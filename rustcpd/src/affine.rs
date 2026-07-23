use nalgebra::DMatrix;

use crate::em::{
    EmConfig, IterationState, SparseIndex, initialize_sigma2, normalized_cloud, posterior_stats,
    shared_frame, validate_clouds, variance_floor, weighted_mean,
};
use crate::{Error, Result};

/// Configuration for affine registration.
#[derive(Clone, Debug, Default)]
pub struct AffineConfig {
    /// Shared EM settings.
    pub em: EmConfig,
    /// Condition the fit by internally centering and scaling both clouds
    /// to a shared unit-scale frame (taken from the target), then map the
    /// result back to the original coordinates. Recommended for clouds in
    /// large or awkward physical units; leaves the returned transform,
    /// points, and `sigma2` in the original frame. (`objective` is reported
    /// in the internal normalized frame.)
    pub normalize: bool,
}

/// Output of [`AffineRegistration::register`].
#[derive(Clone, Debug)]
pub struct AffineResult {
    /// Transformed source points `Y·B + t` (`M×D`).
    pub points: DMatrix<f64>,
    /// Affine matrix in row-vector convention: apply as `y·B`.
    pub transform: DMatrix<f64>,
    /// Translation vector (length `D`).
    pub translation: Vec<f64>,
    /// Final Gaussian variance.
    pub sigma2: f64,
    /// Number of EM iterations performed.
    pub iterations: usize,
    /// Final EM objective value.
    pub objective: f64,
    /// Final convergence-criterion value.
    pub difference: f64,
}

/// Affine point-set registration; `x` is the fixed target cloud (`N×D`),
/// `y` the moving source cloud (`M×D`).
pub struct AffineRegistration<'a> {
    x: &'a DMatrix<f64>,
    y: &'a DMatrix<f64>,
    config: AffineConfig,
}

impl<'a> AffineRegistration<'a> {
    /// Validate the inputs and build a registration; `x` is the fixed
    /// target, `y` the moving source.
    pub fn new(x: &'a DMatrix<f64>, y: &'a DMatrix<f64>, config: AffineConfig) -> Result<Self> {
        validate_clouds(x, y)?;
        config.em.validate()?;
        Ok(Self { x, y, config })
    }

    /// Run EM to convergence or `max_iterations`.
    pub fn register(&self) -> Result<AffineResult> {
        self.register_with(|_| true)
    }

    /// Run EM as [`Self::register`], invoking `callback` after each
    /// iteration with an [`IterationState`]; return `false` from it to stop
    /// early.
    pub fn register_with<F: FnMut(&IterationState) -> bool>(
        &self,
        mut callback: F,
    ) -> Result<AffineResult> {
        if self.config.normalize {
            return self.register_normalized(callback);
        }
        let d = self.x.ncols();
        let mut b = DMatrix::identity(d, d);
        let mut t = vec![0.0; d];
        let mut ty = self.y.clone();
        let mut sigma2 = self
            .config
            .em
            .sigma2
            .unwrap_or(initialize_sigma2(self.x, self.y)?);
        let mut q = f64::INFINITY;
        let mut diff = f64::INFINITY;
        let mut iterations = 0;
        // Keep sigma2 strictly positive even when `tolerance == 0`; see the
        // rigid path for the rationale.
        let variance_floor = variance_floor(self.x, d);
        let sparse_index = self.config.em.k.map(|k| SparseIndex::new(self.x, k));
        while iterations < self.config.em.max_iterations && diff > self.config.em.tolerance {
            let stats = posterior_stats(
                self.x,
                &ty,
                sigma2,
                &self.config.em,
                false,
                sparse_index.as_ref(),
            );
            if stats.np <= f64::MIN_POSITIVE {
                return Err(Error::SingularSystem);
            }
            let mux: Vec<_> = (0..d)
                .map(|j| (0..stats.px.nrows()).map(|i| stats.px[(i, j)]).sum::<f64>() / stats.np)
                .collect();
            let muy = weighted_mean(self.y, &stats.p1, stats.np);
            let mut a = DMatrix::zeros(d, d);
            let mut ypy = DMatrix::zeros(d, d);
            for mi in 0..self.y.nrows() {
                for row in 0..d {
                    for col in 0..d {
                        ypy[(row, col)] += stats.p1[mi]
                            * (self.y[(mi, row)] - muy[row])
                            * (self.y[(mi, col)] - muy[col]);
                    }
                }
                for row in 0..d {
                    for col in 0..d {
                        a[(row, col)] += (stats.px[(mi, row)] - stats.p1[mi] * mux[row])
                            * (self.y[(mi, col)] - muy[col]);
                    }
                }
            }
            // Scale-relative jitter: a fixed absolute value would dominate
            // for small-magnitude clouds and vanish for large ones.
            let mean_diagonal: f64 = ypy.trace() / d as f64;
            let jitter = mean_diagonal.max(f64::MIN_POSITIVE) * 1e-12;
            let mut solve_matrix = ypy.clone();
            for j in 0..d {
                solve_matrix[(j, j)] += jitter;
            }
            let chol = solve_matrix.cholesky().ok_or(Error::SingularSystem)?;
            // Row-vector convention: with A = Xᵀ Pᵀ Y and
            // S = Yᵀ diag(P1) Y, the M-step solves S·B = Aᵀ.
            b = chol.solve(&a.transpose());
            for j in 0..d {
                t[j] = mux[j] - (0..d).map(|q| muy[q] * b[(q, j)]).sum::<f64>();
            }
            apply_affine(self.y, &b, &t, &mut ty);
            let mut xpx = 0.0;
            for ni in 0..self.x.nrows() {
                xpx += stats.pt1[ni]
                    * (0..d)
                        .map(|j| (self.x[(ni, j)] - mux[j]).powi(2))
                        .sum::<f64>();
            }
            let tr_ab = (&a * &b).trace();
            let quad = (b.transpose() * &ypy * &b).trace();
            let previous = q;
            q = (xpx - 2.0 * tr_ab + quad) / (2.0 * sigma2)
                + d as f64 * stats.np / 2.0 * sigma2.ln();
            diff = (q - previous).abs() / q.abs().max(1.0);
            // Jitter means B is not the exact minimizer of the unregularized
            // quadratic, so the collapsed xPx - tr(A·B) identity does not
            // hold. Use the full residual energy.
            sigma2 = ((xpx - 2.0 * tr_ab + quad) / (stats.np * d as f64))
                .max(self.config.em.tolerance / 10.0)
                .max(variance_floor);
            iterations += 1;
            let state = IterationState {
                iteration: iterations,
                sigma2,
                difference: diff,
                points: &ty,
            };
            if !callback(&state) {
                break;
            }
        }
        Ok(AffineResult {
            points: ty,
            transform: b,
            translation: t,
            sigma2,
            iterations,
            objective: q,
            difference: diff,
        })
    }

    /// Run the fit in a shared unit-scale frame taken from the target, then
    /// map the affine transform back to the original coordinates.
    fn register_normalized<F: FnMut(&IterationState) -> bool>(
        &self,
        callback: F,
    ) -> Result<AffineResult> {
        let (cx, cy, scale) = shared_frame(self.x, self.y);
        let xn = normalized_cloud(self.x, &cx, scale);
        let yn = normalized_cloud(self.y, &cy, scale);
        let mut inner_config = self.config.clone();
        inner_config.normalize = false;
        let mut result =
            AffineRegistration::new(&xn, &yn, inner_config)?.register_with(callback)?;
        // With x -> (x - cx)/s and y -> (y - cy)/s, a normalized-frame
        // affine `B_n, t_n` has the same matrix and translation
        // t = s·t_n + cx - cy·B_n. Points and variance rescale by s and s².
        let d = self.x.ncols();
        let b = &result.transform;
        let mut translation = vec![0.0; d];
        for (j, slot) in translation.iter_mut().enumerate() {
            let mapped_source_centroid: f64 = (0..d).map(|q| cy[q] * b[(q, j)]).sum();
            *slot = scale * result.translation[j] + cx[j] - mapped_source_centroid;
        }
        result.translation = translation;
        for value in result.points.iter_mut() {
            *value *= scale;
        }
        for i in 0..result.points.nrows() {
            for (j, &c) in cx.iter().enumerate() {
                result.points[(i, j)] += c;
            }
        }
        result.sigma2 *= scale * scale;
        Ok(result)
    }
}

fn apply_affine(y: &DMatrix<f64>, b: &DMatrix<f64>, t: &[f64], out: &mut DMatrix<f64>) {
    for i in 0..y.nrows() {
        for j in 0..y.ncols() {
            out[(i, j)] = (0..y.ncols()).map(|q| y[(i, q)] * b[(q, j)]).sum::<f64>() + t[j];
        }
    }
}
