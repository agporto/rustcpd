use nalgebra::DMatrix;

use crate::em::{
    EmConfig, IterationState, SparseIndex, initialize_sigma2, normalized_cloud, posterior_stats,
    shared_frame, validate_clouds, variance_floor, weighted_mean,
};
use crate::{Error, Result};

/// Configuration for rigid (similarity) registration.
#[derive(Clone, Debug)]
pub struct RigidConfig {
    /// Shared EM settings.
    pub em: EmConfig,
    /// Estimate an isotropic scale factor; `false` fixes scale at 1.
    pub scale: bool,
    /// Condition the fit by internally centering and scaling both clouds
    /// to a shared unit-scale frame (taken from the target), then map the
    /// result back to the original coordinates. Recommended for clouds in
    /// large or awkward physical units; leaves the returned transform,
    /// points, and `sigma2` in the original frame. (`objective` is reported
    /// in the internal normalized frame.)
    pub normalize: bool,
}

impl Default for RigidConfig {
    fn default() -> Self {
        Self {
            em: EmConfig::default(),
            scale: true,
            normalize: false,
        }
    }
}

/// Output of [`RigidRegistration::register`].
#[derive(Clone, Debug)]
pub struct RigidResult {
    /// Transformed source points `s·Y·R + t` (`M×D`).
    pub points: DMatrix<f64>,
    /// Rotation in row-vector convention: apply as `y·R`.
    pub rotation: DMatrix<f64>,
    /// Translation vector (length `D`).
    pub translation: Vec<f64>,
    /// Estimated isotropic scale (1 when disabled).
    pub scale: f64,
    /// Final Gaussian variance.
    pub sigma2: f64,
    /// Number of EM iterations performed.
    pub iterations: usize,
    /// Final EM objective value.
    pub objective: f64,
    /// Final convergence-criterion value.
    pub difference: f64,
}

/// Rigid (rotation + translation + optional scale) point-set registration.
///
/// `x` is the fixed target cloud (`N×D`), `y` the moving source cloud
/// (`M×D`); only `D` = 2 or 3 is supported.
pub struct RigidRegistration<'a> {
    x: &'a DMatrix<f64>,
    y: &'a DMatrix<f64>,
    config: RigidConfig,
}

impl<'a> RigidRegistration<'a> {
    /// Validate the inputs and build a registration; `x` is the fixed
    /// target, `y` the moving source.
    pub fn new(x: &'a DMatrix<f64>, y: &'a DMatrix<f64>, config: RigidConfig) -> Result<Self> {
        validate_clouds(x, y)?;
        config.em.validate()?;
        if !matches!(x.ncols(), 2 | 3) {
            return Err(Error::UnsupportedRigidDimension);
        }
        Ok(Self { x, y, config })
    }

    /// Run EM to convergence or `max_iterations`.
    pub fn register(&self) -> Result<RigidResult> {
        self.register_with(|_| true)
    }

    /// Run EM as [`Self::register`], invoking `callback` after each
    /// iteration with an [`IterationState`]; return `false` from it to stop
    /// early. Useful for progress reporting, live visualization, or a
    /// custom stopping rule.
    pub fn register_with<F: FnMut(&IterationState) -> bool>(
        &self,
        mut callback: F,
    ) -> Result<RigidResult> {
        if self.config.normalize {
            return self.register_normalized(callback);
        }
        let d = self.x.ncols();
        let mut r = DMatrix::identity(d, d);
        let mut t = vec![0.0; d];
        let mut scale = 1.0;
        let mut ty = self.y.clone();
        let mut sigma2 = self
            .config
            .em
            .sigma2
            .unwrap_or(initialize_sigma2(self.x, self.y)?);
        let mut q = f64::INFINITY;
        let mut diff = f64::INFINITY;
        let mut iterations = 0;
        // Absolute, extent-scaled floor for sigma2 so it stays strictly
        // positive even when `tolerance == 0` (a legitimate "run to
        // max_iterations" setting); the tolerance-derived floor alone
        // collapses to 0 there and lets a later E-step divide by zero.
        let variance_floor = variance_floor(self.x, d);
        // Scale-relative floor for the source spread `ypy`: computing the
        // weighted mean and subtracting it leaves per-coordinate rounding
        // noise of order `eps·max|y|`, so a coincident (or near-coincident)
        // source yields `ypy` up to ~`count·d·(eps·max|y|)²` of pure noise —
        // far above `f64::MIN_POSITIVE`. Estimating scale from such an
        // `ypy` divides noise by noise. Below this floor the scale is
        // unidentifiable and is left unchanged. A genuinely tiny-but-real
        // source spread (points whose magnitudes are themselves tiny) stays
        // above the floor, since then `max|y|` shrinks with the spread.
        let max_abs_y = self.y.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
        let ypy_floor = (64.0
            * (self.y.nrows().max(self.x.nrows()) * d) as f64
            * (f64::EPSILON * max_abs_y).powi(2))
        .max(f64::MIN_POSITIVE);
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
            let mux = weighted_mean_from_px(&stats.px, stats.np);
            let muy = weighted_mean(self.y, &stats.p1, stats.np);
            let mut a = DMatrix::zeros(d, d);
            let mut ypy = 0.0;
            for mi in 0..self.y.nrows() {
                let mut ynorm = 0.0;
                for (j, _) in muy.iter().enumerate().take(d) {
                    ynorm += (self.y[(mi, j)] - muy[j]).powi(2);
                }
                ypy += stats.p1[mi] * ynorm;
                for row in 0..d {
                    for col in 0..d {
                        a[(row, col)] += (stats.px[(mi, row)] - stats.p1[mi] * mux[row])
                            * (self.y[(mi, col)] - muy[col]);
                    }
                }
            }
            let svd = a.clone().svd(true, true);
            let u = svd.u.ok_or(Error::SingularSystem)?;
            let vt = svd.v_t.ok_or(Error::SingularSystem)?;
            let mut correction: DMatrix<f64> = DMatrix::identity(d, d);
            let det_uv: f64 = (&u * &vt).determinant();
            correction[(d - 1, d - 1)] = det_uv.signum();
            r = (&u * correction * &vt).transpose();
            // tr(A Rᵀ_paper) = tr(Aᵀ R_paper); `r` stores the row-vector
            // (transposed) rotation, so this single trace serves both the
            // scale estimate and the objective.
            let tr_ar = (&a * &r).trace();
            // `ypy` is the posterior-weighted source spread about its mean;
            // for a degenerate source (a single point, or all points
            // coincident) it collapses to rounding noise and scale is
            // unidentifiable. Leave the scale unchanged there rather than
            // dividing noise by noise (see `ypy_floor` above).
            if self.config.scale && ypy > ypy_floor {
                scale = tr_ar / ypy;
            }
            for j in 0..d {
                let rotated_mean: f64 = (0..d).map(|q| r[(q, j)] * muy[q]).sum();
                t[j] = mux[j] - scale * rotated_mean;
            }
            apply_rigid(self.y, &r, scale, &t, &mut ty);
            let mut xpx = 0.0;
            for ni in 0..self.x.nrows() {
                let norm: f64 = (0..d).map(|j| (self.x[(ni, j)] - mux[j]).powi(2)).sum();
                xpx += stats.pt1[ni] * norm;
            }
            let previous = q;
            q = (xpx - 2.0 * scale * tr_ar + scale * scale * ypy) / (2.0 * sigma2)
                + d as f64 * stats.np / 2.0 * sigma2.ln();
            diff = (q - previous).abs();
            // With scale optimized, s = tr(AᵀR)/yPy collapses the M-step
            // variance to (xPx − s·tr(AᵀR))/(Np·d). With scale fixed at 1
            // that collapse is invalid; use the general quadratic form.
            sigma2 = if self.config.scale {
                ((xpx - scale * tr_ar) / (stats.np * d as f64))
                    .max(self.config.em.tolerance / 10.0)
                    .max(variance_floor)
            } else {
                ((xpx - 2.0 * tr_ar + ypy) / (stats.np * d as f64))
                    .max(self.config.em.tolerance / 10.0)
                    .max(variance_floor)
            };
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
        Ok(RigidResult {
            points: ty,
            rotation: r,
            translation: t,
            scale,
            sigma2,
            iterations,
            objective: q,
            difference: diff,
        })
    }

    /// Run the fit in a shared unit-scale frame taken from the target, then
    /// map the transform back to the original coordinates.
    fn register_normalized<F: FnMut(&IterationState) -> bool>(
        &self,
        callback: F,
    ) -> Result<RigidResult> {
        let (cx, cy, scale) = shared_frame(self.x, self.y);
        let xn = normalized_cloud(self.x, &cx, scale);
        let yn = normalized_cloud(self.y, &cy, scale);
        let mut inner_config = self.config.clone();
        inner_config.normalize = false;
        let mut result = RigidRegistration::new(&xn, &yn, inner_config)?.register_with(callback)?;
        // With x -> (x - cx)/s and y -> (y - cy)/s, a normalized-frame
        // similarity `s_n·R + t_n` denormalizes to the same rotation and
        // scale, with translation t = s·t_n + cx - scale·(cy·R). Points and
        // variance rescale by s and s² respectively.
        let d = self.x.ncols();
        let r = &result.rotation;
        let mut translation = vec![0.0; d];
        for (j, slot) in translation.iter_mut().enumerate() {
            let rotated_source_centroid: f64 = (0..d).map(|q| cy[q] * r[(q, j)]).sum();
            *slot = scale * result.translation[j] + cx[j] - result.scale * rotated_source_centroid;
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

fn weighted_mean_from_px(px: &DMatrix<f64>, total: f64) -> Vec<f64> {
    (0..px.ncols())
        .map(|j| (0..px.nrows()).map(|i| px[(i, j)]).sum::<f64>() / total)
        .collect()
}

pub(crate) fn apply_rigid(
    y: &DMatrix<f64>,
    r: &DMatrix<f64>,
    scale: f64,
    t: &[f64],
    out: &mut DMatrix<f64>,
) {
    for i in 0..y.nrows() {
        for j in 0..y.ncols() {
            out[(i, j)] = scale * (0..y.ncols()).map(|q| y[(i, q)] * r[(q, j)]).sum::<f64>() + t[j];
        }
    }
}
