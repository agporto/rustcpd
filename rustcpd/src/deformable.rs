use nalgebra::{DMatrix, DVector};

use crate::em::{
    EmConfig, IterationState, SparseIndex, gaussian_kernel_with_parallel, initialize_sigma2,
    normalized_cloud, posterior_stats, shared_frame, validate_clouds,
};
use crate::solve::{lu_solve, matmul_into, symmetric_eigen};
use crate::{Error, Result};

/// A hard correspondence pinning source point `source` to target point
/// `target` during deformable registration. Identical `(source, target)`
/// pairs are deduplicated; when a `source` is pinned to several distinct
/// targets, their masses accumulate (so the source is drawn toward the
/// mean of its targets), matching the reference implementation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Constraint {
    /// Row index into the moving source cloud `y`.
    pub source: usize,
    /// Row index into the fixed target cloud `x`.
    pub target: usize,
}

/// Strategy for building the low-rank Gaussian-kernel approximation used
/// by the deformable M-step.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum LowRankMethod {
    /// Full symmetric eigendecomposition, keeping the leading `rank`
    /// eigenpairs. Exact up to truncation but costs `O(M³)` to build.
    #[default]
    Eigen,
    /// Greedy pivoted (incomplete) Cholesky factorization `G ≈ LLᵀ` with
    /// `rank` pivots, costing `O(M·rank²)` and built matrix-free (the dense
    /// kernel is never formed). Much cheaper than the eigendecomposition
    /// for large `M`; the M-step works on the factor `L` directly, so the
    /// two methods produce the same registration (up to rounding).
    PivotedCholesky,
}

/// Configuration for (optionally constrained) deformable registration.
#[derive(Clone, Debug)]
pub struct DeformableConfig {
    /// Shared EM settings.
    pub em: EmConfig,
    /// Regularization strength: larger values produce a more coherent
    /// (stiffer) deformation field.
    pub alpha: f64,
    /// Gaussian kernel width of the deformation field.
    pub beta: f64,
    /// `Some(rank)` approximates the kernel with a rank-`rank` factor
    /// when `rank < M`, reducing the per-iteration solve cost.
    pub low_rank: Option<usize>,
    /// How the low-rank kernel approximation is constructed.
    pub low_rank_method: LowRankMethod,
    /// Residual-diagonal early-stop tolerance for the pivoted-Cholesky
    /// factorization, in `[0, 1)`. Pivoting halts once the largest
    /// remaining residual diagonal falls to this value; `0.0` (the default)
    /// uses only an epsilon-scaled numerical floor. Ignored by the eigen
    /// method.
    pub pivoted_cholesky_tolerance: f64,
    /// Condition the fit by internally centering and scaling both clouds
    /// to a shared unit-scale frame (taken from the target), then map the
    /// result back to the original coordinates. Recommended for clouds in
    /// large physical units, where the absolute `beta` kernel width would
    /// otherwise be mis-scaled. `points`, `sigma2`, and `transform` are in
    /// the original frame; `weights`/`kernel`/`source` remain in the
    /// internal normalized frame. Note that the absolute length-scale
    /// parameters (`beta`, `constraint_error`, `em.sigma2`, `em.tolerance`)
    /// are interpreted in the normalized frame when this is set.
    pub normalize: bool,
    /// Hard point-to-point correspondences.
    pub constraints: Vec<Constraint>,
    /// Variance assigned to constraint residuals; smaller values pin
    /// constrained points more tightly.
    pub constraint_error: f64,
}

impl Default for DeformableConfig {
    fn default() -> Self {
        Self {
            em: EmConfig::default(),
            alpha: 2.0,
            beta: 2.0,
            low_rank: Some(300),
            low_rank_method: LowRankMethod::default(),
            pivoted_cholesky_tolerance: 0.0,
            normalize: false,
            constraints: Vec::new(),
            constraint_error: 1e-8,
        }
    }
}

/// Output of [`DeformableRegistration::register`].
#[derive(Clone, Debug)]
pub struct DeformableResult {
    /// Deformed source points (`M×D`): `Y + G·W` for a full-rank fit, or
    /// `Y + Q·diag(Λ)·Qᵀ·W` for a low-rank fit (the approximate kernel is
    /// what the EM loop optimized against). Note [`Self::transform`]
    /// evaluates the *exact* kernel, so for a low-rank fit
    /// `transform(source)` differs from `points` by the kernel
    /// approximation error, matching the reference implementation.
    pub points: DMatrix<f64>,
    /// Deformation coefficients `W` (`M×D`).
    pub weights: DMatrix<f64>,
    /// Dense Gaussian kernel `G` on the source cloud (`M×M`) — populated
    /// only when the fit ran at full rank (`low_rank = None`, or a
    /// requested rank `≥ M`). For a low-rank fit it is **empty** (`0×0`)
    /// and the approximation is exposed through [`Self::low_rank_basis`] /
    /// [`Self::low_rank_eigenvalues`] instead (`G ≈ Q·diag(Λ)·Qᵀ`).
    pub kernel: DMatrix<f64>,
    /// Low-rank kernel factor `L` (`M×r`, `G ≈ L Lᵀ`) for a low-rank fit;
    /// `None` for a full-rank fit. The training path works on this factor
    /// directly; the orthonormal `(Q, Λ)` form is derived on demand by
    /// [`Self::low_rank_basis`] / [`Self::low_rank_eigenvalues`].
    low_rank_factor: Option<DMatrix<f64>>,
    /// The moving source cloud `Y` (`M×D`), retained so the learned
    /// deformation can be evaluated at new points via [`Self::transform`].
    pub source: DMatrix<f64>,
    /// Gaussian kernel width `beta` used to fit the deformation.
    pub beta: f64,
    /// Final Gaussian variance.
    pub sigma2: f64,
    /// Number of EM iterations performed.
    pub iterations: usize,
    /// Final convergence-criterion value.
    pub difference: f64,
    /// Internal normalization frame `(target_centroid, source_centroid,
    /// scale)` when `normalize` was set. `weights`/`kernel`/`source` are
    /// expressed in this frame; [`Self::transform`] applies and undoes it
    /// so new points can be supplied in the original coordinates.
    normalization: Option<(Vec<f64>, Vec<f64>, f64)>,
}

impl DeformableResult {
    /// The low-rank kernel factor `L` (`M×r`, `G ≈ L Lᵀ`) for a low-rank
    /// fit; `None` for a full-rank fit. This is the representation the fit
    /// actually used.
    pub fn low_rank_factor(&self) -> Option<&DMatrix<f64>> {
        self.low_rank_factor.as_ref()
    }

    /// The orthonormal low-rank basis `Q` (`M×r`, `G ≈ Q diag(Λ) Qᵀ`),
    /// **computed on demand** from the stored factor; `None` for a
    /// full-rank fit. Pair with [`Self::low_rank_eigenvalues`]; use
    /// [`Self::low_rank_spectrum`] to get both at once without recomputing.
    pub fn low_rank_basis(&self) -> Option<DMatrix<f64>> {
        self.low_rank_spectrum().map(|(q, _)| q)
    }

    /// The low-rank eigenvalues `Λ`, computed on demand from the stored
    /// factor; `None` for a full-rank fit.
    pub fn low_rank_eigenvalues(&self) -> Option<Vec<f64>> {
        self.low_rank_spectrum().map(|(_, s)| s)
    }

    /// Both the orthonormal basis `Q` and eigenvalues `Λ` of the low-rank
    /// kernel approximation, computed on demand from the stored factor;
    /// `None` for a full-rank fit.
    pub fn low_rank_spectrum(&self) -> Option<(DMatrix<f64>, Vec<f64>)> {
        self.low_rank_factor
            .as_ref()
            .and_then(|l| factor_spectrum(l).ok())
    }

    /// The internal normalization frame `(target_centroid, source_centroid,
    /// scale)` the warp was fit in, if [`DeformableConfig::normalize`] was
    /// set. `weights`, `kernel`, and `source` are expressed in this frame.
    pub fn normalization(&self) -> Option<(&[f64], &[f64], f64)> {
        self.normalization
            .as_ref()
            .map(|(target_centroid, source_centroid, scale)| {
                (
                    target_centroid.as_slice(),
                    source_centroid.as_slice(),
                    *scale,
                )
            })
    }

    /// Evaluate the learned continuous deformation at arbitrary points
    /// `z` (shape `(P, D)`), returning `z + G(z, Y)·W`.
    ///
    /// The registration warps only the source points it was fit on; this
    /// applies the same displacement field to any points in the source
    /// frame — a full-resolution mesh, a set of landmarks, a grid — so a
    /// coarse registration can drive a dense warp. `z` must have the same
    /// dimensionality as the registered clouds. If the fit was normalized,
    /// `z` is supplied in the original coordinate frame.
    pub fn transform(&self, z: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        apply_deformation(
            &self.source,
            &self.weights,
            self.beta,
            self.normalization(),
            z,
        )
    }
}

/// Deformable (non-rigid) point-set registration; `x` is the fixed target
/// cloud (`N×D`), `y` the moving source cloud (`M×D`).
pub struct DeformableRegistration<'a> {
    x: &'a DMatrix<f64>,
    y: &'a DMatrix<f64>,
    config: DeformableConfig,
}

impl<'a> DeformableRegistration<'a> {
    /// Validate the inputs and build a registration; `x` is the fixed
    /// target, `y` the moving source.
    pub fn new(x: &'a DMatrix<f64>, y: &'a DMatrix<f64>, config: DeformableConfig) -> Result<Self> {
        validate_clouds(x, y)?;
        config.em.validate()?;
        if !config.alpha.is_finite() || config.alpha <= 0.0 {
            return Err(Error::PositiveParameter("alpha"));
        }
        if !config.beta.is_finite() || config.beta <= 0.0 {
            return Err(Error::PositiveParameter("beta"));
        }
        if !config.constraint_error.is_finite() || config.constraint_error <= 0.0 {
            return Err(Error::PositiveParameter("constraint_error"));
        }
        if !config.pivoted_cholesky_tolerance.is_finite()
            || !(0.0..1.0).contains(&config.pivoted_cholesky_tolerance)
        {
            return Err(Error::PositiveParameter("pivoted_cholesky_tolerance"));
        }
        // A rank-0 approximation is degenerate: the eigen path would
        // silently return an all-zero displacement (points never move,
        // weights inconsistent) and the pivoted path a misleading
        // SingularSystem. Reject it up front like the reference does.
        if config.low_rank == Some(0) {
            return Err(Error::PositiveParameter("low_rank"));
        }
        if config
            .constraints
            .iter()
            .any(|c| c.source >= y.nrows() || c.target >= x.nrows())
        {
            return Err(Error::ConstraintOutOfBounds);
        }
        Ok(Self { x, y, config })
    }

    /// Run EM to convergence or `max_iterations`.
    pub fn register(&self) -> Result<DeformableResult> {
        self.register_with(|_| true)
    }

    /// Run EM as [`Self::register`], invoking `callback` after each
    /// iteration with an [`IterationState`]; return `false` from it to stop
    /// early. Useful for progress on the long deformable fits.
    pub fn register_with<F: FnMut(&IterationState) -> bool>(
        &self,
        mut callback: F,
    ) -> Result<DeformableResult> {
        if self.config.normalize {
            return self.register_normalized(callback);
        }
        let (m, d) = (self.y.nrows(), self.y.ncols());
        let parallel = self.config.em.parallel;
        // Build the kernel representation. A full-rank fit forms the dense
        // `M×M` kernel; a low-rank fit keeps only a compact factor
        // `G ≈ Q·diag(Λ)·Qᵀ`. The pivoted-Cholesky factor is built
        // matrix-free (kernel columns evaluated on demand), so the dense
        // kernel is never allocated for that path.
        let requested_rank = self.config.low_rank.filter(|&rank| rank < m);
        let (mut low_rank, dense_kernel) = match requested_rank {
            Some(rank) => {
                let factor = match self.config.low_rank_method {
                    LowRankMethod::Eigen => {
                        let g = gaussian_kernel_with_parallel(
                            self.y,
                            self.y,
                            self.config.beta,
                            parallel,
                        )?;
                        eigen_factor(&g, rank, parallel)?
                    }
                    LowRankMethod::PivotedCholesky => pivoted_cholesky_factor(
                        self.y,
                        self.config.beta,
                        rank,
                        self.config.pivoted_cholesky_tolerance,
                    )?,
                };
                (Some(LowRankSolver::new(factor, d, parallel)), None)
            }
            None => {
                let g = gaussian_kernel_with_parallel(self.y, self.y, self.config.beta, parallel)?;
                (None, Some(g))
            }
        };
        let mut weights = DMatrix::zeros(m, d);
        let mut ty = self.y.clone();
        let mut sigma2 = self
            .config
            .em
            .sigma2
            .unwrap_or(initialize_sigma2(self.x, self.y)?);
        let centered_target = center(self.x);
        let variance_floor = f64::EPSILON * centered_target.iter().map(|v| v * v).sum::<f64>()
            / (self.x.nrows() * d) as f64;
        let mut diff = f64::INFINITY;
        let mut iterations = 0;
        let sparse_index = self.config.em.k.map(|k| SparseIndex::new(self.x, k));
        // Deduplicate identical (source, target) pairs, then accumulate the
        // mass (a count) and sum the target coordinates per source, so a
        // source pinned to several distinct targets is drawn toward their
        // mean. Mirrors the reference's `np.unique` + `bincount` + `add.at`.
        let mut constraint_mass = vec![0.0; m];
        let mut constraint_px: DMatrix<f64> = DMatrix::zeros(m, d);
        let mut seen = std::collections::HashSet::new();
        for constraint in &self.config.constraints {
            if !seen.insert((constraint.source, constraint.target)) {
                continue;
            }
            constraint_mass[constraint.source] += 1.0;
            for j in 0..d {
                constraint_px[(constraint.source, j)] += self.x[(constraint.target, j)];
            }
        }
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
            let constraint_scale = sigma2 / self.config.constraint_error;
            let mut row_weights = stats.p1.clone();
            let mut f = DMatrix::zeros(m, d);
            for i in 0..m {
                row_weights[i] += constraint_scale * constraint_mass[i];
                for j in 0..d {
                    f[(i, j)] = stats.px[(i, j)] - stats.p1[i] * self.y[(i, j)]
                        + constraint_scale
                            * (constraint_px[(i, j)] - constraint_mass[i] * self.y[(i, j)]);
                }
            }
            let lambda = self.config.alpha * sigma2;
            // Advance `ty = Y + G·W`. The low-rank branch works directly on
            // the factor `L` (`G ≈ L Lᵀ`) via a reused buffer, so no dense
            // kernel is touched; the full-rank branch uses the dense solve.
            match &mut low_rank {
                Some(solver) => {
                    solver.solve(&row_weights, &f, lambda)?;
                    ty = self.y + &solver.displacement;
                    weights.copy_from(&solver.weights);
                }
                None => {
                    let g = dense_kernel
                        .as_ref()
                        .expect("full-rank fit builds the kernel");
                    let mut a = DMatrix::zeros(m, m);
                    for j in 0..m {
                        for i in 0..m {
                            a[(i, j)] = row_weights[i] * g[(i, j)];
                        }
                    }
                    for i in 0..m {
                        a[(i, i)] += lambda;
                    }
                    weights = lu_solve(a, f, parallel).ok_or(Error::SingularSystem)?;
                    ty = self.y + g * &weights;
                }
            }
            let previous = sigma2;
            let xpx: f64 = (0..self.x.nrows())
                .map(|i| stats.pt1[i] * (0..d).map(|j| self.x[(i, j)].powi(2)).sum::<f64>())
                .sum();
            let ypy: f64 = (0..m)
                .map(|i| stats.p1[i] * (0..d).map(|j| ty[(i, j)].powi(2)).sum::<f64>())
                .sum();
            let mut cross = 0.0;
            for i in 0..m {
                for j in 0..d {
                    cross += ty[(i, j)] * stats.px[(i, j)];
                }
            }
            sigma2 = (xpx - 2.0 * cross + ypy) / (stats.np * d as f64);
            if !sigma2.is_finite() || sigma2 <= 0.0 {
                sigma2 = (self.config.em.tolerance / 10.0)
                    .max(variance_floor)
                    .max(f64::MIN_POSITIVE);
            }
            diff = (sigma2 - previous).abs();
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
        let (kernel, low_rank_factor) = match low_rank {
            Some(solver) => (DMatrix::zeros(0, 0), Some(solver.factor)),
            None => (dense_kernel.expect("full-rank fit builds the kernel"), None),
        };
        Ok(DeformableResult {
            points: ty,
            weights,
            kernel,
            low_rank_factor,
            source: self.y.clone(),
            beta: self.config.beta,
            sigma2,
            iterations,
            difference: diff,
            normalization: None,
        })
    }

    /// Run the fit in a shared unit-scale frame taken from the target,
    /// then map `points` and `sigma2` back to the original coordinates.
    /// The warp itself (`weights`/`kernel`/`source`) stays in the
    /// normalized frame; [`DeformableResult::transform`] and the returned
    /// `points` account for that.
    fn register_normalized<F: FnMut(&IterationState) -> bool>(
        &self,
        callback: F,
    ) -> Result<DeformableResult> {
        let (cx, cy, scale) = shared_frame(self.x, self.y);
        let xn = normalized_cloud(self.x, &cx, scale);
        let yn = normalized_cloud(self.y, &cy, scale);
        let mut inner_config = self.config.clone();
        inner_config.normalize = false;
        let mut result =
            DeformableRegistration::new(&xn, &yn, inner_config)?.register_with(callback)?;
        for value in result.points.iter_mut() {
            *value *= scale;
        }
        for i in 0..result.points.nrows() {
            for (j, &c) in cx.iter().enumerate() {
                result.points[(i, j)] += c;
            }
        }
        result.sigma2 *= scale * scale;
        result.normalization = Some((cx, cy, scale));
        Ok(result)
    }
}

/// Evaluate a learned deformation `z + G(z, source)·W` at arbitrary
/// points `z`, given its parts. `normalization` is the frame the warp was
/// fit in (if any), as `(target_centroid, source_centroid, scale)`: `z` is
/// mapped in by the source centroid and scale, the displacement applied,
/// and the result mapped back out by the target centroid and scale.
/// Exposed so binding layers that store the warp piecemeal can evaluate it
/// without holding a whole [`DeformableResult`].
pub fn apply_deformation(
    source: &DMatrix<f64>,
    weights: &DMatrix<f64>,
    beta: f64,
    normalization: Option<(&[f64], &[f64], f64)>,
    z: &DMatrix<f64>,
) -> Result<DMatrix<f64>> {
    if z.ncols() != source.ncols() {
        return Err(Error::DimensionMismatch);
    }
    if !z.iter().all(|v| v.is_finite()) {
        return Err(Error::NonFiniteInput);
    }
    // When the fit was normalized, `source`/`weights` live in the
    // normalized frame; map the query in, warp, and map back out.
    let zn = match normalization {
        Some((_, source_centroid, scale)) => normalized_cloud(z, source_centroid, scale),
        None => z.clone(),
    };
    // g_zy[(p, m)] = exp(-||z_p - y_m||^2 / (2 beta^2)); displacement is
    // g_zy · W, matching the training-time `G · W` when z == source.
    let g_zy = gaussian_kernel_with_parallel(&zn, source, beta, true)?;
    let warped = &zn + g_zy * weights;
    Ok(match normalization {
        Some((target_centroid, _, scale)) => {
            DMatrix::from_fn(warped.nrows(), warped.ncols(), |i, j| {
                warped[(i, j)] * scale + target_centroid[j]
            })
        }
        None => warped,
    })
}

fn center(points: &DMatrix<f64>) -> DMatrix<f64> {
    let mut result = points.clone();
    for j in 0..points.ncols() {
        let mean = (0..points.nrows()).map(|i| points[(i, j)]).sum::<f64>() / points.nrows() as f64;
        for i in 0..points.nrows() {
            result[(i, j)] -= mean;
        }
    }
    result
}

/// Low-rank kernel factor `L` (`M×k`, `G ≈ L Lᵀ`) from the leading `rank`
/// eigenpairs of the dense kernel `g`: with `g ≈ Q diag(Λ) Qᵀ`, set
/// `L = Q diag(√Λ)`. Eigenpairs below the decomposition's noise floor are
/// dropped.
fn eigen_factor(g: &DMatrix<f64>, rank: usize, parallel: bool) -> Result<DMatrix<f64>> {
    let (eigenvectors, eigenvalues) = symmetric_eigen(g, parallel).ok_or(Error::SingularSystem)?;
    let mut order: Vec<usize> = (0..g.nrows()).collect();
    order.sort_unstable_by(|&a, &b| eigenvalues[b].total_cmp(&eigenvalues[a]));
    let floor = eigenvalues
        .iter()
        .fold(0.0_f64, |a, &b| a.max(b))
        .max(f64::MIN_POSITIVE)
        * f64::EPSILON;
    let selected: Vec<_> = order
        .into_iter()
        .filter(|&i| eigenvalues[i] > floor)
        .take(rank)
        .collect();
    if selected.is_empty() {
        return Err(Error::SingularSystem);
    }
    Ok(DMatrix::from_fn(g.nrows(), selected.len(), |row, col| {
        eigenvectors[(row, selected[col])] * eigenvalues[selected[col]].sqrt()
    }))
}

/// Greedy pivoted (incomplete) Cholesky low-rank kernel factor `L`
/// (`M×k`, `G ≈ L Lᵀ`), built **matrix-free** from the source points.
///
/// Repeatedly pivots on the largest residual diagonal entry — the classic
/// `O(M·rank²)` scheme. Each pivot's kernel column is evaluated on demand
/// (no dense `M×M` kernel; memory is `O(M·rank)`), and the rank-1 downdate
/// is expressed as contiguous matrix-vector products so it runs on SIMD
/// GEMV rather than a scalar loop. Pivoting stops once the largest residual
/// diagonal falls to `tolerance` (floored at a small epsilon-scaled value).
fn pivoted_cholesky_factor(
    points: &DMatrix<f64>,
    beta: f64,
    rank: usize,
    tolerance: f64,
) -> Result<DMatrix<f64>> {
    let m = points.nrows();
    let max_rank = rank.min(m);
    let squared_norms: DVector<f64> =
        DVector::from_fn(m, |i, _| points.row(i).iter().map(|v| v * v).sum());
    // The Gaussian kernel diagonal is exactly 1 (`exp(0)`); the residual
    // diagonal starts there and is deflated as pivots are taken.
    let mut diag = vec![1.0_f64; m];
    let stop = tolerance.max(10.0 * f64::EPSILON);
    let inv_two_beta_sq = -1.0 / (2.0 * beta * beta);
    let mut l = DMatrix::zeros(m, max_rank);
    let mut used = 0usize;
    for k in 0..max_rank {
        // Pivot on the largest residual diagonal entry.
        let (pivot, best) =
            diag.iter()
                .enumerate()
                .fold((0usize, f64::NEG_INFINITY), |(bi, bv), (i, &v)| {
                    if v > bv { (i, v) } else { (bi, bv) }
                });
        if best <= stop {
            break;
        }
        // Kernel column for `pivot`, evaluated on demand as a GEMV:
        // dist²_i = ‖p_i‖² + ‖p_pivot‖² − 2·(P·p_pivot)_i, then exp.
        let pivot_point = points.row(pivot).transpose();
        let dots = points * pivot_point;
        let sq_pivot = squared_norms[pivot];
        let mut column = DVector::from_fn(m, |i, _| {
            let dist2 = (squared_norms[i] + sq_pivot - 2.0 * dots[i]).max(0.0);
            (dist2 * inv_two_beta_sq).exp()
        });
        // Rank-1 downdate as a GEMV: column -= L[:, :k] · L[pivot, :k]ᵀ.
        if k > 0 {
            let prefix = l.columns(0, k);
            let pivot_row = prefix.row(pivot).transpose();
            column -= prefix * pivot_row;
        }
        column /= best.sqrt();
        // Deflate the residual diagonal (clamped at zero against roundoff).
        for i in 0..m {
            diag[i] = (diag[i] - column[i] * column[i]).max(0.0);
        }
        diag[pivot] = 0.0;
        l.set_column(k, &column);
        used = k + 1;
    }
    if used == 0 {
        return Err(Error::SingularSystem);
    }
    Ok(l.columns(0, used).into_owned())
}

/// Re-express a low-rank factor `L` (`G ≈ L Lᵀ`) in orthonormal eigenpair
/// form `(Q, Λ)` with `G ≈ Q diag(Λ) Qᵀ`, via the small `k×k` eigenproblem
/// `LᵀL = V diag(S) Vᵀ` (`Q = L V S^{-1/2}`, `Λ = S`). Used only when the
/// caller asks for the public spectrum; it is *not* on the training path.
/// Public wrapper over the low-rank factor → `(Q, Λ)` conversion, for
/// binding layers that store a factor and derive its spectrum on demand.
/// Returns `None` if the (small) eigenproblem fails to converge.
pub fn low_rank_spectrum(factor: &DMatrix<f64>) -> Option<(DMatrix<f64>, Vec<f64>)> {
    factor_spectrum(factor).ok()
}

pub(crate) fn factor_spectrum(l: &DMatrix<f64>) -> Result<(DMatrix<f64>, Vec<f64>)> {
    let m = l.nrows();
    let k = l.ncols();
    let gram = l.tr_mul(l);
    let (vectors, s) = symmetric_eigen(&gram, true).ok_or(Error::SingularSystem)?;
    let mut order: Vec<usize> = (0..k).collect();
    order.sort_unstable_by(|&a, &b| s[b].total_cmp(&s[a]));
    let floor = s
        .iter()
        .fold(0.0_f64, |a, &b| a.max(b))
        .max(f64::MIN_POSITIVE)
        * f64::EPSILON;
    let selected: Vec<_> = order.into_iter().filter(|&i| s[i] > floor).collect();
    let v_sel = DMatrix::from_fn(k, selected.len(), |row, col| vectors[(row, selected[col])]);
    let mut q = l * v_sel;
    for (col, &idx) in selected.iter().enumerate() {
        let inv_sqrt = 1.0 / s[idx].sqrt();
        for row in 0..m {
            q[(row, col)] *= inv_sqrt;
        }
    }
    let values = selected.iter().map(|&i| s[i]).collect();
    Ok((q, values))
}

/// Reusable low-rank CPD M-step working directly on the factor `L`
/// (`G ≈ L Lᵀ`), so no `(Q, Λ)` conversion is needed during training. All
/// scratch is preallocated once and reused each iteration; the three large
/// `M×k` products go through `faer`'s SIMD GEMM.
///
/// Direct-`L` Woodbury: with `D = diag(w)` and `λ > 0`,
/// `H = (LᵀDL + λI)⁻¹ LᵀF`, `W = (F − D·L·H)/λ`, and the displacement
/// `G·W = L·H`. `LᵀDL` is formed as the symmetric self-product
/// `SᵀS` with `S = √D·L` (positive-semidefinite by construction), so the
/// `k×k` system is SPD and Cholesky-solvable.
struct LowRankSolver {
    factor: DMatrix<f64>,
    scaled: DMatrix<f64>,
    system: DMatrix<f64>,
    rhs: DMatrix<f64>,
    displacement: DMatrix<f64>,
    weights: DMatrix<f64>,
    parallel: bool,
}

impl LowRankSolver {
    fn new(factor: DMatrix<f64>, d: usize, parallel: bool) -> Self {
        let (m, k) = (factor.nrows(), factor.ncols());
        Self {
            factor,
            scaled: DMatrix::zeros(m, k),
            system: DMatrix::zeros(k, k),
            rhs: DMatrix::zeros(k, d),
            displacement: DMatrix::zeros(m, d),
            weights: DMatrix::zeros(m, d),
            parallel,
        }
    }

    /// Solve the M-step for the current row weights `w` (all `≥ 0`), target
    /// residual `f`, and regularization `lambda`, filling `self.weights`
    /// (`W`) and `self.displacement` (`G·W`).
    fn solve(&mut self, w: &[f64], f: &DMatrix<f64>, lambda: f64) -> Result<()> {
        let (m, k) = (self.factor.nrows(), self.factor.ncols());
        let d = f.ncols();
        // scaled = √D · L (columns scaled elementwise by the shared √w).
        let sqrt_w = DVector::from_iterator(m, w.iter().map(|value| value.sqrt()));
        for col in 0..k {
            let scaled_col = self.factor.column(col).component_mul(&sqrt_w);
            self.scaled.set_column(col, &scaled_col);
        }
        // system = scaledᵀ·scaled (= LᵀDL), symmetric PSD, then + λI.
        matmul_into(
            &mut self.system,
            &self.scaled,
            true,
            &self.scaled,
            false,
            self.parallel,
        );
        for a in 0..k {
            self.system[(a, a)] += lambda;
        }
        // rhs = Lᵀ·F
        matmul_into(&mut self.rhs, &self.factor, true, f, false, self.parallel);
        // H = system⁻¹·rhs  (k×k Cholesky; k is small).
        let h = self
            .system
            .clone()
            .cholesky()
            .ok_or(Error::SingularSystem)?
            .solve(&self.rhs);
        // displacement = L·H  (= G·W).
        matmul_into(
            &mut self.displacement,
            &self.factor,
            false,
            &h,
            false,
            self.parallel,
        );
        // W = (F − D·displacement)/λ, column by column.
        let w_vec = DVector::from_column_slice(w);
        for col in 0..d {
            let weighted = w_vec.component_mul(&self.displacement.column(col));
            let column = (f.column(col) - weighted) / lambda;
            self.weights.set_column(col, &column);
        }
        Ok(())
    }
}
