use nalgebra::DMatrix;

use crate::em::{
    EmConfig, SparseIndex, initialize_sigma2, posterior_stats, validate_clouds, variance_floor,
};
use crate::{Error, Result};

/// Configuration for statistical-shape-model (atlas) registration.
#[derive(Clone, Debug)]
pub struct AtlasConfig {
    /// Shared EM settings.
    pub em: EmConfig,
    /// Shape-model eigenvalues, one per mode column.
    pub eigenvalues: Vec<f64>,
    /// Strength of the Mahalanobis prior on the shape coefficients.
    pub lambda_regularization: f64,
    /// Center and scale both clouds internally before registration;
    /// results are mapped back to the original frame.
    pub normalize: bool,
    /// Re-estimate rotation/scale/translation each iteration.
    pub optimize_similarity: bool,
    /// Include an isotropic scale in the similarity estimate.
    pub with_scale: bool,
    /// Opt-in adaptive sparse E-step (requires `em.k = Some(k)`). When
    /// `Some(scale)`, the E-step runs **dense** while the variance is large
    /// and permanently switches to the `k`-nearest-neighbor sparse E-step
    /// once `sigma2 < (‖extent(X)‖ / scale)²`, where `extent(X)` is the
    /// per-axis span of the target cloud. This mirrors the reference
    /// implementation's `use_kdtree` default (`scale = 10`); it avoids the
    /// bias of a fixed-`k` truncation while the variance is still large.
    /// `None` (default) keeps the current behavior: a set `em.k` applies
    /// from the first iteration, `em.k = None` stays exact.
    pub kdtree_radius_scale: Option<f64>,
    /// Starting shape coefficients (defaults to zero).
    pub initial_coefficients: Option<Vec<f64>>,
    /// Starting rotation (defaults to identity).
    pub initial_rotation: Option<DMatrix<f64>>,
    /// Starting scale (used only when `with_scale` is set).
    pub initial_scale: f64,
    /// Starting translation (defaults to zero).
    pub initial_translation: Option<Vec<f64>>,
    /// Anchored keypoint correspondences: each entry pairs a source-model
    /// vertex index with its known target coordinate (in the original target
    /// frame). Unlike [`Self::initial_rotation`] these persist through every
    /// EM iteration as a soft data term, keeping the fit in the correct basin
    /// while shape and pose are optimized. Empty (default) disables the term.
    pub landmarks: Vec<(usize, Vec<f64>)>,
    /// Gain on the landmark term, expressed as a multiple of the average
    /// per-source posterior mass. Each landmark contributes
    /// `landmark_weight · (Np / M)` of correspondence mass, so its influence
    /// tracks the annealing level rather than being swamped as `sigma2`
    /// shrinks. `0` disables the term even when `landmarks` is non-empty.
    ///
    /// This is a heuristic scaling: the effective landmark variance is
    /// `sigma2 / mass`, so it *shrinks* with the surface fit, the constraint
    /// strength drifts with posterior occupancy / outlier weight, and the
    /// landmark residuals leak into [`AtlasResult::sigma2`]. Prefer
    /// [`Self::landmark_error`] for a principled, transferable formulation.
    pub landmark_weight: f64,
    /// Landmark localization variance `τ²` (in squared target-coordinate
    /// units), the principled alternative to [`Self::landmark_weight`]. When
    /// `Some`, each landmark is folded in with mass `a = sigma2 / τ²`, giving a
    /// *fixed* effective landmark variance `τ²` independent of annealing —
    /// exactly the `DeformableConfig::constraint_error` scheme. Surface
    /// `sigma2` is then estimated from the ordinary CPD correspondences only,
    /// so [`AtlasResult::sigma2`] stays a clean surface-residual variance and
    /// the shape prior is not incidentally relaxed by the anchors. Takes
    /// precedence over `landmark_weight` when set. `None` (default) keeps the
    /// heuristic weight behavior.
    pub landmark_error: Option<f64>,
}

impl Default for AtlasConfig {
    fn default() -> Self {
        Self {
            em: EmConfig::default(),
            eigenvalues: Vec::new(),
            lambda_regularization: 0.1,
            normalize: false,
            optimize_similarity: true,
            with_scale: true,
            kdtree_radius_scale: None,
            initial_coefficients: None,
            initial_rotation: None,
            initial_scale: 1.0,
            initial_translation: None,
            landmarks: Vec::new(),
            landmark_weight: 0.0,
            landmark_error: None,
        }
    }
}

/// Output of [`AtlasRegistration::register`].
#[derive(Clone, Debug)]
pub struct AtlasResult {
    /// Deformed-and-transformed model points in the target frame (`M×D`).
    pub points: DMatrix<f64>,
    /// Estimated shape coefficients, one per mode.
    pub coefficients: Vec<f64>,
    /// Rotation in row-vector convention: apply as `y·R`.
    pub rotation: DMatrix<f64>,
    /// Estimated isotropic scale (1 when disabled).
    pub scale: f64,
    /// Translation vector (length `D`).
    pub translation: Vec<f64>,
    /// Final Gaussian variance.
    pub sigma2: f64,
    /// Number of EM iterations performed.
    pub iterations: usize,
    /// Final convergence-criterion value.
    pub difference: f64,
    /// Negative log-likelihood of the mixture at the final E-step (the
    /// "trajectory" data objective). `f64::INFINITY` if no EM step ran.
    pub negative_log_likelihood: f64,
    /// RMS of the anchored-landmark residuals `‖similarity(mean+modes·b)ᵢ − qᵢ‖`
    /// at the final iteration, in the original target frame. `f64::NAN` when no
    /// landmarks were supplied. Reported separately from [`Self::sigma2`] so the
    /// surface-fit variance and the landmark fit can be read independently.
    pub landmark_rms: f64,
}

impl AtlasResult {
    /// Reconstruct the fitted shape at full resolution.
    ///
    /// Applies the estimated coefficients to a (typically denser)
    /// `mean`/`modes` pair, then the fitted similarity transform, giving
    /// `similarity(mean + modes·b)` in the target frame. The mesh the atlas
    /// was *registered* on can be a subsample; pass the full-resolution
    /// mean and modes here to rebuild the dense fitted model exactly.
    ///
    /// `mean` is `(P, D)` and `modes` is `(P·D, rank)` in point-major row
    /// order — the same layout [`AtlasRegistration::new`] expects, at
    /// whatever resolution you like. The number of modes must match the
    /// fitted coefficients.
    pub fn reconstruct(&self, mean: &DMatrix<f64>, modes: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        let d = self.rotation.nrows();
        if mean.ncols() != d {
            return Err(Error::DimensionMismatch);
        }
        if modes.nrows() != mean.len() {
            return Err(Error::InvalidShape("modes"));
        }
        if modes.ncols() != self.coefficients.len() {
            return Err(Error::InvalidShape("modes"));
        }
        if !mean.iter().chain(modes.iter()).all(|v| v.is_finite()) {
            return Err(Error::NonFiniteInput);
        }
        let deformed = add_deformation(mean, modes, &self.coefficients);
        let mut out = DMatrix::zeros(mean.nrows(), d);
        apply_similarity(
            &deformed,
            &self.rotation,
            self.scale,
            &self.translation,
            &mut out,
        );
        Ok(out)
    }

    /// Apply only the fitted similarity transform (rotation, scale,
    /// translation) to arbitrary points `z` of shape `(P, D)`, returning
    /// `scale · z · R + t`.
    ///
    /// Unlike [`Self::reconstruct`], this ignores the shape deformation —
    /// use it to move points that live in the model frame (not model
    /// vertices) into the target frame under the recovered pose.
    pub fn apply_similarity(&self, z: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        let d = self.rotation.nrows();
        if z.ncols() != d {
            return Err(Error::DimensionMismatch);
        }
        if !z.iter().all(|v| v.is_finite()) {
            return Err(Error::NonFiniteInput);
        }
        let mut out = DMatrix::zeros(z.nrows(), d);
        apply_similarity(z, &self.rotation, self.scale, &self.translation, &mut out);
        Ok(out)
    }
}

/// Statistical-shape-model registration: fits `mean + modes·b` plus a
/// similarity transform to the fixed target cloud `x`. `modes` is
/// `(M·D)×rank` with rows in point-major order.
pub struct AtlasRegistration<'a> {
    x: &'a DMatrix<f64>,
    mean: &'a DMatrix<f64>,
    modes: &'a DMatrix<f64>,
    config: AtlasConfig,
}

impl<'a> AtlasRegistration<'a> {
    /// Validate the inputs and build a registration; `x` is the fixed
    /// target, `mean` the model mean shape, `modes` the deformation modes.
    pub fn new(
        x: &'a DMatrix<f64>,
        mean: &'a DMatrix<f64>,
        modes: &'a DMatrix<f64>,
        config: AtlasConfig,
    ) -> Result<Self> {
        validate_clouds(x, mean)?;
        config.em.validate()?;
        if modes.nrows() != mean.len() {
            return Err(Error::InvalidShape("modes"));
        }
        if !modes.iter().all(|value| value.is_finite()) {
            return Err(Error::NonFiniteInput);
        }
        if modes.ncols() != config.eigenvalues.len() || config.eigenvalues.is_empty() {
            return Err(Error::InvalidShape("eigenvalues"));
        }
        if config
            .eigenvalues
            .iter()
            .any(|v| !v.is_finite() || *v <= 0.0)
        {
            return Err(Error::PositiveParameter("eigenvalues"));
        }
        if !config.lambda_regularization.is_finite() || config.lambda_regularization < 0.0 {
            return Err(Error::PositiveParameter("lambda_regularization"));
        }
        if config
            .kdtree_radius_scale
            .is_some_and(|scale| !scale.is_finite() || scale <= 0.0)
        {
            return Err(Error::PositiveParameter("kdtree_radius_scale"));
        }
        let d = x.ncols();
        if config
            .initial_rotation
            .as_ref()
            .is_some_and(|r| r.shape() != (d, d))
        {
            return Err(Error::InvalidShape("initial_rotation"));
        }
        if config
            .initial_translation
            .as_ref()
            .is_some_and(|t| t.len() != d)
        {
            return Err(Error::InvalidShape("initial_translation"));
        }
        if config
            .initial_coefficients
            .as_ref()
            .is_some_and(|b| b.len() != modes.ncols())
        {
            return Err(Error::InvalidShape("initial_coefficients"));
        }
        if config.with_scale && (!config.initial_scale.is_finite() || config.initial_scale <= 0.0) {
            return Err(Error::PositiveParameter("initial_scale"));
        }
        if !config.landmark_weight.is_finite() || config.landmark_weight < 0.0 {
            return Err(Error::PositiveParameter("landmark_weight"));
        }
        if config
            .landmark_error
            .is_some_and(|t2| !t2.is_finite() || t2 <= 0.0)
        {
            return Err(Error::PositiveParameter("landmark_error"));
        }
        for (index, point) in &config.landmarks {
            if *index >= mean.nrows() {
                return Err(Error::InvalidShape("landmarks"));
            }
            if point.len() != d || !point.iter().all(|value| value.is_finite()) {
                return Err(Error::InvalidShape("landmarks"));
            }
        }
        if config
            .initial_rotation
            .as_ref()
            .is_some_and(|r| !r.iter().all(|value| value.is_finite()))
            || config
                .initial_translation
                .as_ref()
                .is_some_and(|t| !t.iter().all(|value| value.is_finite()))
            || config
                .initial_coefficients
                .as_ref()
                .is_some_and(|b| !b.iter().all(|value| value.is_finite()))
        {
            return Err(Error::NonFiniteInput);
        }
        Ok(Self {
            x,
            mean,
            modes,
            config,
        })
    }

    /// Run EM to convergence or `max_iterations`.
    pub fn register(&self) -> Result<AtlasResult> {
        let (m, d, rank) = (self.mean.nrows(), self.mean.ncols(), self.modes.ncols());
        let (x, y, modes, centroid, target_scale) = if self.config.normalize {
            let (xn, centroid, scale) = normalize(self.x);
            let mut yn = self.mean.clone();
            for i in 0..m {
                for j in 0..d {
                    yn[(i, j)] = (yn[(i, j)] - centroid[j]) / scale;
                }
            }
            (xn, yn, self.modes / scale, centroid, scale)
        } else {
            (
                self.x.clone(),
                self.mean.clone(),
                self.modes.clone(),
                vec![0.0; d],
                1.0,
            )
        };
        let mut coefficients = self
            .config
            .initial_coefficients
            .clone()
            .unwrap_or_else(|| vec![0.0; rank]);
        let mut previous_coefficients = coefficients.clone();
        let mut r = self
            .config
            .initial_rotation
            .clone()
            .unwrap_or_else(|| DMatrix::identity(d, d));
        let mut scale = if self.config.with_scale {
            self.config.initial_scale
        } else {
            1.0
        };
        let mut t = self
            .config
            .initial_translation
            .clone()
            .unwrap_or_else(|| vec![0.0; d]);
        if self.config.normalize && self.config.initial_translation.is_some() {
            let tw = t.clone();
            for j in 0..d {
                t[j] = (scale * (0..d).map(|q| centroid[q] * r[(q, j)]).sum::<f64>() + tw[j]
                    - centroid[j])
                    / target_scale;
            }
        }
        let mut deformed = add_deformation(&y, &modes, &coefficients);
        let mut ty = DMatrix::zeros(m, d);
        apply_similarity(&deformed, &r, scale, &t, &mut ty);
        let mut sigma2 = self.config.em.sigma2.unwrap_or(initialize_sigma2(&x, &y)?);
        let extent2: f64 = (0..d)
            .map(|j| {
                let lo = (0..x.nrows())
                    .map(|i| x[(i, j)])
                    .fold(f64::INFINITY, f64::min);
                let hi = (0..x.nrows())
                    .map(|i| x[(i, j)])
                    .fold(f64::NEG_INFINITY, f64::max);
                (hi - lo).powi(2)
            })
            .sum();
        let variance_floor = (1e-6 * extent2 / d as f64).max(variance_floor(&x, d));
        // Anchored-keypoint term is enabled by either a positive heuristic
        // weight or an explicit landmark variance.
        let landmark_error_work = self.config.landmark_error.map(|t2| {
            if self.config.normalize {
                t2 / (target_scale * target_scale)
            } else {
                t2
            }
        });
        let landmarks_active = self.config.landmark_weight > 0.0 || landmark_error_work.is_some();
        // Landmark target coordinates mapped into the working frame.
        let landmarks_work: Vec<(usize, Vec<f64>)> = if landmarks_active {
            self.config
                .landmarks
                .iter()
                .map(|(index, point)| {
                    let mapped = if self.config.normalize {
                        (0..d)
                            .map(|j| (point[j] - centroid[j]) / target_scale)
                            .collect()
                    } else {
                        point.clone()
                    };
                    (*index, mapped)
                })
                .collect()
        } else {
            Vec::new()
        };
        let mut landmark_rms = f64::NAN;
        let mut diff = f64::INFINITY;
        let mut iterations = 0;
        let mut last_nll = f64::INFINITY;
        let sparse_index = self.config.em.k.map(|k| SparseIndex::new(&x, k));
        // Adaptive sparse switch: with `kdtree_radius_scale = Some(scale)`
        // the E-step stays dense until `sigma2 < (‖extent(X)‖ / scale)²`,
        // then permanently uses the k-NN sparse index. `extent2` is
        // `‖max(X) − min(X)‖²`. Without it, a set `k` applies from the
        // start (`sparse_active` begins true).
        let sparse_threshold = self
            .config
            .kdtree_radius_scale
            .map(|scale| extent2 / (scale * scale));
        let mut sparse_active = sparse_threshold.is_none();
        // While adaptive and still dense, the E-step must dispatch on a
        // `k = None` config (the sparse/dense choice keys on `config.k`, not
        // on whether an index is passed).
        let dense_em = EmConfig {
            k: None,
            ..self.config.em.clone()
        };
        // Per-iteration scratch, allocated once and overwritten each pass to
        // avoid re-allocating on every EM step (pose initialization runs
        // this loop hundreds of times).
        let mut residual_matrix = DMatrix::zeros(m * d, 1);
        let mut scaled_modes = modes.clone();
        while iterations < self.config.em.max_iterations && diff > self.config.em.tolerance {
            if !sparse_active && sparse_threshold.is_some_and(|threshold| sigma2 < threshold) {
                sparse_active = true;
            }
            let (em, active_index) = if sparse_active {
                (&self.config.em, sparse_index.as_ref())
            } else {
                (&dense_em, None)
            };
            let mut stats = posterior_stats(&x, &ty, sigma2, em, false, active_index);
            if stats.np <= f64::MIN_POSITIVE {
                return Err(Error::SingularSystem);
            }
            last_nll = stats.negative_log_likelihood;
            // Anchored-keypoint term: fold each landmark into the sufficient
            // statistics as a virtual correspondence between source vertex
            // `index` and its known target point `q`. It then flows into both
            // M-steps — the coefficient solve (via the per-source target
            // estimate `px/p1`) and the weighted-similarity solve — with no
            // separate solver.
            //
            // Mass `a` sets the effective landmark variance `sigma2 / a`:
            //   * `landmark_error = Some(τ²)`  →  `a = sigma2 / τ²`  (fixed
            //     variance `τ²`, independent of annealing; the principled mode).
            //   * else                          →  `a = weight · Np / M`  (the
            //     heuristic, occupancy-scaled mode).
            // `np_surface` is the pre-augmentation posterior mass; it and the
            // landmark contributions below let the variance update stay
            // surface-only in the principled mode.
            let np_surface = stats.np;
            let landmark_mass = if landmarks_work.is_empty() {
                0.0
            } else {
                match landmark_error_work {
                    Some(t2) => sigma2 / t2.max(f64::MIN_POSITIVE),
                    None => self.config.landmark_weight * np_surface / m as f64,
                }
            };
            for (index, q) in &landmarks_work {
                for (j, value) in q.iter().enumerate() {
                    stats.px[(*index, j)] += landmark_mass * value;
                }
                stats.p1[*index] += landmark_mass;
                stats.np += landmark_mass;
            }
            // residual = model-frame target estimate minus the mean shape,
            // written straight into the (M·D)×1 right-hand side buffer.
            for i in 0..m {
                for j in 0..d {
                    let inverse = (0..d)
                        .map(|q| centered_at(&stats.px, i, q, stats.p1[i], &t) * r[(j, q)])
                        .sum::<f64>()
                        / scale;
                    residual_matrix[(i * d + j, 0)] = inverse - y[(i, j)];
                }
            }
            // system = Uᵀ·diag(w)·U and rhs = (diag(w)·U)ᵀ·residual through
            // one row-scaled copy of the modes and two GEMMs.
            for (mut destination, source) in scaled_modes.column_iter_mut().zip(modes.column_iter())
            {
                for (flat, (value, &original)) in
                    destination.iter_mut().zip(source.iter()).enumerate()
                {
                    *value = original * stats.p1[flat / d];
                }
            }
            let mut system = modes.tr_mul(&scaled_modes);
            let rhs = scaled_modes.tr_mul(&residual_matrix);
            let gamma = self.config.lambda_regularization * sigma2 / scale.powi(2);
            for a in 0..rank {
                system[(a, a)] += gamma / self.config.eigenvalues[a].max(f64::EPSILON);
            }
            let solved = system.cholesky().ok_or(Error::SingularSystem)?.solve(&rhs);
            for a in 0..rank {
                coefficients[a] = solved[(a, 0)];
            }
            add_deformation_into(&mut deformed, &y, &modes, &coefficients);
            if self.config.optimize_similarity {
                let (nr, ns, nt) = weighted_similarity(
                    &deformed,
                    &stats.p1,
                    &stats.px,
                    stats.np,
                    self.config.with_scale,
                    scale,
                )?;
                r = nr;
                scale = ns;
                t = nt;
            }
            apply_similarity(&deformed, &r, scale, &t, &mut ty);
            let bdiff = coefficients
                .iter()
                .zip(&previous_coefficients)
                .map(|(a, b)| (a - b).abs())
                .sum::<f64>()
                / rank as f64;
            previous_coefficients.copy_from_slice(&coefficients);
            let previous_sigma = sigma2;
            let xpx_surface: f64 = (0..x.nrows())
                .map(|i| stats.pt1[i] * (0..d).map(|j| x[(i, j)].powi(2)).sum::<f64>())
                .sum();
            let ypy_aug: f64 = (0..m)
                .map(|i| stats.p1[i] * (0..d).map(|j| ty[(i, j)].powi(2)).sum::<f64>())
                .sum();
            let cross_aug: f64 = (0..m)
                .map(|i| (0..d).map(|j| ty[(i, j)] * stats.px[(i, j)]).sum::<f64>())
                .sum();
            // Landmark contributions to the (augmented) sums, plus the raw
            // landmark residual for reporting.
            let (mut lm_ypy, mut lm_cross, mut lm_xpx, mut lm_resid2) = (0.0, 0.0, 0.0, 0.0);
            for (index, q) in &landmarks_work {
                let ty_norm2 = (0..d).map(|j| ty[(*index, j)].powi(2)).sum::<f64>();
                let dot = (0..d).map(|j| ty[(*index, j)] * q[j]).sum::<f64>();
                lm_ypy += landmark_mass * ty_norm2;
                lm_cross += landmark_mass * dot;
                lm_xpx += landmark_mass * q.iter().map(|v| v * v).sum::<f64>();
                lm_resid2 += (0..d)
                    .map(|j| (ty[(*index, j)] - q[j]).powi(2))
                    .sum::<f64>();
            }
            // Principled (`landmark_error`) mode estimates `sigma2` from the
            // surface correspondences ONLY: drop the landmark energy/mass so the
            // variance stays a clean surface residual and does not relax the
            // shape prior. Heuristic (`landmark_weight`) mode keeps the legacy
            // behavior where landmarks enter the variance too.
            let (xpx, ypy, cross, denom) = if landmark_error_work.is_some() {
                (
                    xpx_surface,
                    ypy_aug - lm_ypy,
                    cross_aug - lm_cross,
                    np_surface,
                )
            } else {
                (xpx_surface + lm_xpx, ypy_aug, cross_aug, stats.np)
            };
            sigma2 = ((xpx - 2.0 * cross + ypy) / (denom * d as f64)).max(variance_floor);
            if !landmarks_work.is_empty() {
                let rms = (lm_resid2 / landmarks_work.len() as f64).sqrt();
                landmark_rms = if self.config.normalize {
                    rms * target_scale
                } else {
                    rms
                };
            }
            diff = ((sigma2 - previous_sigma).abs() / (sigma2 + 1e-8)).max(bdiff);
            iterations += 1;
        }
        let mut points = ty.clone();
        if self.config.normalize {
            for i in 0..m {
                for j in 0..d {
                    points[(i, j)] = points[(i, j)] * target_scale + centroid[j];
                }
            }
        }
        let translation = if self.config.normalize {
            (0..d)
                .map(|j| {
                    -scale * (0..d).map(|q| centroid[q] * r[(q, j)]).sum::<f64>()
                        + target_scale * t[j]
                        + centroid[j]
                })
                .collect()
        } else {
            t
        };
        let sigma2 = if self.config.normalize {
            sigma2 * target_scale * target_scale
        } else {
            sigma2
        };
        Ok(AtlasResult {
            points,
            coefficients,
            rotation: r,
            scale,
            translation,
            sigma2,
            iterations,
            difference: diff,
            negative_log_likelihood: last_nll,
            landmark_rms,
        })
    }
}

fn normalize(points: &DMatrix<f64>) -> (DMatrix<f64>, Vec<f64>, f64) {
    let mut centroid = vec![0.0; points.ncols()];
    for j in 0..points.ncols() {
        centroid[j] =
            (0..points.nrows()).map(|i| points[(i, j)]).sum::<f64>() / points.nrows() as f64;
    }
    let scale = ((0..points.nrows())
        .map(|i| {
            (0..points.ncols())
                .map(|j| (points[(i, j)] - centroid[j]).powi(2))
                .sum::<f64>()
        })
        .sum::<f64>()
        / points.nrows() as f64)
        .sqrt()
        .max(f64::EPSILON);
    let normalized = DMatrix::from_fn(points.nrows(), points.ncols(), |i, j| {
        (points[(i, j)] - centroid[j]) / scale
    });
    (normalized, centroid, scale)
}

fn add_deformation(y: &DMatrix<f64>, modes: &DMatrix<f64>, b: &[f64]) -> DMatrix<f64> {
    let mut out = DMatrix::zeros(y.nrows(), y.ncols());
    add_deformation_into(&mut out, y, modes, b);
    out
}

/// `out = y + reshape(modes · b)`, writing into a preallocated buffer.
fn add_deformation_into(out: &mut DMatrix<f64>, y: &DMatrix<f64>, modes: &DMatrix<f64>, b: &[f64]) {
    let d = y.ncols();
    for i in 0..y.nrows() {
        for j in 0..d {
            out[(i, j)] = y[(i, j)]
                + (0..b.len())
                    .map(|k| modes[(i * d + j, k)] * b[k])
                    .sum::<f64>();
        }
    }
}

fn apply_similarity(
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

fn centered_at(px: &DMatrix<f64>, row: usize, col: usize, mass: f64, t: &[f64]) -> f64 {
    px[(row, col)] / mass.max(f64::MIN_POSITIVE) - t[col]
}

fn weighted_similarity(
    y: &DMatrix<f64>,
    weights: &[f64],
    px: &DMatrix<f64>,
    total: f64,
    with_scale: bool,
    current_scale: f64,
) -> Result<(DMatrix<f64>, f64, Vec<f64>)> {
    let d = y.ncols();
    let mux: Vec<_> = (0..d)
        .map(|j| (0..px.nrows()).map(|i| px[(i, j)]).sum::<f64>() / total)
        .collect();
    let muy: Vec<_> = (0..d)
        .map(|j| (0..y.nrows()).map(|i| weights[i] * y[(i, j)]).sum::<f64>() / total)
        .collect();
    let mut c = DMatrix::zeros(d, d);
    for row in 0..d {
        for col in 0..d {
            c[(row, col)] = (0..y.nrows())
                .map(|i| y[(i, row)] * px[(i, col)])
                .sum::<f64>()
                - total * muy[row] * mux[col];
        }
    }
    let svd = c.clone().svd(true, true);
    let u = svd.u.ok_or(Error::SingularSystem)?;
    let vt = svd.v_t.ok_or(Error::SingularSystem)?;
    let mut correction: DMatrix<f64> = DMatrix::identity(d, d);
    correction[(d - 1, d - 1)] = (u.determinant() * vt.determinant()).signum();
    let a = &u * correction * &vt;
    // C = Yᵀ P X, so the direct row-vector rotation is U·Vᵀ.
    let r = a.clone();
    let scale = if with_scale {
        let numerator = (a.transpose() * &c).trace();
        let denominator = (0..y.nrows())
            .map(|i| weights[i] * (0..d).map(|j| y[(i, j)].powi(2)).sum::<f64>())
            .sum::<f64>()
            - total * muy.iter().map(|v| v * v).sum::<f64>();
        let max_abs_y = y
            .iter()
            .fold(0.0_f64, |maximum, &value| maximum.max(value.abs()));
        let spread_floor = (64.0 * (y.nrows() * d) as f64 * (f64::EPSILON * max_abs_y).powi(2))
            .max(f64::MIN_POSITIVE);
        if denominator > spread_floor {
            numerator / denominator
        } else {
            current_scale
        }
    } else {
        1.0
    };
    let translation = (0..d)
        .map(|j| mux[j] - scale * (0..d).map(|q| muy[q] * r[(q, j)]).sum::<f64>())
        .collect();
    Ok((r, scale, translation))
}

#[cfg(test)]
mod landmark_tests {
    use super::{AtlasConfig, AtlasRegistration};
    use crate::EmConfig;
    use nalgebra::DMatrix;

    // Deterministic 8-point source and a rank-2 mode basis (point-major).
    fn model() -> (DMatrix<f64>, DMatrix<f64>, Vec<f64>) {
        let source = DMatrix::from_fn(8, 3, |i, j| {
            let i = i as f64;
            let j = j as f64;
            (0.7 * i + 1.3 * j).sin() * 1.4 + 0.11 * i - 0.2 * j
        });
        let modes = DMatrix::from_fn(8 * 3, 2, |row, k| {
            let row = row as f64;
            let k = k as f64;
            ((row + 1.0 + 5.0 * k) * 0.21).sin() * 0.5
        });
        (source, modes, vec![1.0, 1.0])
    }

    fn base_config() -> AtlasConfig {
        AtlasConfig {
            em: EmConfig {
                // A short fixed budget: the Mahalanobis prior vanishes as
                // sigma2 anneals to zero, so the regularization only bites
                // before convergence. Six iterations keeps the plain fit
                // under-shooting, giving the landmark term visible work to do.
                max_iterations: 6,
                tolerance: 0.0,
                outlier_weight: 0.0,
                ..Default::default()
            },
            eigenvalues: vec![1.0, 1.0],
            // High regularization deliberately under-fits the shape, so the
            // landmark data term has visible work to do.
            lambda_regularization: 6.0,
            normalize: false,
            optimize_similarity: false,
            with_scale: false,
            ..Default::default()
        }
    }

    fn landmark_residual(recon: &DMatrix<f64>, landmarks: &[(usize, Vec<f64>)]) -> f64 {
        landmarks
            .iter()
            .map(|(i, q)| (0..3).map(|j| (recon[(*i, j)] - q[j]).powi(2)).sum::<f64>())
            .sum::<f64>()
            .sqrt()
    }

    #[test]
    fn landmarks_pull_anchored_vertices_toward_targets() {
        let (source, modes, _) = model();
        let truth = [0.9_f64, -0.7];
        // Pure-shape target: source + modes·truth (identity pose).
        let target = DMatrix::from_fn(8, 3, |i, j| {
            source[(i, j)]
                + (0..2)
                    .map(|k| modes[(i * 3 + j, k)] * truth[k])
                    .sum::<f64>()
        });
        let landmarks: Vec<(usize, Vec<f64>)> = [1usize, 4, 6]
            .iter()
            .map(|&i| (i, (0..3).map(|j| target[(i, j)]).collect()))
            .collect();

        let plain = AtlasRegistration::new(&target, &source, &modes, base_config())
            .unwrap()
            .register()
            .unwrap();
        let anchored_cfg = AtlasConfig {
            landmarks: landmarks.clone(),
            landmark_weight: 40.0,
            ..base_config()
        };
        let anchored = AtlasRegistration::new(&target, &source, &modes, anchored_cfg)
            .unwrap()
            .register()
            .unwrap();

        let plain_recon = plain.reconstruct(&source, &modes).unwrap();
        let anchored_recon = anchored.reconstruct(&source, &modes).unwrap();
        let plain_res = landmark_residual(&plain_recon, &landmarks);
        let anchored_res = landmark_residual(&anchored_recon, &landmarks);

        // The anchored fit places its landmark vertices much closer to their
        // targets than the over-regularized plain fit, and does so by using
        // more of the true coefficient (it stops under-fitting).
        assert!(
            anchored_res < 0.25 * plain_res,
            "anchored {anchored_res} vs plain {plain_res}"
        );
        let plain_norm: f64 = plain.coefficients.iter().map(|c| c * c).sum();
        let anchored_norm: f64 = anchored.coefficients.iter().map(|c| c * c).sum();
        assert!(anchored_norm > plain_norm);
    }

    #[test]
    fn zero_landmark_weight_is_a_noop() {
        let (source, modes, _) = model();
        let target = DMatrix::from_fn(8, 3, |i, j| source[(i, j)] + 0.05 * (i as f64 - j as f64));
        let landmarks: Vec<(usize, Vec<f64>)> =
            vec![(0, vec![0.0, 0.0, 0.0]), (5, vec![9.0, 9.0, 9.0])];
        let without = AtlasRegistration::new(&target, &source, &modes, base_config())
            .unwrap()
            .register()
            .unwrap();
        // Landmarks present, but weight 0 must not perturb the result at all.
        let disabled_cfg = AtlasConfig {
            landmarks,
            landmark_weight: 0.0,
            ..base_config()
        };
        let disabled = AtlasRegistration::new(&target, &source, &modes, disabled_cfg)
            .unwrap()
            .register()
            .unwrap();
        for (a, b) in without.coefficients.iter().zip(&disabled.coefficients) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn atlas_rejects_invalid_landmarks() {
        let (source, modes, _) = model();
        let target = source.clone();
        let bad_index = AtlasConfig {
            landmarks: vec![(99, vec![0.0, 0.0, 0.0])],
            landmark_weight: 1.0,
            ..base_config()
        };
        assert!(AtlasRegistration::new(&target, &source, &modes, bad_index).is_err());
        let bad_dim = AtlasConfig {
            landmarks: vec![(0, vec![0.0, 0.0])],
            landmark_weight: 1.0,
            ..base_config()
        };
        assert!(AtlasRegistration::new(&target, &source, &modes, bad_dim).is_err());
        let bad_weight = AtlasConfig {
            landmarks: vec![(0, vec![0.0, 0.0, 0.0])],
            landmark_weight: -1.0,
            ..base_config()
        };
        assert!(AtlasRegistration::new(&target, &source, &modes, bad_weight).is_err());
        let bad_error = AtlasConfig {
            landmarks: vec![(0, vec![0.0, 0.0, 0.0])],
            landmark_error: Some(-1.0),
            ..base_config()
        };
        assert!(AtlasRegistration::new(&target, &source, &modes, bad_error).is_err());
    }

    #[test]
    fn landmark_error_keeps_sigma2_surface_clean() {
        let (source, modes, _) = model();
        // Rigid target plus a small NON-model surface perturbation, so there is
        // a real surface residual for sigma2 to measure.
        let (c, s) = (0.2_f64.cos(), 0.2_f64.sin());
        let rmat = DMatrix::from_row_slice(3, 3, &[c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0]);
        let target = DMatrix::from_fn(8, 3, |i, j| {
            (0..3).map(|q| source[(i, q)] * rmat[(q, j)]).sum::<f64>()
                + 0.01 * ((i * 3 + j) as f64).sin()
        });
        let cfg = || AtlasConfig {
            em: EmConfig {
                max_iterations: 200,
                tolerance: 1e-10,
                outlier_weight: 0.0,
                ..Default::default()
            },
            eigenvalues: vec![1.0, 1.0],
            lambda_regularization: 0.5,
            optimize_similarity: true,
            with_scale: false,
            ..Default::default()
        };
        let base = AtlasRegistration::new(&target, &source, &modes, cfg())
            .unwrap()
            .register()
            .unwrap();
        assert!(base.landmark_rms.is_nan());
        // Landmarks placed exactly where the no-landmark fit put those vertices:
        // consistent constraints must not move the surface fit.
        let landmarks: Vec<(usize, Vec<f64>)> = [1usize, 4, 6]
            .iter()
            .map(|&i| (i, (0..3).map(|j| base.points[(i, j)]).collect()))
            .collect();
        let principled = AtlasRegistration::new(
            &target,
            &source,
            &modes,
            AtlasConfig {
                landmarks: landmarks.clone(),
                landmark_error: Some(1e-4),
                ..cfg()
            },
        )
        .unwrap()
        .register()
        .unwrap();
        let heavy = AtlasRegistration::new(
            &target,
            &source,
            &modes,
            AtlasConfig {
                landmarks,
                landmark_weight: 200.0,
                ..cfg()
            },
        )
        .unwrap()
        .register()
        .unwrap();
        // Consistent landmarks are already satisfied by both fits.
        assert!(principled.landmark_rms < 0.02);
        // Principled surface sigma2 is (essentially) unchanged by the anchors.
        assert!(
            (principled.sigma2 - base.sigma2).abs() < 0.1 * base.sigma2,
            "principled {} vs base {}",
            principled.sigma2,
            base.sigma2
        );
        // Heuristic sigma2 is diluted by the added landmark mass in its
        // denominator even though the landmark residual is ~0 — the
        // contamination the principled mode removes.
        assert!(
            heavy.sigma2 < 0.5 * base.sigma2,
            "heavy {} vs base {}",
            heavy.sigma2,
            base.sigma2
        );
    }
}
