//! Shape completion and calibrated per-point uncertainty (optional).
//!
//! The atlas is a linear-Gaussian shape model: a shape is `μ + U·b` with
//! `b ~ N(0, Λ)`, plus a similarity transform and isotropic noise. Given a
//! *partial* observation of the shape, the posterior over the coefficients
//! `b` is Gaussian and closed form. This module keeps that posterior rather
//! than collapsing it to a point, which yields both a completion (the
//! posterior mean shape, evaluated everywhere) and per-point predictive
//! uncertainty (the pushforward of the coefficient covariance).
//!
//! Everything here is post-hoc: it consumes a fitted [`crate::AtlasResult`]
//! or explicit model inputs and never runs inside `register()`, so the
//! complete-data registration paths are unaffected.
//!
//! # Frames
//!
//! The model (`μ`, `U`, `Λ`) lives in the *model frame*; the observed data
//! lives in the *target frame*, related by a similarity `z ↦ scale·z·R + t`.
//! The posterior is computed in the model frame and mapped to the target
//! frame for outputs. The effective model-frame noise is `σ²_target/scale²`.
//!
//! # Prior temperature
//!
//! The atlas `lambda_regularization` multiplies the prior precision: the atlas
//! M-step equals the true Bayesian posterior mean only when it is `1`.
//! Completion uses the statistically correct prior (`temperature = 1`) by
//! default; set [`PosteriorOptions::prior_temperature`] (the legacy name for
//! this precision multiplier) to the fitted `lambda_regularization` for the
//! same shape prior. Coefficient agreement additionally requires convergence
//! and identical conditioning information, including any visibility filtering.
//!
//! # Discrepancy (model inadequacy)
//!
//! The Gaussian predictive variance is only trustworthy if the model can
//! actually represent the shape. When enabled (the default), completion
//! estimates a *discrepancy variance* from the part of the observed
//! residual the fitted model cannot explain, and folds it into the
//! predictive uncertainty (a simple Kennedy–O'Hagan discrepancy). This
//! inflates the intervals for out-of-distribution fragments — a shape the
//! model fits poorly where it *was* observed is one to distrust where it
//! was *not*. It affects only the predictive variance, not the coefficient
//! posterior. Conformal calibration (in the Python package) remains the
//! way to turn these variances into intervals with guaranteed coverage.

use nalgebra::DMatrix;

use crate::atlas_state::{
    MixingWeightMap, effective_outlier_weight, mixing_factors, validate_mixing_weights,
};
use crate::em::{EmConfig, posterior_stats_weighted, validate_clouds};
use crate::{AtlasResult, Error, Result};

/// Options controlling how a [`ShapePosterior`] is built from a fit.
#[derive(Clone, Debug)]
pub struct PosteriorOptions {
    /// Expected fraction of the model actually observed, in `(0, 1]`. When
    /// set, the highest-mass `completeness` fraction of model points is
    /// treated as observed and the rest as purely predicted — a robust
    /// alternative to a fixed weight threshold. `None` keeps every point
    /// whose posterior mass exceeds [`Self::visibility_floor`].
    pub completeness: Option<f64>,
    /// Absolute posterior-mass floor below which a model point is treated
    /// as unobserved (used only when `completeness` is `None`).
    pub visibility_floor: f64,
    /// Prior precision multiplier (legacy name; see module docs). `1.0` is
    /// the original shape prior; `lambda_regularization` matches the fit's
    /// prior strength. Precision is `prior_temperature * Λ⁻¹`.
    pub prior_temperature: f64,
    /// Override the fitted outlier weight when forming the posterior.
    /// `None` inherits the registration's observation model; `Some(0.0)`
    /// explicitly requests clean soft assignments.
    pub outlier_weight: Option<f64>,
    /// Estimate a model-inadequacy (discrepancy) variance from the observed
    /// region and fold it into the predictive uncertainty. This guards
    /// against overconfidence when the shape model cannot represent what
    /// was actually observed (out-of-distribution fragments). On by
    /// default; see [`ShapePosterior::discrepancy_variance`].
    pub estimate_discrepancy: bool,
}

impl Default for PosteriorOptions {
    fn default() -> Self {
        Self {
            completeness: None,
            visibility_floor: 1e-6,
            prior_temperature: 1.0,
            outlier_weight: None,
            estimate_discrepancy: true,
        }
    }
}

/// Gaussian posterior over shape coefficients given a partial observation,
/// with methods for completion and per-point predictive uncertainty.
#[derive(Clone, Debug)]
pub struct ShapePosterior {
    mean: DMatrix<f64>,                   // (M, D) model mean shape
    modes: DMatrix<f64>,                  // (M·D, k) modes, point-major rows
    coefficient_mean: Vec<f64>,           // (k,)
    coefficient_covariance: DMatrix<f64>, // (k, k) Σ_b
    rotation: DMatrix<f64>,               // (D, D) row-vector convention
    scale: f64,
    translation: Vec<f64>,
    sigma_eff2: f64,           // model-frame assumed noise variance
    discrepancy_variance: f64, // model-frame model-inadequacy variance (≥ 0)
    d: usize,
}

impl ShapePosterior {
    /// Estimated shape coefficients (posterior mean).
    pub fn coefficient_mean(&self) -> &[f64] {
        &self.coefficient_mean
    }

    /// Posterior covariance of the shape coefficients, `Σ_b` (`k×k`).
    pub fn coefficient_covariance(&self) -> &DMatrix<f64> {
        &self.coefficient_covariance
    }

    /// Assumed observation-noise variance in the model frame (`σ_eff²`).
    pub fn noise_variance(&self) -> f64 {
        self.sigma_eff2
    }

    /// Estimated model-inadequacy (discrepancy) variance in the model
    /// frame: the excess observed-residual variance the shape model could
    /// not explain. `0` when discrepancy estimation was disabled or the
    /// model fits the observed region within noise. Larger values inflate
    /// the predictive uncertainty and flag out-of-distribution fragments.
    pub fn discrepancy_variance(&self) -> f64 {
        self.discrepancy_variance
    }

    /// Total per-coordinate model-frame variance folded into predictions:
    /// noise plus discrepancy.
    fn residual_variance(&self) -> f64 {
        self.sigma_eff2 + self.discrepancy_variance
    }

    /// Completed shape in the model frame: `μ + U·b_mean` (`M×D`).
    pub fn predict_model_frame(&self) -> DMatrix<f64> {
        let (m, d) = (self.mean.nrows(), self.d);
        DMatrix::from_fn(m, d, |i, j| {
            self.mean[(i, j)]
                + (0..self.coefficient_mean.len())
                    .map(|k| self.modes[(i * d + j, k)] * self.coefficient_mean[k])
                    .sum::<f64>()
        })
    }

    /// Completed shape in the target frame: the model-frame completion put
    /// through the fitted similarity transform (`M×D`). Observed regions
    /// reproduce the data; missing regions are filled by the model's
    /// learned covariation.
    pub fn predict(&self) -> DMatrix<f64> {
        let model = self.predict_model_frame();
        apply_similarity(&model, &self.rotation, self.scale, &self.translation)
    }

    /// Full `D×D` predictive covariance at model point `i`, in the target
    /// frame: `scale²·Rᵀ (Uᵢ Σ_b Uᵢᵀ + (σ_eff² + δ²) I) R`, where `δ²` is
    /// the discrepancy variance.
    pub fn predictive_covariance(&self, i: usize) -> DMatrix<f64> {
        let d = self.d;
        let k = self.coefficient_mean.len();
        // Uᵢ is the D×k block of modes for point i.
        let ui = DMatrix::from_fn(d, k, |row, col| self.modes[(i * d + row, col)]);
        let mut c_model = &ui * &self.coefficient_covariance * ui.transpose();
        let residual = self.residual_variance();
        for a in 0..d {
            c_model[(a, a)] += residual;
        }
        // Row-vector pushforward to the target frame: scale² Rᵀ C_model R.
        let scaled = self.scale * self.scale;
        scaled * (self.rotation.transpose() * c_model * &self.rotation)
    }

    /// Per-point total predictive variance (trace of the target-frame
    /// covariance), length `M`. Cheap confidence map: small where the
    /// fragment constrains the shape, large where the model is guessing.
    pub fn predictive_variance(&self) -> Vec<f64> {
        let (m, d) = (self.mean.nrows(), self.d);
        let k = self.coefficient_mean.len();
        let scaled = self.scale * self.scale;
        let residual = self.residual_variance();
        // trace is rotation-invariant, so total variance =
        // scale²·(trace(Uᵢ Σ_b Uᵢᵀ) + D·(σ_eff² + δ²)).
        (0..m)
            .map(|i| {
                let mut coefficient_part = 0.0;
                for row in 0..d {
                    // uᵀ Σ_b u for this row of Uᵢ.
                    for a in 0..k {
                        let ua = self.modes[(i * d + row, a)];
                        for b in 0..k {
                            coefficient_part += ua
                                * self.coefficient_covariance[(a, b)]
                                * self.modes[(i * d + row, b)];
                        }
                    }
                }
                scaled * (coefficient_part + d as f64 * residual)
            })
            .collect()
    }
}

/// Deterministic `splitmix64` state, seeded per sampling call so results
/// are fully reproducible and independent of thread scheduling.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `(0, 1]` (never returns exactly 0, so `ln` is safe).
    fn next_open_unit(&mut self) -> f64 {
        let bits = self.next_u64() >> 11; // 53-bit mantissa
        ((bits + 1) as f64) * (1.0 / ((1u64 << 53) as f64 + 1.0))
    }

    /// One standard normal via Box–Muller (deterministic pairing).
    fn next_standard_normal(&mut self) -> f64 {
        let u1 = self.next_open_unit();
        let u2 = self.next_open_unit();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

impl ShapePosterior {
    /// Draw `count` coefficient vectors from the Gaussian posterior
    /// `N(b_mean, Σ_b)`, deterministic in `seed`. Each is `b_mean + L·z`
    /// with `L` the Cholesky factor of `Σ_b` and `z` standard normal.
    pub fn sample_coefficients(&self, count: usize, seed: u64) -> Result<Vec<Vec<f64>>> {
        let k = self.coefficient_mean.len();
        let l = self
            .coefficient_covariance
            .clone()
            .cholesky()
            .ok_or(Error::SingularSystem)?
            .l();
        let mut rng = SplitMix64(seed ^ 0x5DEE_CE66_D3A1_1F3B);
        let mut samples = Vec::with_capacity(count);
        for _ in 0..count {
            let z: Vec<f64> = (0..k).map(|_| rng.next_standard_normal()).collect();
            let sample: Vec<f64> = (0..k)
                .map(|a| self.coefficient_mean[a] + (0..=a).map(|b| l[(a, b)] * z[b]).sum::<f64>())
                .collect();
            samples.push(sample);
        }
        Ok(samples)
    }

    /// Draw `count` completed shapes (target frame, `M×D` each) from the
    /// posterior — an ensemble of plausible completions. Deterministic in
    /// `seed`.
    pub fn sample_shapes(&self, count: usize, seed: u64) -> Result<Vec<DMatrix<f64>>> {
        let coefficients = self.sample_coefficients(count, seed)?;
        Ok(coefficients
            .into_iter()
            .map(|b| {
                let model = self.shape_from_coefficients(&b);
                apply_similarity(&model, &self.rotation, self.scale, &self.translation)
            })
            .collect())
    }

    /// `μ + U·b` for an arbitrary coefficient vector (model frame).
    fn shape_from_coefficients(&self, b: &[f64]) -> DMatrix<f64> {
        let (m, d) = (self.mean.nrows(), self.d);
        DMatrix::from_fn(m, d, |i, j| {
            self.mean[(i, j)]
                + (0..b.len())
                    .map(|k| self.modes[(i * d + j, k)] * b[k])
                    .sum::<f64>()
        })
    }
}

/// Draw completed shapes from a *mixture* of posteriors — e.g. one per
/// competing pose hypothesis from the pose initializer — turning pose
/// ambiguity into honest multi-modal predictive uncertainty. `components`
/// pairs each posterior with a non-negative weight (need not sum to one).
/// Deterministic in `seed`.
pub fn mixture_sample_shapes(
    components: &[(f64, &ShapePosterior)],
    count: usize,
    seed: u64,
) -> Result<Vec<DMatrix<f64>>> {
    if components.is_empty() {
        return Err(Error::InvalidShape("components"));
    }
    let total: f64 = components.iter().map(|(w, _)| w).sum();
    if !(total.is_finite()) || total <= 0.0 || components.iter().any(|(w, _)| *w < 0.0) {
        return Err(Error::PositiveParameter("mixture weights"));
    }
    // Deterministic per-draw component choice, then one shape from it.
    let mut rng = SplitMix64(seed ^ 0x2545_F491_4F6C_DD1D);
    let mut shapes = Vec::with_capacity(count);
    for index in 0..count {
        let pick = rng.next_open_unit() * total;
        let mut acc = 0.0;
        let mut chosen = components.len() - 1;
        for (c, (w, _)) in components.iter().enumerate() {
            acc += w;
            if pick <= acc {
                chosen = c;
                break;
            }
        }
        // Sub-seed per draw so the stream is independent of component choice.
        let shape = components[chosen]
            .1
            .sample_shapes(
                1,
                seed ^ ((index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)),
            )?
            .pop()
            .unwrap();
        shapes.push(shape);
    }
    Ok(shapes)
}

/// Build a shape posterior directly from explicit model-frame inputs.
///
/// `mean` is `(M, D)`, `modes` is `(M·D, k)` (point-major rows), and
/// `eigenvalues` are the `k` mode variances. `residual` is `(M, D)`, the
/// observed model-frame position minus the mean (only meaningful where
/// `weight > 0`); `weight` is the per-point soft observation weight
/// (`0` = unobserved). The similarity `(rotation, scale, translation)`
/// maps the model frame to the target frame; `sigma_eff2` is the
/// model-frame noise variance. When `estimate_discrepancy` is set, a
/// model-inadequacy variance is estimated from the observed residual and
/// folded into the predictive uncertainty. This is the testable core that
/// [`AtlasResult::posterior`] wraps.
#[allow(clippy::too_many_arguments)]
pub fn complete_shape(
    mean: &DMatrix<f64>,
    modes: &DMatrix<f64>,
    eigenvalues: &[f64],
    residual: &DMatrix<f64>,
    weight: &[f64],
    sigma_eff2: f64,
    rotation: &DMatrix<f64>,
    scale: f64,
    translation: &[f64],
    prior_temperature: f64,
    estimate_discrepancy: bool,
) -> Result<ShapePosterior> {
    let (m, d) = (mean.nrows(), mean.ncols());
    let k = modes.ncols();
    if modes.nrows() != m * d {
        return Err(Error::InvalidShape("modes"));
    }
    if eigenvalues.len() != k {
        return Err(Error::InvalidShape("eigenvalues"));
    }
    if residual.nrows() != m || residual.ncols() != d {
        return Err(Error::InvalidShape("residual"));
    }
    if weight.len() != m {
        return Err(Error::InvalidShape("weight"));
    }
    if rotation.shape() != (d, d) || translation.len() != d {
        return Err(Error::InvalidShape("pose"));
    }
    if !sigma_eff2.is_finite() || sigma_eff2 <= 0.0 {
        return Err(Error::PositiveParameter("sigma_eff2"));
    }
    if !prior_temperature.is_finite() || prior_temperature <= 0.0 {
        return Err(Error::PositiveParameter("prior_temperature"));
    }
    if eigenvalues.iter().any(|&v| !v.is_finite() || v <= 0.0) {
        return Err(Error::PositiveParameter("eigenvalues"));
    }
    if weight.iter().any(|&w| !w.is_finite() || w < 0.0) {
        return Err(Error::PositiveParameter("weight"));
    }

    // Data precision A_data = Uᵀ W U and rhs Uᵀ W r via a √W-scaled copy of
    // the modes (W repeats each point weight across its D rows).
    let mut scaled_modes = modes.clone();
    let mut weighted_residual = DMatrix::zeros(m * d, 1);
    for i in 0..m {
        let w = weight[i];
        let sqrt_w = w.sqrt();
        for j in 0..d {
            let flat = i * d + j;
            for a in 0..k {
                scaled_modes[(flat, a)] *= sqrt_w;
            }
            weighted_residual[(flat, 0)] = w * residual[(i, j)];
        }
    }
    let mut precision = scaled_modes.tr_mul(&scaled_modes); // Uᵀ W U (k×k)
    precision /= sigma_eff2;
    for a in 0..k {
        precision[(a, a)] += prior_temperature / eigenvalues[a];
    }
    let rhs = modes.tr_mul(&weighted_residual) / sigma_eff2; // (k×1)

    // Σ_b = A⁻¹, b_mean = Σ_b rhs, via a Cholesky of the SPD precision.
    let chol = precision.clone().cholesky().ok_or(Error::SingularSystem)?;
    let covariance = chol.inverse();
    let coefficient_mean: Vec<f64> = (&covariance * rhs).column(0).iter().copied().collect();

    // Discrepancy: the observed residual the fitted model cannot explain.
    // If the weighted mean squared residual over observed coordinates
    // exceeds the assumed noise σ_eff², the excess is a model-inadequacy
    // variance δ² (Kennedy–O'Hagan). It only inflates predictions; the
    // coefficient posterior above is unchanged.
    let discrepancy_variance = if estimate_discrepancy {
        let mut weighted_sq = 0.0;
        let mut total_weight = 0.0;
        for i in 0..m {
            let w = weight[i];
            if w <= 0.0 {
                continue;
            }
            total_weight += w;
            for j in 0..d {
                let predicted: f64 = (0..k)
                    .map(|a| modes[(i * d + j, a)] * coefficient_mean[a])
                    .sum();
                let e = residual[(i, j)] - predicted;
                weighted_sq += w * e * e;
            }
        }
        if total_weight > 0.0 {
            let mean_sq_per_coord = weighted_sq / (total_weight * d as f64);
            (mean_sq_per_coord - sigma_eff2).max(0.0)
        } else {
            0.0
        }
    } else {
        0.0
    };

    Ok(ShapePosterior {
        mean: mean.clone(),
        modes: modes.clone(),
        coefficient_mean,
        coefficient_covariance: covariance,
        rotation: rotation.clone(),
        scale,
        translation: translation.to_vec(),
        sigma_eff2,
        discrepancy_variance,
        d,
    })
}

impl AtlasResult {
    /// Build a [`ShapePosterior`] from this fit for shape completion and
    /// per-point uncertainty. `target` is the observed (possibly partial)
    /// cloud the atlas was registered to; `mean`/`modes`/`eigenvalues` are
    /// the model at the resolution you want to complete at (may be denser
    /// than what was registered, as with [`Self::reconstruct`]).
    ///
    /// Visibility is inferred from the fitted correspondence, optionally
    /// anchored by [`PosteriorOptions::completeness`]. All heavy work is
    /// post-hoc; `register()` is never modified.
    pub fn posterior(
        &self,
        target: &DMatrix<f64>,
        mean: &DMatrix<f64>,
        modes: &DMatrix<f64>,
        eigenvalues: &[f64],
        options: &PosteriorOptions,
    ) -> Result<ShapePosterior> {
        validate_clouds(target, mean)?;
        let (m, d) = (mean.nrows(), mean.ncols());
        if modes.nrows() != m * d || modes.ncols() != eigenvalues.len() {
            return Err(Error::InvalidShape("modes"));
        }
        if self.rotation.shape() != (d, d) {
            return Err(Error::InvalidShape("rotation"));
        }
        let outlier_weight = options.outlier_weight.unwrap_or(self.outlier_weight);
        if !(0.0..1.0).contains(&outlier_weight) {
            return Err(Error::InvalidOutlierWeight);
        }
        if !self.sigma2.is_finite() || self.sigma2 <= 0.0 {
            return Err(Error::PositiveParameter("sigma2"));
        }
        if !self.outlier_density.is_finite() || self.outlier_density <= 0.0 {
            return Err(Error::PositiveParameter("outlier_density"));
        }

        // Deformed + posed model in the target frame (the converged fit at
        // this resolution).
        let ty = self.reconstruct(mean, modes)?;

        // Condition on the fitted observation model. Reinitializing sigma2
        // from all pairs reheats fragment fits; uniform weights also discard
        // the occupancy model learned by adaptive registration. Transfer the
        // relative occupancies only when the model sampling has changed.
        let pi = match (&self.mixing_weights, &self.mixing_reference) {
            (Some(weights), Some(reference)) => {
                validate_mixing_weights(weights)?;
                if weights.len() != reference.nrows() {
                    return Err(Error::InvalidShape("mixing_weights"));
                }
                Some(MixingWeightMap::new(reference, mean)?.apply(weights))
            }
            (Some(_), None) => return Err(Error::InvalidShape("mixing_reference")),
            _ => None,
        };
        let mixing = mixing_factors(pi.as_deref());
        let config = EmConfig {
            outlier_weight: effective_outlier_weight(
                outlier_weight,
                self.outlier_density,
                target.nrows(),
                1.0,
                d,
            ),
            k: None,
            ..EmConfig::default()
        };
        let stats = posterior_stats_weighted(
            target,
            &ty,
            self.sigma2,
            &config,
            false,
            None,
            mixing.as_deref(),
        );
        let sigma_eff2 = (self.sigma2 / (self.scale * self.scale)).max(f64::MIN_POSITIVE);

        // Per-model-point observed position (target frame) = px/p1, inverse
        // posed to the model frame; residual against the mean; weight = p1.
        let mut residual = DMatrix::zeros(m, d);
        let mut weight = vec![0.0; m];
        for i in 0..m {
            let mass = stats.p1[i];
            weight[i] = mass;
            if mass <= f64::MIN_POSITIVE {
                continue;
            }
            // target-frame expected position, then z = (x − t)·Rᵀ / scale.
            let mut posed = vec![0.0f64; d];
            for (j, slot) in posed.iter_mut().enumerate() {
                *slot = stats.px[(i, j)] / mass - self.translation[j];
            }
            for j in 0..d {
                let model_coord: f64 = (0..d)
                    .map(|q| posed[q] * self.rotation[(j, q)])
                    .sum::<f64>()
                    / self.scale;
                residual[(i, j)] = model_coord - mean[(i, j)];
            }
        }
        apply_visibility(&mut weight, options);

        complete_shape(
            mean,
            modes,
            eigenvalues,
            &residual,
            &weight,
            sigma_eff2,
            &self.rotation,
            self.scale,
            &self.translation,
            options.prior_temperature,
            options.estimate_discrepancy,
        )
    }
}

/// Apply the completeness prior / visibility floor to per-point weights.
fn apply_visibility(weight: &mut [f64], options: &PosteriorOptions) {
    match options.completeness {
        Some(fraction) => {
            let fraction = fraction.clamp(0.0, 1.0);
            let keep = ((fraction * weight.len() as f64).round() as usize).clamp(1, weight.len());
            // Threshold at the keep-th largest weight (deterministic order).
            let mut sorted: Vec<f64> = weight.to_vec();
            sorted.sort_by(|a, b| b.total_cmp(a));
            let threshold = sorted[keep - 1];
            for w in weight.iter_mut() {
                if *w < threshold {
                    *w = 0.0;
                }
            }
        }
        None => {
            for w in weight.iter_mut() {
                if *w < options.visibility_floor {
                    *w = 0.0;
                }
            }
        }
    }
}

/// `out = scale · y · R + t` (row-vector convention, matching the atlas).
fn apply_similarity(y: &DMatrix<f64>, r: &DMatrix<f64>, scale: f64, t: &[f64]) -> DMatrix<f64> {
    let d = y.ncols();
    DMatrix::from_fn(y.nrows(), d, |i, j| {
        scale * (0..d).map(|q| y[(i, q)] * r[(q, j)]).sum::<f64>() + t[j]
    })
}
