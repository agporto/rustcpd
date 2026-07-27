use nalgebra::DMatrix;
use rayon::prelude::*;

use crate::fastexp::exp_non_positive;
use crate::{AtlasConfig, AtlasRegistration, EmConfig, Error, Result};

/// Coarse-stage scoring for pose screening.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PoseScoreMode {
    /// Recompute the dense CPD objective on the final coarse iterate.
    Final,
    /// Use the running EM negative log-likelihood plus the shape prior
    /// (cheaper; matches the reference "trajectory" mode).
    Trajectory,
}

/// Configuration for pose-marginalized initialization: a coarse-to-fine
/// sweep over a rotation lattice that scores atlas registrations started
/// from each hypothesis.
///
/// The coarse stage is an optional three-step funnel: when
/// `coarse_screen_iterations < coarse_iterations` and
/// `coarse_survivor_count < rotation_count`, every rotation is *screened*
/// with a short EM run, the best `coarse_survivor_count` survivors (identity
/// always kept) are *completed* to the full `coarse_iterations` budget, and
/// only those advance. Otherwise every rotation runs the full coarse budget
/// directly (the funnel degenerates to a single pass).
#[derive(Clone, Debug)]
pub struct PoseMarginalizedConfig {
    /// Number of rotation hypotheses (the first is always the identity).
    pub rotation_count: usize,
    /// Source subsample size for the coarse pass.
    pub coarse_source_count: usize,
    /// Target subsample size for the coarse pass.
    pub coarse_target_count: usize,
    /// Number of shape modes used in the coarse pass.
    pub coarse_rank: usize,
    /// EM iterations per coarse hypothesis (the full coarse budget).
    pub coarse_iterations: usize,
    /// EM iterations for the initial cheap screening pass over every
    /// rotation. `>= coarse_iterations` disables screening.
    pub coarse_screen_iterations: usize,
    /// Number of screened hypotheses carried into the full coarse pass.
    /// `>= rotation_count` disables pruning.
    pub coarse_survivor_count: usize,
    /// Coarse-stage scoring mode used to rank screened / coarse hypotheses.
    pub coarse_score_mode: PoseScoreMode,
    /// Number of best coarse hypotheses carried into refinement.
    pub refine_count: usize,
    /// Source subsample size for refinement (`None` uses every point).
    pub refine_source_count: Option<usize>,
    /// Target subsample size for refinement.
    pub refine_target_count: usize,
    /// EM iterations per refined hypothesis.
    pub refine_iterations: usize,
    /// Strength of the Mahalanobis prior on shape coefficients.
    pub lambda_regularization: f64,
    /// Uniform-outlier mixture weight in `[0, 1)`.
    pub outlier_weight: f64,
    /// Prior probability of the identity rotation, in `(0, 1)`.
    pub identity_prior_probability: f64,
    /// Anchored keypoint correspondences steering the global search: each
    /// entry pairs a source-model vertex index with its known target
    /// coordinate. Every hypothesis pays a penalty proportional to how far it
    /// leaves these vertices from their targets, so screening, survivor
    /// selection, and refinement all prefer the keypoint-consistent basin —
    /// the discrete decision local refinement alone cannot revisit. Empty
    /// (default) disables the term.
    pub landmarks: Vec<(usize, Vec<f64>)>,
    /// Weight of the keypoint penalty, in units of effective target points
    /// per landmark (the penalty is `0.5 · landmark_weight · ‖fitted −
    /// target‖² / sigma2`, commensurate with the per-point CPD data cost).
    /// `0` disables the term even when `landmarks` is non-empty.
    pub landmark_weight: f64,
    /// Include a residual isotropic scale in each pose hypothesis.
    ///
    /// Set this to `false` when the source and modes have already been
    /// pre-scaled from an external physical-size estimate. Rotation and
    /// translation remain optimized.
    pub with_scale: bool,
    /// Offset applied to the low-discrepancy rotation sequence.
    pub seed: u64,
    /// Evaluate hypotheses on the crate's Rayon pool.
    pub parallel: bool,
    /// Run the coarse and refinement E-steps in single precision; see
    /// [`crate::EmConfig::single_precision`].
    pub single_precision: bool,
}

impl Default for PoseMarginalizedConfig {
    fn default() -> Self {
        Self {
            rotation_count: 193,
            coarse_source_count: 400,
            coarse_target_count: 400,
            coarse_rank: 12,
            coarse_iterations: 8,
            coarse_screen_iterations: 8,
            coarse_survivor_count: 193,
            coarse_score_mode: PoseScoreMode::Final,
            refine_count: 12,
            refine_source_count: None,
            refine_target_count: 1600,
            refine_iterations: 30,
            lambda_regularization: 0.1,
            outlier_weight: 0.05,
            identity_prior_probability: 0.2,
            landmarks: Vec::new(),
            landmark_weight: 0.0,
            with_scale: true,
            seed: 0,
            parallel: true,
            single_precision: false,
        }
    }
}

/// Best pose hypothesis found by [`PoseMarginalizedConfig::initialize`],
/// with diagnostics describing how decisive the selection was.
#[derive(Clone, Debug)]
pub struct PoseMarginalizedInitialization {
    /// Shape coefficients of the winning hypothesis.
    pub coefficients: Vec<f64>,
    /// Rotation of the winning hypothesis (row-vector convention: `y·R`).
    pub rotation: DMatrix<f64>,
    /// Scale of the winning hypothesis.
    pub scale: f64,
    /// Translation of the winning hypothesis.
    pub translation: Vec<f64>,
    /// Negative-log-posterior score of the winner (lower is better).
    pub score: f64,
    /// Score gap to the runner-up; larger means more decisive.
    pub score_margin: f64,
    /// Shannon entropy of the refined-hypothesis posterior.
    pub posterior_entropy: f64,
    /// `exp(posterior_entropy)` — the effective number of competitive
    /// hypotheses.
    pub effective_hypotheses: f64,
    /// Total rotation hypotheses evaluated in the coarse pass.
    pub hypotheses_evaluated: usize,
    /// Hypotheses carried through refinement.
    pub hypotheses_refined: usize,
}

#[derive(Clone)]
struct Candidate {
    /// Lattice rotation index this hypothesis started from (0 = identity).
    index: usize,
    score: f64,
    coefficients: Vec<f64>,
    rotation: DMatrix<f64>,
    scale: f64,
    translation: Vec<f64>,
    prior_cost: f64,
}

impl PoseMarginalizedConfig {
    /// Search the rotation lattice and return the best-scoring similarity
    /// transform and shape coefficients for starting a full registration.
    pub fn initialize(
        &self,
        source: &DMatrix<f64>,
        target: &DMatrix<f64>,
        modes: &DMatrix<f64>,
        eigenvalues: &[f64],
    ) -> Result<PoseMarginalizedInitialization> {
        self.validate(source, target, modes, eigenvalues)?;
        let source_indices = subset_indices(source, Some(self.coarse_source_count));
        let target_indices = subset_indices(target, Some(self.coarse_target_count));
        let coarse_source = select_rows(source, &source_indices);
        let coarse_target = select_rows(target, &target_indices);
        let rank = self.coarse_rank.min(eigenvalues.len());
        let coarse_modes = select_modes(modes, source.ncols(), &source_indices, rank);
        let rotations = rotation_lattice(self.rotation_count, self.seed);
        let nonidentity_prior =
            (1.0 - self.identity_prior_probability) / (rotations.len() - 1).max(1) as f64;
        // Evaluate one rotation hypothesis: run `max_iters` of coarse atlas EM
        // from its initial similarity and score it under the configured mode.
        let evaluate =
            |index: usize, rotation: &DMatrix<f64>, max_iters: usize| -> Result<Candidate> {
                let prior = if index == 0 {
                    self.identity_prior_probability
                } else {
                    nonidentity_prior
                };
                let (scale, translation) =
                    initial_similarity(&coarse_source, &coarse_target, rotation, self.with_scale);
                let config = AtlasConfig {
                    em: EmConfig {
                        max_iterations: max_iters,
                        tolerance: 0.0,
                        outlier_weight: self.outlier_weight,
                        parallel: false,
                        single_precision: self.single_precision,
                        ..Default::default()
                    },
                    eigenvalues: eigenvalues[..rank].to_vec(),
                    lambda_regularization: self.lambda_regularization,
                    normalize: true,
                    optimize_similarity: true,
                    with_scale: self.with_scale,
                    initial_rotation: Some(rotation.clone()),
                    initial_scale: scale,
                    initial_translation: Some(translation),
                    ..Default::default()
                };
                let result =
                    AtlasRegistration::new(&coarse_target, &coarse_source, &coarse_modes, config)?
                        .register()?;
                let data_cost = match self.coarse_score_mode {
                    // score_candidate already folds in the shape-prior term.
                    PoseScoreMode::Final => score_candidate(
                        &coarse_target,
                        &result.points,
                        result.sigma2,
                        self.outlier_weight,
                        &result.coefficients,
                        &eigenvalues[..rank],
                        self.lambda_regularization,
                    ),
                    // Running EM data objective plus the shape prior, matching the
                    // reference "trajectory" score. Comparable across hypotheses
                    // because every candidate shares the same normalized target.
                    PoseScoreMode::Trajectory => {
                        let shape_cost = 0.5
                            * self.lambda_regularization
                            * result
                                .coefficients
                                .iter()
                                .zip(&eigenvalues[..rank])
                                .map(|(b, e)| b * b / e.max(f64::EPSILON))
                                .sum::<f64>();
                        result.negative_log_likelihood + shape_cost
                    }
                };
                let kp_penalty = self.keypoint_penalty(
                    source,
                    modes,
                    &result.coefficients,
                    &result.rotation,
                    result.scale,
                    &result.translation,
                    result.sigma2,
                );
                Ok(Candidate {
                    index,
                    score: data_cost - prior.ln() + kp_penalty,
                    coefficients: result.coefficients,
                    rotation: result.rotation,
                    scale: result.scale,
                    translation: result.translation,
                    prior_cost: -prior.ln(),
                })
            };
        // Map `evaluate` over a set of (index, rotation) items at a given budget.
        let map_eval = |items: &[(usize, &DMatrix<f64>)], iters: usize| -> Result<Vec<Candidate>> {
            if self.parallel {
                items
                    .par_iter()
                    .map(|&(i, r)| evaluate(i, r, iters))
                    .collect()
            } else {
                items.iter().map(|&(i, r)| evaluate(i, r, iters)).collect()
            }
        };

        let indexed: Vec<(usize, &DMatrix<f64>)> = rotations.iter().enumerate().collect();
        // Screening runs a prefix of the coarse budget; cap it so an over-large
        // (e.g. defaulted) value simply disables screening instead of erroring.
        let screen_iters = self.coarse_screen_iterations.min(self.coarse_iterations);
        let use_funnel =
            screen_iters < self.coarse_iterations && self.coarse_survivor_count < rotations.len();
        let mut coarse: Vec<Candidate> = if use_funnel {
            // Screen every rotation cheaply, keep the best survivors (identity
            // always retained), then complete only those to the full budget.
            let screened = map_eval(&indexed, screen_iters)?;
            let survivor_indices = select_survivor_indices(&screened, self.coarse_survivor_count);
            let survivors: Vec<(usize, &DMatrix<f64>)> = survivor_indices
                .iter()
                .map(|&i| (i, &rotations[i]))
                .collect();
            map_eval(&survivors, self.coarse_iterations)?
        } else {
            map_eval(&indexed, self.coarse_iterations)?
        };
        // Take the best `refine_count`, always keeping the identity hypothesis.
        let identity = coarse.iter().find(|c| c.index == 0).cloned();
        coarse.sort_by(|a, b| a.score.total_cmp(&b.score));
        coarse.truncate(self.refine_count.min(coarse.len()));
        if let Some(identity) = identity {
            if coarse.len() > 1 && !coarse.iter().any(|c| c.index == 0) {
                let last = coarse.len() - 1;
                coarse[last] = identity;
            }
        }
        let target_indices = subset_indices(target, Some(self.refine_target_count));
        let source_indices = subset_indices(source, self.refine_source_count);
        let refined_target = select_rows(target, &target_indices);
        let refined_source = select_rows(source, &source_indices);
        let refined_modes = select_modes(modes, source.ncols(), &source_indices, eigenvalues.len());
        let refine = |initial: &Candidate| -> Result<Candidate> {
            let mut coefficients = vec![0.0; eigenvalues.len()];
            coefficients[..initial.coefficients.len()].copy_from_slice(&initial.coefficients);
            let config = AtlasConfig {
                em: EmConfig {
                    max_iterations: self.refine_iterations,
                    tolerance: 0.0,
                    outlier_weight: self.outlier_weight,
                    parallel: false,
                    single_precision: self.single_precision,
                    ..Default::default()
                },
                eigenvalues: eigenvalues.to_vec(),
                lambda_regularization: self.lambda_regularization,
                normalize: true,
                optimize_similarity: true,
                with_scale: self.with_scale,
                kdtree_radius_scale: None,
                initial_coefficients: Some(coefficients),
                initial_rotation: Some(initial.rotation.clone()),
                initial_scale: initial.scale,
                initial_translation: Some(initial.translation.clone()),
                ..Default::default()
            };
            let result =
                AtlasRegistration::new(&refined_target, &refined_source, &refined_modes, config)?
                    .register()?;
            let full_deformed = apply_model(
                source,
                modes,
                &result.coefficients,
                &result.rotation,
                result.scale,
                &result.translation,
            );
            let score = score_candidate(
                target,
                &full_deformed,
                result.sigma2,
                self.outlier_weight,
                &result.coefficients,
                eigenvalues,
                self.lambda_regularization,
            ) + initial.prior_cost
                + self.keypoint_penalty(
                    source,
                    modes,
                    &result.coefficients,
                    &result.rotation,
                    result.scale,
                    &result.translation,
                    result.sigma2,
                );
            Ok(Candidate {
                index: initial.index,
                score,
                coefficients: result.coefficients,
                rotation: result.rotation,
                scale: result.scale,
                translation: result.translation,
                prior_cost: initial.prior_cost,
            })
        };
        let mut refined: Vec<Candidate> = if self.parallel {
            coarse.par_iter().map(refine).collect::<Result<Vec<_>>>()?
        } else {
            coarse.iter().map(refine).collect::<Result<Vec<_>>>()?
        };
        refined.sort_by(|a, b| a.score.total_cmp(&b.score));
        let min_score = refined[0].score;
        let normalizer = refined
            .iter()
            .map(|candidate| (min_score - candidate.score).exp())
            .sum::<f64>();
        let entropy = refined
            .iter()
            .map(|candidate| {
                let p = (min_score - candidate.score).exp() / normalizer;
                if p > 0.0 { -p * p.ln() } else { 0.0 }
            })
            .sum::<f64>();
        let best = refined.remove(0);
        Ok(PoseMarginalizedInitialization {
            coefficients: best.coefficients,
            rotation: best.rotation,
            scale: best.scale,
            translation: best.translation,
            score: best.score,
            score_margin: refined
                .first()
                .map_or(f64::INFINITY, |second| second.score - best.score),
            posterior_entropy: entropy,
            effective_hypotheses: entropy.exp(),
            hypotheses_evaluated: rotations.len(),
            hypotheses_refined: refined.len() + 1,
        })
    }

    /// Keypoint-consistency penalty for one hypothesis, in the same "nats"
    /// as the per-point CPD data cost (`0.5 · ‖·‖² / sigma2` per unit weight).
    /// Zero when the term is disabled.
    #[allow(clippy::too_many_arguments)]
    fn keypoint_penalty(
        &self,
        source: &DMatrix<f64>,
        modes: &DMatrix<f64>,
        coefficients: &[f64],
        rotation: &DMatrix<f64>,
        scale: f64,
        translation: &[f64],
        sigma2: f64,
    ) -> f64 {
        if self.landmarks.is_empty() || self.landmark_weight <= 0.0 {
            return 0.0;
        }
        let residual = landmark_residual_sq(
            source,
            modes,
            coefficients,
            rotation,
            scale,
            translation,
            &self.landmarks,
        );
        0.5 * self.landmark_weight * residual / sigma2.max(f64::MIN_POSITIVE)
    }

    fn validate(
        &self,
        source: &DMatrix<f64>,
        target: &DMatrix<f64>,
        modes: &DMatrix<f64>,
        eigenvalues: &[f64],
    ) -> Result<()> {
        if source.ncols() != 3 || target.ncols() != 3 || source.nrows() == 0 || target.nrows() == 0
        {
            return Err(Error::InvalidShape(
                "source and target must have non-empty shape (N, 3)",
            ));
        }
        if modes.shape() != (source.len(), eigenvalues.len()) || eigenvalues.is_empty() {
            return Err(Error::InvalidShape("modes and eigenvalues"));
        }
        if self.rotation_count == 0
            || self.refine_count == 0
            || self.coarse_rank == 0
            || self.coarse_iterations == 0
            || self.refine_iterations == 0
            || self.coarse_screen_iterations == 0
        {
            return Err(Error::PositiveParameter("pose counts"));
        }
        // Survivors must at least cover the requested finalists (identity is
        // always retained). Screening is capped at the coarse budget in
        // `initialize`, so `coarse_screen_iterations > coarse_iterations` is
        // tolerated (it simply disables screening) rather than rejected.
        if self.coarse_survivor_count < self.refine_count.min(self.rotation_count) {
            return Err(Error::PositiveParameter("pose survivor count"));
        }
        if !(0.0..1.0).contains(&self.identity_prior_probability)
            || self.identity_prior_probability == 0.0
        {
            return Err(Error::InvalidOutlierWeight);
        }
        if !self.landmark_weight.is_finite() || self.landmark_weight < 0.0 {
            return Err(Error::PositiveParameter("landmark_weight"));
        }
        for (index, point) in &self.landmarks {
            if *index >= source.nrows()
                || point.len() != source.ncols()
                || !point.iter().all(|value| value.is_finite())
            {
                return Err(Error::InvalidShape("landmarks"));
            }
        }
        Ok(())
    }
}

fn subset_indices(points: &DMatrix<f64>, count: Option<usize>) -> Vec<usize> {
    let count = count.unwrap_or(points.nrows()).clamp(1, points.nrows());
    if count == points.nrows() {
        return (0..count).collect();
    }
    let centroid: Vec<_> = (0..points.ncols())
        .map(|j| (0..points.nrows()).map(|i| points[(i, j)]).sum::<f64>() / points.nrows() as f64)
        .collect();
    let first = (0..points.nrows())
        .max_by(|&a, &b| {
            squared_from(points, a, &centroid).total_cmp(&squared_from(points, b, &centroid))
        })
        .unwrap();
    let mut selected = vec![first];
    let mut nearest: Vec<_> = (0..points.nrows())
        .map(|i| squared_pair(points, i, first))
        .collect();
    while selected.len() < count {
        let next = (0..points.nrows())
            .max_by(|&a, &b| nearest[a].total_cmp(&nearest[b]))
            .unwrap();
        selected.push(next);
        for (i, value) in nearest.iter_mut().enumerate() {
            *value = value.min(squared_pair(points, i, next));
        }
    }
    selected
}
fn squared_from(p: &DMatrix<f64>, i: usize, q: &[f64]) -> f64 {
    (0..p.ncols()).map(|j| (p[(i, j)] - q[j]).powi(2)).sum()
}
fn squared_pair(p: &DMatrix<f64>, i: usize, j: usize) -> f64 {
    (0..p.ncols())
        .map(|q| (p[(i, q)] - p[(j, q)]).powi(2))
        .sum()
}
fn select_rows(p: &DMatrix<f64>, indices: &[usize]) -> DMatrix<f64> {
    DMatrix::from_fn(indices.len(), p.ncols(), |i, j| p[(indices[i], j)])
}
fn select_modes(m: &DMatrix<f64>, d: usize, indices: &[usize], rank: usize) -> DMatrix<f64> {
    DMatrix::from_fn(indices.len() * d, rank, |row, k| {
        m[(indices[row / d] * d + row % d, k)]
    })
}

fn rotation_lattice(count: usize, seed: u64) -> Vec<DMatrix<f64>> {
    let mut rotations = vec![DMatrix::identity(3, 3)];
    if count == 1 {
        return rotations;
    }
    let remaining = count - 1;
    let local = remaining / 2;
    let global = remaining - local;
    for i in 0..global {
        let index = i as f64 + 1.0 + seed as f64;
        let u1 = (index * 0.7548776662466927).fract();
        let u2 = (index * 0.5698402909980532).fract();
        let u3 = (index * 0.438579021).fract();
        let q = [
            (1. - u1).sqrt() * (2. * std::f64::consts::PI * u2).sin(),
            (1. - u1).sqrt() * (2. * std::f64::consts::PI * u2).cos(),
            u1.sqrt() * (2. * std::f64::consts::PI * u3).sin(),
            u1.sqrt() * (2. * std::f64::consts::PI * u3).cos(),
        ];
        rotations.push(quaternion_matrix(q));
    }
    let (base, extra) = (local / 3, local % 3);
    for (shell, angle) in [20.0_f64, 40.0, 60.0].into_iter().enumerate() {
        let n = base + usize::from(shell < extra);
        for i in 0..n {
            let z = 1. - 2. * (i as f64 + 0.5) / n as f64;
            let radius = (1. - z * z).max(0.).sqrt();
            let golden = std::f64::consts::PI * (3. - 5f64.sqrt());
            rotations.push(axis_angle(
                [
                    radius * (golden * i as f64).cos(),
                    radius * (golden * i as f64).sin(),
                    z,
                ],
                angle.to_radians(),
            ));
        }
    }
    rotations
}
fn quaternion_matrix([x, y, z, w]: [f64; 4]) -> DMatrix<f64> {
    DMatrix::from_row_slice(
        3,
        3,
        &[
            1. - 2. * (y * y + z * z),
            2. * (x * y - z * w),
            2. * (x * z + y * w),
            2. * (x * y + z * w),
            1. - 2. * (x * x + z * z),
            2. * (y * z - x * w),
            2. * (x * z - y * w),
            2. * (y * z + x * w),
            1. - 2. * (x * x + y * y),
        ],
    )
}
fn axis_angle([x, y, z]: [f64; 3], a: f64) -> DMatrix<f64> {
    let h = a / 2.;
    quaternion_matrix([x * h.sin(), y * h.sin(), z * h.sin(), h.cos()])
}
/// Lattice indices of the best `count` screened hypotheses, always including
/// the identity (index 0). Mirrors the reference finalist selection.
fn select_survivor_indices(screened: &[Candidate], count: usize) -> Vec<usize> {
    let keep = count.clamp(1, screened.len());
    let mut order: Vec<usize> = (0..screened.len()).collect();
    order.sort_by(|&a, &b| screened[a].score.total_cmp(&screened[b].score));
    let mut survivors: Vec<usize> = order[..keep].iter().map(|&p| screened[p].index).collect();
    if !survivors.contains(&0) {
        if let Some(last) = survivors.last_mut() {
            *last = 0;
        }
    }
    survivors
}
fn initial_similarity(
    source: &DMatrix<f64>,
    target: &DMatrix<f64>,
    rotation: &DMatrix<f64>,
    with_scale: bool,
) -> (f64, Vec<f64>) {
    let source_centroid: Vec<_> = (0..3)
        .map(|j| (0..source.nrows()).map(|i| source[(i, j)]).sum::<f64>() / source.nrows() as f64)
        .collect();
    let target_centroid: Vec<_> = (0..3)
        .map(|j| (0..target.nrows()).map(|i| target[(i, j)]).sum::<f64>() / target.nrows() as f64)
        .collect();
    let scale = if with_scale {
        let source_radius = ((0..source.nrows())
            .map(|i| {
                (0..3)
                    .map(|j| (source[(i, j)] - source_centroid[j]).powi(2))
                    .sum::<f64>()
            })
            .sum::<f64>()
            / source.nrows() as f64)
            .sqrt();
        let target_radius = ((0..target.nrows())
            .map(|i| {
                (0..3)
                    .map(|j| (target[(i, j)] - target_centroid[j]).powi(2))
                    .sum::<f64>()
            })
            .sum::<f64>()
            / target.nrows() as f64)
            .sqrt();
        target_radius / source_radius.max(f64::EPSILON)
    } else {
        1.0
    };
    let translation = (0..3)
        .map(|j| {
            target_centroid[j]
                - scale
                    * (0..3)
                        .map(|q| source_centroid[q] * rotation[(q, j)])
                        .sum::<f64>()
        })
        .collect();
    (scale, translation)
}
fn apply_model(
    s: &DMatrix<f64>,
    m: &DMatrix<f64>,
    b: &[f64],
    r: &DMatrix<f64>,
    scale: f64,
    t: &[f64],
) -> DMatrix<f64> {
    let deformed = DMatrix::from_fn(s.nrows(), 3, |i, j| {
        s[(i, j)] + (0..b.len()).map(|k| m[(i * 3 + j, k)] * b[k]).sum::<f64>()
    });
    DMatrix::from_fn(s.nrows(), 3, |i, j| {
        scale * (0..3).map(|q| deformed[(i, q)] * r[(q, j)]).sum::<f64>() + t[j]
    })
}

/// Sum of squared distances between each landmark source vertex — deformed by
/// `b` and mapped through the similarity `(r, scale, t)` — and its target
/// coordinate. Evaluates only the landmark rows straight from the full
/// `source`/`modes`, so it needs no subsample bookkeeping.
fn landmark_residual_sq(
    source: &DMatrix<f64>,
    modes: &DMatrix<f64>,
    b: &[f64],
    r: &DMatrix<f64>,
    scale: f64,
    t: &[f64],
    landmarks: &[(usize, Vec<f64>)],
) -> f64 {
    let d = source.ncols();
    landmarks
        .iter()
        .map(|(index, target)| {
            (0..d)
                .map(|j| {
                    let fitted = scale
                        * (0..d)
                            .map(|q| {
                                (source[(*index, q)]
                                    + (0..b.len())
                                        .map(|k| modes[(index * d + q, k)] * b[k])
                                        .sum::<f64>())
                                    * r[(q, j)]
                            })
                            .sum::<f64>()
                        + t[j];
                    (fitted - target[j]).powi(2)
                })
                .sum::<f64>()
        })
        .sum()
}
fn score_candidate(
    x: &DMatrix<f64>,
    ty: &DMatrix<f64>,
    sigma2: f64,
    w: f64,
    b: &[f64],
    eigenvalues: &[f64],
    lambda: f64,
) -> f64 {
    let (n, m, d) = (x.nrows(), ty.nrows(), x.ncols());
    let sigma2 = sigma2.max(f64::MIN_POSITIVE);
    let inorm = (1. - w).ln()
        - (m as f64).ln()
        - 0.5 * d as f64 * (2. * std::f64::consts::PI * sigma2).ln();
    let out = if w > 0. {
        w.ln() - (n as f64).ln()
    } else {
        f64::NEG_INFINITY
    };
    let inverse_two_sigma2 = 0.5 / sigma2;
    let mut objective = 0.;
    // Reused across target rows to avoid a per-row allocation.
    let mut kernels = vec![0.0; m];
    for i in 0..n {
        let mut max_log = out;
        for (j, kernel) in kernels.iter_mut().enumerate() {
            let mut distance = 0.0;
            for q in 0..d {
                let delta = x[(i, q)] - ty[(j, q)];
                distance += delta * delta;
            }
            let value = inorm - distance * inverse_two_sigma2;
            max_log = max_log.max(value);
            *kernel = value;
        }
        // Shift into (-inf, 0] and exponentiate with the vectorized kernel.
        for kernel in kernels.iter_mut() {
            *kernel -= max_log;
        }
        exp_non_positive(&mut kernels);
        let mut sum = (out - max_log).exp();
        for &kernel in kernels.iter() {
            sum += kernel;
        }
        objective -= max_log + sum.ln();
    }
    objective
        + 0.5
            * lambda
            * b.iter()
                .zip(eigenvalues)
                .map(|(value, eigen)| value * value / eigen)
                .sum::<f64>()
}

#[cfg(test)]
mod tests {
    use super::{
        PoseMarginalizedConfig, axis_angle, initial_similarity, quaternion_matrix, rotation_lattice,
    };
    use nalgebra::DMatrix;

    fn is_proper_rotation(r: &DMatrix<f64>) -> bool {
        if r.nrows() != 3 || r.ncols() != 3 {
            return false;
        }
        let identity = r.transpose() * r;
        let orthonormal = (0..3).all(|i| {
            (0..3).all(|j| {
                let expected = if i == j { 1.0 } else { 0.0 };
                (identity[(i, j)] - expected).abs() < 1e-9
            })
        });
        orthonormal && (r.determinant() - 1.0).abs() < 1e-9
    }

    #[test]
    fn fixed_scale_initialization_uses_centroid_translation_for_a_fragment() {
        let source = DMatrix::from_row_slice(
            6,
            3,
            &[
                -2.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0,
                2.0, 0.0,
            ],
        );
        let fragment =
            DMatrix::from_row_slice(3, 3, &[3.0, -2.0, 0.5, 4.0, -2.0, 0.5, 3.0, -1.0, 0.5]);
        let rotation = DMatrix::identity(3, 3);
        let (free_scale, _) = initial_similarity(&source, &fragment, &rotation, true);
        let (fixed_scale, translation) = initial_similarity(&source, &fragment, &rotation, false);

        assert!(
            free_scale < 0.75,
            "free scale did not contract: {free_scale}"
        );
        assert_eq!(fixed_scale, 1.0);

        let source_centroid = [0.0, 1.0 / 3.0, 0.0];
        let fragment_centroid = [10.0 / 3.0, -5.0 / 3.0, 0.5];
        for j in 0..3 {
            assert!((source_centroid[j] + translation[j] - fragment_centroid[j]).abs() < 1e-12);
        }
    }

    #[test]
    fn fixed_scale_pose_still_optimizes_rotation_and_translation() {
        let source = DMatrix::from_row_slice(
            8,
            3,
            &[
                -1.4, -0.2, 0.1, -0.7, 0.9, -0.3, 0.1, -1.1, 0.4, 0.8, 0.2, 0.7, 1.5, 1.0, -0.4,
                -1.0, 1.4, 0.8, 0.4, -0.6, -0.9, 1.1, -0.8, 0.2,
            ],
        );
        let rotation = axis_angle([0.0, 0.0, 1.0], 0.25);
        let offset = [0.7, -0.4, 0.2];
        let target = DMatrix::from_fn(source.nrows(), 3, |i, j| {
            (0..3)
                .map(|q| source[(i, q)] * rotation[(q, j)])
                .sum::<f64>()
                + offset[j]
        });
        let modes = DMatrix::zeros(source.len(), 1);
        let result = PoseMarginalizedConfig {
            rotation_count: 1,
            coarse_source_count: source.nrows(),
            coarse_target_count: target.nrows(),
            coarse_rank: 1,
            coarse_iterations: 8,
            coarse_screen_iterations: 8,
            coarse_survivor_count: 1,
            refine_count: 1,
            refine_source_count: None,
            refine_target_count: target.nrows(),
            refine_iterations: 20,
            with_scale: false,
            parallel: false,
            ..Default::default()
        }
        .initialize(&source, &target, &modes, &[1.0])
        .unwrap();

        assert_eq!(result.scale, 1.0);
        let fitted = DMatrix::from_fn(source.nrows(), 3, |i, j| {
            (0..3)
                .map(|q| source[(i, q)] * result.rotation[(q, j)])
                .sum::<f64>()
                + result.translation[j]
        });
        let rms = (fitted - target).norm() / (source.len() as f64).sqrt();
        assert!(rms < 1e-4, "fixed-scale pose RMS was {rms}");
        assert!(result.translation.iter().any(|value| value.abs() > 0.1));
    }

    #[test]
    fn rotation_lattice_entries_are_proper_rotations() {
        // Every lattice entry — identity, the global Shoemake quaternions,
        // and the 20/40/60 degree Fibonacci shells — must be a proper
        // rotation (orthonormal, det = +1), across a range of sizes.
        for count in [1usize, 2, 7, 13, 64, 193] {
            let lattice = rotation_lattice(count, 0);
            assert_eq!(lattice.len(), count, "count {count}");
            // The identity is always first.
            assert!(
                (&lattice[0] - DMatrix::<f64>::identity(3, 3)).amax() < 1e-12,
                "first lattice entry is not the identity"
            );
            for (i, r) in lattice.iter().enumerate() {
                assert!(
                    is_proper_rotation(r),
                    "lattice[{i}] (count {count}) not a proper rotation"
                );
            }
        }
    }

    #[test]
    fn quaternion_and_axis_angle_are_proper_rotations() {
        // A unit quaternion maps to a proper rotation.
        let n = (0.25f64 + 0.36 + 0.04 + 0.16_f64).sqrt();
        let q = quaternion_matrix([0.5 / n, 0.6 / n, 0.2 / n, 0.4 / n]);
        assert!(is_proper_rotation(&q));
        // A rotation of 40 degrees about a unit axis.
        let a = axis_angle([0.0, 0.0, 1.0], 40.0_f64.to_radians());
        assert!(is_proper_rotation(&a));
        // A 40 degree z-rotation maps (1,0,0) as expected.
        let c = 40.0_f64.to_radians().cos();
        let s = 40.0_f64.to_radians().sin();
        assert!((a[(0, 0)] - c).abs() < 1e-12 && (a[(1, 0)] - s).abs() < 1e-12);
    }
}

#[cfg(test)]
mod landmark_pose_tests {
    use super::{PoseMarginalizedConfig, axis_angle};
    use nalgebra::DMatrix;

    fn is_proper(r: &DMatrix<f64>) -> bool {
        let gram = r.transpose() * r;
        let ortho = (0..3).all(|i| (0..3).all(|j| (gram[(i, j)] - f64::from(i == j)).abs() < 1e-8));
        ortho && (r.determinant() - 1.0).abs() < 1e-8
    }

    // 30-point deterministic source with a rank-1 mode basis.
    fn model() -> (DMatrix<f64>, DMatrix<f64>, Vec<f64>) {
        let source = DMatrix::from_fn(30, 3, |i, j| {
            let z = i as f64 + 1.0;
            match j {
                0 => (z * 0.37).sin() * 1.7 + 0.02 * z,
                1 => (z * 0.23).cos() * 0.9,
                _ => (z * 0.11).sin() * (z * 0.07).cos(),
            }
        });
        let modes = DMatrix::from_fn(30 * 3, 1, |row, _| ((row as f64 + 1.0) * 0.17).sin() * 0.3);
        (source, modes, vec![1.0])
    }

    fn config(landmarks: Vec<(usize, Vec<f64>)>, weight: f64) -> PoseMarginalizedConfig {
        PoseMarginalizedConfig {
            rotation_count: 25,
            coarse_source_count: 30,
            coarse_target_count: 30,
            coarse_rank: 1,
            coarse_iterations: 6,
            coarse_screen_iterations: 6,
            coarse_survivor_count: 25,
            refine_count: 4,
            refine_source_count: None,
            refine_target_count: 30,
            refine_iterations: 20,
            with_scale: false,
            parallel: false,
            landmarks,
            landmark_weight: weight,
            ..Default::default()
        }
    }

    #[test]
    fn keypoints_guide_pose_to_consistent_basin() {
        let (source, modes, ev) = model();
        let r = axis_angle([0.2, -0.3, 0.9], 0.7);
        let t = [0.4_f64, -0.25, 0.15];
        let target = DMatrix::from_fn(30, 3, |i, j| {
            (0..3).map(|q| source[(i, q)] * r[(q, j)]).sum::<f64>() + t[j]
        });
        let landmarks: Vec<(usize, Vec<f64>)> = [0usize, 8, 17, 25]
            .iter()
            .map(|&i| (i, (0..3).map(|j| target[(i, j)]).collect()))
            .collect();

        let init = config(landmarks.clone(), 20.0)
            .initialize(&source, &target, &modes, &ev)
            .unwrap();
        assert!(is_proper(&init.rotation));
        // Every anchored keypoint lands on its target under the recovered pose.
        let mut worst = 0.0_f64;
        for (i, q) in &landmarks {
            for j in 0..3 {
                let fitted = init.scale
                    * (0..3)
                        .map(|s| source[(*i, s)] * init.rotation[(s, j)])
                        .sum::<f64>()
                    + init.translation[j];
                worst = worst.max((fitted - q[j]).abs());
            }
        }
        assert!(worst < 0.05, "worst keypoint miss {worst}");
    }

    #[test]
    fn zero_landmark_weight_matches_no_landmarks() {
        let (source, modes, ev) = model();
        let target = DMatrix::from_fn(30, 3, |i, j| source[(i, j)] + 0.3 * (j as f64) - 0.1);
        let plain = config(Vec::new(), 0.0)
            .initialize(&source, &target, &modes, &ev)
            .unwrap();
        // Landmarks provided but weight 0 must reproduce the no-landmark winner.
        let disabled = config(
            vec![(0, vec![1.0, 2.0, 3.0]), (10, vec![0.0, 0.0, 0.0])],
            0.0,
        )
        .initialize(&source, &target, &modes, &ev)
        .unwrap();
        assert!((&plain.rotation - &disabled.rotation).amax() < 1e-12);
        for (a, b) in plain.translation.iter().zip(&disabled.translation) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn pose_rejects_invalid_landmarks() {
        let (source, modes, ev) = model();
        let target = source.clone();
        assert!(
            config(vec![(999, vec![0.0, 0.0, 0.0])], 5.0)
                .initialize(&source, &target, &modes, &ev)
                .is_err()
        );
        assert!(
            config(vec![(0, vec![0.0, 0.0])], 5.0)
                .initialize(&source, &target, &modes, &ev)
                .is_err()
        );
        assert!(
            config(vec![(0, vec![0.0, 0.0, 0.0])], -2.0)
                .initialize(&source, &target, &modes, &ev)
                .is_err()
        );
    }
}
