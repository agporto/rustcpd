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
    /// Heuristic weight of the keypoint scoring penalty, in units of effective
    /// target points per landmark (the penalty is `0.5 · landmark_weight ·
    /// ‖fitted − target‖² / sigma2`). Because it divides by `sigma2`, the
    /// effective landmark variance is `sigma2 / landmark_weight`, which drifts
    /// with annealing. Prefer [`Self::landmark_sigma`]. `0` disables it.
    pub landmark_weight: f64,
    /// Fixed keypoint localization standard deviation `τ` (a distance) for the
    /// scoring penalty, the principled alternative to [`Self::landmark_weight`].
    /// When `Some`, each hypothesis pays `0.5 · ‖fitted − target‖² / τ²` — a
    /// fixed landmark precision independent of `sigma2`, matching the atlas
    /// [`crate::AtlasConfig::landmark_sigma`]. Takes precedence over
    /// `landmark_weight`. `None` (default) keeps the heuristic weight.
    pub landmark_sigma: Option<f64>,
    /// Anchor the keypoints during the refinement EM as well, using the atlas
    /// heuristic [`crate::AtlasConfig::landmark_weight`] gain. With the default
    /// `0`, keypoints only *score* hypotheses — they choose the basin but do
    /// not hold the refined fit in place; a following anchored `register_atlas`
    /// polish is then expected to supply that. Setting this makes `initialize`
    /// return a keypoint-anchored result on its own. Landmark vertices are
    /// always retained in the refinement source subsample when enabled.
    pub refine_landmark_weight: f64,
    /// Fixed-std alternative to [`Self::refine_landmark_weight`]: anchors the
    /// refinement EM through the atlas [`crate::AtlasConfig::landmark_sigma`]
    /// with localization std `τ`. Takes precedence over
    /// `refine_landmark_weight`. `None` (default) uses the heuristic weight.
    pub refine_landmark_sigma: Option<f64>,
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
    /// Number of translation seeds per rotation (default `1`).
    ///
    /// Every rotation hypothesis needs a starting translation. The classic
    /// seed places the *model centroid* on the target centroid, which is
    /// exactly right for a complete target and exactly wrong for a
    /// fragment: the proximal third of a femur has its centroid a third of
    /// the way along the bone, not in the middle. Values `> 1` add
    /// fragment-sized **local centroids** of the model as extra anchors —
    /// farthest-point samples of the model, each replaced by the centroid of
    /// the model points around it whose RMS radius matches the target's —
    /// and seed one translation per anchor: `t = c_target − s·(a_k·R)`,
    /// i.e. "the target is the part of the model around `a_k`". Anchor 0 is
    /// always the model centroid, so the classic hypothesis is never lost.
    ///
    /// Seeding activates only when the target is demonstrably smaller than
    /// the model (see [`Self::anchor_completeness_threshold`]) **and** the
    /// scale is pinned down — `with_scale = false` or [`Self::scale_bounds`]
    /// set. With a free scale the RMS-radius ratio absorbs the size
    /// difference (the model simply shrinks into the fragment), the
    /// completeness estimate reads 1, and the seeds collapse to the centroid.
    /// The coarse cost scales with the number of anchors actually used;
    /// refinement cost does not.
    pub translation_anchor_count: usize,
    /// Target-to-model RMS-radius ratio (after the seed scale) at or above
    /// which the target is treated as complete and no extra translation
    /// anchors are added, regardless of [`Self::translation_anchor_count`].
    /// Default `0.9`.
    pub anchor_completeness_threshold: f64,
    /// Optional `(min, max)` bounds on the similarity scale, applied to every
    /// seed and every EM scale estimate (coarse and refinement); forwarded to
    /// [`crate::AtlasConfig::scale_bounds`]. Strongly recommended for partial
    /// targets when `with_scale` is on. Ignored when `with_scale = false`.
    pub scale_bounds: Option<(f64, f64)>,
    /// Adaptive per-source mixing proportions for the coarse and refinement
    /// EM; forwarded to [`crate::AtlasConfig::adaptive_mixing`]. Lets model
    /// points with no supporting data switch off, which removes the
    /// centering bias for partial targets. The refined proportions also enter
    /// the final full-cloud score. `None` (default) is classic CPD.
    pub adaptive_mixing: Option<f64>,
    /// Refined hypotheses whose fitted models differ by an RMS distance below
    /// `merge_tolerance × RMS-radius(target)` are treated as the same solution
    /// when computing [`PoseMarginalizedInitialization::score_margin`],
    /// `posterior_entropy`, `effective_hypotheses` and `distinct_hypotheses`.
    /// Default `0.02`. `0` disables merging.
    pub merge_tolerance: f64,
    /// Starting variance for every coarse and refinement EM, in the
    /// normalized frame the atlas runs in (target centred, RMS radius 1).
    /// `None` (default) uses the classic pairwise estimate from the posed
    /// model, which for a fragment is dominated by the *whole model's*
    /// extent and starts the annealing so soft that the first M-steps pull
    /// the model back toward centering on the fragment. With translation
    /// seeding the correct hypotheses already start close, so a variance on
    /// the fragment's own scale — `Some(0.1)`–`Some(0.3)` — is both safe for
    /// them and more discriminating against the wrong ones.
    pub initial_sigma2: Option<f64>,
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
            landmark_sigma: None,
            refine_landmark_weight: 0.0,
            refine_landmark_sigma: None,
            with_scale: true,
            seed: 0,
            parallel: true,
            single_precision: false,
            translation_anchor_count: 1,
            anchor_completeness_threshold: 0.9,
            scale_bounds: None,
            adaptive_mixing: None,
            merge_tolerance: 0.02,
            initial_sigma2: None,
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
    /// Negative-log-posterior score of the winner (lower is better). This is an
    /// unnormalized comparative quantity — it only ranks hypotheses *within a
    /// single run*, and includes the keypoint penalty when landmarks are set.
    /// It is not comparable across runs, across a different keypoint count `k`,
    /// or across a different `τ` / `landmark_weight`; for a cross-fit quality or
    /// confidence signal use [`crate::AtlasResult::sigma2`] via the calibrator
    /// instead.
    pub score: f64,
    /// Score gap to the runner-up; larger means more decisive. Same within-run
    /// caveat as [`Self::score`].
    pub score_margin: f64,
    /// Shannon entropy of the posterior over *distinct* refined solutions
    /// (starts that converged to the same fit are merged first, see
    /// [`PoseMarginalizedConfig::merge_tolerance`]).
    pub posterior_entropy: f64,
    /// `exp(posterior_entropy)` — the effective number of competitive
    /// distinct solutions. `≈ 1` means the winner stands alone; larger
    /// values mean genuinely different fits score comparably (symmetry).
    pub effective_hypotheses: f64,
    /// Total hypotheses evaluated in the coarse pass (rotations × translation
    /// anchors).
    pub hypotheses_evaluated: usize,
    /// Hypotheses carried through refinement (before merging).
    pub hypotheses_refined: usize,
    /// Distinct solutions among the refined hypotheses after merging.
    pub distinct_hypotheses: usize,
    /// Refined starts that converged to the winning solution. Several starts
    /// agreeing is *evidence for* the winner, not ambiguity.
    pub winner_support: usize,
    /// Translation anchors actually used per rotation (1 = centroid only).
    pub translation_anchors_used: usize,
}

#[derive(Clone)]
struct Candidate {
    /// Flat hypothesis id `rotation_index · anchors + anchor_index`
    /// (0 = identity rotation with the centroid anchor).
    index: usize,
    /// Lattice rotation index this hypothesis started from (0 = identity).
    rotation_index: usize,
    /// Translation anchor this hypothesis started from (0 = model centroid).
    #[allow(dead_code)]
    anchor_index: usize,
    score: f64,
    coefficients: Vec<f64>,
    rotation: DMatrix<f64>,
    scale: f64,
    translation: Vec<f64>,
    prior_cost: f64,
    /// Number of refined starts merged into this candidate.
    support: usize,
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
        let scale_bounds = if self.with_scale {
            self.scale_bounds
        } else {
            None
        };
        // Translation seeds. Anchor 0 is always the model centroid (today's
        // behavior); further anchors are fragment-sized local centroids of
        // the model, added only when the target is demonstrably smaller than
        // the (scale-constrained) model. See `translation_anchors`.
        let seed = SeedFrame::new(&coarse_source, &coarse_target, self.with_scale, scale_bounds);
        let anchors = translation_anchors(
            &coarse_source,
            &seed,
            self.translation_anchor_count,
            self.anchor_completeness_threshold,
        );
        let anchor_count = anchors.len();
        // Flat hypothesis list: id = rotation_index · anchor_count + anchor_index,
        // so id 0 is (identity rotation, centroid anchor).
        let hypotheses: Vec<(usize, usize)> = (0..rotations.len())
            .flat_map(|r| (0..anchor_count).map(move |a| (r, a)))
            .collect();
        // Evaluate one hypothesis: run `max_iters` of coarse atlas EM from
        // its initial similarity and score it under the configured mode.
        let evaluate = |id: usize, max_iters: usize| -> Result<Candidate> {
            let (rotation_index, anchor_index) = hypotheses[id];
            let rotation = &rotations[rotation_index];
            let prior = if rotation_index == 0 {
                self.identity_prior_probability
            } else {
                nonidentity_prior
            };
            let translation = seed.translation(rotation, &anchors[anchor_index]);
            let config = AtlasConfig {
                em: EmConfig {
                    sigma2: self.initial_sigma2,
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
                initial_scale: seed.scale,
                initial_translation: Some(translation),
                scale_bounds,
                adaptive_mixing: self.adaptive_mixing,
                ..Default::default()
            };
            let result =
                AtlasRegistration::new(&coarse_target, &coarse_source, &coarse_modes, config)?
                    .register()?;
            let mixing = mixing_factors(result.mixing_weights.as_deref());
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
                    mixing.as_deref(),
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
                index: id,
                rotation_index,
                anchor_index,
                score: data_cost - prior.ln() + kp_penalty,
                coefficients: result.coefficients,
                rotation: result.rotation,
                scale: result.scale,
                translation: result.translation,
                prior_cost: -prior.ln(),
                support: 1,
            })
        };
        // Map `evaluate` over a set of hypothesis ids at a given budget.
        let map_eval = |ids: &[usize], iters: usize| -> Result<Vec<Candidate>> {
            if self.parallel {
                ids.par_iter().map(|&id| evaluate(id, iters)).collect()
            } else {
                ids.iter().map(|&id| evaluate(id, iters)).collect()
            }
        };

        let all_ids: Vec<usize> = (0..hypotheses.len()).collect();
        // Screening runs a prefix of the coarse budget; cap it so an over-large
        // (e.g. defaulted) value simply disables screening instead of erroring.
        let screen_iters = self.coarse_screen_iterations.min(self.coarse_iterations);
        // The survivor count is stated per rotation; with translation seeding
        // it scales with the anchor count so the funnel keeps the same
        // fraction of the (larger) hypothesis set.
        let survivor_count = self.coarse_survivor_count.saturating_mul(anchor_count);
        let use_funnel =
            screen_iters < self.coarse_iterations && survivor_count < hypotheses.len();
        let mut coarse: Vec<Candidate> = if use_funnel {
            // Screen every hypothesis cheaply, keep the best survivors (the
            // identity/centroid hypothesis always retained), then complete
            // only those to the full budget.
            let screened = map_eval(&all_ids, screen_iters)?;
            let survivors = select_survivor_indices(&screened, survivor_count);
            map_eval(&survivors, self.coarse_iterations)?
        } else {
            map_eval(&all_ids, self.coarse_iterations)?
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
        let mut source_indices = subset_indices(source, self.refine_source_count);
        // Anchored refinement needs every landmark vertex present in the
        // refinement subsample, with its landmark index remapped from the
        // full-source row to its subsample position.
        let anchor_refine = (self.refine_landmark_weight > 0.0
            || self.refine_landmark_sigma.is_some())
            && !self.landmarks.is_empty();
        let refine_landmarks: Vec<(usize, Vec<f64>)> = if anchor_refine {
            self.landmarks
                .iter()
                .map(|(index, point)| {
                    let position = source_indices
                        .iter()
                        .position(|&s| s == *index)
                        .unwrap_or_else(|| {
                            source_indices.push(*index);
                            source_indices.len() - 1
                        });
                    (position, point.clone())
                })
                .collect()
        } else {
            Vec::new()
        };
        let refined_target = select_rows(target, &target_indices);
        let refined_source = select_rows(source, &source_indices);
        let refined_modes = select_modes(modes, source.ncols(), &source_indices, eigenvalues.len());
        let refine = |initial: &Candidate| -> Result<Candidate> {
            let mut coefficients = vec![0.0; eigenvalues.len()];
            coefficients[..initial.coefficients.len()].copy_from_slice(&initial.coefficients);
            let config = AtlasConfig {
                em: EmConfig {
                    sigma2: self.initial_sigma2,
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
                landmarks: refine_landmarks.clone(),
                landmark_weight: self.refine_landmark_weight,
                landmark_sigma: self.refine_landmark_sigma,
                scale_bounds,
                adaptive_mixing: self.adaptive_mixing,
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
            // The refined mixing proportions live on the refinement
            // subsample; lift them to the full source by nearest subsample
            // row so the full-cloud score sees the same switched-off points.
            let mixing_full = result.mixing_weights.as_deref().map(|pi| {
                let lifted = lift_to_full(pi, source, &refined_source);
                mixing_factors(Some(&lifted)).expect("lifted proportions are present")
            });
            let score = score_candidate(
                target,
                &full_deformed,
                result.sigma2,
                self.outlier_weight,
                &result.coefficients,
                eigenvalues,
                self.lambda_regularization,
                mixing_full.as_deref(),
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
                rotation_index: initial.rotation_index,
                anchor_index: initial.anchor_index,
                score,
                coefficients: result.coefficients,
                rotation: result.rotation,
                scale: result.scale,
                translation: result.translation,
                prior_cost: initial.prior_cost,
                support: 1,
            })
        };
        let mut refined: Vec<Candidate> = if self.parallel {
            coarse.par_iter().map(refine).collect::<Result<Vec<_>>>()?
        } else {
            coarse.iter().map(refine).collect::<Result<Vec<_>>>()?
        };
        refined.sort_by(|a, b| a.score.total_cmp(&b.score));
        let hypotheses_refined = refined.len();
        // Merge refined starts that converged to the same fit, so the
        // posterior/entropy below is over *distinct* solutions. Without this,
        // several starts landing in the winner's basin (strong evidence for
        // it) would read as ambiguity.
        let merge_radius = self.merge_tolerance * rms_radius(&refined_target);
        let mut clusters =
            merge_converged(refined, &refined_source, &refined_modes, merge_radius);
        let min_score = clusters[0].score;
        let normalizer = clusters
            .iter()
            .map(|candidate| (min_score - candidate.score).exp())
            .sum::<f64>();
        let entropy = clusters
            .iter()
            .map(|candidate| {
                let p = (min_score - candidate.score).exp() / normalizer;
                if p > 0.0 { -p * p.ln() } else { 0.0 }
            })
            .sum::<f64>();
        let distinct_hypotheses = clusters.len();
        let score_margin = clusters
            .get(1)
            .map_or(f64::INFINITY, |second| second.score - min_score);
        let best = clusters.remove(0);
        Ok(PoseMarginalizedInitialization {
            coefficients: best.coefficients,
            rotation: best.rotation,
            scale: best.scale,
            translation: best.translation,
            score: best.score,
            score_margin,
            posterior_entropy: entropy,
            effective_hypotheses: entropy.exp(),
            hypotheses_evaluated: hypotheses.len(),
            hypotheses_refined,
            distinct_hypotheses,
            winner_support: best.support,
            translation_anchors_used: anchor_count,
        })
    }

    /// Keypoint-consistency penalty for one hypothesis, in the same "nats" as
    /// the per-point CPD data cost. With [`Self::landmark_sigma`] `= τ` (a std,
    /// squared internally) it is `0.5 · ‖·‖² / τ²` (fixed landmark precision,
    /// independent of `sigma2`); otherwise it falls back to
    /// the heuristic `0.5 · landmark_weight · ‖·‖² / sigma2`. Zero when the term
    /// is disabled.
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
        let active = !self.landmarks.is_empty()
            && (self.landmark_sigma.is_some() || self.landmark_weight > 0.0);
        if !active {
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
        match self.landmark_sigma {
            Some(sigma) => 0.5 * residual / (sigma * sigma).max(f64::MIN_POSITIVE),
            None => 0.5 * self.landmark_weight * residual / sigma2.max(f64::MIN_POSITIVE),
        }
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
        if self.translation_anchor_count == 0 {
            return Err(Error::PositiveParameter("translation_anchor_count"));
        }
        if !self.anchor_completeness_threshold.is_finite()
            || self.anchor_completeness_threshold <= 0.0
        {
            return Err(Error::PositiveParameter("anchor_completeness_threshold"));
        }
        if let Some((low, high)) = self.scale_bounds {
            if !low.is_finite() || !high.is_finite() || low <= 0.0 || high < low {
                return Err(Error::PositiveParameter("scale_bounds"));
            }
        }
        if self
            .adaptive_mixing
            .is_some_and(|alpha| !alpha.is_finite() || alpha < 0.0)
        {
            return Err(Error::PositiveParameter("adaptive_mixing"));
        }
        if !self.merge_tolerance.is_finite() || self.merge_tolerance < 0.0 {
            return Err(Error::PositiveParameter("merge_tolerance"));
        }
        if self
            .initial_sigma2
            .is_some_and(|value| !value.is_finite() || value <= 0.0)
        {
            return Err(Error::PositiveParameter("initial_sigma2"));
        }
        if !self.landmark_weight.is_finite() || self.landmark_weight < 0.0 {
            return Err(Error::PositiveParameter("landmark_weight"));
        }
        if self
            .landmark_sigma
            .is_some_and(|t2| !t2.is_finite() || t2 <= 0.0)
        {
            return Err(Error::PositiveParameter("landmark_sigma"));
        }
        if !self.refine_landmark_weight.is_finite() || self.refine_landmark_weight < 0.0 {
            return Err(Error::PositiveParameter("refine_landmark_weight"));
        }
        if self
            .refine_landmark_sigma
            .is_some_and(|t2| !t2.is_finite() || t2 <= 0.0)
        {
            return Err(Error::PositiveParameter("refine_landmark_sigma"));
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
/// Flat hypothesis ids of the best `count` screened hypotheses, always
/// including the identity/centroid hypothesis (id 0). Mirrors the reference
/// finalist selection.
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
/// Centroids, RMS radii and the reference scale shared by every translation
/// seed of one search.
struct SeedFrame {
    source_centroid: Vec<f64>,
    target_centroid: Vec<f64>,
    source_radius: f64,
    target_radius: f64,
    /// Scale every seed starts from: the target/source RMS-radius ratio
    /// (the classic estimate) clamped into `scale_bounds`, or exactly 1
    /// when scale is fixed.
    scale: f64,
}

impl SeedFrame {
    fn new(
        source: &DMatrix<f64>,
        target: &DMatrix<f64>,
        with_scale: bool,
        scale_bounds: Option<(f64, f64)>,
    ) -> Self {
        let source_centroid = centroid(source);
        let target_centroid = centroid(target);
        let source_radius = rms_radius(source);
        let target_radius = rms_radius(target);
        let scale = if with_scale {
            let ratio = target_radius / source_radius.max(f64::EPSILON);
            match scale_bounds {
                Some((low, high)) => ratio.clamp(low, high),
                None => ratio,
            }
        } else {
            1.0
        };
        Self {
            source_centroid,
            target_centroid,
            source_radius,
            target_radius,
            scale,
        }
    }

    /// Translation placing model point `anchor` on the target centroid under
    /// `rotation` and the seed scale: `t = c_target − s·(anchor·R)`.
    fn translation(&self, rotation: &DMatrix<f64>, anchor: &[f64]) -> Vec<f64> {
        (0..3)
            .map(|j| {
                self.target_centroid[j]
                    - self.scale * (0..3).map(|q| anchor[q] * rotation[(q, j)]).sum::<f64>()
            })
            .collect()
    }

    /// Fraction of the model the target appears to cover: its RMS radius over
    /// the model's, measured in the seed's scale. `1` for a complete target
    /// (or whenever a free scale absorbed the difference).
    fn completeness(&self) -> f64 {
        self.target_radius / (self.scale * self.source_radius).max(f64::MIN_POSITIVE)
    }
}

fn centroid(points: &DMatrix<f64>) -> Vec<f64> {
    (0..points.ncols())
        .map(|j| (0..points.nrows()).map(|i| points[(i, j)]).sum::<f64>() / points.nrows() as f64)
        .collect()
}

/// Root-mean-square distance of `points` from their centroid.
fn rms_radius(points: &DMatrix<f64>) -> f64 {
    let c = centroid(points);
    ((0..points.nrows())
        .map(|i| squared_from(points, i, &c))
        .sum::<f64>()
        / points.nrows().max(1) as f64)
        .sqrt()
}

/// Translation anchors in the model frame. Anchor 0 is the model centroid.
/// When `count > 1` and the target covers less than `threshold` of the
/// model, farthest-point samples of the model are each replaced by the
/// centroid of the model points around them whose RMS radius matches the
/// target's (in model units), giving "where would a fragment of this size,
/// located here, have its centroid". Near-duplicate anchors are dropped.
fn translation_anchors(
    source: &DMatrix<f64>,
    seed: &SeedFrame,
    count: usize,
    threshold: f64,
) -> Vec<Vec<f64>> {
    let mut anchors = vec![seed.source_centroid.clone()];
    if count <= 1 || seed.completeness() >= threshold {
        return anchors;
    }
    let window_rms = seed.target_radius / seed.scale.max(f64::MIN_POSITIVE);
    let min_separation = 0.25 * window_rms;
    for index in subset_indices(source, Some(count - 1)) {
        let candidate = local_centroid(source, index, window_rms);
        let distinct = anchors.iter().all(|existing| {
            (0..3)
                .map(|j| (existing[j] - candidate[j]).powi(2))
                .sum::<f64>()
                .sqrt()
                > min_separation
        });
        if distinct {
            anchors.push(candidate);
        }
    }
    anchors
}

/// Centroid of the smallest ball of model points around `center` whose RMS
/// radius reaches `target_rms` (at least three points; the whole model if
/// the radius is never reached).
fn local_centroid(source: &DMatrix<f64>, center: usize, target_rms: f64) -> Vec<f64> {
    let m = source.nrows();
    let d = source.ncols();
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&a, &b| {
        squared_pair(source, a, center).total_cmp(&squared_pair(source, b, center))
    });
    // `d` only sizes the accumulators; the loops below iterate them directly.
    let mut sum = vec![0.0; d];
    let mut sum_squared = 0.0;
    let mut centroid = vec![0.0; d];
    for (k, &index) in order.iter().enumerate() {
        for (j, slot) in sum.iter_mut().enumerate() {
            let value = source[(index, j)];
            *slot += value;
            sum_squared += value * value;
        }
        let n = (k + 1) as f64;
        for (mean, &total) in centroid.iter_mut().zip(&sum) {
            *mean = total / n;
        }
        let rms = (sum_squared / n - centroid.iter().map(|v| v * v).sum::<f64>())
            .max(0.0)
            .sqrt();
        if k + 1 >= 3 && rms >= target_rms {
            break;
        }
    }
    centroid
}

/// Relative mixing factors `M·π_m` (floored) from proportions, or `None`.
fn mixing_factors(pi: Option<&[f64]>) -> Option<Vec<f64>> {
    pi.map(|values| {
        let m = values.len() as f64;
        values.iter().map(|&p| (p * m).max(1e-9)).collect()
    })
}

/// Lift per-row proportions on a subsample to the full cloud by nearest
/// subsample row, renormalized to sum to one.
fn lift_to_full(pi: &[f64], full: &DMatrix<f64>, subsample: &DMatrix<f64>) -> Vec<f64> {
    if subsample.nrows() == full.nrows() {
        return pi.to_vec();
    }
    let mut lifted: Vec<f64> = (0..full.nrows())
        .map(|i| {
            let point: Vec<f64> = (0..full.ncols()).map(|j| full[(i, j)]).collect();
            let nearest = (0..subsample.nrows())
                .min_by(|&a, &b| {
                    squared_from(subsample, a, &point)
                        .total_cmp(&squared_from(subsample, b, &point))
                })
                .unwrap_or(0);
            pi[nearest]
        })
        .collect();
    let total: f64 = lifted.iter().sum();
    if total > 0.0 {
        for value in &mut lifted {
            *value /= total;
        }
    }
    lifted
}

/// Merge refined candidates (sorted by score) whose fitted models agree to
/// within `radius` (RMS over the source rows). The best-scoring member
/// represents each cluster and carries the cluster's `support`.
fn merge_converged(
    refined: Vec<Candidate>,
    source: &DMatrix<f64>,
    modes: &DMatrix<f64>,
    radius: f64,
) -> Vec<Candidate> {
    let mut clusters: Vec<Candidate> = Vec::with_capacity(refined.len());
    let mut fits: Vec<DMatrix<f64>> = Vec::with_capacity(refined.len());
    for candidate in refined {
        let fit = apply_model(
            source,
            modes,
            &candidate.coefficients,
            &candidate.rotation,
            candidate.scale,
            &candidate.translation,
        );
        let existing = fits.iter().position(|other| {
            let sum: f64 = fit
                .iter()
                .zip(other.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum();
            (sum / fit.nrows().max(1) as f64).sqrt() <= radius
        });
        match existing {
            Some(k) => clusters[k].support += 1,
            None => {
                clusters.push(candidate);
                fits.push(fit);
            }
        }
    }
    clusters
}

/// Classic seed: model centroid on the target centroid, RMS-radius scale.
/// Equivalent to `SeedFrame::translation` with the centroid anchor and no
/// scale bounds; retained for the unit tests.
#[cfg(test)]
fn initial_similarity(
    source: &DMatrix<f64>,
    target: &DMatrix<f64>,
    rotation: &DMatrix<f64>,
    with_scale: bool,
) -> (f64, Vec<f64>) {
    let seed = SeedFrame::new(source, target, with_scale, None);
    let translation = seed.translation(rotation, &seed.source_centroid);
    (seed.scale, translation)
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
/// Dense CPD negative log-likelihood of `x` under the mixture centred at
/// `ty`, plus the shape prior. `mixing`, when given, holds the relative
/// per-source factors `M·π_m` (see `posterior_stats_weighted`), so the
/// score is the likelihood of the *adaptive* mixture the EM actually fit.
#[allow(clippy::too_many_arguments)]
fn score_candidate(
    x: &DMatrix<f64>,
    ty: &DMatrix<f64>,
    sigma2: f64,
    w: f64,
    b: &[f64],
    eigenvalues: &[f64],
    lambda: f64,
    mixing: Option<&[f64]>,
) -> f64 {
    let (n, m, d) = (x.nrows(), ty.nrows(), x.ncols());
    let sigma2 = sigma2.max(f64::MIN_POSITIVE);
    let inorm = (1. - w).ln()
        - (m as f64).ln()
        - 0.5 * d as f64 * (2. * std::f64::consts::PI * sigma2).ln();
    let log_mixing: Option<Vec<f64>> =
        mixing.map(|values| values.iter().map(|&v| v.max(f64::MIN_POSITIVE).ln()).collect());
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
            let mut value = inorm - distance * inverse_two_sigma2;
            if let Some(log_mixing) = &log_mixing {
                value += log_mixing[j];
            }
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
            for (j, qj) in q.iter().enumerate() {
                let fitted = init.scale
                    * (0..3)
                        .map(|s| source[(*i, s)] * init.rotation[(s, j)])
                        .sum::<f64>()
                    + init.translation[j];
                worst = worst.max((fitted - qj).abs());
            }
        }
        assert!(worst < 0.05, "worst keypoint miss {worst}");
    }

    #[test]
    fn landmark_sigma_scoring_and_refinement_recover() {
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
        // Fixed-variance scoring AND fixed-variance refinement, with a reduced
        // refinement subsample to exercise the landmark remap path.
        let mut cfg = config(landmarks.clone(), 0.0);
        cfg.landmark_sigma = Some(1e-2);
        cfg.refine_landmark_sigma = Some(1e-2);
        cfg.refine_source_count = Some(12);
        let init = cfg.initialize(&source, &target, &modes, &ev).unwrap();
        assert!(is_proper(&init.rotation));
        let mut worst = 0.0_f64;
        for (i, q) in &landmarks {
            for (j, qj) in q.iter().enumerate() {
                let fitted = init.scale
                    * (0..3)
                        .map(|s| source[(*i, s)] * init.rotation[(s, j)])
                        .sum::<f64>()
                    + init.translation[j];
                worst = worst.max((fitted - qj).abs());
            }
        }
        assert!(worst < 0.05, "worst keypoint miss {worst}");
        // A negative fixed variance (scoring or refinement) is rejected.
        let mut bad = config(landmarks.clone(), 0.0);
        bad.landmark_sigma = Some(-1.0);
        assert!(bad.initialize(&source, &target, &modes, &ev).is_err());
        let mut bad2 = config(landmarks, 0.0);
        bad2.refine_landmark_sigma = Some(-1.0);
        assert!(bad2.initialize(&source, &target, &modes, &ev).is_err());
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
    fn anchored_refinement_recovers_and_is_off_by_default() {
        let (source, modes, ev) = model();
        let r = axis_angle([0.1, 0.2, 0.97], 0.9);
        let t = [0.3_f64, -0.2, 0.15];
        let target = DMatrix::from_fn(30, 3, |i, j| {
            (0..3).map(|q| source[(i, q)] * r[(q, j)]).sum::<f64>() + t[j]
        });
        let landmarks: Vec<(usize, Vec<f64>)> = [2usize, 9, 14, 21, 27]
            .iter()
            .map(|&i| (i, (0..3).map(|j| target[(i, j)]).collect()))
            .collect();
        // Force the refinement subsample to drop points so the landmark index
        // remap / extend path runs (refine_source_count < source rows, and
        // some landmark vertices fall outside the farthest-point subsample).
        let mut cfg = config(landmarks.clone(), 20.0);
        cfg.refine_source_count = Some(12);
        cfg.refine_landmark_weight = 25.0;
        let anchored = cfg.initialize(&source, &target, &modes, &ev).unwrap();
        assert!(is_proper(&anchored.rotation));
        let mut worst = 0.0_f64;
        for (i, q) in &landmarks {
            for (j, qj) in q.iter().enumerate() {
                let fitted = anchored.scale
                    * (0..3)
                        .map(|s| source[(*i, s)] * anchored.rotation[(s, j)])
                        .sum::<f64>()
                    + anchored.translation[j];
                worst = worst.max((fitted - qj).abs());
            }
        }
        assert!(worst < 0.05, "anchored refinement keypoint miss {worst}");

        // refine_landmark_weight defaults to 0 and then has no effect.
        let mut off = config(landmarks, 20.0);
        off.refine_source_count = Some(12);
        let base = off
            .clone()
            .initialize(&source, &target, &modes, &ev)
            .unwrap();
        assert_eq!(off.refine_landmark_weight, 0.0);
        // A negative refine weight is rejected.
        let mut bad = off.clone();
        bad.refine_landmark_weight = -1.0;
        assert!(bad.initialize(&source, &target, &modes, &ev).is_err());
        // Sanity: the off run still returns a proper rotation.
        assert!(is_proper(&base.rotation));
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

    // Geodesic angle (degrees) between two rotation matrices.
    fn rotation_error_deg(a: &DMatrix<f64>, b: &DMatrix<f64>) -> f64 {
        let m = a.transpose() * b;
        let trace = (0..3).map(|i| m[(i, i)]).sum::<f64>();
        (((trace - 1.0) / 2.0).clamp(-1.0, 1.0)).acos().to_degrees()
    }

    #[test]
    fn keypoint_penalty_is_fixed_variance_sigma2_independent() {
        // Exercises the scoring term directly. The fixed-variance branch
        // (`landmark_sigma = τ`) uses precision `1/τ²` and must NOT read the
        // annealing `sigma2` at all; the heuristic-weight branch scales as
        // `1/sigma2`. Also covers precedence (both set → fixed wins) and the
        // tiny-variance guard (finite score for an absurdly small τ).
        let (source, modes, _) = model();
        let ident = DMatrix::<f64>::identity(3, 3);
        let b = [0.0];
        let t = [0.0_f64, 0.0, 0.0];
        // Slightly offset landmarks so the residual is strictly positive.
        let landmarks: Vec<(usize, Vec<f64>)> = [0usize, 8, 17]
            .iter()
            .map(|&i| {
                (
                    i,
                    vec![source[(i, 0)] + 0.1, source[(i, 1)], source[(i, 2)]],
                )
            })
            .collect();

        let mut fixed = config(landmarks.clone(), 0.0);
        fixed.landmark_sigma = Some(0.05);
        let p_lo = fixed.keypoint_penalty(&source, &modes, &b, &ident, 1.0, &t, 1e-4);
        let p_hi = fixed.keypoint_penalty(&source, &modes, &b, &ident, 1.0, &t, 9.0);
        assert!(p_lo > 0.0);
        assert!(
            (p_lo - p_hi).abs() < 1e-12,
            "fixed-variance penalty moved with sigma2: {p_lo} vs {p_hi}"
        );

        // Heuristic weight penalty tracks 1/sigma2 (the coupling the
        // reformulation removes): a 9e4× larger sigma2 shrinks it ~9e4×.
        let heur = config(landmarks.clone(), 10.0);
        let h_lo = heur.keypoint_penalty(&source, &modes, &b, &ident, 1.0, &t, 1e-4);
        let h_hi = heur.keypoint_penalty(&source, &modes, &b, &ident, 1.0, &t, 9.0);
        assert!(
            h_lo > 100.0 * h_hi,
            "heuristic penalty should scale with 1/sigma2"
        );

        // Precedence: both set → fixed-variance branch wins, weight ignored.
        let mut both = config(landmarks.clone(), 10.0);
        both.landmark_sigma = Some(0.05);
        let p_both = both.keypoint_penalty(&source, &modes, &b, &ident, 1.0, &t, 1e-4);
        assert!(
            (p_both - p_lo).abs() < 1e-12,
            "precedence: {p_both} vs {p_lo}"
        );

        // Tiny finite variance stays finite (the .max(MIN_POSITIVE) guard).
        let mut tiny = config(landmarks, 0.0);
        tiny.landmark_sigma = Some(1e-12);
        let p_tiny = tiny.keypoint_penalty(&source, &modes, &b, &ident, 1.0, &t, 1e-4);
        assert!(
            p_tiny.is_finite() && p_tiny > 0.0,
            "tiny-variance penalty {p_tiny}"
        );
    }

    #[test]
    fn landmark_sigma_pose_is_scale_equivariant() {
        // Scaling coords by `c` and the landmark std by `c` must recover the
        // identical rotation and coefficients, with translation scaling by `c`.
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
        let mut cfg = config(landmarks.clone(), 0.0);
        cfg.landmark_sigma = Some(0.03);
        cfg.refine_landmark_sigma = Some(0.03);
        let unit = cfg.initialize(&source, &target, &modes, &ev).unwrap();

        let c = 3.0_f64;
        let source_c = source.map(|v| v * c);
        let modes_c = modes.map(|v| v * c);
        let target_c = target.map(|v| v * c);
        let landmarks_c: Vec<(usize, Vec<f64>)> = landmarks
            .iter()
            .map(|(i, q)| (*i, q.iter().map(|v| v * c).collect()))
            .collect();
        let mut cfg_c = config(landmarks_c, 0.0);
        cfg_c.landmark_sigma = Some(0.03 * c);
        cfg_c.refine_landmark_sigma = Some(0.03 * c);
        let scaled = cfg_c
            .initialize(&source_c, &target_c, &modes_c, &ev)
            .unwrap();

        assert!(
            (&unit.rotation - &scaled.rotation).amax() < 1e-6,
            "rotation not scale-equivariant"
        );
        for (a, b) in unit.coefficients.iter().zip(&scaled.coefficients) {
            assert!((a - b).abs() < 1e-5, "coeff {a} vs {b}");
        }
        for (a, b) in unit.translation.iter().zip(&scaled.translation) {
            assert!((a * c - b).abs() < 1e-5 * c, "t {a}*c vs {b}");
        }
    }

    #[test]
    fn fixed_variance_recovers_basin_that_blind_search_misses() {
        // A shape-divergent, rotated target where the blind (surface-only) pose
        // search lands in the WRONG rotation basin. Fixed-variance keypoint
        // scoring steers the search to the correct basin. This is the
        // discriminative version of the recovery test: it asserts the blind
        // baseline actually fails, not merely that keypoints succeed.
        let (source, modes, ev) = model();
        let r = axis_angle([0.15, 0.25, 0.96], 2.5);
        let t = [0.5_f64, -0.3, 0.2];
        let bcoef = 4.0_f64;
        // Deformed shape (mean + modes·b), then rotated + translated.
        let shape = DMatrix::from_fn(30, 3, |i, j| source[(i, j)] + modes[(i * 3 + j, 0)] * bcoef);
        let target = DMatrix::from_fn(30, 3, |i, j| {
            (0..3).map(|q| shape[(i, q)] * r[(q, j)]).sum::<f64>() + t[j]
        });
        let landmarks: Vec<(usize, Vec<f64>)> = [0usize, 8, 17, 25]
            .iter()
            .map(|&i| (i, (0..3).map(|j| target[(i, j)]).collect()))
            .collect();

        let blind = config(Vec::new(), 0.0)
            .initialize(&source, &target, &modes, &ev)
            .unwrap();
        let blind_err = rotation_error_deg(&blind.rotation, &r);

        let mut cfg = config(landmarks.clone(), 0.0);
        cfg.landmark_sigma = Some(0.02);
        cfg.refine_landmark_sigma = Some(0.02);
        let guided = cfg.initialize(&source, &target, &modes, &ev).unwrap();
        let guided_err = rotation_error_deg(&guided.rotation, &r);

        assert!(
            blind_err > 20.0,
            "blind search should miss the basin, got {blind_err}°"
        );
        assert!(
            guided_err < 8.0,
            "fixed-variance search should recover the basin, got {guided_err}°"
        );
    }
}

#[cfg(test)]
mod fragment_seeding_tests {
    use super::{
        Candidate, PoseMarginalizedConfig, SeedFrame, axis_angle, merge_converged,
        translation_anchors,
    };
    use nalgebra::DMatrix;

    /// A tapered, twisted rod: thin at x = 0, thick at x = 4. The taper makes
    /// every segment unique, so a fragment has exactly one home.
    fn tapered_rod(count: usize) -> DMatrix<f64> {
        DMatrix::from_fn(count, 3, |i, j| {
            let z = i as f64 / (count - 1) as f64;
            let radius = 0.1 + 0.6 * z;
            match j {
                0 => 4.0 * z,
                1 => radius * (7.0 * z).sin(),
                _ => radius * (7.0 * z).cos(),
            }
        })
    }

    fn transform(points: &DMatrix<f64>, rotation: &DMatrix<f64>, t: &[f64]) -> DMatrix<f64> {
        DMatrix::from_fn(points.nrows(), 3, |i, j| {
            (0..3)
                .map(|q| points[(i, q)] * rotation[(q, j)])
                .sum::<f64>()
                + t[j]
        })
    }

    fn rms(a: &DMatrix<f64>, b: &DMatrix<f64>) -> f64 {
        (a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            / a.len() as f64)
            .sqrt()
    }

    #[test]
    fn anchors_collapse_to_the_centroid_for_complete_targets() {
        let model = tapered_rod(60);
        let seed = SeedFrame::new(&model, &model, false, None);
        assert!((seed.completeness() - 1.0).abs() < 1e-12);
        let anchors = translation_anchors(&model, &seed, 6, 0.9);
        assert_eq!(anchors.len(), 1, "complete target must not be seeded");
        // A free, unbounded scale absorbs the size difference: no seeding.
        let fragment = DMatrix::from_fn(20, 3, |i, j| model[(i, j)]);
        let free = SeedFrame::new(&model, &fragment, true, None);
        assert!((free.completeness() - 1.0).abs() < 1e-12);
        assert_eq!(translation_anchors(&model, &free, 6, 0.9).len(), 1);
        // count = 1 never seeds.
        let fixed = SeedFrame::new(&model, &fragment, false, None);
        assert_eq!(translation_anchors(&model, &fixed, 1, 0.9).len(), 1);
    }

    #[test]
    fn anchors_include_a_fragment_sized_local_centroid_near_the_true_one() {
        let model = tapered_rod(60);
        // Proximal third; its true centroid sits at x ≈ 0.667.
        let fragment = DMatrix::from_fn(20, 3, |i, j| model[(i, j)]);
        let seed = SeedFrame::new(&model, &fragment, false, None);
        assert!(seed.completeness() < 0.5, "completeness={}", seed.completeness());
        let anchors = translation_anchors(&model, &seed, 6, 0.9);
        assert!(anchors.len() > 1, "fragment must produce extra anchors");
        // Anchor 0 is the model centroid.
        assert!((anchors[0][0] - 2.0).abs() < 1e-9);
        let truth = super::centroid(&fragment);
        let closest = anchors[1..]
            .iter()
            .map(|a| {
                (0..3)
                    .map(|j| (a[j] - truth[j]).powi(2))
                    .sum::<f64>()
                    .sqrt()
            })
            .fold(f64::INFINITY, f64::min);
        assert!(
            closest < 0.25,
            "no anchor near the fragment centroid (closest {closest}); anchors={anchors:?}"
        );
        // Seeding with the true anchor recovers the exact translation.
        let identity = DMatrix::identity(3, 3);
        let t = seed.translation(&identity, &truth);
        assert!(t.iter().all(|v| v.abs() < 1e-9), "t={t:?}");
        // With scale bounds the clamped scale is used for the seed.
        let bounded = SeedFrame::new(&model, &fragment, true, Some((0.8, 1.25)));
        assert!((bounded.scale - 0.8).abs() < 1e-12, "scale={}", bounded.scale);
        assert!(bounded.completeness() < 0.9);
    }

    #[test]
    fn displaced_fragment_pose_is_recovered_with_translation_seeding() {
        let model = tapered_rod(60);
        let modes = DMatrix::zeros(model.len(), 1);
        let rotation = axis_angle([0.0, 0.0, 1.0], 0.3);
        let offset = [0.5, -0.3, 0.2];
        let fragment_model = DMatrix::from_fn(20, 3, |i, j| model[(i, j)]);
        let target = transform(&fragment_model, &rotation, &offset);
        // Plain seeding, then the full fragment recipe (seeding + adaptive
        // mixing + fragment-scale starting variance).
        for (initial_sigma2, adaptive) in [(None, None), (Some(0.2), Some(1.0))] {
            let config = PoseMarginalizedConfig {
                rotation_count: 9,
                coarse_source_count: 60,
                coarse_target_count: 20,
                coarse_rank: 1,
                coarse_iterations: 10,
                refine_count: 6,
                refine_source_count: None,
                refine_target_count: 20,
                refine_iterations: 30,
                with_scale: false,
                parallel: false,
                translation_anchor_count: 6,
                adaptive_mixing: adaptive,
                initial_sigma2,
                ..Default::default()
            };
            let init = config.initialize(&model, &target, &modes, &[1.0]).unwrap();
            assert!(init.translation_anchors_used > 1, "seeding did not activate");
            assert_eq!(init.hypotheses_evaluated, 9 * init.translation_anchors_used);
            let posed = DMatrix::from_fn(20, 3, |i, j| {
                init.scale
                    * (0..3)
                        .map(|q| model[(i, q)] * init.rotation[(q, j)])
                        .sum::<f64>()
                    + init.translation[j]
            });
            let error = rms(&posed, &target);
            assert!(
                error < 0.1,
                "adaptive={adaptive:?}: fragment not placed at the thin end, rms={error}, \
                 translation={:?}",
                init.translation
            );
            assert!(init.distinct_hypotheses >= 1);
            assert!(init.winner_support >= 1);
            assert!(init.distinct_hypotheses <= init.hypotheses_refined);
        }
    }

    #[test]
    fn merging_collapses_starts_that_converged_to_the_same_fit() {
        let model = tapered_rod(30);
        let modes = DMatrix::zeros(model.len(), 1);
        let make = |score: f64, angle: f64, t: [f64; 3]| Candidate {
            index: 0,
            rotation_index: 0,
            anchor_index: 0,
            score,
            coefficients: vec![0.0],
            rotation: axis_angle([0.0, 0.0, 1.0], angle),
            scale: 1.0,
            translation: t.to_vec(),
            prior_cost: 0.0,
            support: 1,
        };
        let refined = vec![
            make(10.0, 0.0, [0.0, 0.0, 0.0]),
            make(10.5, 1e-4, [1e-4, 0.0, 0.0]), // same fit, converged from elsewhere
            make(40.0, 1.5, [0.7, 0.0, 0.0]),  // a genuinely different basin
        ];
        let clusters = merge_converged(refined, &model, &modes, 0.02);
        assert_eq!(clusters.len(), 2);
        assert_eq!(clusters[0].support, 2);
        assert_eq!(clusters[1].support, 1);
        assert!((clusters[0].score - 10.0).abs() < 1e-12, "best member represents");
    }
}
