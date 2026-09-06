# rustcpd

Coherent Point Drift point-set registration — rigid, affine, deformable,
constrained deformable, and statistical-shape-model/atlas — powered by a
fast, deterministic, pure-Rust core (no BLAS, no compiled dependencies at
install time: wheels ship for Linux, macOS, and Windows, CPython ≥ 3.9).

```bash
pip install rustcpd
```

```python
import numpy as np
import rustcpd as cpd

# Rigid: recover the similarity transform mapping source onto target.
result = cpd.register_rigid(target, source)          # (N,3)/(M,3) arrays
aligned = result.points                              # (M, 3)
R, t, s = result.rotation, result.translation, result.scale

# Deformable with landmark constraints and an exact full-rank solve.
result = cpd.register_deformable(
    target, source,
    alpha=2.0, beta=2.0, low_rank=None,
    constraints=[(0, 0), (25, 25)],
)

# Large clouds: rank-300 kernel via pivoted Cholesky (much cheaper to
# build than the default "eigen" method as the source grows).
result = cpd.register_deformable(
    target, source, low_rank=300, low_rank_method="pivoted_cholesky",
)

# Statistical shape model (atlas): modes is (M*3, rank), point-major.
result = cpd.register_atlas(target, mean, modes, eigenvalues)
b = result.coefficients
posed = result.scale * (mean @ result.rotation) + result.translation

# Global pose search for atlas initialization (3-D).
init = cpd.pose_initialize(source, target, modes, eigenvalues)

# If source and modes were pre-scaled from a physical-size estimate (for
# example, a target-completeness prior), keep residual scale fixed while
# Pose-EM continues to optimize rotation and translation.
fragment_init = cpd.pose_initialize(
    prescaled_source, fragment, prescaled_modes, eigenvalues, with_scale=False,
)

# A few corresponding keypoints (same locations on the model and the fragment)
# steer the global pose search toward the keypoint-consistent basin, then keep
# those vertices anchored while register_atlas optimizes shape + pose. Helpful
# for fragments whose shape diverges from the mean. Both accept
# landmark_indices (source-vertex indices) + landmark_targets (their observed
# coordinates); off by default. Set the strength with landmark_sigma (a
# physical localization std, preferred) or the heuristic landmark_weight.
guided = cpd.pose_initialize(
    source, fragment, modes, eigenvalues, with_scale=False,
    # Fixed keypoint std τ (here the localization noise ~ 0.02·radius; squared
    # to a variance τ² internally) for both the basin scoring and the refinement
    # anchoring — the principled form, matching register_atlas(landmark_sigma=...)
    # below, so the whole pipeline uses one physical τ. (landmark_weight /
    # refine_landmark_weight remain as heuristic fallbacks.)
    landmark_indices=kp_idx, landmark_targets=kp_xyz,
    landmark_sigma=0.02 * radius, refine_landmark_sigma=0.02 * radius,
)
fit = cpd.register_atlas(
    fragment, source, modes, eigenvalues, with_scale=False,
    initial_rotation=guided.rotation, initial_translation=guided.translation,
    # Prefer landmark_sigma (an explicit localization std τ, here the keypoint
    # noise ~ 0.02·radius, squared internally) over the heuristic landmark_weight:
    # it gives a fixed constraint strength and keeps fit.sigma2 a clean surface
    # residual. fit.landmark_rms reports the landmark fit separately.
    landmark_indices=kp_idx, landmark_targets=kp_xyz, landmark_sigma=0.02 * radius,
)

# The fit's residual variance is a strong failure signal: a wrong pose basin
# cannot fit the fragment. Calibrate sigma2 -> P(correct) on a few labelled
# fits (recalibrate per dataset), then flag low-confidence fragments.
from rustcpd import calibration
cal = calibration.PoseConfidenceCalibrator.fit(sigma2_array, correct_array)
if not cal.trust(fit.sigma2):
    ...  # low confidence: review or collect more keypoints
```

After a **deformable** fit, apply the learned continuous warp to points it
was never trained on — drive a dense mesh from a coarse registration, or
move landmarks:

```python
fit = cpd.register_deformable(target_subsample, source_subsample, beta=2.0)
warped_full = fit.transform(full_resolution_points)   # any (P, D) array
```

Read off **soft correspondences** from any registration — the best target
match per source point and its confidence, plus the full posterior:

```python
match = cpd.correspondences(target, result.points, result.sigma2)
match.matches        # (M,) best target index per source point
match.probability    # (M,) confidence in [0, 1]
match.posterior      # (M, N) full soft assignment matrix
```

**Continue pose initialization without reheating a fragment fit.** Pass the
complete state and retain the same model/EM options:

```python
common = dict(with_scale=False, adaptive_mixing=1.0, outlier_weight=0.05,
              lambda_regularization=0.1)
init = cpd.pose_initialize(mean, partial_target, modes, eigenvalues,
                           translation_anchor_count=6, **common)
fit = cpd.register_atlas(partial_target, mean, modes, eigenvalues,
                         initial_state=init.state, normalize=True, **common)
```

Both `init.state` and `fit.state` contain the pose, shape coefficients, variance,
and mixture. Variance uses original target units and is converted automatically
when the receiving fit normalizes its inputs. Individual initial-pose arguments
cannot be combined with `initial_state`. Saved mixture weights remain active
but fixed if `adaptive_mixing` is omitted in the next call.

**Complete a partial shape and get per-point uncertainty.** The atlas is a
linear-Gaussian shape model, so a partial observation yields a closed-form
posterior over its coefficients:

```python
fit  = cpd.register_atlas(partial_target, mean, modes, eigenvalues)
post = fit.posterior(partial_target, mean, modes, eigenvalues, completeness=0.55)
completed  = post.predict()               # (M, D) filled-in shape
confidence = post.predictive_variance()   # (M,) per-point uncertainty
ensemble   = post.sample_shapes(200, seed=0)   # plausible completions
```

Visibility is inferred from the fitted correspondence; the optional
`completeness ∈ (0, 1]` anchors it (roughly what fraction of the object was
observed). Calibrate the uncertainty to nominal coverage with the
`rustcpd.calibration` submodule (split-conformal). None of
this touches the complete-data registration paths.

`posterior()` uses the fitted variance and mixture, including adaptive weights.
Its `outlier_weight=None` default inherits the fit's outlier model; explicitly
pass `0.0` for clean assignments. Weights transfer to a reordered or denser mean
by nearest-neighbor interpolation of occupancies and renormalization, assuming
comparable surface sampling. `prior_temperature` still defaults to `1.0`; use
the fitted `lambda_regularization` when you want the same shape-prior strength.

Shared keyword arguments on every registration: `sigma2` (initial
variance; estimated when omitted), `max_iterations` (default 100),
`tolerance`, `outlier_weight` (uniform-outlier mixture weight in `[0, 1)`),
`k` (k-nearest-neighbor sparse E-step; `None` = exact), `parallel`
(default `True`; results are bitwise-identical to serial execution), and
`single_precision` (opt-in `f32` E-step, ~1e-7 accuracy, faster on large
2-D/3-D clouds).

Two more on rigid, affine, and deformable (atlas already had the first):
`normalize=True` conditions the fit in an internal unit-scale frame and
maps the result back to your coordinates — recommended for clouds in
large physical units, where the absolute `beta`/`sigma2` defaults would
otherwise be mis-scaled. `callback=` is a per-iteration hook receiving
`{"iteration", "sigma2", "difference", "points"}`; return `False` to stop
early, `True` or `None` to continue:

```python
result = cpd.register_deformable(
    target, source, normalize=True,
    callback=lambda s: s["sigma2"] > 1e-8,   # custom stopping rule
)
```

The heavy lifting happens in Rust with the GIL released, so other Python
threads keep running. Invalid inputs raise `ValueError`. Bundled type
stubs give editors and type checkers full signatures.

Parameter-selection guidance: see `docs/TUNING.md` in the repository.

Source, benchmarks, and the Rust API: see the
[repository](https://github.com/agporto/rustcpd).

License: BSD 2-Clause.
