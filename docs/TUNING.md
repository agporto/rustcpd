# Choosing parameters

Coherent Point Drift has a handful of knobs. Defaults are sensible, but a
few choices matter for accuracy and speed. This guide covers the ones you
will actually reach for. Parameter names are shared between the Rust
configs (`EmConfig`, `DeformableConfig`, …) and the Python keyword
arguments.

## Which algorithm

- **Rigid** — rotation, translation, and (optionally) a single uniform
  scale. Use when the two clouds differ only by pose/size. Set
  `scale = false` to lock scale at 1.
- **Affine** — adds shear and non-uniform scaling. Use for linear
  distortions.
- **Deformable** — a smooth non-linear warp. Use when the shape itself
  changes (soft tissue, growth, articulation).
- **Atlas / SSM** — fit a statistical shape model (`mean + modes @ b`)
  plus a similarity transform. Use when you have a trained model and want
  its coefficients.

## Preprocessing

CPD is not scale-invariant in its parameters: `beta`, `sigma2`, and the
convergence `tolerance` are all in the units of your coordinates. If your
clouds span thousands of units in one dataset and fractions of one in
another, the fixed defaults will mis-scale the kernel. Pass `normalize =
true` (rigid, affine, deformable, and atlas) and the library conditions
the fit in an internal unit-scale frame and maps the result — points,
`sigma2`, and the returned transform — back to your original coordinates
automatically. Prefer this over hand-rolling the center/scale/un-scale
dance. (For deformable, `DeformableResult::transform` on new points is
frame-aware too, so warping a full-resolution mesh still takes raw
coordinates.)

## Deformable: `alpha` and `beta`

These are the two that shape the warp.

- **`beta`** (default 2.0) is the width of the Gaussian smoothing kernel —
  roughly *how far apart two points must be before they can move
  independently*. Large `beta` ⇒ stiffer, more globally coherent motion;
  small `beta` ⇒ localized, wigglier deformation. Scale it to the spacing
  of your points: a good starting point is the typical nearest-neighbor
  distance times a small factor.
- **`alpha`** (default 2.0) is the regularization strength — *how much*
  smoothness is enforced. Larger `alpha` ⇒ more rigid; smaller ⇒ the warp
  follows the data more closely (and can overfit noise).

If the warp is too loose and chases noise, raise `alpha` or `beta`. If it
is too stiff to capture the real deformation, lower them.

## Outliers: `outlier_weight`

The expected fraction of points with no true match, in `[0, 1)`. `0.0`
(default) assumes clean, fully-overlapping clouds. Raise it (say 0.1–0.5)
when there is clutter, partial overlap, or noise; each source point then
has a uniform "background" alternative to matching a specific target
point. Too high and genuine matches get explained away as outliers.

## Speed knobs

None of these change the algorithm's meaning; they trade compute for the
same (or nearly the same) result.

- **`k`** (sparse E-step) — `None` (default) is the exact posterior over
  all pairs. `Some(k)` restricts each source point to its `k` nearest
  targets via a k-d tree, turning an `O(N·M)` step into roughly
  `O(N·log M)`. For large clouds (tens of thousands of points) this is the
  single biggest win; `k` between 10 and 50 is typical. The result is an
  approximation, so verify on your data.
- **`kdtree_radius_scale`** (atlas only) — an *adaptive* companion to `k`.
  When set (with `k`), the atlas E-step stays dense while the variance is
  large and only switches to the fixed-`k` sparse step once
  `sigma2 < (‖extent(X)‖ / scale)²`. This avoids the bias a fixed-`k`
  truncation introduces early on, when the mixture is still broad and far
  points genuinely matter; `scale = 10` matches the reference default.
  Leave unset to apply `k` from the first iteration.
- **`low_rank`** (deformable only) — approximate the `M×M` kernel with a
  rank-`Some(rank)` factor. Default `Some(300)`; `None` forces the exact
  full-rank solve. For large source clouds, low rank is dramatically
  faster; drop the rank until accuracy suffers.
- **`low_rank_method`** (deformable only) — how that factor is built.
  `"eigen"` (default) keeps the leading `rank` eigenpairs via a full
  symmetric eigendecomposition; it is the optimal rank-`rank`
  approximation but costs `O(M³)` to build. `"pivoted_cholesky"` uses a
  greedy pivoted incomplete Cholesky, `O(M·rank²)` — much cheaper to build
  as `M` grows (roughly 7× faster at 3k points, 12× at 5k), converging to
  the same registration. Prefer it for large clouds; at a *heavily*
  truncated rank the eigen basis reconstructs slightly better, so if you
  push the rank very low, compare the two on your data. With **either**
  method a low-rank fit returns compact factors — `low_rank_basis` (`Q`)
  and `low_rank_eigenvalues` (`Λ`) — and an empty `.kernel`. Only
  `"pivoted_cholesky"` also avoids *building* the dense `M×M` kernel
  (columns are evaluated on demand, `O(M·rank)` memory); `"eigen"` forms
  it temporarily for the decomposition and then drops it.
- **`low_rank_tolerance`** (deformable, pivoted Cholesky only) — the
  residual-diagonal early-stop tolerance in `[0, 1)`. Pivoting halts once
  the largest remaining residual diagonal falls to this value; `0.0`
  (default) keeps pivoting to the requested rank subject to a numerical
  floor. Raise it to trade a little accuracy for a smaller factor on very
  smooth kernels.
- **`single_precision`** — evaluate the dense E-step's distances and
  exponentials in `f32` (statistics still accumulate in `f64`). Roughly
  halves E-step cost at ~1e-7 relative accuracy, deterministically. Leave
  off when you need full double precision; turn on for large 2-D/3-D
  workloads where 6–7 significant digits is plenty.
- **`parallel`** — `true` (default) uses the internal thread pool. Results
  are bitwise-identical to serial, so only set `false` to avoid spawning
  threads (e.g. inside your own parallel loop).

## Convergence: `max_iterations` and `tolerance`

`max_iterations` (default 100) caps the EM loop; `tolerance` (default
1e-3) stops it early once the per-iteration change falls below the
threshold. For a tighter fit raise `max_iterations` and/or lower
`tolerance` (e.g. 1e-6). Because `tolerance` is compared against
coordinate-scale quantities, normalize first (see Preprocessing) so a
single tolerance behaves consistently across datasets.

## After registration

- `result.points` is the aligned/deformed source.
- Deformable results carry the learned field: `result.transform(z)`
  applies it to *new* points `z` — warp a full-resolution mesh from a
  coarse registration, or move landmarks.
- `correspondences(target, result.points, result.sigma2)` returns, per
  source point, the best-matching target index and a confidence in
  `[0, 1]` (plus the full soft posterior). Use the confidence to reject
  weak matches.

### Applying a fit to new points

The deformable and atlas results differ in how their fitted model extends
to points beyond the ones registered:

- **Deformable** learns a *continuous* displacement field, so
  `result.transform(z)` warps *any* points `z` — a full-resolution mesh, a
  grid, landmarks. Register a coarse subsample, then drive a dense warp.
- **Atlas** fits a *discrete* shape basis (the modes are defined only at
  the mean-shape vertices), so there is no continuous `transform(z)`.
  Instead, `result.reconstruct(full_mean, full_modes)` rebuilds the fitted
  shape at full resolution by applying the estimated coefficients to a
  denser mean/modes pair and then the fitted pose — the standard SSM
  workflow of fitting on a subsample and reconstructing the dense model.
  `result.apply_similarity(z)` applies only the recovered rotation, scale,
  and translation to arbitrary points, ignoring the shape deformation.

## Pose search on partial objects

The default pose search seeds every rotation with the model centroid on the
target centroid. When the target is a fragment that sits away from the
model's centre (a proximal third, a distal end), the correct pose needs a
translation the search never proposes, and EM has to slide there during
annealing — which it usually does not, because at large `sigma2` the pose
update keeps re-centring the whole model on the fragment. Symptoms: the fit
looks fine for central fragments and "stays centred" for end fragments.

The fragment recipe, in order of importance:

1. **Pin the scale** — `with_scale=False` if the model is already in physical
   units, otherwise `scale_bounds=(lo, hi)` around 1. With a free scale the
   closed-form estimate shrinks the whole model into the fragment, and the
   search cannot tell a fragment from a small complete object.
2. **`translation_anchor_count=4–8`.** Adds fragment-sized local centroids of
   the model as translation seeds ("the target is the part of the model
   around here"). Activates only when the target's RMS radius is below
   `anchor_completeness_threshold` (0.9) of the model's, so complete targets
   are unchanged. Coarse cost scales with the anchors actually used; use the
   screening funnel (`coarse_screen_iterations < coarse_iterations`,
   `coarse_survivor_count` per rotation) to keep it cheap.
3. **Starting variance.** When seeding activates, `initial_sigma2` defaults
   to 0.25 (normalized frame, target RMS radius = 1) instead of the classic
   whole-model estimate. This matters more than anything else on the list:
   on a tapered test rod the seeds alone recovered nothing (the first
   soft M-steps re-centred the model before any seed could take hold), while
   a fragment-scale start recovered the pose with or without adaptive mixing.
   Override with `initial_sigma2=` if your fragment is unusually noisy
   (larger) or the rotation lattice is dense (smaller).
4. **`adaptive_mixing=1.0`** (optional). Lets model points with no data switch
   off, which removes the residual centring pull and gives the fragment's
   distinctive points their proper weight in the fit and the score. Smaller
   `alpha` switches off faster; larger stays closer to classic CPD.

Read `translation_anchors_used` to confirm seeding activated, and
`winner_support` / `distinct_hypotheses` to see how many refined starts
agreed on the winner versus how many genuinely different fits survived.

## Shape completion & uncertainty (partial objects)

When you register an atlas to a *partial* observation and want the missing
region filled in with a confidence estimate, build a posterior from the fit
(`AtlasResult.posterior` / `.posterior`; Rust needs the `completion`
feature). Notes on the knobs:

- **`completeness`** — roughly what fraction of the object was observed, in
  `(0, 1]`. Visibility is inferred from the fitted correspondence: the
  highest-mass `completeness` fraction of model points is treated as
  observed and the rest as predicted. This is a light, robust ask — one
  scalar — and it separates "genuinely unobserved" from "observed but
  poorly fit" better than a fixed threshold. Omit it to fall back to a
  weight floor (`visibility_floor`).
- **`prior_temperature`** — the atlas `lambda_regularization` is a prior
  *temperature*; completion uses the statistically correct prior
  (`temperature = 1`) by default. Set it equal to the `lambda_regularization`
  you fit with if you need the completion's point estimate to reproduce the
  fitted coefficients exactly.
- **Confidence is coefficient-space, not per-point-independent.** A rank-`k`
  model has only `k` degrees of freedom, so once the fragment pins the
  coefficients, *all* points — observed and missing — are similarly
  constrained; per-point variance differences come only from how the modes
  amplify coefficient uncertainty locally. Do not expect missing regions to
  look dramatically more uncertain in a low-rank model.
- **Calibrate before you trust the numbers.** The predictive variances are
  model-based and will be overconfident out-of-distribution. Use
  `rustcpd.calibration` (split-conformal) on held-out complete
  shapes so a stated 90% interval actually covers at 90%.

The completion is the most probable shape *in the model's span* consistent
with the fragment; it interpolates learned covariation and does not
hallucinate novel geometry.
