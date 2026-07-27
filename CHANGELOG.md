# Changelog

All notable changes to this project are documented here. The Rust crate
and the Python package share a version number.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Optional anchored-keypoint (landmark) terms for fragment workflows whose
  shape diverges from the model mean. `AtlasConfig::landmarks` /
  `landmark_weight` keep a handful of corresponding vertices anchored as a soft
  data term through every atlas EM iteration (folded into the E-step
  sufficient statistics, so they steer both the shape-coefficient and
  similarity M-steps). `PoseMarginalizedConfig::landmarks` / `landmark_weight`
  add a keypoint-consistency penalty to every rotation hypothesis, steering the
  global search toward the keypoint-consistent basin. Both are exposed through
  the Python `register_atlas` and `pose_initialize` bindings as
  `landmark_indices` / `landmark_targets` / `landmark_weight`, and are disabled
  by default (empty `landmarks`, `landmark_weight = 0`).

## [3.0.0] - 2026-07-24

### Changed

- **Breaking:** standardized every returned rotation on the direct
  row-vector convention
  `transformed = scale · points · rotation + translation`. Atlas, pose
  initialization, and completion previously exposed the transposed matrix;
  code that applied those rotations as `points @ rotation.T` must drop the
  transpose. This is the reason for the major version bump.
- Raised the declared Rust MSRV to 1.87 to match the current numerical
  dependency stack.
- Added explicit Python 3.12 wheel tests on Linux, macOS, and Windows while
  retaining coverage for Python 3.9 and 3.13.

### Fixed

- Corrected the affine M-step and objective for non-symmetric transforms.
- Rejected non-finite atlas modes and initializers, preserved scale for
  degenerate model clouds, kept atlas variance strictly positive, and
  returned normalized-fit variance in the original coordinate frame.
- Gated the completion example behind the `completion` feature so default
  builds, Clippy, and tests compile.
- Included the advertised PEP 561 `py.typed` marker in Python packages.
- Moved the development benchmark harness from `src/bin/` to `examples/`
  so `cargo install rustcpd` no longer installs a `benchmark` executable.

### Added

- CI parity gate: the cross-implementation parity example now runs in
  serial and parallel on every push and the outputs must be byte-identical.
- Bitwise serial-vs-parallel determinism tests for the affine and atlas
  paths (previously rigid-only), plus an affine determinism property test.
- The MSRV CI job now runs the full test suite rather than `cargo check`.

## [2.1.0]

### Changed

- **Faster low-rank deformable M-step.** The per-iteration solve was
  rewritten to work directly on the pivoted-Cholesky / eigen factor `L`
  (`G ≈ L Lᵀ`) via a direct-`L` Woodbury solve, with all scratch buffers
  reused across iterations and the three large `M×rank` products routed
  through `faer`'s SIMD GEMM (the `k×k` system uses a symmetric `√w`
  self-product, guaranteed positive-definite). Results are unchanged (up
  to floating-point rounding). Measured speedups on 30-iteration fits vs.
  2.0.0: pivoted Cholesky ~2.4× at 2k points, ~2.2× at 3k, ~1.5× at 5k
  (eigen ~1.2–1.8×). The pivoted factorization's rank-1 update is now a
  contiguous GEMV rather than a scalar loop.
- **`DeformableResult` low-rank fields are now computed on demand.** The
  fit keeps the compact factor `L` and derives the orthonormal `(Q, Λ)`
  form only when requested. In Python, `.low_rank_basis` /
  `.low_rank_eigenvalues` are unchanged (still attribute access). In Rust
  they are now methods — `result.low_rank_basis()` /
  `.low_rank_eigenvalues()` / `.low_rank_spectrum()` — plus
  `result.low_rank_factor()` for the raw factor.

### Added

- **Adaptive sparse E-step for atlas registration** (opt-in,
  `AtlasConfig::kdtree_radius_scale` / Python
  `register_atlas(kdtree_radius_scale=...)`). When set, the atlas E-step
  runs dense while the variance is large and permanently switches to the
  `k`-nearest-neighbor sparse E-step (requires `k`) once
  `sigma2 < (‖extent(X)‖ / scale)²`, mirroring the reference's `use_kdtree`
  default (`scale = 10`). This avoids the bias of a fixed-`k` truncation
  while the mixture is still broad. The default is unchanged (exact dense,
  or `k`-sparse from iteration 1 if `k` is set without this option).

## [2.0.0]

First public release under the name **`rustcpd`** (Rust crate, PyPI
package, and Python import — `import rustcpd`). The project is licensed
under **BSD 2-Clause**. This release also contains two behavioral breaks
(the `.kernel` contract for low-rank fits and the constraint-accumulation
semantics), detailed below.

### Fixed

- **Callbacks no longer hold the GIL for the whole registration.** The
  `callback=` path now releases the GIL while the Rust core runs and
  reacquires it only for each callback invocation, so other Python threads
  (GUIs, progress bars) keep running — previously they were starved for the
  entire fit.
- **Callback return values are validated.** `None` or `True` continues,
  `False` stops; any other return (e.g. `0`) now raises `TypeError`
  instead of being silently treated as "continue".
- **Rigid degenerate-source guard is scale-relative.** The previous
  `ypy > f64::MIN_POSITIVE` check missed the common case where a
  coincident source leaves `ypy` at rounding-noise level (≈ `eps²·|y|²`),
  still producing a garbage scale estimate. The guard now floors `ypy`
  relative to the source magnitude, holding the scale whenever it is
  numerically unidentifiable.
- **`low_rank = Some(0)` is rejected** (`PositiveParameter("low_rank")`).
  Previously the eigen path silently returned unmoved points with
  inconsistent weights, and the pivoted path a misleading
  `SingularSystem`.

### Changed

- **Low-rank deformable fits no longer materialize the dense kernel.** The
  low-rank M-step now reconstructs `TY = Y + Q·coefficients` from the
  compact factor (exactly the low-rank kernel action, reusing the small
  per-iteration solve) instead of the dense `M×M` kernel, and the
  pivoted-Cholesky factor is built **matrix-free** — evaluating kernel
  columns on demand from the source points, so memory is `O(M·rank)`
  rather than `O(M²)`. At the 5,000-point scale this removes a ~200 MB
  allocation per fit.
  - `DeformableResult.kernel` (Rust and Python) is populated only for a
    **full-rank** fit (`low_rank = None`); for a low-rank fit it is now an
    empty `0×0` array. The approximation is exposed instead as
    `low_rank_basis` (`Q`) and `low_rank_eigenvalues` (`Λ`), with
    `G ≈ Q·diag(Λ)·Qᵀ`. **Breaking**: code that read `.kernel` after a
    low-rank fit must use the factor fields (or pass `low_rank=None`).
- **Constrained deformable registration deduplicates and accumulates
  constraints.** Identical `(source, target)` pairs are removed, and a
  source pinned to several distinct targets now accumulates their mass
  (a count) and sums the target coordinates — so it is drawn toward the
  mean of its targets — matching the reference. **Breaking** (behavioral):
  previously the last constraint for a shared source won, so inputs with
  duplicate sources now produce different (correct) results.

### Added

- **Configurable pivoted-Cholesky tolerance**
  (`DeformableConfig::pivoted_cholesky_tolerance`, Python
  `register_deformable(low_rank_tolerance=...)`), a residual-diagonal
  early-stop tolerance in `[0, 1)`; `0.0` (default) uses only the
  epsilon-scaled numerical floor. Larger values keep fewer pivots.

## [1.8.0]

### Added

- **Pivoted Cholesky low-rank kernel** for deformable registration, as a
  selectable alternative to the full eigendecomposition. The deformable
  M-step approximates the `M×M` Gaussian kernel with a rank-`low_rank`
  factor; the existing path builds it from a full symmetric
  eigendecomposition (`O(M³)`), which dominates runtime for large clouds.
  The new option builds it with a greedy pivoted (incomplete) Cholesky
  (`O(M·rank²)`), then re-expresses the factor in the identical
  orthonormal `(Q, Λ)` form the solver already consumes, so the EM loop
  and results are otherwise unchanged.
  - Rust: `DeformableConfig::low_rank_method: LowRankMethod`
    (`Eigen` — default, preserves prior behavior — or `PivotedCholesky`).
    `LowRankMethod` is re-exported from the crate root.
  - Python: `register_deformable(..., low_rank_method="eigen" |
    "pivoted_cholesky")` (aliases: `"full"` → eigen;
    `"cholesky"`/`"pivoted"` → pivoted). Default `"eigen"`.
  - Factorization-cost speedup grows with `M`: ~2.6× at 1k, ~4.8× at 2k,
    ~7.5× at 3k, ~12.5× at 5k points (removing ~19 s of one-time setup at
    `M=5000`). Both methods converge to the same solution once the rank
    captures the kernel's energy.
- **Per-iteration callback / early-stop hook** on rigid, affine, and
  deformable. New `register_with(callback)` methods on the Rust registrations
  invoke a closure after each EM iteration with an `IterationState`
  (`iteration`, `sigma2`, `difference`, and the current transformed
  `points`); returning `false` stops early. Exposed to Python as a
  `callback=` argument receiving a dict and returning `False` to stop —
  for progress reporting, live visualization, or a custom stopping rule.
  (The GIL is held while a Python callback runs.)
- **Opt-in input normalization for rigid, affine, and deformable**
  (`normalize`, default off — atlas already had it). Each cloud is
  internally centered on its own centroid and divided by a shared scale
  taken from the target, the fit runs in that unit-scale frame, and the
  result is mapped back: `points`, `sigma2`, and the returned transform
  (rigid `R/scale/t`, affine `B/t`, and `DeformableResult::transform` for
  new points) all come back in the original coordinates. This makes the
  absolute deformable `beta`/`alpha` length scales meaningful for clouds in
  large physical units (e.g. CT coordinates in millimetres), where the raw
  defaults would otherwise mis-scale the kernel. The shared scale preserves
  rigid's isotropic-scale contract. New public helper
  `apply_deformation(source, weights, beta, normalization, z)` evaluates a
  warp (frame-aware) from its parts.

### Fixed

- **Rigid registration no longer returns a non-finite transform on a
  degenerate source.** With `scale` optimization enabled and a source
  whose points coincide (or a single point), the posterior-weighted source
  spread is zero and the scale update formed `0/0 = NaN`, silently
  returning a non-finite result. The scale is now left unchanged when it is
  unidentifiable.
- **Shape completion works for arbitrary dimensionality.** A hard-coded
  3-element scratch buffer panicked for `D ≥ 4` models; it is now sized to
  the data.
- **`sigma2` stays strictly positive when `tolerance == 0`.** Rigid and
  affine derived their variance floor from `tolerance`, so the legitimate
  "run to `max_iterations`" setting (`tolerance = 0`) collapsed the floor
  to zero and could drive a later E-step to divide by zero. Both now use
  an absolute, extent-scaled floor, matching the deformable and atlas
  paths.

### Tests

- k-NN sparse E-step with `k ≪ N` (real neighbor truncation, previously
  only exercised with `k == N`), truncated and rank-deficient pivoted
  Cholesky, degenerate-source rigid regression, and Python coverage for
  `low_rank_method` (both methods, an alias, and the unknown-method
  `ValueError`).

## [1.7.0]

### Added

- **Shape completion and calibrated per-point uncertainty** (opt-in). The
  atlas is a linear-Gaussian shape model, so the posterior over its
  coefficients given a *partial* observation is Gaussian and closed form.
  New machinery keeps that posterior instead of collapsing it to a point:
  - `ShapePosterior` (Rust, behind the `completion` cargo feature; Python
    always on) with `predict()` (completed shape), `predictive_variance()`
    and `predictive_covariance(i)` (per-point confidence map),
    `coefficient_mean` / `coefficient_covariance`, and deterministic
    `sample_coefficients` / `sample_shapes` ensembles.
  - `AtlasResult::posterior(...)` / `.posterior(...)` builds it post-hoc
    from a fit; visibility is inferred from the fitted correspondence,
    optionally anchored by a scalar `completeness` prior. `complete_shape`
    is the explicit-input constructor.
  - **Discrepancy (model-inadequacy) variance** (`estimate_discrepancy`, on
    by default): the observed residual the model cannot explain inflates
    the predictive uncertainty and flags out-of-distribution fragments
    (`discrepancy_variance` / `noise_variance` accessors).
  - `mixture_sample_shapes` (Rust **and** Python) turns competing pose
    hypotheses into honest multi-modal completions.
  - `rustcpd.calibration` (Python): split-conformal
    calibration so predictive intervals achieve nominal coverage, with a
    worked end-to-end example (`python/examples/calibration.py`).
- The whole feature is post-hoc and additive: it never runs inside
  `register()`, so the complete-data paths are byte-for-byte unaffected
  (verified: parity bitwise-identical, default build carries no completion
  symbols).

### Notes

- `AtlasConfig::lambda_regularization` is a prior *temperature*; completion
  uses the statistically correct prior (`temperature = 1`) by default.
- Completion is validated against an independent brute-force Gaussian
  -conditioning reference (agreement to ~1e-9) in both Rust and Python.

## [1.6.0]

### Added

- **Python bindings** (`pip install rustcpd`): `register_rigid`,
  `register_affine`, `register_deformable`, `register_atlas`,
  `pose_initialize`, `gaussian_kernel`, `initialize_sigma2`, plus the
  `transform` and `correspondences` APIs below. Built with PyO3 + maturin
  against the CPython stable ABI (`abi3-py39`) — one wheel per platform
  serves every CPython ≥ 3.9. Type stubs and a `py.typed` marker included.
- `DeformableResult::transform` (Rust) / `DeformableResult.transform`
  (Python): evaluate the learned continuous deformation at arbitrary
  points, so a coarse registration can drive a dense warp. `source` and
  `beta` are now exposed on the result.
- `correspondences` (Rust and Python): soft point-to-point matches from a
  fitted registration — the CPD posterior plus the best target match and
  its confidence per source point.
- `AtlasResult::reconstruct` / `.reconstruct` (Python): rebuild the fitted
  statistical shape model at full resolution from a subsampled
  registration, and `AtlasResult::apply_similarity` / `.apply_similarity`
  to apply only the recovered pose to arbitrary points.
- `EmConfig::single_precision` (and the `single_precision` keyword in
  Python): opt-in `f32` dense E-step (distances and `exp` in single
  precision, statistics accumulated in `f64`) for ~1e-7 accuracy at lower
  cost; off by default, still deterministic.

### Changed

- σ²-adaptive truncated E-step: once the Gaussian variance is small, terms
  below double-precision resolution are skipped via k-d-tree range
  queries, collapsing the cost of long registrations. Exact at `f64`.
- Streaming dense E-step never materializes the `M×N` posterior;
  const-generic 2-D/3-D inner loops and unrolled deterministic reductions.
- `faer` powers the deformable path's large `M×M` LU and symmetric
  eigendecomposition; low-rank and atlas systems are GEMM-shaped.
- Vectorized `exp` (Cephes, ≤ 2 ulp) with AVX2+FMA runtime dispatch.
- Repository restructured as a Cargo workspace (`rustcpd`
  core crate + `python` bindings crate).

### Fixed

- Correct σ² M-step when rigid registration runs with `scale = false`.
- Scale-relative diagonal jitter in the affine solve.
- Publishable crate metadata (SPDX `license`, `readme`, `rust-version`,
  `docs.rs` configuration); `LICENSE` vendored into the crate.

### Performance

Relative to the first Rust implementation, dense rigid and atlas
registration at 3,000 points are ~7× faster in `f64` (≈8–9× with
`single_precision`); 100-iteration atlas runs at 1,000 points are ~14–16×
faster. Numerical agreement with the original is ≤ 3e-12 relative across a
parity suite covering every registration family. See
[`rustcpd/BENCHMARKS.md`](rustcpd/BENCHMARKS.md).

[3.0.0]: https://github.com/agporto/rustcpd/releases/tag/v3.0.0

<!-- 2.1.0 and earlier predate the public repository and have no tags. -->
