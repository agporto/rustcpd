# Coherent Point Drift for Rust

A native Rust library for rigid, affine, deformable, constrained deformable,
statistical-shape-model/atlas, and pose-marginalized point-set registration.

## Design

- `f64` numerical core with explicit validation and typed results.
- Exact dense CPD posterior and k-nearest-neighbor sparse posterior.
- Streaming dense E-step: posterior columns are processed in fixed blocks
  and folded directly into the fused sufficient statistics, so the `M×N`
  posterior matrix is never materialized (memory is `O(M + N)`, not
  `O(M·N)`) unless the full posterior is explicitly requested.
- σ²-adaptive truncated E-step (2-D/3-D): once σ² shrinks, Gaussian terms
  below double-precision resolution are skipped through anchored k-d-tree
  range queries, so late EM iterations cost `O(active pairs)` instead of
  `O(N·M)` while producing the same statistics up to rounding. The switch
  is automatic, probe-based, and deterministic.
- Deterministic parallelism through one Rayon pool: work is partitioned
  into blocks that depend only on the problem size and combined in a fixed
  order, so results are bitwise-identical across thread counts and to
  serial execution. Set `EmConfig::parallel = false` to avoid worker
  threads entirely.
- Vectorized elementwise `exp` (the E-step's dominant cost) using the
  Cephes rational approximation (≤ 2 ulp), auto-vectorized under
  AVX2+FMA with runtime detection and a scalar-libm fallback.
- Opt-in single-precision E-step (`EmConfig::single_precision`): 2-D/3-D
  dense distances and exponentials evaluate in `f32` while every
  statistic still accumulates in `f64` (~1e-7 relative accuracy, still
  deterministic), and the truncated E-step prunes at single-precision
  resolution.
- Sparse registrations build the fixed-target k-d tree once and reuse it
  across EM iterations; the source-cloud Gaussian kernel exploits symmetry
  and computes only one triangle.
- Pure-Rust linear algebra: `nalgebra` for small fixed-size systems and
  `faer` for the large `M×M` factorizations of the deformable path
  (parallel LU and symmetric eigendecomposition). No BLAS runtime, no
  nested native thread pools.
- A k-d tree for two- and three-dimensional sparse registration, with a
  dimension-independent exact fallback.

## Public API

| Algorithm | Rust API |
|---|---|
| Rigid | `RigidRegistration` / `RigidConfig` |
| Affine | `AffineRegistration` / `AffineConfig` |
| Deformable | `DeformableRegistration` / `DeformableConfig` |
| Constrained deformable | `DeformableConfig::constraints` |
| Atlas/SSM | `AtlasRegistration` / `AtlasConfig` |
| Pose-marginalized initialization | `PoseMarginalizedConfig` |

Every returned rotation uses the same row-vector convention:
`transformed = scale · points · rotation + translation`.

`PoseMarginalizedConfig::with_scale = false` fixes the residual pose scale at
1.0 while continuing to optimize rotation and translation. This is intended
for fragment workflows that pre-scale the source shape and modes from an
external physical-size estimate before pose initialization.

A handful of corresponding keypoints — the same anatomical locations marked on
the model and on the target — sharpen fragment fits that diverge in shape from
the mean. Set `AtlasConfig::landmarks` (source-vertex index paired with its
target coordinate) and a landmark strength to keep those vertices anchored as a
soft data term through every EM iteration, and set
`PoseMarginalizedConfig::landmarks` (plus a strength) to let the same
correspondences steer the global rotation search toward the keypoint-consistent
basin. Prefer `landmark_sigma` (a physical keypoint-localization standard
deviation `τ`, squared to a variance internally) over the heuristic
`landmark_weight`; when both are set, `landmark_sigma` takes precedence. Both
terms are off by default (empty `landmarks`, no strength).
Three non-collinear keypoints are enough to resolve the rotation basin; the
partial mesh does the rest.

`EmConfig::k = None` selects the dense E-step. `Some(k)` selects a
source-to-target k-nearest-neighbor approximation.

## Example

```rust
use rustcpd::{DMatrix, EmConfig, RigidConfig, RigidRegistration};

let source = DMatrix::from_row_slice(3, 2, &[0., 0., 1., 0., 0., 1.]);
let target = DMatrix::from_row_slice(3, 2, &[1., 2., 2., 2., 1., 3.]);
let config = RigidConfig {
    em: EmConfig { max_iterations: 100, tolerance: 1e-6, ..Default::default() },
    scale: true,
};
let result = RigidRegistration::new(&target, &source, config)?.register()?;
# Ok::<(), rustcpd::Error>(())
```

## Python bindings

The `python/` crate packages this library for Python (CPython ≥ 3.9,
Linux/macOS/Windows) through PyO3 and maturin, with NumPy arrays in and
out and the GIL released during registration:

```bash
pip install rustcpd
```

```python
import rustcpd as cpd
result = cpd.register_rigid(target, source)   # (N, D) numpy arrays
```

Wheels use the `abi3-py39` stable ABI, so one wheel per platform serves
every supported Python version. `.github/workflows/wheels.yml` builds and
tests Linux (x86_64, aarch64), macOS (universal2), and Windows (x64)
wheels plus an sdist, and publishes to PyPI on version tags via trusted
publishing. Local development: `pip install maturin && maturin develop
--release` inside `python/`.

## Verification

```bash
cargo test --release
cargo clippy --all-targets -- -D warnings
cd ..
CPD_REFERENCE_PACKAGE=<package> python benchmarks/reference_comparison.py
```

The comparison harness constructs identical analytical inputs in both
languages and reports timing and final RMS error. See [BENCHMARKS.md](BENCHMARKS.md).

## Compatibility scope

The algorithms and public results are equivalent, but this is a Rust-native API,
not a drop-in replacement for the Python classes. Python callbacks and mutable
intermediate attributes are intentionally absent. The low-rank deformable path
uses a deterministic symmetric eigendecomposition. Pose initialization uses a
global/local rotation mixture and priors, with a deterministic low-discrepancy
sequence.
