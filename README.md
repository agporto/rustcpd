# rustcpd

[![CI](https://github.com/agporto/rustcpd/actions/workflows/wheels.yml/badge.svg)](https://github.com/agporto/rustcpd/actions/workflows/wheels.yml)
[![PyPI](https://img.shields.io/pypi/v/rustcpd.svg)](https://pypi.org/project/rustcpd/)
[![License: BSD-2-Clause](https://img.shields.io/badge/License-BSD_2--Clause-blue.svg)](LICENSE)

Fast, deterministic [Coherent Point Drift](https://arxiv.org/abs/0905.2635)
point-set registration — **rigid, affine, deformable, constrained
deformable, and statistical-shape-model / atlas** — with a pure-Rust
numerical core and first-class Python bindings. Plus shape **completion**
and **calibrated per-point uncertainty** for partial objects.

Python: `pip install rustcpd` then `import rustcpd`. Rust: the `rustcpd`
crate in this repository (usable as a path or git dependency).

- **Pure Rust core** — no BLAS, no system dependencies. Wheels ship for
  Linux, macOS, and Windows on CPython ≥ 3.9.
- **Deterministic** — results are bitwise-identical across thread counts
  and to serial execution.
- **Fast** — streaming E-step, σ²-adaptive exact truncation, vectorized
  `exp`, `faer`-backed solves, an optional single-precision mode, and a
  matrix-free pivoted-Cholesky low-rank deformable path (`O(M·rank)`
  memory; ~3.6× faster than v1.7 at 5,000 points).
- **Honest uncertainty** — turn the atlas into a posterior shape model:
  complete partial shapes and get per-point confidence you can calibrate.

```bash
pip install rustcpd
```

---

## Quickstart

Every function takes `(N, D)` NumPy arrays (`float64`; lists and other
dtypes are converted) and releases the GIL while the Rust core runs.
`target` is the fixed cloud, `source` the moving one.

```python
import numpy as np
import rustcpd as cpd

result = cpd.register_rigid(target, source)   # (N, 3) and (M, 3) arrays
aligned = result.points                        # (M, 3) — source aligned to target
R, t, s = result.rotation, result.translation, result.scale
```

Every registration returns the transformed `points` plus the fitted
parameters, the final `sigma2`, and `iterations`.

---

## Registration

### Rigid — rotation, translation, optional scale

```python
result = cpd.register_rigid(target, source, scale=True)
# result.rotation @-convention: aligned = scale * (source @ rotation) + translation
```

Set `scale=False` to lock scale at 1.

### Affine — linear map + translation

```python
result = cpd.register_affine(target, source)
# aligned = source @ result.transform + result.translation
```

### Deformable — smooth non-rigid warp

```python
result = cpd.register_deformable(target, source, alpha=2.0, beta=2.0)
warped = result.points
```

`beta` sets the kernel width (stiffness), `alpha` the regularization
strength.

**Low rank.** By default the `M×M` kernel is approximated with a
rank-`low_rank` factor (default 300) when the source is larger than that.
`low_rank_method="pivoted_cholesky"` builds the factor matrix-free —
`O(M·rank)` memory, dramatically cheaper than the default `"eigen"`
eigendecomposition as `M` grows (~2.3× end-to-end at 3k points, ~3.6× at
5k) — and converges to the same registration. A low-rank fit returns the
compact factors (`result.low_rank_basis`, `result.low_rank_eigenvalues`,
with `G ≈ Q @ diag(Λ) @ Q.T`) and an **empty** `result.kernel`; pass
`low_rank=None` for the exact full-rank solve and a dense kernel.

```python
# Recommended for large clouds:
result = cpd.register_deformable(
    target, source, low_rank=300, low_rank_method="pivoted_cholesky",
)

# Pin known landmark correspondences (source_index -> target_index).
# A source listed with several distinct targets is drawn to their mean:
result = cpd.register_deformable(target, source, constraints=[(0, 0), (25, 25)])

# Fit a coarse subsample, then apply the learned continuous warp to a
# full-resolution mesh or to landmarks:
fit = cpd.register_deformable(target_sub, source_sub, beta=2.0)
warped_full = fit.transform(full_resolution_points)   # any (P, 3) array
```

### Atlas / statistical shape model

Fit `mean + modes @ b` plus a similarity transform. `modes` is
`(M*D, rank)` in point-major row order; `eigenvalues` are the mode
variances.

```python
result = cpd.register_atlas(target, mean, modes, eigenvalues)
b = result.coefficients                         # fitted shape coefficients
# Direct row-vector convention for every variant:
# posed = result.scale * (points @ result.rotation) + result.translation

# Registered on a subsample? Rebuild the dense fitted model:
dense = result.reconstruct(full_mean, full_modes)
# Or apply just the recovered pose to arbitrary points:
posed = result.apply_similarity(points)
```

### Pose-marginalized initialization (3-D global search)

For atlas registration when the initial orientation is unknown — a scored
search over a rotation lattice that also reports how ambiguous the answer
was.

```python
init = cpd.pose_initialize(source, target, modes, eigenvalues)
init.rotation, init.scale, init.translation
init.score_margin          # gap to the runner-up hypothesis
init.effective_hypotheses  # ~1 = unambiguous; larger = near-symmetric
```

---

## Reading a fit

### Soft correspondences

The CPD posterior — which target point each source point matches, and how
confidently:

```python
match = cpd.correspondences(target, result.points, result.sigma2)
match.matches        # (M,) best target index per source point (int64)
match.probability    # (M,) confidence in [0, 1]
match.posterior      # (M, N) full soft assignment matrix
```

---

## Shape completion & calibrated uncertainty

The atlas is a linear-Gaussian shape model, so a **partial** observation
yields a closed-form Gaussian posterior over its coefficients. Keeping that
posterior (instead of collapsing it to a point) gives both a completion and
per-point uncertainty. In Python this is always available; in Rust it is
behind the `completion` cargo feature. It is entirely post-hoc — it never
touches the complete-data registration paths.

### Complete a partial shape

```python
# Register the atlas to a partial scan, then build the posterior.
fit  = cpd.register_atlas(partial_target, mean, modes, eigenvalues)
post = fit.posterior(partial_target, mean, modes, eigenvalues, completeness=0.55)

completed  = post.predict()               # (M, D) filled-in shape (target frame)
confidence = post.predictive_variance()   # (M,) per-point total variance
```

`completeness ∈ (0, 1]` is roughly what fraction of the object was
observed; visibility is otherwise inferred from the fitted correspondence.
You never label points by hand.

### Per-point uncertainty and model inadequacy

```python
post.predictive_covariance(i)   # full (D, D) covariance at point i
post.noise_variance             # assumed observation noise
post.discrepancy_variance       # model-inadequacy variance (see below)
```

By default `posterior(...)` also estimates a **discrepancy** variance — the
part of the observed data the shape model could not explain — and folds it
into the predictive uncertainty. A large `discrepancy_variance` flags an
out-of-distribution fragment the model fits poorly; the intervals widen
accordingly. Disable with `estimate_discrepancy=False`.

### Ensembles and multi-modal completions

```python
ensemble = post.sample_shapes(500, seed=0)   # 500 plausible completions
coeffs   = post.sample_coefficients(500, seed=0)

# Turn competing pose hypotheses into an honestly multi-modal completion:
shapes = cpd.mixture_sample_shapes([(0.6, post_a), (0.4, post_b)], 500, seed=0)
```

### Calibrate the uncertainty (conformal)

Predictive variances are model-based, so they can be overconfident. The
`calibration` submodule wraps split-conformal prediction so a stated 90%
interval actually covers at 90%:

```python
from rustcpd import calibration

# On held-out complete shapes: hide a region, complete it, and record the
# per-point error and predicted variance on the held-out points. Then:
cal = calibration.ConformalCalibrator.fit(errors, variances, alpha=0.1)
radius  = cal.interval_radius(new_variances)          # calibrated 90% radius
covered = cal.covers(new_errors, new_variances)       # per-point booleans
```

See `python/examples/calibration.py` for the full held-out-shape workflow.

---

## Common options

Shared across the registrations (Rust `EmConfig` fields / Python keyword
arguments):

| Option | Meaning |
|---|---|
| `sigma2` | Initial Gaussian variance (estimated when omitted). |
| `max_iterations` | EM iteration cap (default 100). |
| `tolerance` | Early-stop threshold on the convergence criterion. |
| `outlier_weight` | Uniform-outlier mixture weight in `[0, 1)`; raise for clutter / partial overlap. |
| `k` | k-nearest-neighbor sparse E-step (`None` = exact); big speedup on large clouds. |
| `parallel` | Use the internal thread pool (default `True`; bitwise-identical to serial). |
| `single_precision` | Opt-in `f32` E-step (~1e-7 accuracy) — faster on large 2-D/3-D clouds. |
| `normalize` | Condition the fit in an internal unit-scale frame and map the result back — recommended for clouds in large physical units (e.g. CT in mm), where the absolute `beta`/`sigma2` scales would otherwise be wrong. Rigid, affine, deformable, and atlas. |
| `callback` | Per-iteration hook (rigid/affine/deformable): called with `{"iteration", "sigma2", "difference", "points"}`; return `False` to stop early. The GIL is released while the core runs. |

```python
# Progress reporting / custom stopping:
result = cpd.register_deformable(
    target, source, normalize=True,
    callback=lambda s: print(s["iteration"], s["sigma2"]) or s["sigma2"] > 1e-8,
)
```

Parameter-picking guidance lives in [`docs/TUNING.md`](docs/TUNING.md).

---

## Rust

```toml
[dependencies]
rustcpd = "2.1"
# completion + uncertainty:
# rustcpd = { version = "2.1", features = ["completion"] }
```

```rust
use rustcpd::{DMatrix, RigidConfig, RigidRegistration};

let result = RigidRegistration::new(&target, &source, RigidConfig::default())?
    .register()?;
```

Runnable examples are in
[`rustcpd/examples/`](rustcpd/examples/) (rigid
through completion); Python examples in
[`python/examples/`](python/examples/).

---

## Notes and limitations

- **Determinism** is a first-class guarantee: parallel and serial execution
  produce bitwise-identical results, and sampling is reproducible from its
  seed.
- **Completion is within-model.** It returns the most probable shape *in the
  model's span* consistent with the fragment; it interpolates learned
  covariation and does not invent novel geometry. In a low-rank model the
  uncertainty lives in coefficient space, so observed and missing regions
  can look similarly confident — calibrate before trusting the numbers.
- **Sparse (`k`) and `single_precision`** are approximations that trade
  accuracy for speed; verify on your data.

Performance history and methodology:
[`rustcpd/BENCHMARKS.md`](rustcpd/BENCHMARKS.md).
Release history: [`CHANGELOG.md`](CHANGELOG.md).

## Repository layout

| Path | What it is |
|---|---|
| [`rustcpd/`](rustcpd/) | The core Rust library (the `rustcpd` crate). |
| [`python/`](python/) | Python bindings (PyPI: `rustcpd`), PyO3 + maturin. |
| [`benchmarks/`](benchmarks/) | Cross-language comparison harness. |

## Development

```bash
cargo test --release                                    # core tests
cargo test --release --features completion              # + completion
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check

cd python && maturin develop --release                  # build + install bindings
python -m pytest tests -q
```

CI builds and tests wheels for Linux (x86_64, aarch64), macOS
(universal2), and Windows (x64); tagging `v*` publishes to PyPI via trusted
publishing.

## License

BSD 2-Clause. See [`LICENSE`](LICENSE).
