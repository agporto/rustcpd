# Rust versus Python reference implementation

Measurements are medians of seven complete construction-and-registration runs
in release mode. Both implementations use `f64`, identical analytical inputs,
the same iteration count, and the same dense or k=10 posterior approximation.
The benchmark ran on the Codex Linux environment with nine available AMD EPYC
cores, Python 3.12.13, NumPy 2.3.5, SciPy 1.17.0, and OpenBLAS 0.3.30.

| Method | Points | Iterations | Python | Rust | Python/Rust | Python RMS | Rust RMS |
|---|---:|---:|---:|---:|---:|---:|---:|
| Rigid, dense | 300 | 10 | 8.71 ms | 6.63 ms | **1.31×** | 6.43423e-3 | 6.43423e-3 |
| Rigid, dense | 1,000 | 10 | 108.14 ms | 34.92 ms | **3.10×** | 2.26312e-2 | 2.26312e-2 |
| Rigid, k=10 | 1,000 | 10 | 19.09 ms | 7.03 ms | **2.72×** | 4.27359e-8 | 4.27359e-8 |
| Deformable, dense | 200 | 10 | 6.46 ms | 10.97 ms | 0.59× | 3.84598e-2 | 3.84598e-2 |
| Deformable, k=10 | 200 | 10 | 9.72 ms | 9.54 ms | **1.02×** | 1.34370e-15 | 1.59557e-15 |
| Atlas, dense | 1,000 | 10 | 64.54 ms | 40.66 ms | **1.59×** | 9.87668e-4 | 9.87668e-4 |
| Atlas, k=10 | 1,000 | 10 | 19.85 ms | 12.24 ms | **1.62×** | 3.31930e-8 | 3.31930e-8 |

The numerical agreement is the stronger result: every paired final RMS value
agrees at the displayed precision. Rust helps the sparse paths where the fused
k-d-tree/posterior/statistics pipeline avoids SciPy CSR construction. It also
parallelizes independent dense-posterior columns through the same Rayon pool,
which makes dense rigid and atlas registration faster than these NumPy baselines.
Full-rank deformable registration remains slower because its repeated dense LU
factorization is pure Rust rather than LAPACK-backed.

The Python comparison above predates the second optimization audit below;
the Rust column has since improved by the factors listed there.

## Optimization audit (first pass)

Relative to the original Rust implementation measured on the same runner, the
1,000-point dense rigid benchmark fell from 107.80 ms to 34.92 ms and dense
atlas from 129.80 ms to 40.66 ms. The audit made three math-preserving changes:

- normalize independent dense posterior columns in parallel;
- build a sparse registration's fixed-target k-d tree once, outside the EM loop;
- traverse Gaussian-kernel and deformable-system storage contiguously, with
  independent kernel columns sharing the existing Rayon pool.

## Optimization audit (second pass, v1.4)

Measured on a two-core AMD EPYC cloud sandbox (medians over the
`benchmark` example (`cargo run --release --example benchmark`)'s repeats, 10 iterations per run; times in ms).
Speedups on machines with more cores should be larger for the dense and
deformable paths, which parallelize across the Rayon pool.

| Method | Points | Before | After | Speedup |
|---|---:|---:|---:|---:|
| Rigid, dense | 300 | 8.3 | 5.8 | 1.4× |
| Rigid, dense | 1,000 | 100.6 | 52.6 | 1.9× |
| Rigid, dense | 3,000 | 2,074.9 | 485.6 | 4.3× |
| Rigid, k=10 | 20,000 | 240.1 | 215.7 | 1.1× |
| Deformable, dense | 500 | 169.7 | 66.3 | 2.6× |
| Deformable, dense | 1,000 | 1,310.9 | 296.6 | 4.4× |
| Deformable, k=10 | 1,000 | 1,269.9 | 253.3 | 5.0× |
| Deformable, low-rank 300 | 500 | 600.0 | 58.8 | 10.2× |
| Deformable, low-rank 300 | 1,000 | 2,185.6 | 261.8 | 8.3× |
| Atlas, dense | 1,000 | 120.5 | 58.6 | 2.1× |
| Atlas, dense | 3,000 | 2,060.1 | 482.1 | 4.3× |
| Atlas, k=10 | 5,000 | 84.5 | 68.5 | 1.2× |

The changes, in decreasing order of impact:

- **Streaming dense E-step.** The `M×N` posterior matrix is no longer
  materialized; fixed-size column blocks fold posterior columns straight
  into the fused statistics from a length-`M` scratch buffer. This removes
  the GEMM plus three full passes over `M×N` memory per iteration and cut
  dense rigid/atlas at 3,000 points by ~3×. Block boundaries depend only on
  the problem size, so parallel results remain bitwise-identical to serial.
- **Vectorized `exp`.** The E-step and Gaussian kernel evaluate the Cephes
  rational approximation (≤ 2 ulp) in an auto-vectorized loop under
  runtime-detected AVX2+FMA, falling back to scalar libm elsewhere.
- **`faer` for the deformable `M×M` factorizations.** The per-iteration
  full-rank solve uses faer's parallel partial-pivoting LU, and the
  low-rank path uses faer's parallel symmetric eigendecomposition instead
  of nalgebra's serial one.
- **GEMM-shaped small systems.** The low-rank `QᵀWQ` system, its
  right-hand side, and the atlas mode Gram matrix are built with one
  row-scaled copy and `tr_mul` GEMMs instead of nested scalar loops —
  this took the low-rank path from slower than full-rank to strictly
  faster at these sizes.
- **Symmetric-kernel construction.** `gaussian_kernel(y, y, β)` computes
  one triangle and mirrors it, halving its `exp` count.
- **Thin LTO** (previously discarded, re-tested after the rewrite: now a
  ~10% win on dense rigid). Fat LTO fails to link against faer's gemm
  microkernels and stays off.

Numerical impact: across a 245-value parity suite covering every
registration family (dense/sparse/low-rank/constrained, with and without
outlier weights), the worst relative deviation from the pre-optimization
implementation is 2.7e-12, dominated by the deliberate re-association of
dot products; the vectorized `exp` contributes ≤ 2 ulp per element.

## Optimization audit (third pass, v1.5): σ²-adaptive truncated E-step

A Gaussian term whose exponent lies more than `2σ²·(ln 1e16 + ln M)` beyond
its target's nearest-source distance is smaller than one unit in the last
place of that target's posterior denominator; dropping it cannot change any
double-precision statistic. Once EM drives σ² low enough (a deterministic
probe of 32 targets checks that fewer than ~15% of pairs survive), the
dense E-step switches from streaming over all `N·M` pairs to anchored
k-d-tree range queries whose cost scales with the surviving pairs only.
Registration workloads reach that regime within roughly ten iterations, so
long runs — the realistic case at default `max_iterations = 100` —
collapse in cost while producing identical results up to rounding.

Same two-core environment, 100 iterations (medians, ms):

| Method | Points | Before | After | Speedup |
|---|---:|---:|---:|---:|
| Atlas, dense | 1,000 | 623.8 | 72.5 | 8.6× |
| Atlas, dense | 3,000 | 1,386.7 | 745.8 | 1.9× |
| Rigid, dense | 1,000 | 126.1 | 72.6 | 1.7× |
| Deformable, dense | 500 | 757.4 | 614.6 | 1.2× |

Ten-iteration workloads are unchanged (±3%): early large-σ² iterations
fall back to the streaming path automatically. Atlas benefits most because
its per-iteration cost is E-step-dominated; deformable remains bounded by
its `M×M` solve, and rigid runs stop early once the objective reaches its
f64 fixed point. Determinism is unchanged — the switch depends only on the
inputs, the probe targets are fixed, and per-block edge accumulation
combines in a fixed order, so parallel results remain bitwise-identical to
serial.

## Optimization audit (fourth pass, v1.6): specialization, reductions, f32

Three further changes, same two-core environment:

- **2-D/3-D const-generic inner loops.** The streaming dense pass is
  compiled per dimensionality, unrolling the dot product and `P·X`
  update with an accumulation order identical to the generic pass —
  outputs are bitwise-unchanged.
- **Unrolled deterministic reductions.** The posterior denominator and
  mass were serial floating-point dependency chains (~4 cycles/element);
  eight fixed-order partial accumulators let them vectorize, and the
  column mass reuses the already-computed sum instead of a second pass.
- **Opt-in `EmConfig::single_precision`.** Dense 2-D/3-D distances and
  exponentials evaluate in `f32` (Cephes single-precision `exp`, ≤ ~2e-7
  relative) while all statistics accumulate in `f64`; the truncated
  E-step's provable cutoff also tightens to single-precision resolution.
  Off by default; results remain deterministic and thread-count
  independent. Deformable registration is bounded by its `M×M` solve and
  does not benefit.

| Method | Points | Iter | v1.5 | v1.6 `f64` | v1.6 `f32` |
|---|---:|---:|---:|---:|---:|
| Rigid, dense | 1,000 | 10 | 52.6 | 31.5 | 26.0 |
| Rigid, dense | 3,000 | 10 | 485.6 | 295.9 | 247.6 |
| Atlas, dense | 3,000 | 10 | 527.8¹ | 317.4 | 257.4 |
| Atlas, dense | 1,000 | 100 | 72.5 | 45.9 | 39.5 |
| Atlas, dense | 3,000 | 100 | 745.8 | 460.4 | 402.0 |

¹ v1.5 short-run atlas figure from the second-audit table.

Relative to the original v1.3 implementation, dense rigid/atlas at 3,000
points are now ~7× faster in `f64` and ~8–9× with `single_precision`;
100-iteration atlas runs at 1,000 points are ~14–16× faster. Final RMS
values in `f32` mode agree with `f64` to ~7 significant digits.

**EM acceleration (SQUAREM/Anderson): evaluated and rejected.** After the
truncated E-step, the late iterations an accelerator would eliminate cost
almost nothing: at 1,000 points, iterations 11–100 of an atlas run cost
~0.13 ms each (11 ms of a 46 ms run). Even a perfect accelerator is
bounded at ~1.25× here, against real risks (non-monotone objectives,
rotation-manifold extrapolation). The expensive early iterations are the
ones acceleration cannot skip.

**Deformable full-rank `M×M` solve: kept as faer LU.** A Woodbury-style
cross-iteration update does not apply (the diagonal weight matrix changes
completely every iteration), and Cholesky symmetrization requires
dividing by per-point posterior masses that can be zero.

## Low-rank deformable (v2.0–2.1): pivoted Cholesky and a direct-`L` M-step

The deformable path's cost is dominated, at scale, by its low-rank kernel
approximation: building it and then applying it each EM iteration. Two
changes reshaped this path.

**v2.0 — pivoted Cholesky.** The `low_rank_method = "pivoted_cholesky"`
option replaces the full symmetric eigendecomposition (`O(M³)` to build)
with a matrix-free greedy pivoted incomplete Cholesky (`O(M·rank²)`, kernel
columns evaluated on demand). The dense `M×M` kernel is never allocated, so
memory grows as `O(M·rank)` instead of `O(M²)`.

**v2.1 — direct-`L` M-step.** The per-iteration solve was rewritten to work
on the factor `L` (`G ≈ L Lᵀ`) directly (`H = (LᵀDL + λI)⁻¹ LᵀF`,
`W = (F − D·L·H)/λ`, displacement `= L·H`) with all scratch reused across
iterations and the three large `M×rank` products routed through `faer`'s
SIMD GEMM. Results are unchanged up to rounding.

Deformable, `low_rank = 300`, 30 iterations, `f64` unless noted, on the
same two-core AMD EPYC sandbox. Times in seconds; peak process RSS in MB.

| Points | Eigen (s) | Pivoted (s) | Pivoted `f32` (s) | Pivoted vs eigen | Pivoted RSS | Eigen RSS |
|---:|---:|---:|---:|---:|---:|---:|
| 1,000 | 0.47 | 0.37 | 0.37 | 1.3× | 57 | 81 |
| 2,000 | 2.39 | 0.96 | 0.93 | 2.5× | 126 | 209 |
| 3,000 | 7.19 | 2.22 | 1.73 | 3.2× | 240 | 411 |
| 5,000 | 28.18 | 5.12 | 3.90 | **5.5×** | 607 | 1,030 |

The eigen column is effectively the pre-2.0 behavior — its `O(M³)`
eigendecomposition dominates and was not changed — so the pivoted/eigen
ratio is roughly the cumulative improvement over the old default. At 5,000
points, pivoted Cholesky is 5.5× faster and uses ~40% less memory; adding
`single_precision` trims the still-`f64` factorization's E-step a further
~1.3×, for ~7× end-to-end (28.2 s → 3.9 s). All registration results match
the eigen path to floating-point rounding (verified by the
`pivoted_cholesky_matches_eigen_low_rank` and `low_rank_deformable_tracks_
full_rank_solution` tests). Reproduce with
`python benchmarks/lowrank_pinned_iters.py`.

The M-step rewrite alone (v2.0 → v2.1, pivoted Cholesky) was ~2.4× at
2,000 points, ~2.2× at 3,000, ~1.5× at 5,000; the win shrinks with `M` as
the (unchanged) E-step grows to occupy a larger share of each iteration.

Run `CPD_REFERENCE_PACKAGE=<package> python benchmarks/reference_comparison.py`
from the repository root to reproduce the Python table on another machine. The
named Python package must expose the registration classes used by the harness.
Treat small timings and speedups as environment-specific; benchmark
representative point-cloud data before choosing a backend.
