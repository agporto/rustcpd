# Atlas Gram assembly: accuracy and speed

Measured on 2026-09-06 against `fragment-pose-seeding` at
[`57ed8684e5e61395231a08b4bf586783c44932a3`](https://github.com/agporto/rustcpd/commit/57ed8684e5e61395231a08b4bf586783c44932a3).
That reference includes the state-continuation, adaptive-completion, and
candidate-diversity fixes.

The retained optimization preserves every checked floating-point result
bit for bit. Parallel fragment registration was about 1.4x, 2.1x, and 3.1x
faster at ranks 32, 64, and 128. A direct `faer` replacement was also faster,
but failed strict agreement checks on several fixtures, so it is not used.

## Implementation

The atlas M-step constructs `U^T diag(w) U` and solves a regularized system
with nalgebra's Cholesky factorization. Here **rank** is the number of shape
modes, or columns of `U`; it is separate from the number of model points.

Cholesky consumes only the lower triangle. For at least 16 modes and 256
coordinate rows, `atlas_gram` evaluates only `r(r+1)/2` column dot products
instead of `r^2`. Every retained dot product uses the same nalgebra operation
and operand order as the original `tr_mul`. The upper triangle is unused.
Independent output columns run through Rayon when `parallel` is enabled and
`rows * r * r >= 1,000,000`. There is no parallel reduction within a dot product.
Smaller products retain the original path. Pose search can still parallelize
across hypotheses with each hypothesis's inner parallelism disabled.

The right-hand side, Cholesky solve, priors, EM settings, model rank, candidate
budgets, and completion implementation are unchanged. This optimization does
not approximate or truncate the fitted model.

## Numerical comparison

Identical Rust example source was linked separately against the reference and
candidate libraries. Both used the same lockfile, compiler, release profile,
and `completion` feature. The comparison checks numerical snapshots, not just
the final objective:

- Fitted coordinates, coefficients, rotation, translation, scale, variance,
  objective, iteration count, and adaptive mixture.
- Intermediate iteration budgets 1, 2, 4, and 8 for direct registration.
- Continuation from saved state while changing the normalization setting.
- Selected pose, score, score margin, and candidate diagnostics.
- Completion coefficient mean/covariance, predicted coordinates/variances,
  and samples drawn with an identical seed.

The accuracy grid uses 768 model points, ranks 8/12/32/64/128, and both serial
and parallel execution. It covers 35% fragments, complete targets with automatic
initial variance, raw coordinates, a 10,000-unit coordinate offset, sparse
correspondences, stopping by convergence, nearly duplicate modes, tight
landmarks, and pose search followed by registration. Ordinary fixtures use
orthonormal DCT modes on an asymmetric synthetic surface. The deliberately
nearly duplicate stress modes are not an orthonormal PCA basis.

**All 26 successful cases matched bit for bit: 652,948 finite values checked,
maximum absolute difference zero.** The convergence fixture stopped after
9 iterations in both versions. Two additional nearly singular, weakly
regularized cases returned `SingularSystem` in both versions; these are
reported separately and are not counted as successful fits.

The unit tests additionally compare every lower-triangle entry and the
Cholesky solution to the original implementation, including threshold cases
and weights spanning zero to `1e6`. Thread-count checks cover 1/2/4/8 threads.
The existing parity example agrees byte for byte across both versions and
both serial/parallel settings.

Validation on Rust 1.87.0 passed: 100 tests with `completion`, 90 without it,
`cargo fmt --check`, Clippy over all targets with and without `completion`
(`-D warnings`), and all-feature documentation with warnings denied.

## Whole-registration timings

Hardware: AMD EPYC 9V74; Linux x86-64; eight CPUs in the process affinity and
an eight-CPU cgroup quota; `RAYON_NUM_THREADS=8`. Rust 1.87.0, `--release`,
`codegen-units=1`, thin LTO, and the default target CPU. Compilation was stopped
during timing runs. This is a shared virtualized environment, not an isolated
physical benchmark host.

Each result uses seven paired trials, alternating which binary runs first.
Each process performs an untimed warm-up and then times one complete
registration, including registration setup. Synthetic input generation,
snapshot writing, and post-hoc completion are outside the timer. Direct-fit
timings use 12 EM iterations with the same settings in both versions.
Pose timings include initialization/search and the subsequent registration.

Times below are medians in milliseconds. Speedup is the median of the paired
reference/candidate ratios, so it need not equal the ratio of the two displayed
medians.

| Workload, parallel enabled | Model points | Rank | Original ms | Optimized ms | Paired speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| 35% fragment | 2,000 | 12 | 12.06 | 12.58 | 0.97x |
| 35% fragment | 2,000 | 32 | 29.25 | 21.01 | 1.39x |
| 35% fragment | 2,000 | 64 | 79.94 | 37.32 | 2.13x |
| 35% fragment | 2,000 | 128 | 278.07 | 94.17 | 3.11x |
| Complete target, automatic initial variance | 2,000 | 12 | 46.46 | 52.26 | 0.92x |
| Complete target, automatic initial variance | 2,000 | 32 | 71.92 | 69.32 | 1.11x |
| Complete target, automatic initial variance | 2,000 | 64 | 126.82 | 113.07 | 1.12x |
| Complete target, automatic initial variance | 2,000 | 128 | 314.76 | 131.61 | 2.58x |
| 35% fragment, k=32 sparse correspondences | 5,000 | 128 | 796.04 | 256.12 | 3.09x |
| Pose search, landmarks, then registration | 2,000 | 64 | 418.15 | 386.89 | 1.08x |

Rank 12 takes the original matrix path. The initially slower complete-target
measurement was repeated for **21 pairs**: original median 48.21 ms,
candidate 49.62 ms, median paired speedup **0.99x**, with individual ratios
from 0.84x to 1.36x. There is no demonstrated gain at this rank, and these
measurements do not establish a repeatable regression either. Both the initial
measurement and follow-up are retained in the raw data.

With parallelism disabled, fragment speedups at ranks 12/32/64/128 were
1.00x/1.33x/1.51x/1.65x. Complete-target speedups were
1.00x/1.02x/1.13x/1.32x. Other registration work limits the overall gain;
low-rank coarse pose hypotheses also receive little benefit.

These are synthetic workloads. They demonstrate the opportunity and preserve
the original results on the tested cases; they do not establish a universal
speedup or accuracy guarantee for other hardware, compiler versions, or a
particular anatomical training model. No real anatomical dataset was supplied
for these measurements.

## Why the direct faer replacement was rejected

The experiment changed only Gram assembly to the existing `faer` bridge for
rank >= 16 and at least 256 coordinate rows. The right-hand side and solver
were unchanged. Serial faer was used inside each fit, avoiding small-product
threading overhead and nested pose-search parallelism.

Isolated Gram products were much faster. For example, at 6,000 rows and rank
64, the medians were 10.30 ms for nalgebra and 1.02 ms for serial faer (10.12x).
The relative Frobenius errors were around double-precision roundoff. This
does **not** mean a full registration is 10x faster: separate whole-fit trials
gave 2.43x for the rank-128 fragment, 2.14x for the rank-128 complete target,
and 1.25x for pose search. These were separate runs from the retained
candidate's trials, not a head-to-head ranking of the two optimizations.

The preselected comparison criterion was `atol=rtol=1e-8`, with exact discrete
decisions. **16 of the 26 successful faer cases failed at least one check.**
Examples of maximum absolute differences:

| Case / quantity | Maximum difference |
| --- | ---: |
| Tight-landmark stress, coefficients at iteration 8 | 1.58e-5 |
| Tight-landmark stress, coordinates at iteration 8 | 2.24e-7 |
| Tight-landmark stress, posterior covariance | 4.75e-7 |
| Large-offset continuation, coefficients | 2.83e-5 |
| Large-offset continuation, translation | 1.25e-2 |

The large translation discrepancy is a parameter difference in the offset
coordinate frame, not a claim of the same displacement in fitted coordinates.
These results show sensitivity to changed floating-point accumulation in
poorly conditioned solves. They do not establish that faer is less accurate
against mathematical ground truth. The goal here is agreement with the original
implementation. The retained triangular method achieves that stronger agreement
without changing accumulation order.

## Reproduce

Run from a checkout containing this change with Rust 1.87 and Python + NumPy.
Use an executable temporary directory for build outputs. The following commands
build the exact same harness against two separate library revisions:

```bash
atlas_repo="$PWD"
atlas_run=$(mktemp -d /tmp/rustcpd-atlas-bench.XXXXXX)
git worktree add --detach "$atlas_run/reference-src" 57ed8684e5e61395231a08b4bf586783c44932a3
cp rustcpd/examples/atlas_fit_bench.rs "$atlas_run/reference-src/rustcpd/examples/"

CARGO_BUILD_JOBS=2 cargo build --manifest-path "$atlas_run/reference-src/Cargo.toml" \
  -p rustcpd --release --locked --features completion --example atlas_fit_bench \
  --target-dir "$atlas_run/reference-target"
cp "$atlas_run/reference-target/release/examples/atlas_fit_bench" "$atlas_run/reference"

CARGO_BUILD_JOBS=2 cargo build -p rustcpd --release --locked --features completion \
  --example atlas_fit_bench --target-dir "$atlas_run/candidate-target"
cp "$atlas_run/candidate-target/release/examples/atlas_fit_bench" "$atlas_run/candidate"

taskset -c 0-7 python benchmarks/atlas_faer_comparison.py \
  "$atlas_run/reference" "$atlas_run/candidate" --threads 8 --pairs 7 \
  --require-bitwise --output "$atlas_run/results"
```

Choose CPUs allowed on the local machine; omit `taskset` on non-Linux systems.
For a single workload, add `--case fragment,128,2000,true`. To separate accuracy
from timing, use `--validate-only` or `--timing-only`. Every snapshot value is
written with sufficient decimal digits to round-trip to its original `f64`.
The report includes raw paired times, comparison failures, and binary hashes.

To reproduce the rejected faer algorithm, apply the included patch to the
reference worktree and build a third binary. Its accuracy command is expected
to exit nonzero:

```bash
git -C "$atlas_run/reference-src" apply "$atlas_repo/benchmarks/atlas_faer_trial.patch"
CARGO_BUILD_JOBS=2 cargo build --manifest-path "$atlas_run/reference-src/Cargo.toml" \
  -p rustcpd --release --locked --features completion --example atlas_fit_bench \
  --target-dir "$atlas_run/faer-target"
cp "$atlas_run/faer-target/release/examples/atlas_fit_bench" "$atlas_run/faer"

taskset -c 0-7 python benchmarks/atlas_faer_comparison.py \
  "$atlas_run/reference" "$atlas_run/faer" --validate-only \
  --output "$atlas_run/faer-accuracy"

cargo build -p rustcpd --release --locked --example atlas_product_bench \
  --target-dir "$atlas_run/candidate-target"
RAYON_NUM_THREADS=8 taskset -c 0-7 \
  "$atlas_run/candidate-target/release/examples/atlas_product_bench" 6000 64 7
```

The patch expresses the same faer operations as the measured prototype; the
prototype wrapped them in a private helper. Timings are hardware-dependent.
The initial isolated-product sweep used five pairs and affinity CPUs 0-8
under the same eight-CPU quota; whole-registration runs used CPUs 0-7.

The complete timing samples, per-case accuracy summaries, binary hashes, and
faer microbenchmarks are in
[`results/atlas_gram_20260906.json`](results/atlas_gram_20260906.json).
