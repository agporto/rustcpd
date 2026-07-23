"""Isolate the one-time kernel-factorization cost (the only thing that
differs between eigen and pivoted Cholesky). For each method we fit
time = factor_cost + iters * per_iter_cost from two iteration counts run
in the SAME process, so the intercept is the factorization time.
"""

import time
import numpy as np
import rustcpd as cpd

RNG = np.random.default_rng(0)
BETA, ALPHA, LOW_RANK = 2.0, 2.0, 300
SIZES = [1000, 2000, 3000, 5000]
LO, HI = 3, 13   # two iteration counts to fit the line


def make_pair(m, d=3):
    t = np.linspace(0, 1, m)
    y = np.column_stack([
        np.sin(2 * np.pi * t) * 1.7 + 0.15 * RNG.standard_normal(m),
        np.cos(2 * np.pi * t) * 1.3 + 0.15 * RNG.standard_normal(m),
        t * 2.0 + 0.15 * RNG.standard_normal(m),
    ]).astype(np.float64)
    g = cpd.gaussian_kernel(y, y, BETA)
    return y + g @ (0.02 * RNG.standard_normal((m, d))), y


def timed(x, y, method, iters):
    best = float("inf")
    for _ in range(3):                     # take the min of 3 to cut noise
        t0 = time.perf_counter()
        cpd.register_deformable(
            x, y, alpha=ALPHA, beta=BETA, low_rank=LOW_RANK,
            low_rank_method=method, max_iterations=iters,
            tolerance=1e-12, parallel=True)
        best = min(best, time.perf_counter() - t0)
    return best


print(f"{'M':>6} {'method':>17} {'factor(s)':>10} {'per_iter(s)':>12}")
print("-" * 50)
for m in SIZES:
    x, y = make_pair(m)
    for method in ("eigen", "pivoted_cholesky"):
        t_lo = timed(x, y, method, LO)
        t_hi = timed(x, y, method, HI)
        per_iter = (t_hi - t_lo) / (HI - LO)
        factor = t_lo - LO * per_iter
        print(f"{m:>6} {method:>17} {factor:>10.3f} {per_iter:>12.4f}")
    print()
