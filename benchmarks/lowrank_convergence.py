"""Convergence-based benchmark: run each config to its natural stopping
point (real tolerance) so the numbers reflect actual usage, and confirm
pivoted Cholesky and full eigen converge to the same registration.
"""

import time
import numpy as np
import rustcpd as cpd

RNG = np.random.default_rng(0)
BETA = 2.0
ALPHA = 2.0
LOW_RANK = 300
MAX_ITERS = 150
TOL = 1e-5
SIZES = [1000, 2000, 3000, 5000]


def make_pair(m, d=3):
    t = np.linspace(0, 1, m)
    y = np.column_stack([
        np.sin(2 * np.pi * t) * 1.7 + 0.15 * RNG.standard_normal(m),
        np.cos(2 * np.pi * t) * 1.3 + 0.15 * RNG.standard_normal(m),
        t * 2.0 + 0.15 * RNG.standard_normal(m),
    ]).astype(np.float64)
    g = cpd.gaussian_kernel(y, y, BETA)
    w = 0.02 * RNG.standard_normal((m, d))
    x = y + g @ w
    return x, y


def run(x, y, method, single):
    t0 = time.perf_counter()
    r = cpd.register_deformable(
        x, y, alpha=ALPHA, beta=BETA,
        low_rank=LOW_RANK, low_rank_method=method,
        max_iterations=MAX_ITERS, tolerance=TOL,
        parallel=True, single_precision=single,
    )
    return time.perf_counter() - t0, r


def rms(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


print(f"{'M':>6} {'method':>17} {'prec':>5} {'time(s)':>9} {'iters':>6} "
      f"{'rms_vs_target':>14} {'rms_vs_eigen64':>15} {'speedup':>8}")
print("-" * 90)

for m in SIZES:
    x, y = make_pair(m)
    base_dt, base = run(x, y, "eigen", False)
    rows = [
        ("eigen", False, base_dt, base),
        ("eigen", True, *run(x, y, "eigen", True)),
        ("pivoted_cholesky", False, *run(x, y, "pivoted_cholesky", False)),
        ("pivoted_cholesky", True, *run(x, y, "pivoted_cholesky", True)),
    ]
    for method, single, dt, r in rows:
        prec = "f32" if single else "f64"
        print(f"{m:>6} {method:>17} {prec:>5} {dt:>9.3f} {r.iterations:>6} "
              f"{rms(r.points, x):>14.3e} {rms(r.points, base.points):>15.3e} "
              f"{base_dt / dt:>7.2f}x")
    print()
