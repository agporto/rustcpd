"""End-to-end conformal calibration of shape-completion uncertainty.

Generates a population of complete shapes from a linear model, and for each
one: hides a contiguous region, registers the atlas to the partial cloud,
completes it, and records the per-point error and predicted variance on the
held-out region. Those feed split-conformal calibration, after which a
stated 90% interval covers the truth at ~90% on a held-out split.

Run: python examples/calibration.py
"""

import numpy as np

import rustcpd as cpd
from rustcpd import calibration


def base_model(m, k):
    z = (np.arange(m) + 1.0)[:, None]
    mean = np.column_stack(
        (np.sin(z[:, 0] * 0.3) * 1.5, np.cos(z[:, 0] * 0.2), np.sin(z[:, 0] * 0.13))
    )
    rows = np.arange(m * 3)[:, None]
    modes = np.sin((rows + 1 + 5 * np.arange(k)) * 0.17) + 0.3 * np.cos(
        (rows + 2 * np.arange(k) + 1) * 0.09
    )
    modes, _ = np.linalg.qr(modes)
    eigenvalues = np.array([0.6 / (c + 1) for c in range(k)])
    return mean, modes, eigenvalues


def complete_one(mean, modes, eigenvalues, coeffs, observed, noise, rng):
    """Return (held-out errors, held-out predicted variances) for one shape."""
    m, k = mean.shape[0], len(eigenvalues)
    truth = mean + (modes @ coeffs).reshape(m, 3) + noise * rng.randn(m, 3)
    partial = truth[:observed]
    fit = cpd.register_atlas(
        partial, mean, modes, eigenvalues,
        lambda_regularization=1.0, optimize_similarity=True,
        tolerance=1e-8, max_iterations=200,
    )
    post = fit.posterior(partial, mean, modes, eigenvalues, completeness=observed / m)
    completed = post.predict()
    var = post.predictive_variance()
    held = slice(observed, m)
    errors = np.sqrt(np.sum((completed[held] - truth[held]) ** 2, axis=1))
    return errors, var[held]


def main():
    m, k = 60, 5
    mean, modes, eigenvalues = base_model(m, k)
    observed = 42
    rng = np.random.RandomState(0)

    # A population of shapes: sample coefficients from the model prior.
    n_shapes = 60
    all_err, all_var = [], []
    for _ in range(n_shapes):
        coeffs = rng.randn(k) * np.sqrt(eigenvalues)
        errors, variances = complete_one(
            mean, modes, eigenvalues, coeffs, observed, noise=0.01, rng=rng
        )
        all_err.append(errors)
        all_var.append(variances)
    errors = np.concatenate(all_err)
    variances = np.concatenate(all_var)

    # Split calibration / test and calibrate at 90%.
    half = len(errors) // 2
    cal = calibration.ConformalCalibrator.fit(errors[:half], variances[:half], alpha=0.1)
    coverage = calibration.empirical_coverage(errors[half:], variances[half:], cal)
    print(f"target coverage 90%, empirical {coverage * 100:.1f}%  (scale q = {cal.scale:.3f})")

    curve = calibration.coverage_curve(
        errors[:half], variances[:half], errors[half:], variances[half:],
        alphas=[0.2, 0.1, 0.05],
    )
    print("nominal -> empirical:", [(round(n, 2), round(e, 3)) for n, e in curve])


if __name__ == "__main__":
    main()
