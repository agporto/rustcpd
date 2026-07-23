"""Tests for shape completion, uncertainty, and calibration."""

import numpy as np
import pytest

import rustcpd as cpd
from rustcpd import calibration


def model(m, d, k):
    z = (np.arange(m) + 1.0)[:, None]
    mean = np.column_stack(
        (np.sin(z[:, 0] * 0.3) * 1.5, np.cos(z[:, 0] * 0.2), np.sin(z[:, 0] * 0.13) * np.cos(z[:, 0] * 0.07))
    )
    rng = np.arange(m * d)[:, None]
    modes = np.sin((rng + 1 + 5 * np.arange(k)) * 0.17) + 0.3 * np.cos((rng + 2 * np.arange(k) + 1) * 0.09)
    modes, _ = np.linalg.qr(modes)  # orthonormal columns
    eigenvalues = np.array([0.6 / (c + 1) for c in range(k)])
    return mean, modes, eigenvalues


def reference_posterior(modes, eigenvalues, observed, residual, d, sigma2):
    """Independent numpy Gaussian-conditioning reference."""
    k = modes.shape[1]
    rows = np.array([p * d + j for p in observed for j in range(d)])
    u = modes[rows, :]
    r = residual.reshape(-1)[rows]
    lam = np.diag(eigenvalues)
    cov_y = u @ lam @ u.T + sigma2 * np.eye(len(rows))
    gain = lam @ u.T @ np.linalg.inv(cov_y)
    b_mean = gain @ r
    sigma_b = lam - gain @ u @ lam
    return b_mean, sigma_b


def test_complete_shape_matches_numpy_conditioning():
    m, d, k = 12, 3, 4
    mean, modes, eigenvalues = model(m, d, k)
    truth = np.array([0.4, -0.25, 0.15, -0.1])
    shape = mean + (modes @ truth).reshape(m, d)
    residual = shape - mean
    observed = list(range(8))
    weight = np.zeros(m)
    weight[observed] = 1.0
    sigma2 = 0.01
    post = cpd.complete_shape(
        mean, modes, eigenvalues, residual, weight, sigma2,
        np.eye(d), 1.0, np.zeros(d),
    )
    ref_mean, ref_cov = reference_posterior(modes, eigenvalues, observed, residual, d, sigma2)
    assert np.allclose(post.coefficient_mean, ref_mean, atol=1e-9)
    assert np.allclose(post.coefficient_covariance, ref_cov, atol=1e-9)


def test_atlas_posterior_completes_partial_shape():
    m, d, k = 60, 3, 5
    mean, modes, eigenvalues = model(m, d, k)
    truth = np.array([0.4, -0.3, 0.2, -0.12, 0.08])
    deformed = mean + (modes @ truth).reshape(m, d)
    angle = 0.2
    r = np.array([[np.cos(angle), -np.sin(angle), 0.0],
                  [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]])
    full_target = 1.1 * (deformed @ r) + np.array([0.3, -0.2, 0.1])
    observed_count = 40
    partial = full_target[:observed_count]

    fit = cpd.register_atlas(
        partial, mean, modes, eigenvalues,
        lambda_regularization=1.0, optimize_similarity=True,
        tolerance=1e-8, max_iterations=200,
    )
    post = fit.posterior(partial, mean, modes, eigenvalues,
                         completeness=observed_count / m)
    completed = post.predict()
    posed_mean = fit.scale * (mean @ fit.rotation) + fit.translation
    held = slice(observed_count, m)
    completion_err = np.sqrt(np.sum((completed[held] - full_target[held]) ** 2))
    mean_err = np.sqrt(np.sum((posed_mean[held] - full_target[held]) ** 2))
    assert completion_err < 0.5 * mean_err

    # Confidence map: observed region more certain than held-out region.
    var = post.predictive_variance()
    assert var.shape == (m,)
    assert var[:observed_count].mean() < var[observed_count:].mean()
    # Full per-point covariance is symmetric PSD-ish.
    c0 = post.predictive_covariance(0)
    assert c0.shape == (d, d)
    assert np.allclose(c0, c0.T, atol=1e-10)


def test_sampling_moments_and_determinism():
    m, d, k = 12, 3, 4
    mean, modes, eigenvalues = model(m, d, k)
    truth = np.array([0.4, -0.25, 0.15, -0.1])
    residual = (modes @ truth).reshape(m, d)
    weight = np.zeros(m)
    weight[:8] = 1.0
    post = cpd.complete_shape(
        mean, modes, eigenvalues, residual, weight, 0.02, np.eye(d), 1.0, np.zeros(d),
    )
    samples = post.sample_coefficients(40000, 12345)
    assert samples.shape == (40000, k)
    assert np.allclose(samples.mean(axis=0), post.coefficient_mean, atol=5e-3)
    emp_cov = np.cov(samples, rowvar=False)
    assert np.allclose(emp_cov, post.coefficient_covariance, atol=5e-3)
    # Deterministic in the seed.
    a = post.sample_shapes(20, 7)
    b = post.sample_shapes(20, 7)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_calibration_achieves_nominal_coverage():
    # Synthetic: per-point errors are N(0, sd^2) with sd = sqrt(variance).
    # (errors here are the Euclidean magnitudes of 1-D residuals => |N(0,sd)|.)
    rng = np.random.RandomState(0)
    n = 5000
    variances = rng.uniform(0.5, 4.0, size=n)
    errors = np.abs(rng.randn(n) * np.sqrt(variances))
    # Split calibration / test.
    cal_err, test_err = errors[: n // 2], errors[n // 2 :]
    cal_var, test_var = variances[: n // 2], variances[n // 2 :]
    for alpha in (0.2, 0.1, 0.05):
        cal = calibration.ConformalCalibrator.fit(cal_err, cal_var, alpha=alpha)
        cov = calibration.empirical_coverage(test_err, test_var, cal)
        # Coverage should be close to and at least ~ nominal.
        assert cov >= (1 - alpha) - 0.03, f"alpha={alpha} coverage {cov}"
        assert cov <= (1 - alpha) + 0.06, f"alpha={alpha} coverage {cov}"


def test_calibration_coverage_curve_monotone():
    rng = np.random.RandomState(1)
    variances = rng.uniform(0.5, 3.0, size=4000)
    errors = np.abs(rng.randn(4000) * np.sqrt(variances))
    half = 2000
    curve = calibration.coverage_curve(
        errors[:half], variances[:half], errors[half:], variances[half:],
        alphas=[0.2, 0.1, 0.05],
    )
    nominal = [c[0] for c in curve]
    empirical = [c[1] for c in curve]
    # Higher nominal target => higher empirical coverage.
    assert nominal == sorted(nominal)
    assert empirical == sorted(empirical)


def test_discrepancy_flags_out_of_model_data():
    m, d, k = 16, 3, 4
    mean, modes, eigenvalues = model(m, d, k)
    truth = np.array([0.4, -0.25, 0.15, -0.1])
    in_model = (modes @ truth).reshape(m, d)
    weight = np.ones(m)
    clean = cpd.complete_shape(
        mean, modes, eigenvalues, in_model, weight, 1e-6, np.eye(d), 1.0, np.zeros(d),
    )
    assert clean.discrepancy_variance < 1e-3
    # Add signal outside the 4-mode span.
    contaminated = in_model + 0.3 * np.sin(np.arange(m * d).reshape(m, d) * 1.7)
    dirty = cpd.complete_shape(
        mean, modes, eigenvalues, contaminated, weight, 1e-6, np.eye(d), 1.0, np.zeros(d),
    )
    assert dirty.discrepancy_variance > clean.discrepancy_variance + 1e-3
    assert dirty.predictive_variance().mean() > clean.predictive_variance().mean()
    # Disabling zeroes it.
    off = cpd.complete_shape(
        mean, modes, eigenvalues, contaminated, weight, 1e-6, np.eye(d), 1.0, np.zeros(d),
        estimate_discrepancy=False,
    )
    assert off.discrepancy_variance == 0.0


def test_mixture_sample_shapes_spans_components():
    m, d, k = 8, 3, 3
    mean, modes, eigenvalues = model(m, d, k)
    weight = np.ones(m)
    ra = np.full((m, d), 0.5)
    rb = np.full((m, d), -0.5)
    pa = cpd.complete_shape(mean, modes, eigenvalues, ra, weight, 0.01, np.eye(d), 1.0, np.zeros(d))
    pb = cpd.complete_shape(mean, modes, eigenvalues, rb, weight, 0.01, np.eye(d), 1.0, np.zeros(d))
    shapes = cpd.mixture_sample_shapes([(1.0, pa), (1.0, pb)], 200, 42)
    assert len(shapes) == 200
    # Deterministic.
    again = cpd.mixture_sample_shapes([(1.0, pa), (1.0, pb)], 200, 42)
    assert all(np.array_equal(x, y) for x, y in zip(shapes, again))
    overall = np.mean([s[0, 0] for s in shapes])
    lo, hi = sorted([pa.predict()[0, 0], pb.predict()[0, 0]])
    assert lo < overall < hi


def test_end_to_end_calibration_coverage():
    from rustcpd import calibration

    m, k = 50, 4
    z = (np.arange(m) + 1.0)[:, None]
    mean = np.column_stack((np.sin(z[:, 0] * 0.3) * 1.5, np.cos(z[:, 0] * 0.2), np.sin(z[:, 0] * 0.13)))
    rows = np.arange(m * 3)[:, None]
    modes, _ = np.linalg.qr(np.sin((rows + 1 + 5 * np.arange(k)) * 0.17))
    eigenvalues = np.array([0.6 / (c + 1) for c in range(k)])
    observed = 36
    rng = np.random.RandomState(1)

    errs, vars_ = [], []
    for _ in range(40):
        coeffs = rng.randn(k) * np.sqrt(eigenvalues)
        truth = mean + (modes @ coeffs).reshape(m, 3) + 0.01 * rng.randn(m, 3)
        fit = cpd.register_atlas(truth[:observed], mean, modes, eigenvalues,
                                 lambda_regularization=1.0, tolerance=1e-8, max_iterations=200)
        post = fit.posterior(truth[:observed], mean, modes, eigenvalues, completeness=observed / m)
        completed = post.predict()
        held = slice(observed, m)
        errs.append(np.sqrt(np.sum((completed[held] - truth[held]) ** 2, axis=1)))
        vars_.append(post.predictive_variance()[held])
    errors = np.concatenate(errs)
    variances = np.concatenate(vars_)
    half = len(errors) // 2

    curve = calibration.coverage_curve(
        errors[:half], variances[:half], errors[half:], variances[half:],
        alphas=[0.2, 0.1, 0.05],
    )
    nominal = [c[0] for c in curve]
    empirical = [c[1] for c in curve]
    # Higher nominal target => at least as much empirical coverage, and each
    # lands in a sane band around nominal (loose: within-shape correlation).
    assert empirical == sorted(empirical)
    for nom, emp in curve:
        assert emp >= nom - 0.2, f"nominal {nom} empirical {emp} undercovers badly"
        assert emp <= 1.0
