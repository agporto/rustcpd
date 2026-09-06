"""Installed-wheel regressions for the atlas Gram optimization.

384 points give 1,152 coordinate rows. Ranks 15/16 straddle the triangular
path threshold; ranks 32/64 also exceed the 1,000,000-work parallel threshold.
No reference output is rounded and same-build comparisons include signed zero.
"""

import numpy as np
import pytest

import rustcpd as cpd


def model(rank, count=384):
    z = np.linspace(0.0, 1.0, count)
    mean = np.column_stack((3.0 * z, (0.2 + z) * np.sin(13.0 * z),
                            (0.4 + z * z) * np.cos(7.0 * z)))
    rows = mean.size
    modes = np.sqrt(2.0 / rows) * np.cos(
        np.pi * (np.arange(rows)[:, None] + 0.5)
        * (np.arange(rank)[None, :] + 1.0) / rows
    )
    eigenvalues = 0.1 / (1.0 + np.arange(rank) / 8.0)
    coefficients = 0.03 * np.sin(np.arange(rank) + 0.4)
    deformed = mean + (modes @ coefficients).reshape(mean.shape)
    target = deformed[:160] + np.array([0.02, -0.015, 0.01])
    return mean, modes, eigenvalues, target


def assert_bits(actual, expected):
    actual = np.ascontiguousarray(np.asarray(actual, dtype=np.float64))
    expected = np.ascontiguousarray(np.asarray(expected, dtype=np.float64))
    assert actual.shape == expected.shape
    assert np.isfinite(actual).all()
    assert np.isfinite(expected).all()
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))


def fit_snapshot(fit):
    result = {name: getattr(fit, name) for name in
              ("points", "coefficients", "rotation", "translation", "scale", "sigma2",
               "difference", "mixing_weights", "outlier_weight", "iterations")}
    result["state_density"] = fit.state.outlier_density
    return result


def pipeline_snapshot(rank, normalize, parallel):
    """Shared by wheel tests and the two-installed-wheel release comparison."""
    mean, modes, eigenvalues, target = model(rank)
    options = dict(with_scale=False, adaptive_mixing=0.05, outlier_weight=0.1,
                   lambda_regularization=0.7, tolerance=0.0, parallel=parallel)
    first = cpd.register_atlas(target, mean, modes, eigenvalues, sigma2=0.03,
                               normalize=normalize, max_iterations=5, **options)
    resumed = cpd.register_atlas(target, mean, modes, eigenvalues,
                                 initial_state=first.state, normalize=not normalize,
                                 max_iterations=3, **options)
    post = resumed.posterior(target, mean, modes, eigenvalues,
                              prior_temperature=0.7, visibility_floor=0.0,
                              estimate_discrepancy=False)
    result = {}
    for stage, fit in (("first", first), ("resumed", resumed)):
        for name, values in fit_snapshot(fit).items():
            result[stage + "." + name] = values
    result.update({
        "post.mean": post.coefficient_mean,
        "post.covariance": post.coefficient_covariance,
        "post.prediction": post.predict(),
        "post.variance": post.predictive_variance(),
        "post.samples": post.sample_coefficients(3, seed=41),
    })
    return result


@pytest.mark.parametrize("rank", [15, 16, 32, 64])
@pytest.mark.parametrize("normalize", [False, True])
def test_high_rank_wheel_pipeline_is_bitwise_parallel_deterministic(rank, normalize):
    serial = pipeline_snapshot(rank, normalize, False)
    parallel = pipeline_snapshot(rank, normalize, True)
    assert serial.keys() == parallel.keys()
    for name in serial:
        assert_bits(parallel[name], serial[name])
    assert serial["first.iterations"] == 5
    assert serial["resumed.iterations"] == 3
    assert np.ptp(serial["first.mixing_weights"]) > 1e-4


@pytest.mark.parametrize("rank", [16, 64])
@pytest.mark.parametrize("parallel", [False, True])
def test_high_rank_one_step_matches_independent_dense_reference(rank, parallel):
    """Full NumPy Gram/solve oracle, independent of the Rust triangular helper."""
    mean, modes, eigenvalues, target = model(rank)
    sigma2, outlier, regularization = 0.03, 0.1, 0.7
    distance2 = np.sum((mean[:, None, :] - target[None, :, :]) ** 2, axis=2)
    kernel = np.exp(-distance2 / (2.0 * sigma2))
    background = ((2.0 * np.pi * sigma2) ** (mean.shape[1] / 2.0)
                  * outlier / (1.0 - outlier) * len(mean) / len(target))
    responsibilities = kernel / (kernel.sum(axis=0) + background)
    mass = responsibilities.sum(axis=1)
    weighted_target = responsibilities @ target
    system = modes.T @ (np.repeat(mass, mean.shape[1])[:, None] * modes)
    system += np.diag(regularization * sigma2 / eigenvalues)
    rhs = modes.T @ (weighted_target - mass[:, None] * mean).ravel()
    expected = np.linalg.solve(system, rhs)
    fit = cpd.register_atlas(target, mean, modes, eigenvalues,
                             sigma2=sigma2, outlier_weight=outlier,
                             lambda_regularization=regularization,
                             optimize_similarity=False, with_scale=False,
                             normalize=False, max_iterations=1, tolerance=0.0,
                             parallel=parallel)
    assert fit.iterations == 1
    assert np.isfinite(fit.coefficients).all()
    np.testing.assert_allclose(fit.coefficients, expected, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(fit.points, mean + (modes @ expected).reshape(mean.shape),
                               rtol=1e-9, atol=1e-10)
