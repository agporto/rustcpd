"""Regression coverage for pose -> atlas -> completion state transfer."""

import numpy as np
import pytest

import rustcpd as cpd


def rod(n=60):
    z = np.linspace(0.0, 1.0, n)
    r = 0.1 + 0.6 * z
    return np.column_stack((4 * z, r * np.sin(7 * z), r * np.cos(7 * z)))


def fragment(mean):
    c, s = np.cos(0.3), np.sin(0.3)
    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    translation = np.array([0.5, -0.3, 0.2])
    return mean[:20] @ rotation + translation, rotation, translation


@pytest.mark.parametrize("normalize", [False, True])
def test_atlas_state_resumes_across_coordinate_frames(normalize):
    mean = rod()
    target, rotation, translation = fragment(mean)
    modes = (0.01 * np.sin(np.arange(mean.size) * 0.1)).reshape(-1, 1)
    common = dict(with_scale=False, adaptive_mixing=0.1, outlier_weight=0.05,
                  tolerance=0.0, parallel=False)
    start = dict(initial_rotation=rotation, initial_translation=translation,
                 sigma2=0.25, normalize=True)
    full = cpd.register_atlas(target, mean, modes, [1.0], max_iterations=12, **start, **common)
    first = cpd.register_atlas(target, mean, modes, [1.0], max_iterations=4, **start, **common)
    assert isinstance(first.state, cpd.AtlasState)
    resumed = cpd.register_atlas(target, mean, modes, [1.0], initial_state=first.state,
                                 max_iterations=8, normalize=normalize, **common)
    np.testing.assert_allclose(resumed.points, full.points, atol=1e-8)
    np.testing.assert_allclose(resumed.mixing_weights, full.mixing_weights, atol=1e-8)
    assert resumed.sigma2 == pytest.approx(full.sigma2, abs=1e-10)
    with pytest.raises(ValueError, match="conflict"):
        cpd.register_atlas(target, mean, modes, [1.0], initial_state=first.state,
                           initial_rotation=rotation, **common)


def test_pose_state_survives_final_atlas_registration():
    mean = rod()
    target, _, _ = fragment(mean)
    modes = np.zeros((mean.size, 1))
    common = dict(with_scale=False, adaptive_mixing=0.1, outlier_weight=0.05, parallel=False)
    init = cpd.pose_initialize(
        mean, target, modes, [1.0], rotation_count=9,
        coarse_source_count=60, coarse_target_count=20, coarse_rank=1,
        coarse_iterations=10, refine_count=6, refine_target_count=20,
        refine_iterations=30, translation_anchor_count=6, **common,
    )
    assert init.sigma2 == init.state.sigma2
    np.testing.assert_array_equal(init.mixing_weights, init.state.mixing_weights)
    fit = cpd.register_atlas(target, mean, modes, [1.0], initial_state=init.state,
                             normalize=True, max_iterations=100, tolerance=0.0, **common)
    assert np.sqrt(np.mean((fit.points[:20] - target) ** 2)) < 0.1


def conditional_reference(fit, target, mean, modes, eigenvalues, weights, prior, outlier_weight):
    """Independent dense GMM responsibilities and Gaussian conditioning."""
    posed = fit.reconstruct(mean, modes)
    distance2 = np.sum((posed[:, None, :] - target[None, :, :]) ** 2, axis=2)
    kernels = ((1.0 - outlier_weight) * weights[:, None]
               * np.exp(-distance2 / (2 * fit.sigma2))
               / (2 * np.pi * fit.sigma2) ** (mean.shape[1] / 2))
    posterior = kernels / (kernels.sum(axis=0) + outlier_weight * fit.state.outlier_density)
    mass = posterior.sum(axis=1)
    positions = np.divide(posterior @ target, mass[:, None],
                          out=np.zeros_like(mean), where=mass[:, None] > 0)
    residual = ((positions - fit.translation) @ fit.rotation.T / fit.scale - mean).ravel()
    sigma_eff2 = fit.sigma2 / fit.scale ** 2
    precision = np.diag(prior / np.asarray(eigenvalues))
    precision += modes.T @ (np.repeat(mass, mean.shape[1])[:, None] * modes) / sigma_eff2
    covariance = np.linalg.inv(precision)
    coefficients = covariance @ (modes.T @ (np.repeat(mass, mean.shape[1]) * residual)) / sigma_eff2
    return coefficients, covariance


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("sampling", ["same", "permuted", "denser"])
def test_adaptive_completion_matches_independent_conditioning(normalize, sampling):
    mean = rod(30)
    target = mean[:12] + [0.04, 0.03, -0.02]
    modes = np.zeros((mean.size, 1))
    modes[1::3, 0] = 0.1
    fit = cpd.register_atlas(target, mean, modes, [1.0], normalize=normalize,
                             optimize_similarity=False, with_scale=False,
                             adaptive_mixing=0.05, outlier_weight=0.2,
                             max_iterations=2, tolerance=0.0, parallel=False)
    assert np.ptp(fit.mixing_weights) > 1e-3
    assert fit.outlier_weight == 0.2
    destination = mean if sampling == "same" else mean[::-1].copy() if sampling == "permuted" else rod(71)
    destination_modes = np.zeros((destination.size, 1))
    destination_modes[1::3, 0] = 0.1
    if sampling == "same":
        weights = fit.mixing_weights
    else:
        nearest = np.argmin(np.sum((destination[:, None, :] - mean[None, :, :]) ** 2, axis=2), axis=1)
        weights = np.maximum(fit.mixing_weights[nearest], 1e-9 / len(mean))
        weights /= weights.sum()
    for override in [None, 0.0]:
        prior = 0.4
        post = fit.posterior(target, destination, destination_modes, [1.0],
                             prior_temperature=prior, visibility_floor=0.0,
                             estimate_discrepancy=False, outlier_weight=override)
        expected_mean, expected_cov = conditional_reference(
            fit, target, destination, destination_modes, [1.0], weights, prior,
            fit.outlier_weight if override is None else override,
        )
        assert post.noise_variance == fit.sigma2 / fit.scale ** 2
        np.testing.assert_allclose(post.coefficient_mean, expected_mean, atol=1e-9)
        np.testing.assert_allclose(post.coefficient_covariance, expected_cov, atol=1e-9)
    # A uniform-mixture calculation must give a different posterior on this fixture.
    uniform_mean, _ = conditional_reference(fit, target, destination, destination_modes,
                                             [1.0], np.full(len(destination), 1 / len(destination)),
                                             0.4, fit.outlier_weight)
    adaptive_mean, _ = conditional_reference(fit, target, destination, destination_modes,
                                              [1.0], weights, 0.4, fit.outlier_weight)
    assert np.max(np.abs(uniform_mean - adaptive_mean)) > 1e-5


def test_state_reorders_weights_and_pads_new_modes():
    mean = rod(30)
    modes = np.zeros((mean.size, 1))
    target = mean[:10]
    first = cpd.register_atlas(target, mean, modes, [1.0], adaptive_mixing=0.05,
                               with_scale=False, max_iterations=2, tolerance=0.0)
    resumed = cpd.register_atlas(target, mean[::-1], np.zeros((mean.size, 2)), [1.0, 0.5],
                                 initial_state=first.state, with_scale=False, max_iterations=0)
    np.testing.assert_allclose(resumed.mixing_weights, first.mixing_weights[::-1], atol=1e-12)
    np.testing.assert_array_equal(resumed.coefficients, [0.0, 0.0])
