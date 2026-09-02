"""End-to-end tests for the Python bindings."""

from importlib.resources import files

import numpy as np
import pytest

import rustcpd as cpd


def test_package_declares_pep561_typing():
    assert files("rustcpd").joinpath("py.typed").is_file()


def cloud(count):
    z = np.arange(1, count + 1, dtype=np.float64)
    return np.column_stack(
        (
            np.sin(z * 0.37) * 1.7 + 0.01 * z,
            np.cos(z * 0.23) * 0.9,
            np.sin(z * 0.11) * np.cos(z * 0.07),
        )
    )


def rms(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def rotation_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def test_rigid_recovers_similarity():
    y = cloud(60)
    r = rotation_z(0.16)
    x = 1.06 * (y @ r) + np.array([0.18, -0.12, 0.08])
    result = cpd.register_rigid(x, y, tolerance=1e-8, max_iterations=200)
    assert rms(result.points, x) < 1e-8
    assert result.scale == pytest.approx(1.06, abs=1e-8)
    # The returned transform reproduces the aligned points.
    rebuilt = result.scale * (y @ result.rotation) + result.translation
    assert rms(rebuilt, result.points) < 1e-12
    assert result.iterations > 0


def test_rigid_sparse_matches_dense():
    y = cloud(40)
    x = y + np.array([0.04, -0.02, 0.01])
    dense = cpd.register_rigid(x, y, tolerance=0.0, max_iterations=6)
    sparse = cpd.register_rigid(x, y, tolerance=0.0, max_iterations=6, k=40)
    assert rms(dense.points, sparse.points) < 1e-11


def test_parallel_is_deterministic():
    y = cloud(80)
    x = y + np.array([0.04, -0.02, 0.01])
    kwargs = dict(tolerance=0.0, max_iterations=8)
    serial = cpd.register_rigid(x, y, parallel=False, **kwargs)
    parallel_a = cpd.register_rigid(x, y, parallel=True, **kwargs)
    parallel_b = cpd.register_rigid(x, y, parallel=True, **kwargs)
    assert np.array_equal(parallel_a.points, parallel_b.points)
    assert np.array_equal(serial.points, parallel_a.points)


def test_affine_recovers_transform():
    y = cloud(60)
    b = np.array([[1.03, 0.04, 0.0], [-0.025, 0.98, 0.01], [0.0, 0.02, 1.01]])
    x = y @ b + np.array([0.08, -0.05, 0.03])
    result = cpd.register_affine(x, y, tolerance=1e-8, max_iterations=250)
    assert rms(result.points, x) < 1e-7
    assert np.linalg.norm(result.transform - b) < 1e-6
    assert np.linalg.norm(result.transform - b.T) > 1e-2
    rebuilt = y @ result.transform + result.translation
    assert rms(rebuilt, result.points) < 1e-12


def test_deformable_full_rank_and_transform_identity():
    y = cloud(30)
    g = cpd.gaussian_kernel(y, y, 1.5)
    w = 0.004 * np.sin(0.41 * (np.arange(90).reshape(30, 3) + 1))
    x = y + g @ w
    result = cpd.register_deformable(
        x, y, beta=1.5, low_rank=None, tolerance=1e-7, max_iterations=200
    )
    assert rms(result.points, x) < 1e-6
    # points == y + kernel @ weights
    assert rms(y + result.kernel @ result.weights, result.points) < 1e-12


def test_deformable_constraints_pin_landmarks():
    y = cloud(50)
    x = y + np.array([0.3, -0.2, 0.1])
    result = cpd.register_deformable(
        x,
        y,
        low_rank=None,
        constraints=[(0, 0), (25, 25)],
        tolerance=1e-8,
        max_iterations=200,
    )
    for index in (0, 25):
        assert np.linalg.norm(result.points[index] - x[index]) < 1e-6


def test_deformable_low_rank_registers():
    y = cloud(90)
    g = cpd.gaussian_kernel(y, y, 2.0)
    w = 0.004 * np.sin(0.41 * (np.arange(270).reshape(90, 3) + 1))
    x = y + g @ w
    result = cpd.register_deformable(
        x, y, beta=2.0, low_rank=30, tolerance=1e-9, max_iterations=150
    )
    assert rms(result.points, x) < 5e-2


def test_deformable_pivoted_cholesky_matches_eigen():
    y = cloud(120)
    g = cpd.gaussian_kernel(y, y, 2.0)
    w = 0.004 * np.sin(0.41 * (np.arange(360).reshape(120, 3) + 1))
    x = y + g @ w
    kw = dict(beta=2.0, low_rank=60, tolerance=1e-9, max_iterations=150)
    eigen = cpd.register_deformable(x, y, low_rank_method="eigen", **kw)
    pivoted = cpd.register_deformable(x, y, low_rank_method="pivoted_cholesky", **kw)
    # Converged, the two low-rank builders agree to approximation accuracy.
    assert rms(pivoted.points, eigen.points) < 1e-3
    # Alias accepted.
    aliased = cpd.register_deformable(x, y, low_rank_method="cholesky", **kw)
    assert rms(aliased.points, pivoted.points) == 0.0


def test_rigid_normalize_recovers_transform_on_large_coordinates():
    base = cloud(60)
    y = base * 700.0 + np.array([60_000.0, -25_000.0, 8000.0])
    r = rotation_z(0.19)
    scale, t = 1.1, np.array([1500.0, -900.0, 400.0])
    x = scale * (y @ r) + t
    result = cpd.register_rigid(
        x, y, scale=True, normalize=True, tolerance=1e-10, max_iterations=300
    )
    rel = rms(result.points, x) / rms(x, y)
    assert rel < 1e-6, rel
    assert abs(result.scale - scale) < 1e-5
    # Returned transform reproduces the points from raw y.
    reapplied = result.scale * (y @ np.asarray(result.rotation)) + np.asarray(result.translation)
    assert rms(reapplied, result.points) / rms(x, y) < 1e-9


def test_deformable_normalize_transform_round_trips():
    base = cloud(90)
    y = base * 500.0 + np.array([20_000.0, 8000.0, -5000.0])
    g = cpd.gaussian_kernel(y, y, 1000.0)
    w = 0.002 * np.sin(0.41 * (np.arange(270).reshape(90, 3) + 1))
    x = y + g @ w
    result = cpd.register_deformable(
        x, y, beta=2.0, low_rank=None, normalize=True,
        tolerance=1e-10, max_iterations=200,
    )
    assert np.all(np.isfinite(result.points))
    # transform() on new points (here, the source) accounts for the frame.
    warped = result.transform(y)
    assert rms(warped, result.points) / rms(x, y) < 1e-9


def test_callback_observes_iterations():
    y = cloud(50)
    x = y + np.array([0.05, -0.03, 0.02])
    seen = []

    def cb(state):
        assert set(state) == {"iteration", "sigma2", "difference", "points"}
        assert state["points"].shape == (50, 3)
        seen.append(state["iteration"])
        return True

    result = cpd.register_rigid(x, y, tolerance=1e-9, max_iterations=40, callback=cb)
    assert seen == list(range(1, result.iterations + 1))


def test_callback_can_stop_early():
    y = cloud(50)
    x = y + np.array([0.05, -0.03, 0.02])

    def cb(state):
        return state["iteration"] < 4  # stop after the 4th iteration

    result = cpd.register_deformable(
        x, y, low_rank=None, tolerance=0.0, max_iterations=100, callback=cb
    )
    assert result.iterations == 4


def test_callback_exception_propagates():
    y = cloud(30)
    x = y + 0.01

    def cb(_state):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        cpd.register_rigid(x, y, max_iterations=10, callback=cb)


def test_callback_non_bool_return_raises():
    y = cloud(30)
    x = y + 0.01

    def cb(_state):
        return 0  # not None, not a bool: must raise, never silently continue

    with pytest.raises(TypeError, match="None or a bool"):
        cpd.register_rigid(x, y, max_iterations=10, callback=cb)


def test_callback_none_return_continues():
    y = cloud(30)
    x = y + 0.01
    seen = []

    def cb(state):
        seen.append(state["iteration"])  # no return -> None -> continue

    result = cpd.register_rigid(x, y, tolerance=1e-9, max_iterations=15, callback=cb)
    assert len(seen) == result.iterations


def test_deformable_low_rank_zero_raises():
    y = cloud(30)
    with pytest.raises(ValueError):
        cpd.register_deformable(y + 0.01, y, low_rank=0)


def test_deformable_low_rank_exposes_factors_not_dense_kernel():
    y = cloud(80)
    g = cpd.gaussian_kernel(y, y, 2.0)
    w = 0.004 * np.sin(0.41 * (np.arange(240).reshape(80, 3) + 1))
    x = y + g @ w
    low = cpd.register_deformable(
        x, y, beta=2.0, low_rank=40, low_rank_method="pivoted_cholesky",
        tolerance=1e-9, max_iterations=100,
    )
    # No dense kernel; factors present and orthonormal.
    assert low.kernel.shape == (0, 0)
    assert low.low_rank_basis is not None
    assert low.low_rank_eigenvalues is not None
    q = low.low_rank_basis
    assert q.shape[1] == low.low_rank_eigenvalues.shape[0]
    assert np.allclose(q.T @ q, np.eye(q.shape[1]), atol=1e-6)

    # Full-rank fit still returns the dense kernel and no factors.
    full = cpd.register_deformable(x, y, beta=2.0, low_rank=None, max_iterations=50)
    assert full.kernel.shape == (80, 80)
    assert full.low_rank_basis is None
    assert full.low_rank_eigenvalues is None


def test_deformable_unknown_low_rank_method_raises():
    y = cloud(40)
    with pytest.raises(ValueError):
        cpd.register_deformable(y + 0.01, y, low_rank=20, low_rank_method="bogus")


def test_atlas_recovers_coefficients():
    y = cloud(40)
    rank = 3
    modes = np.sin((np.arange(y.size)[:, None] + 1 + 7 * np.arange(rank)) * 0.19)
    modes += 0.3 * np.cos((np.arange(y.size)[:, None] + 3 * np.arange(rank) + 2) * 0.11)
    # Orthonormalize columns, then scale.
    q, _ = np.linalg.qr(modes)
    modes = q * np.array([0.5, 0.4, 0.3])
    truth = np.array([0.35, -0.2, 0.15])
    x = y + (modes @ truth).reshape(y.shape)
    result = cpd.register_atlas(
        x,
        y,
        modes,
        [0.5, 0.3, 0.2],
        lambda_regularization=0.001,
        optimize_similarity=False,
        tolerance=1e-5,
        max_iterations=200,
    )
    assert rms(result.points, x) < 1e-5
    assert np.allclose(result.coefficients, truth, atol=2e-3)


def test_atlas_adaptive_sparse_matches_dense():
    y = cloud(80)
    rank = 3
    modes = np.sin((np.arange(y.size)[:, None] + 1 + 7 * np.arange(rank)) * 0.19)
    modes += 0.3 * np.cos((np.arange(y.size)[:, None] + 3 * np.arange(rank) + 2) * 0.11)
    q, _ = np.linalg.qr(modes)
    modes = q * np.array([0.5, 0.4, 0.3])
    truth = np.array([0.3, -0.18, 0.12])
    x = y + (modes @ truth).reshape(y.shape)
    common = dict(
        lambda_regularization=0.001, optimize_similarity=False,
        tolerance=1e-6, max_iterations=200,
    )
    dense = cpd.register_atlas(x, y, modes, [0.5, 0.3, 0.2], **common)
    # Dense until sigma2 < (extent/10)^2, then k=20 sparse.
    adaptive = cpd.register_atlas(
        x, y, modes, [0.5, 0.3, 0.2], k=20, kdtree_radius_scale=10.0, **common
    )
    assert np.allclose(adaptive.coefficients, dense.coefficients, atol=1e-3)
    assert np.allclose(adaptive.coefficients, truth, atol=3e-3)


def test_pose_initialize_recovers_similarity():
    y = cloud(36)
    r = rotation_z(0.35)
    target = 1.1 * (y @ r) + np.array([0.2, -0.1, 0.05])
    modes = np.zeros((y.size, 1))
    init = cpd.pose_initialize(
        y,
        target,
        modes,
        [1.0],
        rotation_count=9,
        coarse_source_count=36,
        coarse_target_count=36,
        coarse_rank=1,
        coarse_iterations=5,
        refine_count=3,
        refine_target_count=36,
        refine_iterations=12,
        parallel=False,
    )
    points = init.scale * (y @ init.rotation) + init.translation
    assert rms(points, target) < 0.08
    assert init.hypotheses_evaluated == 9
    assert init.hypotheses_refined == 3


def test_pose_initialize_can_fix_residual_scale():
    y = cloud(36)
    r = rotation_z(0.25)
    target = y @ r + np.array([0.7, -0.4, 0.2])
    modes = np.zeros((y.size, 1))
    init = cpd.pose_initialize(
        y,
        target,
        modes,
        [1.0],
        rotation_count=1,
        coarse_source_count=36,
        coarse_target_count=36,
        coarse_rank=1,
        coarse_iterations=8,
        refine_count=1,
        refine_target_count=36,
        refine_iterations=20,
        with_scale=False,
        parallel=False,
    )
    points = y @ init.rotation + init.translation
    assert init.scale == 1.0
    assert rms(points, target) < 1e-4
    assert np.linalg.norm(init.translation) > 0.1


def test_pose_compatibility_wrapper_forwards_fixed_scale():
    y = cloud(24)
    target = y + np.array([0.35, -0.2, 0.1])
    modes = np.zeros((len(y), 3, 1))
    init = cpd.pose_marginalized_initialization(
        y,
        target,
        modes,
        [1.0],
        rotation_count=1,
        coarse_source_count=len(y),
        coarse_target_count=len(y),
        coarse_rank=1,
        coarse_iterations=4,
        coarse_screen_iterations=4,
        coarse_survivor_count=1,
        refine_count=1,
        refine_target_count=len(y),
        refine_iterations=6,
        with_scale=False,
        n_jobs=1,
    )
    assert init.scale == 1.0


def test_initialize_sigma2_matches_pairwise_definition():
    x, y = cloud(9), cloud(7)
    direct = np.mean(
        np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1)
    ) / 3.0
    assert cpd.initialize_sigma2(x, y) == pytest.approx(direct, abs=1e-12)


def test_gaussian_kernel_matches_numpy():
    x, y = cloud(8), cloud(11)
    beta = 1.5
    d2 = np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1)
    expected = np.exp(-d2 / (2 * beta * beta))
    assert np.allclose(cpd.gaussian_kernel(x, y, beta), expected, rtol=1e-14)


def test_accepts_lists_and_f32():
    y = cloud(20)
    x = y + 0.01
    as_list = cpd.register_rigid(x.tolist(), y.tolist(), max_iterations=5)
    as_f32 = cpd.register_rigid(
        x.astype(np.float32), y.astype(np.float32), max_iterations=5
    )
    assert as_list.points.shape == (20, 3)
    assert as_f32.points.shape == (20, 3)


def test_invalid_inputs_raise_value_error():
    y = cloud(10)
    with pytest.raises(ValueError):
        cpd.register_rigid(np.zeros((0, 3)), y)
    with pytest.raises(ValueError):
        cpd.register_rigid(np.zeros((10, 2)), y)
    bad = y.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        cpd.register_rigid(bad, y)
    with pytest.raises(ValueError):
        cpd.register_rigid(y, y, outlier_weight=1.0)
    with pytest.raises(ValueError):
        cpd.register_deformable(y, y, constraints=[(99, 0)])


def test_version_exposed():
    assert isinstance(cpd.__version__, str) and cpd.__version__


def test_single_precision_close_to_double():
    y = cloud(100)
    r = rotation_z(0.14)
    x = 1.05 * (y @ r) + np.array([0.15, -0.1, 0.05])
    double = cpd.register_rigid(x, y, tolerance=1e-8, max_iterations=200)
    single = cpd.register_rigid(
        x, y, tolerance=1e-8, max_iterations=200, single_precision=True
    )
    assert rms(single.points, double.points) < 1e-5
    # Deterministic: two f32 runs match exactly.
    again = cpd.register_rigid(
        x, y, tolerance=1e-8, max_iterations=200, single_precision=True
    )
    assert np.array_equal(single.points, again.points)


def test_deformable_transform_matches_points_and_warps_new():
    y = cloud(40)
    g = cpd.gaussian_kernel(y, y, 1.5)
    w = 0.004 * np.sin(0.41 * (np.arange(120).reshape(40, 3) + 1))
    x = y + g @ w
    result = cpd.register_deformable(
        x, y, beta=1.5, low_rank=None, tolerance=1e-8, max_iterations=200
    )
    # Evaluating the field at training points reproduces `points`.
    assert rms(result.transform(y), result.points) < 1e-12
    # New points (midpoints) warp consistently with the deformed midpoints.
    mid = 0.5 * (y[:-1] + y[1:])
    warped = result.transform(mid)
    deformed_mid = 0.5 * (result.points[:-1] + result.points[1:])
    assert rms(warped, deformed_mid) < 5e-3
    assert result.beta == 1.5
    assert result.source.shape == (40, 3)


def test_deformable_transform_wrong_dimension_raises():
    y = cloud(20)
    result = cpd.register_deformable(y + 0.01, y, low_rank=None, max_iterations=10)
    with pytest.raises(ValueError):
        result.transform(np.zeros((5, 2)))


def test_correspondences_recover_matching():
    y = cloud(30)
    x = y + np.array([0.03, -0.02, 0.01])
    result = cpd.register_rigid(
        x, y, scale=False, tolerance=1e-10, max_iterations=200
    )
    match = cpd.correspondences(x, result.points, result.sigma2)
    assert match.matches.shape == (30,)
    assert match.matches.dtype == np.int64
    correct = int(np.sum(match.matches == np.arange(30)))
    assert correct >= 28
    # Columns are normalized over sources (plus outlier mass) -> <= 1.
    assert np.all(match.posterior.sum(axis=0) <= 1.0 + 1e-9)
    assert np.all((match.probability >= 0) & (match.probability <= 1))


def test_correspondences_validate():
    y = cloud(10)
    with pytest.raises(ValueError):
        cpd.correspondences(y, y, -1.0)
    with pytest.raises(ValueError):
        cpd.correspondences(y, y, 1.0, outlier_weight=1.0)


def _orthonormal_modes(n_points, rank):
    m = np.sin((np.arange(n_points * 3)[:, None] + 1 + 7 * np.arange(rank)) * 0.19)
    m += 0.3 * np.cos((np.arange(n_points * 3)[:, None] + 3 * np.arange(rank) + 2) * 0.11)
    q, _ = np.linalg.qr(m)
    return q * np.array([0.5, 0.4, 0.3][:rank])


def test_atlas_reconstruct_reproduces_points():
    full = cloud(60)
    rank = 3
    modes = _orthonormal_modes(len(full), rank)
    truth = np.array([0.3, -0.18, 0.12])
    deformed = full + (modes @ truth).reshape(full.shape)
    r = rotation_z(0.25)
    target = 1.2 * (deformed @ r) + np.array([0.3, -0.2, 0.1])
    for normalize in (False, True):
        result = cpd.register_atlas(
            target, full, modes, [0.5, 0.4, 0.3],
            lambda_regularization=1e-4, normalize=normalize,
            tolerance=1e-9, max_iterations=300,
        )
        rebuilt = result.reconstruct(full, modes)
        assert rms(rebuilt, result.points) < 1e-9
        # apply_similarity on the deformed model == reconstruct.
        deformed_fit = full + (modes @ result.coefficients).reshape(full.shape)
        assert rms(result.apply_similarity(deformed_fit), result.points) < 1e-9
        expected = result.scale * (deformed_fit @ result.rotation) + result.translation
        assert rms(result.apply_similarity(deformed_fit), expected) < 1e-12


def test_atlas_reconstruct_upsamples_to_higher_resolution():
    # Register on a subsample, reconstruct on the full model.
    rank = 3
    full = cloud(80)
    modes_full = _orthonormal_modes(len(full), rank)
    truth = np.array([0.25, -0.15, 0.1])
    deformed = full + (modes_full @ truth).reshape(full.shape)
    target_full = 1.1 * deformed + np.array([0.2, -0.1, 0.05])
    idx = np.arange(0, 80, 2)  # 40-point subsample
    sub = full[idx]
    modes_sub = modes_full.reshape(80, 3, rank)[idx].reshape(len(idx) * 3, rank)
    target_sub = target_full[idx]
    result = cpd.register_atlas(
        target_sub, sub, modes_sub, [0.5, 0.4, 0.3],
        lambda_regularization=1e-4, optimize_similarity=True,
        tolerance=1e-9, max_iterations=300,
    )
    dense = result.reconstruct(full, modes_full)
    assert dense.shape == (80, 3)
    # The dense reconstruction should track the true full-resolution target.
    assert rms(dense, target_full) < 5e-2


def test_atlas_reconstruct_validates():
    y = cloud(30)
    modes = _orthonormal_modes(len(y), 2)[:, :2]
    result = cpd.register_atlas(
        y, y, modes, [0.5, 0.3], optimize_similarity=False, max_iterations=5
    )
    with pytest.raises(ValueError):
        result.reconstruct(y, np.zeros((y.size, 3)))  # wrong rank
    with pytest.raises(ValueError):
        result.apply_similarity(np.zeros((5, 2)))  # wrong dim


def test_register_atlas_landmarks_improve_underregularized_fit():
    # Under heavy regularization the plain fit under-shoots the true shape;
    # the landmark data term pulls the anchored vertices back onto target.
    y = cloud(40)
    rank = 2
    modes = _orthonormal_modes(len(y), rank)[:, :rank]
    truth = np.array([0.9, -0.7])
    target = y + (modes @ truth).reshape(y.shape)
    idx = [1, 12, 27, 33]
    # A short, fixed EM budget under heavy regularization: the plain fit has
    # not yet converged (the Mahalanobis prior still bites), while the landmark
    # data term drives the anchored vertices onto their targets almost at once.
    common = dict(
        optimize_similarity=False, with_scale=False,
        lambda_regularization=6.0, max_iterations=6, tolerance=0.0,
    )
    plain = cpd.register_atlas(target, y, modes, [1.0, 1.0], **common)
    anchored = cpd.register_atlas(
        target, y, modes, [1.0, 1.0],
        landmark_indices=idx, landmark_targets=target[idx], landmark_weight=40.0,
        **common,
    )

    def resid(res):
        recon = np.asarray(res.reconstruct(y, modes))
        return rms(recon[idx], target[idx])

    assert resid(anchored) < 0.25 * resid(plain)


def test_register_atlas_landmark_weight_zero_is_noop():
    y = cloud(30)
    target = y + 0.05
    modes = _orthonormal_modes(len(y), 2)[:, :2]
    common = dict(optimize_similarity=False, lambda_regularization=1.0, max_iterations=120)
    base = cpd.register_atlas(target, y, modes, [1.0, 1.0], **common)
    disabled = cpd.register_atlas(
        target, y, modes, [1.0, 1.0],
        landmark_indices=[0, 5], landmark_targets=np.zeros((2, 3)),
        landmark_weight=0.0, **common,
    )
    assert np.allclose(base.coefficients, disabled.coefficients, atol=1e-12)


def test_register_atlas_landmark_validation():
    y = cloud(20)
    modes = _orthonormal_modes(len(y), 2)[:, :2]
    with pytest.raises(ValueError):  # targets missing
        cpd.register_atlas(y, y, modes, [1.0, 1.0], landmark_indices=[0], landmark_weight=5.0)
    with pytest.raises(ValueError):  # length mismatch
        cpd.register_atlas(
            y, y, modes, [1.0, 1.0],
            landmark_indices=[0, 1], landmark_targets=y[:1], landmark_weight=5.0,
        )


def test_pose_initialize_landmarks_guide_basin():
    y = cloud(30)
    r = rotation_z(0.7)
    target = (y @ r) + np.array([0.4, -0.25, 0.15])
    modes = np.zeros((y.size, 1))
    idx = [0, 8, 17, 25]
    init = cpd.pose_initialize(
        y, target, modes, [1.0],
        rotation_count=25, coarse_source_count=30, coarse_target_count=30,
        coarse_rank=1, coarse_iterations=6, coarse_screen_iterations=6,
        coarse_survivor_count=25, refine_count=4, refine_target_count=30,
        refine_iterations=20, with_scale=False,
        landmark_indices=idx, landmark_targets=target[idx], landmark_weight=20.0,
        parallel=False,
    )
    fitted = init.scale * (y[idx] @ init.rotation) + init.translation
    assert rms(fitted, target[idx]) < 0.08


def test_pose_marginalized_wrapper_forwards_landmarks():
    # The compatibility wrapper must forward keypoints to pose_initialize.
    y = cloud(30)
    r = rotation_z(0.7)
    target = (y @ r) + np.array([0.4, -0.25, 0.15])
    modes = np.zeros((y.size, 1))
    idx = [0, 8, 17, 25]
    common = dict(
        rotation_count=25, coarse_source_count=30, coarse_target_count=30,
        coarse_rank=1, coarse_iterations=6, coarse_screen_iterations=6,
        coarse_survivor_count=25, refine_count=4, refine_target_count=30,
        refine_iterations=20, with_scale=False, n_jobs=1,
    )
    guided = cpd.pose_marginalized_initialization(
        y, target, modes, [1.0],
        landmark_indices=idx, landmark_targets=target[idx],
        landmark_sigma=0.02, refine_landmark_sigma=0.02, **common,
    )
    fitted = guided.scale * (y[idx] @ guided.rotation) + guided.translation
    assert rms(fitted, target[idx]) < 0.08
    # blind wrapper (no landmarks) is unaffected and still returns a proper pose
    blind = cpd.pose_marginalized_initialization(y, target, modes, [1.0], **common)
    assert abs(np.linalg.det(np.asarray(blind.rotation)) - 1.0) < 1e-6


def test_pose_initialize_landmark_validation():
    y = cloud(20)
    modes = np.zeros((y.size, 1))
    with pytest.raises(ValueError):
        cpd.pose_initialize(
            y, y, modes, [1.0],
            landmark_indices=[0, 1], landmark_targets=y[:1], landmark_weight=5.0,
        )


def test_pose_confidence_calibrator_maps_sigma2_to_probability():
    from rustcpd import calibration as cal
    rng = np.random.default_rng(0)
    # correct fits have small residual variance, failures large (well separated)
    good = rng.uniform(1e-5, 1e-4, 60)
    bad = rng.uniform(1e-2, 1e-1, 40)
    sigma2 = np.concatenate([good, bad])
    correct = np.concatenate([np.ones(60), np.zeros(40)]).astype(bool)

    auc = cal.failure_detection_auc(sigma2, correct)
    assert auc > 0.99  # sigma2 separates the two classes here

    c = cal.PoseConfidenceCalibrator.fit(sigma2, correct)
    assert c.slope < 0  # lower sigma2 -> higher confidence
    assert c.probability(good.mean()) > 0.9
    assert c.probability(bad.mean()) < 0.1
    # monotone decreasing in sigma2
    grid = np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
    p = c.probability(grid)
    assert np.all(np.diff(p) < 0)
    trust = c.trust(sigma2)
    assert trust[:60].mean() > 0.95 and trust[60:].mean() < 0.05


def test_pose_confidence_calibrator_validates():
    from rustcpd import calibration as cal
    with pytest.raises(ValueError):  # one class only
        cal.PoseConfidenceCalibrator.fit(np.array([1e-4, 2e-4]), np.array([True, True]))
    with pytest.raises(ValueError):  # non-positive sigma2
        cal.PoseConfidenceCalibrator.fit(np.array([1e-4, 0.0]), np.array([True, False]))
    with pytest.raises(ValueError):  # AUC needs both classes
        cal.failure_detection_auc(np.array([1e-4, 2e-4]), np.array([True, True]))


def test_pose_initialize_anchored_refinement():
    y = cloud(40)
    r = rotation_z(0.6)
    target = (y @ r) + np.array([0.3, -0.2, 0.1])
    modes = np.zeros((y.size, 1))
    idx = [0, 9, 18, 27, 36]
    common = dict(
        rotation_count=25, coarse_source_count=40, coarse_target_count=40,
        coarse_rank=1, coarse_iterations=6, coarse_screen_iterations=6,
        coarse_survivor_count=25, refine_count=4, refine_source_count=15,
        refine_target_count=40, refine_iterations=20, with_scale=False,
        landmark_indices=idx, landmark_targets=target[idx], landmark_weight=15.0,
        parallel=False,
    )
    anchored = cpd.pose_initialize(y, target, modes, [1.0], refine_landmark_weight=25.0, **common)
    fitted = anchored.scale * (y[idx] @ anchored.rotation) + anchored.translation
    assert rms(fitted, target[idx]) < 0.05
    # default (0) is a no-op vs explicitly passing 0
    a = cpd.pose_initialize(y, target, modes, [1.0], **common)
    b = cpd.pose_initialize(y, target, modes, [1.0], refine_landmark_weight=0.0, **common)
    assert np.allclose(np.asarray(a.rotation), np.asarray(b.rotation))
    with pytest.raises(ValueError):
        cpd.pose_initialize(y, target, modes, [1.0], refine_landmark_weight=-1.0, **common)


def test_register_atlas_landmark_sigma_mode():
    y = cloud(40)
    r = rotation_z(0.3)
    # rigid target plus small non-model surface noise
    target = (y @ r) + np.array([0.4, -0.2, 0.1]) + 0.01 * np.sin(np.arange(y.size).reshape(y.shape))
    rank = 2
    modes = _orthonormal_modes(len(y), rank)[:, :rank]
    common = dict(optimize_similarity=True, with_scale=False, lambda_regularization=0.5,
                  max_iterations=200, tolerance=1e-9)
    base = cpd.register_atlas(target, y, modes, [1.0, 1.0], **common)
    assert np.isnan(base.landmark_rms)  # no landmarks -> NaN
    idx = [1, 12, 27]
    tgt = np.asarray(base.points)[idx]  # consistent landmarks (already satisfied)
    principled = cpd.register_atlas(target, y, modes, [1.0, 1.0],
                                    landmark_indices=idx, landmark_targets=tgt,
                                    landmark_sigma=1e-2, **common)
    heavy = cpd.register_atlas(target, y, modes, [1.0, 1.0],
                               landmark_indices=idx, landmark_targets=tgt,
                               landmark_weight=200.0, **common)
    assert principled.landmark_rms < 0.02
    # principled surface sigma2 ~ unchanged; heuristic diluted by landmark mass
    assert abs(principled.sigma2 - base.sigma2) < 0.1 * base.sigma2
    assert heavy.sigma2 < 0.5 * base.sigma2
    # tighter variance anchors harder (smaller landmark residual) on a divergent case
    tgt2 = target[idx]
    loose = cpd.register_atlas(target, y, modes, [1.0, 1.0], landmark_indices=idx,
                               landmark_targets=tgt2, landmark_sigma=0.2, **common)
    tight = cpd.register_atlas(target, y, modes, [1.0, 1.0], landmark_indices=idx,
                               landmark_targets=tgt2, landmark_sigma=0.02, **common)
    assert tight.landmark_rms < loose.landmark_rms


def test_register_atlas_landmark_sigma_validation():
    y = cloud(20)
    modes = _orthonormal_modes(len(y), 2)[:, :2]
    with pytest.raises(ValueError):
        cpd.register_atlas(y, y, modes, [1.0, 1.0], landmark_indices=[0],
                           landmark_targets=y[:1], landmark_sigma=-1.0)


def test_pose_initialize_landmark_sigma_scoring_and_refine():
    y = cloud(30)
    r = rotation_z(0.6)
    target = (y @ r) + np.array([0.4, -0.2, 0.1])
    modes = np.zeros((y.size, 1))
    idx = [0, 9, 18, 27]
    common = dict(
        rotation_count=25, coarse_source_count=30, coarse_target_count=30,
        coarse_rank=1, coarse_iterations=6, coarse_screen_iterations=6,
        coarse_survivor_count=25, refine_count=4, refine_source_count=15,
        refine_target_count=30, refine_iterations=20, with_scale=False,
        landmark_indices=idx, landmark_targets=target[idx], parallel=False,
    )
    # fixed-variance scoring + fixed-variance refinement (fully principled pose)
    init = cpd.pose_initialize(y, target, modes, [1.0], landmark_sigma=1e-2,
                               refine_landmark_sigma=1e-2, **common)
    fitted = init.scale * (y[idx] @ init.rotation) + init.translation
    assert rms(fitted, target[idx]) < 0.05
    # scoring-only fixed variance still returns a proper pose
    scored = cpd.pose_initialize(y, target, modes, [1.0], landmark_sigma=1e-2, **common)
    assert abs(np.linalg.det(np.asarray(scored.rotation)) - 1.0) < 1e-6
    with pytest.raises(ValueError):
        cpd.pose_initialize(y, target, modes, [1.0], landmark_sigma=-1.0, **common)
    with pytest.raises(ValueError):
        cpd.pose_initialize(y, target, modes, [1.0], refine_landmark_sigma=-1.0, **common)


def test_register_atlas_landmark_sigma_is_scale_equivariant():
    # landmark_sigma is a standard deviation, so scaling the whole problem by c
    # and the std by c must recover identical coefficients/rotation, with
    # translation scaling by c. A unit-free weight would not do this.
    y = cloud(40)
    r = rotation_z(0.3)
    rank = 2
    modes = _orthonormal_modes(len(y), rank)[:, :rank]
    truth = np.array([0.6, -0.4])
    shape = y + (modes @ truth).reshape(y.shape)
    target = shape @ r
    idx = [1, 12, 27]
    common = dict(optimize_similarity=True, with_scale=False,
                  lambda_regularization=0.3, max_iterations=300, tolerance=1e-11,
                  normalize=False)
    unit = cpd.register_atlas(target, y, modes, [1.0, 1.0],
                              landmark_indices=idx, landmark_targets=target[idx],
                              landmark_sigma=0.05, **common)
    c = 2.5
    scaled = cpd.register_atlas(target * c, y * c, modes * c, [1.0, 1.0],
                                landmark_indices=idx, landmark_targets=target[idx] * c,
                                landmark_sigma=0.05 * c, **common)
    assert np.allclose(unit.coefficients, scaled.coefficients, atol=1e-6)
    assert np.allclose(np.asarray(unit.rotation), np.asarray(scaled.rotation), atol=1e-6)
    assert np.allclose(np.asarray(unit.translation) * c, np.asarray(scaled.translation),
                       atol=1e-6 * c)
    assert abs(unit.landmark_rms * c - scaled.landmark_rms) < 1e-6 * c


def test_register_atlas_landmark_sigma_takes_precedence_over_weight():
    # Passing BOTH landmark_sigma and landmark_weight behaves like sigma alone:
    # the fixed-variance term wins and the weight is ignored (no silent blend).
    y = cloud(40)
    rank = 2
    modes = _orthonormal_modes(len(y), rank)[:, :rank]
    truth = np.array([0.7, -0.5])
    target = y + (modes @ truth).reshape(y.shape)
    idx = [1, 12, 27]
    tgt = target[idx] + 0.05  # inconsistent, so the two branches truly differ
    common = dict(optimize_similarity=False, lambda_regularization=6.0,
                  max_iterations=6, tolerance=0.0)
    sigma_only = cpd.register_atlas(target, y, modes, [1.0, 1.0], landmark_indices=idx,
                                    landmark_targets=tgt, landmark_sigma=0.02, **common)
    both = cpd.register_atlas(target, y, modes, [1.0, 1.0], landmark_indices=idx,
                              landmark_targets=tgt, landmark_sigma=0.02,
                              landmark_weight=123.0, **common)
    weight_only = cpd.register_atlas(target, y, modes, [1.0, 1.0], landmark_indices=idx,
                                     landmark_targets=tgt, landmark_weight=123.0, **common)
    assert np.allclose(sigma_only.coefficients, both.coefficients, atol=1e-12)
    # and the fixed-variance branch is genuinely different from weight-only
    assert not np.allclose(sigma_only.coefficients, weight_only.coefficients, atol=1e-6)


def tapered_rod(count=60):
    """Thin at x=0, thick at x=4, twisted: every segment is unique."""
    z = np.linspace(0.0, 1.0, count)
    radius = 0.1 + 0.6 * z
    return np.column_stack((4.0 * z, radius * np.sin(7 * z), radius * np.cos(7 * z)))


def test_pose_initialize_seeds_translations_for_a_displaced_fragment():
    model = tapered_rod()
    modes = np.zeros((model.size, 1))
    r = rotation_z(0.3)
    offset = np.array([0.5, -0.3, 0.2])
    target = model[:20] @ r + offset  # proximal third, displaced
    common = dict(
        rotation_count=9,
        coarse_source_count=60,
        coarse_target_count=20,
        coarse_rank=1,
        coarse_iterations=10,
        refine_count=6,
        refine_target_count=20,
        refine_iterations=30,
        with_scale=False,
        parallel=False,
    )
    init = cpd.pose_initialize(
        model,
        target,
        modes,
        [1.0],
        translation_anchor_count=6,
        adaptive_mixing=1.0,
        initial_sigma2=0.2,
        **common,
    )
    assert init.translation_anchors_used > 1
    assert init.hypotheses_evaluated == 9 * init.translation_anchors_used
    assert 1 <= init.distinct_hypotheses <= init.hypotheses_refined
    assert init.winner_support >= 1
    posed = init.scale * (model[:20] @ init.rotation) + init.translation
    assert rms(posed, target) < 0.1

    # A complete target never seeds, whatever the anchor count.
    full = cpd.pose_initialize(
        model,
        model @ r + offset,
        modes,
        [1.0],
        translation_anchor_count=6,
        **{**common, "coarse_target_count": 60, "refine_target_count": 60},
    )
    assert full.translation_anchors_used == 1
    assert full.hypotheses_evaluated == 9

    # The compatibility wrapper forwards the new options.
    wrapped = cpd.pose_marginalized_initialization(
        model,
        target,
        modes,
        [1.0],
        translation_anchor_count=6,
        with_scale=False,
        rotation_count=9,
        coarse_source_count=60,
        coarse_target_count=20,
        coarse_rank=1,
        coarse_iterations=10,
        refine_count=6,
        refine_target_count=20,
        refine_iterations=30,
    )
    assert wrapped.translation_anchors_used > 1


def test_register_atlas_adaptive_mixing_and_scale_bounds():
    model = tapered_rod()
    modes = np.zeros((model.size, 1))
    target = model[:20]  # fragment already in place
    plain = cpd.register_atlas(
        target, model, modes, [1.0], optimize_similarity=False, max_iterations=15, tolerance=0.0
    )
    assert plain.mixing_weights is None
    adaptive = cpd.register_atlas(
        target,
        model,
        modes,
        [1.0],
        optimize_similarity=False,
        adaptive_mixing=0.01,
        outlier_weight=0.05,
        max_iterations=15,
        tolerance=0.0,
    )
    pi = adaptive.mixing_weights
    assert pi.shape == (60,)
    assert abs(pi.sum() - 1.0) < 1e-9
    assert pi[:20].sum() > 0.9 and pi[20:].sum() < 0.1

    bounded = cpd.register_atlas(
        model[40:],
        model,
        modes,
        [1.0],
        with_scale=True,
        initial_scale=0.3,
        scale_bounds=(0.9, 1.1),
        max_iterations=30,
        tolerance=0.0,
    )
    assert 0.9 <= bounded.scale <= 1.1
    with pytest.raises(ValueError):
        cpd.register_atlas(target, model, modes, [1.0], scale_bounds=(1.2, 0.8))
    with pytest.raises(ValueError):
        cpd.register_atlas(target, model, modes, [1.0], adaptive_mixing=-1.0)
