"""Coherent Point Drift registration with a fast, deterministic Rust core.

Registers a moving ``source`` point cloud onto a fixed ``target`` cloud.
All functions accept ``(N, D)`` NumPy arrays (``float64``; other dtypes and
lists are converted) and release the GIL while the Rust core runs.

Quick start::

    import numpy as np
    import rustcpd as cpd

    result = cpd.register_rigid(target, source)   # target/source: (N, 3)
    aligned = result.points                        # (M, 3) ndarray
    R, t, s = result.rotation, result.translation, result.scale

Available registrations: :func:`register_rigid`, :func:`register_affine`,
:func:`register_deformable` (with optional landmark ``constraints`` and a
``low_rank`` kernel approximation), :func:`register_atlas` (statistical
shape models), and :func:`pose_initialize` (global pose search). Helpers:
:func:`gaussian_kernel` and :func:`initialize_sigma2`.

Common keyword arguments (mirroring the Rust ``EmConfig``): ``sigma2``,
``max_iterations``, ``tolerance``, ``outlier_weight``, ``k`` (k-nearest
-neighbor sparse E-step; ``None`` = exact), and ``parallel`` (results are
bitwise-identical to serial execution).
"""

from ._core import (
    AffineResult,
    AtlasResult,
    Correspondences,
    DeformableResult,
    PoseInitialization,
    RigidResult,
    ShapePosterior,
    __version__,
    complete_shape,
    correspondences,
    gaussian_kernel,
    initialize_sigma2,
    mixture_sample_shapes,
    pose_initialize,
    register_affine,
    register_atlas,
    register_deformable,
    register_rigid,
)
from . import calibration


def pose_marginalized_initialization(
    source,
    target,
    modes,
    eigenvalues,
    *,
    rotation_count=193,
    coarse_source_count=400,
    coarse_target_count=400,
    coarse_rank=12,
    coarse_iterations=8,
    coarse_screen_iterations=8,
    coarse_survivor_count=193,
    coarse_score_mode="trajectory",
    refine_count=12,
    refine_source_count=None,
    refine_target_count=1600,
    refine_iterations=30,
    lambda_reg=0.1,
    outlier_weight=0.05,
    identity_prior_probability=0.2,
    landmark_indices=None,
    landmark_targets=None,
    landmark_weight=0.0,
    landmark_sigma=None,
    refine_landmark_weight=0.0,
    refine_landmark_sigma=None,
    with_scale=True,
    seed=0,
    n_jobs=1,
):
    """Drop-in-compatible wrapper over :func:`pose_initialize`.

    Mirrors the reference ``pose_marginalized_initialization`` signature so
    callers written against it work unchanged. Translates the naming
    differences (``lambda_reg`` -> ``lambda_regularization``) and maps
    ``n_jobs`` onto the Rust core's deterministic parallel flag (``n_jobs == 1``
    runs serially; any other value, including ``-1``, runs in parallel). Set
    ``with_scale=False`` for pre-scaled fragments to keep residual scale fixed
    while rotation and translation remain optimized. Accepts ``modes`` as
    ``(M, 3, K)`` or ``(3M, K)``.

    Anchored keypoints are forwarded too: pass ``landmark_indices`` /
    ``landmark_targets`` and set the strength with ``landmark_sigma`` (a physical
    localization std, preferred) or the heuristic ``landmark_weight``; the
    ``refine_landmark_*`` pair anchors the refinement EM. All default off, so
    existing callers are unaffected.
    """
    import numpy as _np

    modes = _np.asarray(modes, dtype=_np.float64)
    if modes.ndim == 3:
        modes = modes.reshape(_np.asarray(source).size, modes.shape[2])
    return pose_initialize(
        source,
        target,
        modes,
        eigenvalues,
        rotation_count=rotation_count,
        coarse_source_count=coarse_source_count,
        coarse_target_count=coarse_target_count,
        coarse_rank=coarse_rank,
        coarse_iterations=coarse_iterations,
        coarse_screen_iterations=coarse_screen_iterations,
        coarse_survivor_count=coarse_survivor_count,
        coarse_score_mode=coarse_score_mode,
        refine_count=refine_count,
        refine_source_count=refine_source_count,
        refine_target_count=refine_target_count,
        refine_iterations=refine_iterations,
        lambda_regularization=lambda_reg,
        outlier_weight=outlier_weight,
        identity_prior_probability=identity_prior_probability,
        landmark_indices=landmark_indices,
        landmark_targets=landmark_targets,
        landmark_weight=landmark_weight,
        landmark_sigma=landmark_sigma,
        refine_landmark_weight=refine_landmark_weight,
        refine_landmark_sigma=refine_landmark_sigma,
        with_scale=with_scale,
        seed=seed,
        parallel=(n_jobs != 1),
    )


__all__ = [
    "AffineResult",
    "AtlasResult",
    "Correspondences",
    "DeformableResult",
    "PoseInitialization",
    "RigidResult",
    "ShapePosterior",
    "__version__",
    "calibration",
    "complete_shape",
    "correspondences",
    "mixture_sample_shapes",
    "gaussian_kernel",
    "initialize_sigma2",
    "pose_initialize",
    "pose_marginalized_initialization",
    "register_affine",
    "register_atlas",
    "register_deformable",
    "register_rigid",
]
