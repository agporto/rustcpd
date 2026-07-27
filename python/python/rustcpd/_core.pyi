"""Type stubs for the compiled `rustcpd._core` extension."""

from typing import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

__version__: str

# Array-like accepted on input: any (N, D) NumPy array or nested sequence.
_ArrayLike = NDArray[np.float64] | Sequence[Sequence[float]]

class RigidResult:
    points: NDArray[np.float64]
    rotation: NDArray[np.float64]
    translation: NDArray[np.float64]
    scale: float
    sigma2: float
    iterations: int
    objective: float
    difference: float

class AffineResult:
    points: NDArray[np.float64]
    transform: NDArray[np.float64]
    translation: NDArray[np.float64]
    sigma2: float
    iterations: int
    objective: float
    difference: float

class DeformableResult:
    points: NDArray[np.float64]
    weights: NDArray[np.float64]
    kernel: NDArray[np.float64]
    low_rank_basis: NDArray[np.float64] | None
    low_rank_eigenvalues: NDArray[np.float64] | None
    source: NDArray[np.float64]
    beta: float
    sigma2: float
    iterations: int
    difference: float
    def transform(self, z: _ArrayLike) -> NDArray[np.float64]: ...

class AtlasResult:
    points: NDArray[np.float64]
    coefficients: NDArray[np.float64]
    rotation: NDArray[np.float64]
    scale: float
    translation: NDArray[np.float64]
    sigma2: float
    iterations: int
    difference: float
    landmark_rms: float
    def reconstruct(
        self, mean: _ArrayLike, modes: _ArrayLike
    ) -> NDArray[np.float64]: ...
    def apply_similarity(self, z: _ArrayLike) -> NDArray[np.float64]: ...
    def posterior(
        self,
        target: _ArrayLike,
        mean: _ArrayLike,
        modes: _ArrayLike,
        eigenvalues: Sequence[float],
        *,
        completeness: float | None = ...,
        visibility_floor: float = ...,
        prior_temperature: float = ...,
        outlier_weight: float = ...,
        estimate_discrepancy: bool = ...,
    ) -> ShapePosterior: ...

class PoseInitialization:
    coefficients: NDArray[np.float64]
    rotation: NDArray[np.float64]
    scale: float
    translation: NDArray[np.float64]
    score: float
    score_margin: float
    posterior_entropy: float
    effective_hypotheses: float
    hypotheses_evaluated: int
    hypotheses_refined: int

class Correspondences:
    matches: NDArray[np.int64]
    probability: NDArray[np.float64]
    posterior: NDArray[np.float64]

class ShapePosterior:
    @property
    def coefficient_mean(self) -> NDArray[np.float64]: ...
    @property
    def coefficient_covariance(self) -> NDArray[np.float64]: ...
    @property
    def noise_variance(self) -> float: ...
    @property
    def discrepancy_variance(self) -> float: ...
    def predict(self) -> NDArray[np.float64]: ...
    def predict_model_frame(self) -> NDArray[np.float64]: ...
    def predictive_variance(self) -> NDArray[np.float64]: ...
    def predictive_covariance(self, i: int) -> NDArray[np.float64]: ...
    def sample_coefficients(self, count: int, seed: int) -> NDArray[np.float64]: ...
    def sample_shapes(self, count: int, seed: int) -> list[NDArray[np.float64]]: ...

def complete_shape(
    mean: _ArrayLike,
    modes: _ArrayLike,
    eigenvalues: Sequence[float],
    residual: _ArrayLike,
    weight: Sequence[float],
    sigma_eff2: float,
    rotation: _ArrayLike,
    scale: float,
    translation: Sequence[float],
    *,
    prior_temperature: float = ...,
    estimate_discrepancy: bool = ...,
) -> ShapePosterior: ...
def mixture_sample_shapes(
    components: Sequence[tuple[float, ShapePosterior]],
    count: int,
    seed: int,
) -> list[NDArray[np.float64]]: ...

def register_rigid(
    target: _ArrayLike,
    source: _ArrayLike,
    *,
    scale: bool = ...,
    normalize: bool = ...,
    sigma2: float | None = ...,
    max_iterations: int = ...,
    tolerance: float = ...,
    outlier_weight: float = ...,
    k: int | None = ...,
    parallel: bool = ...,
    single_precision: bool = ...,
    callback: Callable[[dict], bool | None] | None = ...,
) -> RigidResult: ...
def register_affine(
    target: _ArrayLike,
    source: _ArrayLike,
    *,
    normalize: bool = ...,
    sigma2: float | None = ...,
    max_iterations: int = ...,
    tolerance: float = ...,
    outlier_weight: float = ...,
    k: int | None = ...,
    parallel: bool = ...,
    single_precision: bool = ...,
    callback: Callable[[dict], bool | None] | None = ...,
) -> AffineResult: ...
def register_deformable(
    target: _ArrayLike,
    source: _ArrayLike,
    *,
    alpha: float = ...,
    beta: float = ...,
    low_rank: int | None = ...,
    low_rank_method: str = ...,
    low_rank_tolerance: float = ...,
    normalize: bool = ...,
    constraints: Sequence[tuple[int, int]] | None = ...,
    constraint_error: float = ...,
    sigma2: float | None = ...,
    max_iterations: int = ...,
    tolerance: float = ...,
    outlier_weight: float = ...,
    k: int | None = ...,
    parallel: bool = ...,
    single_precision: bool = ...,
    callback: Callable[[dict], bool | None] | None = ...,
) -> DeformableResult: ...
def register_atlas(
    target: _ArrayLike,
    mean: _ArrayLike,
    modes: _ArrayLike,
    eigenvalues: Sequence[float],
    *,
    lambda_regularization: float = ...,
    normalize: bool = ...,
    optimize_similarity: bool = ...,
    with_scale: bool = ...,
    kdtree_radius_scale: float | None = ...,
    initial_coefficients: Sequence[float] | None = ...,
    initial_rotation: _ArrayLike | None = ...,
    initial_scale: float = ...,
    initial_translation: Sequence[float] | None = ...,
    sigma2: float | None = ...,
    landmark_indices: Sequence[int] | None = ...,
    landmark_targets: _ArrayLike | None = ...,
    landmark_weight: float = ...,
    landmark_error: float | None = ...,
    max_iterations: int = ...,
    tolerance: float = ...,
    outlier_weight: float = ...,
    k: int | None = ...,
    parallel: bool = ...,
    single_precision: bool = ...,
) -> AtlasResult: ...
def pose_initialize(
    source: _ArrayLike,
    target: _ArrayLike,
    modes: _ArrayLike,
    eigenvalues: Sequence[float],
    *,
    rotation_count: int = ...,
    coarse_source_count: int = ...,
    coarse_target_count: int = ...,
    coarse_rank: int = ...,
    coarse_iterations: int = ...,
    coarse_screen_iterations: int = ...,
    coarse_survivor_count: int = ...,
    coarse_score_mode: str = ...,
    refine_count: int = ...,
    refine_source_count: int | None = ...,
    refine_target_count: int = ...,
    refine_iterations: int = ...,
    lambda_regularization: float = ...,
    outlier_weight: float = ...,
    identity_prior_probability: float = ...,
    landmark_indices: Sequence[int] | None = ...,
    landmark_targets: _ArrayLike | None = ...,
    landmark_weight: float = ...,
    refine_landmark_weight: float = ...,
    with_scale: bool = ...,
    seed: int = ...,
    parallel: bool = ...,
    single_precision: bool = ...,
) -> PoseInitialization: ...
def correspondences(
    target: _ArrayLike,
    aligned_source: _ArrayLike,
    sigma2: float,
    *,
    outlier_weight: float = ...,
) -> Correspondences: ...
def gaussian_kernel(
    x: _ArrayLike, y: _ArrayLike, beta: float
) -> NDArray[np.float64]: ...
def initialize_sigma2(x: _ArrayLike, y: _ArrayLike) -> float: ...
