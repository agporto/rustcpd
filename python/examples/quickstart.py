"""Quickstart for the rustcpd Python bindings.

Builds two synthetic 3-D clouds related by a known transform, recovers it
with rigid CPD, reads off correspondences, then fits a deformable warp and
applies it to new points.

Run with:  python examples/quickstart.py
"""

import numpy as np

import rustcpd as cpd


def synthetic_cloud(count: int) -> np.ndarray:
    z = np.arange(1, count + 1, dtype=np.float64)
    return np.column_stack(
        (
            np.sin(z * 0.37) * 1.7 + 0.01 * z,
            np.cos(z * 0.23) * 0.9,
            np.sin(z * 0.11) * np.cos(z * 0.07),
        )
    )


def rms(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main() -> None:
    source = synthetic_cloud(200)

    # --- Rigid: recover a known rotation + scale + translation ----------
    angle = 0.3
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    target = 1.15 * (source @ rotation) + np.array([0.4, -0.2, 0.1])

    rigid = cpd.register_rigid(target, source)
    print(
        f"rigid: recovered scale = {rigid.scale:.4f} (truth 1.15), "
        f"fit rms = {rms(rigid.points, target):.2e}, {rigid.iterations} iters"
    )

    # Soft correspondences: which target point each source point matched.
    match = cpd.correspondences(target, rigid.points, rigid.sigma2)
    self_matched = int(np.sum(match.matches == np.arange(len(source))))
    print(f"correspondences: {self_matched}/{len(source)} matched to their true target")

    # --- Deformable: fit a smooth warp, then apply it to new points -----
    beta = 2.0
    g = cpd.gaussian_kernel(source, source, beta)
    control = 0.02 * np.sin(0.41 * (np.arange(source.size).reshape(source.shape) + 1))
    warped_target = source + g @ control

    fit = cpd.register_deformable(warped_target, source, beta=beta, low_rank=None)
    print(f"deformable: fit rms = {rms(fit.points, warped_target):.2e}, {fit.iterations} iters")

    # Evaluate the learned field at brand-new points (cloud midpoints).
    new_points = 0.5 * (source[:-1] + source[1:])
    warped_new = fit.transform(new_points)
    print(f"transform: warped {len(warped_new)} new points not seen during registration")


if __name__ == "__main__":
    main()
