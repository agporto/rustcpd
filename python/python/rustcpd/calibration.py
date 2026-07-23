"""Split-conformal calibration for shape-completion uncertainty.

The predictive variances from :class:`ShapePosterior` are *model-based*:
they are only as trustworthy as the linear-Gaussian assumptions behind the
atlas. This module makes them **defensible** by calibrating against held-out
data, so a stated ``1 - alpha`` interval actually contains the truth at that
rate.

The method is split (inductive) conformal prediction. For each calibration
example you need three per-point quantities on the *held-out* (masked)
points: the actual Euclidean error between the completion and the truth, and
the model's predicted total variance at that point. The nonconformity score
is ``error / sqrt(variance)``; its finite-sample ``1 - alpha`` quantile is a
multiplier ``q`` such that the calibrated radius ``q * sqrt(variance)``
achieves nominal marginal coverage on fresh data.

Typical use::

    cal = ConformalCalibrator.fit(errors, variances, alpha=0.1)
    radius = cal.interval_radius(new_variances)   # 90% per-point radius
    covered = cal.covers(new_errors, new_variances)

The primitives (:func:`nonconformity_scores`, :func:`conformal_quantile`,
:func:`empirical_coverage`) are exposed for custom workflows. A driver that
wires an atlas registration + completion into this loop is intentionally left
to the caller, since the model-point/observation correspondence is
application specific; :func:`calibrate_completion` shows the shape of it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

_TINY = 1e-12


def nonconformity_scores(
    errors: NDArray[np.float64], variances: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Per-point scores ``error / sqrt(variance)``.

    ``errors`` are Euclidean distances between completion and truth at the
    held-out points; ``variances`` are the matching per-point total
    predictive variances (``ShapePosterior.predictive_variance()``).
    """
    errors = np.asarray(errors, dtype=np.float64)
    variances = np.asarray(variances, dtype=np.float64)
    if errors.shape != variances.shape:
        raise ValueError("errors and variances must have the same shape")
    if np.any(variances < 0):
        raise ValueError("variances must be non-negative")
    return errors / np.sqrt(np.maximum(variances, _TINY))


def conformal_quantile(scores: NDArray[np.float64], alpha: float) -> float:
    """Finite-sample ``1 - alpha`` conformal quantile of ``scores``.

    Uses the standard ``ceil((n + 1)(1 - alpha)) / n`` level so that, for
    exchangeable data, the resulting interval has coverage at least
    ``1 - alpha``. Returns ``+inf`` if the level exceeds 1 (too few points
    for the requested ``alpha``).
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    scores = np.asarray(scores, dtype=np.float64).ravel()
    n = scores.size
    if n == 0:
        raise ValueError("need at least one calibration score")
    rank = np.ceil((n + 1) * (1.0 - alpha))
    if rank > n:
        return float("inf")
    # kth smallest (1-indexed rank) == (rank-1) index of the sorted scores.
    return float(np.partition(scores, int(rank) - 1)[int(rank) - 1])


@dataclass(frozen=True)
class ConformalCalibrator:
    """A fitted per-point interval scale from split-conformal calibration."""

    scale: float
    alpha: float

    @classmethod
    def fit(
        cls,
        errors: NDArray[np.float64],
        variances: NDArray[np.float64],
        alpha: float = 0.1,
    ) -> "ConformalCalibrator":
        """Calibrate on held-out per-point errors and predicted variances."""
        scores = nonconformity_scores(errors, variances)
        return cls(scale=conformal_quantile(scores, alpha), alpha=alpha)

    def interval_radius(self, variances: NDArray[np.float64]) -> NDArray[np.float64]:
        """Calibrated per-point radius: ``scale * sqrt(variance)``."""
        variances = np.asarray(variances, dtype=np.float64)
        return self.scale * np.sqrt(np.maximum(variances, _TINY))

    def covers(
        self, errors: NDArray[np.float64], variances: NDArray[np.float64]
    ) -> NDArray[np.bool_]:
        """Per-point boolean: is the truth within the calibrated radius?"""
        errors = np.asarray(errors, dtype=np.float64)
        return errors <= self.interval_radius(variances)


def empirical_coverage(
    errors: NDArray[np.float64],
    variances: NDArray[np.float64],
    calibrator: ConformalCalibrator,
) -> float:
    """Fraction of points whose truth falls within the calibrated radius."""
    return float(np.mean(calibrator.covers(errors, variances)))


def coverage_curve(
    calib_errors: NDArray[np.float64],
    calib_variances: NDArray[np.float64],
    test_errors: NDArray[np.float64],
    test_variances: NDArray[np.float64],
    alphas: Sequence[float],
) -> list[tuple[float, float]]:
    """(nominal, empirical) coverage pairs across ``alphas``.

    Fits a calibrator on the calibration split at each ``alpha`` and reports
    the empirical coverage on the disjoint test split. A well-calibrated
    system tracks the diagonal ``empirical ≈ 1 - alpha``.
    """
    curve = []
    for alpha in alphas:
        cal = ConformalCalibrator.fit(calib_errors, calib_variances, alpha)
        curve.append((1.0 - alpha, empirical_coverage(test_errors, test_variances, cal)))
    return curve


def calibrate_completion(
    examples: Iterable[object],
    complete: Callable[[object], tuple[NDArray[np.float64], NDArray[np.float64]]],
    alpha: float = 0.1,
) -> ConformalCalibrator:
    """Driver: calibrate over held-out complete shapes.

    ``examples`` is any iterable of application objects (e.g. complete
    shapes with an ablation policy baked in). ``complete`` maps one example
    to ``(errors, variances)``: the per-point Euclidean completion errors on
    the held-out region and the matching predicted total variances. This
    keeps the registration/correspondence wiring — which is application
    specific — on the caller's side while owning the conformal statistics.
    """
    all_errors: list[NDArray[np.float64]] = []
    all_variances: list[NDArray[np.float64]] = []
    for example in examples:
        errors, variances = complete(example)
        all_errors.append(np.asarray(errors, dtype=np.float64).ravel())
        all_variances.append(np.asarray(variances, dtype=np.float64).ravel())
    if not all_errors:
        raise ValueError("no calibration examples provided")
    return ConformalCalibrator.fit(
        np.concatenate(all_errors), np.concatenate(all_variances), alpha
    )
