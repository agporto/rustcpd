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


def failure_detection_auc(
    sigma2: NDArray[np.float64], correct: NDArray[np.bool_]
) -> float:
    """AUC for using a low atlas residual ``sigma2`` to predict a correct fit.

    ``correct`` is a boolean/0-1 label per fit (e.g. pose within tolerance).
    Returns the rank-based (Mann-Whitney) area under the ROC of the score
    ``-log(sigma2)``; 0.5 is chance, 1.0 is perfect separation. Use it to check
    that ``sigma2`` actually discriminates failures on *your* data before
    trusting a calibrated probability.
    """
    sigma2 = np.asarray(sigma2, dtype=np.float64).ravel()
    y = np.asarray(correct).ravel().astype(bool)
    if sigma2.shape != y.shape:
        raise ValueError("sigma2 and correct must have the same shape")
    if y.all() or not y.any():
        raise ValueError("need both correct and failed examples")
    score = -np.log(np.maximum(sigma2, _TINY))
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, score.size + 1)
    # average ranks for ties so the statistic is exact under ties
    _, inv, counts = np.unique(score, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    avg = (csum - (counts - 1) / 2.0)
    ranks = avg[inv]
    n_pos = int(y.sum()); n_neg = int((~y).sum())
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


@dataclass(frozen=True)
class PoseConfidenceCalibrator:
    """Maps an atlas fit's residual variance ``sigma2`` to ``P(pose correct)``.

    A wrong pose basin cannot fit the fragment, so it leaves a large residual
    variance; ``AtlasResult.sigma2`` is therefore a strong, *unsupervised*
    failure signal (empirically near-perfect separation on synthetic SSM
    fragments). This calibrator turns that monotone signal into a probability
    with one-dimensional logistic regression on ``log(sigma2)``, fit on a set of
    completed fragments you have labelled correct/failed.

    ``sigma2`` is in squared target-coordinate units, so the fit is **dataset
    specific**: always recalibrate on your own data and coordinate frame rather
    than reusing coefficients across problems. Check
    :func:`failure_detection_auc` first to confirm ``sigma2`` discriminates at
    all on your data.

    Example::

        cal = PoseConfidenceCalibrator.fit(sigma2_array, correct_array)
        p = cal.probability(new_fit.sigma2)     # P(pose correct)
        if not cal.trust(new_fit.sigma2):       # flag for more keypoints
            ...
    """

    intercept: float
    slope: float  # coefficient on log(sigma2); negative (low sigma2 -> confident)

    @classmethod
    def fit(
        cls,
        sigma2: NDArray[np.float64],
        correct: NDArray[np.bool_],
        *,
        max_iter: int = 100,
        tol: float = 1e-10,
    ) -> "PoseConfidenceCalibrator":
        """Fit ``P(correct) = sigmoid(intercept + slope * log(sigma2))`` by IRLS."""
        sigma2 = np.asarray(sigma2, dtype=np.float64).ravel()
        y = np.asarray(correct).ravel().astype(np.float64)
        if sigma2.shape != y.shape:
            raise ValueError("sigma2 and correct must have the same shape")
        if sigma2.size == 0:
            raise ValueError("need at least one example")
        if np.any(sigma2 <= 0):
            raise ValueError("sigma2 must be positive")
        if y.min() == y.max():
            raise ValueError("need both correct and failed examples to fit")
        x = np.log(sigma2)
        design = np.column_stack([np.ones_like(x), x])
        w = np.zeros(2)
        for _ in range(max_iter):
            p = 1.0 / (1.0 + np.exp(-(design @ w)))
            grad = design.T @ (p - y)
            weights = np.maximum(p * (1.0 - p), _TINY)
            hess = (design * weights[:, None]).T @ design
            step = np.linalg.solve(hess + 1e-9 * np.eye(2), grad)
            w = w - step
            if np.max(np.abs(step)) < tol:
                break
        return cls(intercept=float(w[0]), slope=float(w[1]))

    def probability(self, sigma2: NDArray[np.float64]) -> NDArray[np.float64]:
        """Calibrated ``P(pose correct)`` for one or more ``sigma2`` values."""
        s = np.asarray(sigma2, dtype=np.float64)
        x = np.log(np.maximum(s, _TINY))
        return 1.0 / (1.0 + np.exp(-(self.intercept + self.slope * x)))

    def trust(
        self, sigma2: NDArray[np.float64], threshold: float = 0.5
    ) -> NDArray[np.bool_]:
        """Boolean: is the fit's calibrated confidence at least ``threshold``?

        Fragments where this is ``False`` are the ones to flag for review or for
        collecting additional keypoints.
        """
        return self.probability(sigma2) >= threshold


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
