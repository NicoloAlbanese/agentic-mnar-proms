"""Evaluation metrics for comparing analysis methods."""

import numpy as np
from dataclasses import dataclass


@dataclass
class ReplicationMetrics:
    """Metrics from a single Monte Carlo replication."""

    true_theta: float  # True mean health gain (from complete data)
    estimate: float  # Point estimate from method
    se: float  # Standard error
    ci_lower: float  # Lower CI bound
    ci_upper: float  # Upper CI bound
    method: str
    delta: float  # True MNAR delta used in simulation
    missing_rate: float  # Actual missing rate


@dataclass
class AggregatedMetrics:
    """Aggregated metrics across replications for one method × scenario."""

    method: str
    delta: float
    missing_rate: float
    n_replications: int
    bias: float
    rmse: float
    coverage: float  # Proportion of CIs containing true theta
    mean_ci_width: float
    mean_estimate: float
    se_of_estimate: float  # Monte Carlo SE of the point estimate


def compute_aggregated_metrics(
    replications: list[ReplicationMetrics],
) -> AggregatedMetrics:
    """Aggregate metrics across Monte Carlo replications.

    Args:
        replications: List of per-replication results (same method & scenario)

    Returns:
        AggregatedMetrics summary
    """
    if not replications:
        raise ValueError("Empty replications list")

    method = replications[0].method
    delta = replications[0].delta
    missing_rate = replications[0].missing_rate

    true_thetas = np.array([r.true_theta for r in replications])
    estimates = np.array([r.estimate for r in replications])
    ci_lowers = np.array([r.ci_lower for r in replications])
    ci_uppers = np.array([r.ci_upper for r in replications])

    # Bias: E[θ̂ - θ]
    errors = estimates - true_thetas
    bias = errors.mean()

    # RMSE: √E[(θ̂ - θ)²]
    rmse = np.sqrt((errors ** 2).mean())

    # Coverage: proportion of CIs containing true θ
    covers = (ci_lowers <= true_thetas) & (true_thetas <= ci_uppers)
    coverage = covers.mean()

    # Mean CI width
    ci_widths = ci_uppers - ci_lowers
    mean_ci_width = ci_widths.mean()

    # Monte Carlo SE of the estimate
    mean_estimate = estimates.mean()
    se_of_estimate = estimates.std(ddof=1) / np.sqrt(len(estimates))

    return AggregatedMetrics(
        method=method,
        delta=delta,
        missing_rate=missing_rate,
        n_replications=len(replications),
        bias=bias,
        rmse=rmse,
        coverage=coverage,
        mean_ci_width=mean_ci_width,
        mean_estimate=mean_estimate,
        se_of_estimate=se_of_estimate,
    )


@dataclass
class TippingPointMetrics:
    """Metrics for evaluating tipping-point identification accuracy."""

    true_tipping_point: float | None
    estimated_tipping_point: float | None
    absolute_error: float | None  # |δ̂_tip - δ_true_tip|
    concordance: bool  # Same robustness conclusion?
    n_deltas_evaluated: int  # Efficiency measure
    wall_time_seconds: float


def compute_tipping_point_accuracy(
    true_tp: float | None,
    estimated_tp: float | None,
    n_deltas: int,
    wall_time: float,
) -> TippingPointMetrics:
    """Compute accuracy of tipping-point identification.

    Args:
        true_tp: True tipping point (None if truly robust)
        estimated_tp: Estimated tipping point (None if method says robust)
        n_deltas: Number of delta values evaluated
        wall_time: Computation time in seconds
    """
    # Concordance: both say robust, or both identify a tipping point
    concordance = (true_tp is None) == (estimated_tp is None)

    # Absolute error (only if both identify a tipping point)
    if true_tp is not None and estimated_tp is not None:
        absolute_error = abs(estimated_tp - true_tp)
    else:
        absolute_error = None

    return TippingPointMetrics(
        true_tipping_point=true_tp,
        estimated_tipping_point=estimated_tp,
        absolute_error=absolute_error,
        concordance=concordance,
        n_deltas_evaluated=n_deltas,
        wall_time_seconds=wall_time,
    )
