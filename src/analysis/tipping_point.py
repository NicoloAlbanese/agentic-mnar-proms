"""Tipping-point sensitivity analysis."""

import numpy as np
import pandas as pd
from dataclasses import dataclass

from src.analysis.pattern_mixture import pattern_mixture_analysis, PatternMixtureResult


@dataclass
class TippingPointResult:
    """Result of a tipping-point analysis."""

    tipping_point: float | None  # δ value where conclusion reverses (None if robust)
    results_by_delta: list[PatternMixtureResult]
    is_robust: bool  # True if conclusion holds across all δ values tested
    delta_grid: list[float]


def run_tipping_point_analysis(
    df: pd.DataFrame,
    observed: np.ndarray,
    delta_grid: list[float],
    pre_col: str = "ohs_pre",
    post_col: str = "ohs_post",
    n_imputations: int = 20,
    rng_seed: int = 42,
) -> TippingPointResult:
    """Run tipping-point analysis across a grid of delta values.

    The tipping point is the smallest |δ| at which the treatment effect
    (health gain) is no longer statistically significant (95% CI includes 0).

    Args:
        df: Full dataset
        observed: Boolean mask
        delta_grid: List of δ values to evaluate (should include 0)
        pre_col: Pre-operative score column
        post_col: Post-operative score column
        n_imputations: Number of MI imputations per δ
        rng_seed: Base random seed

    Returns:
        TippingPointResult with identified tipping point
    """
    results = []
    for i, delta in enumerate(sorted(delta_grid)):
        result = pattern_mixture_analysis(
            df=df,
            observed=observed,
            delta=delta,
            pre_col=pre_col,
            post_col=post_col,
            n_imputations=n_imputations,
            rng_seed=rng_seed + i * 100,
        )
        results.append(result)

    # Find tipping point: smallest |δ| where significance is lost
    # Start from δ=0 and move outward in negative direction (clinically relevant)
    sorted_results = sorted(results, key=lambda r: abs(r.delta))

    # Check if result at δ=0 is significant
    zero_result = next((r for r in results if r.delta == 0.0), results[0])
    baseline_significant = zero_result.significant

    tipping_point = None
    if baseline_significant:
        # Look for first δ where significance is lost
        # Focus on negative δ (worse outcomes for missing patients)
        negative_results = sorted(
            [r for r in results if r.delta <= 0],
            key=lambda r: r.delta,
            reverse=True,  # From 0 toward more negative
        )
        for r in negative_results:
            if not r.significant:
                tipping_point = r.delta
                break

    is_robust = tipping_point is None

    return TippingPointResult(
        tipping_point=tipping_point,
        results_by_delta=results,
        is_robust=is_robust,
        delta_grid=sorted(delta_grid),
    )


def find_tipping_point_precise(
    df: pd.DataFrame,
    observed: np.ndarray,
    bracket_lo: float,
    bracket_hi: float,
    precision: float = 0.05,
    pre_col: str = "ohs_pre",
    post_col: str = "ohs_post",
    n_imputations: int = 20,
    rng_seed: int = 42,
) -> float:
    """Refine tipping point using bisection within a bracket.

    Given that the tipping point lies between bracket_lo and bracket_hi,
    use bisection to find it to the specified precision.
    """
    lo, hi = bracket_lo, bracket_hi

    while (hi - lo) > precision:
        mid = (lo + hi) / 2.0
        result = pattern_mixture_analysis(
            df=df, observed=observed, delta=mid,
            pre_col=pre_col, post_col=post_col,
            n_imputations=n_imputations, rng_seed=rng_seed,
        )
        if result.significant:
            hi = mid  # Tipping point is more negative
        else:
            lo = mid  # Tipping point is less negative

    return (lo + hi) / 2.0
