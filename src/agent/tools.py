"""Strands Agent tools for sensitivity analysis.

These are the statistical tools the agent can invoke autonomously
to explore the MNAR sensitivity parameter space.
"""

import json
import numpy as np
import pandas as pd
from strands import tool

from src.analysis.pattern_mixture import pattern_mixture_analysis
from src.analysis.tipping_point import run_tipping_point_analysis, find_tipping_point_precise
from src.analysis.baselines import complete_case_analysis, mean_imputation, multiple_imputation_mar


# Module-level state (set by orchestrator before agent runs)
_current_df: pd.DataFrame | None = None
_current_observed: np.ndarray | None = None


def set_analysis_context(df: pd.DataFrame, observed: np.ndarray):
    """Set the current dataset and missingness for agent tools."""
    global _current_df, _current_observed
    _current_df = df
    _current_observed = observed


@tool
def run_baseline_analyses() -> str:
    """Run all baseline methods (complete-case, mean imputation, MI-MAR) on the current dataset.

    Returns JSON with estimates, standard errors, and confidence intervals for each method.
    Use this first to understand the MAR-based estimates before exploring MNAR sensitivity.
    """
    if _current_df is None or _current_observed is None:
        return json.dumps({"error": "No analysis context set"})

    results = {}

    cc = complete_case_analysis(_current_df, _current_observed)
    results["complete_case"] = {
        "estimate": round(cc.estimate, 4),
        "se": round(cc.se, 4),
        "ci": [round(cc.ci_lower, 4), round(cc.ci_upper, 4)],
    }

    mi = mean_imputation(_current_df, _current_observed)
    results["mean_imputation"] = {
        "estimate": round(mi.estimate, 4),
        "se": round(mi.se, 4),
        "ci": [round(mi.ci_lower, 4), round(mi.ci_upper, 4)],
    }

    mar = multiple_imputation_mar(_current_df, _current_observed)
    results["mi_mar"] = {
        "estimate": round(mar.estimate, 4),
        "se": round(mar.se, 4),
        "ci": [round(mar.ci_lower, 4), round(mar.ci_upper, 4)],
    }

    n_obs = int(_current_observed.sum())
    n_mis = int((~_current_observed).sum())
    results["data_summary"] = {
        "n_total": len(_current_df),
        "n_observed": n_obs,
        "n_missing": n_mis,
        "missing_rate": round(n_mis / len(_current_df), 3),
    }

    return json.dumps(results, indent=2)


@tool
def run_sensitivity_at_delta(delta: float) -> str:
    """Run pattern-mixture sensitivity analysis at a specific delta value.

    Args:
        delta: The MNAR sensitivity parameter.
            delta < 0 means missing patients have WORSE outcomes than MAR predicts.
            delta > 0 means missing patients have BETTER outcomes than MAR predicts.
            delta = 0 is equivalent to MAR assumption.

    Returns JSON with the adjusted estimate, CI, and whether the health gain
    remains statistically significant at this delta.
    """
    if _current_df is None or _current_observed is None:
        return json.dumps({"error": "No analysis context set"})

    result = pattern_mixture_analysis(
        df=_current_df,
        observed=_current_observed,
        delta=delta,
    )

    return json.dumps({
        "delta": delta,
        "estimate": round(result.estimate, 4),
        "se": round(result.se, 4),
        "ci": [round(result.ci_lower, 4), round(result.ci_upper, 4)],
        "significant": bool(result.significant),
        "interpretation": (
            f"At delta={delta:.2f}, the estimated health gain is {result.estimate:.2f} "
            f"(95% CI: {result.ci_lower:.2f} to {result.ci_upper:.2f}). "
            f"{'Statistically significant.' if result.significant else 'NOT statistically significant - conclusion reversed.'}"
        ),
    }, indent=2)


@tool
def run_tipping_point_grid(delta_values: str) -> str:
    """Run tipping-point analysis across multiple delta values simultaneously.

    Args:
        delta_values: Comma-separated list of delta values to evaluate.
            Example: "-2.0,-1.5,-1.0,-0.5,0.0,0.5,1.0"

    Returns JSON with results for each delta and the identified tipping point.
    """
    if _current_df is None or _current_observed is None:
        return json.dumps({"error": "No analysis context set"})

    delta_grid = [float(d.strip()) for d in delta_values.split(",")]

    tp_result = run_tipping_point_analysis(
        df=_current_df,
        observed=_current_observed,
        delta_grid=delta_grid,
    )

    results_list = []
    for r in tp_result.results_by_delta:
        results_list.append({
            "delta": r.delta,
            "estimate": round(r.estimate, 4),
            "ci": [round(r.ci_lower, 4), round(r.ci_upper, 4)],
            "significant": bool(r.significant),
        })

    return json.dumps({
        "tipping_point": tp_result.tipping_point,
        "is_robust": bool(tp_result.is_robust),
        "results": results_list,
        "summary": (
            f"Tipping point: delta={tp_result.tipping_point:.2f}. "
            if tp_result.tipping_point is not None
            else "No tipping point found - conclusion is robust across all tested delta values."
        ),
    }, indent=2)


@tool
def refine_tipping_point(bracket_lo: float, bracket_hi: float, precision: float = 0.05) -> str:
    """Refine the tipping point location using bisection between two delta values.

    Use this after identifying a bracket where the tipping point lies
    (one delta is significant, the adjacent one is not).

    Args:
        bracket_lo: Lower bound of bracket (more negative delta, significant)
        bracket_hi: Upper bound of bracket (less negative delta, not significant)
        precision: Desired precision for tipping point location

    Returns the refined tipping point value.
    """
    if _current_df is None or _current_observed is None:
        return json.dumps({"error": "No analysis context set"})

    tp = find_tipping_point_precise(
        df=_current_df,
        observed=_current_observed,
        bracket_lo=bracket_lo,
        bracket_hi=bracket_hi,
        precision=precision,
    )

    return json.dumps({
        "refined_tipping_point": round(tp, 3),
        "bracket": [bracket_lo, bracket_hi],
        "precision": precision,
        "interpretation": (
            f"The tipping point is at delta ≈ {tp:.3f}. "
            f"This means the conclusion of positive health gain would be reversed "
            f"if missing patients had outcomes {abs(tp):.3f} standard deviations "
            f"{'worse' if tp < 0 else 'better'} than MAR-imputed values predict."
        ),
    }, indent=2)


@tool
def get_clinical_context() -> str:
    """Get clinical context about the dataset and analysis to inform interpretation.

    Returns information about the Oxford Hip Score, typical health gains,
    and what constitutes a clinically meaningful difference.
    """
    return json.dumps({
        "instrument": "Oxford Hip Score (OHS)",
        "range": "0-48 (higher = better function)",
        "mcid": "5 points (Minimal Clinically Important Difference)",
        "typical_gain": "20-22 points for primary hip replacement",
        "context": (
            "The OHS measures hip-related function and pain. "
            "A score change of 5+ points is considered clinically meaningful. "
            "Typical health gain after primary hip replacement is 20-22 points. "
            "Patients with worse pre-operative scores tend to gain more. "
            "Non-responders to post-operative questionnaires may have worse outcomes "
            "due to complications, dissatisfaction, or poor mobility."
        ),
        "mnar_plausibility": (
            "Negative delta values (worse outcomes for non-responders) are clinically "
            "plausible because: (1) patients with complications may be less able to "
            "complete questionnaires, (2) dissatisfied patients may be less motivated "
            "to respond, (3) patients with poor mobility may have difficulty attending "
            "follow-up appointments."
        ),
    }, indent=2)
