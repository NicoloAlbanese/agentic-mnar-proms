"""MNAR missingness simulation using selection models."""

import numpy as np
import pandas as pd
from scipy.special import expit


def calibrate_intercept(
    y_mis: np.ndarray,
    x: np.ndarray,
    alpha1: np.ndarray,
    delta: float,
    target_rate: float,
    tol: float = 0.001,
    max_iter: int = 100,
) -> float:
    """Calibrate intercept α₀ to achieve target missingness rate.

    Uses bisection to find α₀ such that:
        E[1 - expit(α₀ + α₁·X + δ·Y_mis)] ≈ target_rate

    Args:
        y_mis: Outcome values (would be unobserved for missing)
        x: Covariate matrix
        alpha1: Covariate coefficients
        delta: MNAR sensitivity parameter
        target_rate: Desired proportion of missing values
        tol: Convergence tolerance
        max_iter: Maximum bisection iterations

    Returns:
        Calibrated intercept value α₀
    """
    linear_pred_no_intercept = x @ alpha1 + delta * y_mis

    # Bisection search
    lo, hi = -20.0, 20.0
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        prob_observed = expit(mid + linear_pred_no_intercept)
        actual_missing_rate = 1.0 - prob_observed.mean()

        if abs(actual_missing_rate - target_rate) < tol:
            return mid
        elif actual_missing_rate < target_rate:
            hi = mid  # Need lower intercept → more missing
        else:
            lo = mid  # Need higher intercept → less missing

    return (lo + hi) / 2.0


def generate_mnar_missingness(
    df: pd.DataFrame,
    delta: float,
    target_missing_rate: float,
    outcome_col: str = "ohs_post",
    covariate_cols: list[str] | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate MNAR missingness indicators using a selection model.

    Model: P(R=1 | X, Y_mis) = expit(α₀ + α₁·X + δ·Y_mis)

    Where R=1 means OBSERVED, so R=0 means MISSING.

    Args:
        df: DataFrame with outcome and covariates
        delta: MNAR sensitivity parameter
            δ < 0: worse outcomes → more likely missing (clinically plausible)
            δ = 0: MAR (missingness independent of outcome given covariates)
            δ > 0: better outcomes → more likely missing
        target_missing_rate: Desired proportion of missing outcomes
        outcome_col: Name of the outcome column
        covariate_cols: Covariate columns to include in selection model
        rng: Random number generator

    Returns:
        Boolean array where True = OBSERVED, False = MISSING
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(df)
    y_mis = df[outcome_col].values.astype(float)

    # Standardize outcome for numerical stability
    y_std = (y_mis - y_mis.mean()) / (y_mis.std() + 1e-8)

    # Build covariate matrix
    if covariate_cols and any(c in df.columns for c in covariate_cols):
        available_covs = [c for c in covariate_cols if c in df.columns]
        x = df[available_covs].copy()
        # Encode categoricals
        x = pd.get_dummies(x, drop_first=True).values.astype(float)
        # Standardize
        x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-8)
        # Simple coefficients (small effect of covariates)
        alpha1 = np.full(x.shape[1], 0.1)
    else:
        # No covariates - pure outcome-dependent missingness
        x = np.zeros((n, 1))
        alpha1 = np.array([0.0])

    # Calibrate intercept
    alpha0 = calibrate_intercept(y_std, x, alpha1, delta, target_missing_rate)

    # Generate observation probabilities
    linear_pred = alpha0 + x @ alpha1 + delta * y_std
    prob_observed = expit(linear_pred)

    # Generate missingness indicators
    u = rng.uniform(size=n)
    observed = u < prob_observed

    actual_rate = 1.0 - observed.mean()
    return observed
