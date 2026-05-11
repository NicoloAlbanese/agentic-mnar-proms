"""Baseline analysis methods for handling missing PROMs data."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer


@dataclass
class AnalysisResult:
    """Result from a single analysis method on one dataset."""

    estimate: float  # Point estimate of mean health gain
    se: float  # Standard error
    ci_lower: float  # Lower 95% CI bound
    ci_upper: float  # Upper 95% CI bound
    method: str  # Method name


def complete_case_analysis(
    df: pd.DataFrame,
    observed: np.ndarray,
    pre_col: str = "ohs_pre",
    post_col: str = "ohs_post",
) -> AnalysisResult:
    """Analyze only cases with observed post-operative scores.

    This is biased under MNAR but serves as a common baseline.
    """
    df_obs = df[observed].copy()
    change = df_obs[post_col] - df_obs[pre_col]

    n = len(change)
    estimate = change.mean()
    se = change.std(ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se

    return AnalysisResult(
        estimate=estimate, se=se, ci_lower=ci_lower, ci_upper=ci_upper,
        method="complete_case"
    )


def mean_imputation(
    df: pd.DataFrame,
    observed: np.ndarray,
    pre_col: str = "ohs_pre",
    post_col: str = "ohs_post",
) -> AnalysisResult:
    """Replace missing post-op scores with observed mean.

    Underestimates variance; biased under MNAR.
    """
    df_imp = df.copy()
    df_imp[post_col] = df_imp[post_col].astype(float)
    obs_mean = df_imp.loc[observed, post_col].mean()
    df_imp.loc[~observed, post_col] = obs_mean

    change = df_imp[post_col] - df_imp[pre_col]
    n = len(change)
    estimate = change.mean()
    se = change.std(ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se

    return AnalysisResult(
        estimate=estimate, se=se, ci_lower=ci_lower, ci_upper=ci_upper,
        method="mean_imputation"
    )


def multiple_imputation_mar(
    df: pd.DataFrame,
    observed: np.ndarray,
    pre_col: str = "ohs_pre",
    post_col: str = "ohs_post",
    n_imputations: int = 20,
    rng_seed: int = 42,
) -> AnalysisResult:
    """Multiple imputation under MAR using fully conditional specification.

    Uses sklearn IterativeImputer (MICE algorithm).
    Pools results using Rubin's rules.
    """
    # Prepare data with missing values
    df_mi = df[[pre_col, post_col]].copy()
    # Add covariates if available
    cov_cols = [c for c in ["age_band", "gender", "imd_quintile"] if c in df.columns]
    if cov_cols:
        df_encoded = pd.get_dummies(df[cov_cols], drop_first=True)
        df_mi = pd.concat([df_mi, df_encoded], axis=1)

    # Set missing values
    df_mi.loc[~observed, post_col] = np.nan

    # Run multiple imputations
    estimates = []
    variances = []

    for m in range(n_imputations):
        imputer = IterativeImputer(
            max_iter=10,
            random_state=rng_seed + m,
            sample_posterior=True,
        )
        imputed = imputer.fit_transform(df_mi)
        # Extract imputed post-op scores
        post_idx = df_mi.columns.get_loc(post_col)
        post_imputed = imputed[:, post_idx]
        pre_values = df_mi[pre_col].values

        change = post_imputed - pre_values
        estimates.append(change.mean())
        variances.append(change.var(ddof=1) / len(change))

    # Rubin's rules
    m = n_imputations
    q_bar = np.mean(estimates)  # Pooled estimate
    u_bar = np.mean(variances)  # Within-imputation variance
    b = np.var(estimates, ddof=1)  # Between-imputation variance
    total_var = u_bar + (1 + 1 / m) * b  # Total variance
    se = np.sqrt(total_var)

    # Degrees of freedom (Barnard-Rubin)
    if b > 0:
        r = (1 + 1 / m) * b / u_bar
        df_br = (m - 1) * (1 + 1 / r) ** 2
        t_crit = stats.t.ppf(0.975, df=max(df_br, 2))
    else:
        t_crit = 1.96

    ci_lower = q_bar - t_crit * se
    ci_upper = q_bar + t_crit * se

    return AnalysisResult(
        estimate=q_bar, se=se, ci_lower=ci_lower, ci_upper=ci_upper,
        method="mi_mar"
    )
