"""Pattern-mixture model for MNAR sensitivity analysis."""

import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass

from src.analysis.baselines import AnalysisResult


@dataclass
class PatternMixtureResult:
    """Result from pattern-mixture model at a specific delta."""

    delta: float
    estimate: float
    se: float
    ci_lower: float
    ci_upper: float
    significant: bool  # Whether CI excludes zero


def pattern_mixture_analysis(
    df: pd.DataFrame,
    observed: np.ndarray,
    delta: float,
    pre_col: str = "ohs_pre",
    post_col: str = "ohs_post",
    n_imputations: int = 20,
    rng_seed: int = 42,
) -> PatternMixtureResult:
    """Run pattern-mixture model with delta-adjustment.

    Under the pattern-mixture framework, the distribution of the missing
    outcomes is shifted by delta relative to the MAR-imputed distribution:

        Y_mis ~ f(Y | R=1, X) + δ

    This means: if δ < 0, missing patients are assumed to have outcomes
    that are δ units worse than what MAR imputation would predict.

    Args:
        df: Full dataset
        observed: Boolean mask (True = observed)
        delta: Sensitivity parameter (shift applied to imputed values)
        pre_col: Pre-operative score column
        post_col: Post-operative score column
        n_imputations: Number of imputations
        rng_seed: Random seed

    Returns:
        PatternMixtureResult with adjusted estimates
    """
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer

    # Prepare data
    df_mi = df[[pre_col, post_col]].copy()
    cov_cols = [c for c in ["age_band", "gender", "imd_quintile"] if c in df.columns]
    if cov_cols:
        df_encoded = pd.get_dummies(df[cov_cols], drop_first=True)
        df_mi = pd.concat([df_mi, df_encoded], axis=1)

    df_mi.loc[~observed, post_col] = np.nan

    estimates = []
    variances = []

    for m in range(n_imputations):
        imputer = IterativeImputer(
            max_iter=10,
            random_state=rng_seed + m,
            sample_posterior=True,
        )
        imputed = imputer.fit_transform(df_mi)
        post_idx = df_mi.columns.get_loc(post_col)
        post_imputed = imputed[:, post_idx]

        # Apply delta-adjustment to imputed (missing) values only
        post_adjusted = post_imputed.copy()
        post_adjusted[~observed] += delta

        pre_values = df_mi[pre_col].values
        change = post_adjusted - pre_values

        estimates.append(change.mean())
        variances.append(change.var(ddof=1) / len(change))

    # Rubin's rules
    q_bar = np.mean(estimates)
    u_bar = np.mean(variances)
    b = np.var(estimates, ddof=1)
    total_var = u_bar + (1 + 1 / n_imputations) * b
    se = np.sqrt(total_var)

    if b > 0:
        r = (1 + 1 / n_imputations) * b / u_bar
        df_br = (n_imputations - 1) * (1 + 1 / r) ** 2
        t_crit = stats.t.ppf(0.975, df=max(df_br, 2))
    else:
        t_crit = 1.96

    ci_lower = q_bar - t_crit * se
    ci_upper = q_bar + t_crit * se
    significant = ci_lower > 0 or ci_upper < 0

    return PatternMixtureResult(
        delta=delta,
        estimate=q_bar,
        se=se,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        significant=significant,
    )
