"""Generate synthetic PROMs-like data for development and testing.

This creates a realistic synthetic dataset mimicking the statistical
properties of NHS PROMs hip replacement data, allowing the full pipeline
to run without downloading real data.

The synthetic data preserves:
- Realistic OHS score distributions (0-48 scale)
- Typical health gain (~20 points)
- Correlation between pre/post scores
- Covariate structure (age, gender, deprivation)
"""

import numpy as np
import pandas as pd

from src.config import PROCESSED_DATA_DIR, RANDOM_SEED


def generate_synthetic_proms(
    n: int = 10_000,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Generate synthetic hip replacement PROMs data.

    Based on published summary statistics from NHS PROMs reports:
    - Mean pre-op OHS ≈ 18 (SD ≈ 8)
    - Mean post-op OHS ≈ 38 (SD ≈ 9)
    - Mean health gain ≈ 20 (SD ≈ 10)
    - Correlation(pre, post) ≈ 0.3

    Args:
        n: Number of patients to generate
        seed: Random seed

    Returns:
        DataFrame with ohs_pre, ohs_post, ohs_change, age_band, gender, imd_quintile
    """
    rng = np.random.default_rng(seed)

    # Generate correlated pre/post scores using bivariate normal
    mean_pre, sd_pre = 18.0, 8.0
    mean_post, sd_post = 38.0, 9.0
    correlation = 0.3

    # Covariance matrix
    cov = [[sd_pre**2, correlation * sd_pre * sd_post],
           [correlation * sd_pre * sd_post, sd_post**2]]

    scores = rng.multivariate_normal([mean_pre, mean_post], cov, size=n)
    ohs_pre = scores[:, 0]
    ohs_post = scores[:, 1]

    # Clip to valid OHS range [0, 48]
    ohs_pre = np.clip(ohs_pre, 0, 48).round().astype(int)
    ohs_post = np.clip(ohs_post, 0, 48).round().astype(int)

    # Covariates
    age_bands = ["50-59", "60-69", "70-79", "80+"]
    age_probs = [0.20, 0.35, 0.30, 0.15]
    age_band = rng.choice(age_bands, size=n, p=age_probs)

    gender = rng.choice(["Male", "Female"], size=n, p=[0.42, 0.58])

    imd_quintile = rng.choice([1, 2, 3, 4, 5], size=n, p=[0.15, 0.20, 0.22, 0.22, 0.21])

    # Introduce covariate effects on outcomes
    # Older patients and more deprived patients have slightly worse post-op scores
    age_effect = np.where(age_band == "80+", -3, np.where(age_band == "70-79", -1, 0))
    imd_effect = np.where(imd_quintile >= 4, -2, 0)
    ohs_post = np.clip(ohs_post + age_effect + imd_effect, 0, 48)

    df = pd.DataFrame({
        "ohs_pre": ohs_pre,
        "ohs_post": ohs_post,
        "ohs_change": ohs_post - ohs_pre,
        "age_band": age_band,
        "gender": gender,
        "imd_quintile": imd_quintile,
    })

    return df


def get_or_create_synthetic(n: int = 10_000) -> pd.DataFrame:
    """Load synthetic data from cache or generate fresh.

    Saves to PROCESSED_DATA_DIR for reuse.
    """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DATA_DIR / "proms_hip_complete.parquet"

    if path.exists():
        return pd.read_parquet(path)

    print("Generating synthetic PROMs data ...")
    df = generate_synthetic_proms(n=n)
    df.to_parquet(path, index=False)
    print(f"Saved {len(df)} records to {path}")
    return df
