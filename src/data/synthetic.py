"""Generate synthetic PROMs-like data for development and testing.

This creates a realistic synthetic dataset mimicking the statistical
properties of NHS PROMs hip replacement data, allowing the full pipeline
to run without downloading real data.

The synthetic data preserves:
- Realistic OHS score distributions (0-48 scale)
- Configurable health gain (effect size)
- Correlation between pre/post scores
- Covariate structure (age, gender, deprivation)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass

from src.config import PROCESSED_DATA_DIR, RANDOM_SEED


@dataclass
class SyntheticConfig:
    """Configuration for synthetic data generation."""

    n: int = 10_000
    mean_pre: float = 18.0
    sd_pre: float = 8.0
    mean_post: float = 38.0
    sd_post: float = 9.0
    correlation: float = 0.3
    seed: int = RANDOM_SEED
    label: str = "default"

    @property
    def mean_gain(self) -> float:
        return self.mean_post - self.mean_pre


# Predefined scenarios
SCENARIO_HIGH_EFFECT = SyntheticConfig(
    mean_pre=18.0, sd_pre=8.0,
    mean_post=38.0, sd_post=9.0,
    label="high_effect",
)

SCENARIO_LOW_EFFECT = SyntheticConfig(
    mean_pre=18.0, sd_pre=8.0,
    mean_post=25.0, sd_post=8.0,
    label="low_effect",
)


def generate_synthetic_proms(config: SyntheticConfig | None = None) -> pd.DataFrame:
    """Generate synthetic hip replacement PROMs data.

    Args:
        config: SyntheticConfig with distribution parameters.
                Defaults to high-effect scenario (standard hip replacement).

    Returns:
        DataFrame with ohs_pre, ohs_post, ohs_change, age_band, gender, imd_quintile
    """
    if config is None:
        config = SCENARIO_HIGH_EFFECT

    rng = np.random.default_rng(config.seed)

    # Generate correlated pre/post scores using bivariate normal
    cov = [
        [config.sd_pre**2, config.correlation * config.sd_pre * config.sd_post],
        [config.correlation * config.sd_pre * config.sd_post, config.sd_post**2],
    ]

    scores = rng.multivariate_normal(
        [config.mean_pre, config.mean_post], cov, size=config.n
    )
    ohs_pre = scores[:, 0]
    ohs_post = scores[:, 1]

    # Clip to valid OHS range [0, 48]
    ohs_pre = np.clip(ohs_pre, 0, 48).round().astype(int)
    ohs_post = np.clip(ohs_post, 0, 48).round().astype(int)

    # Covariates
    age_bands = ["50-59", "60-69", "70-79", "80+"]
    age_probs = [0.20, 0.35, 0.30, 0.15]
    age_band = rng.choice(age_bands, size=config.n, p=age_probs)

    gender = rng.choice(["Male", "Female"], size=config.n, p=[0.42, 0.58])

    imd_quintile = rng.choice(
        [1, 2, 3, 4, 5], size=config.n, p=[0.15, 0.20, 0.22, 0.22, 0.21]
    )

    # Covariate effects on outcomes
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


def get_or_create_synthetic(
    n: int = 10_000,
    config: SyntheticConfig | None = None,
) -> pd.DataFrame:
    """Load synthetic data from cache or generate fresh.

    Args:
        n: Number of records (used if config is None)
        config: Full configuration. If None, uses high-effect default with n.
    """
    if config is None:
        config = SyntheticConfig(n=n)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"proms_hip_{config.label}.parquet"
    path = PROCESSED_DATA_DIR / filename

    if path.exists():
        return pd.read_parquet(path)

    print(f"Generating synthetic PROMs data ({config.label}, gain≈{config.mean_gain:.0f}) ...")
    df = generate_synthetic_proms(config)
    df.to_parquet(path, index=False)
    print(f"Saved {len(df)} records to {path}")
    return df
