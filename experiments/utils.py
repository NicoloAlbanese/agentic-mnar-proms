"""Shared utilities for experiment phases.

Provides common sampling, missingness generation, and result persistence
to avoid duplication across phase modules.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import RANDOM_SEED, PROCESSED_DATA_DIR
from src.data.synthetic import get_or_create_synthetic
from src.simulation.mechanisms import generate_mnar_missingness
from experiments.config import ExperimentConfig


def load_population(config: ExperimentConfig) -> pd.DataFrame:
    """Load population data based on scenario config.

    Prefers real NHS PROMs data when available (high_effect only),
    falls back to synthetic generation.
    """
    real_path = PROCESSED_DATA_DIR / "proms_hip_real.parquet"
    if real_path.exists() and config.scenario != "low_effect":
        df = pd.read_parquet(real_path)
        print(f"Using REAL NHS PROMs data: {len(df)} patients")
        return df

    return get_or_create_synthetic(config=config.synthetic_config)


def draw_sample_with_missingness(
    df_population: pd.DataFrame,
    true_delta: float,
    missing_rate: float,
    sample_size: int,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray, float, np.random.Generator]:
    """Draw a bootstrap sample and generate MNAR missingness.

    Returns:
        (df_sample, observed_mask, true_theta, rng)
    """
    rng = np.random.default_rng(seed)

    idx = rng.choice(len(df_population), size=sample_size, replace=True)
    df_sample = df_population.iloc[idx].reset_index(drop=True)

    true_theta = (df_sample["ohs_post"] - df_sample["ohs_pre"]).mean()

    observed = generate_mnar_missingness(
        df=df_sample,
        delta=true_delta,
        target_missing_rate=missing_rate,
        rng=rng,
    )

    return df_sample, observed, true_theta, rng


def save_incremental(results: list[dict], path: Path):
    """Save intermediate results to parquet for crash recovery."""
    if results:
        pd.DataFrame(results).to_parquet(path, index=False)
