"""Preprocess NHS PROMs data to extract complete-case cohort."""

import pandas as pd
import numpy as np
from pathlib import Path

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR


def load_raw_proms(data_dir: Path | None = None) -> pd.DataFrame:
    """Load raw PROMs CSV files from directory.

    Handles the NHS PROMs CSV format with record-level data.
    """
    if data_dir is None:
        data_dir = RAW_DATA_DIR / "hip_2223"

    csv_files = list(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    dfs = [pd.read_csv(f, low_memory=False) for f in csv_files]
    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(df)} records from {len(csv_files)} file(s)")
    return df


def extract_complete_cases(df: pd.DataFrame) -> pd.DataFrame:
    """Extract complete-case cohort for simulation ground truth.

    Identifies OHS columns flexibly and drops rows with any missing items.
    """
    # Try standard NHS PROMs column patterns
    q1_ohs = sorted([c for c in df.columns if "Q1" in c and "OHS" in c])
    q2_ohs = sorted([c for c in df.columns if "Q2" in c and "OHS" in c])

    if not q1_ohs or not q2_ohs:
        # Try alternative naming: Pre_Op_Q_Oxford_Hip_Score etc.
        q1_ohs = [c for c in df.columns if "Pre" in c and "Oxford" in c]
        q2_ohs = [c for c in df.columns if "Post" in c and "Oxford" in c]

    if not q1_ohs or not q2_ohs:
        raise ValueError(
            "Cannot identify OHS columns. "
            f"Available columns: {list(df.columns[:20])}..."
        )

    # Drop rows with any missing OHS items
    complete = df.dropna(subset=q1_ohs + q2_ohs)
    print(f"Complete OHS cases: {len(complete)} / {len(df)} "
          f"({100 * len(complete) / len(df):.1f}%)")
    return complete


def build_analysis_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Build the final analysis dataset with computed scores and covariates.

    Returns DataFrame with columns:
    - ohs_pre: Pre-operative Oxford Hip Score (0-48)
    - ohs_post: Post-operative Oxford Hip Score (0-48)
    - ohs_change: Health gain (post - pre)
    - age_band, gender, imd_quintile (if available)
    """
    result = pd.DataFrame()

    # Compute OHS total scores from item-level data
    q1_ohs = sorted([c for c in df.columns if "Q1" in c and "OHS" in c])
    q2_ohs = sorted([c for c in df.columns if "Q2" in c and "OHS" in c])

    if q1_ohs and q2_ohs:
        result["ohs_pre"] = df[q1_ohs].sum(axis=1)
        result["ohs_post"] = df[q2_ohs].sum(axis=1)
    else:
        # Handle pre-computed total score columns
        pre_col = [c for c in df.columns if "Pre" in c and "Hip" in c and "Score" in c]
        post_col = [c for c in df.columns if "Post" in c and "Hip" in c and "Score" in c]
        if pre_col and post_col:
            result["ohs_pre"] = df[pre_col[0]].values
            result["ohs_post"] = df[post_col[0]].values
        else:
            raise ValueError("Cannot identify OHS score columns in data")

    result["ohs_change"] = result["ohs_post"] - result["ohs_pre"]

    # Covariates - flexible column matching
    for target, patterns in [
        ("age_band", ["Age", "age"]),
        ("gender", ["Gender", "Sex", "gender", "sex"]),
        ("imd_quintile", ["IMD", "imd", "Deprivation"]),
    ]:
        for pat in patterns:
            matches = [c for c in df.columns if pat in c]
            if matches:
                result[target] = df[matches[0]].values
                break

    result = result.reset_index(drop=True)
    print(f"Analysis dataset: {len(result)} rows, columns: {list(result.columns)}")
    return result


def preprocess_pipeline(data_dir: Path | None = None) -> pd.DataFrame:
    """Full preprocessing pipeline: load → complete cases → analysis dataset.

    Saves processed data to PROCESSED_DATA_DIR.
    """
    df_raw = load_raw_proms(data_dir)
    df_complete = extract_complete_cases(df_raw)
    df_analysis = build_analysis_dataset(df_complete)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DATA_DIR / "proms_hip_complete.parquet"
    df_analysis.to_parquet(out_path, index=False)
    print(f"Saved processed data to {out_path}")

    return df_analysis


if __name__ == "__main__":
    preprocess_pipeline()
