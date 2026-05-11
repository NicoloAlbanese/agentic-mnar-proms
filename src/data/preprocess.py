"""Preprocess NHS PROMs data to extract complete-case cohort.

Expects the Hip Replacement Provider CSV from NHS Digital PROMs 2023/24.
Place in: data/raw/Hip Replacement Provider 2324 upload.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR


RAW_FILE = RAW_DATA_DIR / "Hip Replacement Provider 2324 upload.csv"

PRE_SCORE_COL = "Hip Replacement Pre-Op Q Score"
POST_SCORE_COL = "Hip Replacement Post-Op Q Score"


def load_raw_proms() -> pd.DataFrame:
    """Load raw NHS PROMs CSV."""
    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Raw data not found at {RAW_FILE}\n"
            "Download from: https://digital.nhs.uk/data-and-information/"
            "publications/statistical/patient-reported-outcome-measures-proms/"
            "final-2023-24-data\n"
            "File: 'CSV Hip replacement Provider' → data/raw/"
        )
    df = pd.read_csv(RAW_FILE, low_memory=False)
    print(f"Loaded {len(df)} records from {RAW_FILE.name}")
    return df


def build_analysis_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Build analysis dataset from raw PROMs data.

    Selects complete cases (both pre and post OHS available),
    primary hip replacements only, and extracts covariates.

    Returns DataFrame with: ohs_pre, ohs_post, ohs_change, age_band, gender
    """
    # Filter to primary hip replacements (revision flag = 0)
    if "Revision Flag" in df.columns:
        df = df[df["Revision Flag"] == 0].copy()
        print(f"  Primary replacements: {len(df)}")

    # Complete cases: both pre and post OHS scores available
    complete = df.dropna(subset=[PRE_SCORE_COL, POST_SCORE_COL]).copy()
    print(f"  Complete OHS cases: {len(complete)}")

    # Remove suppressed rows (marked with *)
    # Age Band and Gender use * for suppressed small counts
    complete = complete[complete["Age Band"] != "*"].copy()
    print(f"  After removing suppressed: {len(complete)}")

    # Build output
    result = pd.DataFrame({
        "ohs_pre": complete[PRE_SCORE_COL].astype(float).values,
        "ohs_post": complete[POST_SCORE_COL].astype(float).values,
    })
    result["ohs_change"] = result["ohs_post"] - result["ohs_pre"]

    # Age band
    if "Age Band" in complete.columns:
        result["age_band"] = complete["Age Band"].values

    # Gender (1=Male, 2=Female in NHS data)
    if "Gender" in complete.columns:
        gender_map = {"1": "Male", "2": "Female"}
        result["gender"] = complete["Gender"].astype(str).map(gender_map).values

    result = result.dropna(subset=["age_band", "gender"]).reset_index(drop=True)
    print(f"  Final dataset: {len(result)} rows")
    print(f"  Mean gain: {result['ohs_change'].mean():.2f} (SD={result['ohs_change'].std():.2f})")

    return result


def preprocess_pipeline() -> pd.DataFrame:
    """Full pipeline: load → filter → build analysis dataset → save."""
    df_raw = load_raw_proms()
    df_analysis = build_analysis_dataset(df_raw)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DATA_DIR / "proms_hip_real.parquet"
    df_analysis.to_parquet(out_path, index=False)
    print(f"  Saved to {out_path}")

    return df_analysis


if __name__ == "__main__":
    preprocess_pipeline()
