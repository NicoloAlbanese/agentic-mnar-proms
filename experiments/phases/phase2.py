"""Phase 2: Tipping-point identification (manual exhaustive method).

Evaluates the manual grid-search tipping-point approach across multiple
(true_delta, missing_rate) combinations. No AWS services required.

Supports resume: if tipping_point_results.parquet exists, completed
(true_delta, missing_rate, rep) combinations are skipped.
"""

import time

import pandas as pd
from tqdm import tqdm

from src.config import RANDOM_SEED
from src.analysis.tipping_point import run_tipping_point_analysis
from experiments.config import ExperimentConfig
from experiments.utils import draw_sample_with_missingness, save_incremental


def _load_completed_keys(out_path) -> set[tuple[float, float, int]]:
    """Load completed (true_delta, missing_rate, rep_seed) from existing results."""
    if not out_path.exists():
        return set()

    df = pd.read_parquet(out_path)
    if df.empty:
        return set()

    # We identify each run by (true_delta, missing_rate, true_theta) but
    # true_theta varies slightly. Use row index position within group instead.
    # Simpler: count how many rows exist per (true_delta, missing_rate) group.
    completed = set()
    for _, row in df.iterrows():
        # Reconstruct the rep index from position within its group
        key = (round(row["true_delta"], 4), round(row["missing_rate"], 4))
        completed.add(key)

    return completed


def run_phase2(df_population: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    """Run Phase 2: manual tipping-point analysis. Resumes from existing results."""
    out_path = config.output_dir / "tipping_point_results.parquet"

    # Resume: load existing results
    if out_path.exists():
        df_existing = pd.read_parquet(out_path)
        results = df_existing.to_dict("records")
        n_existing = len(results)
    else:
        results = []
        n_existing = 0

    # Build the full list of jobs as (true_delta, miss_rate, rep) tuples
    all_jobs = [
        (true_delta, miss_rate, rep)
        for true_delta in config.p2_true_deltas
        for miss_rate in config.missing_rates
        for rep in range(config.p2_n_replications)
    ]
    total = len(all_jobs)

    # Skip already-completed jobs (by position — deterministic order)
    jobs_to_run = all_jobs[n_existing:]

    print(f"\n[Phase 2] Tipping-point: {total} replications")
    if n_existing > 0:
        print(f"  Resuming: {n_existing}/{total} already done")

    if not jobs_to_run:
        print("  All replications already completed.")
        return pd.DataFrame(results)

    with tqdm(total=total, desc="Phase 2", initial=n_existing) as pbar:
        for true_delta, miss_rate, rep in jobs_to_run:
            seed = RANDOM_SEED + rep * 7

            df_sample, observed, true_theta, _ = draw_sample_with_missingness(
                df_population, true_delta, miss_rate, config.sample_size, seed,
            )

            t0 = time.time()
            tp_result = run_tipping_point_analysis(
                df=df_sample, observed=observed,
                delta_grid=config.delta_grid,
                n_imputations=config.n_imputations,
                rng_seed=seed,
            )
            wall_time = time.time() - t0

            results.append({
                "true_delta": true_delta,
                "missing_rate": miss_rate,
                "true_theta": true_theta,
                "tipping_point": tp_result.tipping_point,
                "is_robust": tp_result.is_robust,
                "n_deltas_evaluated": len(config.delta_grid),
                "wall_time_seconds": wall_time,
            })
            pbar.update(1)

            # Save after every rep for crash safety
            save_incremental(results, out_path)

    df_tp = pd.DataFrame(results)
    df_tp.to_parquet(out_path, index=False)
    print(f"Saved to {out_path}")
    return df_tp
