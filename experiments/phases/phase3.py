"""Phase 3: Agent-driven sensitivity analysis.

Tests the Strands agent across multiple (true_delta, missing_rate) combinations
to evaluate generalizability. Requires AWS Bedrock credentials.

Supports resume: if agent_results.parquet exists, completed runs are skipped.
"""

import pandas as pd
from tqdm import tqdm

from src.config import RANDOM_SEED
from experiments.config import ExperimentConfig
from experiments.utils import draw_sample_with_missingness, save_incremental


def run_phase3(df_population: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    """Run Phase 3: agent-driven analysis. Resumes from existing results."""
    from src.agent.orchestrator import run_agent_sensitivity_analysis

    out_path = config.output_dir / "agent_results.parquet"

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
        for true_delta in config.p3_true_deltas
        for miss_rate in config.missing_rates
        for rep in range(config.p3_n_replications)
    ]
    total = len(all_jobs)

    # Skip already-completed jobs (by position — deterministic order)
    jobs_to_run = all_jobs[n_existing:]

    print(f"\n[Phase 3] Agent: {total} total runs")
    print(f"  true_deltas={config.p3_true_deltas}")
    print(f"  missing_rates={config.missing_rates}")
    print(f"  replications={config.p3_n_replications}")
    if n_existing > 0:
        print(f"  Resuming: {n_existing}/{total} already done")

    if not jobs_to_run:
        print("  All runs already completed.")
        return pd.DataFrame(results)

    with tqdm(total=total, desc="Phase 3", initial=n_existing) as pbar:
        for true_delta, miss_rate, rep in jobs_to_run:
            seed = RANDOM_SEED + rep * 13

            df_sample, observed, _, _ = draw_sample_with_missingness(
                df_population, true_delta, miss_rate, config.sample_size, seed,
            )

            try:
                agent_run = run_agent_sensitivity_analysis(df_sample, observed)
                results.append({
                    "rep": rep,
                    "true_delta": true_delta,
                    "missing_rate": miss_rate,
                    "tipping_point": agent_run.tipping_point,
                    "is_robust": agent_run.is_robust,
                    "estimate_mar": agent_run.estimate_mar,
                    "n_tool_calls": agent_run.n_tool_calls,
                    "n_deltas_evaluated": agent_run.n_deltas_evaluated,
                    "wall_time_seconds": agent_run.wall_time_seconds,
                    "summary": agent_run.final_summary[:500],
                })
            except Exception as e:
                print(f"\n  Rep {rep} (delta={true_delta}, miss={miss_rate}) failed: {e}")
                results.append({
                    "rep": rep,
                    "true_delta": true_delta,
                    "missing_rate": miss_rate,
                    "tipping_point": None, "is_robust": None,
                    "estimate_mar": 0, "n_tool_calls": 0,
                    "n_deltas_evaluated": 0, "wall_time_seconds": 0,
                    "summary": f"ERROR: {e}",
                })

            pbar.update(1)
            save_incremental(results, out_path)

    df_agent = pd.DataFrame(results)
    print(f"Saved to {out_path}")
    return df_agent
