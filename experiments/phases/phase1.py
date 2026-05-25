"""Phase 1: Baseline methods comparison under MNAR.

Evaluates complete-case, mean imputation, and MI-MAR across the full
delta × missing_rate grid. No AWS services required.

Supports resume: if baseline_results.parquet exists, completed scenarios
are skipped automatically.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.analysis.baselines import (
    complete_case_analysis, mean_imputation, multiple_imputation_mar,
)
from src.evaluation.metrics import ReplicationMetrics, compute_aggregated_metrics
from src.simulation.mechanisms import generate_mnar_missingness
from src.simulation.scenarios import generate_all_scenarios, SimulationScenario
from experiments.config import ExperimentConfig


def _run_single_replication(
    df_population: pd.DataFrame,
    scenario: SimulationScenario,
    rep_idx: int,
) -> list[ReplicationMetrics]:
    """Run all baseline methods on one bootstrap sample."""
    rng = np.random.default_rng(scenario.seed + rep_idx)

    sample_idx = rng.choice(len(df_population), size=scenario.sample_size, replace=True)
    df_sample = df_population.iloc[sample_idx].reset_index(drop=True)

    true_theta = (df_sample["ohs_post"] - df_sample["ohs_pre"]).mean()

    observed = generate_mnar_missingness(
        df=df_sample,
        delta=scenario.delta,
        target_missing_rate=scenario.missing_rate,
        rng=rng,
    )

    actual_missing_rate = 1.0 - observed.mean()
    results = []

    for method_fn, method_name in [
        (complete_case_analysis, "complete_case"),
        (mean_imputation, "mean_imputation"),
    ]:
        r = method_fn(df_sample, observed)
        results.append(ReplicationMetrics(
            true_theta=true_theta, estimate=r.estimate, se=r.se,
            ci_lower=r.ci_lower, ci_upper=r.ci_upper,
            method=method_name, delta=scenario.delta,
            missing_rate=actual_missing_rate,
        ))

    mar = multiple_imputation_mar(
        df_sample, observed, rng_seed=scenario.seed + rep_idx
    )
    results.append(ReplicationMetrics(
        true_theta=true_theta, estimate=mar.estimate, se=mar.se,
        ci_lower=mar.ci_lower, ci_upper=mar.ci_upper,
        method="mi_mar", delta=scenario.delta,
        missing_rate=actual_missing_rate,
    ))

    return results


def _load_completed_scenarios(out_path) -> set[tuple[float, float]]:
    """Load already-completed (delta, missing_rate) pairs from existing results.

    A scenario is considered complete if it has results for all 3 methods.
    """
    if not out_path.exists():
        return set()

    df = pd.read_parquet(out_path)
    if df.empty:
        return set()

    # Group by delta and count unique methods — a complete scenario has 3 methods
    # We use the nominal delta (from the scenario) not the actual missing rate
    completed = set()
    for delta, group in df.groupby("delta"):
        # Each scenario produces 3 rows per rep (one per method)
        # A full scenario = n_reps * 3 rows
        n_methods = group["method"].nunique()
        if n_methods == 3:
            # Check each missing_rate bucket within this delta
            for _, sub in group.groupby(group["missing_rate"].round(2)):
                if sub["method"].nunique() == 3:
                    # Use the delta and the rounded missing_rate as key
                    rate = sub["missing_rate"].median()
                    completed.add((round(delta, 4), round(rate, 2)))

    return completed


def run_phase1(df_population: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    """Run Phase 1: baseline methods comparison. Resumes from existing results."""
    scenarios = generate_all_scenarios(
        delta_grid=config.delta_grid,
        missing_rates=config.missing_rates,
        sample_size=config.sample_size,
    )

    out_path = config.output_dir / "baseline_results.parquet"

    # Resume: load existing results
    if out_path.exists():
        df_existing = pd.read_parquet(out_path)
        existing_records = df_existing.to_dict("records")
        n_existing = len(df_existing)
        # Count completed scenarios (each scenario = n_reps * 3 methods)
        reps_per_scenario = config.p1_n_replications * 3
        n_completed_scenarios = n_existing // reps_per_scenario if reps_per_scenario > 0 else 0
        print(f"  Resuming: {n_completed_scenarios}/{len(scenarios)} scenarios already done")
    else:
        existing_records = []
        n_completed_scenarios = 0

    # Skip already-completed scenarios
    scenarios_to_run = scenarios[n_completed_scenarios:]

    all_results: list[ReplicationMetrics] = []
    total = len(scenarios)
    print(f"\n[Phase 1] {total} scenarios × {config.p1_n_replications} reps")

    if not scenarios_to_run:
        print("  All scenarios already completed.")
        return pd.DataFrame(existing_records)

    for scenario in tqdm(scenarios_to_run, desc="Phase 1",
                         initial=n_completed_scenarios, total=total):
        for rep in range(config.p1_n_replications):
            rep_results = _run_single_replication(df_population, scenario, rep)
            all_results.extend(rep_results)

        # Incremental save: existing + new
        new_records = [
            {"method": r.method, "delta": r.delta, "missing_rate": r.missing_rate,
             "true_theta": r.true_theta, "estimate": r.estimate, "se": r.se,
             "ci_lower": r.ci_lower, "ci_upper": r.ci_upper}
            for r in all_results
        ]
        pd.DataFrame(existing_records + new_records).to_parquet(out_path, index=False)

    all_records = existing_records + new_records
    df_results = pd.DataFrame(all_records)
    print(f"Saved {len(df_results)} results to {out_path}")
    return df_results


def summarize_results(df_results: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    """Compute aggregated metrics (bias, RMSE, coverage) per method × scenario."""
    df_results = df_results.copy()

    # Round missing rates to nearest target for grouping
    targets = np.array(config.missing_rates)
    df_results["missing_rate"] = df_results["missing_rate"].apply(
        lambda x: targets[np.argmin(np.abs(targets - x))]
    )

    summaries = []
    for (method, delta, rate), group in df_results.groupby(
        ["method", "delta", "missing_rate"]
    ):
        replications = [
            ReplicationMetrics(
                true_theta=row["true_theta"], estimate=row["estimate"],
                se=row["se"], ci_lower=row["ci_lower"], ci_upper=row["ci_upper"],
                method=method, delta=delta, missing_rate=rate,
            )
            for _, row in group.iterrows()
        ]
        agg = compute_aggregated_metrics(replications)
        summaries.append({
            "method": agg.method, "delta": agg.delta,
            "missing_rate": agg.missing_rate, "n_reps": agg.n_replications,
            "bias": round(agg.bias, 4), "rmse": round(agg.rmse, 4),
            "coverage": round(agg.coverage, 4), "mean_ci_width": round(agg.mean_ci_width, 4),
        })

    df_summary = pd.DataFrame(summaries)
    out_path = config.output_dir / "summary_metrics.parquet"
    df_summary.to_parquet(out_path, index=False)
    print(f"Saved summary to {out_path}")
    return df_summary
