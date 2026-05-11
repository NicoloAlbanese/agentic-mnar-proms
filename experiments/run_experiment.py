"""Main experiment runner for the MNAR sensitivity analysis study.

Compares:
1. Manual exhaustive sensitivity analysis (fixed grid)
2. Agent-driven adaptive sensitivity analysis (Strands + Bedrock)

Both approaches identify the tipping point where the conclusion of
positive health gain reverses under MNAR assumptions.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import (
    RANDOM_SEED, N_SIMULATIONS, DELTA_GRID, MISSINGNESS_RATES,
    RESULTS_DIR, PROCESSED_DATA_DIR,
)
from src.data.synthetic import get_or_create_synthetic
from src.simulation.mechanisms import generate_mnar_missingness
from src.simulation.scenarios import generate_all_scenarios, SimulationScenario
from src.analysis.baselines import (
    complete_case_analysis, mean_imputation, multiple_imputation_mar,
)
from src.analysis.tipping_point import run_tipping_point_analysis
from src.evaluation.metrics import (
    ReplicationMetrics, compute_aggregated_metrics,
    compute_tipping_point_accuracy, TippingPointMetrics,
)


def load_population() -> pd.DataFrame:
    """Load population data (real or synthetic)."""
    real_path = PROCESSED_DATA_DIR / "proms_hip_complete.parquet"
    if real_path.exists():
        df = pd.read_parquet(real_path)
        print(f"Loaded real data: {len(df)} records")
        return df

    print("Real data not found, using synthetic data")
    return get_or_create_synthetic(n=10_000)


def run_single_replication(
    df_population: pd.DataFrame,
    scenario: SimulationScenario,
    rep_idx: int,
) -> list[ReplicationMetrics]:
    """Run a single Monte Carlo replication for one scenario.

    Steps:
    1. Bootstrap sample from population
    2. Compute true theta (full-data mean health gain)
    3. Generate MNAR missingness
    4. Apply each analysis method
    5. Return metrics for each method
    """
    rng = np.random.default_rng(scenario.seed + rep_idx)

    # Bootstrap sample
    sample_idx = rng.choice(len(df_population), size=scenario.sample_size, replace=True)
    df_sample = df_population.iloc[sample_idx].reset_index(drop=True)

    # True theta (full data)
    true_theta = (df_sample["ohs_post"] - df_sample["ohs_pre"]).mean()

    # Generate MNAR missingness
    observed = generate_mnar_missingness(
        df=df_sample,
        delta=scenario.delta,
        target_missing_rate=scenario.missing_rate,
        rng=rng,
    )

    actual_missing_rate = 1.0 - observed.mean()

    # Apply analysis methods
    results = []

    # 1. Complete-case analysis
    cc = complete_case_analysis(df_sample, observed)
    results.append(ReplicationMetrics(
        true_theta=true_theta, estimate=cc.estimate, se=cc.se,
        ci_lower=cc.ci_lower, ci_upper=cc.ci_upper,
        method="complete_case", delta=scenario.delta,
        missing_rate=actual_missing_rate,
    ))

    # 2. Mean imputation
    mi = mean_imputation(df_sample, observed)
    results.append(ReplicationMetrics(
        true_theta=true_theta, estimate=mi.estimate, se=mi.se,
        ci_lower=mi.ci_lower, ci_upper=mi.ci_upper,
        method="mean_imputation", delta=scenario.delta,
        missing_rate=actual_missing_rate,
    ))

    # 3. Multiple imputation under MAR
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


def run_baseline_experiment(
    df_population: pd.DataFrame,
    n_replications: int = 50,
    sample_size: int = 2000,
    delta_grid: list[float] | None = None,
    missing_rates: list[float] | None = None,
) -> pd.DataFrame:
    """Run the baseline simulation experiment.

    Args:
        df_population: Complete-case population
        n_replications: Monte Carlo replications per scenario
        sample_size: Bootstrap sample size per replication
        delta_grid: MNAR delta values to test
        missing_rates: Target missingness rates

    Returns:
        DataFrame with all replication results
    """
    scenarios = generate_all_scenarios(
        delta_grid=delta_grid,
        missing_rates=missing_rates,
        n_replications=n_replications,
        sample_size=sample_size,
    )

    all_results = []
    total = len(scenarios) * n_replications

    print(f"\nRunning {len(scenarios)} scenarios × {n_replications} reps = {total} total")
    print(f"Sample size per rep: {sample_size}")

    for scenario in tqdm(scenarios, desc="Scenarios"):
        for rep in range(n_replications):
            rep_results = run_single_replication(df_population, scenario, rep)
            all_results.extend(rep_results)

    # Convert to DataFrame
    records = [
        {
            "method": r.method,
            "delta": r.delta,
            "missing_rate": r.missing_rate,
            "true_theta": r.true_theta,
            "estimate": r.estimate,
            "se": r.se,
            "ci_lower": r.ci_lower,
            "ci_upper": r.ci_upper,
        }
        for r in all_results
    ]

    df_results = pd.DataFrame(records)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "baseline_results.parquet"
    df_results.to_parquet(out_path, index=False)
    print(f"\nSaved {len(df_results)} results to {out_path}")

    return df_results


def run_tipping_point_experiment(
    df_population: pd.DataFrame,
    n_replications: int = 20,
    sample_size: int = 2000,
    missing_rates: list[float] | None = None,
    delta_grid: list[float] | None = None,
) -> pd.DataFrame:
    """Run tipping-point identification experiment.

    For each replication:
    - Generate MNAR data at a known delta
    - Run manual tipping-point analysis
    - Record accuracy and efficiency

    This provides the ground truth for comparing against the agent.
    """
    if delta_grid is None:
        delta_grid = DELTA_GRID
    if missing_rates is None:
        missing_rates = [0.20, 0.30]

    # Use a subset of true deltas where tipping points are likely
    true_deltas = [-1.5, -1.0, -0.5]

    results = []
    rng = np.random.default_rng(RANDOM_SEED)

    total = len(true_deltas) * len(missing_rates) * n_replications
    print(f"\nTipping-point experiment: {total} replications")

    with tqdm(total=total, desc="Tipping-point") as pbar:
        for true_delta in true_deltas:
            for miss_rate in missing_rates:
                for rep in range(n_replications):
                    seed = RANDOM_SEED + rep * 7
                    rep_rng = np.random.default_rng(seed)

                    # Sample
                    idx = rep_rng.choice(len(df_population), size=sample_size, replace=True)
                    df_sample = df_population.iloc[idx].reset_index(drop=True)

                    # True theta
                    true_theta = (df_sample["ohs_post"] - df_sample["ohs_pre"]).mean()

                    # Generate missingness at true_delta
                    observed = generate_mnar_missingness(
                        df=df_sample,
                        delta=true_delta,
                        target_missing_rate=miss_rate,
                        rng=rep_rng,
                    )

                    # Run manual tipping-point analysis
                    t0 = time.time()
                    tp_result = run_tipping_point_analysis(
                        df=df_sample,
                        observed=observed,
                        delta_grid=delta_grid,
                        n_imputations=10,
                        rng_seed=seed,
                    )
                    wall_time = time.time() - t0

                    results.append({
                        "true_delta": true_delta,
                        "missing_rate": miss_rate,
                        "true_theta": true_theta,
                        "tipping_point": tp_result.tipping_point,
                        "is_robust": tp_result.is_robust,
                        "n_deltas_evaluated": len(delta_grid),
                        "wall_time_seconds": wall_time,
                    })

                    pbar.update(1)

    df_tp = pd.DataFrame(results)
    out_path = RESULTS_DIR / "tipping_point_results.parquet"
    df_tp.to_parquet(out_path, index=False)
    print(f"Saved tipping-point results to {out_path}")

    return df_tp


def summarize_results(df_results: pd.DataFrame) -> pd.DataFrame:
    """Compute aggregated metrics from raw replication results."""
    summaries = []

    for (method, delta, rate), group in df_results.groupby(
        ["method", "delta", "missing_rate"]
    ):
        replications = [
            ReplicationMetrics(
                true_theta=row["true_theta"],
                estimate=row["estimate"],
                se=row["se"],
                ci_lower=row["ci_lower"],
                ci_upper=row["ci_upper"],
                method=method,
                delta=delta,
                missing_rate=rate,
            )
            for _, row in group.iterrows()
        ]

        agg = compute_aggregated_metrics(replications)
        summaries.append({
            "method": agg.method,
            "delta": agg.delta,
            "missing_rate": agg.missing_rate,
            "n_reps": agg.n_replications,
            "bias": round(agg.bias, 4),
            "rmse": round(agg.rmse, 4),
            "coverage": round(agg.coverage, 4),
            "mean_ci_width": round(agg.mean_ci_width, 4),
        })

    df_summary = pd.DataFrame(summaries)

    out_path = RESULTS_DIR / "summary_metrics.parquet"
    df_summary.to_parquet(out_path, index=False)
    print(f"Saved summary to {out_path}")

    return df_summary


def run_agent_experiment(
    df_population: pd.DataFrame,
    n_replications: int = 10,
    sample_size: int = 2000,
    missing_rate: float = 0.25,
    true_delta: float = -1.0,
) -> pd.DataFrame:
    """Run the agent-driven sensitivity analysis experiment.

    For each replication, the Strands agent autonomously explores the
    delta parameter space and identifies the tipping point.

    Requires AWS credentials configured for Bedrock access.
    """
    from src.agent.orchestrator import run_agent_sensitivity_analysis

    results = []
    rng = np.random.default_rng(RANDOM_SEED)

    print(f"\nAgent experiment: {n_replications} replications")
    print(f"True delta={true_delta}, missing_rate={missing_rate}")

    for rep in tqdm(range(n_replications), desc="Agent runs"):
        seed = RANDOM_SEED + rep * 13
        rep_rng = np.random.default_rng(seed)

        # Sample
        idx = rep_rng.choice(len(df_population), size=sample_size, replace=True)
        df_sample = df_population.iloc[idx].reset_index(drop=True)

        # Generate missingness
        observed = generate_mnar_missingness(
            df=df_sample,
            delta=true_delta,
            target_missing_rate=missing_rate,
            rng=rep_rng,
        )

        # Run agent
        try:
            agent_run = run_agent_sensitivity_analysis(df_sample, observed)
            results.append({
                "rep": rep,
                "true_delta": true_delta,
                "missing_rate": missing_rate,
                "tipping_point": agent_run.tipping_point,
                "is_robust": agent_run.is_robust,
                "n_tool_calls": agent_run.n_tool_calls,
                "wall_time_seconds": agent_run.wall_time_seconds,
                "summary": agent_run.final_summary[:500],
            })
        except Exception as e:
            print(f"  Rep {rep} failed: {e}")
            results.append({
                "rep": rep,
                "true_delta": true_delta,
                "missing_rate": missing_rate,
                "tipping_point": None,
                "is_robust": None,
                "n_tool_calls": 0,
                "wall_time_seconds": 0,
                "summary": f"ERROR: {e}",
            })

    df_agent = pd.DataFrame(results)
    out_path = RESULTS_DIR / "agent_results.parquet"
    df_agent.to_parquet(out_path, index=False)
    print(f"Saved agent results to {out_path}")

    return df_agent


def main():
    """Run the full experiment pipeline."""
    print("=" * 60)
    print("MNAR Sensitivity Analysis - Simulation Study")
    print("=" * 60)

    # Load data
    df_pop = load_population()
    true_gain = (df_pop["ohs_post"] - df_pop["ohs_pre"]).mean()
    print(f"Population: {len(df_pop)} complete cases")
    print(f"Mean health gain: {true_gain:.2f}")
    print(f"SD health gain: {(df_pop['ohs_post'] - df_pop['ohs_pre']).std():.2f}")

    # --- Phase 1: Baseline methods comparison ---
    print("\n" + "=" * 60)
    print("Phase 1: Baseline methods under MNAR")
    print("=" * 60)

    # Use reduced grid for faster initial run
    quick_deltas = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0]
    quick_rates = [0.20, 0.30]

    df_results = run_baseline_experiment(
        df_pop,
        n_replications=30,
        sample_size=2000,
        delta_grid=quick_deltas,
        missing_rates=quick_rates,
    )

    df_summary = summarize_results(df_results)
    print("\nBias by method and delta (20% missing):")
    subset = df_summary[df_summary["missing_rate"].between(0.18, 0.22)]
    pivot = subset.pivot_table(values="bias", index="delta", columns="method")
    print(pivot.to_string())

    # --- Phase 2: Tipping-point identification ---
    print("\n" + "=" * 60)
    print("Phase 2: Tipping-point analysis (manual)")
    print("=" * 60)

    df_tp = run_tipping_point_experiment(
        df_pop,
        n_replications=10,
        sample_size=2000,
        delta_grid=quick_deltas,
        missing_rates=quick_rates,
    )

    print("\nTipping-point summary:")
    tp_summary = df_tp.groupby(["true_delta", "missing_rate"]).agg(
        mean_tp=("tipping_point", "mean"),
        mean_time=("wall_time_seconds", "mean"),
    ).round(3)
    print(tp_summary.to_string())

    # --- Phase 3: Agent comparison (requires Bedrock) ---
    print("\n" + "=" * 60)
    print("Phase 3: Agent-driven analysis")
    print("=" * 60)

    try:
        import boto3
        # Quick check if Bedrock is accessible
        client = boto3.client("bedrock-runtime", region_name="us-east-1")
        print("Bedrock accessible - running agent experiment")

        df_agent = run_agent_experiment(
            df_pop,
            n_replications=5,
            sample_size=2000,
        )

        print("\nAgent results:")
        print(df_agent[["rep", "tipping_point", "n_tool_calls", "wall_time_seconds"]].to_string())

    except Exception as e:
        print(f"Skipping agent experiment (Bedrock not available): {e}")
        print("Run with AWS credentials configured for Bedrock access.")

    print("\n" + "=" * 60)
    print("Experiment complete. Results saved to:", RESULTS_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
