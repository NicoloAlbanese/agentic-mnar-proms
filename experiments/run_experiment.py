"""Main experiment runner for the MNAR sensitivity analysis study.

Phases can be run independently:
    uv run python -m experiments.run_experiment --phase 1
    uv run python -m experiments.run_experiment --phase 2
    uv run python -m experiments.run_experiment --phase 3
    uv run python -m experiments.run_experiment              # all phases

Options:
    --run NAME          Run name (creates results/NAME/ directory)
    --scenario NAME     Synthetic data scenario: "high_effect" or "low_effect"
    --phase {1,2,3}     Run a specific phase

Phase 1: Baseline methods comparison (no AWS needed)
Phase 2: Tipping-point identification (no AWS needed)
Phase 3: Agent-driven analysis (requires Bedrock credentials)
"""

import argparse
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import RANDOM_SEED, RESULTS_DIR
from src.data.synthetic import (
    get_or_create_synthetic, SyntheticConfig,
    SCENARIO_HIGH_EFFECT, SCENARIO_LOW_EFFECT,
)
from src.simulation.mechanisms import generate_mnar_missingness
from src.simulation.scenarios import generate_all_scenarios, SimulationScenario
from src.analysis.baselines import (
    complete_case_analysis, mean_imputation, multiple_imputation_mar,
)
from src.analysis.tipping_point import run_tipping_point_analysis
from src.evaluation.metrics import ReplicationMetrics, compute_aggregated_metrics

warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""

    run_name: str = "run_x"
    scenario: str = "high_effect"

    # Phase 1
    p1_n_replications: int = 30
    p1_sample_size: int = 2000
    p1_delta_grid: list[float] | None = None
    p1_missing_rates: list[float] | None = None

    # Phase 2
    p2_n_replications: int = 20
    p2_sample_size: int = 2000
    p2_true_deltas: list[float] | None = None
    p2_delta_grid: list[float] | None = None
    p2_missing_rates: list[float] | None = None

    # Phase 3
    p3_n_replications: int = 10
    p3_sample_size: int = 2000
    p3_true_delta: float = -1.0
    p3_missing_rate: float = 0.25

    def __post_init__(self):
        if self.p1_delta_grid is None:
            if self.scenario == "low_effect":
                self.p1_delta_grid = [-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0]
            else:
                self.p1_delta_grid = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0]
        if self.p1_missing_rates is None:
            if self.scenario == "low_effect":
                self.p1_missing_rates = [0.30, 0.40]
            else:
                self.p1_missing_rates = [0.20, 0.30]
        if self.p2_true_deltas is None:
            if self.scenario == "low_effect":
                self.p2_true_deltas = [-3.0, -2.0, -1.0]
            else:
                self.p2_true_deltas = [-1.5, -1.0, -0.5]
        if self.p2_delta_grid is None:
            if self.scenario == "low_effect":
                self.p2_delta_grid = [-6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0]
            else:
                self.p2_delta_grid = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0]
        if self.p2_missing_rates is None:
            if self.scenario == "low_effect":
                self.p2_missing_rates = [0.30, 0.40]
            else:
                self.p2_missing_rates = [0.20, 0.30]
        # Phase 3 defaults for low_effect
        if self.scenario == "low_effect":
            if self.p3_true_delta == -1.0:  # only override if still default
                self.p3_true_delta = -2.0
            if self.p3_missing_rate == 0.25:
                self.p3_missing_rate = 0.35

    @property
    def output_dir(self) -> Path:
        return RESULTS_DIR / self.run_name

    @property
    def synthetic_config(self) -> SyntheticConfig:
        if self.scenario == "low_effect":
            return SCENARIO_LOW_EFFECT
        return SCENARIO_HIGH_EFFECT

    def save_parameters(self, df_pop: pd.DataFrame):
        """Save experiment parameters to file."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sc = self.synthetic_config
        gain = (df_pop["ohs_post"] - df_pop["ohs_pre"]).mean()
        gain_sd = (df_pop["ohs_post"] - df_pop["ohs_pre"]).std()

        lines = [
            f"# {self.run_name}: {self.scenario} scenario",
            f"# Date: {time.strftime('%Y-%m-%d %H:%M')}",
            "",
            "[data]",
            f"source = synthetic ({sc.label})",
            f"n_population = {sc.n}",
            f"mean_pre_ohs = {sc.mean_pre}",
            f"sd_pre_ohs = {sc.sd_pre}",
            f"mean_post_ohs = {sc.mean_post}",
            f"sd_post_ohs = {sc.sd_post}",
            f"actual_mean_health_gain = {gain:.2f}",
            f"actual_sd_health_gain = {gain_sd:.2f}",
            f"correlation_pre_post = {sc.correlation}",
            "",
            "[phase1]",
            f"n_replications = {self.p1_n_replications}",
            f"sample_size = {self.p1_sample_size}",
            f"delta_grid = {self.p1_delta_grid}",
            f"missing_rates = {self.p1_missing_rates}",
            "",
            "[phase2]",
            f"n_replications = {self.p2_n_replications}",
            f"sample_size = {self.p2_sample_size}",
            f"true_deltas = {self.p2_true_deltas}",
            f"delta_grid = {self.p2_delta_grid}",
            f"missing_rates = {self.p2_missing_rates}",
            "",
            "[phase3]",
            f"n_replications = {self.p3_n_replications}",
            f"sample_size = {self.p3_sample_size}",
            f"true_delta = {self.p3_true_delta}",
            f"missing_rate = {self.p3_missing_rate}",
            f"model_id = us.anthropic.claude-sonnet-4-20250514-v1:0",
            "",
        ]
        (self.output_dir / "parameters.txt").write_text("\n".join(lines))


def load_population(config: ExperimentConfig) -> pd.DataFrame:
    """Load population data based on scenario config."""
    from src.config import PROCESSED_DATA_DIR

    # Try real data first
    real_path = PROCESSED_DATA_DIR / "proms_hip_real.parquet"
    if real_path.exists() and config.scenario != "low_effect":
        df = pd.read_parquet(real_path)
        print(f"Using REAL NHS PROMs data: {len(df)} patients")
        return df

    # Fall back to synthetic
    return get_or_create_synthetic(config=config.synthetic_config)


def run_single_replication(
    df_population: pd.DataFrame,
    scenario: SimulationScenario,
    rep_idx: int,
) -> list[ReplicationMetrics]:
    """Run a single Monte Carlo replication for one scenario."""
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

    cc = complete_case_analysis(df_sample, observed)
    results.append(ReplicationMetrics(
        true_theta=true_theta, estimate=cc.estimate, se=cc.se,
        ci_lower=cc.ci_lower, ci_upper=cc.ci_upper,
        method="complete_case", delta=scenario.delta,
        missing_rate=actual_missing_rate,
    ))

    mi = mean_imputation(df_sample, observed)
    results.append(ReplicationMetrics(
        true_theta=true_theta, estimate=mi.estimate, se=mi.se,
        ci_lower=mi.ci_lower, ci_upper=mi.ci_upper,
        method="mean_imputation", delta=scenario.delta,
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


# ============================================================
# Phase 1
# ============================================================

def run_phase1(df_population: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    """Phase 1: Baseline methods comparison under MNAR."""
    scenarios = generate_all_scenarios(
        delta_grid=config.p1_delta_grid,
        missing_rates=config.p1_missing_rates,
        n_replications=config.p1_n_replications,
        sample_size=config.p1_sample_size,
    )

    all_results = []
    out_path = config.output_dir / "baseline_results.parquet"

    print(f"\n[Phase 1] {len(scenarios)} scenarios × {config.p1_n_replications} reps")

    for scenario in tqdm(scenarios, desc="Phase 1"):
        for rep in range(config.p1_n_replications):
            rep_results = run_single_replication(df_population, scenario, rep)
            all_results.extend(rep_results)

        records = [
            {"method": r.method, "delta": r.delta, "missing_rate": r.missing_rate,
             "true_theta": r.true_theta, "estimate": r.estimate, "se": r.se,
             "ci_lower": r.ci_lower, "ci_upper": r.ci_upper}
            for r in all_results
        ]
        pd.DataFrame(records).to_parquet(out_path, index=False)

    df_results = pd.DataFrame(records)
    print(f"Saved {len(df_results)} results to {out_path}")
    return df_results


def summarize_results(df_results: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    """Compute aggregated metrics."""
    df_results = df_results.copy()
    df_results["missing_rate"] = (df_results["missing_rate"] * 20).round() / 20

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


# ============================================================
# Phase 2
# ============================================================

def run_phase2(df_population: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    """Phase 2: Tipping-point analysis (manual exhaustive method)."""
    results = []
    total = len(config.p2_true_deltas) * len(config.p2_missing_rates) * config.p2_n_replications
    out_path = config.output_dir / "tipping_point_results.parquet"

    print(f"\n[Phase 2] Tipping-point: {total} replications")

    with tqdm(total=total, desc="Phase 2") as pbar:
        for true_delta in config.p2_true_deltas:
            for miss_rate in config.p2_missing_rates:
                for rep in range(config.p2_n_replications):
                    seed = RANDOM_SEED + rep * 7
                    rep_rng = np.random.default_rng(seed)

                    idx = rep_rng.choice(len(df_population), size=config.p2_sample_size, replace=True)
                    df_sample = df_population.iloc[idx].reset_index(drop=True)
                    true_theta = (df_sample["ohs_post"] - df_sample["ohs_pre"]).mean()

                    observed = generate_mnar_missingness(
                        df=df_sample, delta=true_delta,
                        target_missing_rate=miss_rate, rng=rep_rng,
                    )

                    t0 = time.time()
                    tp_result = run_tipping_point_analysis(
                        df=df_sample, observed=observed,
                        delta_grid=config.p2_delta_grid,
                        n_imputations=10, rng_seed=seed,
                    )
                    wall_time = time.time() - t0

                    results.append({
                        "true_delta": true_delta,
                        "missing_rate": miss_rate,
                        "true_theta": true_theta,
                        "tipping_point": tp_result.tipping_point,
                        "is_robust": tp_result.is_robust,
                        "n_deltas_evaluated": len(config.p2_delta_grid),
                        "wall_time_seconds": wall_time,
                    })
                    pbar.update(1)

                pd.DataFrame(results).to_parquet(out_path, index=False)

    df_tp = pd.DataFrame(results)
    df_tp.to_parquet(out_path, index=False)
    print(f"Saved to {out_path}")
    return df_tp


# ============================================================
# Phase 3
# ============================================================

def run_phase3(df_population: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    """Phase 3: Agent-driven sensitivity analysis."""
    from src.agent.orchestrator import run_agent_sensitivity_analysis

    results = []
    out_path = config.output_dir / "agent_results.parquet"

    print(f"\n[Phase 3] Agent: {config.p3_n_replications} replications")
    print(f"  true_delta={config.p3_true_delta}, missing_rate={config.p3_missing_rate}")

    for rep in tqdm(range(config.p3_n_replications), desc="Phase 3"):
        seed = RANDOM_SEED + rep * 13
        rep_rng = np.random.default_rng(seed)

        idx = rep_rng.choice(len(df_population), size=config.p3_sample_size, replace=True)
        df_sample = df_population.iloc[idx].reset_index(drop=True)

        observed = generate_mnar_missingness(
            df=df_sample, delta=config.p3_true_delta,
            target_missing_rate=config.p3_missing_rate, rng=rep_rng,
        )

        try:
            agent_run = run_agent_sensitivity_analysis(df_sample, observed)
            results.append({
                "rep": rep,
                "true_delta": config.p3_true_delta,
                "missing_rate": config.p3_missing_rate,
                "tipping_point": agent_run.tipping_point,
                "is_robust": agent_run.is_robust,
                "estimate_mar": agent_run.estimate_mar,
                "n_tool_calls": agent_run.n_tool_calls,
                "n_deltas_evaluated": agent_run.n_deltas_evaluated,
                "wall_time_seconds": agent_run.wall_time_seconds,
                "summary": agent_run.final_summary[:500],
            })
        except Exception as e:
            print(f"  Rep {rep} failed: {e}")
            results.append({
                "rep": rep,
                "true_delta": config.p3_true_delta,
                "missing_rate": config.p3_missing_rate,
                "tipping_point": None, "is_robust": None,
                "estimate_mar": 0, "n_tool_calls": 0,
                "n_deltas_evaluated": 0, "wall_time_seconds": 0,
                "summary": f"ERROR: {e}",
            })

        pd.DataFrame(results).to_parquet(out_path, index=False)

    df_agent = pd.DataFrame(results)
    print(f"Saved to {out_path}")
    return df_agent


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="MNAR Sensitivity Analysis Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python -m experiments.run_experiment --run run_2 --scenario low_effect
  uv run python -m experiments.run_experiment --run run_2 --scenario low_effect --phase 3
        """,
    )
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], default=None)
    parser.add_argument("--run", type=str, default="run_x", help="Run name for output directory")
    parser.add_argument("--scenario", type=str, default="high_effect",
                        choices=["high_effect", "low_effect"],
                        help="Synthetic data scenario")
    args = parser.parse_args()

    config = ExperimentConfig(run_name=args.run, scenario=args.scenario)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"MNAR Sensitivity Analysis — {args.run} ({args.scenario})")
    print("=" * 60)

    df_pop = load_population(config)
    true_gain = (df_pop["ohs_post"] - df_pop["ohs_pre"]).mean()
    print(f"Population: {len(df_pop)} records")
    print(f"Mean health gain: {true_gain:.2f} (SD={(df_pop['ohs_post'] - df_pop['ohs_pre']).std():.2f})")

    config.save_parameters(df_pop)

    run_all = args.phase is None

    if run_all or args.phase == 1:
        print("\n" + "=" * 60)
        print("Phase 1: Baseline methods under MNAR")
        print("=" * 60)
        df_results = run_phase1(df_pop, config)
        df_summary = summarize_results(df_results, config)
        print("\nBias by method and delta (20% missing):")
        subset = df_summary[df_summary["missing_rate"].between(0.18, 0.22)]
        if not subset.empty:
            pivot = subset.pivot_table(values="bias", index="delta", columns="method")
            print(pivot.to_string())

    if run_all or args.phase == 2:
        print("\n" + "=" * 60)
        print("Phase 2: Tipping-point analysis (manual)")
        print("=" * 60)
        df_tp = run_phase2(df_pop, config)
        print("\nTipping-point summary:")
        print(df_tp.groupby(["true_delta", "missing_rate"]).agg(
            mean_tp=("tipping_point", "mean"),
            mean_time=("wall_time_seconds", "mean"),
        ).round(3).to_string())

    if run_all or args.phase == 3:
        print("\n" + "=" * 60)
        print("Phase 3: Agent-driven analysis")
        print("=" * 60)
        try:
            import boto3
            boto3.client("bedrock-runtime", region_name="us-east-1")
            print("Bedrock accessible")
            df_agent = run_phase3(df_pop, config)
            print("\nAgent results:")
            cols = ["rep", "tipping_point", "estimate_mar", "n_tool_calls",
                    "n_deltas_evaluated", "wall_time_seconds"]
            available = [c for c in cols if c in df_agent.columns]
            print(df_agent[available].to_string())
        except Exception as e:
            print(f"Phase 3 failed: {e}")

    print("\n" + "=" * 60)
    print(f"Done. Results in: {config.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
