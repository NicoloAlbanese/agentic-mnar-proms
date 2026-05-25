"""Main experiment runner — CLI entry point.

Usage:
    uv run python -m experiments.runner --phase 1
    uv run python -m experiments.runner --run run_4 --scenario low_effect
    uv run python -m experiments.runner --run run_4 --sample-size 3000 --reps 50
"""

import argparse
import sys
import warnings
from pathlib import Path

from src.config import BEDROCK_REGION
from experiments.config import ExperimentConfig
from experiments.utils import load_population
from experiments.phases import run_phase1, run_phase2, run_phase3, summarize_results

warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")


class TeeWriter:
    """Write to both stdout/stderr and a log file simultaneously."""

    def __init__(self, log_path: Path):
        self._terminal = sys.stdout
        self._log = open(log_path, "w", encoding="utf-8")

    def write(self, message: str):
        self._terminal.write(message)
        # Strip carriage returns for clean log (tqdm uses \r for progress)
        clean = message.replace("\r", "")
        if clean.strip():
            self._log.write(clean)
            self._log.flush()

    def flush(self):
        self._terminal.flush()
        self._log.flush()

    def fileno(self):
        return self._terminal.fileno()

    def isatty(self):
        return self._terminal.isatty()

    def close(self):
        self._log.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MNAR Sensitivity Analysis Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python -m experiments.runner --run run_4 --scenario low_effect
  uv run python -m experiments.runner --run run_4 --phase 3
  uv run python -m experiments.runner --run run_4 --sample-size 3000 --reps 50
        """,
    )
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], default=None,
                        help="Run a specific phase (default: all)")
    parser.add_argument("--run", type=str, default="run_x",
                        help="Run name for output directory")
    parser.add_argument("--scenario", type=str, default="high_effect",
                        choices=["high_effect", "low_effect"],
                        help="Synthetic data scenario")
    parser.add_argument("--sample-size", type=int, default=2000,
                        help="Sample size per replication (default: 2000)")
    parser.add_argument("--reps", type=int, default=None,
                        help="Override replication count for all phases")
    parser.add_argument("--n-imputations", type=int, default=20,
                        help="Number of MI imputations (default: 20)")
    return parser


def main():
    args = _build_parser().parse_args()

    config = ExperimentConfig(
        run_name=args.run,
        scenario=args.scenario,
        sample_size=args.sample_size,
        n_imputations=args.n_imputations,
    )
    if args.reps is not None:
        config.p1_n_replications = args.reps
        config.p2_n_replications = args.reps
        config.p3_n_replications = args.reps

    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Tee all output to a log file in the results directory
    log_path = config.output_dir / "experiment.log"
    tee = TeeWriter(log_path)
    sys.stdout = tee
    sys.stderr = tee

    try:
        _run(config, args)
    finally:
        sys.stdout = tee._terminal
        sys.stderr = tee._terminal
        tee.close()
        print(f"Log saved to: {log_path}")


def _run(config: ExperimentConfig, args):
    """Core experiment logic."""
    print("=" * 60)
    print(f"MNAR Sensitivity Analysis — {config.run_name} ({config.scenario})")
    print("=" * 60)

    df_pop = load_population(config)
    true_gain = (df_pop["ohs_post"] - df_pop["ohs_pre"]).mean()
    gain_sd = (df_pop["ohs_post"] - df_pop["ohs_pre"]).std()
    print(f"Population: {len(df_pop)} records")
    print(f"Mean health gain: {true_gain:.2f} (SD={gain_sd:.2f})")
    print(f"Delta grid: {config.delta_grid}")
    print(f"Missing rates: {config.missing_rates}")
    print(f"Sample size: {config.sample_size}, Imputations: {config.n_imputations}")

    config.save_parameters(df_pop)

    run_all = args.phase is None

    if run_all or args.phase == 1:
        print("\n" + "=" * 60)
        print("Phase 1: Baseline methods under MNAR")
        print("=" * 60)
        df_results = run_phase1(df_pop, config)
        df_summary = summarize_results(df_results, config)
        _print_phase1_summary(df_summary)

    if run_all or args.phase == 2:
        print("\n" + "=" * 60)
        print("Phase 2: Tipping-point analysis (manual)")
        print("=" * 60)
        df_tp = run_phase2(df_pop, config)
        _print_phase2_summary(df_tp)

    if run_all or args.phase == 3:
        print("\n" + "=" * 60)
        print("Phase 3: Agent-driven analysis")
        print("=" * 60)
        _run_phase3_with_check(df_pop, config)

    print("\n" + "=" * 60)
    print(f"Done. Results in: {config.output_dir}")
    print("=" * 60)


def _print_phase1_summary(df_summary):
    """Print a readable bias table for the middle missing rate."""
    print("\nBias by method and delta (25% missing):")
    subset = df_summary[df_summary["missing_rate"].between(0.23, 0.27)]
    if not subset.empty:
        pivot = subset.pivot_table(values="bias", index="delta", columns="method")
        print(pivot.to_string())


def _print_phase2_summary(df_tp):
    """Print tipping-point summary grouped by scenario."""
    print("\nTipping-point summary:")
    print(df_tp.groupby(["true_delta", "missing_rate"]).agg(
        mean_tp=("tipping_point", "mean"),
        mean_time=("wall_time_seconds", "mean"),
    ).round(3).to_string())


def _run_phase3_with_check(df_pop, config):
    """Run Phase 3 with Bedrock connectivity check."""
    try:
        import boto3
        boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)
        print("Bedrock accessible")
    except Exception as e:
        print(f"Phase 3 requires AWS Bedrock access. Error: {e}")
        return

    df_agent = run_phase3(df_pop, config)
    print("\nAgent results:")
    cols = ["rep", "true_delta", "missing_rate", "tipping_point",
            "estimate_mar", "n_tool_calls", "n_deltas_evaluated",
            "wall_time_seconds"]
    available = [c for c in cols if c in df_agent.columns]
    print(df_agent[available].to_string())


if __name__ == "__main__":
    main()
