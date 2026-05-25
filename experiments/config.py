"""Experiment configuration dataclass.

Design principles:
- A SINGLE delta grid is used across all phases for consistency.
  The grid spans [-5, +1] which covers clinically plausible MNAR
  departures for both high and low effect scenarios.
- The same missing rates are used across phases.
- true_deltas for Phase 2/3 are chosen to test different regimes
  (mild, moderate, severe MNAR) and are NOT required to be in the
  search grid — the tipping point is an emergent property, not the
  true_delta itself.
- n_imputations is consistent across all phases.
"""

import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import RESULTS_DIR, BEDROCK_MODEL_ID
from src.data.synthetic import SyntheticConfig, SCENARIO_HIGH_EFFECT, SCENARIO_LOW_EFFECT


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""

    run_name: str = "run_x"
    scenario: str = "high_effect"

    # Shared across phases — single consistent grid
    delta_grid: list[float] | None = None
    missing_rates: list[float] | None = None
    sample_size: int = 2000
    n_imputations: int = 20

    # Phase 1: Baseline methods comparison
    p1_n_replications: int = 30

    # Phase 2: Tipping-point identification (manual)
    p2_n_replications: int = 30
    p2_true_deltas: list[float] | None = None

    # Phase 3: Agent-driven analysis
    p3_n_replications: int = 10
    p3_true_deltas: list[float] | None = None

    def __post_init__(self):
        if self.delta_grid is None:
            self.delta_grid = [-5.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0]

        if self.missing_rates is None:
            self.missing_rates = [0.15, 0.25, 0.35]

        if self.p2_true_deltas is None:
            self.p2_true_deltas = [-3.0, -2.0, -1.0, -0.5]

        if self.p3_true_deltas is None:
            self.p3_true_deltas = [-2.0, -1.0, -0.5]

    @property
    def output_dir(self) -> Path:
        return RESULTS_DIR / self.run_name

    @property
    def synthetic_config(self) -> SyntheticConfig:
        if self.scenario == "low_effect":
            return SCENARIO_LOW_EFFECT
        return SCENARIO_HIGH_EFFECT

    def save_parameters(self, df_pop: pd.DataFrame):
        """Save experiment parameters to file for reproducibility."""
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
            "[shared_design]",
            f"delta_grid = {self.delta_grid}",
            f"missing_rates = {self.missing_rates}",
            f"sample_size = {self.sample_size}",
            f"n_imputations = {self.n_imputations}",
            "",
            "[phase1]",
            f"n_replications = {self.p1_n_replications}",
            "",
            "[phase2]",
            f"n_replications = {self.p2_n_replications}",
            f"true_deltas = {self.p2_true_deltas}",
            "",
            "[phase3]",
            f"n_replications = {self.p3_n_replications}",
            f"true_deltas = {self.p3_true_deltas}",
            f"model_id = {BEDROCK_MODEL_ID}",
            "",
        ]
        (self.output_dir / "parameters.txt").write_text("\n".join(lines))
