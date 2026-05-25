"""Scenario configurations for the simulation study."""

from dataclasses import dataclass
from itertools import product

from src.config import DELTA_GRID, MISSINGNESS_RATES, RANDOM_SEED


@dataclass
class SimulationScenario:
    """A single simulation scenario (one delta × missing_rate combination)."""

    delta: float
    missing_rate: float
    sample_size: int
    seed: int

    @property
    def label(self) -> str:
        return f"delta={self.delta:.2f}_miss={self.missing_rate:.0%}"


def generate_all_scenarios(
    delta_grid: list[float] | None = None,
    missing_rates: list[float] | None = None,
    sample_size: int = 2000,
    base_seed: int = RANDOM_SEED,
) -> list[SimulationScenario]:
    """Generate all scenario combinations for the experiment.

    Returns list of SimulationScenario objects (one per delta × missing_rate combo).
    """
    if delta_grid is None:
        delta_grid = DELTA_GRID
    if missing_rates is None:
        missing_rates = MISSINGNESS_RATES

    scenarios = []
    for i, (delta, rate) in enumerate(product(delta_grid, missing_rates)):
        scenarios.append(
            SimulationScenario(
                delta=delta,
                missing_rate=rate,
                sample_size=sample_size,
                seed=base_seed + i * 1000,
            )
        )

    return scenarios
