"""Agent orchestration for autonomous sensitivity analysis."""

import time
import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from strands import Agent

from src.agent.prompts import SENSITIVITY_ANALYST_PROMPT
from src.agent.tools import (
    set_analysis_context,
    run_baseline_analyses,
    run_sensitivity_at_delta,
    run_tipping_point_grid,
    refine_tipping_point,
    get_clinical_context,
)
from src.config import BEDROCK_MODEL_ID, BEDROCK_REGION


@dataclass
class AgentRun:
    """Record of a single agent sensitivity analysis run."""

    tipping_point: float | None = None
    is_robust: bool = False
    estimate_mar: float = 0.0
    n_tool_calls: int = 0
    n_deltas_evaluated: int = 0
    wall_time_seconds: float = 0.0
    reasoning_trace: list[str] = field(default_factory=list)
    final_summary: str = ""


def create_sensitivity_agent() -> Agent:
    """Create a Strands agent configured for sensitivity analysis."""
    agent = Agent(
        model=f"bedrock/{BEDROCK_MODEL_ID}",
        system_prompt=SENSITIVITY_ANALYST_PROMPT,
        tools=[
            run_baseline_analyses,
            run_sensitivity_at_delta,
            run_tipping_point_grid,
            refine_tipping_point,
            get_clinical_context,
        ],
    )
    return agent


def run_agent_sensitivity_analysis(
    df: pd.DataFrame,
    observed: np.ndarray,
) -> AgentRun:
    """Run the agent-driven sensitivity analysis on a dataset.

    The agent autonomously:
    1. Examines baseline results
    2. Explores the delta parameter space
    3. Identifies and refines the tipping point
    4. Generates an interpretive summary

    Args:
        df: Full dataset with pre/post scores
        observed: Boolean mask (True = post-op score observed)

    Returns:
        AgentRun with results and metadata
    """
    # Set context for tools
    set_analysis_context(df, observed)

    # Create agent
    agent = create_sensitivity_agent()

    # Run agent
    start_time = time.time()

    prompt = (
        "Conduct a complete MNAR sensitivity analysis for this dataset. "
        "There are post-operative Oxford Hip Score outcomes that may be "
        "missing not at random. Determine whether the conclusion of "
        "positive health gain is robust to plausible MNAR departures. "
        "Find the tipping point if one exists."
    )

    response = agent(prompt)
    wall_time = time.time() - start_time

    # Extract results from agent response
    run_result = AgentRun(
        wall_time_seconds=wall_time,
        final_summary=str(response),
    )

    # Parse tool usage from agent's conversation
    # (Strands tracks this internally)
    if hasattr(agent, "messages"):
        for msg in agent.messages:
            if msg.get("role") == "assistant" and "tool_use" in str(msg.get("content", "")):
                run_result.n_tool_calls += 1

    return run_result


def run_manual_sensitivity_analysis(
    df: pd.DataFrame,
    observed: np.ndarray,
    delta_grid: list[float],
) -> dict:
    """Run the manual (non-agent) exhaustive sensitivity analysis.

    This is the comparator: a fixed protocol that evaluates all delta values
    without adaptive exploration.

    Returns dict with tipping point and all results.
    """
    from src.analysis.tipping_point import run_tipping_point_analysis

    start_time = time.time()

    tp_result = run_tipping_point_analysis(
        df=df,
        observed=observed,
        delta_grid=delta_grid,
    )

    wall_time = time.time() - start_time

    return {
        "tipping_point": tp_result.tipping_point,
        "is_robust": tp_result.is_robust,
        "n_deltas_evaluated": len(delta_grid),
        "wall_time_seconds": wall_time,
        "results": [
            {
                "delta": r.delta,
                "estimate": r.estimate,
                "ci_lower": r.ci_lower,
                "ci_upper": r.ci_upper,
                "significant": r.significant,
            }
            for r in tp_result.results_by_delta
        ],
    }
