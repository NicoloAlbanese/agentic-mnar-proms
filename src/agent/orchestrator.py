"""Agent orchestration for autonomous sensitivity analysis."""

import time
import json
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from botocore.exceptions import ClientError
from strands import Agent
from strands.models import BedrockModel

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

logger = logging.getLogger(__name__)

# Bedrock error codes that warrant retry
RETRYABLE_ERROR_CODES = {
    "ThrottlingException",
    "ModelTimeoutException",
    "ServiceUnavailableException",
    "InternalFailure",
}

# Max retries at the experiment level (on top of Strands' built-in retries)
MAX_EXPERIMENT_RETRIES = 3
INITIAL_RETRY_DELAY = 30  # seconds
MAX_RETRY_DELAY = 300  # seconds


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


def _is_retryable_error(exc: Exception) -> bool:
    """Check if an exception is retryable based on Bedrock error codes."""
    if isinstance(exc, ClientError):
        error_code = exc.response.get("Error", {}).get("Code", "")
        return error_code in RETRYABLE_ERROR_CODES
    # Strands may wrap errors
    error_str = str(exc)
    return any(code in error_str for code in RETRYABLE_ERROR_CODES)


def _retry_with_backoff(func, *args, **kwargs):
    """Execute func with exponential backoff on retryable errors.

    Uses jitter to avoid thundering herd.
    """
    delay = INITIAL_RETRY_DELAY
    for attempt in range(MAX_EXPERIMENT_RETRIES + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt == MAX_EXPERIMENT_RETRIES or not _is_retryable_error(e):
                raise
            jitter = np.random.uniform(0.5, 1.5)
            wait = min(delay * jitter, MAX_RETRY_DELAY)
            logger.warning(
                f"Retryable error (attempt {attempt + 1}/{MAX_EXPERIMENT_RETRIES}): {e}. "
                f"Waiting {wait:.0f}s..."
            )
            time.sleep(wait)
            delay *= 2  # exponential backoff


def create_sensitivity_agent() -> Agent:
    """Create a Strands agent configured for sensitivity analysis."""
    bedrock_model = BedrockModel(
        model_id=BEDROCK_MODEL_ID,
        region_name=BEDROCK_REGION,
    )

    agent = Agent(
        model=bedrock_model,
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


def _extract_tipping_point_from_messages(messages: list) -> tuple[float | None, bool, int, float]:
    """Parse tool results from agent conversation to extract structured data.

    Returns (tipping_point, is_robust, n_deltas_evaluated, estimate_mar)
    """
    tipping_point = None
    is_robust = True
    n_deltas = 0
    estimate_mar = 0.0

    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or "toolResult" not in block:
                continue
            tool_content = block["toolResult"].get("content", [])
            for item in tool_content:
                if not isinstance(item, dict) or "text" not in item:
                    continue
                try:
                    data = json.loads(item["text"])
                except (json.JSONDecodeError, TypeError):
                    continue

                # Extract MAR estimate from baseline results
                if "mi_mar" in data and isinstance(data["mi_mar"], dict):
                    estimate_mar = data["mi_mar"].get("estimate", estimate_mar)

                # Extract from refine_tipping_point result
                if "refined_tipping_point" in data:
                    tipping_point = data["refined_tipping_point"]
                    is_robust = False

                # Extract from run_tipping_point_grid result
                if "tipping_point" in data and data["tipping_point"] is not None:
                    tipping_point = data["tipping_point"]
                    is_robust = False
                elif "is_robust" in data and data["is_robust"]:
                    is_robust = True

                # Count deltas evaluated
                if "results" in data and isinstance(data["results"], list):
                    n_deltas += len(data["results"])
                elif "delta" in data:
                    n_deltas += 1

    return tipping_point, is_robust, n_deltas, estimate_mar


def _run_agent_once(df: pd.DataFrame, observed: np.ndarray) -> AgentRun:
    """Single agent run (called within retry wrapper)."""
    set_analysis_context(df, observed)
    agent = create_sensitivity_agent()

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

    run_result = AgentRun(
        wall_time_seconds=wall_time,
        final_summary=str(response),
    )

    # Parse structured data from tool results
    if hasattr(agent, "messages"):
        tp, robust, n_deltas, mar_est = _extract_tipping_point_from_messages(agent.messages)
        run_result.tipping_point = tp
        run_result.is_robust = robust
        run_result.n_deltas_evaluated = n_deltas
        run_result.estimate_mar = mar_est

        # Count tool calls
        for msg in agent.messages:
            if msg.get("role") == "assistant":
                content = msg.get("content", [])
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and "toolUse" in block:
                            run_result.n_tool_calls += 1

    return run_result


def run_agent_sensitivity_analysis(
    df: pd.DataFrame,
    observed: np.ndarray,
) -> AgentRun:
    """Run the agent-driven sensitivity analysis with retry logic.

    The agent autonomously:
    1. Examines baseline results
    2. Explores the delta parameter space
    3. Identifies and refines the tipping point
    4. Generates an interpretive summary

    Retries on ThrottlingException, ModelTimeoutException, ServiceUnavailableException
    with exponential backoff.
    """
    return _retry_with_backoff(_run_agent_once, df, observed)


def run_manual_sensitivity_analysis(
    df: pd.DataFrame,
    observed: np.ndarray,
    delta_grid: list[float],
) -> dict:
    """Run the manual (non-agent) exhaustive sensitivity analysis.

    This is the comparator: a fixed protocol that evaluates all delta values
    without adaptive exploration.
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
                "significant": bool(r.significant),
            }
            for r in tp_result.results_by_delta
        ],
    }
