# Agentic AI for MNAR Sensitivity Analysis in PROMs

An agentic AI system that autonomously conducts sensitivity analysis for Missing Not At Random (MNAR) data in Patient-Reported Outcome Measures (PROMs).

## Stack

- **Agent**: Strands Agents SDK + Amazon Bedrock (Claude Sonnet 4)
- **Stats**: scipy, statsmodels, scikit-learn
- **Dataset**: NHS England PROMs (Open Government Licence v3)

## Project Structure

```
src/
├── agent/          # Strands agent tools & orchestration
├── analysis/       # Statistical methods (pattern-mixture, tipping-point)
├── data/           # Data download, preprocessing, synthetic generation
├── evaluation/     # Metrics (bias, RMSE, coverage)
├── simulation/     # MNAR simulation engine
└── config.py
experiments/
└── run_experiment.py
```

## Setup

```bash
uv sync
```

## Running Experiments

Phases can be run independently:

```bash
# All phases
uv run python -m experiments.run_experiment

# Individual phases
uv run python -m experiments.run_experiment --phase 1   # Baselines (no AWS)
uv run python -m experiments.run_experiment --phase 2   # Tipping-point (no AWS)
uv run python -m experiments.run_experiment --phase 3   # Agent (requires Bedrock)
```

Phase 1 and 2 are pure local computation. Phase 3 requires AWS credentials with Bedrock model access:

```powershell
$Env:AWS_ACCESS_KEY_ID="..."
$Env:AWS_SECRET_ACCESS_KEY="..."
$Env:AWS_SESSION_TOKEN="..."
uv run python -m experiments.run_experiment --phase 3
```

Results are saved incrementally to `results/` as parquet files.

## Retry Logic (Phase 3)

The agent orchestrator retries on transient Bedrock errors:
- ThrottlingException (429)
- ModelTimeoutException
- ServiceUnavailableException (503)
- InternalFailure (500)

Uses exponential backoff with jitter (30s initial, 300s max, 3 attempts) on top of Strands SDK's built-in retry (6 attempts, 4s–240s).

## Dataset

NHS England PROMs: https://digital.nhs.uk/data-and-information/publications/statistical/patient-reported-outcome-measures-proms

Licence: Open Government Licence v3
