# Agentic AI for MNAR Sensitivity Analysis in PROMs

An agentic AI system that autonomously conducts sensitivity analysis for Missing Not At Random (MNAR) data in Patient-Reported Outcome Measures (PROMs).

## Background

### The problem

Patient-Reported Outcome Measures (PROMs) collect patient self-assessments before and after surgery. The primary estimand is the mean health gain:

```
θ = E[Y_post - Y_pre]
```

where `Y_post` is the post-operative Oxford Hip Score (OHS, 0-48) and `Y_pre` is the pre-operative score. In NHS England PROMs 2023/24, 62% of hip replacement patients did not return post-operative questionnaires. If non-response depends on the unobserved outcome itself (e.g., patients with complications are less likely to respond), the data are Missing Not At Random (MNAR) and standard methods (complete-case, multiple imputation under MAR) produce biased estimates of θ.

### Sensitivity analysis via pattern-mixture models

Under the pattern-mixture framework, the distribution of missing outcomes is modelled as a shift relative to the MAR-imputed distribution:

```
f(Y_mis | R=0, X) = f(Y_mis | R=1, X) + δ
```

where δ is the sensitivity parameter. When δ < 0, missing patients are assumed to have worse outcomes than MAR predicts. The tipping-point analysis identifies the smallest |δ| at which the conclusion reverses (i.e., the 95% CI for θ includes zero):

```
δ_tip = min{|δ| : CI_lower(θ̂(δ)) ≤ 0}
```

If δ_tip is large relative to clinically plausible values, the conclusion is robust.

### The MNAR simulation model

We simulate missingness using a selection model:

```
P(R_i = 1 | X_i, Y_i) = expit(α₀ + α₁·X_i + δ·Y_i*)
```

where R=1 means observed, X are covariates, Y* is the standardized outcome, and δ controls MNAR severity. The intercept α₀ is calibrated via bisection to achieve a target missingness rate.

### What this repo validates

1. That an LLM-based agent can autonomously navigate the sensitivity parameter space (choosing which δ values to evaluate, when to refine via bisection, when to stop)
2. That the agent's adaptive strategy achieves the same statistical conclusions as exhaustive grid search, with fewer evaluations
3. That the agent produces clinically interpretable summaries without human guidance

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

## Data

This project uses NHS England PROMs data (Open Government Licence v3).

**To obtain the data:**
1. Visit https://digital.nhs.uk/data-and-information/publications/statistical/patient-reported-outcome-measures-proms/final-2023-24-data
2. Download:
   - "CSV Hip replacement Provider" → `data/raw/hip_provider_2324.csv`
   - "CSV Hip and Knee Replacements Key Facts" → `data/raw/key_facts_2324.csv`
3. The experiment can also run with synthetic data calibrated to these statistics

Key facts from 2023/24: 78,000 hip replacement episodes, 55,000 pre-op responses, 21,000 post-op responses (62% post-operative non-response).

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

NHS England PROMs (Final 2023/24): https://digital.nhs.uk/data-and-information/publications/statistical/patient-reported-outcome-measures-proms/final-2023-24-data

Licence: Open Government Licence v3
