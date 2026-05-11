# Agentic AI for MNAR Sensitivity Analysis in PROMs

An agentic AI system that autonomously conducts sensitivity analysis for Missing Not At Random (MNAR) data in Patient-Reported Outcome Measures (PROMs).

## Stack

- **Agent**: Strands Agents SDK + Amazon Bedrock
- **Stats**: scipy, statsmodels, scikit-learn
- **Dataset**: NHS England PROMs (Open Government Licence v3)

## Project Structure

```
src/
├── agent/          # Strands agent tools & orchestration
├── analysis/       # Statistical methods (pattern-mixture, tipping-point)
├── data/           # Data download & preprocessing
├── evaluation/     # Metrics
├── simulation/     # MNAR simulation engine
└── config.py
experiments/
└── run_experiment.py
```

## Setup

```bash
uv sync
python -m experiments.run_experiment
```

## Dataset

NHS England PROMs: https://digital.nhs.uk/data-and-information/publications/statistical/patient-reported-outcome-measures-proms

Licence: Open Government Licence v3
