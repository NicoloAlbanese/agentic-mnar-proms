# Agentic AI for MNAR Sensitivity Analysis in PROMs

An agentic AI system that autonomously conducts sensitivity analysis for Missing Not At Random (MNAR) data in Patient-Reported Outcome Measures (PROMs).

## Table of Contents

- [Background](#background)
  - [Patient-Reported Outcome Measures](#patient-reported-outcome-measures)
  - [The Missing Data Problem](#the-missing-data-problem)
  - [Why Standard Methods Fail Under MNAR](#why-standard-methods-fail-under-mnar)
  - [Sensitivity Analysis and Tipping Points](#sensitivity-analysis-and-tipping-points)
  - [The Agentic Approach](#the-agentic-approach)
- [Results](#results)
  - [High-Effect Scenario](#high-effect-scenario-real-nhs-data)
  - [Marginal-Benefit Scenario](#marginal-benefit-scenario-synthetic)
  - [Why These Results Matter](#why-these-results-matter)
- [Setup](#setup)
- [Data](#data)
- [Running Experiments](#running-experiments)
- [Project Structure](#project-structure)
- [Licence](#licence)

---

## Background

### Patient-Reported Outcome Measures

Patient-Reported Outcome Measures (PROMs) are standardized questionnaires that capture the patient's own assessment of their health before and after a clinical intervention. Unlike clinician-reported outcomes (e.g., surgical success rates), PROMs reflect what matters to the patient, including pain, mobility, and quality of life.

In the context of hip replacement surgery, the most widely used instrument is the Oxford Hip Score (OHS), a 12-item questionnaire that produces a score from 0 to 48, where 48 represents perfect hip function. The primary outcome of interest is the health gain, defined as the difference between post-operative and pre-operative OHS. A gain of 5 or more points is considered the Minimal Clinically Important Difference (MCID), meaning the patient perceives a meaningful improvement ([Bell and Fairclough, 2014](https://doi.org/10.1177/0962280212460837)).

The NHS England PROMs programme has collected these questionnaires for all publicly funded hip and knee replacements since 2009, making it one of the largest routine outcome measurement systems in the world ([NHS Digital, PROMs 2023/24](https://digital.nhs.uk/data-and-information/publications/statistical/patient-reported-outcome-measures-proms/final-2023-24-data)).

### The Missing Data Problem

The fundamental challenge with PROMs is non-response. Patients complete the pre-operative questionnaire at the hospital before surgery, but the post-operative questionnaire is mailed to them 6 months later. Many never return it.

In the 2023/24 NHS PROMs data for hip replacement, approximately 78,000 procedures were performed, 55,000 patients completed the pre-operative questionnaire, but only 21,000 returned the post-operative one. This means 62% of patients have missing post-operative outcomes.

If the patients who do not respond are a random subset of all patients, this is merely a loss of statistical power. But if non-responders are systematically different from responders, specifically if they tend to have worse outcomes, then the observed data paint an overly optimistic picture of surgical effectiveness.

There are three categories of missing data mechanisms, formalized by [Rubin (1976)](https://doi.org/10.1093/biomet/63.3.581).

- **MCAR** (Missing Completely At Random). The probability of non-response is the same for everyone, regardless of any patient characteristics or outcomes. Example: the postal service loses questionnaires at random.
- **MAR** (Missing At Random). The probability of non-response depends on observed variables (age, sex, pre-operative score) but not on the unobserved post-operative outcome. Example: older patients respond less, but among patients of the same age, responders and non-responders have similar outcomes.
- **MNAR** (Missing Not At Random). The probability of non-response depends on the unobserved outcome itself. Example: patients who experienced complications or poor recovery are less likely to respond precisely because they are doing poorly.

The critical issue is that MNAR cannot be distinguished from MAR using observed data alone. There is no statistical test that can tell you whether your data are MAR or MNAR. This is why sensitivity analysis is necessary.

### Why Standard Methods Fail Under MNAR

Standard approaches to missing data, including complete-case analysis (analyzing only patients with both scores) and multiple imputation under MAR, assume that missingness does not depend on the unobserved outcome. Under MNAR, these methods produce biased estimates of the health gain because they effectively ignore the possibility that non-responders are systematically worse off ([Ayilara et al., 2019](https://doi.org/10.1186/s12955-019-1181-2)).

The bias can be substantial. In our simulations, MI-MAR underestimates the true health gain by 1.2 to 2.4 points (depending on MNAR severity) when 25% of outcomes are missing. For a marginal intervention with a true gain of only 5.5 points, this represents 22 to 43% of the effect, potentially leading to incorrect conclusions about treatment effectiveness.

Regulatory bodies recognize this problem. The ICH E9(R1) addendum on estimands and sensitivity analysis (2019) explicitly requires that clinical evaluations assess the robustness of conclusions to assumptions about missing data. NICE and the EMA recommend sensitivity analysis for MNAR as part of health technology assessments. Yet in practice, these analyses are rarely performed. [Rombach et al. (2016)](https://doi.org/10.1007/s11136-015-1206-1) found that the majority of PROMs studies in RCTs do not conduct MNAR sensitivity analysis, primarily because it requires specialized statistical expertise.

### Sensitivity Analysis and Tipping Points

The pattern-mixture model provides a principled framework for MNAR sensitivity analysis ([Little, 1993](https://doi.org/10.2307/2290705); [Carpenter et al., 2013](https://doi.org/10.1080/10543406.2013.834911)). The idea is straightforward. We start with the best estimate we can produce under MAR (using multiple imputation), and then ask: "what if the imputed values for non-responders are systematically wrong by δ points?"

The parameter δ represents a departure from the MAR assumption.

- δ = 0 corresponds to MAR (no departure).
- δ = −1 means non-responders have outcomes 1 point worse than MAR predicts.
- δ = −3 means non-responders have outcomes 3 points worse than MAR predicts.
- Positive δ values (non-responders doing better) are generally considered clinically implausible for surgical PROMs.

For each value of δ, we recompute the health gain estimate and its confidence interval using Rubin's rules ([Cro et al., 2020](https://doi.org/10.1002/sim.8569)). The **tipping point** is the value of δ at which the confidence interval first includes zero, meaning we can no longer conclude that the intervention is effective.

If the tipping point is at δ = −0.5, the conclusion is fragile (a tiny MNAR departure invalidates it). If the tipping point is at δ = −4.5, the conclusion is robust (only an implausibly extreme MNAR mechanism could change it). If no tipping point exists within the tested range, the conclusion is very robust.

### The Agentic Approach

The agent is not an alternative statistical method. It implements the same pattern-mixture tipping-point analysis described above, using the same mathematics (MICE imputation, δ-shift, Rubin's rules). The difference is in how it navigates the parameter space.

A fixed grid evaluates all δ values in a predetermined set regardless of the data. The agent adapts its strategy based on intermediate results. It starts with a coarse exploration, identifies the region where the tipping point might lie, and refines via bisection to locate it precisely. It also produces a clinically contextualized interpretation without human guidance.

The agent is built with the [Strands Agents SDK](https://github.com/strands-agents/sdk-python) (open-source, AWS) and uses Claude Sonnet 4 via Amazon Bedrock as its reasoning engine. It has five statistical tools.

1. **Baseline analyses.** Runs complete-case, mean imputation, and MI-MAR to establish the starting point.
2. **Sensitivity at δ.** Evaluates the pattern-mixture model at a single δ value.
3. **Tipping-point grid.** Evaluates multiple δ values simultaneously and identifies the tipping point.
4. **Bisection refinement.** Narrows the tipping point location between two bracketing values.
5. **Clinical context.** Provides OHS scale information and MCID thresholds for interpretation.

---

## Results

### High-Effect Scenario (Real NHS Data)

Population of N=11,629 complete hip replacement cases from NHS PROMs 2023/24. Mean health gain 22.6 points (SD=10.2).

**Phase 1 (Baseline bias).** Under simulated MNAR at 25% missingness, all methods are unbiased under MAR (δ=0) but increasingly biased as MNAR severity grows.

| δ | Complete-case | Mean imputation | MI-MAR |
|--:|--:|--:|--:|
| -5.0 | -1.53 | -2.36 | -2.06 |
| -3.0 | -1.42 | -2.16 | -1.90 |
| -2.0 | -1.30 | -1.94 | -1.68 |
| -1.0 | -0.97 | -1.41 | -1.24 |
| 0.0 | -0.03 | +0.01 | -0.01 |
| +1.0 | +1.63 | +2.24 | +2.09 |


### Marginal-Benefit Scenario (Synthetic)

Synthetic population calibrated to mean health gain 5.5 points (SD=9.6), representing a subgroup with borderline clinical benefit.

**Phase 2 (Exhaustive grid).** Tipping points found only at 35% missingness. At true δ=−3.0, tipping point at −4.03 (found in 97% of reps). At true δ=−2.0, tipping point at −4.63 (found in 53% of reps). All other conditions remained robust.

**Phase 3 (Agent).** The agent identified tipping points in 100% of 90 runs across all conditions.


| Condition | Grid tipping point | Grid detection | Agent tipping point | Agent detection | Agent tool calls | Agent time |
|---|---|---|---|---|---|---|
| true δ=−2.0, 15% miss | none | robust | −21.6 | 100% | 9.1 | 120s |
| true δ=−2.0, 25% miss | none | robust | −9.7 | 100% | 6.8 | 90s |
| true δ=−2.0, 35% miss | −4.63 | 53% | −4.52 (SD=0.47) | 100% | 5.6 | 71s |
| true δ=−1.0, 25% miss | none | robust | −13.1 | 100% | 7.3 | 101s |
| true δ=−1.0, 35% miss | none | robust | −7.5 | 100% | 7.3 | 91s |
| true δ=−0.5, 35% miss | none | robust | −10.3 | 100% | 7.0 | 117s |


The agent achieved higher detection rates than the fixed grid (100% vs 53% at the boundary case) because its adaptive exploration extends beyond predefined grid boundaries.

### Why These Results Matter

1. **Feasibility.** An LLM agent can autonomously conduct a complete MNAR sensitivity analysis workflow without human guidance.
2. **Accuracy.** The agent's conclusions are concordant with exhaustive methods across all tested conditions.
3. **Reliability.** Adaptive exploration achieves higher tipping-point detection than fixed grids in boundary cases.
4. **Interpretability.** The agent produces clinically contextualized summaries suitable for non-specialist audiences.
5. **Practical relevance.** The 62% non-response rate in NHS PROMs is a real problem. This approach lowers the expertise barrier for recommended sensitivity analysis practices.

---

## Setup

```bash
uv sync
```

Requires Python 3.11+. Phase 3 requires AWS credentials with Amazon Bedrock model access (Claude Sonnet 4). The model can be configured through the model id parameter.

## Data

This project uses NHS England PROMs data published under the [Open Government Licence v3](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).

**To obtain the data.**
1. Visit the [NHS PROMs 2023/24 publication page](https://digital.nhs.uk/data-and-information/publications/statistical/patient-reported-outcome-measures-proms/final-2023-24-data)
2. Download "CSV Hip replacement Provider" and place in `data/raw/`
3. Run `uv run python -m src.data.preprocess` to generate the analysis dataset

The experiment can also run entirely on synthetic data (generated automatically if real data is not present).

## Running Experiments

```bash
# All phases
uv run python -m experiments --run run_1 --scenario high_effect

# Note:
#   - the name "run_1" is arbitrary and can be changed
#   - it guides results folder nomenclature
#   - maintain consistency across experiments in the same run

# Individual phases
uv run python -m experiments --run run_1 --phase 1   # Baselines (no AWS needed)
uv run python -m experiments --run run_1 --phase 2   # Tipping-point (no AWS needed)
uv run python -m experiments --run run_1 --phase 3   # Agent (requires Amazon Bedrock)

# Options
uv run python -m experiments --run run_1 --scenario low_effect
uv run python -m experiments --sample-size 3000 --reps 50 --n-imputations 30
```

Phase 3 requires AWS credentials.

```bash
# Example
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_SESSION_TOKEN="..."

uv run python -m experiments --run run_4 --phase 3
```

All phases support automatic resume. If interrupted, re-run the same command and completed work is skipped. To force a fresh run, delete the relevant parquet file in `results/`.

## Project Structure

```
src/
├── agent/          # Strands agent, tools, prompts, orchestration
├── analysis/       # Pattern-mixture model, tipping-point, baselines
├── data/           # Data acquisition, preprocessing, synthetic generation
├── evaluation/     # Metrics (bias, RMSE, coverage, concordance)
├── simulation/     # MNAR missingness engine (selection model)
└── config.py       # Global configuration
experiments/
├── config.py       # ExperimentConfig dataclass
├── utils.py        # Shared helpers (sampling, persistence)
├── runner.py       # CLI entry point
└── phases/         # Phase 1, 2, 3 implementations
data/
├── raw/            # NHS PROMs CSVs (see Data section)
└── processed/      # Parquet files from preprocessing or synthetic generation
results/
└── run_<name>/     # One directory per experiment run
    ├── parameters.txt                 # Full config snapshot for reproducibility
    ├── baseline_results.parquet       # Phase 1 per-replication results
    ├── summary_metrics.parquet        # Phase 1 aggregated metrics
    ├── tipping_point_results.parquet  # Phase 2 results
    ├── agent_results.parquet          # Phase 3 results
    └── experiment.log                 # Full console output
```

## Licence

Code in this repository is provided for research purposes. The NHS PROMs data is used under the [Open Government Licence v3](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).
