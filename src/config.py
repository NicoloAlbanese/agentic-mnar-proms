"""Global configuration for the experiment."""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# NHS PROMs data source
NHS_PROMS_BASE_URL = (
    "https://digital.nhs.uk/data-and-information/publications/statistical/"
    "patient-reported-outcome-measures-proms"
)

# Experiment parameters
RANDOM_SEED = 42
N_SIMULATIONS = 500  # Monte Carlo replications per scenario
CONFIDENCE_LEVEL = 0.95

# MNAR simulation parameters
DELTA_GRID = [-2.0, -1.5, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 1.5, 2.0]
MISSINGNESS_RATES = [0.10, 0.20, 0.30, 0.40]

# Bedrock model configuration
BEDROCK_MODEL_ID = "anthropic.claude-sonnet-4-20250514"
BEDROCK_REGION = "us-east-1"

# Ensure directories exist
for d in [DATA_DIR, RESULTS_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)
