"""NHS PROMs data acquisition.

NOTE: Record-level (individual patient) PROMs data is not publicly downloadable
via automated means. NHS Digital publishes aggregate/provider-level CSVs.

To obtain the data:
1. Go to: https://digital.nhs.uk/data-and-information/publications/statistical/
   patient-reported-outcome-measures-proms/final-2023-24-data
2. Download:
   - "CSV Hip replacement Provider" (4 MB) → data/raw/hip_provider_2324.csv
   - "CSV Hip and Knee Replacements Key Facts" (1 MB) → data/raw/key_facts_2324.csv
3. Licence: Open Government Licence v3
   (https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/)

The provider-level data contains mean OHS scores, response rates, and volumes
per hospital trust. We use this to:
- Calibrate synthetic data to real-world distributions
- Report actual missingness rates
- Validate that our simulation parameters are realistic

For record-level data (individual patient OHS item responses), a formal data
access request to NHS Digital is required.
"""

from pathlib import Path
from src.config import RAW_DATA_DIR


EXPECTED_FILES = {
    "hip_provider": RAW_DATA_DIR / "Hip Replacement Provider 2324 upload.csv",
    "key_facts": RAW_DATA_DIR / "Key Facts Hip and Knee Replacements 2324 upload.csv",
}


def check_data_available() -> dict[str, bool]:
    """Check which raw data files are available."""
    return {name: path.exists() for name, path in EXPECTED_FILES.items()}


def get_data_path(name: str) -> Path:
    """Get path to a raw data file, raising if not found."""
    path = EXPECTED_FILES[name]
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {path}\n"
            f"Download from NHS Digital PROMs page and place in data/raw/\n"
            f"See src/data/download.py docstring for instructions."
        )
    return path
