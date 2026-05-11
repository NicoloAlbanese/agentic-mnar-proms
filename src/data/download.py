"""Download NHS PROMs data from NHS Digital."""

import urllib.request
import zipfile
from pathlib import Path

from src.config import RAW_DATA_DIR

# NHS PROMs CSV download URLs
# Final 2022/23 publication - record-level CSV
# Source: https://digital.nhs.uk/data-and-information/publications/statistical/
#         patient-reported-outcome-measures-proms/finalised-patient-reported-outcome-measures-proms---2022-23
# Licence: Open Government Licence v3
PROMS_URLS = {
    "hip_2223": (
        "https://files.digital.nhs.uk/5B/B98B03/"
        "Hip Replacement PROMS 2223 - CSV.zip"
    ),
}


def download_file(url: str, dest: Path) -> Path:
    """Download a file from URL to destination path."""
    if dest.exists():
        print(f"  Already downloaded: {dest.name}")
        return dest
    print(f"  Downloading {dest.name} ...")
    urllib.request.urlretrieve(url, dest)
    print(f"  Saved to {dest}")
    return dest


def extract_zip(zip_path: Path, extract_to: Path) -> list[Path]:
    """Extract a zip file and return list of extracted files."""
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
        return [extract_to / name for name in zf.namelist()]


def download_proms_data(dataset_key: str = "hip_2223") -> Path:
    """Download and extract NHS PROMs dataset.

    Returns path to the extracted CSV directory.
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    url = PROMS_URLS[dataset_key]
    zip_path = RAW_DATA_DIR / f"{dataset_key}.zip"

    download_file(url, zip_path)
    extracted = extract_zip(zip_path, RAW_DATA_DIR / dataset_key)
    print(f"  Extracted {len(extracted)} files to {RAW_DATA_DIR / dataset_key}")

    return RAW_DATA_DIR / dataset_key
