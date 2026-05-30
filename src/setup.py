"""Utility helpers for local setup.

This file documents the expected data layout. The actual dataset archive is not
shipped with the repository.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "datasets"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def print_expected_paths():
    expected = [
        DATA_DIR / "train_metadata_full.csv",
        DATA_DIR / "ml_small_metadata_full.csv",
        DATA_DIR / "posters",
        DATA_DIR / "ml-latest-small" / "ratings.csv",
    ]
    for path in expected:
        print(path)


if __name__ == "__main__":
    print_expected_paths()
