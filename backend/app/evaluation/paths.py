from __future__ import annotations

from pathlib import Path

# Shared location for the golden dataset artifact, kept next to the framework so
# it travels with the code and is easy to review/curate.
DEFAULT_DATASET_PATH = Path(__file__).parent / "data" / "golden_dataset.json"
