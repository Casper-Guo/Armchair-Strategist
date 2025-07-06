"""Shared variables for f1_visualization module."""

from datetime import datetime
from pathlib import Path

import tomli

ROOT_PATH = Path(__file__).absolute().parents[1]
DATA_PATH = ROOT_PATH / "Data"

CURRENT_SEASON = datetime.now().year  # noqa: DTZ005

# Number of fully completed rounds in the current season is computed dynamically
# Calculating this from fastf1 event schedule is non-trivial due to cancelled races
NUM_ROUNDS = {2018: 21, 2019: 21, 2020: 17, 2021: 22, 2022: 22, 2023: 22, 2024: 24}

SPRINT_ROUNDS = {
    2021: {10, 14, 19},
    2022: {4, 11, 21},
    2023: {4, 9, 12, 17, 18, 20},
    2024: {5, 6, 11, 19, 21, 23},
    2025: {2, 6, 13, 19, 21, 23},
}

# Map session ids to full session names, and reverse
SESSION_IDS = {"R": "grand_prix", "S": "sprint"}
SESSION_NAMES = {name: session_id for session_id, name in SESSION_IDS.items()}
SPRINT_FORMATS = {"sprint", "sprint_shootout", "sprint_qualifying"}
SPRINT_QUALI_ORDINAL = 2
SPRINT_RACE_ORDINAL = 3
RACE_QUALI_ORDINAL = 4
GRAND_PRIX_ORDINAL = 5

with open(DATA_PATH / "compound_selection.toml", "rb") as toml:
    COMPOUND_SELECTION: dict[str, dict[str, list[str]]] = tomli.load(toml)
with open(DATA_PATH / "visualization_config.toml", "rb") as toml:
    VISUAL_CONFIG = tomli.load(toml)
