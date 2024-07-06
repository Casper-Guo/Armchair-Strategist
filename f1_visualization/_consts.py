from datetime import datetime
from pathlib import Path

import tomli

ROOT_PATH = Path(__file__).absolute().parents[1]
DATA_PATH = ROOT_PATH / "Data"

CURRENT_SEASON = datetime.now().year

# Number of completed rounds in the current season is computed dynamically
# Calculating this from fastf1 event schedule is non-trivial due to cancelled races
NUM_ROUNDS = {2018: 21, 2019: 21, 2020: 17, 2021: 22, 2022: 22, 2023: 22}

# Map session ids to full session names, and reverse
SESSION_IDS = {"R": "grand_prix", "S": "sprint"}
SESSION_NAMES = {name: session_id for session_id, name in SESSION_IDS.items()}
SPRINT_FORMATS = set(("sprint", "sprint_shootout", "sprint_qualifying"))

with open(DATA_PATH / "compound_selection.toml", "rb") as toml:
    COMPOUND_SELECTION = tomli.load(toml)
with open(DATA_PATH / "visualization_config.toml", "rb") as toml:
    VISUAL_CONFIG = tomli.load(toml)
