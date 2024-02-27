"""Make up-to-date visualizations for README."""
import logging
import shutil
import warnings
from pathlib import Path

import fastf1 as f
import fastf1.plotting as p
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import visualization as viz
from preprocess import CURRENT_SEASON, ROOT_PATH, get_last_round_number

logging.basicConfig(
    level=logging.INFO, format="%(levelname)s\t%(filename)s\t%(message)s"
)

# plotting setup
visuals_path = ROOT_PATH / "Docs" / "visuals"
mpl.use("Agg")
sns.set(rc={"figure.dpi": 300, "savefig.dpi": 300})
plt.style.use("dark_background")

# Suppress pandas SettingWithCopy warning
pd.options.mode.chained_assignment = None

# Suppress Seaborn false positive warnings
# TODO: This is dangerous
warnings.filterwarnings("ignore")

completed_round = get_last_round_number()
session = f.get_session(CURRENT_SEASON, completed_round, "R")
session.load(telemetry=False, weather=False, messages=False)
logging.info("Visualizing %s", session)

logging.info("Making podium gap graph...")
podium_finishers = viz.get_drivers(session, drivers=3)
race_winner = podium_finishers[0]
viz.add_gap(CURRENT_SEASON, race_winner)
podium_gap = viz.driver_stats_lineplot(
    season=CURRENT_SEASON,
    event=completed_round,
    drivers=podium_finishers,
    y=f"GapTo{race_winner}",
    grid="both",
)
plt.savefig(visuals_path / "podium_gap.png")

logging.info("Making lap time graph...")
laptime = viz.driver_stats_scatterplot(
    season=CURRENT_SEASON, event=completed_round, drivers=10
)
plt.savefig(visuals_path / "laptime.png")

logging.info("Making strategy graph...")
strategy = viz.strategy_barplot(season=CURRENT_SEASON, event=completed_round)
plt.savefig(visuals_path / "strategy.png")

logging.info("Making position change graph...")
position = viz.driver_stats_lineplot(season=CURRENT_SEASON, event=completed_round)
plt.savefig(visuals_path / "position.png")

logging.info("Making teammate comparison boxplot...")
teammate_box = viz.driver_stats_distplot(
    season=CURRENT_SEASON,
    event=completed_round,
    violin=False,
    swarm=False,
    teammate_comp=True,
    drivers=20,
)
plt.savefig(visuals_path / "teammate_box.png")

logging.info("Making teammate comp violinplot...")
teammate_violin = viz.driver_stats_distplot(
    season=CURRENT_SEASON,
    event=completed_round,
    teammate_comp=True,
    drivers=20,
    upper_bound=7,
)
plt.savefig(visuals_path / "teammate_violin.png")

# use basic fastf1 to make team pace comparison plot
logging.info("Making team pace comparison graph...")
p.setup_mpl(misc_mpl_mods=False)
laps = session.laps.pick_wo_box()
event_name = session.event["EventName"]

laps["LapTime (s)"] = laps["LapTime"].dt.total_seconds()
team_order = (
    laps[["Team", "LapTime (s)"]]
    .groupby("Team")
    .median(numeric_only=True)["LapTime (s)"]
    .sort_values()
    .index
)
team_palette = {team: p.team_color(team) for team in team_order}

fig, ax = plt.subplots(figsize=(15, 10))
sns.boxplot(
    data=laps,
    x="Team",
    y="LapTime (s)",
    order=team_order,
    palette=team_palette,
    whiskerprops={"color": "white"},
    boxprops={"edgecolor": "white"},
    medianprops={"color": "grey"},
    capprops={"color": "white"},
    showfliers=False,
)
plt.title(f"{CURRENT_SEASON} {event_name}")
plt.grid(visible=False)
ax.set(xlabel=None)
plt.savefig(visuals_path / "team_pace.png")

# Copy the visualizations
dest = ROOT_PATH / "Visualizations" / f"{CURRENT_SEASON}" / f"{event_name}"
Path.mkdir(dest, parents=True, exist_ok=True)
shutil.copytree(visuals_path, dest, dirs_exist_ok=True)
