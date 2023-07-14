"""Make up-to-date visualizations for README."""
import warnings

import fastf1 as f
import fastf1.plotting as p
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import visualization as viz
from preprocess import CURRENT_SEASON, ROOT_PATH, get_last_round_number

# plotting setup
mpl.use("Agg")
sns.set(rc={"figure.dpi": 300, "savefig.dpi": 300})
plt.style.use("dark_background")

# Suppress pandas SettingWithCopy warning
pd.options.mode.chained_assignment = None

# Suppress Seaborn false positive warnings
# TODO: This is dangerous
warnings.filterwarnings("ignore")

completed_round = get_last_round_number()

laptime = viz.driver_stats_scatterplot(
    season=CURRENT_SEASON, event=completed_round, drivers=10
)
plt.savefig(ROOT_PATH / "docs/visuals/laptime.png")

strategy = viz.strategy_barplot(season=CURRENT_SEASON, event=completed_round)
plt.savefig(ROOT_PATH / "docs/visuals/strategy.png")

position = viz.driver_stats_lineplot(season=CURRENT_SEASON, event=completed_round)
plt.savefig(ROOT_PATH / "docs/visuals/position.png")

teammate_box = viz.driver_stats_distplot(
    season=CURRENT_SEASON,
    event=completed_round,
    violin=False,
    swarm=False,
    teammate_comp=True,
    drivers=20,
)
plt.savefig(ROOT_PATH / "docs/visuals/teammate_box.png")

teammate_violin = viz.driver_stats_distplot(
    season=CURRENT_SEASON,
    event=completed_round,
    teammate_comp=True,
    drivers=20,
    upper_bound=7,
)
plt.savefig(ROOT_PATH / "docs/visuals/teammate_violin.png")

# use basic fastf1 to make team pace comparison plot
p.setup_mpl(misc_mpl_mods=False)
session = f.get_session(CURRENT_SEASON, completed_round, "R")
session.load(telemetry=False, weather=False, messages=False)
laps = session.laps.pick_wo_box()

laps["LapTime (s)"] = laps["LapTime"].dt.total_seconds()
team_order = (
    laps[["Team", "LapTime (s)"]]
    .groupby("Team")
    .median()["LapTime (s)"]
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
    whiskerprops=dict(color="white"),
    boxprops=dict(edgecolor="white"),
    medianprops=dict(color="grey"),
    capprops=dict(color="white"),
    showfliers=False,
)
plt.title(f"{CURRENT_SEASON} {session.event['EventName']}")
plt.grid(visible=False)
ax.set(xlabel=None)
plt.savefig(ROOT_PATH / "docs/visuals/team_pace.png")

# Add podium gap to winner plot
