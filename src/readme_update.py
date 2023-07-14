"""Make up-to-date visualizations for README."""
import warnings

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
    season=CURRENT_SEASON, event=completed_round, teammate_comp=True, drivers=20
)
plt.savefig(ROOT_PATH / "docs/visuals/teammate_violin.png")

# Add podium gap to winner plot
