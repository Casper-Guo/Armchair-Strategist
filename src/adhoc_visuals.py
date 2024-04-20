"""Make up-to-date visualizations for README."""

import logging
import shutil
import warnings
from pathlib import Path

import click
import fastf1 as f
import fastf1.plotting as p
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import visualization as viz
from preprocess import CURRENT_SEASON, ROOT_PATH, get_last_round_number

logging.basicConfig(level=logging.INFO, format="%(levelname)s\t%(filename)s\t%(message)s")

# plotting setup
VISUALS_PATH = ROOT_PATH / "Docs" / "visuals"
mpl.use("Agg")
sns.set_theme(rc={"figure.dpi": 300, "savefig.dpi": 300})
plt.style.use("dark_background")
COMPLETED_ROUND = get_last_round_number()

# Suppress pandas SettingWithCopy warning
pd.options.mode.chained_assignment = None

# Suppress Seaborn false positive warnings
# TODO: This is dangerous
warnings.filterwarnings("ignore")


@click.command()
@click.option("-s", "--season", "season", default=CURRENT_SEASON, type=int)
@click.option("-r", "--round", "round_number", default=COMPLETED_ROUND, type=int)
@click.option(
    "--grand-prix/--sprint-race",
    "-g",
    " /-S",
    default=True,
    help="Toggle between plotting the sprint race or the grand prix",
)
def main(season: int, round_number: int, grand_prix: bool):
    """Make the README suite of visualizations."""
    global VISUALS_PATH
    session = f.get_session(season, round_number, "R" if grand_prix else "S")
    session.load(telemetry=False, weather=False, messages=False)
    event_name = session.event["EventName"]

    cli_call = not (
        (season == CURRENT_SEASON) and (round_number == COMPLETED_ROUND) and grand_prix
    )
    dest = ROOT_PATH / "Visualizations" / f"{season}" / f"{event_name}"

    if cli_call:
        if dest.is_dir():
            overwrite_confirmation = input(
                (
                    "WARNING:\n"
                    f"{dest} may already contain the desired visualizations.\n"
                    "Enter Y if you wish to overwrite them: "
                )
            )
            if overwrite_confirmation.upper() != "Y":
                logging.info("Overwriting permission not given, aborting.")
                return
        else:
            Path.mkdir(dest, parents=True, exist_ok=True)

        VISUALS_PATH = dest

    logging.info("Visualizing %s", session)

    logging.info("Making podium gap graph...")
    podium_finishers = viz.get_drivers(session, drivers=3)
    race_winner = podium_finishers[0]
    viz.add_gap(season, race_winner)
    viz.driver_stats_lineplot(
        season=season,
        event=round_number,
        drivers=podium_finishers,
        y=f"GapTo{race_winner}",
        grid="both",
    )
    plt.savefig(VISUALS_PATH / "podium_gap.png")

    logging.info("Making lap time graph...")
    viz.driver_stats_scatterplot(season=season, event=round_number, drivers=10)
    plt.savefig(VISUALS_PATH / "laptime.png")

    logging.info("Making strategy graph...")
    viz.strategy_barplot(season=season, event=round_number)
    plt.savefig(VISUALS_PATH / "strategy.png")

    logging.info("Making position change graph...")
    viz.driver_stats_lineplot(season=season, event=round_number)
    plt.savefig(VISUALS_PATH / "position.png")

    logging.info("Making teammate comparison boxplot...")
    # TODO: remove dependency on hard-coded driver quantity
    viz.driver_stats_distplot(
        season=season,
        event=round_number,
        violin=False,
        swarm=False,
        teammate_comp=True,
        drivers=20,
    )
    plt.savefig(VISUALS_PATH / "teammate_box.png")

    logging.info("Making teammate comp violinplot...")
    viz.driver_stats_distplot(
        season=season,
        event=round_number,
        teammate_comp=True,
        drivers=20,
        upper_bound=7,
    )
    plt.savefig(VISUALS_PATH / "teammate_violin.png")

    # use basic fastf1 to make team pace comparison plot
    logging.info("Making team pace comparison graph...")
    p.setup_mpl(misc_mpl_mods=False)
    laps = session.laps.pick_wo_box()

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
    plt.savefig(VISUALS_PATH / "team_pace.png")

    # Copy the visualizations
    if not cli_call:
        shutil.copytree(VISUALS_PATH, dest, dirs_exist_ok=True)


if __name__ == "__main__":
    main()
