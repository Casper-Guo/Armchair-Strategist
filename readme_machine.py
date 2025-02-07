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

import f1_visualization.visualization as viz
from f1_visualization._consts import CURRENT_SEASON, NUM_ROUNDS, ROOT_PATH, SPRINT_ROUNDS
from f1_visualization.preprocess import get_last_round

logging.basicConfig(level=logging.INFO, format="%(levelname)s\t%(filename)s\t%(message)s")
logger = logging.getLogger(__name__)

# plotting setup
DOC_VISUALS_PATH = ROOT_PATH / "Docs" / "visuals"
mpl.use("Agg")
sns.set_theme(rc={"figure.dpi": 300, "savefig.dpi": 300})
plt.style.use("dark_background")

# Suppress pandas SettingWithCopy warning
pd.options.mode.chained_assignment = None

# Suppress Seaborn false positive warnings
# TODO: This is dangerous
warnings.filterwarnings("ignore")


def process_round_number(season: int, round_number: int, grand_prix: bool) -> int:
    """Get the last available round number of the requested session type in a season."""
    if season > CURRENT_SEASON:
        raise ValueError(f"The latest season is {CURRENT_SEASON}.")
    if season < 2018:
        raise ValueError("Only 2018 and later seasons are available.")
    if round_number < 1 and round_number != -1:
        raise ValueError("Round number must be positive.")

    if season == CURRENT_SEASON:
        last_round_number = get_last_round(session_cutoff=5 if grand_prix else 3)

        if last_round_number == 0:
            # this means there is no completed round in the current season yet
            # default to the last season's final round
            raise ValueError(f"No session of the requested type in the {season} season yet.")
        if grand_prix:
            return min(round_number, last_round_number)
        if last_round_number <= round_number:
            return last_round_number
            # if round_number < latest_round_number, extra logic needed to find the last sprint
            # round before the requested value

    if round_number > NUM_ROUNDS[season]:
        raise ValueError(f"{season} season only has {NUM_ROUNDS[season]} rounds.")

    if round_number == -1:
        round_number = NUM_ROUNDS[season]
    if grand_prix:
        return round_number

    try:
        round_number = max(
            sprint_round
            for sprint_round in SPRINT_ROUNDS[season]
            if sprint_round <= round_number
        )
    except KeyError as exc:
        raise ValueError(f"{season} season doesn't have any sprint race.") from exc
    except ValueError as exc:
        # no sprint rounds before the requested round
        raise ValueError(
            f"{season} season only has these sprint races: {sorted(SPRINT_ROUNDS[season])}."
        ) from exc

    return round_number


@click.command()
@click.argument("season", nargs=1, default=CURRENT_SEASON, type=int)
@click.argument("round_number", nargs=1, default=-1, type=int)
@click.option(
    "--grand-prix/--sprint-race", "-g/-s", default=True, help="Default to grand prix."
)
@click.option("--update-readme", is_flag=True)
@click.option(
    "-r", "--reddit-machine", is_flag=True, help="Write plotted session name to text file."
)
def main(
    season: int, round_number: int, grand_prix: bool, update_readme: bool, reddit_machine: bool
):
    """
    Make the README suite of visualizations for the latest event.

    \b
    Args:
        SEASON (int): Default to the current season (equivalent to current calendar year).
        ROUND_NUMBER (int): Default to the last completed round of the matching session_type.
    """  # noqa: D301 click requires the \b sequence to not line wrap this paragraph
    global DOC_VISUALS_PATH

    round_number = process_round_number(season, round_number, grand_prix)

    session_type = "R" if grand_prix else "S"
    session = f.get_session(season, round_number, session_type)
    session.load(telemetry=False, weather=False, messages=False)
    event_name = f"{session.event['EventName']} - {session.name}"

    dest = ROOT_PATH / "Visualizations" / f"{season}" / f"{event_name}"

    if dest.is_dir():
        if update_readme:
            shutil.copytree(dest, DOC_VISUALS_PATH, dirs_exist_ok=True)
            logger.info("Copied visualizations from %s to %s", dest, DOC_VISUALS_PATH)
            return

        overwrite_confirmation = input(
            (
                "WARNING:\n"
                f"{dest} may already contain the desired visualizations.\n"
                "Enter Y if you wish to overwrite them, otherwise, enter N: "
            )
        )
        if overwrite_confirmation.upper() != "Y":
            logger.info("Overwriting permission not given, aborting.")
            return
    else:
        Path.mkdir(dest, parents=True, exist_ok=True)

    logger.info("Visualizing %s", session)

    logger.info("Making podium gap graph...")
    podium_finishers = viz.get_drivers(session, drivers=3)
    race_winner = podium_finishers[0]
    viz.add_gap(race_winner, modify_global=True, season=season, session_type=session_type)
    viz.driver_stats_lineplot(
        season=season,
        event=round_number,
        session_type=session_type,
        drivers=podium_finishers,
        y=f"GapTo{race_winner}",
        grid="both",
    )
    plt.tight_layout()
    plt.savefig(dest / "podium_gap.png")

    logger.info("Making lap time graph...")
    viz.driver_stats_scatterplot(
        season=season, event=round_number, session_type=session_type, drivers=10
    )
    plt.tight_layout()
    plt.savefig(dest / "laptime.png")

    logger.info("Making strategy graph...")
    viz.strategy_barplot(
        season=season,
        event=round_number,
        session_type=session_type,
    )
    plt.tight_layout()
    plt.savefig(dest / "strategy.png")

    logger.info("Making position change graph...")
    viz.driver_stats_lineplot(
        season=season,
        event=round_number,
        session_type=session_type,
    )
    plt.tight_layout()
    plt.savefig(dest / "position.png")

    logger.info("Making teammate comparison boxplot...")
    viz.driver_stats_distplot(
        season=season,
        event=round_number,
        session_type=session_type,
        violin=False,
        swarm=False,
        teammate_comp=True,
    )
    plt.tight_layout()
    plt.savefig(dest / "teammate_box.png")

    logger.info("Making teammate comp violinplot...")
    viz.driver_stats_distplot(
        season=season,
        event=round_number,
        session_type=session_type,
        teammate_comp=True,
        upper_bound=7,
    )
    plt.tight_layout()
    plt.savefig(dest / "teammate_violin.png")

    logger.info("Making driver pace plot...")
    viz.driver_stats_distplot(
        season=season,
        event=round_number,
        session_type=session_type,
        upper_bound=7,
    )
    plt.tight_layout()
    plt.savefig(dest / "driver_pace.png")

    # use basic fastf1 to make team pace comparison plot
    logger.info("Making team pace comparison graph...")
    p.setup_mpl(misc_mpl_mods=False)

    laps = session.laps.pick_wo_box().pick_track_status("467", how="none")

    laps["LapTime (s)"] = laps["LapTime"].dt.total_seconds()
    team_order = (
        laps[["Team", "LapTime (s)"]]
        .groupby("Team")["LapTime (s)"]
        .median()
        .sort_values()
        .index
    )
    team_palette = {team: p.team_color(team) for team in team_order}

    _, ax = plt.subplots(figsize=(15, 10))
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
    plt.tight_layout()
    plt.savefig(dest / "team_pace.png")

    # Copy the visualizations
    if update_readme:
        shutil.copytree(dest, DOC_VISUALS_PATH, dirs_exist_ok=True)

    if reddit_machine:
        # write to temp text file
        Path(ROOT_PATH / "tmp").mkdir(exist_ok=True)
        with open(ROOT_PATH / "tmp" / "event_name.txt", "w") as fout:
            fout.write(event_name)


if __name__ == "__main__":
    main()
