"""Plotting functions and other visualization helpers."""

import logging
from collections.abc import Iterable
from functools import lru_cache
from math import ceil
from typing import Literal

import fastf1 as f
import fastf1.plotting as p
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import rcParams

from f1_visualization.annotations import Axes, Figure, PlotArgs, Session
from f1_visualization.consts import (
    COMPOUND_SELECTION,
    DATA_PATH,
    SESSION_NAMES,
    VISUAL_CONFIG,
)

logging.basicConfig(level=logging.INFO, format="%(filename)s\t%(levelname)s\t%(message)s")
logger = logging.getLogger(__name__)


def _correct_dtype(df_laps: pd.DataFrame) -> pd.DataFrame:
    """
    Fix incorrectly parsed data types.

    Requires:
        df_laps has the following columns: [`Time`,
                                            `PitInTime`,
                                            `PitOutTime`,
                                            "TrackStatus`,
                                            "FreshTyre",
                                            ]
    """
    # convert from object (string) to timedelta
    df_laps[["Time", "PitInTime", "PitOutTime"]] = df_laps[
        ["Time", "PitInTime", "PitOutTime"]
    ].apply(pd.to_timedelta)

    # TrackStatus column makes more sense as strings
    df_laps["TrackStatus"] = df_laps["TrackStatus"].astype(str)

    # When there is no missing FreshTyre value, it will be parsed
    # as a boolean column. We need it as a string to be compatible
    # with the visualization_config
    df_laps["FreshTyre"] = df_laps["FreshTyre"].astype(str)

    return df_laps


def load_laps() -> dict[int, dict[str, pd.DataFrame]]:
    """Load transformed data by season."""
    dfs = {}

    for file in DATA_PATH.glob("**/transformed_*.csv"):
        season = int(file.stem.split("_")[-1])
        session_type = SESSION_NAMES[file.parent.name]
        df = pd.read_csv(
            file,
            header=0,
            true_values=["True"],
            false_values=["False"],
        )
        _correct_dtype(df)

        if season not in dfs:
            dfs[season] = {}

        dfs[season][session_type] = df

    return dfs


DF_DICT = load_laps()


def _find_legend_order(labels: Iterable[str]) -> list[int]:
    """
    Provide the index of a list of compounds sorted from soft to hard.

    Args:
        labels: A list of string representing the tyre compounds.

    Returns:
        A list of ints corresponding to the original index of the
        compound names if they were in sorted order (softest to hardest,
        slick compounds first).

    Examples:
        labels = ["MEDIUM", "HARD", "SOFT"]
        desired = ["SOFT", "MEDIUM", "HARD"]
        return [2, 0, 1]

        labels = ["C3", "C1", "WET"]
        desired = ["C1", "C3", "WET"],
        return [1, 0, 2]
    """
    old_indices = list(range(len(labels)))
    sorted_labels = []

    if any(name in labels for name in ("HYPERSOFT", "ULTRASOFT", "SUPERSOFT", "SUPERHARD")):
        # 2018 absolute compound names
        sorted_labels = VISUAL_CONFIG["absolute"]["labels"]["18"]
    elif any(label.startswith("C") for label in labels):
        # 19_22 absolute compound names
        sorted_labels = VISUAL_CONFIG["absolute"]["labels"]["19_22"]
    else:
        # default to relative names
        sorted_labels = VISUAL_CONFIG["relative"]["labels"]

    pos = [sorted_labels.index(label) for label in labels]

    return [old_index for _, old_index in sorted(zip(pos, old_indices, strict=True))]


def _plot_args(season: int, absolute_compound: bool) -> PlotArgs:
    """
    Get plotting arguments based on the season and compound type.

    Args:
        season: Championship season

        absolute_compound: If true, use absolute compound names
                           (C1, C2 ...) in legend
                           Else, use relative compound names
                           (SOFT, MEDIUM, HARD) in legend
    """
    if absolute_compound:
        if season == 2018:  # noqa: PLR2004
            return PlotArgs(
                "CompoundName",
                VISUAL_CONFIG["absolute"]["palette"]["18"],
                VISUAL_CONFIG["absolute"]["markers"]["18"],
                VISUAL_CONFIG["absolute"]["labels"]["18"],
            )
        if season < 2023:  # noqa: PLR2004
            return PlotArgs(
                "CompoundName",
                VISUAL_CONFIG["absolute"]["palette"]["19_22"],
                VISUAL_CONFIG["absolute"]["markers"]["19_22"],
                VISUAL_CONFIG["absolute"]["labels"]["19_22"],
            )
        if season < 2025:  # noqa: PLR2004
            return PlotArgs(
                "CompoundName",
                VISUAL_CONFIG["absolute"]["palette"]["23_24"],
                VISUAL_CONFIG["absolute"]["markers"]["23_24"],
                VISUAL_CONFIG["absolute"]["labels"]["23_24"],
            )
        return PlotArgs(
            "CompoundName",
            VISUAL_CONFIG["absolute"]["palette"]["25_"],
            VISUAL_CONFIG["absolute"]["markers"]["25_"],
            VISUAL_CONFIG["absolute"]["labels"]["25_"],
        )

    return PlotArgs(
        "Compound",
        VISUAL_CONFIG["relative"]["palette"],
        VISUAL_CONFIG["relative"]["markers"],
        VISUAL_CONFIG["relative"]["labels"],
    )


def get_drivers(
    session: Session,
    drivers: Iterable[str | int] | str | int | None = None,
    by: str = "Position",
) -> list[str]:
    """
    Find driver three-letter abbreviations.

    Assumes:
        session.results is sorted by finishing position

    Args:
        session: The race session object, relevant for determining finishing order.

        drivers: The following argument formats are accepted:
            - A single integer retrieve the highest ordered drivers
              e.g. drivers = 10 with by = "Position" will fetch the point finishiers

              drivers = 20 will return all available drivers
            - A string representing either the driver's three letter abbreviation
              or driver number.
              e.g. "VER" or "44"
            - A list of integers and/or strings representing either the driver's
              three letter abbreviation or driver number.
              e.g. ["VER", "44", 14]
            - None returns all drivers who appear in the session

        by: The key by which the drivers are sorted. Default is sorting by finishing position.
            See all available options in FastF1 `Session.results` documentation.

    Returns:
        The drivers' three-letter abbreviations, in the order requested.
        (Or in the case of int argument, in the finishing order.)
    """
    result = session.results.sort_values(by=by, kind="stable")
    if drivers is None:
        return list(result["Abbreviation"].unique())
    if isinstance(drivers, int):
        drivers = result["Abbreviation"].unique()[:drivers]
        return list(drivers)
    if isinstance(drivers, str):
        drivers = [drivers]

    ret = []
    for driver in drivers:
        if isinstance(driver, (int, float)):
            ret.append(session.get_driver(str(int(driver)))["Abbreviation"])
        else:
            ret.append(session.get_driver(driver)["Abbreviation"])

    return ret


def infer_ergast_data(session: Session) -> Session:
    """When Ergast API is not updated yet, some session results data need to be inferred."""
    session.load(laps=True, telemetry=False, weather=False)

    # The laps dataframe is returned in ascending LapNumber order per driver
    # This keeps the final lap of each driver
    final_laps = session.laps.drop_duplicates(subset="DriverNumber", keep="last")

    # Drivers who have completed more laps always finishes higher
    # For drivers who finished the same number of laps, the tie is broken by who finished
    # the lap earlier
    # The ignore_index=True option has the same effect as calling reset_index
    final_order = final_laps.sort_values(
        by=["LapNumber", "Time"], ascending=[False, True], ignore_index=True
    )

    # Convert from Pandas index 0-index to finishing order 1-index
    final_order["Position"] = final_order.index + 1
    final_order = final_order.set_index("DriverNumber")
    session.results.loc[:, ["Position"]] = final_order["Position"]

    # TODO: find a way to infer GridPosition as well
    return session


# From 2018 to the end of 2025, there will be roughly 256 total race-like sessions.
# This is equivalent to the number of possible combinations of (season, event, session_type)
# inputs as the dashboard sanitizes and blocks requests for non-existent sessions.
@lru_cache(maxsize=256)
def get_session_info(
    season: int,
    event: int | str,
    session_type: str,
    drivers: tuple[str | int] | str | int | None = None,
    teammate_comp: bool = False,
) -> tuple[int, str, tuple[str], Session]:
    """
    Retrieve session information based on season, event number/name, and session identifier.

    Args:
        season: Championship season

        event: Round number or event name. If this argument is provided as a string, then
        the fuzzy matching is done by Fastf1

        session_type: Currently support R for the race and S for sprint race

        drivers: See `get_drivers` for all accepted formats.

        teammate_comp: If True, the drivers are returned next to their teammates. Else,
        the drivers are returned in the finishing order.

    Returns:
        A tuple containing the round number, event name, and the drivers in the specified order.
    """
    session = f.get_session(season, event, session_type)
    session.load(laps=False, telemetry=False, weather=False)

    if session.results["Position"].isna().all():
        logger.warning(
            "Session results not available. Starting and finishing positions are inferred."
        )
        session = infer_ergast_data(session)

    round_number = session.event["RoundNumber"]
    event_name = f"{session.event['EventName']} - {session.name}"

    if teammate_comp:
        drivers = get_drivers(session, drivers, by="TeamName")
    else:
        drivers = get_drivers(session, drivers)

    return round_number, event_name, tuple(drivers), session


def add_gap(
    driver: str,
    df_laps: pd.DataFrame | None = None,
    modify_global: bool = False,
    **kwargs,  # noqa: ANN003
) -> pd.DataFrame:
    """
    Calculate the gap to a certain driver.

    Args:
        driver: The driver to whom the gaps will be calculated

        df_laps: The dataframe to modify. Default behaviored explained later

        modify_global: Copies the modified dataframe to the global variable DF_DICT
            - requires season and session_type to be provided as keyword arguments
            - overrides any provided df_laps
            - copies the return value to DF_DICT[season][session_type]

    Returns:
        Modified dataframe with the gap column under the name GapTo{driver}
    """
    assert not (
        not modify_global and df_laps is None
    ), "df_laps must be provided if not editing in-place."

    if modify_global:
        assert (
            "season" in kwargs and "session_type" in kwargs
        ), "Setting modify_global=True requires specifying season and session_type."
        season, session_type = kwargs["season"], kwargs["session_type"]
        df_laps = DF_DICT[season][session_type]

    assert driver.upper() in df_laps["Driver"].unique(), "Driver not available."

    df_driver = df_laps[df_laps["Driver"] == driver][["RoundNumber", "LapNumber", "Time"]]
    timing_column_name = f"{driver}Time"
    df_driver = df_driver.rename(columns={"Time": timing_column_name})

    # although the Time column has not had NaT value thus far
    # for consistency these are filled
    df_driver[timing_column_name] = df_driver[timing_column_name].ffill()

    df_laps = df_laps.merge(
        df_driver, how="left", on=["RoundNumber", "LapNumber"], validate="many_to_one"
    )
    df_laps[f"GapTo{driver}"] = (
        df_laps["Time"] - df_laps[timing_column_name]
    ).dt.total_seconds()
    df_laps = df_laps.drop(columns=timing_column_name)

    if modify_global:
        DF_DICT[kwargs["season"]][kwargs["session_type"]] = df_laps

    return df_laps


def remove_low_data_drivers(
    included_laps: pd.DataFrame, drivers: tuple[str], min_laps: int
) -> tuple[str]:
    """
    Return drivers who appear at least min_laps times in included_laps.

    Guarantees the return value is in the same order as the drivers argument.
    """
    lap_counts = included_laps["Driver"].value_counts()
    qualifying_drivers = []

    for driver in drivers:
        if lap_counts.get(driver, 0) >= min_laps:
            qualifying_drivers.append(driver)
        else:
            logger.info("Dropping %s for having less than %d laps.", driver, min_laps)
    return tuple(qualifying_drivers)


def teammate_comp_order(
    included_laps: pd.DataFrame, drivers: tuple[str], by: str
) -> tuple[str]:
    """
    Reorder teammates by the median gap in some metric in descending order.

    For example, if by is LapTime, then the teammates with the biggest median laptime
    difference will appear first.

    This function is not integrated with get_session_info because get_session_info operates
    at the session level, while teammate_comp_order considers a subset of laps.

    For example, in a mixed condition race with DNFs. We can use get_session_info to
    acquire the full list of drivers who participated in the session, while using
    teammate_comp_order to compare driver performance during the wet period.

    Assumes:
        - teammates are next to each other in the drivers tuple
          This assumption is enforced if drivers is returned from get_session_info
          with teammate_comp=True argument.
          This assumption may be invalidated by remove_low_data_drivers
        - by is a column in included_laps.
    """
    metric_median = included_laps.groupby("Driver")[by].median(numeric_only=True)
    team_median_gaps = []

    # deal with odd number of drivers case
    odd_driver_out = [] if len(drivers) % 2 == 0 else [drivers[-1]]

    # TODO: Python 3.12 added itertools.batched to simplify the following logic
    for i in range(0, len(drivers) - 1, 2):
        teammates = drivers[i], drivers[i + 1]
        if all(driver in metric_median for driver in teammates):
            median_gap = metric_median[teammates[0]] - metric_median[teammates[1]]
            if median_gap < 0:
                # faster driver on the left
                team_median_gaps.append([teammates, -median_gap])
            else:
                team_median_gaps.append([teammates[::-1], median_gap])
        else:
            team_median_gaps.append([teammates, 0])

    team_median_gaps.sort(key=lambda x: x[1], reverse=True)
    drivers = [driver for team in team_median_gaps for driver in team[0]]
    drivers.extend(odd_driver_out)
    return tuple(drivers)


def _lap_filter_sc(row: pd.Series) -> bool:
    """
    Check if any part of a lap is ran under safety car.

    Track status 4 stands for safety car.

    Caveats:
        Unsure if the lap after "safety car in this lap" will be included.
    """
    return "4" in row.loc["TrackStatus"] and row.loc["Position"] == 1


def _lap_filter_vsc(row: pd.Series) -> bool:
    """
    Check if any part of a lap is ran under virtual safety car.

    A lap is only counted if none of it is ran under full safety car

    Track status 6 is VSC deployed.
    Track status 7 is VSC ending.
    """
    return (("6" in row.loc["TrackStatus"]) or ("7" in row.loc["TrackStatus"])) and (
        "4" not in row.loc["TrackStatus"] and row.loc["Position"] == 1
    )


def find_sc_laps(df_laps: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Find the unique lap numbers that is ran under SC or VSC.

    The resulting arrays are sorted before they are returned.
    """
    sc_laps = np.sort(df_laps[df_laps.apply(_lap_filter_sc, axis=1)]["LapNumber"].unique())
    vsc_laps = np.sort(df_laps[df_laps.apply(_lap_filter_vsc, axis=1)]["LapNumber"].unique())

    return sc_laps, vsc_laps


def _shade_sc_periods(sc_laps: np.ndarray, vsc_laps: np.ndarray) -> None:
    """
    Shade SC and VSC periods.

    Args:
        sc_laps: Sorted array of integers indicating laps under safety car

        vsc_laps: sorted array of integers indicating laps under virtual safety car
    """
    sc_laps = np.append(sc_laps, [-1])
    vsc_laps = np.append(vsc_laps, [-1])

    def plot_periods(laps: np.ndarray, label: str, hatch: str | None = None) -> None:
        start = 0
        end = 1

        while end < len(laps):
            # check if the current SC period is still ongoing
            if laps[end] == laps[end - 1] + 1:
                end += 1
            else:
                plt.axvspan(
                    # if there is SC on laps 14, 15, 16
                    # the shading should extend from 13 to 16
                    xmin=laps[start] - 1,
                    xmax=laps[end - 1],
                    alpha=0.5,
                    color="orange",
                    # only produce one label in legend
                    label=label if start == 0 else "_",
                    hatch=hatch,
                )
                start = end
                end += 1

    plot_periods(sc_laps, "SC")
    plot_periods(vsc_laps, "VSC", "xx")


def _deduplicate_legend_labels(ax: Axes, **kwargs) -> None:  # noqa: ANN003
    """
    Add legend to the current plot after deduplicating labels.

    Useful when labelled elements are added one-by-one, such as when showing SC periods.
    """
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        deduplicate_labels_handles = dict(zip(labels, handles, strict=True))
        plt.legend(
            handles=deduplicate_labels_handles.values(),
            labels=deduplicate_labels_handles.keys(),
            **kwargs,
        )


def _convert_compound_name(
    season: int, round_number: int, compounds: Iterable[str]
) -> tuple[str]:
    """
    Convert relative compound names to absolute compound names.

    Args:
        season: Championship season

        round_number: Grand Prix round number.

        compounds: Relative compound names to convert.

    Examples:
        2023 round 1 selects C1, C2, C3 compounds.

        Then _convert_compound_name(
        2023, 1, ["SOFT", "HARD"]
        ) = ["C1", "C3"]
    """
    compound_to_index = {"SOFT": 2, "MEDIUM": 1, "HARD": 0}
    if season == 2018:  # noqa: PLR2004
        compound_to_index = {"SOFT": 0, "MEDIUM": 1, "HARD": 2}

    return_vals = [
        COMPOUND_SELECTION[str(season)][str(round_number)][compound_to_index[compound]]
        for compound in compounds
    ]

    return tuple(return_vals)


def _process_input(
    seasons: int | Iterable[int],
    events: int | str | Iterable[str | int],
    session_types: str | Iterable[str],
    y: str,
    compounds: Iterable[str],
    x: str,
    upper_bound: int | float,
    absolute_compound: bool,
) -> tuple[list[f.events.Event], list[pd.DataFrame]]:
    """
    Sanitize input parameters to compound plots.

    Returns:
        event_objects: List of event objects corresponding to each requested race

        included_laps_list: List of dataframes corresponding to each requested race
    """
    # unpack
    compounds = [compound.upper() for compound in compounds]

    for compound in compounds:
        assert compound in {
            "SOFT",
            "MEDIUM",
            "HARD",
        }, f"requested compound {compound} is not valid"

    if x not in {"LapNumber", "TyreLife"}:
        logger.warning(
            "Using %s as the x-axis is not recommended. (Recommended x: LapNumber, TyreLife)",
            x,
        )

    if not absolute_compound and len(events) > 1:
        logger.warning(
            """
            Different events may use different compounds under the same name!
            e.g. SOFT may be any of C3 to C5 dependinging on the event
            """
        )

    if isinstance(seasons, (int, str)):
        seasons = [seasons]

    if isinstance(events, (int, str)):
        events = [events]

    if isinstance(session_types, str):
        session_types = [session_types]

    assert (
        len(seasons) == len(events) == len(session_types)
    ), f"Arguments {seasons}, {events}, {session_types} have different lengths."

    # Combine seasons and events and get FastF1 event objects
    event_objects = [f.get_event(seasons[i], events[i]) for i in range(len(seasons))]

    included_laps_list = []

    for season, event, session_type in zip(seasons, event_objects, session_types, strict=True):
        df_all = DF_DICT[season][session_type]
        df_laps = df_all[
            (df_all["RoundNumber"] == event["RoundNumber"])
            & (df_all["IsValid"])
            & (df_all["Compound"].isin(compounds))
            & (df_all["PctFromFastest"] < upper_bound)
        ]

        # LapRep columns have outliers that can skew the graph y-axis
        # The high outlier values are filtered by upper_bound
        # Using a lower bound of -5 on PctFromLapRep will retain 95+% of all laps
        if y in {"PctFromLapRep", "DeltaToLapRep"}:
            df_laps = df_laps[df_laps["PctFromLapRep"] > -5]  # noqa: PLR2004

        included_laps_list.append(df_laps)

    return event_objects, included_laps_list


def driver_stats_scatterplot(
    season: int,
    event: int | str,
    session_type: str = "R",
    drivers: Iterable[str | int] | str | int | None = None,
    y: str = "LapTime",
    upper_bound: int | float = 10,
    absolute_compound: bool = False,
    teammate_comp: bool = False,
    lap_numbers: list[int] | None = None,
) -> Figure:
    """
    Visualize driver data during a race as a scatterplot.

    Args:
        season: Championship season

        event: Round number or name of the event.
        Name is fuzzy matched by fastf1.get_event().

        session_type: Follow Fastf1 session identifier convention.

        drivers: See `get_drivers` for all accepted formats.
        By default, all drivers are plotted.

        y: Name of the column to be used as the y-axis.

        upper_bound: The upper bound on included laps as a percentage of the fastest lap.
        By default, only laps that are within 110% of the fastest lap are plotted.

        absolute_compound: If true, group tyres by absolute compound names (C1, C2 etc.).
        Else, group tyres by relative compound names (SOFT, MEDIUM, HARD).

        teammate_comp: Toggles teammate comparison mode. See teammate_comp_order
        for explanation. If False, the drivers are plotted by finishing order
        (higher finishing to the left).

        lap_numbers: A list of consecutive lap numbers representing a segment of the event.
        Recommend constructing this argument from a range object.

    Caveat:
        Providing a list of numbers that is not consecutive as lap_numbers will cause
        undefined behavior.
    """
    plt.style.use("dark_background")
    fontdict = {
        "fontsize": rcParams["axes.titlesize"],
        "fontweight": rcParams["axes.titleweight"],
        "color": rcParams["axes.titlecolor"],
        "verticalalignment": "baseline",
        "horizontalalignment": "center",
    }

    if not isinstance(drivers, (int, str)) and drivers is not None:
        drivers = tuple(drivers)

    round_number, event_name, drivers, session = get_session_info(
        season, event, session_type, drivers, teammate_comp
    )
    included_laps = DF_DICT[season][session_type]
    included_laps = included_laps[
        (included_laps["RoundNumber"] == round_number) & (included_laps["Driver"].isin(drivers))
    ]

    if teammate_comp:
        drivers = teammate_comp_order(included_laps, drivers, y)

    if lap_numbers is not None:
        assert sorted(lap_numbers) == list(range(lap_numbers[0], lap_numbers[-1] + 1))
        included_laps = included_laps[included_laps["LapNumber"].isin(lap_numbers)]

    max_width = 4 if teammate_comp else 5
    num_row = ceil(len(drivers) / max_width)
    num_col = min(max_width, len(drivers))
    fig, axs = plt.subplots(
        nrows=num_row,
        ncols=num_col,
        sharey=True,
        sharex=True,
        figsize=(5 * num_col, 5 * num_row),
    )

    args = _plot_args(season, absolute_compound)

    # Prevent TypeError when only one driver is plotted
    if len(drivers) == 1:
        axs = np.array([axs])

    # LapRep columns have outliers that can skew the graph y-axis
    # The high outlier values are filtered by upper_bound
    # Using a lower bound of -5 on PctFromLapRep will retain 95+% of all laps
    if y in {"PctFromLapRep", "DeltaToLapRep"}:
        included_laps = included_laps[included_laps["PctFromLapRep"] > -5]  # noqa: PLR2004

    for index, driver in enumerate(drivers):
        row, col = divmod(index, max_width)

        ax = axs[row][col] if num_row > 1 else axs[col]

        driver_laps = included_laps[included_laps["Driver"] == driver]
        pit_in_laps = driver_laps[driver_laps["PitInTime"].notna()]["LapNumber"].to_numpy()

        # After pitstops are identified,
        # filter out laps that doesn't meet the upper_bound
        driver_laps = driver_laps[driver_laps["PctFromFastest"] < upper_bound]

        if driver_laps.shape[0] < 5:  # noqa: PLR2004
            logger.warning("%s HAS LESS THAN 5 LAPS ON RECORD FOR THIS EVENT", driver)

        sns.scatterplot(
            data=driver_laps,
            x="LapNumber",
            y=y,
            ax=ax,
            hue=args.hue,
            palette=args.palette,
            hue_order=args.labels,
            style="FreshTyre",
            style_order=["True", "False", "Unknown"],
            markers=VISUAL_CONFIG["fresh"]["markers"],
            legend="auto" if index == num_col - 1 else False,
        )
        ax.vlines(
            ymin=plt.yticks()[0][1],
            ymax=plt.yticks()[0][-2],
            x=pit_in_laps,
            label="Pitstop",
            linestyle="dashed",
        )

        driver_color = p.get_driver_color(driver, session)
        fontdict["color"] = driver_color
        ax.set_title(label=driver, fontdict=fontdict, fontsize=12)

        ax.grid(color=driver_color, which="both", axis="both")
        sns.despine(left=True, bottom=True)

    fig.suptitle(t=f"{season} {event_name}", fontsize=20)
    axs.flatten()[num_col - 1].legend(loc="best", fontsize=8, framealpha=0.5)

    return fig


def driver_stats_lineplot(
    season: int,
    event: int | str,
    session_type: str = "R",
    drivers: Iterable[str | int] | str | int | None = None,
    y: str = "Position",
    upper_bound: int | float | None = None,
    grid: Literal["both", "x", "y"] | None = None,
    lap_numbers: list[int] | None = None,
) -> Figure:
    """
    Visualize driver data during a race as a lineplot.

    Args:
        season: Championship season

        event: Round number or name of the event.
        Name is fuzzy matched by fastf1.get_event().

        session_type: Follow Fastf1 session identifier convention.

        drivers: See `get_drivers` for all accepted formats.
        By default, all drivers are plotted.

        y: Name of the column to be used as the y-axis.

        upper_bound: The upper bound on included laps as a percentage of the fastest lap.
        Defaults to none in signature to enable checking whether a value is explicitly passed.
        Usually, the value is set to 10 in function body.

        grid: Provided to plt.grid() axis argument.
        Leave empty to plot no grid.

        lap_numbers: A list of consecutive lap numbers representing a segment of the event.
        Recommend constructing this argument from a range object.

    Caveat:
        Providing a list of numbers that is not consecutive as lap_numbers will cause
        undefined behavior.
    """
    plt.style.use("dark_background")

    if not isinstance(drivers, (int, str)) and drivers is not None:
        drivers = tuple(drivers)

    round_number, event_name, drivers, session = get_session_info(
        season, event, session_type, drivers
    )
    starting_grid = dict(
        zip(session.results["Abbreviation"], session.results["GridPosition"], strict=True)
    )
    included_laps = DF_DICT[season][session_type]
    included_laps = included_laps[
        (included_laps["RoundNumber"] == round_number) & (included_laps["Driver"].isin(drivers))
    ]

    if lap_numbers is not None:
        assert sorted(lap_numbers) == list(range(lap_numbers[0], lap_numbers[-1] + 1))
        included_laps = included_laps[included_laps["LapNumber"].isin(lap_numbers)]

    sc_laps, vsc_laps = find_sc_laps(included_laps)

    if upper_bound is None:
        upper_bound = 100 if y == "Position" or y.startswith("GapTo") else 10

    # do upper bound filtering after SC periods are identified
    included_laps = included_laps[
        (included_laps["RoundNumber"] == round_number)
        & (included_laps["Driver"].isin(drivers))
        & (included_laps["PctFromFastest"] < upper_bound)
    ]

    # adjust plot size based on number of laps
    num_laps = included_laps["LapNumber"].nunique()
    fig, ax = plt.subplots(figsize=(ceil(num_laps * 0.25), 8))

    if y == "Position":
        plt.yticks(range(2, 21, 2))

    if y == "Position" or y.startswith("GapTo"):
        ax.invert_yaxis()

    for driver in drivers:
        driver_laps = included_laps[included_laps["Driver"] == driver]

        if driver_laps[y].count() == 0:
            # nothing to plot for this driver
            logger.warning("%s has no data entry for %s", driver, y)
            continue

        driver_style = p.get_driver_style(
            identifier=driver,
            session=session,
            style=[
                {"color": "auto", "linestyle": "solid"},
                {"color": "auto", "linestyle": (0, (5, 5))},
            ],
        )

        # stitch in driver starting position (lap zero)
        if y == "Position" and pd.notna(starting_grid[driver]):
            sns.lineplot(
                x=pd.concat([pd.Series([0]), driver_laps["LapNumber"]]),
                y=pd.concat([pd.Series([starting_grid[driver]]), driver_laps[y]]),
                ax=ax,
                errorbar=None,
                **driver_style,
            )
        else:
            sns.lineplot(driver_laps, x="LapNumber", y=y, ax=ax, errorbar=None, **driver_style)
        last_lap = driver_laps["LapNumber"].max()
        last_pos = driver_laps[y][driver_laps["LapNumber"] == last_lap].iloc[0]

        annotation_x_margin = driver_laps.shape[0] / 100
        ax.annotate(
            xy=(last_lap + annotation_x_margin, last_pos + 0.25),
            text=driver,
            color=p.get_driver_color(driver, session),
            fontsize=12,
        )
        sns.despine(left=True, bottom=True)

    # shade SC periods
    _shade_sc_periods(sc_laps, vsc_laps)
    _deduplicate_legend_labels(ax, loc="lower right", fontsize=10)

    if grid in {"both", "x", "y"}:
        plt.grid(visible=True, which="major", axis=grid)
    else:
        plt.grid(visible=False)

    fig.suptitle(t=f"{season} {event_name}", fontsize=20)
    return fig


def driver_stats_distplot(
    season: int,
    event: int | str,
    session_type: str = "R",
    drivers: Iterable[str | int] | str | int | None = None,
    y: str = "LapTime",
    upper_bound: float | int = 10,
    swarm: bool = True,
    violin: bool = True,
    absolute_compound: bool = False,
    teammate_comp: bool = False,
) -> Figure:
    """
    Visualize race data distribution as a violinplot or boxplot + optional swarmplot.

    Only drivers who have completed more than 5 laps are shown.

    Args:
        season: Championship season

        event: Round number or name of the event.
        Name is fuzzy matched by fastf1.get_event().

        session_type: Follow Fastf1 session identifier convention.

        drivers: See `get_drivers` for all accepted formats.
        By default, all drivers are plotted.

        y: Name of the column to be used as the y-axis.

        upper_bound: The upper bound on included laps as a percentage of the fastest lap.
        By default, only laps that are less than 10% slower than the fastest lap are plotted.

        swarm: Toggle swarmplot visibility.

        violin: Toggles between violinplot and boxplot.

        absolute_compound: If true, group tyres by absolute compound names (C1, C2 etc.).
        Else, group tyres by relative compound names (SOFT, MEDIUM, HARD).

        teammate_comp: Toggles teammate comparison mode. See teammate_comp_order
        for explanation. If False, the drivers are plotted by finishing order
        (higher finishing to the left).
    """
    plt.style.use("dark_background")

    if not isinstance(drivers, (int, str)) and drivers is not None:
        drivers = tuple(drivers)

    round_number, event_name, drivers, session = get_session_info(
        season, event, session_type, drivers, teammate_comp
    )

    included_laps = DF_DICT[season][session_type]
    included_laps = included_laps[
        (included_laps["RoundNumber"] == round_number)
        & (included_laps["Driver"].isin(drivers))
        & (included_laps["PctFromFastest"] < upper_bound)
    ]

    if teammate_comp:
        drivers = teammate_comp_order(included_laps, drivers, y)
    drivers = remove_low_data_drivers(included_laps, drivers, 6)

    # Adjust plot size based on number of drivers plotted
    fig, ax = plt.subplots(figsize=(len(drivers) * 1.5, 10))
    args = _plot_args(season, absolute_compound)

    driver_colors = [p.get_driver_color(driver, session) for driver in drivers]

    if violin:
        sns.violinplot(
            data=included_laps,
            x="Driver",
            y=y,
            inner=None,
            scale="area",
            palette=driver_colors,
            order=drivers,
        )
    else:
        sns.boxplot(
            data=included_laps,
            x="Driver",
            y=y,
            palette=driver_colors,
            order=drivers,
            whiskerprops={"color": "white"},
            boxprops={"edgecolor": "white"},
            medianprops={"color": "white"},
            capprops={"color": "white"},
            showfliers=False,
        )

    if swarm:
        sns.swarmplot(
            data=included_laps,
            x="Driver",
            y=y,
            hue=args.hue,
            palette=args.palette,
            order=drivers,
            linewidth=0,
            size=5,
        )

        handles, labels = ax.get_legend_handles_labels()
        order = _find_legend_order(labels)
        ax.legend(
            handles=[handles[idx] for idx in order],
            labels=[labels[idx] for idx in order],
            loc="best",
            title=args.hue,
            frameon=True,
            fontsize=10,
            framealpha=0.5,
        )

    ax.grid(visible=False)

    fig.suptitle(t=f"{season} {event_name}", fontsize=20)

    return fig


def strategy_barplot(
    season: int,
    event: int | str,
    session_type: str = "R",
    drivers: Iterable[str] | int | None = None,
    absolute_compound: bool = False,
) -> Figure:
    """
    Visualize tyre strategies as a horizontal barplot.

    Args:
        season: Championship season

        event: Round number or name of the event.
        Name is fuzzy matched by fastf1.get_event().

        session_type: Follow Fastf1 session identifier convention.

        drivers: See `get_drivers` for all accepted formats.
        By default, all drivers are plotted.

        absolute_compound: If true, group tyres by absolute compound names (C1, C2 etc.).
        Else, group tyres by relative compound names (SOFT, MEDIUM, HARD).
    """
    if not isinstance(drivers, int) and drivers is not None:
        drivers = tuple(drivers)

    round_number, event_name, drivers, _session = get_session_info(
        season, event, session_type, drivers
    )
    included_laps = DF_DICT[season][session_type]
    included_laps = included_laps[
        (included_laps["RoundNumber"] == round_number) & (included_laps["Driver"].isin(drivers))
    ]

    fig, ax = plt.subplots(figsize=(5, len(drivers) // 3 + 1))
    plt.style.use("dark_background")

    driver_stints = (
        included_laps[["Driver", "Stint", "Compound", "CompoundName", "FreshTyre", "LapNumber"]]
        .groupby(["Driver", "Stint", "Compound", "CompoundName", "FreshTyre"])
        .count()
        .reset_index()
    )
    driver_stints = driver_stints.rename(columns={"LapNumber": "StintLength"})
    driver_stints = driver_stints.sort_values(by=["Stint"])

    args = _plot_args(season, absolute_compound)

    for driver in drivers:
        stints = driver_stints.loc[driver_stints["Driver"] == driver]

        previous_stint_end = 0
        for _, stint in stints.iterrows():
            plt.barh(
                [driver],
                stint["StintLength"],
                left=previous_stint_end,
                color=args.palette[stint[args.hue]],
                edgecolor="black",
                fill=True,
                hatch=VISUAL_CONFIG["fresh"]["hatch"][stint["FreshTyre"]],
                label="Fresh" if stint["FreshTyre"] == "True" else "Used",
            )

            previous_stint_end += stint["StintLength"]

    _shade_sc_periods(*find_sc_laps(included_laps))
    _deduplicate_legend_labels(ax, loc="lower right", fontsize=10)

    plt.title(f"{season} {event_name}", fontsize=16)
    plt.xlabel("Lap Number")
    plt.grid(False)

    # Invert y-axis
    ax.invert_yaxis()

    # Remove frame from plot
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    return fig


def compounds_lineplot(
    seasons: int | Iterable[int],
    events: int | str | Iterable[int | str],
    session_types: str | Iterable[str] | None = None,
    y: str = "LapTime",
    compounds: Iterable[str] = ["SOFT", "MEDIUM", "HARD"],
    x: str = "TyreLife",
    upper_bound: int | float = 10,
    absolute_compound: bool = True,
) -> Figure:
    """
    Visualize compound performances as a lineplot.

    Caveats:
        Only laps with `IsValid=True` are considered

    Args:
        seasons: Championship seasons of the events

        events: A mix of round numbers or names of the events
        Name is fuzzy matched by fastf1.get_event()

        Each (season, event) pair should uniquely identify an event.

        session_types: Follow Fastf1 session identifier convention.

        y: The column to use as the y-axis.

        compounds: The compounds to plot.

        x: The column to use as the x-axis.
        `TyreLife` or `LapNumber` recommended.

        upper_bound: The upper bound on included laps as a percentage of the fastest lap.
        By default, only laps that are less than 10% slower than the fastest lap are plotted.

        absolute_compound: If true, group tyres by absolute compound names (C1, C2 etc.).
        Else, group tyres by relative compound names (SOFT, MEDIUM, HARD).
    """
    plt.style.use("dark_background")

    if isinstance(seasons, int):
        seasons = [seasons]

    event_objects, included_laps_list = _process_input(
        seasons, events, session_types, y, compounds, x, upper_bound, absolute_compound
    )

    fig, axs = plt.subplots(
        nrows=len(event_objects),
        sharex=True,
        ncols=1,
        figsize=(5, 5 * len(event_objects)),
    )

    # Prevent TypeError when only one event is plotted
    if len(event_objects) == 1:
        axs = [axs]

    # Copy compounds values
    # May need to convert from relative to absolute names when plotting
    compounds_copy = compounds.copy()

    for idx, event in enumerate(event_objects):
        ax = axs[idx]
        args = _plot_args(seasons[idx], absolute_compound)
        included_laps = included_laps_list[idx]
        medians = included_laps.groupby([args.hue, x])[y].median(numeric_only=True)

        round_number = event["RoundNumber"]
        event_name = event["EventName"]

        if absolute_compound:
            compounds_copy = _convert_compound_name(seasons[idx], round_number, compounds)

        for compound in compounds_copy:
            if compound in medians.index:
                sns.lineplot(
                    x=medians.loc[compound].index,
                    y=medians.loc[compound].values,
                    ax=ax,
                    color=args.palette[compound],
                    marker=args.markers[compound],
                    markersize=4,
                    label=compound,
                )
            else:
                logger.warning((
                    "%s is not plotted for %s %s because there is not enough data",
                    compounds[idx],
                    seasons[idx],
                    event_name,
                ))

        ax.set_ylabel(y, fontsize=12)

        handles, labels = ax.get_legend_handles_labels()
        order = _find_legend_order(labels)
        ax.legend(
            handles=[handles[i] for i in order],
            labels=[labels[i] for i in order],
            loc="best",
            title=args.hue,
            frameon=True,
            fontsize=10,
            framealpha=0.5,
        )

        ax.set_title(label=f"{seasons[idx]} {event_name}", fontsize=12)
        ax.grid(which="both", axis="y")
        sns.despine(left=True, bottom=True)

    # reorder compound names for title
    compounds = [compounds[i] for i in _find_legend_order(compounds)]

    fig.suptitle(t=" VS ".join(compounds), fontsize=14)

    return fig


def compounds_distplot(
    seasons: int | Iterable[int],
    events: int | str | Iterable[int | str],
    session_types: str | Iterable[str] | None = None,
    y: str = "LapTime",
    compounds: Iterable[str] = ["SOFT", "MEDIUM", "HARD"],
    violin_plot: bool = False,
    x: str = "TyreLife",
    upper_bound: int | float = 10,
    absolute_compound: bool = True,
) -> Figure:
    """
    Visualize compound performance as a boxplot or violinplot.

    Caveats:
        Only laps with `IsValid=True` are considered

    Args:
        seasons: Championship seasons of the events

        events: A mix of round numbers or names of the events
        Name is fuzzy matched by fastf1.get_event()

        Each (season, event) pair should uniquely identify an event.

        session_types: Follow Fastf1 session identifier convention.

        y: The column to use as the y-axis.

        compounds: The compounds to plot.

        violin_plot: Toggles violinplot and boxplot.

        x: The column to use as the x-axis.
        `TyreLife` or `LapNumber` recommended.

        upper_bound: The upper bound on included laps as a percentage of the fastest lap.
        By default, only laps that are less than 10% slower than the fastest lap are plotted.

        absolute_compound: If true, group tyres by absolute compound names (C1, C2 etc.).
        Else, group tyres by relative compound names (SOFT, MEDIUM, HARD).
    """
    plt.style.use("dark_background")

    if isinstance(seasons, int):
        seasons = [seasons]

    event_objects, included_laps_list = _process_input(
        seasons, events, session_types, y, compounds, x, upper_bound, absolute_compound
    )

    # adjust plot size based on the chosen x-axis
    x_ticks = max(laps[x].nunique() for laps in included_laps_list)
    fig, axs = plt.subplots(
        nrows=len(event_objects),
        sharex=True,
        ncols=1,
        figsize=(ceil(x_ticks * 0.75), 5 * len(event_objects)),
    )

    # Prevent TypeError when only one event is plotted
    if len(event_objects) == 1:
        axs = [axs]

    # Copy compounds values
    # May need to convert from relative to absolute names when plotting
    compounds_copy = compounds.copy()

    for idx, event in enumerate(event_objects):
        ax = axs[idx]
        args = _plot_args(seasons[idx], absolute_compound)
        included_laps = included_laps_list[idx]

        plotted_compounds = included_laps[args.hue].unique()
        event_name = event["EventName"]
        round_number = event["RoundNumber"]

        if absolute_compound:
            compounds_copy = _convert_compound_name(seasons[idx], round_number, compounds)

        for compound in compounds_copy:
            if compound not in plotted_compounds:
                logger.warning((
                    "%s is not plotted for %s %s because there is not enough data",
                    compounds[idx],
                    seasons[idx],
                    event_name,
                ))

        if violin_plot:
            sns.violinplot(
                data=included_laps, x=x, y=y, ax=ax, hue=args.hue, palette=args.palette
            )
        else:
            sns.boxplot(data=included_laps, x=x, y=y, ax=ax, hue=args.hue, palette=args.palette)

        ax.set_ylabel(y, fontsize=12)
        xticks = ax.get_xticks()
        xticks = [tick + 1 for tick in xticks if tick % 5 == 0]
        ax.set_xticks(xticks)
        ax.grid(which="both", axis="y")

        handles, labels = ax.get_legend_handles_labels()
        order = _find_legend_order(labels)
        ax.legend(
            handles=[handles[i] for i in order],
            labels=[labels[i] for i in order],
            loc="best",
            title=args.hue,
            frameon=True,
            fontsize=12,
            framealpha=0.5,
        )

        ax.set_title(label=f"{seasons[idx]} {event_name}", fontsize=12)
        sns.despine(left=True, bottom=True)

    # reorder compound names for title
    compounds = [compounds[i] for i in _find_legend_order(compounds)]

    fig.suptitle(t=" VS ".join(compounds), fontsize="16")

    return fig
