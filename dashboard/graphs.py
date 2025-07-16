"""Produce Plotly graphs to be inserted with callbacks."""

from math import ceil
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import tomli
from fastf1.plotting import get_driver_color, get_driver_style
from plotly.subplots import make_subplots

from f1_visualization.annotations import PlotArgs, Session
from f1_visualization.visualization import find_sc_laps

with open(Path(__file__).absolute().parent / "visualization_config.toml", "rb") as toml:
    DASH_VISUAL_CONFIG = tomli.load(toml)


def _plot_args() -> PlotArgs:
    """Get plotting arguments based on the season and compound type."""
    return PlotArgs(
        "Compound",
        DASH_VISUAL_CONFIG["relative"]["palette"],
        DASH_VISUAL_CONFIG["relative"]["markers"],
        DASH_VISUAL_CONFIG["relative"]["labels"],
    )


def shade_sc_periods(fig: go.Figure, sc_laps: np.ndarray, vsc_laps: np.ndarray) -> go.Figure:
    """Shade SC and VSC periods."""
    sc_laps = np.append(sc_laps, [-1])
    vsc_laps = np.append(vsc_laps, [-1])

    def plot_periods(laps: np.ndarray, annotation: Literal["SC", "VSC"]) -> None:
        start = 0
        end = 1

        while end < len(laps):
            # check if the current SC period is still ongoing
            if laps[end] == laps[end - 1] + 1:
                end += 1
            else:
                # if there is SC on laps 14, 15, 16
                # the shading should extend from 13 to 16
                fig.add_vrect(
                    x0=laps[start] - 1,
                    x1=laps[end - 1],
                    opacity=0.5,
                    fillcolor="yellow",
                    annotation_text=annotation,
                    annotation_position="top",
                    annotation={"font_size": 20, "font_color": "black"},
                )
                start = end
                end += 1

    plot_periods(sc_laps, "SC")
    plot_periods(vsc_laps, "VSC")
    return fig


def strategy_barplot(
    included_laps: pd.DataFrame,
    drivers: list[str],
) -> go.Figure:
    """Make horizontal stacked barplot of driver strategies."""
    fig = go.Figure()

    driver_stints = (
        included_laps[["Driver", "Stint", "Compound", "CompoundName", "FreshTyre", "LapNumber"]]
        .groupby(["Driver", "Stint", "Compound", "CompoundName", "FreshTyre"])
        .count()
        .reset_index()
    )
    driver_stints = driver_stints.rename(columns={"LapNumber": "StintLength"})
    driver_stints = driver_stints.sort_values(by=["Stint"])

    args = _plot_args()

    # plotly puts the first trace at the bottom
    # so we need to reverse the list of the drivers to get them ordered by finishing position
    for driver in reversed(drivers):
        stints = driver_stints.loc[driver_stints["Driver"] == driver]
        stint_num = 1
        for _, stint in stints.iterrows():
            fig.add_trace(
                go.Bar(
                    y=[driver],
                    x=[stint["StintLength"]],
                    orientation="h",
                    marker={"color": args.palette[stint[args.hue]]},
                    marker_pattern_shape=DASH_VISUAL_CONFIG["fresh"]["hatch"][
                        stint["FreshTyre"]
                    ],
                    name=(
                        f"Stint {stint_num}, "
                        f"{'Fresh' if stint['FreshTyre'] == 'True' else 'Used'}"
                    ),
                )
            )
            stint_num += 1

    num_laps = included_laps["LapNumber"].max()
    fig = shade_sc_periods(fig, *find_sc_laps(included_laps))
    fig.update_layout(
        barmode="stack",
        template="plotly_dark",
        xaxis={
            "tickmode": "array",
            "tickvals": list(range(5, int(num_laps), 5)),
            "title": "Lap Number",
        },
        yaxis={"type": "category"},
        showlegend=False,
        autosize=False,
        width=1250,
        height=max(300, 50 * len(drivers)),
    )
    return fig


def stats_scatterplot(
    included_laps: pd.DataFrame,
    drivers: list[str],
    y: str,
) -> go.Figure:
    """Make scatterplots showing a statistic, one subplot for each driver."""
    args = _plot_args()

    # LapRep columns have outliers that can skew the graph y-axis
    # The high outlier values are filtered by upper_bound
    # Using a lower bound of -5 on PctFromLapRep will retain 95+% of all laps
    if y in {"PctFromLapRep", "DeltaToLapRep"}:
        included_laps = included_laps[included_laps["PctFromLapRep"] > -5]  # noqa: PLR2004

    num_row = ceil(len(drivers) / 4)
    num_col = min(4, len(drivers))
    fig = make_subplots(
        rows=num_row,
        cols=num_col,
        horizontal_spacing=0.1 / num_row,
        vertical_spacing=0.1 / num_col,
        shared_xaxes="all",
        shared_yaxes="all",
        subplot_titles=drivers,
        x_title="Lap Number",
        y_title=y,
    )

    for index, driver in enumerate(drivers):
        driver_laps = included_laps[(included_laps["Driver"] == driver)]

        # the top left subplot is indexed (1, 1)
        row, col = divmod(index, 4)
        row += 1
        col += 1

        fig.add_trace(
            go.Scatter(
                x=driver_laps["LapNumber"],
                y=driver_laps[y],
                mode="markers",
                marker={
                    "color": driver_laps[args.hue].map(args.palette),
                    "symbol": driver_laps["FreshTyre"].map(
                        DASH_VISUAL_CONFIG["fresh"]["markers"]
                    ),
                },
                name=driver,
            ),
            row=row,
            col=col,
        )

    fig.update_layout(
        template="plotly_dark",
        showlegend=False,
        autosize=False,
        width=1250,
        height=400 if num_row == 1 else (300 * num_row),
    )
    return fig


def stats_lineplot(
    included_laps: pd.DataFrame,
    drivers: list[str],
    y: str,
    upper_bound: int,
    session: Session,
    starting_grid: dict[str, int],
) -> go.Figure:
    """Make lineplots showing a statistic."""
    # Identify SC and VSC laps before filtering for upper bound
    sc_laps, vsc_laps = find_sc_laps(included_laps)

    # keep all laps if y is position to eliminate any implicit gaps
    if y != "Position":
        included_laps = included_laps[included_laps["PctFromFastest"] <= (upper_bound - 100)]

    fig = go.Figure()

    # LapRep columns have outliers that can skew the graph y-axis
    # The high outlier values are filtered by upper_bound
    # Using a lower bound of -5 on PctFromLapRep will retain 95+% of all laps
    if y in {"PctFromLapRep", "DeltaToLapRep"}:
        included_laps = included_laps[included_laps["PctFromLapRep"] > -5]  # noqa: PLR2004

    for _, driver in enumerate(reversed(drivers)):
        driver_laps = included_laps[(included_laps["Driver"] == driver)]
        driver_line_style = get_driver_style(
            identifier=driver,
            session=session,
            style=[
                {"color": "auto", "dash": "solid"},
                {"color": "auto", "dash": "longdash"},
            ],
        )

        if starting_grid and y == "Position":
            fig.add_trace(
                go.Scatter(
                    x=pd.concat([pd.Series([0]), driver_laps["LapNumber"]]),
                    y=pd.concat([pd.Series([starting_grid[driver]]), driver_laps[y]]),
                    mode="lines",
                    line=driver_line_style,
                    name=driver,
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=driver_laps["LapNumber"],
                    y=driver_laps[y],
                    mode="lines",
                    line=driver_line_style,
                    name=driver,
                )
            )

    fig = shade_sc_periods(fig, sc_laps, vsc_laps)
    if y == "Position" or y.startswith("Gap"):
        fig.update_yaxes(autorange="reversed")

    num_laps = included_laps["LapNumber"].max()
    fig.update_layout(
        template="plotly_dark",
        xaxis={
            "tickmode": "array",
            "tickvals": list(range(5, int(num_laps), 5)),
            "title": "Lap Number",
        },
        yaxis_title=y,
        autosize=False,
        width=1250,
        height=max(50 * len(drivers), 500),
        legend_traceorder="reversed",
    )

    if y == "Position":
        fig.update_yaxes(tickmode="array", tickvals=[1, *list(range(5, 21, 5))])
    return fig


def stats_distplot(
    included_laps: pd.DataFrame, drivers: list[str], boxplot: bool, session: Session
) -> go.Figure:
    """Make distribution plot of lap times, either as boxplot or as violin plot."""
    fig = go.Figure()

    for driver in drivers:
        driver_laps = included_laps[included_laps["Driver"] == driver]

        if boxplot:
            fig.add_trace(
                go.Box(
                    y=driver_laps["LapTime"],
                    boxmean=True,
                    boxpoints="outliers",
                    pointpos=0,
                    fillcolor=get_driver_color(driver, session),
                    line={"color": "lightslategray"},
                    name=driver,
                    showwhiskers=True,
                )
            )
        else:
            fig.add_trace(
                go.Violin(
                    y=driver_laps["LapTime"],
                    fillcolor=get_driver_color(driver, session),
                    line={"color": "lightslategray"},
                    meanline_visible=True,
                    name=driver,
                    opacity=0.9,
                )
            )

    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Driver",
        yaxis_title="Lap Time (s)",
        showlegend=False,
        autosize=False,
        width=1250,
        height=500,
    )
    return fig


def compounds_lineplot(included_laps: pd.DataFrame, y: str, compounds: list[str]) -> go.Figure:
    """Plot compound degradation curve as a lineplot."""
    fig = go.Figure()
    yaxis_title = "Seconds to LRT" if y == "DeltaToLapRep" else "Percent from LRT"

    _, palette, marker, _ = _plot_args()
    max_stint_length = 0

    for compound in compounds:
        compound_laps = included_laps[included_laps["Compound"] == compound]

        # clip tyre life range to where there are at least three records
        # if a driver does a very long stint, not all of it will be plotted
        tyre_life_range = compound_laps.groupby("TyreLife").size()
        tyre_life_range = tyre_life_range[tyre_life_range >= 3].index  # noqa: PLR2004

        # use the max instead of the length because tyre life range is
        # not guaranteed to start at 0
        max_stint_length = max(max_stint_length, tyre_life_range.max())
        median_LRT = compound_laps.groupby("TyreLife")[y].median()  # noqa: N806
        median_LRT = median_LRT.loc[tyre_life_range]  # noqa: N806

        fig.add_trace(
            go.Scatter(
                x=tyre_life_range,
                y=median_LRT,
                line={"color": palette[compound]},
                marker={
                    "line": {"width": 1, "color": "white"},
                    "color": palette[compound],
                    "symbol": marker[compound],
                    "size": 8,
                },
                mode="lines+markers",
                name=compound,
            )
        )

    fig.update_layout(
        template="plotly_dark",
        xaxis={
            "tickmode": "array",
            "tickvals": list(range(5, int(max_stint_length), 5)),
            "title": "Tyre Age",
        },
        yaxis_title=yaxis_title,
        showlegend=False,
        autosize=False,
        width=1250,
        height=500,
    )
    return fig
