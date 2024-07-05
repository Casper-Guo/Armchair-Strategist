"""Produce Plotly graphs to be inserted with callbacks."""

import tomli
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from f1_visualization.visualization import pick_driver_color, _find_sc_laps
from pathlib import Path


with open(Path(__file__).absolute().parent / "visualization_config.toml", "rb") as toml:
    VISUAL_CONFIG = tomli.load(toml)


def _plot_args(season: int, absolute_compound: bool) -> tuple:
    """
    Get plotting arguments based on the season and compound type.

    Args:
        season: Championship season

        absolute_compound: If true, use absolute compound names
                           (C1, C2 ...) in legend
                           Else, use relative compound names
                           (SOFT, MEDIUM, HARD) in legend

    Returns:
        (hue, palette, marker, labels)
    """
    if absolute_compound:
        if season == 2018:
            return (
                "CompoundName",
                VISUAL_CONFIG["absolute"]["palette"]["18"],
                VISUAL_CONFIG["absolute"]["markers"]["18"],
                VISUAL_CONFIG["absolute"]["labels"]["18"],
            )
        if season < 2023:
            return (
                "CompoundName",
                VISUAL_CONFIG["absolute"]["palette"]["19_22"],
                VISUAL_CONFIG["absolute"]["markers"]["19_22"],
                VISUAL_CONFIG["absolute"]["labels"]["19_22"],
            )

        return (
            "CompoundName",
            VISUAL_CONFIG["absolute"]["palette"]["23_"],
            VISUAL_CONFIG["absolute"]["markers"]["23_"],
            VISUAL_CONFIG["absolute"]["labels"]["23_"],
        )

    return (
        "Compound",
        VISUAL_CONFIG["relative"]["palette"],
        VISUAL_CONFIG["relative"]["markers"],
        VISUAL_CONFIG["relative"]["labels"],
    )


def shade_sc_periods(fig: go.Figure, sc_laps: np.ndarray, vsc_laps: np.ndarray):
    """Shade SC and VSC periods."""
    sc_laps = np.append(sc_laps, [-1])
    vsc_laps = np.append(vsc_laps, [-1])

    def plot_periods(laps, label, hatch=None):
        start = 0
        end = 1

        while end < len(laps):
            # check if the current SC period is still ongoing
            if laps[end] == laps[end - 1] + 1:
                end += 1
            else:
                if end - start > 1:
                    # the latest SC period lasts for more than one lap
                    fig.add_vrect(
                        x0=laps[start] - 1,
                        x1=laps[end - 1] - 1,
                        opacity=0.25,
                        fillcolor="yellow",
                        annotation_text="SC",
                        annotation_position="top",
                        annotation={"font_size": 12, "font_color": "black"},
                    )
                else:
                    # end = start + 1, the latest SC period lasts only one lap
                    fig.add_vrect(
                        x0=laps[start] - 1,
                        x1=laps[end - 1] - 1,
                        opacity=0.5,
                        fillcolor="yellow",
                        annotation_text="VSC",
                        annotation_position="top",
                        annotation={"font_size": 12, "font_color": "black"},
                    )
                start = end
                end += 1

    plot_periods(sc_laps, "SC")
    plot_periods(vsc_laps, "VSC", "-")
    return fig


def strategy_barplot(
    included_laps: pd.DataFrame, season: int, drivers: list[str], absolute_compound: bool
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

    args = _plot_args(season, absolute_compound)

    # plotly puts the first trace at the bottom
    # so we need to reverse the list of the drivers to get them ordered by finishing position
    for driver in reversed(drivers):
        stints = driver_stints.loc[driver_stints["Driver"] == driver]

        for _, stint in stints.iterrows():
            fig.add_trace(
                go.Bar(
                    y=[driver],
                    x=[stint["StintLength"]],
                    orientation="h",
                    marker={"color": args[1][stint[args[0]]]},
                    marker_pattern_shape=VISUAL_CONFIG["fresh"]["hatch"][stint["FreshTyre"]],
                )
            )

    num_laps = included_laps["LapNumber"].max()
    fig = shade_sc_periods(fig, *_find_sc_laps(included_laps))
    fig.update_layout(
        barmode="stack",
        template="plotly_dark",
        showlegend=False,
        xaxis={
            "tickmode": "array",
            "tickvals": [1] + list(range(5, num_laps, 5)),
            "title": "Lap Number",
        },
        yaxis={"type": "category"},
    )
    return fig
