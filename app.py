"""Layout Dash app structure."""

from typing import TypeAlias

import dash_bootstrap_components as dbc
import fastf1 as f
import pandas as pd
from dash import Dash, Input, Output, State, callback, dcc, html
from plotly import graph_objects as go

from f1_visualization._consts import CURRENT_SEASON, SPRINT_FORMATS
from f1_visualization.plotly_dash.graphs import (
    stats_distplot,
    stats_lineplot,
    stats_scatterplot,
    strategy_barplot,
)
from f1_visualization.visualization import get_session_info, load_laps

Session_info: TypeAlias = tuple[int, str, list[str]]

# must not be modified
DF_DICT = load_laps()


def upper_bound_slider(slider_id: str, **kwargs) -> dcc.Slider:
    """Generate generic slider for setting upper bound."""
    return dcc.Slider(
        min=100,
        max=150,
        marks={i: str(i) for i in range(100, 116, 5)} | {150: "Show All"},
        value=107,
        tooltip={"placement": "top"},
        id=slider_id,
        **kwargs,
    )


def lap_numbers_slider(slider_id: str, **kwargs) -> dcc.RangeSlider:
    """Generate generic range slider for setting lap numbers."""
    return dcc.RangeSlider(
        min=1, step=1, allowCross=False, tooltip={"placement": "bottom"}, id=slider_id, **kwargs
    )


def configure_lap_numbers_slider(data: dict) -> tuple[int, list[int], dict[int, str]]:
    """Configure range slider based on the number of laps in a session."""
    if not data:
        return 60, [1, 60], {i: str(i) for i in range(1, 61, 5)}
    df = pd.DataFrame.from_dict(data)
    num_laps = df["LapNumber"].max()

    marks = {i: str(i) for i in [1] + list(range(5, num_laps + 1, 5))}
    return num_laps, [1, num_laps], marks


strategy_tab = dbc.Tab(
    dbc.Card(dbc.CardBody(dcc.Loading(dcc.Graph(id="strategy-plot")))), label="strategy"
)

scatterplot_tab = dbc.Tab(
    dbc.Card(
        dbc.CardBody(
            [
                dbc.Row(
                    dcc.Dropdown(
                        options=[
                            {"label": "Lap Time", "value": "LapTime"},
                            {"label": "Seconds to Median", "value": "DeltaToRep"},
                            {"label": "Percent from Median", "value": "PctFromRep"},
                            {"label": "Seconds to Fastest", "value": "DeltaToFastest"},
                            {"label": "Percent from Fastest", "value": "PctFromFastest"},
                            {
                                "label": "Seconds to Adjusted Representative Time",
                                "value": "DeltaToLapRep",
                            },
                            {
                                "label": "Percent from Adjusted Representative Time",
                                "value": "PctFromLapRep",
                            },
                        ],
                        value="LapTime",
                        placeholder="Select the variable to put in y-axis",
                        clearable=False,
                        id="scatter-y",
                    )
                ),
                html.Br(),
                dbc.Row(dcc.Loading(dcc.Graph(id="scatterplot"))),
                html.Br(),
                html.P("Filter out slow laps (default is 107% of the fastest lap):"),
                dbc.Row(upper_bound_slider(slider_id="upper-bound-scatter")),
                html.Br(),
                html.P("Select the range of lap numbers to include:"),
                dbc.Row(lap_numbers_slider(slider_id="lap-numbers-scatter")),
            ]
        )
    ),
    label="scatterplot",
)

lineplot_tab = dbc.Tab(
    dbc.Card(
        dbc.CardBody(
            [
                dbc.Row(
                    dcc.Dropdown(
                        options=[
                            {"label": "Position", "value": "Position"},
                            {"label": "Lap Time", "value": "LapTime"},
                            {"label": "Seconds to Median", "value": "DeltaToRep"},
                            {"label": "Percent from Median", "value": "PctFromRep"},
                            {"label": "Seconds to Fastest", "value": "DeltaToFastest"},
                            {"label": "Percent from Fastest", "value": "PctFromFastest"},
                            {
                                "label": "Seconds to Adjusted Representative Time",
                                "value": "DeltaToLapRep",
                            },
                            {
                                "label": "Percent from Adjusted Representative Time",
                                "value": "PctFromLapRep",
                            },
                        ],
                        value="Position",
                        placeholder="Select the variable to put in y-axis",
                        clearable=False,
                        id="line-y",
                    )
                ),
                html.Br(),
                dbc.Row(dcc.Loading(dcc.Graph(id="lineplot"))),
                html.Br(),
                html.P("Filter out slow laps (default is 107% of the fastest lap):"),
                dbc.Row(upper_bound_slider(slider_id="upper-bound-line")),
                html.Br(),
                html.P("Select the range of lap numbers to include:"),
                dbc.Row(lap_numbers_slider(slider_id="lap-numbers-line")),
            ]
        )
    ),
    label="lineplot",
)

distplot_tab = dbc.Tab(
    dbc.Card(
        dbc.CardBody(
            [
                dbc.Row(
                    [
                        dcc.Dropdown(
                            options=[
                                {"label": " Show boxplot", "value": True},
                                {"label": " Show violin plot", "value": False},
                            ],
                            value=True,
                            clearable=False,
                            id="boxplot",
                        )
                    ]
                ),
                html.Br(),
                dbc.Row(dcc.Loading(dcc.Graph(id="distplot"))),
                html.Br(),
                html.P("Filter out slow laps (default is 107% of the fastest lap):"),
                dbc.Row(upper_bound_slider(slider_id="upper-bound-dist")),
            ]
        )
    ),
    label="Distribution Plot",
)

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.SANDSTONE],
    title="Armchair Strategist - A F1 Strategy Dashboard",
)
server = app.server
app.layout = dbc.Container(
    [
        html.H1("Armchair Strategist"),
        dbc.Row(
            [
                dbc.Col(
                    dcc.Dropdown(
                        options=list(range(CURRENT_SEASON, 2017, -1)),
                        placeholder="Select a season",
                        value=None,
                        id="season",
                    )
                ),
                dbc.Col(
                    dcc.Dropdown(
                        options=[],
                        placeholder="Select a event",
                        value=None,
                        id="event",
                    ),
                ),
                dbc.Col(
                    dcc.Dropdown(
                        options=[],
                        placeholder="Select a session",
                        value=None,
                        id="session",
                    ),
                ),
                dbc.Col(
                    dcc.Dropdown(
                        options=[
                            {"label": "Finishing order", "value": False},
                            {"label": "Teammate side-by-side", "value": True},
                        ],
                        value=False,
                        clearable=False,
                        id="teammate-comp",
                    )
                ),
                dbc.Col(
                    dbc.Button(
                        children="Load Session / Reorder Drivers",
                        n_clicks=0,
                        disabled=True,
                        color="success",
                        id="load-session",
                    )
                ),
                dcc.Store(id="event-schedule"),
                dcc.Store(id="session-info"),
                dcc.Store(id="laps"),
            ],
        ),
        html.Br(),
        dbc.Row(
            [
                dbc.Col(
                    dcc.Loading(
                        dcc.Dropdown(
                            options=[],
                            value=[],
                            placeholder="Select drivers",
                            disabled=True,
                            multi=True,
                            id="drivers",
                        )
                    )
                )
            ]
        ),
        html.Br(),
        dbc.Tabs([strategy_tab, scatterplot_tab, lineplot_tab, distplot_tab]),
        html.Br(),
        dbc.Row(
            dcc.Markdown("""
            All data provided by [FastF1](https://github.com/theOehrly/Fast-F1).

            Feature requests and bug reports etc. are welcome at the [source repository](https://github.com/Casper-Guo/Armchair-Strategist).
        """)
        ),
    ]
)


@callback(
    Output("event", "options"),
    Output("event", "value"),
    Output("event-schedule", "data"),
    Input("season", "value"),
)
def set_event_options(
    season: int | None,
) -> tuple[list[str], None, dict]:
    """Get the names of all events in the selected season."""
    if season is None:
        return [], None, None

    schedule = f.get_event_schedule(season, include_testing=False)

    if season == CURRENT_SEASON:
        # only include events for which we have processed data
        last_round = DF_DICT[CURRENT_SEASON]["R"]["RoundNumber"].max()
        schedule = schedule[schedule["RoundNumber"] <= last_round]

    return (
        list(schedule["EventName"]),
        None,
        schedule.set_index("EventName").to_dict(orient="index"),
    )


@callback(
    Output("session", "options"),
    Output("session", "value"),
    Input("event", "value"),
    State("event-schedule", "data"),
)
def set_session_options(event: str | None, schedule: dict) -> tuple[list[dict], None]:
    """
    Return the sessions contained in an event.

    Event schedule is passed in as a dictionary with the event names as keys. The values map
    column labels to the corresponding entry.
    """
    if event is None:
        return [], None

    return [
        {"label": "Race", "value": "R"},
        {
            "label": "Sprint",
            "value": "S",
            "disabled": schedule[event]["EventFormat"] not in SPRINT_FORMATS,
        },
    ], None


@callback(
    Output("load-session", "disabled"),
    Input("season", "value"),
    Input("event", "value"),
    Input("session", "value"),
)
def enable_load_session(season: int | None, event: str | None, session: str | None) -> bool:
    """Toggles load session button on when the previous three fields are filled."""
    return not (season is not None and event is not None and session is not None)


@callback(
    Output("drivers", "options"),
    Output("drivers", "value"),
    Output("drivers", "disabled"),
    Output("session-info", "data"),
    Output("laps", "data"),
    Input("load-session", "n_clicks"),
    State("season", "value"),
    State("event", "value"),
    State("session", "value"),
    State("teammate-comp", "value"),
)
def get_driver_list(
    n_clicks: int, season: int, event: str, session: str, teammate_comp: bool
) -> tuple[list[str], list, bool, Session_info, dict]:
    """
    Populate the drivers dropdown boxes.

    Since this requires loading the session, we will save some metadata at the same tine.
    """
    # return default values on startup
    if n_clicks == 0:
        return [], [], True, (), {}
    # We expect these three variables to be set subsequently
    round_number, event_name, drivers = get_session_info(
        season, event, session, drivers=20, teammate_comp=teammate_comp
    )

    included_laps = DF_DICT[season][session]
    included_laps = included_laps[included_laps["RoundNumber"] == round_number]

    return (
        drivers,
        drivers,
        False,
        (round_number, event_name, drivers),
        included_laps.to_dict(),
    )


@callback(
    Output("lap-numbers-scatter", "max"),
    Output("lap-numbers-scatter", "value"),
    Output("lap-numbers-scatter", "marks"),
    Input("laps", "data"),
)
def set_scatterplot_slider(data: dict) -> tuple[int, list[int], dict[int, str]]:
    """Set up scatterplot tab lap numbers slider."""
    return configure_lap_numbers_slider(data)


@callback(
    Output("lap-numbers-line", "max"),
    Output("lap-numbers-line", "value"),
    Output("lap-numbers-line", "marks"),
    Input("laps", "data"),
)
def set_lineplot_slider(data: dict) -> tuple[int, list[int], dict[int, str]]:
    """Set up lineplot tab lap numbers slider."""
    return configure_lap_numbers_slider(data)


@callback(
    Output("strategy-plot", "figure"),
    Input("drivers", "value"),
    State("season", "value"),
    State("laps", "data"),
    State("session-info", "data"),
)
def render_strategy_plot(
    drivers: list[str],
    season: int,
    included_laps: dict,
    session_info: Session_info,
) -> go.Figure:
    """Filter laps and configure strategy plot title."""
    # return empty figure on startup
    if not included_laps or not drivers:
        return go.Figure()
    included_laps = pd.DataFrame.from_dict(included_laps)
    included_laps = included_laps[included_laps["Driver"].isin(drivers)]

    event_name = session_info[1]
    fig = strategy_barplot(included_laps, season, drivers)
    fig.update_layout(title=event_name)
    return fig


@callback(
    Output("scatterplot", "figure"),
    Input("drivers", "value"),
    Input("scatter-y", "value"),
    Input("upper-bound-scatter", "value"),
    Input("lap-numbers-scatter", "value"),
    State("season", "value"),
    State("laps", "data"),
    State("session-info", "data"),
)
def render_scatterplot(
    drivers: list[str],
    y: str,
    upper_bound: float,
    lap_numbers: list[int],
    season: int,
    included_laps: dict,
    session_info: Session_info,
) -> go.Figure:
    """Filter laps and configure scatterplot title."""
    if not included_laps or not drivers:
        return go.Figure()

    minimum, maximum = lap_numbers
    lap_interval = range(minimum, maximum + 1)
    included_laps = pd.DataFrame.from_dict(included_laps)
    included_laps = included_laps[
        (included_laps["Driver"].isin(drivers))
        & (included_laps["PctFromFastest"] < (upper_bound - 100))
        & (included_laps["LapNumber"].isin(lap_interval))
    ]

    fig = stats_scatterplot(included_laps, season, drivers, y)
    event_name = session_info[1]
    fig.update_layout(title=event_name)

    return fig


@callback(
    Output("lineplot", "figure"),
    Input("drivers", "value"),
    Input("line-y", "value"),
    Input("upper-bound-line", "value"),
    Input("lap-numbers-line", "value"),
    State("laps", "data"),
    State("session-info", "data"),
)
def render_lineplot(
    drivers: list[str],
    y: str,
    upper_bound: float,
    lap_numbers: list[int],
    included_laps: dict,
    session_info: Session_info,
) -> go.Figure:
    """Filter laps and configure lineplot title."""
    if not included_laps or not drivers:
        return go.Figure()

    minimum, maximum = lap_numbers
    lap_interval = range(minimum, maximum + 1)
    included_laps = pd.DataFrame.from_dict(included_laps)

    # upper bound not filtered here because we need to identify SC/VSC laps
    # inside the function
    included_laps = included_laps[
        (included_laps["Driver"].isin(drivers))
        & (included_laps["LapNumber"].isin(lap_interval))
    ]

    fig = stats_lineplot(included_laps, drivers, y, upper_bound)
    event_name = session_info[1]
    fig.update_layout(title=event_name)

    return fig


@callback(
    Output("distplot", "figure"),
    Input("drivers", "value"),
    Input("upper-bound-dist", "value"),
    Input("boxplot", "value"),
    State("laps", "data"),
    State("session-info", "data"),
)
def render_distplot(
    drivers: list[str],
    upper_bound: int,
    boxplot: bool,
    included_laps: dict,
    session_info: Session_info,
) -> go.Figure:
    """Filter laps and render distribution plot."""
    if not included_laps or not drivers:
        return go.Figure()

    included_laps = pd.DataFrame.from_dict(included_laps)
    included_laps = included_laps[
        (included_laps["Driver"].isin(drivers))
        & (included_laps["PctFromFastest"] < (upper_bound - 100))
    ]

    fig = stats_distplot(included_laps, drivers, boxplot)
    event_name = session_info[1]
    fig.update_layout(title=event_name)

    return fig


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
