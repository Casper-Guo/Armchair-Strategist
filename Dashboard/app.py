"""Layout Dash app structure."""

from datetime import datetime, timedelta

import dash_bootstrap_components as dbc
import fastf1 as f
import pandas as pd
from dash import Dash, Input, Output, State, callback, html, dash_table, dcc
from plotly import graph_objects as go

from f1_visualization._consts import CURRENT_SEASON, SPRINT_FORMATS
from f1_visualization.visualization import get_session_info, load_laps

import graphs

# must not be modified
DF_DICT = load_laps()


tab0 = dbc.Tab(
    dbc.Card(
        dbc.CardBody(
            [
                dbc.Row(dash_table.DataTable(page_size=25, id="data-table")),
                dbc.Row(
                    dcc.Slider(
                        min=101,
                        max=120,
                        marks={i: str(i) for i in [101] + list(range(105, 121, 5))},
                        value=107,
                        tooltip={"placement": "top"},
                        id="upper-bound-debug",
                    )
                ),
                dbc.Row(
                    dcc.RangeSlider(
                        min=1,
                        step=1,
                        allowCross=False,
                        tooltip={"placement": "bottom"},
                        id="lap-numbers-debug",
                    )
                ),
            ]
        )
    ),
    label="debug",
)

tab1 = dbc.Tab(
    dbc.Card(dbc.CardBody(dcc.Loading(dcc.Graph(id="strategy-plot")))), label="strategy"
)

app = Dash(__name__, external_stylesheets=[dbc.themes.SANDSTONE])

app.layout = dbc.Container(
    [
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
                    dbc.Button(
                        children="Load Session",
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
                    dcc.Dropdown(
                        options=[],
                        value=[],
                        placeholder="Select drivers",
                        disabled=True,
                        multi=True,
                        id="drivers",
                    )
                ),
                dbc.Col(
                    dcc.RadioItems(
                        options=[
                            {"label": "Absolute Compound", "value": True},
                            {"label": "Relative Compound", "value": False},
                        ],
                        value=False,
                        inline=True,
                        id="absolute-compound",
                    )
                ),
            ]
        ),
        html.Br(),
        dbc.Tabs([tab0, tab1]),
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

    schedule = f.get_event_schedule(season)
    # drops testing sessions
    schedule = schedule[schedule["RoundNumber"] != 0]

    if season == CURRENT_SEASON:
        # TODO: calculate this properly
        # complicated by pandas incompatibility with some datetime calculation
        utc_six_hours_ago = datetime.now() - timedelta(hours=2)
        schedule = schedule[schedule["Session5DateUtc"] < utc_six_hours_ago]

    return (
        list(schedule["EventName"]),
        None,
        schedule.set_index("EventName").to_dict(orient="index"),
    )


@callback(
    Output("session", "options"),
    Output("session", "value"),
    Input("event", "value"),
    Input("event-schedule", "data"),
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
)
def get_driver_list(
    n_clicks: int, season: int, event: str, session: str
) -> tuple[list[str], list, bool, tuple[int, str, list[str]], dict]:
    """
    Populate the drivers dropdown boxes.

    Since this requires loading the session, we will save some metadata at the same tine.
    """
    # return default values on startup
    if n_clicks == 0:
        return [], [], True, (), {}
    # We expect these three variables to be set subsequently
    round_number, event_name, drivers = get_session_info(season, event, session, drivers=20)

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
    Output("lap-numbers-debug", "max"),
    Output("lap-numbers-debug", "value"),
    Output("lap-numbers-debug", "marks"),
    Input("laps", "data"),
)
def set_lap_numbers_slider(data: dict) -> tuple[int, list[int], dict[int, str]]:
    """Configure range slider based on the number of laps in a session."""
    if not data:
        return 20, [1, 20], {i: str(i) for i in range(1, 21, 5)}
    df = pd.DataFrame.from_dict(data)
    num_laps = df["LapNumber"].max()

    marks = {i: str(i) for i in [1] + list(range(5, num_laps + 1, 5))}
    return num_laps, [1, num_laps], marks


@callback(
    Output("data-table", "data"),
    Output("data-table", "columns"),
    Input("drivers", "value"),
    Input("upper-bound-debug", "value"),
    Input("lap-numbers-debug", "value"),
    State("laps", "data"),
)
def refresh_data_table(
    drivers: list[str], upper_bound: int, lap_numbers: list[int], data: dict
) -> dict:
    """
    Reload the data table after loading session.

    For debugging only.
    """
    if not data:
        # placeholder
        data = DF_DICT[2024]["R"]
        return data.to_dict("records"), [{"name": i, "id": i} for i in data.columns]

    df = pd.DataFrame.from_dict(data)
    minimum, maximum = lap_numbers
    lap_interval = range(minimum, maximum + 1)
    df = df[
        (df["Driver"].isin(drivers))
        & (df["PctFromFastest"] < (upper_bound - 100))
        & (df["LapNumber"].isin(lap_interval))
    ]
    return df.to_dict("records"), [{"name": i, "id": i} for i in df.columns]


@callback(
    Output("strategy-plot", "figure"),
    Input("drivers", "value"),
    Input("absolute-compound", "value"),
    State("season", "value"),
    State("laps", "data"),
    State("session-info", "data"),
)
def render_strategy_plot(
    drivers: list[str],
    absolute_compound: bool,
    season: int,
    included_laps: dict,
    session_info: tuple[int, str, list[str]],
) -> go.Figure:
    """Configure strategy plot title dynamically."""
    # return empty figure on startup
    if not included_laps:
        return go.Figure()
    included_laps = pd.DataFrame.from_dict(included_laps)
    included_laps = included_laps[included_laps["Driver"].isin(drivers)]

    event_name = session_info[1]
    fig = graphs.strategy_barplot(included_laps, season, drivers, absolute_compound)
    fig.update_layout(title=event_name)
    return fig


if __name__ == "__main__":
    app.run(debug=True)
