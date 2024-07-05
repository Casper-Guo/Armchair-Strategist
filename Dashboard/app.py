"""Layout Dash app structure."""

from datetime import datetime, timedelta

import dash_bootstrap_components as dbc
import fastf1 as f
import pandas as pd
from dash import Dash, Input, Output, State, callback, dash_table, dcc

from f1_visualization._consts import CURRENT_SEASON, SPRINT_FORMATS
from f1_visualization.visualization import _filter_round_driver, get_session_info, load_laps

# must not be modified
DF_DICT = load_laps()

app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    dcc.Dropdown(
                        options=list(range(CURRENT_SEASON, 2017, -1)),
                        placeholder="Select a season",
                        value=None,
                        id="season-dropdown",
                    )
                ),
                dbc.Col(
                    dcc.Dropdown(
                        options=[],
                        placeholder="Select a event",
                        value=None,
                        id="event-dropdown",
                    ),
                ),
                dbc.Col(
                    dcc.Dropdown(
                        options=[],
                        placeholder="Select a session",
                        value=None,
                        id="session-dropdown",
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
        dbc.Row(
            [
                dbc.Col(
                    dcc.Dropdown(options=[], value=[], disabled=True, multi=True, id="drivers")
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
                dbc.Col(
                    dcc.RadioItems(
                        options=[
                            {"label": "Teammate Side-by-side", "value": True},
                            {"label": "Finishing Order", "value": False},
                        ],
                        value=False,
                        inline=True,
                        id="teammate-comp",
                    )
                ),
            ]
        ),
        dbc.Row(dash_table.DataTable(page_size=10, id="data-table")),
        # dbc.Row(
        #     [dbc.Col(dcc.Loading(type="default", children=dcc.Graph(id="strategy-barplot")))]
        # ),
        dbc.Row(
            dcc.Slider(
                min=101,
                max=120,
                marks={i: str(i) for i in [101] + list(range(105, 121, 5))},
                value=107,
                tooltip={"placement": "top"},
                id="upper-bound",
            )
        ),
        dbc.Row(
            dcc.RangeSlider(
                min=1,
                step=1,
                allowCross=False,
                id="lap-numbers",
            )
        ),
    ]
)


@callback(
    Output("event-dropdown", "options"),
    Output("event-dropdown", "value"),
    Output("event-schedule", "data"),
    Input("season-dropdown", "value"),
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
    Output("session-dropdown", "options"),
    Output("session-dropdown", "value"),
    Input("event-dropdown", "value"),
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
    Input("season-dropdown", "value"),
    Input("event-dropdown", "value"),
    Input("session-dropdown", "value"),
)
def enable_load_session(season: int | None, event: str | None, session: str | None) -> bool:
    """Toggles load session button on when the previous three fields are filled."""
    return not (season is not None and event is not None and session is not None)


@callback(
    Output("data-table", "data"),
    Output("data-table", "columns"),
    Input("laps", "data"),
    Input("drivers", "value"),
    Input("upper-bound", "value"),
    Input("lap-numbers", "value"),
)
def refresh_data_table(
    data: dict, drivers: list[str], upper_bound: int, lap_numbers: list[int]
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
    Output("drivers", "options"),
    Output("drivers", "value"),
    Output("drivers", "disabled"),
    Output("session-info", "data"),
    Output("laps", "data"),
    Input("load-session", "n_clicks"),
    State("season-dropdown", "value"),
    State("event-dropdown", "value"),
    State("session-dropdown", "value"),
    State("teammate-comp", "value"),
)
def get_driver_list(
    n_clicks: int, season: int, event: str, session: str, teammate_comp: bool
) -> tuple[list[str], list, bool, tuple[int, str, list[str]], dict]:
    """
    Populate the drivers dropdown boxes.

    Since this requires loading the session, we will save some metadata at the same tine.
    """
    # return default values on startup
    if n_clicks == 0:
        return [], [], True, (), {}
    # We expect these three variables to be set subsequently
    round_number, event_name, drivers = get_session_info(
        season, event, session, teammate_comp=teammate_comp, drivers=20
    )

    included_laps = DF_DICT[season][session]
    included_laps = _filter_round_driver(included_laps, round_number, drivers)

    return (
        drivers,
        [],
        False,
        (round_number, event_name, drivers),
        included_laps.to_dict(),
    )


@callback(
    Output("lap-numbers", "max"),
    Output("lap-numbers", "value"),
    Output("lap-numbers", "marks"),
    Input("laps", "data"),
)
def set_lap_numbers_slider(data: dict) -> tuple[int, list[int], dict[int, str]]:
    """Configure range slider based on the number of laps in a session."""
    if not data:
        return 20, [1, 20]
    df = pd.DataFrame.from_dict(data)
    num_laps = df["LapNumber"].max()

    marks = {i: str(i) for i in [1] + list(range(stop=num_laps + 1, step=5))[1:]}
    return num_laps, [1, num_laps], marks


if __name__ == "__main__":
    app.run(debug=True)
