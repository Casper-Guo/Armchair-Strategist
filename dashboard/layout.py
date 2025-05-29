"""Dash app static layout specifications."""

import dash_bootstrap_components as dbc
from dash import dcc, html


def upper_bound_slider(slider_id: str, **kwargs) -> dcc.Slider:  # noqa: ANN003
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


def lap_numbers_slider(slider_id: str, **kwargs) -> dcc.RangeSlider:  # noqa: ANN003
    """Generate generic range slider for setting lap numbers."""
    return dcc.RangeSlider(
        min=1, step=1, allowCross=False, tooltip={"placement": "bottom"}, id=slider_id, **kwargs
    )


session_picker_row = dbc.Row(
    [
        dbc.Col(
            dcc.Dropdown(
                options=[],
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
    ],
)

add_gap_row = dbc.Row(
    dbc.Card(
        [
            dbc.CardHeader("Calculate gaps between drivers"),
            dbc.CardBody(
                [
                    dbc.Row(
                        dcc.Dropdown(
                            options=[],
                            value=[],
                            placeholder="Select drivers",
                            disabled=True,
                            multi=True,
                            id="gap-drivers",
                        )
                    ),
                    html.Br(),
                    dbc.Row(
                        dbc.Col(
                            dbc.Button(
                                "Add Gap",
                                color="success",
                                disabled=True,
                                n_clicks=0,
                                id="add-gap",
                            ),
                        )
                    ),
                ]
            ),
        ]
    )
)

strategy_hint = dbc.Alert(
    [
        html.H4("Hint", className="alert-heading"),
        html.P("Try reloading the session if no plot is shown."),
    ],
    color="info",
    dismissable=True,
)

strategy_tab = dbc.Tab(
    dbc.Card(
        dbc.CardBody([strategy_hint, html.Br(), dcc.Loading(dcc.Graph(id="strategy-plot"))])
    ),
    label="Strategy",
)

scatter_y_options = [
    {"label": "Lap Time", "value": "LapTime"},
    {"label": "Seconds to Same Condition (Dry/Wet) Median", "value": "DeltaToRep"},
    {"label": "Percent from Same Condition (Dry/Wet) Median", "value": "PctFromRep"},
    {"label": "Seconds to Fastest", "value": "DeltaToFastest"},
    {"label": "Percent from Fastest", "value": "PctFromFastest"},
    {
        "label": "Seconds to Same Lap Median",
        "value": "DeltaToLapRep",
    },
    {
        "label": "Percent from Same Lap Median",
        "value": "PctFromLapRep",
    },
    {"label": "Fuel-Adjusted Lap Time", "value": "FuelAdjLapTime"},
]

scatterplot_tab = dbc.Tab(
    dbc.Card(
        dbc.CardBody(
            [
                dbc.Row(
                    dcc.Dropdown(
                        options=scatter_y_options,
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
    label="Stats Scatterplot",
)

line_y_options = [{"label": "Position", "value": "Position"}] + scatter_y_options

lineplot_tab = dbc.Tab(
    dbc.Card(
        dbc.CardBody(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            dcc.Dropdown(
                                options=line_y_options,
                                value="Position",
                                placeholder="Select the variable to put in y-axis",
                                clearable=False,
                                id="line-y",
                            ),
                            width=9,
                        ),
                        dbc.Col(
                            dbc.Checklist(
                                options=[
                                    {
                                        "label": "Show starting positions",
                                        "value": 1,
                                        "disabled": False,
                                    }
                                ],
                                value=[],
                                id="show-starting-grid",
                                inline=True,
                                switch=True,
                            ),
                            width=3,
                        ),
                    ],
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
    label="Stats Lineplot",
)

distplot_caveat = dbc.Alert(
    [
        html.H4("Caveats", className="alert-heading"),
        html.P(
            (
                "Only drivers who have completed more than 5 laps are shown. "
                "Try adjusting the slow lap filter if no plot is shown."
            )
        ),
    ],
    color="info",
    dismissable=True,
)

distplot_tab = dbc.Tab(
    dbc.Card(
        dbc.CardBody(
            [
                distplot_caveat,
                html.Br(),
                dbc.Row(
                    dcc.Dropdown(
                        options=[
                            {"label": " Show boxplot", "value": True},
                            {"label": " Show violin plot", "value": False},
                        ],
                        value=True,
                        clearable=False,
                        id="boxplot",
                    )
                ),
                html.Br(),
                dbc.Row(dcc.Loading(dcc.Graph(id="distplot"))),
                html.Br(),
                html.P("Filter out slow laps (default is 107% of the fastest lap):"),
                dbc.Row(upper_bound_slider(slider_id="upper-bound-dist")),
            ]
        )
    ),
    label="Lap Time Distribution Plot",
)

compound_plot_explanation = dbc.Alert(
    [
        html.H4("Methodology", className="alert-heading"),
        html.P(
            "The metric behind this graph is delta to lap representative time (DLRT). "
            "It is a measure of how good a lap time is compared to other cars on track "
            "at the same time, thus accounting for fuel load and track evolution."
        ),
        html.Hr(),
        html.P(
            "Since this metric is relative, this plot is best used for understanding "
            "how different compounds degrade at different rates."
        ),
    ],
    color="info",
    dismissable=True,
)

compound_plot_caveats = dbc.Alert(
    [
        html.H4("Caveats", className="alert-heading"),
        html.P(
            "The driver selections does not apply to this plot. "
            "This plot always considers laps driven by all drivers."
        ),
        html.Hr(),
        html.P(
            "Tyre life does not always correspond to stint length. "
            "As the same tyre may have been used in qualifying sessions."
        ),
        html.Hr(),
        html.P(
            # 5% is estimated as three drivers each completing one third race length
            "Only compounds that completed at least 5% of all laps can be shown. "
            "Outlier laps are filtered out."
        ),
        html.Hr(),
        html.P(
            "For each compound, the range of shown tyre life is limited by "
            "the number of drivers who completed a stint of that length. This is to avoid "
            "the plot being stretched by one driver doing a very long stint."
        ),
    ],
    color="info",
    dismissable=True,
)

compound_plot_tab = dbc.Tab(
    dbc.Card(
        dbc.CardBody(
            [
                compound_plot_explanation,
                compound_plot_caveats,
                html.Br(),
                dbc.Row(
                    dbc.Col(
                        dcc.Dropdown(
                            options=[
                                {"label": "Show delta as seconds", "value": True},
                                {"label": "Show delta as percentages", "value": False},
                            ],
                            value=True,
                            clearable=False,
                            placeholder="Select a unit",
                            id="compound-unit",
                        )
                    )
                ),
                html.Br(),
                dbc.Row(
                    dcc.Loading(
                        dcc.Dropdown(
                            options=[],
                            value=[],
                            placeholder="Select compounds",
                            disabled=True,
                            multi=True,
                            id="compounds",
                        )
                    )
                ),
                html.Br(),
                dbc.Row(dcc.Loading(dcc.Graph(id="compound-plot"))),
            ]
        )
    ),
    label="Compound Performance Plot",
)

compound_color_scheme_card = dbc.Card(
    [
        dbc.CardHeader(html.B("Compounds Color Scheme"), style={"textAlign": "center"}),
        dbc.ListGroup(
            [
                dbc.ListGroupItem(
                    html.B("SOFT"),
                    color="#da291c",
                    style={"textAlign": "center"},
                ),
                dbc.ListGroupItem(
                    html.B("MEDIUM"),
                    color="#ffd12e",
                    style={"textAlign": "center"},
                ),
                dbc.ListGroupItem(
                    html.B("HARD"),
                    color="#f0f0ec",
                    style={"textAlign": "center"},
                ),
                dbc.ListGroupItem(
                    html.B("INTERMEDIATE"),
                    color="#43b02a",
                    style={"textAlign": "center"},
                ),
                dbc.ListGroupItem(
                    html.B("WET"),
                    color="#0067ad",
                    style={"textAlign": "center"},
                ),
                dbc.ListGroupItem(
                    html.B("UNKNOWN"),
                    color="#00ffff",
                    style={"textAlign": "center"},
                ),
            ],
        ),
    ],
)

fresh_used_scheme_card = dbc.Card(
    [
        dbc.Accordion(
            [
                dbc.AccordionItem(
                    dbc.Progress(
                        value=100,
                        color="warning",
                        style={"height": "40px"},
                    ),
                    title="Fresh Tyre Bar",
                ),
                dbc.AccordionItem(
                    html.P("⚫", style={"textAlign": "center"}),
                    title="Fresh Tyre Marker",
                ),
                dbc.AccordionItem(
                    dbc.Progress(
                        value=100,
                        color="warning",
                        striped=True,
                        style={"height": "40px"},
                    ),
                    title="Used Tyre Bar (Striped)",
                ),
                dbc.AccordionItem(
                    html.P(html.B("X"), style={"textAlign": "center"}),
                    title="Used Tyre Marker",
                ),
            ],
            start_collapsed=True,
            always_open=True,
        ),
    ]
)

legends_tab = dbc.Tab(
    [compound_color_scheme_card, html.Br(), fresh_used_scheme_card],
    label="Graph Legends",
)

external_links = dbc.Alert(
    [
        "Made by ",
        html.A("Casper Guo", href="https://casper-guo.dev", className="alert-link"),
        " ✌️",
        html.Hr(),
        "All data provided by ",
        html.A(
            "FastF1",
            href="https://github.com/theOehrly/Fast-F1",
            className="alert-link",
        ),
        html.Hr(),
        "Feature requests and bug reports etc. are welcome at the ",
        html.A(
            "source repository",
            href="https://github.com/Casper-Guo/Armchair-Strategist",
            className="alert-link",
        ),
    ],
    color="info",
)

app_layout = dbc.Container(
    [
        html.H1("Armchair Strategist"),
        session_picker_row,
        dcc.Store(id="event-schedule"),
        dcc.Store(id="session-info"),
        dcc.Store(id="last-race-round"),
        dcc.Store(id="last-sprint-round"),
        dcc.Store(id="laps"),
        html.Br(),
        dbc.Row(
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
        ),
        html.Br(),
        add_gap_row,
        html.Br(),
        dbc.Tabs(
            [
                strategy_tab,
                scatterplot_tab,
                lineplot_tab,
                distplot_tab,
                compound_plot_tab,
                legends_tab,
            ]
        ),
        html.Br(),
        dbc.Row(external_links),
        # this component exists solely to enforce callback order
        html.Span(id="laps-data-sequencer", hidden=True),
    ]
)
