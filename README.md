# F1-Data-Visualization

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

A repository for engineered F1 data and visualization tools.

## Requirements

Use `pip install -r requirements.txt` to install all dependencies.

## General Information

### Data Source

All data sourced from the [FastF1](https://github.com/theOehrly/Fast-F1) package.

### Data Availability

Data from all races beginning in the 2018 season, excluding test sessions, are available. This repository will be regularly updated during the F1 season.

You can use `data_loading.ipynb` and `data_transformation.ipynb` to renew your local data.

### Usage

There are six plotting functions provided in `visualizations.ipynb`. Documentation and examples are provided in the same notebook. Here is an overview:

- `tyre_usage_pie`: Visualize the frequency of compound usage within select races or entire seasons.
- `driver_stats_scatterplot`: Visualize various pace data within a single race as a scatterplot.
- `driver_stats_lineplot`: Visualize various pace data within a single race as a lineplot.
- `driver_stats_distplot`: Visualize the distribution of various pace data within a single race as a violin plot with optional swarm plot.
- `strategy_barplot`: Visualize tyre strategies within a single race with SC and VSC periods highlighted as a barplot.
- `compounds_lineplot`: Visualize performance over time by compound in multiple races as line plots.
- `compounds_distribution`: Visualize performance distribution over time by compound in multiple races with either boxplots or violin plots.

There is also a helper function, `add_gap`, for calculating gap to a specific driver which can then be plotted.

### Metrics Definitions

Detailed metric definitions can be found in the `data_transformation.ipynb` file. All columns are accurate to three digits.

- All columns provided by the FastF1 [Laps](https://theoehrly.github.io/Fast-F1/core.html?highlight=session#fastf1.core.Laps) object. Note that the `LapTime` column is converted from the native timedelta type to float type equal to the total seconds in the timedelta entry.
- `RoundNumber`: (str) Round number of the event that the lap belongs to
- `EventName`: (str) Short / common name of the Grand Prix that the lap belongs to
- `IsSlick`: (bool) Whether the lap is completed on slick tyres
- `CompoundName`: (str) The name of the compound used for this lap (C1, C2, C3, C4, C5, INTERMEDIATE, WET)
- `IsValid`: (bool) See documentation
- `Position`: (int) drivers' positions at the end of each lap
- `DeltaToRep`: (float) Difference between the lap time and the representative lap time (see definition in the notebook) of the race in seconds
- `DeltaToFastest`: (float) Difference between the current lap time and fastest lap time of the race in seconds
- `DeltaToLapRep`: (float) Difference between the current lap time and the lap representative time in seconds
- `PctFromRep`: (float) Difference between the lap time and the representative lap time of the race as a percentage
- `PctFromFastest`: (float) Difference between the current lap time and fastest lap time of the race as a percentage
- `PctFromLapRep`: (float) Difference between the current lap time and the lap representative time as a percentage
