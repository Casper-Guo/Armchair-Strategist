# F1-Data-Visualization

A repository for engineered F1 data and visualization tools.

Website being built in this [repo](https://github.com/brianmakesthings/F1-Web-Server.git).

## Requirements

Use `pip install -r requirements.txt` to install all dependencies.

## General Information

### Data Source

All data sourced from the [FastF1](https://github.com/theOehrly/Fast-F1) package.

### Data Availability

Data from all races beginning in the 2018 season, excluding test sessions, are available. This repository will be regularly updated during the F1 season.

You can use `data_loading.ipynb` and `data_transformation.ipynb` to renew your local data.

### Usage

There are five plotting functions provided in `visualizations.ipynb`. Documentation and examples are provided in the same notebook. Here is an overview:

- `tyre_usage_pie`: Visualize the frequency of compound usage within select races or an entire season
- `driver_stats_scatterplot`: Visualize various driver pace data within a single race.
- `strategy_barplot`: Visualize tyre strategy within a single race with SC and VSC periods highlighted.
- `compounds_lineplot`: Visualize performance over time by compound as line charts. Allows plotting multiple races simultaneously for easier comparison
- `compounds_distribution`: Visualize performance distribution over time by compound with either boxplots or violin plots. Allows plotting multiple races simultaneously for easier comparison

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
