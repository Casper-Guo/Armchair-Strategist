# F1-Data-Visualization
A repository of engineered F1 data and visualization tools.

Website being built in this [repo](https://github.com/brianmakesthings/F1-Web-Server.git).

Repository under refactoring.

## Requirements 
Use `pip install -r requirements.txt` to install all dependencies.

## General Information

### Data Source
All data sourced from the [FastF1](https://github.com/theOehrly/Fast-F1) package.

### Data Availability
Data from all the races beginning in the 2018 season are available. This repository will be regularly updated during the F1 season.

You can use `data_loading.ipynb` and `data_transformation.ipynb` to renew your local data.

### Usage
There are five plotting functions provided in `visualizations.ipynb`. Their usage will hopefully be clear through the examples provided at the end of the notebook and accompanying documentation. Here is an overview:

- `tyre_usage_pie`: Visualize the frequency of compound usage within a season and allows filtering by events and drivers
- `plot_driver_lap_times`: Visualize driver lap time data within a single event. Allows selecting any number of drivers
- `plot_strategy_barplot`: Visualize tyre strategy within a single event with SC and VSC highlighted. Allows selecting any number of drivers
- `plot_compounds_lineplot`: Visualize tyre performance over time by compound as line charts. Allows selecting multiple events
- `plot_compounds_distribution`: Visualize tyre performance distribution over time by compound with boxplots or violin plots. Allows selecting multiple events

### Metrics Definitions
Detailed metric definitions can be found in the `data_transformation.ipynb` file. 

- All columns provided by the FastF1 [Laps](https://theoehrly.github.io/Fast-F1/core.html?highlight=session#fastf1.core.Laps) object
- `RoundNumber`: (str) Round number of the event that the lap belongs to 
- `EventName`: (str) Short / common name of the Grand Prix that the lap belongs to 
- `IsSlick`: (bool) Whether the lap is completed on slick tyres
- `CompoundName`: (str) The name of the compound used for this lap (C1, C2, C3, C4, C5, INTERMEDIATE, WET)
- `IsValid`: (bool) See documentation
- `DeltaToRep`: (timedelta) Time differential between the current lap time and the representative lap time of the event
- `DeltaToFastest`: (timedelta) Time differential between the current lap time and the fastest lap time of the event
- `PctFromRep`: (float) Percent difference between the current lap time and the representative lap time of the event, accurate to three digits
- `PctFromFastest`: (float) Percent difference between the current lap time and fastest lap time of the event, accurate to three digits
- `DeltaToLapRep`: (timedelta) Time differential between the current lap time and the representative lap time for all laps of the same lap number
- `PctFromLapRep`: (float) Percent difference between the current lap time and the representative lap time for all laps of the same lap number, accurate to three digits
- `sLapTime`: (float) Lap time in seconds, accurate to three digits
- `sDeltaToRep`: (float) `DeltaToRep` in seconds, accurate to three digits
- `sDeltaToFastest`: (float) `DeltaToFastest` in seconds, accurate to three digits
- `sDeltaToLapRep`: (float) `DeltaToLapRep` in seconds, accurate to three digits
