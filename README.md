# F1-Data-Visualization

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

Engineered F1 data from the 2018 season onwards and visualization tools.

![](docs/staic/laptime.png "laptime")
![](docs/staic/position.png "position")
![](docs/staic/tyre.png "tyre")
![](docs/staic/strategy.png "strategy")

## Requirements

Use `python3 -m pip install -r requirements.txt` to install all dependencies.

## Data Source

All data sourced from the [FastF1](https://github.com/theOehrly/Fast-F1) package.

## Data Availability

Data from all races beginning in the 2018 season, excluding test sessions, are available. This repository will be regularly updated during the F1 season.

## Metrics Definitions

See `SCHEMA.md` for details on the columns provided in `all_laps_*.csv` and `transformed_laps_*.csv` files.

## Directory Structure

```
.
├── Data
│   ├── all_laps_2018.csv
│   ├── all_laps_2019.csv
│   ├── all_laps_2020.csv
│   ├── all_laps_2021.csv
│   ├── all_laps_2022.csv
│   ├── all_laps_2023.csv
│   ├── compound_selection.toml
│   ├── transformed_laps_2018.csv
│   ├── transformed_laps_2019.csv
│   ├── transformed_laps_2020.csv
│   ├── transformed_laps_2021.csv
│   ├── transformed_laps_2022.csv
│   ├── transformed_laps_2023.csv
│   └── compound_selection.toml
└── Notebooks
    ├── data_loading.ipynb
    ├── data_transformation.ipynb
    └── visualization.ipynb
├── README.md
├── requirements.txt
├── SCHEMA.md
└── src
    ├── main.py
    ├── preprocess.py
    └── visualization.py
```

- `data_loading.ipynb` and `data_transformation.ipynb` will be deprecated in the future. Use `preprocess.py` instead to load and preprocess data.
- See `visualization.ipynb` for example usages and documentations of all plotting functions.
