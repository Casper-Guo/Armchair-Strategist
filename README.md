# F1-Data-Visualization

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

Engineered F1 data from the 2018 season onwards and visualization tools. Visualizations automatically updated to reflect the latest race on the Monday after the race at midnight EDT.

## Visualizations of the Most Recent Race

<details>
<summary><b>Pit Stop Strategies</b></summary>

![](Docs/visuals/strategy.png "strategy")

</details>

<details>
<summary><b>Position Changes</b></summary>

![](Docs/visuals/position.png "position")

</details>

<details>
<summary><b>Point Finishers Race Pace</b></summary>

![](Docs/visuals/laptime.png "laptime")

</details>

<details>
<summary><b>Podium Finishers Gap to Winner</b></summary>

![](Docs/visuals/podium_gap.png "podium_gap")

</details>

<details>
<summary><b>Teammate Pace Comparisons</b></summary>

![](Docs/visuals/teammate_box.png "teammate_box")

Violinplot with all laptimes:
![](Docs/visuals/teammate_violin.png "teammate_violin")

</details>

<details>
<summary><b>Team Pace Comparisons</b></summary>

![](Docs/visuals/team_pace.png "team_pace")

</details>

## Requirements

Use `python3 -m pip install -r requirements.txt` to install all dependencies.

## Data Source

All data sourced from the [FastF1](https://github.com/theOehrly/Fast-F1) package.

## Data Availability

Data from all races beginning in the 2018 season, excluding test sessions, are available. This repository will be regularly updated during the F1 season.

## Metrics Definitions

See `SCHEMA.md` for details on the columns provided in `Data/all_laps_*.csv` and `Data/transformed_laps_*.csv` files.

## Important Files

- Use `src/main.py` or `Notebooks/visualization.ipynb` to make your own visualizations. `Notebooks/visualization.ipynb` contains some example visualizations towards the end of the file.
- `Notebooks/data_loading.ipynb` and `Notebooks/data_transformation.ipynb` are _planned for removal_. Prefer `src/preprocessing.py` for acquiring and processing data.
