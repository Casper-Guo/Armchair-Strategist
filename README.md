# F1-Data-Visualization

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

This repository contains engineered F1 data for all grand prixs and sprint races from the 2018 season onwards and some handy visualization tools. Visualizations in the README are automatically updated to reflect the latest race on the Monday after the race at midnight EDT.

## Visualizations of the Most Recent Race/Examples

<details>
    <summary>
        <b>Pit Stop Strategies</b>
    </summary>
    <img src="Docs/visuals/strategy.png", alt="strategy">
    <details>
        <summary>
            <b>Function call:</b>
        </summary>
        <code>strategy_barplot(season, event)</code>
    </details>
</details>

<details>
    <summary>
        <b>Position Changes</b>
    </summary>
    <img src="Docs/visuals/position.png" alt="position">
    <details>
        <summary>
            <b>Function call:</b>
        </summary>
        <code>driver_stats_scatterplot(season, event, drivers=10)</code>
    </details>
</details>

<details>
    <summary>
        <b>Point Finishers Race Pace</b>
    </summary>
    <img src="Docs/visuals/laptime.png" alt="laptime">
    <details>
        <summary>
            <b>Function call:</b>
        </summary>
        <code>strategy_barplot(season, event)</code>
    </details>
</details>

<details>
    <summary>
        <b>Podium Finishers Gap to Winner</b>
    </summary>
    <img src="Docs/visuals/podium_gap.png">
    <details>
        <summary>
            <b>Function call:</b>
        </summary>
        See <code>src/readme_machine.py</code>
    </details>
</details>

<details>
    <summary>
        <b>Teammate Pace Comparisons</b>
    </summary>
    Boxplot visualization:
    <img src="Docs/visuals/teammate_box.png">
    <details>
        <summary>
            <b>Function call:</b>
        </summary>
        <code>driver_stats_distplot(season, event, violin=False, swarm=False, teammate_comp=True, drivers=20)</code>
    </details>
    Violinplot with all laptimes:
    <img src="Docs/visuals/teammate_violin.png">
    <details>
        <summary>
            <b>Function call:</b>
        </summary>
        <code>driver_stats_distplot(season, event, violin=False, swarm=False, teammate_comp=True, drivers=20)</code>
    </details>
</details>

<details>
    <summary>
        <b>Team Pace Comparisons</b>
    </summary>
    <img src="Docs/visuals/team_pace.png">
    <details>
        <summary>
            <b>Function call:</b>
        </summary>
        See <code>src/readme_machine.py</code>
    </details>
</details>

## Requirements

Use `python3 -m pip install -r requirements.txt` to install all dependencies.

## Data Source

All data sourced from the [FastF1](https://github.com/theOehrly/Fast-F1) package.

## Data Availability

Data from all races beginning in the 2018 season, excluding test sessions, are available. This repository will be automatically updated during the F1 season.

## Metrics Definitions

See `SCHEMA.md` for details on the columns provided in `Data/all_laps_*.csv` and `Data/transformed_laps_*.csv` files.

## Usage Guide

- Use `src/adhoc_visuals.py` to make your own visualizations.

## Additional Examples
<details>
    <summary>
        <b>Tyre Degradation Lineplot</b>
    </summary>
    <img src="Docs/examples/tyre_line.png">
    <details>
        <summary>
            <b>Function call:</b>
        </summary>
        <code>compounds_lineplot(seasons, events)</code>
    </details>
</details>

<details>
    <summary>
        <b>Tyre Degradation Distribution Plot</b>
    </summary>
    <img src="Docs/examples/tyre_dist.png">
    <details>
        <summary>
            <b>Function call:</b>
        </summary>
        <code>compounds_distplot(seasons, events)</code>
    </details>
</details>
