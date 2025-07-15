"""Shared type annotations."""

from typing import NamedTuple, TypeAlias

import fastf1
import matplotlib as mpl

Session: TypeAlias = fastf1.core.Session
Figure: TypeAlias = mpl.figure.Figure
Axes: TypeAlias = mpl.axes.Axes


class PlotArgs(NamedTuple):
    """Data class for plot styling configuration."""

    hue: str
    palette: dict[str, str]
    markers: dict[str, str]
    labels: list[str]
