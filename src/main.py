"""Visualization playground."""

import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# uncomment this
# import visualization as viz


def main():
    """Set up visualizations."""
    # plotting setup
    sns.set(rc={"figure.dpi": 300, "savefig.dpi": 300})
    plt.style.use("dark_background")

    # Suppress pandas SettingWithCopy warning
    pd.options.mode.chained_assignment = None

    # Suppress Seaborn false positive warnings
    # TODO: This is dangerous
    warnings.filterwarnings("ignore")

    # Visualizations below


if __name__ == "__main__":
    main()
