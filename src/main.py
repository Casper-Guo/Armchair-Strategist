import warnings
from visualization import *


def main():
    root_path = Path(__file__).absolute().parents[1]

    # connect to cache
    cache_path = root_path / "Cache"
    f.Cache.enable_cache(cache_path)

    # plotting setup
    sns.set(rc={"figure.dpi": 300, "savefig.dpi": 300})
    plt.style.use("dark_background")

    # Suppress pandas SettingWithCopy warning
    pd.options.mode.chained_assignment = None

    # Suppress Seaborn false positive warnings
    warnings.filterwarnings("ignore")

    # Visualizations below


if __name__ == "__main__":
    main()
