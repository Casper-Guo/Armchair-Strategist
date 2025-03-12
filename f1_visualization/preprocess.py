"""Load and transform F1 data from the FastF1 API."""

import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import fastf1 as f
import numpy as np
import pandas as pd
from fastf1.core import InvalidSessionError, NoLapDataError
from fastf1.ergast.interface import ErgastError
from fastf1.req import RateLimitExceededError

from f1_visualization._consts import (
    COMPOUND_SELECTION,
    CURRENT_SEASON,
    DATA_PATH,
    GRAND_PRIX_ORDINAL,
    NUM_ROUNDS,
    SESSION_IDS,
    SESSION_NAMES,
    SPRINT_RACE_ORDINAL,
    SPRINT_ROUNDS,
    VISUAL_CONFIG,
)

logging.basicConfig(level=logging.INFO, format="%(filename)s\t%(levelname)s\t%(message)s")
logger = logging.getLogger(__name__)


class OutdatedTOMLError(Exception):  # noqa: N801
    """Raised when Data/compound_selection.toml is not up to date."""


def load_all_data(season: int, path: Path, session_type: str):
    """
    Load all available data in a season.

    Assumes:
        None of the data for the season is already loaded.

    Args:
        season: The season to load

        path: The path to a csv file where the data will be stored.

        session_type: Follow FastF1 session identifier convention
    """
    dfs = []
    schedule = f.get_event_schedule(season)
    all_rounds = set(range(1, NUM_ROUNDS[season] + 1))

    if session_type == "S":
        if season == CURRENT_SEASON:
            all_rounds = SPRINT_ROUNDS[CURRENT_SEASON].intersection(
                range(1, get_last_round(session_cutoff=SPRINT_RACE_ORDINAL) + 1)
            )
        else:
            all_rounds = SPRINT_ROUNDS.get(season, set())

    for i in sorted(all_rounds):
        try:
            session = f.get_session(season, i, session_type)
            session.load(telemetry=False)
        except (NoLapDataError, InvalidSessionError, RateLimitExceededError, ErgastError) as e:
            logger.error("Cannot load %s", session)
            raise e

        laps = session.laps
        laps["RoundNumber"] = i
        laps["EventName"] = schedule[schedule["RoundNumber"] == i]["EventName"].item()
        dfs.append(laps)

    if dfs:
        all_laps = pd.concat(dfs, ignore_index=True)
        all_laps.to_csv(path, index=False)
        logger.info("Finished loading %d season %s data.", season, SESSION_IDS[session_type])
    else:
        logger.info(
            "No data available for %d season %s yet.", season, SESSION_IDS[session_type]
        )


def update_data(season: int, path: Path, session_type: str):
    """
    Update the data for a season.

    Assumes:
        Some of that season's data is already loaded.

    Args:
        season: The season to update.

        path: The path to a csv file where some of that season's data

        should already by loaded.

        session_type: Follow FastF1 session identifier convention
    """
    existing_data = pd.read_csv(path, index_col=0, header=0)
    loaded_rounds = set(pd.unique(existing_data["RoundNumber"]))
    all_rounds = set(range(1, NUM_ROUNDS[season] + 1))

    if session_type == "S":
        if season == CURRENT_SEASON:
            all_rounds = SPRINT_ROUNDS[CURRENT_SEASON].intersection(
                range(1, get_last_round(session_cutoff=SPRINT_RACE_ORDINAL) + 1)
            )
        else:
            all_rounds = SPRINT_ROUNDS.get(season, set())

    missing_rounds = sorted(all_rounds.difference(loaded_rounds))

    if not missing_rounds:
        logger.info(
            "%d season %s data is already up to date.", season, SESSION_IDS[session_type]
        )
        return

    logger.info(
        "Existing %d season %s data coverage: %s",
        season,
        SESSION_IDS[session_type],
        loaded_rounds,
    )
    logger.info("Coverage to be added: %s", missing_rounds)

    schedule = f.get_event_schedule(season)
    dfs = []

    for i in missing_rounds:
        try:
            session = f.get_session(season, i, session_type)
            session.load(telemetry=False)
        except (NoLapDataError, InvalidSessionError, RateLimitExceededError, ErgastError) as e:
            logger.error("Cannot load %s", session)
            raise e

        laps = session.laps
        laps["RoundNumber"] = i
        laps["EventName"] = schedule.loc[schedule["RoundNumber"] == i]["EventName"].item()
        dfs.append(laps)

    if dfs:
        all_laps = pd.concat(dfs, ignore_index=True)
        all_laps.to_csv(path, mode="a", index=False, header=False)
    else:
        logger.info(
            "All available %d season %s data are already loaded",
            season,
            SESSION_IDS[session_type],
        )

    logger.info("Finished updating %d season %s data.", season, SESSION_IDS[session_type])

    return


def correct_dtype(df_laps: pd.DataFrame) -> pd.DataFrame:
    """
    Fix columns with incorrect data types or missing values.

    Requires:
        df_laps has the following columns: [`Time`,
                                            `LapTime`,
                                            `PitInTime`,
                                            `PitOutTime`,
                                            `IsPersonalBest`
                                            `TrackStatus`,
                                            `FreshTyre`
                                            ]

    Returns:
        The transformed dataframe.
    """
    # convert from object (string) to timedelta
    df_laps[["Time", "LapTime", "PitInTime", "PitOutTime"]] = df_laps[
        ["Time", "LapTime", "PitInTime", "PitOutTime"]
    ].apply(pd.to_timedelta)
    df_laps["LapTime"] = df_laps["LapTime"].dt.total_seconds()

    df_laps["IsPersonalBest"] = df_laps["IsPersonalBest"].fillna(value="False").astype(bool)

    # make sure TrackStatus is stored as ints so it can be easily converted to strings later
    df_laps["TrackStatus"] = df_laps["TrackStatus"].fillna(0.0).astype(int)

    df_laps["FreshTyre"] = df_laps["FreshTyre"].fillna("Unknown").astype(str)

    return df_laps


def fill_compound(df_laps: pd.DataFrame) -> pd.DataFrame:
    """
    Infer missing `Compound` values as `UNKNOWN`.

    Requires:
        df_laps has the `Compound` column.
    """
    df_laps["Compound"] = df_laps["Compound"].fillna(value="UNKNOWN")

    return df_laps


def parse_csv_path(path: Path) -> tuple[int, str, str]:
    """Parse a data path and calculate season, session, and type."""
    filename_splits = path.stem.split("_")
    season = int(filename_splits[-1])
    parent_dir = path.parent.name
    data_type = filename_splits[0]
    return season, SESSION_NAMES[parent_dir], data_type


def load_laps() -> defaultdict[int, defaultdict[str, pd.DataFrame]]:
    """
    Parse the data directory and load all available data csvs.

    Examples:
        grand_prix
            - all_grand_prix_laps_2024.csv
            - all_grand_prix_laps_2022.csv
            - transformed_grand_prix_laps_2022.csv
        sprint
            - all_sprint_laps_2024.csv
            - transformed_sprint_laps_2024.csv

        reads to
        {
            2024: {
                    S: {"all": df, "transformed": df},
                    R: {"all": df}
                  }
            2022: {R: {"all": df, "transformed": df}}
        }
    """
    df_dict = defaultdict(lambda: defaultdict(lambda: defaultdict()))

    for file in DATA_PATH.glob("**/*.csv"):
        season, session, data_type = parse_csv_path(file)

        df = pd.read_csv(file, header=0, true_values=["TRUE"], false_values=["FALSE"])

        if data_type == "all":
            df = df[
                [
                    "Time",
                    "Driver",
                    "DriverNumber",
                    "LapTime",
                    "LapNumber",
                    "Stint",
                    "PitOutTime",
                    "PitInTime",
                    "IsPersonalBest",
                    "Compound",
                    "TyreLife",
                    "FreshTyre",
                    "Team",
                    "TrackStatus",
                    "Position",
                    "IsAccurate",
                    "RoundNumber",
                    "EventName",
                ]
            ]
            df = correct_dtype(df)
            df = fill_compound(df)

        df = df[df["LapNumber"] != 0]
        df_dict[season][session][data_type] = df

    return df_dict


def add_is_slick(season: int, df_laps: pd.DataFrame) -> pd.DataFrame:
    """
    Add a `IsSlick` column to df_laps in place.

    All compounds that are not intermediate or wet are considered slick.

    Requires:
        df_laps has the `Compound` column.

    Returns:
        The modified dataframe.
    """
    slick_names = []

    if season == 2018:
        slick_names = VISUAL_CONFIG["slick_names"]["18"]
    else:
        slick_names = VISUAL_CONFIG["slick_names"]["19_"]

    df_laps["IsSlick"] = df_laps["Compound"].apply(lambda x: x in slick_names)

    return df_laps


def add_compound_name(
    df_laps: pd.DataFrame,
    season_selection: dict[str, dict[str, list[str]]],
    season: int,
) -> pd.DataFrame:
    """
    Infer the underlying compound names and add it to df_laps in place.

    Args:
        df_laps: A pandas dataframe containing data from a single season.

        season_selection: The underlying slick compounds selection for one particular season

        by Grand Prix round number, in the order from the softest to hardest.

        season: The season to which df_laps and compound_selection refer to.

    Requires:
        df_laps has the following columns: [`Compound`, `RoundNumber`]

    Returns:
        The modified dataframe.
    """
    if season == 2018:
        df_laps["CompoundName"] = df_laps["Compound"]

        return df_laps

    def convert_compound_name(row):
        compound_to_index = {"SOFT": 2, "MEDIUM": 1, "HARD": 0}

        try:
            if row.loc["Compound"] not in compound_to_index:
                return row.loc["Compound"]

            return season_selection[str(row.loc["RoundNumber"])][
                compound_to_index[row.loc["Compound"]]
            ]
        except KeyError as exc:
            # error handling for when compound_selection.toml is not up-to-date
            raise OutdatedTOMLError(
                (
                    "Compound selection record is missing for "
                    f"{season} season round {row.loc['RoundNumber']}"
                )
            ) from exc

    df_laps["CompoundName"] = df_laps.apply(convert_compound_name, axis=1)

    return df_laps


def convert_compound(df_laps: pd.DataFrame) -> pd.DataFrame:
    """
    Add the relative compound names (SOFT, MEDIUM, HARD) to 2018 data in place.

    The 2018 data only has the underlying compound names (ultrasoft etc.)
    but sometimes we want access to the relative compound names as well.

    Args:
        df_laps: A pandas dataframe containing data from the 2018 season.

    Requires:
        df_laps has the following columns: [1Compound1, `RoundNumber`].

    Example:
        2018 round 1 uses the following compound selection:
        ["ULTRASOFT", "SUPERSOFT", "SOFT"]
        So the following mapping is applied:
        {
            "ULTRASOFT": "SOFT",
            "SUPERSOFT": "MEDIUM",
            "SOFT": "HARD"
        }

    Returns:
        The 2018 dataframe, with the `Compound` column overwritten
        with relative compound names.
    """
    compounds_2018 = COMPOUND_SELECTION["2018"]

    def convert_helper(row):
        index_to_compound = {0: "SOFT", 1: "MEDIUM", 2: "HARD"}

        try:
            if row.loc["Compound"] not in VISUAL_CONFIG["slick_names"]["18"]:
                return row.loc["Compound"]

            return index_to_compound[
                compounds_2018[str(row.loc["RoundNumber"])].index(row.loc["Compound"])
            ]
        except KeyError as exc:
            # error handling for when compound_selection.toml is not up-to-date
            raise OutdatedTOMLError(
                (
                    "Compound selection record is missing for 2018 season round "
                    f"{row.loc['RoundNumber']}"
                )
            ) from exc

    df_laps["Compound"] = df_laps.apply(convert_helper, axis=1)

    return df_laps


def add_is_valid(df_laps: pd.DataFrame) -> pd.DataFrame:
    """
    Add a `IsValid` column in place to identify fast laps.

    Requires:
        df_laps has the following columns: [`IsSlick`, `IsAccurate`, `TrackStatus`]
    """
    df_laps["IsValid"] = (
        (df_laps["IsSlick"]) & (df_laps["IsAccurate"]) & (df_laps["TrackStatus"] == 1)
    )

    return df_laps


def add_rep_deltas(df_laps: pd.DataFrame) -> pd.DataFrame:
    """
    Add two columns that calculate the difference to the representative lap time.

    Requires:
        df_laps has the following columns: [`RoundNumber`, `LapTime`]
    """
    rep_times = (
        df_laps[df_laps["IsValid"]].groupby("RoundNumber")["LapTime"].median().round(decimals=3)
    )
    df_laps = df_laps.merge(
        rep_times,
        how="left",
        on="RoundNumber",
        suffixes=(None, "_Rep"),
        validate="many_to_one",
    )

    df_laps["DeltaToRep"] = (df_laps["LapTime"] - df_laps["LapTime_Rep"]).round(decimals=3)
    df_laps["PctFromRep"] = (
        (df_laps["LapTime"] - df_laps["LapTime_Rep"]) / df_laps["LapTime_Rep"] * 100
    ).round(decimals=3)

    return df_laps.drop(columns=["LapTime_Rep"])


def add_fastest_deltas(df_laps: pd.DataFrame) -> pd.DataFrame:
    """
    Add two columns that calculate the difference to the fastest lap time.

    Requires:
        df_laps has the following columns: [`RoundNumber`, `LapTime`]
    """
    fastest_times = df_laps[df_laps["IsPersonalBest"]].groupby("RoundNumber")["LapTime"].min()
    df_laps = df_laps.merge(
        fastest_times,
        how="left",
        on="RoundNumber",
        suffixes=(None, "_Fastest"),
        validate="many_to_one",
    )

    df_laps["DeltaToFastest"] = (df_laps["LapTime"] - df_laps["LapTime_Fastest"]).round(
        decimals=3
    )
    df_laps["PctFromFastest"] = (
        (df_laps["LapTime"] - df_laps["LapTime_Fastest"]) / df_laps["LapTime_Fastest"] * 100
    ).round(decimals=3)

    return df_laps.drop(columns=["LapTime_Fastest"])


def add_lap_rep_deltas(df_laps: pd.DataFrame) -> pd.DataFrame:
    """
    Add two columns that calculate the difference to the lap representative time.

    Requires:
        df_laps has the following columns: [`RoundNumber`, `LapTime`]
    """
    lap_reps = (
        df_laps.groupby(["RoundNumber", "LapNumber"])["LapTime"].median().round(decimals=3)
    )

    df_laps = df_laps.merge(
        lap_reps,
        how="left",
        on=["RoundNumber", "LapNumber"],
        suffixes=(None, "_LapRep"),
        validate="many_to_one",
    )

    df_laps["DeltaToLapRep"] = (df_laps["LapTime"] - df_laps["LapTime_LapRep"]).round(
        decimals=3
    )
    df_laps["PctFromLapRep"] = (
        (df_laps["LapTime"] - df_laps["LapTime_LapRep"]) / df_laps["LapTime_LapRep"] * 100
    ).round(decimals=3)

    return df_laps.drop(columns=["LapTime_LapRep"])


def find_diff(season: int, dfs: dict[str, pd.DataFrame], session_type: str) -> pd.DataFrame:
    """
    Find the rows present in all_laps but missing in transformed_laps.

    Args:
        season: championship season

        dfs: a dictionary where the key is either "all" or "transformed"

        session_type: Follow FastF1 session identifier convention

    Assumes:
        - all_laps have at least as many rows as transformed_laps
        - The ith row in transformed_laps correspond to the ith row in all_laps

    Returns:
        The part of all_laps that is missing in transformed_laps.
    """
    if len(dfs) == 1:
        # If there is only one pair, the key should be "all"
        assert "all" in dfs

        logger.info("%d: No transfromed_laps found", season)

        # If no transformed_laps is found, the entirety of all_laps is in the diff
        return dfs["all"]

    if len(dfs) == 2:
        # "all" should be the key for the first pair in items
        # but we will not rely on this
        assert "all" in dfs and "transformed" in dfs

        num_row_all = dfs["all"].shape[0]
        num_row_transformed = dfs["transformed"].shape[0]

        # see assumption
        assert num_row_all >= num_row_transformed

        if num_row_all == num_row_transformed:
            logger.info(
                "transformed_%s_laps_%d is up-to-date",
                SESSION_IDS[session_type],
                season,
            )
        else:
            logger.info(
                "%d rows will be added to transformed_%s_laps_%d",
                num_row_all - num_row_transformed,
                SESSION_IDS[session_type],
                season,
            )

        return dfs["all"].iloc[num_row_transformed:]

    raise ValueError("Unexpected input length")


def get_last_round(session_cutoff: int = GRAND_PRIX_ORDINAL) -> int:
    """
    Return the last finished round number in the current season.

    Args:
        session_cutoff: The round is considered completed when a set amount of time
                        (five hours currently) has past after this session.

                        For example, if it is set to 3 on a sprint weekend, then the
                        round would be considered completed after the sprint race, even
                        if the grand prix has not taken place.

    Assumes:
        - session_cutoff is between 1 and 5 inclusive.
    """
    current_schedule = f.get_event_schedule(CURRENT_SEASON)

    # only load a session that is at most five hours old
    # this allows for a max session length of three hours
    # and a two-hour window for Fastf1 to make the session available
    five_hours_past = datetime.now(timezone.utc) - timedelta(hours=5)

    # Numpy might issue a deprecation warning or user warning for this conversion
    # From numpy documentation:
    #
    # if the string contains a trailing timezone (A ‘Z’ or a timezone offset),
    # the timezone will be dropped and a User Warning is given
    # Datetime64 objects should be considered to be UTC and therefore have an offset of +0000
    #
    # This is our use case and the warning can be ignored
    five_hours_past = np.datetime64(five_hours_past)

    rounds_completed = current_schedule[
        current_schedule[f"Session{session_cutoff}DateUtc"] <= five_hours_past
    ]["RoundNumber"].max()

    if pd.isna(rounds_completed):
        rounds_completed = 0

    return rounds_completed


def transform(season: int, dfs: dict[str, pd.DataFrame], session_type: str):
    """
    Update transformed_laps if it doesn't match all_laps.

    Args:
        season: championship season

        dfs: a dictionary where the key is either "all" or "transformed"

        session_type: Follow FastF1 session identifier convention

    Effects:
        Write transformed csv to path
    """
    df_transform = find_diff(season, dfs, session_type)

    if df_transform.shape[0] != 0:
        df_transform = add_is_slick(season, df_transform)
        df_transform = add_compound_name(df_transform, COMPOUND_SELECTION[str(season)], season)

        if season == 2018:
            df_transform = convert_compound(df_transform)

        df_transform = add_is_valid(df_transform)
        df_transform = add_rep_deltas(df_transform)
        df_transform = add_fastest_deltas(df_transform)
        df_transform = add_lap_rep_deltas(df_transform)

        path = (
            DATA_PATH
            / SESSION_IDS[session_type]
            / f"transformed_{SESSION_IDS[session_type]}_laps_{season}.csv"
        )

        if Path.is_file(path):
            # if the file already exists, then don't need to write header again
            df_transform.to_csv(path, mode="a", index=False, header=False)
        else:
            df_transform.to_csv(path, index=False)


def main() -> int:
    """Load and transform all newly available data."""
    Path.mkdir(DATA_PATH / "sprint", parents=True, exist_ok=True)
    Path.mkdir(DATA_PATH / "grand_prix", parents=True, exist_ok=True)

    load_seasons = list(range(2018, CURRENT_SEASON + 1))
    num_race_completed = get_last_round()
    num_sprint_completed = get_last_round(session_cutoff=SPRINT_RACE_ORDINAL)

    logger.info(
        "Correctness Check: %d races of the %d season have been completed",
        num_race_completed,
        CURRENT_SEASON,
    )
    NUM_ROUNDS[CURRENT_SEASON] = num_race_completed

    logger.info(
        "Correctness Check: %d sprint races of the %d season have been completed",
        num_sprint_completed,
        CURRENT_SEASON,
    )

    for season in load_seasons:
        for session_type, session_name in SESSION_IDS.items():
            path = DATA_PATH / session_name / f"all_{session_name}_laps_{season}.csv"

            try:
                if Path.is_file(path):
                    update_data(season, path, session_type)
                else:
                    load_all_data(season, path, session_type)
            except OutdatedTOMLError as exc:
                # exc carries info about the first round number with missing compound info
                logger.warning("%s. All later rounds of the same season are not loaded", exc)

    # Suppress SettingWithCopy Warning
    pd.options.mode.chained_assignment = None

    data = load_laps()
    for season in data:
        for session_type, dfs in data[season].items():
            transform(season, dfs, session_type)

    return 0


if __name__ == "__main__":
    main()
