import tomli
import fastf1 as f
import pandas as pd
from datetime import datetime
from pathlib import Path


root_path = Path(__file__).absolute().parents[1]
cache_path = root_path / "Cache"
data_path = root_path / "Data"
current_season = 2023
num_rounds = {2018: 21, 2019: 21, 2020: 17, 2021: 22, 2022: 22, 2023: 24}

with open(root_path / "Data" / "compound_selection.toml", "rb") as toml:
    compound_selection = tomli.load(toml)
with open(root_path / "Data" / "visualization_config.toml", "rb") as toml:
    visual_config = tomli.load(toml)


def load_all_data(season, path):
    # assumes there is no data for the season yet
    # data will be stored at the location specified by path as a csv

    race_dfs = []
    schedule = f.get_event_schedule(season)

    for i in range(1, num_rounds[season] + 1):
        race = f.get_session(season, i, 'R')
        race.load()
        laps = race.laps
        laps["RoundNumber"] = i
        laps["EventName"] = schedule[schedule["RoundNumber"]
                                     == i]["EventName"].item()
        race_dfs.append(laps)

    if race_dfs:
        all_laps = pd.concat(race_dfs, ignore_index=True)
        all_laps.to_csv(path)
        print(f"Finished loading {season} season data.")
    else:
        print(f"No data available for {season} season yet.")

    return None


def update_data(season, path):
    existing_data = pd.read_csv(path, index_col=0, header=0)

    schedule = f.get_event_schedule(season)

    loaded_rounds = set(pd.unique(existing_data["RoundNumber"]))
    newest_round = num_rounds[season]
    all_rounds = set(range(1, newest_round + 1))
    missing_rounds = all_rounds.difference(loaded_rounds)

    if not missing_rounds:
        print(f"{season} season is already up to date.")
        return None
    else:
        # correctness check
        print("Existing coverage: ", loaded_rounds)
        print("Coverage to be added: ", missing_rounds)

    race_dfs = []

    for i in missing_rounds:
        race = f.get_session(2022, i, 'R')
        race.load()
        laps = race.laps
        laps["RoundNumber"] = i
        laps["EventName"] = schedule.loc[schedule["RoundNumber"]
                                         == i]["EventName"].item()
        race_dfs.append(laps)

    all_laps = pd.concat(race_dfs, ignore_index=True)

    all_laps.to_csv(path, mode='a')
    print(f"Finished updating {season} season data.")
    return None


def read_csv(path):
    '''
    Read csv file at path location and filter for relevant columns

    Requires:
    csv file located at path location is derived from a fastf1 laps object
    '''

    return pd.read_csv(path,
                       header=0,
                       true_values=["TRUE"],
                       false_values=["FALSE"],
                       usecols=["Time", "DriverNumber", "LapTime", "LapNumber", "Stint",
                                "PitOutTime", "PitInTime", "IsPersonalBest", "Compound", "TyreLife", "FreshTyre",
                                "Team", "Driver", "TrackStatus", "IsAccurate", "RoundNumber",
                                "EventName"]
                       )


def correct_dtype(df_laps):
    '''
    Requires: 
    df_laps has the following columns: ["Time", "LapTime", "PitInTime", "PitOutTime", "IsPersonalBest"]
    '''

    # convert from object (string) to timedelta
    df_laps[["Time", "LapTime", "PitInTime", "PitOutTime"]] = df_laps[[
        "Time", "LapTime", "PitInTime", "PitOutTime"]].apply(pd.to_timedelta)
    df_laps["LapTime"] = df_laps["LapTime"].apply(lambda x: x.total_seconds())

    # convert from object (string) to bool
    # treat missing entries as False
    df_laps["IsPersonalBest"] = df_laps["IsPersonalBest"].fillna(value="False")
    df_laps["IsPersonalBest"] = df_laps["IsPersonalBest"].astype(bool)

    return df_laps


def fill_compound(df_laps):
    df_laps["Compound"] = df_laps["Compound"].fillna(value="UNKNOWN")

    return df_laps


def load_laps():
    df_dict = {}

    for file in Path.iterdir(root_path / "Data"):
        if file.suffix == ".csv":
            splits = file.stem.split("_")

            # "all" or "transformed"
            type = splits[0]
            season = int(splits[2])

            df = read_csv(file)

            if type == "all":
                correct_dtype(df)
                fill_compound(df)

            if season not in df_dict:
                df_dict[season] = {}

            df_dict[season][type] = df

    return df_dict


def add_is_slick(season, df_laps):
    '''
    Requires:
    df_laps has the following column: ["Compound"]
    '''

    slick_names = []

    if season == 2018:
        slick_names = visual_config["slick_names"]["18"]
    else:
        slick_names = visual_config["slick_names"]["19_22"]

    df_laps["IsSlick"] = df_laps["Compound"].apply(lambda x: x in slick_names)

    return None


def add_compound_name(df_laps, compound_selection, season):
    '''
    Requires:
    df_laps has the following columns: ["Compound", "RoundNumber"]

    Assumes:
        - all data contained in compound_selection is from the same season 
        - df_laps contain data from the same season as compound_selection
    '''
    if season == 2018:
        df_laps["CompoundName"] = df_laps["Compound"]

        return df_laps

    def convert_compound_name(row):
        compound_to_index = {"SOFT": 2, "MEDIUM": 1, "HARD": 0}

        try:
            if row.loc["Compound"] not in compound_to_index:
                return row.loc["Compound"]
            else:
                return compound_selection[str(row.loc["RoundNumber"])][compound_to_index[row.loc["Compound"]]]
        except KeyError:
            # error handling for when compound_selection.toml is not up-to-date
            print("Compound selection record is missing for round " +
                  str(row.loc["RoundNumber"]))

            # terminate cell
            assert False

    df_laps["CompoundName"] = df_laps.apply(convert_compound_name, axis=1)

    return df_laps


def convert_compound(df_laps):
    '''
    Requires:
        df_laps must be the 2018 dataset
        df_laps has the following columns: ["Compound", "RoundNumber"]
    '''

    compounds_2018 = compound_selection["2018"]

    def convert_compound(row):
        index_to_compound = {0: "SOFT", 1: "MEDIUM", 2: "HARD"}

        try:
            if row.loc["Compound"] not in visual_config["slick_names"]["18"]:
                return row.loc["Compound"]
            else:
                return index_to_compound[compounds_2018[str(row.loc["RoundNumber"])].index(row.loc["Compound"])]
        except KeyError:
            # error handling for when compound_selection.toml is not up-to-date
            print("Compound selection record is missing for 2018 season round " +
                  str(row.loc["RoundNumber"]))

            # terminate cell
            assert False

    df_laps["Compound"] = df_laps.apply(convert_compound, axis=1)

    return df_laps


def add_pos(df_laps):
    df_laps["Position"] = pd.Series(dtype="int")

    for round in range(1, int(df_laps["RoundNumber"].max()) + 1):
        df_round = df_laps[df_laps["RoundNumber"] == round]

        for lap in range(1, int(df_round["LapNumber"].max()) + 1):
            ranks = df_round[df_round["LapNumber"]
                             == lap]["Time"].rank(method="first")
            df_laps.loc[ranks.index, "Position"] = ranks.values

    return df_laps


def add_is_valid(df_laps):
    '''
    Requires:
    df_laps has the following columns: ["IsSlick", "IsAccurate", "TrackStatus"]
    '''

    def check_lap_valid(row):
        return row.loc["IsSlick"] and row.loc["IsAccurate"] and row.loc["TrackStatus"] == 1

    df_laps["IsValid"] = df_laps.apply(check_lap_valid, axis=1)

    return df_laps


def find_rep_times(df_laps):
    '''
    Requires:
    df_laps has the following columns: ["RoundNumber", "IsValid", "LapTime"]
    '''

    rounds = df_laps["RoundNumber"].unique()
    rep_times = {}

    for round_number in rounds:
        median = df_laps[(df_laps["RoundNumber"] == round_number) & (
            df_laps["IsValid"] == True)]["LapTime"].median()
        rep_times[round_number] = round(median, 3)

    return rep_times


def add_rep_deltas(df_laps):
    '''
    Requires:
    df_laps has the following columns: ["RoundNumber", "IsValid", "LapTime"]
    '''

    rep_times = find_rep_times(df_laps)

    def delta_to_rep(row):
        return row.loc["LapTime"] - rep_times[row.loc["RoundNumber"]]

    def pct_from_rep(row):
        delta = row.loc["LapTime"] - rep_times[row.loc["RoundNumber"]]
        return round(delta / rep_times[row.loc["RoundNumber"]] * 100, 3)

    df_laps["DeltaToRep"] = df_laps.apply(delta_to_rep, axis=1)
    df_laps["PctFromRep"] = df_laps.apply(pct_from_rep, axis=1)

    return df_laps


def find_fastest_times(df_laps):
    '''
    Requires:
    df_laps has the following columns: ["RoundNumber", "IsPersonalBest", "LapTime"]
    '''

    rounds = df_laps["RoundNumber"].unique()
    fastest_times = {}

    for round_number in rounds:
        fastest = df_laps[(df_laps["RoundNumber"] == round_number) & (
            df_laps["IsPersonalBest"] == True)]["LapTime"].min()
        fastest_times[round_number] = round(fastest, 3)

    return fastest_times


def add_fastest_deltas(df_laps):
    '''
    Requires:
    df_laps has the following columns: ["RoundNumber", "IsPersonalBest", "LapTime"]
    '''

    fastest_times = find_fastest_times(df_laps)

    def delta_to_fastest(row):
        return row.loc["LapTime"] - fastest_times[row.loc["RoundNumber"]]

    def pct_from_fastest(row):
        delta = row.loc["LapTime"] - fastest_times[row.loc["RoundNumber"]]
        return round(delta / fastest_times[row.loc["RoundNumber"]] * 100, 3)

    df_laps["DeltaToFastest"] = df_laps.apply(delta_to_fastest, axis=1)
    df_laps["PctFromFastest"] = df_laps.apply(pct_from_fastest, axis=1)

    return df_laps


def find_lap_reps(df_laps):
    '''
    Requires:
    df_laps has the following columns: ["RoundNumber", "LapNumber", "IsValid", "LapTime"]
    '''

    lap_reps = {}

    for round_number in df_laps["RoundNumber"].unique():
        round_lap_reps = {}
        round_laps = df_laps[df_laps["RoundNumber"] == round_number]
        lap_numbers = round_laps["LapNumber"].unique()

        for lap_number in lap_numbers:
            median = round_laps[round_laps["LapNumber"]
                                == lap_number]["LapTime"].median()
            round_lap_reps[lap_number] = round(median, 3)

        lap_reps[round_number] = round_lap_reps

    return lap_reps


def add_lap_rep_deltas(df_laps):
    '''
    Requires:
    df_laps has the following columns: ["RoundNumber", "LapNumber", "IsValid", "LapTime"]
    '''

    lap_reps = find_lap_reps(df_laps)

    def delta_to_lap_rep(row):
        return row.loc["LapTime"] - lap_reps[row.loc["RoundNumber"]][row.loc["LapNumber"]]

    def pct_from_lap_rep(row):
        delta = row.loc["LapTime"] - \
            lap_reps[row.loc["RoundNumber"]][row.loc["LapNumber"]]
        return round(delta / lap_reps[row.loc["RoundNumber"]][row.loc["LapNumber"]] * 100, 3)

    df_laps["DeltaToLapRep"] = df_laps.apply(delta_to_lap_rep, axis=1)
    df_laps["PctFromLapRep"] = df_laps.apply(pct_from_lap_rep, axis=1)

    return df_laps


def find_diff(items):
    '''
    Find the rows present in all_laps but missing in transformed_laps 
    for a given season

    Args:
        items: list
        list derived from a dict_items object containing all key value 
        pairs in a key in the return value of load_laps()

        i.e all dataframes associated with a certain F1 season

    Assumes:
        all_laps have at least as many rows as transformed_laps
        The ith row in transformed_laps correspond to the ith row in all_laps
    '''

    if len(items) == 1:
        # If there is only one pair, the key should be "all"
        assert items[0][0] == "all"

        print("No transfromed_laps found")

        # If no transformed_laps is found, the entirety of all_laps is in the diff
        return items[0][1]

    elif len(items) == 2:
        # "all" should be the key for the first pair in items
        # but we will not rely on this

        num_row_all = 0
        num_row_transformed = 0
        diff = pd.DataFrame()

        if items[0][0] == "all":
            num_row_all = items[0][1].shape[0]
            num_row_transformed = items[1][1].shape[0]
            diff = items[0][1]
        elif items[0][0] == "transformed":
            num_row_all = items[1][1].shape[0]
            num_row_transformed = items[0][1].shape[0]
            diff = items[1][1]
        else:
            raise ValueError("Unexpected key")

        # see assumption
        assert num_row_all >= num_row_transformed

        if num_row_all == num_row_transformed:
            print("transformed_laps is up-to-date")
        else:
            print(
                F"{num_row_all - num_row_transformed} rows will be added to transformed_laps")

        return diff.iloc[num_row_transformed:]
    else:
        raise ValueError("Unexpected input length")


def main():
    Path.mkdir(cache_path, exist_ok=True)
    f.Cache.enable_cache(cache_path)
    Path.mkdir(data_path, exist_ok=True)

    load_seasons = list(range(2018, current_season + 1))

    current_schedule = f.get_event_schedule(current_season)
    rounds_completed = current_schedule[current_schedule["EventDate"] < datetime.now(
    )]["RoundNumber"].max()

    if pd.isna(rounds_completed):
        rounds_completed = 0

    print((f"Correctness Check: {rounds_completed} rounds of the {current_season}"
           "season have been completed"))
    num_rounds[current_season] = rounds_completed

    for season in load_seasons:
        path = root_path / "Data" / ("all_laps_" + str(season) + ".csv")

        if Path.is_file(path):
            update_data(season, path)
        else:
            load_all_data(season, path)

    # Suppress SettingWithCopy Warning
    pd.options.mode.chained_assignment = None

    data = load_laps()

    for season, dfs in data.items():
        print(str(season) + ":")
        df_transform = find_diff(list(dfs.items()))

        if df_transform.shape[0] != 0:
            add_is_slick(season, df_transform)
            add_compound_name(df_transform,
                              compound_selection[str(season)],
                              season
                              )

            if season == 2018:
                convert_compound(df_transform)

            add_pos(df_transform)
            add_is_valid(df_transform)
            add_rep_deltas(df_transform)
            add_fastest_deltas(df_transform)
            add_lap_rep_deltas(df_transform)

            path = root_path / "Data" / "transformed_laps_{season}.csv"

            if Path.is_file(path):
                # if the file already exists, then don't need to write header again
                # need to move the cursor onto a new line before appending
                with open(path, 'a') as append:
                    append.write("\n")

                df_transform.to_csv(path, mode="a", header=False)
            else:
                df_transform.to_csv(path)

    return 0


if __name__ == "__main__":
    main()