# `all_laps_*.csv`

- All columns provided by [Fastf1.laps](https://docs.fastf1.dev/core.html#laps)
- **RoundNumber** (`int`): Championship round number.
- **EventName** (`str`): Name of the Grand Prix, as provided by Fastf1 [Event Schedule](https://docs.fastf1.dev/events.html#event-schedule-data)'s `EventName` column.

# `transformed_laps_*.csv`

The following columns from `all_laps_*.csv`:

- **Time** (`pd.Timedelta`)
- **Driver** (`string`)
- **DriverNumber** (`str`)
- **LapTime** (`float`): Cast from `pd.Timedelta`
- **LapNumber** (`float`)
- **Stint** (`float`)
- **PitOutTime** (`pd.Timedelta`)
- **PitInTime** (`pd.Timedelta`)
- **IsPersonalBest** (`bool`)
- **Compound** (`string`)
- **TyreLife** (`float`)
- **FreshTyre** (`bool`)
- **Team** (`string`)
- **TrackStatus** (`int`)
- **Position** (`float`)
- **IsAccurate** (`bool`)
- **RoundNumber** (`int`)
- **EventName** (`str`)

The following columns are added. All numerical columns are accurate to three places.

- **IsSlick** (`bool`): A flag for whether the lap is completed on any of the slick compounds.
- **CompoundName** (`string`): The name of the underlying compound.
- **IsValid** (`bool`): A lap is valid if it is completed on a slick compound, the timing is accurate, and the entire lap is under green flags.
- **DeltaToRep** (`float`): Difference in seconds to the median lap time of all valid laps in the race.
- **PctFromRep** (`float`): Difference from the median lap time of all valid laps in the race as a percentage of the median lap time.
- **DeltaToFastest** (`float`): Difference in seconds to the fastest, non-deleted lap in the race.
- **PctFromFastest** (`float`): Difference from the fastest, non-deleted lap in the race as a percentage of the fastest lap time.
- **DeltaToLapRep** (`float`): Difference in seconds to the median lap time of all valid laps in the race with the same lap number.
- **PctFromLapRep** (`float`): Difference from the median lap time of all valid laps in the race with the same lap number as a percentage of the lap representative time.
