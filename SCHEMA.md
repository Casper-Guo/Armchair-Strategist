# `all_laps_*.csv`

**Caution**: Retroactive data accuracy fixes may not always be applied to `all_laps_*.csv`! Always depend on `transformed_laps_*.csv` whenever possible.

- All columns provided by [Fastf1.laps](https://docs.fastf1.dev/core.html#laps)
- **RoundNumber** (`int`): Championship round number.
- **EventName** (`string`): Name of the Grand Prix, as provided by Fastf1 [Event Schedule](https://docs.fastf1.dev/events.html#event-schedule-data)'s `EventName` column.

# `transformed_laps_*.csv`

The following columns from `all_laps_*.csv`, note when reading from CSVs `pd.Timedelta` is interpreted as `string` and thus require explicit casting:

- **Time** (`pd.Timedelta`)
- **Driver** (`string`)
- **DriverNumber** (`int`)
- **LapTime** (`float`): Cast from `pd.Timedelta`
- **LapNumber** (`float`)
- **Stint** (`float`)
- **PitOutTime** (`pd.Timedelta`)
- **PitInTime** (`pd.Timedelta`)
- **IsPersonalBest** (`bool`)
- **Compound** (`string`)
- **TyreLife** (`float`)
- **FreshTyre** (`string`)
- **Team** (`string`)
- **TrackStatus** (`int`)
- **Position** (`float`)
- **IsAccurate** (`bool`)
- **RoundNumber** (`int`)
- **EventName** (`string`)

The following columns are added. All numerical columns are accurate to three places.

- **IsSlick** (`bool`): A flag for whether the lap is completed on any of the slick compounds.
- **CompoundName** (`string`): The name of the underlying compound.
- **IsValid** (`bool`): A lap is valid if it is completed on a slick compound, the timing is accurate, and the entire lap is under green flags.

For the next two features, a representative time is defined as the median of all lap times in the same race, given that:

- The lap is accurate (as determined by `IsAccurate`)
- The lap is completed under green flags

The representative time is calculated separately for slick and non-slick compounds.

- **DeltaToRep** (`float`): Difference in seconds to the representative time.
- **PctFromRep** (`float`): Percentage difference from the representative time
- **DeltaToFastest** (`float`): Difference in seconds to the fastest, non-deleted lap in the race.
- **PctFromFastest** (`float`): Percentage difference from the fastest, non-deleted lap in the same race.
- **DeltaToLapRep** (`float`): Difference in seconds to the median of all lap times from the same lap and is neither an in lap or an out lap.
- **PctFromLapRep** (`float`): Percentage difference from the median of all lap times from the same lap and is neither an in lap or an out lap.
- **FuelAdjLapTime** (`float`): Lap time adjusted for fuel consumption. Assumes an initial fuel load of 110kg and a performance gain of 0.03s per lap per kilogram of fuel consumed
