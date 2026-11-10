# Temporal: function notes

Up: [function matrix](matrix.md) · [compute support matrix](../../compute-support-matrix.md)

Arrow C++'s "Temporal component extraction", "Temporal difference", "Temporal rounding" and
"Timezone handling" functions, all unary or binary scalar functions on arrays, chunked arrays (a
chunked result) and scalars, nulls propagated. Families: extraction (`src/compute/temporal.zig:86`),
rounding and differences (`src/compute/temporal_round.zig:81`), timezones
(`src/compute/temporal_zone.zig:35`). Calendar math is proleptic Gregorian, floored before 1970
(`civil`, `src/compute/temporal.zig:366`). Timezone-naive and UTC timestamps are read as UTC; a
timestamp in another timezone (an IANA name or a fixed offset `+HH:MM` / `+HHMM`) is read as its
local time by every function here (DST-aware). Zones come from the system zoneinfo (Linux and macOS,
with `ExecConfig.io`) or the bundled IANA 2026f table (`tools/gen_tzdata.py`), both through a
bounds-checked TZif reader (`src/compute/tz.zig:286`; RFC 8536 v1-v4, footer POSIX TZ rules with the
v3 extensions); an unknown or malformed zone is `InvalidInput`. Checked against pyarrow 26 on 27,993
cases around the offset changes of 14 zones in 1995, 2011, 2021, 2037 and 2045: 0 unexplained
differences. Explained: 84 is_dst rows of Europe/Dublin and Africa/Casablanca (IANA's negative DST;
ICU flips the flag), 390 Africa/Casablanca rows from 2037 (ICU drops the predicted Ramadan
transitions) and 167 Pacific/Apia rows of September 2011 to April 2012 (ICU lacks Samoa's daylight
time around its date-line skip); 678 `%Z` rows belong to strftime. Evidence: `src/compute/test_temporal.zig:"temporal oracle matches a per-row reference loop on random sliced inputs with nulls for every input type and unit"`, `src/compute/test_temporal.zig:"temporal chunked inputs with misaligned chunks give a chunked result equal to the array result"`, `src/compute/test_temporal.zig:"temporal scalars, empty and all-null inputs"`, `src/compute/test_temporal_zone.zig:"timezones: zoned temporal functions and assume_timezone match pyarrow 26 around offset changes (tools/tz_corpus.py)"`, `src/compute/test_temporal_zone.zig:"timezones: ExecConfig.io resolves names (system zoneinfo on Linux and macOS, else the bundled table)"`.

## Summary

| Function | Arity | Input types | Output type | Options |
|---|---|---|---|---|
| [`year`](#year) | Unary | date32, date64, timestamp | int64 | none |
| [`month`](#month) | Unary | date32, date64, timestamp | int64 | none |
| [`day`](#day) | Unary | date32, date64, timestamp | int64 | none |
| [`day_of_week`](#day_of_week) | Unary | date32, date64, timestamp | int64 | `DayOfWeekOptions` |
| [`day_of_year`](#day_of_year) | Unary | date32, date64, timestamp | int64 | none |
| [`hour`](#hour) | Unary | date64, timestamp, time32, time64 | int64 | none |
| [`minute`](#minute) | Unary | date64, timestamp, time32, time64 | int64 | none |
| [`second`](#second) | Unary | date64, timestamp, time32, time64 | int64 | none |
| [`millisecond`](#millisecond) | Unary | date64, timestamp, time32, time64 | int64 | none |
| [`microsecond`](#microsecond) | Unary | date64, timestamp, time32, time64 | int64 | none |
| [`nanosecond`](#nanosecond) | Unary | date64, timestamp, time32, time64 | int64 | none |
| [`subsecond`](#subsecond) | Unary | timestamp, time32, time64 | float64 | none |
| [`quarter`](#quarter) | Unary | date32, date64, timestamp | int64 | none |
| [`week`](#week) | Unary | date32, date64, timestamp | int64 | `WeekOptions` |
| [`us_week`](#us_week) | Unary | date32, date64, timestamp | int64 | none |
| [`us_year`](#us_year) | Unary | date32, date64, timestamp | int64 | none |
| [`iso_week`](#iso_week) | Unary | date32, date64, timestamp | int64 | none |
| [`iso_year`](#iso_year) | Unary | date32, date64, timestamp | int64 | none |
| [`iso_calendar`](#iso_calendar) | Unary | date32, date64, timestamp | struct<iso_year, iso_week, iso_day_of_week: int64> | none |
| [`year_month_day`](#year_month_day) | Unary | date32, date64, timestamp | struct<year, month, day: int64> | none |
| [`is_leap_year`](#is_leap_year) | Unary | date32, date64, timestamp | bool | none |
| [`is_dst`](#is_dst) | Unary | zoned timestamp | bool | none |
| [`years_between`](#years_between) | Binary | date32, date64, timestamp | int64 | none |
| [`quarters_between`](#quarters_between) | Binary | date32, date64, timestamp | int64 | none |
| [`month_interval_between`](#month_interval_between) | Binary | date32, date64, timestamp | month_interval | none |
| [`weeks_between`](#weeks_between) | Binary | date32, date64, timestamp | int64 | `DayOfWeekOptions` |
| [`days_between`](#days_between) | Binary | date32, date64, timestamp | int64 | none |
| [`hours_between`](#hours_between) | Binary | date32, date64, timestamp, time32, time64 | int64 | none |
| [`minutes_between`](#minutes_between) | Binary | date32, date64, timestamp, time32, time64 | int64 | none |
| [`seconds_between`](#seconds_between) | Binary | date32, date64, timestamp, time32, time64 | int64 | none |
| [`milliseconds_between`](#milliseconds_between) | Binary | date32, date64, timestamp, time32, time64 | int64 | none |
| [`microseconds_between`](#microseconds_between) | Binary | date32, date64, timestamp, time32, time64 | int64 | none |
| [`nanoseconds_between`](#nanoseconds_between) | Binary | date32, date64, timestamp, time32, time64 | int64 | none |
| [`day_time_interval_between`](#day_time_interval_between) | Binary | date32, date64, timestamp, time32, time64 | day_time_interval | none |
| [`month_day_nano_interval_between`](#month_day_nano_interval_between) | Binary | date32, date64, timestamp, time32, time64 | month_day_nano_interval | none |
| [`ceil_temporal`](#ceil_temporal) | Unary | date32, date64, timestamp, time32, time64, duration | input type | `RoundTemporalOptions` |
| [`floor_temporal`](#floor_temporal) | Unary | date32, date64, timestamp, time32, time64, duration | input type | `RoundTemporalOptions` |
| [`round_temporal`](#round_temporal) | Unary | date32, date64, timestamp, time32, time64, duration | input type | `RoundTemporalOptions` |
| [`assume_timezone`](#assume_timezone) | Unary | naive timestamp | timestamp(unit, timezone) | `AssumeTimezoneOptions` |
| [`local_timestamp`](#local_timestamp) | Unary | timestamp | naive timestamp | none |

## year

The Gregorian year. Accepted on date32, date64 and timestamp (every unit). Time32 / time64 are not accepted.

Evidence: `src/compute/temporal.zig:53`, `src/compute/temporal.zig:126`, `src/compute/test_temporal.zig:"temporal hand-written dates: components of known days, leap years, ISO weeks and week options"`, `src/compute/test_temporal.zig:"temporal chunked inputs with misaligned chunks give a chunked result equal to the array result"`, `src/compute/test_temporal.zig:"temporal oracle matches a per-row reference loop on random sliced inputs with nulls for every input type and unit"`, `src/compute/test_temporal.zig:"temporal 32-bit and vector calendar paths agree with the 64-bit algorithm across their whole range"`

## month

The month, 1..12. Accepted on date32, date64 and timestamp (every unit). Time32 / time64 are not accepted.

Evidence: `src/compute/temporal.zig:54`, `src/compute/temporal.zig:126`, `src/compute/test_temporal.zig:"temporal hand-written dates: components of known days, leap years, ISO weeks and week options"`, `src/compute/test_temporal.zig:"temporal scalars, empty and all-null inputs"`, `src/compute/test_temporal.zig:"temporal 32-bit and vector calendar paths agree with the 64-bit algorithm across their whole range"`

## day

The day of the month, 1..31. Accepted on date32, date64 and timestamp (every unit). Time32 / time64 are not accepted.

Evidence: `src/compute/temporal.zig:55`, `src/compute/temporal.zig:126`, `src/compute/test_temporal.zig:"temporal hand-written dates: components of known days, leap years, ISO weeks and week options"`, `src/compute/test_temporal.zig:"temporal 32-bit and vector calendar paths agree with the 64-bit algorithm across their whole range"`, `src/compute/test_temporal.zig:"temporal week, us_week, subsecond and year_month_day: every WeekOptions combination (Arrow C++ results)"`

## day_of_week

The weekday number. `DayOfWeekOptions`: `count_from_zero` (default true: 0..6, else 1..7) and
`week_start` (default 1 = Monday ... 7 = Sunday); a `week_start` outside 1..7 is `InvalidOptions`.
Accepted on date32, date64 and timestamp (every unit). Time32 / time64 are not accepted.

Evidence: `src/compute/temporal.zig:97`, `src/compute/temporal.zig:126`, `src/compute/test_temporal.zig:"temporal hand-written dates: components of known days, leap years, ISO weeks and week options"`, `src/compute/test_temporal.zig:"temporal errors: unknown timezone, time parts of date32, options, arity and malformed input"`

## day_of_year

The day of the year, 1..366. Accepted on date32, date64 and timestamp (every unit). Time32 / time64 are not accepted.

Evidence: `src/compute/temporal.zig:295`, `src/compute/temporal.zig:126`, `src/compute/test_temporal.zig:"temporal hand-written dates: components of known days, leap years, ISO weeks and week options"`, `src/compute/test_temporal.zig:"temporal oracle matches a per-row reference loop on random sliced inputs with nulls for every input type and unit"`

## hour

The hour component of the time of day (an integer field, not a total). Accepted on date64, timestamp, time32 and time64 (every unit); date32 is `NoMatchingKernel` (no time parts). A time past a day or below zero wraps, as Arrow.

Evidence: `src/compute/temporal.zig:63`, `src/compute/temporal.zig:126`, `src/compute/test_temporal.zig:"temporal hand-written dates: components of known days, leap years, ISO weeks and week options"`, `src/compute/test_temporal.zig:"temporal oracle matches a per-row reference loop on random sliced inputs with nulls for every input type and unit"`, `src/compute/test_temporal.zig:"temporal errors: unknown timezone, time parts of date32, options, arity and malformed input"`, `src/compute/test_temporal_round.zig:"temporal time32 / time64 inputs, us_year and interval differences (Arrow C++ results)"`

## minute

The minute component of the time of day (an integer field, not a total). Accepted on date64, timestamp, time32 and time64 (every unit); date32 is `NoMatchingKernel` (no time parts). A time past a day or below zero wraps, as Arrow.

Evidence: `src/compute/temporal.zig:63`, `src/compute/temporal.zig:126`, `src/compute/test_temporal.zig:"temporal hand-written dates: components of known days, leap years, ISO weeks and week options"`, `src/compute/test_temporal.zig:"temporal oracle matches a per-row reference loop on random sliced inputs with nulls for every input type and unit"`, `src/compute/test_temporal.zig:"temporal errors: unknown timezone, time parts of date32, options, arity and malformed input"`, `src/compute/test_temporal_round.zig:"temporal time32 / time64 inputs, us_year and interval differences (Arrow C++ results)"`

## second

The second component of the time of day (an integer field, not a total). Accepted on date64, timestamp, time32 and time64 (every unit); date32 is `NoMatchingKernel` (no time parts). A time past a day or below zero wraps, as Arrow.

Evidence: `src/compute/temporal.zig:63`, `src/compute/temporal.zig:126`, `src/compute/test_temporal.zig:"temporal hand-written dates: components of known days, leap years, ISO weeks and week options"`, `src/compute/test_temporal.zig:"temporal oracle matches a per-row reference loop on random sliced inputs with nulls for every input type and unit"`, `src/compute/test_temporal.zig:"temporal errors: unknown timezone, time parts of date32, options, arity and malformed input"`, `src/compute/test_temporal_round.zig:"temporal time32 / time64 inputs, us_year and interval differences (Arrow C++ results)"`

## millisecond

The millisecond component of the time of day (an integer field, not a total). Accepted on date64, timestamp, time32 and time64 (every unit); date32 is `NoMatchingKernel` (no time parts). A time past a day or below zero wraps, as Arrow.

Evidence: `src/compute/temporal.zig:63`, `src/compute/temporal.zig:126`, `src/compute/test_temporal.zig:"temporal hand-written dates: components of known days, leap years, ISO weeks and week options"`, `src/compute/test_temporal.zig:"temporal oracle matches a per-row reference loop on random sliced inputs with nulls for every input type and unit"`, `src/compute/test_temporal.zig:"temporal errors: unknown timezone, time parts of date32, options, arity and malformed input"`, `src/compute/test_temporal_round.zig:"temporal time32 / time64 inputs, us_year and interval differences (Arrow C++ results)"`

## microsecond

The microsecond component of the time of day (an integer field, not a total). Accepted on date64, timestamp, time32 and time64 (every unit); date32 is `NoMatchingKernel` (no time parts). A time past a day or below zero wraps, as Arrow.

Evidence: `src/compute/temporal.zig:63`, `src/compute/temporal.zig:126`, `src/compute/test_temporal.zig:"temporal hand-written dates: components of known days, leap years, ISO weeks and week options"`, `src/compute/test_temporal.zig:"temporal oracle matches a per-row reference loop on random sliced inputs with nulls for every input type and unit"`, `src/compute/test_temporal.zig:"temporal errors: unknown timezone, time parts of date32, options, arity and malformed input"`, `src/compute/test_temporal_round.zig:"temporal time32 / time64 inputs, us_year and interval differences (Arrow C++ results)"`

## nanosecond

The nanosecond component of the time of day (an integer field, not a total). Accepted on date64, timestamp, time32 and time64 (every unit); date32 is `NoMatchingKernel` (no time parts). A time past a day or below zero wraps, as Arrow.

Evidence: `src/compute/temporal.zig:63`, `src/compute/temporal.zig:126`, `src/compute/test_temporal.zig:"temporal hand-written dates: components of known days, leap years, ISO weeks and week options"`, `src/compute/test_temporal.zig:"temporal oracle matches a per-row reference loop on random sliced inputs with nulls for every input type and unit"`, `src/compute/test_temporal.zig:"temporal errors: unknown timezone, time parts of date32, options, arity and malformed input"`, `src/compute/test_temporal_round.zig:"temporal time32 / time64 inputs, us_year and interval differences (Arrow C++ results)"`

## subsecond

The fraction of the second as float64. Accepted on timestamp, time32 and time64; date32 and date64
are not accepted (Arrow registers it for timestamps and times only).

Evidence: `src/compute/temporal.zig:63`, `src/compute/temporal.zig:126`, `src/compute/test_temporal.zig:"temporal week, us_week, subsecond and year_month_day: every WeekOptions combination (Arrow C++ results)"`, `src/compute/test_temporal_round.zig:"temporal time32 / time64 inputs, us_year and interval differences (Arrow C++ results)"`, `src/compute/test_temporal.zig:"temporal oracle matches a per-row reference loop on random sliced inputs with nulls for every input type and unit"`

## quarter

The quarter, 1..4. Accepted on date32, date64 and timestamp (every unit). Time32 / time64 are not accepted.

Evidence: `src/compute/temporal.zig:254`, `src/compute/temporal.zig:126`, `src/compute/test_temporal.zig:"temporal hand-written dates: components of known days, leap years, ISO weeks and week options"`, `src/compute/test_temporal.zig:"temporal oracle matches a per-row reference loop on random sliced inputs with nulls for every input type and unit"`

## week

The week number (`week`, `src/compute/temporal.zig:439`). `WeekOptions` (`src/compute/options.zig:286`):
`week_starts_monday` (default true), `count_from_zero` (default false: days before week 1 belong to
the previous year's last week; true: week 0), `first_week_is_fully_in_year` (default false: week 1
holds the 4th day of the year; true: it starts on the first week-start day). Defaults equal
`iso_week`. Every combination checked against Arrow C++. Accepted on date32, date64 and timestamp (every unit). Time32 / time64 are not accepted.

Evidence: `src/compute/temporal.zig:98`, `src/compute/temporal.zig:126`, `src/compute/test_temporal.zig:"temporal hand-written dates: components of known days, leap years, ISO weeks and week options"`, `src/compute/test_temporal.zig:"temporal week, us_week, subsecond and year_month_day: every WeekOptions combination (Arrow C++ results)"`

## us_week

`week` with Arrow's `WeekOptions::USWeek()`: weeks start on Sunday, not counted from zero, week 1
holds January 4th. Accepted on date32, date64 and timestamp (every unit). Time32 / time64 are not accepted.

Evidence: `src/compute/temporal.zig:145`, `src/compute/temporal.zig:126`, `src/compute/test_temporal.zig:"temporal week, us_week, subsecond and year_month_day: every WeekOptions combination (Arrow C++ results)"`, `src/compute/test_temporal.zig:"temporal hand-written dates: components of known days, leap years, ISO weeks and week options"`

## us_year

The year that the `us_week` numbering assigns to the day. Accepted on date32, date64 and timestamp (every unit). Time32 / time64 are not accepted.

Evidence: `src/compute/temporal.zig:301`, `src/compute/temporal.zig:126`, `src/compute/test_temporal_round.zig:"temporal time32 / time64 inputs, us_year and interval differences (Arrow C++ results)"`

## iso_week

The ISO 8601 week, 1..53 (`isoWeek`, `src/compute/temporal.zig:423`). Accepted on date32, date64 and timestamp (every unit). Time32 / time64 are not accepted.

Evidence: `src/compute/temporal.zig:47`, `src/compute/temporal.zig:126`, `src/compute/test_temporal.zig:"temporal hand-written dates: components of known days, leap years, ISO weeks and week options"`, `src/compute/test_temporal.zig:"temporal oracle matches a per-row reference loop on random sliced inputs with nulls for every input type and unit"`

## iso_year

The ISO 8601 week-numbering year. Accepted on date32, date64 and timestamp (every unit). Time32 / time64 are not accepted.

Evidence: `src/compute/temporal.zig:46`, `src/compute/temporal.zig:126`, `src/compute/test_temporal.zig:"temporal hand-written dates: components of known days, leap years, ISO weeks and week options"`, `src/compute/test_temporal.zig:"temporal oracle matches a per-row reference loop on random sliced inputs with nulls for every input type and unit"`

## iso_calendar

A struct of the ISO year, ISO week and ISO weekday (1 = Monday). Accepted on date32, date64 and timestamp (every unit). Time32 / time64 are not accepted.

Evidence: `src/compute/temporal.zig:71`, `src/compute/temporal.zig:126`, `src/compute/test_temporal.zig:"temporal hand-written dates: components of known days, leap years, ISO weeks and week options"`, `src/compute/test_temporal.zig:"temporal scalars, empty and all-null inputs"`

## year_month_day

A struct of year, month and day. Accepted on date32, date64 and timestamp (every unit). Time32 / time64 are not accepted.

Evidence: `src/compute/temporal.zig:72`, `src/compute/temporal.zig:126`, `src/compute/test_temporal.zig:"temporal week, us_week, subsecond and year_month_day: every WeekOptions combination (Arrow C++ results)"`, `src/compute/test_temporal.zig:"temporal scalars, empty and all-null inputs"`

## is_leap_year

Whether the year is a Gregorian leap year. Accepted on date32, date64 and timestamp (every unit). Time32 / time64 are not accepted.

Evidence: `src/compute/temporal.zig:70`, `src/compute/temporal.zig:126`, `src/compute/test_temporal.zig:"temporal hand-written dates: components of known days, leap years, ISO weeks and week options"`, `src/compute/test_temporal.zig:"temporal oracle matches a per-row reference loop on random sliced inputs with nulls for every input type and unit"`

## is_dst

Whether daylight saving time is in effect at the instant, from the timestamp's zone. A naive
timestamp is `InvalidInput`. Deviation (explained): 84 rows of Europe/Dublin and Africa/Casablanca
differ from pyarrow 26 on Windows, where ICU uses the rearguard form with the flag flipped.

Evidence: `src/compute/temporal_zone.zig:44`, `src/compute/temporal_zone.zig:35`, `src/compute/test_temporal_zone.zig:"timezones: assume_timezone, local_timestamp and is_dst on nulls, sliced, chunked and scalar inputs (pyarrow 26 results)"`, `src/compute/test_temporal_zone.zig:"timezones: zoned temporal functions and assume_timezone match pyarrow 26 around offset changes (tools/tz_corpus.py)"`

## years_between

Calendar-year boundaries crossed from the first operand to the second: `year2 - year1` of the civil dates, the day of year ignored (pyarrow 26: 2020-02-29 to 2021-02-28 is 1, 2019-12-31 to 2020-01-01 is 1). Two operands of the same type; differences past 64 bits wrap like Arrow.

Evidence: `src/compute/temporal_round.zig:62`, `src/compute/temporal_round.zig:416`, `src/compute/test_temporal_round.zig:"temporal rounding and differences hand-written values (Arrow C++ results)"`, `src/compute/test_temporal_round.zig:"temporal rounding and differences on chunked inputs with misaligned chunks equal the array result"`, `src/compute/test_temporal_round.zig:"temporal rounding and differences errors: options, timezone, mismatched types, arity and malformed input"`, `src/compute/test_temporal_round.zig:"years, quarters, weeks, days and month_interval between: end of month, leap day, negative spans, week start, sliced, chunked (pyarrow 26 results)"`

## quarters_between

Quarter boundaries crossed from the first operand to the second: `(4 * year2 + (month2 - 1) / 3) - (4 * year1 + (month1 - 1) / 3)`, the day ignored (pyarrow 26: 2020-02-29 to 2021-02-28 is 4, 2021-03-31 to 2020-12-31 is -1). Two operands of the same type; differences past 64 bits wrap like Arrow.

Evidence: `src/compute/temporal_round.zig:62`, `src/compute/temporal_round.zig:416`, `src/compute/test_temporal_round.zig:"temporal rounding and differences hand-written values (Arrow C++ results)"`, `src/compute/test_temporal_round.zig:"temporal rounding and differences on chunked inputs with misaligned chunks equal the array result"`, `src/compute/test_temporal_round.zig:"years, quarters, weeks, days and month_interval between: end of month, leap day, negative spans, week start, sliced, chunked (pyarrow 26 results)"`

## month_interval_between

Month boundaries crossed, as a month_interval: `(12 * year2 + month2) - (12 * year1 + month1)`, the day ignored (pyarrow 26: 2020-01-31 to 2020-02-29 is 1M, 2021-03-31 to 2020-12-31 is -3M). Two operands of the same type; differences past 64 bits wrap like Arrow.

Evidence: `src/compute/temporal_round.zig:62`, `src/compute/temporal_round.zig:416`, `src/compute/test_temporal_round.zig:"temporal rounding and differences hand-written values (Arrow C++ results)"`, `src/compute/test_temporal_round.zig:"temporal time32 / time64 inputs, us_year and interval differences (Arrow C++ results)"`, `src/compute/test_temporal_round.zig:"years, quarters, weeks, days and month_interval between: end of month, leap day, negative spans, week start, sliced, chunked (pyarrow 26 results)"`

## weeks_between

Week boundaries crossed: `floor((day2 + 3 - (week_start - 1)) / 7) - floor((day1 + 3 - (week_start - 1)) / 7)` on days since 1970-01-01 (a Thursday); the week starts on `DayOfWeekOptions.week_start` (default Monday; pyarrow 26: Sunday 2024-01-07 to Monday 2024-01-08 is 1 with Monday, 0 with Sunday). Two operands of the same type; differences past 64 bits wrap like Arrow.

Evidence: `src/compute/temporal_round.zig:62`, `src/compute/temporal_round.zig:416`, `src/compute/test_temporal_round.zig:"temporal rounding and differences hand-written values (Arrow C++ results)"`, `src/compute/test_temporal_round.zig:"temporal rounding and differences on chunked inputs with misaligned chunks equal the array result"`, `src/compute/test_temporal_round.zig:"years, quarters, weeks, days and month_interval between: end of month, leap day, negative spans, week start, sliced, chunked (pyarrow 26 results)"`

## days_between

Day boundaries crossed (local dates): `floor(t2 / day) - floor(t1 / day)` (pyarrow 26: 2020-02-29 to 2021-02-28 is 365). Two operands of the same type; differences past 64 bits wrap like Arrow.

Evidence: `src/compute/temporal_round.zig:62`, `src/compute/temporal_round.zig:416`, `src/compute/test_temporal_round.zig:"temporal rounding and differences hand-written values (Arrow C++ results)"`, `src/compute/test_temporal_round.zig:"temporal rounding and differences on scalars, empty and all-null inputs"`, `src/compute/test_temporal_round.zig:"temporal rounding oracle matches a per-row reference on random sliced inputs with nulls for every unit"`, `src/compute/test_temporal_round.zig:"years, quarters, weeks, days and month_interval between: end of month, leap day, negative spans, week start, sliced, chunked (pyarrow 26 results)"`

## hours_between

The number of hours boundaries crossed; also on time32 / time64. Two operands of the same type; differences past 64 bits wrap like Arrow.

Evidence: `src/compute/temporal_round.zig:509`, `src/compute/temporal_round.zig:416`, `src/compute/test_temporal_round.zig:"temporal rounding and differences hand-written values (Arrow C++ results)"`, `src/compute/test_temporal_round.zig:"temporal time32 / time64 inputs, us_year and interval differences (Arrow C++ results)"`, `src/compute/test_temporal_round.zig:"temporal rounding and differences on chunked inputs with misaligned chunks equal the array result"`

## minutes_between

The number of minutes boundaries crossed; also on time32 / time64. Two operands of the same type; differences past 64 bits wrap like Arrow.

Evidence: `src/compute/temporal_round.zig:510`, `src/compute/temporal_round.zig:416`, `src/compute/test_temporal_round.zig:"temporal rounding and differences hand-written values (Arrow C++ results)"`, `src/compute/test_temporal_round.zig:"temporal time32 / time64 inputs, us_year and interval differences (Arrow C++ results)"`, `src/compute/test_temporal_round.zig:"temporal rounding and differences on chunked inputs with misaligned chunks equal the array result"`

## seconds_between

The number of seconds boundaries crossed; also on time32 / time64. Two operands of the same type; differences past 64 bits wrap like Arrow.

Evidence: `src/compute/temporal_round.zig:511`, `src/compute/temporal_round.zig:416`, `src/compute/test_temporal_round.zig:"temporal rounding and differences hand-written values (Arrow C++ results)"`, `src/compute/test_temporal_round.zig:"temporal time32 / time64 inputs, us_year and interval differences (Arrow C++ results)"`, `src/compute/test_temporal_round.zig:"temporal rounding and differences on chunked inputs with misaligned chunks equal the array result"`

## milliseconds_between

The number of milliseconds boundaries crossed; also on time32 / time64. Two operands of the same type; differences past 64 bits wrap like Arrow.

Evidence: `src/compute/temporal_round.zig:512`, `src/compute/temporal_round.zig:416`, `src/compute/test_temporal_round.zig:"temporal rounding and differences hand-written values (Arrow C++ results)"`, `src/compute/test_temporal_round.zig:"temporal time32 / time64 inputs, us_year and interval differences (Arrow C++ results)"`, `src/compute/test_temporal_round.zig:"temporal rounding and differences on chunked inputs with misaligned chunks equal the array result"`

## microseconds_between

The number of microseconds boundaries crossed; also on time32 / time64. Two operands of the same type; differences past 64 bits wrap like Arrow.

Evidence: `src/compute/temporal_round.zig:513`, `src/compute/temporal_round.zig:416`, `src/compute/test_temporal_round.zig:"temporal rounding and differences hand-written values (Arrow C++ results)"`, `src/compute/test_temporal_round.zig:"temporal time32 / time64 inputs, us_year and interval differences (Arrow C++ results)"`, `src/compute/test_temporal_round.zig:"temporal rounding and differences on chunked inputs with misaligned chunks equal the array result"`

## nanoseconds_between

The number of nanoseconds boundaries crossed; also on time32 / time64. Two operands of the same type; differences past 64 bits wrap like Arrow.

Evidence: `src/compute/temporal_round.zig:5`, `src/compute/temporal_round.zig:416`, `src/compute/test_temporal_round.zig:"temporal rounding and differences hand-written values (Arrow C++ results)"`, `src/compute/test_temporal_round.zig:"temporal time32 / time64 inputs, us_year and interval differences (Arrow C++ results)"`, `src/compute/test_temporal_round.zig:"temporal rounding and differences on chunked inputs with misaligned chunks equal the array result"`

## day_time_interval_between

Days, then milliseconds of the day, as a day_time_interval (Arrow C++'s definition). pyarrow 26
cannot materialize a day_time_interval array, so these expected values are not pyarrow-checked. Two operands of the same type; differences past 64 bits wrap like Arrow.

Evidence: `src/compute/temporal_round.zig:73`, `src/compute/temporal_round.zig:416`, `src/compute/test_temporal_round.zig:"temporal time32 / time64 inputs, us_year and interval differences (Arrow C++ results)"`, `src/compute/test_temporal_round.zig:"temporal rounding and differences hand-written values (Arrow C++ results)"`

## month_day_nano_interval_between

Months, then days of the month and nanoseconds of the day, as a month_day_nano_interval; also on times. Two operands of the same type; differences past 64 bits wrap like Arrow.

Evidence: `src/compute/temporal_round.zig:74`, `src/compute/temporal_round.zig:416`, `src/compute/test_temporal_round.zig:"temporal time32 / time64 inputs, us_year and interval differences (Arrow C++ results)"`, `src/compute/test_temporal_round.zig:"temporal rounding and differences hand-written values (Arrow C++ results)"`

## ceil_temporal

Rounds up to the next multiple of the unit; a value on a boundary stays unless `ceil_is_strictly_greater`. See `round_temporal` for the shared options and rules.

Evidence: `src/compute/temporal_round.zig:290`, `src/compute/temporal_round.zig:309`, `src/compute/temporal_round.zig:300`, `src/compute/test_temporal_round.zig:"temporal rounding and differences hand-written values (Arrow C++ results)"`, `src/compute/test_temporal_round.zig:"temporal rounding oracle matches a per-row reference on random sliced inputs with nulls for every unit"`, `src/compute/test_temporal_round.zig:"floor_temporal, ceil_temporal and round_temporal on duration: multiples from zero, finer units, strict ceil, weeks, sliced, chunked, scalars (pyarrow 26 results)"`

## floor_temporal

Rounds down to a multiple of the unit. See `round_temporal` for the shared options and rules.

Evidence: `src/compute/temporal_round.zig:288`, `src/compute/temporal_round.zig:309`, `src/compute/temporal_round.zig:300`, `src/compute/test_temporal_round.zig:"temporal rounding and differences hand-written values (Arrow C++ results)"`, `src/compute/test_temporal_round.zig:"temporal rounding oracle matches a per-row reference on random sliced inputs with nulls for every unit"`, `src/compute/test_temporal_round.zig:"floor_temporal, ceil_temporal and round_temporal on duration: multiples from zero, finer units, strict ceil, weeks, sliced, chunked, scalars (pyarrow 26 results)"`

## round_temporal

Rounds to the nearest multiple of the unit.

Keeps the input type: date32, date64, timestamp, time32 / time64 (rounded like a timestamp of their
unit) and duration. `RoundTemporalOptions` (`src/compute/options.zig:273`): `multiple` (default 1,
> 0), `unit` (nanosecond .. year, default day), `week_starts_monday` (default true),
`ceil_is_strictly_greater` (default false), `calendar_based_origin` (default false: multiples counted
from 1970-01-01; true: from the start of the next larger unit). Arrow quirks are kept and were
checked against pyarrow 26 on random values (ceil of a month / quarter / year on a boundary moves up;
week origins with calendar_based_origin; a unit finer than the input truncates back). A zoned
timestamp rounds its local time and maps back to UTC; an ambiguous or nonexistent local result (also
the floor or ceil it is chosen from) is `InvalidInput` (Arrow: "Local time is ambiguous").
On duration: multiples of nanosecond .. week counted from zero (a week is seven days), a unit finer
than the input keeps the value, ties round up; month, quarter, year and `calendar_based_origin` are
`InvalidOptions` (Arrow rejects them). pyarrow 26: the default one day ceils 1 ms to 1 day and
floors it to 0.

Evidence: `src/compute/temporal_round.zig:130`, `src/compute/temporal_round.zig:309`, `src/compute/temporal_round.zig:300`, `src/compute/test_temporal_round.zig:"temporal rounding and differences hand-written values (Arrow C++ results)"`, `src/compute/test_temporal_round.zig:"temporal rounding oracle matches a per-row reference on random sliced inputs with nulls for every unit"`, `src/compute/test_temporal_round.zig:"temporal rounding and differences on chunked inputs with misaligned chunks equal the array result"`, `src/compute/test_temporal_round.zig:"temporal rounding and differences on scalars, empty and all-null inputs"`, `src/compute/test_temporal_round.zig:"temporal rounding and differences errors: options, timezone, mismatched types, arity and malformed input"`, `src/compute/test_temporal_round.zig:"floor_temporal, ceil_temporal and round_temporal on duration: multiples from zero, finer units, strict ceil, weeks, sliced, chunked, scalars (pyarrow 26 results)"`

## assume_timezone

Reads a naive timestamp as local time in `AssumeTimezoneOptions.timezone` (an IANA name or a fixed
offset `+HH:MM`) and returns the UTC instant typed with that zone. `ambiguous` and `nonexistent`
(each `raise` (default), `earliest` or `latest`) choose the instant for a local time that occurs
twice or never; `raise` makes it `InvalidInput`. A zoned input is `InvalidInput`. Every choice is
checked on the tz corpus.

Evidence: `src/compute/temporal_zone.zig:36`, `src/compute/temporal_zone.zig:122`, `src/compute/test_temporal_zone.zig:"timezones: assume_timezone, local_timestamp and is_dst on nulls, sliced, chunked and scalar inputs (pyarrow 26 results)"`, `src/compute/test_temporal_zone.zig:"timezones: zoned temporal functions and assume_timezone match pyarrow 26 around offset changes (tools/tz_corpus.py)"`

## local_timestamp

The local wall-clock time of a timestamp as a naive timestamp of the same unit; a naive input is returned as is and a UTC input
keeps its values.

Evidence: `src/compute/temporal_zone.zig:40`, `src/compute/temporal_zone.zig:99`, `src/compute/test_temporal_zone.zig:"timezones: assume_timezone, local_timestamp and is_dst on nulls, sliced, chunked and scalar inputs (pyarrow 26 results)"`, `src/compute/test_temporal_zone.zig:"timezones: zoned temporal functions and assume_timezone match pyarrow 26 around offset changes (tools/tz_corpus.py)"`
