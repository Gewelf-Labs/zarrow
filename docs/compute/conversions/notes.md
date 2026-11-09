# Conversions: function notes

Up: [function matrix](matrix.md) · [compute support matrix](../../compute-support-matrix.md)

Arrow C++'s "Conversions": `cast` (`src/compute/cast.zig:71`) and the temporal text functions
`strftime` / `strptime` (`src/compute/temporal_text.zig:66`). All are unary scalar functions on arrays,
chunked arrays and scalars, nulls propagated. The temporal rounding functions (`floor_temporal`,
`ceil_temporal`, `round_temporal`) are documented under [Temporal](../temporal/notes.md).
Evidence: `src/compute/test_cast.zig:"cast oracle matches a per-element loop on random sliced inputs with nulls for every numeric pair"`, `src/compute/test_cast.zig:"cast dictionary decode and encode, null type, identity, chunked and scalar inputs"`, `src/compute/test_temporal_text.zig:"temporal text: strftime and casts to text on nulls, sliced, chunked and scalar inputs, every string layout (pyarrow 26 results)"`.

## Summary

| Function | Arity | Input types | Output type | Options |
|---|---|---|---|---|
| [`cast`](#cast) | Unary | see below | `CastOptions.to_type` | `CastOptions` |
| [`strptime`](#strptime) | Unary | string, large_string, string_view | timestamp(unit, tz) | `StrptimeOptions` |
| [`strftime`](#strftime) | Unary | timestamp, date32, date64, time32, time64 | string | `StrftimeOptions` |

## cast

Converts to `CastOptions.to_type` (`src/compute/options.zig:127`). Safe mode applies when both
`CastOptions.safe` and `ExecConfig.safe_cast` are true. Identity returns the input; the null type
casts to every supported target. Results match pyarrow 26 unless noted.

- **Numbers and bool** (int8..int64, uint8..uint64, float32 / float64, bool) among themselves. Safe:
  out-of-range, fractional and NaN values, and integers past 2^24 / 2^53 to float, are
  `InvalidCast`; unsafe wraps integers, truncates and saturates floats.
- **To text** (string / large_string / string_view): floats print shortest round-trip digits in
  Python `repr` layout without `.0`. **From text** to numbers and bool: bad text is `InvalidCast` in
  both modes, as Arrow.
- **String and binary layouts** among themselves; UTF-8 is checked for a string target in both
  modes (Arrow skips the check when unsafe).
- **Temporal**: date32 / date64 / timestamp units and timestamp to dates, time32 / time64 units,
  timestamp to time32 / time64 (time of day), duration units (safe: a lost fraction or an overflow is
  `InvalidCast`; unsafe truncates and wraps). A timestamp in another timezone (a name or a fixed
  offset) cast to a date or time takes its local time (DST-aware, as Arrow). int32 <-> date32 /
  time32 and int64 <-> date64 / timestamp / time64 / duration keep the stored values and share
  buffers (Arrow's zero-copy casts); other integer / temporal pairs are `UnsupportedType`, as Arrow.
- **Temporal to and from text** (`src/compute/temporal_text.zig:521`, `src/compute/temporal_text.zig:735`):
  timestamp / date32 / date64 / time32 / time64 / duration to the string layouts in Arrow's forms
  (`YYYY-MM-DD HH:MM:SS.fff`, `Z` for `UTC`, local time plus `+HHMM` for another zone,
  `<value out of range: N>` past the years -32767..32767 or a day), and back from Arrow's ISO 8601
  forms (a zone offset required for a zoned target, refused for a naive one). time32 / time64 and
  duration from text too, which pyarrow 26 lacks.
- **Dictionaries**: dictionary to values (indices checked); values to `dictionary<I, T>` for every
  integer index type (too many values for `I` is `InvalidCast`); dictionary to dictionary of another
  index or value type. A dictionary of a nested type decodes, then casts.
- **Decimals** (decimal32..decimal256): decimal to decimal (scaling down truncates; safe: a lost
  digit or a value past the target precision is `InvalidCast`, unsafe wraps at the storage); decimal
  to integer (truncates; safe: a fraction or out of range is `InvalidCast`, unsafe wraps); integer to
  decimal (the precision must hold the integer's digits plus the scale; a negative scale is
  `InvalidCast`, both modes, as Arrow); decimal to float32 / float64 (Arrow's `ToReal`, bit for bit);
  float to decimal (Arrow's `FromReal`: the exact binary value rounded half to even, a negative scale
  rounded in floating point; safe: NaN, inf or past the precision is `InvalidCast`, unsafe gives 0);
  decimal to the string layouts (Arrow's `ToString`, `1.2E+3` form for a negative scale or an
  adjusted exponent below -6) and string / binary layouts to decimal (Arrow's `FromString` grammar;
  a parsed scale below -38 / -76 is `InvalidCast` in both modes). A target precision outside the
  width is `InvalidOptions`. About 14,000 random cases per seed checked against pyarrow 26.
  Deliberate differences: safe casts reject every overflowing scale-up (Arrow C++'s sign-based test
  lets some wrapped values through), and float to decimal128 at scales of 32 and more rounds the
  exact value where Arrow C++ loses its last digits.
- **float16** to and from every number type, bool and the string layouts: to text the exact value
  in float64 digits (`0.0999755859375`, as pyarrow 26); from a float rounded to nearest even
  (overflow to inf, as pyarrow 26); from an integer exact up to 2048 and past it `InvalidCast` when
  safe (pyarrow 26 rounds silently; zarrow's rule for float32 / float64 too). pyarrow 26 has no
  float16 kernel for bool and decimals; zarrow casts bool, not decimals.
- **Nested** (`castList`, `src/compute/cast.zig:849`; `castStruct`, `src/compute/cast.zig:1024`):
  list, large_list, list_view, large_list_view, fixed_size_list and map to list, large_list,
  fixed_size_list, list_view and large_list_view (the last two pyarrow 26 cannot cast to), the child
  cast recursively in the same mode. To fixed_size_list every valid row must have the size
  (`InvalidCast`; a null row of another length gets `size` null values); two fixed sizes must match
  (`UnsupportedType`, Arrow's TypeError); offsets past 32 bits are `Overflow`. A map to a map or to a
  list of a two-field struct casts its key and item by position. Struct to struct by field name in
  any order: each source field taken once, a missing nullable field all null, a missing non-nullable
  one `UnsupportedType`, a non-nullable target field holding a null `InvalidCast`; the struct
  validity is kept. The null type casts to list, large_list, fixed_size_list, struct and map. Every
  child read passes `validateLayout` first. pyarrow 26 returns empty lists for large_list_view to
  list (a pyarrow bug); zarrow keeps the values.
- **Absent**: a list of structs to a map and a nested type to a dictionary (`UnsupportedType`, as
  Arrow).

Evidence: `src/compute/cast.zig:72`, `src/compute/cast.zig:849`, `src/compute/cast.zig:1024`, `src/compute/decimal.zig:708`, `src/compute/test_cast.zig:"cast hand-written values: numbers safe and unsafe, bool, number formatting and parsing"`, `src/compute/test_cast.zig:"cast oracle matches a per-element loop on random sliced inputs with nulls for every numeric pair"`, `src/compute/test_cast.zig:"cast among string, large_string, string_view, binary, large_binary and binary_view checks UTF-8 for string targets"`, `src/compute/test_cast.zig:"cast temporal: date32, date64 and timestamp units with truncation and overflow in safe and unsafe mode"`, `src/compute/test_cast.zig:"cast dictionary decode and encode, null type, identity, chunked and scalar inputs"`, `src/compute/test_cast.zig:"cast errors: unsupported pairs, options, unparsable text, mismatched scalars and malformed input"`, `src/compute/test_cast.zig:"cast times, durations, timestamp to time of day, integer reinterpretation and dictionary index widths (Arrow C++ results)"`, `src/compute/test_cast.zig:"cast nested lists: list, large_list, list_view and fixed_size_list among themselves, the child cast recursively (pyarrow 26 results)"`, `src/compute/test_cast.zig:"cast struct to struct by field name and map to map or list of struct (pyarrow 26 results)"`, `src/compute/test_cast.zig:"float16: cast to and from numbers and text, promotion to float32 in arithmetic and comparisons, is_nan / is_inf / is_finite (pyarrow 26 results)"`, `src/compute/test_decimal.zig:"decimal casts: rescale, integers, floats and text (pyarrow 26 results)"`, `src/compute/test_temporal_text.zig:"temporal text: strftime, strptime and casts match pyarrow 26 on the generated corpus (tools/temporal_text_corpus.py)"`

## strptime

Parses text into a timestamp. `StrptimeOptions` (no defaults for `format` and `unit`, as Arrow):
`format`, `unit`, `error_is_null` (default false: a row that does not parse is `InvalidInput`;
true: null). Uses the musl parser Arrow ships (`%C %d %e %D %H %I %j %m %M %n %t %R %S %T %U %w %W
%y %Y %%`, widths, Arrow's day / month roll-over) plus glibc's `%z`; with `%z` the result is
`timestamp(unit, "UTC")`. Checked on the generated corpus; 53 `%z` rows differ from pyarrow 26 on
Windows (which uses musl's strptime, lacking `%z`), where zarrow parses it as Arrow on Linux.

Evidence: `src/compute/temporal_text.zig:74`, `src/compute/temporal_text.zig:998`, `src/compute/test_temporal_text.zig:"temporal text: strptime and casts from text on nulls, sliced, chunked and scalar inputs; error_is_null; times and durations"`, `src/compute/test_temporal_text.zig:"temporal text: strftime, strptime and casts match pyarrow 26 on the generated corpus (tools/temporal_text_corpus.py)"`

## strftime

Formats temporal values as text. `StrftimeOptions`: `format` (default `%Y-%m-%dT%H:%M:%S`) and
`locale` ("C" only; another locale is `NotImplemented`). Specifiers: pyarrow 26's `%a %A %b %h %B %c
%C %d %D %e %F %g %G %H %I %j %m %M %n %p %r %R %S %t %T %u %U %V %w %W %x %X %y %Y %z %Z %%` and the
`E` / `O` forms it accepts. `%S` / `%T` print the unit's fraction digits; `%c` / `%x` / `%g` / `%G`
/ `%E*` print `?` outside the years 0..9999, as pyarrow 26. `%z` / `%Z` on a value without a
timezone is `InvalidInput`; an unknown specifier, or a format past 4096 bytes or not UTF-8, is
`InvalidOptions`. A zoned timestamp prints its local time (DST-aware). Corpus: 16,702 cases, 0
unexplained differences; 90 rows whose format does not start with `%` differ from pyarrow 26 on
Windows (MSVC's std::format reads it as fill / width or rejects it; zarrow prints the text).

Evidence: `src/compute/temporal_text.zig:68`, `src/compute/temporal_text.zig:66`, `src/compute/test_temporal_text.zig:"temporal text: strftime and casts to text on nulls, sliced, chunked and scalar inputs, every string layout (pyarrow 26 results)"`, `src/compute/test_temporal_text.zig:"temporal text: strftime option and timezone errors (Arrow's rules)"`, `src/compute/test_temporal_text.zig:"temporal text: strftime, strptime and casts match pyarrow 26 on the generated corpus (tools/temporal_text_corpus.py)"`
