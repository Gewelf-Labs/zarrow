# Compute API

This document describes the compute framework API in `zarrow.compute`.

It is for anyone adding kernels to this repository (core and compute live together, see [repo-layout.md](repo-layout.md)) and for applications that register and invoke them.

Chinese version: [compute-api-zh.md](compute-api-zh.md)

---

## 1. Scope and Boundary

`zarrow` provides the **compute framework layer**:

- function registration and dispatch;
- signature matching (arity/type/options);
- result type inference hooks;
- execution context and aggregate lifecycle abstractions;
- common helper utilities for array/scalar/chunked execution;
- chunk-aware permutation helpers.

Concrete kernels live in this repository under `src/compute/`, one file per family. `FunctionRegistry.init`
returns an empty registry; each family exposes `register(registry)`, and `registerAll(registry)`
(`compute.registerAll`, `zarrow.computeRegisterAll`) registers every family, the equivalent of
`arrow::compute::Initialize()`:

```zig
var registry = zarrow.compute.FunctionRegistry.init(allocator);
defer registry.deinit();
try zarrow.compute.registerAll(&registry); // or compute.kernels.arithmetic.register(&registry), ...
var ctx = zarrow.compute.ExecContext.init(allocator, &registry);
var sum = try ctx.invoke("add", .scalar, &.{ lhs, rhs }, .none); // lhs, rhs: Datum
defer sum.release();
```

| Family | File | Functions |
|---|---|---|
| Arithmetic | `arithmetic.zig` | `add`, `subtract`, `multiply`, `divide`, `negate`, `abs` and their `_checked` variants, `sign` (int8..int64, uint8..uint64, float32/64) |
| Decimals | `decimal.zig` | the decimal kernels of `add`, `subtract`, `multiply`, `divide` (+ `_checked`; decimal32..decimal256, also with an integer), `negate`, `abs` (+ `_checked`) and `sign` (decimal128 / decimal256); the decimal casts behind `cast`; the decimal helpers of the comparison and aggregate families |
| Comparisons | `comparison.zig` | `equal`, `not_equal`, `less`, `less_equal`, `greater`, `greater_equal` (numeric, bool, string and binary layouts, fixed_size_binary, temporal, decimal; mixed byte layouts and mixed decimals compared in place); `min_element_wise`, `max_element_wise` (numeric, temporal, decimal, fixed_size_binary, string and binary layouts; `ElementWiseAggregateOptions`) |
| Logical | `logical.zig` | `and`, `or`, `xor`, `and_not`, `invert`, `and_kleene`, `or_kleene`, `and_not_kleene` |
| Aggregations | `aggregate_basic.zig` | `sum`, `mean`, `min`, `max`, `min_max`, `any`, `all` (`ScalarAggregateOptions`; sum, mean, min, max, min_max also on decimal128 / decimal256; min, max, min_max also on the null type, bool, temporal types, decimal32 / decimal64, string / binary layouts and fixed_size_binary), `count` (`CountOptions`), `count_all` (one argument or none) |
| More aggregations | `aggregate_stats.zig` | `product`, `first`, `last`, `first_last` (`ScalarAggregateOptions`), `variance`, `stddev` (`VarianceOptions`), `skew`, `kurtosis` (`SkewOptions`; numbers and decimal32..decimal256), `count_distinct` (`CountOptions`), `index` (`IndexOptions`); vector `mode` (`ModeOptions`) and `quantile` (`QuantileOptions`) |
| Math | `math.zig` | `sqrt`, `exp`, `expm1`, `ln`, `log10`, `log2`, `log1p`, `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh` (+ `_checked` where Arrow has one), `atan2`, `logb` (+ `_checked`), `power` (+ `_checked`), `bit_wise_and`, `bit_wise_or`, `bit_wise_xor`, `bit_wise_not`, `shift_left`, `shift_right` (+ `_checked`) |
| Rounding | `round.zig` | `round` (`RoundOptions`), `round_to_multiple` (`RoundToMultipleOptions`), `round_binary` (`RoundBinaryOptions`), `ceil`, `floor`, `trunc` (floats, integers, decimal128 / decimal256) |
| Temporal arithmetic | `temporal_arith.zig` | the timestamp, date, time and duration kernels of `add`, `subtract`, `multiply`, `divide`, `negate`, `abs` (+ `_checked`) |
| Associative transforms | `associative.zig` | `unique`, `value_counts`, `dictionary_encode` (`DictionaryEncodeOptions`); dictionary inputs keep their dictionary (see Dictionary inputs) |
| Selecting and null replacement | `selection.zig` | `coalesce`, `if_else`, `case_when`, `choose` (scalar); `drop_null` (also record batches and tables), `fill_null_forward`, `fill_null_backward`, `replace_with_mask`, `scatter` (`ScatterOptions`), `inverse_permutation` (`InversePermutationOptions`), `indices_nonzero` (vector) |
| Cumulative and pairwise | `cumulative.zig` | `cumulative_sum`, `cumulative_prod` (+ `_checked`), `cumulative_min`, `cumulative_max`, `cumulative_mean` (`CumulativeOptions`), `pairwise_diff` (+ `_checked`, `PairwiseOptions`), `winsorize` (`WinsorizeOptions`) (vector); `random` (`RandomOptions`, nullary scalar) |
| Sorts and partitions | `sort.zig` | `sort_indices` (array / chunked columns, or one record batch / table by key names), `array_sort_indices` (`SortOptions`), `partition_nth_indices` (`PartitionNthOptions`), `select_k_unstable` (`SelectKOptions`; also record batches / tables), `rank` (`RankOptions`), `rank_quantile`, `rank_normal` (`RankQuantileOptions`), `search_sorted` (`SearchSortedOptions`); `sortIndicesRecordBatch`, `sortIndicesTable` |
| Categorizations | `is_null.zig` | `is_null` (`NullOptions`), `is_valid`, `true_unless_null`, `is_nan`, `is_finite`, `is_inf` (also float16) |
| Conversions | `cast.zig` | `cast` (`CastOptions.to_type`): numbers and bool among themselves, to and from text, string and binary layouts, date32 / date64 / timestamp / time32 / time64 / duration units, timestamp to time of day, same-width integers and temporal types, dictionary decode and encode (every index width), the null type; float16 with the numbers and text; nested: the list layouts (list, large_list, list_view, large_list_view, fixed_size_list, map) among themselves with the child cast recursively, struct to struct by field name; decimals to and from decimals, integers, floats and text (`decimal.zig`) |
| Strings | `string.zig` | `ascii_upper`, `ascii_lower`, `ascii_swapcase`, `ascii_capitalize`, `ascii_title`, `ascii_reverse` (ASCII letters), `utf8_upper`, `utf8_lower`, `utf8_swapcase`, `utf8_capitalize`, `utf8_title` (Unicode simple case mappings), `utf8_length`, `binary_length`, `ascii_trim`, `ascii_ltrim`, `ascii_rtrim`, `utf8_trim`, `utf8_ltrim`, `utf8_rtrim`, `ascii_*trim_whitespace`, `utf8_*trim_whitespace`, `utf8_slice_codeunits`, `utf8_reverse`, `binary_join_element_wise`, `starts_with`, `ends_with`, `match_substring`, `string_is_ascii`, `ascii_is_alnum`, `ascii_is_alpha`, `ascii_is_decimal`, `ascii_is_lower`, `ascii_is_printable`, `ascii_is_space`, `ascii_is_title`, `ascii_is_upper`, `utf8_is_alnum`, `utf8_is_alpha`, `utf8_is_decimal`, `utf8_is_digit`, `utf8_is_lower`, `utf8_is_numeric`, `utf8_is_printable`, `utf8_is_space`, `utf8_is_title`, `utf8_is_upper` (string, large_string, string_view) |
| String transforms | `string_transform.zig` | `replace_substring`, `split_pattern`, `ascii_split_whitespace`, `utf8_split_whitespace` (`list<T>` result), `ascii_lpad`, `ascii_rpad`, `ascii_center`, `utf8_lpad`, `utf8_rpad`, `utf8_center`, `utf8_zero_fill`, `utf8_replace_slice`, `count_substring`, `find_substring`, `binary_repeat` (string and binary layouts; padding, zero fill, replace_slice and whitespace splits on the string layouts), `binary_slice`, `binary_reverse`, `binary_replace_slice` (binary layouts), `binary_join` (`list<T>` / `large_list<T>` with a `T` separator) |
| Temporal | `temporal.zig` | `year`, `month`, `day`, `day_of_week`, `day_of_year`, `iso_year`, `iso_week`, `us_week`, `us_year`, `week`, `iso_calendar`, `year_month_day`, `quarter`, `is_leap_year` (date32, date64, timestamp); `hour`, `minute`, `second`, `millisecond`, `microsecond`, `nanosecond` (date64, timestamp, time32, time64); `subsecond` (timestamp, time32, time64) |
| Temporal rounding and differences | `temporal_round.zig` | `floor_temporal`, `ceil_temporal`, `round_temporal`; `years_between`, `quarters_between`, `month_interval_between`, `weeks_between`, `days_between` (date32, date64, timestamp); `hours_between`, `minutes_between`, `seconds_between`, `milliseconds_between`, `microseconds_between`, `nanoseconds_between`, `day_time_interval_between`, `month_day_nano_interval_between` and the rounding functions (also time32, time64) |
| Temporal text | `temporal_text.zig` | `strftime` (`StrftimeOptions`), `strptime` (`StrptimeOptions`), and the casts between the temporal types and the string layouts (called by `cast`) |
| Timezones | `temporal_zone.zig`, `tz.zig` | `assume_timezone` (`AssumeTimezoneOptions`), `local_timestamp`, `is_dst`; the localization every temporal function applies to a non-UTC timestamp; the TZif reader and the bundled IANA table |
| Normalization | `string_normalize.zig` | `utf8_normalize` (`Utf8NormalizeOptions`: NFC, NFD, NFKC, NFKD) |
| Set lookup | `set_lookup.zig` | `is_in`, `index_in` (numbers, bool, temporal, intervals, decimals, fixed_size_binary, string and binary layouts; a value set of another type is cast) |
| Approximate quantiles | `tdigest.zig` | `tdigest` (`TDigestOptions`), `approximate_median` (`ScalarAggregateOptions`), `hash_tdigest`, `hash_approximate_median` (Arrow's merging t-digest) |
| Pivot | `pivot.zig` | `pivot_wider`, `hash_pivot_wider` (`PivotWiderOptions`) |
| Grouped aggregations | `hash_aggregate.zig` | `Grouper`, `groupBy` (per-aggregate options); kind `hash_aggregate`: `hash_sum`, `hash_mean`, `hash_min`, `hash_max`, `hash_min_max`, `hash_any`, `hash_all`, `hash_product`, `hash_first`, `hash_last`, `hash_first_last` (`ScalarAggregateOptions`; sum, mean, min, max, min_max also on decimal128 / decimal256), `hash_variance`, `hash_stddev` (`VarianceOptions`), `hash_skew`, `hash_kurtosis` (`SkewOptions`), `hash_count`, `hash_count_distinct`, `hash_distinct` (`CountOptions`), `hash_one`, `hash_list`, `hash_count_all` (values plus uint32 group ids, one row per group); hash_min, hash_max, hash_min_max also on the ordered non-numeric types of min_max |

Element-wise functions take arrays, chunked arrays or scalars (a chunked input gives a chunked output, all
scalars give a scalar); aggregates take one array, chunked array or scalar and return a scalar; grouped
aggregates take a value array (or chunked array) and the uint32 group ids of a `Grouper` and return one
row per group:

```zig
const hash = zarrow.compute.kernels.hash_aggregate;
var batch = try hash.groupBy(&ctx, &.{.{ .name = "k", .array = keys }}, &.{
    .{ .function = "hash_sum", .input = values, .name = "total" },
    .{ .function = "hash_count_all", .name = "rows" },
});
defer batch.deinit(); // columns: k (one row per group, first-seen order), total, rows
```

### Implicit casts

Arrow C++ functions pick a kernel with `DispatchBest`, which may cast arguments first. Here a function
registered with an `ImplicitCast` mode (`FunctionRegistry.setImplicitCast`) does the same when no kernel
matches: dictionary arguments are decoded to their value type and (not for `min_element_wise` /
`max_element_wise`) a null-type argument takes the type of the first other argument; then the arguments
are cast with the `cast` function to a common type and the call is resolved again. The common type:

- numbers (`commonNumericType`, Arrow's `CommonNumeric`): any float64 gives float64, else any float32 or
  float16 gives float32 (even for int64, as Arrow; float16 also against itself: there is no float16
  arithmetic or comparison kernel, Arrow promotes it the same way); integers of one signedness give the widest; signed with unsigned
  gives the signed type wider than every unsigned one (int8 + uint8 is int16, int32 + uint32 is int64,
  uint64 + int64 is int64);
- a decimal with a float: float64 for every argument (`commonDecimalFloatType`, as Arrow);
- temporal types of different units or kinds (`commonTemporalType`, Arrow's `CommonTemporal`; comparisons
  and min / max_element_wise only): a timestamp of the finest unit in the shared timezone (with dates;
  naive and zoned never mix), date64 over date32, the finest duration or time.

When only dictionaries were decoded or null types replaced, each argument keeps its own decoded type (so a
dictionary of strings compares with a string_view through the mixed-layout kernel). The cast is safe
unless `ExecConfig.safe_cast` is false, so a value the common type cannot hold (2^60 to float64, 2^63 as
int64, a timestamp past the finer unit) is `InvalidCast`, as in Arrow. `add`, `subtract`, `multiply`,
`divide` (and `_checked`), the bit-wise functions and the shifts use `.numeric`, the six comparisons
`.comparison`, `min_element_wise` / `max_element_wise` `.element_wise`. Three more modes follow Arrow's
floating-point dispatch: `.floating` (integers and decimals count as float64, then the widest float:
`sqrt`, `ln`, `sin`, ..., `atan2`, `logb`; atan2(float32, int32) is float64, as Arrow),
`.decimal_to_float` (`commonNumericType`, else a decimal makes float64: `power`) and
`.integer_to_float` (an integer goes to float64: `ceil`, `floor`, `trunc`; decimals keep their own
kernels). `.decode` only decodes dictionaries (`is_in`, `index_in`: a dictionary input is looked up by
its values, as pyarrow 26). `resolveResultType` reports the type of the cast call.

The multiplexers (`coalesce`, `if_else`, `case_when`, `choose`) cast their value arguments themselves
(`selection.commonValueType`): numbers to `commonNumericType`, temporal types to `commonTemporalType`,
string / binary layouts to binary or the large layout when one argument is, decimals to the precision and
scale that hold every value. `is_in` / `index_in` cast a value set of another type to the input type,
and the input to the value set's type when that fails (pyarrow 26).

Some mixed types need no cast because a kernel takes them directly: decimals of different widths,
precisions and scales and a decimal with an integer (arithmetic with Arrow's result types; comparisons by
exact value), and two different string / binary / fixed_size_binary layouts in the comparisons (bytes
compared in place). Exact semantics, including the differences from Arrow C++, are in
[compute-support-matrix.md](compute-support-matrix.md).

### Decimals

Decimal values are the unscaled integers of their storage (i32, i64, i128, i256), read whole through the
typed accessors after `validateLayout`; every kernel first checks that the precision is within the width
(1..9, 18, 38, 76; `InvalidInput` otherwise, a bad cast target is `InvalidOptions`). Arithmetic computes in
i128 (decimal128 results) or i256 (decimal256 results) with Arrow C++'s result precision and scale; the
unchecked functions wrap like Arrow C++, the `_checked` ones also bound every valid row by the result
precision. Sums are exact (256 / 512 bits) until finalize, which applies `ExecConfig.overflow_mode`. Casts
follow Arrow C++ bit for bit where pyarrow 26 can express the case (text, floats, rescaling); the two
deliberate differences (every overflowing safe scale-up is rejected; float to decimal128 at scales of 32
and more is rounded exactly) are listed in the support matrix.

### Dictionary inputs

What a function does with a dictionary-encoded argument follows pyarrow 26:

- decoded first (implicit casts): the comparisons, arithmetic, `is_in` / `index_in`, `min_element_wise`
  / `max_element_wise`; `sort_indices` and the other sorts compare decoded values;
- on the indices, keeping the dictionary: `unique` and `value_counts` (the result is of the input's
  dictionary type, unused entries stay; a null index and an index to a null entry are two values),
  group-by keys (`Grouper`, the key column keeps the dictionary) and the values of `hash_count`,
  `hash_count_distinct`, `hash_distinct`; chunks with different dictionaries are unified first for
  `unique` / `value_counts` (Arrow's `DictionaryUnifier`: the first dictionary, then the new values of
  the next ones, the index type kept; dictionaries with null entries cannot be unified, `InvalidInput`),
  and a later group-by batch over a different dictionary is `NotImplemented` (as Arrow);
- the identity: `dictionary_encode`;
- no kernel (`NoMatchingKernel`, as Arrow): `min`, `max`, `min_max`, `sum`, `count_distinct`, `first`,
  `last` and the other numeric aggregates.

Before any index is read, a dictionary argument passes `hash_aggregate.validateDictionary` (the
dictionary's layout and value type, every non-null index below its length) or the equivalent check of
`cast`; `validateLayout` alone checks neither.

### Unicode

The `utf8_*` case functions and predicates read `unicode_tables.zig`, generated from the Unicode
Character Database (`UnicodeData.txt`, Unicode 17.0.0, the version of pyarrow 26's utf8proc) by
`python -I tools/gen_unicode_tables.py UnicodeData.txt 17.0.0`: general-category runs (4144) and
simple upper / lower case ranges with one delta and stride 1 or 2 (205 and 186), each with a bucket
index of 128 code points below U+20000, about 26 KiB of data. A lookup binary-searches between two
bucket entries (code points past U+20000 search the tail of the category runs); about 25 KiB in all. The case kernels
also build, at compile time, a direct table of the 1- and 2-byte code points (U+0000..U+07FF, 8 KiB
per direction) from those ranges. Mappings are simple (one code point to one), as in Arrow's utf8proc
kernels, with the one Arrow difference from UnicodeData.txt (U+00DF uppercases to U+1E9E); special
casing (`ß` to `SS`, final sigma, ligatures) is not applied, in Arrow neither. A mapping may change a
row's UTF-8 length (at most 3/2 of it; the generator checks this and that every mapping stays a scalar
value), so the output is always valid UTF-8. To move to a newer Unicode version, download that
UnicodeData.txt into an empty directory, rerun the generator with the version and check the
`string Unicode case mapping` test against pyarrow.

This repository also ships take/filter/select (`datumTake`, `datumFilter`,
`datumSelect`), concatenation (`concatArrayRefs`), slicing, `dictionaryEncode`, validation and null-like /
empty-like construction. [compute-support-matrix.md](compute-support-matrix.md) maps every category of the
Arrow C++ function catalog to its status here.

### Arrow C++ to zarrow glossary

| Arrow C++ | zarrow |
|---|---|
| `arrow::compute::Function` (name + kernels) | `Function` |
| `arrow::compute::Initialize()` (built-in function catalog) | `registerAll` (per family: `kernels.<family>.register`) |
| `ScalarKernel` / `VectorKernel` / `ScalarAggregateKernel` / `HashAggregateKernel` | `Kernel` with `FunctionKind.scalar` / `.vector` / `.aggregate` / `.hash_aggregate` |
| `Grouper`, `Table::GroupBy` | `kernels.hash_aggregate.Grouper`, `kernels.hash_aggregate.groupBy` |
| `KernelSignature` (`InputType`, `OutputType`, type matchers) | `KernelSignature` (`arity`, `type_check`, `result_type_fn`) |
| `FunctionOptions` | `Options` (tagged union, section 3) |
| `ExecContext` + `MemoryPool` | `ExecContext` + `std.mem.Allocator` (section 6) |
| `Datum` | `Datum` (array, chunked, scalar, record_batch, table) |
| `ExecBatch` / `ExecSpan` chunk-wise execution | exec chunk iterators (section 8) |
| `Status` / `Result<T>` | Zig error unions (`KernelError!T`, section 5) |

---

## 2. Core Types

Main public framework types:

- `FunctionRegistry`
- `KernelSignature`
- `Kernel`
- `ExecContext`
- `Datum`

`Datum` has the five forms of Arrow's `Datum` (`DatumKind`):

- `array: ArrayRef`
- `chunked: ChunkedArray`
- `scalar: Scalar`
- `record_batch: SharedRecordBatch`
- `table: SharedTable`

`retain` / `release`, `dataType`, `length` (rows; 1 for a scalar, as Arrow) and `kind` work on every
form. A record batch or table datum owns its value through a reference-counted handle
(`SharedRecordBatch` / `SharedTable`, `get()` borrows it): `Datum.fromRecordBatch(allocator, batch)` /
`Datum.fromTable(allocator, table)` take the value over after `validateRecordBatch` / `validateTable`
(one column per schema field, of the field's type, every column `num_rows` long, `validateLayout` of
every column and chunk; `InvalidInput` otherwise, and the caller keeps the value). `dataType` of a
tabular datum is the struct of its schema fields (borrowed from the schema; Arrow gives no type).

Tabular datums go to the functions Arrow defines on record batches and tables: `datumTake` /
`datumTakeNullable`, `datumFilter` / `datumFilterChunkAware`, `datumSlice`, `datumSelectColumns`,
`datumStructField` (a column), and the registered functions whose signature sets `tabular`
(`drop_null`, `sort_indices`, `select_k_unstable`). A kernel signature without `tabular` never matches a
tabular argument, and the registry then answers `NotImplemented` (pyarrow raises TypeError /
NotImplemented); helpers, iterators (`inferNaryExecLen`, the chunk iterators) and kernel switches over
`Datum` answer `InvalidInput`. Every `switch` over a `Datum` names the five forms.

---

## 3. Typed Options Model

Compute options are type-safe and tag-based.

### Options families

`OptionsTag`:

- `none`
- `cast`
- `arithmetic`
- `filter`
- `sort`
- `day_of_week`
- `match_substring`
- `slice`
- `join`
- `trim`
- `replace_substring`
- `split_pattern`
- `split`
- `pad`
- `round_temporal`
- `week`
- `set_lookup`
- `null_options`
- `scalar_aggregate`
- `count`
- `element_wise_aggregate`
- `replace_slice`
- `zero_fill`
- `round`, `round_to_multiple`, `round_binary`
- `mode`, `variance`, `skew`, `quantile`, `index`
- `partition_nth`, `select_k`, `rank`, `rank_quantile`, `search_sorted`
- `tdigest`, `pivot_wider`
- `dictionary_encode`
- `scatter`, `inverse_permutation`
- `cumulative`, `pairwise`, `winsorize`, `random`
- `strftime`, `strptime`, `assume_timezone`, `utf8_normalize`
- `custom`

### Payload types

- `CastOptions` (`to_type`, `safe`)
- `ArithmeticOptions`
- `FilterOptions`
- `SortOptions`
- `DayOfWeekOptions` (`count_from_zero`, `week_start`)
- `MatchSubstringOptions` (`pattern`, `ignore_case`; a literal, an RE2 pattern for the `*_regex`
  functions or a LIKE pattern for `match_like`), `ExtractRegexOptions` (`pattern`) for
  `extract_regex` and `extract_regex_span`
- `SliceOptions` (`start`, `stop`, `step`)
- `JoinOptions` (`null_handling`: `emit_null | skip | replace`, `null_replacement`)
- `TrimOptions` (`characters`)
- `ReplaceSubstringOptions` (`pattern`, `replacement`, `max_replacements`)
- `SplitPatternOptions` (`pattern`, `max_splits`, `reverse`), `SplitOptions` (`max_splits`, `reverse`)
- `PadOptions` (`width`, `padding`, `lean_left_on_odd_padding`)
- `RoundTemporalOptions` (`multiple`, `unit`: `CalendarUnit` nanosecond .. year, `week_starts_monday`,
  `ceil_is_strictly_greater`, `calendar_based_origin`)
- `WeekOptions` (`week_starts_monday`, `count_from_zero`, `first_week_is_fully_in_year`)
- `SetLookupOptions` (`value_set`: a borrowed array or chunked array `Datum`, `null_matching_behavior`:
  `match | skip | emit_null | inconclusive`)
- `NullOptions` (`nan_is_null`) for `is_null`
- `ScalarAggregateOptions` (`skip_nulls` = true, `min_count` = 1) for `sum`, `mean`, `min`, `max`,
  `min_max`, `any`, `all`
- `CountOptions` (`mode`: `only_valid | only_null | all`) for `count`
- `ElementWiseAggregateOptions` (`skip_nulls` = true) for `min_element_wise`, `max_element_wise`
- `ReplaceSliceOptions` (`start`, `stop`, `replacement`) for `utf8_replace_slice`, `binary_replace_slice`
- `ZeroFillOptions` (`width`, `padding` = `"0"`) for `utf8_zero_fill`
- `TrimOptions` also serves `utf8_trim` / `utf8_ltrim` / `utf8_rtrim` (`characters` as UTF-8 code points)
- `RoundOptions` (`ndigits` = 0, `round_mode` = `half_to_even`; `RoundMode`: `down`, `up`, `towards_zero`,
  `towards_infinity`, `half_down`, `half_up`, `half_towards_zero`, `half_towards_infinity`,
  `half_to_even`, `half_to_odd`), `RoundToMultipleOptions` (`multiple`: a `Scalar` cast to the input
  type, `round_mode`), `RoundBinaryOptions` (`round_mode`)
- `ModeOptions` (`n` = 1, `skip_nulls`, `min_count` = 0), `VarianceOptions` (`ddof` = 0, `skip_nulls`,
  `min_count`), `SkewOptions` (`skip_nulls`, `biased` = true, `min_count`), `QuantileOptions` (`q` =
  `{0.5}`, `interpolation`: `linear | lower | higher | nearest | midpoint`, `skip_nulls`, `min_count`),
  `IndexOptions` (`value`: a `Scalar`)
- `PartitionNthOptions` (`pivot`, `null_placement`), `SelectKOptions` (`k`, `sort_keys`), `RankOptions`
  (`order`, `null_placement`, `tiebreaker`: `min | max | first | dense`), `RankQuantileOptions` (`order`,
  `null_placement`) for `rank_quantile` / `rank_normal`, `SearchSortedOptions` (`side`: `left | right`)
- `TDigestOptions` (`q` = `{0.5}`, each in [0, 1], at most 65536; `delta` = 100, at most 2^20, at least
  10 used; `buffer_size` = 500, at most 2^24, at least 1 used; `skip_nulls`; `min_count` = 0)
- `PivotWiderOptions` (`key_names`, distinct, at most 65536; `unexpected_key_behavior`: `ignore | raise`)
- `DictionaryEncodeOptions` (`null_encoding`: `mask | encode`)
- `ScatterOptions` (`max_index` = -1: the indices' length), `InversePermutationOptions` (`max_index` = -1,
  `output_type`: a signed integer type, null for the index type); more output rows than
  max(indices, 2^24) is `InvalidOptions`
- `CumulativeOptions` (`start`: a numeric `Scalar` cast safely to the input type, null for the identity;
  `skip_nulls` = false), `PairwiseOptions` (`period` = 1), `WinsorizeOptions` (`lower_limit` = 0,
  `upper_limit` = 1; required, as Arrow), `RandomOptions` (`length`, `initializer`: `seed | system`
  (`system` draws the seed through `ExecConfig.io`), `seed`; required)
- `StrftimeOptions` (`format` = `%Y-%m-%dT%H:%M:%S`, `locale` = `"C"`, the only one), `StrptimeOptions`
  (`format`, `unit`, `error_is_null` = false; required, as Arrow)
- `AssumeTimezoneOptions` (`timezone`: an IANA name or `+HH:MM`, `ambiguous` / `nonexistent`:
  `raise | earliest | latest`, both `raise` by default)
- `Utf8NormalizeOptions` (`form`: `nfc | nfkc | nfd | nfkd`, `nfc` by default)
- `CustomOptions`

Kernels whose options are optional in Arrow (`day_of_week`, `binary_join_element_wise`, `week`,
`weeks_between`, the whitespace splits, the `*_temporal` rounding functions, `is_null`, the scalar
aggregates, `count` and `min_element_wise` / `max_element_wise`) also accept `.none` for the defaults.

Use `KernelSignature.options_check` to validate allowed option families for each kernel.

### SortOptions

`SortOptions` (sort_indices, array_sort_indices), as Arrow 26:

- `order`: `ascending | descending` (array_sort_indices, and every column when `sort_keys` is empty)
- `sort_keys`: one `SortKey` (`name`, `order`, `null_placement` = `at_end`) per array / chunked argument,
  or the columns of one record batch / table argument by `name` (at least one key, else
  `InvalidOptions`); `SelectKOptions.sort_keys` the same
- `null_placement`: `?SortNullPlacement` = null; when set it overrides every key's null placement
  (Arrow's `SortOptions.null_placement`); null: each key's own (`at_end` without keys)

Arrow 26 has no NaN placement option: NaN always sits next to the nulls, between them and the values
(`at_end`: values, NaNs, nulls; `at_start`: nulls, NaNs, values). The former `nan_placement` and
`stable` fields, which changed nothing, are removed; the sorts are always stable (equal rows keep their
input order). A struct key sorts by its fields in order with a null struct row null in every field (Arrow
flattens it); a dictionary key by its decoded values.

---

## 4. Signature and Type Inference

`KernelSignature` supports:

- exact arity;
- variadic minimum arity (`at_least`);
- bounded arity range (`range`);
- optional `type_check` callback;
- optional `options_check` callback;
- optional `result_type_fn` callback;
- optional `result_type_alloc_fn` callback, for a result type that needs storage (a struct over the input
  type: decimal `min_max`, `value_counts`, `first_last`, `mode`; a dictionary over it: `dictionary_encode`).

Result type APIs:

- `KernelSignature.inferResultType(args, options)` (`NotImplemented` for a kernel with only
  `result_type_alloc_fn`)
- `KernelSignature.inferResultTypeAlloc(allocator, args, options)`
- `FunctionRegistry.resolveResultType(...)`
- `FunctionRegistry.resolveResultTypeAlloc(allocator, ...)`: the type is allocated in `allocator` (an
  arena the caller frees) and may borrow from the arguments.

A kernel returning a struct over a run-time type builds it with `dense.structArray`: the field list and
child types live in a refcounted block owned by the array and its slices (owner hooks), so the result
(and a scalar over it) keeps its type valid as long as it lives; `dense.structTypeAlloc` builds the same
type in an arena for `result_type_alloc_fn`.

When signature or options checks fail, errors are reported through `KernelError`.

---

## 5. Error Model

`KernelError` includes framework and execution-level categories such as:

- `InvalidArity`
- `InvalidOptions`
- `InvalidInput`
- `UnsupportedType`
- `InvalidCast`
- `Overflow`
- `DivideByZero`
- `NoMatchingKernel`

Helpful utility helpers are provided for common arithmetic/cast behavior:

- `intCastOrInvalidCast(...)`
- `arithmeticDivI64(...)`

Error conventions, mirroring Arrow C++ (`Status`, `Result<T>`, no exceptions across the API):

- every fallible kernel entry point returns an error union; a kernel never panics on input it was given
  through the public API, including malformed arrays (validation errors are returned, not asserted);
- `std.debug.assert` is reserved for internal invariants that cannot fail on valid input (Arrow's `DCHECK`),
  and is stripped in release builds, so it must not guard input;
- `OutOfMemory` passes through unchanged; kernels do not retry or fall back silently;
- an unsupported type is `UnsupportedType`, a malformed value is `InvalidInput`, and an arity or option
  mismatch is reported through the signature errors above, so callers can distinguish "cannot" from
  "will not".

---

## 6. ExecContext and Config

`ExecConfig` currently includes:

- `safe_cast`: `cast` is safe only when both `safe_cast` and `CastOptions.safe` are true
- `overflow_mode` (`checked | wrapping | saturating`): the unchecked arithmetic functions (`add`, ...) wrap
  like Arrow C++ under `checked` and `wrapping` and saturate under `saturating` (the `_checked` names always
  return `Overflow`); `sum`, which has no `_checked` twin, returns `Overflow` under `checked`, wraps under
  `wrapping` and clamps under `saturating`
- `threads`
- `arena_allocator`
- `io` (`?std.Io`, null by default): timezone names are read from the system zoneinfo
  (`/usr/share/zoneinfo`, Linux and macOS) when set, otherwise from the bundled IANA table
  (`src/compute/tzdata/tzdata.bin`); `random` with `initializer = system` draws its seed through it

Context construction:

- `ExecContext.init(allocator, &registry)`
- `ExecContext.initWithConfig(allocator, &registry, config)`

For borrowed scalar string/binary data, use:

- `ExecContext.dupScalarString(...)`
- `ExecContext.dupScalarBinary(...)`

Memory conventions (Arrow's `default_memory_pool()` has no implicit counterpart here):

- every allocation is made through the allocator the caller passed (`ExecContext.allocator`, or the datum's
  allocator through `inferDatumAllocator`); kernels never reach for a global allocator;
- `ExecContext.tempAllocator()` is for scratch that dies with the call (uses `arena_allocator` when set);
- output buffers are built with `OwnedBuffer` and handed over with `toShared`; a kernel that produces an array
  returns it with `ArrayRef.fromOwnedInlineBuffers` so node and buffers are one allocation;
- builders may keep a spare buffer from a previous batch, but reuse it only once `SharedBuffer.reclaim`
  shows no live array references it (nothing may alias a buffer a live `ArrayRef` still holds).

---

## 7. Aggregate Lifecycle Interface

For stateful aggregate kernels:

- `AggregateLifecycle.init`
- `AggregateLifecycle.update`
- `AggregateLifecycle.merge`
- `AggregateLifecycle.finalize`
- `AggregateLifecycle.deinit`

Runtime flow:

1. register kernel with `aggregate_lifecycle`;
2. call `ctx.beginAggregate(...)` to create `AggregateSession`;
3. call `update/merge/finalize/deinit`.

If lifecycle is missing, `beginAggregate` returns `error.MissingLifecycle`. `beginAggregate` also
resolves the `hash_*` functions (kind `hash_aggregate`): `update` takes (values, group ids) batches,
`merge` adds a state over the same group ids, `finalize` returns one row per group seen so far.

---

## 8. Execution and Chunk-Aware Helpers

Framework helpers include:

- null propagation: `unaryNullPropagates`, `binaryNullPropagates`, `naryNullPropagates`;
- scalar broadcast length inference: `inferBinaryExecLen`, `inferNaryExecLen`;
- chunk-aligned iterators: `UnaryExecChunkIterator`, `BinaryExecChunkIterator`, `NaryExecChunkIterator`;
- `dense.execElementwise` / `dense.execElementwiseWith` (any arity, options passed to every chunk): the
  driver of the element-wise families (array / chunked / scalar dispatch, `validateLayout` of every input
  chunk, null scalars under intersection), with `dense.Bits` / `dense.mapBits` for packed-bit operands,
  `dense.StringOperand` for the rows of the string and binary layouts, `dense.VarBinaryOutput` for a
  string / large_string / string_view output (offsets checked against the 32-bit range) and
  `dense.chunkArray` to read a scalar operand as a one-row array. A string or binary scalar result owns
  its bytes through `Scalar.payload`.

Permutation-family helpers include:

- `chunkedResolveLogicalIndices`
- `datumTake`, `datumTakeNullable`
- `datumFilterSelectionIndices`
- `datumFilterChunkAware`
- `datumSelect`, `datumSelectNullable`
- `datumSlice`, `datumSelectColumns` (and the tabular forms of `datumTake` / `datumFilter`: every column of
  a record batch or table, a table kept chunked)

These are intended to help kernels preserve chunked execution/output where possible, instead of pre-concatenating chunked inputs.

### Null handling

Arrow C++ kernels declare a `NullHandling` policy; here the policy is a choice of helper:

- **intersection** (the default for element-wise kernels): combine input validity once, bitmap word by
  word, then run the value loop over plain typed slices with no per-element null check. Null slots may
  compute garbage that the output validity hides. Use `dense.DenseBinaryOutput`, `dense.validityBits` and
  `dense.laneMask`; `unaryNullPropagates` / `binaryNullPropagates` / `naryNullPropagates` answer the
  question for scalar inputs.
- **computed**: the kernel writes its own validity (comparisons with Kleene logic, `is_null`-style
  categorizations). Write the bitmap explicitly; do not derive it per row with `isNull`.
- **output not null**: aggregates and count-like kernels return a scalar; use `Scalar.init`.

Per-element `isNull` calls inside a hot loop are the one pattern to avoid: they were measured at 4 to
15 ns per row against under 1 ns for the dense path.

### Validity at trust boundaries

Kernels may receive arrays that came from IPC or the C data interface. They must bounds-check what they
read (buffer counts, offsets windows, index ranges, dictionary lengths) and return `InvalidInput` rather than
trust `length` or `null_count`. The selection and concat kernels in this repository are the reference for
how much checking is expected; nothing in a kernel may weaken what `validateLayout` establishes.
`validateLayout` does not check UTF-8 or dictionary indices: the string kernels that decode code points
reject a non-null row that is not UTF-8 (`InvalidInput`), a cast to a string type runs the UTF-8 check of
`validateFull` on a binary source, and `cast` checks dictionary indices before decoding.

---

## 9. Recommended Extension Path

For new sort/take/filter kernels:

1. keep one logical behavior for both array and chunked inputs;
2. drive behavior through typed options (especially `SortOptions`), not hardcoded defaults;
3. reuse core chunk-aware helpers for index resolution, null handling, and gather/filter execution;
4. test array/chunked logical equivalence, including misaligned chunk-boundary cases.

### Adding a function, step by step

This follows the Arrow C++ developer workflow (one function, its kernels, options, tests, benchmark, docs)
translated to this repository's layout. Kernels are added here, not in a separate repository.

1. **Name it like Arrow.** Use the Arrow C++ function name (`add`, `less`, `sum`, `sort_indices`), with
   the `_checked` suffix for the overflow-checking variant and `hash_` prefix for grouped aggregates, so
   users can read the Arrow reference. Options types end in `Options`.
2. **Place the code.** One file per family, underscores in file names (`arithmetic.zig`,
   `comparison.zig`, `aggregate_basic.zig`); tests in `test_<family>.zig` next to it; a benchmark in
   `benchmarks/<family>_benchmark.zig`. Public symbols are re-exported from the family's `mod.zig`;
   anything not re-exported is internal.
3. **Write the signature first.** `KernelSignature.unary` / `.binary` / `.atLeast` with a `type_check`
   that accepts exactly the types the kernel handles (`unaryNumeric`, `binarySameNumeric`, or a custom
   check) and a `result_type_fn` when the output type differs from the input.
4. **Pick the kernel kind and options family.** `registerScalarKernel` for element-wise,
   `registerVectorKernel` for whole-array (sort, unique, cumulative), `registerAggregateKernel` with an
   `AggregateLifecycle` for reductions. Declare the options family the kernel reads
   (`options_check`); read `ExecContext.overflowMode()` and `safeCastEnabled()` instead of re-deriving them.
5. **Execute chunk-wise.** Iterate with the exec chunk iterators so a chunked input yields a chunked
   output; never concatenate inputs first. Use the dense helpers for validity (section 8).
6. **Return errors, do not assert.** See section 5. Overflow in `checked` mode is `Overflow`, division by
   zero is `DivideByZero`, a narrowing cast that loses data in safe mode is `InvalidCast`.
7. **Test.** For every kernel: a hand-written expected-value test; an oracle test comparing the kernel
   against a simple per-element loop on random data with nulls and sliced inputs (offset != 0); array
   versus chunked equivalence with chunk boundaries that do not align; empty input; all-null input; the
   error cases of step 6. Register the test file in the family's `test.zig`.
8. **Benchmark.** One case per kernel and type at 200k rows in `benchmarks/<family>_benchmark.zig`,
   printing the common CSV line; allocate through `benchmarks/bench_allocator.zig` and make one untimed
   warm-up call so output memory is reused across iterations (as `compute_benchmark.zig` does); register
   the case name in `tools/benchmark_ci_check.py` (`EXPECTED_BENCHES`), `tools/benchmark_ci_thresholds.json`
   and `tools/test_benchmark_ci_check.py` (`VALID_ROWS`) so the CI smoke run covers it.
9. **Document.** Add a `///` docstring on the public entry point (summary line in the infinitive, as in
   Arrow: "Add two numeric arrays", not "Adds"), and flip the function's row in
   [compute-support-matrix.md](compute-support-matrix.md).

### Source conventions

Taken from the Arrow C++ [conventions](https://arrow.apache.org/docs/developers/cpp/conventions.html) where
they map onto Zig:

- file names use underscores; test files are `test_*.zig`; benchmark executables are `<name>_benchmark`;
- `///` docstrings on every `pub` declaration, summary in the infinitive; `//` for ordinary comments;
- no hidden work in constructors: anything that can fail or allocate is a `pub fn init(...) !T` factory;
- public functions take an explicit `std.mem.Allocator` or an `ExecContext`, never a global;
- errors are returned, invariants are asserted (section 5);
- code must build on both supported Zig versions (see the README), so no builtins or std APIs that exist in
  only one of them.

---

## 10. Diagnostics

Readable mismatch diagnostics are available:

- `KernelSignature.explainMismatch`
- `KernelSignature.explainInferResultTypeFailure`
- `FunctionRegistry.explainResolveKernelFailure`
- `FunctionRegistry.explainResolveResultTypeFailure`

---

## 11. References

- Example: [`examples/compute_lifecycle.zig`](../examples/compute_lifecycle.zig)
- Kernel family benchmarks: [`benchmarks/compute_benchmark.zig`](../benchmarks/compute_benchmark.zig)
- Support status against the Arrow catalog: [compute-support-matrix.md](compute-support-matrix.md)
- Readiness note: [ecosystem-readiness.md](ecosystem-readiness.md)
- Arrow C++: [compute functions](https://arrow.apache.org/docs/cpp/compute.html),
  [compute tutorial](https://arrow.apache.org/docs/cpp/tutorials/compute_tutorial.html),
  [developer conventions](https://arrow.apache.org/docs/developers/cpp/conventions.html)
