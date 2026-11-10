# Aggregations: function notes

Up: [function matrix](matrix.md) · [compute support matrix](../../compute-support-matrix.md)

Scalar aggregate functions reduce an array, a chunked array or a scalar (one row) to one scalar, as Arrow's scalar aggregate kernels; vector `mode` and `quantile` return arrays. Every kernel has an `AggregateLifecycle` (update by batch, merge, finalize), so chunks merge exactly as one array; results are checked against pyarrow 26 (floats within 1e-12 relative unless stated). Evidence: `src/compute/aggregate_basic.zig:63`, `src/compute/test_aggregate_basic.zig:"aggregate chunked input with misaligned chunks and the update / merge lifecycle equal the array result"`.

## Summary

| Function | Arity | Input types | Output type | Options |
|---|---|---|---|---|
| [`all`](#all) | Unary | bool | bool | `ScalarAggregateOptions` |
| [`any`](#any) | Unary | bool | bool | `ScalarAggregateOptions` |
| [`approximate_median`](#approximate_median) | Unary | numbers, decimal128 / decimal256 | float64 | `ScalarAggregateOptions` |
| [`count`](#count) | Unary | any type | int64 | `CountOptions` |
| [`count_all`](#count_all) | Nullary or Unary | any type, or none | int64 | none |
| [`count_distinct`](#count_distinct) | Unary | every key type | int64 | `CountOptions` |
| [`first`](#first) | Unary | numbers, bool, temporal, decimals, string / binary layouts, fixed_size_binary | input type | `ScalarAggregateOptions` |
| [`first_last`](#first_last) | Unary | as `first` | `struct<first, last>` of the input type | `ScalarAggregateOptions` |
| [`index`](#index) | Unary | every key type | int64 | `IndexOptions` |
| [`kurtosis`](#kurtosis) | Unary | numbers, decimal32..decimal256 | float64 | `SkewOptions` |
| [`last`](#last) | Unary | as `first` | input type | `ScalarAggregateOptions` |
| [`max`](#max) | Unary | numbers, decimal128 / decimal256, null, bool, date32, date64, timestamp, time32, time64, string / binary layouts, fixed_size_binary, plus duration, string_view / binary_view, decimal32 / decimal64 | input type | `ScalarAggregateOptions` |
| [`mean`](#mean) | Unary | int8..int64, uint8..uint64, float32/64, decimal128 / decimal256, bool, null | float64; decimals keep the input type | `ScalarAggregateOptions` |
| [`min`](#min) | Unary | numbers, decimal128 / decimal256, null, bool, date32, date64, timestamp, time32, time64, string / binary layouts, fixed_size_binary, plus duration, string_view / binary_view, decimal32 / decimal64 | input type | `ScalarAggregateOptions` |
| [`min_max`](#min_max) | Unary | numbers, decimal128 / decimal256, null, bool, date32, date64, timestamp, time32, time64, string / binary layouts, fixed_size_binary, plus duration, string_view / binary_view, decimal32 / decimal64 | `struct<min, max>` of the input type | `ScalarAggregateOptions` |
| [`pivot_wider`](#pivot_wider) | Binary | keys: string / binary / fixed_size_binary / integer; values: any type | struct scalar, one field per key name | `PivotWiderOptions` |
| [`product`](#product) | Unary | int8..int64, uint8..uint64, float32/64, decimal128 / decimal256, bool, null | int64 / uint64 / float64; decimals keep the input type | `ScalarAggregateOptions` |
| [`skew`](#skew) | Unary | numbers, decimal32..decimal256 | float64 | `SkewOptions` |
| [`stddev`](#stddev) | Unary | numbers, decimal32..decimal256 | float64 | `VarianceOptions` |
| [`sum`](#sum) | Unary | int8..int64, uint8..uint64, float32/64, decimal128 / decimal256, bool, null | int64 / uint64 / float64; decimals at precision 38 / 76 | `ScalarAggregateOptions` |
| [`tdigest`](#tdigest) | Unary | numbers, decimal128 / decimal256 | float64 array, one value per q | `TDigestOptions` |
| [`variance`](#variance) | Unary | numbers, decimal32..decimal256 | float64 | `VarianceOptions` |
| [`mode`](#mode) | Unary | numbers, bool, decimals | `struct<mode: T, count: int64>` array | `ModeOptions` |
| [`quantile`](#quantile) | Unary | numbers, decimals | float64 (`linear`, `midpoint`) or input type (`lower`, `higher`, `nearest`) | `QuantileOptions` |

## all

`all` is true when every valid row is true. Input: bool, array, chunked or scalar.

- Options: `ScalarAggregateOptions` (`skip_nulls` default true, `min_count` default 1; Arrow defaults when `none`). A null row without `skip_nulls`, or fewer valid rows than `min_count`, gives a null result.
- Without `skip_nulls` it uses Kleene logic: a false row gives false even with nulls present, otherwise a null makes the result null.
- An empty or all-null input gives null under the default `min_count = 1`.

Evidence: `src/compute/aggregate_basic.zig:80`, `src/compute/aggregate_basic.zig:63`, `src/compute/test_aggregate_basic.zig:"aggregate hand-written values: sum, mean, min, max, min_max, count, count_all, any, all"`, `src/compute/test_aggregate_basic.zig:"aggregate oracle matches a per-element loop on random sliced inputs with nulls for every numeric type"`, `src/compute/test_aggregate_basic.zig:"aggregate chunked input with misaligned chunks and the update / merge lifecycle equal the array result"`

## any

`any` is true when at least one valid row is true. Input: bool, array, chunked or scalar.

- Options: `ScalarAggregateOptions` (`skip_nulls` default true, `min_count` default 1; Arrow defaults when `none`). A null row without `skip_nulls`, or fewer valid rows than `min_count`, gives a null result.
- Without `skip_nulls` it uses Kleene logic: a true row gives true even with nulls present, otherwise a null makes the result null.

Evidence: `src/compute/aggregate_basic.zig:79`, `src/compute/aggregate_basic.zig:63`, `src/compute/test_aggregate_basic.zig:"aggregate hand-written values: sum, mean, min, max, min_max, count, count_all, any, all"`, `src/compute/test_aggregate_basic.zig:"aggregate oracle matches a per-element loop on random sliced inputs with nulls for every numeric type"`, `src/compute/test_aggregate_basic.zig:"aggregate chunked input with misaligned chunks and the update / merge lifecycle equal the array result"`

## approximate_median

`approximate_median` is `tdigest` with `q = [0.5]`, returned as one float64 scalar. It uses the same merging t-digest (see [`tdigest`](#tdigest)): NaN is skipped but counts as a valid row (Arrow), and the result equals pyarrow 26 within 1e-15 relative on 5000 rows with nulls and NaN (observed; the tests check 1e-9).

Options: `ScalarAggregateOptions` (`skip_nulls`, `min_count`).

Evidence: `src/compute/tdigest.zig:535`, `src/compute/tdigest.zig:531`, `src/compute/test_tdigest.zig:"tdigest and approximate_median hand-written values, options and types (pyarrow 26 results)"`, `src/compute/test_tdigest.zig:"tdigest on 5000 rows with nulls and NaN: array, chunked (merged per chunk) and small delta follow pyarrow 26"`

## count

`count` counts rows of any type with a validity bitmap. Options: `CountOptions.mode` = `only_valid` (default), `only_null` or `all`.

- An empty or all-null input gives 0 (not null).
- A union or run-end encoded array counts its logical nulls (the selected child's row, the run's value) through `is_null.logicalNullCount`, as Arrow.
- A scalar counts as one row.

Evidence: `src/compute/aggregate_basic.zig:76`, `src/compute/aggregate_basic.zig:63`, `src/compute/test_aggregate_basic.zig:"aggregate hand-written values: sum, mean, min, max, min_max, count, count_all, any, all"`, `src/compute/test_aggregate_basic.zig:"aggregate oracle matches a per-element loop on random sliced inputs with nulls for every numeric type"`, `src/compute/test_aggregate_basic.zig:"aggregate chunked input with misaligned chunks and the update / merge lifecycle equal the array result"`

## count_all

`count_all` counts every row, null or not. It takes its one argument or none: with no argument it follows Arrow's nullary calling convention (no batch, 0 rows), so the result is 0.

Evidence: `src/compute/aggregate_basic.zig:77`, `src/compute/aggregate_basic.zig:63`, `src/compute/test_aggregate_basic.zig:"aggregate hand-written values: sum, mean, min, max, min_max, count, count_all, any, all"`, `src/compute/test_aggregate_basic.zig:"aggregate oracle matches a per-element loop on random sliced inputs with nulls for every numeric type"`, `src/compute/test_aggregate_basic.zig:"aggregate chunked input with misaligned chunks and the update / merge lifecycle equal the array result"`

## count_distinct

`count_distinct` counts distinct values on every key type (numbers, float16, bool, temporal, intervals, decimals, fixed_size_binary, string / binary layouts). A null is one value; `CountOptions.mode` chooses whether the null is counted (`only_valid` default, `only_null`, `all`). Floats compare by their bits, as pyarrow 26 hashes them: NaN with another sign or payload is another value, -0.0 and 0.0 two (`count_distinct([0.0, -0.0, nan, -nan])` is 4); evidence `src/compute/test_associative.zig:"hash-based kernels compare floats by bits: NaN sign and payload, signed zeros; nulls, sliced, chunked (pyarrow 26 results)"`.

Evidence: `src/compute/aggregate_stats.zig:84`, `src/compute/aggregate_basic.zig:63`, `src/compute/test_aggregate_stats.zig:"first, last, first_last, count_distinct and index hand-written values and options (pyarrow 26 results)"`, `src/compute/test_aggregate_stats.zig:"aggregate stats on random sliced inputs with nulls: chunked inputs and the update / merge lifecycle equal the array result"`, `src/compute/test_aggregate_stats.zig:"aggregate stats errors: options, types, empty and all-null inputs, malformed input"`

## first

`first` returns the first row of the input in row order across chunks.

- With `skip_nulls` (default) the first valid row; without it the first row, which may be null.
- fixed_size_binary gives a scalar of the input width.
- Options: `ScalarAggregateOptions` (`skip_nulls` default true, `min_count` default 1; Arrow defaults when `none`). A null row without `skip_nulls`, or fewer valid rows than `min_count`, gives a null result.

Evidence: `src/compute/aggregate_stats.zig:81`, `src/compute/aggregate_basic.zig:63`, `src/compute/test_aggregate_stats.zig:"first, last, first_last, count_distinct and index hand-written values and options (pyarrow 26 results)"`, `src/compute/test_aggregate_stats.zig:"aggregate stats on random sliced inputs with nulls: chunked inputs and the update / merge lifecycle equal the array result"`, `src/compute/test_aggregate_stats.zig:"aggregate stats errors: options, types, empty and all-null inputs, malformed input"`

## first_last

`first_last` returns both ends at once as a struct scalar; the rules of [`first`](#first) and [`last`](#last) apply to each field.

Evidence: `src/compute/aggregate_stats.zig:83`, `src/compute/aggregate_basic.zig:63`, `src/compute/test_aggregate_stats.zig:"first, last, first_last, count_distinct and index hand-written values and options (pyarrow 26 results)"`, `src/compute/test_aggregate_stats.zig:"first, last and first_last on fixed_size_binary: nulls, skip_nulls, sliced and chunked (pyarrow 26 results)"`

## index

`index` returns the position of the first row equal to `IndexOptions.value`, or -1 when absent.

- The value is cast to the input type first.
- A null or NaN value is never found (-1); floats compare by value, so `index([0.0, -0.0], -0.0)` is 0 (pyarrow 26; `src/compute/test_associative.zig:"hash-based kernels compare floats by bits: NaN sign and payload, signed zeros; nulls, sliced, chunked (pyarrow 26 results)"`).

Evidence: `src/compute/aggregate_stats.zig:86`, `src/compute/aggregate_basic.zig:63`, `src/compute/test_aggregate_stats.zig:"first, last, first_last, count_distinct and index hand-written values and options (pyarrow 26 results)"`, `src/compute/test_aggregate_stats.zig:"aggregate stats errors: options, types, empty and all-null inputs, malformed input"`

## kurtosis

`kurtosis` returns the excess kurtosis as float64. Options: `SkewOptions` (`skip_nulls` default true, `biased` default true, `min_count` default 0).

- Two-pass moments per chunk, merged with the pairwise formulas.
- NaN propagates; constant data gives NaN.
- The unbiased form needs at least 4 rows, else null.
- Decimals (decimal32..decimal256) are read as float64 (pyarrow 26 results).

Evidence: `src/compute/aggregate_stats.zig:79`, `src/compute/aggregate_basic.zig:63`, `src/compute/test_aggregate_stats.zig:"product, variance, stddev, skew and kurtosis hand-written values and options (pyarrow 26 results)"`, `src/compute/test_aggregate_stats.zig:"aggregate stats on random sliced inputs with nulls: chunked inputs and the update / merge lifecycle equal the array result"`, `src/compute/test_aggregate_stats.zig:"variance, stddev, skew and kurtosis and their hash_ forms on decimal32 / decimal64 (pyarrow 26 results)"`

## last

`last` mirrors [`first`](#first): the last valid row with `skip_nulls`, the last row (maybe null) without it.

Evidence: `src/compute/aggregate_stats.zig:82`, `src/compute/aggregate_basic.zig:63`, `src/compute/test_aggregate_stats.zig:"first, last, first_last, count_distinct and index hand-written values and options (pyarrow 26 results)"`, `src/compute/test_aggregate_stats.zig:"aggregate stats on random sliced inputs with nulls: chunked inputs and the update / merge lifecycle equal the array result"`, `src/compute/test_aggregate_stats.zig:"first, last and first_last on fixed_size_binary: nulls, skip_nulls, sliced and chunked (pyarrow 26 results)"`

## max

`max` returns the largest valid value with the input type kept.

- Types: int8..int64, uint8..uint64, float32/64; decimal128 / decimal256; and the ordered non-numeric types: the null type, bool (`false < true`), date32, date64, timestamp, time32, time64, string / large_string / binary / large_binary and fixed_size_binary (byte order), as pyarrow 26.
- Beyond pyarrow 26 (which has no kernel there): duration, string_view / binary_view and decimal32 / decimal64, through `OrderedMinMax` (the state keeps one-row slices of the extremes; byte rows compare 8-byte big-endian prefixes in blocks and the full bytes only on a tie).
- Interval types have no order: `NoMatchingKernel`, as Arrow.
- Options: `ScalarAggregateOptions` (`skip_nulls` default true, `min_count` default 1; Arrow defaults when `none`). A null row without `skip_nulls`, or fewer valid rows than `min_count`, gives a null result. With `min_count = 0` an empty input still gives null (pyarrow 26).

Evidence: `src/compute/aggregate_basic.zig:69`, `src/compute/aggregate_basic.zig:63`, `src/compute/test_aggregate_basic.zig:"aggregate hand-written values: sum, mean, min, max, min_max, count, count_all, any, all"`, `src/compute/test_aggregate_basic.zig:"aggregate oracle matches a per-element loop on random sliced inputs with nulls for every numeric type"`, `src/compute/test_aggregate_basic.zig:"aggregate chunked input with misaligned chunks and the update / merge lifecycle equal the array result"`

## mean

`mean` returns the arithmetic mean.

- Integers and floats give float64; with `min_count = 0` an empty mean is NaN.
- decimal128 / decimal256 give the input type, rounded half away from zero; an empty decimal mean is null whatever `min_count`.
- bool gives the float64 fraction of true rows; the null type gives null by the options, else 0.0 (pyarrow 26).
- Options: `ScalarAggregateOptions` (`skip_nulls` default true, `min_count` default 1; Arrow defaults when `none`). A null row without `skip_nulls`, or fewer valid rows than `min_count`, gives a null result.

Evidence: `src/compute/aggregate_basic.zig:67`, `src/compute/aggregate_basic.zig:63`, `src/compute/test_aggregate_basic.zig:"aggregate hand-written values: sum, mean, min, max, min_max, count, count_all, any, all"`, `src/compute/test_aggregate_basic.zig:"aggregate oracle matches a per-element loop on random sliced inputs with nulls for every numeric type"`, `src/compute/test_aggregate_basic.zig:"aggregate chunked input with misaligned chunks and the update / merge lifecycle equal the array result"`

## min

`min` returns the smallest valid value with the input type kept.

- Types: int8..int64, uint8..uint64, float32/64; decimal128 / decimal256; and the ordered non-numeric types: the null type, bool (`false < true`), date32, date64, timestamp, time32, time64, string / large_string / binary / large_binary and fixed_size_binary (byte order), as pyarrow 26.
- Beyond pyarrow 26 (which has no kernel there): duration, string_view / binary_view and decimal32 / decimal64, through `OrderedMinMax` (the state keeps one-row slices of the extremes; byte rows compare 8-byte big-endian prefixes in blocks and the full bytes only on a tie).
- Interval types have no order: `NoMatchingKernel`, as Arrow.
- Options: `ScalarAggregateOptions` (`skip_nulls` default true, `min_count` default 1; Arrow defaults when `none`). A null row without `skip_nulls`, or fewer valid rows than `min_count`, gives a null result. With `min_count = 0` an empty input still gives null (pyarrow 26).

Evidence: `src/compute/aggregate_basic.zig:68`, `src/compute/aggregate_basic.zig:63`, `src/compute/test_aggregate_basic.zig:"aggregate hand-written values: sum, mean, min, max, min_max, count, count_all, any, all"`, `src/compute/test_aggregate_basic.zig:"aggregate oracle matches a per-element loop on random sliced inputs with nulls for every numeric type"`, `src/compute/test_aggregate_basic.zig:"aggregate chunked input with misaligned chunks and the update / merge lifecycle equal the array result"`

## min_max

`min_max` returns a `struct<min, max>` scalar, with the type and option rules of [`min`](#min) and [`max`](#max). The struct type is stored with its payload array (`dense.structArray`) and `resolveResultTypeAlloc` builds it in an arena.

- The null fields of a null result reuse a null input row. For an empty fixed_size_binary input wider than 16 MiB there is no such row, and the call is `UnsupportedType` rather than an allocation the input does not justify (`src/compute/aggregate_basic.zig:1073`; backed by code only, no test).

Evidence: `src/compute/aggregate_basic.zig:70`, `src/compute/aggregate_basic.zig:63`, `src/compute/test_aggregate_basic.zig:"aggregate hand-written values: sum, mean, min, max, min_max, count, count_all, any, all"`, `src/compute/test_aggregate_basic.zig:"aggregate chunked input with misaligned chunks and the update / merge lifecycle equal the array result"`, `src/compute/test_aggregate_basic.zig:"aggregate ScalarAggregateOptions and CountOptions: skip_nulls, min_count, count modes, nullary count_all (Arrow C++ results)"`

## pivot_wider

`pivot_wider` builds one struct scalar from (key, value) rows: one field per entry of `PivotWiderOptions.key_names` (names parsed as the key type for integer keys), null when the key never appears.

- `unexpected_key_behavior`: `ignore` (default) drops keys not in `key_names`; `raise` makes them `InvalidInput`.
- A null key, or two non-null values for one key, is `InvalidInput`; duplicate key names are `InvalidOptions`.

Evidence: `src/compute/pivot.zig:377`, `src/compute/aggregate_basic.zig:63`, `src/compute/test_pivot.zig:"pivot_wider hand-written values, null and unexpected keys, key types (pyarrow 26 results)"`

## product

`product` multiplies the valid rows: int -> int64, uint -> uint64, float -> float64.

- With `min_count = 0` an empty product is 1.
- Integer overflow follows `overflow_mode` as for [`sum`](#sum) (`wrapping` gives Arrow C++'s wrapped product).
- Decimals (decimal128 / decimal256) keep the input type: every step is rescaled to the input scale, rounding half away from zero (Arrow's `MultiplyTraits`); every chunk is a product of its own, then merged (pyarrow 26: chunks [1.01], [1.01, 1.29] give 1.31, one chunk 1.32). Past the precision `overflow_mode` decides (`checked`: `Overflow`; `wrapping`: Arrow C++'s 128 / 256-bit wrapped value; `saturating`: the largest value of the sign). A negative scale is `NotImplemented` (undefined in Arrow).
- bool gives uint64 (0 when a valid row is false, else 1); the null type gives int64 null, or 1 with `min_count = 0` (pyarrow 26).
- Options: `ScalarAggregateOptions` (`skip_nulls` default true, `min_count` default 1; Arrow defaults when `none`). A null row without `skip_nulls`, or fewer valid rows than `min_count`, gives a null result.

Evidence: `src/compute/aggregate_stats.zig:65`, `src/compute/aggregate_basic.zig:63`, `src/compute/test_aggregate_stats.zig:"product, variance, stddev, skew and kurtosis hand-written values and options (pyarrow 26 results)"`, `src/compute/test_aggregate_stats.zig:"aggregate stats on random sliced inputs with nulls: chunked inputs and the update / merge lifecycle equal the array result"`, `src/compute/test_aggregate_stats.zig:"aggregate stats errors: options, types, empty and all-null inputs, malformed input"`

## skew

`skew` returns the sample skewness as float64, with the rules of [`kurtosis`](#kurtosis); the unbiased form needs at least 3 rows.

Evidence: `src/compute/aggregate_stats.zig:78`, `src/compute/aggregate_basic.zig:63`, `src/compute/test_aggregate_stats.zig:"product, variance, stddev, skew and kurtosis hand-written values and options (pyarrow 26 results)"`, `src/compute/test_aggregate_stats.zig:"aggregate stats on random sliced inputs with nulls: chunked inputs and the update / merge lifecycle equal the array result"`, `src/compute/test_aggregate_stats.zig:"aggregate stats errors: options, types, empty and all-null inputs, malformed input"`

## stddev

`stddev` is the square root of [`variance`](#variance), with the same options and rules.

Evidence: `src/compute/aggregate_stats.zig:77`, `src/compute/aggregate_basic.zig:63`, `src/compute/test_aggregate_stats.zig:"product, variance, stddev, skew and kurtosis hand-written values and options (pyarrow 26 results)"`, `src/compute/test_aggregate_stats.zig:"aggregate stats on random sliced inputs with nulls: chunked inputs and the update / merge lifecycle equal the array result"`, `src/compute/test_aggregate_stats.zig:"variance, stddev, skew and kurtosis and their hash_ forms on decimal32 / decimal64 (pyarrow 26 results)"`

## sum

`sum` adds the valid rows.

- Integer sums are exact until finalize, which applies `overflow_mode` (`checked`: `Overflow`; `wrapping`: Arrow C++'s result; `saturating`: clamp).
- With `min_count = 0` an empty sum is 0.
- decimal128 / decimal256 give precision 38 / 76 at the input scale, exact until finalize; `checked` rejects a sum past the precision, `wrapping` keeps the low bits as Arrow, `saturating` clamps.
- bool gives the uint64 count of true rows; the null type gives int64 null by the options, else 0 (pyarrow 26).
- Options: `ScalarAggregateOptions` (`skip_nulls` default true, `min_count` default 1; Arrow defaults when `none`). A null row without `skip_nulls`, or fewer valid rows than `min_count`, gives a null result.

Evidence: `src/compute/aggregate_basic.zig:66`, `src/compute/aggregate_basic.zig:63`, `src/compute/test_aggregate_basic.zig:"aggregate hand-written values: sum, mean, min, max, min_max, count, count_all, any, all"`, `src/compute/test_aggregate_basic.zig:"aggregate oracle matches a per-element loop on random sliced inputs with nulls for every numeric type"`, `src/compute/test_aggregate_basic.zig:"aggregate chunked input with misaligned chunks and the update / merge lifecycle equal the array result"`

## tdigest

`tdigest` approximates quantiles with Arrow's merging t-digest: buffered input, arcsine scale function, every chunk a digest of its own, merged as Arrow's executor merges batch states.

- Options: `TDigestOptions` (`q` default [0.5], each in [0, 1]; `delta` default 100; `buffer_size` default 500; `skip_nulls`; `min_count` default 0).
- Equal to pyarrow 26 within 1e-15 relative on 5000 rows with nulls and NaN; the tests allow 1e-9 since another libm's asin / sin may move a centroid boundary.
- NaN is skipped but counts as a valid row (Arrow).
- A q outside [0, 1] (or NaN) gives a NaN quantile, as Arrow (pyarrow 26: `tdigest(q=[-0.1, 0.5])` -> `[nan, ...]`); `delta` past 2^20 and `buffer_size` past 2^24 are `InvalidOptions` (a zarrow bound against huge allocations).

Evidence: `src/compute/tdigest.zig:534`, `src/compute/tdigest.zig:531`, `src/compute/test_tdigest.zig:"tdigest and approximate_median hand-written values, options and types (pyarrow 26 results)"`, `src/compute/test_tdigest.zig:"tdigest on 5000 rows with nulls and NaN: array, chunked (merged per chunk) and small delta follow pyarrow 26"`

## variance

`variance` returns the variance as float64. Options: `VarianceOptions` (`ddof` default 0, `skip_nulls` default true, `min_count` default 0).

- Two-pass moments per chunk merged with the pairwise formulas; NaN propagates.
- `count <= ddof` gives null.
- Decimals (decimal32..decimal256) are read as float64 (pyarrow 26 results).

Evidence: `src/compute/aggregate_stats.zig:76`, `src/compute/aggregate_basic.zig:63`, `src/compute/test_aggregate_stats.zig:"product, variance, stddev, skew and kurtosis hand-written values and options (pyarrow 26 results)"`, `src/compute/test_aggregate_stats.zig:"aggregate stats on random sliced inputs with nulls: chunked inputs and the update / merge lifecycle equal the array result"`, `src/compute/test_aggregate_stats.zig:"aggregate stats errors: options, types, empty and all-null inputs, malformed input"`

## mode

`mode` (vector) returns the `n` most common values ordered by count descending, then by value; NaN sorts last. Floats compare by value with every NaN one value and -0.0 equal to 0.0, unlike the hash kernels, as pyarrow 26 (`mode([nan, -nan, 1, 1], n=3)` gives `[1, nan]` with counts `[2, 2]`; `src/compute/test_associative.zig:"hash-based kernels compare floats by bits: NaN sign and payload, signed zeros; nulls, sliced, chunked (pyarrow 26 results)"`). The result is empty when a null row is seen without `skip_nulls` or fewer valid rows than `min_count` remain. Options: `ModeOptions` (`n` default 1, `skip_nulls` default true, `min_count` default 0).

Evidence: `src/compute/aggregate_stats.zig:91`, `src/compute/aggregate_basic.zig:63`, `src/compute/test_aggregate_stats.zig:"mode and quantile hand-written values and options (pyarrow 26 results)"`, `src/compute/test_aggregate_stats.zig:"aggregate stats errors: options, types, empty and all-null inputs, malformed input"`

## quantile

`quantile` (vector) returns one value per entry of `QuantileOptions.q` (default [0.5]) with every interpolation and Arrow's formulas: `linear` (default) and `midpoint` give float64 (`f * hi + (1 - f) * lo`, `lo / 2 + hi / 2`), `lower`, `higher` and `nearest` (ties to the even index) the input type. Each q must be in [0, 1]. Nulls and NaN are ignored; a row is null when no value is left, fewer than `min_count` remain, or a null row is seen without `skip_nulls`. Options also carry `skip_nulls` (default true) and `min_count` (default 0).

Evidence: `src/compute/aggregate_stats.zig:95`, `src/compute/aggregate_basic.zig:63`, `src/compute/test_aggregate_stats.zig:"mode and quantile hand-written values and options (pyarrow 26 results)"`, `src/compute/test_aggregate_stats.zig:"aggregate stats errors: options, types, empty and all-null inputs, malformed input"`
