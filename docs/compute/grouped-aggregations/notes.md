# Grouped aggregations: function notes

Up: [function matrix](matrix.md) · [compute support matrix](../../compute-support-matrix.md)

Hash aggregate functions (kind `hash_aggregate`) take values plus uint32 group ids and return one row per group, as Arrow's `hash_*` kernels behind `group_by`. `Grouper` maps one or more key columns (numbers, bool, temporal, intervals, decimals, fixed_size_binary, string / binary layouts) to dense ids in first-seen order: a null key is its own group, floats compare by their bits, as pyarrow 26 hashes them: NaN with another sign or payload is another value, -0.0 and 0.0 two (`count_distinct([0.0, -0.0, nan, -nan])` is 4); `src/compute/test_associative.zig:"hash-based kernels compare floats by bits: NaN sign and payload, signed zeros; nulls, sliced, chunked (pyarrow 26 results)"`. Dictionary keys group by index (a null index and an index to a null entry are two groups), and every batch of a key column must carry the dictionary of the first, else `NotImplemented`. `groupBy` returns the key columns then one column per aggregate. A one-shot call needs every id below its row count. Evidence: `src/compute/hash_aggregate.zig:497`, `src/compute/test_hash_aggregate.zig:"hash aggregate on empty and all-null inputs"`.

## Summary

| Function | Arity | Input types | Output type | Options |
|---|---|---|---|---|
| [`hash_all`](#hash_all) | Unary + ids | bool | bool | `ScalarAggregateOptions` |
| [`hash_any`](#hash_any) | Unary + ids | bool | bool | `ScalarAggregateOptions` |
| [`hash_approximate_median`](#hash_approximate_median) | Unary + ids | numbers, decimal128 / decimal256 | float64 | `ScalarAggregateOptions` |
| [`hash_count`](#hash_count) | Unary + ids | any type | int64 | `CountOptions` |
| [`hash_count_all`](#hash_count_all) | ids only | group ids | int64 | none |
| [`hash_count_distinct`](#hash_count_distinct) | Unary + ids | every key type | int64 | `CountOptions` |
| [`hash_distinct`](#hash_distinct) | Unary + ids | every key type | `list<T>` | `CountOptions` |
| [`hash_first`](#hash_first) | Unary + ids | every type the take kernels handle | input type | `ScalarAggregateOptions` |
| [`hash_first_last`](#hash_first_last) | Unary + ids | every type the take kernels handle | `struct<first, last>` | `ScalarAggregateOptions` |
| [`hash_kurtosis`](#hash_kurtosis) | Unary + ids | numbers, decimal32..decimal256 | float64 | `SkewOptions` |
| [`hash_last`](#hash_last) | Unary + ids | every type the take kernels handle | input type | `ScalarAggregateOptions` |
| [`hash_list`](#hash_list) | Unary + ids | every type the take kernels handle | `list<T>` | none |
| [`hash_max`](#hash_max) | Unary + ids | numbers, decimals, ordered non-numeric types | input type | `ScalarAggregateOptions` |
| [`hash_mean`](#hash_mean) | Unary + ids | int8..int64, uint8..uint64, float32/64, decimal128 / decimal256 | float64; decimals keep the input type | `ScalarAggregateOptions` |
| [`hash_min`](#hash_min) | Unary + ids | numbers, decimals, ordered non-numeric types | input type | `ScalarAggregateOptions` |
| [`hash_min_max`](#hash_min_max) | Unary + ids | numbers, decimals, ordered non-numeric types | `struct<min, max>` | `ScalarAggregateOptions` |
| [`hash_one`](#hash_one) | Unary + ids | every type the take kernels handle | input type | none |
| [`hash_pivot_wider`](#hash_pivot_wider) | Binary + ids | keys and values as `pivot_wider` | struct, one row per group | `PivotWiderOptions` |
| [`hash_product`](#hash_product) | Unary + ids | numbers, decimal128 / decimal256 | int64 / uint64 / float64; decimals keep the input type | `ScalarAggregateOptions` |
| [`hash_skew`](#hash_skew) | Unary + ids | numbers, decimal32..decimal256 | float64 | `SkewOptions` |
| [`hash_stddev`](#hash_stddev) | Unary + ids | numbers, decimal32..decimal256 | float64 | `VarianceOptions` |
| [`hash_sum`](#hash_sum) | Unary + ids | int8..int64, uint8..uint64, float32/64, decimal128 / decimal256 | int64 / uint64 / float64; decimals at precision 38 / 76 | `ScalarAggregateOptions` |
| [`hash_tdigest`](#hash_tdigest) | Unary + ids | numbers, decimal128 / decimal256 | `fixed_size_list<double>[q.len]` | `TDigestOptions` |
| [`hash_variance`](#hash_variance) | Unary + ids | numbers, decimal32..decimal256 | float64 | `VarianceOptions` |

## hash_all

Per-group [`all`](../aggregations/notes.md#all); Kleene logic without `skip_nulls`. Options: `ScalarAggregateOptions` (`skip_nulls`, `min_count`): a group with a null row and no `skip_nulls`, or fewer valid rows than `min_count`, is null. Options of another class are `InvalidOptions`.

Evidence: `src/compute/hash_aggregate.zig:516`, `src/compute/hash_aggregate.zig:497`, `src/compute/test_hash_aggregate.zig:"hash aggregate hand-written values: group by one and two keys, every hash_ function (Arrow C++ results)"`, `src/compute/test_hash_aggregate.zig:"hash aggregate on empty and all-null inputs"`, `src/compute/test_hash_aggregate.zig:"hash aggregates honour ScalarAggregateOptions and CountOptions (pyarrow 26 group_by results)"`

## hash_any

Per-group [`any`](../aggregations/notes.md#any); Kleene logic without `skip_nulls`. Options: `ScalarAggregateOptions` (`skip_nulls`, `min_count`): a group with a null row and no `skip_nulls`, or fewer valid rows than `min_count`, is null. Options of another class are `InvalidOptions`.

Evidence: `src/compute/hash_aggregate.zig:515`, `src/compute/hash_aggregate.zig:497`, `src/compute/test_hash_aggregate.zig:"hash aggregate hand-written values: group by one and two keys, every hash_ function (Arrow C++ results)"`, `src/compute/test_hash_aggregate.zig:"hash aggregates honour ScalarAggregateOptions and CountOptions (pyarrow 26 group_by results)"`

## hash_approximate_median

Per-group [`approximate_median`](../aggregations/notes.md#approximate_median), the t-digest with q = 0.5.

Evidence: `src/compute/tdigest.zig:538`, `src/compute/hash_aggregate.zig:497`, `src/compute/test_tdigest.zig:"hash_tdigest and hash_approximate_median per group (pyarrow 26 group_by, use_threads=False)"`

## hash_count

Per-group [`count`](../aggregations/notes.md#count) with `CountOptions.mode` (`only_valid`, `only_null`, `all`). Dictionary values are counted by index validity.

Evidence: `src/compute/hash_aggregate.zig:513`, `src/compute/hash_aggregate.zig:497`, `src/compute/test_dictionary.zig:"dictionary group-by keys and hash_count, hash_count_distinct, hash_distinct values group by index (pyarrow 26 group_by)"`, `src/compute/test_hash_aggregate.zig:"hash aggregate hand-written values: group by one and two keys, every hash_ function (Arrow C++ results)"`, `src/compute/test_hash_aggregate.zig:"hash aggregate oracle matches a per-group reference on random sliced keys and values with nulls"`

## hash_count_all

Counts the rows of each group. Its only argument is the group ids, as Arrow's `hash_count_all` (a nullary hash function: the group ids are its one argument; pyarrow 26 `group_by("k").aggregate([([], "count_all")])` on keys `[1, 1, 2]` gives `[2, 1]`). `groupBy` takes it with a null `input`.

Evidence: `src/compute/hash_aggregate.zig:514`, `src/compute/hash_aggregate.zig:497`, `src/compute/test_hash_aggregate.zig:"hash aggregate hand-written values: group by one and two keys, every hash_ function (Arrow C++ results)"`, `src/compute/test_hash_aggregate.zig:"hash aggregate oracle matches a per-group reference on random sliced keys and values with nulls"`, `src/compute/test_hash_aggregate.zig:"hash aggregate errors: group ids, key types, lengths, overflow and malformed input"`

## hash_count_distinct

Distinct values per group; a null is one distinct value, counted by `CountOptions.mode`. Dictionary values are taken by index validity.

Evidence: `src/compute/hash_aggregate.zig:517`, `src/compute/hash_aggregate.zig:497`, `src/compute/test_dictionary.zig:"dictionary group-by keys and hash_count, hash_count_distinct, hash_distinct values group by index (pyarrow 26 group_by)"`, `src/compute/test_hash_aggregate.zig:"hash aggregate hand-written values: group by one and two keys, every hash_ function (Arrow C++ results)"`, `src/compute/test_hash_aggregate.zig:"hash aggregate oracle matches a per-group reference on random sliced keys and values with nulls"`

## hash_distinct

The distinct values of each group as `list<T>` in first-seen order; a null is one value, kept or dropped by `CountOptions.mode`.

Evidence: `src/compute/hash_aggregate.zig:518`, `src/compute/hash_aggregate.zig:497`, `src/compute/test_dictionary.zig:"dictionary group-by keys and hash_count, hash_count_distinct, hash_distinct values group by index (pyarrow 26 group_by)"`, `src/compute/test_hash_aggregate.zig:"hash aggregate hand-written values: group by one and two keys, every hash_ function (Arrow C++ results)"`, `src/compute/test_hash_aggregate.zig:"hash aggregate on chunked inputs with misaligned chunks and the update / merge lifecycle equal the array result"`

## hash_first

First row of each group: the first valid row with `skip_nulls`, else the first row. `min_count` is not applied, as pyarrow 26 `group_by`.

Evidence: `src/compute/hash_aggregate.zig:1331`, `src/compute/hash_aggregate.zig:497`, `src/compute/test_aggregate_stats.zig:"hash_product, hash_variance, hash_stddev, hash_skew, hash_kurtosis, hash_first, hash_last, hash_first_last, hash_one and hash_list (pyarrow 26 group_by results)"`, `src/compute/test_aggregate_stats.zig:"aggregate stats errors: options, types, empty and all-null inputs, malformed input"`

## hash_first_last

[`hash_first`](#hash_first) and [`hash_last`](#hash_last) in one struct per group; `min_count` is not applied.

Evidence: `src/compute/hash_aggregate.zig:1333`, `src/compute/hash_aggregate.zig:497`, `src/compute/test_aggregate_stats.zig:"hash_product, hash_variance, hash_stddev, hash_skew, hash_kurtosis, hash_first, hash_last, hash_first_last, hash_one and hash_list (pyarrow 26 group_by results)"`, `src/compute/test_aggregate_stats.zig:"aggregate stats errors: options, types, empty and all-null inputs, malformed input"`

## hash_kurtosis

Per-group [`kurtosis`](../aggregations/notes.md#kurtosis) with `SkewOptions`.

Evidence: `src/compute/hash_aggregate.zig:1324`, `src/compute/hash_aggregate.zig:497`, `src/compute/test_aggregate_stats.zig:"hash_product, hash_variance, hash_stddev, hash_skew, hash_kurtosis, hash_first, hash_last, hash_first_last, hash_one and hash_list (pyarrow 26 group_by results)"`

## hash_last

Mirror of [`hash_first`](#hash_first); `min_count` is not applied.

Evidence: `src/compute/hash_aggregate.zig:1332`, `src/compute/hash_aggregate.zig:497`, `src/compute/test_aggregate_stats.zig:"hash_product, hash_variance, hash_stddev, hash_skew, hash_kurtosis, hash_first, hash_last, hash_first_last, hash_one and hash_list (pyarrow 26 group_by results)"`

## hash_list

Every row of each group, nulls included, in row order, as `list<T>`.

Evidence: `src/compute/hash_aggregate.zig:1335`, `src/compute/hash_aggregate.zig:497`, `src/compute/test_aggregate_stats.zig:"hash_product, hash_variance, hash_stddev, hash_skew, hash_kurtosis, hash_first, hash_last, hash_first_last, hash_one and hash_list (pyarrow 26 group_by results)"`, `src/compute/test_aggregate_stats.zig:"aggregate stats errors: options, types, empty and all-null inputs, malformed input"`

## hash_max

`hash_max` is [`max`](../aggregations/notes.md#max) per group, on int8..int64, uint8..uint64, float32/64, decimal128 / decimal256 and the ordered non-numeric types of `min_max` (null, bool, temporal, decimal32 / decimal64, string / binary layouts, fixed_size_binary). For the non-numeric types `OrderedGroups` retains the value batches, each group keeps the (batch, row) of its extremes, and one take builds each output (pyarrow 26 `group_by` results). Options: `ScalarAggregateOptions`: a group with a null row and no `skip_nulls` is null; `min_count` is not applied, as pyarrow 26 `group_by` (`max` with `min_count=2` on a group `[1, null]` gives 1, while the scalar `max` gives null). Options of another class are `InvalidOptions`.

Evidence: `src/compute/hash_aggregate.zig:503`, `src/compute/hash_aggregate.zig:497`, `src/compute/test_aggregate_basic.zig:"hash_min, hash_max and hash_min_max on ordered non-numeric types: strings, bool, dates, fixed_size_binary, chunked and lifecycle (pyarrow 26 group_by results)"`, `src/compute/test_hash_aggregate.zig:"hash aggregate hand-written values: group by one and two keys, every hash_ function (Arrow C++ results)"`, `src/compute/test_hash_aggregate.zig:"hash aggregate oracle matches a per-group reference on random sliced keys and values with nulls"`, `src/compute/test_hash_aggregate.zig:"hash aggregates honour ScalarAggregateOptions and CountOptions (pyarrow 26 group_by results)"`

## hash_mean

Per-group [`mean`](../aggregations/notes.md#mean); with `min_count = 0` an empty group's mean is NaN. Options: `ScalarAggregateOptions` (`skip_nulls`, `min_count`): a group with a null row and no `skip_nulls`, or fewer valid rows than `min_count`, is null. Options of another class are `InvalidOptions`.

Evidence: `src/compute/hash_aggregate.zig:501`, `src/compute/hash_aggregate.zig:497`, `src/compute/test_hash_aggregate.zig:"hash aggregate hand-written values: group by one and two keys, every hash_ function (Arrow C++ results)"`, `src/compute/test_hash_aggregate.zig:"hash aggregate oracle matches a per-group reference on random sliced keys and values with nulls"`, `src/compute/test_hash_aggregate.zig:"hash aggregate on empty and all-null inputs"`

## hash_min

`hash_min` is [`min`](../aggregations/notes.md#min) per group, on int8..int64, uint8..uint64, float32/64, decimal128 / decimal256 and the ordered non-numeric types of `min_max` (null, bool, temporal, decimal32 / decimal64, string / binary layouts, fixed_size_binary). For the non-numeric types `OrderedGroups` retains the value batches, each group keeps the (batch, row) of its extremes, and one take builds each output (pyarrow 26 `group_by` results). Options: `ScalarAggregateOptions`: a group with a null row and no `skip_nulls` is null; `min_count` is not applied, as pyarrow 26 `group_by` (`max` with `min_count=2` on a group `[1, null]` gives 1, while the scalar `max` gives null). Options of another class are `InvalidOptions`.

Evidence: `src/compute/hash_aggregate.zig:502`, `src/compute/hash_aggregate.zig:497`, `src/compute/test_aggregate_basic.zig:"hash_min, hash_max and hash_min_max on ordered non-numeric types: strings, bool, dates, fixed_size_binary, chunked and lifecycle (pyarrow 26 group_by results)"`, `src/compute/test_hash_aggregate.zig:"hash aggregate hand-written values: group by one and two keys, every hash_ function (Arrow C++ results)"`, `src/compute/test_hash_aggregate.zig:"hash aggregate oracle matches a per-group reference on random sliced keys and values with nulls"`, `src/compute/test_hash_aggregate.zig:"hash aggregates honour ScalarAggregateOptions and CountOptions (pyarrow 26 group_by results)"`

## hash_min_max

Per-group `struct<min, max>` of the input type, stored with the array; type rules of [`hash_min`](#hash_min). Options: `ScalarAggregateOptions`: a group with a null row and no `skip_nulls` is null; `min_count` is not applied, as pyarrow 26 `group_by` (`max` with `min_count=2` on a group `[1, null]` gives 1, while the scalar `max` gives null). Options of another class are `InvalidOptions`.

Evidence: `src/compute/hash_aggregate.zig:504`, `src/compute/hash_aggregate.zig:497`, `src/compute/test_aggregate_basic.zig:"hash_min, hash_max and hash_min_max on ordered non-numeric types: strings, bool, dates, fixed_size_binary, chunked and lifecycle (pyarrow 26 group_by results)"`, `src/compute/test_aggregate_stats.zig:"decimal min_max and hash_min_max keep the input precision and scale (pyarrow 26 results)"`, `src/compute/test_hash_aggregate.zig:"hash aggregate hand-written values: group by one and two keys, every hash_ function (Arrow C++ results)"`, `src/compute/test_hash_aggregate.zig:"hash aggregates honour ScalarAggregateOptions and CountOptions (pyarrow 26 group_by results)"`

## hash_one

The first non-null value of each group. Arrow returns an arbitrary one, so results agree with pyarrow only up to that choice.

Evidence: `src/compute/hash_aggregate.zig:1334`, `src/compute/hash_aggregate.zig:497`, `src/compute/test_aggregate_stats.zig:"hash_product, hash_variance, hash_stddev, hash_skew, hash_kurtosis, hash_first, hash_last, hash_first_last, hash_one and hash_list (pyarrow 26 group_by results)"`, `src/compute/test_aggregate_stats.zig:"aggregate stats errors: options, types, empty and all-null inputs, malformed input"`

## hash_pivot_wider

Per-group [`pivot_wider`](../aggregations/notes.md#pivot_wider). Call it with the ids of a `Grouper`: `groupBy` passes one input per aggregate, so it does not fit `groupBy`.

Evidence: `src/compute/pivot.zig:377`, `src/compute/hash_aggregate.zig:497`, `src/compute/test_pivot.zig:"hash_pivot_wider per group and the update / merge lifecycle (pyarrow 26 group_by)"`

## hash_product

Per-group [`product`](../aggregations/notes.md#product), with its overflow rules. Decimal products step row by row with rescaling, as Arrow's grouped product. Options: `ScalarAggregateOptions` (`skip_nulls`, `min_count`): a group with a null row and no `skip_nulls`, or fewer valid rows than `min_count`, is null. Options of another class are `InvalidOptions`.

Evidence: `src/compute/hash_aggregate.zig:1315`, `src/compute/hash_aggregate.zig:497`, `src/compute/test_aggregate_stats.zig:"hash_product, hash_variance, hash_stddev, hash_skew, hash_kurtosis, hash_first, hash_last, hash_first_last, hash_one and hash_list (pyarrow 26 group_by results)"`, `src/compute/test_aggregate_stats.zig:"decimal product and hash_product: input type, Arrow rounding per step and per chunk, overflow modes (pyarrow 26 results)"`

## hash_skew

Per-group [`skew`](../aggregations/notes.md#skew) with `SkewOptions`.

Evidence: `src/compute/hash_aggregate.zig:1323`, `src/compute/hash_aggregate.zig:497`, `src/compute/test_aggregate_stats.zig:"hash_product, hash_variance, hash_stddev, hash_skew, hash_kurtosis, hash_first, hash_last, hash_first_last, hash_one and hash_list (pyarrow 26 group_by results)"`

## hash_stddev

Per-group [`stddev`](../aggregations/notes.md#stddev) with `VarianceOptions`.

Evidence: `src/compute/hash_aggregate.zig:1322`, `src/compute/hash_aggregate.zig:497`, `src/compute/test_aggregate_stats.zig:"hash_product, hash_variance, hash_stddev, hash_skew, hash_kurtosis, hash_first, hash_last, hash_first_last, hash_one and hash_list (pyarrow 26 group_by results)"`, `src/compute/test_aggregate_stats.zig:"variance, stddev, skew and kurtosis and their hash_ forms on decimal32 / decimal64 (pyarrow 26 results)"`

## hash_sum

Per-group [`sum`](../aggregations/notes.md#sum), `overflow_mode` for integer sums; with `min_count = 0` an empty group sums to 0. Options: `ScalarAggregateOptions` (`skip_nulls`, `min_count`): a group with a null row and no `skip_nulls`, or fewer valid rows than `min_count`, is null. Options of another class are `InvalidOptions`.

Evidence: `src/compute/hash_aggregate.zig:500`, `src/compute/hash_aggregate.zig:497`, `src/compute/test_dictionary.zig:"dictionary group-by keys and hash_count, hash_count_distinct, hash_distinct values group by index (pyarrow 26 group_by)"`, `src/compute/test_hash_aggregate.zig:"hash aggregate hand-written values: group by one and two keys, every hash_ function (Arrow C++ results)"`, `src/compute/test_hash_aggregate.zig:"hash aggregate oracle matches a per-group reference on random sliced keys and values with nulls"`

## hash_tdigest

Per-group [`tdigest`](../aggregations/notes.md#tdigest). A group without a result holds null items in a valid list, as Arrow; a q outside [0, 1] gives NaN items in every group with values (pyarrow 26).

Evidence: `src/compute/tdigest.zig:537`, `src/compute/hash_aggregate.zig:497`, `src/compute/test_tdigest.zig:"hash_tdigest and hash_approximate_median per group (pyarrow 26 group_by, use_threads=False)"`

## hash_variance

Per-group [`variance`](../aggregations/notes.md#variance) with `VarianceOptions`.

Evidence: `src/compute/hash_aggregate.zig:1321`, `src/compute/hash_aggregate.zig:497`, `src/compute/test_aggregate_stats.zig:"hash_product, hash_variance, hash_stddev, hash_skew, hash_kurtosis, hash_first, hash_last, hash_first_last, hash_one and hash_list (pyarrow 26 group_by results)"`, `src/compute/test_aggregate_stats.zig:"decimal min_max and hash_min_max keep the input precision and scale (pyarrow 26 results)"`, `src/compute/test_aggregate_stats.zig:"variance, stddev, skew and kurtosis and their hash_ forms on decimal32 / decimal64 (pyarrow 26 results)"`
