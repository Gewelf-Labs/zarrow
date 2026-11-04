# Compute support matrix

Arrow C++ documents its compute functions in
[Compute Functions](https://arrow.apache.org/docs/cpp/compute.html) and shows the calling model in the
[compute tutorial](https://arrow.apache.org/docs/cpp/tutorials/compute_tutorial.html). This page maps every
category of that catalog to what `zarrow` provides today. It is a status page, not a roadmap.

Legend:

- **yes**: a public operation exists in this repository with the same semantics.
- **partial**: a building block exists (types, helpers, accessors) but no callable function.
- **no**: nothing in this repository. Kernels in this state belong downstream (see
  [compute-api.md](compute-api.md), section 1).

The framework itself (function registry, signatures, options, exec context, aggregate lifecycle, chunk-aware
iterators) is complete and is what downstream kernel libraries build on. `FunctionRegistry.init` starts
empty: there is no built-in kernel catalog, and `arrow::compute::Initialize()` has no counterpart.

## Calling model

| Arrow C++ | zarrow | Status |
|---|---|---|
| `CallFunction(name, {args}, &options)` | `ExecContext.invoke(name, args, options)` (`invokeScalar` / `invokeVector` / `invokeAggregate`) | yes |
| `Datum` (Array, ChunkedArray, Scalar, RecordBatch, Table) | `Datum` with `array`, `chunked`, `scalar` | partial (no batch / table datums) |
| `FunctionOptions` subclasses | `Options` tagged union: `cast`, `arithmetic`, `filter`, `sort`, `custom` | yes |
| `Function::Kind` scalar / vector / scalar aggregate / hash aggregate | `FunctionKind` scalar / vector / aggregate | partial (no hash aggregate) |
| `ExecContext` (memory pool, thread pool) | `ExecContext` (`allocator`, `arena_allocator`, `threads`, `overflow_mode`, `safe_cast`) | yes |
| chunked execution (`ExecBatch` / `ExecSpan`) | `UnaryExecChunkIterator`, `BinaryExecChunkIterator`, `NaryExecChunkIterator` | yes |
| `arrow::compute::Initialize()` | none; downstream registers its kernels into a `FunctionRegistry` | no (by design) |

## Function catalog

| Arrow category | Arrow functions (abridged) | zarrow | Notes |
|---|---|---|---|
| Aggregations | all, any, count, count_distinct, first, last, index, max, mean, min, min_max, mode, product, quantile, stddev, sum, tdigest, variance, approximate_median, kurtosis, skew, pivot_wider | no | `AggregateLifecycle` (init/update/merge/finalize) is the hook. `benchmarks/compute_benchmark.zig` carries reference `sum_int64`, `sum_float64`, `min_max_int32` kernels that are not part of the library. |
| Grouped aggregations | hash_* | no | No hash aggregate kind in `FunctionKind`. |
| Arithmetic | add, subtract, multiply, divide, negate, abs, power, sqrt, sign, exp, remainder (+ `_checked`) | partial | `ArithmeticOptions`, `OverflowMode` (checked / wrapping / saturating), `arithmeticDivI64`, `intCastOrInvalidCast`, `dense.DenseBinaryOutput` for validity-combined loops. Reference `add_int32` lives in the benchmark only. |
| Bit-wise | bit_wise_and/or/xor/not, shift_left/right | no | |
| Rounding, logarithmic, trigonometric, hyperbolic | ceil, floor, round, trunc, ln, log2, log10, sin, cos, tan, sinh, ... | no | |
| Comparisons | equal, not_equal, less, less_equal, greater, greater_equal, min/max_element_wise | no | Reference `less_int32` in the benchmark only. |
| Logical | and, or, xor, invert, and_kleene, or_kleene, and_not | no | |
| String predicates | ascii_is_*, utf8_is_*, string_is_ascii | no | `validateFull` validates UTF-8 of string arrays (not a function). |
| String transforms, padding, trimming, splitting, slicing, joining, regex | ascii_upper, utf8_lower, utf8_length, binary_length, replace_substring, ..._trim, split_pattern, binary_slice, binary_join, extract_regex | no | Builders and per-value accessors only (`StringArray`, `StringViewArray`). |
| Containment | is_in, index_in, match_substring, starts_with, ends_with, count_substring, find_substring | no | |
| Categorizations | is_null, is_valid, is_nan, is_finite, is_inf, true_unless_null | partial | Per-element `isNull` / `isValid` on arrays, scalars and `ValidityBitmap`; no function returning a boolean array. |
| Selecting / multiplexing | if_else, case_when, coalesce, choose | no | |
| Structural transforms | list_element, list_flatten, list_parent_indices, list_slice, list_value_length, make_struct, struct_field, map_lookup | partial | `datumListValueAt`, `datumLargeListValueAt`, `datumFixedSizeListValueAt`, `datumStructField`, `datumElementArrayAt` give per-row access; no array-producing kernels. |
| Conversions | cast, strptime, strftime, *_temporal rounding | partial | `CastOptions` (`safe`, `to_type`) and `ExecConfig.safe_cast`; no cast kernel. |
| Temporal | year, month, day, hour, ..., *_between, assume_timezone, local_timestamp | no | Temporal types exist in `DataType`; no functions. |
| Random | random | no | |
| Cumulative, statistical, pairwise | cumulative_sum/prod/max/min/mean, winsorize, pairwise_diff | no | |
| Associative transforms | dictionary_encode, unique, value_counts | partial | `dictionaryEncode` (int32 indices, first-seen order, fixed-width and string/binary values). `unique` / `value_counts` absent. |
| Selections | take, array_take, filter, array_filter, drop_null, inverse_permutation, scatter | yes | `datumTake` / `datumTakeNullable`, `datumFilter` / `datumFilterChunkAware`, `datumSelect` / `datumSelectNullable` for arrays and chunked arrays, direct kernels for every layout (nested, unions, run-end encoded, views, dictionaries). `drop_null`, `inverse_permutation`, `scatter` absent. |
| Sorts and partitions | sort_indices, array_sort_indices, rank, partition_nth_indices, select_k_unstable, search_sorted | partial | `SortOptions` (order, null and NaN placement, stable) only. |
| Replace functions | fill_null_forward, fill_null_backward, replace_with_mask | no | |
| Indices | indices_nonzero | no | |

## Operations outside the catalog that zarrow provides

| Operation | Where | Arrow equivalent |
|---|---|---|
| Concatenate arrays (N-ary, every layout, dictionary unification) | `concatArrayRefs` (`src/concat_array_refs.zig`) | `arrow::Concatenate` |
| Slice | `ArrayRef.slice`, `ChunkedArray.slice`, `RecordBatch.slice`, `Table.slice` | `Array::Slice` |
| Column projection | `Table.select` | `Table::SelectColumns` |
| Validation | `validateLayout`, `validateFull` (offsets, child lengths, dictionary indices, UTF-8) | `Array::Validate`, `ValidateFull` |
| Null-like / empty-like construction | `datumBuildNullLike`, `datumBuildEmptyLike` | `MakeArrayOfNull`, `MakeEmptyArray` |

## Keeping this page current

Update the matching row when a kernel lands here or when a helper becomes a callable function. The
`compute-compat-check` build step and `src/compute/test*.zig` are the source of truth for the "yes" rows.
