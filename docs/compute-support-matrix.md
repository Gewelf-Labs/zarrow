# Compute support matrix

Arrow C++ documents its compute functions in
[Compute Functions](https://arrow.apache.org/docs/cpp/compute.html) and shows the calling model in the
[compute tutorial](https://arrow.apache.org/docs/cpp/tutorials/compute_tutorial.html). This page maps every
category of that catalog to what `zarrow` provides today. It is a status page, not a roadmap.

Legend:

- **yes**: a public operation exists in this repository with the same semantics, and a test exercises it.
- **partial**: a building block exists (types, helpers, accessors) but no callable function, or the
  operation exists without a test.
- **no**: nothing in this repository. Kernels in this state belong downstream (see
  [compute-api.md](compute-api.md), section 1).

## Evidence rule

Every `yes` or `partial` row cites its evidence: at least one **source** reference
(a backticked `path` ending in `.zig`, a colon and the line number of the implementing `pub fn` or type) and at least one **test** reference
(the same path, a colon and the exact test name in double quotes). A claim without a test is at most `partial`. CI runs
`tools/check_doc_evidence.py`, which fails when a cited file, line or test name does not exist or a claim
row lacks either kind of reference. The rule itself is stated in [`CLAUDE.md`](../CLAUDE.md).

The framework itself (function registry, signatures, options, exec context, aggregate lifecycle, chunk-aware
iterators) is complete and is what downstream kernel libraries build on. `FunctionRegistry.init` starts
empty: there is no built-in kernel catalog, and `arrow::compute::Initialize()` has no counterpart.

## Calling model

| Arrow C++ | zarrow | Status | Source | Tests |
|---|---|---|---|---|
| `CallFunction(name, {args}, &options)` | `ExecContext.invoke` (`invokeScalar` / `invokeVector` / `invokeAggregate`) | yes | `src/compute/registry.zig:447`, `src/compute/registry.zig:297` | `src/compute/test_registry.zig:"compute registry registers and invokes scalar kernel"`, `src/compute/test_registry.zig:"compute invoke helpers cover vector and aggregate kernels"` |
| `Datum` (Array, ChunkedArray, Scalar, RecordBatch, Table) | `Datum` with `array`, `chunked`, `scalar` | partial (no batch / table datums) | `src/compute/core.zig:128` | `src/compute/test_datum.zig:"compute datum chunked variant retain and release"`, `src/compute/test_datum.zig:"compute datum helpers and shared type predicates"` |
| `FunctionOptions` subclasses | `Options` tagged union: `cast`, `arithmetic`, `filter`, `sort`, `custom` | yes | `src/compute/options.zig:83`, `src/compute/options.zig:7` | `src/compute/test_options.zig:"compute sort options defaults and tag dispatch"`, `src/compute/test_options.zig:"compute kernel signature result type inference and typed options"` |
| `Function::Kind` scalar / vector / scalar aggregate / hash aggregate | `FunctionKind` scalar / vector / aggregate | partial (no hash aggregate) | `src/compute/core.zig:13` | `src/compute/test_registry.zig:"compute registry keeps separate indices per function kind"` |
| `ExecContext` (memory pool, thread pool) | `ExecContext` (`allocator`, `arena_allocator`, `threads`, `overflow_mode`, `safe_cast`) | yes | `src/compute/registry.zig:402`, `src/compute/registry.zig:391` | `src/compute/test_datum.zig:"compute exec context config and scalar payload duplication"` |
| Aggregate kernel state (`KernelState` init / consume / merge / finalize) | `AggregateLifecycle`, `AggregateSession`, `beginAggregate` | yes | `src/compute/registry.zig:23`, `src/compute/registry.zig:42`, `src/compute/registry.zig:341` | `src/compute/test_registry.zig:"compute aggregate lifecycle session supports init/update/merge/finalize"`, `src/compute/test_registry.zig:"compute beginAggregate returns MissingLifecycle when not provided"` |
| Kernel signature matching and dispatch diagnostics | `KernelSignature.matches`, `explainMismatch`, `FunctionRegistry.resolveKernel`, `explainResolveKernelFailure` | yes | `src/compute/signature.zig:157`, `src/compute/signature.zig:173`, `src/compute/registry.zig:183`, `src/compute/registry.zig:217` | `src/compute/test_registry.zig:"compute resolveKernel prefers exact then range then at_least arity"`, `src/compute/test_options.zig:"compute explain helpers provide readable diagnostics"` |
| chunked execution (`ExecBatch` / `ExecSpan`) | `UnaryExecChunkIterator`, `BinaryExecChunkIterator`, `NaryExecChunkIterator` | yes | `src/compute/iterators.zig:237`, `src/compute/iterators.zig:269`, `src/compute/iterators.zig:308` | `src/compute/test_iterators.zig:"compute execution helpers align chunked chunks for binary kernels"`, `src/compute/test_iterators.zig:"compute nary execution helper aligns misaligned chunk boundaries"`, `src/compute/test_iterators.zig:"compute unary execution helper propagates nulls over chunked input"` |
| `NullHandling::INTERSECTION` | `unaryNullPropagates` / `binaryNullPropagates` / `naryNullPropagates`, `dense.DenseBinaryOutput`, `dense.validityBits`, `dense.laneMask` | yes | `src/compute/iterators.zig:43`, `src/compute/dense.zig:31`, `src/compute/dense.zig:111`, `src/compute/dense.zig:119` | `src/compute/test_iterators.zig:"compute nary execution helper propagates nulls across all inputs"`, `src/compute/dense.zig:"DenseBinaryOutput validity equals per-row null propagation (sliced, nullable, empty inputs)"`, `src/compute/dense.zig:"laneMask matches isNull per row (random, sliced, all-null, no-null, tail block)"` |
| `arrow::compute::Initialize()` | none; downstream registers its kernels into a `FunctionRegistry` | no (by design) | | |

## Function catalog

| Arrow category | Arrow functions (abridged) | zarrow | Notes | Source | Tests |
|---|---|---|---|---|---|
| Aggregations | all, any, count, count_distinct, first, last, index, max, mean, min, min_max, mode, product, quantile, stddev, sum, tdigest, variance, approximate_median, kurtosis, skew, pivot_wider | no | `AggregateLifecycle` is the hook (see Calling model). `benchmarks/compute_benchmark.zig` carries reference `sum_int64`, `sum_float64`, `min_max_int32` kernels that are not part of the library. | | |
| Grouped aggregations | hash_* | no | No hash aggregate kind in `FunctionKind`. | | |
| Arithmetic | add, subtract, multiply, divide, negate, abs, power, sqrt, sign, exp, remainder (+ `_checked`) | partial | `ArithmeticOptions`, `OverflowMode` (checked / wrapping / saturating), `arithmeticDivI64`, `intCastOrInvalidCast`, dense validity helpers. Reference `add_int32` lives in the benchmark only. | `src/compute/options.zig:31`, `src/compute/registry.zig:384`, `src/compute/iterators.zig:375`, `src/compute/iterators.zig:370` | `src/compute/test_options.zig:"compute arithmetic kernel maps divide-by-zero to DivideByZero"`, `src/compute/dense.zig:"DenseBinaryOutput validity equals per-row null propagation (sliced, nullable, empty inputs)"` |
| Bit-wise | bit_wise_and/or/xor/not, shift_left/right | no | | | |
| Rounding, logarithmic, trigonometric, hyperbolic | ceil, floor, round, trunc, ln, log2, log10, sin, cos, tan, sinh, ... | no | | | |
| Comparisons | equal, not_equal, less, less_equal, greater, greater_equal, min/max_element_wise | no | Reference `less_int32` in the benchmark only. | | |
| Logical | and, or, xor, invert, and_kleene, or_kleene, and_not | no | | | |
| String predicates | ascii_is_*, utf8_is_*, string_is_ascii | no | `validateFull` validates UTF-8 of string arrays (see Operations outside the catalog), not a function. | | |
| String transforms, padding, trimming, splitting, slicing, joining, regex | ascii_upper, utf8_lower, utf8_length, binary_length, replace_substring, ..._trim, split_pattern, binary_slice, binary_join, extract_regex | no | Builders and per-value accessors only. | | |
| Containment | is_in, index_in, match_substring, starts_with, ends_with, count_substring, find_substring | no | | | |
| Categorizations | is_null, is_valid, is_nan, is_finite, is_inf, true_unless_null | partial | Per-element `isNull` / `isValid` on arrays, scalars and `ValidityBitmap`, and `laneMask` for 8-row blocks; no function returning a boolean array. | `src/array/array_data.zig:118`, `src/array/array_data.zig:129`, `src/compute/dense.zig:119` | `src/bitmap.zig:"validity bitmap reads values"`, `src/compute/dense.zig:"laneMask matches isNull per row (random, sliced, all-null, no-null, tail block)"` |
| Selecting / multiplexing | if_else, case_when, coalesce, choose | no | | | |
| Structural transforms | list_element, list_flatten, list_parent_indices, list_slice, list_value_length, make_struct, struct_field, map_lookup | partial | `datumListValueAt`, `datumLargeListValueAt`, `datumFixedSizeListValueAt`, `datumStructField`, `datumElementArrayAt` give per-row access; no array-producing kernels. | `src/compute/datum_accessors.zig:179`, `src/compute/datum_accessors.zig:206`, `src/compute/datum_accessors.zig:233`, `src/compute/datum_accessors.zig:264`, `src/compute/datum_accessors.zig:330` | `src/compute/test_datum.zig:"compute datum list extraction supports array chunked and scalar inputs"`, `src/compute/test_datum.zig:"compute datumStructField supports scalar struct bool fields"` |
| Conversions | cast, strptime, strftime, *_temporal rounding | partial | `CastOptions` (`safe`, `to_type`) and `ExecConfig.safe_cast`; no cast kernel. | `src/compute/options.zig:23`, `src/compute/registry.zig:391` | `src/compute/test_options.zig:"compute cast kernel succeeds with valid conversion"`, `src/compute/test_options.zig:"compute cast kernel maps invalid conversion to InvalidCast"` |
| Temporal | year, month, day, hour, ..., *_between, assume_timezone, local_timestamp | no | Temporal types exist in `DataType`; no functions. | | |
| Random | random | no | | | |
| Cumulative, statistical, pairwise | cumulative_sum/prod/max/min/mean, winsorize, pairwise_diff | no | | | |
| Associative transforms | dictionary_encode, unique, value_counts | partial | `dictionaryEncode` (int32 indices, first-seen order, fixed-width and string/binary values). `unique` / `value_counts` absent. | `src/concat_array_refs.zig:624` | `src/compute/test_datum.zig:"dictionaryEncode maps rows to distinct values in first-seen order; null rows stay null"`, `src/compute/test_datum.zig:"dictionaryEncode fast paths: every fixed width and large strings, past the reserved capacity"` |
| Selections | take, array_take, filter, array_filter, drop_null, inverse_permutation, scatter | yes (take, filter, select) | `datumTake` / `datumTakeNullable`, `datumFilter` / `datumFilterChunkAware`, `datumSelect` / `datumSelectNullable` for arrays and chunked arrays, direct kernels for every layout (nested, unions, run-end encoded, views, dictionaries). `drop_null`, `inverse_permutation`, `scatter` absent. | `src/compute/datum_permutation.zig:1113`, `src/compute/datum_permutation.zig:1134`, `src/compute/datum_permutation.zig:1147`, `src/compute/datum_permutation.zig:1178`, `src/compute/datum_permutation.zig:1212`, `src/compute/datum_permutation.zig:1223` | `src/compute/test_datum.zig:"compute datumTake keeps chunked output and matches array logical result on misaligned boundaries"`, `src/compute/test_datum.zig:"compute datumFilter supports scalar array chunked and fixed_size_list null alignment"`, `src/compute/test_datum.zig:"compute datumSelectNullable and datumBuildNullLikeWithAllocator support null output rows"`, `src/compute/test_datum.zig:"compute datumTake direct gather for string and list on sliced input with null indices"`, `src/compute/test_datum.zig:"compute datumFilterSelectionIndices and datumFilterChunkAware stay consistent for array and chunked"` |
| Sorts and partitions | sort_indices, array_sort_indices, rank, partition_nth_indices, select_k_unstable, search_sorted | partial | `SortOptions` (order, null and NaN placement, stable) only. | `src/compute/options.zig:63` | `src/compute/test_options.zig:"compute sort options defaults and tag dispatch"`, `src/compute/test_options.zig:"compute kernel signature options_check validates sort options"` |
| Replace functions | fill_null_forward, fill_null_backward, replace_with_mask | no | | | |
| Indices | indices_nonzero | no | | | |

## Operations outside the catalog that zarrow provides

| Operation | Where | Arrow equivalent | Status | Source | Tests |
|---|---|---|---|---|---|
| Concatenate arrays (N-ary, every layout, dictionary unification) | `concatArrayRefs` | `arrow::Concatenate` | yes | `src/concat_array_refs.zig:19` | `src/concat_array_refs.zig:"concatArrayRefs supports null bool and fixed-width with sliced inputs"`, `src/compute/test_datum.zig:"concatArrayRefs unifies parts with different dictionaries"` |
| Slice | `ArrayRef.slice`, `ChunkedArray.slice`, `RecordBatch.slice`, `Table.slice` | `Array::Slice` | yes | `src/array/array_ref.zig:109`, `src/chunked_array.zig:109`, `src/record_batch.zig:133`, `src/table.zig:94` | `src/array/array_ref.zig:"array ref slice retains buffers"`, `src/chunked_array.zig:"chunked array slice spans chunk boundary"`, `src/record_batch.zig:"record batch slice returns sliced columns"`, `src/table.zig:"table slice keeps schema and rows"` |
| Column projection | `Table.select` | `Table::SelectColumns` | partial (no test) | `src/table.zig:127` | `src/table.zig:"table slice keeps schema and rows"` (covers the table type, not `select`) |
| Validation | `validateLayout`, `validateFull` (offsets, child lengths, dictionary indices, UTF-8) | `Array::Validate`, `ValidateFull` | yes | `src/array/array_data.zig:490`, `src/array/array_data.zig:734` | `src/array/array_data.zig:"array data validateFull checks UTF-8 of non-null string values like a per-value check"`, `src/array/array_data.zig:"array data validateFull checks null count matches bitmap"`, `src/array/array_data.zig:"array data validateFull rejects dictionary index beyond dictionary length"` |
| Null-like / empty-like construction | `datumBuildNullLike`, `datumBuildEmptyLike` | `MakeArrayOfNull`, `MakeEmptyArray` | yes | `src/compute/datum_builders.zig:263`, `src/compute/datum_builders.zig:273` | `src/compute/test_datum.zig:"compute datumBuildEmptyLike and datumSliceEmpty preserve nested layout"`, `src/compute/test_datum.zig:"compute datumSelectNullable and datumBuildNullLikeWithAllocator support null output rows"` |
| Builder buffer reuse without aliasing live arrays | `SharedBuffer.reclaim`, builders' `finishReset` | none | yes | `src/buffer.zig:120` | `src/buffer.zig:"shared buffer reclaim returns the allocation only to the last handle"`, `src/array/test_builder_reuse.zig:"string/binary/view builders reuse finishReset buffers without aliasing live arrays"`, `src/array/test_builder_reuse.zig:"primitive builder reuses finishReset values buffer without aliasing live arrays"` |
| Fuzzing of every selection / concat / encode kernel on decoded IPC input | fuzz target `compute` | none | yes | `fuzz/fuzz.zig:213` | `fuzz/fuzz.zig:"fuzz compute"` |
| PyArrow compute alignment runner | `zig build compute-compat-check` | none | partial (runner, no Zig test block) | `build.zig:271` | `build.zig:"compute-compat-check"` |

## Keeping this page current

Update the matching row and its evidence when a kernel lands here or when a helper becomes a callable
function. `python -I tools/check_doc_evidence.py` must pass before the change is committed.
