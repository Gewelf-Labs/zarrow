# Selections: function notes

Up: [function matrix](matrix.md) · [compute support matrix](../../compute-support-matrix.md)

Arrow C++'s "Selections" vector functions, registered by `src/compute/selection.zig:64`. Kernels exist
for every layout (nested types, unions, run-end encoded, views, dictionaries). A chunked argument gives
a chunked result and an empty input keeps its type (pyarrow 26 results). The Zig helpers
`datumTake` / `datumTakeNullable` (`src/compute/datum_permutation.zig:1115`), `datumFilter` /
`datumFilterChunkAware` (`src/compute/datum_permutation.zig:1230`) and `datumSelect` /
`datumSelectNullable` (`src/compute/datum_permutation.zig:1153`) serve arrays and chunked arrays, and
record batches and tables column by column (a table stays chunked). Evidence: `src/compute/test_datum.zig:"compute chunked take and filter run the array kernels per chunk and match the flat result"`, `src/compute/test_datum.zig:"compute datumFilterSelectionIndices and datumFilterChunkAware stay consistent for array and chunked"`.

## Summary

| Function | Arity | Input types | Output type | Options |
|---|---|---|---|---|
| [`take`](#take) | Binary | any values; integer indices | values type | `TakeOptions` |
| [`array_take`](#array_take) | Binary | any array values; integer indices | values type | `TakeOptions` |
| [`filter`](#filter) | Binary | any values; bool predicate | values type | `FilterOptions` |
| [`array_filter`](#array_filter) | Binary | array or chunked values; bool predicate | values type | `FilterOptions` |
| [`drop_null`](#drop_null) | Unary | any array, chunked, record batch or table | input type | none |
| [`inverse_permutation`](#inverse_permutation) | Unary | signed integer indices | signed integer | `InversePermutationOptions` |
| [`scatter`](#scatter) | Binary | any values; signed integer indices | values type | `ScatterOptions` |

## take

The rows of the values at the given indices (Arrow's meta function). Values may be an array, a
chunked array, a record batch or a table (every column taken; a table stays chunked). Indices of any integer type, array or chunked. A null index gives a null row. A negative index or
one past the values is `InvalidInput` (Arrow's IndexError), also with `TakeOptions.boundscheck`
false (default true), where Arrow skips the check; zarrow always checks. A take with no index keeps
the input's type and layout (an empty slice).

Evidence: `src/compute/selection.zig:109`, `src/compute/selection.zig:1027`, `src/compute/datum_permutation.zig:1268`, `src/compute/test_selection.zig:"take, array_take, filter and array_filter on nulls, sliced, chunked and every index type (pyarrow 26 results)"`, `src/compute/test_selection.zig:"take and filter errors: index bounds, index and predicate types, lengths, datum kinds, options"`, `src/compute/test_tabular.zig:"take, filter, drop_null, slice and column selection on record batches and tables (pyarrow 26 results)"`, `src/compute/test_datum.zig:"compute datumTake nested kernels match the per-row oracle on sliced inputs with null indices"`

## array_take

As `take`, but the values must be an array: a chunked argument is `NoMatchingKernel` (Arrow:
"Vector kernel cannot execute chunkwise"), a record batch or table `NotImplemented`. Indices of any integer type, array or chunked. A null index gives a null row. A negative index or
one past the values is `InvalidInput` (Arrow's IndexError), also with `TakeOptions.boundscheck`
false (default true), where Arrow skips the check; zarrow always checks. A take with no index keeps
the input's type and layout (an empty slice).

Evidence: `src/compute/selection.zig:109`, `src/compute/selection.zig:1027`, `src/compute/test_selection.zig:"take, array_take, filter and array_filter on nulls, sliced, chunked and every index type (pyarrow 26 results)"`, `src/compute/test_selection.zig:"take and filter errors: index bounds, index and predicate types, lengths, datum kinds, options"`, `src/compute/test_tabular.zig:"functions without tabular kernels reject record batches and tables"`

## filter

The rows whose predicate is true (Arrow's meta function), on arrays, chunked arrays, record batches
and tables (every column filtered). The predicate is a bool array or chunked array of the values' length (`InvalidInput` otherwise);
a scalar predicate is `NoMatchingKernel` (Arrow: "Filter should be array-like").
`FilterOptions.drop_nulls` (default true) drops rows whose predicate is null; false keeps them as
null rows (Arrow's `emit_null`).

Evidence: `src/compute/selection.zig:116`, `src/compute/selection.zig:1090`, `src/compute/options.zig:143`, `src/compute/test_selection.zig:"take, array_take, filter and array_filter on nulls, sliced, chunked and every index type (pyarrow 26 results)"`, `src/compute/test_selection.zig:"take and filter errors: index bounds, index and predicate types, lengths, datum kinds, options"`, `src/compute/test_tabular.zig:"take, filter, drop_null, slice and column selection on record batches and tables (pyarrow 26 results)"`

## array_filter

As `filter`, for arrays and chunked arrays only; a record batch or table is `NotImplemented`.
The predicate is a bool array or chunked array of the values' length (`InvalidInput` otherwise);
a scalar predicate is `NoMatchingKernel` (Arrow: "Filter should be array-like").
`FilterOptions.drop_nulls` (default true) drops rows whose predicate is null; false keeps them as
null rows (Arrow's `emit_null`).

Evidence: `src/compute/selection.zig:116`, `src/compute/selection.zig:1090`, `src/compute/test_selection.zig:"take, array_take, filter and array_filter on nulls, sliced, chunked and every index type (pyarrow 26 results)"`, `src/compute/test_selection.zig:"take and filter errors: index bounds, index and predicate types, lengths, datum kinds, options"`, `src/compute/test_tabular.zig:"functions without tabular kernels reject record batches and tables"`

## drop_null

The valid rows of an array or chunked array. On a record batch or table, drops the rows with a
null in any column, as pyarrow 26. A scalar is `NoMatchingKernel`, as Arrow.

Evidence: `src/compute/selection.zig:81`, `src/compute/selection.zig:382`, `src/compute/selection.zig:429`, `src/compute/test_selection.zig:"drop_null, fill_null_forward and fill_null_backward (pyarrow 26 results), chunked across boundaries"`, `src/compute/test_tabular.zig:"take, filter, drop_null, slice and column selection on record batches and tables (pyarrow 26 results)"`

## inverse_permutation

For each position `i` of the output, the input row whose value is `i`. Indices must be signed
integers (unsigned: `NoMatchingKernel`, as Arrow), each bounds-checked before any write (out of
range is `InvalidInput`, Arrow's IndexError). A later row wins on a repeated index, a null index is
skipped, and a position no index names is null.

Options: `max_index` (default -1): the output has `max_index + 1` rows, or as many rows as indices
when negative; capped at max(indices, 2^24) rows (`InvalidOptions`: no allocation the input does not
justify). `output_type` (default: the index type) must be a signed integer type holding the input
length; another type is `InvalidOptions`, a length past its maximum `InvalidInput`. Chunked input
gives chunked output (pyarrow 26 results).

Evidence: `src/compute/selection.zig:103`, `src/compute/selection.zig:823`, `src/compute/selection.zig:747`, `src/compute/test_selection.zig:"replace_with_mask, scatter and inverse_permutation hand-written values (pyarrow 26 results)"`

## scatter

Writes value row `i` to output position `indices[i]`; values and indices must have equal length
(`InvalidInput`). Takes every layout the take kernels handle. Index rules as
`inverse_permutation`: signed integers only (unsigned `NoMatchingKernel`), bounds-checked before any
write (`InvalidInput`), a later row wins, a null index is skipped, an unnamed position is null.
`ScatterOptions.max_index` (default -1) sizes the output the same way, with the same cap. Chunked
input gives chunked output (pyarrow 26 results).

Evidence: `src/compute/selection.zig:95`, `src/compute/selection.zig:792`, `src/compute/selection.zig:747`, `src/compute/test_selection.zig:"replace_with_mask, scatter and inverse_permutation hand-written values (pyarrow 26 results)"`
