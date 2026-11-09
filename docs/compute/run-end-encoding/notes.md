# Run-end encoding: function notes

Up: [function matrix](matrix.md) · [compute support matrix](../../compute-support-matrix.md)

Vector functions over arrays and chunked arrays in `run_end.zig`, chunk by chunk as Arrow: runs do not cross chunks and empty chunks are dropped as pyarrow 26. Every input chunk passes `validateLayout`. Dictionaries, unions and extension types are `NoMatchingKernel` (Arrow rejects them too).

## Summary

| Function | Arity | Input types | Output type | Options |
|---|---|---|---|---|
| [`run_end_encode`](#run_end_encode) | Unary | null, bool, numbers, float16, temporal, intervals, decimals, fixed_size_binary, string / binary layouts and views, nested | `run_end_encoded<run_end_type, T>` | `RunEndEncodeOptions` |
| [`run_end_decode`](#run_end_decode) | Unary | run_end_encoded | the values type | none |

## run_end_encode

`run_end_encode` collapses runs of equal consecutive rows; nulls form a run of their own (`[1, 1, 2, null, null]` gives run ends `[2, 3, 5]` and values `[1, 2, null]`).

- Options: `RunEndEncodeOptions.run_end_type` int16, int32 (default) or int64. A chunk longer than the run end type holds is `InvalidInput` (Arrow's "Cannot run-end encode Arrays with more elements than the run end type can hold"). A zero-length input gives an empty run-end encoded array.
- Floats compare by bits as Arrow (0.0 and -0.0 are two runs).
- Beyond pyarrow 26: string_view / binary_view (no pyarrow kernel).
- Nested types (list, large_list, list_view, large_list_view, fixed_size_list, struct, map) compare rows by the logical equality of `integration/compare.zig` (`rowsEqual`: a null row equals a null row, a null struct is not a struct of null fields, item nulls equal item nulls); results match pyarrow 26 on lists and structs.
- Deviation: a float item inside a nested row compares by bits, so NaN items with the same bits are one run where Arrow, comparing by value, makes two.

Evidence: `src/compute/run_end.zig:44`, `src/compute/run_end.zig:43`, `src/compute/test_is_null.zig:"is_null, is_valid, true_unless_null and count on run-end encoded and union arrays use the children's nulls: sliced, chunked (pyarrow 26 results)"`, `src/compute/test_run_end.zig:"run_end_encode and run_end_decode hand-written values, nulls, sliced, every run end type (pyarrow 26 results)"`, `src/compute/test_run_end.zig:"run_end_decode of a sliced run_end_encoded array and a sliced values child (pyarrow 26 results)"`

## run_end_decode

`run_end_decode` returns the plain values of a run-end encoded array, also a sliced one or one over a sliced values child. The values child passes `validateLayout` too.

Evidence: `src/compute/run_end.zig:48`, `src/compute/run_end.zig:43`, `src/compute/test_run_end.zig:"run_end_encode and run_end_decode hand-written values, nulls, sliced, every run end type (pyarrow 26 results)"`, `src/compute/test_run_end.zig:"run_end_decode of a sliced run_end_encoded array and a sliced values child (pyarrow 26 results)"`, `src/compute/test_run_end.zig:"run_end_encode and run_end_decode on chunked inputs: per chunk, empty chunks dropped (pyarrow 26 results)"`
