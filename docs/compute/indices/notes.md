# Indices: function notes

Up: [function matrix](matrix.md) · [compute support matrix](../../compute-support-matrix.md)

The vector function that lists positions of nonzero rows, as Arrow's `indices_nonzero`.

## Summary

| Function | Arity | Input types | Output type | Options |
|---|---|---|---|---|
| [`indices_nonzero`](#indices_nonzero) | Unary | numbers, bool, decimal32..decimal256 | uint64 | none |

## indices_nonzero

`indices_nonzero` returns the uint64 positions of the valid rows that are not zero (bool: true; a float NaN is nonzero, -0.0 is zero).

- Chunked inputs are indexed across every chunk as one array; a scalar gives zero or one row (pyarrow 26 results).
- Beyond pyarrow 26: decimal32 / decimal64.

Evidence: `src/compute/selection.zig:99`, `src/compute/selection.zig:64`, `src/compute/test_selection.zig:"indices_nonzero on numbers, bool, floats and decimals, sliced, chunked and scalar (pyarrow 26 results)"`
