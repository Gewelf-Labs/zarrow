# Replace functions: function notes

Up: [function matrix](matrix.md) · [compute support matrix](../../compute-support-matrix.md)

Vector functions that replace rows: null filling and masked replacement, as Arrow's replace functions; results are checked against pyarrow 26. Arrow's `fill_null` is a pyarrow helper, not a compute function (absent from pyarrow 26 `pc.list_functions()`), so zarrow does not register it.

## Summary

| Function | Arity | Input types | Output type | Options |
|---|---|---|---|---|
| [`fill_null_forward`](#fill_null_forward) | Unary | every type the take kernels handle | input type | none |
| [`fill_null_backward`](#fill_null_backward) | Unary | every type the take kernels handle | input type | none |
| [`replace_with_mask`](#replace_with_mask) | Ternary | values: any; mask: bool; replacements: values' type | values' type | none |

## fill_null_forward

`fill_null_forward`: each null row takes the previous valid value, across chunk boundaries as Arrow. Leading nulls stay null; an input with no null comes back as is; a scalar is returned unchanged.

Evidence: `src/compute/selection.zig:85`, `src/compute/selection.zig:64`, `src/compute/test_selection.zig:"drop_null, fill_null_forward and fill_null_backward (pyarrow 26 results), chunked across boundaries"`, `src/compute/test_selection.zig:"selection errors: case_when null struct rows, choose index out of range, scalars, arity and malformed input"`

## fill_null_backward

`fill_null_backward`: each null row takes the next valid value, across chunk boundaries. Trailing nulls stay null; otherwise as [`fill_null_forward`](#fill_null_forward).

Evidence: `src/compute/selection.zig:85`, `src/compute/selection.zig:64`, `src/compute/test_selection.zig:"drop_null, fill_null_forward and fill_null_backward (pyarrow 26 results), chunked across boundaries"`

## replace_with_mask

`replace_with_mask(values, mask, replacements)`: each true mask row takes the next replacement (a scalar replacement: always it), a null mask row gives null, a false row keeps its value.

- values: array or chunked; mask: a bool array of the values' length or a bool scalar; replacements: an array or scalar of the values' type, else `NoMatchingKernel`.
- Fewer replacements than true rows, a chunked mask or a chunked replacement is `InvalidInput` (Arrow: Invalid).
- Fixed-width values go through one pass over 8-row mask bytes; every other layout, also nested and view layouts (which Arrow lacks), through one take.

Evidence: `src/compute/selection.zig:91`, `src/compute/selection.zig:64`, `src/compute/test_selection.zig:"replace_with_mask, scatter and inverse_permutation hand-written values (pyarrow 26 results)"`
