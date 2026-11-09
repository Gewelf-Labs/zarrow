# Selecting / multiplexing: function notes

Up: [function matrix](matrix.md) · [compute support matrix](../../compute-support-matrix.md)

Arrow C++'s "Selecting / multiplexing" scalar functions, registered by `src/compute/selection.zig:64`
and executed by `Select` (`src/compute/selection.zig:202`); none takes options. Operands may be arrays,
chunked arrays (a chunked result) or scalars. The value arguments are first cast to a common type
(`commonValueType`, `src/compute/selection.zig:175`): numbers (`commonNumericType`), temporal types
(`commonTemporalType`), string / binary layouts (binary when one is, the large layout when one is),
decimals (the precision and scale that hold every value, decimal256 past 38 digits), and the null type
to the others' type. Values of any layout the take kernels handle (nested types and views; wider
than Arrow, which lacks views here) go through one take. Evidence: `src/compute/test_selection.zig:"selection common types: numbers, strings, temporal, decimals and the null type (pyarrow 26 results)"`, `src/compute/test_selection.zig:"selection oracle matches a per-row loop on random sliced inputs with nulls, chunked and scalar operands"`.

## Summary

| Function | Arity | Input types | Output type | Options |
|---|---|---|---|---|
| [`if_else`](#if_else) | Ternary | bool or null cond; any values | common value type | none |
| [`case_when`](#case_when) | Varargs | struct of bool; any values | common value type | none |
| [`coalesce`](#coalesce) | Varargs | any (common type) | common value type | none |
| [`choose`](#choose) | Varargs | integer, bool or null index; any values | common value type | none |

## if_else

Per row: the second argument where the condition is true, the third where it is false. A null
condition row gives a null result row. The condition may be of the null type (every row null).
When both values are of the null type the result is of the null type without reading the
condition (`registry.null_type_kernel`, `src/compute/selection.zig:234`).

Evidence: `src/compute/selection.zig:70`, `src/compute/selection.zig:234`, `src/compute/test_selection.zig:"coalesce, if_else, case_when and choose hand-written values (pyarrow 26 results)"`, `src/compute/test_selection.zig:"if_else and choose with null-type arguments and bool indices: nulls, sliced, chunked and scalars (pyarrow 26 results)"`, `src/compute/test_selection.zig:"selection oracle matches a per-row loop on random sliced inputs with nulls, chunked and scalar operands"`

## case_when

The first argument is a struct of bool conditions; then one value per condition field and an
optional extra `else` value. Each row takes the value of the first true field; a null field counts
as false; with no true field the `else` value, or null without one. A null struct row is
`InvalidInput` (Arrow raises too).

Evidence: `src/compute/selection.zig:71`, `src/compute/selection.zig:202`, `src/compute/test_selection.zig:"coalesce, if_else, case_when and choose hand-written values (pyarrow 26 results)"`, `src/compute/test_selection.zig:"selection errors: case_when null struct rows, choose index out of range, scalars, arity and malformed input"`, `src/compute/test_selection.zig:"selection oracle matches a per-row loop on random sliced inputs with nulls, chunked and scalar operands"`

## coalesce

Per row, the first valid value among the arguments, null when all are null. At least one argument.

Evidence: `src/compute/selection.zig:69`, `src/compute/selection.zig:175`, `src/compute/test_selection.zig:"coalesce, if_else, case_when and choose hand-written values (pyarrow 26 results)"`, `src/compute/test_selection.zig:"selection common types: numbers, strings, temporal, decimals and the null type (pyarrow 26 results)"`, `src/compute/test_selection.zig:"selection oracle matches a per-row loop on random sliced inputs with nulls, chunked and scalar operands"`

## choose

The first argument selects, per row, which of the following value arguments to read. The index may
be of any integer type, bool (as 0 / 1) or the null type; it is cast to int64 as Arrow's implicit
cast. A null index gives null; a valid index outside the values is `InvalidInput` (Arrow's
IndexError). When every value argument is of the null type the result is of the null type without
checking the index (pyarrow 26: `choose([5], null, null)` is `[null]`).

Evidence: `src/compute/selection.zig:71`, `src/compute/selection.zig:234`, `src/compute/test_selection.zig:"coalesce, if_else, case_when and choose hand-written values (pyarrow 26 results)"`, `src/compute/test_selection.zig:"if_else and choose with null-type arguments and bool indices: nulls, sliced, chunked and scalars (pyarrow 26 results)"`, `src/compute/test_selection.zig:"selection errors: case_when null struct rows, choose index out of range, scalars, arity and malformed input"`
