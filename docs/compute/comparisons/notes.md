# Comparisons: function notes

Up: [function matrix](matrix.md) · [compute support matrix](../../compute-support-matrix.md)

The Arrow C++ "Comparisons" in `comparison.zig`: six binary comparisons with a bool result and the
variadic `min_element_wise` / `max_element_wise`. All work element-wise on array, chunked or scalar
operands; operands of different types are unified as Arrow's `DispatchBest` (dictionaries are
decoded, a null-type operand takes the other type, numbers go to their common type, float16 to
float32). Evidence: `src/compute/comparison.zig:68`, `src/compute/registry.zig:253`,
`src/compute/test_comparison.zig:"implicit casts decode dictionaries and give null-type operands the other type (pyarrow 26)"`,
`src/compute/test_cast.zig:"float16: cast to and from numbers and text, promotion to float32 in arithmetic and comparisons, is_nan / is_inf / is_finite (pyarrow 26 results)"`.

## Summary

| Function | Arity | Input types | Output type | Options |
|---|---|---|---|---|
| [`equal`](#equal) | Binary | numeric, bool, string / binary layouts, fixed_size_binary, temporal, decimal | bool | none |
| [`not_equal`](#not_equal) | Binary | as `equal` | bool | none |
| [`less`](#less) | Binary | as `equal` | bool | none |
| [`less_equal`](#less_equal) | Binary | as `equal` | bool | none |
| [`greater`](#greater) | Binary | as `equal` | bool | none |
| [`greater_equal`](#greater_equal) | Binary | as `equal` | bool | none |
| [`max_element_wise`](#max_element_wise) | Varargs | numeric, temporal, decimal, fixed_size_binary, string, binary | first operand type (after unification) | `ElementWiseAggregateOptions` |
| [`min_element_wise`](#min_element_wise) | Varargs | as `max_element_wise` | as `max_element_wise` | `ElementWiseAggregateOptions` |

## equal

`x == y` per row, bool output (a bool scalar for scalar inputs), null intersection. Supported:
int8..int64, uint8..uint64, float32 / float64, bool, string / large_string / string_view, binary /
large_binary / binary_view, fixed_size_binary (same width), date32, date64, timestamp, time32,
time64, duration and decimal32..decimal256.

- Floats follow IEEE 754: NaN is unequal to everything, itself included.
- Strings and binaries compare their bytes. Two different string / binary / fixed_size_binary
  layouts compare in place with no cast; this includes a view with another layout, which Arrow has
  no kernel for. Two fixed_size_binary of different widths stay `NoMatchingKernel`, as Arrow.
- Temporal values and decimals compare their stored integers. Temporal operands of different types
  go to Arrow's `CommonTemporal` (the finer unit, date64 over date32, a timestamp over a date; a
  value past the finer unit is `InvalidCast` when the cast is safe). Timestamps of one unit compare
  instants across timezones; a naive with a zoned timestamp has no kernel (`NoMatchingKernel`;
  Arrow: Invalid); timestamps of different units in different timezones have no common type.
- Decimals of any widths, precisions and scales, and a decimal with an integer, compare by exact
  value (the operand of the smaller scale is scaled up in 256-bit arithmetic; a product past that
  range is beyond every value of the other side). Arrow rejects a common precision past 38 / 76 or
  a negative scale; zarrow compares those too. A decimal with a float goes to float64.

Evidence: `src/compute/comparison.zig:212`, `src/compute/comparison.zig:427`, `src/compute/comparison.zig:335`, `src/compute/comparison.zig:377`,
`src/compute/test_comparison.zig:"comparison hand-written values for integers, floats with NaN, bools and strings of every layout"`,
`src/compute/test_comparison.zig:"comparison oracle matches a per-element loop on random sliced inputs with nulls for every type"`,
`src/compute/test_comparison.zig:"comparison chunked inputs with misaligned chunks give a chunked result equal to the array result"`,
`src/compute/test_comparison.zig:"comparison scalar broadcast, scalar-scalar and null scalars"`,
`src/compute/test_comparison.zig:"comparisons unify mixed string layouts, temporal units and decimal scales (pyarrow 26 results)"`

## not_equal

`x != y` per row, with the types and unification of [`equal`](#equal). For a NaN operand the
result is true.

Evidence: `src/compute/comparison.zig:212`, `src/compute/test_comparison.zig:"not_equal, less, less_equal, greater, greater_equal by name: nulls, sliced, chunked, scalar (pyarrow 26 results)"`,
`src/compute/test_comparison.zig:"comparison hand-written values for integers, floats with NaN, bools and strings of every layout"`,
`src/compute/test_comparison.zig:"comparison empty and all-null inputs"`

## less

`x < y` per row, with the types and unification of [`equal`](#equal). Bool orders `false < true`;
strings and binaries order by bytes; a NaN operand gives false. Mismatched types with no common
type, wrong arity, options or lengths are errors.

Evidence: `src/compute/comparison.zig:233`, `src/compute/comparison.zig:307`, `src/compute/test_comparison.zig:"not_equal, less, less_equal, greater, greater_equal by name: nulls, sliced, chunked, scalar (pyarrow 26 results)"`,
`src/compute/test_comparison.zig:"comparison hand-written values for integers, floats with NaN, bools and strings of every layout"`,
`src/compute/test_comparison.zig:"comparison errors: mismatched types, arity, options, lengths and malformed scalars"`

## less_equal

`x <= y` per row, as [`less`](#less).

Evidence: `src/compute/comparison.zig:212`, `src/compute/test_comparison.zig:"not_equal, less, less_equal, greater, greater_equal by name: nulls, sliced, chunked, scalar (pyarrow 26 results)"`,
`src/compute/test_comparison.zig:"comparison implicit casts and binary, fixed-size, temporal and decimal operands"`

## greater

`x > y` per row, as [`less`](#less).

Evidence: `src/compute/comparison.zig:212`, `src/compute/test_comparison.zig:"not_equal, less, less_equal, greater, greater_equal by name: nulls, sliced, chunked, scalar (pyarrow 26 results)"`,
`src/compute/test_comparison.zig:"comparison oracle matches a per-element loop on random sliced inputs with nulls for every type"`

## greater_equal

`x >= y` per row, as [`less`](#less).

Evidence: `src/compute/comparison.zig:212`, `src/compute/test_comparison.zig:"not_equal, less, less_equal, greater, greater_equal by name: nulls, sliced, chunked, scalar (pyarrow 26 results)"`,
`src/compute/test_comparison.zig:"comparison oracle matches a per-element loop on random sliced inputs with nulls for every type"`

## max_element_wise

The largest value of each row over one or more operands (arrays, chunked arrays or scalars) of one
numeric, temporal, decimal, fixed_size_binary, string or binary type. Numbers are promoted to their
common type and temporal types unified as for [`equal`](#equal) (as Arrow); decimals and
fixed_size_binary must have equal types and string layouts are not mixed (as Arrow). decimal32 /
decimal64 also have a kernel (Arrow has none).

`ElementWiseAggregateOptions.skip_nulls` (default true): null operands are left out and the row is
null only when all are; with false, any null operand makes the row null. A NaN is left out unless
every value of the row is NaN; -0.0 is smaller than 0.0 (pyarrow 26 results).

Evidence: `src/compute/comparison.zig:459`, `src/compute/comparison.zig:522`, `src/compute/options.zig:336`,
`src/compute/test_comparison.zig:"min_element_wise and max_element_wise: nulls, NaN, signed zeros, strings, promotion (Arrow C++ results)"`,
`src/compute/test_comparison.zig:"min_element_wise and max_element_wise on decimals, fixed_size_binary and mixed temporal units"`

## min_element_wise

The smallest value of each row, otherwise as [`max_element_wise`](#max_element_wise).

Evidence: `src/compute/comparison.zig:522`,
`src/compute/test_comparison.zig:"min_element_wise and max_element_wise: nulls, NaN, signed zeros, strings, promotion (Arrow C++ results)"`
