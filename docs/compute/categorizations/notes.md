# Categorizations: function notes

Up: [function matrix](matrix.md) · [compute support matrix](../../compute-support-matrix.md)

Arrow C++'s "Categorizations" functions, all unary scalar functions with a bool result, registered
by `src/compute/is_null.zig:38`. The null predicates (`is_null`, `is_valid`, `true_unless_null`) take
every type and read Arrow's logical nulls (`src/compute/is_null.zig:210`); the float predicates
(`is_nan`, `is_finite`, `is_inf`) propagate input nulls (`src/compute/is_null.zig:77`). Each accepts an
array, a chunked array (chunks kept) or a scalar. Evidence: `src/compute/test_is_null.zig:"is_null family oracle matches isNull per row on random sliced inputs of several types"`, `src/compute/test_is_null.zig:"is_null family on chunked input with misaligned chunks keeps the chunks"`.

## Summary

| Function | Arity | Input types | Output type | Options |
|---|---|---|---|---|
| [`is_null`](#is_null) | Unary | any | bool | `NullOptions` |
| [`is_valid`](#is_valid) | Unary | any | bool | none |
| [`is_nan`](#is_nan) | Unary | floats, integers, decimals, duration, null | bool | none |
| [`is_finite`](#is_finite) | Unary | floats, integers, decimals, duration, null | bool | none |
| [`is_inf`](#is_inf) | Unary | floats, integers, decimals, duration, null | bool | none |
| [`true_unless_null`](#true_unless_null) | Unary | any | bool | none |

## is_null

True where the row is null, false elsewhere; the result itself is never null. Logical nulls: the top-level validity bitmap, every row of the null type, for a union the selected
child's row and for a run-end encoded array its run's value (recursively). pyarrow 26:
`is_null(ree [1, 1, null, 4, 4])` is `[F, F, T, F, F]`.

Options: `NullOptions.nan_is_null` (default false); when true a NaN of float16, float32 or float64
also counts as null. Any other options value is `InvalidOptions`. A null scalar gives a true scalar.

Evidence: `src/compute/is_null.zig:44`, `src/compute/options.zig:314`, `src/compute/test_is_null.zig:"is_null, is_valid and true_unless_null hand-written values for primitive, string, null and dictionary arrays"`, `src/compute/test_is_null.zig:"is_null, is_valid, true_unless_null and count on run-end encoded and union arrays use the children's nulls: sliced, chunked (pyarrow 26 results)"`, `src/compute/test_is_null.zig:"is_null family on scalars: a null scalar is null, the result is never null for is_null and is_valid"`, `src/compute/test_is_null.zig:"is_nan, is_finite, is_inf and is_null with nan_is_null on floats, integers, decimals and the null type (Arrow C++ results)"`, `src/compute/test_is_null.zig:"is_null family errors: arity and options"`

## is_valid

The negation of `is_null` without options: true where the row is valid. Never null. Logical nulls: the top-level validity bitmap, every row of the null type, for a union the selected
child's row and for a run-end encoded array its run's value (recursively). pyarrow 26:
`is_null(ree [1, 1, null, 4, 4])` is `[F, F, T, F, F]`.

Evidence: `src/compute/is_null.zig:220`, `src/compute/is_null.zig:210`, `src/compute/test_is_null.zig:"is_null, is_valid and true_unless_null hand-written values for primitive, string, null and dictionary arrays"`, `src/compute/test_is_null.zig:"is_null, is_valid, true_unless_null and count on run-end encoded and union arrays use the children's nulls: sliced, chunked (pyarrow 26 results)"`, `src/compute/test_is_null.zig:"is_null family on scalars: a null scalar is null, the result is never null for is_null and is_valid"`, `src/compute/test_is_null.zig:"is_null family on empty and all-null inputs"`

## is_nan

True where a floating-point value is NaN. Accepted types: float16, float32, float64 (IEEE 754), every integer type and the decimal types
(always finite, never NaN or infinite), duration (likewise; pyarrow 26), and the null type (all
null). A null input row gives a null output row (pyarrow 26 results).

Evidence: `src/compute/is_null.zig:86`, `src/compute/is_null.zig:77`, `src/compute/test_is_null.zig:"is_nan, is_finite, is_inf and is_null with nan_is_null on floats, integers, decimals and the null type (Arrow C++ results)"`, `src/compute/test_is_null.zig:"is_finite, is_inf and is_nan on duration: finite, nulls kept, sliced and chunked (pyarrow 26 results)"`, `src/compute/test_cast.zig:"float16: cast to and from numbers and text, promotion to float32 in arithmetic and comparisons, is_nan / is_inf / is_finite (pyarrow 26 results)"`

## is_finite

True where the value is neither NaN nor infinite; integers, decimals and durations are always
true. Accepted types: float16, float32, float64 (IEEE 754), every integer type and the decimal types
(always finite, never NaN or infinite), duration (likewise; pyarrow 26), and the null type (all
null). A null input row gives a null output row (pyarrow 26 results).

Evidence: `src/compute/is_null.zig:87`, `src/compute/is_null.zig:77`, `src/compute/test_is_null.zig:"is_nan, is_finite, is_inf and is_null with nan_is_null on floats, integers, decimals and the null type (Arrow C++ results)"`, `src/compute/test_is_null.zig:"is_finite, is_inf and is_nan on duration: finite, nulls kept, sliced and chunked (pyarrow 26 results)"`, `src/compute/test_cast.zig:"float16: cast to and from numbers and text, promotion to float32 in arithmetic and comparisons, is_nan / is_inf / is_finite (pyarrow 26 results)"`

## is_inf

True where a floating-point value is positive or negative infinity. Accepted types: float16, float32, float64 (IEEE 754), every integer type and the decimal types
(always finite, never NaN or infinite), duration (likewise; pyarrow 26), and the null type (all
null). A null input row gives a null output row (pyarrow 26 results).

Evidence: `src/compute/is_null.zig:88`, `src/compute/is_null.zig:77`, `src/compute/test_is_null.zig:"is_nan, is_finite, is_inf and is_null with nan_is_null on floats, integers, decimals and the null type (Arrow C++ results)"`, `src/compute/test_is_null.zig:"is_finite, is_inf and is_nan on duration: finite, nulls kept, sliced and chunked (pyarrow 26 results)"`, `src/compute/test_cast.zig:"float16: cast to and from numbers and text, promotion to float32 in arithmetic and comparisons, is_nan / is_inf / is_finite (pyarrow 26 results)"`

## true_unless_null

True where the row is valid and null where it is null (the input's logical nulls carried to the
output, unlike `is_valid` which is never null). Logical nulls: the top-level validity bitmap, every row of the null type, for a union the selected
child's row and for a run-end encoded array its run's value (recursively). pyarrow 26:
`is_null(ree [1, 1, null, 4, 4])` is `[F, F, T, F, F]`.

Evidence: `src/compute/is_null.zig:221`, `src/compute/is_null.zig:210`, `src/compute/test_is_null.zig:"is_null, is_valid and true_unless_null hand-written values for primitive, string, null and dictionary arrays"`, `src/compute/test_is_null.zig:"is_null, is_valid, true_unless_null and count on run-end encoded and union arrays use the children's nulls: sliced, chunked (pyarrow 26 results)"`, `src/compute/test_is_null.zig:"is_null family on chunked input with misaligned chunks keeps the chunks"`, `src/compute/test_is_null.zig:"is_null family on empty and all-null inputs"`
