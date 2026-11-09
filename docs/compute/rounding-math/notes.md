# Rounding, logarithmic, trigonometric, hyperbolic: function notes

Up: [function matrix](matrix.md) · [compute support matrix](../../compute-support-matrix.md)

The Arrow C++ "Rounding functions" (`round.zig`) and the floating-point "Arithmetic", "Logarithmic", "Trigonometric" and "Hyperbolic trigonometric" functions (`math.zig`). The math functions run on float32 / float64; integer and decimal arguments are cast to float64 first (Arrow's `ArithmeticFloatingPointFunction`). Unchecked functions follow IEEE 754 (NaN for a domain error, -inf for a logarithm of zero); the `_checked` variants Arrow has return `InvalidInput` on a valid row outside the domain. All functions work element-wise on array, chunked or scalar inputs with null intersection, and arguments that are all of the null type give a null-type result. Evidence: `src/compute/math.zig:58`, `src/compute/round.zig:48`, `src/compute/test_math.zig:"math scalars, empty and all-null inputs, errors: types, options, arity and malformed input"`, `src/compute/test_arithmetic.zig:"arithmetic, math, rounding, bit-wise and pairwise_diff functions on null-type arguments give a null-type result: arrays, sliced, chunked, scalars (pyarrow 26)"`.

## Summary

| Function | Arity | Input types | Output type | Options |
|---|---|---|---|---|
| [`ceil`](#ceil) | Unary | float32/64, decimal128/256 (integers cast to float64) | input type (float64 for integers) | none |
| [`floor`](#floor) | Unary | as `ceil` | as `ceil` | none |
| [`trunc`](#trunc) | Unary | as `ceil` | as `ceil` | none |
| [`round`](#round) | Unary | float32/64, int8..uint64, decimal128/256 | input type | `RoundOptions` |
| [`round_binary`](#round_binary) | Binary | value as `round`; digits integer or bool (decimal values also take decimal or float digits) | input type | `RoundBinaryOptions` |
| [`round_to_multiple`](#round_to_multiple) | Unary | as `round` | input type | `RoundToMultipleOptions` |
| [`exp`](#exp) | Unary | float32 / float64 (integers and decimals cast to float64) | float32 / float64 | none |
| [`expm1`](#expm1) | Unary | float32 / float64 (integers and decimals cast to float64) | float32 / float64 | none |
| [`sqrt`](#sqrt) | Unary | float32 / float64 (integers and decimals cast to float64) | float32 / float64 | none |
| [`sqrt_checked`](#sqrt_checked) | Unary | float32 / float64 (integers and decimals cast to float64) | float32 / float64 | none |
| [`ln`](#ln) | Unary | float32 / float64 (integers and decimals cast to float64) | float32 / float64 | none |
| [`ln_checked`](#ln_checked) | Unary | float32 / float64 (integers and decimals cast to float64) | float32 / float64 | none |
| [`log2`](#log2) | Unary | float32 / float64 (integers and decimals cast to float64) | float32 / float64 | none |
| [`log2_checked`](#log2_checked) | Unary | float32 / float64 (integers and decimals cast to float64) | float32 / float64 | none |
| [`log10`](#log10) | Unary | float32 / float64 (integers and decimals cast to float64) | float32 / float64 | none |
| [`log10_checked`](#log10_checked) | Unary | float32 / float64 (integers and decimals cast to float64) | float32 / float64 | none |
| [`log1p`](#log1p) | Unary | float32 / float64 (integers and decimals cast to float64) | float32 / float64 | none |
| [`log1p_checked`](#log1p_checked) | Unary | float32 / float64 (integers and decimals cast to float64) | float32 / float64 | none |
| [`logb`](#logb) | Binary | float32 / float64 (integers and decimals cast to float64) | float32 / float64 | none |
| [`logb_checked`](#logb_checked) | Binary | float32 / float64 (integers and decimals cast to float64) | float32 / float64 | none |
| [`sin`](#sin) | Unary | float32 / float64 (integers and decimals cast to float64) | float32 / float64 | none |
| [`sin_checked`](#sin_checked) | Unary | float32 / float64 (integers and decimals cast to float64) | float32 / float64 | none |
| [`cos`](#cos) | Unary | float32 / float64 (integers and decimals cast to float64) | float32 / float64 | none |
| [`cos_checked`](#cos_checked) | Unary | float32 / float64 (integers and decimals cast to float64) | float32 / float64 | none |
| [`tan`](#tan) | Unary | float32 / float64 (integers and decimals cast to float64) | float32 / float64 | none |
| [`tan_checked`](#tan_checked) | Unary | float32 / float64 (integers and decimals cast to float64) | float32 / float64 | none |
| [`asin`](#asin) | Unary | float32 / float64 (integers and decimals cast to float64) | float32 / float64 | none |
| [`asin_checked`](#asin_checked) | Unary | float32 / float64 (integers and decimals cast to float64) | float32 / float64 | none |
| [`acos`](#acos) | Unary | float32 / float64 (integers and decimals cast to float64) | float32 / float64 | none |
| [`acos_checked`](#acos_checked) | Unary | float32 / float64 (integers and decimals cast to float64) | float32 / float64 | none |
| [`atanh`](#atanh) | Unary | float32 / float64 (integers and decimals cast to float64) | float32 / float64 | none |
| [`atanh_checked`](#atanh_checked) | Unary | float32 / float64 (integers and decimals cast to float64) | float32 / float64 | none |
| [`acosh`](#acosh) | Unary | float32 / float64 (integers and decimals cast to float64) | float32 / float64 | none |
| [`acosh_checked`](#acosh_checked) | Unary | float32 / float64 (integers and decimals cast to float64) | float32 / float64 | none |
| [`atan`](#atan) | Unary | float32 / float64 (integers and decimals cast to float64) | float32 / float64 | none |
| [`atan2`](#atan2) | Binary | float32 / float64 (integers and decimals cast to float64) | float32 / float64 | none |
| [`sinh`](#sinh) | Unary | float32 / float64 (integers and decimals cast to float64) | float32 / float64 | none |
| [`cosh`](#cosh) | Unary | float32 / float64 (integers and decimals cast to float64) | float32 / float64 | none |
| [`tanh`](#tanh) | Unary | float32 / float64 (integers and decimals cast to float64) | float32 / float64 | none |
| [`asinh`](#asinh) | Unary | float32 / float64 (integers and decimals cast to float64) | float32 / float64 | none |
| [`hypot`](#hypot) | Binary | float32 / float64 (integers and decimals cast to float64) | float32 / float64 | none |

## ceil

Smallest integral value not below x (rounding mode `up`). Floats keep their type and pass NaN and infinities through. Integers have no kernel: they are cast to float64 first, as Arrow (an int32 argument gives float64). decimal128 / decimal256 keep their type and are rounded to an integral value (mode `up`); a result past the precision is `Overflow`. Null handling is intersection; sliced, chunked and scalar inputs; null-type arguments give a null-type result.

Evidence: `src/compute/round.zig:48`, `src/compute/round.zig:237`, `src/compute/test_round.zig:"round hand-written values for every mode on floats, integers and decimals (pyarrow 26 results)"`, `src/compute/test_round.zig:"round scalars, empty and all-null inputs"`

## floor

Largest integral value not above x (rounding mode `down`). Floats keep their type and pass NaN and infinities through. Integers have no kernel: they are cast to float64 first, as Arrow (an int32 argument gives float64). decimal128 / decimal256 keep their type and are rounded to an integral value (mode `down`); a result past the precision is `Overflow`. Null handling is intersection; sliced, chunked and scalar inputs; null-type arguments give a null-type result.

Evidence: `src/compute/round.zig:48`, `src/compute/round.zig:237`, `src/compute/test_round.zig:"round hand-written values for every mode on floats, integers and decimals (pyarrow 26 results)"`, `src/compute/test_round.zig:"round scalars, empty and all-null inputs"`

## trunc

Integral part of x, rounding toward zero (mode `towards_zero`). Floats keep their type and pass NaN and infinities through. Integers have no kernel: they are cast to float64 first, as Arrow (an int32 argument gives float64). decimal128 / decimal256 keep their type and are rounded to an integral value (mode `towards_zero`); a result past the precision is `Overflow`. Null handling is intersection; sliced, chunked and scalar inputs; null-type arguments give a null-type result.

Evidence: `src/compute/round.zig:48`, `src/compute/round.zig:237`, `src/compute/test_round.zig:"round hand-written values for every mode on floats, integers and decimals (pyarrow 26 results)"`, `src/compute/test_round.zig:"round scalars, empty and all-null inputs"`

## round

Rounds to `RoundOptions.ndigits` digits (default 0; negative rounds to tens, hundreds, ...) with `RoundOptions.round_mode` (default `half_to_even`; all ten Arrow modes: `down`, `up`, `towards_zero`, `towards_infinity`, `half_down`, `half_up`, `half_towards_zero`, `half_towards_infinity`, `half_to_even`, `half_to_odd`).

- float32 / float64 follow Arrow's algorithm bit for bit: the value is scaled by a power of ten built like `RoundUtil::Pow10` (a float32 table up to 1e15, then repeated multiplication in double, cast to the input type), rounded (only `frac == 0.5` breaks ties by the mode, any other fraction rounds to nearest) and scaled back. NaN and infinities pass through; a value that becomes infinite is `Overflow` (Arrow's "overflow occurred during rounding"). A huge `ndigits` stops the power-of-ten loop at infinity, where Arrow would loop for ever.
- int8..int64, uint8..uint64 keep their type: `ndigits >= 0` is the identity, a negative `ndigits` rounds to a multiple of `10^-ndigits`; more digits than the type holds (Arrow's `digits10`) is `InvalidOptions`, also for an empty input, as Arrow; a result past the type is `Overflow`.
- decimal128 / decimal256 keep their type: the stored value is rounded to a multiple of `10^(scale - ndigits)`; when that exponent reaches the precision every valid row is `InvalidOptions`, when it is not positive the value is kept; a result past the precision is `Overflow`. decimal32 / decimal64 have no kernel (as Arrow).

Checked against pyarrow 26 by hashes over every int8 / uint8 value, float grids at ndigits -3..4 and decimal grids for all ten modes. Errors are only raised for valid rows.

Evidence: `src/compute/round.zig:237`, `src/compute/round.zig:186`, `src/compute/round.zig:198`, `src/compute/options.zig:363`, `src/compute/test_round.zig:"round hand-written values for every mode on floats, integers and decimals (pyarrow 26 results)"`, `src/compute/test_round.zig:"round matches pyarrow 26 hashes over every int8 / uint8 value, float grids and decimals for every mode"`, `src/compute/test_round.zig:"round errors: digits out of range, multiple not positive, precision overflow, float overflow, types and malformed input"`, `src/compute/test_round.zig:"round scalars, empty and all-null inputs"`

## round_binary

As [`round`](#round), but the digits come per row from the second argument; `RoundBinaryOptions` carries only `round_mode` (default `half_to_even`). The digits may be any integer type or bool and are cast to int32 (Arrow's implicit cast: `round_binary([1.25, 2.5], [True, False])` is `[1.2, 2.0]`). For a decimal128 / decimal256 value they may also be a decimal of any width and scale or a float, cast safely to int32 as pyarrow 26 (a fraction is `InvalidCast`; pyarrow: "Rescaling Decimal value would cause data loss"). A row whose digits are out of range for an integer or decimal input is `InvalidInput`. Null handling is intersection of the value and the digits.

Evidence: `src/compute/round.zig:237`, `src/compute/round.zig:81`, `src/compute/test_round.zig:"round_binary per-row digits equal round with constant digits on random sliced inputs with nulls, chunked and scalars"`, `src/compute/test_round.zig:"round_binary on decimal128 / decimal256 with decimal digits of any width and scale: nulls, modes, sliced, chunked (pyarrow 26 results)"`, `src/compute/test_round.zig:"round_binary with bool digits (cast to int32 0 / 1): nulls, sliced, chunked and scalar (pyarrow 26 results)"`

## round_to_multiple

Rounds to a multiple of `RoundToMultipleOptions.multiple` (a scalar, default float64 1.0) with `round_mode` (default `half_to_even`). The multiple is cast to the input type with a safe cast (a lost fraction is `InvalidCast`); a null, zero, negative or NaN multiple is `InvalidOptions`. Floats divide by the multiple, round as [`round`](#round) and multiply back (`Overflow` when the result becomes infinite); integers and decimal128 / decimal256 round to a neighbouring multiple in a wider type, ties by mode, and a result past the type or precision is `Overflow`.

Evidence: `src/compute/round.zig:237`, `src/compute/round.zig:198`, `src/compute/options.zig:371`, `src/compute/test_round.zig:"round hand-written values for every mode on floats, integers and decimals (pyarrow 26 results)"`, `src/compute/test_round.zig:"round errors: digits out of range, multiple not positive, precision overflow, float overflow, types and malformed input"`

## exp

E raised to x. Integers and decimals are cast to float64 first (Arrow's `ArithmeticFloatingPointFunction`). Results follow IEEE 754 and stay within 2 ulp of pyarrow 26 (Zig's std.math and the C library Arrow calls may differ in the last bit). NaN passes through. Null handling is intersection; sliced, chunked and scalar inputs; null-type arguments give a null-type result. There is no `_checked` variant (none in Arrow).

Evidence: `src/compute/math.zig:151`, `src/compute/math.zig:113`, `src/compute/test_math.zig:"math unary functions on float64 and float32 match pyarrow 26 (NaN, infinities, zeros, nulls)"`, `src/compute/test_math.zig:"math oracle matches a per-element loop on random sliced inputs with nulls, chunked and scalar operands"`

## expm1

E^x - 1, accurate near 0. Integers and decimals are cast to float64 first (Arrow's `ArithmeticFloatingPointFunction`). Results follow IEEE 754 and stay within 2 ulp of pyarrow 26 (Zig's std.math and the C library Arrow calls may differ in the last bit). NaN passes through. Null handling is intersection; sliced, chunked and scalar inputs; null-type arguments give a null-type result. There is no `_checked` variant (none in Arrow).

Evidence: `src/compute/math.zig:151`, `src/compute/math.zig:113`, `src/compute/test_math.zig:"math unary functions on float64 and float32 match pyarrow 26 (NaN, infinities, zeros, nulls)"`, `src/compute/test_math.zig:"math oracle matches a per-element loop on random sliced inputs with nulls, chunked and scalar operands"`

## sqrt

Square root; a negative argument gives NaN. Integers and decimals are cast to float64 first (Arrow's `ArithmeticFloatingPointFunction`). Results follow IEEE 754 and stay within 2 ulp of pyarrow 26 (Zig's std.math and the C library Arrow calls may differ in the last bit). NaN passes through. Null handling is intersection; sliced, chunked and scalar inputs; null-type arguments give a null-type result. The checked variant is [`sqrt_checked`](#sqrt_checked).

Evidence: `src/compute/math.zig:151`, `src/compute/math.zig:113`, `src/compute/test_math.zig:"math unary functions on float64 and float32 match pyarrow 26 (NaN, infinities, zeros, nulls)"`, `src/compute/test_math.zig:"math oracle matches a per-element loop on random sliced inputs with nulls, chunked and scalar operands"`

## sqrt_checked

As [`sqrt`](#sqrt), but a negative argument in a valid row is `InvalidInput` (Arrow's domain check); NaN passes without an error and null rows never raise.

Evidence: `src/compute/math.zig:138`, `src/compute/test_math.zig:"math checked variants: domain errors only on valid rows, NaN passes (pyarrow 26)"`

## ln

Natural logarithm; 0 gives -inf, a negative argument NaN. Integers and decimals are cast to float64 first (Arrow's `ArithmeticFloatingPointFunction`). Results follow IEEE 754 and stay within 2 ulp of pyarrow 26 (Zig's std.math and the C library Arrow calls may differ in the last bit). NaN passes through. Null handling is intersection; sliced, chunked and scalar inputs; null-type arguments give a null-type result. The checked variant is [`ln_checked`](#ln_checked).

Evidence: `src/compute/math.zig:151`, `src/compute/math.zig:113`, `src/compute/test_math.zig:"math unary functions on float64 and float32 match pyarrow 26 (NaN, infinities, zeros, nulls)"`, `src/compute/test_math.zig:"math oracle matches a per-element loop on random sliced inputs with nulls, chunked and scalar operands"`

## ln_checked

As [`ln`](#ln), but zero or a negative argument in a valid row is `InvalidInput` (Arrow's domain check); NaN passes without an error and null rows never raise.

Evidence: `src/compute/math.zig:138`, `src/compute/test_math.zig:"math checked variants: domain errors only on valid rows, NaN passes (pyarrow 26)"`

## log2

Base-2 logarithm; 0 gives -inf, a negative argument NaN. Integers and decimals are cast to float64 first (Arrow's `ArithmeticFloatingPointFunction`). Results follow IEEE 754 and stay within 2 ulp of pyarrow 26 (Zig's std.math and the C library Arrow calls may differ in the last bit). NaN passes through. Null handling is intersection; sliced, chunked and scalar inputs; null-type arguments give a null-type result. The checked variant is [`log2_checked`](#log2_checked).

Evidence: `src/compute/math.zig:151`, `src/compute/math.zig:113`, `src/compute/test_math.zig:"math unary functions on float64 and float32 match pyarrow 26 (NaN, infinities, zeros, nulls)"`, `src/compute/test_math.zig:"math oracle matches a per-element loop on random sliced inputs with nulls, chunked and scalar operands"`

## log2_checked

As [`log2`](#log2), but zero or a negative argument in a valid row is `InvalidInput` (Arrow's domain check); NaN passes without an error and null rows never raise.

Evidence: `src/compute/math.zig:138`, `src/compute/test_math.zig:"math checked variants: domain errors only on valid rows, NaN passes (pyarrow 26)"`

## log10

Base-10 logarithm; 0 gives -inf, a negative argument NaN. Integers and decimals are cast to float64 first (Arrow's `ArithmeticFloatingPointFunction`). Results follow IEEE 754 and stay within 2 ulp of pyarrow 26 (Zig's std.math and the C library Arrow calls may differ in the last bit). NaN passes through. Null handling is intersection; sliced, chunked and scalar inputs; null-type arguments give a null-type result. The checked variant is [`log10_checked`](#log10_checked).

Evidence: `src/compute/math.zig:151`, `src/compute/math.zig:113`, `src/compute/test_math.zig:"math unary functions on float64 and float32 match pyarrow 26 (NaN, infinities, zeros, nulls)"`, `src/compute/test_math.zig:"math oracle matches a per-element loop on random sliced inputs with nulls, chunked and scalar operands"`

## log10_checked

As [`log10`](#log10), but zero or a negative argument in a valid row is `InvalidInput` (Arrow's domain check); NaN passes without an error and null rows never raise.

Evidence: `src/compute/math.zig:138`, `src/compute/test_math.zig:"math checked variants: domain errors only on valid rows, NaN passes (pyarrow 26)"`

## log1p

Ln(1 + x), accurate near 0; -1 gives -inf, below -1 NaN. Integers and decimals are cast to float64 first (Arrow's `ArithmeticFloatingPointFunction`). Results follow IEEE 754 and stay within 2 ulp of pyarrow 26 (Zig's std.math and the C library Arrow calls may differ in the last bit). NaN passes through. Null handling is intersection; sliced, chunked and scalar inputs; null-type arguments give a null-type result. The checked variant is [`log1p_checked`](#log1p_checked).

Evidence: `src/compute/math.zig:151`, `src/compute/math.zig:113`, `src/compute/test_math.zig:"math unary functions on float64 and float32 match pyarrow 26 (NaN, infinities, zeros, nulls)"`, `src/compute/test_math.zig:"math oracle matches a per-element loop on random sliced inputs with nulls, chunked and scalar operands"`

## log1p_checked

As [`log1p`](#log1p), but an argument at or below -1 in a valid row is `InvalidInput` (Arrow's domain check); NaN passes without an error and null rows never raise.

Evidence: `src/compute/math.zig:138`, `src/compute/test_math.zig:"math checked variants: domain errors only on valid rows, NaN passes (pyarrow 26)"`

## logb

`log(x) / log(base)` with Arrow's special cases: `x == 0` gives -inf (NaN when `base <= 0`), `x < 0` gives NaN. Integer and decimal operands are cast to float64. NaN passes through; null handling is intersection; array, chunked and scalar operands. The checked variant is [`logb_checked`](#logb_checked).

Evidence: `src/compute/math.zig:240`, `src/compute/math.zig:234`, `src/compute/test_math.zig:"math binary functions, integer power, bit-wise and shifts match pyarrow 26"`, `src/compute/test_math.zig:"math oracle matches a per-element loop on random sliced inputs with nulls, chunked and scalar operands"`

## logb_checked

As [`logb`](#logb), but a valid row with `x <= 0` or `base <= 0` is `InvalidInput`.

Evidence: `src/compute/math.zig:240`, `src/compute/test_math.zig:"math checked variants: domain errors only on valid rows, NaN passes (pyarrow 26)"`, `src/compute/test_math.zig:"math binary functions, integer power, bit-wise and shifts match pyarrow 26"`

## sin

Sine (radians); an infinity gives NaN. Integers and decimals are cast to float64 first (Arrow's `ArithmeticFloatingPointFunction`). Results follow IEEE 754 and stay within 2 ulp of pyarrow 26 (Zig's std.math and the C library Arrow calls may differ in the last bit). NaN passes through. Null handling is intersection; sliced, chunked and scalar inputs; null-type arguments give a null-type result. The checked variant is [`sin_checked`](#sin_checked).

Evidence: `src/compute/math.zig:151`, `src/compute/math.zig:113`, `src/compute/test_math.zig:"math unary functions on float64 and float32 match pyarrow 26 (NaN, infinities, zeros, nulls)"`, `src/compute/test_math.zig:"math oracle matches a per-element loop on random sliced inputs with nulls, chunked and scalar operands"`

## sin_checked

As [`sin`](#sin), but an infinity in a valid row is `InvalidInput` (Arrow's domain check); NaN passes without an error and null rows never raise.

Evidence: `src/compute/math.zig:138`, `src/compute/test_math.zig:"math checked variants: domain errors only on valid rows, NaN passes (pyarrow 26)"`

## cos

Cosine (radians); an infinity gives NaN. Integers and decimals are cast to float64 first (Arrow's `ArithmeticFloatingPointFunction`). Results follow IEEE 754 and stay within 2 ulp of pyarrow 26 (Zig's std.math and the C library Arrow calls may differ in the last bit). NaN passes through. Null handling is intersection; sliced, chunked and scalar inputs; null-type arguments give a null-type result. The checked variant is [`cos_checked`](#cos_checked).

Evidence: `src/compute/math.zig:151`, `src/compute/math.zig:113`, `src/compute/test_math.zig:"math unary functions on float64 and float32 match pyarrow 26 (NaN, infinities, zeros, nulls)"`, `src/compute/test_math.zig:"math oracle matches a per-element loop on random sliced inputs with nulls, chunked and scalar operands"`

## cos_checked

As [`cos`](#cos), but an infinity in a valid row is `InvalidInput` (Arrow's domain check); NaN passes without an error and null rows never raise.

Evidence: `src/compute/math.zig:138`, `src/compute/test_math.zig:"math checked variants: domain errors only on valid rows, NaN passes (pyarrow 26)"`

## tan

Tangent (radians); an infinity gives NaN. Integers and decimals are cast to float64 first (Arrow's `ArithmeticFloatingPointFunction`). Results follow IEEE 754 and stay within 2 ulp of pyarrow 26 (Zig's std.math and the C library Arrow calls may differ in the last bit). NaN passes through. Null handling is intersection; sliced, chunked and scalar inputs; null-type arguments give a null-type result. The checked variant is [`tan_checked`](#tan_checked).

Evidence: `src/compute/math.zig:151`, `src/compute/math.zig:113`, `src/compute/test_math.zig:"math unary functions on float64 and float32 match pyarrow 26 (NaN, infinities, zeros, nulls)"`, `src/compute/test_math.zig:"math oracle matches a per-element loop on random sliced inputs with nulls, chunked and scalar operands"`

## tan_checked

As [`tan`](#tan), but an infinity in a valid row is `InvalidInput` (Arrow's domain check); NaN passes without an error and null rows never raise.

Evidence: `src/compute/math.zig:138`, `src/compute/test_math.zig:"math checked variants: domain errors only on valid rows, NaN passes (pyarrow 26)"`

## asin

Arcsine; outside [-1, 1] gives NaN. Integers and decimals are cast to float64 first (Arrow's `ArithmeticFloatingPointFunction`). Results follow IEEE 754 and stay within 2 ulp of pyarrow 26 (Zig's std.math and the C library Arrow calls may differ in the last bit). NaN passes through. Null handling is intersection; sliced, chunked and scalar inputs; null-type arguments give a null-type result. The checked variant is [`asin_checked`](#asin_checked).

Evidence: `src/compute/math.zig:151`, `src/compute/math.zig:113`, `src/compute/test_math.zig:"math unary functions on float64 and float32 match pyarrow 26 (NaN, infinities, zeros, nulls)"`, `src/compute/test_math.zig:"math oracle matches a per-element loop on random sliced inputs with nulls, chunked and scalar operands"`

## asin_checked

As [`asin`](#asin), but an argument outside [-1, 1] in a valid row is `InvalidInput` (Arrow's domain check); NaN passes without an error and null rows never raise.

Evidence: `src/compute/math.zig:138`, `src/compute/test_math.zig:"math checked variants: domain errors only on valid rows, NaN passes (pyarrow 26)"`

## acos

Arccosine; outside [-1, 1] gives NaN. Integers and decimals are cast to float64 first (Arrow's `ArithmeticFloatingPointFunction`). Results follow IEEE 754 and stay within 2 ulp of pyarrow 26 (Zig's std.math and the C library Arrow calls may differ in the last bit). NaN passes through. Null handling is intersection; sliced, chunked and scalar inputs; null-type arguments give a null-type result. The checked variant is [`acos_checked`](#acos_checked).

Evidence: `src/compute/math.zig:151`, `src/compute/math.zig:113`, `src/compute/test_math.zig:"math unary functions on float64 and float32 match pyarrow 26 (NaN, infinities, zeros, nulls)"`, `src/compute/test_math.zig:"math oracle matches a per-element loop on random sliced inputs with nulls, chunked and scalar operands"`

## acos_checked

As [`acos`](#acos), but an argument outside [-1, 1] in a valid row is `InvalidInput` (Arrow's domain check); NaN passes without an error and null rows never raise.

Evidence: `src/compute/math.zig:138`, `src/compute/test_math.zig:"math checked variants: domain errors only on valid rows, NaN passes (pyarrow 26)"`

## atanh

Inverse hyperbolic tangent; +-1 gives +-inf, outside [-1, 1] NaN. Integers and decimals are cast to float64 first (Arrow's `ArithmeticFloatingPointFunction`). Results follow IEEE 754 and stay within 2 ulp of pyarrow 26 (Zig's std.math and the C library Arrow calls may differ in the last bit). NaN passes through. Null handling is intersection; sliced, chunked and scalar inputs; null-type arguments give a null-type result. The checked variant is [`atanh_checked`](#atanh_checked).

Evidence: `src/compute/math.zig:151`, `src/compute/math.zig:113`, `src/compute/test_math.zig:"math unary functions on float64 and float32 match pyarrow 26 (NaN, infinities, zeros, nulls)"`, `src/compute/test_math.zig:"math oracle matches a per-element loop on random sliced inputs with nulls, chunked and scalar operands"`

## atanh_checked

As [`atanh`](#atanh), but an argument at or beyond -1 or 1 in a valid row is `InvalidInput` (Arrow's domain check); NaN passes without an error and null rows never raise.

Evidence: `src/compute/math.zig:138`, `src/compute/test_math.zig:"math checked variants: domain errors only on valid rows, NaN passes (pyarrow 26)"`

## acosh

Inverse hyperbolic cosine; below 1 gives NaN. Integers and decimals are cast to float64 first (Arrow's `ArithmeticFloatingPointFunction`). Results follow IEEE 754 and stay within 2 ulp of pyarrow 26 (Zig's std.math and the C library Arrow calls may differ in the last bit). NaN passes through. Null handling is intersection; sliced, chunked and scalar inputs; null-type arguments give a null-type result. The checked variant is [`acosh_checked`](#acosh_checked).

Evidence: `src/compute/math.zig:151`, `src/compute/math.zig:113`, `src/compute/test_math.zig:"math unary functions on float64 and float32 match pyarrow 26 (NaN, infinities, zeros, nulls)"`, `src/compute/test_math.zig:"math oracle matches a per-element loop on random sliced inputs with nulls, chunked and scalar operands"`

## acosh_checked

As [`acosh`](#acosh), but an argument below 1 in a valid row is `InvalidInput` (Arrow's domain check); NaN passes without an error and null rows never raise.

Evidence: `src/compute/math.zig:138`, `src/compute/test_math.zig:"math checked variants: domain errors only on valid rows, NaN passes (pyarrow 26)"`

## atan

Arctangent. Integers and decimals are cast to float64 first (Arrow's `ArithmeticFloatingPointFunction`). Results follow IEEE 754 and stay within 2 ulp of pyarrow 26 (Zig's std.math and the C library Arrow calls may differ in the last bit). NaN passes through. Null handling is intersection; sliced, chunked and scalar inputs; null-type arguments give a null-type result. There is no `_checked` variant (none in Arrow).

Evidence: `src/compute/math.zig:151`, `src/compute/math.zig:113`, `src/compute/test_math.zig:"math unary functions on float64 and float32 match pyarrow 26 (NaN, infinities, zeros, nulls)"`, `src/compute/test_math.zig:"math oracle matches a per-element loop on random sliced inputs with nulls, chunked and scalar operands"`

## atan2

Two-argument arctangent `atan2(y, x)` (the first argument is y). Integer and decimal operands are cast to float64, and a float32 with an int32 also gives float64. NaN passes through; null handling is intersection; array, chunked and scalar operands. No `_checked` variant.

Evidence: `src/compute/math.zig:240`, `src/compute/test_math.zig:"math binary functions, integer power, bit-wise and shifts match pyarrow 26"`, `src/compute/test_math.zig:"math implicit casts: integers and decimals to float64, power and bit-wise common types (pyarrow 26)"`

## sinh

Hyperbolic sine. Integers and decimals are cast to float64 first (Arrow's `ArithmeticFloatingPointFunction`). Results follow IEEE 754 and stay within 2 ulp of pyarrow 26 (Zig's std.math and the C library Arrow calls may differ in the last bit). NaN passes through. Null handling is intersection; sliced, chunked and scalar inputs; null-type arguments give a null-type result. There is no `_checked` variant (none in Arrow).

Evidence: `src/compute/math.zig:151`, `src/compute/math.zig:113`, `src/compute/test_math.zig:"math unary functions on float64 and float32 match pyarrow 26 (NaN, infinities, zeros, nulls)"`, `src/compute/test_math.zig:"math oracle matches a per-element loop on random sliced inputs with nulls, chunked and scalar operands"`

## cosh

Hyperbolic cosine. Integers and decimals are cast to float64 first (Arrow's `ArithmeticFloatingPointFunction`). Results follow IEEE 754 and stay within 2 ulp of pyarrow 26 (Zig's std.math and the C library Arrow calls may differ in the last bit). NaN passes through. Null handling is intersection; sliced, chunked and scalar inputs; null-type arguments give a null-type result. There is no `_checked` variant (none in Arrow).

Evidence: `src/compute/math.zig:151`, `src/compute/math.zig:113`, `src/compute/test_math.zig:"math unary functions on float64 and float32 match pyarrow 26 (NaN, infinities, zeros, nulls)"`, `src/compute/test_math.zig:"math oracle matches a per-element loop on random sliced inputs with nulls, chunked and scalar operands"`

## tanh

Hyperbolic tangent. Integers and decimals are cast to float64 first (Arrow's `ArithmeticFloatingPointFunction`). Results follow IEEE 754 and stay within 2 ulp of pyarrow 26 (Zig's std.math and the C library Arrow calls may differ in the last bit). NaN passes through. Null handling is intersection; sliced, chunked and scalar inputs; null-type arguments give a null-type result. There is no `_checked` variant (none in Arrow).

Evidence: `src/compute/math.zig:151`, `src/compute/math.zig:113`, `src/compute/test_math.zig:"math unary functions on float64 and float32 match pyarrow 26 (NaN, infinities, zeros, nulls)"`, `src/compute/test_math.zig:"math oracle matches a per-element loop on random sliced inputs with nulls, chunked and scalar operands"`

## asinh

Inverse hyperbolic sine. Integers and decimals are cast to float64 first (Arrow's `ArithmeticFloatingPointFunction`). Results follow IEEE 754 and stay within 2 ulp of pyarrow 26 (Zig's std.math and the C library Arrow calls may differ in the last bit). NaN passes through. Null handling is intersection; sliced, chunked and scalar inputs; null-type arguments give a null-type result. There is no `_checked` variant (none in Arrow).

Evidence: `src/compute/math.zig:151`, `src/compute/math.zig:113`, `src/compute/test_math.zig:"math unary functions on float64 and float32 match pyarrow 26 (NaN, infinities, zeros, nulls)"`, `src/compute/test_math.zig:"math oracle matches a per-element loop on random sliced inputs with nulls, chunked and scalar operands"`

## hypot

`sqrt(x^2 + y^2)` via `std.math.hypot`: no intermediate overflow, and +inf when either argument is an infinity even if the other is NaN (as C's `hypot`); otherwise NaN passes through. Integer and mixed operands go to a float type as for [`atan2`](#atan2); float16 goes to float32. No `_checked` variant.

Evidence: `src/compute/math.zig:240`, `src/compute/test_math.zig:"hypot: no intermediate overflow, infinities over NaN, integer and mixed operands, nulls, sliced, chunked, scalars (pyarrow 26 results)"`
