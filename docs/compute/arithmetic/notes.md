# Arithmetic: function notes

Up: [function matrix](matrix.md) · [compute support matrix](../../compute-support-matrix.md)

The Arrow C++ "Arithmetic functions" on int8..int64, uint8..uint64, float32 and float64, plus the
decimal kernels of `decimal.zig` and the temporal kernels of `temporal_arith.zig`. Every function
works element-wise on array, chunked or scalar operands (null intersection); operands of different
numeric types are first cast to their common type (Arrow's `DispatchBest` / `CommonNumeric`, a safe
cast unless `ExecConfig.safe_cast` is false), and arguments that are all of the null type give a
null-type result of their length (Arrow's null kernels). Unchecked integer results wrap, or saturate
when `ExecConfig.overflow_mode` is `saturating`; the `_checked` variants return `Overflow`. Errors
are only raised for valid rows. Evidence: `src/compute/arithmetic.zig:42`,
`src/compute/registry.zig:261`, `src/compute/test_arithmetic.zig:"arithmetic implicit casts: mixed numeric operands use Arrow's common type (pyarrow 26 table)"`,
`src/compute/test_arithmetic.zig:"arithmetic, math, rounding, bit-wise and pairwise_diff functions on null-type arguments give a null-type result: arrays, sliced, chunked, scalars (pyarrow 26)"`.

## Summary

| Function | Arity | Input types | Output type | Options |
|---|---|---|---|---|
| [`add`](#add) | Binary | numeric, decimal, temporal + duration | common type / Arrow decimal rule / temporal | none |
| [`add_checked`](#add_checked) | Binary | as `add` | as `add` | none |
| [`subtract`](#subtract) | Binary | numeric, decimal, temporal | common type / Arrow decimal rule / temporal or duration | none |
| [`subtract_checked`](#subtract_checked) | Binary | as `subtract` | as `subtract` | none |
| [`multiply`](#multiply) | Binary | numeric, decimal, duration and integer | common type / Arrow decimal rule / duration | none |
| [`multiply_checked`](#multiply_checked) | Binary | as `multiply` | as `multiply` | none |
| [`divide`](#divide) | Binary | numeric, decimal, duration by integer or duration | common type / Arrow decimal rule / duration or float64 | none |
| [`divide_checked`](#divide_checked) | Binary | as `divide` | as `divide` | none |
| [`modulo`](#modulo) | Binary | numeric, decimal | common type / decimal | none |
| [`modulo_checked`](#modulo_checked) | Binary | as `modulo` | as `modulo` | none |
| [`remainder`](#remainder) | Binary | numeric, decimal | common type / decimal | none |
| [`remainder_checked`](#remainder_checked) | Binary | as `remainder` | as `remainder` | none |
| [`negate`](#negate) | Unary | numeric, float16, decimal128/256, duration | input type | none |
| [`negate_checked`](#negate_checked) | Unary | signed integers, floats, float16, decimal128/256, duration | input type | none |
| [`abs`](#abs) | Unary | numeric, float16, decimal128/256, duration | input type | none |
| [`abs_checked`](#abs_checked) | Unary | as `abs` | input type | none |
| [`power`](#power) | Binary | numeric (decimals go to float64) | common type | none |
| [`power_checked`](#power_checked) | Binary | as `power` | as `power` | none |
| [`sign`](#sign) | Unary | numeric, float16, decimal128/256, duration | int8 (integers, durations), int64 (decimals) or the float type | none |

## add

Element-wise `x + y`. Integers wrap on overflow (Arrow C++), or saturate when
`ExecConfig.overflow_mode` is `saturating`. Float addition never reports overflow. Mixed numeric
operands are cast to their common type first (int32 + int64 is int64, int + float32 is float32,
uint32 + int32 is int64, uint64 + int64 is int64; every pair checked against pyarrow 26); a value
the cast cannot represent is `InvalidCast`. float16 operands go to float32 (also float16 with
float16, as Arrow's `CommonNumeric`: there is no float16 arithmetic kernel); bool has no kernel.

Decimals (`decimal.zig`): two decimal32..decimal256 operands, or a decimal and an integer (the
integer is `decimal(digits, 0)`); result scale `max(s1, s2)`, precision
`max(p1 - s1, p2 - s2) + max(s1, s2) + 1`; decimal256 when an operand is, else decimal128; a
precision past 38 / 76 is `Overflow`, a negative scale `NotImplemented`. Unchecked wraps at the
storage width as Arrow. A decimal with a float goes to float64.

Temporal (`temporal_arith.zig`, pyarrow 26 results): timestamp + duration (timezone kept),
duration + duration, date32 / date64 + duration (a timestamp of seconds / milliseconds or finer),
time32 / time64 + duration (a time; outside one day is `InvalidInput` in both variants). Operands
of different units go to the finer one (a value that does not fit is `InvalidCast`). Interval
arithmetic is absent, as in pyarrow 26.

Null rows may hold any value: no error is raised for them. Sliced, chunked (misaligned chunks give a
chunked result) and scalar operands are covered by the tests.

Evidence: `src/compute/arithmetic.zig:118`, `src/compute/decimal.zig:287`, `src/compute/temporal_arith.zig:55`,
`src/compute/test_arithmetic.zig:"arithmetic hand-written values: wrapping, saturating, truncating division, float division by zero"`,
`src/compute/test_arithmetic.zig:"arithmetic oracle matches a per-element loop on random sliced inputs with nulls for every numeric type"`,
`src/compute/test_arithmetic.zig:"arithmetic chunked inputs with misaligned chunks give a chunked result equal to the array result"`,
`src/compute/test_arithmetic.zig:"arithmetic scalar broadcast: array-scalar, scalar-array, scalar-scalar and null scalars"`,
`src/compute/test_decimal.zig:"decimal arithmetic follows Arrow's precision and scale rules (nulls, sliced, chunked, scalars)"`,
`src/compute/test_temporal_arith.zig:"temporal arithmetic hand-written values: timestamps, durations, dates and times (pyarrow 26 results)"`,
`src/compute/test_cast.zig:"float16: cast to and from numbers and text, promotion to float32 in arithmetic and comparisons, is_nan / is_inf / is_finite (pyarrow 26 results)"`

## add_checked

As [`add`](#add), but integer overflow in a valid row is `Overflow` (also for temporal operands);
decimal results past the storage or the result precision are `Overflow`. Float results are the same
as `add`. The overflow scan is branch free and only revisits rows when some row overflowed, so null
rows never raise.

Evidence: `src/compute/arithmetic.zig:105`, `src/compute/decimal.zig:331`,
`src/compute/test_arithmetic.zig:"arithmetic errors: Overflow, DivideByZero, mismatched types, options, lengths and malformed input"`,
`src/compute/test_arithmetic.zig:"arithmetic empty and all-null inputs: no error from overflow or zero divisors in null rows"`,
`src/compute/test_decimal.zig:"decimal arithmetic errors: precision overflow, negative scale, zero divisor, checked overflow, bad precision"`

## subtract

Element-wise `x - y`, same types, casts, overflow behaviour and decimal result type as
[`add`](#add). Temporal: timestamp - duration, duration - duration, date / time - duration as for
`add`, and the difference of two timestamps (both naive or both zoned, else `InvalidInput`), of two
dates or of two times (a duration).

Evidence: `src/compute/arithmetic.zig:118`, `src/compute/decimal.zig:287`, `src/compute/temporal_arith.zig:121`,
`src/compute/test_arithmetic.zig:"arithmetic hand-written values: wrapping, saturating, truncating division, float division by zero"`,
`src/compute/test_arithmetic.zig:"arithmetic oracle matches a per-element loop on random sliced inputs with nulls for every numeric type"`,
`src/compute/test_temporal_arith.zig:"temporal arithmetic oracle matches a per-row loop on random sliced inputs with nulls, chunked and scalars"`

## subtract_checked

As [`subtract`](#subtract), with `Overflow` for a valid row that overflows (integers, durations,
timestamps, decimals).

Evidence: `src/compute/arithmetic.zig:105`,
`src/compute/test_arithmetic.zig:"arithmetic errors: Overflow, DivideByZero, mismatched types, options, lengths and malformed input"`,
`src/compute/test_temporal_arith.zig:"temporal arithmetic errors: overflow, unit casts, time range, zoned and naive, zero divisor, types"`

## multiply

Element-wise `x * y` with the casts and overflow modes of [`add`](#add). Decimals: scale `s1 + s2`,
precision `p1 + p2 + 1` (Arrow C++). Temporal: a duration times an integer is a duration.

Evidence: `src/compute/arithmetic.zig:118`, `src/compute/decimal.zig:287`,
`src/compute/test_arithmetic.zig:"arithmetic hand-written values: wrapping, saturating, truncating division, float division by zero"`,
`src/compute/test_decimal.zig:"decimal arithmetic follows Arrow's precision and scale rules (nulls, sliced, chunked, scalars)"`,
`src/compute/test_temporal_arith.zig:"temporal arithmetic hand-written values: timestamps, durations, dates and times (pyarrow 26 results)"`

## multiply_checked

As [`multiply`](#multiply), with `Overflow` for a valid row whose product overflows.

Evidence: `src/compute/arithmetic.zig:105`,
`src/compute/test_arithmetic.zig:"arithmetic errors: Overflow, DivideByZero, mismatched types, options, lengths and malformed input"`

## divide

Integer division truncates toward zero. A zero divisor in a valid row is `DivideByZero` in both
variants (as Arrow C++); null rows holding a zero divisor are skipped. `divide(minInt, -1)` is 0
unchecked (Arrow C++), `maxInt` when saturating. Float division by zero gives inf / NaN.

Decimals: the dividend is scaled up by `max(4, s1 + p2 - s2 + 1) + s2 - s1` digits and the quotient
truncated (Arrow C++); a zero divisor is `DivideByZero`. Temporal: a duration divided by an integer
(truncating, a duration) or by a duration (float64).

Evidence: `src/compute/arithmetic.zig:118`, `src/compute/decimal.zig:287`,
`src/compute/test_arithmetic.zig:"arithmetic hand-written values: wrapping, saturating, truncating division, float division by zero"`,
`src/compute/test_arithmetic.zig:"arithmetic empty and all-null inputs: no error from overflow or zero divisors in null rows"`,
`src/compute/test_decimal.zig:"decimal arithmetic errors: precision overflow, negative scale, zero divisor, checked overflow, bad precision"`

## divide_checked

As [`divide`](#divide), but `divide_checked(minInt, -1)` is `Overflow` and a zero float divisor in
a valid row is `DivideByZero`.

Evidence: `src/compute/arithmetic.zig:105`,
`src/compute/test_arithmetic.zig:"arithmetic errors: Overflow, DivideByZero, mismatched types, options, lengths and malformed input"`

## modulo

Floored modulo: the result has the sign of the divisor (Python's `%`). `modulo(-7, 2)` is 1,
`modulo(-7.5, 2)` is 0.5, `modulo(-5, inf)` is inf, and a zero float result takes the divisor's
sign. A zero integer divisor in a valid row is `DivideByZero` in both variants (Arrow: "divide by
zero"); `modulo(minInt, -1)` is 0; a zero float divisor gives NaN. Same implicit casts as
`divide` (int8 with uint8 is int16); no float16 or bool kernel, as Arrow.

Decimals: both operands scaled to `max(s1, s2)`, precision `max(p1 - s1, p2 - s2) + max(s1, s2)`
(pyarrow 26), never overflowing; integers are accepted as `decimal(digits, 0)`.

Evidence: `src/compute/arithmetic.zig:302`, `src/compute/decimal.zig:287`,
`src/compute/test_arithmetic.zig:"modulo, remainder and their checked variants: signs, zero divisors, overflow, floats, nulls, sliced, chunked, scalars (pyarrow 26 results)"`,
`src/compute/test_decimal.zig:"decimal modulo and remainder: Arrow's scale rule, signs, nulls, sliced, chunked, integers, zero divisor (pyarrow 26 results)"`

## modulo_checked

As [`modulo`](#modulo), but `modulo_checked(minInt, -1)` is `Overflow` and a zero float divisor in
a valid row is `DivideByZero`.

Evidence: `src/compute/arithmetic.zig:118`,
`src/compute/test_arithmetic.zig:"modulo, remainder and their checked variants: signs, zero divisors, overflow, floats, nulls, sliced, chunked, scalars (pyarrow 26 results)"`

## remainder

Truncated remainder: the result has the sign of the dividend (C's `%` and `fmod`):
`remainder(-7.5, 2)` is -1.5. Zero divisors, `minInt % -1`, casts and the decimal rule are as for
[`modulo`](#modulo).

Evidence: `src/compute/arithmetic.zig:302`,
`src/compute/test_arithmetic.zig:"modulo, remainder and their checked variants: signs, zero divisors, overflow, floats, nulls, sliced, chunked, scalars (pyarrow 26 results)"`,
`src/compute/test_decimal.zig:"decimal modulo and remainder: Arrow's scale rule, signs, nulls, sliced, chunked, integers, zero divisor (pyarrow 26 results)"`

## remainder_checked

As [`remainder`](#remainder), with `Overflow` for `minInt` and -1 and `DivideByZero` for a zero
float divisor in a valid row.

Evidence: `src/compute/arithmetic.zig:118`,
`src/compute/test_arithmetic.zig:"modulo, remainder and their checked variants: signs, zero divisors, overflow, floats, nulls, sliced, chunked, scalars (pyarrow 26 results)"`

## negate

`-x` in the input type, no implicit cast. Signed `negate(minInt)` wraps to `minInt` (saturates to
`maxInt` when saturating); unsigned negate wraps (`0 -% x`, or 0 when saturating). float16 keeps its
type (pyarrow 26 has a float16 negate). decimal128 / decimal256 and durations are supported too.

Evidence: `src/compute/arithmetic.zig:118`, `src/compute/decimal.zig:482`,
`src/compute/test_arithmetic.zig:"arithmetic hand-written values: wrapping, saturating, truncating division, float division by zero"`,
`src/compute/test_decimal.zig:"decimal negate, abs, sign and their checked variants"`,
`src/compute/test_associative.zig:"float16 keys and signs: unique, value_counts, dictionary_encode, count_distinct, is_in, sorts, negate, abs and sign (pyarrow 26 results)"`

## negate_checked

As [`negate`](#negate), but `minInt` in a valid row is `Overflow`. There is no unsigned kernel
(as Arrow C++), so an unsigned argument is `NoMatchingKernel`.

Evidence: `src/compute/arithmetic.zig:42`, `src/compute/test_arithmetic.zig:"abs_checked and negate_checked by name: nulls, sliced, chunked, overflow, unsigned (pyarrow 26 results)"`,
`src/compute/test_arithmetic.zig:"arithmetic errors: Overflow, DivideByZero, mismatched types, options, lengths and malformed input"`,
`src/compute/test_decimal.zig:"decimal negate, abs, sign and their checked variants"`

## abs

Absolute value in the input type. Unsigned is the identity; signed `abs(minInt)` wraps to `minInt`
(saturates to `maxInt`). float16 keeps its type; this kernel is a zarrow extension (pyarrow 26 has
no float16 `abs`). decimal128 / decimal256 and durations are supported.

Evidence: `src/compute/arithmetic.zig:118`, `src/compute/decimal.zig:482`,
`src/compute/test_arithmetic.zig:"arithmetic oracle matches a per-element loop on random sliced inputs with nulls for every numeric type"`,
`src/compute/test_associative.zig:"float16 keys and signs: unique, value_counts, dictionary_encode, count_distinct, is_in, sorts, negate, abs and sign (pyarrow 26 results)"`

## abs_checked

As [`abs`](#abs), but `minInt` in a valid row is `Overflow`.

Evidence: `src/compute/arithmetic.zig:42`, `src/compute/test_arithmetic.zig:"abs_checked and negate_checked by name: nulls, sliced, chunked, overflow, unsigned (pyarrow 26 results)"`,
`src/compute/test_decimal.zig:"decimal negate, abs, sign and their checked variants"`

## power

`base ^ exponent`. Integers use Arrow's left-to-right squaring and wrap on overflow; a negative
exponent is `InvalidInput` wherever the exponent is valid (Arrow checks it before the null
intersection with the base), in both variants. Floats use `pow` and never fail. A decimal operand
makes both float64 (Arrow's `ArithmeticDecimalToFloatingPointFunction`); other mixed operands go to
their common numeric type. Implemented in `math.zig`.

Evidence: `src/compute/math.zig:214`, `src/compute/math.zig:240`,
`src/compute/test_math.zig:"math binary functions, integer power, bit-wise and shifts match pyarrow 26"`,
`src/compute/test_math.zig:"math implicit casts: integers and decimals to float64, power and bit-wise common types (pyarrow 26)"`,
`src/compute/test_math.zig:"math oracle matches a per-element loop on random sliced inputs with nulls, chunked and scalar operands"`

## power_checked

As [`power`](#power), but an integer overflow at any squaring step in a valid row is `Overflow`.

Evidence: `src/compute/math.zig:58`, `src/compute/test_math.zig:"power_checked by name: nulls, sliced, chunked, overflow and negative exponents (pyarrow 26 results)"`,
`src/compute/test_math.zig:"math binary functions, integer power, bit-wise and shifts match pyarrow 26"`

## sign

-1, 0 or 1. Integers and durations give int8 (Arrow C++, pyarrow 26); decimal128 / decimal256
give int64 (pyarrow 26); floats give their own type, NaN stays NaN and both zeros give 0.0; float16 keeps float16. No `_checked` variant.

Evidence: `src/compute/arithmetic.zig:70`, `src/compute/decimal.zig:482`,
`src/compute/test_arithmetic.zig:"sign on duration gives int8: nulls, sliced, chunked and scalars (pyarrow 26 results)"`,
`src/compute/test_decimal.zig:"decimal negate, abs, sign and their checked variants"`
