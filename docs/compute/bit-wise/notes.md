# Bit-wise: function notes

Up: [function matrix](matrix.md) · [compute support matrix](../../compute-support-matrix.md)

The Arrow C++ "Bit-wise functions" on int8..int64 and uint8..uint64, implemented in `math.zig`.
All work element-wise on array, chunked or scalar operands with null intersection; mixed integer
operands of the binary functions go to their common type (int8 with int32 is int32, uint8 with int8
is int16); floats have no kernel, as Arrow. Arguments that are all of the null type give a null-type
result of their length (Arrow's null kernels). Evidence: `src/compute/math.zig:58`,
`src/compute/test_math.zig:"math implicit casts: integers and decimals to float64, power and bit-wise common types (pyarrow 26)"`,
`src/compute/test_arithmetic.zig:"arithmetic, math, rounding, bit-wise and pairwise_diff functions on null-type arguments give a null-type result: arrays, sliced, chunked, scalars (pyarrow 26)"`.

## Summary

| Function | Arity | Input types | Output type | Options |
|---|---|---|---|---|
| [`bit_wise_and`](#bit_wise_and) | Binary | integers | common integer type | none |
| [`bit_wise_not`](#bit_wise_not) | Unary | integers | input type | none |
| [`bit_wise_or`](#bit_wise_or) | Binary | integers | common integer type | none |
| [`bit_wise_xor`](#bit_wise_xor) | Binary | integers | common integer type | none |
| [`shift_left`](#shift_left) | Binary | integers | common integer type | none |
| [`shift_left_checked`](#shift_left_checked) | Binary | integers | common integer type | none |
| [`shift_right`](#shift_right) | Binary | integers | common integer type | none |
| [`shift_right_checked`](#shift_right_checked) | Binary | integers | common integer type | none |

## bit_wise_and

`x & y` per row. Never fails.

Evidence: `src/compute/math.zig:240`,
`src/compute/test_math.zig:"math binary functions, integer power, bit-wise and shifts match pyarrow 26"`,
`src/compute/test_math.zig:"math oracle matches a per-element loop on random sliced inputs with nulls, chunked and scalar operands"`

## bit_wise_not

`~x` per row in the input type, no implicit cast (unary). Never fails.

Evidence: `src/compute/math.zig:190`,
`src/compute/test_math.zig:"math binary functions, integer power, bit-wise and shifts match pyarrow 26"`,
`src/compute/test_math.zig:"math oracle matches a per-element loop on random sliced inputs with nulls, chunked and scalar operands"`

## bit_wise_or

`x | y` per row. Never fails.

Evidence: `src/compute/math.zig:240`,
`src/compute/test_math.zig:"math binary functions, integer power, bit-wise and shifts match pyarrow 26"`

## bit_wise_xor

`x ^ y` per row. Never fails.

Evidence: `src/compute/math.zig:240`,
`src/compute/test_math.zig:"math binary functions, integer power, bit-wise and shifts match pyarrow 26"`

## shift_left

`x << y`, wrapping (bits shifted out are lost; the shift runs on the unsigned bit pattern, so a
signed result may change sign). A shift amount outside `[0, digits)` leaves the value unchanged,
where digits is Arrow's `std::numeric_limits<T>::digits`: 7 for int8, 8 for uint8, 63 for int64,
64 for uint64 (pyarrow 26 results).

Evidence: `src/compute/math.zig:240`,
`src/compute/test_math.zig:"math binary functions, integer power, bit-wise and shifts match pyarrow 26"`,
`src/compute/test_math.zig:"math oracle matches a per-element loop on random sliced inputs with nulls, chunked and scalar operands"`

## shift_left_checked

As [`shift_left`](#shift_left), but a shift amount outside `[0, digits)` in a valid row is
`InvalidInput`. Overflow of the shifted value itself is not reported.

Evidence: `src/compute/math.zig:58`,
`src/compute/test_math.zig:"math checked variants: domain errors only on valid rows, NaN passes (pyarrow 26)"`

## shift_right

`x >> y`: arithmetic (sign-extending) for signed types, logical for unsigned. A shift amount outside
`[0, digits)` leaves the value unchanged, as for [`shift_left`](#shift_left).

Evidence: `src/compute/math.zig:240`,
`src/compute/test_math.zig:"math binary functions, integer power, bit-wise and shifts match pyarrow 26"`

## shift_right_checked

As [`shift_right`](#shift_right), but a shift amount outside `[0, digits)` in a valid row is
`InvalidInput`.

Evidence: `src/compute/math.zig:58`,
`src/compute/test_math.zig:"math checked variants: domain errors only on valid rows, NaN passes (pyarrow 26)"`
