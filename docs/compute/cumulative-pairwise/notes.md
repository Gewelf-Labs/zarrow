# Cumulative, statistical, pairwise: function notes

Up: [function matrix](matrix.md) · [compute support matrix](../../compute-support-matrix.md)

Vector functions whose output row depends on earlier rows: the `cumulative_*` scans, `pairwise_diff` and `winsorize`, as Arrow's cumulative, pairwise and statistical vector kernels. Results are checked against pyarrow 26. Evidence: `src/compute/cumulative.zig:1`, `src/compute/test_cumulative.zig:"cumulative_sum, cumulative_prod, cumulative_min, cumulative_max and cumulative_mean: options, nulls, NaN, overflow (pyarrow 26 results)"`.

## Summary

| Function | Arity | Input types | Output type | Options |
|---|---|---|---|---|
| [`cumulative_max`](#cumulative_max) | Unary | int8..int64, uint8..uint64, float32/64 | input type | `CumulativeOptions` |
| [`cumulative_mean`](#cumulative_mean) | Unary | int8..int64, uint8..uint64, float32/64 | float64 | `CumulativeOptions` |
| [`cumulative_min`](#cumulative_min) | Unary | int8..int64, uint8..uint64, float32/64 | input type | `CumulativeOptions` |
| [`cumulative_prod`](#cumulative_prod) | Unary | int8..int64, uint8..uint64, float32/64 | input type | `CumulativeOptions` |
| [`cumulative_prod_checked`](#cumulative_prod_checked) | Unary | int8..int64, uint8..uint64, float32/64 | input type | `CumulativeOptions` |
| [`cumulative_sum`](#cumulative_sum) | Unary | int8..int64, uint8..uint64, float32/64 | input type | `CumulativeOptions` |
| [`cumulative_sum_checked`](#cumulative_sum_checked) | Unary | int8..int64, uint8..uint64, float32/64 | input type | `CumulativeOptions` |
| [`winsorize`](#winsorize) | Unary | numbers, decimals | input type | `WinsorizeOptions` (required) |
| [`pairwise_diff`](#pairwise_diff) | Unary | numbers, temporal, decimal128, null | numbers: input type; temporal: duration; decimal128(p + 1, s) | `PairwiseOptions` |
| [`pairwise_diff_checked`](#pairwise_diff_checked) | Unary | as `pairwise_diff` | as `pairwise_diff` | `PairwiseOptions` |

## cumulative_max

`cumulative_max` keeps a running maximum over int8..int64, uint8..uint64, float32/64 (pyarrow 26 has no other input type) and returns the input type.

- Options: `CumulativeOptions` (`start` default none, `skip_nulls` default false). `start` is a numeric scalar cast safely to the input type (1000 for int8 or 1.5 for an integer is `InvalidCast`); a non-numeric start is `InvalidOptions`, where Arrow parses a string.
- Without `skip_nulls` every row from the first null on is null; with it a null row is null and the accumulation goes on.
- The running value is carried across chunks and the chunks are kept; a scalar gives a one-row array (Arrow).
- NaN is skipped (an all-NaN prefix gives the identity, -inf); of two equal values the later wins (-0.0 then 0.0 gives 0.0).

Evidence: `src/compute/cumulative.zig:65`, `src/compute/cumulative.zig:57`, `src/compute/test_cumulative.zig:"cumulative_sum, cumulative_prod, cumulative_min, cumulative_max and cumulative_mean: options, nulls, NaN, overflow (pyarrow 26 results)"`

## cumulative_mean

`cumulative_mean` keeps a running mean over int8..int64, uint8..uint64, float32/64 (pyarrow 26 has no other input type) and returns the input type.

- Options: `CumulativeOptions` (`start` default none, `skip_nulls` default false). `start` is a numeric scalar cast safely to the input type (1000 for int8 or 1.5 for an integer is `InvalidCast`); a non-numeric start is `InvalidOptions`, where Arrow parses a string.
- Without `skip_nulls` every row from the first null on is null; with it a null row is null and the accumulation goes on.
- The running value is carried across chunks and the chunks are kept; a scalar gives a one-row array (Arrow).
- Deviation from the template above: the output is float64 over a float64 sum and `start` is ignored, as Arrow.

Evidence: `src/compute/cumulative.zig:66`, `src/compute/cumulative.zig:57`, `src/compute/test_cumulative.zig:"cumulative_sum, cumulative_prod, cumulative_min, cumulative_max and cumulative_mean: options, nulls, NaN, overflow (pyarrow 26 results)"`

## cumulative_min

`cumulative_min` keeps a running minimum over int8..int64, uint8..uint64, float32/64 (pyarrow 26 has no other input type) and returns the input type.

- Options: `CumulativeOptions` (`start` default none, `skip_nulls` default false). `start` is a numeric scalar cast safely to the input type (1000 for int8 or 1.5 for an integer is `InvalidCast`); a non-numeric start is `InvalidOptions`, where Arrow parses a string.
- Without `skip_nulls` every row from the first null on is null; with it a null row is null and the accumulation goes on.
- The running value is carried across chunks and the chunks are kept; a scalar gives a one-row array (Arrow).
- NaN is skipped (an all-NaN prefix gives the identity, inf); of two equal values the later wins.

Evidence: `src/compute/cumulative.zig:64`, `src/compute/cumulative.zig:57`, `src/compute/test_cumulative.zig:"cumulative_sum, cumulative_prod, cumulative_min, cumulative_max and cumulative_mean: options, nulls, NaN, overflow (pyarrow 26 results)"`

## cumulative_prod

`cumulative_prod` keeps a running product over int8..int64, uint8..uint64, float32/64 (pyarrow 26 has no other input type) and returns the input type.

- Options: `CumulativeOptions` (`start` default none, `skip_nulls` default false). `start` is a numeric scalar cast safely to the input type (1000 for int8 or 1.5 for an integer is `InvalidCast`); a non-numeric start is `InvalidOptions`, where Arrow parses a string.
- Without `skip_nulls` every row from the first null on is null; with it a null row is null and the accumulation goes on.
- The running value is carried across chunks and the chunks are kept; a scalar gives a one-row array (Arrow).
- Integer overflow wraps (Arrow C++) or saturates with `overflow_mode` saturating; floats never fail.

Evidence: `src/compute/cumulative.zig:62`, `src/compute/cumulative.zig:57`, `src/compute/test_cumulative.zig:"cumulative_sum, cumulative_prod, cumulative_min, cumulative_max and cumulative_mean: options, nulls, NaN, overflow (pyarrow 26 results)"`

## cumulative_prod_checked

As [`cumulative_prod`](#cumulative_prod), but integer overflow returns `Overflow`.

Evidence: `src/compute/cumulative.zig:63`, `src/compute/cumulative.zig:57`, `src/compute/test_cumulative.zig:"cumulative_sum, cumulative_prod, cumulative_min, cumulative_max and cumulative_mean: options, nulls, NaN, overflow (pyarrow 26 results)"`

## cumulative_sum

`cumulative_sum` keeps a running sum over int8..int64, uint8..uint64, float32/64 (pyarrow 26 has no other input type) and returns the input type.

- Options: `CumulativeOptions` (`start` default none, `skip_nulls` default false). `start` is a numeric scalar cast safely to the input type (1000 for int8 or 1.5 for an integer is `InvalidCast`); a non-numeric start is `InvalidOptions`, where Arrow parses a string.
- Without `skip_nulls` every row from the first null on is null; with it a null row is null and the accumulation goes on.
- The running value is carried across chunks and the chunks are kept; a scalar gives a one-row array (Arrow).
- Integer overflow wraps (Arrow C++) or saturates with `overflow_mode` saturating; floats never fail.

Evidence: `src/compute/cumulative.zig:60`, `src/compute/cumulative.zig:57`, `src/compute/test_cumulative.zig:"cumulative_sum, cumulative_prod, cumulative_min, cumulative_max and cumulative_mean: options, nulls, NaN, overflow (pyarrow 26 results)"`

## cumulative_sum_checked

As [`cumulative_sum`](#cumulative_sum), but integer overflow returns `Overflow`.

Evidence: `src/compute/cumulative.zig:61`, `src/compute/cumulative.zig:57`, `src/compute/test_cumulative.zig:"cumulative_sum, cumulative_prod, cumulative_min, cumulative_max and cumulative_mean: options, nulls, NaN, overflow (pyarrow 26 results)"`

## winsorize

`winsorize` clamps values to quantile bounds: a value below the lower or above the upper quantile takes that quantile (quantile's `nearest` interpolation over the valid non-NaN values, ties to the even index).

- Options: `WinsorizeOptions` (`lower_limit`, `upper_limit`), required as in Arrow; 0 <= lower_limit <= upper_limit <= 1, else `InvalidOptions`.
- NaN and nulls stay; an all-null or all-NaN input comes back as is; chunks are kept (pyarrow 26 results).

Evidence: `src/compute/cumulative.zig:81`, `src/compute/cumulative.zig:57`, `src/compute/test_cumulative.zig:"winsorize: nearest quantile limits, NaN and nulls, decimals, chunked, options (pyarrow 26 results)"`

## pairwise_diff

`pairwise_diff` returns `x[i] - x[i - period]` through `subtract`.

- Options: `PairwiseOptions.period` (default 1); a negative period looks ahead, and |period| at least the length gives all null.
- Numbers keep their type (integer overflow wraps); temporal types give durations (date32: seconds, date64: milliseconds); decimal128(p, s) gives decimal128(p + 1, s).
- The null type gives a null-type result of its length (Arrow's null kernel); a null-type scalar gives a one-row null-type array.
- A chunked input is `InvalidInput`, as Arrow (pyarrow 26: "Vector kernel cannot execute chunkwise and no chunked exec function was defined"), also for the null type.

Evidence: `src/compute/cumulative.zig:73`, `src/compute/cumulative.zig:57`, `src/compute/test_arithmetic.zig:"arithmetic, math, rounding, bit-wise and pairwise_diff functions on null-type arguments give a null-type result: arrays, sliced, chunked, scalars (pyarrow 26)"`, `src/compute/test_cumulative.zig:"pairwise_diff and pairwise_diff_checked: periods, overflow, temporal and decimal types, chunked (pyarrow 26 results)"`

## pairwise_diff_checked

As [`pairwise_diff`](#pairwise_diff), but through `subtract_checked`: integer overflow returns `Overflow`.

Evidence: `src/compute/cumulative.zig:73`, `src/compute/cumulative.zig:57`, `src/compute/test_cumulative.zig:"pairwise_diff and pairwise_diff_checked: periods, overflow, temporal and decimal types, chunked (pyarrow 26 results)"`
