# Random: function notes

Up: [function matrix](matrix.md) · [compute support matrix](../../compute-support-matrix.md)

The one random-number function, a nullary scalar function as in Arrow (Arrow passes the length as the batch length).

## Summary

| Function | Arity | Input types | Output type | Options |
|---|---|---|---|---|
| [`random`](#random) | Nullary | none | float64 | `RandomOptions` |

## random

`random` returns `length` float64 values in [0, 1).

- Options: `RandomOptions` (`length` default 0, `initializer` `seed` (default) or `system`, `seed` default 0).
- `seed`: `std.Random.DefaultPrng` seeded with `seed`, deterministic per seed. The sequence differs from Arrow's generator, which Arrow does not specify either.
- `system`: seeds from the operating system through `ExecConfig.io` (`std.Io.random`); without an I/O source it is `NotImplemented`.

Evidence: `src/compute/math.zig:58`, `src/compute/test_math.zig:"random: seeded values in [0, 1), length, system initializer"`
