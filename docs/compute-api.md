# Compute API

This document describes the compute framework API in `zarrow.compute`.

It is for anyone adding kernels to this repository (core and compute live together, see [repo-layout.md](repo-layout.md)) and for applications that register and invoke them.

Chinese version: [compute-api-zh.md](compute-api-zh.md)

---

## 1. Scope and Boundary

`zarrow` provides the **compute framework layer**:

- function registration and dispatch;
- signature matching (arity/type/options);
- result type inference hooks;
- execution context and aggregate lifecycle abstractions;
- common helper utilities for array/scalar/chunked execution;
- chunk-aware permutation helpers.

Concrete kernels live in this repository under `src/compute/`, one file per family. `FunctionRegistry.init`
returns an empty registry; each family exposes `register(registry)`, and `registerAll(registry)`
(`compute.registerAll`, `zarrow.computeRegisterAll`) registers every family, the equivalent of
`arrow::compute::Initialize()`:

```zig
var registry = zarrow.compute.FunctionRegistry.init(allocator);
defer registry.deinit();
try zarrow.compute.registerAll(&registry); // or compute.kernels.arithmetic.register(&registry), ...
var ctx = zarrow.compute.ExecContext.init(allocator, &registry);
var sum = try ctx.invoke("add", .scalar, &.{ lhs, rhs }, .none); // lhs, rhs: Datum
defer sum.release();
```

| Family | File | Functions |
|---|---|---|
| Arithmetic | `arithmetic.zig` | `add`, `subtract`, `multiply`, `divide`, `negate`, `abs` and their `_checked` variants (int8..int64, uint8..uint64, float32/64) |
| Comparisons | `comparison.zig` | `equal`, `not_equal`, `less`, `less_equal`, `greater`, `greater_equal` (numeric, bool, string, large_string, string_view) |
| Logical | `logical.zig` | `and`, `or`, `xor`, `and_not`, `invert`, `and_kleene`, `or_kleene` |
| Aggregations | `aggregate_basic.zig` | `sum`, `mean`, `min`, `max`, `min_max`, `count`, `count_all`, `any`, `all` |
| Categorizations | `is_null.zig` | `is_null`, `is_valid`, `true_unless_null` |
| Conversions | `cast.zig` | `cast` (`CastOptions.to_type`): numbers and bool among themselves, to and from text, string and binary layouts, date32 / date64 / timestamp units, dictionary decode and encode, the null type |
| Strings | `string.zig` | `ascii_upper`, `ascii_lower`, `utf8_upper`, `utf8_lower`, `ascii_swapcase`, `utf8_swapcase`, `ascii_capitalize`, `utf8_capitalize` (ASCII letters), `utf8_length`, `binary_length`, `ascii_trim`, `ascii_ltrim`, `ascii_rtrim`, `ascii_*trim_whitespace`, `utf8_*trim_whitespace`, `utf8_slice_codeunits`, `utf8_reverse`, `binary_join_element_wise`, `starts_with`, `ends_with`, `match_substring`, `string_is_ascii`, `ascii_is_alpha`, `ascii_is_decimal`, `ascii_is_space`, `utf8_is_alpha`, `utf8_is_digit`, `utf8_is_space` (string, large_string, string_view) |
| String transforms | `string_transform.zig` | `replace_substring`, `split_pattern`, `ascii_split_whitespace`, `utf8_split_whitespace` (`list<T>` result), `ascii_lpad`, `ascii_rpad`, `ascii_center`, `utf8_lpad`, `utf8_rpad`, `utf8_center`, `count_substring`, `find_substring`, `binary_repeat` (string and binary layouts; padding and whitespace splits on the string layouts), `binary_slice`, `binary_reverse` (binary layouts) |
| Temporal | `temporal.zig` | `year`, `month`, `day`, `day_of_week`, `day_of_year`, `iso_year`, `iso_week`, `us_week`, `week`, `iso_calendar`, `year_month_day`, `quarter`, `is_leap_year` (date32, date64, timestamp); `hour`, `minute`, `second`, `millisecond`, `microsecond`, `nanosecond` (date64, timestamp); `subsecond` (timestamp) |
| Temporal rounding and differences | `temporal_round.zig` | `floor_temporal`, `ceil_temporal`, `round_temporal`; `years_between`, `quarters_between`, `month_interval_between`, `weeks_between`, `days_between`, `hours_between`, `minutes_between`, `seconds_between`, `milliseconds_between`, `microseconds_between`, `nanoseconds_between` (date32, date64, timestamp) |
| Set lookup | `set_lookup.zig` | `is_in`, `index_in` (numbers, bool, temporal, string and binary layouts) |
| Grouped aggregations | `hash_aggregate.zig` | `Grouper`, `groupBy`; kind `hash_aggregate`: `hash_sum`, `hash_mean`, `hash_min`, `hash_max`, `hash_min_max`, `hash_count`, `hash_count_all`, `hash_any`, `hash_all`, `hash_count_distinct`, `hash_distinct` (values plus uint32 group ids, one row per group) |

Element-wise functions take arrays, chunked arrays or scalars (a chunked input gives a chunked output, all
scalars give a scalar); aggregates take one array, chunked array or scalar and return a scalar; grouped
aggregates take a value array (or chunked array) and the uint32 group ids of a `Grouper` and return one
row per group:

```zig
const hash = zarrow.compute.kernels.hash_aggregate;
var batch = try hash.groupBy(&ctx, &.{.{ .name = "k", .array = keys }}, &.{
    .{ .function = "hash_sum", .input = values, .name = "total" },
    .{ .function = "hash_count_all", .name = "rows" },
});
defer batch.deinit(); // columns: k (one row per group, first-seen order), total, rows
```

Operands of a binary function must have the same type: there are no implicit casts (call `cast`). Exact semantics,
including the differences from Arrow C++, are in [compute-support-matrix.md](compute-support-matrix.md).

This repository also ships take/filter/select (`datumTake`, `datumFilter`,
`datumSelect`), concatenation (`concatArrayRefs`), slicing, `dictionaryEncode`, validation and null-like /
empty-like construction. [compute-support-matrix.md](compute-support-matrix.md) maps every category of the
Arrow C++ function catalog to its status here.

### Arrow C++ to zarrow glossary

| Arrow C++ | zarrow |
|---|---|
| `arrow::compute::Function` (name + kernels) | `Function` |
| `arrow::compute::Initialize()` (built-in function catalog) | `registerAll` (per family: `kernels.<family>.register`) |
| `ScalarKernel` / `VectorKernel` / `ScalarAggregateKernel` / `HashAggregateKernel` | `Kernel` with `FunctionKind.scalar` / `.vector` / `.aggregate` / `.hash_aggregate` |
| `Grouper`, `Table::GroupBy` | `kernels.hash_aggregate.Grouper`, `kernels.hash_aggregate.groupBy` |
| `KernelSignature` (`InputType`, `OutputType`, type matchers) | `KernelSignature` (`arity`, `type_check`, `result_type_fn`) |
| `FunctionOptions` | `Options` (tagged union, section 3) |
| `ExecContext` + `MemoryPool` | `ExecContext` + `std.mem.Allocator` (section 6) |
| `Datum` | `Datum` (array, chunked, scalar) |
| `ExecBatch` / `ExecSpan` chunk-wise execution | exec chunk iterators (section 8) |
| `Status` / `Result<T>` | Zig error unions (`KernelError!T`, section 5) |

---

## 2. Core Types

Main public framework types:

- `FunctionRegistry`
- `KernelSignature`
- `Kernel`
- `ExecContext`
- `Datum`

`Datum` currently supports all three forms:

- `array: ArrayRef`
- `chunked: ChunkedArray`
- `scalar: Scalar`

Lifecycle helpers (`retain/release`) and type access (`dataType`) are implemented for all three variants.

---

## 3. Typed Options Model

Compute options are type-safe and tag-based.

### Options families

`OptionsTag`:

- `none`
- `cast`
- `arithmetic`
- `filter`
- `sort`
- `day_of_week`
- `match_substring`
- `slice`
- `join`
- `trim`
- `replace_substring`
- `split_pattern`
- `split`
- `pad`
- `round_temporal`
- `week`
- `set_lookup`
- `custom`

### Payload types

- `CastOptions` (`to_type`, `safe`)
- `ArithmeticOptions`
- `FilterOptions`
- `SortOptions`
- `DayOfWeekOptions` (`count_from_zero`, `week_start`)
- `MatchSubstringOptions` (`pattern`, `ignore_case`)
- `SliceOptions` (`start`, `stop`, `step`)
- `JoinOptions` (`null_handling`: `emit_null | skip | replace`, `null_replacement`)
- `TrimOptions` (`characters`)
- `ReplaceSubstringOptions` (`pattern`, `replacement`, `max_replacements`)
- `SplitPatternOptions` (`pattern`, `max_splits`, `reverse`), `SplitOptions` (`max_splits`, `reverse`)
- `PadOptions` (`width`, `padding`, `lean_left_on_odd_padding`)
- `RoundTemporalOptions` (`multiple`, `unit`: `CalendarUnit` nanosecond .. year, `week_starts_monday`,
  `ceil_is_strictly_greater`, `calendar_based_origin`)
- `WeekOptions` (`week_starts_monday`, `count_from_zero`, `first_week_is_fully_in_year`)
- `SetLookupOptions` (`value_set`: a borrowed array or chunked array `Datum`, `null_matching_behavior`:
  `match | skip | emit_null | inconclusive`)
- `CustomOptions`

Kernels whose options are optional in Arrow (`day_of_week`, `binary_join_element_wise`, `week`,
`weeks_between`, the whitespace splits and the `*_temporal` rounding functions) also accept `.none` for
the defaults.

Use `KernelSignature.options_check` to validate allowed option families for each kernel.

### SortOptions

`SortOptions` currently includes:

- `order`: `ascending | descending`
- `null_placement`: `at_start | at_end`
- `nan_placement`: `?at_start | ?at_end`
- `stable`: `bool`

---

## 4. Signature and Type Inference

`KernelSignature` supports:

- exact arity;
- variadic minimum arity (`at_least`);
- bounded arity range (`range`);
- optional `type_check` callback;
- optional `options_check` callback;
- optional `result_type_fn` callback.

Result type APIs:

- `KernelSignature.inferResultType(args, options)`
- `FunctionRegistry.resolveResultType(...)`

When signature or options checks fail, errors are reported through `KernelError`.

---

## 5. Error Model

`KernelError` includes framework and execution-level categories such as:

- `InvalidArity`
- `InvalidOptions`
- `InvalidInput`
- `UnsupportedType`
- `InvalidCast`
- `Overflow`
- `DivideByZero`
- `NoMatchingKernel`

Helpful utility helpers are provided for common arithmetic/cast behavior:

- `intCastOrInvalidCast(...)`
- `arithmeticDivI64(...)`

Error conventions, mirroring Arrow C++ (`Status`, `Result<T>`, no exceptions across the API):

- every fallible kernel entry point returns an error union; a kernel never panics on input it was given
  through the public API, including malformed arrays (validation errors are returned, not asserted);
- `std.debug.assert` is reserved for internal invariants that cannot fail on valid input (Arrow's `DCHECK`),
  and is stripped in release builds, so it must not guard input;
- `OutOfMemory` passes through unchanged; kernels do not retry or fall back silently;
- an unsupported type is `UnsupportedType`, a malformed value is `InvalidInput`, and an arity or option
  mismatch is reported through the signature errors above, so callers can distinguish "cannot" from
  "will not".

---

## 6. ExecContext and Config

`ExecConfig` currently includes:

- `safe_cast`: `cast` is safe only when both `safe_cast` and `CastOptions.safe` are true
- `overflow_mode` (`checked | wrapping | saturating`): the unchecked arithmetic functions (`add`, ...) wrap
  like Arrow C++ under `checked` and `wrapping` and saturate under `saturating` (the `_checked` names always
  return `Overflow`); `sum`, which has no `_checked` twin, returns `Overflow` under `checked`, wraps under
  `wrapping` and clamps under `saturating`
- `threads`
- `arena_allocator`

Context construction:

- `ExecContext.init(allocator, &registry)`
- `ExecContext.initWithConfig(allocator, &registry, config)`

For borrowed scalar string/binary data, use:

- `ExecContext.dupScalarString(...)`
- `ExecContext.dupScalarBinary(...)`

Memory conventions (Arrow's `default_memory_pool()` has no implicit counterpart here):

- every allocation is made through the allocator the caller passed (`ExecContext.allocator`, or the datum's
  allocator through `inferDatumAllocator`); kernels never reach for a global allocator;
- `ExecContext.tempAllocator()` is for scratch that dies with the call (uses `arena_allocator` when set);
- output buffers are built with `OwnedBuffer` and handed over with `toShared`; a kernel that produces an array
  returns it with `ArrayRef.fromOwnedInlineBuffers` so node and buffers are one allocation;
- builders may keep a spare buffer from a previous batch, but reuse it only once `SharedBuffer.reclaim`
  shows no live array references it (nothing may alias a buffer a live `ArrayRef` still holds).

---

## 7. Aggregate Lifecycle Interface

For stateful aggregate kernels:

- `AggregateLifecycle.init`
- `AggregateLifecycle.update`
- `AggregateLifecycle.merge`
- `AggregateLifecycle.finalize`
- `AggregateLifecycle.deinit`

Runtime flow:

1. register kernel with `aggregate_lifecycle`;
2. call `ctx.beginAggregate(...)` to create `AggregateSession`;
3. call `update/merge/finalize/deinit`.

If lifecycle is missing, `beginAggregate` returns `error.MissingLifecycle`. `beginAggregate` also
resolves the `hash_*` functions (kind `hash_aggregate`): `update` takes (values, group ids) batches,
`merge` adds a state over the same group ids, `finalize` returns one row per group seen so far.

---

## 8. Execution and Chunk-Aware Helpers

Framework helpers include:

- null propagation: `unaryNullPropagates`, `binaryNullPropagates`, `naryNullPropagates`;
- scalar broadcast length inference: `inferBinaryExecLen`, `inferNaryExecLen`;
- chunk-aligned iterators: `UnaryExecChunkIterator`, `BinaryExecChunkIterator`, `NaryExecChunkIterator`;
- `dense.execElementwise` / `dense.execElementwiseWith` (any arity, options passed to every chunk): the
  driver of the element-wise families (array / chunked / scalar dispatch, `validateLayout` of every input
  chunk, null scalars under intersection), with `dense.Bits` / `dense.mapBits` for packed-bit operands,
  `dense.StringOperand` for the rows of the string and binary layouts, `dense.VarBinaryOutput` for a
  string / large_string / string_view output (offsets checked against the 32-bit range) and
  `dense.chunkArray` to read a scalar operand as a one-row array. A string or binary scalar result owns
  its bytes through `Scalar.payload`.

Permutation-family helpers include:

- `chunkedResolveLogicalIndices`
- `datumTake`, `datumTakeNullable`
- `datumFilterSelectionIndices`
- `datumFilterChunkAware`
- `datumSelect`, `datumSelectNullable`

These are intended to help kernels preserve chunked execution/output where possible, instead of pre-concatenating chunked inputs.

### Null handling

Arrow C++ kernels declare a `NullHandling` policy; here the policy is a choice of helper:

- **intersection** (the default for element-wise kernels): combine input validity once, bitmap word by
  word, then run the value loop over plain typed slices with no per-element null check. Null slots may
  compute garbage that the output validity hides. Use `dense.DenseBinaryOutput`, `dense.validityBits` and
  `dense.laneMask`; `unaryNullPropagates` / `binaryNullPropagates` / `naryNullPropagates` answer the
  question for scalar inputs.
- **computed**: the kernel writes its own validity (comparisons with Kleene logic, `is_null`-style
  categorizations). Write the bitmap explicitly; do not derive it per row with `isNull`.
- **output not null**: aggregates and count-like kernels return a scalar; use `Scalar.init`.

Per-element `isNull` calls inside a hot loop are the one pattern to avoid: they were measured at 4 to
15 ns per row against under 1 ns for the dense path.

### Validity at trust boundaries

Kernels may receive arrays that came from IPC or the C data interface. They must bounds-check what they
read (buffer counts, offsets windows, index ranges, dictionary lengths) and return `InvalidInput` rather than
trust `length` or `null_count`. The selection and concat kernels in this repository are the reference for
how much checking is expected; nothing in a kernel may weaken what `validateLayout` establishes.
`validateLayout` does not check UTF-8 or dictionary indices: the string kernels that decode code points
reject a non-null row that is not UTF-8 (`InvalidInput`), a cast to a string type runs the UTF-8 check of
`validateFull` on a binary source, and `cast` checks dictionary indices before decoding.

---

## 9. Recommended Extension Path

For new sort/take/filter kernels:

1. keep one logical behavior for both array and chunked inputs;
2. drive behavior through typed options (especially `SortOptions`), not hardcoded defaults;
3. reuse core chunk-aware helpers for index resolution, null handling, and gather/filter execution;
4. test array/chunked logical equivalence, including misaligned chunk-boundary cases.

### Adding a function, step by step

This follows the Arrow C++ developer workflow (one function, its kernels, options, tests, benchmark, docs)
translated to this repository's layout. Kernels are added here, not in a separate repository.

1. **Name it like Arrow.** Use the Arrow C++ function name (`add`, `less`, `sum`, `sort_indices`), with
   the `_checked` suffix for the overflow-checking variant and `hash_` prefix for grouped aggregates, so
   users can read the Arrow reference. Options types end in `Options`.
2. **Place the code.** One file per family, underscores in file names (`arithmetic.zig`,
   `comparison.zig`, `aggregate_basic.zig`); tests in `test_<family>.zig` next to it; a benchmark in
   `benchmarks/<family>_benchmark.zig`. Public symbols are re-exported from the family's `mod.zig`;
   anything not re-exported is internal.
3. **Write the signature first.** `KernelSignature.unary` / `.binary` / `.atLeast` with a `type_check`
   that accepts exactly the types the kernel handles (`unaryNumeric`, `binarySameNumeric`, or a custom
   check) and a `result_type_fn` when the output type differs from the input.
4. **Pick the kernel kind and options family.** `registerScalarKernel` for element-wise,
   `registerVectorKernel` for whole-array (sort, unique, cumulative), `registerAggregateKernel` with an
   `AggregateLifecycle` for reductions. Declare the options family the kernel reads
   (`options_check`); read `ExecContext.overflowMode()` and `safeCastEnabled()` instead of re-deriving them.
5. **Execute chunk-wise.** Iterate with the exec chunk iterators so a chunked input yields a chunked
   output; never concatenate inputs first. Use the dense helpers for validity (section 8).
6. **Return errors, do not assert.** See section 5. Overflow in `checked` mode is `Overflow`, division by
   zero is `DivideByZero`, a narrowing cast that loses data in safe mode is `InvalidCast`.
7. **Test.** For every kernel: a hand-written expected-value test; an oracle test comparing the kernel
   against a simple per-element loop on random data with nulls and sliced inputs (offset != 0); array
   versus chunked equivalence with chunk boundaries that do not align; empty input; all-null input; the
   error cases of step 6. Register the test file in the family's `test.zig`.
8. **Benchmark.** One case per kernel and type at 200k rows in `benchmarks/<family>_benchmark.zig`,
   printing the common CSV line; allocate through `benchmarks/bench_allocator.zig` and make one untimed
   warm-up call so output memory is reused across iterations (as `compute_benchmark.zig` does); register
   the case name in `tools/benchmark_ci_check.py` (`EXPECTED_BENCHES`), `tools/benchmark_ci_thresholds.json`
   and `tools/test_benchmark_ci_check.py` (`VALID_ROWS`) so the CI smoke run covers it.
9. **Document.** Add a `///` docstring on the public entry point (summary line in the infinitive, as in
   Arrow: "Add two numeric arrays", not "Adds"), and flip the function's row in
   [compute-support-matrix.md](compute-support-matrix.md).

### Source conventions

Taken from the Arrow C++ [conventions](https://arrow.apache.org/docs/developers/cpp/conventions.html) where
they map onto Zig:

- file names use underscores; test files are `test_*.zig`; benchmark executables are `<name>_benchmark`;
- `///` docstrings on every `pub` declaration, summary in the infinitive; `//` for ordinary comments;
- no hidden work in constructors: anything that can fail or allocate is a `pub fn init(...) !T` factory;
- public functions take an explicit `std.mem.Allocator` or an `ExecContext`, never a global;
- errors are returned, invariants are asserted (section 5);
- code must build on both supported Zig versions (see the README), so no builtins or std APIs that exist in
  only one of them.

---

## 10. Diagnostics

Readable mismatch diagnostics are available:

- `KernelSignature.explainMismatch`
- `KernelSignature.explainInferResultTypeFailure`
- `FunctionRegistry.explainResolveKernelFailure`
- `FunctionRegistry.explainResolveResultTypeFailure`

---

## 11. References

- Example: [`examples/compute_lifecycle.zig`](../examples/compute_lifecycle.zig)
- Kernel family benchmarks: [`benchmarks/compute_benchmark.zig`](../benchmarks/compute_benchmark.zig)
- Support status against the Arrow catalog: [compute-support-matrix.md](compute-support-matrix.md)
- Readiness note: [ecosystem-readiness.md](ecosystem-readiness.md)
- Arrow C++: [compute functions](https://arrow.apache.org/docs/cpp/compute.html),
  [compute tutorial](https://arrow.apache.org/docs/cpp/tutorials/compute_tutorial.html),
  [developer conventions](https://arrow.apache.org/docs/developers/cpp/conventions.html)
