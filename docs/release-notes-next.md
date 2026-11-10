# Release Notes Draft (v0.0.2)

## Summary

`v0.0.2` focuses on interoperability completeness and C ABI robustness:

- expanded Arrow Go IPC interop coverage and docs;
- introduced and documented C ABI smoke/readiness workflow;
- moved C Stream interop to true streaming transfer (no full-stream materialization);
- improved interop error handling and cross-ecosystem smoke reliability.

## New Features

- Implicit casts check for a kernel before casting (Arrow's DispatchBest): `binary_repeat` on a
  dictionary of a type without a kernel is `NoMatchingKernel`, no longer a cast error.
- Added `zarrow.c_api` export for direct C ABI surface access from package root.
- Added `examples/c_abi_roundtrip.zig` for end-to-end stream roundtrip via C ABI.
- IPC stream and file readers read big-endian IPC: every body buffer is byte-swapped to
  little-endian and the schema is presented as little-endian (Arrow C++ `ensure_native_endian`);
  all 22 `1.0.0-bigendian` gold cases now pass ([docs/integration-suite.md](integration-suite.md)).
- Compute, closing the per-type gaps against pyarrow 26
  ([docs/compute-support-matrix.md](compute-support-matrix.md)):
  - every arithmetic, math, bit-wise and rounding function (and `_checked` variant) and
    `pairwise_diff` take null-type arguments and return a null-type result of their length
    (`FunctionRegistry.setNullTypeResult`); `sum` / `product` / `mean` of the null type give int64 /
    int64 / float64; `sort_indices` / `array_sort_indices` of the null type give 0 .. n-1;
  - `is_null`, `is_valid`, `true_unless_null` and `count` take unions and run-end encoded arrays, with
    Arrow's logical nulls (`is_null.logicalValidity`);
  - `first`, `last`, `first_last` on fixed_size_binary; `sum`, `mean`, `product` on bool;
  - `floor_temporal`, `ceil_temporal`, `round_temporal`, `is_finite`, `is_inf`, `is_nan` and `sign` on
    duration;
  - `round_binary` on decimal128 / decimal256 with decimal (or float) digits;
  - `run_end_encode` / `run_end_decode` on list, large_list, list_view, large_list_view,
    fixed_size_list, struct and map (rows compared by `integration/compare.zig`'s `rowsEqual`);
  - `binary_join_element_wise` on binary and large_binary (any `null_replacement` bytes);
  - `binary_repeat` with counts of any integer type (array or scalar) and on dictionary<string>
    (decoded first, string result);
  - null-type arguments in `if_else`, `choose` (null-type result), `is_in` / `index_in` (all null
    rows: `is_in(null, value_set = nulls)` is all true, `index_in` all 0) and
    `partition_nth_indices` (0 .. n-1 for any pivot); `choose` takes a bool index as 0 / 1;
  - `count_substring_regex` / `find_substring_regex` on fixed_size_binary;
  - `round_binary` with bool digits (cast to int32).
- Compute behaviour now matching pyarrow 26 (each a change from the previous zarrow result):
  - `tdigest`, `approximate_median`'s options and `hash_tdigest`: a q outside [0, 1] (or NaN) gives
    a NaN quantile instead of `InvalidOptions`;
  - `pairwise_diff` / `pairwise_diff_checked` reject a chunked input with `InvalidInput` (Arrow:
    "Vector kernel cannot execute chunkwise") instead of concatenating it; a null-type scalar gives a
    one-row null-type array;
  - `run_end_encode` on nested values: a NaN float item never equals another row, so `[[nan], [nan]]`
    is two runs (`integration/compare.zig` `rowsEqualNanUnequal`);
  - `hash_min`, `hash_max`, `hash_min_max` no longer apply `min_count` (Arrow's grouped min / max
    ignore it; the scalar forms still apply it).
  - hash-based kernels (group keys, `hash_count_distinct`, `hash_distinct`, `count_distinct`,
    `unique`, `value_counts`, `dictionary_encode`, `is_in`, `index_in`) compare floats by their bits
    as pyarrow 26 hashes them: NaN with another sign or payload is another value (NaN was folded
    before); `mode` and `index` compare by value (every NaN one value, -0.0 equal to 0.0).
- Added C ABI interop guides:
  - `docs/interop-c-abi.md`
  - `docs/interop-c-abi-zh.md`

## Bug Fixes and Reliability

- Fixed C ABI smoke integration with Arrow C++.
- Fixed arrow-rs interop binary invocation ambiguity (`cargo run --bin ...`).
- Improved Go/Rust interop tooling and CI smoke consistency.
- Strengthened error-path handling in array/value/interop validation paths.

## Breaking Changes

- `IpcStreamReaderOptions` / `IpcFileReader` `ReaderOptions.endianness_mode` now defaults to
  `.normalize_to_little` (byte-swap big-endian input); pass `.strict` to keep rejecting it.
- `zarrow_c_export_stream` now requires mutable handle:
  - from: `const zarrow_c_stream_handle*`
  - to: `zarrow_c_stream_handle*`
- IPC readers/writers now use Zig 0.17 `std.Io` interfaces:
  - `IpcStreamReader` / `IpcStreamWriter` / `IpcFileWriter` take `*std.Io.Reader` / `*std.Io.Writer`
    (were generic over a reader/writer type; now plain struct types, no `Ipc*(T)` call);
  - `IpcFileReader` takes `[]const u8` (load or mmap the whole file; it no longer reads from a stream);
  - writer failures surface as `error.WriteFailed` (check the underlying writer for the cause).

## Migration Guidance

- Regenerate/update downstream FFI bindings for `zarrow_c_export_stream`.
- Update callers to pass mutable stream handles.
- Do not reuse a stream handle after successful export:
  - ownership is transferred;
  - repeated export returns `ZARROW_C_STATUS_RELEASED`.
- IPC migration:
  - replace `zarrow.IpcStreamReader(T).init(a, r)` with `zarrow.IpcStreamReader.init(a, &reader)`
    where `reader: std.Io.Reader` (e.g. `.fixed(bytes)`);
  - replace `zarrow.IpcStreamWriter(T).init(a, w)` with `zarrow.IpcStreamWriter.init(a, w_ptr)`
    where `w_ptr: *std.Io.Writer` (e.g. `&allocating.writer` or `&file_writer.interface`); `writeEnd` flushes;
  - for `IpcFileReader`, pass the file bytes: `zarrow.IpcFileReader.init(a, bytes)`;
  - treat `error.WriteFailed` from a writer as an I/O failure and inspect the sink for details;
  - bytes passed to `IpcFileReader.init` must outlive the reader;
  - only `writeEnd` flushes the sink; streaming consumers (sockets/pipes) must call `flush()` on their writer between messages;
  - reader errors now include `error.ReadFailed`;
  - after any write error, `IpcFileWriter` returns `error.InvalidState` for later writes; create a new writer to retry.

## Compatibility Notes

- Minimum Zig: `0.16.0` (0.16.x and 0.17.x supported).
- Arrow Go IPC:
  - main lane targets Go `1.24.x`;
  - version-smoke lane validates Go `1.23.x` and `1.24.x`.
- C ABI smoke lanes:
  - `interop-c-abi-cpp-smoke`
  - `interop-c-abi-rs-smoke`
