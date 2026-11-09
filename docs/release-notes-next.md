# Release Notes Draft (v0.0.2)

## Summary

`v0.0.2` focuses on interoperability completeness and C ABI robustness:

- expanded Arrow Go IPC interop coverage and docs;
- introduced and documented C ABI smoke/readiness workflow;
- moved C Stream interop to true streaming transfer (no full-stream materialization);
- improved interop error handling and cross-ecosystem smoke reliability.

## New Features

- Added `zarrow.c_api` export for direct C ABI surface access from package root.
- Added `examples/c_abi_roundtrip.zig` for end-to-end stream roundtrip via C ABI.
- IPC stream and file readers read big-endian IPC: every body buffer is byte-swapped to
  little-endian and the schema is presented as little-endian (Arrow C++ `ensure_native_endian`);
  all 22 `1.0.0-bigendian` gold cases now pass ([docs/integration-suite.md](integration-suite.md)).
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
