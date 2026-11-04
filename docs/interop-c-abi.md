# Arrow C ABI Interop Guide

This guide explains how `zarrow_c` interoperates with the Arrow C Data / C Stream ABI and how to run local smoke checks.

## Scope

Public C header:

- `include/zarrow_c_api.h`

Supported object boundaries:

- `ArrowSchema`
- `ArrowArray` (with imported schema handle)
- `ArrowArrayStream`

## Capabilities

Evidence follows the rule in [`CLAUDE.md`](../CLAUDE.md): every `yes` / `partial` row cites a source reference
and a test reference, checked by `python -I tools/check_doc_evidence.py`. A CI job is cited by its line in
`.github/workflows/ci.yml`; a row backed only by a CI job (no Zig test) is `partial`, marked "CI job only".
Which data types cross the C data interface is tracked per type in the C ABI column of
[type-coverage.md](type-coverage.md).

| Capability | zarrow | Status | Source | Tests |
|---|---|---|---|---|
| Schema import / export through the C API | `zarrow_c_import_schema`, `zarrow_c_export_schema` | yes | `src/c_api.zig:74`, `src/c_api.zig:90` | `src/c_api.zig:"c api schema and array import/export roundtrip"` |
| Array import / export through the C API (with an imported schema handle) | `zarrow_c_import_array`, `zarrow_c_export_array` | yes | `src/c_api.zig:105`, `src/c_api.zig:133` | `src/c_api.zig:"c api schema and array import/export roundtrip"` |
| Stream import / export without materializing batches (callbacks forwarded lazily) | `zarrow_c_import_stream`, `zarrow_c_export_stream` | yes | `src/c_api.zig:149`, `src/c_api.zig:169` | `src/c_api.zig:"c api stream import/export keeps callbacks and avoids materialization"` |
| Stream export is one-shot: a second export returns `ZARROW_C_STATUS_RELEASED` | `zarrow_c_export_stream` | yes | `src/c_api.zig:169` | `src/c_api.zig:"c api stream import/export keeps callbacks and avoids materialization"` |
| Null and released arguments map to `INVALID_ARGUMENT` / `RELEASED` status codes | C API entry points | yes | `src/c_api.zig:149`, `src/c_api.zig:169` | `src/c_api.zig:"c api invalid-argument and invalid-data paths"` |
| Status code to string | `zarrow_c_status_string` | yes | `src/c_api.zig:62` | `src/c_api.zig:"c api status string smoke"` |
| Handle release functions | `zarrow_c_release_schema`, `zarrow_c_release_array`, `zarrow_c_release_stream` | partial (called by the tests; the C API uses `c_allocator`, so a leak would not fail them) | `src/c_api.zig:99`, `src/c_api.zig:142`, `src/c_api.zig:181` | `src/c_api.zig:"c api schema and array import/export roundtrip"`, `src/c_api.zig:"c api stream import/export keeps callbacks and avoids materialization"` |
| ABI version query | `zarrow_c_abi_version` | partial (exported, no test calls it) | `src/c_api.zig:58` | `build.zig:"c-api-lib"` |
| Zig API: `ArrowSchema` export / import | `exportSchema`, `importSchemaOwned` | yes | `src/ffi/c_data.zig:120`, `src/ffi/c_data.zig:126` | `src/ffi/c_data.zig:"c data schema export/import roundtrip"` |
| Zig API: `ArrowArray` export / import, zero-copy | `exportArray`, `importArray` | yes | `src/ffi/c_data.zig:153`, `src/ffi/c_data.zig:157` | `src/ffi/c_data.zig:"c data array export/import zero-copy smoke"` |
| Zig API: `ArrowArrayStream` export / import | `exportRecordBatchStream`, `importRecordBatchStreamOwned` | yes | `src/ffi/c_stream.zig:61`, `src/ffi/c_stream.zig:100` | `src/ffi/c_stream.zig:"c stream export callbacks yield schema, batch, and eos"`, `src/ffi/c_stream.zig:"c stream import drains stream into owned record batches"` |
| Released input structs rejected on import | `importSchemaOwned`, `importArray` | yes | `src/ffi/c_data.zig:126`, `src/ffi/c_data.zig:157` | `src/ffi/c_data.zig:"c data import schema rejects released input"`, `src/ffi/c_data.zig:"c data import array rejects released input"` |
| Non-little-endian schema rejected on export | `exportSchema` | yes | `src/ffi/c_data.zig:120` | `src/ffi/c_data.zig:"c data export schema rejects non-little-endian"` |
| Overflowing buffer lengths rejected on import | `importArray` | yes | `src/ffi/c_data.zig:157` | `src/ffi/c_data.zig:"c data import array rejects overflowing required buffer length"` |
| Extension name and metadata carried across the interface | field metadata keys | yes | `src/ffi/c_data.zig:1225`, `src/ffi/c_data.zig:1303` | `src/ffi/c_data.zig:"c data schema export/import roundtrip preserves extension metadata"` |
| Per-type format strings and buffers | `formatFromDataType`, `importDataType` | partial (format strings for every type; array tests for some types, see [type-coverage.md](type-coverage.md)) | `src/ffi/c_data.zig:1087`, `src/ffi/c_data.zig:507` | `src/ffi/c_data.zig:"c data export emits canonical temporal and decimal formats"`, `src/ffi/c_data.zig:"c data array import supports run-end-encoded"` |
| Zig round-trip example, run in CI by `zig build examples` (`.github/workflows/ci.yml:110`) | `examples/c_abi_roundtrip.zig` | yes | `examples/c_abi_roundtrip.zig:24`, `.github/workflows/ci.yml:110` | `build.zig:"examples"` |
| Arrow C++ smoke (`zarrow_c` <-> Arrow C++ 23.0.1) | `tools/interop/cpp/c_abi_smoke.cpp`, CI job `interop-c-abi-cpp-smoke` | partial (CI job only) | `tools/interop/cpp/c_abi_smoke.cpp:172`, `.github/workflows/ci.yml:871` | `build.zig:"c-api-lib"` (the library the job links; no Zig test runs the C++ side) |
| arrow-rs smoke (`zarrow_c` <-> arrow-rs) | `tools/interop/arrow-rs/src/bin/c_abi_smoke.rs`, CI job `interop-c-abi-rs-smoke` | partial (CI job only) | `tools/interop/arrow-rs/src/bin/c_abi_smoke.rs:180`, `.github/workflows/ci.yml:921` | `build.zig:"c-api-lib"` (the library the job links; no Zig test runs the Rust side) |

## Streaming Semantics

`zarrow_c_import_stream` and `zarrow_c_export_stream` are streaming-preserving operations:

- no full materialization into an in-memory batch list;
- original `ArrowArrayStream` callbacks are transferred and forwarded lazily;
- this avoids large memory spikes for long streams.

## Ownership Rules

- `zarrow_c_import_*` transfers ownership of input Arrow C structs into a zarrow handle.
- `zarrow_c_export_*` transfers ownership from handle to output Arrow C structs.
- release zarrow handles with:
  - `zarrow_c_release_schema`
  - `zarrow_c_release_array`
  - `zarrow_c_release_stream`
- release exported Arrow C structs using their own `release` callback.

For stream handles, export is one-shot: exporting again after ownership is moved out returns `ZARROW_C_STATUS_RELEASED`.

## Local Smoke Checks

Run from repository root.

### 1) Build shared library

```bash
zig build c-api-lib
```

### 2) Zig roundtrip example

```bash
zig build example-c_abi_roundtrip
```

### 3) Rust smoke (`zarrow_c <-> arrow-rs`)

```bash
export LD_LIBRARY_PATH="$PWD/zig-out/lib:${LD_LIBRARY_PATH:-}"
export RUSTFLAGS="-L native=$PWD/zig-out/lib"
cargo run --manifest-path tools/interop/arrow-rs/Cargo.toml --bin c_abi_smoke
```

### 4) C++ smoke (`zarrow_c <-> Arrow C++`)

Requires Arrow C++ development environment and `pkg-config` entry for `arrow`.

```bash
mkdir -p .interop-bin
c++ -std=c++17 tools/interop/cpp/c_abi_smoke.cpp \
  -Iinclude \
  -Lzig-out/lib \
  -Wl,-rpath,"$PWD/zig-out/lib" \
  -lzarrow_c \
  -o .interop-bin/c_abi_smoke \
  $(pkg-config --cflags --libs arrow)

export LD_LIBRARY_PATH="$PWD/zig-out/lib:${LD_LIBRARY_PATH:-}"
./.interop-bin/c_abi_smoke
```

## CI Lanes

In `.github/workflows/ci.yml`:

- `interop-c-abi-cpp-smoke`
- `interop-c-abi-rs-smoke`

## Troubleshooting

`cargo run could not determine which binary to run`:

- use `--bin c_abi_smoke` or `--bin zarrow-interop-arrow-rs`.

Shared library not found at runtime:

- ensure `LD_LIBRARY_PATH` includes `$PWD/zig-out/lib`.
