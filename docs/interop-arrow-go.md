# Arrow Go IPC Interop Guide

This document explains how to run zarrow <-> Arrow Go IPC interop locally and what coverage is enforced in CI.

## Scope

Interop tool path:

- `tools/interop/arrow-go`

Supported fixtures:

- `canonical`
- `dict-delta` (stream only)
- `ree`
- `ree-int16`
- `ree-int64`
- `complex`
- `extension`
- `view`

Supported containers:

- `stream`
- `file` (all fixtures except `dict-delta`)

## Coverage Matrix

Each fixture runs in both directions in the `interop-arrow-go` CI job: zarrow writes it and Arrow Go
validates it, then Arrow Go writes it and zarrow checks it (`zig build interop-fixture-check`). Evidence follows
the rule in [`CLAUDE.md`](../CLAUDE.md): every `yes` / `partial` row cites a source reference (the zarrow
fixture writer and checker, the Arrow Go runner and the CI step) and a test reference (the fixture-check build
step plus the zarrow unit test for the same shape), checked by `python -I tools/check_doc_evidence.py`.

| Fixture | Content | stream | file | Status | Source | Tests |
|---|---|---|---|---|---|---|
| canonical | `id: int32` (non-null), `name: utf8` (nullable), 3 rows | yes | yes | yes | `tools/interop_fixture_writer.zig:272`, `tools/interop_fixture_check.zig:247`, `tools/interop/arrow-go/main.go:312`, `tools/interop/arrow-go/main.go:576`, `.github/workflows/ci.yml:630`, `.github/workflows/ci.yml:649`, `.github/workflows/ci.yml:683` | `build.zig:"interop-fixture-check"`, `src/ipc/stream_reader.zig:"ipc stream roundtrip schema and batch"`, `src/ipc/file_reader.zig:"ipc file reader roundtrips batches via stream reader"` |
| dict-delta | `color: dictionary<int32, utf8>` with a delta dictionary batch | yes | no | yes (stream only by design) | `tools/interop_fixture_writer.zig:310`, `tools/interop_fixture_check.zig:273`, `tools/interop/arrow-go/main.go:335`, `tools/interop/arrow-go/main.go:617`, `.github/workflows/ci.yml:649` | `build.zig:"interop-fixture-check"`, `src/ipc/stream_reader.zig:"ipc reader merges dictionary delta batches"`, `src/ipc/stream_writer.zig:"ipc writer emits dictionary delta on append-only dictionary growth"` |
| ree | `run_end_encoded<int32 run ends, int32 values>`, 5 logical rows | yes | yes | yes | `tools/interop_fixture_writer.zig:227`, `tools/interop_fixture_check.zig:218`, `tools/interop/arrow-go/main.go:394`, `tools/interop/arrow-go/main.go:666`, `.github/workflows/ci.yml:649` | `build.zig:"interop-fixture-check"`, `src/ipc/stream_reader.zig:"ipc stream roundtrip run-end encoded int32 values"` |
| ree-int16 | as `ree` with int16 run ends | yes | yes | yes | `tools/interop_fixture_writer.zig:227`, `tools/interop_fixture_check.zig:218`, `tools/interop/arrow-go/main.go:140`, `.github/workflows/ci.yml:649` | `build.zig:"interop-fixture-check"`, `src/array/test_type_coverage.zig:"ipc stream and file round trip every type sample, full and sliced"` |
| ree-int64 | as `ree` with int64 run ends | yes | yes | yes | `tools/interop_fixture_writer.zig:227`, `tools/interop_fixture_check.zig:218`, `tools/interop/arrow-go/main.go:140`, `.github/workflows/ci.yml:649` | `build.zig:"interop-fixture-check"`, `src/array/test_type_coverage.zig:"ipc stream and file round trip every type sample, full and sliced"` |
| complex | `list<int32>`, `struct<id: int32, name: utf8>`, `map<int32, int32>`, `dense_union<int32, bool>`, `decimal128(10, 2)`, `timestamp(ms, UTC)` | yes | yes | yes | `tools/interop_fixture_writer.zig:382`, `tools/interop_fixture_check.zig:304`, `tools/interop/arrow-go/main.go:421`, `tools/interop/arrow-go/main.go:713`, `.github/workflows/ci.yml:649` | `build.zig:"interop-fixture-check"`, `src/ipc/stream_reader.zig:"stream reader zero-copy read equals copying read for nested, dictionary and compressed streams"`, `src/ipc/stream_reader.zig:"ipc stream roundtrip map int32 to int32"`, `src/ipc/stream_reader.zig:"ipc stream roundtrip dense union int32/bool"`, `src/ipc/stream_reader.zig:"ipc stream roundtrip temporal and decimal primitives"` |
| extension | `com.example.int32_ext` over int32, metadata `v1` | yes | yes | yes | `tools/interop_fixture_writer.zig:189`, `tools/interop_fixture_check.zig:187`, `tools/interop/arrow-go/main.go:501`, `tools/interop/arrow-go/main.go:865`, `.github/workflows/ci.yml:649` | `build.zig:"interop-fixture-check"`, `src/ipc/stream_writer.zig:"ipc writer and reader roundtrip extension field metadata and values"` |
| view | `utf8_view`, `binary_view` with inline and out-of-line values | yes | yes | yes | `tools/interop_fixture_writer.zig:154`, `tools/interop_fixture_check.zig:157`, `tools/interop/arrow-go/main.go:528`, `tools/interop/arrow-go/main.go:914`, `.github/workflows/ci.yml:649` | `build.zig:"interop-fixture-check"`, `src/ipc/stream_writer.zig:"ipc writer roundtrip supports view types and variadicBufferCounts"` |
| tensor | schema `id: int32` plus one `int32[2, 3]` tensor message; also pyarrow's `write_tensor` messages (row- and column-major, strided, empty, 0-dimensional, float16) | yes | no | yes (with PyArrow both ways: PyArrow reads zarrow's message with `ipc.read_tensor`, zarrow reads PyArrow's; Arrow Go has no tensor IPC and Arrow files have no tensor blocks) | `tools/interop_fixture_writer.zig:116`, `tools/interop_fixture_check.zig:131`, `tools/interop/pyarrow_generate.py:184`, `tools/interop/pyarrow_validate.py:261`, `.github/workflows/ci.yml:331`, `src/ipc/tensor_types.zig:48` | `build.zig:"interop-fixture-check"`, `src/ipc/stream_reader.zig:"ipc reader reads pyarrow 26 tensor messages and writes them back"`, `src/ipc/stream_reader.zig:"ipc tensor validation: shape, strides and data size checked without overflow on write and read"`, `src/ipc/stream_writer.zig:"ipc writer emits tensor message via public tensor-like api"`, `src/ipc/file_reader.zig:"ipc file roundtrips tensor and sparse tensor messages"` |

`dict-delta` is stream-only because IPC file format does not support dictionary replacement across record batches.

The `ree`, `ree-int16` and `ree-int64` cases share one zarrow check, which asserts the run-end width the
case names (32, 16 or 64 bits) and the decoded values. The zarrow unit test for every width is the IPC
round trip of the type samples, which include int16, int32 and int64 run ends.

The `tensor` fixture is written by zarrow and validated by PyArrow in the `interop-pyarrow` job
(`.github/workflows/ci.yml:368`, `.github/workflows/ci.yml:403`); there is no Arrow Go counterpart.

## Local Full Verification

Run from repository root:

```bash
set -euo pipefail

mkdir -p .interop-fixtures

stream_cases=(canonical dict-delta ree ree-int16 ree-int64 complex extension view)
file_cases=(canonical ree ree-int16 ree-int64 complex extension view)

name_suffix() {
  local case_name="$1"
  if [[ "$case_name" == "canonical" ]]; then
    printf '%s' ""
  else
    printf '%s' "_${case_name//-/_}"
  fi
}

# 1) zarrow generates fixtures, Arrow Go validates.
for case_name in "${stream_cases[@]}"; do
  suffix="$(name_suffix "$case_name")"
  zig build interop-fixture-writer -- ".interop-fixtures/zarrow${suffix}.arrow" "$case_name" stream
  (cd tools/interop/arrow-go && GOCACHE=/tmp/go-build GOMODCACHE=/tmp/go-mod go run . validate "../../../.interop-fixtures/zarrow${suffix}.arrow" "$case_name" stream)
done
for case_name in "${file_cases[@]}"; do
  suffix="$(name_suffix "$case_name")"
  zig build interop-fixture-writer -- ".interop-fixtures/zarrow${suffix}_file.arrow" "$case_name" file
  (cd tools/interop/arrow-go && GOCACHE=/tmp/go-build GOMODCACHE=/tmp/go-mod go run . validate "../../../.interop-fixtures/zarrow${suffix}_file.arrow" "$case_name" file)
done

# 2) Arrow Go generates fixtures, zarrow validates.
for case_name in "${stream_cases[@]}"; do
  suffix="$(name_suffix "$case_name")"
  (cd tools/interop/arrow-go && GOCACHE=/tmp/go-build GOMODCACHE=/tmp/go-mod go run . generate "../../../.interop-fixtures/arrow_go${suffix}.arrow" "$case_name" stream)
  zig build interop-fixture-check -- ".interop-fixtures/arrow_go${suffix}.arrow" "$case_name" stream
done
for case_name in "${file_cases[@]}"; do
  suffix="$(name_suffix "$case_name")"
  (cd tools/interop/arrow-go && GOCACHE=/tmp/go-build GOMODCACHE=/tmp/go-mod go run . generate "../../../.interop-fixtures/arrow_go${suffix}_file.arrow" "$case_name" file)
  zig build interop-fixture-check -- ".interop-fixtures/arrow_go${suffix}_file.arrow" "$case_name" file
done
```

## CI Lanes

| Lane | Coverage | Status | Source | Tests |
|---|---|---|---|---|
| `interop-arrow-go` | full case matrix above, Go `1.24.x` | yes | `.github/workflows/ci.yml:613` | `build.zig:"interop-fixture-check"`, `build.zig:"interop-fixture-writer"` |
| `interop-arrow-go-version-smoke` | `canonical` stream only, Go `1.23.x` and `1.24.x` | yes | `.github/workflows/ci.yml:701` | `build.zig:"interop-fixture-check"` |

## Troubleshooting

`go: cannot find main module`:

- Run Go commands under `tools/interop/arrow-go`, not repository root.

`go.mod requires go >= 1.24`:

- Verify you are on current branch and `tools/interop/arrow-go/go.mod` is up to date.
- Use Go `1.24.x` (same as main CI lane), or run version smoke locally with `1.23.x` only when module/dependencies allow it.
- If your environment pins `GOTOOLCHAIN=local`, ensure local Go version satisfies the module requirement.
