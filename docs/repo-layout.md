# Repository layout: core and compute in one repository

This replaces the earlier multi-repository split plan. `zarrow` is a single repository that holds both the
core Arrow implementation (types, arrays, builders, IPC, C data interface) and the compute layer (framework
and kernels). Nothing compute-related is planned for a separate `zarrow-compute` repository.

## Why one repository

- Kernels and the framework change together: a new kernel kind, a null-handling helper or an options
  family lands with the kernels that use it, in one change, with one test run on both Zig versions.
- The selection, concat, dictionary-encode and validation kernels already live here and are tuned against
  the array layouts in `src/`; splitting them out would re-create the cross-repository drift the old plan
  tried to manage.
- One fuzzing harness, one benchmark CI and one evidence checker cover everything.

## Where things live

| Area | Path | Notes |
|---|---|---|
| Core types, arrays, builders, buffers | `src/datatype.zig`, `src/array/`, `src/buffer.zig`, `src/bitmap.zig` | |
| Chunked arrays, record batches, tables | `src/chunked_array.zig`, `src/record_batch.zig`, `src/table.zig` | |
| IPC stream and file | `src/ipc/` | |
| C data and C stream interface | `src/ffi/`, `src/c_api.zig` | |
| Compute framework | `src/compute/` (registry, signature, options, context, iterators, datum helpers, dense helpers) | documented in [compute-api.md](compute-api.md) |
| Compute kernels | `src/compute/` as one file per family (`src/compute/<family>.zig`, tests in `src/compute/test_<family>.zig`), plus `src/concat_array_refs.zig` for concat and dictionary encode | status in [compute-support-matrix.md](compute-support-matrix.md) |
| Benchmarks | `benchmarks/`, registered in `tools/benchmark_ci_check.py`, `tools/benchmark_ci_thresholds.json`, `tools/test_benchmark_ci_check.py` | |
| Fuzzing | `fuzz/`, `tools/fuzz*.py`, `tools/fuzz_docker.sh` | [fuzzing.md](fuzzing.md) |
| Interop fixtures and runners | `tools/interop/` | |

## Build modules

`build.zig` exposes two modules. Both come from this repository; the second is a subset, not a separate
project.

- `zarrow`: everything.
- `zarrow-core`: the core data structures and the compute layer without IPC and FFI, for consumers that do
  not want the zstd and C ABI dependencies.

## Registry

`FunctionRegistry.init` returns an empty registry. Kernel families expose a `register(registry)` function;
an application registers the families it uses. A convenience that registers every family in the repository
is the intended equivalent of `arrow::compute::Initialize()` and should be added with the first kernel
family.

## Still separate

Parquet and Flight are not part of this repository. [zarrow-parquet](https://github.com/tylitianrui/zarrow-parquet)
depends on `zarrow`; a Flight implementation would do the same. The rules for those projects are the ones the
old plan stated: depend on a released `zarrow`, follow its minor versions, keep their own CI on the same Zig
matrix, and never copy core type definitions.

## Versioning inside the repository

Public symbols exported from `src/root.zig` follow [api-stability.md](api-stability.md). Kernel families are
part of that surface once their matrix row says `yes`; a `partial` row is not a stability promise.
