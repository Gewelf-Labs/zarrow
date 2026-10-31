# Contributing to zarrow

Thank you for contributing to `zarrow`.

## Development Setup

1. Install Zig `0.16.x` or `0.17.x` (minimum `0.16.0`).
2. Clone the repository.
3. Run:

```sh
zig build test
```

## Before Opening a PR

1. Keep changes focused and reviewable.
2. Add or update tests for behavior changes.
3. Run formatting and tests locally:

```sh
zig fmt src tools examples benchmarks build.zig
zig build test
```

Format with Zig 0.16's `zig fmt`: Zig 0.17's formatter rewrites `@intFromEnum`/`@enumFromInt` to `@backingInt`/`@fromBackingInt`, which does not compile on 0.16.

4. If the change touches a reader, validation or the C ABI, replay the fuzz corpus
   (`zig build fuzz-corpus`) and, for new parsing code, run a short campaign. Fuzz findings are
   fixed at the root cause with a regression seed: see [docs/fuzzing.md](docs/fuzzing.md).

5. If the change is performance-sensitive, also run:

```sh
zig build benchmark-smoke
```

## PR Guidelines

1. Explain what changed and why.
2. Mention compatibility impact (API/IPC/interop).
3. Include benchmark notes for perf changes.
4. Keep CI green.

## Coding Guidelines

1. Prefer clear, explicit code over clever shortcuts.
2. Preserve ownership/ref-count invariants.
3. Keep Arrow layout and IPC behavior spec-aligned.
4. Add tests for error paths, not only happy paths.

## API Stability

See [docs/api-stability.md](docs/api-stability.md).

## Release Process

See [docs/release-checklist.md](docs/release-checklist.md).

## Interop Verification

For Arrow Go IPC interop coverage and local full-matrix commands, see:

- [docs/interop-arrow-go.md](docs/interop-arrow-go.md)
