# zarrow project rules

## Evidence rule for compute and compatibility claims

Every claim that zarrow supports, partially supports, or is compatible with something (an Arrow
compute function, an IPC / C data interface feature, a type, another implementation) must cite
evidence in the same table row or paragraph:

- at least one **source reference** in the form `` `src/path/file.zig:LINE` `` pointing at the
  implementing `pub fn` or type, and
- at least one **test reference** in the form `` `src/path/file.zig:"exact test name"` `` naming a
  test that exercises the claim (prefer tests covering nulls, sliced inputs and chunked inputs), or
  a build step `` `build.zig:"step-name"` ``.

A claim with no test is not "yes"; downgrade it to "partial" and say that the test is missing.
`python -I tools/check_doc_evidence.py` verifies that every reference resolves and that every
`yes` / `partial` row in `docs/compute-support-matrix.md` carries both kinds; CI runs it. Update the
references when code moves.

## Repository scope

Core and compute are one repository. Compute kernels belong in `src/compute/` (one file per family) with
their tests and benchmarks here; never describe kernels as belonging to a downstream `zarrow-compute`
repository. Parquet and Flight stay separate projects. See `docs/repo-layout.md`.

## Language

Write all new documentation, comments, commit messages and reports in English. The existing `*-zh.md`
files are legacy mirrors; do not add Chinese text to them or anywhere else, and link to the English
page instead of translating.

## Toolchains

Code must build on Zig 0.16 and 0.17 (see README). Run `zig build test -j2` on both before claiming
tests pass; the CI lint uses 0.16 `zig fmt`.
