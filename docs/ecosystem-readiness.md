# zarrow ecosystem extension readiness (compute refresh)

> Refresh date: 2026-05-02  
> Scope: the compute core and the boundary between the framework and the kernel families in `src/compute/`
> (superseded in part: core and compute are one repository, see [repo-layout.md](repo-layout.md))

---

## 0. Executive conclusion

Current conclusion: **the compute framework has array / scalar / chunked entry points for data and provides
chunk-aware permutation primitives; kernel families implement concrete kernels on top of it, in this
repository.**

- `Datum` is now three-state (`array / scalar / chunked`); it is no longer limited to a single chunk.
- The compute core is responsible for the type and dispatch framework, shared helpers and the options
  conventions. It does not promise a complete function catalog (for example every cast / aggregate / sort
  kernel implementation).
- For `sort / take / filter`, reuse the core's chunk-aware helpers first; avoid kernels that `concat` chunked
  input before processing it.
- The options family already includes `sort` (`order / null_placement / nan_placement / stable`), which gives
  one shared sort semantics.

---

## 1. Current compute module structure (public API view)

`src/compute/` is split by responsibility, with `core.zig` as the single export entry point:

- `core.zig`: unified public API exports (types, registry, context, helpers)
- `options.zig`: `OptionsTag` and each options payload
- `signature.zig`: `KernelSignature` with matching and inference logic
- `registry.zig`: `FunctionRegistry / Kernel / ExecContext`
- `datum*.zig`: Datum access, construction and permutation helpers
- `iterators.zig`: array / chunked / scalar execution iterators
- `helpers.zig`: aggregated helper exports

---

## 2. Current state of Datum (array / scalar / chunked)

Current definition and behavior of `Datum`:

- Variants: `array: ArrayRef`, `scalar: Scalar`, `chunked: ChunkedArray`
- Lifecycle: `retain()` / `release()` cover all three variants
- Type access: `dataType()` covers all three variants
- Discrimination and extraction: `isArray / isScalar / isChunked`, `asArray / asScalar / asChunked`

This means:

- kernels can accept `chunked` input directly;
- "support chunked" is not a prerequisite refactoring task;
- the focus shifts to "how to use the chunked helpers efficiently" rather than "is there a chunked entry
  point".

---

## 3. Responsibility boundary: compute framework vs kernel families

### 3.1 What the compute core already provides

- Dispatch and registration framework: `FunctionRegistry`, `Kernel`, `KernelSignature`, `ExecContext`
- One error domain: `KernelError` (including `InvalidOptions / InvalidInput / UnsupportedType` and others)
- Execution helpers: `unary / binary / nary` iteration and null-propagation helpers
- Datum helpers:
  - access: `datumListValueAt`, `datumStructField` and others
  - construction: `datumBuildNullLike`, `datumBuildEmptyLike`
  - permutation: `datumTake`, `datumTakeNullable`, `datumSelect`, `datumFilterSelectionIndices`,
    `datumFilterChunkAware`
- Chunk-aware primitives:
  - `chunkedResolveLogicalIndices`
  - consistent index and null handling on the chunked take / filter paths

### 3.2 What kernel families own

- The semantics and implementation of concrete business kernels (for example sort key interpretation,
  aggregation strategy, expression-level optimization)
- The function catalog and naming conventions (which functions are exposed as stable API)
- Performance strategy (parallelism, local sort algorithms, cross-chunk merge strategy)
- Integration with higher-level engines and execution plans (those remain separate projects)

---

## 4. Recommended extension path for sort / take / filter

### 4.1 sort (recommended)

1. Implement `sort_indices` as a kernel family, with one logical semantics shared by array and chunked input.
2. Reuse the core `Datum` and chunk helpers to map logical indices to chunk-local ones.
3. Drive sort behavior through `SortOptions`; do not hard-code "ascending + nulls last" inside the kernel.
4. If `array_sort_indices` is needed, make it an array specialization wrapping `sort_indices` rather than a
   copied semantic branch.

### 4.2 take (recommended)

- Call `datumTake` / `datumTakeNullable` first.
- Keep chunked output for chunked input; avoid "concat first, then gather".
- On the nullable-index path, keep the core's null-row output semantics.

### 4.3 filter (recommended)

- Predicate to selection vector: `datumFilterSelectionIndices`
- Chunk-aware execution: `datumFilterChunkAware`
- When the legacy array output shape is required, use `datumFilter` (compatibility path)

### 4.4 Consistency checklist (for kernel implementations)

- Do array and chunked inputs yield the same logical result?
- Misaligned chunk boundaries (contiguous and non-contiguous indices across chunks)
- Are null and NaN semantics driven only by options?
- Do out-of-range indices and invalid options return consistent errors?

---

## 5. Options family: current state and direction

### 5.1 Options families exposed today

`OptionsTag`:

- `none`
- `cast`
- `arithmetic`
- `filter`
- `sort`
- `custom`

Matching payloads:

- `CastOptions`
- `ArithmeticOptions`
- `FilterOptions`
- `SortOptions`
- `CustomOptions`

### 5.2 Current SortOptions fields

- `order`: `ascending | descending`
- `null_placement`: `at_start | at_end`
- `nan_placement`: `?at_start | ?at_end`
- `stable`: `bool`

### 5.3 Near-term extension directions (compatible with the current model)

- Multi-key sort (compose several `SortOptions` at a higher level, or extend a custom payload)
- Sort metadata aligned with the execution-plan layer (for example column selection, comparator
  configuration)
- Keep extending without breaking `Options.noneValue()` compatibility

---

## 6. Onboarding advice for a new kernel family (short version)

- Step 1: use only the framework and helpers exported by `compute/core.zig`.
- Step 2: land the three paths `sort_indices + take + filter` first, reusing the existing chunk-aware
  infrastructure.
- Step 3: put options validation in `KernelSignature.options_check` instead of scattering type checks
  inside exec.
- Step 4: fill in regression tests:
  - defaults
  - tag dispatch
  - invalid option mismatch
  - array / chunked consistency

---

## 7. Facts aligned by this refresh

- The document matches the current compute public API: `Datum` is `array / scalar / chunked`.
- The old statement "chunked datum support is missing" has been removed.
- Clarified: the compute core provides the framework and helpers; kernel families in this repository
  implement concrete kernels.
- Added: the recommended extension path for `sort / take / filter` and the options family (including `sort`).
