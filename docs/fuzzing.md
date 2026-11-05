# Fuzzing zarrow

zarrow's fuzzing follows Apache Arrow C++'s setup: one entry point per untrusted-input reader,
every decoded array goes through `validateFull`, a returned error is a pass, and a crash, hang,
leak or allocation blow-up is a bug. Seeds come from
[apache/arrow-testing](https://github.com/apache/arrow-testing) and from zarrow's own writers;
every fixed bug leaves its reproducer in `fuzz/corpus/` so CI replays it forever.

## Pieces

| Piece | What it does |
|---|---|
| `fuzz/fuzz.zig` | The harness: `runOne(target, allocator, bytes)` for every target, the replay / mutation CLI, seed generation, and `std.testing.fuzz` tests. |
| `tools/fuzz.py` | Supervisor: runs the harness in a loop, classifies crashes, saves and dedupes reproducers. |
| `tools/fuzz_triage.py` | Markdown report of a fuzz output dir. |
| `tools/fuzz_import_seeds.py` | Copies arrow-testing IPC reproducers and integration fixtures into the corpus. |
| `fuzz/corpus/<dir>/` | Committed seeds (`at-*` from arrow-testing, `gen-*` from `fuzz-gen-seeds`, `regress-*` fixed findings). |

Targets (`zig build fuzz -- <target> ...`):

| Target | Input bytes are | Corpus dir |
|---|---|---|
| `ipc-stream` | an IPC stream for `IpcStreamReader.init` (tensor streams via `nextTensorLikeMessage`) | `ipc-stream` |
| `ipc-stream-zero-copy` | the same, through `IpcStreamReader.initZeroCopy` | `ipc-stream` |
| `ipc-file` | an IPC file for `IpcFileReader` (batches, tensors, sparse tensors) | `ipc-file` |
| `array-layout` | a recipe for a hand-built `ArrayData` tree (type, lengths, buffers, children); `validateLayout`, then `validateFull`, then compute | `array-layout` |
| `compute` | an IPC stream; every column that passes `validateFull` goes through slice / take / filter / concat with indices derived from the input and, where the type fits, the kernels `add`, `less`, `sum`, `min_max`, `and_kleene`, `any`, `is_null`, `count`, `cast` (to a few targets), `utf8_upper`, `utf8_slice_codeunits`, `binary_join_element_wise`, `utf8_split_whitespace`, `replace_substring`, `year`, `iso_calendar`, `floor_temporal`, `round_temporal`, `is_in` (value set: the slice) and a group by of the column on itself (`Grouper`, `hash_count_distinct`, `hash_sum`) (on the column and a slice of it); the kernels must succeed (a cast may fail with `InvalidCast`, a timestamp in a non-UTC timezone with `NotImplemented`) and every array result must pass `validateFull` with the expected length (one row per group for a grouped aggregate) | `ipc-stream` |
| `c-abi` | a recipe for `ArrowSchema` / `ArrowArray` trees imported with `importSchemaFromC` / `importArrayFromC`; an accepted array goes through compute like above, then is exported and re-imported | `c-abi` |

What counts as a finding (the harness panics, the supervisor saves the input):

- any panic: safety checks in Debug / ReleaseSafe are our sanitizer (bounds, overflow, invalid
  enum, unreachable, use of a released ref usually segfaults on the 0xaa fill);
- `fuzz-leak`: bytes still allocated after one input (a counting allocator wraps every input);
- `fuzz-oom`: more than `--alloc-cap` live bytes (default 256 MiB) for one input of at most
  1 MiB, i.e. an allocation sized by an unchecked length;
- `fuzz-invariant`: a kernel turned a valid array into an invalid one or the wrong length;
- `fuzz-hang`: one input running longer than `--timeout` seconds (default 10; exit code 77);
- RSS over `--rss-mb` (supervisor, default 2048).

## Run a campaign

```sh
# 10 minutes per target, all targets; Debug by default, --optimize ReleaseSafe is ~5x faster.
python tools/fuzz.py --out .fuzz-out run --targets all --seconds 600
# one target, faster build
python tools/fuzz.py --optimize ReleaseSafe --out .fuzz-out run --targets ipc-file --seconds 300
# replay a directory one process per file (e.g. new arrow-testing reproducers)
python tools/fuzz.py --out .fuzz-out replay --target ipc-stream path/to/dir
```

The supervisor builds `zig build fuzz-install` (zig-out/bin/zarrow-fuzz), then runs
`zarrow-fuzz <target> run <chunk>s --corpus fuzz/corpus/<dir> --out .fuzz-out/<target>` in
chunks. The harness first runs every corpus input once, then mutates (bit/byte flips, arithmetic,
interesting integers, insert / erase / copy, splices between corpus entries, Arrow tokens such as
`ARROW1`, the 0xFFFFFFFF continuation marker and end-of-stream). Without coverage feedback an
input joins the corpus when it produces a new signature: error name, batch count, `validateFull`
error and the set of decoded type ids. Growth goes to `.fuzz-out/<target>/corpus/` and is
reloaded by the next chunk.

Before each execution the harness writes the input to `.fuzz-out/<target>/current.bin`, so a
crash always leaves its reproducer. The supervisor then saves `<sha1-16>.bin` + `.txt` (stack
trace) per unique signature (kind + panic message + top three stack frames), keeps the smallest
input, counts duplicates in `index.json` and marks the input in `skip/` so the restarted harness
does not crash on it again while loading the corpus.

The harness can be used directly too:

```sh
zig build fuzz -- ipc-stream replay fuzz/corpus/ipc-stream            # every file, one process
zig build fuzz -- compute run 100000 --corpus fuzz/corpus/ipc-stream --out .fuzz-out/compute --seed 1
```

## Read the triage report

```sh
python tools/fuzz_triage.py .fuzz-out > triage.md
```

Per target, one section per signature: kind and count, the panic message, the smallest
reproducer and its trace file, the first ten non-std stack frames, and the first frame inside
`src/` as the suggested place to start. For a leak the frames are the error return trace of the
failing path (the stack at the leak check is just the harness): the first `src/` frame is where
the error started, the leak is in a caller's cleanup on that path.

## Reproduce one crash

```sh
zig build fuzz -- ipc-stream replay .fuzz-out/ipc-stream/0123456789abcdef.bin
```

Under a debugger, run the installed harness with the same arguments:

```sh
zig build fuzz-install
gdb --args zig-out/bin/zarrow-fuzz ipc-stream replay crash.bin    # or lldb -- ...
# Windows: devenv /debugexe zig-out\bin\zarrow-fuzz.exe ipc-stream replay crash.bin (or WinDbg)
```

Check whether a bounds panic is a memory-safety bug in release builds by replaying with
`zig build fuzz -Doptimize=ReleaseFast -- <target> replay <file>`: there the check is gone and the
read is out of bounds (it may or may not fault).

## Fix it

1. Fix the root cause at the trust boundary, in the shared function every caller goes through:
   the IPC bytes and the C ABI structs are untrusted, everything decoded from them must be
   validated before it is used. For example: verify a flatbuffer once before any generated
   unpacker reads it, instead of guarding each accessor; never let a length prefix size an
   allocation before the bytes exist; bound child offsets by what every consumer indexes.
2. Never make a finding go away by loosening validation, catching the panic, raising the caps,
   or special-casing the reproducer. A rejected input must return an error.
3. Error paths free what they allocated: put one `errdefer` at the scope that owns the resource;
   an `errdefer` inside an `if` block stops covering it when the block ends.
4. Add the regression seed: copy the reproducer to `fuzz/corpus/<corpus dir>/regress-<first 8
   hex of its name>.bin` (the arrow-testing convention: reproducers stay in the corpus). Where a
   unit test states the rule more clearly, add one next to the fix too.
5. Re-run on both compilers:

```sh
zig build test && zig build fuzz-corpus
C:\path\to\zig-0.16\zig.exe build test --cache-dir .zig-cache16    # or zig-0.16 on PATH
zig-0.16 build fuzz-corpus --cache-dir .zig-cache16
python tools/fuzz.py --out .fuzz-out run --targets <target> --seconds 300
```

If an input is valid Arrow that zarrow rejects (an Arrow writer produced it), that is a
compatibility gap, not a fuzz bug: open an issue, do not add it as a regression seed that
expects rejection.

## Docker (the Linux campaign from any host)

`--fuzz` is Linux-only and Linux builds the vendored codecs with UBSan, so the complete campaign
runs in a container. The image holds just Debian, Python and a sha256-checked Zig release; the
checkout is mounted, so findings land in `.fuzz-out` on the host; Zig's caches live in a named
Docker volume (`zarrow-fuzz-cache-<version>`), because Zig renames cache directories and a bind
mount of a Windows checkout refuses that:

```sh
sh tools/fuzz_docker.sh 600 2M               # 10 min mutation per target, then 2M coverage-guided iterations
ZIG_VERSION=0.16.0 sh tools/fuzz_docker.sh   # same on the other compiler
```

It replays the corpus, runs `tools/fuzz.py` on every target, runs `zig build fuzz-test --fuzz`,
prints the triage report and exits 1 if anything was found. Reproduce a Linux-only finding the
same way: `docker run --rm -v "$PWD:/work" -v zarrow-fuzz-cache-0.17.0:/cache --entrypoint zig zarrow-fuzz:0.17.0 build fuzz -- <target> replay <file>`.

## Coverage-guided fuzzing (Zig's built-in fuzzer)

Each target is also a `std.testing.fuzz` test in `fuzz/fuzz.zig`. Plain `zig build test` runs
each once on its embedded seed and on empty input. On Linux the real coverage-guided engine runs
headless with an iteration limit:

```sh
zig build fuzz-test --fuzz=1M                        # Zig 0.17 (Debug works)
zig build fuzz-test --fuzz=1M -Doptimize=ReleaseSafe # Zig 0.16: Debug fuzz builds hit a test-runner compile error
zig build fuzz-test --fuzz                           # no limit: opens the web UI
```

Limitations: `--fuzz` is not implemented on Windows (both 0.16 and 0.17 print
`--fuzz not yet implemented for windows`); a failing input is reported by the fuzzer but is not
written as a file by zarrow's tooling, so reproduce it through the portable harness once found.

## Seeds

```sh
python -I tools/fuzz_import_seeds.py <arrow-testing checkout>   # at-* files + NOTICE
zig build fuzz-gen-seeds                                        # gen-* files
```

`fuzz_import_seeds.py` copies arrow-testing `data/arrow-ipc-stream`, `data/arrow-ipc-file`,
`data/arrow-ipc-tensor-stream` and a selection of `integration/` fixtures (Arrow 0.14.1, 1.0.0
big-endian, 2.0.0 compression, 4.0.0 shared dictionaries, C++ 21) up to 32 KiB each, deduped by
content hash; the `NOTICE` in each corpus dir records origin, revision and the Apache-2.0
license. `fuzz-gen-seeds` re-encodes every valid stream in the corpus with zarrow's writers
(plain, lz4, zstd; stream and file; plus a sliced copy of each batch) so mutations start from
zarrow-produced inputs for every type the readers support. Keep the committed corpus under about
2 MB.

## CI

- `ci.yml` (every push / PR): `zig build fuzz-corpus` replays the committed corpus through every
  target, then a 20-second mutation smoke per target; `zig build test` runs the
  `std.testing.fuzz` tests once on all OS / compiler lanes.
- `nightly-fuzz.yml`: 10 minutes per target (ReleaseSafe) with crash artifacts and the triage
  report in the job summary, plus an informational `zig build fuzz-test --fuzz=2M` job.
