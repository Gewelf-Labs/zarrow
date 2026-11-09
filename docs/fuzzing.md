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
| `ipc-stream` | an IPC stream for `IpcStreamReader.init` (tensor streams via `nextTensorLikeMessage`; a decoded tensor must pass `validateIpcTensor`) | `ipc-stream` |
| `ipc-stream-zero-copy` | the same, through `IpcStreamReader.initZeroCopy` | `ipc-stream` |
| `ipc-file` | an IPC file for `IpcFileReader` (batches, tensors, sparse tensors; a decoded tensor must pass `validateIpcTensor`) | `ipc-file` |
| `array-layout` | a recipe for a hand-built `ArrayData` tree (type, lengths, buffers, children); `validateLayout`, then `validateFull`, then compute, then the record batch / table exercise of `compute` with the array as the one column | `array-layout` |
| `compute` | an IPC stream; every column that passes `validateFull` goes through slice / take / filter / concat with indices derived from the input and, where the type fits, the kernels `add`, `less`, `sum`, `min_max`, `and_kleene`, `any`, `is_null`, `count`, `cast` (to a few targets), `utf8_upper`, `utf8_slice_codeunits`, `binary_join_element_wise`, `utf8_split_whitespace`, `replace_substring`, `year`, `iso_calendar`, `floor_temporal`, `round_temporal`, `is_in` (value set: the slice), the Unicode case functions and predicates (`utf8_lower`, `utf8_swapcase`, `utf8_title`, `utf8_capitalize`, `ascii_title`, `utf8_is_lower`, `utf8_is_title`, `utf8_is_alnum`, `utf8_is_printable`, `utf8_is_numeric`, `ascii_is_upper`), `utf8_trim`, `utf8_replace_slice`, `utf8_zero_fill`, `binary_replace_slice`, `binary_join` (list columns of a string or binary type), `add` / `less` / `max_element_wise` with an operand of another numeric type (implicit casts), `min_element_wise` / `max_element_wise`, `is_nan`, `is_finite`, `mean` / `min_max` / `all` with `ScalarAggregateOptions`, `count` with `CountOptions`, `is_null` with `nan_is_null`, `and_not_kleene`, `us_year`, `day_time_interval_between`, `month_day_nano_interval_between`, comparisons of temporal, decimal, duration and fixed-size binary columns, time32 / time64 extraction, differences, rounding and casts, a timestamp to time64 and a dictionary to 16-bit indices, a group by of the column on itself (`Grouper`, `hash_count_distinct`, `hash_sum`), the decimal kernels (`add` .. `divide` and `_checked` with itself and an integer, `add` with a float, `negate`, `abs`, `sign`, `sum`, `mean`, `min`, `max` with options, `hash_sum`, `hash_mean`, `hash_min`, `hash_max`, `hash_count` with options, casts to text, floats, integers and other decimals) and comparisons across layouts: a string / binary column against itself cast to another layout, a timestamp / date32 against itself in a finer unit, a decimal against itself at a larger scale, a dictionary against itself (decoded), and the round 16 kernels: `sqrt`, `power`, `round` (ndigits
-1), `round_to_multiple`, `variance`, `kurtosis`, `product`, `mode`, `quantile` (numbers), `round`,
`floor`, `min_max`, `stddev` (decimal128 / decimal256), `subtract_checked` and `add` with a duration
(timestamps), `unique`, `value_counts`, `count_distinct`, `dictionary_encode` (key types),
`coalesce`, `drop_null`, `fill_null_backward` (every type), `sort_indices` (two keys, both orders),
`rank`, `select_k_unstable` (sortable types) and `hash_first_last`, `hash_list`, `hash_variance`,
`hash_product` in the group by (on the column and a slice of it), round 24 (`exerciseRound24`, `fuzz/fuzz.zig:2273`): `is_null`, `is_valid`, `true_unless_null` and `count` in every mode on every column (also unions and run-end encoded; count must agree with is_null) and the arithmetic / math / rounding / `pairwise_diff` smoke on null-type arguments (a null-type result of the column length), round 23 (`exerciseRound23`, `fuzz/fuzz.zig:2324`): `take` (reversed rows, null indices), `filter`, `dictionary_decode`, `run_end_encode` / `run_end_decode` (decode gives the column back), `modulo` / `remainder` (and `_checked`; `DivideByZero` / `Overflow` allowed) and `hypot` on number columns, round 20: `strftime` (a format of many specifiers) and the cast to string_view on date and timestamp columns (an unknown timezone name, or `%Z` without a timezone, may be `InvalidInput`) and `utf8_normalize` in all four forms on string columns (must succeed and be idempotent), and round 19: the regex kernels (`match_substring_regex`, `count_substring_regex`, `find_substring_regex`, `replace_substring_regex` with group references and with `max_replacements`, `split_pattern_regex`, `extract_regex`, `extract_regex_span`, `match_like`, case-insensitive `match_substring`) on string and binary columns; the kernels must succeed (a cast may fail with `InvalidCast`, an implicit cast of a value the common type cannot hold with `InvalidCast`, a timestamp with an unknown timezone name (or a rounding to an ambiguous or nonexistent local time) with `InvalidInput`, decimal arithmetic with `Overflow`, `DivideByZero` or `NotImplemented`), the cross-layout comparisons must be true on every valid row, a decimal cast to text and back (unsafe) must give its values back, every array result must pass `validateFull` with the expected length (one row per group for a grouped aggregate), a `sort_indices` result must be a permutation of the rows and every `rank` in `[1, rows]`; round 17: every decoded batch whose columns pass `validateFull` also goes through a record batch datum and a table datum chunked at a random cut (take with null indices, filter, `drop_null` (no null may stay), slice, column selection, `sort_indices` and `select_k_unstable` on every sortable column by name, per-key order and null placement: outputs validate with the expected rows, sorts are permutations, select_k distinct rows), dictionary columns go through `unique`, `value_counts` (counts add up to the rows, the dictionary kept), `is_in` / `index_in` against their own dictionary and the group by (dictionary keys and values), every sortable column through `select_k_unstable` (must equal the prefix of the stable sort), `partition_nth_indices` (the sort's row at the pivot, its first rows before it), `rank_quantile` (in (0, 1]) and `search_sorted` (positions in [0, rows]) for both null placements, number and decimal columns through `tdigest` / `approximate_median` (quantiles non-decreasing within 1e-9 of the value range, the median between the extreme quantiles) and `hash_tdigest` / `hash_approximate_median`, decimals through `product` / `hash_product` (a negative scale is `NotImplemented`), and key-type columns through `pivot_wider` / `hash_pivot_wider` with the column as keys and values; round 18 (`exerciseRound18`, also on `array-layout` arrays): `min` / `max` / `min_max` and `hash_min_max` (ids `row % 3`) of the ordered non-numeric types (an empty fixed_size_binary wider than 16 MiB may give `UnsupportedType`), `scatter` and `inverse_permutation` of the reversed rows (the inverse must be the reversed rows again), `replace_with_mask` (every third row true, every fifth null, the column as its own replacements), `cumulative_sum` / `cumulative_max` / `cumulative_prod_checked` (`Overflow` allowed) / `cumulative_mean` / `pairwise_diff` / `winsorize` / `indices_nonzero` on numbers, `pairwise_diff_checked` on temporal columns, float16 casts, `add`, `less` and `is_nan`, and nested casts (`exerciseNestedCasts`): every list layout to list, large_list, list_view, large_list_view and fixed_size_list of its value field, a struct to its fields reversed and a map to a list of its entries and to a sorted map; a nested cast may fail only with `UnsupportedType`, `InvalidCast`, `Overflow` or `NotImplemented`; and the round 21 structural kernels (`exerciseRound21`): `make_struct` of the column with itself and a scalar (every type), `list_value_length`, `list_element` (index 0; only `InvalidInput` for an empty row), `list_flatten` (also recursive), `list_parent_indices`, `list_slice` (input layout and fixed_size_list) on every list layout, `binary_join` on every list layout of a string or binary item, `struct_field` through structs and unions, `map_lookup` (first, last, all) on integer- and string-keyed maps, `binary_replace_slice` on the string layouts (only `InvalidInput` when a code point would split), and `binary_length`, `count_substring`, `find_substring`, `binary_slice`, `binary_replace_slice` on fixed_size_binary columns | `ipc-stream` |
| `c-abi` | a recipe for `ArrowSchema` / `ArrowArray` trees imported with `importSchemaFromC` / `importArrayFromC`; an accepted array goes through compute like above, then is exported and re-imported | `c-abi` |
| `regex` | a flag byte (Latin-1, case-insensitive, literal), a u16 pattern length, the pattern and a text (first 4 KiB): the regex compiler must reject or stay inside its limits (`src/regex/parse.zig` `max_*`, `src/regex/prog.zig` `max_insts`), and `isMatch`, `find` (lazy DFA, forward and reverse), the Pike VM and `captures` must agree on every match with spans inside the text and groups inside their match | `regex` |
| `temporal` | a flag byte (unit, string_view rows, timezone, TZif), a u16 format length, the format and a text (first 4 KiB): `strptime` (`error_is_null`) on the lines of the text, the casts of the lines to every temporal type, `strftime` (the format and the default) and the cast to text of the 8-byte words as timestamps in one of eight zones (naive, UTC, fixed offsets, New York, Lord Howe, Casablanca, Apia), the text cast back must give a naive timestamp back; zoned timestamps go through `local_timestamp`, `is_dst`, fields, rounding (an ambiguous or nonexistent local result may be `InvalidInput`), casts and `assume_timezone` with every choice; with the TZif bit the text is read by the TZif reader (`src/compute/tz.zig`): a file it accepts must give intervals that hold their instant, offsets within RFC 8536's range, a cursor that agrees with the lookup and local times that map back. Every call must succeed or fail with its documented error; outputs must pass `validateFull` with one row per input row | `temporal` |
| `integration-json` | an Arrow integration JSON document for `integration.json.read` (docs/integration-suite.md): an accepted document must have every column pass `validateFull`, write back with `integration.json.write` (only `NestingTooDeep` may stop it), read back equal (`integration.golden.same`), and, when zarrow's IPC stream writer and reader take it, come back from IPC equal | `integration-json` |

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
