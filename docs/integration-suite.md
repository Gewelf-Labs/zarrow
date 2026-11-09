# Arrow integration suite

Apache Arrow checks that its implementations agree with each other through the
[integration tests](https://arrow.apache.org/docs/format/Integration.html): `archery` makes every
implementation convert between an implementation-neutral JSON description of record batches and the IPC
stream and file formats, and compares what the others produced. The `apache/arrow-testing` repository
keeps "gold" files written by known Arrow versions (`data/arrow-ipc-stream/integration/<version>/<case>`:
`.json.gz`, `.arrow_file`, `.stream`). zarrow implements the JSON format, a tester with archery's
command set, and a golden-file check over that repository; this page is its status, how to run it, and
what it found.

Evidence follows the rule in [`CLAUDE.md`](../CLAUDE.md): every `yes` / `partial` row names the
implementing code and a test; `python -I tools/check_doc_evidence.py` checks the references.

## What zarrow implements

| Capability | Status | Source | Tests |
|---|---|---|---|
| Integration JSON reader: schema (every type of the format, dictionary encoding, extension types through `ARROW:extension:name` / `metadata`, schema and field metadata, duplicate field names), dictionaries (shared, nested, `isDelta` appended in order) and batches into record batches whose every column passed `validateFull` | yes | `src/integration/json.zig:84`, `src/integration/json.zig:451` | `src/integration/test_json.zig:"integration json reads and round-trips null, bool, integer and float columns"`, `src/integration/test_json.zig:"integration json reads and round-trips decimal, date, time, timestamp, duration and interval columns"`, `src/integration/test_json.zig:"integration json reads binary, string, fixed-size binary and view columns"`, `src/integration/test_json.zig:"integration json reads and round-trips list, list-view, fixed-size list, struct, map, union and run-end encoded columns"`, `src/integration/test_json.zig:"integration json resolves shared, nested and delta dictionaries"`, `src/integration/test_json.zig:"integration json reads extension types, field and schema metadata"`, `src/integration/test_json.zig:"integration json reads schemas without batches"` |
| Reader as a trust boundary: malformed documents are errors, depth is bounded, any truncation or byte flip is read or rejected without a crash | yes | `src/integration/json.zig:46`, `src/integration/json.zig:42` | `src/integration/test_json.zig:"integration json rejects malformed documents"`, `src/integration/test_json.zig:"integration json rejects nesting deeper than max_depth"`, `src/integration/test_json.zig:"integration json survives every truncation and byte flip of valid documents"`, `build.zig:"fuzz-corpus"` (target `integration-json`, [fuzzing](fuzzing.md)) |
| Integration JSON writer: Arrow C++'s member order and names; one dictionary per id (deltas folded in); sliced batches written as their rows | yes | `src/integration/json_writer.zig:44` | `src/integration/test_json.zig:"integration json writes sliced batches as their logical rows"`, `src/integration/test_json.zig:"integration json reads and round-trips null, bool, integer and float columns"` |
| Logical comparison with Arrow's equality (dictionary ids and list / map internal field names ignored, metadata compared) and a message naming the first difference | yes | `src/integration/compare.zig:47`, `src/integration/compare.zig:108` | `src/integration/test_json.zig:"integration compare follows Arrow's type equality"`, `src/integration/test_json.zig:"integration golden check reports a case whose IPC files differ from its JSON"` |
| Tester with archery's commands (`--integration --mode=VALIDATE / JSON_TO_ARROW / ARROW_TO_JSON`, `--stream-to-file`, `--file-to-stream`) | yes | `tools/integration_runner.zig:31` | `build.zig:"integration-runner"`, `tools/interop/integration_pyarrow_check.py:1` (pyarrow cross-check, below) |
| Golden-file check: JSON, IPC file and IPC stream agree; JSON through zarrow's stream and file writers; file to stream to file; JSON writer output read back | yes | `src/integration/golden.zig:63`, `tools/integration_runner.zig:159` | `src/integration/test_json.zig:"integration golden check passes on the arrow-testing cases in src/ipc/testdata"`, `build.zig:"integration-golden"` |
| Big-endian gold files (`1.0.0-bigendian`) | no (rejected as expected: zarrow reads little-endian IPC only, `UnsupportedType`) | `src/ipc/stream_reader.zig:1222` | `build.zig:"integration-golden"` |
| Unsigned dictionary index types (`generated_dictionary_unsigned`; uint8 .. uint64) in the IPC reader and writer and the C data interface, every index bounded by its dictionary when a batch is decoded or imported | yes | `src/ipc/stream_reader.zig:1613`, `src/ipc/stream_reader.zig:1876`, `src/array/array_data.zig:489` | `src/ipc/stream_reader.zig:"stream and file readers bound unsigned dictionary indices by the dictionary length"`, `src/ipc/stream_writer.zig:"ipc writer writes a dictionary with an unsigned index type"`, `src/ffi/c_data.zig:"c data unsigned dictionary index formats round trip and bound every index"`, `src/integration/test_json.zig:"integration golden check passes on generated_dictionary_unsigned with unsigned index types"` |

## Running it

```sh
# The pinned gold files (the CI job does the same):
git clone https://github.com/apache/arrow-testing
git -C arrow-testing checkout 5a26d35785ff8761e56d79eebff971f6f737e18c

zig build integration-golden -Darrow-testing=arrow-testing -j2   # the table below; exit 1 on a FAIL
zig build integration-runner -j2                                 # zig-out/bin/integration-runner
zig-out/bin/integration-runner --integration --json=case.json --arrow=case.arrow_file --mode=VALIDATE
zig-out/bin/integration-runner --stream-to-file < case.stream > case.arrow_file

# Cross-check against pyarrow (reads every file zarrow writes and the gold JSON):
python -P tools/interop/integration_pyarrow_check.py zig-out/bin/integration-runner arrow-testing .integration-out
```

The runner's flags follow Arrow C++'s `arrow-json-integration-test` (gflags style: `--arrow=x`,
`-arrow x`), so archery can drive it like the C++ tester; `--json-to-arrow`, `--arrow-to-json` and
`--validate` are shortcuts for `--mode`. `zig build test` runs the same golden check on the 16 gold
cases copied into `src/ipc/testdata` (`at-*`, see its `NOTICE`), and CI runs `integration-golden`
on every push (job `integration-golden` in `.github/workflows/ci.yml`).

Per case, the golden check (`src/integration/golden.zig`) runs: **(a)** the JSON, the IPC file and the
IPC stream give equal schemas (with metadata), dictionaries and batches, every column passing
`validateFull`; **(b)** JSON → zarrow stream writer → stream reader equals the JSON; **(c)** the same
through the file writer and reader; **(d)** IPC file → zarrow stream → zarrow file equals the JSON;
**(e)** zarrow's JSON writer output, read back, equals the JSON. The compression cases run with the
vendored lz4 / zstd that `configureIpcCompression` links into the runner, so none is skipped.

## Results

Local run, Windows x86_64, `zig build integration-golden -Darrow-testing=<clone> -j2` on Zig 0.17.0 and
on Zig 0.16.0 (identical tables), arrow-testing 5a26d35785ff8761e56d79eebff971f6f737e18c:

| directory | case | result | detail |
|---|---|---|---|
| 0.14.1 | generated_datetime | pass |  |
| 0.14.1 | generated_decimal | pass |  |
| 0.14.1 | generated_dictionary | pass |  |
| 0.14.1 | generated_interval | pass |  |
| 0.14.1 | generated_map | pass |  |
| 0.14.1 | generated_nested | pass |  |
| 0.14.1 | generated_primitive | pass |  |
| 0.14.1 | generated_primitive_no_batches | pass |  |
| 0.14.1 | generated_primitive_zerolength | pass |  |
| 0.17.1 | generated_union | pass |  |
| 1.0.0-bigendian | generated_custom_metadata | rejected-as-expected | file: UnsupportedType, stream: UnsupportedType |
| 1.0.0-bigendian | generated_datetime | rejected-as-expected | file: UnsupportedType, stream: UnsupportedType |
| 1.0.0-bigendian | generated_decimal | rejected-as-expected | file: UnsupportedType, stream: UnsupportedType |
| 1.0.0-bigendian | generated_decimal256 | rejected-as-expected | file: UnsupportedType, stream: UnsupportedType |
| 1.0.0-bigendian | generated_dictionary | rejected-as-expected | file: UnsupportedType, stream: UnsupportedType |
| 1.0.0-bigendian | generated_dictionary_unsigned | rejected-as-expected | file: UnsupportedType, stream: UnsupportedType |
| 1.0.0-bigendian | generated_duplicate_fieldnames | rejected-as-expected | file: UnsupportedType, stream: UnsupportedType |
| 1.0.0-bigendian | generated_extension | rejected-as-expected | file: UnsupportedType, stream: UnsupportedType |
| 1.0.0-bigendian | generated_interval | rejected-as-expected | file: UnsupportedType, stream: UnsupportedType |
| 1.0.0-bigendian | generated_map | rejected-as-expected | file: UnsupportedType, stream: UnsupportedType |
| 1.0.0-bigendian | generated_map_non_canonical | rejected-as-expected | file: UnsupportedType, stream: UnsupportedType |
| 1.0.0-bigendian | generated_nested | rejected-as-expected | file: UnsupportedType, stream: UnsupportedType |
| 1.0.0-bigendian | generated_nested_dictionary | rejected-as-expected | file: UnsupportedType, stream: UnsupportedType |
| 1.0.0-bigendian | generated_nested_large_offsets | rejected-as-expected | file: UnsupportedType, stream: UnsupportedType |
| 1.0.0-bigendian | generated_null | rejected-as-expected | file: UnsupportedType, stream: UnsupportedType |
| 1.0.0-bigendian | generated_null_trivial | rejected-as-expected | file: UnsupportedType, stream: UnsupportedType |
| 1.0.0-bigendian | generated_primitive | rejected-as-expected | file: UnsupportedType, stream: UnsupportedType |
| 1.0.0-bigendian | generated_primitive_large_offsets | rejected-as-expected | file: UnsupportedType, stream: UnsupportedType |
| 1.0.0-bigendian | generated_primitive_no_batches | rejected-as-expected | file: UnsupportedType, stream: UnsupportedType |
| 1.0.0-bigendian | generated_primitive_zerolength | rejected-as-expected | file: UnsupportedType, stream: UnsupportedType |
| 1.0.0-bigendian | generated_recursive_nested | rejected-as-expected | file: UnsupportedType, stream: UnsupportedType |
| 1.0.0-bigendian | generated_union | rejected-as-expected | file: UnsupportedType, stream: UnsupportedType |
| 1.0.0-littleendian | generated_custom_metadata | pass |  |
| 1.0.0-littleendian | generated_datetime | pass |  |
| 1.0.0-littleendian | generated_decimal | pass |  |
| 1.0.0-littleendian | generated_decimal256 | pass |  |
| 1.0.0-littleendian | generated_dictionary | pass |  |
| 1.0.0-littleendian | generated_dictionary_unsigned | pass |  |
| 1.0.0-littleendian | generated_duplicate_fieldnames | pass |  |
| 1.0.0-littleendian | generated_extension | pass |  |
| 1.0.0-littleendian | generated_interval | pass |  |
| 1.0.0-littleendian | generated_map | pass |  |
| 1.0.0-littleendian | generated_map_non_canonical | pass |  |
| 1.0.0-littleendian | generated_nested | pass |  |
| 1.0.0-littleendian | generated_nested_dictionary | pass |  |
| 1.0.0-littleendian | generated_nested_large_offsets | pass |  |
| 1.0.0-littleendian | generated_null | pass |  |
| 1.0.0-littleendian | generated_null_trivial | pass |  |
| 1.0.0-littleendian | generated_primitive | pass |  |
| 1.0.0-littleendian | generated_primitive_large_offsets | pass |  |
| 1.0.0-littleendian | generated_primitive_no_batches | pass |  |
| 1.0.0-littleendian | generated_primitive_zerolength | pass |  |
| 1.0.0-littleendian | generated_recursive_nested | pass |  |
| 1.0.0-littleendian | generated_union | pass |  |
| 2.0.0-compression | generated_lz4 | pass |  |
| 2.0.0-compression | generated_uncompressible_lz4 | pass |  |
| 2.0.0-compression | generated_uncompressible_zstd | pass |  |
| 2.0.0-compression | generated_zstd | pass |  |
| 4.0.0-shareddict | generated_shared_dict | pass |  |
| cpp-21.0.0 | generated_binary | pass |  |
| cpp-21.0.0 | generated_binary_no_batches | pass |  |
| cpp-21.0.0 | generated_binary_view | pass |  |
| cpp-21.0.0 | generated_binary_zerolength | pass |  |
| cpp-21.0.0 | generated_custom_metadata | pass |  |
| cpp-21.0.0 | generated_datetime | pass |  |
| cpp-21.0.0 | generated_decimal | pass |  |
| cpp-21.0.0 | generated_decimal256 | pass |  |
| cpp-21.0.0 | generated_decimal32 | pass |  |
| cpp-21.0.0 | generated_decimal64 | pass |  |
| cpp-21.0.0 | generated_dictionary | pass |  |
| cpp-21.0.0 | generated_dictionary_unsigned | pass |  |
| cpp-21.0.0 | generated_duplicate_fieldnames | pass |  |
| cpp-21.0.0 | generated_duration | pass |  |
| cpp-21.0.0 | generated_extension | pass |  |
| cpp-21.0.0 | generated_interval | pass |  |
| cpp-21.0.0 | generated_interval_mdn | pass |  |
| cpp-21.0.0 | generated_large_binary | pass |  |
| cpp-21.0.0 | generated_list_view | pass |  |
| cpp-21.0.0 | generated_map | pass |  |
| cpp-21.0.0 | generated_map_non_canonical | pass |  |
| cpp-21.0.0 | generated_nested | pass |  |
| cpp-21.0.0 | generated_nested_dictionary | pass |  |
| cpp-21.0.0 | generated_nested_large_offsets | pass |  |
| cpp-21.0.0 | generated_null | pass |  |
| cpp-21.0.0 | generated_null_trivial | pass |  |
| cpp-21.0.0 | generated_primitive | pass |  |
| cpp-21.0.0 | generated_primitive_no_batches | pass |  |
| cpp-21.0.0 | generated_primitive_zerolength | pass |  |
| cpp-21.0.0 | generated_recursive_nested | pass |  |
| cpp-21.0.0 | generated_run_end_encoded | pass |  |
| cpp-21.0.0 | generated_union | pass |  |

91 cases: 69 pass, 0 FAIL, 22 rejected (big-endian as expected, or by zarrow validation as listed)

`python -P tools/interop/integration_pyarrow_check.py` (pyarrow 26.0.0) on the same clone: all 69
little-endian cases check out (JSON → zarrow file equals the gold file in pyarrow,
gold file → zarrow JSON equals the gold JSON after a normalising parse, zarrow's `--stream-to-file` /
`--file-to-stream` output equals the gold data in pyarrow).

## What the suite found in zarrow

| Bug | Cause | Fix | Regression test |
|---|---|---|---|
| A dictionary nested in another dictionary's values (`generated_nested_dictionary`) was never written by the IPC writers: the stream had only the outer dictionary and reading it back failed (`InvalidMetadata`) | `collectDictionaryIdsFromField` / `collectDictionaryArraysFromData` did not descend into dictionary value types / values | both descend first, so inner dictionaries are numbered and sent before the dictionaries that use them (`src/ipc/stream_writer.zig:897`) | `src/integration/test_json.zig:"IPC writers emit a dictionary nested in another dictionary's values before it"` |
| An unchanged dictionary with nested or view values was re-sent with every batch; in an IPC file that is a replacement, which pyarrow rejects ("Unsupported dictionary replacement in IPC file") | `arraysEqualForDelta` answered "different" for every non-flat value type | the same dictionary (same `ArrayRef`) is skipped, other value types compare logically (`src/ipc/stream_writer.zig:1047`) | `src/integration/test_json.zig:"IPC file writer writes an unchanged list-valued dictionary once"` |
| zarrow's file writer wrote dictionary replacements, producing files other readers reject | no check | the file writer now fails with `DictionaryReplacement` (deltas still allowed) (`src/ipc/file_writer.zig:31`) | `src/integration/test_json.zig:"IPC file writer rejects a dictionary replacement, the stream writer sends it"` |
| A map's entries field name (`generated_map_non_canonical`: `some_entries`) was lost: zarrow wrote `entries` to IPC and the C data interface, so an implementation that compares field names strictly (arrow-rs) sees another schema | `MapType` had no entries name | `MapType.entries_name` (default `entries`), carried by the IPC reader and writer, the C data import and export and the JSON reader and writer (`src/datatype.zig:186`) | `src/integration/test_json.zig:"IPC keeps a non-canonical map entries field name"`, `src/ffi/c_data.zig:"c data schema keeps a non-canonical map entries field name"` |

A follow-up closed a gap the suite exposed: zarrow's IPC reader rejected unsigned dictionary index
types (`InvalidMetadata`) and its writer refused them, although `Schema.fbs` allows any integer index
type and Arrow C++ reads and writes them (`generated_dictionary_unsigned`). Both now accept uint8 ..
uint64 indices; this is not a loosening: every non-null index, signed or unsigned, is still bounded by the
dictionary length when a batch is decoded (`validateDictionaryIndices` in `readArrayFromMeta`, an
out-of-range index is `InvalidBody`: index 255 of a uint8 against a 2-value dictionary, a uint64 above
i64 max), and the C data interface, which already accepted the unsigned formats `C` / `S` / `I` / `L`,
bounds them the same way on import (`InvalidOffset`). The C ABI needed no change; tests were added.

Seeds for the reader paths of these fixes: `fuzz/corpus/ipc-stream/zarrow-integration-*.stream` and
`fuzz/corpus/ipc-file/zarrow-integration-*.arrow_file` (written by the fixed writers), and
`fuzz/corpus/ipc-stream/at-1.0.0-littleendian-generated_dictionary_unsigned.stream` (the cpp-21.0.0 one
was in the corpus already).

## Differences from Arrow that are not bugs

- **Dictionary ids and internal field names.** zarrow's `DataType.eql` compares dictionary ids and the
  names of list item and map fields; Arrow's type equality ignores them. Arrow C++ numbers the
  dictionaries of `generated_nested_dictionary` differently in the JSON (inner first) and the IPC file
  (outer first, one dictionary per field), and its `.stream` of `generated_map_non_canonical` has
  canonical map field names while the `.arrow_file` keeps `some_entries` / `some_key` / `some_value`. The
  suite therefore compares with Arrow's equality (`compare.typesEqual`), and zarrow's own equality is
  unchanged.
- **Floats** are written as Arrow C++ writes them: the double value of the float (`-870.8419799804688` for
  the float32 `-870.842`), whole numbers with `.0`; NaN and the infinities, which JSON numbers cannot hold,
  as the strings `"NaN"`, `"Infinity"`, `"-Infinity"` (read back by the reader). Float16 values are their
  16-bit pattern as an integer.
- **Decimals** are written as `"name": "decimal"` with `bitWidth` (the format's spelling); the reader
  also accepts Arrow C++'s current `decimal32` / `decimal64` / `decimal256` names.
- **Null slots** are written as Arrow C++ writes them (`0`, `"0"`, `false`, `{}`), except that a null
  string or view is written empty instead of with whatever bytes its slot holds.
- **Pre-1.0 unions** carry a `VALIDITY` member in the gold JSON (0.17.1); the reader ignores it, as
  Arrow C++ does, and the writer does not write it.

## Rejected inputs (validation not loosened)

- **Big-endian IPC** (`1.0.0-bigendian`, 22 cases): zarrow's IPC readers accept little-endian data only
  (`UnsupportedType`); the runner reports these cases as `rejected-as-expected`. The JSON side of these
  cases reads fine.
- **A record batch without columns but with rows** (Arrow allows it; no gold case has one) is accepted:
  the readers take the row count from the message when there are no columns
  (`src/ipc/stream_reader.zig:"ipc stream and file readers keep the row count of a record batch without columns"`);
  with columns, a length that differs from theirs is still `InvalidMetadata`.
