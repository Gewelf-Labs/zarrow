# Canonical extension types: coverage

Arrow defines a set of [canonical extension types](https://arrow.apache.org/docs/format/CanonicalExtensions.html):
named extensions in the reserved `arrow.` namespace, each over a fixed storage type, carried through IPC and the
C data interface as the field metadata keys `ARROW:extension:name` and `ARROW:extension:metadata`. An
implementation is not required to support any of them; one that does not recognize a name must keep the
storage type usable and pass the metadata along.

This page records what `zarrow` does with them. Evidence follows the rule in [`CLAUDE.md`](../CLAUDE.md):
every `yes` / `partial` row cites a source reference and a test reference, checked by
`tools/check_doc_evidence.py`.

Legend: **yes** = the behaviour exists and a test exercises it; **partial** = the type passes through the
generic extension mechanism but zarrow does not recognize the name, validate its metadata or check its storage
type; **no** = nothing.

## Generic extension mechanism

| Behaviour | zarrow | Status | Source | Tests |
|---|---|---|---|---|
| Extension type representation (name, storage type, optional metadata) | `DataType.extension` / `ExtensionType` | yes | `src/datatype.zig:204`, `src/datatype.zig:253` | `src/datatype.zig:"data type physical type unwraps dictionary and extension"` |
| Unwrap to storage for layout, validation and compute (nested extensions unwrapped in a loop) | `storageDataType`, `layoutDataType` | yes | `src/datatype.zig:214`, `src/array/array_data.zig:76` | `src/array/array_data.zig:"array data validateLayout accepts extension by delegating to storage layout"` |
| Build an extension array over a storage array; storage type must equal the declared one | `ExtensionBuilder.finish` (`StorageTypeMismatch` otherwise) | yes | `src/array/extension_array.zig:67` | `src/array/extension_array.zig:"extension builder wraps storage array and preserves layout"` |
| IPC write: name always, metadata when present, type encoded as the storage type | stream writer field custom metadata | yes | `src/ipc/stream_writer.zig:764`, `src/ipc/stream_writer.zig:660` | `src/ipc/stream_writer.zig:"ipc writer and reader roundtrip extension field metadata and values"` |
| IPC read: any name accepted and kept as an extension over the decoded storage type; keys removed from user metadata; metadata without a name or duplicate keys rejected | stream reader `buildFieldFromFlatbuf`, `parseFieldMetadataFromFlatbuf` | yes | `src/ipc/stream_reader.zig:1157`, `src/ipc/stream_reader.zig:1234` | `src/ipc/stream_reader.zig:"ipc schema decodes extension metadata into extension datatype"`, `src/ipc/stream_reader.zig:"ipc schema rejects extension metadata without extension name"` |
| IPC read: extension over a dictionary wraps the dictionary value type | stream reader | partial (no test) | `src/ipc/stream_reader.zig:1158` | `src/ipc/stream_reader.zig:"ipc schema decodes extension metadata into extension datatype"` (plain storage only) |
| C data interface export and import of both keys; duplicates and metadata without a name rejected | `src/ffi/c_data.zig` | yes | `src/ffi/c_data.zig:1264`, `src/ffi/c_data.zig:1343` | `src/ffi/c_data.zig:"c data schema export/import roundtrip preserves extension metadata"`, `src/ffi/c_data.zig:"c data array export/import handles extension layout via storage type"` |
| Compute kernels (take, filter, concat) on extension arrays operate on the storage layout | `storageDataType` dispatch in the kernels | yes | `src/concat_array_refs.zig:44`, `src/compute/datum_permutation.zig:1113` | `src/compute/test_datum.zig:"compute datumTake nested kernels match the per-row oracle on sliced inputs with null indices"`, `src/compute/test_datum.zig:"concatArrayRefs N-ary path matches the source rows for every layout, sliced parts and child windows"` (both use a `test.ext` over int32 source) |
| `validateFull` on an extension array (validates the storage) | `validateFull` via `layoutDataType` | partial (no extension-specific test) | `src/array/array_data.zig:795` | `src/array/array_data.zig:"array data validateLayout accepts extension by delegating to storage layout"` |
| Fuzzing of extension metadata on the IPC path | fuzz target `ipc-stream` | yes | `fuzz/fuzz.zig:719` | `fuzz/fuzz.zig:"fuzz ipc-stream"` |
| Registry of known extension names, equality by name plus metadata, typed accessors | none | no | | |

## Canonical types

For every canonical type the status is the same: the field round-trips through IPC and the C data interface
with its name and metadata intact, and the array is usable as its storage type. zarrow does not recognize the
name, does not parse or validate the metadata, does not check that the storage type is the one the
specification requires, and offers no typed accessor. The evidence for the pass-through is the generic
mechanism above; the `Storage in zarrow` column says whether the required storage type exists at all.

| Canonical name | Required storage | Metadata | Storage in zarrow | Status | Source | Tests |
|---|---|---|---|---|---|---|
| `arrow.fixed_shape_tensor` | FixedSizeList of the value type, list size = product of `shape` | JSON: `shape` required, optional `dim_names`, `permutation` | yes, `fixed_size_list` | partial (pass-through) | `src/ipc/stream_reader.zig:1157` | `src/ipc/stream_writer.zig:"ipc writer and reader roundtrip extension field metadata and values"` |
| `arrow.variable_shape_tensor` | Struct of `data` (List of value type) and `shape` (FixedSizeList<int32>) | JSON, may be empty: `dim_names`, `permutation`, `uniform_shape` | yes, `struct_`, `list`, `fixed_size_list` | partial (pass-through) | `src/ipc/stream_reader.zig:1157` | `src/ipc/stream_writer.zig:"ipc writer and reader roundtrip extension field metadata and values"` |
| `arrow.json` | String, LargeString or StringView (UTF-8 JSON) | empty string or `{}` | yes, `string`, `large_string`, `string_view` | partial (pass-through; `validateFull` checks UTF-8 of the storage, not JSON syntax) | `src/ipc/stream_reader.zig:1157`, `src/array/array_data.zig:795` | `src/ipc/stream_writer.zig:"ipc writer and reader roundtrip extension field metadata and values"`, `src/array/array_data.zig:"array data validateFull checks UTF-8 of non-null string values like a per-value check"` |
| `arrow.uuid` | FixedSizeBinary(16), big-endian | not specified | yes, `fixed_size_binary` | partial (pass-through) | `src/ipc/stream_reader.zig:1157` | `src/ipc/stream_writer.zig:"ipc writer and reader roundtrip extension field metadata and values"` |
| `arrow.opaque` | any | JSON: `type_name`, `vendor_name` | yes | partial (pass-through) | `src/ipc/stream_reader.zig:1157` | `src/ipc/stream_writer.zig:"ipc writer and reader roundtrip extension field metadata and values"` |
| `arrow.bool8` | Int8 (0 = false, non-zero = true) | empty string | yes, `int8` | partial (pass-through) | `src/ipc/stream_reader.zig:1157` | `src/ipc/stream_writer.zig:"ipc writer and reader roundtrip extension field metadata and values"` |
| `arrow.parquet.variant` | Struct of non-nullable `metadata` (Binary / LargeBinary / BinaryView) plus `value` and/or `typed_value`; fields may be dictionary- or run-end-encoded | empty string | yes, `struct_`, `binary`, `large_binary`, `binary_view`, `dictionary`, `run_end_encoded` | partial (pass-through) | `src/ipc/stream_reader.zig:1157` | `src/ipc/stream_writer.zig:"ipc writer and reader roundtrip extension field metadata and values"` |
| `arrow.timestamp_with_offset` | Struct of non-nullable `timestamp` (Timestamp(unit, "UTC")) and non-nullable `offset_minutes` (Int16) | empty string | yes, `struct_`, `timestamp`, `int16` | partial (pass-through) | `src/ipc/stream_reader.zig:1157` | `src/ipc/stream_writer.zig:"ipc writer and reader roundtrip extension field metadata and values"` |

Community (non-canonical) types such as GeoArrow get the same pass-through.

## What "yes" would take

Per type: a constructor that checks the storage type and parses the metadata, a name match on IPC and C ABI
import that returns `InvalidMetadata` when the storage type or metadata is wrong, a `validateFull` hook for
the value constraints (UTF-8 JSON for `arrow.json`, list size equal to the shape product for
`arrow.fixed_shape_tensor`, UTC unit and offset range for `arrow.timestamp_with_offset`), and a round-trip
test against PyArrow's registered canonical types in `tools/interop`. The interop fixtures today use only the
non-canonical `com.example.int32_ext`.
