# Apache Arrow Specification and IPC: C / Rust Implementation Notes (Implementation-Oriented)

> Updated: 2026-04-16  
> Goal: give implementers a directly usable "specification + implementation" description, focusing on the Arrow columnar memory specification, the IPC wire protocol, how the C interface is implemented, and the layered implementation in Rust (arrow-rs).

## 1. Arrow Columnar Specification (must know)

### 1.1 Unified Data Model

By specification, an Arrow array consists of the following elements:

- `data_type`
- `buffers[]`
- `length`
- `null_count`
- `children[]` (nested types)
- `dictionary` (dictionary encoding)

In implementation, any column can be viewed as an `ArrayData tree`: the parent node describes the column itself, and child nodes describe the nested sub-columns.

### 1.2 Alignment and Padding

- For memory allocation, 8- or 64-byte alignment and padding are recommended.
- During IPC serialization, these alignment and padding requirements are enforced.
- Community guidance prefers 64-byte alignment (better for SIMD / cache-line behavior).

### 1.3 Validity Bitmap

- Most types use a separate validity bitmap.
- Bit semantics: `1 = valid (non-null)`, `0 = null`.
- Bit order: `LSB-first` (`bitmap[j / 8] & (1 << (j % 8))`).
- When `null_count == 0`, the bitmap may be omitted (implementations may still choose to allocate it).

### 1.4 Core Constraints of Variable-Length Layouts (Binary/List)

- Variable-length values are represented by `offsets` + `data`.
- `offsets` must be monotonically non-decreasing.
- Null slots may correspond to non-empty physical intervals (the semantics are determined by validity).

## 2. IPC Specification (Stream / File)

### 2.1 IPC Message Types and Structures

Arrow IPC transmits a stream of messages. The core messages are:

- `Schema`
- `RecordBatch`
- `DictionaryBatch`

#### 2.1.1 Structures of Schema / RecordBatch / DictionaryBatch

The following structures come from the Arrow IPC FlatBuffers definitions (`src/format/Schema.fbs`, `src/format/Message.fbs`).

#### Schema (Logical Structure Definition)

`Schema` describes "what the columns look like" and does not contain actual column data:

```text
Schema {
  endianness: Endianness = Little
  fields: [Field]
  custom_metadata: [KeyValue]
  features: [Feature]
}
```

`Field` describes each column recursively:

```text
Field {
  name: string
  nullable: bool
  type: Type
  dictionary: DictionaryEncoding?   // present when dictionary-encoded
  children: [Field]                 // nested types such as List/Struct/Union
  custom_metadata: [KeyValue]
}
```

Key fields of `DictionaryEncoding`:

```text
DictionaryEncoding {
  id: long
  indexType: Int
  isOrdered: bool
}
```

Implementation notes:

- `Schema` determines the flattening order of `nodes/buffers` in subsequent `RecordBatch` messages.
- The index type of a dictionary column is fixed in `Field.dictionary.indexType`; the actual dictionary values are provided by a later `DictionaryBatch`.

#### RecordBatch (A Batch of Column Data)

`RecordBatch` describes "the data layout of this batch of rows and how it maps onto the message body":

```text
RecordBatch {
  length: long                 // number of rows
  nodes: [FieldNode]           // logical nodes in pre-order traversal
  buffers: [Buffer]            // physical buffer ranges in pre-order traversal
  compression: BodyCompression?
  variadicBufferCounts: [long] // buffer counts for variable-count buffers such as BinaryView/Utf8View
}

FieldNode {
  length: long
  null_count: long
}

Buffer {
  offset: long
  length: long
}
```

Implementation notes:

- Both `nodes` and `buffers` are produced by a "pre-order flatten" of the schema tree.
- `Buffer.offset/length` are ranges relative to the body of this message.
- When `compression` is enabled, compression is applied per buffer, and each buffer carries a raw-length prefix as specified.

#### DictionaryBatch (Dictionary Payload)

`DictionaryBatch` is a dedicated message that "sends the dictionary values for a given dictionary id":

```text
DictionaryBatch {
  id: long
  data: RecordBatch
  isDelta: bool = false
}
```

Semantics:

- `id`: corresponds to `Field.dictionary.id`.
- `data`: the dictionary values themselves (also a `RecordBatch`).
- `isDelta=false`: replaces the existing dictionary.
- `isDelta=true`: appends this batch's dictionary values to the end of the existing dictionary.

Implementation notes:

- The writer must maintain a `dictionary id -> last dictionary` state to decide whether to send a full or a delta dictionary.
- The reader must consume and update the dictionary first, before decoding subsequent `RecordBatch` messages that use that id.

### 2.2 Encapsulated Message Format

Each IPC message (stream or file) is wrapped with the following structure:

1. 32-bit continuation indicator (valid value `0xFFFFFFFF`)
2. 32-bit little-endian metadata length
3. FlatBuffers `Message` metadata
4. padding (aligned to an 8-byte boundary)
5. body (length must be a multiple of 8 bytes)

This ensures that messages are relocatable and that structures can be rebuilt with zero copy.

### 2.3 Streaming Format

Order in the stream format:

1. `Schema`
2. zero or more `DictionaryBatch` messages
3. zero or more `RecordBatch` messages
4. optional `EOS`: `0xFFFFFFFF 0x00000000`

Key rules:

- `DictionaryBatch` and `RecordBatch` messages may be interleaved;
- but in a stream, the corresponding dictionary usually must appear before a dictionary key is used;
- the writer can end the stream with EOS or by closing the stream.

### 2.4 File Format (Random Access)

File format = stream format + footer:

- File header magic: `ARROW1`
- File trailer magic: `ARROW1`
- The trailer contains a footer (with a copy of the schema and the offset/length of each data block), which enables random access to RecordBatches

Additional notes:

- In a file, dictionaries may appear in a looser order than in a stream (they only need to be defined somewhere in the file);
- but the same dictionary id must not have more than one "non-delta" replacement batch.

## 3. C Implementation Principles (Recommended Path)

### 3.1 The C Data Interface as the ABI Core

The minimal interoperable unit on the C side is two structs:

- `ArrowSchema`: describes the type and metadata
- `ArrowArray`: describes the data itself (buffers/children/dictionary)

The key point is not calling some large library API, but strictly following the field semantics and lifetime contract of these two structs.

### 3.2 Lifetime and Ownership Model

Core mechanics of the C Data Interface:

- The producer fills in the struct and sets the `release` callback;
- The consumer calls `release` after use;
- The "released" state of a struct is represented by `release == NULL`;
- Move semantics are implemented by shallow-copying the struct and then marking the source as released.

Implementation conclusion: the stability of a C implementation depends critically on whether the **release callback correctly and recursively releases children/dictionary/private_data**.

### 3.3 Multi-Batch Transfer: C Stream Interface

`ArrowArrayStream` provides a pull model through callbacks:

- `get_schema`
- `get_next`
- `get_last_error`
- `release`

Returning a released array from `get_next` signals EOS.  
This layer can directly carry batch output from a database or execution engine, without an IPC deserialization step.

### 3.4 Practical Recommendation: nanoarrow

If the goal is a "lightweight C/C++ Arrow API", prefer nanoarrow:

- It is built directly around the C Data Interface;
- It is small and easy to vendor;
- It suits the C ABI boundary layer (rather than a full analytics engine).

## 4. Rust (arrow-rs) Implementation Principles

### 4.1 Layered Architecture (Most Important)

arrow-rs is not a single crate but a set of layers:

- `arrow-schema`: logical types (`DataType` / `Field` / `Schema`)
- `arrow-buffer`: low-level shared memory and bit/byte buffer abstractions
- `arrow-data`: `ArrayData` and layout validation, low-level construction
- `arrow-array`: strongly typed arrays and builders, `dyn Array` type erasure
- `arrow-ipc`: Stream/File reading and writing, IPC encoding/decoding

Implementation advice: first adopt the `arrow-data` mental model (buffers/children/layout), then move up to the `arrow-array` API.

### 4.2 IPC in Rust

`arrow-ipc` explicitly distinguishes two kinds:

1. Streaming: `StreamReader` / `StreamWriter`
2. File: `FileReader` / `FileWriter`

These correspond one-to-one with the specification:

- A stream is designed for sequential reading;
- A file relies on the footer to support random access;
- The writer is responsible for writing the termination/footer at the end (for example via `finish` / `close`).

### 4.3 Rust and C ABI Interoperability

arrow-rs provides C interface bridging in the `ffi` / `ffi_stream` modules:

- C Data Interface: `ArrowSchema` / `ArrowArray` correspond to the Rust FFI structs;
- C Stream Interface: `ArrowArrayStream` corresponds to `FFI_ArrowArrayStream`;
- C-side batch streams can be imported as a Rust `RecordBatchReader`, and the reverse export is also possible.

This lets Rust serve as the "compute kernel" and the C side as the "system boundary layer".

## 5. Recommended Division of Work Between C and Rust

Recommended layering:

1. C layer: the in-process ABI boundary (C Data/C Stream, responsible for lifetime and zero-copy handoff)
2. Rust layer: type-safe construction, validation, computation, and IPC encoding/decoding
3. Transport layer: IPC Stream/File when crossing processes or writing to disk

One-sentence summary:

- **In-process**: prefer C Data/C Stream (no FlatBuffers dependency, zero-copy).
- **Inter-process/persistence**: use IPC (streamable, random-access, compressible).

## 6. Correspondence with This Repository (zarrow)

This project already covers the key paths above; the following are worth cross-referencing:

- IPC: `src/ipc/`
- FlatBuffers protocol definitions: `src/format/*.fbs`
- C Data Interface related: `src/ffi/c_data.zig`
- ArrayData and layout validation: `src/array/array_data.zig`

## 7. Examples (Runnable and Modifiable)

### 7.1 Zig: IPC Stream Write/Read Round Trip (zarrow)

Existing example in the repository: `examples/ipc_stream.zig`  
Run:

```sh
zig build example-ipc_stream
```

Core flow code (matches the example):

```zig
const std = @import("std");
const zarrow = @import("zarrow");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    const id_type = zarrow.DataType{ .int32 = {} };
    const name_type = zarrow.DataType{ .string = {} };
    const fields = [_]zarrow.Field{
        .{ .name = "id", .data_type = &id_type, .nullable = false },
        .{ .name = "name", .data_type = &name_type, .nullable = true },
    };
    const schema = zarrow.Schema{ .fields = fields[0..] };

    var id_builder = try zarrow.Int32Builder.init(allocator, 3);
    defer id_builder.deinit();
    try id_builder.append(1);
    try id_builder.append(2);
    try id_builder.append(3);
    var id_ref = try id_builder.finish();
    defer id_ref.release();

    var name_builder = try zarrow.StringBuilder.init(allocator, 3, 12);
    defer name_builder.deinit();
    try name_builder.append("alice");
    try name_builder.appendNull();
    try name_builder.append("bob");
    var name_ref = try name_builder.finish();
    defer name_ref.release();

    var batch = try zarrow.RecordBatch.initBorrowed(allocator, schema, &[_]zarrow.ArrayRef{ id_ref, name_ref });
    defer batch.deinit();

    var out: std.Io.Writer.Allocating = .init(allocator);
    defer out.deinit();

    var writer = zarrow.IpcStreamWriter.init(allocator, &out.writer);
    defer writer.deinit();
    try writer.writeSchema(schema);
    try writer.writeRecordBatch(batch);
    try writer.writeEnd();

    var in: std.Io.Reader = .fixed(out.written());
    var reader = zarrow.IpcStreamReader.init(allocator, &in);
    defer reader.deinit();

    _ = try reader.readSchema();
    while (try reader.nextRecordBatch()) |*rb| {
        var one = rb.*;
        defer one.deinit();
        std.debug.print("rows={d}\n", .{one.numRows()});
    }
}
```

### 7.2 Zig: C Data Interface Export/Import Example

This example shows the core path of "zero-copy in-process handoff":

```zig
const std = @import("std");
const zarrow = @import("zarrow");

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    const id_type = zarrow.DataType{ .int32 = {} };

    var b = try zarrow.Int32Builder.init(allocator, 3);
    defer b.deinit();
    try b.append(10);
    try b.appendNull();
    try b.append(30);
    var arr = try b.finish();
    defer arr.release();

    // Export
    var c_arr = try zarrow.exportArrayToC(allocator, arr);
    defer if (c_arr.release) |rel| rel(&c_arr);

    // Import (ownership transferred to importer)
    var imported = try zarrow.importArrayFromC(allocator, &id_type, &c_arr);
    defer imported.release();

    const view = zarrow.Int32Array{ .data = imported.data() };
    std.debug.print("len={d}, v0={d}, isNull1={any}\n", .{
        view.len(),
        view.value(0),
        view.isNull(1),
    });
}
```

### 7.3 Rust: arrow-rs IPC Stream Example

```rust
use std::sync::Arc;
use arrow_array::{Int32Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use arrow_ipc::writer::StreamWriter;
use arrow_ipc::reader::StreamReader;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, true),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            Arc::new(StringArray::from(vec![Some("alice"), None, Some("bob")])),
        ],
    )?;

    let mut buf = Vec::new();
    {
        let mut w = StreamWriter::try_new(&mut buf, &schema)?;
        w.write(&batch)?;
        w.finish()?;
    }

    let mut r = StreamReader::try_new(std::io::Cursor::new(buf), None)?;
    while let Some(batch) = r.next() {
        let b = batch?;
        println!("rows={}", b.num_rows());
    }
    Ok(())
}
```

## 8. Code Logic Implementation (zarrow Source Level)

### 8.1 IPC Writer Logic Chain

Corresponding file: `src/ipc/stream_writer.zig`

1. `writeSchema()`: builds the Schema FlatBuffers and writes the message header  
   Entry: `writeSchema` (:186)  
   Core: `buildSchemaT` (:556) -> `writeMessage` (:328)
2. `writeRecordBatch()`: scans fields, handles dictionaries, builds the RecordBatch metadata and body  
   Entry: `writeRecordBatch` (:216)  
   Key actions: dictionary state caching (`dictionary_values`) + body compression selection
3. `writeEnd()`: emits the stream end marker (EOS)  
   Entry: `writeEnd` (:321)

In one sentence: the Writer's responsibility is to "linearize the `ArrayData tree` into an IPC message sequence and maintain the dictionary state machine".

### 8.2 IPC Reader Logic Chain

Corresponding file: `src/ipc/stream_reader.zig`

1. `readSchema()`: reads the first Schema message and establishes a `SchemaRef`  
   Entry: `readSchema` (:174)  
   Core: `readMessage` (:352) -> `buildSchemaFromFlatbufWithEndiannessMode` (:687)
2. `nextRecordBatch()`: loops over messages, consumes dictionaries first, then yields a batch  
   Entry: `nextRecordBatch` (:193)  
   Core: `readMessageOptional` (:281)  
   Dictionary branch: `ingestDictionaryBatchWithMap` (:1702)  
   RecordBatch branch: `buildRecordBatchFromFlatbuf` (:1119)

In one sentence: the Reader's responsibility is to "restore the schema and dictionary context in message order, then rebuild the column data".

### 8.3 C Data Import/Export Logic Chain

Corresponding file: `src/ffi/c_data.zig`

1. Schema:
- Export: `exportSchema` (:120) -> `releaseExportedSchema` (:338)
- Import: `importSchemaOwned` (:126)
2. Array:
- Export: `exportArray` (:153) -> `exportArrayRecursive` (:368) -> `releaseExportedArray` (:426)
- Import: `importArray` (:157) -> `importArrayRecursive` (:671)
3. Key lifetime points:
- After the importer takes ownership, it sets the input `c_array.release = null` (to avoid double release)
- `ownerRelease` (:108) is responsible for actually triggering the C-side release and destroying the owner

In one sentence: the core of FFI is not "copying data" but "strictly transferring ownership and ensuring the release callback is reentrant and recursive".

## 9. Primary Sources

- Arrow Columnar Format (main specification document, Version 1.5):  
  https://arrow.apache.org/docs/format/Columnar.html
- Arrow C Data Interface:  
  https://arrow.apache.org/docs/format/CDataInterface.html
- Arrow C Stream Interface:  
  https://arrow.apache.org/docs/format/CStreamInterface.html
- Arrow C++ IPC API:  
  https://arrow.apache.org/docs/cpp/api/ipc.html
- Arrow C++ Memory Management:  
  https://arrow.apache.org/docs/cpp/memory.html
- Arrow C++ C Interfaces (Import/Export C ABI):  
  https://arrow.apache.org/docs/cpp/api/c_abi.html
- arrow-rs crate docs (overall):  
  https://docs.rs/arrow/latest/arrow/
- arrow-ipc crate docs:  
  https://docs.rs/arrow-ipc/latest/arrow_ipc/
- arrow-array crate docs:  
  https://docs.rs/arrow-array/latest/arrow_array/
- arrow-data crate docs:  
  https://docs.rs/arrow-data/latest/arrow_data/
- arrow-buffer crate docs:  
  https://docs.rs/arrow-buffer/latest/arrow_buffer/
- nanoarrow documentation (lightweight C/C++ practice):  
  https://arrow.apache.org/nanoarrow/latest/getting-started/cpp.html
