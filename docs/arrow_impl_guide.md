# Apache Arrow Implementation Principles Explained: Rust and C++ Perspectives

This document is based on this project's source code (zarrow) and systematically explains the implementation principles of Apache Arrow in **Rust** (arrow-rs) and **C++** (Apache Arrow C++), covering the core data structures, IPC serialization, the C Data Interface FFI, and key optimization details.

---

## Table of Contents

1. [Arrow Columnar Storage Core Principles](#1-arrow-columnar-storage-core-principles)
2. [Memory Model and Buffer Management](#2-memory-model-and-buffer-management)
3. [Type System and Schema](#3-type-system-and-schema)
4. [Array Layout in Detail](#4-array-layout-in-detail)
5. [Builder Construction Patterns](#5-builder-construction-patterns)
6. [IPC Serialization Format](#6-ipc-serialization-format)
7. [C Data Interface (FFI)](#7-c-data-interface-ffi)
8. [Rust Implementation: arrow-rs](#8-rust-implementation-arrow-rs)
9. [C++ Implementation: Apache Arrow C++](#9-c-implementation-apache-arrow-c)
10. [Optimization Details](#10-optimization-details)
11. [Cross-Language Interoperability Summary](#11-cross-language-interoperability-summary)

---

## 1. Arrow Columnar Storage Core Principles

### 1.1 Row-Oriented vs. Column-Oriented

```
Row-oriented storage:
  Row0: [id=1, name="alice", age=30]
  Row1: [id=2, name="bob",   age=25]

Arrow column-oriented storage:
  id-buffer:   [1,   2  ]   <- contiguous int32 memory
  name-buffer: ["alice","bob"]
  age-buffer:  [30,  25 ]   <- contiguous int32 memory
```

Advantages of the columnar layout:
- **SIMD vectorization**: When operating on a column in bulk, the CPU can process 8 to 16 elements at once with AVX2/AVX-512.
- **Higher compression ratio**: Columns share the same data type and similar value ranges, so run-length and dictionary encoding are highly efficient.
- **Zero-copy sharing**: Multiple consumers share the same block of memory through pointers, with no serialization needed.

### 1.2 Physical Layout Hierarchy

```
Schema
  └─ Field[]          <- column name + type + nullable + metadata
        │
        ▼
RecordBatch
  └─ ArrayData[]      <- each column corresponds to one ArrayData tree
        │
        ▼
ArrayData
  ├─ data_type        <- logical type
  ├─ length           <- logical row count
  ├─ offset           <- slice offset (supports zero-copy slicing)
  ├─ null_count       <- cached null count
  ├─ buffers[]        <- raw byte buffers (validity/offsets/values)
  ├─ children[]       <- child ArrayData for nested types
  └─ dictionary       <- dictionary ArrayData for dictionary encoding
```

---

## 2. Memory Model and Buffer Management

### 2.1 64-Byte Alignment

The Arrow specification requires that all buffers be **aligned to 64 bytes**, and that the actual length be **rounded up to a multiple of 64 bytes** (padding).

```
Reasons:
  - AVX-512 SIMD instructions need 64-byte alignment to use the fastest aligned load (_mm512_load_ps)
  - A cache line is typically 64 bytes; alignment ensures single-element accesses do not cross a cache line
  - Padding ensures SIMD reads of the tail do not go out of bounds
```

Alignment calculation (this project, `src/buffer.zig`):

```zig
pub const ALIGNMENT: usize = 64;

pub fn alignedSize(size: usize) usize {
    // round up to a multiple of 64
    // e.g.: size=5  -> (5+63) & ~63 = 64
    // e.g.: size=64 -> (64+63) & ~63 = 64
    // e.g.: size=65 -> (65+63) & ~63 = 128
    return (size + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
}
```

**Rust (arrow-rs) equivalent implementation:**

```rust
// arrow-rs internally uses arrow_buffer::alloc::ALIGNMENT = 64
// Buffer::from_vec aligns automatically
use arrow_buffer::Buffer;

// manually allocate aligned memory
let layout = std::alloc::Layout::from_size_align(size, 64).unwrap();
let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
```

**C++ equivalent implementation:**

```cpp
// arrow/memory_pool.h
// DefaultMemoryPool uses posix_memalign or _aligned_malloc under the hood
ARROW_ASSIGN_OR_RAISE(auto buffer,
    arrow::AllocateBuffer(byte_size, pool));
// buffer->data() is guaranteed to be 64-byte aligned
assert(reinterpret_cast<uintptr_t>(buffer->data()) % 64 == 0);
```

### 2.2 SharedBuffer: Reference-Counted Shared Memory

A core feature of Arrow is **zero-copy**. Multiple ArrayData objects can share the same buffer, and the memory is only actually freed when the last reference is released.

This project's Zig implementation (`src/buffer.zig`):

```zig
// internal control block (similar to the C++ shared_ptr control block)
const BufferStorage = struct {
    allocator: std.mem.Allocator,
    data: []align(64) u8,
    ref_count: std.atomic.Value(u32),   // atomic reference count
    release_fn: *const fn (*BufferStorage) void,  // supports external memory (FFI)
};

pub const SharedBuffer = struct {
    storage: ?*BufferStorage,  // null means borrowed (not owned)
    data: []const u8,          // logical view, may be a subslice of storage.data

    // retain: increment the reference count
    pub fn retain(self: Self) Self {
        if (self.storage) |storage| {
            _ = storage.ref_count.fetchAdd(1, .monotonic);
        }
        return self;
    }

    // release: decrement the reference count; when it reaches zero, call release_fn to free the memory
    pub fn release(self: *Self) void {
        if (self.storage) |storage| {
            // acq_rel ensures writes made before release are visible to the thread that destroys the buffer
            if (storage.ref_count.fetchSub(1, .acq_rel) == 1) {
                storage.release_fn(storage);
            }
        }
        self.storage = null;
        self.data = &.{};
    }
};
```

**Rust equivalent (arrow_buffer::Buffer):**

```rust
// In arrow-rs, Buffer internally wraps Arc<Bytes>
// Arc provides thread-safe reference counting
#[derive(Clone, Debug, PartialEq)]
pub struct Buffer {
    data: Arc<Bytes>,      // reference-counted underlying byte block
    ptr: *const u8,        // pointer into data (supports slice offsets)
    length: usize,
}

// Zero-copy slicing: only changes ptr and length, does not copy data
impl Buffer {
    pub fn slice(&self, offset: usize) -> Self {
        assert!(offset <= self.len());
        Buffer {
            data: self.data.clone(),  // Arc clone: only increments the reference count
            ptr: unsafe { self.ptr.add(offset) },
            length: self.length - offset,
        }
    }
}
```

**C++ equivalent (arrow::Buffer):**

```cpp
// arrow/buffer.h
class Buffer {
 public:
    // shared_ptr provides reference counting
    static std::shared_ptr<Buffer> FromString(std::string data);

    const uint8_t* data() const { return data_; }
    int64_t size() const { return size_; }

    // zero-copy slicing
    static std::shared_ptr<Buffer> SliceBuffer(
        const std::shared_ptr<Buffer>& buffer,
        int64_t offset, int64_t length) {
        // The returned Buffer holds a shared_ptr to the original buffer;
        // its internal data_ pointer is just an offset, no new memory is allocated
        return std::make_shared<Buffer>(buffer, buffer->data() + offset, length);
    }
};
```

---

## 3. Type System and Schema

### 3.1 Type ID Enumeration

Arrow defines 45 logical types (this project, `src/datatype.zig`):

```zig
pub const TypeId = enum(u8) {
    null = 0,          // all-null column
    bool = 1,          // stored as a bitmap
    int8 = 3,          // 1 byte
    int32 = 7,         // 4 bytes
    int64 = 9,         // 8 bytes
    float = 11,        // IEEE 754 single precision
    double = 12,       // IEEE 754 double precision
    string = 13,       // UTF-8 variable length, i32 offsets
    binary = 14,       // arbitrary byte sequence
    timestamp = 18,    // int64 + timezone + time unit
    list = 25,         // variable-length nested list
    struct_ = 26,      // named-field struct
    dictionary = 29,   // dictionary encoding (indices + dictionary values)
    run_end_encoded=38,// run-end encoding
    string_view = 39,  // new format since Arrow 1.5, 16-byte fixed header
    // ...45 types in total
};
```

### 3.2 DataType Union

```zig
pub const DataType = union(TypeId) {
    // parameterless types use void directly
    int32: void,
    string: void,

    // parameterized types carry their configuration
    timestamp: TimestampType,   // { unit: TimeUnit, timezone: ?[]u8 }
    decimal128: DecimalParams,  // { precision: u8, scale: i32 }
    list: ListType,             // { value_field: Field }
    struct_: StructType,        // { fields: []Field }
    dictionary: DictionaryType, // { index_type, value_type, ordered }
    run_end_encoded: RunEndEncodedType,
    extension: ExtensionType,   // user-defined extension type
};
```

**Type system in Rust (arrow_schema::DataType):**

```rust
// arrow-rs: arrow_schema/src/datatype.rs
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DataType {
    Null,
    Boolean,
    Int8, Int16, Int32, Int64,
    UInt8, UInt16, UInt32, UInt64,
    Float16, Float32, Float64,
    Utf8, LargeUtf8,
    Binary, LargeBinary,
    // parameterized types
    Timestamp(TimeUnit, Option<Arc<str>>),  // unit + timezone
    Decimal128(u8, i8),                     // precision, scale
    List(Arc<Field>),
    Struct(Fields),                         // Fields = Arc<[Arc<Field>]>
    Dictionary(Box<DataType>, Box<DataType>),
    RunEndEncoded(Arc<Field>, Arc<Field>),
}

// Field contains name, type, nullable, metadata
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Field {
    name: Arc<str>,
    data_type: DataType,
    nullable: bool,
    metadata: HashMap<String, String>,
}

// Schema is an ordered collection of Fields
pub struct Schema {
    fields: Fields,
    metadata: HashMap<String, String>,
}
```

**Type system in C++:**

```cpp
// arrow/type.h
// C++ uses an inheritance hierarchy; all types derive from DataType
class DataType {
 public:
    virtual Type::type id() const = 0;
    virtual std::string ToString() const = 0;
    virtual bool Equals(const DataType& other) const = 0;
};

// concrete type example
class Int32Type : public NumberType<int32_t> {
 public:
    static constexpr Type::type type_id = Type::INT32;
    static constexpr int bit_width = 32;
};

class TimestampType : public TemporalType {
 public:
    TimestampType(TimeUnit::type unit, std::string timezone = "")
        : unit_(unit), timezone_(std::move(timezone)) {}
    TimeUnit::type unit() const { return unit_; }
    const std::string& timezone() const { return timezone_; }
};

// factory functions (the style used in this project's C++ interop)
auto int32_t  = arrow::int32();   // returns shared_ptr<Int32Type>
auto utf8_t   = arrow::utf8();    // returns shared_ptr<StringType>
auto ts_type  = arrow::timestamp(arrow::TimeUnit::MICRO, "UTC");
auto dict_t   = arrow::dictionary(arrow::int32(), arrow::utf8());
auto ree_t    = arrow::run_end_encoded(arrow::int32(), arrow::int32());
```

---

## 4. Array Layout in Detail

Arrow defines a precise buffer layout for every type. Understanding these layouts is the foundation of efficient computation.

### 4.1 Validity Bitmap (Null Bitmap)

Every nullable type has a validity bitmap, **with one bit per row**: 1 = valid, 0 = null.
Arrow uses **LSB-first** (least significant bit first) bit order.

```
Logical values:  [valid, null, valid, valid, null, valid, valid, valid]
Index:             0      1     2      3      4      5      6      7

Byte 0: bit0=1, bit1=0, bit2=1, bit3=1, bit4=0, bit5=1, bit6=1, bit7=1
        = 0b11011101 = 0xDD

Byte length = ceil(n / 8), trailing padding bits are 0
```

Bitmap implementation in this project (`src/bitmap.zig`):

```zig
// Read bit i: locate the byte, then check the corresponding bit
pub fn bitIsSet(data: []const u8, bit_index: usize) bool {
    const byte = data[bit_index >> 3];          // bit_index / 8
    const mask = @as(u8, 1) << @as(u3, @intCast(bit_index & 7));  // bit_index % 8
    return (byte & mask) != 0;
}

// Bulk count of valid bits (uses the popcount hardware instruction)
pub fn countSetBit(data: []const u8, bit_len: usize) usize {
    var count: usize = 0;
    const full_bytes = bit_len >> 3;
    for (data[0..full_bytes]) |byte| {
        count += @popCount(byte);  // compiles to a POPCNT instruction
    }
    // Handle the trailing partial byte
    if (bit_len & 7 > 0) {
        const mask = (@as(u8, 1) << @as(u3, @intCast(bit_len & 7))) - 1;
        count += @popCount(data[full_bytes] & mask);
    }
    return count;
}
```

**In Rust:**

```rust
// arrow_buffer::bit_util
pub fn get_bit(data: &[u8], i: usize) -> bool {
    (data[i >> 3] >> (i & 7)) & 1 != 0
}

pub fn count_set_bits(data: &[u8]) -> usize {
    data.iter().map(|b| b.count_ones() as usize).sum()
}
```

**In C++:**

```cpp
// arrow/util/bit_util.h
inline bool GetBit(const uint8_t* bits, int64_t i) {
    return (bits[i >> 3] >> (i & 0x07)) & 1;
}

// SIMD-accelerated version (uses AVX2 internally)
int64_t CountSetBits(const uint8_t* data, int64_t bit_offset, int64_t length);
```

### 4.2 Primitive Array (Fixed-Width Basic Types)

Layout: `[validity bitmap] [values buffer]`

```
int32 array [1, null, 30]:

Buffer 0 (validity): 0b00000101 = 0x05  (bit0=1, bit1=0, bit2=1)
Buffer 1 (values):   [01 00 00 00] [00 00 00 00] [1E 00 00 00]
                      ^^^^^^^^^^^^   ^^^^^^^^^^^^   ^^^^^^^^^^^^
                         i32=1         i32=0 (invalid)  i32=30
```

This project's Zig implementation (`src/array/primitive_array.zig`):

```zig
pub fn PrimitiveArray(comptime T: type) type {
    return struct {
        data: *const ArrayData,

        pub fn values(self: Self) []const T {
            // Buffer[1] is the values buffer, interpreted as type T
            const raw = self.data.buffers[1].typedSlice(T);
            // offset supports sliced views (zero-copy)
            return raw[self.data.offset .. self.data.offset + self.data.length];
        }

        pub fn isNull(self: Self, i: usize) bool {
            return self.data.isNull(i);  // reads the validity bitmap
        }
    };
}
```

**In Rust (arrow_array::PrimitiveArray):**

```rust
// arrow_array::array::primitive_array.rs
pub struct PrimitiveArray<T: ArrowPrimitiveType> {
    data: ArrayData,
    raw_values: RawPtrBox<T::Native>,  // raw pointer into the values buffer
}

impl<T: ArrowPrimitiveType> PrimitiveArray<T> {
    // Zero-cost access: direct pointer offset, no bounds check (unsafe internally)
    pub fn value(&self, i: usize) -> T::Native {
        assert!(i < self.len());
        // SAFETY: i < len has been verified
        unsafe { *self.raw_values.get().add(i) }
    }

    pub fn values(&self) -> &[T::Native] {
        // Return a slice reference over the entire values buffer
        let len = self.len();
        // SAFETY: ArrowPrimitiveType::Native guarantees a correct memory layout
        unsafe {
            std::slice::from_raw_parts(self.raw_values.get(), len)
        }
    }
}

// Usage example (from this project's tools/interop/arrow-rs/src/main.rs)
let ids: ArrayRef = Arc::new(Int32Array::from(vec![1, 2, 3]));
let ids = batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
assert_eq!(ids.value(0), 1);
assert!(ids.is_valid(0));
```

**In C++:**

```cpp
// arrow/array/array_primitive.h
// Usage example (from this project's tools/interop/cpp/interop.cpp)
arrow::Int32Builder id_builder;
ARROW_RETURN_NOT_OK(id_builder.AppendValues({1, 2, 3}));
std::shared_ptr<arrow::Array> ids;
ARROW_RETURN_NOT_OK(id_builder.Finish(&ids));

// Read values
auto int_arr = std::static_pointer_cast<arrow::Int32Array>(ids);
int32_t val = int_arr->Value(0);  // direct memory access
bool is_null = int_arr->IsNull(1);
```

### 4.3 String / Binary Array (Variable-Length Types)

Layout: `[validity] [offsets: i32 × (n+1)] [data: bytes]`

```
string array ["alice", null, "bob"]:

offsets: [0,  5,  5,  8]    <- n+1 = 4 int32 values
          |   |   |   |
data:    "alice" "bob"
          0    5      8

- row0: data[offsets[0]..offsets[1]] = data[0..5] = "alice"
- row1: is_null (validity bit = 0), data[5..5] = ""
- row2: data[offsets[2]..offsets[3]] = data[5..8] = "bob"
```

**In Rust:**

```rust
// arrow_array::StringArray internals
pub struct GenericStringArray<O: OffsetSizeTrait> {
    data: ArrayData,
    value_offsets: RawPtrBox<O>,  // i32 or i64
    value_data: RawPtrBox<u8>,
}

impl StringArray {
    pub fn value(&self, i: usize) -> &str {
        let offsets = self.value_offsets();
        let start = offsets[i] as usize;
        let end = offsets[i + 1] as usize;
        // SAFETY: Arrow guarantees UTF-8 validity
        unsafe {
            std::str::from_utf8_unchecked(&self.value_data()[start..end])
        }
    }
}

// Usage example (from this project's main.rs)
let names = Arc::new(StringArray::from(vec![Some("alice"), None, Some("bob")]));
let names = batch.column(1).as_any().downcast_ref::<StringArray>().unwrap();
assert_eq!(names.value(0), "alice");
assert!(names.is_null(1));
assert_eq!(names.value(2), "bob");
```

**In C++:**

```cpp
// Usage example (from this project's interop.cpp)
arrow::StringBuilder name_builder;
ARROW_RETURN_NOT_OK(name_builder.Append("alice"));
ARROW_RETURN_NOT_OK(name_builder.AppendNull());
ARROW_RETURN_NOT_OK(name_builder.Append("bob"));
std::shared_ptr<arrow::Array> names;
ARROW_RETURN_NOT_OK(name_builder.Finish(&names));

auto str_arr = std::static_pointer_cast<arrow::StringArray>(names);
std::string_view val = str_arr->GetView(0);  // zero-copy string_view
bool is_null = str_arr->IsNull(1);
```

### 4.4 Dictionary Array (Dictionary Encoding)

Layout: an indices buffer plus a separate dictionary array, which avoids storing repeated strings.

```
Color column ["red","blue","red","green","blue"] after dictionary encoding:

dictionary (StringArray): ["red", "blue", "green"]
indices   (Int32Array):   [0, 1, 0, 2, 1]

Memory savings: 5 × average string length → 5 × 4 bytes (indices) + 1 × dictionary
```

**Rust implementation (from this project's main.rs):**

```rust
// Build a dictionary array
let mut builder = StringDictionaryBuilder::<Int32Type>::new();
builder.append("red")?;
builder.append("blue")?;
let dict_array: ArrayRef = Arc::new(builder.finish());

// Read dictionary values: look up the dictionary by key
let dict = batch.column(0)
    .as_any()
    .downcast_ref::<DictionaryArray<Int32Type>>()
    .unwrap();
let keys = dict.keys();                           // Int32Array (indices)
let values = dict.values()                        // StringArray (dictionary)
    .as_any()
    .downcast_ref::<StringArray>()
    .unwrap();
let key_for_row0 = keys.value(0) as usize;        // = 0
let decoded = values.value(key_for_row0);          // = "red"
```

**C++ implementation (from this project's interop.cpp):**

```cpp
// Create a dictionary array from a StringArray via compute::dictionary_encode
auto dict_type = arrow::dictionary(arrow::int32(), arrow::utf8(), false);
ARROW_ASSIGN_OR_RAISE(
    auto encoded_datum,
    arrow::compute::CallFunction("dictionary_encode", {plain_string_array}));
auto dict_encoded = encoded_datum.make_array();

// Decode a single row
auto dict_arr = std::static_pointer_cast<arrow::DictionaryArray>(column);
auto keys    = std::static_pointer_cast<arrow::Int32Array>(dict_arr->indices());
auto values  = std::static_pointer_cast<arrow::StringArray>(dict_arr->dictionary());
int32_t key  = keys->Value(row);            // index
auto decoded = values->GetString(key);      // dictionary lookup
```

### 4.5 Run-End Encoded Array (Run-Length Encoding)

**REE** compresses consecutive equal values into `(run_end, value)` pairs. It has no independent buffer and only 2 child arrays.

```
Logical values:  [100, 100, 200, 200, 200]

run_ends (Int32Array): [2, 5]   <- row number where each run ends (exclusive)
values   (Int32Array): [100, 200]

To read row i: binary-search run_ends for the first position j where run_ends[j] >= i+1,
            the value is values[j]
```

**Rust implementation (from this project's main.rs):**

```rust
// Build an REE array
let run_ends = Int32Array::from(vec![2, 5]);
let values   = Int32Array::from(vec![100, 200]);
let ree: ArrayRef = Arc::new(
    RunArray::<Int32Type>::try_new(&run_ends, &values)?
);

// Iterate (runs are expanded automatically)
let typed = ree.downcast::<Int32Array>().unwrap();
let actual: Vec<Option<i32>> = typed.into_iter().collect();
// = [Some(100), Some(100), Some(200), Some(200), Some(200)]
```

**C++ implementation (from this project's interop.cpp):**

```cpp
// Build an REE array
arrow::Int32Builder run_ends_builder, values_builder;
ARROW_RETURN_NOT_OK(run_ends_builder.AppendValues({2, 5}));
ARROW_RETURN_NOT_OK(values_builder.AppendValues({100, 200}));
std::shared_ptr<arrow::Array> run_ends, values;
ARROW_RETURN_NOT_OK(run_ends_builder.Finish(&run_ends));
ARROW_RETURN_NOT_OK(values_builder.Finish(&values));

// The third argument is the logical length (5, not 2)
ARROW_ASSIGN_OR_RAISE(
    auto ree_arr,
    arrow::RunEndEncodedArray::Make(5, run_ends, values, /*offset=*/0));

// Read the child arrays (not expanded)
auto ree_data = batch->column(0)->data();
auto run_ends_child = arrow::MakeArray(ree_data->child_data[0]);
auto values_child   = arrow::MakeArray(ree_data->child_data[1]);
```

### 4.6 StringView / BinaryView (New Format in Arrow 1.5+)

Each element is represented by a **fixed 16-byte header**. Strings longer than 12 bytes are stored in variadic data buffers.

```
16-byte view structure (if length <= 12, stored inline):
  [length: i32][data: 12 bytes]   <- inline, does not reference an external buffer

If length > 12:
  [length: i32][prefix: 4 bytes][buf_index: i32][offset: i32]
                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                  points into the variadic data buffers

Advantage: short strings (<= 12 bytes) are accessed without a pointer jump, improving cache hit rates
```

Layout: `[validity] [views: u128 × n] [variadic data buffers…] [variadic lengths: i64 × n_bufs]`

### 4.7 Quick Reference: Memory Diagrams for All Types (45 Real Types)

Notes:
- `B0/B1/B2...` in the diagrams means `buffers[0]/buffers[1]/buffers[2]...`
- `validity` is optional: it can be omitted when `null_count == 0`
- When `offset` is present, logical index `i` actually accesses physical index `offset + i`

#### A. No-Parameter / Basic Type Families

1) Null

```text
┌──────────── NullArray ────────────┐
│ length = n, null_count = n        │
├────────────────────────────────────┤
│ buffers: []                        │
│ children: []                       │
└────────────────────────────────────┘
```

2) Boolean

```text
┌──────────── BooleanArray ─────────┐
│ B0: validity bitmap (optional)    │
│ B1: values bitmap (LSB-first)     │
└────────────────────────────────────┘
```

3) Int8
4) UInt8
5) Int16
6) UInt16
7) Int32
8) UInt32
9) Int64
10) UInt64
11) Float16
12) Float32
13) Float64
14) Date32
15) Date64
16) Time32
17) Time64
18) Timestamp
19) Duration

```text
┌──── Primitive Fixed-Width Array ───┐
│ B0: validity bitmap (optional)     │
│ B1: values (little-endian, fixed width) │
│     byte_width = 1/2/4/8           │
└─────────────────────────────────────┘
```

20) Decimal32
21) Decimal64
22) Decimal128
23) Decimal256

```text
┌────────── Decimal Array ───────────┐
│ B0: validity bitmap (optional)     │
│ B1: values (little-endian integer) │
│     width = 4/8/16/32 bytes        │
│     logical value = integer * 10^(-scale) │
└─────────────────────────────────────┘
```

24) FixedSizeBinary

```text
┌────── FixedSizeBinaryArray ────────┐
│ B0: validity bitmap (optional)     │
│ B1: values, byte_width bytes per element │
│     [elem0][elem1]...[elemN-1]     │
└─────────────────────────────────────┘
```

#### B. Variable-Length Binary / String Families

25) String (Utf8)
26) Binary

```text
┌──────── String/Binary Array ───────┐
│ B0: validity bitmap (optional)     │
│ B1: offsets i32, length n+1        │
│ B2: data bytes                     │
│ row i = data[offsets[i]..offsets[i+1]]
└─────────────────────────────────────┘
```

27) LargeString (LargeUtf8)
28) LargeBinary

```text
┌────── LargeString/LargeBinary ─────┐
│ B0: validity bitmap (optional)     │
│ B1: offsets i64, length n+1        │
│ B2: data bytes                     │
└─────────────────────────────────────┘
```

29) StringView
30) BinaryView

```text
┌────── StringView/BinaryView ───────┐
│ B0: validity bitmap (optional)     │
│ B1: views (16 bytes * n)           │
│     <=12 bytes: stored inline      │
│     >12 bytes: [prefix, buf_idx, off] -> variadic buffers
│ B2..Bk: variadic data buffers      │
│ B(k+1): variadic lengths (i64 each) │
└─────────────────────────────────────┘
```

#### C. Nested Structure Families

31) List

```text
┌────────────── ListArray ───────────┐
│ B0: validity bitmap (optional)     │
│ B1: offsets i32, length n+1        │
│ children[0]: values child array    │
│ row i -> child[offsets[i]..offsets[i+1]]
└─────────────────────────────────────┘
```

32) LargeList

```text
┌─────────── LargeListArray ─────────┐
│ B0: validity bitmap (optional)     │
│ B1: offsets i64, length n+1        │
│ children[0]: values child array    │
└─────────────────────────────────────┘
```

33) ListView
34) LargeListView

```text
┌──────────── ListView Array ────────┐
│ B0: validity bitmap (optional)     │
│ B1: offsets (i32 or i64)           │
│ B2: sizes   (i32 or i64)           │
│ children[0]: values child array    │
│ row i -> child[offsets[i] .. offsets[i] + sizes[i]]
└─────────────────────────────────────┘
```

35) FixedSizeList

```text
┌───────── FixedSizeListArray ───────┐
│ B0: validity bitmap (optional)     │
│ children[0]: values child array    │
│ fixed length per row = list_size   │
│ row i -> child[i*list_size .. (i+1)*list_size]
└─────────────────────────────────────┘
```

36) Struct

```text
┌──────────── StructArray ───────────┐
│ B0: validity bitmap (optional)     │
│ children[k]: one child per field   │
│ row i's struct = aggregate of each child at the same index │
└─────────────────────────────────────┘
```

37) Map

```text
┌────────────── MapArray ────────────┐
│ B0: validity bitmap (optional)     │
│ B1: offsets i32, length n+1        │
│ children[0]: entries StructArray   │
│   entries.children[0] = key array  │
│   entries.children[1] = item array │
│ row i -> entries[offsets[i]..offsets[i+1]]
└─────────────────────────────────────┘
```

38) SparseUnion

```text
┌─────────── SparseUnionArray ───────┐
│ B0: type_ids (i8 * n)              │
│ children[k]: every child has length n │
│ row i: selects a child based on type_ids[i] │
│ (no validity bitmap)               │
└─────────────────────────────────────┘
```

39) DenseUnion

```text
┌──────────── DenseUnionArray ───────┐
│ B0: type_ids (i8 * n)              │
│ B1: value_offsets (i32 * n)        │
│ children[k]: each child may have a compact length │
│ row i -> child(type_ids[i])[value_offsets[i]]
│ (no validity bitmap)               │
└─────────────────────────────────────┘
```

#### D. Encoded / Extension Families

40) Dictionary

```text
┌────────── DictionaryArray ─────────┐
│ B0: validity bitmap (optional)     │
│ B1: indices (int8/16/32/64)        │
│ dictionary: separate values array  │
│ row i -> dictionary[indices[i]]    │
└─────────────────────────────────────┘
```

41) RunEndEncoded

```text
┌────── RunEndEncodedArray (REE) ────┐
│ buffers: []                        │
│ children[0]: run_ends (increasing, >0) │
│ children[1]: values                │
│ row i: find the smallest j such that run_ends[j] > i │
│        return values[j]            │
└─────────────────────────────────────┘
```

42) Extension

```text
┌────────── ExtensionArray ──────────┐
│ physical layout is exactly the same as storage_type │
│ extra metadata: extension_name/metadata │
│ computation and transport follow storage_type │
└─────────────────────────────────────┘
```

#### E. Interval: Three Physical Layouts

43) Interval[Months]

```text
┌────── Interval(Months) Array ──────┐
│ B0: validity bitmap (optional)     │
│ B1: values int32 (months)          │
└─────────────────────────────────────┘
```

44) Interval[DayTime]

```text
┌────── Interval(DayTime) Array ─────┐
│ B0: validity bitmap (optional)     │
│ B1: values, 8 bytes per element    │
│     {days:i32, milliseconds:i32}   │
└─────────────────────────────────────┘
```

45) Interval[MonthDayNano]

```text
┌── Interval(MonthDayNano) Array ────┐
│ B0: validity bitmap (optional)     │
│ B1: values, 16 bytes per element   │
│     {months:i32, days:i32, nanos:i64}
└─────────────────────────────────────┘
```

Note: this section covers 45 real data types; the upper-bound sentinel is now a separate constant `max_type_id`, not an instantiable type.

### 4.8 One-Page Cheat Sheet (Print Version)

Purpose: compress all 45 types onto one page. Memorize by "layout template" first, then look up via "type mapping".

```text
Arrow Memory Layout One-Page Cheat Sheet

Legend:
  B0/B1/B2... = buffers[i]
  V = validity bitmap (optional when null_count==0)
  off = offset(s), val = values, ch = children, dict = dictionary

Templates (sorted by frequency of occurrence)

T0  []
    Null

T1  [V][bitmap-values]
    Boolean

T2  [V][fixed-width values]
    Int8/16/32/64, UInt8/16/32/64, Float16/32/64,
    Date32/64, Time32/64, Timestamp, Duration

T3  [V][decimal-values]
    Decimal32/64/128/256

T4  [V][fixed-size-bytes]
    FixedSizeBinary

T5  [V][off32(n+1)][data]
    String, Binary

T6  [V][off64(n+1)][data]
    LargeString, LargeBinary

T7  [V][views16*n][variadic-data...][variadic-lengths]
    StringView, BinaryView

T8  [V][off32(n+1)] + ch[0]
    List

T9  [V][off64(n+1)] + ch[0]
    LargeList

T10 [V][off][size] + ch[0]
    ListView, LargeListView

T11 [V] + ch[0] (row i => ch[i*list_size .. (i+1)*list_size])
    FixedSizeList

T12 [V] + ch[field0..fieldN]
    Struct

T13 [V][off32(n+1)] + ch[0]=entries(struct<key,item>)
    Map

T14 [type_ids] + ch[all len=n]            (no V)
    SparseUnion

T15 [type_ids][value_offsets] + ch[compact] (no V)
    DenseUnion

T16 [V][indices] + dict(values-array)
    Dictionary

T17 [] + ch[0]=run_ends, ch[1]=values
    RunEndEncoded

T18 same as storage_type + extension metadata
    Extension

T19 [V][int32 months]
    Interval[Months]

T20 [V][{days:i32, ms:i32} * n]
    Interval[DayTime]

T21 [V][{months:i32, days:i32, nanos:i64} * n]
    Interval[MonthDayNano]

Quick Rules:
  1) Variable-length = offsets + data
  2) Nested = the parent describes the slice range; the real values are in the children
  3) Encoded = Dictionary/REE fetch values indirectly via "index or run"
  4) Unions have no validity; null semantics are expressed by the type system/convention
  5) `max_type_id` is an internal upper-bound sentinel, not an instantiable type
```

---

## 5. Builder Construction Patterns

A Builder is a factory that constructs immutable Arrays. Internally it maintains **mutable** buffers, and `finish()` converts them into immutable SharedBuffers.

### 5.1 Rust Builder

```rust
// Rust: type-safe Builder API
use arrow_array::builder::Int32Builder;

let mut builder = Int32Builder::with_capacity(1024);
builder.append_value(1);       // append a valid value
builder.append_null();         // append a null
builder.append_value(30);      // append a valid value

// finish() consumes the builder and returns Arc<Int32Array>
let array: Int32Array = builder.finish();
// Memory has moved from the mutable Vec into an immutable Buffer (Arc)
```

**StringDictionaryBuilder (from this project's main.rs):**

```rust
// Automatically maintains the dictionary; repeated values use the same key
let mut builder = StringDictionaryBuilder::<Int32Type>::new();
builder.append("red")?;
builder.append("blue")?;
// batch 2: append a delta using an existing dictionary
let bootstrap = StringArray::from(vec!["red", "blue"]);
let mut builder2 = StringDictionaryBuilder::<Int32Type>::new_with_dictionary(
    1,            // initial capacity
    &bootstrap    // inherit the existing dictionary
)?;
builder2.append("green")?;  // only new words are added to the dictionary
```

### 5.2 C++ Builder

```cpp
// C++ Builder API (from this project's interop.cpp)
arrow::Int32Builder id_builder;
ARROW_RETURN_NOT_OK(id_builder.AppendValues({1, 2, 3}));
std::shared_ptr<arrow::Array> ids;
ARROW_RETURN_NOT_OK(id_builder.Finish(&ids));  // consumes the builder

// StringBuilder
arrow::StringBuilder name_builder;
ARROW_RETURN_NOT_OK(name_builder.Append("alice"));
ARROW_RETURN_NOT_OK(name_builder.AppendNull());
ARROW_RETURN_NOT_OK(name_builder.Append("bob"));
std::shared_ptr<arrow::Array> names;
ARROW_RETURN_NOT_OK(name_builder.Finish(&names));

// Bulk append (more efficient; capacity is grown once internally)
ARROW_RETURN_NOT_OK(id_builder.AppendValues(
    std::vector<int32_t>{1, 2, 3, 4, 5},
    std::vector<bool>{true, false, true, true, true}  // validity
));
```

### 5.3 Zig Builder Internals (This Project)

```zig
// src/array/primitive_array.zig
pub fn PrimitiveBuilder(comptime T: type, comptime dtype: DataType) type {
    return struct {
        values: OwnedBuffer,          // mutable value buffer
        validity: ?OwnedBuffer = null, // lazily initialized: not allocated when all values are valid
        len: usize = 0,
        null_count: usize = 0,

        pub fn append(self: *Self, value: T) !void {
            const next_len = self.len + 1;
            try self.ensureValuesCapacity(next_len);  // dynamic growth (2x)
            const slice = std.mem.bytesAsSlice(T, self.values.data);
            slice[self.len] = value;
            // When validity is null, all values are implicitly valid, so no bitmap write is needed
            try self.setValidBit(self.len);
            self.len = next_len;
        }

        pub fn appendNull(self: *Self) !void {
            const next_len = self.len + 1;
            try self.ensureValuesCapacity(next_len);
            // First null: lazily initialize the validity bitmap,
            // marking all existing elements as valid (memset 0xFF)
            try self.ensureValidityForNull(next_len);
            self.len = next_len;
        }

        pub fn finish(self: *Self) !ArrayRef {
            // OwnedBuffer -> SharedBuffer (transfers ownership and empties self)
            const validity_buf = if (self.validity) |*buf|
                try buf.toShared(bitmap.byteLength(self.len))
            else
                SharedBuffer.empty;
            const values_buf = try self.values.toShared(self.len * @sizeOf(T));
            // Construct immutable ArrayData
            // ...
        }
    };
}
```

**Key optimization: Lazy Validity (lazily initialized validity bitmap)**

The validity bitmap is allocated only when the first null appears. If all values are valid, the bitmap stays empty, which saves memory and speeds up processing.

---

## 6. IPC Serialization Format

Arrow defines two IPC formats for passing data across processes and networks. The data can be used without deserialization (zero-copy read).

### 6.1 Stream Format

```
+------------------+
| Schema Message   |  <- describes column names and types (FlatBuffers encoded)
+------------------+
| RecordBatch Msg  |  <- first batch of data
+------------------+
| RecordBatch Msg  |  <- more batches...
+------------------+
| EOS (4 zero bytes) |  <- end-of-stream marker
+------------------+

Structure of each message:
  [0xFFFFFFFF]       <- continuation marker (4 bytes)
  [metadata_len]     <- FlatBuffers metadata length (4 bytes, little-endian)
  [metadata]         <- FlatBuffers-encoded message header (aligned to 8 bytes)
  [body buffers]     <- raw data buffers (each buffer 8-byte aligned)
```

**Rust Stream write (from this project's main.rs):**

```rust
use arrow_ipc::writer::StreamWriter;

// Create the stream writer; it writes the Schema message automatically
let mut writer = StreamWriter::try_new(file, &schema)?;

// Write each RecordBatch
writer.write(&batch)?;

// Write the EOS marker and flush
writer.finish()?;
```

**Rust Stream read:**

```rust
use arrow_ipc::reader::StreamReader;

// Reads and parses the Schema automatically
let mut reader = StreamReader::try_new(file, None)?;
let schema: Arc<Schema> = reader.schema();

// Iterate over all RecordBatches (implements Iterator<Item=Result<RecordBatch>>)
while let Some(batch) = reader.next().transpose()? {
    println!("rows: {}", batch.num_rows());
}
```

**C++ Stream write (from this project's interop.cpp):**

```cpp
ARROW_ASSIGN_OR_RAISE(auto out, arrow::io::FileOutputStream::Open(path));
ARROW_ASSIGN_OR_RAISE(
    auto writer,
    arrow::ipc::MakeStreamWriter(out.get(), schema));

ARROW_RETURN_NOT_OK(writer->WriteRecordBatch(*batch));
ARROW_RETURN_NOT_OK(writer->Close());
ARROW_RETURN_NOT_OK(out->Close());
```

**C++ Stream read (from this project's interop.cpp):**

```cpp
ARROW_ASSIGN_OR_RAISE(auto in, arrow::io::ReadableFile::Open(path));
ARROW_ASSIGN_OR_RAISE(
    auto reader,
    arrow::ipc::RecordBatchStreamReader::Open(in));

std::shared_ptr<arrow::Schema> schema = reader->schema();
std::shared_ptr<arrow::RecordBatch> batch;
ARROW_RETURN_NOT_OK(reader->ReadNext(&batch));  // null indicates EOS
```

### 6.2 File Format (Random Access)

```
+------------------+
| "ARROW1\0"       |  <- magic number (8 bytes)
+------------------+
| Schema Message   |
+------------------+
| RecordBatch Msg  |
| RecordBatch Msg  |
| ...              |
+------------------+
| Footer           |  <- contains the file offset index of all RecordBatches
+------------------+
| Footer length    |  <- i32 (4 bytes)
+------------------+
| "ARROW1\0"       |  <- trailing magic number (8 bytes)
+------------------+

The File format supports random access:
  reader->num_record_batches()    <- total number of batches
  reader->ReadRecordBatch(i)      <- read any batch by index (seek + read)
```

**C++ File read (from this project's interop.cpp):**

```cpp
ARROW_ASSIGN_OR_RAISE(auto reader, arrow::ipc::RecordBatchFileReader::Open(in));
int n = reader->num_record_batches();
std::shared_ptr<arrow::RecordBatch> batch;
ARROW_ASSIGN_OR_RAISE(batch, reader->ReadRecordBatch(0));  // random access
```

### 6.3 Message Alignment and Padding

This project's `src/ipc/format.zig`:

```zig
pub const Alignment: usize = 8;  // IPC message body is aligned to 8 bytes

pub fn padLen(len: usize) usize {
    const rem = len % Alignment;
    return if (rem == 0) 0 else Alignment - rem;
}

// When writing a message, insert padding bytes between metadata and body
pub fn writePadding(writer: anytype, pad_len: usize) !void {
    var zeros: [Alignment]u8 = [_]u8{0} ** Alignment;
    // ...
}
```

---

## 7. C Data Interface (FFI)

The Arrow C Data Interface allows **zero-copy transfer** of Arrow data between different runtimes, without any serialization.
The specification defines two C structs: `ArrowSchema` and `ArrowArray`.

### 7.1 C Struct Definitions

This project's `src/ffi/c_data.zig`:

```zig
// extern structs that exactly match the Arrow C Data Interface specification
pub const ArrowSchema = extern struct {
    format: [*c]const u8,      // type format string (e.g. "i"=int32, "+s"=struct)
    name: [*c]const u8,        // field name
    metadata: [*c]const u8,    // binary key-value metadata (custom format)
    flags: i64,                // NULLABLE=2, DICT_ORDERED=1, MAP_KEYS_SORTED=4
    n_children: i64,
    children: [*c]?*ArrowSchema,
    dictionary: ?*ArrowSchema, // value schema for dictionary types
    release: ?*const fn (?*ArrowSchema) callconv(.c) void,  // lifetime callback
    private_data: ?*anyopaque, // used internally by the implementation
};

pub const ArrowArray = extern struct {
    length: i64,
    null_count: i64,           // -1 means unknown
    offset: i64,               // slice offset
    n_buffers: i64,
    n_children: i64,
    buffers: [*c]?*const anyopaque,  // array of pointers to the raw memory
    children: [*c]?*ArrowArray,
    dictionary: ?*ArrowArray,
    release: ?*const fn (?*ArrowArray) callconv(.c) void,
    private_data: ?*anyopaque,
};
```

### 7.2 Format Strings

```
Primitive types:
  "n" = null,    "b" = bool
  "c" = int8,    "C" = uint8
  "s" = int16,   "S" = uint16
  "i" = int32,   "I" = uint32
  "l" = int64,   "L" = uint64
  "e" = float16, "f" = float32, "g" = float64
  "u" = utf8,    "U" = large_utf8
  "z" = binary,  "Z" = large_binary
  "vu"= utf8_view, "vz"= binary_view

Temporal types:
  "tss:UTC"  = timestamp[second, "UTC"]
  "tsm:"     = timestamp[ms, no timezone]
  "ttu"      = time64[microsecond]
  "tDn"      = duration[nanosecond]
  "tiM"      = interval[months]

Precision types:
  "d:20,4"       = decimal128(precision=20, scale=4)
  "d:9,2,32"     = decimal32(9,2)
  "w:16"         = fixed_size_binary(16 bytes)

Nested types:
  "+l"       = list
  "+L"       = large_list
  "+s"       = struct
  "+m"       = map
  "+r"       = run_end_encoded
  "+us:0,1"  = sparse_union (type_ids=0,1)
  "+ud:0,1"  = dense_union
  "+w:4"     = fixed_size_list(4)
```

### 7.3 Ownership Transfer Semantics

Core rule of the C Data Interface: **whoever calls release owns the data**.

```zig
// Export (Producer: Zig -> external)
pub fn exportArray(allocator: std.mem.Allocator, arr: ArrayRef) Error!ArrowArray {
    const priv = try allocator.create(ExportedArrayPrivate);
    priv.* = .{
        .retained_ref = arr.retain(),   // increment reference count to prevent premature release
        .buffers_ptrs = ...,
    };
    return ArrowArray{
        .buffers = priv.buffers_ptrs.ptr,
        .release = releaseExportedArray,  // set the release callback
        .private_data = priv,
    };
    // Note: after exportArray returns, the original arr and the exported ArrowArray share the same memory
}

// Release callback (triggered when the consumer calls release)
fn releaseExportedArray(raw: ?*ArrowArray) callconv(.c) void {
    const priv: *ExportedArrayPrivate = @ptrCast(@alignCast(raw.?.private_data.?));
    // Release child arrays
    for (priv.children_storage) |*child| {
        if (child.release) |r| r(child);
    }
    // Release the Zig-side reference (may trigger buffer deallocation)
    var retained = priv.retained_ref;
    retained.release();  // decrement reference count
    priv.allocator.destroy(priv);
    raw.?.release = null;  // mark as released to prevent double release
}

// Import (Consumer: external -> Zig, zero-copy)
pub fn importArray(...) Error!ArrayRef {
    // Take over release responsibility (clear the original c_array.release)
    c_array.release = null;  // <- key: transfers ownership

    // Wrap the external pointer directly as a SharedBuffer without copying data
    const ptr: [*]const u8 = @ptrCast(c_array.buffers[i].?);
    buffers[i] = SharedBuffer.init(ptr[0..needed_len]);

    // Create an owner object; ref_count tracks all derived ArrayRefs
    // When the last ArrayRef is released, the original release function is called
}
```

**FFI in Rust (arrow-rs FFI module):**

```rust
use arrow_array::ffi::{FFI_ArrowArray, FFI_ArrowSchema};
use arrow_array::ffi_stream::ArrowArrayStreamReader;

// Export to C
let ffi_schema = FFI_ArrowSchema::try_from(&schema)?;
let ffi_array  = FFI_ArrowArray::try_from(&array_data)?;

// Pass raw pointers to C code
let schema_ptr = &ffi_schema as *const FFI_ArrowSchema;
let array_ptr  = &ffi_array  as *const FFI_ArrowArray;

// Import from C (transfers ownership)
let array_data = unsafe {
    arrow_array::ffi::from_ffi(ffi_array, &ffi_schema)?
};
```

**FFI in C++:**

```cpp
// arrow/c/bridge.h
// Export
struct ArrowArray c_array;
struct ArrowSchema c_schema;
ARROW_RETURN_NOT_OK(arrow::ExportArray(*array, &c_array, &c_schema));

// After handing off to external code, the external code calls c_array.release(&c_array) to free it

// Import
std::shared_ptr<arrow::Array> imported;
ARROW_ASSIGN_OR_RAISE(imported,
    arrow::ImportArray(&c_array, &c_schema));  // takes ownership
```

---

## 8. Rust Implementation: arrow-rs

### 8.1 Crate Structure

```
arrow/               <- Main facade crate (re-exports all sub-crates)
├── arrow_buffer     <- Buffer, MutableBuffer, ScalarBuffer
├── arrow_data       <- ArrayData (raw layout container)
├── arrow_schema     <- DataType, Field, Schema
├── arrow_array      <- Array types + Builders
└── arrow_ipc        <- IPC Stream/File reader and writer
    └── arrow_schema (FlatBuffers generated)
```

### 8.2 ArrayData: General-Purpose Layout Container

```rust
// arrow_data/src/data.rs
pub struct ArrayData {
    data_type: DataType,
    len: usize,
    null_count: usize,
    offset: usize,               // supports sliced views
    buffers: Vec<Buffer>,        // raw byte buffers
    child_data: Vec<ArrayData>,  // child arrays for nested types
    nulls: Option<NullBuffer>,   // wraps the validity bitmap
}

impl ArrayData {
    // Zero-copy slice: only changes offset and len
    pub fn slice(&self, offset: usize, length: usize) -> Self {
        ArrayData {
            offset: self.offset + offset,
            len: length,
            // All buffers are Clone (Arc::clone)
            buffers: self.buffers.clone(),
            ..self.clone()
        }
    }
}
```

### 8.3 Type-Safe Array Access

arrow-rs uses Rust's trait system to guarantee type safety:

```rust
// The ArrowPrimitiveType trait binds a Rust native type to an Arrow type
pub trait ArrowPrimitiveType: 'static + Send + Sync + Clone {
    type Native: NativeType;
    const DATA_TYPE: DataType;
}

// Concrete implementation
impl ArrowPrimitiveType for Int32Type {
    type Native = i32;
    const DATA_TYPE: DataType = DataType::Int32;
}

// PrimitiveArray<T> resolves its type at compile time, with no runtime dynamic dispatch
let arr: Int32Array = Int32Array::from(vec![1, 2, 3]);
let v: i32 = arr.value(0);  // The compiler knows this returns i32

// Dynamic dispatch requires a downcast
let arr: &dyn Array = batch.column(0).as_ref();
let int_arr = arr.as_any().downcast_ref::<Int32Array>().unwrap();
```

### 8.4 Error Handling: The Result Chain

arrow-rs uses `Result<T, ArrowError>` throughout:

```rust
// Example (error propagation from this project's main.rs)
fn generate(path: &Path, container: ContainerMode)
    -> Result<(), Box<dyn std::error::Error>>
{
    let schema = canonical_schema();
    let ids: ArrayRef = Arc::new(Int32Array::from(vec![1, 2, 3]));
    let names: ArrayRef = Arc::new(
        StringArray::from(vec![Some("alice"), None, Some("bob")]));

    // ? operator: errors propagate upward automatically
    let batch = RecordBatch::try_new(schema.clone(), vec![ids, names])?;
    let file  = File::create(path)?;
    let mut writer = StreamWriter::try_new(file, &schema)?;
    writer.write(&batch)?;
    writer.finish()?;
    Ok(())
}
```

---

## 9. C++ Implementation: Apache Arrow C++

### 9.1 Core Class Hierarchy

```
arrow::Array               <- Base class for all arrays (virtual interface)
├── arrow::PrimitiveArray<T>
│   ├── Int32Array
│   ├── DoubleArray
│   └── ...
├── arrow::BinaryArray      <- string / binary
├── arrow::LargeBinaryArray
├── arrow::StringArray      (= BinaryArray<int32_t>)
├── arrow::DictionaryArray
├── arrow::RunEndEncodedArray
├── arrow::StructArray
└── arrow::ListArray

arrow::ArrayBuilder         <- Base class for builders
├── Int32Builder
├── StringBuilder
├── DictionaryBuilder<>
└── ...
```

### 9.2 Status and Result Error Handling

C++ uses `arrow::Status` + `arrow::Result<T>` instead of exceptions:

```cpp
// Macros that simplify error handling (from this project's interop.cpp)

// ARROW_RETURN_NOT_OK(expr): equivalent to if (!st.ok()) return st;
ARROW_RETURN_NOT_OK(builder.Append(42));

// ARROW_ASSIGN_OR_RAISE(var, expr): similar to map/and_then on Result
ARROW_ASSIGN_OR_RAISE(auto out, arrow::io::FileOutputStream::Open(path));
// Expands to:
// auto result = arrow::io::FileOutputStream::Open(path);
// if (!result.ok()) return result.status();
// auto out = result.MoveValueUnsafe();

// Handling it explicitly is also possible
arrow::Status st = builder.Finish(&array);
if (!st.ok()) {
    std::cerr << "Error: " << st.ToString() << std::endl;
    return st;
}
```

### 9.3 Memory Management: MemoryPool

```cpp
// C++ abstracts memory allocation through MemoryPool
arrow::MemoryPool* pool = arrow::default_memory_pool();

// Custom memory pool (for tracking, limiting, or using jemalloc/mimalloc)
arrow::MemoryPool* jemalloc_pool = arrow::jemalloc_memory_pool();

// Allocate an aligned buffer
ARROW_ASSIGN_OR_RAISE(
    std::unique_ptr<arrow::Buffer> buf,
    arrow::AllocateBuffer(size_bytes, pool));
// buf->data() is aligned to 64 bytes

// Builders accept an optional MemoryPool
arrow::Int32Builder builder(pool);
```

### 9.4 Compute Function Framework

C++ provides vectorized compute functions (as used in this project's interop.cpp):

```cpp
// arrow/compute/api.h
// Dynamically dispatches to a registered function by name
ARROW_ASSIGN_OR_RAISE(
    auto result_datum,
    arrow::compute::CallFunction("dictionary_encode", {plain_array}));

// Type-safe APIs are also available
ARROW_ASSIGN_OR_RAISE(
    auto filtered,
    arrow::compute::Filter(array, boolean_filter));

ARROW_ASSIGN_OR_RAISE(
    auto sorted,
    arrow::compute::SortIndices(array));

// Built-in SIMD-accelerated implementations:
// - sum/mean/min/max (using AVX2)
// - cast (type conversion, vectorized)
// - compare (produces a boolean bitmap)
// - filter (selects rows based on a bitmap)
```

---

## 10. Optimization Details

### 10.1 Memory Alignment and SIMD

```
Purpose of 64-byte alignment + 64-byte padding:

Plain addition (scalar): processes 1 int32 per operation
for (int i = 0; i < n; i++) out[i] = a[i] + b[i];  // n operations

AVX2 vectorized (no alignment guarantee): processes 8 int32 per operation, but needs vmovdqu (unaligned)
AVX2 vectorized (64-byte aligned): processes 16 int32 per operation, using vmovdqa (aligned, faster)
AVX-512 (64-byte aligned): processes 16 int32 per operation, highest throughput

Measured speedup (int32 array sum, n=1M):
  Scalar:    ~2ms
  AVX2:      ~0.3ms  (about 7x)
  AVX-512:   ~0.15ms (about 13x)
```

### 10.2 Lazy Initialization of the Validity Bitmap

```
Traditional approach:
  Every append writes a validity bit -> wastes memory bandwidth even when all values are valid

Arrow approach (this project's PrimitiveBuilder):
  - The validity buffer starts as null
  - The bitmap is allocated only on the first appendNull() call
  - At that point, all of [0, len) is marked valid (memset 0xFF)
  - Then the current bit is cleared

Benefits:
  - All-valid column: zero memory overhead + zero bitmap writes
  - On read: when null_count==0, the bitmap check can be skipped
  - During SIMD processing: columns without nulls can use more aggressive branch-free implementations
```

### 10.3 Zero-Copy Slicing (Slice without Copy)

```
A slice operation only modifies offset and length; the reference count of every buffer is incremented by 1:

original: offset=0, length=100, buffer=[0..6400 bytes]
sliced:   offset=10, length=20, buffer=[same memory block]

Reading the i-th element:
  values()[i] = buffer[(offset + i) × sizeof(T)]

Costs of zero-copy:
  - The buffer must stay aligned (it was aligned before slicing, but the offset may be unaligned)
  - Handling unaligned offsets requires an extra copy kernel (handled in the compute layer)
```

### 10.4 Atomic Reference Counting and Memory Ordering

This project's buffer reference counting uses correct memory ordering:

```zig
// retain: only monotonic is needed (no data synchronization required)
_ = storage.ref_count.fetchAdd(1, .monotonic);

// release: requires acq_rel
// - release semantics: ensure all writes before the release are visible to the destroying thread
// - acquire semantics: when fetchSub returns 1 (about to destroy),
//                      see all writes other threads made before their releases
if (storage.ref_count.fetchSub(1, .acq_rel) == 1) {
    storage.release_fn(storage);
}
```

Equivalent to the implementation of `Arc<T>` in Rust:

```rust
// Logic equivalent to the std::sync::Arc source
fn drop(&mut self) {
    // Release: fetch_sub with Release ordering
    if self.inner().strong.fetch_sub(1, Release) != 1 {
        return;
    }
    // Acquire fence: synchronize with all Release decrements
    fence(Acquire);
    // Now it is safe to destroy the data
    unsafe { self.drop_slow() }
}
```

### 10.5 Dictionary Encoding Optimization

```
String column with low cardinality (e.g. status: "active"/"inactive"/"deleted"):

Raw storage:
  N rows × average 8 bytes = 8N bytes
  Each comparison: strcmp (O(len))

After dictionary encoding:
  indices: N × 4 bytes (int32) = 4N bytes
  dictionary: 3 × 8 bytes = 24 bytes
  Total: 4N + 24 bytes (~50% savings)

  Comparison becomes: integer comparison (O(1))
  Group by becomes: group by indices (3-5x faster than string hashing)
```

### 10.6 Run-End Encoding Compression Effect

```
Time-series data (1 million rows, each value repeated consecutively 1000 times):

Raw int32: 1 million × 4 bytes = 4MB
REE: 1000 pairs × (4+4) bytes = 8KB (99.8% savings)

Random access to row i:
  Binary search over the run_ends array (1000 elements), O(log 1000) ≈ 10 comparisons
  Far faster than a sequential scan
```

### 10.7 IPC Zero-Copy Reads

```
The body buffers of the Stream/File formats are mmap'ed directly into memory:

Traditional deserialization:
  File bytes -> parse -> allocate new memory -> copy data -> use
  
Arrow IPC zero-copy:
  mmap file -> validate metadata -> access directly via pointers
  (no extra memory allocated)

C++ implementation:
  auto mmap = arrow::io::MemoryMappedFile::Open(path, arrow::io::FileMode::READ);
  // The reader directly holds slices of the mmap'ed Buffer internally
  // RecordBatch column buffers point into the mmap'ed memory
```

### 10.8 Safe SIMD Reads of Padding Bytes

```
Arrow requires that padding bytes be 0

Reason: SIMD operations often read in 64-byte units, even when only some elements are valid.
If padding bytes were random, then:
  - sum operations would read in garbage values
  - bitmap operations would read in invalid bits

This project's OwnedBuffer allocation ensures zeroing:
  const data = try allocator.alignedAlloc(...);
  @memset(data, 0);  // <- Key: zero everything so padding is 0
```

---

## 11. Cross-Language Interoperability Summary

This project achieves interoperability between Zig / Rust / C++ / Python through the IPC format and the C Data Interface:

```
                 ┌─────────────┐
                 │   zarrow    │  <- This project (Zig)
                 │  (writer)   │
                 └──────┬──────┘
                        │ IPC Stream/File (.arrow files)
           ┌────────────┼────────────┐
           ▼            ▼            ▼
    ┌─────────────┐ ┌─────────┐ ┌──────────┐
    │  arrow-rs   │ │arrow C++│ │ pyarrow  │
    │   (Rust)    │ │  (C++)  │ │ (Python) │
    │ (validator) │ │(validor)│ │(generat.)│
    └─────────────┘ └─────────┘ └──────────┘

C Data Interface path (zero-copy, in-process):
  Zig exportArray() -> ArrowArray struct -> C++ ImportArray()
                                         -> Rust from_ffi()
                                         -> Python pa.Array._import_from_c()
```

### Key Consistency Points in Interoperability

| Point | Zig | Rust | C++ |
|---|---|---|---|
| Buffer alignment | 64 bytes | 64 bytes | 64 bytes |
| IPC alignment | 8 bytes | 8 bytes | 8 bytes |
| Bitmap order | LSB-first | LSB-first | LSB-first |
| Byte order | little-endian | little-endian | little-endian |
| Offset type | i32 (string) / i64 (large_string) | Same as left | Same as left |
| Padding | Filled with 0 | Filled with 0 | Filled with 0 |
| REE child order | [run_ends, values] | [run_ends, values] | [run_ends, values] |

---

## References

- [Apache Arrow Columnar Format Specification](https://arrow.apache.org/docs/format/Columnar.html)
- [Arrow IPC Format Specification](https://arrow.apache.org/docs/format/IPC.html)
- [Arrow C Data Interface Specification](https://arrow.apache.org/docs/format/CDataInterface.html)
- [arrow-rs source code](https://github.com/apache/arrow-rs)
- [Apache Arrow C++ Documentation](https://arrow.apache.org/docs/cpp/)
- Project source code: `src/` `tools/interop/`
