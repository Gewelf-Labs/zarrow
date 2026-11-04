# Apache Arrow Specification, Array Types, and Implementation Overview

## 0. Basic Concepts
### Memory Alignment Requirements

The Arrow specification requires that the starting address of every buffer be **64-byte aligned** (recommended; minimum 8-byte alignment). The length of each buffer must be a multiple of **8 bytes** (any shortfall is zero-padded with 0). This requirement allows SIMD operations to safely read boundary bytes at vector width.

### General Semantics of Buffers

Each Array is composed of the following metadata and buffer collection:

- `length`: number of logical elements (non-negative integer)
- `offset`: logical starting offset (used for slicing, non-negative integer, initially 0)
- `null_count`: number of null values (C++: -1 means unknown; Zig: null requires a scan to compute)
- `buffers`: ordered list of physical buffers; each type has a fixed count and fixed semantics
- `children`: list of child Arrays (used by nested types)
- `dictionary`: dictionary values Array (used by dictionary types)

The range of logical element indices is `[offset, offset + length)`. Data in the physical buffer at `[0, offset)` is legacy data or padding; it is valid but does not belong to the current Array.

### Validity Bitmap

The `buffers[0]` of almost all types is the validity bitmap. The rules are as follows:

- Uses **LSB (least significant bit) first** bit order: the i-th logical element corresponds to bit `i%8` of `byte[i/8]`
- **1 means valid**, 0 means null
- Number of bytes stored is `ceil((offset + length) / 8)`
- Padding bits in the range `[offset + length, ...)` must be **0**
- When `null_count == 0`, the validity bitmap may be omitted (`buffers[0]` is empty or null), and all elements are treated as valid
- The **Null type** and the **Union type** have no validity bitmap

---

## 1. The Unified Model of Arrow

At its core, Arrow can be viewed uniformly as:

```text
Schema
  -> Field[]
  -> RecordBatch
  -> ArrayData tree
       - data_type
       - length
       - offset
       - null_count
       - buffers[]
       - children[]
       - dictionary?
```

Key points:

- `Schema / Field / DataType` describe the logical structure
- `ArrayData` describes the physical memory
- `RecordBatch` groups equal-length columns into a batch
- IPC and FFI transfers are both essentially this `ArrayData` tree

General rules:

- Fixed-width numeric values use little-endian
- The validity bitmap uses LSB-first bit order
- Most types use `buffers[0]` to represent validity
- `offset` is the basis for zero-copy slicing
- Null and values buffers are decoupled

---

## 2. Schema, Field, and DataType

A `Field` typically contains:

- `name`
- `data_type`
- `nullable`
- `metadata`

Common `DataType` categories:

- `Null`
- `Boolean`
- fixed-width primitive: `Int*`, `UInt*`, `Float*`
- decimal: `Decimal32/64/128/256`
- temporal: `Date32/64`, `Time32/64`, `Timestamp`, `Duration`
- interval: `YearMonth`, `DayTime`, `MonthDayNano`
- binary families: `FixedSizeBinary`, `Binary`, `String`, `LargeBinary`, `LargeString`
- view families: `BinaryView`, `StringView`
- nested: `List`, `LargeList`, `FixedSizeList`, `ListView`, `LargeListView`, `Struct`, `Map`
- special: `SparseUnion`, `DenseUnion`, `Dictionary`, `RunEndEncoded`
- `Extension`

---

## 3. Array Types and Memory Layouts

### 3.1 Null

**Explanation**: `Null` means every element of the column is null. It is the simplest and most special Arrow type.

**Implementation**: Suitable as the minimal type for validating the `ArrayData` base path.

| Field | Content |
|---|---|
| `TypeID` | `Null` |
| `Physical buffers` | None |
| `children` | None |
| `Byte width` | No values buffer |

**Access formula**:

```text
value(i) = null
```

**Details and constraints**:
- `null_count` must equal `length`
- No validity bitmap
- No values buffer
- After a slice, only the `offset/length` semantics change; no additional physical data is required

**Memory diagram**:

```text
┌──────────────────────── NullArray ────────────────────────┐
│ length = 4                                                │
│ offset = 0                                                │
│ null_count = 4                                            │
├────────────────────────────────────────────────────────────┤
│ no buffers                                                │
│ no children                                               │
└────────────────────────────────────────────────────────────┘
```

**Implementation notes and common pitfalls**:
- Keep `null_count` correct at construction time; do not compute it afterwards
- Do not mistakenly allocate a validity bitmap for the `Null` type
- After a slice, if only a sub-view is kept, `length` and `null_count` must still be consistent

**C++ / Rust equivalents**:
- C++: `arrow::NullArray` / `arrow::NullBuilder` / `arrow::ArrayData`
- Rust: `arrow_array::NullArray` / `arrow_array::builder::NullBuilder` / `arrow_data::ArrayData` / `arrow_buffer::Buffer`

### 3.2 Boolean

**Explanation**: `Boolean` is a bit-packed boolean array. Unlike other fixed-width types, its values buffer is not byte-per-value.

**Implementation**: Usually requires a separate bitmap read/write utility. It is not suitable for reusing the ordinary fixed-width access logic directly.

| Field | Content |
|---|---|
| `TypeID` | `Bool` |
| `Physical buffers` | `buffer[0]` validity bitmap; `buffer[1]` values bitmap |
| `children` | None |
| `Byte width` | Validity is per bit; values are per bit |

**Access formula**:

```text
bit_index = offset + i
if validity(bit_index) == 0:
    null
else:
    values(bit_index) ? true : false
```

**Details and constraints**:
- `offset` applies to the bit index
- Both validity and values use LSB-first
- The values bit of a null slot does not affect semantics

**Memory diagram**:

```text
┌────────────────────── BooleanArray ───────────────────────┐
│ length = 4                                                │
│ offset = 0                                                │
│ null_count = 1                                            │
├────────────────────────────────────────────────────────────┤
│ buffer[0] validity bitmap                                 │
│   slot:  0  1  2  3                                       │
│   bit :  1  1  0  1                                       │
├────────────────────────────────────────────────────────────┤
│ buffer[1] values bitmap                                   │
│   slot:  0  1  2  3                                       │
│   bit :  1  0  1  1                                       │
└────────────────────────────────────────────────────────────┘
```

**Implementation notes and common pitfalls**:
- Bitmap reads and writes must uniformly use LSB-first, otherwise results will be wrong across languages
- `offset` is a bit offset, not a byte offset
- Do not merge the validity bitmap and the values bitmap into one buffer

**C++ / Rust equivalents**:
- C++: `arrow::BooleanArray` / `arrow::BooleanBuilder` / `arrow::ArrayData` / bitmap helpers in Arrow C++
- Rust: `arrow_array::BooleanArray` / `arrow_array::builder::BooleanBuilder` / `arrow_data::ArrayData` / `arrow_buffer::BooleanBuffer`

### 3.3 Fixed-Width Primitive

**Explanation**: This is the most basic and general group of physical layouts in Arrow. Integers, floats and similar types mostly share this template.

**Implementation**: Best implemented first, because their buffer rules are the most stable and they most easily carry over to later builder, slice and IPC logic.

| Field | Content |
|---|---|
| `TypeID` | `Int8/16/32/64`, `UInt8/16/32/64`, `HalfFloat`, `Float`, `Double` |
| `Physical buffers` | `buffer[0]` validity bitmap; `buffer[1]` values |
| `children` | None |
| `Byte width` | `Int8/UInt8=1`, `Int16/UInt16/HalfFloat=2`, `Int32/UInt32/Float=4`, `Int64/UInt64/Double=8` |

**Access formula**:

```text
idx = offset + i
if invalid(idx):
    null
else:
    read_little_endian<T>(buffer[1], idx * byte_width)
```

**Details and constraints**:
- Values use little-endian
- The bytes of a null slot may be any value
- Reads use `offset + i`
- The minimum buffer length should cover `(offset + length) * byte_width`

**Memory diagram**:

`Int32Array`

```text
┌──────────────────────── Int32Array ───────────────────────┐
│ length = 4                                                │
│ offset = 0                                                │
│ null_count = 1                                            │
├────────────────────────────────────────────────────────────┤
│ buffer[0] validity bitmap                                 │
│   slot:  0  1  2  3                                       │
│   bit :  1  0  1  1                                       │
├────────────────────────────────────────────────────────────┤
│ buffer[1] values (int32)                                  │
│   index:  0    1    2    3                                │
│   value: [10] [20] [30] [40]                              │
│   slot1 is null, but values[1] may be any value           │
└────────────────────────────────────────────────────────────┘
```

`Float64Array`

```text
┌──────────────────────── Float64Array ─────────────────────┐
│ length = 3                                                │
│ offset = 0                                                │
│ null_count = 1                                            │
├────────────────────────────────────────────────────────────┤
│ buffer[0] validity bitmap                                 │
│   slot:  0  1  2                                          │
│   bit :  1  0  1                                          │
├────────────────────────────────────────────────────────────┤
│ buffer[1] values (f64)                                    │
│   index:  0      1      2                                 │
│   value: [1.5] [9.9] [3.25]                               │
│   slot1 is null, but values[1] may be any value           │
└────────────────────────────────────────────────────────────┘
```

**Implementation notes and common pitfalls**:
- Abstract the fixed-width template first, then derive the concrete typed array views
- The bytes of a null slot must not take part in comparison, hashing or semantic decisions
- On slice, do not copy values; only move the logical start point

**C++ / Rust equivalents**:
- C++: `arrow::Int32Array`, `arrow::DoubleArray` and other typed arrays / `arrow::NumericBuilder<T>` / `arrow::ArrayData`
- Rust: `arrow_array::PrimitiveArray<T>`, `arrow_array::types::*` / `arrow_array::builder::PrimitiveBuilder<T>` / `arrow_data::ArrayData` / `arrow_buffer::ScalarBuffer`

### 3.4 Decimal

**Explanation**: Logically, Decimal is a decimal fixed-point number. Physically, it is still a fixed-width integer, which must be interpreted together with `precision/scale`.

**Implementation**: The fixed-width values buffer management can be reused, but `precision` and the type width must be additionally validated to match.

| Field | Content |
|---|---|
| `TypeID` | `Decimal32`, `Decimal64`, `Decimal128`, `Decimal256` |
| `Physical buffers` | `buffer[0]` validity bitmap; `buffer[1]` fixed-width integer storage |
| `children` | None |
| `Byte width` | `Decimal32=4`, `Decimal64=8`, `Decimal128=16`, `Decimal256=32` |

**Access formula**:

```text
idx = offset + i
if invalid(idx):
    null
else:
    raw = read_little_endian_integer(buffer[1], idx * byte_width, byte_width)
    value = raw * 10^(-scale)
```

**Details and constraints**:
- True value = `raw * 10^(-scale)`
- Memory is stored little-endian
- `precision` determines the maximum number of decimal digits allowed
- `Decimal32` has `precision ∈ [1, 9]`
- `Decimal64` has `precision ∈ [1, 18]`
- `Decimal128` has `precision ∈ [1, 38]`
- `Decimal256` has `precision ∈ [1, 76]`

**Memory diagram**:

`Decimal32Array`

```text
┌────────────────────── Decimal32Array ─────────────────────┐
│ precision = 5, scale = 1                                  │
│ length = 3, null_count = 1                                │
├────────────────────────────────────────────────────────────┤
│ buffer[0] validity bitmap                                 │
│   slot:  0  1   2                                         │
│   bit :  1  0   1                                         │
├────────────────────────────────────────────────────────────┤
│ buffer[1] raw values (i32)                                │
│   index:  0    1    2                                     │
│   raw  : [123] [77] [-45]                                 │
│   logical value: [12.3] [null] [-4.5]                     │
└────────────────────────────────────────────────────────────┘
```

`Decimal64Array`

```text
┌────────────────────── Decimal64Array ─────────────────────┐
│ precision = 10, scale = 2                                 │
│ length = 2                                                │
├────────────────────────────────────────────────────────────┤
│ buffer[0] validity bitmap omitted                         │
├────────────────────────────────────────────────────────────┤
│ buffer[1] raw values (i64)                                │
│   index:  0        1                                      │
│   raw  : [100001] [200]                                   │
│   logical value: [1000.01] [2.00]                         │
└────────────────────────────────────────────────────────────┘
```

`Decimal128Array`

```text
┌────────────────────── Decimal128Array ────────────────────┐
│ precision = 10, scale = 2                                 │
│ length = 3, null_count = 1                                │
├────────────────────────────────────────────────────────────┤
│ buffer[0] validity bitmap                                 │
│   slot:  0  1   2                                         │
│   bit :  1  0   1                                         │
├────────────────────────────────────────────────────────────┤
│ buffer[1] raw values (16 bytes per slot)                  │
│   index:  0       1      2                                │
│   raw  : [12345] [88] [-700]                              │
│   logical value: [123.45] [null] [-7.00]                  │
└────────────────────────────────────────────────────────────┘
```

`Decimal256Array`

```text
┌────────────────────── Decimal256Array ────────────────────┐
│ precision = 50, scale = 6                                 │
│ length = 2, null_count = 1                                │
├────────────────────────────────────────────────────────────┤
│ buffer[0] validity bitmap                                 │
│   slot:  0  1                                             │
│   bit :  1  0                                             │
├────────────────────────────────────────────────────────────┤
│ buffer[1] raw values (32 bytes per slot)                  │
│   slot0 -> huge_raw_0                                     │
│   slot1 -> huge_raw_1 (ignored because null)              │
└────────────────────────────────────────────────────────────┘
```

**Implementation notes and common pitfalls**:
- Width validation must consider precision/scale together, not only the buffer length
- Little-endian decoding of negative numbers and large integers is easy to get wrong
- Do not write a decimal's "display value" back into the underlying raw buffer semantics

  

**C++ / Rust equivalents**:
- C++: `arrow::Decimal32Array`, `arrow::Decimal64Array`, `arrow::Decimal128Array`, `arrow::Decimal256Array` / corresponding decimal builders / `arrow::ArrayData`
- Rust: `arrow_array::PrimitiveArray<Decimal*Type>` / `arrow_array::builder::Decimal*Builder` / `arrow_data::ArrayData` / `arrow_buffer::ScalarBuffer`

### 3.5 Date / Time / Timestamp / Duration

**Explanation**: These types have time semantics logically, and physically they are mostly `int32` or `int64`.

**Implementation**: The fixed-width template can be reused. The differences lie mainly in how schema parameters are interpreted, not in the buffer layout.

| Field | Content |
|---|---|
| `TypeID` | `Date32`, `Date64`, `Time32`, `Time64`, `Timestamp`, `Duration` |
| `Physical buffers` | `buffer[0]` validity bitmap; `buffer[1]` fixed-width integer storage |
| `children` | None |
| `Byte width` | `Date32=4`, `Date64=8`, `Time32=4`, `Time64=8`, `Timestamp=8`, `Duration=8` |

**Access formula**:

```text
idx = offset + i
if invalid(idx):
    null
else:
    raw = read_little_endian_integer(buffer[1], idx * byte_width, byte_width)
    interpret(raw, unit, timezone?)
```

**Details and constraints**:
- `Date32` counts days
- `Date64` counts milliseconds
- `Time32/64` are constrained by the unit parameter
- `Timestamp` may carry timezone metadata
- `Duration` is a signed time interval with a unit
- The unit of `Date32` is fixed to `DAY`
- The unit of `Date64` is fixed to `MILLISECOND`
- The value of `Date64` should be an integer multiple of `86_400_000`
- The unit of `Time32` can only be `SECOND` or `MILLISECOND`
- The unit of `Time64` can only be `MICROSECOND` or `NANOSECOND`
- The value range of `Time32` is `[0, 86400)` seconds or `[0, 86400000)` milliseconds, respectively
- The value range of `Time64` is `[0, 86400000000)` microseconds or `[0, 86400000000000)` nanoseconds, respectively
- The unit of `Timestamp` can be `SECOND`, `MILLISECOND`, `MICROSECOND` or `NANOSECOND`
- The timezone of `Timestamp` may be empty, an IANA timezone string, or a UTC offset string
- `timezone == null` means local time interpretation without a timezone
- `timezone == "UTC"` means UTC time interpretation
- The unit of `Duration` may also be `SECOND`, `MILLISECOND`, `MICROSECOND` or `NANOSECOND`

**Memory diagram**:

`Date32Array`

```text
┌──────────────────────── Date32Array ──────────────────────┐
│ values = int32 days since epoch                           │
│ length = 3, null_count = 1                                │
├────────────────────────────────────────────────────────────┤
│ buffer[0] validity bitmap                                 │
│   slot:  0      1    2                                    │
│   bit :  1      0    1                                    │
├────────────────────────────────────────────────────────────┤
│ buffer[1] values                                          │
│   index:  0       1     2                                 │
│   raw  : [19723] [0] [19725]                              │
│   logical date: [2024-01-01] [null] [2024-01-03]          │
└────────────────────────────────────────────────────────────┘
```

`Date64Array`

```text
┌──────────────────────── Date64Array ──────────────────────┐
│ values = int64 millis since epoch                         │
│ length = 2, null_count = 1                                │
├────────────────────────────────────────────────────────────┤
│ buffer[0] validity bitmap                                 │
│   slot:  0               1                                │
│   bit :  1               0                                │
├────────────────────────────────────────────────────────────┤
│ buffer[1] values                                          │
│   index:  0               1                               │
│   raw  : [1704067200000] [1]                              │
│   logical date: [2024-01-01] [null]                       │
└────────────────────────────────────────────────────────────┘
```

`Time32Array`

```text
┌──────────────────────── Time32Array ──────────────────────┐
│ unit = millisecond                                        │
│ length = 3, null_count = 1                                │
├────────────────────────────────────────────────────────────┤
│ buffer[0] validity bitmap                                 │
│   slot:  0     1   2                                      │
│   bit :  1     0   1                                      │
├────────────────────────────────────────────────────────────┤
│ buffer[1] values                                          │
│   index:  0      1   2                                    │
│   raw  : [1000] [9] [2500]                                │
│   logical time: [00:00:01.000] [null] [00:00:02.500]      │
└────────────────────────────────────────────────────────────┘
```

`Time64Array`

```text
┌──────────────────────── Time64Array ──────────────────────┐
│ unit = microsecond                                        │
│ length = 2                                                │
├────────────────────────────────────────────────────────────┤
│ buffer[0] validity bitmap omitted                         │
├────────────────────────────────────────────────────────────┤
│ buffer[1] values                                          │
│   index:  0         1                                     │

┌────────────────────────────────────────────────────────────┐
│   raw  : [1000000] [2000000]                              │
│   logical time: [00:00:01.000000] [00:00:02.000000]       │
└────────────────────────────────────────────────────────────┘
```

`TimestampArray`

```text
┌────────────────────── TimestampArray ─────────────────────┐
│ unit = millisecond, timezone = null                       │
│ length = 3, null_count = 1                                │
├────────────────────────────────────────────────────────────┤
│ buffer[0] validity bitmap                                 │
│   slot:  0              1   2                             │
│   bit :  1              0   1                             │
├────────────────────────────────────────────────────────────┤
│ buffer[1] values                                          │
│   index:  0              1   2                            │
│   raw  : [1700000000000] [0] [1700000000123]              │
│   logical ts: [t0] [null] [t0+123ms]                      │
└────────────────────────────────────────────────────────────┘
```

`DurationArray`

```text
┌──────────────────────── DurationArray ────────────────────┐
│ unit = millisecond                                        │
│ length = 3, null_count = 1                                │
├────────────────────────────────────────────────────────────┤
│ buffer[0] validity bitmap                                 │
│   slot:  0    1   2                                        │
│   bit :  1    0   1                                        │
├────────────────────────────────────────────────────────────┤
│ buffer[1] values                                          │
│   index:  0    1   2                                       │
│   raw  : [100] [7] [-25]                                  │
│   logical duration: [100ms] [null] [-25ms]                │
└────────────────────────────────────────────────────────────┘
```

**Implementation notes and common pitfalls**:
- The main differences between these types are in the schema parameters, not in the underlying buffer shape
- The timezone of `Timestamp` does not change the underlying integer; it only changes how the value is interpreted
- Do not confuse the units of `Date64` / `Time*` / `Duration` in readers and writers

**C++ / Rust equivalents**:
- C++: `arrow::Date32Array`, `arrow::Date64Array`, `arrow::Time32Array`, `arrow::Time64Array`, `arrow::TimestampArray`, `arrow::DurationArray` / corresponding builders / `arrow::ArrayData`
- Rust: `arrow_array::PrimitiveArray<Date32Type/Date64Type/Time32*Type/Time64*Type/Timestamp*Type/Duration*Type>` / corresponding `PrimitiveBuilder` / `arrow_data::ArrayData` / `arrow_buffer::ScalarBuffer`

### 3.6 Interval

**Explanation**: An Interval is a time span with several logical components, but physically it is still a fixed-width slot.

**Implementation**: When reading, the fixed-width field combination must be interpreted according to the specific interval variant.

| Field | Content |
|---|---|
| `TypeID` | `Interval(Months)`, `Interval(DayTime)`, `Interval(MonthDayNano)` |
| `Physical buffers` | `buffer[0]` validity bitmap; `buffer[1]` fixed-width interval storage |
| `children` | None |
| `Byte width` | `YearMonth=4`, `DayTime=8`, `MonthDayNano=16` |

**Access formula**:

```text
idx = offset + i
if invalid(idx):
    null
else:
    decode_interval_record(buffer[1], idx * byte_width, interval_kind)
```

**Details and constraints**:
- `YearMonth` is essentially a number of months
- `DayTime` usually represents days + milliseconds
- `MonthDayNano` represents months + days + nanoseconds
- The physical type of `YearMonth` is `int32`
- The physical type of `DayTime` is an 8-byte record `{days: int32, milliseconds: int32}`
- The physical type of `MonthDayNano` is a 16-byte record `{months: int32, days: int32, nanoseconds: int64}`
- All three support positive and negative interval values

**Memory diagrams**:

`IntervalYearMonthArray`

```text
┌────────────────── IntervalYearMonthArray ────────────────┐
│ length = 3, null_count = 1                                │
├────────────────────────────────────────────────────────────┤
│ buffer[0] validity bitmap                                 │
│   slot:  0   1   2                                         │
│   bit :  1   0   1                                         │
├────────────────────────────────────────────────────────────┤
│ buffer[1] values (int32 months)                           │
│   index:  0    1   2                                       │
│   raw  : [14] [1] [3]                                     │
│   logical interval: [1y2m] [null] [3m]                    │
└────────────────────────────────────────────────────────────┘
```

`IntervalDayTimeArray`

```text
┌─────────────────── IntervalDayTimeArray ──────────────────┐
│ length = 3, null_count = 1                                │
├────────────────────────────────────────────────────────────┤
│ buffer[0] validity bitmap                                 │
│   slot:  0   1   2                                         │
│   bit :  1   0   1                                         │
├────────────────────────────────────────────────────────────┤
│ buffer[1] values (8 bytes per slot)                       │
│   slot0 -> [days=1, ms=500]                               │
│   slot1 -> [days=9, ms=9] ignored                         │
│   slot2 -> [days=0, ms=200]                               │
└────────────────────────────────────────────────────────────┘
```

`IntervalMonthDayNanoArray`

```text
┌──────────────── IntervalMonthDayNanoArray ────────────────┐
│ length = 2, null_count = 1                                │
├────────────────────────────────────────────────────────────┤
│ buffer[0] validity bitmap                                 │
│   slot:  0   1                                             │
│   bit :  1   0                                             │
├────────────────────────────────────────────────────────────┤
│ buffer[1] values (16 bytes per slot)                      │
│   slot0 -> [months=1, days=2, nanos=30]                   │
│   slot1 -> [months=7, days=8, nanos=9] ignored            │
└────────────────────────────────────────────────────────────┘
```

**Implementation notes and common pitfalls**:
- The three interval types have different physical widths; readers must not be shared between them
- `DayTime` and `MonthDayNano` are both composite records; pay attention to field boundaries and order
- Comparison and formatting logic should be based on the logical structure; do not compare raw bytes directly

**C++ / Rust equivalents**:
- C++: `arrow::MonthIntervalArray`, `arrow::DayTimeIntervalArray`, `arrow::MonthDayNanoIntervalArray` / corresponding builders / `arrow::ArrayData`
- Rust: `arrow_array::PrimitiveArray<IntervalYearMonthType/IntervalDayTimeType/IntervalMonthDayNanoType>` / `PrimitiveBuilder` / `arrow_data::ArrayData`

### 3.7 FixedSizeBinary

**Explanation**: Each element is a fixed-length block of bytes, suitable for hashes, digests, and fixed-length encoding scenarios.

**Implementation**: Fixed-width index calculation can be reused, but the value itself is interpreted as `u8[]` rather than as a number.

| Field | Content |
|---|---|
| `TypeID` | `FixedSizeBinary` |
| `Physical buffers` | `buffer[0]` validity bitmap; `buffer[1]` values bytes |
| `children` | None |
| `Byte width` | Specified by `byte_width` in the schema |

**Access formula**:

```text
idx = offset + i
if invalid(idx):
    null
else:
    start = idx * byte_width
    end = start + byte_width
    value = buffer[1][start..end]
```

**Details and constraints**:
- The starting point of `slot i` is `(offset + i) * byte_width`
- The byte block for a null slot can be ignored
- `byte_width` must be greater than 0
- `buffer[1]` should cover at least `(offset + length) * byte_width` bytes

**Memory diagram**:

```text
┌────────────────── FixedSizeBinaryArray ───────────────────┐
│ byte_width = 2                                            │
│ length = 4, null_count = 1                                │
├────────────────────────────────────────────────────────────┤
│ buffer[0] validity bitmap                                 │
│   slot:  0  1  2  3                                       │
│   bit :  1  0  1  1                                       │
├────────────────────────────────────────────────────────────┤
│ buffer[1] values bytes                                    │
│   slot0 -> [AA BB]                                        │
│   slot1 -> [?? ??]                                        │
│   slot2 -> [CC DD]                                        │
│   slot3 -> [EE FF]                                        │
└────────────────────────────────────────────────────────────┘
```

**Implementation notes and common pitfalls**:
- `byte_width` is a schema constraint; do not guess it only from the runtime buffer length
- Slicing still cuts by element width; it is not an arbitrary byte slice
- The payload bytes of a null slot can be ignored; they do not need to be zeroed

**C++ / Rust equivalents**:
- C++: `arrow::FixedSizeBinaryArray` / `arrow::FixedSizeBinaryBuilder` / `arrow::ArrayData`
- Rust: `arrow_array::FixedSizeBinaryArray` / `arrow_array::builder::FixedSizeBinaryBuilder` / `arrow_data::ArrayData` / `arrow_buffer::Buffer`

### 3.8 String / Binary

**Explanation**: This is the classic variable-length layout: offsets + data.

**Implementation**: Strictly verify the monotonicity, count, and bounds of the offsets.

| Field | Content |
|---|---|
| `TypeID` | `String`, `Binary` |
| `Physical buffers` | `buffer[0]` validity bitmap; `buffer[1]` offsets(`i32`); `buffer[2]` data |
| `children` | None |
| `Byte width` | Offsets are 4 bytes each; data is stored contiguously, byte by byte |

**Access formula**:

```text
idx = offset + i
if invalid(idx):
    null
else:
    start = offsets_i32[idx]
    end = offsets_i32[idx + 1]
    value = data[start..end]
```

**Details and constraints**:
- At least `length + offset + 1` offset slots are required
- `String` requires the data region to conform to UTF-8
- `Binary` imposes no text constraint on the data content
- All values in the offsets must be non-negative
- The offsets must be monotonically non-decreasing
- `offsets[0]` is not required to be 0, because a slice can share the original buffer
- `buffer[2].len >= offsets[offset + length]`
- This layout is limited by `int32` offsets; total data is typically no more than about 2 GB

**Memory diagrams**:

`StringArray`

```text
┌──────────────────────── StringArray ──────────────────────┐
│ length = 4, null_count = 1                                │
├────────────────────────────────────────────────────────────┤
│ buffer[0] validity bitmap                                 │
│   slot:  0  1  2  3                                       │
│   bit :  1  0  1  1                                       │
├────────────────────────────────────────────────────────────┤
│ buffer[1] offsets (i32)                                   │
│   [0] [5] [5] [8] [8]                                     │
├────────────────────────────────────────────────────────────┤
│ buffer[2] data                                            │
│   "alicebob"                                              │
│   slot0 -> "alice"                                        │
│   slot1 -> null                                           │
│   slot2 -> "bob"                                          │
│   slot3 -> ""                                             │
└────────────────────────────────────────────────────────────┘
```

`BinaryArray`

```text
┌──────────────────────── BinaryArray ──────────────────────┐
│ length = 4, null_count = 1                                │
├────────────────────────────────────────────────────────────┤
│ buffer[0] validity bitmap                                 │
│   slot:  0  1  2  3                                       │
│   bit :  1  0  1  1                                       │
├────────────────────────────────────────────────────────────┤
│ buffer[1] offsets (i32)                                   │
│   [0] [2] [2] [3] [6]                                     │
├────────────────────────────────────────────────────────────┤
│ buffer[2] data                                            │
│   [01 02 AA FF 00 10]                                     │
│   slot0 -> [01 02]                                        │
│   slot1 -> null                                           │
│   slot2 -> [AA]                                           │
│   slot3 -> [FF 00 10]                                     │
└────────────────────────────────────────────────────────────┘
```

**Implementation notes and common pitfalls**:
- The offsets count must be at least `offset + length + 1`
- The offsets must be non-decreasing, and the last offset must not run past the end of the data buffer
- `String` also needs UTF-8 validation; `Binary` does not

**C++ / Rust equivalents**:
- C++: `arrow::StringArray`, `arrow::BinaryArray` / `arrow::StringBuilder`, `arrow::BinaryBuilder` / `arrow::ArrayData`
- Rust: `arrow_array::StringArray`, `arrow_array::BinaryArray` / `arrow_array::builder::StringBuilder`, `BinaryBuilder` / `arrow_data::ArrayData` / `arrow_buffer::OffsetBuffer`

### 3.9 LargeString / LargeBinary

**Explanation**: Structurally identical to the regular variable-length layout, except that the offsets are extended to `i64`.

**Implementation**: The logic is the same as `String/Binary`; pay attention to 64-bit offsets and larger capacity scenarios.

| Field | Content |
|---|---|
| `TypeID` | `LargeString`, `LargeBinary` |
| `Physical buffers` | `buffer[0]` validity bitmap; `buffer[1]` offsets(`i64`); `buffer[2]` data |
| `children` | None |
| `Byte width` | Offsets are 8 bytes each; data is byte by byte |

**Access formula**:

```text
idx = offset + i
if invalid(idx):
    null
else:
    start = offsets_i64[idx]
    end = offsets_i64[idx + 1]
    value = data[start..end]
```

**Details and constraints**:
- Suitable for very large data buffers
- The read formula is the same as `String/Binary`; only the offset type differs
- All values in the offsets must be non-negative
- The offsets must be monotonically non-decreasing
- `buffer[2].len >= offsets[offset + length]`
- Compared with the regular variable-length layout, the only core difference is that the offset width changes from `int32` to `int64`

**Memory diagrams**:

`LargeStringArray`

```text
┌────────────────────── LargeStringArray ───────────────────┐
│ length = 2                                                │
├────────────────────────────────────────────────────────────┤
│ buffer[0] validity bitmap omitted                         │
├────────────────────────────────────────────────────────────┤
│ buffer[1] offsets (i64)                                   │
│   [0] [5] [10]                                            │
├────────────────────────────────────────────────────────────┤
│ buffer[2] data = "helloworld"                             │
│   slot0 -> "hello"                                        │
│   slot1 -> "world"                                        │
└────────────────────────────────────────────────────────────┘
```

`LargeBinaryArray`

```text
┌────────────────────── LargeBinaryArray ───────────────────┐
│ length = 2                                                │
├────────────────────────────────────────────────────────────┤
│ buffer[0] validity bitmap omitted                         │
├────────────────────────────────────────────────────────────┤
│ buffer[1] offsets (i64)                                   │
│   [0] [1] [4]                                             │
├────────────────────────────────────────────────────────────┤
│ buffer[2] data = [AA BB CC DD]                            │
│   slot0 -> [AA]                                           │
│   slot1 -> [BB CC DD]                                     │
└────────────────────────────────────────────────────────────┘
```

**Implementation notes and common pitfalls**:
- Apart from the offset width becoming `i64`, do not introduce a different set of semantics
- With very large data, watch the usize / i64 conversion boundaries
- In writers / IPC, do not accidentally encode the large variant with regular 32-bit offsets

**C++ / Rust equivalents**:
- C++: `arrow::LargeStringArray`, `arrow::LargeBinaryArray` / `arrow::LargeStringBuilder`, `arrow::LargeBinaryBuilder` / `arrow::ArrayData`
- Rust: `arrow_array::LargeStringArray`, `arrow_array::LargeBinaryArray` / `arrow_array::builder::LargeStringBuilder`, `LargeBinaryBuilder` / `arrow_data::ArrayData` / `arrow_buffer::OffsetBuffer<i64>`

### 3.10 StringView / BinaryView

**Explanation**: The View family is not offsets + data. Instead, each slot holds a fixed-size view record that either inlines a small value or references a data region.

**Implementation**: The view record must be parsed first; the variable-length offsets model cannot be applied directly.

| Field | Content |
|---|---|
| `TypeID` | `StringView`, `BinaryView` |
| `Physical buffers` | `buffer[0]` validity bitmap; `buffer[1]` views; optional additional data region |
| `children` | None |
| `Byte width` | Each view record has a fixed width; long payloads reside in an external data region |

**Access formula**:

```text
idx = offset + i
if invalid(idx):
    null
else:
    view = read_view_record(buffer[1], idx)
    if view.is_inline:
        value = view.inline_bytes[0..view.length]
    else:
        value = external_data[view.start .. view.start + view.length]
```

**Details and constraints**:
- Small values can be inlined
- Large values are referenced through a pointer/offset-style description of the data region

- Validation must ensure that the view does not go out of bounds
- Each view record in `buffer[1]` is a fixed 16 bytes
- When `length <= 12`, the view can inline the value content directly
- When `length > 12`, the view records a prefix, buffer index, and offset
- On the long-value path, `buf_index >= 0` and `offset >= 0`
- On the long-value path, `offset + length <= variadic_buffer.len`
- The long-value prefix must match the first 4 bytes of the actual value
- The number of `buffers[2..]` is usually recorded in IPC through the variadic buffer count
- Different elements may legally share the same variadic buffer, even referencing overlapping ranges

**Memory diagram**:

`StringViewArray`

```text
┌────────────────────── StringViewArray ────────────────────┐
│ length = 3, null_count = 1                                │
├────────────────────────────────────────────────────────────┤
│ buffer[0] validity bitmap                                 │
├────────────────────────────────────────────────────────────┤
│ buffer[1] views (fixed-size records)                      │
│   slot0 -> inline small string "hi"                       │
│   slot1 -> external data reference                        │
│   slot2 -> null                                           │
├────────────────────────────────────────────────────────────┤
│ extra data region stores long payload bytes               │
└────────────────────────────────────────────────────────────┘
```

`BinaryViewArray`

```text
┌────────────────────── BinaryViewArray ────────────────────┐
│ length = 3, null_count = 1                                │
├────────────────────────────────────────────────────────────┤
│ buffer[0] validity bitmap                                 │
├────────────────────────────────────────────────────────────┤
│ buffer[1] views (fixed-size records)                      │
│   slot0 -> inline bytes [AA BB]                           │
│   slot1 -> external data reference                        │
│   slot2 -> null                                           │
├────────────────────────────────────────────────────────────┤
│ extra data region stores long payload bytes               │
└────────────────────────────────────────────────────────────┘
```

**Implementation notes and common pitfalls**:
- This is one of the types most easily mistaken for the offsets template
- Test both the inline and out-of-line paths
- When a view record points into the external data region, bounds checking must be strict

**C++ / Rust cross-reference**:
- C++: `arrow::StringViewArray`, `arrow::BinaryViewArray` / corresponding view-family builder / `arrow::ArrayData`
- Rust: `arrow_array::StringViewArray`, `arrow_array::BinaryViewArray` / corresponding view-family builder / `arrow_data::ArrayData` / `arrow_buffer::Buffer`

### 3.11 List / LargeList

**Explanation**: The parent array uses offsets to describe which segment of the child values each slot corresponds to.

**Implementation**: When reading a value, the parent offsets must be mapped to a child slice, and a child array view must be constructed.

| Field | Content |
|---|---|
| `TypeID` | `List`, `LargeList` |
| `Physical buffers` | `buffer[0]` validity bitmap; `buffer[1]` offsets; `children[0]` values child |
| `children` | 1 |
| `Byte width` | `List` offsets=4; `LargeList` offsets=8; the child is determined by its own type |

**Access formula**:

```text
idx = offset + i
if invalid(idx):
    null
else:
    start = offsets[idx]
    end = offsets[idx + 1]
    value = child.slice(start, end - start)
```

**Details and constraints**:
- `offsets[i]..offsets[i+1]` is the logical range within the child
- The offsets for null slots must still be valid
- Offsets must be monotonically non-decreasing
- All values in offsets must be non-negative
- `children[0].length >= offsets[offset + length]`
- `List` uses `int32` offsets, `LargeList` uses `int64`

**Memory diagram**:

`ListArray`

```text
┌───────────────────────── ListArray ───────────────────────┐
│ length = 4, null_count = 1                                │
├────────────────────────────────────────────────────────────┤
│ buffer[0] validity bitmap                                 │
│   slot:  0  1  2  3                                       │
│   bit :  1  0  1  1                                       │
├────────────────────────────────────────────────────────────┤
│ buffer[1] offsets (i32)                                   │
│   [0] [2] [2] [2] [3]                                     │
├────────────────────────────────────────────────────────────┤
│ child[0] values = Int32Array [1, 2, 3]                    │
│   slot0 -> child[0..2]                                    │
│   slot1 -> null                                           │
│   slot2 -> child[2..2]                                    │
│   slot3 -> child[2..3]                                    │
└────────────────────────────────────────────────────────────┘
```

`LargeListArray`

```text
┌─────────────────────── LargeListArray ────────────────────┐
│ length = 2                                                │
├────────────────────────────────────────────────────────────┤
│ buffer[0] validity bitmap omitted                         │
├────────────────────────────────────────────────────────────┤
│ buffer[1] offsets (i64)                                   │
│   [0] [1] [3]                                             │
├────────────────────────────────────────────────────────────┤
│ child[0] values = Int32Array [10, 20, 30]                 │
│   slot0 -> [10]                                           │
│   slot1 -> [20, 30]                                       │
└────────────────────────────────────────────────────────────┘
```

**Implementation notes and common pitfalls**:
- The relationship between parent offsets and child length must be validated
- The offset/length of the child slice must be propagated correctly
- The offsets for null slots must still be valid

**C++ / Rust cross-reference**:
- C++: `arrow::ListArray`, `arrow::LargeListArray` / `arrow::ListBuilder`, `arrow::LargeListBuilder` / `arrow::ArrayData`
- Rust: `arrow_array::ListArray`, `arrow_array::LargeListArray` / `arrow_array::builder::ListBuilder` / `arrow_data::ArrayData` / `arrow_buffer::OffsetBuffer`

### 3.12 FixedSizeList

**Explanation**: Each slot corresponds to a fixed number of elements in the child, so offsets are no longer needed.

**Implementation**: The most important step is verifying that the child capacity covers `(offset + length) * list_size`.

| Field | Content |
|---|---|
| `TypeID` | `FixedSizeList` |
| `Physical buffers` | `buffer[0]` validity bitmap; `children[0]` values child |
| `children` | 1 |
| `Byte width` | The parent node has no fixed values width; the logical width per slot is specified by `list_size` |

**Access formula**:

```text
idx = offset + i
if invalid(idx):
    null
else:
    start = idx * list_size
    value = child.slice(start, list_size)
```

**Details and constraints**:
- `slot i` maps to a fixed span in the child
- The physical child segment for a null slot may be ignored
- `list_size` must be greater than 0
- `children[0].length` should at least cover `(offset + length) * list_size`

**Memory diagram**:

```text
┌────────────────────── FixedSizeListArray ─────────────────┐
│ list_size = 2                                             │
│ length = 3, null_count = 1                                │
├────────────────────────────────────────────────────────────┤
│ buffer[0] validity bitmap                                 │
│   slot:  0  1  2                                          │
│   bit :  1  0  1                                          │
├────────────────────────────────────────────────────────────┤
│ child[0] values = Int32Array                              │
│   [1, 2, ??, ??, 5, 6]                                    │
│   slot0 -> child[0..2]                                    │
│   slot1 -> ignored because null                           │
│   slot2 -> child[4..6]                                    │
└────────────────────────────────────────────────────────────┘
```

**Implementation notes and common pitfalls**:
- `list_size` is a schema constant and must not be inferred from the data
- Child capacity validation is the easiest to miss
- After slicing, the mapping from logical index to child must be recomputed based on `offset + i`

**C++ / Rust cross-reference**:
- C++: `arrow::FixedSizeListArray` / `arrow::FixedSizeListBuilder` / `arrow::ArrayData`
- Rust: `arrow_array::FixedSizeListArray` / `arrow_array::builder::FixedSizeListBuilder` / `arrow_data::ArrayData`

### 3.13 ListView / LargeListView

**Explanation**: Compared with ordinary lists, this family explicitly records the start and length of each slot.

**Implementation**: Offsets and sizes must be validated together; the logical model is more like an explicit view table.

| Field | Content |
|---|---|
| `TypeID` | `ListView`, `LargeListView` |
| `Physical buffers` | `buffer[0]` validity bitmap; `buffer[1]` offsets; `buffer[2]` sizes; `children[0]` values child |
| `children` | 1 |
| `Byte width` | `ListView` offsets/sizes=4; `LargeListView` offsets/sizes=8 |

**Access formula**:

```text
idx = offset + i
if invalid(idx):
    null
else:
    start = offsets[idx]
    len = sizes[idx]
    value = child.slice(start, len)
```

**Details and constraints**:
- Start and length are recorded independently
- An empty list is represented by size=0
- Null slots must still carry valid metadata
- Offsets are not required to be monotonic
- View ranges of different rows may overlap
- `offsets[idx] >= 0`
- `sizes[idx] >= 0`
- `offsets[idx] + sizes[idx] <= children[0].length + children[0].offset`
- `ListView` uses `int32` offsets/sizes, `LargeListView` uses `int64`
- By specification, the offset/size of a null slot may be uninterpreted, but implementations should still keep them as valid values

**Memory diagram**:

`ListViewArray`

```text
┌──────────────────────── ListViewArray ────────────────────┐
│ length = 4, null_count = 1                                │
├────────────────────────────────────────────────────────────┤
│ buffer[0] validity bitmap                                 │
│   slot:  0  1  2  3                                       │
│   bit :  1  0  1  1                                       │
├────────────────────────────────────────────────────────────┤
│ buffer[1] offsets = [0, 2, 2, 3]                          │
│ buffer[2] sizes   = [2, 0, 1, 0]                          │
├────────────────────────────────────────────────────────────┤
│ child[0] values = Int32Array [1, 2, 3]                    │
│   slot0 -> [1, 2]                                         │
│   slot1 -> null                                           │
│   slot2 -> [3]                                            │
│   slot3 -> []                                             │
└────────────────────────────────────────────────────────────┘
```

`LargeListViewArray`

```text
┌────────────────────── LargeListViewArray ─────────────────┐
│ length = 3, null_count = 1                                │
├────────────────────────────────────────────────────────────┤
│ buffer[0] validity bitmap                                 │
│   slot:  0  1  2                                          │
│   bit :  1  1  0                                          │
├────────────────────────────────────────────────────────────┤
│ buffer[1] offsets (i64) = [0, 1, 3]                       │
│ buffer[2] sizes   (i64) = [1, 2, 0]                       │
├────────────────────────────────────────────────────────────┤
│ child[0] values = Int32Array [10, 20, 30]                 │
│   slot0 -> [10]                                           │
│   slot1 -> [20, 30]                                       │
│   slot2 -> null                                           │
└────────────────────────────────────────────────────────────┘
```

**Implementation notes and common pitfalls**:
- Unlike ordinary lists, the length here comes from `sizes[idx]`; do not borrow `offsets[idx+1] - offsets[idx]`
- Offsets and sizes must both be bounds-checked
- Metadata for null slots must also stay valid

**C++ / Rust cross-reference**:
- C++: `arrow::ListViewArray`, `arrow::LargeListViewArray` / corresponding list-view builders / `arrow::ArrayData`
- Rust: `arrow_array::ListViewArray`, `arrow_array::LargeListViewArray` / corresponding list-view builders / `arrow_data::ArrayData`

### 3.14 Struct

**Explanation**: A Struct stores no values itself; all fields are distributed across the children.

**Implementation**: The parent-level validity controls whether the whole struct exists; each child can also have its own nulls.

| Field | Content |
|---|---|
| `TypeID` | `Struct` |
| `Physical buffers` | `buffer[0]` validity bitmap |
| `children` | `n = field count` |
| `Byte width` | The parent node has no fixed values width; each child is determined by its own type |

**Access formula**:

```text
idx = offset + i
if invalid(idx):
    null
else:
    value = {
        field_0 = child_0[idx],
        field_1 = child_1[idx],
        ...
    }
```

**Details and constraints**:
- A parent null does not require the corresponding child positions to also be null
- On read, check the parent validity first, then decide whether to interpret the children
- The struct parent node has no values buffer of its own
- The order of all children must match the schema field order

**Memory diagram**:

```text
┌──────────────────────── StructArray ──────────────────────┐
│ length = 3, null_count = 1                                │
├────────────────────────────────────────────────────────────┤
│ buffer[0] validity bitmap                                 │
│   slot:  0  1  2                                          │
│   bit :  1  0  1                                          │
├────────────────────────────────────────────────────────────┤
│ child[0] = Int32Array  [1, 2, 3]                          │
│ child[1] = StringArray ["aa", "bb", "cc"]                 │
│   slot0 -> {id=1, name="aa"}                              │
│   slot1 -> null                                           │
│   slot2 -> {id=3, name="cc"}                              │
│ parent slot1 is null; child bytes remain but ignored      │
└────────────────────────────────────────────────────────────┘
```

**Implementation notes and common pitfalls**:
- Parent null and child null are two separate layers of semantics and cannot substitute for each other
- Do not require the child positions to be zeroed just because the parent is null
- The number, order, and types of fields must match the schema

**C++ / Rust cross-reference**:
- C++: `arrow::StructArray` / `arrow::StructBuilder` / `arrow::ArrayData`
- Rust: `arrow_array::StructArray` / `arrow_array::builder::StructBuilder` / `arrow_data::ArrayData`

### 3.15 Map

**Explanation**: Physically, a Map is usually equivalent to `list<struct<key, value>>`.

**Implementation**: The list + struct validation logic can be reused; only map semantics are applied at the schema level.

| Field | Content |
|---|---|
| `TypeID` | `Map` |
| `Physical buffers` | `buffer[0]` validity bitmap; `buffer[1]` offsets; `children[0]` entries struct |
| `children` | 1 entries child |
| `Byte width` | offsets=4; the entries child is determined by the struct's children |

**Access formula**:

```text
idx = offset + i
if invalid(idx):
    null
else:
    start = offsets[idx]
    end = offsets[idx + 1]
    value = entries.slice(start, end - start)
```

**Details and constraints**:
- The key is usually required to be non-null
- Each map slot corresponds to a range within the entries child
- Physically, it can be understood directly as `list<struct<key, value>>`
- `children[0]` must be an entries struct
- The entries struct must contain exactly two children, key and value

**Memory diagram**:

```text
┌───────────────────────── MapArray ────────────────────────┐
│ length = 3, null_count = 1                                │
├────────────────────────────────────────────────────────────┤
│ buffer[0] validity bitmap                                 │
│   slot:  0  1  2                                          │
│   bit :  1  0  1                                          │
├────────────────────────────────────────────────────────────┤
│ buffer[1] offsets = [0, 2, 2, 3]                          │
├────────────────────────────────────────────────────────────┤
│ child[0] = entries StructArray                            │
│   entries.key   = Int32Array [1, 2, 3]                    │
│   entries.value = Int32Array [10, 20, 30]                 │
│   slot0 -> {1:10, 2:20}                                   │
│   slot1 -> null                                           │
│   slot2 -> {3:30}                                         │
└────────────────────────────────────────────────────────────┘
```

**Implementation notes and common pitfalls**:
- In implementation, it is best to truly understand it as `list<struct<key,value>>`

- Do not drop the key non-null constraint
- Validate offsets together with the entries child

**C++ / Rust comparison**:
- C++: `arrow::MapArray` / `arrow::MapBuilder` / `arrow::ArrayData`
- Rust: `arrow_array::MapArray` / `arrow_array::builder::MapBuilder` / `arrow_data::ArrayData`

### 3.16 DenseUnion

**Explanation**: Each slot points to a position in a specific child through `type_id + value_offset`.

**Implementation**: Do not treat a union as a struct with nulls; it is closer to a "tagged dispatcher".

| Field | Content |
|---|---|
| `TypeID` | `DenseUnion` |
| `Physical buffers` | `buffer[0]` type_ids; `buffer[1]` value_offsets |
| `children` | `n` union children |
| `Byte width` | type_id usually 1; value_offset usually 4 |

**Access formula**:

```text
idx = offset + i
type_id = type_ids[idx]
child = child_for_type_id(type_id)
child_idx = value_offsets[idx]
value = child[child_idx]
```

**Details and constraints**:
- A child does not need to have the same length as the parent
- The mapping from `type_id` to child comes from the schema
- A dense union has no ordinary validity bitmap
- Every `value_offset` must fall within the valid range of its corresponding child

**Memory diagram**:

```text
┌────────────────────── DenseUnionArray ────────────────────┐
│ length = 3                                                │
├────────────────────────────────────────────────────────────┤
│ buffer[0] type_ids      = [5, 7, 5]                       │
│ buffer[1] value_offsets = [0, 0, 1]                       │
├────────────────────────────────────────────────────────────┤
│ child[Int32]  = [1, 2]                                    │
│ child[String] = ["aa"]                                    │
│ slot0 -> Int32[0], slot1 -> String[0], slot2 -> Int32[1]  │
│ logical values -> [1, "aa", 2]                            │
└────────────────────────────────────────────────────────────┘
```

**Implementation notes and common pitfalls**:
- The `type_id -> child` mapping comes from the schema and should not be hardcoded in the reader
- Out-of-range `value_offsets` is a high-risk bug for dense unions
- Unions do not use an ordinary validity bitmap; do not apply null rules to them by mistake

**C++ / Rust comparison**:
- C++: `arrow::DenseUnionArray` / `arrow::DenseUnionBuilder` / `arrow::ArrayData`
- Rust: `arrow_array::UnionArray` (dense mode) / corresponding union builder / `arrow_data::ArrayData`

### 3.17 SparseUnion

**Explanation**: Every child has the same length as the parent; `type_id` only determines which child the current slot reads from.

**Implementation**: It needs one fewer offsets buffer than a dense union, but has more empty child slots.

| Field | Content |
|---|---|
| `TypeID` | `SparseUnion` |
| `Physical buffers` | `buffer[0]` type_ids |
| `children` | `n` union children |
| `Byte width` | type_id usually 1 |

**Access formula**:

```text
idx = offset + i
type_id = type_ids[idx]
child = child_for_type_id(type_id)
value = child[idx]
```

**Details and constraints**:
- All children have the same length as the parent
- For slots whose child is not the current type, the value must be ignored
- A sparse union has no ordinary validity bitmap
- Each child's length should match the parent's length

**Memory diagram**:

```text
┌────────────────────── SparseUnionArray ───────────────────┐
│ length = 3                                                │
├────────────────────────────────────────────────────────────┤
│ buffer[0] type_ids = [5, 7, 5]                            │
├────────────────────────────────────────────────────────────┤
│ child[Int32]  = [1, ?, 2]                                 │
│ child[String] = ["", "aa", ""]                            │
│ every child has parent length                             │
│ logical values -> [1, "aa", 2]                            │
└────────────────────────────────────────────────────────────┘
```

**Implementation notes and common pitfalls**:
- All children have the same length as the parent, so memory usage is usually higher than a dense union
- Child slots for non-active branches must be ignored
- The type id dispatch table must match the schema

**C++ / Rust comparison**:
- C++: `arrow::SparseUnionArray` / `arrow::BasicUnionBuilder` or the corresponding sparse-union builder / `arrow::ArrayData`
- Rust: `arrow_array::UnionArray` (sparse mode) / corresponding union builder / `arrow_data::ArrayData`

### 3.18 Dictionary

**Explanation**: The parent array stores indices; the real values are stored in the `dictionary` array.

**Implementation**: The lifetimes of the index buffer and the dictionary values must both be managed; in IPC, dictionary batches and deltas must also be handled.

| Field | Content |
|---|---|
| `TypeID` | `Dictionary` |
| `Physical buffers` | `buffer[0]` validity bitmap; `buffer[1]` indices; `dictionary` values array |
| `children` | No ordinary children, but has `dictionary` |
| `Byte width` | Indices are determined by the index type: 1/2/4/8; dictionary values are determined by the value type |

**Access formula**:

```text
idx = offset + i
if invalid(idx):
    null
else:
    dict_idx = read_index(buffer[1], idx, index_type)
    value = dictionary[dict_idx]
```

**Details and constraints**:
- Indices must be within the dictionary range
- The dictionary value type can be primitive, string, list, struct, etc.
- The index type must be an integer type
- The schema usually also carries a dictionary id and an ordered flag
- `ordered` affects semantic interpretation but does not change the base physical layout

**Memory diagram**:

```text
┌────────────────────── DictionaryArray ────────────────────┐
│ index_type = int32                                        │
│ length = 4, null_count = 1                                │
├────────────────────────────────────────────────────────────┤
│ buffer[0] validity bitmap                                 │
│   slot:  0  1  2  3                                       │
│   bit :  1  0  1  1                                       │
├────────────────────────────────────────────────────────────┤
│ buffer[1] indices = [1, 0, 0, 1]                          │
├────────────────────────────────────────────────────────────┤
│ dictionary = StringArray                                  │
│   dict[0] = "red"                                         │
│   dict[1] = "blue"                                        │
│   logical values -> ["blue", null, "red", "blue"]         │
└────────────────────────────────────────────────────────────┘
```

**Implementation notes and common pitfalls**:
- The index buffer width is determined by the dictionary index type; do not default to int32
- The lifetime of the dictionary values must be tied to the parent
- In IPC, dictionary batches, replacements, deltas, and state replay during random access must also be handled

**C++ / Rust comparison**:
- C++: `arrow::DictionaryArray` / dictionary-capable builders (commonly paired with `DictionaryBuilder` or compute encode) / `arrow::ArrayData`
- Rust: `arrow_array::DictionaryArray<K>` / `arrow_array::builder::StringDictionaryBuilder`, `PrimitiveDictionaryBuilder`, etc. / `arrow_data::ArrayData`

### 3.19 RunEndEncoded

**Explanation**: REE uses two children, `run_ends` and `values`, to express the compressed logical sequence.

**Implementation**: On read, first locate which run the current logical index belongs to, then fetch the value from the `values` child.

| Field | Content |
|---|---|
| `TypeID` | `RunEndEncoded` |
| `Physical buffers` | No direct buffers |
| `children` | `child[0]` run_ends; `child[1]` values |
| `Byte width` | run_ends are determined by the run-end type, commonly 4 or 8; values are determined by the value type |

**Access formula**:

```text
logical_idx = offset + i
run_idx = lower_bound(run_ends, logical_idx + 1)
value = values[run_idx]
```

**Details and constraints**:
- `run_ends` must be monotonically increasing
- The last run_end should cover the logical length
- This is the read model that differs most from an ordinary array
- A REE parent node has no direct buffers; all data lives in its two children
- The `run_ends` child and the `values` child should have the same length

**Memory diagram**:

```text
┌──────────────────── RunEndEncodedArray ───────────────────┐
│ length = 5                                                │
├────────────────────────────────────────────────────────────┤
│ no direct buffers                                         │
├────────────────────────────────────────────────────────────┤
│ child[0] run_ends = Int32Array [2, 5]                     │
│ child[1] values   = Int32Array [100, 200]                 │
│ [0,2) -> 100, [2,5) -> 200                                │
│ logical values -> [100, 100, 200, 200, 200]               │
└────────────────────────────────────────────────────────────┘
```

**Implementation notes and common pitfalls**:
- Looking up `run_ends` usually requires binary search; do not implement hot paths as linear scans
- The last run_end must cover the entire logical length
- The writer must ensure the run merging rules are correct at construction time

**C++ / Rust comparison**:
- C++: `arrow::RunEndEncodedArray` / run-end-encoded builder / `arrow::ArrayData`
- Rust: `arrow_array::RunArray<RunEndType>` / corresponding run-array construction helpers / `arrow_data::ArrayData`

### 3.20 Extension

**Explanation**: Extension is not a new physical layout; it adds custom semantics on top of an existing storage type.

**Implementation**: Layout validation goes through the storage type; the extension name and metadata are responsible for semantic identification.

| Field | Content |
|---|---|
| `TypeID` | `Extension` |
| `Physical buffers` | Inherited from the storage type |
| `children` | Inherited from the storage type |
| `Byte width` | Inherited from the storage type |

**Access formula**:

```text
value(i) = interpret_as_extension(storage_value(i), extension_name, metadata)
```

**Details and constraints**:
- The extension name determines type identity
- Metadata can carry a version or extra parameters
- In IPC / FFI, extension information is usually passed through metadata
- The physical layout of an extension is always exactly the same as its storage type
- In IPC, extension information is usually transmitted through schema / field metadata
- In the C Data Interface, extension information is also usually transmitted through schema metadata

**Memory diagram**:

```text
┌──────────────────────── ExtensionArray ───────────────────┐
│ extension name = "com.example.int32_ext"                  │
│ storage type   = Int32                                    │
│ length = 3, null_count = 1                                │
├────────────────────────────────────────────────────────────┤
│ physical layout is identical to Int32Array                │
├────────────────────────────────────────────────────────────┤
│ buffer[0] validity bitmap                                 │
│   slot:  0  1  2                                          │
│   bit :  1  0  1                                          │
│ buffer[1] values = [7, ?, 11]                             │
│ logical values -> [ext(7), null, ext(11)]                 │
└────────────────────────────────────────────────────────────┘
```

**Implementation notes and common pitfalls**:
- An extension is not a new layout; implementing the storage type correctly matters most
- Identity depends on the extension name and metadata; do not look only at the storage type
- Extension metadata must be preserved across IPC / FFI round trips

**C++ / Rust comparison**:
- C++: The storage maps to ordinary `arrow::*Array` types; extension semantics usually live in `arrow::ExtensionType` / `Field` metadata / `arrow::ArrayData`
- Rust: The storage maps to ordinary `arrow_array::*Array` types; extension semantics usually live in `arrow_schema::DataType` metadata or an extension wrapper / `arrow_data::ArrayData`

### 3.21 Summary table

| Type group | TypeID | buffers | children | Typical byte width |
|---|---|---:|---:|---|
| Null | `Null` | 0 | 0 | none |
| Boolean | `Bool` | 2 | 0 | bit-packed |
| Fixed-width primitive | `Int*` `UInt*` `Float*` | 2 | 0 | 1/2/4/8 |
| Decimal | `Decimal*` | 2 | 0 | 4/8/16/32 |
| Date/Time/Timestamp/Duration | temporal | 2 | 0 | 4 or 8 |
| Interval | interval | 2 | 0 | 4/8/16 |
| FixedSizeBinary | `FixedSizeBinary` | 2 | 0 | set by schema |
| String/Binary | `String` `Binary` | 3 | 0 | offsets 4 |
| LargeString/LargeBinary | `LargeString` `LargeBinary` | 3 | 0 | offsets 8 |
| StringView/BinaryView | `StringView` `BinaryView` | 2+ | 0 | view record fixed |
| List/LargeList | `List` `LargeList` | 2 | 1 | offsets 4/8 |
| FixedSizeList | `FixedSizeList` | 1 | 1 | list_size determines logical width |
| ListView/LargeListView | `ListView` `LargeListView` | 3 | 1 | offsets/sizes 4/8 |
| Struct | `Struct` | 1 | N | determined by children |
| Map | `Map` | 2 | 1 | offsets 4 |
| SparseUnion | `SparseUnion` | 1 | N | type_ids 1 |
| DenseUnion | `DenseUnion` | 2 | N | type_ids 1, offsets 4 |
| Dictionary | `Dictionary` | 2 + dictionary | 0 | indices 1/2/4/8 |
| RunEndEncoded | `RunEndEncoded` | 0 | 2 | run_end 4/8 |
| Extension | `Extension` | inherited | inherited | inherited |

---

## 4. Builder and Memory Management

The responsibility of a Builder is to safely produce a valid `ArrayData`, not simply to mirror the final array type.

It is usually responsible for:

- allocating and growing buffers
- writing values / offsets / validity
- managing child builders
- producing a structurally correct array on finish

Common optimizations:

- lazy allocation of the validity bitmap
- reference counting of shared buffers
- slicing without copying, only adjusting `offset` and `length`

---

## 5. Validation Checklist

An implementation must verify at least the following:

- the number of buffers matches what the type requires
- the number of children matches what the type requires
- the validity bitmap covers `[offset, offset + length)`
- the fixed-width values buffer is large enough
- the offsets array is monotonic and does not go out of bounds
- child boundaries for list/map/struct/union are correct
- dictionary indices are not out of bounds
- the REE `run_ends` are strictly increasing

---

## 6. IPC Stream

The IPC stream suits sequential transmission. Its structure is roughly:

```text
[message length / continuation]
[flatbuffer metadata]
[body]
...
[EOS]
```

Core messages:

- `Schema`
- `DictionaryBatch`
- `RecordBatch`

Implementation focus:

- continuation marker
- separation of metadata and body
- body padding
- EOS handling

---

## 7. IPC File

The IPC file adds a footer that supports random access, on top of the stream-style message area:

```text
magic
messages...
footer
footer_length
magic
```

Implementation focus:

- validation of the header / trailer magic
- validation of the footer block index
- checks on block offset / metadata length / body length
- `recordBatchCount`
- random reads and rebuilding of dictionary state

---

## 8. Dictionary Batch and Delta

Dictionary encoding introduces additional state in IPC:

- the schema binds a field to a dictionary id
- dictionary values are sent separately through dictionary batches
- record batches store only the indices

The reader needs to maintain:

- `dictionary_id -> dictionary_values`
- delta merging
- replay of prior state when random-accessing a file

---

## 9. C Data Interface

The core structures of the FFI:

- `ArrowSchema`
- `ArrowArray`
- `ArrowArrayStream`

Meaning:

- `ArrowSchema` expresses the logical type
- `ArrowArray` expresses the physical layout
- the `release callback` manages the lifetime

Key implementation points:

- mapping of format strings
- holding `private_data`
- cascading release of children / dictionary
- distinguishing borrowed from owned

---

## 10. Implementation Advice

Recommended implementation order:

1. aligned buffer and bitmap utilities
2. `DataType / Field / Schema`
3. `ArrayData`
4. primitive / string / list / struct
5. record batch
6. IPC stream
7. IPC file
8. dictionary / union / REE / views / extension
9. FFI

The most common pitfalls:

- incomplete handling of `offset`
- treating null as part of the values bytes
- insufficiently strict validation of varlen offsets
- a file reader that supports only sequential reads and not random reads
- confused release semantics in FFI

---

## 11. Final Conclusion

The core of Arrow is not an object model in any one language, but the cross-language shared convention for physical layout.

If you only remember one thing, let it be this:

Get `ArrayData`, the buffer rules, the offset rules and layout validation right first, and only then implement the higher-level APIs, IPC, FFI and optimizations.
