const std = @import("std");
const reader = @import("reader.zig");

pub const Error = error{
    InvalidMetadata,
};

pub const MessageEnvelope = struct {
    version: i16,
    header_type: u8,
    body_length: i64,
};

/// Fast structural sanity check for any FlatBuffer table blob.
pub fn isSaneTable(buf: []const u8) bool {
    _ = reader.rootTable(buf) catch return false;
    return true;
}

/// Parse + validate Arrow IPC Message envelope fields without full unpack.
///
/// Message.fbs field ids:
/// 0 = version (enum MetadataVersion: short)
/// 1 = header_type (union discriminator)
/// 3 = bodyLength (long)
pub fn parseArrowMessageEnvelope(buf: []const u8) Error!MessageEnvelope {
    const table = reader.rootTable(buf) catch return error.InvalidMetadata;

    const version = table.readFieldI16(0) orelse 0;
    if (version < 0 or version > 4) return error.InvalidMetadata;

    const header_type = table.readFieldU8(1) orelse return error.InvalidMetadata;
    if (header_type == 0) return error.InvalidMetadata;

    const body_length = table.readFieldI64(3) orelse 0;

    return .{
        .version = version,
        .header_type = header_type,
        .body_length = body_length,
    };
}

// ---------------------------------------------------------------------------------------------
// Full verifier for Message and Footer flatbuffers.
//
// The generated readers in src/ipc_schema (and fbs_runtime's Table) trust every offset, length
// and enum value they read: a bad offset slices out of bounds (a panic in safe builds, an
// out-of-bounds read in ReleaseFast) and an unknown enum value panics. `unpackMessage` /
// `unpackFooter` therefore run these checks first, like Arrow C++ runs flatbuffers' Verifier:
// every table, vtable, field, string, vector and union reachable by the unpackers must lie inside
// the buffer, every enum / union tag must be a value the generated enum declares, nesting depth
// is capped and the number of tables visited is bounded by the buffer size (shared subtables
// cannot make the unpack exponential).
//
// When the schema is regenerated with new fields, the comptime field counts below stop compiling:
// add the new fields here before the unpackers can read them unchecked.

const fbs = struct {
    const s = @import("ipc_schema");
    const Message = s.org_apache_arrow_flatbuf_Message;
    const MessageHeader = s.org_apache_arrow_flatbuf_MessageHeader.MessageHeader;
    const MetadataVersion = s.org_apache_arrow_flatbuf_MetadataVersion.MetadataVersion;
    const Type = s.org_apache_arrow_flatbuf_Type.Type;
    const SparseTensorIndex = s.org_apache_arrow_flatbuf_SparseTensorIndex.SparseTensorIndex;
    const Endianness = s.org_apache_arrow_flatbuf_Endianness.Endianness;
    const Precision = s.org_apache_arrow_flatbuf_Precision.Precision;
    const DateUnit = s.org_apache_arrow_flatbuf_DateUnit.DateUnit;
    const TimeUnit = s.org_apache_arrow_flatbuf_TimeUnit.TimeUnit;
    const IntervalUnit = s.org_apache_arrow_flatbuf_IntervalUnit.IntervalUnit;
    const UnionMode = s.org_apache_arrow_flatbuf_UnionMode.UnionMode;
    const DictionaryKind = s.org_apache_arrow_flatbuf_DictionaryKind.DictionaryKind;
    const CompressionType = s.org_apache_arrow_flatbuf_CompressionType.CompressionType;
    const BodyCompressionMethod = s.org_apache_arrow_flatbuf_BodyCompressionMethod.BodyCompressionMethod;
    const SparseMatrixCompressedAxis = s.org_apache_arrow_flatbuf_SparseMatrixCompressedAxis.SparseMatrixCompressedAxis;
};

comptime {
    // Field count of every table the verifier walks (union type+value pairs count once).
    const counts = .{
        .{ fbs.s.org_apache_arrow_flatbuf_Message.MessageT, 4 },
        .{ fbs.s.org_apache_arrow_flatbuf_Schema.SchemaT, 4 },
        .{ fbs.s.org_apache_arrow_flatbuf_Field.FieldT, 6 },
        .{ fbs.s.org_apache_arrow_flatbuf_KeyValue.KeyValueT, 2 },
        .{ fbs.s.org_apache_arrow_flatbuf_DictionaryEncoding.DictionaryEncodingT, 4 },
        .{ fbs.s.org_apache_arrow_flatbuf_RecordBatch.RecordBatchT, 5 },
        .{ fbs.s.org_apache_arrow_flatbuf_BodyCompression.BodyCompressionT, 2 },
        .{ fbs.s.org_apache_arrow_flatbuf_DictionaryBatch.DictionaryBatchT, 3 },
        .{ fbs.s.org_apache_arrow_flatbuf_Tensor.TensorT, 4 },
        .{ fbs.s.org_apache_arrow_flatbuf_TensorDim.TensorDimT, 2 },
        .{ fbs.s.org_apache_arrow_flatbuf_SparseTensor.SparseTensorT, 5 },
        .{ fbs.s.org_apache_arrow_flatbuf_SparseTensorIndexCOO.SparseTensorIndexCOOT, 4 },
        .{ fbs.s.org_apache_arrow_flatbuf_SparseMatrixIndexCSX.SparseMatrixIndexCSXT, 5 },
        .{ fbs.s.org_apache_arrow_flatbuf_SparseTensorIndexCSF.SparseTensorIndexCSFT, 5 },
        .{ fbs.s.org_apache_arrow_flatbuf_Footer.FooterT, 5 },
        .{ fbs.s.org_apache_arrow_flatbuf_Int.IntT, 2 },
        .{ fbs.s.org_apache_arrow_flatbuf_Decimal.DecimalT, 3 },
        .{ fbs.s.org_apache_arrow_flatbuf_Time.TimeT, 2 },
        .{ fbs.s.org_apache_arrow_flatbuf_Timestamp.TimestampT, 2 },
        .{ fbs.s.org_apache_arrow_flatbuf_Union.UnionT, 2 },
    };
    for (counts) |c| {
        if (std.meta.fieldNames(c[0]).len != c[1]) @compileError("fbs_lite/verify.zig does not cover every field of " ++ @typeName(c[0]));
    }
    if (std.meta.fieldNames(fbs.Type).len != 27) @compileError("verifyType does not cover every Type union member");
}

const max_depth = 128;

const Verifier = struct {
    buf: []const u8,
    depth: u32 = 0,
    tables_left: usize,

    fn need(v: *const Verifier, pos: usize, len: usize) Error!void {
        const end = std.math.add(usize, pos, len) catch return error.InvalidMetadata;
        if (end > v.buf.len) return error.InvalidMetadata;
    }

    fn int(v: *const Verifier, comptime T: type, pos: usize) Error!T {
        try v.need(pos, @sizeOf(T));
        return std.mem.readInt(T, v.buf[pos..][0..@sizeOf(T)], .little);
    }

    const Tab = struct { pos: usize, vtable: usize, vtable_len: usize };

    fn table(v: *Verifier, pos: usize) Error!Tab {
        if (v.tables_left == 0) return error.InvalidMetadata;
        v.tables_left -= 1;
        const vtable_i = @as(i64, @intCast(pos)) - try v.int(i32, pos);
        const vtable = std.math.cast(usize, vtable_i) orelse return error.InvalidMetadata;
        const vtable_len = try v.int(u16, vtable);
        if (vtable_len < 4 or vtable_len % 2 != 0) return error.InvalidMetadata;
        try v.need(vtable, vtable_len);
        return .{ .pos = pos, .vtable = vtable, .vtable_len = vtable_len };
    }

    /// Absolute position of field `id` (null when absent); its `size` bytes must be in bounds.
    fn field(v: *Verifier, t: Tab, id: usize, size: usize) Error!?usize {
        const slot = 4 + 2 * id;
        if (slot >= t.vtable_len) return null;
        const rel = try v.int(u16, t.vtable + slot);
        if (rel == 0) return null;
        try v.need(t.pos + rel, size);
        return t.pos + rel;
    }

    fn scalar(v: *Verifier, t: Tab, id: usize, comptime T: type) Error!void {
        _ = try v.field(t, id, @sizeOf(T));
    }

    /// Enum (or union tag) field: must decode to a declared value of `E`, as fbs_runtime reads it.
    fn enumField(v: *Verifier, t: Tab, id: usize, comptime E: type) Error!?E {
        const tag_int = @typeInfo(@typeInfo(E).@"enum".tag_type).int;
        const I = @Int(tag_int.signedness, std.math.ceilPowerOfTwo(u16, @max(tag_int.bits, 8)) catch unreachable);
        const pos = try v.field(t, id, @sizeOf(I)) orelse return null;
        return std.enums.fromInt(E, try v.int(I, pos)) orelse error.InvalidMetadata;
    }

    /// Target of the uoffset stored in field `id` (null when absent).
    fn indirect(v: *Verifier, t: Tab, id: usize) Error!?usize {
        const pos = try v.field(t, id, 4) orelse return null;
        return try v.follow(pos);
    }

    fn follow(v: *Verifier, pos: usize) Error!usize {
        const target = std.math.add(usize, pos, try v.int(u32, pos)) catch return error.InvalidMetadata;
        try v.need(target, 4);
        return target;
    }

    fn nested(v: *Verifier, pos: usize, comptime verifyFn: fn (*Verifier, Tab) Error!void) Error!void {
        if (v.depth >= max_depth) return error.InvalidMetadata;
        v.depth += 1;
        defer v.depth -= 1;
        try verifyFn(v, try v.table(pos));
    }

    fn subTable(v: *Verifier, t: Tab, id: usize, comptime verifyFn: fn (*Verifier, Tab) Error!void) Error!void {
        if (try v.indirect(t, id)) |pos| try v.nested(pos, verifyFn);
    }

    /// Vector of `elem_size`-byte elements (strings are byte vectors); returns start and length.
    fn vector(v: *Verifier, t: Tab, id: usize, elem_size: usize) Error!?struct { start: usize, len: usize } {
        const pos = try v.indirect(t, id) orelse return null;
        const len = try v.int(u32, pos);
        try v.need(pos + 4, std.math.mul(usize, len, elem_size) catch return error.InvalidMetadata);
        return .{ .start = pos + 4, .len = len };
    }

    fn tableVector(v: *Verifier, t: Tab, id: usize, comptime verifyFn: fn (*Verifier, Tab) Error!void) Error!void {
        const vec = try v.vector(t, id, 4) orelse return;
        for (0..vec.len) |i| try v.nested(try v.follow(vec.start + 4 * i), verifyFn);
    }

    fn unionValue(v: *Verifier, t: Tab, type_id: usize, comptime U: type, comptime verifyFn: fn (*Verifier, std.meta.Tag(U), Tab) Error!void) Error!void {
        // The unpackers follow the value offset whenever it is present, even for a NONE tag.
        const pos = try v.indirect(t, type_id + 1) orelse return;
        const tag = try v.enumField(t, type_id, std.meta.Tag(U)) orelse return;
        if (v.depth >= max_depth) return error.InvalidMetadata;
        v.depth += 1;
        defer v.depth -= 1;
        try verifyFn(v, tag, try v.table(pos));
    }
};

fn verifyKeyValue(v: *Verifier, t: Verifier.Tab) Error!void {
    _ = try v.vector(t, 0, 1);
    _ = try v.vector(t, 1, 1);
}

fn verifyInt(v: *Verifier, t: Verifier.Tab) Error!void {
    try v.scalar(t, 0, i32);
    try v.scalar(t, 1, bool);
}

fn verifyType(v: *Verifier, tag: std.meta.Tag(fbs.Type), t: Verifier.Tab) Error!void {
    switch (tag) {
        .NONE, .Null, .Binary, .Utf8, .Bool, .List, .Struct_, .LargeBinary, .LargeUtf8, .LargeList, .RunEndEncoded, .BinaryView, .Utf8View, .ListView, .LargeListView => {},
        .Int => try verifyInt(v, t),
        .FloatingPoint => _ = try v.enumField(t, 0, fbs.Precision),
        .Decimal => for (0..3) |id| try v.scalar(t, id, i32),
        .Date => _ = try v.enumField(t, 0, fbs.DateUnit),
        .Time => {
            _ = try v.enumField(t, 0, fbs.TimeUnit);
            try v.scalar(t, 1, i32);
        },
        .Timestamp => {
            _ = try v.enumField(t, 0, fbs.TimeUnit);
            _ = try v.vector(t, 1, 1);
        },
        .Interval => _ = try v.enumField(t, 0, fbs.IntervalUnit),
        .Union => {
            _ = try v.enumField(t, 0, fbs.UnionMode);
            _ = try v.vector(t, 1, 4);
        },
        .FixedSizeBinary, .FixedSizeList => try v.scalar(t, 0, i32),
        .Map => try v.scalar(t, 0, bool),
        .Duration => _ = try v.enumField(t, 0, fbs.TimeUnit),
    }
}

fn verifyDictionaryEncoding(v: *Verifier, t: Verifier.Tab) Error!void {
    try v.scalar(t, 0, i64);
    try v.subTable(t, 1, verifyInt);
    try v.scalar(t, 2, bool);
    _ = try v.enumField(t, 3, fbs.DictionaryKind);
}

fn verifyField(v: *Verifier, t: Verifier.Tab) Error!void {
    _ = try v.vector(t, 0, 1);
    try v.scalar(t, 1, bool);
    try v.unionValue(t, 2, fbs.Type, verifyType);
    try v.subTable(t, 4, verifyDictionaryEncoding);
    try v.tableVector(t, 5, verifyField);
    try v.tableVector(t, 6, verifyKeyValue);
}

fn verifySchema(v: *Verifier, t: Verifier.Tab) Error!void {
    _ = try v.enumField(t, 0, fbs.Endianness);
    try v.tableVector(t, 1, verifyField);
    try v.tableVector(t, 2, verifyKeyValue);
    _ = try v.vector(t, 3, 8);
}

fn verifyBodyCompression(v: *Verifier, t: Verifier.Tab) Error!void {
    _ = try v.enumField(t, 0, fbs.CompressionType);
    _ = try v.enumField(t, 1, fbs.BodyCompressionMethod);
}

fn verifyRecordBatch(v: *Verifier, t: Verifier.Tab) Error!void {
    try v.scalar(t, 0, i64);
    _ = try v.vector(t, 1, 16); // FieldNode
    _ = try v.vector(t, 2, 16); // Buffer
    try v.subTable(t, 3, verifyBodyCompression);
    _ = try v.vector(t, 4, 8);
}

fn verifyDictionaryBatch(v: *Verifier, t: Verifier.Tab) Error!void {
    try v.scalar(t, 0, i64);
    try v.subTable(t, 1, verifyRecordBatch);
    try v.scalar(t, 2, bool);
}

fn verifyTensorDim(v: *Verifier, t: Verifier.Tab) Error!void {
    try v.scalar(t, 0, i64);
    _ = try v.vector(t, 1, 1);
}

fn verifyTensor(v: *Verifier, t: Verifier.Tab) Error!void {
    try v.unionValue(t, 0, fbs.Type, verifyType);
    try v.tableVector(t, 2, verifyTensorDim);
    _ = try v.vector(t, 3, 8);
    _ = try v.field(t, 4, 16); // inline Buffer
}

fn verifySparseIndex(v: *Verifier, tag: std.meta.Tag(fbs.SparseTensorIndex), t: Verifier.Tab) Error!void {
    switch (tag) {
        .NONE => {},
        .SparseTensorIndexCOO => {
            try v.subTable(t, 0, verifyInt);
            _ = try v.vector(t, 1, 8);
            _ = try v.field(t, 2, 16);
            try v.scalar(t, 3, bool);
        },
        .SparseMatrixIndexCSX => {
            _ = try v.enumField(t, 0, fbs.SparseMatrixCompressedAxis);
            try v.subTable(t, 1, verifyInt);
            _ = try v.field(t, 2, 16);
            try v.subTable(t, 3, verifyInt);
            _ = try v.field(t, 4, 16);
        },
        .SparseTensorIndexCSF => {
            try v.subTable(t, 0, verifyInt);
            _ = try v.vector(t, 1, 16);
            try v.subTable(t, 2, verifyInt);
            _ = try v.vector(t, 3, 16);
            _ = try v.vector(t, 4, 4);
        },
    }
}

fn verifySparseTensor(v: *Verifier, t: Verifier.Tab) Error!void {
    try v.unionValue(t, 0, fbs.Type, verifyType);
    try v.tableVector(t, 2, verifyTensorDim);
    try v.scalar(t, 3, i64);
    try v.unionValue(t, 4, fbs.SparseTensorIndex, verifySparseIndex);
    _ = try v.field(t, 6, 16);
}

fn verifyHeader(v: *Verifier, tag: std.meta.Tag(fbs.MessageHeader), t: Verifier.Tab) Error!void {
    switch (tag) {
        .NONE => {},
        .Schema => try verifySchema(v, t),
        .DictionaryBatch => try verifyDictionaryBatch(v, t),
        .RecordBatch => try verifyRecordBatch(v, t),
        .Tensor => try verifyTensor(v, t),
        .SparseTensor => try verifySparseTensor(v, t),
    }
}

fn verifyMessageTable(v: *Verifier, t: Verifier.Tab) Error!void {
    _ = try v.enumField(t, 0, fbs.MetadataVersion);
    try v.unionValue(t, 1, fbs.MessageHeader, verifyHeader);
    try v.scalar(t, 3, i64);
    try v.tableVector(t, 4, verifyKeyValue);
}

fn verifyFooterTable(v: *Verifier, t: Verifier.Tab) Error!void {
    _ = try v.enumField(t, 0, fbs.MetadataVersion);
    try v.subTable(t, 1, verifySchema);
    _ = try v.vector(t, 2, 24); // Block
    _ = try v.vector(t, 3, 24);
    try v.tableVector(t, 4, verifyKeyValue);
}

fn verifyRoot(buf: []const u8, comptime verifyFn: fn (*Verifier, Verifier.Tab) Error!void) Error!void {
    // Offsets are u32 and fbs_runtime does i32 arithmetic on table positions.
    if (buf.len > std.math.maxInt(i32)) return error.InvalidMetadata;
    var v: Verifier = .{ .buf = buf, .tables_left = 8 * buf.len };
    try v.nested(try v.follow(0), verifyFn);
}

/// Verifies a Message flatbuffer before `fbs_lite/builder.unpackMessage` reads it.
pub fn verifyMessage(buf: []const u8) Error!void {
    return verifyRoot(buf, verifyMessageTable);
}

/// Verifies a Footer flatbuffer before `fbs_lite/builder.unpackFooter` reads it.
pub fn verifyFooter(buf: []const u8) Error!void {
    return verifyRoot(buf, verifyFooterTable);
}

test "verify rejects malformed table bytes" {
    const malformed = [_]u8{
        0x10, 0x00, 0x00, 0x00,
        0xff, 0xff, 0xff, 0xff,
    };
    try std.testing.expect(!isSaneTable(malformed[0..]));
    try std.testing.expectError(error.InvalidMetadata, parseArrowMessageEnvelope(malformed[0..]));
}
