//! Integration JSON writer (the inverse of `json.read`), following Arrow C++'s SchemaWriter and
//! ArrayWriter in `cpp/src/arrow/integration/json_internal.cc`: member order and names as Arrow
//! C++ writes them. See docs/integration-suite.md.
const std = @import("std");
const json = @import("json.zig");
const compare = @import("compare.zig");
const datatype = @import("../datatype.zig");
const array_data = @import("../array/array_data.zig");
const array_mod = @import("../array/array.zig");
const bitmap = @import("../bitmap.zig");
const concat = @import("../concat_array_refs.zig");

const Allocator = std.mem.Allocator;
const Stringify = std.json.Stringify;
const DataType = datatype.DataType;
const Field = datatype.Field;
const KeyValue = datatype.KeyValue;
const IntType = datatype.IntType;
const Schema = json.Schema;
const ArrayData = array_data.ArrayData;
const ArrayRef = json.ArrayRef;
const RecordBatch = json.RecordBatch;
const storageDataType = datatype.storageDataType;
const max_depth = json.max_depth;

pub const WriteError = error{
    NestingTooDeep,
    /// A name, metadata entry or string value that is not UTF-8 (JSON strings are).
    InvalidUtf8,
    UnsupportedType,
    /// A dictionary-encoded type without an id, or a column without its dictionary.
    MissingDictionary,
    /// A dictionary that a later batch replaces instead of extending: one document holds one
    /// dictionary per id.
    InconsistentDictionary,
    /// An array longer than `json.max_count`.
    ArrayTooLong,
    OutOfMemory,
} || std.Io.Writer.Error || array_data.AccessorError;

/// Writes `sc` and `batches` (columns carrying their dictionaries) as one document, members in
/// Arrow C++'s order. Each dictionary is written once, as the last batch has it; every earlier
/// batch's dictionary must be a prefix of it (deltas), otherwise `InconsistentDictionary`.
pub fn write(allocator: Allocator, out: *std.Io.Writer, sc: Schema, batches: []const RecordBatch) WriteError!void {
    var w: JsonWriter = .{ .gpa = allocator, .s = .{ .writer = out, .options = .{ .whitespace = .indent_1 } } };
    defer w.deinit();

    // Columns with a slice offset somewhere are written from a compacted copy.
    for (batches) |*batch| {
        if (batch.columns.len != sc.fields.len) return error.InvalidChildren;
        try w.columns.ensureUnusedCapacity(allocator, batch.columns.len);
        for (batch.columns) |col| w.columns.appendAssumeCapacity(try compacted(allocator, col));
    }
    for (w.columns.items) |col| try w.collectDictionaries(col.data(), 0);

    try w.s.beginObject();
    try w.s.objectField("schema");
    try w.schema(sc);
    if (w.dict_ids.items.len > 0) {
        try w.s.objectField("dictionaries");
        try w.s.beginArray();
        for (w.dict_ids.items) |id| {
            const dict = w.dicts.get(id).?;
            try w.s.beginObject();
            try w.s.objectField("id");
            try w.s.write(id);
            try w.s.objectField("data");
            try w.s.beginObject();
            try w.s.objectField("count");
            try w.s.write(try countOf(dict.data().length));
            try w.s.objectField("columns");
            try w.s.beginArray();
            try w.column("dictionary", dict.data(), 0);
            try w.s.endArray();
            try w.s.endObject();
            try w.s.endObject();
        }
        try w.s.endArray();
    }
    try w.s.objectField("batches");
    try w.s.beginArray();
    var next: usize = 0;
    for (batches) |*batch| {
        try w.s.beginObject();
        try w.s.objectField("count");
        try w.s.write(try countOf(batch.numRows()));
        try w.s.objectField("columns");
        try w.s.beginArray();
        for (sc.fields) |f| {
            try w.column(f.name, w.columns.items[next].data(), 0);
            next += 1;
        }
        try w.s.endArray();
        try w.s.endObject();
    }
    try w.s.endArray();
    try w.s.endObject();
    try out.writeByte('\n');
}

/// `col`, or a compacted copy when it or a child has a slice offset.
fn compacted(allocator: Allocator, col: ArrayRef) WriteError!ArrayRef {
    if (!hasSliceOffset(col.data())) return col.retain();
    const parts = [_]ArrayRef{col};
    return concat.concatArrayRefs(allocator, col.data().data_type, &parts) catch |e| switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        else => error.UnsupportedType,
    };
}

fn hasSliceOffset(data: *const ArrayData) bool {
    if (data.offset != 0) return true;
    for (data.children) |c| if (hasSliceOffset(c.data())) return true;
    return false;
}

fn countOf(n: usize) WriteError!i64 {
    if (n > json.max_count) return error.ArrayTooLong;
    return @intCast(n);
}

fn intDataType(t: IntType) DataType {
    return switch (t.bit_width) {
        8 => if (t.signed) .{ .int8 = {} } else .{ .uint8 = {} },
        16 => if (t.signed) .{ .int16 = {} } else .{ .uint16 = {} },
        32 => if (t.signed) .{ .int32 = {} } else .{ .uint32 = {} },
        else => if (t.signed) .{ .int64 = {} } else .{ .uint64 = {} },
    };
}

const JsonWriter = struct {
    gpa: Allocator,
    s: Stringify,
    columns: std.ArrayList(ArrayRef) = .empty,
    /// Dictionary ids in writing order (inner dictionaries first) and their dictionaries.
    dict_ids: std.ArrayList(i64) = .empty,
    dicts: std.AutoHashMapUnmanaged(i64, ArrayRef) = .empty,

    fn deinit(w: *JsonWriter) void {
        for (w.columns.items) |*c| c.release();
        w.columns.deinit(w.gpa);
        w.dict_ids.deinit(w.gpa);
        var it = w.dicts.valueIterator();
        while (it.next()) |d| d.release();
        w.dicts.deinit(w.gpa);
    }

    fn collectDictionaries(w: *JsonWriter, data: *const ArrayData, depth: usize) WriteError!void {
        if (depth >= max_depth) return error.NestingTooDeep;
        const st = storageDataType(data.data_type);
        if (st == .dictionary) {
            const id = st.dictionary.id orelse return error.MissingDictionary;
            const dict = data.dictionary orelse return error.MissingDictionary;
            if (w.dicts.getPtr(id)) |prev| {
                if (prev.node != dict.node) {
                    const grows = prev.data().length <= dict.data().length;
                    const shorter = if (grows) prev.* else dict;
                    const longer = if (grows) dict else prev.*;
                    if (!compare.arrayIsPrefix(shorter.data(), longer.data(), null)) return error.InconsistentDictionary;
                    if (grows) {
                        const next = try compacted(w.gpa, dict);
                        prev.release();
                        prev.* = next;
                    }
                }
            } else {
                var copy = try compacted(w.gpa, dict);
                errdefer copy.release();
                // Nested dictionaries first, so that a reader meets them before their users.
                try w.collectDictionaries(copy.data(), depth + 1);
                try w.dict_ids.ensureUnusedCapacity(w.gpa, 1);
                try w.dicts.put(w.gpa, id, copy);
                w.dict_ids.appendAssumeCapacity(id);
            }
        }
        for (data.children) |c| try w.collectDictionaries(c.data(), depth + 1);
    }

    fn str(w: *JsonWriter, v: []const u8) WriteError!void {
        if (!std.unicode.utf8ValidateSlice(v)) return error.InvalidUtf8;
        try w.s.write(v);
    }

    fn hex(w: *JsonWriter, bytes: []const u8) WriteError!void {
        const digits = "0123456789ABCDEF";
        try w.s.beginWriteRaw();
        try w.s.writer.writeByte('"');
        for (bytes) |b| {
            try w.s.writer.writeByte(digits[b >> 4]);
            try w.s.writer.writeByte(digits[b & 15]);
        }
        try w.s.writer.writeByte('"');
        w.s.endWriteRaw();
    }

    fn quotedInt(w: *JsonWriter, v: anytype) WriteError!void {
        try w.s.print("\"{d}\"", .{v});
    }

    /// A float as Arrow C++ writes it (the double value, shortest round-trip form); NaN and
    /// infinities, which JSON numbers cannot hold, as the strings "NaN", "Infinity", "-Infinity".
    fn float(w: *JsonWriter, v: f64) WriteError!void {
        if (std.math.isNan(v)) return w.s.write("NaN");
        if (std.math.isInf(v)) return w.s.write(if (v > 0) "Infinity" else "-Infinity");
        var buf: [64]u8 = undefined;
        const mag = @abs(v);
        const text = (if (mag >= 1e16 or (mag != 0 and mag < 1e-6))
            std.fmt.bufPrint(&buf, "{e}", .{v})
        else
            std.fmt.bufPrint(&buf, "{d}", .{v})) catch unreachable; // a double fits in 64 bytes
        // A whole number keeps a ".0", as Arrow C++ (rapidjson) writes it, so it reads back as a
        // float; this also keeps the sign of -0.0.
        if (std.mem.indexOfAny(u8, text, ".e") == null) return w.s.print("{s}.0", .{text});
        try w.s.print("{s}", .{text});
    }

    fn schema(w: *JsonWriter, sc: Schema) WriteError!void {
        try w.s.beginObject();
        try w.s.objectField("fields");
        try w.s.beginArray();
        for (sc.fields) |f| try w.field(f, 0);
        try w.s.endArray();
        try w.metadata(sc.metadata, null);
        try w.s.endObject();
    }

    fn metadata(w: *JsonWriter, kvs: ?[]const KeyValue, ext: ?datatype.ExtensionType) WriteError!void {
        const user = kvs orelse &.{};
        if (user.len == 0 and ext == null) return;
        try w.s.objectField("metadata");
        try w.s.beginArray();
        for (user) |kv| try w.keyValue(kv.key, kv.value);
        if (ext) |e| {
            try w.keyValue(json.extension_name_key, e.name);
            if (e.metadata) |m| try w.keyValue(json.extension_metadata_key, m);
        }
        try w.s.endArray();
    }

    fn keyValue(w: *JsonWriter, key: []const u8, value: []const u8) WriteError!void {
        try w.s.beginObject();
        try w.s.objectField("key");
        try w.str(key);
        try w.s.objectField("value");
        try w.str(value);
        try w.s.endObject();
    }

    fn field(w: *JsonWriter, f: Field, depth: usize) WriteError!void {
        if (depth >= max_depth) return error.NestingTooDeep;
        // Extension and dictionary wrappers go to the metadata and "dictionary"; "type" and
        // "children" describe the storage (value) type.
        var dt = f.data_type.*;
        var ext: ?datatype.ExtensionType = null;
        if (dt == .extension) {
            ext = dt.extension;
            dt = dt.extension.storage_type.*;
        }
        var dict: ?datatype.DictionaryType = null;
        if (dt == .dictionary) {
            dict = dt.dictionary;
            dt = dt.dictionary.value_type.*;
            if (dt == .extension) {
                if (ext != null) return error.UnsupportedType;
                ext = dt.extension;
                dt = dt.extension.storage_type.*;
            }
        }
        if (dt == .extension or dt == .dictionary) return error.UnsupportedType;

        try w.s.beginObject();
        try w.s.objectField("name");
        try w.str(f.name);
        try w.s.objectField("nullable");
        try w.s.write(f.nullable);
        try w.s.objectField("type");
        try w.typeObject(dt);
        try w.s.objectField("children");
        try w.s.beginArray();
        switch (dt) {
            .list, .large_list => |l| try w.field(l.value_field, depth + 1),
            .fixed_size_list => |l| try w.field(l.value_field, depth + 1),
            .list_view, .large_list_view => |l| try w.field(l.value_field, depth + 1),
            .struct_ => |st| for (st.fields) |c| try w.field(c, depth + 1),
            .sparse_union, .dense_union => |u| for (u.fields) |c| try w.field(c, depth + 1),
            .map => |m| {
                const kv = [2]Field{ m.key_field, m.item_field };
                const entries_dt: DataType = .{ .struct_ = .{ .fields = &kv } };
                try w.field(.{ .name = m.entries_name, .nullable = false, .data_type = m.entries_type orelse &entries_dt }, depth + 1);
            },
            .run_end_encoded => |ree| {
                const ends_dt = intDataType(ree.run_end_type);
                try w.field(.{ .name = "run_ends", .nullable = false, .data_type = &ends_dt }, depth + 1);
                try w.field(.{ .name = "values", .nullable = true, .data_type = ree.value_type }, depth + 1);
            },
            else => {},
        }
        try w.s.endArray();
        if (dict) |d| {
            try w.s.objectField("dictionary");
            try w.s.beginObject();
            try w.s.objectField("id");
            try w.s.write(d.id orelse return error.MissingDictionary);
            try w.s.objectField("indexType");
            try w.intType(d.index_type);
            try w.s.objectField("isOrdered");
            try w.s.write(d.ordered);
            try w.s.endObject();
        }
        try w.metadata(f.metadata, ext);
        try w.s.endObject();
    }

    fn intType(w: *JsonWriter, t: IntType) WriteError!void {
        try w.s.beginObject();
        try w.named("int");
        try w.s.objectField("bitWidth");
        try w.s.write(t.bit_width);
        try w.s.objectField("isSigned");
        try w.s.write(t.signed);
        try w.s.endObject();
    }

    fn named(w: *JsonWriter, name: []const u8) WriteError!void {
        try w.s.objectField("name");
        try w.s.write(name);
    }

    fn unit(w: *JsonWriter, u: datatype.TimeUnit) WriteError!void {
        try w.s.objectField("unit");
        try w.s.write(switch (u) {
            .second => "SECOND",
            .millisecond => "MILLISECOND",
            .microsecond => "MICROSECOND",
            .nanosecond => "NANOSECOND",
        });
    }

    fn typeObject(w: *JsonWriter, dt: DataType) WriteError!void {
        if (IntType.fromTypeId(dt.id())) |t| return w.intType(t);
        try w.s.beginObject();
        switch (dt) {
            .null => try w.named("null"),
            .bool => try w.named("bool"),
            .string => try w.named("utf8"),
            .binary => try w.named("binary"),
            .large_string => try w.named("largeutf8"),
            .large_binary => try w.named("largebinary"),
            .string_view => try w.named("utf8view"),
            .binary_view => try w.named("binaryview"),
            .half_float, .float, .double => {
                try w.named("floatingpoint");
                try w.s.objectField("precision");
                try w.s.write(switch (dt) {
                    .half_float => "HALF",
                    .float => "SINGLE",
                    else => "DOUBLE",
                });
            },
            .fixed_size_binary => |fsb| {
                try w.named("fixedsizebinary");
                try w.s.objectField("byteWidth");
                try w.s.write(fsb.byte_width);
            },
            .decimal32, .decimal64, .decimal128, .decimal256 => |d| {
                try w.named("decimal");
                try w.s.objectField("precision");
                try w.s.write(d.precision);
                try w.s.objectField("scale");
                try w.s.write(d.scale);
                try w.s.objectField("bitWidth");
                try w.s.write(@as(u16, switch (dt) {
                    .decimal32 => 32,
                    .decimal64 => 64,
                    .decimal128 => 128,
                    else => 256,
                }));
            },
            .date32, .date64 => {
                try w.named("date");
                try w.s.objectField("unit");
                try w.s.write(if (dt == .date32) "DAY" else "MILLISECOND");
            },
            .time32, .time64 => |t| {
                try w.named("time");
                try w.unit(t.unit);
                try w.s.objectField("bitWidth");
                try w.s.write(@as(u8, if (dt == .time32) 32 else 64));
            },
            .timestamp => |ts| {
                try w.named("timestamp");
                try w.unit(ts.unit);
                if (ts.timezone) |tz| if (tz.len > 0) {
                    try w.s.objectField("timezone");
                    try w.str(tz);
                };
            },
            .duration => |d| {
                try w.named("duration");
                try w.unit(d.unit);
            },
            .interval_months, .interval_day_time, .interval_month_day_nano => {
                try w.named("interval");
                try w.s.objectField("unit");
                try w.s.write(switch (dt) {
                    .interval_months => "YEAR_MONTH",
                    .interval_day_time => "DAY_TIME",
                    else => "MONTH_DAY_NANO",
                });
            },
            .list => try w.named("list"),
            .large_list => try w.named("largelist"),
            .list_view => try w.named("listview"),
            .large_list_view => try w.named("largelistview"),
            .fixed_size_list => |l| {
                try w.named("fixedsizelist");
                try w.s.objectField("listSize");
                try w.s.write(l.list_size);
            },
            .struct_ => try w.named("struct"),
            .map => |m| {
                try w.named("map");
                try w.s.objectField("keysSorted");
                try w.s.write(m.keys_sorted);
            },
            .sparse_union, .dense_union => |u| {
                try w.named("union");
                try w.s.objectField("mode");
                try w.s.write(if (dt == .sparse_union) "SPARSE" else "DENSE");
                try w.s.objectField("typeIds");
                try w.s.beginArray();
                for (u.type_ids) |id| try w.s.write(id);
                try w.s.endArray();
            },
            .run_end_encoded => try w.named("runendencoded"),
            else => return error.UnsupportedType,
        }
        try w.s.endObject();
    }

    /// One `FieldData` object (Arrow C++'s ArrayWriter).
    fn column(w: *JsonWriter, name: []const u8, data: *const ArrayData, depth: usize) WriteError!void {
        if (depth >= max_depth) return error.NestingTooDeep;
        const st = storageDataType(data.data_type);
        try w.s.beginObject();
        try w.s.objectField("name");
        try w.str(name);
        try w.s.objectField("count");
        try w.s.write(try countOf(data.length));
        switch (st) {
            .null, .sparse_union, .dense_union, .run_end_encoded => {},
            else => {
                try w.s.objectField("VALIDITY");
                try w.s.beginArray();
                for (0..data.length) |i| try w.s.write(@as(u8, if (data.isNull(i)) 0 else 1));
                try w.s.endArray();
            },
        }
        switch (st) {
            .null => {},
            .bool => {
                const bits = (try data.bufferAt(1)).data;
                try w.s.objectField("DATA");
                try w.s.beginArray();
                for (0..data.length) |i| {
                    const p = data.offset + i;
                    if (p / 8 >= bits.len) return error.BufferTooSmall;
                    try w.s.write(!data.isNull(i) and bitmap.bitIsSet(bits, p));
                }
                try w.s.endArray();
            },
            .int8 => try w.ints(i8, data, false),
            .uint8 => try w.ints(u8, data, false),
            .int16 => try w.ints(i16, data, false),
            // float16 as its bit pattern
            .uint16, .half_float => try w.ints(u16, data, false),
            .int32, .date32, .time32, .interval_months => try w.ints(i32, data, false),
            .uint32 => try w.ints(u32, data, false),
            .int64, .date64, .time64, .timestamp, .duration => try w.ints(i64, data, true),
            .uint64 => try w.ints(u64, data, true),
            .decimal32 => try w.ints(i32, data, true),
            .decimal64 => try w.ints(i64, data, true),
            .decimal128 => try w.ints(i128, data, true),
            .decimal256 => try w.ints(i256, data, true),
            .float => try w.floats(f32, data),
            .double => try w.floats(f64, data),
            .interval_day_time => {
                const values = try (array_mod.PrimitiveArray(i64){ .data = data }).values();
                try w.s.objectField("DATA");
                try w.s.beginArray();
                for (values, 0..) |v, i| {
                    try w.s.beginObject();
                    if (!data.isNull(i)) {
                        const bits: u64 = @bitCast(v);
                        try w.s.objectField("days");
                        try w.s.write(@as(i32, @bitCast(@as(u32, @truncate(bits)))));
                        try w.s.objectField("milliseconds");
                        try w.s.write(@as(i32, @bitCast(@as(u32, @truncate(bits >> 32)))));
                    }
                    try w.s.endObject();
                }
                try w.s.endArray();
            },
            .interval_month_day_nano => {
                const values = try (array_mod.PrimitiveArray(i128){ .data = data }).values();
                try w.s.objectField("DATA");
                try w.s.beginArray();
                for (values, 0..) |v, i| {
                    try w.s.beginObject();
                    if (!data.isNull(i)) {
                        const bits: u128 = @bitCast(v);
                        try w.s.objectField("months");
                        try w.s.write(@as(i32, @bitCast(@as(u32, @truncate(bits)))));
                        try w.s.objectField("days");
                        try w.s.write(@as(i32, @bitCast(@as(u32, @truncate(bits >> 32)))));
                        try w.s.objectField("nanoseconds");
                        try w.s.write(@as(i64, @bitCast(@as(u64, @truncate(bits >> 64)))));
                    }
                    try w.s.endObject();
                }
                try w.s.endArray();
            },
            .fixed_size_binary => |fsb| {
                const width: usize = @intCast(fsb.byte_width);
                const bytes = (try data.bufferAt(1)).data;
                try w.s.objectField("DATA");
                try w.s.beginArray();
                for (0..data.length) |i| {
                    const start = std.math.mul(usize, data.offset + i, width) catch return error.BufferTooSmall;
                    if (start + width > bytes.len) return error.BufferTooSmall;
                    try w.hex(bytes[start..][0..width]);
                }
                try w.s.endArray();
            },
            .string, .binary => try w.varBinary(i32, data, st == .string),
            .large_string, .large_binary => try w.varBinary(i64, data, st == .large_string),
            .string_view, .binary_view => try w.views(data, st == .string_view),
            .list => |l| {
                try w.rawInts(i32, data, 1, "OFFSET", data.length + 1, false);
                try w.childColumns(data, &.{l.value_field.name}, depth);
            },
            .map => |m| {
                try w.rawInts(i32, data, 1, "OFFSET", data.length + 1, false);
                try w.childColumns(data, &.{m.entries_name}, depth);
            },
            .large_list => |l| {
                try w.rawInts(i64, data, 1, "OFFSET", data.length + 1, true);
                try w.childColumns(data, &.{l.value_field.name}, depth);
            },
            .list_view => |l| {
                try w.rawInts(i32, data, 1, "OFFSET", data.length, false);
                try w.rawInts(i32, data, 2, "SIZE", data.length, false);
                try w.childColumns(data, &.{l.value_field.name}, depth);
            },
            .large_list_view => |l| {
                try w.rawInts(i64, data, 1, "OFFSET", data.length, true);
                try w.rawInts(i64, data, 2, "SIZE", data.length, true);
                try w.childColumns(data, &.{l.value_field.name}, depth);
            },
            .fixed_size_list => |l| try w.childColumns(data, &.{l.value_field.name}, depth),
            .struct_ => |s| try w.fieldColumns(data, s.fields, depth),
            .sparse_union, .dense_union => |u| {
                try w.rawInts(i8, data, 0, "TYPE_ID", data.length, false);
                if (st == .dense_union) try w.rawInts(i32, data, 1, "OFFSET", data.length, false);
                try w.fieldColumns(data, u.fields, depth);
            },
            .run_end_encoded => try w.childColumns(data, &.{ "run_ends", "values" }, depth),
            .dictionary => |d| switch (d.index_type.bit_width) {
                8 => if (d.index_type.signed) try w.ints(i8, data, false) else try w.ints(u8, data, false),
                16 => if (d.index_type.signed) try w.ints(i16, data, false) else try w.ints(u16, data, false),
                32 => if (d.index_type.signed) try w.ints(i32, data, false) else try w.ints(u32, data, false),
                else => if (d.index_type.signed) try w.ints(i64, data, true) else try w.ints(u64, data, true),
            },
            .extension => unreachable, // storageDataType unwraps extensions
        }
        try w.s.endObject();
    }

    /// DATA of fixed-width integers; null rows as 0 (as Arrow C++).
    fn ints(w: *JsonWriter, comptime T: type, data: *const ArrayData, quoted: bool) WriteError!void {
        const values = try (array_mod.PrimitiveArray(T){ .data = data }).values();
        try w.s.objectField("DATA");
        try w.s.beginArray();
        for (values, 0..) |v, i| {
            const x: T = if (data.isNull(i)) 0 else v;
            if (quoted) try w.quotedInt(x) else try w.s.write(x);
        }
        try w.s.endArray();
    }

    fn floats(w: *JsonWriter, comptime T: type, data: *const ArrayData) WriteError!void {
        const values = try (array_mod.PrimitiveArray(T){ .data = data }).values();
        try w.s.objectField("DATA");
        try w.s.beginArray();
        for (values, 0..) |v, i| try w.float(if (data.isNull(i)) 0 else v);
        try w.s.endArray();
    }

    /// `n` integers of buffer `index` from the array offset on, as stored.
    fn rawInts(w: *JsonWriter, comptime T: type, data: *const ArrayData, index: usize, key: []const u8, n: usize, quoted: bool) WriteError!void {
        const all = try data.typedBufferAt(index, T);
        if (data.offset + n > all.len) return error.BufferTooSmall;
        try w.s.objectField(key);
        try w.s.beginArray();
        for (all[data.offset..][0..n]) |v| {
            if (quoted) try w.quotedInt(v) else try w.s.write(v);
        }
        try w.s.endArray();
    }

    fn varBinary(w: *JsonWriter, comptime O: type, data: *const ArrayData, utf8: bool) WriteError!void {
        try w.rawInts(O, data, 1, "OFFSET", data.length + 1, O == i64);
        try w.s.objectField("DATA");
        try w.s.beginArray();
        for (0..data.length) |i| {
            const v = if (O == i32) try (array_mod.BinaryArray{ .data = data }).value(i) else try (array_mod.LargeBinaryArray{ .data = data }).value(i);
            if (utf8) try w.str(if (data.isNull(i)) "" else v) else try w.hex(v);
        }
        try w.s.endArray();
    }

    fn views(w: *JsonWriter, data: *const ArrayData, utf8: bool) WriteError!void {
        const records = (try data.bufferAt(1)).data;
        try w.s.objectField("VIEWS");
        try w.s.beginArray();
        for (0..data.length) |i| {
            const start = std.math.mul(usize, data.offset + i, 16) catch return error.BufferTooSmall;
            if (start + 16 > records.len) return error.BufferTooSmall;
            const rec = records[start..][0..16];
            const size = std.mem.readInt(u32, rec[0..4], .little);
            try w.s.beginObject();
            try w.s.objectField("SIZE");
            if (data.isNull(i)) {
                // A null row's view is not data: written as the empty inline view.
                try w.s.write(0);
                try w.s.objectField("INLINED");
                try w.s.write("");
            } else if (size <= 12) {
                try w.s.write(size);
                try w.s.objectField("INLINED");
                if (utf8) try w.str(rec[4..][0..size]) else try w.hex(rec[4..][0..size]);
            } else {
                try w.s.write(size);
                try w.s.objectField("PREFIX_HEX");
                try w.hex(rec[4..8]);
                try w.s.objectField("BUFFER_INDEX");
                try w.s.write(std.mem.readInt(u32, rec[8..12], .little));
                try w.s.objectField("OFFSET");
                try w.s.write(std.mem.readInt(u32, rec[12..16], .little));
            }
            try w.s.endObject();
        }
        try w.s.endArray();
        try w.s.objectField("VARIADIC_DATA_BUFFERS");
        try w.s.beginArray();
        if (data.buffers.len > 2) for (data.buffers[2..]) |buf| try w.hex(buf.data);
        try w.s.endArray();
    }

    fn fieldColumns(w: *JsonWriter, data: *const ArrayData, fields: []const Field, depth: usize) WriteError!void {
        if (data.children.len != fields.len) return error.InvalidChildren;
        if (fields.len == 0) return; // Arrow C++ leaves out an empty "children"
        try w.s.objectField("children");
        try w.s.beginArray();
        for (fields, data.children) |f, c| try w.column(f.name, c.data(), depth + 1);
        try w.s.endArray();
    }

    fn childColumns(w: *JsonWriter, data: *const ArrayData, names: []const []const u8, depth: usize) WriteError!void {
        if (data.children.len != names.len) return error.InvalidChildren;
        try w.s.objectField("children");
        try w.s.beginArray();
        for (names, data.children) |n, c| try w.column(n, c.data(), depth + 1);
        try w.s.endArray();
    }
};
