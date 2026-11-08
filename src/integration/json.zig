//! The Arrow integration-test JSON format (https://arrow.apache.org/docs/format/Integration.html,
//! Arrow C++ `cpp/src/arrow/integration/json_internal.cc`): `read` turns a document into a schema
//! and record batches (dictionaries resolved, `isDelta` dictionaries appended in order), `write`
//! is the inverse. The reader is a trust boundary: any input gives a document whose every column
//! passed `validateFull`, or an error. See docs/integration-suite.md.
const std = @import("std");
const datatype = @import("../datatype.zig");
const schema_mod = @import("../schema.zig");
const array_data = @import("../array/array_data.zig");
const array_ref = @import("../array/array_ref.zig");
const buffer = @import("../buffer.zig");
const bitmap = @import("../bitmap.zig");
const record_batch = @import("../record_batch.zig");
const canonical = @import("../extension/canonical.zig");
const concat = @import("../concat_array_refs.zig");

const Allocator = std.mem.Allocator;
const Value = std.json.Value;
const ObjectMap = std.json.ObjectMap;
const Stringify = std.json.Stringify;
pub const DataType = datatype.DataType;
pub const Field = datatype.Field;
pub const KeyValue = datatype.KeyValue;
pub const IntType = datatype.IntType;
pub const Schema = schema_mod.Schema;
pub const SchemaRef = schema_mod.SchemaRef;
pub const ArrayData = array_data.ArrayData;
pub const ArrayRef = array_ref.ArrayRef;
pub const RecordBatch = record_batch.RecordBatch;
const SharedBuffer = buffer.SharedBuffer;
const OwnedBuffer = buffer.OwnedBuffer;
const storageDataType = datatype.storageDataType;

/// The writer (src/integration/json_writer.zig).
pub const write = @import("json_writer.zig").write;
pub const WriteError = @import("json_writer.zig").WriteError;

pub const extension_name_key = "ARROW:extension:name";
pub const extension_metadata_key = "ARROW:extension:metadata";

/// Deepest field / column nesting read or written; deeper documents are `NestingTooDeep`.
pub const max_depth: usize = 64;
/// Largest `count` (Arrow C++ reads it as an int32).
pub const max_count: usize = std.math.maxInt(i32);

pub const ReadError = error{
    /// Not JSON.
    InvalidJson,
    /// A required member is missing or has the wrong JSON type.
    InvalidDocument,
    /// An unknown type name or invalid type parameters.
    UnknownType,
    /// A buffer array whose length disagrees with `count`.
    LengthMismatch,
    /// A value that does not parse (bad hex, integer out of range, ...).
    InvalidValue,
    /// A dictionary-encoded field or dictionary batch whose id has no dictionary.
    MissingDictionary,
    NestingTooDeep,
    OutOfMemory,
} || array_data.FullValidationError || record_batch.RecordBatchError;

/// A parsed document: the schema and the record batches (each column carries its dictionary).
pub const Document = struct {
    allocator: Allocator,
    schema_ref: SchemaRef,
    batches: []RecordBatch,

    pub fn schema(self: *const Document) *const Schema {
        return self.schema_ref.schema();
    }

    pub fn deinit(self: *Document) void {
        for (self.batches) |*b| b.deinit();
        self.allocator.free(self.batches);
        self.schema_ref.release();
    }
};

// ---------------------------------------------------------------------------------------------
// Reading

/// Parses an integration JSON document. Every column (and dictionary) passed `validateFull`.
pub fn read(allocator: Allocator, text: []const u8) ReadError!Document {
    var parsed = std.json.parseFromSlice(Value, allocator, text, .{}) catch |e| return switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        else => error.InvalidJson,
    };
    defer parsed.deinit();
    const root = try asObject(parsed.value);

    // The parsed types live in this arena until the SchemaRef has its own copy; the arrays
    // point into the SchemaRef's types.
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var r: Reader = .{ .gpa = allocator, .arena = arena.allocator() };
    defer r.deinit();
    var schema_ref = try SchemaRef.fromBorrowed(allocator, try r.parseSchema(try asObject(try member(root, "schema"))));
    defer schema_ref.release();
    for (schema_ref.schema().fields) |f| try r.registerDictionaries(f.data_type, 0);

    if (root.get("dictionaries")) |dicts| {
        if (dicts != .null) for (try asArray(dicts)) |d| try r.parseDictionary(try asObject(d));
    }

    const batches_json = try asArray(try member(root, "batches"));
    const batches = try allocator.alloc(RecordBatch, batches_json.len);
    var filled: usize = 0;
    errdefer {
        for (batches[0..filled]) |*b| b.deinit();
        allocator.free(batches);
    }
    for (batches_json) |bj| {
        batches[filled] = try r.parseBatch(schema_ref, try asObject(bj));
        filled += 1;
    }
    return .{ .allocator = allocator, .schema_ref = schema_ref.retain(), .batches = batches };
}

const Reader = struct {
    gpa: Allocator,
    arena: Allocator,
    /// Dictionary id -> value type, from the schema.
    dict_types: std.AutoHashMapUnmanaged(i64, *const DataType) = .empty,
    /// Dictionary id -> current dictionary values.
    dicts: std.AutoHashMapUnmanaged(i64, ArrayRef) = .empty,

    fn deinit(self: *Reader) void {
        self.dict_types.deinit(self.gpa);
        var it = self.dicts.valueIterator();
        while (it.next()) |d| d.release();
        self.dicts.deinit(self.gpa);
    }

    fn dupe(self: *Reader, s: []const u8) error{OutOfMemory}![]const u8 {
        return self.arena.dupe(u8, s);
    }

    fn create(self: *Reader, dt: DataType) error{OutOfMemory}!*const DataType {
        const p = try self.arena.create(DataType);
        p.* = dt;
        return p;
    }

    fn parseSchema(self: *Reader, obj: ObjectMap) ReadError!Schema {
        const fields_json = try asArray(try member(obj, "fields"));
        const fields = try self.arena.alloc(Field, fields_json.len);
        for (fields_json, fields) |fj, *f| f.* = try self.parseField(fj, 0);
        const meta = try self.parseMetadata(obj.get("metadata"));
        if (meta.extension_name != null or meta.extension_metadata != null) {
            // Schema metadata has no extension meaning: keep the entries as they are.
            return .{ .fields = fields, .metadata = try self.parseMetadataRaw(obj.get("metadata")) };
        }
        return .{ .fields = fields, .metadata = meta.user };
    }

    fn parseField(self: *Reader, v: Value, depth: usize) ReadError!Field {
        if (depth >= max_depth) return error.NestingTooDeep;
        const obj = try asObject(v);
        const name = try self.dupe(try memberString(obj, "name"));
        const nullable = try memberBool(obj, "nullable");
        const type_obj = try asObject(try member(obj, "type"));
        const children_json = try asArray(try member(obj, "children"));
        const children = try self.arena.alloc(Field, children_json.len);
        for (children_json, children) |cj, *c| c.* = try self.parseField(cj, depth + 1);

        var dt = try self.parseType(type_obj, children);
        const meta = try self.parseMetadata(obj.get("metadata"));

        if (obj.get("dictionary")) |dv| {
            if (dv != .null) {
                const d = try asObject(dv);
                const id = try memberInt(i64, d, "id");
                const ordered = try memberBool(d, "isOrdered");
                const index_obj = try asObject(try member(d, "indexType"));
                if (!std.mem.eql(u8, try memberString(index_obj, "name"), "int")) return error.UnknownType;
                const index = try parseIntType(index_obj);
                const value_type = try self.create(dt); // before `dt` is overwritten
                dt = .{ .dictionary = .{ .id = id, .index_type = index, .value_type = value_type, .ordered = ordered } };
            }
        }

        // ARROW:extension:name wraps the storage type; over a dictionary it wraps the value type
        // (as the IPC reader does: src/ipc/stream_reader.zig buildFieldFromFlatbuf).
        if (meta.extension_name) |ext_name| {
            if (dt == .dictionary) {
                const ext = try self.create(.{ .extension = .{ .name = ext_name, .storage_type = dt.dictionary.value_type, .metadata = meta.extension_metadata } });
                dt.dictionary.value_type = ext;
            } else {
                const storage = try self.create(dt); // before `dt` is overwritten
                dt = .{ .extension = .{ .name = ext_name, .storage_type = storage, .metadata = meta.extension_metadata } };
            }
        } else if (meta.extension_metadata != null) return error.InvalidValue;
        const ext: ?datatype.ExtensionType = switch (dt) {
            .extension => |e| e,
            .dictionary => |d| if (d.value_type.* == .extension) d.value_type.extension else null,
            else => null,
        };
        if (ext) |e| canonical.validateType(e) catch return error.InvalidValue;

        return .{ .name = name, .data_type = try self.create(dt), .nullable = nullable, .metadata = meta.user };
    }

    /// Records the value type of every dictionary id in `dt` (of the SchemaRef's copy).
    fn registerDictionaries(self: *Reader, dt: *const DataType, depth: usize) ReadError!void {
        if (depth >= max_depth) return error.NestingTooDeep;
        switch (dt.*) {
            .dictionary => |d| {
                const gop = try self.dict_types.getOrPut(self.gpa, d.id.?);
                if (gop.found_existing) {
                    // A dictionary shared by several fields must have one value type.
                    if (!gop.value_ptr.*.eql(d.value_type.*)) return error.InvalidValue;
                } else gop.value_ptr.* = d.value_type;
                try self.registerDictionaries(d.value_type, depth + 1);
            },
            .extension => |e| try self.registerDictionaries(e.storage_type, depth + 1),
            .list, .large_list => |l| try self.registerDictionaries(l.value_field.data_type, depth + 1),
            .list_view, .large_list_view => |l| try self.registerDictionaries(l.value_field.data_type, depth + 1),
            .fixed_size_list => |l| try self.registerDictionaries(l.value_field.data_type, depth + 1),
            .struct_ => |st| for (st.fields) |f| try self.registerDictionaries(f.data_type, depth + 1),
            .sparse_union, .dense_union => |u| for (u.fields) |f| try self.registerDictionaries(f.data_type, depth + 1),
            .map => |m| try self.registerDictionaries(m.entries_type.?, depth + 1),
            .run_end_encoded => |ree| try self.registerDictionaries(ree.value_type, depth + 1),
            else => {},
        }
    }

    const ParsedMetadata = struct {
        user: ?[]const KeyValue = null,
        extension_name: ?[]const u8 = null,
        extension_metadata: ?[]const u8 = null,
    };

    /// Field metadata with the extension keys split off (null when there are no other keys).
    fn parseMetadata(self: *Reader, v: ?Value) ReadError!ParsedMetadata {
        const all = try self.parseMetadataRaw(v) orelse return .{};
        var out: ParsedMetadata = .{};
        var user: std.ArrayList(KeyValue) = .empty;
        for (all) |kv| {
            if (std.mem.eql(u8, kv.key, extension_name_key)) {
                if (out.extension_name != null) return error.InvalidValue;
                out.extension_name = kv.value;
            } else if (std.mem.eql(u8, kv.key, extension_metadata_key)) {
                if (out.extension_metadata != null) return error.InvalidValue;
                out.extension_metadata = kv.value;
            } else try user.append(self.arena, kv);
        }
        if (user.items.len > 0) out.user = user.items;
        return out;
    }

    fn parseMetadataRaw(self: *Reader, v: ?Value) ReadError!?[]const KeyValue {
        const val = v orelse return null;
        if (val == .null) return null;
        const items = try asArray(val);
        if (items.len == 0) return null;
        const out = try self.arena.alloc(KeyValue, items.len);
        for (items, out) |item, *kv| {
            const o = try asObject(item);
            kv.* = .{ .key = try self.dupe(try memberString(o, "key")), .value = try self.dupe(try memberString(o, "value")) };
        }
        return out;
    }

    fn parseType(self: *Reader, obj: ObjectMap, children: []const Field) ReadError!DataType {
        const name = try memberString(obj, "name");
        const leaf: ?DataType = if (eql(name, "null"))
            .{ .null = {} }
        else if (eql(name, "bool"))
            .{ .bool = {} }
        else if (eql(name, "utf8"))
            .{ .string = {} }
        else if (eql(name, "binary"))
            .{ .binary = {} }
        else if (eql(name, "largeutf8"))
            .{ .large_string = {} }
        else if (eql(name, "largebinary"))
            .{ .large_binary = {} }
        else if (eql(name, "utf8view"))
            .{ .string_view = {} }
        else if (eql(name, "binaryview"))
            .{ .binary_view = {} }
        else if (eql(name, "int"))
            intDataType(try parseIntType(obj))
        else if (eql(name, "floatingpoint"))
            try parseFloatingPoint(obj)
        else if (eql(name, "fixedsizebinary"))
            try parseFixedSizeBinary(obj)
        else if (eql(name, "decimal"))
            try parseDecimal(obj, null)
        else if (eql(name, "decimal32"))
            try parseDecimal(obj, 32)
        else if (eql(name, "decimal64"))
            try parseDecimal(obj, 64)
        else if (eql(name, "decimal256"))
            try parseDecimal(obj, 256)
        else if (eql(name, "date"))
            try parseDate(obj)
        else if (eql(name, "time"))
            try parseTime(obj)
        else if (eql(name, "timestamp"))
            try self.parseTimestamp(obj)
        else if (eql(name, "duration"))
            DataType{ .duration = .{ .unit = try parseTimeUnit(try memberString(obj, "unit")) } }
        else if (eql(name, "interval"))
            try parseInterval(obj)
        else
            null;
        if (leaf) |dt| {
            if (children.len != 0) return error.UnknownType;
            return dt;
        }

        if (eql(name, "list") or eql(name, "largelist") or eql(name, "listview") or eql(name, "largelistview")) {
            if (children.len != 1) return error.UnknownType;
            const lt: datatype.ListType = .{ .value_field = children[0] };
            if (eql(name, "list")) return .{ .list = lt };
            if (eql(name, "largelist")) return .{ .large_list = lt };
            if (eql(name, "listview")) return .{ .list_view = .{ .value_field = children[0] } };
            return .{ .large_list_view = .{ .value_field = children[0] } };
        }
        if (eql(name, "fixedsizelist")) {
            if (children.len != 1) return error.UnknownType;
            const size = try memberInt(i32, obj, "listSize");
            if (size < 0) return error.UnknownType;
            return .{ .fixed_size_list = .{ .value_field = children[0], .list_size = size } };
        }
        if (eql(name, "struct")) return .{ .struct_ = .{ .fields = children } };
        if (eql(name, "map")) {
            // One non-nullable struct child of a non-nullable key and an item (as the IPC reader).
            if (children.len != 1) return error.UnknownType;
            const entries = children[0];
            if (entries.nullable or entries.data_type.* != .struct_) return error.UnknownType;
            const kv = entries.data_type.struct_.fields;
            if (kv.len != 2 or kv[0].nullable) return error.UnknownType;
            return .{ .map = .{ .key_field = kv[0], .item_field = kv[1], .keys_sorted = try memberBool(obj, "keysSorted"), .entries_type = entries.data_type, .entries_name = entries.name } };
        }
        if (eql(name, "union")) {
            const mode = try memberString(obj, "mode");
            const ids_json = try asArray(try member(obj, "typeIds"));
            const ids = try self.arena.alloc(i8, children.len);
            if (ids_json.len == 0) {
                for (ids, 0..) |*id, i| id.* = std.math.cast(i8, i) orelse return error.UnknownType;
            } else {
                if (ids_json.len != children.len) return error.UnknownType;
                for (ids_json, ids) |j, *id| id.* = try intValue(i8, j);
            }
            const ut: datatype.UnionType = .{ .type_ids = ids, .fields = children, .mode = undefined };
            if (eql(mode, "SPARSE")) {
                var u = ut;
                u.mode = .sparse;
                return .{ .sparse_union = u };
            }
            if (eql(mode, "DENSE")) {
                var u = ut;
                u.mode = .dense;
                return .{ .dense_union = u };
            }
            return error.UnknownType;
        }
        if (eql(name, "runendencoded")) {
            if (children.len != 2) return error.UnknownType;
            const run_ends = children[0];
            if (!eql(run_ends.name, "run_ends") or !eql(children[1].name, "values") or run_ends.nullable) return error.UnknownType;
            const ret: IntType = switch (run_ends.data_type.*) {
                .int16 => .{ .bit_width = 16, .signed = true },
                .int32 => .{ .bit_width = 32, .signed = true },
                .int64 => .{ .bit_width = 64, .signed = true },
                else => return error.UnknownType,
            };
            return .{ .run_end_encoded = .{ .run_end_type = ret, .value_type = children[1].data_type } };
        }
        return error.UnknownType;
    }

    fn parseTimestamp(self: *Reader, obj: ObjectMap) ReadError!DataType {
        const unit = try parseTimeUnit(try memberString(obj, "unit"));
        var tz: ?[]const u8 = null;
        if (obj.get("timezone")) |v| {
            if (v != .null) {
                const s = try asString(v);
                // An empty timezone is no timezone (the IPC reader reads "" as none).
                if (s.len > 0) tz = try self.dupe(s);
            }
        }
        return .{ .timestamp = .{ .unit = unit, .timezone = tz } };
    }

    fn parseDictionary(self: *Reader, obj: ObjectMap) ReadError!void {
        const id = try memberInt(i64, obj, "id");
        const value_type = self.dict_types.get(id) orelse return error.MissingDictionary;
        const data = try asObject(try member(obj, "data"));
        const count = try memberCount(data, "count");
        const columns = try asArray(try member(data, "columns"));
        if (columns.len != 1) return error.InvalidDocument;
        var values = try self.parseColumn(value_type.*, columns[0], 0);
        defer values.release();
        if (values.data().length != count) return error.LengthMismatch;
        try values.data().validateFull();

        const is_delta = if (obj.get("isDelta")) |v| (if (v == .null) false else try asBool(v)) else false;
        const gop = try self.dicts.getOrPut(self.gpa, id);
        if (!gop.found_existing) {
            if (is_delta) {
                self.dicts.removeByPtr(gop.key_ptr);
                return error.MissingDictionary;
            }
            gop.value_ptr.* = values.retain();
            return;
        }
        if (!is_delta) {
            gop.value_ptr.release();
            gop.value_ptr.* = values.retain();
            return;
        }
        const parts = [_]ArrayRef{ gop.value_ptr.*, values };
        const merged = concat.concatArrayRefs(self.gpa, value_type.*, &parts) catch |e| return switch (e) {
            error.OutOfMemory => error.OutOfMemory,
            else => error.InvalidValue,
        };
        gop.value_ptr.release();
        gop.value_ptr.* = merged;
    }

    fn parseBatch(self: *Reader, schema_ref: SchemaRef, obj: ObjectMap) ReadError!RecordBatch {
        const count = try memberCount(obj, "count");
        const columns_json = try asArray(try member(obj, "columns"));
        const fields = schema_ref.schema().fields;
        if (columns_json.len != fields.len) return error.LengthMismatch;
        const columns = try self.gpa.alloc(ArrayRef, fields.len);
        defer self.gpa.free(columns);
        var filled: usize = 0;
        defer for (columns[0..filled]) |*c| c.release();
        for (columns_json, fields) |cj, f| {
            columns[filled] = try self.parseColumn(f.data_type.*, cj, 0);
            filled += 1;
            const col = columns[filled - 1].data();
            if (col.length != count) return error.LengthMismatch;
            try col.validateFull();
        }
        var ref = schema_ref.retain();
        var batch = RecordBatch.init(self.gpa, ref, columns) catch |e| {
            ref.release(); // init takes the reference only when it succeeds
            return e;
        };
        batch.num_rows = count;
        return batch;
    }

    /// One `FieldData` object of type `dt`; children are parsed with their field types.
    fn parseColumn(self: *Reader, dt: DataType, v: Value, depth: usize) ReadError!ArrayRef {
        if (depth >= max_depth) return error.NestingTooDeep;
        const obj = try asObject(v);
        const count = try memberCount(obj, "count");
        const storage = storageDataType(dt);

        var b: Build = .{ .gpa = self.gpa, .dt = dt, .length = count };
        defer b.deinit();

        switch (storage) {
            .null, .sparse_union, .dense_union, .run_end_encoded => {},
            else => try b.validityFrom(obj),
        }

        switch (storage) {
            .null => b.null_count = count,
            .bool => {
                const data = try dataArray(obj, "DATA", count);
                const bits = try b.bitmapBuffer();
                for (data, 0..) |item, i| {
                    if (!b.isValid(i)) continue;
                    const set = switch (item) {
                        .bool => |x| x,
                        else => (try intValue(u8, item)) != 0,
                    };
                    if (set) bitmap.setBit(bits, i);
                }
                try b.addBitmap();
            },
            .int8 => try b.fixed(i8, obj, intValue),
            .uint8 => try b.fixed(u8, obj, intValue),
            .int16 => try b.fixed(i16, obj, intValue),
            .uint16 => try b.fixed(u16, obj, intValue),
            .int32, .date32, .time32, .interval_months => try b.fixed(i32, obj, intValue),
            .uint32 => try b.fixed(u32, obj, intValue),
            .int64, .date64, .time64, .timestamp, .duration => try b.fixed(i64, obj, intValue),
            .uint64 => try b.fixed(u64, obj, intValue),
            .half_float => try b.fixed(u16, obj, intValue),
            .float => try b.fixed(f32, obj, floatValue),
            .double => try b.fixed(f64, obj, floatValue),
            .decimal32 => try b.fixed(i32, obj, decimalValue),
            .decimal64 => try b.fixed(i64, obj, decimalValue),
            .decimal128 => try b.fixed(i128, obj, decimalValue),
            .decimal256 => try b.fixed(i256, obj, decimalValue),
            .interval_day_time => try b.fixed(i64, obj, dayTimeValue),
            .interval_month_day_nano => try b.fixed(i128, obj, monthDayNanoValue),
            .fixed_size_binary => |fsb| {
                const width: usize = @intCast(fsb.byte_width);
                const data = try dataArray(obj, "DATA", count);
                const total = std.math.mul(usize, width, count) catch return error.InvalidValue;
                const out = try b.addBytes(total);
                for (data, 0..) |item, i| {
                    if (!b.isValid(i)) continue;
                    try hexInto(out[i * width ..][0..width], try asString(item));
                }
                try b.finishBytes();
            },
            .string, .binary => try b.varBinary(i32, obj, storage == .string),
            .large_string, .large_binary => try b.varBinary(i64, obj, storage == .large_string),
            .string_view, .binary_view => try b.views(obj, storage == .string_view),
            .list, .map => {
                try b.offsets(i32, obj, "OFFSET", count + 1, true);
                const child_dt = if (storage == .map) storage.map.entries_type.?.* else storage.list.value_field.data_type.*;
                try self.parseChildren(&b, obj, &.{child_dt}, depth);
            },
            .large_list => |lt| {
                try b.offsets(i64, obj, "OFFSET", count + 1, true);
                try self.parseChildren(&b, obj, &.{lt.value_field.data_type.*}, depth);
            },
            .list_view => |lt| {
                try b.offsets(i32, obj, "OFFSET", count, false);
                try b.offsets(i32, obj, "SIZE", count, false);
                try self.parseChildren(&b, obj, &.{lt.value_field.data_type.*}, depth);
            },
            .large_list_view => |lt| {
                try b.offsets(i64, obj, "OFFSET", count, false);
                try b.offsets(i64, obj, "SIZE", count, false);
                try self.parseChildren(&b, obj, &.{lt.value_field.data_type.*}, depth);
            },
            .fixed_size_list => |lt| try self.parseChildren(&b, obj, &.{lt.value_field.data_type.*}, depth),
            .struct_ => |st| try self.fieldChildren(&b, obj, st.fields, depth),
            .sparse_union, .dense_union => |ut| {
                try b.offsets(i8, obj, "TYPE_ID", count, false);
                if (storage == .dense_union) try b.offsets(i32, obj, "OFFSET", count, false);
                try self.fieldChildren(&b, obj, ut.fields, depth);
            },
            .run_end_encoded => |ree| {
                const run_end_dt = intDataType(ree.run_end_type);
                try self.parseChildren(&b, obj, &.{ run_end_dt, ree.value_type.* }, depth);
            },
            .dictionary => |dict| {
                switch (dict.index_type.bit_width) {
                    8 => if (dict.index_type.signed) try b.fixed(i8, obj, intValue) else try b.fixed(u8, obj, intValue),
                    16 => if (dict.index_type.signed) try b.fixed(i16, obj, intValue) else try b.fixed(u16, obj, intValue),
                    32 => if (dict.index_type.signed) try b.fixed(i32, obj, intValue) else try b.fixed(u32, obj, intValue),
                    64 => if (dict.index_type.signed) try b.fixed(i64, obj, intValue) else try b.fixed(u64, obj, intValue),
                    else => return error.UnknownType,
                }
                const d = self.dicts.get(dict.id orelse return error.MissingDictionary) orelse return error.MissingDictionary;
                b.dictionary = d.retain();
            },
            .extension => unreachable, // storageDataType unwraps extensions
        }
        return b.finish();
    }

    fn fieldChildren(self: *Reader, b: *Build, obj: ObjectMap, fields: []const Field, depth: usize) ReadError!void {
        var types_buf: [16]DataType = undefined;
        const types = if (fields.len <= types_buf.len) types_buf[0..fields.len] else try self.gpa.alloc(DataType, fields.len);
        defer if (fields.len > types_buf.len) self.gpa.free(types);
        for (fields, types) |f, *t| t.* = f.data_type.*;
        return self.parseChildren(b, obj, types, depth);
    }

    fn parseChildren(self: *Reader, b: *Build, obj: ObjectMap, types: []const DataType, depth: usize) ReadError!void {
        const items: []const Value = if (obj.get("children")) |c| (if (c == .null) &.{} else try asArray(c)) else &.{};
        if (items.len != types.len) return error.LengthMismatch;
        b.children = try self.gpa.alloc(ArrayRef, types.len);
        for (items, types) |item, t| {
            b.children[b.children_filled] = try self.parseColumn(t, item, depth + 1);
            b.children_filled += 1;
        }
    }
};

/// Buffers of one array under construction (released on error).
const Build = struct {
    gpa: Allocator,
    dt: DataType,
    length: usize,
    null_count: usize = 0,
    /// Per-row validity, null when the array has no validity bitmap.
    valid: ?[]bool = null,
    buffers: [3]SharedBuffer = undefined,
    buffer_count: usize = 0,
    /// Variadic buffers of a view array (after `buffers`).
    extra: std.ArrayList(SharedBuffer) = .empty,
    pending: ?OwnedBuffer = null,
    pending_len: usize = 0,
    children: []ArrayRef = &.{},
    children_filled: usize = 0,
    dictionary: ?ArrayRef = null,

    fn deinit(self: *Build) void {
        if (self.valid) |v| self.gpa.free(v);
        for (self.buffers[0..self.buffer_count]) |*buf| buf.release();
        for (self.extra.items) |*buf| buf.release();
        self.extra.deinit(self.gpa);
        if (self.pending) |*p| p.deinit();
        for (self.children[0..self.children_filled]) |*c| c.release();
        if (self.children.len > 0) self.gpa.free(self.children);
        if (self.dictionary) |*d| d.release();
    }

    fn isValid(self: *const Build, i: usize) bool {
        const v = self.valid orelse return true;
        return v[i];
    }

    fn push(self: *Build, buf: SharedBuffer) void {
        self.buffers[self.buffer_count] = buf;
        self.buffer_count += 1;
    }

    /// VALIDITY -> `valid`, and buffer 0 (empty when there is no null).
    fn validityFrom(self: *Build, obj: ObjectMap) ReadError!void {
        const items = try dataArray(obj, "VALIDITY", self.length);
        const valid = try self.gpa.alloc(bool, items.len);
        self.valid = valid;
        for (items, valid) |item, *v| {
            v.* = switch (item) {
                .bool => |x| x,
                else => (try intValue(i64, item)) != 0,
            };
            if (!v.*) self.null_count += 1;
        }
        if (self.null_count == 0) return self.push(SharedBuffer.empty);
        const bits = try self.bitmapBuffer();
        for (valid, 0..) |v, i| if (v) bitmap.setBit(bits, i);
        try self.addBitmap();
    }

    fn bitmapBuffer(self: *Build) error{OutOfMemory}![]u8 {
        const len = bitmap.byteLength(self.length);
        self.pending = try OwnedBuffer.init(self.gpa, len);
        self.pending_len = len;
        return self.pending.?.data[0..len];
    }

    fn addBitmap(self: *Build) error{OutOfMemory}!void {
        return self.finishBytes();
    }

    fn addBytes(self: *Build, len: usize) error{OutOfMemory}![]u8 {
        self.pending = try OwnedBuffer.init(self.gpa, len);
        self.pending_len = len;
        return self.pending.?.data[0..len];
    }

    fn finishBytes(self: *Build) error{OutOfMemory}!void {
        var p = self.pending.?;
        self.pending = null;
        errdefer p.deinit();
        self.push(try p.toShared(self.pending_len));
    }

    fn typed(self: *Build, comptime T: type, n: usize) error{OutOfMemory}![]T {
        const bytes = std.math.mul(usize, n, @sizeOf(T)) catch return error.OutOfMemory;
        _ = try self.addBytes(bytes);
        return self.pending.?.typedSlice(T)[0..n];
    }

    /// DATA of `length` values parsed by `parse` (null rows are zero, their JSON value unread).
    fn fixed(self: *Build, comptime T: type, obj: ObjectMap, comptime parse: anytype) ReadError!void {
        const data = try dataArray(obj, "DATA", self.length);
        const out = try self.typed(T, data.len);
        for (data, out, 0..) |item, *o, i| o.* = if (self.isValid(i)) try parse(T, item) else 0;
        try self.finishBytes();
    }

    /// `n` integers of member `key` taken as they are (offsets, sizes, type ids).
    /// `list_offsets`: the count + 1 offsets of a list, which an older writer leaves empty for an
    /// empty array.
    fn offsets(self: *Build, comptime T: type, obj: ObjectMap, key: []const u8, n: usize, list_offsets: bool) ReadError!void {
        const items = if (list_offsets) try offsetItems(obj, key, n) else try dataArray(obj, key, n);
        const out = try self.typed(T, n);
        if (items.len == 0) {
            if (n > 0) out[0] = 0; // older writers omit the one offset of an empty array
        } else for (items, out) |item, *o| o.* = try intValue(T, item);
        try self.finishBytes();
    }

    /// string / binary: the values of DATA (hex for binary); OFFSET only gives each value's
    /// length, which must match (Arrow C++ rebuilds the offsets the same way).
    fn varBinary(self: *Build, comptime O: type, obj: ObjectMap, utf8: bool) ReadError!void {
        const data = try dataArray(obj, "DATA", self.length);
        const offs = try offsetItems(obj, "OFFSET", self.length + 1);
        var total: usize = 0;
        for (data, 0..) |item, i| {
            if (!self.isValid(i)) continue;
            const s = try asString(item);
            if (!utf8 and s.len % 2 != 0) return error.InvalidValue;
            const len = if (utf8) s.len else s.len / 2;
            if (offs.len > 0) {
                const start = try intValue(i64, offs[i]);
                const end = try intValue(i64, offs[i + 1]);
                if (end < start or end - start != len) return error.LengthMismatch;
            }
            total = std.math.add(usize, total, len) catch return error.InvalidValue;
        }
        if (total > std.math.maxInt(O)) return error.InvalidValue;

        const out_offsets = try self.typed(O, self.length + 1);
        var pos: usize = 0;
        out_offsets[0] = 0;
        for (data, 0..) |item, i| {
            if (self.isValid(i)) {
                const s = try asString(item);
                pos += if (utf8) s.len else s.len / 2;
            }
            out_offsets[i + 1] = @intCast(pos);
        }
        try self.finishBytes();

        const bytes = try self.addBytes(total);
        pos = 0;
        for (data, 0..) |item, i| {
            if (!self.isValid(i)) continue;
            const s = try asString(item);
            if (utf8) {
                @memcpy(bytes[pos..][0..s.len], s);
                pos += s.len;
            } else {
                try hexInto(bytes[pos..][0 .. s.len / 2], s);
                pos += s.len / 2;
            }
        }
        try self.finishBytes();
    }

    /// utf8_view / binary_view: VIEWS (16-byte records as written) and VARIADIC_DATA_BUFFERS.
    fn views(self: *Build, obj: ObjectMap, utf8: bool) ReadError!void {
        const items = try dataArray(obj, "VIEWS", self.length);
        const data_bufs = try asArray(try member(obj, "VARIADIC_DATA_BUFFERS"));
        // Variadic buffer sizes first: the views are checked against them.
        const sizes = try self.gpa.alloc(usize, data_bufs.len);
        defer self.gpa.free(sizes);
        for (data_bufs, sizes) |db, *size| {
            const s = try asString(db);
            if (s.len % 2 != 0) return error.InvalidValue;
            size.* = s.len / 2;
        }

        const records = try self.addBytes(std.math.mul(usize, self.length, 16) catch return error.OutOfMemory);
        for (items, 0..) |item, i| {
            if (!self.isValid(i)) continue;
            const rec = records[i * 16 ..][0..16];
            const o = try asObject(item);
            const size = try memberInt(i32, o, "SIZE");
            if (size < 0) return error.InvalidValue;
            const len: usize = @intCast(size);
            std.mem.writeInt(u32, rec[0..4], @intCast(len), .little);
            if (len <= 12) {
                const inlined = try memberString(o, "INLINED");
                if (utf8) {
                    if (inlined.len != len) return error.LengthMismatch;
                    @memcpy(rec[4..][0..len], inlined);
                } else try hexInto(rec[4..][0..len], inlined);
                continue;
            }
            try hexInto(rec[4..8], try memberString(o, "PREFIX_HEX"));
            const index = try memberInt(u32, o, "BUFFER_INDEX");
            const offset = try memberInt(u32, o, "OFFSET");
            if (index >= sizes.len) return error.InvalidValue;
            if (offset > sizes[index] or len > sizes[index] - offset) return error.InvalidValue;
            std.mem.writeInt(u32, rec[8..12], index, .little);
            std.mem.writeInt(u32, rec[12..16], offset, .little);
        }
        try self.finishBytes();

        try self.extra.ensureTotalCapacity(self.gpa, data_bufs.len);
        for (data_bufs, sizes) |db, size| {
            const bytes = try self.addBytes(size);
            try hexInto(bytes, try asString(db));
            var p = self.pending.?;
            self.pending = null;
            errdefer p.deinit();
            self.extra.appendAssumeCapacity(try p.toShared(size));
        }
    }

    fn finish(self: *Build) ReadError!ArrayRef {
        const bufs = try self.gpa.alloc(SharedBuffer, self.buffer_count + self.extra.items.len);
        defer self.gpa.free(bufs);
        @memcpy(bufs[0..self.buffer_count], self.buffers[0..self.buffer_count]);
        @memcpy(bufs[self.buffer_count..], self.extra.items);
        const out = try ArrayRef.fromOwned(self.gpa, .{
            .data_type = self.dt,
            .length = self.length,
            .null_count = self.null_count,
            .buffers = bufs,
            .children = self.children[0..self.children_filled],
            .dictionary = self.dictionary,
        });
        // The node holds these references now.
        self.buffer_count = 0;
        self.extra.clearRetainingCapacity();
        self.children_filled = 0;
        self.dictionary = null;
        return out;
    }
};

/// The `n` items of an offset-like member; an empty array where one 0 offset is due (older
/// writers' empty arrays) comes back empty.
fn offsetItems(obj: ObjectMap, key: []const u8, n: usize) ReadError![]const Value {
    const items: []const Value = if (obj.get(key)) |v| try asArray(v) else if (n == 1) &.{} else return error.InvalidDocument;
    if (items.len == 0 and n == 1) return items;
    if (items.len != n) return error.LengthMismatch;
    return items;
}

// ---------------------------------------------------------------------------------------------
// JSON value helpers

fn eql(a: []const u8, b: []const u8) bool {
    return std.mem.eql(u8, a, b);
}

fn member(obj: ObjectMap, key: []const u8) ReadError!Value {
    return obj.get(key) orelse error.InvalidDocument;
}

fn asObject(v: Value) ReadError!ObjectMap {
    return switch (v) {
        .object => |o| o,
        else => error.InvalidDocument,
    };
}

fn asArray(v: Value) ReadError![]const Value {
    return switch (v) {
        .array => |a| a.items,
        else => error.InvalidDocument,
    };
}

fn asString(v: Value) ReadError![]const u8 {
    return switch (v) {
        .string => |s| s,
        else => error.InvalidDocument,
    };
}

fn asBool(v: Value) ReadError!bool {
    return switch (v) {
        .bool => |b| b,
        else => error.InvalidDocument,
    };
}

fn memberString(obj: ObjectMap, key: []const u8) ReadError![]const u8 {
    return asString(try member(obj, key));
}

fn memberBool(obj: ObjectMap, key: []const u8) ReadError!bool {
    return asBool(try member(obj, key));
}

fn memberInt(comptime T: type, obj: ObjectMap, key: []const u8) ReadError!T {
    return intValue(T, try member(obj, key));
}

fn memberCount(obj: ObjectMap, key: []const u8) ReadError!usize {
    const n = try memberInt(i64, obj, key);
    if (n < 0 or n > max_count) return error.InvalidValue;
    return @intCast(n);
}

/// The array member `key`, which must have `count` items.
fn dataArray(obj: ObjectMap, key: []const u8, count: usize) ReadError![]const Value {
    const items = try asArray(try member(obj, key));
    if (items.len != count) return error.LengthMismatch;
    return items;
}

/// An integer given as a JSON number or a decimal string (64-bit values are strings).
fn intValue(comptime T: type, v: Value) ReadError!T {
    return switch (v) {
        .integer => |i| std.math.cast(T, i) orelse error.InvalidValue,
        .string, .number_string => |s| std.fmt.parseInt(T, s, 10) catch error.InvalidValue,
        else => error.InvalidValue,
    };
}

fn floatValue(comptime T: type, v: Value) ReadError!T {
    const f: f64 = switch (v) {
        .float => |f| f,
        .integer => |i| @floatFromInt(i),
        .number_string => |s| std.fmt.parseFloat(f64, s) catch return error.InvalidValue,
        // Not JSON numbers: written as strings (see `writeFloat`).
        .string => |s| if (eql(s, "NaN")) std.math.nan(f64) else if (eql(s, "Infinity")) std.math.inf(f64) else if (eql(s, "-Infinity")) -std.math.inf(f64) else return error.InvalidValue,
        else => return error.InvalidValue,
    };
    return @floatCast(f);
}

fn decimalValue(comptime T: type, v: Value) ReadError!T {
    return std.fmt.parseInt(T, try asString(v), 10) catch error.InvalidValue;
}

/// {"days": i32, "milliseconds": i32} as days in the low 32 bits.
fn dayTimeValue(comptime T: type, v: Value) ReadError!T {
    const o = try asObject(v);
    const days: u32 = @bitCast(try memberInt(i32, o, "days"));
    const ms: u32 = @bitCast(try memberInt(i32, o, "milliseconds"));
    return @bitCast(@as(u64, days) | (@as(u64, ms) << 32));
}

/// {"months": i32, "days": i32, "nanoseconds": i64}: months, days, nanoseconds from the low bits.
fn monthDayNanoValue(comptime T: type, v: Value) ReadError!T {
    const o = try asObject(v);
    const months: u32 = @bitCast(try memberInt(i32, o, "months"));
    const days: u32 = @bitCast(try memberInt(i32, o, "days"));
    const nanos: u64 = @bitCast(try memberInt(i64, o, "nanoseconds"));
    return @bitCast(@as(u128, months) | (@as(u128, days) << 32) | (@as(u128, nanos) << 64));
}

fn hexInto(out: []u8, hex: []const u8) ReadError!void {
    if (hex.len != out.len * 2) return error.InvalidValue;
    _ = std.fmt.hexToBytes(out, hex) catch return error.InvalidValue;
}

fn parseIntType(obj: ObjectMap) ReadError!IntType {
    const signed = try memberBool(obj, "isSigned");
    const width = try memberInt(i64, obj, "bitWidth");
    return switch (width) {
        8, 16, 32, 64 => .{ .bit_width = @intCast(width), .signed = signed },
        else => error.UnknownType,
    };
}

fn intDataType(t: IntType) DataType {
    return switch (t.bit_width) {
        8 => if (t.signed) .{ .int8 = {} } else .{ .uint8 = {} },
        16 => if (t.signed) .{ .int16 = {} } else .{ .uint16 = {} },
        32 => if (t.signed) .{ .int32 = {} } else .{ .uint32 = {} },
        else => if (t.signed) .{ .int64 = {} } else .{ .uint64 = {} },
    };
}

fn parseFloatingPoint(obj: ObjectMap) ReadError!DataType {
    const p = try memberString(obj, "precision");
    if (eql(p, "HALF")) return .{ .half_float = {} };
    if (eql(p, "SINGLE")) return .{ .float = {} };
    if (eql(p, "DOUBLE")) return .{ .double = {} };
    return error.UnknownType;
}

fn parseFixedSizeBinary(obj: ObjectMap) ReadError!DataType {
    const width = try memberInt(i32, obj, "byteWidth");
    // As the IPC reader: a positive width.
    if (width <= 0) return error.UnknownType;
    return .{ .fixed_size_binary = .{ .byte_width = width } };
}

fn parseDecimal(obj: ObjectMap, implied_width: ?i64) ReadError!DataType {
    const precision = try memberInt(i64, obj, "precision");
    const scale = try memberInt(i32, obj, "scale");
    const width = if (obj.get("bitWidth")) |w| try intValue(i64, w) else implied_width orelse 128;
    if (implied_width) |iw| if (iw != width) return error.UnknownType;
    const max: i64 = switch (width) {
        32 => 9,
        64 => 18,
        128 => 38,
        256 => 76,
        else => return error.UnknownType,
    };
    if (precision < 1 or precision > max) return error.UnknownType;
    const params: datatype.DecimalParams = .{ .precision = @intCast(precision), .scale = scale };
    return switch (width) {
        32 => .{ .decimal32 = params },
        64 => .{ .decimal64 = params },
        128 => .{ .decimal128 = params },
        else => .{ .decimal256 = params },
    };
}

fn parseDate(obj: ObjectMap) ReadError!DataType {
    const unit = try memberString(obj, "unit");
    if (eql(unit, "DAY")) return .{ .date32 = {} };
    if (eql(unit, "MILLISECOND")) return .{ .date64 = {} };
    return error.UnknownType;
}

fn parseTime(obj: ObjectMap) ReadError!DataType {
    const unit = try parseTimeUnit(try memberString(obj, "unit"));
    const width = try memberInt(i64, obj, "bitWidth");
    return switch (unit) {
        .second, .millisecond => if (width == 32) .{ .time32 = .{ .unit = unit } } else error.UnknownType,
        .microsecond, .nanosecond => if (width == 64) .{ .time64 = .{ .unit = unit } } else error.UnknownType,
    };
}

fn parseTimeUnit(s: []const u8) ReadError!datatype.TimeUnit {
    if (eql(s, "SECOND")) return .second;
    if (eql(s, "MILLISECOND")) return .millisecond;
    if (eql(s, "MICROSECOND")) return .microsecond;
    if (eql(s, "NANOSECOND")) return .nanosecond;
    return error.UnknownType;
}

fn parseInterval(obj: ObjectMap) ReadError!DataType {
    const unit = try memberString(obj, "unit");
    if (eql(unit, "YEAR_MONTH")) return .{ .interval_months = .{ .unit = .months } };
    if (eql(unit, "DAY_TIME")) return .{ .interval_day_time = .{ .unit = .day_time } };
    if (eql(unit, "MONTH_DAY_NANO")) return .{ .interval_month_day_nano = .{ .unit = .month_day_nano } };
    return error.UnknownType;
}
