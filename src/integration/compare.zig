//! Logical equality of schemas, arrays and record batches for the integration suite (what Arrow
//! C++'s `RecordBatch::Equals` checks): types, validity and the values of valid rows; null rows'
//! bytes, offsets bases and slicing do not matter. A mismatch is described in `Mismatch`.
const std = @import("std");
const datatype = @import("../datatype.zig");
const schema_mod = @import("../schema.zig");
const array_data = @import("../array/array_data.zig");
const array_mod = @import("../array/array.zig");
const record_batch = @import("../record_batch.zig");
const concat = @import("../concat_array_refs.zig");

const DataType = datatype.DataType;
const Schema = schema_mod.Schema;
const ArrayData = array_data.ArrayData;
const RecordBatch = record_batch.RecordBatch;
const storageDataType = datatype.storageDataType;

/// Deepest nesting compared (deeper arrays compare unequal).
pub const max_depth: usize = 64;

/// Where two values first differ: a `/`-separated path and a reason.
pub const Mismatch = struct {
    buf: [1024]u8 = undefined,
    len: usize = 0,

    pub fn message(self: *const Mismatch) []const u8 {
        return self.buf[0..self.len];
    }
};

/// Schemas equal as Arrow C++ compares them (`typesEqual`), plus schema and field metadata at
/// every level.
pub fn schemasEqual(a: Schema, b: Schema, why: ?*Mismatch) bool {
    var c: Cmp = .{ .why = why };
    if (a.endianness != b.endianness) return c.fail("schema: endianness differs", .{});
    if (!metadataEqual(a.metadata, b.metadata)) return c.fail("schema: metadata differs", .{});
    if (a.fields.len != b.fields.len) return c.fail("schema: {d} vs {d} fields", .{ a.fields.len, b.fields.len });
    for (a.fields, b.fields, 0..) |fa, fb, i| {
        if (!fieldsEqual(fa, fb, true, 0)) return c.fail("schema: field {d} differs: {f} vs {f}", .{ i, FieldFmt{ .f = fa }, FieldFmt{ .f = fb } });
    }
    return true;
}

/// Arrow's type equality (what `DataType::Equals` and pyarrow's `==` check), which differs from
/// zarrow's `DataType.eql` in what it ignores: dictionary ids (an IPC detail), the names of list
/// item fields and of map entries / key / item fields. Field metadata is compared.
pub fn typesEqual(a: DataType, b: DataType, depth: usize) bool {
    if (depth >= max_depth) return false;
    if (a.id() != b.id()) return false;
    const d = depth + 1;
    return switch (a) {
        .list, .large_list => |l| fieldsEqual(l.value_field, switch (b) {
            .list, .large_list => |r| r.value_field,
            else => unreachable,
        }, false, d),
        .list_view, .large_list_view => |l| fieldsEqual(l.value_field, switch (b) {
            .list_view, .large_list_view => |r| r.value_field,
            else => unreachable,
        }, false, d),
        .fixed_size_list => |l| l.list_size == b.fixed_size_list.list_size and fieldsEqual(l.value_field, b.fixed_size_list.value_field, false, d),
        .struct_ => |l| fieldListEqual(l.fields, b.struct_.fields, d),
        .sparse_union, .dense_union => |l| blk: {
            const r = if (b == .sparse_union) b.sparse_union else b.dense_union;
            break :blk std.mem.eql(i8, l.type_ids, r.type_ids) and fieldListEqual(l.fields, r.fields, d);
        },
        .map => |l| l.keys_sorted == b.map.keys_sorted and
            fieldsEqual(l.key_field, b.map.key_field, false, d) and
            fieldsEqual(l.item_field, b.map.item_field, false, d),
        .dictionary => |l| l.index_type.bit_width == b.dictionary.index_type.bit_width and
            l.index_type.signed == b.dictionary.index_type.signed and
            l.ordered == b.dictionary.ordered and
            typesEqual(l.value_type.*, b.dictionary.value_type.*, d),
        .run_end_encoded => |l| l.run_end_type.bit_width == b.run_end_encoded.run_end_type.bit_width and
            l.run_end_type.signed == b.run_end_encoded.run_end_type.signed and
            typesEqual(l.value_type.*, b.run_end_encoded.value_type.*, d),
        .extension => |l| std.mem.eql(u8, l.name, b.extension.name) and
            optionalEql(l.metadata, b.extension.metadata) and
            typesEqual(l.storage_type.*, b.extension.storage_type.*, d),
        else => datatype.dataTypeEql(a, b),
    };
}

fn fieldsEqual(a: datatype.Field, b: datatype.Field, check_name: bool, depth: usize) bool {
    if (check_name and !std.mem.eql(u8, a.name, b.name)) return false;
    return a.nullable == b.nullable and metadataEqual(a.metadata, b.metadata) and typesEqual(a.data_type.*, b.data_type.*, depth);
}

fn fieldListEqual(a: []const datatype.Field, b: []const datatype.Field, depth: usize) bool {
    if (a.len != b.len) return false;
    for (a, b) |x, y| if (!fieldsEqual(x, y, true, depth)) return false;
    return true;
}

fn metadataEqual(a: ?[]const datatype.KeyValue, b: ?[]const datatype.KeyValue) bool {
    const x = a orelse &.{};
    const y = b orelse &.{};
    if (x.len != y.len) return false;
    for (x, y) |p, q| if (!std.mem.eql(u8, p.key, q.key) or !std.mem.eql(u8, p.value, q.value)) return false;
    return true;
}

fn optionalEql(a: ?[]const u8, b: ?[]const u8) bool {
    if (a == null or b == null) return a == null and b == null;
    return std.mem.eql(u8, a.?, b.?);
}

/// Same schema, row count and columns.
pub fn batchesEqual(a: *const RecordBatch, b: *const RecordBatch, why: ?*Mismatch) bool {
    var c: Cmp = .{ .why = why };
    if (!schemasEqual(a.schema().*, b.schema().*, why)) return false;
    if (a.numRows() != b.numRows()) return c.fail("{d} vs {d} rows", .{ a.numRows(), b.numRows() });
    for (a.columns, b.columns, a.schema().fields) |ca, cb, f| {
        c.depth = 0;
        c.path[0] = f.name;
        c.depth = 1;
        if (ca.data().length != cb.data().length) return c.fail("length {d} vs {d}", .{ ca.data().length, cb.data().length });
        if (!c.range(ca.data(), 0, cb.data(), 0, ca.data().length)) return false;
    }
    return true;
}

/// Arrays of one type with equal length and rows.
pub fn arraysEqual(a: *const ArrayData, b: *const ArrayData, why: ?*Mismatch) bool {
    var c: Cmp = .{ .why = why };
    if (a.length != b.length) return c.fail("length {d} vs {d}", .{ a.length, b.length });
    return c.range(a, 0, b, 0, a.length);
}

/// `a` equals the first `a.length` rows of `b`.
pub fn arrayIsPrefix(a: *const ArrayData, b: *const ArrayData, why: ?*Mismatch) bool {
    var c: Cmp = .{ .why = why };
    if (a.length > b.length) return c.fail("length {d} > {d}", .{ a.length, b.length });
    return c.range(a, 0, b, 0, a.length);
}

const Cmp = struct {
    why: ?*Mismatch,
    path: [max_depth + 1][]const u8 = undefined,
    depth: usize = 0,
    /// Dictionary pairs already found equal.
    seen: [8][2]*const ArrayData = undefined,
    seen_len: usize = 0,

    fn fail(self: *Cmp, comptime fmt: []const u8, args: anytype) bool {
        const m = self.why orelse return false;
        var w: std.Io.Writer = .fixed(&m.buf);
        for (self.path[0..self.depth]) |p| w.print("{s}/", .{p}) catch {};
        w.print(fmt, args) catch {};
        m.len = w.end;
        return false;
    }

    fn child(self: *Cmp, name: []const u8, a: *const ArrayData, ai: usize, b: *const ArrayData, bi: usize, n: usize) bool {
        return self.childChecked(name, a, ai, b, bi, n, true);
    }

    /// `check_type` false: the caller compared the types already (map entries, whose struct
    /// field names Arrow does not compare).
    fn childChecked(self: *Cmp, name: []const u8, a: *const ArrayData, ai: usize, b: *const ArrayData, bi: usize, n: usize, check_type: bool) bool {
        if (self.depth >= max_depth) return self.fail("nested too deep", .{});
        self.path[self.depth] = name;
        self.depth += 1;
        defer self.depth -= 1;
        return self.rangeChecked(a, ai, b, bi, n, check_type);
    }

    /// Rows [ai, ai + n) of `a` (logical, the array offset not included) against rows of `b`.
    fn range(self: *Cmp, a: *const ArrayData, ai: usize, b: *const ArrayData, bi: usize, n: usize) bool {
        return self.rangeChecked(a, ai, b, bi, n, true);
    }

    fn rangeChecked(self: *Cmp, a: *const ArrayData, ai: usize, b: *const ArrayData, bi: usize, n: usize, check_type: bool) bool {
        if (check_type and !typesEqual(a.data_type, b.data_type, 0)) return self.fail("type {f} vs {f}", .{ TypeFmt{ .t = a.data_type }, TypeFmt{ .t = b.data_type } });
        if (ai + n > a.length or bi + n > b.length) return self.fail("rows out of range", .{});
        const st = storageDataType(a.data_type);
        if (st == .dictionary and !self.dictionaries(a, b)) return false;
        for (0..n) |k| {
            const i = ai + k;
            const j = bi + k;
            const null_a = st == .null or a.isNull(i);
            const null_b = st == .null or b.isNull(j);
            if (null_a != null_b) return self.fail("row {d}: null {} vs {}", .{ i, null_a, null_b });
            if (null_a) continue;
            if (!self.row(st, a, i, b, j)) return false;
        }
        return true;
    }

    fn dictionaries(self: *Cmp, a: *const ArrayData, b: *const ArrayData) bool {
        const da = (a.dictionary orelse return self.fail("no dictionary", .{})).data();
        const db = (b.dictionary orelse return self.fail("no dictionary", .{})).data();
        for (self.seen[0..self.seen_len]) |pair| if (pair[0] == da and pair[1] == db) return true;
        if (da.length != db.length) return self.fail("dictionary length {d} vs {d}", .{ da.length, db.length });
        if (!self.child("dictionary", da, 0, db, 0, da.length)) return false;
        if (self.seen_len < self.seen.len) {
            self.seen[self.seen_len] = .{ da, db };
            self.seen_len += 1;
        }
        return true;
    }

    /// One valid row (i of `a`, j of `b`, logical).
    fn row(self: *Cmp, st: DataType, a: *const ArrayData, i: usize, b: *const ArrayData, j: usize) bool {
        switch (st) {
            .null => return true,
            .bool => {
                const va = bitAt(a, 1, a.offset + i) orelse return self.fail("row {d}: bad buffer", .{i});
                const vb = bitAt(b, 1, b.offset + j) orelse return self.fail("row {d}: bad buffer", .{j});
                if (va != vb) return self.fail("row {d}: {} vs {}", .{ i, va, vb });
                return true;
            },
            .string, .binary => return self.bytes(i, (array_mod.BinaryArray{ .data = a }).value(i), (array_mod.BinaryArray{ .data = b }).value(j)),
            .large_string, .large_binary => return self.bytes(i, (array_mod.LargeBinaryArray{ .data = a }).value(i), (array_mod.LargeBinaryArray{ .data = b }).value(j)),
            .string_view, .binary_view => return self.bytes(i, (array_mod.BinaryViewArray{ .data = a }).value(i), (array_mod.BinaryViewArray{ .data = b }).value(j)),
            .list, .map => return self.listRow(i32, false, a, i, b, j),
            .large_list => return self.listRow(i64, false, a, i, b, j),
            .list_view => return self.listRow(i32, true, a, i, b, j),
            .large_list_view => return self.listRow(i64, true, a, i, b, j),
            .fixed_size_list => |fsl| {
                const size: usize = @intCast(fsl.list_size);
                if (a.children.len != 1 or b.children.len != 1) return self.fail("row {d}: no child", .{i});
                return self.child(fsl.value_field.name, a.children[0].data(), (a.offset + i) * size, b.children[0].data(), (b.offset + j) * size, size);
            },
            .struct_ => |s| {
                if (a.children.len != s.fields.len or b.children.len != s.fields.len) return self.fail("row {d}: children", .{i});
                for (s.fields, a.children, b.children) |f, ca, cb| {
                    if (!self.child(f.name, ca.data(), structRow(a, ca.data(), i), cb.data(), structRow(b, cb.data(), j), 1)) return false;
                }
                return true;
            },
            .sparse_union, .dense_union => |u| {
                const ta = typeId(a, i) orelse return self.fail("row {d}: bad type ids", .{i});
                const tb = typeId(b, j) orelse return self.fail("row {d}: bad type ids", .{j});
                if (ta != tb) return self.fail("row {d}: type id {d} vs {d}", .{ i, ta, tb });
                const table = concat.unionChildTable(u) catch return self.fail("bad union type", .{});
                const k = table[@as(u8, @bitCast(ta))];
                if (k >= a.children.len or k >= b.children.len) return self.fail("row {d}: type id {d} has no child", .{ i, ta });
                const pa = if (st == .dense_union) unionOffset(a, i) else a.offset + i;
                const pb = if (st == .dense_union) unionOffset(b, j) else b.offset + j;
                return self.child(u.fields[k].name, a.children[k].data(), pa orelse return self.fail("row {d}: bad offsets", .{i}), b.children[k].data(), pb orelse return self.fail("row {d}: bad offsets", .{j}), 1);
            },
            .run_end_encoded => {
                const pa = runIndex(a, i) orelse return self.fail("row {d}: bad run ends", .{i});
                const pb = runIndex(b, j) orelse return self.fail("row {d}: bad run ends", .{j});
                return self.child("values", a.children[1].data(), pa, b.children[1].data(), pb, 1);
            },
            .dictionary => |d| {
                const ia = indexAt(a, d.index_type, i) orelse return self.fail("row {d}: bad indices", .{i});
                const ib = indexAt(b, d.index_type, j) orelse return self.fail("row {d}: bad indices", .{j});
                if (ia != ib) return self.fail("row {d}: index {d} vs {d}", .{ i, ia, ib });
                return true;
            },
            .extension => unreachable,
            else => {
                // Fixed width: the value bytes.
                const w = ArrayData.fixedWidthByteSize(st) orelse return self.fail("unsupported type {s}", .{st.name()});
                const va = fixedAt(a, w, a.offset + i) orelse return self.fail("row {d}: bad buffer", .{i});
                const vb = fixedAt(b, w, b.offset + j) orelse return self.fail("row {d}: bad buffer", .{j});
                if (!std.mem.eql(u8, va, vb)) return self.fail("row {d}: value bytes {X} vs {X}", .{ i, va, vb });
                return true;
            },
        }
    }

    fn bytes(self: *Cmp, i: usize, va: array_data.AccessorError![]const u8, vb: array_data.AccessorError![]const u8) bool {
        const x = va catch return self.fail("row {d}: bad offsets", .{i});
        const y = vb catch return self.fail("row {d}: bad offsets", .{i});
        if (!std.mem.eql(u8, x, y)) return self.fail("row {d}: '{s}' vs '{s}'", .{ i, x[0..@min(x.len, 64)], y[0..@min(y.len, 64)] });
        return true;
    }

    fn listRow(self: *Cmp, comptime O: type, comptime view: bool, a: *const ArrayData, i: usize, b: *const ArrayData, j: usize) bool {
        const ra = listRange(O, view, a, i) orelse return self.fail("row {d}: bad offsets", .{i});
        const rb = listRange(O, view, b, j) orelse return self.fail("row {d}: bad offsets", .{j});
        if (ra[1] != rb[1]) return self.fail("row {d}: {d} vs {d} items", .{ i, ra[1], rb[1] });
        if (a.children.len != 1 or b.children.len != 1) return self.fail("row {d}: no child", .{i});
        const name = switch (a.data_type) {
            .list, .large_list => |l| l.value_field.name,
            .list_view, .large_list_view => |l| l.value_field.name,
            .map => |m| m.entries_name,
            else => "entries",
        };
        return self.childChecked(name, a.children[0].data(), ra[0], b.children[0].data(), rb[0], ra[1], storageDataType(a.data_type) != .map);
    }
};

/// Child start and length of list row `i` (logical).
fn listRange(comptime O: type, comptime view: bool, d: *const ArrayData, i: usize) ?[2]usize {
    const offsets = d.typedBufferAt(1, O) catch return null;
    const p = d.offset + i;
    if (view) {
        const sizes = d.typedBufferAt(2, O) catch return null;
        if (p >= offsets.len or p >= sizes.len) return null;
        return .{ std.math.cast(usize, offsets[p]) orelse return null, std.math.cast(usize, sizes[p]) orelse return null };
    }
    if (p + 1 >= offsets.len) return null;
    const start = std.math.cast(usize, offsets[p]) orelse return null;
    const end = std.math.cast(usize, offsets[p + 1]) orelse return null;
    if (end < start) return null;
    return .{ start, end - start };
}

/// Child row of struct row `i`: zarrow's rule (StructArray.field) is that a child as long as
/// the struct is row-aligned with it, any other child is in the struct's buffer coordinates.
fn structRow(parent: *const ArrayData, child_data: *const ArrayData, i: usize) usize {
    return if (child_data.length == parent.length) i else parent.offset + i;
}

fn bitAt(d: *const ArrayData, buf: usize, p: usize) ?bool {
    if (buf >= d.buffers.len) return null;
    const data = d.buffers[buf].data;
    if (p / 8 >= data.len) return null;
    return (data[p / 8] >> @intCast(p % 8)) & 1 == 1;
}

fn fixedAt(d: *const ArrayData, width: usize, p: usize) ?[]const u8 {
    if (d.buffers.len < 2) return null;
    const data = d.buffers[1].data;
    const start = std.math.mul(usize, p, width) catch return null;
    if (start + width > data.len) return null;
    return data[start..][0..width];
}

fn typeId(d: *const ArrayData, i: usize) ?i8 {
    const ids = d.typedBufferAt(0, i8) catch return null;
    if (d.offset + i >= ids.len) return null;
    return ids[d.offset + i];
}

fn unionOffset(d: *const ArrayData, i: usize) ?usize {
    const offsets = d.typedBufferAt(1, i32) catch return null;
    if (d.offset + i >= offsets.len) return null;
    return std.math.cast(usize, offsets[d.offset + i]);
}

fn indexAt(d: *const ArrayData, t: datatype.IntType, i: usize) ?i128 {
    return switch (t.bit_width) {
        8 => if (t.signed) intAt(i8, d, i) else intAt(u8, d, i),
        16 => if (t.signed) intAt(i16, d, i) else intAt(u16, d, i),
        32 => if (t.signed) intAt(i32, d, i) else intAt(u32, d, i),
        64 => if (t.signed) intAt(i64, d, i) else intAt(u64, d, i),
        else => null,
    };
}

fn intAt(comptime T: type, d: *const ArrayData, i: usize) ?i128 {
    const values = array_mod.PrimitiveArray(T){ .data = d };
    return values.value(i) catch null;
}

/// Physical row of the values child for logical row `i` of a run-end encoded array.
fn runIndex(d: *const ArrayData, i: usize) ?usize {
    if (d.children.len != 2) return null;
    const ends = d.children[0].data();
    const pos = d.offset + i;
    return switch (ends.data_type) {
        .int16 => findRun(i16, ends, pos),
        .int32 => findRun(i32, ends, pos),
        .int64 => findRun(i64, ends, pos),
        else => null,
    };
}

fn findRun(comptime T: type, ends: *const ArrayData, pos: usize) ?usize {
    const values = (array_mod.PrimitiveArray(T){ .data = ends }).values() catch return null;
    return concat.findRun(T, values, pos) catch null;
}

/// Arrow-style text of a field: `name: type [not null] {metadata}` (types recursively).
pub const FieldFmt = struct {
    f: datatype.Field,
    depth: usize = 0,

    pub fn format(self: FieldFmt, w: *std.Io.Writer) std.Io.Writer.Error!void {
        try w.print("{s}: {f}", .{ self.f.name, TypeFmt{ .t = self.f.data_type.*, .depth = self.depth } });
        if (!self.f.nullable) try w.writeAll(" not null");
        if (self.f.metadata) |m| {
            try w.writeAll(" {");
            for (m, 0..) |kv, i| try w.print("{s}{s}={s}", .{ if (i > 0) ", " else "", kv.key, kv.value });
            try w.writeAll("}");
        }
    }
};

pub const TypeFmt = struct {
    t: DataType,
    depth: usize = 0,

    pub fn format(self: TypeFmt, w: *std.Io.Writer) std.Io.Writer.Error!void {
        if (self.depth >= max_depth) return w.writeAll("...");
        const d = self.depth + 1;
        switch (self.t) {
            .list, .large_list => |l| try w.print("{s}<{f}>", .{ self.t.name(), FieldFmt{ .f = l.value_field, .depth = d } }),
            .list_view, .large_list_view => |l| try w.print("{s}<{f}>", .{ self.t.name(), FieldFmt{ .f = l.value_field, .depth = d } }),
            .fixed_size_list => |l| try w.print("fixed_size_list<{f}>[{d}]", .{ FieldFmt{ .f = l.value_field, .depth = d }, l.list_size }),
            .struct_, .sparse_union, .dense_union => {
                const fields = switch (self.t) {
                    .struct_ => |st| st.fields,
                    .sparse_union, .dense_union => |u| u.fields,
                    else => unreachable,
                };
                try w.print("{s}<", .{self.t.name()});
                for (fields, 0..) |f, i| try w.print("{s}{f}", .{ if (i > 0) ", " else "", FieldFmt{ .f = f, .depth = d } });
                try w.writeAll(">");
                switch (self.t) {
                    .sparse_union, .dense_union => |u| try w.print("{any}", .{u.type_ids}),
                    else => {},
                }
            },
            .map => |m| {
                try w.print("map<{f}, {f}", .{ FieldFmt{ .f = m.key_field, .depth = d }, FieldFmt{ .f = m.item_field, .depth = d } });
                if (m.keys_sorted) try w.writeAll(", keys_sorted");
                if (m.entries_type) |e| try w.print(", entries={f}", .{TypeFmt{ .t = e.*, .depth = d }});
                try w.writeAll(">");
            },
            .dictionary => |x| try w.print("dictionary<values={f}, indices={s}int{d}, id={?d}, ordered={}>", .{ TypeFmt{ .t = x.value_type.*, .depth = d }, if (x.index_type.signed) "" else "u", x.index_type.bit_width, x.id, x.ordered }),
            .run_end_encoded => |r| try w.print("run_end_encoded<int{d}, {f}>", .{ r.run_end_type.bit_width, TypeFmt{ .t = r.value_type.*, .depth = d } }),
            .extension => |e| try w.print("extension<{s}, {f}, {?s}>", .{ e.name, TypeFmt{ .t = e.storage_type.*, .depth = d }, e.metadata }),
            .timestamp => |ts| try w.print("timestamp[{s}, {?s}]", .{ @tagName(ts.unit), ts.timezone }),
            .decimal32, .decimal64, .decimal128, .decimal256 => |x| try w.print("{s}({d}, {d})", .{ self.t.name(), x.precision, x.scale }),
            .fixed_size_binary => |x| try w.print("fixed_size_binary[{d}]", .{x.byte_width}),
            .time32, .time64 => |x| try w.print("{s}[{s}]", .{ self.t.name(), @tagName(x.unit) }),
            .duration => |x| try w.print("duration[{s}]", .{@tagName(x.unit)}),
            else => try w.writeAll(self.t.name()),
        }
    }
};
