//! Arrow canonical extension types (format/CanonicalExtensions.rst): the names in the reserved
//! `arrow.` namespace, their required storage types and metadata, value constraints and typed
//! accessors. IPC and C data import call `validateType` for every extension field: a recognized
//! name with the wrong storage type or malformed metadata is `InvalidMetadata`; an unknown name
//! passes through untouched (the specification's rule). `validateFull` calls `validateValues`
//! for the value constraints (JSON syntax of `arrow.json`, shape products of
//! `arrow.variable_shape_tensor`, the offset range of `arrow.timestamp_with_offset`).
//!
//! Metadata JSON is checked by a strict, non-recursive RFC 8259 validator (`validJson`): at most
//! `max_metadata_len` bytes and nesting depth `max_metadata_depth`, then read with std.json.
const std = @import("std");
const datatype = @import("../datatype.zig");

const DataType = datatype.DataType;
const ExtensionType = datatype.ExtensionType;
const ArrayData = @import("../array/array_data.zig").ArrayData;

pub const Kind = enum {
    fixed_shape_tensor,
    variable_shape_tensor,
    json,
    uuid,
    @"opaque",
    bool8,
    parquet_variant,
    timestamp_with_offset,

    /// The registered name (`arrow.` prefix included).
    pub fn name(k: Kind) []const u8 {
        return switch (k) {
            .fixed_shape_tensor => "arrow.fixed_shape_tensor",
            .variable_shape_tensor => "arrow.variable_shape_tensor",
            .json => "arrow.json",
            .uuid => "arrow.uuid",
            .@"opaque" => "arrow.opaque",
            .bool8 => "arrow.bool8",
            .parquet_variant => "arrow.parquet.variant",
            .timestamp_with_offset => "arrow.timestamp_with_offset",
        };
    }
};

/// The canonical kind of an extension name, null for any other name.
pub fn kindOf(name: []const u8) ?Kind {
    for (std.meta.tags(Kind)) |k| {
        if (std.mem.eql(u8, k.name(), name)) return k;
    }
    return null;
}

pub const max_metadata_len = 64 * 1024;
pub const max_metadata_depth = 8;
/// Deepest nesting of an `arrow.json` value.
pub const max_value_depth = 1024;
/// Most dimensions of a tensor type.
pub const max_ndim = 32;
/// Largest |offset_minutes| of `arrow.timestamp_with_offset`: under a day. The specification
/// gives -779..780 as the usual range, not a bound.
pub const max_offset_minutes = 24 * 60 - 1;

pub const Error = error{InvalidMetadata};

// ------------------------------------------------------------------------------ JSON

/// Whether `s` is exactly one RFC 8259 JSON value (surrounding whitespace allowed), nested at
/// most `max_depth` deep. Iterative: the depth costs one bit of a fixed stack, no recursion.
pub fn validJson(s: []const u8, max_depth: usize) bool {
    std.debug.assert(max_depth <= max_value_depth);
    var stack: [max_value_depth / 8]u8 = undefined; // bit set: object, clear: array
    var depth: usize = 0;
    var i = ws(s, 0);
    value: while (true) {
        if (i >= s.len) return false;
        switch (s[i]) {
            '{', '[' => {
                if (depth >= max_depth) return false;
                const obj = s[i] == '{';
                setBit(&stack, depth, obj);
                depth += 1;
                i = ws(s, i + 1);
                if (i < s.len and s[i] == (if (obj) @as(u8, '}') else ']')) {
                    i += 1;
                    depth -= 1;
                } else {
                    if (obj) i = key(s, i) orelse return false;
                    continue :value;
                }
            },
            '"' => i = string(s, i) orelse return false,
            't' => i = literal(s, i, "true") orelse return false,
            'f' => i = literal(s, i, "false") orelse return false,
            'n' => i = literal(s, i, "null") orelse return false,
            else => i = number(s, i) orelse return false,
        }
        // After a value: close containers, or a comma and the next member.
        while (true) {
            i = ws(s, i);
            if (depth == 0) return i == s.len;
            if (i >= s.len) return false;
            const obj = getBit(&stack, depth - 1);
            if (s[i] == ',') {
                i = ws(s, i + 1);
                if (obj) i = key(s, i) orelse return false;
                continue :value;
            }
            if (s[i] != (if (obj) @as(u8, '}') else ']')) return false;
            i += 1;
            depth -= 1;
        }
    }
}

fn setBit(stack: []u8, k: usize, v: bool) void {
    const mask = @as(u8, 1) << @intCast(k % 8);
    if (v) stack[k / 8] |= mask else stack[k / 8] &= ~mask;
}

fn getBit(stack: []const u8, k: usize) bool {
    return stack[k / 8] & (@as(u8, 1) << @intCast(k % 8)) != 0;
}

fn ws(s: []const u8, start: usize) usize {
    var i = start;
    while (i < s.len and (s[i] == ' ' or s[i] == '\t' or s[i] == '\n' or s[i] == '\r')) i += 1;
    return i;
}

/// A member key and its colon; the position of the value.
fn key(s: []const u8, start: usize) ?usize {
    if (start >= s.len or s[start] != '"') return null;
    const i = ws(s, string(s, start) orelse return null);
    if (i >= s.len or s[i] != ':') return null;
    return ws(s, i + 1);
}

fn literal(s: []const u8, i: usize, word: []const u8) ?usize {
    return if (std.mem.startsWith(u8, s[i..], word)) i + word.len else null;
}

fn string(s: []const u8, start: usize) ?usize {
    var i = start + 1;
    while (i < s.len) {
        const c = s[i];
        if (c == '"') return i + 1;
        if (c < 0x20) return null;
        if (c == '\\') {
            if (i + 1 >= s.len) return null;
            switch (s[i + 1]) {
                '"', '\\', '/', 'b', 'f', 'n', 'r', 't' => i += 2,
                'u' => {
                    if (i + 6 > s.len) return null;
                    for (s[i + 2 .. i + 6]) |h| if (!std.ascii.isHex(h)) return null;
                    i += 6;
                },
                else => return null,
            }
            continue;
        }
        if (c >= 0x80) {
            const n = std.unicode.utf8ByteSequenceLength(c) catch return null;
            if (i + n > s.len) return null;
            _ = std.unicode.utf8Decode(s[i..][0..n]) catch return null;
            i += n;
            continue;
        }
        i += 1;
    }
    return null;
}

fn number(s: []const u8, start: usize) ?usize {
    var i = start;
    if (i < s.len and s[i] == '-') i += 1;
    if (i >= s.len) return null;
    if (s[i] == '0') {
        i += 1;
    } else if (s[i] >= '1' and s[i] <= '9') {
        while (i < s.len and std.ascii.isDigit(s[i])) i += 1;
    } else return null;
    if (i < s.len and s[i] == '.') {
        i += 1;
        const d = i;
        while (i < s.len and std.ascii.isDigit(s[i])) i += 1;
        if (i == d) return null;
    }
    if (i < s.len and (s[i] == 'e' or s[i] == 'E')) {
        i += 1;
        if (i < s.len and (s[i] == '+' or s[i] == '-')) i += 1;
        const d = i;
        while (i < s.len and std.ascii.isDigit(s[i])) i += 1;
        if (i == d) return null;
    }
    return i;
}

/// The metadata parsed as a JSON object (validated, bounded), in `arena`.
fn parseObject(arena: std.mem.Allocator, md: []const u8) Error!std.json.ObjectMap {
    if (md.len > max_metadata_len or !validJson(md, max_metadata_depth)) return error.InvalidMetadata;
    const v = std.json.parseFromSliceLeaky(std.json.Value, arena, md, .{}) catch return error.InvalidMetadata;
    return if (v == .object) v.object else error.InvalidMetadata;
}

// ------------------------------------------------------------------------------ types

/// The parsed parameters of `arrow.fixed_shape_tensor` / `arrow.variable_shape_tensor`.
pub const TensorParams = struct {
    ndim: usize = 0,
    /// fixed_shape_tensor: the physical shape (row-major). variable: `uniform_shape` (-1 for a
    /// non-uniform dimension), all -1 when absent.
    shape: [max_ndim]i64 = undefined,
    has_shape: bool = false,
    permutation: ?[max_ndim]u32 = null,
    dim_names: usize = 0,

    /// Row-major strides in bytes of the physical shape for elements of `elem_size` bytes.
    pub fn strides(self: *const TensorParams, elem_size: i64) [max_ndim]i64 {
        var out: [max_ndim]i64 = undefined;
        var acc = elem_size;
        var k = self.ndim;
        while (k > 0) {
            k -= 1;
            out[k] = acc;
            acc *|= @max(self.shape[k], 0);
        }
        return out;
    }
};

/// Parse the metadata of a tensor type. `fixed`: `shape` is required (`arrow.fixed_shape_tensor`);
/// otherwise `ndim` comes from the storage (`arrow.variable_shape_tensor`).
pub fn tensorParams(md: []const u8, fixed: bool, storage_ndim: ?usize) Error!TensorParams {
    var arena: std.heap.ArenaAllocator = .init(std.heap.page_allocator);
    defer arena.deinit();
    var p: TensorParams = .{};
    if (!fixed and md.len == 0) {
        p.ndim = storage_ndim orelse return error.InvalidMetadata;
        if (p.ndim > max_ndim) return error.InvalidMetadata;
        @memset(p.shape[0..p.ndim], -1);
        return p;
    }
    const obj = try parseObject(arena.allocator(), md);
    if (fixed) {
        const shape = obj.get("shape") orelse return error.InvalidMetadata;
        if (shape != .array or shape.array.items.len > max_ndim) return error.InvalidMetadata;
        p.ndim = shape.array.items.len;
        for (shape.array.items, 0..) |d, k| {
            if (d != .integer or d.integer < 0) return error.InvalidMetadata;
            p.shape[k] = d.integer;
        }
        p.has_shape = true;
    } else {
        p.ndim = storage_ndim orelse return error.InvalidMetadata;
        if (p.ndim > max_ndim) return error.InvalidMetadata;
        @memset(p.shape[0..p.ndim], -1);
        if (obj.get("uniform_shape")) |u| {
            if (u != .array or u.array.items.len != p.ndim) return error.InvalidMetadata;
            for (u.array.items, 0..) |d, k| switch (d) {
                .null => {},
                .integer => |n| {
                    if (n < 0 or n > std.math.maxInt(i32)) return error.InvalidMetadata;
                    p.shape[k] = n;
                },
                else => return error.InvalidMetadata,
            };
            p.has_shape = true;
        }
    }
    if (obj.get("dim_names")) |names| {
        if (names != .array or names.array.items.len != p.ndim) return error.InvalidMetadata;
        for (names.array.items) |n| if (n != .string) return error.InvalidMetadata;
        p.dim_names = p.ndim;
    }
    if (obj.get("permutation")) |perm| {
        if (perm != .array or perm.array.items.len != p.ndim) return error.InvalidMetadata;
        var seen: [max_ndim]bool = @splat(false);
        var out: [max_ndim]u32 = undefined;
        for (perm.array.items, 0..) |x, k| {
            if (x != .integer or x.integer < 0 or x.integer >= p.ndim or seen[@intCast(x.integer)]) return error.InvalidMetadata;
            seen[@intCast(x.integer)] = true;
            out[k] = @intCast(x.integer);
        }
        p.permutation = out;
    }
    return p;
}

/// `type_name` and `vendor_name` of `arrow.opaque` (both required strings), copied into `out`.
pub fn opaqueNames(md: []const u8, type_name: *std.ArrayList(u8), vendor_name: *std.ArrayList(u8), allocator: std.mem.Allocator) (Error || error{OutOfMemory})!void {
    var arena: std.heap.ArenaAllocator = .init(allocator);
    defer arena.deinit();
    const obj = try parseObject(arena.allocator(), md);
    const t = obj.get("type_name") orelse return error.InvalidMetadata;
    const v = obj.get("vendor_name") orelse return error.InvalidMetadata;
    if (t != .string or v != .string) return error.InvalidMetadata;
    try type_name.appendSlice(allocator, t.string);
    try vendor_name.appendSlice(allocator, v.string);
}

fn emptyMetadata(md: ?[]const u8) bool {
    return md == null or md.?.len == 0;
}

fn isBinaryLike(dt: DataType) bool {
    return switch (dt) {
        .binary, .large_binary, .binary_view => true,
        else => false,
    };
}

/// `dt` or the value type of a dictionary / run-end encoded `dt`.
fn decoded(dt: DataType) DataType {
    return switch (dt) {
        .dictionary => |d| d.value_type.*,
        .run_end_encoded => |r| r.value_type.*,
        else => dt,
    };
}

fn fieldNamed(fields: []const datatype.Field, name: []const u8) ?datatype.Field {
    for (fields) |f| if (std.mem.eql(u8, f.name, name)) return f;
    return null;
}

/// Check the storage type and metadata of `ext` when its name is canonical; any other name
/// passes. `InvalidMetadata` otherwise.
pub fn validateType(ext: ExtensionType) Error!void {
    const kind = kindOf(ext.name) orelse return;
    const md = ext.metadata orelse "";
    const storage = ext.storage_type.*;
    switch (kind) {
        .fixed_shape_tensor => {
            if (storage != .fixed_size_list) return error.InvalidMetadata;
            const p = try tensorParams(md, true, null);
            var product: i64 = 1;
            for (p.shape[0..p.ndim]) |d| product = std.math.mul(i64, product, d) catch return error.InvalidMetadata;
            if (product != storage.fixed_size_list.list_size) return error.InvalidMetadata;
        },
        .variable_shape_tensor => {
            if (storage != .struct_) return error.InvalidMetadata;
            const fields = storage.struct_.fields;
            const data = fieldNamed(fields, "data") orelse return error.InvalidMetadata;
            const shape = fieldNamed(fields, "shape") orelse return error.InvalidMetadata;
            if (fields.len != 2 or data.data_type.* != .list or shape.data_type.* != .fixed_size_list) return error.InvalidMetadata;
            const fsl = shape.data_type.fixed_size_list;
            if (fsl.value_field.data_type.* != .int32 or fsl.list_size < 0) return error.InvalidMetadata;
            _ = try tensorParams(md, false, @intCast(fsl.list_size));
        },
        .json => {
            switch (storage) {
                .string, .large_string, .string_view => {},
                else => return error.InvalidMetadata,
            }
            // Empty, or an object (future fields are allowed, never required).
            if (md.len > 0) _ = try parseObjectChecked(md);
        },
        .uuid => {
            if (storage != .fixed_size_binary or storage.fixed_size_binary.byte_width != 16) return error.InvalidMetadata;
            if (!emptyMetadata(ext.metadata)) return error.InvalidMetadata;
        },
        .@"opaque" => {
            var t: std.ArrayList(u8) = .empty;
            var v: std.ArrayList(u8) = .empty;
            const a = std.heap.page_allocator;
            defer t.deinit(a);
            defer v.deinit(a);
            opaqueNames(md, &t, &v, a) catch |e| return switch (e) {
                error.OutOfMemory => error.InvalidMetadata,
                error.InvalidMetadata => error.InvalidMetadata,
            };
        },
        .bool8 => if (storage != .int8 or !emptyMetadata(ext.metadata)) return error.InvalidMetadata,
        .parquet_variant => {
            if (storage != .struct_ or !emptyMetadata(ext.metadata)) return error.InvalidMetadata;
            const fields = storage.struct_.fields;
            const meta = fieldNamed(fields, "metadata") orelse return error.InvalidMetadata;
            if (meta.nullable or !isBinaryLike(decoded(meta.data_type.*))) return error.InvalidMetadata;
            const value = fieldNamed(fields, "value");
            const typed = fieldNamed(fields, "typed_value");
            if (value == null and typed == null) return error.InvalidMetadata;
            if (value) |v| if (!isBinaryLike(v.data_type.*)) return error.InvalidMetadata;
        },
        .timestamp_with_offset => {
            if (storage != .struct_ or !emptyMetadata(ext.metadata)) return error.InvalidMetadata;
            const fields = storage.struct_.fields;
            if (fields.len != 2 or !std.mem.eql(u8, fields[0].name, "timestamp") or !std.mem.eql(u8, fields[1].name, "offset_minutes")) return error.InvalidMetadata;
            if (fields[0].nullable or fields[1].nullable) return error.InvalidMetadata;
            const ts = fields[0].data_type.*;
            if (ts != .timestamp or ts.timestamp.timezone == null or !std.mem.eql(u8, ts.timestamp.timezone.?, "UTC")) return error.InvalidMetadata;
            if (decoded(fields[1].data_type.*) != .int16) return error.InvalidMetadata;
        },
    }
}

fn parseObjectChecked(md: []const u8) Error!void {
    var arena: std.heap.ArenaAllocator = .init(std.heap.page_allocator);
    defer arena.deinit();
    _ = try parseObject(arena.allocator(), md);
}

/// Equality of two extension types: the same name, equal storage types and equal metadata
/// (null and empty metadata are the same; `arrow.json` also treats `{}` as empty).
pub fn eql(a: ExtensionType, b: ExtensionType) bool {
    if (!std.mem.eql(u8, a.name, b.name) or !a.storage_type.eql(b.storage_type.*)) return false;
    const am = normalized(a);
    const bm = normalized(b);
    return std.mem.eql(u8, am, bm);
}

fn normalized(e: ExtensionType) []const u8 {
    const md = e.metadata orelse "";
    if (kindOf(e.name) == .json and std.mem.eql(u8, std.mem.trim(u8, md, " \t\r\n"), "{}")) return "";
    return md;
}

// ------------------------------------------------------------------------------ values

pub const ValueError = error{InvalidExtensionValue};

/// Value constraints of a canonical extension array that passed `validateLayout` (the storage
/// buffers are in bounds): every `arrow.json` value is JSON, every `arrow.variable_shape_tensor`
/// row has as many values as its shape's product (non-negative dimensions, equal to the
/// uniform ones), every `arrow.timestamp_with_offset` offset is under a day.
pub fn validateValues(data: *const ArrayData) ValueError!void {
    const ext = switch (data.data_type) {
        .extension => |e| e,
        else => return,
    };
    const kind = kindOf(ext.name) orelse return;
    // A type built without import checks: its storage must still be the required one.
    validateType(ext) catch return error.InvalidExtensionValue;
    switch (kind) {
        .json => {
            const storage = ext.storage_type.*;
            for (0..data.length) |i| {
                if (data.isNull(i)) continue;
                const v = stringAt(data, storage, i) orelse return error.InvalidExtensionValue;
                if (!validJson(v, max_value_depth)) return error.InvalidExtensionValue;
            }
        },
        .variable_shape_tensor => {
            const fields = ext.storage_type.struct_.fields;
            const data_i: usize = if (std.mem.eql(u8, fields[0].name, "data")) 0 else 1;
            const ndim: usize = @intCast(fields[1 - data_i].data_type.fixed_size_list.list_size);
            const p = tensorParams(ext.metadata orelse "", false, ndim) catch return error.InvalidExtensionValue;
            const list = data.children[data_i].data();
            const shapes = data.children[1 - data_i].data();
            const offsets = list.buffers[1].typedSlice(i32) catch return error.InvalidExtensionValue;
            const dims = shapes.children[0].data();
            const dvals = dims.buffers[1].typedSlice(i32) catch return error.InvalidExtensionValue;
            for (0..data.length) |i| {
                if (data.isNull(i)) continue;
                const row = data.offset + i;
                const lo = offsets[list.offset + row];
                const hi = offsets[list.offset + row + 1];
                var product: i64 = 1;
                for (0..p.ndim) |k| {
                    const d = dvals[dims.offset + (shapes.offset + row) * p.ndim + k];
                    if (d < 0 or (p.shape[k] >= 0 and d != p.shape[k])) return error.InvalidExtensionValue;
                    product *|= d;
                }
                if (product != hi - lo) return error.InvalidExtensionValue;
            }
        },
        .timestamp_with_offset => {
            const off = data.children[1].data();
            if (off.data_type != .int16) return; // dictionary / run-end encoded offsets
            const vals = off.buffers[1].typedSlice(i16) catch return error.InvalidExtensionValue;
            for (0..data.length) |i| {
                if (data.isNull(i)) continue;
                const v = vals[off.offset + data.offset + i];
                if (@abs(v) > max_offset_minutes) return error.InvalidExtensionValue;
            }
        },
        else => {},
    }
}

fn stringAt(data: *const ArrayData, storage: DataType, i: usize) ?[]const u8 {
    const row = data.offset + i;
    switch (storage) {
        inline .string, .large_string => |_, tag| {
            const O = if (tag == .string) i32 else i64;
            const offs = data.buffers[1].typedSlice(O) catch return null;
            return data.buffers[2].data[@intCast(offs[row])..@intCast(offs[row + 1])];
        },
        .string_view => {
            const v = data.buffers[1].data[row * 16 ..][0..16];
            const n = std.mem.readInt(u32, v[0..4], .little);
            if (n <= 12) return v[4..][0..n];
            const buf = std.mem.readInt(u32, v[8..12], .little);
            const at = std.mem.readInt(u32, v[12..16], .little);
            return data.buffers[2 + buf].data[at..][0..n];
        },
        else => return null,
    }
}

// ------------------------------------------------------------------------------ accessors

/// The 16 bytes of row `i` of an `arrow.uuid` array (big-endian, as stored), null for a null row.
pub fn uuidAt(data: *const ArrayData, i: usize) ?[16]u8 {
    std.debug.assert(i < data.length);
    if (data.isNull(i)) return null;
    return data.buffers[1].data[(data.offset + i) * 16 ..][0..16].*;
}

/// Row `i` of an `arrow.bool8` array: false for 0, true for any other value, null for a null row.
pub fn bool8At(data: *const ArrayData, i: usize) ?bool {
    std.debug.assert(i < data.length);
    if (data.isNull(i)) return null;
    return data.buffers[1].data[data.offset + i] != 0;
}

test {
    _ = @import("test_canonical.zig");
}
