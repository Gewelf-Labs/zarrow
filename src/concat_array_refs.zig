const std = @import("std");
const datatype = @import("datatype.zig");
const array_ref_mod = @import("array/array_ref.zig");
const array_data = @import("array/array_data.zig");
const array_mod = @import("array/array.zig");
const buffer = @import("buffer.zig");
const bitmap = @import("bitmap.zig");

pub const DataType = datatype.DataType;
pub const ArrayRef = array_ref_mod.ArrayRef;
pub const ArrayData = array_data.ArrayData;

pub const ConcatArrayError = error{ OutOfMemory, InvalidInput, UnsupportedType };
pub const Error = ConcatArrayError || array_data.ValidationError;

/// Concatenates `parts` (all of type `data_type`) into one fresh array at offset 0, N-ary for
/// every layout. Dictionary parts with different dictionaries get one merged dictionary
/// (UnsupportedType for value types other than fixed-width and string/binary).
pub fn concatArrayRefs(
    allocator: std.mem.Allocator,
    data_type: DataType,
    parts: []const ArrayRef,
) Error!ArrayRef {
    if (parts.len == 0) return error.InvalidInput;
    return concatParts(allocator, data_type, parts);
}

/// Concat of ArrayRefs or Part windows; every path checks the part types itself.
fn concatParts(allocator: std.mem.Allocator, dt: DataType, parts: anytype) Error!ArrayRef {
    if (try concatManyFlat(allocator, dt, parts)) |out| return out;
    return switch (storageDataType(dt)) {
        .string_view, .binary_view => concatManyViews(allocator, dt, parts),
        .dictionary => |d| switch (d.index_type.bit_width) { // parts with different dictionaries
            8 => if (d.index_type.signed) concatDictionaries(i8, allocator, dt, parts) else concatDictionaries(u8, allocator, dt, parts),
            16 => if (d.index_type.signed) concatDictionaries(i16, allocator, dt, parts) else concatDictionaries(u16, allocator, dt, parts),
            32 => if (d.index_type.signed) concatDictionaries(i32, allocator, dt, parts) else concatDictionaries(u32, allocator, dt, parts),
            64 => if (d.index_type.signed) concatDictionaries(i64, allocator, dt, parts) else concatDictionaries(u64, allocator, dt, parts),
            else => error.InvalidInput,
        },
        else => concatManyNested(allocator, dt, parts),
    };
}

pub fn storageDataType(dt: DataType) DataType {
    return switch (dt) {
        .extension => |ext| storageDataType(ext.storage_type.*),
        else => dt,
    };
}

/// Union type id -> child index (0xFF: undeclared). Without declared ids the id is the child
/// index; on duplicate ids the first wins (as the validator).
pub fn unionChildTable(uni: datatype.UnionType) error{InvalidInput}![256]u8 {
    const n = uni.fields.len;
    if (n > 128) return error.InvalidInput;
    var child_of: [256]u8 = @splat(0xFF);
    if (uni.type_ids.len == 0) {
        for (0..n) |c| child_of[c] = @intCast(c);
    } else {
        if (uni.type_ids.len != n) return error.InvalidInput;
        var c = n;
        while (c > 0) : (c -= 1) child_of[@as(u8, @bitCast(uni.type_ids[c - 1]))] = @intCast(c - 1);
    }
    return child_of;
}

/// First run whose end is past `pos`. Non-monotone (hostile) run ends still give an in-bounds run.
pub fn findRun(comptime T: type, ends: []const T, pos: usize) error{InvalidInput}!usize {
    const p = std.math.cast(T, pos) orelse return error.InvalidInput;
    if (ends.len == 0) return error.InvalidInput;
    var base: usize = 0;
    var n = ends.len;
    while (n > 1) {
        const half = n / 2;
        if (ends[base + half - 1] <= p) base += half;
        n -= half;
    }
    base += @intFromBool(ends[base] <= p);
    if (base == ends.len) return error.InvalidInput;
    return base;
}

/// std.meta.eql first: copies of one type (parts sliced from one array) share its pointers.
inline fn checkPartType(d: *const ArrayData, dt: DataType) Error!void {
    if (!std.meta.eql(d.data_type, dt) and !datatype.dataTypeEql(d.data_type, dt)) return error.InvalidInput;
}

/// N-ary concat for string_view/binary_view: the 16-byte views are copied and the variadic data
/// buffers are shared (retained), each part's out-of-line views rebased onto the output index of
/// the part's first data buffer. A part with the same data buffers as the previous one (slices
/// of one array) reuses them instead of adding them again.
fn concatManyViews(allocator: std.mem.Allocator, dt: DataType, parts: anytype) Error!ArrayRef {
    const bases = try allocator.alloc(usize, parts.len);
    defer allocator.free(bases);
    var total_len: usize = 0;
    var maybe_nulls = false;
    var n_data: usize = 0;
    var prev: []const array_data.SharedBuffer = &.{};
    var base: usize = 0;
    var view: ArrayData = undefined;
    for (parts, bases) |*part, *b| {
        const d = partData(part, &view);
        try checkPartType(d, dt);
        if (d.buffers.len < 2) return error.InvalidInput;
        const end = std.math.add(usize, d.offset, d.length) catch return error.InvalidInput;
        if (d.buffers[1].len() / 16 < end) return error.InvalidInput;
        total_len = std.math.add(usize, total_len, d.length) catch return error.InvalidInput;
        maybe_nulls = maybe_nulls or ((d.null_count orelse 1) != 0 and !d.buffers[0].isEmpty());
        const data_bufs = d.buffers[2..];
        if (!sameBuffers(data_bufs, prev)) {
            base = n_data;
            n_data += data_bufs.len;
            prev = data_bufs;
        }
        b.* = base;
    }
    if (n_data > std.math.maxInt(u32)) return error.InvalidInput;

    const used = bitmap.byteLength(total_len);
    var validity = try buffer.OwnedBuffer.init(allocator, if (maybe_nulls) used else 0);
    errdefer validity.deinit();
    const views_len = std.math.mul(usize, total_len, 16) catch return error.InvalidInput;
    var views = try buffer.OwnedBuffer.initUninit(allocator, views_len);
    errdefer views.deinit();
    const bufs = try allocator.alloc(array_data.SharedBuffer, 2 + n_data);
    defer allocator.free(bufs);
    var row: usize = 0;
    var filled: usize = 0;
    for (parts, bases) |*part, b| {
        const d = partData(part, &view);
        if (maybe_nulls) try appendValidityBits(validity.data, row, d);
        const data_bufs = d.buffers[2..];
        if (b == filled) { // first part with these data buffers
            @memcpy(bufs[2 + filled ..][0..data_bufs.len], data_bufs);
            filled += data_bufs.len;
        }
        const src = std.mem.bytesAsSlice([16]u8, d.buffers[1].data[d.offset * 16 ..][0 .. d.length * 16]);
        for (src, std.mem.bytesAsSlice([16]u8, views.data[row * 16 ..][0 .. d.length * 16])) |v, *o| {
            o.* = v;
            const len = std.mem.readInt(u32, v[0..4], .little);
            if (len <= 12) continue;
            const index = std.mem.readInt(u32, v[8..12], .little);
            const start = std.mem.readInt(u32, v[12..16], .little);
            if (index >= data_bufs.len or @as(u64, start) + len > data_bufs[index].len()) return error.InvalidInput;
            std.mem.writeInt(u32, o[8..12], @intCast(b + index), .little);
        }
        row += d.length;
    }
    const null_count = if (maybe_nulls) total_len - bitmap.countSetBit(validity.data, total_len) else 0;

    bufs[0] = array_data.SharedBuffer.empty;
    bufs[1] = array_data.SharedBuffer.empty;
    errdefer for (bufs[0..2]) |*b| b.release();
    if (null_count == 0) validity.deinit() else bufs[0] = try validity.toShared(used);
    bufs[1] = try views.toShared(views_len);
    const data = ArrayData{ .data_type = dt, .length = total_len, .null_count = null_count, .buffers = bufs };
    try data.validateLayout();
    const out = try ArrayRef.fromOwnedInlineBuffers(allocator, data);
    for (bufs[2..]) |b| _ = b.retain();
    return out;
}

fn sameBuffers(a: []const array_data.SharedBuffer, b: []const array_data.SharedBuffer) bool {
    if (a.len != b.len) return false;
    for (a, b) |x, y| if (x.data.ptr != y.data.ptr or x.data.len != y.data.len) return false;
    return true;
}

const FlatKind = union(enum) {
    null_type,
    boolean,
    fixed: usize,
    binary32,
    binary64,
};

fn flatKind(layout_dt: DataType) Error!?FlatKind {
    return switch (layout_dt) {
        .null => .null_type,
        .bool => .boolean,
        .uint8, .int8 => .{ .fixed = 1 },
        .uint16, .int16, .half_float => .{ .fixed = 2 },
        .uint32, .int32, .float, .date32, .time32, .interval_months, .decimal32 => .{ .fixed = 4 },
        .uint64, .int64, .double, .date64, .time64, .timestamp, .duration, .interval_day_time, .decimal64 => .{ .fixed = 8 },
        .decimal128, .interval_month_day_nano => .{ .fixed = 16 },
        .decimal256 => .{ .fixed = 32 },
        .fixed_size_binary => |fsb| .{ .fixed = std.math.cast(usize, fsb.byte_width) orelse return error.InvalidInput },
        .dictionary => |d| .{ .fixed = d.index_type.bit_width / 8 }, // indices; parts share one dictionary
        .string, .binary => .binary32,
        .large_string, .large_binary => .binary64,
        else => null,
    };
}

/// N-ary concat for flat layouts (null, bool, fixed-width, binary/string) in two passes over
/// `parts`: one sizing pass (types, rows, bytes, may-have-nulls), then one copy pass that writes
/// validity and values of each part together. Nulls are counted once on the output bitmap.
/// `parts` are ArrayRefs or Part windows. Dictionary parts must share one dictionary.
/// Returns null for nested and view types and for dictionaries that differ.
fn concatManyFlat(allocator: std.mem.Allocator, dt: DataType, parts: anytype) Error!?ArrayRef {
    const kind = (try flatKind(storageDataType(dt))) orelse return null;
    var view: ArrayData = undefined;
    const dictionary = if (storageDataType(dt) == .dictionary) partData(&parts[0], &view).dictionary else null;

    // Copying a DataType for dataTypeEql touches a second cache line of every part's node;
    // for parameterless types the tag compare is the whole equality.
    const tag_only = switch (dt) {
        .null, .bool, .uint8, .int8, .uint16, .int16, .uint32, .int32, .uint64, .int64, .half_float, .float, .double, .string, .binary, .large_string, .large_binary, .date32, .date64 => true,
        else => false,
    };
    var total_len: usize = 0;
    var total_data: usize = 0;
    var maybe_nulls = false;
    for (parts) |*part| {
        const d = partData(part, &view);
        if (std.meta.activeTag(d.data_type) != std.meta.activeTag(dt)) return error.InvalidInput;
        if (!tag_only and !datatype.dataTypeEql(d.data_type, dt)) return error.InvalidInput;
        if (dictionary) |dict| {
            if (d.dictionary == null or d.dictionary.?.node != dict.node) return null;
        }
        total_len = std.math.add(usize, total_len, d.length) catch return error.InvalidInput;
        maybe_nulls = maybe_nulls or ((d.null_count orelse 1) != 0 and d.buffers.len > 0 and !d.buffers[0].isEmpty());
        const r: ByteRange = switch (kind) {
            .binary32 => try binaryDataRange(i32, d),
            .binary64 => try binaryDataRange(i64, d),
            else => .{ .start = 0, .end = 0 },
        };
        total_data = std.math.add(usize, total_data, r.end - r.start) catch return error.InvalidInput;
    }
    if (kind == .null_type) {
        return try ArrayRef.fromOwnedInlineBuffers(allocator, .{ .data_type = dt, .length = total_len, .null_count = total_len, .buffers = &.{} });
    }

    // Every output buffer is allocated once. Validity only if some part may hold a null.
    const used = bitmap.byteLength(total_len);
    const width: usize = switch (kind) {
        .fixed => |w| w,
        .binary32 => @sizeOf(i32),
        .binary64 => @sizeOf(i64),
        else => 0,
    };
    const values_len = switch (kind) {
        .boolean => used,
        .fixed => std.math.mul(usize, total_len, width) catch return error.InvalidInput,
        else => (total_len + 1) * width, // offsets
    };
    const is_binary = kind == .binary32 or kind == .binary64;
    if (total_data > (if (kind == .binary32) @as(usize, std.math.maxInt(i32)) else @as(usize, std.math.maxInt(i64)))) return error.InvalidInput;
    var validity = try buffer.OwnedBuffer.init(allocator, if (maybe_nulls) used else 0);
    errdefer validity.deinit();
    // Bool values are OR-ed bits (need zeroed memory); other slots are all written below.
    var values = try (if (kind == .boolean) buffer.OwnedBuffer.init(allocator, values_len) else buffer.OwnedBuffer.initUninit(allocator, values_len + 16));
    errdefer values.deinit();
    var data_bytes = try buffer.OwnedBuffer.initUninit(allocator, total_data + 16); // wildCopy slack
    errdefer data_bytes.deinit();

    var row: usize = 0;
    var byte_pos: usize = 0;
    for (parts) |*part| {
        const d = partData(part, &view);
        if (maybe_nulls) try appendValidityBits(validity.data, row, d);
        switch (kind) {
            .null_type => unreachable,
            .boolean => {
                try requireBufferCount(d, 2);
                try requireBitmapBits(d.buffers[1], d.offset + d.length);
                bitmap.copyBits(values.data, row, d.buffers[1].data, d.offset, d.length);
            },
            .fixed => {
                try requireBufferCount(d, 2);
                const bytes = try dataBytesForFixedWidth(d, width);
                wildCopy(values.data[row * width ..], d.buffers[1].data, d.offset * width, bytes.len);
            },
            .binary32 => byte_pos = try copyBinaryPart(i32, values.typedSlice(i32), data_bytes.data, row, byte_pos, d),
            .binary64 => byte_pos = try copyBinaryPart(i64, values.typedSlice(i64), data_bytes.data, row, byte_pos, d),
        }
        row += d.length;
    }
    if (kind == .binary32) values.typedSlice(i32)[0] = 0;
    if (kind == .binary64) values.typedSlice(i64)[0] = 0;
    const null_count = if (maybe_nulls) total_len - bitmap.countSetBit(validity.data, total_len) else 0;

    var bufs: [3]array_data.SharedBuffer = .{ array_data.SharedBuffer.empty, array_data.SharedBuffer.empty, array_data.SharedBuffer.empty };
    errdefer for (&bufs) |*b| b.release();
    if (null_count == 0) validity.deinit() else bufs[0] = try validity.toShared(used);
    bufs[1] = try values.toShared(values_len);
    if (is_binary) bufs[2] = try data_bytes.toShared(total_data) else data_bytes.deinit();
    const buf_count: usize = if (is_binary) 3 else 2;

    const data = ArrayData{
        .data_type = dt,
        .length = total_len,
        .null_count = null_count,
        .buffers = bufs[0..buf_count],
        .dictionary = dictionary,
    };
    try data.validateLayout();
    const out = try ArrayRef.fromOwnedInlineBuffers(allocator, data);
    if (dictionary) |dict| _ = dict.retain();
    return out;
}

/// Concat of dictionary parts that do not share one dictionary (index type T). The merged
/// dictionary holds the first dictionary as is, then each new distinct value of the next ones in
/// first-seen order (null entries share one slot). Each distinct dictionary is read once, when a
/// part first refers to it: its values are hashed into an old -> new index table (per value, not
/// per row) and new values are appended to the merged values; then each part's indices are
/// gathered through its table. The merged dictionary must fit T, and every non-null index must
/// lie in its part's dictionary (InvalidInput otherwise).
fn concatDictionaries(comptime T: type, allocator: std.mem.Allocator, dt: DataType, parts: anytype) Error!ArrayRef {
    const value_type = storageDataType(dt).dictionary.value_type.*;
    const kind = (try flatKind(storageDataType(value_type))) orelse return error.UnsupportedType;
    if (kind == .null_type or kind == .boolean or storageDataType(value_type) == .dictionary) return error.UnsupportedType;
    if (kind == .fixed and kind.fixed == 0) return error.UnsupportedType;

    // Pass 1: types, rows, and each distinct dictionary (by node) with its rows in `remap`.
    const Table = struct { base: usize, len: usize, merged: bool = false };
    var tables: std.AutoHashMapUnmanaged(usize, Table) = .empty; // dictionary node -> remap rows
    defer tables.deinit(allocator);
    var remap_len: usize = 0;
    var total_len: usize = 0;
    var maybe_nulls = false;
    var view: ArrayData = undefined;
    var last_node: usize = 0;
    const width: usize = switch (kind) {
        .fixed => |w| w,
        .binary32 => 4,
        else => 8,
    };
    for (parts) |*part| {
        const d = partData(part, &view);
        try checkPartType(d, dt);
        const dict = d.dictionary orelse return error.InvalidInput;
        total_len = std.math.add(usize, total_len, d.length) catch return error.InvalidInput;
        maybe_nulls = maybe_nulls or ((d.null_count orelse 1) != 0 and d.buffers.len > 0 and !d.buffers[0].isEmpty());
        if (@intFromPtr(dict.node) == last_node) continue;
        last_node = @intFromPtr(dict.node);
        const gop = try tables.getOrPut(allocator, last_node);
        if (gop.found_existing) continue;
        // The values buffer bounds the length before anything is sized by it (checked fully when merged).
        const vd = dict.data();
        if (!datatype.dataTypeEql(vd.data_type, value_type)) return error.InvalidInput; // the dictionary must have the declared value type
        const end = std.math.add(usize, vd.offset, vd.length) catch return error.InvalidInput;
        if (vd.buffers.len < 2 or vd.buffers[1].len() / width < end) return error.InvalidInput;
        gop.value_ptr.* = .{ .base = remap_len, .len = vd.length };
        remap_len = std.math.add(usize, remap_len, vd.length) catch return error.InvalidInput;
    }
    const used = bitmap.byteLength(total_len);
    var validity = try buffer.OwnedBuffer.init(allocator, if (maybe_nulls) used else 0);
    errdefer validity.deinit();
    const values_len = std.math.mul(usize, total_len, @sizeOf(T)) catch return error.InvalidInput;
    var values = try buffer.OwnedBuffer.initUninit(allocator, values_len);
    errdefer values.deinit();

    var merged: MergedValues = .{ .width = if (kind == .fixed) kind.fixed else 0 };
    defer merged.deinit(allocator);
    // Every scratch array is sized once (growing them costs more than the hashing on Windows).
    if (remap_len > std.math.maxInt(u32)) return error.InvalidInput;
    var keys: KeyMap = .empty;
    defer keys.deinit(allocator);
    try keys.ensureTotalCapacityContext(allocator, @intCast(remap_len), EntryContext{ .v = &merged });
    if (kind == .fixed) try merged.data.ensureTotalCapacity(allocator, std.math.mul(usize, remap_len, width) catch return error.InvalidInput) else try merged.ends.ensureTotalCapacity(allocator, remap_len);
    var null_slot: ?u32 = null;
    const remap = try allocator.alloc(T, remap_len);
    defer allocator.free(remap);
    var any_merged = false;
    var table: Table = undefined;
    last_node = 0;

    const out = values.typedSlice(T);
    var row: usize = 0;
    for (parts) |*part| {
        const d = partData(part, &view);
        if (maybe_nulls) try appendValidityBits(validity.data, row, d);
        const dict = d.dictionary.?;
        if (@intFromPtr(dict.node) != last_node) {
            last_node = @intFromPtr(dict.node);
            const t = tables.getPtr(last_node).?;
            if (!t.merged) {
                try mergeDictionary(T, allocator, kind, dict.data(), &merged, &keys, &null_slot, !any_merged, remap[t.base..][0..t.len]);
                t.merged = true;
                any_merged = true;
            }
            table = t.*;
        }
        const map = remap[table.base..][0..table.len];

        try requireBufferCount(d, 2);
        const idx = d.buffers[1].typedSlice(T) catch return error.InvalidInput;
        const end = std.math.add(usize, d.offset, d.length) catch return error.InvalidInput;
        if (end > idx.len) return error.InvalidInput;
        for (idx[d.offset..end], out[row..][0..d.length], row..) |x, *o, r| {
            if (maybe_nulls and !bitmap.bitIsSet(validity.data, r)) {
                o.* = 0; // null rows: index 0, whatever the part held
                continue;
            }
            const ix = std.math.cast(usize, x) orelse return error.InvalidInput;
            if (ix >= map.len) return error.InvalidInput;
            o.* = map[ix];
        }
        row += d.length;
    }
    const null_count = if (maybe_nulls) total_len - bitmap.countSetBit(validity.data, total_len) else 0;

    var dictionary = try merged.finish(allocator, value_type, kind);
    errdefer dictionary.release();
    var bufs: [2]array_data.SharedBuffer = .{ array_data.SharedBuffer.empty, array_data.SharedBuffer.empty };
    errdefer for (&bufs) |*b| b.release();
    if (null_count == 0) validity.deinit() else bufs[0] = try validity.toShared(used);
    bufs[1] = try values.toShared(values_len);
    const data = ArrayData{ .data_type = dt, .length = total_len, .null_count = null_count, .buffers = &bufs, .dictionary = dictionary };
    try data.validateLayout();
    return try ArrayRef.fromOwnedInlineBuffers(allocator, data);
}

/// Hashes the values of `vd` into `keys`, writing old index -> merged index to `remap`. The
/// first dictionary is appended as is (duplicates keep their own index); later ones only add
/// values not seen before. Null values share one entry (`null_slot`) or, without it, get none.
fn mergeDictionary(
    comptime T: type,
    allocator: std.mem.Allocator,
    kind: FlatKind,
    vd: *const ArrayData,
    merged: *MergedValues,
    keys: *KeyMap,
    null_slot: ?*?u32, // null: null values get no entry (index 0)
    first: bool,
    remap: []T,
) Error!void {
    const end = std.math.add(usize, vd.offset, vd.length) catch return error.InvalidInput;
    if (vd.buffers.len > 0 and !vd.buffers[0].isEmpty()) try requireBitmapBits(vd.buffers[0], end); // for isNull
    switch (kind) {
        .binary32 => _ = try binaryDataRange(i32, vd), // buffer count and the offsets window
        .binary64 => _ = try binaryDataRange(i64, vd),
        else => {},
    }
    const fixed: []const u8 = if (kind == .fixed) try dataBytesForFixedWidth(vd, kind.fixed) else &.{};
    for (remap, 0..) |*slot, i| {
        const bytes: ?[]const u8 = if (vd.isNull(i)) null else switch (kind) {
            .fixed => |w| fixed[i * w ..][0..w],
            .binary32 => try binaryValue(i32, vd, i),
            else => try binaryValue(i64, vd, i),
        };
        var m: u32 = undefined;
        if (bytes) |b| {
            const gop = try keys.getOrPutContextAdapted(allocator, b, BytesProbe{ .v = merged }, EntryContext{ .v = merged });
            if (!gop.found_existing) gop.key_ptr.* = @intCast(merged.count);
            if (first or !gop.found_existing) try merged.append(allocator, b);
            m = if (first) @intCast(merged.count - 1) else gop.key_ptr.*;
        } else if (null_slot) |ns| {
            if (first or ns.* == null) try merged.append(allocator, null);
            if (ns.* == null) ns.* = @intCast(merged.count - 1);
            m = if (first) @intCast(merged.count - 1) else ns.*.?;
        } else m = 0; // masked: the row stays null, no dictionary entry
        if (m > std.math.maxInt(T)) return error.InvalidInput; // the merged dictionary does not fit T
        slot.* = @intCast(m);
    }
}

/// Distinct values of a merged dictionary: entry numbers, hashed through their bytes.
const KeyMap = std.HashMapUnmanaged(u32, void, EntryContext, std.hash_map.default_max_load_percentage);

/// Map context (rehash on growth): an entry number's bytes.
const EntryContext = struct {
    v: *const MergedValues,

    pub fn hash(c: EntryContext, m: u32) u64 {
        return std.hash.Wyhash.hash(0, c.v.at(m));
    }
    pub fn eql(_: EntryContext, a: u32, b: u32) bool {
        return a == b;
    }
};

/// Lookup of a value's bytes among the entries.
const BytesProbe = struct {
    v: *const MergedValues,

    pub fn hash(_: BytesProbe, b: []const u8) u64 {
        return std.hash.Wyhash.hash(0, b);
    }
    pub fn eql(p: BytesProbe, b: []const u8, m: u32) bool {
        return std.mem.eql(u8, b, p.v.at(m));
    }
};

/// Values of a merged dictionary (fixed-width or string/binary bytes) as they are appended.
const MergedValues = struct {
    width: usize, // fixed width, 0 for string/binary
    data: std.ArrayList(u8) = .empty,
    ends: std.ArrayList(usize) = .empty, // string/binary: end offset of each entry
    count: usize = 0,
    nulls: std.ArrayList(usize) = .empty, // entries that are null

    fn deinit(self: *MergedValues, allocator: std.mem.Allocator) void {
        self.data.deinit(allocator);
        self.ends.deinit(allocator);
        self.nulls.deinit(allocator);
    }

    fn at(self: MergedValues, m: u32) []const u8 {
        if (self.width != 0) return self.data.items[m * self.width ..][0..self.width];
        return self.data.items[if (m == 0) 0 else self.ends.items[m - 1]..self.ends.items[m]];
    }

    fn append(self: *MergedValues, allocator: std.mem.Allocator, bytes: ?[]const u8) Error!void {
        if (self.count >= std.math.maxInt(u32)) return error.InvalidInput;
        if (bytes) |b| try self.data.appendSlice(allocator, b) else {
            try self.nulls.append(allocator, self.count);
            if (self.width != 0) try self.data.appendNTimes(allocator, 0, self.width);
        }
        if (self.width == 0) try self.ends.append(allocator, self.data.items.len);
        self.count += 1;
    }

    fn finish(self: *MergedValues, allocator: std.mem.Allocator, value_type: DataType, kind: FlatKind) Error!ArrayRef {
        const n = self.count;
        const used = bitmap.byteLength(n);
        var validity = try buffer.OwnedBuffer.init(allocator, if (self.nulls.items.len > 0) used else 0);
        errdefer validity.deinit();
        if (self.nulls.items.len > 0) {
            bitmap.setBits(validity.data, 0, n);
            for (self.nulls.items) |k| bitmap.clearBit(validity.data, k);
        }
        const ow: usize = if (kind == .binary32) 4 else 8;
        const offsets_len = if (self.width == 0) (n + 1) * ow else 0;
        if (kind == .binary32 and self.data.items.len > std.math.maxInt(i32)) return error.InvalidInput;
        var offsets = try buffer.OwnedBuffer.initUninit(allocator, offsets_len);
        errdefer offsets.deinit();
        if (self.width == 0) {
            if (ow == 4) {
                const o = offsets.typedSlice(i32);
                o[0] = 0;
                for (o[1..][0..n], self.ends.items) |*x, e| x.* = @intCast(e);
            } else {
                const o = offsets.typedSlice(i64);
                o[0] = 0;
                for (o[1..][0..n], self.ends.items) |*x, e| x.* = @intCast(e);
            }
        }
        var bytes = try buffer.OwnedBuffer.initUninit(allocator, self.data.items.len);
        errdefer bytes.deinit();
        @memcpy(bytes.data[0..self.data.items.len], self.data.items);

        var bufs: [3]array_data.SharedBuffer = .{ array_data.SharedBuffer.empty, array_data.SharedBuffer.empty, array_data.SharedBuffer.empty };
        errdefer for (&bufs) |*b| b.release();
        if (self.nulls.items.len > 0) bufs[0] = try validity.toShared(used) else validity.deinit();
        if (self.width == 0) {
            bufs[1] = try offsets.toShared(offsets_len);
            bufs[2] = try bytes.toShared(self.data.items.len);
        } else {
            offsets.deinit();
            bufs[1] = try bytes.toShared(self.data.items.len);
        }
        const data = ArrayData{
            .data_type = value_type,
            .length = n,
            .null_count = self.nulls.items.len,
            .buffers = bufs[0..if (self.width == 0) 3 else 2],
        };
        try data.validateLayout();
        return try ArrayRef.fromOwnedInlineBuffers(allocator, data);
    }
};

/// Dictionary-encodes a fixed-width or string/binary array: int32 indices into its distinct
/// values in first-seen order (each row hashed once); null rows stay null and add no entry.
/// `value_type` must equal the array's type and outlive the result (as for DictionaryBuilder).
pub fn dictionaryEncode(allocator: std.mem.Allocator, array: ArrayRef, value_type: *const DataType) Error!ArrayRef {
    const d = array.data();
    if (!datatype.dataTypeEql(value_type.*, d.data_type)) return error.InvalidInput;
    const kind = (try flatKind(storageDataType(d.data_type))) orelse return error.UnsupportedType;
    if (kind == .null_type or kind == .boolean or storageDataType(d.data_type) == .dictionary) return error.UnsupportedType;
    if (kind == .fixed and kind.fixed == 0) return error.UnsupportedType;
    const width: usize = switch (kind) {
        .fixed => |w| w,
        .binary32 => 4,
        else => 8,
    };
    const end = std.math.add(usize, d.offset, d.length) catch return error.InvalidInput;
    if (d.buffers.len < 2 or d.buffers[1].len() / width < end) return error.InvalidInput; // bounds the sizes below

    const maybe_nulls = (d.null_count orelse 1) != 0 and !d.buffers[0].isEmpty();
    const used = bitmap.byteLength(d.length);
    var validity = try buffer.OwnedBuffer.init(allocator, if (maybe_nulls) used else 0);
    errdefer validity.deinit();
    if (maybe_nulls) try appendValidityBits(validity.data, 0, d);
    var indices = try buffer.OwnedBuffer.initUninit(allocator, d.length * @sizeOf(i32));
    errdefer indices.deinit();
    var merged: MergedValues = .{ .width = if (kind == .fixed) kind.fixed else 0 };
    defer merged.deinit(allocator);
    var keys: KeyMap = .empty; // grows with the distinct values (a reserved table would be cold)
    defer keys.deinit(allocator);
    try mergeDictionary(i32, allocator, kind, d, &merged, &keys, null, false, indices.typedSlice(i32)[0..d.length]);
    const null_count = if (maybe_nulls) d.length - bitmap.countSetBit(validity.data, d.length) else 0;

    var dictionary = try merged.finish(allocator, value_type.*, kind);
    errdefer dictionary.release();
    var bufs: [2]array_data.SharedBuffer = .{ array_data.SharedBuffer.empty, array_data.SharedBuffer.empty };
    errdefer for (&bufs) |*b| b.release();
    if (null_count == 0) validity.deinit() else bufs[0] = try validity.toShared(used);
    bufs[1] = try indices.toShared(d.length * @sizeOf(i32));
    const out = ArrayData{
        .data_type = .{ .dictionary = .{ .index_type = .{ .bit_width = 32, .signed = true }, .value_type = value_type } },
        .length = d.length,
        .null_count = null_count,
        .buffers = &bufs,
        .dictionary = dictionary,
    };
    try out.validateLayout();
    return try ArrayRef.fromOwnedInlineBuffers(allocator, out);
}

/// Row `i` of a string/binary array whose offsets passed binaryDataRange (each value checked).
fn binaryValue(comptime O: type, d: *const ArrayData, i: usize) Error![]const u8 {
    const offsets = d.buffers[1].typedSlice(O) catch return error.InvalidInput;
    const start = std.math.cast(usize, offsets[d.offset + i]) orelse return error.InvalidInput;
    const end = std.math.cast(usize, offsets[d.offset + i + 1]) orelse return error.InvalidInput;
    if (end < start or end > d.buffers[2].len()) return error.InvalidInput;
    return d.buffers[2].data[start..end];
}

/// Rows [start, start + len) of `ref`: nested concat records every part's child windows in one
/// pass over the parts and hands them down instead of allocating an ArrayRef slice per window.
const Part = struct { ref: ArrayRef, start: usize, len: usize };

/// A part's rows (`part` points to an ArrayRef or a Part); a partial window is sliced into `view`.
inline fn partData(part: anytype, view: *ArrayData) *const ArrayData {
    if (@TypeOf(part) == *const ArrayRef) return part.data();
    const d = part.ref.data();
    if (part.start == 0 and part.len == d.length) return d;
    view.* = d.slice(part.start, part.len);
    return view;
}

/// N-ary concat for the nested layouts (`parts`: ArrayRefs or Parts): list, large_list, map,
/// fixed_size_list, struct, list_view, large_list_view, sparse/dense union and run_end_encoded.
/// Pass 1 checks the parts and records every child's window, pass 2 writes validity (unions:
/// type ids) and the offsets, view offsets or run ends rebased onto the output, then each child
/// is concatenated once from its windows (REE: the run ends child is the one pass 2 wrote).
fn concatManyNested(allocator: std.mem.Allocator, dt: DataType, parts: anytype) Error!ArrayRef {
    const layout = storageDataType(dt);
    const n_children: usize = switch (layout) {
        .list, .large_list, .map, .fixed_size_list, .list_view, .large_list_view => 1,
        .struct_ => |s| s.fields.len,
        .sparse_union, .dense_union => |u| u.fields.len,
        .run_end_encoded => 2,
        else => return error.UnsupportedType,
    };
    // Width of what pass 2 writes: list / list_view / dense union offsets, or run ends.
    const width: usize = switch (layout) {
        .list, .map, .list_view, .dense_union => 4,
        .large_list, .large_list_view => 8,
        .run_end_encoded => |ree| switch (ree.run_end_type.bit_width) {
            16, 32, 64 => |bits| if (ree.run_end_type.signed) bits / 8 else return error.InvalidInput,
            else => return error.InvalidInput,
        },
        else => 0,
    };
    const max_pos: usize = switch (width) {
        2 => std.math.maxInt(i16),
        4 => std.math.maxInt(i32),
        else => std.math.maxInt(i64),
    };
    const buf_count: usize = switch (layout) {
        .fixed_size_list, .struct_, .sparse_union => 1,
        .list_view, .large_list_view => 3,
        .run_end_encoded => 0,
        else => 2,
    };
    const is_union = layout == .sparse_union or layout == .dense_union;
    const has_validity = buf_count > 0 and !is_union;
    const child_of: [256]u8 = if (layout == .dense_union) try unionChildTable(layout.dense_union) else undefined;
    const n = parts.len;

    // windows[c * n + p]: the rows of child c that part p covers.
    const windows = try allocator.alloc(Part, n_children * n);
    defer allocator.free(windows);
    var total_len: usize = 0;
    var maybe_nulls = false;
    var view: ArrayData = undefined;
    for (parts, 0..) |*part, p| {
        const d = partData(part, &view);
        try checkPartType(d, dt);
        if (d.buffers.len < buf_count or d.children.len != n_children) return error.InvalidInput;
        total_len = std.math.add(usize, total_len, d.length) catch return error.InvalidInput;
        if (has_validity) maybe_nulls = maybe_nulls or ((d.null_count orelse 1) != 0 and !d.buffers[0].isEmpty());
        if (layout == .dense_union) {
            try unionWindows(d, &child_of, windows, n, p);
        } else for (0..n_children) |c| {
            windows[c * n + p] = if (layout == .run_end_encoded and c == 1)
                .{ .ref = d.children[1], .start = windows[p].start, .len = windows[p].len } // runRange checked the lengths
            else
                try childWindow(layout, part, d, c);
        }
    }
    var runs: usize = 0; // REE: one output run per values window row (<= rows, see runRange)
    if (layout == .run_end_encoded) {
        if (total_len > max_pos) return error.InvalidInput;
        for (windows[0..n]) |w| runs += w.len;
    }

    const used = bitmap.byteLength(total_len);
    var validity = try buffer.OwnedBuffer.init(allocator, if (maybe_nulls) used else 0);
    errdefer validity.deinit();
    var ids = try buffer.OwnedBuffer.initUninit(allocator, if (is_union) total_len else 0);
    errdefer ids.deinit();
    const offsets_count: usize = switch (layout) {
        .list, .large_list, .map => std.math.add(usize, total_len, 1) catch return error.InvalidInput,
        .run_end_encoded => runs,
        else => if (width > 0) total_len else 0,
    };
    // Parts without buffers to bound their length (unions without children) may claim any length.
    const offsets_len = std.math.mul(usize, offsets_count, width) catch return error.InvalidInput;
    var offsets = try buffer.OwnedBuffer.initUninit(allocator, offsets_len);
    errdefer offsets.deinit();
    var sizes = try buffer.OwnedBuffer.initUninit(allocator, if (buf_count == 3) offsets_len else 0);
    errdefer sizes.deinit();
    if (maybe_nulls or width > 0 or is_union) {
        var row: usize = 0;
        var pos: [128]usize = @splat(0); // output rows so far of each offset-indexed child (REE: runs)
        for (parts, 0..) |*part, p| {
            const d = partData(part, &view);
            if (maybe_nulls) try appendValidityBits(validity.data, row, d);
            if (is_union) {
                const end = std.math.add(usize, d.offset, d.length) catch return error.InvalidInput;
                if (d.buffers[0].len() < end) return error.InvalidInput;
                @memcpy(ids.data[row..][0..d.length], d.buffers[0].data[d.offset..end]);
            }
            switch (layout) {
                .list, .map => rebaseOffsets(i32, offsets.typedSlice(i32), row, pos[0], d),
                .large_list => rebaseOffsets(i64, offsets.typedSlice(i64), row, pos[0], d),
                .list_view => rebaseViews(i32, offsets.typedSlice(i32), sizes.typedSlice(i32), row, pos[0], windows[p].start, d),
                .large_list_view => rebaseViews(i64, offsets.typedSlice(i64), sizes.typedSlice(i64), row, pos[0], windows[p].start, d),
                .dense_union => try rebaseUnionOffsets(offsets.typedSlice(i32)[row..][0..d.length], &child_of, windows, n, p, &pos, d),
                .run_end_encoded => switch (width) {
                    2 => try rebaseRunEnds(i16, offsets.typedSlice(i16)[pos[0]..][0..windows[p].len], row, windows[p], d),
                    4 => try rebaseRunEnds(i32, offsets.typedSlice(i32)[pos[0]..][0..windows[p].len], row, windows[p], d),
                    else => try rebaseRunEnds(i64, offsets.typedSlice(i64)[pos[0]..][0..windows[p].len], row, windows[p], d),
                },
                else => {},
            }
            // Offsets are checked against the output type after the part (the rebase wraps).
            if (width > 0) for (0..if (layout == .dense_union) n_children else 1) |c| {
                const len = windows[c * n + p].len;
                if (len > max_pos - pos[c]) return error.InvalidInput;
                pos[c] += len;
            };
            row += d.length;
        }
    }
    switch (layout) {
        .list, .map => offsets.typedSlice(i32)[0] = 0,
        .large_list => offsets.typedSlice(i64)[0] = 0,
        else => {},
    }
    const null_count = if (maybe_nulls) total_len - bitmap.countSetBit(validity.data, total_len) else 0;

    const children = try allocator.alloc(ArrayRef, n_children);
    var built: usize = 0;
    errdefer {
        for (children[0..built]) |*child| child.release();
        allocator.free(children);
    }
    for (children, 0..) |*child, c| {
        if (layout == .run_end_encoded and c == 0) {
            var ends_bufs = [2]array_data.SharedBuffer{ array_data.SharedBuffer.empty, try offsets.toShared(offsets_len) };
            child.* = ArrayRef.fromOwnedInlineBuffers(allocator, .{
                .data_type = switch (width) {
                    2 => .{ .int16 = {} },
                    4 => .{ .int32 = {} },
                    else => .{ .int64 = {} },
                },
                .length = runs,
                .null_count = 0,
                .buffers = &ends_bufs,
            }) catch |err| {
                ends_bufs[1].release();
                return err;
            };
        } else {
            const child_windows = windows[c * n ..][0..n];
            child.* = try concatParts(allocator, child_windows[0].ref.data().data_type, child_windows);
        }
        built += 1;
    }

    var bufs: [3]array_data.SharedBuffer = .{ array_data.SharedBuffer.empty, array_data.SharedBuffer.empty, array_data.SharedBuffer.empty };
    errdefer for (&bufs) |*b| b.release();
    if (is_union) bufs[0] = try ids.toShared(total_len) else if (null_count != 0) bufs[0] = try validity.toShared(used);
    if (buf_count >= 2) bufs[1] = try offsets.toShared(offsets_len);
    if (buf_count == 3) bufs[2] = try sizes.toShared(offsets_len);
    // Whatever was not handed to a buffer above is unused (deinit of a handed-over buffer is a no-op).
    validity.deinit();
    ids.deinit();
    offsets.deinit();
    sizes.deinit();

    const data = ArrayData{
        .data_type = dt,
        .length = total_len,
        .null_count = null_count,
        .buffers = bufs[0..buf_count],
        .children = children,
    };
    try data.validateLayout();
    return try ArrayRef.fromOwnedInlineBuffers(allocator, data);
}

/// Rows of child `c` that `part` (rows `d`) covers: the list offsets range, the [min, max) of the
/// list views, the runs overlapping the part (REE),
/// the fixed_size_list rows times list_size, or the struct / sparse union rows (row-aligned or
/// parent coordinates).
fn childWindow(layout: DataType, part: anytype, d: *const ArrayData, c: usize) Error!Part {
    const child = d.children[c];
    const cd = child.data();
    const r: ByteRange = switch (layout) {
        .list, .map => try offsetsRange(i32, d, cd.length),
        .large_list => try offsetsRange(i64, d, cd.length),
        .list_view => try viewsRange(i32, d, cd.length),
        .large_list_view => try viewsRange(i64, d, cd.length),
        .run_end_encoded => |ree| switch (ree.run_end_type.bit_width) {
            16 => try runRange(i16, d),
            32 => try runRange(i32, d),
            else => try runRange(i64, d),
        },
        .fixed_size_list => |f| blk: {
            const size = std.math.cast(usize, f.list_size) orelse return error.InvalidInput;
            const end = std.math.add(usize, d.offset, d.length) catch return error.InvalidInput;
            break :blk .{
                .start = std.math.mul(usize, d.offset, size) catch return error.InvalidInput,
                .end = std.math.mul(usize, end, size) catch return error.InvalidInput,
            };
        },
        .struct_, .sparse_union => blk: {
            // Like ArrayRef.slice: a child as long as the struct node is row-aligned with it.
            const node = if (@TypeOf(part) == *const ArrayRef) d else part.ref.data();
            const start = if (cd.length == node.length) d.offset - node.offset else d.offset;
            break :blk .{ .start = start, .end = std.math.add(usize, start, d.length) catch return error.InvalidInput };
        },
        else => unreachable,
    };
    if (r.end > cd.length) return error.InvalidInput;
    return .{ .ref = child, .start = r.start, .len = r.end - r.start };
}

/// [min offset, max offset + size) of a list view part; every view must lie inside the child.
fn viewsRange(comptime T: type, d: *const ArrayData, limit: usize) Error!ByteRange {
    const offsets = d.buffers[1].typedSlice(T) catch return error.InvalidInput;
    const sizes = d.buffers[2].typedSlice(T) catch return error.InvalidInput;
    const end = std.math.add(usize, d.offset, d.length) catch return error.InvalidInput;
    if (offsets.len < end or sizes.len < end) return error.InvalidInput;
    var lo: usize = std.math.maxInt(usize);
    var hi: usize = 0;
    for (offsets[d.offset..end], sizes[d.offset..end]) |o, s| {
        const start = std.math.cast(usize, o) orelse return error.InvalidInput;
        const size = std.math.cast(usize, s) orelse return error.InvalidInput;
        if (size > limit or start > limit - size) return error.InvalidInput;
        lo = @min(lo, start);
        hi = @max(hi, start + size);
    }
    return if (d.length == 0) .{ .start = 0, .end = 0 } else .{ .start = lo, .end = hi };
}

/// Writes `d`'s list views moved from their child window (starting at `start`) to `pos`.
fn rebaseViews(comptime T: type, offsets: []T, sizes: []T, row: usize, pos: usize, start: usize, d: *const ArrayData) void {
    const src_offsets = (d.buffers[1].typedSlice(T) catch unreachable)[d.offset..][0..d.length];
    const src_sizes = (d.buffers[2].typedSlice(T) catch unreachable)[d.offset..][0..d.length];
    const shift: T = @as(T, @intCast(pos)) - @as(T, @intCast(start));
    for (src_offsets, offsets[row..][0..d.length]) |o, *dst| dst.* = o +% shift;
    @memcpy(sizes[row..][0..d.length], src_sizes);
}

/// Child windows of dense union part p (child k at `windows[k * n + p]`; a union without
/// children has none): [min, max + 1) of the offsets of its rows of that child. Every type id
/// must be declared, every offset inside its child.
fn unionWindows(d: *const ArrayData, child_of: *const [256]u8, windows: []Part, n: usize, p: usize) Error!void {
    const offsets = d.buffers[1].typedSlice(i32) catch return error.InvalidInput;
    const end = std.math.add(usize, d.offset, d.length) catch return error.InvalidInput;
    if (d.buffers[0].len() < end or offsets.len < end) return error.InvalidInput;
    const nc = d.children.len;
    var lo_buf: [128]usize = undefined;
    var hi_buf: [128]usize = undefined;
    var limit_buf: [128]usize = undefined;
    const lo = lo_buf[0..nc];
    const hi = hi_buf[0..nc];
    const limit = limit_buf[0..nc];
    @memset(lo, std.math.maxInt(usize));
    @memset(hi, 0);
    for (d.children, limit) |child, *l| l.* = child.data().length;
    for (d.buffers[0].data[d.offset..end], offsets[d.offset..end]) |id, o| {
        const k = child_of[id];
        if (k == 0xFF) return error.InvalidInput;
        const r = std.math.cast(usize, o) orelse return error.InvalidInput;
        if (r >= limit[k]) return error.InvalidInput;
        lo[k] = @min(lo[k], r);
        hi[k] = @max(hi[k], r + 1);
    }
    for (d.children, lo, hi, 0..) |child, l, h, k| {
        const start = if (h == 0) 0 else l;
        windows[k * n + p] = .{ .ref = child, .start = start, .len = h - start };
    }
}

/// Writes `d`'s dense union offsets moved from each child's window (`windows[k * n + p]`) to `pos[k]`.
fn rebaseUnionOffsets(out: []i32, child_of: *const [256]u8, windows: []const Part, n: usize, p: usize, pos: *const [128]usize, d: *const ArrayData) Error!void {
    const offsets = d.buffers[1].typedSlice(i32) catch return error.InvalidInput; // bounds: unionWindows
    for (d.buffers[0].data[d.offset..][0..d.length], offsets[d.offset..][0..d.length], out) |id, o, *dst| {
        const k = child_of[id];
        if (k == 0xFF) return error.InvalidInput;
        dst.* = o -% @as(i32, @intCast(windows[k * n + p].start)) +% @as(i32, @intCast(pos[k]));
    }
}

/// Runs [k0, k1) of a run-end encoded part: the runs overlapping rows [offset, offset + length).
/// Strictly increasing run ends cover at least one row each, so k1 - k0 <= length.
fn runRange(comptime T: type, d: *const ArrayData) Error!ByteRange {
    const run_ends = d.children[0].data();
    if (run_ends.buffers.len < 2 or d.children[1].data().length != run_ends.length) return error.InvalidInput;
    const all = run_ends.buffers[1].typedSlice(T) catch return error.InvalidInput;
    const ends_end = std.math.add(usize, run_ends.offset, run_ends.length) catch return error.InvalidInput;
    if (all.len < ends_end) return error.InvalidInput;
    if (d.length == 0) return .{ .start = 0, .end = 0 };
    const ends = all[run_ends.offset..ends_end];
    const last = std.math.add(usize, d.offset, d.length - 1) catch return error.InvalidInput;
    const k0 = try findRun(T, ends, d.offset);
    const k1 = try findRun(T, ends, last) + 1;
    if (k1 <= k0 or k1 - k0 > d.length) return error.InvalidInput;
    return .{ .start = k0, .end = k1 };
}

/// Writes the run ends of `d`'s runs (window `w` of its run ends child) in output rows: clamped
/// to the part's rows and shifted to `row`. They must increase strictly inside the part.
fn rebaseRunEnds(comptime T: type, out: []T, row: usize, w: Part, d: *const ArrayData) Error!void {
    const run_ends = w.ref.data();
    const ends = (run_ends.buffers[1].typedSlice(T) catch unreachable)[run_ends.offset + w.start ..][0..w.len];
    const stop = d.offset + d.length;
    var prev = d.offset;
    for (ends, out, 1..) |e, *o, k| {
        const v = if (k == ends.len) stop else std.math.cast(usize, e) orelse return error.InvalidInput;
        if (v <= prev or v > stop) return error.InvalidInput;
        o.* = @intCast(v - d.offset + row);
        prev = v;
    }
}

/// Copies `src[lo..][0..n]` to the front of `dst` in 16-byte chunks when both sides have room
/// for the rounded-up length (bytes past `n` in `dst` are scratch); else one @memcpy.
inline fn wildCopy(dst: []u8, src: []const u8, lo: usize, n: usize) void {
    const rounded = (n + 15) & ~@as(usize, 15);
    if (n <= 256 and lo + rounded <= src.len and rounded <= dst.len) {
        var k: usize = 0;
        while (k < n) : (k += 16) dst[k..][0..16].* = src[lo + k ..][0..16].*;
    } else @memcpy(dst[0..n], src[lo..][0..n]);
}

/// Appends one string/binary part: offsets `out_offsets[row + 1 ..]` rebased onto `byte_pos`,
/// and its bytes at `out_bytes[byte_pos..]`. Returns the new byte position.
fn copyBinaryPart(comptime T: type, out_offsets: []T, out_bytes: []u8, row: usize, byte_pos: usize, d: *const ArrayData) Error!usize {
    const r = try binaryDataRange(T, d);
    rebaseOffsets(T, out_offsets, row, byte_pos, d);
    wildCopy(out_bytes[byte_pos..], d.buffers[2].data, r.start, r.end - r.start);
    return byte_pos + (r.end - r.start);
}

/// Writes `d`'s offsets rebased onto `pos` to `out[row + 1 ..][0..d.length]` (after offsetsRange).
fn rebaseOffsets(comptime T: type, out: []T, row: usize, pos: usize, d: *const ArrayData) void {
    const src = (d.buffers[1].typedSlice(T) catch unreachable)[d.offset .. d.offset + d.length + 1];
    // Wrapping math: malformed (non-monotonic) input yields offsets that validateLayout rejects.
    const shift: T = @as(T, @intCast(pos)) -% src[0];
    for (src[1..], out[row + 1 ..][0..d.length]) |o, *dst| dst.* = o +% shift;
}

const ByteRange = struct { start: usize, end: usize };

fn binaryDataRange(comptime T: type, d: *const ArrayData) Error!ByteRange {
    try requireBufferCount(d, 3);
    return offsetsRange(T, d, d.buffers[2].len());
}

/// [offsets[offset], offsets[offset + length]) of an offsets layout, checked against `limit`.
fn offsetsRange(comptime T: type, d: *const ArrayData, limit: usize) Error!ByteRange {
    const offsets = d.buffers[1].typedSlice(T) catch return error.InvalidInput;
    const last = std.math.add(usize, d.offset, d.length) catch return error.InvalidInput;
    if (last >= offsets.len) return error.InvalidInput;
    const start = std.math.cast(usize, offsets[d.offset]) orelse return error.InvalidInput;
    const end = std.math.cast(usize, offsets[last]) orelse return error.InvalidInput;
    if (end < start or end > limit) return error.InvalidInput;
    return .{ .start = start, .end = end };
}

/// Writes `data`'s validity bits into zeroed `dst` at bit `dst_off` (all-null parts stay zero).
fn appendValidityBits(dst: []u8, dst_off: usize, data: *const ArrayData) ConcatArrayError!void {
    if (data.length == 0) return;
    if (data.null_count) |count| {
        if (count == data.length) return;
        if (count == 0) return bitmap.setBits(dst, dst_off, data.length);
    }
    // Every caller's layout has a top-level bitmap, so read buffers[0] directly (cheaper than validity()).
    if (data.buffers.len == 0 or data.buffers[0].isEmpty()) return bitmap.setBits(dst, dst_off, data.length);
    const bits = data.buffers[0].data;
    if (bitmap.byteLength(data.offset + data.length) > bits.len) return error.InvalidInput;
    bitmap.copyBits(dst, dst_off, bits, data.offset, data.length);
}

fn dataBytesForFixedWidth(data: *const ArrayData, byte_width: usize) ConcatArrayError![]const u8 {
    const start = std.math.mul(usize, data.offset, byte_width) catch return error.InvalidInput;
    const data_len = std.math.mul(usize, data.length, byte_width) catch return error.InvalidInput;
    const end = std.math.add(usize, start, data_len) catch return error.InvalidInput;
    if (data.buffers.len < 2) return error.InvalidInput;
    if (end > data.buffers[1].len()) return error.InvalidInput;
    return data.buffers[1].data[start..end];
}

fn requireBufferCount(data: *const ArrayData, min_count: usize) ConcatArrayError!void {
    if (data.buffers.len < min_count) return error.InvalidInput;
}

fn requireBitmapBits(buf: array_data.SharedBuffer, bit_len: usize) ConcatArrayError!void {
    if (bitmap.byteLength(bit_len) > buf.len()) return error.InvalidInput;
}

test "concatArrayRefs supports null bool and fixed-width with sliced inputs" {
    const allocator = std.testing.allocator;
    const arr = @import("array/array.zig");

    var null_left_builder = try arr.NullBuilder.init(allocator, 0);
    try null_left_builder.appendNulls(4);
    var null_left = try null_left_builder.finish();
    defer null_left.release();

    var null_right_builder = try arr.NullBuilder.init(allocator, 0);
    try null_right_builder.appendNulls(3);
    var null_right = try null_right_builder.finish();
    defer null_right.release();

    var null_left_slice = try null_left.slice(1, 2);
    defer null_left_slice.release();
    var null_right_slice = try null_right.slice(1, 1);
    defer null_right_slice.release();

    var merged_null = try concatArrayRefs(allocator, .{ .null = {} }, &[_]ArrayRef{ null_left_slice, null_right_slice });
    defer merged_null.release();
    try merged_null.data().validateLayout();
    try std.testing.expectEqual(@as(usize, 3), merged_null.data().length);
    try std.testing.expectEqual(@as(usize, 3), merged_null.data().null_count.?);

    var bool_left_builder = try arr.BooleanBuilder.init(allocator, 4);
    defer bool_left_builder.deinit();
    try bool_left_builder.append(true);
    try bool_left_builder.append(false);
    try bool_left_builder.appendNull();
    try bool_left_builder.append(true);
    var bool_left = try bool_left_builder.finish();
    defer bool_left.release();

    var bool_right_builder = try arr.BooleanBuilder.init(allocator, 3);
    defer bool_right_builder.deinit();
    try bool_right_builder.appendNull();
    try bool_right_builder.append(false);
    try bool_right_builder.append(true);
    var bool_right = try bool_right_builder.finish();
    defer bool_right.release();

    var bool_left_slice = try bool_left.slice(1, 3);
    defer bool_left_slice.release();
    var bool_right_slice = try bool_right.slice(0, 2);
    defer bool_right_slice.release();

    var merged_bool = try concatArrayRefs(allocator, .{ .bool = {} }, &[_]ArrayRef{ bool_left_slice, bool_right_slice });
    defer merged_bool.release();
    try merged_bool.data().validateLayout();

    const bool_view = arr.BooleanArray{ .data = merged_bool.data() };
    try std.testing.expectEqual(@as(usize, 5), bool_view.len());
    try std.testing.expectEqual(false, try bool_view.value(0));
    try std.testing.expect(bool_view.isNull(1));
    try std.testing.expectEqual(true, try bool_view.value(2));
    try std.testing.expect(bool_view.isNull(3));
    try std.testing.expectEqual(false, try bool_view.value(4));

    var i32_left_builder = try arr.Int32Builder.init(allocator, 4);
    defer i32_left_builder.deinit();
    try i32_left_builder.append(1);
    try i32_left_builder.append(2);
    try i32_left_builder.append(3);
    try i32_left_builder.append(4);
    var i32_left = try i32_left_builder.finish();
    defer i32_left.release();

    var i32_right_builder = try arr.Int32Builder.init(allocator, 3);
    defer i32_right_builder.deinit();
    try i32_right_builder.append(5);
    try i32_right_builder.append(6);
    try i32_right_builder.append(7);
    var i32_right = try i32_right_builder.finish();
    defer i32_right.release();

    var i32_left_slice = try i32_left.slice(1, 2);
    defer i32_left_slice.release();
    var i32_right_slice = try i32_right.slice(1, 2);
    defer i32_right_slice.release();

    var merged_i32 = try concatArrayRefs(allocator, .{ .int32 = {} }, &[_]ArrayRef{ i32_left_slice, i32_right_slice });
    defer merged_i32.release();
    try merged_i32.data().validateLayout();

    const i32_view = arr.Int32Array{ .data = merged_i32.data() };
    try std.testing.expectEqual(@as(usize, 4), i32_view.len());
    try std.testing.expectEqual(@as(i32, 2), try i32_view.value(0));
    try std.testing.expectEqual(@as(i32, 3), try i32_view.value(1));
    try std.testing.expectEqual(@as(i32, 6), try i32_view.value(2));
    try std.testing.expectEqual(@as(i32, 7), try i32_view.value(3));
}

test "concatArrayRefs supports string and binary families" {
    const allocator = std.testing.allocator;
    const arr = @import("array/array.zig");

    var str_left_builder = try arr.StringBuilder.init(allocator, 3, 8);
    defer str_left_builder.deinit();
    try str_left_builder.append("aa");
    try str_left_builder.append("bbb");
    try str_left_builder.append("c");
    var str_left = try str_left_builder.finish();
    defer str_left.release();

    var str_right_builder = try arr.StringBuilder.init(allocator, 2, 3);
    defer str_right_builder.deinit();
    try str_right_builder.append("d");
    try str_right_builder.append("ee");
    var str_right = try str_right_builder.finish();
    defer str_right.release();

    var str_left_slice = try str_left.slice(1, 2);
    defer str_left_slice.release();

    var merged_str = try concatArrayRefs(allocator, .{ .string = {} }, &[_]ArrayRef{ str_left_slice, str_right });
    defer merged_str.release();
    try merged_str.data().validateLayout();

    const str_view = arr.StringArray{ .data = merged_str.data() };
    try std.testing.expectEqual(@as(usize, 4), str_view.len());
    try std.testing.expectEqualStrings("bbb", try str_view.value(0));
    try std.testing.expectEqualStrings("c", try str_view.value(1));
    try std.testing.expectEqualStrings("d", try str_view.value(2));
    try std.testing.expectEqualStrings("ee", try str_view.value(3));

    var bin_left_builder = try arr.BinaryBuilder.init(allocator, 3, 8);
    defer bin_left_builder.deinit();
    try bin_left_builder.append("x1");
    try bin_left_builder.append("y22");
    try bin_left_builder.append("z");
    var bin_left = try bin_left_builder.finish();
    defer bin_left.release();

    var bin_right_builder = try arr.BinaryBuilder.init(allocator, 1, 2);
    defer bin_right_builder.deinit();
    try bin_right_builder.append("qq");
    var bin_right = try bin_right_builder.finish();
    defer bin_right.release();

    var bin_left_slice = try bin_left.slice(1, 2);
    defer bin_left_slice.release();

    var merged_bin = try concatArrayRefs(allocator, .{ .binary = {} }, &[_]ArrayRef{ bin_left_slice, bin_right });
    defer merged_bin.release();
    try merged_bin.data().validateLayout();

    const bin_view = arr.BinaryArray{ .data = merged_bin.data() };
    try std.testing.expectEqualStrings("y22", try bin_view.value(0));
    try std.testing.expectEqualStrings("z", try bin_view.value(1));
    try std.testing.expectEqualStrings("qq", try bin_view.value(2));

    var lstr_left_builder = try arr.LargeStringBuilder.init(allocator, 2, 4);
    defer lstr_left_builder.deinit();
    try lstr_left_builder.append("l1");
    try lstr_left_builder.append("l22");
    var lstr_left = try lstr_left_builder.finish();
    defer lstr_left.release();

    var lstr_right_builder = try arr.LargeStringBuilder.init(allocator, 1, 2);
    defer lstr_right_builder.deinit();
    try lstr_right_builder.append("l3");
    var lstr_right = try lstr_right_builder.finish();
    defer lstr_right.release();

    var merged_lstr = try concatArrayRefs(allocator, .{ .large_string = {} }, &[_]ArrayRef{ lstr_left, lstr_right });
    defer merged_lstr.release();
    try merged_lstr.data().validateLayout();

    const lstr_view = arr.LargeStringArray{ .data = merged_lstr.data() };
    try std.testing.expectEqualStrings("l1", try lstr_view.value(0));
    try std.testing.expectEqualStrings("l22", try lstr_view.value(1));
    try std.testing.expectEqualStrings("l3", try lstr_view.value(2));

    var lbin_left_builder = try arr.LargeBinaryBuilder.init(allocator, 2, 4);
    defer lbin_left_builder.deinit();
    try lbin_left_builder.append("ab");
    try lbin_left_builder.append("c");
    var lbin_left = try lbin_left_builder.finish();
    defer lbin_left.release();

    var lbin_right_builder = try arr.LargeBinaryBuilder.init(allocator, 2, 4);
    defer lbin_right_builder.deinit();
    try lbin_right_builder.append("de");
    try lbin_right_builder.append("f");
    var lbin_right = try lbin_right_builder.finish();
    defer lbin_right.release();

    var lbin_right_slice = try lbin_right.slice(0, 1);
    defer lbin_right_slice.release();

    var merged_lbin = try concatArrayRefs(allocator, .{ .large_binary = {} }, &[_]ArrayRef{ lbin_left, lbin_right_slice });
    defer merged_lbin.release();
    try merged_lbin.data().validateLayout();

    const lbin_view = arr.LargeBinaryArray{ .data = merged_lbin.data() };
    try std.testing.expectEqualStrings("ab", try lbin_view.value(0));
    try std.testing.expectEqualStrings("c", try lbin_view.value(1));
    try std.testing.expectEqualStrings("de", try lbin_view.value(2));

    var sv_left_builder = try arr.StringViewBuilder.init(allocator, 3, 32);
    defer sv_left_builder.deinit();
    try sv_left_builder.append("small");
    try sv_left_builder.appendNull();
    try sv_left_builder.append("this string is longer than twelve");
    var sv_left = try sv_left_builder.finish();
    defer sv_left.release();

    var sv_right_builder = try arr.StringViewBuilder.init(allocator, 2, 32);
    defer sv_right_builder.deinit();
    try sv_right_builder.append("tail");
    try sv_right_builder.appendNull();
    var sv_right = try sv_right_builder.finish();
    defer sv_right.release();

    var sv_left_slice = try sv_left.slice(1, 2);
    defer sv_left_slice.release();
    var sv_right_slice = try sv_right.slice(0, 1);
    defer sv_right_slice.release();

    var merged_sv = try concatArrayRefs(allocator, .{ .string_view = {} }, &[_]ArrayRef{ sv_left_slice, sv_right_slice });
    defer merged_sv.release();
    try merged_sv.data().validateLayout();
    const sv_view = arr.StringViewArray{ .data = merged_sv.data() };
    try std.testing.expectEqual(@as(usize, 3), sv_view.len());
    try std.testing.expect(sv_view.isNull(0));
    try std.testing.expectEqualStrings("this string is longer than twelve", try sv_view.value(1));
    try std.testing.expectEqualStrings("tail", try sv_view.value(2));

    var bv_left_builder = try arr.BinaryViewBuilder.init(allocator, 3, 32);
    defer bv_left_builder.deinit();
    try bv_left_builder.append("aa");
    try bv_left_builder.appendNull();
    try bv_left_builder.append("this-binary-view-is-long");
    var bv_left = try bv_left_builder.finish();
    defer bv_left.release();

    var bv_right_builder = try arr.BinaryViewBuilder.init(allocator, 1, 8);
    defer bv_right_builder.deinit();
    try bv_right_builder.append("zz");
    var bv_right = try bv_right_builder.finish();
    defer bv_right.release();

    var bv_left_slice = try bv_left.slice(1, 2);
    defer bv_left_slice.release();
    var merged_bv = try concatArrayRefs(allocator, .{ .binary_view = {} }, &[_]ArrayRef{ bv_left_slice, bv_right });
    defer merged_bv.release();
    try merged_bv.data().validateLayout();
    const bv_view = arr.BinaryViewArray{ .data = merged_bv.data() };
    try std.testing.expectEqual(@as(usize, 3), bv_view.len());
    try std.testing.expect(bv_view.isNull(0));
    try std.testing.expectEqualStrings("this-binary-view-is-long", try bv_view.value(1));
    try std.testing.expectEqualStrings("zz", try bv_view.value(2));
}

test "concatArrayRefs supports list large_list and fixed_size_list" {
    const allocator = std.testing.allocator;
    const arr = @import("array/array.zig");

    const int_type = DataType{ .int32 = {} };
    const value_field = datatype.Field{ .name = "item", .data_type = &int_type, .nullable = true };

    var list_left_values_builder = try arr.Int32Builder.init(allocator, 4);
    defer list_left_values_builder.deinit();
    try list_left_values_builder.append(10);
    try list_left_values_builder.append(11);
    try list_left_values_builder.append(12);
    try list_left_values_builder.append(13);
    var list_left_values = try list_left_values_builder.finish();
    defer list_left_values.release();

    var list_left_builder = try arr.ListBuilder.init(allocator, 3, value_field);
    defer list_left_builder.deinit();
    try list_left_builder.appendLen(1);
    try list_left_builder.appendLen(2);
    try list_left_builder.appendLen(1);
    var list_left = try list_left_builder.finish(list_left_values);
    defer list_left.release();

    var list_right_values_builder = try arr.Int32Builder.init(allocator, 3);
    defer list_right_values_builder.deinit();
    try list_right_values_builder.append(20);
    try list_right_values_builder.append(21);
    try list_right_values_builder.append(22);
    var list_right_values = try list_right_values_builder.finish();
    defer list_right_values.release();

    var list_right_builder = try arr.ListBuilder.init(allocator, 2, value_field);
    defer list_right_builder.deinit();
    try list_right_builder.appendLen(2);
    try list_right_builder.appendLen(1);
    var list_right = try list_right_builder.finish(list_right_values);
    defer list_right.release();

    var list_left_slice = try list_left.slice(1, 2);
    defer list_left_slice.release();
    var list_right_slice = try list_right.slice(0, 1);
    defer list_right_slice.release();

    var merged_list = try concatArrayRefs(allocator, .{ .list = .{ .value_field = value_field } }, &[_]ArrayRef{ list_left_slice, list_right_slice });
    defer merged_list.release();
    try merged_list.data().validateLayout();
    const list_offsets = try merged_list.data().buffers[1].typedSlice(i32);
    try std.testing.expectEqual(@as(i32, 0), list_offsets[0]);
    try std.testing.expectEqual(@as(i32, 2), list_offsets[1]);
    try std.testing.expectEqual(@as(i32, 3), list_offsets[2]);
    try std.testing.expectEqual(@as(i32, 5), list_offsets[3]);

    const list_child = arr.Int32Array{ .data = merged_list.data().children[0].data() };
    try std.testing.expectEqual(@as(i32, 11), try list_child.value(0));
    try std.testing.expectEqual(@as(i32, 12), try list_child.value(1));
    try std.testing.expectEqual(@as(i32, 13), try list_child.value(2));
    try std.testing.expectEqual(@as(i32, 20), try list_child.value(3));
    try std.testing.expectEqual(@as(i32, 21), try list_child.value(4));

    var llist_left_values_builder = try arr.Int32Builder.init(allocator, 3);
    defer llist_left_values_builder.deinit();
    try llist_left_values_builder.append(1);
    try llist_left_values_builder.append(2);
    try llist_left_values_builder.append(3);
    var llist_left_values = try llist_left_values_builder.finish();
    defer llist_left_values.release();

    var llist_left_builder = try arr.LargeListBuilder.init(allocator, 2, value_field);
    defer llist_left_builder.deinit();
    try llist_left_builder.appendLen(2);
    try llist_left_builder.appendLen(1);
    var llist_left = try llist_left_builder.finish(llist_left_values);
    defer llist_left.release();

    var llist_right_values_builder = try arr.Int32Builder.init(allocator, 2);
    defer llist_right_values_builder.deinit();
    try llist_right_values_builder.append(4);
    try llist_right_values_builder.append(5);
    var llist_right_values = try llist_right_values_builder.finish();
    defer llist_right_values.release();

    var llist_right_builder = try arr.LargeListBuilder.init(allocator, 1, value_field);
    defer llist_right_builder.deinit();
    try llist_right_builder.appendLen(2);
    var llist_right = try llist_right_builder.finish(llist_right_values);
    defer llist_right.release();

    var llist_left_slice = try llist_left.slice(1, 1);
    defer llist_left_slice.release();

    var merged_llist = try concatArrayRefs(allocator, .{ .large_list = .{ .value_field = value_field } }, &[_]ArrayRef{ llist_left_slice, llist_right });
    defer merged_llist.release();
    try merged_llist.data().validateLayout();
    const llist_offsets = try merged_llist.data().buffers[1].typedSlice(i64);
    try std.testing.expectEqual(@as(i64, 0), llist_offsets[0]);
    try std.testing.expectEqual(@as(i64, 1), llist_offsets[1]);
    try std.testing.expectEqual(@as(i64, 3), llist_offsets[2]);

    const llist_child = arr.Int32Array{ .data = merged_llist.data().children[0].data() };
    try std.testing.expectEqual(@as(i32, 3), try llist_child.value(0));
    try std.testing.expectEqual(@as(i32, 4), try llist_child.value(1));
    try std.testing.expectEqual(@as(i32, 5), try llist_child.value(2));

    var fsl_left_values_builder = try arr.Int32Builder.init(allocator, 4);
    defer fsl_left_values_builder.deinit();
    try fsl_left_values_builder.append(1);
    try fsl_left_values_builder.append(2);
    try fsl_left_values_builder.append(3);
    try fsl_left_values_builder.append(4);
    var fsl_left_values = try fsl_left_values_builder.finish();
    defer fsl_left_values.release();

    var fsl_left_builder = try arr.FixedSizeListBuilder.init(allocator, value_field, 2);
    defer fsl_left_builder.deinit();
    try fsl_left_builder.appendValid();
    try fsl_left_builder.appendValid();
    var fsl_left = try fsl_left_builder.finish(fsl_left_values);
    defer fsl_left.release();

    var fsl_right_values_builder = try arr.Int32Builder.init(allocator, 4);
    defer fsl_right_values_builder.deinit();
    try fsl_right_values_builder.append(5);
    try fsl_right_values_builder.append(6);
    try fsl_right_values_builder.append(7);
    try fsl_right_values_builder.append(8);
    var fsl_right_values = try fsl_right_values_builder.finish();
    defer fsl_right_values.release();

    var fsl_right_builder = try arr.FixedSizeListBuilder.init(allocator, value_field, 2);
    defer fsl_right_builder.deinit();
    try fsl_right_builder.appendValid();
    try fsl_right_builder.appendValid();
    var fsl_right = try fsl_right_builder.finish(fsl_right_values);
    defer fsl_right.release();

    var fsl_left_slice = try fsl_left.slice(1, 1);
    defer fsl_left_slice.release();
    var fsl_right_slice = try fsl_right.slice(0, 1);
    defer fsl_right_slice.release();

    var merged_fsl = try concatArrayRefs(
        allocator,
        .{ .fixed_size_list = .{ .list_size = 2, .value_field = value_field } },
        &[_]ArrayRef{ fsl_left_slice, fsl_right_slice },
    );
    defer merged_fsl.release();
    try merged_fsl.data().validateLayout();

    const fsl_view = arr.FixedSizeListArray{ .data = merged_fsl.data() };
    try std.testing.expectEqual(@as(usize, 2), fsl_view.len());
    const fsl_child = arr.Int32Array{ .data = (try fsl_view.valuesRef()).data() };
    try std.testing.expectEqual(@as(i32, 3), try fsl_child.value(0));
    try std.testing.expectEqual(@as(i32, 4), try fsl_child.value(1));
    try std.testing.expectEqual(@as(i32, 5), try fsl_child.value(2));
    try std.testing.expectEqual(@as(i32, 6), try fsl_child.value(3));

    var fsl_null_left_values_builder = try arr.Int32Builder.init(allocator, 6);
    defer fsl_null_left_values_builder.deinit();
    try fsl_null_left_values_builder.append(101);
    try fsl_null_left_values_builder.append(102);
    try fsl_null_left_values_builder.append(103);
    try fsl_null_left_values_builder.append(104);
    try fsl_null_left_values_builder.append(105);
    try fsl_null_left_values_builder.append(106);
    var fsl_null_left_values = try fsl_null_left_values_builder.finish();
    defer fsl_null_left_values.release();

    var fsl_null_left_builder = try arr.FixedSizeListBuilder.init(allocator, value_field, 2);
    defer fsl_null_left_builder.deinit();
    try fsl_null_left_builder.appendNull();
    try fsl_null_left_builder.appendValid();
    try fsl_null_left_builder.appendValid();
    var fsl_null_left = try fsl_null_left_builder.finish(fsl_null_left_values);
    defer fsl_null_left.release();

    var fsl_null_right_values_builder = try arr.Int32Builder.init(allocator, 2);
    defer fsl_null_right_values_builder.deinit();
    try fsl_null_right_values_builder.append(201);
    try fsl_null_right_values_builder.append(202);
    var fsl_null_right_values = try fsl_null_right_values_builder.finish();
    defer fsl_null_right_values.release();

    var fsl_null_right_builder = try arr.FixedSizeListBuilder.init(allocator, value_field, 2);
    defer fsl_null_right_builder.deinit();
    try fsl_null_right_builder.appendValid();
    var fsl_null_right = try fsl_null_right_builder.finish(fsl_null_right_values);
    defer fsl_null_right.release();

    var fsl_null_left_slice = try fsl_null_left.slice(1, 1);
    defer fsl_null_left_slice.release();
    var fsl_null_merged = try concatArrayRefs(
        allocator,
        .{ .fixed_size_list = .{ .list_size = 2, .value_field = value_field } },
        &[_]ArrayRef{ fsl_null_left_slice, fsl_null_right },
    );
    defer fsl_null_merged.release();
    try fsl_null_merged.data().validateLayout();
    const fsl_null_view = arr.FixedSizeListArray{ .data = fsl_null_merged.data() };
    try std.testing.expectEqual(@as(usize, 2), fsl_null_view.len());
    try std.testing.expect(!fsl_null_view.isNull(0));
    try std.testing.expect(!fsl_null_view.isNull(1));
    try std.testing.expectEqual(@as(usize, 0), fsl_null_merged.data().null_count.?);
}

test "concatArrayRefs supports recursive struct" {
    const allocator = std.testing.allocator;
    const arr = @import("array/array.zig");

    const int_type = DataType{ .int32 = {} };
    const list_value_field = datatype.Field{ .name = "item", .data_type = &int_type, .nullable = true };
    const list_type = DataType{ .list = .{ .value_field = list_value_field } };

    const struct_fields = [_]datatype.Field{
        .{ .name = "id", .data_type = &int_type, .nullable = false },
        .{ .name = "nums", .data_type = &list_type, .nullable = true },
    };

    var left_ids_builder = try arr.Int32Builder.init(allocator, 3);
    defer left_ids_builder.deinit();
    try left_ids_builder.append(10);
    try left_ids_builder.append(20);
    try left_ids_builder.append(30);
    var left_ids = try left_ids_builder.finish();
    defer left_ids.release();

    var left_vals_builder = try arr.Int32Builder.init(allocator, 4);
    defer left_vals_builder.deinit();
    try left_vals_builder.append(1);
    try left_vals_builder.append(2);
    try left_vals_builder.append(3);
    try left_vals_builder.append(4);
    var left_vals = try left_vals_builder.finish();
    defer left_vals.release();

    var left_list_builder = try arr.ListBuilder.init(allocator, 3, list_value_field);
    defer left_list_builder.deinit();
    try left_list_builder.appendLen(1);
    try left_list_builder.appendLen(2);
    try left_list_builder.appendLen(1);
    var left_list = try left_list_builder.finish(left_vals);
    defer left_list.release();

    var left_struct_builder = arr.StructBuilder.init(allocator, struct_fields[0..]);
    defer left_struct_builder.deinit();
    try left_struct_builder.appendValid();
    try left_struct_builder.appendValid();
    try left_struct_builder.appendValid();
    var left_struct = try left_struct_builder.finish(&[_]ArrayRef{ left_ids, left_list });
    defer left_struct.release();

    var right_ids_builder = try arr.Int32Builder.init(allocator, 2);
    defer right_ids_builder.deinit();
    try right_ids_builder.append(40);
    try right_ids_builder.append(50);
    var right_ids = try right_ids_builder.finish();
    defer right_ids.release();

    var right_vals_builder = try arr.Int32Builder.init(allocator, 3);
    defer right_vals_builder.deinit();
    try right_vals_builder.append(5);
    try right_vals_builder.append(6);
    try right_vals_builder.append(7);
    var right_vals = try right_vals_builder.finish();
    defer right_vals.release();

    var right_list_builder = try arr.ListBuilder.init(allocator, 2, list_value_field);
    defer right_list_builder.deinit();
    try right_list_builder.appendLen(2);
    try right_list_builder.appendLen(1);
    var right_list = try right_list_builder.finish(right_vals);
    defer right_list.release();

    var right_struct_builder = arr.StructBuilder.init(allocator, struct_fields[0..]);
    defer right_struct_builder.deinit();
    try right_struct_builder.appendValid();
    try right_struct_builder.appendValid();
    var right_struct = try right_struct_builder.finish(&[_]ArrayRef{ right_ids, right_list });
    defer right_struct.release();

    var left_struct_slice = try left_struct.slice(1, 2);
    defer left_struct_slice.release();
    var right_struct_slice = try right_struct.slice(0, 1);
    defer right_struct_slice.release();

    var merged = try concatArrayRefs(
        allocator,
        .{ .struct_ = .{ .fields = struct_fields[0..] } },
        &[_]ArrayRef{ left_struct_slice, right_struct_slice },
    );
    defer merged.release();
    try merged.data().validateLayout();

    try std.testing.expectEqual(@as(usize, 3), merged.data().length);
    const ids = arr.Int32Array{ .data = merged.data().children[0].data() };
    try std.testing.expectEqual(@as(i32, 20), try ids.value(0));
    try std.testing.expectEqual(@as(i32, 30), try ids.value(1));
    try std.testing.expectEqual(@as(i32, 40), try ids.value(2));

    const nums = merged.data().children[1];
    const nums_offsets = try nums.data().buffers[1].typedSlice(i32);
    try std.testing.expectEqual(@as(i32, 0), nums_offsets[0]);
    try std.testing.expectEqual(@as(i32, 2), nums_offsets[1]);
    try std.testing.expectEqual(@as(i32, 3), nums_offsets[2]);
    try std.testing.expectEqual(@as(i32, 5), nums_offsets[3]);

    const nums_values = arr.Int32Array{ .data = nums.data().children[0].data() };
    try std.testing.expectEqual(@as(i32, 2), try nums_values.value(0));
    try std.testing.expectEqual(@as(i32, 3), try nums_values.value(1));
    try std.testing.expectEqual(@as(i32, 4), try nums_values.value(2));
    try std.testing.expectEqual(@as(i32, 5), try nums_values.value(3));
    try std.testing.expectEqual(@as(i32, 6), try nums_values.value(4));
}

test "concatArrayRefs rejects empty input" {
    try std.testing.expectError(
        error.InvalidInput,
        concatArrayRefs(std.testing.allocator, .{ .int32 = {} }, &[_]ArrayRef{}),
    );
}

test "concatArrayRefs merges many sliced parts in order" {
    const allocator = std.testing.allocator;
    var builder = try array_mod.Int32Builder.init(allocator, 16);
    defer builder.deinit();
    for (0..16) |i| {
        if (i % 3 == 0) try builder.appendNull() else try builder.append(@intCast(i));
    }
    var full = try builder.finish();
    defer full.release();

    // 7 parts (odd count exercises the retained trailing ref), all sliced from `full`.
    var parts: [7]ArrayRef = undefined;
    for (&parts, 0..) |*part, i| part.* = try full.slice(i * 2, 2);
    defer for (&parts) |*part| part.release();

    var out = try concatArrayRefs(allocator, DataType{ .int32 = {} }, parts[0..]);
    defer out.release();
    const view = array_mod.Int32Array{ .data = out.data() };
    try std.testing.expectEqual(@as(usize, 14), view.len());
    for (0..14) |i| {
        try std.testing.expectEqual(i % 3 == 0, view.isNull(i));
        if (i % 3 != 0) try std.testing.expectEqual(@as(i32, @intCast(i)), try view.value(i));
    }
}

test "concatArrayRefs N-ary path matches per-row values for int32, string and bool parts" {
    const allocator = std.testing.allocator;

    var ib = try array_mod.Int32Builder.init(allocator, 0);
    defer ib.deinit();
    var sb = try array_mod.StringBuilder.init(allocator, 0, 0);
    defer sb.deinit();
    var bb = try array_mod.BooleanBuilder.init(allocator, 0);
    defer bb.deinit();
    const n = 67;
    for (0..n) |i| {
        if (i % 5 == 2) {
            try ib.appendNull();
            try sb.appendNull();
            try bb.appendNull();
        } else {
            try ib.append(@intCast(i));
            try sb.append(if (i % 3 == 0) "xy" else "z");
            try bb.append(i % 4 == 1);
        }
    }
    var ints = try ib.finish();
    defer ints.release();
    var strs = try sb.finish();
    defer strs.release();
    var bools = try bb.finish();
    defer bools.release();

    // Uneven cuts so parts start at every bit phase; part [0, 2) has no nulls.
    const cuts = [_]usize{ 0, 2, 9, 10, 11, 30, 33, 34, 50, 67 };
    const sources = [_]ArrayRef{ ints, strs, bools };
    for (sources) |src| {
        var parts: [cuts.len - 1]ArrayRef = undefined;
        for (&parts, 0..) |*p, k| p.* = try src.slice(cuts[k], cuts[k + 1] - cuts[k]);
        defer for (&parts) |*p| p.release();

        var out = try concatArrayRefs(allocator, src.data().data_type, parts[0..]);
        defer out.release();
        try std.testing.expectEqual(@as(usize, n), out.data().length);
        try std.testing.expectEqual(@as(?usize, 13), out.data().null_count);
        for (0..n) |i| {
            try std.testing.expectEqual(src.data().isNull(i), out.data().isNull(i));
            if (out.data().isNull(i)) continue;
            switch (src.data().data_type) {
                .int32 => try std.testing.expectEqual(
                    try (array_mod.Int32Array{ .data = src.data() }).value(i),
                    try (array_mod.Int32Array{ .data = out.data() }).value(i),
                ),
                .string => try std.testing.expectEqualStrings(
                    try (array_mod.StringArray{ .data = src.data() }).value(i),
                    try (array_mod.StringArray{ .data = out.data() }).value(i),
                ),
                .bool => try std.testing.expectEqual(
                    try (array_mod.BooleanArray{ .data = src.data() }).value(i),
                    try (array_mod.BooleanArray{ .data = out.data() }).value(i),
                ),
                else => unreachable,
            }
        }
    }
}

test "concatArrayRefs rejects parts whose type differs in tag or parameters" {
    const allocator = std.testing.allocator;
    const values = [_]i64{ 1, 2 };
    const bufs = [_]array_data.SharedBuffer{ array_data.SharedBuffer.empty, array_data.SharedBuffer.fromSlice(std.mem.sliceAsBytes(values[0..])) };
    const sec = DataType{ .timestamp = .{ .unit = .second, .timezone = null } };
    const ms = DataType{ .timestamp = .{ .unit = .millisecond, .timezone = null } };
    var a = try ArrayRef.fromBorrowed(allocator, .{ .data_type = sec, .length = 2, .null_count = 0, .buffers = bufs[0..] });
    defer a.release();
    var b = try ArrayRef.fromBorrowed(allocator, .{ .data_type = ms, .length = 2, .null_count = 0, .buffers = bufs[0..] });
    defer b.release();
    var c = try ArrayRef.fromBorrowed(allocator, .{ .data_type = .{ .int64 = {} }, .length = 2, .null_count = 0, .buffers = bufs[0..] });
    defer c.release();
    // Same tag, different unit (full dataTypeEql) and different tag (tag compare) are both rejected.
    try std.testing.expectError(error.InvalidInput, concatArrayRefs(allocator, sec, &[_]ArrayRef{ a, b }));
    try std.testing.expectError(error.InvalidInput, concatArrayRefs(allocator, .{ .int64 = {} }, &[_]ArrayRef{ c, a }));
    var ok = try concatArrayRefs(allocator, sec, &[_]ArrayRef{ a, a });
    defer ok.release();
    try std.testing.expectEqual(@as(usize, 4), ok.data().length);
}
