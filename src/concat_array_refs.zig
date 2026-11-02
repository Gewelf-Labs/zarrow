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

pub fn concatArrayRefs(
    allocator: std.mem.Allocator,
    data_type: DataType,
    parts: []const ArrayRef,
) Error!ArrayRef {
    if (parts.len == 0) return error.InvalidInput;

    // Flat layouts: two passes over `parts`, each output buffer allocated once; checks part types itself.
    if (try concatManyFlat(allocator, data_type, parts)) |out| return out;

    // Nested layouts check the part types themselves, in the pass that reads the parts.
    if (isNestedLayout(data_type)) return concatManyNested(allocator, data_type, parts);

    for (parts) |part| {
        if (!datatype.dataTypeEql(part.data().data_type, data_type)) return error.InvalidInput;
    }

    if (parts.len == 1) {
        // Single part still returns a fresh, compacted copy (offset 0), as before.
        var empty = try parts[0].slice(0, 0);
        defer empty.release();
        return concatTwoArrayRefs(allocator, empty, parts[0]);
    }

    // View types: balanced pairwise merge, O(total_len * log(parts.len)).
    var level = try mergePairs(allocator, parts);
    while (level.items.len > 1) {
        const next = mergePairs(allocator, level.items) catch |err| {
            releaseAll(allocator, &level);
            return err;
        };
        releaseAll(allocator, &level);
        level = next;
    }
    const out = level.items[0];
    level.deinit(allocator);
    return out;
}

/// Concatenates `refs` two at a time; an odd trailing ref is retained as-is.
fn mergePairs(allocator: std.mem.Allocator, refs: []const ArrayRef) Error!std.ArrayList(ArrayRef) {
    var out = try std.ArrayList(ArrayRef).initCapacity(allocator, (refs.len + 1) / 2);
    errdefer releaseAll(allocator, &out);
    var i: usize = 0;
    while (i + 1 < refs.len) : (i += 2) {
        out.appendAssumeCapacity(try concatTwoArrayRefs(allocator, refs[i], refs[i + 1]));
    }
    if (i < refs.len) out.appendAssumeCapacity(refs[i].retain());
    return out;
}

fn releaseAll(allocator: std.mem.Allocator, list: *std.ArrayList(ArrayRef)) void {
    for (list.items) |*ref| ref.release();
    list.deinit(allocator);
}

fn concatTwoArrayRefs(
    allocator: std.mem.Allocator,
    left: ArrayRef,
    right: ArrayRef,
) Error!ArrayRef {
    if (!datatype.dataTypeEql(left.data().data_type, right.data().data_type)) return error.InvalidInput;

    if (try concatManyFlat(allocator, left.data().data_type, &[_]ArrayRef{ left, right })) |out| return out;

    const dt = left.data().data_type;
    const layout_dt = storageDataType(dt);
    var merged_storage = try switch (layout_dt) {
        .string_view => try concatStringViewArray(allocator, left.data(), right.data()),
        .binary_view => try concatBinaryViewArray(allocator, left.data(), right.data()),
        else => error.UnsupportedType,
    };

    if (datatype.dataTypeEql(dt, layout_dt)) return merged_storage;
    const retagged = try retagArrayRefDataType(allocator, merged_storage, dt);
    merged_storage.release();
    return retagged;
}

pub fn storageDataType(dt: DataType) DataType {
    return switch (dt) {
        .extension => |ext| storageDataType(ext.storage_type.*),
        else => dt,
    };
}

fn retagArrayRefDataType(allocator: std.mem.Allocator, src: ArrayRef, out_dt: DataType) error{OutOfMemory}!ArrayRef {
    var out = src.data().*;
    out.data_type = out_dt;
    return ArrayRef.fromBorrowed(allocator, out);
}

fn viewDataCapacityHint(data: *const ArrayData) usize {
    if (data.buffers.len >= 3) return data.buffers[2].len();
    return 0;
}

fn concatStringViewArray(
    allocator: std.mem.Allocator,
    left: *const ArrayData,
    right: *const ArrayData,
) Error!ArrayRef {
    try left.validateLayout();
    try right.validateLayout();
    const total_len = std.math.add(usize, left.length, right.length) catch return error.InvalidInput;
    const data_capacity = std.math.add(usize, viewDataCapacityHint(left), viewDataCapacityHint(right)) catch return error.InvalidInput;

    var builder = try array_mod.StringViewBuilder.init(allocator, total_len, data_capacity);
    defer builder.deinit();

    const left_view = array_mod.StringViewArray{ .data = left };
    var i: usize = 0;
    while (i < left.length) : (i += 1) {
        if (left_view.isNull(i)) {
            builder.appendNull() catch |err| return mapViewBuilderError(err);
        } else {
            const value = left_view.value(i) catch return error.InvalidInput;
            builder.append(value) catch |err| return mapViewBuilderError(err);
        }
    }

    const right_view = array_mod.StringViewArray{ .data = right };
    i = 0;
    while (i < right.length) : (i += 1) {
        if (right_view.isNull(i)) {
            builder.appendNull() catch |err| return mapViewBuilderError(err);
        } else {
            const value = right_view.value(i) catch return error.InvalidInput;
            builder.append(value) catch |err| return mapViewBuilderError(err);
        }
    }

    var out = builder.finish() catch |err| return mapViewBuilderError(err);
    errdefer out.release();
    try out.data().validateLayout();
    return out;
}

fn concatBinaryViewArray(
    allocator: std.mem.Allocator,
    left: *const ArrayData,
    right: *const ArrayData,
) Error!ArrayRef {
    try left.validateLayout();
    try right.validateLayout();
    const total_len = std.math.add(usize, left.length, right.length) catch return error.InvalidInput;
    const data_capacity = std.math.add(usize, viewDataCapacityHint(left), viewDataCapacityHint(right)) catch return error.InvalidInput;

    var builder = try array_mod.BinaryViewBuilder.init(allocator, total_len, data_capacity);
    defer builder.deinit();

    const left_view = array_mod.BinaryViewArray{ .data = left };
    var i: usize = 0;
    while (i < left.length) : (i += 1) {
        if (left_view.isNull(i)) {
            builder.appendNull() catch |err| return mapViewBuilderError(err);
        } else {
            const value = left_view.value(i) catch return error.InvalidInput;
            builder.append(value) catch |err| return mapViewBuilderError(err);
        }
    }

    const right_view = array_mod.BinaryViewArray{ .data = right };
    i = 0;
    while (i < right.length) : (i += 1) {
        if (right_view.isNull(i)) {
            builder.appendNull() catch |err| return mapViewBuilderError(err);
        } else {
            const value = right_view.value(i) catch return error.InvalidInput;
            builder.append(value) catch |err| return mapViewBuilderError(err);
        }
    }

    var out = builder.finish() catch |err| return mapViewBuilderError(err);
    errdefer out.release();
    try out.data().validateLayout();
    return out;
}

fn mapViewBuilderError(err: anyerror) ConcatArrayError {
    return switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        else => error.InvalidInput,
    };
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

fn isNestedLayout(dt: DataType) bool {
    return switch (storageDataType(dt)) {
        .list, .large_list, .map, .fixed_size_list, .struct_ => true,
        else => false,
    };
}

/// Concat of windows: flat and nested layouts N-ary, other layouts through slices + concatArrayRefs.
fn concatParts(allocator: std.mem.Allocator, dt: DataType, parts: []const Part) Error!ArrayRef {
    if (try concatManyFlat(allocator, dt, parts)) |out| return out;
    if (isNestedLayout(dt)) return concatManyNested(allocator, dt, parts);
    const refs = try allocator.alloc(ArrayRef, parts.len);
    var made: usize = 0;
    defer {
        for (refs[0..made]) |*ref| ref.release();
        allocator.free(refs);
    }
    for (parts, refs) |*part, *ref| {
        ref.* = try part.ref.slice(part.start, part.len);
        made += 1;
    }
    return concatArrayRefs(allocator, dt, refs);
}

/// N-ary concat for list/large_list/map/fixed_size_list/struct (`parts`: ArrayRefs or Parts).
/// Pass 1 checks the parts and records every child's window, pass 2 writes validity and rebased
/// list offsets, then each child is concatenated once from its windows.
fn concatManyNested(allocator: std.mem.Allocator, dt: DataType, parts: anytype) Error!ArrayRef {
    const layout = storageDataType(dt);
    const n_children: usize = if (layout == .struct_) layout.struct_.fields.len else 1;
    const offset_width: usize = switch (layout) {
        .list, .map => @sizeOf(i32),
        .large_list => @sizeOf(i64),
        else => 0,
    };
    const max_child: usize = if (offset_width == 4) std.math.maxInt(i32) else std.math.maxInt(i64);
    const buf_count: usize = if (offset_width > 0) 2 else 1;
    const n = parts.len;

    // windows[c * n + p]: the rows of child c that part p covers.
    const windows = try allocator.alloc(Part, n_children * n);
    defer allocator.free(windows);
    var total_len: usize = 0;
    var maybe_nulls = false;
    var view: ArrayData = undefined;
    for (parts, 0..) |*part, p| {
        const d = partData(part, &view);
        // std.meta.eql first: copies of one type (parts sliced from one array) share its pointers.
        if (!std.meta.eql(d.data_type, dt) and !datatype.dataTypeEql(d.data_type, dt)) return error.InvalidInput;
        if (d.buffers.len < buf_count or d.children.len != n_children) return error.InvalidInput;
        total_len = std.math.add(usize, total_len, d.length) catch return error.InvalidInput;
        maybe_nulls = maybe_nulls or ((d.null_count orelse 1) != 0 and !d.buffers[0].isEmpty());
        for (0..n_children) |c| windows[c * n + p] = try childWindow(layout, part, d, c);
    }

    const used = bitmap.byteLength(total_len);
    var validity = try buffer.OwnedBuffer.init(allocator, if (maybe_nulls) used else 0);
    errdefer validity.deinit();
    const offsets_len = if (offset_width > 0) (total_len + 1) * offset_width else 0;
    var offsets = try buffer.OwnedBuffer.initUninit(allocator, offsets_len);
    errdefer offsets.deinit();
    if (maybe_nulls or offset_width > 0) {
        var row: usize = 0;
        var pos: usize = 0;
        for (parts, 0..) |*part, p| {
            const d = partData(part, &view);
            if (maybe_nulls) try appendValidityBits(validity.data, row, d);
            if (offset_width > 0) { // lists: one child, windows[p] is its range
                if (windows[p].len > max_child - pos) return error.InvalidInput;
                if (offset_width == 4) rebaseOffsets(i32, offsets.typedSlice(i32), row, pos, d) else rebaseOffsets(i64, offsets.typedSlice(i64), row, pos, d);
                pos += windows[p].len;
            }
            row += d.length;
        }
    }
    switch (offset_width) {
        4 => offsets.typedSlice(i32)[0] = 0,
        8 => offsets.typedSlice(i64)[0] = 0,
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
        const child_windows = windows[c * n ..][0..n];
        child.* = try concatParts(allocator, child_windows[0].ref.data().data_type, child_windows);
        built += 1;
    }

    var bufs: [2]array_data.SharedBuffer = .{ array_data.SharedBuffer.empty, array_data.SharedBuffer.empty };
    errdefer for (&bufs) |*b| b.release();
    if (null_count == 0) validity.deinit() else bufs[0] = try validity.toShared(used);
    if (offset_width > 0) bufs[1] = try offsets.toShared(offsets_len) else offsets.deinit();

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

/// Rows of child `c` that `part` (rows `d`) covers: the list offsets range, the fixed_size_list
/// rows times list_size, or the struct rows (row-aligned or parent coordinates).
fn childWindow(layout: DataType, part: anytype, d: *const ArrayData, c: usize) Error!Part {
    const child = d.children[c];
    const cd = child.data();
    const r: ByteRange = switch (layout) {
        .list, .map => try offsetsRange(i32, d, cd.length),
        .large_list => try offsetsRange(i64, d, cd.length),
        .fixed_size_list => |f| blk: {
            const size = std.math.cast(usize, f.list_size) orelse return error.InvalidInput;
            const end = std.math.add(usize, d.offset, d.length) catch return error.InvalidInput;
            break :blk .{
                .start = std.math.mul(usize, d.offset, size) catch return error.InvalidInput,
                .end = std.math.mul(usize, end, size) catch return error.InvalidInput,
            };
        },
        .struct_ => blk: {
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
