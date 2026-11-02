const std = @import("std");
const array_mod = @import("../array/array.zig");
const bitmap = @import("../bitmap.zig");
const buffer = @import("../buffer.zig");
const chunked_array_mod = @import("../chunked_array.zig");
const concat_array_refs = @import("../concat_array_refs.zig");
const core = @import("core.zig");
const iterators = @import("iterators.zig");
const accessors = @import("datum_accessors.zig");
const builders = @import("datum_builders.zig");
const common = @import("datum_common.zig");

const DataType = core.DataType;
const storageDataType = concat_array_refs.storageDataType;
const ArrayRef = core.ArrayRef;
const ChunkedArray = core.ChunkedArray;
const Datum = core.Datum;
const KernelError = core.KernelError;
const FilterOptions = core.FilterOptions;

const inferBinaryExecLen = iterators.inferBinaryExecLen;
const inferNaryExecLen = iterators.inferNaryExecLen;
const sameDataTypes = iterators.sameDataTypes;

const mapArrayReadError = common.mapArrayReadError;

const ChunkLookup = struct {
    chunk: *const ArrayRef,
    local_index: usize,
};

fn lookupChunkAt(chunks: ChunkedArray, logical_index: usize) ?ChunkLookup {
    if (logical_index >= chunks.len()) return null;

    var remaining = logical_index;
    var chunk_index: usize = 0;
    while (chunk_index < chunks.numChunks()) : (chunk_index += 1) {
        const chunk_ref = chunks.chunk(chunk_index);
        const chunk_len = chunk_ref.data().length;
        if (remaining < chunk_len) {
            return .{
                .chunk = chunk_ref,
                .local_index = remaining,
            };
        }
        remaining -= chunk_len;
    }
    return null;
}

const PredicateDecision = enum {
    keep,
    drop,
    emit_null,
};

fn datumTakeChunkedNullable(
    allocator: std.mem.Allocator,
    chunks: ChunkedArray,
    out_type: DataType,
    indices: anytype,
) KernelError!Datum {
    if (indices.len == 0) {
        return Datum.fromChunked(chunks.slice(allocator, 0, 0) catch |err| return common.mapChunkedError(err));
    }
    var pieces: std.ArrayList(ArrayRef) = .empty;
    defer releasePieces(allocator, &pieces);
    try takeChunked(allocator, chunks, out_type, indices, &pieces);
    return Datum.fromChunked(ChunkedArray.init(allocator, out_type, pieces.items) catch |err| return common.mapChunkedError(err));
}

fn releasePieces(allocator: std.mem.Allocator, pieces: *std.ArrayList(ArrayRef)) void {
    for (pieces.items) |*piece| piece.release();
    pieces.deinit(allocator);
}

/// Chunk holding logical row `idx` (< total); tries `hint` first so sorted indices stay O(1).
fn chunkOf(starts: []const usize, hint: usize, idx: usize) usize {
    if (starts[hint] <= idx and idx < starts[hint + 1]) return hint;
    var lo: usize = 0;
    var hi: usize = starts.len - 1;
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        if (starts[mid + 1] <= idx) lo = mid + 1 else hi = mid;
    }
    return lo;
}

/// Take from a chunked datum, appending the output pieces in order. Indices are split into runs
/// that stay inside one chunk: few runs (filters, sorted or clustered takes) take each run from
/// its chunk with the array kernels; many runs (random takes) concat the chunks once and take
/// from that, unless the chunks do not concat (e.g. different dictionaries).
fn takeChunked(
    allocator: std.mem.Allocator,
    chunks: ChunkedArray,
    out_type: DataType,
    indices: anytype,
    pieces: *std.ArrayList(ArrayRef),
) KernelError!void {
    const n = chunks.numChunks();
    if (n == 0 or !hasTakeKernel(out_type)) {
        // Per-row path: one 1-row slice (or null row) per index.
        try pieces.ensureUnusedCapacity(allocator, indices.len);
        for (indices) |m| pieces.appendAssumeCapacity(if (@as(?usize, m)) |idx|
            try accessors.datumElementArrayAt(allocator, Datum.fromChunked(chunks), idx)
        else
            try builders.buildNullLikeArray(allocator, out_type, 1));
        return;
    }
    const starts = allocator.alloc(usize, n + 1) catch return error.OutOfMemory;
    defer allocator.free(starts);
    starts[0] = 0;
    for (chunks.chunks(), 0..) |c, i| starts[i + 1] = std.math.add(usize, starts[i], c.data().length) catch return error.Overflow;
    try checkIndices(indices, starts[n]);

    // ponytail: 64 ~ cost of one kernel call per run vs. one concat row; retune if mixed shapes get hot.
    const max_runs = starts[n] / 64;
    var runs: usize = 1;
    var cur: usize = 0;
    var have = false; // the current run holds a non-null index
    for (indices) |m| {
        const c = chunkOf(starts, cur, @as(?usize, m) orelse continue);
        if (have and c != cur) runs += 1;
        if (runs > max_runs) break;
        cur = c;
        have = true;
    }
    if (n > 1 and runs > max_runs) concat_first: {
        var flat = concat_array_refs.concatArrayRefs(allocator, out_type, chunks.chunks()) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            else => break :concat_first,
        };
        defer flat.release();
        return appendPiece(allocator, pieces, try takeChild(allocator, flat, indices));
    }

    const E = @typeInfo(@TypeOf(indices)).pointer.child;
    const local = allocator.alloc(E, indices.len) catch return error.OutOfMemory;
    defer allocator.free(local);
    var run_start: usize = 0;
    cur = 0;
    have = false;
    for (indices, local, 0..) |m, *l, i| {
        if (@as(?usize, m)) |idx| {
            const c = chunkOf(starts, cur, idx);
            if (have and c != cur) {
                try appendPiece(allocator, pieces, try takeChild(allocator, chunks.chunk(cur).*, @as([]const E, local[run_start..i])));
                run_start = i;
            }
            cur = c;
            have = true;
            l.* = if (comptime E == u32) @intCast(idx - starts[c]) else idx - starts[c];
        } else l.* = m;
    }
    try appendPiece(allocator, pieces, try takeChild(allocator, chunks.chunk(cur).*, @as([]const E, local[run_start..])));
}

fn appendPiece(allocator: std.mem.Allocator, pieces: *std.ArrayList(ArrayRef), piece: ArrayRef) KernelError!void {
    pieces.append(allocator, piece) catch {
        var owned = piece;
        owned.release();
        return error.OutOfMemory;
    };
}

/// Takes `indices` from `child` with the array kernels (or the per-row fallback).
fn takeChild(allocator: std.mem.Allocator, child: ArrayRef, indices: anytype) KernelError!ArrayRef {
    var taken = try datumTakeArrayLikeNullable(allocator, Datum.fromArray(child), child.data().data_type, indices);
    if (taken != .array) {
        taken.release();
        return error.InvalidInput;
    }
    return taken.array;
}

/// Direct gather for fixed-width and boolean arrays: output buffers are allocated once, every
/// slot copies its value without a branch, and validity is packed 8 rows per byte (gatherBits).
/// `indices` is `[]const ?usize` (null = null row) or dense `[]const usize` / `[]const u32`.
/// Dictionary arrays gather their indices and share the dictionary.
/// Returns null for layouts it does not handle so callers fall back to slice+concat.
fn takeFixedWidthArray(
    allocator: std.mem.Allocator,
    src: *const array_mod.ArrayData,
    indices: anytype,
) KernelError!?ArrayRef {
    const layout = storageDataType(src.data_type);
    const byte_width = (try fixedByteWidth(layout)) orelse return null;
    if (src.buffers.len != 2) return null;
    const is_bool = layout == .bool;
    const src_bytes = src.buffers[1].data;
    const src_end = std.math.add(usize, src.offset, src.length) catch return error.Overflow;
    const src_need = if (is_bool) bitmap.byteLength(src_end) else std.math.mul(usize, src_end, byte_width) catch return error.Overflow;
    if (src_bytes.len < src_need) return error.InvalidInput;
    const src_validity = try validityBytes(src, src_end);
    try checkIndices(indices, src.length);

    const out_len = indices.len;
    const values_len = if (is_bool) bitmap.byteLength(out_len) else out_len * byte_width;
    // Bool values are OR-ed bits (need zeroed memory); fixed-width slots are all written below.
    var values = (if (is_bool) buffer.OwnedBuffer.init(allocator, values_len) else buffer.OwnedBuffer.initUninit(allocator, values_len)) catch return error.OutOfMemory;
    errdefer values.deinit();
    var validity = buffer.OwnedBuffer.init(allocator, bitmap.byteLength(out_len)) catch return error.OutOfMemory;
    errdefer validity.deinit();

    const null_count = out_len - gatherBits(validity.data, src_validity, src.offset, indices);
    if (is_bool) {
        _ = gatherBits(values.data, src_bytes, src.offset, indices);
    } else switch (byte_width) {
        inline 1, 2, 4, 8, 16, 32 => |w| {
            const E = [w]u8;
            const s = std.mem.bytesAsSlice(E, src_bytes[src.offset * w ..][0 .. src.length * w]);
            const d = std.mem.bytesAsSlice(E, values.data[0..values_len]);
            for (indices, d) |m, *o| o.* = if (@as(?usize, m)) |j| s[j] else @splat(0);
        },
        else => for (indices, 0..) |m, i| {
            const o = values.data[i * byte_width ..][0..byte_width];
            if (@as(?usize, m)) |j| @memcpy(o, src_bytes[(src.offset + j) * byte_width ..][0..byte_width]) else @memset(o, 0);
        },
    }

    var buffers: [2]buffer.SharedBuffer = undefined;
    buffers[0] = if (null_count == 0) blk: {
        validity.deinit();
        break :blk buffer.SharedBuffer.empty;
    } else validity.toShared(bitmap.byteLength(out_len)) catch return error.OutOfMemory;
    errdefer buffers[0].release();
    buffers[1] = values.toShared(values_len) catch return error.OutOfMemory;
    errdefer buffers[1].release();

    const out = ArrayRef.fromOwnedInlineBuffers(allocator, .{
        .data_type = src.data_type,
        .length = out_len,
        .null_count = null_count,
        .buffers = buffers[0..],
        .dictionary = src.dictionary,
    }) catch return error.OutOfMemory;
    if (src.dictionary) |dict| _ = dict.retain();
    return out;
}

/// Value width of a fixed-width layout (0 for bool, index width for dictionaries), else null.
fn fixedByteWidth(layout: DataType) KernelError!?usize {
    const width: usize = switch (layout) {
        .bool => return 0,
        .uint8, .int8 => 1,
        .uint16, .int16, .half_float => 2,
        .uint32, .int32, .float, .date32, .time32, .interval_months, .decimal32 => 4,
        .uint64, .int64, .double, .date64, .time64, .timestamp, .duration, .interval_day_time, .decimal64 => 8,
        .decimal128, .interval_month_day_nano => 16,
        .decimal256 => 32,
        .fixed_size_binary => |f| std.math.cast(usize, f.byte_width) orelse return error.InvalidInput,
        .dictionary => |d| d.index_type.bit_width / 8,
        else => return null,
    };
    return if (width == 0) null else width; // zero-width: per-row path, as before
}

/// The source validity bytes (null when there is no bitmap), checked to cover `src_end` bits.
fn validityBytes(src: *const array_mod.ArrayData, src_end: usize) KernelError!?[]const u8 {
    const v = src.validity() orelse return null;
    if (v.data.len < bitmap.byteLength(src_end)) return error.InvalidInput;
    return v.data;
}

/// Index slices without null entries (`[]const usize`, `[]const u32`).
fn isDense(comptime I: type) bool {
    return I == []const usize or I == []const u32;
}

/// Every non-null index must be < len. Dense slices use a max-reduce, which vectorizes.
fn checkIndices(indices: anytype, len: usize) KernelError!void {
    const I = @TypeOf(indices);
    if (comptime isDense(I)) {
        if (indices.len > 0 and std.mem.max(@typeInfo(I).pointer.child, indices) >= len) return error.InvalidInput;
    } else for (indices) |maybe| if (maybe) |idx| {
        if (idx >= len) return error.InvalidInput;
    };
}

/// Bit i of zeroed `dst` = indices[i] is non-null and bit `offset + indices[i]` of `src` is set
/// (`src == null` means all set). 8 indices per output byte, no data-dependent branch
/// (arrow-rs take_bits). Returns the number of set bits. Callers bounds-check indices and `src`.
fn gatherBits(dst: []u8, src: ?[]const u8, offset: usize, indices: anytype) usize {
    var set: usize = 0;
    var i: usize = 0;
    while (i + 8 <= indices.len) : (i += 8) {
        var byte: u8 = 0;
        inline for (0..8) |k| byte |= bitOf(src, offset, indices[i + k]) << k;
        dst[i >> 3] = byte;
        set += @popCount(byte);
    }
    if (i < indices.len) {
        var byte: u8 = 0;
        for (indices[i..], 0..) |m, k| byte |= bitOf(src, offset, m) << @intCast(k);
        dst[i >> 3] = byte;
        set += @popCount(byte);
    }
    return set;
}

inline fn bitOf(src: ?[]const u8, offset: usize, maybe: anytype) u8 {
    const idx = @as(?usize, maybe) orelse return 0;
    const s = src orelse return 1;
    const j = offset + idx;
    return (s[j >> 3] >> @intCast(j & 7)) & 1;
}

fn validAt(src_validity: ?bitmap.ValidityBitmap, j: usize) bool {
    const v = src_validity orelse return true;
    return bitmap.bitIsSet(v.data, j);
}

/// Value range of row `j` (physical index) in an offsets buffer; `limit` bounds the end.
fn offsetRange(comptime T: type, offsets: []const T, j: usize, limit: usize) KernelError![2]usize {
    const lo = std.math.cast(usize, offsets[j]) orelse return error.InvalidInput;
    const hi = std.math.cast(usize, offsets[j + 1]) orelse return error.InvalidInput;
    if (hi < lo or hi > limit) return error.InvalidInput;
    return .{ lo, hi };
}

/// Direct gather for string/binary (i32 or i64 offsets): pass 1 bounds-checks offsets and sizes
/// the value buffer, pass 2 copies offsets + bytes, validity is packed by gatherBits.
/// Null source rows keep their bytes (usually none), like arrow-rs. Returns null for other layouts.
fn takeBinaryArray(
    comptime T: type,
    allocator: std.mem.Allocator,
    src: *const array_mod.ArrayData,
    indices: anytype,
) KernelError!?ArrayRef {
    if (src.buffers.len != 3) return null;
    const offsets = src.buffers[1].typedSlice(T) catch return error.InvalidInput;
    const bytes = src.buffers[2].data;
    const src_end = std.math.add(usize, src.offset, src.length) catch return error.Overflow;
    if (offsets.len <= src_end) return error.InvalidInput;
    const src_validity = try validityBytes(src, src_end);
    try checkIndices(indices, src.length);

    var total: usize = 0;
    for (indices) |maybe| {
        const idx: usize = @as(?usize, maybe) orelse continue;
        const r = try offsetRange(T, offsets, src.offset + idx, bytes.len);
        total = std.math.add(usize, total, r[1] - r[0]) catch return error.Overflow;
    }
    if (total > std.math.maxInt(T)) return error.Overflow;

    const out_len = indices.len;
    var out_offsets = buffer.OwnedBuffer.initUninit(allocator, (out_len + 1) * @sizeOf(T)) catch return error.OutOfMemory;
    errdefer out_offsets.deinit();
    // 16 bytes of slack so short values move as one fixed 16-byte copy (copyShort).
    var out_bytes = buffer.OwnedBuffer.initUninit(allocator, total + 16) catch return error.OutOfMemory;
    errdefer out_bytes.deinit();
    var validity = buffer.OwnedBuffer.init(allocator, bitmap.byteLength(out_len)) catch return error.OutOfMemory;
    errdefer validity.deinit();

    const o = out_offsets.typedSlice(T);
    o[0] = 0;
    var pos: usize = 0;
    for (indices, 0..) |maybe, i| {
        if (@as(?usize, maybe)) |idx| {
            const j = src.offset + idx;
            const lo: usize = @intCast(offsets[j]);
            const n: usize = @intCast(offsets[j + 1] - offsets[j]);
            copyShort(out_bytes.data[pos..], bytes, lo, n);
            pos += n;
        }
        o[i + 1] = @intCast(pos);
    }
    const null_count = out_len - gatherBits(validity.data, src_validity, src.offset, indices);

    var buffers: [3]buffer.SharedBuffer = undefined;
    buffers[0] = if (null_count == 0) blk: {
        validity.deinit();
        break :blk buffer.SharedBuffer.empty;
    } else validity.toShared(bitmap.byteLength(out_len)) catch return error.OutOfMemory;
    errdefer buffers[0].release();
    buffers[1] = out_offsets.toShared((out_len + 1) * @sizeOf(T)) catch return error.OutOfMemory;
    errdefer buffers[1].release();
    buffers[2] = out_bytes.toShared(total) catch return error.OutOfMemory;
    errdefer buffers[2].release();

    return ArrayRef.fromOwnedInlineBuffers(allocator, .{
        .data_type = src.data_type,
        .length = out_len,
        .null_count = null_count,
        .buffers = buffers[0..],
    }) catch return error.OutOfMemory;
}

/// Copies `src[lo..][0..n]` to the front of `dst`. Values of at most 16 bytes move as one fixed
/// 16-byte load/store when both sides have room; `dst` bytes past `n` are scratch.
inline fn copyShort(dst: []u8, src: []const u8, lo: usize, n: usize) void {
    if (n <= 16 and lo + 16 <= src.len and dst.len >= 16) {
        dst[0..16].* = src[lo..][0..16].*;
    } else @memcpy(dst[0..n], src[lo..][0..n]);
}

/// Direct gather for list/large_list: rebuilds offsets + validity and takes the child once
/// with the expanded child indices (recursing into the fast kernels). Returns null for other layouts.
fn takeListArray(
    comptime T: type,
    allocator: std.mem.Allocator,
    src: *const array_mod.ArrayData,
    indices: anytype,
) KernelError!?ArrayRef {
    if (src.buffers.len != 2 or src.children.len != 1) return null;
    const offsets = src.buffers[1].typedSlice(T) catch return error.InvalidInput;
    const child = src.children[0];
    const child_len = child.data().length;
    const src_end = std.math.add(usize, src.offset, src.length) catch return error.Overflow;
    if (offsets.len <= src_end) return error.InvalidInput;
    const src_validity = src.validity();

    var total: usize = 0;
    var null_count: usize = 0;
    for (indices) |maybe| {
        const idx: usize = @as(?usize, maybe) orelse {
            null_count += 1;
            continue;
        };
        if (idx >= src.length) return error.InvalidInput;
        const j = src.offset + idx;
        if (!validAt(src_validity, j)) {
            null_count += 1;
            continue;
        }
        const r = try offsetRange(T, offsets, j, child_len);
        total = std.math.add(usize, total, r[1] - r[0]) catch return error.Overflow;
    }
    if (total > std.math.maxInt(T)) return error.Overflow;

    const child_indices = allocator.alloc(usize, total) catch return error.OutOfMemory;
    defer allocator.free(child_indices);
    const out_len = indices.len;
    var out_offsets = buffer.OwnedBuffer.initUninit(allocator, (out_len + 1) * @sizeOf(T)) catch return error.OutOfMemory;
    errdefer out_offsets.deinit();
    var validity = buffer.OwnedBuffer.init(allocator, bitmap.byteLength(out_len)) catch return error.OutOfMemory;
    errdefer validity.deinit();

    const o = out_offsets.typedSlice(T);
    o[0] = 0;
    var pos: usize = 0;
    for (indices, 0..) |maybe, i| {
        if (@as(?usize, maybe)) |idx| {
            const j = src.offset + idx;
            if (validAt(src_validity, j)) {
                const lo: usize = @intCast(offsets[j]);
                const hi: usize = @intCast(offsets[j + 1]);
                for (child_indices[pos..][0 .. hi - lo], lo..) |*ci, k| ci.* = k;
                pos += hi - lo;
                bitmap.setBit(validity.data, i);
            }
        }
        o[i + 1] = @intCast(pos);
    }

    const children = try takeChildren(allocator, src.children, null, @as([]const usize, child_indices));
    errdefer releaseChildren(allocator, children);

    var buffers: [2]buffer.SharedBuffer = undefined;
    buffers[0] = if (null_count == 0) blk: {
        validity.deinit();
        break :blk buffer.SharedBuffer.empty;
    } else validity.toShared(bitmap.byteLength(out_len)) catch return error.OutOfMemory;
    errdefer buffers[0].release();
    buffers[1] = out_offsets.toShared((out_len + 1) * @sizeOf(T)) catch return error.OutOfMemory;
    errdefer buffers[1].release();

    return ArrayRef.fromOwnedInlineBuffers(allocator, .{
        .data_type = src.data_type,
        .length = out_len,
        .null_count = null_count,
        .buffers = buffers[0..],
        .children = children,
    }) catch return error.OutOfMemory;
}

/// Takes the same `indices` from every child. With a struct/union `parent`, children are either
/// row-aligned (same length as the parent, as `ArrayRef.slice` makes them) or in parent
/// coordinates (row = offset + idx); without one, indices are child rows.
fn takeChildren(allocator: std.mem.Allocator, src_children: []const ArrayRef, parent: ?*const array_mod.ArrayData, indices: anytype) KernelError![]ArrayRef {
    const children = allocator.alloc(ArrayRef, src_children.len) catch return error.OutOfMemory;
    var built: usize = 0;
    errdefer {
        for (children[0..built]) |*child| child.release();
        allocator.free(children);
    }
    for (src_children, children) |child, *out| {
        const p = parent orelse child.data();
        if (child.data().length == p.length) {
            out.* = try takeChild(allocator, child, indices);
        } else {
            if (child.data().length < p.offset + p.length) return error.InvalidInput;
            var window = child.slice(p.offset, p.length) catch return error.OutOfMemory;
            defer window.release();
            out.* = try takeChild(allocator, window, indices);
        }
        built += 1;
    }
    return children;
}

fn releaseChildren(allocator: std.mem.Allocator, children: []ArrayRef) void {
    for (children) |*child| child.release();
    allocator.free(children);
}

/// Direct take for struct and sparse union: every child is taken with the same indices (null
/// indices take null child rows; null struct rows keep their child rows, like arrow-rs).
/// Struct validity is gathered; union type ids are gathered (null index: the first type id).
fn takeStructLikeArray(allocator: std.mem.Allocator, src: *const array_mod.ArrayData, indices: anytype) KernelError!ArrayRef {
    const is_union = storageDataType(src.data_type) == .sparse_union;
    if (src.buffers.len < 1) return error.InvalidInput;
    const src_end = std.math.add(usize, src.offset, src.length) catch return error.Overflow;
    try checkIndices(indices, src.length);
    const out_len = indices.len;

    var head = buffer.OwnedBuffer.init(allocator, if (is_union) out_len else bitmap.byteLength(out_len)) catch return error.OutOfMemory;
    errdefer head.deinit();
    var null_count: usize = 0;
    if (is_union) {
        const ids = src.buffers[0].data;
        if (ids.len < src_end) return error.InvalidInput;
        const declared = storageDataType(src.data_type).sparse_union.type_ids;
        if (declared.len == 0 and out_len > 0) return error.InvalidInput;
        const first: u8 = if (declared.len > 0) @bitCast(declared[0]) else 0;
        for (indices, head.data[0..out_len]) |m, *o| o.* = if (@as(?usize, m)) |j| ids[src.offset + j] else first;
    } else {
        null_count = out_len - gatherBits(head.data, try validityBytes(src, src_end), src.offset, indices);
    }

    const children = try takeChildren(allocator, src.children, src, indices);
    errdefer releaseChildren(allocator, children);
    var buffers: [1]buffer.SharedBuffer = .{if (!is_union and null_count == 0) blk: {
        head.deinit();
        break :blk buffer.SharedBuffer.empty;
    } else head.toShared(if (is_union) out_len else bitmap.byteLength(out_len)) catch return error.OutOfMemory};
    errdefer buffers[0].release();

    return ArrayRef.fromOwnedInlineBuffers(allocator, .{
        .data_type = src.data_type,
        .length = out_len,
        .null_count = null_count,
        .buffers = buffers[0..],
        .children = children,
    }) catch return error.OutOfMemory;
}

/// Direct take for fixed_size_list: validity by gatherBits, the child taken once with each
/// index expanded to its `list_size` child rows (null indices take null child rows).
fn takeFixedSizeListArray(allocator: std.mem.Allocator, src: *const array_mod.ArrayData, indices: anytype) KernelError!ArrayRef {
    if (src.buffers.len != 1 or src.children.len != 1) return error.InvalidInput;
    const size = std.math.cast(usize, storageDataType(src.data_type).fixed_size_list.list_size) orelse return error.InvalidInput;
    const src_end = std.math.add(usize, src.offset, src.length) catch return error.Overflow;
    const child_end = std.math.mul(usize, src_end, size) catch return error.Overflow;
    if (src.children[0].data().length < child_end) return error.InvalidInput;
    const src_validity = try validityBytes(src, src_end);
    try checkIndices(indices, src.length);
    const out_len = indices.len;

    // A fixed-width child moves each row's values as one copy; other children are taken with the
    // expanded indices (dense ones as u32 child rows when the child fits: half the scratch).
    const child = src.children[0].data();
    const child_layout = storageDataType(child.data_type);
    const width = (try fixedByteWidth(child_layout)) orelse 0;
    const children = if (width > 0 and child_layout != .dictionary and child.buffers.len == 2) blk: {
        const one = allocator.alloc(ArrayRef, 1) catch return error.OutOfMemory;
        one[0] = takeFixedSizeListValues(allocator, child, width, size, src.offset, indices) catch |err| {
            allocator.free(one);
            return err;
        };
        break :blk one;
    } else if (!comptime isDense(@TypeOf(indices)))
        try takeExpandedChild(?usize, allocator, src, size, indices)
    else if (child_end <= std.math.maxInt(u32))
        try takeExpandedChild(u32, allocator, src, size, indices)
    else
        try takeExpandedChild(usize, allocator, src, size, indices);
    errdefer releaseChildren(allocator, children);

    var validity = buffer.OwnedBuffer.init(allocator, bitmap.byteLength(out_len)) catch return error.OutOfMemory;
    errdefer validity.deinit();
    const null_count = out_len - gatherBits(validity.data, src_validity, src.offset, indices);
    var buffers: [1]buffer.SharedBuffer = .{if (null_count == 0) blk: {
        validity.deinit();
        break :blk buffer.SharedBuffer.empty;
    } else validity.toShared(bitmap.byteLength(out_len)) catch return error.OutOfMemory};
    errdefer buffers[0].release();

    return ArrayRef.fromOwnedInlineBuffers(allocator, .{
        .data_type = src.data_type,
        .length = out_len,
        .null_count = null_count,
        .buffers = buffers[0..],
        .children = children,
    }) catch return error.OutOfMemory;
}

/// The fixed-width child of a fixed_size_list taken by list rows: row `base + idx` moves its
/// `size` values with one copy and their validity bits one by one; null indices give null rows.
/// The caller checked the child covers every row.
fn takeFixedSizeListValues(allocator: std.mem.Allocator, child: *const array_mod.ArrayData, width: usize, size: usize, base: usize, indices: anytype) KernelError!ArrayRef {
    const child_end = std.math.add(usize, child.offset, child.length) catch return error.Overflow;
    const src_bytes = child.buffers[1].data;
    if (src_bytes.len < std.math.mul(usize, child_end, width) catch return error.Overflow) return error.InvalidInput;
    const child_validity = try validityBytes(child, child_end);
    const out_len = indices.len * size;
    const row_bytes = size * width;

    var values = buffer.OwnedBuffer.initUninit(allocator, out_len * width) catch return error.OutOfMemory;
    errdefer values.deinit();
    var validity = buffer.OwnedBuffer.init(allocator, bitmap.byteLength(out_len)) catch return error.OutOfMemory;
    errdefer validity.deinit();
    var valid: usize = 0;
    for (indices, 0..) |m, i| {
        const out = values.data[i * row_bytes ..][0..row_bytes];
        const idx = @as(?usize, m) orelse {
            @memset(out, 0);
            continue;
        };
        const first = child.offset + (base + idx) * size;
        @memcpy(out, src_bytes[first * width ..][0..row_bytes]);
        for (0..size) |k| {
            if (child_validity) |v| if (!bitmap.bitIsSet(v, first + k)) continue;
            bitmap.setBit(validity.data, i * size + k);
            valid += 1;
        }
    }

    const null_count = out_len - valid;
    var buffers: [2]buffer.SharedBuffer = undefined;
    buffers[0] = if (null_count == 0) blk: {
        validity.deinit();
        break :blk buffer.SharedBuffer.empty;
    } else validity.toShared(bitmap.byteLength(out_len)) catch return error.OutOfMemory;
    errdefer buffers[0].release();
    buffers[1] = values.toShared(out_len * width) catch return error.OutOfMemory;
    errdefer buffers[1].release();
    return ArrayRef.fromOwnedInlineBuffers(allocator, .{
        .data_type = child.data_type,
        .length = out_len,
        .null_count = null_count,
        .buffers = buffers[0..],
    }) catch return error.OutOfMemory;
}

/// Takes child rows (offset + idx) * size + k, k < size, for every index (null: null child rows).
fn takeExpandedChild(comptime E: type, allocator: std.mem.Allocator, src: *const array_mod.ArrayData, size: usize, indices: anytype) KernelError![]ArrayRef {
    const child_indices = allocator.alloc(E, std.math.mul(usize, indices.len, size) catch return error.Overflow) catch return error.OutOfMemory;
    defer allocator.free(child_indices);
    for (indices, 0..) |m, i| {
        const dst = child_indices[i * size ..][0..size];
        if (@as(?usize, m)) |idx| {
            for (dst, (src.offset + idx) * size..) |*c, k| c.* = if (comptime E == u32) @intCast(k) else k;
        } else if (comptime E != ?usize) unreachable else @memset(dst, null);
    }
    return takeChildren(allocator, src.children, null, @as([]const E, child_indices));
}

/// Direct take for list_view/large_list_view: gathers offsets and sizes (bounds-checked against
/// the child); the child is shared, not copied. Null indices are empty null views.
fn takeListViewArray(comptime T: type, allocator: std.mem.Allocator, src: *const array_mod.ArrayData, indices: anytype) KernelError!ArrayRef {
    if (src.buffers.len != 3 or src.children.len != 1) return error.InvalidInput;
    const offsets = src.buffers[1].typedSlice(T) catch return error.InvalidInput;
    const sizes = src.buffers[2].typedSlice(T) catch return error.InvalidInput;
    const child_len = src.children[0].data().length;
    const src_end = std.math.add(usize, src.offset, src.length) catch return error.Overflow;
    if (offsets.len < src_end or sizes.len < src_end) return error.InvalidInput;
    const src_validity = try validityBytes(src, src_end);
    try checkIndices(indices, src.length);
    const out_len = indices.len;

    var out_offsets = buffer.OwnedBuffer.initUninit(allocator, out_len * @sizeOf(T)) catch return error.OutOfMemory;
    errdefer out_offsets.deinit();
    var out_sizes = buffer.OwnedBuffer.initUninit(allocator, out_len * @sizeOf(T)) catch return error.OutOfMemory;
    errdefer out_sizes.deinit();
    for (indices, out_offsets.typedSlice(T)[0..out_len], out_sizes.typedSlice(T)[0..out_len]) |m, *o, *s| {
        const j = src.offset + (@as(?usize, m) orelse {
            o.* = 0;
            s.* = 0;
            continue;
        });
        const lo = std.math.cast(usize, offsets[j]) orelse return error.InvalidInput;
        const n = std.math.cast(usize, sizes[j]) orelse return error.InvalidInput;
        if (n > child_len or lo > child_len - n) return error.InvalidInput;
        o.* = offsets[j];
        s.* = sizes[j];
    }
    var validity = buffer.OwnedBuffer.init(allocator, bitmap.byteLength(out_len)) catch return error.OutOfMemory;
    errdefer validity.deinit();
    const null_count = out_len - gatherBits(validity.data, src_validity, src.offset, indices);

    const children = allocator.alloc(ArrayRef, 1) catch return error.OutOfMemory;
    errdefer allocator.free(children);
    var buffers: [3]buffer.SharedBuffer = undefined;
    buffers[0] = if (null_count == 0) blk: {
        validity.deinit();
        break :blk buffer.SharedBuffer.empty;
    } else validity.toShared(bitmap.byteLength(out_len)) catch return error.OutOfMemory;
    errdefer buffers[0].release();
    buffers[1] = out_offsets.toShared(out_len * @sizeOf(T)) catch return error.OutOfMemory;
    errdefer buffers[1].release();
    buffers[2] = out_sizes.toShared(out_len * @sizeOf(T)) catch return error.OutOfMemory;
    errdefer buffers[2].release();

    children[0] = src.children[0];
    const out = ArrayRef.fromOwnedInlineBuffers(allocator, .{
        .data_type = src.data_type,
        .length = out_len,
        .null_count = null_count,
        .buffers = buffers[0..],
        .children = children,
    }) catch return error.OutOfMemory;
    _ = children[0].retain();
    return out;
}

/// Layouts with a direct take kernel; the rest (views, dense unions, run-end encoded) take row
/// by row (slice + concat).
fn hasTakeKernel(dt: DataType) bool {
    return switch (storageDataType(dt)) {
        .string_view, .binary_view, .dense_union, .run_end_encoded => false,
        else => true,
    };
}

/// Direct take by layout; extension arrays take their storage layout and keep their type.
fn takeArray(allocator: std.mem.Allocator, src: *const array_mod.ArrayData, indices: anytype) KernelError!?ArrayRef {
    if (!hasTakeKernel(src.data_type)) return null;
    return switch (storageDataType(src.data_type)) {
        .null => blk: {
            try checkIndices(indices, src.length);
            break :blk ArrayRef.fromOwnedInlineBuffers(allocator, .{
                .data_type = src.data_type,
                .length = indices.len,
                .null_count = indices.len,
                .buffers = &.{},
            }) catch return error.OutOfMemory;
        },
        .string, .binary => try takeBinaryArray(i32, allocator, src, indices),
        .large_string, .large_binary => try takeBinaryArray(i64, allocator, src, indices),
        .list, .map => try takeListArray(i32, allocator, src, indices),
        .large_list => try takeListArray(i64, allocator, src, indices),
        .list_view => try takeListViewArray(i32, allocator, src, indices),
        .large_list_view => try takeListViewArray(i64, allocator, src, indices),
        .fixed_size_list => try takeFixedSizeListArray(allocator, src, indices),
        .struct_, .sparse_union => try takeStructLikeArray(allocator, src, indices),
        else => try takeFixedWidthArray(allocator, src, indices),
    };
}

fn datumTakeArrayLikeNullable(
    allocator: std.mem.Allocator,
    datum: Datum,
    out_type: DataType,
    indices: anytype,
) KernelError!Datum {
    if (datum == .array) {
        if (try takeArray(allocator, datum.array.data(), indices)) |out| return Datum.fromArray(out);
    }
    if (indices.len == 0) return builders.datumBuildEmptyLikeWithAllocator(allocator, out_type);

    var pieces: std.ArrayList(ArrayRef) = .empty;
    defer {
        for (pieces.items) |*piece| piece.release();
        pieces.deinit(allocator);
    }
    try pieces.ensureTotalCapacity(allocator, indices.len);

    for (indices) |choice| {
        const piece = if (@as(?usize, choice)) |idx|
            try accessors.datumElementArrayAt(allocator, datum, idx)
        else
            try builders.buildNullLikeArray(allocator, out_type, 1);
        pieces.appendAssumeCapacity(piece);
    }

    const out = concat_array_refs.concatArrayRefs(allocator, out_type, pieces.items) catch |err| return common.mapConcatError(err);
    return Datum.fromArray(out);
}

fn predicateDecisionAt(predicate: Datum, logical_index: usize, options: FilterOptions) KernelError!PredicateDecision {
    return switch (predicate) {
        .scalar => |s| blk: {
            if (s.data_type != .bool) break :blk error.InvalidInput;
            if (s.isNull()) break :blk if (options.drop_nulls) .drop else .emit_null;
            break :blk if (s.value.bool) .keep else .drop;
        },
        .array => |arr| blk: {
            if (arr.data().data_type != .bool) break :blk error.InvalidInput;
            if (logical_index >= arr.data().length) break :blk error.InvalidInput;
            if (arr.data().isNull(logical_index)) break :blk if (options.drop_nulls) .drop else .emit_null;
            const bool_array = array_mod.BooleanArray{ .data = arr.data() };
            const keep = bool_array.value(logical_index) catch |err| return mapArrayReadError(err);
            break :blk if (keep) .keep else .drop;
        },
        .chunked => |chunks| blk: {
            if (chunks.dataType() != .bool) break :blk error.InvalidInput;
            const located = lookupChunkAt(chunks, logical_index) orelse break :blk error.InvalidInput;
            if (located.chunk.data().isNull(located.local_index)) break :blk if (options.drop_nulls) .drop else .emit_null;
            const bool_array = array_mod.BooleanArray{ .data = located.chunk.data() };
            const keep = bool_array.value(located.local_index) catch |err| return mapArrayReadError(err);
            break :blk if (keep) .keep else .drop;
        },
    };
}

const SelectionByte = struct { emit: u8, valid: u8 };

/// Rows `row .. row+8` of a bool predicate: which rows emit an entry and which of those are valid.
fn selectionByte(values: []const u8, validity: []const u8, bit: usize, rows_left: usize, drop_nulls: bool) SelectionByte {
    const tail_mask: u8 = if (rows_left >= 8) 0xFF else (@as(u8, 1) << @intCast(rows_left)) - 1;
    const valid = if (validity.len > 0) bitmap.byteAtBit(validity, bit) else 0xFF;
    var emit = bitmap.byteAtBit(values, bit) & valid;
    if (!drop_nulls) emit |= ~valid;
    return .{ .emit = emit & tail_mask, .valid = valid };
}

/// Appends selections for rows [0, len) of a bool predicate array, 8 rows per step.
/// `base` is the logical index of the predicate's row 0.
fn appendBoolSelections(
    allocator: std.mem.Allocator,
    selections: anytype,
    pred: *const array_mod.ArrayData,
    len: usize,
    base: usize,
    options: FilterOptions,
) KernelError!void {
    if (pred.buffers.len < 2) return error.InvalidInput;
    const values = pred.buffers[1].data;
    const end = std.math.add(usize, pred.offset, len) catch return error.Overflow;
    if (bitmap.byteLength(end) > values.len) return error.InvalidInput;
    const validity: []const u8 = if (pred.validity()) |v| v.data else &.{};
    if (validity.len > 0 and bitmap.byteLength(end) > validity.len) return error.InvalidInput;

    var count: usize = 0;
    var row: usize = 0;
    while (row < len) : (row += 8) {
        count += @popCount(selectionByte(values, validity, pred.offset + row, len - row, options.drop_nulls).emit);
    }
    try selections.ensureTotalCapacityPrecise(allocator, selections.items.len + count);

    row = 0;
    while (row < len) : (row += 8) {
        const sb = selectionByte(values, validity, pred.offset + row, len - row, options.drop_nulls);
        var emit = sb.emit;
        while (emit != 0) : (emit &= emit - 1) {
            const bit: u3 = @intCast(@ctz(emit));
            const is_valid = (sb.valid >> bit) & 1 != 0;
            if (@TypeOf(selections.items[0]) != ?usize) {
                std.debug.assert(is_valid); // dense lists are only built with drop_nulls
                selections.appendAssumeCapacity(@intCast(base + row + bit));
            } else selections.appendAssumeCapacity(if (is_valid) base + row + bit else null);
        }
    }
}

/// Build nullable selection indices from a boolean predicate.
///
/// `keep` emits the logical input index, `drop` emits nothing, and `emit_null`
/// emits `null` to preserve output row count for null-predicate semantics.
pub fn datumFilterSelectionIndices(
    allocator: std.mem.Allocator,
    predicate: Datum,
    logical_len: usize,
    options: FilterOptions,
) KernelError![]?usize {
    var selections: std.ArrayList(?usize) = .empty;
    defer selections.deinit(allocator);

    switch (predicate) {
        .array => |arr| {
            if (arr.data().data_type != .bool) return error.InvalidInput;
            if (logical_len > arr.data().length) return error.InvalidInput;
            try appendBoolSelections(allocator, &selections, arr.data(), logical_len, 0, options);
        },
        .chunked => |chunks| {
            if (chunks.dataType() != .bool) return error.InvalidInput;
            var done: usize = 0;
            for (chunks.chunks()) |chunk| {
                if (done == logical_len) break;
                const take = @min(chunk.data().length, logical_len - done);
                try appendBoolSelections(allocator, &selections, chunk.data(), take, done, options);
                done += take;
            }
            if (done != logical_len) return error.InvalidInput;
        },
        .scalar => {
            var i: usize = 0;
            while (i < logical_len) : (i += 1) {
                switch (try predicateDecisionAt(predicate, i, options)) {
                    .drop => {},
                    .keep => try selections.append(allocator, i),
                    .emit_null => try selections.append(allocator, null),
                }
            }
        },
    }
    return selections.toOwnedSlice(allocator);
}

/// Gather/take rows from a datum according to logical indices.
///
/// For chunked inputs this helper preserves chunked output and avoids forcing
/// concat pre-normalization.
pub fn datumTake(datum: Datum, indices: []const usize) KernelError!Datum {
    const allocator = common.inferDatumAllocator(datum) orelse std.heap.page_allocator;
    const out_type = datum.dataType();

    if (indices.len == 0) {
        return switch (datum) {
            .chunked => |chunks| Datum.fromChunked(chunks.slice(allocator, 0, 0) catch |err| return common.mapChunkedError(err)),
            else => builders.datumBuildEmptyLikeWithAllocator(allocator, out_type),
        };
    }

    return switch (datum) {
        .chunked => |chunks| datumTakeChunkedNullable(allocator, chunks, out_type, indices),
        else => datumTakeArrayLikeNullable(allocator, datum, out_type, indices),
    };
}

/// Gather/take rows with nullable indices where `null` emits an all-null row.
///
/// For chunked inputs this helper preserves chunked output and avoids forcing
/// concat pre-normalization.
pub fn datumTakeNullable(datum: Datum, indices: []const ?usize) KernelError!Datum {
    const allocator = common.inferDatumAllocator(datum) orelse std.heap.page_allocator;
    const out_type = datum.dataType();
    return switch (datum) {
        .chunked => |chunks| datumTakeChunkedNullable(allocator, chunks, out_type, indices),
        else => datumTakeArrayLikeNullable(allocator, datum, out_type, indices),
    };
}

/// Row-wise selection primitive shared by choose/case_when/filter style operators.
///
/// `indices[i]` selects which entry in `values` contributes output row `i`.
/// Each candidate value may be array/chunked/scalar (scalar broadcasts).
pub fn datumSelect(indices: []const usize, values: []const Datum) KernelError!Datum {
    if (values.len == 0) return error.InvalidArity;
    if (!sameDataTypes(values)) return error.InvalidInput;

    const output_len = try inferNaryExecLen(values);
    if (indices.len != output_len) return error.InvalidInput;

    const allocator = common.inferDatumsAllocator(values);
    const out_type = values[0].dataType();
    if (indices.len == 0) return builders.datumBuildEmptyLikeWithAllocator(allocator, out_type);

    var pieces: std.ArrayList(ArrayRef) = .empty;
    defer {
        for (pieces.items) |*piece| piece.release();
        pieces.deinit(allocator);
    }
    try pieces.ensureTotalCapacity(allocator, indices.len);

    for (indices, 0..) |choice, logical_index| {
        if (choice >= values.len) return error.InvalidInput;
        const piece = try accessors.datumElementArrayAt(allocator, values[choice], logical_index);
        pieces.appendAssumeCapacity(piece);
    }

    const out = concat_array_refs.concatArrayRefs(allocator, out_type, pieces.items) catch |err| return common.mapConcatError(err);
    return Datum.fromArray(out);
}

/// Row-wise selection primitive with nullable index support.
///
/// `indices[i] == null` emits a null row at output position `i`.
pub fn datumSelectNullable(indices: []const ?usize, values: []const Datum) KernelError!Datum {
    if (values.len == 0) return error.InvalidArity;
    if (!sameDataTypes(values)) return error.InvalidInput;

    const output_len = try inferNaryExecLen(values);
    if (indices.len != output_len) return error.InvalidInput;

    const allocator = common.inferDatumsAllocator(values);
    const out_type = values[0].dataType();
    if (indices.len == 0) return builders.datumBuildEmptyLikeWithAllocator(allocator, out_type);

    var pieces: std.ArrayList(ArrayRef) = .empty;
    defer {
        for (pieces.items) |*piece| piece.release();
        pieces.deinit(allocator);
    }
    try pieces.ensureTotalCapacity(allocator, indices.len);

    for (indices, 0..) |choice, logical_index| {
        const piece = if (choice) |idx| blk: {
            if (idx >= values.len) return error.InvalidInput;
            break :blk try accessors.datumElementArrayAt(allocator, values[idx], logical_index);
        } else try builders.buildNullLikeArray(allocator, out_type, 1);
        pieces.appendAssumeCapacity(piece);
    }

    const out = concat_array_refs.concatArrayRefs(allocator, out_type, pieces.items) catch |err| return common.mapConcatError(err);
    return Datum.fromArray(out);
}

/// Chunk-aware filter helper for array/chunked/scalar datums with bool predicates.
///
/// This helper composes `datumFilterSelectionIndices` + `datumTakeNullable`,
/// preserving chunked output when the input datum is chunked.
pub fn datumFilterChunkAware(datum: Datum, predicate: Datum, options: FilterOptions) KernelError!Datum {
    const input_len = try inferBinaryExecLen(datum, predicate);
    const allocator = common.inferDatumsAllocator(&[_]Datum{ datum, predicate });
    const selections = try datumFilterSelectionIndices(allocator, predicate, input_len, options);
    defer allocator.free(selections);
    return datumTakeNullable(datum, selections);
}

/// Generic filter primitive for array/chunked/scalar datums with bool predicates.
///
/// This preserves the historical array-output shape for compatibility.
pub fn datumFilter(datum: Datum, predicate: Datum, options: FilterOptions) KernelError!Datum {
    const output_len = try inferBinaryExecLen(datum, predicate);
    const allocator = common.inferDatumsAllocator(&[_]Datum{ datum, predicate });
    const out_type = datum.dataType();

    if (output_len == 0) return builders.datumBuildEmptyLikeWithAllocator(allocator, out_type);
    if (predicate == .array and options.drop_nulls and output_len <= std.math.maxInt(u32)) {
        // drop_nulls: every kept row is a valid index, so the selections are dense u32 (4 B per
        // kept row, exact size) and the take runs the branch-free dense kernels.
        const pred = predicate.array.data();
        if (pred.data_type != .bool or output_len > pred.length) return error.InvalidInput;
        var selections: std.ArrayList(u32) = .empty;
        defer selections.deinit(allocator);
        try appendBoolSelections(allocator, &selections, pred, output_len, 0, options);
        return takeAsArray(allocator, datum, out_type, @as([]const u32, selections.items));
    }
    const selections = try datumFilterSelectionIndices(allocator, predicate, output_len, options);
    defer allocator.free(selections);
    return takeAsArray(allocator, datum, out_type, selections);
}

/// Take with array output for every datum form (chunked pieces are concatenated once).
fn takeAsArray(allocator: std.mem.Allocator, datum: Datum, out_type: DataType, indices: anytype) KernelError!Datum {
    if (datum != .chunked) return datumTakeArrayLikeNullable(allocator, datum, out_type, indices);
    var pieces: std.ArrayList(ArrayRef) = .empty;
    defer releasePieces(allocator, &pieces);
    try takeChunked(allocator, datum.chunked, out_type, indices, &pieces);
    if (pieces.items.len == 1) return Datum.fromArray(pieces.items[0].retain());
    if (pieces.items.len == 0) return builders.datumBuildEmptyLikeWithAllocator(allocator, out_type);
    return Datum.fromArray(concat_array_refs.concatArrayRefs(allocator, out_type, pieces.items) catch |err| return common.mapConcatError(err));
}
