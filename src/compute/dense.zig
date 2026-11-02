const std = @import("std");
const bitmap = @import("../bitmap.zig");
const buffer = @import("../buffer.zig");
const array_data = @import("../array/array_data.zig");
const array_ref = @import("../array/array_ref.zig");
const datatype = @import("../datatype.zig");

// Dense execution helpers for kernels: the validity of the inputs is combined once, bitmap word
// by bitmap word, so the value loop runs over plain typed slices with no per-element null check
// (null slots compute garbage that the validity bitmap hides, like arrow-rs / Arrow C++).

const ArrayData = array_data.ArrayData;
const ArrayRef = array_ref.ArrayRef;
const DataType = datatype.DataType;
const OwnedBuffer = buffer.OwnedBuffer;
const SharedBuffer = buffer.SharedBuffer;

/// Bits `[offset, offset + len)` of `src` ANDed into `dst` (bit 0 = row 0).
fn andValidityBits(dst: []u8, src: []const u8, offset: usize, len: usize) void {
    const bytes = bitmap.byteLength(len);
    if (offset % 8 == 0) {
        const aligned = src[offset / 8 ..];
        for (dst[0..bytes], aligned[0..bytes]) |*d, s| d.* &= s;
    } else {
        for (dst[0..bytes], 0..) |*d, k| d.* &= bitmap.byteAtBit(src, offset + k * 8);
    }
}

/// Output of a dense unary/binary kernel over primitive inputs: a values buffer (`[]T`, or the
/// packed bits of a bool array when `T == bool`) plus the AND of the inputs' validity bitmaps.
pub fn DenseBinaryOutput(comptime T: type) type {
    return struct {
        allocator: std.mem.Allocator,
        len: usize,
        values_buf: OwnedBuffer,
        validity: ?OwnedBuffer,
        out_type: DataType,

        const Self = @This();
        const is_bits = T == bool;

        /// `rhs` may be null for a unary kernel. Rows are `lhs.length`; `rhs` must have the same length.
        pub fn init(allocator: std.mem.Allocator, lhs: *const ArrayData, rhs: ?*const ArrayData, out_type: DataType) !Self {
            const len = lhs.length;
            if (rhs) |r| if (r.length != len) return error.InvalidInput;
            var values_buf = try OwnedBuffer.initUninit(allocator, if (is_bits) bitmap.byteLength(len) else len * @sizeOf(T));
            errdefer values_buf.deinit();

            var validity: ?OwnedBuffer = null;
            const inputs = [2]?*const ArrayData{ lhs, rhs };
            for (inputs) |maybe| {
                const input = maybe orelse continue;
                if (!input.hasNulls()) continue;
                const bits = (input.validity() orelse continue).data;
                if (validity == null) {
                    var v = try OwnedBuffer.initUninit(allocator, bitmap.byteLength(len));
                    @memset(v.data[0..bitmap.byteLength(len)], 0xFF);
                    validity = v;
                }
                andValidityBits(validity.?.data, bits, input.offset, len);
            }
            // Padding bits past `len` may hold source garbage: clear them.
            if (validity) |*v| if (len % 8 != 0) {
                const keep: u8 = (@as(u8, 1) << @as(u3, @intCast(len % 8))) - 1;
                v.data[bitmap.byteLength(len) - 1] &= keep;
            };
            return .{ .allocator = allocator, .len = len, .values_buf = values_buf, .validity = validity, .out_type = out_type };
        }

        pub fn deinit(self: *Self) void {
            self.values_buf.deinit();
            if (self.validity) |*v| v.deinit();
        }

        /// The value slots to fill: `[]T`, or the `byteLength(len)` packed bytes for `bool`.
        pub fn values(self: *Self) if (is_bits) []u8 else []T {
            if (is_bits) return self.values_buf.data[0..bitmap.byteLength(self.len)];
            return std.mem.bytesAsSlice(T, self.values_buf.data)[0..self.len];
        }

        /// True when row `i` is valid in every input (for checked-overflow decisions on null slots).
        pub fn isValid(self: *const Self, i: usize) bool {
            const v = self.validity orelse return true;
            return bitmap.bitIsSet(v.data, i);
        }

        /// Hands the buffers to a new array (`null_count` counted from the combined bitmap).
        pub fn finish(self: *Self) !ArrayRef {
            var buffers: [2]SharedBuffer = undefined;
            var null_count: usize = 0;
            buffers[0] = .empty;
            if (self.validity) |*v| {
                null_count = bitmap.countUnsetBit(v.data, self.len);
                buffers[0] = try v.toShared(bitmap.byteLength(self.len));
                self.validity = null;
            }
            errdefer buffers[0].release();
            buffers[1] = try self.values_buf.toShared(if (is_bits) bitmap.byteLength(self.len) else self.len * @sizeOf(T));
            errdefer buffers[1].release();
            return ArrayRef.fromOwnedInlineBuffers(self.allocator, .{
                .data_type = self.out_type,
                .length = self.len,
                .null_count = null_count,
                .buffers = buffers[0..],
            });
        }
    };
}

/// Validity bitmap bytes of `data` (null when no row is null), for `laneMask`.
pub fn validityBits(data: *const ArrayData) ?[]const u8 {
    if (!data.hasNulls()) return null;
    return if (data.validity()) |v| v.data else null;
}

/// Validity of rows `[i, i + 8)` as a lane mask: null rows and rows past `data.length` are false.
/// Aggregates run `@select(T, mask, values[i..][0..8], neutral)` over whole blocks with no branch
/// per row (`bitmap.byteAtBit` reads 8 bits at any bit position, so sliced inputs cost nothing).
pub inline fn laneMask(data: *const ArrayData, bits: ?[]const u8, i: usize) @Vector(8, bool) {
    std.debug.assert(i < data.length);
    const live: u8 = if (i + 8 <= data.length) 0xFF else @as(u8, 0xFF) >> @as(u3, @intCast(8 - (data.length - i)));
    const byte: u8 = (if (bits) |b| bitmap.byteAtBit(b, data.offset + i) else 0xFF) & live;
    const bytes: @Vector(8, u8) = @splat(byte);
    const shifts = @Vector(8, u3){ 0, 1, 2, 3, 4, 5, 6, 7 };
    return ((bytes >> shifts) & @as(@Vector(8, u8), @splat(1))) == @as(@Vector(8, u8), @splat(1));
}

// ------------------------------------------------------------------------------------------ tests

const PrimitiveBuilder = @import("../array/primitive_array.zig").PrimitiveBuilder;

fn randomInt32(allocator: std.mem.Allocator, rng: std.Random, len: usize, null_pct: u8) !ArrayRef {
    var b = try PrimitiveBuilder(i32, .{ .int32 = {} }).init(allocator, len);
    defer b.deinit();
    for (0..len) |_| {
        if (rng.uintLessThan(u8, 100) < null_pct) try b.appendNull() else try b.append(rng.int(i32));
    }
    return b.finish();
}

test "DenseBinaryOutput validity equals per-row null propagation (sliced, nullable, empty inputs)" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(8);
    const rng = prng.random();
    const cases = [_]struct { len: usize, lp: u8, rp: u8, lo: usize, ro: usize }{
        .{ .len = 1000, .lp = 10, .rp = 30, .lo = 3, .ro = 11 },
        .{ .len = 1000, .lp = 0, .rp = 50, .lo = 0, .ro = 5 },
        .{ .len = 777, .lp = 100, .rp = 0, .lo = 8, .ro = 0 },
        .{ .len = 64, .lp = 20, .rp = 20, .lo = 0, .ro = 0 },
        .{ .len = 0, .lp = 20, .rp = 20, .lo = 0, .ro = 0 },
    };
    for (cases) |c| {
        var l_full = try randomInt32(allocator, rng, c.len + c.lo, c.lp);
        defer l_full.release();
        var r_full = try randomInt32(allocator, rng, c.len + c.ro, c.rp);
        defer r_full.release();
        var l = try l_full.slice(c.lo, c.len);
        defer l.release();
        var r = try r_full.slice(c.ro, c.len);
        defer r.release();

        var out = try DenseBinaryOutput(i32).init(allocator, l.data(), r.data(), .{ .int32 = {} });
        errdefer out.deinit();
        const a = std.mem.bytesAsSlice(i32, l.data().buffers[1].data)[l.data().offset..][0..c.len];
        const b = std.mem.bytesAsSlice(i32, r.data().buffers[1].data)[r.data().offset..][0..c.len];
        for (a, b, out.values()) |x, y, *d| d.* = x +% y;
        for (0..c.len) |i| try std.testing.expectEqual(!(l.data().isNull(i) or r.data().isNull(i)), out.isValid(i));
        var arr = try out.finish();
        defer arr.release();
        try arr.data().validateFull();

        var expected_nulls: usize = 0;
        for (0..c.len) |i| {
            const is_null = l.data().isNull(i) or r.data().isNull(i);
            if (is_null) expected_nulls += 1;
            try std.testing.expectEqual(is_null, arr.data().isNull(i));
            if (!is_null) try std.testing.expectEqual(a[i] +% b[i], std.mem.bytesAsSlice(i32, arr.data().buffers[1].data)[i]);
        }
        try std.testing.expectEqual(expected_nulls, arr.data().null_count.?);
        if (c.lp == 0 and c.rp == 0) try std.testing.expect(arr.data().buffers[0].isEmpty());
    }
}

test "DenseBinaryOutput(bool) builds a valid boolean array" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(9);
    var l = try randomInt32(allocator, prng.random(), 100, 10);
    defer l.release();
    var r = try randomInt32(allocator, prng.random(), 100, 10);
    defer r.release();
    var out = try DenseBinaryOutput(bool).init(allocator, l.data(), r.data(), .{ .bool = {} });
    errdefer out.deinit();
    const a = std.mem.bytesAsSlice(i32, l.data().buffers[1].data)[0..100];
    const b = std.mem.bytesAsSlice(i32, r.data().buffers[1].data)[0..100];
    @memset(out.values(), 0);
    for (0..100) |i| if (a[i] < b[i]) bitmap.setBit(out.values(), i);
    var arr = try out.finish();
    defer arr.release();
    try arr.data().validateFull();
    for (0..100) |i| {
        try std.testing.expectEqual(l.data().isNull(i) or r.data().isNull(i), arr.data().isNull(i));
        if (!arr.data().isNull(i)) try std.testing.expectEqual(a[i] < b[i], bitmap.bitIsSet(arr.data().buffers[1].data, i));
    }
}

test "laneMask matches isNull per row (random, sliced, all-null, no-null, tail block)" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(10);
    const rng = prng.random();
    const cases = [_]struct { len: usize, pct: u8, off: usize }{
        .{ .len = 2000, .pct = 10, .off = 0 },
        .{ .len = 2001, .pct = 50, .off = 5 },
        .{ .len = 301, .pct = 90, .off = 13 },
        .{ .len = 300, .pct = 100, .off = 1 },
        .{ .len = 300, .pct = 0, .off = 7 },
        .{ .len = 3, .pct = 10, .off = 3 },
    };
    for (cases) |c| {
        var full = try randomInt32(allocator, rng, c.len + c.off, c.pct);
        defer full.release();
        var arr = try full.slice(c.off, c.len);
        defer arr.release();
        const bits = validityBits(arr.data());
        if (c.pct == 0) try std.testing.expect(bits == null);
        var i: usize = 0;
        while (i < c.len) : (i += 8) {
            const mask: [8]bool = laneMask(arr.data(), bits, i);
            for (mask, 0..) |m, k| {
                const expected = if (i + k < c.len) !arr.data().isNull(i + k) else false;
                try std.testing.expectEqual(expected, m);
            }
        }
    }
}
