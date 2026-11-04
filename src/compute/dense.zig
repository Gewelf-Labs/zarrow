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
            return initValidity(allocator, lhs.length, lhs, rhs, out_type);
        }

        /// Allocate the output for `len` rows; a null input stands for a non-null scalar operand
        /// (it adds no nulls). Every input that is present must have `len` rows.
        pub fn initValidity(allocator: std.mem.Allocator, len: usize, lhs: ?*const ArrayData, rhs: ?*const ArrayData, out_type: DataType) error{ OutOfMemory, InvalidInput }!Self {
            const inputs = [2]?*const ArrayData{ lhs, rhs };
            for (inputs) |maybe| if (maybe) |input| if (input.length != len) return error.InvalidInput;
            var values_buf = try OwnedBuffer.initUninit(allocator, if (is_bits) bitmap.byteLength(len) else len * @sizeOf(T));
            errdefer values_buf.deinit();

            var validity: ?OwnedBuffer = null;
            errdefer if (validity) |*v| v.deinit();
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

// ------------------------------------------------------------------- element-wise kernel driver

const core = @import("core.zig");
const PrimitiveArray = @import("../array/primitive_array.zig").PrimitiveArray;

const ChunkedArray = core.ChunkedArray;
const Datum = core.Datum;
const ExecChunkValue = core.ExecChunkValue;
const ExecContext = core.ExecContext;
const KernelError = core.KernelError;
const Options = core.Options;
const Scalar = core.Scalar;
const ScalarValue = core.ScalarValue;
const TypeTag = std.meta.Tag(DataType);

/// Zig element types of the numeric kernels (int8..int64, uint8..uint64, float32, float64).
pub const numeric_types = [_]type{ i8, i16, i32, i64, u8, u16, u32, u64, f32, f64 };

/// Return the Arrow type of the Zig primitive `T`.
pub fn dataTypeOf(comptime T: type) DataType {
    return switch (T) {
        bool => .{ .bool = {} },
        i8 => .{ .int8 = {} },
        i16 => .{ .int16 = {} },
        i32 => .{ .int32 = {} },
        i64 => .{ .int64 = {} },
        u8 => .{ .uint8 = {} },
        u16 => .{ .uint16 = {} },
        u32 => .{ .uint32 = {} },
        u64 => .{ .uint64 = {} },
        f32 => .{ .float = {} },
        f64 => .{ .double = {} },
        else => @compileError("no Arrow type for " ++ @typeName(T)),
    };
}

/// `ScalarValue` field holding a `T` (`i32`, `f64`, `bool`, ...).
fn scalarField(comptime T: type) []const u8 {
    return switch (T) {
        f32 => "f32",
        f64 => "f64",
        else => @typeName(T),
    };
}

/// Return the scalar value holding the Zig primitive `v`.
pub fn scalarValueOf(comptime T: type, v: T) ScalarValue {
    return @unionInit(ScalarValue, scalarField(T), v);
}

/// Return the value of a non-null scalar as `T`; `InvalidInput` when the payload does not hold a `T`.
pub fn scalarAs(comptime T: type, s: Scalar) KernelError!T {
    const field = comptime scalarField(T);
    if (std.meta.activeTag(s.value) != @field(std.meta.Tag(ScalarValue), field)) return error.InvalidInput;
    return @field(s.value, field);
}

/// Return a type check accepting exactly `arity` arguments of type `tag` (array, chunked or scalar).
pub fn argsOfType(comptime tag: TypeTag, comptime arity: usize) *const fn ([]const Datum) bool {
    return &struct {
        fn check(args: []const Datum) bool {
            if (args.len != arity) return false;
            for (args) |arg| if (std.meta.activeTag(arg.dataType()) != tag) return false;
            return true;
        }
    }.check;
}

/// Return a result type callback that always answers `dt`.
pub fn resultTypeOf(comptime dt: DataType) *const fn ([]const Datum, Options) KernelError!DataType {
    return &struct {
        fn infer(_: []const Datum, _: Options) KernelError!DataType {
            return dt;
        }
    }.infer;
}

/// Accept only `Options.none` (kernels that read no options).
pub fn noOptions(options: Options) bool {
    return options == .none;
}

/// Return the array of an exec chunk value, or null for a scalar.
pub fn arrayData(v: ExecChunkValue) ?*const ArrayData {
    return switch (v) {
        .array => |arr| arr.data(),
        .scalar => null,
    };
}

/// A numeric operand: the typed values of an array chunk (offset applied) or a non-null scalar.
pub fn Operand(comptime T: type) type {
    return union(enum) { slice: []const T, scalar: T };
}

/// Return the numeric operand of an exec chunk value (`InvalidInput` on a malformed buffer or scalar).
pub fn operand(comptime T: type, v: ExecChunkValue) KernelError!Operand(T) {
    return switch (v) {
        .array => |arr| .{ .slice = (PrimitiveArray(T){ .data = arr.data() }).values() catch return error.InvalidInput },
        .scalar => |s| .{ .scalar = try scalarAs(T, s) },
    };
}

fn Elem(comptime V: type) type {
    return if (@typeInfo(V) == .pointer) std.meta.Child(V) else V;
}

/// Return row `i` of an operand payload: `v[i]` for a slice, `v` for a broadcast scalar.
pub inline fn at(v: anytype, i: usize) Elem(@TypeOf(v)) {
    return if (comptime @typeInfo(@TypeOf(v)) == .pointer) v[i] else v;
}

/// Kernel callback that computes one output chunk from aligned input chunks of `len` rows.
pub const ChunkFn = fn (ctx: *ExecContext, values: []const ExecChunkValue, len: usize) KernelError!ArrayRef;

/// Run a unary or binary element-wise kernel chunk by chunk: all-scalar inputs give a scalar, any
/// chunked input gives a chunked output (chunk boundaries are the union of the inputs'), otherwise
/// an array. Every array chunk passes `validateLayout` first, so `chunk_fn` may index its buffers.
/// With `null_scalars_propagate` (null intersection) a null scalar input yields an all-null chunk
/// without calling `chunk_fn`.
pub fn execElementwise(
    ctx: *ExecContext,
    args: []const Datum,
    out_type: DataType,
    comptime null_scalars_propagate: bool,
    comptime chunk_fn: ChunkFn,
) KernelError!Datum {
    if (args.len == 0 or args.len > 2) return error.InvalidArity;
    const len = try core.inferNaryExecLen(args);
    var any_chunked = false;
    var all_scalar = true;
    for (args) |arg| {
        if (arg == .chunked) any_chunked = true;
        if (arg != .scalar) all_scalar = false;
    }
    if (!any_chunked) {
        var values: [2]ExecChunkValue = undefined;
        for (args, 0..) |arg, i| values[i] = switch (arg) {
            .array => |arr| .{ .array = arr },
            .scalar => |s| .{ .scalar = s },
            .chunked => unreachable,
        };
        var out = try runChunk(ctx, values[0..args.len], len, out_type, null_scalars_propagate, chunk_fn);
        if (!all_scalar) return .{ .array = out };
        defer out.release();
        return .{ .scalar = try scalarAt0(out) };
    }

    var it = try core.NaryExecChunkIterator.init(ctx.allocator, args);
    defer it.deinit();
    var outs: std.ArrayList(ArrayRef) = .empty;
    defer {
        for (outs.items) |*o| o.release();
        outs.deinit(ctx.allocator);
    }
    while (try it.next()) |next| {
        var chunk = next;
        defer chunk.deinit();
        var out = try runChunk(ctx, chunk.values, chunk.len, out_type, null_scalars_propagate, chunk_fn);
        outs.append(ctx.allocator, out) catch {
            out.release();
            return error.OutOfMemory;
        };
    }
    const chunked = ChunkedArray.init(ctx.allocator, out_type, outs.items) catch |err| return switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        else => error.InvalidInput,
    };
    return .{ .chunked = chunked };
}

fn runChunk(
    ctx: *ExecContext,
    values: []const ExecChunkValue,
    len: usize,
    out_type: DataType,
    comptime null_scalars_propagate: bool,
    comptime chunk_fn: ChunkFn,
) KernelError!ArrayRef {
    for (values) |v| if (v == .array) v.array.data().validateLayout() catch return error.InvalidInput;
    if (null_scalars_propagate) for (values) |v| {
        if (v == .scalar and v.scalar.isNull()) return (try core.datumBuildNullLikeWithAllocator(ctx.allocator, out_type, len)).array;
    };
    return chunk_fn(ctx, values, len);
}

/// Return row 0 of a numeric or bool kernel output as a scalar.
fn scalarAt0(arr: ArrayRef) KernelError!Scalar {
    const data = arr.data();
    if (data.length != 1) return error.InvalidInput;
    if (data.isNull(0)) return Scalar.init(data.data_type, .null);
    if (data.data_type == .bool) return Scalar.init(data.data_type, .{ .bool = bitmap.bitIsSet(data.buffers[1].data, data.offset) });
    inline for (numeric_types) |T| {
        if (std.meta.activeTag(data.data_type) == comptime std.meta.activeTag(dataTypeOf(T))) {
            const values = (PrimitiveArray(T){ .data = data }).values() catch return error.InvalidInput;
            return Scalar.init(data.data_type, scalarValueOf(T, values[0]));
        }
    }
    return error.UnsupportedType;
}

/// Packed bits read 8 rows at a time: an array buffer at a bit offset, or a broadcast byte.
pub const Bits = struct {
    /// Null for a broadcast operand (`splat` in every byte).
    bytes: ?[]const u8 = null,
    /// Bit position of row 0 in `bytes`.
    offset: usize = 0,
    splat: u8 = 0xFF,

    /// Return rows `[8 * j, 8 * j + 8)` as one byte (bit 0 = first row).
    pub inline fn byte(self: Bits, j: usize) u8 {
        const b = self.bytes orelse return self.splat;
        return bitmap.byteAtBit(b, self.offset + j * 8);
    }

    /// Return the value bits of a bool array.
    pub fn values(data: *const ArrayData) Bits {
        return .{ .bytes = data.buffers[1].data, .offset = data.offset };
    }

    /// Return the validity of `data` (all valid when it has no null; all null for the null type).
    pub fn validity(data: *const ArrayData) Bits {
        if (datatype.storageDataType(data.data_type) == .null) return .{ .splat = 0 };
        const bits = validityBits(data) orelse return .{};
        return .{ .bytes = bits, .offset = data.offset };
    }

    /// Return all-set bits for `true`, all-clear bits for `false`.
    pub fn of(b: bool) Bits {
        return .{ .splat = if (b) 0xFF else 0 };
    }
};

/// Set `out[j] = f(operand bytes j)` for the `out.len` bytes of a packed bool output of `len`
/// rows; bits past `len` in the last byte are cleared.
pub fn mapBits(comptime N: usize, out: []u8, ops: [N]Bits, len: usize, comptime f: fn ([N]u8) u8) void {
    var byte_aligned = true;
    for (ops) |o| {
        if (o.bytes != null and o.offset % 8 != 0) byte_aligned = false;
    }
    if (byte_aligned) {
        var bases: [N]usize = undefined;
        for (ops, &bases) |o, *b| b.* = o.offset / 8;
        for (out, 0..) |*d, j| {
            var in: [N]u8 = undefined;
            inline for (0..N) |k| in[k] = if (ops[k].bytes) |b| b[bases[k] + j] else ops[k].splat;
            d.* = f(in);
        }
    } else for (out, 0..) |*d, j| {
        var in: [N]u8 = undefined;
        inline for (0..N) |k| in[k] = ops[k].byte(j);
        d.* = f(in);
    }
    if (len % 8 != 0) out[out.len - 1] &= (@as(u8, 1) << @as(u3, @intCast(len % 8))) - 1;
}

/// Return a bool array of `len` rows built from packed value bits and an optional validity
/// bitmap; takes ownership of both buffers (also on error).
pub fn boolArray(allocator: std.mem.Allocator, len: usize, values_buf: OwnedBuffer, validity: ?OwnedBuffer) KernelError!ArrayRef {
    var out = DenseBinaryOutput(bool){ .allocator = allocator, .len = len, .values_buf = values_buf, .validity = validity, .out_type = .{ .bool = {} } };
    errdefer out.deinit();
    return out.finish() catch error.OutOfMemory;
}

/// Return an uninitialized packed-bit buffer for `len` rows.
pub fn bitsBuffer(allocator: std.mem.Allocator, len: usize) KernelError!OwnedBuffer {
    return OwnedBuffer.initUninit(allocator, bitmap.byteLength(len)) catch error.OutOfMemory;
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
