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

/// Return the AND of the validity bitmaps of `inputs` over `len` rows (bit 0 = row 0, padding
/// bits clear), or null when no input has a null. A null entry stands for a non-null scalar
/// operand (it adds no nulls); every input that is present must have `len` rows.
pub fn combinedValidity(allocator: std.mem.Allocator, len: usize, inputs: []const ?*const ArrayData) error{ OutOfMemory, InvalidInput }!?OwnedBuffer {
    for (inputs) |maybe| if (maybe) |input| if (input.length != len) return error.InvalidInput;
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
    return validity;
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
            var validity = try combinedValidity(allocator, len, &.{ lhs, rhs });
            errdefer if (validity) |*v| v.deinit();
            const values_buf = try OwnedBuffer.initUninit(allocator, if (is_bits) bitmap.byteLength(len) else len * @sizeOf(T));
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
/// Like `ChunkFn`, with the options of the call.
pub const ChunkOptionsFn = fn (ctx: *ExecContext, values: []const ExecChunkValue, len: usize, options: Options) KernelError!ArrayRef;

/// Run an element-wise kernel chunk by chunk: all-scalar inputs give a scalar, any chunked input
/// gives a chunked output (chunk boundaries are the union of the inputs'), otherwise an array.
/// Every array chunk passes `validateLayout` first, so `chunk_fn` may index its buffers. With
/// `null_scalars_propagate` (null intersection) a null scalar input yields an all-null chunk
/// without calling `chunk_fn`.
pub fn execElementwise(
    ctx: *ExecContext,
    args: []const Datum,
    out_type: DataType,
    comptime null_scalars_propagate: bool,
    comptime chunk_fn: ChunkFn,
) KernelError!Datum {
    return execElementwiseWith(ctx, args, Options.noneValue(), out_type, null_scalars_propagate, struct {
        fn f(c: *ExecContext, values: []const ExecChunkValue, len: usize, _: Options) KernelError!ArrayRef {
            return chunk_fn(c, values, len);
        }
    }.f);
}

/// `execElementwise` for any number of arguments, handing `options` to every chunk.
pub fn execElementwiseWith(
    ctx: *ExecContext,
    args: []const Datum,
    options: Options,
    out_type: DataType,
    comptime null_scalars_propagate: bool,
    comptime chunk_fn: ChunkOptionsFn,
) KernelError!Datum {
    if (args.len == 0) return error.InvalidArity;
    const len = try core.inferNaryExecLen(args);
    var any_chunked = false;
    var all_scalar = true;
    for (args) |arg| {
        if (arg == .chunked) any_chunked = true;
        if (arg != .scalar) all_scalar = false;
    }
    if (!any_chunked) {
        var stack: [4]ExecChunkValue = undefined;
        const values = if (args.len <= stack.len) stack[0..args.len] else ctx.allocator.alloc(ExecChunkValue, args.len) catch return error.OutOfMemory;
        defer if (args.len > stack.len) ctx.allocator.free(values);
        for (args, values) |arg, *v| v.* = switch (arg) {
            .array => |arr| .{ .array = arr },
            .scalar => |s| .{ .scalar = s },
            .chunked => unreachable,
        };
        var out = try runChunk(ctx, values, len, options, out_type, null_scalars_propagate, chunk_fn);
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
        var out = try runChunk(ctx, chunk.values, chunk.len, options, out_type, null_scalars_propagate, chunk_fn);
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
    options: Options,
    out_type: DataType,
    comptime null_scalars_propagate: bool,
    comptime chunk_fn: ChunkOptionsFn,
) KernelError!ArrayRef {
    for (values) |v| if (v == .array) v.array.data().validateLayout() catch return error.InvalidInput;
    if (null_scalars_propagate) for (values) |v| {
        if (v == .scalar and v.scalar.isNull()) return nullArray(ctx.allocator, out_type, len);
    };
    return chunk_fn(ctx, values, len, options);
}

/// Return an all-null array of `len` rows of a primitive, bool, string / binary or struct type.
pub fn nullArray(allocator: std.mem.Allocator, out_type: DataType, len: usize) KernelError!ArrayRef {
    switch (datatype.storageDataType(out_type)) {
        .string, .large_string, .string_view, .binary, .large_binary, .binary_view => {
            var out = try VarBinaryOutput.init(allocator, out_type, len, try zeroBits(allocator, len), 0);
            errdefer out.deinit();
            for (0..len) |_| try out.commit(0);
            return out.finish();
        },
        .bool => {
            var values_buf = try zeroBits(allocator, len);
            const validity = zeroBits(allocator, len) catch |err| {
                values_buf.deinit();
                return err;
            };
            return boolArray(allocator, len, values_buf, validity);
        },
        .struct_, .list, .large_list, .fixed_size_list, .null => return (try core.datumBuildNullLikeWithAllocator(allocator, out_type, len)).array,
        .dictionary, .extension => return error.UnsupportedType,
        else => {
            const width = (out_type.bitWidth() orelse return error.UnsupportedType) / 8;
            var validity = try zeroBits(allocator, len);
            var values_buf = OwnedBuffer.init(allocator, len * width) catch {
                validity.deinit();
                return error.OutOfMemory;
            };
            return finishFixed(allocator, out_type, len, &values_buf, validity);
        },
    }
}

/// Return a zeroed packed-bit buffer for `len` rows (an all-null validity bitmap).
fn zeroBits(allocator: std.mem.Allocator, len: usize) KernelError!OwnedBuffer {
    return OwnedBuffer.init(allocator, bitmap.byteLength(len)) catch error.OutOfMemory;
}

/// Hand a fixed-width values buffer of `len` rows and a validity bitmap (null: no nulls) to a new
/// array of `out_type`; takes ownership of both buffers (also on error).
pub fn finishFixed(allocator: std.mem.Allocator, out_type: DataType, len: usize, values_buf: *OwnedBuffer, validity: ?OwnedBuffer) KernelError!ArrayRef {
    const width = (out_type.bitWidth() orelse 8) / 8;
    var v = validity;
    errdefer if (v) |*b| b.deinit();
    errdefer values_buf.deinit();
    var buffers: [2]SharedBuffer = .{ .empty, .empty };
    var null_count: usize = 0;
    if (v) |*b| {
        null_count = bitmap.countUnsetBit(b.data, len);
        buffers[0] = b.toShared(bitmap.byteLength(len)) catch return error.OutOfMemory;
        v = null;
    }
    errdefer buffers[0].release();
    buffers[1] = values_buf.toShared(len * width) catch return error.OutOfMemory;
    errdefer buffers[1].release();
    return ArrayRef.fromOwnedInlineBuffers(allocator, .{ .data_type = out_type, .length = len, .null_count = null_count, .buffers = &buffers }) catch error.OutOfMemory;
}

/// Return row 0 of a kernel output as a scalar. A string / binary scalar borrows its bytes from
/// `payload`, which it retains; a struct or list scalar is a nested scalar over the row.
fn scalarAt0(arr: ArrayRef) KernelError!Scalar {
    const data = arr.data();
    if (data.length != 1) return error.InvalidInput;
    if (data.isNull(0)) return Scalar.init(data.data_type, .null);
    switch (datatype.storageDataType(data.data_type)) {
        .bool => return Scalar.init(data.data_type, .{ .bool = bitmap.bitIsSet(data.buffers[1].data, data.offset) }),
        .string, .large_string, .string_view, .binary, .large_binary, .binary_view => {
            const bytes = switch (try StringOperand.init(.{ .array = arr })) {
                inline else => |op| op.get(0),
            };
            const text = switch (datatype.storageDataType(data.data_type)) {
                .string, .large_string, .string_view => true,
                else => false,
            };
            var s = Scalar.init(data.data_type, if (text) .{ .string = bytes } else .{ .binary = bytes });
            s.payload = arr.retain();
            return s;
        },
        .struct_, .list => return Scalar.initNested(data.data_type, arr),
        inline .date32, .date64, .timestamp, .time32, .time64, .duration, .interval_months, .interval_day_time, .interval_month_day_nano, .decimal32, .decimal64, .decimal128, .decimal256 => |_, tag| {
            const T = switch (tag) {
                .date32, .time32, .interval_months, .decimal32 => i32,
                .interval_month_day_nano, .decimal128 => i128,
                .decimal256 => i256,
                else => i64,
            };
            const values = (PrimitiveArray(T){ .data = data }).values() catch return error.InvalidInput;
            return Scalar.init(data.data_type, @unionInit(ScalarValue, @tagName(tag), values[0]));
        },
        else => {},
    }
    inline for (numeric_types) |T| {
        if (std.meta.activeTag(data.data_type) == comptime std.meta.activeTag(dataTypeOf(T))) {
            const values = (PrimitiveArray(T){ .data = data }).values() catch return error.InvalidInput;
            return Scalar.init(data.data_type, scalarValueOf(T, values[0]));
        }
    }
    return error.UnsupportedType;
}

/// Return the array of an exec chunk value: the array itself (retained), or a one-row array
/// holding a scalar of a primitive, bool, temporal, decimal or string / binary type. The caller
/// releases it.
pub fn chunkArray(allocator: std.mem.Allocator, v: ExecChunkValue) KernelError!ArrayRef {
    const s = switch (v) {
        .array => |arr| return arr.retain(),
        .scalar => |s| s,
    };
    if (s.payload) |p| return if (p.data().length == 1) p.retain() else error.InvalidInput;
    if (s.isNull()) return nullArray(allocator, s.data_type, 1);
    switch (datatype.storageDataType(s.data_type)) {
        .string, .large_string, .string_view, .binary, .large_binary, .binary_view => {
            const bytes = switch (s.value) {
                .string, .binary => |b| b,
                else => return error.InvalidInput,
            };
            var out = try VarBinaryOutput.init(allocator, s.data_type, 1, null, bytes.len);
            errdefer out.deinit();
            try out.append(bytes);
            return out.finish();
        },
        .bool => {
            const b = try scalarAs(bool, s);
            var bits = try bitsBuffer(allocator, 1);
            bits.data[0] = @intFromBool(b);
            return boolArray(allocator, 1, bits, null);
        },
        else => {},
    }
    const width = (s.data_type.bitWidth() orelse return error.UnsupportedType) / 8;
    // The payload must be the field of the scalar's own type (an int32 scalar holds `.i32`).
    const expected: std.meta.Tag(ScalarValue) = switch (datatype.storageDataType(s.data_type)) {
        .int8 => .i8,
        .int16 => .i16,
        .int32 => .i32,
        .int64 => .i64,
        .uint8 => .u8,
        .uint16 => .u16,
        .uint32 => .u32,
        .uint64 => .u64,
        .float => .f32,
        .double => .f64,
        .date32 => .date32,
        .date64 => .date64,
        .timestamp => .timestamp,
        .time32 => .time32,
        .time64 => .time64,
        .duration => .duration,
        .decimal32 => .decimal32,
        .decimal64 => .decimal64,
        .decimal128 => .decimal128,
        .decimal256 => .decimal256,
        else => return error.UnsupportedType,
    };
    if (std.meta.activeTag(s.value) != expected) return error.InvalidInput;
    var values_buf = OwnedBuffer.init(allocator, width) catch return error.OutOfMemory;
    switch (s.value) {
        inline .i8, .i16, .i32, .i64, .u8, .u16, .u32, .u64, .f32, .f64, .date32, .date64, .timestamp, .time32, .time64, .duration, .decimal32, .decimal64, .decimal128, .decimal256 => |x| @memcpy(values_buf.data[0..width], std.mem.asBytes(&x)),
        else => unreachable,
    }
    return finishFixed(allocator, s.data_type, 1, &values_buf, null);
}

// ------------------------------------------------------------------- variable-width rows

/// Rows of a string / binary operand: a scalar, or an array of string / large_string /
/// string_view (or the binary twins) that passed `validateLayout` (offsets and views are in
/// bounds, so rows are sliced without further checks).
pub const StringOperand = union(enum) {
    scalar: ScalarBytes,
    offsets32: OffsetBytes(i32),
    offsets64: OffsetBytes(i64),
    views: ViewBytes,

    pub fn init(v: ExecChunkValue) KernelError!StringOperand {
        const data = switch (v) {
            .scalar => |s| return switch (s.value) {
                .string, .binary => |bytes| .{ .scalar = .{ .bytes = bytes } },
                else => error.InvalidInput,
            },
            .array => |arr| arr.data(),
        };
        switch (datatype.storageDataType(data.data_type)) {
            inline .string, .large_string, .binary, .large_binary => |_, tag| {
                const O = if (tag == .string or tag == .binary) i32 else i64;
                const all = data.buffers[1].typedSlice(O) catch return error.InvalidInput;
                const op: OffsetBytes(O) = .{ .offsets = all[data.offset..][0 .. data.length + 1], .data = data.buffers[2].data };
                return if (O == i32) .{ .offsets32 = op } else .{ .offsets64 = op };
            },
            .string_view, .binary_view => return .{ .views = ViewBytes.init(data) },
            else => return error.UnsupportedType,
        }
    }
};

pub const ScalarBytes = struct {
    bytes: []const u8,
    pub inline fn get(self: ScalarBytes, _: usize) []const u8 {
        return self.bytes;
    }
};

pub fn OffsetBytes(comptime O: type) type {
    return struct {
        offsets: []const O,
        data: []const u8,
        pub inline fn get(self: @This(), i: usize) []const u8 {
            return self.data[@intCast(self.offsets[i])..@intCast(self.offsets[i + 1])];
        }
    };
}

pub const ViewBytes = struct {
    views: []const u8,
    /// The variadic data buffers (`validateLayout` checked every out-of-line view against them),
    /// or one empty buffer when there is none.
    buffers: []const SharedBuffer,
    /// Start of the only data buffer (null with several): one load less per out-of-line row.
    single: ?[*]const u8,

    fn init(data: *const ArrayData) ViewBytes {
        const buffers = data.buffers[2..];
        return .{
            .views = data.buffers[1].data[data.offset * 16 ..][0 .. data.length * 16],
            .buffers = buffers,
            .single = if (buffers.len == 1) buffers[0].data.ptr else null,
        };
    }

    pub inline fn get(self: ViewBytes, i: usize) []const u8 {
        const view = self.views[i * 16 ..][0..16];
        const len = std.mem.readInt(u32, view[0..4], .little);
        if (len <= 12) return view[4..][0..len];
        const off = std.mem.readInt(u32, view[12..16], .little);
        if (self.single) |base| return base[off..][0..len];
        const buf = std.mem.readInt(u32, view[8..12], .little);
        return self.buffers[buf].data[off..][0..len];
    }
};

/// Total bytes of rows `[0, len)` of an array operand (an upper bound of a transform's output).
pub fn totalBytes(op: StringOperand, len: usize) usize {
    return switch (op) {
        .scalar => |s| s.bytes.len * len,
        inline .offsets32, .offsets64 => |o| @intCast(o.offsets[len] - o.offsets[0]),
        .views => |o| blk: {
            var n: usize = 0;
            for (0..len) |i| n += std.mem.readInt(u32, o.views[i * 16 ..][0..4], .little);
            break :blk n;
        },
    };
}

/// Builder of a string / large_string / string_view (or binary twin) output, row by row in
/// order: `reserve` a slot, fill it, `commit` the bytes used (0 for a null row). Offsets past
/// `maxInt(i32)` (string / binary) or a data buffer past 4 GiB (views) are `Overflow`.
pub const VarBinaryOutput = struct {
    allocator: std.mem.Allocator,
    out_type: DataType,
    len: usize,
    validity: ?OwnedBuffer,
    /// Offsets (`len + 1` entries) or 16-byte views (`len` entries).
    index: OwnedBuffer,
    data: OwnedBuffer,
    kind: Layout,
    pos: usize = 0,
    row: usize = 0,

    const Self = @This();
    const Layout = enum { offsets32, offsets64, views };

    inline fn layout(self: *const Self) Layout {
        return self.kind;
    }

    /// Takes ownership of `validity` (also on error); `data_capacity` is the expected byte total.
    pub fn init(allocator: std.mem.Allocator, out_type: DataType, len: usize, validity: ?OwnedBuffer, data_capacity: usize) KernelError!Self {
        var v = validity;
        errdefer if (v) |*b| b.deinit();
        const kind: Layout = switch (datatype.storageDataType(out_type)) {
            .string, .binary => .offsets32,
            .large_string, .large_binary => .offsets64,
            .string_view, .binary_view => .views,
            else => return error.UnsupportedType,
        };
        const index_bytes: usize = switch (kind) {
            .offsets32 => (len + 1) * 4,
            .offsets64 => (len + 1) * 8,
            .views => len * 16,
        };
        var index = OwnedBuffer.initUninit(allocator, index_bytes) catch return error.OutOfMemory;
        errdefer index.deinit();
        const data = OwnedBuffer.initUninit(allocator, data_capacity) catch return error.OutOfMemory;
        var self: Self = .{ .allocator = allocator, .out_type = out_type, .len = len, .validity = v, .index = index, .data = data, .kind = kind };
        switch (self.layout()) {
            .offsets32 => self.index.typedSlice(i32)[0] = 0,
            .offsets64 => self.index.typedSlice(i64)[0] = 0,
            .views => {},
        }
        return self;
    }

    pub fn deinit(self: *Self) void {
        if (self.validity) |*v| v.deinit();
        self.index.deinit();
        self.data.deinit();
    }

    /// Return a writable slot of `n` bytes for the next row.
    pub fn reserve(self: *Self, n: usize) KernelError![]u8 {
        const need = std.math.add(usize, self.pos, n) catch return error.Overflow;
        if (need > self.data.data.len) self.data.resize(@max(need, self.data.data.len * 2)) catch return error.OutOfMemory;
        return self.data.data[self.pos..need];
    }

    /// Close the next row with the first `n` bytes of its slot.
    pub fn commit(self: *Self, n: usize) KernelError!void {
        std.debug.assert(self.row < self.len);
        const end = self.pos + n;
        switch (self.layout()) {
            .offsets32 => self.index.typedSlice(i32)[self.row + 1] = std.math.cast(i32, end) orelse return error.Overflow,
            .offsets64 => self.index.typedSlice(i64)[self.row + 1] = std.math.cast(i64, end) orelse return error.Overflow,
            .views => {
                const view = self.index.data[self.row * 16 ..][0..16];
                @memset(view, 0);
                std.mem.writeInt(u32, view[0..4], @intCast(n), .little);
                if (n <= 12) {
                    if (n > 0) @memcpy(view[4..][0..n], self.data.data[self.pos..end]);
                    self.row += 1;
                    return;
                }
                @memcpy(view[4..8], self.data.data[self.pos..][0..4]);
                std.mem.writeInt(u32, view[12..16], std.math.cast(u32, self.pos) orelse return error.Overflow, .little);
                _ = std.math.cast(u32, end) orelse return error.Overflow;
            },
        }
        self.pos = end;
        self.row += 1;
    }

    /// Close every row at once over bytes already written at the start of the data buffer: row
    /// `i` spans `[offsets[i], offsets[i + 1])` rebased to `offsets[0]` (offsets layouts only).
    pub fn bulkOffsets(self: *Self, offsets: anytype) KernelError!void {
        std.debug.assert(self.row == 0 and self.pos == 0 and offsets.len == self.len + 1);
        const base = offsets[0];
        switch (self.layout()) {
            inline .offsets32, .offsets64 => |l| {
                const O = if (l == .offsets32) i32 else i64;
                for (self.index.typedSlice(O)[0..offsets.len], offsets) |*d, s| d.* = std.math.cast(O, s - base) orelse return error.Overflow;
            },
            .views => unreachable,
        }
        self.pos = @intCast(offsets[self.len] - base);
        self.row = self.len;
    }

    /// The `len + 1` offsets of an offsets layout (`O` = i32 or i64 by layout; entry 0 is 0), for
    /// a caller that writes every row at once and then calls `closeRows`.
    pub fn offsetSlots(self: *Self, comptime O: type) []O {
        std.debug.assert(self.layout() == if (O == i32) Layout.offsets32 else Layout.offsets64);
        return self.index.typedSlice(O)[0 .. self.len + 1];
    }

    /// Close every row after the caller wrote `bytes` bytes at the start of the data buffer
    /// (through `reserve`) and filled `offsetSlots` (offsets layouts only).
    pub fn closeRows(self: *Self, bytes: usize) void {
        std.debug.assert(self.row == 0 and self.pos == 0 and self.layout() != .views and bytes <= self.data.data.len);
        self.pos = bytes;
        self.row = self.len;
    }

    /// Copy `bytes` into the next row.
    pub fn append(self: *Self, bytes: []const u8) KernelError!void {
        copyBytes(try self.reserve(bytes.len), bytes);
        try self.commit(bytes.len);
    }

    /// Hand the buffers to a new array (every row must have been committed).
    pub fn finish(self: *Self) KernelError!ArrayRef {
        std.debug.assert(self.row == self.len);
        errdefer self.deinit();
        var buffers: [3]SharedBuffer = .{ .empty, .empty, .empty };
        var count: usize = 3;
        var null_count: usize = 0;
        errdefer for (buffers[0..count]) |*b| b.release();
        if (self.validity) |*v| {
            null_count = bitmap.countUnsetBit(v.data, self.len);
            buffers[0] = v.toShared(bitmap.byteLength(self.len)) catch return error.OutOfMemory;
            self.validity = null;
        }
        const index_bytes = switch (self.layout()) {
            .offsets32 => (self.len + 1) * 4,
            .offsets64 => (self.len + 1) * 8,
            .views => self.len * 16,
        };
        buffers[1] = self.index.toShared(index_bytes) catch return error.OutOfMemory;
        if (self.layout() == .views and self.pos == 0) {
            count = 2;
            self.data.deinit();
        } else {
            buffers[2] = self.data.toShared(self.pos) catch return error.OutOfMemory;
        }
        return ArrayRef.fromOwnedInlineBuffers(self.allocator, .{
            .data_type = self.out_type,
            .length = self.len,
            .null_count = null_count,
            .buffers = buffers[0..count],
        }) catch error.OutOfMemory;
    }
};

/// `@memcpy(dst[0..src.len], src)` without a call for short rows: up to 16 bytes move as two
/// overlapping loads and stores (the common size of a string row or part).
pub inline fn copyBytes(dst: []u8, src: []const u8) void {
    const n = src.len;
    if (n >= 8 and n <= 16) {
        std.mem.writeInt(u64, dst[0..8], std.mem.readInt(u64, src[0..8], .little), .little);
        std.mem.writeInt(u64, dst[n - 8 ..][0..8], std.mem.readInt(u64, src[n - 8 ..][0..8], .little), .little);
    } else if (n >= 4 and n < 8) {
        std.mem.writeInt(u32, dst[0..4], std.mem.readInt(u32, src[0..4], .little), .little);
        std.mem.writeInt(u32, dst[n - 4 ..][0..4], std.mem.readInt(u32, src[n - 4 ..][0..4], .little), .little);
    } else if (n < 4) {
        for (dst[0..n], src) |*d, c| d.* = c;
    } else @memcpy(dst[0..n], src);
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
