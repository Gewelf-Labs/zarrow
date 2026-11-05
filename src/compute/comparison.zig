//! Comparison kernels (Arrow C++ "Comparisons"): equal, not_equal, less, less_equal, greater,
//! greater_equal for int8..int64, uint8..uint64, float32, float64, bool, the string and binary
//! layouts, fixed_size_binary, date32, date64, timestamp, time32, time64, duration and the decimal
//! types. The output is a bool array (scalar for scalar inputs); null handling is intersection.
//! Strings and binaries compare by bytes; floats follow IEEE 754 (NaN is unequal to everything,
//! `not_equal` is true for it); `false < true`; temporal values and decimals compare their stored
//! integers. Operands of different types are unified as in Arrow's comparison `DispatchBest`:
//! - numbers are cast to their common numeric type (`registry.commonNumericType`);
//! - two string / binary layouts (string, large_string, string_view, binary, large_binary,
//!   binary_view, fixed_size_binary) compare their bytes directly, with no cast (Arrow casts the
//!   offset layouts to a common one and has no kernel for a view mixed with another layout; two
//!   fixed_size_binary of different widths stay `NoMatchingKernel`, as Arrow);
//! - decimals of any widths, precisions and scales, and a decimal with an integer, compare by
//!   exact value (the operand of the smaller scale is scaled up in 256-bit arithmetic; a product
//!   past that range is beyond every value of the other side). Arrow rescales to a common decimal
//!   type and rejects a common precision past 38 / 76 and negative scales; here both compare;
//! - a decimal with a float casts both to float64 (`registry.commonDecimalFloatType`);
//! - temporal operands of different types are cast to `registry.commonTemporalType` (Arrow's
//!   `CommonTemporal`: a timestamp of the finer unit, date64 over date32, the finer time or
//!   duration; a value that overflows the finer unit is `InvalidCast`, unsafe wraps).
//!   Timestamps of the same unit compare their instants whatever their timezones, but a naive
//!   timestamp never compares with a zoned one (`NoMatchingKernel`; Arrow: Invalid), and
//!   timestamps of different units in different timezones have no common type (as Arrow).
//!
//! min_element_wise / max_element_wise (`ElementWiseAggregateOptions.skip_nulls`, default true)
//! take one or more operands of one numeric, temporal, decimal, fixed_size_binary, string or
//! binary type (numbers promoted and temporal types unified like above, as Arrow; decimals and
//! fixed_size_binary must have equal types and string layouts are not mixed, as Arrow) and return
//! the smallest / largest of each row: with `skip_nulls` the null operands are left out (null only
//! when all are), otherwise a null operand makes the row null. A NaN is left out unless every
//! value of the row is NaN; -0.0 is smaller than 0.0.
const std = @import("std");
const bitmap = @import("../bitmap.zig");
const core = @import("core.zig");
const datatype = @import("../datatype.zig");
const decimal = @import("decimal.zig");
const dense = @import("dense.zig");
const PrimitiveArray = @import("../array/primitive_array.zig").PrimitiveArray;
const OwnedBuffer = @import("../buffer.zig").OwnedBuffer;

const ArrayRef = core.ArrayRef;
const DataType = core.DataType;
const Datum = core.Datum;
const ExecChunkValue = core.ExecChunkValue;
const ExecContext = core.ExecContext;
const FunctionRegistry = core.FunctionRegistry;
const Kernel = core.Kernel;
const KernelError = core.KernelError;
const Options = core.Options;
const TypeTag = std.meta.Tag(DataType);

const Op = enum { equal, not_equal, less, less_equal, greater, greater_equal };
const Extreme = enum { min_element_wise, max_element_wise };
const bool_type = DataType{ .bool = {} };
const string_tags = [_]TypeTag{ .string, .large_string, .string_view };
const binary_tags = [_]TypeTag{ .binary, .large_binary, .binary_view };
/// Temporal and decimal types compared by their stored value, with its Zig type.
const temporal_types = .{
    .{ TypeTag.date32, i32 }, .{ TypeTag.date64, i64 }, .{ TypeTag.timestamp, i64 },
    .{ TypeTag.time32, i32 }, .{ TypeTag.time64, i64 }, .{ TypeTag.duration, i64 },
};
const decimal_types = .{
    .{ TypeTag.decimal32, i32 },   .{ TypeTag.decimal64, i64 },
    .{ TypeTag.decimal128, i128 }, .{ TypeTag.decimal256, i256 },
};

/// Register the comparisons and min_element_wise / max_element_wise.
pub fn register(registry: *FunctionRegistry) KernelError!void {
    @setEvalBranchQuota(10_000);
    inline for (comptime std.meta.tags(Op)) |op| {
        inline for (dense.numeric_types) |T| {
            try registry.registerScalarKernel(@tagName(op), kernel(std.meta.activeTag(dense.dataTypeOf(T)), Numeric(T, op).chunk));
        }
        try registry.registerScalarKernel(@tagName(op), kernel(.bool, Boolean(op).chunk));
        inline for (string_tags ++ binary_tags) |tag| {
            try registry.registerScalarKernel(@tagName(op), kernel(tag, Strings(op).chunk));
        }
        inline for (temporal_types ++ decimal_types) |pt| {
            try registry.registerScalarKernel(@tagName(op), kernel(pt[0], Numeric(pt[1], op).chunk));
        }
        try registry.registerScalarKernel(@tagName(op), kernel(.fixed_size_binary, FixedBinary(op).chunk));
        try registry.registerScalarKernel(@tagName(op), mixedKernel(mixedBytes, MixedBytes(op).chunk));
        try registry.registerScalarKernel(@tagName(op), mixedKernel(mixedDecimals, MixedDecimals(op).chunk));
        try registry.setImplicitCast(@tagName(op), .scalar, .comparison);
    }
    inline for (comptime std.meta.tags(Extreme)) |which| {
        inline for (dense.numeric_types) |T| {
            try registry.registerScalarKernel(@tagName(which), extremeKernel(std.meta.activeTag(dense.dataTypeOf(T)), which));
        }
        inline for (temporal_types) |pt| try registry.registerScalarKernel(@tagName(which), extremeKernel(pt[0], which));
        inline for (string_tags ++ binary_tags) |tag| try registry.registerScalarKernel(@tagName(which), extremeKernel(tag, which));
        inline for (decimal_types) |pt| try registry.registerScalarKernel(@tagName(which), extremeKernel(pt[0], which));
        try registry.registerScalarKernel(@tagName(which), extremeKernel(.fixed_size_binary, which));
        try registry.setImplicitCast(@tagName(which), .scalar, .element_wise);
    }
}

/// `count` arguments (at least one when `variadic`) of type `tag`, all equal (same unit, width,
/// precision and scale; timestamps may differ in timezone when all have one or none has one).
fn sameTypes(comptime tag: TypeTag, comptime count: usize, comptime variadic: bool) core.TypeCheckFn {
    return &struct {
        fn check(args: []const Datum) bool {
            if (if (variadic) args.len < count else args.len != count) return false;
            for (args) |arg| if (std.meta.activeTag(arg.dataType()) != tag or !sameType(arg.dataType(), args[0].dataType())) return false;
            return true;
        }
    }.check;
}

fn sameType(a: DataType, b: DataType) bool {
    if (a == .timestamp and b == .timestamp) {
        return a.timestamp.unit == b.timestamp.unit and (a.timestamp.timezone == null) == (b.timestamp.timezone == null);
    }
    return a.eql(b);
}

fn isBytes(dt: DataType) bool {
    return switch (dt) {
        .string, .large_string, .string_view, .binary, .large_binary, .binary_view, .fixed_size_binary => true,
        else => false,
    };
}

/// Two string / binary layouts of different types (two fixed_size_binary never: Arrow does not
/// compare different widths).
fn mixedBytes(args: []const Datum) bool {
    if (args.len != 2) return false;
    const a = args[0].dataType();
    const b = args[1].dataType();
    return isBytes(a) and isBytes(b) and !a.eql(b) and !(a == .fixed_size_binary and b == .fixed_size_binary);
}

/// Two integer or decimal operands of different types, at least one a decimal.
fn mixedDecimals(args: []const Datum) bool {
    if (args.len != 2) return false;
    const a = args[0].dataType();
    const b = args[1].dataType();
    if (a.eql(b) or !(decimal.isDecimal(a) or decimal.isDecimal(b))) return false;
    return (decimal.isDecimal(a) or decimal.isInteger(a)) and (decimal.isDecimal(b) or decimal.isInteger(b));
}

fn mixedKernel(comptime check: core.TypeCheckFn, comptime chunk: dense.ChunkFn) Kernel {
    return .{
        .signature = .{ .arity = 2, .type_check = check, .options_check = dense.noOptions, .result_type_fn = dense.resultTypeOf(bool_type) },
        .exec = struct {
            fn exec(ctx: *ExecContext, args: []const Datum, _: Options) KernelError!Datum {
                return dense.execElementwise(ctx, args, bool_type, true, chunk);
            }
        }.exec,
    };
}

fn kernel(comptime tag: TypeTag, comptime chunk: dense.ChunkFn) Kernel {
    return .{
        .signature = .{
            .arity = 2,
            .type_check = sameTypes(tag, 2, false),
            .options_check = dense.noOptions,
            .result_type_fn = dense.resultTypeOf(bool_type),
        },
        .exec = struct {
            /// Compare two operands element-wise (array, chunked or scalar).
            fn exec(ctx: *ExecContext, args: []const Datum, _: Options) KernelError!Datum {
                return dense.execElementwise(ctx, args, bool_type, true, chunk);
            }
        }.exec,
    };
}

/// The values of a fixed-width operand as `T`: an array chunk (offset applied) or a non-null
/// scalar whose payload has type `T` (an int32, date32, time32 or decimal32 scalar for `i32`).
fn physOperand(comptime T: type, v: ExecChunkValue) KernelError!dense.Operand(T) {
    return switch (v) {
        .array => |arr| .{ .slice = (PrimitiveArray(T){ .data = arr.data() }).values() catch return error.InvalidInput },
        .scalar => |s| switch (s.value) {
            inline else => |x| if (@TypeOf(x) == T) .{ .scalar = x } else error.InvalidInput,
        },
    };
}

inline fn compare(comptime op: Op, x: anytype, y: @TypeOf(x)) bool {
    return switch (op) {
        .equal => x == y,
        .not_equal => x != y,
        .less => x < y,
        .less_equal => x <= y,
        .greater => x > y,
        .greater_equal => x >= y,
    };
}

/// Return a bool output whose validity is the intersection of the operands'.
fn boolOutput(ctx: *ExecContext, values: []const ExecChunkValue, len: usize) KernelError!dense.DenseBinaryOutput(bool) {
    return dense.DenseBinaryOutput(bool).initValidity(ctx.allocator, len, dense.arrayData(values[0]), dense.arrayData(values[1]), bool_type);
}

/// Pack `pred(i)` for `i < len` into `bits` (8 rows per byte, unused high bits clear).
inline fn packBits(bits: []u8, len: usize, ctx: anytype, comptime pred: anytype) void {
    var i: usize = 0;
    while (i + 8 <= len) : (i += 8) {
        var byte: u8 = 0;
        inline for (0..8) |k| byte |= @as(u8, @intFromBool(pred(ctx, i + k))) << k;
        bits[i / 8] = byte;
    }
    if (i < len) {
        var byte: u8 = 0;
        for (i..len) |j| byte |= @as(u8, @intFromBool(pred(ctx, j))) << @as(u3, @intCast(j - i));
        bits[i / 8] = byte;
    }
}

fn Numeric(comptime T: type, comptime op: Op) type {
    return struct {
        fn chunk(ctx: *ExecContext, values: []const ExecChunkValue, len: usize) KernelError!ArrayRef {
            var out = try boolOutput(ctx, values, len);
            errdefer out.deinit();
            const lhs = try physOperand(T, values[0]);
            const rhs = try physOperand(T, values[1]);
            switch (lhs) {
                inline else => |a| switch (rhs) {
                    inline else => |b| packBits(out.values(), len, .{ a, b }, pred),
                },
            }
            return out.finish() catch error.OutOfMemory;
        }

        inline fn pred(ab: anytype, i: usize) bool {
            return compare(op, dense.at(ab[0], i), dense.at(ab[1], i));
        }
    };
}

fn Boolean(comptime op: Op) type {
    return struct {
        fn bits(v: ExecChunkValue) KernelError!dense.Bits {
            return switch (v) {
                .array => |arr| dense.Bits.values(arr.data()),
                .scalar => |s| dense.Bits.of(try dense.scalarAs(bool, s)),
            };
        }

        fn f(in: [2]u8) u8 {
            const a = in[0];
            const b = in[1];
            return switch (op) {
                .equal => ~(a ^ b),
                .not_equal => a ^ b,
                .less => ~a & b,
                .less_equal => ~a | b,
                .greater => a & ~b,
                .greater_equal => a | ~b,
            };
        }

        fn chunk(ctx: *ExecContext, values: []const ExecChunkValue, len: usize) KernelError!ArrayRef {
            var out = try boolOutput(ctx, values, len);
            errdefer out.deinit();
            dense.mapBits(2, out.values(), .{ try bits(values[0]), try bits(values[1]) }, len, f);
            return out.finish() catch error.OutOfMemory;
        }
    };
}

/// Return the byte-wise order of `a` and `b`, 8 bytes at a time (big-endian words order like bytes).
fn orderBytes(a: []const u8, b: []const u8) std.math.Order {
    const n = @min(a.len, b.len);
    var i: usize = 0;
    while (i + 8 <= n) : (i += 8) {
        const x = std.mem.readInt(u64, a[i..][0..8], .big);
        const y = std.mem.readInt(u64, b[i..][0..8], .big);
        if (x != y) return std.math.order(x, y);
    }
    if (i < n) {
        if (n >= 8) {
            // The last 8 bytes overlap the equal prefix, so their words order like the tail.
            const x = std.mem.readInt(u64, a[n - 8 ..][0..8], .big);
            const y = std.mem.readInt(u64, b[n - 8 ..][0..8], .big);
            if (x != y) return std.math.order(x, y);
        } else while (i < n) : (i += 1) if (a[i] != b[i]) return std.math.order(a[i], b[i]);
    }
    return std.math.order(a.len, b.len);
}

/// Rows of a fixed_size_binary operand: an array chunk or a scalar of `width` bytes.
const FixedBytes = struct {
    bytes: []const u8,
    width: usize,
    scalar: bool,

    fn init(v: ExecChunkValue) KernelError!FixedBytes {
        const width: usize = std.math.cast(usize, v.dataType().fixed_size_binary.byte_width) orelse return error.InvalidInput;
        return switch (v) {
            .scalar => |s| switch (s.value) {
                .binary, .string => |b| if (b.len == width) .{ .bytes = b, .width = width, .scalar = true } else error.InvalidInput,
                else => error.InvalidInput,
            },
            // validateLayout checked that the values buffer holds offset + length rows.
            .array => |arr| .{ .bytes = arr.data().buffers[1].data[arr.data().offset * width ..], .width = width, .scalar = false },
        };
    }

    inline fn get(self: FixedBytes, i: usize) []const u8 {
        return if (self.scalar) self.bytes else self.bytes[i * self.width ..][0..self.width];
    }
};

fn FixedBinary(comptime op: Op) type {
    return struct {
        fn chunk(ctx: *ExecContext, values: []const ExecChunkValue, len: usize) KernelError!ArrayRef {
            var out = try boolOutput(ctx, values, len);
            errdefer out.deinit();
            packBits(out.values(), len, .{ try FixedBytes.init(values[0]), try FixedBytes.init(values[1]) }, Strings(op).pred);
            return out.finish() catch error.OutOfMemory;
        }
    };
}

/// Rows of any string / binary operand, fixed_size_binary included.
const AnyBytes = union(enum) {
    scalar: dense.ScalarBytes,
    offsets32: dense.OffsetBytes(i32),
    offsets64: dense.OffsetBytes(i64),
    views: dense.ViewBytes,
    fixed: FixedBytes,

    fn init(v: ExecChunkValue) KernelError!AnyBytes {
        if (datatype.storageDataType(v.dataType()) == .fixed_size_binary) return .{ .fixed = try FixedBytes.init(v) };
        return switch (try dense.StringOperand.init(v)) {
            inline else => |r, tag| @unionInit(AnyBytes, @tagName(tag), r),
        };
    }
};

/// Comparisons of two different string / binary layouts: the bytes, with no cast.
fn MixedBytes(comptime op: Op) type {
    return struct {
        fn chunk(ctx: *ExecContext, values: []const ExecChunkValue, len: usize) KernelError!ArrayRef {
            var out = try boolOutput(ctx, values, len);
            errdefer out.deinit();
            switch (try AnyBytes.init(values[0])) {
                inline else => |x| switch (try AnyBytes.init(values[1])) {
                    inline else => |y| run(x, y, out.values(), len),
                },
            }
            return out.finish() catch error.OutOfMemory;
        }

        /// The operands as plain parameters (their fields stay in registers across rows).
        fn run(x: anytype, y: anytype, bits: []u8, len: usize) void {
            var i: usize = 0;
            while (i + 8 <= len) : (i += 8) {
                var byte: u8 = 0;
                inline for (0..8) |k| byte |= @as(u8, @intFromBool(holds(orderBytes(x.get(i + k), y.get(i + k))))) << k;
                bits[i / 8] = byte;
            }
            if (i < len) {
                var byte: u8 = 0;
                for (i..len) |j| byte |= @as(u8, @intFromBool(holds(orderBytes(x.get(j), y.get(j))))) << @as(u3, @intCast(j - i));
                bits[i / 8] = byte;
            }
        }

        inline fn holds(o: std.math.Order) bool {
            return switch (op) {
                .equal => o == .eq,
                .not_equal => o != .eq,
                .less => o == .lt,
                .less_equal => o != .gt,
                .greater => o == .gt,
                .greater_equal => o != .lt,
            };
        }
    };
}

/// Comparisons of two integer / decimal operands of different types by exact value.
fn MixedDecimals(comptime op: Op) type {
    return struct {
        fn chunk(ctx: *ExecContext, values: []const ExecChunkValue, len: usize) KernelError!ArrayRef {
            for (values) |v| if (decimal.isDecimal(v.dataType())) {
                _ = try decimal.checkedParams(v.dataType());
            };
            var out = try boolOutput(ctx, values, len);
            errdefer out.deinit();
            const a = try decimal.widen(i256, ctx.allocator, values[0]);
            defer a.deinit(ctx.allocator);
            const b = try decimal.widen(i256, ctx.allocator, values[1]);
            defer b.deinit(ctx.allocator);
            const sa = decimal.scaleOf(values[0].dataType());
            const sb = decimal.scaleOf(values[1].dataType());
            // The operand of the smaller scale is scaled up by k digits.
            const k: u32 = @intCast(@min(@abs(sa - sb), 1000));
            switch (a.op) {
                inline else => |x| switch (b.op) {
                    inline else => |y| if (sa >= sb)
                        packBits(out.values(), len, .{ y, x, k }, predSwapped)
                    else
                        packBits(out.values(), len, .{ x, y, k }, pred),
                },
            }
            return out.finish() catch error.OutOfMemory;
        }

        /// Left operand scaled: `c[0][i] * 10^k` against `c[1][i]`.
        inline fn pred(c: anytype, i: usize) bool {
            return holds(decimal.orderScaled(dense.at(c[0], i), c[2], dense.at(c[1], i)));
        }

        /// Right operand scaled (passed first): the left `c[1][i]` against `c[0][i] * 10^k`.
        inline fn predSwapped(c: anytype, i: usize) bool {
            return holds(decimal.orderScaled(dense.at(c[0], i), c[2], dense.at(c[1], i)).invert());
        }

        inline fn holds(o: std.math.Order) bool {
            return switch (op) {
                .equal => o == .eq,
                .not_equal => o != .eq,
                .less => o == .lt,
                .less_equal => o != .gt,
                .greater => o == .gt,
                .greater_equal => o != .lt,
            };
        }
    };
}

fn Strings(comptime op: Op) type {
    return struct {
        fn chunk(ctx: *ExecContext, values: []const ExecChunkValue, len: usize) KernelError!ArrayRef {
            var out = try boolOutput(ctx, values, len);
            errdefer out.deinit();
            const lhs = try dense.StringOperand.init(values[0]);
            const rhs = try dense.StringOperand.init(values[1]);
            switch (lhs) {
                inline else => |x| switch (rhs) {
                    inline else => |y| packBits(out.values(), len, .{ x, y }, pred),
                },
            }
            return out.finish() catch error.OutOfMemory;
        }

        fn pred(ops: anytype, i: usize) bool {
            const x = ops[0].get(i);
            const y = ops[1].get(i);
            return switch (op) {
                .equal => std.mem.eql(u8, x, y),
                .not_equal => !std.mem.eql(u8, x, y),
                .less => orderBytes(x, y) == .lt,
                .less_equal => orderBytes(x, y) != .gt,
                .greater => orderBytes(x, y) == .gt,
                .greater_equal => orderBytes(x, y) != .lt,
            };
        }
    };
}

// ------------------------------------------------------------------------------ element-wise min / max

fn extremeKernel(comptime tag: TypeTag, comptime which: Extreme) Kernel {
    return .{
        .signature = .{
            .arity = 1,
            .variadic = true,
            .type_check = sameTypes(tag, 1, true),
            .options_check = elementWiseOptions,
            .result_type_fn = firstType,
        },
        .exec = ElementWise(which).exec,
    };
}

fn firstType(args: []const Datum, _: Options) KernelError!DataType {
    if (args.len == 0) return error.InvalidArity;
    return args[0].dataType();
}

/// `none` (skip_nulls) or `ElementWiseAggregateOptions`.
fn elementWiseOptions(options: Options) bool {
    return options == .none or options == .element_wise_aggregate;
}

/// Whether row `i` of an operand is valid.
inline fn validAt(v: ExecChunkValue, i: usize) bool {
    return switch (v) {
        .scalar => |s| !s.isNull(),
        .array => |arr| !arr.data().isNull(i),
    };
}

/// Validity of an element-wise min / max over `values`: the AND of the operands' (`skip_nulls`
/// false) or their OR (null when some operand has no null); bit 0 = row 0, padding bits clear.
fn extremeValidity(allocator: std.mem.Allocator, values: []const ExecChunkValue, len: usize, skip_nulls: bool) KernelError!?OwnedBuffer {
    const n = bitmap.byteLength(len);
    if (!skip_nulls) {
        const inputs = allocator.alloc(?*const @import("../array/array_data.zig").ArrayData, values.len) catch return error.OutOfMemory;
        defer allocator.free(inputs);
        for (values, inputs) |v, *in| {
            // A null scalar: no row is valid (`init` zeroes the bits).
            if (v == .scalar and v.scalar.isNull()) return OwnedBuffer.init(allocator, n) catch error.OutOfMemory;
            in.* = dense.arrayData(v);
        }
        return dense.combinedValidity(allocator, len, inputs);
    }
    for (values) |v| switch (v) {
        .scalar => |s| if (!s.isNull()) return null,
        .array => |arr| if (!arr.data().hasNulls()) return null,
    };
    var out = OwnedBuffer.init(allocator, n) catch return error.OutOfMemory;
    for (values) |v| if (v == .array) {
        const data = v.array.data();
        const bits = dense.validityBits(data) orelse continue;
        if (data.length != len) {
            out.deinit();
            return error.InvalidInput;
        }
        for (out.data[0..n], 0..) |*d, k| d.* |= bitmap.byteAtBit(bits, data.offset + k * 8);
    };
    if (len % 8 != 0) out.data[n - 1] &= (@as(u8, 1) << @as(u3, @intCast(len % 8))) - 1;
    return out;
}

fn ElementWise(comptime which: Extreme) type {
    return struct {
        /// The smallest / largest operand of each row (array, chunked or scalar operands).
        fn exec(ctx: *ExecContext, args: []const Datum, options: Options) KernelError!Datum {
            return dense.execElementwiseWith(ctx, args, options, args[0].dataType(), false, chunk);
        }

        fn chunk(ctx: *ExecContext, values: []const ExecChunkValue, len: usize, options: Options) KernelError!ArrayRef {
            const skip = if (options == .element_wise_aggregate) options.element_wise_aggregate.skip_nulls else true;
            const out_type = values[0].dataType();
            return switch (datatype.storageDataType(out_type)) {
                .string, .large_string, .string_view, .binary, .large_binary, .binary_view => bytesChunk(ctx, values, len, skip, out_type),
                .int8 => fixedChunk(i8, ctx, values, len, skip, out_type),
                .int16 => fixedChunk(i16, ctx, values, len, skip, out_type),
                .int32, .date32, .time32 => fixedChunk(i32, ctx, values, len, skip, out_type),
                .int64, .date64, .timestamp, .time64, .duration => fixedChunk(i64, ctx, values, len, skip, out_type),
                .uint8 => fixedChunk(u8, ctx, values, len, skip, out_type),
                .uint16 => fixedChunk(u16, ctx, values, len, skip, out_type),
                .uint32 => fixedChunk(u32, ctx, values, len, skip, out_type),
                .uint64 => fixedChunk(u64, ctx, values, len, skip, out_type),
                .float => fixedChunk(f32, ctx, values, len, skip, out_type),
                .double => fixedChunk(f64, ctx, values, len, skip, out_type),
                .decimal32 => fixedChunk(i32, ctx, values, len, skip, out_type),
                .decimal64 => fixedChunk(i64, ctx, values, len, skip, out_type),
                .decimal128 => fixedChunk(i128, ctx, values, len, skip, out_type),
                .decimal256 => fixedChunk(i256, ctx, values, len, skip, out_type),
                .fixed_size_binary => fixedBytesChunk(ctx, values, len, skip, out_type),
                else => error.UnsupportedType,
            };
        }

        /// The extreme of `a` and `b`: NaN only when both are, -0.0 below 0.0.
        inline fn pick(comptime T: type, a: T, b: T) T {
            if (@typeInfo(T) != .float) return if (which == .min_element_wise) @min(a, b) else @max(a, b);
            if (std.math.isNan(a)) return b;
            if (std.math.isNan(b)) return a;
            if (a == b) {
                // Equal values: only the signs of two zeros can differ.
                const a_neg = std.math.signbit(a);
                return if ((which == .min_element_wise) == a_neg) a else b;
            }
            return if ((a < b) == (which == .min_element_wise)) a else b;
        }

        fn fixedChunk(comptime T: type, ctx: *ExecContext, values: []const ExecChunkValue, len: usize, skip: bool, out_type: DataType) KernelError!ArrayRef {
            var out: dense.DenseBinaryOutput(T) = .{
                .allocator = ctx.allocator,
                .len = len,
                .values_buf = OwnedBuffer.initUninit(ctx.allocator, len * @sizeOf(T)) catch return error.OutOfMemory,
                .validity = null,
                .out_type = out_type,
            };
            errdefer out.deinit();
            out.validity = try extremeValidity(ctx.allocator, values, len, skip);
            const dst = out.values();
            const start: T = if (@typeInfo(T) == .float) std.math.nan(T) else if (which == .min_element_wise) std.math.maxInt(T) else std.math.minInt(T);
            @memset(dst, start);
            for (values) |v| {
                if (v == .scalar and v.scalar.isNull()) continue;
                const bits = if (v == .array) dense.validityBits(v.array.data()) else null;
                switch (try physOperand(T, v)) {
                    inline else => |x| for (dst, 0..) |*d, i| {
                        if (bits) |b| if (!bitmap.bitIsSet(b, v.array.data().offset + i)) continue;
                        d.* = pick(T, d.*, dense.at(x, i));
                    },
                }
            }
            return out.finish() catch error.OutOfMemory;
        }

        fn fixedBytesChunk(ctx: *ExecContext, values: []const ExecChunkValue, len: usize, skip: bool, out_type: DataType) KernelError!ArrayRef {
            const width: usize = std.math.cast(usize, out_type.fixed_size_binary.byte_width) orelse return error.InvalidInput;
            const ops = ctx.allocator.alloc(?FixedBytes, values.len) catch return error.OutOfMemory;
            defer ctx.allocator.free(ops);
            for (values, ops) |v, *op| op.* = if (v == .scalar and v.scalar.isNull()) null else try FixedBytes.init(v);
            const bytes = std.math.mul(usize, len, width) catch return error.Overflow;
            var validity = try extremeValidity(ctx.allocator, values, len, skip);
            var vals = OwnedBuffer.init(ctx.allocator, bytes) catch {
                if (validity) |*b| b.deinit();
                return error.OutOfMemory;
            };
            for (0..len) |i| {
                if (validity) |b| if (!bitmap.bitIsSet(b.data, i)) continue;
                var best: ?[]const u8 = null;
                for (values, ops) |v, maybe| {
                    const op = maybe orelse continue;
                    if (!validAt(v, i)) continue;
                    const row = op.get(i);
                    const better: std.math.Order = if (which == .min_element_wise) .lt else .gt;
                    if (best == null or orderBytes(row, best.?) == better) best = row;
                }
                if (best) |row| @memcpy(vals.data[i * width ..][0..width], row);
            }
            return dense.finishFixed(ctx.allocator, out_type, len, &vals, validity);
        }

        fn bytesChunk(ctx: *ExecContext, values: []const ExecChunkValue, len: usize, skip: bool, out_type: DataType) KernelError!ArrayRef {
            const ops = ctx.allocator.alloc(?dense.StringOperand, values.len) catch return error.OutOfMemory;
            defer ctx.allocator.free(ops);
            for (values, ops) |v, *op| op.* = if (v == .scalar and v.scalar.isNull()) null else try dense.StringOperand.init(v);
            var out = try dense.VarBinaryOutput.init(ctx.allocator, out_type, len, try extremeValidity(ctx.allocator, values, len, skip), if (ops[0]) |op| dense.totalBytes(op, len) else 0);
            errdefer out.deinit();
            for (0..len) |i| {
                if (out.validity) |v| if (!bitmap.bitIsSet(v.data, i)) {
                    try out.commit(0);
                    continue;
                };
                var best: ?[]const u8 = null;
                for (values, ops) |v, maybe| {
                    const op = maybe orelse continue;
                    if (!validAt(v, i)) continue;
                    const row = switch (op) {
                        inline else => |r| r.get(i),
                    };
                    const better: std.math.Order = if (which == .min_element_wise) .lt else .gt;
                    if (best == null or orderBytes(row, best.?) == better) best = row;
                }
                try out.append(best orelse "");
            }
            return out.finish();
        }
    };
}
