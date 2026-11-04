//! Comparison kernels: equal, not_equal, less, less_equal, greater, greater_equal for
//! int8..int64, uint8..uint64, float32, float64, bool, string, large_string and string_view
//! (Arrow C++ "Comparisons"). The output is a bool array (scalar for scalar inputs); null
//! handling is intersection. Strings compare by bytes; floats follow IEEE 754 (NaN is unequal to
//! everything, `not_equal` is true for it); `false < true`. Both operands must have the same type.
const std = @import("std");
const core = @import("core.zig");
const dense = @import("dense.zig");

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
const bool_type = DataType{ .bool = {} };
const string_tags = [_]TypeTag{ .string, .large_string, .string_view };

/// Register equal, not_equal, less, less_equal, greater and greater_equal.
pub fn register(registry: *FunctionRegistry) KernelError!void {
    inline for (comptime std.meta.tags(Op)) |op| {
        inline for (dense.numeric_types) |T| {
            try registry.registerScalarKernel(@tagName(op), kernel(std.meta.activeTag(dense.dataTypeOf(T)), Numeric(T, op).chunk));
        }
        try registry.registerScalarKernel(@tagName(op), kernel(.bool, Boolean(op).chunk));
        inline for (string_tags) |tag| {
            try registry.registerScalarKernel(@tagName(op), kernel(tag, Strings(op).chunk));
        }
    }
}

fn kernel(comptime tag: TypeTag, comptime chunk: dense.ChunkFn) Kernel {
    return .{
        .signature = .{
            .arity = 2,
            .type_check = dense.argsOfType(tag, 2),
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
            const lhs = try dense.operand(T, values[0]);
            const rhs = try dense.operand(T, values[1]);
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
    while (i < n) : (i += 1) if (a[i] != b[i]) return std.math.order(a[i], b[i]);
    return std.math.order(a.len, b.len);
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
