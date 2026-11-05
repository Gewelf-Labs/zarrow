//! Arithmetic kernels: add, subtract, multiply, divide, negate, abs and their `_checked` variants
//! for int8..int64, uint8..uint64, float32 and float64 (Arrow C++ "Arithmetic functions").
//!
//! Null handling is intersection. The unchecked functions wrap on integer overflow, or saturate when
//! `ExecConfig.overflow_mode` is `saturating`; the `_checked` functions return `Overflow`. Integer
//! division truncates toward zero and a zero divisor in a valid row is `DivideByZero` in both
//! variants (as in Arrow C++); `divide(minInt, -1)` is 0 (Arrow C++) or `maxInt` when saturating.
//! Float division by zero gives inf / nan unless checked. Float add / subtract / multiply never
//! report overflow. Operands of different numeric types are first cast to their common type
//! (Arrow's implicit casts, `registry.commonNumericType`: int32 + int64 is int64, int + float32 is
//! float32, uint32 + int32 is int64; a value the cast cannot represent is `InvalidCast`).
const std = @import("std");
const core = @import("core.zig");
const dense = @import("dense.zig");

const ArrayRef = core.ArrayRef;
const Datum = core.Datum;
const ExecChunkValue = core.ExecChunkValue;
const ExecContext = core.ExecContext;
const FunctionRegistry = core.FunctionRegistry;
const Kernel = core.Kernel;
const KernelError = core.KernelError;
const Options = core.Options;

const Op = enum { add, subtract, multiply, divide, negate, abs };
const Mode = enum { wrapping, saturating, checked };

/// Register add, subtract, multiply, divide, negate, abs and their `_checked` variants.
pub fn register(registry: *FunctionRegistry) KernelError!void {
    inline for (.{ Op.add, Op.subtract, Op.multiply, Op.divide, Op.negate, Op.abs }) |op| {
        inline for (dense.numeric_types) |T| {
            try registry.registerScalarKernel(@tagName(op), kernel(T, op, false));
            // Arrow C++ has no negate_checked kernel for unsigned integers.
            if (!(op == .negate and @typeInfo(T) == .int and @typeInfo(T).int.signedness == .unsigned)) {
                try registry.registerScalarKernel(@tagName(op) ++ "_checked", kernel(T, op, true));
            }
        }
        if (op != .negate and op != .abs) {
            try registry.setImplicitCast(@tagName(op), .scalar, .numeric);
            try registry.setImplicitCast(@tagName(op) ++ "_checked", .scalar, .numeric);
        }
    }
}

fn kernel(comptime T: type, comptime op: Op, comptime checked: bool) Kernel {
    const arity: usize = if (op == .negate or op == .abs) 1 else 2;
    return .{
        .signature = .{
            .arity = arity,
            .type_check = dense.argsOfType(std.meta.activeTag(dense.dataTypeOf(T)), arity),
            .options_check = dense.noOptions,
            .result_type_fn = dense.resultTypeOf(dense.dataTypeOf(T)),
        },
        .exec = Arith(T, op, checked).exec,
    };
}

fn Arith(comptime T: type, comptime op: Op, comptime checked: bool) type {
    const is_float = @typeInfo(T) == .float;
    const signed = !is_float and @typeInfo(T).int.signedness == .signed;
    const unary = op == .negate or op == .abs;
    return struct {
        /// Apply the operation element-wise (array, chunked or scalar operands).
        fn exec(ctx: *ExecContext, args: []const Datum, _: Options) KernelError!Datum {
            return dense.execElementwise(ctx, args, dense.dataTypeOf(T), true, chunk);
        }

        fn chunk(ctx: *ExecContext, values: []const ExecChunkValue, len: usize) KernelError!ArrayRef {
            const mode: Mode = if (checked) .checked else switch (ctx.overflowMode()) {
                .saturating => .saturating,
                .wrapping, .checked => .wrapping,
            };
            const rhs_data = if (unary) null else dense.arrayData(values[1]);
            var out = try dense.DenseBinaryOutput(T).initValidity(ctx.allocator, len, dense.arrayData(values[0]), rhs_data, dense.dataTypeOf(T));
            errdefer out.deinit();
            const lhs = try dense.operand(T, values[0]);
            if (unary) {
                switch (lhs) {
                    inline else => |a| switch (mode) {
                        inline else => |m| try runUnary(m, a, &out),
                    },
                }
            } else {
                const rhs = try dense.operand(T, values[1]);
                switch (lhs) {
                    inline else => |a| switch (rhs) {
                        inline else => |b| switch (mode) {
                            inline else => |m| try runBinary(m, a, b, &out),
                        },
                    },
                }
            }
            return out.finish() catch error.OutOfMemory;
        }

        const Out = dense.DenseBinaryOutput(T);
        const at = dense.at;

        inline fn wrapOp(x: T, y: T) T {
            if (is_float) return switch (op) {
                .add => x + y,
                .subtract => x - y,
                .multiply => x * y,
                else => unreachable,
            };
            return switch (op) {
                .add => x +% y,
                .subtract => x -% y,
                .multiply => x *% y,
                else => unreachable,
            };
        }

        inline fn satOp(x: T, y: T) T {
            return switch (op) {
                .add => x +| y,
                .subtract => x -| y,
                .multiply => x *| y,
                else => unreachable,
            };
        }

        /// Overflow flag of `r = wrapOp(x, y)`, branch free for add / subtract.
        inline fn overflowed(x: T, y: T, r: T) bool {
            return switch (op) {
                .add => if (signed) ((x ^ r) & (y ^ r)) < 0 else r < x,
                .subtract => if (signed) ((x ^ y) & (x ^ r)) < 0 else x < y,
                .multiply => @mulWithOverflow(x, y)[1] != 0,
                else => unreachable,
            };
        }

        fn runBinary(comptime mode: Mode, a: anytype, b: anytype, out: *Out) KernelError!void {
            const dst = out.values();
            if (op == .divide) return divide(mode, a, b, out);
            if (is_float or mode == .wrapping) {
                for (dst, 0..) |*d, i| d.* = wrapOp(at(a, i), at(b, i));
            } else if (mode == .saturating) {
                for (dst, 0..) |*d, i| d.* = satOp(at(a, i), at(b, i));
            } else {
                // Dense pass with a branch-free overflow flag; only when some row overflowed,
                // find out whether a valid one did (null rows may hold anything).
                var any: u8 = 0;
                for (dst, 0..) |*d, i| {
                    const x = at(a, i);
                    const y = at(b, i);
                    const r = wrapOp(x, y);
                    d.* = r;
                    any |= @intFromBool(overflowed(x, y, r));
                }
                if (any != 0) for (dst, 0..) |r, i| {
                    if (overflowed(at(a, i), at(b, i), r) and out.isValid(i)) return error.Overflow;
                };
            }
        }

        fn divide(comptime mode: Mode, a: anytype, b: anytype, out: *Out) KernelError!void {
            const dst = out.values();
            if (is_float) {
                for (dst, 0..) |*d, i| d.* = at(a, i) / at(b, i);
                if (mode == .checked) for (0..dst.len) |i| {
                    if (at(b, i) == 0 and out.isValid(i)) return error.DivideByZero;
                };
                return;
            }
            // Null rows may hold a zero divisor or minInt / -1: divide by 1 there instead of trapping.
            var zero: u8 = 0;
            var over: u8 = 0;
            for (dst, 0..) |*d, i| {
                const x = at(a, i);
                const y = at(b, i);
                const z = y == 0;
                const o = if (signed) x == std.math.minInt(T) and y == -1 else false;
                zero |= @intFromBool(z);
                over |= @intFromBool(o);
                d.* = if (o) (if (mode == .saturating) std.math.maxInt(T) else 0) else @divTrunc(x, if (z) 1 else y);
            }
            if (zero != 0) for (0..dst.len) |i| {
                if (at(b, i) == 0 and out.isValid(i)) return error.DivideByZero;
            };
            if (signed and mode == .checked and over != 0) for (0..dst.len) |i| {
                if (at(a, i) == std.math.minInt(T) and at(b, i) == -1 and out.isValid(i)) return error.Overflow;
            };
        }

        fn runUnary(comptime mode: Mode, a: anytype, out: *Out) KernelError!void {
            const dst = out.values();
            if (is_float) {
                for (dst, 0..) |*d, i| d.* = if (op == .negate) -at(a, i) else @abs(at(a, i));
                return;
            }
            if (!signed) {
                // abs is the identity; negate wraps (or saturates to 0).
                for (dst, 0..) |*d, i| d.* = if (op == .abs) at(a, i) else if (mode == .saturating) 0 -| at(a, i) else 0 -% at(a, i);
                return;
            }
            const min = std.math.minInt(T);
            var any: u8 = 0;
            for (dst, 0..) |*d, i| {
                const x = at(a, i);
                any |= @intFromBool(x == min);
                const wrapped: T = if (op == .negate) 0 -% x else @bitCast(@abs(x));
                d.* = if (mode == .saturating and x == min) std.math.maxInt(T) else wrapped;
            }
            if (mode == .checked and any != 0) for (0..dst.len) |i| {
                if (at(a, i) == min and out.isValid(i)) return error.Overflow;
            };
        }
    };
}
