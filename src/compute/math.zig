//! Math kernels (Arrow C++ "Arithmetic functions", "Bit-wise functions", "Logarithmic
//! functions", "Trigonometric functions", "Hyperbolic trigonometric functions"):
//!
//! - float32 / float64 (integers and decimals are cast to float64 first, Arrow's
//!   `ArithmeticFloatingPointFunction`): sqrt, exp, expm1, ln, log10, log2, log1p, sin, cos, tan,
//!   asin, acos, atan, sinh, cosh, tanh, asinh, acosh, atanh; binary atan2 and logb. The unchecked
//!   functions follow IEEE 754 (NaN for a domain error, -inf for a logarithm of zero); the
//!   `_checked` variants Arrow has (sqrt, ln, log10, log2, log1p, logb, sin, cos, tan, asin, acos,
//!   acosh, atanh) return `InvalidInput` on a valid row outside the domain (a negative square
//!   root, a logarithm of zero or of a negative number, the sine of an infinity, an arcsine past
//!   1). NaN passes through every function.
//! - power (+ `_checked`) on int8..int64, uint8..uint64 (a negative exponent in a valid row is
//!   `InvalidInput` in both variants; unchecked wraps, checked returns `Overflow` like Arrow's
//!   left-to-right squaring) and on floats (`pow`, never an error); a decimal operand makes both
//!   float64 (Arrow's `ArithmeticDecimalToFloatingPointFunction`), other mixed operands go to
//!   their common numeric type.
//! - bit_wise_and, bit_wise_or, bit_wise_xor, bit_wise_not, shift_left, shift_right (+ `_checked`)
//!   on the integer types (mixed operands: common numeric type). A shift amount outside
//!   `[0, digits)` (digits: 7 for int8, 8 for uint8, ...) leaves the value unchanged; the
//!   `_checked` shifts return `InvalidInput` for it. shift_left wraps; shift_right is arithmetic
//!   for signed types.
//!
//! Null handling is intersection; array, chunked or scalar operands.
const std = @import("std");
const core = @import("core.zig");
const dense = @import("dense.zig");

const ArrayRef = core.ArrayRef;
const ArrayData = @import("../array/array_data.zig").ArrayData;
const Datum = core.Datum;
const ExecChunkValue = core.ExecChunkValue;
const ExecContext = core.ExecContext;
const FunctionRegistry = core.FunctionRegistry;
const Kernel = core.Kernel;
const KernelError = core.KernelError;
const Options = core.Options;

const float_types = [_]type{ f32, f64 };
const int_types = [_]type{ i8, i16, i32, i64, u8, u16, u32, u64 };

/// The unary floating-point functions, by Arrow name.
pub const Unary = enum { sqrt, exp, expm1, ln, log10, log2, log1p, sin, cos, tan, asin, acos, atan, sinh, cosh, tanh, asinh, acosh, atanh };

/// Whether Arrow has a `_checked` variant of `op`.
fn hasChecked(comptime op: Unary) bool {
    return switch (op) {
        .sqrt, .ln, .log10, .log2, .log1p, .sin, .cos, .tan, .asin, .acos, .acosh, .atanh => true,
        else => false,
    };
}

/// Binary operations of this file.
const Binary = enum { power, atan2, logb, bit_wise_and, bit_wise_or, bit_wise_xor, shift_left, shift_right };

/// Register the math, bit-wise and shift functions.
pub fn register(registry: *FunctionRegistry) KernelError!void {
    inline for (comptime std.meta.tags(Unary)) |op| {
        inline for (float_types) |T| {
            try registry.registerScalarKernel(@tagName(op), unaryKernel(T, op, false));
            if (comptime hasChecked(op)) try registry.registerScalarKernel(@tagName(op) ++ "_checked", unaryKernel(T, op, true));
        }
        try registry.setImplicitCast(@tagName(op), .scalar, .floating);
        if (comptime hasChecked(op)) try registry.setImplicitCast(@tagName(op) ++ "_checked", .scalar, .floating);
    }
    inline for (float_types) |T| {
        try registry.registerScalarKernel("atan2", binaryKernel(T, .atan2, false));
        try registry.registerScalarKernel("logb", binaryKernel(T, .logb, false));
        try registry.registerScalarKernel("logb_checked", binaryKernel(T, .logb, true));
    }
    inline for (.{ "atan2", "logb", "logb_checked" }) |name| try registry.setImplicitCast(name, .scalar, .floating);
    inline for (dense.numeric_types) |T| {
        try registry.registerScalarKernel("power", binaryKernel(T, .power, false));
        try registry.registerScalarKernel("power_checked", binaryKernel(T, .power, true));
    }
    try registry.setImplicitCast("power", .scalar, .decimal_to_float);
    try registry.setImplicitCast("power_checked", .scalar, .decimal_to_float);
    inline for (int_types) |T| {
        inline for (.{ Binary.bit_wise_and, Binary.bit_wise_or, Binary.bit_wise_xor }) |op| {
            try registry.registerScalarKernel(@tagName(op), binaryKernel(T, op, false));
        }
        inline for (.{ Binary.shift_left, Binary.shift_right }) |op| {
            try registry.registerScalarKernel(@tagName(op), binaryKernel(T, op, false));
            try registry.registerScalarKernel(@tagName(op) ++ "_checked", binaryKernel(T, op, true));
        }
        try registry.registerScalarKernel("bit_wise_not", bitNotKernel(T));
    }
    inline for (.{ "bit_wise_and", "bit_wise_or", "bit_wise_xor", "shift_left", "shift_left_checked", "shift_right", "shift_right_checked" }) |name| {
        try registry.setImplicitCast(name, .scalar, .numeric);
    }
}

fn signature(comptime T: type, comptime arity: usize) core.KernelSignature {
    return .{
        .arity = arity,
        .type_check = dense.argsOfType(std.meta.activeTag(dense.dataTypeOf(T)), arity),
        .options_check = dense.noOptions,
        .result_type_fn = dense.resultTypeOf(dense.dataTypeOf(T)),
    };
}

/// `op(x)` in `T` (IEEE 754; the C library functions Arrow calls).
pub inline fn unaryValue(comptime T: type, comptime op: Unary, x: T) T {
    return switch (op) {
        .sqrt => @sqrt(x),
        .exp => @exp(x),
        .expm1 => std.math.expm1(x),
        .ln => @log(x),
        .log10 => @log10(x),
        .log2 => @log2(x),
        .log1p => std.math.log1p(x),
        .sin => @sin(x),
        .cos => @cos(x),
        .tan => @tan(x),
        .asin => std.math.asin(x),
        .acos => std.math.acos(x),
        .atan => std.math.atan(x),
        .sinh => std.math.sinh(x),
        .cosh => std.math.cosh(x),
        .tanh => std.math.tanh(x),
        .asinh => std.math.asinh(x),
        .acosh => std.math.acosh(x),
        .atanh => std.math.atanh(x),
    };
}

/// Whether the checked variant of `op` rejects `x` (Arrow's domain checks).
inline fn outOfDomain(comptime T: type, comptime op: Unary, x: T) bool {
    return switch (op) {
        .sqrt => x < 0,
        .ln, .log10, .log2 => x <= 0,
        .log1p => x <= -1,
        .sin, .cos, .tan => std.math.isInf(x),
        .asin, .acos => x < -1 or x > 1,
        .acosh => x < 1,
        .atanh => x <= -1 or x >= 1,
        else => false,
    };
}

fn unaryKernel(comptime T: type, comptime op: Unary, comptime checked: bool) Kernel {
    return .{
        .signature = signature(T, 1),
        .exec = struct {
            fn exec(ctx: *ExecContext, args: []const Datum, _: Options) KernelError!Datum {
                return dense.execElementwise(ctx, args, dense.dataTypeOf(T), true, chunk);
            }

            fn chunk(ctx: *ExecContext, values: []const ExecChunkValue, len: usize) KernelError!ArrayRef {
                var out = try dense.DenseBinaryOutput(T).initValidity(ctx.allocator, len, dense.arrayData(values[0]), null, dense.dataTypeOf(T));
                errdefer out.deinit();
                const dst = out.values();
                switch (try dense.operand(T, values[0])) {
                    .slice => |a| if (op == .sqrt) {
                        // Whole vectors: LLVM leaves the scalar loop unvectorized.
                        const V = @Vector(8, T);
                        var i: usize = 0;
                        while (i + 8 <= dst.len) : (i += 8) {
                            const x: V = a[i..][0..8].*;
                            const r: V = @sqrt(x);
                            dst[i..][0..8].* = r;
                        }
                        while (i < dst.len) : (i += 1) dst[i] = @sqrt(a[i]);
                    } else for (dst, a) |*d, x| {
                        d.* = unaryValue(T, op, x);
                    },
                    .scalar => |x| @memset(dst, unaryValue(T, op, x)),
                }
                if (checked) switch (try dense.operand(T, values[0])) {
                    inline else => |a| for (0..len) |i| {
                        if (outOfDomain(T, op, dense.at(a, i)) and out.isValid(i)) return error.InvalidInput;
                    },
                };
                return out.finish() catch error.OutOfMemory;
            }
        }.exec,
    };
}

fn bitNotKernel(comptime T: type) Kernel {
    return .{
        .signature = signature(T, 1),
        .exec = struct {
            fn exec(ctx: *ExecContext, args: []const Datum, _: Options) KernelError!Datum {
                return dense.execElementwise(ctx, args, dense.dataTypeOf(T), true, chunk);
            }

            fn chunk(ctx: *ExecContext, values: []const ExecChunkValue, len: usize) KernelError!ArrayRef {
                var out = try dense.DenseBinaryOutput(T).initValidity(ctx.allocator, len, dense.arrayData(values[0]), null, dense.dataTypeOf(T));
                errdefer out.deinit();
                switch (try dense.operand(T, values[0])) {
                    inline else => |a| for (out.values(), 0..) |*d, i| {
                        d.* = ~dense.at(a, i);
                    },
                }
                return out.finish() catch error.OutOfMemory;
            }
        }.exec,
    };
}

/// Arrow's integer power: `base^exp` by left-to-right squaring, wrapping; `overflow` is set when
/// a step overflowed (power_checked). `exp` must not be negative.
pub fn intPower(comptime T: type, base: T, exp: T, overflow: *bool) T {
    if (exp == 0) return 1;
    const U = @Int(.unsigned, @bitSizeOf(T));
    const e: U = @intCast(exp);
    var mask: U = @as(U, 1) << @intCast(@bitSizeOf(U) - 1 - @clz(e));
    var pow: T = 1;
    while (mask != 0) : (mask >>= 1) {
        const sq = @mulWithOverflow(pow, pow);
        pow = sq[0];
        overflow.* = overflow.* or sq[1] != 0;
        if (e & mask != 0) {
            const m = @mulWithOverflow(pow, base);
            pow = m[0];
            overflow.* = overflow.* or m[1] != 0;
        }
    }
    return pow;
}

/// Arrow's logb: log(x) / log(base), with its special cases for x == 0 and x < 0 (unchecked).
inline fn logbValue(comptime T: type, x: T, base: T) T {
    if (x == 0) return if (base <= 0) std.math.nan(T) else -std.math.inf(T);
    if (x < 0) return std.math.nan(T);
    return @log(x) / @log(base);
}

fn binaryKernel(comptime T: type, comptime op: Binary, comptime checked: bool) Kernel {
    const is_float = @typeInfo(T) == .float;
    const signed = !is_float and @typeInfo(T).int.signedness == .signed;
    // Shift amounts below `digits` change the value (Arrow's std::numeric_limits<T>::digits).
    const digits: comptime_int = if (is_float) 0 else @typeInfo(T).int.bits - @intFromBool(signed);
    return .{
        .signature = signature(T, 2),
        .exec = struct {
            fn exec(ctx: *ExecContext, args: []const Datum, _: Options) KernelError!Datum {
                return dense.execElementwise(ctx, args, dense.dataTypeOf(T), true, chunk);
            }

            fn chunk(ctx: *ExecContext, values: []const ExecChunkValue, len: usize) KernelError!ArrayRef {
                const rhs_data = dense.arrayData(values[1]);
                var out = try dense.DenseBinaryOutput(T).initValidity(ctx.allocator, len, dense.arrayData(values[0]), rhs_data, dense.dataTypeOf(T));
                errdefer out.deinit();
                const lhs = try dense.operand(T, values[0]);
                const rhs = try dense.operand(T, values[1]);
                switch (lhs) {
                    inline else => |a| switch (rhs) {
                        inline else => |b| try run(a, b, rhs_data, &out),
                    },
                }
                return out.finish() catch error.OutOfMemory;
            }

            fn run(a: anytype, b: anytype, rhs_data: ?*const ArrayData, out: *dense.DenseBinaryOutput(T)) KernelError!void {
                const dst = out.values();
                const at = dense.at;
                switch (op) {
                    .atan2 => for (dst, 0..) |*d, i| {
                        d.* = std.math.atan2(at(a, i), at(b, i));
                    },
                    .logb => {
                        for (dst, 0..) |*d, i| d.* = logbValue(T, at(a, i), at(b, i));
                        if (checked) for (0..dst.len) |i| {
                            const x = at(a, i);
                            const y = at(b, i);
                            if ((x <= 0 or y <= 0) and out.isValid(i)) return error.InvalidInput;
                        };
                    },
                    .power => if (is_float) {
                        for (dst, 0..) |*d, i| d.* = std.math.pow(T, at(a, i), at(b, i));
                    } else {
                        // A negative exponent is an error wherever the exponent is valid (Arrow
                        // checks it before the null intersection of the base).
                        if (signed) for (0..dst.len) |i| {
                            if (at(b, i) < 0 and (rhs_data == null or !rhs_data.?.isNull(i))) return error.InvalidInput;
                        };
                        var any = false;
                        for (dst, 0..) |*d, i| {
                            var over = false;
                            const e = at(b, i);
                            d.* = if (e < 0) 0 else intPower(T, at(a, i), e, &over);
                            if (checked and over and out.isValid(i)) any = true;
                        }
                        if (any) return error.Overflow;
                    },
                    .bit_wise_and => for (dst, 0..) |*d, i| {
                        d.* = at(a, i) & at(b, i);
                    },
                    .bit_wise_or => for (dst, 0..) |*d, i| {
                        d.* = at(a, i) | at(b, i);
                    },
                    .bit_wise_xor => for (dst, 0..) |*d, i| {
                        d.* = at(a, i) ^ at(b, i);
                    },
                    .shift_left, .shift_right => {
                        const U = @Int(.unsigned, @bitSizeOf(T));
                        const S = std.math.Log2Int(T);
                        for (dst, 0..) |*d, i| {
                            const x = at(a, i);
                            const s = at(b, i);
                            if (s < 0 or s >= digits) {
                                if (checked and out.isValid(i)) return error.InvalidInput;
                                d.* = x;
                            } else if (op == .shift_left) {
                                d.* = @bitCast(@as(U, @bitCast(x)) << @as(S, @intCast(s)));
                            } else {
                                d.* = x >> @as(S, @intCast(s));
                            }
                        }
                    },
                }
            }
        }.exec,
    };
}
