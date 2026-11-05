//! Decimal kernels (Arrow C++ "Arithmetic functions" and "Conversions" on decimals) and the
//! decimal helpers that the comparison and aggregate families share.
//!
//! Arithmetic: add, subtract, multiply, divide and their `_checked` variants take two operands
//! that are decimal32..decimal256 or an integer (an integer is `decimal(digits, 0)`, Arrow's
//! `MaxDecimalDigitsForInteger`: 3, 5, 10, 19 digits, 20 for uint64) with at least one decimal. The
//! result type follows Arrow C++ (`CastBinaryDecimalArgs` and the output resolvers, Redshift's
//! rules), decimal256 when an operand is decimal256 and decimal128 otherwise:
//! - add / subtract: scale `max(s1, s2)`, precision `max(p1 - s1, p2 - s2) + max(s1, s2) + 1`;
//!   the operand of the smaller scale is scaled up first.
//! - multiply: scale `s1 + s2`, precision `p1 + p2 + 1`.
//! - divide: the dividend is scaled up by `k = max(4, s1 + p2 - s2 + 1) + s2 - s1` digits, then
//!   divided (truncated toward zero): scale `s1 + k - s2`, precision `p1 + k`.
//! A result precision past 38 (decimal128) or 76 (decimal256) is `Overflow` (Arrow: "Decimal
//! precision out of range"); a negative scale is `NotImplemented` (as Arrow). A zero divisor in a
//! valid row is `DivideByZero` in both variants. The unchecked functions wrap at the storage
//! width (Arrow C++; operands within their precision never wrap under these rules); the
//! `_checked` functions return `Overflow` when a valid row overflows the storage width or its
//! result does not fit the result precision. A decimal and a float are cast to float64 first
//! (`registry.commonDecimalFloatType`, as Arrow).
//!
//! negate, negate_checked, abs, abs_checked (same type) and sign (int64) take decimal128 and
//! decimal256 (Arrow registers no decimal32 / decimal64 kernels); the checked variants report the
//! one storage value that has no negation.
//!
//! Casts (`castArray`, called by `cast`; safe when both `CastOptions.safe` and
//! `ExecConfig.safe_cast`): decimal to decimal (scaling down truncates toward zero; safe: a lost
//! digit is `InvalidCast`, and so is a value past the target precision), decimal to integer
//! (truncates; safe: a fraction or a value out of range is `InvalidCast`; unsafe wraps), integer to
//! decimal (the target precision must hold `digits + scale` and the scale must not be negative, in
//! both modes, as Arrow), decimal to float (correctly rounded), float to decimal (the exact binary
//! value rounded half to even; safe: NaN, inf and a value past the precision are `InvalidCast`,
//! unsafe gives 0, as Arrow), decimal to the string layouts (Arrow's `ToString`: plain digits, or
//! `1.23E+4` form when the scale is negative or the adjusted exponent is below -6) and string /
//! binary layouts to decimal (Arrow's `FromString` grammar: sign, digits, optional fraction and
//! exponent; text that does not parse or does not fit the storage is `InvalidCast` in both modes,
//! dropped digits and a precision overflow only in safe mode).
//!
//! Every decimal type must have a precision in `[1, 9 / 18 / 38 / 76]` (`InvalidInput`
//! otherwise); values are read as whole 4 / 8 / 16 / 32-byte elements through the typed accessors.
const std = @import("std");
const bitmap = @import("../bitmap.zig");
const core = @import("core.zig");
const datatype = @import("../datatype.zig");
const dense = @import("dense.zig");
const PrimitiveArray = @import("../array/primitive_array.zig").PrimitiveArray;
const ArrayData = @import("../array/array_data.zig").ArrayData;

const ArrayRef = core.ArrayRef;
const DataType = core.DataType;
const Datum = core.Datum;
const ExecChunkValue = core.ExecChunkValue;
const ExecContext = core.ExecContext;
const FunctionRegistry = core.FunctionRegistry;
const Kernel = core.Kernel;
const KernelError = core.KernelError;
const Options = core.Options;
const Scalar = core.Scalar;
const ScalarValue = core.ScalarValue;
const TypeTag = std.meta.Tag(DataType);

pub const Params = datatype.DecimalParams;

/// The decimal type tags with their storage integers.
pub const decimal_types = .{
    .{ TypeTag.decimal32, i32 },   .{ TypeTag.decimal64, i64 },
    .{ TypeTag.decimal128, i128 }, .{ TypeTag.decimal256, i256 },
};

/// Integer and decimal type tags with their storage integers (the operands `widen` reads).
const int_like = .{
    .{ TypeTag.int8, i8 },       .{ TypeTag.int16, i16 },     .{ TypeTag.int32, i32 },       .{ TypeTag.int64, i64 },
    .{ TypeTag.uint8, u8 },      .{ TypeTag.uint16, u16 },    .{ TypeTag.uint32, u32 },      .{ TypeTag.uint64, u64 },
    .{ TypeTag.decimal32, i32 }, .{ TypeTag.decimal64, i64 }, .{ TypeTag.decimal128, i128 }, .{ TypeTag.decimal256, i256 },
};

pub fn isDecimal(dt: DataType) bool {
    return switch (dt) {
        .decimal32, .decimal64, .decimal128, .decimal256 => true,
        else => false,
    };
}

/// Precision and scale of a decimal type (null for anything else).
pub fn params(dt: DataType) ?Params {
    return switch (dt) {
        .decimal32, .decimal64, .decimal128, .decimal256 => |p| p,
        else => null,
    };
}

/// Largest precision of a decimal type: 9, 18, 38 or 76.
pub fn maxPrecision(dt: DataType) u8 {
    return switch (dt) {
        .decimal32 => 9,
        .decimal64 => 18,
        .decimal128 => 38,
        else => 76,
    };
}

/// The parameters of a decimal type whose precision is in `[1, maxPrecision]`; `InvalidInput`
/// for another precision or a type that is not a decimal.
pub fn checkedParams(dt: DataType) KernelError!Params {
    const p = params(dt) orelse return error.InvalidInput;
    if (p.precision < 1 or p.precision > maxPrecision(dt)) return error.InvalidInput;
    return p;
}

/// Digits of the widest value of an integer type (Arrow `MaxDecimalDigitsForInteger`).
pub fn integerDigits(dt: DataType) ?u8 {
    return switch (dt) {
        .int8, .uint8 => 3,
        .int16, .uint16 => 5,
        .int32, .uint32 => 10,
        .int64 => 19,
        .uint64 => 20,
        else => null,
    };
}

/// Whether `dt` is an integer type.
pub fn isInteger(dt: DataType) bool {
    return integerDigits(dt) != null;
}

const pow10_table: [77]i256 = blk: {
    var t: [77]i256 = undefined;
    t[0] = 1;
    for (1..77) |k| t[k] = t[k - 1] * 10;
    break :blk t;
};

/// 10^k as `T` (`k` at most 76, and small enough for `T`).
pub inline fn pow10(comptime T: type, k: usize) T {
    return @intCast(pow10_table[k]);
}

/// Most decimal digits every value below `2^(bits - 1)` can have (9, 18, 38, 76).
fn digitsOf(comptime T: type) u8 {
    return switch (@bitSizeOf(T)) {
        32 => 9,
        64 => 18,
        128 => 38,
        256 => 76,
        else => @compileError("no decimal storage of this width"),
    };
}

/// Whether `|v| < 10^p` (always true when `p` is past the digits of `v`'s type).
pub inline fn fits(v: anytype, p: u8) bool {
    const T = @TypeOf(v);
    if (p > digitsOf(T)) return true;
    return @abs(v) < @as(@Int(.unsigned, @bitSizeOf(T)), @intCast(pow10_table[p]));
}

/// The rows of an integer or decimal operand as `Acc`: a borrowed slice when the storage already
/// is `Acc`, a widened copy (`owned`, freed by `deinit`) otherwise, or a broadcast scalar.
pub fn Widened(comptime Acc: type) type {
    return struct {
        op: dense.Operand(Acc),
        owned: ?[]Acc = null,

        pub fn deinit(self: @This(), allocator: std.mem.Allocator) void {
            if (self.owned) |o| allocator.free(o);
        }
    };
}

/// `ScalarValue` field of the scalars of an integer or decimal type.
fn payloadTag(dt: DataType) ?std.meta.Tag(ScalarValue) {
    return switch (dt) {
        .int8 => .i8,
        .int16 => .i16,
        .int32 => .i32,
        .int64 => .i64,
        .uint8 => .u8,
        .uint16 => .u16,
        .uint32 => .u32,
        .uint64 => .u64,
        .decimal32 => .decimal32,
        .decimal64 => .decimal64,
        .decimal128 => .decimal128,
        .decimal256 => .decimal256,
        else => null,
    };
}

/// The value of a non-null integer or decimal scalar as `Acc` (its payload must be the field of
/// its own type; `InvalidInput` otherwise).
pub fn scalarInt(comptime Acc: type, s: Scalar) KernelError!Acc {
    if (payloadTag(s.data_type) != std.meta.activeTag(s.value)) return error.InvalidInput;
    switch (s.value) {
        inline .i8, .i16, .i32, .i64, .u8, .u16, .u32, .u64, .decimal32, .decimal64, .decimal128, .decimal256 => |x| {
            return if (comptime @bitSizeOf(@TypeOf(x)) > @bitSizeOf(Acc)) (std.math.cast(Acc, x) orelse error.InvalidInput) else x;
        },
        else => return error.InvalidInput,
    }
}

/// Read an integer or decimal operand (array chunk past `validateLayout`, or non-null scalar) as `Acc`.
pub fn widen(comptime Acc: type, allocator: std.mem.Allocator, v: ExecChunkValue) KernelError!Widened(Acc) {
    const tag = std.meta.activeTag(v.dataType());
    inline for (int_like) |pair| if (tag == pair[0]) {
        const T = pair[1];
        switch (v) {
            .scalar => |s| return .{ .op = .{ .scalar = try scalarInt(Acc, s) } },
            .array => |arr| {
                const vals = (PrimitiveArray(T){ .data = arr.data() }).values() catch return error.InvalidInput;
                if (T == Acc) return .{ .op = .{ .slice = vals } };
                if (comptime @bitSizeOf(T) > @bitSizeOf(Acc)) {
                    return error.UnsupportedType;
                } else {
                    const buf = allocator.alloc(Acc, vals.len) catch return error.OutOfMemory;
                    for (buf, vals) |*d, x| d.* = x;
                    return .{ .op = .{ .slice = buf }, .owned = buf };
                }
            },
        }
    };
    return error.UnsupportedType;
}

const at = dense.at;

// ------------------------------------------------------------------------------ arithmetic

const Op = enum { add, subtract, multiply, divide };
const UnaryOp = enum { negate, abs, sign };

/// Register the decimal kernels of add, subtract, multiply, divide (+ `_checked`), negate, abs
/// (+ `_checked`) and sign (after `arithmetic.register`, which creates the functions).
pub fn register(registry: *FunctionRegistry) KernelError!void {
    inline for (comptime std.meta.tags(Op)) |op| {
        try registry.registerScalarKernel(@tagName(op), arithKernel(op, false));
        try registry.registerScalarKernel(@tagName(op) ++ "_checked", arithKernel(op, true));
    }
    inline for (.{ TypeTag.decimal128, TypeTag.decimal256 }) |tag| {
        const T = if (tag == .decimal128) i128 else i256;
        try registry.registerScalarKernel("negate", unaryKernel(tag, T, .negate, false));
        try registry.registerScalarKernel("negate_checked", unaryKernel(tag, T, .negate, true));
        try registry.registerScalarKernel("abs", unaryKernel(tag, T, .abs, false));
        try registry.registerScalarKernel("abs_checked", unaryKernel(tag, T, .abs, true));
        try registry.registerScalarKernel("sign", unaryKernel(tag, T, .sign, false));
    }
}

/// Two integer or decimal operands, at least one of them a decimal.
fn decimalPair(args: []const Datum) bool {
    if (args.len != 2) return false;
    const a = args[0].dataType();
    const b = args[1].dataType();
    if (!isDecimal(a) and !isDecimal(b)) return false;
    return (isDecimal(a) or isInteger(a)) and (isDecimal(b) or isInteger(b));
}

fn arithKernel(comptime op: Op, comptime checked: bool) Kernel {
    const A = Arith(op, checked);
    return .{
        .signature = .{ .arity = 2, .type_check = decimalPair, .options_check = dense.noOptions, .result_type_fn = A.resultType },
        .exec = A.exec,
    };
}

/// Result type of a binary decimal operation and the scale-ups of its operands.
const Plan = struct {
    out: DataType,
    precision: u8,
    ka: u8 = 0,
    kb: u8 = 0,
};

/// Precision and scale of a decimal operand, or `(digits, 0)` for an integer.
fn operandParams(dt: DataType) KernelError!struct { p: i64, s: i64 } {
    if (integerDigits(dt)) |d| return .{ .p = d, .s = 0 };
    const p = try checkedParams(dt);
    return .{ .p = p.precision, .s = p.scale };
}

fn plan(op: Op, a: DataType, b: DataType) KernelError!Plan {
    const x = try operandParams(a);
    const y = try operandParams(b);
    if (x.s < 0 or y.s < 0) return error.NotImplemented;
    const wide = a == .decimal256 or b == .decimal256;
    const max: i64 = if (wide) 76 else 38;
    var ka: i64 = 0;
    var kb: i64 = 0;
    var p: i64 = undefined;
    var s: i64 = undefined;
    switch (op) {
        .add, .subtract => {
            s = @max(x.s, y.s);
            ka = s - x.s;
            kb = s - y.s;
            p = @max(x.p - x.s, y.p - y.s) + s + 1;
        },
        .multiply => {
            s = x.s + y.s;
            p = x.p + y.p + 1;
        },
        .divide => {
            ka = @max(4, x.s + y.p - y.s + 1) + y.s - x.s;
            s = x.s + ka - y.s;
            p = x.p + ka;
        },
    }
    // Every scale-up is below the result precision, so it is at most 76 too.
    if (p > max) return error.Overflow;
    const out: Params = .{ .precision = @intCast(p), .scale = std.math.cast(i32, s) orelse return error.Overflow };
    return .{
        .out = if (wide) .{ .decimal256 = out } else .{ .decimal128 = out },
        .precision = out.precision,
        .ka = @intCast(ka),
        .kb = @intCast(kb),
    };
}

fn Arith(comptime op: Op, comptime checked: bool) type {
    return struct {
        fn resultType(args: []const Datum, _: Options) KernelError!DataType {
            if (args.len != 2) return error.InvalidArity;
            return (try plan(op, args[0].dataType(), args[1].dataType())).out;
        }

        /// Apply the operation element-wise (array, chunked or scalar operands).
        fn exec(ctx: *ExecContext, args: []const Datum, options: Options) KernelError!Datum {
            return dense.execElementwiseWith(ctx, args, options, try resultType(args, options), true, chunk);
        }

        fn chunk(ctx: *ExecContext, values: []const ExecChunkValue, len: usize, _: Options) KernelError!ArrayRef {
            const pl = try plan(op, values[0].dataType(), values[1].dataType());
            return switch (pl.out) {
                .decimal256 => run(i256, ctx, values, len, pl),
                else => run(i128, ctx, values, len, pl),
            };
        }

        fn run(comptime Acc: type, ctx: *ExecContext, values: []const ExecChunkValue, len: usize, pl: Plan) KernelError!ArrayRef {
            const a = try widen(Acc, ctx.allocator, values[0]);
            defer a.deinit(ctx.allocator);
            const b = try widen(Acc, ctx.allocator, values[1]);
            defer b.deinit(ctx.allocator);
            var out = try dense.DenseBinaryOutput(Acc).initValidity(ctx.allocator, len, dense.arrayData(values[0]), dense.arrayData(values[1]), pl.out);
            errdefer out.deinit();
            const fa = pow10(Acc, pl.ka);
            const fb = pow10(Acc, pl.kb);
            switch (a.op) {
                inline else => |x| switch (b.op) {
                    inline else => |y| if (op == .divide)
                        try divide(Acc, x, y, fa, pl.precision, &out)
                    else if (pl.ka == 0 and pl.kb == 0)
                        try loop(Acc, false, x, y, fa, fb, pl.precision, &out)
                    else
                        try loop(Acc, true, x, y, fa, fb, pl.precision, &out),
                },
            }
            return out.finish() catch error.OutOfMemory;
        }

        /// One checked row: the scaled operands, then the operation, each overflow-checked, then
        /// the result precision.
        inline fn checkedRow(comptime Acc: type, comptime scaled: bool, x: Acc, y: Acc, fa: Acc, fb: Acc, precision: u8, over: *bool) Acc {
            var u = x;
            var v = y;
            var o: u1 = 0;
            if (scaled) {
                const mu = @mulWithOverflow(x, fa);
                const mv = @mulWithOverflow(y, fb);
                u = mu[0];
                v = mv[0];
                o = mu[1] | mv[1];
            }
            const r = switch (op) {
                .add => @addWithOverflow(u, v),
                .subtract => @subWithOverflow(u, v),
                .multiply => @mulWithOverflow(u, v),
                .divide => unreachable,
            };
            over.* = (o | r[1]) != 0 or !fits(r[0], precision);
            return r[0];
        }

        fn loop(comptime Acc: type, comptime scaled: bool, x: anytype, y: anytype, fa: Acc, fb: Acc, precision: u8, out: *dense.DenseBinaryOutput(Acc)) KernelError!void {
            const dst = out.values();
            if (!checked) {
                for (dst, 0..) |*d, i| {
                    const u = if (scaled) at(x, i) *% fa else at(x, i);
                    const v = if (scaled) at(y, i) *% fb else at(y, i);
                    d.* = switch (op) {
                        .add => u +% v,
                        .subtract => u -% v,
                        .multiply => u *% v,
                        .divide => unreachable,
                    };
                }
                return;
            }
            // Dense pass with an overflow flag; only when some row overflowed, look for a valid one.
            var any = false;
            for (dst, 0..) |*d, i| {
                var over = false;
                d.* = checkedRow(Acc, scaled, at(x, i), at(y, i), fa, fb, precision, &over);
                any = any or over;
            }
            if (any) for (0..dst.len) |i| {
                var over = false;
                _ = checkedRow(Acc, scaled, at(x, i), at(y, i), fa, fb, precision, &over);
                if (over and out.isValid(i)) return error.Overflow;
            };
        }

        /// `x * 10^ka / y` truncated toward zero. Null rows may hold a zero divisor or the one
        /// overflowing quotient: they divide by 1 or wrap instead of trapping.
        fn divide(comptime Acc: type, x: anytype, y: anytype, fa: Acc, precision: u8, out: *dense.DenseBinaryOutput(Acc)) KernelError!void {
            const dst = out.values();
            var zero = false;
            var any = false;
            for (dst, 0..) |*d, i| {
                var over = false;
                d.* = divRow(Acc, at(x, i), at(y, i), fa, precision, &over);
                zero = zero or at(y, i) == 0;
                any = any or over;
            }
            if (zero) for (0..dst.len) |i| {
                if (at(y, i) == 0 and out.isValid(i)) return error.DivideByZero;
            };
            if (checked and any) for (0..dst.len) |i| {
                var over = false;
                _ = divRow(Acc, at(x, i), at(y, i), fa, precision, &over);
                if (over and out.isValid(i)) return error.Overflow;
            };
        }

        inline fn divRow(comptime Acc: type, x: Acc, y: Acc, fa: Acc, precision: u8, over: *bool) Acc {
            const m = @mulWithOverflow(x, fa);
            const den: Acc = if (y == 0) 1 else y;
            const minus = m[0] == std.math.minInt(Acc) and den == -1;
            const q = if (minus) m[0] else @divTrunc(m[0], den);
            over.* = m[1] != 0 or minus or !fits(q, precision);
            return q;
        }
    };
}

fn unaryKernel(comptime tag: TypeTag, comptime T: type, comptime which: UnaryOp, comptime checked: bool) Kernel {
    const U = Unary(T, which, checked);
    return .{
        .signature = .{ .arity = 1, .type_check = dense.argsOfType(tag, 1), .options_check = dense.noOptions, .result_type_fn = U.resultType },
        .exec = U.exec,
    };
}

fn Unary(comptime T: type, comptime which: UnaryOp, comptime checked: bool) type {
    const Out = if (which == .sign) i64 else T;
    return struct {
        fn resultType(args: []const Datum, _: Options) KernelError!DataType {
            if (args.len != 1) return error.InvalidArity;
            if (which == .sign) return .{ .int64 = {} };
            _ = try checkedParams(args[0].dataType());
            return args[0].dataType();
        }

        fn exec(ctx: *ExecContext, args: []const Datum, options: Options) KernelError!Datum {
            _ = try checkedParams(args[0].dataType());
            return dense.execElementwiseWith(ctx, args, options, try resultType(args, options), true, chunk);
        }

        fn chunk(ctx: *ExecContext, values: []const ExecChunkValue, len: usize, _: Options) KernelError!ArrayRef {
            const out_type = if (which == .sign) DataType{ .int64 = {} } else values[0].dataType();
            const a = try widen(T, ctx.allocator, values[0]);
            defer a.deinit(ctx.allocator);
            var out = try dense.DenseBinaryOutput(Out).initValidity(ctx.allocator, len, dense.arrayData(values[0]), null, out_type);
            errdefer out.deinit();
            const dst = out.values();
            const min = std.math.minInt(T);
            var any = false;
            switch (a.op) {
                inline else => |x| for (dst, 0..) |*d, i| {
                    const v = at(x, i);
                    d.* = switch (which) {
                        .negate => 0 -% v,
                        .abs => if (v < 0) 0 -% v else v,
                        .sign => @as(i64, @intFromBool(v > 0)) - @intFromBool(v < 0),
                    };
                    any = any or v == min;
                },
            }
            if (checked and any) switch (a.op) {
                inline else => |x| for (0..dst.len) |i| {
                    if (at(x, i) == min and out.isValid(i)) return error.Overflow;
                },
            };
            return out.finish() catch error.OutOfMemory;
        }
    };
}

// ------------------------------------------------------------------------------ text

/// Write the decimal digits of `m` right-aligned in `buf`; return them.
fn digitsOfMagnitude(buf: []u8, m: u256) []const u8 {
    var i = buf.len;
    var rest = m;
    // 19 digits per u64 chunk keep the wide divisions few.
    while (rest >= 10_000_000_000_000_000_000) {
        var chunk: u64 = @intCast(rest % 10_000_000_000_000_000_000);
        rest /= 10_000_000_000_000_000_000;
        for (0..19) |_| {
            i -= 1;
            buf[i] = @intCast('0' + chunk % 10);
            chunk /= 10;
        }
    }
    var small: u64 = @intCast(rest);
    while (true) {
        i -= 1;
        buf[i] = @intCast('0' + small % 10);
        small /= 10;
        if (small == 0) break;
    }
    return buf[i..];
}

/// Text of `v` at `scale` as Arrow's `Decimal::ToString`: plain digits with `scale` fraction
/// digits, or `d.dddE+x` when the scale is negative or the adjusted exponent is below -6 (Java
/// `BigDecimal` rules).
pub fn format(buf: *[128]u8, v: i256, scale: i32) []const u8 {
    var digit_buf: [80]u8 = undefined;
    const digits = digitsOfMagnitude(&digit_buf, @abs(v));
    var w: usize = 0;
    if (v < 0) {
        buf[0] = '-';
        w = 1;
    }
    const n: i64 = @intCast(digits.len);
    if (scale == 0) {
        @memcpy(buf[w..][0..digits.len], digits);
        return buf[0 .. w + digits.len];
    }
    const adjusted: i64 = n - 1 - scale;
    if (scale < 0 or adjusted < -6) {
        buf[w] = digits[0];
        w += 1;
        if (digits.len > 1) {
            buf[w] = '.';
            @memcpy(buf[w + 1 ..][0 .. digits.len - 1], digits[1..]);
            w += digits.len;
        }
        const e = std.fmt.bufPrint(buf[w..], "E{s}{d}", .{ if (adjusted >= 0) "+" else "", adjusted }) catch unreachable;
        return buf[0 .. w + e.len];
    }
    if (n > scale) {
        const int_len: usize = @intCast(n - scale);
        @memcpy(buf[w..][0..int_len], digits[0..int_len]);
        buf[w + int_len] = '.';
        @memcpy(buf[w + int_len + 1 ..][0 .. digits.len - int_len], digits[int_len..]);
        return buf[0 .. w + digits.len + 1];
    }
    // 0.000ddd: at most 6 zeros after the point (adjusted >= -6 here).
    const zeros: usize = @intCast(scale - n);
    @memcpy(buf[w..][0..2], "0.");
    @memset(buf[w + 2 ..][0..zeros], '0');
    @memcpy(buf[w + 2 + zeros ..][0..digits.len], digits);
    return buf[0 .. w + 2 + zeros + digits.len];
}

/// A parsed decimal text: `value * 10^-scale`.
pub const Parsed = struct { value: i256, scale: i64 };

/// Parse Arrow's `Decimal::FromString` grammar: an optional sign, digits with an optional `.`
/// and fraction (at least one digit in all), an optional `e` / `E` exponent with an optional sign.
/// Null on anything else (spaces, `inf`, `1,5`) or a value past the i256 range (the magnitude is
/// read as a u256, so the text of `minInt(i256)` parses back: fuzz finding,
/// fuzz/corpus/ipc-stream/regress-138f0637.bin).
pub fn parse(s: []const u8) ?Parsed {
    var i: usize = 0;
    var neg = false;
    if (i < s.len and (s[i] == '+' or s[i] == '-')) {
        neg = s[i] == '-';
        i += 1;
    }
    var value: u256 = 0;
    var digits: usize = 0;
    var frac: i64 = 0;
    var seen_point = false;
    while (i < s.len) : (i += 1) {
        const c = s[i];
        if (c == '.' and !seen_point) {
            seen_point = true;
            continue;
        }
        if (c < '0' or c > '9') break;
        digits += 1;
        if (seen_point) frac += 1;
        const m = @mulWithOverflow(value, 10);
        const a = @addWithOverflow(m[0], @as(u256, c - '0'));
        if (m[1] != 0 or a[1] != 0) return null;
        value = a[0];
    }
    if (digits == 0) return null;
    var exp: i64 = 0;
    if (i < s.len and (s[i] == 'e' or s[i] == 'E')) {
        i += 1;
        var exp_neg = false;
        if (i < s.len and (s[i] == '+' or s[i] == '-')) {
            exp_neg = s[i] == '-';
            i += 1;
        }
        const start = i;
        while (i < s.len and s[i] >= '0' and s[i] <= '9') : (i += 1) {
            // Clamped: an exponent this large cannot be represented anyway.
            exp = @min(exp * 10 + (s[i] - '0'), 1 << 40);
        }
        if (i == start) return null;
        if (exp_neg) exp = -exp;
    }
    if (i != s.len) return null;
    const limit: u256 = @as(u256, 1) << 255;
    if (value > limit or (!neg and value == limit)) return null;
    const signed: i256 = @bitCast(if (neg) 0 -% value else value);
    return .{ .value = signed, .scale = frac - exp };
}

// ------------------------------------------------------------------------------ casts

/// Result of moving `v` from one scale to another.
const Rescaled = struct { value: i256, lost: bool = false, over: bool = false };

/// Scale change of one cast: multiply by `factor` (`up`), or divide by it.
const Scaling = struct {
    up: bool,
    /// 10^|delta| when it fits an i256; null when every nonzero value overflows (up) or every
    /// value divides to 0 (down).
    factor: ?i256,
    /// 10^|delta| mod 2^256 (the wrapping product of an unsafe up-scale).
    wrap: i256,

    fn init(delta: i64) Scaling {
        const k = @abs(delta);
        if (k <= 76) return .{ .up = delta >= 0, .factor = pow10_table[@intCast(k)], .wrap = 0 };
        // 10^k mod 2^256 is 0 from k = 256 (2^k divides it).
        var wrap: i256 = 0;
        if (delta > 0 and k < 256) {
            wrap = 1;
            for (0..@intCast(k)) |_| wrap *%= 10;
        }
        return .{ .up = delta >= 0, .factor = null, .wrap = wrap };
    }

    inline fn apply(self: Scaling, v: i256) Rescaled {
        if (self.up) {
            const f = self.factor orelse return .{ .value = v *% self.wrap, .over = v != 0 };
            const m = @mulWithOverflow(v, f);
            return .{ .value = m[0], .over = m[1] != 0 };
        }
        const f = self.factor orelse return .{ .value = 0, .lost = v != 0 };
        return .{ .value = @divTrunc(v, f), .lost = @rem(v, f) != 0 };
    }
};

const Kind = enum { decimal, integer, float, text, other };

fn kindOf(dt: DataType) Kind {
    if (isDecimal(dt)) return .decimal;
    if (isInteger(dt)) return .integer;
    return switch (dt) {
        .float, .double => .float,
        .string, .large_string, .string_view, .binary, .large_binary, .binary_view => .text,
        else => .other,
    };
}

/// Whether `cast` hands a `from` -> `to` pair to `castArray`.
pub fn handlesCast(from: DataType, to: DataType) bool {
    return isDecimal(from) or isDecimal(to);
}

/// Cast an array that passed `validateLayout` when `from` or `to` is a decimal.
pub fn castArray(ctx: *ExecContext, arr: ArrayRef, to: DataType, safe: bool) KernelError!ArrayRef {
    const data = arr.data();
    const from = data.data_type;
    const fk = kindOf(from);
    const tk = kindOf(to);
    if (fk == .decimal) _ = try checkedParams(from);
    if (tk == .decimal) _ = checkedParams(to) catch return error.InvalidOptions;
    if (fk == .text and tk == .decimal) return fromText(ctx, arr, to, safe);
    if (fk == .float and tk == .decimal) return fromFloat(ctx, data, to, safe);
    if (fk == .decimal and tk == .float) return toFloat(ctx, arr, to);
    if (fk == .decimal and tk == .text) return switch (to) {
        .string, .large_string, .string_view => toText(ctx, arr, to),
        else => error.UnsupportedType,
    };
    if (fk == .integer and tk == .decimal) {
        const p = params(to).?;
        if (p.scale < 0 or p.precision < @as(i64, integerDigits(from).?) + p.scale) return error.InvalidCast;
    }
    if ((fk == .decimal or fk == .integer) and (tk == .decimal or tk == .integer)) return rescale(ctx, arr, to, safe);
    return error.UnsupportedType;
}

/// Storage-width target of a cast: a decimal type or an integer type, with its Zig storage.
fn rescale(ctx: *ExecContext, arr: ArrayRef, to: DataType, safe: bool) KernelError!ArrayRef {
    const from_scale: i64 = if (params(arr.data().data_type)) |p| p.scale else 0;
    const to_scale: i64 = if (params(to)) |p| p.scale else 0;
    const src = try widen(i256, ctx.allocator, .{ .array = arr });
    defer src.deinit(ctx.allocator);
    const tag = std.meta.activeTag(to);
    inline for (int_like) |pair| if (tag == pair[0]) {
        return rescaleTo(pair[1], ctx, arr.data(), src.op.slice, to, Scaling.init(to_scale - from_scale), safe);
    };
    return error.UnsupportedType;
}

fn rescaleTo(comptime T: type, ctx: *ExecContext, data: *const ArrayData, src: []const i256, to: DataType, scaling: Scaling, safe: bool) KernelError!ArrayRef {
    var out = try dense.DenseBinaryOutput(T).initValidity(ctx.allocator, data.length, data, null, to);
    errdefer out.deinit();
    const precision: ?u8 = if (params(to)) |p| p.precision else null;
    for (out.values(), src, 0..) |*d, v, i| {
        const r = scaling.apply(v);
        d.* = @bitCast(@as(@Int(.signed, @bitSizeOf(T)), @truncate(r.value)));
        if (!safe) continue;
        const bad = r.lost or r.over or (if (precision) |p| !fits(r.value, p) else false) or std.math.cast(T, r.value) == null;
        if (bad and out.isValid(i)) return error.InvalidCast;
    }
    return out.finish() catch error.OutOfMemory;
}

/// Text to decimal: `parse`, then the target scale like a decimal cast (safe: a dropped digit,
/// an overflow or a value past the precision is `InvalidCast`; unsafe truncates and wraps). A
/// parsed scale below minus the largest precision is `InvalidCast` in both modes (Arrow: "cannot
/// be represented").
fn fromText(ctx: *ExecContext, arr: ArrayRef, to: DataType, safe: bool) KernelError!ArrayRef {
    const data = arr.data();
    const p = params(to).?;
    const min_scale = -@as(i64, maxPrecision(to));
    const rows = try dense.StringOperand.init(.{ .array = arr });
    const bits = dense.validityBits(data);
    inline for (decimal_types) |pair| if (std.meta.activeTag(to) == pair[0]) {
        const T = pair[1];
        var out = try dense.DenseBinaryOutput(T).initValidity(ctx.allocator, data.length, data, null, to);
        errdefer out.deinit();
        switch (rows) {
            inline else => |r| for (out.values(), 0..) |*d, i| {
                d.* = 0;
                if (bits) |b| if (!bitmap.bitIsSet(b, data.offset + i)) continue;
                const parsed = parse(r.get(i)) orelse return error.InvalidCast;
                if (parsed.scale < min_scale) return error.InvalidCast;
                const delta = std.math.sub(i64, p.scale, parsed.scale) catch return error.InvalidCast;
                const x = Scaling.init(delta).apply(parsed.value);
                if (safe and (x.lost or x.over or !fits(x.value, p.precision) or std.math.cast(T, x.value) == null)) return error.InvalidCast;
                d.* = @truncate(x.value);
            },
        }
        return out.finish() catch error.OutOfMemory;
    };
    return error.UnsupportedType;
}

/// Decimal to string / large_string / string_view (`format`).
fn toText(ctx: *ExecContext, arr: ArrayRef, to: DataType) KernelError!ArrayRef {
    const data = arr.data();
    const scale = params(data.data_type).?.scale;
    const src = try widen(i256, ctx.allocator, .{ .array = arr });
    defer src.deinit(ctx.allocator);
    const bits = dense.validityBits(data);
    var out = try dense.VarBinaryOutput.init(ctx.allocator, to, data.length, try dense.combinedValidity(ctx.allocator, data.length, &.{data}), data.length * 8);
    errdefer out.deinit();
    for (src.op.slice, 0..) |v, i| {
        if (bits) |b| if (!bitmap.bitIsSet(b, data.offset + i)) {
            try out.commit(0);
            continue;
        };
        var buf: [128]u8 = undefined;
        try out.append(format(&buf, v, scale));
    }
    return out.finish();
}

/// 10^-76 .. 10^76 as `F`, each correctly rounded (0 / inf past the range of `F`): Arrow's
/// `kFloatPowersOfTen` / `kDoublePowersOfTen`.
fn powersOfTen(comptime F: type) [153]F {
    @setEvalBranchQuota(2_000_000);
    var t: [153]F = undefined;
    for (&t, 0..) |*p, i| p.* = std.fmt.parseFloat(F, std.fmt.comptimePrint("1e{d}", .{@as(i32, @intCast(i)) - 76})) catch unreachable;
    return t;
}

const powers_f32 = powersOfTen(f32);
const powers_f64 = powersOfTen(f64);

fn powerOfTen(comptime F: type, exp: i32) F {
    if (exp >= -76 and exp <= 76) return (if (F == f32) powers_f32 else powers_f64)[@intCast(exp + 76)];
    return std.math.pow(F, 10, @floatFromInt(exp));
}

/// Arrow's `ToRealPositiveNoSplit`: the 64-bit words of `m` as `F`, most significant first, times
/// `10^-scale`.
fn noSplit(comptime F: type, m: u256, scale: i32) F {
    var x: F = 0;
    var k: usize = 4;
    while (k > 0) {
        k -= 1;
        const word: u64 = @truncate(m >> @intCast(64 * k));
        x = x * 18446744073709551616.0 + @as(F, @floatFromInt(word));
    }
    return x * powerOfTen(F, -scale);
}

/// `v * 10^-scale` as `F` the way Arrow C++ computes it (`DecimalRealConversion::ToReal`, so the
/// results are bit-identical to pyarrow's): an integer, or a value below 2^24 / 2^53, is
/// converted then multiplied by the power of ten; a larger value is split into its whole and
/// fraction parts first.
pub fn toFloatValue(comptime F: type, v: i256, scale: i32) F {
    const m: u256 = @abs(v);
    const precise: u256 = if (F == f32) (1 << 24) - 1 else (1 << 53) - 1;
    var x: F = undefined;
    if (scale <= 0 or m <= precise) {
        x = noSplit(F, m, scale);
    } else if (scale <= 76) {
        const p: u256 = @intCast(pow10_table[@intCast(scale)]);
        x = noSplit(F, m / p, 0) + noSplit(F, m % p, scale);
    } else {
        x = noSplit(F, m, scale);
    }
    return if (v < 0) -x else x;
}

fn toFloat(ctx: *ExecContext, arr: ArrayRef, to: DataType) KernelError!ArrayRef {
    const data = arr.data();
    const scale = params(data.data_type).?.scale;
    const src = try widen(i256, ctx.allocator, .{ .array = arr });
    defer src.deinit(ctx.allocator);
    if (to == .float) return floatRun(f32, ctx, data, src.op.slice, scale);
    return floatRun(f64, ctx, data, src.op.slice, scale);
}

fn floatRun(comptime F: type, ctx: *ExecContext, data: *const ArrayData, src: []const i256, scale: i32) KernelError!ArrayRef {
    var out = try dense.DenseBinaryOutput(F).initValidity(ctx.allocator, data.length, data, null, dense.dataTypeOf(F));
    errdefer out.deinit();
    for (out.values(), src) |*d, v| d.* = toFloatValue(F, v, scale);
    return out.finish() catch error.OutOfMemory;
}

/// `x * 10^scale` as Arrow C++'s `FromReal`: for a scale of 0 or more the exact binary value of
/// `x` rounded half to even; for a negative scale `nearbyint(x * 10^scale)` computed in `F`
/// (Arrow's `FromPositiveRealApprox`). Null for NaN, inf, a result of `precision` digits or more,
/// or a scale past 76.
pub fn fromFloatValue(comptime F: type, x: F, precision: u8, scale: i32) ?i256 {
    if (!std.math.isFinite(x)) return null;
    if (scale < 0) return fromFloatApprox(F, x, precision, scale);
    return fromFloatExact(x, precision, scale);
}

fn fromFloatApprox(comptime F: type, x: F, precision: u8, scale: i32) ?i256 {
    const y = @abs(x) * powerOfTen(F, scale);
    // nearbyint in the default rounding mode: half to even.
    const r = if (@abs(y - @trunc(y)) == 0.5) 2 * @round(y / 2) else @round(y);
    if (r >= powerOfTen(F, precision)) return null;
    const v: i256 = @intFromFloat(r);
    return if (x < 0) -v else v;
}

fn fromFloatExact(x: f64, precision: u8, scale: i32) ?i256 {
    if (x == 0) return 0;
    if (scale > 76) return null;
    const bits: u64 = @bitCast(x);
    const exp_bits: i32 = @intCast((bits >> 52) & 0x7FF);
    const frac = bits & ((@as(u64, 1) << 52) - 1);
    const m: u64 = if (exp_bits == 0) frac else frac | (@as(u64, 1) << 52);
    const e: i32 = if (exp_bits == 0) -1074 else exp_bits - 1075;
    // |x| >= 2^(52 + e) for a normal x: past e = 460 even 10^-76 of it exceeds 2^256.
    if (e > 460) return null;
    // Below 2^-547 the result rounds to 0 whatever the scale (10^76 < 2^253).
    if (e < -600) return 0;
    var num: u1024 = @as(u1024, m) * @as(u1024, @intCast(pow10_table[@intCast(@max(scale, 0))]));
    if (e > 0) num <<= @intCast(e);
    var den: u1024 = @intCast(pow10_table[@intCast(@max(-scale, 0))]);
    if (e < 0) den <<= @intCast(-e);
    const q = num / den;
    const r = num - q * den;
    const twice = r << 1;
    const n = q + @intFromBool(twice > den or (twice == den and q & 1 == 1));
    if (n >= @as(u1024, @intCast(pow10_table[precision]))) return null;
    const v: i256 = @intCast(n);
    return if (bits >> 63 != 0) -v else v;
}

fn fromFloat(ctx: *ExecContext, data: *const ArrayData, to: DataType, safe: bool) KernelError!ArrayRef {
    const p = params(to).?;
    inline for (decimal_types) |pair| if (std.meta.activeTag(to) == pair[0]) {
        const T = pair[1];
        var out = try dense.DenseBinaryOutput(T).initValidity(ctx.allocator, data.length, data, null, to);
        errdefer out.deinit();
        inline for (.{ f32, f64 }) |F| if (std.meta.activeTag(data.data_type) == comptime std.meta.activeTag(dense.dataTypeOf(F))) {
            const src = (PrimitiveArray(F){ .data = data }).values() catch return error.InvalidInput;
            for (out.values(), src, 0..) |*d, x, i| {
                if (fromFloatValue(F, x, p.precision, p.scale)) |v| {
                    d.* = @intCast(v);
                } else {
                    d.* = 0;
                    if (safe and out.isValid(i)) return error.InvalidCast;
                }
            }
        };
        return out.finish() catch error.OutOfMemory;
    };
    return error.UnsupportedType;
}

// ------------------------------------------------------------------------------ comparison helpers

/// Order of `a * 10^k` and `b` (exact: a product past the i256 range is beyond every `b`).
pub inline fn orderScaled(a: i256, k: u32, b: i256) std.math.Order {
    if (k == 0) return std.math.order(a, b);
    if (k > 76) {
        if (a == 0) return std.math.order(0, b);
        return if (a > 0) .gt else .lt;
    }
    const m = @mulWithOverflow(a, pow10_table[k]);
    if (m[1] != 0) return if (a > 0) .gt else .lt;
    return std.math.order(m[0], b);
}

/// Scale of a decimal operand (0 for an integer) for exact comparisons.
pub fn scaleOf(dt: DataType) i64 {
    return if (params(dt)) |p| p.scale else 0;
}

test "decimal format and parse follow Arrow's ToString / FromString" {
    var buf: [128]u8 = undefined;
    try std.testing.expectEqualStrings("1.25", format(&buf, 125, 2));
    try std.testing.expectEqualStrings("-0.05", format(&buf, -5, 2));
    try std.testing.expectEqualStrings("100.00", format(&buf, 10000, 2));
    try std.testing.expectEqualStrings("1.2E+3", format(&buf, 12, -2));
    try std.testing.expectEqualStrings("-5E+2", format(&buf, -5, -2));
    try std.testing.expectEqualStrings("1E-25", format(&buf, 1, 25));
    try std.testing.expectEqualStrings("0.000001", format(&buf, 1, 6));
    try std.testing.expectEqualStrings("1E-7", format(&buf, 1, 7));
    try std.testing.expectEqualStrings("-1.23E-7", format(&buf, -123, 9));
    try std.testing.expectEqualStrings("0E-31", format(&buf, 0, 31));
    try std.testing.expectEqualStrings("-57896044618658097711785492504343953926634992332820282019728792003956564819968", format(&buf, std.math.minInt(i256), 0));
    const p = parse("-1.50e-1").?;
    try std.testing.expectEqual(@as(i256, -150), p.value);
    try std.testing.expectEqual(@as(i64, 3), p.scale);
    try std.testing.expectEqual(@as(i64, -2), parse("1.2E+3").?.scale);
    for ([_][]const u8{ "", "+.", "-", "1e", "e5", "1.2.3", " 1", "1,5", "inf", "1e+", "0x10" }) |bad| try std.testing.expect(parse(bad) == null);
    var ones: [78]u8 = undefined;
    @memset(&ones, '1');
    try std.testing.expect(parse(&ones) == null);
    try std.testing.expectEqual(@as(i256, 5), parse(".5").?.value);
    // Every i256 formats and parses back, minInt included.
    for ([_]i256{ std.math.minInt(i256), std.math.maxInt(i256), -1, 0 }) |v| {
        try std.testing.expectEqual(v, parse(format(&buf, v, 0)).?.value);
        const q = parse(format(&buf, v, 40)).?;
        try std.testing.expectEqual(v, q.value);
        try std.testing.expectEqual(@as(i64, 40), q.scale);
    }
    try std.testing.expect(parse("57896044618658097711785492504343953926634992332820282019728792003956564819968") == null);
    try std.testing.expectEqual(@as(i256, 5), parse("5.").?.value);
}

test "decimal float conversions match Arrow's FromReal / ToReal" {
    try std.testing.expectEqual(@as(?i256, 12), fromFloatValue(f64, 0.125, 10, 2));
    try std.testing.expectEqual(@as(?i256, 38), fromFloatValue(f64, 0.375, 10, 2));
    try std.testing.expectEqual(@as(?i256, -12), fromFloatValue(f64, -0.125, 10, 2));
    try std.testing.expectEqual(@as(?i256, 2), fromFloatValue(f64, 2.5, 10, 0));
    try std.testing.expectEqual(@as(?i256, 1), fromFloatValue(f64, 0.05, 10, 1));
    try std.testing.expectEqual(@as(?i256, 12), fromFloatValue(f64, 1250.0, 5, -2));
    try std.testing.expectEqual(@as(?i256, 125), fromFloatValue(f64, 0.00125, 3, 5));
    try std.testing.expectEqual(@as(?i256, null), fromFloatValue(f64, 1e10, 5, 2));
    try std.testing.expectEqual(@as(?i256, null), fromFloatValue(f64, std.math.nan(f64), 5, 2));
    try std.testing.expectEqual(@as(?i256, 0), fromFloatValue(f64, 5e-324, 76, 76));
    try std.testing.expectEqual(@as(f64, 0.1), toFloatValue(f64, 1, 1));
    try std.testing.expectEqual(@as(f64, 1200.0), toFloatValue(f64, 12, -2));
    // Arrow multiplies by the float 1e-2: 0.099999994, one ulp below the nearest float to 0.1.
    try std.testing.expectEqual(@as(f32, 0.099999994), toFloatValue(f32, 10, 2));
    try std.testing.expectEqual(@as(f64, 1.2345678901234568e43), toFloatValue(f64, 123456789012345678901234567890123456789012345, 1));
}
