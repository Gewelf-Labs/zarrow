//! Rounding kernels (Arrow C++ "Rounding functions"): round (`RoundOptions`: ndigits,
//! round_mode), round_to_multiple (`RoundToMultipleOptions`: multiple, round_mode), round_binary
//! (the digits as a second, integer argument; `RoundBinaryOptions`), ceil, floor and trunc.
//!
//! - float32 / float64 keep their type and follow Arrow's algorithm bit for bit: the value is
//!   scaled by a power of ten built like `RoundUtil::Pow10` (a float32 table up to 1e15, then
//!   repeated multiplication in double, cast to the input type), rounded (`frac == 0.5` breaks
//!   ties by `round_mode`, any other fraction rounds to nearest) and scaled back; NaN and
//!   infinities pass through; a value that becomes infinite is `Overflow` (Arrow's "overflow
//!   occurred during rounding"). round_to_multiple divides by the multiple instead.
//! - int8..int64, uint8..uint64 keep their type (round and round_to_multiple; ceil / floor /
//!   trunc cast integers to float64 first, as Arrow): `ndigits >= 0` is the identity, a negative
//!   `ndigits` rounds to a multiple of `10^-ndigits`; more digits than the type holds is
//!   `InvalidOptions` (also for an empty input, as Arrow); a result past the type is `Overflow`.
//! - decimal128 / decimal256 keep their type (as Arrow; decimal32 / decimal64 have no rounding
//!   kernels there): the stored value is rounded to a multiple of `10^(scale - ndigits)`; when
//!   that exponent reaches the precision every valid row is `InvalidOptions`, when it is negative
//!   the value is kept; a result past the precision is `Overflow`.
//! - round_to_multiple casts its multiple to the input type (safe: a lost fraction is
//!   `InvalidCast`); a null, zero, negative or NaN multiple is `InvalidOptions`.
//! - round_binary takes its digits per row (any integer type, cast to int32; for a decimal128 /
//!   decimal256 input also a decimal of any width and scale or a float, cast safely to int32 as
//!   pyarrow 26: a fraction is `InvalidCast`); a row whose digits are out of range for an integer
//!   or decimal input is `InvalidInput`.
//!
//! Errors are only raised for valid rows. Null handling is intersection. Arguments that are all
//! of the null type give a null-type result of their length (Arrow's null kernels).
const std = @import("std");
const core = @import("core.zig");
const dense = @import("dense.zig");
const decimal = @import("decimal.zig");

const ArrayRef = core.ArrayRef;
const PrimitiveArray = @import("../array/primitive_array.zig").PrimitiveArray;
const DataType = core.DataType;
const Datum = core.Datum;
const ExecChunkValue = core.ExecChunkValue;
const ExecContext = core.ExecContext;
const FunctionRegistry = core.FunctionRegistry;
const Kernel = core.Kernel;
const KernelError = core.KernelError;
const Options = core.Options;
const RoundMode = core.RoundMode;

const Func = enum { round, round_to_multiple, round_binary, ceil, floor, trunc };

/// Register round, round_to_multiple, round_binary, ceil, floor and trunc.
pub fn register(registry: *FunctionRegistry) KernelError!void {
    inline for (.{ f32, f64, i8, i16, i32, i64, u8, u16, u32, u64, i128, i256 }) |T| {
        const is_int = @typeInfo(T) == .int and @bitSizeOf(T) <= 64;
        inline for (comptime std.meta.tags(Func)) |f| {
            // Integers have no ceil / floor / trunc kernels: they are cast to float64.
            if (comptime !(is_int and (f == .ceil or f == .floor or f == .trunc))) {
                try registry.registerScalarKernel(@tagName(f), kernel(T, f));
            }
        }
    }
    inline for (.{ "ceil", "floor", "trunc" }) |name| try registry.setImplicitCast(name, .scalar, .integer_to_float);
    // Null-type arguments give a null-type result (Arrow's null kernels).
    inline for (comptime std.meta.tags(Func)) |f| try registry.setNullTypeResult(@tagName(f), .scalar);
}

fn storageTag(comptime T: type) std.meta.Tag(DataType) {
    return switch (T) {
        i128 => .decimal128,
        i256 => .decimal256,
        else => std.meta.activeTag(dense.dataTypeOf(T)),
    };
}

fn isIntegerTag(tag: std.meta.Tag(DataType)) bool {
    return switch (tag) {
        .int8, .int16, .int32, .int64, .uint8, .uint16, .uint32, .uint64 => true,
        else => false,
    };
}

/// The digits of round_binary on decimal128 / decimal256 `tag`: an integer, or a decimal of any
/// width or a float (cast to int32 like the integers; a fraction or a value past int32 fails the
/// cast, as pyarrow 26).
fn digitsOk(tag: std.meta.Tag(DataType), digits: std.meta.Tag(DataType)) bool {
    if (isIntegerTag(digits)) return true;
    if (tag != .decimal128 and tag != .decimal256) return false;
    return switch (digits) {
        .decimal32, .decimal64, .decimal128, .decimal256, .half_float, .float, .double => true,
        else => false,
    };
}

fn kernel(comptime T: type, comptime f: Func) Kernel {
    const tag = comptime storageTag(T);
    return .{
        .signature = .{
            .arity = if (f == .round_binary) 2 else 1,
            .type_check = struct {
                fn check(args: []const Datum) bool {
                    if (f == .round_binary) return args.len == 2 and std.meta.activeTag(args[0].dataType()) == tag and digitsOk(tag, std.meta.activeTag(args[1].dataType()));
                    return args.len == 1 and std.meta.activeTag(args[0].dataType()) == tag;
                }
            }.check,
            .options_check = struct {
                fn check(options: Options) bool {
                    return options == .none or switch (f) {
                        .round => options == .round,
                        .round_to_multiple => options == .round_to_multiple,
                        .round_binary => options == .round_binary,
                        else => false,
                    };
                }
            }.check,
            .result_type_fn = struct {
                fn infer(args: []const Datum, _: Options) KernelError!DataType {
                    if (args.len == 0) return error.InvalidArity;
                    if (T == i128 or T == i256) _ = try decimal.checkedParams(args[0].dataType());
                    return args[0].dataType();
                }
            }.infer,
        },
        .exec = Rounder(T, f).exec,
    };
}

/// Wide type of the integer / decimal rounding arithmetic (the floor and ceil of any stored
/// value and multiple fit).
fn Wide(comptime T: type) type {
    return switch (T) {
        i128 => i256,
        i256 => i512,
        else => i128,
    };
}

/// What every row of one call shares.
fn Params(comptime T: type) type {
    const is_float = @typeInfo(T) == .float;
    return struct {
        mode: RoundMode,
        /// floats: 10^|ndigits| (round) or the multiple; integers / decimals: the multiple in
        /// stored units (1: keep the value).
        m: if (is_float) T else Wide(T),
        nd: i64 = 0,
        /// Decimals: results must stay below 10^precision.
        precision: u8 = 0,
        /// Decimals: every valid row is an error (the multiple reaches the precision).
        reject: bool = false,
    };
}

/// Arrow's `RoundUtil::Pow10(|n|)` as `T`: a float32 table, then repeated multiplication in
/// double (stopped at infinity: Arrow would loop on a huge `n`).
pub fn pow10Float(comptime T: type, n: u64) T {
    const lut = comptime blk: {
        var t: [16]f64 = undefined;
        var p: f32 = 1;
        for (&t) |*e| {
            e.* = p;
            p *= 10;
        }
        break :blk t;
    };
    var p: f64 = lut[@min(n, 15)];
    var a = n;
    while (a >= 16 and !std.math.isInf(p)) : (a -= 1) p *= 10;
    return @floatCast(p);
}

inline fn isHalf(mode: RoundMode) bool {
    return @intFromEnum(mode) >= @intFromEnum(RoundMode.half_down);
}

/// Arrow's `RoundImpl` for a float already scaled.
inline fn roundScaled(comptime T: type, v: T, mode: RoundMode) T {
    return switch (mode) {
        .down, .half_down => @floor(v),
        .up, .half_up => @ceil(v),
        .towards_zero, .half_towards_zero => @trunc(v),
        .towards_infinity => if (std.math.signbit(v)) @floor(v) else @ceil(v),
        .half_towards_infinity => @round(v),
        .half_to_even => @round(v * 0.5) * 2,
        .half_to_odd => @floor(v * 0.5) + @ceil(v * 0.5),
    };
}

/// Round `x` with `ndigits = nd` (`p` = 10^|nd|), or to a multiple `p` when `multiple`; null
/// when the result overflows.
pub inline fn roundFloat(comptime T: type, x: T, nd: i64, p: T, mode: RoundMode, comptime multiple: bool) ?T {
    if (!std.math.isFinite(x)) return x;
    var v = if (multiple or nd < 0) x / p else x * p;
    const frac = v - @floor(v);
    if (frac == 0) return x;
    v = if (isHalf(mode) and frac != 0.5) @round(v) else roundScaled(T, v, mode);
    v = if (!multiple and nd > 0) v / p else v * p;
    return if (std.math.isFinite(v)) v else null;
}

/// Round the integer `v` to a multiple of `m > 0` (Arrow's integer and decimal rounding: floor and
/// ceil are the neighbouring multiples; ties by `mode`).
pub inline fn roundInt(comptime W: type, v: W, m: W, mode: RoundMode) W {
    const r = @mod(v, m);
    if (r == 0) return v;
    const fl = v - r;
    const ce = fl + m;
    const pos = v > 0;
    return switch (mode) {
        .down => fl,
        .up => ce,
        .towards_zero => if (pos) fl else ce,
        .towards_infinity => if (pos) ce else fl,
        else => blk: {
            const d = 2 * r;
            if (d < m) break :blk fl;
            if (d > m) break :blk ce;
            break :blk switch (mode) {
                .half_down => fl,
                .half_up => ce,
                .half_towards_zero => if (pos) fl else ce,
                .half_towards_infinity => if (pos) ce else fl,
                .half_to_even => if (@mod(@divFloor(fl, m), 2) == 0) fl else ce,
                else => if (@mod(@divFloor(fl, m), 2) == 0) ce else fl,
            };
        },
    };
}

/// Most decimal digits of every value of an integer type (Arrow's `digits10`).
fn digits10(comptime T: type) u8 {
    return switch (T) {
        i8, u8 => 2,
        i16, u16 => 4,
        i32, u32 => 9,
        i64 => 18,
        u64 => 19,
        else => unreachable,
    };
}

fn Rounder(comptime T: type, comptime f: Func) type {
    const is_float = @typeInfo(T) == .float;
    const is_decimal = T == i128 or T == i256;
    const W = if (is_float) T else Wide(T);
    const P = Params(T);
    return struct {
        fn exec(ctx: *ExecContext, args: []const Datum, options: Options) KernelError!Datum {
            const out_type = args[0].dataType();
            const precision: u8 = if (is_decimal) (try decimal.checkedParams(out_type)).precision else 0;
            const scale: i64 = if (is_decimal) decimal.scaleOf(out_type) else 0;
            if (f == .round_binary) {
                const mode = if (options == .round_binary) options.round_binary.round_mode else RoundMode.half_to_even;
                var p: P = .{ .mode = mode, .m = 1, .precision = precision };
                const call: Call = .{ .params = &p, .scale = scale };
                // The digits as int32 (Arrow casts them).
                var digits = if (args[1].dataType() == .int32) args[1].retain() else try ctx.invoke("cast", .scalar, &.{args[1]}, .{ .cast = .{ .to_type = .{ .int32 = {} } } });
                defer digits.release();
                return dense.execElementwiseWith(ctx, &.{ args[0], digits }, .{ .custom = .{ .tag = "round", .payload = &call } }, out_type, true, chunk);
            }
            var p: P = .{ .mode = switch (f) {
                .round => if (options == .round) options.round.round_mode else .half_to_even,
                .round_to_multiple => if (options == .round_to_multiple) options.round_to_multiple.round_mode else .half_to_even,
                .ceil => .up,
                .floor => .down,
                .trunc => .towards_zero,
                .round_binary => unreachable,
            }, .m = 1, .precision = precision };
            if (f == .round_to_multiple) {
                const multiple = if (options == .round_to_multiple) options.round_to_multiple.multiple else core.Scalar.init(.{ .double = {} }, .{ .f64 = 1.0 });
                p.m = try castMultiple(ctx, multiple, out_type);
            } else {
                const nd: i64 = if (f == .round and options == .round) options.round.ndigits else 0;
                p.nd = nd;
                if (is_float) {
                    p.m = pow10Float(T, @abs(nd));
                } else if (is_decimal) {
                    p.m, p.reject = decimalMultiple(scale, nd, precision);
                } else if (nd < 0) {
                    if (@abs(nd) > digits10(T)) return error.InvalidOptions;
                    p.m = decimal.pow10(W, @intCast(@abs(nd)));
                }
            }
            const call: Call = .{ .params = &p, .scale = scale };
            return dense.execElementwiseWith(ctx, args, .{ .custom = .{ .tag = "round", .payload = &call } }, out_type, true, chunk);
        }

        const Call = struct { params: *const P, scale: i64 };

        /// The multiple of a decimal round to `nd` digits at `scale` (1 when nothing is rounded)
        /// and whether every valid row is rejected (the multiple reaches the precision).
        fn decimalMultiple(scale: i64, nd: i64, precision: u8) struct { W, bool } {
            const pow = @as(i128, scale) - nd;
            if (pow >= precision) return .{ 1, true };
            if (pow <= 0) return .{ 1, false };
            return .{ decimal.pow10(W, @intCast(pow)), false };
        }

        /// The multiple of round_to_multiple as the input type (safe cast), checked positive.
        fn castMultiple(ctx: *ExecContext, multiple: core.Scalar, out_type: DataType) KernelError!@FieldType(P, "m") {
            if (multiple.isNull()) return error.InvalidOptions;
            var cast = try ctx.invoke("cast", .scalar, &.{.{ .scalar = multiple }}, .{ .cast = .{ .to_type = out_type } });
            defer cast.release();
            const s = cast.asScalar() orelse return error.InvalidOptions;
            if (s.isNull()) return error.InvalidOptions;
            const m: T = if (is_decimal) try decimal.scalarInt(T, s) else try dense.scalarAs(T, s);
            if (!(m > 0)) return error.InvalidOptions;
            return m;
        }

        fn operandOf(comptime V: type, v: ExecChunkValue) KernelError!dense.Operand(V) {
            return switch (v) {
                .array => |arr| .{ .slice = (PrimitiveArray(V){ .data = arr.data() }).values() catch return error.InvalidInput },
                .scalar => |s| .{ .scalar = if (V == i128 or V == i256) try decimal.scalarInt(V, s) else try dense.scalarAs(V, s) },
            };
        }

        fn chunk(ctx: *ExecContext, values: []const ExecChunkValue, len: usize, options: Options) KernelError!ArrayRef {
            const call: *const Call = @ptrCast(@alignCast(options.custom.payload.?));
            const p = call.params;
            const out_type = values[0].dataType();
            const rhs_data = if (f == .round_binary) dense.arrayData(values[1]) else null;
            var out = try dense.DenseBinaryOutput(T).initValidity(ctx.allocator, len, dense.arrayData(values[0]), rhs_data, out_type);
            errdefer out.deinit();
            const dst = out.values();
            const xs = try operandOf(T, values[0]);
            if (f == .round_binary) {
                const ds = try operandOf(i32, values[1]);
                switch (xs) {
                    inline else => |a| switch (ds) {
                        inline else => |b| try roundRows(a, b, call, &out),
                    },
                }
            } else switch (xs) {
                inline else => |a| {
                    var bad = false;
                    if (is_decimal and p.reject) {
                        @memset(dst, 0);
                        bad = true;
                    } else for (dst, 0..) |*d, i| {
                        const r = one(dense.at(a, i), p.*, p.nd);
                        d.* = r orelse 0;
                        bad = bad or r == null;
                    }
                    if (bad) for (0..len) |i| {
                        if (out.isValid(i) and (p.reject or one(dense.at(a, i), p.*, p.nd) == null)) {
                            return if (p.reject) error.InvalidOptions else error.Overflow;
                        }
                    };
                },
            }
            return out.finish() catch error.OutOfMemory;
        }

        /// Row `x` rounded with the call's parameters; null when the result does not fit.
        inline fn one(x: T, p: P, nd: i64) ?T {
            if (is_float) {
                if (f == .ceil) return @ceil(x);
                if (f == .floor) return @floor(x);
                if (f == .trunc) return @trunc(x);
                return roundFloat(T, x, nd, p.m, p.mode, f == .round_to_multiple);
            }
            if (p.m == 1) return x;
            const r = roundInt(W, x, p.m, p.mode);
            if (is_decimal) {
                const lim = decimal.pow10(W, p.precision);
                return if (r < lim and r > -lim) @intCast(r) else null;
            }
            return std.math.cast(T, r);
        }

        /// round_binary: the digits of every row (`b`) build the row's parameters.
        fn roundRows(a: anytype, b: anytype, call: *const Call, out: *dense.DenseBinaryOutput(T)) KernelError!void {
            const p = call.params.*;
            var last_nd: i64 = 0;
            var row: P = p;
            if (is_float) row.m = 1;
            for (out.values(), 0..) |*d, i| {
                const nd: i64 = dense.at(b, i);
                if (i == 0 or nd != last_nd) {
                    last_nd = nd;
                    row = p;
                    row.nd = nd;
                    if (is_float) {
                        row.m = pow10Float(T, @abs(nd));
                    } else if (is_decimal) {
                        row.m, row.reject = decimalMultiple(call.scale, nd, p.precision);
                    } else {
                        row.m = 1;
                        row.reject = nd < 0 and @abs(nd) > digits10(T);
                        if (nd < 0 and !row.reject) row.m = decimal.pow10(W, @intCast(@abs(nd)));
                    }
                }
                const valid = out.isValid(i);
                if (row.reject) {
                    if (valid) return error.InvalidInput;
                    d.* = 0;
                    continue;
                }
                const r = one(dense.at(a, i), row, nd);
                if (r == null and valid) return error.Overflow;
                d.* = r orelse 0;
            }
        }
    };
}
