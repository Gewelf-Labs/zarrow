//! Temporal arithmetic (Arrow C++ "Arithmetic functions" on temporal types), registered on add,
//! subtract, multiply, divide, negate, abs and their `_checked` variants:
//!
//! - add: timestamp + duration and duration + timestamp (timestamp, its timezone kept), duration +
//!   duration (duration), date32 / date64 + duration and duration + date (a timestamp: date32 as
//!   seconds, date64 as milliseconds, no timezone), time32 / time64 + duration (a time).
//! - subtract: timestamp - duration (timestamp), timestamp - timestamp (duration; both naive or
//!   both with a timezone, else `InvalidInput` like Arrow's "zoned and non-zoned" error),
//!   duration - duration, date - date (duration: seconds for date32, milliseconds with a date64),
//!   date - duration (timestamp), time - duration (time), time - time (duration).
//! - multiply: duration * integer and integer * duration (duration); divide: duration / integer
//!   (duration, truncating; a zero divisor is `DivideByZero`) and duration / duration (float64).
//! - negate, abs: duration.
//!
//! Operands of different units go to the finer unit first; a value that does not fit the finer
//! unit is `InvalidCast` in both variants (Arrow's cast error). Unchecked results wrap (saturate
//! with `ExecConfig.overflow_mode = saturating`), checked ones return `Overflow`. A time result
//! outside one day is `InvalidInput` in both variants (Arrow's range check). Integer operands of
//! multiply / divide are cast to int64 first. Null handling is intersection; errors only come
//! from valid rows.
const std = @import("std");
const core = @import("core.zig");
const dense = @import("dense.zig");
const datatype = @import("../datatype.zig");

const ArrayRef = core.ArrayRef;
const PrimitiveArray = @import("../array/primitive_array.zig").PrimitiveArray;
const DataType = core.DataType;
const Datum = core.Datum;
const ExecChunkValue = core.ExecChunkValue;
const ExecContext = core.ExecContext;
const FunctionRegistry = core.FunctionRegistry;
const KernelError = core.KernelError;
const Options = core.Options;
const Scalar = core.Scalar;
const TimeUnit = datatype.TimeUnit;

const Op = enum { add, subtract, multiply, divide, negate, abs };

/// Operand kinds of a kernel: timestamp, duration, date, time, integer.
const K = enum { ts, dur, date, time, int };

/// The operand patterns registered per function.
const patterns = .{
    .{ Op.add, .{ .{ K.ts, K.dur }, .{ K.dur, K.ts }, .{ K.dur, K.dur }, .{ K.date, K.dur }, .{ K.dur, K.date }, .{ K.time, K.dur } } },
    .{ Op.subtract, .{ .{ K.ts, K.dur }, .{ K.ts, K.ts }, .{ K.dur, K.dur }, .{ K.date, K.date }, .{ K.date, K.dur }, .{ K.time, K.dur }, .{ K.time, K.time } } },
    .{ Op.multiply, .{ .{ K.dur, K.int }, .{ K.int, K.dur } } },
    .{ Op.divide, .{ .{ K.dur, K.int }, .{ K.dur, K.dur } } },
    .{ Op.negate, .{.{K.dur}} },
    .{ Op.abs, .{.{K.dur}} },
};

/// Register the temporal kernels of add, subtract, multiply, divide, negate and abs (+ `_checked`;
/// after `arithmetic.register`, which creates the functions).
pub fn register(registry: *FunctionRegistry) KernelError!void {
    inline for (patterns) |entry| {
        const op = entry[0];
        inline for (entry[1]) |kinds| {
            inline for (.{ false, true }) |checked| {
                const name = @tagName(op) ++ if (checked) "_checked" else "";
                try registry.registerScalarKernel(name, .{
                    .signature = .{
                        .arity = kinds.len,
                        .type_check = Kernel(op, kinds.len, kinds, checked).check,
                        .options_check = dense.noOptions,
                        .result_type_fn = Kernel(op, kinds.len, kinds, checked).resultType,
                    },
                    .exec = Kernel(op, kinds.len, kinds, checked).exec,
                });
            }
        }
    }
}

fn kindOf(dt: DataType) ?K {
    return switch (dt) {
        .timestamp => .ts,
        .duration => .dur,
        .date32, .date64 => .date,
        .time32, .time64 => .time,
        .int8, .int16, .int32, .int64, .uint8, .uint16, .uint32, .uint64 => .int,
        else => null,
    };
}

/// Unit of a temporal operand (date32 counts as seconds, date64 as milliseconds).
fn unitOf(dt: DataType) TimeUnit {
    return switch (dt) {
        .timestamp => |t| t.unit,
        .duration => |d| d.unit,
        .time32, .time64 => |t| t.unit,
        .date32 => .second,
        .date64 => .millisecond,
        else => .second,
    };
}

fn unitsPerSecond(u: TimeUnit) i64 {
    return switch (u) {
        .second => 1,
        .millisecond => 1_000,
        .microsecond => 1_000_000,
        .nanosecond => 1_000_000_000,
    };
}

fn finer(a: TimeUnit, b: TimeUnit) TimeUnit {
    return if (@intFromEnum(b) > @intFromEnum(a)) b else a;
}

/// What one call computes.
const Plan = struct {
    /// Factor of each operand to the common unit (date32 also counts days).
    factors: [2]i64 = .{ 1, 1 },
    out_type: DataType,
    /// Units per day of a time result (null: no range check).
    day: ?i64 = null,
    mode: enum { wrapping, saturating, checked },
};

fn Kernel(comptime op: Op, comptime arity: usize, comptime kinds: [arity]K, comptime checked: bool) type {
    return struct {
        fn check(args: []const Datum) bool {
            if (args.len != arity) return false;
            for (args, kinds) |a, k| if (kindOf(a.dataType()) != k) return false;
            return true;
        }

        fn resultType(args: []const Datum, _: Options) KernelError!DataType {
            const p = try plan(args, .wrapping);
            return p.out_type;
        }

        fn plan(args: []const Datum, mode: @FieldType(Plan, "mode")) KernelError!Plan {
            if (args.len != arity) return error.InvalidArity;
            const t0 = args[0].dataType();
            if (arity == 1) return .{ .out_type = t0, .mode = mode };
            const t1 = args[1].dataType();
            if (op == .multiply or (op == .divide and kinds[1] == .int)) {
                return .{ .out_type = if (kinds[0] == .dur) t0 else t1, .mode = mode };
            }
            const unit = finer(unitOf(t0), unitOf(t1));
            var p: Plan = .{ .out_type = undefined, .mode = mode };
            for ([2]DataType{ t0, t1 }, &p.factors) |t, *f| {
                f.* = @divExact(unitsPerSecond(unit), unitsPerSecond(unitOf(t)));
                if (t == .date32) f.* *= 86_400;
            }
            if (op == .divide) {
                p.out_type = .{ .double = {} };
            } else if (kinds[0] == .ts and kinds[1] == .ts) {
                if ((t0.timestamp.timezone == null) != (t1.timestamp.timezone == null)) return error.InvalidInput;
                p.out_type = .{ .duration = .{ .unit = unit } };
            } else if (kinds[0] == .ts or kinds[1] == .ts) {
                const tz = if (kinds[0] == .ts) t0.timestamp.timezone else t1.timestamp.timezone;
                p.out_type = .{ .timestamp = .{ .unit = unit, .timezone = tz } };
            } else if (kinds[0] == .date and kinds[1] == .date) {
                p.out_type = .{ .duration = .{ .unit = unit } };
            } else if (kinds[0] == .date or kinds[1] == .date) {
                p.out_type = .{ .timestamp = .{ .unit = unit, .timezone = null } };
            } else if (kinds[0] == .time and kinds[1] == .dur) {
                p.out_type = if (unit == .second or unit == .millisecond) .{ .time32 = .{ .unit = unit } } else .{ .time64 = .{ .unit = unit } };
                p.day = 86_400 * unitsPerSecond(unit);
            } else {
                p.out_type = .{ .duration = .{ .unit = unit } };
            }
            return p;
        }

        fn exec(ctx: *ExecContext, args: []const Datum, _: Options) KernelError!Datum {
            const mode: @FieldType(Plan, "mode") = if (checked) .checked else if (ctx.overflowMode() == .saturating) .saturating else .wrapping;
            const p = try plan(args, mode);
            // Integer operands of multiply / divide as int64 (Arrow's implicit cast).
            var cast: [2]?Datum = .{ null, null };
            defer for (&cast) |*c| if (c.*) |*d| d.release();
            var call: [2]Datum = undefined;
            for (args, 0..) |a, i| {
                call[i] = a;
                if (kinds[i] == .int and a.dataType() != .int64) {
                    cast[i] = try ctx.invoke("cast", .scalar, &.{a}, .{ .cast = .{ .to_type = .{ .int64 = {} } } });
                    call[i] = cast[i].?;
                }
            }
            return dense.execElementwiseWith(ctx, call[0..arity], .{ .custom = .{ .tag = "temporal_arith", .payload = &p } }, p.out_type, true, chunk);
        }

        fn chunk(ctx: *ExecContext, values: []const ExecChunkValue, len: usize, options: Options) KernelError!ArrayRef {
            const p: *const Plan = @ptrCast(@alignCast(options.custom.payload.?));
            const rhs = if (arity == 2) dense.arrayData(values[1]) else null;
            const a = try Ints.init(values[0]);
            const b = if (arity == 2) try Ints.init(values[1]) else a;
            switch (p.out_type) {
                .double => {
                    var out = try dense.DenseBinaryOutput(f64).initValidity(ctx.allocator, len, dense.arrayData(values[0]), rhs, p.out_type);
                    errdefer out.deinit();
                    for (out.values(), 0..) |*d, i| {
                        const x = try scaled(a.get(i), p.factors[0], &out, i);
                        const y = try scaled(b.get(i), p.factors[1], &out, i);
                        d.* = @as(f64, @floatFromInt(x)) / @as(f64, @floatFromInt(y));
                    }
                    return out.finish() catch error.OutOfMemory;
                },
                .time32 => {
                    var out = try dense.DenseBinaryOutput(i32).initValidity(ctx.allocator, len, dense.arrayData(values[0]), rhs, p.out_type);
                    errdefer out.deinit();
                    for (out.values(), 0..) |*d, i| d.* = @truncate(try row(p, a, b, i, &out));
                    return out.finish() catch error.OutOfMemory;
                },
                else => {
                    var out = try dense.DenseBinaryOutput(i64).initValidity(ctx.allocator, len, dense.arrayData(values[0]), rhs, p.out_type);
                    errdefer out.deinit();
                    for (out.values(), 0..) |*d, i| d.* = try row(p, a, b, i, &out);
                    return out.finish() catch error.OutOfMemory;
                },
            }
        }

        /// `v` in the common unit; a valid row that does not fit is `InvalidCast`.
        inline fn scaled(v: i64, factor: i64, out: anytype, i: usize) KernelError!i64 {
            if (factor == 1) return v;
            const r = @mulWithOverflow(v, factor);
            if (r[1] != 0 and out.isValid(i)) return error.InvalidCast;
            return r[0];
        }

        fn row(p: *const Plan, a: Ints, b: Ints, i: usize, out: anytype) KernelError!i64 {
            const valid = out.isValid(i);
            const x = try scaled(a.get(i), p.factors[0], out, i);
            if (arity == 1) {
                const min = std.math.minInt(i64);
                if (x != min) return if (op == .negate) -x else @intCast(@abs(x));
                if (p.mode == .checked and valid) return error.Overflow;
                return if (p.mode == .saturating) std.math.maxInt(i64) else min;
            }
            const y = try scaled(b.get(i), p.factors[1], out, i);
            const r: i64 = switch (op) {
                .add => try combine(p.mode, @addWithOverflow(x, y), x +| y, valid),
                .subtract => try combine(p.mode, @subWithOverflow(x, y), x -| y, valid),
                .multiply => try combine(p.mode, @mulWithOverflow(x, y), x *| y, valid),
                .divide => blk: {
                    if (y == 0) {
                        if (valid) return error.DivideByZero;
                        break :blk 0;
                    }
                    if (x == std.math.minInt(i64) and y == -1) {
                        if (p.mode == .checked and valid) return error.Overflow;
                        break :blk if (p.mode == .saturating) std.math.maxInt(i64) else 0;
                    }
                    break :blk @divTrunc(x, y);
                },
                else => unreachable,
            };
            if (p.day) |day| if (valid and (r < 0 or r >= day)) return error.InvalidInput;
            return r;
        }

        inline fn combine(mode: @FieldType(Plan, "mode"), wrapped: struct { i64, u1 }, saturated: i64, valid: bool) KernelError!i64 {
            if (mode == .checked and wrapped[1] != 0 and valid) return error.Overflow;
            return if (mode == .saturating) saturated else wrapped[0];
        }
    };
}

/// An int32- or int64-stored operand (array chunk past validateLayout or non-null scalar),
/// read as i64.
const Ints = union(enum) {
    i32s: []const i32,
    i64s: []const i64,
    scalar: i64,

    fn init(v: ExecChunkValue) KernelError!Ints {
        switch (v) {
            .scalar => |s| return .{ .scalar = switch (s.value) {
                .date32, .time32, .i32 => |x| x,
                .date64, .time64, .timestamp, .duration, .i64 => |x| x,
                else => return error.InvalidInput,
            } },
            .array => |arr| {
                const data = arr.data();
                return switch (data.data_type) {
                    .date32, .time32, .int32 => .{ .i32s = (PrimitiveArray(i32){ .data = data }).values() catch return error.InvalidInput },
                    else => .{ .i64s = (PrimitiveArray(i64){ .data = data }).values() catch return error.InvalidInput },
                };
            },
        }
    }

    inline fn get(self: Ints, i: usize) i64 {
        return switch (self) {
            .i32s => |s| s[i],
            .i64s => |s| s[i],
            .scalar => |x| x,
        };
    }
};
