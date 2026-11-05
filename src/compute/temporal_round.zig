//! Temporal rounding and differences (Arrow C++ "Temporal rounding" and "Temporal difference"):
//! floor_temporal, ceil_temporal and round_temporal (`RoundTemporalOptions`) keep the input type;
//! years_between, quarters_between, weeks_between (`DayOfWeekOptions.week_start`), days_between,
//! hours_between, minutes_between, seconds_between, milliseconds_between, microseconds_between and
//! nanoseconds_between return int64, month_interval_between returns month_interval,
//! day_time_interval_between returns day_time_interval (whole days apart, then the difference of
//! the milliseconds of the day) and month_day_nano_interval_between month_day_nano_interval (months
//! apart, then the difference of the days of the month and of the nanoseconds of the day), as in
//! Arrow C++. Inputs are date32, date64 and timestamp (both operands of a difference of the same
//! type), and time32 / time64 for the rounding functions, hours_between .. nanoseconds_between and
//! the two interval differences (a time is a timestamp of its unit on 1970-01-01, as in Arrow);
//! timestamps without a timezone or in UTC only (another timezone is `NotImplemented`). Null
//! handling is intersection.
//!
//! The calendar math is temporal.zig's (proleptic Gregorian, days floored). Rounding follows
//! Arrow C++ step by step: multiples count from 1970-01-01 (weeks from the Monday or Sunday
//! before it, months from 1970-01, years from year 0), or with `calendar_based_origin` from the
//! start of the next larger unit (days from the start of the month, weeks from the start of week
//! 1 of the week year, months and quarters from January; a multiple of 1 ignores the origin); a
//! unit finer than the input is floored in that unit and truncated back, like `duration_cast`;
//! ceil is floor plus one multiple (truncated to the input unit) unless floor already equals the
//! value and `ceil_is_strictly_greater` is false; round picks ceil on a tie. Arrow quirks kept
//! (checked against pyarrow 26 on random values): ceil of a month, quarter or year on a boundary
//! still moves one multiple up (`ceil_is_strictly_greater` has no effect there); a week with
//! `calendar_based_origin` is counted from the start of week 1 of the year of the value shifted
//! by its weekday offset (3 or 4 days), with truncating division and no shift back, so it can
//! land after the value; years round by truncating division of the year number. Values whose result does not fit in 64 bits wrap (Arrow does the same).
const std = @import("std");
const core = @import("core.zig");
const dense = @import("dense.zig");
const temporal = @import("temporal.zig");

const ArrayData = @import("../array/array_data.zig").ArrayData;
const PrimitiveArray = @import("../array/primitive_array.zig").PrimitiveArray;
const ArrayRef = core.ArrayRef;
const DataType = core.DataType;
const Datum = core.Datum;
const ExecChunkValue = core.ExecChunkValue;
const ExecContext = core.ExecContext;
const FunctionRegistry = core.FunctionRegistry;
const KernelError = core.KernelError;
const Options = core.Options;
const RoundTemporalOptions = core.RoundTemporalOptions;
const TypeTag = std.meta.Tag(DataType);

const temporal_tags = [_]TypeTag{ .date32, .date64, .timestamp, .time32, .time64 };
const ns_per_day: i64 = 86_400_000_000_000;

/// The rounding functions, by Arrow name.
pub const RoundOp = enum { floor_temporal, ceil_temporal, round_temporal };

/// The difference functions, by Arrow name.
pub const Between = enum { years_between, quarters_between, month_interval_between, weeks_between, days_between, hours_between, minutes_between, seconds_between, milliseconds_between, microseconds_between, nanoseconds_between, day_time_interval_between, month_day_nano_interval_between };

/// Whether Arrow registers `b` for time32 / time64 (the differences of time-of-day units).
fn forTimes(comptime b: Between) bool {
    return switch (b) {
        .years_between, .quarters_between, .month_interval_between, .weeks_between, .days_between => false,
        else => true,
    };
}

const day_time_type = DataType{ .interval_day_time = .{ .unit = .day_time } };
const month_day_nano_type = DataType{ .interval_month_day_nano = .{ .unit = .month_day_nano } };

fn betweenType(comptime b: Between) DataType {
    return switch (b) {
        .month_interval_between => month_interval_type,
        .day_time_interval_between => day_time_type,
        .month_day_nano_interval_between => month_day_nano_type,
        else => .{ .int64 = {} },
    };
}

/// Register the rounding and difference functions for date32, date64, timestamp, time32 and time64.
pub fn register(registry: *FunctionRegistry) KernelError!void {
    @setEvalBranchQuota(10_000);
    inline for (temporal_tags) |tag| {
        inline for (comptime std.meta.tags(RoundOp)) |op| {
            try registry.registerScalarKernel(@tagName(op), .{
                .signature = .{ .arity = 1, .type_check = dense.argsOfType(tag, 1), .options_check = roundOptions, .result_type_fn = firstType },
                .exec = Round(op).exec,
            });
        }
        inline for (comptime std.meta.tags(Between)) |b| {
            if (forTimes(b) or !(tag == .time32 or tag == .time64)) try registry.registerScalarKernel(@tagName(b), .{
                .signature = .{
                    .arity = 2,
                    .type_check = sameTemporal(tag),
                    .options_check = if (b == .weeks_between) weekStartOptions else dense.noOptions,
                    .result_type_fn = dense.resultTypeOf(betweenType(b)),
                },
                .exec = Diff(b).exec,
            });
        }
    }
}

const month_interval_type = DataType{ .interval_months = .{ .unit = .months } };

fn firstType(args: []const Datum, _: Options) KernelError!DataType {
    if (args.len == 0) return error.InvalidArity;
    return args[0].dataType();
}

/// Two arguments of type `tag` that are equal (same unit and timezone).
fn sameTemporal(comptime tag: TypeTag) core.TypeCheckFn {
    return &struct {
        fn check(args: []const Datum) bool {
            return args.len == 2 and std.meta.activeTag(args[0].dataType()) == tag and args[0].dataType().eql(args[1].dataType());
        }
    }.check;
}

/// `RoundTemporalOptions` with `multiple > 0`, or `none` (one day).
fn roundOptions(options: Options) bool {
    return switch (options) {
        .none => true,
        .round_temporal => |o| o.multiple > 0,
        else => false,
    };
}

/// `none` (Monday) or `DayOfWeekOptions` with `week_start` in 1..7.
fn weekStartOptions(options: Options) bool {
    return switch (options) {
        .none => true,
        .day_of_week => |o| o.week_start >= 1 and o.week_start <= 7,
        else => false,
    };
}

/// `NotImplemented` for a timestamp in a timezone other than UTC.
fn checkTimezone(dt: DataType) KernelError!void {
    switch (dt) {
        .timestamp => |ts| if (ts.timezone) |tz| if (!temporal.isUtc(tz)) return error.NotImplemented,
        else => {},
    }
}

/// Nanoseconds per value of a date32 / date64 / timestamp / time32 / time64 type.
fn nsPerTick(dt: DataType) i64 {
    const unit = switch (dt) {
        .date32 => return ns_per_day,
        .date64 => return 1_000_000,
        .timestamp => |ts| ts.unit,
        .time32, .time64 => |t| t.unit,
        else => unreachable,
    };
    return switch (unit) {
        .second => 1_000_000_000,
        .millisecond => 1_000_000,
        .microsecond => 1_000,
        .nanosecond => 1,
    };
}

/// Values of a temporal chunk operand as `V` (i32 for date32 and time32, i64 otherwise).
fn operandOf(comptime V: type, v: ExecChunkValue) KernelError!dense.Operand(V) {
    return switch (v) {
        .array => |arr| .{ .slice = (PrimitiveArray(V){ .data = arr.data() }).values() catch return error.InvalidInput },
        .scalar => |s| .{ .scalar = switch (s.value) {
            .date32, .time32 => |x| if (V == i32) x else return error.InvalidInput,
            .date64, .timestamp, .time64 => |x| if (V == i64) x else return error.InvalidInput,
            else => return error.InvalidInput,
        } },
    };
}

// ------------------------------------------------------------------------------ rounding

const unit_ns = [_]i64{ 1, 1_000, 1_000_000, 1_000_000_000, 60_000_000_000, 3_600_000_000_000, ns_per_day };

/// Per-call constants of a rounding: values are taken `k` working ticks per input tick.
const Plan = struct {
    o: RoundTemporalOptions,
    /// Working ticks per input tick (> 1 when the unit is finer than the input).
    k: i64,
    /// Working ticks per day.
    day: i64,
    /// One multiple of a fixed-length unit (nanosecond .. week) in working ticks.
    u: i64,
    /// Next larger unit in working ticks (calendar origin of nanosecond .. hour).
    larger: i64,
    /// Week origin shift in working ticks (Monday- or Sunday-based weeks).
    shift: i64,

    fn init(options: RoundTemporalOptions, tick_ns: i64) KernelError!Plan {
        // One multiple rounds from the epoch whatever the origin option (Arrow C++).
        var o = options;
        if (o.multiple == 1) o.calendar_based_origin = false;
        const idx = @intFromEnum(o.unit);
        const fixed_ns: i64 = if (o.unit == .week) 7 * ns_per_day else if (idx < unit_ns.len) unit_ns[idx] else ns_per_day;
        const base_ns = if (o.unit == .week) ns_per_day else fixed_ns; // smallest step of the unit
        const w_ns = @min(tick_ns, base_ns);
        const per = @divExact(fixed_ns, w_ns);
        const u = std.math.mul(i64, per, o.multiple) catch return error.InvalidOptions;
        const larger_ns: i64 = if (idx + 1 < unit_ns.len) unit_ns[idx + 1] else ns_per_day;
        return .{
            .o = o,
            .k = @divExact(tick_ns, w_ns),
            .day = @divExact(ns_per_day, w_ns),
            .u = u,
            .larger = @divExact(larger_ns, w_ns),
            .shift = (if (o.week_starts_monday) @as(i64, 3) else 4) * @divExact(ns_per_day, w_ns),
        };
    }

    /// Whether the unit has a fixed length in working ticks (not month, quarter or year).
    fn fixed(p: Plan) bool {
        return switch (p.o.unit) {
            .month, .quarter, .year => false,
            else => true,
        };
    }

    /// Floor of `t` (input ticks) in working ticks: a fixed-length unit only.
    fn floorFixed(p: Plan, t: i64) i64 {
        const tw = t *% p.k;
        if (!p.o.calendar_based_origin) {
            if (p.o.unit == .week) return @divFloor(tw +% p.shift, p.u) *% p.u -% p.shift;
            return @divFloor(tw, p.u) *% p.u;
        }
        const origin = switch (p.o.unit) {
            .day => blk: {
                const c = temporal.civil(@divFloor(tw, p.day));
                break :blk temporal.daysFromCivil(c.year, c.month, 1) *% p.day;
            },
            .week => {
                // Arrow C++ shifts the value by the weekday offset, counts weeks from the start of
                // week 1 of the shifted value's year and does not shift back.
                const shifted = tw +% p.shift;
                const y = temporal.civil(@divFloor(shifted, p.day)).year;
                const start = temporal.weekOneStart(y, .{ .week_starts_monday = p.o.week_starts_monday }) *% p.day;
                return start +% @divTrunc(shifted -% start, p.u) *% p.u;
            },
            else => @divFloor(tw, p.larger) *% p.larger,
        };
        return origin +% @divTrunc(tw -% origin, p.u) *% p.u;
    }

    /// Months of a calendar unit multiple (`quarter` counts three months).
    fn months(p: Plan) i64 {
        return if (p.o.unit == .quarter) p.o.multiple *| 3 else p.o.multiple;
    }

    /// Floor (`up` false) or one multiple past it (`up` true) of a calendar unit, in input ticks.
    fn calendar(p: Plan, t: i64, up: bool) i64 {
        const per_day = @divExact(p.day, p.k); // input ticks per day (calendar units: k == 1)
        const c = temporal.civil(@divFloor(t, per_day));
        var y: i64 = undefined;
        var m0: i64 = undefined; // month - 1
        if (p.o.unit == .year) {
            y = @divTrunc(c.year, p.o.multiple) *% p.o.multiple +% if (up) p.o.multiple else 0;
            m0 = 0;
        } else {
            const n = p.months();
            var total: i64 = if (p.o.calendar_based_origin) c.month - 1 else (c.year -% 1970) *% 12 +% (c.month - 1);
            total = @divFloor(total, n) *% n +% if (up) n else 0;
            const base: i64 = if (p.o.calendar_based_origin) c.year else 1970;
            y = base +% @divFloor(total, 12);
            m0 = @mod(total, 12);
        }
        return temporal.daysFromCivil(y, m0 + 1, 1) *% per_day;
    }

    /// floor_temporal of `t` in input ticks.
    fn floor(p: Plan, t: i64) i64 {
        if (!p.fixed()) return p.calendar(t, false);
        return @divTrunc(p.floorFixed(t), p.k);
    }

    /// ceil_temporal of `t`, given its floor `f`: one multiple, truncated to the input unit, past
    /// the floor (Arrow C++). Months, quarters and years always move one multiple past the
    /// floor, even from a boundary.
    fn ceil(p: Plan, t: i64, f: i64) i64 {
        if (!p.fixed()) return p.calendar(t, true);
        if (!(p.o.ceil_is_strictly_greater or f < t)) return f;
        return f +% @divTrunc(p.u, p.k);
    }

    fn apply(p: Plan, comptime op: RoundOp, t: i64) i64 {
        const f = p.floor(t);
        if (op == .floor_temporal) return f;
        const c = p.ceil(t, f);
        if (op == .ceil_temporal) return c;
        return if (t -% f >= c -% t) c else f;
    }
};

fn Round(comptime op: RoundOp) type {
    return struct {
        /// Round each row (array, chunked or scalar) to a multiple of the unit.
        fn exec(ctx: *ExecContext, args: []const Datum, options: Options) KernelError!Datum {
            if (args.len != 1) return error.InvalidArity;
            try checkTimezone(args[0].dataType());
            const o: RoundTemporalOptions = if (options == .round_temporal) options.round_temporal else .{};
            _ = try Plan.init(o, nsPerTick(args[0].dataType()));
            return dense.execElementwiseWith(ctx, args, options, args[0].dataType(), true, chunk);
        }

        fn chunk(ctx: *ExecContext, values: []const ExecChunkValue, len: usize, options: Options) KernelError!ArrayRef {
            var arr = try dense.chunkArray(ctx.allocator, values[0]);
            defer arr.release();
            const data = arr.data();
            const o: RoundTemporalOptions = if (options == .round_temporal) options.round_temporal else .{};
            const p = try Plan.init(o, nsPerTick(data.data_type));
            return switch (data.data_type) {
                .date32, .time32 => run(i32, ctx, data, len, p),
                else => run(i64, ctx, data, len, p),
            };
        }

        fn run(comptime V: type, ctx: *ExecContext, data: *const ArrayData, len: usize, p: Plan) KernelError!ArrayRef {
            const vals = (PrimitiveArray(V){ .data = data }).values() catch return error.InvalidInput;
            var out = try dense.DenseBinaryOutput(V).initValidity(ctx.allocator, len, data, null, data.data_type);
            errdefer out.deinit();
            const dst = out.values();
            if (V == i64 and p.fixed() and p.k == 1 and !p.o.calendar_based_origin) {
                // Epoch-based fixed units: one division per row, through doubles when every value
                // (null slots included) is below 2^52 so the loop vectorizes.
                const shift = if (p.o.unit == .week) p.shift else 0;
                var lo: i64 = 0;
                var hi: i64 = 0;
                for (vals) |v| {
                    lo = @min(lo, v);
                    hi = @max(hi, v);
                }
                if (lo > -(1 << 52) + shift and hi < (1 << 52) - shift) {
                    fastFixed(vals, dst, p.u, shift, p.o.ceil_is_strictly_greater);
                    return out.finish() catch error.OutOfMemory;
                }
            }
            for (vals, dst) |v, *d| d.* = @truncate(p.apply(op, v));
            return out.finish() catch error.OutOfMemory;
        }

        fn fastFixed(vals: []const i64, dst: []i64, u: i64, shift: i64, strict: bool) void {
            const L = 8;
            const V = @Vector(L, i64);
            const inv: @Vector(L, f64) = @splat(1.0 / @as(f64, @floatFromInt(u)));
            const uv: V = @splat(u);
            const sv: V = @splat(shift);
            const one: V = @splat(1);
            const zero: V = @splat(0);
            var i: usize = 0;
            while (i + L <= vals.len) : (i += L) {
                const t: V = vals[i..][0..L].*;
                const x = t +% sv;
                var q: V = @intFromFloat(@floor(@as(@Vector(L, f64), @floatFromInt(x)) * inv));
                const r = x -% q *% uv;
                q +%= @select(i64, r >= uv, one, zero) -% @select(i64, r < zero, one, zero);
                const f = q *% uv -% sv;
                if (op == .floor_temporal) {
                    dst[i..][0..L].* = f;
                    continue;
                }
                const bump = if (strict) @as(@Vector(L, bool), @splat(true)) else f < t;
                const c = @select(i64, bump, f +% uv, f);
                dst[i..][0..L].* = if (op == .ceil_temporal) c else @select(i64, t -% f >= c -% t, c, f);
            }
            for (vals[i..], dst[i..]) |t, *d| {
                const f = @divFloor(t +% shift, u) *% u -% shift;
                const c = if (strict or f < t) f +% u else f;
                d.* = switch (op) {
                    .floor_temporal => f,
                    .ceil_temporal => c,
                    .round_temporal => if (t -% f >= c -% t) c else f,
                };
            }
        }
    };
}

// ------------------------------------------------------------------------------ differences

fn Diff(comptime b: Between) type {
    return struct {
        /// Return the number of unit boundaries from the first to the second argument of each row.
        fn exec(ctx: *ExecContext, args: []const Datum, options: Options) KernelError!Datum {
            if (args.len != 2) return error.InvalidArity;
            try checkTimezone(args[0].dataType());
            return dense.execElementwiseWith(ctx, args, options, betweenType(b), true, chunk);
        }

        fn chunk(ctx: *ExecContext, values: []const ExecChunkValue, len: usize, options: Options) KernelError!ArrayRef {
            const week_start: i64 = if (options == .day_of_week) options.day_of_week.week_start else 1;
            return switch (values[0].dataType()) {
                .date32, .time32 => run(i32, ctx, values, len, week_start),
                else => run(i64, ctx, values, len, week_start),
            };
        }

        fn run(comptime V: type, ctx: *ExecContext, values: []const ExecChunkValue, len: usize, week_start: i64) KernelError!ArrayRef {
            const tick_ns = nsPerTick(values[0].dataType());
            const lhs = try operandOf(V, values[0]);
            const rhs = try operandOf(V, values[1]);
            const O = switch (b) {
                .month_interval_between => i32,
                .month_day_nano_interval_between => i128,
                else => i64,
            };
            var out = try dense.DenseBinaryOutput(O).initValidity(ctx.allocator, len, dense.arrayData(values[0]), dense.arrayData(values[1]), betweenType(b));
            errdefer out.deinit();
            const dst = out.values();
            switch (lhs) {
                inline else => |l| switch (rhs) {
                    inline else => |r| for (dst, 0..) |*d, i| {
                        d.* = switch (b) {
                            .day_time_interval_between, .month_day_nano_interval_between => interval(dense.at(l, i), dense.at(r, i), tick_ns),
                            else => @truncate(diff(dense.at(l, i), dense.at(r, i), tick_ns, week_start)),
                        };
                    },
                },
            }
            return out.finish() catch error.OutOfMemory;
        }

        /// day_time_interval_between: `{days, milliseconds}` as one i64 (days in the low 32 bits,
        /// the Arrow memory layout); month_day_nano_interval_between: `{months, days, nanoseconds}`
        /// as one i128 (months in the low 32 bits, days in the next 32, nanoseconds in the high 64).
        /// Fields that do not fit wrap, like Arrow's 32-bit casts.
        fn interval(a: i64, z: i64, tick_ns: i64) if (b == .day_time_interval_between) i64 else i128 {
            const per_day = @divExact(ns_per_day, tick_ns);
            const da = @divFloor(a, per_day);
            const dz = @divFloor(z, per_day);
            // Ticks into the day, in [0, per_day): at most one day of nanoseconds.
            const ns_a = (a -% da *% per_day) *% tick_ns;
            const ns_z = (z -% dz *% per_day) *% tick_ns;
            if (b == .day_time_interval_between) {
                const days: u32 = @truncate(@as(u64, @bitCast(dz -% da)));
                const ms: u32 = @bitCast(@as(i32, @intCast(@divTrunc(ns_z, 1_000_000) - @divTrunc(ns_a, 1_000_000))));
                return @bitCast(@as(u64, ms) << 32 | days);
            }
            const ca = temporal.civil(da);
            const cz = temporal.civil(dz);
            const months: u32 = @truncate(@as(u64, @bitCast((cz.year *% 12 +% cz.month) -% (ca.year *% 12 +% ca.month))));
            const days: u32 = @bitCast(@as(i32, @intCast(cz.day - ca.day)));
            const nanos: u64 = @bitCast(ns_z - ns_a);
            return @bitCast(@as(u128, nanos) << 64 | @as(u128, days) << 32 | months);
        }

        fn diff(a: i64, z: i64, tick_ns: i64, week_start: i64) i64 {
            const per_day = @divExact(ns_per_day, tick_ns);
            switch (b) {
                .years_between, .quarters_between, .month_interval_between => {
                    const ca = temporal.civil(@divFloor(a, per_day));
                    const cz = temporal.civil(@divFloor(z, per_day));
                    return switch (b) {
                        .years_between => cz.year -% ca.year,
                        .quarters_between => (cz.year *% 4 +% @divTrunc(cz.month - 1, 3)) -% (ca.year *% 4 +% @divTrunc(ca.month - 1, 3)),
                        else => (cz.year *% 12 +% cz.month) -% (ca.year *% 12 +% ca.month),
                    };
                },
                .weeks_between => {
                    // Week index with boundaries on `week_start` (1 = Monday ... 7 = Sunday).
                    const shift = 3 - (week_start - 1);
                    return @divFloor(@divFloor(z, per_day) +% shift, 7) -% @divFloor(@divFloor(a, per_day) +% shift, 7);
                },
                .days_between => return @divFloor(z, per_day) -% @divFloor(a, per_day),
                else => {
                    const u_ns: i64 = switch (b) {
                        .hours_between => 3_600_000_000_000,
                        .minutes_between => 60_000_000_000,
                        .seconds_between => 1_000_000_000,
                        .milliseconds_between => 1_000_000,
                        .microseconds_between => 1_000,
                        else => 1,
                    };
                    // A unit finer than the input: exact multiple of the tick difference.
                    if (u_ns <= tick_ns) return (z -% a) *% @divExact(tick_ns, u_ns);
                    const u = @divExact(u_ns, tick_ns);
                    return @divFloor(z, u) -% @divFloor(a, u);
                },
            }
        }
    };
}
