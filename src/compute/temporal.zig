//! Temporal component extraction (Arrow C++ "Temporal component extraction"): year, month, day,
//! day_of_week, day_of_year, iso_year, iso_week, iso_calendar, quarter and is_leap_year on date32,
//! date64 and timestamp; hour, minute, second, millisecond, microsecond and nanosecond on date64
//! and timestamp. Results are int64 (is_leap_year: bool; iso_calendar: struct of int64
//! `iso_year`, `iso_week`, `iso_day_of_week`); null handling is intersection.
//!
//! Dates follow the proleptic Gregorian calendar over the whole input range (days are floored, so
//! instants before 1970 land on the right day). Timestamps without a timezone and with a UTC
//! timezone (`UTC`, `Etc/UTC`, `+00:00`) are read as UTC; any other timezone is `NotImplemented`
//! (Arrow C++ converts to local time through the timezone database). millisecond, microsecond and
//! nanosecond are the 0..999 sub-fields (of the second, millisecond and microsecond), as in Arrow.
const std = @import("std");
const bitmap = @import("../bitmap.zig");
const core = @import("core.zig");
const datatype = @import("../datatype.zig");
const dense = @import("dense.zig");

const ArrayData = @import("../array/array_data.zig").ArrayData;
const PrimitiveArray = @import("../array/primitive_array.zig").PrimitiveArray;
const SharedBuffer = @import("../buffer.zig").SharedBuffer;
const ArrayRef = core.ArrayRef;
const DataType = core.DataType;
const Datum = core.Datum;
const DayOfWeekOptions = core.DayOfWeekOptions;
const ExecChunkValue = core.ExecChunkValue;
const ExecContext = core.ExecContext;
const FunctionRegistry = core.FunctionRegistry;
const KernelError = core.KernelError;
const Options = core.Options;
const TypeTag = std.meta.Tag(DataType);

/// The extraction functions, by Arrow C++ name.
pub const Field = enum { year, month, day, day_of_week, day_of_year, hour, minute, second, millisecond, microsecond, nanosecond, iso_year, iso_week, iso_calendar, quarter, is_leap_year };

const int64_type = DataType{ .int64 = {} };
const bool_type = DataType{ .bool = {} };
const iso_fields = [_]datatype.Field{
    .{ .name = "iso_year", .data_type = &int64_type },
    .{ .name = "iso_week", .data_type = &int64_type },
    .{ .name = "iso_day_of_week", .data_type = &int64_type },
};
/// Result type of iso_calendar.
pub const iso_calendar_type = DataType{ .struct_ = .{ .fields = &iso_fields } };

fn isTimePart(comptime f: Field) bool {
    return switch (f) {
        .hour, .minute, .second, .millisecond, .microsecond, .nanosecond => true,
        else => false,
    };
}

fn outType(comptime f: Field) DataType {
    return switch (f) {
        .is_leap_year => bool_type,
        .iso_calendar => iso_calendar_type,
        else => int64_type,
    };
}

/// Register every extraction function for date32 (date parts only), date64 and timestamp.
pub fn register(registry: *FunctionRegistry) KernelError!void {
    inline for (comptime std.meta.tags(Field)) |f| {
        inline for (.{ TypeTag.date32, TypeTag.date64, TypeTag.timestamp }) |tag| {
            if (!(isTimePart(f) and tag == .date32)) try registry.registerScalarKernel(@tagName(f), .{
                .signature = .{
                    .arity = 1,
                    .type_check = dense.argsOfType(tag, 1),
                    .options_check = if (f == .day_of_week) dayOfWeekOptions else dense.noOptions,
                    .result_type_fn = dense.resultTypeOf(outType(f)),
                },
                .exec = Extract(f).exec,
            });
        }
    }
}

/// Accept `none` (defaults) or `DayOfWeekOptions` with `week_start` in 1..7.
fn dayOfWeekOptions(options: Options) bool {
    return switch (options) {
        .none => true,
        .day_of_week => |o| o.week_start >= 1 and o.week_start <= 7,
        else => false,
    };
}

/// Return whether `tz` names UTC (a timezone-naive timestamp is read as UTC too).
fn isUtc(tz: []const u8) bool {
    for ([_][]const u8{ "UTC", "Etc/UTC", "+00:00" }) |name| if (std.mem.eql(u8, tz, name)) return true;
    return false;
}

fn Extract(comptime f: Field) type {
    return struct {
        /// Extract the component from each row (array, chunked or scalar).
        fn exec(ctx: *ExecContext, args: []const Datum, options: Options) KernelError!Datum {
            if (args.len != 1) return error.InvalidArity;
            switch (args[0].dataType()) {
                .timestamp => |ts| if (ts.timezone) |tz| if (!isUtc(tz)) return error.NotImplemented,
                else => {},
            }
            return dense.execElementwiseWith(ctx, args, options, outType(f), true, chunk);
        }

        fn chunk(ctx: *ExecContext, values: []const ExecChunkValue, len: usize, options: Options) KernelError!ArrayRef {
            var arr = try dense.chunkArray(ctx.allocator, values[0]);
            defer arr.release();
            const dow: DayOfWeekOptions = if (options == .day_of_week) options.day_of_week else .{};
            const data = arr.data();
            return switch (data.data_type) {
                .date32 => run(i32, 1, 0, ctx, data, len, dow),
                .date64 => run(i64, 86_400_000, 1_000_000, ctx, data, len, dow),
                .timestamp => |ts| switch (ts.unit) {
                    .second => run(i64, 86_400, 1_000_000_000, ctx, data, len, dow),
                    .millisecond => run(i64, 86_400_000, 1_000_000, ctx, data, len, dow),
                    .microsecond => run(i64, 86_400_000_000, 1_000, ctx, data, len, dow),
                    .nanosecond => run(i64, 86_400_000_000_000, 1, ctx, data, len, dow),
                },
                else => error.UnsupportedType,
            };
        }

        /// `per_day`: units per day; `ns_per_unit`: nanoseconds per unit (0 for date32).
        fn run(comptime V: type, comptime per_day: i64, comptime ns_per_unit: i64, ctx: *ExecContext, data: *const ArrayData, len: usize, dow: DayOfWeekOptions) KernelError!ArrayRef {
            const vals = (PrimitiveArray(V){ .data = data }).values() catch return error.InvalidInput;
            switch (f) {
                .is_leap_year => {
                    var out = try dense.DenseBinaryOutput(bool).initValidity(ctx.allocator, len, data, null, bool_type);
                    errdefer out.deinit();
                    const bits = out.values();
                    @memset(bits, 0);
                    for (vals, 0..) |v, i| {
                        if (isLeap(civil(@divFloor(@as(i64, v), per_day)).year)) bits[i / 8] |= @as(u8, 1) << @intCast(i % 8);
                    }
                    return out.finish() catch error.OutOfMemory;
                },
                .iso_calendar => {
                    var parts: [3]dense.DenseBinaryOutput(i64) = undefined;
                    var made: usize = 0;
                    defer for (parts[0..made]) |*p| p.deinit();
                    for (&parts) |*p| {
                        p.* = try dense.DenseBinaryOutput(i64).initValidity(ctx.allocator, len, null, null, int64_type);
                        made += 1;
                    }
                    for (vals, 0..) |v, i| {
                        const days = @divFloor(@as(i64, v), per_day);
                        const iw = isoWeek(days);
                        parts[0].values()[i] = iw.year;
                        parts[1].values()[i] = iw.week;
                        parts[2].values()[i] = @mod(days + 3, 7) + 1;
                    }
                    var children: [3]ArrayRef = undefined;
                    var built: usize = 0;
                    errdefer for (children[0..built]) |*c| c.release();
                    for (&parts, &children) |*p, *c| {
                        c.* = p.finish() catch return error.OutOfMemory;
                        built += 1;
                    }
                    var validity = try dense.combinedValidity(ctx.allocator, len, &.{data});
                    errdefer if (validity) |*b| b.deinit();
                    var buffers = [1]SharedBuffer{.empty};
                    var null_count: usize = 0;
                    if (validity) |*b| {
                        null_count = bitmap.countUnsetBit(b.data, len);
                        buffers[0] = b.toShared(bitmap.byteLength(len)) catch return error.OutOfMemory;
                        validity = null;
                    }
                    errdefer buffers[0].release();
                    return ArrayRef.fromOwnedInline(ctx.allocator, .{
                        .data_type = iso_calendar_type,
                        .length = len,
                        .null_count = null_count,
                        .buffers = &buffers,
                        .children = &children,
                    }) catch error.OutOfMemory;
                },
                else => {
                    var out = try dense.DenseBinaryOutput(i64).initValidity(ctx.allocator, len, data, null, int64_type);
                    errdefer out.deinit();
                    // When every value (null slots included) lands in the 32-bit calendar range, the
                    // loop has no per-row range branch.
                    var lo: i64 = 0;
                    var hi: i64 = 0;
                    for (vals) |v| {
                        lo = @min(lo, v);
                        hi = @max(hi, v);
                    }
                    const margin = 7; // isoWeek looks up to 3 days around the row
                    const small = lo > -(1 << 52) and hi < (1 << 52); // exact in f64, see floorDay
                    if (small and @divFloor(lo, per_day) >= fast_min + margin and @divFloor(hi, per_day) <= fast_max - margin) {
                        const dst = out.values();
                        var i: usize = 0;
                        if (comptime f == .year or f == .month or f == .day or f == .quarter) {
                            while (i + lanes <= vals.len) : (i += lanes) {
                                const x: @Vector(lanes, i64) = vals[i..][0..lanes].*;
                                const c = civilVec(floorDayVec(per_day, x));
                                dst[i..][0..lanes].* = switch (f) {
                                    .year => c.year,
                                    .month => c.month,
                                    .day => c.day,
                                    else => (c.month + @as(@Vector(lanes, i64), @splat(2))) / @as(@Vector(lanes, i64), @splat(3)),
                                };
                            }
                        }
                        for (vals[i..], dst[i..]) |v, *o| o.* = component(civilFast, per_day, ns_per_unit, v, dow);
                    } else {
                        for (vals, out.values()) |v, *o| o.* = component(civil, per_day, ns_per_unit, v, dow);
                    }
                    return out.finish() catch error.OutOfMemory;
                },
            }
        }

        inline fn component(comptime civ: fn (i64) Civil, comptime per_day: i64, comptime ns_per_unit: i64, v: anytype, dow: DayOfWeekOptions) i64 {
            const x: i64 = v;
            if (comptime isTimePart(f)) {
                const ns = @mod(x, per_day) * ns_per_unit;
                return switch (f) {
                    .hour => @divTrunc(ns, 3_600_000_000_000),
                    .minute => @mod(@divTrunc(ns, 60_000_000_000), 60),
                    .second => @mod(@divTrunc(ns, 1_000_000_000), 60),
                    .millisecond => @mod(@divTrunc(ns, 1_000_000), 1000),
                    .microsecond => @mod(@divTrunc(ns, 1_000), 1000),
                    .nanosecond => @mod(ns, 1000),
                    else => unreachable,
                };
            }
            const days = if (civ == civilFast) floorDay(per_day, x) else @divFloor(x, per_day);
            return switch (f) {
                .year => civ(days).year,
                .month => civ(days).month,
                .day => civ(days).day,
                .quarter => @divTrunc(civ(days).month - 1, 3) + 1,
                .day_of_year => dayOfYearWith(civ, days),
                .day_of_week => @mod(@mod(days + 3, 7) + 1 + 7 - @as(i64, dow.week_start), 7) + @intFromBool(!dow.count_from_zero),
                .iso_year => isoWeekWith(civ, days).year,
                .iso_week => isoWeekWith(civ, days).week,
                else => unreachable,
            };
        }
    };
}

/// `@divFloor(x, per_day)` for `|x| < 2^52` through a double multiply and a one-step correction
/// (vectorizes, where the 64-bit integer division does not).
inline fn floorDay(comptime per_day: i64, x: i64) i64 {
    if (per_day == 1) return x;
    var q: i64 = @intFromFloat(@floor(@as(f64, @floatFromInt(x)) * (1.0 / @as(f64, per_day))));
    const r = x - q * per_day;
    q += @as(i64, @intFromBool(r >= per_day)) - @intFromBool(r < 0);
    return q;
}

const lanes = 8;

/// `floorDay` on a vector of values.
inline fn floorDayVec(comptime per_day: i64, x: @Vector(lanes, i64)) @Vector(lanes, i64) {
    if (per_day == 1) return x;
    const V = @Vector(lanes, i64);
    const xf: @Vector(lanes, f64) = @floatFromInt(x);
    var q: V = @intFromFloat(@floor(xf * @as(@Vector(lanes, f64), @splat(1.0 / @as(f64, per_day)))));
    const r = x - q * @as(V, @splat(per_day));
    const one: V = @splat(1);
    const zero: V = @splat(0);
    q += @select(i64, r >= @as(V, @splat(per_day)), one, zero) - @select(i64, r < zero, one, zero);
    return q;
}

/// `civilFast` on a vector of days (every lane in `[fast_min, fast_max]`, or a wrong lane).
inline fn civilVec(days: @Vector(lanes, i64)) struct { year: @Vector(lanes, i64), month: @Vector(lanes, i64), day: @Vector(lanes, i64) } {
    const U = @Vector(lanes, u32);
    const W = @Vector(lanes, u64);
    const I = @Vector(lanes, i64);
    const N: U = @truncate(@as(W, @bitCast(days -% @as(I, @splat(fast_min)))));
    const N1 = N *% @as(U, @splat(4)) +% @as(U, @splat(3));
    const C = N1 / @as(U, @splat(146_097));
    const NC = (N1 - C * @as(U, @splat(146_097))) / @as(U, @splat(4));
    const P2 = @as(W, @splat(2_939_745)) * @as(W, NC * @as(U, @splat(4)) + @as(U, @splat(3)));
    const Z: U = @truncate(P2 >> @as(@Vector(lanes, u6), @splat(32)));
    const NY = @as(U, @truncate(P2)) / @as(U, @splat(2_939_745 * 4));
    const N3 = NY * @as(U, @splat(2141)) + @as(U, @splat(197_913));
    const J = NY >= @as(U, @splat(306));
    const M: I = N3 >> @as(@Vector(lanes, u5), @splat(16));
    const one: I = @splat(1);
    const zero: I = @splat(0);
    const year = @as(I, C * @as(U, @splat(100)) + Z) - @as(I, @splat(400 * 82)) + @select(i64, J, one, zero);
    return .{
        .year = year,
        .month = M - @select(i64, J, @as(I, @splat(12)), zero),
        .day = @as(I, (N3 & @as(U, @splat(0xFFFF))) / @as(U, @splat(2141))) + one,
    };
}

/// A proleptic Gregorian date.
pub const Civil = struct { year: i64, month: i64, day: i64 };

/// Days after 1970-01-01 that `civilFast` handles: about years -32800 to 2.9 million.
pub const fast_min: i64 = -(719_468 + 146_097 * 82);
pub const fast_max: i64 = 1_073_741_823 + fast_min;

/// Return the date `days` days after 1970-01-01 (`civilFast` in its range, `civilWide` beyond).
pub fn civil(days: i64) Civil {
    if (days < fast_min or days > fast_max) return civilWide(days);
    return civilFast(days);
}

/// The date of `days` in `[fast_min, fast_max]` in 32-bit arithmetic (Neri and Schneider,
/// "Euclidean affine functions and their application to calendar algorithms", 2022). Wrapping
/// operations: a value out of range gives a wrong date, never a trap (null slots may hold anything).
pub fn civilFast(days: i64) Civil {
    const s = 82; // 400-year cycles that shift every day in range to a non-negative count
    const N: u32 = @truncate(@as(u64, @bitCast(days -% fast_min)));
    const N1 = N *% 4 +% 3;
    const C = N1 / 146_097;
    const NC = N1 % 146_097 / 4;
    const P2 = @as(u64, 2_939_745) * (4 * NC + 3);
    const Z: u32 = @intCast(P2 >> 32);
    const NY = @as(u32, @truncate(P2)) / 2_939_745 / 4; // day of the year, March 1 = 0
    const N3 = 2141 * NY + 197_913;
    const J = NY >= 306;
    const M = N3 / 65536;
    return .{
        .year = @as(i64, 100 * @as(i64, C) + Z) - 400 * s + @intFromBool(J),
        .month = if (J) M - 12 else M,
        .day = N3 % 65536 / 2141 + 1,
    };
}

/// `civil` over the whole i64 range (Howard Hinnant's `civil_from_days` in 64-bit arithmetic).
pub fn civilWide(days: i64) Civil {
    const z = days + 719_468;
    const era = @divFloor(z, 146_097);
    const doe = z - era * 146_097; // [0, 146096]
    const yoe = @divTrunc(doe - @divTrunc(doe, 1460) + @divTrunc(doe, 36_524) - @divTrunc(doe, 146_096), 365); // [0, 399]
    const doy = doe - (365 * yoe + @divTrunc(yoe, 4) - @divTrunc(yoe, 100)); // [0, 365], March 1 = 0
    const mp = @divTrunc(5 * doy + 2, 153); // [0, 11], March = 0
    const month = if (mp < 10) mp + 3 else mp - 9;
    return .{ .year = yoe + era * 400 + @intFromBool(month <= 2), .month = month, .day = doy - @divTrunc(153 * mp + 2, 5) + 1 };
}

/// Return whether `year` is a Gregorian leap year.
pub fn isLeap(year: i64) bool {
    return @mod(year, 4) == 0 and (@mod(year, 100) != 0 or @mod(year, 400) == 0);
}

/// Return the 1-based day of the year of `days` after 1970-01-01.
pub fn dayOfYear(days: i64) i64 {
    return dayOfYearWith(civil, days);
}

inline fn dayOfYearWith(comptime civ: fn (i64) Civil, days: i64) i64 {
    const c = civ(days);
    const cumulative = [_]i64{ 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334 };
    return cumulative[@intCast(c.month - 1)] + c.day + @intFromBool(c.month > 2 and isLeap(c.year));
}

/// Return the ISO 8601 week-numbering year and week (1..53) of `days` after 1970-01-01: the
/// year and week of the Thursday of its Monday-based week.
pub fn isoWeek(days: i64) IsoWeek {
    return isoWeekWith(civil, days);
}

/// ISO 8601 week-numbering year and week.
pub const IsoWeek = struct { year: i64, week: i64 };

inline fn isoWeekWith(comptime civ: fn (i64) Civil, days: i64) IsoWeek {
    const thursday = days - @mod(days + 3, 7) + 3;
    return .{ .year = civ(thursday).year, .week = @divTrunc(dayOfYearWith(civ, thursday) - 1, 7) + 1 };
}
