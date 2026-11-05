const std = @import("std");
const common = @import("test_common.zig");

const ArrayRef = common.ArrayRef;
const DataType = common.DataType;
const Datum = common.Datum;
const ExecContext = common.ExecContext;
const Options = common.Options;
const Scalar = common.Scalar;
const none = common.none;
const CalendarUnit = common.core.CalendarUnit;

const date32 = DataType{ .date32 = {} };
const date64 = DataType{ .date64 = {} };
const ts_s = DataType{ .timestamp = .{ .unit = .second, .timezone = null } };
const ts_ms = DataType{ .timestamp = .{ .unit = .millisecond, .timezone = null } };
const ts_us_utc = DataType{ .timestamp = .{ .unit = .microsecond, .timezone = "UTC" } };
const ts_ns = DataType{ .timestamp = .{ .unit = .nanosecond, .timezone = null } };

fn isLeap(y: i64) bool {
    return @mod(y, 4) == 0 and (@mod(y, 100) != 0 or @mod(y, 400) == 0);
}

/// Days from 1970-01-01 to `y`-`m`-`d`, walking years and months (reference, no shortcut).
fn days(y: i64, m: i64, d: i64) i64 {
    var n: i64 = 0;
    var k: i64 = 1970;
    while (k < y) : (k += 1) n += if (isLeap(k)) 366 else 365;
    while (k > y) : (k -= 1) n -= if (isLeap(k - 1)) 366 else 365;
    const lens = [_]i64{ 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 };
    for (0..@intCast(m - 1)) |i| n += lens[i] + @intFromBool(i == 1 and isLeap(y));
    return n + d - 1;
}

/// Milliseconds since the epoch of a UTC date and time.
fn ms(y: i64, m: i64, d: i64, hh: i64, mm: i64, ss: i64) i64 {
    return ((days(y, m, d) * 24 + hh) * 60 + mm) * 60_000 + ss * 1000;
}

fn round(multiple: i64, unit: CalendarUnit) Options {
    return .{ .round_temporal = .{ .multiple = multiple, .unit = unit } };
}

fn roundWith(o: common.core.RoundTemporalOptions) Options {
    return .{ .round_temporal = o };
}

test "temporal rounding and differences hand-written values (Arrow C++ results)" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);

    // Expected values from pyarrow 26 on 2023-11-14 22:13:20.123, 1969-12-31 23:59:59,
    // 2024-02-29 12:00, 2021-01-01 00:00 and a null (timestamp[ms]).
    var t = try common.makeTyped(i64, ts_ms, allocator, &.{ 1_700_000_000_123, -1000, ms(2024, 2, 29, 12, 0, 0), ms(2021, 1, 1, 0, 0, 0), null });
    defer t.release();
    const a = [_]Datum{.{ .array = t }};
    const Case = struct { f: []const u8, o: Options, want: [4]i64 };
    const cases = [_]Case{
        .{ .f = "floor_temporal", .o = round(3, .second), .want = .{ ms(2023, 11, 14, 22, 13, 18), ms(1969, 12, 31, 23, 59, 57), ms(2024, 2, 29, 12, 0, 0), ms(2021, 1, 1, 0, 0, 0) } },
        .{ .f = "ceil_temporal", .o = round(3, .second), .want = .{ ms(2023, 11, 14, 22, 13, 21), 0, ms(2024, 2, 29, 12, 0, 0), ms(2021, 1, 1, 0, 0, 0) } },
        .{ .f = "floor_temporal", .o = round(3, .minute), .want = .{ ms(2023, 11, 14, 22, 12, 0), ms(1969, 12, 31, 23, 57, 0), ms(2024, 2, 29, 12, 0, 0), ms(2021, 1, 1, 0, 0, 0) } },
        .{ .f = "floor_temporal", .o = round(3, .hour), .want = .{ ms(2023, 11, 14, 21, 0, 0), ms(1969, 12, 31, 21, 0, 0), ms(2024, 2, 29, 12, 0, 0), ms(2021, 1, 1, 0, 0, 0) } },
        .{ .f = "ceil_temporal", .o = round(3, .hour), .want = .{ ms(2023, 11, 15, 0, 0, 0), 0, ms(2024, 2, 29, 12, 0, 0), ms(2021, 1, 1, 0, 0, 0) } },
        .{ .f = "floor_temporal", .o = round(1, .day), .want = .{ ms(2023, 11, 14, 0, 0, 0), ms(1969, 12, 31, 0, 0, 0), ms(2024, 2, 29, 0, 0, 0), ms(2021, 1, 1, 0, 0, 0) } },
        .{ .f = "ceil_temporal", .o = round(1, .day), .want = .{ ms(2023, 11, 15, 0, 0, 0), 0, ms(2024, 3, 1, 0, 0, 0), ms(2021, 1, 1, 0, 0, 0) } },
        .{ .f = "round_temporal", .o = round(1, .day), .want = .{ ms(2023, 11, 15, 0, 0, 0), 0, ms(2024, 3, 1, 0, 0, 0), ms(2021, 1, 1, 0, 0, 0) } },
        .{ .f = "floor_temporal", .o = round(3, .day), .want = .{ ms(2023, 11, 13, 0, 0, 0), ms(1969, 12, 29, 0, 0, 0), ms(2024, 2, 29, 0, 0, 0), ms(2020, 12, 31, 0, 0, 0) } },
        .{ .f = "round_temporal", .o = round(3, .day), .want = .{ ms(2023, 11, 16, 0, 0, 0), 0, ms(2024, 2, 29, 0, 0, 0), ms(2020, 12, 31, 0, 0, 0) } },
        .{ .f = "floor_temporal", .o = round(1, .week), .want = .{ ms(2023, 11, 13, 0, 0, 0), ms(1969, 12, 29, 0, 0, 0), ms(2024, 2, 26, 0, 0, 0), ms(2020, 12, 28, 0, 0, 0) } },
        .{ .f = "ceil_temporal", .o = round(3, .week), .want = .{ ms(2023, 12, 4, 0, 0, 0), ms(1970, 1, 19, 0, 0, 0), ms(2024, 3, 18, 0, 0, 0), ms(2021, 1, 18, 0, 0, 0) } },
        .{ .f = "floor_temporal", .o = roundWith(.{ .unit = .week, .week_starts_monday = false }), .want = .{ ms(2023, 11, 12, 0, 0, 0), ms(1969, 12, 28, 0, 0, 0), ms(2024, 2, 25, 0, 0, 0), ms(2020, 12, 27, 0, 0, 0) } },
        .{ .f = "floor_temporal", .o = roundWith(.{ .multiple = 2, .unit = .week, .week_starts_monday = false }), .want = .{ ms(2023, 11, 5, 0, 0, 0), ms(1969, 12, 28, 0, 0, 0), ms(2024, 2, 25, 0, 0, 0), ms(2020, 12, 20, 0, 0, 0) } },
        .{ .f = "floor_temporal", .o = round(3, .month), .want = .{ ms(2023, 10, 1, 0, 0, 0), ms(1969, 10, 1, 0, 0, 0), ms(2024, 1, 1, 0, 0, 0), ms(2021, 1, 1, 0, 0, 0) } },
        .{ .f = "ceil_temporal", .o = round(3, .month), .want = .{ ms(2024, 1, 1, 0, 0, 0), 0, ms(2024, 4, 1, 0, 0, 0), ms(2021, 4, 1, 0, 0, 0) } },
        .{ .f = "round_temporal", .o = round(3, .month), .want = .{ ms(2023, 10, 1, 0, 0, 0), 0, ms(2024, 4, 1, 0, 0, 0), ms(2021, 1, 1, 0, 0, 0) } },
        .{ .f = "floor_temporal", .o = round(3, .quarter), .want = .{ ms(2023, 4, 1, 0, 0, 0), ms(1969, 4, 1, 0, 0, 0), ms(2024, 1, 1, 0, 0, 0), ms(2021, 1, 1, 0, 0, 0) } },
        .{ .f = "round_temporal", .o = round(3, .quarter), .want = .{ ms(2024, 1, 1, 0, 0, 0), 0, ms(2024, 1, 1, 0, 0, 0), ms(2021, 1, 1, 0, 0, 0) } },
        .{ .f = "floor_temporal", .o = round(3, .year), .want = .{ ms(2022, 1, 1, 0, 0, 0), ms(1968, 1, 1, 0, 0, 0), ms(2022, 1, 1, 0, 0, 0), ms(2019, 1, 1, 0, 0, 0) } },
        .{ .f = "ceil_temporal", .o = round(3, .year), .want = .{ ms(2025, 1, 1, 0, 0, 0), ms(1971, 1, 1, 0, 0, 0), ms(2025, 1, 1, 0, 0, 0), ms(2022, 1, 1, 0, 0, 0) } },
        .{ .f = "round_temporal", .o = round(1, .year), .want = .{ ms(2024, 1, 1, 0, 0, 0), 0, ms(2024, 1, 1, 0, 0, 0), ms(2021, 1, 1, 0, 0, 0) } },
        // calendar_based_origin: multiples from the start of the next larger unit.
        .{ .f = "floor_temporal", .o = roundWith(.{ .multiple = 10, .unit = .day, .calendar_based_origin = true }), .want = .{ ms(2023, 11, 11, 0, 0, 0), ms(1969, 12, 31, 0, 0, 0), ms(2024, 2, 21, 0, 0, 0), ms(2021, 1, 1, 0, 0, 0) } },
        .{ .f = "ceil_temporal", .o = roundWith(.{ .multiple = 10, .unit = .day, .calendar_based_origin = true }), .want = .{ ms(2023, 11, 21, 0, 0, 0), ms(1970, 1, 10, 0, 0, 0), ms(2024, 3, 2, 0, 0, 0), ms(2021, 1, 1, 0, 0, 0) } },
        .{ .f = "floor_temporal", .o = roundWith(.{ .multiple = 5, .unit = .hour, .calendar_based_origin = true }), .want = .{ ms(2023, 11, 14, 20, 0, 0), ms(1969, 12, 31, 20, 0, 0), ms(2024, 2, 29, 10, 0, 0), ms(2021, 1, 1, 0, 0, 0) } },
        .{ .f = "floor_temporal", .o = roundWith(.{ .multiple = 7, .unit = .second, .calendar_based_origin = true }), .want = .{ ms(2023, 11, 14, 22, 13, 14), ms(1969, 12, 31, 23, 59, 56), ms(2024, 2, 29, 12, 0, 0), ms(2021, 1, 1, 0, 0, 0) } },
        // Weeks from week 1 of the year: a value before the origin rounds to it (Arrow quirk).
        .{ .f = "floor_temporal", .o = roundWith(.{ .multiple = 2, .unit = .week, .calendar_based_origin = true }), .want = .{ ms(2023, 11, 6, 0, 0, 0), ms(1969, 12, 29, 0, 0, 0), ms(2024, 2, 26, 0, 0, 0), ms(2021, 1, 4, 0, 0, 0) } },
        .{ .f = "floor_temporal", .o = roundWith(.{ .multiple = 2, .unit = .week, .calendar_based_origin = true, .week_starts_monday = false }), .want = .{ ms(2023, 11, 5, 0, 0, 0), ms(1970, 1, 4, 0, 0, 0), ms(2024, 2, 25, 0, 0, 0), ms(2021, 1, 3, 0, 0, 0) } },
        .{ .f = "ceil_temporal", .o = roundWith(.{ .multiple = 2, .unit = .week, .calendar_based_origin = true }), .want = .{ ms(2023, 11, 20, 0, 0, 0), ms(1970, 1, 12, 0, 0, 0), ms(2024, 3, 11, 0, 0, 0), ms(2021, 1, 4, 0, 0, 0) } },
        .{ .f = "round_temporal", .o = roundWith(.{ .multiple = 2, .unit = .week, .calendar_based_origin = true }), .want = .{ ms(2023, 11, 20, 0, 0, 0), ms(1969, 12, 29, 0, 0, 0), ms(2024, 2, 26, 0, 0, 0), ms(2021, 1, 4, 0, 0, 0) } },
        .{ .f = "floor_temporal", .o = roundWith(.{ .multiple = 5, .unit = .month, .calendar_based_origin = true }), .want = .{ ms(2023, 11, 1, 0, 0, 0), ms(1969, 11, 1, 0, 0, 0), ms(2024, 1, 1, 0, 0, 0), ms(2021, 1, 1, 0, 0, 0) } },
        .{ .f = "ceil_temporal", .o = roundWith(.{ .multiple = 5, .unit = .month, .calendar_based_origin = true }), .want = .{ ms(2024, 4, 1, 0, 0, 0), ms(1970, 4, 1, 0, 0, 0), ms(2024, 6, 1, 0, 0, 0), ms(2021, 6, 1, 0, 0, 0) } },
        .{ .f = "floor_temporal", .o = roundWith(.{ .multiple = 3, .unit = .quarter, .calendar_based_origin = true }), .want = .{ ms(2023, 10, 1, 0, 0, 0), ms(1969, 10, 1, 0, 0, 0), ms(2024, 1, 1, 0, 0, 0), ms(2021, 1, 1, 0, 0, 0) } },
    };
    for (cases) |c| {
        common.expectInvokeWith(i64, &ctx, c.f, &a, c.o, &.{ c.want[0], c.want[1], c.want[2], c.want[3], null }) catch |err| {
            std.debug.print("{s} {any}\n", .{ c.f, c.o.round_temporal });
            return err;
        };
    }
    // `none` rounds to one day.
    try common.expectInvokeWith(i64, &ctx, "floor_temporal", &a, none, &.{ ms(2023, 11, 14, 0, 0, 0), ms(1969, 12, 31, 0, 0, 0), ms(2024, 2, 29, 0, 0, 0), ms(2021, 1, 1, 0, 0, 0), null });

    // ceil_is_strictly_greater moves a value on a boundary to the next one.
    var exact = try common.makeTyped(i64, ts_s, allocator, &.{@divExact(ms(2021, 1, 1, 0, 0, 0), 1000)});
    defer exact.release();
    const e = [_]Datum{.{ .array = exact }};
    const day_s = @divExact(ms(2021, 1, 2, 0, 0, 0), 1000);
    try common.expectInvokeWith(i64, &ctx, "ceil_temporal", &e, roundWith(.{ .unit = .day, .ceil_is_strictly_greater = true }), &.{day_s});
    try common.expectInvokeWith(i64, &ctx, "ceil_temporal", &e, round(1, .day), &.{@divExact(ms(2021, 1, 1, 0, 0, 0), 1000)});
    try common.expectInvokeWith(i64, &ctx, "ceil_temporal", &e, roundWith(.{ .unit = .month, .ceil_is_strictly_greater = true }), &.{@divExact(ms(2021, 2, 1, 0, 0, 0), 1000)});

    // A unit finer than the input: computed in the unit, truncated back (duration_cast).
    var s = try common.makeTyped(i64, ts_s, allocator, &.{ -7, 7, -1 });
    defer s.release();
    try common.expectInvokeWith(i64, &ctx, "floor_temporal", &.{.{ .array = s }}, round(1500, .millisecond), &.{ -7, 6, -1 });
    try common.expectInvokeWith(i64, &ctx, "ceil_temporal", &.{.{ .array = s }}, round(1500, .millisecond), &.{ -7, 7, -1 });
    // ceil adds the multiple truncated to the input unit (1500 ms -> 1 s, 2500 ms -> 2 s).
    try common.expectInvokeWith(i64, &ctx, "ceil_temporal", &.{.{ .array = s }}, roundWith(.{ .multiple = 1500, .unit = .millisecond, .ceil_is_strictly_greater = true }), &.{ -6, 7, 0 });
    try common.expectInvokeWith(i64, &ctx, "ceil_temporal", &.{.{ .array = s }}, roundWith(.{ .multiple = 1, .unit = .nanosecond, .ceil_is_strictly_greater = true }), &.{ -7, 7, -1 });
    var s2 = try common.makeTyped(i64, ts_s, allocator, &.{ -6, 8 });
    defer s2.release();
    try common.expectInvokeWith(i64, &ctx, "ceil_temporal", &.{.{ .array = s2 }}, round(2500, .millisecond), &.{ -5, 9 });

    // Weeks with calendar_based_origin count from week 1 of the year of the value shifted by its
    // weekday offset, without shifting back (Arrow C++); a multiple of 1 ignores the origin.
    var w = try common.makeTyped(i32, date32, allocator, &.{ @intCast(days(1937, 2, 26)), @intCast(days(2021, 1, 1)) });
    defer w.release();
    try common.expectInvokeWith(i32, &ctx, "floor_temporal", &.{.{ .array = w }}, roundWith(.{ .multiple = 2, .unit = .week, .calendar_based_origin = true }), &.{ @intCast(days(1937, 3, 1)), @intCast(days(2021, 1, 4)) });
    try common.expectInvokeWith(i32, &ctx, "floor_temporal", &.{.{ .array = w }}, roundWith(.{ .multiple = 2, .unit = .week, .calendar_based_origin = true, .week_starts_monday = false }), &.{ @intCast(days(1937, 2, 28)), @intCast(days(2021, 1, 3)) });
    try common.expectInvokeWith(i32, &ctx, "floor_temporal", &.{.{ .array = w }}, roundWith(.{ .multiple = 1, .unit = .week, .calendar_based_origin = true }), &.{ @intCast(days(1937, 2, 22)), @intCast(days(2020, 12, 28)) });
    try common.expectInvokeWith(i64, &ctx, "floor_temporal", &.{.{ .array = s }}, round(3, .hour), &.{ -10800, 0, -10800 });

    // date32 and date64 keep their type.
    var d = try common.makeTyped(i32, date32, allocator, &.{ @intCast(days(2023, 11, 14)), -1, null });
    defer d.release();
    const da = [_]Datum{.{ .array = d }};
    try common.expectInvokeWith(i32, &ctx, "floor_temporal", &da, round(1, .month), &.{ @intCast(days(2023, 11, 1)), @intCast(days(1969, 12, 1)), null });
    try common.expectInvokeWith(i32, &ctx, "floor_temporal", &da, round(5, .hour), &.{ @intCast(days(2023, 11, 14)), -1, null });
    try common.expectInvokeWith(i32, &ctx, "ceil_temporal", &da, round(5, .hour), &.{ @intCast(days(2023, 11, 14)), -1, null });
    try common.expectInvokeWith(i32, &ctx, "ceil_temporal", &da, round(1, .week), &.{ @intCast(days(2023, 11, 20)), 4, null });
    var d64 = try common.makeTyped(i64, date64, allocator, &.{ms(2023, 11, 14, 0, 0, 0)});
    defer d64.release();
    try common.expectInvokeWith(i64, &ctx, "floor_temporal", &.{.{ .array = d64 }}, round(1, .week), &.{ms(2023, 11, 13, 0, 0, 0)});
    var out = try ctx.invokeScalar("floor_temporal", &.{.{ .array = d64 }}, none);
    defer out.release();
    try std.testing.expect(out.array.data().data_type == .date64);

    // Differences (pyarrow 26): [2020-12-31 23:00, 2021-01-04, null] to [2021-01-01 01:00, 2020-12-27, 2020-01-01].
    var x = try common.makeTyped(i64, ts_s, allocator, &.{ @divExact(ms(2020, 12, 31, 23, 0, 0), 1000), @divExact(ms(2021, 1, 4, 0, 0, 0), 1000), null });
    defer x.release();
    var y = try common.makeTyped(i64, ts_s, allocator, &.{ @divExact(ms(2021, 1, 1, 1, 0, 0), 1000), @divExact(ms(2020, 12, 27, 0, 0, 0), 1000), @divExact(ms(2020, 1, 1, 0, 0, 0), 1000) });
    defer y.release();
    const xy = [_]Datum{ .{ .array = x }, .{ .array = y } };
    const B = struct { f: []const u8, want: [2]i64 };
    for ([_]B{
        .{ .f = "years_between", .want = .{ 1, -1 } },
        .{ .f = "quarters_between", .want = .{ 1, -1 } },
        .{ .f = "weeks_between", .want = .{ 0, -2 } },
        .{ .f = "days_between", .want = .{ 1, -8 } },
        .{ .f = "hours_between", .want = .{ 2, -192 } },
        .{ .f = "minutes_between", .want = .{ 120, -11_520 } },
        .{ .f = "seconds_between", .want = .{ 7200, -691_200 } },
        .{ .f = "milliseconds_between", .want = .{ 7_200_000, -691_200_000 } },
        .{ .f = "microseconds_between", .want = .{ 7_200_000_000, -691_200_000_000 } },
        .{ .f = "nanoseconds_between", .want = .{ 7_200_000_000_000, -691_200_000_000_000 } },
    }) |c| try common.expectInvokeWith(i64, &ctx, c.f, &xy, none, &.{ c.want[0], c.want[1], null });
    try common.expectInvokeWith(i64, &ctx, "weeks_between", &xy, .{ .day_of_week = .{ .week_start = 7 } }, &.{ 0, -1, null });
    try common.expectInvokeWith(i32, &ctx, "month_interval_between", &xy, none, &.{ 1, -1, null });
    var months = try ctx.invokeScalar("month_interval_between", &xy, none);
    defer months.release();
    try std.testing.expect(months.array.data().data_type == .interval_months);

    var d1 = try common.makeTyped(i32, date32, allocator, &.{ @intCast(days(2020, 1, 31)), @intCast(days(1, 1, 1)) });
    defer d1.release();
    var d2 = try common.makeTyped(i32, date32, allocator, &.{ @intCast(days(2020, 2, 1)), @intCast(days(9999, 1, 1)) });
    defer d2.release();
    // Out-of-range differences wrap like Arrow's (9999-01-01 minus 0001-01-01 in nanoseconds).
    try common.expectInvokeWith(i64, &ctx, "nanoseconds_between", &.{ .{ .array = d1 }, .{ .array = d2 } }, none, &.{ 86_400_000_000_000, 1_911_712_346_937_622_528 });
    try common.expectInvokeWith(i64, &ctx, "hours_between", &.{ .{ .array = d1 }, .{ .array = d2 } }, none, &.{ 24, 87_640_656 });
}

/// Random timestamps of `unit` (`per_s` ticks per second) within about +-200 years, 20% null.
fn randomRows(allocator: std.mem.Allocator, rng: std.Random, len: usize, per_s: i64) ![]?i64 {
    const rows = try allocator.alloc(?i64, len);
    for (rows) |*r| {
        const sec = rng.intRangeAtMost(i64, -6_000_000_000, 6_000_000_000);
        r.* = if (rng.uintLessThan(u8, 5) == 0) null else sec * per_s + rng.intRangeLessThan(i64, 0, per_s);
    }
    return rows;
}

/// Reference floor of `t` (ticks of `per_s` per second) to `multiple` fixed units of `unit_s`
/// seconds since the epoch (weeks from the Monday 1969-12-29).
fn refFloor(t: i64, per_s: i64, unit_s: i64, multiple: i64, shift_s: i64) i64 {
    const u = unit_s * per_s * multiple;
    return @divFloor(t + shift_s * per_s, u) * u - shift_s * per_s;
}

test "temporal rounding oracle matches a per-row reference on random sliced inputs with nulls for every unit" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    var prng = std.Random.DefaultPrng.init(0x7E50);
    const rng = prng.random();
    const Input = struct { dt: DataType, per_s: i64 };
    const inputs = [_]Input{ .{ .dt = ts_s, .per_s = 1 }, .{ .dt = ts_ms, .per_s = 1000 }, .{ .dt = ts_us_utc, .per_s = 1_000_000 }, .{ .dt = ts_ns, .per_s = 1_000_000_000 }, .{ .dt = date64, .per_s = 1000 } };
    const Unit = struct { unit: CalendarUnit, seconds: i64, multiple: i64, shift: i64 = 0 };
    const units = [_]Unit{
        .{ .unit = .second, .seconds = 1, .multiple = 7 },
        .{ .unit = .minute, .seconds = 60, .multiple = 15 },
        .{ .unit = .hour, .seconds = 3600, .multiple = 5 },
        .{ .unit = .day, .seconds = 86_400, .multiple = 1 },
        .{ .unit = .day, .seconds = 86_400, .multiple = 3 },
        .{ .unit = .week, .seconds = 7 * 86_400, .multiple = 2, .shift = 3 * 86_400 },
    };
    inline for (inputs) |in| {
        const rows = try randomRows(allocator, rng, 400, in.per_s);
        defer allocator.free(rows);
        var full = try common.makeTyped(i64, in.dt, allocator, rows);
        defer full.release();
        var arr = try full.slice(9, 390);
        defer arr.release();
        const args = [_]Datum{.{ .array = arr }};
        for (units) |u| {
            const per_s = in.per_s;
            var floors: [390]?i64 = undefined;
            var ceils: [390]?i64 = undefined;
            var rounds: [390]?i64 = undefined;
            for (rows[9..][0..390], &floors, &ceils, &rounds) |r, *f, *c, *rd| {
                const v = r orelse {
                    f.* = null;
                    c.* = null;
                    rd.* = null;
                    continue;
                };
                const fl = refFloor(v, per_s, u.seconds, u.multiple, u.shift);
                const step = u.seconds * per_s * u.multiple;
                const ce = if (fl < v) fl + step else fl;
                f.* = fl;
                c.* = ce;
                rd.* = if (v - fl >= ce - v) ce else fl;
            }
            const o = round(u.multiple, u.unit);
            try common.expectInvokeWith(i64, &ctx, "floor_temporal", &args, o, &floors);
            try common.expectInvokeWith(i64, &ctx, "ceil_temporal", &args, o, &ceils);
            try common.expectInvokeWith(i64, &ctx, "round_temporal", &args, o, &rounds);
        }
        // Months: the first day of the floored month (reference calendar).
        var floors: [390]?i64 = undefined;
        for (rows[9..][0..390], &floors) |r, *f| f.* = if (r) |v| blk: {
            const day = @divFloor(v, 86_400 * in.per_s);
            var yy: i64 = 1970 + @divFloor(day * 400, 146_097) - 1;
            while (days(yy + 1, 1, 1) <= day) yy += 1;
            while (days(yy, 1, 1) > day) yy -= 1;
            var mm: i64 = 1;
            while (mm < 12 and days(yy, mm + 1, 1) <= day) mm += 1;
            const total = @divFloor((yy - 1970) * 12 + mm - 1, 4) * 4;
            break :blk days(1970 + @divFloor(total, 12), @mod(total, 12) + 1, 1) * 86_400 * in.per_s;
        } else null;
        try common.expectInvokeWith(i64, &ctx, "floor_temporal", &args, round(4, .month), &floors);
    }
}

test "temporal rounding and differences on chunked inputs with misaligned chunks equal the array result" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    var prng = std.Random.DefaultPrng.init(0x7E51);
    const rows = try randomRows(allocator, prng.random(), 300, 1000);
    defer allocator.free(rows);
    var arr = try common.makeTyped(i64, ts_ms, allocator, rows);
    defer arr.release();
    var a = try common.chunkedAt(allocator, arr, &.{ 5, 64, 200 });
    defer a.release();
    var b = try common.chunkedAt(allocator, arr, &.{ 100, 299 });
    defer b.release();
    const T = struct { f: []const u8, o: Options };
    for ([_]T{
        .{ .f = "floor_temporal", .o = round(1, .day) },
        .{ .f = "round_temporal", .o = round(2, .month) },
        .{ .f = "ceil_temporal", .o = roundWith(.{ .multiple = 3, .unit = .week, .calendar_based_origin = true }) },
    }) |c| {
        var whole = try ctx.invokeScalar(c.f, &.{.{ .array = arr }}, c.o);
        defer whole.release();
        const expected = try common.collect(i64, allocator, whole);
        defer allocator.free(expected);
        var out = try ctx.invokeScalar(c.f, &.{.{ .chunked = a }}, c.o);
        defer out.release();
        try std.testing.expectEqual(@as(usize, 4), out.chunked.numChunks());
        const got = try common.collect(i64, allocator, out);
        defer allocator.free(got);
        try common.expectRows(i64, expected, got);
    }
    var whole = try ctx.invokeScalar("days_between", &.{ .{ .array = arr }, .{ .array = arr } }, none);
    defer whole.release();
    var out = try ctx.invokeScalar("days_between", &.{ .{ .chunked = a }, .{ .chunked = b } }, none);
    defer out.release();
    try std.testing.expectEqual(@as(usize, 6), out.chunked.numChunks());
    const got = try common.collect(i64, allocator, out);
    defer allocator.free(got);
    for (got, rows) |g, r| try std.testing.expectEqual(if (r == null) null else @as(?i64, 0), g);
}

test "temporal rounding and differences on scalars, empty and all-null inputs" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);

    var f = try ctx.invokeScalar("floor_temporal", &.{.{ .scalar = Scalar.init(ts_s, .{ .timestamp = 90_061 }) }}, round(1, .hour));
    defer f.release();
    try std.testing.expectEqual(@as(i64, 90_000), f.scalar.value.timestamp);
    var c = try ctx.invokeScalar("ceil_temporal", &.{.{ .scalar = Scalar.init(date32, .{ .date32 = 3 }) }}, round(1, .month));
    defer c.release();
    try std.testing.expectEqual(@as(i32, 31), c.scalar.value.date32);
    var n = try ctx.invokeScalar("round_temporal", &.{.{ .scalar = Scalar.init(ts_s, .null) }}, none);
    defer n.release();
    try std.testing.expect(n.scalar.isNull());
    var arr = try common.makeTyped(i32, date32, allocator, &.{ 0, 40, null });
    defer arr.release();
    // A scalar operand broadcasts against an array.
    try common.expectInvokeWith(i64, &ctx, "days_between", &.{ .{ .scalar = Scalar.init(date32, .{ .date32 = 10 }) }, .{ .array = arr } }, none, &.{ -10, 30, null });
    var mi = try ctx.invokeScalar("month_interval_between", &.{ .{ .scalar = Scalar.init(date32, .{ .date32 = 0 }) }, .{ .scalar = Scalar.init(date32, .{ .date32 = 40 }) } }, none);
    defer mi.release();
    try std.testing.expectEqual(@as(i32, 1), mi.scalar.value.interval_months);

    var empty = try common.makeTyped(i64, ts_ns, allocator, &.{});
    defer empty.release();
    try common.expectInvokeWith(i64, &ctx, "floor_temporal", &.{.{ .array = empty }}, round(1, .year), &.{});
    try common.expectInvokeWith(i64, &ctx, "hours_between", &.{ .{ .array = empty }, .{ .array = empty } }, none, &.{});
    // Null rows hold extreme values; nothing panics and they stay null.
    const values = [_]i64{ std.math.minInt(i64), std.math.maxInt(i64), 0 };
    const all_null = [_]u8{0};
    const bufs = [_]common.SharedBuffer{ .fromSlice(&all_null), .fromSlice(std.mem.sliceAsBytes(&values)) };
    var nulls = try ArrayRef.fromBorrowed(allocator, .{ .data_type = ts_ns, .length = 3, .null_count = 3, .buffers = &bufs });
    defer nulls.release();
    for ([_]CalendarUnit{ .nanosecond, .second, .day, .week, .month, .quarter, .year }) |u| {
        for ([_]bool{ false, true }) |cal| {
            const o = roundWith(.{ .multiple = 7, .unit = u, .calendar_based_origin = cal });
            inline for (.{ "floor_temporal", "ceil_temporal", "round_temporal" }) |name| try common.expectInvokeWith(i64, &ctx, name, &.{.{ .array = nulls }}, o, &.{ null, null, null });
        }
    }
    inline for (.{ "years_between", "weeks_between", "nanoseconds_between", "month_interval_between" }) |name| {
        var out = try ctx.invokeScalar(name, &.{ .{ .array = nulls }, .{ .array = nulls } }, none);
        defer out.release();
        try std.testing.expectEqual(@as(?usize, 3), out.array.data().null_count);
    }
}

test "temporal rounding and differences errors: options, timezone, mismatched types, arity and malformed input" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    var t = try common.makeTyped(i64, ts_s, allocator, &.{0});
    defer t.release();
    const a = [_]Datum{.{ .array = t }};
    try std.testing.expectError(error.InvalidOptions, ctx.invokeScalar("floor_temporal", &a, round(0, .day)));
    try std.testing.expectError(error.InvalidOptions, ctx.invokeScalar("ceil_temporal", &a, round(-2, .hour)));
    try std.testing.expectError(error.InvalidOptions, ctx.invokeScalar("round_temporal", &a, .{ .day_of_week = .{} }));
    // One multiple of the unit does not fit in 64-bit working ticks.
    var t_ns = try common.makeTyped(i64, ts_ns, allocator, &.{0});
    defer t_ns.release();
    try std.testing.expectError(error.InvalidOptions, ctx.invokeScalar("floor_temporal", &.{.{ .array = t_ns }}, round(std.math.maxInt(i64) / 1000, .day)));
    var d = try common.makeTyped(i32, date32, allocator, &.{0});
    defer d.release();
    try std.testing.expectError(error.InvalidOptions, ctx.invokeScalar("weeks_between", &.{ a[0], a[0] }, .{ .day_of_week = .{ .week_start = 0 } }));
    try std.testing.expectError(error.InvalidOptions, ctx.invokeScalar("days_between", &.{ a[0], a[0] }, round(1, .day)));
    const ny = DataType{ .timestamp = .{ .unit = .second, .timezone = "America/New_York" } };
    var tz = try common.makeTyped(i64, ny, allocator, &.{0});
    defer tz.release();
    try std.testing.expectError(error.NotImplemented, ctx.invokeScalar("floor_temporal", &.{.{ .array = tz }}, none));
    try std.testing.expectError(error.NotImplemented, ctx.invokeScalar("days_between", &.{ .{ .array = tz }, .{ .array = tz } }, none));
    var t_ms = try common.makeTyped(i64, ts_ms, allocator, &.{0});
    defer t_ms.release();
    try std.testing.expectError(error.NoMatchingKernel, ctx.invokeScalar("days_between", &.{ a[0], .{ .array = t_ms } }, none));
    try std.testing.expectError(error.NoMatchingKernel, ctx.invokeScalar("days_between", &.{ a[0], .{ .array = d } }, none));
    try std.testing.expectError(error.InvalidArity, ctx.invokeScalar("days_between", &a, none));
    var i = try common.makeArray(i64, allocator, &.{0});
    defer i.release();
    try std.testing.expectError(error.NoMatchingKernel, ctx.invokeScalar("floor_temporal", &.{.{ .array = i }}, none));
    const short = [_]i64{1};
    const bufs = [_]common.SharedBuffer{ .empty, .fromSlice(std.mem.sliceAsBytes(&short)) };
    var malformed = try ArrayRef.fromBorrowed(allocator, .{ .data_type = ts_s, .length = 2, .null_count = 0, .buffers = &bufs });
    defer malformed.release();
    try std.testing.expectError(error.InvalidInput, ctx.invokeScalar("floor_temporal", &.{.{ .array = malformed }}, none));
    try std.testing.expectError(error.InvalidInput, ctx.invokeScalar("days_between", &.{ .{ .array = malformed }, .{ .array = malformed } }, none));
}

/// day_time_interval as stored: days in the low 32 bits, milliseconds in the high 32.
fn dayTime(d: i32, millis: i32) i64 {
    return @bitCast(@as(u64, @as(u32, @bitCast(millis))) << 32 | @as(u32, @bitCast(d)));
}

/// month_day_nano_interval as stored: months, days, then nanoseconds in the high 64 bits.
fn monthDayNano(months: i32, d: i32, nanos: i64) i128 {
    return @bitCast(@as(u128, @as(u64, @bitCast(nanos))) << 64 | @as(u128, @as(u32, @bitCast(d))) << 32 | @as(u32, @bitCast(months)));
}

test "temporal time32 / time64 inputs, us_year and interval differences (Arrow C++ results)" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);

    // Time parts of time32 / time64 (pyarrow 26; a value past a day or below zero wraps).
    const t32s = DataType{ .time32 = .{ .unit = .second } };
    const t32ms = DataType{ .time32 = .{ .unit = .millisecond } };
    const t64us = DataType{ .time64 = .{ .unit = .microsecond } };
    const t64ns = DataType{ .time64 = .{ .unit = .nanosecond } };
    var s = try common.makeTyped(i32, t32s, allocator, &.{ 3723, -1, 90000, null });
    defer s.release();
    var m = try common.makeTyped(i32, t32ms, allocator, &.{ 3723, -1, 90000, null });
    defer m.release();
    var u = try common.makeTyped(i64, t64us, allocator, &.{ 3723, -1, 90000, null });
    defer u.release();
    var n = try common.makeTyped(i64, t64ns, allocator, &.{ 3723, -1, 90000, null });
    defer n.release();
    const P = struct { name: []const u8, want: [4][4]?i64 };
    for ([_]P{
        .{ .name = "hour", .want = .{ .{ 1, 23, 1, null }, .{ 0, 23, 0, null }, .{ 0, 23, 0, null }, .{ 0, 23, 0, null } } },
        .{ .name = "minute", .want = .{ .{ 2, 59, 0, null }, .{ 0, 59, 1, null }, .{ 0, 59, 0, null }, .{ 0, 59, 0, null } } },
        .{ .name = "second", .want = .{ .{ 3, 59, 0, null }, .{ 3, 59, 30, null }, .{ 0, 59, 0, null }, .{ 0, 59, 0, null } } },
        .{ .name = "millisecond", .want = .{ .{ 0, 0, 0, null }, .{ 723, 999, 0, null }, .{ 3, 999, 90, null }, .{ 0, 999, 0, null } } },
        .{ .name = "microsecond", .want = .{ .{ 0, 0, 0, null }, .{ 0, 0, 0, null }, .{ 723, 999, 0, null }, .{ 3, 999, 90, null } } },
        .{ .name = "nanosecond", .want = .{ .{ 0, 0, 0, null }, .{ 0, 0, 0, null }, .{ 0, 0, 0, null }, .{ 723, 999, 0, null } } },
    }) |p| {
        for ([_]ArrayRef{ s, m, u, n }, p.want) |arr, want| try common.expectInvoke(i64, &ctx, .scalar, p.name, &.{.{ .array = arr }}, &want);
    }
    try common.expectInvoke(f64, &ctx, .scalar, "subsecond", &.{.{ .array = m }}, &.{ 0.723, 0.999, 0.0, null });
    try common.expectInvoke(f64, &ctx, .scalar, "subsecond", &.{.{ .array = n }}, &.{ 3.723e-06, 0.999999999, 9e-05, null });
    try common.expectInvoke(i64, &ctx, .scalar, "hour", &.{.{ .scalar = Scalar.init(t32s, .{ .time32 = 7200 }) }}, &.{2});
    try std.testing.expectError(error.NoMatchingKernel, ctx.invokeScalar("year", &.{.{ .array = m }}, none));
    try std.testing.expectError(error.NoMatchingKernel, ctx.invokeScalar("days_between", &.{ .{ .array = m }, .{ .array = m } }, none));

    // Differences of time-of-day units on time32[ms] (pyarrow 26).
    var from = try common.makeTyped(i32, t32ms, allocator, &.{ 3723000, -1, null });
    defer from.release();
    var to = try common.makeTyped(i32, t32ms, allocator, &.{ 0, 86399999, 5 });
    defer to.release();
    const ft = [_]Datum{ .{ .array = from }, .{ .array = to } };
    try common.expectInvoke(i64, &ctx, .scalar, "hours_between", &ft, &.{ -1, 24, null });
    try common.expectInvoke(i64, &ctx, .scalar, "minutes_between", &ft, &.{ -62, 1440, null });
    try common.expectInvoke(i64, &ctx, .scalar, "seconds_between", &ft, &.{ -3723, 86400, null });
    try common.expectInvoke(i64, &ctx, .scalar, "milliseconds_between", &ft, &.{ -3723000, 86400000, null });
    try common.expectInvoke(i64, &ctx, .scalar, "nanoseconds_between", &ft, &.{ -3723000000000, 86400000000000, null });

    // Rounding a time like a timestamp of its unit (pyarrow 26).
    var r = try common.makeTyped(i32, t32ms, allocator, &.{ 3723456, -1 });
    defer r.release();
    try common.expectInvokeWith(i32, &ctx, "floor_temporal", &.{.{ .array = r }}, round(1, .minute), &.{ 3720000, -60000 });
    try common.expectInvokeWith(i32, &ctx, "ceil_temporal", &.{.{ .array = r }}, round(2, .hour), &.{ 7200000, 0 });
    try common.expectInvokeWith(i32, &ctx, "floor_temporal", &.{.{ .array = r }}, round(1, .day), &.{ 0, -86400000 });
    try common.expectInvokeWith(i32, &ctx, "ceil_temporal", &.{.{ .array = r }}, round(1, .day), &.{ 86400000, 0 });

    var ru = try common.makeTyped(i64, t64us, allocator, &.{3723456});
    defer ru.release();
    try common.expectInvokeWith(i64, &ctx, "round_temporal", &.{.{ .array = ru }}, round(1, .second), &.{4000000});

    // month_day_nano_interval_between: months, then days of the month, then nanoseconds of the day (pyarrow 26).
    var d1 = try common.makeTyped(i32, date32, allocator, &.{ 0, 40, -1, null });
    defer d1.release();
    var d2 = try common.makeTyped(i32, date32, allocator, &.{ 31, 1, 59, 0 });
    defer d2.release();
    try common.expectInvoke(i128, &ctx, .scalar, "month_day_nano_interval_between", &.{ .{ .array = d1 }, .{ .array = d2 } }, &.{ monthDayNano(1, 0, 0), monthDayNano(-1, -8, 0), monthDayNano(3, -30, 0), null });
    const ts_us = DataType{ .timestamp = .{ .unit = .microsecond, .timezone = null } };
    var us_a = try common.makeTyped(i64, ts_us, allocator, &.{ 0, 1500 });
    defer us_a.release();
    var us_b = try common.makeTyped(i64, ts_us, allocator, &.{ 2678400000001, 0 });
    defer us_b.release();
    try common.expectInvoke(i128, &ctx, .scalar, "month_day_nano_interval_between", &.{ .{ .array = us_a }, .{ .array = us_b } }, &.{ monthDayNano(1, 0, 1000), monthDayNano(0, 0, -1500000) });
    var s1 = try common.makeTyped(i64, ts_s, allocator, &.{-1});
    defer s1.release();
    var s2 = try common.makeTyped(i64, ts_s, allocator, &.{86400 * 40});
    defer s2.release();
    try common.expectInvoke(i128, &ctx, .scalar, "month_day_nano_interval_between", &.{ .{ .array = s1 }, .{ .array = s2 } }, &.{monthDayNano(2, -21, -86399000000000)});
    var tm1 = try common.makeTyped(i32, t32ms, allocator, &.{3723000});
    defer tm1.release();
    var tm2 = try common.makeTyped(i32, t32ms, allocator, &.{0});
    defer tm2.release();
    try common.expectInvoke(i128, &ctx, .scalar, "month_day_nano_interval_between", &.{ .{ .array = tm1 }, .{ .array = tm2 } }, &.{monthDayNano(0, 0, -3723000000000)});
    try common.expectInvokeWith(i32, &ctx, "floor_temporal", &.{.{ .array = tm1 }}, round(1, .month), &.{0});

    // day_time_interval_between: whole days, then milliseconds of the day (Arrow C++'s
    // DayTimeBetween; pyarrow 26 cannot materialize a day_time_interval array to compare).
    var m1 = try common.makeTyped(i64, ts_ms, allocator, &.{ 0, 86400001, -1, 172800000, null });
    defer m1.release();
    var m2 = try common.makeTyped(i64, ts_ms, allocator, &.{ 90061001, 0, 1, 86400000 * 3 + 5, 0 });
    defer m2.release();
    try common.expectInvoke(i64, &ctx, .scalar, "day_time_interval_between", &.{ .{ .array = m1 }, .{ .array = m2 } }, &.{ dayTime(1, 3661001), dayTime(-1, -1), dayTime(1, -86399998), dayTime(1, 5), null });
    try common.expectInvoke(i64, &ctx, .scalar, "day_time_interval_between", &.{ .{ .array = d1 }, .{ .array = d2 } }, &.{ dayTime(31, 0), dayTime(-39, 0), dayTime(60, 0), null });
    try common.expectInvoke(i64, &ctx, .scalar, "day_time_interval_between", &.{ .{ .array = tm1 }, .{ .array = tm2 } }, &.{dayTime(0, -3723000)});
    // Scalars give interval scalars.
    var sc = try ctx.invokeScalar("day_time_interval_between", &.{ .{ .scalar = Scalar.init(date32, .{ .date32 = 0 }) }, .{ .scalar = Scalar.init(date32, .{ .date32 = 2 }) } }, none);
    defer sc.release();
    try std.testing.expectEqual(dayTime(2, 0), sc.scalar.value.interval_day_time);
    var mc = try common.chunkedAt(allocator, m1, &.{2});
    defer mc.release();
    var chunked = try ctx.invokeScalar("month_day_nano_interval_between", &.{ .{ .chunked = mc }, .{ .array = m2 } }, none);
    defer chunked.release();
    try std.testing.expectEqual(@as(usize, 2), chunked.chunked.numChunks());

    // us_year: the year of the Sunday-based us_week (pyarrow 26).
    var ud = try common.makeTyped(i32, date32, allocator, &.{ 0, 3, 4, 365, -1, 10957, 10962, null });
    defer ud.release();
    try common.expectInvoke(i64, &ctx, .scalar, "us_year", &.{.{ .array = ud }}, &.{ 1969, 1970, 1970, 1970, 1969, 1999, 2000, null });
    var ut = try common.makeTyped(i64, ts_ms, allocator, &.{ 0, 3 * 86400000 });
    defer ut.release();
    try common.expectInvoke(i64, &ctx, .scalar, "us_year", &.{.{ .array = ut }}, &.{ 1969, 1970 });
    var ud64 = try common.makeTyped(i64, date64, allocator, &.{ 0, 3 * 86400000 });
    defer ud64.release();
    try common.expectInvoke(i64, &ctx, .scalar, "us_year", &.{.{ .array = ud64 }}, &.{ 1969, 1970 });
}
