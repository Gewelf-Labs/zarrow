const std = @import("std");
const common = @import("test_common.zig");

const ArrayRef = common.ArrayRef;
const DataType = common.DataType;
const Datum = common.Datum;
const ExecContext = common.ExecContext;
const Scalar = common.Scalar;
const none = common.none;

const date32 = DataType{ .date32 = {} };
const date64 = DataType{ .date64 = {} };
const ts_s = DataType{ .timestamp = .{ .unit = .second, .timezone = null } };
const ts_ms = DataType{ .timestamp = .{ .unit = .millisecond, .timezone = null } };
const ts_us_utc = DataType{ .timestamp = .{ .unit = .microsecond, .timezone = "UTC" } };
const ts_ns = DataType{ .timestamp = .{ .unit = .nanosecond, .timezone = null } };

const date_fields = [_][]const u8{ "year", "month", "day", "day_of_week", "day_of_year", "iso_year", "iso_week", "quarter" };
const time_fields = [_][]const u8{ "hour", "minute", "second", "millisecond", "microsecond", "nanosecond" };

test "temporal hand-written dates: components of known days, leap years, ISO weeks and week options" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);

    // 1970-01-01, 1969-12-31, 2000-02-29, 2021-01-01, 2020-12-31, 2008-12-29, 0001-01-01,
    // 9999-12-31, 1900-01-01, 2023-01-01, null (expected values from Python's datetime).
    var d = try common.makeTyped(i32, date32, allocator, &.{ 0, -1, 11016, 18628, 18627, 14242, -719162, 2932896, -25567, 19358, null });
    defer d.release();
    const args = [_]Datum{.{ .array = d }};
    try common.expectInvoke(i64, &ctx, .scalar, "year", &args, &.{ 1970, 1969, 2000, 2021, 2020, 2008, 1, 9999, 1900, 2023, null });
    try common.expectInvoke(i64, &ctx, .scalar, "month", &args, &.{ 1, 12, 2, 1, 12, 12, 1, 12, 1, 1, null });
    try common.expectInvoke(i64, &ctx, .scalar, "day", &args, &.{ 1, 31, 29, 1, 31, 29, 1, 31, 1, 1, null });
    try common.expectInvoke(i64, &ctx, .scalar, "day_of_week", &args, &.{ 3, 2, 1, 4, 3, 0, 0, 4, 0, 6, null });
    try common.expectInvoke(i64, &ctx, .scalar, "day_of_year", &args, &.{ 1, 365, 60, 1, 366, 364, 1, 365, 1, 1, null });
    try common.expectInvoke(i64, &ctx, .scalar, "iso_year", &args, &.{ 1970, 1970, 2000, 2020, 2020, 2009, 1, 9999, 1900, 2022, null });
    try common.expectInvoke(i64, &ctx, .scalar, "iso_week", &args, &.{ 1, 1, 9, 53, 53, 1, 1, 52, 1, 52, null });
    try common.expectInvoke(i64, &ctx, .scalar, "quarter", &args, &.{ 1, 4, 1, 1, 4, 4, 1, 4, 1, 1, null });
    try common.expectInvoke(bool, &ctx, .scalar, "is_leap_year", &args, &.{ false, false, true, false, true, true, false, false, false, false, null });

    // DayOfWeekOptions: Sunday-based weeks counted from 1 (Sunday = 1 ... Saturday = 7).
    const sunday_one = common.Options{ .day_of_week = .{ .count_from_zero = false, .week_start = 7 } };
    try common.expectInvokeWith(i64, &ctx, "day_of_week", &args, sunday_one, &.{ 5, 4, 3, 6, 5, 2, 2, 6, 2, 1, null });

    // iso_calendar: struct of iso_year, iso_week, iso_day_of_week; null rows stay null.
    var cal = try ctx.invokeScalar("iso_calendar", &args, none);
    defer cal.release();
    try cal.array.data().validateFull();
    try std.testing.expect(cal.array.data().data_type.eql(common.kernels.temporal.iso_calendar_type));
    try std.testing.expect(cal.array.data().isNull(10));
    const iso_day = try common.collect(i64, allocator, .{ .array = cal.array.data().children[2] });
    defer allocator.free(iso_day);
    try common.expectRows(i64, &.{ 4, 3, 2, 5, 4, 1, 1, 5, 1, 7 }, iso_day[0..10]);
    const iso_week = try common.collect(i64, allocator, .{ .array = cal.array.data().children[1] });
    defer allocator.free(iso_week);
    try common.expectRows(i64, &.{ 1, 1, 9, 53, 53, 1, 1, 52, 1, 52 }, iso_week[0..10]);

    // Timestamps: -1 ms is 1969-12-31 23:59:59.999; 1700000000123 ms is 2023-11-14 22:13:20.123;
    // 951786123004 ms is 2000-02-29 01:02:03.004.
    var t = try common.makeTyped(i64, ts_ms, allocator, &.{ -1, 1700000000123, 951786123004, null });
    defer t.release();
    const targs = [_]Datum{.{ .array = t }};
    try common.expectInvoke(i64, &ctx, .scalar, "year", &targs, &.{ 1969, 2023, 2000, null });
    try common.expectInvoke(i64, &ctx, .scalar, "day_of_year", &targs, &.{ 365, 318, 60, null });
    try common.expectInvoke(i64, &ctx, .scalar, "hour", &targs, &.{ 23, 22, 1, null });
    try common.expectInvoke(i64, &ctx, .scalar, "minute", &targs, &.{ 59, 13, 2, null });
    try common.expectInvoke(i64, &ctx, .scalar, "second", &targs, &.{ 59, 20, 3, null });
    try common.expectInvoke(i64, &ctx, .scalar, "millisecond", &targs, &.{ 999, 123, 4, null });
    try common.expectInvoke(i64, &ctx, .scalar, "microsecond", &targs, &.{ 0, 0, 0, null });
    try common.expectInvoke(i64, &ctx, .scalar, "iso_week", &targs, &.{ 1, 46, 9, null });

    // Nanosecond timestamps split into millisecond / microsecond / nanosecond sub-fields; a UTC
    // timezone reads like no timezone.
    var tn = try common.makeTyped(i64, ts_ns, allocator, &.{ 1_234_567_891, -1 });
    defer tn.release();
    try common.expectInvoke(i64, &ctx, .scalar, "second", &.{.{ .array = tn }}, &.{ 1, 59 });
    try common.expectInvoke(i64, &ctx, .scalar, "millisecond", &.{.{ .array = tn }}, &.{ 234, 999 });
    try common.expectInvoke(i64, &ctx, .scalar, "microsecond", &.{.{ .array = tn }}, &.{ 567, 999 });
    try common.expectInvoke(i64, &ctx, .scalar, "nanosecond", &.{.{ .array = tn }}, &.{ 891, 999 });
    var tu = try common.makeTyped(i64, ts_us_utc, allocator, &.{ 86_400_000_000 + 3_600_000_000, std.math.maxInt(i64), std.math.minInt(i64) });
    defer tu.release();
    try common.expectInvoke(i64, &ctx, .scalar, "day", &.{.{ .array = tu }}, &.{ 2, 10, 21 });
    try common.expectInvoke(i64, &ctx, .scalar, "hour", &.{.{ .array = tu }}, &.{ 1, 4, 19 });
    try common.expectInvoke(i64, &ctx, .scalar, "year", &.{.{ .array = tu }}, &.{ 1970, 294247, -290308 });

    // date64 is milliseconds since the epoch.
    var d64 = try common.makeTyped(i64, date64, allocator, &.{ 951782400000, -86400000 });
    defer d64.release();
    try common.expectInvoke(i64, &ctx, .scalar, "month", &.{.{ .array = d64 }}, &.{ 2, 12 });
    try common.expectInvoke(i64, &ctx, .scalar, "hour", &.{.{ .array = d64 }}, &.{ 0, 0 });
}

/// Days of the year `y` (reference: no shortcut through the implementation).
fn yearLen(y: i64) i64 {
    const leap = @mod(y, 4) == 0 and (@mod(y, 100) != 0 or @mod(y, 400) == 0);
    return if (leap) 366 else 365;
}

fn monthLen(y: i64, m: i64) i64 {
    const lens = [_]i64{ 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 };
    return lens[@intCast(m - 1)] + @intFromBool(m == 2 and yearLen(y) == 366);
}

/// Days from 1970-01-01 to January 1 of `y`, walking year by year.
fn yearStart(y: i64) i64 {
    var days: i64 = 0;
    var k: i64 = 1970;
    while (k < y) : (k += 1) days += yearLen(k);
    while (k > y) : (k -= 1) days -= yearLen(k - 1);
    return days;
}

const RefDate = struct { y: i64, m: i64, d: i64, doy: i64 };

/// Reference calendar: walk years, then months, from 1970-01-01.
fn refDate(days: i64) RefDate {
    var y: i64 = 1970;
    var rem = days;
    while (rem < 0) {
        y -= 1;
        rem += yearLen(y);
    }
    while (rem >= yearLen(y)) {
        rem -= yearLen(y);
        y += 1;
    }
    const doy = rem + 1;
    var m: i64 = 1;
    while (rem >= monthLen(y, m)) {
        rem -= monthLen(y, m);
        m += 1;
    }
    return .{ .y = y, .m = m, .d = rem + 1, .doy = doy };
}

/// Monday of ISO week 1 of `y`: the week holding January 4.
fn isoWeekOneMonday(y: i64) i64 {
    const jan4 = yearStart(y) + 3;
    return jan4 - @mod(jan4 + 3, 7);
}

/// Reference ISO year and week from the week-1 definition.
fn refIso(days: i64) [2]i64 {
    var y = refDate(days).y;
    if (days < isoWeekOneMonday(y)) y -= 1 else if (days >= isoWeekOneMonday(y + 1)) y += 1;
    return .{ y, @divFloor(days - isoWeekOneMonday(y), 7) + 1 };
}

fn refField(name: []const u8, days: i64, ns_of_day: i64) i64 {
    const eq = std.mem.eql;
    const r = refDate(days);
    if (eq(u8, name, "year")) return r.y;
    if (eq(u8, name, "month")) return r.m;
    if (eq(u8, name, "day")) return r.d;
    if (eq(u8, name, "day_of_week")) return @mod(days + 3, 7);
    if (eq(u8, name, "day_of_year")) return r.doy;
    if (eq(u8, name, "iso_year")) return refIso(days)[0];
    if (eq(u8, name, "iso_week")) return refIso(days)[1];
    if (eq(u8, name, "quarter")) return @divFloor(r.m + 2, 3);
    if (eq(u8, name, "hour")) return @divFloor(ns_of_day, 3600 * std.time.ns_per_s);
    if (eq(u8, name, "minute")) return @mod(@divFloor(ns_of_day, 60 * std.time.ns_per_s), 60);
    if (eq(u8, name, "second")) return @mod(@divFloor(ns_of_day, std.time.ns_per_s), 60);
    if (eq(u8, name, "millisecond")) return @mod(@divFloor(ns_of_day, std.time.ns_per_ms), 1000);
    if (eq(u8, name, "microsecond")) return @mod(@divFloor(ns_of_day, std.time.ns_per_us), 1000);
    return @mod(ns_of_day, 1000);
}

/// Random temporal values: days within about +-220 years plus a random time of day, about 20% null.
fn randomRows(allocator: std.mem.Allocator, rng: std.Random, len: usize, per_day: i64) ![]?i64 {
    const rows = try allocator.alloc(?i64, len);
    for (rows) |*r| {
        const day = rng.intRangeAtMost(i64, -80_000, 80_000);
        r.* = if (rng.uintLessThan(u8, 5) == 0) null else day * per_day + if (per_day > 1) rng.intRangeLessThan(i64, 0, per_day) else 0;
    }
    return rows;
}

test "temporal oracle matches a per-row reference loop on random sliced inputs with nulls for every input type and unit" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    var prng = std.Random.DefaultPrng.init(0x7E4D);
    const rng = prng.random();

    const Case = struct { dt: DataType, per_day: i64, ns_per_unit: i64 };
    const cases = [_]Case{
        .{ .dt = date32, .per_day = 1, .ns_per_unit = 0 },
        .{ .dt = date64, .per_day = 86_400_000, .ns_per_unit = 1_000_000 },
        .{ .dt = ts_s, .per_day = 86_400, .ns_per_unit = 1_000_000_000 },
        .{ .dt = ts_ms, .per_day = 86_400_000, .ns_per_unit = 1_000_000 },
        .{ .dt = ts_us_utc, .per_day = 86_400_000_000, .ns_per_unit = 1_000 },
        .{ .dt = ts_ns, .per_day = 86_400_000_000_000, .ns_per_unit = 1 },
    };
    inline for (cases) |c| {
        const rows = try randomRows(allocator, rng, 400, c.per_day);
        defer allocator.free(rows);
        var full = if (c.dt == .date32) blk: {
            var narrow: [400]?i32 = undefined;
            for (rows, &narrow) |r, *n| n.* = if (r) |x| @intCast(x) else null;
            break :blk try common.makeTyped(i32, c.dt, allocator, &narrow);
        } else try common.makeTyped(i64, c.dt, allocator, rows);
        defer full.release();
        var sliced = try full.slice(13, 380);
        defer sliced.release();
        const args = [_]Datum{.{ .array = sliced }};
        const fields = if (c.dt == .date32) date_fields else date_fields ++ time_fields;
        inline for (fields) |name| {
            var expected: [380]?i64 = undefined;
            for (&expected, rows[13..][0..380]) |*e, r| e.* = if (r) |v| refField(name, @divFloor(v, c.per_day), @mod(v, c.per_day) * c.ns_per_unit) else null;
            try common.expectInvoke(i64, &ctx, .scalar, name, &args, &expected);
        }
        var leap: [380]?bool = undefined;
        for (&leap, rows[13..][0..380]) |*e, r| e.* = if (r) |v| yearLen(refDate(@divFloor(v, c.per_day)).y) == 366 else null;
        try common.expectInvoke(bool, &ctx, .scalar, "is_leap_year", &args, &leap);
    }
}

test "temporal chunked inputs with misaligned chunks give a chunked result equal to the array result" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    var prng = std.Random.DefaultPrng.init(0x7E4E);
    const rows = try randomRows(allocator, prng.random(), 300, 86_400_000);
    defer allocator.free(rows);
    var arr = try common.makeTyped(i64, ts_ms, allocator, rows);
    defer arr.release();
    var chunked = try common.chunkedAt(allocator, arr, &.{ 5, 64, 200 });
    defer chunked.release();
    inline for (.{ "year", "day_of_week", "hour", "iso_week" }) |name| {
        var whole = try ctx.invokeScalar(name, &.{.{ .array = arr }}, none);
        defer whole.release();
        const expected = try common.collect(i64, allocator, whole);
        defer allocator.free(expected);
        var out = try ctx.invokeScalar(name, &.{.{ .chunked = chunked }}, none);
        defer out.release();
        try std.testing.expectEqual(@as(usize, 4), out.chunked.numChunks());
        const got = try common.collect(i64, allocator, out);
        defer allocator.free(got);
        try common.expectRows(i64, expected, got);
    }
    var cal = try ctx.invokeScalar("iso_calendar", &.{.{ .chunked = chunked }}, none);
    defer cal.release();
    for (cal.chunked.chunks()) |c| try c.data().validateFull();
}

test "temporal scalars, empty and all-null inputs" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);

    var y = try ctx.invokeScalar("year", &.{.{ .scalar = Scalar.init(date32, .{ .date32 = 11016 }) }}, none);
    defer y.release();
    try std.testing.expectEqual(@as(i64, 2000), y.scalar.value.i64);
    var h = try ctx.invokeScalar("hour", &.{.{ .scalar = Scalar.init(ts_s, .{ .timestamp = 7200 }) }}, none);
    defer h.release();
    try std.testing.expectEqual(@as(i64, 2), h.scalar.value.i64);
    var leap = try ctx.invokeScalar("is_leap_year", &.{.{ .scalar = Scalar.init(date64, .{ .date64 = 951782400000 }) }}, none);
    defer leap.release();
    try std.testing.expect(leap.scalar.value.bool);
    var n = try ctx.invokeScalar("month", &.{.{ .scalar = Scalar.init(date32, .null) }}, none);
    defer n.release();
    try std.testing.expect(n.scalar.isNull() and n.scalar.data_type == .int64);
    var cal = try ctx.invokeScalar("iso_calendar", &.{.{ .scalar = Scalar.init(date32, .{ .date32 = 18628 }) }}, none);
    defer cal.release();
    const payload = cal.scalar.payload.?.data();
    try std.testing.expectEqual(@as(i64, 2020), std.mem.bytesAsSlice(i64, payload.children[0].data().buffers[1].data)[payload.offset]);
    var cal_null = try ctx.invokeScalar("iso_calendar", &.{.{ .scalar = Scalar.init(date32, .null) }}, none);
    defer cal_null.release();
    try std.testing.expect(cal_null.scalar.isNull());

    var empty = try common.makeTyped(i32, date32, allocator, &.{});
    defer empty.release();
    try common.expectInvoke(i64, &ctx, .scalar, "year", &.{.{ .array = empty }}, &.{});
    // Null rows hold extreme values; nothing is computed into the valid output.
    const values = [_]i64{ std.math.minInt(i64), std.math.maxInt(i64), 0 };
    const all_null = [_]u8{0};
    const bufs = [_]common.SharedBuffer{ .fromSlice(&all_null), .fromSlice(std.mem.sliceAsBytes(&values)) };
    var nulls = try ArrayRef.fromBorrowed(allocator, .{ .data_type = ts_ns, .length = 3, .null_count = 3, .buffers = &bufs });
    defer nulls.release();
    inline for (date_fields ++ time_fields) |name| try common.expectInvoke(i64, &ctx, .scalar, name, &.{.{ .array = nulls }}, &.{ null, null, null });
}

test "temporal errors: non-UTC timezone, time parts of date32, options, arity and malformed input" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    const ny = DataType{ .timestamp = .{ .unit = .second, .timezone = "America/New_York" } };
    var t = try common.makeTyped(i64, ny, allocator, &.{0});
    defer t.release();
    try std.testing.expectError(error.NotImplemented, ctx.invokeScalar("year", &.{.{ .array = t }}, none));
    var d = try common.makeTyped(i32, date32, allocator, &.{0});
    defer d.release();
    try std.testing.expectError(error.NoMatchingKernel, ctx.invokeScalar("hour", &.{.{ .array = d }}, none));
    try std.testing.expectError(error.InvalidOptions, ctx.invokeScalar("day_of_week", &.{.{ .array = d }}, .{ .day_of_week = .{ .week_start = 0 } }));
    try std.testing.expectError(error.InvalidOptions, ctx.invokeScalar("day_of_week", &.{.{ .array = d }}, .{ .day_of_week = .{ .week_start = 8 } }));
    try std.testing.expectError(error.InvalidOptions, ctx.invokeScalar("year", &.{.{ .array = d }}, .{ .day_of_week = .{} }));
    try std.testing.expectError(error.InvalidArity, ctx.invokeScalar("year", &.{ .{ .array = d }, .{ .array = d } }, none));
    var i = try common.makeArray(i32, allocator, &.{0});
    defer i.release();
    try std.testing.expectError(error.NoMatchingKernel, ctx.invokeScalar("year", &.{.{ .array = i }}, none));
    // A values buffer shorter than the array fails validateLayout instead of being read.
    const short = [_]i32{1};
    const bufs = [_]common.SharedBuffer{ .empty, .fromSlice(std.mem.sliceAsBytes(&short)) };
    var malformed = try ArrayRef.fromBorrowed(allocator, .{ .data_type = date32, .length = 2, .null_count = 0, .buffers = &bufs });
    defer malformed.release();
    try std.testing.expectError(error.InvalidInput, ctx.invokeScalar("year", &.{.{ .array = malformed }}, none));
}

test "temporal 32-bit and vector calendar paths agree with the 64-bit algorithm across their whole range" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    const temporal = common.kernels.temporal;
    var prng = std.Random.DefaultPrng.init(0x7E4F);
    const rng = prng.random();

    const lo = temporal.fast_min;
    const hi = temporal.fast_max;
    for ([_]i64{ lo, lo + 1, hi - 1, hi, 0, -1, -719_468, 2_932_896 }) |d| try std.testing.expectEqual(temporal.civilWide(d), temporal.civilFast(d));
    for (0..20_000) |_| {
        const d = rng.intRangeAtMost(i64, lo, hi);
        try std.testing.expectEqual(temporal.civilWide(d), temporal.civilFast(d));
        try std.testing.expectEqual(temporal.civilWide(d), temporal.civil(d));
    }
    try std.testing.expectEqual(temporal.civilWide(hi + 1), temporal.civil(hi + 1));
    try std.testing.expectEqual(temporal.civilWide(lo - 1), temporal.civil(lo - 1));

    // The 8-lane kernel path (every value in range) against civilWide, with a scalar tail.
    var values: [1003]?i32 = undefined;
    for (&values) |*v| v.* = @intCast(rng.intRangeAtMost(i64, lo + 7, hi - 7));
    var d = try common.makeTyped(i32, date32, allocator, &values);
    defer d.release();
    const names = [_][]const u8{ "year", "month", "day", "quarter" };
    inline for (names, 0..) |name, k| {
        var expected: [1003]?i64 = undefined;
        for (&expected, values) |*e, v| {
            const c = temporal.civilWide(v.?);
            e.* = switch (k) {
                0 => c.year,
                1 => c.month,
                2 => c.day,
                else => @divTrunc(c.month - 1, 3) + 1,
            };
        }
        try common.expectInvoke(i64, &ctx, .scalar, name, &.{.{ .array = d }}, &expected);
    }
}

test "temporal week, us_week, subsecond and year_month_day: every WeekOptions combination (Arrow C++ results)" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);

    // 2021-01-01, 2021-01-03, 2021-01-04, 2020-12-31, 2019-12-30, 2023-01-01, 2022-01-01,
    // 2024-12-30, null; expected values from pyarrow 26.
    var d = try common.makeTyped(i32, date32, allocator, &.{ 18628, 18630, 18631, 18627, 18260, 19358, 18993, 20087, null });
    defer d.release();
    const args = [_]Datum{.{ .array = d }};
    try common.expectInvoke(i64, &ctx, .scalar, "us_week", &args, &.{ 53, 1, 1, 53, 1, 1, 52, 1, null });
    try common.expectInvoke(i64, &ctx, .scalar, "week", &args, &.{ 53, 53, 1, 53, 1, 52, 52, 1, null });
    const Case = struct { monday: bool, zero: bool, full: bool, want: [8]i64 };
    for ([_]Case{
        .{ .monday = true, .zero = true, .full = true, .want = .{ 0, 0, 1, 52, 52, 0, 0, 53 } },
        .{ .monday = true, .zero = true, .full = false, .want = .{ 0, 0, 1, 53, 53, 0, 0, 53 } },
        .{ .monday = true, .zero = false, .full = true, .want = .{ 52, 52, 1, 52, 52, 52, 52, 53 } },
        .{ .monday = true, .zero = false, .full = false, .want = .{ 53, 53, 1, 53, 1, 52, 52, 1 } },
        .{ .monday = false, .zero = true, .full = true, .want = .{ 0, 1, 1, 52, 52, 1, 0, 52 } },
        .{ .monday = false, .zero = true, .full = false, .want = .{ 0, 1, 1, 53, 53, 1, 0, 53 } },
        .{ .monday = false, .zero = false, .full = true, .want = .{ 52, 1, 1, 52, 52, 1, 52, 52 } },
        .{ .monday = false, .zero = false, .full = false, .want = .{ 53, 1, 1, 53, 1, 1, 52, 1 } },
    }) |c| {
        const o = common.Options{ .week = .{ .week_starts_monday = c.monday, .count_from_zero = c.zero, .first_week_is_fully_in_year = c.full } };
        common.expectInvokeWith(i64, &ctx, "week", &args, o, &.{ c.want[0], c.want[1], c.want[2], c.want[3], c.want[4], c.want[5], c.want[6], c.want[7], null }) catch |err| {
            std.debug.print("week {any}\n", .{o.week});
            return err;
        };
    }

    // year_month_day: struct of int64 year, month, day; null rows stay null.
    var ymd = try ctx.invokeScalar("year_month_day", &args, none);
    defer ymd.release();
    try ymd.array.data().validateFull();
    try std.testing.expect(ymd.array.data().data_type.eql(common.kernels.temporal.year_month_day_type));
    try std.testing.expect(ymd.array.data().isNull(8));
    inline for (.{ [_]i64{ 2021, 2021, 2021, 2020, 2019, 2023, 2022, 2024 }, [_]i64{ 1, 1, 1, 12, 12, 1, 1, 12 }, [_]i64{ 1, 3, 4, 31, 30, 1, 1, 30 } }, 0..) |want, k| {
        const got = try common.collect(i64, allocator, .{ .array = ymd.array.data().children[k] });
        defer allocator.free(got);
        try common.expectRows(i64, &.{ want[0], want[1], want[2], want[3], want[4], want[5], want[6], want[7] }, got[0..8]);
    }

    // subsecond: the fraction of the second as float64 (timestamps only, as in Arrow).
    var t = try common.makeTyped(i64, ts_ms, allocator, &.{ 1500, -1, null, 7000 });
    defer t.release();
    try common.expectInvoke(f64, &ctx, .scalar, "subsecond", &.{.{ .array = t }}, &.{ 0.5, 0.999, null, 0 });
    var tn = try common.makeTyped(i64, ts_ns, allocator, &.{ 1_234_567_891, -1 });
    defer tn.release();
    try common.expectInvoke(f64, &ctx, .scalar, "subsecond", &.{.{ .array = tn }}, &.{ 0.234567891, 0.999999999 });
    var d64 = try common.makeTyped(i64, date64, allocator, &.{0});
    defer d64.release();
    try std.testing.expectError(error.NoMatchingKernel, ctx.invokeScalar("subsecond", &.{.{ .array = d64 }}, none));
    try std.testing.expectError(error.InvalidOptions, ctx.invokeScalar("us_week", &args, .{ .week = .{} }));
    try std.testing.expectError(error.InvalidOptions, ctx.invokeScalar("week", &args, .{ .day_of_week = .{} }));

    // Oracle: week with the default options is iso_week, on random sliced dates with nulls; the
    // ymd fields equal year / month / day.
    var prng = std.Random.DefaultPrng.init(0x7E52);
    const rows = try randomRows(allocator, prng.random(), 400, 1);
    defer allocator.free(rows);
    var narrow: [400]?i32 = undefined;
    for (rows, &narrow) |r, *n| n.* = if (r) |x| @intCast(x) else null;
    var full = try common.makeTyped(i32, date32, allocator, &narrow);
    defer full.release();
    var sliced = try full.slice(7, 390);
    defer sliced.release();
    const sargs = [_]Datum{.{ .array = sliced }};
    var iso = try ctx.invokeScalar("iso_week", &sargs, none);
    defer iso.release();
    const expected = try common.collect(i64, allocator, iso);
    defer allocator.free(expected);
    try common.expectInvoke(i64, &ctx, .scalar, "week", &sargs, expected);
    var parts = try ctx.invokeScalar("year_month_day", &sargs, none);
    defer parts.release();
    inline for (.{ "year", "month", "day" }, 0..) |name, k| {
        var whole = try ctx.invokeScalar(name, &sargs, none);
        defer whole.release();
        const want = try common.collect(i64, allocator, whole);
        defer allocator.free(want);
        const got = try common.collect(i64, allocator, .{ .array = parts.array.data().children[k] });
        defer allocator.free(got);
        for (want, got) |w, g| if (w) |x| try std.testing.expectEqual(x, g.?);
    }
}
