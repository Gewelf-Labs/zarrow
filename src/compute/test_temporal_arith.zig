//! Temporal arithmetic on add, subtract, multiply, divide, negate and abs. Expected values are
//! pyarrow 26 results.
const std = @import("std");
const common = @import("test_common.zig");

const ArrayRef = common.ArrayRef;
const DataType = common.DataType;
const Datum = common.Datum;
const ExecContext = common.ExecContext;
const none = common.none;

const ts_ms = DataType{ .timestamp = .{ .unit = .millisecond, .timezone = null } };
const ts_s = DataType{ .timestamp = .{ .unit = .second, .timezone = null } };
const ts_utc = DataType{ .timestamp = .{ .unit = .millisecond, .timezone = "UTC" } };
const dur_s = DataType{ .duration = .{ .unit = .second } };
const dur_ms = DataType{ .duration = .{ .unit = .millisecond } };
const dur_us = DataType{ .duration = .{ .unit = .microsecond } };

/// Invoke `name`, check the result type and the rows (stored as `T`).
fn expectTyped(comptime T: type, ctx: *ExecContext, name: []const u8, args: []const Datum, dt: DataType, expected: []const ?T) !void {
    var out = try ctx.invokeScalar(name, args, none);
    defer out.release();
    if (!out.dataType().eql(dt)) {
        std.debug.print("{s}: got type {any}, expected {any}\n", .{ name, out.dataType(), dt });
        return error.TestExpectedEqual;
    }
    try std.testing.expect((try ctx.registry.resolveResultType(name, .scalar, args, none)).eql(dt));
    switch (out) {
        .array => |arr| try arr.data().validateFull(),
        .chunked => |c| for (c.chunks()) |part| try part.data().validateFull(),
        .scalar => {},
        .record_batch, .table => return error.InvalidInput,
    }
    const rows = try collectAs(T, out);
    defer std.testing.allocator.free(rows);
    try common.expectRows(T, expected, rows);
}

/// Rows of a datum of any type stored as `T` (temporal scalars read through their own field).
fn collectAs(comptime T: type, out: Datum) ![]?T {
    if (out == .scalar) {
        const r = try std.testing.allocator.alloc(?T, 1);
        r[0] = if (out.scalar.isNull()) null else switch (out.scalar.value) {
            inline .timestamp, .duration, .date32, .date64, .time32, .time64, .f64 => |x| if (@TypeOf(x) == T) x else unreachable,
            else => return error.TestUnexpectedResult,
        };
        return r;
    }
    const allocator = std.testing.allocator;
    const parts: []const ArrayRef = if (out == .array) &.{out.array} else out.chunked.chunks();
    var total: usize = 0;
    for (parts) |p| total += p.data().length;
    const r = try allocator.alloc(?T, total);
    errdefer allocator.free(r);
    var pos: usize = 0;
    for (parts) |p| {
        const vals = try (common.array_mod.PrimitiveArray(T){ .data = p.data() }).values();
        for (vals, 0..) |v, i| r[pos + i] = if (p.data().isNull(i)) null else v;
        pos += vals.len;
    }
    return r;
}

test "temporal arithmetic hand-written values: timestamps, durations, dates and times (pyarrow 26 results)" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    var ts = try common.makeTyped(i64, ts_ms, allocator, &.{ 0, 1000, null });
    defer ts.release();
    var dur = try common.makeTyped(i64, dur_s, allocator, &.{ 1, 2, 3 });
    defer dur.release();
    var durms = try common.makeTyped(i64, dur_ms, allocator, &.{ 1, 2, 3 });
    defer durms.release();
    const t = Datum{ .array = ts };
    const d = Datum{ .array = dur };
    const dm = Datum{ .array = durms };
    try expectTyped(i64, &ctx, "add", &.{ t, d }, ts_ms, &.{ 1000, 3000, null });
    try expectTyped(i64, &ctx, "add_checked", &.{ d, t }, ts_ms, &.{ 1000, 3000, null });
    try expectTyped(i64, &ctx, "subtract", &.{ t, d }, ts_ms, &.{ -1000, -1000, null });
    try expectTyped(i64, &ctx, "subtract_checked", &.{ t, t }, dur_ms, &.{ 0, 0, null });
    try expectTyped(i64, &ctx, "add", &.{ d, dm }, dur_ms, &.{ 1001, 2002, 3003 });
    try expectTyped(i64, &ctx, "subtract", &.{ d, dm }, dur_ms, &.{ 999, 1998, 2997 });
    try expectTyped(i64, &ctx, "add", &.{ t, dm }, ts_ms, &.{ 1, 1002, null });
    // The timezone of the timestamp is kept; two zoned timestamps subtract.
    var tz = try common.makeTyped(i64, ts_utc, allocator, &.{ 0, 1000, 5 });
    defer tz.release();
    try expectTyped(i64, &ctx, "add", &.{ .{ .array = tz }, d }, ts_utc, &.{ 1000, 3000, 3005 });
    try expectTyped(i64, &ctx, "subtract", &.{ .{ .array = tz }, .{ .array = tz } }, dur_ms, &.{ 0, 0, 0 });

    var d32 = try common.makeTyped(i32, .{ .date32 = {} }, allocator, &.{ 1, 2, null });
    defer d32.release();
    var d64 = try common.makeTyped(i64, .{ .date64 = {} }, allocator, &.{ 86400000, 0, null });
    defer d64.release();
    try expectTyped(i64, &ctx, "add", &.{ .{ .array = d32 }, d }, ts_s, &.{ 86401, 172802, null });
    try expectTyped(i64, &ctx, "add", &.{ d, .{ .array = d32 } }, ts_s, &.{ 86401, 172802, null });
    try expectTyped(i64, &ctx, "add", &.{ .{ .array = d64 }, d }, ts_ms, &.{ 86401000, 2000, null });
    try expectTyped(i64, &ctx, "add", &.{ .{ .array = d32 }, dm }, ts_ms, &.{ 86400001, 172800002, null });
    try expectTyped(i64, &ctx, "subtract", &.{ .{ .array = d32 }, .{ .array = d32 } }, dur_s, &.{ 0, 0, null });
    try expectTyped(i64, &ctx, "subtract", &.{ .{ .array = d64 }, .{ .array = d64 } }, dur_ms, &.{ 0, 0, null });
    try expectTyped(i64, &ctx, "subtract", &.{ .{ .array = d32 }, .{ .array = d64 } }, dur_ms, &.{ 0, 172800000, null });
    try expectTyped(i64, &ctx, "subtract", &.{ .{ .array = d32 }, d }, ts_s, &.{ 86399, 172798, null });

    var t32 = try common.makeTyped(i32, .{ .time32 = .{ .unit = .second } }, allocator, &.{ 1, 2, 3 });
    defer t32.release();
    var t64 = try common.makeTyped(i64, .{ .time64 = .{ .unit = .microsecond } }, allocator, &.{ 1, 2, 3 });
    defer t64.release();
    try expectTyped(i32, &ctx, "add", &.{ .{ .array = t32 }, d }, .{ .time32 = .{ .unit = .second } }, &.{ 2, 4, 6 });
    try expectTyped(i64, &ctx, "add", &.{ .{ .array = t64 }, d }, .{ .time64 = .{ .unit = .microsecond } }, &.{ 1000001, 2000002, 3000003 });
    try expectTyped(i32, &ctx, "subtract_checked", &.{ .{ .array = t32 }, d }, .{ .time32 = .{ .unit = .second } }, &.{ 0, 0, 0 });
    try expectTyped(i64, &ctx, "subtract", &.{ .{ .array = t32 }, .{ .array = t32 } }, dur_s, &.{ 0, 0, 0 });
    try expectTyped(i64, &ctx, "subtract", &.{ .{ .array = t64 }, .{ .array = t64 } }, dur_us, &.{ 0, 0, 0 });

    var n64 = try common.makeArray(i64, allocator, &.{ 2, 3, 4 });
    defer n64.release();
    var n32 = try common.makeArray(i32, allocator, &.{ 2, 3, 4 });
    defer n32.release();
    try expectTyped(i64, &ctx, "multiply", &.{ d, .{ .array = n64 } }, dur_s, &.{ 2, 6, 12 });
    try expectTyped(i64, &ctx, "multiply_checked", &.{ .{ .array = n32 }, d }, dur_s, &.{ 2, 6, 12 });
    try expectTyped(i64, &ctx, "divide", &.{ d, .{ .array = n32 } }, dur_s, &.{ 0, 0, 0 });
    try expectTyped(f64, &ctx, "divide", &.{ d, d }, .{ .double = {} }, &.{ 1.0, 1.0, 1.0 });
    try expectTyped(f64, &ctx, "divide_checked", &.{ d, dm }, .{ .double = {} }, &.{ 1000.0, 1000.0, 1000.0 });
    try expectTyped(i64, &ctx, "negate", &.{d}, dur_s, &.{ -1, -2, -3 });
    try expectTyped(i64, &ctx, "abs_checked", &.{.{ .scalar = common.Scalar.init(dur_s, .{ .duration = -4 }) }}, dur_s, &.{4});
}

test "temporal arithmetic oracle matches a per-row loop on random sliced inputs with nulls, chunked and scalars" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    var prng = std.Random.DefaultPrng.init(0x7E3A);
    const rng = prng.random();
    var raw_a = try common.randomArray(i64, allocator, rng, 300, 4, 20, .{ .small = true });
    defer raw_a.release();
    var raw_b = try common.randomArray(i64, allocator, rng, 300, 9, 20, .{ .small = true });
    defer raw_b.release();
    const xs = try common.collect(i64, allocator, .{ .array = raw_a });
    defer allocator.free(xs);
    const ys = try common.collect(i64, allocator, .{ .array = raw_b });
    defer allocator.free(ys);
    var tsa = try common.makeTyped(i64, ts_s, allocator, xs);
    defer tsa.release();
    var dub = try common.makeTyped(i64, dur_ms, allocator, ys);
    defer dub.release();
    // ts[s] + dur[ms]: the timestamp is scaled by 1000.
    const expected = try allocator.alloc(?i64, xs.len);
    defer allocator.free(expected);
    for (expected, xs, ys) |*e, x, y| e.* = if (x == null or y == null) null else x.? * 1000 + y.?;
    try expectTyped(i64, &ctx, "add_checked", &.{ .{ .array = tsa }, .{ .array = dub } }, ts_ms, expected);
    var ca = try common.chunkedAt(allocator, tsa, &.{ 10, 200 });
    defer ca.release();
    var cb = try common.chunkedAt(allocator, dub, &.{150});
    defer cb.release();
    try expectTyped(i64, &ctx, "add", &.{ .{ .chunked = ca }, .{ .chunked = cb } }, ts_ms, expected);
    for (expected, xs) |*e, x| e.* = if (x == null) null else x.? * 1000 - 7;
    try expectTyped(i64, &ctx, "subtract", &.{ .{ .array = tsa }, .{ .scalar = common.Scalar.init(dur_ms, .{ .duration = 7 }) } }, ts_ms, expected);
    for (expected, xs, ys) |*e, x, y| e.* = if (x == null or y == null) null else x.? * 1000 - y.?;
    var tsb = try common.makeTyped(i64, ts_ms, allocator, ys);
    defer tsb.release();
    try expectTyped(i64, &ctx, "subtract_checked", &.{ .{ .array = tsa }, .{ .array = tsb } }, dur_ms, expected);
    try expectTyped(i64, &ctx, "add", &.{ .{ .scalar = common.Scalar.init(ts_s, .{ .timestamp = 2 }) }, .{ .scalar = common.Scalar.init(dur_ms, .{ .duration = 5 }) } }, ts_ms, &.{2005});
    try expectTyped(i64, &ctx, "add", &.{ .{ .scalar = common.Scalar.init(ts_s, .null) }, .{ .scalar = common.Scalar.init(dur_ms, .{ .duration = 5 }) } }, ts_ms, &.{null});
}

test "temporal arithmetic errors: overflow, unit casts, time range, zoned and naive, zero divisor, types" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    var max = try common.makeTyped(i64, ts_s, allocator, &.{std.math.maxInt(i64)});
    defer max.release();
    var one_s = try common.makeTyped(i64, dur_s, allocator, &.{1});
    defer one_s.release();
    var one_ms = try common.makeTyped(i64, dur_ms, allocator, &.{1});
    defer one_ms.release();
    try std.testing.expectError(error.Overflow, ctx.invokeScalar("add_checked", &.{ .{ .array = max }, .{ .array = one_s } }, none));
    try expectTyped(i64, &ctx, "add", &.{ .{ .array = max }, .{ .array = one_s } }, ts_s, &.{std.math.minInt(i64)});
    var sat = ExecContext.initWithConfig(allocator, &registry, .{ .overflow_mode = .saturating });
    try expectTyped(i64, &sat, "add", &.{ .{ .array = max }, .{ .array = one_s } }, ts_s, &.{std.math.maxInt(i64)});
    // Scaling to the finer unit fails in both variants.
    try std.testing.expectError(error.InvalidCast, ctx.invokeScalar("add", &.{ .{ .array = max }, .{ .array = one_ms } }, none));
    try std.testing.expectError(error.InvalidCast, ctx.invokeScalar("add_checked", &.{ .{ .array = max }, .{ .array = one_ms } }, none));
    // Times must stay within one day.
    var t = try common.makeTyped(i32, .{ .time32 = .{ .unit = .second } }, allocator, &.{ 1, 0 });
    defer t.release();
    var day = try common.makeTyped(i64, dur_s, allocator, &.{ 86400, 1 });
    defer day.release();
    try std.testing.expectError(error.InvalidInput, ctx.invokeScalar("add", &.{ .{ .array = t }, .{ .array = day } }, none));
    try std.testing.expectError(error.InvalidInput, ctx.invokeScalar("subtract_checked", &.{ .{ .array = t }, .{ .array = day } }, none));
    // A null row is never checked.
    var masked = try common.makeTyped(i64, dur_s, allocator, &.{ null, 1 });
    defer masked.release();
    try expectTyped(i32, &ctx, "add", &.{ .{ .array = t }, .{ .array = masked } }, .{ .time32 = .{ .unit = .second } }, &.{ null, 1 });
    // Zoned minus naive is ambiguous.
    var naive = try common.makeTyped(i64, ts_ms, allocator, &.{0});
    defer naive.release();
    var zoned = try common.makeTyped(i64, ts_utc, allocator, &.{0});
    defer zoned.release();
    try std.testing.expectError(error.InvalidInput, ctx.invokeScalar("subtract", &.{ .{ .array = naive }, .{ .array = zoned } }, none));
    var zero = try common.makeArray(i64, allocator, &.{0});
    defer zero.release();
    try std.testing.expectError(error.DivideByZero, ctx.invokeScalar("divide", &.{ .{ .array = one_s }, .{ .array = zero } }, none));
    var big = try common.makeArray(i64, allocator, &.{std.math.maxInt(i64)});
    defer big.release();
    var two = try common.makeTyped(i64, dur_s, allocator, &.{2});
    defer two.release();
    try std.testing.expectError(error.Overflow, ctx.invokeScalar("multiply_checked", &.{ .{ .array = two }, .{ .array = big } }, none));
    // No kernel: timestamp + timestamp, duration + integer, time + time, duration * float.
    try std.testing.expectError(error.NoMatchingKernel, ctx.invokeScalar("add", &.{ .{ .array = naive }, .{ .array = naive } }, none));
    try std.testing.expectError(error.NoMatchingKernel, ctx.invokeScalar("add", &.{ .{ .array = one_s }, .{ .array = zero } }, none));
    try std.testing.expectError(error.NoMatchingKernel, ctx.invokeScalar("add", &.{ .{ .array = t }, .{ .array = t } }, none));
    try std.testing.expectError(error.NoMatchingKernel, ctx.invokeScalar("multiply", &.{ .{ .array = two }, .{ .scalar = common.scalarOf(f64, 2.0) } }, none));
    try std.testing.expectError(error.InvalidOptions, ctx.invokeScalar("add", &.{ .{ .array = naive }, .{ .array = one_s } }, .{ .arithmetic = .{} }));
}
