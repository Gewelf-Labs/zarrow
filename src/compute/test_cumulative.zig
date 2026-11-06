//! cumulative_*, pairwise_diff, winsorize and random. Expected values are pyarrow 26 results.
const std = @import("std");
const common = @import("test_common.zig");

const ArrayRef = common.ArrayRef;
const DataType = common.DataType;
const Datum = common.Datum;
const ExecContext = common.ExecContext;
const Options = common.Options;
const Scalar = common.Scalar;
const none = common.none;

const inf = std.math.inf(f64);
const nan = std.math.nan(f64);

/// Invoke vector function `name`; every array part passes validateFull; the rows (of storage
/// `T`) equal `expected` (NaN equals NaN, -0.0 differs from 0.0).
fn expectVector(comptime T: type, ctx: *ExecContext, name: []const u8, args: []const Datum, options: Options, expected: []const ?T) !void {
    const allocator = std.testing.allocator;
    var out = try ctx.invokeVector(name, args, options);
    defer out.release();
    switch (out) {
        .array => |a| try a.data().validateFull(),
        .chunked => |c| for (c.chunks()) |p| try p.data().validateFull(),
        else => return error.TestUnexpectedResult,
    }
    const rows = try common.collect(T, allocator, out);
    defer allocator.free(rows);
    try std.testing.expectEqual(expected.len, rows.len);
    for (expected, rows, 0..) |e, g, i| {
        const same = if (e == null or g == null) e == null and g == null else if (@typeInfo(T) == .float)
            (std.math.isNan(e.?) and std.math.isNan(g.?)) or (e.? == g.? and std.math.signbit(e.?) == std.math.signbit(g.?))
        else
            e.? == g.?;
        if (!same) {
            std.debug.print("{s} row {d}: expected {any}, got {any}\n", .{ name, i, e, g });
            return error.TestExpectedEqual;
        }
    }
}

fn cum(skip_nulls: bool, start: ?Scalar) Options {
    return .{ .cumulative = .{ .skip_nulls = skip_nulls, .start = start } };
}

test "cumulative_sum, cumulative_prod, cumulative_min, cumulative_max and cumulative_mean: options, nulls, NaN, overflow (pyarrow 26 results)" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);

    var v = try common.makeArray(i64, allocator, &.{ 1, 2, null, 4 });
    defer v.release();
    const dv = [_]Datum{.{ .array = v }};
    const ten: Scalar = common.scalarOf(i64, 10);
    try expectVector(i64, &ctx, "cumulative_sum", &dv, none, &.{ 1, 3, null, null });
    try expectVector(i64, &ctx, "cumulative_sum", &dv, cum(true, null), &.{ 1, 3, null, 7 });
    try expectVector(i64, &ctx, "cumulative_sum", &dv, cum(false, ten), &.{ 11, 13, null, null });
    try expectVector(i64, &ctx, "cumulative_sum", &dv, cum(true, ten), &.{ 11, 13, null, 17 });
    try expectVector(i64, &ctx, "cumulative_prod", &dv, cum(true, null), &.{ 1, 2, null, 8 });
    try expectVector(i64, &ctx, "cumulative_prod", &dv, cum(true, ten), &.{ 10, 20, null, 80 });
    try expectVector(i64, &ctx, "cumulative_min", &dv, cum(true, null), &.{ 1, 1, null, 1 });
    try expectVector(i64, &ctx, "cumulative_max", &dv, cum(true, ten), &.{ 10, 10, null, 10 });
    try expectVector(i64, &ctx, "cumulative_max", &dv, none, &.{ 1, 2, null, null });
    try expectVector(f64, &ctx, "cumulative_mean", &dv, cum(true, ten), &.{ 1.0, 1.5, null, 7.0 / 3.0 });
    try expectVector(i64, &ctx, "cumulative_sum", &dv, cum(true, common.scalarOf(i64, null)), &.{ 1, 3, null, 7 });
    // The start is cast to the input type: an int8 start on doubles, 1000 on int8 fails.
    var d = try common.makeArray(f64, allocator, &.{1.5});
    defer d.release();
    try expectVector(f64, &ctx, "cumulative_sum", &.{.{ .array = d }}, cum(false, common.scalarOf(i8, 1)), &.{2.5});
    var small = try common.makeArray(i8, allocator, &.{ 100, 100 });
    defer small.release();
    const ds = [_]Datum{.{ .array = small }};
    try std.testing.expectError(error.InvalidCast, ctx.invokeVector("cumulative_sum", &ds, cum(false, common.scalarOf(i64, 1000))));
    try std.testing.expectError(error.InvalidCast, ctx.invokeVector("cumulative_sum", &ds, cum(false, common.scalarOf(f64, 1.5))));
    try std.testing.expectError(error.InvalidOptions, ctx.invokeVector("cumulative_sum", &ds, cum(false, Scalar.init(.{ .string = {} }, .{ .string = "1" }))));
    // Integer overflow wraps (unchecked), fails (checked) or saturates (overflow_mode).
    try expectVector(i8, &ctx, "cumulative_sum", &ds, none, &.{ 100, -56 });
    try expectVector(i8, &ctx, "cumulative_prod", &ds, none, &.{ 100, 16 });
    try std.testing.expectError(error.Overflow, ctx.invokeVector("cumulative_sum_checked", &ds, none));
    try std.testing.expectError(error.Overflow, ctx.invokeVector("cumulative_prod_checked", &ds, none));
    var u = try common.makeArray(u8, allocator, &.{ 200, 100 });
    defer u.release();
    try expectVector(u8, &ctx, "cumulative_sum", &.{.{ .array = u }}, none, &.{ 200, 44 });
    try expectVector(f64, &ctx, "cumulative_mean", &.{.{ .array = u }}, none, &.{ 200, 150 });
    try std.testing.expectError(error.Overflow, ctx.invokeVector("cumulative_sum_checked", &.{.{ .array = u }}, none));
    var sat = ExecContext.initWithConfig(allocator, &registry, .{ .overflow_mode = .saturating });
    try expectVector(i8, &sat, "cumulative_sum", &ds, none, &.{ 100, 127 });
    // Floats: NaN propagates through sum / prod / mean; min / max skip it (identity first);
    // the later of equal values wins (-0.0 then 0.0).
    var f = try common.makeArray(f64, allocator, &.{ 1.5, nan, null, -1.0 });
    defer f.release();
    const df = [_]Datum{.{ .array = f }};
    try expectVector(f64, &ctx, "cumulative_sum", &df, cum(true, null), &.{ 1.5, nan, null, nan });
    try expectVector(f64, &ctx, "cumulative_min", &df, cum(true, null), &.{ 1.5, 1.5, null, -1.0 });
    try expectVector(f64, &ctx, "cumulative_max", &df, cum(true, null), &.{ 1.5, 1.5, null, 1.5 });
    var lead = try common.makeArray(f64, allocator, &.{ nan, 1.0, nan, 0.5 });
    defer lead.release();
    try expectVector(f64, &ctx, "cumulative_min", &.{.{ .array = lead }}, none, &.{ inf, 1.0, 1.0, 0.5 });
    try expectVector(f64, &ctx, "cumulative_max", &.{.{ .array = lead }}, none, &.{ -inf, 1.0, 1.0, 1.0 });
    try expectVector(f64, &ctx, "cumulative_min", &.{.{ .array = lead }}, cum(false, common.scalarOf(f64, nan)), &.{ nan, 1.0, 1.0, 0.5 });
    var zeros = try common.makeArray(f64, allocator, &.{ -0.0, 0.0 });
    defer zeros.release();
    try expectVector(f64, &ctx, "cumulative_max", &.{.{ .array = zeros }}, none, &.{ -0.0, 0.0 });
    var zeros2 = try common.makeArray(f64, allocator, &.{ 0.0, -0.0 });
    defer zeros2.release();
    try expectVector(f64, &ctx, "cumulative_min", &.{.{ .array = zeros2 }}, none, &.{ 0.0, -0.0 });
    var f32s = try common.makeArray(f32, allocator, &.{ 1.5, 2 });
    defer f32s.release();
    try expectVector(f64, &ctx, "cumulative_mean", &.{.{ .array = f32s }}, none, &.{ 1.5, 1.75 });
    try expectVector(f32, &ctx, "cumulative_prod", &.{.{ .array = f32s }}, none, &.{ 1.5, 3.0 });
    // Chunked (state carried across chunks, chunks kept), sliced, scalar; a leading null.
    var long = try common.makeArray(i32, allocator, &.{ 5, 1, null, 4, 2, 9, null, 3 });
    defer long.release();
    var chunked = try common.chunkedAt(allocator, long, &.{ 2, 3, 6 });
    defer chunked.release();
    var co = try ctx.invokeVector("cumulative_sum", &.{.{ .chunked = chunked }}, cum(true, null));
    defer co.release();
    try std.testing.expectEqual(@as(usize, 4), co.chunked.numChunks());
    try expectVector(i32, &ctx, "cumulative_sum", &.{.{ .chunked = chunked }}, cum(true, null), &.{ 5, 6, null, 10, 12, 21, null, 24 });
    try expectVector(i32, &ctx, "cumulative_max", &.{.{ .chunked = chunked }}, none, &.{ 5, 5, null, null, null, null, null, null });
    var mid = try long.slice(3, 4);
    defer mid.release();
    try expectVector(i32, &ctx, "cumulative_min", &.{.{ .array = mid }}, cum(true, null), &.{ 4, 2, 2, null });
    try expectVector(i64, &ctx, "cumulative_sum", &.{.{ .scalar = common.scalarOf(i64, 3) }}, none, &.{3});
    var lnull = try common.makeArray(i64, allocator, &.{ null, 1 });
    defer lnull.release();
    try expectVector(i64, &ctx, "cumulative_sum", &.{.{ .array = lnull }}, cum(true, common.scalarOf(i64, 5)), &.{ null, 6 });
    // No kernel for other types (pyarrow 26).
    var s = try common.makeStrings(.string, allocator, &.{"a"});
    defer s.release();
    try std.testing.expectError(error.NoMatchingKernel, ctx.invokeVector("cumulative_sum", &.{.{ .array = s }}, none));
    try std.testing.expectError(error.InvalidOptions, ctx.invokeVector("cumulative_sum", &dv, .{ .pairwise = .{} }));
}

test "pairwise_diff and pairwise_diff_checked: periods, overflow, temporal and decimal types, chunked (pyarrow 26 results)" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);

    var v = try common.makeArray(i64, allocator, &.{ 1, 4, null, 10, 20 });
    defer v.release();
    const dv = [_]Datum{.{ .array = v }};
    const cases = [_]struct { i64, [5]?i64 }{
        .{ 1, .{ null, 3, null, null, 10 } },
        .{ 2, .{ null, null, null, 6, null } },
        .{ -1, .{ -3, null, null, -10, null } },
        .{ 0, .{ 0, 0, null, 0, 0 } },
        .{ 5, .{ null, null, null, null, null } },
        .{ -6, .{ null, null, null, null, null } },
        .{ std.math.minInt(i64), .{ null, null, null, null, null } },
    };
    for (cases) |c| try expectVector(i64, &ctx, "pairwise_diff", &dv, .{ .pairwise = .{ .period = c[0] } }, &c[1]);
    try expectVector(i64, &ctx, "pairwise_diff", &dv, none, &.{ null, 3, null, null, 10 });
    var u = try common.makeArray(u8, allocator, &.{ 4, 1 });
    defer u.release();
    try expectVector(u8, &ctx, "pairwise_diff", &.{.{ .array = u }}, none, &.{ null, 253 });
    try std.testing.expectError(error.Overflow, ctx.invokeVector("pairwise_diff_checked", &.{.{ .array = u }}, none));
    var i8s = try common.makeArray(i8, allocator, &.{ -100, 100 });
    defer i8s.release();
    try expectVector(i8, &ctx, "pairwise_diff", &.{.{ .array = i8s }}, none, &.{ null, -56 });
    try std.testing.expectError(error.Overflow, ctx.invokeVector("pairwise_diff_checked", &.{.{ .array = i8s }}, none));
    var fl = try common.makeArray(f64, allocator, &.{ 1.5, 4 });
    defer fl.release();
    try expectVector(f64, &ctx, "pairwise_diff", &.{.{ .array = fl }}, none, &.{ null, 2.5 });
    // Temporal types give durations, decimal128(5, 2) gives decimal128(6, 2).
    const ts_ms = DataType{ .timestamp = .{ .unit = .millisecond, .timezone = null } };
    var ts = try common.makeTyped(i64, ts_ms, allocator, &.{ 1, 4 });
    defer ts.release();
    var d32 = try common.makeTyped(i32, .{ .date32 = {} }, allocator, &.{ 1, 4 });
    defer d32.release();
    var t32 = try common.makeTyped(i32, .{ .time32 = .{ .unit = .second } }, allocator, &.{ 1, 4 });
    defer t32.release();
    var dec = try common.makeTyped(i128, .{ .decimal128 = .{ .precision = 5, .scale = 2 } }, allocator, &.{ 150, 225 });
    defer dec.release();
    const want_types = [_]DataType{ .{ .duration = .{ .unit = .millisecond } }, .{ .duration = .{ .unit = .second } }, .{ .duration = .{ .unit = .second } }, .{ .decimal128 = .{ .precision = 6, .scale = 2 } } };
    const want_vals = [_]i128{ 3, 3 * 86400, 3, 75 };
    for ([_]ArrayRef{ ts, d32, t32, dec }, want_types, want_vals) |a, wt, wv| {
        var out = try ctx.invokeVector("pairwise_diff", &.{.{ .array = a }}, none);
        defer out.release();
        try out.array.data().validateFull();
        try std.testing.expect(out.array.data().data_type.eql(wt));
        try std.testing.expect(out.array.data().isNull(0));
        const got: i128 = if (wt == .decimal128) (try (common.array_mod.PrimitiveArray(i128){ .data = out.array.data() }).values())[1] else (try (common.array_mod.PrimitiveArray(i64){ .data = out.array.data() }).values())[1];
        try std.testing.expectEqual(wv, got);
    }
    // Chunked input (concatenated: Arrow rejects it), sliced input, a scalar.
    var vc = try common.chunkedAt(allocator, v, &.{ 1, 3 });
    defer vc.release();
    try expectVector(i64, &ctx, "pairwise_diff", &.{.{ .chunked = vc }}, none, &.{ null, 3, null, null, 10 });
    var tail = try v.slice(2, 3);
    defer tail.release();
    try expectVector(i64, &ctx, "pairwise_diff", &.{.{ .array = tail }}, .{ .pairwise = .{ .period = -1 } }, &.{ null, -10, null });
    try expectVector(i64, &ctx, "pairwise_diff", &.{.{ .scalar = common.scalarOf(i64, 1) }}, none, &.{null});
    var s = try common.makeStrings(.string, allocator, &.{"a"});
    defer s.release();
    try std.testing.expectError(error.NoMatchingKernel, ctx.invokeVector("pairwise_diff", &.{.{ .array = s }}, none));
}

test "winsorize: nearest quantile limits, NaN and nulls, decimals, chunked, options (pyarrow 26 results)" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);

    var w = try common.makeArray(i64, allocator, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, null });
    defer w.release();
    const dw = [_]Datum{.{ .array = w }};
    try expectVector(i64, &ctx, "winsorize", &dw, .{ .winsorize = .{ .lower_limit = 0.1, .upper_limit = 0.9 } }, &.{ 2, 2, 3, 4, 5, 6, 7, 8, 9, 9, null });
    try expectVector(i64, &ctx, "winsorize", &dw, .{ .winsorize = .{ .lower_limit = 0.15, .upper_limit = 0.8 } }, &.{ 2, 2, 3, 4, 5, 6, 7, 8, 8, 8, null });
    try expectVector(i64, &ctx, "winsorize", &dw, .{ .winsorize = .{ .lower_limit = 0, .upper_limit = 1 } }, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, null });
    var t = try common.makeArray(i64, allocator, &.{ 1, 2, 3, 4 });
    defer t.release();
    try expectVector(i64, &ctx, "winsorize", &.{.{ .array = t }}, .{ .winsorize = .{ .lower_limit = 0.5, .upper_limit = 0.5 } }, &.{ 3, 3, 3, 3 });
    var f = try common.makeArray(f64, allocator, &.{ 1.5, nan, 3, null, -2 });
    defer f.release();
    try expectVector(f64, &ctx, "winsorize", &.{.{ .array = f }}, .{ .winsorize = .{ .lower_limit = 0.2, .upper_limit = 0.8 } }, &.{ 1.5, nan, 3, null, -2 });
    try expectVector(f64, &ctx, "winsorize", &.{.{ .array = f }}, .{ .winsorize = .{ .lower_limit = 0.5, .upper_limit = 0.5 } }, &.{ 1.5, nan, 1.5, null, 1.5 });
    var dec = try common.makeTyped(i128, .{ .decimal128 = .{ .precision = 5, .scale = 2 } }, allocator, &.{ 150, 225, 900 });
    defer dec.release();
    try expectVector(i128, &ctx, "winsorize", &.{.{ .array = dec }}, .{ .winsorize = .{ .lower_limit = 0.1, .upper_limit = 0.5 } }, &.{ 150, 225, 225 });
    var wc = try common.chunkedAt(allocator, w, &.{3});
    defer wc.release();
    var cout = try ctx.invokeVector("winsorize", &.{.{ .chunked = wc }}, .{ .winsorize = .{ .lower_limit = 0.1, .upper_limit = 0.9 } });
    defer cout.release();
    try std.testing.expectEqual(@as(usize, 2), cout.chunked.numChunks());
    var nulls = try common.makeArray(i64, allocator, &.{ null, null });
    defer nulls.release();
    try expectVector(i64, &ctx, "winsorize", &.{.{ .array = nulls }}, .{ .winsorize = .{ .lower_limit = 0.1, .upper_limit = 0.9 } }, &.{ null, null });
    try std.testing.expectError(error.InvalidOptions, ctx.invokeVector("winsorize", &dw, .{ .winsorize = .{ .lower_limit = 0.9, .upper_limit = 0.1 } }));
    try std.testing.expectError(error.InvalidOptions, ctx.invokeVector("winsorize", &dw, .{ .winsorize = .{ .lower_limit = -0.1, .upper_limit = 1 } }));
    try std.testing.expectError(error.InvalidOptions, ctx.invokeVector("winsorize", &dw, .{ .winsorize = .{ .lower_limit = nan, .upper_limit = 1 } }));
    try std.testing.expectError(error.InvalidOptions, ctx.invokeVector("winsorize", &dw, none));
}

test "random: seeded values in [0, 1), length, system initializer" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    var a = try ctx.invokeScalar("random", &.{}, .{ .random = .{ .length = 100, .seed = 42 } });
    defer a.release();
    var b = try ctx.invokeScalar("random", &.{}, .{ .random = .{ .length = 100, .seed = 42 } });
    defer b.release();
    try a.array.data().validateFull();
    const xs = try (common.array_mod.PrimitiveArray(f64){ .data = a.array.data() }).values();
    const ys = try (common.array_mod.PrimitiveArray(f64){ .data = b.array.data() }).values();
    try std.testing.expectEqual(@as(usize, 100), xs.len);
    try std.testing.expectEqualSlices(f64, xs, ys);
    for (xs) |x| try std.testing.expect(x >= 0 and x < 1);
    try std.testing.expectError(error.NotImplemented, ctx.invokeScalar("random", &.{}, .{ .random = .{ .length = 1, .initializer = .system } }));
    try std.testing.expectError(error.InvalidOptions, ctx.invokeScalar("random", &.{}, none));
}
