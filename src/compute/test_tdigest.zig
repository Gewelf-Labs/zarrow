//! tdigest, approximate_median, hash_tdigest and hash_approximate_median. Expected values are
//! pyarrow 26 results; the 5000-row data comes from the same LCG in Python and here.
const std = @import("std");
const common = @import("test_common.zig");
const hash = @import("hash_aggregate.zig");

const ArrayRef = common.ArrayRef;
const DataType = common.DataType;
const Datum = common.Datum;
const ExecContext = common.ExecContext;
const Options = common.Options;
const none = common.none;

const nan = std.math.nan(f64);

/// Rows of an aggregate result: a float64 array, or a float64 scalar as one row.
fn rowsOf(d: Datum) ![]?f64 {
    if (d == .array) try d.array.data().validateFull();
    return common.collect(f64, std.testing.allocator, d);
}

/// Every expected row within `rel` relative error (nulls exactly).
fn expectClose(expected: []const ?f64, got: []const ?f64, rel: f64) !void {
    try std.testing.expectEqual(expected.len, got.len);
    for (expected, got, 0..) |e, g, i| {
        if (e == null or g == null) {
            if (e != null or g != null) {
                std.debug.print("row {d}: expected {?d}, got {?d}\n", .{ i, e, g });
                return error.TestExpectedEqual;
            }
            continue;
        }
        if (std.math.isNan(e.?)) {
            try std.testing.expect(std.math.isNan(g.?));
            continue;
        }
        std.testing.expectApproxEqRel(e.?, g.?, rel) catch |err| {
            std.debug.print("row {d}: expected {d}, got {d}\n", .{ i, e.?, g.? });
            return err;
        };
    }
}

fn expectAggregate(ctx: *ExecContext, name: []const u8, args: []const Datum, options: Options, expected: []const ?f64) !void {
    var out = try ctx.invokeAggregate(name, args, options);
    defer out.release();
    const got = try rowsOf(out);
    defer std.testing.allocator.free(got);
    try expectClose(expected, got, 1e-12);
}

/// 5000 float64 rows: every 17th null, every 29th (not 17th) NaN, else uniform in [-500, 500).
fn lcgRows(allocator: std.mem.Allocator) ![]?f64 {
    const out = try allocator.alloc(?f64, 5000);
    var x: u64 = 42;
    for (out, 0..) |*o, i| {
        x = x *% 6364136223846793005 +% 1442695040888963407;
        const v = @as(f64, @floatFromInt(x >> 11)) * std.math.pow(f64, 2, -53) * 1000.0 - 500.0;
        o.* = if (i % 17 == 0) null else if (i % 29 == 0) nan else v;
    }
    return out;
}

test "tdigest and approximate_median hand-written values, options and types (pyarrow 26 results)" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    var a = try common.makeArray(f64, allocator, &.{ 1, 2, 3, 4, null, 5 });
    defer a.release();
    const d: Datum = .{ .array = a };
    try expectAggregate(&ctx, "tdigest", &.{d}, none, &.{3.0});
    try expectAggregate(&ctx, "tdigest", &.{d}, .{ .tdigest = .{ .q = &.{ 0, 0.25, 0.5, 1 } } }, &.{ 1, 2, 3, 5 });
    try expectAggregate(&ctx, "tdigest", &.{d}, .{ .tdigest = .{ .skip_nulls = false } }, &.{null});
    try expectAggregate(&ctx, "tdigest", &.{d}, .{ .tdigest = .{ .min_count = 6 } }, &.{null});
    // delta below 10 and buffer_size 0 work (Arrow uses 10 and 1).
    try expectAggregate(&ctx, "tdigest", &.{d}, .{ .tdigest = .{ .delta = 0, .buffer_size = 0 } }, &.{3.0});
    try expectAggregate(&ctx, "approximate_median", &.{d}, none, &.{3.0});
    try expectAggregate(&ctx, "approximate_median", &.{d}, .{ .scalar_aggregate = .{ .skip_nulls = false } }, &.{null});
    var empty = try common.makeArray(f64, allocator, &.{});
    defer empty.release();
    try expectAggregate(&ctx, "tdigest", &.{.{ .array = empty }}, none, &.{null});
    try expectAggregate(&ctx, "approximate_median", &.{.{ .array = empty }}, none, &.{null});
    // NaN is skipped but counts as a valid row.
    var with_nan = try common.makeArray(f64, allocator, &.{ nan, 1.0 });
    defer with_nan.release();
    try expectAggregate(&ctx, "tdigest", &.{.{ .array = with_nan }}, none, &.{1.0});
    var only_nan = try common.makeArray(f64, allocator, &.{nan});
    defer only_nan.release();
    try expectAggregate(&ctx, "tdigest", &.{.{ .array = only_nan }}, none, &.{null});
    try expectAggregate(&ctx, "tdigest", &.{.{ .array = only_nan }}, .{ .tdigest = .{ .min_count = 1 } }, &.{null});
    var ints = try common.makeArray(i8, allocator, &.{ 1, 2, 3 });
    defer ints.release();
    try expectAggregate(&ctx, "tdigest", &.{.{ .array = ints }}, none, &.{2.0});
    var dec = try common.makeTyped(i128, .{ .decimal128 = .{ .precision = 5, .scale = 2 } }, allocator, &.{ 100, 200, 300 });
    defer dec.release();
    try expectAggregate(&ctx, "tdigest", &.{.{ .array = dec }}, none, &.{2.0});
    try expectAggregate(&ctx, "tdigest", &.{.{ .scalar = common.scalarOf(f64, 3.0) }}, none, &.{3.0});
    // q outside [0, 1] or NaN gives NaN (pyarrow 26: tdigest(q=[-0.1, 0.5]) -> [nan, ...];
    // q=[nan, 2] -> [nan, nan]); no value still gives null.
    try expectAggregate(&ctx, "tdigest", &.{d}, .{ .tdigest = .{ .q = &.{ -0.1, 0.5, 1.5, nan } } }, &.{ nan, 3.0, nan, nan });
    try expectAggregate(&ctx, "tdigest", &.{.{ .array = empty }}, .{ .tdigest = .{ .q = &.{2} } }, &.{null});
    // Oversized options are rejected.
    try std.testing.expectError(error.InvalidOptions, ctx.invokeAggregate("tdigest", &.{d}, .{ .tdigest = .{ .delta = 1 << 21 } }));
    try std.testing.expectError(error.InvalidOptions, ctx.invokeAggregate("tdigest", &.{d}, .{ .tdigest = .{ .buffer_size = 1 << 25 } }));
    var bools = try common.makeArray(bool, allocator, &.{true});
    defer bools.release();
    try std.testing.expectError(error.NoMatchingKernel, ctx.invokeAggregate("tdigest", &.{.{ .array = bools }}, none));
}

test "tdigest on 5000 rows with nulls and NaN: array, chunked (merged per chunk) and small delta follow pyarrow 26" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    const rows = try lcgRows(allocator);
    defer allocator.free(rows);
    var a = try common.makeArray(f64, allocator, rows);
    defer a.release();
    var chunked = try common.chunkedAt(allocator, a, &.{ 1500, 1501 });
    defer chunked.release();
    const qs = [_]f64{ 0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999 };
    const q: Options = .{ .tdigest = .{ .q = &qs } };
    const cases = [_]struct { arg: Datum, options: Options, want: []const ?f64 }{
        .{ .arg = .{ .array = a }, .options = q, .want = &.{ -498.81027309949457, -489.6893524065866, -401.6536412303856, -245.03552092525433, -2.1712703000838154, 243.1528805022142, 399.5963828074915, 488.800903273284, 498.35767732658627 } },
        .{ .arg = .{ .chunked = chunked }, .options = q, .want = &.{ -498.7536610643192, -489.372658938864, -400.81878211445195, -244.70356192574786, -2.49376623778911, 243.60348383385218, 400.5379589022455, 488.68006862727225, 498.08196697469975 } },
        .{ .arg = .{ .array = a }, .options = .{ .tdigest = .{ .q = &qs, .delta = 20, .buffer_size = 37 } }, .want = &.{ -497.09175141232856, -482.7642198120109, -386.8831500399914, -243.90187982840595, -1.0452058347741655, 241.48465080269904, 393.6822959265317, 487.0322317247009, 498.11618703638055 } },
    };
    for (cases) |c| {
        var out = try ctx.invokeAggregate("tdigest", &.{c.arg}, c.options);
        defer out.release();
        const got = try rowsOf(out);
        defer allocator.free(got);
        try expectClose(c.want, got, 1e-9);
    }
    var median = try ctx.invokeAggregate("approximate_median", &.{.{ .chunked = chunked }}, none);
    defer median.release();
    try std.testing.expectApproxEqRel(-2.49376623778911, median.scalar.value.f64, 1e-9);

    // The update / merge lifecycle: one session per chunk merged equals the chunked call.
    var whole = try ctx.beginAggregate("tdigest", &.{.{ .array = a }}, q);
    defer whole.deinit();
    var part = try ctx.beginAggregate("tdigest", &.{.{ .array = a }}, q);
    defer part.deinit();
    try whole.update(&.{.{ .array = chunked.chunks()[0] }});
    try whole.update(&.{.{ .array = chunked.chunks()[1] }});
    try part.update(&.{.{ .array = chunked.chunks()[2] }});
    try whole.merge(&part);
    var merged = try whole.finalize();
    defer merged.release();
    const got = try rowsOf(merged);
    defer allocator.free(got);
    try expectClose(cases[1].want, got, 1e-9);
}

test "hash_tdigest and hash_approximate_median per group (pyarrow 26 group_by, use_threads=False)" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    var keys = try common.makeArray(i64, allocator, &.{ 1, 1, 2, 2, 2, 3 });
    defer keys.release();
    var v = try common.makeArray(f64, allocator, &.{ 1, 2, 3, null, 5, null });
    defer v.release();
    var by = try hash.groupBy(&ctx, &.{.{ .name = "k", .array = keys }}, &.{
        .{ .name = "td", .function = "hash_tdigest", .input = v },
        .{ .name = "td2", .function = "hash_tdigest", .input = v, .options = .{ .tdigest = .{ .q = &.{ 0.1, 0.9 }, .skip_nulls = false } } },
        .{ .name = "med", .function = "hash_approximate_median", .input = v },
    });
    defer by.deinit();
    // fixed_size_list<double>[1]: [[1.0], [3.0], [None]] (the list valid, its item null).
    const td = by.columns[1].data();
    try by.columns[1].data().validateFull();
    try std.testing.expect(td.data_type == .fixed_size_list and td.data_type.fixed_size_list.list_size == 1 and !td.hasNulls());
    const items = try rowsOf(.{ .array = td.children[0] });
    defer allocator.free(items);
    try expectClose(&.{ 1.0, 3.0, null }, items, 1e-12);
    const items2 = try rowsOf(.{ .array = by.columns[2].data().children[0] });
    defer allocator.free(items2);
    try expectClose(&.{ 1.0, 2.0, null, null, null, null }, items2, 1e-12);
    const med = try rowsOf(.{ .array = by.columns[3] });
    defer allocator.free(med);
    try expectClose(&.{ 1.0, 3.0, null }, med, 1e-12);

    // 5000 rows in three groups (row % 3).
    const rows = try lcgRows(allocator);
    defer allocator.free(rows);
    var big = try common.makeArray(f64, allocator, rows);
    defer big.release();
    const ks = try allocator.alloc(?i64, rows.len);
    defer allocator.free(ks);
    for (ks, 0..) |*k, i| k.* = @intCast(i % 3);
    var big_keys = try common.makeArray(i64, allocator, ks);
    defer big_keys.release();
    var big_by = try hash.groupBy(&ctx, &.{.{ .name = "k", .array = big_keys }}, &.{
        .{ .name = "td", .function = "hash_tdigest", .input = big, .options = .{ .tdigest = .{ .q = &.{ 0.1, 0.5, 0.9 } } } },
        .{ .name = "med", .function = "hash_approximate_median", .input = big },
    });
    defer big_by.deinit();
    const quantiles = try rowsOf(.{ .array = big_by.columns[1].data().children[0] });
    defer allocator.free(quantiles);
    try expectClose(&.{ -398.844165700193, -0.27444398285586735, 405.2396468712288, -403.3786977043621, 5.07235258371462, 393.93858347254866, -401.19633240961474, -4.091138756330174, 401.70492241903366 }, quantiles, 1e-9);
    const medians = try rowsOf(.{ .array = big_by.columns[2] });
    defer allocator.free(medians);
    try expectClose(&.{ -0.27444398285586735, 5.07235258371462, -4.091138756330174 }, medians, 1e-9);
    var ids = try common.makeArray(u32, allocator, &.{ 0, 0, 1, 1, 1, 2 });
    defer ids.release();
    // q outside [0, 1]: NaN in every group (pyarrow 26 group_by tdigest(q=2) -> [[nan], [nan]]).
    var out_q = try ctx.invoke("hash_tdigest", .hash_aggregate, &.{ .{ .array = v }, .{ .array = ids } }, .{ .tdigest = .{ .q = &.{-0.5} } });
    defer out_q.release();
    const nan_items = try rowsOf(.{ .array = out_q.array.data().children[0] });
    defer allocator.free(nan_items);
    try expectClose(&.{ nan, nan, null }, nan_items, 1e-12);
    try std.testing.expectError(error.InvalidOptions, ctx.invoke("hash_tdigest", .hash_aggregate, &.{ .{ .array = v }, .{ .array = ids } }, .{ .tdigest = .{ .delta = 1 << 21 } }));
}
