//! product, variance, stddev, skew, kurtosis, first, last, first_last, count_distinct, index,
//! mode, quantile and their hash_ forms. Expected values are pyarrow 26 results.
const std = @import("std");
const common = @import("test_common.zig");
const hash = @import("hash_aggregate.zig");

const ArrayRef = common.ArrayRef;
const DataType = common.DataType;
const Datum = common.Datum;
const ExecContext = common.ExecContext;
const Options = common.Options;
const Scalar = common.Scalar;
const none = common.none;

const nan = std.math.nan(f64);

fn agg(ctx: *ExecContext, name: []const u8, arg: Datum, options: Options) !Datum {
    return ctx.invokeAggregate(name, &.{arg}, options);
}

/// The f64 result of aggregate `name` within a relative 1e-12 of `expected` (null: a null
/// result; NaN equals NaN).
fn expectF64(ctx: *ExecContext, name: []const u8, arg: Datum, options: Options, expected: ?f64) !void {
    var out = try agg(ctx, name, arg, options);
    defer out.release();
    const s = out.scalar;
    try std.testing.expect(s.data_type == .double);
    if (expected == null or s.isNull()) {
        if (!(expected == null and s.isNull())) {
            std.debug.print("{s}: expected {any}, got {any}\n", .{ name, expected, s.value });
            return error.TestExpectedEqual;
        }
        return;
    }
    try expectNear(expected.?, s.value.f64);
}

fn expectNear(e: f64, g: f64) !void {
    if (std.math.isNan(e) and std.math.isNan(g)) return;
    if (!std.math.approxEqRel(f64, e, g, 1e-12) and !std.math.approxEqAbs(f64, e, g, 1e-12)) {
        std.debug.print("expected {d}, got {d}\n", .{ e, g });
        return error.TestExpectedEqual;
    }
}

fn expectI64(ctx: *ExecContext, name: []const u8, arg: Datum, options: Options, expected: ?i64) !void {
    var out = try agg(ctx, name, arg, options);
    defer out.release();
    if (expected) |e| try std.testing.expectEqual(e, out.scalar.value.i64) else try std.testing.expect(out.scalar.isNull());
}

fn variance(ddof: i32, skip_nulls: bool, min_count: u32) Options {
    return .{ .variance = .{ .ddof = ddof, .skip_nulls = skip_nulls, .min_count = min_count } };
}

fn scalarAgg(skip_nulls: bool, min_count: u32) Options {
    return .{ .scalar_aggregate = .{ .skip_nulls = skip_nulls, .min_count = min_count } };
}

test "product, variance, stddev, skew and kurtosis hand-written values and options (pyarrow 26 results)" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    var a = try common.makeArray(i32, allocator, &.{ 3, 1, null, 3, 2, null, 1 });
    defer a.release();
    const da: Datum = .{ .array = a };
    try expectI64(&ctx, "product", da, none, 18);
    try expectI64(&ctx, "product", da, scalarAgg(false, 1), null);
    var empty = try common.makeArray(i32, allocator, &.{});
    defer empty.release();
    try expectI64(&ctx, "product", .{ .array = empty }, none, null);
    try expectI64(&ctx, "product", .{ .array = empty }, scalarAgg(true, 0), 1);
    var big = try common.makeArray(i64, allocator, &.{ 1 << 40, 1 << 40, 0 });
    defer big.release();
    var big2 = try big.slice(0, 2);
    defer big2.release();
    var wrap = ExecContext.initWithConfig(allocator, &registry, .{ .overflow_mode = .wrapping });
    try expectI64(&wrap, "product", .{ .array = big2 }, none, 0);
    try std.testing.expectError(error.Overflow, agg(&ctx, "product", .{ .array = big2 }, none));
    // A zero factor makes the product exact again.
    try expectI64(&ctx, "product", .{ .array = big }, none, 0);
    var sat = ExecContext.initWithConfig(allocator, &registry, .{ .overflow_mode = .saturating });
    try expectI64(&sat, "product", .{ .array = big2 }, none, std.math.maxInt(i64));
    var u = try common.makeArray(u8, allocator, &.{ 2, 3 });
    defer u.release();
    var pu = try agg(&ctx, "product", .{ .array = u }, none);
    defer pu.release();
    try std.testing.expectEqual(@as(u64, 6), pu.scalar.value.u64);
    var f32s = try common.makeArray(f32, allocator, &.{ 1.5, null, 4.0 });
    defer f32s.release();
    try expectF64(&ctx, "product", .{ .array = f32s }, none, 6.0);

    try expectF64(&ctx, "variance", da, none, 0.8);
    try expectF64(&ctx, "variance", da, variance(1, true, 0), 1.0);
    try expectF64(&ctx, "stddev", da, variance(1, true, 0), 1.0);
    try expectF64(&ctx, "variance", da, variance(0, false, 0), null);
    try expectF64(&ctx, "variance", da, variance(0, true, 10), null);
    try expectF64(&ctx, "skew", da, none, 0.0);
    try expectF64(&ctx, "kurtosis", da, none, -1.75);
    try expectF64(&ctx, "skew", da, .{ .skew = .{ .biased = false } }, 0.0);
    try expectF64(&ctx, "kurtosis", da, .{ .skew = .{ .biased = false } }, -3.0);
    var one = try common.makeArray(i64, allocator, &.{1});
    defer one.release();
    try expectF64(&ctx, "variance", .{ .array = one }, variance(1, true, 0), null);
    var twelve = try common.makeArray(i64, allocator, &.{ 1, 2 });
    defer twelve.release();
    try expectF64(&ctx, "variance", .{ .array = twelve }, variance(2, true, 0), null);
    try expectF64(&ctx, "skew", .{ .array = twelve }, .{ .skew = .{ .biased = false } }, null);
    var ones = try common.makeArray(i64, allocator, &.{ 1, 1, 1 });
    defer ones.release();
    try expectF64(&ctx, "skew", .{ .array = ones }, none, nan);
    try expectF64(&ctx, "kurtosis", .{ .array = ones }, none, nan);
    try expectF64(&ctx, "kurtosis", .{ .array = ones }, .{ .skew = .{ .biased = false } }, null);
    var big_mean = try common.makeArray(f64, allocator, &.{ 1e9 + 1, 1e9 + 2, 1e9 + 3 });
    defer big_mean.release();
    try expectF64(&ctx, "variance", .{ .array = big_mean }, none, 0.6666666666666666);
    var f = try common.makeArray(f64, allocator, &.{ 1.5, 2.0, null, 4.0, -1.0, 2.5, 3.0, 8.0 });
    defer f.release();
    const df: Datum = .{ .array = f };
    try expectF64(&ctx, "variance", df, none, 6.479591836734692);
    try expectF64(&ctx, "stddev", df, none, 2.5455042401722086);
    try expectF64(&ctx, "skew", df, none, 0.6665625933692185);
    try expectF64(&ctx, "kurtosis", df, none, 0.15263066526133162);
    try expectF64(&ctx, "variance", df, variance(1, true, 0), 7.5595238095238075);
    try expectF64(&ctx, "stddev", df, variance(1, true, 0), 2.749458821208968);
    try expectF64(&ctx, "skew", df, .{ .skew = .{ .biased = false } }, 0.8639638653768367);
    try expectF64(&ctx, "kurtosis", df, .{ .skew = .{ .biased = false } }, 2.1663135966271954);
    var with_nan = try common.makeArray(f64, allocator, &.{ 1.0, nan, 2.0 });
    defer with_nan.release();
    try expectF64(&ctx, "variance", .{ .array = with_nan }, none, nan);
    const d52 = DataType{ .decimal128 = .{ .precision = 5, .scale = 2 } };
    var d = try common.makeTyped(i128, d52, allocator, &.{ 100, 200 });
    defer d.release();
    try expectF64(&ctx, "variance", .{ .array = d }, none, 0.25);
    var d3 = try common.makeTyped(i128, d52, allocator, &.{ 100, 200, 400 });
    defer d3.release();
    try expectF64(&ctx, "skew", .{ .array = d3 }, none, 0.3818017741606059);
}

test "first, last, first_last, count_distinct and index hand-written values and options (pyarrow 26 results)" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    var a = try common.makeArray(i32, allocator, &.{ 3, 1, null, 3, 2, null, 1 });
    defer a.release();
    const da: Datum = .{ .array = a };
    var first = try agg(&ctx, "first", da, none);
    defer first.release();
    try std.testing.expectEqual(@as(i32, 3), first.scalar.value.i32);
    var last = try agg(&ctx, "last", da, none);
    defer last.release();
    try std.testing.expectEqual(@as(i32, 1), last.scalar.value.i32);
    var lead = try common.makeArray(i64, allocator, &.{ null, 2, 3 });
    defer lead.release();
    var f0 = try agg(&ctx, "first", .{ .array = lead }, scalarAgg(false, 1));
    defer f0.release();
    try std.testing.expect(f0.scalar.isNull());
    var f1 = try agg(&ctx, "first", .{ .array = lead }, scalarAgg(true, 3));
    defer f1.release();
    try std.testing.expect(f1.scalar.isNull());
    var f2 = try agg(&ctx, "first", .{ .array = lead }, scalarAgg(true, 2));
    defer f2.release();
    try std.testing.expectEqual(@as(i64, 2), f2.scalar.value.i64);
    var fl = try agg(&ctx, "first_last", da, none);
    defer fl.release();
    const payload = fl.scalar.payload.?.data();
    try std.testing.expectEqualStrings("first", payload.data_type.struct_.fields[0].name);
    try expectRowsOf(i32, payload.children[0], &.{3});
    try expectRowsOf(i32, payload.children[1], &.{1});
    var s = try common.makeStrings(.string, allocator, &.{ null, "b", "a" });
    defer s.release();
    var sf = try agg(&ctx, "first", .{ .array = s }, none);
    defer sf.release();
    try std.testing.expectEqualStrings("b", sf.scalar.value.string);
    var sl = try agg(&ctx, "last", .{ .array = s }, none);
    defer sl.release();
    try std.testing.expectEqualStrings("a", sl.scalar.value.string);
    // Chunked input with a leading null chunk.
    var chunked = try common.chunkedAt(allocator, lead, &.{1});
    defer chunked.release();
    var cf = try agg(&ctx, "first", .{ .chunked = chunked }, none);
    defer cf.release();
    try std.testing.expectEqual(@as(i64, 2), cf.scalar.value.i64);

    try expectI64(&ctx, "count_distinct", da, none, 3);
    try expectI64(&ctx, "count_distinct", da, .{ .count = .{ .mode = .all } }, 4);
    try expectI64(&ctx, "count_distinct", da, .{ .count = .{ .mode = .only_null } }, 1);
    var zeros = try common.makeArray(f64, allocator, &.{ 0.0, -0.0, nan, -nan });
    defer zeros.release();
    try expectI64(&ctx, "count_distinct", .{ .array = zeros }, none, 3);

    try expectI64(&ctx, "index", da, .{ .index = .{ .value = common.scalarOf(i32, 3) } }, 0);
    try expectI64(&ctx, "index", da, .{ .index = .{ .value = common.scalarOf(i32, 7) } }, -1);
    try expectI64(&ctx, "index", da, .{ .index = .{ .value = common.scalarOf(i64, 2) } }, 4);
    try expectI64(&ctx, "index", .{ .array = s }, .{ .index = .{ .value = Scalar.init(.{ .string = {} }, .{ .string = "a" }) } }, 2);
    var fv = try common.makeArray(f64, allocator, &.{ 1.0, nan, 2.0 });
    defer fv.release();
    try expectI64(&ctx, "index", .{ .array = fv }, .{ .index = .{ .value = common.scalarOf(f64, nan) } }, -1);
    var c3 = try common.makeArray(i64, allocator, &.{ 1, 2, 3 });
    defer c3.release();
    var cc3 = try common.chunkedAt(allocator, c3, &.{1});
    defer cc3.release();
    try expectI64(&ctx, "index", .{ .chunked = cc3 }, .{ .index = .{ .value = common.scalarOf(i64, 3) } }, 2);
    try expectI64(&ctx, "index", da, .{ .index = .{ .value = common.scalarOf(i32, null) } }, -1);
}

/// The rows of `arr` (of `T`) equal `expected`.
fn expectRowsOf(comptime T: type, arr: ArrayRef, expected: []const ?T) !void {
    const rows = try common.collect(T, std.testing.allocator, .{ .array = arr });
    defer std.testing.allocator.free(rows);
    try common.expectRows(T, expected, rows);
}

test "mode and quantile hand-written values and options (pyarrow 26 results)" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    var a = try common.makeArray(i32, allocator, &.{ 3, 1, null, 3, 2, null, 1 });
    defer a.release();
    const da: Datum = .{ .array = a };
    var m1 = try ctx.invokeVector("mode", &.{da}, none);
    defer m1.release();
    try expectRowsOf(i32, m1.array.data().children[0], &.{1});
    try expectRowsOf(i64, m1.array.data().children[1], &.{2});
    var m10 = try ctx.invokeVector("mode", &.{da}, .{ .mode = .{ .n = 10 } });
    defer m10.release();
    try m10.array.data().validateFull();
    try expectRowsOf(i32, m10.array.data().children[0], &.{ 1, 3, 2 });
    try expectRowsOf(i64, m10.array.data().children[1], &.{ 2, 2, 1 });
    var m0 = try ctx.invokeVector("mode", &.{da}, .{ .mode = .{ .skip_nulls = false } });
    defer m0.release();
    try std.testing.expectEqual(@as(usize, 0), m0.array.data().length);
    var mc = try ctx.invokeVector("mode", &.{da}, .{ .mode = .{ .min_count = 10 } });
    defer mc.release();
    try std.testing.expectEqual(@as(usize, 0), mc.array.data().length);
    var b = try common.makeArray(i64, allocator, &.{ 5, 3, 5, 3, 1, null, 1, 7 });
    defer b.release();
    var m3 = try ctx.invokeVector("mode", &.{.{ .array = b }}, .{ .mode = .{ .n = 3 } });
    defer m3.release();
    try expectRowsOf(i64, m3.array.data().children[0], &.{ 1, 3, 5 });
    var fn_ = try common.makeArray(f64, allocator, &.{ nan, 1.0, nan, 2.0, 2.0 });
    defer fn_.release();
    var mn = try ctx.invokeVector("mode", &.{.{ .array = fn_ }}, .{ .mode = .{ .n = 3 } });
    defer mn.release();
    try expectRowsOf(f64, mn.array.data().children[0], &.{ 2.0, nan, 1.0 });
    try expectRowsOf(i64, mn.array.data().children[1], &.{ 2, 2, 1 });
    var bools = try common.makeArray(bool, allocator, &.{ true, false, true });
    defer bools.release();
    var mb = try ctx.invokeVector("mode", &.{.{ .array = bools }}, none);
    defer mb.release();
    try expectRowsOf(bool, mb.array.data().children[0], &.{true});

    var q = try common.makeArray(i32, allocator, &.{ 1, 2, 3, 4 });
    defer q.release();
    const qs = [_]f64{ 0, 0.1, 0.25, 0.5, 0.75, 0.9, 1 };
    try expectQuantile(f64, &ctx, .{ .array = q }, .{ .q = &qs, .interpolation = .linear }, &.{ 1.0, 1.3, 1.75, 2.5, 3.25, 3.7, 4.0 });
    try expectQuantile(i32, &ctx, .{ .array = q }, .{ .q = &qs, .interpolation = .lower }, &.{ 1, 1, 1, 2, 3, 3, 4 });
    try expectQuantile(i32, &ctx, .{ .array = q }, .{ .q = &qs, .interpolation = .higher }, &.{ 1, 2, 2, 3, 4, 4, 4 });
    try expectQuantile(i32, &ctx, .{ .array = q }, .{ .q = &qs, .interpolation = .nearest }, &.{ 1, 1, 2, 3, 3, 4, 4 });
    try expectQuantile(f64, &ctx, .{ .array = q }, .{ .q = &qs, .interpolation = .midpoint }, &.{ 1.0, 1.5, 1.5, 2.5, 3.5, 3.5, 4.0 });
    var six = try common.makeArray(i64, allocator, &.{ 1, 2, 3, 4, 5, 6 });
    defer six.release();
    try expectQuantile(i64, &ctx, .{ .array = six }, .{ .q = &.{ 0.5, 0.1, 0.3, 0.7 }, .interpolation = .nearest }, &.{ 3, 1, 3, 5 });
    var f = try common.makeArray(f64, allocator, &.{ 1.5, 2.0, null, 4.0, -1.0, 2.5, 3.0, 8.0 });
    defer f.release();
    try expectQuantile(f64, &ctx, .{ .array = f }, .{ .q = &.{ 0.0, 0.33, 0.5, 0.99, 1.0 } }, &.{ -1.0, 1.99, 2.5, 7.759999999999998, 8.0 });
    try expectQuantile(f64, &ctx, .{ .array = f }, .{ .q = &.{ 0.33, 0.5 }, .interpolation = .midpoint }, &.{ 1.75, 2.5 });
    var with_nan = try common.makeArray(f64, allocator, &.{ 1.0, nan, 2.0, null, 4.0 });
    defer with_nan.release();
    try expectQuantile(f64, &ctx, .{ .array = with_nan }, .{}, &.{2.0});
    try expectQuantile(f64, &ctx, .{ .array = with_nan }, .{ .skip_nulls = false }, &.{null});
    var empty = try common.makeArray(i32, allocator, &.{});
    defer empty.release();
    try expectQuantile(f64, &ctx, .{ .array = empty }, .{}, &.{null});
    const d52 = DataType{ .decimal128 = .{ .precision = 5, .scale = 2 } };
    var d = try common.makeTyped(i128, d52, allocator, &.{ 100, 200 });
    defer d.release();
    try expectQuantile(f64, &ctx, .{ .array = d }, .{}, &.{1.5});
    var dq = try ctx.invokeVector("quantile", &.{.{ .array = d }}, .{ .quantile = .{ .interpolation = .lower } });
    defer dq.release();
    try std.testing.expect(dq.array.data().data_type.eql(d52));
}

fn expectQuantile(comptime T: type, ctx: *ExecContext, arg: Datum, o: common.core.QuantileOptions, expected: []const ?T) !void {
    var out = try ctx.invokeVector("quantile", &.{arg}, .{ .quantile = o });
    defer out.release();
    try out.array.data().validateFull();
    try std.testing.expect((try ctx.registry.resolveResultType("quantile", .vector, &.{arg}, .{ .quantile = o })).eql(out.array.data().data_type));
    const rows = try common.collect(T, std.testing.allocator, out);
    defer std.testing.allocator.free(rows);
    if (T == f64) {
        try std.testing.expectEqual(expected.len, rows.len);
        for (expected, rows) |e, r| {
            if (e == null or r == null) {
                try std.testing.expect(e == null and r == null);
            } else try expectNear(e.?, r.?);
        }
    } else try common.expectRows(T, expected, rows);
}

test "aggregate stats on random sliced inputs with nulls: chunked inputs and the update / merge lifecycle equal the array result" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    var prng = std.Random.DefaultPrng.init(0x57A7);
    const rng = prng.random();
    inline for (.{ i16, u32, f32, f64 }) |T| {
        var a = try common.randomArray(T, allocator, rng, 600, 7, 15, .{ .small = true });
        defer a.release();
        const rows = try common.collect(T, allocator, .{ .array = a });
        defer allocator.free(rows);
        // Reference variance (two passes over the valid values).
        var n: f64 = 0;
        var sum: f64 = 0;
        var finite = true;
        for (rows) |r| if (r) |v| {
            const x: f64 = if (@typeInfo(T) == .float) @floatCast(v) else @floatFromInt(v);
            if (!std.math.isFinite(x)) finite = false;
            n += 1;
            sum += x;
        };
        const mean = sum / n;
        var m2: f64 = 0;
        for (rows) |r| if (r) |v| {
            const x: f64 = if (@typeInfo(T) == .float) @floatCast(v) else @floatFromInt(v);
            m2 += (x - mean) * (x - mean);
        };
        var chunked = try common.chunkedAt(allocator, a, &.{ 3, 290, 291 });
        defer chunked.release();
        inline for (.{ "variance", "stddev", "skew", "kurtosis", "product", "count_distinct", "first", "last" }) |name| {
            var whole = try agg(&ctx, name, .{ .array = a }, none);
            defer whole.release();
            var parts = try agg(&ctx, name, .{ .chunked = chunked }, none);
            defer parts.release();
            try expectSameScalar(whole.scalar, parts.scalar);
            // update / merge: two sessions over the chunks, merged.
            var s1 = try ctx.beginAggregate(name, &.{.{ .array = a }}, none);
            defer s1.deinit();
            var s2 = try ctx.beginAggregate(name, &.{.{ .array = a }}, none);
            defer s2.deinit();
            for (chunked.chunks(), 0..) |c, k| try (if (k < 2) &s1 else &s2).update(&.{.{ .array = c }});
            try s1.merge(&s2);
            var merged = try s1.finalize();
            defer merged.release();
            try expectSameScalar(whole.scalar, merged.scalar);
            if (comptime std.mem.eql(u8, name, "variance")) {
                if (finite) try expectNear(m2 / n, whole.scalar.value.f64);
            }
        }
    }
}

fn expectSameScalar(a: Scalar, b: Scalar) !void {
    try std.testing.expectEqual(a.isNull(), b.isNull());
    if (a.isNull()) return;
    switch (a.value) {
        .f64 => |x| try expectNear(x, b.value.f64),
        .i64 => |x| try std.testing.expectEqual(x, b.value.i64),
        .u64 => |x| try std.testing.expectEqual(x, b.value.u64),
        .i16 => |x| try std.testing.expectEqual(x, b.value.i16),
        .u32 => |x| try std.testing.expectEqual(x, b.value.u32),
        .f32 => |x| try std.testing.expect(x == b.value.f32 or (std.math.isNan(x) and std.math.isNan(b.value.f32))),
        else => return error.TestUnexpectedResult,
    }
}

fn hashCall(ctx: *ExecContext, name: []const u8, values: ArrayRef, ids: ArrayRef, options: Options) !ArrayRef {
    var out = try ctx.invoke(name, .hash_aggregate, &.{ .{ .array = values }, .{ .array = ids } }, options);
    errdefer out.release();
    try out.array.data().validateFull();
    return out.array;
}

test "hash_product, hash_variance, hash_stddev, hash_skew, hash_kurtosis, hash_first, hash_last, hash_first_last, hash_one and hash_list (pyarrow 26 group_by results)" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    // Keys 1, 2, 1, 3, 2, 1, 3, 3 as group ids 0, 1, 0, 2, 1, 0, 2, 2.
    var ids = try common.makeArray(u32, allocator, &.{ 0, 1, 0, 2, 1, 0, 2, 2 });
    defer ids.release();
    var v = try common.makeArray(i32, allocator, &.{ 4, null, 2, 7, 5, 1, null, null });
    defer v.release();
    var f = try common.makeArray(f64, allocator, &.{ 1.5, 2.0, null, 4.0, -1.0, 2.5, 3.0, 8.0 });
    defer f.release();
    const F64Case = struct { name: []const u8, values: ArrayRef, options: Options, expected: [3]?f64 };
    const f64_cases = [_]F64Case{
        .{ .name = "hash_variance", .values = v, .options = none, .expected = .{ 1.5555555555555556, 0.0, 0.0 } },
        .{ .name = "hash_stddev", .values = v, .options = none, .expected = .{ 1.247219128924647, 0.0, 0.0 } },
        .{ .name = "hash_skew", .values = v, .options = none, .expected = .{ 0.3818017741606059, nan, nan } },
        .{ .name = "hash_kurtosis", .values = v, .options = none, .expected = .{ -1.4999999999999998, nan, nan } },
        .{ .name = "hash_variance", .values = f, .options = none, .expected = .{ 0.25, 2.25, 4.666666666666667 } },
        .{ .name = "hash_skew", .values = f, .options = none, .expected = .{ 0.0, 0.0, 0.5951700641394974 } },
        .{ .name = "hash_kurtosis", .values = f, .options = none, .expected = .{ -2.0, -2.0, -1.5 } },
        .{ .name = "hash_product", .values = f, .options = none, .expected = .{ 3.75, -2.0, 96.0 } },
        .{ .name = "hash_variance", .values = f, .options = variance(1, true, 0), .expected = .{ 0.5, 4.5, 7.0 } },
        .{ .name = "hash_variance", .values = f, .options = variance(0, false, 0), .expected = .{ null, 2.25, 4.666666666666667 } },
        .{ .name = "hash_variance", .values = f, .options = variance(0, true, 3), .expected = .{ null, null, 4.666666666666667 } },
        .{ .name = "hash_skew", .values = f, .options = .{ .skew = .{ .biased = false } }, .expected = .{ null, null, 1.4578629673213046 } },
        .{ .name = "hash_kurtosis", .values = f, .options = .{ .skew = .{ .biased = false } }, .expected = .{ null, null, null } },
    };
    for (f64_cases) |c| {
        var out = try hashCall(&ctx, c.name, c.values, ids, c.options);
        defer out.release();
        const rows = try common.collect(f64, allocator, .{ .array = out });
        defer allocator.free(rows);
        for (c.expected, rows) |e, r| {
            if (e == null or r == null) {
                try std.testing.expect(e == null and r == null);
            } else try expectNear(e.?, r.?);
        }
    }
    var prod = try hashCall(&ctx, "hash_product", v, ids, none);
    defer prod.release();
    try expectRowsOf(i64, prod, &.{ 8, 5, 7 });
    var prod_nulls = try hashCall(&ctx, "hash_product", v, ids, scalarAgg(false, 1));
    defer prod_nulls.release();
    try expectRowsOf(i64, prod_nulls, &.{ 8, null, null });
    var prod_min = try hashCall(&ctx, "hash_product", v, ids, scalarAgg(true, 2));
    defer prod_min.release();
    try expectRowsOf(i64, prod_min, &.{ 8, null, null });

    var first = try hashCall(&ctx, "hash_first", v, ids, none);
    defer first.release();
    try expectRowsOf(i32, first, &.{ 4, 5, 7 });
    var last = try hashCall(&ctx, "hash_last", v, ids, none);
    defer last.release();
    try expectRowsOf(i32, last, &.{ 1, 5, 7 });
    var one = try hashCall(&ctx, "hash_one", v, ids, none);
    defer one.release();
    try expectRowsOf(i32, one, &.{ 4, 5, 7 });
    var first_nulls = try hashCall(&ctx, "hash_first", v, ids, scalarAgg(false, 1));
    defer first_nulls.release();
    try expectRowsOf(i32, first_nulls, &.{ 4, null, 7 });
    var last_nulls = try hashCall(&ctx, "hash_last", v, ids, scalarAgg(false, 1));
    defer last_nulls.release();
    try expectRowsOf(i32, last_nulls, &.{ 1, 5, null });
    // pyarrow 26 group_by applies no min_count to hash_first.
    var first_min = try hashCall(&ctx, "hash_first", v, ids, scalarAgg(true, 2));
    defer first_min.release();
    try expectRowsOf(i32, first_min, &.{ 4, 5, 7 });
    var fl = try hashCall(&ctx, "hash_first_last", v, ids, scalarAgg(false, 1));
    defer fl.release();
    try expectRowsOf(i32, fl.data().children[0], &.{ 4, null, 7 });
    try expectRowsOf(i32, fl.data().children[1], &.{ 1, 5, null });
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const fl_type = try registry.resolveResultTypeAlloc(arena.allocator(), "hash_first_last", .hash_aggregate, &.{ .{ .array = v }, .{ .array = ids } }, none);
    try std.testing.expect(fl_type.eql(fl.data().data_type));
    var list = try hashCall(&ctx, "hash_list", v, ids, none);
    defer list.release();
    try expectRowsOf(i32, list.data().children[0], &.{ 4, 2, 1, null, 5, 7, null, null });
    var s = try common.makeStrings(.string, allocator, &.{ "a", "b", null, "c", "d", "e", "f", null });
    defer s.release();
    var sfirst = try hashCall(&ctx, "hash_first", s, ids, none);
    defer sfirst.release();
    const srows = try common.collectStrings(allocator, .{ .array = sfirst });
    defer allocator.free(srows);
    try common.expectStrings(&.{ "a", "b", "c" }, srows);

    // The lifecycle: two batches merged equal the one-shot result.
    var ids_a = try ids.slice(0, 5);
    defer ids_a.release();
    var ids_b = try ids.slice(5, 3);
    defer ids_b.release();
    var f_a = try f.slice(0, 5);
    defer f_a.release();
    var f_b = try f.slice(5, 3);
    defer f_b.release();
    inline for (.{ "hash_variance", "hash_last", "hash_list", "hash_product" }) |name| {
        var s1 = try ctx.beginAggregate(name, &.{ .{ .array = f }, .{ .array = ids } }, none);
        defer s1.deinit();
        var s2 = try ctx.beginAggregate(name, &.{ .{ .array = f }, .{ .array = ids } }, none);
        defer s2.deinit();
        try s1.update(&.{ .{ .array = f_a }, .{ .array = ids_a } });
        try s2.update(&.{ .{ .array = f_b }, .{ .array = ids_b } });
        try s1.merge(&s2);
        var merged = try s1.finalize();
        defer merged.release();
        var whole = try hashCall(&ctx, name, f, ids, none);
        defer whole.release();
        const x = try common.collect(f64, allocator, if (comptime std.mem.eql(u8, name, "hash_list")) .{ .array = merged.array.data().children[0] } else merged);
        defer allocator.free(x);
        const y = try common.collect(f64, allocator, .{ .array = if (comptime std.mem.eql(u8, name, "hash_list")) whole.data().children[0] else whole });
        defer allocator.free(y);
        try std.testing.expectEqual(y.len, x.len);
        for (x, y) |p, r| {
            if (p == null or r == null) try std.testing.expect(p == null and r == null) else try expectNear(r.?, p.?);
        }
    }
}

test "decimal min_max and hash_min_max keep the input precision and scale (pyarrow 26 results)" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    const d52 = DataType{ .decimal128 = .{ .precision = 5, .scale = 2 } };
    var d = try common.makeTyped(i128, d52, allocator, &.{ 150, -225, null });
    defer d.release();
    var mm = try agg(&ctx, "min_max", .{ .array = d }, none);
    defer mm.release();
    const p = mm.scalar.payload.?.data();
    try std.testing.expect(p.data_type.struct_.fields[0].data_type.eql(d52));
    try std.testing.expect(mm.scalar.data_type.struct_.fields[1].data_type.eql(d52));
    const lo = try (common.array_mod.PrimitiveArray(i128){ .data = p.children[0].data() }).values();
    const hi = try (common.array_mod.PrimitiveArray(i128){ .data = p.children[1].data() }).values();
    try std.testing.expectEqual(@as(i128, -225), lo[0]);
    try std.testing.expectEqual(@as(i128, 150), hi[0]);
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const rt = try registry.resolveResultTypeAlloc(arena.allocator(), "min_max", .aggregate, &.{.{ .array = d }}, none);
    try std.testing.expect(rt.eql(mm.scalar.data_type));
    try std.testing.expectError(error.NotImplemented, registry.resolveResultType("min_max", .aggregate, &.{.{ .array = d }}, none));
    var all_null = try common.makeTyped(i128, d52, allocator, &.{null});
    defer all_null.release();
    var mn = try agg(&ctx, "min_max", .{ .array = all_null }, none);
    defer mn.release();
    try std.testing.expect(mn.scalar.payload.?.data().children[0].data().isNull(0));
    // A retained copy of the scalar keeps its struct type valid.
    var kept = mm.scalar.retain();
    defer kept.release();
    try std.testing.expect(kept.data_type.struct_.fields[0].data_type.eql(d52));

    var ids = try common.makeArray(u32, allocator, &.{ 0, 0, 1 });
    defer ids.release();
    var h = try hashCall(&ctx, "hash_min_max", d, ids, none);
    defer h.release();
    try std.testing.expect(h.data().data_type.struct_.fields[0].data_type.eql(d52));
    const hlo = try (common.array_mod.PrimitiveArray(i128){ .data = h.data().children[0].data() }).values();
    try std.testing.expectEqual(@as(i128, -225), hlo[0]);
    try std.testing.expect(h.data().children[1].data().isNull(1));
    var hv = try hashCall(&ctx, "hash_variance", d, ids, none);
    defer hv.release();
    const hvr = try common.collect(f64, allocator, .{ .array = hv });
    defer allocator.free(hvr);
    try expectNear(3.515625, hvr[0].?);
    try std.testing.expect(hvr[1] == null);
}

test "aggregate stats errors: options, types, empty and all-null inputs, malformed input" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    var a = try common.makeArray(i32, allocator, &.{ 1, 2 });
    defer a.release();
    const da: Datum = .{ .array = a };
    try std.testing.expectError(error.InvalidOptions, agg(&ctx, "variance", da, .{ .variance = .{ .ddof = -1 } }));
    try std.testing.expectError(error.InvalidOptions, agg(&ctx, "variance", da, .{ .count = .{} }));
    try std.testing.expectError(error.InvalidOptions, agg(&ctx, "index", da, none));
    try std.testing.expectError(error.InvalidOptions, ctx.invokeVector("mode", &.{da}, .{ .mode = .{ .n = 0 } }));
    try std.testing.expectError(error.InvalidOptions, ctx.invokeVector("quantile", &.{da}, .{ .quantile = .{ .q = &.{1.5} } }));
    try std.testing.expectError(error.InvalidOptions, ctx.invokeVector("quantile", &.{da}, .{ .quantile = .{ .q = &.{nan} } }));
    var s = try common.makeStrings(.string, allocator, &.{ "a", "b" });
    defer s.release();
    try std.testing.expectError(error.NoMatchingKernel, ctx.invokeVector("mode", &.{.{ .array = s }}, none));
    try std.testing.expectError(error.NoMatchingKernel, agg(&ctx, "variance", .{ .array = s }, none));
    try std.testing.expectError(error.NoMatchingKernel, ctx.invokeVector("quantile", &.{.{ .array = s }}, none));
    var nulls = try common.makeArray(f64, allocator, &.{ null, null });
    defer nulls.release();
    try expectF64(&ctx, "variance", .{ .array = nulls }, none, null);
    var pn = try agg(&ctx, "first", .{ .array = nulls }, none);
    defer pn.release();
    try std.testing.expect(pn.scalar.isNull());
    try expectI64(&ctx, "count_distinct", .{ .array = nulls }, none, 0);
    try expectF64(&ctx, "skew", .{ .scalar = common.scalarOf(f64, 2.0) }, none, nan);
    try expectI64(&ctx, "product", .{ .scalar = common.scalarOf(i32, 5) }, none, 5);
    var ids = try common.makeArray(u32, allocator, &.{ 0, 5 });
    defer ids.release();
    try std.testing.expectError(error.InvalidInput, ctx.invoke("hash_first", .hash_aggregate, &.{ da, .{ .array = ids } }, none));
    // Fuzz finding (fuzz/corpus/ipc-stream/at-0.14.1-generated_datetime.stream): no group of a
    // date64 column built its empty output through a builder that lacks date64.
    var no_dates = try common.makeTyped(i64, .{ .date64 = {} }, allocator, &.{});
    defer no_dates.release();
    var no_ids = try common.makeArray(u32, allocator, &.{});
    defer no_ids.release();
    inline for (.{ "hash_first_last", "hash_list", "hash_one" }) |name| {
        var out = try hashCall(&ctx, name, no_dates, no_ids, none);
        defer out.release();
        try std.testing.expectEqual(@as(usize, 0), out.data().length);
    }
    const short = [_]i32{1};
    const bufs = [_]common.SharedBuffer{ .empty, .fromSlice(std.mem.sliceAsBytes(&short)) };
    var malformed = try ArrayRef.fromBorrowed(allocator, .{ .data_type = .{ .int32 = {} }, .length = 2, .null_count = 0, .buffers = &bufs });
    defer malformed.release();
    inline for (.{ "product", "variance", "first", "count_distinct" }) |name| {
        try std.testing.expectError(error.InvalidInput, agg(&ctx, name, .{ .array = malformed }, none));
    }
    try std.testing.expectError(error.InvalidInput, ctx.invokeVector("mode", &.{.{ .array = malformed }}, none));
    try std.testing.expectError(error.InvalidInput, ctx.invokeVector("quantile", &.{.{ .array = malformed }}, none));
}

fn decimalProduct(ctx: *ExecContext, args: []const Datum, options: Options) !?i256 {
    var out = try ctx.invokeAggregate("product", args, options);
    defer out.release();
    try std.testing.expect(out.scalar.data_type.eql(args[0].dataType()));
    return switch (out.scalar.value) {
        .null => null,
        .decimal128 => |v| v,
        .decimal256 => |v| v,
        else => error.TestUnexpectedResult,
    };
}

test "decimal product and hash_product: input type, Arrow rounding per step and per chunk, overflow modes (pyarrow 26 results)" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    const d10_2 = DataType{ .decimal128 = .{ .precision = 10, .scale = 2 } };
    const d40_3 = DataType{ .decimal256 = .{ .precision = 40, .scale = 3 } };
    var a = try common.makeTyped(i128, d10_2, allocator, &.{ 150, 200, null, -325 });
    defer a.release();
    try std.testing.expectEqual(@as(?i256, -975), try decimalProduct(&ctx, &.{.{ .array = a }}, none));
    try std.testing.expectEqual(@as(?i256, null), try decimalProduct(&ctx, &.{.{ .array = a }}, .{ .scalar_aggregate = .{ .skip_nulls = false } }));
    try std.testing.expectEqual(@as(?i256, null), try decimalProduct(&ctx, &.{.{ .array = a }}, .{ .scalar_aggregate = .{ .min_count = 4 } }));
    var w = try common.makeTyped(i256, d40_3, allocator, &.{ 1500, 2000, null, -3250 });
    defer w.release();
    try std.testing.expectEqual(@as(?i256, -9750), try decimalProduct(&ctx, &.{.{ .array = w }}, none));
    // The empty product is 1 at the input scale with min_count 0.
    var empty = try common.makeTyped(i128, d10_2, allocator, &.{});
    defer empty.release();
    try std.testing.expectEqual(@as(?i256, null), try decimalProduct(&ctx, &.{.{ .array = empty }}, none));
    try std.testing.expectEqual(@as(?i256, 100), try decimalProduct(&ctx, &.{.{ .array = empty }}, .{ .scalar_aggregate = .{ .min_count = 0 } }));
    var all_null = try common.makeTyped(i128, d10_2, allocator, &.{null});
    defer all_null.release();
    try std.testing.expectEqual(@as(?i256, 100), try decimalProduct(&ctx, &.{.{ .array = all_null }}, .{ .scalar_aggregate = .{ .min_count = 0 } }));
    // Each step rescales to the input scale, rounding half away from zero.
    const rounding = [_]struct { in: []const ?i128, out: i256 }{
        .{ .in = &.{ 125, 125 }, .out = 156 },
        .{ .in = &.{ 115, 115, 115 }, .out = 152 },
        .{ .in = &.{ -125, 125 }, .out = -156 },
        .{ .in = &.{ 5, 10 }, .out = 1 },
        .{ .in = &.{ 15, 10 }, .out = 2 },
        .{ .in = &.{ 101, 101, 129 }, .out = 132 },
    };
    for (rounding) |r| {
        var arr = try common.makeTyped(i128, d10_2, allocator, r.in);
        defer arr.release();
        try std.testing.expectEqual(@as(?i256, r.out), try decimalProduct(&ctx, &.{.{ .array = arr }}, none));
    }
    // A chunked input: each chunk is a product of its own, merged (Arrow: 1.31, flat 1.32).
    var three = try common.makeTyped(i128, d10_2, allocator, &.{ 101, 101, 129 });
    defer three.release();
    var chunked = try common.chunkedAt(allocator, three, &.{1});
    defer chunked.release();
    try std.testing.expectEqual(@as(?i256, 131), try decimalProduct(&ctx, &.{.{ .chunked = chunked }}, none));
    var scale0 = try common.makeTyped(i128, .{ .decimal128 = .{ .precision = 5, .scale = 0 } }, allocator, &.{ 12, -3 });
    defer scale0.release();
    try std.testing.expectEqual(@as(?i256, -36), try decimalProduct(&ctx, &.{.{ .array = scale0 }}, none));

    // Past the precision: checked is Overflow, wrapping gives Arrow's value (pyarrow 26 returns a
    // decimal128(10, 2) of 999999999700000000020000.00), saturating the largest value.
    var big = try common.makeTyped(i128, d10_2, allocator, &.{ 9999999999, 9999999999, 9999999999 });
    defer big.release();
    try std.testing.expectError(error.Overflow, ctx.invokeAggregate("product", &.{.{ .array = big }}, none));
    var wrap_ctx = ExecContext.initWithConfig(allocator, &registry, .{ .overflow_mode = .wrapping });
    try std.testing.expectEqual(@as(?i256, 99999999970000000002000000), try decimalProduct(&wrap_ctx, &.{.{ .array = big }}, none));
    var sat_ctx = ExecContext.initWithConfig(allocator, &registry, .{ .overflow_mode = .saturating });
    try std.testing.expectEqual(@as(?i256, 9999999999), try decimalProduct(&sat_ctx, &.{.{ .array = big }}, none));
    // decimal128(38, 2): Arrow's 128-bit product wraps (pyarrow 26: 8147088236464949819059335416690165.94).
    const nines: i128 = 99999999999999999999999999999999999999;
    var huge = try common.makeTyped(i128, .{ .decimal128 = .{ .precision = 38, .scale = 2 } }, allocator, &.{ nines, nines });
    defer huge.release();
    try std.testing.expectEqual(@as(?i256, 814708823646494981905933541669016594), try decimalProduct(&wrap_ctx, &.{.{ .array = huge }}, none));
    try std.testing.expectEqual(@as(?i256, nines), try decimalProduct(&sat_ctx, &.{.{ .array = huge }}, none));

    // hash_product: g = [1, 1, 2, 2, 3] over [1.50, 2.00, None, -3.25, None].
    var keys = try common.makeArray(i64, allocator, &.{ 1, 1, 2, 2, 3 });
    defer keys.release();
    var hv = try common.makeTyped(i128, d10_2, allocator, &.{ 150, 200, null, -325, null });
    defer hv.release();
    var by = try hash.groupBy(&ctx, &.{.{ .name = "g", .array = keys }}, &.{
        .{ .name = "p", .function = "hash_product", .input = hv },
        .{ .name = "p2", .function = "hash_product", .input = hv, .options = .{ .scalar_aggregate = .{ .skip_nulls = false, .min_count = 0 } } },
    });
    defer by.deinit();
    try std.testing.expect(by.columns[1].data().data_type.eql(d10_2));
    const p = try common.collect(i128, allocator, .{ .array = by.columns[1] });
    defer allocator.free(p);
    try common.expectRows(i128, &.{ 300, -325, null }, p);
    const p2 = try common.collect(i128, allocator, .{ .array = by.columns[2] });
    defer allocator.free(p2);
    try common.expectRows(i128, &.{ 300, null, null }, p2);
    var one_key = try common.makeArray(i64, allocator, &.{ 1, 1 });
    defer one_key.release();
    var two = try common.makeTyped(i128, d10_2, allocator, &.{ 115, 115 });
    defer two.release();
    var by2 = try hash.groupBy(&ctx, &.{.{ .name = "g", .array = one_key }}, &.{.{ .name = "p", .function = "hash_product", .input = two }});
    defer by2.deinit();
    const p3 = try common.collect(i128, allocator, .{ .array = by2.columns[1] });
    defer allocator.free(p3);
    try common.expectRows(i128, &.{132}, p3);
    var three_keys = try common.makeArray(i64, allocator, &.{ 1, 1, 1 });
    defer three_keys.release();
    try std.testing.expectError(error.Overflow, hash.groupBy(&ctx, &.{.{ .name = "g", .array = three_keys }}, &.{.{ .name = "p", .function = "hash_product", .input = big }}));
}

test "variance, stddev, skew and kurtosis and their hash_ forms on decimal32 / decimal64 (pyarrow 26 results)" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    var d32 = try common.makeTyped(i32, .{ .decimal32 = .{ .precision = 5, .scale = 2 } }, allocator, &.{ 150, null, -225 });
    defer d32.release();
    var d64 = try common.makeTyped(i64, .{ .decimal64 = .{ .precision = 12, .scale = 2 } }, allocator, &.{ 150, null, -225 });
    defer d64.release();
    var ids = try common.makeArray(u32, allocator, &.{ 0, 0, 1 });
    defer ids.release();
    for ([_]ArrayRef{ d32, d64 }) |d| {
        try expectF64(&ctx, "variance", .{ .array = d }, none, 3.515625);
        try expectF64(&ctx, "stddev", .{ .array = d }, none, 1.875);
        try expectF64(&ctx, "skew", .{ .array = d }, none, 0.0);
        try expectF64(&ctx, "kurtosis", .{ .array = d }, none, -2.0);
        try expectF64(&ctx, "variance", .{ .array = d }, variance(1, true, 0), 7.03125);
        try expectF64(&ctx, "variance", .{ .array = d }, variance(0, false, 0), null);
        // Sliced and chunked inputs, a scalar.
        var tail = try d.slice(1, 2);
        defer tail.release();
        try expectF64(&ctx, "variance", .{ .array = tail }, none, 0.0);
        var chunked = try common.chunkedAt(allocator, d, &.{1});
        defer chunked.release();
        try expectF64(&ctx, "stddev", .{ .chunked = chunked }, none, 1.875);
        var hv = try hashCall(&ctx, "hash_variance", d, ids, none);
        defer hv.release();
        const rows = try common.collect(f64, allocator, .{ .array = hv });
        defer allocator.free(rows);
        try expectNear(0.0, rows[0].?);
        try expectNear(0.0, rows[1].?);
        var hk = try hashCall(&ctx, "hash_stddev", d, ids, variance(1, true, 0));
        defer hk.release();
        try std.testing.expect(hk.data().isNull(0) and hk.data().isNull(1));
    }
    const s: Datum = .{ .scalar = Scalar.init(.{ .decimal32 = .{ .precision = 5, .scale = 2 } }, .{ .decimal32 = 150 }) };
    try expectF64(&ctx, "variance", s, none, 0.0);
}
