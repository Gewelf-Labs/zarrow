const std = @import("std");
const common = @import("test_common.zig");

const ArrayRef = common.ArrayRef;
const Datum = common.Datum;
const ExecContext = common.ExecContext;
const Scalar = common.Scalar;
const dense = common.dense;
const none = common.none;

fn aggregate(ctx: *ExecContext, name: []const u8, arg: Datum) !Scalar {
    var out = try ctx.invokeAggregate(name, &.{arg}, none);
    errdefer out.release();
    if (out != .scalar) return error.TestUnexpectedResult;
    return out.scalar;
}

fn expectScalar(comptime T: type, ctx: *ExecContext, name: []const u8, arg: Datum, want: ?T) !void {
    var s = try aggregate(ctx, name, arg);
    defer s.release();
    const got: ?T = if (s.isNull()) null else try dense.scalarAs(T, s);
    try common.expectRows(T, &.{want}, &.{got});
}

/// min and max fields of a min_max struct scalar.
fn minMax(comptime T: type, s: Scalar) ![2]?T {
    const payload = s.payload orelse return error.TestUnexpectedResult;
    try payload.data().validateFull();
    var out: [2]?T = undefined;
    for (&out, payload.data().children) |*o, child| {
        o.* = if (child.data().isNull(0)) null else (try (common.array_mod.PrimitiveArray(T){ .data = child.data() }).values())[0];
    }
    return out;
}

fn expectMinMax(comptime T: type, ctx: *ExecContext, arg: Datum, want: [2]?T) !void {
    var s = try aggregate(ctx, "min_max", arg);
    defer s.release();
    try std.testing.expect(s.data_type == .struct_);
    try common.expectRows(T, &want, &(try minMax(T, s)));
}

/// Random finite floats (the float sum / mean oracle compares within a tolerance).
fn finiteFloats(comptime T: type, allocator: std.mem.Allocator, rng: std.Random, len: usize, off: usize) !ArrayRef {
    const rows = try allocator.alloc(?T, len + off);
    defer allocator.free(rows);
    for (rows) |*r| r.* = if (rng.uintLessThan(u8, 5) == 0) null else @floatCast((rng.float(f64) - 0.5) * 1000);
    var full = try common.makeArray(T, allocator, rows);
    defer full.release();
    return full.slice(off, len);
}

test "aggregate hand-written values: sum, mean, min, max, min_max, count, count_all, any, all" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    var a = try common.makeArray(i32, allocator, &.{ 4, null, -10, 7, null });
    defer a.release();
    const d: Datum = .{ .array = a };
    try expectScalar(i64, &ctx, "sum", d, 1);
    try expectScalar(f64, &ctx, "mean", d, 1.0 / 3.0);
    try expectScalar(i32, &ctx, "min", d, -10);
    try expectScalar(i32, &ctx, "max", d, 7);
    try expectMinMax(i32, &ctx, d, .{ -10, 7 });
    try expectScalar(i64, &ctx, "count", d, 3);
    try expectScalar(i64, &ctx, "count_all", d, 5);

    var u = try common.makeArray(u8, allocator, &.{ 200, 100, 255 });
    defer u.release();
    try expectScalar(u64, &ctx, "sum", .{ .array = u }, 555);
    var f = try common.makeArray(f32, allocator, &.{ 1.5, null, 2.25 });
    defer f.release();
    try expectScalar(f64, &ctx, "sum", .{ .array = f }, 3.75);
    try expectScalar(f64, &ctx, "mean", .{ .array = f }, 1.875);

    // NaN is skipped by min / max unless every value is NaN (Arrow C++).
    const nan = std.math.nan(f64);
    var fn1 = try common.makeArray(f64, allocator, &.{ nan, 1.0, null, -2.0 });
    defer fn1.release();
    try expectScalar(f64, &ctx, "min", .{ .array = fn1 }, -2.0);
    try expectScalar(f64, &ctx, "max", .{ .array = fn1 }, 1.0);
    var all_nan = try common.makeArray(f64, allocator, &.{ nan, nan });
    defer all_nan.release();
    try expectScalar(f64, &ctx, "max", .{ .array = all_nan }, nan);

    // The mean of large int64 values is exact (no 64-bit wrap of the sum).
    var big = try common.makeArray(i64, allocator, &.{ 1 << 62, 1 << 62 });
    defer big.release();
    try expectScalar(f64, &ctx, "mean", .{ .array = big }, 4611686018427387904.0);

    var b = try common.makeArray(bool, allocator, &.{ true, null, false });
    defer b.release();
    try expectScalar(bool, &ctx, "any", .{ .array = b }, true);
    try expectScalar(bool, &ctx, "all", .{ .array = b }, false);
    var t = try common.makeArray(bool, allocator, &.{ true, null, true });
    defer t.release();
    try expectScalar(bool, &ctx, "all", .{ .array = t }, true);
    var fls = try common.makeArray(bool, allocator, &.{ false, null });
    defer fls.release();
    try expectScalar(bool, &ctx, "any", .{ .array = fls }, false);
}

test "aggregate oracle matches a per-element loop on random sliced inputs with nulls for every numeric type" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.initWithConfig(allocator, &registry, .{ .overflow_mode = .wrapping });
    var prng = std.Random.DefaultPrng.init(0xA66);
    const rng = prng.random();
    inline for (dense.numeric_types) |T| {
        const is_float = @typeInfo(T) == .float;
        for ([_][2]usize{ .{ 1000, 0 }, .{ 1003, 5 }, .{ 7, 3 } }) |shape| {
            var a = try common.randomArray(T, allocator, rng, shape[0], shape[1], 20, .{});
            defer a.release();
            const rows = try common.collect(T, allocator, .{ .array = a });
            defer allocator.free(rows);
            var count: i64 = 0;
            var lo: ?T = null;
            var hi: ?T = null;
            var exact: i128 = 0;
            for (rows) |r| {
                const v = r orelse continue;
                count += 1;
                if (is_float) {
                    if (std.math.isNan(v)) {
                        if (lo == null) lo = v;
                        if (hi == null) hi = v;
                        continue;
                    }
                    lo = if (lo == null or std.math.isNan(lo.?)) v else @min(lo.?, v);
                    hi = if (hi == null or std.math.isNan(hi.?)) v else @max(hi.?, v);
                } else {
                    exact += v;
                    lo = if (lo) |m| @min(m, v) else v;
                    hi = if (hi) |m| @max(m, v) else v;
                }
            }
            const d: Datum = .{ .array = a };
            try expectScalar(T, &ctx, "min", d, lo);
            try expectScalar(T, &ctx, "max", d, hi);
            try expectMinMax(T, &ctx, d, .{ lo, hi });
            try expectScalar(i64, &ctx, "count", d, count);
            try expectScalar(i64, &ctx, "count_all", d, @intCast(shape[0]));
            if (!is_float) {
                const Sum = if (@typeInfo(T).int.signedness == .signed) i64 else u64;
                try expectScalar(Sum, &ctx, "sum", d, if (count == 0) null else @as(Sum, @bitCast(@as(u64, @truncate(@as(u128, @bitCast(exact)))))));
                try expectScalar(f64, &ctx, "mean", d, if (count == 0) null else @as(f64, @floatFromInt(exact)) / @as(f64, @floatFromInt(count)));
            }
        }
        if (is_float) {
            var a = try finiteFloats(T, allocator, rng, 2001, 3);
            defer a.release();
            const rows = try common.collect(T, allocator, .{ .array = a });
            defer allocator.free(rows);
            var sum: f64 = 0;
            var count: f64 = 0;
            for (rows) |r| if (r) |v| {
                sum += v;
                count += 1;
            };
            var s = try aggregate(&ctx, "sum", .{ .array = a });
            defer s.release();
            try std.testing.expectApproxEqRel(sum, s.value.f64, 1e-9);
            var m = try aggregate(&ctx, "mean", .{ .array = a });
            defer m.release();
            try std.testing.expectApproxEqRel(sum / count, m.value.f64, 1e-9);
        }
    }
    var b = try common.randomArray(bool, allocator, rng, 1001, 5, 30, .{});
    defer b.release();
    const rows = try common.collect(bool, allocator, .{ .array = b });
    defer allocator.free(rows);
    var any = false;
    var all = true;
    for (rows) |r| if (r) |v| {
        any = any or v;
        all = all and v;
    };
    try expectScalar(bool, &ctx, "any", .{ .array = b }, any);
    try expectScalar(bool, &ctx, "all", .{ .array = b }, all);
}

test "aggregate chunked input with misaligned chunks and the update / merge lifecycle equal the array result" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    var prng = std.Random.DefaultPrng.init(0xA67);
    const rng = prng.random();
    var a = try common.randomArray(i64, allocator, rng, 900, 3, 20, .{ .small = true });
    defer a.release();
    var b = try common.randomArray(bool, allocator, rng, 900, 1, 20, .{});
    defer b.release();
    const names = [_][]const u8{ "sum", "mean", "min", "max", "count", "count_all" };
    for ([_]ArrayRef{ a, b }) |arr| {
        var chunked = try common.chunkedAt(allocator, arr, &.{ 1, 64, 65, 700 });
        defer chunked.release();
        const fn_names: []const []const u8 = if (arr.data().data_type == .bool) &.{ "any", "all", "count", "count_all" } else &names;
        for (fn_names) |name| {
            var whole = try aggregate(&ctx, name, .{ .array = arr });
            defer whole.release();
            var from_chunks = try aggregate(&ctx, name, .{ .chunked = chunked });
            defer from_chunks.release();
            try std.testing.expectEqual(whole.value, from_chunks.value);

            // Two sessions fed different chunks, then merged.
            var left = try ctx.beginAggregate(name, &.{.{ .array = arr }}, none);
            defer left.deinit();
            var right = try ctx.beginAggregate(name, &.{.{ .array = arr }}, none);
            defer right.deinit();
            for (chunked.chunks(), 0..) |chunk, k| {
                const session = if (k % 2 == 0) &left else &right;
                try session.update(&.{.{ .array = chunk }});
            }
            try left.merge(&right);
            var merged = try left.finalize();
            defer merged.release();
            try std.testing.expectEqual(whole.value, merged.scalar.value);
        }
        if (arr.data().data_type == .int64) {
            var whole = try aggregate(&ctx, "min_max", .{ .array = arr });
            defer whole.release();
            var from_chunks = try aggregate(&ctx, "min_max", .{ .chunked = chunked });
            defer from_chunks.release();
            try std.testing.expectEqual(try minMax(i64, whole), try minMax(i64, from_chunks));
        }
    }
}

test "aggregate scalar inputs count as one row" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    const five: Datum = .{ .scalar = common.scalarOf(i16, 5) };
    const null16: Datum = .{ .scalar = common.scalarOf(i16, null) };
    try expectScalar(i64, &ctx, "sum", five, 5);
    try expectScalar(f64, &ctx, "mean", five, 5);
    try expectScalar(i16, &ctx, "max", five, 5);
    try expectMinMax(i16, &ctx, five, .{ 5, 5 });
    try expectScalar(i64, &ctx, "count", five, 1);
    try expectScalar(i64, &ctx, "count", null16, 0);
    try expectScalar(i64, &ctx, "count_all", null16, 1);
    try expectScalar(i64, &ctx, "sum", null16, null);
    try expectScalar(bool, &ctx, "all", .{ .scalar = common.scalarOf(bool, false) }, false);
    try expectScalar(bool, &ctx, "any", .{ .scalar = common.scalarOf(bool, null) }, null);
    try expectScalar(i64, &ctx, "count", .{ .scalar = Scalar.init(.{ .string = {} }, .{ .string = "s" }) }, 1);
}

test "aggregate empty and all-null inputs give null (count gives 0)" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    var empty = try common.makeArray(u32, allocator, &.{});
    defer empty.release();
    var nulls = try common.makeArray(u32, allocator, &.{ null, null, null });
    defer nulls.release();
    var no_chunks = try common.ChunkedArray.init(allocator, .{ .uint32 = {} }, &.{});
    defer no_chunks.release();
    for ([_]Datum{ .{ .array = empty }, .{ .array = nulls }, .{ .chunked = no_chunks } }) |d| {
        try expectScalar(u64, &ctx, "sum", d, null);
        try expectScalar(f64, &ctx, "mean", d, null);
        try expectScalar(u32, &ctx, "min", d, null);
        try expectScalar(u32, &ctx, "max", d, null);
        try expectMinMax(u32, &ctx, d, .{ null, null });
        try expectScalar(i64, &ctx, "count", d, 0);
    }
    try expectScalar(i64, &ctx, "count_all", .{ .array = nulls }, 3);
    var bool_nulls = try common.makeArray(bool, allocator, &.{ null, null });
    defer bool_nulls.release();
    try expectScalar(bool, &ctx, "any", .{ .array = bool_nulls }, null);
    try expectScalar(bool, &ctx, "all", .{ .array = bool_nulls }, null);
    var bool_empty = try common.makeArray(bool, allocator, &.{});
    defer bool_empty.release();
    try expectScalar(bool, &ctx, "all", .{ .array = bool_empty }, null);
}

test "aggregate errors and overflow modes: checked sum Overflow, wrapping, saturating, mismatched types, options" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var big = try common.makeArray(i64, allocator, &.{ std.math.maxInt(i64), 1, null });
    defer big.release();
    const d: Datum = .{ .array = big };
    var checked = ExecContext.init(allocator, &registry);
    try std.testing.expectError(error.Overflow, checked.invokeAggregate("sum", &.{d}, none));
    var wrapping = ExecContext.initWithConfig(allocator, &registry, .{ .overflow_mode = .wrapping });
    try expectScalar(i64, &wrapping, "sum", d, std.math.minInt(i64));
    var saturating = ExecContext.initWithConfig(allocator, &registry, .{ .overflow_mode = .saturating });
    try expectScalar(i64, &saturating, "sum", d, std.math.maxInt(i64));
    // The exact sum decides: an intermediate overflow that comes back in range is not an error.
    var back = try common.makeArray(i64, allocator, &.{ std.math.maxInt(i64), 1, -2 });
    defer back.release();
    try expectScalar(i64, &checked, "sum", .{ .array = back }, std.math.maxInt(i64) - 1);
    var ubig = try common.makeArray(u64, allocator, &.{ std.math.maxInt(u64), 1 });
    defer ubig.release();
    try std.testing.expectError(error.Overflow, checked.invokeAggregate("sum", &.{.{ .array = ubig }}, none));
    try expectScalar(u64, &wrapping, "sum", .{ .array = ubig }, 0);
    // The lifecycle reports it at finalize.
    var session = try checked.beginAggregate("sum", &.{d}, none);
    defer session.deinit();
    try session.update(&.{d});
    try std.testing.expectError(error.Overflow, session.finalize());

    var prng = std.Random.DefaultPrng.init(1);
    var s = try common.randomStrings(.string, allocator, prng.random(), 3, 0, 0);
    defer s.release();
    try std.testing.expectError(error.NoMatchingKernel, checked.invokeAggregate("sum", &.{.{ .array = s }}, none));
    try std.testing.expectError(error.NoMatchingKernel, checked.invokeAggregate("any", &.{d}, none));
    try std.testing.expectError(error.InvalidArity, checked.invokeAggregate("min", &.{ d, d }, none));
    try std.testing.expectError(error.InvalidOptions, checked.invokeAggregate("max", &.{d}, .{ .sort = .{} }));
    var mismatched = try checked.beginAggregate("min", &.{d}, none);
    defer mismatched.deinit();
    try std.testing.expectError(error.NoMatchingKernel, mismatched.update(&.{.{ .array = s }}));
    const bad = Scalar.init(.{ .int64 = {} }, .{ .f64 = 1 });
    try std.testing.expectError(error.InvalidInput, checked.invokeAggregate("sum", &.{.{ .scalar = bad }}, none));
    const short = [_]i64{1};
    const bufs = [_]common.SharedBuffer{ .empty, .fromSlice(std.mem.sliceAsBytes(&short)) };
    var malformed = try ArrayRef.fromBorrowed(allocator, .{ .data_type = .{ .int64 = {} }, .length = 4, .null_count = 0, .buffers = &bufs });
    defer malformed.release();
    try std.testing.expectError(error.InvalidInput, checked.invokeAggregate("sum", &.{.{ .array = malformed }}, none));
}

fn aggOptions(skip_nulls: bool, min_count: u32) common.Options {
    return .{ .scalar_aggregate = .{ .skip_nulls = skip_nulls, .min_count = min_count } };
}

/// Invoke an aggregate with options and return its scalar as `?T` (null for a null scalar).
fn aggWith(comptime T: type, ctx: *ExecContext, name: []const u8, args: []const Datum, options: common.Options) !?T {
    var out = try ctx.invokeAggregate(name, args, options);
    defer out.release();
    return if (out.scalar.isNull()) null else try dense.scalarAs(T, out.scalar);
}

test "aggregate ScalarAggregateOptions and CountOptions: skip_nulls, min_count, count modes, nullary count_all (Arrow C++ results)" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);

    // Expected values from pyarrow 26.
    var with_null = try common.makeArray(i64, allocator, &.{ 1, null, 3 });
    defer with_null.release();
    var three = try common.makeArray(i64, allocator, &.{ 1, 2, 3 });
    defer three.release();
    var empty = try common.makeArray(i32, allocator, &.{});
    defer empty.release();
    var one_null = try common.makeArray(i32, allocator, &.{ 1, null });
    defer one_null.release();
    const wn = [_]Datum{.{ .array = with_null }};
    try std.testing.expectEqual(@as(?i64, null), try aggWith(i64, &ctx, "sum", &wn, aggOptions(false, 1)));
    try std.testing.expectEqual(@as(?i64, 4), try aggWith(i64, &ctx, "sum", &wn, aggOptions(true, 1)));
    try std.testing.expectEqual(@as(?i64, 4), try aggWith(i64, &ctx, "sum", &wn, aggOptions(true, 2)));
    try std.testing.expectEqual(@as(?i64, null), try aggWith(i64, &ctx, "sum", &wn, aggOptions(true, 3)));
    try std.testing.expectEqual(@as(?i64, null), try aggWith(i64, &ctx, "sum", &.{.{ .array = three }}, aggOptions(true, 4)));
    try std.testing.expectEqual(@as(?i64, 0), try aggWith(i64, &ctx, "sum", &.{.{ .array = empty }}, aggOptions(true, 0)));
    try std.testing.expect(std.math.isNan((try aggWith(f64, &ctx, "mean", &.{.{ .array = empty }}, aggOptions(true, 0))).?));
    try std.testing.expectEqual(@as(?i32, null), try aggWith(i32, &ctx, "min", &.{.{ .array = empty }}, aggOptions(true, 0)));
    try std.testing.expectEqual(@as(?i32, null), try aggWith(i32, &ctx, "min", &.{.{ .array = one_null }}, aggOptions(false, 0)));
    try std.testing.expectEqual(@as(?f64, null), try aggWith(f64, &ctx, "mean", &.{.{ .array = one_null }}, aggOptions(false, 1)));
    try std.testing.expectEqual(@as(?i32, 1), try aggWith(i32, &ctx, "max", &.{.{ .array = one_null }}, aggOptions(true, 1)));
    inline for (.{ .{ &empty, aggOptions(true, 0) }, .{ &one_null, aggOptions(false, 1) } }) |c| {
        var mm = try ctx.invokeAggregate("min_max", &.{.{ .array = c[0].* }}, c[1]);
        defer mm.release();
        try common.expectRows(i32, &.{ null, null }, &(try minMax(i32, mm.scalar)));
    }

    // any / all: Kleene logic without skip_nulls, min_count on valid rows.
    var tn = try common.makeArray(bool, allocator, &.{ true, null });
    defer tn.release();
    var fn_ = try common.makeArray(bool, allocator, &.{ false, null });
    defer fn_.release();
    var tf = try common.makeArray(bool, allocator, &.{ true, false });
    defer tf.release();
    var nn = try common.makeArray(bool, allocator, &.{ null, null });
    defer nn.release();
    var no_bools = try common.makeArray(bool, allocator, &.{});
    defer no_bools.release();
    const B = struct { name: []const u8, arr: *ArrayRef, o: common.Options, want: ?bool };
    for ([_]B{
        .{ .name = "any", .arr = &tn, .o = aggOptions(false, 1), .want = true },
        .{ .name = "any", .arr = &fn_, .o = aggOptions(false, 1), .want = null },
        .{ .name = "all", .arr = &fn_, .o = aggOptions(false, 1), .want = false },
        .{ .name = "all", .arr = &tn, .o = aggOptions(false, 1), .want = null },
        .{ .name = "all", .arr = &no_bools, .o = aggOptions(true, 0), .want = true },
        .{ .name = "any", .arr = &no_bools, .o = aggOptions(true, 0), .want = false },
        .{ .name = "any", .arr = &tf, .o = aggOptions(true, 3), .want = null },
        .{ .name = "any", .arr = &nn, .o = none, .want = null },
        .{ .name = "all", .arr = &nn, .o = none, .want = null },
        .{ .name = "all", .arr = &tn, .o = aggOptions(false, 0), .want = null },
        .{ .name = "all", .arr = &tn, .o = aggOptions(true, 1), .want = true },
    }) |c| {
        const got = try aggWith(bool, &ctx, c.name, &.{.{ .array = c.arr.* }}, c.o);
        std.testing.expectEqual(c.want, got) catch |err| {
            std.debug.print("{s}\n", .{c.name});
            return err;
        };
    }

    // CountOptions modes; count_all with Arrow's nullary calling convention counts no rows.
    try std.testing.expectEqual(@as(?i64, 2), try aggWith(i64, &ctx, "count", &wn, .{ .count = .{ .mode = .only_valid } }));
    try std.testing.expectEqual(@as(?i64, 1), try aggWith(i64, &ctx, "count", &wn, .{ .count = .{ .mode = .only_null } }));
    try std.testing.expectEqual(@as(?i64, 3), try aggWith(i64, &ctx, "count", &wn, .{ .count = .{ .mode = .all } }));
    try std.testing.expectEqual(@as(?i64, 3), try aggWith(i64, &ctx, "count_all", &wn, none));
    try std.testing.expectEqual(@as(?i64, 0), try aggWith(i64, &ctx, "count_all", &.{}, none));
    var nullary = try ctx.beginAggregate("count_all", &.{}, none);
    defer nullary.deinit();
    try nullary.update(&.{});
    var zero = try nullary.finalize();
    defer zero.release();
    try std.testing.expectEqual(@as(i64, 0), zero.scalar.value.i64);

    // The lifecycle and chunked inputs see the nulls of every batch.
    var parts = try common.chunkedAt(allocator, with_null, &.{1});
    defer parts.release();
    try std.testing.expectEqual(@as(?i64, null), try aggWith(i64, &ctx, "sum", &.{.{ .chunked = parts }}, aggOptions(false, 1)));
    var session = try ctx.beginAggregate("max", &.{.{ .array = three }}, aggOptions(false, 1));
    defer session.deinit();
    try session.update(&.{.{ .array = three }});
    var other = try ctx.beginAggregate("max", &.{.{ .array = three }}, aggOptions(false, 1));
    defer other.deinit();
    try other.update(&.{.{ .scalar = common.scalarOf(i64, null) }});
    try session.merge(&other);
    var merged = try session.finalize();
    defer merged.release();
    try std.testing.expect(merged.scalar.isNull());

    // Options are checked per function.
    try std.testing.expectError(error.InvalidOptions, ctx.invokeAggregate("sum", &wn, .{ .count = .{} }));
    try std.testing.expectError(error.InvalidOptions, ctx.invokeAggregate("count", &wn, aggOptions(true, 1)));
    try std.testing.expectError(error.InvalidOptions, ctx.invokeAggregate("count_all", &wn, .{ .count = .{} }));
}
