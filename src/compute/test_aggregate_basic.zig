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

// ------------------------------------------------------------------------------ ordered types

/// Bytes of a string / binary / fixed_size_binary scalar (null: a null scalar).
fn scalarBytes(s: Scalar) ?[]const u8 {
    return switch (s.value) {
        .string, .binary => |b| b,
        else => null,
    };
}

/// min, max and min_max of `arg` as bytes; every result passes validateFull.
fn expectBytesMinMax(ctx: *ExecContext, arg: Datum, options: common.Options, lo: ?[]const u8, hi: ?[]const u8) !void {
    for ([_][]const u8{ "min", "max" }, [_]?[]const u8{ lo, hi }) |name, want| {
        var out = try ctx.invokeAggregate(name, &.{arg}, options);
        defer out.release();
        try std.testing.expect(out.scalar.data_type.eql(arg.dataType()));
        try common.expectStrings(&.{want}, &.{scalarBytes(out.scalar)});
    }
    var mm = try ctx.invokeAggregate("min_max", &.{arg}, options);
    defer mm.release();
    const payload = mm.scalar.payload.?.data();
    try payload.validateFull();
    try std.testing.expect(payload.children[0].data().data_type.eql(arg.dataType()));
    try common.expectStrings(&.{ lo, hi }, &.{ try rowBytes(payload.children[0].data()), try rowBytes(payload.children[1].data()) });
}

/// Bytes of row 0 of a string / binary / fixed_size_binary array (null for a null row).
fn rowBytes(data: *const common.array_mod.ArrayData) !?[]const u8 {
    if (data.data_type != .fixed_size_binary) return common.bytesAt(data, 0);
    if (data.isNull(0)) return null;
    const w: usize = @intCast(data.data_type.fixed_size_binary.byte_width);
    return data.buffers[1].data[data.offset * w ..][0..w];
}

fn fixedBinary(allocator: std.mem.Allocator, width: usize, values: []const ?[]const u8) !ArrayRef {
    var b = try common.array_mod.FixedSizeBinaryBuilder.init(allocator, width, values.len);
    defer b.deinit();
    for (values) |v| if (v) |x| try b.append(x) else try b.appendNull();
    return b.finish();
}

test "min, max and min_max on bool, temporal, decimal32 / decimal64, string / binary layouts, fixed_size_binary and null (pyarrow 26 results)" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);

    var b = try common.makeArray(bool, allocator, &.{ true, null, false });
    defer b.release();
    try expectScalar(bool, &ctx, "min", .{ .array = b }, false);
    try expectScalar(bool, &ctx, "max", .{ .array = b }, true);
    var bmm = try aggregate(&ctx, "min_max", .{ .array = b });
    defer bmm.release();
    try std.testing.expect(bmm.payload.?.data().children[1].data().data_type == .bool);
    var trues = try b.slice(0, 2);
    defer trues.release();
    try expectScalar(bool, &ctx, "min", .{ .array = trues }, true);

    // Temporal types and decimal32 / decimal64 by value; the result keeps the input type.
    const ts = common.DataType{ .timestamp = .{ .unit = .second, .timezone = null } };
    var d32 = try common.makeTyped(i32, .{ .date32 = {} }, allocator, &.{ 3, null, 1 });
    defer d32.release();
    var t64 = try common.makeTyped(i64, ts, allocator, &.{ 3, null, -1 });
    defer t64.release();
    var tm32 = try common.makeTyped(i32, .{ .time32 = .{ .unit = .second } }, allocator, &.{ 3, null, 1 });
    defer tm32.release();
    var dur = try common.makeTyped(i64, .{ .duration = .{ .unit = .millisecond } }, allocator, &.{ -3, null, 1 });
    defer dur.release();
    var dec32 = try common.makeTyped(i32, .{ .decimal32 = .{ .precision = 5, .scale = 2 } }, allocator, &.{ 150, null, -225 });
    defer dec32.release();
    var dec64 = try common.makeTyped(i64, .{ .decimal64 = .{ .precision = 12, .scale = 2 } }, allocator, &.{ 150, null, -225 });
    defer dec64.release();
    for ([_]ArrayRef{ d32, t64, tm32, dur, dec32, dec64 }, [_][2]i64{ .{ 1, 3 }, .{ -1, 3 }, .{ 1, 3 }, .{ -3, 1 }, .{ -225, 150 }, .{ -225, 150 } }) |arr, want| {
        for ([_][]const u8{ "min", "max" }, want) |name, w| {
            var s = try aggregate(&ctx, name, .{ .array = arr });
            defer s.release();
            try std.testing.expect(s.data_type.eql(arr.data().data_type));
            const got: i64 = switch (s.value) {
                inline .date32, .time32, .decimal32 => |x| x,
                inline .timestamp, .duration, .decimal64 => |x| x,
                else => return error.TestUnexpectedResult,
            };
            try std.testing.expectEqual(w, got);
        }
        var mm = try aggregate(&ctx, "min_max", .{ .array = arr });
        defer mm.release();
        try std.testing.expect(mm.data_type.struct_.fields[0].data_type.eql(arr.data().data_type));
        var arena = std.heap.ArenaAllocator.init(allocator);
        defer arena.deinit();
        const rt = try registry.resolveResultTypeAlloc(arena.allocator(), "min_max", .aggregate, &.{.{ .array = arr }}, none);
        try std.testing.expect(rt.eql(mm.data_type));
    }

    // Every string and binary layout by bytes (0xff sorts last).
    const rows = [_]?[]const u8{ "b", null, "a", "ab" };
    inline for (.{ .string, .large_string, .string_view }) |tag| {
        var s = try common.makeStrings(tag, allocator, &rows);
        defer s.release();
        try expectBytesMinMax(&ctx, .{ .array = s }, none, "a", "b");
    }
    const bins = [_]?[]const u8{ "b", null, "\xff", "ab" };
    inline for (.{ .binary, .large_binary, .binary_view }) |tag| {
        var s = try common.makeStrings(tag, allocator, &bins);
        defer s.release();
        try expectBytesMinMax(&ctx, .{ .array = s }, none, "ab", "\xff");
    }
    var fsb = try fixedBinary(allocator, 2, &.{ "bb", null, "\xff\x00", "ab" });
    defer fsb.release();
    try expectBytesMinMax(&ctx, .{ .array = fsb }, none, "ab", "\xff\x00");

    // Sliced, chunked (one chunk all null), scalar, options, empty and all-null inputs.
    var long = try common.makeStrings(.string, allocator, &.{ "zz", "m", null, "c", "x", null, "d", "a" });
    defer long.release();
    var mid = try long.slice(1, 6);
    defer mid.release();
    try expectBytesMinMax(&ctx, .{ .array = mid }, none, "c", "x");
    var chunked = try common.chunkedAt(allocator, long, &.{ 2, 3, 5, 6 });
    defer chunked.release();
    try expectBytesMinMax(&ctx, .{ .chunked = chunked }, none, "a", "zz");
    const one: Datum = .{ .scalar = Scalar.init(.{ .string = {} }, .{ .string = "q" }) };
    try expectBytesMinMax(&ctx, one, none, "q", "q");
    try expectBytesMinMax(&ctx, .{ .array = mid }, .{ .scalar_aggregate = .{ .skip_nulls = false } }, null, null);
    try expectBytesMinMax(&ctx, .{ .array = mid }, .{ .scalar_aggregate = .{ .min_count = 5 } }, null, null);
    try expectBytesMinMax(&ctx, .{ .array = mid }, .{ .scalar_aggregate = .{ .min_count = 0 } }, "c", "x");
    var empty = try long.slice(0, 0);
    defer empty.release();
    try expectBytesMinMax(&ctx, .{ .array = empty }, .{ .scalar_aggregate = .{ .min_count = 0 } }, null, null);
    var nulls = try long.slice(5, 1);
    defer nulls.release();
    try expectBytesMinMax(&ctx, .{ .array = nulls }, none, null, null);

    // Two sessions fed alternate chunks, then merged, equal the whole input.
    for ([_][]const u8{ "min", "max" }, [_][]const u8{ "a", "zz" }) |name, want| {
        var left = try ctx.beginAggregate(name, &.{.{ .array = long }}, none);
        defer left.deinit();
        var right = try ctx.beginAggregate(name, &.{.{ .array = long }}, none);
        defer right.deinit();
        for (chunked.chunks(), 0..) |chunk, k| try (if (k % 2 == 0) &left else &right).update(&.{.{ .array = chunk }});
        try left.merge(&right);
        var merged = try left.finalize();
        defer merged.release();
        try common.expectStrings(&.{want}, &.{scalarBytes(merged.scalar)});
    }

    // The null type: null, and min_max of two nulls (struct<min: null, max: null>).
    var nul = try common.core.datumBuildNullLikeWithAllocator(allocator, .{ .null = {} }, 3);
    defer nul.release();
    var nm = try aggregate(&ctx, "min", nul);
    defer nm.release();
    try std.testing.expect(nm.isNull() and nm.data_type == .null);
    var nmm = try aggregate(&ctx, "min_max", nul);
    defer nmm.release();
    try std.testing.expect(nmm.payload.?.data().children[0].data().data_type == .null);

    // Intervals have no order (pyarrow 26: no kernel).
    var iv = try common.makeTyped(i32, .{ .interval_months = .{ .unit = .months } }, allocator, &.{1});
    defer iv.release();
    try std.testing.expectError(error.NoMatchingKernel, ctx.invokeAggregate("min", &.{.{ .array = iv }}, none));
}

fn hashOf(ctx: *ExecContext, name: []const u8, values: Datum, ids: Datum, options: common.Options) !ArrayRef {
    var out = try ctx.invoke(name, .hash_aggregate, &.{ values, ids }, options);
    errdefer out.release();
    try out.array.data().validateFull();
    return out.array;
}

test "hash_min, hash_max and hash_min_max on ordered non-numeric types: strings, bool, dates, fixed_size_binary, chunked and lifecycle (pyarrow 26 group_by results)" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);

    var v = try common.makeStrings(.large_string, allocator, &.{ "b", "a", null, "c", null, "zz", "e" });
    defer v.release();
    var ids = try common.makeArray(u32, allocator, &.{ 0, 0, 1, 1, 2, 0, 3 });
    defer ids.release();
    var lo = try hashOf(&ctx, "hash_min", .{ .array = v }, .{ .array = ids }, none);
    defer lo.release();
    const lo_rows = try common.collectStrings(allocator, .{ .array = lo });
    defer allocator.free(lo_rows);
    try common.expectStrings(&.{ "a", "c", null, "e" }, lo_rows);
    var mm = try hashOf(&ctx, "hash_min_max", .{ .array = v }, .{ .array = ids }, none);
    defer mm.release();
    try std.testing.expect(mm.data().data_type.struct_.fields[1].data_type.* == .large_string);
    const hi_rows = try common.collectStrings(allocator, .{ .array = mm.data().children[1] });
    defer allocator.free(hi_rows);
    try common.expectStrings(&.{ "zz", "c", null, "e" }, hi_rows);
    // skip_nulls = false: a group with a null row is null.
    var strict = try hashOf(&ctx, "hash_max", .{ .array = v }, .{ .array = ids }, .{ .scalar_aggregate = .{ .skip_nulls = false } });
    defer strict.release();
    const strict_rows = try common.collectStrings(allocator, .{ .array = strict });
    defer allocator.free(strict_rows);
    try common.expectStrings(&.{ "zz", null, null, "e" }, strict_rows);

    // Chunked values and ids with different boundaries equal the array result, and so do two
    // lifecycle sessions merged.
    var cv = try common.chunkedAt(allocator, v, &.{ 1, 4 });
    defer cv.release();
    var ci = try common.chunkedAt(allocator, ids, &.{ 2, 3, 6 });
    defer ci.release();
    var from_chunks = try hashOf(&ctx, "hash_min", .{ .chunked = cv }, .{ .chunked = ci }, none);
    defer from_chunks.release();
    const chunk_rows = try common.collectStrings(allocator, .{ .array = from_chunks });
    defer allocator.free(chunk_rows);
    try common.expectStrings(lo_rows, chunk_rows);
    var s1 = try ctx.beginAggregate("hash_min_max", &.{ .{ .array = v }, .{ .array = ids } }, none);
    defer s1.deinit();
    var s2 = try ctx.beginAggregate("hash_min_max", &.{ .{ .array = v }, .{ .array = ids } }, none);
    defer s2.deinit();
    var v1 = try v.slice(0, 4);
    defer v1.release();
    var v2 = try v.slice(4, 3);
    defer v2.release();
    var id1 = try ids.slice(0, 4);
    defer id1.release();
    var id2 = try ids.slice(4, 3);
    defer id2.release();
    try s1.update(&.{ .{ .array = v1 }, .{ .array = id1 } });
    try s2.update(&.{ .{ .array = v2 }, .{ .array = id2 } });
    try s1.merge(&s2);
    var merged = try s1.finalize();
    defer merged.release();
    try merged.array.data().validateFull();
    const merged_hi = try common.collectStrings(allocator, .{ .array = merged.array.data().children[1] });
    defer allocator.free(merged_hi);
    try common.expectStrings(hi_rows, merged_hi);

    // bool, date32 and fixed_size_binary.
    var bools = try common.makeArray(bool, allocator, &.{ true, false, null, true, null, true, false });
    defer bools.release();
    var bmin = try hashOf(&ctx, "hash_min", .{ .array = bools }, .{ .array = ids }, none);
    defer bmin.release();
    const bvals = try common.collect(bool, allocator, .{ .array = bmin });
    defer allocator.free(bvals);
    try common.expectRows(bool, &.{ false, true, null, false }, bvals);
    var dates = try common.makeTyped(i32, .{ .date32 = {} }, allocator, &.{ 5, 2, null, 7, null, 9, -1 });
    defer dates.release();
    var dmax = try hashOf(&ctx, "hash_max", .{ .array = dates }, .{ .array = ids }, none);
    defer dmax.release();
    try std.testing.expect(dmax.data().data_type == .date32);
    const dvals = try common.collect(i32, allocator, .{ .array = dmax });
    defer allocator.free(dvals);
    try common.expectRows(i32, &.{ 9, 7, null, -1 }, dvals);
    var fsb = try fixedBinary(allocator, 1, &.{ "b", "a", null, "c", null, "\xff", "e" });
    defer fsb.release();
    var fmax = try hashOf(&ctx, "hash_max", .{ .array = fsb }, .{ .array = ids }, none);
    defer fmax.release();
    try std.testing.expectEqualStrings("\xff", fmax.data().buffers[1].data[0..1]);
    // The null type: every group null (pyarrow 26: struct<min: null, max: null>).
    var nul = try common.core.datumBuildNullLikeWithAllocator(allocator, .{ .null = {} }, 7);
    defer nul.release();
    var nmm = try hashOf(&ctx, "hash_min_max", nul, .{ .array = ids }, none);
    defer nmm.release();
    try std.testing.expect(nmm.data().children[0].data().data_type == .null and nmm.data().length == 4);
}

test "min and max of byte rows match std.mem.order on random rows sharing prefixes, with nulls and every layout" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    var prng = std.Random.DefaultPrng.init(0x18);
    const rng = prng.random();
    const alphabet = [_]u8{ 0, 1, 'a', 'b', 0x7f, 0xfe, 0xff };
    var storage: [64][24]u8 = undefined;
    var rows: [64]?[]const u8 = undefined;
    for (0..40) |_| {
        const prefix_len = rng.uintLessThan(usize, 12);
        const shared = rng.uintLessThan(u8, 3);
        for (&rows, &storage) |*r, *buf| {
            if (rng.uintLessThan(u8, 10) == 0) {
                r.* = null;
                continue;
            }
            const len = prefix_len + rng.uintLessThan(usize, 12);
            for (buf[0..len], 0..) |*c, k| c.* = if (k < prefix_len) alphabet[shared] else alphabet[rng.uintLessThan(usize, alphabet.len)];
            r.* = buf[0..len];
        }
        var want_lo: ?[]const u8 = null;
        var want_hi: ?[]const u8 = null;
        for (rows) |r| if (r) |x| {
            if (want_lo == null or std.mem.order(u8, x, want_lo.?) == .lt) want_lo = x;
            if (want_hi == null or std.mem.order(u8, x, want_hi.?) == .gt) want_hi = x;
        };
        inline for (.{ .binary, .large_binary, .binary_view }) |tag| {
            var arr = try common.makeStrings(tag, allocator, &rows);
            defer arr.release();
            try expectBytesMinMax(&ctx, .{ .array = arr }, none, want_lo, want_hi);
        }
    }
}

test "min and max of byte rows across scan blocks: 3000 rows, extremes in later blocks, nulls, sliced" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    var prng = std.Random.DefaultPrng.init(0x1818);
    const rng = prng.random();
    const n = 3000;
    const storage = try allocator.alloc([20]u8, n);
    defer allocator.free(storage);
    const rows = try allocator.alloc(?[]const u8, n);
    defer allocator.free(rows);
    for (0..6) |round| {
        for (rows, storage, 0..) |*r, *buf, i| {
            if (rng.uintLessThan(u8, 8) == 0) {
                r.* = null;
                continue;
            }
            // "row-XXXXXXXX" with a shared prefix; later rows reach further in odd rounds.
            const hi_digit: u8 = if (round % 2 == 1 and i > 2000) '9' else '0' + rng.uintLessThan(u8, 9);
            const len = 9 + rng.uintLessThan(usize, 11);
            @memcpy(buf[0..4], "row-");
            buf[4] = hi_digit;
            for (buf[5..len]) |*c| c.* = 'a' + rng.uintLessThan(u8, 3);
            r.* = buf[0..len];
        }
        const off = round * 7;
        var want_lo: ?[]const u8 = null;
        var want_hi: ?[]const u8 = null;
        for (rows[off..]) |r| if (r) |x| {
            if (want_lo == null or std.mem.order(u8, x, want_lo.?) == .lt) want_lo = x;
            if (want_hi == null or std.mem.order(u8, x, want_hi.?) == .gt) want_hi = x;
        };
        var arr = try common.makeStrings(.string, allocator, rows);
        defer arr.release();
        var sliced = try arr.slice(off, n - off);
        defer sliced.release();
        try expectBytesMinMax(&ctx, .{ .array = sliced }, none, want_lo, want_hi);
    }
}
