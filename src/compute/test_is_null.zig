const std = @import("std");
const common = @import("test_common.zig");

const ArrayRef = common.ArrayRef;
const DataType = common.DataType;
const Datum = common.Datum;
const ExecContext = common.ExecContext;
const Scalar = common.Scalar;
const none = common.none;

const ops = [_][]const u8{ "is_null", "is_valid", "true_unless_null" };

fn reference(comptime op: []const u8, is_null: bool) ?bool {
    if (comptime std.mem.eql(u8, op, "is_null")) return is_null;
    if (comptime std.mem.eql(u8, op, "is_valid")) return !is_null;
    return if (is_null) null else true;
}

fn expectMatchesReference(ctx: *ExecContext, arr: ArrayRef) !void {
    const allocator = std.testing.allocator;
    const expected = try allocator.alloc(?bool, arr.data().length);
    defer allocator.free(expected);
    inline for (ops) |op| {
        for (expected, 0..) |*e, i| e.* = reference(op, arr.data().isNull(i));
        try common.expectInvoke(bool, ctx, .scalar, op, &.{.{ .array = arr }}, expected);
    }
}

test "is_null, is_valid and true_unless_null hand-written values for primitive, string, null and dictionary arrays" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    var a = try common.makeArray(f64, allocator, &.{ 1.0, null, std.math.nan(f64), null });
    defer a.release();
    // NaN is a value, not a null.
    try common.expectInvoke(bool, &ctx, .scalar, "is_null", &.{.{ .array = a }}, &.{ false, true, false, true });
    try common.expectInvoke(bool, &ctx, .scalar, "is_valid", &.{.{ .array = a }}, &.{ true, false, true, false });
    try common.expectInvoke(bool, &ctx, .scalar, "true_unless_null", &.{.{ .array = a }}, &.{ true, null, true, null });

    var no_nulls = try common.makeArray(i8, allocator, &.{ 1, 2 });
    defer no_nulls.release();
    try common.expectInvoke(bool, &ctx, .scalar, "is_null", &.{.{ .array = no_nulls }}, &.{ false, false });
    try common.expectInvoke(bool, &ctx, .scalar, "true_unless_null", &.{.{ .array = no_nulls }}, &.{ true, true });

    var nb = try common.array_mod.NullBuilder.init(allocator, 3);
    defer nb.deinit();
    try nb.appendNulls(3);
    var null_arr = try nb.finish();
    defer null_arr.release();
    try common.expectInvoke(bool, &ctx, .scalar, "is_null", &.{.{ .array = null_arr }}, &.{ true, true, true });
    try common.expectInvoke(bool, &ctx, .scalar, "is_valid", &.{.{ .array = null_arr }}, &.{ false, false, false });
    try common.expectInvoke(bool, &ctx, .scalar, "true_unless_null", &.{.{ .array = null_arr }}, &.{ null, null, null });

    const string_type = DataType{ .string = {} };
    var dict_values_builder = try common.array_mod.StringBuilder.init(allocator, 2, 8);
    defer dict_values_builder.deinit();
    try dict_values_builder.append("red");
    try dict_values_builder.append("blue");
    var dict_values = try dict_values_builder.finish();
    defer dict_values.release();
    var dict_builder = try common.array_mod.DictionaryBuilder.init(allocator, .{ .bit_width = 32, .signed = true }, &string_type, 3);
    defer dict_builder.deinit();
    try dict_builder.appendIndex(1);
    try dict_builder.appendNull();
    try dict_builder.appendIndex(0);
    var dict = try dict_builder.finish(dict_values);
    defer dict.release();
    try common.expectInvoke(bool, &ctx, .scalar, "is_valid", &.{.{ .array = dict }}, &.{ true, false, true });
}

test "is_null family oracle matches isNull per row on random sliced inputs of several types" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    var prng = std.Random.DefaultPrng.init(0x15A1);
    const rng = prng.random();
    for ([_][2]usize{ .{ 1000, 0 }, .{ 1001, 3 }, .{ 13, 9 } }) |shape| {
        for ([_]u8{ 0, 30, 100 }) |pct| {
            var ints = try common.randomArray(u32, allocator, rng, shape[0], shape[1], pct, .{});
            defer ints.release();
            try expectMatchesReference(&ctx, ints);
            var bools = try common.randomArray(bool, allocator, rng, shape[0], shape[1], pct, .{});
            defer bools.release();
            try expectMatchesReference(&ctx, bools);
            var views = try common.randomStrings(.string_view, allocator, rng, shape[0], shape[1], pct);
            defer views.release();
            try expectMatchesReference(&ctx, views);
        }
    }
}

test "is_null family on chunked input with misaligned chunks keeps the chunks" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    var prng = std.Random.DefaultPrng.init(0x15A2);
    var a = try common.randomArray(i16, allocator, prng.random(), 500, 7, 30, .{});
    defer a.release();
    var ca = try common.chunkedAt(allocator, a, &.{ 1, 100, 333 });
    defer ca.release();
    inline for (ops) |op| {
        var whole = try ctx.invokeScalar(op, &.{.{ .array = a }}, none);
        defer whole.release();
        const expected = try common.collect(bool, allocator, whole);
        defer allocator.free(expected);
        var chunked = try ctx.invokeScalar(op, &.{.{ .chunked = ca }}, none);
        defer chunked.release();
        try std.testing.expectEqual(@as(usize, 4), chunked.chunked.numChunks());
        const got = try common.collect(bool, allocator, chunked);
        defer allocator.free(got);
        try common.expectRows(bool, expected, got);
    }
}

test "is_null family on scalars: a null scalar is null, the result is never null for is_null and is_valid" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    const cases = [_]struct { s: Scalar, null_in: bool }{
        .{ .s = common.scalarOf(i32, null), .null_in = true },
        .{ .s = common.scalarOf(i32, 4), .null_in = false },
        .{ .s = Scalar.init(.{ .string = {} }, .{ .string = "x" }), .null_in = false },
        .{ .s = Scalar.init(.{ .null = {} }, .null), .null_in = true },
    };
    for (cases) |c| {
        inline for (ops) |op| {
            var out = try ctx.invokeScalar(op, &.{.{ .scalar = c.s }}, none);
            defer out.release();
            try std.testing.expect(out == .scalar);
            const want = reference(op, c.null_in);
            try std.testing.expectEqual(want, if (out.scalar.isNull()) null else out.scalar.value.bool);
        }
    }
}

test "is_null family on empty and all-null inputs" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    var empty = try common.makeArray(u64, allocator, &.{});
    defer empty.release();
    var nulls = try common.makeArray(u64, allocator, &.{ null, null, null, null, null, null, null, null, null });
    defer nulls.release();
    inline for (ops) |op| {
        try common.expectInvoke(bool, &ctx, .scalar, op, &.{.{ .array = empty }}, &.{});
        var chunked = try common.ChunkedArray.init(allocator, .{ .uint64 = {} }, &.{});
        defer chunked.release();
        var out = try ctx.invokeScalar(op, &.{.{ .chunked = chunked }}, none);
        defer out.release();
        try std.testing.expectEqual(@as(usize, 0), out.chunked.len());
    }
    const all_true: [9]?bool = @splat(true);
    const all_null: [9]?bool = @splat(null);
    try common.expectInvoke(bool, &ctx, .scalar, "is_null", &.{.{ .array = nulls }}, &all_true);
    try common.expectInvoke(bool, &ctx, .scalar, "true_unless_null", &.{.{ .array = nulls }}, &all_null);
}

test "is_null family errors: arity and options" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    var a = try common.makeArray(i32, allocator, &.{1});
    defer a.release();
    try std.testing.expectError(error.InvalidArity, ctx.invokeScalar("is_valid", &.{ .{ .array = a }, .{ .array = a } }, none));
    try std.testing.expectError(error.InvalidOptions, ctx.invokeScalar("is_valid", &.{.{ .array = a }}, .{ .cast = .{} }));
}

test "is_nan, is_finite, is_inf and is_null with nan_is_null on floats, integers, decimals and the null type (Arrow C++ results)" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    const inf = std.math.inf(f64);
    // pyarrow 26 results.
    var f = try common.makeArray(f64, allocator, &.{ 1.0, std.math.nan(f64), inf, -inf, null, -0.0 });
    defer f.release();
    const fa = [_]Datum{.{ .array = f }};
    try common.expectInvoke(bool, &ctx, .scalar, "is_nan", &fa, &.{ false, true, false, false, null, false });
    try common.expectInvoke(bool, &ctx, .scalar, "is_finite", &fa, &.{ true, false, false, false, null, true });
    try common.expectInvoke(bool, &ctx, .scalar, "is_inf", &fa, &.{ false, false, true, true, null, false });
    // Sliced float32.
    var g_full = try common.makeArray(f32, allocator, &.{ 0, 0, std.math.nan(f32), 2, -std.math.inf(f32) });
    defer g_full.release();
    var g = try g_full.slice(2, 3);
    defer g.release();
    try common.expectInvoke(bool, &ctx, .scalar, "is_nan", &.{.{ .array = g }}, &.{ true, false, false });
    try common.expectInvoke(bool, &ctx, .scalar, "is_inf", &.{.{ .array = g }}, &.{ false, false, true });
    // Integers: finite, never NaN or infinite; nulls stay null.
    var ints = try common.makeArray(i32, allocator, &.{ 1, null, std.math.minInt(i32) });
    defer ints.release();
    try common.expectInvoke(bool, &ctx, .scalar, "is_finite", &.{.{ .array = ints }}, &.{ true, null, true });
    try common.expectInvoke(bool, &ctx, .scalar, "is_nan", &.{.{ .array = ints }}, &.{ false, null, false });
    // Decimals likewise; the null type gives nulls.
    const dec = DataType{ .decimal128 = .{ .precision = 5, .scale = 1 } };
    const dvals = [_]i128{ 15, 0 };
    const dbufs = [_]common.SharedBuffer{ .empty, .fromSlice(std.mem.sliceAsBytes(&dvals)) };
    var d = try ArrayRef.fromBorrowed(allocator, .{ .data_type = dec, .length = 2, .null_count = 0, .buffers = &dbufs });
    defer d.release();
    try common.expectInvoke(bool, &ctx, .scalar, "is_inf", &.{.{ .array = d }}, &.{ false, false });
    try common.expectInvoke(bool, &ctx, .scalar, "is_finite", &.{.{ .array = d }}, &.{ true, true });
    var nb = try common.array_mod.NullBuilder.init(allocator, 2);
    defer nb.deinit();
    try nb.appendNulls(2);
    var null_arr = try nb.finish();
    defer null_arr.release();
    try common.expectInvoke(bool, &ctx, .scalar, "is_nan", &.{.{ .array = null_arr }}, &.{ null, null });
    // Scalars and chunked input.
    try common.expectInvoke(bool, &ctx, .scalar, "is_inf", &.{.{ .scalar = common.scalarOf(f32, std.math.inf(f32)) }}, &.{true});
    try common.expectInvoke(bool, &ctx, .scalar, "is_nan", &.{.{ .scalar = common.scalarOf(f64, null) }}, &.{null});
    var cf = try common.chunkedAt(allocator, f, &.{ 2, 5 });
    defer cf.release();
    var out = try ctx.invokeScalar("is_finite", &.{.{ .chunked = cf }}, none);
    defer out.release();
    try std.testing.expectEqual(@as(usize, 3), out.chunked.numChunks());
    const rows = try common.collect(bool, allocator, out);
    defer allocator.free(rows);
    try common.expectRows(bool, &.{ true, false, false, false, null, true }, rows);

    // is_null: a NaN is null only with nan_is_null; other types ignore the option.
    const nan_null = common.Options{ .null_options = .{ .nan_is_null = true } };
    var h = try common.makeArray(f64, allocator, &.{ 1.0, std.math.nan(f64), null });
    defer h.release();
    try common.expectInvokeWith(bool, &ctx, "is_null", &.{.{ .array = h }}, nan_null, &.{ false, true, true });
    try common.expectInvokeWith(bool, &ctx, "is_null", &.{.{ .array = h }}, .{ .null_options = .{} }, &.{ false, false, true });
    try common.expectInvokeWith(bool, &ctx, "is_null", &.{.{ .array = ints }}, nan_null, &.{ false, true, false });
    try common.expectInvokeWith(bool, &ctx, "is_null", &.{.{ .scalar = common.scalarOf(f32, std.math.nan(f32)) }}, nan_null, &.{true});
    var hc = try common.chunkedAt(allocator, h, &.{1});
    defer hc.release();
    try common.expectInvokeWith(bool, &ctx, "is_null", &.{.{ .chunked = hc }}, nan_null, &.{ false, true, true });

    // Errors: bool has no kernel (as in Arrow); options are checked.
    var flags = try common.makeArray(bool, allocator, &.{true});
    defer flags.release();
    try std.testing.expectError(error.NoMatchingKernel, ctx.invokeScalar("is_nan", &.{.{ .array = flags }}, none));
    try std.testing.expectError(error.InvalidOptions, ctx.invokeScalar("is_nan", &fa, nan_null));
    try std.testing.expectError(error.InvalidOptions, ctx.invokeScalar("is_valid", &fa, nan_null));
}

/// The count aggregate of `d` with `mode`.
fn countOf(ctx: *ExecContext, d: Datum, mode: common.core.CountMode) !i64 {
    var out = try ctx.invokeAggregate("count", &.{d}, .{ .count = .{ .mode = mode } });
    defer out.release();
    return out.scalar.value.i64;
}

/// is_null / is_valid / true_unless_null of `d` follow `nulls`, and count agrees.
fn expectLogicalNulls(ctx: *ExecContext, d: Datum, nulls: []const bool) !void {
    const allocator = std.testing.allocator;
    const expected = try allocator.alloc(?bool, nulls.len);
    defer allocator.free(expected);
    var n: i64 = 0;
    inline for (ops) |op| {
        for (expected, nulls) |*e, is_null| e.* = reference(op, is_null);
        try common.expectInvoke(bool, ctx, .scalar, op, &.{d}, expected);
    }
    for (nulls) |x| n += @intFromBool(x);
    try std.testing.expectEqual(@as(i64, @intCast(nulls.len)) - n, try countOf(ctx, d, .only_valid));
    try std.testing.expectEqual(n, try countOf(ctx, d, .only_null));
    try std.testing.expectEqual(@as(i64, @intCast(nulls.len)), try countOf(ctx, d, .all));
}

test "is_null, is_valid, true_unless_null and count on run-end encoded and union arrays use the children's nulls: sliced, chunked (pyarrow 26 results)" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);

    // run_end_encode(int32 [1, 1, null, 4, 4]): is_null [F, F, T, F, F], count 4.
    var ints = try common.makeArray(i32, allocator, &.{ 1, 1, null, 4, 4 });
    defer ints.release();
    var ree = try ctx.invoke("run_end_encode", .vector, &.{.{ .array = ints }}, none);
    defer ree.release();
    try expectLogicalNulls(&ctx, ree, &.{ false, false, true, false, false });
    var ree_mid = try ree.array.slice(1, 3);
    defer ree_mid.release();
    try expectLogicalNulls(&ctx, .{ .array = ree_mid }, &.{ false, true, false });
    var ree_chunked = try common.chunkedAt(allocator, ree.array, &.{ 2, 3 });
    defer ree_chunked.release();
    try expectLogicalNulls(&ctx, .{ .chunked = ree_chunked }, &.{ false, false, true, false, false });

    // sparse_union<int32, string> type ids [0, 1, 0, 1] over [1, null, null, 4] and
    // ["a", "b", null, null]: rows 2 and 3 select a null.
    const int_type = DataType{ .int32 = {} };
    const string_type = DataType{ .string = {} };
    const fields = [_]common.datatype.Field{ .{ .name = "0", .data_type = &int_type }, .{ .name = "1", .data_type = &string_type } };
    const ids = [_]i8{ 0, 1 };
    var si = try common.makeArray(i32, allocator, &.{ 1, null, null, 4 });
    defer si.release();
    var ss = try common.makeStrings(.string, allocator, &.{ "a", "b", null, null });
    defer ss.release();
    var sb = try common.array_mod.SparseUnionBuilder.init(allocator, .{ .type_ids = &ids, .fields = &fields, .mode = .sparse }, 4);
    defer sb.deinit();
    for ([_]i8{ 0, 1, 0, 1 }) |t| try sb.appendTypeId(t);
    var su = try sb.finish(&.{ si, ss });
    defer su.release();
    try expectLogicalNulls(&ctx, .{ .array = su }, &.{ false, false, true, true });
    var su_tail = try su.slice(1, 3);
    defer su_tail.release();
    try expectLogicalNulls(&ctx, .{ .array = su_tail }, &.{ false, true, true });
    var su_chunked = try common.chunkedAt(allocator, su, &.{1});
    defer su_chunked.release();
    try expectLogicalNulls(&ctx, .{ .chunked = su_chunked }, &.{ false, false, true, true });

    // dense_union, offsets [0, 0, 1, 1] over [1, null] and ["a", null].
    var di = try common.makeArray(i32, allocator, &.{ 1, null });
    defer di.release();
    var ds = try common.makeStrings(.string, allocator, &.{ "a", null });
    defer ds.release();
    var dbld = try common.array_mod.DenseUnionBuilder.init(allocator, .{ .type_ids = &ids, .fields = &fields, .mode = .dense }, 4);
    defer dbld.deinit();
    for ([_]i8{ 0, 1, 0, 1 }, [_]i32{ 0, 0, 1, 1 }) |t, o| try dbld.append(t, o);
    var du = try dbld.finish(&.{ di, ds });
    defer du.release();
    try expectLogicalNulls(&ctx, .{ .array = du }, &.{ false, false, true, true });
    var du_tail = try du.slice(2, 2);
    defer du_tail.release();
    try expectLogicalNulls(&ctx, .{ .array = du_tail }, &.{ true, true });

    // A run-end encoded array over a dense union: the run's selected child decides.
    var ends_b = try common.array_mod.RunEndEncodedBuilder.init(allocator, .{ .bit_width = 32, .signed = true }, &du.data().data_type, 4);
    defer ends_b.deinit();
    for ([_]i64{ 2, 3, 4, 6 }) |e| try ends_b.appendRunEnd(e);
    var nested = try ends_b.finish(du);
    defer nested.release();
    try expectLogicalNulls(&ctx, .{ .array = nested }, &.{ false, false, false, true, true, true });

    // Scalars keep their own null flag.
    try common.expectInvoke(bool, &ctx, .scalar, "is_null", &.{.{ .scalar = Scalar.init(su.data().data_type, .null) }}, &.{true});
}

test "is_finite, is_inf and is_nan on duration: finite, nulls kept, sliced and chunked (pyarrow 26 results)" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    var d = try common.makeTyped(i64, .{ .duration = .{ .unit = .millisecond } }, allocator, &.{ 1, 2, 4, null, -5 });
    defer d.release();
    try common.expectInvoke(bool, &ctx, .scalar, "is_finite", &.{.{ .array = d }}, &.{ true, true, true, null, true });
    try common.expectInvoke(bool, &ctx, .scalar, "is_inf", &.{.{ .array = d }}, &.{ false, false, false, null, false });
    try common.expectInvoke(bool, &ctx, .scalar, "is_nan", &.{.{ .array = d }}, &.{ false, false, false, null, false });
    var tail = try d.slice(3, 2);
    defer tail.release();
    try common.expectInvoke(bool, &ctx, .scalar, "is_finite", &.{.{ .array = tail }}, &.{ null, true });
    var c = try common.chunkedAt(allocator, d, &.{2});
    defer c.release();
    try common.expectInvoke(bool, &ctx, .scalar, "is_nan", &.{.{ .chunked = c }}, &.{ false, false, false, null, false });
}
