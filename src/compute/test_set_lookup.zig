const std = @import("std");
const common = @import("test_common.zig");

const ArrayRef = common.ArrayRef;
const DataType = common.DataType;
const Datum = common.Datum;
const ExecContext = common.ExecContext;
const Options = common.Options;
const Scalar = common.Scalar;
const NullMatchingBehavior = common.core.NullMatchingBehavior;

fn lookup(value_set: Datum, b: NullMatchingBehavior) Options {
    return .{ .set_lookup = .{ .value_set = value_set, .null_matching_behavior = b } };
}

test "set lookup hand-written values for every null matching behavior (Arrow C++ results)" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);

    var v = try common.makeArray(i64, allocator, &.{ 1, 2, null, 3, 2 });
    defer v.release();
    var vs = try common.makeArray(i64, allocator, &.{ 2, null, 5, 2 });
    defer vs.release();
    var vs_no_null = try common.makeArray(i64, allocator, &.{ 2, 5 });
    defer vs_no_null.release();
    const args = [_]Datum{.{ .array = v }};
    const set: Datum = .{ .array = vs };
    const plain: Datum = .{ .array = vs_no_null };
    // match / skip from pyarrow 26 (skip_nulls false / true); emit_null and inconclusive from the
    // Arrow C++ SetLookupOptions documentation.
    try common.expectInvokeWith(bool, &ctx, "is_in", &args, lookup(set, .match), &.{ false, true, true, false, true });
    try common.expectInvokeWith(i32, &ctx, "index_in", &args, lookup(set, .match), &.{ null, 0, 1, null, 0 });
    try common.expectInvokeWith(bool, &ctx, "is_in", &args, lookup(set, .skip), &.{ false, true, false, false, true });
    try common.expectInvokeWith(i32, &ctx, "index_in", &args, lookup(set, .skip), &.{ null, 0, null, null, 0 });
    try common.expectInvokeWith(bool, &ctx, "is_in", &args, lookup(set, .emit_null), &.{ false, true, null, false, true });
    try common.expectInvokeWith(i32, &ctx, "index_in", &args, lookup(set, .emit_null), &.{ null, 0, null, null, 0 });
    try common.expectInvokeWith(bool, &ctx, "is_in", &args, lookup(set, .inconclusive), &.{ null, true, null, null, true });
    try common.expectInvokeWith(i32, &ctx, "index_in", &args, lookup(set, .inconclusive), &.{ null, 0, null, null, 0 });
    try common.expectInvokeWith(bool, &ctx, "is_in", &args, lookup(plain, .match), &.{ false, true, false, false, true });
    try common.expectInvokeWith(i32, &ctx, "index_in", &args, lookup(plain, .match), &.{ null, 0, null, null, 0 });
    try common.expectInvokeWith(bool, &ctx, "is_in", &args, lookup(plain, .inconclusive), &.{ false, true, null, false, true });

    // Floats: NaN matches NaN, -0.0 does not match 0.0 (pyarrow 26).
    var f = try common.makeArray(f64, allocator, &.{ std.math.nan(f64), -0.0, 0.0, 1.5 });
    defer f.release();
    var fs = try common.makeArray(f64, allocator, &.{ -std.math.nan(f64), 0.0 });
    defer fs.release();
    try common.expectInvokeWith(bool, &ctx, "is_in", &.{.{ .array = f }}, lookup(.{ .array = fs }, .match), &.{ true, false, true, false });
    var f32s = try common.makeArray(f32, allocator, &.{ std.math.nan(f32), 2 });
    defer f32s.release();
    try common.expectInvokeWith(i32, &ctx, "index_in", &.{.{ .array = f32s }}, lookup(.{ .array = f32s }, .match), &.{ 0, 1 });

    // Strings of every layout: the position of the first occurrence.
    inline for (.{ .string, .large_string, .string_view, .binary, .large_binary, .binary_view }) |tag| {
        var s = try common.makeStrings(tag, allocator, &.{ "a", "b", "c", null, "a long value past twelve" });
        defer s.release();
        var ss = try common.makeStrings(tag, allocator, &.{ "c", "a", "c", "a long value past twelve" });
        defer ss.release();
        try common.expectInvokeWith(i32, &ctx, "index_in", &.{.{ .array = s }}, lookup(.{ .array = ss }, .match), &.{ 1, null, 0, null, 3 });
    }
    var b = try common.makeArray(bool, allocator, &.{ true, false, null });
    defer b.release();
    var bs = try common.makeArray(bool, allocator, &.{true});
    defer bs.release();
    try common.expectInvokeWith(bool, &ctx, "is_in", &.{.{ .array = b }}, lookup(.{ .array = bs }, .match), &.{ true, false, false });

    // Timestamps and dates, and a chunked value set (positions count across its chunks).
    const ts = DataType{ .timestamp = .{ .unit = .millisecond, .timezone = "UTC" } };
    var t = try common.makeTyped(i64, ts, allocator, &.{ 10, 20, 30 });
    defer t.release();
    var tset = try common.makeTyped(i64, ts, allocator, &.{ 30, 99, 10 });
    defer tset.release();
    var tchunks = try common.chunkedAt(allocator, tset, &.{1});
    defer tchunks.release();
    try common.expectInvokeWith(i32, &ctx, "index_in", &.{.{ .array = t }}, lookup(.{ .chunked = tchunks }, .match), &.{ 2, null, 0 });
    var d = try common.makeTyped(i32, .{ .date32 = {} }, allocator, &.{ 1, 2 });
    defer d.release();
    try common.expectInvokeWith(bool, &ctx, "is_in", &.{.{ .array = d }}, lookup(.{ .array = d }, .skip), &.{ true, true });
}

/// Reference lookup: the first position of `x` in `set` (null matches null), or null.
fn refIndex(comptime T: type, set: []const ?T, x: ?T) ?usize {
    for (set, 0..) |s, i| {
        if (x == null and s == null) return i;
        if (x != null and s != null and s.? == x.?) return i;
    }
    return null;
}

test "set lookup oracle matches a linear search on random sliced inputs with nulls" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    var prng = std.Random.DefaultPrng.init(0x5E71);
    const rng = prng.random();

    var vals: [520]?i32 = undefined;
    for (&vals) |*x| x.* = if (rng.uintLessThan(u8, 8) == 0) null else rng.intRangeAtMost(i32, -40, 40);
    var set_vals: [33]?i32 = undefined;
    for (&set_vals) |*x| x.* = if (rng.uintLessThan(u8, 16) == 0) null else rng.intRangeAtMost(i32, -40, 40);
    var full = try common.makeArray(i32, allocator, &vals);
    defer full.release();
    var arr = try full.slice(13, 500);
    defer arr.release();
    var set_full = try common.makeArray(i32, allocator, &set_vals);
    defer set_full.release();
    var set_arr = try set_full.slice(1, 32);
    defer set_arr.release();
    const set = set_vals[1..];
    var has_null = false;
    for (set) |s| has_null = has_null or s == null;
    for ([_]NullMatchingBehavior{ .match, .skip, .emit_null, .inconclusive }) |behavior| {
        var want_in: [500]?bool = undefined;
        var want_index: [500]?i32 = undefined;
        for (vals[13..][0..500], &want_in, &want_index) |x, *wi, *wx| {
            const found = if (x == null and behavior != .match) null else refIndex(i32, set, x);
            wx.* = if (found) |p| @intCast(p) else null;
            wi.* = if (x == null) switch (behavior) {
                .match => found != null,
                .skip => false,
                else => null,
            } else if (found != null) true else if (behavior == .inconclusive and has_null) null else false;
        }
        try common.expectInvokeWith(bool, &ctx, "is_in", &.{.{ .array = arr }}, lookup(.{ .array = set_arr }, behavior), &want_in);
        try common.expectInvokeWith(i32, &ctx, "index_in", &.{.{ .array = arr }}, lookup(.{ .array = set_arr }, behavior), &want_index);
    }
}

test "set lookup on chunked inputs with misaligned chunks equals the array result" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    var prng = std.Random.DefaultPrng.init(0x5E72);
    const rng = prng.random();
    const rows = try common.randomStrings(.string_view, allocator, rng, 300, 3, 10);
    defer {
        var r = rows;
        r.release();
    }
    var set = try rows.slice(0, 40);
    defer set.release();
    var chunked = try common.chunkedAt(allocator, rows, &.{ 7, 128, 255 });
    defer chunked.release();
    inline for (.{ "is_in", "index_in" }) |name| {
        const o = lookup(.{ .array = set }, .match);
        var whole = try ctx.invokeScalar(name, &.{.{ .array = rows }}, o);
        defer whole.release();
        var out = try ctx.invokeScalar(name, &.{.{ .chunked = chunked }}, o);
        defer out.release();
        try std.testing.expectEqual(@as(usize, 4), out.chunked.numChunks());
        const T = if (comptime std.mem.eql(u8, name, "is_in")) bool else i32;
        const expected = try common.collect(T, allocator, whole);
        defer allocator.free(expected);
        const got = try common.collect(T, allocator, out);
        defer allocator.free(got);
        try common.expectRows(T, expected, got);
    }
}

test "set lookup on scalars, empty value sets, empty and all-null inputs" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    var vs = try common.makeArray(i32, allocator, &.{ 4, null });
    defer vs.release();
    const set: Datum = .{ .array = vs };
    var hit = try ctx.invokeScalar("index_in", &.{.{ .scalar = common.scalarOf(i32, 4) }}, lookup(set, .match));
    defer hit.release();
    try std.testing.expectEqual(@as(i32, 0), hit.scalar.value.i32);
    // A null scalar matches the null of the value set under `match`, is null under emit_null.
    var null_in = try ctx.invokeScalar("is_in", &.{.{ .scalar = common.scalarOf(i32, null) }}, lookup(set, .match));
    defer null_in.release();
    try std.testing.expect(null_in.scalar.value.bool);
    var null_index = try ctx.invokeScalar("index_in", &.{.{ .scalar = common.scalarOf(i32, null) }}, lookup(set, .match));
    defer null_index.release();
    try std.testing.expectEqual(@as(i32, 1), null_index.scalar.value.i32);
    var emit = try ctx.invokeScalar("is_in", &.{.{ .scalar = common.scalarOf(i32, null) }}, lookup(set, .emit_null));
    defer emit.release();
    try std.testing.expect(emit.scalar.isNull());

    var empty_set = try common.makeArray(i32, allocator, &.{});
    defer empty_set.release();
    var x = try common.makeArray(i32, allocator, &.{ 1, null });
    defer x.release();
    try common.expectInvokeWith(bool, &ctx, "is_in", &.{.{ .array = x }}, lookup(.{ .array = empty_set }, .match), &.{ false, false });
    try common.expectInvokeWith(i32, &ctx, "index_in", &.{.{ .array = x }}, lookup(.{ .array = empty_set }, .match), &.{ null, null });
    try common.expectInvokeWith(bool, &ctx, "is_in", &.{.{ .array = empty_set }}, lookup(set, .match), &.{});
    var nulls = try common.makeArray(i32, allocator, &.{ null, null });
    defer nulls.release();
    try common.expectInvokeWith(bool, &ctx, "is_in", &.{.{ .array = nulls }}, lookup(set, .skip), &.{ false, false });
    try common.expectInvokeWith(i32, &ctx, "index_in", &.{.{ .array = nulls }}, lookup(set, .match), &.{ 1, 1 });
}

test "set lookup errors: options, value set type, unsupported types and malformed input" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    var x = try common.makeArray(i32, allocator, &.{1});
    defer x.release();
    var y = try common.makeArray(i64, allocator, &.{1});
    defer y.release();
    const args = [_]Datum{.{ .array = x }};
    try std.testing.expectError(error.InvalidOptions, ctx.invokeScalar("is_in", &args, common.none));
    try std.testing.expectError(error.InvalidOptions, ctx.invokeScalar("is_in", &args, lookup(.{ .scalar = common.scalarOf(i32, 1) }, .match)));
    // Arrow casts the value set; here it must have the input's type.
    try std.testing.expectError(error.InvalidOptions, ctx.invokeScalar("index_in", &args, lookup(.{ .array = y }, .match)));
    var n = try common.datumBuildNullLike(.{ .null = {} }, 2);
    defer n.release();
    try std.testing.expectError(error.NoMatchingKernel, ctx.invokeScalar("is_in", &.{n}, lookup(n, .match)));
    try std.testing.expectError(error.InvalidArity, ctx.invokeScalar("is_in", &.{ args[0], args[0] }, lookup(args[0], .match)));
    const short = [_]i32{1};
    const bufs = [_]common.SharedBuffer{ .empty, .fromSlice(std.mem.sliceAsBytes(&short)) };
    var malformed = try ArrayRef.fromBorrowed(allocator, .{ .data_type = .{ .int32 = {} }, .length = 2, .null_count = 0, .buffers = &bufs });
    defer malformed.release();
    try std.testing.expectError(error.InvalidInput, ctx.invokeScalar("is_in", &.{.{ .array = malformed }}, lookup(args[0], .match)));
    try std.testing.expectError(error.InvalidInput, ctx.invokeScalar("is_in", &args, lookup(.{ .array = malformed }, .match)));
}
