const std = @import("std");
const common = @import("test_common.zig");

const ArrayRef = common.ArrayRef;
const Datum = common.Datum;
const ExecContext = common.ExecContext;
const Scalar = common.Scalar;
const dense = common.dense;
const none = common.none;

const ops = [_][]const u8{ "equal", "not_equal", "less", "less_equal", "greater", "greater_equal" };

fn order(comptime op: []const u8, o: std.math.Order) bool {
    const eq = std.mem.eql;
    if (comptime eq(u8, op, "equal")) return o == .eq;
    if (comptime eq(u8, op, "not_equal")) return o != .eq;
    if (comptime eq(u8, op, "less")) return o == .lt;
    if (comptime eq(u8, op, "less_equal")) return o != .gt;
    if (comptime eq(u8, op, "greater")) return o == .gt;
    return o != .lt;
}

/// Per-element reference: IEEE comparisons for floats (any NaN: only not_equal is true).
fn reference(comptime T: type, comptime op: []const u8, x: T, y: T) bool {
    if (T == bool) return order(op, std.math.order(@intFromBool(x), @intFromBool(y)));
    if (@typeInfo(T) == .float and (std.math.isNan(x) or std.math.isNan(y))) return comptime std.mem.eql(u8, op, "not_equal");
    return order(op, std.math.order(x, y));
}

fn expectMatchesReference(comptime T: type, comptime op: []const u8, ctx: *ExecContext, args: []const Datum) !void {
    const allocator = std.testing.allocator;
    const a = try common.collect(T, allocator, args[0]);
    defer allocator.free(a);
    const b = try common.collect(T, allocator, args[1]);
    defer allocator.free(b);
    const len = @max(a.len, b.len);
    const expected = try allocator.alloc(?bool, len);
    defer allocator.free(expected);
    for (expected, 0..) |*e, i| {
        const x = a[if (a.len == 1) 0 else i];
        const y = b[if (b.len == 1) 0 else i];
        e.* = if (x == null or y == null) null else reference(T, op, x.?, y.?);
    }
    try common.expectInvoke(bool, ctx, .scalar, op, args, expected);
}

fn stringRows(allocator: std.mem.Allocator, arr: ArrayRef) ![]?[]const u8 {
    const out = try allocator.alloc(?[]const u8, arr.data().length);
    for (out, 0..) |*o, i| o.* = try common.stringAt(arr.data(), i);
    return out;
}

test "comparison hand-written values for integers, floats with NaN, bools and strings of every layout" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);

    var a = try common.makeArray(i32, allocator, &.{ 1, 2, 3, null, -5 });
    defer a.release();
    var b = try common.makeArray(i32, allocator, &.{ 2, 2, 1, 4, null });
    defer b.release();
    const ab = [_]Datum{ .{ .array = a }, .{ .array = b } };
    try common.expectInvoke(bool, &ctx, .scalar, "equal", &ab, &.{ false, true, false, null, null });
    try common.expectInvoke(bool, &ctx, .scalar, "not_equal", &ab, &.{ true, false, true, null, null });
    try common.expectInvoke(bool, &ctx, .scalar, "less", &ab, &.{ true, false, false, null, null });
    try common.expectInvoke(bool, &ctx, .scalar, "less_equal", &ab, &.{ true, true, false, null, null });
    try common.expectInvoke(bool, &ctx, .scalar, "greater", &ab, &.{ false, false, true, null, null });
    try common.expectInvoke(bool, &ctx, .scalar, "greater_equal", &ab, &.{ false, true, true, null, null });

    const nan = std.math.nan(f64);
    var fa = try common.makeArray(f64, allocator, &.{ nan, nan, 1.0, -0.0, std.math.inf(f64) });
    defer fa.release();
    var fb = try common.makeArray(f64, allocator, &.{ nan, 1.0, nan, 0.0, 1e308 });
    defer fb.release();
    const fab = [_]Datum{ .{ .array = fa }, .{ .array = fb } };
    try common.expectInvoke(bool, &ctx, .scalar, "equal", &fab, &.{ false, false, false, true, false });
    try common.expectInvoke(bool, &ctx, .scalar, "not_equal", &fab, &.{ true, true, true, false, true });
    try common.expectInvoke(bool, &ctx, .scalar, "less_equal", &fab, &.{ false, false, false, true, false });
    try common.expectInvoke(bool, &ctx, .scalar, "greater", &fab, &.{ false, false, false, false, true });

    var ba = try common.makeArray(bool, allocator, &.{ false, false, true, true, null });
    defer ba.release();
    var bb = try common.makeArray(bool, allocator, &.{ false, true, false, true, true });
    defer bb.release();
    const bab = [_]Datum{ .{ .array = ba }, .{ .array = bb } };
    try common.expectInvoke(bool, &ctx, .scalar, "less", &bab, &.{ false, true, false, false, null });
    try common.expectInvoke(bool, &ctx, .scalar, "greater_equal", &bab, &.{ true, false, true, true, null });
    try common.expectInvoke(bool, &ctx, .scalar, "equal", &bab, &.{ true, false, false, true, null });

    const left = [_]?[]const u8{ "a", "ab", "", "same-but-longer-than-12", "b", null, "zz" };
    const right = [_]?[]const u8{ "ab", "a", "a", "same-but-longer-than-12", "abc", "x", "zz-but-longer-than-12" };
    inline for (.{ common.array_mod.StringBuilder, common.array_mod.LargeStringBuilder, common.array_mod.StringViewBuilder }) |Builder| {
        var arrays: [2]ArrayRef = undefined;
        for ([_][]const ?[]const u8{ &left, &right }, 0..) |rows, k| {
            var builder = try Builder.init(allocator, rows.len, 64);
            defer builder.deinit();
            for (rows) |r| if (r) |v| try builder.append(v) else try builder.appendNull();
            arrays[k] = try builder.finish();
        }
        defer for (&arrays) |*x| x.release();
        const sab = [_]Datum{ .{ .array = arrays[0] }, .{ .array = arrays[1] } };
        try common.expectInvoke(bool, &ctx, .scalar, "less", &sab, &.{ true, false, true, false, false, null, true });
        try common.expectInvoke(bool, &ctx, .scalar, "equal", &sab, &.{ false, false, false, true, false, null, false });
        try common.expectInvoke(bool, &ctx, .scalar, "greater_equal", &sab, &.{ false, true, false, true, true, null, false });
    }
}

test "comparison oracle matches a per-element loop on random sliced inputs with nulls for every type" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    var prng = std.Random.DefaultPrng.init(0xC0FE);
    const rng = prng.random();
    inline for (dense.numeric_types ++ [_]type{bool}) |T| {
        var a = try common.randomArray(T, allocator, rng, 333, 5, 20, .{ .small = true });
        defer a.release();
        var b = try common.randomArray(T, allocator, rng, 333, 11, 20, .{ .small = true });
        defer b.release();
        const s = common.scalarOf(T, common.randomValue(T, rng));
        inline for (ops) |op| {
            try expectMatchesReference(T, op, &ctx, &.{ .{ .array = a }, .{ .array = b } });
            try expectMatchesReference(T, op, &ctx, &.{ .{ .array = a }, .{ .array = a } });
            try expectMatchesReference(T, op, &ctx, &.{ .{ .array = a }, .{ .scalar = s } });
            try expectMatchesReference(T, op, &ctx, &.{ .{ .scalar = s }, .{ .array = b } });
        }
    }
    inline for (.{ .string, .large_string, .string_view }) |tag| {
        var a = try common.randomStrings(tag, allocator, rng, 401, 9, 15);
        defer a.release();
        var b = try common.randomStrings(tag, allocator, rng, 401, 2, 15);
        defer b.release();
        const left = try stringRows(allocator, a);
        defer allocator.free(left);
        const right = try stringRows(allocator, b);
        defer allocator.free(right);
        inline for (ops) |op| {
            const expected = try allocator.alloc(?bool, left.len);
            defer allocator.free(expected);
            for (expected, left, right) |*e, x, y| e.* = if (x == null or y == null) null else order(op, std.mem.order(u8, x.?, y.?));
            try common.expectInvoke(bool, &ctx, .scalar, op, &.{ .{ .array = a }, .{ .array = b } }, expected);
            // Scalar operand: compare every row with "ab".
            const s = Scalar.init(@unionInit(common.DataType, @tagName(tag), {}), .{ .string = "ab" });
            for (expected, left) |*e, x| e.* = if (x == null) null else order(op, std.mem.order(u8, x.?, "ab"));
            try common.expectInvoke(bool, &ctx, .scalar, op, &.{ .{ .array = a }, .{ .scalar = s } }, expected);
        }
    }
}

test "comparison chunked inputs with misaligned chunks give a chunked result equal to the array result" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    var prng = std.Random.DefaultPrng.init(0xC0C0);
    const rng = prng.random();
    var ints_a = try common.randomArray(i64, allocator, rng, 700, 1, 10, .{ .small = true });
    defer ints_a.release();
    var ints_b = try common.randomArray(i64, allocator, rng, 700, 6, 10, .{ .small = true });
    defer ints_b.release();
    var strs_a = try common.randomStrings(.string, allocator, rng, 700, 3, 10);
    defer strs_a.release();
    var strs_b = try common.randomStrings(.string, allocator, rng, 700, 0, 10);
    defer strs_b.release();
    var bools_a = try common.randomArray(bool, allocator, rng, 700, 3, 10, .{});
    defer bools_a.release();
    var bools_b = try common.randomArray(bool, allocator, rng, 700, 7, 10, .{});
    defer bools_b.release();
    for ([_][2]ArrayRef{ .{ ints_a, ints_b }, .{ strs_a, strs_b }, .{ bools_a, bools_b } }) |pair| {
        var ca = try common.chunkedAt(allocator, pair[0], &.{ 3, 129, 640 });
        defer ca.release();
        var cb = try common.chunkedAt(allocator, pair[1], &.{ 64, 129, 513 });
        defer cb.release();
        inline for (ops) |op| {
            var whole = try ctx.invokeScalar(op, &.{ .{ .array = pair[0] }, .{ .array = pair[1] } }, none);
            defer whole.release();
            const expected = try common.collect(bool, allocator, whole);
            defer allocator.free(expected);
            var chunked = try ctx.invokeScalar(op, &.{ .{ .chunked = ca }, .{ .chunked = cb } }, none);
            defer chunked.release();
            // Boundaries 3, 64, 129, 513, 640.
            try std.testing.expectEqual(@as(usize, 6), chunked.chunked.numChunks());
            const got = try common.collect(bool, allocator, chunked);
            defer allocator.free(got);
            try common.expectRows(bool, expected, got);
        }
    }
}

test "comparison scalar broadcast, scalar-scalar and null scalars" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    var a = try common.makeArray(u16, allocator, &.{ 1, 5, null, 9 });
    defer a.release();
    const five = common.scalarOf(u16, 5);
    try common.expectInvoke(bool, &ctx, .scalar, "less", &.{ .{ .array = a }, .{ .scalar = five } }, &.{ true, false, null, false });
    try common.expectInvoke(bool, &ctx, .scalar, "less", &.{ .{ .scalar = five }, .{ .array = a } }, &.{ false, false, null, true });
    try common.expectInvoke(bool, &ctx, .scalar, "equal", &.{ .{ .array = a }, .{ .scalar = common.scalarOf(u16, null) } }, &.{ null, null, null, null });

    var out = try ctx.invokeScalar("greater_equal", &.{ .{ .scalar = five }, .{ .scalar = common.scalarOf(u16, 5) } }, none);
    defer out.release();
    try std.testing.expect(out == .scalar);
    try std.testing.expect(out.scalar.data_type == .bool);
    try std.testing.expect(out.scalar.value.bool);
    var null_out = try ctx.invokeScalar("equal", &.{ .{ .scalar = five }, .{ .scalar = common.scalarOf(u16, null) } }, none);
    defer null_out.release();
    try std.testing.expect(null_out.scalar.isNull());

    const s = Scalar.init(.{ .string = {} }, .{ .string = "m" });
    var prng = std.Random.DefaultPrng.init(1);
    var strs = try common.randomStrings(.string, allocator, prng.random(), 0, 0, 0);
    defer strs.release();
    try common.expectInvoke(bool, &ctx, .scalar, "less", &.{ .{ .array = strs }, .{ .scalar = s } }, &.{});
    var str_out = try ctx.invokeScalar("less", &.{ .{ .scalar = s }, .{ .scalar = Scalar.init(.{ .string = {} }, .{ .string = "n" }) } }, none);
    defer str_out.release();
    try std.testing.expect(str_out.scalar.value.bool);
}

test "comparison empty and all-null inputs" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    var empty = try common.makeArray(f32, allocator, &.{});
    defer empty.release();
    var nulls = try common.makeArray(f32, allocator, &.{ null, null, null });
    defer nulls.release();
    var bool_nulls = try common.makeArray(bool, allocator, &.{ null, null });
    defer bool_nulls.release();
    inline for (ops) |op| {
        try common.expectInvoke(bool, &ctx, .scalar, op, &.{ .{ .array = empty }, .{ .array = empty } }, &.{});
        try common.expectInvoke(bool, &ctx, .scalar, op, &.{ .{ .array = nulls }, .{ .array = nulls } }, &.{ null, null, null });
        try common.expectInvoke(bool, &ctx, .scalar, op, &.{ .{ .array = bool_nulls }, .{ .scalar = common.scalarOf(bool, true) } }, &.{ null, null });
    }
}

test "comparison errors: mismatched types, arity, options, lengths and malformed scalars" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    var i = try common.makeArray(i32, allocator, &.{ 1, 2 });
    defer i.release();
    var l = try common.makeArray(i64, allocator, &.{ 1, 2 });
    defer l.release();
    try std.testing.expectError(error.NoMatchingKernel, ctx.invokeScalar("less", &.{ .{ .array = i }, .{ .array = l } }, none));
    var prng = std.Random.DefaultPrng.init(2);
    var s = try common.randomStrings(.string, allocator, prng.random(), 2, 0, 0);
    defer s.release();
    var ls = try common.randomStrings(.large_string, allocator, prng.random(), 2, 0, 0);
    defer ls.release();
    try std.testing.expectError(error.NoMatchingKernel, ctx.invokeScalar("equal", &.{ .{ .array = s }, .{ .array = ls } }, none));
    try std.testing.expectError(error.InvalidArity, ctx.invokeScalar("equal", &.{.{ .array = i }}, none));
    try std.testing.expectError(error.InvalidOptions, ctx.invokeScalar("equal", &.{ .{ .array = i }, .{ .array = i } }, .{ .sort = .{} }));
    var three = try common.makeArray(i32, allocator, &.{ 1, 2, 3 });
    defer three.release();
    try std.testing.expectError(error.InvalidInput, ctx.invokeScalar("less", &.{ .{ .array = i }, .{ .array = three } }, none));
    const bad = Scalar.init(.{ .string = {} }, .{ .i32 = 7 });
    try std.testing.expectError(error.InvalidInput, ctx.invokeScalar("less", &.{ .{ .array = s }, .{ .scalar = bad } }, none));
    const bad_bool = Scalar.init(.{ .bool = {} }, .{ .i32 = 1 });
    var b = try common.makeArray(bool, allocator, &.{ true, false });
    defer b.release();
    try std.testing.expectError(error.InvalidInput, ctx.invokeScalar("equal", &.{ .{ .array = b }, .{ .scalar = bad_bool } }, none));
    // Offsets past the data buffer fail validateLayout instead of being sliced.
    const offsets = [_]i32{ 0, 1, 40 };
    const bufs = [_]common.SharedBuffer{ .empty, .fromSlice(std.mem.sliceAsBytes(&offsets)), .fromSlice("ab") };
    var malformed = try ArrayRef.fromBorrowed(allocator, .{ .data_type = .{ .string = {} }, .length = 2, .null_count = 0, .buffers = &bufs });
    defer malformed.release();
    try std.testing.expectError(error.InvalidInput, ctx.invokeScalar("less", &.{ .{ .array = malformed }, .{ .array = s } }, none));
}
