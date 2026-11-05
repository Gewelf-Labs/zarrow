const std = @import("std");
const common = @import("test_common.zig");

const ArrayRef = common.ArrayRef;
const Datum = common.Datum;
const ExecContext = common.ExecContext;
const Scalar = common.Scalar;
const dense = common.dense;
const none = common.none;
const DataType = common.DataType;
const Options = common.Options;
const TypeTag = std.meta.Tag(DataType);

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
    // Numeric operands are cast to their common type; a string never compares with a number.
    try common.expectInvoke(bool, &ctx, .scalar, "less", &.{ .{ .array = i }, .{ .array = l } }, &.{ false, false });
    var prng = std.Random.DefaultPrng.init(2);
    var s = try common.randomStrings(.string, allocator, prng.random(), 2, 0, 0);
    defer s.release();
    var ls = try common.randomStrings(.large_string, allocator, prng.random(), 2, 0, 0);
    defer ls.release();
    try std.testing.expectError(error.NoMatchingKernel, ctx.invokeScalar("equal", &.{ .{ .array = s }, .{ .array = i } }, none));
    try std.testing.expectError(error.NoMatchingKernel, ctx.invokeScalar("less", &.{ .{ .array = ls }, .{ .array = l } }, none));
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

/// Borrowed array of `dt` over `values` (all valid): temporal, decimal and fixed-size test inputs.
fn typedArray(allocator: std.mem.Allocator, dt: DataType, values: []const u8, len: usize) !ArrayRef {
    const bufs = [_]common.SharedBuffer{ .empty, .fromSlice(values) };
    return ArrayRef.fromBorrowed(allocator, .{ .data_type = dt, .length = len, .null_count = 0, .buffers = &bufs });
}

test "comparison implicit casts and binary, fixed-size, temporal and decimal operands" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);

    // int8 vs uint8 compares as int16 (pyarrow 26: less(-1, 255) is true).
    var neg = try common.makeArray(i8, allocator, &.{ -1, 5, null });
    defer neg.release();
    var big = try common.makeArray(u8, allocator, &.{ 255, 5, 1 });
    defer big.release();
    try common.expectInvoke(bool, &ctx, .scalar, "less", &.{ .{ .array = neg }, .{ .array = big } }, &.{ true, false, null });
    try common.expectInvoke(bool, &ctx, .scalar, "equal", &.{ .{ .array = big }, .{ .scalar = common.scalarOf(f32, 5.0) } }, &.{ false, true, false });
    var huge = try common.makeArray(u64, allocator, &.{1 << 63});
    defer huge.release();
    try std.testing.expectError(error.InvalidCast, ctx.invokeScalar("equal", &.{ .{ .array = huge }, .{ .scalar = common.scalarOf(i64, 1) } }, none));

    // Binary layouts compare by bytes like strings.
    inline for (.{ TypeTag.binary, TypeTag.large_binary, TypeTag.binary_view }) |tag| {
        var x = try common.makeStrings(tag, allocator, &.{ "\x00\xff", "abc", null, "a long binary value past twelve" });
        defer x.release();
        var y = try common.makeStrings(tag, allocator, &.{ "\x00", "abd", "q", "a long binary value past twelve" });
        defer y.release();
        const xy = [_]Datum{ .{ .array = x }, .{ .array = y } };
        try common.expectInvoke(bool, &ctx, .scalar, "greater", &xy, &.{ true, false, null, false });
        try common.expectInvoke(bool, &ctx, .scalar, "equal", &xy, &.{ false, false, null, true });
    }

    // fixed_size_binary(3), sliced, against an array and a scalar.
    const fsb = DataType{ .fixed_size_binary = .{ .byte_width = 3 } };
    var f_full = try typedArray(allocator, fsb, "zzzabcabdxyz", 4);
    defer f_full.release();
    var f = try f_full.slice(1, 3);
    defer f.release();
    var g = try typedArray(allocator, fsb, "abcabcxyz", 3);
    defer g.release();
    try common.expectInvoke(bool, &ctx, .scalar, "less_equal", &.{ .{ .array = f }, .{ .array = g } }, &.{ true, false, true });
    try common.expectInvoke(bool, &ctx, .scalar, "not_equal", &.{ .{ .array = f }, .{ .scalar = Scalar.init(fsb, .{ .binary = "abd" }) } }, &.{ true, false, true });
    try std.testing.expectError(error.InvalidInput, ctx.invokeScalar("equal", &.{ .{ .array = f }, .{ .scalar = Scalar.init(fsb, .{ .binary = "ab" }) } }, none));
    var w4 = try typedArray(allocator, .{ .fixed_size_binary = .{ .byte_width = 4 } }, "abcdabcdabcd", 3);
    defer w4.release();
    try std.testing.expectError(error.NoMatchingKernel, ctx.invokeScalar("equal", &.{ .{ .array = g }, .{ .array = w4 } }, none));

    // Temporal types compare their stored integers (on a common unit when they differ).
    var d1 = try common.makeTyped(i32, .{ .date32 = {} }, allocator, &.{ -1, 0, null, 20000 });
    defer d1.release();
    var d2 = try common.makeTyped(i32, .{ .date32 = {} }, allocator, &.{ 0, 0, 5, 19999 });
    defer d2.release();
    try common.expectInvoke(bool, &ctx, .scalar, "less", &.{ .{ .array = d1 }, .{ .array = d2 } }, &.{ true, false, null, false });
    try common.expectInvoke(bool, &ctx, .scalar, "greater_equal", &.{ .{ .array = d1 }, .{ .scalar = Scalar.init(.{ .date32 = {} }, .{ .date32 = 0 }) } }, &.{ false, true, null, true });
    const ts_ms = DataType{ .timestamp = .{ .unit = .millisecond, .timezone = null } };
    var t1 = try common.makeTyped(i64, ts_ms, allocator, &.{ 1000, -5, 7 });
    defer t1.release();
    var t2 = try common.makeTyped(i64, ts_ms, allocator, &.{ 1000, 5, null });
    defer t2.release();
    try common.expectInvoke(bool, &ctx, .scalar, "equal", &.{ .{ .array = t1 }, .{ .array = t2 } }, &.{ true, false, null });
    var t_s = try common.makeTyped(i64, .{ .timestamp = .{ .unit = .second, .timezone = null } }, allocator, &.{ 1, 2, 3 });
    defer t_s.release();
    // Different units compare on the finer one (Arrow's CommonTemporal): 1 s == 1000 ms.
    try common.expectInvoke(bool, &ctx, .scalar, "equal", &.{ .{ .array = t1 }, .{ .array = t_s } }, &.{ true, false, false });
    inline for (.{ DataType{ .time32 = .{ .unit = .millisecond } }, DataType{ .time64 = .{ .unit = .nanosecond } }, DataType{ .duration = .{ .unit = .second } }, DataType{ .date64 = {} } }) |dt| {
        const V = if (dt == .time32) i32 else i64;
        var x = try common.makeTyped(V, dt, allocator, &.{ 3, null, -2 });
        defer x.release();
        var y = try common.makeTyped(V, dt, allocator, &.{ 2, 2, -2 });
        defer y.release();
        try common.expectInvoke(bool, &ctx, .scalar, "greater", &.{ .{ .array = x }, .{ .array = y } }, &.{ true, null, false });
    }

    // Decimals with the same precision and scale compare their unscaled integers.
    const dec = DataType{ .decimal128 = .{ .precision = 10, .scale = 2 } };
    const dvals = [_]i128{ -150, 0, 99999 };
    var dx = try typedArray(allocator, dec, std.mem.sliceAsBytes(&dvals), 3);
    defer dx.release();
    try common.expectInvoke(bool, &ctx, .scalar, "less", &.{ .{ .array = dx }, .{ .scalar = Scalar.init(dec, .{ .decimal128 = 0 }) } }, &.{ true, false, false });
    const dec256 = DataType{ .decimal256 = .{ .precision = 40, .scale = 0 } };
    const wvals = [_]i256{ -(1 << 200), 7 };
    var wx = try typedArray(allocator, dec256, std.mem.sliceAsBytes(&wvals), 2);
    defer wx.release();
    try common.expectInvoke(bool, &ctx, .scalar, "less", &.{ .{ .array = wx }, .{ .scalar = Scalar.init(dec256, .{ .decimal256 = 0 }) } }, &.{ true, false });
    const other_scale = DataType{ .decimal128 = .{ .precision = 10, .scale = 3 } };
    var dy = try typedArray(allocator, other_scale, std.mem.sliceAsBytes(&dvals), 3);
    defer dy.release();
    // Different scales compare by exact value: -1.50 vs -0.150, 0 vs 0, 999.99 vs 99.999.
    try common.expectInvoke(bool, &ctx, .scalar, "less", &.{ .{ .array = dx }, .{ .array = dy } }, &.{ true, false, false });
}

fn withNulls(skip: bool) Options {
    return .{ .element_wise_aggregate = .{ .skip_nulls = skip } };
}

test "min_element_wise and max_element_wise: nulls, NaN, signed zeros, strings, promotion (Arrow C++ results)" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);

    var a = try common.makeArray(i64, allocator, &.{ 1, null, 3, null });
    defer a.release();
    var b = try common.makeArray(i64, allocator, &.{ 2, 2, null, null });
    defer b.release();
    const ab = [_]Datum{ .{ .array = a }, .{ .array = b } };
    try common.expectInvokeWith(i64, &ctx, "min_element_wise", &ab, none, &.{ 1, 2, 3, null });
    try common.expectInvokeWith(i64, &ctx, "max_element_wise", &ab, withNulls(true), &.{ 2, 2, 3, null });
    try common.expectInvokeWith(i64, &ctx, "min_element_wise", &ab, withNulls(false), &.{ 1, null, null, null });
    // One operand is the identity; a scalar broadcasts; a null scalar is skipped or poisons.
    try common.expectInvokeWith(i64, &ctx, "max_element_wise", &.{.{ .array = a }}, none, &.{ 1, null, 3, null });
    try common.expectInvokeWith(i64, &ctx, "min_element_wise", &.{ .{ .array = a }, .{ .scalar = common.scalarOf(i64, 2) } }, none, &.{ 1, 2, 2, 2 });
    try common.expectInvokeWith(i64, &ctx, "min_element_wise", &.{ .{ .array = a }, .{ .scalar = common.scalarOf(i64, null) } }, none, &.{ 1, null, 3, null });
    try common.expectInvokeWith(i64, &ctx, "min_element_wise", &.{ .{ .array = a }, .{ .scalar = common.scalarOf(i64, null) } }, withNulls(false), &.{ null, null, null, null });
    // int32 with int64 gives int64 (pyarrow 26); three operands.
    var c = try common.makeArray(i32, allocator, &.{ 0, 9, -4, 8 });
    defer c.release();
    var mixed = try ctx.invokeScalar("max_element_wise", &.{ .{ .array = c }, .{ .array = a }, .{ .array = b } }, none);
    defer mixed.release();
    try std.testing.expectEqual(TypeTag.int64, std.meta.activeTag(mixed.dataType()));
    const got = try common.collect(i64, allocator, mixed);
    defer allocator.free(got);
    try common.expectRows(i64, &.{ 2, 9, 3, 8 }, got);

    // NaN is left out unless every value is NaN; -0.0 < 0.0 (pyarrow 26).
    const nan = std.math.nan(f64);
    var x = try common.makeArray(f64, allocator, &.{ 1.0, nan, nan, null, -0.0, 0.0 });
    defer x.release();
    var y = try common.makeArray(f64, allocator, &.{ nan, 2.0, nan, nan, 0.0, -0.0 });
    defer y.release();
    const xy = [_]Datum{ .{ .array = x }, .{ .array = y } };
    try common.expectInvokeWith(f64, &ctx, "max_element_wise", &xy, none, &.{ 1.0, 2.0, nan, nan, 0.0, 0.0 });
    var mins = try ctx.invokeScalar("min_element_wise", &xy, none);
    defer mins.release();
    const min_rows = try common.collect(f64, allocator, mins);
    defer allocator.free(min_rows);
    try common.expectRows(f64, &.{ 1.0, 2.0, nan, nan, -0.0, -0.0 }, min_rows);
    try std.testing.expect(std.math.signbit(min_rows[4].?) and std.math.signbit(min_rows[5].?));
    var maxs = try ctx.invokeScalar("max_element_wise", &xy, none);
    defer maxs.release();
    const max_rows = try common.collect(f64, allocator, maxs);
    defer allocator.free(max_rows);
    try std.testing.expect(!std.math.signbit(max_rows[4].?) and !std.math.signbit(max_rows[5].?));

    // Strings and binaries by bytes, every layout.
    inline for (.{ TypeTag.string, TypeTag.large_string, TypeTag.string_view, TypeTag.binary }) |tag| {
        var s = try common.makeStrings(tag, allocator, &.{ "b", null, "a", "a value longer than twelve" });
        defer s.release();
        var t = try common.makeStrings(tag, allocator, &.{ "a", "c", null, "a value longer than twelvf" });
        defer t.release();
        const st = [_]Datum{ .{ .array = s }, .{ .array = t } };
        try common.expectInvokeStrings(&ctx, "min_element_wise", &st, none, &.{ "a", "c", "a", "a value longer than twelve" });
        try common.expectInvokeStrings(&ctx, "max_element_wise", &st, none, &.{ "b", "c", "a", "a value longer than twelvf" });
        try common.expectInvokeStrings(&ctx, "max_element_wise", &st, withNulls(false), &.{ "b", null, null, "a value longer than twelvf" });
    }

    // Temporal operands keep their type.
    var d1 = try common.makeTyped(i32, .{ .date32 = {} }, allocator, &.{ 5, null, -3 });
    defer d1.release();
    var d2 = try common.makeTyped(i32, .{ .date32 = {} }, allocator, &.{ 4, 1, null });
    defer d2.release();
    var dmin = try ctx.invokeScalar("min_element_wise", &.{ .{ .array = d1 }, .{ .array = d2 } }, none);
    defer dmin.release();
    try std.testing.expect(dmin.dataType() == .date32);
    const drows = try common.collect(i32, allocator, dmin);
    defer allocator.free(drows);
    try common.expectRows(i32, &.{ 4, 1, -3 }, drows);

    // Chunked operands equal the array result.
    var prng = std.Random.DefaultPrng.init(0xE1E);
    var r1 = try common.randomArray(i16, allocator, prng.random(), 200, 3, 30, .{});
    defer r1.release();
    var r2 = try common.randomArray(i16, allocator, prng.random(), 200, 0, 30, .{});
    defer r2.release();
    const v1 = try common.collect(i16, allocator, .{ .array = r1 });
    defer allocator.free(v1);
    const v2 = try common.collect(i16, allocator, .{ .array = r2 });
    defer allocator.free(v2);
    const want = try allocator.alloc(?i16, 200);
    defer allocator.free(want);
    for (want, v1, v2) |*w, p, q| w.* = if (p == null) q else if (q == null) p else @max(p.?, q.?);
    try common.expectInvokeWith(i16, &ctx, "max_element_wise", &.{ .{ .array = r1 }, .{ .array = r2 } }, none, want);
    var ch = try common.chunkedAt(allocator, r1, &.{ 17, 99 });
    defer ch.release();
    var chunked = try ctx.invokeScalar("max_element_wise", &.{ .{ .chunked = ch }, .{ .array = r2 } }, none);
    defer chunked.release();
    try std.testing.expectEqual(@as(usize, 3), chunked.chunked.numChunks());
    const chunked_rows = try common.collect(i16, allocator, chunked);
    defer allocator.free(chunked_rows);
    try common.expectRows(i16, want, chunked_rows);

    // Errors: bool (no kernel, as in Arrow), mixed non-numeric types, options.
    var flags = try common.makeArray(bool, allocator, &.{ true, false });
    defer flags.release();
    try std.testing.expectError(error.NoMatchingKernel, ctx.invokeScalar("min_element_wise", &.{ .{ .array = flags }, .{ .array = flags } }, none));
    try std.testing.expectError(error.NoMatchingKernel, ctx.invokeScalar("min_element_wise", &.{ .{ .array = d1 }, .{ .array = c } }, none));
    try std.testing.expectError(error.InvalidOptions, ctx.invokeScalar("min_element_wise", &ab, .{ .sort = .{} }));
    try std.testing.expectError(error.InvalidArity, ctx.invokeScalar("max_element_wise", &.{}, none));
}

fn ts(comptime unit: anytype, comptime tz: ?[]const u8) DataType {
    return .{ .timestamp = .{ .unit = unit, .timezone = tz } };
}

test "comparisons unify mixed string layouts, temporal units and decimal scales (pyarrow 26 results)" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);

    // Mixed string / binary layouts compare their bytes with no cast (views too; Arrow casts the
    // offset layouts and has no kernel for views).
    var s = try common.makeStrings(.string, allocator, &.{ "a", "b", null });
    defer s.release();
    var ls = try common.makeStrings(.large_string, allocator, &.{ "a", "c", "x" });
    defer ls.release();
    var sv_full = try common.makeStrings(.string_view, allocator, &.{ "zz", "a", "a", "a string past twelve bytes" });
    defer sv_full.release();
    var sv = try sv_full.slice(1, 3);
    defer sv.release();
    var bv = try common.makeStrings(.binary_view, allocator, &.{ "a", "a", "x" });
    defer bv.release();
    var lb = try common.makeStrings(.large_binary, allocator, &.{ "a", "c", "x" });
    defer lb.release();
    const fsb1 = DataType{ .fixed_size_binary = .{ .byte_width = 1 } };
    var fsb = try typedArray(allocator, fsb1, "abc", 3);
    defer fsb.release();
    try common.expectInvoke(bool, &ctx, .scalar, "equal", &.{ .{ .array = s }, .{ .array = ls } }, &.{ true, false, null });
    try common.expectInvoke(bool, &ctx, .scalar, "less", &.{ .{ .array = s }, .{ .array = ls } }, &.{ false, true, null });
    try common.expectInvoke(bool, &ctx, .scalar, "equal", &.{ .{ .array = s }, .{ .array = sv } }, &.{ true, false, null });
    try common.expectInvoke(bool, &ctx, .scalar, "greater", &.{ .{ .array = s }, .{ .array = sv } }, &.{ false, true, null });
    try common.expectInvoke(bool, &ctx, .scalar, "greater", &.{ .{ .array = ls }, .{ .array = sv } }, &.{ false, true, true });
    try common.expectInvoke(bool, &ctx, .scalar, "equal", &.{ .{ .array = s }, .{ .array = lb } }, &.{ true, false, null });
    try common.expectInvoke(bool, &ctx, .scalar, "equal", &.{ .{ .array = s }, .{ .array = fsb } }, &.{ true, true, null });
    try common.expectInvoke(bool, &ctx, .scalar, "less_equal", &.{ .{ .array = bv }, .{ .array = fsb } }, &.{ true, true, false });
    try common.expectInvoke(bool, &ctx, .scalar, "equal", &.{ .{ .scalar = Scalar.init(.{ .string = {} }, .{ .string = "a" }) }, .{ .array = ls } }, &.{ true, false, false });
    var chunked_s = try common.chunkedAt(allocator, s, &.{1});
    defer chunked_s.release();
    try common.expectInvoke(bool, &ctx, .scalar, "not_equal", &.{ .{ .chunked = chunked_s }, .{ .array = bv } }, &.{ false, true, null });

    // Timestamps of different units compare on the finer one; same unit, any two timezones
    // compare instants; naive with zoned, or different units in different zones, do not.
    var t_s = try common.makeTyped(i64, comptime ts(.second, null), allocator, &.{ 1, 2, null });
    defer t_s.release();
    var t_ms = try common.makeTyped(i64, comptime ts(.millisecond, null), allocator, &.{ 1000, 1500, 3000 });
    defer t_ms.release();
    try common.expectInvoke(bool, &ctx, .scalar, "equal", &.{ .{ .array = t_s }, .{ .array = t_ms } }, &.{ true, false, null });
    try common.expectInvoke(bool, &ctx, .scalar, "less", &.{ .{ .array = t_s }, .{ .array = t_ms } }, &.{ false, false, null });
    var utc_s = try common.makeTyped(i64, comptime ts(.second, "UTC"), allocator, &.{ 1, 2, 3 });
    defer utc_s.release();
    var utc_us = try common.makeTyped(i64, comptime ts(.microsecond, "UTC"), allocator, &.{ 1_000_000, 2, 3 });
    defer utc_us.release();
    var ny_s = try common.makeTyped(i64, comptime ts(.second, "America/New_York"), allocator, &.{ 1, 2, 3 });
    defer ny_s.release();
    var ny_ms = try common.makeTyped(i64, comptime ts(.millisecond, "America/New_York"), allocator, &.{ 1000, 2, 3 });
    defer ny_ms.release();
    try common.expectInvoke(bool, &ctx, .scalar, "equal", &.{ .{ .array = utc_s }, .{ .array = utc_us } }, &.{ true, false, false });
    try common.expectInvoke(bool, &ctx, .scalar, "equal", &.{ .{ .array = utc_s }, .{ .array = ny_s } }, &.{ true, true, true });
    try std.testing.expectError(error.NoMatchingKernel, ctx.invokeScalar("equal", &.{ .{ .array = t_s }, .{ .array = ny_s } }, none));
    try std.testing.expectError(error.NoMatchingKernel, ctx.invokeScalar("equal", &.{ .{ .array = t_s }, .{ .array = utc_us } }, none));
    try std.testing.expectError(error.NoMatchingKernel, ctx.invokeScalar("equal", &.{ .{ .array = utc_s }, .{ .array = ny_ms } }, none));
    // A value past the finer unit's range: InvalidCast when safe (Arrow: out of bounds).
    var far = try common.makeTyped(i64, comptime ts(.second, null), allocator, &.{1 << 62});
    defer far.release();
    var one_ns = try common.makeTyped(i64, comptime ts(.nanosecond, null), allocator, &.{1});
    defer one_ns.release();
    try std.testing.expectError(error.InvalidCast, ctx.invokeScalar("equal", &.{ .{ .array = far }, .{ .array = one_ns } }, none));
    var unsafe = ExecContext.initWithConfig(allocator, &registry, .{ .safe_cast = false });
    try common.expectInvoke(bool, &unsafe, .scalar, "equal", &.{ .{ .array = far }, .{ .array = one_ns } }, &.{false});

    // Dates, times and durations: Arrow's CommonTemporal.
    var d32 = try common.makeTyped(i32, .{ .date32 = {} }, allocator, &.{ 1, 2, 3 });
    defer d32.release();
    var d64 = try common.makeTyped(i64, .{ .date64 = {} }, allocator, &.{ 86_400_000, 0, 3 });
    defer d64.release();
    try common.expectInvoke(bool, &ctx, .scalar, "equal", &.{ .{ .array = d32 }, .{ .array = d64 } }, &.{ true, false, false });
    try common.expectInvoke(bool, &ctx, .scalar, "equal", &.{ .{ .array = t_s }, .{ .array = d32 } }, &.{ false, false, null });
    var day_ms = try common.makeTyped(i64, comptime ts(.millisecond, "UTC"), allocator, &.{ 86_400_000, 0, 0 });
    defer day_ms.release();
    try common.expectInvoke(bool, &ctx, .scalar, "equal", &.{ .{ .array = d32 }, .{ .array = day_ms } }, &.{ true, false, false });
    var t32 = try common.makeTyped(i32, .{ .time32 = .{ .unit = .second } }, allocator, &.{ 1, 2, 3 });
    defer t32.release();
    var t64 = try common.makeTyped(i64, .{ .time64 = .{ .unit = .microsecond } }, allocator, &.{ 1_000_000, 0, 3 });
    defer t64.release();
    try common.expectInvoke(bool, &ctx, .scalar, "equal", &.{ .{ .array = t32 }, .{ .array = t64 } }, &.{ true, false, false });
    var dur_s = try common.makeTyped(i64, .{ .duration = .{ .unit = .second } }, allocator, &.{ 1, 2, 3 });
    defer dur_s.release();
    var dur_ms = try common.makeTyped(i64, .{ .duration = .{ .unit = .millisecond } }, allocator, &.{ 1000, 0, 3 });
    defer dur_ms.release();
    try common.expectInvoke(bool, &ctx, .scalar, "greater", &.{ .{ .array = dur_s }, .{ .array = dur_ms } }, &.{ false, true, true });
    try std.testing.expectError(error.NoMatchingKernel, ctx.invokeScalar("equal", &.{ .{ .array = t32 }, .{ .array = dur_s } }, none));
    try std.testing.expectError(error.UnsupportedType, ctx.invokeScalar("equal", &.{ .{ .array = d32 }, .{ .array = dur_s } }, none));

    // Decimals of different scales, widths and precisions and with integers: exact values.
    var x = try common.makeTyped(i128, .{ .decimal128 = .{ .precision = 10, .scale = 2 } }, allocator, &.{ 125, 250, null });
    defer x.release();
    var y = try common.makeTyped(i128, .{ .decimal128 = .{ .precision = 12, .scale = 3 } }, allocator, &.{ 1250, 2499, 1000 });
    defer y.release();
    try common.expectInvoke(bool, &ctx, .scalar, "equal", &.{ .{ .array = x }, .{ .array = y } }, &.{ true, false, null });
    try common.expectInvoke(bool, &ctx, .scalar, "less", &.{ .{ .array = x }, .{ .array = y } }, &.{ false, false, null });
    try common.expectInvoke(bool, &ctx, .scalar, "greater", &.{ .{ .array = y }, .{ .array = x } }, &.{ false, false, null });
    var x32 = try common.makeTyped(i32, .{ .decimal32 = .{ .precision = 5, .scale = 2 } }, allocator, &.{ 125, 250, 100 });
    defer x32.release();
    try common.expectInvoke(bool, &ctx, .scalar, "equal", &.{ .{ .array = x32 }, .{ .array = x } }, &.{ true, true, null });
    var w = try common.makeTyped(i256, .{ .decimal256 = .{ .precision = 40, .scale = 1 } }, allocator, &.{ 12, 25, 10 });
    defer w.release();
    try common.expectInvoke(bool, &ctx, .scalar, "less", &.{ .{ .array = w }, .{ .array = x } }, &.{ true, false, null });
    var ints = try common.makeArray(i32, allocator, &.{ 1, 2, 3 });
    defer ints.release();
    try common.expectInvoke(bool, &ctx, .scalar, "equal", &.{ .{ .array = x }, .{ .array = ints } }, &.{ false, false, null });
    var longs = try common.makeArray(i64, allocator, &.{ 1, 3, 3 });
    defer longs.release();
    try common.expectInvoke(bool, &ctx, .scalar, "less", &.{ .{ .array = x }, .{ .array = longs } }, &.{ false, true, null });
    try common.expectInvoke(bool, &ctx, .scalar, "greater_equal", &.{ .{ .scalar = common.scalarOf(u64, 2) }, .{ .array = x } }, &.{ true, false, null });
    var floats = try common.makeArray(f64, allocator, &.{ 1.25, 2.5, 3.0 });
    defer floats.release();
    try common.expectInvoke(bool, &ctx, .scalar, "equal", &.{ .{ .array = x }, .{ .array = floats } }, &.{ true, true, null });
    // A common precision past 38 (Arrow rejects it) and a negative scale (Arrow: not
    // implemented) still compare exactly; a product past 256 bits orders by sign.
    var p38 = try common.makeTyped(i128, .{ .decimal128 = .{ .precision = 38, .scale = 0 } }, allocator, &.{ 99999999999999999999999999999999999999, -1, 0 });
    defer p38.release();
    var p38s = try common.makeTyped(i128, .{ .decimal128 = .{ .precision = 38, .scale = 37 } }, allocator, &.{ 1, -10_000_000_000_000_000_000_000_000_000_000_000_000, 0 });
    defer p38s.release();
    try common.expectInvoke(bool, &ctx, .scalar, "greater", &.{ .{ .array = p38 }, .{ .array = p38s } }, &.{ true, false, false });
    try common.expectInvoke(bool, &ctx, .scalar, "equal", &.{ .{ .array = p38 }, .{ .array = p38s } }, &.{ false, true, true });
    var hundreds = try common.makeTyped(i128, .{ .decimal128 = .{ .precision = 5, .scale = -2 } }, allocator, &.{ 1, 2, -1 });
    defer hundreds.release();
    var plain = try common.makeTyped(i128, .{ .decimal128 = .{ .precision = 5, .scale = 0 } }, allocator, &.{ 100, 201, -100 });
    defer plain.release();
    try common.expectInvoke(bool, &ctx, .scalar, "less", &.{ .{ .array = hundreds }, .{ .array = plain } }, &.{ false, true, false });
    var huge = try common.makeTyped(i256, .{ .decimal256 = .{ .precision = 76, .scale = 0 } }, allocator, &.{ -(1 << 250), 1 << 250 });
    defer huge.release();
    var tiny = try common.makeTyped(i256, .{ .decimal256 = .{ .precision = 76, .scale = 76 } }, allocator, &.{ 1 << 250, -(1 << 250) });
    defer tiny.release();
    try common.expectInvoke(bool, &ctx, .scalar, "less", &.{ .{ .array = huge }, .{ .array = tiny } }, &.{ true, false });
    var bad = try common.makeTyped(i128, .{ .decimal128 = .{ .precision = 40, .scale = 0 } }, allocator, &.{ 1, 2, 3 });
    defer bad.release();
    try std.testing.expectError(error.InvalidInput, ctx.invokeScalar("equal", &.{ .{ .array = bad }, .{ .array = x } }, none));
    try std.testing.expectError(error.NoMatchingKernel, ctx.invokeScalar("equal", &.{ .{ .array = x }, .{ .array = s } }, none));
}

test "min_element_wise and max_element_wise on decimals, fixed_size_binary and mixed temporal units" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);

    const dec = DataType{ .decimal128 = .{ .precision = 5, .scale = 1 } };
    var a = try common.makeTyped(i128, dec, allocator, &.{ 15, null, 30 });
    defer a.release();
    var b = try common.makeTyped(i128, dec, allocator, &.{ -15, null, 40 });
    defer b.release();
    var out = try ctx.invokeScalar("min_element_wise", &.{ .{ .array = a }, .{ .array = b } }, .{ .element_wise_aggregate = .{ .skip_nulls = false } });
    defer out.release();
    try std.testing.expect(out.dataType().eql(dec));
    try out.array.data().validateFull();
    const mins = (try (common.array_mod.PrimitiveArray(i128){ .data = out.array.data() }).values());
    try std.testing.expectEqual(@as(i128, -15), mins[0]);
    try std.testing.expect(out.array.data().isNull(1));
    try std.testing.expectEqual(@as(i128, 30), mins[2]);
    var w = try common.makeTyped(i256, .{ .decimal256 = .{ .precision = 60, .scale = 0 } }, allocator, &.{ 1 << 200, -5 });
    defer w.release();
    var w2 = try common.makeTyped(i256, .{ .decimal256 = .{ .precision = 60, .scale = 0 } }, allocator, &.{ 3, null });
    defer w2.release();
    var wmax = try ctx.invokeScalar("max_element_wise", &.{ .{ .array = w }, .{ .array = w2 } }, none);
    defer wmax.release();
    const wv = try (common.array_mod.PrimitiveArray(i256){ .data = wmax.array.data() }).values();
    try std.testing.expectEqual(@as(i256, 1 << 200), wv[0]);
    try std.testing.expectEqual(@as(i256, -5), wv[1]);
    // Decimals of different scales are not unified (Arrow: "Different input types").
    var c = try common.makeTyped(i128, .{ .decimal128 = .{ .precision = 5, .scale = 2 } }, allocator, &.{ 1, 2, 3 });
    defer c.release();
    try std.testing.expectError(error.NoMatchingKernel, ctx.invokeScalar("max_element_wise", &.{ .{ .array = a }, .{ .array = c } }, none));

    // fixed_size_binary: byte order, nulls skipped or propagated; widths must match.
    const fsb1 = DataType{ .fixed_size_binary = .{ .byte_width = 1 } };
    var f1_full = try typedArray(allocator, fsb1, "zab", 3);
    defer f1_full.release();
    var f1 = try f1_full.slice(1, 2);
    defer f1.release();
    var f2 = try typedArray(allocator, fsb1, "ba", 2);
    defer f2.release();
    var fmax = try ctx.invokeScalar("max_element_wise", &.{ .{ .array = f1 }, .{ .array = f2 } }, none);
    defer fmax.release();
    try fmax.array.data().validateFull();
    try std.testing.expectEqualStrings("bb", fmax.array.data().buffers[1].data[0..2]);
    var fmin = try ctx.invokeScalar("min_element_wise", &.{ .{ .array = f1 }, .{ .scalar = Scalar.init(fsb1, .null) } }, .{ .element_wise_aggregate = .{ .skip_nulls = true } });
    defer fmin.release();
    try std.testing.expectEqualStrings("ab", fmin.array.data().buffers[1].data[0..2]);
    var fnull = try ctx.invokeScalar("min_element_wise", &.{ .{ .array = f1 }, .{ .scalar = Scalar.init(fsb1, .null) } }, .{ .element_wise_aggregate = .{ .skip_nulls = false } });
    defer fnull.release();
    try std.testing.expectEqual(@as(?usize, 2), fnull.array.data().null_count);
    var f3 = try typedArray(allocator, .{ .fixed_size_binary = .{ .byte_width = 2 } }, "abcd", 2);
    defer f3.release();
    try std.testing.expectError(error.NoMatchingKernel, ctx.invokeScalar("max_element_wise", &.{ .{ .array = f1 }, .{ .array = f3 } }, none));

    // Temporal operands of different units or kinds go to their common type (Arrow).
    var t_s = try common.makeTyped(i64, .{ .timestamp = .{ .unit = .second, .timezone = null } }, allocator, &.{ 1, 2, null });
    defer t_s.release();
    var t_ms = try common.makeTyped(i64, .{ .timestamp = .{ .unit = .millisecond, .timezone = null } }, allocator, &.{ 1000, 1500, 3000 });
    defer t_ms.release();
    var tmax = try ctx.invokeScalar("max_element_wise", &.{ .{ .array = t_s }, .{ .array = t_ms } }, none);
    defer tmax.release();
    try std.testing.expect(tmax.dataType().eql(.{ .timestamp = .{ .unit = .millisecond, .timezone = null } }));
    try common.expectInvoke(i64, &ctx, .scalar, "max_element_wise", &.{ .{ .array = t_s }, .{ .array = t_ms } }, &.{ 1000, 2000, 3000 });
    var d32 = try common.makeTyped(i32, .{ .date32 = {} }, allocator, &.{ 1, 2, 3 });
    defer d32.release();
    var d64 = try common.makeTyped(i64, .{ .date64 = {} }, allocator, &.{ 86_400_000, 0, 3 });
    defer d64.release();
    try common.expectInvoke(i64, &ctx, .scalar, "max_element_wise", &.{ .{ .array = d32 }, .{ .array = d64 } }, &.{ 86_400_000, 172_800_000, 259_200_000 });
    var t32 = try common.makeTyped(i32, .{ .time32 = .{ .unit = .second } }, allocator, &.{1});
    defer t32.release();
    var t64 = try common.makeTyped(i64, .{ .time64 = .{ .unit = .microsecond } }, allocator, &.{1000});
    defer t64.release();
    try common.expectInvoke(i64, &ctx, .scalar, "max_element_wise", &.{ .{ .array = t32 }, .{ .array = t64 } }, &.{1_000_000});
    var day = try d32.slice(0, 1);
    defer day.release();
    var ts1000 = try common.makeTyped(i64, .{ .timestamp = .{ .unit = .second, .timezone = null } }, allocator, &.{1000});
    defer ts1000.release();
    try common.expectInvoke(i64, &ctx, .scalar, "max_element_wise", &.{ .{ .array = day }, .{ .array = ts1000 } }, &.{86_400});
    var zoned = try common.makeTyped(i64, .{ .timestamp = .{ .unit = .millisecond, .timezone = "UTC" } }, allocator, &.{ 1, 2, 3 });
    defer zoned.release();
    try std.testing.expectError(error.NoMatchingKernel, ctx.invokeScalar("max_element_wise", &.{ .{ .array = t_s }, .{ .array = zoned } }, none));
    // String layouts are not mixed here (as Arrow).
    var s = try common.makeStrings(.string, allocator, &.{"a"});
    defer s.release();
    var ls = try common.makeStrings(.large_string, allocator, &.{"b"});
    defer ls.release();
    try std.testing.expectError(error.NoMatchingKernel, ctx.invokeScalar("max_element_wise", &.{ .{ .array = s }, .{ .array = ls } }, none));
}

test "implicit casts decode dictionaries and give null-type operands the other type (pyarrow 26)" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);

    const string_type = DataType{ .string = {} };
    const int64_type = DataType{ .int64 = {} };
    const dict_string = DataType{ .dictionary = .{ .index_type = .{ .bit_width = 32, .signed = true }, .value_type = &string_type } };
    const dict_int64 = DataType{ .dictionary = .{ .index_type = .{ .bit_width = 8, .signed = true }, .value_type = &int64_type } };
    var ab = try common.makeStrings(.string, allocator, &.{ "a", "b" });
    defer ab.release();
    var ac = try common.makeStrings(.string, allocator, &.{ "a", "c" });
    defer ac.release();
    var d_ab = try ctx.invokeScalar("cast", &.{.{ .array = ab }}, .{ .cast = .{ .to_type = dict_string } });
    defer d_ab.release();
    var d_ac = try ctx.invokeScalar("cast", &.{.{ .array = ac }}, .{ .cast = .{ .to_type = dict_string } });
    defer d_ac.release();
    try common.expectInvoke(bool, &ctx, .scalar, "equal", &.{ d_ab, .{ .array = ac } }, &.{ true, false });
    try common.expectInvoke(bool, &ctx, .scalar, "equal", &.{ d_ab, d_ac }, &.{ true, false });
    var sv = try common.makeStrings(.string_view, allocator, &.{ "a", "c" });
    defer sv.release();
    try common.expectInvoke(bool, &ctx, .scalar, "less", &.{ d_ab, .{ .array = sv } }, &.{ false, true });

    var ints = try common.makeArray(i64, allocator, &.{ 1, 2 });
    defer ints.release();
    var d_ints = try ctx.invokeScalar("cast", &.{.{ .array = ints }}, .{ .cast = .{ .to_type = dict_int64 } });
    defer d_ints.release();
    var small = try common.makeArray(i8, allocator, &.{ 1, 3 });
    defer small.release();
    var other = try common.makeArray(i64, allocator, &.{ 1, 3 });
    defer other.release();
    try common.expectInvoke(bool, &ctx, .scalar, "equal", &.{ d_ints, .{ .array = small } }, &.{ true, false });
    try common.expectInvoke(i64, &ctx, .scalar, "max_element_wise", &.{ d_ints, .{ .array = other } }, &.{ 1, 3 });
    try common.expectInvoke(i64, &ctx, .scalar, "add", &.{ d_ints, .{ .array = other } }, &.{ 2, 5 });

    // A null-type operand takes the other operand's type (all rows null), except in
    // min / max_element_wise; two null-type operands have no kernel (as Arrow).
    var nulls = try dense.nullArray(allocator, .{ .null = {} }, 2);
    defer nulls.release();
    try common.expectInvoke(bool, &ctx, .scalar, "less", &.{ .{ .array = nulls }, .{ .array = ab } }, &.{ null, null });
    try common.expectInvoke(i64, &ctx, .scalar, "add", &.{ .{ .array = nulls }, .{ .array = other } }, &.{ null, null });
    try std.testing.expect((try registry.resolveResultType("add", .scalar, &.{ .{ .array = nulls }, .{ .array = other } }, none)) == .int64);
    var dec = try common.makeTyped(i128, .{ .decimal128 = .{ .precision = 5, .scale = 0 } }, allocator, &.{ 1, 2 });
    defer dec.release();
    try common.expectInvoke(bool, &ctx, .scalar, "less", &.{ .{ .array = nulls }, .{ .array = dec } }, &.{ null, null });
    try std.testing.expectError(error.NoMatchingKernel, ctx.invokeScalar("max_element_wise", &.{ .{ .array = nulls }, .{ .array = other } }, none));
    try std.testing.expectError(error.NoMatchingKernel, ctx.invokeScalar("equal", &.{ .{ .array = nulls }, .{ .array = nulls } }, none));
}
