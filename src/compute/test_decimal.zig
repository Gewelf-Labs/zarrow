//! Decimal arithmetic, casts and aggregates. Expected values are pyarrow 26 results.
const std = @import("std");
const common = @import("test_common.zig");

const ArrayRef = common.ArrayRef;
const DataType = common.DataType;
const Datum = common.Datum;
const ExecContext = common.ExecContext;
const Options = common.Options;
const Scalar = common.Scalar;
const PrimitiveArray = common.array_mod.PrimitiveArray;
const none = common.none;

fn d32(p: u8, s: i32) DataType {
    return .{ .decimal32 = .{ .precision = p, .scale = s } };
}
fn d64(p: u8, s: i32) DataType {
    return .{ .decimal64 = .{ .precision = p, .scale = s } };
}
fn d128(p: u8, s: i32) DataType {
    return .{ .decimal128 = .{ .precision = p, .scale = s } };
}
fn d256(p: u8, s: i32) DataType {
    return .{ .decimal256 = .{ .precision = p, .scale = s } };
}

fn castTo(dt: DataType, safe: bool) Options {
    return .{ .cast = .{ .to_type = dt, .safe = safe } };
}

/// Rows of a decimal / integer array, chunked or scalar datum stored as `T`; caller frees.
pub fn rows(comptime T: type, datum: Datum) ![]?T {
    const allocator = std.testing.allocator;
    switch (datum) {
        .scalar => |s| {
            const out = try allocator.alloc(?T, 1);
            errdefer allocator.free(out);
            out[0] = if (s.isNull()) null else switch (s.value) {
                inline .i8, .i16, .i32, .i64, .u8, .u16, .u32, .u64, .decimal32, .decimal64, .decimal128, .decimal256 => |x| if (@TypeOf(x) == T) x else return error.TestUnexpectedResult,
                else => return error.TestUnexpectedResult,
            };
            return out;
        },
        .array => |arr| return partRows(T, &.{arr}),
        .chunked => |c| return partRows(T, c.chunks()),
        .record_batch, .table => return error.InvalidInput,
    }
}

fn partRows(comptime T: type, parts: []const ArrayRef) ![]?T {
    var total: usize = 0;
    for (parts) |p| total += p.data().length;
    const out = try std.testing.allocator.alloc(?T, total);
    errdefer std.testing.allocator.free(out);
    var pos: usize = 0;
    for (parts) |p| {
        try p.data().validateFull();
        const vals = try (PrimitiveArray(T){ .data = p.data() }).values();
        for (vals, 0..) |v, i| out[pos + i] = if (p.data().isNull(i)) null else v;
        pos += vals.len;
    }
    return out;
}

/// Invoke scalar function `name` and check the result type and rows (stored as `T`).
pub fn expectDec(comptime T: type, ctx: *ExecContext, name: []const u8, args: []const Datum, options: Options, dt: DataType, expected: []const ?T) !void {
    var out = try ctx.invokeScalar(name, args, options);
    defer out.release();
    if (!out.dataType().eql(dt)) {
        std.debug.print("{s}: type {any}, expected {any}\n", .{ name, out.dataType(), dt });
        return error.TestExpectedEqual;
    }
    try std.testing.expect((try ctx.registry.resolveResultType(name, .scalar, args, options)).eql(dt));
    const got = try rows(T, out);
    defer std.testing.allocator.free(got);
    try common.expectRows(T, expected, got);
}

fn arr128(comptime dt: DataType, values: []const ?i128) !ArrayRef {
    return common.makeTyped(i128, dt, std.testing.allocator, values);
}

test "decimal arithmetic follows Arrow's precision and scale rules (nulls, sliced, chunked, scalars)" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);

    // decimal128(10, 2) [1.25, -3.50, null] with decimal128(6, 1) [2.5, 1.0, 1.0].
    var a = try arr128(comptime d128(10, 2), &.{ 125, -350, null });
    defer a.release();
    var b = try arr128(comptime d128(6, 1), &.{ 25, 10, 10 });
    defer b.release();
    const ab = [_]Datum{ .{ .array = a }, .{ .array = b } };
    inline for (.{ "", "_checked" }) |suffix| {
        try expectDec(i128, &ctx, "add" ++ suffix, &ab, none, d128(11, 2), &.{ 375, -250, null });
        try expectDec(i128, &ctx, "subtract" ++ suffix, &ab, none, d128(11, 2), &.{ -125, -450, null });
        try expectDec(i128, &ctx, "multiply" ++ suffix, &ab, none, d128(17, 3), &.{ 3125, -3500, null });
        try expectDec(i128, &ctx, "divide" ++ suffix, &ab, none, d128(17, 8), &.{ 50_000_000, -350_000_000, null });
    }

    // decimal32 / decimal64 operands give decimal128 (Arrow casts them up).
    var s32 = try common.makeTyped(i32, comptime d32(5, 2), allocator, &.{ 125, -350 });
    defer s32.release();
    var t32 = try common.makeTyped(i32, comptime d32(3, 1), allocator, &.{ 25, 10 });
    defer t32.release();
    var t64 = try common.makeTyped(i64, comptime d64(3, 1), allocator, &.{ 25, 10 });
    defer t64.release();
    try expectDec(i128, &ctx, "add", &.{ .{ .array = s32 }, .{ .array = t32 } }, none, d128(6, 2), &.{ 375, -250 });
    try expectDec(i128, &ctx, "multiply", &.{ .{ .array = s32 }, .{ .array = t64 } }, none, d128(9, 3), &.{ 3125, -3500 });
    try expectDec(i128, &ctx, "divide", &.{ .{ .array = s32 }, .{ .array = t32 } }, none, d128(9, 5), &.{ 50_000, -350_000 });

    // decimal256 operands give decimal256.
    var w = try common.makeTyped(i256, comptime d256(10, 2), allocator, &.{ 125, -350, null });
    defer w.release();
    var v = try common.makeTyped(i256, comptime d256(6, 1), allocator, &.{ 25, 10, 10 });
    defer v.release();
    try expectDec(i256, &ctx, "add", &.{ .{ .array = w }, .{ .array = v } }, none, d256(11, 2), &.{ 375, -250, null });
    try expectDec(i256, &ctx, "divide_checked", &.{ .{ .array = w }, .{ .array = v } }, none, d256(17, 8), &.{ 50_000_000, -350_000_000, null });
    var c5 = try arr128(comptime d128(5, 1), &.{15});
    defer c5.release();
    var c52 = try common.makeTyped(i256, comptime d256(5, 2), allocator, &.{125});
    defer c52.release();
    try expectDec(i256, &ctx, "add", &.{ .{ .array = c5 }, .{ .array = c52 } }, none, d256(7, 2), &.{275});

    // Truncating division and Arrow's dividend scale-up.
    var n = try arr128(comptime d128(10, 2), &.{ 100, -200, 777 });
    defer n.release();
    var dv = try arr128(comptime d128(5, 2), &.{ 300, 300, -7 });
    defer dv.release();
    try expectDec(i128, &ctx, "divide", &.{ .{ .array = n }, .{ .array = dv } }, none, d128(16, 6), &.{ 333_333, -666_666, -111_000_000 });
    var n0 = try arr128(comptime d128(10, 0), &.{ 1, -2 });
    defer n0.release();
    var dv4 = try arr128(comptime d128(5, 4), &.{ 30_000, 30_000 });
    defer dv4.release();
    try expectDec(i128, &ctx, "divide", &.{ .{ .array = n0 }, .{ .array = dv4 } }, none, d128(18, 4), &.{ 3333, -6666 });
    var big_n = try arr128(comptime d128(30, 10), &.{10_000_000_000});
    defer big_n.release();
    var three = try arr128(comptime d128(5, 2), &.{300});
    defer three.release();
    try expectDec(i128, &ctx, "divide", &.{ .{ .array = big_n }, .{ .array = three } }, none, d128(36, 14), &.{33_333_333_333_333});

    // Integers act as decimal(digits, 0): int32 10, int64 19, int8 3, uint64 20.
    var i32a = try common.makeArray(i32, allocator, &.{2});
    defer i32a.release();
    var i64a = try common.makeArray(i64, allocator, &.{2});
    defer i64a.release();
    var u64a = try common.makeArray(u64, allocator, &.{2});
    defer u64a.release();
    try expectDec(i128, &ctx, "add", &.{ .{ .array = c5 }, .{ .array = i32a } }, none, d128(12, 1), &.{35});
    try expectDec(i128, &ctx, "add", &.{ .{ .array = c5 }, .{ .array = i64a } }, none, d128(21, 1), &.{35});
    try expectDec(i128, &ctx, "add", &.{ .{ .array = c5 }, .{ .array = u64a } }, none, d128(22, 1), &.{35});
    try expectDec(i128, &ctx, "multiply", &.{ .{ .array = c5 }, .{ .array = i32a } }, none, d128(16, 1), &.{30});
    try expectDec(i128, &ctx, "divide", &.{ .{ .array = c5 }, .{ .array = i32a } }, none, d128(16, 12), &.{750_000_000_000});
    var s15 = try common.makeTyped(i32, comptime d32(5, 1), allocator, &.{15});
    defer s15.release();
    try expectDec(i128, &ctx, "add", &.{ .{ .array = s15 }, .{ .scalar = common.scalarOf(i8, 2) } }, none, d128(6, 1), &.{35});

    // A decimal with a float goes to float64.
    const fx = [_]Datum{ .{ .array = c5 }, .{ .scalar = common.scalarOf(f32, 2.25) } };
    try common.expectInvoke(f64, &ctx, .scalar, "add", &fx, &.{3.75});
    try std.testing.expect((try registry.resolveResultType("add", .scalar, &fx, none)) == .double);

    // Scalars: two scalars give a scalar; a null scalar gives nulls.
    const x = Scalar.init(d128(5, 1), .{ .decimal128 = 15 });
    const y = Scalar.init(d128(5, 2), .{ .decimal128 = 125 });
    try expectDec(i128, &ctx, "add", &.{ .{ .scalar = x }, .{ .scalar = y } }, none, d128(7, 2), &.{275});
    try expectDec(i128, &ctx, "add", &.{ .{ .array = a }, .{ .scalar = Scalar.init(d128(5, 2), .null) } }, none, d128(11, 2), &.{ null, null, null });

    // Sliced and chunked (misaligned chunks) inputs equal the array result.
    var a_full = try arr128(comptime d128(10, 2), &.{ 9, 125, -350, null, 7 });
    defer a_full.release();
    var a_slice = try a_full.slice(1, 3);
    defer a_slice.release();
    try expectDec(i128, &ctx, "add", &.{ .{ .array = a_slice }, .{ .array = b } }, none, d128(11, 2), &.{ 375, -250, null });
    var ca = try common.chunkedAt(allocator, a, &.{1});
    defer ca.release();
    var cb = try common.chunkedAt(allocator, b, &.{2});
    defer cb.release();
    var chunked = try ctx.invokeScalar("multiply_checked", &.{ .{ .chunked = ca }, .{ .chunked = cb } }, none);
    defer chunked.release();
    try std.testing.expectEqual(@as(usize, 3), chunked.chunked.numChunks());
    const got = try rows(i128, chunked);
    defer allocator.free(got);
    try common.expectRows(i128, &.{ 3125, -3500, null }, got);
}

test "decimal arithmetic errors: precision overflow, negative scale, zero divisor, checked overflow, bad precision" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);

    // decimal128(38, 0) + itself needs precision 39 (Arrow: "Decimal precision out of range").
    var p38 = try arr128(comptime d128(38, 0), &.{1});
    defer p38.release();
    try std.testing.expectError(error.Overflow, ctx.invokeScalar("add", &.{ .{ .array = p38 }, .{ .array = p38 } }, none));
    try std.testing.expectError(error.Overflow, registry.resolveResultType("add", .scalar, &.{ .{ .array = p38 }, .{ .array = p38 } }, none));
    var p20 = try arr128(comptime d128(20, 0), &.{1});
    defer p20.release();
    try std.testing.expectError(error.Overflow, ctx.invokeScalar("multiply", &.{ .{ .array = p20 }, .{ .array = p20 } }, none));
    var p19 = try arr128(comptime d128(19, 0), &.{1});
    defer p19.release();
    var p18 = try arr128(comptime d128(18, 0), &.{1});
    defer p18.release();
    try expectDec(i128, &ctx, "multiply", &.{ .{ .array = p19 }, .{ .array = p18 } }, none, d128(38, 0), &.{1});
    // Negative scales are not implemented (as Arrow).
    var neg = try arr128(comptime d128(5, -2), &.{1});
    defer neg.release();
    try std.testing.expectError(error.NotImplemented, ctx.invokeScalar("add", &.{ .{ .array = neg }, .{ .array = p19 } }, none));

    // A zero divisor in a valid row fails both variants; in a null row it is skipped.
    var one = try arr128(comptime d128(10, 2), &.{ 100, null });
    defer one.release();
    var zeros = try arr128(comptime d128(5, 2), &.{ 0, 0 });
    defer zeros.release();
    try std.testing.expectError(error.DivideByZero, ctx.invokeScalar("divide", &.{ .{ .array = one }, .{ .array = zeros } }, none));
    try std.testing.expectError(error.DivideByZero, ctx.invokeScalar("divide_checked", &.{ .{ .array = one }, .{ .array = zeros } }, none));
    var only_null = try one.slice(1, 1);
    defer only_null.release();
    var zero1 = try zeros.slice(1, 1);
    defer zero1.release();
    try expectDec(i128, &ctx, "divide", &.{ .{ .array = only_null }, .{ .array = zero1 } }, none, d128(16, 6), &.{null});

    // Stored values past the declared precision: unchecked wraps (Arrow C++), checked reports.
    var huge = try arr128(comptime d128(37, 0), &.{ std.math.maxInt(i128), null });
    defer huge.release();
    var ones = try arr128(comptime d128(1, 0), &.{ 1, 1 });
    defer ones.release();
    try expectDec(i128, &ctx, "add", &.{ .{ .array = huge }, .{ .array = ones } }, none, d128(38, 0), &.{ std.math.minInt(i128), null });
    try std.testing.expectError(error.Overflow, ctx.invokeScalar("add_checked", &.{ .{ .array = huge }, .{ .array = ones } }, none));
    try std.testing.expectError(error.Overflow, ctx.invokeScalar("multiply_checked", &.{ .{ .array = huge }, .{ .array = ones } }, none));
    try std.testing.expectError(error.Overflow, ctx.invokeScalar("subtract_checked", &.{ .{ .array = ones }, .{ .array = huge } }, none));
    try std.testing.expectError(error.Overflow, ctx.invokeScalar("divide_checked", &.{ .{ .array = huge }, .{ .array = ones } }, none));
    // A huge value in a null row does not fail the checked variant.
    var huge_null = try huge.slice(1, 1);
    defer huge_null.release();
    var one1 = try ones.slice(1, 1);
    defer one1.release();
    try expectDec(i128, &ctx, "add_checked", &.{ .{ .array = huge_null }, .{ .array = one1 } }, none, d128(38, 0), &.{null});

    // Precision outside [1, 38] is malformed input.
    var bad = try arr128(comptime d128(39, 0), &.{1});
    defer bad.release();
    try std.testing.expectError(error.InvalidInput, ctx.invokeScalar("add", &.{ .{ .array = bad }, .{ .array = ones } }, none));
    try std.testing.expectError(error.InvalidInput, ctx.invokeScalar("negate", &.{.{ .array = bad }}, none));
    // A scalar whose payload is not the field of its type is malformed.
    try std.testing.expectError(error.InvalidInput, ctx.invokeScalar("add", &.{ .{ .array = ones }, .{ .scalar = Scalar.init(d128(5, 0), .{ .i64 = 1 }) } }, none));
    // Strings and decimals do not mix; decimal32 has no negate (as Arrow).
    var s = try common.makeStrings(.string, allocator, &.{"1"});
    defer s.release();
    try std.testing.expectError(error.NoMatchingKernel, ctx.invokeScalar("add", &.{ .{ .array = ones }, .{ .array = s } }, none));
    var small = try common.makeTyped(i32, comptime d32(5, 2), allocator, &.{1});
    defer small.release();
    try std.testing.expectError(error.NoMatchingKernel, ctx.invokeScalar("negate", &.{.{ .array = small }}, none));
}

test "decimal negate, abs, sign and their checked variants" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);

    var a = try arr128(comptime d128(2, 0), &.{ -99, 0, 5, null });
    defer a.release();
    try expectDec(i128, &ctx, "negate", &.{.{ .array = a }}, none, d128(2, 0), &.{ 99, 0, -5, null });
    try expectDec(i128, &ctx, "abs_checked", &.{.{ .array = a }}, none, d128(2, 0), &.{ 99, 0, 5, null });
    try common.expectInvoke(i64, &ctx, .scalar, "sign", &.{.{ .array = a }}, &.{ -1, 0, 1, null });
    var w = try common.makeTyped(i256, comptime d256(70, 3), allocator, &.{ -(1 << 200), null });
    defer w.release();
    try expectDec(i256, &ctx, "abs", &.{.{ .array = w }}, none, d256(70, 3), &.{ 1 << 200, null });
    try common.expectInvoke(i64, &ctx, .scalar, "sign", &.{.{ .array = w }}, &.{ -1, null });
    // The one storage value without a negation: wraps unchecked, Overflow checked (valid rows).
    var m = try arr128(comptime d128(38, 0), &.{ std.math.minInt(i128), null });
    defer m.release();
    try expectDec(i128, &ctx, "negate", &.{.{ .array = m }}, none, d128(38, 0), &.{ std.math.minInt(i128), null });
    try std.testing.expectError(error.Overflow, ctx.invokeScalar("negate_checked", &.{.{ .array = m }}, none));
    try std.testing.expectError(error.Overflow, ctx.invokeScalar("abs_checked", &.{.{ .array = m }}, none));
    try expectDec(i128, &ctx, "negate", &.{.{ .scalar = Scalar.init(d128(5, 1), .{ .decimal128 = 15 }) }}, none, d128(5, 1), &.{-15});

    // Numeric sign: int8 for integers, the float type for floats (NaN stays NaN, -0.0 gives 0).
    var ints = try common.makeArray(i32, allocator, &.{ 7, -7, 0, null });
    defer ints.release();
    try common.expectInvoke(i8, &ctx, .scalar, "sign", &.{.{ .array = ints }}, &.{ 1, -1, 0, null });
    var u = try common.makeArray(u8, allocator, &.{ 1, 0 });
    defer u.release();
    try common.expectInvoke(i8, &ctx, .scalar, "sign", &.{.{ .array = u }}, &.{ 1, 0 });
    var f = try common.makeArray(f64, allocator, &.{ 1.5, -0.0, 0.0, std.math.nan(f64), -2.0, null });
    defer f.release();
    try common.expectInvoke(f64, &ctx, .scalar, "sign", &.{.{ .array = f }}, &.{ 1.0, 0.0, 0.0, std.math.nan(f64), -1.0, null });
}

test "decimal casts: rescale, integers, floats and text (pyarrow 26 results)" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);

    // decimal128(10, 2) [1.25, -3.75, 0.05, -0.05, 1.35, null].
    var x = try arr128(comptime d128(10, 2), &.{ 125, -375, 5, -5, 135, null });
    defer x.release();
    const xa = [_]Datum{.{ .array = x }};
    try std.testing.expectError(error.InvalidCast, ctx.invokeScalar("cast", &xa, castTo(d128(10, 1), true)));
    try expectDec(i128, &ctx, "cast", &xa, castTo(d128(10, 1), false), d128(10, 1), &.{ 12, -37, 0, 0, 13, null });
    try expectDec(i128, &ctx, "cast", &xa, castTo(d128(12, 4), true), d128(12, 4), &.{ 12500, -37500, 500, -500, 13500, null });
    try expectDec(i32, &ctx, "cast", &xa, castTo(d32(6, 2), true), d32(6, 2), &.{ 125, -375, 5, -5, 135, null });
    try expectDec(i256, &ctx, "cast", &xa, castTo(d256(40, 5), true), d256(40, 5), &.{ 125000, -375000, 5000, -5000, 135000, null });
    // Precision: safe rejects, unsafe keeps the value (and wraps past the storage).
    var wide = try arr128(comptime d128(10, 2), &.{12345});
    defer wide.release();
    try std.testing.expectError(error.InvalidCast, ctx.invokeScalar("cast", &.{.{ .array = wide }}, castTo(d128(4, 2), true)));
    try expectDec(i128, &ctx, "cast", &.{.{ .array = wide }}, castTo(d128(4, 2), false), d128(4, 2), &.{12345});
    var to32 = try arr128(comptime d128(10, 2), &.{1234567845});
    defer to32.release();
    try std.testing.expectError(error.InvalidCast, ctx.invokeScalar("cast", &.{.{ .array = to32 }}, castTo(d32(9, 2), true)));
    try expectDec(i32, &ctx, "cast", &.{.{ .array = to32 }}, castTo(d32(9, 2), false), d32(9, 2), &.{1234567845});
    var near_max = try arr128(comptime d128(34, 2), &.{9999999999999999999999999999999999});
    defer near_max.release();
    try expectDec(i128, &ctx, "cast", &.{.{ .array = near_max }}, castTo(d128(38, 10), false), d128(38, 10), &.{-89876380638144118857971241966873469184});

    // Decimal to integer: truncates (unsafe) or rejects a fraction / out of range (safe).
    var di = try arr128(comptime d128(10, 2), &.{ 12300, -700, null });
    defer di.release();
    try expectDec(i32, &ctx, "cast", &.{.{ .array = di }}, castTo(.{ .int32 = {} }, true), .{ .int32 = {} }, &.{ 123, -7, null });
    var frac = try arr128(comptime d128(10, 2), &.{ 12345, -12399 });
    defer frac.release();
    try std.testing.expectError(error.InvalidCast, ctx.invokeScalar("cast", &.{.{ .array = frac }}, castTo(.{ .int32 = {} }, true)));
    try expectDec(i32, &ctx, "cast", &.{.{ .array = frac }}, castTo(.{ .int32 = {} }, false), .{ .int32 = {} }, &.{ 123, -123 });
    var big = try arr128(comptime d128(10, 2), &.{ 30000, -30050 });
    defer big.release();
    try std.testing.expectError(error.InvalidCast, ctx.invokeScalar("cast", &.{.{ .array = big }}, castTo(.{ .int8 = {} }, true)));
    try expectDec(i8, &ctx, "cast", &.{.{ .array = big }}, castTo(.{ .int8 = {} }, false), .{ .int8 = {} }, &.{ 44, -44 });
    var minus = try arr128(comptime d128(10, 2), &.{-100});
    defer minus.release();
    try std.testing.expectError(error.InvalidCast, ctx.invokeScalar("cast", &.{.{ .array = minus }}, castTo(.{ .uint8 = {} }, true)));
    var umax = try arr128(comptime d128(25, 2), &.{1844674407370955161500});
    defer umax.release();
    try expectDec(u64, &ctx, "cast", &.{.{ .array = umax }}, castTo(.{ .uint64 = {} }, true), .{ .uint64 = {} }, &.{std.math.maxInt(u64)});

    // Integer to decimal: the precision must hold digits + scale, in both modes.
    var i8s = try common.makeArray(i8, allocator, &.{ 1, -2, null, 127 });
    defer i8s.release();
    try expectDec(i128, &ctx, "cast", &.{.{ .array = i8s }}, castTo(d128(5, 2), true), d128(5, 2), &.{ 100, -200, null, 12700 });
    var i32s = try common.makeArray(i32, allocator, &.{1000});
    defer i32s.release();
    try std.testing.expectError(error.InvalidCast, ctx.invokeScalar("cast", &.{.{ .array = i32s }}, castTo(d128(5, 2), false)));
    try std.testing.expectError(error.InvalidCast, ctx.invokeScalar("cast", &.{.{ .array = i32s }}, castTo(d32(9, 2), true)));
    try std.testing.expectError(error.InvalidCast, ctx.invokeScalar("cast", &.{.{ .array = i32s }}, castTo(d128(15, -2), true)));
    var u64s = try common.makeArray(u64, allocator, &.{std.math.maxInt(u64)});
    defer u64s.release();
    try expectDec(i128, &ctx, "cast", &.{.{ .array = u64s }}, castTo(d128(22, 2), true), d128(22, 2), &.{1844674407370955161500});

    // Decimal to float: Arrow's ToReal (the value times a float power of ten), bit for bit.
    var fx = try arr128(comptime d128(10, 2), &.{ 125, -375, 10, null });
    defer fx.release();
    try common.expectInvokeWith(f64, &ctx, "cast", &.{.{ .array = fx }}, castTo(.{ .double = {} }, true), &.{ 1.25, -3.75, 0.1, null });
    try common.expectInvokeWith(f32, &ctx, "cast", &.{.{ .array = fx }}, castTo(.{ .float = {} }, true), &.{ 1.25, -3.75, 0.099999994, null });
    var w46 = try common.makeTyped(i256, comptime d256(46, 1), allocator, &.{123456789012345678901234567890123456789012345});
    defer w46.release();
    try common.expectInvokeWith(f64, &ctx, "cast", &.{.{ .array = w46 }}, castTo(.{ .double = {} }, true), &.{1.2345678901234568e43});
    var negs = try arr128(comptime d128(5, -2), &.{12});
    defer negs.release();
    try common.expectInvokeWith(f64, &ctx, "cast", &.{.{ .array = negs }}, castTo(.{ .double = {} }, true), &.{1200.0});

    // Float to decimal: the exact binary value rounded half to even (Arrow's FromReal; a negative
    // scale rounds `x * 10^scale` in floating point, as Arrow).
    var fl = try common.makeArray(f64, allocator, &.{ 1.25, -3.755, 0.1, 1e10, null, 2.5, 0.125, -0.125, 0.375 });
    defer fl.release();
    try expectDec(i128, &ctx, "cast", &.{.{ .array = fl }}, castTo(d128(15, 2), true), d128(15, 2), &.{ 125, -375, 10, 1_000_000_000_000, null, 250, 12, -12, 38 });
    var halves = try common.makeArray(f64, allocator, &.{ 0.5, 1.5, 2.5, -0.5, -2.5, 0.05, 0.15, 0.25 });
    defer halves.release();
    try expectDec(i128, &ctx, "cast", &.{.{ .array = halves }}, castTo(d128(10, 0), true), d128(10, 0), &.{ 0, 2, 2, 0, -2, 0, 0, 0 });
    var tenths = try common.makeArray(f64, allocator, &.{ 0.05, 0.15, 0.25, 0.35, 1.005 });
    defer tenths.release();
    try expectDec(i128, &ctx, "cast", &.{.{ .array = tenths }}, castTo(d128(10, 1), true), d128(10, 1), &.{ 1, 1, 2, 3, 10 });
    var f32s = try common.makeArray(f32, allocator, &.{ 1.1, 0.3 });
    defer f32s.release();
    try expectDec(i128, &ctx, "cast", &.{.{ .array = f32s }}, castTo(d128(15, 10), true), d128(15, 10), &.{ 11000000238, 3000000119 });
    var huge_f = try common.makeArray(f64, allocator, &.{1.25e30});
    defer huge_f.release();
    try expectDec(i256, &ctx, "cast", &.{.{ .array = huge_f }}, castTo(d256(50, 2), true), d256(50, 2), &.{124999999999999995448703687065600});
    var thousands = try common.makeArray(f64, allocator, &.{1250.0});
    defer thousands.release();
    try expectDec(i128, &ctx, "cast", &.{.{ .array = thousands }}, castTo(d128(5, -2), true), d128(5, -2), &.{12});
    var bad_f = try common.makeArray(f64, allocator, &.{ 1e10, std.math.nan(f64), std.math.inf(f64) });
    defer bad_f.release();
    for (0..3) |k| {
        var one = try bad_f.slice(k, 1);
        defer one.release();
        try std.testing.expectError(error.InvalidCast, ctx.invokeScalar("cast", &.{.{ .array = one }}, castTo(d128(5, 2), true)));
    }
    try expectDec(i128, &ctx, "cast", &.{.{ .array = bad_f }}, castTo(d128(5, 2), false), d128(5, 2), &.{ 0, 0, 0 });

    // Decimal to text (Arrow ToString) in every string layout.
    var tx = try arr128(comptime d128(10, 2), &.{ 125, -375, 5, -5, 10000, null });
    defer tx.release();
    try common.expectInvokeStrings(&ctx, "cast", &.{.{ .array = tx }}, castTo(.{ .string = {} }, true), &.{ "1.25", "-3.75", "0.05", "-0.05", "100.00", null });
    try common.expectInvokeStrings(&ctx, "cast", &.{.{ .array = negs }}, castTo(.{ .large_string = {} }, true), &.{"1.2E+3"});
    var tiny = try arr128(comptime d128(30, 25), &.{1});
    defer tiny.release();
    try common.expectInvokeStrings(&ctx, "cast", &.{.{ .array = tiny }}, castTo(.{ .string_view = {} }, true), &.{"1E-25"});
    try common.expectInvokeStrings(&ctx, "cast", &.{.{ .array = w46 }}, castTo(.{ .string = {} }, true), &.{"12345678901234567890123456789012345678901234.5"});
    var small = try common.makeTyped(i32, comptime d32(3, 1), allocator, &.{-15});
    defer small.release();
    try common.expectInvokeStrings(&ctx, "cast", &.{.{ .array = small }}, castTo(.{ .string = {} }, true), &.{"-1.5"});

    // Text to decimal (Arrow FromString grammar).
    var texts = try common.makeStrings(.string, allocator, &.{ "1.25", "-3.7", "1.2e2", "1E-2", "+5", ".5", "5.", "-0", "0.00", "1.2E+3", null });
    defer texts.release();
    try expectDec(i128, &ctx, "cast", &.{.{ .array = texts }}, castTo(d128(10, 2), true), d128(10, 2), &.{ 125, -370, 12000, 1, 500, 50, 500, 0, 0, 120000, null });
    var bin = try common.makeStrings(.binary, allocator, &.{"1.25"});
    defer bin.release();
    try expectDec(i32, &ctx, "cast", &.{.{ .array = bin }}, castTo(d32(5, 2), true), d32(5, 2), &.{125});
    for ([_][]const u8{ "  1.2", "abc", "1e400", "1,5", "", "inf", "+." }) |t| {
        var one = try common.makeStrings(.string, allocator, &.{t});
        defer one.release();
        try std.testing.expectError(error.InvalidCast, ctx.invokeScalar("cast", &.{.{ .array = one }}, castTo(d128(10, 2), false)));
    }
    var lossy = try common.makeStrings(.large_string, allocator, &.{ "1.234", "12345678901" });
    defer lossy.release();
    try std.testing.expectError(error.InvalidCast, ctx.invokeScalar("cast", &.{.{ .array = lossy }}, castTo(d128(10, 2), true)));
    try expectDec(i128, &ctx, "cast", &.{.{ .array = lossy }}, castTo(d128(10, 2), false), d128(10, 2), &.{ 123, 1234567890100 });
    var digits39 = try common.makeStrings(.string_view, allocator, &.{"111111111111111111111111111111111111111"});
    defer digits39.release();
    try std.testing.expectError(error.InvalidCast, ctx.invokeScalar("cast", &.{.{ .array = digits39 }}, castTo(d128(38, 0), true)));
    try expectDec(i256, &ctx, "cast", &.{.{ .array = digits39 }}, castTo(d256(40, 0), true), d256(40, 0), &.{111111111111111111111111111111111111111});

    // Scalars and chunked inputs; bool, binary targets and bad target precision are rejected.
    try expectDec(i128, &ctx, "cast", &.{.{ .scalar = Scalar.init(d128(5, 1), .{ .decimal128 = 15 }) }}, castTo(d128(6, 2), true), d128(6, 2), &.{150});
    var ch = try common.chunkedAt(allocator, x, &.{ 2, 4 });
    defer ch.release();
    var cc = try ctx.invokeScalar("cast", &.{.{ .chunked = ch }}, castTo(d128(12, 4), true));
    defer cc.release();
    const cr = try rows(i128, cc);
    defer allocator.free(cr);
    try common.expectRows(i128, &.{ 12500, -37500, 500, -500, 13500, null }, cr);
    try std.testing.expectError(error.UnsupportedType, ctx.invokeScalar("cast", &xa, castTo(.{ .bool = {} }, true)));
    try std.testing.expectError(error.UnsupportedType, ctx.invokeScalar("cast", &xa, castTo(.{ .binary = {} }, true)));
    try std.testing.expectError(error.InvalidOptions, ctx.invokeScalar("cast", &xa, castTo(d128(0, 0), true)));
    try std.testing.expectError(error.InvalidOptions, ctx.invokeScalar("cast", &xa, castTo(d32(10, 0), true)));
}

fn aggregateScalar(ctx: *ExecContext, name: []const u8, arg: Datum, options: Options) !Scalar {
    var out = try ctx.invokeAggregate(name, &.{arg}, options);
    errdefer out.release();
    if (out != .scalar) return error.TestUnexpectedResult;
    return out.scalar;
}

fn expectAgg(comptime T: type, ctx: *ExecContext, name: []const u8, arg: Datum, options: Options, dt: DataType, want: ?T) !void {
    var s = try aggregateScalar(ctx, name, arg, options);
    defer s.release();
    try std.testing.expect(s.data_type.eql(dt));
    try std.testing.expect((try ctx.registry.resolveResultType(name, .aggregate, &.{arg}, options)).eql(dt));
    const got = try rows(T, .{ .scalar = s });
    defer std.testing.allocator.free(got);
    try common.expectRows(T, &.{want}, got);
}

test "decimal sum, mean, min and max: options, rounding, overflow modes, chunked and scalars" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);

    // [1.25, -3.50, null, 2.01]: sum -0.24 (precision 38), mean -0.08, min -3.50, max 2.01.
    var x = try arr128(comptime d128(10, 2), &.{ 125, -350, null, 201 });
    defer x.release();
    try expectAgg(i128, &ctx, "sum", .{ .array = x }, none, d128(38, 2), -24);
    try expectAgg(i128, &ctx, "mean", .{ .array = x }, none, d128(10, 2), -8);
    try expectAgg(i128, &ctx, "min", .{ .array = x }, none, d128(10, 2), -350);
    try expectAgg(i128, &ctx, "max", .{ .array = x }, none, d128(10, 2), 201);
    var w = try common.makeTyped(i256, comptime d256(40, 2), allocator, &.{ 125, -350, null, 201 });
    defer w.release();
    try expectAgg(i256, &ctx, "sum", .{ .array = w }, none, d256(76, 2), -24);
    try expectAgg(i256, &ctx, "mean", .{ .array = w }, none, d256(40, 2), -8);
    try expectAgg(i256, &ctx, "max", .{ .array = w }, none, d256(40, 2), 201);
    const no_skip: Options = .{ .scalar_aggregate = .{ .skip_nulls = false } };
    try expectAgg(i128, &ctx, "sum", .{ .array = x }, no_skip, d128(38, 2), null);
    try expectAgg(i128, &ctx, "min", .{ .array = x }, no_skip, d128(10, 2), null);
    try expectAgg(i128, &ctx, "max", .{ .array = x }, .{ .scalar_aggregate = .{ .min_count = 4 } }, d128(10, 2), null);

    // Mean rounds half away from zero.
    const means = [_]struct { v: []const ?i128, want: i128 }{
        .{ .v = &.{ 100, 1 }, .want = 51 },
        .{ .v = &.{ 100, 2, 3 }, .want = 35 },
        .{ .v = &.{ -100, -1 }, .want = -51 },
        .{ .v = &.{ 1, 1, 2 }, .want = 1 },
        .{ .v = &.{ 1, 1, 0 }, .want = 1 },
        .{ .v = &.{ -1, -1, 0 }, .want = -1 },
        .{ .v = &.{ 1, 0 }, .want = 1 },
        .{ .v = &.{ -1, 0 }, .want = -1 },
        .{ .v = &.{ 3, 0 }, .want = 2 },
        .{ .v = &.{ -3, 0 }, .want = -2 },
    };
    for (means) |m| {
        var arr = try arr128(comptime d128(10, 2), m.v);
        defer arr.release();
        try expectAgg(i128, &ctx, "mean", .{ .array = arr }, none, d128(10, 2), m.want);
    }

    // Empty input: null, or 0 with min_count 0 (sum); mean stays null (Arrow).
    var empty = try arr128(comptime d128(10, 2), &.{});
    defer empty.release();
    const mc0: Options = .{ .scalar_aggregate = .{ .min_count = 0 } };
    try expectAgg(i128, &ctx, "sum", .{ .array = empty }, none, d128(38, 2), null);
    try expectAgg(i128, &ctx, "sum", .{ .array = empty }, mc0, d128(38, 2), 0);
    try expectAgg(i128, &ctx, "mean", .{ .array = empty }, mc0, d128(10, 2), null);
    try expectAgg(i128, &ctx, "min", .{ .array = empty }, mc0, d128(10, 2), null);

    // A sum past precision 38: checked fails, wrapping keeps the low 128 bits (Arrow), saturating clamps.
    const nines: i128 = 99999999999999999999999999999999999999;
    var over = try arr128(comptime d128(38, 0), &.{ nines, nines });
    defer over.release();
    try std.testing.expectError(error.Overflow, ctx.invokeAggregate("sum", &.{.{ .array = over }}, none));
    var wrapping = ExecContext.initWithConfig(allocator, &registry, .{ .overflow_mode = .wrapping });
    try expectAgg(i128, &wrapping, "sum", .{ .array = over }, none, d128(38, 0), -140282366920938463463374607431768211458);
    var saturating = ExecContext.initWithConfig(allocator, &registry, .{ .overflow_mode = .saturating });
    try expectAgg(i128, &saturating, "sum", .{ .array = over }, none, d128(38, 0), nines);

    // Chunked, sliced and scalar inputs; a lifecycle merges states.
    var ch = try common.chunkedAt(allocator, x, &.{ 1, 3 });
    defer ch.release();
    try expectAgg(i128, &ctx, "sum", .{ .chunked = ch }, none, d128(38, 2), -24);
    var sl = try x.slice(1, 3);
    defer sl.release();
    try expectAgg(i128, &ctx, "min", .{ .array = sl }, none, d128(10, 2), -350);
    try expectAgg(i128, &ctx, "mean", .{ .scalar = Scalar.init(d128(5, 1), .{ .decimal128 = 15 }) }, none, d128(5, 1), 15);
    try expectAgg(i128, &ctx, "sum", .{ .scalar = Scalar.init(d128(5, 1), .null) }, none, d128(38, 1), null);
    var s1 = try ctx.beginAggregate("sum", &.{.{ .array = x }}, none);
    defer s1.deinit();
    var s2 = try ctx.beginAggregate("sum", &.{.{ .array = x }}, none);
    defer s2.deinit();
    try s1.update(&.{.{ .array = x }});
    try s2.update(&.{.{ .array = sl }});
    try s1.merge(&s2);
    var merged = try s1.finalize();
    defer merged.release();
    try std.testing.expectEqual(@as(i128, -24 - 149), merged.scalar.value.decimal128);

    // decimal32 / decimal64 have no aggregate kernels (as Arrow); a bad precision is malformed.
    var small = try common.makeTyped(i32, comptime d32(5, 2), allocator, &.{1});
    defer small.release();
    try std.testing.expectError(error.NoMatchingKernel, ctx.invokeAggregate("sum", &.{.{ .array = small }}, none));
    var bad = try arr128(comptime d128(39, 0), &.{1});
    defer bad.release();
    try std.testing.expectError(error.InvalidInput, ctx.invokeAggregate("sum", &.{.{ .array = bad }}, none));
}

test "decimal modulo and remainder: Arrow's scale rule, signs, nulls, sliced, chunked, integers, zero divisor (pyarrow 26 results)" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);

    // decimal128(5, 1) [7.5, -7.5, null] with decimal128(5, 2) [2.25, 2.25, 1.00] -> decimal128(6, 2).
    var a = try arr128(comptime d128(5, 1), &.{ 75, -75, null });
    defer a.release();
    var b = try arr128(comptime d128(5, 2), &.{ 225, 225, 100 });
    defer b.release();
    const ab = [_]Datum{ .{ .array = a }, .{ .array = b } };
    inline for (.{ "", "_checked" }) |suffix| {
        try expectDec(i128, &ctx, "modulo" ++ suffix, &ab, none, d128(6, 2), &.{ 75, 150, null });
        try expectDec(i128, &ctx, "remainder" ++ suffix, &ab, none, d128(6, 2), &.{ 75, -75, null });
    }
    // The sign of the divisor (modulo) or of the dividend (remainder): (5, 1) % (3, 0) -> (5, 1).
    var p = try arr128(comptime d128(5, 1), &.{ -75, 75 });
    defer p.release();
    var q = try arr128(comptime d128(3, 0), &.{ 2, -2 });
    defer q.release();
    try expectDec(i128, &ctx, "modulo", &.{ .{ .array = p }, .{ .array = q } }, none, d128(5, 1), &.{ 5, -5 });
    try expectDec(i128, &ctx, "remainder", &.{ .{ .array = p }, .{ .array = q } }, none, d128(5, 1), &.{ -15, 15 });
    var r = try arr128(comptime d128(3, 0), &.{1});
    defer r.release();
    var t = try arr128(comptime d128(5, 1), &.{30});
    defer t.release();
    try expectDec(i128, &ctx, "modulo", &.{ .{ .array = r }, .{ .array = t } }, none, d128(5, 1), &.{10});
    // Sliced and chunked.
    var ps = try p.slice(1, 1);
    defer ps.release();
    var qs = try q.slice(1, 1);
    defer qs.release();
    try expectDec(i128, &ctx, "modulo", &.{ .{ .array = ps }, .{ .array = qs } }, none, d128(5, 1), &.{-5});
    var cp = try common.chunkedAt(allocator, p, &.{1});
    defer cp.release();
    try expectDec(i128, &ctx, "remainder_checked", &.{ .{ .chunked = cp }, .{ .array = q } }, none, d128(5, 1), &.{ -15, 15 });
    // decimal32 operands give decimal128, decimal256 gives decimal256, an int32 is decimal(10, 0).
    var s32 = try common.makeTyped(i32, comptime d32(5, 1), allocator, &.{75});
    defer s32.release();
    var t32 = try common.makeTyped(i32, comptime d32(3, 0), allocator, &.{2});
    defer t32.release();
    try expectDec(i128, &ctx, "modulo", &.{ .{ .array = s32 }, .{ .array = t32 } }, none, d128(5, 1), &.{15});
    var w = try common.makeTyped(i256, comptime d256(5, 1), allocator, &.{ -75, 75 });
    defer w.release();
    try expectDec(i256, &ctx, "modulo", &.{ .{ .array = w }, .{ .array = q } }, none, d256(5, 1), &.{ 5, -5 });
    var i = try common.makeArray(i32, allocator, &.{7});
    defer i.release();
    var dd = try arr128(comptime d128(3, 1), &.{25});
    defer dd.release();
    try expectDec(i128, &ctx, "remainder", &.{ .{ .array = i }, .{ .array = dd } }, none, d128(11, 1), &.{20});
    try expectDec(i128, &ctx, "modulo", &.{ .{ .array = a }, .{ .scalar = Scalar.init(d128(3, 0), .{ .decimal128 = 2 }) } }, none, d128(5, 1), &.{ 15, 5, null });
    // A zero divisor in a valid row (Arrow: "divide by zero"); the precision limit.
    var z = try arr128(comptime d128(3, 0), &.{ 0, 1, 1 });
    defer z.release();
    try std.testing.expectError(error.DivideByZero, ctx.invokeScalar("modulo", &.{ .{ .array = a }, .{ .array = z } }, none));
    try std.testing.expectError(error.DivideByZero, ctx.invokeScalar("remainder_checked", &.{ .{ .array = a }, .{ .array = z } }, none));
    var zn = try arr128(comptime d128(3, 0), &.{ 1, 1, 0 });
    defer zn.release();
    try expectDec(i128, &ctx, "modulo", &.{ .{ .array = a }, .{ .array = zn } }, none, d128(5, 1), &.{ 5, 5, null });
    var big = try arr128(comptime d128(38, 0), &.{1});
    defer big.release();
    var big_s = try arr128(comptime d128(38, 10), &.{3});
    defer big_s.release();
    try std.testing.expectError(error.Overflow, ctx.invokeScalar("modulo", &.{ .{ .array = big }, .{ .array = big_s } }, none));
}
