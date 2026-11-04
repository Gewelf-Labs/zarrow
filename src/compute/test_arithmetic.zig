const std = @import("std");
const common = @import("test_common.zig");

const ArrayRef = common.ArrayRef;
const Datum = common.Datum;
const ExecContext = common.ExecContext;
const dense = common.dense;
const none = common.none;

const binary_ops = [_][]const u8{ "add", "subtract", "multiply", "divide" };
const unary_ops = [_][]const u8{ "negate", "abs" };

/// Per-element reference of the unchecked functions (`sat`: ExecConfig.overflow_mode = saturating).
fn reference(comptime T: type, comptime op: []const u8, sat: bool, x: T, y: T) T {
    const eq = std.mem.eql;
    if (@typeInfo(T) == .float) {
        if (comptime eq(u8, op, "add")) return x + y;
        if (comptime eq(u8, op, "subtract")) return x - y;
        if (comptime eq(u8, op, "multiply")) return x * y;
        if (comptime eq(u8, op, "divide")) return x / y;
        if (comptime eq(u8, op, "negate")) return -x;
        return @abs(x);
    }
    const signed = @typeInfo(T).int.signedness == .signed;
    const min = std.math.minInt(T);
    const max = std.math.maxInt(T);
    if (comptime eq(u8, op, "add")) return if (sat) x +| y else x +% y;
    if (comptime eq(u8, op, "subtract")) return if (sat) x -| y else x -% y;
    if (comptime eq(u8, op, "multiply")) return if (sat) x *| y else x *% y;
    if (comptime eq(u8, op, "divide")) {
        if (signed) {
            if (x == min and y == -1) return if (sat) max else 0;
        }
        return @divTrunc(x, y);
    }
    if (comptime eq(u8, op, "negate")) return if (sat) 0 -| x else 0 -% x;
    if (!signed) return x;
    if (x == min) return if (sat) max else min;
    return @intCast(@abs(x));
}

fn referenceRows(comptime T: type, comptime op: []const u8, sat: bool, allocator: std.mem.Allocator, a: []const ?T, b: ?[]const ?T) ![]?T {
    const len = if (a.len == 1 and b != null) b.?.len else a.len;
    const out = try allocator.alloc(?T, len);
    for (out, 0..) |*o, i| {
        const x = a[if (a.len == 1) 0 else i];
        const y: ?T = if (b) |bb| bb[if (bb.len == 1) 0 else i] else 0;
        o.* = if (x == null or y == null) null else reference(T, op, sat, x.?, y.?);
    }
    return out;
}

fn expectMatchesReference(comptime T: type, comptime op: []const u8, ctx: *ExecContext, name: []const u8, args: []const Datum) !void {
    const allocator = std.testing.allocator;
    const a = try common.collect(T, allocator, args[0]);
    defer allocator.free(a);
    const b = if (args.len > 1) try common.collect(T, allocator, args[1]) else null;
    defer if (b) |bb| allocator.free(bb);
    const expected = try referenceRows(T, op, ctx.overflowMode() == .saturating, allocator, a, b);
    defer allocator.free(expected);
    try common.expectInvoke(T, ctx, .scalar, name, args, expected);
}

test "arithmetic hand-written values: wrapping, saturating, truncating division, float division by zero" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);

    var a = try common.makeArray(i32, allocator, &.{ 7, -7, null, 100, std.math.minInt(i32) });
    defer a.release();
    var b = try common.makeArray(i32, allocator, &.{ 2, 2, 5, null, -1 });
    defer b.release();
    const ab = [_]Datum{ .{ .array = a }, .{ .array = b } };
    try common.expectInvoke(i32, &ctx, .scalar, "add", &ab, &.{ 9, -5, null, null, std.math.maxInt(i32) });
    try common.expectInvoke(i32, &ctx, .scalar, "subtract", &ab, &.{ 5, -9, null, null, std.math.minInt(i32) + 1 });
    try common.expectInvoke(i32, &ctx, .scalar, "multiply", &ab, &.{ 14, -14, null, null, std.math.minInt(i32) });
    // Truncation toward zero; minInt / -1 is 0 like Arrow C++.
    try common.expectInvoke(i32, &ctx, .scalar, "divide", &ab, &.{ 3, -3, null, null, 0 });
    try common.expectInvoke(i32, &ctx, .scalar, "negate", &.{.{ .array = a }}, &.{ -7, 7, null, -100, std.math.minInt(i32) });
    try common.expectInvoke(i32, &ctx, .scalar, "abs", &.{.{ .array = a }}, &.{ 7, 7, null, 100, std.math.minInt(i32) });

    var i8a = try common.makeArray(i8, allocator, &.{ 127, -128, 100 });
    defer i8a.release();
    var i8b = try common.makeArray(i8, allocator, &.{ 1, -1, 2 });
    defer i8b.release();
    const i8ab = [_]Datum{ .{ .array = i8a }, .{ .array = i8b } };
    try common.expectInvoke(i8, &ctx, .scalar, "add", &i8ab, &.{ -128, 127, 102 });
    try common.expectInvoke(i8, &ctx, .scalar, "multiply", &i8ab, &.{ 127, -128, -56 });

    var u8a = try common.makeArray(u8, allocator, &.{ 0, 1, 200 });
    defer u8a.release();
    var u8b = try common.makeArray(u8, allocator, &.{ 1, 1, 100 });
    defer u8b.release();
    const u8ab = [_]Datum{ .{ .array = u8a }, .{ .array = u8b } };
    try common.expectInvoke(u8, &ctx, .scalar, "subtract", &u8ab, &.{ 255, 0, 100 });
    try common.expectInvoke(u8, &ctx, .scalar, "negate", &.{.{ .array = u8a }}, &.{ 0, 255, 56 });
    try common.expectInvoke(u8, &ctx, .scalar, "abs_checked", &.{.{ .array = u8a }}, &.{ 0, 1, 200 });

    // Saturating mode clamps the unchecked functions; the checked ones still fail.
    var sat = ExecContext.initWithConfig(allocator, &registry, .{ .overflow_mode = .saturating });
    try common.expectInvoke(i8, &sat, .scalar, "add", &i8ab, &.{ 127, -128, 102 });
    try common.expectInvoke(i8, &sat, .scalar, "multiply", &i8ab, &.{ 127, 127, 127 });
    try common.expectInvoke(i8, &sat, .scalar, "divide", &i8ab, &.{ 127, 127, 50 });
    try common.expectInvoke(i8, &sat, .scalar, "negate", &.{.{ .array = i8a }}, &.{ -127, 127, -100 });
    try common.expectInvoke(i8, &sat, .scalar, "abs", &.{.{ .array = i8a }}, &.{ 127, 127, 100 });
    try common.expectInvoke(u8, &sat, .scalar, "subtract", &u8ab, &.{ 0, 0, 100 });
    try std.testing.expectError(error.Overflow, sat.invokeScalar("add_checked", &i8ab, none));

    var fa = try common.makeArray(f64, allocator, &.{ 1.0, 0.0, -1.0, 1.5, null });
    defer fa.release();
    var fb = try common.makeArray(f64, allocator, &.{ 0.0, 0.0, 0.0, -0.5, 1.0 });
    defer fb.release();
    const fab = [_]Datum{ .{ .array = fa }, .{ .array = fb } };
    try common.expectInvoke(f64, &ctx, .scalar, "divide", &fab, &.{ std.math.inf(f64), std.math.nan(f64), -std.math.inf(f64), -3.0, null });
    try common.expectInvoke(f64, &ctx, .scalar, "add_checked", &fab, &.{ 1.0, 0.0, -1.0, 1.0, null });
    try common.expectInvoke(f64, &ctx, .scalar, "abs", &.{.{ .array = fb }}, &.{ 0.0, 0.0, 0.0, 0.5, 1.0 });
}

test "arithmetic oracle matches a per-element loop on random sliced inputs with nulls for every numeric type" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var prng = std.Random.DefaultPrng.init(0xA417);
    const rng = prng.random();
    const configs = [_]common.core.ExecConfig{ .{ .overflow_mode = .wrapping }, .{ .overflow_mode = .saturating }, .{ .overflow_mode = .checked } };
    inline for (dense.numeric_types) |T| {
        for (configs) |config| {
            var ctx = ExecContext.initWithConfig(allocator, &registry, config);
            var a = try common.randomArray(T, allocator, rng, 517, 3, 20, .{});
            defer a.release();
            var b = try common.randomArray(T, allocator, rng, 517, 13, 20, .{});
            defer b.release();
            var nz = try common.randomArray(T, allocator, rng, 517, 5, 20, .{ .nonzero = true });
            defer nz.release();
            const s = common.scalarOf(T, if (@typeInfo(T) == .float) common.randomValue(T, rng) else 3);
            inline for (binary_ops) |op| {
                const rhs: ArrayRef = if (comptime std.mem.eql(u8, op, "divide")) nz else b;
                try expectMatchesReference(T, op, &ctx, op, &.{ .{ .array = a }, .{ .array = rhs } });
                try expectMatchesReference(T, op, &ctx, op, &.{ .{ .array = a }, .{ .scalar = s } });
                try expectMatchesReference(T, op, &ctx, op, &.{ .{ .scalar = s }, .{ .array = rhs } });
            }
            inline for (unary_ops) |op| try expectMatchesReference(T, op, &ctx, op, &.{.{ .array = a }});

            // The checked variants agree with the unchecked ones when nothing overflows.
            var sa = try common.randomArray(T, allocator, rng, 300, 7, 20, .{ .small = true });
            defer sa.release();
            var sb = try common.randomArray(T, allocator, rng, 300, 1, 20, .{ .small = true, .nonzero = true });
            defer sb.release();
            const unsigned = @typeInfo(T) == .int and @typeInfo(T).int.signedness == .unsigned;
            inline for (binary_ops) |op| {
                // Small unsigned values still underflow when subtracted (covered by the error test).
                if (unsigned and comptime std.mem.eql(u8, op, "subtract")) continue;
                try expectMatchesReference(T, op, &ctx, op ++ "_checked", &.{ .{ .array = sa }, .{ .array = sb } });
                try expectMatchesReference(T, op, &ctx, op ++ "_checked", &.{ .{ .scalar = common.scalarOf(T, 2) }, .{ .array = sb } });
            }
            try expectMatchesReference(T, "abs", &ctx, "abs_checked", &.{.{ .array = sa }});
            if (@typeInfo(T) == .float or @typeInfo(T).int.signedness == .signed) {
                try expectMatchesReference(T, "negate", &ctx, "negate_checked", &.{.{ .array = sa }});
            }
        }
    }
}

test "arithmetic chunked inputs with misaligned chunks give a chunked result equal to the array result" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    var prng = std.Random.DefaultPrng.init(0xC4A1);
    const rng = prng.random();
    inline for (.{ i32, u16, f64 }) |T| {
        var a = try common.randomArray(T, allocator, rng, 600, 3, 15, .{ .small = true });
        defer a.release();
        var b = try common.randomArray(T, allocator, rng, 600, 0, 15, .{ .small = true, .nonzero = true });
        defer b.release();
        var ca = try common.chunkedAt(allocator, a, &.{ 7, 300, 301 });
        defer ca.release();
        var cb = try common.chunkedAt(allocator, b, &.{ 64, 500 });
        defer cb.release();
        inline for (binary_ops ++ [_][]const u8{"add_checked"}) |name| {
            var whole = try ctx.invokeScalar(name, &.{ .{ .array = a }, .{ .array = b } }, none);
            defer whole.release();
            const expected = try common.collect(T, allocator, whole);
            defer allocator.free(expected);

            var chunked = try ctx.invokeScalar(name, &.{ .{ .chunked = ca }, .{ .chunked = cb } }, none);
            defer chunked.release();
            try std.testing.expect(chunked == .chunked);
            // Boundaries are the union of both inputs: 7, 64, 300, 301, 500.
            try std.testing.expectEqual(@as(usize, 6), chunked.chunked.numChunks());
            const got = try common.collect(T, allocator, chunked);
            defer allocator.free(got);
            try common.expectRows(T, expected, got);

            var mixed = try ctx.invokeScalar(name, &.{ .{ .array = a }, .{ .chunked = cb } }, none);
            defer mixed.release();
            try std.testing.expectEqual(@as(usize, 3), mixed.chunked.numChunks());
            const got_mixed = try common.collect(T, allocator, mixed);
            defer allocator.free(got_mixed);
            try common.expectRows(T, expected, got_mixed);
        }
        var neg = try ctx.invokeScalar("negate", &.{.{ .chunked = ca }}, none);
        defer neg.release();
        try std.testing.expectEqual(@as(usize, 4), neg.chunked.numChunks());
    }
}

test "arithmetic scalar broadcast: array-scalar, scalar-array, scalar-scalar and null scalars" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    var a = try common.makeArray(i64, allocator, &.{ 10, null, -4 });
    defer a.release();
    const three = common.scalarOf(i64, 3);
    const null_scalar = common.scalarOf(i64, null);
    try common.expectInvoke(i64, &ctx, .scalar, "subtract", &.{ .{ .array = a }, .{ .scalar = three } }, &.{ 7, null, -7 });
    try common.expectInvoke(i64, &ctx, .scalar, "subtract", &.{ .{ .scalar = three }, .{ .array = a } }, &.{ -7, null, 7 });
    try common.expectInvoke(i64, &ctx, .scalar, "divide", &.{ .{ .array = a }, .{ .scalar = three } }, &.{ 3, null, -1 });
    try common.expectInvoke(i64, &ctx, .scalar, "add", &.{ .{ .array = a }, .{ .scalar = null_scalar } }, &.{ null, null, null });
    // A null divisor scalar is not a division by zero.
    try common.expectInvoke(i64, &ctx, .scalar, "divide_checked", &.{ .{ .array = a }, .{ .scalar = null_scalar } }, &.{ null, null, null });

    var out = try ctx.invokeScalar("multiply", &.{ .{ .scalar = three }, .{ .scalar = common.scalarOf(i64, -5) } }, none);
    defer out.release();
    try std.testing.expect(out == .scalar);
    try std.testing.expectEqual(@as(i64, -15), out.scalar.value.i64);
    var neg = try ctx.invokeScalar("negate", &.{.{ .scalar = null_scalar }}, none);
    defer neg.release();
    try std.testing.expect(neg.scalar.isNull());
    try std.testing.expect(neg.scalar.data_type == .int64);
}

test "arithmetic empty and all-null inputs: no error from overflow or zero divisors in null rows" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    var empty = try common.makeArray(i16, allocator, &.{});
    defer empty.release();
    inline for (binary_ops) |op| {
        try common.expectInvoke(i16, &ctx, .scalar, op, &.{ .{ .array = empty }, .{ .array = empty } }, &.{});
        try common.expectInvoke(i16, &ctx, .scalar, op ++ "_checked", &.{ .{ .array = empty }, .{ .scalar = common.scalarOf(i16, 0) } }, &.{});
    }
    // Null rows keep values that would overflow or divide by zero: nothing is reported for them.
    const lhs_values = [_]i16{ std.math.minInt(i16), std.math.maxInt(i16), 5, 1 };
    const rhs_values = [_]i16{ -1, 1, 0, 0 };
    const all_null = [_]u8{0};
    var a = try common.borrowedArray(i16, allocator, &lhs_values, &all_null);
    defer a.release();
    var b = try common.borrowedArray(i16, allocator, &rhs_values, &all_null);
    defer b.release();
    const ab = [_]Datum{ .{ .array = a }, .{ .array = b } };
    inline for (binary_ops) |op| {
        try common.expectInvoke(i16, &ctx, .scalar, op, &ab, &.{ null, null, null, null });
        try common.expectInvoke(i16, &ctx, .scalar, op ++ "_checked", &ab, &.{ null, null, null, null });
    }
    try common.expectInvoke(i16, &ctx, .scalar, "negate_checked", &.{.{ .array = a }}, &.{ null, null, null, null });
    try common.expectInvoke(i16, &ctx, .scalar, "abs_checked", &.{.{ .array = a }}, &.{ null, null, null, null });

    // Only row 3 is valid (1 / 0, 1 + 0): only that row can fail.
    const last_valid = [_]u8{0b1000};
    var a3 = try common.borrowedArray(i16, allocator, &lhs_values, &last_valid);
    defer a3.release();
    var b3 = try common.borrowedArray(i16, allocator, &rhs_values, &last_valid);
    defer b3.release();
    const ab3 = [_]Datum{ .{ .array = a3 }, .{ .array = b3 } };
    try std.testing.expectError(error.DivideByZero, ctx.invokeScalar("divide", &ab3, none));
    try common.expectInvoke(i16, &ctx, .scalar, "add_checked", &ab3, &.{ null, null, null, 1 });
    try common.expectInvoke(i16, &ctx, .scalar, "multiply_checked", &ab3, &.{ null, null, null, 0 });
}

test "arithmetic errors: Overflow, DivideByZero, mismatched types, options, lengths and malformed input" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    var max8 = try common.makeArray(i8, allocator, &.{ 1, 127 });
    defer max8.release();
    var min8 = try common.makeArray(i8, allocator, &.{ 1, -128 });
    defer min8.release();
    var ones8 = try common.makeArray(i8, allocator, &.{ 1, 1 });
    defer ones8.release();
    var neg8 = try common.makeArray(i8, allocator, &.{ 1, -1 });
    defer neg8.release();
    var zero8 = try common.makeArray(i8, allocator, &.{ 1, 0 });
    defer zero8.release();
    try std.testing.expectError(error.Overflow, ctx.invokeScalar("add_checked", &.{ .{ .array = max8 }, .{ .array = ones8 } }, none));
    try std.testing.expectError(error.Overflow, ctx.invokeScalar("subtract_checked", &.{ .{ .array = min8 }, .{ .array = ones8 } }, none));
    try std.testing.expectError(error.Overflow, ctx.invokeScalar("multiply_checked", &.{ .{ .array = max8 }, .{ .array = max8 } }, none));
    try std.testing.expectError(error.Overflow, ctx.invokeScalar("divide_checked", &.{ .{ .array = min8 }, .{ .array = neg8 } }, none));
    try std.testing.expectError(error.Overflow, ctx.invokeScalar("negate_checked", &.{.{ .array = min8 }}, none));
    try std.testing.expectError(error.Overflow, ctx.invokeScalar("abs_checked", &.{.{ .array = min8 }}, none));
    try std.testing.expectError(error.Overflow, ctx.invokeScalar("add_checked", &.{ .{ .array = max8 }, .{ .scalar = common.scalarOf(i8, 1) } }, none));
    try std.testing.expectError(error.Overflow, ctx.invokeScalar("add_checked", &.{ .{ .scalar = common.scalarOf(i8, 127) }, .{ .scalar = common.scalarOf(i8, 1) } }, none));
    // Integer division by zero fails in both variants (Arrow C++); float only when checked.
    try std.testing.expectError(error.DivideByZero, ctx.invokeScalar("divide", &.{ .{ .array = ones8 }, .{ .array = zero8 } }, none));
    try std.testing.expectError(error.DivideByZero, ctx.invokeScalar("divide_checked", &.{ .{ .array = ones8 }, .{ .array = zero8 } }, none));
    try std.testing.expectError(error.DivideByZero, ctx.invokeScalar("divide", &.{ .{ .array = ones8 }, .{ .scalar = common.scalarOf(i8, 0) } }, none));
    var f = try common.makeArray(f32, allocator, &.{ 1, 2 });
    defer f.release();
    try std.testing.expectError(error.DivideByZero, ctx.invokeScalar("divide_checked", &.{ .{ .array = f }, .{ .scalar = common.scalarOf(f32, 0) } }, none));
    var u = try common.makeArray(u8, allocator, &.{ 0, 1 });
    defer u.release();
    try std.testing.expectError(error.Overflow, ctx.invokeScalar("subtract_checked", &.{ .{ .array = u }, .{ .scalar = common.scalarOf(u8, 1) } }, none));

    // Arrow C++ has no negate_checked for unsigned integers; there are no implicit casts.
    try std.testing.expectError(error.NoMatchingKernel, ctx.invokeScalar("negate_checked", &.{.{ .array = u }}, none));
    var wide = try common.makeArray(i16, allocator, &.{ 1, 2 });
    defer wide.release();
    try std.testing.expectError(error.NoMatchingKernel, ctx.invokeScalar("add", &.{ .{ .array = ones8 }, .{ .array = wide } }, none));
    var b = try common.makeArray(bool, allocator, &.{ true, false });
    defer b.release();
    try std.testing.expectError(error.NoMatchingKernel, ctx.invokeScalar("add", &.{ .{ .array = b }, .{ .array = b } }, none));
    try std.testing.expectError(error.InvalidArity, ctx.invokeScalar("add", &.{.{ .array = ones8 }}, none));
    try std.testing.expectError(error.InvalidOptions, ctx.invokeScalar("add", &.{ .{ .array = ones8 }, .{ .array = ones8 } }, .{ .arithmetic = .{} }));
    var three = try common.makeArray(i8, allocator, &.{ 1, 2, 3 });
    defer three.release();
    try std.testing.expectError(error.InvalidInput, ctx.invokeScalar("add", &.{ .{ .array = ones8 }, .{ .array = three } }, none));
    // A scalar whose payload does not match its type is rejected, not reinterpreted.
    const bad = common.Scalar.init(.{ .int8 = {} }, .{ .i64 = 1 });
    try std.testing.expectError(error.InvalidInput, ctx.invokeScalar("add", &.{ .{ .array = ones8 }, .{ .scalar = bad } }, none));
    // A values buffer shorter than the array fails validateLayout instead of being read.
    const short = [_]i8{1};
    const bufs = [_]common.SharedBuffer{ .empty, .fromSlice(std.mem.sliceAsBytes(&short)) };
    var malformed = try ArrayRef.fromBorrowed(allocator, .{ .data_type = .{ .int8 = {} }, .length = 2, .null_count = 0, .buffers = &bufs });
    defer malformed.release();
    try std.testing.expectError(error.InvalidInput, ctx.invokeScalar("add", &.{ .{ .array = malformed }, .{ .array = ones8 } }, none));
}
