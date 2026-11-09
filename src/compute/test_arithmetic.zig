const std = @import("std");
const common = @import("test_common.zig");

const ArrayRef = common.ArrayRef;
const Datum = common.Datum;
const ExecContext = common.ExecContext;
const dense = common.dense;
const none = common.none;
const Options = common.Options;
const TypeTag = std.meta.Tag(common.DataType);

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

    // Arrow C++ has no negate_checked for unsigned integers; numeric operands of different types
    // are cast to their common type (int8 + int16 is int16), other types are not.
    try std.testing.expectError(error.NoMatchingKernel, ctx.invokeScalar("negate_checked", &.{.{ .array = u }}, none));
    var wide = try common.makeArray(i16, allocator, &.{ 1, 2 });
    defer wide.release();
    try common.expectInvoke(i16, &ctx, .scalar, "add", &.{ .{ .array = ones8 }, .{ .array = wide } }, &.{ 2, 3 });
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

test "arithmetic implicit casts: mixed numeric operands use Arrow's common type (pyarrow 26 table)" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    const types = [_]type{ i8, i16, i32, i64, u8, u16, u32, u64, f32, f64 };
    // pyarrow 26: pc.add(a, b).type for every pair (row a, column b).
    const table = [10][10]TypeTag{
        .{ .int8, .int16, .int32, .int64, .int16, .int32, .int64, .int64, .float, .double },
        .{ .int16, .int16, .int32, .int64, .int16, .int32, .int64, .int64, .float, .double },
        .{ .int32, .int32, .int32, .int64, .int32, .int32, .int64, .int64, .float, .double },
        .{ .int64, .int64, .int64, .int64, .int64, .int64, .int64, .int64, .float, .double },
        .{ .int16, .int16, .int32, .int64, .uint8, .uint16, .uint32, .uint64, .float, .double },
        .{ .int32, .int32, .int32, .int64, .uint16, .uint16, .uint32, .uint64, .float, .double },
        .{ .int64, .int64, .int64, .int64, .uint32, .uint32, .uint32, .uint64, .float, .double },
        .{ .int64, .int64, .int64, .int64, .uint64, .uint64, .uint64, .uint64, .float, .double },
        .{ .float, .float, .float, .float, .float, .float, .float, .float, .float, .double },
        .{ .double, .double, .double, .double, .double, .double, .double, .double, .double, .double },
    };
    const to_double = Options{ .cast = .{ .to_type = .{ .double = {} } } };
    inline for (types, 0..) |A, i| {
        var a = try common.makeArray(A, allocator, &.{ 1, null, 3, 4 });
        defer a.release();
        inline for (types, 0..) |B, j| {
            var b = try common.makeArray(B, allocator, &.{ 2, 5, null, 6 });
            defer b.release();
            const args = [_]Datum{ .{ .array = a }, .{ .array = b } };
            try std.testing.expectEqual(table[i][j], std.meta.activeTag(try registry.resolveResultType("add", .scalar, &args, none)));
            var sum = try ctx.invokeScalar("add_checked", &args, none);
            defer sum.release();
            try std.testing.expectEqual(table[i][j], std.meta.activeTag(sum.dataType()));
            try common.expectInvokeWith(f64, &ctx, "cast", &.{sum}, to_double, &.{ 3, null, null, 10 });
        }
    }

    // Scalars are cast too; the common type decides overflow (int8 127 + int16 1 is 128).
    var small = try common.makeArray(i32, allocator, &.{ 1, 2 });
    defer small.release();
    var out = try ctx.invokeScalar("add", &.{ .{ .array = small }, .{ .scalar = common.scalarOf(i64, 1 << 40) } }, none);
    defer out.release();
    try std.testing.expectEqual(TypeTag.int64, std.meta.activeTag(out.dataType()));
    try common.expectInvoke(i16, &ctx, .scalar, "add_checked", &.{ .{ .scalar = common.scalarOf(i8, 127) }, .{ .scalar = common.scalarOf(i16, 1) } }, &.{128});
    try common.expectInvoke(i32, &ctx, .scalar, "divide", &.{ .{ .scalar = common.scalarOf(i8, 7) }, .{ .array = small } }, &.{ 7, 3 });
    // Chunked operands keep their chunks.
    var wide = try common.makeArray(i64, allocator, &.{ 10, 20, 30, 40, 50 });
    defer wide.release();
    var narrow = try common.makeArray(u16, allocator, &.{ 1, 2, 3, 4, 5 });
    defer narrow.release();
    var chunks = try common.chunkedAt(allocator, narrow, &.{2});
    defer chunks.release();
    var mixed = try ctx.invokeScalar("subtract", &.{ .{ .array = wide }, .{ .chunked = chunks } }, none);
    defer mixed.release();
    try std.testing.expectEqual(@as(usize, 2), mixed.chunked.numChunks());
    const got = try common.collect(i64, allocator, mixed);
    defer allocator.free(got);
    try common.expectRows(i64, &.{ 9, 18, 27, 36, 45 }, got);
    // The implicit cast is safe (Arrow): values it cannot represent are InvalidCast.
    var big = try common.makeArray(i64, allocator, &.{1 << 60});
    defer big.release();
    var one_f = try common.makeArray(f64, allocator, &.{1.0});
    defer one_f.release();
    try std.testing.expectError(error.InvalidCast, ctx.invokeScalar("add", &.{ .{ .array = big }, .{ .array = one_f } }, none));
    var huge = try common.makeArray(u64, allocator, &.{1 << 63});
    defer huge.release();
    var one = try common.makeArray(i64, allocator, &.{1});
    defer one.release();
    try std.testing.expectError(error.InvalidCast, ctx.invokeScalar("multiply", &.{ .{ .array = huge }, .{ .array = one } }, none));
    var unsafe_ctx = ExecContext.initWithConfig(allocator, &registry, .{ .safe_cast = false });
    var wrapped = try unsafe_ctx.invokeScalar("add", &.{ .{ .array = huge }, .{ .array = one } }, none);
    wrapped.release();
    // Non-numeric mismatches stay errors; negate / abs take one operand and never cast.
    var flags = try common.makeArray(bool, allocator, &.{ true, false });
    defer flags.release();
    try std.testing.expectError(error.NoMatchingKernel, ctx.invokeScalar("add", &.{ .{ .array = small }, .{ .array = flags } }, none));
    try std.testing.expectEqual(@as(?common.DataType, null), common.core.commonNumericType(&.{ .{ .array = small }, .{ .array = small } }));
}

test "modulo, remainder and their checked variants: signs, zero divisors, overflow, floats, nulls, sliced, chunked, scalars (pyarrow 26 results)" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);

    // pc.modulo([7, -7, 7, -7, None], [2, 2, -2, -2, 1]) -> [1, 1, -1, -1, null];
    // pc.remainder(...) -> [1, -1, 1, -1, null].
    var x = try common.makeArray(i64, allocator, &.{ 7, -7, 7, -7, null });
    defer x.release();
    var y = try common.makeArray(i64, allocator, &.{ 2, 2, -2, -2, 1 });
    defer y.release();
    const xy = [_]Datum{ .{ .array = x }, .{ .array = y } };
    inline for (.{ "", "_checked" }) |suffix| {
        try common.expectInvoke(i64, &ctx, .scalar, "modulo" ++ suffix, &xy, &.{ 1, 1, -1, -1, null });
        try common.expectInvoke(i64, &ctx, .scalar, "remainder" ++ suffix, &xy, &.{ 1, -1, 1, -1, null });
    }
    // Sliced operands.
    var xs = try x.slice(2, 3);
    defer xs.release();
    var ys = try y.slice(1, 3);
    defer ys.release();
    try common.expectInvoke(i64, &ctx, .scalar, "modulo", &.{ .{ .array = xs }, .{ .array = ys } }, &.{ 1, -1, null });
    // Chunked operands give a chunked result.
    var cx = try common.chunkedAt(allocator, x, &.{ 1, 3 });
    defer cx.release();
    var out = try ctx.invokeScalar("remainder", &.{ .{ .chunked = cx }, .{ .array = y } }, none);
    defer out.release();
    try std.testing.expect(out == .chunked);
    const got = try common.collect(i64, allocator, out);
    defer allocator.free(got);
    try common.expectRows(i64, &.{ 1, -1, 1, -1, null }, got);
    // Scalars: pc.modulo(7, -2) -> -1, pc.remainder(7, -2) -> 1.
    try common.expectInvoke(i64, &ctx, .scalar, "modulo", &.{ .{ .scalar = common.scalarOf(i64, 7) }, .{ .scalar = common.scalarOf(i64, -2) } }, &.{-1});
    try common.expectInvoke(i64, &ctx, .scalar, "remainder", &.{ .{ .scalar = common.scalarOf(i64, 7) }, .{ .scalar = common.scalarOf(i64, -2) } }, &.{1});
    try common.expectInvoke(i64, &ctx, .scalar, "modulo", &.{ .{ .array = x }, .{ .scalar = common.scalarOf(i64, null) } }, &.{ null, null, null, null, null });

    // int8 minInt by -1: 0 unchecked, Overflow checked (Arrow: "overflow").
    var m = try common.makeArray(i8, allocator, &.{ -128, 5 });
    defer m.release();
    var n1 = try common.makeArray(i8, allocator, &.{ -1, 3 });
    defer n1.release();
    const mn = [_]Datum{ .{ .array = m }, .{ .array = n1 } };
    try common.expectInvoke(i8, &ctx, .scalar, "modulo", &mn, &.{ 0, 2 });
    try common.expectInvoke(i8, &ctx, .scalar, "remainder", &mn, &.{ 0, 2 });
    try std.testing.expectError(error.Overflow, ctx.invokeScalar("modulo_checked", &mn, none));
    try std.testing.expectError(error.Overflow, ctx.invokeScalar("remainder_checked", &mn, none));
    // A zero divisor is an error in both variants (Arrow: "divide by zero"), not in a null row.
    var z = try common.makeArray(u8, allocator, &.{ 5, 3 });
    defer z.release();
    var zd = try common.makeArray(u8, allocator, &.{ 3, 0 });
    defer zd.release();
    inline for (.{ "modulo", "modulo_checked", "remainder", "remainder_checked" }) |name| {
        try std.testing.expectError(error.DivideByZero, ctx.invokeScalar(name, &.{ .{ .array = z }, .{ .array = zd } }, none));
    }
    var zn = try common.makeArray(i32, allocator, &.{ null, 9 });
    defer zn.release();
    var zz = try common.makeArray(i32, allocator, &.{ 0, 4 });
    defer zz.release();
    try common.expectInvoke(i32, &ctx, .scalar, "modulo_checked", &.{ .{ .array = zn }, .{ .array = zz } }, &.{ null, 1 });
    // Unsigned and mixed types (int8 % uint8 is int16: -7 % 3 = 2, remainder -1).
    var big = try common.makeArray(u64, allocator, &.{(1 << 63) + 5});
    defer big.release();
    var seven = try common.makeArray(u64, allocator, &.{7});
    defer seven.release();
    try common.expectInvoke(u64, &ctx, .scalar, "modulo", &.{ .{ .array = big }, .{ .array = seven } }, &.{6});
    var s8 = try common.makeArray(i8, allocator, &.{-7});
    defer s8.release();
    var u8a = try common.makeArray(u8, allocator, &.{3});
    defer u8a.release();
    try common.expectInvoke(i16, &ctx, .scalar, "modulo", &.{ .{ .array = s8 }, .{ .array = u8a } }, &.{2});
    try common.expectInvoke(i16, &ctx, .scalar, "remainder", &.{ .{ .array = s8 }, .{ .array = u8a } }, &.{-1});

    // Floats: modulo is Python's %, remainder is fmod; a zero divisor gives NaN unchecked.
    const inf = std.math.inf(f64);
    const nan = std.math.nan(f64);
    var fx = try common.makeArray(f64, allocator, &.{ 7.5, -7.5, 7.5, -7.5, 1.0, inf, 5.0, -5.0, 5.0, -0.0, 6.0, -6.0, nan, null });
    defer fx.release();
    var fy = try common.makeArray(f64, allocator, &.{ 2.0, 2.0, -2.0, -2.0, 0.0, 2.0, inf, inf, -inf, -2.0, -3.0, 3.0, 2.0, 1.0 });
    defer fy.release();
    const fxy = [_]Datum{ .{ .array = fx }, .{ .array = fy } };
    try common.expectInvoke(f64, &ctx, .scalar, "modulo", &fxy, &.{ 1.5, 0.5, -0.5, -1.5, nan, nan, 5.0, inf, -inf, -0.0, -0.0, 0.0, nan, null });
    try common.expectInvoke(f64, &ctx, .scalar, "remainder", &fxy, &.{ 1.5, -1.5, 1.5, -1.5, nan, nan, 5.0, -5.0, 5.0, -0.0, 0.0, -0.0, nan, null });
    // Signs of the zeros (expectRows compares -0.0 == 0.0).
    var fm = try ctx.invokeScalar("modulo", &fxy, none);
    defer fm.release();
    const fmv = try (common.array_mod.PrimitiveArray(f64){ .data = fm.array.data() }).values();
    try std.testing.expect(std.math.signbit(fmv[9]) and std.math.signbit(fmv[10]) and !std.math.signbit(fmv[11]));
    var fr = try ctx.invokeScalar("remainder", &fxy, none);
    defer fr.release();
    const frv = try (common.array_mod.PrimitiveArray(f64){ .data = fr.array.data() }).values();
    try std.testing.expect(std.math.signbit(frv[9]) and !std.math.signbit(frv[10]) and std.math.signbit(frv[11]));
    try std.testing.expectError(error.DivideByZero, ctx.invokeScalar("modulo_checked", &fxy, none));
    try std.testing.expectError(error.DivideByZero, ctx.invokeScalar("remainder_checked", &fxy, none));
    var f32x = try common.makeArray(f32, allocator, &.{-7.5});
    defer f32x.release();
    try common.expectInvoke(f32, &ctx, .scalar, "modulo", &.{ .{ .array = f32x }, .{ .scalar = common.scalarOf(f32, 2) } }, &.{0.5});
    // int32 % double is double: pc.modulo([7], [2.5]) -> [2.0].
    var seven_i = try common.makeArray(i32, allocator, &.{7});
    defer seven_i.release();
    try common.expectInvoke(f64, &ctx, .scalar, "modulo_checked", &.{ .{ .array = seven_i }, .{ .scalar = common.scalarOf(f64, 2.5) } }, &.{2.0});
    // No bool or float16 kernel (as Arrow).
    var bools = try common.makeArray(bool, allocator, &.{true});
    defer bools.release();
    try std.testing.expectError(error.NoMatchingKernel, ctx.invokeScalar("modulo", &.{ .{ .array = bools }, .{ .array = bools } }, none));
}
