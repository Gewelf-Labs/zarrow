const std = @import("std");
const common = @import("test_common.zig");

const ArrayRef = common.ArrayRef;
const Datum = common.Datum;
const ExecContext = common.ExecContext;
const none = common.none;

const binary_ops = [_][]const u8{ "and", "or", "xor", "and_not", "and_kleene", "or_kleene" };

/// Per-element reference: null intersection, or Kleene logic for the `_kleene` functions.
fn reference(comptime op: []const u8, x: ?bool, y: ?bool) ?bool {
    const eq = std.mem.eql;
    if (comptime eq(u8, op, "and_kleene")) {
        if (x == false or y == false) return false;
        if (x == null or y == null) return null;
        return true;
    }
    if (comptime eq(u8, op, "or_kleene")) {
        if (x == true or y == true) return true;
        if (x == null or y == null) return null;
        return false;
    }
    if (x == null or y == null) return null;
    if (comptime eq(u8, op, "and")) return x.? and y.?;
    if (comptime eq(u8, op, "or")) return x.? or y.?;
    if (comptime eq(u8, op, "xor")) return x.? != y.?;
    return x.? and !y.?;
}

fn expectMatchesReference(comptime op: []const u8, ctx: *ExecContext, args: []const Datum) !void {
    const allocator = std.testing.allocator;
    const a = try common.collect(bool, allocator, args[0]);
    defer allocator.free(a);
    const b = try common.collect(bool, allocator, args[1]);
    defer allocator.free(b);
    const expected = try allocator.alloc(?bool, @max(a.len, b.len));
    defer allocator.free(expected);
    for (expected, 0..) |*e, i| e.* = reference(op, a[if (a.len == 1) 0 else i], b[if (b.len == 1) 0 else i]);
    try common.expectInvoke(bool, ctx, .scalar, op, args, expected);
}

test "logical hand-written truth tables including Kleene logic (Arrow C++ results)" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    // Every pair of {true, false, null}.
    var a = try common.makeArray(bool, allocator, &.{ true, true, true, false, false, false, null, null, null });
    defer a.release();
    var b = try common.makeArray(bool, allocator, &.{ true, false, null, true, false, null, true, false, null });
    defer b.release();
    const ab = [_]Datum{ .{ .array = a }, .{ .array = b } };
    try common.expectInvoke(bool, &ctx, .scalar, "and", &ab, &.{ true, false, null, false, false, null, null, null, null });
    try common.expectInvoke(bool, &ctx, .scalar, "or", &ab, &.{ true, true, null, true, false, null, null, null, null });
    try common.expectInvoke(bool, &ctx, .scalar, "xor", &ab, &.{ false, true, null, true, false, null, null, null, null });
    try common.expectInvoke(bool, &ctx, .scalar, "and_not", &ab, &.{ false, true, null, false, false, null, null, null, null });
    try common.expectInvoke(bool, &ctx, .scalar, "and_kleene", &ab, &.{ true, false, null, false, false, false, null, false, null });
    try common.expectInvoke(bool, &ctx, .scalar, "or_kleene", &ab, &.{ true, true, true, true, false, null, true, null, null });
    try common.expectInvoke(bool, &ctx, .scalar, "invert", &.{.{ .array = a }}, &.{ false, false, false, true, true, true, null, null, null });
}

test "logical oracle matches a per-element loop on random inputs at unaligned bit offsets" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    var prng = std.Random.DefaultPrng.init(0x10C1);
    const rng = prng.random();
    // Offsets 0 / 8 take the byte-aligned path, the others the shifted one; lengths end mid-byte.
    const shapes = [_][3]usize{ .{ 1000, 0, 8 }, .{ 1001, 3, 5 }, .{ 77, 13, 0 }, .{ 9, 1, 1 } };
    for (shapes) |shape| {
        var a = try common.randomArray(bool, allocator, rng, shape[0], shape[1], 25, .{});
        defer a.release();
        var b = try common.randomArray(bool, allocator, rng, shape[0], shape[2], 25, .{});
        defer b.release();
        inline for (binary_ops) |op| {
            try expectMatchesReference(op, &ctx, &.{ .{ .array = a }, .{ .array = b } });
            for ([_]?bool{ true, false, null }) |s| {
                try expectMatchesReference(op, &ctx, &.{ .{ .array = a }, .{ .scalar = common.scalarOf(bool, s) } });
                try expectMatchesReference(op, &ctx, &.{ .{ .scalar = common.scalarOf(bool, s) }, .{ .array = b } });
            }
        }
        const rows = try common.collect(bool, allocator, .{ .array = a });
        defer allocator.free(rows);
        const inverted = try allocator.alloc(?bool, rows.len);
        defer allocator.free(inverted);
        for (inverted, rows) |*o, r| o.* = if (r) |v| !v else null;
        try common.expectInvoke(bool, &ctx, .scalar, "invert", &.{.{ .array = a }}, inverted);
    }
}

test "logical chunked inputs with misaligned chunks give a chunked result equal to the array result" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    var prng = std.Random.DefaultPrng.init(0x10C2);
    const rng = prng.random();
    var a = try common.randomArray(bool, allocator, rng, 900, 5, 20, .{});
    defer a.release();
    var b = try common.randomArray(bool, allocator, rng, 900, 2, 20, .{});
    defer b.release();
    var ca = try common.chunkedAt(allocator, a, &.{ 1, 17, 450 });
    defer ca.release();
    var cb = try common.chunkedAt(allocator, b, &.{ 9, 450, 899 });
    defer cb.release();
    inline for (binary_ops) |op| {
        var whole = try ctx.invokeScalar(op, &.{ .{ .array = a }, .{ .array = b } }, none);
        defer whole.release();
        const expected = try common.collect(bool, allocator, whole);
        defer allocator.free(expected);
        var chunked = try ctx.invokeScalar(op, &.{ .{ .chunked = ca }, .{ .chunked = cb } }, none);
        defer chunked.release();
        // Boundaries 1, 9, 17, 450, 899.
        try std.testing.expectEqual(@as(usize, 6), chunked.chunked.numChunks());
        const got = try common.collect(bool, allocator, chunked);
        defer allocator.free(got);
        try common.expectRows(bool, expected, got);
    }
    var inv = try ctx.invokeScalar("invert", &.{.{ .chunked = ca }}, none);
    defer inv.release();
    try std.testing.expectEqual(@as(usize, 4), inv.chunked.numChunks());
}

test "logical scalar operands: Kleene keeps the deciding side of a null scalar, scalar-scalar gives a scalar" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    var a = try common.makeArray(bool, allocator, &.{ true, false, null });
    defer a.release();
    const null_bool = common.scalarOf(bool, null);
    try common.expectInvoke(bool, &ctx, .scalar, "and_kleene", &.{ .{ .array = a }, .{ .scalar = null_bool } }, &.{ null, false, null });
    try common.expectInvoke(bool, &ctx, .scalar, "or_kleene", &.{ .{ .scalar = null_bool }, .{ .array = a } }, &.{ true, null, null });
    try common.expectInvoke(bool, &ctx, .scalar, "and", &.{ .{ .array = a }, .{ .scalar = null_bool } }, &.{ null, null, null });
    try common.expectInvoke(bool, &ctx, .scalar, "xor", &.{ .{ .array = a }, .{ .scalar = common.scalarOf(bool, true) } }, &.{ false, true, null });

    const cases = [_]struct { name: []const u8, x: ?bool, y: ?bool, want: ?bool }{
        .{ .name = "and_kleene", .x = false, .y = null, .want = false },
        .{ .name = "and_kleene", .x = true, .y = null, .want = null },
        .{ .name = "or_kleene", .x = null, .y = true, .want = true },
        .{ .name = "and", .x = false, .y = null, .want = null },
        .{ .name = "and_not", .x = true, .y = false, .want = true },
    };
    for (cases) |c| {
        var out = try ctx.invokeScalar(c.name, &.{ .{ .scalar = common.scalarOf(bool, c.x) }, .{ .scalar = common.scalarOf(bool, c.y) } }, none);
        defer out.release();
        try std.testing.expect(out == .scalar);
        try std.testing.expectEqual(c.want, if (out.scalar.isNull()) null else out.scalar.value.bool);
    }
    var inv = try ctx.invokeScalar("invert", &.{.{ .scalar = common.scalarOf(bool, true) }}, none);
    defer inv.release();
    try std.testing.expect(!inv.scalar.value.bool);
}

test "logical empty and all-null inputs" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    var empty = try common.makeArray(bool, allocator, &.{});
    defer empty.release();
    var nulls = try common.makeArray(bool, allocator, &.{ null, null, null });
    defer nulls.release();
    var mixed = try common.makeArray(bool, allocator, &.{ true, false, null });
    defer mixed.release();
    inline for (binary_ops) |op| try common.expectInvoke(bool, &ctx, .scalar, op, &.{ .{ .array = empty }, .{ .array = empty } }, &.{});
    try common.expectInvoke(bool, &ctx, .scalar, "invert", &.{.{ .array = empty }}, &.{});
    try common.expectInvoke(bool, &ctx, .scalar, "and", &.{ .{ .array = nulls }, .{ .array = mixed } }, &.{ null, null, null });
    try common.expectInvoke(bool, &ctx, .scalar, "and_kleene", &.{ .{ .array = nulls }, .{ .array = mixed } }, &.{ null, false, null });
    try common.expectInvoke(bool, &ctx, .scalar, "or_kleene", &.{ .{ .array = nulls }, .{ .array = mixed } }, &.{ true, null, null });
    try common.expectInvoke(bool, &ctx, .scalar, "invert", &.{.{ .array = nulls }}, &.{ null, null, null });
}

test "logical errors: non-bool operands, arity, options, lengths and malformed scalars" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    var b = try common.makeArray(bool, allocator, &.{ true, false });
    defer b.release();
    var i = try common.makeArray(i8, allocator, &.{ 1, 0 });
    defer i.release();
    try std.testing.expectError(error.NoMatchingKernel, ctx.invokeScalar("and", &.{ .{ .array = i }, .{ .array = i } }, none));
    try std.testing.expectError(error.NoMatchingKernel, ctx.invokeScalar("and_kleene", &.{ .{ .array = b }, .{ .array = i } }, none));
    try std.testing.expectError(error.InvalidArity, ctx.invokeScalar("invert", &.{ .{ .array = b }, .{ .array = b } }, none));
    try std.testing.expectError(error.InvalidOptions, ctx.invokeScalar("or", &.{ .{ .array = b }, .{ .array = b } }, .{ .filter = .{} }));
    var three = try common.makeArray(bool, allocator, &.{ true, false, true });
    defer three.release();
    try std.testing.expectError(error.InvalidInput, ctx.invokeScalar("or_kleene", &.{ .{ .array = b }, .{ .array = three } }, none));
    const bad = common.Scalar.init(.{ .bool = {} }, .{ .u8 = 1 });
    try std.testing.expectError(error.InvalidInput, ctx.invokeScalar("and_kleene", &.{ .{ .array = b }, .{ .scalar = bad } }, none));
    // A value bitmap shorter than the array fails validateLayout instead of being read.
    const bufs = [_]common.SharedBuffer{ .empty, .empty };
    var malformed = try ArrayRef.fromBorrowed(allocator, .{ .data_type = .{ .bool = {} }, .length = 2, .null_count = 0, .buffers = &bufs });
    defer malformed.release();
    try std.testing.expectError(error.InvalidInput, ctx.invokeScalar("xor", &.{ .{ .array = malformed }, .{ .array = b } }, none));
}
