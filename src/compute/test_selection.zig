//! coalesce, if_else, case_when, choose, drop_null, fill_null_forward and fill_null_backward.
//! Expected values are pyarrow 26 results.
const std = @import("std");
const common = @import("test_common.zig");

const ArrayRef = common.ArrayRef;
const DataType = common.DataType;
const Datum = common.Datum;
const ExecContext = common.ExecContext;
const Scalar = common.Scalar;
const none = common.none;

fn expectType(ctx: *ExecContext, name: []const u8, args: []const Datum, dt: DataType) !void {
    var out = try ctx.invokeScalar(name, args, none);
    defer out.release();
    if (!out.dataType().eql(dt)) {
        std.debug.print("{s}: type {any}, expected {any}\n", .{ name, out.dataType(), dt });
        return error.TestExpectedEqual;
    }
    try std.testing.expect((try ctx.registry.resolveResultType(name, .scalar, args, none)).eql(dt));
}

/// A struct array of bool fields `a`, `b` (the case_when condition).
fn conditions(allocator: std.mem.Allocator, a: []const ?bool, b: []const ?bool) !ArrayRef {
    var fa = try common.makeArray(bool, allocator, a);
    const fb = common.makeArray(bool, allocator, b) catch |err| {
        fa.release();
        return err;
    };
    return common.dense.structArray(allocator, &.{ "a", "b" }, &.{ fa, fb }, a.len, null);
}

test "coalesce, if_else, case_when and choose hand-written values (pyarrow 26 results)" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);

    var cond = try common.makeArray(bool, allocator, &.{ true, false, null, true });
    defer cond.release();
    var x = try common.makeArray(i64, allocator, &.{ 1, 2, 3, null });
    defer x.release();
    var y = try common.makeArray(i64, allocator, &.{ 10, 20, 30, 40 });
    defer y.release();
    try common.expectInvoke(i64, &ctx, .scalar, "if_else", &.{ .{ .array = cond }, .{ .array = x }, .{ .array = y } }, &.{ 1, 20, null, null });
    try common.expectInvoke(i64, &ctx, .scalar, "if_else", &.{ .{ .scalar = common.scalarOf(bool, true) }, .{ .array = x }, .{ .array = y } }, &.{ 1, 2, 3, null });
    try common.expectInvoke(i64, &ctx, .scalar, "if_else", &.{ .{ .scalar = common.scalarOf(bool, null) }, .{ .array = x }, .{ .array = y } }, &.{ null, null, null, null });
    try common.expectInvoke(i64, &ctx, .scalar, "if_else", &.{ .{ .array = cond }, .{ .scalar = common.scalarOf(i64, 7) }, .{ .array = y } }, &.{ 7, 20, null, 7 });
    try common.expectInvoke(i64, &ctx, .scalar, "if_else", &.{ .{ .scalar = common.scalarOf(bool, false) }, .{ .scalar = common.scalarOf(i64, 7) }, .{ .scalar = common.scalarOf(i64, 8) } }, &.{8});

    var cw = try conditions(allocator, &.{ true, false, null, false }, &.{ false, true, true, null });
    defer cw.release();
    var e = try common.makeArray(i64, allocator, &.{ 100, 200, 300, 400 });
    defer e.release();
    try common.expectInvoke(i64, &ctx, .scalar, "case_when", &.{ .{ .array = cw }, .{ .array = x }, .{ .array = y } }, &.{ 1, 20, 30, null });
    try common.expectInvoke(i64, &ctx, .scalar, "case_when", &.{ .{ .array = cw }, .{ .array = x }, .{ .array = y }, .{ .array = e } }, &.{ 1, 20, 30, 400 });
    var y8 = try common.makeArray(i8, allocator, &.{ 10, 20, 30, 40 });
    defer y8.release();
    try common.expectInvoke(i64, &ctx, .scalar, "case_when", &.{ .{ .array = cw }, .{ .scalar = common.scalarOf(i64, 1) }, .{ .array = y8 } }, &.{ 1, 20, 30, null });

    var idx = try common.makeArray(i8, allocator, &.{ 0, 1, null, 2, 1 });
    defer idx.release();
    var c0 = try common.makeArray(i64, allocator, &.{ 1, 2, 3, 4, 5 });
    defer c0.release();
    var c1 = try common.makeArray(i64, allocator, &.{ 10, 20, 30, 40, null });
    defer c1.release();
    try common.expectInvoke(i64, &ctx, .scalar, "choose", &.{ .{ .array = idx }, .{ .array = c0 }, .{ .array = c1 }, .{ .scalar = common.scalarOf(i64, 100) } }, &.{ 1, 20, null, 100, null });

    var k0 = try common.makeArray(i64, allocator, &.{ 1, null, null });
    defer k0.release();
    var k1 = try common.makeArray(i64, allocator, &.{ null, 2, null });
    defer k1.release();
    try common.expectInvoke(i64, &ctx, .scalar, "coalesce", &.{ .{ .array = k0 }, .{ .array = k1 }, .{ .scalar = common.scalarOf(i64, 9) } }, &.{ 1, 2, 9 });
    try common.expectInvoke(i64, &ctx, .scalar, "coalesce", &.{ .{ .array = k0 }, .{ .scalar = common.scalarOf(i64, null) } }, &.{ 1, null, null });
    try common.expectInvoke(i64, &ctx, .scalar, "coalesce", &.{.{ .array = k1 }}, &.{ null, 2, null });

    // Strings and nested values go through the take kernels.
    var s1 = try common.makeStrings(.string, allocator, &.{ "a", null });
    defer s1.release();
    var s2 = try common.makeStrings(.large_string, allocator, &.{ "b", "c" });
    defer s2.release();
    var two = try common.makeArray(bool, allocator, &.{ true, false });
    defer two.release();
    try common.expectInvokeStrings(&ctx, "if_else", &.{ .{ .array = two }, .{ .array = s1 }, .{ .array = s2 } }, none, &.{ "a", "c" });
    try common.expectInvokeStrings(&ctx, "coalesce", &.{ .{ .array = s1 }, .{ .array = s2 } }, none, &.{ "a", "c" });
    var bin = try common.makeStrings(.binary, allocator, &.{ null, "x" });
    defer bin.release();
    const ys = Scalar.init(.{ .string = {} }, .{ .string = "y" });
    try common.expectInvokeStrings(&ctx, "coalesce", &.{ .{ .array = bin }, .{ .scalar = ys } }, none, &.{ "y", "x" });
}

test "selection common types: numbers, strings, temporal, decimals and the null type (pyarrow 26 results)" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    var two = try common.makeArray(bool, allocator, &.{ true, false });
    defer two.release();
    var i32s = try common.makeArray(i32, allocator, &.{ 10, 20 });
    defer i32s.release();
    try common.expectInvoke(i64, &ctx, .scalar, "if_else", &.{ .{ .array = two }, .{ .scalar = common.scalarOf(i64, 1) }, .{ .array = i32s } }, &.{ 1, 20 });
    var i8s = try common.makeArray(i8, allocator, &.{ 1, 2 });
    defer i8s.release();
    var f64s = try common.makeArray(f64, allocator, &.{ 1.5, 2.5 });
    defer f64s.release();
    try common.expectInvoke(f64, &ctx, .scalar, "if_else", &.{ .{ .array = two }, .{ .array = i8s }, .{ .array = f64s } }, &.{ 1.0, 2.5 });
    var n8 = try common.makeArray(i8, allocator, &.{ null, 2 });
    defer n8.release();
    var u64s = try common.makeArray(u64, allocator, &.{ 1, 3 });
    defer u64s.release();
    try common.expectInvoke(i64, &ctx, .scalar, "coalesce", &.{ .{ .array = n8 }, .{ .array = u64s } }, &.{ 1, 2 });

    const d52 = DataType{ .decimal128 = .{ .precision = 5, .scale = 2 } };
    const d61 = DataType{ .decimal128 = .{ .precision = 6, .scale = 1 } };
    var da = try common.makeTyped(i128, d52, allocator, &.{ 100, 200 });
    defer da.release();
    var db = try common.makeTyped(i128, d61, allocator, &.{ 10, 20 });
    defer db.release();
    try expectType(&ctx, "if_else", &.{ .{ .array = two }, .{ .array = da }, .{ .array = db } }, .{ .decimal128 = .{ .precision = 7, .scale = 2 } });
    const ts_s = DataType{ .timestamp = .{ .unit = .second, .timezone = null } };
    const ts_ms = DataType{ .timestamp = .{ .unit = .millisecond, .timezone = null } };
    var ta = try common.makeTyped(i64, ts_s, allocator, &.{ 1, 2 });
    defer ta.release();
    var tb = try common.makeTyped(i64, ts_ms, allocator, &.{ 3, 4 });
    defer tb.release();
    try expectType(&ctx, "if_else", &.{ .{ .array = two }, .{ .array = ta }, .{ .array = tb } }, ts_ms);
    var out = try ctx.invokeScalar("if_else", &.{ .{ .array = two }, .{ .array = ta }, .{ .array = tb } }, none);
    defer out.release();
    const vals = try (common.array_mod.PrimitiveArray(i64){ .data = out.array.data() }).values();
    try std.testing.expectEqualSlices(i64, &.{ 1000, 4 }, vals);
    var dd = try common.makeTyped(i32, .{ .date32 = {} }, allocator, &.{ null, 2 });
    defer dd.release();
    var de = try common.makeTyped(i64, .{ .date64 = {} }, allocator, &.{ 1, 3 });
    defer de.release();
    try expectType(&ctx, "coalesce", &.{ .{ .array = dd }, .{ .array = de } }, .{ .date64 = {} });
    var s1 = try common.makeStrings(.string, allocator, &.{ "a", null });
    defer s1.release();
    var s2 = try common.makeStrings(.large_string, allocator, &.{ "b", "c" });
    defer s2.release();
    try expectType(&ctx, "coalesce", &.{ .{ .array = s1 }, .{ .array = s2 } }, .{ .large_string = {} });
    var nulls = try common.datumBuildNullLike(.{ .null = {} }, 2);
    defer nulls.release();
    try common.expectInvoke(i32, &ctx, .scalar, "coalesce", &.{ nulls, .{ .array = i32s } }, &.{ 10, 20 });
    // A list value goes through the nested take.
    const item = DataType{ .int64 = {} };
    const list_type = DataType{ .list = .{ .value_field = .{ .name = "item", .data_type = &item } } };
    var lists = try common.datumBuildNullLike(list_type, 2);
    defer lists.release();
    var lo = try ctx.invokeScalar("if_else", &.{ .{ .array = two }, lists, lists }, none);
    defer lo.release();
    try lo.array.data().validateFull();
    try std.testing.expectEqual(@as(usize, 2), lo.array.data().null_count.?);
    // No common type.
    try std.testing.expectError(error.NoMatchingKernel, ctx.invokeScalar("if_else", &.{ .{ .array = two }, .{ .array = s1 }, .{ .array = i32s } }, none));
    try std.testing.expectError(error.NoMatchingKernel, ctx.invokeScalar("coalesce", &.{ .{ .array = ta }, .{ .array = da } }, none));
}

test "selection oracle matches a per-row loop on random sliced inputs with nulls, chunked and scalar operands" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    var prng = std.Random.DefaultPrng.init(0x5E1E);
    const rng = prng.random();
    const n = 300;
    var cond = try common.randomArray(bool, allocator, rng, n, 3, 20, .{});
    defer cond.release();
    var a = try common.randomArray(i32, allocator, rng, n, 5, 30, .{});
    defer a.release();
    var b = try common.randomArray(i32, allocator, rng, n, 11, 30, .{});
    defer b.release();
    var c = try common.randomArray(i32, allocator, rng, n, 1, 30, .{});
    defer c.release();
    const cs = try common.collect(bool, allocator, .{ .array = cond });
    defer allocator.free(cs);
    const as = try common.collect(i32, allocator, .{ .array = a });
    defer allocator.free(as);
    const bs = try common.collect(i32, allocator, .{ .array = b });
    defer allocator.free(bs);
    const ccs = try common.collect(i32, allocator, .{ .array = c });
    defer allocator.free(ccs);
    const expected = try allocator.alloc(?i32, n);
    defer allocator.free(expected);
    for (expected, cs, as, bs) |*e, k, x, y| e.* = if (k) |t| (if (t) x else y) else null;
    try common.expectInvoke(i32, &ctx, .scalar, "if_else", &.{ .{ .array = cond }, .{ .array = a }, .{ .array = b } }, expected);
    var cc = try common.chunkedAt(allocator, cond, &.{ 7, 150 });
    defer cc.release();
    var ca = try common.chunkedAt(allocator, a, &.{100});
    defer ca.release();
    try common.expectInvoke(i32, &ctx, .scalar, "if_else", &.{ .{ .chunked = cc }, .{ .chunked = ca }, .{ .array = b } }, expected);
    for (expected, as, bs, ccs) |*e, x, y, z| e.* = x orelse y orelse z;
    try common.expectInvoke(i32, &ctx, .scalar, "coalesce", &.{ .{ .array = a }, .{ .chunked = ca }, .{ .array = b }, .{ .array = c } }, expected);
    // choose with indices 0..2 (some null) over the three arrays.
    var picks: [n]?u8 = undefined;
    for (&picks, 0..) |*p, i| p.* = if (i % 13 == 5) null else @intCast(i % 3);
    var pick = try common.makeArray(u8, allocator, &picks);
    defer pick.release();
    for (expected, picks, as, bs, ccs) |*e, p, x, y, z| e.* = if (p) |k| switch (k) {
        0 => x,
        1 => y,
        else => z,
    } else null;
    try common.expectInvoke(i32, &ctx, .scalar, "choose", &.{ .{ .array = pick }, .{ .array = a }, .{ .array = b }, .{ .array = c } }, expected);
    // case_when over two random conditions with an else value.
    var cond2 = try common.randomArray(bool, allocator, rng, n, 0, 20, .{});
    defer cond2.release();
    const c2s = try common.collect(bool, allocator, .{ .array = cond2 });
    defer allocator.free(c2s);
    var cond1 = try common.randomArray(bool, allocator, rng, n, 0, 20, .{});
    defer cond1.release();
    const c1s = try common.collect(bool, allocator, .{ .array = cond1 });
    defer allocator.free(c1s);
    var st = try common.dense.structArray(allocator, &.{ "x", "y" }, &.{ cond1.retain(), cond2.retain() }, n, null);
    defer st.release();
    var st_slice = try st.slice(10, n - 10);
    defer st_slice.release();
    for (expected, c1s, c2s, as, bs, ccs) |*e, k1, k2, x, y, z| e.* = if (k1 orelse false) x else if (k2 orelse false) y else z;
    try common.expectInvoke(i32, &ctx, .scalar, "case_when", &.{ .{ .array = st }, .{ .array = a }, .{ .array = b }, .{ .array = c } }, expected);
    var a10 = try a.slice(10, n - 10);
    defer a10.release();
    var b10 = try b.slice(10, n - 10);
    defer b10.release();
    var c10 = try c.slice(10, n - 10);
    defer c10.release();
    try common.expectInvoke(i32, &ctx, .scalar, "case_when", &.{ .{ .array = st_slice }, .{ .array = a10 }, .{ .array = b10 }, .{ .array = c10 } }, expected[10..]);
}

test "drop_null, fill_null_forward and fill_null_backward (pyarrow 26 results), chunked across boundaries" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    var a = try common.makeArray(i32, allocator, &.{ 3, 1, null, 3, 2, null, 1 });
    defer a.release();
    const da = [_]Datum{.{ .array = a }};
    try common.expectInvoke(i32, &ctx, .vector, "drop_null", &da, &.{ 3, 1, 3, 2, 1 });
    try common.expectInvoke(i32, &ctx, .vector, "fill_null_forward", &da, &.{ 3, 1, 1, 3, 2, 2, 1 });
    try common.expectInvoke(i32, &ctx, .vector, "fill_null_backward", &da, &.{ 3, 1, 3, 3, 2, 1, 1 });
    var lead = try common.makeArray(i64, allocator, &.{ null, 1, null });
    defer lead.release();
    try common.expectInvoke(i64, &ctx, .vector, "fill_null_forward", &.{.{ .array = lead }}, &.{ null, 1, 1 });
    try common.expectInvoke(i64, &ctx, .vector, "fill_null_backward", &.{.{ .array = lead }}, &.{ 1, 1, null });
    var f = try common.makeArray(i64, allocator, &.{ 1, null, null, 2, null });
    defer f.release();
    var fc = try common.chunkedAt(allocator, f, &.{ 2, 4 });
    defer fc.release();
    try common.expectInvoke(i64, &ctx, .vector, "fill_null_forward", &.{.{ .chunked = fc }}, &.{ 1, 1, 1, 2, 2 });
    var g = try common.makeArray(i64, allocator, &.{ null, 1, null, null, 2 });
    defer g.release();
    var gc = try common.chunkedAt(allocator, g, &.{ 2, 3 });
    defer gc.release();
    try common.expectInvoke(i64, &ctx, .vector, "fill_null_backward", &.{.{ .chunked = gc }}, &.{ 1, 1, 2, 2, 2 });
    var dropped = try ctx.invokeVector("drop_null", &.{.{ .chunked = fc }}, none);
    defer dropped.release();
    try std.testing.expectEqual(@as(usize, 3), dropped.chunked.numChunks());
    try common.expectInvoke(i64, &ctx, .vector, "drop_null", &.{.{ .chunked = fc }}, &.{ 1, 2 });
    var s = try common.makeStrings(.string, allocator, &.{ "a", null, "b", null });
    defer s.release();
    var sf = try ctx.invokeVector("fill_null_forward", &.{.{ .array = s }}, none);
    defer sf.release();
    const srows = try common.collectStrings(allocator, sf);
    defer allocator.free(srows);
    try common.expectStrings(&.{ "a", "a", "b", "b" }, srows);
    // No null: the input comes back; a scalar comes back as is; the null type drops to empty.
    var full = try common.makeArray(i32, allocator, &.{ 1, 2 });
    defer full.release();
    var same = try ctx.invokeVector("fill_null_backward", &.{.{ .array = full }}, none);
    defer same.release();
    try std.testing.expect(same.array.node == full.node);
    try common.expectInvoke(i32, &ctx, .vector, "fill_null_forward", &.{.{ .scalar = common.scalarOf(i32, null) }}, &.{null});
    // Fuzz finding (fuzz/corpus/array-layout/oob_offset_layout.bin): every row of a uint8 array
    // null left no index, and the empty take had no builder for uint8 (UnsupportedType).
    var u8_nulls = try common.makeArray(u8, allocator, &.{ null, null });
    defer u8_nulls.release();
    var u8_dropped = try ctx.invokeVector("drop_null", &.{.{ .array = u8_nulls }}, none);
    defer u8_dropped.release();
    try u8_dropped.array.data().validateFull();
    try std.testing.expect(u8_dropped.array.data().data_type == .uint8 and u8_dropped.array.data().length == 0);
    var nulls = try common.datumBuildNullLike(.{ .null = {} }, 3);
    defer nulls.release();
    var nd = try ctx.invokeVector("drop_null", &.{nulls}, none);
    defer nd.release();
    try std.testing.expectEqual(@as(usize, 0), nd.array.data().length);
}

test "selection errors: case_when null struct rows, choose index out of range, scalars, arity and malformed input" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    var x = try common.makeArray(i64, allocator, &.{ 1, 2 });
    defer x.release();
    // A struct row that is null is rejected (Arrow: top-level nulls are not allowed).
    const fa = try common.makeArray(bool, allocator, &.{ true, false });
    const fb = try common.makeArray(bool, allocator, &.{ false, true });
    var validity = try common.dense.bitsBuffer(allocator, 2);
    validity.data[0] = 0b10;
    var masked = try common.dense.structArray(allocator, &.{ "a", "b" }, &.{ fa, fb }, 2, validity);
    defer masked.release();
    try std.testing.expectError(error.InvalidInput, ctx.invokeScalar("case_when", &.{ .{ .array = masked }, .{ .array = x }, .{ .array = x } }, none));
    // The number of values must equal the fields (or one more for the else value).
    try std.testing.expectError(error.NoMatchingKernel, ctx.invokeScalar("case_when", &.{ .{ .array = masked }, .{ .array = x }, .{ .array = x }, .{ .array = x }, .{ .array = x } }, none));
    var i8s = try common.makeArray(i8, allocator, &.{ 0, 3 });
    defer i8s.release();
    try std.testing.expectError(error.InvalidInput, ctx.invokeScalar("choose", &.{ .{ .array = i8s }, .{ .array = x }, .{ .array = x } }, none));
    try std.testing.expectError(error.InvalidInput, ctx.invokeScalar("choose", &.{ .{ .scalar = common.scalarOf(i8, -1) }, .{ .array = x } }, none));
    // An out-of-range index in a null row is not checked.
    const picks = [_]i8{ 0, 9 };
    var masked_picks = try common.borrowedArray(i8, allocator, &picks, &.{0b01});
    defer masked_picks.release();
    try common.expectInvoke(i64, &ctx, .scalar, "choose", &.{ .{ .array = masked_picks }, .{ .array = x } }, &.{ 1, null });
    var f = try common.makeArray(f64, allocator, &.{ 0, 1 });
    defer f.release();
    try std.testing.expectError(error.NoMatchingKernel, ctx.invokeScalar("choose", &.{ .{ .array = f }, .{ .array = x } }, none));
    try std.testing.expectError(error.NoMatchingKernel, ctx.invokeScalar("if_else", &.{ .{ .array = x }, .{ .array = x }, .{ .array = x } }, none));
    try std.testing.expectError(error.NoMatchingKernel, ctx.invokeVector("drop_null", &.{.{ .scalar = common.scalarOf(i32, 1) }}, none));
    try std.testing.expectError(error.InvalidArity, ctx.invokeScalar("if_else", &.{ .{ .array = fa }, .{ .array = x } }, none));
    try std.testing.expectError(error.InvalidOptions, ctx.invokeScalar("coalesce", &.{.{ .array = x }}, .{ .count = .{} }));
    var three = try common.makeArray(i64, allocator, &.{ 1, 2, 3 });
    defer three.release();
    try std.testing.expectError(error.InvalidInput, ctx.invokeScalar("coalesce", &.{ .{ .array = x }, .{ .array = three } }, none));
    const short = [_]i64{1};
    const bufs = [_]common.SharedBuffer{ .empty, .fromSlice(std.mem.sliceAsBytes(&short)) };
    var malformed = try ArrayRef.fromBorrowed(allocator, .{ .data_type = .{ .int64 = {} }, .length = 2, .null_count = 0, .buffers = &bufs });
    defer malformed.release();
    try std.testing.expectError(error.InvalidInput, ctx.invokeScalar("coalesce", &.{ .{ .array = malformed }, .{ .array = x } }, none));
    try std.testing.expectError(error.InvalidInput, ctx.invokeVector("drop_null", &.{.{ .array = malformed }}, none));
    try std.testing.expectError(error.InvalidInput, ctx.invokeVector("fill_null_forward", &.{.{ .array = malformed }}, none));
}
