//! coalesce, if_else, case_when, choose, drop_null, fill_null_forward, fill_null_backward,
//! replace_with_mask, scatter and inverse_permutation.
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

/// Invoke vector function `name` with `options`; every array part passes validateFull; compare
/// the rows (of storage `T`) with `expected`.
fn expectVector(comptime T: type, ctx: *ExecContext, name: []const u8, args: []const Datum, options: common.Options, expected: []const ?T) !void {
    const allocator = std.testing.allocator;
    var out = try ctx.invokeVector(name, args, options);
    defer out.release();
    switch (out) {
        .array => |a| try a.data().validateFull(),
        .chunked => |c| for (c.chunks()) |p| try p.data().validateFull(),
        else => return error.TestUnexpectedResult,
    }
    const rows = try common.collect(T, allocator, out);
    defer allocator.free(rows);
    try common.expectRows(T, expected, rows);
}

test "replace_with_mask, scatter and inverse_permutation hand-written values (pyarrow 26 results)" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);

    var v = try common.makeArray(i64, allocator, &.{ 1, 2, null, 4, 5 });
    defer v.release();
    var m = try common.makeArray(bool, allocator, &.{ true, false, null, true, false });
    defer m.release();
    var r2 = try common.makeArray(i64, allocator, &.{ 10, 20 });
    defer r2.release();
    var r3 = try common.makeArray(i64, allocator, &.{ 10, null, 30 });
    defer r3.release();
    var r1 = try common.makeArray(i64, allocator, &.{10});
    defer r1.release();
    const dv: Datum = .{ .array = v };
    const dm: Datum = .{ .array = m };
    try expectVector(i64, &ctx, "replace_with_mask", &.{ dv, dm, .{ .array = r2 } }, none, &.{ 10, 2, null, 20, 5 });
    try expectVector(i64, &ctx, "replace_with_mask", &.{ dv, dm, .{ .array = r3 } }, none, &.{ 10, 2, null, null, 5 });
    try std.testing.expectError(error.InvalidInput, ctx.invokeVector("replace_with_mask", &.{ dv, dm, .{ .array = r1 } }, none));
    const seven: Datum = .{ .scalar = common.scalarOf(i64, 7) };
    try expectVector(i64, &ctx, "replace_with_mask", &.{ dv, dm, seven }, none, &.{ 7, 2, null, 7, 5 });
    try expectVector(i64, &ctx, "replace_with_mask", &.{ dv, .{ .scalar = common.scalarOf(bool, true) }, seven }, none, &.{ 7, 7, 7, 7, 7 });
    try expectVector(i64, &ctx, "replace_with_mask", &.{ dv, .{ .scalar = common.scalarOf(bool, null) }, seven }, none, &.{ null, null, null, null, null });
    try expectVector(i64, &ctx, "replace_with_mask", &.{ dv, .{ .scalar = common.scalarOf(bool, false) }, .{ .array = r1 } }, none, &.{ 1, 2, null, 4, 5 });
    try expectVector(i64, &ctx, "replace_with_mask", &.{ dv, dm, .{ .scalar = common.scalarOf(i64, null) } }, none, &.{ null, 2, null, null, 5 });
    // Chunked values (a chunked result), sliced inputs, strings (the generic path) and bool.
    var vc = try common.chunkedAt(allocator, v, &.{2});
    defer vc.release();
    var chunked_out = try ctx.invokeVector("replace_with_mask", &.{ .{ .chunked = vc }, dm, .{ .array = r2 } }, none);
    defer chunked_out.release();
    try std.testing.expect(chunked_out == .chunked);
    try expectVector(i64, &ctx, "replace_with_mask", &.{ .{ .chunked = vc }, dm, .{ .array = r2 } }, none, &.{ 10, 2, null, 20, 5 });
    var vs = try v.slice(1, 3);
    defer vs.release();
    var ms = try m.slice(2, 3);
    defer ms.release();
    try expectVector(i64, &ctx, "replace_with_mask", &.{ .{ .array = vs }, .{ .array = ms }, .{ .array = r3 } }, none, &.{ null, 10, 4 });
    var s = try common.makeStrings(.string_view, allocator, &.{ "a", "b", null });
    defer s.release();
    var sr = try common.makeStrings(.string_view, allocator, &.{ "x", "a long replacement string" });
    defer sr.release();
    var sm = try common.makeArray(bool, allocator, &.{ true, true, false });
    defer sm.release();
    var so = try ctx.invokeVector("replace_with_mask", &.{ .{ .array = s }, .{ .array = sm }, .{ .array = sr } }, none);
    defer so.release();
    try so.array.data().validateFull();
    const srows = try common.collectStrings(allocator, so);
    defer allocator.free(srows);
    try common.expectStrings(&.{ "x", "a long replacement string", null }, srows);
    var bv = try common.makeArray(bool, allocator, &.{ true, null });
    defer bv.release();
    var br = try common.makeArray(bool, allocator, &.{ false, null });
    defer br.release();
    var bm = try common.makeArray(bool, allocator, &.{ true, true });
    defer bm.release();
    try expectVector(bool, &ctx, "replace_with_mask", &.{ .{ .array = bv }, .{ .array = bm }, .{ .array = br } }, none, &.{ false, null });
    // A mask of another length, a chunked mask or replacement, another replacement type.
    try std.testing.expectError(error.InvalidInput, ctx.invokeVector("replace_with_mask", &.{ dv, .{ .array = bm }, .{ .array = r2 } }, none));
    var mc = try common.chunkedAt(allocator, m, &.{1});
    defer mc.release();
    try std.testing.expectError(error.InvalidInput, ctx.invokeVector("replace_with_mask", &.{ dv, .{ .chunked = mc }, .{ .array = r2 } }, none));
    var r8 = try common.makeArray(i8, allocator, &.{ 1, 2 });
    defer r8.release();
    try std.testing.expectError(error.NoMatchingKernel, ctx.invokeVector("replace_with_mask", &.{ dv, dm, .{ .array = r8 } }, none));

    // scatter: a later row wins on a repeated index, a null index drops its row.
    var sv = try common.makeArray(i64, allocator, &.{ 10, 20, 30 });
    defer sv.release();
    const dsv: Datum = .{ .array = sv };
    for ([_][3]?i32{ .{ 2, 0, 1 }, .{ 2, 0, null }, .{ 2, 0, 2 } }, [_][3]?i64{ .{ 20, 30, 10 }, .{ 20, null, 10 }, .{ 20, null, 30 } }) |ix, want| {
        var idx = try common.makeArray(i32, allocator, &ix);
        defer idx.release();
        try expectVector(i64, &ctx, "scatter", &.{ dsv, .{ .array = idx } }, none, &want);
    }
    var perm = try common.makeArray(i8, allocator, &.{ 2, 0, 1 });
    defer perm.release();
    try expectVector(i64, &ctx, "scatter", &.{ dsv, .{ .array = perm } }, .{ .scatter = .{ .max_index = 5 } }, &.{ 20, 30, 10, null, null, null });
    try std.testing.expectError(error.InvalidInput, ctx.invokeVector("scatter", &.{ dsv, .{ .array = perm } }, .{ .scatter = .{ .max_index = 1 } }));
    var bad = try common.makeArray(i64, allocator, &.{ 5, 0, 1 });
    defer bad.release();
    try std.testing.expectError(error.InvalidInput, ctx.invokeVector("scatter", &.{ dsv, .{ .array = bad } }, none));
    var neg = try common.makeArray(i64, allocator, &.{ -1, 0, 1 });
    defer neg.release();
    try std.testing.expectError(error.InvalidInput, ctx.invokeVector("scatter", &.{ dsv, .{ .array = neg } }, none));
    var short = try common.makeArray(i64, allocator, &.{ 1, 0 });
    defer short.release();
    try std.testing.expectError(error.InvalidInput, ctx.invokeVector("scatter", &.{ dsv, .{ .array = short } }, none));
    var unsigned = try common.makeArray(u8, allocator, &.{ 1, 0, 2 });
    defer unsigned.release();
    try std.testing.expectError(error.NoMatchingKernel, ctx.invokeVector("scatter", &.{ dsv, .{ .array = unsigned } }, none));
    // A hostile max_index past max(length, 2^24) is refused before any allocation.
    try std.testing.expectError(error.InvalidOptions, ctx.invokeVector("scatter", &.{ dsv, .{ .array = perm } }, .{ .scatter = .{ .max_index = 1 << 40 } }));
    // Chunked values and indices with other boundaries; strings.
    var svc = try common.chunkedAt(allocator, sv, &.{1});
    defer svc.release();
    var pc = try common.chunkedAt(allocator, perm, &.{2});
    defer pc.release();
    try expectVector(i64, &ctx, "scatter", &.{ .{ .chunked = svc }, .{ .chunked = pc } }, none, &.{ 20, 30, 10 });
    var strs = try common.makeStrings(.string, allocator, &.{ "a", null, "c" });
    defer strs.release();
    var so2 = try ctx.invokeVector("scatter", &.{ .{ .array = strs }, .{ .array = perm } }, none);
    defer so2.release();
    const s2rows = try common.collectStrings(allocator, so2);
    defer allocator.free(s2rows);
    try common.expectStrings(&.{ null, "c", "a" }, s2rows);

    // inverse_permutation.
    var p3 = try common.makeArray(i64, allocator, &.{ 2, 0, 1 });
    defer p3.release();
    try expectVector(i64, &ctx, "inverse_permutation", &.{.{ .array = p3 }}, none, &.{ 1, 2, 0 });
    var pn = try common.makeArray(i64, allocator, &.{ 2, 0, null });
    defer pn.release();
    try expectVector(i64, &ctx, "inverse_permutation", &.{.{ .array = pn }}, none, &.{ 1, null, 0 });
    var pd = try common.makeArray(i64, allocator, &.{ 2, 0, 2 });
    defer pd.release();
    try expectVector(i64, &ctx, "inverse_permutation", &.{.{ .array = pd }}, none, &.{ 1, null, 2 });
    var p2 = try common.makeArray(i64, allocator, &.{ 1, 0 });
    defer p2.release();
    try expectVector(i64, &ctx, "inverse_permutation", &.{.{ .array = p2 }}, .{ .inverse_permutation = .{ .max_index = 4 } }, &.{ 1, 0, null, null, null });
    try std.testing.expectError(error.InvalidInput, ctx.invokeVector("inverse_permutation", &.{.{ .array = p2 }}, .{ .inverse_permutation = .{ .max_index = 0 } }));
    try expectVector(i8, &ctx, "inverse_permutation", &.{.{ .array = p2 }}, .{ .inverse_permutation = .{ .output_type = .{ .int8 = {} } } }, &.{ 1, 0 });
    try std.testing.expectError(error.InvalidOptions, ctx.invokeVector("inverse_permutation", &.{.{ .array = p2 }}, .{ .inverse_permutation = .{ .output_type = .{ .uint8 = {} } } }));
    try std.testing.expectError(error.InvalidInput, ctx.invokeVector("inverse_permutation", &.{.{ .array = bad }}, none));
    try std.testing.expectError(error.InvalidInput, ctx.invokeVector("inverse_permutation", &.{.{ .array = neg }}, none));
    const many = try allocator.alloc(?i64, 128);
    defer allocator.free(many);
    for (many, 0..) |*x, i| x.* = @intCast(i);
    var p128 = try common.makeArray(i64, allocator, many);
    defer p128.release();
    try std.testing.expectError(error.InvalidInput, ctx.invokeVector("inverse_permutation", &.{.{ .array = p128 }}, .{ .inverse_permutation = .{ .output_type = .{ .int8 = {} } } }));
    var p3c = try common.chunkedAt(allocator, p3, &.{1});
    defer p3c.release();
    var ic = try ctx.invokeVector("inverse_permutation", &.{.{ .chunked = p3c }}, none);
    defer ic.release();
    try std.testing.expect(ic == .chunked);
    try expectVector(i64, &ctx, "inverse_permutation", &.{.{ .chunked = p3c }}, none, &.{ 1, 2, 0 });
    var empty = try p3.slice(0, 0);
    defer empty.release();
    try expectVector(i64, &ctx, "inverse_permutation", &.{.{ .array = empty }}, none, &.{});
}

test "indices_nonzero on numbers, bool, floats and decimals, sliced, chunked and scalar (pyarrow 26 results)" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);

    var i = try common.makeArray(i64, allocator, &.{ 0, 1, null, -2, 0 });
    defer i.release();
    try expectVector(u64, &ctx, "indices_nonzero", &.{.{ .array = i }}, none, &.{ 1, 3 });
    var b = try common.makeArray(bool, allocator, &.{ true, false, null, true, true, false, false, false, false, true });
    defer b.release();
    try expectVector(u64, &ctx, "indices_nonzero", &.{.{ .array = b }}, none, &.{ 0, 3, 4, 9 });
    var bs = try b.slice(3, 6);
    defer bs.release();
    try expectVector(u64, &ctx, "indices_nonzero", &.{.{ .array = bs }}, none, &.{ 0, 1 });
    var f = try common.makeArray(f64, allocator, &.{ 0.0, -0.0, std.math.nan(f64), null, 1.5 });
    defer f.release();
    try expectVector(u64, &ctx, "indices_nonzero", &.{.{ .array = f }}, none, &.{ 2, 4 });
    var d = try common.makeTyped(i128, .{ .decimal128 = .{ .precision = 5, .scale = 2 } }, allocator, &.{ 0, 150, null });
    defer d.release();
    try expectVector(u64, &ctx, "indices_nonzero", &.{.{ .array = d }}, none, &.{1});
    var d32 = try common.makeTyped(i32, .{ .decimal32 = .{ .precision = 5, .scale = 2 } }, allocator, &.{ 7, 0 });
    defer d32.release();
    try expectVector(u64, &ctx, "indices_nonzero", &.{.{ .array = d32 }}, none, &.{0});
    var u = try common.makeArray(u8, allocator, &.{ 0, 3, 0, 1, 2 });
    defer u.release();
    var uc = try common.chunkedAt(allocator, u, &.{ 2, 3 });
    defer uc.release();
    try expectVector(u64, &ctx, "indices_nonzero", &.{.{ .chunked = uc }}, none, &.{ 1, 3, 4 });
    try expectVector(u64, &ctx, "indices_nonzero", &.{.{ .scalar = common.scalarOf(i32, 1) }}, none, &.{0});
    try expectVector(u64, &ctx, "indices_nonzero", &.{.{ .scalar = common.scalarOf(i32, null) }}, none, &.{});
    var s = try common.makeStrings(.string, allocator, &.{"a"});
    defer s.release();
    try std.testing.expectError(error.NoMatchingKernel, ctx.invokeVector("indices_nonzero", &.{.{ .array = s }}, none));
}

/// Invoke vector function `name` with `options`; the result rows (array or chunked, every chunk
/// validated) must equal `expected`, and the result must be chunked exactly when `chunked`.
fn expectSelected(comptime T: type, ctx: *ExecContext, name: []const u8, args: []const Datum, options: common.Options, chunked: bool, expected: []const ?T) !void {
    const allocator = std.testing.allocator;
    var out = try ctx.invokeVector(name, args, options);
    defer out.release();
    switch (out) {
        .array => |arr| try arr.data().validateFull(),
        .chunked => |c| for (c.chunks()) |p| try p.data().validateFull(),
        else => return error.TestUnexpectedResult,
    }
    try std.testing.expectEqual(chunked, out == .chunked);
    const rows = try common.collect(T, allocator, out);
    defer allocator.free(rows);
    try common.expectRows(T, expected, rows);
}

test "take, array_take, filter and array_filter on nulls, sliced, chunked and every index type (pyarrow 26 results)" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);

    var v = try common.makeArray(i64, allocator, &.{ 1, 2, 3 });
    defer v.release();
    var idx = try common.makeArray(i64, allocator, &.{ 2, null, 0 });
    defer idx.release();
    // pc.take(pa.array([1, 2, 3]), pa.array([2, None, 0])) -> [3, null, 1]
    try expectSelected(i64, &ctx, "take", &.{ .{ .array = v }, .{ .array = idx } }, none, false, &.{ 3, null, 1 });
    try expectSelected(i64, &ctx, "array_take", &.{ .{ .array = v }, .{ .array = idx } }, none, false, &.{ 3, null, 1 });
    try expectSelected(i64, &ctx, "take", &.{ .{ .array = v }, .{ .array = idx } }, .{ .take = .{ .boundscheck = false } }, false, &.{ 3, null, 1 });
    // Every integer index type.
    inline for (.{ i8, i16, i32, u8, u16, u32, u64 }) |I| {
        var ix = try common.makeArray(I, allocator, &.{ 1, 1, null });
        defer ix.release();
        try expectSelected(i64, &ctx, "take", &.{ .{ .array = v }, .{ .array = ix } }, none, false, &.{ 2, 2, null });
    }
    // Sliced values and sliced indices.
    var vs = try v.slice(1, 2);
    defer vs.release();
    var is = try idx.slice(1, 2);
    defer is.release();
    try expectSelected(i64, &ctx, "take", &.{ .{ .array = vs }, .{ .array = is } }, none, false, &.{ null, 2 });
    // Chunked values and / or chunked indices give a chunked result.
    var cv = try common.chunkedAt(allocator, v, &.{2});
    defer cv.release();
    var ci = try common.chunkedAt(allocator, idx, &.{1});
    defer ci.release();
    try expectSelected(i64, &ctx, "take", &.{ .{ .chunked = cv }, .{ .array = idx } }, none, true, &.{ 3, null, 1 });
    try expectSelected(i64, &ctx, "take", &.{ .{ .chunked = cv }, .{ .chunked = ci } }, none, true, &.{ 3, null, 1 });
    try expectSelected(i64, &ctx, "take", &.{ .{ .array = v }, .{ .chunked = ci } }, none, true, &.{ 3, null, 1 });
    // Strings go through the same take kernels.
    var sv = try common.makeStrings(.string_view, allocator, &.{ "a", null, "a string longer than twelve" });
    defer sv.release();
    var st = try ctx.invokeVector("take", &.{ .{ .array = sv }, .{ .array = idx } }, none);
    defer st.release();
    const srows = try common.collectStrings(allocator, st);
    defer allocator.free(srows);
    try common.expectStrings(&.{ "a string longer than twelve", null, "a" }, srows);

    // filter: a null predicate row is dropped, or null with emit_null (drop_nulls = false).
    var pred = try common.makeArray(bool, allocator, &.{ true, null, false });
    defer pred.release();
    try expectSelected(i64, &ctx, "filter", &.{ .{ .array = v }, .{ .array = pred } }, none, false, &.{1});
    try expectSelected(i64, &ctx, "array_filter", &.{ .{ .array = v }, .{ .array = pred } }, none, false, &.{1});
    try expectSelected(i64, &ctx, "filter", &.{ .{ .array = v }, .{ .array = pred } }, .{ .filter = .{ .drop_nulls = false } }, false, &.{ 1, null });
    var p2 = try common.makeArray(bool, allocator, &.{ true, false, true });
    defer p2.release();
    var cp = try common.chunkedAt(allocator, p2, &.{1});
    defer cp.release();
    try expectSelected(i64, &ctx, "filter", &.{ .{ .chunked = cv }, .{ .array = p2 } }, none, true, &.{ 1, 3 });
    try expectSelected(i64, &ctx, "filter", &.{ .{ .chunked = cv }, .{ .chunked = cp } }, none, true, &.{ 1, 3 });
    try expectSelected(i64, &ctx, "filter", &.{ .{ .array = v }, .{ .chunked = cp } }, none, true, &.{ 1, 3 });
    try expectSelected(i64, &ctx, "array_filter", &.{ .{ .chunked = cv }, .{ .array = p2 } }, none, true, &.{ 1, 3 });
    try expectSelected(i64, &ctx, "array_filter", &.{ .{ .array = v }, .{ .chunked = cp } }, none, true, &.{ 1, 3 });
    // Sliced values and predicate.
    var ps = try p2.slice(1, 2);
    defer ps.release();
    try expectSelected(i64, &ctx, "filter", &.{ .{ .array = vs }, .{ .array = ps } }, none, false, &.{3});
    // An empty input keeps its type (fuzz finding: date64 and extension arrays).
    var dates = try common.makeTyped(i64, .{ .date64 = {} }, allocator, &.{});
    defer dates.release();
    var no_pred = try p2.slice(0, 0);
    defer no_pred.release();
    var empty = try ctx.invokeVector("filter", &.{ .{ .array = dates }, .{ .array = no_pred } }, none);
    defer empty.release();
    try std.testing.expect(empty.array.data().data_type == .date64 and empty.array.data().length == 0);

    // Record batch and table values (take / filter only).
    const int64_type = DataType{ .int64 = {} };
    const fields = [_]common.datatype.Field{.{ .name = "a", .data_type = &int64_type }};
    var batch = try @import("../record_batch.zig").RecordBatch.initBorrowed(allocator, .{ .fields = &fields }, &.{v});
    var rb = Datum.fromRecordBatch(allocator, batch) catch |err| {
        batch.deinit();
        return err;
    };
    defer rb.release();
    var table = try @import("../table.zig").Table.init(allocator, try @import("../schema.zig").SchemaRef.fromBorrowed(allocator, .{ .fields = &fields }), &.{cv});
    var tb = Datum.fromTable(allocator, table) catch |err| {
        table.deinit();
        return err;
    };
    defer tb.release();
    inline for (.{ "take", "filter" }) |name| {
        const is_take = comptime std.mem.eql(u8, name, "take");
        const second: Datum = if (is_take) .{ .chunked = ci } else .{ .array = p2 };
        const want: []const ?i64 = if (is_take) &.{ 3, null, 1 } else &.{ 1, 3 };
        for ([_]Datum{ rb, tb }) |values| {
            var out = try ctx.invokeVector(name, &.{ values, second }, none);
            defer out.release();
            try std.testing.expectEqual(std.meta.activeTag(values), std.meta.activeTag(out));
            var col = try common.core.datumStructField(out, 0);
            defer col.release();
            const rows = try common.collect(i64, allocator, col);
            defer allocator.free(rows);
            try common.expectRows(i64, want, rows);
        }
    }
    try std.testing.expectError(error.NotImplemented, ctx.invokeVector("array_take", &.{ rb, .{ .array = idx } }, none));
    try std.testing.expectError(error.NotImplemented, ctx.invokeVector("array_filter", &.{ tb, .{ .array = p2 } }, none));
    try std.testing.expect((try registry.resolveResultType("take", .vector, &.{ .{ .array = v }, .{ .array = idx } }, none)) == .int64);
}

test "take and filter errors: index bounds, index and predicate types, lengths, datum kinds, options" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);

    var v = try common.makeArray(i64, allocator, &.{ 1, 2, 3 });
    defer v.release();
    // Arrow: IndexError "Index -1 out of bounds" / "Index 3 out of bounds", also without boundscheck.
    inline for (.{ @as(i64, -1), @as(i64, 3) }) |bad| {
        var bx = try common.makeArray(i64, allocator, &.{ 0, bad });
        defer bx.release();
        try std.testing.expectError(error.InvalidInput, ctx.invokeVector("take", &.{ .{ .array = v }, .{ .array = bx } }, none));
        try std.testing.expectError(error.InvalidInput, ctx.invokeVector("array_take", &.{ .{ .array = v }, .{ .array = bx } }, .{ .take = .{ .boundscheck = false } }));
    }
    var huge = try common.makeArray(u64, allocator, &.{std.math.maxInt(u64)});
    defer huge.release();
    try std.testing.expectError(error.InvalidInput, ctx.invokeVector("take", &.{ .{ .array = v }, .{ .array = huge } }, none));
    // Float indices, scalar values or indices, chunked arguments to array_take.
    var fx = try common.makeArray(f64, allocator, &.{1.0});
    defer fx.release();
    try std.testing.expectError(error.NoMatchingKernel, ctx.invokeVector("take", &.{ .{ .array = v }, .{ .array = fx } }, none));
    var ix = try common.makeArray(i32, allocator, &.{0});
    defer ix.release();
    try std.testing.expectError(error.NoMatchingKernel, ctx.invokeVector("take", &.{ .{ .array = v }, .{ .scalar = common.scalarOf(i32, 0) } }, none));
    try std.testing.expectError(error.NoMatchingKernel, ctx.invokeVector("take", &.{ .{ .scalar = common.scalarOf(i64, 1) }, .{ .array = ix } }, none));
    var cv = try common.chunkedAt(allocator, v, &.{1});
    defer cv.release();
    try std.testing.expectError(error.NoMatchingKernel, ctx.invokeVector("array_take", &.{ .{ .chunked = cv }, .{ .array = ix } }, none));
    try std.testing.expectError(error.InvalidOptions, ctx.invokeVector("take", &.{ .{ .array = v }, .{ .array = ix } }, .{ .filter = .{} }));
    // Arrow: "Filter argument must be boolean type", "Filter should be array-like", unequal lengths.
    var pred = try common.makeArray(bool, allocator, &.{ true, false });
    defer pred.release();
    try std.testing.expectError(error.NoMatchingKernel, ctx.invokeVector("filter", &.{ .{ .array = v }, .{ .array = v } }, none));
    try std.testing.expectError(error.NoMatchingKernel, ctx.invokeVector("filter", &.{ .{ .array = v }, .{ .scalar = common.scalarOf(bool, true) } }, none));
    try std.testing.expectError(error.InvalidInput, ctx.invokeVector("filter", &.{ .{ .array = v }, .{ .array = pred } }, none));
    try std.testing.expectError(error.InvalidInput, ctx.invokeVector("array_filter", &.{ .{ .chunked = cv }, .{ .array = pred } }, none));
    try std.testing.expectError(error.InvalidOptions, ctx.invokeVector("filter", &.{ .{ .array = pred }, .{ .array = pred } }, .{ .take = .{} }));
    try std.testing.expectError(error.InvalidArity, ctx.invokeVector("take", &.{.{ .array = v }}, none));
}
