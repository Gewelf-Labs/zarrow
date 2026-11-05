//! unique, value_counts and dictionary_encode. Expected values are pyarrow 26 results.
const std = @import("std");
const common = @import("test_common.zig");

const ArrayRef = common.ArrayRef;
const DataType = common.DataType;
const Datum = common.Datum;
const ExecContext = common.ExecContext;
const none = common.none;

const nan = std.math.nan(f64);

/// Invoke vector function `name` and return the array result after validateFull.
fn vector(ctx: *ExecContext, name: []const u8, arg: Datum, options: common.Options) !ArrayRef {
    var out = try ctx.invokeVector(name, &.{arg}, options);
    errdefer out.release();
    const arr = out.asArray() orelse return error.TestUnexpectedResult;
    try arr.data().validateFull();
    return arr;
}

/// The values and counts of a value_counts result.
fn expectCounts(comptime T: type, out: ArrayRef, values: []const ?T, counts: []const i64) !void {
    const allocator = std.testing.allocator;
    const st = out.data().data_type.struct_;
    try std.testing.expectEqualStrings("values", st.fields[0].name);
    try std.testing.expectEqualStrings("counts", st.fields[1].name);
    const v = try common.collect(T, allocator, .{ .array = out.data().children[0] });
    defer allocator.free(v);
    try common.expectRows(T, values, v);
    const c = try common.collect(i64, allocator, .{ .array = out.data().children[1] });
    defer allocator.free(c);
    try std.testing.expectEqual(counts.len, c.len);
    for (counts, c) |e, g| try std.testing.expectEqual(e, g.?);
}

test "unique, value_counts and dictionary_encode hand-written values (pyarrow 26 results)" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);

    var a = try common.makeArray(i32, allocator, &.{ 3, 1, null, 3, 2, null, 1 });
    defer a.release();
    var u = try vector(&ctx, "unique", .{ .array = a }, none);
    defer u.release();
    try expectArray(i32, u, &.{ 3, 1, null, 2 });
    var vc = try vector(&ctx, "value_counts", .{ .array = a }, none);
    defer vc.release();
    try expectCounts(i32, vc, &.{ 3, 1, null, 2 }, &.{ 2, 2, 2, 1 });
    // The struct type lives with the array: a slice keeps it after the array is released.
    var vc_slice = try vc.slice(1, 2);
    defer vc_slice.release();
    try vc_slice.data().validateFull();
    try std.testing.expect(vc_slice.data().data_type.struct_.fields[0].data_type.* == .int32);

    // Floats: NaN is one value, -0.0 and 0.0 are two.
    var f = try common.makeArray(f64, allocator, &.{ nan, 0.0, -0.0, -nan, null });
    defer f.release();
    var fu = try vector(&ctx, "value_counts", .{ .array = f }, none);
    defer fu.release();
    try expectCounts(f64, fu, &.{ nan, 0.0, -0.0, null }, &.{ 2, 1, 1, 1 });

    // Chunked input: one result over all chunks.
    var c = try common.makeArray(i64, allocator, &.{ 1, 2, 2, null, 3 });
    defer c.release();
    var cc = try common.chunkedAt(allocator, c, &.{2});
    defer cc.release();
    var cu = try vector(&ctx, "unique", .{ .chunked = cc }, none);
    defer cu.release();
    try expectArray(i64, cu, &.{ 1, 2, null, 3 });
    var cvc = try vector(&ctx, "value_counts", .{ .chunked = cc }, none);
    defer cvc.release();
    try expectCounts(i64, cvc, &.{ 1, 2, null, 3 }, &.{ 1, 2, 1, 1 });

    // Strings, bool, decimals, fixed_size_binary and the null type.
    var s = try common.makeStrings(.string_view, allocator, &.{ "b", "a", null, "b", "a long value past twelve" });
    defer s.release();
    var su = try vector(&ctx, "unique", .{ .array = s }, none);
    defer su.release();
    const su_rows = try common.collectStrings(allocator, .{ .array = su });
    defer allocator.free(su_rows);
    try common.expectStrings(&.{ "b", "a", null, "a long value past twelve" }, su_rows);
    var b = try common.makeArray(bool, allocator, &.{ true, null, false, true });
    defer b.release();
    var bu = try vector(&ctx, "unique", .{ .array = b }, none);
    defer bu.release();
    try expectArray(bool, bu, &.{ true, null, false });
    const d52 = DataType{ .decimal128 = .{ .precision = 5, .scale = 2 } };
    var d = try common.makeTyped(i128, d52, allocator, &.{ 100, 200, 100 });
    defer d.release();
    var dvc = try vector(&ctx, "value_counts", .{ .array = d }, none);
    defer dvc.release();
    try std.testing.expect(dvc.data().data_type.struct_.fields[0].data_type.eql(d52));
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    try std.testing.expect((try registry.resolveResultTypeAlloc(arena.allocator(), "value_counts", .vector, &.{.{ .array = d }}, none)).struct_.fields[0].data_type.eql(d52));
    var nulls = try common.datumBuildNullLike(.{ .null = {} }, 2);
    defer nulls.release();
    var nu = try vector(&ctx, "unique", nulls, none);
    defer nu.release();
    try std.testing.expectEqual(@as(usize, 1), nu.data().length);
    var nvc = try vector(&ctx, "value_counts", nulls, none);
    defer nvc.release();
    const ncounts = try common.collect(i64, allocator, .{ .array = nvc.data().children[1] });
    defer allocator.free(ncounts);
    try common.expectRows(i64, &.{2}, ncounts);

    // dictionary_encode: mask (default) keeps nulls out of the dictionary; encode adds one.
    var e = try vector(&ctx, "dictionary_encode", .{ .array = a }, none);
    defer e.release();
    try std.testing.expect(e.data().data_type == .dictionary);
    try expectArray(i32, e.data().dictionary.?, &.{ 3, 1, 2 });
    try expectIndices(e, &.{ 0, 1, null, 0, 2, null, 1 });
    var enc = try vector(&ctx, "dictionary_encode", .{ .array = a }, .{ .dictionary_encode = .{ .null_encoding = .encode } });
    defer enc.release();
    try expectArray(i32, enc.data().dictionary.?, &.{ 3, 1, null, 2 });
    try expectIndices(enc, &.{ 0, 1, 2, 0, 3, 2, 1 });
    // A chunked input gives chunks sharing one dictionary.
    var ce = try ctx.invokeVector("dictionary_encode", &.{.{ .chunked = cc }}, none);
    defer ce.release();
    try std.testing.expectEqual(@as(usize, 2), ce.chunked.numChunks());
    try std.testing.expect(ce.chunked.chunks()[0].data().dictionary.?.node == ce.chunked.chunks()[1].data().dictionary.?.node);
    var bool_enc = try vector(&ctx, "dictionary_encode", .{ .array = b }, none);
    defer bool_enc.release();
    try expectArray(bool, bool_enc.data().dictionary.?, &.{ true, false });
}

/// The rows of `arr` (of `T`) equal `expected`.
fn expectArray(comptime T: type, arr: ArrayRef, expected: []const ?T) !void {
    const rows = try common.collect(T, std.testing.allocator, .{ .array = arr });
    defer std.testing.allocator.free(rows);
    try common.expectRows(T, expected, rows);
}

/// The int32 indices of a dictionary array equal `expected`.
fn expectIndices(dict: ArrayRef, expected: []const ?i32) !void {
    const d = dict.data();
    var idx = try ArrayRef.fromBorrowed(std.testing.allocator, .{ .data_type = .{ .int32 = {} }, .length = d.length, .offset = d.offset, .null_count = d.null_count, .buffers = d.buffers });
    defer idx.release();
    try expectArray(i32, idx, expected);
}

test "associative oracle on random sliced inputs with nulls: unique and counts equal a linear scan, dictionary_encode decodes back" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    var prng = std.Random.DefaultPrng.init(0xA550);
    const rng = prng.random();
    inline for (.{ i8, u16, i64, f32 }) |T| {
        var a = try common.randomArray(T, allocator, rng, 400, 9, 15, .{ .small = @typeInfo(T) == .int });
        defer a.release();
        const rows = try common.collect(T, allocator, .{ .array = a });
        defer allocator.free(rows);
        // Reference: first-seen distinct values (bit equality with one NaN) and their counts.
        var distinct: std.ArrayList(?T) = .empty;
        defer distinct.deinit(allocator);
        var counts: std.ArrayList(i64) = .empty;
        defer counts.deinit(allocator);
        for (rows) |r| {
            const k = for (distinct.items, 0..) |d, k| {
                if (same(T, d, r)) break k;
            } else null;
            if (k) |i| counts.items[i] += 1 else {
                try distinct.append(allocator, r);
                try counts.append(allocator, 1);
            }
        }
        var vc = try vector(&ctx, "value_counts", .{ .array = a }, none);
        defer vc.release();
        try expectCounts(T, vc, distinct.items, counts.items);
        var chunked = try common.chunkedAt(allocator, a, &.{ 1, 200, 399 });
        defer chunked.release();
        var cu = try vector(&ctx, "unique", .{ .chunked = chunked }, none);
        defer cu.release();
        const urows = try common.collect(T, allocator, .{ .array = cu });
        defer allocator.free(urows);
        try common.expectRows(T, distinct.items, urows);
        // Decoding the dictionary gives the input back.
        var enc = try ctx.invokeVector("dictionary_encode", &.{.{ .chunked = chunked }}, none);
        defer enc.release();
        for (enc.chunked.chunks()) |part| try part.data().validateFull();
        var back = try ctx.invokeScalar("cast", &.{enc}, .{ .cast = .{ .to_type = common.dense.dataTypeOf(T) } });
        defer back.release();
        const brows = try common.collect(T, allocator, back);
        defer allocator.free(brows);
        try common.expectRows(T, rows, brows);
    }
}

fn same(comptime T: type, a: ?T, b: ?T) bool {
    if (a == null or b == null) return a == null and b == null;
    if (@typeInfo(T) == .float) {
        if (std.math.isNan(a.?) and std.math.isNan(b.?)) return true;
        const U = @Int(.unsigned, @bitSizeOf(T));
        return @as(U, @bitCast(a.?)) == @as(U, @bitCast(b.?));
    }
    return a.? == b.?;
}

test "associative edge cases and errors: empty, scalars, unsupported types, options and malformed input" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    var empty = try common.makeArray(i32, allocator, &.{});
    defer empty.release();
    var eu = try vector(&ctx, "unique", .{ .array = empty }, none);
    defer eu.release();
    try std.testing.expectEqual(@as(usize, 0), eu.data().length);
    var evc = try vector(&ctx, "value_counts", .{ .array = empty }, none);
    defer evc.release();
    try std.testing.expectEqual(@as(usize, 0), evc.data().length);
    var su = try vector(&ctx, "unique", .{ .scalar = common.scalarOf(i32, 7) }, none);
    defer su.release();
    try expectArray(i32, su, &.{7});
    var all_null = try common.makeArray(i32, allocator, &.{ null, null });
    defer all_null.release();
    var an = try vector(&ctx, "dictionary_encode", .{ .array = all_null }, none);
    defer an.release();
    try std.testing.expectEqual(@as(usize, 0), an.data().dictionary.?.data().length);
    try std.testing.expectEqual(@as(usize, 2), an.data().null_count.?);

    const item = DataType{ .int32 = {} };
    var list = try common.datumBuildNullLike(.{ .list = .{ .value_field = .{ .name = "item", .data_type = &item } } }, 1);
    defer list.release();
    try std.testing.expectError(error.NoMatchingKernel, ctx.invokeVector("unique", &.{list}, none));
    var nulls = try common.datumBuildNullLike(.{ .null = {} }, 1);
    defer nulls.release();
    try std.testing.expectError(error.NoMatchingKernel, ctx.invokeVector("dictionary_encode", &.{nulls}, none));
    try std.testing.expectError(error.InvalidOptions, ctx.invokeVector("unique", &.{.{ .array = empty }}, .{ .count = .{} }));
    try std.testing.expectError(error.InvalidArity, ctx.invokeVector("value_counts", &.{ .{ .array = empty }, .{ .array = empty } }, none));
    // The plain result-type path cannot hold a struct over the input type.
    try std.testing.expectError(error.NotImplemented, registry.resolveResultType("value_counts", .vector, &.{.{ .array = empty }}, none));
    const short = [_]i32{1};
    const bufs = [_]common.SharedBuffer{ .empty, .fromSlice(std.mem.sliceAsBytes(&short)) };
    var malformed = try ArrayRef.fromBorrowed(allocator, .{ .data_type = .{ .int32 = {} }, .length = 2, .null_count = 0, .buffers = &bufs });
    defer malformed.release();
    inline for (.{ "unique", "value_counts", "dictionary_encode" }) |name| {
        try std.testing.expectError(error.InvalidInput, ctx.invokeVector(name, &.{.{ .array = malformed }}, none));
    }
}
