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
    // The null type (pyarrow 26): a dictionary of one null; mask leaves every index null,
    // encode points every row at it.
    var nulls = try common.datumBuildNullLike(.{ .null = {} }, 3);
    defer nulls.release();
    for ([_]bool{ false, true }) |encode| {
        const o: common.Options = if (encode) .{ .dictionary_encode = .{ .null_encoding = .encode } } else none;
        var ne = try ctx.invokeVector("dictionary_encode", &.{nulls}, o);
        defer ne.release();
        try ne.array.data().validateFull();
        try std.testing.expect(ne.array.data().dictionary.?.data().data_type == .null);
        try std.testing.expectEqual(@as(usize, 1), ne.array.data().dictionary.?.data().length);
        try std.testing.expectEqual(@as(usize, if (encode) 0 else 3), ne.array.data().null_count.?);
    }
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

test "float16 keys and signs: unique, value_counts, dictionary_encode, count_distinct, is_in, sorts, negate, abs and sign (pyarrow 26 results)" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    const h_nan = std.math.nan(f16);
    const h_inf = std.math.inf(f16);
    // [1.5, -0.0, 0.0, NaN, 1.5, NaN, inf, null], sliced off a longer array.
    var full = try common.makeArray(f16, allocator, &.{ 9, 1.5, -0.0, 0.0, h_nan, 1.5, h_nan, h_inf, null });
    defer full.release();
    var a = try full.slice(1, 8);
    defer a.release();
    const d: Datum = .{ .array = a };
    // NaN is one value, -0.0 and 0.0 are two (pyarrow 26: unique, value_counts, dictionary_encode).
    var uniq = try vector(&ctx, "unique", d, none);
    defer uniq.release();
    const u = try common.collect(f16, allocator, .{ .array = uniq });
    defer allocator.free(u);
    try common.expectRows(f16, &.{ 1.5, -0.0, 0.0, h_nan, h_inf, null }, u);
    try std.testing.expect(std.math.signbit(u[1].?) and !std.math.signbit(u[2].?));
    var vc = try vector(&ctx, "value_counts", d, none);
    defer vc.release();
    const counts = try common.collect(i64, allocator, .{ .array = vc.data().children[1] });
    defer allocator.free(counts);
    try common.expectRows(i64, &.{ 2, 1, 1, 2, 1, 1 }, counts);
    var enc = try vector(&ctx, "dictionary_encode", d, none);
    defer enc.release();
    const ix = try enc.data().buffers[1].typedSlice(i32);
    try std.testing.expectEqualSlices(i32, &.{ 0, 1, 2, 3, 0, 3, 4 }, ix[enc.data().offset..][0..7]);
    try std.testing.expect(enc.data().isNull(7));
    try std.testing.expectEqual(@as(usize, 5), enc.data().dictionary.?.data().length);
    var cd = try ctx.invokeAggregate("count_distinct", &.{d}, none);
    defer cd.release();
    try std.testing.expectEqual(@as(i64, 5), cd.scalar.value.i64);
    // is_in / index_in (pyarrow 26 has no float16 kernel; zarrow's float rules: NaN matches NaN).
    var set = try common.makeArray(f16, allocator, &.{ 1.5, h_nan });
    defer set.release();
    try common.expectInvokeWith(bool, &ctx, "is_in", &.{d}, .{ .set_lookup = .{ .value_set = .{ .array = set } } }, &.{ true, false, false, true, true, true, false, false });
    // sort_indices: -0.0 equals 0.0, NaN after the values, nulls last (pyarrow 26 has no kernel).
    var order = try vector(&ctx, "sort_indices", d, none);
    defer order.release();
    const o = try common.collect(u64, allocator, .{ .array = order });
    defer allocator.free(o);
    try common.expectRows(u64, &.{ 1, 2, 0, 4, 6, 3, 5, 7 }, o);
    // negate, sign (pyarrow 26) and abs.
    try common.expectInvokeWith(f16, &ctx, "negate", &.{d}, none, &.{ -1.5, 0.0, -0.0, h_nan, -1.5, h_nan, -h_inf, null });
    try common.expectInvokeWith(f16, &ctx, "negate_checked", &.{d}, none, &.{ -1.5, 0.0, -0.0, h_nan, -1.5, h_nan, -h_inf, null });
    try common.expectInvokeWith(f16, &ctx, "sign", &.{d}, none, &.{ 1, 0, 0, h_nan, 1, h_nan, 1, null });
    try common.expectInvokeWith(f16, &ctx, "abs", &.{d}, none, &.{ 1.5, 0.0, 0.0, h_nan, 1.5, h_nan, h_inf, null });
}

test "dictionary_decode: nulls, null entries, sliced, chunked, non-dictionary and extension inputs returned as is (pyarrow 26 results)" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);

    // DictionaryArray.from_arrays([0, None, 1, 0], ['a', None]) -> ['a', null, null, 'a'].
    var dict = try common.makeStrings(.string, allocator, &.{ "a", null });
    defer dict.release();
    var d = try common.makeDictionary(allocator, dict, .{ .bit_width = 64, .signed = true }, &.{ 0, null, 1, 0 });
    defer d.release();
    var out = try ctx.invokeVector("dictionary_decode", &.{.{ .array = d }}, none);
    defer out.release();
    try std.testing.expect(out.dataType() == .string);
    const got = try common.collectStrings(allocator, out);
    defer allocator.free(got);
    try common.expectStrings(&.{ "a", null, null, "a" }, got);
    try std.testing.expect((try registry.resolveResultType("dictionary_decode", .vector, &.{.{ .array = d }}, none)) == .string);

    // Sliced and chunked (each chunk decoded on its own; the chunks may have other dictionaries).
    var nums = try common.makeArray(i32, allocator, &.{ 10, 20, 30 });
    defer nums.release();
    var di = try common.makeDictionary(allocator, nums, .{ .bit_width = 8, .signed = false }, &.{ 2, 0, null, 1, 1 });
    defer di.release();
    var ds = try di.slice(1, 3);
    defer ds.release();
    try common.expectInvoke(i32, &ctx, .vector, "dictionary_decode", &.{.{ .array = ds }}, &.{ 10, null, 20 });
    var other = try common.makeArray(i32, allocator, &.{ 7, 8, 9 });
    defer other.release();
    var d2 = try common.makeDictionary(allocator, other, .{ .bit_width = 8, .signed = false }, &.{ 2, 2 });
    defer d2.release();
    var parts = [_]ArrayRef{ di, ds, d2 };
    var c = try common.ChunkedArray.init(allocator, di.data().data_type, &parts);
    defer c.release();
    var dc = try ctx.invokeVector("dictionary_decode", &.{.{ .chunked = c }}, none);
    defer dc.release();
    try std.testing.expectEqual(@as(usize, 3), dc.chunked.numChunks());
    try common.expectInvoke(i32, &ctx, .vector, "dictionary_decode", &.{.{ .chunked = c }}, &.{ 30, 10, null, 20, 20, 10, null, 20, 9, 9 });

    // A non-dictionary input comes back as is (the same array).
    var plain = try ctx.invokeVector("dictionary_decode", &.{.{ .array = nums }}, none);
    defer plain.release();
    try std.testing.expect(plain.array.node == nums.node);
    // An extension type over a dictionary is not a dictionary type: returned as is (pyarrow 26).
    const ext_type = DataType{ .extension = .{ .name = "my.ext", .storage_type = &d.data().data_type } };
    var ext_data = d.data().*;
    ext_data.data_type = ext_type;
    var ext = try ArrayRef.fromBorrowed(allocator, ext_data);
    defer ext.release();
    var eo = try ctx.invokeVector("dictionary_decode", &.{.{ .array = ext }}, none);
    defer eo.release();
    try std.testing.expect(eo.array.node == ext.node);
    // Scalars: Arrow's TypeError "Expected an Array or a Chunked Array".
    try std.testing.expectError(error.NoMatchingKernel, ctx.invokeVector("dictionary_decode", &.{.{ .scalar = common.scalarOf(i32, 1) }}, none));
    try std.testing.expectError(error.InvalidOptions, ctx.invokeVector("dictionary_decode", &.{.{ .array = d }}, .{ .take = .{} }));

    // An index past the dictionary fails the dictionary check before any read.
    var bad = try common.makeDictionary(allocator, nums, .{ .bit_width = 32, .signed = true }, &.{0});
    defer bad.release();
    var short = try nums.slice(0, 0);
    defer short.release();
    var bad_data = bad.data().*;
    bad_data.dictionary = short;
    var malformed = try ArrayRef.fromBorrowed(allocator, bad_data);
    defer malformed.release();
    try std.testing.expectError(error.InvalidInput, ctx.invokeVector("dictionary_decode", &.{.{ .array = malformed }}, none));
}
