//! Dictionary-encoded inputs: unique, value_counts and dictionary_encode (on the indices, the
//! dictionary kept; chunks with different dictionaries unified), is_in / index_in and the
//! comparisons (decoded), sort_indices (decoded), group-by keys and hash_count /
//! hash_count_distinct / hash_distinct values (by index), and the functions Arrow has no
//! dictionary kernel for. Expected values are pyarrow 26 results.
const std = @import("std");
const common = @import("test_common.zig");
const hash = @import("hash_aggregate.zig");

const ArrayRef = common.ArrayRef;
const ChunkedArray = common.ChunkedArray;
const DataType = common.DataType;
const Datum = common.Datum;
const ExecContext = common.ExecContext;
const Options = common.Options;
const none = common.none;

const int8_index: common.datatype.IntType = .{ .bit_width = 8, .signed = true };
const int16_index: common.datatype.IntType = .{ .bit_width = 16, .signed = true };

/// The rows of a dictionary datum (array or chunked) as decoded strings (`validateFull` on every
/// part) and the index of every row (null: a null index).
fn expectDictionary(d: Datum, want_dict: []const ?[]const u8, want_indices: []const ?i64) !void {
    const allocator = std.testing.allocator;
    const arr = switch (d) {
        .array => |a| a,
        else => return error.TestUnexpectedResult,
    };
    try arr.data().validateFull();
    try std.testing.expect(arr.data().data_type == .dictionary);
    const dict = try common.collectStrings(allocator, .{ .array = arr.data().dictionary.? });
    defer allocator.free(dict);
    try common.expectStrings(want_dict, dict);
    try std.testing.expectEqual(want_indices.len, arr.data().length);
    for (want_indices, 0..) |w, i| {
        if (w == null) {
            try std.testing.expect(arr.data().isNull(i));
            continue;
        }
        try std.testing.expect(!arr.data().isNull(i));
        const got: i64 = switch (arr.data().data_type.dictionary.index_type.bit_width) {
            8 => (try (common.array_mod.PrimitiveArray(i8){ .data = arr.data() }).values())[i],
            16 => (try (common.array_mod.PrimitiveArray(i16){ .data = arr.data() }).values())[i],
            32 => (try (common.array_mod.PrimitiveArray(i32){ .data = arr.data() }).values())[i],
            else => return error.TestUnexpectedResult,
        };
        try std.testing.expectEqual(w.?, got);
    }
}

/// d = dictionary<int8, string>: indices [1, 0, None, 1, 3, 2] over ['b', 'a', None, 'z'], so
/// the rows are ['a', 'b', None, 'a', 'z', None] (a null index and an index to the null entry).
fn sample(allocator: std.mem.Allocator) !ArrayRef {
    var dict = try common.makeStrings(.string, allocator, &.{ "b", "a", null, "z" });
    defer dict.release();
    return common.makeDictionary(allocator, dict, int8_index, &.{ 1, 0, null, 1, 3, 2 });
}

test "unique, value_counts and dictionary_encode on dictionary arrays keep the dictionary (pyarrow 26 results)" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    var d = try sample(allocator);
    defer d.release();
    const dict = [_]?[]const u8{ "b", "a", null, "z" };

    var u = try ctx.invokeVector("unique", &.{.{ .array = d }}, none);
    defer u.release();
    try std.testing.expect(u.dataType().eql(d.data().data_type));
    try expectDictionary(u, &dict, &.{ 1, 0, null, 3, 2 });
    // A sliced input: rows ['b', None, 'a'].
    var s = try d.slice(1, 3);
    defer s.release();
    var us = try ctx.invokeVector("unique", &.{.{ .array = s }}, none);
    defer us.release();
    try expectDictionary(us, &dict, &.{ 0, null, 1 });

    var vc = try ctx.invokeVector("value_counts", &.{.{ .array = d }}, none);
    defer vc.release();
    try vc.array.data().validateFull();
    var values = try common.core.datumStructField(vc, 0);
    defer values.release();
    try expectDictionary(values, &dict, &.{ 1, 0, null, 3, 2 });
    var counts = try common.core.datumStructField(vc, 1);
    defer counts.release();
    const got_counts = try common.collect(i64, allocator, counts);
    defer allocator.free(got_counts);
    try common.expectRows(i64, &.{ 2, 1, 1, 1, 1 }, got_counts);
    var vc_type_arena = std.heap.ArenaAllocator.init(allocator);
    defer vc_type_arena.deinit();
    const vct = try registry.resolveResultTypeAlloc(vc_type_arena.allocator(), "value_counts", .vector, &.{.{ .array = d }}, none);
    try std.testing.expect(vct.struct_.fields[0].data_type.* == .dictionary);

    // dictionary_encode of a dictionary is the identity, whatever the null encoding.
    for ([_]Options{ none, .{ .dictionary_encode = .{ .null_encoding = .encode } } }) |o| {
        var e = try ctx.invokeVector("dictionary_encode", &.{.{ .array = d }}, o);
        defer e.release();
        try std.testing.expect(e.array.node == d.node);
        try std.testing.expect((try registry.resolveResultTypeAlloc(vc_type_arena.allocator(), "dictionary_encode", .vector, &.{.{ .array = d }}, o)).eql(d.data().data_type));
    }

    // Unused entries stay: indices [3, 3, None, 1] over ['u', 'a', 'b', 'c'].
    var dict2 = try common.makeStrings(.string, allocator, &.{ "u", "a", "b", "c" });
    defer dict2.release();
    var d2 = try common.makeDictionary(allocator, dict2, int8_index, &.{ 3, 3, null, 1 });
    defer d2.release();
    var unused_u = try ctx.invokeVector("unique", &.{.{ .array = d2 }}, none);
    defer unused_u.release();
    try expectDictionary(unused_u, &.{ "u", "a", "b", "c" }, &.{ 3, null, 1 });
    // No rows: no values, the dictionary kept.
    var empty = try d2.slice(0, 0);
    defer empty.release();
    var ue = try ctx.invokeVector("unique", &.{.{ .array = empty }}, none);
    defer ue.release();
    try expectDictionary(ue, &.{ "u", "a", "b", "c" }, &.{});

    // A dictionary far longer than the input groups through the hash table instead of a direct
    // table over the indices: the same result.
    var long_builder = try common.array_mod.StringBuilder.init(allocator, 5000, 30000);
    defer long_builder.deinit();
    var name: [16]u8 = undefined;
    for (0..5000) |k| try long_builder.append(try std.fmt.bufPrint(&name, "v{d}", .{k}));
    var long_dict = try long_builder.finish();
    defer long_dict.release();
    var sparse = try common.makeDictionary(allocator, long_dict, int16_index, &.{ 4999, 3, 4999, null });
    defer sparse.release();
    var us_long = try ctx.invokeVector("unique", &.{.{ .array = sparse }}, none);
    defer us_long.release();
    try us_long.array.data().validateFull();
    try std.testing.expect(us_long.array.data().dictionary.?.node == long_dict.node);
    const idx = try (common.array_mod.PrimitiveArray(i16){ .data = us_long.array.data() }).values();
    try std.testing.expectEqual(@as(usize, 3), idx.len);
    try std.testing.expectEqualSlices(i16, &.{ 4999, 3 }, idx[0..2]);
    try std.testing.expect(us_long.array.data().isNull(2));
    var vc_long = try ctx.invokeVector("value_counts", &.{.{ .array = sparse }}, none);
    defer vc_long.release();
    var long_counts = try common.core.datumStructField(vc_long, 1);
    defer long_counts.release();
    const lc = try common.collect(i64, allocator, long_counts);
    defer allocator.free(lc);
    try common.expectRows(i64, &.{ 2, 1, 1 }, lc);
}

test "unique and value_counts unify the dictionaries of chunks; dictionary_encode keeps the chunks (pyarrow 26 results)" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    var xy = try common.makeStrings(.string, allocator, &.{ "x", "y" });
    defer xy.release();
    var zx = try common.makeStrings(.string, allocator, &.{ "z", "x" });
    defer zx.release();
    var c0 = try common.makeDictionary(allocator, xy, int8_index, &.{ 0, 1, 0 });
    defer c0.release();
    var c1 = try common.makeDictionary(allocator, zx, int8_index, &.{ 1, 0, null });
    defer c1.release();
    var chunked = try ChunkedArray.init(allocator, c0.data().data_type, &.{ c0, c1 });
    defer chunked.release();
    var u = try ctx.invokeVector("unique", &.{.{ .chunked = chunked }}, none);
    defer u.release();
    try expectDictionary(u, &.{ "x", "y", "z" }, &.{ 0, 1, 2, null });
    var vc = try ctx.invokeVector("value_counts", &.{.{ .chunked = chunked }}, none);
    defer vc.release();
    var counts = try common.core.datumStructField(vc, 1);
    defer counts.release();
    const got = try common.collect(i64, allocator, counts);
    defer allocator.free(got);
    try common.expectRows(i64, &.{ 3, 1, 1, 1 }, got);
    var enc = try ctx.invokeVector("dictionary_encode", &.{.{ .chunked = chunked }}, none);
    defer enc.release();
    try std.testing.expect(enc == .chunked and enc.chunked.node == chunked.node);

    // Unused entries of every chunk are kept: ['x', 'y', 'w'] then ['q', 'x'].
    var xyw = try common.makeStrings(.string, allocator, &.{ "x", "y", "w" });
    defer xyw.release();
    var qx = try common.makeStrings(.string, allocator, &.{ "q", "x" });
    defer qx.release();
    var e0 = try common.makeDictionary(allocator, xyw, int8_index, &.{1});
    defer e0.release();
    var e1 = try common.makeDictionary(allocator, qx, int8_index, &.{0});
    defer e1.release();
    var unused = try ChunkedArray.init(allocator, e0.data().data_type, &.{ e0, e1 });
    defer unused.release();
    var uu = try ctx.invokeVector("unique", &.{.{ .chunked = unused }}, none);
    defer uu.release();
    try expectDictionary(uu, &.{ "x", "y", "w", "q" }, &.{ 1, 3 });

    // Chunks over one dictionary (or equal ones) are not unified.
    var xy_copy = try common.makeStrings(.string, allocator, &.{ "x", "y" });
    defer xy_copy.release();
    var same0 = try common.makeDictionary(allocator, xy, int8_index, &.{ 1, 0 });
    defer same0.release();
    var same1 = try common.makeDictionary(allocator, xy_copy, int8_index, &.{0});
    defer same1.release();
    var same = try ChunkedArray.init(allocator, same0.data().data_type, &.{ same0, same1 });
    defer same.release();
    var us = try ctx.invokeVector("unique", &.{.{ .chunked = same }}, none);
    defer us.release();
    try expectDictionary(us, &.{ "x", "y" }, &.{ 1, 0 });

    // No chunks: an empty dictionary.
    var no_chunks = try ChunkedArray.init(allocator, c0.data().data_type, &.{});
    defer no_chunks.release();
    var un = try ctx.invokeVector("unique", &.{.{ .chunked = no_chunks }}, none);
    defer un.release();
    try expectDictionary(un, &.{}, &.{});

    // Different dictionaries with null entries: Arrow "Cannot yet unify dictionaries with nulls".
    var nx = try common.makeStrings(.string, allocator, &.{ null, "x" });
    defer nx.release();
    var xn = try common.makeStrings(.string, allocator, &.{ "x", null });
    defer xn.release();
    var n0 = try common.makeDictionary(allocator, nx, int16_index, &.{ 0, 1 });
    defer n0.release();
    var n1 = try common.makeDictionary(allocator, xn, int16_index, &.{ 1, 0 });
    defer n1.release();
    var with_nulls = try ChunkedArray.init(allocator, n0.data().data_type, &.{ n0, n1 });
    defer with_nulls.release();
    try std.testing.expectError(error.InvalidInput, ctx.invokeVector("unique", &.{.{ .chunked = with_nulls }}, none));
    try std.testing.expectError(error.InvalidInput, ctx.invokeVector("value_counts", &.{.{ .chunked = with_nulls }}, none));
}

test "dictionary inputs to is_in, index_in, comparisons and sort_indices are decoded (pyarrow 26 results)" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    var d = try sample(allocator);
    defer d.release();
    const dd: Datum = .{ .array = d };
    var az = try common.makeStrings(.string, allocator, &.{ "a", "z" });
    defer az.release();
    var a_null = try common.makeStrings(.string, allocator, &.{ "a", null });
    defer a_null.release();
    var za = try common.makeStrings(.string, allocator, &.{ "z", "a" });
    defer za.release();
    try common.expectInvokeWith(bool, &ctx, "is_in", &.{dd}, .{ .set_lookup = .{ .value_set = .{ .array = az } } }, &.{ true, false, false, true, true, false });
    try common.expectInvokeWith(bool, &ctx, "is_in", &.{dd}, .{ .set_lookup = .{ .value_set = .{ .array = a_null }, .null_matching_behavior = .match } }, &.{ true, false, true, true, false, true });
    try common.expectInvokeWith(i32, &ctx, "index_in", &.{dd}, .{ .set_lookup = .{ .value_set = .{ .array = za } } }, &.{ 1, null, null, 1, 0, null });
    try common.expectInvoke(bool, &ctx, .scalar, "equal", &.{ dd, dd }, &.{ true, true, null, true, true, null });
    const a_scalar: Datum = .{ .scalar = common.Scalar.init(.{ .string = {} }, .{ .string = "a" }) };
    try common.expectInvoke(bool, &ctx, .scalar, "equal", &.{ dd, a_scalar }, &.{ true, false, null, true, false, null });

    var out = try ctx.invokeVector("sort_indices", &.{dd}, none);
    defer out.release();
    try std.testing.expectEqualSlices(u64, &.{ 0, 3, 1, 4, 2, 5 }, try (common.array_mod.PrimitiveArray(u64){ .data = out.array.data() }).values());

    // An index past the dictionary is rejected before any lookup.
    var bad_dict = try common.makeStrings(.string, allocator, &.{"a"});
    defer bad_dict.release();
    var bad = try common.makeDictionary(allocator, bad_dict, int8_index, &.{ 0, 5 });
    defer bad.release();
    try std.testing.expectError(error.InvalidInput, ctx.invokeVector("unique", &.{.{ .array = bad }}, none));
    try std.testing.expectError(error.InvalidInput, ctx.invokeVector("value_counts", &.{.{ .array = bad }}, none));
    try std.testing.expectError(error.InvalidInput, ctx.invokeVector("dictionary_encode", &.{.{ .array = bad }}, none));
    try std.testing.expectError(error.InvalidInput, ctx.invokeScalar("is_in", &.{.{ .array = bad }}, .{ .set_lookup = .{ .value_set = .{ .array = az } } }));
    try std.testing.expectError(error.InvalidInput, ctx.invokeVector("sort_indices", &.{.{ .array = bad }}, none));
    var keys = try common.makeArray(i64, allocator, &.{ 1, 1 });
    defer keys.release();
    try std.testing.expectError(error.InvalidInput, hash.groupBy(&ctx, &.{.{ .name = "k", .array = bad }}, &.{}));

    // Arrow has no dictionary kernel for min / max / min_max / sum / count_distinct / first.
    for ([_][]const u8{ "min", "max", "min_max", "sum", "count_distinct", "first" }) |name| {
        try std.testing.expectError(error.NoMatchingKernel, ctx.invokeAggregate(name, &.{dd}, none));
    }
}

test "dictionary group-by keys and hash_count, hash_count_distinct, hash_distinct values group by index (pyarrow 26 group_by)" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    var d = try sample(allocator);
    defer d.release();
    var v = try common.makeArray(i64, allocator, &.{ 1, 2, 3, 4, 5, 6 });
    defer v.release();

    // Keys: first-seen groups by index (pyarrow lists the same groups with the null index last).
    var by_key = try hash.groupBy(&ctx, &.{.{ .name = "k", .array = d }}, &.{.{ .name = "v_sum", .function = "hash_sum", .input = v }});
    defer by_key.deinit();
    try expectDictionary(.{ .array = by_key.columns[0] }, &.{ "b", "a", null, "z" }, &.{ 1, 0, null, 3, 2 });
    const sums = try common.collect(i64, allocator, .{ .array = by_key.columns[1] });
    defer allocator.free(sums);
    try common.expectRows(i64, &.{ 5, 2, 3, 5, 6 }, sums);
    // With a second key.
    var k2 = try common.makeArray(i64, allocator, &.{ 1, 1, 1, 1, 2, 2 });
    defer k2.release();
    var two = try hash.groupBy(&ctx, &.{ .{ .name = "k1", .array = d }, .{ .name = "k2", .array = k2 } }, &.{.{ .name = "v_sum", .function = "hash_sum", .input = v }});
    defer two.deinit();
    try std.testing.expectEqual(@as(usize, 5), two.numRows());

    // Values: k = [1, 1, 2, 2, 3, 3]; a null index is null, an index to the null entry valid.
    var k = try common.makeArray(i64, allocator, &.{ 1, 1, 2, 2, 3, 3 });
    defer k.release();
    var by_value = try hash.groupBy(&ctx, &.{.{ .name = "k", .array = k }}, &.{
        .{ .name = "count", .function = "hash_count", .input = d },
        .{ .name = "count_distinct", .function = "hash_count_distinct", .input = d },
        .{ .name = "distinct", .function = "hash_distinct", .input = d },
    });
    defer by_value.deinit();
    for ([_]usize{ 1, 2 }) |col| {
        const got = try common.collect(i64, allocator, .{ .array = by_value.columns[col] });
        defer allocator.free(got);
        try common.expectRows(i64, &.{ 2, 1, 2 }, got);
    }
    const lists = by_value.columns[3];
    try lists.data().validateFull();
    try std.testing.expect(lists.data().data_type.list.value_field.data_type.* == .dictionary);
    var decoded = try ctx.invokeScalar("cast", &.{.{ .array = lists.data().children[0] }}, .{ .cast = .{ .to_type = .{ .string = {} } } });
    defer decoded.release();
    const flat = try common.collectStrings(allocator, decoded);
    defer allocator.free(flat);
    try common.expectStrings(&.{ "a", "b", "a", "z", null }, flat);

    // A second batch over a different dictionary: Arrow "Unifying differing dictionaries".
    var grouper = try hash.Grouper.init(allocator, &.{d.data().data_type});
    defer grouper.deinit();
    var ids = try grouper.consume(&.{d});
    ids.release();
    var other_dict = try common.makeStrings(.string, allocator, &.{ "q", "a", null, "z" });
    defer other_dict.release();
    var other = try common.makeDictionary(allocator, other_dict, int8_index, &.{0});
    defer other.release();
    try std.testing.expectError(error.NotImplemented, grouper.consume(&.{other}));
    // An equal dictionary in another array is the same dictionary.
    var equal_dict = try common.makeStrings(.string, allocator, &.{ "b", "a", null, "z" });
    defer equal_dict.release();
    var again = try common.makeDictionary(allocator, equal_dict, int8_index, &.{ 0, 3 });
    defer again.release();
    var ids2 = try grouper.consume(&.{again});
    defer ids2.release();
    try std.testing.expectEqualSlices(u32, &.{ 1, 3 }, try (common.array_mod.PrimitiveArray(u32){ .data = ids2.data() }).values());
}
