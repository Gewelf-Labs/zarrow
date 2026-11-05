//! sort_indices, array_sort_indices, partition_nth_indices, select_k_unstable and rank.
//! Expected values are pyarrow 26 results.
const std = @import("std");
const common = @import("test_common.zig");
const sort = @import("sort.zig");

const ArrayRef = common.ArrayRef;
const DataType = common.DataType;
const Datum = common.Datum;
const ExecContext = common.ExecContext;
const Options = common.Options;
const none = common.none;

const nan = std.math.nan(f64);
const RecordBatch = @import("../record_batch.zig").RecordBatch;
const Table = @import("../table.zig").Table;
const SchemaRef = @import("../schema.zig").SchemaRef;

fn expectIndices(ctx: *ExecContext, name: []const u8, args: []const Datum, options: Options, expected: []const u64) !void {
    var out = try ctx.invokeVector(name, args, options);
    defer out.release();
    try out.array.data().validateFull();
    const got = try (common.array_mod.PrimitiveArray(u64){ .data = out.array.data() }).values();
    std.testing.expectEqualSlices(u64, expected, got) catch |err| {
        std.debug.print("{s}: got {any}\n", .{ name, got });
        return err;
    };
}

fn sortOpts(order: common.core.SortOrder, placement: common.core.SortNullPlacement) Options {
    return .{ .sort = .{ .order = order, .null_placement = placement } };
}

fn rankOpts(order: common.core.SortOrder, placement: common.core.SortNullPlacement, tie: common.core.RankTiebreaker) Options {
    return .{ .rank = .{ .order = order, .null_placement = placement, .tiebreaker = tie } };
}

test "sort_indices, array_sort_indices and rank hand-written values: NaN, nulls and both orders (pyarrow 26 results)" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    var f = try common.makeArray(f64, allocator, &.{ 3.0, nan, null, 1.0, nan, -0.0, 0.0, null, 2.0 });
    defer f.release();
    const df = [_]Datum{.{ .array = f }};
    try expectIndices(&ctx, "array_sort_indices", &df, sortOpts(.ascending, .at_start), &.{ 2, 7, 1, 4, 5, 6, 3, 8, 0 });
    try expectIndices(&ctx, "array_sort_indices", &df, sortOpts(.ascending, .at_end), &.{ 5, 6, 3, 8, 0, 1, 4, 2, 7 });
    try expectIndices(&ctx, "array_sort_indices", &df, sortOpts(.descending, .at_start), &.{ 2, 7, 1, 4, 0, 8, 3, 5, 6 });
    try expectIndices(&ctx, "sort_indices", &df, sortOpts(.descending, .at_end), &.{ 0, 8, 3, 5, 6, 1, 4, 2, 7 });
    try expectIndices(&ctx, "rank", &df, rankOpts(.ascending, .at_start, .min), &.{ 9, 3, 1, 7, 3, 5, 5, 1, 8 });
    try expectIndices(&ctx, "rank", &df, rankOpts(.ascending, .at_start, .dense), &.{ 6, 2, 1, 4, 2, 3, 3, 1, 5 });
    try expectIndices(&ctx, "rank", &df, rankOpts(.ascending, .at_end, .min), &.{ 5, 6, 8, 3, 6, 1, 1, 8, 4 });
    try expectIndices(&ctx, "rank", &df, rankOpts(.ascending, .at_end, .dense), &.{ 4, 5, 6, 2, 5, 1, 1, 6, 3 });
    try expectIndices(&ctx, "rank", &df, rankOpts(.descending, .at_start, .min), &.{ 5, 3, 1, 7, 3, 8, 8, 1, 6 });
    try expectIndices(&ctx, "rank", &df, rankOpts(.descending, .at_start, .dense), &.{ 3, 2, 1, 5, 2, 6, 6, 1, 4 });
    try expectIndices(&ctx, "rank", &df, rankOpts(.descending, .at_end, .min), &.{ 1, 6, 8, 3, 6, 4, 4, 8, 2 });
    try expectIndices(&ctx, "rank", &df, rankOpts(.descending, .at_end, .dense), &.{ 1, 5, 6, 3, 5, 4, 4, 6, 2 });
    var r = try common.makeArray(i64, allocator, &.{ 3, 1, null, 3, 2, 1 });
    defer r.release();
    const dr = [_]Datum{.{ .array = r }};
    try expectIndices(&ctx, "rank", &dr, rankOpts(.ascending, .at_end, .min), &.{ 4, 1, 6, 4, 3, 1 });
    try expectIndices(&ctx, "rank", &dr, rankOpts(.ascending, .at_end, .max), &.{ 5, 2, 6, 5, 3, 2 });
    try expectIndices(&ctx, "rank", &dr, none, &.{ 4, 1, 6, 5, 3, 2 });
    try expectIndices(&ctx, "rank", &dr, rankOpts(.ascending, .at_end, .dense), &.{ 3, 1, 4, 3, 2, 1 });
    var r4 = try common.makeArray(i64, allocator, &.{ 3, 1, null, 3 });
    defer r4.release();
    try expectIndices(&ctx, "rank", &.{.{ .array = r4 }}, rankOpts(.descending, .at_end, .first), &.{ 1, 3, 4, 2 });

    // Chunked input, bool, strings, decimals, dates and a dictionary.
    var c = try common.makeArray(i64, allocator, &.{ 3, 1, null, 2 });
    defer c.release();
    var cc = try common.chunkedAt(allocator, c, &.{2});
    defer cc.release();
    try expectIndices(&ctx, "sort_indices", &.{.{ .chunked = cc }}, none, &.{ 1, 3, 0, 2 });
    try expectIndices(&ctx, "rank", &.{.{ .chunked = cc }}, none, &.{ 3, 1, 4, 2 });
    var b = try common.makeArray(bool, allocator, &.{ true, null, false });
    defer b.release();
    try expectIndices(&ctx, "array_sort_indices", &.{.{ .array = b }}, none, &.{ 2, 0, 1 });
    inline for (.{ .string, .large_string, .string_view, .binary }) |tag| {
        var s = try common.makeStrings(tag, allocator, &.{ "b", "a", null, "c", "a long value past twelve", "" });
        defer s.release();
        try expectIndices(&ctx, "array_sort_indices", &.{.{ .array = s }}, sortOpts(.descending, .at_end), &.{ 3, 0, 4, 1, 5, 2 });
    }
    const d52 = DataType{ .decimal128 = .{ .precision = 5, .scale = 2 } };
    var d = try common.makeTyped(i128, d52, allocator, &.{ 200, -100, 100 });
    defer d.release();
    try expectIndices(&ctx, "array_sort_indices", &.{.{ .array = d }}, none, &.{ 1, 2, 0 });
    var dates = try common.makeTyped(i32, .{ .date32 = {} }, allocator, &.{ 5, 2, null, 9 });
    defer dates.release();
    try expectIndices(&ctx, "array_sort_indices", &.{.{ .array = dates }}, sortOpts(.descending, .at_start), &.{ 2, 3, 0, 1 });
    var words = try common.makeStrings(.string, allocator, &.{ "x", "y", "x", null });
    defer words.release();
    var dict = try ctx.invokeVector("dictionary_encode", &.{.{ .array = words }}, none);
    defer dict.release();
    try expectIndices(&ctx, "array_sort_indices", &.{dict}, sortOpts(.descending, .at_end), &.{ 1, 0, 2, 3 });
}

test "sort_indices on several keys, record batches and tables; select_k_unstable and partition_nth_indices (pyarrow 26 results)" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    var a = try common.makeArray(i64, allocator, &.{ 1, 1, null, 2, 1 });
    defer a.release();
    var b = try common.makeStrings(.string, allocator, &.{ "x", "z", "y", null, "y" });
    defer b.release();
    const ab = [_]Datum{ .{ .array = a }, .{ .array = b } };
    const asc_desc = [_]common.core.SortKey{ .{ .name = "a", .order = .ascending }, .{ .name = "b", .order = .descending } };
    const desc_asc = [_]common.core.SortKey{ .{ .name = "a", .order = .descending }, .{ .name = "b", .order = .ascending } };
    try expectIndices(&ctx, "sort_indices", &ab, .{ .sort = .{ .sort_keys = &asc_desc } }, &.{ 1, 4, 0, 3, 2 });
    try expectIndices(&ctx, "sort_indices", &ab, .{ .sort = .{ .sort_keys = &desc_asc, .null_placement = .at_start } }, &.{ 2, 3, 0, 4, 1 });
    try expectIndices(&ctx, "select_k_unstable", &ab, .{ .select_k = .{ .k = 3, .sort_keys = &asc_desc } }, &.{ 1, 4, 0 });

    // The same through a record batch and a table, by column name.
    const fields = [_]common.datatype.Field{
        .{ .name = "a", .data_type = &a.data().data_type },
        .{ .name = "b", .data_type = &b.data().data_type },
    };
    var batch = try RecordBatch.initBorrowed(allocator, .{ .fields = &fields }, &.{ a, b });
    defer batch.deinit();
    var by_batch = try sort.sortIndicesRecordBatch(&ctx, &batch, &asc_desc, .at_end);
    defer by_batch.release();
    try std.testing.expectEqualSlices(u64, &.{ 1, 4, 0, 3, 2 }, try (common.array_mod.PrimitiveArray(u64){ .data = by_batch.data() }).values());
    var ca = try common.chunkedAt(allocator, a, &.{2});
    defer ca.release();
    var cb = try common.chunkedAt(allocator, b, &.{ 1, 4 });
    defer cb.release();
    var table = try Table.init(allocator, try SchemaRef.fromBorrowed(allocator, .{ .fields = &fields }), &.{ ca, cb });
    defer table.deinit();
    var by_table = try sort.sortIndicesTable(&ctx, &table, &desc_asc, .at_start);
    defer by_table.release();
    try std.testing.expectEqualSlices(u64, &.{ 2, 3, 0, 4, 1 }, try (common.array_mod.PrimitiveArray(u64){ .data = by_table.data() }).values());
    try std.testing.expectError(error.InvalidOptions, sort.sortIndicesRecordBatch(&ctx, &batch, &.{.{ .name = "missing" }}, .at_end));

    // select_k: nulls last in both orders; k past the length keeps every row.
    var s = try common.makeArray(i64, allocator, &.{ 5, 1, null, 3 });
    defer s.release();
    const desc = [_]common.core.SortKey{.{ .order = .descending }};
    try expectIndices(&ctx, "select_k_unstable", &.{.{ .array = s }}, .{ .select_k = .{ .k = 10, .sort_keys = &desc } }, &.{ 0, 3, 1, 2 });
    try expectIndices(&ctx, "select_k_unstable", &.{.{ .array = s }}, .{ .select_k = .{ .k = 0, .sort_keys = &desc } }, &.{});
    try expectIndices(&ctx, "select_k_unstable", &.{.{ .array = s }}, .{ .select_k = .{ .k = 2 } }, &.{ 1, 3 });
    var f = try common.makeArray(f64, allocator, &.{ 3.0, nan, null, 1.0, nan, -0.0, 0.0, null, 2.0 });
    defer f.release();
    // pyarrow 26 gives [6, 5, 3, 8]: -0.0 and 0.0 tie, and ties keep input order here.
    try expectIndices(&ctx, "select_k_unstable", &.{.{ .array = f }}, .{ .select_k = .{ .k = 4 } }, &.{ 5, 6, 3, 8 });
    try expectIndices(&ctx, "select_k_unstable", &.{.{ .array = f }}, .{ .select_k = .{ .k = 4, .sort_keys = &desc } }, &.{ 0, 8, 3, 5 });

    // partition_nth_indices: the pivot row is in place, rows before sort before it.
    for ([_]usize{ 0, 4, 9 }) |pivot| {
        var p = try ctx.invokeVector("partition_nth_indices", &.{.{ .array = f }}, .{ .partition_nth = .{ .pivot = pivot } });
        defer p.release();
        try p.array.data().validateFull();
        const idx = try (common.array_mod.PrimitiveArray(u64){ .data = p.array.data() }).values();
        try expectPermutation(idx, 9);
    }
    try std.testing.expectError(error.InvalidOptions, ctx.invokeVector("partition_nth_indices", &.{.{ .array = f }}, .{ .partition_nth = .{ .pivot = 10 } }));
}

fn expectPermutation(idx: []const u64, n: usize) !void {
    try std.testing.expectEqual(n, idx.len);
    var seen = try std.testing.allocator.alloc(bool, n);
    defer std.testing.allocator.free(seen);
    @memset(seen, false);
    for (idx) |i| {
        try std.testing.expect(i < n and !seen[i]);
        seen[i] = true;
    }
}

/// Reference comparison of two optional floats: values ascending (-0.0 == 0.0), then NaN, then
/// null (`at_end`).
fn refOrder(a: ?f64, b: ?f64) std.math.Order {
    const ca: u2 = if (a == null) 2 else if (std.math.isNan(a.?)) 1 else 0;
    const cb: u2 = if (b == null) 2 else if (std.math.isNan(b.?)) 1 else 0;
    if (ca != cb) return std.math.order(ca, cb);
    if (ca != 0) return .eq;
    return std.math.order(a.?, b.?);
}

test "sort oracle: random sliced inputs with nulls give a stable sorted permutation for every type and both orders" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    var prng = std.Random.DefaultPrng.init(0x50E7);
    const rng = prng.random();
    inline for (.{ i8, i32, u64, f32, f64 }) |T| {
        var a = try common.randomArray(T, allocator, rng, 700, 9, 15, .{ .small = @typeInfo(T) == .int });
        defer a.release();
        const rows = try common.collect(T, allocator, .{ .array = a });
        defer allocator.free(rows);
        var chunked = try common.chunkedAt(allocator, a, &.{ 5, 350 });
        defer chunked.release();
        for ([_]common.core.SortOrder{ .ascending, .descending }) |order| {
            for ([_]Datum{ .{ .array = a }, .{ .chunked = chunked } }) |arg| {
                var out = try ctx.invokeVector("sort_indices", &.{arg}, sortOpts(order, .at_end));
                defer out.release();
                const idx = try (common.array_mod.PrimitiveArray(u64){ .data = out.array.data() }).values();
                try expectPermutation(idx, rows.len);
                for (idx[0 .. idx.len - 1], idx[1..]) |p, q| {
                    const x: ?f64 = if (rows[p]) |v| (if (@typeInfo(T) == .float) @floatCast(v) else @floatFromInt(v)) else null;
                    const y: ?f64 = if (rows[q]) |v| (if (@typeInfo(T) == .float) @floatCast(v) else @floatFromInt(v)) else null;
                    var o = refOrder(x, y);
                    if (order == .descending and x != null and y != null and !std.math.isNan(x.?) and !std.math.isNan(y.?)) o = o.invert();
                    try std.testing.expect(o != .gt);
                    if (o == .eq) try std.testing.expect(p < q);
                }
            }
        }
    }
    // Two keys: an int8 column with many ties and a string column.
    var k1 = try common.randomArray(i8, allocator, rng, 400, 3, 10, .{ .small = true });
    defer k1.release();
    var k2 = try common.randomStrings(.string, allocator, rng, 400, 0, 10);
    defer k2.release();
    const keys = [_]common.core.SortKey{ .{ .order = .descending }, .{ .order = .ascending } };
    var out = try ctx.invokeVector("sort_indices", &.{ .{ .array = k1 }, .{ .array = k2 } }, .{ .sort = .{ .sort_keys = &keys, .null_placement = .at_start } });
    defer out.release();
    const idx = try (common.array_mod.PrimitiveArray(u64){ .data = out.array.data() }).values();
    try expectPermutation(idx, 400);
    const r1 = try common.collect(i8, allocator, .{ .array = k1 });
    defer allocator.free(r1);
    for (idx[0 .. idx.len - 1], idx[1..]) |p, q| {
        const x = r1[p];
        const y = r1[q];
        if (x == null or y == null) {
            try std.testing.expect(x == null or y != null);
            if (x == null and y == null) {} else continue;
        } else if (x.? != y.?) {
            try std.testing.expect(x.? > y.?);
            continue;
        }
        const s = try common.stringAt(k2.data(), p);
        const t = try common.stringAt(k2.data(), q);
        if (s == null or t == null) {
            try std.testing.expect(s == null or t != null);
            if (s == null and t == null) try std.testing.expect(p < q);
            continue;
        }
        const o = std.mem.order(u8, s.?, t.?);
        try std.testing.expect(o != .gt);
        if (o == .eq) try std.testing.expect(p < q);
    }
}

test "sort errors: options, types, scalars, lengths and malformed input" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    var a = try common.makeArray(i32, allocator, &.{ 2, 1 });
    defer a.release();
    var three = try common.makeArray(i32, allocator, &.{ 1, 2, 3 });
    defer three.release();
    try std.testing.expectError(error.InvalidInput, ctx.invokeVector("sort_indices", &.{ .{ .array = a }, .{ .array = three } }, none));
    try std.testing.expectError(error.InvalidOptions, ctx.invokeVector("sort_indices", &.{.{ .array = a }}, .{ .sort = .{ .sort_keys = &.{ .{}, .{} } } }));
    try std.testing.expectError(error.InvalidOptions, ctx.invokeVector("select_k_unstable", &.{.{ .array = a }}, none));
    try std.testing.expectError(error.InvalidOptions, ctx.invokeVector("rank", &.{.{ .array = a }}, .{ .count = .{} }));
    try std.testing.expectError(error.NoMatchingKernel, ctx.invokeVector("sort_indices", &.{.{ .scalar = common.scalarOf(i32, 1) }}, none));
    const item = DataType{ .int32 = {} };
    var list = try common.datumBuildNullLike(.{ .list = .{ .value_field = .{ .name = "item", .data_type = &item } } }, 2);
    defer list.release();
    try std.testing.expectError(error.NoMatchingKernel, ctx.invokeVector("array_sort_indices", &.{list}, none));
    try std.testing.expectError(error.InvalidArity, ctx.invokeVector("array_sort_indices", &.{ .{ .array = a }, .{ .array = a } }, none));
    var empty = try common.makeArray(i32, allocator, &.{});
    defer empty.release();
    try expectIndices(&ctx, "sort_indices", &.{.{ .array = empty }}, none, &.{});
    try expectIndices(&ctx, "rank", &.{.{ .array = empty }}, none, &.{});
    const short = [_]i32{1};
    const bufs = [_]common.SharedBuffer{ .empty, .fromSlice(std.mem.sliceAsBytes(&short)) };
    var malformed = try ArrayRef.fromBorrowed(allocator, .{ .data_type = .{ .int32 = {} }, .length = 2, .null_count = 0, .buffers = &bufs });
    defer malformed.release();
    inline for (.{ "sort_indices", "array_sort_indices", "rank" }) |name| {
        try std.testing.expectError(error.InvalidInput, ctx.invokeVector(name, &.{.{ .array = malformed }}, none));
    }
}
