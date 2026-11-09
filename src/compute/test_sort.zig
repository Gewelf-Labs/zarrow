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

fn indicesOf(ctx: *ExecContext, name: []const u8, args: []const Datum, options: Options) ![]u64 {
    var out = try ctx.invokeVector(name, args, options);
    defer out.release();
    try out.array.data().validateFull();
    return std.testing.allocator.dupe(u64, try (common.array_mod.PrimitiveArray(u64){ .data = out.array.data() }).values());
}

/// select_k_unstable must give the first k rows of the stable sort (ties broken by row, as the
/// sort), partition_nth_indices a permutation whose pivot row is the one of the sort and whose
/// rows before the pivot are the first rows of the sort.
fn expectSelections(ctx: *ExecContext, args: []const Datum, keys: []const common.core.SortKey) !void {
    const allocator = std.testing.allocator;
    const sorted = try indicesOf(ctx, "sort_indices", args, .{ .sort = .{ .sort_keys = keys } });
    defer allocator.free(sorted);
    const n = sorted.len;
    for ([_]usize{ 0, 1, 7, n / 2, n, n + 3 }) |k| {
        const top = try indicesOf(ctx, "select_k_unstable", args, .{ .select_k = .{ .k = k, .sort_keys = keys } });
        defer allocator.free(top);
        try std.testing.expectEqualSlices(u64, sorted[0..@min(k, n)], top);
    }
    if (args.len != 1 or keys[0].order != .ascending) return;
    const seen = try allocator.alloc(bool, n);
    defer allocator.free(seen);
    for ([_]usize{ 0, 1, n / 3, n / 2 + 1, n -| 1, n }) |pivot| {
        if (pivot > n) continue;
        const part = try indicesOf(ctx, "partition_nth_indices", args, .{ .partition_nth = .{ .pivot = pivot, .null_placement = keys[0].null_placement } });
        defer allocator.free(part);
        try expectPermutation(part, n);
        if (pivot < n) try std.testing.expectEqual(sorted[pivot], part[pivot]);
        @memset(seen, false);
        for (sorted[0..pivot]) |r| seen[r] = true;
        for (part[0..pivot]) |r| try std.testing.expect(seen[r]);
    }
}

test "partition_nth_indices and select_k_unstable oracle: selections agree with the stable sort for radix and comparison columns, both orders and null placements" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    var prng = std.Random.DefaultPrng.init(0x5E1EC7);
    const rng = prng.random();
    var arrays: [6]ArrayRef = undefined;
    arrays[0] = try common.randomArray(i32, allocator, rng, 600, 3, 12, .{ .small = true });
    arrays[1] = try common.randomArray(f64, allocator, rng, 500, 5, 10, .{});
    arrays[2] = try common.randomArray(u8, allocator, rng, 300, 0, 20, .{});
    arrays[3] = try common.randomArray(i64, allocator, rng, 200, 1, 0, .{ .small = true });
    arrays[4] = try common.randomArray(bool, allocator, rng, 100, 2, 25, .{});
    arrays[5] = try common.randomStrings(.string, allocator, rng, 300, 4, 10);
    defer for (&arrays) |*a| a.release();
    for (arrays) |a| {
        var chunked = try common.chunkedAt(allocator, a, &.{ 1, a.data().length / 2 });
        defer chunked.release();
        for ([_]Datum{ .{ .array = a }, .{ .chunked = chunked } }) |arg| {
            for ([_]common.core.SortOrder{ .ascending, .descending }) |order| {
                for ([_]common.core.SortNullPlacement{ .at_end, .at_start }) |placement| {
                    try expectSelections(&ctx, &.{arg}, &.{.{ .order = order, .null_placement = placement }});
                }
            }
        }
    }
    // Two keys go through the comparison heap.
    const two = [_]common.core.SortKey{ .{ .order = .descending, .null_placement = .at_start }, .{} };
    try expectSelections(&ctx, &.{ .{ .array = arrays[2] }, .{ .array = arrays[5] } }, &two);
    // Small inputs and the empty input.
    var empty = try common.makeArray(i32, allocator, &.{});
    defer empty.release();
    try expectSelections(&ctx, &.{.{ .array = empty }}, &.{.{}});
    var one = try common.makeArray(f32, allocator, &.{std.math.nan(f32)});
    defer one.release();
    try expectSelections(&ctx, &.{.{ .array = one }}, &.{.{ .null_placement = .at_start }});
}

fn expectFloats(ctx: *ExecContext, name: []const u8, args: []const Datum, options: Options, expected: []const f64) !void {
    var out = try ctx.invokeVector(name, args, options);
    defer out.release();
    try out.array.data().validateFull();
    try std.testing.expect(out.array.data().data_type == .double);
    const got = try common.collect(f64, std.testing.allocator, out);
    defer std.testing.allocator.free(got);
    try std.testing.expectEqual(expected.len, got.len);
    for (expected, got) |e, g| try std.testing.expectApproxEqAbs(e, g.?, 1e-12);
}

fn expectPositions(ctx: *ExecContext, args: []const Datum, options: Options, expected: []const ?u64) !void {
    var out = try ctx.invokeVector("search_sorted", args, options);
    defer out.release();
    try out.array.data().validateFull();
    const got = try common.collect(u64, std.testing.allocator, out);
    defer std.testing.allocator.free(got);
    try common.expectRows(u64, expected, got);
}

test "rank_quantile, rank_normal and search_sorted hand-written values (pyarrow 26 results)" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    var a = try common.makeArray(f64, allocator, &.{ 3, null, 1, 1, nan, 2 });
    defer a.release();
    const d: Datum = .{ .array = a };
    const desc_start: Options = .{ .rank_quantile = .{ .order = .descending, .null_placement = .at_start } };
    try expectFloats(&ctx, "rank_quantile", &.{d}, none, &.{ 0.5833333333333334, 0.9166666666666666, 0.16666666666666666, 0.16666666666666666, 0.75, 0.4166666666666667 });
    try expectFloats(&ctx, "rank_quantile", &.{d}, desc_start, &.{ 0.4166666666666667, 0.08333333333333333, 0.8333333333333334, 0.8333333333333334, 0.25, 0.5833333333333334 });
    try expectFloats(&ctx, "rank_normal", &.{d}, none, &.{ 0.21042839424792484, 1.382994127100638, -0.9674215661017014, -0.9674215661017014, 0.6744897501960817, -0.2104283942479247 });
    try expectFloats(&ctx, "rank_normal", &.{d}, desc_start, &.{ -0.2104283942479247, -1.3829941271006387, 0.9674215661017014, 0.9674215661017014, -0.6744897501960817, 0.21042839424792484 });
    var chunked = try common.chunkedAt(allocator, a, &.{ 1, 4 });
    defer chunked.release();
    try expectFloats(&ctx, "rank_quantile", &.{.{ .chunked = chunked }}, none, &.{ 0.5833333333333334, 0.9166666666666666, 0.16666666666666666, 0.16666666666666666, 0.75, 0.4166666666666667 });
    var s = try common.makeStrings(.string, allocator, &.{ "b", "a", null, "a", "c", "d", "e" });
    defer s.release();
    try expectFloats(&ctx, "rank_normal", &.{.{ .array = s }}, none, &.{ -0.3661063568005697, -1.0675705238781417, 1.4652337926855223, -1.0675705238781417, 0.0, 0.3661063568005697, 0.7916386077433746 });
    var five = try common.makeArray(i64, allocator, &.{5});
    defer five.release();
    try expectFloats(&ctx, "rank_normal", &.{.{ .array = five }}, none, &.{0.0});
    var empty = try common.makeArray(i64, allocator, &.{});
    defer empty.release();
    try expectFloats(&ctx, "rank_quantile", &.{.{ .array = empty }}, none, &.{});
    try std.testing.expectError(error.InvalidOptions, ctx.invokeVector("rank_normal", &.{d}, .{ .rank = .{} }));

    // search_sorted: positions among the non-null values (nulls at the start or the end).
    const left: Options = .none;
    const right: Options = .{ .search_sorted = .{ .side = .right } };
    var v = try common.makeArray(i64, allocator, &.{ 1, 2, 2, 3, null, null });
    defer v.release();
    var needles = try common.makeArray(i64, allocator, &.{ 2, 0, 4, null, 3 });
    defer needles.release();
    const vn = [_]Datum{ .{ .array = v }, .{ .array = needles } };
    try expectPositions(&ctx, &vn, left, &.{ 1, 0, 4, null, 3 });
    try expectPositions(&ctx, &vn, right, &.{ 3, 0, 4, null, 4 });
    var lead = try common.makeArray(i64, allocator, &.{ null, null, 1, 2, 2 });
    defer lead.release();
    var two_zero = try common.makeArray(i64, allocator, &.{ 2, 0 });
    defer two_zero.release();
    try expectPositions(&ctx, &.{ .{ .array = lead }, .{ .array = two_zero } }, right, &.{ 5, 2 });
    var found = try ctx.invokeVector("search_sorted", &.{ .{ .array = v }, .{ .scalar = common.scalarOf(i64, 2) } }, left);
    defer found.release();
    try std.testing.expectEqual(@as(u64, 1), found.scalar.value.u64);
    var missing = try ctx.invokeVector("search_sorted", &.{ .{ .array = v }, .{ .scalar = common.scalarOf(i64, null) } }, left);
    defer missing.release();
    try std.testing.expect(missing.scalar.isNull());
    var f = try common.makeArray(f64, allocator, &.{ -0.0, 0.0, 1.0, nan, nan });
    defer f.release();
    var fneedles = try common.makeArray(f64, allocator, &.{ 0.0, -0.0, nan });
    defer fneedles.release();
    try expectPositions(&ctx, &.{ .{ .array = f }, .{ .array = fneedles } }, left, &.{ 0, 0, 3 });
    try expectPositions(&ctx, &.{ .{ .array = f }, .{ .array = fneedles } }, right, &.{ 2, 2, 5 });
    var strs = try common.makeStrings(.string, allocator, &.{ "a", "b", "b", "c" });
    defer strs.release();
    var sneedles = try common.makeStrings(.string, allocator, &.{ "b", "bb", null, "" });
    defer sneedles.release();
    try expectPositions(&ctx, &.{ .{ .array = strs }, .{ .array = sneedles } }, left, &.{ 1, 3, null, 0 });
    try expectPositions(&ctx, &.{ .{ .array = strs }, .{ .array = sneedles } }, right, &.{ 3, 3, null, 0 });
    var c = try common.makeArray(i64, allocator, &.{ 1, 2, 2, 3 });
    defer c.release();
    var cv = try common.chunkedAt(allocator, c, &.{2});
    defer cv.release();
    var cneedle_arr = try common.makeArray(i64, allocator, &.{ 2, 3, 9 });
    defer cneedle_arr.release();
    var cn = try common.chunkedAt(allocator, cneedle_arr, &.{1});
    defer cn.release();
    try expectPositions(&ctx, &.{ .{ .chunked = cv }, .{ .chunked = cn } }, right, &.{ 3, 4, 4 });
    var bools = try common.makeArray(bool, allocator, &.{ false, true, true });
    defer bools.release();
    var bneedles = try common.makeArray(bool, allocator, &.{ true, false });
    defer bneedles.release();
    try expectPositions(&ctx, &.{ .{ .array = bools }, .{ .array = bneedles } }, left, &.{ 1, 0 });
    var all_null = try common.makeArray(i64, allocator, &.{ null, null });
    defer all_null.release();
    try expectPositions(&ctx, &.{ .{ .array = all_null }, .{ .array = two_zero } }, left, &.{ 0, 0 });
    try expectPositions(&ctx, &.{ .{ .array = empty }, .{ .array = two_zero } }, left, &.{ 0, 0 });
    // Types must match (Arrow: "matching logical types"); nulls inside the values are rejected
    // (Arrow leaves the position unspecified).
    var i32s = try common.makeArray(i32, allocator, &.{ 1, 2 });
    defer i32s.release();
    try std.testing.expectError(error.NoMatchingKernel, ctx.invokeVector("search_sorted", &.{ .{ .array = i32s }, .{ .array = two_zero } }, left));
    var mid_null = try common.makeArray(i64, allocator, &.{ 1, null, 2 });
    defer mid_null.release();
    try std.testing.expectError(error.InvalidInput, ctx.invokeVector("search_sorted", &.{ .{ .array = mid_null }, .{ .array = two_zero } }, left));
}

test "sort_indices and array_sort_indices of the null type: every row ties, 0 .. n-1; sliced, chunked, with another key (pyarrow 26 results)" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    var n = try common.dense.nullArray(allocator, .{ .null = {} }, 5);
    defer n.release();
    inline for (.{ "sort_indices", "array_sort_indices" }) |name| {
        try expectIndices(&ctx, name, &.{.{ .array = n }}, none, &.{ 0, 1, 2, 3, 4 });
        try expectIndices(&ctx, name, &.{.{ .array = n }}, sortOpts(.descending, .at_start), &.{ 0, 1, 2, 3, 4 });
    }
    var tail = try n.slice(2, 3);
    defer tail.release();
    try expectIndices(&ctx, "array_sort_indices", &.{.{ .array = tail }}, none, &.{ 0, 1, 2 });
    var c = try common.chunkedAt(allocator, n, &.{ 1, 3 });
    defer c.release();
    try expectIndices(&ctx, "sort_indices", &.{.{ .chunked = c }}, none, &.{ 0, 1, 2, 3, 4 });
    // A null-type key ties everywhere: the next key decides.
    var ints = try common.makeArray(i32, allocator, &.{ 3, null, 1, 2, 1 });
    defer ints.release();
    try expectIndices(&ctx, "sort_indices", &.{ .{ .array = n }, .{ .array = ints } }, none, &.{ 2, 4, 3, 0, 1 });
    try std.testing.expect((try registry.resolveResultType("array_sort_indices", .vector, &.{.{ .array = n }}, none)) == .uint64);
    // rank and the selections keep rejecting it.
    try std.testing.expectError(error.NoMatchingKernel, ctx.invokeVector("rank", &.{.{ .array = n }}, none));
}

test "partition_nth_indices of the null type: 0 .. n-1 for any pivot; sliced, chunked and empty (pyarrow 26 results)" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    var n = try common.dense.nullArray(allocator, .{ .null = {} }, 3);
    defer n.release();
    // partition_nth_indices(nulls(3), pivot) -> uint64 [0, 1, 2] for pivot 0, 1 and 4.
    for ([_]usize{ 0, 1, 3, 4 }) |pivot| {
        try expectIndices(&ctx, "partition_nth_indices", &.{.{ .array = n }}, .{ .partition_nth = .{ .pivot = pivot } }, &.{ 0, 1, 2 });
        try expectIndices(&ctx, "partition_nth_indices", &.{.{ .array = n }}, .{ .partition_nth = .{ .pivot = pivot, .null_placement = .at_start } }, &.{ 0, 1, 2 });
    }
    var tail = try n.slice(1, 2);
    defer tail.release();
    try expectIndices(&ctx, "partition_nth_indices", &.{.{ .array = tail }}, .{ .partition_nth = .{ .pivot = 1 } }, &.{ 0, 1 });
    var c = try common.chunkedAt(allocator, n, &.{2});
    defer c.release();
    try expectIndices(&ctx, "partition_nth_indices", &.{.{ .chunked = c }}, .{ .partition_nth = .{ .pivot = 1 } }, &.{ 0, 1, 2 });
    var empty = try common.dense.nullArray(allocator, .{ .null = {} }, 0);
    defer empty.release();
    try expectIndices(&ctx, "partition_nth_indices", &.{.{ .array = empty }}, .{ .partition_nth = .{ .pivot = 0 } }, &.{});
    try std.testing.expect((try registry.resolveResultType("partition_nth_indices", .vector, &.{.{ .array = n }}, .{ .partition_nth = .{ .pivot = 0 } })) == .uint64);
    // Options are still required; a scalar is still rejected.
    try std.testing.expectError(error.InvalidOptions, ctx.invokeVector("partition_nth_indices", &.{.{ .array = n }}, none));
    try std.testing.expectError(error.NoMatchingKernel, ctx.invokeVector("partition_nth_indices", &.{.{ .scalar = common.Scalar.init(.{ .null = {} }, .null) }}, .{ .partition_nth = .{ .pivot = 0 } }));
}
