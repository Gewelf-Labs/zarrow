const std = @import("std");
const common = @import("test_common.zig");

const ArrayRef = common.ArrayRef;
const DataType = common.DataType;
const Datum = common.Datum;
const ExecContext = common.ExecContext;
const hash = common.kernels.hash_aggregate;
const none = common.none;

/// Rows of column `name` of a group-by result as optionals of `T`; caller frees.
fn column(comptime T: type, batch: anytype, name: []const u8) ![]?T {
    return common.collect(T, std.testing.allocator, .{ .array = batch.columnByName(name).?.* });
}

fn expectColumn(comptime T: type, batch: anytype, name: []const u8, expected: []const ?T) !void {
    const got = try column(T, batch, name);
    defer std.testing.allocator.free(got);
    common.expectRows(T, expected, got) catch |err| {
        std.debug.print("column {s}\n", .{name});
        return err;
    };
}

/// Group ids of `keys` through a fresh Grouper; caller releases.
fn groupIds(keys: []const ArrayRef) !ArrayRef {
    var types: [4]DataType = undefined;
    for (keys, 0..) |k, i| types[i] = k.data().data_type;
    var g = try hash.Grouper.init(std.testing.allocator, types[0..keys.len]);
    defer g.deinit();
    return g.consume(keys);
}

test "hash aggregate hand-written values: group by one and two keys, every hash_ function (Arrow C++ results)" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);

    // Expected values from pyarrow 26 `Table.group_by(...).aggregate(...)`.
    var k = try common.makeArray(i64, allocator, &.{ 1, 2, 1, null, 2, null, 3 });
    defer k.release();
    var k2 = try common.makeStrings(.string, allocator, &.{ "a", "b", "a", "x", "c", "x", "z" });
    defer k2.release();
    var v = try common.makeArray(f64, allocator, &.{ 1.0, 2.0, null, 4.0, std.math.nan(f64), 6.0, null });
    defer v.release();
    var i = try common.makeArray(i64, allocator, &.{ 1, 2, 3, 4, 5, 6, null });
    defer i.release();
    var b = try common.makeArray(bool, allocator, &.{ true, false, null, true, true, null, null });
    defer b.release();
    var batch = try hash.groupBy(&ctx, &.{.{ .name = "k", .array = k }}, &.{
        .{ .function = "hash_sum", .input = i, .name = "i_sum" },
        .{ .function = "hash_sum", .input = v, .name = "v_sum" },
        .{ .function = "hash_mean", .input = i, .name = "i_mean" },
        .{ .function = "hash_min", .input = v, .name = "v_min" },
        .{ .function = "hash_max", .input = v, .name = "v_max" },
        .{ .function = "hash_min_max", .input = i, .name = "i_min_max" },
        .{ .function = "hash_count", .input = i, .name = "i_count" },
        .{ .function = "hash_count_all", .name = "count_all" },
        .{ .function = "hash_any", .input = b, .name = "b_any" },
        .{ .function = "hash_all", .input = b, .name = "b_all" },
        .{ .function = "hash_count_distinct", .input = i, .name = "i_count_distinct" },
        .{ .function = "hash_distinct", .input = i, .name = "i_distinct" },
    });
    defer batch.deinit();
    try std.testing.expectEqual(@as(usize, 4), batch.numRows());
    for (batch.columns) |c| try c.data().validateFull();
    try expectColumn(i64, &batch, "k", &.{ 1, 2, null, 3 });
    try expectColumn(i64, &batch, "i_sum", &.{ 4, 7, 10, null });
    try expectColumn(f64, &batch, "v_sum", &.{ 1.0, std.math.nan(f64), 10.0, null });
    try expectColumn(f64, &batch, "i_mean", &.{ 2.0, 3.5, 5.0, null });
    try expectColumn(f64, &batch, "v_min", &.{ 1.0, 2.0, 4.0, null });
    try expectColumn(f64, &batch, "v_max", &.{ 1.0, 2.0, 6.0, null });
    try expectColumn(i64, &batch, "i_count", &.{ 2, 2, 2, 0 });
    try expectColumn(i64, &batch, "count_all", &.{ 2, 2, 2, 1 });
    try expectColumn(bool, &batch, "b_any", &.{ true, true, true, null });
    try expectColumn(bool, &batch, "b_all", &.{ true, false, true, null });
    try expectColumn(i64, &batch, "i_count_distinct", &.{ 2, 2, 2, 0 });
    const mm = batch.columnByName("i_min_max").?.data();
    const mins = try common.collect(i64, allocator, .{ .array = mm.children[0] });
    defer allocator.free(mins);
    try common.expectRows(i64, &.{ 1, 2, 4, null }, mins);
    const maxs = try common.collect(i64, allocator, .{ .array = mm.children[1] });
    defer allocator.free(maxs);
    try common.expectRows(i64, &.{ 3, 5, 6, null }, maxs);
    const distinct = batch.columnByName("i_distinct").?.data();
    try std.testing.expect(distinct.data_type == .list);
    const offsets = try distinct.buffers[1].typedSlice(i32);
    try std.testing.expectEqualSlices(i32, &.{ 0, 2, 4, 6, 6 }, offsets[0..5]);
    const items = try common.collect(i64, allocator, .{ .array = distinct.children[0] });
    defer allocator.free(items);
    try common.expectRows(i64, &.{ 1, 3, 2, 5, 4, 6 }, items);

    // Two keys (int64 and string): (1, a), (2, b), (null, x), (2, c), (3, z).
    var two = try hash.groupBy(&ctx, &.{ .{ .name = "k", .array = k }, .{ .name = "k2", .array = k2 } }, &.{.{ .function = "hash_sum", .input = i, .name = "i_sum" }});
    defer two.deinit();
    try expectColumn(i64, &two, "k", &.{ 1, 2, null, 2, 3 });
    try expectColumn(i64, &two, "i_sum", &.{ 4, 2, 10, 5, null });
    const names = try common.collectStrings(allocator, .{ .array = two.columnByName("k2").?.* });
    defer allocator.free(names);
    try common.expectStrings(&.{ "a", "b", "x", "c", "z" }, names);

    // Nulls are not distinct values; float keys: NaN is one group, -0.0 another than 0.0.
    var dk = try common.makeArray(i64, allocator, &.{ 1, 1, 1, 2 });
    defer dk.release();
    var dv = try common.makeArray(i64, allocator, &.{ null, 1, null, null });
    defer dv.release();
    var dist = try hash.groupBy(&ctx, &.{.{ .name = "k", .array = dk }}, &.{
        .{ .function = "hash_count_distinct", .input = dv, .name = "n" },
        .{ .function = "hash_sum", .input = dv, .name = "s" },
        .{ .function = "hash_distinct", .input = dv, .name = "d" },
    });
    defer dist.deinit();
    try expectColumn(i64, &dist, "n", &.{ 1, 0 });
    try expectColumn(i64, &dist, "s", &.{ 1, null });
    try std.testing.expectEqualSlices(i32, &.{ 0, 1, 1 }, (try dist.columnByName("d").?.data().buffers[1].typedSlice(i32))[0..3]);
    var fk = try common.makeArray(f64, allocator, &.{ 0.0, -0.0, std.math.nan(f64), -std.math.nan(f64) });
    defer fk.release();
    var fv = try common.makeArray(i64, allocator, &.{ 1, 2, 3, 4 });
    defer fv.release();
    var floats = try hash.groupBy(&ctx, &.{.{ .name = "k", .array = fk }}, &.{.{ .function = "hash_sum", .input = fv, .name = "s" }});
    defer floats.deinit();
    try expectColumn(i64, &floats, "s", &.{ 1, 2, 7 });

    // The kernels on their own: values and group ids in, one row per group out.
    var ids = try groupIds(&.{k});
    defer ids.release();
    try common.expectInvoke(i64, &ctx, .hash_aggregate, "hash_sum", &.{ .{ .array = i }, .{ .array = ids } }, &.{ 4, 7, 10, null });
    try common.expectInvoke(i64, &ctx, .hash_aggregate, "hash_count_all", &.{.{ .array = ids }}, &.{ 2, 2, 2, 1 });
    try std.testing.expect((try registry.resolveResultType("hash_mean", .hash_aggregate, &.{ .{ .array = i }, .{ .array = ids } }, none)) == .double);
    try std.testing.expect((try registry.resolveResultType("hash_distinct", .hash_aggregate, &.{ .{ .array = i }, .{ .array = ids } }, none)) == .list);

    // Integer sums follow overflow_mode like sum (Arrow wraps).
    var big = try common.makeArray(i64, allocator, &.{ 1 << 62, 1 << 62 });
    defer big.release();
    var same = try common.makeArray(u32, allocator, &.{ 0, 0 });
    defer same.release();
    try std.testing.expectError(error.Overflow, ctx.invoke("hash_sum", .hash_aggregate, &.{ .{ .array = big }, .{ .array = same } }, none));
    var wrap_ctx = ExecContext.initWithConfig(allocator, &registry, .{ .overflow_mode = .wrapping });
    try common.expectInvoke(i64, &wrap_ctx, .hash_aggregate, "hash_sum", &.{ .{ .array = big }, .{ .array = same } }, &.{std.math.minInt(i64)});
}

const Ref = struct { sum: i128 = 0, count: i64 = 0, rows: i64 = 0, min: i64 = std.math.maxInt(i64), max: i64 = std.math.minInt(i64) };

test "hash aggregate oracle matches a per-group reference on random sliced keys and values with nulls" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    var prng = std.Random.DefaultPrng.init(0x6A66);
    const rng = prng.random();
    const n = 1000;

    var keys: [n + 11]?i32 = undefined;
    for (&keys) |*x| x.* = if (rng.uintLessThan(u8, 20) == 0) null else rng.intRangeAtMost(i32, -30, 30);
    var vals: [n + 5]?i64 = undefined;
    for (&vals) |*x| x.* = if (rng.uintLessThan(u8, 6) == 0) null else rng.intRangeAtMost(i64, -1_000_000, 1_000_000);
    var key_full = try common.makeArray(i32, allocator, &keys);
    defer key_full.release();
    var key_arr = try key_full.slice(11, n);
    defer key_arr.release();
    var val_full = try common.makeArray(i64, allocator, &vals);
    defer val_full.release();
    var val_arr = try val_full.slice(5, n);
    defer val_arr.release();

    // Reference: groups in first-seen order of the key (null is a key of its own).
    var order: std.ArrayList(?i32) = .empty;
    defer order.deinit(allocator);
    var refs: std.ArrayList(Ref) = .empty;
    defer refs.deinit(allocator);
    for (keys[11..][0..n], vals[5..][0..n]) |key, val| {
        const g = for (order.items, 0..) |o, gi| {
            if (o == key) break gi;
        } else blk: {
            try order.append(allocator, key);
            try refs.append(allocator, .{});
            break :blk order.items.len - 1;
        };
        const r = &refs.items[g];
        r.rows += 1;
        if (val) |x| {
            r.sum += x;
            r.count += 1;
            r.min = @min(r.min, x);
            r.max = @max(r.max, x);
        }
    }
    var batch = try hash.groupBy(&ctx, &.{.{ .name = "k", .array = key_arr }}, &.{
        .{ .function = "hash_sum", .input = val_arr, .name = "sum" },
        .{ .function = "hash_mean", .input = val_arr, .name = "mean" },
        .{ .function = "hash_min", .input = val_arr, .name = "min" },
        .{ .function = "hash_max", .input = val_arr, .name = "max" },
        .{ .function = "hash_count", .input = val_arr, .name = "count" },
        .{ .function = "hash_count_all", .name = "rows" },
        .{ .function = "hash_count_distinct", .input = key_arr, .name = "one" },
    });
    defer batch.deinit();
    const g = refs.items.len;
    try std.testing.expectEqual(g, batch.numRows());
    const exp_sum = try allocator.alloc(?i64, g);
    defer allocator.free(exp_sum);
    const exp_mean = try allocator.alloc(?f64, g);
    defer allocator.free(exp_mean);
    const exp_min = try allocator.alloc(?i64, g);
    defer allocator.free(exp_min);
    const exp_max = try allocator.alloc(?i64, g);
    defer allocator.free(exp_max);
    const exp_count = try allocator.alloc(?i64, g);
    defer allocator.free(exp_count);
    const exp_rows = try allocator.alloc(?i64, g);
    defer allocator.free(exp_rows);
    const exp_one = try allocator.alloc(?i64, g);
    defer allocator.free(exp_one);
    for (refs.items, order.items, 0..) |r, key, gi| {
        const valid = r.count > 0;
        exp_sum[gi] = if (valid) @intCast(r.sum) else null;
        exp_mean[gi] = if (valid) @as(f64, @floatFromInt(r.sum)) / @as(f64, @floatFromInt(r.count)) else null;
        exp_min[gi] = if (valid) r.min else null;
        exp_max[gi] = if (valid) r.max else null;
        exp_count[gi] = r.count;
        exp_rows[gi] = r.rows;
        exp_one[gi] = if (key == null) 0 else 1;
    }
    try expectColumn(i32, &batch, "k", order.items);
    try expectColumn(i64, &batch, "sum", exp_sum);
    try expectColumn(f64, &batch, "mean", exp_mean);
    try expectColumn(i64, &batch, "min", exp_min);
    try expectColumn(i64, &batch, "max", exp_max);
    try expectColumn(i64, &batch, "count", exp_count);
    try expectColumn(i64, &batch, "rows", exp_rows);
    try expectColumn(i64, &batch, "one", exp_one);

    // Two keys of string layouts give the groups of their pairs.
    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    inline for (.{ .string, .large_string, .string_view }) |tag| {
        var s = try common.randomStrings(tag, allocator, rng, n, 0, 10);
        defer s.release();
        var two = try hash.groupBy(&ctx, &.{ .{ .name = "s", .array = s }, .{ .name = "k", .array = key_arr } }, &.{.{ .function = "hash_count_all", .name = "rows" }});
        defer two.deinit();
        var pairs: std.ArrayList(struct { s: ?[]const u8, k: ?i32 }) = .empty;
        defer pairs.deinit(allocator);
        for (0..n) |row| {
            const sv = try common.stringAt(s.data(), row);
            const kv = keys[11 + row];
            for (pairs.items) |p| {
                const same_s = if (p.s == null or sv == null) p.s == null and sv == null else std.mem.eql(u8, p.s.?, sv.?);
                if (same_s and p.k == kv) break;
            } else try pairs.append(allocator, .{ .s = sv, .k = kv });
        }
        try std.testing.expectEqual(pairs.items.len, two.numRows());
        const got_k = try column(i32, &two, "k");
        defer allocator.free(got_k);
        for (pairs.items, got_k) |p, gk| try std.testing.expectEqual(p.k, gk);
        var total: i64 = 0;
        const rows = try column(i64, &two, "rows");
        defer allocator.free(rows);
        for (rows) |r| total += r.?;
        try std.testing.expectEqual(@as(i64, n), total);
    }
}

test "hash aggregate on chunked inputs with misaligned chunks and the update / merge lifecycle equal the array result" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    var prng = std.Random.DefaultPrng.init(0x6A67);
    const rng = prng.random();
    var keys = try common.randomArray(i16, allocator, rng, 600, 0, 5, .{ .small = true });
    defer keys.release();
    var vals = try common.randomArray(i32, allocator, rng, 600, 3, 10, .{});
    defer vals.release();
    var ids = try groupIds(&.{keys});
    defer ids.release();
    var cv = try common.chunkedAt(allocator, vals, &.{ 10, 300 });
    defer cv.release();
    var ci = try common.chunkedAt(allocator, ids, &.{ 128, 512 });
    defer ci.release();
    inline for (.{ "hash_sum", "hash_min_max", "hash_count", "hash_count_distinct", "hash_distinct" }) |name| {
        var whole = try ctx.invoke(name, .hash_aggregate, &.{ .{ .array = vals }, .{ .array = ids } }, none);
        defer whole.release();
        var chunked = try ctx.invoke(name, .hash_aggregate, &.{ .{ .chunked = cv }, .{ .chunked = ci } }, none);
        defer chunked.release();
        try expectSameArray(whole.array, chunked.array);

        // Lifecycle: a Grouper and the kernel state fed batch by batch, then merged with a second
        // state over the same group ids.
        var grouper = try hash.Grouper.init(allocator, &.{.{ .int16 = {} }});
        defer grouper.deinit();
        var s1 = try ctx.beginAggregate(name, &.{ .{ .array = vals }, .{ .array = ids } }, none);
        defer s1.deinit();
        var s2 = try ctx.beginAggregate(name, &.{ .{ .array = vals }, .{ .array = ids } }, none);
        defer s2.deinit();
        for ([_][2]usize{ .{ 0, 250 }, .{ 250, 100 }, .{ 350, 250 } }, 0..) |part, p| {
            var kp = try keys.slice(part[0], part[1]);
            defer kp.release();
            var vp = try vals.slice(part[0], part[1]);
            defer vp.release();
            var gp = try grouper.consume(&.{kp});
            defer gp.release();
            try (if (p == 1) &s2 else &s1).update(&.{ .{ .array = vp }, .{ .array = gp } });
        }
        try s1.merge(&s2);
        var merged = try s1.finalize();
        defer merged.release();
        // A merged distinct list holds the same values, in the order the merge met them.
        if (comptime std.mem.eql(u8, name, "hash_distinct")) {
            try sortLists(merged.array);
            try sortLists(whole.array);
        }
        try expectSameArray(whole.array, merged.array);
    }
}

/// Sort the int32 values of each row of a list array in place (test-owned buffers).
fn sortLists(arr: ArrayRef) !void {
    const data = arr.data();
    const offsets = try data.buffers[1].typedSlice(i32);
    const child = data.children[0].data();
    const vals: []i32 = @constCast(try child.buffers[1].typedSlice(i32))[child.offset..];
    for (0..data.length) |i| std.mem.sort(i32, vals[@intCast(offsets[i])..@intCast(offsets[i + 1])], {}, std.sort.asc(i32));
}

/// Equal rows (validateFull, same type and length, same values and nulls via a take-free compare).
fn expectSameArray(a: ArrayRef, b: ArrayRef) !void {
    try a.data().validateFull();
    try b.data().validateFull();
    try std.testing.expect(a.data().data_type.eql(b.data().data_type));
    try std.testing.expectEqual(a.data().length, b.data().length);
    switch (a.data().data_type) {
        .list => {
            const ao = try a.data().buffers[1].typedSlice(i32);
            const bo = try b.data().buffers[1].typedSlice(i32);
            try std.testing.expectEqualSlices(i32, ao[0 .. a.data().length + 1], bo[0 .. b.data().length + 1]);
            try expectSameArray(a.data().children[0], b.data().children[0]);
        },
        .struct_ => for (a.data().children, b.data().children) |x, y| try expectSameArray(x, y),
        inline .int64, .int32 => |_, tag| {
            const T = if (tag == .int64) i64 else i32;
            const x = try common.collect(T, std.testing.allocator, .{ .array = a });
            defer std.testing.allocator.free(x);
            const y = try common.collect(T, std.testing.allocator, .{ .array = b });
            defer std.testing.allocator.free(y);
            try common.expectRows(T, x, y);
        },
        else => return error.TestUnexpectedResult,
    }
}

test "hash aggregate on empty and all-null inputs" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    var empty_k = try common.makeArray(i32, allocator, &.{});
    defer empty_k.release();
    var empty_v = try common.makeArray(f32, allocator, &.{});
    defer empty_v.release();
    var e = try hash.groupBy(&ctx, &.{.{ .name = "k", .array = empty_k }}, &.{
        .{ .function = "hash_mean", .input = empty_v, .name = "m" },
        .{ .function = "hash_distinct", .input = empty_v, .name = "d" },
    });
    defer e.deinit();
    try std.testing.expectEqual(@as(usize, 0), e.numRows());
    for (e.columns) |c| try c.data().validateFull();

    var k = try common.makeArray(u8, allocator, &.{ 7, null, 7 });
    defer k.release();
    var v = try common.makeArray(u64, allocator, &.{ null, null, null });
    defer v.release();
    var b = try common.makeArray(bool, allocator, &.{ null, null, null });
    defer b.release();
    var nulls = try hash.groupBy(&ctx, &.{.{ .name = "k", .array = k }}, &.{
        .{ .function = "hash_sum", .input = v, .name = "s" },
        .{ .function = "hash_min_max", .input = v, .name = "mm" },
        .{ .function = "hash_count", .input = v, .name = "c" },
        .{ .function = "hash_all", .input = b, .name = "all" },
        .{ .function = "hash_count_distinct", .input = v, .name = "n" },
    });
    defer nulls.deinit();
    for (nulls.columns) |c| try c.data().validateFull();
    try expectColumn(u8, &nulls, "k", &.{ 7, null });
    try expectColumn(u64, &nulls, "s", &.{ null, null });
    try expectColumn(i64, &nulls, "c", &.{ 0, 0 });
    try expectColumn(bool, &nulls, "all", &.{ null, null });
    try expectColumn(i64, &nulls, "n", &.{ 0, 0 });
    try std.testing.expectEqual(@as(?usize, 2), nulls.columnByName("mm").?.data().children[0].data().null_count);
}

test "hash aggregate errors: group ids, key types, lengths, overflow and malformed input" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    var v = try common.makeArray(i64, allocator, &.{ 1, 2 });
    defer v.release();
    var null_ids = try common.makeArray(u32, allocator, &.{ 0, null });
    defer null_ids.release();
    try std.testing.expectError(error.InvalidInput, ctx.invoke("hash_sum", .hash_aggregate, &.{ .{ .array = v }, .{ .array = null_ids } }, none));
    // A one-shot call bounds the ids by its rows (dense ids from a Grouper always are).
    var far = try common.makeArray(u32, allocator, &.{ 0, 4_000_000_000 });
    defer far.release();
    try std.testing.expectError(error.InvalidInput, ctx.invoke("hash_sum", .hash_aggregate, &.{ .{ .array = v }, .{ .array = far } }, none));
    try std.testing.expectError(error.InvalidInput, ctx.invoke("hash_count_all", .hash_aggregate, &.{.{ .array = far }}, none));
    try std.testing.expectError(error.InvalidInput, ctx.invoke("hash_sum", .hash_aggregate, &.{ .{ .array = v }, .{ .scalar = common.scalarOf(u32, 0) } }, none));
    var i32_ids = try common.makeArray(i32, allocator, &.{ 0, 1 });
    defer i32_ids.release();
    try std.testing.expectError(error.NoMatchingKernel, ctx.invoke("hash_sum", .hash_aggregate, &.{ .{ .array = v }, .{ .array = i32_ids } }, none));
    var short_ids = try common.makeArray(u32, allocator, &.{0});
    defer short_ids.release();
    try std.testing.expectError(error.InvalidInput, ctx.invoke("hash_sum", .hash_aggregate, &.{ .{ .array = v }, .{ .array = short_ids } }, none));
    try std.testing.expectError(error.InvalidOptions, ctx.invoke("hash_sum", .hash_aggregate, &.{ .{ .array = v }, .{ .array = short_ids } }, .{ .day_of_week = .{} }));
    try std.testing.expectError(error.FunctionNotFound, ctx.invoke("hash_sum", .aggregate, &.{ .{ .array = v }, .{ .array = short_ids } }, none));

    try std.testing.expectError(error.UnsupportedType, hash.Grouper.init(allocator, &.{.{ .null = {} }}));
    try std.testing.expectError(error.InvalidArity, hash.Grouper.init(allocator, &.{}));
    var g = try hash.Grouper.init(allocator, &.{.{ .int64 = {} }});
    defer g.deinit();
    var w = try common.makeArray(i32, allocator, &.{1});
    defer w.release();
    try std.testing.expectError(error.InvalidInput, g.consume(&.{w}));
    try std.testing.expectError(error.InvalidArity, g.consume(&.{ v, v }));
    var k = try common.makeArray(i64, allocator, &.{1});
    defer k.release();
    try std.testing.expectError(error.InvalidInput, hash.groupBy(&ctx, &.{.{ .name = "k", .array = k }}, &.{.{ .function = "hash_sum", .input = v, .name = "s" }}));
    try std.testing.expectError(error.InvalidArity, hash.groupBy(&ctx, &.{}, &.{}));
    const short = [_]i64{1};
    const bufs = [_]common.SharedBuffer{ .empty, .fromSlice(std.mem.sliceAsBytes(&short)) };
    var malformed = try ArrayRef.fromBorrowed(allocator, .{ .data_type = .{ .int64 = {} }, .length = 2, .null_count = 0, .buffers = &bufs });
    defer malformed.release();
    var ids = try common.makeArray(u32, allocator, &.{ 0, 1 });
    defer ids.release();
    try std.testing.expectError(error.InvalidInput, ctx.invoke("hash_sum", .hash_aggregate, &.{ .{ .array = malformed }, .{ .array = ids } }, none));
    try std.testing.expectError(error.InvalidInput, g.consume(&.{malformed}));
}

fn aggOpts(skip_nulls: bool, min_count: u32) common.Options {
    return .{ .scalar_aggregate = .{ .skip_nulls = skip_nulls, .min_count = min_count } };
}

fn countOpts(mode: common.core.CountMode) common.Options {
    return .{ .count = .{ .mode = mode } };
}

test "hash aggregates honour ScalarAggregateOptions and CountOptions (pyarrow 26 group_by results)" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);

    // k = [1, 1, 2, 2, 3], v = [1, null, 3, 4, null], b = [true, null, false, true, null].
    var k = try common.makeArray(i64, allocator, &.{ 1, 1, 2, 2, 3 });
    defer k.release();
    var v = try common.makeArray(i32, allocator, &.{ 1, null, 3, 4, null });
    defer v.release();
    var b = try common.makeArray(bool, allocator, &.{ true, null, false, true, null });
    defer b.release();
    var batch = try hash.groupBy(&ctx, &.{.{ .name = "k", .array = k }}, &.{
        .{ .function = "hash_sum", .input = v, .name = "sum_keep", .options = aggOpts(false, 1) },
        .{ .function = "hash_sum", .input = v, .name = "sum_mc2", .options = aggOpts(true, 2) },
        .{ .function = "hash_sum", .input = v, .name = "sum_mc0", .options = aggOpts(true, 0) },
        .{ .function = "hash_mean", .input = v, .name = "mean_mc0", .options = aggOpts(true, 0) },
        .{ .function = "hash_min", .input = v, .name = "min_keep", .options = aggOpts(false, 1) },
        .{ .function = "hash_min", .input = v, .name = "min_mc0", .options = aggOpts(true, 0) },
        .{ .function = "hash_max", .input = v, .name = "max_mc2", .options = aggOpts(true, 2) },
        .{ .function = "hash_min_max", .input = v, .name = "mm_keep", .options = aggOpts(false, 1) },
        .{ .function = "hash_count", .input = v, .name = "count_null", .options = countOpts(.only_null) },
        .{ .function = "hash_count", .input = v, .name = "count_all", .options = countOpts(.all) },
        .{ .function = "hash_count", .input = v, .name = "count_valid", .options = countOpts(.only_valid) },
        .{ .function = "hash_any", .input = b, .name = "any_keep", .options = aggOpts(false, 1) },
        .{ .function = "hash_all", .input = b, .name = "all_keep", .options = aggOpts(false, 1) },
        .{ .function = "hash_any", .input = b, .name = "any_mc0", .options = aggOpts(true, 0) },
        .{ .function = "hash_all", .input = b, .name = "all_mc2", .options = aggOpts(true, 2) },
        .{ .function = "hash_count_distinct", .input = v, .name = "cd_all", .options = countOpts(.all) },
        .{ .function = "hash_count_distinct", .input = v, .name = "cd_null", .options = countOpts(.only_null) },
        .{ .function = "hash_distinct", .input = v, .name = "d_all", .options = countOpts(.all) },
        .{ .function = "hash_distinct", .input = v, .name = "d_null", .options = countOpts(.only_null) },
    });
    defer batch.deinit();
    for (batch.columns) |c| try c.data().validateFull();
    try expectColumn(i64, &batch, "k", &.{ 1, 2, 3 });
    try expectColumn(i64, &batch, "sum_keep", &.{ null, 7, null });
    try expectColumn(i64, &batch, "sum_mc2", &.{ null, 7, null });
    try expectColumn(i64, &batch, "sum_mc0", &.{ 1, 7, 0 });
    try expectColumn(f64, &batch, "mean_mc0", &.{ 1.0, 3.5, std.math.nan(f64) });
    try expectColumn(i32, &batch, "min_keep", &.{ null, 3, null });
    try expectColumn(i32, &batch, "min_mc0", &.{ 1, 3, null });
    try expectColumn(i32, &batch, "max_mc2", &.{ null, 4, null });
    const mm = batch.columnByName("mm_keep").?.data();
    try std.testing.expectEqual(@as(?usize, 0), mm.null_count);
    for (mm.children, [2][]const ?i32{ &.{ null, 3, null }, &.{ null, 4, null } }) |child, want| {
        const got = try common.collect(i32, allocator, .{ .array = child });
        defer allocator.free(got);
        try common.expectRows(i32, want, got);
    }
    try expectColumn(i64, &batch, "count_null", &.{ 1, 0, 1 });
    try expectColumn(i64, &batch, "count_all", &.{ 2, 2, 1 });
    try expectColumn(i64, &batch, "count_valid", &.{ 1, 2, 0 });
    try expectColumn(bool, &batch, "any_keep", &.{ true, true, null });
    try expectColumn(bool, &batch, "all_keep", &.{ null, false, null });
    try expectColumn(bool, &batch, "any_mc0", &.{ true, true, false });
    try expectColumn(bool, &batch, "all_mc2", &.{ null, false, null });
    try expectColumn(i64, &batch, "cd_all", &.{ 2, 2, 1 });
    try expectColumn(i64, &batch, "cd_null", &.{ 1, 0, 1 });
    // hash_distinct with all: [[1, null], [3, 4], [null]]; with only_null: [[null], [], [null]].
    const d_all = batch.columnByName("d_all").?.data();
    try std.testing.expectEqualSlices(i32, &.{ 0, 2, 4, 5 }, (try d_all.buffers[1].typedSlice(i32))[0..4]);
    const all_items = try common.collect(i32, allocator, .{ .array = d_all.children[0] });
    defer allocator.free(all_items);
    try common.expectRows(i32, &.{ 1, null, 3, 4, null }, all_items);
    const d_null = batch.columnByName("d_null").?.data();
    try std.testing.expectEqualSlices(i32, &.{ 0, 1, 1, 2 }, (try d_null.buffers[1].typedSlice(i32))[0..4]);

    // The same options through a lifecycle with batches and a merge.
    var ids = try groupIds(&.{k});
    defer ids.release();
    var s1 = try ctx.beginAggregate("hash_sum", &.{ .{ .array = v }, .{ .array = ids } }, aggOpts(false, 0));
    defer s1.deinit();
    var s2 = try ctx.beginAggregate("hash_sum", &.{ .{ .array = v }, .{ .array = ids } }, aggOpts(false, 0));
    defer s2.deinit();
    var v_head = try v.slice(0, 2);
    defer v_head.release();
    var ids_head = try ids.slice(0, 2);
    defer ids_head.release();
    var v_tail = try v.slice(2, 3);
    defer v_tail.release();
    var ids_tail = try ids.slice(2, 3);
    defer ids_tail.release();
    try s1.update(&.{ .{ .array = v_head }, .{ .array = ids_head } });
    try s2.update(&.{ .{ .array = v_tail }, .{ .array = ids_tail } });
    try s1.merge(&s2);
    var merged = try s1.finalize();
    defer merged.release();
    const sums = try common.collect(i64, allocator, merged);
    defer allocator.free(sums);
    try common.expectRows(i64, &.{ null, 7, null }, sums);

    // Options are validated: wrong option kinds are InvalidOptions; hash_count_all takes none.
    try std.testing.expectError(error.InvalidOptions, ctx.invoke("hash_sum", .hash_aggregate, &.{ .{ .array = v }, .{ .array = ids } }, countOpts(.all)));
    try std.testing.expectError(error.InvalidOptions, ctx.invoke("hash_count", .hash_aggregate, &.{ .{ .array = v }, .{ .array = ids } }, aggOpts(true, 1)));
    try std.testing.expectError(error.InvalidOptions, ctx.invoke("hash_count_all", .hash_aggregate, &.{.{ .array = ids }}, countOpts(.all)));
}

test "hash_sum, hash_mean, hash_min and hash_max on decimals (pyarrow 26 group_by results)" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);

    // k = [1, 1, 2], v = decimal128(10, 2) [1.50, 2.25, null]: sum decimal128(38, 2) [3.75, null],
    // mean decimal128(10, 2) [1.88, null] (half away from zero), min [1.50, null], max [2.25, null].
    const dec = DataType{ .decimal128 = .{ .precision = 10, .scale = 2 } };
    var k = try common.makeArray(i64, allocator, &.{ 1, 1, 2 });
    defer k.release();
    var v = try common.makeTyped(i128, dec, allocator, &.{ 150, 225, null });
    defer v.release();
    var batch = try hash.groupBy(&ctx, &.{.{ .name = "k", .array = k }}, &.{
        .{ .function = "hash_sum", .input = v, .name = "sum" },
        .{ .function = "hash_mean", .input = v, .name = "mean" },
        .{ .function = "hash_min", .input = v, .name = "min" },
        .{ .function = "hash_max", .input = v, .name = "max" },
        .{ .function = "hash_sum", .input = v, .name = "sum_mc0", .options = aggOpts(true, 0) },
    });
    defer batch.deinit();
    for (batch.columns) |c| try c.data().validateFull();
    try std.testing.expect(batch.columnByName("sum").?.data().data_type.eql(.{ .decimal128 = .{ .precision = 38, .scale = 2 } }));
    try std.testing.expect(batch.columnByName("mean").?.data().data_type.eql(dec));
    inline for (.{ .{ "sum", [2]?i128{ 375, null } }, .{ "mean", [2]?i128{ 188, null } }, .{ "min", [2]?i128{ 150, null } }, .{ "max", [2]?i128{ 225, null } }, .{ "sum_mc0", [2]?i128{ 375, 0 } } }) |c| {
        const data = batch.columnByName(c[0]).?.data();
        const vals = try (common.array_mod.PrimitiveArray(i128){ .data = data }).values();
        for (c[1], 0..) |want, i| {
            if (want) |w| try std.testing.expectEqual(w, vals[i]) else try std.testing.expect(data.isNull(i));
        }
    }
    // decimal256 and the result type of the kernel on its own.
    var w = try common.makeTyped(i256, .{ .decimal256 = .{ .precision = 50, .scale = 1 } }, allocator, &.{ 1 << 160, 5, -5 });
    defer w.release();
    var ids = try groupIds(&.{k});
    defer ids.release();
    try std.testing.expect((try registry.resolveResultType("hash_sum", .hash_aggregate, &.{ .{ .array = w }, .{ .array = ids } }, none)).eql(.{ .decimal256 = .{ .precision = 76, .scale = 1 } }));
    var sums = try ctx.invoke("hash_sum", .hash_aggregate, &.{ .{ .array = w }, .{ .array = ids } }, none);
    defer sums.release();
    const sv = try (common.array_mod.PrimitiveArray(i256){ .data = sums.array.data() }).values();
    try std.testing.expectEqual(@as(i256, (1 << 160) + 5), sv[0]);
    try std.testing.expectEqual(@as(i256, -5), sv[1]);
    // A sum past precision 38 is Overflow when checked.
    const nines: i128 = 99999999999999999999999999999999999999;
    var over = try common.makeTyped(i128, .{ .decimal128 = .{ .precision = 38, .scale = 0 } }, allocator, &.{ nines, nines, 1 });
    defer over.release();
    try std.testing.expectError(error.Overflow, ctx.invoke("hash_sum", .hash_aggregate, &.{ .{ .array = over }, .{ .array = ids } }, none));
    var small = try common.makeTyped(i32, .{ .decimal32 = .{ .precision = 5, .scale = 2 } }, allocator, &.{ 1, 2, 3 });
    defer small.release();
    try std.testing.expectError(error.NoMatchingKernel, ctx.invoke("hash_sum", .hash_aggregate, &.{ .{ .array = small }, .{ .array = ids } }, none));
}
