//! Record batch and table datums: constructors, accessors and the functions that take them
//! (take, filter, drop_null, slice, column selection, sort_indices, select_k_unstable).
//! Expected values are pyarrow 26 results.
const std = @import("std");
const common = @import("test_common.zig");

const ArrayRef = common.ArrayRef;
const Datum = common.Datum;
const ExecContext = common.ExecContext;
const Options = common.Options;
const core = common.core;
const RecordBatch = @import("../record_batch.zig").RecordBatch;
const Table = @import("../table.zig").Table;
const SchemaRef = @import("../schema.zig").SchemaRef;
const ChunkedArray = common.ChunkedArray;

const nan = std.math.nan(f64);
const int64_type = common.DataType{ .int64 = {} };
const string_type = common.DataType{ .string = {} };
const fields = [_]common.datatype.Field{
    .{ .name = "a", .data_type = &int64_type },
    .{ .name = "b", .data_type = &string_type },
};

/// {'a': [3, None, 1, 2], 'b': ['x', 'y', None, 'w']} as a record batch over `a` / `b`.
fn batchOf(allocator: std.mem.Allocator, a: ArrayRef, b: ArrayRef) !Datum {
    var batch = try RecordBatch.initBorrowed(allocator, .{ .fields = &fields }, &.{ a, b });
    errdefer batch.deinit();
    return Datum.fromRecordBatch(allocator, batch);
}

/// The same rows as a table whose columns are chunked at `cuts` (different cuts per column).
fn tableOf(allocator: std.mem.Allocator, a: ArrayRef, b: ArrayRef, cuts_a: []const usize, cuts_b: []const usize) !Datum {
    var ca = try common.chunkedAt(allocator, a, cuts_a);
    defer ca.release();
    var cb = try common.chunkedAt(allocator, b, cuts_b);
    defer cb.release();
    var table = try Table.init(allocator, try SchemaRef.fromBorrowed(allocator, .{ .fields = &fields }), &.{ ca, cb });
    errdefer table.deinit();
    return Datum.fromTable(allocator, table);
}

/// Column `i` of a record batch or table datum as int64 rows (caller frees).
fn ints(allocator: std.mem.Allocator, d: Datum, i: usize) ![]?i64 {
    var col = try core.datumStructField(d, i);
    defer col.release();
    return common.collect(i64, allocator, col);
}

fn expectTabular(d: Datum, want_a: []const ?i64, want_b: []const ?[]const u8) !void {
    const allocator = std.testing.allocator;
    try std.testing.expectEqual(want_a.len, d.length());
    switch (d) {
        .record_batch => |b| {
            try core.validateRecordBatch(b.get());
            for (b.get().columns) |c| try c.data().validateFull();
        },
        .table => |t| {
            try core.validateTable(t.get());
            for (t.get().columns) |c| for (c.chunks()) |part| try part.data().validateFull();
        },
        else => return error.TestUnexpectedResult,
    }
    const got_a = try ints(allocator, d, 0);
    defer allocator.free(got_a);
    try common.expectRows(i64, want_a, got_a);
    var col_b = try core.datumStructField(d, 1);
    defer col_b.release();
    const got_b = try common.collectStrings(allocator, col_b);
    defer allocator.free(got_b);
    try common.expectStrings(want_b, got_b);
}

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

test "tabular datum constructors, kind, length, dataType, retain and release; malformed batches and tables are rejected" {
    const allocator = std.testing.allocator;
    var a = try common.makeArray(i64, allocator, &.{ 3, null, 1, 2 });
    defer a.release();
    var b = try common.makeStrings(.string, allocator, &.{ "x", "y", null, "w" });
    defer b.release();

    var rb = try batchOf(allocator, a, b);
    defer rb.release();
    try std.testing.expectEqual(core.DatumKind.record_batch, rb.kind());
    try std.testing.expect(rb.isTabular() and !rb.isArray() and rb.asTable() == null);
    try std.testing.expectEqual(@as(usize, 4), rb.length());
    try std.testing.expectEqual(@as(usize, 2), rb.asRecordBatch().?.numColumns());
    const dt = rb.dataType();
    try std.testing.expect(dt == .struct_ and dt.struct_.fields.len == 2 and std.mem.eql(u8, dt.struct_.fields[1].name, "b"));
    var shared = rb.retain();
    shared.release();
    try std.testing.expectEqual(@as(usize, 4), rb.length());

    var t = try tableOf(allocator, a, b, &.{2}, &.{ 1, 3 });
    defer t.release();
    try std.testing.expectEqual(core.DatumKind.table, t.kind());
    try std.testing.expectEqual(@as(usize, 4), t.length());
    try std.testing.expect(t.asTable() != null and t.asRecordBatch() == null);
    try std.testing.expectEqual(@as(usize, 1), (Datum{ .scalar = common.Scalar.init(int64_type, .{ .i64 = 1 }) }).length());

    // datumStructField gives a column: an array of a record batch, a chunked array of a table.
    var col = try core.datumStructField(t, 1);
    defer col.release();
    try std.testing.expect(col == .chunked and col.chunked.numChunks() == 3);
    try std.testing.expectError(error.InvalidInput, core.datumStructField(rb, 2));

    // A batch whose column is shorter than num_rows, of another type, or one column short.
    var short = try a.slice(0, 3);
    defer short.release();
    var schema_ref = try SchemaRef.fromBorrowed(allocator, .{ .fields = &fields });
    defer schema_ref.release();
    var cols_bad_len = [_]ArrayRef{ short, b };
    try std.testing.expectError(error.InvalidInput, Datum.fromRecordBatch(allocator, .{ .allocator = allocator, .schema_ref = schema_ref, .columns = &cols_bad_len, .num_rows = 4 }));
    var cols_bad_type = [_]ArrayRef{ b, b };
    try std.testing.expectError(error.InvalidInput, Datum.fromRecordBatch(allocator, .{ .allocator = allocator, .schema_ref = schema_ref, .columns = &cols_bad_type, .num_rows = 4 }));
    var cols_one = [_]ArrayRef{a};
    try std.testing.expectError(error.InvalidInput, Datum.fromRecordBatch(allocator, .{ .allocator = allocator, .schema_ref = schema_ref, .columns = &cols_one, .num_rows = 4 }));
    // A column whose buffers are too short for its length fails validateLayout.
    var broken_data = a.data().*;
    broken_data.length = 1000;
    var broken = try common.array_mod.ArrayRef.fromBorrowed(allocator, broken_data);
    defer broken.release();
    var cols_broken = [_]ArrayRef{ broken, b };
    try std.testing.expectError(error.InvalidInput, Datum.fromRecordBatch(allocator, .{ .allocator = allocator, .schema_ref = schema_ref, .columns = &cols_broken, .num_rows = 1000 }));
    // A table whose columns disagree with num_rows.
    var ca = try ChunkedArray.fromSingle(allocator, a);
    defer ca.release();
    var cs = try ChunkedArray.fromSingle(allocator, short);
    defer cs.release();
    var tcols = [_]ChunkedArray{ ca, cs };
    try std.testing.expectError(error.InvalidInput, Datum.fromTable(allocator, .{ .allocator = allocator, .schema_ref = schema_ref, .columns = &tcols, .num_rows = 4 }));
}

test "take, filter, drop_null, slice and column selection on record batches and tables (pyarrow 26 results)" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    // A sliced source: the rows are [3, None, 1, 2] / ['x', 'y', None, 'w'].
    var a_full = try common.makeArray(i64, allocator, &.{ 9, 3, null, 1, 2 });
    defer a_full.release();
    var b_full = try common.makeStrings(.string, allocator, &.{ "q", "x", "y", null, "w" });
    defer b_full.release();
    var a = try a_full.slice(1, 4);
    defer a.release();
    var b = try b_full.slice(1, 4);
    defer b.release();
    var rb = try batchOf(allocator, a, b);
    defer rb.release();
    var t = try tableOf(allocator, a, b, &.{2}, &.{ 1, 3 });
    defer t.release();

    var mask = try common.makeBoolArray(allocator, &.{ true, null, false, true });
    defer mask.release();
    for ([_]Datum{ rb, t }) |d| {
        var took = try core.datumTakeNullable(d, &.{ 2, 0, null });
        defer took.release();
        try std.testing.expectEqual(d.kind(), took.kind());
        try expectTabular(took, &.{ 1, 3, null }, &.{ null, "x", null });

        var took_dense = try core.datumTake(d, &.{ 3, 3, 1 });
        defer took_dense.release();
        try expectTabular(took_dense, &.{ 2, 2, null }, &.{ "w", "w", "y" });

        var none_taken = try core.datumTake(d, &.{});
        defer none_taken.release();
        try expectTabular(none_taken, &.{}, &.{});

        var kept = try core.datumFilter(d, .{ .array = mask }, .{});
        defer kept.release();
        try expectTabular(kept, &.{ 3, 2 }, &.{ "x", "w" });
        var emitted = try core.datumFilterChunkAware(d, .{ .array = mask }, .{ .drop_nulls = false });
        defer emitted.release();
        try expectTabular(emitted, &.{ 3, null, 2 }, &.{ "x", null, "w" });

        var dropped = try ctx.invokeVector("drop_null", &.{d}, .none);
        defer dropped.release();
        try std.testing.expectEqual(d.kind(), dropped.kind());
        try expectTabular(dropped, &.{ 3, 2 }, &.{ "x", "w" });

        var sliced = try core.datumSlice(d, 1, 2);
        defer sliced.release();
        try expectTabular(sliced, &.{ null, 1 }, &.{ "y", null });
        // drop_null of rows without nulls gives the input back.
        var last = try core.datumSlice(d, 3, 1);
        defer last.release();
        var same = try ctx.invokeVector("drop_null", &.{last}, .none);
        defer same.release();
        try expectTabular(same, &.{2}, &.{"w"});

        var picked = try core.datumSelectColumns(d, &.{ "b", "a", "b" });
        defer picked.release();
        try std.testing.expectEqual(@as(usize, 3), picked.dataType().struct_.fields.len);
        const got = try ints(allocator, picked, 1);
        defer allocator.free(got);
        try common.expectRows(i64, &.{ 3, null, 1, 2 }, got);

        try std.testing.expectError(error.InvalidInput, core.datumSlice(d, 3, 2));
        try std.testing.expectError(error.InvalidInput, core.datumSelectColumns(d, &.{"zz"}));
        try std.testing.expectError(error.InvalidInput, core.datumTake(d, &.{4}));
        // The predicate must have the row count and may not be tabular.
        var short_mask = try mask.slice(0, 3);
        defer short_mask.release();
        try std.testing.expectError(error.InvalidInput, core.datumFilter(d, .{ .array = short_mask }, .{}));
        try std.testing.expectError(error.InvalidInput, core.datumFilter(d, d, .{}));
    }
    // A scalar predicate keeps or drops every row.
    var all = try core.datumFilter(t, .{ .scalar = common.Scalar.init(.{ .bool = {} }, .{ .bool = true }) }, .{});
    defer all.release();
    try expectTabular(all, &.{ 3, null, 1, 2 }, &.{ "x", "y", null, "w" });
    // A take with a null index into a non-nullable field is rejected.
    const strict = [_]common.datatype.Field{.{ .name = "a", .data_type = &int64_type, .nullable = false }};
    var full = try a_full.slice(0, 2);
    defer full.release();
    var strict_batch = try RecordBatch.initBorrowed(allocator, .{ .fields = &strict }, &.{full});
    errdefer strict_batch.deinit();
    var strict_datum = try Datum.fromRecordBatch(allocator, strict_batch);
    defer strict_datum.release();
    try std.testing.expectError(error.InvalidInput, core.datumTakeNullable(strict_datum, &.{null}));
}

test "sort_indices and select_k_unstable on record batches and tables with per-key null placement (pyarrow 26 results)" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    const double_type = common.DataType{ .double = {} };
    const f2 = [_]common.datatype.Field{
        .{ .name = "a", .data_type = &double_type },
        .{ .name = "b", .data_type = &string_type },
    };
    var a = try common.makeArray(f64, allocator, &.{ 3, null, 1, nan, 2 });
    defer a.release();
    var b = try common.makeStrings(.string, allocator, &.{ null, "y", null, "x", "w" });
    defer b.release();
    var batch = try RecordBatch.initBorrowed(allocator, .{ .fields = &f2 }, &.{ a, b });
    errdefer batch.deinit();
    var rb = try Datum.fromRecordBatch(allocator, batch);
    defer rb.release();
    var ca = try common.chunkedAt(allocator, a, &.{ 1, 3 });
    defer ca.release();
    var cb = try common.chunkedAt(allocator, b, &.{2});
    defer cb.release();
    var table = try Table.init(allocator, try SchemaRef.fromBorrowed(allocator, .{ .fields = &f2 }), &.{ ca, cb });
    errdefer table.deinit();
    var t = try Datum.fromTable(allocator, table);
    defer t.release();

    const keys = [_]core.SortKey{ .{ .name = "a", .null_placement = .at_start }, .{ .name = "b", .order = .descending } };
    const a_start = [_]core.SortKey{.{ .name = "a", .null_placement = .at_start }};
    const a_end = [_]core.SortKey{.{ .name = "a" }};
    const b_desc = [_]core.SortKey{.{ .name = "b", .order = .descending }};
    for ([_]Datum{ rb, t }) |d| {
        try expectIndices(&ctx, "sort_indices", &.{d}, .{ .sort = .{ .sort_keys = &keys } }, &.{ 1, 3, 2, 4, 0 });
        // SortOptions.null_placement overrides the keys' (pyarrow: at_start over at_end).
        try expectIndices(&ctx, "sort_indices", &.{d}, .{ .sort = .{ .sort_keys = &a_end, .null_placement = .at_start } }, &.{ 1, 3, 2, 4, 0 });
        try expectIndices(&ctx, "sort_indices", &.{d}, .{ .sort = .{ .sort_keys = &a_end } }, &.{ 2, 4, 0, 3, 1 });
        try expectIndices(&ctx, "sort_indices", &.{d}, .{ .sort = .{ .sort_keys = &b_desc } }, &.{ 1, 3, 4, 0, 2 });
        try expectIndices(&ctx, "select_k_unstable", &.{d}, .{ .select_k = .{ .k = 3, .sort_keys = &a_start } }, &.{ 1, 3, 2 });
        try expectIndices(&ctx, "select_k_unstable", &.{d}, .{ .select_k = .{ .k = 2, .sort_keys = &a_end } }, &.{ 2, 4 });
        // Arrow: "Must specify one or more sort keys"; unknown column names.
        try std.testing.expectError(error.InvalidOptions, ctx.invokeVector("sort_indices", &.{d}, .none));
        try std.testing.expectError(error.InvalidOptions, ctx.invokeVector("sort_indices", &.{d}, .{ .sort = .{ .sort_keys = &.{.{ .name = "zz" }} } }));
        try std.testing.expectError(error.InvalidOptions, ctx.invokeVector("select_k_unstable", &.{d}, .{ .select_k = .{ .k = 1 } }));
    }
    // A name two fields share is ambiguous (Arrow: FieldRef "Multiple matches").
    const twice = [_]common.datatype.Field{
        .{ .name = "a", .data_type = &double_type },
        .{ .name = "a", .data_type = &string_type },
    };
    var dup_batch = try RecordBatch.initBorrowed(allocator, .{ .fields = &twice }, &.{ a, b });
    errdefer dup_batch.deinit();
    var dup = try Datum.fromRecordBatch(allocator, dup_batch);
    defer dup.release();
    try std.testing.expectError(error.InvalidOptions, ctx.invokeVector("sort_indices", &.{dup}, .{ .sort = .{ .sort_keys = &a_end } }));
    // A tabular argument is one argument: never mixed with arrays.
    try std.testing.expectError(error.NotImplemented, ctx.invokeVector("sort_indices", &.{ rb, .{ .array = a } }, .{ .sort = .{ .sort_keys = &a_end } }));
}

test "functions without tabular kernels reject record batches and tables" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    var a = try common.makeArray(i64, allocator, &.{ 3, null, 1, 2 });
    defer a.release();
    var b = try common.makeStrings(.string, allocator, &.{ "x", "y", null, "w" });
    defer b.release();
    var rb = try batchOf(allocator, a, b);
    defer rb.release();
    var t = try tableOf(allocator, a, b, &.{2}, &.{2});
    defer t.release();
    for ([_]Datum{ rb, t }) |d| {
        // pyarrow 26: TypeError / NotImplemented for every one of these.
        try std.testing.expectError(error.NotImplemented, ctx.invokeScalar("is_null", &.{d}, .none));
        try std.testing.expectError(error.NotImplemented, ctx.invokeScalar("add", &.{ d, d }, .none));
        try std.testing.expectError(error.NotImplemented, ctx.invokeScalar("cast", &.{d}, .{ .cast = .{ .to_type = int64_type } }));
        try std.testing.expectError(error.NotImplemented, ctx.invokeAggregate("sum", &.{d}, .none));
        try std.testing.expectError(error.NotImplemented, ctx.invokeVector("unique", &.{d}, .none));
        try std.testing.expectError(error.NotImplemented, ctx.invokeVector("rank", &.{d}, .none));
        try std.testing.expectError(error.NotImplemented, ctx.invokeVector("partition_nth_indices", &.{d}, .{ .partition_nth = .{ .pivot = 1 } }));
        try std.testing.expectError(error.NotImplemented, ctx.invoke("hash_sum", .hash_aggregate, &.{ d, .{ .array = a } }, .none));
        try std.testing.expectError(error.NotImplemented, ctx.beginAggregate("sum", &.{d}, .none));
        // The helpers and iterators answer InvalidInput.
        try std.testing.expectError(error.InvalidInput, core.datumListValueAt(d, 0));
        try std.testing.expectError(error.InvalidInput, core.datumSelect(&.{0}, &.{d}));
        try std.testing.expectError(error.InvalidInput, core.inferNaryExecLen(&.{ d, .{ .array = a } }));
        try std.testing.expectError(error.InvalidInput, core.datumFilter(.{ .array = a }, d, .{}));
        try std.testing.expectError(error.InvalidInput, core.datumSlice(.{ .scalar = common.Scalar.init(int64_type, .null) }, 0, 0));
        var it = core.UnaryExecChunkIterator.init(d);
        try std.testing.expectError(error.InvalidInput, it.next());
    }
}

test "struct sort keys sort by their fields, a null struct row null in every field (pyarrow 26 results)" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    // s = [{1, 'x'}, None, {None, 'y'}, {1, 'a'}, {0, None}, {None, None}]; the null row holds
    // field values that must not count.
    const fa = try common.makeArray(i64, allocator, &.{ 1, 7, null, 1, 0, null });
    const fb = try common.makeStrings(.string, allocator, &.{ "x", "q", "y", "a", null, null });
    var validity = try common.dense.bitsBuffer(allocator, 6);
    validity.data[0] = 0b111101;
    var s = try common.dense.structArray(allocator, &.{ "a", "b" }, &.{ fa, fb }, 6, validity);
    defer s.release();
    var k = try common.makeArray(i64, allocator, &.{ 5, 4, 3, 2, 1, 0 });
    defer k.release();
    const sk_fields = [_]common.datatype.Field{
        .{ .name = "s", .data_type = &s.data().data_type },
        .{ .name = "k", .data_type = &int64_type },
    };
    var batch = try RecordBatch.initBorrowed(allocator, .{ .fields = &sk_fields }, &.{ s, k });
    errdefer batch.deinit();
    var rb = try Datum.fromRecordBatch(allocator, batch);
    defer rb.release();
    var t = try tableOf2(allocator, &sk_fields, s, k);
    defer t.release();
    for ([_]Datum{ rb, t }) |d| {
        try expectIndices(&ctx, "sort_indices", &.{d}, .{ .sort = .{ .sort_keys = &.{.{ .name = "s" }} } }, &.{ 4, 3, 0, 2, 1, 5 });
        try expectIndices(&ctx, "sort_indices", &.{d}, .{ .sort = .{ .sort_keys = &.{.{ .name = "s", .order = .descending }} } }, &.{ 0, 3, 4, 2, 1, 5 });
        try expectIndices(&ctx, "sort_indices", &.{d}, .{ .sort = .{ .sort_keys = &.{.{ .name = "s", .null_placement = .at_start }} } }, &.{ 1, 5, 2, 4, 3, 0 });
        try expectIndices(&ctx, "sort_indices", &.{d}, .{ .sort = .{ .sort_keys = &.{ .{ .name = "s", .order = .descending, .null_placement = .at_start }, .{ .name = "k" } } } }, &.{ 5, 1, 2, 0, 3, 4 });
        try expectIndices(&ctx, "sort_indices", &.{d}, .{ .sort = .{ .sort_keys = &.{ .{ .name = "k", .order = .descending }, .{ .name = "s" } } } }, &.{ 0, 1, 2, 3, 4, 5 });
        // select_k on a struct key (Arrow 26 has no struct select_k kernel): the sort prefix.
        try expectIndices(&ctx, "select_k_unstable", &.{d}, .{ .select_k = .{ .k = 3, .sort_keys = &.{.{ .name = "s", .order = .descending }} } }, &.{ 0, 3, 4 });
    }
    // A struct array argument sorts by all its fields in order.
    try expectIndices(&ctx, "sort_indices", &.{.{ .array = s }}, .none, &.{ 4, 3, 0, 2, 1, 5 });
    // A dictionary field sorts by its decoded values.
    var bd = try common.makeStrings(.string, allocator, &.{ "b", "a" });
    defer bd.release();
    const dict_field = try common.makeDictionary(allocator, bd, .{ .bit_width = 8, .signed = true }, &.{ 0, 1, null });
    var sd = try common.dense.structArray(allocator, &.{"d"}, &.{dict_field}, 3, null);
    defer sd.release();
    try expectIndices(&ctx, "sort_indices", &.{.{ .array = sd }}, .none, &.{ 1, 0, 2 });
    // A struct with a field that cannot be sorted has no kernel.
    const nulls = try common.dense.nullArray(allocator, .{ .null = {} }, 6);
    var bad = try common.dense.structArray(allocator, &.{"l"}, &.{nulls}, 6, null);
    defer bad.release();
    try std.testing.expectError(error.NoMatchingKernel, ctx.invokeVector("sort_indices", &.{.{ .array = bad }}, .none));
}

fn tableOf2(allocator: std.mem.Allocator, f: []const common.datatype.Field, a: ArrayRef, b: ArrayRef) !Datum {
    var ca = try common.chunkedAt(allocator, a, &.{ 1, 4 });
    defer ca.release();
    var cb = try common.chunkedAt(allocator, b, &.{3});
    defer cb.release();
    var table = try Table.init(allocator, try SchemaRef.fromBorrowed(allocator, .{ .fields = f }), &.{ ca, cb });
    errdefer table.deinit();
    return Datum.fromTable(allocator, table);
}
