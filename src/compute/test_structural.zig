const std = @import("std");
const common = @import("test_common.zig");
const dense = @import("dense.zig");
const bitmap = @import("../bitmap.zig");
const buffer = @import("../buffer.zig");

const ArrayRef = common.ArrayRef;
const ArrayData = common.array_mod.ArrayData;
const DataType = common.DataType;
const Datum = common.Datum;
const ExecContext = common.ExecContext;
const Field = common.datatype.Field;
const Options = common.Options;
const Scalar = common.Scalar;
const TypeTag = std.meta.Tag(DataType);

/// A list row of int32 items (null: a null row).
const Row = ?[]const ?i32;

const none = Options.noneValue();
const int32_type = DataType{ .int32 = {} };
const int64_type = DataType{ .int64 = {} };
const string_type = DataType{ .string = {} };
const item32 = Field{ .name = "item", .data_type = &int32_type };
const list_tags = [_]TypeTag{ .list, .large_list, .list_view, .large_list_view };

fn listType(comptime tag: TypeTag, size: i32) DataType {
    return switch (tag) {
        .list => .{ .list = .{ .value_field = item32 } },
        .large_list => .{ .large_list = .{ .value_field = item32 } },
        .list_view => .{ .list_view = .{ .value_field = item32 } },
        .large_list_view => .{ .large_list_view = .{ .value_field = item32 } },
        .fixed_size_list => .{ .fixed_size_list = .{ .value_field = item32, .list_size = size } },
        else => unreachable,
    };
}

fn ownedBytes(allocator: std.mem.Allocator, bytes: []const u8) !buffer.SharedBuffer {
    var b = try buffer.OwnedBuffer.init(allocator, bytes.len);
    @memcpy(b.data[0..bytes.len], bytes);
    return b.toShared(bytes.len);
}

/// A list array of layout `tag` over int32 `rows`: list and large_list store the items in row
/// order, the view layouts in reverse row order (not contiguous), a fixed_size_list stores `size`
/// zeros for a null row.
fn makeList(comptime tag: TypeTag, allocator: std.mem.Allocator, rows: []const Row, size: usize) !ArrayRef {
    const n = rows.len;
    const view = tag == .list_view or tag == .large_list_view;
    var vals: std.ArrayList(?i32) = .empty;
    defer vals.deinit(allocator);
    const starts = try allocator.alloc(usize, n + 1);
    defer allocator.free(starts);
    for (0..n) |k| {
        const i = if (view) n - 1 - k else k;
        starts[i] = vals.items.len;
        if (rows[i]) |r| try vals.appendSlice(allocator, r) else if (tag == .fixed_size_list) try vals.appendNTimes(allocator, 0, size);
    }
    starts[n] = vals.items.len;
    const values = try common.makeArray(i32, allocator, vals.items);
    const validity = try allocator.alloc(u8, bitmap.byteLength(n));
    defer allocator.free(validity);
    @memset(validity, 0);
    var nulls: usize = 0;
    for (rows, 0..) |r, i| {
        if (r != null) bitmap.setBit(validity, i) else nulls += 1;
    }
    var bufs: [3]buffer.SharedBuffer = .{ try ownedBytes(allocator, validity), .empty, .empty };
    var nbufs: usize = 1;
    switch (tag) {
        .list, .large_list => {
            const O = if (tag == .list) i32 else i64;
            const offs = try allocator.alloc(O, n + 1);
            defer allocator.free(offs);
            for (offs, starts) |*o, s| o.* = @intCast(s);
            bufs[1] = try ownedBytes(allocator, std.mem.sliceAsBytes(offs));
            nbufs = 2;
        },
        .list_view, .large_list_view => {
            const O = if (tag == .list_view) i32 else i64;
            const offs = try allocator.alloc(O, 2 * n);
            defer allocator.free(offs);
            for (rows, 0..) |r, i| {
                offs[i] = if (r == null) 0 else @intCast(starts[i]);
                offs[n + i] = if (r) |x| @intCast(x.len) else 0;
            }
            bufs[1] = try ownedBytes(allocator, std.mem.sliceAsBytes(offs[0..n]));
            bufs[2] = try ownedBytes(allocator, std.mem.sliceAsBytes(offs[n..]));
            nbufs = 3;
        },
        .fixed_size_list => {},
        else => unreachable,
    }
    const arr = try ArrayRef.fromOwnedInline(allocator, .{
        .data_type = listType(tag, @intCast(size)),
        .length = n,
        .null_count = nulls,
        .buffers = bufs[0..nbufs],
        .children = &.{values},
    });
    try arr.data().validateFull();
    return arr;
}

/// The rows of a list-like array of int32 items, read from its buffers (the arena owns them).
fn listRows(arena: std.mem.Allocator, d: *const ArrayData) ![]Row {
    try d.validateFull();
    const child = try common.collect(i32, arena, .{ .array = d.children[0] });
    const out = try arena.alloc(Row, d.length);
    for (out, 0..) |*o, i| {
        if (d.isNull(i)) {
            o.* = null;
            continue;
        }
        const j = d.offset + i;
        const start: usize, const len: usize = switch (d.data_type) {
            .list, .map => blk: {
                const offs = try d.typedBufferAt(1, i32);
                break :blk .{ @intCast(offs[j]), @intCast(offs[j + 1] - offs[j]) };
            },
            .large_list => blk: {
                const offs = try d.typedBufferAt(1, i64);
                break :blk .{ @intCast(offs[j]), @intCast(offs[j + 1] - offs[j]) };
            },
            .list_view => .{ @intCast((try d.typedBufferAt(1, i32))[j]), @intCast((try d.typedBufferAt(2, i32))[j]) },
            .large_list_view => .{ @intCast((try d.typedBufferAt(1, i64))[j]), @intCast((try d.typedBufferAt(2, i64))[j]) },
            .fixed_size_list => |f| .{ j * @as(usize, @intCast(f.list_size)), @intCast(f.list_size) },
            else => return error.TestUnexpectedResult,
        };
        o.* = child[start..][0..len];
    }
    return out;
}

/// The rows of a list datum (array, chunked or scalar).
fn collectLists(arena: std.mem.Allocator, datum: Datum) ![]Row {
    switch (datum) {
        .array => |a| return listRows(arena, a.data()),
        .scalar => |s| {
            if (s.payload) |p| return listRows(arena, p.data());
            const out = try arena.alloc(Row, 1);
            out[0] = null;
            return out;
        },
        .chunked => |c| {
            var all: std.ArrayList(Row) = .empty;
            for (c.chunks()) |chunk| try all.appendSlice(arena, try listRows(arena, chunk.data()));
            return all.items;
        },
        else => return error.TestUnexpectedResult,
    }
}

fn expectLists(expected: []const Row, actual: []const Row) !void {
    try std.testing.expectEqual(expected.len, actual.len);
    for (expected, actual, 0..) |e, a, i| {
        if (e == null or a == null) {
            if (e == null and a == null) continue;
            std.debug.print("row {d}: expected {any}, got {any}\n", .{ i, e, a });
            return error.TestExpectedEqual;
        }
        common.expectRows(i32, e.?, a.?) catch |err| {
            std.debug.print("in row {d}\n", .{i});
            return err;
        };
    }
}

fn expectListsOf(arena: std.mem.Allocator, ctx: *ExecContext, kind: common.core.FunctionKind, name: []const u8, args: []const Datum, options: Options, out_tag: TypeTag, expected: []const Row) !void {
    var out = try ctx.invoke(name, kind, args, options);
    defer out.release();
    try std.testing.expectEqual(out_tag, std.meta.activeTag(out.dataType()));
    try expectLists(expected, try collectLists(arena, out));
}

fn expectValues(comptime T: type, ctx: *ExecContext, kind: common.core.FunctionKind, name: []const u8, args: []const Datum, options: Options, expected: []const ?T) !void {
    const allocator = std.testing.allocator;
    var out = try ctx.invoke(name, kind, args, options);
    defer out.release();
    switch (out) {
        .array => |a| try a.data().validateFull(),
        .chunked => |c| for (c.chunks()) |chunk| try chunk.data().validateFull(),
        else => {},
    }
    const rows = try common.collect(T, allocator, out);
    defer allocator.free(rows);
    try common.expectRows(T, expected, rows);
}

fn sliceOpts(start: i64, stop: ?i64, step: i64, fixed: ?bool) Options {
    return .{ .list_slice = .{ .start = start, .stop = stop, .step = step, .return_fixed_size_list = fixed } };
}

fn idx(comptime T: type, v: T) Datum {
    return .{ .scalar = common.scalarOf(T, v) };
}

const recursive: Options = .{ .list_flatten = .{ .recursive = true } };

test "structural list functions: list_value_length, list_element, list_flatten, list_parent_indices, list_slice on every list layout, sliced (pyarrow 26 results)" {
    const allocator = std.testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);

    // R = [[1, 2, 3], null, [], [4, null, 6, 7]], E = R without the empty row. pyarrow 26
    // registers list_element for list_view but rejects every index ("should be in [0, -2)"), so
    // the view layouts are checked against the list results.
    const r_rows = [_]Row{ &.{ 1, 2, 3 }, null, &.{}, &.{ 4, null, 6, 7 } };
    const e_rows = [_]Row{ &.{ 1, 2, 3 }, null, &.{ 4, null, 6, 7 } };
    inline for (list_tags) |tag| {
        const wide = tag == .large_list or tag == .large_list_view;
        var r = try makeList(tag, allocator, &r_rows, 0);
        defer r.release();
        var tail = try r.slice(1, 3);
        defer tail.release();
        const L = if (wide) i64 else i32;
        try expectValues(L, &ctx, .scalar, "list_value_length", &.{.{ .array = r }}, none, &.{ 3, null, 0, 4 });
        try expectValues(L, &ctx, .scalar, "list_value_length", &.{.{ .array = tail }}, none, &.{ null, 0, 4 });
        try expectValues(i32, &ctx, .vector, "list_flatten", &.{.{ .array = r }}, none, &.{ 1, 2, 3, 4, null, 6, 7 });
        try expectValues(i32, &ctx, .vector, "list_flatten", &.{.{ .array = tail }}, none, &.{ 4, null, 6, 7 });
        try expectValues(i64, &ctx, .vector, "list_parent_indices", &.{.{ .array = r }}, none, &.{ 0, 0, 0, 3, 3, 3, 3 });
        try expectValues(i64, &ctx, .vector, "list_parent_indices", &.{.{ .array = tail }}, none, &.{ 2, 2, 2, 2 });

        var e = try makeList(tag, allocator, &e_rows, 0);
        defer e.release();
        try expectValues(i32, &ctx, .scalar, "list_element", &.{ .{ .array = e }, idx(i64, 0) }, none, &.{ 1, null, 4 });
        try expectValues(i32, &ctx, .scalar, "list_element", &.{ .{ .array = e }, idx(u8, 1) }, none, &.{ 2, null, null });
        try expectValues(i32, &ctx, .scalar, "list_element", &.{ .{ .array = e }, idx(i16, 2) }, none, &.{ 3, null, 6 });
        var e_tail = try e.slice(1, 2);
        defer e_tail.release();
        try expectValues(i32, &ctx, .scalar, "list_element", &.{ .{ .array = e_tail }, idx(i32, 1) }, none, &.{ null, null });
        // Index 0 of the empty row and index 3 of a 3-item row: Arrow's "Index ... is out of bounds".
        try std.testing.expectError(error.InvalidInput, ctx.invokeScalar("list_element", &.{ .{ .array = r }, idx(i64, 0) }, none));
        try std.testing.expectError(error.InvalidInput, ctx.invokeScalar("list_element", &.{ .{ .array = e }, idx(i64, 3) }, none));

        const same = tag;
        try expectListsOf(arena, &ctx, .scalar, "list_slice", &.{.{ .array = r }}, sliceOpts(1, null, 1, null), same, &.{ &.{ 2, 3 }, null, &.{}, &.{ null, 6, 7 } });
        try expectListsOf(arena, &ctx, .scalar, "list_slice", &.{.{ .array = r }}, sliceOpts(1, 3, 1, null), same, &.{ &.{ 2, 3 }, null, &.{}, &.{ null, 6 } });
        try expectListsOf(arena, &ctx, .scalar, "list_slice", &.{.{ .array = r }}, sliceOpts(0, 4, 2, null), same, &.{ &.{ 1, 3 }, null, &.{}, &.{ 4, 6 } });
        try expectListsOf(arena, &ctx, .scalar, "list_slice", &.{.{ .array = r }}, sliceOpts(0, null, 2, null), same, &.{ &.{ 1, 3 }, null, &.{}, &.{ 4, 6 } });
        try expectListsOf(arena, &ctx, .scalar, "list_slice", &.{.{ .array = r }}, sliceOpts(1, null, 1, false), same, &.{ &.{ 2, 3 }, null, &.{}, &.{ null, 6, 7 } });
        try expectListsOf(arena, &ctx, .scalar, "list_slice", &.{.{ .array = r }}, sliceOpts(0, 5, 1, null), same, &.{ &.{ 1, 2, 3 }, null, &.{}, &.{ 4, null, 6, 7 } });
        try expectListsOf(arena, &ctx, .scalar, "list_slice", &.{.{ .array = r }}, sliceOpts(2, 2, 1, null), same, &.{ &.{}, null, &.{}, &.{} });
        try expectListsOf(arena, &ctx, .scalar, "list_slice", &.{.{ .array = r }}, sliceOpts(1, 3, 1, true), .fixed_size_list, &.{ &.{ 2, 3 }, null, &.{ null, null }, &.{ null, 6 } });
        try expectListsOf(arena, &ctx, .scalar, "list_slice", &.{.{ .array = tail }}, sliceOpts(1, 3, 1, null), same, &.{ null, &.{}, &.{ null, 6 } });
        // Arrow: "Unable to produce FixedSizeListArray from non-FixedSizeListArray without `stop`".
        try std.testing.expectError(error.InvalidOptions, ctx.invokeScalar("list_slice", &.{.{ .array = r }}, sliceOpts(1, null, 1, true)));
    }

    // fixed_size_list<int32>[3]: F = [[1, 2, 3], null, [4, null, 6], [7, 8, 9]].
    const f_rows = [_]Row{ &.{ 1, 2, 3 }, null, &.{ 4, null, 6 }, &.{ 7, 8, 9 } };
    var f = try makeList(.fixed_size_list, allocator, &f_rows, 3);
    defer f.release();
    var f_tail = try f.slice(1, 3);
    defer f_tail.release();
    try expectValues(i32, &ctx, .scalar, "list_value_length", &.{.{ .array = f }}, none, &.{ 3, null, 3, 3 });
    try expectValues(i32, &ctx, .vector, "list_flatten", &.{.{ .array = f }}, none, &.{ 1, 2, 3, 4, null, 6, 7, 8, 9 });
    try expectValues(i32, &ctx, .vector, "list_flatten", &.{.{ .array = f_tail }}, none, &.{ 4, null, 6, 7, 8, 9 });
    try expectValues(i64, &ctx, .vector, "list_parent_indices", &.{.{ .array = f }}, none, &.{ 0, 0, 0, 2, 2, 2, 3, 3, 3 });
    try expectValues(i64, &ctx, .vector, "list_parent_indices", &.{.{ .array = f_tail }}, none, &.{ 1, 1, 1, 2, 2, 2 });
    try expectValues(i32, &ctx, .scalar, "list_element", &.{ .{ .array = f }, idx(i64, 0) }, none, &.{ 1, null, 4, 7 });
    try expectValues(i32, &ctx, .scalar, "list_element", &.{ .{ .array = f }, idx(i64, 2) }, none, &.{ 3, null, 6, 9 });
    try expectValues(i32, &ctx, .scalar, "list_element", &.{ .{ .array = f_tail }, idx(i64, 1) }, none, &.{ null, null, 8 });
    try std.testing.expectError(error.InvalidInput, ctx.invokeScalar("list_element", &.{ .{ .array = f }, idx(i64, 3) }, none));
    try expectListsOf(arena, &ctx, .scalar, "list_slice", &.{.{ .array = f }}, sliceOpts(1, null, 1, null), .fixed_size_list, &.{ &.{ 2, 3 }, null, &.{ null, 6 }, &.{ 8, 9 } });
    try expectListsOf(arena, &ctx, .scalar, "list_slice", &.{.{ .array = f }}, sliceOpts(0, 4, 2, null), .fixed_size_list, &.{ &.{ 1, 3 }, null, &.{ 4, 6 }, &.{ 7, 9 } });
    try expectListsOf(arena, &ctx, .scalar, "list_slice", &.{.{ .array = f }}, sliceOpts(1, null, 1, false), .list, &.{ &.{ 2, 3 }, null, &.{ null, 6 }, &.{ 8, 9 } });
    try expectListsOf(arena, &ctx, .scalar, "list_slice", &.{.{ .array = f }}, sliceOpts(0, 5, 1, null), .fixed_size_list, &.{ &.{ 1, 2, 3, null, null }, null, &.{ 4, null, 6, null, null }, &.{ 7, 8, 9, null, null } });
    try expectListsOf(arena, &ctx, .scalar, "list_slice", &.{.{ .array = f }}, sliceOpts(2, 2, 1, null), .fixed_size_list, &.{ &.{}, null, &.{}, &.{} });
    // A start past the size (stop defaults to it): 0 items; past it with a stop: nulls.
    try expectListsOf(arena, &ctx, .scalar, "list_slice", &.{.{ .array = f }}, sliceOpts(5, null, 1, null), .fixed_size_list, &.{ &.{}, null, &.{}, &.{} });
    try expectListsOf(arena, &ctx, .scalar, "list_slice", &.{.{ .array = f }}, sliceOpts(5, 7, 1, null), .fixed_size_list, &.{ &.{ null, null }, null, &.{ null, null }, &.{ null, null } });
    try expectListsOf(arena, &ctx, .scalar, "list_slice", &.{.{ .array = f_tail }}, sliceOpts(1, 3, 1, null), .fixed_size_list, &.{ null, &.{ null, 6 }, &.{ 8, 9 } });

    // Nested: list<list<int32>> [[[1, 2], null], null, [[3], []]]; recursive flattening; a
    // list<fixed_size_list<int32>[2]> [[[1, 2], null], [[3, 4]]] flattens to [1, 2, 3, 4].
    var inner = try makeList(.list, allocator, &.{ &.{ 1, 2 }, null, &.{3}, &.{} }, 0);
    defer inner.release();
    var lb = try common.array_mod.ListBuilder.init(allocator, 3, .{ .name = "item", .data_type = &inner.data().data_type });
    defer lb.deinit();
    try lb.appendLen(2);
    try lb.appendNull();
    try lb.appendLen(2);
    var nested = try lb.finish(inner);
    defer nested.release();
    try expectListsOf(arena, &ctx, .vector, "list_flatten", &.{.{ .array = nested }}, none, .list, &.{ &.{ 1, 2 }, null, &.{3}, &.{} });
    try expectValues(i32, &ctx, .vector, "list_flatten", &.{.{ .array = nested }}, recursive, &.{ 1, 2, 3 });
    try expectValues(i64, &ctx, .vector, "list_parent_indices", &.{.{ .array = nested }}, none, &.{ 0, 0, 2, 2 });
    try expectListsOf(arena, &ctx, .scalar, "list_element", &.{ .{ .array = nested }, idx(i8, 1) }, none, .list, &.{ null, null, &.{} });
    var pairs = try makeList(.fixed_size_list, allocator, &.{ &.{ 1, 2 }, null, &.{ 3, 4 } }, 2);
    defer pairs.release();
    var lb2 = try common.array_mod.ListBuilder.init(allocator, 2, .{ .name = "item", .data_type = &pairs.data().data_type });
    defer lb2.deinit();
    try lb2.appendLen(2);
    try lb2.appendLen(1);
    var of_fsl = try lb2.finish(pairs);
    defer of_fsl.release();
    try expectValues(i32, &ctx, .vector, "list_flatten", &.{.{ .array = of_fsl }}, recursive, &.{ 1, 2, 3, 4 });
    var rt = try registry.resolveResultType("list_flatten", .vector, &.{.{ .array = of_fsl }}, recursive);
    try std.testing.expect(rt.eql(int32_type));
    rt = try registry.resolveResultType("list_slice", .scalar, &.{.{ .array = of_fsl }}, sliceOpts(0, 3, 2, true));
    try std.testing.expectEqual(@as(i32, 2), rt.fixed_size_list.list_size);

    // list_element over list<string>: [["x", "yy"], null] index 1 is ["yy", null].
    var words = try common.makeStrings(.string, allocator, &.{ "x", "yy" });
    defer words.release();
    var wb = try common.array_mod.ListBuilder.init(allocator, 2, .{ .name = "item", .data_type = &string_type });
    defer wb.deinit();
    try wb.appendLen(2);
    try wb.appendNull();
    var word_lists = try wb.finish(words);
    defer word_lists.release();
    try common.expectInvokeStrings(&ctx, "list_element", &.{ .{ .array = word_lists }, idx(u64, 1) }, none, &.{ "yy", null });
}

/// A struct array over `children` (retained) with field names `names` and row validity `valid`
/// (null: no null rows).
fn makeStruct(allocator: std.mem.Allocator, names: []const []const u8, children: []const ArrayRef, valid: ?[]const bool) !ArrayRef {
    const len = children[0].data().length;
    var validity: ?buffer.OwnedBuffer = null;
    if (valid) |v| {
        const b = try buffer.OwnedBuffer.init(allocator, bitmap.byteLength(len));
        for (v, 0..) |ok, i| if (ok) bitmap.setBit(b.data, i);
        validity = b;
    }
    for (children) |c| _ = c.retain();
    return dense.structArray(allocator, names, children, len, validity);
}

const key_str = Field{ .name = "key", .data_type = &string_type, .nullable = false };
const key_i32 = Field{ .name = "key", .data_type = &int32_type, .nullable = false };
const value_i64 = Field{ .name = "value", .data_type = &int64_type };
const value_str = Field{ .name = "value", .data_type = &string_type };
const value_i32 = Field{ .name = "value", .data_type = &int32_type };

/// A map array over `keys` and `items` (retained) with row lengths `lens` (null: a null row).
fn makeMap(allocator: std.mem.Allocator, key: Field, item: Field, keys: ArrayRef, items: ArrayRef, lens: []const ?usize) !ArrayRef {
    _ = keys.retain();
    _ = items.retain();
    var entries = try dense.structArrayFields(allocator, &.{ key, item }, &.{ keys, items }, keys.data().length, null);
    defer entries.release();
    var mb = try common.array_mod.MapBuilder.init(allocator, lens.len, key, item, false);
    defer mb.deinit();
    for (lens) |l| if (l) |n| try mb.appendLen(n) else try mb.appendNull();
    const m = try mb.finish(entries);
    try m.data().validateFull();
    return m;
}

fn lookup(key: Scalar, occurrence: common.core.MapLookupOptions.Occurrence) Options {
    return .{ .map_lookup = .{ .query_key = key, .occurrence = occurrence } };
}

fn strScalar(s: []const u8) Scalar {
    return Scalar.init(string_type, .{ .string = s });
}

fn fieldPath(indices: []const i32) Options {
    return .{ .struct_field = .{ .indices = indices } };
}

test "structural make_struct, struct_field and map_lookup hand-written values on structs, unions and maps (pyarrow 26 results)" {
    const allocator = std.testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);

    // make_struct(a, b): struct<a: int32, b: string> with the arguments as fields and no null row.
    var a = try common.makeArray(i32, allocator, &.{ 1, null, 3 });
    defer a.release();
    var b = try common.makeStrings(.string, allocator, &.{ "x", "y", null });
    defer b.release();
    const md = [_]common.datatype.KeyValue{.{ .key = "k", .value = "v" }};
    const names_ab = [_][]const u8{ "a", "b" };
    var made = try ctx.invokeScalar("make_struct", &.{ .{ .array = a }, .{ .array = b } }, .{ .make_struct = .{ .field_names = &names_ab, .field_nullability = &.{ true, true }, .field_metadata = &.{ &md, null } } });
    defer made.release();
    try made.array.data().validateFull();
    const st = made.array.data().data_type.struct_;
    try std.testing.expectEqualStrings("a", st.fields[0].name);
    try std.testing.expectEqualStrings("b", st.fields[1].name);
    try std.testing.expectEqualStrings("v", st.fields[0].metadata.?[0].value);
    try std.testing.expect(st.fields[1].metadata == null);
    try std.testing.expectEqual(@as(usize, 0), made.array.data().null_count.?);
    try expectValues(i32, &ctx, .scalar, "struct_field", &.{made}, fieldPath(&.{0}), &.{ 1, null, 3 });
    try common.expectInvokeStrings(&ctx, "struct_field", &.{made}, .{ .struct_field = .{ .names = &.{"b"} } }, &.{ "x", "y", null });
    // pyarrow's default names "0", "1"; a scalar repeated (int64 5); every argument a scalar.
    var dflt = try ctx.invokeScalar("make_struct", &.{ .{ .array = a }, .{ .scalar = common.scalarOf(i64, 5) } }, none);
    defer dflt.release();
    try std.testing.expectEqualStrings("1", dflt.array.data().data_type.struct_.fields[1].name);
    try expectValues(i64, &ctx, .scalar, "struct_field", &.{dflt}, fieldPath(&.{1}), &.{ 5, 5, 5 });
    var all_scalar = try ctx.invokeScalar("make_struct", &.{ .{ .scalar = common.scalarOf(i64, 1) }, .{ .scalar = common.scalarOf(i8, null) } }, none);
    defer all_scalar.release();
    try std.testing.expect(all_scalar == .scalar and !all_scalar.scalar.isNull());
    try expectValues(i8, &ctx, .scalar, "struct_field", &.{all_scalar}, fieldPath(&.{1}), &.{null});
    try expectValues(i64, &ctx, .scalar, "struct_field", &.{all_scalar}, fieldPath(&.{0}), &.{1});
    // A not-nullable field: struct<a: string not null> over a valid argument (with a null,
    // Arrow's "does not allow nulls" is InvalidInput, see the errors test).
    var b_head = try b.slice(0, 2);
    defer b_head.release();
    var strict = try ctx.invokeScalar("make_struct", &.{.{ .array = b_head }}, .{ .make_struct = .{ .field_names = &.{"a"}, .field_nullability = &.{false} } });
    defer strict.release();
    try std.testing.expect(!strict.array.data().data_type.struct_.fields[0].nullable);
    var none_args = try ctx.invokeScalar("make_struct", &.{}, none);
    defer none_args.release();
    try std.testing.expectEqual(@as(usize, 0), none_args.array.data().length);

    // struct<a: int32, b: struct<c: string>>: [{a: 1, b: {c: "x"}}, null, {a: null, b: null},
    // {a: 4, b: {c: null}}]; a null struct row makes every field row null.
    var cs = try common.makeStrings(.string, allocator, &.{ "x", "q", "r", null });
    defer cs.release();
    var inner = try makeStruct(allocator, &.{"c"}, &.{cs}, &.{ true, true, false, true });
    defer inner.release();
    var as = try common.makeArray(i32, allocator, &.{ 1, 9, null, 4 });
    defer as.release();
    var s = try makeStruct(allocator, &.{ "a", "b" }, &.{ as, inner }, &.{ true, false, true, true });
    defer s.release();
    try expectValues(i32, &ctx, .scalar, "struct_field", &.{.{ .array = s }}, fieldPath(&.{0}), &.{ 1, null, null, 4 });
    try common.expectInvokeStrings(&ctx, "struct_field", &.{.{ .array = s }}, fieldPath(&.{ 1, 0 }), &.{ "x", null, null, null });
    try common.expectInvokeStrings(&ctx, "struct_field", &.{.{ .array = s }}, .{ .struct_field = .{ .names = &.{ "b", "c" } } }, &.{ "x", null, null, null });
    var s_tail = try s.slice(1, 3);
    defer s_tail.release();
    try common.expectInvokeStrings(&ctx, "struct_field", &.{.{ .array = s_tail }}, fieldPath(&.{ 1, 0 }), &.{ null, null, null });
    var itself = try ctx.invokeScalar("struct_field", &.{.{ .array = s }}, fieldPath(&.{}));
    defer itself.release();
    try std.testing.expectEqual(@as(usize, 4), itself.array.data().length);
    // Arrow: out-of-bounds field reference, no match, multiple matches.
    try std.testing.expectError(error.InvalidOptions, ctx.invokeScalar("struct_field", &.{.{ .array = s }}, fieldPath(&.{2})));
    try std.testing.expectError(error.InvalidOptions, ctx.invokeScalar("struct_field", &.{.{ .array = s }}, fieldPath(&.{-1})));
    try std.testing.expectError(error.InvalidOptions, ctx.invokeScalar("struct_field", &.{.{ .array = s }}, fieldPath(&.{ 0, 0 })));
    try std.testing.expectError(error.InvalidOptions, ctx.invokeScalar("struct_field", &.{.{ .array = s }}, .{ .struct_field = .{ .names = &.{"zz"} } }));
    var twin = try makeStruct(allocator, &.{ "a", "a" }, &.{ as, as }, null);
    defer twin.release();
    try std.testing.expectError(error.InvalidOptions, ctx.invokeScalar("struct_field", &.{.{ .array = twin }}, .{ .struct_field = .{ .names = &.{"a"} } }));

    // Unions: sparse_union<x: int64, y: string> type ids [0, 1, 0, 1]; dense_union with offsets
    // [0, 0, 1, 1] over [1, 2] and ["a", null]; a row of the other child is null.
    const ufields = [_]Field{ .{ .name = "x", .data_type = &int64_type }, .{ .name = "y", .data_type = &string_type } };
    const ids = [_]i8{ 0, 1 };
    var ints4 = try common.makeArray(i64, allocator, &.{ 1, 2, 3, 4 });
    defer ints4.release();
    var strs4 = try common.makeStrings(.string, allocator, &.{ "a", "b", null, "d" });
    defer strs4.release();
    var sb = try common.array_mod.SparseUnionBuilder.init(allocator, .{ .type_ids = &ids, .fields = &ufields, .mode = .sparse }, 4);
    defer sb.deinit();
    for ([_]i8{ 0, 1, 0, 1 }) |t| try sb.appendTypeId(t);
    var su = try sb.finish(&.{ ints4, strs4 });
    defer su.release();
    try expectValues(i64, &ctx, .scalar, "struct_field", &.{.{ .array = su }}, fieldPath(&.{0}), &.{ 1, null, 3, null });
    try common.expectInvokeStrings(&ctx, "struct_field", &.{.{ .array = su }}, .{ .struct_field = .{ .names = &.{"y"} } }, &.{ null, "b", null, "d" });
    var su_tail = try su.slice(1, 3);
    defer su_tail.release();
    try expectValues(i64, &ctx, .scalar, "struct_field", &.{.{ .array = su_tail }}, fieldPath(&.{0}), &.{ null, 3, null });
    var ints2 = try common.makeArray(i64, allocator, &.{ 1, 2 });
    defer ints2.release();
    var strs2 = try common.makeStrings(.string, allocator, &.{ "a", null });
    defer strs2.release();
    var db = try common.array_mod.DenseUnionBuilder.init(allocator, .{ .type_ids = &ids, .fields = &ufields, .mode = .dense }, 4);
    defer db.deinit();
    for ([_]i8{ 0, 1, 0, 1 }, [_]i32{ 0, 0, 1, 1 }) |t, o| try db.append(t, o);
    var du = try db.finish(&.{ ints2, strs2 });
    defer du.release();
    try expectValues(i64, &ctx, .scalar, "struct_field", &.{.{ .array = du }}, fieldPath(&.{0}), &.{ 1, null, 2, null });
    try common.expectInvokeStrings(&ctx, "struct_field", &.{.{ .array = du }}, fieldPath(&.{1}), &.{ null, "a", null, null });
    var du_tail = try du.slice(1, 3);
    defer du_tail.release();
    try expectValues(i64, &ctx, .scalar, "struct_field", &.{.{ .array = du_tail }}, fieldPath(&.{0}), &.{ null, 2, null });

    // map<string, int64> [[(a, 1), (b, 2), (a, 3)], null, [], [(b, null), (a, 5)]].
    var keys = try common.makeStrings(.string, allocator, &.{ "a", "b", "a", "b", "a" });
    defer keys.release();
    var items = try common.makeArray(i64, allocator, &.{ 1, 2, 3, null, 5 });
    defer items.release();
    var m = try makeMap(allocator, key_str, value_i64, keys, items, &.{ 3, null, 0, 2 });
    defer m.release();
    const ma: Datum = .{ .array = m };
    try expectValues(i64, &ctx, .scalar, "map_lookup", &.{ma}, lookup(strScalar("a"), .first), &.{ 1, null, null, 5 });
    try expectValues(i64, &ctx, .scalar, "map_lookup", &.{ma}, lookup(strScalar("a"), .last), &.{ 3, null, null, 5 });
    try expectValues(i64, &ctx, .scalar, "map_lookup", &.{ma}, lookup(strScalar("b"), .first), &.{ 2, null, null, null });
    try expectValues(i64, &ctx, .scalar, "map_lookup", &.{ma}, lookup(strScalar("zz"), .last), &.{ null, null, null, null });
    var all_a = try ctx.invokeScalar("map_lookup", &.{ma}, lookup(strScalar("a"), .all));
    defer all_a.release();
    try all_a.array.data().validateFull();
    const all_a_items = try common.collect(i64, arena, .{ .array = all_a.array.data().children[0] });
    try common.expectRows(i64, &.{ 1, 3, 5 }, all_a_items);
    try std.testing.expect(!all_a.array.data().isNull(0) and all_a.array.data().isNull(1) and all_a.array.data().isNull(2) and !all_a.array.data().isNull(3));
    var all_b = try ctx.invokeScalar("map_lookup", &.{ma}, lookup(strScalar("b"), .all));
    defer all_b.release();
    const all_b_items = try common.collect(i64, arena, .{ .array = all_b.array.data().children[0] });
    try common.expectRows(i64, &.{ 2, null }, all_b_items);
    var all_zz = try ctx.invokeScalar("map_lookup", &.{ma}, lookup(strScalar("zz"), .all));
    defer all_zz.release();
    try std.testing.expectEqual(@as(usize, 4), all_zz.array.data().null_count.?);
    var m_tail = try m.slice(1, 3);
    defer m_tail.release();
    try expectValues(i64, &ctx, .scalar, "map_lookup", &.{.{ .array = m_tail }}, lookup(strScalar("a"), .first), &.{ null, null, 5 });
    // map<int32, string> [[(1, x), (2, y)], [(2, z)]]: key 2 gives [y, z].
    var ikeys = try common.makeArray(i32, allocator, &.{ 1, 2, 2 });
    defer ikeys.release();
    var sitems = try common.makeStrings(.string, allocator, &.{ "x", "y", "z" });
    defer sitems.release();
    var mi = try makeMap(allocator, key_i32, value_str, ikeys, sitems, &.{ 2, 1 });
    defer mi.release();
    try common.expectInvokeStrings(&ctx, "map_lookup", &.{.{ .array = mi }}, lookup(common.scalarOf(i32, 2), .first), &.{ "y", "z" });
    // Arrow: a null query key is Invalid, another key type a TypeError, as is a float key.
    try std.testing.expectError(error.InvalidOptions, ctx.invokeScalar("map_lookup", &.{.{ .array = mi }}, lookup(common.scalarOf(i32, null), .first)));
    try std.testing.expectError(error.UnsupportedType, ctx.invokeScalar("map_lookup", &.{.{ .array = mi }}, lookup(common.scalarOf(i64, 2), .first)));
    try std.testing.expectError(error.UnsupportedType, ctx.invokeScalar("map_lookup", &.{ma}, lookup(Scalar.init(.{ .large_string = {} }, .{ .string = "a" }), .first)));
    const double_type = DataType{ .double = {} };
    var fkeys = try common.makeArray(f64, allocator, &.{1.5});
    defer fkeys.release();
    var fitems = try common.makeArray(i32, allocator, &.{1});
    defer fitems.release();
    var mf = try makeMap(allocator, .{ .name = "key", .data_type = &double_type, .nullable = false }, value_i32, fkeys, fitems, &.{1});
    defer mf.release();
    try std.testing.expectError(error.UnsupportedType, ctx.invokeScalar("map_lookup", &.{.{ .array = mf }}, lookup(common.scalarOf(f64, 1.5), .first)));
}

/// Random list rows: up to 5 int32 items in [-50, 50) (exactly `fixed` when given); a row and an
/// item are null with probability `null_pct` percent.
fn randomRows(arena: std.mem.Allocator, rng: std.Random, n: usize, null_pct: u8, fixed: ?usize) ![]Row {
    const rows = try arena.alloc(Row, n);
    for (rows) |*r| {
        if (rng.uintLessThan(u8, 100) < null_pct) {
            r.* = null;
            continue;
        }
        const len = fixed orelse rng.uintLessThan(usize, 6);
        const items = try arena.alloc(?i32, len);
        for (items) |*it| it.* = if (rng.uintLessThan(u8, 100) < null_pct) null else rng.intRangeLessThan(i32, -50, 50);
        r.* = items;
    }
    return rows;
}

/// Per-row reference of list_slice (`stop` null: the row end; `size`: fixed_size_list output).
fn sliceRef(arena: std.mem.Allocator, rows: []const Row, start: usize, stop: ?usize, step: usize, size: ?usize) ![]Row {
    const out = try arena.alloc(Row, rows.len);
    for (rows, out) |r, *o| {
        const row = r orelse {
            o.* = null;
            continue;
        };
        var items: std.ArrayList(?i32) = .empty;
        if (size) |k| {
            for (0..k) |j| {
                const pos = start + j * step;
                try items.append(arena, if (pos < row.len) row[pos] else null);
            }
        } else {
            var pos = start;
            while (pos < @min(row.len, stop orelse row.len)) : (pos += step) try items.append(arena, row[pos]);
        }
        o.* = items.items;
    }
    return out;
}

test "structural oracle matches a per-row loop on random sliced inputs with nulls for every layout" {
    const allocator = std.testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    var prng = std.Random.DefaultPrng.init(0x5717c7);
    const rng = prng.random();

    for (0..6) |round| {
        const n = 150 + rng.uintLessThan(usize, 100);
        const off = rng.uintLessThan(usize, 20);
        const null_pct: u8 = if (round == 0) 0 else 5 + rng.uintLessThan(u8, 30);
        inline for (list_tags ++ [_]TypeTag{.fixed_size_list}) |tag| {
            const fixed: ?usize = if (tag == .fixed_size_list) 4 else null;
            const all = try randomRows(arena, rng, n + off, null_pct, fixed);
            const rows = all[off..];
            var full = try makeList(tag, allocator, all, fixed orelse 0);
            defer full.release();
            var arr = try full.slice(off, n);
            defer arr.release();
            const in: Datum = .{ .array = arr };

            // list_value_length, list_flatten, list_parent_indices.
            const wide = tag == .large_list or tag == .large_list_view;
            const lens = try arena.alloc(?i64, n);
            var flat: std.ArrayList(?i32) = .empty;
            var parents: std.ArrayList(?i64) = .empty;
            for (rows, lens, 0..) |r, *l, i| {
                l.* = if (r) |x| @intCast(x.len) else null;
                if (r) |x| {
                    try flat.appendSlice(arena, x);
                    try parents.appendNTimes(arena, @intCast(i), x.len);
                }
            }
            if (wide) {
                try expectValues(i64, &ctx, .scalar, "list_value_length", &.{in}, none, lens);
            } else {
                const lens32 = try arena.alloc(?i32, n);
                for (lens32, lens) |*d, l| d.* = if (l) |x| @intCast(x) else null;
                try expectValues(i32, &ctx, .scalar, "list_value_length", &.{in}, none, lens32);
            }
            try expectValues(i32, &ctx, .vector, "list_flatten", &.{in}, none, flat.items);
            try expectValues(i64, &ctx, .vector, "list_parent_indices", &.{in}, none, parents.items);

            // list_element at an index every valid row has.
            var min_len: usize = std.math.maxInt(usize);
            for (rows) |r| if (r) |x| {
                min_len = @min(min_len, x.len);
            };
            if (min_len != std.math.maxInt(usize) and min_len > 0) {
                const k = rng.uintLessThan(usize, min_len);
                const want = try arena.alloc(?i32, n);
                for (rows, want) |r, *w| w.* = if (r) |x| x[k] else null;
                try expectValues(i32, &ctx, .scalar, "list_element", &.{ in, idx(i64, @intCast(k)) }, none, want);
            }

            // list_slice with random options, as the input layout and as fixed_size_list.
            const start = rng.uintLessThan(usize, 4);
            const stop: ?usize = if (rng.boolean()) start + rng.uintLessThan(usize, 5) else null;
            const step = 1 + rng.uintLessThan(usize, 3);
            const o = sliceOpts(@intCast(start), if (stop) |x| @intCast(x) else null, @intCast(step), null);
            if (tag == .fixed_size_list) {
                const end = stop orelse 4;
                const size = if (end > start) (end - start + step - 1) / step else 0;
                try expectListsOf(arena, &ctx, .scalar, "list_slice", &.{in}, o, .fixed_size_list, try sliceRef(arena, rows, start, null, step, size));
            } else {
                try expectListsOf(arena, &ctx, .scalar, "list_slice", &.{in}, o, tag, try sliceRef(arena, rows, start, stop, step, null));
                if (stop) |x| {
                    const size = if (x > start) (x - start + step - 1) / step else 0;
                    const of = sliceOpts(@intCast(start), @intCast(x), @intCast(step), true);
                    try expectListsOf(arena, &ctx, .scalar, "list_slice", &.{in}, of, .fixed_size_list, try sliceRef(arena, rows, start, null, step, size));
                }
            }
        }

        // struct_field on a random struct with null rows, sliced: a field row is valid where the
        // struct row and the child row are.
        const total = n + off;
        const vals = try arena.alloc(?i32, total);
        const valid = try arena.alloc(bool, total);
        for (vals, valid) |*v, *ok| {
            v.* = if (rng.uintLessThan(u8, 100) < null_pct) null else rng.int(i32);
            ok.* = rng.uintLessThan(u8, 100) >= null_pct;
        }
        var child = try common.makeArray(i32, allocator, vals);
        defer child.release();
        var st_full = try makeStruct(allocator, &.{"v"}, &.{child}, valid);
        defer st_full.release();
        var st = try st_full.slice(off, n);
        defer st.release();
        const want = try arena.alloc(?i32, n);
        for (want, vals[off..], valid[off..]) |*w, v, ok| w.* = if (ok) v else null;
        try expectValues(i32, &ctx, .scalar, "struct_field", &.{.{ .array = st }}, fieldPath(&.{0}), want);

        // map_lookup on random map<int32, int32> with keys in [0, 4), sliced.
        const map_rows = try randomRows(arena, rng, total, null_pct, null);
        var mkeys: std.ArrayList(?i32) = .empty;
        var mitems: std.ArrayList(?i32) = .empty;
        const mlens = try arena.alloc(?usize, total);
        for (map_rows, mlens) |r, *l| {
            l.* = if (r) |x| x.len else null;
            if (r) |x| for (x) |it| {
                try mkeys.append(arena, rng.intRangeLessThan(i32, 0, 4));
                try mitems.append(arena, it);
            };
        }
        var kk = try common.makeArray(i32, allocator, mkeys.items);
        defer kk.release();
        var ii = try common.makeArray(i32, allocator, mitems.items);
        defer ii.release();
        var map_full = try makeMap(allocator, key_i32, value_i32, kk, ii, mlens);
        defer map_full.release();
        var map = try map_full.slice(off, n);
        defer map.release();
        const q = rng.intRangeLessThan(i32, 0, 4);
        var base: usize = 0;
        for (map_rows[0..off]) |r| if (r) |x| {
            base += x.len;
        };
        const first = try arena.alloc(?i32, n);
        const last = try arena.alloc(?i32, n);
        const every = try arena.alloc(Row, n);
        for (map_rows[off..], first, last, every) |r, *fi, *la, *ev| {
            fi.* = null;
            la.* = null;
            ev.* = null;
            const x = r orelse continue;
            var hits: std.ArrayList(?i32) = .empty;
            for (0..x.len) |j| if (mkeys.items[base + j].? == q) try hits.append(arena, mitems.items[base + j]);
            base += x.len;
            if (hits.items.len == 0) continue;
            fi.* = hits.items[0];
            la.* = hits.items[hits.items.len - 1];
            ev.* = hits.items;
        }
        const mdat: Datum = .{ .array = map };
        try expectValues(i32, &ctx, .scalar, "map_lookup", &.{mdat}, lookup(common.scalarOf(i32, q), .first), first);
        try expectValues(i32, &ctx, .scalar, "map_lookup", &.{mdat}, lookup(common.scalarOf(i32, q), .last), last);
        try expectListsOf(arena, &ctx, .scalar, "map_lookup", &.{mdat}, lookup(common.scalarOf(i32, q), .all), .list, every);
    }
}

test "structural chunked inputs with misaligned chunks give a chunked result equal to the array result" {
    const allocator = std.testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    var prng = std.Random.DefaultPrng.init(0xc4);
    const rng = prng.random();

    inline for (list_tags ++ [_]TypeTag{.fixed_size_list}) |tag| {
        const fixed: ?usize = if (tag == .fixed_size_list) 3 else null;
        const rows = try randomRows(arena, rng, 60, 15, fixed);
        for (rows) |*r| if (r.* != null and r.*.?.len == 0) {
            r.* = &.{7};
        };
        var arr = try makeList(tag, allocator, rows, fixed orelse 0);
        defer arr.release();
        var chunks = try common.chunkedAt(allocator, arr, &.{ 7, 30, 31 });
        defer chunks.release();
        const Case = struct { name: []const u8, kind: common.core.FunctionKind, index: bool, o: Options };
        const cases = [_]Case{
            .{ .name = "list_value_length", .kind = .scalar, .index = false, .o = none },
            .{ .name = "list_element", .kind = .scalar, .index = true, .o = none },
            .{ .name = "list_flatten", .kind = .vector, .index = false, .o = none },
            .{ .name = "list_parent_indices", .kind = .vector, .index = false, .o = none },
            .{ .name = "list_slice", .kind = .scalar, .index = false, .o = sliceOpts(0, 2, 1, null) },
        };
        for (cases) |c| {
            const args_a = [2]Datum{ .{ .array = arr }, idx(i64, 0) };
            const args_c = [2]Datum{ .{ .chunked = chunks }, idx(i64, 0) };
            const nargs: usize = if (c.index) 2 else 1;
            var out_a = try ctx.invoke(c.name, c.kind, args_a[0..nargs], c.o);
            defer out_a.release();
            var out_c = try ctx.invoke(c.name, c.kind, args_c[0..nargs], c.o);
            defer out_c.release();
            try std.testing.expect(out_c == .chunked);
            try std.testing.expect(out_c.dataType().eql(out_a.dataType()));
            if (std.mem.eql(u8, c.name, "list_slice")) {
                try expectLists(try collectLists(arena, out_a), try collectLists(arena, out_c));
            } else if (out_a.dataType() == .int64) {
                try common.expectRows(i64, try common.collect(i64, arena, out_a), try common.collect(i64, arena, out_c));
            } else {
                try common.expectRows(i32, try common.collect(i32, arena, out_a), try common.collect(i32, arena, out_c));
            }
        }
    }

    // make_struct, struct_field and map_lookup over chunked inputs.
    var a = try common.makeArray(i32, allocator, &.{ 1, null, 3, 4, 5 });
    defer a.release();
    var ca = try common.chunkedAt(allocator, a, &.{2});
    defer ca.release();
    var b = try common.makeArray(i64, allocator, &.{ 10, 20, null, 40, 50 });
    defer b.release();
    var cb = try common.chunkedAt(allocator, b, &.{ 1, 4 });
    defer cb.release();
    var zipped = try ctx.invokeScalar("make_struct", &.{ .{ .chunked = ca }, .{ .chunked = cb }, .{ .scalar = common.scalarOf(i32, 7) } }, none);
    defer zipped.release();
    try std.testing.expectEqual(@as(usize, 4), zipped.chunked.numChunks());
    try expectValues(i64, &ctx, .scalar, "struct_field", &.{zipped}, fieldPath(&.{1}), &.{ 10, 20, null, 40, 50 });
    try expectValues(i32, &ctx, .scalar, "struct_field", &.{zipped}, fieldPath(&.{2}), &.{ 7, 7, 7, 7, 7 });
    var keys = try common.makeArray(i32, allocator, &.{ 1, 2, 1, 2 });
    defer keys.release();
    var items = try common.makeArray(i32, allocator, &.{ 10, 20, 30, 40 });
    defer items.release();
    var m = try makeMap(allocator, key_i32, value_i32, keys, items, &.{ 2, null, 1, 1 });
    defer m.release();
    var cm = try common.chunkedAt(allocator, m, &.{ 1, 3 });
    defer cm.release();
    try expectValues(i32, &ctx, .scalar, "map_lookup", &.{.{ .chunked = cm }}, lookup(common.scalarOf(i32, 2), .first), &.{ 20, null, null, 40 });
    try expectListsOf(arena, &ctx, .scalar, "map_lookup", &.{.{ .chunked = cm }}, lookup(common.scalarOf(i32, 1), .all), .list, &.{ &.{10}, null, &.{30}, null });
    // list_parent_indices counts rows from the start of the first chunk (pyarrow 26:
    // [[[1, 2]], [[3, 4], null]] gives [[0, 0], [1, 1]]).
    var two = try makeList(.list, allocator, &.{ &.{ 1, 2 }, &.{ 3, 4 }, null }, 0);
    defer two.release();
    var c2 = try common.chunkedAt(allocator, two, &.{1});
    defer c2.release();
    try expectValues(i64, &ctx, .vector, "list_parent_indices", &.{.{ .chunked = c2 }}, none, &.{ 0, 0, 1, 1 });
    // A chunked array without chunks.
    var empty = try common.ChunkedArray.init(allocator, two.data().data_type, &.{});
    defer empty.release();
    var flat = try ctx.invokeVector("list_flatten", &.{.{ .chunked = empty }}, none);
    defer flat.release();
    try std.testing.expectEqual(@as(usize, 0), flat.chunked.len());
}

test "structural scalars, empty and all-null inputs" {
    const allocator = std.testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);

    // The list scalar [1, 2, 3] and a null list scalar.
    var one = try makeList(.list, allocator, &.{&.{ 1, 2, 3 }}, 0);
    defer one.release();
    var ls = try Scalar.initNested(one.data().data_type, one);
    defer ls.release();
    const null_list = Scalar.init(listType(.list, 0), .null);
    try expectValues(i32, &ctx, .scalar, "list_value_length", &.{.{ .scalar = ls }}, none, &.{3});
    try expectValues(i32, &ctx, .scalar, "list_value_length", &.{.{ .scalar = null_list }}, none, &.{null});
    try expectValues(i32, &ctx, .scalar, "list_element", &.{ .{ .scalar = ls }, idx(i64, 1) }, none, &.{2});
    try expectValues(i32, &ctx, .scalar, "list_element", &.{ .{ .scalar = null_list }, idx(i64, 5) }, none, &.{null});
    var sliced = try ctx.invokeScalar("list_slice", &.{.{ .scalar = ls }}, sliceOpts(1, null, 1, null));
    defer sliced.release();
    try std.testing.expect(sliced == .scalar);
    try expectLists(&.{&.{ 2, 3 }}, try collectLists(arena, sliced));
    var null_slice = try ctx.invokeScalar("list_slice", &.{.{ .scalar = null_list }}, sliceOpts(1, 4, 1, null));
    defer null_slice.release();
    try std.testing.expect(null_slice == .scalar and null_slice.scalar.isNull());
    // A scalar flattens to an array (Arrow); list_parent_indices of a scalar is NotImplemented.
    try expectValues(i32, &ctx, .vector, "list_flatten", &.{.{ .scalar = ls }}, none, &.{ 1, 2, 3 });
    try expectValues(i32, &ctx, .vector, "list_flatten", &.{.{ .scalar = null_list }}, none, &.{});
    try std.testing.expectError(error.NotImplemented, ctx.invokeVector("list_parent_indices", &.{.{ .scalar = ls }}, none));

    // Empty and all-null arrays of every layout.
    inline for (list_tags ++ [_]TypeTag{.fixed_size_list}) |tag| {
        var empty = try makeList(tag, allocator, &.{}, 2);
        defer empty.release();
        var nulls = try makeList(tag, allocator, &.{ null, null, null }, 2);
        defer nulls.release();
        const wide = tag == .large_list or tag == .large_list_view;
        const L = if (wide) i64 else i32;
        try expectValues(L, &ctx, .scalar, "list_value_length", &.{.{ .array = empty }}, none, &.{});
        try expectValues(L, &ctx, .scalar, "list_value_length", &.{.{ .array = nulls }}, none, &.{ null, null, null });
        try expectValues(i32, &ctx, .scalar, "list_element", &.{ .{ .array = empty }, idx(i64, 3) }, none, &.{});
        try expectValues(i32, &ctx, .scalar, "list_element", &.{ .{ .array = nulls }, idx(i64, 3) }, none, &.{ null, null, null });
        try expectValues(i32, &ctx, .vector, "list_flatten", &.{.{ .array = nulls }}, none, &.{});
        try expectValues(i64, &ctx, .vector, "list_parent_indices", &.{.{ .array = empty }}, none, &.{});
        try expectListsOf(arena, &ctx, .scalar, "list_slice", &.{.{ .array = nulls }}, sliceOpts(0, 1, 1, true), .fixed_size_list, &.{ null, null, null });
    }

    // struct_field of a struct scalar and of a null struct scalar.
    var a = try common.makeArray(i32, allocator, &.{7});
    defer a.release();
    var s1 = try makeStruct(allocator, &.{"a"}, &.{a}, null);
    defer s1.release();
    var ss = try Scalar.initNested(s1.data().data_type, s1);
    defer ss.release();
    try expectValues(i32, &ctx, .scalar, "struct_field", &.{.{ .scalar = ss }}, fieldPath(&.{0}), &.{7});
    try expectValues(i32, &ctx, .scalar, "struct_field", &.{.{ .scalar = Scalar.init(s1.data().data_type, .null) }}, fieldPath(&.{0}), &.{null});
    // make_struct of a null list scalar repeated over an array argument.
    var three = try common.makeArray(i32, allocator, &.{ 1, 2, 3 });
    defer three.release();
    var with_null = try ctx.invokeScalar("make_struct", &.{ .{ .array = three }, .{ .scalar = null_list } }, none);
    defer with_null.release();
    try with_null.array.data().validateFull();
    var lists = try ctx.invokeScalar("struct_field", &.{with_null}, fieldPath(&.{1}));
    defer lists.release();
    try expectLists(&.{ null, null, null }, try collectLists(arena, lists));
    // A null map row and an empty map row give null.
    var keys = try common.makeArray(i32, allocator, &.{});
    defer keys.release();
    var m = try makeMap(allocator, key_i32, value_i32, keys, keys, &.{ null, 0 });
    defer m.release();
    try expectValues(i32, &ctx, .scalar, "map_lookup", &.{.{ .array = m }}, lookup(common.scalarOf(i32, 1), .first), &.{ null, null });
}

test "structural errors: indices, options, types, arity and malformed input" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);

    var l = try makeList(.list, allocator, &.{ &.{ 1, 2 }, null }, 0);
    defer l.release();
    const la: Datum = .{ .array = l };
    // list_element: a null or negative index (Arrow: "Index must not be null", "should be
    // greater than or equal to 0"), a huge index, an index array (NotImplemented), a float index.
    try std.testing.expectError(error.InvalidInput, ctx.invokeScalar("list_element", &.{ la, .{ .scalar = common.scalarOf(i64, null) } }, none));
    try std.testing.expectError(error.InvalidInput, ctx.invokeScalar("list_element", &.{ la, idx(i64, -1) }, none));
    try std.testing.expectError(error.InvalidInput, ctx.invokeScalar("list_element", &.{ la, idx(u64, std.math.maxInt(u64)) }, none));
    var ix = try common.makeArray(i64, allocator, &.{ 0, 0 });
    defer ix.release();
    try std.testing.expectError(error.NotImplemented, ctx.invokeScalar("list_element", &.{ la, .{ .array = ix } }, none));
    try std.testing.expectError(error.NoMatchingKernel, ctx.invokeScalar("list_element", &.{ la, .{ .scalar = common.scalarOf(f64, 0) } }, none));
    // list_slice: start < 0, stop < start, step < 1 (Arrow's checks), no options.
    try std.testing.expectError(error.InvalidOptions, ctx.invokeScalar("list_slice", &.{la}, sliceOpts(-1, null, 1, null)));
    try std.testing.expectError(error.InvalidOptions, ctx.invokeScalar("list_slice", &.{la}, sliceOpts(3, 1, 1, null)));
    try std.testing.expectError(error.InvalidOptions, ctx.invokeScalar("list_slice", &.{la}, sliceOpts(1, 3, 0, null)));
    try std.testing.expectError(error.InvalidOptions, ctx.invokeScalar("list_slice", &.{la}, none));
    // Types: a map is not a list for these functions (Arrow), an int32 array is neither.
    var keys = try common.makeArray(i32, allocator, &.{1});
    defer keys.release();
    var m = try makeMap(allocator, key_i32, value_i32, keys, keys, &.{1});
    defer m.release();
    try std.testing.expectError(error.NoMatchingKernel, ctx.invokeVector("list_flatten", &.{.{ .array = m }}, none));
    try std.testing.expectError(error.NoMatchingKernel, ctx.invokeScalar("list_value_length", &.{.{ .array = m }}, none));
    try std.testing.expectError(error.NoMatchingKernel, ctx.invokeScalar("struct_field", &.{.{ .array = keys }}, fieldPath(&.{0})));
    try std.testing.expectError(error.NoMatchingKernel, ctx.invokeScalar("map_lookup", &.{la}, lookup(common.scalarOf(i32, 1), .first)));
    try std.testing.expectError(error.InvalidArity, ctx.invokeScalar("list_value_length", &.{ la, la }, none));
    // make_struct: option lengths must match the arguments; array lengths must match; a field
    // declared not nullable must get no null (Arrow's Invalid).
    try std.testing.expectError(error.InvalidOptions, ctx.invokeScalar("make_struct", &.{ la, la }, .{ .make_struct = .{ .field_names = &.{"a"} } }));
    try std.testing.expectError(error.InvalidOptions, ctx.invokeScalar("make_struct", &.{la}, .{ .make_struct = .{ .field_nullability = &.{ true, false } } }));
    try std.testing.expectError(error.InvalidInput, ctx.invokeScalar("make_struct", &.{ la, .{ .array = keys } }, none));
    var nn = try common.makeArray(i32, allocator, &.{ 1, null });
    defer nn.release();
    try std.testing.expectError(error.InvalidInput, ctx.invokeScalar("make_struct", &.{.{ .array = nn }}, .{ .make_struct = .{ .field_names = &.{"a"}, .field_nullability = &.{false} } }));
    try std.testing.expectError(error.InvalidInput, ctx.invokeScalar("make_struct", &.{.{ .scalar = common.scalarOf(i32, null) }}, .{ .make_struct = .{ .field_names = &.{"a"}, .field_nullability = &.{false} } }));

    // Malformed input: list offsets past the child, a list_view size past the child, a
    // fixed_size_list child that is too short, a dense union offset past its child.
    var items = try common.makeArray(i32, allocator, &.{ 1, 2 });
    defer items.release();
    const bad_offs = [_]i32{ 0, 5 };
    const bufs = [_]buffer.SharedBuffer{ .empty, .fromSlice(std.mem.sliceAsBytes(&bad_offs)) };
    var bad = try ArrayRef.fromBorrowed(allocator, .{ .data_type = listType(.list, 0), .length = 1, .null_count = 0, .buffers = &bufs, .children = &.{items} });
    defer bad.release();
    try std.testing.expectError(error.InvalidInput, ctx.invokeScalar("list_value_length", &.{.{ .array = bad }}, none));
    try std.testing.expectError(error.InvalidInput, ctx.invokeScalar("list_slice", &.{.{ .array = bad }}, sliceOpts(0, null, 1, null)));
    try std.testing.expectError(error.InvalidInput, ctx.invokeScalar("list_element", &.{ .{ .array = bad }, idx(i64, 0) }, none));
    try std.testing.expectError(error.InvalidInput, ctx.invokeVector("list_flatten", &.{.{ .array = bad }}, none));
    try std.testing.expectError(error.InvalidInput, ctx.invokeVector("list_parent_indices", &.{.{ .array = bad }}, none));
    const voffs = [_]i32{0};
    const vsizes = [_]i32{3};
    const vbufs = [_]buffer.SharedBuffer{ .empty, .fromSlice(std.mem.sliceAsBytes(&voffs)), .fromSlice(std.mem.sliceAsBytes(&vsizes)) };
    var bad_view = try ArrayRef.fromBorrowed(allocator, .{ .data_type = listType(.list_view, 0), .length = 1, .null_count = 0, .buffers = &vbufs, .children = &.{items} });
    defer bad_view.release();
    try std.testing.expectError(error.InvalidInput, ctx.invokeVector("list_flatten", &.{.{ .array = bad_view }}, none));
    const fbufs = [_]buffer.SharedBuffer{.empty};
    var bad_fsl = try ArrayRef.fromBorrowed(allocator, .{ .data_type = listType(.fixed_size_list, 3), .length = 1, .null_count = 0, .buffers = &fbufs, .children = &.{items} });
    defer bad_fsl.release();
    try std.testing.expectError(error.InvalidInput, ctx.invokeScalar("list_element", &.{ .{ .array = bad_fsl }, idx(i64, 2) }, none));
    const ufields = [_]Field{.{ .name = "x", .data_type = &int32_type }};
    const uids = [_]i8{0};
    const type_ids = [_]i8{0};
    const uoffs = [_]i32{9};
    const ubufs = [_]buffer.SharedBuffer{ .fromSlice(std.mem.sliceAsBytes(&type_ids)), .fromSlice(std.mem.sliceAsBytes(&uoffs)) };
    var bad_union = try ArrayRef.fromBorrowed(allocator, .{ .data_type = .{ .dense_union = .{ .type_ids = &uids, .fields = &ufields, .mode = .dense } }, .length = 1, .buffers = &ubufs, .children = &.{items} });
    defer bad_union.release();
    try std.testing.expectError(error.InvalidInput, ctx.invokeScalar("struct_field", &.{.{ .array = bad_union }}, fieldPath(&.{0})));
}
