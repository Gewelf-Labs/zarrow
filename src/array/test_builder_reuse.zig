const std = @import("std");
const array = @import("array.zig");
const array_ref = @import("array_ref.zig");
const datatype = @import("../datatype.zig");

// `finishReset` keeps the finished batch's buffers as spares and reuses their memory for the next
// batch once every array built from them is released. These tests pin the two guarantees:
// a live array is never aliased by a later batch, and a reused builder produces the same bytes
// as a fresh one.

const ArrayRef = array_ref.ArrayRef;

fn valueOf(view: anytype, i: usize) ![]const u8 {
    return view.value(i);
}

fn expectBytesBatch(comptime View: type, arr: ArrayRef, values: []const []const u8, null_every: usize) !void {
    const view = View{ .data = arr.data() };
    try arr.data().validateFull();
    try std.testing.expectEqual(values.len, view.len());
    for (values, 0..) |v, i| {
        if (null_every != 0 and i % null_every == 0) {
            try std.testing.expect(view.isNull(i));
        } else {
            try std.testing.expect(!view.isNull(i));
            try std.testing.expectEqualStrings(v, try valueOf(view, i));
        }
    }
}

fn appendBytesBatch(builder: anytype, values: []const []const u8, null_every: usize) !void {
    for (values, 0..) |v, i| {
        if (null_every != 0 and i % null_every == 0) try builder.appendNull() else try builder.append(v);
    }
}

fn checkBytesBuilderReuse(comptime Builder: type, comptime View: type) !void {
    const allocator = std.testing.allocator;
    var pool: [40][20]u8 = undefined;
    var values: [40][]const u8 = undefined;
    for (&pool, 0..) |*p, i| values[i] = std.fmt.bufPrint(p, "v{d}-{s}", .{ i, "xxxxxxxxxxxxxx" }) catch unreachable;
    const batch1 = values[0..30];
    const batch2 = values[10..40];

    var builder = try Builder.init(allocator, 4, 16);
    defer builder.deinit();

    // Batch 1 stays alive while batch 2 is built: batch 2 must get its own memory.
    try appendBytesBatch(&builder, batch1, 3);
    var a1 = try builder.finishReset();
    defer a1.release();
    try appendBytesBatch(&builder, batch2, 0);
    var a2 = try builder.finishReset();
    try expectBytesBatch(View, a1, batch1, 3);
    try expectBytesBatch(View, a2, batch2, 0);
    for (a1.data().buffers[1..], a2.data().buffers[1..]) |b1, b2| {
        if (b1.len() > 0 and b2.len() > 0) try std.testing.expect(b1.data.ptr != b2.data.ptr);
    }

    // Batch 2 released: batch 3 reuses its memory (same pointer) and still reads back exactly.
    const a2_data_ptr = a2.data().buffers[a2.data().buffers.len - 1].data.ptr;
    a2.release();
    try appendBytesBatch(&builder, batch1, 5);
    var a3 = try builder.finishReset();
    defer a3.release();
    try expectBytesBatch(View, a3, batch1, 5);
    try expectBytesBatch(View, a1, batch1, 3);
    try std.testing.expect(a3.data().buffers[a3.data().buffers.len - 1].data.ptr == a2_data_ptr);

    // Same bytes as a fresh builder.
    var fresh = try Builder.init(allocator, 0, 0);
    defer fresh.deinit();
    try appendBytesBatch(&fresh, batch1, 5);
    var f = try fresh.finish();
    defer f.release();
    for (f.data().buffers, a3.data().buffers) |fb, rb| try std.testing.expectEqualSlices(u8, fb.data, rb.data);

    // Empty batch after reuse and a null-only batch after that.
    var e = try builder.finishReset();
    defer e.release();
    try expectBytesBatch(View, e, &.{}, 0);
    try builder.appendNull();
    var n = try builder.finishReset();
    defer n.release();
    try expectBytesBatch(View, n, &.{"ignored"}, 1);
}

test "string/binary/view builders reuse finishReset buffers without aliasing live arrays" {
    try checkBytesBuilderReuse(array.StringBuilder, array.StringArray);
    try checkBytesBuilderReuse(array.LargeStringBuilder, array.LargeStringArray);
    try checkBytesBuilderReuse(array.BinaryBuilder, array.BinaryArray);
    try checkBytesBuilderReuse(array.LargeBinaryBuilder, array.LargeBinaryArray);
    try checkBytesBuilderReuse(array.StringViewBuilder, array.StringViewArray);
    try checkBytesBuilderReuse(array.BinaryViewBuilder, array.BinaryViewArray);
}

test "primitive builder reuses finishReset values buffer without aliasing live arrays" {
    const allocator = std.testing.allocator;
    var builder = try array.Int64Builder.init(allocator, 2);
    defer builder.deinit();

    for (0..100) |i| try builder.append(@intCast(i));
    var a1 = try builder.finishReset();
    defer a1.release();
    for (0..100) |i| {
        if (i % 7 == 0) try builder.appendNull() else try builder.append(@intCast(1000 + i));
    }
    var a2 = try builder.finishReset();
    const v1 = array.Int64Array{ .data = a1.data() };
    const v2 = array.Int64Array{ .data = a2.data() };
    try std.testing.expect(a1.data().buffers[1].data.ptr != a2.data().buffers[1].data.ptr);
    for (0..100) |i| {
        try std.testing.expectEqual(@as(i64, @intCast(i)), try v1.value(i));
        if (i % 7 == 0) try std.testing.expect(v2.isNull(i)) else try std.testing.expectEqual(@as(i64, @intCast(1000 + i)), try v2.value(i));
    }

    // Reclaimed memory: null slots must read 0 like a fresh (zeroed) buffer.
    const a2_ptr = a2.data().buffers[1].data.ptr;
    a2.release();
    for (0..100) |i| {
        if (i % 3 == 0) try builder.appendNull() else try builder.append(@intCast(i * 2));
    }
    var a3 = try builder.finishReset();
    defer a3.release();
    try a3.data().validateFull();
    try std.testing.expect(a3.data().buffers[1].data.ptr == a2_ptr);
    const raw3 = std.mem.bytesAsSlice(i64, a3.data().buffers[1].data);
    for (0..100) |i| {
        if (i % 3 == 0) try std.testing.expectEqual(@as(i64, 0), raw3[i]) else try std.testing.expectEqual(@as(i64, @intCast(i * 2)), raw3[i]);
        try std.testing.expectEqual(@as(i64, @intCast(i)), try v1.value(i));
    }

    var fresh = try array.Int64Builder.init(allocator, 0);
    defer fresh.deinit();
    for (0..100) |i| {
        if (i % 3 == 0) try fresh.appendNull() else try fresh.append(@intCast(i * 2));
    }
    var f = try fresh.finish();
    defer f.release();
    for (f.data().buffers, a3.data().buffers) |fb, rb| try std.testing.expectEqualSlices(u8, fb.data, rb.data);
}

test "struct builder finish and slices keep children in the node allocation" {
    const allocator = std.testing.allocator;
    const int_type = datatype.DataType{ .int32 = {} };
    const fields = [_]datatype.Field{
        .{ .name = "a", .data_type = &int_type, .nullable = true },
        .{ .name = "b", .data_type = &int_type, .nullable = true },
    };
    var sb = array.StructBuilder.init(allocator, fields[0..]);
    defer sb.deinit();
    var ab = try array.Int32Builder.init(allocator, 4);
    defer ab.deinit();
    var bb = try array.Int32Builder.init(allocator, 4);
    defer bb.deinit();
    for (0..6) |i| {
        try ab.append(@intCast(i));
        try bb.append(@intCast(i * 10));
        if (i == 2) try sb.appendNull() else try sb.appendValid();
    }
    var a = try ab.finish();
    defer a.release();
    var b = try bb.finish();
    defer b.release();
    var s = try sb.finishReset(&[_]ArrayRef{ a, b });
    defer s.release();
    try s.data().validateFull();
    try std.testing.expectEqual(@as(usize, 2), s.data().children.len);
    try std.testing.expectEqual(@as(u32, 2), a.node.ref_count.load(.monotonic));

    var sl = try s.slice(1, 4);
    defer sl.release();
    try sl.data().validateFull();
    try std.testing.expect(sl.data().isNull(1));
    const child = array.Int32Array{ .data = sl.data().children[1].data() };
    try std.testing.expectEqual(@as(i32, 40), try child.value(3));
    // A struct slice slices its children (new nodes over the same buffers): the children's own
    // nodes are not retained again, and releasing the slice frees only what the slice allocated.
    sl.release();
    sl = try s.slice(0, 0);
    try std.testing.expectEqual(@as(u32, 2), a.node.ref_count.load(.monotonic));
}
