//! pivot_wider and hash_pivot_wider. Expected values are pyarrow 26 results.
const std = @import("std");
const common = @import("test_common.zig");

const ArrayRef = common.ArrayRef;
const DataType = common.DataType;
const Datum = common.Datum;
const ExecContext = common.ExecContext;
const Options = common.Options;
const none = common.none;

fn pivot(names: []const []const u8) Options {
    return .{ .pivot_wider = .{ .key_names = names } };
}

/// The fields of a pivot_wider struct scalar (or of row `row` of a struct array) as optional
/// float64 values, with their names.
fn expectFields(d: Datum, row: usize, names: []const []const u8, want: []const ?f64) !void {
    const arr = switch (d) {
        .scalar => |s| s.payload.?,
        .array => |a| a,
        else => return error.TestUnexpectedResult,
    };
    try arr.data().validateFull();
    const st = arr.data().data_type.struct_;
    try std.testing.expectEqual(names.len, st.fields.len);
    for (names, st.fields, want, 0..) |n, f, w, k| {
        try std.testing.expectEqualStrings(n, f.name);
        var field = try common.core.datumStructField(.{ .array = arr }, k);
        defer field.release();
        const got = try common.collect(f64, std.testing.allocator, field);
        defer std.testing.allocator.free(got);
        try std.testing.expectEqual(w, got[row]);
    }
}

test "pivot_wider hand-written values, null and unexpected keys, key types (pyarrow 26 results)" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    var keys = try common.makeStrings(.string, allocator, &.{ "height", "width", "depth", "height", null, "other" });
    defer keys.release();
    var vals = try common.makeArray(f64, allocator, &.{ 10.5, 11.5, null, 12.0, 1.0, 2.0 });
    defer vals.release();
    const hw = [_][]const u8{ "height", "width" };
    var k3 = try keys.slice(0, 3);
    defer k3.release();
    var v3 = try vals.slice(0, 3);
    defer v3.release();
    var out = try ctx.invokeAggregate("pivot_wider", &.{ .{ .array = k3 }, .{ .array = v3 } }, pivot(&hw));
    defer out.release();
    try expectFields(out, 0, &hw, &.{ 10.5, 11.5 });
    // The struct names are copied: the result outlives the options.
    {
        var names = [_][]const u8{ "height", "x" };
        var missing = try ctx.invokeAggregate("pivot_wider", &.{ .{ .array = k3 }, .{ .array = v3 } }, pivot(&names));
        defer missing.release();
        names[1] = "changed";
        try expectFields(missing, 0, &.{ "height", "x" }, &.{ 10.5, null });
    }
    // Two non-null values for one key; a null key; an unexpected key with `raise`.
    var k4 = try keys.slice(0, 4);
    defer k4.release();
    var v4 = try vals.slice(0, 4);
    defer v4.release();
    try std.testing.expectError(error.InvalidInput, ctx.invokeAggregate("pivot_wider", &.{ .{ .array = k4 }, .{ .array = v4 } }, pivot(&hw)));
    var k_null = try keys.slice(4, 1);
    defer k_null.release();
    var v_null = try vals.slice(4, 1);
    defer v_null.release();
    try std.testing.expectError(error.InvalidInput, ctx.invokeAggregate("pivot_wider", &.{ .{ .array = k_null }, .{ .array = v_null } }, pivot(&.{"height"})));
    var k_other = try keys.slice(5, 1);
    defer k_other.release();
    var v_other = try vals.slice(5, 1);
    defer v_other.release();
    var ignored = try ctx.invokeAggregate("pivot_wider", &.{ .{ .array = k_other }, .{ .array = v_other } }, pivot(&.{"height"}));
    defer ignored.release();
    try expectFields(ignored, 0, &.{"height"}, &.{null});
    try std.testing.expectError(error.InvalidInput, ctx.invokeAggregate("pivot_wider", &.{ .{ .array = k_other }, .{ .array = v_other } }, .{ .pivot_wider = .{ .key_names = &.{"height"}, .unexpected_key_behavior = .raise } }));
    // Duplicate key names (Arrow: KeyError); no key names: struct<>.
    try std.testing.expectError(error.InvalidOptions, ctx.invokeAggregate("pivot_wider", &.{ .{ .array = k3 }, .{ .array = v3 } }, pivot(&.{ "height", "height" })));
    var empty_struct = try ctx.invokeAggregate("pivot_wider", &.{ .{ .array = k3 }, .{ .array = v3 } }, pivot(&.{}));
    defer empty_struct.release();
    try expectFields(empty_struct, 0, &.{}, &.{});

    // A null value does not count: ['a', 'a'] with [None, 1] and [1, None] both give a = 1.
    var aa = try common.makeStrings(.string, allocator, &.{ "a", "a", "b" });
    defer aa.release();
    var n1 = try common.makeArray(f64, allocator, &.{ null, 1, null });
    defer n1.release();
    var o1 = try ctx.invokeAggregate("pivot_wider", &.{ .{ .array = aa }, .{ .array = n1 } }, pivot(&.{ "a", "b" }));
    defer o1.release();
    try expectFields(o1, 0, &.{ "a", "b" }, &.{ 1, null });

    // Integer keys match the parsed names; a name that does not parse is rejected.
    var ints = try common.makeArray(i8, allocator, &.{ -1, 2 });
    defer ints.release();
    var iv = try common.makeArray(f64, allocator, &.{ 5, 6 });
    defer iv.release();
    var oi = try ctx.invokeAggregate("pivot_wider", &.{ .{ .array = ints }, .{ .array = iv } }, pivot(&.{ "2", "-1" }));
    defer oi.release();
    try expectFields(oi, 0, &.{ "2", "-1" }, &.{ 6, 5 });
    try std.testing.expectError(error.InvalidOptions, ctx.invokeAggregate("pivot_wider", &.{ .{ .array = ints }, .{ .array = iv } }, pivot(&.{"x"})));
    try std.testing.expectError(error.InvalidOptions, ctx.invokeAggregate("pivot_wider", &.{ .{ .array = ints }, .{ .array = iv } }, pivot(&.{ "2", "02" })));
    // Binary keys, string values.
    var bk = try common.makeStrings(.large_binary, allocator, &.{ "a", "b" });
    defer bk.release();
    var sv = try common.makeStrings(.string, allocator, &.{ "x", "y" });
    defer sv.release();
    var os = try ctx.invokeAggregate("pivot_wider", &.{ .{ .array = bk }, .{ .array = sv } }, pivot(&.{ "b", "a" }));
    defer os.release();
    var field_b = try common.core.datumStructField(.{ .array = os.scalar.payload.? }, 0);
    defer field_b.release();
    const strs = try common.collectStrings(allocator, field_b);
    defer allocator.free(strs);
    try common.expectStrings(&.{"y"}, strs);
    // Chunked inputs: the rows of every chunk ([['a'], ['b', 'a']] with [[1], [2, 3]]: 'a' twice).
    var ck_arr = try common.makeStrings(.string, allocator, &.{ "a", "b", "a" });
    defer ck_arr.release();
    var ck = try common.chunkedAt(allocator, ck_arr, &.{1});
    defer ck.release();
    var cv_arr = try common.makeArray(f64, allocator, &.{ 1, 2, 3 });
    defer cv_arr.release();
    var cv = try common.chunkedAt(allocator, cv_arr, &.{2});
    defer cv.release();
    try std.testing.expectError(error.InvalidInput, ctx.invokeAggregate("pivot_wider", &.{ .{ .chunked = ck }, .{ .chunked = cv } }, pivot(&.{ "a", "b" })));
    var ck2 = try ck.slice(allocator, 0, 2);
    defer ck2.release();
    var cv2 = try cv.slice(allocator, 0, 2);
    defer cv2.release();
    var oc = try ctx.invokeAggregate("pivot_wider", &.{ .{ .chunked = ck2 }, .{ .chunked = cv2 } }, pivot(&.{ "a", "b" }));
    defer oc.release();
    try expectFields(oc, 0, &.{ "a", "b" }, &.{ 1, 2 });
    // No rows: every field null.
    var ek = try k3.slice(0, 0);
    defer ek.release();
    var ev = try v3.slice(0, 0);
    defer ev.release();
    var oe = try ctx.invokeAggregate("pivot_wider", &.{ .{ .array = ek }, .{ .array = ev } }, pivot(&.{"a"}));
    defer oe.release();
    try expectFields(oe, 0, &.{"a"}, &.{null});
    // Null fields of a fixed-width value type still take their width: 8 fields of 2^30 bytes
    // from zero rows is past the 4 GiB output bound.
    const bufs = [_]common.SharedBuffer{ .empty, .empty };
    var wide = try ArrayRef.fromBorrowed(allocator, .{ .data_type = .{ .fixed_size_binary = .{ .byte_width = 1 << 30 } }, .length = 0, .null_count = 0, .buffers = &bufs });
    defer wide.release();
    try std.testing.expectError(error.InvalidInput, ctx.invokeAggregate("pivot_wider", &.{ .{ .array = ek }, .{ .array = wide } }, pivot(&.{ "a", "b", "c", "d", "e", "f", "g", "h" })));
    // Key types without a kernel (Arrow: no dictionary or float keys).
    try std.testing.expectError(error.NoMatchingKernel, ctx.invokeAggregate("pivot_wider", &.{ .{ .array = iv }, .{ .array = iv } }, pivot(&.{"1"})));
}

test "hash_pivot_wider per group and the update / merge lifecycle (pyarrow 26 group_by)" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    var keys = try common.makeStrings(.string, allocator, &.{ "h", "w", "w", "zz" });
    defer keys.release();
    var vals = try common.makeArray(f64, allocator, &.{ 1, 2, 4, 9 });
    defer vals.release();
    var ids = try common.makeArray(u32, allocator, &.{ 0, 0, 1, 1 });
    defer ids.release();
    const hw = [_][]const u8{ "h", "w" };
    const args = [_]Datum{ .{ .array = keys }, .{ .array = vals }, .{ .array = ids } };
    var out = try ctx.invoke("hash_pivot_wider", .hash_aggregate, &args, pivot(&hw));
    defer out.release();
    try std.testing.expectEqual(@as(usize, 2), out.array.data().length);
    try expectFields(out, 0, &hw, &.{ 1, 2 });
    try expectFields(out, 1, &hw, &.{ null, 4 });
    try std.testing.expectError(error.InvalidInput, ctx.invoke("hash_pivot_wider", .hash_aggregate, &args, .{ .pivot_wider = .{ .key_names = &hw, .unexpected_key_behavior = .raise } }));
    // One group with 'w' twice.
    var same = try common.makeArray(u32, allocator, &.{ 0, 0, 0, 0 });
    defer same.release();
    try std.testing.expectError(error.InvalidInput, ctx.invoke("hash_pivot_wider", .hash_aggregate, &.{ .{ .array = keys }, .{ .array = vals }, .{ .array = same } }, pivot(&hw)));
    // The lifecycle: two states over halves, merged.
    var a = try ctx.beginAggregate("hash_pivot_wider", &args, pivot(&hw));
    defer a.deinit();
    var b = try ctx.beginAggregate("hash_pivot_wider", &args, pivot(&hw));
    defer b.deinit();
    var k0 = try keys.slice(0, 2);
    defer k0.release();
    var v0 = try vals.slice(0, 2);
    defer v0.release();
    var ids0 = try ids.slice(0, 2);
    defer ids0.release();
    var k1 = try keys.slice(2, 2);
    defer k1.release();
    var v1 = try vals.slice(2, 2);
    defer v1.release();
    var ids1 = try ids.slice(2, 2);
    defer ids1.release();
    try a.update(&.{ .{ .array = k0 }, .{ .array = v0 }, .{ .array = ids0 } });
    try b.update(&.{ .{ .array = k1 }, .{ .array = v1 }, .{ .array = ids1 } });
    try a.merge(&b);
    var merged = try a.finalize();
    defer merged.release();
    try expectFields(merged, 0, &hw, &.{ 1, 2 });
    try expectFields(merged, 1, &hw, &.{ null, 4 });
    // Group ids must be below the rows of a one-shot call.
    var far = try common.makeArray(u32, allocator, &.{ 0, 0, 0, 9 });
    defer far.release();
    try std.testing.expectError(error.InvalidInput, ctx.invoke("hash_pivot_wider", .hash_aggregate, &.{ .{ .array = keys }, .{ .array = vals }, .{ .array = far } }, pivot(&hw)));
}
