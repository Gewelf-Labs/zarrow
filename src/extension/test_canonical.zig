const std = @import("std");
const canonical = @import("canonical.zig");
const datatype = @import("../datatype.zig");
const StreamReader = @import("../ipc/stream_reader.zig").StreamReader;
const StreamWriter = @import("../ipc/stream_writer.zig").StreamWriter;
const c_data = @import("../ffi/c_data.zig");
const ArrayRef = @import("../array/array.zig").ArrayRef;
const ExtensionBuilder = @import("../array/extension_array.zig").ExtensionBuilder;
const string_array = @import("../array/string_array.zig");
const RecordBatch = @import("../record_batch.zig").RecordBatch;
const Schema = @import("../schema.zig").Schema;

const DataType = datatype.DataType;
const Field = datatype.Field;

test "canonical extension metadata: strict JSON validator (RFC 8259, depth and size caps)" {
    const good = [_][]const u8{ "{}", " [ ] ", "0", "-0.5e+3", "\"a\\u00e9\\n\"", "{\"a\":[1,{\"b\":null}],\"c\":true}", "\"é\"", "[false]" };
    for (good) |s| try std.testing.expect(canonical.validJson(s, 8));
    const bad = [_][]const u8{ "", "{", "}", "[1,]", "{\"a\"}", "{\"a\":1,}", "01", "1.", ".5", "1e", "+1", "\"\\x\"", "\"a", "\"\x01\"", "tru", "nul", "[1] 2", "{a:1}", "'a'", "\"\xff\"", "NaN", "\"\\u12g4\"" };
    for (bad) |s| {
        if (canonical.validJson(s, 8)) {
            std.debug.print("accepted {s}\n", .{s});
            return error.TestUnexpectedResult;
        }
    }
    // Depth: 8 levels pass, 9 do not (metadata); values allow 1024.
    try std.testing.expect(canonical.validJson("[[[[[[[[1]]]]]]]]", 8));
    try std.testing.expect(!canonical.validJson("[[[[[[[[[1]]]]]]]]]", 8));
    var deep: [2050]u8 = undefined;
    @memset(deep[0..1025], '[');
    @memset(deep[1025..2050], ']');
    try std.testing.expect(!canonical.validJson(&deep, canonical.max_value_depth));
    try std.testing.expect(canonical.validJson(deep[1..2049], canonical.max_value_depth));
}

fn ext(name: []const u8, storage: *const DataType, md: ?[]const u8) datatype.ExtensionType {
    return .{ .name = name, .storage_type = storage, .metadata = md };
}

test "canonical extension types: storage and metadata checks for every canonical name; unknown names pass" {
    const I = error.InvalidMetadata;
    const i32_t = DataType{ .int32 = {} };
    const fsl6 = DataType{ .fixed_size_list = .{ .value_field = .{ .name = "item", .data_type = &i32_t }, .list_size = 6 } };
    try canonical.validateType(ext("arrow.fixed_shape_tensor", &fsl6, "{\"shape\":[2,3]}"));
    try canonical.validateType(ext("arrow.fixed_shape_tensor", &fsl6, "{\"shape\":[3,2],\"dim_names\":[\"a\",\"b\"],\"permutation\":[1,0]}"));
    try std.testing.expectError(I, canonical.validateType(ext("arrow.fixed_shape_tensor", &fsl6, "{\"shape\":[2,2]}")));
    try std.testing.expectError(I, canonical.validateType(ext("arrow.fixed_shape_tensor", &fsl6, "{\"dim_names\":[\"a\"]}")));
    try std.testing.expectError(I, canonical.validateType(ext("arrow.fixed_shape_tensor", &fsl6, "{\"shape\":[2,3],\"permutation\":[0,0]}")));
    try std.testing.expectError(I, canonical.validateType(ext("arrow.fixed_shape_tensor", &fsl6, "{\"shape\":[2,3],\"dim_names\":[\"a\"]}")));
    try std.testing.expectError(I, canonical.validateType(ext("arrow.fixed_shape_tensor", &fsl6, "{\"shape\":[-1,-6]}")));
    try std.testing.expectError(I, canonical.validateType(ext("arrow.fixed_shape_tensor", &fsl6, "{\"shape\":[2,3],\"shape\":[6]}")));
    try std.testing.expectError(I, canonical.validateType(ext("arrow.fixed_shape_tensor", &i32_t, "{\"shape\":[]}")));
    const p = try canonical.tensorParams("{\"shape\":[2,3,4]}", true, null);
    try std.testing.expectEqualSlices(i64, &.{ 48, 16, 4 }, p.strides(4)[0..3]);

    const list_t = DataType{ .list = .{ .value_field = .{ .name = "item", .data_type = &i32_t } } };
    const shape2 = DataType{ .fixed_size_list = .{ .value_field = .{ .name = "item", .data_type = &i32_t }, .list_size = 2 } };
    const vfields = [_]Field{ .{ .name = "data", .data_type = &list_t }, .{ .name = "shape", .data_type = &shape2 } };
    const vst = DataType{ .struct_ = .{ .fields = &vfields } };
    try canonical.validateType(ext("arrow.variable_shape_tensor", &vst, ""));
    try canonical.validateType(ext("arrow.variable_shape_tensor", &vst, "{\"uniform_shape\":[null,3]}"));
    try std.testing.expectError(I, canonical.validateType(ext("arrow.variable_shape_tensor", &vst, "{\"uniform_shape\":[3]}")));
    try std.testing.expectError(I, canonical.validateType(ext("arrow.variable_shape_tensor", &fsl6, "")));

    const str = DataType{ .string = {} };
    try canonical.validateType(ext("arrow.json", &str, null));
    try canonical.validateType(ext("arrow.json", &str, "{}"));
    try std.testing.expectError(I, canonical.validateType(ext("arrow.json", &str, "[]")));
    try std.testing.expectError(I, canonical.validateType(ext("arrow.json", &i32_t, "")));

    const fsb16 = DataType{ .fixed_size_binary = .{ .byte_width = 16 } };
    const fsb8 = DataType{ .fixed_size_binary = .{ .byte_width = 8 } };
    try canonical.validateType(ext("arrow.uuid", &fsb16, ""));
    try std.testing.expectError(I, canonical.validateType(ext("arrow.uuid", &fsb8, "")));
    try std.testing.expectError(I, canonical.validateType(ext("arrow.uuid", &fsb16, "x")));

    try canonical.validateType(ext("arrow.opaque", &i32_t, "{\"type_name\":\"g\",\"vendor_name\":\"v\",\"future\":1}"));
    try std.testing.expectError(I, canonical.validateType(ext("arrow.opaque", &i32_t, "{\"type_name\":\"g\"}")));
    try std.testing.expectError(I, canonical.validateType(ext("arrow.opaque", &i32_t, "")));

    const i8_t = DataType{ .int8 = {} };
    try canonical.validateType(ext("arrow.bool8", &i8_t, ""));
    try std.testing.expectError(I, canonical.validateType(ext("arrow.bool8", &i32_t, "")));

    const bin = DataType{ .binary = {} };
    const variant_fields = [_]Field{ .{ .name = "metadata", .data_type = &bin, .nullable = false }, .{ .name = "value", .data_type = &bin } };
    const variant = DataType{ .struct_ = .{ .fields = &variant_fields } };
    try canonical.validateType(ext("arrow.parquet.variant", &variant, ""));
    const no_value = DataType{ .struct_ = .{ .fields = variant_fields[0..1] } };
    try std.testing.expectError(I, canonical.validateType(ext("arrow.parquet.variant", &no_value, "")));
    const nullable_md = [_]Field{ .{ .name = "metadata", .data_type = &bin }, .{ .name = "value", .data_type = &bin } };
    const bad_variant = DataType{ .struct_ = .{ .fields = &nullable_md } };
    try std.testing.expectError(I, canonical.validateType(ext("arrow.parquet.variant", &bad_variant, "")));

    const ts_utc = DataType{ .timestamp = .{ .unit = .microsecond, .timezone = "UTC" } };
    const ts_local = DataType{ .timestamp = .{ .unit = .microsecond, .timezone = null } };
    const i16_t = DataType{ .int16 = {} };
    const two_fields = [_]Field{ .{ .name = "timestamp", .data_type = &ts_utc, .nullable = false }, .{ .name = "offset_minutes", .data_type = &i16_t, .nullable = false } };
    const two = DataType{ .struct_ = .{ .fields = &two_fields } };
    try canonical.validateType(ext("arrow.timestamp_with_offset", &two, ""));
    const local_fields = [_]Field{ .{ .name = "timestamp", .data_type = &ts_local, .nullable = false }, .{ .name = "offset_minutes", .data_type = &i16_t, .nullable = false } };
    const local = DataType{ .struct_ = .{ .fields = &local_fields } };
    try std.testing.expectError(I, canonical.validateType(ext("arrow.timestamp_with_offset", &local, "")));

    // Unknown names pass whatever the storage and metadata.
    try canonical.validateType(ext("com.example.anything", &i32_t, "not json"));
    try canonical.validateType(ext("arrow.not_canonical", &i32_t, "{"));
    try std.testing.expectEqual(canonical.Kind.parquet_variant, canonical.kindOf("arrow.parquet.variant").?);

    // Equality by name, storage and metadata (null = empty; arrow.json: {} = empty).
    try std.testing.expect(canonical.eql(ext("arrow.json", &str, null), ext("arrow.json", &str, "{}")));
    try std.testing.expect(!canonical.eql(ext("arrow.json", &str, null), ext("arrow.uuid", &str, null)));
    try std.testing.expect(!canonical.eql(ext("arrow.fixed_shape_tensor", &fsl6, "{\"shape\":[2,3]}"), ext("arrow.fixed_shape_tensor", &fsl6, "{\"shape\":[3,2]}")));
    try std.testing.expect(!canonical.eql(ext("arrow.uuid", &fsb16, ""), ext("arrow.uuid", &fsb8, "")));
}

test "canonical extension types read from pyarrow 26 fixtures: names, accessors, validateFull, round trip; bad storage rejected" {
    const allocator = std.testing.allocator;
    var stream: std.Io.Reader = .fixed(@embedFile("testdata/pyarrow_canonical.arrows"));
    var reader = StreamReader.init(allocator, &stream);
    defer reader.deinit();
    const schema = try reader.readSchema();
    const names = [_][]const u8{ "arrow.uuid", "arrow.bool8", "arrow.json", "arrow.fixed_shape_tensor", "arrow.opaque", "arrow.variable_shape_tensor", "arrow.timestamp_with_offset", "arrow.parquet.variant" };
    try std.testing.expectEqual(names.len, schema.fields.len);
    for (schema.fields, names) |f, n| {
        try std.testing.expect(f.data_type.* == .extension);
        try std.testing.expectEqualStrings(n, f.data_type.extension.name);
        try canonical.validateType(f.data_type.extension);
    }
    const tensor = try canonical.tensorParams(schema.fields[3].data_type.extension.metadata.?, true, null);
    try std.testing.expectEqual(@as(usize, 2), tensor.ndim);
    try std.testing.expectEqualSlices(i64, &.{ 2, 3 }, tensor.shape[0..2]);
    try std.testing.expectEqualSlices(u32, &.{ 1, 0 }, tensor.permutation.?[0..2]);
    try std.testing.expectEqual(@as(usize, 2), tensor.dim_names);
    var t: std.ArrayList(u8) = .empty;
    defer t.deinit(allocator);
    var v: std.ArrayList(u8) = .empty;
    defer v.deinit(allocator);
    try canonical.opaqueNames(schema.fields[4].data_type.extension.metadata.?, &t, &v, allocator);
    try std.testing.expectEqualStrings("geometry", t.items);
    try std.testing.expectEqualStrings("postgis", v.items);

    var batch = (try reader.nextRecordBatch()).?;
    defer batch.deinit();
    for (batch.columns) |c| try c.data().validateFull();
    const uuid = batch.columns[0].data();
    var want: [16]u8 = undefined;
    for (&want, 0..) |*b, i| b.* = @intCast(i);
    try std.testing.expectEqualSlices(u8, &want, &canonical.uuidAt(uuid, 0).?);
    try std.testing.expect(canonical.uuidAt(uuid, 1) == null);
    const b8 = batch.columns[1].data();
    try std.testing.expectEqual(@as(?bool, true), canonical.bool8At(b8, 0));
    try std.testing.expectEqual(@as(?bool, null), canonical.bool8At(b8, 1));
    try std.testing.expectEqual(@as(?bool, false), canonical.bool8At(b8, 2));

    // Written by zarrow and read back: the same types (name, storage, metadata).
    var out = std.Io.Writer.Allocating.init(allocator);
    defer out.deinit();
    var writer = StreamWriter.init(allocator, &out.writer);
    defer writer.deinit();
    try writer.writeSchema(schema);
    try writer.writeRecordBatch(batch);
    try writer.writeEnd();
    var again: std.Io.Reader = .fixed(out.written());
    var reader2 = StreamReader.init(allocator, &again);
    defer reader2.deinit();
    const schema2 = try reader2.readSchema();
    for (schema.fields, schema2.fields) |a, b| try std.testing.expect(canonical.eql(a.data_type.extension, b.data_type.extension));
    var batch2 = (try reader2.nextRecordBatch()).?;
    defer batch2.deinit();
    for (batch2.columns) |c| try c.data().validateFull();

    // A canonical name over the wrong storage, or a shape that is not the list size.
    inline for (.{ "testdata/pyarrow_canonical_bad_uuid.arrows", "testdata/pyarrow_canonical_bad_tensor.arrows" }) |path| {
        var bad: std.Io.Reader = .fixed(@embedFile(path));
        var r = StreamReader.init(allocator, &bad);
        defer r.deinit();
        try std.testing.expectError(error.InvalidMetadata, r.readSchema());
    }
}

test "canonical extension types on C data import: InvalidMetadata for a wrong storage type" {
    const allocator = std.testing.allocator;
    const fsb8 = DataType{ .fixed_size_binary = .{ .byte_width = 8 } };
    const bad = DataType{ .extension = .{ .name = "arrow.uuid", .storage_type = &fsb8 } };
    const fields = [_]Field{.{ .name = "u", .data_type = &bad }};
    var c_schema = try c_data.exportSchema(allocator, .{ .fields = &fields });
    try std.testing.expectError(error.InvalidMetadata, c_data.importSchemaOwned(allocator, &c_schema));
    if (c_schema.release) |release| release(&c_schema);

    const fsb16 = DataType{ .fixed_size_binary = .{ .byte_width = 16 } };
    const good = DataType{ .extension = .{ .name = "arrow.uuid", .storage_type = &fsb16 } };
    const good_fields = [_]Field{.{ .name = "u", .data_type = &good }};
    var c_good = try c_data.exportSchema(allocator, .{ .fields = &good_fields });
    var owned = try c_data.importSchemaOwned(allocator, &c_good);
    defer owned.deinit();
    try std.testing.expectEqualStrings("arrow.uuid", owned.schema.fields[0].data_type.extension.name);
}

test "canonical extension value constraints in validateFull: arrow.json syntax, tensor shapes, offset range" {
    const allocator = std.testing.allocator;
    const str = DataType{ .string = {} };
    var sb = try string_array.StringBuilder.init(allocator, 3, 16);
    defer sb.deinit();
    try sb.append("{\"a\":1}");
    try sb.appendNull();
    try sb.append("{oops}");
    var storage = try sb.finish();
    defer storage.release();
    var eb = try ExtensionBuilder.init(allocator, .{ .name = "arrow.json", .storage_type = &str });
    defer eb.deinit();
    var json = try eb.finish(storage);
    defer json.release();
    try std.testing.expectError(error.InvalidExtensionValue, json.data().validateFull());
    var head = try json.slice(0, 2);
    defer head.release();
    try head.data().validateFull();
    // An unknown name is not looked at.
    var other_b = try ExtensionBuilder.init(allocator, .{ .name = "com.example.json", .storage_type = &str });
    defer other_b.deinit();
    var other = try other_b.finish(storage);
    defer other.release();
    try other.data().validateFull();
    // Tensor shapes and offsets from pyarrow: each column breaks its constraint.
    var stream: std.Io.Reader = .fixed(@embedFile("testdata/pyarrow_canonical_bad_values.arrows"));
    var reader = StreamReader.init(allocator, &stream);
    defer reader.deinit();
    _ = try reader.readSchema();
    var batch = (try reader.nextRecordBatch()).?;
    defer batch.deinit();
    for (batch.columns) |c| {
        try c.data().validateLayout();
        try std.testing.expectError(error.InvalidExtensionValue, c.data().validateFull());
    }
}
