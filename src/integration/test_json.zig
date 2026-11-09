//! Tests of the integration JSON reader / writer (src/integration/json.zig, json_writer.zig) and
//! the golden check over the arrow-testing files in src/ipc/testdata.
const std = @import("std");
const json = @import("json.zig");
const compare = @import("compare.zig");
const golden = @import("golden.zig");
const array_mod = @import("../array/array.zig");
const datatype = @import("../datatype.zig");

const testing = std.testing;
const Document = json.Document;

/// Reads `text`, checks every column, writes it back, reads that and expects equal batches,
/// then the same through zarrow's IPC stream and file writers and readers.
fn roundTrip(text: []const u8) !Document {
    return roundTripOpts(text, true);
}

fn roundTripOpts(text: []const u8, with_ipc: bool) !Document {
    var doc = try json.read(testing.allocator, text);
    errdefer doc.deinit();
    for (doc.batches) |*b| for (b.columns) |c| try c.data().validateFull();
    var out: std.Io.Writer.Allocating = .init(testing.allocator);
    defer out.deinit();
    try json.write(testing.allocator, &out.writer, doc.schema().*, doc.batches);
    var again = try json.read(testing.allocator, out.written());
    defer again.deinit();
    var r: golden.Result = .{};
    if (!golden.same(&r, "JSON round trip", &doc, &again)) {
        std.debug.print("{s}\n{s}\n", .{ r.message(), out.written() });
        return error.TestUnexpectedResult;
    }
    if (with_ipc) for ([_]bool{ false, true }) |file| {
        const bytes = try golden.writeIpc(testing.allocator, &doc, if (file) .file else .stream);
        defer testing.allocator.free(bytes);
        var back = if (file) try golden.readFile(testing.allocator, bytes) else try golden.readStream(testing.allocator, bytes);
        defer back.deinit();
        if (!golden.same(&r, "IPC round trip", &doc, &back)) {
            std.debug.print("{s}\n", .{r.message()});
            return error.TestUnexpectedResult;
        }
    };
    return doc;
}

fn column(doc: *const Document, batch: usize, col: usize) *const array_mod.ArrayData {
    return doc.batches[batch].columns[col].data();
}

const primitives =
    \\{"schema": {"fields": [
    \\  {"name": "n", "nullable": true, "type": {"name": "null"}, "children": []},
    \\  {"name": "b", "nullable": true, "type": {"name": "bool"}, "children": []},
    \\  {"name": "i8", "nullable": true, "type": {"name": "int", "isSigned": true, "bitWidth": 8}, "children": []},
    \\  {"name": "u16", "nullable": false, "type": {"name": "int", "isSigned": false, "bitWidth": 16}, "children": []},
    \\  {"name": "i32", "nullable": true, "type": {"name": "int", "isSigned": true, "bitWidth": 32}, "children": []},
    \\  {"name": "u32", "nullable": true, "type": {"name": "int", "isSigned": false, "bitWidth": 32}, "children": []},
    \\  {"name": "i64", "nullable": true, "type": {"name": "int", "isSigned": true, "bitWidth": 64}, "children": []},
    \\  {"name": "u64", "nullable": true, "type": {"name": "int", "isSigned": false, "bitWidth": 64}, "children": []},
    \\  {"name": "f16", "nullable": true, "type": {"name": "floatingpoint", "precision": "HALF"}, "children": []},
    \\  {"name": "f32", "nullable": true, "type": {"name": "floatingpoint", "precision": "SINGLE"}, "children": []},
    \\  {"name": "f64", "nullable": true, "type": {"name": "floatingpoint", "precision": "DOUBLE"}, "children": []}
    \\]},
    \\"batches": [
    \\ {"count": 3, "columns": [
    \\  {"name": "n", "count": 3},
    \\  {"name": "b", "count": 3, "VALIDITY": [1, 0, 1], "DATA": [true, true, false]},
    \\  {"name": "i8", "count": 3, "VALIDITY": [1, 1, 0], "DATA": [-128, 127, 999]},
    \\  {"name": "u16", "count": 3, "VALIDITY": [1, 1, 1], "DATA": [0, 65535, 7]},
    \\  {"name": "i32", "count": 3, "VALIDITY": [0, 1, 1], "DATA": [0, -2147483648, 2147483647]},
    \\  {"name": "u32", "count": 3, "VALIDITY": [1, 1, 1], "DATA": [4294967295, 0, 1]},
    \\  {"name": "i64", "count": 3, "VALIDITY": [1, 1, 0], "DATA": ["-9223372036854775808", "9223372036854775807", "0"]},
    \\  {"name": "u64", "count": 3, "VALIDITY": [1, 1, 1], "DATA": ["18446744073709551615", "0", "42"]},
    \\  {"name": "f16", "count": 3, "VALIDITY": [1, 1, 0], "DATA": [15360, 31744, 0]},
    \\  {"name": "f32", "count": 3, "VALIDITY": [1, 1, 1], "DATA": [0.1, -870.842, "NaN"]},
    \\  {"name": "f64", "count": 3, "VALIDITY": [1, 1, 1], "DATA": [1e300, "-Infinity", -0.0]}
    \\ ]},
    \\ {"count": 0, "columns": [
    \\  {"name": "n", "count": 0}, {"name": "b", "count": 0, "VALIDITY": [], "DATA": []},
    \\  {"name": "i8", "count": 0, "VALIDITY": [], "DATA": []}, {"name": "u16", "count": 0, "VALIDITY": [], "DATA": []},
    \\  {"name": "i32", "count": 0, "VALIDITY": [], "DATA": []}, {"name": "u32", "count": 0, "VALIDITY": [], "DATA": []},
    \\  {"name": "i64", "count": 0, "VALIDITY": [], "DATA": []}, {"name": "u64", "count": 0, "VALIDITY": [], "DATA": []},
    \\  {"name": "f16", "count": 0, "VALIDITY": [], "DATA": []}, {"name": "f32", "count": 0, "VALIDITY": [], "DATA": []},
    \\  {"name": "f64", "count": 0, "VALIDITY": [], "DATA": []}
    \\ ]}
    \\]}
;

test "integration json reads and round-trips null, bool, integer and float columns" {
    var doc = try roundTrip(primitives);
    defer doc.deinit();
    try testing.expectEqual(@as(usize, 2), doc.batches.len);
    try testing.expectEqual(@as(usize, 0), doc.batches[1].numRows());
    try testing.expect(column(&doc, 0, 0).isNull(2));
    try testing.expect(column(&doc, 0, 1).isNull(1));
    const i8s = array_mod.Int8Array{ .data = column(&doc, 0, 2) };
    try testing.expectEqual(@as(i8, -128), try i8s.value(0));
    try testing.expect(i8s.isNull(2)); // its out-of-range value is a null slot's, never parsed
    try testing.expectEqual(@as(i64, std.math.minInt(i64)), try (array_mod.Int64Array{ .data = column(&doc, 0, 6) }).value(0));
    try testing.expectEqual(@as(u64, std.math.maxInt(u64)), try (array_mod.UInt64Array{ .data = column(&doc, 0, 7) }).value(0));
    // float16 is its bit pattern: 15360 = 1.0, 31744 = +inf.
    try testing.expectEqual(@as(f16, 1.0), try (array_mod.HalfFloatArray{ .data = column(&doc, 0, 8) }).value(0));
    const f32s = array_mod.Float32Array{ .data = column(&doc, 0, 9) };
    try testing.expectEqual(@as(f32, -870.842), try f32s.value(1));
    try testing.expect(std.math.isNan(try f32s.value(2)));
    const f64s = array_mod.Float64Array{ .data = column(&doc, 0, 10) };
    try testing.expect(std.math.isNegativeInf(try f64s.value(1)));
    try testing.expect(std.math.signbit(try f64s.value(2)));
}

const temporal =
    \\{"schema": {"fields": [
    \\  {"name": "d32", "nullable": true, "type": {"name": "decimal", "precision": 9, "scale": 2, "bitWidth": 32}, "children": []},
    \\  {"name": "d64", "nullable": true, "type": {"name": "decimal", "precision": 18, "scale": 2, "bitWidth": 64}, "children": []},
    \\  {"name": "d128", "nullable": true, "type": {"name": "decimal", "precision": 38, "scale": -3}, "children": []},
    \\  {"name": "d256", "nullable": true, "type": {"name": "decimal256", "precision": 76, "scale": 5}, "children": []},
    \\  {"name": "date32", "nullable": true, "type": {"name": "date", "unit": "DAY"}, "children": []},
    \\  {"name": "date64", "nullable": true, "type": {"name": "date", "unit": "MILLISECOND"}, "children": []},
    \\  {"name": "t32", "nullable": true, "type": {"name": "time", "unit": "MILLISECOND", "bitWidth": 32}, "children": []},
    \\  {"name": "t64", "nullable": true, "type": {"name": "time", "unit": "NANOSECOND", "bitWidth": 64}, "children": []},
    \\  {"name": "ts", "nullable": true, "type": {"name": "timestamp", "unit": "MICROSECOND", "timezone": "Europe/Paris"}, "children": []},
    \\  {"name": "ts_naive", "nullable": true, "type": {"name": "timestamp", "unit": "SECOND"}, "children": []},
    \\  {"name": "dur", "nullable": true, "type": {"name": "duration", "unit": "NANOSECOND"}, "children": []},
    \\  {"name": "ym", "nullable": true, "type": {"name": "interval", "unit": "YEAR_MONTH"}, "children": []},
    \\  {"name": "dt", "nullable": true, "type": {"name": "interval", "unit": "DAY_TIME"}, "children": []},
    \\  {"name": "mdn", "nullable": true, "type": {"name": "interval", "unit": "MONTH_DAY_NANO"}, "children": []}
    \\]},
    \\"batches": [{"count": 2, "columns": [
    \\  {"name": "d32", "count": 2, "VALIDITY": [1, 0], "DATA": ["-999999999", "junk"]},
    \\  {"name": "d64", "count": 2, "VALIDITY": [1, 1], "DATA": ["123456789012345678", "-1"]},
    \\  {"name": "d128", "count": 2, "VALIDITY": [1, 1], "DATA": ["-99999999999999999999999999999999999999", "0"]},
    \\  {"name": "d256", "count": 2, "VALIDITY": [1, 1], "DATA": ["-5789604461865809771178549250434395392663499233282028201972879200395656481996", "1"]},
    \\  {"name": "date32", "count": 2, "VALIDITY": [1, 1], "DATA": [-719162, 2932896]},
    \\  {"name": "date64", "count": 2, "VALIDITY": [1, 0], "DATA": ["1577836800000", "0"]},
    \\  {"name": "t32", "count": 2, "VALIDITY": [1, 1], "DATA": [0, 86399999]},
    \\  {"name": "t64", "count": 2, "VALIDITY": [1, 1], "DATA": ["0", "86399999999999"]},
    \\  {"name": "ts", "count": 2, "VALIDITY": [1, 1], "DATA": ["-62135596800000000", "253402300799999999"]},
    \\  {"name": "ts_naive", "count": 2, "VALIDITY": [0, 1], "DATA": ["0", "1"]},
    \\  {"name": "dur", "count": 2, "VALIDITY": [1, 1], "DATA": ["-9223372036854775808", "1"]},
    \\  {"name": "ym", "count": 2, "VALIDITY": [1, 1], "DATA": [-2147483648, 13]},
    \\  {"name": "dt", "count": 2, "VALIDITY": [1, 0], "DATA": [{"days": -1, "milliseconds": 86399999}, {}]},
    \\  {"name": "mdn", "count": 2, "VALIDITY": [1, 1], "DATA": [{"months": 1, "days": -2, "nanoseconds": -9223372036854775808}, {"months": 0, "days": 0, "nanoseconds": "5"}]}
    \\]}]}
;

test "integration json reads and round-trips decimal, date, time, timestamp, duration and interval columns" {
    var doc = try roundTrip(temporal);
    defer doc.deinit();
    const sc = doc.schema();
    try testing.expectEqualStrings("Europe/Paris", sc.fields[8].data_type.timestamp.timezone.?);
    try testing.expect(sc.fields[9].data_type.timestamp.timezone == null);
    try testing.expectEqual(@as(i32, -3), sc.fields[2].data_type.decimal128.scale);
    try testing.expect(sc.fields[3].data_type.* == .decimal256);
    try testing.expectEqual(@as(i32, -999999999), try (array_mod.Decimal32Array{ .data = column(&doc, 0, 0) }).value(0));
    try testing.expectEqual(@as(i256, -5789604461865809771178549250434395392663499233282028201972879200395656481996), try (array_mod.Decimal256Array{ .data = column(&doc, 0, 3) }).value(0));
    // day_time: days in the low 32 bits; month_day_nano: months, days, nanoseconds.
    const dt: u64 = @bitCast(try (array_mod.IntervalDayTimeArray{ .data = column(&doc, 0, 12) }).value(0));
    try testing.expectEqual(@as(u32, @bitCast(@as(i32, -1))), @as(u32, @truncate(dt)));
    try testing.expectEqual(@as(u32, 86399999), @as(u32, @truncate(dt >> 32)));
    const mdn: u128 = @bitCast(try (array_mod.IntervalMonthDayNanoArray{ .data = column(&doc, 0, 13) }).value(0));
    try testing.expectEqual(@as(u64, @bitCast(@as(i64, std.math.minInt(i64)))), @as(u64, @truncate(mdn >> 64)));
}

const binaries =
    \\{"schema": {"fields": [
    \\  {"name": "bin", "nullable": true, "type": {"name": "binary"}, "children": []},
    \\  {"name": "lbin", "nullable": true, "type": {"name": "largebinary"}, "children": []},
    \\  {"name": "str", "nullable": true, "type": {"name": "utf8"}, "children": []},
    \\  {"name": "lstr", "nullable": true, "type": {"name": "largeutf8"}, "children": []},
    \\  {"name": "fsb", "nullable": true, "type": {"name": "fixedsizebinary", "byteWidth": 3}, "children": []},
    \\  {"name": "bv", "nullable": true, "type": {"name": "binaryview"}, "children": []},
    \\  {"name": "sv", "nullable": true, "type": {"name": "utf8view"}, "children": []}
    \\]},
    \\"batches": [{"count": 3, "columns": [
    \\  {"name": "bin", "count": 3, "VALIDITY": [1, 0, 1], "OFFSET": [0, 2, 9, 9], "DATA": ["00FF", "ignored", ""]},
    \\  {"name": "lbin", "count": 3, "VALIDITY": [1, 1, 1], "OFFSET": ["0", "1", "1", "4"], "DATA": ["AB", "", "010203"]},
    \\  {"name": "str", "count": 3, "VALIDITY": [1, 1, 0], "OFFSET": [0, 6, 9, 9], "DATA": ["héllo", "€", ""]},
    \\  {"name": "lstr", "count": 3, "VALIDITY": [1, 1, 1], "OFFSET": ["0", "0", "3", "7"], "DATA": ["", "abc", "µµ"]},
    \\  {"name": "fsb", "count": 3, "VALIDITY": [1, 0, 1], "DATA": ["000102", "", "FFFEFD"]},
    \\  {"name": "bv", "count": 3, "VALIDITY": [1, 0, 1],
    \\   "VIEWS": [{"SIZE": 2, "INLINED": "F34D"}, {"SIZE": 0, "INLINED": ""}, {"SIZE": 14, "PREFIX_HEX": "00010203", "BUFFER_INDEX": 1, "OFFSET": 2}],
    \\   "VARIADIC_DATA_BUFFERS": ["", "FFFF000102030405060708090A0B0C0D"]},
    \\  {"name": "sv", "count": 3, "VALIDITY": [1, 1, 1],
    \\   "VIEWS": [{"SIZE": 12, "INLINED": "twelve bytes"}, {"SIZE": 14, "PREFIX_HEX": "74686972", "BUFFER_INDEX": 0, "OFFSET": 0}, {"SIZE": 0, "INLINED": ""}],
    \\   "VARIADIC_DATA_BUFFERS": ["746869727465656E206279746573"]}
    \\]}]}
;

test "integration json reads binary, string, fixed-size binary and view columns" {
    var doc = try roundTrip(binaries);
    defer doc.deinit();
    const bin = array_mod.BinaryArray{ .data = column(&doc, 0, 0) };
    try testing.expectEqualSlices(u8, &.{ 0, 255 }, try bin.value(0));
    try testing.expectEqualStrings("h\u{e9}llo", try (array_mod.StringArray{ .data = column(&doc, 0, 2) }).value(0));
    try testing.expectEqualStrings("\u{b5}\u{b5}", try (array_mod.LargeStringArray{ .data = column(&doc, 0, 3) }).value(2));
    const bv = array_mod.BinaryViewArray{ .data = column(&doc, 0, 5) };
    try testing.expectEqualSlices(u8, &.{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13 }, try bv.value(2));
    try testing.expectEqualStrings("thirteen bytes", try (array_mod.StringViewArray{ .data = column(&doc, 0, 6) }).value(1));
}

const nested =
    \\{"schema": {"fields": [
    \\  {"name": "list", "nullable": true, "type": {"name": "list"}, "children": [
    \\    {"name": "item", "nullable": true, "type": {"name": "int", "isSigned": true, "bitWidth": 32}, "children": []}]},
    \\  {"name": "large_list", "nullable": true, "type": {"name": "largelist"}, "children": [
    \\    {"name": "x", "nullable": false, "type": {"name": "bool"}, "children": []}]},
    \\  {"name": "list_view", "nullable": true, "type": {"name": "listview"}, "children": [
    \\    {"name": "item", "nullable": true, "type": {"name": "utf8"}, "children": []}]},
    \\  {"name": "large_list_view", "nullable": true, "type": {"name": "largelistview"}, "children": [
    \\    {"name": "item", "nullable": true, "type": {"name": "int", "isSigned": false, "bitWidth": 8}, "children": []}]},
    \\  {"name": "fsl", "nullable": true, "type": {"name": "fixedsizelist", "listSize": 2}, "children": [
    \\    {"name": "item", "nullable": true, "type": {"name": "int", "isSigned": true, "bitWidth": 16}, "children": []}]},
    \\  {"name": "struct", "nullable": true, "type": {"name": "struct"}, "children": [
    \\    {"name": "a", "nullable": true, "type": {"name": "int", "isSigned": true, "bitWidth": 64}, "children": []},
    \\    {"name": "a", "nullable": true, "type": {"name": "utf8"}, "children": []}]},
    \\  {"name": "map", "nullable": true, "type": {"name": "map", "keysSorted": true}, "children": [
    \\    {"name": "some_entries", "nullable": false, "type": {"name": "struct"}, "children": [
    \\      {"name": "some_key", "nullable": false, "type": {"name": "utf8"}, "children": []},
    \\      {"name": "some_value", "nullable": true, "type": {"name": "int", "isSigned": true, "bitWidth": 32}, "children": []}]}]},
    \\  {"name": "sparse", "nullable": true, "type": {"name": "union", "mode": "SPARSE", "typeIds": [5, 7]}, "children": [
    \\    {"name": "f1", "nullable": true, "type": {"name": "int", "isSigned": true, "bitWidth": 32}, "children": []},
    \\    {"name": "f2", "nullable": true, "type": {"name": "utf8"}, "children": []}]},
    \\  {"name": "dense", "nullable": true, "type": {"name": "union", "mode": "DENSE", "typeIds": [10, 20]}, "children": [
    \\    {"name": "f1", "nullable": true, "type": {"name": "int", "isSigned": true, "bitWidth": 16}, "children": []},
    \\    {"name": "f2", "nullable": true, "type": {"name": "null"}, "children": []}]},
    \\  {"name": "ree", "nullable": true, "type": {"name": "runendencoded"}, "children": [
    \\    {"name": "run_ends", "nullable": false, "type": {"name": "int", "isSigned": true, "bitWidth": 16}, "children": []},
    \\    {"name": "values", "nullable": true, "type": {"name": "utf8"}, "children": []}]}
    \\]},
    \\"batches": [{"count": 3, "columns": [
    \\  {"name": "list", "count": 3, "VALIDITY": [1, 0, 1], "OFFSET": [0, 2, 2, 3], "children": [
    \\    {"name": "item", "count": 3, "VALIDITY": [1, 0, 1], "DATA": [1, 0, 3]}]},
    \\  {"name": "large_list", "count": 3, "VALIDITY": [1, 1, 1], "OFFSET": ["0", "0", "1", "3"], "children": [
    \\    {"name": "x", "count": 3, "VALIDITY": [1, 1, 1], "DATA": [true, false, true]}]},
    \\  {"name": "list_view", "count": 3, "VALIDITY": [1, 1, 0], "OFFSET": [2, 0, 0], "SIZE": [1, 3, 0], "children": [
    \\    {"name": "item", "count": 3, "VALIDITY": [1, 1, 1], "OFFSET": [0, 1, 2, 3], "DATA": ["a", "b", "c"]}]},
    \\  {"name": "large_list_view", "count": 3, "VALIDITY": [1, 1, 1], "OFFSET": ["0", "0", "1"], "SIZE": ["2", "0", "1"], "children": [
    \\    {"name": "item", "count": 2, "VALIDITY": [1, 1], "DATA": [255, 0]}]},
    \\  {"name": "fsl", "count": 3, "VALIDITY": [1, 0, 1], "children": [
    \\    {"name": "item", "count": 6, "VALIDITY": [1, 1, 0, 0, 1, 1], "DATA": [1, 2, 0, 0, 5, 6]}]},
    \\  {"name": "struct", "count": 3, "VALIDITY": [1, 1, 0], "children": [
    \\    {"name": "a", "count": 3, "VALIDITY": [1, 0, 1], "DATA": ["1", "0", "3"]},
    \\    {"name": "a", "count": 3, "VALIDITY": [1, 1, 1], "OFFSET": [0, 1, 2, 3], "DATA": ["x", "y", "z"]}]},
    \\  {"name": "map", "count": 3, "VALIDITY": [1, 1, 0], "OFFSET": [0, 2, 2, 2], "children": [
    \\    {"name": "some_entries", "count": 2, "VALIDITY": [1, 1], "children": [
    \\      {"name": "some_key", "count": 2, "VALIDITY": [1, 1], "OFFSET": [0, 1, 2], "DATA": ["a", "b"]},
    \\      {"name": "some_value", "count": 2, "VALIDITY": [0, 1], "DATA": [0, 2]}]}]},
    \\  {"name": "sparse", "count": 3, "TYPE_ID": [5, 7, 5], "children": [
    \\    {"name": "f1", "count": 3, "VALIDITY": [1, 0, 0], "DATA": [11, 0, 0]},
    \\    {"name": "f2", "count": 3, "VALIDITY": [0, 1, 0], "OFFSET": [0, 0, 2, 2], "DATA": ["", "hi", ""]}]},
    \\  {"name": "dense", "count": 3, "TYPE_ID": [10, 20, 10], "OFFSET": [0, 0, 1], "children": [
    \\    {"name": "f1", "count": 2, "VALIDITY": [1, 1], "DATA": [-1, 1]},
    \\    {"name": "f2", "count": 1}]},
    \\  {"name": "ree", "count": 3, "children": [
    \\    {"name": "run_ends", "count": 2, "VALIDITY": [1, 1], "DATA": [2, 3]},
    \\    {"name": "values", "count": 2, "VALIDITY": [1, 0], "OFFSET": [0, 3, 3], "DATA": ["abc", ""]}]}
    \\]}]}
;

test "integration json reads and round-trips list, list-view, fixed-size list, struct, map, union and run-end encoded columns" {
    var doc = try roundTrip(nested);
    defer doc.deinit();
    const map_t = doc.schema().fields[6].data_type.map;
    try testing.expectEqualStrings("some_entries", map_t.entries_name);
    try testing.expectEqualStrings("some_key", map_t.key_field.name);
    try testing.expect(map_t.keys_sorted);
    // Duplicate field names in a struct are kept.
    const st = doc.schema().fields[5].data_type.struct_;
    try testing.expectEqualStrings(st.fields[0].name, st.fields[1].name);
    try testing.expectEqualSlices(i8, &.{ 5, 7 }, doc.schema().fields[7].data_type.sparse_union.type_ids);
}

test "integration json writes sliced batches as their logical rows" {
    var doc = try roundTrip(nested);
    defer doc.deinit();
    var sliced = try doc.batches[0].slice(1, 2);
    defer sliced.deinit();
    var out: std.Io.Writer.Allocating = .init(testing.allocator);
    defer out.deinit();
    try json.write(testing.allocator, &out.writer, doc.schema().*, &.{sliced});
    var back = try json.read(testing.allocator, out.written());
    defer back.deinit();
    var why: compare.Mismatch = .{};
    if (!compare.batchesEqual(&sliced, &back.batches[0], &why)) {
        std.debug.print("{s}\n", .{why.message()});
        return error.TestUnexpectedResult;
    }
}

const dictionaries =
    \\{"schema": {"fields": [
    \\  {"name": "d", "nullable": true, "type": {"name": "utf8"}, "children": [],
    \\   "dictionary": {"id": 3, "indexType": {"name": "int", "isSigned": true, "bitWidth": 16}, "isOrdered": true}},
    \\  {"name": "shared", "nullable": true, "type": {"name": "utf8"}, "children": [],
    \\   "dictionary": {"id": 3, "indexType": {"name": "int", "isSigned": true, "bitWidth": 8}, "isOrdered": false}},
    \\  {"name": "nested", "nullable": true, "type": {"name": "list"}, "children": [
    \\    {"name": "inner", "nullable": true, "type": {"name": "utf8"}, "children": [],
    \\     "dictionary": {"id": 3, "indexType": {"name": "int", "isSigned": true, "bitWidth": 32}, "isOrdered": false}}],
    \\   "dictionary": {"id": 9, "indexType": {"name": "int", "isSigned": true, "bitWidth": 64}, "isOrdered": false}}
    \\]},
    \\"dictionaries": [
    \\  {"id": 3, "data": {"count": 2, "columns": [{"name": "DICT3", "count": 2, "VALIDITY": [1, 1], "OFFSET": [0, 1, 2], "DATA": ["a", "b"]}]}},
    \\  {"id": 3, "isDelta": true, "data": {"count": 1, "columns": [{"name": "DICT3", "count": 1, "VALIDITY": [0], "OFFSET": [0, 0], "DATA": [""]}]}},
    \\  {"id": 9, "data": {"count": 2, "columns": [{"name": "DICT9", "count": 2, "VALIDITY": [1, 1], "OFFSET": [0, 2, 3], "children": [
    \\    {"name": "inner", "count": 3, "VALIDITY": [1, 1, 1], "DATA": [0, 1, 2]}]}]}}
    \\],
    \\"batches": [
    \\ {"count": 2, "columns": [
    \\  {"name": "d", "count": 2, "VALIDITY": [1, 0], "DATA": [2, 77]},
    \\  {"name": "shared", "count": 2, "VALIDITY": [1, 1], "DATA": [1, 0]},
    \\  {"name": "nested", "count": 2, "VALIDITY": [1, 1], "DATA": ["1", "0"]}]},
    \\ {"count": 1, "columns": [
    \\  {"name": "d", "count": 1, "VALIDITY": [1], "DATA": [0]},
    \\  {"name": "shared", "count": 1, "VALIDITY": [0], "DATA": [0]},
    \\  {"name": "nested", "count": 1, "VALIDITY": [1], "DATA": ["1"]}]}
    \\]}
;

test "integration json resolves shared, nested and delta dictionaries" {
    var doc = try roundTrip(dictionaries);
    defer doc.deinit();
    const d = column(&doc, 0, 0);
    // The delta (one null value) was appended to dictionary 3.
    try testing.expectEqual(@as(usize, 3), d.dictionary.?.data().length);
    try testing.expect(d.dictionary.?.data().isNull(2));
    try testing.expect(d.data_type.dictionary.ordered);
    // Shared by `shared` and the values of `nested`'s dictionary.
    const nested_values = column(&doc, 0, 2).dictionary.?.data().children[0].data();
    try testing.expect(nested_values.dictionary.?.node == column(&doc, 0, 1).dictionary.?.node);
}

const extensions =
    \\{"schema": {"fields": [
    \\  {"name": "uuid", "nullable": true, "type": {"name": "fixedsizebinary", "byteWidth": 16}, "children": [],
    \\   "metadata": [{"key": "ARROW:extension:name", "value": "arrow.uuid"}, {"key": "ARROW:extension:metadata", "value": ""}]},
    \\  {"name": "custom", "nullable": true, "type": {"name": "int", "isSigned": true, "bitWidth": 32}, "children": [],
    \\   "metadata": [{"key": "k", "value": "v"}, {"key": "ARROW:extension:name", "value": "example.custom"}, {"key": "ARROW:integration:allow_unregistered_extension", "value": "true"}]},
    \\  {"name": "dict_ext", "nullable": true, "type": {"name": "utf8"}, "children": [],
    \\   "dictionary": {"id": 0, "indexType": {"name": "int", "isSigned": true, "bitWidth": 8}, "isOrdered": false},
    \\   "metadata": [{"key": "ARROW:extension:name", "value": "dict-extension"}, {"key": "ARROW:extension:metadata", "value": "dict-extension-serialized"}]}
    \\], "metadata": [{"key": "schema-level", "value": "é"}, {"key": "schema-level", "value": "duplicate keys are allowed"}]},
    \\"dictionaries": [{"id": 0, "data": {"count": 1, "columns": [{"name": "DICT0", "count": 1, "VALIDITY": [1], "OFFSET": [0, 1], "DATA": ["x"]}]}}],
    \\"batches": [{"count": 1, "columns": [
    \\  {"name": "uuid", "count": 1, "VALIDITY": [1], "DATA": ["000102030405060708090A0B0C0D0E0F"]},
    \\  {"name": "custom", "count": 1, "VALIDITY": [1], "DATA": [7]},
    \\  {"name": "dict_ext", "count": 1, "VALIDITY": [1], "DATA": [0]}]}]}
;

test "integration json reads extension types, field and schema metadata" {
    var doc = try roundTrip(extensions);
    defer doc.deinit();
    const f = doc.schema().fields;
    try testing.expectEqualStrings("arrow.uuid", f[0].data_type.extension.name);
    try testing.expectEqualStrings("", f[0].data_type.extension.metadata.?);
    try testing.expect(f[0].metadata == null);
    try testing.expectEqualStrings("example.custom", f[1].data_type.extension.name);
    try testing.expect(f[1].data_type.extension.metadata == null);
    try testing.expectEqual(@as(usize, 2), f[1].metadata.?.len);
    // Extension over a dictionary: the dictionary's values are the extension (as IPC reads it).
    try testing.expectEqualStrings("dict-extension", f[2].data_type.dictionary.value_type.extension.name);
    try testing.expectEqual(@as(usize, 2), doc.schema().metadata.?.len);
}

test "integration json reads schemas without batches" {
    var doc = try roundTrip(
        \\{"schema": {"fields": [{"name": "a", "nullable": true, "type": {"name": "utf8"}, "children": []}]}, "batches": []}
    );
    defer doc.deinit();
    try testing.expectEqual(@as(usize, 0), doc.batches.len);
    // A batch without columns keeps its row count, as the IPC readers do.
    var none = try roundTripOpts(
        \\{"schema": {"fields": []}, "batches": [{"count": 4, "columns": []}]}
    , false);
    defer none.deinit();
    try testing.expectEqual(@as(usize, 4), none.batches[0].numRows());
}

test "integration json rejects malformed documents" {
    const field =
        \\{"name": "a", "nullable": true, "type": {"name": "int", "isSigned": true, "bitWidth": 8}, "children": []}
    ;
    const Case = struct { []const u8, anyerror };
    const cases = [_]Case{
        .{ "", error.InvalidJson },
        .{ "{\"schema\": {\"fields\": [", error.InvalidJson },
        .{ "[]", error.InvalidDocument },
        .{ "{\"batches\": []}", error.InvalidDocument },
        .{ "{\"schema\": {\"fields\": []}}", error.InvalidDocument },
        .{ "{\"schema\": {\"fields\": [" ++ field ++ "]}, \"batches\": [{\"count\": 2, \"columns\": [{\"name\": \"a\", \"count\": 2, \"VALIDITY\": [1], \"DATA\": [1, 2]}]}]}", error.LengthMismatch },
        .{ "{\"schema\": {\"fields\": [" ++ field ++ "]}, \"batches\": [{\"count\": 1, \"columns\": [{\"name\": \"a\", \"count\": 2, \"VALIDITY\": [1, 1], \"DATA\": [1, 2]}]}]}", error.LengthMismatch },
        .{ "{\"schema\": {\"fields\": [" ++ field ++ "]}, \"batches\": [{\"count\": 1, \"columns\": [{\"name\": \"a\", \"count\": 1, \"VALIDITY\": [1], \"DATA\": [128]}]}]}", error.InvalidValue },
        .{ "{\"schema\": {\"fields\": [" ++ field ++ "]}, \"batches\": [{\"count\": -1, \"columns\": []}]}", error.InvalidValue },
        .{ "{\"schema\": {\"fields\": [" ++ field ++ "]}, \"batches\": [{\"count\": 1, \"columns\": [{\"name\": \"a\", \"count\": 1, \"VALIDITY\": [1], \"DATA\": [1.5]}]}]}", error.InvalidValue },
        .{ "{\"schema\": {\"fields\": [{\"name\": \"a\", \"nullable\": true, \"type\": {\"name\": \"float128\"}, \"children\": []}]}, \"batches\": []}", error.UnknownType },
        .{ "{\"schema\": {\"fields\": [{\"name\": \"a\", \"nullable\": true, \"type\": {\"name\": \"int\", \"isSigned\": true, \"bitWidth\": 12}, \"children\": []}]}, \"batches\": []}", error.UnknownType },
        .{ "{\"schema\": {\"fields\": [{\"name\": \"a\", \"nullable\": true, \"type\": {\"name\": \"decimal\", \"precision\": 39, \"scale\": 0}, \"children\": []}]}, \"batches\": []}", error.UnknownType },
        .{ "{\"schema\": {\"fields\": [{\"name\": \"a\", \"nullable\": true, \"type\": {\"name\": \"list\"}, \"children\": []}]}, \"batches\": []}", error.UnknownType },
        .{ "{\"schema\": {\"fields\": [{\"name\": \"a\", \"nullable\": true, \"type\": {\"name\": \"binary\"}, \"children\": []}]}, \"batches\": [{\"count\": 1, \"columns\": [{\"name\": \"a\", \"count\": 1, \"VALIDITY\": [1], \"OFFSET\": [0, 1], \"DATA\": [\"GG\"]}]}]}", error.InvalidValue },
        .{ "{\"schema\": {\"fields\": [{\"name\": \"a\", \"nullable\": true, \"type\": {\"name\": \"binary\"}, \"children\": []}]}, \"batches\": [{\"count\": 1, \"columns\": [{\"name\": \"a\", \"count\": 1, \"VALIDITY\": [1], \"OFFSET\": [0, 1], \"DATA\": [\"ABC\"]}]}]}", error.InvalidValue },
        .{ "{\"schema\": {\"fields\": [{\"name\": \"a\", \"nullable\": true, \"type\": {\"name\": \"utf8\"}, \"children\": []}]}, \"batches\": [{\"count\": 1, \"columns\": [{\"name\": \"a\", \"count\": 1, \"VALIDITY\": [1], \"OFFSET\": [0, 5], \"DATA\": [\"abc\"]}]}]}", error.LengthMismatch },
        .{ "{\"schema\": {\"fields\": [{\"name\": \"a\", \"nullable\": true, \"type\": {\"name\": \"utf8\"}, \"children\": [], \"dictionary\": {\"id\": 1, \"indexType\": {\"name\": \"int\", \"isSigned\": true, \"bitWidth\": 8}, \"isOrdered\": false}}]}, \"batches\": [{\"count\": 1, \"columns\": [{\"name\": \"a\", \"count\": 1, \"VALIDITY\": [1], \"DATA\": [0]}]}]}", error.MissingDictionary },
        .{ "{\"schema\": {\"fields\": []}, \"dictionaries\": [{\"id\": 4, \"data\": {\"count\": 0, \"columns\": []}}], \"batches\": []}", error.MissingDictionary },
        .{ "{\"schema\": {\"fields\": [{\"name\": \"a\", \"nullable\": true, \"type\": {\"name\": \"utf8\"}, \"children\": [], \"dictionary\": {\"id\": 1, \"indexType\": {\"name\": \"int\", \"isSigned\": true, \"bitWidth\": 8}, \"isOrdered\": false}}]}, \"dictionaries\": [{\"id\": 1, \"data\": {\"count\": 1, \"columns\": [{\"name\": \"d\", \"count\": 1, \"VALIDITY\": [1], \"OFFSET\": [0, 1], \"DATA\": [\"x\"]}]}}], \"batches\": [{\"count\": 1, \"columns\": [{\"name\": \"a\", \"count\": 1, \"VALIDITY\": [1], \"DATA\": [1]}]}]}", error.InvalidOffsets },
        // A non-nullable column with a null.
        .{ "{\"schema\": {\"fields\": [{\"name\": \"a\", \"nullable\": false, \"type\": {\"name\": \"bool\"}, \"children\": []}]}, \"batches\": [{\"count\": 1, \"columns\": [{\"name\": \"a\", \"count\": 1, \"VALIDITY\": [0], \"DATA\": [true]}]}]}", error.NullInNonNullableField },
        // Invalid UTF-8 is caught by validateFull.
        .{ "{\"schema\": {\"fields\": [{\"name\": \"a\", \"nullable\": true, \"type\": {\"name\": \"utf8view\"}, \"children\": []}]}, \"batches\": [{\"count\": 1, \"columns\": [{\"name\": \"a\", \"count\": 1, \"VALIDITY\": [1], \"VIEWS\": [{\"SIZE\": 13, \"PREFIX_HEX\": \"FFFFFFFF\", \"BUFFER_INDEX\": 0, \"OFFSET\": 0}], \"VARIADIC_DATA_BUFFERS\": [\"FFFFFFFF000000000000000000\"]}]}]}", error.InvalidUtf8 },
    };
    for (cases, 0..) |c, i| {
        const got = json.read(testing.allocator, c[0]);
        if (got) |doc| {
            var d = doc;
            d.deinit();
            std.debug.print("case {d} was accepted\n", .{i});
            return error.TestUnexpectedResult;
        } else |e| {
            if (e != c[1]) {
                std.debug.print("case {d}: {s}, expected {s}\n", .{ i, @errorName(e), @errorName(c[1]) });
                return error.TestUnexpectedResult;
            }
        }
    }
}

test "integration json rejects nesting deeper than max_depth" {
    var text: std.ArrayList(u8) = .empty;
    defer text.deinit(testing.allocator);
    try text.appendSlice(testing.allocator, "{\"schema\": {\"fields\": [");
    for (0..json.max_depth + 1) |_| try text.appendSlice(testing.allocator, "{\"name\": \"l\", \"nullable\": true, \"type\": {\"name\": \"list\"}, \"children\": [");
    try text.appendSlice(testing.allocator, "{\"name\": \"x\", \"nullable\": true, \"type\": {\"name\": \"null\"}, \"children\": []}");
    for (0..json.max_depth + 1) |_| try text.appendSlice(testing.allocator, "]}");
    try text.appendSlice(testing.allocator, "]}, \"batches\": []}");
    try testing.expectError(error.NestingTooDeep, json.read(testing.allocator, text.items));
}

test "integration json survives every truncation and byte flip of valid documents" {
    // A document cut anywhere, or with one byte changed, is read or rejected; whatever is read
    // passed validateFull and writes back.
    var prng = std.Random.DefaultPrng.init(0x22);
    const random = prng.random();
    for ([_][]const u8{ primitives, temporal, nested, dictionaries, extensions }) |doc_text| {
        var len: usize = 0;
        while (len < doc_text.len) : (len += 1 + random.uintLessThan(usize, 7)) try tryRead(doc_text[0..len]);
        const copy = try testing.allocator.dupe(u8, doc_text);
        defer testing.allocator.free(copy);
        for (0..200) |_| {
            const at = random.uintLessThan(usize, copy.len);
            const old = copy[at];
            copy[at] = "0123456789-\"[]{},:tfnINaSe"[random.uintLessThan(usize, 26)];
            try tryRead(copy);
            copy[at] = old;
        }
    }
}

fn tryRead(text: []const u8) !void {
    var doc = json.read(testing.allocator, text) catch return;
    defer doc.deinit();
    for (doc.batches) |*b| for (b.columns) |c| try c.data().validateFull();
    var out: std.Io.Writer.Allocating = .init(testing.allocator);
    defer out.deinit();
    json.write(testing.allocator, &out.writer, doc.schema().*, doc.batches) catch |e| switch (e) {
        // A valid document may hold values JSON strings cannot (it never does here).
        error.InvalidUtf8 => return,
        else => return e,
    };
    var again = try json.read(testing.allocator, out.written());
    again.deinit();
}

// The arrow-testing integration cases copied into src/ipc/testdata (NOTICE): the golden check
// of tools/integration_runner.zig on each.
const golden_cases = [_]struct { name: []const u8, json_gz: []const u8, file: []const u8, stream: []const u8 }{
    .{ .name = "0.14.1 decimal", .json_gz = @embedFile("../ipc/testdata/at-0.14.1-generated_decimal.json.gz"), .file = @embedFile("../ipc/testdata/at-0.14.1-generated_decimal.arrow_file"), .stream = @embedFile("../ipc/testdata/at-0.14.1-generated_decimal.stream") },
    .{ .name = "0.14.1 primitive_zerolength", .json_gz = @embedFile("../ipc/testdata/at-0.14.1-generated_primitive_zerolength.json.gz"), .file = @embedFile("../ipc/testdata/at-0.14.1-generated_primitive_zerolength.arrow_file"), .stream = @embedFile("../ipc/testdata/at-0.14.1-generated_primitive_zerolength.stream") },
    .{ .name = "cpp-21.0.0 nested_dictionary", .json_gz = @embedFile("../ipc/testdata/at-cpp-21.0.0-generated_nested_dictionary.json.gz"), .file = @embedFile("../ipc/testdata/at-cpp-21.0.0-generated_nested_dictionary.arrow_file"), .stream = @embedFile("../ipc/testdata/at-cpp-21.0.0-generated_nested_dictionary.stream") },
    .{ .name = "cpp-21.0.0 null_trivial", .json_gz = @embedFile("../ipc/testdata/at-cpp-21.0.0-generated_null_trivial.json.gz"), .file = @embedFile("../ipc/testdata/at-cpp-21.0.0-generated_null_trivial.arrow_file"), .stream = @embedFile("../ipc/testdata/at-cpp-21.0.0-generated_null_trivial.stream") },
    .{ .name = "cpp-21.0.0 union", .json_gz = @embedFile("../ipc/testdata/at-cpp-21.0.0-generated_union.json.gz"), .file = @embedFile("../ipc/testdata/at-cpp-21.0.0-generated_union.arrow_file"), .stream = @embedFile("../ipc/testdata/at-cpp-21.0.0-generated_union.stream") },
    .{ .name = "cpp-21.0.0 binary_view", .json_gz = @embedFile("../ipc/testdata/at-cpp-21.0.0-generated_binary_view.json.gz"), .file = @embedFile("../ipc/testdata/at-cpp-21.0.0-generated_binary_view.arrow_file"), .stream = @embedFile("../ipc/testdata/at-cpp-21.0.0-generated_binary_view.stream") },
    .{ .name = "cpp-21.0.0 list_view", .json_gz = @embedFile("../ipc/testdata/at-cpp-21.0.0-generated_list_view.json.gz"), .file = @embedFile("../ipc/testdata/at-cpp-21.0.0-generated_list_view.arrow_file"), .stream = @embedFile("../ipc/testdata/at-cpp-21.0.0-generated_list_view.stream") },
    .{ .name = "cpp-21.0.0 run_end_encoded", .json_gz = @embedFile("../ipc/testdata/at-cpp-21.0.0-generated_run_end_encoded.json.gz"), .file = @embedFile("../ipc/testdata/at-cpp-21.0.0-generated_run_end_encoded.arrow_file"), .stream = @embedFile("../ipc/testdata/at-cpp-21.0.0-generated_run_end_encoded.stream") },
    .{ .name = "cpp-21.0.0 interval_mdn", .json_gz = @embedFile("../ipc/testdata/at-cpp-21.0.0-generated_interval_mdn.json.gz"), .file = @embedFile("../ipc/testdata/at-cpp-21.0.0-generated_interval_mdn.arrow_file"), .stream = @embedFile("../ipc/testdata/at-cpp-21.0.0-generated_interval_mdn.stream") },
    .{ .name = "cpp-21.0.0 map", .json_gz = @embedFile("../ipc/testdata/at-cpp-21.0.0-generated_map.json.gz"), .file = @embedFile("../ipc/testdata/at-cpp-21.0.0-generated_map.arrow_file"), .stream = @embedFile("../ipc/testdata/at-cpp-21.0.0-generated_map.stream") },
    .{ .name = "cpp-21.0.0 map_non_canonical", .json_gz = @embedFile("../ipc/testdata/at-cpp-21.0.0-generated_map_non_canonical.json.gz"), .file = @embedFile("../ipc/testdata/at-cpp-21.0.0-generated_map_non_canonical.arrow_file"), .stream = @embedFile("../ipc/testdata/at-cpp-21.0.0-generated_map_non_canonical.stream") },
    .{ .name = "cpp-21.0.0 extension", .json_gz = @embedFile("../ipc/testdata/at-cpp-21.0.0-generated_extension.json.gz"), .file = @embedFile("../ipc/testdata/at-cpp-21.0.0-generated_extension.arrow_file"), .stream = @embedFile("../ipc/testdata/at-cpp-21.0.0-generated_extension.stream") },
    .{ .name = "cpp-21.0.0 duplicate_fieldnames", .json_gz = @embedFile("../ipc/testdata/at-cpp-21.0.0-generated_duplicate_fieldnames.json.gz"), .file = @embedFile("../ipc/testdata/at-cpp-21.0.0-generated_duplicate_fieldnames.arrow_file"), .stream = @embedFile("../ipc/testdata/at-cpp-21.0.0-generated_duplicate_fieldnames.stream") },
    .{ .name = "cpp-21.0.0 custom_metadata", .json_gz = @embedFile("../ipc/testdata/at-cpp-21.0.0-generated_custom_metadata.json.gz"), .file = @embedFile("../ipc/testdata/at-cpp-21.0.0-generated_custom_metadata.arrow_file"), .stream = @embedFile("../ipc/testdata/at-cpp-21.0.0-generated_custom_metadata.stream") },
    .{ .name = "cpp-21.0.0 datetime", .json_gz = @embedFile("../ipc/testdata/at-cpp-21.0.0-generated_datetime.json.gz"), .file = @embedFile("../ipc/testdata/at-cpp-21.0.0-generated_datetime.arrow_file"), .stream = @embedFile("../ipc/testdata/at-cpp-21.0.0-generated_datetime.stream") },
    .{ .name = "cpp-21.0.0 dictionary_unsigned", .json_gz = @embedFile("../ipc/testdata/at-cpp-21.0.0-generated_dictionary_unsigned.json.gz"), .file = @embedFile("../ipc/testdata/at-cpp-21.0.0-generated_dictionary_unsigned.arrow_file"), .stream = @embedFile("../ipc/testdata/at-cpp-21.0.0-generated_dictionary_unsigned.stream") },
};

test "integration golden check passes on generated_dictionary_unsigned with unsigned index types" {
    // Rejected (InvalidMetadata) before zarrow's IPC reader and writer accepted unsigned dictionary
    // index types; every index is still bounded by its dictionary when a batch is decoded.
    const c = golden_cases[golden_cases.len - 1];
    const text = try golden.gunzip(testing.allocator, c.json_gz);
    defer testing.allocator.free(text);
    const r = golden.check(testing.allocator, text, c.file, c.stream, .accept);
    if (r.outcome != .pass) {
        std.debug.print("{s}\n", .{r.message()});
        return error.TestUnexpectedResult;
    }
    var doc = try golden.readStream(testing.allocator, c.stream);
    defer doc.deinit();
    var unsigned: usize = 0;
    for (doc.schema().fields) |f| {
        if (f.data_type.* == .dictionary and !f.data_type.dictionary.index_type.signed) unsigned += 1;
    }
    try testing.expect(unsigned > 0);
}

test "integration golden check passes on the arrow-testing cases in src/ipc/testdata" {
    for (golden_cases) |c| {
        const text = try golden.gunzip(testing.allocator, c.json_gz);
        defer testing.allocator.free(text);
        const r = golden.check(testing.allocator, text, c.file, c.stream, .accept);
        if (r.outcome != .pass) {
            std.debug.print("{s}: {s}\n", .{ c.name, r.message() });
            return error.TestUnexpectedResult;
        }
    }
}

test "integration golden check reports a case whose IPC files differ from its JSON" {
    // The union case's JSON against the decimal case's IPC files.
    const text = try golden.gunzip(testing.allocator, golden_cases[4].json_gz);
    defer testing.allocator.free(text);
    const r = golden.check(testing.allocator, text, golden_cases[0].file, golden_cases[0].stream, .accept);
    try testing.expectEqual(golden.Outcome.fail, r.outcome);
    try testing.expect(std.mem.startsWith(u8, r.message(), "(a) JSON vs IPC file: schema"));
    // And a case expected to be rejected that is accepted fails too.
    const r2 = golden.check(testing.allocator, text, golden_cases[4].file, golden_cases[4].stream, .reject_known);
    try testing.expectEqual(golden.Outcome.fail, r2.outcome);
}

test "integration compare follows Arrow's type equality" {
    const int32: datatype.DataType = .{ .int32 = {} };
    const utf8: datatype.DataType = .{ .string = {} };
    const a: datatype.DataType = .{ .list = .{ .value_field = .{ .name = "item", .data_type = &int32 } } };
    const b: datatype.DataType = .{ .list = .{ .value_field = .{ .name = "element", .data_type = &int32 } } };
    try testing.expect(compare.typesEqual(a, b, 0)); // list item names do not matter
    try testing.expect(!a.eql(b)); // to zarrow's DataType.eql they do
    const d1: datatype.DataType = .{ .dictionary = .{ .id = 0, .index_type = .{ .bit_width = 8, .signed = true }, .value_type = &utf8 } };
    const d2: datatype.DataType = .{ .dictionary = .{ .id = 5, .index_type = .{ .bit_width = 8, .signed = true }, .value_type = &utf8 } };
    try testing.expect(compare.typesEqual(d1, d2, 0)); // nor dictionary ids
    const s1: datatype.DataType = .{ .struct_ = .{ .fields = &.{.{ .name = "a", .data_type = &int32 }} } };
    const s2: datatype.DataType = .{ .struct_ = .{ .fields = &.{.{ .name = "b", .data_type = &int32 }} } };
    try testing.expect(!compare.typesEqual(s1, s2, 0)); // struct field names do
}

// ---------------------------------------------------------------------------------------------
// Regressions of zarrow IPC bugs the golden suite found (docs/integration-suite.md, "Bugs found").

const ipc_stream_writer = @import("../ipc/stream_writer.zig");
const ipc_file_writer = @import("../ipc/file_writer.zig");

const list_dictionary =
    \\{"schema": {"fields": [
    \\  {"name": "list_dict", "nullable": true, "type": {"name": "list"}, "children": [
    \\    {"name": "str_dict", "nullable": true, "type": {"name": "utf8"}, "children": [],
    \\     "dictionary": {"id": 0, "indexType": {"name": "int", "isSigned": true, "bitWidth": 8}, "isOrdered": false}}],
    \\   "dictionary": {"id": 1, "indexType": {"name": "int", "isSigned": true, "bitWidth": 8}, "isOrdered": false}}
    \\]},
    \\"dictionaries": [
    \\  {"id": 0, "data": {"count": 2, "columns": [{"name": "d0", "count": 2, "VALIDITY": [1, 1], "OFFSET": [0, 1, 2], "DATA": ["a", "b"]}]}},
    \\  {"id": 1, "data": {"count": 2, "columns": [{"name": "d1", "count": 2, "VALIDITY": [1, 1], "OFFSET": [0, 1, 3], "children": [
    \\    {"name": "str_dict", "count": 3, "VALIDITY": [1, 1, 1], "DATA": [1, 0, 1]}]}]}}
    \\],
    \\"batches": [
    \\  {"count": 2, "columns": [{"name": "list_dict", "count": 2, "VALIDITY": [1, 1], "DATA": [1, 0]}]},
    \\  {"count": 1, "columns": [{"name": "list_dict", "count": 1, "VALIDITY": [1], "DATA": [1]}]}
    \\]}
;

test "IPC writers emit a dictionary nested in another dictionary's values before it" {
    // Before the fix the stream writer sent only the outer dictionary (id 1); the reader then
    // failed on its values, which use dictionary 0 (InvalidMetadata).
    var doc = try json.read(testing.allocator, list_dictionary);
    defer doc.deinit();
    for ([_]bool{ false, true }) |file| {
        const bytes = try golden.writeIpc(testing.allocator, &doc, if (file) .file else .stream);
        defer testing.allocator.free(bytes);
        var back = if (file) try golden.readFile(testing.allocator, bytes) else try golden.readStream(testing.allocator, bytes);
        defer back.deinit();
        var r: golden.Result = .{};
        if (!golden.same(&r, "nested dictionary", &doc, &back)) {
            std.debug.print("{s}\n", .{r.message()});
            return error.TestUnexpectedResult;
        }
    }
}

test "IPC file writer writes an unchanged list-valued dictionary once" {
    // Before the fix every batch re-sent dictionary 1 (list values were never found equal), a
    // replacement the file format forbids (pyarrow: "Unsupported dictionary replacement").
    var doc = try json.read(testing.allocator, list_dictionary);
    defer doc.deinit();
    var out: std.Io.Writer.Allocating = .init(testing.allocator);
    defer out.deinit();
    var w = try ipc_file_writer.FileWriter.init(testing.allocator, &out.writer);
    defer w.deinit();
    try w.writeSchema(doc.schema().*);
    try w.writeRecordBatch(doc.batches[0]);
    try testing.expectEqual(@as(usize, 2), w.dictionary_blocks.items.len); // ids 0 and 1
    try w.writeRecordBatch(doc.batches[1]);
    try testing.expectEqual(@as(usize, 2), w.dictionary_blocks.items.len); // nothing re-sent
    try w.writeEnd();
}

test "IPC file writer rejects a dictionary replacement, the stream writer sends it" {
    var a = try json.read(testing.allocator, list_dictionary);
    defer a.deinit();
    const other = try std.mem.replaceOwned(u8, testing.allocator, list_dictionary, "\"DATA\": [\"a\", \"b\"]", "\"DATA\": [\"x\", \"y\"]");
    defer testing.allocator.free(other);
    var b = try json.read(testing.allocator, other);
    defer b.deinit();

    var out: std.Io.Writer.Allocating = .init(testing.allocator);
    defer out.deinit();
    var fw = try ipc_file_writer.FileWriter.init(testing.allocator, &out.writer);
    defer fw.deinit();
    try fw.writeSchema(a.schema().*);
    try fw.writeRecordBatch(a.batches[0]);
    try testing.expectError(error.DictionaryReplacement, fw.writeRecordBatch(b.batches[0]));

    var sout: std.Io.Writer.Allocating = .init(testing.allocator);
    defer sout.deinit();
    var sw = ipc_stream_writer.StreamWriter.init(testing.allocator, &sout.writer);
    defer sw.deinit();
    try sw.writeSchema(a.schema().*);
    try sw.writeRecordBatch(a.batches[0]);
    try sw.writeRecordBatch(b.batches[0]);
    try sw.writeEnd();
    var back = try golden.readStream(testing.allocator, sout.written());
    defer back.deinit();
    var why: compare.Mismatch = .{};
    try testing.expect(compare.batchesEqual(&b.batches[0], &back.batches[1], &why));
}

test "IPC keeps a non-canonical map entries field name" {
    // Before the fix MapType had no entries name and the writer always wrote "entries".
    var doc = try json.read(testing.allocator, nested);
    defer doc.deinit();
    for ([_]bool{ false, true }) |file| {
        const bytes = try golden.writeIpc(testing.allocator, &doc, if (file) .file else .stream);
        defer testing.allocator.free(bytes);
        var back = if (file) try golden.readFile(testing.allocator, bytes) else try golden.readStream(testing.allocator, bytes);
        defer back.deinit();
        const m = back.schema().fields[6].data_type.map;
        try testing.expectEqualStrings("some_entries", m.entries_name);
        try testing.expect(doc.schema().fields[6].eql(back.schema().fields[6]));
    }
}
