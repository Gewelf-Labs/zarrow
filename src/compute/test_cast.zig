const std = @import("std");
const common = @import("test_common.zig");
const concat = @import("../concat_array_refs.zig");

const ArrayRef = common.ArrayRef;
const DataType = common.DataType;
const Datum = common.Datum;
const ExecContext = common.ExecContext;
const Options = common.Options;
const Scalar = common.Scalar;
const dense = common.dense;
const TypeTag = std.meta.Tag(DataType);

fn to(dt: DataType) Options {
    return .{ .cast = .{ .to_type = dt } };
}

fn unsafeTo(dt: DataType) Options {
    return .{ .cast = .{ .to_type = dt, .safe = false } };
}

const string_type = DataType{ .string = {} };
const int32_type = DataType{ .int32 = {} };
const dict_string = DataType{ .dictionary = .{ .index_type = .{ .bit_width = 32, .signed = true }, .value_type = &string_type } };
const dict_int32 = DataType{ .dictionary = .{ .index_type = .{ .bit_width = 32, .signed = true }, .value_type = &int32_type } };
const date32 = DataType{ .date32 = {} };
const date64 = DataType{ .date64 = {} };
const ts_s = DataType{ .timestamp = .{ .unit = .second, .timezone = null } };
const ts_s_utc = DataType{ .timestamp = .{ .unit = .second, .timezone = "UTC" } };
const ts_ms = DataType{ .timestamp = .{ .unit = .millisecond, .timezone = null } };
const ts_ns = DataType{ .timestamp = .{ .unit = .nanosecond, .timezone = null } };

test "cast hand-written values: numbers safe and unsafe, bool, number formatting and parsing" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);

    var a = try common.makeArray(i32, allocator, &.{ 1, -1, 300, null, std.math.maxInt(i32) });
    defer a.release();
    const args = [_]Datum{.{ .array = a }};
    try common.expectInvokeWith(i64, &ctx, "cast", &args, to(.{ .int64 = {} }), &.{ 1, -1, 300, null, std.math.maxInt(i32) });
    try common.expectInvokeWith(i8, &ctx, "cast", &args, unsafeTo(.{ .int8 = {} }), &.{ 1, -1, 44, null, -1 });
    try common.expectInvokeWith(u8, &ctx, "cast", &args, unsafeTo(.{ .uint8 = {} }), &.{ 1, 255, 44, null, 255 });
    try std.testing.expectError(error.InvalidCast, ctx.invokeScalar("cast", &args, to(.{ .int8 = {} })));
    try std.testing.expectError(error.InvalidCast, ctx.invokeScalar("cast", &args, to(.{ .uint32 = {} })));
    // ExecConfig.safe_cast = false makes every cast of the context unsafe.
    var lenient = ExecContext.initWithConfig(allocator, &registry, .{ .safe_cast = false });
    try common.expectInvokeWith(i8, &lenient, "cast", &args, to(.{ .int8 = {} }), &.{ 1, -1, 44, null, -1 });

    const nan = std.math.nan(f64);
    var f = try common.makeArray(f64, allocator, &.{ 1.5, -2.7, 3.0, nan, 1e20, -1e20, null });
    defer f.release();
    try common.expectInvokeWith(i32, &ctx, "cast", &.{.{ .array = f }}, unsafeTo(int32_type), &.{ 1, -2, 3, 0, std.math.maxInt(i32), std.math.minInt(i32), null });
    try common.expectInvokeWith(u16, &ctx, "cast", &.{.{ .array = f }}, unsafeTo(.{ .uint16 = {} }), &.{ 1, 0, 3, 0, std.math.maxInt(u16), 0, null });
    try std.testing.expectError(error.InvalidCast, ctx.invokeScalar("cast", &.{.{ .array = f }}, to(int32_type)));
    var whole = try common.makeArray(f64, allocator, &.{ 3.0, -4.0, -0.0, null });
    defer whole.release();
    try common.expectInvokeWith(i32, &ctx, "cast", &.{.{ .array = whole }}, to(int32_type), &.{ 3, -4, 0, null });
    try common.expectInvokeWith(f32, &ctx, "cast", &.{.{ .array = f }}, to(.{ .float = {} }), &.{ 1.5, @floatCast(@as(f64, -2.7)), 3.0, std.math.nan(f32), @floatCast(@as(f64, 1e20)), @floatCast(@as(f64, -1e20)), null });
    var huge = try common.makeArray(f64, allocator, &.{1e300});
    defer huge.release();
    try common.expectInvokeWith(f32, &ctx, "cast", &.{.{ .array = huge }}, to(.{ .float = {} }), &.{std.math.inf(f32)});

    // Integers beyond 2^53 do not fit a double exactly.
    var big = try common.makeArray(i64, allocator, &.{ 1 << 53, (1 << 53) + 1, -(1 << 53) });
    defer big.release();
    try std.testing.expectError(error.InvalidCast, ctx.invokeScalar("cast", &.{.{ .array = big }}, to(.{ .double = {} })));
    try common.expectInvokeWith(f64, &ctx, "cast", &.{.{ .array = big }}, unsafeTo(.{ .double = {} }), &.{ 9007199254740992.0, 9007199254740992.0, -9007199254740992.0 });
    try common.expectInvokeWith(f64, &ctx, "cast", &.{.{ .array = a }}, to(.{ .double = {} }), &.{ 1, -1, 300, null, 2147483647 });

    var b = try common.makeArray(bool, allocator, &.{ true, false, null });
    defer b.release();
    try common.expectInvokeWith(i32, &ctx, "cast", &.{.{ .array = b }}, to(int32_type), &.{ 1, 0, null });
    try common.expectInvokeWith(f64, &ctx, "cast", &.{.{ .array = b }}, to(.{ .double = {} }), &.{ 1, 0, null });
    try common.expectInvokeWith(bool, &ctx, "cast", &args, to(.{ .bool = {} }), &.{ true, true, true, null, true });
    var zeros = try common.makeArray(f64, allocator, &.{ 0.0, -0.0, nan, 0.5 });
    defer zeros.release();
    try common.expectInvokeWith(bool, &ctx, "cast", &.{.{ .array = zeros }}, to(.{ .bool = {} }), &.{ false, false, true, true });

    // Numbers to text: integers in plain decimal, floats in shortest round-trip digits laid out
    // like Python's repr without a trailing ".0" (expected values from Python).
    var ints = try common.makeArray(i64, allocator, &.{ 0, -123, std.math.minInt(i64), std.math.maxInt(i64), null });
    defer ints.release();
    try common.expectInvokeStrings(&ctx, "cast", &.{.{ .array = ints }}, to(string_type), &.{ "0", "-123", "-9223372036854775808", "9223372036854775807", null });
    var u = try common.makeArray(u64, allocator, &.{ std.math.maxInt(u64), 7 });
    defer u.release();
    try common.expectInvokeStrings(&ctx, "cast", &.{.{ .array = u }}, to(.{ .string_view = {} }), &.{ "18446744073709551615", "7" });
    var floats = try common.makeArray(f64, allocator, &.{ 0.0, -0.0, 1.0, 1.5, -2.25, 0.1, 1e16, 1e15, 123456789012345.6, 1e-4, 1e-5, 1.5e-7, 9007199254740992.0, 1e300, 5e-324, std.math.inf(f64), -std.math.inf(f64), nan, 1.0 / 3.0, null });
    defer floats.release();
    try common.expectInvokeStrings(&ctx, "cast", &.{.{ .array = floats }}, to(.{ .large_string = {} }), &.{ "0", "-0", "1", "1.5", "-2.25", "0.1", "1e+16", "1000000000000000", "123456789012345.6", "0.0001", "1e-05", "1.5e-07", "9007199254740992", "1e+300", "5e-324", "inf", "-inf", "nan", "0.3333333333333333", null });
    var f32s = try common.makeArray(f32, allocator, &.{ 0.1, 16777216.0, 3.4028235e38, 1e-5 });
    defer f32s.release();
    try common.expectInvokeStrings(&ctx, "cast", &.{.{ .array = f32s }}, to(string_type), &.{ "0.1", "16777216", "3.4028235e+38", "1e-05" });
    try common.expectInvokeStrings(&ctx, "cast", &.{.{ .array = b }}, to(string_type), &.{ "true", "false", null });

    // Text to numbers: an optional sign and digits (floats: fraction, exponent, inf, nan).
    var text = try common.makeStrings(.string, allocator, &.{ "12", "-7", "+3", null, "0" });
    defer text.release();
    try common.expectInvokeWith(i32, &ctx, "cast", &.{.{ .array = text }}, to(int32_type), &.{ 12, -7, 3, null, 0 });
    try common.expectInvokeWith(f64, &ctx, "cast", &.{.{ .array = text }}, to(.{ .double = {} }), &.{ 12, -7, 3, null, 0 });
    var ftext = try common.makeStrings(.large_string, allocator, &.{ "1.5e3", "-inf", "NaN", ".5", "Infinity", "-0.25" });
    defer ftext.release();
    try common.expectInvokeWith(f64, &ctx, "cast", &.{.{ .array = ftext }}, to(.{ .double = {} }), &.{ 1500, -std.math.inf(f64), nan, 0.5, std.math.inf(f64), -0.25 });
    var btext = try common.makeStrings(.string_view, allocator, &.{ "true", "FALSE", "1", "0", null });
    defer btext.release();
    try common.expectInvokeWith(bool, &ctx, "cast", &.{.{ .array = btext }}, to(.{ .bool = {} }), &.{ true, false, true, false, null });
}

/// Reference conversion (unsafe): integers wrap modulo 2^bits, floats truncate and saturate.
fn refConvert(comptime F: type, comptime T: type, x: F) T {
    if (@typeInfo(T) == .float) return if (@typeInfo(F) == .float) @floatCast(x) else @floatFromInt(x);
    if (@typeInfo(F) == .float) {
        if (std.math.isNan(x)) return 0;
        const t = @trunc(x);
        if (t >= @as(F, @floatFromInt(std.math.maxInt(T)))) return std.math.maxInt(T);
        if (t <= @as(F, @floatFromInt(std.math.minInt(T)))) return std.math.minInt(T);
        return @intFromFloat(t);
    }
    const modulus: i128 = @as(i128, 1) << @bitSizeOf(T);
    var m = @mod(@as(i128, x), modulus);
    if (std.math.minInt(T) < 0 and m > std.math.maxInt(T)) m -= modulus;
    return @intCast(m);
}

/// Reference loss check of a safe cast.
fn refLossy(comptime F: type, comptime T: type, x: F) bool {
    if (@typeInfo(T) == .float) {
        if (@typeInfo(F) == .float) return false;
        const limit: i128 = if (T == f32) 1 << 24 else 1 << 53;
        return @as(i128, x) > limit or @as(i128, x) < -limit;
    }
    if (@typeInfo(F) == .float) {
        if (std.math.isNan(x) or @trunc(x) != x) return true;
        const bits = @bitSizeOf(T) - @as(u8, if (std.math.minInt(T) < 0) 1 else 0);
        return x >= std.math.pow(F, 2, bits) or x < @as(F, @floatFromInt(std.math.minInt(T)));
    }
    return std.math.cast(T, x) == null;
}

fn expectCastMatches(comptime F: type, comptime T: type, ctx: *ExecContext, arr: ArrayRef) !void {
    const allocator = std.testing.allocator;
    const rows = try common.collect(F, allocator, .{ .array = arr });
    defer allocator.free(rows);
    const expected = try allocator.alloc(?T, rows.len);
    defer allocator.free(expected);
    var any_lossy = false;
    for (expected, rows) |*e, r| {
        e.* = if (r) |x| refConvert(F, T, x) else null;
        if (r) |x| any_lossy = any_lossy or refLossy(F, T, x);
    }
    const dt = dense.dataTypeOf(T);
    try common.expectInvokeWith(T, ctx, "cast", &.{.{ .array = arr }}, unsafeTo(dt), expected);
    if (any_lossy) {
        try std.testing.expectError(error.InvalidCast, ctx.invokeScalar("cast", &.{.{ .array = arr }}, to(dt)));
    } else {
        try common.expectInvokeWith(T, ctx, "cast", &.{.{ .array = arr }}, to(dt), expected);
    }
}

test "cast oracle matches a per-element loop on random sliced inputs with nulls for every numeric pair" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    var prng = std.Random.DefaultPrng.init(0xCA57);
    const rng = prng.random();
    inline for (dense.numeric_types) |F| {
        // Full-range values (edges included) and small ones: safe succeeds exactly when nothing is lost.
        var a = try common.randomArray(F, allocator, rng, 300, 7, 20, .{});
        defer a.release();
        var small = try common.randomArray(F, allocator, rng, 300, 3, 20, .{ .small = true });
        defer small.release();
        inline for (dense.numeric_types) |T| {
            if (F != T) {
                try expectCastMatches(F, T, &ctx, a);
                try expectCastMatches(F, T, &ctx, small);
            }
        }
        // Through text and back: integers are exact, floats round-trip bit for bit.
        const rows = try common.collect(F, allocator, .{ .array = a });
        defer allocator.free(rows);
        var text = try ctx.invokeScalar("cast", &.{.{ .array = a }}, to(string_type));
        defer text.release();
        try common.expectInvokeWith(F, &ctx, "cast", &.{text}, to(dense.dataTypeOf(F)), rows);
        var b = try ctx.invokeScalar("cast", &.{.{ .array = a }}, to(.{ .bool = {} }));
        defer b.release();
        const brows = try common.collect(bool, allocator, b);
        defer allocator.free(brows);
        for (brows, rows) |got, r| try std.testing.expectEqual(if (r) |x| x != 0 else null, got);
    }
}

test "cast among string, large_string, string_view, binary, large_binary and binary_view checks UTF-8 for string targets" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    const values = [_]?[]const u8{ "a", null, "", "héllo wörld😀 long value", "0123456789ab", null, "z" };
    const both = values ++ values;
    const tags = [_]TypeTag{ .string, .large_string, .string_view, .binary, .large_binary, .binary_view };
    inline for (tags) |from| {
        var full = try common.makeStrings(from, allocator, &both);
        defer full.release();
        var arr = try full.slice(2, values.len + 3);
        defer arr.release();
        const expected = both[2..][0 .. values.len + 3];
        inline for (tags) |target| {
            const dt = @unionInit(DataType, @tagName(target), {});
            try common.expectInvokeStrings(&ctx, "cast", &.{.{ .array = arr }}, to(dt), expected);
            try common.expectInvokeStrings(&ctx, "cast", &.{.{ .array = arr }}, unsafeTo(dt), expected);
        }
    }
    // Invalid UTF-8 in a valid row fails both modes; in a null row it is never looked at.
    inline for (.{ TypeTag.binary, TypeTag.large_binary, TypeTag.binary_view }) |from| {
        var bad = try common.makeStrings(from, allocator, &.{ "ok", "\xff", "0123456789abcdef\xc3" });
        defer bad.release();
        inline for (.{ TypeTag.string, TypeTag.large_string, TypeTag.string_view }) |target| {
            const dt = @unionInit(DataType, @tagName(target), {});
            try std.testing.expectError(error.InvalidCast, ctx.invokeScalar("cast", &.{.{ .array = bad }}, to(dt)));
            try std.testing.expectError(error.InvalidCast, ctx.invokeScalar("cast", &.{.{ .array = bad }}, unsafeTo(dt)));
        }
        const first_valid = [_]u8{0b001};
        var bufs: [3]common.SharedBuffer = undefined;
        bufs[0] = .fromSlice(&first_valid);
        bufs[1] = bad.data().buffers[1];
        bufs[2] = bad.data().buffers[2];
        var masked = try ArrayRef.fromBorrowed(allocator, .{ .data_type = bad.data().data_type, .length = 3, .null_count = 2, .buffers = &bufs });
        defer masked.release();
        try common.expectInvokeStrings(&ctx, "cast", &.{.{ .array = masked }}, to(string_type), &.{ "ok", null, null });
    }
}

test "cast temporal: date32, date64 and timestamp units with truncation and overflow in safe and unsafe mode" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    const day_ms: i64 = 86_400_000;

    var d32 = try common.makeTyped(i32, date32, allocator, &.{ 0, -1, 11016, null });
    defer d32.release();
    try common.expectInvokeWith(i64, &ctx, "cast", &.{.{ .array = d32 }}, to(date64), &.{ 0, -day_ms, 11016 * day_ms, null });
    try common.expectInvokeWith(i64, &ctx, "cast", &.{.{ .array = d32 }}, to(ts_s), &.{ 0, -86_400, 11016 * 86_400, null });
    var far = try common.makeTyped(i32, date32, allocator, &.{std.math.maxInt(i32)});
    defer far.release();
    try std.testing.expectError(error.InvalidCast, ctx.invokeScalar("cast", &.{.{ .array = far }}, to(ts_ns)));
    try common.expectInvokeWith(i64, &ctx, "cast", &.{.{ .array = far }}, unsafeTo(ts_ns), &.{@as(i64, std.math.maxInt(i32)) *% 86_400_000_000_000});

    var d64 = try common.makeTyped(i64, date64, allocator, &.{ 3 * day_ms, 3 * day_ms + 1, -1, null });
    defer d64.release();
    try std.testing.expectError(error.InvalidCast, ctx.invokeScalar("cast", &.{.{ .array = d64 }}, to(date32)));
    try common.expectInvokeWith(i32, &ctx, "cast", &.{.{ .array = d64 }}, unsafeTo(date32), &.{ 3, 3, -1, null });
    try common.expectInvokeWith(i64, &ctx, "cast", &.{.{ .array = d64 }}, unsafeTo(ts_s), &.{ 3 * 86_400, 3 * 86_400, 0, null });

    var ms = try common.makeTyped(i64, ts_ms, allocator, &.{ 1500, -1500, 2 * day_ms + 5, -1, null });
    defer ms.release();
    try std.testing.expectError(error.InvalidCast, ctx.invokeScalar("cast", &.{.{ .array = ms }}, to(ts_s)));
    // A finer to a coarser unit truncates toward zero (Arrow C++); to a date it floors.
    try common.expectInvokeWith(i64, &ctx, "cast", &.{.{ .array = ms }}, unsafeTo(ts_s), &.{ 1, -1, 2 * 86_400, 0, null });
    try common.expectInvokeWith(i64, &ctx, "cast", &.{.{ .array = ms }}, to(ts_ns), &.{ 1_500_000_000, -1_500_000_000, (2 * day_ms + 5) * 1_000_000, -1_000_000, null });
    try common.expectInvokeWith(i32, &ctx, "cast", &.{.{ .array = ms }}, to(date32), &.{ 0, -1, 2, -1, null });
    try common.expectInvokeWith(i64, &ctx, "cast", &.{.{ .array = ms }}, to(date64), &.{ 0, -day_ms, 2 * day_ms, -day_ms, null });

    var secs = try common.makeTyped(i64, ts_s, allocator, &.{ std.math.maxInt(i64) / 10, 7 });
    defer secs.release();
    try std.testing.expectError(error.InvalidCast, ctx.invokeScalar("cast", &.{.{ .array = secs }}, to(ts_ns)));
    try common.expectInvokeWith(i64, &ctx, "cast", &.{.{ .array = secs }}, unsafeTo(ts_ns), &.{ @as(i64, std.math.maxInt(i64) / 10) *% 1_000_000_000, 7_000_000_000 });
    try std.testing.expectError(error.InvalidCast, ctx.invokeScalar("cast", &.{.{ .array = secs }}, to(date32)));
    // A timezone change keeps the instants.
    var utc = try ctx.invokeScalar("cast", &.{.{ .array = secs }}, to(ts_s_utc));
    defer utc.release();
    try std.testing.expect(utc.array.data().data_type.eql(ts_s_utc));
    const utc_rows = try common.collect(i64, allocator, utc);
    defer allocator.free(utc_rows);
    try common.expectRows(i64, &.{ std.math.maxInt(i64) / 10, 7 }, utc_rows);
}

test "cast dictionary decode and encode, null type, identity, chunked and scalar inputs" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);

    var s = try common.makeStrings(.string, allocator, &.{ "a", "b", "a", null, "c" });
    defer s.release();
    var encoded = try ctx.invokeScalar("cast", &.{.{ .array = s }}, to(dict_string));
    defer encoded.release();
    try encoded.array.data().validateFull();
    try std.testing.expectEqual(@as(usize, 3), encoded.array.data().dictionary.?.data().length);
    var sliced = try encoded.array.slice(1, 4);
    defer sliced.release();
    try common.expectInvokeStrings(&ctx, "cast", &.{.{ .array = sliced }}, to(string_type), &.{ "b", "a", null, "c" });
    try common.expectInvokeStrings(&ctx, "cast", &.{.{ .array = sliced }}, to(.{ .large_binary = {} }), &.{ "b", "a", null, "c" });
    var ints = try common.makeArray(i32, allocator, &.{ 5, null, 5, -3 });
    defer ints.release();
    var dict_ints = try concat.dictionaryEncode(allocator, ints, &int32_type);
    defer dict_ints.release();
    try std.testing.expect(dict_ints.data().data_type.eql(dict_int32));
    try common.expectInvokeWith(i64, &ctx, "cast", &.{.{ .array = dict_ints }}, to(.{ .int64 = {} }), &.{ 5, null, 5, -3 });
    try common.expectInvokeStrings(&ctx, "cast", &.{.{ .array = dict_ints }}, to(string_type), &.{ "5", null, "5", "-3" });
    // An index past the dictionary (never validated) is rejected, not read.
    const bad_index = [_]i32{ 0, 9 };
    const bufs = [_]common.SharedBuffer{ .empty, .fromSlice(std.mem.sliceAsBytes(&bad_index)) };
    var bad = try ArrayRef.fromBorrowed(allocator, .{ .data_type = dict_int32, .length = 2, .null_count = 0, .buffers = &bufs, .dictionary = dict_ints.data().dictionary });
    defer bad.release();
    try std.testing.expectError(error.InvalidInput, ctx.invokeScalar("cast", &.{.{ .array = bad }}, to(int32_type)));

    // The null type casts to all-null arrays.
    var nulls = try common.core.datumBuildNullLikeWithAllocator(allocator, .{ .null = {} }, 3);
    defer nulls.release();
    try common.expectInvokeWith(i32, &ctx, "cast", &.{nulls}, to(int32_type), &.{ null, null, null });
    try common.expectInvokeStrings(&ctx, "cast", &.{nulls}, to(.{ .string_view = {} }), &.{ null, null, null });

    // Identity returns the input itself.
    var same = try ctx.invokeScalar("cast", &.{.{ .array = ints }}, to(int32_type));
    defer same.release();
    try std.testing.expectEqual(ints.node, same.array.node);

    // Chunked in, chunked out with the same boundaries.
    var prng = std.Random.DefaultPrng.init(0xCA58);
    var r = try common.randomArray(i32, allocator, prng.random(), 200, 0, 20, .{});
    defer r.release();
    var chunked = try common.chunkedAt(allocator, r, &.{ 3, 100 });
    defer chunked.release();
    var c64 = try ctx.invokeScalar("cast", &.{.{ .chunked = chunked }}, to(.{ .int64 = {} }));
    defer c64.release();
    try std.testing.expectEqual(@as(usize, 3), c64.chunked.numChunks());
    const want = try common.collect(i32, allocator, .{ .array = r });
    defer allocator.free(want);
    const got = try common.collect(i64, allocator, c64);
    defer allocator.free(got);
    for (want, got) |w, g| try std.testing.expectEqual(if (w) |x| @as(?i64, x) else null, g);
    var ctext = try ctx.invokeScalar("cast", &.{.{ .chunked = chunked }}, to(string_type));
    defer ctext.release();
    try std.testing.expectEqual(@as(usize, 3), ctext.chunked.numChunks());

    // Scalars.
    var s16 = try ctx.invokeScalar("cast", &.{.{ .scalar = common.scalarOf(i32, 300) }}, to(.{ .int16 = {} }));
    defer s16.release();
    try std.testing.expectEqual(@as(i16, 300), s16.scalar.value.i16);
    var stext = try ctx.invokeScalar("cast", &.{.{ .scalar = common.scalarOf(i32, -42) }}, to(string_type));
    defer stext.release();
    try std.testing.expectEqualStrings("-42", stext.scalar.value.string);
    var snum = try ctx.invokeScalar("cast", &.{stext}, to(.{ .int8 = {} }));
    defer snum.release();
    try std.testing.expectEqual(@as(i8, -42), snum.scalar.value.i8);
    var sdate = try ctx.invokeScalar("cast", &.{.{ .scalar = Scalar.init(ts_ms, .{ .timestamp = 2 * 86_400_000 + 1 }) }}, to(date32));
    defer sdate.release();
    try std.testing.expectEqual(@as(i32, 2), sdate.scalar.value.date32);
    var snull = try ctx.invokeScalar("cast", &.{.{ .scalar = common.scalarOf(i32, null) }}, to(string_type));
    defer snull.release();
    try std.testing.expect(snull.scalar.isNull() and snull.scalar.data_type == .string);
    try std.testing.expectError(error.InvalidCast, ctx.invokeScalar("cast", &.{.{ .scalar = common.scalarOf(i32, 300) }}, to(.{ .int8 = {} })));
}

test "cast errors: unsupported pairs, options, unparsable text, mismatched scalars and malformed input" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    var i = try common.makeArray(i32, allocator, &.{1});
    defer i.release();
    const args = [_]Datum{.{ .array = i }};
    // int32 casts to date32 / time32 only (Arrow's zero-copy casts), not to date64.
    try std.testing.expectError(error.UnsupportedType, ctx.invokeScalar("cast", &args, to(.{ .date64 = {} })));
    try std.testing.expectError(error.UnsupportedType, ctx.invokeScalar("cast", &args, to(.{ .binary = {} })));
    // int32 to decimal needs `10 + scale` digits of precision (Arrow: "should be at least 12").
    try std.testing.expectError(error.InvalidCast, ctx.invokeScalar("cast", &args, to(.{ .decimal128 = .{ .precision = 11, .scale = 2 } })));
    try std.testing.expectError(error.UnsupportedType, ctx.invokeScalar("cast", &.{.{ .scalar = Scalar.init(.{ .bool = {} }, .{ .bool = true }) }}, to(.{ .decimal128 = .{ .precision = 10, .scale = 2 } })));
    try std.testing.expectError(error.UnsupportedType, ctx.invokeScalar("cast", &args, to(dict_string)));
    const ordered = DataType{ .dictionary = .{ .index_type = .{ .bit_width = 8, .signed = true }, .value_type = &int32_type, .ordered = true } };
    try std.testing.expectError(error.UnsupportedType, ctx.invokeScalar("cast", &args, to(ordered)));
    var bin = try common.makeStrings(.binary, allocator, &.{"1"});
    defer bin.release();
    try std.testing.expectError(error.UnsupportedType, ctx.invokeScalar("cast", &.{.{ .array = bin }}, to(int32_type)));
    try std.testing.expectError(error.InvalidOptions, ctx.invokeScalar("cast", &args, .{ .cast = .{} }));
    try std.testing.expectError(error.InvalidOptions, ctx.invokeScalar("cast", &args, common.none));
    try std.testing.expectError(error.InvalidArity, ctx.invokeScalar("cast", &.{ args[0], args[0] }, to(string_type)));

    // Text that does not parse or fit is InvalidCast in both modes (Arrow C++).
    inline for (.{ " 1", "1_000", "0x10", "", "-", "1.5", "300", "abc" }) |t| {
        var s = try common.makeStrings(.string, allocator, &.{ "1", t });
        defer s.release();
        try std.testing.expectError(error.InvalidCast, ctx.invokeScalar("cast", &.{.{ .array = s }}, to(.{ .int8 = {} })));
        try std.testing.expectError(error.InvalidCast, ctx.invokeScalar("cast", &.{.{ .array = s }}, unsafeTo(.{ .int8 = {} })));
    }
    inline for (.{ "1e", "1,5", "0x1p3", "nan2", "1_0" }) |t| {
        var s = try common.makeStrings(.string, allocator, &.{t});
        defer s.release();
        try std.testing.expectError(error.InvalidCast, ctx.invokeScalar("cast", &.{.{ .array = s }}, to(.{ .double = {} })));
    }
    var yes = try common.makeStrings(.string, allocator, &.{"yes"});
    defer yes.release();
    try std.testing.expectError(error.InvalidCast, ctx.invokeScalar("cast", &.{.{ .array = yes }}, to(.{ .bool = {} })));
    try std.testing.expectError(error.InvalidCast, ctx.invokeScalar("cast", &.{.{ .array = yes }}, to(.{ .uint64 = {} })));
    // A scalar whose payload does not match its type is rejected, not reinterpreted.
    const mismatched = Scalar.init(int32_type, .{ .f32 = 1.5 });
    try std.testing.expectError(error.InvalidInput, ctx.invokeScalar("cast", &.{.{ .scalar = mismatched }}, to(.{ .int64 = {} })));
    // A values buffer shorter than the array fails validateLayout instead of being read.
    const short = [_]i32{1};
    const bufs = [_]common.SharedBuffer{ .empty, .fromSlice(std.mem.sliceAsBytes(&short)) };
    var malformed = try ArrayRef.fromBorrowed(allocator, .{ .data_type = int32_type, .length = 2, .null_count = 0, .buffers = &bufs });
    defer malformed.release();
    try std.testing.expectError(error.InvalidInput, ctx.invokeScalar("cast", &.{.{ .array = malformed }}, to(.{ .int64 = {} })));
}

const large_string_type = DataType{ .large_string = {} };
const int64_type = DataType{ .int64 = {} };

fn dictOf(comptime bits: u16, comptime signed: bool, value: *const DataType) DataType {
    return .{ .dictionary = .{ .index_type = .{ .bit_width = bits, .signed = signed }, .value_type = value } };
}

test "cast times, durations, timestamp to time of day, integer reinterpretation and dictionary index widths (Arrow C++ results)" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    const t32s = DataType{ .time32 = .{ .unit = .second } };
    const t32ms = DataType{ .time32 = .{ .unit = .millisecond } };
    const t64us = DataType{ .time64 = .{ .unit = .microsecond } };
    const t64ns = DataType{ .time64 = .{ .unit = .nanosecond } };
    const dur_s = DataType{ .duration = .{ .unit = .second } };
    const dur_ms = DataType{ .duration = .{ .unit = .millisecond } };

    // Time units (pyarrow 26): a coarser unit that drops a fraction fails when safe, truncates when not.
    var tu = try common.makeTyped(i64, t64us, allocator, &.{ 1500, -1, null });
    defer tu.release();
    try std.testing.expectError(error.InvalidCast, ctx.invokeScalar("cast", &.{.{ .array = tu }}, to(t32ms)));
    try common.expectInvokeWith(i32, &ctx, "cast", &.{.{ .array = tu }}, unsafeTo(t32ms), &.{ 1, 0, null });
    var tm = try common.makeTyped(i32, t32ms, allocator, &.{1500});
    defer tm.release();
    try common.expectInvokeWith(i64, &ctx, "cast", &.{.{ .array = tm }}, to(t64ns), &.{1500000000});
    try common.expectInvokeWith(i32, &ctx, "cast", &.{.{ .array = tm }}, unsafeTo(t32s), &.{1});

    // A timestamp to its time of day (floored to the day first).
    var ts = try common.makeTyped(i64, ts_ms, allocator, &.{ 90061001, -1 });
    defer ts.release();
    try std.testing.expectError(error.InvalidCast, ctx.invokeScalar("cast", &.{.{ .array = ts }}, to(t32s)));
    try common.expectInvokeWith(i32, &ctx, "cast", &.{.{ .array = ts }}, unsafeTo(t32s), &.{ 3661, 86399 });
    var ts_whole = try common.makeTyped(i64, ts_ms, allocator, &.{ 90061000, -1000 });
    defer ts_whole.release();
    try common.expectInvokeWith(i32, &ctx, "cast", &.{.{ .array = ts_whole }}, to(t32s), &.{ 3661, 86399 });
    try common.expectInvokeWith(i64, &ctx, "cast", &.{.{ .array = ts }}, to(t64us), &.{ 3661001000, 86399999000 });
    // A timezone other than UTC gives the local time of day and date (1969-12-31 19:00, pyarrow 26).
    var ny = try common.makeTyped(i64, .{ .timestamp = .{ .unit = .second, .timezone = "America/New_York" } }, allocator, &.{0});
    defer ny.release();
    try common.expectInvokeWith(i32, &ctx, "cast", &.{.{ .array = ny }}, to(t32s), &.{68400});
    try common.expectInvokeWith(i32, &ctx, "cast", &.{.{ .array = ny }}, to(date32), &.{-1});
    var utc = try common.makeTyped(i64, ts_s_utc, allocator, &.{86401});
    defer utc.release();
    try common.expectInvokeWith(i32, &ctx, "cast", &.{.{ .array = utc }}, to(t32s), &.{1});

    // Durations: loss and overflow fail when safe; unsafe truncates and wraps.
    var dm = try common.makeTyped(i64, dur_ms, allocator, &.{ 1500, -2 });
    defer dm.release();
    try std.testing.expectError(error.InvalidCast, ctx.invokeScalar("cast", &.{.{ .array = dm }}, to(dur_s)));
    try common.expectInvokeWith(i64, &ctx, "cast", &.{.{ .array = dm }}, unsafeTo(dur_s), &.{ 1, 0 });
    var dbig = try common.makeTyped(i64, dur_s, allocator, &.{1 << 62});
    defer dbig.release();
    try std.testing.expectError(error.InvalidCast, ctx.invokeScalar("cast", &.{.{ .array = dbig }}, to(dur_ms)));
    try common.expectInvokeWith(i64, &ctx, "cast", &.{.{ .array = dbig }}, unsafeTo(dur_ms), &.{0});

    // Same-width integers keep the stored values, both ways (the buffers are shared).
    var i32s = try common.makeArray(i32, allocator, &.{ 90000, null, -1 });
    defer i32s.release();
    var as_time = try ctx.invokeScalar("cast", &.{.{ .array = i32s }}, to(t32s));
    defer as_time.release();
    try std.testing.expect(as_time.array.data().data_type.eql(t32s));
    try std.testing.expectEqual(i32s.data().buffers[1].data.ptr, as_time.array.data().buffers[1].data.ptr);
    try common.expectInvokeWith(i32, &ctx, "cast", &.{as_time}, to(int32_type), &.{ 90000, null, -1 });
    try common.expectInvokeWith(i32, &ctx, "cast", &.{.{ .array = i32s }}, to(date32), &.{ 90000, null, -1 });
    var i64s = try common.makeArray(i64, allocator, &.{ 5, null });
    defer i64s.release();
    inline for (.{ ts_ns, date64, t64ns, dur_ms }) |dt| {
        var cast_out = try ctx.invokeScalar("cast", &.{.{ .array = i64s }}, to(dt));
        defer cast_out.release();
        try std.testing.expect(cast_out.array.data().data_type.eql(dt));
        try common.expectInvokeWith(i64, &ctx, "cast", &.{cast_out}, to(int64_type), &.{ 5, null });
    }
    // Pairs Arrow does not cast.
    for ([_]struct { from: ArrayRef, to: DataType }{
        .{ .from = i64s, .to = date32 },
        .{ .from = i64s, .to = t32s },
        .{ .from = i32s, .to = ts_s },
        .{ .from = tm, .to = ts_ms },
        .{ .from = dm, .to = t64us },
        .{ .from = tm, .to = int64_type },
    }) |c| try std.testing.expectError(error.UnsupportedType, ctx.invokeScalar("cast", &.{.{ .array = c.from }}, to(c.to)));
    var d32 = try common.makeTyped(i32, date32, allocator, &.{1});
    defer d32.release();
    try std.testing.expectError(error.UnsupportedType, ctx.invokeScalar("cast", &.{.{ .array = d32 }}, to(t32s)));

    // Dictionary targets of every index width; too many values for the width fail.
    var s = try common.makeStrings(.string, allocator, &.{ "a", "b", null, "a" });
    defer s.release();
    inline for (.{ dictOf(8, true, &string_type), dictOf(16, false, &string_type), dictOf(64, true, &string_type), dictOf(8, false, &string_type) }) |dt| {
        var enc = try ctx.invokeScalar("cast", &.{.{ .array = s }}, to(dt));
        defer enc.release();
        try enc.array.data().validateFull();
        try std.testing.expect(enc.array.data().data_type.eql(dt));
        try common.expectInvokeStrings(&ctx, "cast", &.{enc}, to(string_type), &.{ "a", "b", null, "a" });
    }
    var many_rows: [300]?[]const u8 = undefined;
    var names: [300][4]u8 = undefined;
    for (&many_rows, &names, 0..) |*r, *n, k| r.* = std.fmt.bufPrint(n, "{d}", .{k}) catch unreachable;
    var many = try common.makeStrings(.string, allocator, &many_rows);
    defer many.release();
    try std.testing.expectError(error.InvalidCast, ctx.invokeScalar("cast", &.{.{ .array = many }}, to(dictOf(8, true, &string_type))));
    var wide_enough = try ctx.invokeScalar("cast", &.{.{ .array = many }}, to(dictOf(16, true, &string_type)));
    wide_enough.release();
    // A dictionary to another index and value type: the values are cast first (pyarrow 26).
    var dict = try ctx.invokeScalar("cast", &.{.{ .array = s }}, to(dict_string));
    defer dict.release();
    const target = dictOf(64, true, &large_string_type);
    var redone = try ctx.invokeScalar("cast", &.{dict}, to(target));
    defer redone.release();
    try std.testing.expect(redone.array.data().data_type.eql(target));
    try common.expectInvokeStrings(&ctx, "cast", &.{redone}, to(large_string_type), &.{ "a", "b", null, "a" });
    // A plain array to a dictionary of another value type is not cast (Arrow: unsupported).
    var ints = try common.makeArray(i32, allocator, &.{ 1, 2 });
    defer ints.release();
    try std.testing.expectError(error.UnsupportedType, ctx.invokeScalar("cast", &.{.{ .array = ints }}, to(dictOf(16, true, &int64_type))));
}

// ------------------------------------------------------------------------------ nested casts

const int8_type = DataType{ .int8 = {} };
const item32 = common.datatype.Field{ .name = "item", .data_type = &int32_type };
const item64 = common.datatype.Field{ .name = "item", .data_type = &int64_type };
const item8 = common.datatype.Field{ .name = "item", .data_type = &int8_type };
const item_str = common.datatype.Field{ .name = "item", .data_type = &string_type };
const list32 = DataType{ .list = .{ .value_field = item32 } };
const list64 = DataType{ .list = .{ .value_field = item64 } };
const large32 = DataType{ .large_list = .{ .value_field = item32 } };
const large8 = DataType{ .large_list = .{ .value_field = item8 } };
const view32 = DataType{ .list_view = .{ .value_field = item32 } };
const large_view64 = DataType{ .large_list_view = .{ .value_field = item64 } };
const fsl32_2 = DataType{ .fixed_size_list = .{ .value_field = item32, .list_size = 2 } };
const fsl64_2 = DataType{ .fixed_size_list = .{ .value_field = item64, .list_size = 2 } };
const fsl32_3 = DataType{ .fixed_size_list = .{ .value_field = item32, .list_size = 3 } };
const list_str = DataType{ .list = .{ .value_field = item_str } };

/// A list-like array of `Builder` over `values` with row lengths `lens` (null: a null row).
fn listArray(comptime Builder: type, allocator: std.mem.Allocator, field: common.datatype.Field, lens: []const ?usize, values: ArrayRef) !ArrayRef {
    var b = try Builder.init(allocator, lens.len, field);
    defer b.deinit();
    for (lens) |l| if (l) |n| try b.appendLen(n) else try b.appendNull();
    return b.finish(values);
}

/// The rows of a list-like array of `T` values (null rows null).
fn expectListRows(comptime T: type, arr: ArrayRef, expected: []const ?[]const ?T) !void {
    const allocator = std.testing.allocator;
    const d = arr.data();
    try d.validateFull();
    try std.testing.expectEqual(expected.len, d.length);
    const child = try common.collect(T, allocator, .{ .array = d.children[0] });
    defer allocator.free(child);
    for (expected, 0..) |e, i| {
        if (e == null) {
            try std.testing.expect(d.isNull(i));
            continue;
        }
        try std.testing.expect(!d.isNull(i));
        const start: usize, const len: usize = switch (d.data_type) {
            .list, .map => blk: {
                const o = (try d.typedBufferAt(1, i32))[d.offset + i ..];
                break :blk .{ @intCast(o[0]), @intCast(o[1] - o[0]) };
            },
            .large_list => blk: {
                const o = (try d.typedBufferAt(1, i64))[d.offset + i ..];
                break :blk .{ @intCast(o[0]), @intCast(o[1] - o[0]) };
            },
            .list_view => .{ @intCast((try d.typedBufferAt(1, i32))[d.offset + i]), @intCast((try d.typedBufferAt(2, i32))[d.offset + i]) },
            .large_list_view => .{ @intCast((try d.typedBufferAt(1, i64))[d.offset + i]), @intCast((try d.typedBufferAt(2, i64))[d.offset + i]) },
            .fixed_size_list => |f| .{ (d.offset + i) * @as(usize, @intCast(f.list_size)), @intCast(f.list_size) },
            else => return error.TestUnexpectedResult,
        };
        try common.expectRows(T, e.?, child[start .. start + len]);
    }
}

fn castTo(ctx: *ExecContext, arr: ArrayRef, dt: DataType, safe: bool) !ArrayRef {
    var out = try ctx.invokeScalar("cast", &.{.{ .array = arr }}, if (safe) to(dt) else unsafeTo(dt));
    errdefer out.release();
    try std.testing.expect(out.array.data().data_type.eql(dt));
    return out.array;
}

test "cast nested lists: list, large_list, list_view and fixed_size_list among themselves, the child cast recursively (pyarrow 26 results)" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);

    // [[1, 2], null, [], [300]]
    var vals = try common.makeArray(i32, allocator, &.{ 1, 2, 300 });
    defer vals.release();
    var l = try listArray(common.array_mod.ListBuilder, allocator, item32, &.{ 2, null, 0, 1 }, vals);
    defer l.release();
    const rows = [_]?[]const ?i64{ &.{ 1, 2 }, null, &.{}, &.{300} };
    inline for (.{ list64, large_view64 }) |dt| {
        var out = try castTo(&ctx, l, dt, true);
        defer out.release();
        try expectListRows(i64, out, &rows);
    }
    inline for (.{ large32, view32 }) |dt| {
        var out = try castTo(&ctx, l, dt, true);
        defer out.release();
        try expectListRows(i32, out, &.{ &.{ 1, 2 }, null, &.{}, &.{300} });
    }
    try std.testing.expectError(error.InvalidCast, ctx.invokeScalar("cast", &.{.{ .array = l }}, to(large8)));
    var wrapped = try castTo(&ctx, l, large8, false);
    defer wrapped.release();
    try expectListRows(i8, wrapped, &.{ &.{ 1, 2 }, null, &.{}, &.{44} });
    var text = try castTo(&ctx, l, list_str, true);
    defer text.release();
    const text_child = try common.collectStrings(allocator, .{ .array = text.data().children[0] });
    defer allocator.free(text_child);
    try common.expectStrings(&.{ "1", "2", "300" }, text_child);
    // To fixed_size_list every valid row needs the size; between fixed sizes they must match.
    try std.testing.expectError(error.InvalidCast, ctx.invokeScalar("cast", &.{.{ .array = l }}, to(fsl32_2)));
    try std.testing.expectError(error.UnsupportedType, ctx.invokeScalar("cast", &.{.{ .array = l }}, to(string_type)));
    const nested_list = DataType{ .list = .{ .value_field = .{ .name = "item", .data_type = &list32 } } };
    try std.testing.expectError(error.UnsupportedType, ctx.invokeScalar("cast", &.{.{ .array = l }}, to(nested_list)));
    // Sliced input: the child window is rebased.
    var tail = try l.slice(1, 3);
    defer tail.release();
    var tail_out = try castTo(&ctx, tail, large32, true);
    defer tail_out.release();
    try expectListRows(i32, tail_out, &.{ null, &.{}, &.{300} });

    // [[1, 2], null, [3, 4]] as list (the null row of length 0 gets two null values) and as
    // fixed_size_list (to list, large_list and fixed_size_list<int64>, not to another size).
    var four = try common.makeArray(i32, allocator, &.{ 1, 2, 3, 4 });
    defer four.release();
    var pairs = try listArray(common.array_mod.ListBuilder, allocator, item32, &.{ 2, null, 2 }, four);
    defer pairs.release();
    var as_fsl = try castTo(&ctx, pairs, fsl32_2, true);
    defer as_fsl.release();
    try expectListRows(i32, as_fsl, &.{ &.{ 1, 2 }, null, &.{ 3, 4 } });
    var fsl_vals = try common.makeArray(i32, allocator, &.{ 1, 2, 0, 0, 3, 4 });
    defer fsl_vals.release();
    var fb = try common.array_mod.FixedSizeListBuilder.init(allocator, item32, 2);
    defer fb.deinit();
    try fb.appendValid();
    try fb.appendNull();
    try fb.appendValid();
    var fsl = try fb.finish(fsl_vals);
    defer fsl.release();
    inline for (.{ list64, fsl64_2 }) |dt| {
        var out = try castTo(&ctx, fsl, dt, true);
        defer out.release();
        try expectListRows(i64, out, &.{ &.{ 1, 2 }, null, &.{ 3, 4 } });
    }
    var fsl_sliced = try fsl.slice(1, 2);
    defer fsl_sliced.release();
    var fsl_large = try castTo(&ctx, fsl_sliced, large32, true);
    defer fsl_large.release();
    try expectListRows(i32, fsl_large, &.{ null, &.{ 3, 4 } });
    try std.testing.expectError(error.UnsupportedType, ctx.invokeScalar("cast", &.{.{ .array = fsl }}, to(fsl32_3)));

    // list_view rows out of order: [[3], null, [1, 2]] over [1, 2, 3].
    var lvb = try common.array_mod.ListViewBuilder.init(allocator, 3, item32);
    defer lvb.deinit();
    try lvb.appendLen(2);
    try lvb.appendNull();
    try lvb.appendLen(1);
    var lv = try lvb.finish(vals);
    defer lv.release();
    var from_view = try castTo(&ctx, lv, list64, true);
    defer from_view.release();
    try expectListRows(i64, from_view, &.{ &.{ 1, 2 }, null, &.{300} });

    // Chunked and scalar inputs.
    var chunked = try common.chunkedAt(allocator, l, &.{1});
    defer chunked.release();
    var cout = try ctx.invokeScalar("cast", &.{.{ .chunked = chunked }}, to(large32));
    defer cout.release();
    try std.testing.expectEqual(@as(usize, 2), cout.chunked.numChunks());
    var one = try l.slice(0, 1);
    defer one.release();
    var one_scalar = try Scalar.initNested(list32, one);
    defer one_scalar.release();
    var s = try ctx.invokeScalar("cast", &.{.{ .scalar = one_scalar }}, to(large32));
    defer s.release();
    try std.testing.expect(s == .scalar and s.scalar.data_type == .large_list);
    // The null type to nested types.
    var nulls = try common.datumBuildNullLike(.{ .null = {} }, 2);
    defer nulls.release();
    inline for (.{ list32, fsl32_2 }) |dt| {
        var out = try ctx.invokeScalar("cast", &.{nulls}, to(dt));
        defer out.release();
        try out.array.data().validateFull();
        try std.testing.expectEqual(@as(usize, 2), out.array.data().null_count.?);
    }
    // A list<string> to list<dictionary<int32, string>> and back.
    var words = try common.makeStrings(.string, allocator, &.{ "a", "b", "a" });
    defer words.release();
    var lw = try listArray(common.array_mod.ListBuilder, allocator, item_str, &.{ 2, null, 1 }, words);
    defer lw.release();
    const list_dict = DataType{ .list = .{ .value_field = .{ .name = "item", .data_type = &dict_string } } };
    var encoded = try castTo(&ctx, lw, list_dict, true);
    defer encoded.release();
    try std.testing.expect(encoded.data().children[0].data().data_type == .dictionary);
    var decoded = try castTo(&ctx, encoded, list_str, true);
    defer decoded.release();
    const back = try common.collectStrings(allocator, .{ .array = decoded.data().children[0] });
    defer allocator.free(back);
    try common.expectStrings(&.{ "a", "b", "a" }, back);
    // A dictionary of lists decodes, then casts.
    var dict_of_lists = try common.makeDictionary(allocator, pairs, .{ .bit_width = 32, .signed = true }, &.{ 2, 0, null });
    defer dict_of_lists.release();
    var from_dict = try castTo(&ctx, dict_of_lists, large32, true);
    defer from_dict.release();
    try expectListRows(i32, from_dict, &.{ &.{ 3, 4 }, &.{ 1, 2 }, null });
}

test "cast struct to struct by field name and map to map or list of struct (pyarrow 26 results)" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);

    // struct<a: int32, b: string> [{1, "x"}, null, {3, null}]
    const a = try common.makeArray(i32, allocator, &.{ 1, 0, 3 });
    const b = try common.makeStrings(.string, allocator, &.{ "x", "", null });
    var validity = try dense.bitsBuffer(allocator, 3);
    validity.data[0] = 0b101;
    var s = try dense.structArray(allocator, &.{ "a", "b" }, &.{ a, b }, 3, validity);
    defer s.release();
    const f = common.datatype.Field;
    const cases = [_]struct { []const f, []const ?i64, []const ?[]const u8 }{
        .{ &.{ .{ .name = "a", .data_type = &int64_type }, .{ .name = "b", .data_type = &large_string_type } }, &.{ 1, 0, 3 }, &.{ "x", "", null } },
        .{ &.{ .{ .name = "b", .data_type = &string_type }, .{ .name = "a", .data_type = &int64_type } }, &.{ 1, 0, 3 }, &.{ "x", "", null } },
        .{ &.{ .{ .name = "a", .data_type = &int64_type }, .{ .name = "c", .data_type = &string_type } }, &.{ 1, 0, 3 }, &.{ null, null, null } },
        .{ &.{ .{ .name = "x", .data_type = &int64_type }, .{ .name = "b", .data_type = &string_type } }, &.{ null, null, null }, &.{ "x", "", null } },
    };
    for (cases) |c| {
        const dt = DataType{ .struct_ = .{ .fields = c[0] } };
        var out = try castTo(&ctx, s, dt, true);
        defer out.release();
        try out.data().validateFull();
        try std.testing.expect(out.data().isNull(1) and !out.data().isNull(0));
        for (c[0], 0..) |field, k| {
            const child = out.data().children[k];
            if (field.data_type.* == .int64) {
                const got = try common.collect(i64, allocator, .{ .array = child });
                defer allocator.free(got);
                try common.expectRows(i64, c[1], got);
            } else {
                const got = try common.collectStrings(allocator, .{ .array = child });
                defer allocator.free(got);
                try common.expectStrings(c[2], got);
            }
        }
    }
    // A repeated target name takes the next unused source field (none here: all null).
    const twice = DataType{ .struct_ = .{ .fields = &.{ .{ .name = "a", .data_type = &int32_type }, .{ .name = "a", .data_type = &int32_type } } } };
    var tw = try castTo(&ctx, s, twice, true);
    defer tw.release();
    try std.testing.expectEqual(@as(usize, 3), tw.data().children[1].data().null_count.?);
    // Non-nullable targets: a missing field is UnsupportedType, a field with nulls InvalidCast,
    // a field without nulls casts.
    const missing_nn = DataType{ .struct_ = .{ .fields = &.{.{ .name = "c", .data_type = &int32_type, .nullable = false }} } };
    try std.testing.expectError(error.UnsupportedType, ctx.invokeScalar("cast", &.{.{ .array = s }}, to(missing_nn)));
    const b_nn = DataType{ .struct_ = .{ .fields = &.{.{ .name = "b", .data_type = &string_type, .nullable = false }} } };
    try std.testing.expectError(error.InvalidCast, ctx.invokeScalar("cast", &.{.{ .array = s }}, to(b_nn)));
    const a_nn = DataType{ .struct_ = .{ .fields = &.{.{ .name = "a", .data_type = &int64_type, .nullable = false }} } };
    var ann = try castTo(&ctx, s, a_nn, true);
    defer ann.release();
    // A field cast error surfaces (the null struct row's "" is cast too, as Arrow).
    const b_int = DataType{ .struct_ = .{ .fields = &.{.{ .name = "b", .data_type = &int32_type }} } };
    try std.testing.expectError(error.InvalidCast, ctx.invokeScalar("cast", &.{.{ .array = s }}, to(b_int)));
    // Sliced struct: the validity window and the children follow the slice.
    var tail = try s.slice(1, 2);
    defer tail.release();
    const ba = DataType{ .struct_ = .{ .fields = cases[1][0] } };
    var tail_out = try castTo(&ctx, tail, ba, true);
    defer tail_out.release();
    try tail_out.data().validateFull();
    try std.testing.expect(tail_out.data().isNull(0) and !tail_out.data().isNull(1));
    const tail_a = try common.collect(i64, allocator, .{ .array = tail_out.data().children[1] });
    defer allocator.free(tail_a);
    try common.expectRows(i64, &.{ 0, 3 }, tail_a);
    // The null type to a struct; a struct to a list is unsupported.
    var nulls = try common.datumBuildNullLike(.{ .null = {} }, 2);
    defer nulls.release();
    var ns = try ctx.invokeScalar("cast", &.{nulls}, to(ba));
    defer ns.release();
    try ns.array.data().validateFull();
    try std.testing.expectError(error.UnsupportedType, ctx.invokeScalar("cast", &.{.{ .array = s }}, to(list32)));

    // map<string, int32> [[("k", 1)], null, []] to map<large_string, int64>, to
    // list<struct<k, v>> (entries by position), and to map<int32, int32> ("k" fails).
    const key_str = f{ .name = "key", .data_type = &string_type, .nullable = false };
    const val32 = f{ .name = "value", .data_type = &int32_type };
    const keys = try common.makeStrings(.string, allocator, &.{"k"});
    const items = try common.makeArray(i32, allocator, &.{1});
    var entries = try dense.structArrayFields(allocator, &.{ key_str, val32 }, &.{ keys, items }, 1, null);
    defer entries.release();
    var mb = try common.array_mod.MapBuilder.init(allocator, 3, key_str, val32, false);
    defer mb.deinit();
    try mb.appendLen(1);
    try mb.appendNull();
    try mb.appendLen(0);
    var m = try mb.finish(entries);
    defer m.release();
    try m.data().validateFull();
    const map_wide = DataType{ .map = .{ .key_field = .{ .name = "key", .data_type = &large_string_type, .nullable = false }, .item_field = .{ .name = "value", .data_type = &int64_type } } };
    var mw = try castTo(&ctx, m, map_wide, true);
    defer mw.release();
    try mw.data().validateFull();
    const mw_items = try common.collect(i64, allocator, .{ .array = mw.data().children[0].data().children[1] });
    defer allocator.free(mw_items);
    try common.expectRows(i64, &.{1}, mw_items);
    try std.testing.expect(mw.data().isNull(1));
    const kv = DataType{ .struct_ = .{ .fields = &.{ .{ .name = "k", .data_type = &string_type }, .{ .name = "v", .data_type = &int64_type } } } };
    const list_kv = DataType{ .list = .{ .value_field = .{ .name = "item", .data_type = &kv } } };
    var as_list = try castTo(&ctx, m, list_kv, true);
    defer as_list.release();
    try as_list.data().validateFull();
    const lv_items = try common.collect(i64, allocator, .{ .array = as_list.data().children[0].data().children[1] });
    defer allocator.free(lv_items);
    try common.expectRows(i64, &.{1}, lv_items);
    const map_int = DataType{ .map = .{ .key_field = .{ .name = "key", .data_type = &int32_type, .nullable = false }, .item_field = val32 } };
    try std.testing.expectError(error.InvalidCast, ctx.invokeScalar("cast", &.{.{ .array = m }}, to(map_int)));
    try std.testing.expectError(error.UnsupportedType, ctx.invokeScalar("cast", &.{.{ .array = as_list }}, to(map_wide)));
    var nm = try ctx.invokeScalar("cast", &.{nulls}, to(map_wide));
    defer nm.release();
    try nm.array.data().validateFull();
    try std.testing.expectEqual(@as(usize, 2), nm.array.data().null_count.?);
}

// ------------------------------------------------------------------------------ float16

const half_type = DataType{ .half_float = {} };

test "float16: cast to and from numbers and text, promotion to float32 in arithmetic and comparisons, is_nan / is_inf / is_finite (pyarrow 26 results)" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    const nan16 = std.math.nan(f16);
    const inf16 = std.math.inf(f16);

    var h = try common.makeArray(f16, allocator, &.{ 1.5, null, 65504, inf16, nan16 });
    defer h.release();
    const dh = [_]Datum{.{ .array = h }};
    try common.expectInvokeWith(f32, &ctx, "cast", &dh, to(.{ .float = {} }), &.{ 1.5, null, 65504, std.math.inf(f32), std.math.nan(f32) });
    try common.expectInvokeWith(f64, &ctx, "cast", &dh, to(.{ .double = {} }), &.{ 1.5, null, 65504, std.math.inf(f64), std.math.nan(f64) });
    try common.expectInvokeStrings(&ctx, "cast", &dh, to(string_type), &.{ "1.5", null, "65504", "inf", "nan" });
    var odd = try common.makeArray(f16, allocator, &.{ 0.1, 1.0 / 3.0, 1e-7, -0.0 });
    defer odd.release();
    try common.expectInvokeStrings(&ctx, "cast", &.{.{ .array = odd }}, to(string_type), &.{ "0.0999755859375", "0.333251953125", "1.1920928955078125e-07", "-0" });
    // To integers: a fraction, inf and NaN fail when safe; unsafe truncates and saturates.
    try std.testing.expectError(error.InvalidCast, ctx.invokeScalar("cast", &dh, to(int32_type)));
    try common.expectInvokeWith(i32, &ctx, "cast", &dh, unsafeTo(int32_type), &.{ 1, null, 65504, std.math.maxInt(i32), 0 });
    var whole = try common.makeArray(f16, allocator, &.{ -3, 255, null });
    defer whole.release();
    try common.expectInvokeWith(u8, &ctx, "cast", &.{.{ .array = whole }}, unsafeTo(.{ .uint8 = {} }), &.{ 0, 255, null });
    try std.testing.expectError(error.InvalidCast, ctx.invokeScalar("cast", &.{.{ .array = whole }}, to(.{ .uint8 = {} })));
    try common.expectInvokeWith(i16, &ctx, "cast", &.{.{ .array = whole }}, to(.{ .int16 = {} }), &.{ -3, 255, null });
    // From floats: round to nearest even (0.1 -> 0.0999755859375), overflow to inf; from
    // integers: exact up to 2048, past it InvalidCast when safe (pyarrow 26 rounds silently).
    var d = try common.makeArray(f64, allocator, &.{ 1.5, 70000.0, 1e-8, null, std.math.nan(f64), 0.1 });
    defer d.release();
    try common.expectInvokeWith(f16, &ctx, "cast", &.{.{ .array = d }}, to(half_type), &.{ 1.5, inf16, 0, null, nan16, 0.0999755859375 });
    var ints = try common.makeArray(i32, allocator, &.{ 1, -2048, 2048 });
    defer ints.release();
    try common.expectInvokeWith(f16, &ctx, "cast", &.{.{ .array = ints }}, to(half_type), &.{ 1, -2048, 2048 });
    var big = try common.makeArray(i64, allocator, &.{ 2049, 70000 });
    defer big.release();
    try std.testing.expectError(error.InvalidCast, ctx.invokeScalar("cast", &.{.{ .array = big }}, to(half_type)));
    try common.expectInvokeWith(f16, &ctx, "cast", &.{.{ .array = big }}, unsafeTo(half_type), &.{ 2048, inf16 });
    var text = try common.makeStrings(.string, allocator, &.{ "1.5", "inf", null });
    defer text.release();
    try common.expectInvokeWith(f16, &ctx, "cast", &.{.{ .array = text }}, to(half_type), &.{ 1.5, inf16, null });
    var bad = try common.makeStrings(.string, allocator, &.{"x"});
    defer bad.release();
    try std.testing.expectError(error.InvalidCast, ctx.invokeScalar("cast", &.{.{ .array = bad }}, to(half_type)));
    try common.expectInvokeWith(f16, &ctx, "cast", &.{.{ .scalar = common.scalarOf(f64, 2.5) }}, to(half_type), &.{2.5});

    // Arithmetic and comparisons promote float16 to float32 (also against itself), float64 wins.
    var g = try common.makeArray(f16, allocator, &.{ 2, 1, 3, -inf16, 1 });
    defer g.release();
    const hg = [_]Datum{ .{ .array = h }, .{ .array = g } };
    try common.expectInvokeWith(f32, &ctx, "add", &hg, common.none, &.{ 3.5, null, 65507, std.math.nan(f32), std.math.nan(f32) });
    try common.expectInvokeWith(bool, &ctx, "less", &hg, common.none, &.{ true, null, false, false, false });
    try common.expectInvokeWith(bool, &ctx, "equal", &.{ .{ .array = g }, .{ .scalar = common.scalarOf(i32, 1) } }, common.none, &.{ false, true, false, false, true });
    var f64s = try common.makeArray(f64, allocator, &.{ 0.25, 0, 0, 0, 0 });
    defer f64s.release();
    try common.expectInvokeWith(f64, &ctx, "multiply", &.{ .{ .array = g }, .{ .array = f64s } }, common.none, &.{ 0.5, 0, 0, std.math.nan(f64), 0 });
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    try std.testing.expect((try registry.resolveResultType("add", .scalar, &hg, common.none)).eql(.{ .float = {} }));

    try common.expectInvokeWith(bool, &ctx, "is_nan", &dh, common.none, &.{ false, null, false, false, true });
    try common.expectInvokeWith(bool, &ctx, "is_inf", &dh, common.none, &.{ false, null, false, true, false });
    try common.expectInvokeWith(bool, &ctx, "is_finite", &dh, common.none, &.{ true, null, true, false, false });
    try common.expectInvokeWith(bool, &ctx, "is_null", &dh, .{ .null_options = .{ .nan_is_null = true } }, &.{ false, true, false, false, true });
    // Selections already handle the 2-byte layout.
    var taken = try common.core.datumTake(.{ .array = h }, &.{ 4, 0 });
    defer taken.release();
    const trows = try common.collect(f16, allocator, taken);
    defer allocator.free(trows);
    try common.expectRows(f16, &.{ nan16, 1.5 }, trows);
}
