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
    try std.testing.expectError(error.UnsupportedType, ctx.invokeScalar("cast", &args, to(date32)));
    try std.testing.expectError(error.UnsupportedType, ctx.invokeScalar("cast", &args, to(.{ .binary = {} })));
    try std.testing.expectError(error.UnsupportedType, ctx.invokeScalar("cast", &args, to(.{ .decimal128 = .{ .precision = 10, .scale = 2 } })));
    try std.testing.expectError(error.UnsupportedType, ctx.invokeScalar("cast", &args, to(dict_string)));
    const dict8 = DataType{ .dictionary = .{ .index_type = .{ .bit_width = 8, .signed = true }, .value_type = &int32_type } };
    try std.testing.expectError(error.UnsupportedType, ctx.invokeScalar("cast", &args, to(dict8)));
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
