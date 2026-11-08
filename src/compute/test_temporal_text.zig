const std = @import("std");
const common = @import("test_common.zig");
const temporal_text = @import("temporal_text.zig");

const ArrayRef = common.ArrayRef;
const DataType = common.DataType;
const Datum = common.Datum;
const ExecContext = common.ExecContext;
const Options = common.Options;
const SharedBuffer = @import("../buffer.zig").SharedBuffer;
const TimeUnit = common.datatype.TimeUnit;

fn unitOf(s: []const u8) !TimeUnit {
    inline for (.{ .{ "s", TimeUnit.second }, .{ "ms", TimeUnit.millisecond }, .{ "us", TimeUnit.microsecond }, .{ "ns", TimeUnit.nanosecond } }) |p| {
        if (std.mem.eql(u8, s, p[0])) return p[1];
    }
    return error.InvalidInput;
}

/// The type of a corpus type code (`ts:<unit>:<timezone>`, `d32`, `d64`, `t32:<unit>`,
/// `t64:<unit>`, `dur:<unit>`); the timezone slices the corpus.
fn typeOf(code: []const u8) !DataType {
    var parts = std.mem.splitScalar(u8, code, ':');
    const kind = parts.next().?;
    if (std.mem.eql(u8, kind, "d32")) return .{ .date32 = {} };
    if (std.mem.eql(u8, kind, "d64")) return .{ .date64 = {} };
    const unit = try unitOf(parts.next().?);
    if (std.mem.eql(u8, kind, "t32")) return .{ .time32 = .{ .unit = unit } };
    if (std.mem.eql(u8, kind, "t64")) return .{ .time64 = .{ .unit = unit } };
    if (std.mem.eql(u8, kind, "dur")) return .{ .duration = .{ .unit = unit } };
    const zone = parts.rest();
    return .{ .timestamp = .{ .unit = unit, .timezone = if (zone.len == 0) null else zone } };
}

/// One-row array of `dt` over `storage` (borrowed).
fn oneRow(allocator: std.mem.Allocator, dt: DataType, storage: *[8]u8, v: i64) !ArrayRef {
    const width: usize = if (dt == .date32 or dt == .time32) 4 else 8;
    if (width == 4) std.mem.writeInt(i32, storage[0..4], @intCast(v), .little) else std.mem.writeInt(i64, storage, v, .little);
    const bufs = [_]SharedBuffer{ .empty, .fromSlice(storage[0..width]) };
    return ArrayRef.fromBorrowed(allocator, .{ .data_type = dt, .length = 1, .null_count = 0, .buffers = &bufs });
}

fn hexDecode(allocator: std.mem.Allocator, hex: []const u8) ![]u8 {
    const out = try allocator.alloc(u8, hex.len / 2);
    errdefer allocator.free(out);
    _ = try std.fmt.hexToBytes(out, hex);
    return out;
}

fn hexDecodeTmp(buf: []u8, hex: []const u8) ![]const u8 {
    return std.fmt.hexToBytes(buf[0 .. hex.len / 2], hex);
}

/// The single row of a string result, hex-encoded into `out`.
fn hexRow(allocator: std.mem.Allocator, out: *std.ArrayList(u8), datum: Datum) !void {
    const arr = datum.array;
    try arr.data().validateFull();
    const row = (try common.stringAt(arr.data(), 0)).?;
    const digits = "0123456789abcdef";
    for (row) |b| {
        try out.append(allocator, digits[b >> 4]);
        try out.append(allocator, digits[b & 15]);
    }
}

/// The single row of a temporal result as its stored integer, into `out`.
fn intRow(allocator: std.mem.Allocator, out: *std.ArrayList(u8), datum: Datum) !void {
    const data = datum.array.data();
    try data.validateFull();
    const v: i64 = if (data.data_type == .date32 or data.data_type == .time32)
        (try data.buffers[1].typedSlice(i32))[data.offset]
    else
        (try data.buffers[1].typedSlice(i64))[data.offset];
    try out.print(allocator, "{d}", .{v});
}

const Kind = enum { F, C, P, S };

test "temporal text: strftime, strptime and casts match pyarrow 26 on the generated corpus (tools/temporal_text_corpus.py)" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    const corpus = @embedFile("testdata/temporal_text_corpus.tsv");

    var got: std.ArrayList(u8) = .empty;
    defer got.deinit(allocator);
    var cases = std.enums.EnumArray(Kind, usize).initFill(0);
    var mismatches: usize = 0;
    var strptime_z: usize = 0;
    var leading_text: usize = 0;
    var hex_buf: [256]u8 = undefined;
    var lines = std.mem.splitScalar(u8, corpus, '\n');
    while (lines.next()) |line| {
        if (line.len == 0 or line[0] == '#') continue;
        var fields = std.mem.splitScalar(u8, line, '\t');
        const kind = std.meta.stringToEnum(Kind, fields.next().?).?;
        var f: [4][]const u8 = undefined;
        var n: usize = 0;
        while (fields.next()) |x| : (n += 1) f[n] = x;
        const expected = f[n - 1];
        cases.getPtr(kind).* += 1;
        got.clearRetainingCapacity();
        var storage: [8]u8 = undefined;
        switch (kind) {
            .F => {
                const fmt = try hexDecode(allocator, f[1]);
                defer allocator.free(fmt);
                var arr = try oneRow(allocator, try typeOf(f[0]), &storage, try std.fmt.parseInt(i64, f[2], 10));
                defer arr.release();
                const args = [_]Datum{.{ .array = arr }};
                if (ctx.invoke("strftime", .scalar, &args, .{ .strftime = .{ .format = fmt } })) |res| {
                    var r = res;
                    defer r.release();
                    try hexRow(allocator, &got, r);
                } else |_| try got.append(allocator, 'E');
            },
            .C => {
                var arr = try oneRow(allocator, try typeOf(f[0]), &storage, try std.fmt.parseInt(i64, f[1], 10));
                defer arr.release();
                const args = [_]Datum{.{ .array = arr }};
                if (ctx.invoke("cast", .scalar, &args, .{ .cast = .{ .to_type = .{ .string = {} } } })) |res| {
                    var r = res;
                    defer r.release();
                    try hexRow(allocator, &got, r);
                } else |_| try got.append(allocator, 'E');
            },
            .P, .S => {
                const text = try hexDecode(allocator, if (kind == .P) f[1] else f[2]);
                defer allocator.free(text);
                var arr = try common.makeStrings(.string, allocator, &.{text});
                defer arr.release();
                const args = [_]Datum{.{ .array = arr }};
                var fmt: []u8 = &.{};
                defer allocator.free(fmt);
                const options: Options = if (kind == .P) .{ .cast = .{ .to_type = try typeOf(f[0]) } } else blk: {
                    fmt = try hexDecode(allocator, f[1]);
                    break :blk .{ .strptime = .{ .format = fmt, .unit = try unitOf(f[0]) } };
                };
                if (ctx.invoke(if (kind == .P) "cast" else "strptime", .scalar, &args, options)) |res| {
                    var r = res;
                    defer r.release();
                    try intRow(allocator, &got, r);
                } else |_| try got.append(allocator, 'E');
            },
        }
        if (std.mem.eql(u8, got.items, expected)) continue;
        // pyarrow 26 on Windows parses with the musl strptime Arrow vendors, which lacks `%z`
        // (every such row fails there); zarrow implements glibc's `%z`, as Arrow on Linux.
        if (kind == .S and std.mem.eql(u8, expected, "E") and std.mem.indexOf(u8, try hexDecodeTmp(&hex_buf, f[1]), "%z") != null) {
            strptime_z += 1;
            continue;
        }
        // pyarrow 26 on Windows formats through MSVC's std::format, which reads a format that
        // does not start with `%` as fill / align / width (or rejects it); zarrow prints the
        // text, as Arrow's date-library build on Linux.
        if (kind == .F and !std.mem.startsWith(u8, try hexDecodeTmp(&hex_buf, f[1]), "%")) {
            leading_text += 1;
            continue;
        }
        mismatches += 1;
        if (mismatches <= 400) std.debug.print("mismatch {s}: got {s}\n", .{ line, got.items });
    }
    if (mismatches > 0) std.debug.print("{d} mismatches\n", .{mismatches});
    try std.testing.expectEqual(@as(usize, 0), mismatches);
    try std.testing.expectEqual([_]usize{ 8730, 1392, 3564, 3016 }, cases.values);
    try std.testing.expectEqual(@as(usize, 53), strptime_z);
    try std.testing.expectEqual(@as(usize, 90), leading_text);
}

const ts_ms = DataType{ .timestamp = .{ .unit = .millisecond, .timezone = null } };
const ts_s_offset = DataType{ .timestamp = .{ .unit = .second, .timezone = "+05:30" } };

test "temporal text: strftime and casts to text on nulls, sliced, chunked and scalar inputs, every string layout (pyarrow 26 results)" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);

    var full = try common.makeTyped(i64, ts_ms, allocator, &.{ 99, 1234567891123, null, -1, 0 });
    defer full.release();
    var arr = try full.slice(1, 4);
    defer arr.release();
    const args = [_]Datum{.{ .array = arr }};
    try common.expectInvokeStrings(&ctx, "strftime", &args, .{ .none = {} }, &.{ "2009-02-13T23:31:31.123", null, "1969-12-31T23:59:59.999", "1970-01-01T00:00:00.000" });
    try common.expectInvokeStrings(&ctx, "strftime", &args, .{ .strftime = .{ .format = "%a %d %b %Y %OS %j %%" } }, &.{ "Fri 13 Feb 2009 31 044 %", null, "Wed 31 Dec 1969 59 365 %", "Thu 01 Jan 1970 00 001 %" });
    inline for (.{ DataType{ .string = {} }, DataType{ .large_string = {} }, DataType{ .string_view = {} } }) |to| {
        try common.expectInvokeStrings(&ctx, "cast", &args, .{ .cast = .{ .to_type = to } }, &.{ "2009-02-13 23:31:31.123", null, "1969-12-31 23:59:59.999", "1970-01-01 00:00:00.000" });
    }
    // A chunked input keeps its chunks.
    var chunked = try common.chunkedAt(allocator, arr, &.{2});
    defer chunked.release();
    try common.expectInvokeStrings(&ctx, "strftime", &.{.{ .chunked = chunked }}, .{ .strftime = .{ .format = "%F" } }, &.{ "2009-02-13", null, "1969-12-31", "1970-01-01" });
    // A fixed offset prints the local time, `%z` and `%Z`; a cast appends the offset.
    var zoned = try common.makeTyped(i64, ts_s_offset, allocator, &.{ 0, null });
    defer zoned.release();
    const zargs = [_]Datum{.{ .array = zoned }};
    try common.expectInvokeStrings(&ctx, "strftime", &zargs, .{ .strftime = .{ .format = "%H:%M %z %Ez %Z" } }, &.{ "05:30 +0530 +05:30 +0530", null });
    try common.expectInvokeStrings(&ctx, "cast", &zargs, .{ .cast = .{ .to_type = .{ .string = {} } } }, &.{ "1970-01-01 05:30:00+0530", null });
    // A scalar gives a scalar.
    const s = common.Scalar.init(.{ .date32 = {} }, .{ .date32 = 18321 });
    var out = try ctx.invokeScalar("strftime", &.{.{ .scalar = s }}, .{ .strftime = .{ .format = "%D %e" } });
    defer out.release();
    try std.testing.expectEqualStrings("02/29/20 29", out.scalar.value.string);
    // Times and durations.
    var t = try common.makeTyped(i32, .{ .time32 = .{ .unit = .millisecond } }, allocator, &.{ 3_723_004, -1, 86_400_000, null });
    defer t.release();
    try common.expectInvokeStrings(&ctx, "cast", &.{.{ .array = t }}, .{ .cast = .{ .to_type = .{ .string = {} } } }, &.{ "01:02:03.004", "<value out of range: -1>", "<value out of range: 86400000>", null });
    var d = try common.makeTyped(i64, .{ .duration = .{ .unit = .second } }, allocator, &.{ -5, null });
    defer d.release();
    try common.expectInvokeStrings(&ctx, "cast", &.{.{ .array = d }}, .{ .cast = .{ .to_type = .{ .large_string = {} } } }, &.{ "-5", null });
}

test "temporal text: strptime and casts from text on nulls, sliced, chunked and scalar inputs; error_is_null; times and durations" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);

    const fmt = "%Y-%m-%d %H:%M:%S";
    inline for (.{ .string, .large_string, .string_view }) |tag| {
        var full = try common.makeStrings(tag, allocator, &.{ "x", "2009-02-13 23:31:31", null, "bad", "1970-01-01 00:00:01" });
        defer full.release();
        var arr = try full.slice(1, 4);
        defer arr.release();
        const args = [_]Datum{.{ .array = arr }};
        try common.expectInvokeWith(i64, &ctx, "strptime", &args, .{ .strptime = .{ .format = fmt, .unit = .millisecond, .error_is_null = true } }, &.{ 1234567891000, null, null, 1000 });
        try std.testing.expectError(error.InvalidInput, ctx.invokeScalar("strptime", &args, .{ .strptime = .{ .format = fmt, .unit = .second } }));
        var chunked = try common.chunkedAt(allocator, arr, &.{1});
        defer chunked.release();
        try common.expectInvokeWith(i64, &ctx, "strptime", &.{.{ .chunked = chunked }}, .{ .strptime = .{ .format = fmt, .unit = .second, .error_is_null = true } }, &.{ 1234567891, null, null, 1 });
    }
    // `%z` gives a UTC result, shifted by the offset (glibc's `%z`).
    var z = try common.makeStrings(.string, allocator, &.{ "2009-02-13 23:31:31+0100", "2009-02-13 23:31:31Z", "2009-02-13 23:31:31-01:30" });
    defer z.release();
    const zopts: Options = .{ .strptime = .{ .format = "%Y-%m-%d %H:%M:%S%z", .unit = .second } };
    var zout = try ctx.invokeScalar("strptime", &.{.{ .array = z }}, zopts);
    defer zout.release();
    try std.testing.expectEqualStrings("UTC", zout.array.data().data_type.timestamp.timezone.?);
    try common.expectInvokeWith(i64, &ctx, "strptime", &.{.{ .array = z }}, zopts, &.{ 1234564291, 1234567891, 1234573291 });
    // Options: a format past the cap or holding a NUL is InvalidOptions; strptime has no defaults.
    const long = try allocator.alloc(u8, temporal_text.max_format_len + 1);
    defer allocator.free(long);
    @memset(long, 'x');
    try std.testing.expectError(error.InvalidOptions, ctx.invokeScalar("strptime", &.{.{ .array = z }}, .{ .strptime = .{ .format = long, .unit = .second } }));
    try std.testing.expectError(error.InvalidOptions, ctx.invokeScalar("strptime", &.{.{ .array = z }}, .{ .strptime = .{ .format = "%Y\x00", .unit = .second } }));
    if (ctx.invokeScalar("strptime", &.{.{ .array = z }}, .{ .none = {} })) |r| {
        var rr = r;
        rr.release();
        return error.TestUnexpectedResult;
    } else |_| {}
    // A scalar string to a zoned timestamp.
    const s = common.Scalar.init(.{ .string = {} }, .{ .string = "2020-02-29T12:00:00.5Z" });
    var out = try ctx.invokeScalar("cast", &.{.{ .scalar = s }}, .{ .cast = .{ .to_type = .{ .timestamp = .{ .unit = .millisecond, .timezone = "UTC" } } } });
    defer out.release();
    try std.testing.expectEqual(@as(i64, 1582977600500), out.scalar.value.timestamp);
    // Times and durations from text (Arrow's ParseTime / integer parsing; pyarrow 26 has no such
    // cast).
    var times = try common.makeStrings(.string, allocator, &.{ "01:02", "01:02:03.25", "23:59:59.999999999", null });
    defer times.release();
    try common.expectInvokeWith(i64, &ctx, "cast", &.{.{ .array = times }}, .{ .cast = .{ .to_type = .{ .time64 = .{ .unit = .nanosecond } } } }, &.{ 3_720_000_000_000, 3_723_250_000_000, 86_399_999_999_999, null });
    var two = try times.slice(0, 2);
    defer two.release();
    try common.expectInvokeWith(i32, &ctx, "cast", &.{.{ .array = two }}, .{ .cast = .{ .to_type = .{ .time32 = .{ .unit = .millisecond } } } }, &.{ 3_720_000, 3_723_250 });
    try std.testing.expectError(error.InvalidCast, ctx.invokeScalar("cast", &.{.{ .array = two }}, .{ .cast = .{ .to_type = .{ .time32 = .{ .unit = .second } } } }));
    var bad_time = try common.makeStrings(.string, allocator, &.{"24:00"});
    defer bad_time.release();
    try std.testing.expectError(error.InvalidCast, ctx.invokeScalar("cast", &.{.{ .array = bad_time }}, .{ .cast = .{ .to_type = .{ .time32 = .{ .unit = .second } } } }));
    var durs = try common.makeStrings(.string_view, allocator, &.{ "-12", "+7", null });
    defer durs.release();
    try common.expectInvokeWith(i64, &ctx, "cast", &.{.{ .array = durs }}, .{ .cast = .{ .to_type = .{ .duration = .{ .unit = .microsecond } } } }, &.{ -12, 7, null });
    // A naive target refuses a zone offset.
    var zs = try common.makeStrings(.string, allocator, &.{"1970-01-01T00:00:00Z"});
    defer zs.release();
    try std.testing.expectError(error.InvalidCast, ctx.invokeScalar("cast", &.{.{ .array = zs }}, .{ .cast = .{ .to_type = ts_ms } }));
}

test "temporal text: strftime option and timezone errors (Arrow's rules)" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    var arr = try common.makeTyped(i64, ts_ms, allocator, &.{0});
    defer arr.release();
    const args = [_]Datum{.{ .array = arr }};
    for ([_][]const u8{ "%", "abc%", "%Q", "%Ea", "%OY", "%E", "%5Y", "%-d", "%s" }) |fmt| {
        try std.testing.expectError(error.InvalidOptions, ctx.invokeScalar("strftime", &args, .{ .strftime = .{ .format = fmt } }));
    }
    // `%z` / `%Z` (also inside `%%z`: Arrow's substring test) on a value without a timezone.
    for ([_][]const u8{ "%z", "%Z", "%%z" }) |fmt| {
        try std.testing.expectError(error.InvalidInput, ctx.invokeScalar("strftime", &args, .{ .strftime = .{ .format = fmt } }));
    }
    try std.testing.expectError(error.NotImplemented, ctx.invokeScalar("strftime", &args, .{ .strftime = .{ .locale = "en_US.UTF-8" } }));
    const long = try allocator.alloc(u8, temporal_text.max_format_len + 1);
    defer allocator.free(long);
    @memset(long, 'x');
    try std.testing.expectError(error.InvalidOptions, ctx.invokeScalar("strftime", &args, .{ .strftime = .{ .format = long } }));
    var bad = try common.makeTyped(i64, .{ .timestamp = .{ .unit = .second, .timezone = "+24:00" } }, allocator, &.{0});
    defer bad.release();
    try std.testing.expectError(error.InvalidInput, ctx.invokeScalar("strftime", &.{.{ .array = bad }}, .{ .none = {} }));
}
