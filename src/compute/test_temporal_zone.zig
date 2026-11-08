const std = @import("std");
const common = @import("test_common.zig");

const ArrayRef = common.ArrayRef;
const DataType = common.DataType;
const Datum = common.Datum;
const ExecContext = common.ExecContext;
const Options = common.Options;
const SharedBuffer = @import("../buffer.zig").SharedBuffer;
const TimeUnit = common.datatype.TimeUnit;

fn oneRow(allocator: std.mem.Allocator, dt: DataType, storage: *[8]u8, v: i64) !ArrayRef {
    std.mem.writeInt(i64, storage, v, .little);
    const bufs = [_]SharedBuffer{ .empty, .fromSlice(storage) };
    return ArrayRef.fromBorrowed(allocator, .{ .data_type = dt, .length = 1, .null_count = 0, .buffers = &bufs });
}

/// The single row of `datum` in the corpus notation into `out`.
fn describe(allocator: std.mem.Allocator, out: *std.ArrayList(u8), datum: Datum) !void {
    const data = datum.array.data();
    try data.validateFull();
    switch (data.data_type) {
        .bool => try out.append(allocator, if (@import("../bitmap.zig").bitIsSet(data.buffers[1].data, data.offset)) 'T' else 'F'),
        .string => {
            const row = (try common.stringAt(data, 0)).?;
            const digits = "0123456789abcdef";
            for (row) |b| {
                try out.append(allocator, digits[b >> 4]);
                try out.append(allocator, digits[b & 15]);
            }
        },
        .date32 => try out.print(allocator, "{d}", .{(try data.buffers[1].typedSlice(i32))[data.offset]}),
        else => try out.print(allocator, "{d}", .{(try data.buffers[1].typedSlice(i64))[data.offset]}),
    }
}

fn unitOf(s: []const u8) TimeUnit {
    return if (std.mem.eql(u8, s, "s")) .second else .millisecond;
}

const CalendarUnit = common.core.CalendarUnit;

fn invokeCase(ctx: *ExecContext, allocator: std.mem.Allocator, out: *std.ArrayList(u8), name: []const u8, arr: ArrayRef, v: i64, ts: DataType) !void {
    var storage: [8]u8 = undefined;
    const per: i64 = if (ts.timestamp.unit == .second) 1 else 1000;
    var other = try oneRow(allocator, ts, &storage, v + 90000 * per);
    defer other.release();
    const d: Datum = .{ .array = arr };
    var options: Options = .none;
    var fname = name;
    var args_buf = [2]Datum{ d, .{ .array = other } };
    var args: []const Datum = args_buf[0..1];
    if (std.mem.eql(u8, name, "strftime")) {
        options = .{ .strftime = .{ .format = "%Y-%m-%dT%H:%M:%S%z" } };
    } else if (std.mem.eql(u8, name, "strftime_Z")) {
        fname = "strftime";
        options = .{ .strftime = .{ .format = "%Z" } };
    } else if (std.mem.startsWith(u8, name, "cast_")) {
        fname = "cast";
        const to: DataType = if (std.mem.eql(u8, name, "cast_string")) .{ .string = {} } else if (std.mem.eql(u8, name, "cast_date32")) .{ .date32 = {} } else .{ .time64 = .{ .unit = .microsecond } };
        options = .{ .cast = .{ .to_type = to } };
    } else if (std.mem.endsWith(u8, name, "_between")) {
        args = &args_buf;
    } else if (std.mem.indexOfScalar(u8, name, ':')) |c| {
        var parts = std.mem.splitScalar(u8, name[c + 1 ..], ':');
        const unit = std.meta.stringToEnum(CalendarUnit, parts.next().?).?;
        const multiple = try std.fmt.parseInt(i64, parts.next().?, 10);
        fname = name[0..c];
        options = .{ .round_temporal = .{ .multiple = multiple, .unit = unit } };
    }
    var res = ctx.invokeScalar(fname, args, options) catch {
        try out.append(allocator, 'E');
        return;
    };
    defer res.release();
    try describe(allocator, out, res);
}

test "timezones: zoned temporal functions and assume_timezone match pyarrow 26 around offset changes (tools/tz_corpus.py)" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    const corpus = @embedFile("testdata/tz_corpus.tsv");
    var got: std.ArrayList(u8) = .empty;
    defer got.deinit(allocator);
    var cases: usize = 0;
    var mismatches: usize = 0;
    var abbr: usize = 0;
    var negative_dst: usize = 0;
    var casablanca_2037: usize = 0;
    var apia_2011: usize = 0;
    var lines = std.mem.splitScalar(u8, corpus, '\n');
    while (lines.next()) |line| {
        if (line.len == 0 or line[0] == '#') continue;
        var f = std.mem.splitScalar(u8, line, '\t');
        const kind = f.next().?;
        const zone = f.next().?;
        const unit = unitOf(f.next().?);
        const v = try std.fmt.parseInt(i64, f.next().?, 10);
        const name = f.next().?;
        const expected = f.next().?;
        cases += 1;
        got.clearRetainingCapacity();
        var storage: [8]u8 = undefined;
        if (kind[0] == 'Z') {
            const ts: DataType = .{ .timestamp = .{ .unit = unit, .timezone = zone } };
            var arr = try oneRow(allocator, ts, &storage, v);
            defer arr.release();
            try invokeCase(&ctx, allocator, &got, name, arr, v, ts);
        } else {
            var arr = try oneRow(allocator, .{ .timestamp = .{ .unit = unit, .timezone = null } }, &storage, v);
            defer arr.release();
            const colon = std.mem.indexOfScalar(u8, name, ':').?;
            const Choice = @FieldType(common.core.AssumeTimezoneOptions, "ambiguous");
            const Gap = @FieldType(common.core.AssumeTimezoneOptions, "nonexistent");
            const options: Options = .{ .assume_timezone = .{
                .timezone = zone,
                .ambiguous = std.meta.stringToEnum(Choice, name[0..colon]).?,
                .nonexistent = std.meta.stringToEnum(Gap, name[colon + 1 ..]).?,
            } };
            if (ctx.invokeScalar("assume_timezone", &.{.{ .array = arr }}, options)) |res| {
                var r = res;
                defer r.release();
                try describe(allocator, &got, r);
            } else |_| try got.append(allocator, 'E');
        }
        if (std.mem.eql(u8, got.items, expected)) continue;
        // pyarrow 26 on Windows takes zone abbreviations from the system (ICU: "GMT-5"); zarrow
        // prints the IANA abbreviation ("EST"), as Arrow's date-library build on Linux.
        if (std.mem.eql(u8, name, "strftime_Z")) {
            abbr += 1;
            continue;
        }
        // IANA's main data gives Ireland and Morocco a negative DST (winter time, Ramadan time
        // is the daylight time); ICU (pyarrow 26 on Windows) uses the rearguard form with the
        // flag flipped. zarrow reports the flag of the compiled tzdata, as Arrow on Linux.
        if (std.mem.eql(u8, name, "is_dst") and (std.mem.eql(u8, zone, "Europe/Dublin") or std.mem.eql(u8, zone, "Africa/Casablanca"))) {
            negative_dst += 1;
            continue;
        }
        // IANA 2026f lists Morocco's predicted Ramadan transitions through 2087; ICU's table
        // drops those after 2037, so pyarrow 26 on Windows keeps +01 there.
        const secs = if (unit == .second) v else @divFloor(v, 1000);
        if (std.mem.eql(u8, zone, "Africa/Casablanca") and secs >= 2_136_000_000) {
            casablanca_2037 += 1;
            continue;
        }
        // Samoa kept daylight time (-10, then +14) across its 2011-12-30 date-line skip (IANA);
        // ICU shows standard time (-11, then +13) from September 2011 to April 2012.
        if (std.mem.eql(u8, zone, "Pacific/Apia") and secs >= 1_316_000_000 and secs < 1_334_000_000) {
            apia_2011 += 1;
            continue;
        }
        mismatches += 1;
        if (mismatches <= 60) std.debug.print("mismatch {s}: got {s}\n", .{ line, got.items });
    }
    if (mismatches > 0) std.debug.print("{d} mismatches of {d}\n", .{ mismatches, cases });
    try std.testing.expectEqual(@as(usize, 0), mismatches);
    try std.testing.expectEqual(@as(usize, 27993), cases);
    try std.testing.expectEqual(@as(usize, 678), abbr);
    try std.testing.expectEqual(@as(usize, 84), negative_dst);
    try std.testing.expectEqual(@as(usize, 390), casablanca_2037);
    try std.testing.expectEqual(@as(usize, 167), apia_2011);
}

const ny_s = DataType{ .timestamp = .{ .unit = .second, .timezone = "America/New_York" } };
const naive_s = DataType{ .timestamp = .{ .unit = .second, .timezone = null } };

test "timezones: assume_timezone, local_timestamp and is_dst on nulls, sliced, chunked and scalar inputs (pyarrow 26 results)" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);

    // New York local 2021-03-14 01:59:59, 02:30 (gap), 2021-11-07 01:30 (ambiguous), null, 00:00.
    var full = try common.makeTyped(i64, naive_s, allocator, &.{ 7, 1615687199, 1615689000, 1636248600, null, 0 });
    defer full.release();
    var local = try full.slice(1, 5);
    defer local.release();
    const args = [_]Datum{.{ .array = local }};
    const both = Options{ .assume_timezone = .{ .timezone = "America/New_York", .ambiguous = .earliest, .nonexistent = .latest } };
    try common.expectInvokeWith(i64, &ctx, "assume_timezone", &args, both, &.{ 1615705199, 1615705200, 1636263000, null, 18000 });
    try common.expectInvokeWith(i64, &ctx, "assume_timezone", &args, .{ .assume_timezone = .{ .timezone = "America/New_York", .ambiguous = .latest, .nonexistent = .earliest } }, &.{ 1615705199, 1615705199, 1636266600, null, 18000 });
    // The defaults raise.
    try std.testing.expectError(error.InvalidInput, ctx.invokeScalar("assume_timezone", &args, .{ .assume_timezone = .{ .timezone = "America/New_York" } }));
    var out = try ctx.invokeScalar("assume_timezone", &args, both);
    defer out.release();
    try std.testing.expectEqualStrings("America/New_York", out.array.data().data_type.timestamp.timezone.?);
    // local_timestamp gives the local values back (the gap time moved to 03:00); is_dst.
    try common.expectInvokeWith(i64, &ctx, "local_timestamp", &.{out}, .none, &.{ 1615687199, 1615690800, 1636248600, null, 0 });
    try common.expectInvokeWith(bool, &ctx, "is_dst", &.{out}, .none, &.{ false, true, true, null, false });
    // Chunked.
    var chunked = try common.chunkedAt(allocator, out.array, &.{2});
    defer chunked.release();
    try common.expectInvokeWith(bool, &ctx, "is_dst", &.{.{ .chunked = chunked }}, .none, &.{ false, true, true, null, false });
    try common.expectInvokeWith(i64, &ctx, "hour", &.{.{ .chunked = chunked }}, .none, &.{ 1, 3, 1, null, 0 });
    // Scalars, a fixed offset, a naive local_timestamp (identity).
    const s = common.Scalar.init(ny_s, .{ .timestamp = 1615705200 });
    var dst = try ctx.invokeScalar("is_dst", &.{.{ .scalar = s }}, .none);
    defer dst.release();
    try std.testing.expect(dst.scalar.value.bool);
    try common.expectInvokeWith(i64, &ctx, "assume_timezone", &args, .{ .assume_timezone = .{ .timezone = "+05:30" } }, &.{ 1615667399, 1615669200, 1636228800, null, -19800 });
    try common.expectInvokeWith(i64, &ctx, "local_timestamp", &args, .none, &.{ 1615687199, 1615689000, 1636248600, null, 0 });
    // Errors: a zoned input to assume_timezone, a naive input to is_dst, an unknown zone.
    try std.testing.expectError(error.InvalidInput, ctx.invokeScalar("assume_timezone", &.{out}, both));
    try std.testing.expectError(error.InvalidInput, ctx.invokeScalar("is_dst", &args, .none));
    try std.testing.expectError(error.InvalidInput, ctx.invokeScalar("assume_timezone", &args, .{ .assume_timezone = .{ .timezone = "Nowhere/Land" } }));
    try std.testing.expectError(error.InvalidOptions, ctx.invokeScalar("assume_timezone", &args, .none));
}

test "timezones: ExecConfig.io resolves names (system zoneinfo on Linux and macOS, else the bundled table)" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.initWithConfig(allocator, &registry, .{ .io = std.testing.io });
    // Sydney: 2021-04-04 02:00 local ends AEDT (+11) for AEST (+10); a southern-hemisphere zone
    // whose rules come from the footer of the bundled TZif file.
    var t = try common.makeTyped(i64, .{ .timestamp = .{ .unit = .second, .timezone = "Australia/Sydney" } }, allocator, &.{ 1617465599, 1617465600, 2121955200 });
    defer t.release();
    try common.expectInvokeWith(i64, &ctx, "local_timestamp", &.{.{ .array = t }}, .none, &.{ 1617505199, 1617501600, 2121994800 });
    try common.expectInvokeStrings(&ctx, "strftime", &.{.{ .array = t }}, .{ .strftime = .{ .format = "%H:%M %Z %z" } }, &.{ "02:59 AEDT +1100", "02:00 AEST +1000", "03:00 AEDT +1100" });
    var bad = try common.makeTyped(i64, .{ .timestamp = .{ .unit = .second, .timezone = "../../etc/passwd" } }, allocator, &.{0});
    defer bad.release();
    try std.testing.expectError(error.InvalidInput, ctx.invokeScalar("hour", &.{.{ .array = bad }}, .none));
}
