//! Temporal values as text (Arrow C++ "Conversions"): `strftime`, `strptime` and the casts
//! between timestamp / date32 / date64 / time32 / time64 / duration and string / large_string /
//! string_view.
//!
//! strftime (`StrftimeOptions`: `format`, default `%Y-%m-%dT%H:%M:%S`; `locale` "C" only, any
//! other is `NotImplemented`) on timestamp, date32 (as seconds), date64 (as milliseconds), time32
//! and time64 (an instant of 1970-01-01 of their unit) gives string. A zoned timestamp prints its
//! local time. Conversion specifiers (pyarrow 26's set): `%a %A %b %h %B %c %C %d %D %e %F %g %G
//! %H %I %j %m %M %n %p %r %R %S %t %T %u %U %V %w %W %x %X %y %Y %z %Z %%`, `E` on `c C x X y Y
//! z` and `O` on `d e H I m M S u U V w W y z`; `%S` and `%T` print the fraction digits of the
//! unit (3, 6 or 9; `%OS`, `%c`, `%r`, `%X` do not); `%c` is `%m/%d/%y %H:%M:%S`, `%x` / `%D`
//! `%m/%d/%y`; `%z` is `+HHMM`, `%Ez` / `%Oz` `+HH:MM`; `%Z` the zone abbreviation (`UTC` without
//! a timezone). Another specifier, a trailing `%`, a format past `max_format_len` bytes or not
//! UTF-8 is `InvalidOptions`; a format holding `%z` or `%Z` on a value without a timezone is
//! `InvalidInput`, as Arrow.
//!
//! strptime (`StrptimeOptions`: `format`, `unit`, `error_is_null`) on the string layouts gives
//! `timestamp(unit)`, or `timestamp(unit, "UTC")` when the format holds `%z`. The parser follows
//! the musl strptime Arrow ships (pyarrow 26): `%C %d %e %D %H %I %j %m %M %n %t %R %S %T %U %w
//! %W %y %Y %%`, an optional `+` and a field width after `%`; a whitespace in the format skips any
//! whitespace; only the year, month, day, hour, minute and second are used (Arrow's conversion:
//! day 0 is day 1, a day past the month's end rolls over, month 0 is December of the previous
//! year); `%z` (`Z`, `+HH`, `+HHMM`, `+HH:MM`, glibc's form) shifts to UTC. The whole row must be
//! consumed; a row that does not parse or whose instant overflows the unit is null with
//! `error_is_null`, else `InvalidInput`.
//!
//! Casts to text (Arrow's `FormatValue`): timestamp `YYYY-MM-DD HH:MM:SS[.fff]` (fraction digits
//! of the unit) plus `Z` for the timezone `UTC` or the local time and `+HHMM` for another zone;
//! date32 / date64 `YYYY-MM-DD`; a date or timestamp outside the years -32767..32767 prints
//! `<value out of range: N>`; time32 / time64 `HH:MM:SS[.fff]` (`<value out of range: N>`
//! outside one day); duration the integer. Casts from text: timestamp from Arrow's ISO 8601
//! forms (`YYYY-MM-DD`, then `[ T]hh`, `hh:mm`, `hh:mm:ss` and `.` plus at most the unit's
//! fraction digits, then `Z`, `+hh`, `+hhmm` or `+hh:mm`; a zone offset is required for a zoned
//! target and refused for a naive one), date32 / date64 from `YYYY-MM-DD`, time32 / time64 from
//! `hh:mm`, `hh:mm:ss` and a fraction (Arrow's `ParseTime`; pyarrow 26 has no such cast),
//! duration from an integer (likewise); text that does not parse or overflows is `InvalidCast`
//! in both modes.
const std = @import("std");
const bitmap = @import("../bitmap.zig");
const core = @import("core.zig");
const datatype = @import("../datatype.zig");
const dense = @import("dense.zig");
const temporal = @import("temporal.zig");
const tz = @import("tz.zig");
const temporal_zone = @import("temporal_zone.zig");

const ArrayData = @import("../array/array_data.zig").ArrayData;
const PrimitiveArray = @import("../array/primitive_array.zig").PrimitiveArray;
const OwnedBuffer = @import("../buffer.zig").OwnedBuffer;
const ArrayRef = core.ArrayRef;
const DataType = core.DataType;
const Datum = core.Datum;
const ExecChunkValue = core.ExecChunkValue;
const ExecContext = core.ExecContext;
const FunctionRegistry = core.FunctionRegistry;
const KernelError = core.KernelError;
const Options = core.Options;
const TimeUnit = datatype.TimeUnit;
const TypeTag = std.meta.Tag(DataType);

/// Longest strftime / strptime format accepted (a trust-boundary cap; Arrow has none).
pub const max_format_len = 4096;

const string_type = DataType{ .string = {} };

pub fn register(registry: *FunctionRegistry) KernelError!void {
    inline for (.{ TypeTag.timestamp, TypeTag.date32, TypeTag.date64, TypeTag.time32, TypeTag.time64 }) |tag| {
        try registry.registerScalarKernel("strftime", .{
            .signature = .{ .arity = 1, .type_check = dense.argsOfType(tag, 1), .options_check = strftimeOptions, .result_type_fn = dense.resultTypeOf(string_type) },
            .exec = strftimeExec,
        });
    }
    inline for (.{ TypeTag.string, TypeTag.large_string, TypeTag.string_view }) |tag| {
        try registry.registerScalarKernel("strptime", .{
            .signature = .{ .arity = 1, .type_check = dense.argsOfType(tag, 1), .options_check = strptimeOptions, .result_type_fn = strptimeType },
            .exec = strptimeExec,
        });
    }
}

fn strftimeOptions(options: Options) bool {
    return options == .none or options == .strftime;
}

fn strptimeOptions(options: Options) bool {
    return options == .strptime;
}

/// Units of `unit` per second.
pub fn perSecond(unit: TimeUnit) i64 {
    return switch (unit) {
        .second => 1,
        .millisecond => 1_000,
        .microsecond => 1_000_000,
        .nanosecond => 1_000_000_000,
    };
}

fn fracDigits(unit: TimeUnit) u8 {
    return switch (unit) {
        .second => 0,
        .millisecond => 3,
        .microsecond => 6,
        .nanosecond => 9,
    };
}

/// How the stored values of a temporal type read as instants: `value * mul` is a count of
/// `1 / per` seconds, printed with `digits` fraction digits.
const Scale = struct { mul: i64 = 1, per: i64, digits: u8 };

fn scaleOf(dt: DataType) ?Scale {
    return switch (dt) {
        .timestamp => |t| .{ .per = perSecond(t.unit), .digits = fracDigits(t.unit) },
        .date32 => .{ .mul = 86_400, .per = 1, .digits = 0 },
        .date64 => .{ .per = 1000, .digits = 3 },
        .time32, .time64 => |t| .{ .per = perSecond(t.unit), .digits = fracDigits(t.unit) },
        else => null,
    };
}

// ------------------------------------------------------------------------------ fields

/// The calendar fields of one instant in local time.
const Fields = struct {
    civ: temporal.Civil,
    days: i64,
    /// Seconds of the day.
    sod: i64,
    /// Fraction of the second in units of the type.
    frac: i64,
    info: tz.Info,
};

inline fn fieldsOf(v: i64, scale: Scale, zone: *tz.Cursor) Fields {
    const x = v *% scale.mul;
    const info = zone.info(@divFloor(x, scale.per));
    // The local time in units of the type, wrapping like Arrow's arithmetic at the i64 edges.
    const local = x +% @as(i64, info.offset) *% scale.per;
    const secs = @divFloor(local, scale.per);
    const days = @divFloor(secs, 86_400);
    return .{ .civ = temporal.civil(days), .days = days, .sod = @mod(secs, 86_400), .frac = @mod(local, scale.per), .info = info };
}

const digit_pairs = blk: {
    var pairs: [200]u8 = undefined;
    for (0..100) |i| {
        pairs[2 * i] = '0' + i / 10;
        pairs[2 * i + 1] = '0' + i % 10;
    }
    break :blk pairs;
};

inline fn put2(dst: []u8, n: anytype) void {
    const k: usize = @intCast(n);
    dst[0..2].* = digit_pairs[2 * k ..][0..2].*;
}

/// Write `|v|` with at least `min` digits, after a `-` for a negative `v`; return the length.
fn putSigned(dst: []u8, v: i64, min: usize) usize {
    var tmp: [24]u8 = undefined;
    var m: u64 = @abs(v);
    var n: usize = 0;
    while (m > 0 or n < min) {
        tmp[tmp.len - 1 - n] = @intCast('0' + m % 10);
        m /= 10;
        n += 1;
    }
    var w: usize = 0;
    if (v < 0) {
        dst[0] = '-';
        w = 1;
    }
    @memcpy(dst[w..][0..n], tmp[tmp.len - n ..]);
    return w + n;
}

/// `YYYY` (at least 4 digits, `-` before a negative year).
inline fn putYear(dst: []u8, y: i64) usize {
    if (y >= 0 and y <= 9999) {
        put2(dst[0..2], @as(u64, @intCast(y)) / 100);
        put2(dst[2..4], @as(u64, @intCast(y)) % 100);
        return 4;
    }
    return putSigned(dst, y, 4);
}

/// `.` and `digits` fraction digits of `frac` (nothing for 0 digits).
inline fn putFrac(dst: []u8, frac: i64, digits: u8) usize {
    if (digits == 0) return 0;
    dst[0] = '.';
    var m: u64 = @intCast(frac);
    var i: usize = digits;
    while (i > 0) : (i -= 1) {
        dst[i] = @intCast('0' + m % 10);
        m /= 10;
    }
    return @as(usize, digits) + 1;
}

/// `YYYY-MM-DD`.
inline fn putDate(dst: []u8, c: temporal.Civil) usize {
    const n = putYear(dst, c.year);
    dst[n] = '-';
    put2(dst[n + 1 ..], c.month);
    dst[n + 3] = '-';
    put2(dst[n + 4 ..], c.day);
    return n + 6;
}

/// `HH:MM:SS` of `sod` seconds of the day.
inline fn putTime(dst: []u8, sod: i64) void {
    const s: u64 = @intCast(sod);
    put2(dst[0..], s / 3600);
    dst[2] = ':';
    put2(dst[3..], s / 60 % 60);
    dst[5] = ':';
    put2(dst[6..], s % 60);
}

// ------------------------------------------------------------------------------ strftime

const Spec = enum(u8) { lit, a, A, b, B, c, C, d, e, G, g, H, I, j, m, M, p, S, S_int, u, U, V, w, W, x, y, Y, z, z_colon, Z };
/// `guard`: the op goes through the C library's `put_time` in pyarrow 26 (MSVC), which prints
/// `?` for a year outside 0..9999.
const Op = struct { spec: Spec, lit: []const u8 = "", guard: bool = false };

const weekday_names = [_][]const u8{ "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday" };
const month_names = [_][]const u8{ "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December" };

/// Bytes an op may write at most (abbreviations are capped at `tz.max_abbr_len`).
fn opMax(op: Op) usize {
    return switch (op.spec) {
        .lit => op.lit.len,
        .A, .B => 9,
        .C, .G, .Y, .g, .y => 24,
        .S => 12,
        .c => 17,
        .x => 8,
        .Z => tz.max_abbr_len,
        .z => 5,
        .z_colon => 6,
        else => 4,
    };
}

/// A compiled strftime format.
const Format = struct {
    ops: std.ArrayList(Op) = .empty,
    /// Upper bound of the bytes of one row.
    max_row: usize = 0,
    /// The format names the zone (`%z` / `%Z` as a substring, Arrow's test).
    names_zone: bool = false,

    fn deinit(self: *Format, allocator: std.mem.Allocator) void {
        self.ops.deinit(allocator);
    }

    fn add(self: *Format, allocator: std.mem.Allocator, ops: []const Op) KernelError!void {
        for (ops) |op| {
            self.ops.append(allocator, op) catch return error.OutOfMemory;
            self.max_row += opMax(op);
        }
    }

    fn compile(allocator: std.mem.Allocator, fmt: []const u8) KernelError!Format {
        // The text of the format lands in a string result: it must be UTF-8.
        if (fmt.len > max_format_len or !std.unicode.utf8ValidateSlice(fmt)) return error.InvalidOptions;
        var f: Format = .{};
        errdefer f.deinit(allocator);
        f.names_zone = std.mem.indexOf(u8, fmt, "%z") != null or std.mem.indexOf(u8, fmt, "%Z") != null;
        var i: usize = 0;
        while (i < fmt.len) {
            if (fmt[i] != '%') {
                const end = std.mem.indexOfScalarPos(u8, fmt, i, '%') orelse fmt.len;
                try f.add(allocator, &.{.{ .spec = .lit, .lit = fmt[i..end] }});
                i = end;
                continue;
            }
            if (i + 1 >= fmt.len) return error.InvalidOptions;
            var c = fmt[i + 1];
            var mod: u8 = 0;
            i += 2;
            if (c == 'E' or c == 'O') {
                if (i >= fmt.len) return error.InvalidOptions;
                mod = c;
                c = fmt[i];
                i += 1;
                const allowed: []const u8 = if (mod == 'E') "cCxXyYz" else "deHImMSuUVwWyz";
                if (std.mem.indexOfScalar(u8, allowed, c) == null) return error.InvalidOptions;
            }
            const H: Op = .{ .spec = .H };
            const M: Op = .{ .spec = .M };
            const colon: Op = .{ .spec = .lit, .lit = ":" };
            const slash: Op = .{ .spec = .lit, .lit = "/" };
            const dash: Op = .{ .spec = .lit, .lit = "-" };
            const mdy = [_]Op{ .{ .spec = .m }, slash, .{ .spec = .d }, slash, .{ .spec = .y } };
            const guard = mod == 'E' or (mod == 'O' and c == 'y');
            try f.add(allocator, switch (c) {
                'a' => &.{.{ .spec = .a }},
                'A' => &.{.{ .spec = .A }},
                'b', 'h' => &.{.{ .spec = .b }},
                'B' => &.{.{ .spec = .B }},
                'c' => &.{.{ .spec = .c, .guard = true }},
                'C' => &.{.{ .spec = .C, .guard = guard }},
                'd' => &.{.{ .spec = .d }},
                'D' => &mdy,
                'x' => &.{.{ .spec = .x, .guard = true }},
                'e' => &.{.{ .spec = .e }},
                'F' => &.{ .{ .spec = .Y }, dash, .{ .spec = .m }, dash, .{ .spec = .d } },
                'g' => &.{.{ .spec = .g, .guard = true }},
                'G' => &.{.{ .spec = .G, .guard = true }},
                'H' => &.{H},
                'I' => &.{.{ .spec = .I }},
                'j' => &.{.{ .spec = .j }},
                'm' => &.{.{ .spec = .m }},
                'M' => &.{M},
                'n' => &.{.{ .spec = .lit, .lit = "\n" }},
                'p' => &.{.{ .spec = .p }},
                'r' => &.{ .{ .spec = .I }, colon, M, colon, .{ .spec = .S_int }, .{ .spec = .lit, .lit = " " }, .{ .spec = .p } },
                'R' => &.{ H, colon, M },
                'S' => if (mod == 'O') &.{.{ .spec = .S_int }} else &.{.{ .spec = .S }},
                't' => &.{.{ .spec = .lit, .lit = "\t" }},
                'T' => &.{ H, colon, M, colon, .{ .spec = .S } },
                'u' => &.{.{ .spec = .u }},
                'U' => &.{.{ .spec = .U }},
                'V' => &.{.{ .spec = .V }},
                'w' => &.{.{ .spec = .w }},
                'W' => &.{.{ .spec = .W }},
                // %EX prints like %X (no year).
                'X' => &.{ H, colon, M, colon, .{ .spec = .S_int } },
                'y' => &.{.{ .spec = .y, .guard = guard }},
                'Y' => &.{.{ .spec = .Y, .guard = guard }},
                'z' => if (mod != 0) &.{.{ .spec = .z_colon }} else &.{.{ .spec = .z }},
                'Z' => &.{.{ .spec = .Z }},
                '%' => &.{.{ .spec = .lit, .lit = "%" }},
                else => return error.InvalidOptions,
            });
        }
        return f;
    }
};

/// Write one row of `fmt` for `f` into `dst` (at least `fmt.max_row` bytes); return the length.
fn writeRow(dst: []u8, fmt: *const Format, f: *const Fields, digits: u8) usize {
    var n: usize = 0;
    const c = f.civ;
    for (fmt.ops.items) |op| {
        const d = dst[n..];
        if (op.guard and (c.year < 0 or c.year > 9999)) {
            d[0] = '?';
            n += 1;
            continue;
        }
        switch (op.spec) {
            .lit => {
                @memcpy(d[0..op.lit.len], op.lit);
                n += op.lit.len;
            },
            .Y => n += putYear(d, c.year),
            .m => {
                put2(d, c.month);
                n += 2;
            },
            .d => {
                put2(d, c.day);
                n += 2;
            },
            .e => {
                put2(d, c.day);
                if (c.day < 10) d[0] = ' ';
                n += 2;
            },
            .H => {
                put2(d, @divTrunc(f.sod, 3600));
                n += 2;
            },
            .I => {
                const h = @mod(@divTrunc(f.sod, 3600), 12);
                put2(d, if (h == 0) 12 else h);
                n += 2;
            },
            .M => {
                put2(d, @mod(@divTrunc(f.sod, 60), 60));
                n += 2;
            },
            .S, .S_int => {
                put2(d, @mod(f.sod, 60));
                n += 2;
                if (op.spec == .S) n += putFrac(dst[n..], f.frac, digits);
            },
            .p => {
                d[0..2].* = if (f.sod < 43_200) "AM".* else "PM".*;
                n += 2;
            },
            .a, .A => {
                const name = weekday_names[@intCast(@mod(f.days + 4, 7))];
                const s = if (op.spec == .a) name[0..3] else name;
                @memcpy(d[0..s.len], s);
                n += s.len;
            },
            .b, .B => {
                const name = month_names[@intCast(c.month - 1)];
                const s = if (op.spec == .b) name[0..3] else name;
                @memcpy(d[0..s.len], s);
                n += s.len;
            },
            .c, .x => {
                put2(d, c.month);
                d[2] = '/';
                put2(d[3..], c.day);
                d[5] = '/';
                put2(d[6..], @mod(c.year, 100));
                n += 8;
                if (op.spec == .c) {
                    d[8] = ' ';
                    putTime(d[9..], f.sod);
                    n += 9;
                }
            },
            .C => n += putSigned(d, @divFloor(c.year, 100), 2),
            .y => {
                put2(d, @mod(c.year, 100));
                n += 2;
            },
            .G => n += putSigned(d, temporal.isoWeek(f.days).year, 4),
            .g => {
                put2(d, @mod(temporal.isoWeek(f.days).year, 100));
                n += 2;
            },
            .V => {
                put2(d, temporal.isoWeek(f.days).week);
                n += 2;
            },
            .j => {
                const yd: u64 = @intCast(temporal.dayOfYear(f.days));
                d[0] = @intCast('0' + yd / 100);
                put2(d[1..], yd % 100);
                n += 3;
            },
            .u => {
                d[0] = @intCast('0' + @mod(f.days + 3, 7) + 1);
                n += 1;
            },
            .w => {
                d[0] = @intCast('0' + @mod(f.days + 4, 7));
                n += 1;
            },
            .U, .W => {
                const yday = temporal.dayOfYear(f.days) - 1;
                const wday = @mod(f.days + 4, 7);
                const start = if (op.spec == .U) wday else @mod(wday + 6, 7);
                put2(d, @divTrunc(yday + 7 - start, 7));
                n += 2;
            },
            .z, .z_colon => n += tz.writeOffset(d, f.info.offset, op.spec == .z_colon),
            .Z => {
                @memcpy(d[0..f.info.abbr.len], f.info.abbr);
                n += f.info.abbr.len;
            },
        }
    }
    return n;
}

fn strftimeExec(ctx: *ExecContext, args: []const Datum, options: Options) KernelError!Datum {
    if (args.len != 1) return error.InvalidArity;
    const o: core.StrftimeOptions = if (options == .strftime) options.strftime else .{};
    if (!std.mem.eql(u8, o.locale, "C")) return error.NotImplemented;
    var fmt = try Format.compile(ctx.allocator, o.format);
    defer fmt.deinit(ctx.allocator);
    const dt = args[0].dataType();
    const zone_name: ?[]const u8 = if (dt == .timestamp) dt.timestamp.timezone else null;
    if (zone_name == null and fmt.names_zone) return error.InvalidInput;
    var zone = if (zone_name) |name| try temporal_zone.resolve(ctx, name) else tz.utc;
    defer zone.deinit(ctx.allocator);
    // The chunks get the compiled format and the zone through a custom options payload.
    const state: StrftimeState = .{ .fmt = &fmt, .zone = &zone };
    return dense.execElementwiseWith(ctx, args, .{ .custom = .{ .tag = "strftime", .payload = &state } }, string_type, true, strftimeChunk);
}

const StrftimeState = struct { fmt: *const Format, zone: *const tz.Zone };

fn strftimeChunk(ctx: *ExecContext, values: []const ExecChunkValue, len: usize, options: Options) KernelError!ArrayRef {
    const state: *const StrftimeState = @ptrCast(@alignCast(options.custom.payload.?));
    var arr = try dense.chunkArray(ctx.allocator, values[0]);
    defer arr.release();
    const data = arr.data();
    const scale = scaleOf(data.data_type) orelse return error.UnsupportedType;
    var out = try dense.VarBinaryOutput.init(ctx.allocator, string_type, len, try dense.combinedValidity(ctx.allocator, len, &.{data}), len * @min(state.fmt.max_row, 64));
    errdefer out.deinit();
    switch (data.data_type) {
        .date32, .time32 => try strftimeRows(i32, data, scale, state, &out),
        else => try strftimeRows(i64, data, scale, state, &out),
    }
    return out.finish();
}

fn strftimeRows(comptime V: type, data: *const ArrayData, scale: Scale, state: *const StrftimeState, out: *dense.VarBinaryOutput) KernelError!void {
    const vals = (PrimitiveArray(V){ .data = data }).values() catch return error.InvalidInput;
    const bits = dense.validityBits(data);
    var cursor = tz.Cursor.init(state.zone);
    for (vals, 0..) |v, i| {
        if (bits) |b| if (!bitmap.bitIsSet(b, data.offset + i)) {
            try out.commit(0);
            continue;
        };
        const f = fieldsOf(v, scale, &cursor);
        const slot = try out.reserve(state.fmt.max_row);
        try out.commit(writeRow(slot, state.fmt, &f, scale.digits));
    }
}

// ------------------------------------------------------------------------------ casts to text

/// Days of the years -32767..32767 (Arrow prints other dates as out of range).
const min_days: i64 = -12_687_428;
const max_days: i64 = 11_248_737;

/// Cast a temporal array (it passed `validateLayout`) to a string layout.
pub fn formatCast(ctx: *ExecContext, data: *const ArrayData, to: DataType) KernelError!ArrayRef {
    const len = data.length;
    var out = try dense.VarBinaryOutput.init(ctx.allocator, to, len, try dense.combinedValidity(ctx.allocator, len, &.{data}), len * 24);
    errdefer out.deinit();
    switch (data.data_type) {
        .duration => try castRows(i64, .duration, data, .{ .per = 1, .digits = 0 }, &tz.utc, .none, &out),
        .date32 => try castRows(i32, .date32, data, scaleOf(data.data_type).?, &tz.utc, .none, &out),
        .date64 => try castRows(i64, .date64, data, scaleOf(data.data_type).?, &tz.utc, .none, &out),
        .time32 => try castRows(i32, .time32, data, scaleOf(data.data_type).?, &tz.utc, .none, &out),
        .time64 => try castRows(i64, .time64, data, scaleOf(data.data_type).?, &tz.utc, .none, &out),
        .timestamp => |t| if (t.timezone) |name| {
            var zone = try temporal_zone.resolve(ctx, name);
            defer zone.deinit(ctx.allocator);
            try castRows(i64, .timestamp, data, scaleOf(data.data_type).?, &zone, if (std.mem.eql(u8, name, "UTC")) .z else .offset, &out);
        } else try castRows(i64, .timestamp, data, scaleOf(data.data_type).?, &tz.utc, .none, &out),
        else => return error.UnsupportedType,
    }
    return out.finish();
}

/// What follows a timestamp: nothing (naive), `Z` (the timezone `UTC`) or the offset.
const Suffix = enum { none, z, offset };

fn castRows(comptime V: type, comptime tag: TypeTag, data: *const ArrayData, scale: Scale, zone: *const tz.Zone, suffix: Suffix, out: *dense.VarBinaryOutput) KernelError!void {
    const vals = (PrimitiveArray(V){ .data = data }).values() catch return error.InvalidInput;
    const bits = dense.validityBits(data);
    var cursor = tz.Cursor.init(zone);
    for (vals, 0..) |v, i| {
        if (bits) |b| if (!bitmap.bitIsSet(b, data.offset + i)) {
            try out.commit(0);
            continue;
        };
        const slot = try out.reserve(64);
        const x: i64 = v;
        if (tag == .duration) {
            try out.commit(putSigned(slot, x, 1));
            continue;
        }
        if (tag == .time32 or tag == .time64) {
            if (x < 0 or x >= 86_400 * scale.per) {
                try out.commit(outOfRange(slot, x));
                continue;
            }
            const secs = @divTrunc(x, scale.per);
            putTime(slot, secs);
            try out.commit(8 + putFrac(slot[8..], x - secs * scale.per, scale.digits));
            continue;
        }
        // Arrow checks the floored day but prints a date64 truncated to the day; a zoned timestamp skips the range check and
        // prints its year as the date library's 16-bit year.
        if (suffix == .none) {
            const days = @divFloor(x *% scale.mul, scale.per * 86_400);
            if (days < min_days or days > max_days) {
                try out.commit(outOfRange(slot, x));
                continue;
            }
        }
        var f = fieldsOf(if (tag == .date64) @divTrunc(x, 86_400_000) * 86_400_000 else x, scale, &cursor);
        if (suffix != .none) f.civ.year = @as(i16, @truncate(f.civ.year));
        var n = putDate(slot, f.civ);
        if (tag == .timestamp) {
            slot[n] = ' ';
            putTime(slot[n + 1 ..], f.sod);
            n += 9;
            n += putFrac(slot[n..], f.frac, scale.digits);
            switch (suffix) {
                .none => {},
                .z => {
                    slot[n] = 'Z';
                    n += 1;
                },
                .offset => n += tz.writeOffset(slot[n..], f.info.offset, false),
            }
        }
        try out.commit(n);
    }
}

fn outOfRange(slot: []u8, x: i64) usize {
    const head = "<value out of range: ";
    @memcpy(slot[0..head.len], head);
    var n = head.len + putSigned(slot[head.len..], x, 1);
    slot[n] = '>';
    n += 1;
    return n;
}

// ------------------------------------------------------------------------------ casts from text

inline fn digit(c: u8) ?u8 {
    return if (c >= '0' and c <= '9') c - '0' else null;
}

fn num2(s: []const u8) ?i64 {
    const a = digit(s[0]) orelse return null;
    const b = digit(s[1]) orelse return null;
    return @as(i64, a) * 10 + b;
}

/// `YYYY-MM-DD` (a valid date) as days after 1970-01-01.
fn parseDate(s: []const u8) ?i64 {
    if (s.len < 10 or s[4] != '-' or s[7] != '-') return null;
    const y = (num2(s[0..2]) orelse return null) * 100 + (num2(s[2..4]) orelse return null);
    const m = num2(s[5..7]) orelse return null;
    const d = num2(s[8..10]) orelse return null;
    if (m < 1 or m > 12 or d < 1 or d > daysInMonth(y, m)) return null;
    return temporal.daysFromCivil(y, m, d);
}

fn daysInMonth(y: i64, m: i64) i64 {
    const days = [_]i64{ 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 };
    return days[@intCast(m - 1)] + @intFromBool(m == 2 and temporal.isLeap(y));
}

/// `hh` (0..23) as seconds.
fn parseHH(s: []const u8) ?i64 {
    const h = num2(s[0..2]) orelse return null;
    return if (h <= 23) h * 3600 else null;
}

/// `hh:mm` as seconds.
fn parseHH_MM(s: []const u8) ?i64 {
    if (s[2] != ':') return null;
    const m = num2(s[3..5]) orelse return null;
    if (m > 59) return null;
    return (parseHH(s) orelse return null) + m * 60;
}

/// `hhmm` as seconds.
fn parseHHMM(s: []const u8) ?i64 {
    const m = num2(s[2..4]) orelse return null;
    if (m > 59) return null;
    return (parseHH(s) orelse return null) + m * 60;
}

/// `hh:mm:ss` as seconds.
fn parseHH_MM_SS(s: []const u8) ?i64 {
    if (s[5] != ':') return null;
    const sec = num2(s[6..8]) orelse return null;
    if (sec > 59) return null;
    return (parseHH_MM(s) orelse return null) + sec;
}

/// The fraction digits after a `.` in units of `unit` (at most the unit's digits, at least one).
fn parseSubSeconds(s: []const u8, unit: TimeUnit) ?i64 {
    const digits = fracDigits(unit);
    if (s.len == 0 or s.len > digits) return null;
    var v: i64 = 0;
    for (s) |c| v = v * 10 + (digit(c) orelse return null);
    for (s.len..digits) |_| v *= 10;
    return v;
}

/// `secs` seconds plus `sub` units as a count of `unit`, or null on overflow.
fn toUnit(secs: i64, sub: i64, unit: TimeUnit) ?i64 {
    const scaled = std.math.mul(i64, secs, perSecond(unit)) catch return null;
    return std.math.add(i64, scaled, sub) catch null;
}

/// Arrow's `ParseTimestampISO8601`: the instant and whether a zone offset was present.
pub fn parseTimestamp(s_in: []const u8, unit: TimeUnit) ?struct { value: i64, zoned: bool } {
    var s = s_in;
    if (s.len < 10) return null;
    var secs = (parseDate(s) orelse return null) * 86_400;
    if (s.len == 10) return .{ .value = toUnit(secs, 0, unit) orelse return null, .zoned = false };
    if (s[10] != ' ' and s[10] != 'T') return null;
    var zoned = false;
    var offset: i64 = 0;
    var len = s.len;
    if (s[len - 1] == 'Z') {
        len -= 1;
        zoned = true;
    } else if (s[len - 3] == '+' or s[len - 3] == '-') {
        len -= 3;
        offset = parseHH(s[len + 1 ..]) orelse return null;
        if (s[len] == '+') offset = -offset;
        zoned = true;
    } else if (s[len - 5] == '+' or s[len - 5] == '-') {
        len -= 5;
        offset = parseHHMM(s[len + 1 ..]) orelse return null;
        if (s[len] == '+') offset = -offset;
        zoned = true;
    } else if ((s[len - 6] == '+' or s[len - 6] == '-') and s[len - 3] == ':') {
        len -= 6;
        offset = parseHH_MM(s[len + 1 ..]) orelse return null;
        if (s[len] == '+') offset = -offset;
        zoned = true;
    }
    s = s[0..len];
    secs += switch (len) {
        13 => parseHH(s[11..]) orelse return null,
        16 => parseHH_MM(s[11..]) orelse return null,
        19, 21...29 => parseHH_MM_SS(s[11..]) orelse return null,
        else => return null,
    };
    secs += offset;
    if (len <= 19) return .{ .value = toUnit(secs, 0, unit) orelse return null, .zoned = zoned };
    if (s[19] != '.') return null;
    const sub = parseSubSeconds(s[20..], unit) orelse return null;
    return .{ .value = toUnit(secs, sub, unit) orelse return null, .zoned = zoned };
}

/// Arrow's `ParseTime`: `hh:mm`, `hh:mm:ss`, `hh:mm:ss.f+` in units of `unit`.
fn parseTime(s: []const u8, unit: TimeUnit) ?i64 {
    if (s.len == 5) return (parseHH_MM(s) orelse return null) * perSecond(unit);
    if (s.len < 8) return null;
    const secs = parseHH_MM_SS(s) orelse return null;
    if (s.len == 8) return secs * perSecond(unit);
    if (s[8] != '.') return null;
    return secs * perSecond(unit) + (parseSubSeconds(s[9..], unit) orelse return null);
}

/// Cast a string-layout array (it passed `validateLayout`) to a temporal type; text that does
/// not parse is `InvalidCast`.
pub fn parseCast(ctx: *ExecContext, arr: ArrayRef, to: DataType) KernelError!ArrayRef {
    const data = arr.data();
    const rows = try dense.StringOperand.init(.{ .array = arr });
    const bits = dense.validityBits(data);
    return switch (to) {
        .date32, .time32 => parseRows(i32, ctx, data, rows, bits, to),
        .date64, .timestamp, .time64, .duration => parseRows(i64, ctx, data, rows, bits, to),
        else => error.UnsupportedType,
    };
}

fn parseRows(comptime V: type, ctx: *ExecContext, data: *const ArrayData, rows: dense.StringOperand, bits: ?[]const u8, to: DataType) KernelError!ArrayRef {
    var out = try dense.DenseBinaryOutput(V).initValidity(ctx.allocator, data.length, data, null, to);
    errdefer out.deinit();
    const dst = out.values();
    switch (rows) {
        inline else => |r| for (dst, 0..) |*d, i| {
            d.* = 0;
            if (bits) |b| if (!bitmap.bitIsSet(b, data.offset + i)) continue;
            d.* = @intCast(try parseOne(r.get(i), to));
        },
    }
    return out.finish() catch error.OutOfMemory;
}

fn parseOne(s: []const u8, to: DataType) KernelError!i64 {
    switch (to) {
        .date32, .date64 => {
            if (s.len != 10) return error.InvalidCast;
            const days = parseDate(s) orelse return error.InvalidCast;
            return if (to == .date32) days else days * 86_400_000;
        },
        .timestamp => |t| {
            const r = parseTimestamp(s, t.unit) orelse return error.InvalidCast;
            if (r.zoned != (t.timezone != null)) return error.InvalidCast;
            return r.value;
        },
        .time32, .time64 => |t| return parseTime(s, t.unit) orelse error.InvalidCast,
        .duration => return @import("cast.zig").parseNumber(i64, s),
        else => return error.UnsupportedType,
    }
}

// ------------------------------------------------------------------------------ strptime

fn strptimeType(_: []const Datum, options: Options) KernelError!DataType {
    const o = options.strptime;
    return .{ .timestamp = .{ .unit = o.unit, .timezone = if (formatHasZone(o.format)) "UTC" else null } };
}

/// Arrow's test for a zoned strptime result: a `%z` conversion (`%%z` is a literal).
fn formatHasZone(fmt: []const u8) bool {
    var i: usize = 0;
    while (i + 1 < fmt.len) : (i += 1) {
        if (fmt[i] == '%') {
            if (fmt[i + 1] == 'z') return true;
            i += 1;
        }
    }
    return false;
}

/// The `struct tm` fields strptime fills.
pub const Tm = struct { year: i32 = 0, mon: i32 = 0, mday: i32 = 0, hour: i32 = 0, min: i32 = 0, sec: i32 = 0, gmtoff: i32 = 0 };

inline fn isSpace(c: u8) bool {
    return c == ' ' or (c >= '\t' and c <= '\r');
}

/// musl's strptime over `s` from `start`: the end position, or null when the row does not
/// match. Nested formats (`%D`, `%R`, `%T`) recurse once with their own century state.
pub fn strptimeAt(s: []const u8, start: usize, fmt: []const u8, tm: *Tm) ?usize {
    var p = start;
    var f: usize = 0;
    var century: i64 = 0;
    var relyear: i64 = 0;
    var want_century: u2 = 0;
    while (f < fmt.len) {
        if (fmt[f] != '%') {
            if (isSpace(fmt[f])) {
                while (p < s.len and isSpace(s[p])) p += 1;
            } else {
                if (p >= s.len or s[p] != fmt[f]) return null;
                p += 1;
            }
            f += 1;
            continue;
        }
        f += 1;
        if (f < fmt.len and fmt[f] == '+') f += 1;
        var w: i64 = -1;
        if (f < fmt.len and digit(fmt[f]) != null) {
            w = 0;
            while (f < fmt.len) : (f += 1) {
                const dd = digit(fmt[f]) orelse break;
                w = @min(w * 10 + dd, 1 << 40);
            }
        }
        if (f >= fmt.len) return null;
        const c = fmt[f];
        f += 1;
        // numeric_range: digits up to the magnitude of min + range, value < min + range.
        var range_dest: ?*i32 = null;
        var min: i64 = 0;
        var range: i64 = 0;
        var adj: i64 = 0;
        var dummy: i32 = 0;
        switch (c) {
            'C' => {
                if (w < 0) w = 2;
                want_century |= 2;
                century = numericDigits(s, &p, w) orelse return null;
                continue;
            },
            'y' => {
                want_century |= 1;
                relyear = numericDigits(s, &p, 2) orelse return null;
                continue;
            },
            'Y' => {
                if (w < 0) w = 4;
                want_century = 0;
                const y = numericDigits(s, &p, w) orelse return null;
                tm.year = std.math.cast(i32, y - 1900) orelse return null;
                continue;
            },
            'd', 'e' => {
                range_dest = &tm.mday;
                min = 1;
                range = 31;
            },
            'D' => {
                p = strptimeAt(s, p, "%m/%d/%y", tm) orelse return null;
                continue;
            },
            'H' => {
                range_dest = &tm.hour;
                range = 24;
            },
            'I' => {
                range_dest = &tm.hour;
                min = 1;
                range = 12;
            },
            'j' => {
                range_dest = &dummy;
                min = 1;
                range = 366;
            },
            'm' => {
                range_dest = &tm.mon;
                min = 1;
                range = 12;
                adj = 1;
            },
            'M' => {
                range_dest = &tm.min;
                range = 60;
            },
            'n', 't' => {
                while (p < s.len and isSpace(s[p])) p += 1;
                continue;
            },
            'R' => {
                p = strptimeAt(s, p, "%H:%M", tm) orelse return null;
                continue;
            },
            'S' => {
                range_dest = &tm.sec;
                range = 61;
            },
            'T' => {
                p = strptimeAt(s, p, "%H:%M:%S", tm) orelse return null;
                continue;
            },
            'U', 'W' => {
                range_dest = &dummy;
                range = 54;
            },
            'w' => {
                range_dest = &dummy;
                range = 7;
            },
            'z' => {
                tm.gmtoff = parseGmtoff(s, &p) orelse return null;
                continue;
            },
            '%' => {
                if (p >= s.len or s[p] != '%') return null;
                p += 1;
                continue;
            },
            else => return null,
        }
        if (p >= s.len or digit(s[p]) == null) return null;
        var v: i64 = 0;
        var i: i64 = 1;
        while (i <= min + range and p < s.len) : (i *= 10) {
            const dd = digit(s[p]) orelse break;
            v = v * 10 + dd;
            p += 1;
        }
        if (v - min >= range) return null;
        range_dest.?.* = @intCast(v - adj);
    }
    if (want_century != 0) {
        var y = relyear;
        if (want_century & 2 != 0) y += century * 100 - 1900 else if (y <= 68) y += 100;
        tm.year = std.math.cast(i32, y) orelse return null;
    }
    return p;
}

/// musl's numeric_digits: an optional sign and 1..`w` digits (`w` 0 reads none after the first
/// digit check); null past the `int` range.
fn numericDigits(s: []const u8, p: *usize, w: i64) ?i64 {
    var neg = false;
    if (p.* < s.len and s[p.*] == '+') {
        p.* += 1;
    } else if (p.* < s.len and s[p.*] == '-') {
        neg = true;
        p.* += 1;
    }
    if (p.* >= s.len or digit(s[p.*]) == null) return null;
    var v: i64 = 0;
    var i: i64 = 0;
    while (i < w and p.* < s.len) : (i += 1) {
        const dd = digit(s[p.*]) orelse break;
        v = v * 10 + dd;
        if (v > std.math.maxInt(i32)) return null;
        p.* += 1;
    }
    return if (neg) -v else v;
}

/// glibc's `%z`: optional spaces, then `Z`, or a sign and `hh`, `hhmm` or `hh:mm` (minutes
/// below 60), as seconds east of UTC.
fn parseGmtoff(s: []const u8, p: *usize) ?i32 {
    while (p.* < s.len and isSpace(s[p.*])) p.* += 1;
    if (p.* < s.len and s[p.*] == 'Z') {
        p.* += 1;
        return 0;
    }
    if (p.* >= s.len or (s[p.*] != '+' and s[p.*] != '-')) return null;
    const neg = s[p.*] == '-';
    p.* += 1;
    var v: i32 = 0;
    var n: usize = 0;
    while (n < 4 and p.* < s.len) {
        const dd = digit(s[p.*]) orelse break;
        v = v * 10 + dd;
        p.* += 1;
        n += 1;
        if (n == 2 and p.* + 1 < s.len and s[p.*] == ':' and digit(s[p.* + 1]) != null) p.* += 1;
    }
    if (n == 2) {
        v *= 100;
    } else if (n != 4 or @mod(v, 100) >= 60) return null;
    const off = @divTrunc(v, 100) * 3600 + @mod(v, 100) * 60;
    return if (neg) -off else off;
}

/// Parse one row with `fmt` into a count of `unit` (Arrow's `ParseTimestampStrptime`).
pub fn strptimeValue(s: []const u8, fmt: []const u8, unit: TimeUnit) ?i64 {
    var tm: Tm = .{};
    const end = strptimeAt(s, 0, fmt, &tm) orelse return null;
    if (end != s.len) return null;
    const days = temporal.daysFromCivil(@as(i64, tm.year) + 1900, @as(i64, tm.mon) + 1, @max(tm.mday, 1));
    const secs = days * 86_400 + @as(i64, tm.hour) * 3600 + @as(i64, tm.min) * 60 + tm.sec - tm.gmtoff;
    return toUnit(secs, 0, unit);
}

fn strptimeExec(ctx: *ExecContext, args: []const Datum, options: Options) KernelError!Datum {
    if (args.len != 1) return error.InvalidArity;
    const o = options.strptime;
    if (o.format.len > max_format_len or std.mem.indexOfScalar(u8, o.format, 0) != null) return error.InvalidOptions;
    return dense.execElementwiseWith(ctx, args, options, try strptimeType(args, options), true, strptimeChunk);
}

fn strptimeChunk(ctx: *ExecContext, values: []const ExecChunkValue, len: usize, options: Options) KernelError!ArrayRef {
    const o = options.strptime;
    var arr = try dense.chunkArray(ctx.allocator, values[0]);
    defer arr.release();
    const data = arr.data();
    const rows = try dense.StringOperand.init(.{ .array = arr });
    const bits = dense.validityBits(data);
    var out = try dense.DenseBinaryOutput(i64).initValidity(ctx.allocator, len, data, null, try strptimeType(&.{}, options));
    errdefer out.deinit();
    const dst = out.values();
    switch (rows) {
        inline else => |r| for (dst, 0..) |*d, i| {
            d.* = 0;
            if (bits) |b| if (!bitmap.bitIsSet(b, data.offset + i)) continue;
            d.* = strptimeValue(r.get(i), o.format, o.unit) orelse {
                if (!o.error_is_null) return error.InvalidInput;
                if (out.validity == null) {
                    var v = OwnedBuffer.initUninit(ctx.allocator, bitmap.byteLength(len)) catch return error.OutOfMemory;
                    @memset(v.data[0..bitmap.byteLength(len)], 0);
                    bitmap.setBits(v.data, 0, len);
                    out.validity = v;
                }
                bitmap.clearBit(out.validity.?.data, i);
                continue;
            };
        },
    }
    return out.finish() catch error.OutOfMemory;
}
