//! Timezones of timestamp types (Arrow C++ resolves them through the timezone database): `UTC`,
//! fixed offsets `+HH:MM` / `+HHMM` (hours 0..23, minutes 0..59, as Arrow) and IANA names.
//!
//! A name is read from the system zoneinfo (`/usr/share/zoneinfo/<name>`, Linux and macOS, when
//! `ExecConfig.io` is set) and otherwise from the table bundled in `tzdata/tzdata.bin`
//! (`tools/gen_tzdata.py`, IANA tzdata `bundledVersion()`). Both go through one bounds-checked
//! TZif reader (RFC 8536 versions 1 to 4): every count is checked against the input before a
//! read, the input is capped at `max_tzif_len`, transitions must ascend, offsets stay within
//! RFC 8536's -89999..93599 seconds, abbreviations are printable ASCII of at most `max_abbr_len`
//! bytes, and the footer TZ string (POSIX, with the version 3 extensions) is parsed with its own
//! limits. Leap second records are checked and ignored (as Arrow's date library does).
const std = @import("std");
const builtin = @import("builtin");
const core = @import("core.zig");
const temporal = @import("temporal.zig");

const KernelError = core.KernelError;

/// Longest zone abbreviation kept (longer TZif abbreviations are rejected).
pub const max_abbr_len = 16;
/// Largest TZif input read (system files are about 4 KiB).
pub const max_tzif_len = 256 << 10;
/// Transition instants beyond this magnitude are rejected (keeps offset arithmetic exact).
const max_time: i64 = 1 << 59;
/// Footer rules apply within this many seconds of 1970 (standard time beyond).
const footer_limit: i64 = 1 << 46;

const bundle = @embedFile("tzdata/tzdata.bin");

/// The IANA tzdata release of the bundled table.
pub fn bundledVersion() []const u8 {
    if (bundle.len < 5) return "";
    return bundle[5..][0..@min(bundle[4], bundle.len - 5)];
}

/// The local time of one instant: offset east of UTC in seconds, daylight saving time, and the
/// abbreviation (`%Z`).
pub const Info = struct { offset: i32, is_dst: bool = false, abbr: []const u8 };

/// One interval `[begin, end)` of UTC seconds with a constant local time type.
pub const Span = struct { begin: i64, end: i64, info: Info };

/// A local time type of a TZif file.
const TType = struct { off: i32, dst: bool, abbr: []const u8 };

/// A transition rule of a POSIX TZ string: day `day` of the year counted from 1 without
/// February 29 (`julian`), from 0 with it (`doy`), or day of week `day` of week `week` (5: last)
/// of month `mon` (`mwd`); at `time` seconds of local wall time.
const Rule = struct {
    kind: enum { julian, doy, mwd },
    day: i32,
    week: i32 = 0,
    mon: i32 = 0,
    time: i32 = 7200,
};

/// A POSIX TZ string (the TZif footer): standard time, and daylight time with its two rules.
const Posix = struct {
    std_off: i32,
    std_abbr: []const u8,
    dst: ?struct { off: i32, abbr: []const u8, start: Rule, end: Rule } = null,
};

/// The rules of a named zone (allocated; `Zone.deinit` frees them).
pub const Rules = struct {
    times: []i64,
    idx: []u8,
    types: []TType,
    footer: ?Posix,
    /// Abbreviation bytes `types` and `footer` slice into.
    chars: []u8,

    pub fn deinit(self: *Rules, allocator: std.mem.Allocator) void {
        allocator.free(self.times);
        allocator.free(self.idx);
        allocator.free(self.types);
        allocator.free(self.chars);
    }

    fn typeInfo(self: *const Rules, i: usize) Info {
        const t = self.types[i];
        return .{ .offset = t.off, .is_dst = t.dst, .abbr = t.abbr };
    }

    /// The interval of `at` and its local time type.
    pub fn span(self: *const Rules, at: i64) Span {
        const n = self.times.len;
        const min = std.math.minInt(i64);
        const max = std.math.maxInt(i64);
        if (n == 0) {
            if (self.footer) |p| return footerSpan(p, at);
            return .{ .begin = min, .end = max, .info = self.typeInfo(0) };
        }
        // Before the first transition: time type 0 (RFC 8536 3.2).
        if (at < self.times[0]) return .{ .begin = min, .end = self.times[0], .info = self.typeInfo(0) };
        // The last transition at or before `at`.
        var lo: usize = 0;
        var hi: usize = n;
        while (hi - lo > 1) {
            const mid = lo + (hi - lo) / 2;
            if (self.times[mid] <= at) lo = mid else hi = mid;
        }
        if (lo == n - 1) {
            if (self.footer) |p| {
                var s = footerSpan(p, at);
                s.begin = @max(s.begin, self.times[n - 1]);
                return s;
            }
            return .{ .begin = self.times[n - 1], .end = max, .info = self.typeInfo(self.idx[n - 1]) };
        }
        return .{ .begin = self.times[lo], .end = self.times[lo + 1], .info = self.typeInfo(self.idx[lo]) };
    }
};

/// A resolved timezone: a fixed offset (`rules` null) or the rules of a named zone. Fixed
/// offsets print `+HHMM` as their abbreviation, as Arrow.
pub const Zone = struct {
    offset: i32 = 0,
    abbr_buf: [8]u8 = undefined,
    abbr_len: u8 = 0,
    rules: ?*Rules = null,

    /// The interval of the UTC instant `at` (seconds).
    pub fn span(self: *const Zone, at: i64) Span {
        if (self.rules) |r| return r.span(at);
        return .{ .begin = std.math.minInt(i64), .end = std.math.maxInt(i64), .info = .{ .offset = self.offset, .abbr = self.abbr_buf[0..self.abbr_len] } };
    }

    /// The local time of the UTC instant `at` (seconds).
    pub inline fn info(self: *const Zone, at: i64) Info {
        if (self.rules == null) return .{ .offset = self.offset, .abbr = self.abbr_buf[0..self.abbr_len] };
        return self.span(at).info;
    }

    /// Whether every instant has the same offset (a fixed offset or UTC).
    pub fn isFixed(self: *const Zone) bool {
        return self.rules == null;
    }

    pub fn deinit(self: *Zone, allocator: std.mem.Allocator) void {
        if (self.rules) |r| {
            r.deinit(allocator);
            allocator.destroy(r);
            self.rules = null;
        }
    }

    /// The UTC instants of the local time `l` (seconds).
    pub fn local(self: *const Zone, l: i64) Local {
        if (self.rules == null) return .{ .unique = self.offset };
        // Every solution u of u + offset(u) = l lies within 26 hours of l: walk the intervals
        // that cover that window.
        const window = 27 * 3600;
        var t = l -| window;
        var sols: [2]i32 = undefined;
        var n: usize = 0;
        var gap: ?i64 = null;
        var steps: usize = 0;
        while (steps < 16) : (steps += 1) {
            const s = self.span(t);
            const lo = s.begin +| s.info.offset;
            const hi = s.end +| s.info.offset;
            if (l >= lo and l < hi) {
                if (n < 2) sols[n] = s.info.offset;
                n += 1;
            } else if (l < lo and n == 0 and gap == null) gap = s.begin;
            if (s.end == std.math.maxInt(i64) or s.end > l +| window) break;
            t = s.end;
        }
        return switch (n) {
            0 => .{ .gap = gap orelse l },
            1 => .{ .unique = sols[0] },
            else => .{ .ambiguous = .{ sols[0], sols[1] } },
        };
    }
};

/// How a local time maps to UTC: one offset, two (an ambiguous time: the earlier instant's
/// offset first), or none (a time in a gap: the UTC instant the gap ends at).
pub const Local = union(enum) { unique: i32, ambiguous: [2]i32, gap: i64 };

/// A cache of the last interval for row loops over mostly ordered instants.
pub const Cursor = struct {
    zone: *const Zone,
    begin: i64 = 1,
    end: i64 = 0,
    cached: Info = .{ .offset = 0, .abbr = "" },
    /// Local times in `[lbegin, lend)` map to UTC uniquely with offset `loff`.
    lbegin: i64 = 1,
    lend: i64 = 0,
    loff: i32 = 0,

    pub fn init(zone: *const Zone) Cursor {
        return .{ .zone = zone };
    }

    pub inline fn info(self: *Cursor, at: i64) Info {
        if (self.zone.rules == null) return self.zone.info(at);
        if (at >= self.begin and at < self.end) return self.cached;
        const s = self.zone.span(at);
        self.begin = s.begin;
        self.end = s.end;
        self.cached = s.info;
        return s.info;
    }

    pub inline fn local(self: *Cursor, l: i64) Local {
        if (self.zone.rules == null) return .{ .unique = self.zone.offset };
        if (l >= self.lbegin and l < self.lend) return .{ .unique = self.loff };
        const r = self.zone.local(l);
        if (r == .unique) {
            // The interval of the solution, shrunk by 26 hours on both sides: no neighbor's
            // local range reaches into it.
            const s = self.zone.span(l - r.unique);
            self.lbegin = (s.begin +| s.info.offset) +| 26 * 3600;
            self.lend = (s.end +| s.info.offset) -| 26 * 3600;
            self.loff = r.unique;
        }
        return r;
    }
};

/// The zone of a naive timestamp, a date or a time: UTC.
pub const utc: Zone = blk: {
    var z: Zone = .{};
    z.abbr_buf[0..3].* = "UTC".*;
    z.abbr_len = 3;
    break :blk z;
};

/// Return whether `tz` names UTC without a lookup (`UTC`, `Etc/UTC`, `+00:00`).
pub fn isUtc(tz: []const u8) bool {
    for ([_][]const u8{ "UTC", "Etc/UTC", "+00:00" }) |name| if (std.mem.eql(u8, tz, name)) return true;
    return false;
}

/// Parse `+HH:MM` or `+HHMM` (or `-`) into seconds east of UTC (Arrow's accepted offsets).
pub fn parseOffset(s: []const u8) ?i32 {
    if (s.len != 5 and s.len != 6) return null;
    if (s[0] != '+' and s[0] != '-') return null;
    if (s.len == 6 and s[3] != ':') return null;
    const mm = s[s.len - 2 ..];
    const hh = s[1..3];
    for ([_]u8{ hh[0], hh[1], mm[0], mm[1] }) |c| if (c < '0' or c > '9') return null;
    const h: i32 = (hh[0] - '0') * 10 + (hh[1] - '0');
    const m: i32 = (mm[0] - '0') * 10 + (mm[1] - '0');
    if (h > 23 or m > 59) return null;
    const off = h * 3600 + m * 60;
    return if (s[0] == '-') -off else off;
}

/// Write `+HHMM` (or `+HH:MM` with `colon`) for `offset` seconds into `buf`; return the length.
pub fn writeOffset(buf: []u8, offset: i32, colon: bool) usize {
    const a: u32 = @abs(offset);
    const h = a / 3600;
    const m = a / 60 % 60;
    buf[0] = if (offset < 0) '-' else '+';
    buf[1] = @intCast('0' + h / 10 % 10);
    buf[2] = @intCast('0' + h % 10);
    var i: usize = 3;
    if (colon) {
        buf[3] = ':';
        i = 4;
    }
    buf[i] = @intCast('0' + m / 10);
    buf[i + 1] = @intCast('0' + m % 10);
    return i + 2;
}

/// Whether `name` may name a zoneinfo file: 1..255 bytes of `A-Z a-z 0-9 _ + -` in `/`-separated
/// parts, none empty (no absolute path, no `.` or `..`).
pub fn validName(name: []const u8) bool {
    if (name.len == 0 or name.len > 255) return false;
    var parts = std.mem.splitScalar(u8, name, '/');
    while (parts.next()) |p| {
        if (p.len == 0) return false;
        for (p) |c| if (!(std.ascii.isAlphanumeric(c) or c == '_' or c == '+' or c == '-')) return false;
    }
    return true;
}

/// Resolve the timezone string of a timestamp type: UTC and fixed offsets directly, a name from
/// the system zoneinfo (with `io`, on Linux and macOS) or else the bundled table. A name that is
/// neither an offset nor a known zone is `InvalidInput` (Arrow: "Cannot locate or parse
/// timezone").
pub fn resolve(allocator: std.mem.Allocator, io: ?std.Io, name: []const u8) KernelError!Zone {
    if (std.mem.eql(u8, name, "UTC") or std.mem.eql(u8, name, "Etc/UTC")) return utc;
    if (parseOffset(name)) |off| {
        var z: Zone = .{ .offset = off };
        z.abbr_len = @intCast(writeOffset(&z.abbr_buf, off, false));
        return z;
    }
    if (!validName(name)) return error.InvalidInput;
    if (io) |handle| if (comptime builtin.os.tag == .linux or builtin.os.tag.isDarwin()) {
        if (systemZone(allocator, handle, name)) |z| return z else |err| if (err == error.OutOfMemory) return error.OutOfMemory;
    };
    const blob = bundledZone(name) orelse return error.InvalidInput;
    return zoneOf(allocator, blob);
}

fn systemZone(allocator: std.mem.Allocator, io: std.Io, name: []const u8) KernelError!Zone {
    var path_buf: [300]u8 = undefined;
    const path = std.fmt.bufPrint(&path_buf, "/usr/share/zoneinfo/{s}", .{name}) catch return error.InvalidInput;
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(max_tzif_len)) catch |err| return switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        else => error.InvalidInput,
    };
    defer allocator.free(bytes);
    return zoneOf(allocator, bytes);
}

fn zoneOf(allocator: std.mem.Allocator, bytes: []const u8) KernelError!Zone {
    const r = allocator.create(Rules) catch return error.OutOfMemory;
    errdefer allocator.destroy(r);
    r.* = parseTzif(allocator, bytes) catch |err| return switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        else => error.InvalidInput,
    };
    return .{ .rules = r };
}

/// The TZif blob of `name` in the bundled table (every bound checked), or null.
pub fn bundledZone(name: []const u8) ?[]const u8 {
    return lookupBundle(bundle, name);
}

/// `bundledZone` over any table bytes (the fuzz target feeds untrusted ones).
pub fn lookupBundle(table: []const u8, name: []const u8) ?[]const u8 {
    if (table.len < 5 or !std.mem.eql(u8, table[0..4], "ZTZB")) return null;
    var p: usize = 5 + @as(usize, table[4]);
    if (p + 4 > table.len) return null;
    const count = std.mem.readInt(u32, table[p..][0..4], .little);
    p += 4;
    var i: u32 = 0;
    while (i < count) : (i += 1) {
        if (p >= table.len) return null;
        const n = table[p];
        if (p + 1 + n + 8 > table.len) return null;
        const entry = table[p + 1 ..][0..n];
        const off = std.mem.readInt(u32, table[p + 1 + n ..][0..4], .little);
        const len = std.mem.readInt(u32, table[p + 5 + n ..][0..4], .little);
        p += 1 + n + 8;
        if (!std.mem.eql(u8, entry, name)) continue;
        if (off > table.len or len > table.len - off or len > max_tzif_len) return null;
        return table[off..][0..len];
    }
    return null;
}

// ------------------------------------------------------------------------------ TZif reader

pub const TzifError = error{ OutOfMemory, InvalidTzif };

/// Bounds-checked reader over the input.
const Reader = struct {
    b: []const u8,
    p: usize = 0,

    fn take(self: *Reader, n: usize) TzifError![]const u8 {
        if (n > self.b.len - self.p) return error.InvalidTzif;
        defer self.p += n;
        return self.b[self.p..][0..n];
    }

    fn int(self: *Reader, comptime T: type) TzifError!T {
        return std.mem.readInt(T, (try self.take(@sizeOf(T)))[0..@sizeOf(T)], .big);
    }
};

const Header = struct { version: u8, isut: u32, isstd: u32, leap: u32, timecnt: u32, typecnt: u32, charcnt: u32 };

fn readHeader(r: *Reader) TzifError!Header {
    const magic = try r.take(4);
    if (!std.mem.eql(u8, magic, "TZif")) return error.InvalidTzif;
    const version = (try r.take(1))[0];
    if (version != 0 and (version < '2' or version > '4')) return error.InvalidTzif;
    _ = try r.take(15);
    return .{
        .version = version,
        .isut = try r.int(u32),
        .isstd = try r.int(u32),
        .leap = try r.int(u32),
        .timecnt = try r.int(u32),
        .typecnt = try r.int(u32),
        .charcnt = try r.int(u32),
    };
}

/// Size of a data block with `time_size`-byte times; an error past `limit`.
fn blockSize(h: Header, time_size: usize, limit: usize) TzifError!usize {
    // Each count is below 2^32 and multiplied by at most 12: no overflow in u64.
    const size: u64 = @as(u64, h.timecnt) * (time_size + 1) + @as(u64, h.typecnt) * 6 + h.charcnt + @as(u64, h.leap) * (time_size + 4) + h.isstd + h.isut;
    if (size > limit) return error.InvalidTzif;
    return @intCast(size);
}

/// Parse a TZif file (RFC 8536, versions 1 to 4) into allocated rules.
pub fn parseTzif(allocator: std.mem.Allocator, bytes: []const u8) TzifError!Rules {
    if (bytes.len > max_tzif_len) return error.InvalidTzif;
    var r: Reader = .{ .b = bytes };
    var h = try readHeader(&r);
    var time_size: usize = 4;
    if (h.version != 0) {
        // Skip the version 1 block; the second header describes the 64-bit data.
        _ = try r.take(try blockSize(h, 4, bytes.len));
        const h2 = try readHeader(&r);
        if (h2.version != h.version) return error.InvalidTzif;
        h = h2;
        time_size = 8;
    }
    _ = try blockSize(h, time_size, bytes.len - r.p);
    // RFC 8536 3.1: typecnt and charcnt nonzero, isstdcnt / isutcnt zero or typecnt.
    if (h.typecnt == 0 or h.typecnt > 256 or h.charcnt == 0) return error.InvalidTzif;
    if ((h.isstd != 0 and h.isstd != h.typecnt) or (h.isut != 0 and h.isut != h.typecnt)) return error.InvalidTzif;

    const times = try allocator.alloc(i64, h.timecnt);
    errdefer allocator.free(times);
    for (times, 0..) |*t, i| {
        t.* = if (time_size == 8) try r.int(i64) else try r.int(i32);
        if (t.* < -max_time or t.* > max_time) return error.InvalidTzif;
        if (i > 0 and t.* <= times[i - 1]) return error.InvalidTzif;
    }
    const idx = try allocator.alloc(u8, h.timecnt);
    errdefer allocator.free(idx);
    @memcpy(idx, try r.take(h.timecnt));
    for (idx) |x| if (x >= h.typecnt) return error.InvalidTzif;

    const raw_types = try r.take(@as(usize, h.typecnt) * 6);
    const chars_in = try r.take(h.charcnt);
    _ = try r.take(@as(usize, h.leap) * (time_size + 4));
    for (try r.take(h.isstd)) |x| if (x > 1) return error.InvalidTzif;
    for (try r.take(h.isut)) |x| if (x > 1) return error.InvalidTzif;

    // The footer (version 2+): "\n" TZ string "\n".
    var footer_text: []const u8 = "";
    if (h.version != 0) {
        if ((try r.take(1))[0] != '\n') return error.InvalidTzif;
        const rest = bytes[r.p..];
        const end = std.mem.indexOfScalar(u8, rest, '\n') orelse return error.InvalidTzif;
        footer_text = rest[0..end];
    }

    // Abbreviations live in one allocation: the TZif characters, then the footer's names.
    const chars = try allocator.alloc(u8, h.charcnt + footer_text.len);
    errdefer allocator.free(chars);
    @memcpy(chars[0..h.charcnt], chars_in);
    @memcpy(chars[h.charcnt..], footer_text);

    const types = try allocator.alloc(TType, h.typecnt);
    errdefer allocator.free(types);
    for (types, 0..) |*t, i| {
        const raw = raw_types[i * 6 ..][0..6];
        const off = std.mem.readInt(i32, raw[0..4], .big);
        if (off < -89_999 or off > 93_599 or raw[4] > 1 or raw[5] >= h.charcnt) return error.InvalidTzif;
        const tail = chars[raw[5]..h.charcnt];
        const len = std.mem.indexOfScalar(u8, tail, 0) orelse return error.InvalidTzif;
        const abbr = tail[0..len];
        if (!validAbbr(abbr)) return error.InvalidTzif;
        t.* = .{ .off = off, .dst = raw[4] == 1, .abbr = abbr };
    }
    const footer: ?Posix = if (footer_text.len == 0) null else parsePosix(chars[h.charcnt..]) orelse return error.InvalidTzif;
    return .{ .times = times, .idx = idx, .types = types, .footer = footer, .chars = chars };
}

fn validAbbr(abbr: []const u8) bool {
    if (abbr.len > max_abbr_len) return false;
    for (abbr) |c| if (c < '!' or c > '~') return false;
    return true;
}

// ------------------------------------------------------------------------------ POSIX TZ

const PosixReader = struct {
    s: []const u8,
    p: usize = 0,

    fn peek(self: *const PosixReader) u8 {
        return if (self.p < self.s.len) self.s[self.p] else 0;
    }

    /// An abbreviation: `<...>` of `[A-Za-z0-9+-]`, or 3+ letters.
    fn abbr(self: *PosixReader) ?[]const u8 {
        const start = self.p;
        if (self.peek() == '<') {
            self.p += 1;
            const a = self.p;
            while (self.p < self.s.len and self.s[self.p] != '>') : (self.p += 1) {
                const c = self.s[self.p];
                if (!(std.ascii.isAlphanumeric(c) or c == '+' or c == '-')) return null;
            }
            if (self.p >= self.s.len) return null;
            const out = self.s[a..self.p];
            self.p += 1;
            return if (out.len >= 3 and out.len <= max_abbr_len) out else null;
        }
        while (self.p < self.s.len and std.ascii.isAlphabetic(self.s[self.p])) self.p += 1;
        const out = self.s[start..self.p];
        return if (out.len >= 3 and out.len <= max_abbr_len) out else null;
    }

    /// `[+-]hh[:mm[:ss]]` with hours up to `max_hours`, in seconds.
    fn hms(self: *PosixReader, max_hours: i32) ?i32 {
        var sign: i32 = 1;
        if (self.peek() == '+' or self.peek() == '-') {
            if (self.peek() == '-') sign = -1;
            self.p += 1;
        }
        const h = self.num(3) orelse return null;
        if (h > max_hours) return null;
        var secs = h * 3600;
        if (self.peek() == ':') {
            self.p += 1;
            const m = self.num(2) orelse return null;
            if (m > 59) return null;
            secs += m * 60;
            if (self.peek() == ':') {
                self.p += 1;
                const s = self.num(2) orelse return null;
                if (s > 59) return null;
                secs += s;
            }
        }
        return sign * secs;
    }

    /// 1 to `max_digits` decimal digits.
    fn num(self: *PosixReader, max_digits: usize) ?i32 {
        var v: i32 = 0;
        var n: usize = 0;
        while (n < max_digits and self.p < self.s.len and std.ascii.isDigit(self.s[self.p])) : (n += 1) {
            v = v * 10 + (self.s[self.p] - '0');
            self.p += 1;
        }
        return if (n == 0) null else v;
    }

    fn rule(self: *PosixReader) ?Rule {
        var r: Rule = undefined;
        if (self.peek() == 'J') {
            self.p += 1;
            const d = self.num(3) orelse return null;
            if (d < 1 or d > 365) return null;
            r = .{ .kind = .julian, .day = d };
        } else if (self.peek() == 'M') {
            self.p += 1;
            const m = self.num(2) orelse return null;
            if (self.peek() != '.') return null;
            self.p += 1;
            const w = self.num(1) orelse return null;
            if (self.peek() != '.') return null;
            self.p += 1;
            const d = self.num(1) orelse return null;
            if (m < 1 or m > 12 or w < 1 or w > 5 or d > 6) return null;
            r = .{ .kind = .mwd, .day = d, .week = w, .mon = m };
        } else {
            const d = self.num(3) orelse return null;
            if (d > 365) return null;
            r = .{ .kind = .doy, .day = d };
        }
        r.time = 7200;
        if (self.peek() == '/') {
            self.p += 1;
            // RFC 8536 3.3.1: the time may be negative and up to 167 hours.
            r.time = self.hms(167) orelse return null;
        }
        return r;
    }
};

/// Parse a POSIX TZ string (null when malformed). Offsets are stored east of UTC.
fn parsePosix(s: []const u8) ?Posix {
    var r: PosixReader = .{ .s = s };
    const std_abbr = r.abbr() orelse return null;
    const std_off = -(r.hms(24) orelse return null);
    if (r.p == s.len) return .{ .std_off = std_off, .std_abbr = std_abbr };
    const dst_abbr = r.abbr() orelse return null;
    var dst_off = std_off + 3600;
    if (r.peek() != ',' and r.p < s.len) dst_off = -(r.hms(24) orelse return null);
    // A daylight time without rules takes the US rules (POSIX leaves it to the implementation;
    // Go and glibc do this).
    var start: Rule = .{ .kind = .mwd, .day = 0, .week = 2, .mon = 3 };
    var end: Rule = .{ .kind = .mwd, .day = 0, .week = 1, .mon = 11 };
    if (r.p < s.len) {
        if (r.peek() != ',') return null;
        r.p += 1;
        start = r.rule() orelse return null;
        if (r.peek() != ',') return null;
        r.p += 1;
        end = r.rule() orelse return null;
    }
    if (r.p != s.len) return null;
    return .{ .std_off = std_off, .std_abbr = std_abbr, .dst = .{ .off = dst_off, .abbr = dst_abbr, .start = start, .end = end } };
}

/// Seconds after the start of `year` (UTC) at which `rule` fires, the wall clock read in
/// `off` (Go's `tzruleTime`).
fn ruleTime(year: i64, rule: Rule, off: i32) i64 {
    const jan1 = temporal.daysFromCivil(year, 1, 1);
    const day: i64 = switch (rule.kind) {
        .julian => rule.day - 1 + @as(i64, @intFromBool(temporal.isLeap(year) and rule.day >= 60)),
        .doy => rule.day,
        .mwd => blk: {
            const first = temporal.daysFromCivil(year, rule.mon, 1);
            const dow = @mod(first + 4, 7); // 0 = Sunday
            var d = first + @mod(rule.day - dow, 7);
            const next_month = if (rule.mon == 12) temporal.daysFromCivil(year + 1, 1, 1) else temporal.daysFromCivil(year, rule.mon + 1, 1);
            var w: i32 = 1;
            while (w < rule.week and d + 7 < next_month) : (w += 1) d += 7;
            break :blk d - jan1;
        },
    };
    return day * 86_400 + rule.time - off;
}

/// The interval of `at` under a footer rule (Go's `tzset`: exact around the transitions, the
/// year's start and end otherwise).
fn footerSpan(p: Posix, at: i64) Span {
    const std_info: Info = .{ .offset = p.std_off, .abbr = p.std_abbr };
    const d = p.dst orelse return .{ .begin = std.math.minInt(i64), .end = std.math.maxInt(i64), .info = std_info };
    // Past about 2 million years from 1970 the rule is not evaluated (keeps the year arithmetic
    // in range): standard time.
    if (at >= footer_limit) return .{ .begin = footer_limit, .end = std.math.maxInt(i64), .info = std_info };
    if (at < -footer_limit) return .{ .begin = std.math.minInt(i64), .end = -footer_limit, .info = std_info };
    const year = temporal.civil(@divFloor(at, 86_400)).year;
    const ys = temporal.daysFromCivil(year, 1, 1) * 86_400;
    const ye = temporal.daysFromCivil(year + 1, 1, 1) * 86_400;
    var start = ys + ruleTime(year, d.start, p.std_off);
    var end = ys + ruleTime(year, d.end, d.off);
    var outside = std_info;
    var inside: Info = .{ .offset = d.off, .is_dst = true, .abbr = d.abbr };
    if (end < start) {
        std.mem.swap(i64, &start, &end);
        std.mem.swap(Info, &outside, &inside);
    }
    if (at < start) return .{ .begin = ys, .end = start, .info = outside };
    if (at >= end) return .{ .begin = end, .end = ye, .info = outside };
    return .{ .begin = start, .end = end, .info = inside };
}

test "tz POSIX footer rules and the bundled table" {
    const allocator = std.testing.allocator;
    const p = parsePosix("EST5EDT,M3.2.0,M11.1.0").?;
    // 2021-03-14 07:00 UTC starts EDT, 2021-11-07 06:00 UTC ends it.
    try std.testing.expectEqual(@as(i32, -18000), footerSpan(p, 1615705199).info.offset);
    try std.testing.expectEqual(@as(i32, -14400), footerSpan(p, 1615705200).info.offset);
    try std.testing.expectEqual(@as(i32, -14400), footerSpan(p, 1636264799).info.offset);
    try std.testing.expectEqual(@as(i32, -18000), footerSpan(p, 1636264800).info.offset);
    // Southern hemisphere (Sydney): DST from October to April.
    const syd = parsePosix("AEST-10AEDT,M10.1.0,M4.1.0/3").?;
    try std.testing.expectEqual(@as(i32, 39600), footerSpan(syd, 1609459200).info.offset);
    try std.testing.expectEqual(@as(i32, 36000), footerSpan(syd, 1625097600).info.offset);
    try std.testing.expect(parsePosix("<+0545>-5:45") != null);
    try std.testing.expect(parsePosix("<-03>3<-02>,M3.5.0/-2,M10.5.0/-1") != null);
    for ([_][]const u8{ "", "E", "EST", "EST5EDT,M13.1.0,M1.1.0", "EST5EDT,M3.2.0", "<AB>5", "EST25", "EST5EDT,J0,J1" }) |bad| {
        try std.testing.expect(parsePosix(bad) == null);
    }
    try std.testing.expectEqualStrings("2026f", bundledVersion());
    var z = try resolve(allocator, null, "America/New_York");
    defer z.deinit(allocator);
    try std.testing.expectEqual(@as(i32, -18000), z.info(1615705199).offset);
    try std.testing.expectEqualStrings("EDT", z.info(1615705200).abbr);
    try std.testing.expectEqual(Local{ .gap = 1615705200 }, z.local(1615705200 - 18000 + 1800));
    try std.testing.expectEqual(Local{ .ambiguous = .{ -14400, -18000 } }, z.local(1636263000 - 14400));
    try std.testing.expectEqual(Local{ .unique = -18000 }, z.local(0));
    try std.testing.expectError(error.InvalidInput, resolve(allocator, null, "Bad/Zone"));
    try std.testing.expectError(error.InvalidInput, resolve(allocator, null, "../etc/passwd"));
    try std.testing.expectError(error.InvalidInput, resolve(allocator, null, "/usr/share/zoneinfo/UTC"));
}

test "tz TZif reader rejects malformed input without reading past it" {
    const allocator = std.testing.allocator;
    const good = bundledZone("Europe/Paris").?;
    var rules = try parseTzif(allocator, good);
    rules.deinit(allocator);
    // Every truncation fails cleanly.
    for (0..good.len) |n| {
        if (parseTzif(allocator, good[0..n])) |r| {
            var rr = r;
            rr.deinit(allocator);
            return error.TestUnexpectedResult;
        } else |err| try std.testing.expectEqual(error.InvalidTzif, err);
    }
    // A huge count in the second header neither allocates nor reads.
    const bad = try allocator.dupe(u8, good);
    defer allocator.free(bad);
    std.mem.writeInt(u32, bad[44 + 32 ..][0..4], 0xFFFF_FFFF, .big);
    try std.testing.expectError(error.InvalidTzif, parseTzif(allocator, bad));
    try std.testing.expect(lookupBundle("ZTZB\x05", "UTC") == null);
    try std.testing.expect(lookupBundle("ZTZB\x00\xff\xff\xff\xff\x03UTC", "UTC") == null);
}
