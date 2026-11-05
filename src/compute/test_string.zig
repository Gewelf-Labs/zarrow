const std = @import("std");
const common = @import("test_common.zig");
const unicode_tables = @import("unicode_tables.zig");

const ArrayRef = common.ArrayRef;
const DataType = common.DataType;
const Datum = common.Datum;
const ExecContext = common.ExecContext;
const Options = common.Options;
const Scalar = common.Scalar;
const none = common.none;
const TypeTag = std.meta.Tag(DataType);

const string_tags = [_]TypeTag{ .string, .large_string, .string_view };
const max_i64 = std.math.maxInt(i64);

fn slice(start: i64, stop: i64, step: i64) Options {
    return .{ .slice = .{ .start = start, .stop = stop, .step = step } };
}

fn pattern(p: []const u8, ignore_case: bool) Options {
    return .{ .match_substring = .{ .pattern = p, .ignore_case = ignore_case } };
}

test "string hand-written values: case mapping, trims, slices, reverse, lengths, predicates and join on every string layout" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);

    inline for (string_tags) |tag| {
        // The last value is longer than 12 bytes, so string_view stores it out of line.
        var a = try common.makeStrings(tag, allocator, &.{ "Hello", null, "", "  mIxEd 9 \t", "héllo wörld😀" });
        defer a.release();
        const args = [_]Datum{.{ .array = a }};
        try common.expectInvokeStrings(&ctx, "ascii_upper", &args, none, &.{ "HELLO", null, "", "  MIXED 9 \t", "HéLLO WöRLD😀" });
        try common.expectInvokeStrings(&ctx, "ascii_lower", &args, none, &.{ "hello", null, "", "  mixed 9 \t", "héllo wörld😀" });
        // utf8_upper / utf8_lower map every cased code point (Unicode tables).
        try common.expectInvokeStrings(&ctx, "utf8_upper", &args, none, &.{ "HELLO", null, "", "  MIXED 9 \t", "HÉLLO WÖRLD😀" });
        try common.expectInvokeStrings(&ctx, "utf8_lower", &args, none, &.{ "hello", null, "", "  mixed 9 \t", "héllo wörld😀" });
        try common.expectInvokeStrings(&ctx, "ascii_title", &args, none, &.{ "Hello", null, "", "  Mixed 9 \t", "HéLlo WöRld😀" });
        try common.expectInvokeStrings(&ctx, "utf8_title", &args, none, &.{ "Hello", null, "", "  Mixed 9 \t", "Héllo Wörld😀" });
        try common.expectInvokeStrings(&ctx, "utf8_trim", &args, .{ .trim = .{ .characters = "H😀d \t" } }, &.{ "ello", null, "", "mIxEd 9", "héllo wörl" });
        try common.expectInvokeStrings(&ctx, "utf8_ltrim", &args, .{ .trim = .{ .characters = "hé " } }, &.{ "Hello", null, "", "mIxEd 9 \t", "llo wörld😀" });
        try common.expectInvokeStrings(&ctx, "utf8_rtrim", &args, .{ .trim = .{ .characters = "😀ldr" } }, &.{ "Hello", null, "", "  mIxEd 9 \t", "héllo wö" });
        try common.expectInvokeStrings(&ctx, "ascii_trim_whitespace", &args, none, &.{ "Hello", null, "", "mIxEd 9", "héllo wörld😀" });
        try common.expectInvokeStrings(&ctx, "ascii_ltrim_whitespace", &args, none, &.{ "Hello", null, "", "mIxEd 9 \t", "héllo wörld😀" });
        try common.expectInvokeStrings(&ctx, "ascii_rtrim_whitespace", &args, none, &.{ "Hello", null, "", "  mIxEd 9", "héllo wörld😀" });
        try common.expectInvokeStrings(&ctx, "ascii_trim", &args, .{ .trim = .{ .characters = "Ho \t" } }, &.{ "ell", null, "", "mIxEd 9", "héllo wörld😀" });
        try common.expectInvokeStrings(&ctx, "ascii_ltrim", &args, .{ .trim = .{ .characters = "H " } }, &.{ "ello", null, "", "mIxEd 9 \t", "héllo wörld😀" });
        try common.expectInvokeStrings(&ctx, "ascii_rtrim", &args, .{ .trim = .{ .characters = "o\t " } }, &.{ "Hell", null, "", "  mIxEd 9", "héllo wörld😀" });
        try common.expectInvokeStrings(&ctx, "utf8_reverse", &args, none, &.{ "olleH", null, "", "\t 9 dExIm  ", "😀dlröw olléh" });
        var ascii_rows = try common.makeStrings(tag, allocator, &.{ "Hello", null, "", "  mIxEd 9 \t" });
        defer ascii_rows.release();
        try common.expectInvokeStrings(&ctx, "ascii_reverse", &.{.{ .array = ascii_rows }}, none, &.{ "olleH", null, "", "\t 9 dExIm  " });
        try common.expectInvoke(bool, &ctx, .scalar, "string_is_ascii", &args, &.{ true, null, true, true, false });

        // Slices: Python s[start:stop:step] over code points (expected values from Python).
        var s = try common.makeStrings(tag, allocator, &.{"héllo wörld😀"});
        defer s.release();
        const sargs = [_]Datum{.{ .array = s }};
        const Case = struct { o: Options, want: []const u8 };
        for ([_]Case{
            .{ .o = slice(1, 5, 1), .want = "éllo" },
            .{ .o = slice(-3, max_i64, 1), .want = "ld😀" },
            .{ .o = slice(0, max_i64, 2), .want = "hlowrd" },
            .{ .o = slice(-1, -100, -1), .want = "😀dlröw olléh" },
            .{ .o = slice(5, 1, -2), .want = " l" },
            .{ .o = slice(100, max_i64, 1), .want = "" },
            .{ .o = slice(-100, 2, 1), .want = "hé" },
            .{ .o = slice(2, max_i64, -1), .want = "" },
        }) |c| try common.expectInvokeStrings(&ctx, "utf8_slice_codeunits", &sargs, c.o, &.{c.want});
        const len_t = if (tag == .large_string) i64 else i32;
        try common.expectInvoke(len_t, &ctx, .scalar, "utf8_length", &sargs, &.{12});
        try common.expectInvoke(len_t, &ctx, .scalar, "binary_length", &sargs, &.{17});

        // Unicode White_Space trimming (U+3000, U+2003; U+00A0 inside is kept), like Python's str.strip.
        var w = try common.makeStrings(tag, allocator, &.{ "\u{3000} a\u{00a0}b \t\u{2003}", "\u{2028}", "x" });
        defer w.release();
        try common.expectInvokeStrings(&ctx, "utf8_trim_whitespace", &.{.{ .array = w }}, none, &.{ "a\u{00a0}b", "", "x" });
        try common.expectInvokeStrings(&ctx, "utf8_ltrim_whitespace", &.{.{ .array = w }}, none, &.{ "a\u{00a0}b \t\u{2003}", "", "x" });
        try common.expectInvokeStrings(&ctx, "utf8_rtrim_whitespace", &.{.{ .array = w }}, none, &.{ "\u{3000} a\u{00a0}b", "", "x" });
        try common.expectInvoke(bool, &ctx, .scalar, "utf8_is_space", &.{.{ .array = w }}, &.{ false, true, false });

        // Predicates: true only for a non-empty all-matching row.
        var p = try common.makeStrings(tag, allocator, &.{ "abc", "ab1", "123", "", " \t", "é", "Abc" });
        defer p.release();
        const pargs = [_]Datum{.{ .array = p }};
        try common.expectInvoke(bool, &ctx, .scalar, "ascii_is_alpha", &pargs, &.{ true, false, false, false, false, false, true });
        try common.expectInvoke(bool, &ctx, .scalar, "ascii_is_decimal", &pargs, &.{ false, false, true, false, false, false, false });
        try common.expectInvoke(bool, &ctx, .scalar, "ascii_is_space", &pargs, &.{ false, false, false, false, true, false, false });
        try common.expectInvoke(bool, &ctx, .scalar, "utf8_is_alpha", &pargs, &.{ true, false, false, false, false, true, true });
        try common.expectInvoke(bool, &ctx, .scalar, "utf8_is_digit", &pargs, &.{ false, false, true, false, false, false, false });
        try common.expectInvokeWith(bool, &ctx, "starts_with", &pargs, pattern("ab", false), &.{ true, true, false, false, false, false, false });
        try common.expectInvokeWith(bool, &ctx, "starts_with", &pargs, pattern("ab", true), &.{ true, true, false, false, false, false, true });
        try common.expectInvokeWith(bool, &ctx, "ends_with", &pargs, pattern("C", true), &.{ true, false, false, false, false, false, true });
        try common.expectInvokeWith(bool, &ctx, "match_substring", &pargs, pattern("b", false), &.{ true, true, false, false, false, false, true });
        try common.expectInvokeWith(bool, &ctx, "match_substring", &pargs, pattern("BC", true), &.{ true, false, false, false, false, false, true });
        try common.expectInvokeWith(bool, &ctx, "match_substring", &pargs, pattern("", false), &.{ true, true, true, true, true, true, true });

        // binary_join_element_wise: the last argument is the separator.
        var x = try common.makeStrings(tag, allocator, &.{ "a", null, "c", "d" });
        defer x.release();
        var y = try common.makeStrings(tag, allocator, &.{ "1", "2", null, "4" });
        defer y.release();
        var sep = try common.makeStrings(tag, allocator, &.{ "-", "-", "-", null });
        defer sep.release();
        const jargs = [_]Datum{ .{ .array = x }, .{ .array = y }, .{ .array = sep } };
        try common.expectInvokeStrings(&ctx, "binary_join_element_wise", &jargs, none, &.{ "a-1", null, null, null });
        try common.expectInvokeStrings(&ctx, "binary_join_element_wise", &jargs, .{ .join = .{ .null_handling = .skip } }, &.{ "a-1", "2", "c", null });
        try common.expectInvokeStrings(&ctx, "binary_join_element_wise", &jargs, .{ .join = .{ .null_handling = .replace, .null_replacement = "NA" } }, &.{ "a-1", "NA-2", "c-NA", null });
        const comma = Scalar.init(@unionInit(DataType, @tagName(tag), {}), .{ .string = ", " });
        try common.expectInvokeStrings(&ctx, "binary_join_element_wise", &.{ .{ .array = x }, .{ .array = x }, .{ .scalar = comma } }, none, &.{ "a, a", null, "c, c", "d, d" });
        try common.expectInvokeStrings(&ctx, "binary_join_element_wise", &.{ .{ .array = x }, .{ .scalar = comma } }, none, &.{ "a", null, "c", "d" });
    }

    // binary_length on the binary layouts.
    inline for (.{ TypeTag.binary, TypeTag.large_binary, TypeTag.binary_view }) |tag| {
        var b = try common.makeStrings(tag, allocator, &.{ "\x00\xff", null, "0123456789abcdef" });
        defer b.release();
        const len_t = if (tag == .large_binary) i64 else i32;
        try common.expectInvoke(len_t, &ctx, .scalar, "binary_length", &.{.{ .array = b }}, &.{ 2, null, 16 });
    }
}

// ------------------------------------------------------------------------------ oracle

const pieces = [_][]const u8{ "a", "B", "z", "Q", "0", "7", " ", "\t", "é", "€", "😀", "\u{3000}", "\u{00a0}", "ab", "Zy", "É", "ß", "Σ", "ǅ", "ı", "½", "٣", "ﬁ", "Ⱥ", "\u{10428}", "İ", "\u{212A}", "ⱥ" };

/// Random UTF-8 rows (ASCII letters and digits, ASCII and Unicode spaces, 2-, 3- and 4-byte code
/// points), about 15% null, allocated in `arena`.
fn randomRows(arena: std.mem.Allocator, rng: std.Random, len: usize) ![]?[]const u8 {
    const rows = try arena.alloc(?[]const u8, len);
    for (rows) |*r| {
        if (rng.uintLessThan(u8, 100) < 15) {
            r.* = null;
            continue;
        }
        var buf: std.ArrayList(u8) = .empty;
        for (0..rng.uintAtMost(usize, 9)) |_| try buf.appendSlice(arena, pieces[rng.uintLessThan(usize, pieces.len)]);
        r.* = buf.items;
    }
    return rows;
}

fn codepoints(arena: std.mem.Allocator, s: []const u8) ![][]const u8 {
    var out: std.ArrayList([]const u8) = .empty;
    var it = (try std.unicode.Utf8View.init(s)).iterator();
    while (it.nextCodepointSlice()) |cp| try out.append(arena, cp);
    return out.items;
}

fn isWhiteSpace(cp: []const u8) bool {
    const c = std.unicode.utf8Decode(cp) catch unreachable;
    for ([_]u21{ 9, 10, 11, 12, 13, 0x1C, 0x1D, 0x1E, 0x1F, 32, 0x85, 0xA0, 0x1680, 0x2000, 0x2001, 0x2002, 0x2003, 0x2004, 0x2005, 0x2006, 0x2007, 0x2008, 0x2009, 0x200A, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000 }) |w| {
        if (c == w) return true;
    }
    return false;
}

/// Reference of a simple case mapping: a linear scan of the generated ranges.
fn refCase(ranges: []const unicode_tables.CaseRange, cp: u21) u21 {
    for (ranges) |r| {
        const stride: u21 = if (r.stride2 == 1) 2 else 1;
        if (cp >= r.first and cp <= r.last and (cp - r.first) % stride == 0) return @intCast(@as(i32, cp) + r.delta);
    }
    return cp;
}

/// Reference of the general category: a binary search of the runs without the bucket index.
fn refCategory(cp: u21) unicode_tables.Category {
    var lo: usize = 0;
    var hi: usize = unicode_tables.category_runs.len;
    while (hi - lo > 1) {
        const mid = (lo + hi) / 2;
        if (unicode_tables.category_runs[mid] >> 5 <= cp) lo = mid else hi = mid;
    }
    return @enumFromInt(@as(u5, @truncate(unicode_tables.category_runs[lo])));
}

/// Reference of each string to string function on one row.
fn refTransform(arena: std.mem.Allocator, name: []const u8, s: []const u8, o: Options) ![]const u8 {
    const eq = std.mem.eql;
    if (eq(u8, name, "ascii_upper")) return std.ascii.allocUpperString(arena, s);
    if (eq(u8, name, "ascii_lower")) return std.ascii.allocLowerString(arena, s);
    if (eq(u8, name, "utf8_upper") or eq(u8, name, "utf8_lower")) {
        var out: std.ArrayList(u8) = .empty;
        var it = (try std.unicode.Utf8View.init(s)).iterator();
        while (it.nextCodepoint()) |cp| {
            const m = if (eq(u8, name, "utf8_upper")) refCase(&unicode_tables.upper_ranges, cp) else refCase(&unicode_tables.lower_ranges, cp);
            var buf: [4]u8 = undefined;
            try out.appendSlice(arena, buf[0 .. std.unicode.utf8Encode(m, &buf) catch unreachable]);
        }
        return out.items;
    }
    const ws = " \t\n\r\x0b\x0c";
    if (eq(u8, name, "ascii_trim_whitespace")) return std.mem.trim(u8, s, ws);
    if (eq(u8, name, "ascii_ltrim_whitespace")) return std.mem.trimStart(u8, s, ws);
    if (eq(u8, name, "ascii_rtrim_whitespace")) return std.mem.trimEnd(u8, s, ws);
    if (eq(u8, name, "ascii_trim")) return std.mem.trim(u8, s, o.trim.characters);
    const cps = try codepoints(arena, s);
    var out: std.ArrayList(u8) = .empty;
    if (std.mem.endsWith(u8, name, "trim_whitespace")) {
        var lo: usize = 0;
        var hi: usize = cps.len;
        if (!eq(u8, name, "utf8_rtrim_whitespace")) while (lo < hi and isWhiteSpace(cps[lo])) {
            lo += 1;
        };
        if (!eq(u8, name, "utf8_ltrim_whitespace")) while (hi > lo and isWhiteSpace(cps[hi - 1])) {
            hi -= 1;
        };
        for (cps[lo..hi]) |cp| try out.appendSlice(arena, cp);
    } else if (eq(u8, name, "utf8_reverse")) {
        var i = cps.len;
        while (i > 0) : (i -= 1) try out.appendSlice(arena, cps[i - 1]);
    } else {
        // utf8_slice_codeunits: visit every index in step order, keep those Python's slice selects.
        const n: i64 = @intCast(cps.len);
        const so = o.slice;
        const start = if (so.start < 0) so.start + n else so.start;
        const stop = if (so.stop < 0) so.stop + n else so.stop;
        const step_abs: i64 = @intCast(@abs(so.step));
        for (0..cps.len) |k| {
            const idx: i64 = @intCast(if (so.step > 0) k else cps.len - 1 - k);
            const keep = if (so.step > 0)
                idx >= start and idx < stop and @mod(idx - @max(start, 0), step_abs) == 0
            else
                idx <= start and idx > stop and @mod(@min(start, n - 1) - idx, step_abs) == 0;
            if (keep) try out.appendSlice(arena, cps[@intCast(idx)]);
        }
    }
    return out.items;
}

fn refPredicate(name: []const u8, s: []const u8, o: Options) bool {
    const eq = std.mem.eql;
    if (eq(u8, name, "string_is_ascii")) {
        for (s) |c| if (c >= 0x80) return false;
        return true;
    }
    if (eq(u8, name, "starts_with")) return std.mem.startsWith(u8, s, o.match_substring.pattern);
    if (eq(u8, name, "ends_with")) return std.mem.endsWith(u8, s, o.match_substring.pattern);
    if (eq(u8, name, "match_substring")) return std.mem.indexOf(u8, s, o.match_substring.pattern) != null;
    if (s.len == 0) return false;
    var it = (std.unicode.Utf8View.init(s) catch unreachable).iterator();
    while (it.nextCodepointSlice()) |cp| {
        const ascii = cp.len == 1;
        const cat = refCategory(std.unicode.utf8Decode(cp) catch unreachable);
        const ok = if (eq(u8, name, "ascii_is_space"))
            ascii and std.ascii.isWhitespace(cp[0])
        else if (eq(u8, name, "utf8_is_space"))
            isWhiteSpace(cp)
        else if (eq(u8, name, "ascii_is_alpha"))
            ascii and std.ascii.isAlphabetic(cp[0])
        else if (eq(u8, name, "ascii_is_decimal"))
            ascii and std.ascii.isDigit(cp[0])
        else if (eq(u8, name, "utf8_is_alpha"))
            cat == .Lu or cat == .Ll or cat == .Lt or cat == .Lm or cat == .Lo
        else if (eq(u8, name, "utf8_is_digit"))
            cat == .Nd or cat == .No
        else
            cat == .Nd or cat == .Nl or cat == .No; // utf8_is_numeric
        if (!ok) return false;
    }
    return true;
}

test "string oracle matches a per-row loop on random sliced inputs with nulls for every layout" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var prng = std.Random.DefaultPrng.init(0x57A1);
    const rng = prng.random();

    const T = struct { name: []const u8, o: Options };
    const transforms = [_]T{
        .{ .name = "ascii_upper", .o = none },
        .{ .name = "ascii_lower", .o = none },
        .{ .name = "utf8_upper", .o = none },
        .{ .name = "utf8_lower", .o = none },
        .{ .name = "ascii_trim_whitespace", .o = none },
        .{ .name = "ascii_ltrim_whitespace", .o = none },
        .{ .name = "ascii_rtrim_whitespace", .o = none },
        .{ .name = "ascii_trim", .o = .{ .trim = .{ .characters = "aB 0" } } },
        .{ .name = "utf8_trim_whitespace", .o = none },
        .{ .name = "utf8_ltrim_whitespace", .o = none },
        .{ .name = "utf8_rtrim_whitespace", .o = none },
        .{ .name = "utf8_reverse", .o = none },
        .{ .name = "utf8_slice_codeunits", .o = slice(1, 4, 1) },
        .{ .name = "utf8_slice_codeunits", .o = slice(-5, max_i64, 2) },
        .{ .name = "utf8_slice_codeunits", .o = slice(-1, -8, -3) },
    };
    const predicates = [_]T{
        .{ .name = "string_is_ascii", .o = none },
        .{ .name = "ascii_is_alpha", .o = none },
        .{ .name = "ascii_is_decimal", .o = none },
        .{ .name = "ascii_is_space", .o = none },
        .{ .name = "utf8_is_alpha", .o = none },
        .{ .name = "utf8_is_digit", .o = none },
        .{ .name = "utf8_is_space", .o = none },
        .{ .name = "utf8_is_numeric", .o = none },
        .{ .name = "starts_with", .o = pattern("a", false) },
        .{ .name = "ends_with", .o = pattern("😀", false) },
        .{ .name = "match_substring", .o = pattern("é", false) },
    };
    inline for (string_tags) |tag| {
        const rows = try randomRows(arena, rng, 420);
        var full = try common.makeStrings(tag, allocator, rows);
        defer full.release();
        var arr = try full.slice(11, 400);
        defer arr.release();
        const view = rows[11..][0..400];
        const args = [_]Datum{.{ .array = arr }};
        for (transforms) |t| {
            const expected = try arena.alloc(?[]const u8, 400);
            for (expected, view) |*e, r| e.* = if (r) |s| try refTransform(arena, t.name, s, t.o) else null;
            common.expectInvokeStrings(&ctx, t.name, &args, t.o, expected) catch |err| {
                std.debug.print("{s} on {s}\n", .{ t.name, @tagName(tag) });
                return err;
            };
        }
        for (predicates) |t| {
            const expected = try arena.alloc(?bool, 400);
            for (expected, view) |*e, r| e.* = if (r) |s| refPredicate(t.name, s, t.o) else null;
            try common.expectInvokeWith(bool, &ctx, t.name, &args, t.o, expected);
        }
        const L = if (tag == .large_string) i64 else i32;
        const lengths = try arena.alloc(?L, 400);
        for (lengths, view) |*e, r| e.* = if (r) |s| @intCast(std.unicode.utf8CountCodepoints(s) catch unreachable) else null;
        try common.expectInvoke(L, &ctx, .scalar, "utf8_length", &args, lengths);
        for (lengths, view) |*e, r| e.* = if (r) |s| @intCast(s.len) else null;
        try common.expectInvoke(L, &ctx, .scalar, "binary_length", &args, lengths);

        // Join the column with its reversed copy, separator "|", in every null mode.
        var other = try ctx.invokeScalar("utf8_reverse", &args, none);
        defer other.release();
        const sep = Scalar.init(@unionInit(DataType, @tagName(tag), {}), .{ .string = "|" });
        for ([_]common.core.JoinNullHandling{ .emit_null, .skip, .replace }) |mode| {
            const expected = try arena.alloc(?[]const u8, 400);
            for (expected, view) |*e, r| {
                const joined = if (r) |s| try std.mem.concat(arena, u8, &.{ s, "|", try refTransform(arena, "utf8_reverse", s, none) }) else null;
                e.* = switch (mode) {
                    .emit_null => joined,
                    .skip => joined orelse "",
                    .replace => joined orelse "?|?",
                };
            }
            try common.expectInvokeStrings(&ctx, "binary_join_element_wise", &.{ .{ .array = arr }, other, .{ .scalar = sep } }, .{ .join = .{ .null_handling = mode, .null_replacement = "?" } }, expected);
        }
    }
}

test "string Unicode case mapping and predicates on every case class (Arrow C++ results)" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);

    // Titlecase digraphs, final sigma, dotted / dotless i, ß (uppercase U+1E9E as in Arrow), a
    // ligature with only special casing, numbers of every category, spaces, mappings that change
    // the UTF-8 length (U+023A / U+2C65, U+0250), a combining mark with an uppercase, astral
    // letters (Deseret, Medefaidrin) and a Unicode 16 addition (U+1C89).
    const rows = [_][]const u8{
        "",                         "abc",                      "ABC",             "Abc",                      "aBc",                              "Hello World",
        "hello world",              "HELLO WORLD",              "\xc7\x85ungla",   "\xc7\x86",                 "\xc7\x85",                         "Stra\xc3\x9fe",
        "\xce\xa3\xce\x91\xce\xa3", "\xcf\x83\xce\xb1\xcf\x82", "\xc4\xb0stanbul", "\xc4\xb1I",                "123",                              "\xd9\xa1\xd9\xa2\xd9\xa3",
        "\xc2\xbd",                 "\xe2\x85\xab",             "\xc2\xb2",        "a1",                       "\xc3\xa9",                         "a\xc3\xa9",
        "\xc3\x89a",                " ",                        "\x09",            "\xc2\xa0",                 "\xe3\x80\x80",                     "\xef\xac\x81",
        "\xe2\x84\x8c",             "_A",                       "a_B",             "A1b",                      "Ab cD",                            "1A",
        "\xea\x9e\xb5",             "\xc8\xba",                 "\xe2\xb1\xa5",    "\xc9\x90",                 "\xcd\x85",                         "A\xcc\x80b",
        "\x7f",                     "~",                        "a\xe2\x80\x8b",   "\xc7\xb1\xc7\xb2\xc7\xb3", "\xf0\x90\x90\x80\xf0\x90\x90\xa8", "\xe1\xb2\x89",
        "\xf0\x96\xb9\x80x",
    };
    // One character per row: 1 = true (pyarrow 26).
    const predicates = [_]struct { name: []const u8, want: []const u8 }{
        .{ .name = "ascii_is_alnum", .want = "0111100000000000100001000000000001010000000000000" },
        .{ .name = "ascii_is_alpha", .want = "0111100000000000000000000000000000000000000000000" },
        .{ .name = "ascii_is_decimal", .want = "0000000000000000100000000000000000000000000000000" },
        .{ .name = "ascii_is_lower", .want = "0100001010000010000001011000000000000000000010001" },
        .{ .name = "ascii_is_printable", .want = "1111111100000000100001000100000111110000000100000" },
        .{ .name = "ascii_is_space", .want = "0000000000000000000000000110000000000000000000000" },
        .{ .name = "ascii_is_title", .want = "0001010000000001000000000000000100010000000000000" },
        .{ .name = "ascii_is_upper", .want = "0010000100000001000000000000000100010000000000000" },
        .{ .name = "utf8_is_alnum", .want = "0111100011111111111111111000011001011111000001111" },
        .{ .name = "utf8_is_alpha", .want = "0111100011111111000000111000011000001111000001111" },
        .{ .name = "utf8_is_decimal", .want = "0000000000000000110000000000000000000000000000000" },
        .{ .name = "utf8_is_digit", .want = "0000000000000000111010000000000000000000000000000" },
        .{ .name = "utf8_is_lower", .want = "0100001001000100000001110000010000001011100010000" },
        .{ .name = "utf8_is_numeric", .want = "0000000000000000111110000000000000000000000000000" },
        .{ .name = "utf8_is_printable", .want = "1111111111111111111111111100011111111111110101111" },
        .{ .name = "utf8_is_space", .want = "0000000000000000000000000111100000000000000000000" },
        .{ .name = "utf8_is_title", .want = "0001010010110010000100001000001100010100000000111" },
        .{ .name = "utf8_is_upper", .want = "0010000100001000000100000000001100010100000000010" },
    };
    const transforms = [_]struct { name: []const u8, want: []const []const u8 }{
        .{ .name = "utf8_upper", .want = &.{
            "",                         "ABC",                      "ABC",             "ABC",                      "ABC",                              "HELLO WORLD",
            "HELLO WORLD",              "HELLO WORLD",              "\xc7\x84UNGLA",   "\xc7\x84",                 "\xc7\x84",                         "STRA\xe1\xba\x9eE",
            "\xce\xa3\xce\x91\xce\xa3", "\xce\xa3\xce\x91\xce\xa3", "\xc4\xb0STANBUL", "II",                       "123",                              "\xd9\xa1\xd9\xa2\xd9\xa3",
            "\xc2\xbd",                 "\xe2\x85\xab",             "\xc2\xb2",        "A1",                       "\xc3\x89",                         "A\xc3\x89",
            "\xc3\x89A",                " ",                        "\x09",            "\xc2\xa0",                 "\xe3\x80\x80",                     "\xef\xac\x81",
            "\xe2\x84\x8c",             "_A",                       "A_B",             "A1B",                      "AB CD",                            "1A",
            "\xea\x9e\xb4",             "\xc8\xba",                 "\xc8\xba",        "\xe2\xb1\xaf",             "\xce\x99",                         "A\xcc\x80B",
            "\x7f",                     "~",                        "A\xe2\x80\x8b",   "\xc7\xb1\xc7\xb1\xc7\xb1", "\xf0\x90\x90\x80\xf0\x90\x90\x80", "\xe1\xb2\x89",
            "\xf0\x96\xb9\x80X",
        } },
        .{ .name = "utf8_lower", .want = &.{
            "",                         "abc",                      "abc",           "abc",                      "abc",                              "hello world",
            "hello world",              "hello world",              "\xc7\x86ungla", "\xc7\x86",                 "\xc7\x86",                         "stra\xc3\x9fe",
            "\xcf\x83\xce\xb1\xcf\x83", "\xcf\x83\xce\xb1\xcf\x82", "istanbul",      "\xc4\xb1i",                "123",                              "\xd9\xa1\xd9\xa2\xd9\xa3",
            "\xc2\xbd",                 "\xe2\x85\xbb",             "\xc2\xb2",      "a1",                       "\xc3\xa9",                         "a\xc3\xa9",
            "\xc3\xa9a",                " ",                        "\x09",          "\xc2\xa0",                 "\xe3\x80\x80",                     "\xef\xac\x81",
            "\xe2\x84\x8c",             "_a",                       "a_b",           "a1b",                      "ab cd",                            "1a",
            "\xea\x9e\xb5",             "\xe2\xb1\xa5",             "\xe2\xb1\xa5",  "\xc9\x90",                 "\xcd\x85",                         "a\xcc\x80b",
            "\x7f",                     "~",                        "a\xe2\x80\x8b", "\xc7\xb3\xc7\xb3\xc7\xb3", "\xf0\x90\x90\xa8\xf0\x90\x90\xa8", "\xe1\xb2\x8a",
            "\xf0\x96\xb9\xa0x",
        } },
        .{ .name = "utf8_swapcase", .want = &.{
            "",                         "ABC",                      "abc",           "aBC",                      "AbC",                              "hELLO wORLD",
            "HELLO WORLD",              "hello world",              "\xc7\x85UNGLA", "\xc7\x84",                 "\xc7\x85",                         "sTRA\xe1\xba\x9eE",
            "\xcf\x83\xce\xb1\xcf\x83", "\xce\xa3\xce\x91\xce\xa3", "iSTANBUL",      "Ii",                       "123",                              "\xd9\xa1\xd9\xa2\xd9\xa3",
            "\xc2\xbd",                 "\xe2\x85\xbb",             "\xc2\xb2",      "A1",                       "\xc3\x89",                         "A\xc3\x89",
            "\xc3\xa9A",                " ",                        "\x09",          "\xc2\xa0",                 "\xe3\x80\x80",                     "\xef\xac\x81",
            "\xe2\x84\x8c",             "_a",                       "A_b",           "a1B",                      "aB Cd",                            "1a",
            "\xea\x9e\xb4",             "\xe2\xb1\xa5",             "\xc8\xba",      "\xe2\xb1\xaf",             "\xce\x99",                         "a\xcc\x80B",
            "\x7f",                     "~",                        "A\xe2\x80\x8b", "\xc7\xb3\xc7\xb2\xc7\xb1", "\xf0\x90\x90\xa8\xf0\x90\x90\x80", "\xe1\xb2\x8a",
            "\xf0\x96\xb9\xa0X",
        } },
        .{ .name = "utf8_capitalize", .want = &.{
            "",                         "Abc",                      "Abc",             "Abc",                      "Abc",                              "Hello world",
            "Hello world",              "Hello world",              "\xc7\x84ungla",   "\xc7\x84",                 "\xc7\x84",                         "Stra\xc3\x9fe",
            "\xce\xa3\xce\xb1\xcf\x83", "\xce\xa3\xce\xb1\xcf\x82", "\xc4\xb0stanbul", "Ii",                       "123",                              "\xd9\xa1\xd9\xa2\xd9\xa3",
            "\xc2\xbd",                 "\xe2\x85\xab",             "\xc2\xb2",        "A1",                       "\xc3\x89",                         "A\xc3\xa9",
            "\xc3\x89a",                " ",                        "\x09",            "\xc2\xa0",                 "\xe3\x80\x80",                     "\xef\xac\x81",
            "\xe2\x84\x8c",             "_a",                       "A_b",             "A1b",                      "Ab cd",                            "1a",
            "\xea\x9e\xb4",             "\xc8\xba",                 "\xc8\xba",        "\xe2\xb1\xaf",             "\xce\x99",                         "A\xcc\x80b",
            "\x7f",                     "~",                        "A\xe2\x80\x8b",   "\xc7\xb1\xc7\xb3\xc7\xb3", "\xf0\x90\x90\x80\xf0\x90\x90\xa8", "\xe1\xb2\x89",
            "\xf0\x96\xb9\x80x",
        } },
        .{ .name = "utf8_title", .want = &.{
            "",                         "Abc",                      "Abc",             "Abc",                      "Abc",                              "Hello World",
            "Hello World",              "Hello World",              "\xc7\x84ungla",   "\xc7\x84",                 "\xc7\x84",                         "Stra\xc3\x9fe",
            "\xce\xa3\xce\xb1\xcf\x83", "\xce\xa3\xce\xb1\xcf\x82", "\xc4\xb0stanbul", "Ii",                       "123",                              "\xd9\xa1\xd9\xa2\xd9\xa3",
            "\xc2\xbd",                 "\xe2\x85\xab",             "\xc2\xb2",        "A1",                       "\xc3\x89",                         "A\xc3\xa9",
            "\xc3\x89a",                " ",                        "\x09",            "\xc2\xa0",                 "\xe3\x80\x80",                     "\xef\xac\x81",
            "\xe2\x84\x8c",             "_A",                       "A_B",             "A1B",                      "Ab Cd",                            "1A",
            "\xea\x9e\xb4",             "\xc8\xba",                 "\xc8\xba",        "\xe2\xb1\xaf",             "\xce\x99",                         "A\xcc\x80B",
            "\x7f",                     "~",                        "A\xe2\x80\x8b",   "\xc7\xb1\xc7\xb3\xc7\xb3", "\xf0\x90\x90\x80\xf0\x90\x90\xa8", "\xe1\xb2\x89",
            "\xf0\x96\xb9\x80x",
        } },
        .{ .name = "ascii_title", .want = &.{
            "",                         "Abc",                      "Abc",             "Abc",                      "Abc",                              "Hello World",
            "Hello World",              "Hello World",              "\xc7\x85Ungla",   "\xc7\x86",                 "\xc7\x85",                         "Stra\xc3\x9fE",
            "\xce\xa3\xce\x91\xce\xa3", "\xcf\x83\xce\xb1\xcf\x82", "\xc4\xb0Stanbul", "\xc4\xb1I",                "123",                              "\xd9\xa1\xd9\xa2\xd9\xa3",
            "\xc2\xbd",                 "\xe2\x85\xab",             "\xc2\xb2",        "A1",                       "\xc3\xa9",                         "A\xc3\xa9",
            "\xc3\x89A",                " ",                        "\x09",            "\xc2\xa0",                 "\xe3\x80\x80",                     "\xef\xac\x81",
            "\xe2\x84\x8c",             "_A",                       "A_B",             "A1B",                      "Ab Cd",                            "1A",
            "\xea\x9e\xb5",             "\xc8\xba",                 "\xe2\xb1\xa5",    "\xc9\x90",                 "\xcd\x85",                         "A\xcc\x80B",
            "\x7f",                     "~",                        "A\xe2\x80\x8b",   "\xc7\xb1\xc7\xb2\xc7\xb3", "\xf0\x90\x90\x80\xf0\x90\x90\xa8", "\xe1\xb2\x89",
            "\xf0\x96\xb9\x80X",
        } },
    };
    inline for (string_tags) |tag| {
        var opt: [rows.len]?[]const u8 = undefined;
        for (rows, &opt) |r, *o| o.* = r;
        var a = try common.makeStrings(tag, allocator, &opt);
        defer a.release();
        const args = [_]Datum{.{ .array = a }};
        for (predicates) |p| {
            var want: [rows.len]?bool = undefined;
            for (p.want, &want) |c, *w| w.* = c == '1';
            common.expectInvoke(bool, &ctx, .scalar, p.name, &args, &want) catch |err| {
                std.debug.print("{s} on {s}\n", .{ p.name, @tagName(tag) });
                return err;
            };
        }
        for (transforms) |t| {
            var want: [rows.len]?[]const u8 = undefined;
            for (t.want, &want) |w, *o| o.* = w;
            common.expectInvokeStrings(&ctx, t.name, &args, none, &want) catch |err| {
                std.debug.print("{s} on {s}\n", .{ t.name, @tagName(tag) });
                return err;
            };
        }
    }
    try std.testing.expectEqualStrings("17.0.0", unicode_tables.unicode_version);

    // Many rows without nulls are mapped as one stream; mappings that change the length
    // (U+00DF, U+0131, U+0130, U+023A, U+2C65, U+212A) move the row offsets. Sliced input,
    // against the per-row reference.
    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var prng = std.Random.DefaultPrng.init(0x57A3);
    const rng = prng.random();
    const many = try arena.alloc(?[]const u8, 500);
    for (many) |*r| {
        var buf: std.ArrayList(u8) = .empty;
        for (0..rng.uintAtMost(usize, 9)) |_| try buf.appendSlice(arena, pieces[rng.uintLessThan(usize, pieces.len)]);
        r.* = buf.items;
    }
    inline for (string_tags) |tag| {
        var full = try common.makeStrings(tag, allocator, many);
        defer full.release();
        var arr = try full.slice(7, 480);
        defer arr.release();
        for ([_][]const u8{ "utf8_upper", "utf8_lower" }) |name| {
            const expected = try arena.alloc(?[]const u8, 480);
            for (expected, many[7..][0..480]) |*e, r| e.* = try refTransform(arena, name, r.?, none);
            try common.expectInvokeStrings(&ctx, name, &.{.{ .array = arr }}, none, expected);
        }
    }
}

test "unicode tables: bucketed lookups equal a sequential scan for every code point" {
    const string = common.kernels.string;
    var run: usize = 0;
    var up: usize = 0;
    var low: usize = 0;
    var cp: u21 = 0;
    while (true) : (cp += 1) {
        while (run + 1 < unicode_tables.category_runs.len and unicode_tables.category_runs[run + 1] >> 5 <= cp) run += 1;
        try std.testing.expectEqual(@as(unicode_tables.Category, @enumFromInt(@as(u5, @truncate(unicode_tables.category_runs[run])))), string.category(cp));
        inline for (.{ .{ &unicode_tables.upper_ranges, &up, string.toUpper }, .{ &unicode_tables.lower_ranges, &low, string.toLower } }) |t| {
            const ranges = t[0];
            const k = t[1];
            while (k.* + 1 < ranges.len and ranges[k.*].last < cp) k.* += 1;
            const r = ranges[k.*];
            const stride: u21 = if (r.stride2 == 1) 2 else 1;
            const want: u21 = if (cp >= r.first and cp <= r.last and (cp - r.first) % stride == 0) @intCast(@as(i32, cp) + r.delta) else cp;
            try std.testing.expectEqual(want, t[2](cp));
        }
        if (cp == 0x10FFFF) break;
    }
    // Spot checks of the data (UnicodeData.txt 17.0.0).
    try std.testing.expectEqual(unicode_tables.Category.Lt, string.category(0x01C5));
    try std.testing.expectEqual(unicode_tables.Category.Cn, string.category(0x0378));
    try std.testing.expectEqual(unicode_tables.Category.Lo, string.category(0x20000));
    try std.testing.expectEqual(unicode_tables.Category.Co, string.category(0x10FFFD));
    try std.testing.expectEqual(@as(u21, 0x1E9E), string.toUpper(0xDF));
    try std.testing.expectEqual(@as(u21, 0x1C8A), string.toLower(0x1C89));
    try std.testing.expectEqual(@as(u21, 0x10400), string.toUpper(0x10428));
}

test "string chunked inputs with misaligned chunks give a chunked result equal to the array result" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    var prng = std.Random.DefaultPrng.init(0x57A2);
    inline for (string_tags) |tag| {
        const rows = try randomRows(arena_state.allocator(), prng.random(), 300);
        var arr = try common.makeStrings(tag, allocator, rows);
        defer arr.release();
        var a = try common.chunkedAt(allocator, arr, &.{ 7, 150 });
        defer a.release();
        var b = try common.chunkedAt(allocator, arr, &.{ 64, 299 });
        defer b.release();
        inline for (.{ "utf8_upper", "ascii_trim_whitespace", "utf8_reverse" }) |name| {
            var whole = try ctx.invokeScalar(name, &.{.{ .array = arr }}, none);
            defer whole.release();
            const expected = try common.collectStrings(allocator, whole);
            defer allocator.free(expected);
            var out = try ctx.invokeScalar(name, &.{.{ .chunked = a }}, none);
            defer out.release();
            try std.testing.expectEqual(@as(usize, 3), out.chunked.numChunks());
            const got = try common.collectStrings(allocator, out);
            defer allocator.free(got);
            try common.expectStrings(expected, got);
        }
        // Join over chunked inputs: the boundaries are the union (7, 64, 150, 299).
        var joined = try ctx.invokeScalar("binary_join_element_wise", &.{ .{ .chunked = a }, .{ .chunked = b }, .{ .chunked = b } }, none);
        defer joined.release();
        try std.testing.expectEqual(@as(usize, 5), joined.chunked.numChunks());
        var whole = try ctx.invokeScalar("binary_join_element_wise", &.{ .{ .array = arr }, .{ .array = arr }, .{ .array = arr } }, none);
        defer whole.release();
        const expected = try common.collectStrings(allocator, whole);
        defer allocator.free(expected);
        const got = try common.collectStrings(allocator, joined);
        defer allocator.free(got);
        try common.expectStrings(expected, got);
        var lens = try ctx.invokeScalar("utf8_length", &.{.{ .chunked = a }}, none);
        defer lens.release();
        try std.testing.expectEqual(@as(usize, 3), lens.chunked.numChunks());
    }
}

test "string scalars, empty and all-null inputs" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    const st = DataType{ .string = {} };

    var up = try ctx.invokeScalar("utf8_upper", &.{.{ .scalar = Scalar.init(st, .{ .string = "abc" }) }}, none);
    defer up.release();
    try std.testing.expectEqualStrings("ABC", up.scalar.value.string);
    // The scalar owns its bytes through its payload, so it can feed the next call.
    var rev = try ctx.invokeScalar("utf8_reverse", &.{up}, none);
    defer rev.release();
    try std.testing.expectEqualStrings("CBA", rev.scalar.value.string);
    var n = try ctx.invokeScalar("utf8_length", &.{.{ .scalar = Scalar.init(.{ .large_string = {} }, .{ .string = "€€" }) }}, none);
    defer n.release();
    try std.testing.expectEqual(@as(i64, 2), n.scalar.value.i64);
    var sw = try ctx.invokeScalar("starts_with", &.{.{ .scalar = Scalar.init(st, .{ .string = "abc" }) }}, pattern("ab", false));
    defer sw.release();
    try std.testing.expect(sw.scalar.value.bool);
    var nul = try ctx.invokeScalar("ascii_lower", &.{.{ .scalar = Scalar.init(st, .null) }}, none);
    defer nul.release();
    try std.testing.expect(nul.scalar.isNull() and nul.scalar.data_type == .string);
    var j = try ctx.invokeScalar("binary_join_element_wise", &.{ .{ .scalar = Scalar.init(st, .{ .string = "x" }) }, .{ .scalar = Scalar.init(st, .null) }, .{ .scalar = Scalar.init(st, .{ .string = "+" }) } }, .{ .join = .{ .null_handling = .skip } });
    defer j.release();
    try std.testing.expectEqualStrings("x", j.scalar.value.string);

    inline for (string_tags) |tag| {
        var empty = try common.makeStrings(tag, allocator, &.{});
        defer empty.release();
        try common.expectInvokeStrings(&ctx, "utf8_upper", &.{.{ .array = empty }}, none, &.{});
        try common.expectInvokeStrings(&ctx, "binary_join_element_wise", &.{ .{ .array = empty }, .{ .array = empty } }, none, &.{});
        var nulls = try common.makeStrings(tag, allocator, &.{ null, null, null });
        defer nulls.release();
        try common.expectInvokeStrings(&ctx, "utf8_reverse", &.{.{ .array = nulls }}, none, &.{ null, null, null });
        try common.expectInvokeStrings(&ctx, "ascii_upper", &.{.{ .array = nulls }}, none, &.{ null, null, null });
        try common.expectInvoke(bool, &ctx, .scalar, "utf8_is_alpha", &.{.{ .array = nulls }}, &.{ null, null, null });
    }
}

test "string errors: invalid UTF-8 rows, options, types, arity and malformed input" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);

    // A string array that never passed validateFull: the decoding kernels reject the bad row,
    // the byte kernels pass it through, and a null row with bad bytes is not looked at.
    var a = try common.makeStrings(.binary, allocator, &.{ "ok", "\xff\xfe", "fine" });
    defer a.release();
    var as_string = try ArrayRef.fromBorrowed(allocator, .{ .data_type = .{ .string = {} }, .length = 3, .null_count = 0, .buffers = a.data().buffers });
    defer as_string.release();
    const args = [_]Datum{.{ .array = as_string }};
    inline for (.{ "utf8_upper", "utf8_lower", "utf8_swapcase", "utf8_capitalize", "utf8_title", "utf8_trim_whitespace", "utf8_reverse", "utf8_is_alpha", "utf8_is_space", "utf8_is_lower", "utf8_is_title", "utf8_is_printable" }) |name| {
        try std.testing.expectError(error.InvalidInput, ctx.invokeScalar(name, &args, none));
    }
    try std.testing.expectError(error.InvalidInput, ctx.invokeScalar("utf8_slice_codeunits", &args, slice(0, 1, 1)));
    var lens = try ctx.invokeScalar("binary_length", &args, none);
    lens.release();
    const mask = [_]u8{0b101};
    const masked_bufs = [_]common.SharedBuffer{ .fromSlice(&mask), a.data().buffers[1], a.data().buffers[2] };
    var masked = try ArrayRef.fromBorrowed(allocator, .{ .data_type = .{ .string = {} }, .length = 3, .null_count = 1, .buffers = &masked_bufs });
    defer masked.release();
    try common.expectInvokeStrings(&ctx, "utf8_upper", &.{.{ .array = masked }}, none, &.{ "OK", null, "FINE" });
    // A row boundary inside a code point: the window is UTF-8, its rows are not.
    const split_offsets = [_]i32{ 0, 2, 3 };
    const split_bufs = [_]common.SharedBuffer{ .empty, .fromSlice(std.mem.sliceAsBytes(&split_offsets)), .fromSlice("a\xc3\xa9") };
    var split = try ArrayRef.fromBorrowed(allocator, .{ .data_type = .{ .string = {} }, .length = 2, .null_count = 0, .buffers = &split_bufs });
    defer split.release();
    try std.testing.expectError(error.InvalidInput, ctx.invokeScalar("utf8_lower", &.{.{ .array = split }}, none));

    var s = try common.makeStrings(.string, allocator, &.{"abc"});
    defer s.release();
    const sargs = [_]Datum{.{ .array = s }};
    try std.testing.expectError(error.InvalidOptions, ctx.invokeScalar("utf8_slice_codeunits", &sargs, slice(0, 1, 0)));
    try std.testing.expectError(error.InvalidOptions, ctx.invokeScalar("utf8_slice_codeunits", &sargs, none));
    try std.testing.expectError(error.InvalidOptions, ctx.invokeScalar("starts_with", &sargs, none));
    try std.testing.expectError(error.InvalidOptions, ctx.invokeScalar("ascii_trim", &sargs, .{ .trim = .{ .characters = "\xc3" } }));
    try std.testing.expectError(error.InvalidOptions, ctx.invokeScalar("utf8_trim", &sargs, .{ .trim = .{ .characters = "\xc3" } }));
    try std.testing.expectError(error.InvalidOptions, ctx.invokeScalar("utf8_trim", &sargs, none));
    try std.testing.expectError(error.InvalidInput, ctx.invokeScalar("utf8_trim", &args, .{ .trim = .{ .characters = "x" } }));
    var non_ascii = try common.makeStrings(.string, allocator, &.{ "ab", "é" });
    defer non_ascii.release();
    try std.testing.expectError(error.InvalidInput, ctx.invokeScalar("ascii_reverse", &.{.{ .array = non_ascii }}, none));
    try std.testing.expectError(error.InvalidOptions, ctx.invokeScalar("utf8_upper", &sargs, pattern("a", false)));
    try std.testing.expectError(error.InvalidOptions, ctx.invokeScalar("binary_join_element_wise", &.{ sargs[0], sargs[0] }, .{ .join = .{ .null_handling = .replace, .null_replacement = "\xff" } }));
    var ls = try common.makeStrings(.large_string, allocator, &.{"abc"});
    defer ls.release();
    try std.testing.expectError(error.NoMatchingKernel, ctx.invokeScalar("binary_join_element_wise", &.{ sargs[0], .{ .array = ls } }, none));
    var b = try common.makeStrings(.binary, allocator, &.{"abc"});
    defer b.release();
    try std.testing.expectError(error.NoMatchingKernel, ctx.invokeScalar("utf8_upper", &.{.{ .array = b }}, none));
    var i = try common.makeArray(i32, allocator, &.{1});
    defer i.release();
    try std.testing.expectError(error.NoMatchingKernel, ctx.invokeScalar("utf8_length", &.{.{ .array = i }}, none));
    try std.testing.expectError(error.InvalidArity, ctx.invokeScalar("utf8_upper", &.{ sargs[0], sargs[0] }, none));
    // Offsets past the data buffer fail validateLayout instead of being read.
    const offsets = [_]i32{ 0, 100 };
    const broken_bufs = [_]common.SharedBuffer{ .empty, .fromSlice(std.mem.sliceAsBytes(&offsets)), .fromSlice("abc") };
    var broken = try ArrayRef.fromBorrowed(allocator, .{ .data_type = .{ .string = {} }, .length = 1, .null_count = 0, .buffers = &broken_bufs });
    defer broken.release();
    try std.testing.expectError(error.InvalidInput, ctx.invokeScalar("utf8_upper", &.{.{ .array = broken }}, none));
}
