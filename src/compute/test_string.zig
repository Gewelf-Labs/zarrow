const std = @import("std");
const common = @import("test_common.zig");

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
        // utf8_upper / utf8_lower map ASCII letters only (no Unicode tables in std).
        try common.expectInvokeStrings(&ctx, "utf8_upper", &args, none, &.{ "HELLO", null, "", "  MIXED 9 \t", "HéLLO WöRLD😀" });
        try common.expectInvokeStrings(&ctx, "utf8_lower", &args, none, &.{ "hello", null, "", "  mixed 9 \t", "héllo wörld😀" });
        try common.expectInvokeStrings(&ctx, "ascii_trim_whitespace", &args, none, &.{ "Hello", null, "", "mIxEd 9", "héllo wörld😀" });
        try common.expectInvokeStrings(&ctx, "ascii_ltrim_whitespace", &args, none, &.{ "Hello", null, "", "mIxEd 9 \t", "héllo wörld😀" });
        try common.expectInvokeStrings(&ctx, "ascii_rtrim_whitespace", &args, none, &.{ "Hello", null, "", "  mIxEd 9", "héllo wörld😀" });
        try common.expectInvokeStrings(&ctx, "ascii_trim", &args, .{ .trim = .{ .characters = "Ho \t" } }, &.{ "ell", null, "", "mIxEd 9", "héllo wörld😀" });
        try common.expectInvokeStrings(&ctx, "ascii_ltrim", &args, .{ .trim = .{ .characters = "H " } }, &.{ "ello", null, "", "mIxEd 9 \t", "héllo wörld😀" });
        try common.expectInvokeStrings(&ctx, "ascii_rtrim", &args, .{ .trim = .{ .characters = "o\t " } }, &.{ "Hell", null, "", "  mIxEd 9", "héllo wörld😀" });
        try common.expectInvokeStrings(&ctx, "utf8_reverse", &args, none, &.{ "olleH", null, "", "\t 9 dExIm  ", "😀dlröw olléh" });
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
        try common.expectInvoke(bool, &ctx, .scalar, "utf8_is_alpha", &pargs, &.{ true, false, false, false, false, false, true });
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

const pieces = [_][]const u8{ "a", "B", "z", "Q", "0", "7", " ", "\t", "é", "€", "😀", "\u{3000}", "\u{00a0}", "ab", "Zy" };

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
    for ([_]u21{ 9, 10, 11, 12, 13, 32, 0x85, 0xA0, 0x1680, 0x2000, 0x2001, 0x2002, 0x2003, 0x2004, 0x2005, 0x2006, 0x2007, 0x2008, 0x2009, 0x200A, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000 }) |w| {
        if (c == w) return true;
    }
    return false;
}

/// Reference of each string to string function on one row.
fn refTransform(arena: std.mem.Allocator, name: []const u8, s: []const u8, o: Options) ![]const u8 {
    const eq = std.mem.eql;
    if (eq(u8, name, "ascii_upper") or eq(u8, name, "utf8_upper")) return std.ascii.allocUpperString(arena, s);
    if (eq(u8, name, "ascii_lower") or eq(u8, name, "utf8_lower")) return std.ascii.allocLowerString(arena, s);
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
        const ok = if (eq(u8, name, "ascii_is_space"))
            ascii and std.ascii.isWhitespace(cp[0])
        else if (eq(u8, name, "utf8_is_space"))
            isWhiteSpace(cp)
        else if (std.mem.endsWith(u8, name, "is_alpha"))
            ascii and std.ascii.isAlphabetic(cp[0])
        else
            ascii and std.ascii.isDigit(cp[0]);
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
    inline for (.{ "utf8_upper", "utf8_lower", "utf8_trim_whitespace", "utf8_reverse", "utf8_is_alpha", "utf8_is_space" }) |name| {
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

    var s = try common.makeStrings(.string, allocator, &.{"abc"});
    defer s.release();
    const sargs = [_]Datum{.{ .array = s }};
    try std.testing.expectError(error.InvalidOptions, ctx.invokeScalar("utf8_slice_codeunits", &sargs, slice(0, 1, 0)));
    try std.testing.expectError(error.InvalidOptions, ctx.invokeScalar("utf8_slice_codeunits", &sargs, none));
    try std.testing.expectError(error.InvalidOptions, ctx.invokeScalar("starts_with", &sargs, none));
    try std.testing.expectError(error.InvalidOptions, ctx.invokeScalar("ascii_trim", &sargs, .{ .trim = .{ .characters = "\xc3" } }));
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
