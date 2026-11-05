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
const binary_tags = [_]TypeTag{ .binary, .large_binary, .binary_view };

fn replace(p: []const u8, r: []const u8, max: ?usize) Options {
    return .{ .replace_substring = .{ .pattern = p, .replacement = r, .max_replacements = max } };
}

fn splitOn(p: []const u8, max: ?usize, reverse: bool) Options {
    return .{ .split_pattern = .{ .pattern = p, .max_splits = max, .reverse = reverse } };
}

fn pad(width: usize, padding: []const u8) Options {
    return .{ .pad = .{ .width = width, .padding = padding } };
}

fn match(p: []const u8, ignore_case: bool) Options {
    return .{ .match_substring = .{ .pattern = p, .ignore_case = ignore_case } };
}

/// Rows of a list<string-like> datum (array or chunked): null or the parts of each row. The
/// output must pass validateFull. Caller frees with `freeLists`.
fn collectLists(allocator: std.mem.Allocator, datum: Datum) ![]?[]?[]const u8 {
    const parts: []const ArrayRef = switch (datum) {
        .array => |*a| a[0..1],
        .chunked => |c| c.chunks(),
        .scalar => |*s| (&s.payload.?)[0..1],
    };
    var out: std.ArrayList(?[]?[]const u8) = .empty;
    errdefer freeLists(allocator, out.items);
    for (parts) |p| {
        const data = p.data();
        try data.validateFull();
        const offsets = (try data.buffers[1].typedSlice(i32))[data.offset..][0 .. data.length + 1];
        const child = data.children[0].data();
        for (0..data.length) |i| {
            if (data.isNull(i)) {
                try out.append(allocator, null);
                continue;
            }
            const row = try allocator.alloc(?[]const u8, @intCast(offsets[i + 1] - offsets[i]));
            for (row, @intCast(offsets[i])..) |*r, k| r.* = try common.bytesAt(child, k);
            try out.append(allocator, row);
        }
    }
    return out.toOwnedSlice(allocator);
}

fn freeLists(allocator: std.mem.Allocator, rows: []?[]?[]const u8) void {
    for (rows) |r| if (r) |x| allocator.free(x);
    allocator.free(rows);
}

fn expectLists(expected: []const ?[]const []const u8, actual: []const ?[]?[]const u8) !void {
    try std.testing.expectEqual(expected.len, actual.len);
    for (expected, actual, 0..) |e, a, i| {
        errdefer std.debug.print("row {d}\n", .{i});
        if (e == null or a == null) {
            try std.testing.expect(e == null and a == null);
            continue;
        }
        try std.testing.expectEqual(e.?.len, a.?.len);
        for (e.?, a.?) |x, y| try std.testing.expectEqualStrings(x, y.?);
    }
}

fn expectSplit(ctx: *ExecContext, name: []const u8, args: []const Datum, options: Options, expected: []const ?[]const []const u8) !void {
    const allocator = std.testing.allocator;
    var out = try ctx.invokeScalar(name, args, options);
    defer out.release();
    const rows = try collectLists(allocator, out);
    defer freeLists(allocator, rows);
    try expectLists(expected, rows);
}

test "string transform hand-written values: replace, split, pad, count, find, slice, repeat, reverse, case on every layout" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);

    // Expected values from pyarrow 26.
    inline for (string_tags) |tag| {
        var r = try common.makeStrings(tag, allocator, &.{ "aaa", "abab", null, "abc", "" });
        defer r.release();
        const rargs = [_]Datum{.{ .array = r }};
        try common.expectInvokeStrings(&ctx, "replace_substring", &rargs, replace("a", "xy", null), &.{ "xyxyxy", "xybxyb", null, "xybc", "" });
        try common.expectInvokeStrings(&ctx, "replace_substring", &rargs, replace("a", "x", 2), &.{ "xxa", "xbxb", null, "xbc", "" });
        try common.expectInvokeStrings(&ctx, "replace_substring", &rargs, replace("", "-", 2), &.{ "--aaa", "--abab", null, "--abc", "" });
        try common.expectInvokeStrings(&ctx, "replace_substring", &rargs, replace("ab", "", null), &.{ "aaa", "", null, "c", "" });
        // A match is inside one row: "a" | "b" across a row end is not "ab".
        var cross = try common.makeStrings(tag, allocator, &.{ "xa", "by", "ab", "a", "bab" });
        defer cross.release();
        try common.expectInvokeStrings(&ctx, "replace_substring", &.{.{ .array = cross }}, replace("ab", "R", null), &.{ "xa", "by", "R", "a", "bR" });
        try common.expectInvokeStrings(&ctx, "replace_substring", &.{.{ .array = cross }}, replace("ab", "R", 0), &.{ "xa", "by", "ab", "a", "bab" });

        var s = try common.makeStrings(tag, allocator, &.{ " a  b ", "", "  ", "a,,b", "abc", null, "héllo wörld" });
        defer s.release();
        const sargs = [_]Datum{.{ .array = s }};
        try expectSplit(&ctx, "split_pattern", &sargs, splitOn(",", null, false), &.{ &.{" a  b "}, &.{""}, &.{"  "}, &.{ "a", "", "b" }, &.{"abc"}, null, &.{"héllo wörld"} });
        try expectSplit(&ctx, "ascii_split_whitespace", &sargs, none, &.{ &.{ "", "a", "b", "" }, &.{""}, &.{ "", "" }, &.{"a,,b"}, &.{"abc"}, null, &.{ "héllo", "wörld" } });
        try expectSplit(&ctx, "ascii_split_whitespace", &sargs, .{ .split = .{ .max_splits = 1 } }, &.{ &.{ "", "a  b " }, &.{""}, &.{ "", "" }, &.{"a,,b"}, &.{"abc"}, null, &.{ "héllo", "wörld" } });
        try expectSplit(&ctx, "ascii_split_whitespace", &sargs, .{ .split = .{ .max_splits = 1, .reverse = true } }, &.{ &.{ " a  b", "" }, &.{""}, &.{ "", "" }, &.{"a,,b"}, &.{"abc"}, null, &.{ "héllo", "wörld" } });
        var abc = try common.makeStrings(tag, allocator, &.{ "a,b,c", "aaa" });
        defer abc.release();
        try expectSplit(&ctx, "split_pattern", &.{.{ .array = abc }}, splitOn(",", 1, true), &.{ &.{ "a,b", "c" }, &.{"aaa"} });
        try expectSplit(&ctx, "split_pattern", &.{.{ .array = abc }}, splitOn(",", 1, false), &.{ &.{ "a", "b,c" }, &.{"aaa"} });
        // A self-overlapping separator is searched from the right when reversed.
        try expectSplit(&ctx, "split_pattern", &.{.{ .array = abc }}, splitOn("aa", null, true), &.{ &.{"a,b,c"}, &.{ "a", "" } });
        try expectSplit(&ctx, "split_pattern", &.{.{ .array = abc }}, splitOn("aa", null, false), &.{ &.{"a,b,c"}, &.{ "", "a" } });
        var u = try common.makeStrings(tag, allocator, &.{ " a\u{3000}b ", "", "x\u{00a0}\u{2003}y", "p\x1fq\u{200b}r" });
        defer u.release();
        // U+001F is space for Arrow (utf8proc), U+200B is not (pyarrow 26).
        try expectSplit(&ctx, "utf8_split_whitespace", &.{.{ .array = u }}, none, &.{ &.{ "", "a", "b", "" }, &.{""}, &.{ "x", "y" }, &.{ "p", "q\u{200b}r" } });

        var c = try common.makeStrings(tag, allocator, &.{ "aaaa", "abc", "", null, "héllo", "AaAa" });
        defer c.release();
        const cargs = [_]Datum{.{ .array = c }};
        const L = if (tag == .large_string) i64 else i32;
        try common.expectInvokeWith(L, &ctx, "count_substring", &cargs, match("aa", false), &.{ 2, 0, 0, null, 0, 0 });
        try common.expectInvokeWith(L, &ctx, "count_substring", &cargs, match("aa", true), &.{ 2, 0, 0, null, 0, 2 });
        try common.expectInvokeWith(L, &ctx, "count_substring", &cargs, match("", false), &.{ 5, 4, 1, null, 7, 5 });
        try common.expectInvokeWith(L, &ctx, "find_substring", &cargs, match("l", false), &.{ -1, -1, -1, null, 3, -1 });
        try common.expectInvokeWith(L, &ctx, "find_substring", &cargs, match("BC", true), &.{ -1, 1, -1, null, -1, -1 });
        try common.expectInvokeWith(L, &ctx, "find_substring", &cargs, match("", false), &.{ 0, 0, 0, null, 0, 0 });

        var p = try common.makeStrings(tag, allocator, &.{ "ab", "abcdef", null, "a", "héé" });
        defer p.release();
        const pargs = [_]Datum{.{ .array = p }};
        try common.expectInvokeStrings(&ctx, "ascii_lpad", &pargs, pad(5, "*"), &.{ "***ab", "abcdef", null, "****a", "héé" });
        try common.expectInvokeStrings(&ctx, "ascii_rpad", &pargs, pad(6, "x"), &.{ "abxxxx", "abcdef", null, "axxxxx", "hééx" });
        try common.expectInvokeStrings(&ctx, "ascii_center", &pargs, pad(5, "*"), &.{ "*ab**", "abcdef", null, "**a**", "héé" });
        try common.expectInvokeStrings(&ctx, "ascii_center", &pargs, .{ .pad = .{ .width = 5, .padding = "*", .lean_left_on_odd_padding = false } }, &.{ "**ab*", "abcdef", null, "**a**", "héé" });
        try common.expectInvokeStrings(&ctx, "utf8_rpad", &pargs, pad(5, "x"), &.{ "abxxx", "abcdef", null, "axxxx", "hééxx" });
        try common.expectInvokeStrings(&ctx, "utf8_lpad", &pargs, pad(4, "€"), &.{ "€€ab", "abcdef", null, "€€€a", "€héé" });
        var e = try common.makeStrings(tag, allocator, &.{"é"});
        defer e.release();
        try common.expectInvokeStrings(&ctx, "utf8_center", &.{.{ .array = e }}, pad(4, "€"), &.{"€é€€"});

        // Case: ASCII letters only (Arrow's utf8_capitalize also maps é to É).
        var k = try common.makeStrings(tag, allocator, &.{ "hELLO wORLD", "éa", "", null, "HeLLo 9" });
        defer k.release();
        const kargs = [_]Datum{.{ .array = k }};
        try common.expectInvokeStrings(&ctx, "utf8_capitalize", &kargs, none, &.{ "Hello world", "éa", "", null, "Hello 9" });
        try common.expectInvokeStrings(&ctx, "ascii_capitalize", &kargs, none, &.{ "Hello world", "éa", "", null, "Hello 9" });
        try common.expectInvokeStrings(&ctx, "utf8_swapcase", &kargs, none, &.{ "Hello World", "éA", "", null, "hEllO 9" });
        try common.expectInvokeStrings(&ctx, "ascii_swapcase", &kargs, none, &.{ "Hello World", "éA", "", null, "hEllO 9" });

        var n = try common.makeArray(i64, allocator, &.{ 2, 0, 1, null, 3 });
        defer n.release();
        try common.expectInvokeStrings(&ctx, "binary_repeat", &.{ .{ .array = p }, .{ .array = n } }, none, &.{ "abab", "", null, null, "hééhééhéé" });
        try common.expectInvokeStrings(&ctx, "binary_repeat", &.{ .{ .array = p }, .{ .scalar = common.scalarOf(i64, 2) } }, none, &.{ "abab", "abcdefabcdef", null, "aa", "hééhéé" });
    }

    inline for (binary_tags) |tag| {
        var b = try common.makeStrings(tag, allocator, &.{ "abcdef", null, "", "\x00\xff\x01" });
        defer b.release();
        const bargs = [_]Datum{.{ .array = b }};
        try common.expectInvokeStrings(&ctx, "binary_slice", &bargs, .{ .slice = .{ .start = 1, .stop = 4 } }, &.{ "bcd", null, "", "\xff\x01" });
        try common.expectInvokeStrings(&ctx, "binary_slice", &bargs, .{ .slice = .{ .start = -1, .stop = -10, .step = -2 } }, &.{ "fdb", null, "", "\x01\x00" });
        try common.expectInvokeStrings(&ctx, "binary_reverse", &bargs, none, &.{ "fedcba", null, "", "\x01\xff\x00" });
        try common.expectInvokeStrings(&ctx, "replace_substring", &bargs, replace("\xff", "XY", null), &.{ "abcdef", null, "", "\x00XY\x01" });
        try expectSplit(&ctx, "split_pattern", &bargs, splitOn("\xff", null, false), &.{ &.{"abcdef"}, null, &.{""}, &.{ "\x00", "\x01" } });
        const L = if (tag == .large_binary) i64 else i32;
        try common.expectInvokeWith(L, &ctx, "count_substring", &bargs, match("c", false), &.{ 1, null, 0, 0 });
    }
}

// ------------------------------------------------------------------------------ oracle

const pieces = [_][]const u8{ "a", "b", "ab", " ", "  ", "\t", ",", "é", "😀", "A" };

fn randomRows(arena: std.mem.Allocator, rng: std.Random, len: usize) ![]?[]const u8 {
    const rows = try arena.alloc(?[]const u8, len);
    for (rows) |*r| {
        if (rng.uintLessThan(u8, 100) < 15) {
            r.* = null;
            continue;
        }
        var buf: std.ArrayList(u8) = .empty;
        for (0..rng.uintAtMost(usize, 8)) |_| try buf.appendSlice(arena, pieces[rng.uintLessThan(usize, pieces.len)]);
        r.* = buf.items;
    }
    return rows;
}

/// Reference split of `s` on runs of ASCII whitespace (empty leading / trailing parts kept).
fn refSplitWhitespace(arena: std.mem.Allocator, s: []const u8) ![]const []const u8 {
    var parts: std.ArrayList([]const u8) = .empty;
    var start: usize = 0;
    var i: usize = 0;
    while (i < s.len) {
        if (!std.ascii.isWhitespace(s[i])) {
            i += 1;
            continue;
        }
        try parts.append(arena, s[start..i]);
        while (i < s.len and std.ascii.isWhitespace(s[i])) i += 1;
        start = i;
    }
    try parts.append(arena, s[start..]);
    return parts.items;
}

fn refSplit(arena: std.mem.Allocator, s: []const u8, sep: []const u8) ![]const []const u8 {
    var parts: std.ArrayList([]const u8) = .empty;
    var it = std.mem.splitSequence(u8, s, sep);
    while (it.next()) |p| try parts.append(arena, p);
    return parts.items;
}

fn refPad(arena: std.mem.Allocator, s: []const u8, width: usize, c: []const u8, left: bool, codepoints: bool) ![]const u8 {
    const n = if (codepoints) try std.unicode.utf8CountCodepoints(s) else s.len;
    var out: std.ArrayList(u8) = .empty;
    if (!left) try out.appendSlice(arena, s);
    for (n..@max(n, width)) |_| try out.appendSlice(arena, c);
    if (left) try out.appendSlice(arena, s);
    return out.items;
}

test "string transform oracle matches a per-row loop on random sliced inputs with nulls for every layout" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var prng = std.Random.DefaultPrng.init(0x57A3);
    const rng = prng.random();

    inline for (string_tags) |tag| {
        const rows = try randomRows(arena, rng, 420);
        var full = try common.makeStrings(tag, allocator, rows);
        defer full.release();
        var arr = try full.slice(17, 400);
        defer arr.release();
        const view = rows[17..][0..400];
        const args = [_]Datum{.{ .array = arr }};

        const replaced = try arena.alloc(?[]const u8, 400);
        for (replaced, view) |*e, r| e.* = if (r) |s| try std.mem.replaceOwned(u8, arena, s, "ab", "<é>") else null;
        try common.expectInvokeStrings(&ctx, "replace_substring", &args, replace("ab", "<é>", null), replaced);
        for (replaced, view) |*e, r| e.* = if (r) |s| try std.mem.replaceOwned(u8, arena, s, "a", "") else null;
        try common.expectInvokeStrings(&ctx, "replace_substring", &args, replace("a", "", null), replaced);

        const lpadded = try arena.alloc(?[]const u8, 400);
        for (lpadded, view) |*e, r| e.* = if (r) |s| try refPad(arena, s, 12, ".", true, false) else null;
        try common.expectInvokeStrings(&ctx, "ascii_lpad", &args, pad(12, "."), lpadded);
        const rpadded = try arena.alloc(?[]const u8, 400);
        for (rpadded, view) |*e, r| e.* = if (r) |s| try refPad(arena, s, 9, "😀", false, true) else null;
        try common.expectInvokeStrings(&ctx, "utf8_rpad", &args, pad(9, "😀"), rpadded);

        const by_comma = try arena.alloc(?[]const []const u8, 400);
        for (by_comma, view) |*e, r| e.* = if (r) |s| try refSplit(arena, s, ",") else null;
        try expectSplit(&ctx, "split_pattern", &args, splitOn(",", null, false), by_comma);
        const by_space = try arena.alloc(?[]const []const u8, 400);
        for (by_space, view) |*e, r| e.* = if (r) |s| try refSplitWhitespace(arena, s) else null;
        try expectSplit(&ctx, "ascii_split_whitespace", &args, none, by_space);
        // Only ASCII whitespace in the rows: the utf8 split agrees.
        try expectSplit(&ctx, "utf8_split_whitespace", &args, none, by_space);

        const L = if (tag == .large_string) i64 else i32;
        const counts = try arena.alloc(?L, 400);
        for (counts, view) |*e, r| e.* = if (r) |s| @intCast(std.mem.count(u8, s, "ab")) else null;
        try common.expectInvokeWith(L, &ctx, "count_substring", &args, match("ab", false), counts);
        const finds = try arena.alloc(?L, 400);
        for (finds, view) |*e, r| e.* = if (r) |s| if (std.mem.indexOf(u8, s, "é")) |p| @intCast(p) else -1 else null;
        try common.expectInvokeWith(L, &ctx, "find_substring", &args, match("é", false), finds);

        const swapped = try arena.alloc(?[]const u8, 400);
        for (swapped, view) |*e, r| e.* = if (r) |s| blk: {
            const o = try arena.dupe(u8, s);
            for (o) |*ch| if (std.ascii.isAlphabetic(ch.*)) {
                ch.* ^= 32;
            };
            break :blk o;
        } else null;
        try common.expectInvokeStrings(&ctx, "utf8_swapcase", &args, none, swapped);
    }

    inline for (binary_tags) |tag| {
        const rows = try randomRows(arena, rng, 300);
        var full = try common.makeStrings(tag, allocator, rows);
        defer full.release();
        var arr = try full.slice(5, 290);
        defer arr.release();
        const view = rows[5..][0..290];
        const reversed = try arena.alloc(?[]const u8, 290);
        for (reversed, view) |*e, r| e.* = if (r) |s| blk: {
            const o = try arena.dupe(u8, s);
            std.mem.reverse(u8, o);
            break :blk o;
        } else null;
        try common.expectInvokeStrings(&ctx, "binary_reverse", &.{.{ .array = arr }}, none, reversed);
        const sliced = try arena.alloc(?[]const u8, 290);
        for (sliced, view) |*e, r| e.* = if (r) |s| s[@min(2, s.len)..@min(7, s.len)] else null;
        try common.expectInvokeStrings(&ctx, "binary_slice", &.{.{ .array = arr }}, .{ .slice = .{ .start = 2, .stop = 7 } }, sliced);
    }
}

test "string transform chunked inputs with misaligned chunks give a chunked result equal to the array result" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    var prng = std.Random.DefaultPrng.init(0x57A4);
    inline for (string_tags) |tag| {
        const rows = try randomRows(arena_state.allocator(), prng.random(), 300);
        var arr = try common.makeStrings(tag, allocator, rows);
        defer arr.release();
        var chunked = try common.chunkedAt(allocator, arr, &.{ 9, 150, 299 });
        defer chunked.release();
        const T = struct { name: []const u8, o: Options };
        for ([_]T{
            .{ .name = "replace_substring", .o = replace("a", "xyz", 1) },
            .{ .name = "utf8_center", .o = pad(7, "-") },
            .{ .name = "utf8_capitalize", .o = none },
        }) |t| {
            var whole = try ctx.invokeScalar(t.name, &.{.{ .array = arr }}, t.o);
            defer whole.release();
            const expected = try common.collectStrings(allocator, whole);
            defer allocator.free(expected);
            var out = try ctx.invokeScalar(t.name, &.{.{ .chunked = chunked }}, t.o);
            defer out.release();
            try std.testing.expectEqual(@as(usize, 4), out.chunked.numChunks());
            const got = try common.collectStrings(allocator, out);
            defer allocator.free(got);
            try common.expectStrings(expected, got);
        }
        var whole = try ctx.invokeScalar("split_pattern", &.{.{ .array = arr }}, splitOn(" ", null, false));
        defer whole.release();
        const expected = try collectLists(allocator, whole);
        defer freeLists(allocator, expected);
        var out = try ctx.invokeScalar("split_pattern", &.{.{ .chunked = chunked }}, splitOn(" ", null, false));
        defer out.release();
        try std.testing.expectEqual(@as(usize, 4), out.chunked.numChunks());
        const got = try collectLists(allocator, out);
        defer freeLists(allocator, got);
        try std.testing.expectEqual(expected.len, got.len);
        for (expected, got) |e, g| {
            try std.testing.expectEqual(e == null, g == null);
            if (e) |x| for (x, g.?) |p, q| try std.testing.expectEqualStrings(p.?, q.?);
        }
        var counts = try ctx.invokeScalar("count_substring", &.{.{ .chunked = chunked }}, match("a", false));
        defer counts.release();
        try std.testing.expectEqual(@as(usize, 4), counts.chunked.numChunks());
    }
}

test "string transform scalars, empty and all-null inputs" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    const st = DataType{ .string = {} };

    var r = try ctx.invokeScalar("replace_substring", &.{.{ .scalar = Scalar.init(st, .{ .string = "a-b-c" }) }}, replace("-", "+", null));
    defer r.release();
    try std.testing.expectEqualStrings("a+b+c", r.scalar.value.string);
    var sp = try ctx.invokeScalar("split_pattern", &.{.{ .scalar = Scalar.init(st, .{ .string = "a-b" }) }}, splitOn("-", null, false));
    defer sp.release();
    try std.testing.expect(sp.scalar.data_type == .list and !sp.scalar.isNull());
    const parts = try collectLists(allocator, sp);
    defer freeLists(allocator, parts);
    try expectLists(&.{&.{ "a", "b" }}, parts);
    var c = try ctx.invokeScalar("count_substring", &.{.{ .scalar = Scalar.init(st, .{ .string = "aXa" }) }}, match("a", false));
    defer c.release();
    try std.testing.expectEqual(@as(i32, 2), c.scalar.value.i32);
    var rep = try ctx.invokeScalar("binary_repeat", &.{ .{ .scalar = Scalar.init(st, .{ .string = "ab" }) }, .{ .scalar = common.scalarOf(i64, 3) } }, none);
    defer rep.release();
    try std.testing.expectEqualStrings("ababab", rep.scalar.value.string);
    var nul = try ctx.invokeScalar("ascii_lpad", &.{.{ .scalar = Scalar.init(st, .null) }}, pad(3, " "));
    defer nul.release();
    try std.testing.expect(nul.scalar.isNull());
    var nul_split = try ctx.invokeScalar("utf8_split_whitespace", &.{.{ .scalar = Scalar.init(st, .null) }}, none);
    defer nul_split.release();
    try std.testing.expect(nul_split.scalar.isNull() and nul_split.scalar.data_type == .list);

    inline for (string_tags) |tag| {
        var empty = try common.makeStrings(tag, allocator, &.{});
        defer empty.release();
        try common.expectInvokeStrings(&ctx, "replace_substring", &.{.{ .array = empty }}, replace("a", "b", null), &.{});
        try expectSplit(&ctx, "split_pattern", &.{.{ .array = empty }}, splitOn("a", null, false), &.{});
        var nulls = try common.makeStrings(tag, allocator, &.{ null, null, null });
        defer nulls.release();
        try common.expectInvokeStrings(&ctx, "utf8_center", &.{.{ .array = nulls }}, pad(4, "x"), &.{ null, null, null });
        try expectSplit(&ctx, "ascii_split_whitespace", &.{.{ .array = nulls }}, none, &.{ null, null, null });
        const L = if (tag == .large_string) i64 else i32;
        try common.expectInvokeWith(L, &ctx, "find_substring", &.{.{ .array = nulls }}, match("a", false), &.{ null, null, null });
        var n = try common.makeArray(i64, allocator, &.{ null, null, null });
        defer n.release();
        try common.expectInvokeStrings(&ctx, "binary_repeat", &.{ .{ .array = nulls }, .{ .array = n } }, none, &.{ null, null, null });
    }
}

test "string transform errors: options, invalid UTF-8 rows, negative and huge repeat counts, types and malformed input" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);

    var s = try common.makeStrings(.string, allocator, &.{"abc"});
    defer s.release();
    const sargs = [_]Datum{.{ .array = s }};
    const E = error.InvalidOptions;
    try std.testing.expectError(E, ctx.invokeScalar("replace_substring", &sargs, replace("", "x", null)));
    try std.testing.expectError(E, ctx.invokeScalar("replace_substring", &sargs, replace("\xc3", "x", null)));
    try std.testing.expectError(E, ctx.invokeScalar("replace_substring", &sargs, replace("a", "\xff", null)));
    try std.testing.expectError(E, ctx.invokeScalar("split_pattern", &sargs, splitOn("", null, false)));
    try std.testing.expectError(E, ctx.invokeScalar("split_pattern", &sargs, none));
    try std.testing.expectError(E, ctx.invokeScalar("ascii_lpad", &sargs, pad(3, "ab")));
    try std.testing.expectError(E, ctx.invokeScalar("ascii_lpad", &sargs, pad(3, "é")));
    try std.testing.expectError(E, ctx.invokeScalar("utf8_lpad", &sargs, pad(3, "ab")));
    try std.testing.expectError(E, ctx.invokeScalar("utf8_lpad", &sargs, pad(3, "\xe2\x82")));
    try std.testing.expectError(E, ctx.invokeScalar("utf8_lpad", &sargs, pad(3, "")));
    try std.testing.expectError(E, ctx.invokeScalar("ascii_rpad", &sargs, pad(@as(usize, 1) << 31, " ")));
    try std.testing.expectError(E, ctx.invokeScalar("count_substring", &sargs, none));
    try std.testing.expectError(E, ctx.invokeScalar("ascii_split_whitespace", &sargs, splitOn(" ", null, false)));
    // Binary layouts accept any pattern bytes.
    var b = try common.makeStrings(.binary, allocator, &.{"abc"});
    defer b.release();
    var ok = try ctx.invokeScalar("replace_substring", &.{.{ .array = b }}, replace("\xc3", "\xff", null));
    ok.release();
    try std.testing.expectError(E, ctx.invokeScalar("binary_slice", &.{.{ .array = b }}, .{ .slice = .{ .start = 0, .step = 0 } }));

    // A string array that never passed validateFull: the utf8 functions reject the bad row.
    var bad = try common.makeStrings(.binary, allocator, &.{ "ok", "\xff\xfe" });
    defer bad.release();
    var as_string = try ArrayRef.fromBorrowed(allocator, .{ .data_type = .{ .string = {} }, .length = 2, .null_count = 0, .buffers = bad.data().buffers });
    defer as_string.release();
    try std.testing.expectError(error.InvalidInput, ctx.invokeScalar("utf8_lpad", &.{.{ .array = as_string }}, pad(5, " ")));
    try std.testing.expectError(error.InvalidInput, ctx.invokeScalar("utf8_split_whitespace", &.{.{ .array = as_string }}, none));
    try std.testing.expectError(error.InvalidInput, ctx.invokeScalar("utf8_capitalize", &.{.{ .array = as_string }}, none));

    var counts = try common.makeArray(i64, allocator, &.{-1});
    defer counts.release();
    try std.testing.expectError(error.InvalidInput, ctx.invokeScalar("binary_repeat", &.{ sargs[0], .{ .array = counts } }, none));
    var huge = try common.makeArray(i64, allocator, &.{std.math.maxInt(i64)});
    defer huge.release();
    try std.testing.expectError(error.Overflow, ctx.invokeScalar("binary_repeat", &.{ sargs[0], .{ .array = huge } }, none));
    // A null count (or null row) is not looked at, whatever value its slot holds.
    const masked_counts = [_]i64{-5};
    const no_bits = [_]u8{0};
    var masked = try common.borrowedArray(i64, allocator, &masked_counts, &no_bits);
    defer masked.release();
    try common.expectInvokeStrings(&ctx, "binary_repeat", &.{ sargs[0], .{ .array = masked } }, none, &.{null});
    var i32s = try common.makeArray(i32, allocator, &.{2});
    defer i32s.release();
    try std.testing.expectError(error.NoMatchingKernel, ctx.invokeScalar("binary_repeat", &.{ sargs[0], .{ .array = i32s } }, none));
    try std.testing.expectError(error.NoMatchingKernel, ctx.invokeScalar("binary_slice", &sargs, .{ .slice = .{ .start = 0 } }));
    try std.testing.expectError(error.NoMatchingKernel, ctx.invokeScalar("binary_reverse", &sargs, none));
    try std.testing.expectError(error.NoMatchingKernel, ctx.invokeScalar("ascii_lpad", &.{.{ .array = b }}, pad(3, " ")));
    try std.testing.expectError(error.InvalidArity, ctx.invokeScalar("replace_substring", &.{ sargs[0], sargs[0] }, replace("a", "b", null)));

    // Offsets past the data buffer fail validateLayout instead of being read.
    const offsets = [_]i32{ 0, 100 };
    const broken_bufs = [_]common.SharedBuffer{ .empty, .fromSlice(std.mem.sliceAsBytes(&offsets)), .fromSlice("abc") };
    var broken = try ArrayRef.fromBorrowed(allocator, .{ .data_type = .{ .string = {} }, .length = 1, .null_count = 0, .buffers = &broken_bufs });
    defer broken.release();
    try std.testing.expectError(error.InvalidInput, ctx.invokeScalar("split_pattern", &.{.{ .array = broken }}, splitOn(",", null, false)));
    try std.testing.expectError(error.InvalidInput, ctx.invokeScalar("replace_substring", &.{.{ .array = broken }}, replace("a", "b", null)));
    try std.testing.expectError(error.InvalidInput, ctx.invokeScalar("binary_repeat", &.{ .{ .array = broken }, .{ .array = counts } }, none));
}
