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
        .record_batch, .table => return error.InvalidInput,
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

        // Case: the utf8_* functions map every cased code point, the ascii_* ones ASCII letters only.
        var k = try common.makeStrings(tag, allocator, &.{ "hELLO wORLD", "éa", "", null, "HeLLo 9" });
        defer k.release();
        const kargs = [_]Datum{.{ .array = k }};
        try common.expectInvokeStrings(&ctx, "utf8_capitalize", &kargs, none, &.{ "Hello world", "Éa", "", null, "Hello 9" });
        try common.expectInvokeStrings(&ctx, "ascii_capitalize", &kargs, none, &.{ "Hello world", "éa", "", null, "Hello 9" });
        try common.expectInvokeStrings(&ctx, "utf8_swapcase", &kargs, none, &.{ "Hello World", "ÉA", "", null, "hEllO 9" });
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
        try common.expectInvokeStrings(&ctx, "ascii_swapcase", &args, none, swapped);
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
    var f64s = try common.makeArray(f64, allocator, &.{2});
    defer f64s.release();
    try std.testing.expectError(error.NoMatchingKernel, ctx.invokeScalar("binary_repeat", &.{ sargs[0], .{ .array = f64s } }, none));
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

fn replaceSlice(start: i64, stop: i64, r: []const u8) Options {
    return .{ .replace_slice = .{ .start = start, .stop = stop, .replacement = r } };
}

const string_item = DataType{ .string = {} };
const large_string_item = DataType{ .large_string = {} };
const binary_item = DataType{ .binary = {} };

/// list<item> (or large_list) over `items` with `offsets` and an optional validity bitmap.
fn makeList(comptime large: bool, allocator: std.mem.Allocator, item: *const DataType, items: ArrayRef, offsets: anytype, validity: ?[]const u8, nulls: usize) !ArrayRef {
    const field = common.datatype.Field{ .name = "item", .data_type = item };
    const dt: DataType = if (large) .{ .large_list = .{ .value_field = field } } else .{ .list = .{ .value_field = field } };
    const bufs = [_]common.SharedBuffer{ if (validity) |v| .fromSlice(v) else .empty, .fromSlice(std.mem.sliceAsBytes(offsets)) };
    return ArrayRef.fromBorrowed(allocator, .{ .data_type = dt, .length = offsets.len - 1, .null_count = nulls, .buffers = &bufs, .children = &.{items} });
}

test "string transform replace_slice, zero_fill and binary_join on lists (Arrow C++ results)" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);

    // pyarrow 26 results; positions count code points (utf8) or bytes (binary).
    const R = struct { o: Options, text: [4]?[]const u8, bytes: [4]?[]const u8 };
    const cases = [_]R{
        .{ .o = replaceSlice(1, 3, "XY"), .text = .{ "hXYlo", "XY", null, "aXY" }, .bytes = .{ "hXYllo", "XY", null, "aXY" } },
        .{ .o = replaceSlice(-2, -1, "Z"), .text = .{ "hélZo", "Z", null, "Zb" }, .bytes = .{ "h\xc3\xa9lZo", "Z", null, "Zb" } },
        .{ .o = replaceSlice(3, 1, "Q"), .text = .{ "hélQlo", "Q", null, "abQ" }, .bytes = .{ "h\xc3\xa9Qllo", "Q", null, "abQ" } },
        .{ .o = replaceSlice(0, 0, "P"), .text = .{ "Phéllo", "P", null, "Pab" }, .bytes = .{ "Ph\xc3\xa9llo", "P", null, "Pab" } },
        .{ .o = replaceSlice(10, 20, "R"), .text = .{ "hélloR", "R", null, "abR" }, .bytes = .{ "h\xc3\xa9lloR", "R", null, "abR" } },
        .{ .o = replaceSlice(-10, 2, "S"), .text = .{ "Sllo", "S", null, "S" }, .bytes = .{ "S\xa9llo", "S", null, "S" } },
        .{ .o = replaceSlice(2, -10, "T"), .text = .{ "héTllo", "T", null, "abT" }, .bytes = .{ "h\xc3T\xa9llo", "T", null, "abT" } },
    };
    inline for (string_tags) |tag| {
        var a = try common.makeStrings(tag, allocator, &.{ "héllo", "", null, "ab" });
        defer a.release();
        for (cases) |c| try common.expectInvokeStrings(&ctx, "utf8_replace_slice", &.{.{ .array = a }}, c.o, &c.text);
        var z = try common.makeStrings(tag, allocator, &.{ "12", "-12", "+1", "-", "abc", "é1", "", null, "12345" });
        defer z.release();
        try common.expectInvokeStrings(&ctx, "utf8_zero_fill", &.{.{ .array = z }}, .{ .zero_fill = .{ .width = 4 } }, &.{ "0012", "-012", "+001", "-000", "0abc", "00é1", "0000", null, "12345" });
        try common.expectInvokeStrings(&ctx, "utf8_zero_fill", &.{.{ .array = z }}, .{ .zero_fill = .{ .width = 4, .padding = "é" } }, &.{ "éé12", "-é12", "+éé1", "-ééé", "éabc", "ééé1", "éééé", null, "12345" });
    }
    inline for (binary_tags) |tag| {
        var b = try common.makeStrings(tag, allocator, &.{ "h\xc3\xa9llo", "", null, "ab" });
        defer b.release();
        for (cases) |c| try common.expectInvokeStrings(&ctx, "binary_replace_slice", &.{.{ .array = b }}, c.o, &c.bytes);
    }

    // binary_join: a null list, a null item or a null separator gives null; [] gives "".
    var items = try common.makeStrings(.string, allocator, &.{ "a", "b", "x", null, "c" });
    defer items.release();
    const offs = [_]i32{ 0, 2, 2, 2, 4, 5 };
    const valid = [_]u8{0b11101};
    var lists = try makeList(false, allocator, &string_item, items, &offs, &valid, 1);
    defer lists.release();
    try common.expectInvokeStrings(&ctx, "binary_join", &.{ .{ .array = lists }, .{ .scalar = Scalar.init(string_item, .{ .string = "-" }) } }, none, &.{ "a-b", null, "", null, "c" });
    var seps = try common.makeStrings(.string, allocator, &.{ "+", "+", "+", "+", null });
    defer seps.release();
    try common.expectInvokeStrings(&ctx, "binary_join", &.{ .{ .array = lists }, .{ .array = seps } }, none, &.{ "a+b", null, "", null, null });
    var sliced = try lists.slice(3, 2);
    defer sliced.release();
    var sep_slice = try seps.slice(3, 2);
    defer sep_slice.release();
    try common.expectInvokeStrings(&ctx, "binary_join", &.{ .{ .array = sliced }, .{ .array = sep_slice } }, none, &.{ null, null });
    // large_list<large_string> with a large_string separator; list<binary>.
    var large_items = try common.makeStrings(.large_string, allocator, &.{ "x", "y", "z" });
    defer large_items.release();
    const loffs = [_]i64{ 0, 3, 3 };
    var large = try makeList(true, allocator, &large_string_item, large_items, &loffs, null, 0);
    defer large.release();
    try common.expectInvokeStrings(&ctx, "binary_join", &.{ .{ .array = large }, .{ .scalar = Scalar.init(large_string_item, .{ .string = ", " }) } }, none, &.{ "x, y, z", "" });
    var bin_items = try common.makeStrings(.binary, allocator, &.{ "\x00", "\xff" });
    defer bin_items.release();
    const boffs = [_]i32{ 0, 2 };
    var bins = try makeList(false, allocator, &binary_item, bin_items, &boffs, null, 0);
    defer bins.release();
    try common.expectInvokeStrings(&ctx, "binary_join", &.{ .{ .array = bins }, .{ .scalar = Scalar.init(binary_item, .{ .binary = "-" }) } }, none, &.{"\x00-\xff"});
    var chunks = try common.chunkedAt(allocator, lists, &.{ 1, 4 });
    defer chunks.release();
    var joined = try ctx.invokeScalar("binary_join", &.{ .{ .chunked = chunks }, .{ .scalar = Scalar.init(string_item, .{ .string = "-" }) } }, none);
    defer joined.release();
    try std.testing.expectEqual(@as(usize, 3), joined.chunked.numChunks());
    const got = try common.collectStrings(allocator, joined);
    defer allocator.free(got);
    try common.expectStrings(&.{ "a-b", null, "", null, "c" }, got);

    // Errors: options, a separator of another layout, a child that is not the declared layout.
    var s = try common.makeStrings(.string, allocator, &.{"abc"});
    defer s.release();
    try std.testing.expectError(error.InvalidOptions, ctx.invokeScalar("utf8_replace_slice", &.{.{ .array = s }}, replaceSlice(0, 1, "\xff")));
    try std.testing.expectError(error.InvalidOptions, ctx.invokeScalar("utf8_zero_fill", &.{.{ .array = s }}, .{ .zero_fill = .{ .width = 3, .padding = "00" } }));
    try std.testing.expectError(error.InvalidOptions, ctx.invokeScalar("utf8_zero_fill", &.{.{ .array = s }}, .{ .pad = .{ .width = 3 } }));
    try std.testing.expectError(error.InvalidOptions, ctx.invokeScalar("binary_replace_slice", &.{.{ .array = s }}, replaceSlice(0, 1, "\xff")));
    try std.testing.expectError(error.NoMatchingKernel, ctx.invokeScalar("binary_join", &.{ .{ .array = lists }, .{ .scalar = Scalar.init(large_string_item, .{ .string = "-" }) } }, none));
    var liar = try makeList(false, allocator, &string_item, bin_items, &boffs, null, 0);
    defer liar.release();
    try std.testing.expectError(error.InvalidInput, ctx.invokeScalar("binary_join", &.{ .{ .array = liar }, .{ .scalar = Scalar.init(string_item, .{ .string = "-" }) } }, none));
}

test "string transform binary_join on list_view and view layouts, binary_replace_slice on the string layouts (pyarrow 26 results)" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);

    // binary_replace_slice counts bytes on string, large_string and string_view (pyarrow 26 on
    // string / large_string; it has no string_view kernel). A result that splits "\u{e9}" is not
    // UTF-8: pyarrow writes it, zarrow answers InvalidInput.
    inline for (string_tags) |tag| {
        var a = try common.makeStrings(tag, allocator, &.{ "h\u{e9}llo", "", null, "ab" });
        defer a.release();
        try common.expectInvokeStrings(&ctx, "binary_replace_slice", &.{.{ .array = a }}, replaceSlice(1, 3, "XY"), &.{ "hXYllo", "XY", null, "aXY" });
        try common.expectInvokeStrings(&ctx, "binary_replace_slice", &.{.{ .array = a }}, replaceSlice(-2, -1, "Z"), &.{ "h\u{e9}lZo", "Z", null, "Zb" });
        try common.expectInvokeStrings(&ctx, "binary_replace_slice", &.{.{ .array = a }}, replaceSlice(0, 0, "\u{ff}"), &.{ "\u{ff}h\u{e9}llo", "\u{ff}", null, "\u{ff}ab" });
        var tail = try a.slice(1, 3);
        defer tail.release();
        try common.expectInvokeStrings(&ctx, "binary_replace_slice", &.{.{ .array = tail }}, replaceSlice(0, 1, "Q"), &.{ "Q", null, "Qb" });
        try std.testing.expectError(error.InvalidInput, ctx.invokeScalar("binary_replace_slice", &.{.{ .array = a }}, replaceSlice(2, 3, "Z")));
        try std.testing.expectError(error.InvalidInput, ctx.invokeScalar("binary_replace_slice", &.{.{ .array = a }}, replaceSlice(1, 2, "\u{e9}")));
        var ch = try common.chunkedAt(allocator, a, &.{2});
        defer ch.release();
        try common.expectInvokeStrings(&ctx, "binary_replace_slice", &.{.{ .chunked = ch }}, replaceSlice(1, 3, "XY"), &.{ "hXYllo", "XY", null, "aXY" });
        try common.expectInvokeStrings(&ctx, "binary_replace_slice", &.{.{ .scalar = Scalar.init(@unionInit(DataType, @tagName(tag), {}), .{ .string = "abc" }) }}, replaceSlice(1, 2, "-"), &.{"a-c"});
    }

    // binary_join of [["a", "b"], null, [], ["x", null], ["c"]] for every item layout under
    // list, large_list, list_view and large_list_view (Arrow registers list / large_list of the
    // offset layouts only; the results equal pyarrow 26's list<string> ones): a null list, a
    // null item or a null separator gives null.
    inline for (string_tags ++ binary_tags) |tag| {
        const item = comptime @unionInit(DataType, @tagName(tag), {});
        const field = common.datatype.Field{ .name = "item", .data_type = &item };
        var items = try common.makeStrings(tag, allocator, &.{ "a", "b", "x", null, "c" });
        defer items.release();
        var seps = try common.makeStrings(tag, allocator, &.{ "+", "+", "+", "+", null });
        defer seps.release();
        const text = comptime (tag == .string or tag == .large_string or tag == .string_view);
        const dash = Scalar.init(item, if (text) .{ .string = "-" } else .{ .binary = "-" });
        inline for (.{ common.array_mod.ListBuilder, common.array_mod.LargeListBuilder, common.array_mod.ListViewBuilder, common.array_mod.LargeListViewBuilder }) |B| {
            var b = try B.init(allocator, 5, field);
            defer b.deinit();
            try b.appendLen(2);
            try b.appendNull();
            try b.appendLen(0);
            try b.appendLen(2);
            try b.appendLen(1);
            var lists = try b.finish(items);
            defer lists.release();
            try common.expectInvokeStrings(&ctx, "binary_join", &.{ .{ .array = lists }, .{ .scalar = dash } }, none, &.{ "a-b", null, "", null, "c" });
            try common.expectInvokeStrings(&ctx, "binary_join", &.{ .{ .array = lists }, .{ .array = seps } }, none, &.{ "a+b", null, "", null, null });
            var tail = try lists.slice(2, 3);
            defer tail.release();
            var sep_tail = try seps.slice(2, 3);
            defer sep_tail.release();
            try common.expectInvokeStrings(&ctx, "binary_join", &.{ .{ .array = tail }, .{ .array = sep_tail } }, none, &.{ "", null, null });
        }
    }
    // list_view rows out of order and overlapping: offsets [3, 0, 1], sizes [2, 2, 3] over
    // ["a", "b", "c", "d", "e"] join to ["d-e", "a-b", "b-c-d"].
    var letters = try common.makeStrings(.string_view, allocator, &.{ "a", "b", "c", "d", "e" });
    defer letters.release();
    const sv = DataType{ .string_view = {} };
    const offs = [_]i32{ 3, 0, 1 };
    const sizes = [_]i32{ 2, 2, 3 };
    const bufs = [_]common.SharedBuffer{ .empty, .fromSlice(std.mem.sliceAsBytes(&offs)), .fromSlice(std.mem.sliceAsBytes(&sizes)) };
    var views = try ArrayRef.fromBorrowed(allocator, .{ .data_type = .{ .list_view = .{ .value_field = .{ .name = "item", .data_type = &sv } } }, .length = 3, .null_count = 0, .buffers = &bufs, .children = &.{letters} });
    defer views.release();
    try common.expectInvokeStrings(&ctx, "binary_join", &.{ .{ .array = views }, .{ .scalar = Scalar.init(sv, .{ .string = "-" }) } }, none, &.{ "d-e", "a-b", "b-c-d" });
    // A separator of another layout than the items is NoMatchingKernel (as Arrow).
    try std.testing.expectError(error.NoMatchingKernel, ctx.invokeScalar("binary_join", &.{ .{ .array = views }, .{ .scalar = Scalar.init(.{ .string = {} }, .{ .string = "-" }) } }, none));
}

/// A fixed_size_binary array of `width` bytes per row.
fn makeFixed(allocator: std.mem.Allocator, width: usize, rows: []const ?[]const u8) !ArrayRef {
    var b = try common.array_mod.FixedSizeBinaryBuilder.init(allocator, width, rows.len);
    defer b.deinit();
    for (rows) |r| if (r) |x| try b.append(x) else try b.appendNull();
    return b.finish();
}

/// The rows of a fixed_size_binary datum (array, chunked or scalar) of width `width`; every
/// array part must pass validateFull and have that width.
fn fixedRows(allocator: std.mem.Allocator, datum: Datum, width: usize) ![]?[]const u8 {
    var parts: std.ArrayList(?[]const u8) = .empty;
    errdefer parts.deinit(allocator);
    const arrays: []const ArrayRef = switch (datum) {
        .array => |*a| a[0..1],
        .chunked => |c| c.chunks(),
        .scalar => |s| {
            try std.testing.expectEqual(@as(i32, @intCast(width)), s.data_type.fixed_size_binary.byte_width);
            try parts.append(allocator, if (s.isNull()) null else s.value.binary);
            return parts.toOwnedSlice(allocator);
        },
        else => return error.TestUnexpectedResult,
    };
    for (arrays) |a| {
        const d = a.data();
        try d.validateFull();
        try std.testing.expectEqual(@as(i32, @intCast(width)), d.data_type.fixed_size_binary.byte_width);
        const view = common.array_mod.FixedSizeBinaryArray{ .data = d };
        for (0..d.length) |i| try parts.append(allocator, if (d.isNull(i)) null else try view.value(i));
    }
    return parts.toOwnedSlice(allocator);
}

fn expectFixed(ctx: *ExecContext, name: []const u8, args: []const Datum, options: Options, width: usize, expected: []const ?[]const u8) !void {
    const allocator = std.testing.allocator;
    var out = try ctx.invokeScalar(name, args, options);
    defer out.release();
    const rows = try fixedRows(allocator, out, width);
    defer allocator.free(rows);
    try common.expectStrings(expected, rows);
}

fn sliceBytes(start: i64, stop: i64, step: i64) Options {
    return .{ .slice = .{ .start = start, .stop = stop, .step = step } };
}

test "string transform fixed_size_binary: binary_length, binary_slice, binary_replace_slice, count_substring, find_substring on nulls, sliced, chunked and scalar inputs (pyarrow 26 results)" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);

    // fixed_size_binary[4] [abcd, null, wxab, a\xc3\xa9b]; pyarrow 26 gives binary_slice and
    // binary_replace_slice a fixed_size_binary of the computed width, the others int32.
    var a = try makeFixed(allocator, 4, &.{ "abcd", null, "wxab", "a\xc3\xa9b" });
    defer a.release();
    const ad: Datum = .{ .array = a };
    try common.expectInvokeWith(i32, &ctx, "binary_length", &.{ad}, none, &.{ 4, null, 4, 4 });
    const max = std.math.maxInt(i64);
    const S = struct { o: Options, w: usize, rows: [4]?[]const u8 };
    const slices = [_]S{
        .{ .o = sliceBytes(1, 3, 1), .w = 2, .rows = .{ "bc", null, "xa", "\xc3\xa9" } },
        .{ .o = sliceBytes(0, max, 2), .w = 2, .rows = .{ "ac", null, "wa", "a\xa9" } },
        .{ .o = sliceBytes(-2, max, 1), .w = 2, .rows = .{ "cd", null, "ab", "\xa9b" } },
        .{ .o = sliceBytes(3, 1, -1), .w = 2, .rows = .{ "dc", null, "ba", "b\xa9" } },
        .{ .o = sliceBytes(-10, 10, 1), .w = 4, .rows = .{ "abcd", null, "wxab", "a\xc3\xa9b" } },
    };
    for (slices) |c| try expectFixed(&ctx, "binary_slice", &.{ad}, c.o, c.w, &c.rows);
    const replaces = [_]S{
        .{ .o = replaceSlice(1, 3, "Q"), .w = 3, .rows = .{ "aQd", null, "wQb", "aQb" } },
        .{ .o = replaceSlice(0, 0, "XY"), .w = 6, .rows = .{ "XYabcd", null, "XYwxab", "XYa\xc3\xa9b" } },
        .{ .o = replaceSlice(-1, 4, ""), .w = 3, .rows = .{ "abc", null, "wxa", "a\xc3\xa9" } },
        .{ .o = replaceSlice(3, 1, "Z"), .w = 5, .rows = .{ "abcZd", null, "wxaZb", "a\xc3\xa9Zb" } },
        .{ .o = replaceSlice(10, 20, "R"), .w = 5, .rows = .{ "abcdR", null, "wxabR", "a\xc3\xa9bR" } },
        .{ .o = replaceSlice(-10, 2, "S"), .w = 3, .rows = .{ "Scd", null, "Sab", "S\xa9b" } },
    };
    for (replaces) |c| try expectFixed(&ctx, "binary_replace_slice", &.{ad}, c.o, c.w, &c.rows);
    const M = struct { o: Options, count: [4]?i32, find: [4]?i32 };
    const matches = [_]M{
        .{ .o = match("ab", false), .count = .{ 1, null, 1, 0 }, .find = .{ 0, null, 2, -1 } },
        .{ .o = match("AB", true), .count = .{ 1, null, 1, 0 }, .find = .{ 0, null, 2, -1 } },
        .{ .o = match("", false), .count = .{ 5, null, 5, 5 }, .find = .{ 0, null, 0, 0 } },
        .{ .o = match("b", false), .count = .{ 1, null, 1, 1 }, .find = .{ 1, null, 3, 3 } },
        .{ .o = match("\xc3\xa9", false), .count = .{ 0, null, 0, 1 }, .find = .{ -1, null, -1, 1 } },
        .{ .o = match("\xc3\x89", true), .count = .{ 0, null, 0, 0 }, .find = .{ -1, null, -1, -1 } },
    };
    for (matches) |c| {
        try common.expectInvokeWith(i32, &ctx, "count_substring", &.{ad}, c.o, &c.count);
        try common.expectInvokeWith(i32, &ctx, "find_substring", &.{ad}, c.o, &c.find);
    }

    // Sliced, chunked (one result chunk per input chunk) and scalar inputs; a null scalar.
    var tail = try a.slice(1, 3);
    defer tail.release();
    try common.expectInvokeWith(i32, &ctx, "binary_length", &.{.{ .array = tail }}, none, &.{ null, 4, 4 });
    try expectFixed(&ctx, "binary_slice", &.{.{ .array = tail }}, sliceBytes(1, 3, 1), 2, &.{ null, "xa", "\xc3\xa9" });
    try expectFixed(&ctx, "binary_replace_slice", &.{.{ .array = tail }}, replaceSlice(1, 3, "Q"), 3, &.{ null, "wQb", "aQb" });
    try common.expectInvokeWith(i32, &ctx, "find_substring", &.{.{ .array = tail }}, match("b", false), &.{ null, 3, 3 });
    var chunks = try common.chunkedAt(allocator, a, &.{ 1, 3 });
    defer chunks.release();
    try expectFixed(&ctx, "binary_slice", &.{.{ .chunked = chunks }}, sliceBytes(1, 3, 1), 2, &.{ "bc", null, "xa", "\xc3\xa9" });
    try expectFixed(&ctx, "binary_replace_slice", &.{.{ .chunked = chunks }}, replaceSlice(0, 0, "XY"), 6, &.{ "XYabcd", null, "XYwxab", "XYa\xc3\xa9b" });
    try common.expectInvokeWith(i32, &ctx, "count_substring", &.{.{ .chunked = chunks }}, match("ab", false), &.{ 1, null, 1, 0 });
    try common.expectInvokeWith(i32, &ctx, "binary_length", &.{.{ .chunked = chunks }}, none, &.{ 4, null, 4, 4 });
    const fsb4 = DataType{ .fixed_size_binary = .{ .byte_width = 4 } };
    const sc: Datum = .{ .scalar = Scalar.init(fsb4, .{ .binary = "abcd" }) };
    const null_sc: Datum = .{ .scalar = Scalar.init(fsb4, .null) };
    try expectFixed(&ctx, "binary_slice", &.{sc}, sliceBytes(1, 3, 1), 2, &.{"bc"});
    try expectFixed(&ctx, "binary_slice", &.{null_sc}, sliceBytes(1, 3, 1), 2, &.{null});
    try expectFixed(&ctx, "binary_replace_slice", &.{sc}, replaceSlice(1, 3, "Q"), 3, &.{"aQd"});
    try common.expectInvokeWith(i32, &ctx, "binary_length", &.{sc}, none, &.{4});
    try common.expectInvokeWith(i32, &ctx, "count_substring", &.{null_sc}, match("a", false), &.{null});
    try common.expectInvokeWith(i32, &ctx, "find_substring", &.{sc}, match("cd", false), &.{2});
    // An empty result width: pyarrow 26 gives fixed_size_binary[0] (binary_slice(5, 9) and
    // binary_replace_slice(0, 4, "") give [b"", null, b"", b""]); zarrow has no zero-width
    // fixed_size_binary (validateLayout rejects it), so the call is UnsupportedType.
    try std.testing.expectError(error.UnsupportedType, ctx.invokeScalar("binary_slice", &.{ad}, sliceBytes(5, 9, 1)));
    try std.testing.expectError(error.UnsupportedType, ctx.invokeScalar("binary_replace_slice", &.{ad}, replaceSlice(0, 4, "")));
    // A step of 0 is an option error, as for the binary layouts.
    try std.testing.expectError(error.InvalidOptions, ctx.invokeScalar("binary_slice", &.{ad}, sliceBytes(0, 1, 0)));

    // Oracle: random fixed_size_binary[5] rows with nulls, sliced, against a per-row loop.
    var prng = std.Random.DefaultPrng.init(0xf5b);
    const rng = prng.random();
    for (0..4) |_| {
        const n = 100 + rng.uintLessThan(usize, 60);
        const off = rng.uintLessThan(usize, 9);
        const store = try allocator.alloc(u8, (n + off) * 5);
        defer allocator.free(store);
        for (store) |*c| c.* = "ab\xc3\xa9x"[rng.uintLessThan(usize, 5)];
        const rows = try allocator.alloc(?[]const u8, n + off);
        defer allocator.free(rows);
        for (rows, 0..) |*r, i| r.* = if (rng.uintLessThan(u8, 5) == 0) null else store[i * 5 ..][0..5];
        var full = try makeFixed(allocator, 5, rows);
        defer full.release();
        var arr = try full.slice(off, n);
        defer arr.release();
        const in = rows[off..];
        const want_s = try allocator.alloc(?[]const u8, n);
        defer allocator.free(want_s);
        const buf = try allocator.alloc(u8, n * 8);
        defer allocator.free(buf);
        // binary_slice(1, 4, 2): bytes 1 and 3; binary_replace_slice(2, 4, "Q"): 4 bytes.
        for (want_s, in, 0..) |*w, r, i| w.* = if (r) |x| blk: {
            buf[i * 8] = x[1];
            buf[i * 8 + 1] = x[3];
            break :blk buf[i * 8 ..][0..2];
        } else null;
        try expectFixed(&ctx, "binary_slice", &.{.{ .array = arr }}, sliceBytes(1, 4, 2), 2, want_s);
        for (want_s, in, 0..) |*w, r, i| w.* = if (r) |x| blk: {
            @memcpy(buf[i * 8 ..][0..2], x[0..2]);
            buf[i * 8 + 2] = 'Q';
            buf[i * 8 + 3] = x[4];
            break :blk buf[i * 8 ..][0..4];
        } else null;
        try expectFixed(&ctx, "binary_replace_slice", &.{.{ .array = arr }}, replaceSlice(2, 4, "Q"), 4, want_s);
        const counts = try allocator.alloc(?i32, n);
        defer allocator.free(counts);
        const finds = try allocator.alloc(?i32, n);
        defer allocator.free(finds);
        const lens = try allocator.alloc(?i32, n);
        defer allocator.free(lens);
        for (in, counts, finds, lens) |r, *c, *f, *l| {
            const x = r orelse {
                c.* = null;
                f.* = null;
                l.* = null;
                continue;
            };
            c.* = @intCast(std.mem.count(u8, x, "ab"));
            f.* = if (std.mem.indexOf(u8, x, "\xc3\xa9")) |p| @intCast(p) else -1;
            l.* = 5;
        }
        try common.expectInvokeWith(i32, &ctx, "count_substring", &.{.{ .array = arr }}, match("ab", false), counts);
        try common.expectInvokeWith(i32, &ctx, "find_substring", &.{.{ .array = arr }}, match("\xc3\xa9", false), finds);
        try common.expectInvokeWith(i32, &ctx, "binary_length", &.{.{ .array = arr }}, none, lens);
    }
}

test "binary_repeat with counts of every integer type: nulls, negative counts, sliced, chunked, scalars and dictionaries (pyarrow 26 results)" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    inline for (.{ TypeTag.string, TypeTag.large_string, TypeTag.binary, TypeTag.large_binary }) |tag| {
        const dt = @unionInit(DataType, @tagName(tag), {});
        var v = try common.makeStrings(tag, allocator, &.{ "ab", null, "ba", "c" });
        defer v.release();
        inline for (.{ i8, i16, i32, u8, u16, u32, u64 }) |T| {
            // binary_repeat(["ab", null, "ba", "c"], [2, 1, null, 0]) -> ["abab", null, null, ""].
            var n = try common.makeArray(T, allocator, &.{ 2, 1, null, 0 });
            defer n.release();
            try common.expectInvokeStrings(&ctx, "binary_repeat", &.{ .{ .array = v }, .{ .array = n } }, none, &.{ "abab", null, null, "" });
            try common.expectInvokeStrings(&ctx, "binary_repeat", &.{ .{ .array = v }, .{ .scalar = common.scalarOf(T, 3) } }, none, &.{ "ababab", null, "bababa", "ccc" });
            var out = try ctx.invokeScalar("binary_repeat", &.{ .{ .array = v }, .{ .array = n } }, none);
            defer out.release();
            try std.testing.expect(out.dataType().eql(dt));
            // Sliced values and counts (rows 1..4).
            var sv = try v.slice(1, 3);
            defer sv.release();
            var sn = try n.slice(1, 3);
            defer sn.release();
            try common.expectInvokeStrings(&ctx, "binary_repeat", &.{ .{ .array = sv }, .{ .array = sn } }, none, &.{ null, null, "" });
            // Chunked counts: a chunked result.
            var cn = try common.chunkedAt(allocator, n, &.{1});
            defer cn.release();
            var chunked = try ctx.invokeScalar("binary_repeat", &.{ .{ .array = v }, .{ .chunked = cn } }, none);
            defer chunked.release();
            try std.testing.expectEqual(@as(usize, 2), chunked.chunked.numChunks());
            const rows = try common.collectStrings(allocator, chunked);
            defer allocator.free(rows);
            try common.expectStrings(&.{ "abab", null, null, "" }, rows);
        }
        // A negative count in a valid row is InvalidInput ("Repeat count must be a non-negative
        // integer"); in a null row it is not looked at.
        var neg = try common.makeArray(i8, allocator, &.{ 2, 1, -1, 1 });
        defer neg.release();
        try std.testing.expectError(error.InvalidInput, ctx.invokeScalar("binary_repeat", &.{ .{ .array = v }, .{ .array = neg } }, none));
        var neg_null_row = try common.makeArray(i16, allocator, &.{ 2, -1, 1, 1 });
        defer neg_null_row.release();
        var masked_v = try common.makeStrings(tag, allocator, &.{ "ab", null, null, null });
        defer masked_v.release();
        try common.expectInvokeStrings(&ctx, "binary_repeat", &.{ .{ .array = masked_v }, .{ .array = neg_null_row } }, none, &.{ "abab", null, null, null });
    }
    // A uint64 count past int64 fails the cast (pyarrow 26: "Integer value ... not in range").
    var s = try common.makeStrings(.string, allocator, &.{"ab"});
    defer s.release();
    var big = try common.makeArray(u64, allocator, &.{std.math.maxInt(u64)});
    defer big.release();
    try std.testing.expectError(error.InvalidCast, ctx.invokeScalar("binary_repeat", &.{ .{ .array = s }, .{ .array = big } }, none));
    // dictionary<string> decodes first: binary_repeat(dict ["ab", null, "ba"], int32 [2, 1, null]).
    var dict = try common.makeStrings(.string, allocator, &.{ "ab", "ba" });
    defer dict.release();
    var d = try common.makeDictionary(allocator, dict, .{ .bit_width = 8, .signed = true }, &.{ 0, null, 1 });
    defer d.release();
    var counts = try common.makeArray(i32, allocator, &.{ 2, 1, null });
    defer counts.release();
    var dout = try ctx.invokeScalar("binary_repeat", &.{ .{ .array = d }, .{ .array = counts } }, none);
    defer dout.release();
    try std.testing.expect(dout.dataType() == .string);
    try common.expectInvokeStrings(&ctx, "binary_repeat", &.{ .{ .array = d }, .{ .array = counts } }, none, &.{ "abab", null, null });
}
