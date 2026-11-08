const std = @import("std");
const regex = @import("regex.zig");

const Regex = regex.Regex;

fn compileOpts(pattern: []const u8, options: regex.Options) !*Regex {
    return Regex.compile(std.testing.allocator, pattern, options, null);
}

/// `pattern` on `text`: the leftmost-first match span, or null.
fn expectFind(pattern: []const u8, text: []const u8, expected: ?[2]usize) !void {
    return expectFindOpts(pattern, .{}, text, expected);
}

fn expectFindOpts(pattern: []const u8, options: regex.Options, text: []const u8, expected: ?[2]usize) !void {
    const re = try compileOpts(pattern, options);
    defer re.deinit();
    const got = try re.find(text, 0);
    if (expected) |e| {
        if (got == null) {
            std.debug.print("pattern {s} on {s}: no match, expected [{d}, {d})\n", .{ pattern, text, e[0], e[1] });
            return error.TestExpectedEqual;
        }
        if (got.?.start != e[0] or got.?.end != e[1]) {
            std.debug.print("pattern {s} on {s}: [{d}, {d}), expected [{d}, {d})\n", .{ pattern, text, got.?.start, got.?.end, e[0], e[1] });
            return error.TestExpectedEqual;
        }
    } else if (got) |g| {
        std.debug.print("pattern {s} on {s}: [{d}, {d}), expected no match\n", .{ pattern, text, g.start, g.end });
        return error.TestExpectedEqual;
    }
    try std.testing.expectEqual(expected != null, try re.isMatch(text));
    // The Pike VM agrees with the DFA.
    var slots: [2]usize = undefined;
    const pm = try re.pike.search(&re.forward, text, 0, false, &slots);
    try std.testing.expectEqual(expected != null, pm);
    if (expected) |e| try std.testing.expectEqualSlices(usize, &e, &slots);
}

test "regex literals, classes, anchors and quantifiers (RE2 results)" {
    try expectFind("abc", "xxabcxx", .{ 2, 5 });
    try expectFind("", "abc", .{ 0, 0 });
    try expectFind("a.c", "a\nc abc", .{ 4, 7 });
    try expectFind("(?s)a.c", "a\nc abc", .{ 0, 3 });
    try expectFind("[a-c]+", "xxbcaz", .{ 2, 5 });
    try expectFind("[^a-c]+", "abxyc", .{ 2, 4 });
    try expectFind("[]a]+", "x]a]", .{ 1, 4 });
    try expectFind("[a-]+", "x-a-", .{ 1, 4 });
    try expectFind("\\d+", "ab123c", .{ 2, 5 });
    try expectFind("\\D+", "12ab3", .{ 2, 4 });
    try expectFind("\\w+", "  foo_1 ", .{ 2, 7 });
    try expectFind("\\s+", "a \t\nb", .{ 1, 4 });
    try expectFind("[[:alpha:]]+", "12abC3", .{ 2, 5 });
    try expectFind("[[:^digit:]]+", "12abC3", .{ 2, 5 });
    try expectFind("^abc", "abcabc", .{ 0, 3 });
    try expectFind("^abc", "xabc", null);
    try expectFind("abc$", "abcabc", .{ 3, 6 });
    try expectFind("abc$", "abc\n", null);
    try expectFind("(?m)abc$", "abc\nabc", .{ 0, 3 });
    try expectFind("(?m)^b", "a\nb", .{ 2, 3 });
    try expectFind("\\Aa", "aa", .{ 0, 1 });
    try expectFind("a\\z", "aa", .{ 1, 2 });
    try expectFind("\\bfoo\\b", "afoo foo", .{ 5, 8 });
    try expectFind("\\Boo", "foo", .{ 1, 3 });
    try expectFind("a*", "baaa", .{ 0, 0 });
    try expectFind("a+", "baaa", .{ 1, 4 });
    try expectFind("a+?", "baaa", .{ 1, 2 });
    try expectFind("a*?b", "aaab", .{ 0, 4 });
    try expectFind("a{2}", "aaaa", .{ 0, 2 });
    try expectFind("a{2,}", "aaaa", .{ 0, 4 });
    try expectFind("a{1,3}", "aaaa", .{ 0, 3 });
    try expectFind("a{1,3}?", "aaaa", .{ 0, 1 });
    try expectFind("a{,3}", "a{,3}", .{ 0, 5 });
    try expectFind("(?U)a+", "aaa", .{ 0, 1 });
    try expectFind("(?U)a+?", "aaa", .{ 0, 3 });
    // Leftmost-first: the first alternative wins, not the longest.
    try expectFind("a|ab", "ab", .{ 0, 1 });
    try expectFind("ab|a", "ab", .{ 0, 2 });
    try expectFind("(a|ab)(c|bcd)", "abcd", .{ 0, 4 });
    try expectFind("x*", "", .{ 0, 0 });
    try expectFind("(a*)*", "aa", .{ 0, 2 });
    try expectFind("(a*)+b", "aab", .{ 0, 3 });
    try expectFind("\\x41\\x{42}\\103", "ABC", .{ 0, 3 });
    try expectFind("\\Q.*\\E+", "a.*.**", .{ 1, 3 });
    try expectFind("\\Qa\\E\\Qb", "xab", .{ 1, 3 });
    try expectFind("a\\.b", "axb a.b", .{ 4, 7 });
    try expectFind("[\\d\\-x]+", "a1-x2", .{ 1, 5 });
}

test "regex UTF-8, Latin-1, Unicode classes and case folding (RE2 results)" {
    try expectFind("^.$", "é", .{ 0, 2 });
    try expectFindOpts("^.$", .{ .latin1 = true }, "\xe9", .{ 0, 1 });
    try expectFind("^.$", "\xe9", null);
    try expectFind("\\C", "\xe9", .{ 0, 1 });
    try expectFind("[α-ω]+", "abγδε", .{ 2, 8 });
    try expectFind("\\pL+", "12héllo3", .{ 2, 8 });
    try expectFind("\\p{Lu}", "abcÉ", .{ 3, 5 });
    try expectFind("\\PL+", "ab12", .{ 2, 4 });
    try expectFind("\\p{^L}+", "ab12", .{ 2, 4 });
    try expectFind("[^a]", "a☃", .{ 1, 4 });
    try expectFind("(?i)k", "\u{212A}", .{ 0, 3 });
    try expectFind("(?i)s", "\u{17F}", .{ 0, 2 });
    try expectFind("(?i)i", "\u{131}\u{130}I", .{ 4, 5 });
    try expectFind("(?i)école", "ÉCOLE", .{ 0, 6 });
    try expectFind("(?i)[a-z]+", "ÀBC", .{ 2, 4 });
    try expectFind("(?i)σ", "ς", .{ 0, 2 });
    try expectFind("(?i)ß", "ẞ", .{ 0, 3 });
    try expectFind("(?i:a)b", "AB Ab", .{ 3, 5 });
    try expectFindOpts("é", .{ .case_insensitive = true }, "É", .{ 0, 2 });
    // Latin-1: RE2 folds the ASCII letters only.
    try expectFindOpts("\xe9", .{ .case_insensitive = true, .latin1 = true }, "\xc9", null);
    try expectFindOpts("[a-c]+", .{ .case_insensitive = true, .latin1 = true }, "xAbC", .{ 1, 4 });
    try expectFindOpts("a.b", .{ .literal = true }, "axb a.b", .{ 4, 7 });
}

test "regex captures and group names" {
    const re = try compileOpts("(?P<x>a)(?P<y>b)?(c)", .{});
    defer re.deinit();
    try std.testing.expectEqual(@as(usize, 3), re.groupCount());
    try std.testing.expectEqualStrings("x", re.groupName(1).?);
    try std.testing.expectEqualStrings("y", re.groupName(2).?);
    try std.testing.expect(re.groupName(3) == null);
    var g: [4]?regex.Span = undefined;
    try std.testing.expect(try re.captures("zzac", 0, &g));
    try std.testing.expectEqual(regex.Span{ .start = 2, .end = 4 }, g[0].?);
    try std.testing.expectEqual(regex.Span{ .start = 2, .end = 3 }, g[1].?);
    try std.testing.expect(g[2] == null);
    try std.testing.expectEqual(regex.Span{ .start = 3, .end = 4 }, g[3].?);
    try std.testing.expect(!try re.captures("zzab", 0, &g));
    // The last iteration of a repeated group is kept.
    const rep = try compileOpts("(a|b)+", .{});
    defer rep.deinit();
    var g2: [2]?regex.Span = undefined;
    try std.testing.expect(try rep.captures("xabab", 0, &g2));
    try std.testing.expectEqual(regex.Span{ .start = 4, .end = 5 }, g2[1].?);
    // Searches from a position see the text before it.
    const b = try compileOpts("\\bx", .{});
    defer b.deinit();
    try std.testing.expect(try b.find("ax x", 1) != null);
    try std.testing.expectEqual(@as(usize, 3), (try b.find("ax x", 1)).?.start);
}

test "regex errors: syntax, unsupported constructs and limits" {
    const cases = [_]struct { []const u8, anyerror }{
        .{ "(a", error.RegexSyntax },            .{ "a)", error.RegexSyntax },
        .{ "[a", error.RegexSyntax },            .{ "[z-a]", error.RegexSyntax },
        .{ "*a", error.RegexSyntax },            .{ "a**", error.RegexSyntax },
        .{ "a{1001}", error.RegexSyntax },       .{ "a{2,1}", error.RegexSyntax },
        .{ "(a{100}){100}", error.RegexSyntax }, .{ "\\1", error.RegexUnsupported },
        .{ "(?=a)", error.RegexUnsupported },    .{ "(?<!a)", error.RegexUnsupported },
        .{ "(?P=n)", error.RegexUnsupported },   .{ "\\p{Greek}", error.RegexUnsupported },
        .{ "\\pX", error.RegexSyntax },          .{ "\\q", error.RegexSyntax },
        .{ "\\", error.RegexSyntax },            .{ "(?P<>a)", error.RegexSyntax },
        .{ "(?x)", error.RegexSyntax },          .{ "(?i-)", error.RegexSyntax },
        .{ "\xff", error.RegexSyntax },          .{ "[[:foo:]]", error.RegexSyntax },
        .{ "\\x{110000}", error.RegexSyntax },
    };
    for (cases) |c| {
        var diag: regex.Diagnostic = .{};
        try std.testing.expectError(c[1], Regex.compile(std.testing.allocator, c[0], .{}, &diag));
        try std.testing.expect(diag.message.len > 0);
    }
    // Program size and nesting limits.
    try std.testing.expectError(error.RegexTooLarge, Regex.compile(std.testing.allocator, "(\\pL{1000}){1}\\pL{1000}", .{}, null));
    var deep: [600]u8 = undefined;
    @memset(deep[0..300], '(');
    @memset(deep[300..], ')');
    try std.testing.expectError(error.RegexTooLarge, Regex.compile(std.testing.allocator, &deep, .{}, null));
    const long = try std.testing.allocator.alloc(u8, regex.parse.max_pattern_len + 1);
    defer std.testing.allocator.free(long);
    @memset(long, 'a');
    try std.testing.expectError(error.RegexTooLarge, Regex.compile(std.testing.allocator, long, .{}, null));
}

test "regex matching is linear: no blow-up on patterns that backtrack exponentially" {
    const re = try compileOpts("(a*)*b", .{});
    defer re.deinit();
    var text: [5000]u8 = undefined;
    @memset(&text, 'a');
    try std.testing.expect(!try re.isMatch(&text));
    try std.testing.expect(try re.find(&text, 0) == null);
    const re2 = try compileOpts("(x+x+)+y", .{});
    defer re2.deinit();
    @memset(text[0..3000], 'x');
    try std.testing.expect(!try re2.isMatch(text[0..3000]));
    // A DFA that would need exponentially many states falls back to the Pike VM.
    const re3 = try compileOpts("[ab]*a[ab]{20}", .{});
    defer re3.deinit();
    var buf: [4000]u8 = undefined;
    var rng = std.Random.DefaultPrng.init(7);
    for (&buf) |*c| c.* = if (rng.random().boolean()) 'a' else 'b';
    for (0..8) |_| {
        const span = (try re3.find(&buf, 0)).?;
        try std.testing.expectEqual(@as(usize, 0), span.start);
    }
}

test "regex case folding orbits equal RE2's (pyarrow 26, tools/regex_fold_check.py)" {
    const allocator = std.testing.allocator;
    // Members of each fold key other than the key itself.
    var members: std.AutoHashMapUnmanaged(u21, u32) = .empty;
    defer members.deinit(allocator);
    var c: u21 = 0;
    while (c <= 0x10FFFF) : (c += 1) {
        if (c >= 0xD800 and c <= 0xDFFF) continue;
        const k = regex.parse.foldKey(c);
        if (k == c) continue;
        try std.testing.expectEqual(k, regex.parse.foldKey(k));
        const gop = try members.getOrPut(allocator, k);
        if (!gop.found_existing) gop.value_ptr.* = 0;
        gop.value_ptr.* += 1;
    }
    var orbits: usize = 0;
    var bad: usize = 0;
    var lines = std.mem.splitScalar(u8, @embedFile("testdata/fold_orbits.txt"), '\n');
    while (lines.next()) |line| {
        if (line.len == 0 or line[0] == '#') continue;
        orbits += 1;
        var cps = std.mem.splitScalar(u8, line, ' ');
        var key: ?u21 = null;
        var n: u32 = 0;
        var same = true;
        while (cps.next()) |hex| {
            const k = regex.parse.foldKey(try std.fmt.parseInt(u21, hex, 16));
            if (key) |k0| same = same and k0 == k else key = k;
            n += 1;
        }
        if (!same or n - 1 != (members.get(key.?) orelse 0)) {
            bad += 1;
            std.debug.print("orbit {s} differs\n", .{line});
        }
    }
    try std.testing.expectEqual(@as(usize, 0), bad);
    // Every RE2 orbit is one of ours. Ours also folds the case pairs added in Unicode 16.0 /
    // 17.0 (Garay, Beria Erfe and a few Latin / Cyrillic letters), which pyarrow 26's RE2
    // tables predate; those are the only extra keys.
    var known: std.AutoHashMapUnmanaged(u21, void) = .empty;
    defer known.deinit(allocator);
    var again = std.mem.splitScalar(u8, @embedFile("testdata/fold_orbits.txt"), '\n');
    while (again.next()) |line| {
        if (line.len == 0 or line[0] == '#') continue;
        var cps = std.mem.splitScalar(u8, line, ' ');
        try known.put(allocator, regex.parse.foldKey(try std.fmt.parseInt(u21, cps.next().?, 16)), {});
    }
    var extra: usize = 0;
    var it = members.keyIterator();
    while (it.next()) |k| {
        if (known.contains(k.*)) continue;
        extra += 1;
        const newer = switch (k.*) {
            0x10D70...0x10D85, 0x16EBB...0x16ED3, 0x19B, 0x1C8A, 0x264, 0xA7CD, 0xA7CF, 0xA7D3, 0xA7D5, 0xA7DB => true,
            else => false,
        };
        if (!newer) std.debug.print("fold key {x} unknown to RE2\n", .{k.*});
        try std.testing.expect(newer);
    }
    try std.testing.expectEqual(@as(usize, 55), extra);
}
