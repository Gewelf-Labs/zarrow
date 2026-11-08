//! Parser of the RE2 regular expression syntax (the subset Arrow C++ exposes through RE2) into
//! an AST of character classes, assertions, groups, concatenations, alternations and bounded
//! repetitions. Literals are one-code-point classes; `(?i)` folds every class through the simple
//! case folding orbits (`foldClass`). Backreferences and lookaround are rejected (`RegexUnsupported`),
//! as RE2 does. The parser is bounded: pattern length, group nesting depth, repetition counts and
//! the total number of class ranges are capped; recursion depth is the group nesting depth.
const std = @import("std");
const string = @import("../compute/string.zig");
const unicode = @import("../compute/unicode_tables.zig");
const props = @import("../compute/unicode_props.zig");

pub const Error = error{ OutOfMemory, RegexSyntax, RegexUnsupported, RegexTooLarge };

/// Message and byte offset of a compile error (static strings).
pub const Diagnostic = struct {
    message: []const u8 = "",
    offset: usize = 0,
};

/// Longest accepted pattern in bytes.
pub const max_pattern_len = 64 * 1024;
/// Deepest group nesting (RE2 allows 1000; the parser recurses once per level).
pub const max_depth = 200;
/// Largest count of `{n,m}` and of the product of nested counts (RE2's limits).
pub const max_repeat = 1000;
/// Most class ranges in one pattern (a Unicode class has a few hundred).
pub const max_ranges = 200_000;

/// An inclusive code point range (bytes in Latin-1 mode).
pub const Range = struct { lo: u21, hi: u21 };

/// Empty-width assertions. `\b` / `\B` use ASCII word characters, as RE2.
pub const Assert = enum(u8) { begin_text, end_text, begin_line, end_line, word_boundary, not_word_boundary };

pub const Span32 = struct { start: u32, len: u32 };

pub const Node = union(enum) {
    empty,
    /// Code point ranges `Ast.ranges[start..][0..len]`, sorted and disjoint; empty never matches.
    class: Span32,
    /// `\C`: any byte.
    any_byte,
    assert: Assert,
    /// Capture group `index` (1-based).
    capture: struct { child: u32, index: u32 },
    /// Children `Ast.kids[start..][0..len]`.
    concat: Span32,
    alt: Span32,
    /// `max == inf` for no upper bound.
    repeat: struct { child: u32, min: u32, max: u32, greedy: bool },

    pub const inf = std.math.maxInt(u32);
};

pub const Ast = struct {
    nodes: []const Node,
    ranges: []const Range,
    kids: []const u32,
    root: u32,
    /// Name of capture group `i + 1`, null when unnamed.
    names: []const ?[]const u8,
    latin1: bool,

    pub fn classRanges(self: Ast, s: Span32) []const Range {
        return self.ranges[s.start..][0..s.len];
    }

    pub fn children(self: Ast, s: Span32) []const u32 {
        return self.kids[s.start..][0..s.len];
    }
};

pub const Options = struct {
    /// Case-insensitive from the start (`ignore_case`); `(?i)` sets it inside the pattern.
    case_insensitive: bool = false,
    /// The pattern is literal text (RE2 `literal`).
    literal: bool = false,
    /// Bytes are characters (RE2 Latin-1 encoding, Arrow's binary types); UTF-8 otherwise.
    latin1: bool = false,
};

const Flags = struct {
    i: bool = false,
    m: bool = false,
    s: bool = false,
    U: bool = false,
};

/// Parse `pattern` into an AST allocated in `arena`.
pub fn parse(arena: std.mem.Allocator, pattern: []const u8, options: Options, diag: ?*Diagnostic) Error!Ast {
    var p: Parser = .{ .arena = arena, .src = pattern, .latin1 = options.latin1, .diag = diag, .flags = .{ .i = options.case_insensitive } };
    if (pattern.len > max_pattern_len) return p.fail(error.RegexTooLarge, "pattern too long", 0);
    if (!options.latin1 and !std.unicode.utf8ValidateSlice(pattern)) return p.fail(error.RegexSyntax, "invalid UTF-8", 0);
    const root = if (options.literal) try p.parseLiteralText(pattern.len) else blk: {
        const r = try p.parseAlt(0);
        if (p.pos < p.src.len) return p.fail(error.RegexSyntax, "unexpected )", p.pos);
        break :blk r;
    };
    try p.checkRepeatProduct();
    return .{
        .nodes = p.nodes.items,
        .ranges = p.ranges.items,
        .kids = p.kids.items,
        .root = root,
        .names = p.names.items,
        .latin1 = p.latin1,
    };
}

const Parser = struct {
    arena: std.mem.Allocator,
    src: []const u8,
    pos: usize = 0,
    latin1: bool,
    diag: ?*Diagnostic,
    flags: Flags,
    nodes: std.ArrayList(Node) = .empty,
    ranges: std.ArrayList(Range) = .empty,
    kids: std.ArrayList(u32) = .empty,
    names: std.ArrayList(?[]const u8) = .empty,
    /// Case folding pairs, built on the first case-insensitive class.
    fold: ?[]const FoldPair = null,

    fn fail(p: *Parser, err: Error, message: []const u8, offset: usize) Error {
        if (p.diag) |d| d.* = .{ .message = message, .offset = offset };
        return err;
    }

    fn maxCp(p: *const Parser) u21 {
        return if (p.latin1) 0xFF else 0x10FFFF;
    }

    fn addNode(p: *Parser, n: Node) Error!u32 {
        const id: u32 = @intCast(p.nodes.items.len);
        try p.nodes.append(p.arena, n);
        return id;
    }

    fn addList(p: *Parser, items: []const u32, comptime tag: std.meta.Tag(Node)) Error!u32 {
        if (items.len == 0) return p.addNode(.empty);
        if (items.len == 1) return items[0];
        const start: u32 = @intCast(p.kids.items.len);
        try p.kids.appendSlice(p.arena, items);
        return p.addNode(@unionInit(Node, @tagName(tag), .{ .start = start, .len = @intCast(items.len) }));
    }

    /// Add a class node of `rs` (any order, may overlap), folded when `(?i)` and complemented
    /// when `negate`.
    fn addClass(p: *Parser, rs: *std.ArrayList(Range), negate: bool) Error!u32 {
        try p.normalize(rs);
        if (p.flags.i) {
            try p.foldClass(rs);
            try p.normalize(rs);
        }
        if (negate) try p.complement(rs);
        // Latin-1: only bytes can match.
        var n: usize = 0;
        for (rs.items) |r| {
            if (r.lo > p.maxCp()) break;
            rs.items[n] = .{ .lo = r.lo, .hi = @min(r.hi, p.maxCp()) };
            n += 1;
        }
        rs.shrinkRetainingCapacity(n);
        if (p.ranges.items.len + rs.items.len > max_ranges) return p.fail(error.RegexTooLarge, "pattern too large: too many class ranges", p.pos);
        const start: u32 = @intCast(p.ranges.items.len);
        try p.ranges.appendSlice(p.arena, rs.items);
        return p.addNode(.{ .class = .{ .start = start, .len = @intCast(rs.items.len) } });
    }

    fn addChar(p: *Parser, c: u21) Error!u32 {
        var rs: std.ArrayList(Range) = .empty;
        try rs.append(p.arena, .{ .lo = c, .hi = c });
        return p.addClass(&rs, false);
    }

    fn normalize(p: *Parser, rs: *std.ArrayList(Range)) Error!void {
        _ = p;
        std.sort.pdq(Range, rs.items, {}, struct {
            fn lt(_: void, a: Range, b: Range) bool {
                return a.lo < b.lo;
            }
        }.lt);
        var n: usize = 0;
        for (rs.items) |r| {
            if (n > 0 and @as(u32, r.lo) <= @as(u32, rs.items[n - 1].hi) + 1) {
                rs.items[n - 1].hi = @max(rs.items[n - 1].hi, r.hi);
            } else {
                rs.items[n] = r;
                n += 1;
            }
        }
        rs.shrinkRetainingCapacity(n);
    }

    /// Complement of normalized ranges within [0, 0x10FFFF] (clipped to Latin-1 later).
    fn complement(p: *Parser, rs: *std.ArrayList(Range)) Error!void {
        var out: std.ArrayList(Range) = .empty;
        var next: u32 = 0;
        for (rs.items) |r| {
            if (r.lo > next) try out.append(p.arena, .{ .lo = @intCast(next), .hi = r.lo - 1 });
            next = @as(u32, r.hi) + 1;
        }
        if (next <= 0x10FFFF) try out.append(p.arena, .{ .lo = @intCast(next), .hi = 0x10FFFF });
        rs.* = out;
    }

    fn foldClass(p: *Parser, rs: *std.ArrayList(Range)) Error!void {
        if (p.latin1) {
            // RE2 folds only the ASCII letters in Latin-1 mode (checked with pyarrow 26).
            const base = rs.items.len;
            for (0..base) |k| {
                const r = rs.items[k];
                for ([_][2]u21{ .{ 'a', 'z' }, .{ 'A', 'Z' } }) |l| {
                    const lo = @max(r.lo, l[0]);
                    const hi = @min(r.hi, l[1]);
                    if (lo <= hi) try rs.append(p.arena, .{ .lo = lo ^ 0x20, .hi = hi ^ 0x20 });
                }
            }
            return;
        }
        if (p.fold == null) p.fold = try buildFoldPairs(p.arena);
        const pairs = p.fold.?;
        const base = rs.items.len;
        var i: usize = 0;
        while (i < pairs.len) {
            var j = i;
            var hit = contains(rs.items[0..base], pairs[i].key);
            while (j < pairs.len and pairs[j].key == pairs[i].key) : (j += 1) {
                if (!hit and contains(rs.items[0..base], pairs[j].cp)) hit = true;
            }
            if (hit) {
                try rs.append(p.arena, .{ .lo = pairs[i].key, .hi = pairs[i].key });
                for (pairs[i..j]) |q| try rs.append(p.arena, .{ .lo = q.cp, .hi = q.cp });
            }
            i = j;
        }
    }

    fn eof(p: *const Parser) bool {
        return p.pos >= p.src.len;
    }

    fn peek(p: *const Parser) u8 {
        return p.src[p.pos];
    }

    fn lookingAt(p: *const Parser, s: []const u8) bool {
        return std.mem.startsWith(u8, p.src[p.pos..], s);
    }

    /// Next character (a byte in Latin-1 mode, a code point otherwise; the pattern is valid UTF-8).
    fn nextChar(p: *Parser) u21 {
        if (p.latin1) {
            p.pos += 1;
            return p.src[p.pos - 1];
        }
        const d = string.decodeStrict(p.src, p.pos).?;
        p.pos += d.len;
        return d.cp;
    }

    fn parseLiteralText(p: *Parser, end: usize) Error!u32 {
        var items: std.ArrayList(u32) = .empty;
        while (p.pos < end) try items.append(p.arena, try p.addChar(p.nextChar()));
        return p.addList(items.items, .concat);
    }

    fn parseAlt(p: *Parser, depth: u32) Error!u32 {
        if (depth > max_depth) return p.fail(error.RegexTooLarge, "pattern too deeply nested", p.pos);
        var items: std.ArrayList(u32) = .empty;
        try items.append(p.arena, try p.parseConcat(depth));
        while (!p.eof() and p.peek() == '|') {
            p.pos += 1;
            try items.append(p.arena, try p.parseConcat(depth));
        }
        return p.addList(items.items, .alt);
    }

    fn parseConcat(p: *Parser, depth: u32) Error!u32 {
        var items: std.ArrayList(u32) = .empty;
        while (!p.eof() and p.peek() != '|' and p.peek() != ')') {
            if (p.lookingAt("\\Q")) {
                // \Q...\E: literal characters; a repetition applies to the last one (RE2).
                p.pos += 2;
                const end = std.mem.indexOfPos(u8, p.src, p.pos, "\\E") orelse p.src.len;
                while (p.pos < end) {
                    const c = try p.addChar(p.nextChar());
                    if (p.pos < end) {
                        try items.append(p.arena, c);
                        continue;
                    }
                    p.pos = @min(end + 2, p.src.len);
                    try items.append(p.arena, try p.parseRepeat(c));
                    break;
                }
                p.pos = @max(p.pos, @min(end + 2, p.src.len));
                continue;
            }
            const atom = try p.parseAtom(depth) orelse continue;
            try items.append(p.arena, try p.parseRepeat(atom));
        }
        return p.addList(items.items, .concat);
    }

    /// `{n}`, `{n,}` or `{n,m}` at the cursor: the bounds and the length; null when not one
    /// (then `{` is a literal, as in RE2).
    fn repeatBounds(p: *Parser) Error!?struct { min: u32, max: u32, len: usize } {
        const s = p.src[p.pos..];
        if (s.len == 0 or s[0] != '{') return null;
        var i: usize = 1;
        const min = number(s, &i) orelse return null;
        var max = min;
        if (i < s.len and s[i] == ',') {
            i += 1;
            max = if (i < s.len and s[i] == '}') Node.inf else number(s, &i) orelse return null;
        }
        if (i >= s.len or s[i] != '}') return null;
        i += 1;
        if (min > max_repeat or (max != Node.inf and (max > max_repeat or max < min))) return p.fail(error.RegexSyntax, "invalid repetition size", p.pos);
        return .{ .min = min, .max = max, .len = i };
    }

    fn number(s: []const u8, i: *usize) ?u32 {
        const start = i.*;
        var v: u32 = 0;
        while (i.* < s.len and std.ascii.isDigit(s[i.*])) : (i.* += 1) v = @min(v *| 10 +| (s[i.*] - '0'), 100_000);
        return if (i.* == start) null else v;
    }

    fn parseRepeat(p: *Parser, atom: u32) Error!u32 {
        var min: u32 = 0;
        var max: u32 = 0;
        const start = p.pos;
        if (p.eof()) return atom;
        switch (p.peek()) {
            '*' => {
                max = Node.inf;
                p.pos += 1;
            },
            '+' => {
                min = 1;
                max = Node.inf;
                p.pos += 1;
            },
            '?' => {
                max = 1;
                p.pos += 1;
            },
            '{' => {
                const b = try p.repeatBounds() orelse return atom;
                min = b.min;
                max = b.max;
                p.pos += b.len;
            },
            else => return atom,
        }
        var greedy = true;
        if (!p.eof() and p.peek() == '?') {
            greedy = false;
            p.pos += 1;
        }
        if (p.flags.U) greedy = !greedy;
        // RE2 (Perl mode) rejects stacked repetition operators: a**, a+?+, a{2}*.
        if (!p.eof()) switch (p.peek()) {
            '*', '+', '?' => return p.fail(error.RegexSyntax, "bad repetition operator", start),
            '{' => if (try p.repeatBounds() != null) return p.fail(error.RegexSyntax, "bad repetition operator", start),
            else => {},
        };
        return p.addNode(.{ .repeat = .{ .child = atom, .min = min, .max = max, .greedy = greedy } });
    }

    /// The next atom; null for a flag group `(?i)` that only changes flags.
    fn parseAtom(p: *Parser, depth: u32) Error!?u32 {
        const start = p.pos;
        const c = p.peek();
        switch (c) {
            '(' => return p.parseGroup(depth),
            '[' => return try p.parseClass(),
            '*', '+', '?' => return p.fail(error.RegexSyntax, "missing argument to repetition operator", start),
            '{' => {
                if (try p.repeatBounds() != null) return p.fail(error.RegexSyntax, "missing argument to repetition operator", start);
                p.pos += 1;
                return try p.addChar('{');
            },
            '.' => {
                p.pos += 1;
                var rs: std.ArrayList(Range) = .empty;
                if (p.flags.s) {
                    try rs.append(p.arena, .{ .lo = 0, .hi = 0x10FFFF });
                } else {
                    try rs.append(p.arena, .{ .lo = 0, .hi = '\n' - 1 });
                    try rs.append(p.arena, .{ .lo = '\n' + 1, .hi = 0x10FFFF });
                }
                // `.` is not case folded (it already holds every orbit).
                const saved = p.flags.i;
                p.flags.i = false;
                defer p.flags.i = saved;
                return try p.addClass(&rs, false);
            },
            '^' => {
                p.pos += 1;
                return try p.addNode(.{ .assert = if (p.flags.m) .begin_line else .begin_text });
            },
            '$' => {
                p.pos += 1;
                return try p.addNode(.{ .assert = if (p.flags.m) .end_line else .end_text });
            },
            '\\' => return p.parseEscapeAtom(),
            else => return try p.addChar(p.nextChar()),
        }
    }

    fn parseGroup(p: *Parser, depth: u32) Error!?u32 {
        const start = p.pos;
        p.pos += 1;
        var capture: ?u32 = null;
        const saved = p.flags;
        if (p.lookingAt("?")) {
            if (p.lookingAt("?=") or p.lookingAt("?!") or p.lookingAt("?<=") or p.lookingAt("?<!"))
                return p.fail(error.RegexUnsupported, "lookaround is not supported", start);
            if (p.lookingAt("?P=") or p.lookingAt("?P>"))
                return p.fail(error.RegexUnsupported, "backreferences are not supported", start);
            if (p.lookingAt("?P<") or p.lookingAt("?<")) {
                p.pos += if (p.lookingAt("?P<")) 3 else 2;
                const name_start = p.pos;
                while (!p.eof() and (std.ascii.isAlphanumeric(p.peek()) or p.peek() == '_')) p.pos += 1;
                if (p.eof() or p.peek() != '>' or p.pos == name_start) return p.fail(error.RegexSyntax, "invalid named capture group", start);
                const name = p.src[name_start..p.pos];
                p.pos += 1;
                // RE2 accepts a name twice (Arrow then has two struct fields of that name).
                try p.names.append(p.arena, name);
                capture = @intCast(p.names.items.len);
            } else {
                // Flags: (?imsU-imsU) or (?imsU-imsU:re).
                p.pos += 1;
                var negate = false;
                var any = false;
                var after_neg = false;
                while (true) {
                    if (p.eof()) return p.fail(error.RegexSyntax, "missing )", start);
                    const f = p.peek();
                    p.pos += 1;
                    switch (f) {
                        'i', 'm', 's', 'U' => {
                            const v = !negate;
                            switch (f) {
                                'i' => p.flags.i = v,
                                'm' => p.flags.m = v,
                                's' => p.flags.s = v,
                                else => p.flags.U = v,
                            }
                            any = true;
                            if (negate) after_neg = true;
                        },
                        '-' => {
                            if (negate) return p.fail(error.RegexSyntax, "invalid or unsupported Perl syntax", start);
                            negate = true;
                        },
                        ':', ')' => {
                            // `(?:` needs no flags; `(?)`, `(?-)` and `(?i-:` are errors.
                            const bare_group = f == ':' and !any and !negate;
                            if (!bare_group and (!any or (negate and !after_neg))) return p.fail(error.RegexSyntax, "invalid or unsupported Perl syntax", start);
                            if (f == ')') return null; // flags stay set to the end of the group
                            break;
                        },
                        else => return p.fail(error.RegexSyntax, "invalid or unsupported Perl syntax", start),
                    }
                }
            }
        } else {
            try p.names.append(p.arena, null);
            capture = @intCast(p.names.items.len);
        }
        const body = try p.parseAlt(depth + 1);
        if (p.eof() or p.peek() != ')') return p.fail(error.RegexSyntax, "missing )", start);
        p.pos += 1;
        p.flags = saved;
        if (capture) |index| return try p.addNode(.{ .capture = .{ .child = body, .index = index } });
        return body;
    }

    fn parseEscapeAtom(p: *Parser) Error!?u32 {
        const start = p.pos;
        if (p.pos + 1 >= p.src.len) return p.fail(error.RegexSyntax, "trailing \\", start);
        const c = p.src[p.pos + 1];
        switch (c) {
            'b', 'B', 'A', 'z' => {
                p.pos += 2;
                return try p.addNode(.{ .assert = switch (c) {
                    'b' => .word_boundary,
                    'B' => .not_word_boundary,
                    'A' => .begin_text,
                    else => .end_text,
                } });
            },
            'C' => {
                p.pos += 2;
                return try p.addNode(.any_byte);
            },
            'E' => {
                // A stray \E is ignored (RE2).
                p.pos += 2;
                return null;
            },
            else => {},
        }
        var rs: std.ArrayList(Range) = .empty;
        const neg = try p.parseClassEscape(&rs);
        return try p.addClass(&rs, neg);
    }

    /// An escape inside or outside a class: adds its ranges to `rs`; returns whether the whole
    /// set is negated (\D, \PL outside a class; inside a class negated sets are complemented
    /// before being added).
    fn parseClassEscape(p: *Parser, rs: *std.ArrayList(Range)) Error!bool {
        const start = p.pos;
        p.pos += 1; // backslash
        if (p.eof()) return p.fail(error.RegexSyntax, "trailing \\", start);
        const c = p.peek();
        switch (c) {
            'd', 'D', 's', 'S', 'w', 'W' => {
                p.pos += 1;
                switch (std.ascii.toLower(c)) {
                    'd' => try rs.append(p.arena, .{ .lo = '0', .hi = '9' }),
                    's' => for ([_]u21{ '\t', '\n', 0x0C, '\r', ' ' }) |x| try rs.append(p.arena, .{ .lo = x, .hi = x }),
                    else => try appendWord(p.arena, rs),
                }
                return std.ascii.isUpper(c);
            },
            'p', 'P' => {
                p.pos += 1;
                if (p.eof()) return p.fail(error.RegexSyntax, "invalid character class range", start);
                var name: []const u8 = undefined;
                if (p.peek() == '{') {
                    const end = std.mem.indexOfScalarPos(u8, p.src, p.pos, '}') orelse return p.fail(error.RegexSyntax, "invalid character class range", start);
                    name = p.src[p.pos + 1 .. end];
                    p.pos = end + 1;
                } else {
                    // One character name (a code point in UTF-8 mode).
                    const n0 = p.pos;
                    _ = p.nextChar();
                    name = p.src[n0..p.pos];
                }
                var neg = c == 'P';
                if (name.len > 0 and name[0] == '^') {
                    neg = !neg;
                    name = name[1..];
                }
                try p.appendUnicodeClass(rs, name, start);
                return neg;
            },
            else => {
                try rs.append(p.arena, .{ .lo = try p.parseEscapeChar(), .hi = 0 });
                rs.items[rs.items.len - 1].hi = rs.items[rs.items.len - 1].lo;
                return false;
            },
        }
    }

    /// A single-character escape after the backslash (cursor on the character).
    fn parseEscapeChar(p: *Parser) Error!u21 {
        const start = p.pos - 1;
        const c = p.peek();
        switch (c) {
            '1'...'7' => {
                // A single non-zero digit would be a backreference (unsupported, as RE2).
                if (p.pos + 1 >= p.src.len or p.src[p.pos + 1] < '0' or p.src[p.pos + 1] > '7') return p.fail(error.RegexUnsupported, "invalid escape sequence: backreferences are not supported", start);
                return p.octal();
            },
            '0' => return p.octal(),
            'x' => {
                p.pos += 1;
                if (p.eof()) return p.fail(error.RegexSyntax, "invalid escape sequence", start);
                if (p.peek() == '{') {
                    p.pos += 1;
                    var v: u32 = 0;
                    var n: usize = 0;
                    while (!p.eof() and std.ascii.isHex(p.peek())) : (n += 1) {
                        v = @min(v * 16 + (std.fmt.charToDigit(p.peek(), 16) catch unreachable), 0x110000);
                        p.pos += 1;
                    }
                    if (n == 0 or p.eof() or p.peek() != '}' or v > 0x10FFFF) return p.fail(error.RegexSyntax, "invalid escape sequence", start);
                    p.pos += 1;
                    return @intCast(v);
                }
                if (p.pos + 2 > p.src.len or !std.ascii.isHex(p.src[p.pos]) or !std.ascii.isHex(p.src[p.pos + 1])) return p.fail(error.RegexSyntax, "invalid escape sequence", start);
                const v = std.fmt.parseInt(u8, p.src[p.pos..][0..2], 16) catch unreachable;
                p.pos += 2;
                return v;
            },
            'a' => return p.simple(0x07),
            'f' => return p.simple(0x0C),
            't' => return p.simple('\t'),
            'n' => return p.simple('\n'),
            'r' => return p.simple('\r'),
            'v' => return p.simple(0x0B),
            else => {
                // Escaped punctuation is itself; word characters and non-ASCII are errors.
                if (c < 0x80 and !std.ascii.isAlphanumeric(c) and c != '_') {
                    p.pos += 1;
                    return c;
                }
                return p.fail(error.RegexSyntax, "invalid escape sequence", start);
            },
        }
    }

    fn simple(p: *Parser, v: u21) u21 {
        p.pos += 1;
        return v;
    }

    /// Up to three octal digits.
    fn octal(p: *Parser) u21 {
        var v: u21 = 0;
        var n: usize = 0;
        while (n < 3 and !p.eof() and p.peek() >= '0' and p.peek() <= '7') : (n += 1) {
            v = v * 8 + (p.peek() - '0');
            p.pos += 1;
        }
        return v;
    }

    fn appendUnicodeClass(p: *Parser, rs: *std.ArrayList(Range), name: []const u8, start: usize) Error!void {
        if (std.mem.eql(u8, name, "Any")) return rs.append(p.arena, .{ .lo = 0, .hi = 0x10FFFF });
        const mask = categoryMask(name) orelse {
            // A script (Scripts.txt of Unicode 17.0.0; "Unknown" is not a class, as RE2).
            for (props.script_names[1..], 1..) |script, index| {
                if (!std.mem.eql(u8, script, name)) continue;
                const runs = &props.script_runs;
                for (runs, 0..) |run, k| {
                    if (run & 0xFF != index) continue;
                    const lo: u21 = @intCast(run >> 8);
                    const hi: u21 = if (k + 1 < runs.len) @intCast((runs[k + 1] >> 8) - 1) else 0x10FFFF;
                    try rs.append(p.arena, .{ .lo = lo, .hi = hi });
                }
                return;
            }
            return p.fail(error.RegexSyntax, "invalid character class range", start);
        };
        const runs = &unicode.category_runs;
        for (runs, 0..) |run, k| {
            const cat: u5 = @truncate(run);
            if (mask & (@as(u32, 1) << cat) == 0) continue;
            const lo: u21 = @intCast(run >> 5);
            const hi: u21 = if (k + 1 < runs.len) @intCast((runs[k + 1] >> 5) - 1) else 0x10FFFF;
            try rs.append(p.arena, .{ .lo = lo, .hi = hi });
        }
    }

    fn parseClass(p: *Parser) Error!u32 {
        const start = p.pos;
        p.pos += 1;
        var rs: std.ArrayList(Range) = .empty;
        var negate = false;
        if (!p.eof() and p.peek() == '^') {
            negate = true;
            p.pos += 1;
        }
        var first = true;
        while (true) : (first = false) {
            if (p.eof()) return p.fail(error.RegexSyntax, "missing ]", start);
            const c = p.peek();
            if (c == ']' and !first) break;
            if (p.lookingAt("[:")) {
                if (try p.posixClass(&rs)) continue;
            }
            const item_start = p.pos;
            var lo: u21 = undefined;
            if (c == '\\') {
                var sub: std.ArrayList(Range) = .empty;
                const neg = try p.parseClassEscape(&sub);
                if (sub.items.len != 1 or sub.items[0].lo != sub.items[0].hi or neg) {
                    // A Perl or Unicode class inside the brackets.
                    if (neg) {
                        try p.normalize(&sub);
                        try p.complement(&sub);
                    }
                    try rs.appendSlice(p.arena, sub.items);
                    continue;
                }
                lo = sub.items[0].lo;
            } else {
                lo = p.nextChar();
            }
            var hi = lo;
            if (p.pos + 1 < p.src.len and p.peek() == '-' and p.src[p.pos + 1] != ']') {
                p.pos += 1;
                if (p.peek() == '\\') {
                    var sub: std.ArrayList(Range) = .empty;
                    const neg = try p.parseClassEscape(&sub);
                    if (neg or sub.items.len != 1 or sub.items[0].lo != sub.items[0].hi) return p.fail(error.RegexSyntax, "invalid character class range", item_start);
                    hi = sub.items[0].lo;
                } else {
                    hi = p.nextChar();
                }
                if (hi < lo) return p.fail(error.RegexSyntax, "invalid character class range", item_start);
            }
            try rs.append(p.arena, .{ .lo = lo, .hi = hi });
        }
        p.pos += 1;
        return p.addClass(&rs, negate);
    }

    /// `[:name:]` / `[:^name:]` (ASCII); false when the text is not one (then `[` is literal).
    fn posixClass(p: *Parser, rs: *std.ArrayList(Range)) Error!bool {
        const end = std.mem.indexOfPos(u8, p.src, p.pos + 2, ":]") orelse return false;
        var name = p.src[p.pos + 2 .. end];
        var neg = false;
        if (name.len > 0 and name[0] == '^') {
            neg = true;
            name = name[1..];
        }
        const table = [_]struct { []const u8, []const u8 }{
            .{ "alnum", "09AZaz" },   .{ "alpha", "AZaz" },             .{ "ascii", "\x00\x7f" },
            .{ "blank", "\t\t  " },   .{ "cntrl", "\x00\x1f\x7f\x7f" }, .{ "digit", "09" },
            .{ "graph", "!~" },       .{ "lower", "az" },               .{ "print", " ~" },
            .{ "punct", "!/:@[`{~" }, .{ "space", "\t\r  " },           .{ "upper", "AZ" },
            .{ "word", "09AZaz__" },  .{ "xdigit", "09AFaf" },
        };
        for (table) |entry| {
            if (!std.mem.eql(u8, entry[0], name)) continue;
            var sub: std.ArrayList(Range) = .empty;
            var k: usize = 0;
            while (k < entry[1].len) : (k += 2) try sub.append(p.arena, .{ .lo = entry[1][k], .hi = entry[1][k + 1] });
            if (neg) {
                try p.normalize(&sub);
                try p.complement(&sub);
            }
            try rs.appendSlice(p.arena, sub.items);
            p.pos = end + 2;
            return true;
        }
        return p.fail(error.RegexSyntax, "invalid character class range", p.pos);
    }

    /// RE2 rejects a pattern whose nested `{n,m}` counts multiply past 1000.
    fn checkRepeatProduct(p: *Parser) Error!void {
        const prod = try p.arena.alloc(u32, p.nodes.items.len);
        // Children are created before their parents.
        for (p.nodes.items, 0..) |n, i| {
            var inner: u32 = 1;
            switch (n) {
                .capture => |c| inner = prod[c.child],
                .repeat => |r| inner = prod[r.child],
                .concat, .alt => |s| for (p.kids.items[s.start..][0..s.len]) |k| {
                    inner = @max(inner, prod[k]);
                },
                else => {},
            }
            if (n == .repeat) {
                const r = n.repeat;
                const m = if (r.max == Node.inf) r.min else r.max;
                if (m > 0) inner *|= m;
                if (inner > max_repeat) return p.fail(error.RegexSyntax, "bad repetition operator", 0);
            }
            prod[i] = inner;
        }
    }
};

fn appendWord(arena: std.mem.Allocator, rs: *std.ArrayList(Range)) Error!void {
    for ([_]Range{ .{ .lo = '0', .hi = '9' }, .{ .lo = 'A', .hi = 'Z' }, .{ .lo = '_', .hi = '_' }, .{ .lo = 'a', .hi = 'z' } }) |r| try rs.append(arena, r);
}

fn contains(rs: []const Range, c: u21) bool {
    var lo: usize = 0;
    var hi: usize = rs.len;
    while (lo < hi) {
        const mid = (lo + hi) / 2;
        if (rs[mid].hi < c) lo = mid + 1 else if (rs[mid].lo > c) hi = mid else return true;
    }
    return false;
}

/// Bit mask over `unicode.Category` of a general category name (one or two letters), as RE2:
/// `C` is Cc Cf Co Cs (RE2 has no Cn table).
fn categoryMask(name: []const u8) ?u32 {
    const cats = comptime std.meta.tags(unicode.Category);
    if (name.len == 1) {
        var mask: u32 = 0;
        inline for (cats) |c| {
            if (@tagName(c)[0] == name[0] and c != .Cn) mask |= @as(u32, 1) << @intFromEnum(c);
        }
        return if (mask == 0) null else mask;
    }
    if (std.mem.eql(u8, name, "Cn")) return null;
    inline for (cats) |c| {
        if (std.mem.eql(u8, @tagName(c), name)) return @as(u32, 1) << @intFromEnum(c);
    }
    return null;
}

/// A code point and the key of its simple case folding orbit.
pub const FoldPair = struct { key: u21, cp: u21 };

/// Key of the case folding orbit of `c`: its simple lowercase of its simple uppercase, except
/// U+0130 and U+0131, which fold to nothing else (CaseFolding.txt has only Turkic entries for
/// them), and the three simple foldings that are no case mapping (U+1FD3, U+1FE3, U+FB05).
/// Checked against RE2's `(?i)` orbits from pyarrow 26 (`tools/regex_fold_check.py`).
pub fn foldKey(c: u21) u21 {
    return switch (c) {
        0x130, 0x131 => c,
        0x1FD3 => 0x390,
        0x1FE3 => 0x3B0,
        0xFB05 => 0xFB06,
        else => string.toLower(string.toUpper(c)),
    };
}

/// Every code point whose fold key is another code point, sorted by key.
fn buildFoldPairs(arena: std.mem.Allocator) Error![]const FoldPair {
    var out: std.ArrayList(FoldPair) = .empty;
    inline for (.{ &unicode.upper_ranges, &unicode.lower_ranges }) |table| {
        for (table) |r| {
            var c: u21 = r.first;
            while (c <= r.last) : (c += if (r.stride2 == 1) 2 else 1) {
                const k = foldKey(c);
                if (k != c) try out.append(arena, .{ .key = k, .cp = c });
            }
        }
    }
    for ([_]u21{ 0x1FD3, 0x1FE3, 0xFB05 }) |c| try out.append(arena, .{ .key = foldKey(c), .cp = c });
    std.sort.pdq(FoldPair, out.items, {}, struct {
        fn lt(_: void, a: FoldPair, b: FoldPair) bool {
            return if (a.key != b.key) a.key < b.key else a.cp < b.cp;
        }
    }.lt);
    // A code point in both tables appears twice.
    var n: usize = 0;
    for (out.items) |q| {
        if (n > 0 and out.items[n - 1].key == q.key and out.items[n - 1].cp == q.cp) continue;
        out.items[n] = q;
        n += 1;
    }
    return out.items[0..n];
}
