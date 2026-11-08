//! Linear-time regular expressions in Zig with RE2's syntax and leftmost-first semantics, as
//! Arrow C++ uses them (compute functions `*_regex`, `match_like`, case-insensitive literals).
//!
//! Syntax: literals, `.`, classes `[...]` (ranges, negation, `[:alpha:]` POSIX classes), Perl
//! classes `\d \s \w` and their negations (ASCII, as RE2), Unicode general categories `\pL`,
//! `\p{Lu}`, `\PL`, `\p{^L}`, scripts such as `\p{Greek}` (Unicode 17.0.0 tables),
//! anchors `^ $ \A \z`, word boundaries `\b \B` (ASCII), groups `( )`, `(?: )`, named groups
//! `(?P<name> )` / `(?<name> )`, alternation, greedy and lazy quantifiers `* + ? {n} {n,}
//! {n,m}` (n, m <= 1000), flags `(?imsU)` / `(?imsU-imsU: )`, escapes `\a \f \t \n \r \v \x41
//! \x{263a} \123 \Q...\E \C` and escaped punctuation. Backreferences and lookaround are rejected
//! (`RegexUnsupported`), as RE2 does. `(?i)` folds through simple case folding orbits.
//!
//! Text is UTF-8 (a character is a code point; invalid bytes match only `\C`) or, with
//! `latin1`, bytes (Arrow's binary types). Matching is O(len(text) * len(program)): a lazy DFA
//! finds whether and where a match is (forward for the end, a reverse program for the start)
//! and the Pike VM fills capture groups; neither backtracks. Compilation is bounded
//! (`parse.max_*`, `prog.max_*`), the DFA cache too (`dfa.max_*`).
const std = @import("std");
pub const parse = @import("parse.zig");
pub const prog = @import("prog.zig");
pub const dfa = @import("dfa.zig");

pub const Options = parse.Options;
pub const Diagnostic = parse.Diagnostic;
/// `RegexSyntax`: malformed pattern; `RegexUnsupported`: valid Perl syntax RE2 rejects
/// (backreferences, lookaround); `RegexTooLarge`: past a compile limit.
pub const CompileError = parse.Error;
pub const MatchError = error{OutOfMemory};

/// A match `[start, end)` in bytes.
pub const Span = struct { start: usize, end: usize };

/// Searches after which a DFA that keeps filling its cache is no longer tried.
const max_dfa_failures = 4;

pub const Regex = struct {
    allocator: std.mem.Allocator,
    forward: prog.Prog,
    backward: prog.Prog,
    /// The pattern as bytes when it is a plain case-sensitive literal (memmem fast path).
    literal: ?[]const u8,
    names: [][]const u8,
    named: []bool,
    fwd_dfa: dfa.Dfa,
    rev_dfa: dfa.Dfa,
    pike: prog.Pike,
    dfa_failures: u32 = 0,
    slots: []usize,
    /// The bytes that can start a match; null when the pattern can match empty.
    first_bytes: ?prog.ByteSet,

    /// Compile `pattern`; on error `diag` (when given) holds a message and the byte offset.
    pub fn compile(allocator: std.mem.Allocator, pattern: []const u8, options: Options, diag: ?*Diagnostic) CompileError!*Regex {
        var arena: std.heap.ArenaAllocator = .init(allocator);
        defer arena.deinit();
        const ast = try parse.parse(arena.allocator(), pattern, options, diag);

        const self = try allocator.create(Regex);
        errdefer allocator.destroy(self);
        self.* = undefined;
        self.allocator = allocator;
        self.forward = try prog.compile(allocator, ast, false, diag);
        errdefer prog.deinit(&self.forward, allocator);
        self.backward = try prog.compile(allocator, ast, true, diag);
        errdefer prog.deinit(&self.backward, allocator);
        self.literal = try literalOf(allocator, ast);
        var first: prog.ByteSet = undefined;
        self.first_bytes = if (try firstBytes(allocator, &self.forward, &first)) first else null;
        errdefer if (self.literal) |l| allocator.free(l);
        self.names = try allocator.alloc([]const u8, ast.names.len);
        @memset(self.names, "");
        errdefer {
            for (self.names) |n| allocator.free(n);
            allocator.free(self.names);
        }
        self.named = try allocator.alloc(bool, ast.names.len);
        errdefer allocator.free(self.named);
        for (ast.names, self.names, self.named) |n, *dst, *flag| {
            flag.* = n != null;
            dst.* = try allocator.dupe(u8, n orelse "");
        }
        self.slots = try allocator.alloc(usize, self.forward.nslots);
        errdefer allocator.free(self.slots);
        self.fwd_dfa = try dfa.Dfa.init(allocator, &self.forward, false);
        errdefer self.fwd_dfa.deinit();
        self.rev_dfa = try dfa.Dfa.init(allocator, &self.backward, true);
        errdefer self.rev_dfa.deinit();
        self.pike = .{ .allocator = allocator };
        self.dfa_failures = 0;
        return self;
    }

    pub fn deinit(self: *Regex) void {
        const a = self.allocator;
        self.pike.deinit();
        self.rev_dfa.deinit();
        self.fwd_dfa.deinit();
        a.free(self.slots);
        a.free(self.named);
        for (self.names) |n| a.free(n);
        a.free(self.names);
        if (self.literal) |l| a.free(l);
        prog.deinit(&self.backward, a);
        prog.deinit(&self.forward, a);
        a.destroy(self);
    }

    /// Number of capture groups (group 0, the whole match, not counted).
    pub fn groupCount(self: *const Regex) usize {
        return self.names.len;
    }

    /// Name of group `i` (1-based), null when unnamed.
    pub fn groupName(self: *const Regex, i: usize) ?[]const u8 {
        return if (self.named[i - 1]) self.names[i - 1] else null;
    }

    /// Whether `text` has a match anywhere.
    pub fn isMatch(self: *Regex, text: []const u8) MatchError!bool {
        if (self.literal) |l| return indexOfLiteral(text, 0, l) != null;
        const from = self.firstCandidate(text, 0) orelse return false;
        if (self.dfa_failures < max_dfa_failures) {
            if (self.fwd_dfa.searchForward(text, from, false, true)) |end| return end != null else |err| try self.dfaFailed(err);
        }
        return self.pike.search(&self.forward, text, from, false, self.slots[0..0]);
    }

    /// The first position at or after `start` where a match can begin (null: none), from the
    /// bytes that can start a match of a pattern that cannot match empty.
    inline fn firstCandidate(self: *const Regex, text: []const u8, start: usize) ?usize {
        const set = if (self.first_bytes) |*f| f else return start;
        for (text[start..], start..) |b, i| {
            if (set[b]) return i;
        }
        return null;
    }

    fn dfaFailed(self: *Regex, err: dfa.Error) MatchError!void {
        if (err == error.OutOfMemory) return error.OutOfMemory;
        self.dfa_failures += 1;
    }

    /// The leftmost-first match starting at or after `start`; the whole `text` is the context
    /// of `^`, `$` and `\b`.
    pub fn find(self: *Regex, text: []const u8, start: usize) MatchError!?Span {
        std.debug.assert(start <= text.len);
        if (self.literal) |l| {
            const at = indexOfLiteral(text, start, l) orelse return null;
            return .{ .start = at, .end = at + l.len };
        }
        const from = self.firstCandidate(text, start) orelse return null;
        if (self.dfa_failures < max_dfa_failures) dfa_path: {
            // No match starts before `from`: a match anchored there is the leftmost-first one,
            // found in one scan (the usual case once `firstCandidate` skipped ahead).
            if (self.first_bytes != null) {
                const anchored_end = self.fwd_dfa.searchForward(text, from, true, false) catch |err| {
                    try self.dfaFailed(err);
                    break :dfa_path;
                };
                if (anchored_end) |e| return .{ .start = from, .end = e };
            }
            const end = (self.fwd_dfa.searchForward(text, from, false, false) catch |err| {
                try self.dfaFailed(err);
                break :dfa_path;
            }) orelse return null;
            const begin = self.rev_dfa.searchReverse(text, end, from) catch |err| {
                try self.dfaFailed(err);
                break :dfa_path;
            };
            // The reverse program matches every forward match backwards.
            return .{ .start = begin.?, .end = end };
        }
        const s = self.slots[0..2];
        if (!try self.pike.search(&self.forward, text, from, false, s)) return null;
        return .{ .start = s[0], .end = s[1] };
    }

    /// The leftmost-first match at or after `start` with its groups: `groups[i]` is group `i`
    /// (0: the whole match), null when the group did not take part. `groups.len` must be
    /// `groupCount() + 1`. Returns false when there is no match.
    pub fn captures(self: *Regex, text: []const u8, start: usize, groups: []?Span) MatchError!bool {
        std.debug.assert(groups.len == self.groupCount() + 1);
        const span = try self.find(text, start) orelse return false;
        if (groups.len == 1) {
            groups[0] = span;
            return true;
        }
        // Anchored at the match start, the Pike VM takes the same leftmost-first path.
        const ok = try self.pike.search(&self.forward, text, span.start, true, self.slots);
        std.debug.assert(ok);
        for (groups, 0..) |*g, i| {
            const a = self.slots[2 * i];
            const b = self.slots[2 * i + 1];
            g.* = if (a == prog.Pike.none or b == prog.Pike.none) null else .{ .start = a, .end = b };
        }
        return true;
    }
};

/// `needle` in `hay` at or after `start`: a scan for the first byte, then the rest compared
/// (rows are short; no skip table to build per row). O(len(hay) * len(needle)).
fn indexOfLiteral(hay: []const u8, start: usize, needle: []const u8) ?usize {
    if (needle.len == 0) return start;
    if (hay.len < needle.len) return null;
    const last = hay.len - needle.len;
    var i = start;
    while (i <= last) {
        i = std.mem.indexOfScalarPos(u8, hay[0 .. last + 1], i, needle[0]) orelse return null;
        if (std.mem.eql(u8, hay[i + 1 ..][0 .. needle.len - 1], needle[1..])) return i;
        i += 1;
    }
    return null;
}

/// Fill `set` with the bytes a match can start with and return true, or return false when the
/// pattern can match empty (assertions are taken as passing: a superset, still a sound filter).
fn firstBytes(allocator: std.mem.Allocator, p: *const prog.Prog, set: *prog.ByteSet) error{OutOfMemory}!bool {
    set.* = @splat(false);
    const seen = try allocator.alloc(bool, p.insts.len);
    defer allocator.free(seen);
    @memset(seen, false);
    var stack: std.ArrayList(u32) = .empty;
    defer stack.deinit(allocator);
    try stack.append(allocator, p.start);
    while (stack.pop()) |pc| {
        if (seen[pc]) continue;
        seen[pc] = true;
        const inst = p.insts[pc];
        switch (inst.op) {
            .match => return false,
            .byte => @memset(set[inst.lo .. @as(usize, inst.hi) + 1], true),
            .set => for (set, p.sets[inst.arg]) |*d, x| {
                d.* = d.* or x;
            },
            .split => {
                try stack.append(allocator, inst.arg);
                try stack.append(allocator, inst.out);
            },
            .save, .assert, .nop => try stack.append(allocator, inst.out),
            .fail => {},
        }
    }
    return true;
}

/// The bytes of a pattern made only of single code points (no class, group or assertion).
fn literalOf(allocator: std.mem.Allocator, ast: parse.Ast) error{OutOfMemory}!?[]const u8 {
    const root = ast.nodes[ast.root];
    const items: []const u32 = switch (root) {
        .concat => |s| ast.children(s),
        .class, .empty => &.{ast.root},
        else => return null,
    };
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    for (items) |id| switch (ast.nodes[id]) {
        .empty => {},
        .class => |s| {
            const rs = ast.classRanges(s);
            if (rs.len != 1 or rs[0].lo != rs[0].hi) {
                out.deinit(allocator);
                return null;
            }
            var buf: [4]u8 = undefined;
            const n = if (ast.latin1) blk: {
                buf[0] = @intCast(rs[0].lo);
                break :blk 1;
            } else std.unicode.utf8Encode(rs[0].lo, &buf) catch unreachable;
            try out.appendSlice(allocator, buf[0..n]);
        },
        else => {
            out.deinit(allocator);
            return null;
        },
    };
    return try out.toOwnedSlice(allocator);
}

test {
    _ = @import("test_regex.zig");
}
