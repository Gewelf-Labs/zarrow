//! Regular expression kernels over the zarrow regex engine (`src/regex/`, RE2 syntax and
//! leftmost-first semantics, linear time): match_substring_regex, match_like,
//! count_substring_regex, find_substring_regex, replace_substring_regex, split_pattern_regex,
//! extract_regex and extract_regex_span. Also the case-insensitive forms of the literal
//! functions (match_substring, starts_with, ends_with, count_substring, find_substring with
//! `ignore_case`), which Arrow runs through RE2 as well (`literalMatcher`).
//!
//! Semantics follow Arrow C++ 26 / RE2, checked against pyarrow 26 by a generated corpus
//! (`tools/regex_corpus.py`, `src/compute/test_string_regex.zig`):
//! - The string layouts are UTF-8 (a character is a code point), the binary layouts Latin-1 (a
//!   character is a byte). A match can still start or end inside a code point (`\C`, or an
//!   empty `\B` / `x*` match between two bytes of one, as RE2 searches byte by byte): a valid
//!   string row whose replace / split result would not be UTF-8 is `InvalidInput` (Arrow emits
//!   the invalid bytes).
//! - count_substring_regex and the `max_replacements` path of replace_substring_regex consume
//!   the row as RE2's `FindAndConsume` does: each search sees only the rest of the row, so `^`
//!   and `\b` can match again after a match; an empty match counts and advances one byte.
//! - replace_substring_regex without `max_replacements` (or on an empty row) is RE2's
//!   `GlobalReplace`: `\0`..`\9` and `\\` in the replacement, an empty match right after the
//!   previous match is skipped (one code point in UTF-8 mode). With `max_replacements` each match
//!   is replaced by matching the pattern again inside it (RE2 `Replace`, Arrow); when that fails
//!   (`\Ba` on the isolated match) the result is `InvalidInput`, as Arrow's "replacement failed".
//! - split_pattern_regex: `reverse` is `NotImplemented` (as Arrow). Arrow loops forever on an
//!   empty separator without `max_splits`; here that row is `InvalidInput`. A separator that cuts
//!   a code point of a string row is `InvalidInput` (Arrow would emit invalid UTF-8).
//! - extract_regex: every group must be named (`InvalidOptions` otherwise); a row without a
//!   match is a null struct row, a group that did not take part is "" (Arrow). extract_regex_span
//!   gives `fixed_size_list<int32, 2>` [start, length] per group (int64 for the large layouts),
//!   null for a group that did not take part.
//! - match_like translates the LIKE pattern as Arrow (`%` any run, `_` one character, `\`
//!   escapes the next one; case-sensitive patterns of the forms `%x%`, `x%`, `%x` are plain
//!   substring / prefix / suffix tests, as Arrow, so a backslash there is literal).
//! Layouts: string, large_string, binary, large_binary as Arrow, plus string_view and binary_view
//! (not registered by Arrow); count_substring_regex and find_substring_regex also take
//! fixed_size_binary (as Arrow, int32 results). Null handling is intersection. Every input chunk passes
//! `validateLayout`; a pattern that does not compile is `InvalidOptions`.
const std = @import("std");
const bitmap = @import("../bitmap.zig");
const core = @import("core.zig");
const datatype = @import("../datatype.zig");
const dense = @import("dense.zig");
const regex = @import("../regex/regex.zig");
const string = @import("string.zig");
const string_transform = @import("string_transform.zig");

const ArrayData = @import("../array/array_data.zig").ArrayData;
const ArrayRef = core.ArrayRef;
const DataType = core.DataType;
const Datum = core.Datum;
const ExecChunkValue = core.ExecChunkValue;
const ExecContext = core.ExecContext;
const FunctionRegistry = core.FunctionRegistry;
const KernelError = core.KernelError;
const Options = core.Options;
const OwnedBuffer = @import("../buffer.zig").OwnedBuffer;
const SharedBuffer = @import("../buffer.zig").SharedBuffer;
const Regex = regex.Regex;
const TypeTag = std.meta.Tag(DataType);

const all_tags = [_]TypeTag{ .string, .large_string, .string_view, .binary, .large_binary, .binary_view };
const bool_type = DataType{ .bool = {} };

pub fn register(registry: *FunctionRegistry) KernelError!void {
    inline for (all_tags) |tag| {
        try registry.registerScalarKernel("match_substring_regex", unary(tag, matchOptions, dense.resultTypeOf(bool_type), Match(.regex).exec));
        try registry.registerScalarKernel("match_like", unary(tag, matchOptions, dense.resultTypeOf(bool_type), Match(.like).exec));
        try registry.registerScalarKernel("count_substring_regex", unary(tag, matchOptions, dense.resultTypeOf(indexType(tag)), Search(.count).exec));
        try registry.registerScalarKernel("find_substring_regex", unary(tag, matchOptions, dense.resultTypeOf(indexType(tag)), Search(.find).exec));
        try registry.registerScalarKernel("replace_substring_regex", unary(tag, replaceOptions, dense.resultTypeOf(@unionInit(DataType, @tagName(tag), {})), replaceExec));
        try registry.registerScalarKernel("split_pattern_regex", unary(tag, splitOptions, dense.resultTypeOf(string_transform.listOf(tag)), splitExec));
        inline for (.{ Extract.values, Extract.spans }) |which| {
            try registry.registerScalarKernel(if (which == .values) "extract_regex" else "extract_regex_span", .{
                .signature = .{ .arity = 1, .type_check = dense.argsOfType(tag, 1), .options_check = extractOptions, .result_type_alloc_fn = Extractor(which).resultType },
                .exec = Extractor(which).exec,
            });
        }
    }
    // fixed_size_binary rows as Latin-1 binary rows (Arrow), int32 results.
    inline for (.{ "count_substring_regex", "find_substring_regex" }, .{ Search(.count), Search(.find) }) |name, S| {
        try registry.registerScalarKernel(name, .{
            .signature = .{ .arity = 1, .type_check = dense.argsOfType(.fixed_size_binary, 1), .options_check = matchOptions, .result_type_fn = dense.resultTypeOf(.{ .int32 = {} }) },
            .exec = string_transform.FixedVia(S.chunk, false).exec,
        });
    }
}

fn unary(comptime tag: TypeTag, comptime options_check: core.OptionsCheckFn, comptime result: core.ResultTypeFn, comptime exec: core.KernelExecFn) core.Kernel {
    return .{
        .signature = .{ .arity = 1, .type_check = dense.argsOfType(tag, 1), .options_check = options_check, .result_type_fn = result },
        .exec = exec,
    };
}

/// int64 for the large layouts, int32 otherwise (Arrow C++).
fn indexType(comptime tag: TypeTag) DataType {
    return if (tag == .large_string or tag == .large_binary) .{ .int64 = {} } else .{ .int32 = {} };
}

fn isText(dt: DataType) bool {
    return switch (datatype.storageDataType(dt)) {
        .string, .large_string, .string_view => true,
        else => false,
    };
}

fn matchOptions(options: Options) bool {
    return options == .match_substring;
}

fn replaceOptions(options: Options) bool {
    return options == .replace_substring;
}

fn splitOptions(options: Options) bool {
    return options == .split_pattern;
}

fn extractOptions(options: Options) bool {
    return options == .extract_regex;
}

pub const CompileFlags = struct {
    ignore_case: bool = false,
    literal: bool = false,
};

/// Compile `pattern` for rows of `dt` (UTF-8 for the string layouts, Latin-1 for the binary
/// layouts); a pattern that does not compile is `InvalidOptions`.
pub fn compile(allocator: std.mem.Allocator, pattern: []const u8, dt: DataType, flags: CompileFlags) KernelError!*Regex {
    return Regex.compile(allocator, pattern, .{
        .case_insensitive = flags.ignore_case,
        .literal = flags.literal,
        .latin1 = !isText(dt),
    }, null) catch |err| switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        else => error.InvalidOptions,
    };
}

fn rowOf(op: dense.StringOperand, i: usize) []const u8 {
    return switch (op) {
        inline else => |r| r.get(i),
    };
}

// ------------------------------------------------------------------------------ match

const MatchKind = enum { regex, like };

fn Match(comptime kind: MatchKind) type {
    return struct {
        fn exec(ctx: *ExecContext, args: []const Datum, options: Options) KernelError!Datum {
            return dense.execElementwiseWith(ctx, args, options, bool_type, true, chunk);
        }

        fn chunk(ctx: *ExecContext, values: []const ExecChunkValue, len: usize, options: Options) KernelError!ArrayRef {
            const o = options.match_substring;
            var m = try (if (kind == .like) likeMatcher(ctx.allocator, o, values[0].dataType()) else Matcher.initRegex(ctx.allocator, o.pattern, values[0].dataType(), .{ .ignore_case = o.ignore_case }));
            defer m.deinit();
            return boolRows(ctx, values[0], len, &m);
        }
    };
}

/// A row predicate: a compiled regex or one of the plain tests Arrow uses for LIKE patterns.
pub const Matcher = union(enum) {
    re: *Regex,
    contains: []const u8,
    starts: []const u8,
    ends: []const u8,
    /// `re` anchored at the row start (starts_with with `ignore_case`).
    re_prefix: *Regex,
    /// `re` anchored at the row end (ends_with with `ignore_case`).
    re_suffix: *Regex,

    fn initRegex(allocator: std.mem.Allocator, pattern: []const u8, dt: DataType, flags: CompileFlags) KernelError!Matcher {
        return .{ .re = try compile(allocator, pattern, dt, flags) };
    }

    pub fn deinit(self: *Matcher) void {
        switch (self.*) {
            .re, .re_prefix, .re_suffix => |r| r.deinit(),
            else => {},
        }
    }

    pub fn matches(self: *Matcher, s: []const u8) KernelError!bool {
        return switch (self.*) {
            .re => |r| r.isMatch(s),
            .contains => |p| std.mem.indexOf(u8, s, p) != null,
            .starts => |p| std.mem.startsWith(u8, s, p),
            .ends => |p| std.mem.endsWith(u8, s, p),
            .re_prefix => |r| ((try r.find(s, 0)) orelse return false).start == 0,
            .re_suffix => |r| (try r.find(s, 0)) != null,
        };
    }
};

/// The case-insensitive literal test of starts_with / ends_with / match_substring (Arrow runs
/// them through RE2 with Unicode case folding; Latin-1 for the binary layouts).
pub fn literalMatcher(allocator: std.mem.Allocator, which: enum { contains, starts, ends }, pattern: []const u8, dt: DataType) KernelError!Matcher {
    return switch (which) {
        .contains => .{ .re = try compile(allocator, pattern, dt, .{ .ignore_case = true, .literal = true }) },
        .starts => .{ .re_prefix = try compile(allocator, pattern, dt, .{ .ignore_case = true, .literal = true }) },
        .ends => blk: {
            // Quote the pattern and anchor it at the end of the text.
            var buf: std.ArrayList(u8) = .empty;
            defer buf.deinit(allocator);
            for (pattern) |c| {
                if (c < 0x80 and !std.ascii.isAlphanumeric(c)) {
                    buf.appendSlice(allocator, if (c == 0) "\\x00" else &.{ '\\', c }) catch return error.OutOfMemory;
                } else buf.append(allocator, c) catch return error.OutOfMemory;
            }
            buf.appendSlice(allocator, "\\z") catch return error.OutOfMemory;
            break :blk .{ .re_suffix = try compile(allocator, buf.items, dt, .{ .ignore_case = true }) };
        },
    };
}

/// match_like: Arrow's plain tests for `%x%`, `x%` and `%x` (case-sensitive only), otherwise the
/// LIKE pattern translated to a regex.
fn likeMatcher(allocator: std.mem.Allocator, o: core.MatchSubstringOptions, dt: DataType) KernelError!Matcher {
    const p = o.pattern;
    if (!o.ignore_case) {
        const lead = std.mem.indexOfNone(u8, p, "%") orelse p.len;
        const trail = if (lead == p.len) p.len else std.mem.lastIndexOfNone(u8, p, "%").? + 1;
        const mid = p[lead..trail];
        const plain = std.mem.indexOfAny(u8, mid, "%_") == null;
        const no_trailing_escape = mid.len == 0 or mid[mid.len - 1] != '\\';
        // %+([^%_]*[^\\%_])?%+  /  ([^%_]*[^\\%_])?%+  /  %+([^%_]*)
        if (plain and lead >= 1 and p.len - trail >= 1 and no_trailing_escape and !(lead == p.len and p.len < 2)) return .{ .contains = mid };
        if (plain and lead == 0 and p.len - trail >= 1 and no_trailing_escape) return .{ .starts = mid };
        if (plain and lead >= 1 and trail == p.len) return .{ .ends = mid };
    }
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(allocator);
    buf.appendSlice(allocator, "(?s:^") catch return error.OutOfMemory;
    var escaped = false;
    for (p) |c| {
        if (!escaped and c == '%') {
            buf.appendSlice(allocator, ".*") catch return error.OutOfMemory;
        } else if (!escaped and c == '_') {
            buf.append(allocator, '.') catch return error.OutOfMemory;
        } else if (!escaped and c == '\\') {
            escaped = true;
        } else {
            if (std.mem.indexOfScalar(u8, ".?+*^$\\[{()|", c) != null) buf.append(allocator, '\\') catch return error.OutOfMemory;
            buf.append(allocator, c) catch return error.OutOfMemory;
            escaped = false;
        }
    }
    buf.appendSlice(allocator, "$)") catch return error.OutOfMemory;
    return .{ .re = try compile(allocator, buf.items, dt, .{ .ignore_case = o.ignore_case }) };
}

/// The bool array of `matcher` over the rows of `value`.
pub fn boolRows(ctx: *ExecContext, value: ExecChunkValue, len: usize, matcher: *Matcher) KernelError!ArrayRef {
    var arr = try dense.chunkArray(ctx.allocator, value);
    defer arr.release();
    const data = arr.data();
    var out = try dense.DenseBinaryOutput(bool).initValidity(ctx.allocator, len, data, null, bool_type);
    errdefer out.deinit();
    const out_bits = out.values();
    @memset(out_bits, 0);
    const bits = dense.validityBits(data);
    switch (try dense.StringOperand.init(.{ .array = arr })) {
        inline else => |r| for (0..len) |i| {
            if (bits) |b| if (!bitmap.bitIsSet(b, data.offset + i)) continue;
            if (try matcher.matches(r.get(i))) out_bits[i / 8] |= @as(u8, 1) << @intCast(i % 8);
        },
    }
    return out.finish() catch error.OutOfMemory;
}

// ------------------------------------------------------------------------------ count / find

/// Arrow's count loop: `FindAndConsume` over the rest of the row; an empty match advances one
/// byte (or ends the row).
pub fn countMatches(re: *Regex, s: []const u8) KernelError!usize {
    var n: usize = 0;
    var pos: usize = 0;
    while (try re.find(s[pos..], 0)) |m| {
        n += 1;
        if (m.end == 0) {
            if (pos == s.len) break;
            pos += 1;
        } else pos += m.end;
    }
    return n;
}

fn Search(comptime which: enum { count, find }) type {
    return struct {
        fn exec(ctx: *ExecContext, args: []const Datum, options: Options) KernelError!Datum {
            const tag = std.meta.activeTag(args[0].dataType());
            const out_type: DataType = if (tag == .large_string or tag == .large_binary) .{ .int64 = {} } else .{ .int32 = {} };
            return dense.execElementwiseWith(ctx, args, options, out_type, true, chunk);
        }

        fn chunk(ctx: *ExecContext, values: []const ExecChunkValue, len: usize, options: Options) KernelError!ArrayRef {
            const o = options.match_substring;
            const re = try compile(ctx.allocator, o.pattern, values[0].dataType(), .{ .ignore_case = o.ignore_case });
            defer re.deinit();
            return switch (datatype.storageDataType(values[0].dataType())) {
                .large_string, .large_binary => searchRows(i64, ctx, values[0], len, re, which == .count),
                else => searchRows(i32, ctx, values[0], len, re, which == .count),
            };
        }
    };
}

/// count (non-overlapping matches, Arrow's loop) or find (byte index of the first match, -1)
/// of every row; also the case-insensitive count_substring / find_substring.
pub fn searchRows(comptime O: type, ctx: *ExecContext, value: ExecChunkValue, len: usize, re: *Regex, count: bool) KernelError!ArrayRef {
    var arr = try dense.chunkArray(ctx.allocator, value);
    defer arr.release();
    var out = try dense.DenseBinaryOutput(O).initValidity(ctx.allocator, len, arr.data(), null, dense.dataTypeOf(O));
    errdefer out.deinit();
    switch (try dense.StringOperand.init(.{ .array = arr })) {
        inline else => |r| for (out.values(), 0..) |*dst, i| {
            const s = r.get(i);
            const v: usize = if (count) try countMatches(re, s) else if (try re.find(s, 0)) |m| m.start else {
                dst.* = -1;
                continue;
            };
            dst.* = std.math.cast(O, v) orelse return error.Overflow;
        },
    }
    return out.finish() catch error.OutOfMemory;
}

// ------------------------------------------------------------------------------ replace

/// A parsed replacement: literal bytes and group references (`\0`..`\9`, `\\`), as RE2's
/// `Rewrite`; checked against the group count of the pattern (RE2 `CheckRewriteString`).
const Template = struct {
    pieces: std.ArrayList(Piece) = .empty,
    /// The replacement refers to a group other than 0.
    uses_groups: bool = false,

    const Piece = union(enum) { text: []const u8, group: u8 };

    fn init(allocator: std.mem.Allocator, t: []const u8, groups: usize) KernelError!Template {
        var self: Template = .{};
        errdefer self.pieces.deinit(allocator);
        var i: usize = 0;
        var start: usize = 0;
        while (i < t.len) {
            if (t[i] != '\\') {
                i += 1;
                continue;
            }
            if (i + 1 >= t.len) return error.InvalidOptions;
            const c = t[i + 1];
            if (c != '\\' and !std.ascii.isDigit(c)) return error.InvalidOptions;
            if (std.ascii.isDigit(c) and c - '0' > groups) return error.InvalidOptions;
            if (i > start) self.pieces.append(allocator, .{ .text = t[start..i] }) catch return error.OutOfMemory;
            self.pieces.append(allocator, if (c == '\\') .{ .text = "\\" } else .{ .group = c - '0' }) catch return error.OutOfMemory;
            i += 2;
            start = i;
        }
        if (start < t.len) self.pieces.append(allocator, .{ .text = t[start..] }) catch return error.OutOfMemory;
        for (self.pieces.items) |p| {
            if (p == .group and p.group != 0) self.uses_groups = true;
        }
        return self;
    }

    fn write(self: *const Template, allocator: std.mem.Allocator, out: *std.ArrayList(u8), text: []const u8, groups: []const ?regex.Span) KernelError!void {
        for (self.pieces.items) |p| switch (p) {
            .text => |b| out.appendSlice(allocator, b) catch return error.OutOfMemory,
            .group => |g| if (groups[g]) |sp| out.appendSlice(allocator, text[sp.start..sp.end]) catch return error.OutOfMemory,
        };
    }
};

fn replaceExec(ctx: *ExecContext, args: []const Datum, options: Options) KernelError!Datum {
    return dense.execElementwiseWith(ctx, args, options, args[0].dataType(), true, replaceChunk);
}

/// Length of the code point at `s[0..]` as RE2's `chartorune` decodes it, 1 on a decoding error.
fn runeLen(s: []const u8) usize {
    const c = s[0];
    const n: usize = if (c < 0x80) return 1 else if (c < 0xC0) return 1 else if (c < 0xE0) 2 else if (c < 0xF0) 3 else if (c < 0xF8) 4 else return 1;
    if (s.len < n) return 1;
    var v: u32 = c & (@as(u8, 0x7F) >> @intCast(n));
    for (s[1..n]) |b| {
        if (b & 0xC0 != 0x80) return 1;
        v = v << 6 | (b & 0x3F);
    }
    const min: u32 = switch (n) {
        2 => 0x80,
        3 => 0x800,
        else => 0x10000,
    };
    return if (v < min or v > 0x10FFFF) 1 else n;
}

fn replaceChunk(ctx: *ExecContext, values: []const ExecChunkValue, len: usize, options: Options) KernelError!ArrayRef {
    const allocator = ctx.allocator;
    const o = options.replace_substring;
    const dt = values[0].dataType();
    if (isText(dt) and !std.unicode.utf8ValidateSlice(o.replacement)) return error.InvalidOptions;
    const re = try compile(allocator, o.pattern, dt, .{});
    defer re.deinit();
    var tmpl = try Template.init(allocator, o.replacement, re.groupCount());
    defer tmpl.pieces.deinit(allocator);
    // Only the groups the replacement uses are captured.
    const groups = allocator.alloc(?regex.Span, re.groupCount() + 1) catch return error.OutOfMemory;
    defer allocator.free(groups);
    const utf8 = isText(dt);

    var arr = try dense.chunkArray(allocator, values[0]);
    defer arr.release();
    const data = arr.data();
    const rows = try dense.StringOperand.init(.{ .array = arr });
    var out = try dense.VarBinaryOutput.init(allocator, dt, len, try dense.combinedValidity(allocator, len, &.{data}), dense.totalBytes(rows, len));
    errdefer out.deinit();
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(allocator);
    const bits = dense.validityBits(data);
    for (0..len) |i| {
        if (bits) |b| if (!bitmap.bitIsSet(b, data.offset + i)) {
            try out.commit(0);
            continue;
        };
        const s = rowOf(rows, i);
        buf.clearRetainingCapacity();
        var cut = false;
        if (o.max_replacements == null or s.len == 0) {
            try globalReplace(allocator, re, &tmpl, s, utf8, groups, &buf, &cut);
        } else {
            try boundedReplace(allocator, re, &tmpl, s, o.max_replacements.?, groups, &buf, &cut);
        }
        // A match inside a code point (`\C`, or `\B` / an empty match between two of its bytes)
        // would make a valid row invalid.
        if (utf8 and cut and !string.validUtf8(buf.items) and string.validUtf8(s)) return error.InvalidInput;
        try out.append(buf.items);
    }
    return out.finish();
}

/// Fill `groups` with the match at or after `start` of `text`: only group 0 unless the
/// replacement refers to groups.
fn matchGroups(re: *Regex, tmpl: *const Template, text: []const u8, start: usize, groups: []?regex.Span) KernelError!bool {
    if (!tmpl.uses_groups) {
        groups[0] = try re.find(text, start) orelse return false;
        return true;
    }
    return re.captures(text, start, groups);
}

/// RE2 `GlobalReplace`.
fn globalReplace(allocator: std.mem.Allocator, re: *Regex, tmpl: *const Template, s: []const u8, utf8: bool, groups: []?regex.Span, out: *std.ArrayList(u8), cut: *bool) KernelError!void {
    var p: usize = 0;
    var last_end: ?usize = null;
    while (p <= s.len) {
        if (!try matchGroups(re, tmpl, s, p, groups)) break;
        const m = groups[0].?;
        if (cutsCodePoint(s, m.start) or cutsCodePoint(s, m.end)) cut.* = true;
        out.appendSlice(allocator, s[p..m.start]) catch return error.OutOfMemory;
        if (last_end != null and m.start == last_end.? and m.start == m.end) {
            // No empty match right after the previous match: copy one character and go on.
            const n: usize = if (p >= s.len) 1 else if (utf8) runeLen(s[p..]) else 1;
            if (p < s.len) out.appendSlice(allocator, s[p..@min(p + n, s.len)]) catch return error.OutOfMemory;
            p += n;
            continue;
        }
        try tmpl.write(allocator, out, s, groups);
        p = m.end;
        last_end = p;
    }
    if (p < s.len) out.appendSlice(allocator, s[p..]) catch return error.OutOfMemory;
}

/// Arrow's `max_replacements` loop: `FindAndConsume` over the rest of the row, then RE2 `Replace`
/// of the pattern inside the match.
fn boundedReplace(allocator: std.mem.Allocator, re: *Regex, tmpl: *const Template, s: []const u8, max: usize, groups: []?regex.Span, out: *std.ArrayList(u8), cut: *bool) KernelError!void {
    var i: usize = 0;
    var left = max;
    while (i < s.len and left != 0) : (left -= 1) {
        const piece = s[i..];
        const m = try re.find(piece, 0) orelse break;
        out.appendSlice(allocator, piece[0..m.start]) catch return error.OutOfMemory;
        const found = piece[m.start..m.end];
        if (!try matchGroups(re, tmpl, found, 0, groups)) return error.InvalidInput;
        const r = groups[0].?;
        const at = i + m.start;
        if (cutsCodePoint(s, at + r.start) or cutsCodePoint(s, at + r.end)) cut.* = true;
        out.appendSlice(allocator, found[0..r.start]) catch return error.OutOfMemory;
        try tmpl.write(allocator, out, found, groups);
        out.appendSlice(allocator, found[r.end..]) catch return error.OutOfMemory;
        i += m.end;
    }
    out.appendSlice(allocator, s[i..]) catch return error.OutOfMemory;
}

// ------------------------------------------------------------------------------ split

fn splitExec(ctx: *ExecContext, args: []const Datum, options: Options) KernelError!Datum {
    if (options.split_pattern.reverse) return error.NotImplemented;
    const dt = args[0].dataType();
    const list_type: DataType = switch (std.meta.activeTag(dt)) {
        inline .string, .large_string, .string_view, .binary, .large_binary, .binary_view => |tag| string_transform.listOf(tag),
        else => return error.UnsupportedType,
    };
    return dense.execElementwiseWith(ctx, args, options, list_type, true, splitChunk);
}

fn splitChunk(ctx: *ExecContext, values: []const ExecChunkValue, len: usize, options: Options) KernelError!ArrayRef {
    const allocator = ctx.allocator;
    const o = options.split_pattern;
    const dt = values[0].dataType();
    const re = try compile(allocator, o.pattern, dt, .{});
    defer re.deinit();
    const max = o.max_splits orelse std.math.maxInt(usize);
    const utf8 = isText(dt);

    var arr = try dense.chunkArray(allocator, values[0]);
    defer arr.release();
    const data = arr.data();
    const rows = try dense.StringOperand.init(.{ .array = arr });
    const bits = dense.validityBits(data);
    var offsets = OwnedBuffer.initUninit(allocator, (len + 1) * 4) catch return error.OutOfMemory;
    defer offsets.deinit();
    const offs = offsets.typedSlice(i32)[0 .. len + 1];
    offs[0] = 0;
    var parts: std.ArrayList([]const u8) = .empty;
    defer parts.deinit(allocator);
    for (0..len) |row| {
        if (bits == null or bitmap.bitIsSet(bits.?, data.offset + row)) {
            const s = rowOf(rows, row);
            var i: usize = 0;
            var splits: usize = 0;
            while (splits < max) : (splits += 1) {
                const m = try re.find(s[i..], 0) orelse break;
                // Arrow never advances past an empty separator and loops forever.
                if (m.start == m.end and o.max_splits == null) return error.InvalidInput;
                if (utf8 and (cutsCodePoint(s, i + m.start) or cutsCodePoint(s, i + m.end))) return error.InvalidInput;
                parts.append(allocator, s[i .. i + m.start]) catch return error.OutOfMemory;
                i += m.end;
            }
            parts.append(allocator, s[i..]) catch return error.OutOfMemory;
        }
        offs[row + 1] = std.math.cast(i32, parts.items.len) orelse return error.Overflow;
    }
    return string_transform.partsList(allocator, data, len, &offsets, parts.items, dense.totalBytes(rows, len));
}

fn cutsCodePoint(s: []const u8, at: usize) bool {
    return at < s.len and s[at] & 0xC0 == 0x80;
}

// ------------------------------------------------------------------------------ extract

const Extract = enum { values, spans };

fn Extractor(comptime which: Extract) type {
    return struct {
        /// The struct type of the groups (names from the pattern), allocated in `allocator`.
        fn resultType(allocator: std.mem.Allocator, args: []const Datum, options: Options) KernelError!DataType {
            if (args.len != 1) return error.InvalidArity;
            const dt = args[0].dataType();
            const re = try compileNamed(allocator, options.extract_regex.pattern, dt);
            defer re.deinit();
            const n = re.groupCount();
            const names = allocator.alloc([]const u8, n) catch return error.OutOfMemory;
            for (names, 0..) |*name, i| name.* = allocator.dupe(u8, re.groupName(i + 1).?) catch return error.OutOfMemory;
            const types = allocator.alloc(DataType, n) catch return error.OutOfMemory;
            @memset(types, if (which == .values) dt else spanType(dt));
            return dense.structTypeAlloc(allocator, names, types);
        }

        fn spanType(dt: DataType) DataType {
            const S = struct {
                const i32_type: DataType = .{ .int32 = {} };
                const i64_type: DataType = .{ .int64 = {} };
            };
            const item = switch (datatype.storageDataType(dt)) {
                .large_string, .large_binary => &S.i64_type,
                else => &S.i32_type,
            };
            return .{ .fixed_size_list = .{ .value_field = .{ .name = "item", .data_type = item }, .list_size = 2 } };
        }

        /// One struct row per input row (the struct type lives with the result arrays).
        fn exec(ctx: *ExecContext, args: []const Datum, options: Options) KernelError!Datum {
            if (args.len != 1) return error.InvalidArity;
            const pattern = options.extract_regex.pattern;
            switch (args[0]) {
                .array => |a| return .{ .array = try run(ctx, a, pattern) },
                .scalar => |s| {
                    var a = try dense.chunkArray(ctx.allocator, .{ .scalar = s });
                    defer a.release();
                    var out = try run(ctx, a, pattern);
                    defer out.release();
                    return .{ .scalar = try core.Scalar.initNested(out.data().data_type, out) };
                },
                .chunked => |c| {
                    var outs: std.ArrayList(ArrayRef) = .empty;
                    defer {
                        for (outs.items) |*o| o.release();
                        outs.deinit(ctx.allocator);
                    }
                    for (c.chunks()) |chunk| {
                        const out = try run(ctx, chunk, pattern);
                        outs.append(ctx.allocator, out) catch {
                            var o = out;
                            o.release();
                            return error.OutOfMemory;
                        };
                    }
                    if (outs.items.len == 0) {
                        // No chunk: one empty chunk carries the type.
                        var empty = try core.datumBuildEmptyLikeWithAllocator(ctx.allocator, c.dataType());
                        defer empty.release();
                        const out = try run(ctx, empty.array, pattern);
                        outs.append(ctx.allocator, out) catch {
                            var o = out;
                            o.release();
                            return error.OutOfMemory;
                        };
                    }
                    const chunked = core.ChunkedArray.init(ctx.allocator, outs.items[0].data().data_type, outs.items) catch |err| return switch (err) {
                        error.OutOfMemory => error.OutOfMemory,
                        else => error.InvalidInput,
                    };
                    return .{ .chunked = chunked };
                },
                else => return error.InvalidInput,
            }
        }

        fn run(ctx: *ExecContext, arr: ArrayRef, pattern: []const u8) KernelError!ArrayRef {
            const allocator = ctx.allocator;
            const data = arr.data();
            data.validateLayout() catch return error.InvalidInput;
            const len = data.length;
            const dt = data.data_type;
            const re = try compileNamed(allocator, pattern, dt);
            defer re.deinit();
            const n = re.groupCount();
            const groups = allocator.alloc(?regex.Span, n + 1) catch return error.OutOfMemory;
            defer allocator.free(groups);
            const rows = try dense.StringOperand.init(.{ .array = arr });
            const bits = dense.validityBits(data);

            var validity = try dense.bitsBuffer(allocator, len);
            var validity_owned = true;
            defer if (validity_owned) validity.deinit();
            @memset(validity.data[0..bitmap.byteLength(len)], 0);
            const fields = allocator.alloc(Field, n) catch return error.OutOfMemory;
            var inited: usize = 0;
            defer {
                for (fields[0..inited]) |*f| f.deinit();
                allocator.free(fields);
            }
            for (fields) |*f| {
                f.* = try Field.init(allocator, dt, len);
                inited += 1;
            }
            for (0..len) |i| {
                const valid = if (bits) |b| bitmap.bitIsSet(b, data.offset + i) else true;
                const s = rowOf(rows, i);
                const matched = valid and try re.captures(s, 0, groups);
                if (matched) bitmap.setBit(validity.data, i);
                for (fields, 1..) |*f, g| try f.push(i, s, if (matched) groups[g] else null, matched);
            }

            const names = allocator.alloc([]const u8, n) catch return error.OutOfMemory;
            defer allocator.free(names);
            const children = allocator.alloc(ArrayRef, n) catch return error.OutOfMemory;
            defer allocator.free(children);
            var built: usize = 0;
            errdefer for (children[0..built]) |*c| c.release();
            for (fields, 0..) |*f, g| {
                names[g] = re.groupName(g + 1).?;
                children[g] = try f.finish();
                built += 1;
            }
            validity_owned = false;
            built = 0;
            // structArray takes the children and the validity over, also on error.
            return dense.structArray(allocator, names, children, len, validity);
        }

        /// The output of one group: its bytes ("" when it did not take part) or its
        /// [start, length] (null when it did not take part); null where the row did not match.
        const Field = if (which == .values) struct {
            out: dense.VarBinaryOutput,

            fn init(allocator: std.mem.Allocator, dt: DataType, len: usize) KernelError!@This() {
                var v = try dense.bitsBuffer(allocator, len);
                @memset(v.data[0..bitmap.byteLength(len)], 0);
                return .{ .out = try dense.VarBinaryOutput.init(allocator, dt, len, v, 0) };
            }

            fn deinit(self: *@This()) void {
                self.out.deinit();
            }

            fn push(self: *@This(), i: usize, s: []const u8, group: ?regex.Span, matched: bool) KernelError!void {
                if (!matched) return self.out.commit(0);
                bitmap.setBit(self.out.validity.?.data, i);
                const sp = group orelse regex.Span{ .start = 0, .end = 0 };
                try self.out.append(s[sp.start..sp.end]);
            }

            fn finish(self: *@This()) KernelError!ArrayRef {
                return self.out.finish();
            }
        } else struct {
            allocator: std.mem.Allocator,
            dt: DataType,
            len: usize,
            validity: OwnedBuffer,
            vals: OwnedBuffer,
            large: bool,

            fn init(allocator: std.mem.Allocator, dt: DataType, len: usize) KernelError!@This() {
                const large = switch (datatype.storageDataType(dt)) {
                    .large_string, .large_binary => true,
                    else => false,
                };
                var v = try dense.bitsBuffer(allocator, len);
                errdefer v.deinit();
                @memset(v.data[0..bitmap.byteLength(len)], 0);
                const vals = OwnedBuffer.init(allocator, 2 * len * @as(usize, if (large) 8 else 4)) catch return error.OutOfMemory;
                return .{ .allocator = allocator, .dt = dt, .len = len, .validity = v, .vals = vals, .large = large };
            }

            fn deinit(self: *@This()) void {
                self.validity.deinit();
                self.vals.deinit();
            }

            fn push(self: *@This(), i: usize, s: []const u8, group: ?regex.Span, matched: bool) KernelError!void {
                _ = s;
                _ = matched;
                const sp = group orelse return;
                bitmap.setBit(self.validity.data, i);
                if (self.large) {
                    const v = self.vals.typedSlice(i64);
                    v[2 * i] = std.math.cast(i64, sp.start) orelse return error.Overflow;
                    v[2 * i + 1] = std.math.cast(i64, sp.end - sp.start) orelse return error.Overflow;
                } else {
                    const v = self.vals.typedSlice(i32);
                    v[2 * i] = std.math.cast(i32, sp.start) orelse return error.Overflow;
                    v[2 * i + 1] = std.math.cast(i32, sp.end - sp.start) orelse return error.Overflow;
                }
            }

            /// The fixed_size_list array (the buffers move to it; `deinit` stays safe).
            fn finish(self: *@This()) KernelError!ArrayRef {
                const len = self.len;
                const item_type: DataType = if (self.large) .{ .int64 = {} } else .{ .int32 = {} };
                var item = try dense.finishFixed(self.allocator, item_type, 2 * len, &self.vals, null);
                errdefer item.release();
                var buffers: [1]SharedBuffer = .{.empty};
                const null_count = bitmap.countUnsetBit(self.validity.data, len);
                buffers[0] = self.validity.toShared(bitmap.byteLength(len)) catch return error.OutOfMemory;
                errdefer buffers[0].release();
                var children = [1]ArrayRef{item};
                return ArrayRef.fromOwnedInline(self.allocator, .{
                    .data_type = spanType(self.dt),
                    .length = len,
                    .null_count = null_count,
                    .buffers = &buffers,
                    .children = &children,
                }) catch error.OutOfMemory;
            }
        };

        /// Compile for extraction: every group must be named (Arrow).
        fn compileNamed(allocator: std.mem.Allocator, pattern: []const u8, dt: DataType) KernelError!*Regex {
            const re = try compile(allocator, pattern, dt, .{});
            errdefer re.deinit();
            for (1..re.groupCount() + 1) |i| if (re.groupName(i) == null) return error.InvalidOptions;
            return re;
        }
    };
}
