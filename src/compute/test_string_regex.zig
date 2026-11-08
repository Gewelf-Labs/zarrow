const std = @import("std");
const common = @import("test_common.zig");

const ArrayRef = common.ArrayRef;
const DataType = common.DataType;
const Datum = common.Datum;
const ExecContext = common.ExecContext;
const Options = common.Options;
const Scalar = common.Scalar;
const TypeTag = std.meta.Tag(DataType);

const all_tags = [_]TypeTag{ .string, .large_string, .string_view, .binary, .large_binary, .binary_view };

fn match(p: []const u8, ignore_case: bool) Options {
    return .{ .match_substring = .{ .pattern = p, .ignore_case = ignore_case } };
}

fn replace(p: []const u8, r: []const u8, max: ?usize) Options {
    return .{ .replace_substring = .{ .pattern = p, .replacement = r, .max_replacements = max } };
}

fn splitOn(p: []const u8, max: ?usize) Options {
    return .{ .split_pattern = .{ .pattern = p, .max_splits = max } };
}

fn extract(p: []const u8) Options {
    return .{ .extract_regex = .{ .pattern = p } };
}

fn hexDecode(allocator: std.mem.Allocator, hex: []const u8) ![]u8 {
    const out = try allocator.alloc(u8, hex.len / 2);
    errdefer allocator.free(out);
    _ = try std.fmt.hexToBytes(out, hex);
    return out;
}

fn appendHex(allocator: std.mem.Allocator, out: *std.ArrayList(u8), bytes: []const u8) !void {
    const digits = "0123456789abcdef";
    for (bytes) |b| {
        try out.append(allocator, digits[b >> 4]);
        try out.append(allocator, digits[b & 15]);
    }
}

fn appendInt(allocator: std.mem.Allocator, out: *std.ArrayList(u8), v: i64) !void {
    var buf: [24]u8 = undefined;
    try out.appendSlice(allocator, try std.fmt.bufPrint(&buf, "{d}", .{v}));
}

/// The parts of row 0 of a list<string-like> array, hex, comma-separated.
fn formatSplit(allocator: std.mem.Allocator, out: *std.ArrayList(u8), arr: ArrayRef) !void {
    const data = arr.data();
    try data.validateFull();
    const offsets = (try data.buffers[1].typedSlice(i32))[data.offset..][0..2];
    const child = data.children[0].data();
    var k: usize = @intCast(offsets[0]);
    while (k < offsets[1]) : (k += 1) {
        if (k > @as(usize, @intCast(offsets[0]))) try out.append(allocator, ',');
        try appendHex(allocator, out, (try common.bytesAt(child, k)).?);
    }
}

/// Row 0 of an extract_regex_span result as `start:length` per group (`-` when null), `N` for
/// a null row, `{}` for a valid row without groups.
fn formatSpans(allocator: std.mem.Allocator, out: *std.ArrayList(u8), arr: ArrayRef) !void {
    const data = arr.data();
    try data.validateFull();
    if (data.isNull(0)) return out.append(allocator, 'N');
    if (data.children.len == 0) return out.appendSlice(allocator, "{}");
    for (data.children, 0..) |c, g| {
        if (g > 0) try out.append(allocator, ',');
        const fsl = c.data();
        const row = data.offset;
        if (fsl.isNull(row)) {
            try out.append(allocator, '-');
            continue;
        }
        const item = fsl.children[0].data();
        const vals = (try item.buffers[1].typedSlice(i32))[item.offset..];
        try appendInt(allocator, out, vals[(fsl.offset + row) * 2]);
        try out.append(allocator, ':');
        try appendInt(allocator, out, vals[(fsl.offset + row) * 2 + 1]);
    }
}

/// Run one corpus case; the result in the corpus notation (`E` for an error).
fn runCase(allocator: std.mem.Allocator, ctx: *ExecContext, func: []const u8, opt: []const u8, binary: bool, pattern: []const u8, text: []const u8, out: *std.ArrayList(u8)) !void {
    var arr = if (binary) try common.makeStrings(.binary, allocator, &.{text}) else try common.makeStrings(.string, allocator, &.{text});
    defer arr.release();
    const args = [_]Datum{.{ .array = arr }};
    var name: []const u8 = func;
    var options: Options = undefined;
    var repl: []u8 = &.{};
    defer allocator.free(repl);
    if (std.mem.eql(u8, func, "match") or std.mem.eql(u8, func, "like")) {
        name = if (func[0] == 'm') "match_substring_regex" else "match_like";
        options = match(pattern, opt[0] == '1');
    } else if (std.mem.eql(u8, func, "find") or std.mem.eql(u8, func, "count")) {
        name = if (func[0] == 'f') "find_substring_regex" else "count_substring_regex";
        options = match(pattern, false);
    } else if (std.mem.eql(u8, func, "replace")) {
        name = "replace_substring_regex";
        const colon = std.mem.indexOfScalar(u8, opt, ':').?;
        const max = try std.fmt.parseInt(i64, opt[0..colon], 10);
        repl = try hexDecode(allocator, opt[colon + 1 ..]);
        options = replace(pattern, repl, if (max < 0) null else @intCast(max));
    } else if (std.mem.eql(u8, func, "split")) {
        name = "split_pattern_regex";
        const max = try std.fmt.parseInt(i64, opt, 10);
        options = splitOn(pattern, if (max < 0) null else @intCast(max));
    } else if (std.mem.eql(u8, func, "span")) {
        name = "extract_regex_span";
        options = extract(pattern);
    } else return error.UnknownCase;

    var res = ctx.invoke(name, .scalar, &args, options) catch |err| {
        if (std.mem.eql(u8, func, "split") and err == error.InvalidInput) return out.append(allocator, 'H');
        return out.append(allocator, 'E');
    };
    defer res.release();
    const r = res.array;
    if (std.mem.eql(u8, func, "match") or std.mem.eql(u8, func, "like")) {
        const v = (try common.collect(bool, allocator, res));
        defer allocator.free(v);
        try out.append(allocator, if (v[0].?) '1' else '0');
    } else if (std.mem.eql(u8, func, "find") or std.mem.eql(u8, func, "count")) {
        const v = (try common.collect(i32, allocator, res));
        defer allocator.free(v);
        try appendInt(allocator, out, v[0].?);
    } else if (std.mem.eql(u8, func, "replace")) {
        r.data().validateFull() catch |err| {
            std.debug.print("invalid output: pattern \"{s}\" text \"{s}\" {s}\n", .{ pattern, text, opt });
            return err;
        };
        try appendHex(allocator, out, (try common.bytesAt(r.data(), 0)).?);
    } else if (std.mem.eql(u8, func, "split")) {
        try formatSplit(allocator, out, r);
    } else {
        try formatSpans(allocator, out, r);
    }
}

/// Corpus cases where zarrow knowingly differs from pyarrow 26 (each one documented in
/// docs/compute-support-matrix.md): patterns, any function and text.
const known_mismatches = [_][]const u8{
    // Unicode script classes are not implemented (no Scripts.txt tables): InvalidOptions.
    "\\p{Greek}",
    "\\p{Han}",
};

/// Whether a replace (hex) or split (comma-separated hex) result holds invalid UTF-8.
fn invalidUtf8Result(allocator: std.mem.Allocator, expected: []const u8) !bool {
    if (expected.len == 1) return false;
    var parts = std.mem.splitScalar(u8, expected, ',');
    while (parts.next()) |p| {
        const bytes = hexDecode(allocator, p) catch return false;
        defer allocator.free(bytes);
        if (!std.unicode.utf8ValidateSlice(bytes)) return true;
    }
    return false;
}

fn isKnown(pattern: []const u8) bool {
    for (known_mismatches) |k| {
        if (std.mem.eql(u8, k, pattern)) return true;
    }
    return false;
}

test "string regex kernels match pyarrow 26 on the generated corpus (tools/regex_corpus.py)" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    const corpus = @embedFile("testdata/regex_corpus.tsv");

    var pattern: []u8 = &.{};
    defer allocator.free(pattern);
    var text: []u8 = &.{};
    defer allocator.free(text);
    var binary = false;
    var got: std.ArrayList(u8) = .empty;
    defer got.deinit(allocator);
    var cases: usize = 0;
    var mismatches: usize = 0;
    var scripts: usize = 0;
    var utf8_cuts: usize = 0;
    var lines = std.mem.splitScalar(u8, corpus, '\n');
    while (lines.next()) |line| {
        if (line.len == 0 or line[0] == '#') continue;
        var fields = std.mem.splitScalar(u8, line, '\t');
        const f0 = fields.next().?;
        const f1 = fields.next() orelse "";
        const f2 = fields.next() orelse "";
        if (std.mem.eql(u8, f0, "P")) {
            binary = f1[0] == 'b';
            allocator.free(pattern);
            pattern = try hexDecode(allocator, f2);
            continue;
        }
        if (std.mem.eql(u8, f0, "T")) {
            allocator.free(text);
            text = try hexDecode(allocator, f1);
            continue;
        }
        cases += 1;
        got.clearRetainingCapacity();
        try runCase(allocator, &ctx, f0, f1, binary, pattern, text, &got);
        // An error matches an error of any kind.
        if (std.mem.eql(u8, got.items, f2)) continue;
        // A match inside a code point: a string result that would not be UTF-8 is an error here.
        if (!binary and std.mem.eql(u8, got.items, "E") and try invalidUtf8Result(allocator, f2)) {
            utf8_cuts += 1;
            continue;
        }
        if (isKnown(pattern)) {
            scripts += 1;
            continue;
        }
        mismatches += 1;
        if (mismatches <= 40) std.debug.print("mismatch {s} {s} [{s}] pattern \"{s}\" text \"{s}\": got {s}, pyarrow {s}\n", .{ f0, f1, if (binary) "binary" else "string", pattern, text, got.items, f2 });
    }
    if (mismatches > 0) std.debug.print("{d} mismatches of {d} cases\n", .{ mismatches, cases });
    try std.testing.expectEqual(@as(usize, 33605), cases);
    try std.testing.expectEqual(@as(usize, 0), mismatches);
    // The documented differences: script classes rejected, and results Arrow would emit as
    // invalid UTF-8 (a match inside a code point) rejected.
    try std.testing.expectEqual(@as(usize, 264), scripts);
    try std.testing.expectEqual(@as(usize, 8), utf8_cuts);
}

/// Rows of a list, struct or fixed_size_list result as text: rows joined by `;`, `N` for a null
/// row, list items by `|`, struct fields by `,`, spans as `start:length`. Every part must pass
/// validateFull.
fn describe(allocator: std.mem.Allocator, datum: Datum) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    const parts: []const ArrayRef = switch (datum) {
        .array => |*a| a[0..1],
        .chunked => |c| c.chunks(),
        .scalar => |*s| (&s.payload.?)[0..1],
        else => return error.InvalidInput,
    };
    var first = true;
    for (parts) |p| {
        try p.data().validateFull();
        for (0..p.data().length) |i| {
            if (!first) try out.append(allocator, ';');
            first = false;
            try describeRow(allocator, &out, p.data(), i);
        }
    }
    return out.toOwnedSlice(allocator);
}

fn describeRow(allocator: std.mem.Allocator, out: *std.ArrayList(u8), data: *const common.array_mod.ArrayData, i: usize) !void {
    if (data.isNull(i)) return out.append(allocator, 'N');
    switch (data.data_type) {
        .list => {
            const offsets = (try data.buffers[1].typedSlice(i32))[data.offset..];
            var k: usize = @intCast(offsets[i]);
            while (k < offsets[i + 1]) : (k += 1) {
                if (k > @as(usize, @intCast(offsets[i]))) try out.append(allocator, '|');
                try out.appendSlice(allocator, (try common.bytesAt(data.children[0].data(), k)).?);
            }
        },
        .struct_ => for (data.children, 0..) |c, g| {
            if (g > 0) try out.append(allocator, ',');
            try describeRow(allocator, out, c.data(), data.offset + i);
        },
        .fixed_size_list => {
            const item = data.children[0].data();
            const at = (data.offset + i) * 2;
            const v: [2]i64 = switch (item.data_type) {
                .int32 => blk: {
                    const s = (try item.buffers[1].typedSlice(i32))[item.offset..];
                    break :blk .{ s[at], s[at + 1] };
                },
                else => blk: {
                    const s = (try item.buffers[1].typedSlice(i64))[item.offset..];
                    break :blk .{ s[at], s[at + 1] };
                },
            };
            try appendInt(allocator, out, v[0]);
            try out.append(allocator, ':');
            try appendInt(allocator, out, v[1]);
        },
        else => try out.appendSlice(allocator, (try common.bytesAt(data, i)).?),
    }
}

fn expectDescribe(ctx: *ExecContext, name: []const u8, args: []const Datum, options: Options, expected: []const u8) !void {
    const allocator = std.testing.allocator;
    var res = try ctx.invokeScalar(name, args, options);
    defer res.release();
    const got = try describeAny(allocator, res);
    defer allocator.free(got);
    if (!std.mem.eql(u8, expected, got)) {
        std.debug.print("{s}: expected \"{s}\", got \"{s}\"\n", .{ name, expected, got });
        return error.TestExpectedEqual;
    }
}

/// `describe` for any result: bool, integer and string / binary results too.
fn describeAny(allocator: std.mem.Allocator, datum: Datum) ![]u8 {
    const dt = datum.dataType();
    switch (dt) {
        .list, .struct_, .fixed_size_list => return describe(allocator, datum),
        else => {},
    }
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    switch (dt) {
        .bool => {
            const v = try common.collect(bool, allocator, datum);
            defer allocator.free(v);
            for (v, 0..) |x, i| {
                if (i > 0) try out.append(allocator, ';');
                try out.append(allocator, if (x) |b| (if (b) '1' else '0') else 'N');
            }
        },
        .int32 => try describeInts(i32, allocator, &out, datum),
        .int64 => try describeInts(i64, allocator, &out, datum),
        else => {
            const v = try common.collectStrings(allocator, datum);
            defer allocator.free(v);
            for (v, 0..) |x, i| {
                if (i > 0) try out.append(allocator, ';');
                if (x) |s| try out.appendSlice(allocator, s) else try out.append(allocator, 'N');
            }
        },
    }
    return out.toOwnedSlice(allocator);
}

fn describeInts(comptime T: type, allocator: std.mem.Allocator, out: *std.ArrayList(u8), datum: Datum) !void {
    const v = try common.collect(T, allocator, datum);
    defer allocator.free(v);
    for (v, 0..) |x, i| {
        if (i > 0) try out.append(allocator, ';');
        if (x) |n| try appendInt(allocator, out, n) else try out.append(allocator, 'N');
    }
}

test "string regex hand-written values on every layout: match, like, count, find, replace, split, extract (pyarrow 26 results)" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    inline for (all_tags) |tag| {
        var arr = try common.makeStrings(tag, allocator, &.{ "abc", null, "", "xaxbx", "AbC aBc", "a1b22" });
        defer arr.release();
        const args = [_]Datum{.{ .array = arr }};
        try expectDescribe(&ctx, "match_substring_regex", &args, match("a.c", false), "1;N;0;0;1;0");
        try expectDescribe(&ctx, "match_substring_regex", &args, match("A.C", true), "1;N;0;0;1;0");
        try expectDescribe(&ctx, "match_like", &args, match("a%", false), "1;N;0;0;0;1");
        try expectDescribe(&ctx, "match_like", &args, match("_B%", true), "1;N;0;0;1;0");
        try expectDescribe(&ctx, "count_substring_regex", &args, match("[xb]", false), "1;N;0;4;1;1");
        try expectDescribe(&ctx, "find_substring_regex", &args, match("b|B", false), "1;N;-1;3;1;2");
        try expectDescribe(&ctx, "replace_substring_regex", &args, replace("a(b)?", "<\\1>", null), "<b>c;N;;x<>xbx;AbC <>Bc;<>1b22");
        try expectDescribe(&ctx, "replace_substring_regex", &args, replace("[a-z]", "#", 2), "##c;N;;##xbx;A#C #Bc;#1#22");
        try expectDescribe(&ctx, "split_pattern_regex", &args, splitOn("[b2]+", null), "a|c;N;;xax|x;A|C aBc;a1|");
        try expectDescribe(&ctx, "split_pattern_regex", &args, splitOn("x", 1), "abc;N;;|axbx;AbC aBc;a1b22");
        try expectDescribe(&ctx, "extract_regex", &args, extract("(?P<l>[a-z])(?P<d>\\d)?"), "a,;N;N;x,;b,;a,1");
        try expectDescribe(&ctx, "extract_regex_span", &args, extract("(?P<l>[a-z])(?P<d>\\d)?"), "0:1,N;N;N;0:1,N;1:1,N;0:1,1:1");
    }
}

test "string regex sliced and chunked inputs with nulls equal the per-row results" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    var rng = std.Random.DefaultPrng.init(19);
    const r = rng.random();
    const alphabet = [_][]const u8{ "a", "b", "x", " ", "é", "É", "1", "\n" };
    var rows: [300]?[]const u8 = undefined;
    var store: [300][60]u8 = undefined;
    for (&rows, &store) |*row, *buf| {
        if (r.uintLessThan(u8, 10) == 0) {
            row.* = null;
            continue;
        }
        var n: usize = 0;
        for (0..r.uintAtMost(usize, 12)) |_| {
            const piece = alphabet[r.uintLessThan(usize, alphabet.len)];
            @memcpy(buf[n..][0..piece.len], piece);
            n += piece.len;
        }
        row.* = buf[0..n];
    }
    const cases = [_]struct { []const u8, Options }{
        .{ "match_substring_regex", match("(?i)é+|b\\b", false) },
        .{ "match_like", match("%a_", true) },
        .{ "count_substring_regex", match("a|x+", false) },
        .{ "find_substring_regex", match("\\d", false) },
        .{ "replace_substring_regex", replace("(a)(x)?", "[\\2\\1]", null) },
        .{ "replace_substring_regex", replace("a+", "-", 3) },
        .{ "split_pattern_regex", splitOn("\\s+", null) },
        .{ "extract_regex", extract("(?P<w>\\w)(?P<s>\\s)?") },
        .{ "extract_regex_span", extract("(?P<w>\\w)(?P<s>\\s)?") },
    };
    inline for (.{ TypeTag.string, TypeTag.large_binary }) |tag| {
        var full = try common.makeStrings(tag, allocator, &rows);
        defer full.release();
        var sliced = try full.slice(17, 250);
        defer sliced.release();
        var chunked = try common.chunkedAt(allocator, sliced, &.{ 1, 60, 61, 200 });
        defer chunked.release();
        for (cases) |c| {
            var whole = try ctx.invokeScalar(c[0], &.{.{ .array = sliced }}, c[1]);
            defer whole.release();
            const whole_text = try describeAny(allocator, whole);
            defer allocator.free(whole_text);
            var parts = try ctx.invokeScalar(c[0], &.{.{ .chunked = chunked }}, c[1]);
            defer parts.release();
            try std.testing.expect(parts == .chunked);
            const parts_text = try describeAny(allocator, parts);
            defer allocator.free(parts_text);
            try std.testing.expectEqualStrings(whole_text, parts_text);
            // Row by row through one-row slices.
            var each: std.ArrayList(u8) = .empty;
            defer each.deinit(allocator);
            for (0..250) |i| {
                var one = try sliced.slice(i, 1);
                defer one.release();
                var res = try ctx.invokeScalar(c[0], &.{.{ .array = one }}, c[1]);
                defer res.release();
                const t = try describeAny(allocator, res);
                defer allocator.free(t);
                if (i > 0) try each.append(allocator, ';');
                try each.appendSlice(allocator, t);
            }
            try std.testing.expectEqualStrings(whole_text, each.items);
        }
    }
}

test "string regex scalars, empty and all-null inputs" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    const s = Scalar.init(.{ .string = {} }, .{ .string = "a1b2" });
    const n = Scalar.init(.{ .string = {} }, .null);
    try expectDescribe(&ctx, "match_substring_regex", &.{.{ .scalar = s }}, match("\\d", false), "1");
    try expectDescribe(&ctx, "match_substring_regex", &.{.{ .scalar = n }}, match("\\d", false), "N");
    try expectDescribe(&ctx, "count_substring_regex", &.{.{ .scalar = s }}, match("\\d", false), "2");
    try expectDescribe(&ctx, "replace_substring_regex", &.{.{ .scalar = s }}, replace("\\d", "#", null), "a#b#");
    try expectDescribe(&ctx, "replace_substring_regex", &.{.{ .scalar = n }}, replace("\\d", "#", null), "N");
    try expectDescribe(&ctx, "split_pattern_regex", &.{.{ .scalar = s }}, splitOn("\\d", null), "a|b|");
    try expectDescribe(&ctx, "extract_regex", &.{.{ .scalar = s }}, extract("(?P<x>[a-z])(?P<y>\\d)"), "a,1");
    try expectDescribe(&ctx, "extract_regex", &.{.{ .scalar = n }}, extract("(?P<x>[a-z])(?P<y>\\d)"), "N");
    try expectDescribe(&ctx, "extract_regex_span", &.{.{ .scalar = s }}, extract("(?P<y>\\d)"), "1:1");
    // Empty arrays and all-null arrays, and an empty chunked array (no chunk).
    inline for (.{ TypeTag.string, TypeTag.binary_view }) |tag| {
        var empty = try common.makeStrings(tag, allocator, &.{});
        defer empty.release();
        var nulls = try common.makeStrings(tag, allocator, &.{ null, null });
        defer nulls.release();
        inline for (.{ "match_substring_regex", "count_substring_regex", "find_substring_regex", "match_like" }) |name| {
            try expectDescribe(&ctx, name, &.{.{ .array = empty }}, match("a", false), "");
            try expectDescribe(&ctx, name, &.{.{ .array = nulls }}, match("a", false), "N;N");
        }
        try expectDescribe(&ctx, "replace_substring_regex", &.{.{ .array = nulls }}, replace("a", "b", null), "N;N");
        try expectDescribe(&ctx, "split_pattern_regex", &.{.{ .array = nulls }}, splitOn("a", null), "N;N");
        try expectDescribe(&ctx, "extract_regex", &.{.{ .array = empty }}, extract("(?P<x>a)"), "");
        try expectDescribe(&ctx, "extract_regex_span", &.{.{ .array = nulls }}, extract("(?P<x>a)"), "N;N");
    }
    var none_chunks = try common.ChunkedArray.init(allocator, .{ .string = {} }, &.{});
    defer none_chunks.release();
    var res = try ctx.invokeScalar("extract_regex", &.{.{ .chunked = none_chunks }}, extract("(?P<x>a)"));
    defer res.release();
    try std.testing.expectEqual(@as(usize, 0), res.chunked.len());
    try std.testing.expectEqualStrings("x", res.chunked.dataType().struct_.fields[0].name);
    // A pattern without groups gives a struct without fields.
    var arr = try common.makeStrings(.string, allocator, &.{ "abc", "x" });
    defer arr.release();
    var none_fields = try ctx.invokeScalar("extract_regex", &.{.{ .array = arr }}, extract("b"));
    defer none_fields.release();
    try std.testing.expectEqual(@as(usize, 0), none_fields.array.data().children.len);
    try std.testing.expect(!none_fields.array.data().isNull(0) and none_fields.array.data().isNull(1));
    // The result type of extract_regex comes from the pattern.
    var arena: std.heap.ArenaAllocator = .init(allocator);
    defer arena.deinit();
    const dt = try registry.resolveResultTypeAlloc(arena.allocator(), "extract_regex_span", .scalar, &.{.{ .array = arr }}, extract("(?P<first>a)(?P<second>b)"));
    try std.testing.expectEqual(@as(usize, 2), dt.struct_.fields.len);
    try std.testing.expectEqualStrings("second", dt.struct_.fields[1].name);
    try std.testing.expect(dt.struct_.fields[1].data_type.* == .fixed_size_list);
}

test "string regex errors: patterns, replacements, unnamed groups, reverse and empty separators, types, malformed input" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    var s = try common.makeStrings(.string, allocator, &.{ "abc", "é" });
    defer s.release();
    const args = [_]Datum{.{ .array = s }};
    const E = error.InvalidOptions;
    try std.testing.expectError(E, ctx.invokeScalar("match_substring_regex", &args, match("(a", false)));
    try std.testing.expectError(E, ctx.invokeScalar("match_substring_regex", &args, match("a\\1", false)));
    try std.testing.expectError(E, ctx.invokeScalar("match_substring_regex", &args, match("(?=a)", false)));
    try std.testing.expectError(E, ctx.invokeScalar("count_substring_regex", &args, match("a{1001}", false)));
    try std.testing.expectError(E, ctx.invokeScalar("find_substring_regex", &args, match("\xff", false)));
    try std.testing.expectError(E, ctx.invokeScalar("replace_substring_regex", &args, replace("a", "\\1", null)));
    try std.testing.expectError(E, ctx.invokeScalar("replace_substring_regex", &args, replace("(a)", "\\x", null)));
    try std.testing.expectError(E, ctx.invokeScalar("replace_substring_regex", &args, replace("(a)", "a\\", null)));
    try std.testing.expectError(E, ctx.invokeScalar("replace_substring_regex", &args, replace("(a)", "\xff", null)));
    try std.testing.expectError(E, ctx.invokeScalar("extract_regex", &args, extract("(a)")));
    try std.testing.expectError(E, ctx.invokeScalar("extract_regex_span", &args, extract("(?P<x>a)(b)")));
    try std.testing.expectError(E, ctx.invokeScalar("extract_regex", &args, match("a", false)));
    try std.testing.expectError(E, ctx.invokeScalar("match_like", &args, replace("a", "b", null)));
    try std.testing.expectError(error.NotImplemented, ctx.invokeScalar("split_pattern_regex", &args, .{ .split_pattern = .{ .pattern = "b", .reverse = true } }));
    // Arrow loops forever on an empty separator without max_splits; with a bound it splits.
    try std.testing.expectError(error.InvalidInput, ctx.invokeScalar("split_pattern_regex", &args, splitOn("x*", null)));
    try expectDescribe(&ctx, "split_pattern_regex", &args, splitOn("x*", 2), "||abc;||é");
    // Replace with max_replacements re-matches inside the match (RE2 Replace): \B fails there.
    try std.testing.expectError(error.InvalidInput, ctx.invokeScalar("replace_substring_regex", &args, replace("\\Bb", "x", 1)));
    // A match inside a code point would emit invalid UTF-8 into a string array.
    try std.testing.expectError(error.InvalidInput, ctx.invokeScalar("replace_substring_regex", &args, replace("\\C", "x", 1)));
    try std.testing.expectError(error.InvalidInput, ctx.invokeScalar("split_pattern_regex", &args, splitOn("\\C", null)));
    // The binary layouts are Latin-1: \C and `.` are one byte, so nothing is cut.
    var b = try common.makeStrings(.binary, allocator, &.{"é"});
    defer b.release();
    try expectDescribe(&ctx, "replace_substring_regex", &.{.{ .array = b }}, replace(".", "x", null), "xx");
    try expectDescribe(&ctx, "count_substring_regex", &.{.{ .array = b }}, match("\\C", false), "2");
    // Types: no numeric kernels; arity.
    var ints = try common.makeArray(i32, allocator, &.{1});
    defer ints.release();
    try std.testing.expectError(error.NoMatchingKernel, ctx.invokeScalar("match_substring_regex", &.{.{ .array = ints }}, match("a", false)));
    try std.testing.expectError(error.InvalidArity, ctx.invokeScalar("extract_regex", &.{ args[0], args[0] }, extract("(?P<x>a)")));
    // Offsets past the data buffer fail validateLayout instead of being read.
    const offsets = [_]i32{ 0, 100 };
    const broken_bufs = [_]common.SharedBuffer{ .empty, .fromSlice(std.mem.sliceAsBytes(&offsets)), .fromSlice("abc") };
    var broken = try ArrayRef.fromBorrowed(allocator, .{ .data_type = .{ .string = {} }, .length = 1, .null_count = 0, .buffers = &broken_bufs });
    defer broken.release();
    inline for (.{ "match_substring_regex", "count_substring_regex", "match_like" }) |name| {
        try std.testing.expectError(error.InvalidInput, ctx.invokeScalar(name, &.{.{ .array = broken }}, match("a", false)));
    }
    try std.testing.expectError(error.InvalidInput, ctx.invokeScalar("replace_substring_regex", &.{.{ .array = broken }}, replace("a", "b", null)));
    try std.testing.expectError(error.InvalidInput, ctx.invokeScalar("split_pattern_regex", &.{.{ .array = broken }}, splitOn("a", null)));
    try std.testing.expectError(error.InvalidInput, ctx.invokeScalar("extract_regex", &.{.{ .array = broken }}, extract("(?P<x>a)")));
}

test "case-insensitive literal kernels fold through Unicode simple case folding (pyarrow 26 results)" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    inline for (.{ TypeTag.string, TypeTag.large_string, TypeTag.string_view }) |tag| {
        var s = try common.makeStrings(tag, allocator, &.{ "ÉCOLE École", "ſ S", "K-sign \u{212A}", null });
        defer s.release();
        const args = [_]Datum{.{ .array = s }};
        try expectDescribe(&ctx, "match_substring", &args, match("école", true), "1;0;0;N");
        try expectDescribe(&ctx, "starts_with", &args, match("éc", true), "1;0;0;N");
        try expectDescribe(&ctx, "ends_with", &args, match("OLE", true), "1;0;0;N");
        try expectDescribe(&ctx, "count_substring", &args, match("s", true), "0;2;1;N");
        try expectDescribe(&ctx, "count_substring", &args, match("k", true), "0;0;2;N");
        try expectDescribe(&ctx, "find_substring", &args, match("É", true), "0;-1;-1;N");
        // Regex metacharacters stay literal.
        try expectDescribe(&ctx, "ends_with", &args, match(".", true), "0;0;0;N");
    }
    // Binary layouts: RE2's Latin-1 mode folds the ASCII letters only.
    var b = try common.makeStrings(.binary, allocator, &.{ "\xc9cole", "x\xe9", null });
    defer b.release();
    const bargs = [_]Datum{.{ .array = b }};
    try expectDescribe(&ctx, "match_substring", &bargs, match("\xe9", true), "0;1;N");
    try expectDescribe(&ctx, "count_substring", &bargs, match("\xe9", true), "0;1;N");
    try expectDescribe(&ctx, "ends_with", &bargs, match("\xc9", true), "0;0;N");
    try expectDescribe(&ctx, "starts_with", &bargs, match("\xe9C", true), "0;0;N");
    try expectDescribe(&ctx, "count_substring", &bargs, match("COLE", true), "1;0;N");
}
