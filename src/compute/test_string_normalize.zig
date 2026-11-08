const std = @import("std");
const common = @import("test_common.zig");

const ArrayRef = common.ArrayRef;
const DataType = common.DataType;
const Datum = common.Datum;
const ExecContext = common.ExecContext;
const Options = common.Options;
const Form = common.core.Utf8NormalizeOptions.Form;

const forms = [_]Form{ .nfc, .nfd, .nfkc, .nfkd };

fn formOf(name: []const u8) Form {
    for (forms, [_][]const u8{ "NFC", "NFD", "NFKC", "NFKD" }) |f, n| if (std.mem.eql(u8, n, name)) return f;
    unreachable;
}

fn fnv(h: u64, bytes: []const u8) u64 {
    var x = h;
    for (bytes) |b| x = (x ^ b) *% 0x100000001B3;
    return x;
}

fn hexDecode(allocator: std.mem.Allocator, hex: []const u8) ![]u8 {
    const out = try allocator.alloc(u8, hex.len / 2);
    errdefer allocator.free(out);
    _ = try std.fmt.hexToBytes(out, hex);
    return out;
}

test "utf8_normalize matches Python unicodedata and pyarrow 26 on every code point and on mixed sequences (tools/normalize_corpus.py)" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    const corpus = @embedFile("testdata/normalize_corpus.tsv");

    // Every code point of each 4096 block as one row; the hashes of the results per form.
    var blocks: usize = 0;
    var lines = std.mem.splitScalar(u8, corpus, '\n');
    var seqs: std.ArrayList([]const u8) = .empty;
    defer {
        for (seqs.items) |s| allocator.free(s);
        seqs.deinit(allocator);
    }
    var expected: [4]std.ArrayList([]const u8) = .{ .empty, .empty, .empty, .empty };
    defer for (&expected) |*e| {
        for (e.items) |s| allocator.free(s);
        e.deinit(allocator);
    };
    var bad_blocks: usize = 0;
    while (lines.next()) |line| {
        if (line.len == 0 or line[0] == '#') continue;
        var f = std.mem.splitScalar(u8, line, '\t');
        const kind = f.next().?;
        if (kind[0] == 'S') {
            try seqs.append(allocator, try hexDecode(allocator, f.next().?));
            for (&expected) |*e| try e.append(allocator, try hexDecode(allocator, f.next().?));
            continue;
        }
        const form = formOf(f.next().?);
        const block = try std.fmt.parseInt(u21, f.next().?, 10);
        const want = try std.fmt.parseInt(u64, f.next().?, 16);
        var b = try common.array_mod.StringBuilder.init(allocator, 4096, 4096 * 4);
        defer b.deinit();
        var cp: u21 = block << 12;
        while (cp < (block + 1) << 12 and cp <= 0x10FFFF) : (cp += 1) {
            if (cp >= 0xD800 and cp <= 0xDFFF) continue;
            var buf: [4]u8 = undefined;
            const n = try std.unicode.utf8Encode(cp, &buf);
            try b.append(buf[0..n]);
        }
        var arr = try b.finish();
        defer arr.release();
        var out = try ctx.invokeScalar("utf8_normalize", &.{.{ .array = arr }}, .{ .utf8_normalize = .{ .form = form } });
        defer out.release();
        try out.array.data().validateFull();
        var h: u64 = 0xCBF29CE484222325;
        for (0..out.array.data().length) |i| {
            h = fnv(h, (try common.stringAt(out.array.data(), i)).?);
            h = fnv(h, "\xff");
        }
        blocks += 1;
        if (h != want) {
            bad_blocks += 1;
            std.debug.print("block {d} form {s}: hash {x}, expected {x}\n", .{ block, @tagName(form), h, want });
        }
    }
    try std.testing.expectEqual(@as(usize, 4 * 272), blocks);
    try std.testing.expectEqual(@as(usize, 0), bad_blocks);

    // Mixed sequences, as one array per form.
    var b = try common.array_mod.StringBuilder.init(allocator, seqs.items.len, 64);
    defer b.deinit();
    for (seqs.items) |s| try b.append(s);
    var arr = try b.finish();
    defer arr.release();
    var mismatches: usize = 0;
    for (forms, &expected) |form, *e| {
        var out = try ctx.invokeScalar("utf8_normalize", &.{.{ .array = arr }}, .{ .utf8_normalize = .{ .form = form } });
        defer out.release();
        try out.array.data().validateFull();
        for (e.items, 0..) |want, i| {
            const got = (try common.stringAt(out.array.data(), i)).?;
            if (std.mem.eql(u8, got, want)) continue;
            mismatches += 1;
            if (mismatches < 20) std.debug.print("{s} row {d}: got {x}, expected {x}\n", .{ @tagName(form), i, got, want });
        }
    }
    try std.testing.expectEqual(@as(usize, 4000), seqs.items.len);
    try std.testing.expectEqual(@as(usize, 0), mismatches);
}

test "utf8_normalize on nulls, sliced, chunked and scalar inputs, every string layout, unchanged inputs and invalid UTF-8 (pyarrow 26 results)" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);

    inline for (.{ .string, .large_string, .string_view }) |tag| {
        var full = try common.makeStrings(tag, allocator, &.{ "x", "e\u{301}\u{323}\u{FB01}\u{D55C}", null, "abc", "\u{1100}\u{1161}\u{11A8} \u{212B}", "" });
        defer full.release();
        var arr = try full.slice(1, 5);
        defer arr.release();
        const args = [_]Datum{.{ .array = arr }};
        try common.expectInvokeStrings(&ctx, "utf8_normalize", &args, .none, &.{ "\u{1EB9}\u{301}\u{FB01}\u{D55C}", null, "abc", "\u{AC01} \u{C5}", "" });
        try common.expectInvokeStrings(&ctx, "utf8_normalize", &args, .{ .utf8_normalize = .{ .form = .nfd } }, &.{ "e\u{323}\u{301}\u{FB01}\u{1112}\u{1161}\u{11AB}", null, "abc", "\u{1100}\u{1161}\u{11A8} A\u{30A}", "" });
        try common.expectInvokeStrings(&ctx, "utf8_normalize", &args, .{ .utf8_normalize = .{ .form = .nfkc } }, &.{ "\u{1EB9}\u{301}fi\u{D55C}", null, "abc", "\u{AC01} \u{C5}", "" });
        try common.expectInvokeStrings(&ctx, "utf8_normalize", &args, .{ .utf8_normalize = .{ .form = .nfkd } }, &.{ "e\u{323}\u{301}fi\u{1112}\u{1161}\u{11AB}", null, "abc", "\u{1100}\u{1161}\u{11A8} A\u{30A}", "" });
        var chunked = try common.chunkedAt(allocator, arr, &.{2});
        defer chunked.release();
        try common.expectInvokeStrings(&ctx, "utf8_normalize", &.{.{ .chunked = chunked }}, .none, &.{ "\u{1EB9}\u{301}\u{FB01}\u{D55C}", null, "abc", "\u{AC01} \u{C5}", "" });
    }
    // U+FDFA expands to 18 code points in NFKD (the bound the buffers are sized by).
    const s = common.Scalar.init(.{ .string = {} }, .{ .string = "\u{FDFA}" });
    var out = try ctx.invokeScalar("utf8_normalize", &.{.{ .scalar = s }}, .{ .utf8_normalize = .{ .form = .nfkd } });
    defer out.release();
    try std.testing.expectEqual(@as(usize, 18), try std.unicode.utf8CountCodepoints(out.scalar.value.string));
    // An input already in the form comes back without a copy.
    var ascii = try common.makeStrings(.string, allocator, &.{ "plain", null, "ascii" });
    defer ascii.release();
    var same = try ctx.invokeScalar("utf8_normalize", &.{.{ .array = ascii }}, .{ .utf8_normalize = .{ .form = .nfkd } });
    defer same.release();
    try std.testing.expect(same.array.data().buffers[2].data.ptr == ascii.data().buffers[2].data.ptr);
    // A row that is not UTF-8 is InvalidInput, also when the other rows need no change.
    var bad = try common.makeStrings(.string, allocator, &.{ "ok", "\xC3\x28" });
    defer bad.release();
    for (forms) |form| {
        try std.testing.expectError(error.InvalidInput, ctx.invokeScalar("utf8_normalize", &.{.{ .array = bad }}, .{ .utf8_normalize = .{ .form = form } }));
    }
    try std.testing.expectError(error.InvalidOptions, ctx.invokeScalar("utf8_normalize", &.{.{ .array = bad }}, .{ .trim = .{ .characters = "x" } }));
}
