//! zarrow's Arrow integration tester (docs/integration-suite.md), with the command set of Arrow
//! C++'s `arrow-json-integration-test`, `arrow-stream-to-file` and `arrow-file-to-stream` that
//! archery drives, plus the golden-file check over an apache/arrow-testing clone:
//!
//!   integration-runner --integration --arrow=<file> --json=<file> --mode=VALIDATE|JSON_TO_ARROW|ARROW_TO_JSON
//!   integration-runner --validate|--json-to-arrow|--arrow-to-json --arrow=<file> --json=<file>
//!   integration-runner --stream-to-file [<in.stream> <out.arrow_file>]   (default stdin / stdout)
//!   integration-runner --file-to-stream [<in.arrow_file> <out.stream>]   (default stdin / stdout)
//!   integration-runner golden <arrow-testing-dir>
//!
//! Flags take `--name=value`, `-name=value`, `--name value` or `-name value` (gflags style).
const std = @import("std");
const zarrow = @import("zarrow");

const integration = zarrow.integration;
const json = integration.json;
const golden = integration.golden;
const Allocator = std.mem.Allocator;

const usage =
    \\usage: integration-runner --integration --arrow=<file> --json=<file> --mode=VALIDATE|JSON_TO_ARROW|ARROW_TO_JSON
    \\       integration-runner --validate|--json-to-arrow|--arrow-to-json --arrow=<file> --json=<file>
    \\       integration-runner --stream-to-file [<in> <out>]
    \\       integration-runner --file-to-stream [<in> <out>]
    \\       integration-runner golden <arrow-testing-dir>
    \\
;

const Mode = enum { validate, json_to_arrow, arrow_to_json, stream_to_file, file_to_stream, golden };

pub fn main(init: std.process.Init) !u8 {
    const io = init.io;
    const gpa = init.gpa;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    var args_it = try init.minimal.args.iterateAllocator(arena);
    defer args_it.deinit();
    _ = args_it.next();
    var args: std.ArrayList([]const u8) = .empty;
    while (args_it.next()) |a| try args.append(arena, a);

    var mode: ?Mode = null;
    var arrow_path: ?[]const u8 = null;
    var json_path: ?[]const u8 = null;
    var positional: std.ArrayList([]const u8) = .empty;
    var i: usize = 0;
    while (i < args.items.len) : (i += 1) {
        const arg = args.items[i];
        if (arg.len == 0 or arg[0] != '-') {
            if (mode == null and std.mem.eql(u8, arg, "golden")) mode = .golden else try positional.append(arena, arg);
            continue;
        }
        const body = std.mem.trimStart(u8, arg, "-");
        const eq = std.mem.indexOfScalar(u8, body, '=');
        const name = if (eq) |e| body[0..e] else body;
        const needs_value = std.mem.eql(u8, name, "arrow") or std.mem.eql(u8, name, "json") or std.mem.eql(u8, name, "mode");
        var value: ?[]const u8 = if (eq) |e| body[e + 1 ..] else null;
        if (needs_value and value == null) {
            i += 1;
            if (i >= args.items.len) return fail(io, "missing value for -{s}\n" ++ usage, .{name});
            value = args.items[i];
        }
        if (std.mem.eql(u8, name, "arrow")) {
            arrow_path = value;
        } else if (std.mem.eql(u8, name, "json")) {
            json_path = value;
        } else if (std.mem.eql(u8, name, "mode")) {
            const m = value.?;
            mode = if (std.ascii.eqlIgnoreCase(m, "VALIDATE")) .validate else if (std.ascii.eqlIgnoreCase(m, "JSON_TO_ARROW")) .json_to_arrow else if (std.ascii.eqlIgnoreCase(m, "ARROW_TO_JSON")) .arrow_to_json else return fail(io, "unknown mode {s}\n", .{m});
        } else if (std.mem.eql(u8, name, "validate")) {
            mode = .validate;
        } else if (std.mem.eql(u8, name, "json-to-arrow")) {
            mode = .json_to_arrow;
        } else if (std.mem.eql(u8, name, "arrow-to-json")) {
            mode = .arrow_to_json;
        } else if (std.mem.eql(u8, name, "stream-to-file")) {
            mode = .stream_to_file;
        } else if (std.mem.eql(u8, name, "file-to-stream")) {
            mode = .file_to_stream;
        } else if (std.mem.eql(u8, name, "integration") or std.mem.startsWith(u8, name, "verbose") or std.mem.startsWith(u8, name, "validate_")) {
            // Accepted for archery compatibility; nothing to configure.
        } else if (std.mem.eql(u8, name, "help") or std.mem.eql(u8, name, "h")) {
            try writeAll(io, .stdout, usage);
            return 0;
        } else return fail(io, "unknown flag {s}\n" ++ usage, .{arg});
    }

    const m = mode orelse .validate;
    switch (m) {
        .golden => {
            if (positional.items.len != 1) return fail(io, usage, .{});
            return runGolden(io, gpa, positional.items[0]);
        },
        .stream_to_file, .file_to_stream => {
            const in = if (positional.items.len > 0) try readPath(io, gpa, positional.items[0]) else try readStdin(io, gpa);
            defer gpa.free(in);
            var doc = (if (m == .stream_to_file) golden.readStream(gpa, in) else golden.readFile(gpa, in)) catch |e|
                return fail(io, "read: {s}\n", .{@errorName(e)});
            defer doc.deinit();
            const out = golden.writeIpc(gpa, &doc, if (m == .stream_to_file) .file else .stream) catch |e|
                return fail(io, "write: {s}\n", .{@errorName(e)});
            defer gpa.free(out);
            if (positional.items.len > 1) try writePath(io, positional.items[1], out) else try writeAll(io, .stdout, out);
            return 0;
        },
        else => {},
    }
    const jp = json_path orelse return fail(io, "Must specify json file name\n", .{});
    const ap = arrow_path orelse return fail(io, "Must specify arrow file name\n", .{});
    switch (m) {
        .json_to_arrow => {
            const text = try readPath(io, gpa, jp);
            defer gpa.free(text);
            var doc = json.read(gpa, text) catch |e| return fail(io, "{s}: {s}\n", .{ jp, @errorName(e) });
            defer doc.deinit();
            const out = golden.writeIpc(gpa, &doc, .file) catch |e| return fail(io, "write {s}: {s}\n", .{ ap, @errorName(e) });
            defer gpa.free(out);
            try writePath(io, ap, out);
        },
        .arrow_to_json => {
            const bytes = try readPath(io, gpa, ap);
            defer gpa.free(bytes);
            var doc = golden.readFile(gpa, bytes) catch |e| return fail(io, "{s}: {s}\n", .{ ap, @errorName(e) });
            defer doc.deinit();
            var out: std.Io.Writer.Allocating = .init(gpa);
            defer out.deinit();
            json.write(gpa, &out.writer, doc.schema().*, doc.batches) catch |e| return fail(io, "write {s}: {s}\n", .{ jp, @errorName(e) });
            try writePath(io, jp, out.written());
        },
        .validate => {
            const text = try readPath(io, gpa, jp);
            defer gpa.free(text);
            var doc = json.read(gpa, text) catch |e| return fail(io, "{s}: {s}\n", .{ jp, @errorName(e) });
            defer doc.deinit();
            const bytes = try readPath(io, gpa, ap);
            defer gpa.free(bytes);
            var file_doc = golden.readFile(gpa, bytes) catch |e| return fail(io, "{s}: {s}\n", .{ ap, @errorName(e) });
            defer file_doc.deinit();
            var r: golden.Result = .{};
            if (!golden.same(&r, "JSON vs Arrow file", &doc, &file_doc)) return fail(io, "{s}\n", .{r.message()});
        },
        else => unreachable,
    }
    return 0;
}

// ---------------------------------------------------------------------------------------------
// golden <arrow-testing-dir>

const Row = struct {
    dir: []const u8,
    case: []const u8,
    status: []const u8,
    detail: []const u8,
};

fn runGolden(io: std.Io, gpa: Allocator, root: []const u8) !u8 {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const base = try std.fs.path.join(arena, &.{ root, "data", "arrow-ipc-stream", "integration" });
    var top = std.Io.Dir.cwd().openDir(io, base, .{ .iterate = true }) catch |e|
        return fail(io, "cannot open {s}: {s} (expected an apache/arrow-testing clone)\n", .{ base, @errorName(e) });
    defer top.close(io);

    var dirs: std.ArrayList([]const u8) = .empty;
    var it = top.iterate();
    while (try it.next(io)) |entry| {
        if (entry.kind == .directory) try dirs.append(arena, try arena.dupe(u8, entry.name));
    }
    std.mem.sort([]const u8, dirs.items, {}, lessThan);

    var rows: std.ArrayList(Row) = .empty;
    var failures: usize = 0;
    var passes: usize = 0;
    for (dirs.items) |dir_name| {
        const dir_path = try std.fs.path.join(arena, &.{ base, dir_name });
        var dir = try std.Io.Dir.cwd().openDir(io, dir_path, .{ .iterate = true });
        defer dir.close(io);
        var cases: std.ArrayList([]const u8) = .empty;
        var dit = dir.iterate();
        while (try dit.next(io)) |entry| {
            if (entry.kind != .file or !std.mem.endsWith(u8, entry.name, ".json.gz")) continue;
            try cases.append(arena, try arena.dupe(u8, entry.name[0 .. entry.name.len - ".json.gz".len]));
        }
        std.mem.sort([]const u8, cases.items, {}, lessThan);

        for (cases.items) |case| {
            const expect: golden.Expect = if (knownRejection(case) != null) .reject_known else .accept;
            const row = try runCase(io, gpa, arena, dir_path, dir_name, case, expect);
            if (std.mem.startsWith(u8, row.status, "FAIL")) failures += 1;
            if (std.mem.eql(u8, row.status, "pass")) passes += 1;
            try rows.append(arena, row);
        }
    }

    var buf: [4096]u8 = undefined;
    var out = std.Io.File.stdout().writer(io, &buf);
    const w = &out.interface;
    try w.print("| directory | case | result | detail |\n|---|---|---|---|\n", .{});
    for (rows.items) |row| try w.print("| {s} | {s} | {s} | {s} |\n", .{ row.dir, row.case, row.status, row.detail });
    try w.print("\n{d} cases: {d} pass, {d} FAIL, {d} rejected (by zarrow validation as listed)\n", .{ rows.items.len, passes, failures, rows.items.len - passes - failures });
    try w.flush();
    return if (failures == 0 and rows.items.len > 0) 0 else 1;
}

fn runCase(io: std.Io, gpa: Allocator, arena: Allocator, dir_path: []const u8, dir_name: []const u8, case: []const u8, expect: golden.Expect) !Row {
    const row: Row = .{ .dir = dir_name, .case = case, .status = "", .detail = "" };
    const gz = readPath(io, gpa, try std.fmt.allocPrint(arena, "{s}/{s}.json.gz", .{ dir_path, case })) catch |e|
        return withStatus(row, "FAIL", try std.fmt.allocPrint(arena, "read json.gz: {s}", .{@errorName(e)}));
    defer gpa.free(gz);
    const text = golden.gunzip(gpa, gz) catch |e|
        return withStatus(row, "FAIL", try std.fmt.allocPrint(arena, "gunzip: {s}", .{@errorName(e)}));
    defer gpa.free(text);
    const file = readPath(io, gpa, try std.fmt.allocPrint(arena, "{s}/{s}.arrow_file", .{ dir_path, case })) catch |e|
        return withStatus(row, "FAIL", try std.fmt.allocPrint(arena, "read arrow_file: {s}", .{@errorName(e)}));
    defer gpa.free(file);
    const stream = readPath(io, gpa, try std.fmt.allocPrint(arena, "{s}/{s}.stream", .{ dir_path, case })) catch |e|
        return withStatus(row, "FAIL", try std.fmt.allocPrint(arena, "read stream: {s}", .{@errorName(e)}));
    defer gpa.free(stream);

    const r = golden.check(gpa, text, file, stream, expect);
    const detail = try arena.dupe(u8, r.message());
    return switch (r.outcome) {
        .pass => withStatus(row, "pass", ""),
        .rejected_known => withStatus(row, "rejected-by-validation", try std.fmt.allocPrint(arena, "{s} ({s})", .{ knownRejection(case).?, detail })),
        .fail => withStatus(row, "FAIL", detail),
    };
}

/// Golden cases zarrow's IPC validation rejects on purpose; docs/integration-suite.md has the
/// analysis. Each must stay rejected (an accepted one fails, so this list cannot go stale).
const known_rejections = [_][2][]const u8{
    // (none: the unsigned dictionary index cases pass since zarrow reads and writes unsigned indices)
};

fn knownRejection(case: []const u8) ?[]const u8 {
    for (known_rejections) |k| if (std.mem.eql(u8, k[0], case)) return k[1];
    return null;
}

fn withStatus(row: Row, status: []const u8, detail: []const u8) Row {
    var r = row;
    r.status = status;
    r.detail = detail;
    return r;
}

fn lessThan(_: void, a: []const u8, b: []const u8) bool {
    return std.mem.order(u8, a, b) == .lt;
}

// ---------------------------------------------------------------------------------------------
// IO

/// Largest input read (the integration files are a few MiB at most).
const max_input: usize = 1 << 30;

fn readPath(io: std.Io, gpa: Allocator, path: []const u8) ![]u8 {
    return std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(max_input));
}

fn readStdin(io: std.Io, gpa: Allocator) ![]u8 {
    var buf: [64 * 1024]u8 = undefined;
    var in = std.Io.File.stdin().reader(io, &buf);
    return in.interface.allocRemaining(gpa, .limited(max_input));
}

fn writePath(io: std.Io, path: []const u8, bytes: []const u8) !void {
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = bytes });
}

const Stream = enum { stdout, stderr };

fn writeAll(io: std.Io, which: Stream, bytes: []const u8) !void {
    var buf: [4096]u8 = undefined;
    var out = (if (which == .stdout) std.Io.File.stdout() else std.Io.File.stderr()).writer(io, &buf);
    try out.interface.writeAll(bytes);
    try out.interface.flush();
}

fn fail(io: std.Io, comptime fmt: []const u8, args: anytype) u8 {
    var buf: [4096]u8 = undefined;
    var out = std.Io.File.stderr().writer(io, &buf);
    out.interface.print(fmt, args) catch {};
    out.interface.flush() catch {};
    return 1;
}
