const std = @import("std");

const max_bytes: usize = 16 * 1024 * 1024;

/// Converts backslashes to '/' on `@import("...")` lines only, so Windows
/// path bugs are fixed without touching other backslashes (e.g. `\n` in comments).
fn patchImportBackslashes(gpa: std.mem.Allocator, src: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    var lines = std.mem.splitScalar(u8, src, '\n');
    var first = true;
    while (lines.next()) |line| {
        if (!first) try out.append(gpa, '\n');
        first = false;
        const is_import = std.mem.indexOf(u8, line, "@import(") != null;
        for (line) |c| try out.append(gpa, if (is_import and c == '\\') '/' else c);
    }
    return out.toOwnedSlice(gpa);
}

fn patchUnionMissingSemicolon(gpa: std.mem.Allocator, src: []const u8) ![]u8 {
    return std.mem.replaceOwned(
        u8,
        gpa,
        src,
        "const obj2 = try obj_packed.Unpack(opts)\n",
        "const obj2 = try obj_packed.Unpack(opts);\n",
    );
}

/// Index of the `)` closing a group whose contents start at `from`, or null.
fn matchingParen(src: []const u8, from: usize) ?usize {
    var depth: usize = 1;
    var j = from;
    while (j < src.len) : (j += 1) {
        switch (src[j]) {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if (depth == 0) return j;
            },
            else => {},
        }
    }
    return null;
}

/// `std.ArrayList(T){}` -> `std.ArrayList(T).empty`
/// `name: std.ArrayList(T) = .{},` -> `name: std.ArrayList(T) = .empty,`
fn rewriteArrayListEmpty(gpa: std.mem.Allocator, src: []const u8) ![]u8 {
    const needle = "std.ArrayList(";
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    var i: usize = 0;
    while (std.mem.indexOfPos(u8, src, i, needle)) |start| {
        const close = matchingParen(src, start + needle.len) orelse break;
        const rest = src[close + 1 ..];
        try out.appendSlice(gpa, src[i .. close + 1]);
        if (std.mem.startsWith(u8, rest, "{}")) {
            try out.appendSlice(gpa, ".empty");
            i = close + 3;
        } else if (std.mem.startsWith(u8, rest, " = .{}")) {
            try out.appendSlice(gpa, " = .empty");
            i = close + 7;
        } else {
            i = close + 1;
        }
    }
    try out.appendSlice(gpa, src[i..]);
    return out.toOwnedSlice(gpa);
}

/// Applies every fix to one generated file. Caller owns the returned slice.
fn fixSource(gpa: std.mem.Allocator, src: []const u8) ![]u8 {
    const a = try patchImportBackslashes(gpa, src);
    defer gpa.free(a);
    const b = try patchUnionMissingSemicolon(gpa, a);
    defer gpa.free(b);
    return rewriteArrayListEmpty(gpa, b);
}

/// Usage: fix_flatc_lib <flatc output root dir>
/// Rewrites every .zig file under the directory in place. Unchanged files are not touched.
pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const gpa = init.gpa;

    var args = try init.minimal.args.iterateAllocator(gpa);
    defer args.deinit();

    _ = args.next(); // executable name
    const root_path = args.next() orelse return error.MissingPathArgument;

    var root = try std.Io.Dir.cwd().openDir(io, root_path, .{ .iterate = true });
    defer root.close(io);

    var walker = try root.walk(gpa);
    defer walker.deinit();
    while (try walker.next(io)) |entry| {
        if (entry.kind != .file) continue;
        if (!std.mem.endsWith(u8, entry.basename, ".zig")) continue;

        const src = try entry.dir.readFileAlloc(io, entry.basename, gpa, .limited(max_bytes + 1));
        defer gpa.free(src);
        const fixed = try fixSource(gpa, src);
        defer gpa.free(fixed);
        if (std.mem.eql(u8, src, fixed)) continue;

        try entry.dir.writeFile(io, .{ .sub_path = entry.basename, .data = fixed });
    }
}
