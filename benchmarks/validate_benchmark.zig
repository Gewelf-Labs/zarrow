const std = @import("std");
const zarrow = @import("zarrow");
const csv_meta = @import("csv_meta.zig");

const Mode = enum { default, smoke, full, matrix, matrix_no_header, ci, ci_no_header };

const BenchConfig = struct {
    rows: usize,
    iterations: usize,
};

fn parseMode(init: std.process.Init, allocator: std.mem.Allocator) !Mode {
    var args = try init.minimal.args.iterateAllocator(allocator);
    defer args.deinit();

    _ = args.next();
    const mode_str = args.next() orelse return .default;

    if (std.mem.eql(u8, mode_str, "smoke")) return .smoke;
    if (std.mem.eql(u8, mode_str, "full")) return .full;
    if (std.mem.eql(u8, mode_str, "matrix")) return .matrix;
    if (std.mem.eql(u8, mode_str, "matrix-no-header")) return .matrix_no_header;
    if (std.mem.eql(u8, mode_str, "ci")) return .ci;
    if (std.mem.eql(u8, mode_str, "ci-no-header")) return .ci_no_header;
    return .default;
}

fn printCsvHeader() void {
    std.debug.print("benchmark,rows,iterations,elapsed_ns,rows_per_sec,ns_per_row,checksum,git_sha,timestamp\n", .{});
}

/// Times validateFull on `arr`; ns_per_row / 16 is ns per string byte for the string case.
fn timeValidate(io: std.Io, name: []const u8, arr: zarrow.ArrayRef, cfg: BenchConfig, emit_csv: bool, meta: csv_meta.CsvMeta) !void {
    var checksum: usize = 0;
    const timer_start = std.Io.Timestamp.now(io, .awake);
    for (0..cfg.iterations) |_| {
        try arr.data().validateFull();
        checksum += arr.data().length;
    }
    const elapsed_ns: u64 = @max(1, @as(u64, @intCast(timer_start.untilNow(io, .awake).toNanoseconds())));
    const total_rows = cfg.rows * cfg.iterations;
    const rows_per_sec = (@as(f64, @floatFromInt(total_rows)) * 1_000_000_000.0) / @as(f64, @floatFromInt(elapsed_ns));
    const ns_per_row = @as(f64, @floatFromInt(elapsed_ns)) / @as(f64, @floatFromInt(total_rows));
    if (emit_csv) {
        std.debug.print("{s},{d},{d},{d},{d:.2},{d:.2},{d},{s},{d}\n", .{ name, cfg.rows, cfg.iterations, elapsed_ns, rows_per_sec, ns_per_row, checksum, meta.git_sha, meta.timestamp });
        return;
    }
    std.debug.print("benchmark={s} rows={d} iterations={d} elapsed_ns={d} rows_per_sec={d:.2} ns_per_row={d:.2} checksum={d}\n", .{ name, cfg.rows, cfg.iterations, elapsed_ns, rows_per_sec, ns_per_row, checksum });
}

fn runBenchmark(io: std.Io, allocator: std.mem.Allocator, cfg: BenchConfig, emit_csv: bool, meta: csv_meta.CsvMeta) !void {
    // 16-byte strings, ASCII except every 50th row (two 2-byte chars and one 3-byte char), every 17th null.
    var sb = try zarrow.StringBuilder.init(allocator, cfg.rows, cfg.rows * 16);
    defer sb.deinit();
    var tmp: [16]u8 = undefined;
    for (0..cfg.rows) |i| {
        if (i % 17 == 0) {
            try sb.appendNull();
            continue;
        }
        const s = if (i % 50 == 0)
            std.fmt.bufPrint(&tmp, "\u{e9}t\u{e9}-\u{20ac}{d:0>6}", .{i % 1_000_000}) catch unreachable
        else
            std.fmt.bufPrint(&tmp, "row-value-{d:0>6}", .{i % 1_000_000}) catch unreachable;
        try sb.append(s);
    }
    var strs = try sb.finish();
    defer strs.release();
    try timeValidate(io, "validate_full_string", strs, cfg, emit_csv, meta);

    // list<int32>: 0..3 items per row, every 11th row null (structural checks only).
    var ib = try zarrow.Int32Builder.init(allocator, cfg.rows * 2);
    defer ib.deinit();
    const int_type = zarrow.DataType{ .int32 = {} };
    const field = zarrow.Field{ .name = "a", .data_type = &int_type, .nullable = true };
    var lb = try zarrow.ListBuilder.init(allocator, cfg.rows, field);
    defer lb.deinit();
    var used: usize = 0;
    for (0..cfg.rows) |i| {
        if (i % 11 == 0) {
            try lb.appendNull();
            continue;
        }
        const n = i % 4;
        try lb.appendLen(n);
        for (0..n) |k| if ((used + k) % 7 == 0) try ib.appendNull() else try ib.append(@intCast(used + k));
        used += n;
    }
    var values = try ib.finish();
    defer values.release();
    var lists = try lb.finish(values);
    defer lists.release();
    try timeValidate(io, "validate_full_int32_list", lists, cfg, emit_csv, meta);
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const allocator = std.heap.smp_allocator;
    const mode = try parseMode(init, allocator);
    const meta = try csv_meta.resolve(init, allocator);
    defer meta.deinit(allocator);

    switch (mode) {
        .smoke => try runBenchmark(io, allocator, .{ .rows = 10_000, .iterations = 20 }, false, meta),
        .full => try runBenchmark(io, allocator, .{ .rows = 1_000_000, .iterations = 200 }, false, meta),
        .matrix, .matrix_no_header => {
            if (mode == .matrix) printCsvHeader();
            try runBenchmark(io, allocator, .{ .rows = 1_000, .iterations = 2_000 }, true, meta);
            try runBenchmark(io, allocator, .{ .rows = 10_000, .iterations = 500 }, true, meta);
            try runBenchmark(io, allocator, .{ .rows = 100_000, .iterations = 100 }, true, meta);
        },
        .ci, .ci_no_header => {
            if (mode == .ci) printCsvHeader();
            try runBenchmark(io, allocator, .{ .rows = 1_000, .iterations = 200 }, true, meta);
            try runBenchmark(io, allocator, .{ .rows = 10_000, .iterations = 50 }, true, meta);
            try runBenchmark(io, allocator, .{ .rows = 100_000, .iterations = 10 }, true, meta);
        },
        .default => try runBenchmark(io, allocator, .{ .rows = 200_000, .iterations = 500 }, false, meta),
    }
}
