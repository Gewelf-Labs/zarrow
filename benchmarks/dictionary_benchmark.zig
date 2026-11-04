const std = @import("std");
const zarrow = @import("zarrow");
const csv_meta = @import("csv_meta.zig");
const bench_allocator = @import("bench_allocator.zig");

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

/// Times dictionaryEncode of `arr` (int32 indices into its distinct values).
fn timeEncode(io: std.Io, allocator: std.mem.Allocator, name: []const u8, arr: zarrow.ArrayRef, value_type: *const zarrow.DataType, cfg: BenchConfig, emit_csv: bool, meta: csv_meta.CsvMeta) !void {
    // Untimed warm-up: the timed iterations reuse its output blocks (bench_allocator.zig).
    var warm_up = try zarrow.dictionaryEncode(allocator, arr, value_type);
    warm_up.release();
    var checksum: usize = 0;
    const timer_start = std.Io.Timestamp.now(io, .awake);
    for (0..cfg.iterations) |_| {
        var out = try zarrow.dictionaryEncode(allocator, arr, value_type);
        defer out.release();
        checksum += out.data().dictionary.?.data().length + (out.data().null_count orelse 0);
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

const str_type = zarrow.DataType{ .string = {} };
const int_type = zarrow.DataType{ .int32 = {} };

fn runBenchmark(io: std.Io, allocator: std.mem.Allocator, cfg: BenchConfig, emit_csv: bool, meta: csv_meta.CsvMeta) !void {
    // 1,000 distinct values in scattered order, every 13th row null.
    var sb = try zarrow.StringBuilder.init(allocator, cfg.rows, cfg.rows * 10);
    defer sb.deinit();
    var ib = try zarrow.Int32Builder.init(allocator, cfg.rows);
    defer ib.deinit();
    var tmp: [16]u8 = undefined;
    for (0..cfg.rows) |i| {
        if (i % 13 == 0) {
            try sb.appendNull();
            try ib.appendNull();
            continue;
        }
        const k = (i * 7919) % 1000;
        try sb.append(std.fmt.bufPrint(&tmp, "value-{d}", .{k}) catch unreachable);
        try ib.append(@intCast(k * 1_000_003));
    }
    var strs = try sb.finish();
    defer strs.release();
    var ints = try ib.finish();
    defer ints.release();
    try timeEncode(io, allocator, "dictionary_encode_string", strs, &str_type, cfg, emit_csv, meta);
    try timeEncode(io, allocator, "dictionary_encode_int32", ints, &int_type, cfg, emit_csv, meta);
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const allocator = bench_allocator.allocator;
    const mode = try parseMode(init, allocator);
    const meta = try csv_meta.resolve(init, allocator);
    defer meta.deinit(allocator);

    switch (mode) {
        .smoke => try runBenchmark(io, allocator, .{ .rows = 10_000, .iterations = 20 }, false, meta),
        .full => try runBenchmark(io, allocator, .{ .rows = 1_000_000, .iterations = 50 }, false, meta),
        .matrix, .matrix_no_header => {
            if (mode == .matrix) printCsvHeader();
            try runBenchmark(io, allocator, .{ .rows = 1_000, .iterations = 500 }, true, meta);
            try runBenchmark(io, allocator, .{ .rows = 10_000, .iterations = 120 }, true, meta);
            try runBenchmark(io, allocator, .{ .rows = 100_000, .iterations = 20 }, true, meta);
        },
        .ci, .ci_no_header => {
            if (mode == .ci) printCsvHeader();
            try runBenchmark(io, allocator, .{ .rows = 1_000, .iterations = 80 }, true, meta);
            try runBenchmark(io, allocator, .{ .rows = 10_000, .iterations = 20 }, true, meta);
            try runBenchmark(io, allocator, .{ .rows = 100_000, .iterations = 4 }, true, meta);
        },
        .default => try runBenchmark(io, allocator, .{ .rows = 200_000, .iterations = 50 }, false, meta),
    }
}
