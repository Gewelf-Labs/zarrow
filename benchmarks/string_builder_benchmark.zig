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

const value_len = 16;
const pool_len = 1024;

/// 1024 distinct 16-byte values ("value-0000000000" style), appended round-robin.
fn makePool() [pool_len][value_len]u8 {
    var pool: [pool_len][value_len]u8 = undefined;
    for (&pool, 0..) |*v, i| {
        _ = std.fmt.bufPrint(v, "value-{d:0>10}", .{i * 7919}) catch unreachable;
    }
    return pool;
}

fn report(name: []const u8, cfg: BenchConfig, elapsed: u64, checksum: usize, emit_csv: bool, meta: csv_meta.CsvMeta) void {
    const elapsed_ns = @max(1, elapsed);
    const total_rows = cfg.rows * cfg.iterations;
    const rows_per_sec = (@as(f64, @floatFromInt(total_rows)) * 1_000_000_000.0) / @as(f64, @floatFromInt(elapsed_ns));
    const ns_per_row = @as(f64, @floatFromInt(elapsed_ns)) / @as(f64, @floatFromInt(total_rows));
    if (emit_csv) {
        std.debug.print("{s},{d},{d},{d},{d:.2},{d:.2},{d},{s},{d}\n", .{ name, cfg.rows, cfg.iterations, elapsed_ns, rows_per_sec, ns_per_row, checksum, meta.git_sha, meta.timestamp });
        return;
    }
    std.debug.print("benchmark={s} rows={d} iterations={d} elapsed_ns={d} rows_per_sec={d:.2} ns_per_row={d:.2} checksum={d}\n", .{ name, cfg.rows, cfg.iterations, elapsed_ns, rows_per_sec, ns_per_row, checksum });
}

/// Appends `rows` values (every 10th one null when `nullable`) per iteration, then finishReset.
/// The builder is created once, as a streaming producer would hold it.
fn timeBuilder(comptime Builder: type, io: std.Io, allocator: std.mem.Allocator, name: []const u8, pool: *const [pool_len][value_len]u8, nullable: bool, cfg: BenchConfig, emit_csv: bool, meta: csv_meta.CsvMeta) !void {
    var builder = try Builder.init(allocator, cfg.rows, cfg.rows * value_len);
    defer builder.deinit();

    var checksum: usize = 0;
    const timer_start = std.Io.Timestamp.now(io, .awake);
    for (0..cfg.iterations) |_| {
        for (0..cfg.rows) |i| {
            if (nullable and i % 10 == 0) {
                try builder.appendNull();
            } else {
                try builder.append(&pool[i % pool_len]);
            }
        }
        var arr = try builder.finishReset();
        defer arr.release();
        checksum += arr.data().length + (arr.data().null_count orelse 0);
    }
    const elapsed_ns: u64 = @intCast(timer_start.untilNow(io, .awake).toNanoseconds());
    report(name, cfg, elapsed_ns, checksum, emit_csv, meta);
}

fn runBenchmark(io: std.Io, allocator: std.mem.Allocator, cfg: BenchConfig, emit_csv: bool, meta: csv_meta.CsvMeta) !void {
    const pool = makePool();
    try timeBuilder(zarrow.StringBuilder, io, allocator, "string_builder", &pool, false, cfg, emit_csv, meta);
    try timeBuilder(zarrow.StringBuilder, io, allocator, "string_builder_nullable", &pool, true, cfg, emit_csv, meta);
    try timeBuilder(zarrow.StringViewBuilder, io, allocator, "string_view_builder", &pool, false, cfg, emit_csv, meta);
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const allocator = std.heap.smp_allocator;
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
