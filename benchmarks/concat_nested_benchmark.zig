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

/// Cuts `source` into 16-row parts (unaligned bit offsets), as filter/take/chunk flattening
/// produce, and times concatArrayRefs over all parts.
fn timeConcat(io: std.Io, allocator: std.mem.Allocator, name: []const u8, source: zarrow.ArrayRef, cfg: BenchConfig, emit_csv: bool, meta: csv_meta.CsvMeta) !void {
    const part_rows: usize = 16;
    const part_count = (cfg.rows + part_rows - 1) / part_rows;
    const parts = try allocator.alloc(zarrow.ArrayRef, part_count);
    defer allocator.free(parts);
    for (parts, 0..) |*part, p| {
        const start = p * part_rows;
        part.* = try source.slice(start, @min(part_rows, cfg.rows - start));
    }
    defer for (parts) |*part| part.release();

    var checksum: usize = 0;
    const timer_start = std.Io.Timestamp.now(io, .awake);
    for (0..cfg.iterations) |_| {
        var out = try zarrow.concatArrayRefs(allocator, source.data().data_type, parts);
        defer out.release();
        checksum += out.data().length + (out.data().null_count orelse 0);
    }

    const elapsed_ns: u64 = @intCast(timer_start.untilNow(io, .awake).toNanoseconds());
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
    // Nullable int32 (every 7th null) + string columns.
    var ints_b = try zarrow.Int32Builder.init(allocator, cfg.rows);
    defer ints_b.deinit();
    var strs_b = try zarrow.StringBuilder.init(allocator, cfg.rows, cfg.rows * 8);
    defer strs_b.deinit();
    const names = [_][]const u8{ "alpha", "beta", "gamma", "delta-long-name" };
    for (0..cfg.rows) |i| {
        if (i % 7 == 0) {
            try ints_b.appendNull();
            try strs_b.appendNull();
        } else {
            try ints_b.append(@intCast(i));
            try strs_b.append(names[i & 3]);
        }
    }
    var ints = try ints_b.finish();
    defer ints.release();
    var strs = try strs_b.finish();
    defer strs.release();

    // list<int32>: 0..3 items per row (the int column as values), every 11th row null.
    const int_type = zarrow.DataType{ .int32 = {} };
    const str_type = zarrow.DataType{ .string = {} };
    const fields = [_]zarrow.Field{
        .{ .name = "a", .data_type = &int_type, .nullable = true },
        .{ .name = "b", .data_type = &str_type, .nullable = true },
    };
    var lb = try zarrow.ListBuilder.init(allocator, cfg.rows, fields[0]);
    defer lb.deinit();
    var used: usize = 0;
    for (0..cfg.rows) |i| {
        const n = @min(i % 4, cfg.rows - used);
        if (i % 11 == 0) try lb.appendNull() else try lb.appendLen(n);
        if (i % 11 != 0) used += n;
    }
    var values = try ints.slice(0, used);
    defer values.release();
    var lists = try lb.finish(values);
    defer lists.release();
    try timeConcat(io, allocator, "concat_nested_list", lists, cfg, emit_csv, meta);

    // struct{a: int32, b: string}, every 11th row null.
    var sb = zarrow.StructBuilder.init(allocator, &fields);
    defer sb.deinit();
    for (0..cfg.rows) |i| if (i % 11 == 0) try sb.appendNull() else try sb.appendValid();
    var structs = try sb.finish(&.{ ints, strs });
    defer structs.release();
    try timeConcat(io, allocator, "concat_nested_struct", structs, cfg, emit_csv, meta);

    // string_view: the string column's values (one in four longer than 12 bytes, out of line).
    var svb = try zarrow.StringViewBuilder.init(allocator, cfg.rows, cfg.rows * 4);
    defer svb.deinit();
    for (0..cfg.rows) |i| if (i % 7 == 0) try svb.appendNull() else try svb.append(names[i & 3]);
    var views = try svb.finish();
    defer views.release();
    try timeConcat(io, allocator, "concat_nested_string_view", views, cfg, emit_csv, meta);

    // dense_union<int32 = 5, string = 7>: alternating, child row i / 2.
    const union_ids = [_]i8{ 5, 7 };
    var ub = try zarrow.DenseUnionBuilder.init(allocator, .{ .type_ids = &union_ids, .fields = &fields, .mode = .dense }, cfg.rows);
    defer ub.deinit();
    for (0..cfg.rows) |i| try ub.append(union_ids[i & 1], @intCast(i / 2));
    var unions = try ub.finish(&.{ ints, strs });
    defer unions.release();
    try timeConcat(io, allocator, "concat_nested_dense_union", unions, cfg, emit_csv, meta);

    // run_end_encoded<int32, int32>: runs of 1-8 rows over the int32 column's values.
    var reb = try zarrow.RunEndEncodedBuilder.init(allocator, .{ .bit_width = 32, .signed = true }, &int_type, cfg.rows);
    defer reb.deinit();
    var end: usize = 0;
    var runs: usize = 0;
    while (end < cfg.rows) : (runs += 1) {
        end = @min(cfg.rows, end + 1 + (runs * 7919) % 8);
        try reb.appendRunEnd(@intCast(end));
    }
    var run_values = try ints.slice(0, runs);
    defer run_values.release();
    var rees = try reb.finish(run_values);
    defer rees.release();
    try timeConcat(io, allocator, "concat_nested_ree", rees, cfg, emit_csv, meta);
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    // smp_allocator: small per-part/per-batch allocations; page_allocator would mostly measure syscalls.
    const allocator = std.heap.smp_allocator;
    const mode = try parseMode(init, allocator);
    const meta = try csv_meta.resolve(init, allocator);
    defer meta.deinit(allocator);

    switch (mode) {
        .smoke => try runBenchmark(io, allocator, .{ .rows = 10_000, .iterations = 20 }, false, meta),
        .full => try runBenchmark(io, allocator, .{ .rows = 500_000, .iterations = 120 }, false, meta),
        .matrix => {
            printCsvHeader();
            try runBenchmark(io, allocator, .{ .rows = 1_000, .iterations = 500 }, true, meta);
            try runBenchmark(io, allocator, .{ .rows = 10_000, .iterations = 120 }, true, meta);
            try runBenchmark(io, allocator, .{ .rows = 100_000, .iterations = 20 }, true, meta);
        },
        .matrix_no_header => {
            try runBenchmark(io, allocator, .{ .rows = 1_000, .iterations = 500 }, true, meta);
            try runBenchmark(io, allocator, .{ .rows = 10_000, .iterations = 120 }, true, meta);
            try runBenchmark(io, allocator, .{ .rows = 100_000, .iterations = 20 }, true, meta);
        },
        .ci => {
            printCsvHeader();
            try runBenchmark(io, allocator, .{ .rows = 1_000, .iterations = 80 }, true, meta);
            try runBenchmark(io, allocator, .{ .rows = 10_000, .iterations = 20 }, true, meta);
            try runBenchmark(io, allocator, .{ .rows = 100_000, .iterations = 4 }, true, meta);
        },
        .ci_no_header => {
            try runBenchmark(io, allocator, .{ .rows = 1_000, .iterations = 80 }, true, meta);
            try runBenchmark(io, allocator, .{ .rows = 10_000, .iterations = 20 }, true, meta);
            try runBenchmark(io, allocator, .{ .rows = 100_000, .iterations = 4 }, true, meta);
        },
        .default => try runBenchmark(io, allocator, .{ .rows = 200_000, .iterations = 80 }, false, meta),
    }
}
