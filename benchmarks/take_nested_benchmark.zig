const std = @import("std");
const zarrow = @import("zarrow");
const csv_meta = @import("csv_meta.zig");

const Mode = enum { default, smoke, full, matrix, matrix_no_header, ci, ci_no_header };

const BenchConfig = struct {
    rows: usize,
    iterations: usize,
};

const Datum = zarrow.compute.Datum;

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

fn report(name: []const u8, cfg: BenchConfig, elapsed_ns: u64, checksum: usize, emit_csv: bool, meta: csv_meta.CsvMeta) void {
    const total_rows = cfg.rows * cfg.iterations;
    const rows_per_sec = (@as(f64, @floatFromInt(total_rows)) * 1_000_000_000.0) / @as(f64, @floatFromInt(elapsed_ns));
    const ns_per_row = @as(f64, @floatFromInt(elapsed_ns)) / @as(f64, @floatFromInt(total_rows));
    if (emit_csv) {
        std.debug.print("{s},{d},{d},{d},{d:.2},{d:.2},{d},{s},{d}\n", .{ name, cfg.rows, cfg.iterations, elapsed_ns, rows_per_sec, ns_per_row, checksum, meta.git_sha, meta.timestamp });
        return;
    }
    std.debug.print("benchmark={s} rows={d} iterations={d} elapsed_ns={d} rows_per_sec={d:.2} ns_per_row={d:.2} checksum={d}\n", .{ name, cfg.rows, cfg.iterations, elapsed_ns, rows_per_sec, ns_per_row, checksum });
}

/// Takes `indices` from `source` `cfg.iterations` times and reports ns per taken row.
fn timeTake(io: std.Io, name: []const u8, source: Datum, indices: []const usize, cfg: BenchConfig, emit_csv: bool, meta: csv_meta.CsvMeta) !void {
    var checksum: usize = 0;
    const start = std.Io.Timestamp.now(io, .awake);
    for (0..cfg.iterations) |_| {
        var out = try zarrow.compute.datumTake(source, indices);
        defer out.release();
        checksum += if (out.asArray()) |a| a.data().null_count orelse 0 else out.chunked.len();
    }
    report(name, cfg, @intCast(start.untilNow(io, .awake).toNanoseconds()), checksum, emit_csv, meta);
}

fn timeFilter(io: std.Io, name: []const u8, source: Datum, predicate: Datum, cfg: BenchConfig, emit_csv: bool, meta: csv_meta.CsvMeta) !void {
    var checksum: usize = 0;
    const start = std.Io.Timestamp.now(io, .awake);
    for (0..cfg.iterations) |_| {
        var out = try zarrow.compute.datumFilter(source, predicate, .{});
        defer out.release();
        checksum += out.array.data().length;
    }
    report(name, cfg, @intCast(start.untilNow(io, .awake).toNanoseconds()), checksum, emit_csv, meta);
}

fn runBenchmark(io: std.Io, allocator: std.mem.Allocator, cfg: BenchConfig, emit_csv: bool, meta: csv_meta.CsvMeta) !void {
    // Nullable int32 (every 7th null) + string columns, a random index permutation with repeats.
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
    const indices = try allocator.alloc(usize, cfg.rows);
    defer allocator.free(indices);
    for (indices, 0..) |*idx, i| idx.* = (i * 7919 + 13) % cfg.rows;

    // struct{a: int32, b: string}, every 11th row null.
    const int_type = zarrow.DataType{ .int32 = {} };
    const str_type = zarrow.DataType{ .string = {} };
    const fields = [_]zarrow.Field{
        .{ .name = "a", .data_type = &int_type, .nullable = true },
        .{ .name = "b", .data_type = &str_type, .nullable = true },
    };
    var sb = zarrow.StructBuilder.init(allocator, &fields);
    defer sb.deinit();
    for (0..cfg.rows) |i| if (i % 11 == 0) try sb.appendNull() else try sb.appendValid();
    var structs = try sb.finish(&.{ ints, strs });
    defer structs.release();
    try timeTake(io, "take_nested_struct", Datum.fromArray(structs), indices, cfg, emit_csv, meta);

    // dictionary<int32, string> over 16 values, every 7th index null.
    var dict_b = try zarrow.StringBuilder.init(allocator, 16, 16 * 8);
    defer dict_b.deinit();
    for (0..16) |k| try dict_b.append(names[k & 3]);
    var dict = try dict_b.finish();
    defer dict.release();
    var db = try zarrow.DictionaryBuilder.init(allocator, .{ .bit_width = 32, .signed = true }, &str_type, cfg.rows);
    defer db.deinit();
    for (0..cfg.rows) |i| if (i % 7 == 0) try db.appendNull() else try db.appendIndex(@intCast(i % 16));
    var dicts = try db.finish(dict);
    defer dicts.release();
    try timeTake(io, "take_nested_dictionary", Datum.fromArray(dicts), indices, cfg, emit_csv, meta);

    // fixed_size_list<int32, 4>, every 11th row null.
    var fvals_b = try zarrow.Int32Builder.init(allocator, cfg.rows * 4);
    defer fvals_b.deinit();
    for (0..cfg.rows * 4) |k| if (k % 13 == 0) try fvals_b.appendNull() else try fvals_b.append(@intCast(k));
    var fvals = try fvals_b.finish();
    defer fvals.release();
    var fb = try zarrow.FixedSizeListBuilder.init(allocator, fields[0], 4);
    defer fb.deinit();
    for (0..cfg.rows) |i| if (i % 11 == 0) try fb.appendNull() else try fb.appendValid();
    var fsl = try fb.finish(fvals);
    defer fsl.release();
    try timeTake(io, "take_nested_fixed_size_list", Datum.fromArray(fsl), indices, cfg, emit_csv, meta);

    // The int32 column as 16 chunks: random take, and a ~50% filter (nulls dropped) next to the
    // same filter on the flat column.
    const chunk_count = 16;
    var chunk_refs: [chunk_count]zarrow.ArrayRef = undefined;
    for (&chunk_refs, 0..) |*c, k| {
        const lo = k * cfg.rows / chunk_count;
        c.* = try ints.slice(lo, (k + 1) * cfg.rows / chunk_count - lo);
    }
    defer for (&chunk_refs) |*c| c.release();
    var chunked = try zarrow.ChunkedArray.init(allocator, .{ .int32 = {} }, &chunk_refs);
    defer chunked.release();
    try timeTake(io, "take_nested_chunked_take", Datum.fromChunked(chunked), indices, cfg, emit_csv, meta);

    var pred_b = try zarrow.BooleanBuilder.init(allocator, cfg.rows);
    defer pred_b.deinit();
    for (0..cfg.rows) |i| if (i % 11 == 0) try pred_b.appendNull() else try pred_b.append((i * 7919) % 2 == 0);
    var pred = try pred_b.finish();
    defer pred.release();
    try timeFilter(io, "take_nested_chunked_filter", Datum.fromChunked(chunked), Datum.fromArray(pred), cfg, emit_csv, meta);
    try timeFilter(io, "take_nested_flat_filter", Datum.fromArray(ints), Datum.fromArray(pred), cfg, emit_csv, meta);
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
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
