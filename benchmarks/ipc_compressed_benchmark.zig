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

fn buildSources(allocator: std.mem.Allocator, rows: usize) !struct { zarrow.ArrayRef, zarrow.ArrayRef } {
    var ints = try zarrow.Int32Builder.init(allocator, rows);
    defer ints.deinit();
    var strs = try zarrow.StringBuilder.init(allocator, rows, rows * 8);
    defer strs.deinit();
    const names = [_][]const u8{ "alpha", "beta", "gamma", "delta-long-name" };
    for (0..rows) |i| {
        if (i % 7 == 0) {
            try ints.appendNull();
            try strs.appendNull();
        } else {
            try ints.append(@intCast(i));
            try strs.append(names[i & 3]);
        }
    }
    var int_ref = try ints.finish();
    errdefer int_ref.release();
    return .{ int_ref, try strs.finish() };
}

fn runBenchmark(io: std.Io, allocator: std.mem.Allocator, cfg: BenchConfig, emit_csv: bool, meta: csv_meta.CsvMeta) !void {
    // zstd-compressed stream round trip (write + copying read) of an int32 + string batch.
    const int32_type = zarrow.DataType{ .int32 = {} };
    const string_type = zarrow.DataType{ .string = {} };
    const fields = [_]zarrow.Field{
        .{ .name = "id", .data_type = &int32_type, .nullable = true },
        .{ .name = "name", .data_type = &string_type, .nullable = true },
    };
    const schema = zarrow.Schema{ .fields = fields[0..] };
    var schema_ref = try zarrow.SchemaRef.fromBorrowed(allocator, schema);
    defer schema_ref.release();
    var ints, var strs = try buildSources(allocator, cfg.rows);
    defer ints.release();
    defer strs.release();
    var batch = try zarrow.RecordBatch.init(allocator, schema_ref.retain(), &[_]zarrow.ArrayRef{ ints, strs });
    defer batch.deinit();

    // One sink reused across iterations: times zarrow, not first-touch page faults of a fresh sink.
    var sink: std.Io.Writer.Allocating = try .initCapacity(allocator, cfg.rows * 16 + 4096);
    defer sink.deinit();

    var checksum: usize = 0;
    const timer_start = std.Io.Timestamp.now(io, .awake);

    var iter: usize = 0;
    while (iter < cfg.iterations) : (iter += 1) {
        sink.clearRetainingCapacity();
        {
            var writer = zarrow.IpcStreamWriter.initWithBodyCompression(allocator, &sink.writer, .zstd);
            defer writer.deinit();
            try writer.writeSchema(schema);
            try writer.writeRecordBatch(batch);
            try writer.writeEnd();
        }
        var source: std.Io.Reader = .fixed(sink.written());
        var reader = zarrow.IpcStreamReader.init(allocator, &source);
        defer reader.deinit();
        _ = try reader.readSchema();
        while (try reader.nextRecordBatch()) |*rb| {
            var out_batch = rb.*;
            defer out_batch.deinit();
            checksum += out_batch.numRows();
        }
    }

    const elapsed_ns: u64 = @intCast(timer_start.untilNow(io, .awake).toNanoseconds());
    const total_rows = cfg.rows * cfg.iterations;
    const rows_per_sec = (@as(f64, @floatFromInt(total_rows)) * 1_000_000_000.0) / @as(f64, @floatFromInt(elapsed_ns));
    const ns_per_row = @as(f64, @floatFromInt(elapsed_ns)) / @as(f64, @floatFromInt(total_rows));

    if (emit_csv) {
        std.debug.print("ipc_compressed,{d},{d},{d},{d:.2},{d:.2},{d},{s},{d}\n", .{
            cfg.rows,
            cfg.iterations,
            elapsed_ns,
            rows_per_sec,
            ns_per_row,
            checksum,
            meta.git_sha,
            meta.timestamp,
        });
        return;
    }

    std.debug.print("benchmark=ipc_compressed rows={d} iterations={d} elapsed_ns={d} rows_per_sec={d:.2} ns_per_row={d:.2} checksum={d}\n", .{
        cfg.rows,
        cfg.iterations,
        elapsed_ns,
        rows_per_sec,
        ns_per_row,
        checksum,
    });
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
