const std = @import("std");
const zarrow = @import("zarrow");
const csv_meta = @import("csv_meta.zig");

const Mode = enum { default, smoke, full, matrix, matrix_no_header, ci, ci_no_header };

const BenchConfig = struct {
    rows: usize,
    iterations: usize,
};

/// Record batches written (and read back) per iteration.
const batches_per_iter: usize = 8;

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

/// Builds one batch: id int32 (every 7th null), value int64, ok bool, name utf8.
fn buildBatch(allocator: std.mem.Allocator, schema_ref: zarrow.SchemaRef, rows: usize) !zarrow.RecordBatch {
    var id_builder = try zarrow.Int32Builder.init(allocator, rows);
    defer id_builder.deinit();
    var value_builder = try zarrow.Int64Builder.init(allocator, rows);
    defer value_builder.deinit();
    var ok_builder = try zarrow.BooleanBuilder.init(allocator, rows);
    defer ok_builder.deinit();
    var name_builder = try zarrow.StringBuilder.init(allocator, rows, rows * 8);
    defer name_builder.deinit();

    const names = [_][]const u8{ "alpha", "beta", "gamma", "delta-long-name" };
    for (0..rows) |i| {
        if (i % 7 == 0) try id_builder.appendNull() else try id_builder.append(@intCast(i));
        try value_builder.append(@intCast(i * 3));
        try ok_builder.append((i & 1) == 0);
        try name_builder.append(names[i & 3]);
    }

    var id_ref = try id_builder.finish();
    defer id_ref.release();
    var value_ref = try value_builder.finish();
    defer value_ref.release();
    var ok_ref = try ok_builder.finish();
    defer ok_ref.release();
    var name_ref = try name_builder.finish();
    defer name_ref.release();

    return zarrow.RecordBatch.init(allocator, schema_ref.retain(), &[_]zarrow.ArrayRef{ id_ref, value_ref, ok_ref, name_ref });
}

fn runBenchmark(io: std.Io, allocator: std.mem.Allocator, cfg: BenchConfig, emit_csv: bool, meta: csv_meta.CsvMeta) !void {
    const int32_type = zarrow.DataType{ .int32 = {} };
    const int64_type = zarrow.DataType{ .int64 = {} };
    const bool_type = zarrow.DataType{ .bool = {} };
    const string_type = zarrow.DataType{ .string = {} };
    const fields = [_]zarrow.Field{
        .{ .name = "id", .data_type = &int32_type, .nullable = true },
        .{ .name = "value", .data_type = &int64_type, .nullable = false },
        .{ .name = "ok", .data_type = &bool_type, .nullable = false },
        .{ .name = "name", .data_type = &string_type, .nullable = false },
    };
    const schema = zarrow.Schema{ .fields = fields[0..] };
    var schema_ref = try zarrow.SchemaRef.fromBorrowed(allocator, schema);
    defer schema_ref.release();

    var batch = try buildBatch(allocator, schema_ref, cfg.rows);
    defer batch.deinit();

    var checksum: usize = 0;
    var write_ns: u64 = 0;
    var read_ns: u64 = 0;
    var zero_copy_read_ns: u64 = 0;

    // One pre-sized sink reused across iterations: times zarrow, not Allocating growth or the
    // first-touch page faults of a fresh 5 MB sink every iteration.
    var sink: std.Io.Writer.Allocating = try .initCapacity(allocator, cfg.rows * batches_per_iter * 32 + 4096);
    defer sink.deinit();

    var iter: usize = 0;
    while (iter < cfg.iterations) : (iter += 1) {
        sink.clearRetainingCapacity();

        const write_start = std.Io.Timestamp.now(io, .awake);
        {
            var writer = zarrow.IpcStreamWriter.init(allocator, &sink.writer);
            defer writer.deinit();
            try writer.writeSchema(schema);
            for (0..batches_per_iter) |_| try writer.writeRecordBatch(batch);
            try writer.writeEnd();
        }
        write_ns += @intCast(write_start.untilNow(io, .awake).toNanoseconds());

        const read_start = std.Io.Timestamp.now(io, .awake);
        {
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
        read_ns += @intCast(read_start.untilNow(io, .awake).toNanoseconds());

        // Zero-copy read of the same bytes from an owning SharedBuffer (copy done outside the timer).
        var owned = try zarrow.OwnedBuffer.initUninit(allocator, sink.written().len);
        @memcpy(owned.data[0..sink.written().len], sink.written());
        var source = try owned.toShared(sink.written().len);
        defer source.release();
        const zc_start = std.Io.Timestamp.now(io, .awake);
        {
            var fixed: std.Io.Reader = .fixed(source.data);
            var reader = zarrow.IpcStreamReader.initZeroCopy(allocator, &fixed, source);
            defer reader.deinit();
            _ = try reader.readSchema();
            while (try reader.nextRecordBatch()) |*rb| {
                var out_batch = rb.*;
                defer out_batch.deinit();
                checksum += out_batch.numRows();
            }
        }
        zero_copy_read_ns += @intCast(zc_start.untilNow(io, .awake).toNanoseconds());
    }

    const elapsed_ns = write_ns + read_ns;
    const total_rows = cfg.rows * cfg.iterations * batches_per_iter;
    const rows_per_sec = (@as(f64, @floatFromInt(total_rows)) * 1_000_000_000.0) / @as(f64, @floatFromInt(elapsed_ns));
    const ns_per_row = @as(f64, @floatFromInt(elapsed_ns)) / @as(f64, @floatFromInt(total_rows));

    if (emit_csv) {
        std.debug.print("ipc_roundtrip,{d},{d},{d},{d:.2},{d:.2},{d},{s},{d}\n", .{
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

    std.debug.print("benchmark=ipc_roundtrip rows={d} iterations={d} batches_per_iter={d} elapsed_ns={d} write_ns={d} read_ns={d} zero_copy_read_ns={d} rows_per_sec={d:.2} ns_per_row={d:.2} checksum={d}\n", .{
        cfg.rows,
        cfg.iterations,
        batches_per_iter,
        elapsed_ns,
        write_ns,
        read_ns,
        zero_copy_read_ns,
        rows_per_sec,
        ns_per_row,
        checksum,
    });
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    // smp_allocator: per-batch IPC allocations are small; page_allocator would mostly measure syscalls.
    const allocator = std.heap.smp_allocator;
    const mode = try parseMode(init, allocator);
    const meta = try csv_meta.resolve(init, allocator);
    defer meta.deinit(allocator);

    switch (mode) {
        .smoke => try runBenchmark(io, allocator, .{ .rows = 1_000, .iterations = 20 }, false, meta),
        .full => try runBenchmark(io, allocator, .{ .rows = 100_000, .iterations = 100 }, false, meta),
        .matrix => {
            printCsvHeader();
            try runBenchmark(io, allocator, .{ .rows = 100, .iterations = 500 }, true, meta);
            try runBenchmark(io, allocator, .{ .rows = 10_000, .iterations = 100 }, true, meta);
            try runBenchmark(io, allocator, .{ .rows = 100_000, .iterations = 20 }, true, meta);
        },
        .matrix_no_header => {
            try runBenchmark(io, allocator, .{ .rows = 100, .iterations = 500 }, true, meta);
            try runBenchmark(io, allocator, .{ .rows = 10_000, .iterations = 100 }, true, meta);
            try runBenchmark(io, allocator, .{ .rows = 100_000, .iterations = 20 }, true, meta);
        },
        .ci => {
            printCsvHeader();
            try runBenchmark(io, allocator, .{ .rows = 100, .iterations = 80 }, true, meta);
            try runBenchmark(io, allocator, .{ .rows = 10_000, .iterations = 20 }, true, meta);
        },
        .ci_no_header => {
            try runBenchmark(io, allocator, .{ .rows = 100, .iterations = 80 }, true, meta);
            try runBenchmark(io, allocator, .{ .rows = 10_000, .iterations = 20 }, true, meta);
        },
        .default => try runBenchmark(io, allocator, .{ .rows = 20_000, .iterations = 50 }, false, meta),
    }
}
