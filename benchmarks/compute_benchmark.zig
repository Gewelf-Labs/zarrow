const std = @import("std");
const zarrow = @import("zarrow");
const csv_meta = @import("csv_meta.zig");

const compute = zarrow.compute;
const Datum = compute.Datum;
const Options = compute.Options;
const ExecContext = compute.ExecContext;

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

fn report(name: []const u8, cfg: BenchConfig, elapsed: u64, checksum: u64, emit_csv: bool, meta: csv_meta.CsvMeta) void {
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

/// Folds a Datum into a checksum: array -> length + null count + first/last value bytes; scalar -> value bits.
fn fold(d: Datum) u64 {
    switch (d) {
        .array => |arr| {
            const data = arr.data();
            var sum: u64 = data.length + (data.null_count orelse 0);
            if (data.buffers.len > 1 and data.buffers[1].len() >= 4) {
                const bytes = data.buffers[1].data;
                sum += std.mem.readInt(u32, bytes[0..4], .little);
                sum += std.mem.readInt(u32, bytes[bytes.len - 4 ..][0..4], .little);
            }
            return sum;
        },
        .scalar => |s| return switch (s.value) {
            .i64 => |v| @bitCast(v),
            .f64 => |v| @bitCast(v),
            else => 0,
        },
        .chunked => return 0,
    }
}

fn timeKernel(io: std.Io, ctx: *ExecContext, kind: compute.FunctionKind, bench_name: []const u8, fn_name: []const u8, args: []const Datum, cfg: BenchConfig, emit_csv: bool, meta: csv_meta.CsvMeta) !void {
    return timeKernelWith(io, ctx, kind, bench_name, fn_name, args, Options.noneValue(), cfg, emit_csv, meta);
}

fn timeKernelWith(io: std.Io, ctx: *ExecContext, kind: compute.FunctionKind, bench_name: []const u8, fn_name: []const u8, args: []const Datum, options: Options, cfg: BenchConfig, emit_csv: bool, meta: csv_meta.CsvMeta) !void {
    var checksum: u64 = 0;
    const timer_start = std.Io.Timestamp.now(io, .awake);
    for (0..cfg.iterations) |_| {
        var out = try ctx.invoke(fn_name, kind, args, options);
        defer out.release();
        checksum +%= fold(out);
    }
    const elapsed_ns: u64 = @intCast(timer_start.untilNow(io, .awake).toNanoseconds());
    report(bench_name, cfg, elapsed_ns, checksum, emit_csv, meta);
}

fn buildInt32(allocator: std.mem.Allocator, rows: usize, seed: u32, null_every: usize) !zarrow.ArrayRef {
    var b = try zarrow.Int32Builder.init(allocator, rows);
    defer b.deinit();
    var x: u32 = seed;
    for (0..rows) |i| {
        x = x *% 1664525 +% 1013904223;
        if (null_every != 0 and i % null_every == 0) try b.appendNull() else try b.append(@bitCast(x >> 8));
    }
    return b.finish();
}

fn runBenchmark(io: std.Io, allocator: std.mem.Allocator, cfg: BenchConfig, emit_csv: bool, meta: csv_meta.CsvMeta) !void {
    var registry = compute.FunctionRegistry.init(allocator);
    defer registry.deinit();
    try compute.registerAll(&registry);
    var ctx = ExecContext.initWithConfig(allocator, &registry, .{ .overflow_mode = .checked });

    var a = try buildInt32(allocator, cfg.rows, 1, 0);
    defer a.release();
    var b = try buildInt32(allocator, cfg.rows, 2, 0);
    defer b.release();
    // 10% nulls, offset so the two bitmaps differ.
    var an = try buildInt32(allocator, cfg.rows, 3, 10);
    defer an.release();
    var bn = try buildInt32(allocator, cfg.rows, 4, 10);
    defer bn.release();
    var bn_shift = try bn.slice(3, cfg.rows - 3);
    defer bn_shift.release();
    var an_cut = try an.slice(0, cfg.rows - 3);
    defer an_cut.release();

    var i64b = try zarrow.Int64Builder.init(allocator, cfg.rows);
    defer i64b.deinit();
    var f64b = try zarrow.Float64Builder.init(allocator, cfg.rows);
    defer f64b.deinit();
    for (0..cfg.rows) |i| {
        if (i % 10 == 0) {
            try i64b.appendNull();
            try f64b.appendNull();
        } else {
            try i64b.append(@intCast(i * 7));
            try f64b.append(@as(f64, @floatFromInt(i)) * 0.5);
        }
    }
    var i64arr = try i64b.finish();
    defer i64arr.release();
    var f64arr = try f64b.finish();
    defer f64arr.release();

    var bool_a = try buildBool(allocator, cfg.rows, 5, 10);
    defer bool_a.release();
    var bool_b = try buildBool(allocator, cfg.rows, 6, 7);
    defer bool_b.release();
    var str_a = try buildString(allocator, cfg.rows, 7);
    defer str_a.release();
    var str_b = try buildString(allocator, cfg.rows, 8);
    defer str_b.release();

    const nullable_cfg = BenchConfig{ .rows = cfg.rows - 3, .iterations = cfg.iterations };
    try timeKernel(io, &ctx, .scalar, "compute_add_int32", "add_checked", &.{ .{ .array = a }, .{ .array = b } }, cfg, emit_csv, meta);
    try timeKernel(io, &ctx, .scalar, "compute_add_int32_nullable", "add_checked", &.{ .{ .array = an_cut }, .{ .array = bn_shift } }, nullable_cfg, emit_csv, meta);
    try timeKernel(io, &ctx, .scalar, "compute_cmp_lt_int32", "less", &.{ .{ .array = a }, .{ .array = b } }, cfg, emit_csv, meta);
    try timeKernel(io, &ctx, .aggregate, "compute_sum_int64", "sum", &.{.{ .array = i64arr }}, cfg, emit_csv, meta);
    try timeKernel(io, &ctx, .aggregate, "compute_sum_float64", "sum", &.{.{ .array = f64arr }}, cfg, emit_csv, meta);
    try timeKernel(io, &ctx, .aggregate, "compute_min_max_int32", "min_max", &.{.{ .array = an }}, cfg, emit_csv, meta);
    try timeKernel(io, &ctx, .scalar, "compute_logical_and_kleene", "and_kleene", &.{ .{ .array = bool_a }, .{ .array = bool_b } }, cfg, emit_csv, meta);
    try timeKernel(io, &ctx, .scalar, "compute_is_null", "is_null", &.{.{ .array = an }}, cfg, emit_csv, meta);
    try timeKernel(io, &ctx, .scalar, "compute_cmp_string", "less", &.{ .{ .array = str_a }, .{ .array = str_b } }, cfg, emit_csv, meta);

    var ts = try buildTimestamp(allocator, cfg.rows, 9);
    defer ts.release();
    const to_int64 = Options{ .cast = .{ .to_type = .{ .int64 = {} } } };
    const to_string = Options{ .cast = .{ .to_type = .{ .string = {} } } };
    try timeKernelWith(io, &ctx, .scalar, "compute_cast_int32_to_int64", "cast", &.{.{ .array = a }}, to_int64, cfg, emit_csv, meta);
    try timeKernelWith(io, &ctx, .scalar, "compute_cast_int32_to_string", "cast", &.{.{ .array = a }}, to_string, cfg, emit_csv, meta);
    try timeKernel(io, &ctx, .scalar, "compute_utf8_upper", "utf8_upper", &.{.{ .array = str_a }}, cfg, emit_csv, meta);
    try timeKernel(io, &ctx, .scalar, "compute_temporal_year", "year", &.{.{ .array = ts }}, cfg, emit_csv, meta);
}

/// timestamp[ms] array spread over about 1870..2070, every 10th row null.
fn buildTimestamp(allocator: std.mem.Allocator, rows: usize, seed: u32) !zarrow.ArrayRef {
    var b = try zarrow.TimestampBuilder(.millisecond, null).init(allocator, rows);
    defer b.deinit();
    var x: u32 = seed;
    for (0..rows) |i| {
        x = x *% 1664525 +% 1013904223;
        const ms = (@as(i64, x) - (1 << 31)) * 1470;
        if (i % 10 == 0) try b.appendNull() else try b.append(ms);
    }
    return b.finish();
}

/// Bool array with every `null_every`-th row null.
fn buildBool(allocator: std.mem.Allocator, rows: usize, seed: u32, null_every: usize) !zarrow.ArrayRef {
    var b = try zarrow.BooleanBuilder.init(allocator, rows);
    defer b.deinit();
    var x: u32 = seed;
    for (0..rows) |i| {
        x = x *% 1664525 +% 1013904223;
        if (null_every != 0 and i % null_every == 0) try b.appendNull() else try b.append((x >> 16) & 1 == 1);
    }
    return b.finish();
}

/// Utf8 array of 4..19 byte values sharing an 8-byte prefix (comparisons look past it).
fn buildString(allocator: std.mem.Allocator, rows: usize, seed: u32) !zarrow.ArrayRef {
    var b = try zarrow.StringBuilder.init(allocator, rows, rows * 16);
    defer b.deinit();
    var x: u32 = seed;
    var buf: [20]u8 = undefined;
    for (0..rows) |_| {
        x = x *% 1664525 +% 1013904223;
        const s = std.fmt.bufPrint(&buf, "key-{d:0>4}{d}", .{ (x >> 20) % 16, x % 1000003 }) catch unreachable;
        try b.append(s);
    }
    return b.finish();
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
            try runBenchmark(io, allocator, .{ .rows = 1_000, .iterations = 2000 }, true, meta);
            try runBenchmark(io, allocator, .{ .rows = 10_000, .iterations = 500 }, true, meta);
            try runBenchmark(io, allocator, .{ .rows = 100_000, .iterations = 100 }, true, meta);
        },
        .ci, .ci_no_header => {
            if (mode == .ci) printCsvHeader();
            try runBenchmark(io, allocator, .{ .rows = 1_000, .iterations = 400 }, true, meta);
            try runBenchmark(io, allocator, .{ .rows = 10_000, .iterations = 100 }, true, meta);
            try runBenchmark(io, allocator, .{ .rows = 100_000, .iterations = 20 }, true, meta);
        },
        .default => try runBenchmark(io, allocator, .{ .rows = 200_000, .iterations = 200 }, false, meta),
    }
}
