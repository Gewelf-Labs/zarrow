const std = @import("std");
const zarrow = @import("zarrow");
const csv_meta = @import("csv_meta.zig");

const compute = zarrow.compute;
const Datum = compute.Datum;
const Options = compute.Options;
const ExecContext = compute.ExecContext;
const KernelError = compute.KernelError;

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

// ---------------------------------------------------------------------------------------------
// Reference kernels. zarrow ships the compute *framework* (registry, signatures, chunk iterators,
// null propagation helpers), not a kernel catalog; these are the kernels a downstream library
// would register, written the way docs/compute-api.md recommends, so the benchmark measures the
// framework path end to end (resolve + exec + output array).
// ---------------------------------------------------------------------------------------------

const int32_type = zarrow.DataType{ .int32 = {} };
const int64_type = zarrow.DataType{ .int64 = {} };
const float64_type = zarrow.DataType{ .double = {} };
const bool_type = zarrow.DataType{ .bool = {} };

fn isInt32Pair(args: []const Datum) bool {
    return args.len == 2 and args[0] == .array and args[1] == .array and
        args[0].array.data().data_type == .int32 and args[1].array.data().data_type == .int32;
}

fn isInt32Array(args: []const Datum) bool {
    return args.len == 1 and args[0] == .array and args[0].array.data().data_type == .int32;
}

fn isInt64Array(args: []const Datum) bool {
    return args.len == 1 and args[0] == .array and args[0].array.data().data_type == .int64;
}

fn isFloat64Array(args: []const Datum) bool {
    return args.len == 1 and args[0] == .array and args[0].array.data().data_type == .double;
}

fn int32ResultType(_: []const Datum, _: Options) KernelError!zarrow.DataType {
    return int32_type;
}

fn boolResultType(_: []const Datum, _: Options) KernelError!zarrow.DataType {
    return bool_type;
}

fn int64ResultType(_: []const Datum, _: Options) KernelError!zarrow.DataType {
    return int64_type;
}

fn float64ResultType(_: []const Datum, _: Options) KernelError!zarrow.DataType {
    return float64_type;
}

/// add_int32(array, array) -> int32 array; nulls propagate; overflow per ctx.overflowMode().
fn addInt32Kernel(ctx: *ExecContext, args: []const Datum, _: Options) KernelError!Datum {
    const lhs = zarrow.Int32Array{ .data = args[0].array.data() };
    const rhs = zarrow.Int32Array{ .data = args[1].array.data() };
    const n = lhs.len();
    if (rhs.len() != n) return error.InvalidInput;
    const a = lhs.values() catch return error.InvalidInput;
    const b = rhs.values() catch return error.InvalidInput;

    var out = compute.DenseBinaryOutput(i32).init(ctx.allocator, args[0].array.data(), args[1].array.data(), int32_type) catch return error.OutOfMemory;
    errdefer out.deinit();
    const dst = out.values();
    switch (ctx.overflowMode()) {
        .wrapping => for (a, b, dst) |x, y, *d| {
            d.* = x +% y;
        },
        .saturating => for (a, b, dst) |x, y, *d| {
            d.* = x +| y;
        },
        .checked => {
            // Dense pass: wrapping add plus a branch-free overflow flag (sign trick, vectorizes);
            // only when some row overflowed, find out whether a valid one did.
            var any: i32 = 0;
            for (a, b, dst) |x, y, *d| {
                const r = x +% y;
                d.* = r;
                any |= (x ^ r) & (y ^ r);
            }
            if (any < 0) {
                for (a, b, 0..) |x, y, i| {
                    if (@addWithOverflow(x, y)[1] != 0 and out.isValid(i)) return error.Overflow;
                }
            }
        },
    }
    return .{ .array = out.finish() catch return error.OutOfMemory };
}

/// less_int32(array, array) -> bool array; nulls propagate.
fn lessInt32Kernel(ctx: *ExecContext, args: []const Datum, _: Options) KernelError!Datum {
    const lhs = zarrow.Int32Array{ .data = args[0].array.data() };
    const rhs = zarrow.Int32Array{ .data = args[1].array.data() };
    const n = lhs.len();
    if (rhs.len() != n) return error.InvalidInput;
    const a = lhs.values() catch return error.InvalidInput;
    const b = rhs.values() catch return error.InvalidInput;

    var out = compute.DenseBinaryOutput(bool).init(ctx.allocator, args[0].array.data(), args[1].array.data(), bool_type) catch return error.OutOfMemory;
    errdefer out.deinit();
    const bits = out.values();
    var i: usize = 0;
    while (i + 8 <= n) : (i += 8) {
        var byte: u8 = 0;
        inline for (0..8) |k| byte |= @as(u8, @intFromBool(a[i + k] < b[i + k])) << k;
        bits[i / 8] = byte;
    }
    if (i < n) {
        var byte: u8 = 0;
        for (i..n) |j| byte |= @as(u8, @intFromBool(a[j] < b[j])) << @as(u3, @intCast(j - i));
        bits[i / 8] = byte;
    }
    return .{ .array = out.finish() catch return error.OutOfMemory };
}

/// Loads 8 values starting at `i` (zero past the end).
inline fn lanes(comptime T: type, values: []const T, i: usize) @Vector(8, T) {
    if (i + 8 <= values.len) return values[i..][0..8].*;
    var tmp: [8]T = @splat(0);
    @memcpy(tmp[0 .. values.len - i], values[i..]);
    return tmp;
}

/// sum_int64(array) -> int64 scalar (null when every input is null); error on overflow of the sum.
fn sumInt64Kernel(_: *ExecContext, args: []const Datum, _: Options) KernelError!Datum {
    const data = args[0].array.data();
    const values = (zarrow.Int64Array{ .data = data }).values() catch return error.InvalidInput;
    const bits = compute.validityBits(data);
    const zero: @Vector(8, i64) = @splat(0);
    var sum: i64 = 0;
    var count: usize = 0;
    var i: usize = 0;
    while (i < values.len) : (i += 8) {
        const mask = compute.laneMask(data, bits, i);
        const v = @select(i64, mask, lanes(i64, values, i), zero);
        count += @popCount(@as(u8, @bitCast(mask)));
        // Checked: the running total must stay in i64 at every row (same as a serial checked sum).
        inline for (0..8) |k| {
            const r = @addWithOverflow(sum, v[k]);
            if (r[1] != 0) return error.Overflow;
            sum = r[0];
        }
    }
    return .{ .scalar = compute.Scalar.init(int64_type, if (count == 0) .null else .{ .i64 = sum }) };
}

/// sum_float64(array) -> float64 scalar (null when every input is null).
fn sumFloat64Kernel(_: *ExecContext, args: []const Datum, _: Options) KernelError!Datum {
    const data = args[0].array.data();
    const values = (zarrow.Float64Array{ .data = data }).values() catch return error.InvalidInput;
    const bits = compute.validityBits(data);
    const zero: @Vector(8, f64) = @splat(0);
    var acc = zero;
    var count: usize = 0;
    var i: usize = 0;
    while (i < values.len) : (i += 8) {
        const mask = compute.laneMask(data, bits, i);
        acc += @select(f64, mask, lanes(f64, values, i), zero);
        count += @popCount(@as(u8, @bitCast(mask)));
    }
    return .{ .scalar = compute.Scalar.init(float64_type, if (count == 0) .null else .{ .f64 = @reduce(.Add, acc) }) };
}

/// min_max_int32(array) -> int32 array [min, max] (both null when every input is null).
fn minMaxInt32Kernel(ctx: *ExecContext, args: []const Datum, _: Options) KernelError!Datum {
    const data = args[0].array.data();
    const values = (zarrow.Int32Array{ .data = data }).values() catch return error.InvalidInput;
    const bits = compute.validityBits(data);
    var lo: @Vector(8, i32) = @splat(std.math.maxInt(i32));
    var hi: @Vector(8, i32) = @splat(std.math.minInt(i32));
    var count: usize = 0;
    var i: usize = 0;
    while (i < values.len) : (i += 8) {
        const mask = compute.laneMask(data, bits, i);
        const v = lanes(i32, values, i);
        lo = @min(lo, @select(i32, mask, v, lo));
        hi = @max(hi, @select(i32, mask, v, hi));
        count += @popCount(@as(u8, @bitCast(mask)));
    }
    var builder = zarrow.Int32Builder.init(ctx.allocator, 2) catch return error.OutOfMemory;
    defer builder.deinit();
    if (count == 0) {
        builder.appendNull() catch return error.OutOfMemory;
        builder.appendNull() catch return error.OutOfMemory;
    } else {
        builder.append(@reduce(.Min, lo)) catch return error.OutOfMemory;
        builder.append(@reduce(.Max, hi)) catch return error.OutOfMemory;
    }
    return .{ .array = builder.finish() catch return error.OutOfMemory };
}

fn registerKernels(registry: *compute.FunctionRegistry) !void {
    try registry.registerScalarKernel("add_int32", .{
        .signature = .{ .arity = 2, .type_check = isInt32Pair, .result_type_fn = int32ResultType },
        .exec = addInt32Kernel,
    });
    try registry.registerScalarKernel("less_int32", .{
        .signature = .{ .arity = 2, .type_check = isInt32Pair, .result_type_fn = boolResultType },
        .exec = lessInt32Kernel,
    });
    try registry.registerAggregateKernel("sum_int64", .{
        .signature = .{ .arity = 1, .type_check = isInt64Array, .result_type_fn = int64ResultType },
        .exec = sumInt64Kernel,
    });
    try registry.registerAggregateKernel("sum_float64", .{
        .signature = .{ .arity = 1, .type_check = isFloat64Array, .result_type_fn = float64ResultType },
        .exec = sumFloat64Kernel,
    });
    try registry.registerAggregateKernel("min_max_int32", .{
        .signature = .{ .arity = 1, .type_check = isInt32Array, .result_type_fn = int32ResultType },
        .exec = minMaxInt32Kernel,
    });
}

// ---------------------------------------------------------------------------------------------

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
    var checksum: u64 = 0;
    const timer_start = std.Io.Timestamp.now(io, .awake);
    for (0..cfg.iterations) |_| {
        var out = try ctx.invoke(fn_name, kind, args, Options.noneValue());
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
    try registerKernels(&registry);
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

    const nullable_cfg = BenchConfig{ .rows = cfg.rows - 3, .iterations = cfg.iterations };
    try timeKernel(io, &ctx, .scalar, "compute_add_int32", "add_int32", &.{ .{ .array = a }, .{ .array = b } }, cfg, emit_csv, meta);
    try timeKernel(io, &ctx, .scalar, "compute_add_int32_nullable", "add_int32", &.{ .{ .array = an_cut }, .{ .array = bn_shift } }, nullable_cfg, emit_csv, meta);
    try timeKernel(io, &ctx, .scalar, "compute_cmp_lt_int32", "less_int32", &.{ .{ .array = a }, .{ .array = b } }, cfg, emit_csv, meta);
    try timeKernel(io, &ctx, .aggregate, "compute_sum_int64", "sum_int64", &.{.{ .array = i64arr }}, cfg, emit_csv, meta);
    try timeKernel(io, &ctx, .aggregate, "compute_sum_float64", "sum_float64", &.{.{ .array = f64arr }}, cfg, emit_csv, meta);
    try timeKernel(io, &ctx, .aggregate, "compute_min_max_int32", "min_max_int32", &.{.{ .array = an }}, cfg, emit_csv, meta);
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
