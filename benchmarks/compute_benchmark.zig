const std = @import("std");
const zarrow = @import("zarrow");
const csv_meta = @import("csv_meta.zig");
const bench_allocator = @import("bench_allocator.zig");

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
        .chunked => |c| return c.len(),
        .record_batch, .table => return d.length(),
    }
}

fn timeKernel(io: std.Io, ctx: *ExecContext, kind: compute.FunctionKind, bench_name: []const u8, fn_name: []const u8, args: []const Datum, cfg: BenchConfig, emit_csv: bool, meta: csv_meta.CsvMeta) !void {
    return timeKernelWith(io, ctx, kind, bench_name, fn_name, args, Options.noneValue(), cfg, emit_csv, meta);
}

fn timeKernelWith(io: std.Io, ctx: *ExecContext, kind: compute.FunctionKind, bench_name: []const u8, fn_name: []const u8, args: []const Datum, options: Options, cfg: BenchConfig, emit_csv: bool, meta: csv_meta.CsvMeta) !void {
    // Untimed warm-up: the timed iterations reuse its output blocks (bench_allocator.zig).
    var warm_up = try ctx.invoke(fn_name, kind, args, options);
    warm_up.release();
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

    const dash = Options{ .replace_substring = .{ .pattern = "-", .replacement = "::" } };
    try timeKernelWith(io, &ctx, .scalar, "compute_replace_substring", "replace_substring", &.{.{ .array = str_a }}, dash, cfg, emit_csv, meta);
    var words = try buildWords(allocator, cfg.rows, 10);
    defer words.release();
    try timeKernel(io, &ctx, .scalar, "compute_split_whitespace", "ascii_split_whitespace", &.{.{ .array = words }}, cfg, emit_csv, meta);
    const day = Options{ .round_temporal = .{ .unit = .day } };
    try timeKernelWith(io, &ctx, .scalar, "compute_floor_temporal_day", "floor_temporal", &.{.{ .array = ts }}, day, cfg, emit_csv, meta);
    // Rows and a 128-value set drawn from [0, 384): about a third of the rows match.
    var a_mod = try buildInt32Mod(allocator, cfg.rows, 12, 384);
    defer a_mod.release();
    var mod_set = try buildInt32Mod(allocator, 128, 13, 384);
    defer mod_set.release();
    const mod_lookup = Options{ .set_lookup = .{ .value_set = .{ .array = mod_set } } };
    try timeKernelWith(io, &ctx, .scalar, "compute_is_in_int32", "is_in", &.{.{ .array = a_mod }}, mod_lookup, cfg, emit_csv, meta);
    var keys = try buildInt64Mod(allocator, cfg.rows, 14, 1000);
    defer keys.release();
    try timeGroupedSum(io, &ctx, "compute_hash_sum_int64", keys, i64arr, cfg, emit_csv, meta);

    var words_unicode = try buildUnicode(allocator, cfg.rows, 15);
    defer words_unicode.release();
    try timeKernel(io, &ctx, .scalar, "compute_utf8_upper_unicode", "utf8_upper", &.{.{ .array = words_unicode }}, cfg, emit_csv, meta);
    // int32 + int64: the int32 operand is cast to int64 first (Arrow's implicit cast).
    try timeKernel(io, &ctx, .scalar, "compute_add_promoted", "add_checked", &.{ .{ .array = a }, .{ .array = i64arr } }, cfg, emit_csv, meta);

    // decimal128(18, 2) + decimal128(18, 2) -> decimal128(19, 2).
    var dec_a = try buildDecimal(allocator, cfg.rows, 16);
    defer dec_a.release();
    var dec_b = try buildDecimal(allocator, cfg.rows, 17);
    defer dec_b.release();
    try timeKernel(io, &ctx, .scalar, "compute_decimal128_add", "add", &.{ .{ .array = dec_a }, .{ .array = dec_b } }, cfg, emit_csv, meta);
    // string vs string_view: bytes compared in place, no cast.
    var str_view = try ctx.invoke("cast", .scalar, &.{.{ .array = str_b }}, .{ .cast = .{ .to_type = .{ .string_view = {} } } });
    defer str_view.release();
    try timeKernel(io, &ctx, .scalar, "compute_cmp_mixed_string", "less", &.{ .{ .array = str_a }, str_view }, cfg, emit_csv, meta);
    // hash_sum with explicit ScalarAggregateOptions over int64 values with 10% nulls (1000 groups).
    var grouper = try compute.kernels.hash_aggregate.Grouper.init(allocator, &.{keys.data().data_type});
    defer grouper.deinit();
    var group_ids = try grouper.consume(&.{keys});
    defer group_ids.release();
    const skip_nulls = Options{ .scalar_aggregate = .{ .skip_nulls = true, .min_count = 1 } };
    try timeKernelWith(io, &ctx, .hash_aggregate, "compute_hash_sum_skip_nulls", "hash_sum", &.{ .{ .array = i64arr }, .{ .array = group_ids } }, skip_nulls, cfg, emit_csv, meta);

    // float64 with 10% nulls: square root, rounding to two digits (half to even) and stddev.
    try timeKernel(io, &ctx, .scalar, "compute_sqrt_float64", "sqrt", &.{.{ .array = f64arr }}, cfg, emit_csv, meta);
    const two_digits = Options{ .round = .{ .ndigits = 2 } };
    try timeKernelWith(io, &ctx, .scalar, "compute_round_float64", "round", &.{.{ .array = f64arr }}, two_digits, cfg, emit_csv, meta);
    try timeKernel(io, &ctx, .aggregate, "compute_stddev_float64", "stddev", &.{.{ .array = f64arr }}, cfg, emit_csv, meta);
    // unique over int32 rows drawn from 1000 values; sort_indices over random int32 rows.
    var a_k = try buildInt32Mod(allocator, cfg.rows, 18, 1000);
    defer a_k.release();
    try timeKernel(io, &ctx, .vector, "compute_unique_int32", "unique", &.{.{ .array = a_k }}, cfg, emit_csv, meta);
    try timeKernel(io, &ctx, .vector, "compute_sort_indices_int32", "sort_indices", &.{.{ .array = a }}, cfg, emit_csv, meta);
    // partition_nth_indices at the median and select_k_unstable (k = 100) over random int32 rows.
    try timeKernelWith(io, &ctx, .vector, "compute_partition_nth_int32", "partition_nth_indices", &.{.{ .array = a }}, .{ .partition_nth = .{ .pivot = cfg.rows / 2 } }, cfg, emit_csv, meta);
    try timeKernelWith(io, &ctx, .vector, "compute_select_k_int32", "select_k_unstable", &.{.{ .array = a }}, .{ .select_k = .{ .k = 100 } }, cfg, emit_csv, meta);

    // take of random rows (with repeats) from a record batch of four int32 columns, two with
    // 10% nulls: ns per row covers all four columns.
    const i32_type = zarrow.DataType{ .int32 = {} };
    const take_fields = [_]zarrow.Field{
        .{ .name = "a", .data_type = &i32_type },  .{ .name = "b", .data_type = &i32_type },
        .{ .name = "an", .data_type = &i32_type }, .{ .name = "bn", .data_type = &i32_type },
    };
    const take_batch = try zarrow.RecordBatch.initBorrowed(allocator, .{ .fields = &take_fields }, &.{ a, b, an, bn });
    var rb = try Datum.fromRecordBatch(allocator, take_batch);
    defer rb.release();
    const take_idx = try allocator.alloc(usize, cfg.rows);
    defer allocator.free(take_idx);
    var x: u32 = 19;
    for (take_idx) |*t| {
        x = x *% 1664525 +% 1013904223;
        t.* = (x >> 4) % cfg.rows;
    }
    try timeTake(io, "compute_take_record_batch", rb, take_idx, cfg, emit_csv, meta);

    // unique over a dictionary<int32, string> of 1000 entries (the rows of a_k as indices).
    var dict_values = try zarrow.StringBuilder.init(allocator, 1000, 8000);
    defer dict_values.deinit();
    var name_buf: [16]u8 = undefined;
    for (0..1000) |k| try dict_values.append(std.fmt.bufPrint(&name_buf, "v{d}", .{k}) catch unreachable);
    var dict = try dict_values.finish();
    defer dict.release();
    var dict_builder = try zarrow.DictionaryBuilder.init(allocator, .{ .bit_width = 32, .signed = true }, &dict.data().data_type, cfg.rows);
    defer dict_builder.deinit();
    for (try (zarrow.PrimitiveArray(i32){ .data = a_k.data() }).values()) |k| try dict_builder.appendIndex(k);
    var dict_rows = try dict_builder.finish(dict);
    defer dict_rows.release();
    try timeKernel(io, &ctx, .vector, "compute_unique_dictionary", "unique", &.{.{ .array = dict_rows }}, cfg, emit_csv, meta);
    // min_max over the string rows; cumulative_sum (skip_nulls) and pairwise_diff over int64 with
    // 10% nulls; replace_with_mask of int32 rows under a bool mask with 10% nulls.
    try timeKernel(io, &ctx, .aggregate, "compute_min_max_string", "min_max", &.{.{ .array = str_a }}, cfg, emit_csv, meta);
    try timeKernelWith(io, &ctx, .vector, "compute_cumulative_sum_int64", "cumulative_sum", &.{.{ .array = i64arr }}, .{ .cumulative = .{ .skip_nulls = true } }, cfg, emit_csv, meta);
    try timeKernel(io, &ctx, .vector, "compute_pairwise_diff_int64", "pairwise_diff", &.{.{ .array = i64arr }}, cfg, emit_csv, meta);
    try timeKernel(io, &ctx, .vector, "compute_replace_with_mask_int32", "replace_with_mask", &.{ .{ .array = a }, .{ .array = bool_a }, .{ .array = b } }, cfg, emit_csv, meta);
    // list<int32> of four values per row over the int32 rows to list<int64>: ns per child row.
    const item32 = zarrow.Field{ .name = "item", .data_type = &i32_type };
    const i64_type = zarrow.DataType{ .int64 = {} };
    var list_builder = try zarrow.ListBuilder.init(allocator, cfg.rows / 4, item32);
    defer list_builder.deinit();
    for (0..cfg.rows / 4) |_| try list_builder.appendLen(4);
    var child = try a.slice(0, (cfg.rows / 4) * 4);
    defer child.release();
    var lists = try list_builder.finish(child);
    defer lists.release();
    const to_list64 = Options{ .cast = .{ .to_type = .{ .list = .{ .value_field = .{ .name = "item", .data_type = &i64_type } } } } };
    try timeKernelWith(io, &ctx, .scalar, "compute_cast_list_int32_to_int64", "cast", &.{.{ .array = lists }}, to_list64, .{ .rows = (cfg.rows / 4) * 4, .iterations = cfg.iterations }, emit_csv, meta);
    // indices_nonzero over int32 rows that are 0 or 1 at random.
    var zero_one = try buildInt32Mod(allocator, cfg.rows, 20, 2);
    defer zero_one.release();
    try timeKernel(io, &ctx, .vector, "compute_indices_nonzero", "indices_nonzero", &.{.{ .array = zero_one }}, cfg, emit_csv, meta);
    // Regex over the "key-0007123456" rows: a literal (memmem path), a class pattern (lazy
    // DFA; about 3 rows in 16 match) and a replacement of every digit run (DFA spans).
    try timeKernelWith(io, &ctx, .scalar, "compute_match_regex_literal", "match_substring_regex", &.{.{ .array = str_a }}, .{ .match_substring = .{ .pattern = "y-0012" } }, cfg, emit_csv, meta);
    try timeKernelWith(io, &ctx, .scalar, "compute_match_regex_class", "match_substring_regex", &.{.{ .array = str_a }}, .{ .match_substring = .{ .pattern = "[a-z]+-\\d{3}[1-3]\\d+$" } }, cfg, emit_csv, meta);
    try timeKernelWith(io, &ctx, .scalar, "compute_replace_regex", "replace_substring_regex", &.{.{ .array = str_a }}, .{ .replace_substring = .{ .pattern = "\\d+", .replacement = "#" } }, cfg, emit_csv, meta);
    // Temporal text over millisecond timestamps with 10% nulls: the default strftime format, the
    // cast to string, and strptime of the ISO text that strftime printed (seconds, 19 bytes).
    try timeKernelWith(io, &ctx, .scalar, "compute_strftime_timestamp", "strftime", &.{.{ .array = ts }}, .{ .none = {} }, cfg, emit_csv, meta);
    try timeKernelWith(io, &ctx, .scalar, "compute_cast_timestamp_to_string", "cast", &.{.{ .array = ts }}, to_string, cfg, emit_csv, meta);
    var iso = try ctx.invokeScalar("strftime", &.{.{ .array = ts }}, .{ .strftime = .{ .format = "%Y-%m-%dT%H:%M:%OS" } });
    defer iso.release();
    try timeKernelWith(io, &ctx, .scalar, "compute_strptime", "strptime", &.{iso}, .{ .strptime = .{ .format = "%Y-%m-%dT%H:%M:%S", .unit = .second } }, cfg, emit_csv, meta);
    // assume_timezone (America/New_York, ambiguous / nonexistent earliest) over a naive
    // millisecond series one minute apart from 2021-01-01 (it crosses the March DST gap).
    var minutes = try zarrow.TimestampBuilder(.millisecond, null).init(allocator, cfg.rows);
    defer minutes.deinit();
    for (0..cfg.rows) |i| try minutes.append(1_609_459_200_000 + @as(i64, @intCast(i)) * 60_000);
    var series = try minutes.finish();
    defer series.release();
    try timeKernelWith(io, &ctx, .scalar, "compute_assume_timezone", "assume_timezone", &.{.{ .array = series }}, .{ .assume_timezone = .{ .timezone = "America/New_York", .ambiguous = .earliest, .nonexistent = .earliest } }, cfg, emit_csv, meta);
    // utf8_normalize to NFC: ASCII keys (in NFC already: the input comes back), and mixed
    // Unicode rows of which one in four is decomposed ("cafe\u{301}") and must be composed.
    try timeKernelWith(io, &ctx, .scalar, "compute_utf8_normalize_nfc_ascii", "utf8_normalize", &.{.{ .array = str_a }}, .none, cfg, emit_csv, meta);
    var mixed_norm = try buildNormMixed(allocator, cfg.rows, 23);
    defer mixed_norm.release();
    try timeKernelWith(io, &ctx, .scalar, "compute_utf8_normalize_nfc_mixed", "utf8_normalize", &.{.{ .array = mixed_norm }}, .none, cfg, emit_csv, meta);
    // tdigest (default options: q 0.5, delta 100, buffer_size 500) over float64 with 10% nulls.
    try timeKernel(io, &ctx, .aggregate, "compute_tdigest_float64", "tdigest", &.{.{ .array = f64arr }}, cfg, emit_csv, meta);
}

/// Time `datumTake` of `indices` from `datum` (one untimed warm-up first).
fn timeTake(io: std.Io, bench_name: []const u8, datum: Datum, indices: []const usize, cfg: BenchConfig, emit_csv: bool, meta: csv_meta.CsvMeta) !void {
    var warm_up = try compute.datumTake(datum, indices);
    warm_up.release();
    var checksum: u64 = 0;
    const timer_start = std.Io.Timestamp.now(io, .awake);
    for (0..cfg.iterations) |_| {
        var out = try compute.datumTake(datum, indices);
        defer out.release();
        checksum +%= fold(out);
    }
    const elapsed_ns: u64 = @intCast(timer_start.untilNow(io, .awake).toNanoseconds());
    report(bench_name, cfg, elapsed_ns, checksum, emit_csv, meta);
}

/// decimal128(18, 2) values below 10^16 in magnitude, every tenth row null.
fn buildDecimal(allocator: std.mem.Allocator, rows: usize, seed: u32) !zarrow.ArrayRef {
    var b = try zarrow.Decimal128Builder(.{ .precision = 18, .scale = 2 }).init(allocator, rows);
    defer b.deinit();
    var x: u64 = seed;
    for (0..rows) |i| {
        x = x *% 6364136223846793005 +% 1442695040888963407;
        if (i % 10 == 0) try b.appendNull() else try b.append(@as(i128, @as(i64, @bitCast(x >> 10))) >> 1);
    }
    return b.finish();
}

/// Utf8 rows for utf8_normalize: words in NFC ("größe", "привет", "Ångström") and in NFD
/// ("cafe\u{301}", "A\u{30A}ngstro\u{308}m") with a number, one decomposed row in four.
fn buildNormMixed(allocator: std.mem.Allocator, rows: usize, seed: u32) !zarrow.ArrayRef {
    const words = [_][]const u8{ "größe", "привет", "Ångström", "naïve", "σωκράτης", "東京", "cafe\u{301}", "A\u{30A}ngstro\u{308}m" };
    var b = try zarrow.StringBuilder.init(allocator, rows, rows * 20);
    defer b.deinit();
    var x: u32 = seed;
    var buf: [32]u8 = undefined;
    for (0..rows) |_| {
        x = x *% 1664525 +% 1013904223;
        const s = std.fmt.bufPrint(&buf, "{s}-{d}", .{ words[(x >> 24) % words.len], x % 1000 }) catch unreachable;
        try b.append(s);
    }
    return b.finish();
}

/// Utf8 rows of a lowercase word in a non-Latin or accented script and a number
/// ("привет-417", "größe-3"): 2-byte code points, ASCII digits, a 2-to-3-byte mapping (ß).
fn buildUnicode(allocator: std.mem.Allocator, rows: usize, seed: u32) !zarrow.ArrayRef {
    const words = [_][]const u8{ "привет", "σωκράτης", "größe", "élan", "ǆungla", "ąžuolas", "ölçü", "ξένος" };
    var b = try zarrow.StringBuilder.init(allocator, rows, rows * 20);
    defer b.deinit();
    var x: u32 = seed;
    var buf: [32]u8 = undefined;
    for (0..rows) |_| {
        x = x *% 1664525 +% 1013904223;
        const s = std.fmt.bufPrint(&buf, "{s}-{d}", .{ words[(x >> 24) % words.len], x % 1000 }) catch unreachable;
        try b.append(s);
    }
    return b.finish();
}

/// Group `values` by `keys` (Grouper) and sum each group (hash_sum): the end-to-end group by.
fn timeGroupedSum(io: std.Io, ctx: *ExecContext, bench_name: []const u8, keys: zarrow.ArrayRef, values: zarrow.ArrayRef, cfg: BenchConfig, emit_csv: bool, meta: csv_meta.CsvMeta) !void {
    const hash = compute.kernels.hash_aggregate;
    var checksum: u64 = 0;
    var timer_start = std.Io.Timestamp.now(io, .awake);
    for (0..cfg.iterations + 1) |it| {
        // Iteration 0 is the untimed warm-up.
        if (it == 1) timer_start = std.Io.Timestamp.now(io, .awake);
        var grouper = try hash.Grouper.init(ctx.allocator, &.{keys.data().data_type});
        defer grouper.deinit();
        var ids = try grouper.consume(&.{keys});
        defer ids.release();
        var out = try ctx.invoke("hash_sum", .hash_aggregate, &.{ .{ .array = values }, .{ .array = ids } }, Options.noneValue());
        defer out.release();
        if (it > 0) checksum +%= fold(out);
    }
    const elapsed_ns: u64 = @intCast(timer_start.untilNow(io, .awake).toNanoseconds());
    report(bench_name, cfg, elapsed_ns, checksum, emit_csv, meta);
}

/// Int32 values in `[0, modulus)`.
fn buildInt32Mod(allocator: std.mem.Allocator, rows: usize, seed: u32, modulus: u32) !zarrow.ArrayRef {
    var b = try zarrow.Int32Builder.init(allocator, rows);
    defer b.deinit();
    var x: u32 = seed;
    for (0..rows) |_| {
        x = x *% 1664525 +% 1013904223;
        try b.append(@intCast((x >> 8) % modulus));
    }
    return b.finish();
}

/// Int64 group keys in `[0, groups)`.
fn buildInt64Mod(allocator: std.mem.Allocator, rows: usize, seed: u32, groups: u32) !zarrow.ArrayRef {
    var b = try zarrow.Int64Builder.init(allocator, rows);
    defer b.deinit();
    var x: u32 = seed;
    for (0..rows) |_| {
        x = x *% 1664525 +% 1013904223;
        try b.append((x >> 8) % groups);
    }
    return b.finish();
}

/// Utf8 rows of three space-separated words ("w12 x3456 y7").
fn buildWords(allocator: std.mem.Allocator, rows: usize, seed: u32) !zarrow.ArrayRef {
    var b = try zarrow.StringBuilder.init(allocator, rows, rows * 16);
    defer b.deinit();
    var x: u32 = seed;
    var buf: [32]u8 = undefined;
    for (0..rows) |_| {
        x = x *% 1664525 +% 1013904223;
        const s = std.fmt.bufPrint(&buf, "w{d} x{d} y{d}", .{ x % 100, (x >> 8) % 10000, (x >> 20) % 10 }) catch unreachable;
        try b.append(s);
    }
    return b.finish();
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
    const allocator = bench_allocator.allocator;
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
