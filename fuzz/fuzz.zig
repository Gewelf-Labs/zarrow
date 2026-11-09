//! zarrow fuzz harness: one executable, one `runOne` per target (see docs/fuzzing.md).
//!
//!   zarrow-fuzz <target> replay <file|dir>...
//!   zarrow-fuzz <target> run <N|Ns> --corpus <dir>... --out <dir> [--seed N] [--max-len N]
//!   zarrow-fuzz gen-seeds <corpus-root>
//!
//! Every decoded array goes through validateFull. A returned error is a pass; a panic, hang,
//! leak or allocation over the cap is a finding. The same `runOne` backs the std.testing.fuzz
//! tests at the bottom (`zig build fuzz-test --fuzz=100K`, Linux only).
const std = @import("std");
const zarrow = @import("zarrow");

const Allocator = std.mem.Allocator;
const ArrayRef = zarrow.ArrayRef;
const DataType = zarrow.DataType;
const Field = zarrow.Field;

pub const Target = enum {
    @"ipc-stream",
    @"ipc-stream-zero-copy",
    @"ipc-file",
    @"array-layout",
    compute,
    @"c-abi",
    regex,
    temporal,
    @"integration-json",
};

/// Inputs over this size are skipped (replay) or never produced (run).
pub const max_input_len: usize = 1 << 20;
/// Live bytes one input may allocate before it counts as an OOM finding.
pub var alloc_cap: usize = 256 << 20;
/// Print the error return trace of a rejected input (set by single-file replay).
var explain = false;
const max_batches = 64;
/// In-memory corpus cap for `run`; past it, new signatures are still counted but not kept.
const max_corpus = 1 << 14;

/// What one execution looked like; a new `sig` grows the corpus (our stand-in for coverage).
pub const Feedback = struct {
    err: ?anyerror = null,
    batches: u32 = 0,
    types: u64 = 0, // bit per decoded TypeId (mod 64), children included
    invalid: ?anyerror = null, // first validateFull error

    fn sig(f: Feedback) u64 {
        var h = std.hash.Wyhash.init(0);
        h.update(if (f.err) |e| @errorName(e) else "ok");
        h.update(if (f.invalid) |e| @errorName(e) else "valid");
        h.update(std.mem.asBytes(&@min(f.batches, 8)));
        h.update(std.mem.asBytes(&f.types));
        return h.final();
    }

    fn typeBits(data: *const zarrow.ArrayData) u64 {
        var bits = @as(u64, 1) << @intCast(@intFromEnum(data.data_type.id()) % 64);
        for (data.children) |child| bits |= typeBits(child.data());
        return bits;
    }

    fn noteArray(f: *Feedback, arr: ArrayRef) bool {
        f.types |= typeBits(arr.data());
        arr.data().validateFull() catch |e| {
            if (f.invalid == null) f.invalid = e;
            return false;
        };
        return true;
    }
};

// ---------------------------------------------------------------------------------------------
// Allocation cap + per-input leak check.

const CountingAllocator = struct {
    parent: Allocator,
    live: usize = 0,

    fn allocator(self: *CountingAllocator) Allocator {
        return .{ .ptr = self, .vtable = &.{ .alloc = alloc, .resize = resize, .remap = remap, .free = free } };
    }

    fn charge(self: *CountingAllocator, old: usize, new: usize) void {
        if (new > old and self.live + (new - old) > alloc_cap)
            std.debug.panic("fuzz-oom: allocation of {d} bytes with {d} live exceeds cap {d}", .{ new - old, self.live, alloc_cap });
    }

    fn alloc(ctx: *anyopaque, len: usize, a: std.mem.Alignment, ra: usize) ?[*]u8 {
        const self: *CountingAllocator = @ptrCast(@alignCast(ctx));
        self.charge(0, len);
        const p = self.parent.rawAlloc(len, a, ra) orelse return null;
        self.live += len;
        return p;
    }

    fn resize(ctx: *anyopaque, m: []u8, a: std.mem.Alignment, n: usize, ra: usize) bool {
        const self: *CountingAllocator = @ptrCast(@alignCast(ctx));
        self.charge(m.len, n);
        if (!self.parent.rawResize(m, a, n, ra)) return false;
        self.live = self.live - m.len + n;
        return true;
    }

    fn remap(ctx: *anyopaque, m: []u8, a: std.mem.Alignment, n: usize, ra: usize) ?[*]u8 {
        const self: *CountingAllocator = @ptrCast(@alignCast(ctx));
        self.charge(m.len, n);
        const p = self.parent.rawRemap(m, a, n, ra) orelse return null;
        self.live = self.live - m.len + n;
        return p;
    }

    fn free(ctx: *anyopaque, m: []u8, a: std.mem.Alignment, ra: usize) void {
        const self: *CountingAllocator = @ptrCast(@alignCast(ctx));
        self.parent.rawFree(m, a, ra);
        self.live -= m.len;
    }
};

/// Runs one input against `target`. Panics (= finding) on leaks, cap overruns and broken
/// invariants; library errors are returned in the feedback.
pub fn runOne(target: Target, parent: Allocator, bytes: []const u8) Feedback {
    var counting: CountingAllocator = .{ .parent = parent };
    const gpa = counting.allocator();
    var fb: Feedback = .{};
    const result = switch (target) {
        .@"ipc-stream" => ipcStream(gpa, bytes, false, &fb),
        .@"ipc-stream-zero-copy" => ipcStream(gpa, bytes, true, &fb),
        .@"ipc-file" => ipcFile(gpa, bytes, &fb),
        .compute => computeTarget(gpa, bytes, &fb),
        .@"array-layout" => arrayLayout(gpa, bytes, &fb),
        .@"c-abi" => cAbi(gpa, bytes, &fb),
        .regex => regexTarget(gpa, bytes, &fb),
        .temporal => temporalTarget(gpa, bytes, &fb),
        .@"integration-json" => integrationJsonTarget(gpa, bytes, &fb),
    };
    result catch |e| {
        fb.err = e;
        // Single-file replay: show where the input was rejected (compat-gap triage).
        if (explain) if (@errorReturnTrace()) |trace| std.debug.dumpErrorReturnTrace(trace);
    };
    if (counting.live != 0) std.debug.panic("fuzz-leak: {d} bytes still allocated after the input", .{counting.live});
    return fb;
}

// ---------------------------------------------------------------------------------------------
// IPC targets.

fn noteBatch(fb: *Feedback, batch: *const zarrow.RecordBatch) void {
    fb.batches += 1;
    for (0..batch.numColumns()) |i| _ = fb.noteArray(batch.column(i).*);
}

fn ipcStream(gpa: Allocator, bytes: []const u8, zero_copy: bool, fb: *Feedback) !void {
    const zc = zero_copy and bytes.len > 0;
    var source = zarrow.SharedBuffer.empty;
    if (zc) {
        var owned = try zarrow.OwnedBuffer.initUninit(gpa, bytes.len);
        @memcpy(owned.data[0..bytes.len], bytes);
        source = try owned.toShared(bytes.len);
    }
    defer source.release();
    var fixed: std.Io.Reader = .fixed(if (zc) source.data else bytes);
    var reader = if (zc) zarrow.IpcStreamReader.initZeroCopy(gpa, &fixed, source) else zarrow.IpcStreamReader.init(gpa, &fixed);
    defer reader.deinit();

    _ = reader.readSchema() catch |e| {
        // Tensor streams have no schema message: walk them with the tensor API instead.
        var fixed2: std.Io.Reader = .fixed(bytes);
        var tensors = zarrow.IpcStreamReader.init(gpa, &fixed2);
        defer tensors.deinit();
        while (fb.batches < max_batches) {
            var msg = (tensors.nextTensorLikeMessage() catch break) orelse break;
            defer msg.deinit();
            // A decoded tensor holds its every element inside its body (round 20).
            if (msg.metadata == .tensor) zarrow.validateIpcTensor(msg.metadata.tensor, msg.body.len()) catch std.debug.panic("fuzz-invariant: decoded tensor fails validateTensor", .{});
            fb.batches += 1;
        }
        return e;
    };
    while (fb.batches < max_batches) {
        var batch = (try reader.nextRecordBatch()) orelse return;
        defer batch.deinit();
        noteBatch(fb, &batch);
    }
}

fn ipcFile(gpa: Allocator, bytes: []const u8, fb: *Feedback) !void {
    var reader = zarrow.IpcFileReader.init(gpa, bytes);
    defer reader.deinit();
    _ = try reader.readSchema();
    const n = try reader.recordBatchCount();
    for (0..@min(n, max_batches)) |i| {
        var batch = try reader.readRecordBatchAt(i);
        defer batch.deinit();
        noteBatch(fb, &batch);
    }
    for (0..@min(try reader.tensorCount(), max_batches)) |i| {
        var t = try reader.readTensorAt(i);
        defer t.deinit();
        if (t.metadata == .tensor) zarrow.validateIpcTensor(t.metadata.tensor, t.body.len()) catch std.debug.panic("fuzz-invariant: decoded tensor fails validateTensor", .{});
    }
    for (0..@min(try reader.sparseTensorCount(), max_batches)) |i| {
        var t = try reader.readSparseTensorAt(i);
        t.deinit();
    }
}

fn computeTarget(gpa: Allocator, bytes: []const u8, fb: *Feedback) !void {
    var prng = std.Random.DefaultPrng.init(std.hash.Wyhash.hash(1, bytes));
    var fixed: std.Io.Reader = .fixed(bytes);
    var reader = zarrow.IpcStreamReader.init(gpa, &fixed);
    defer reader.deinit();
    _ = try reader.readSchema();
    while (fb.batches < max_batches) {
        var batch = (try reader.nextRecordBatch()) orelse return;
        defer batch.deinit();
        fb.batches += 1;
        var all_valid = true;
        for (0..batch.numColumns()) |i| {
            const col = batch.column(i).*;
            if (fb.noteArray(col)) try exerciseCompute(gpa, col, prng.random()) else all_valid = false;
        }
        if (all_valid) try exerciseTabular(gpa, &batch, prng.random());
    }
}

/// slice/take/filter/concat/dictionaryEncode and the element-wise / aggregate kernels on an
/// array that passed validateFull; every output must validate and have the expected length,
/// otherwise the kernel broke an invariant (a finding).
fn exerciseCompute(gpa: Allocator, arr: ArrayRef, rng: std.Random) !void {
    const len = arr.data().length;
    const off = rng.uintAtMost(usize, len);
    const sub_len = rng.uintAtMost(usize, len - off);
    var sliced = try arr.slice(off, sub_len);
    defer sliced.release();
    expectValid("slice", sliced, sub_len);
    exerciseKernels(gpa, arr, sliced);

    var indices: [32]usize = undefined;
    const n_idx = if (len == 0) 0 else rng.uintAtMost(usize, indices.len);
    for (indices[0..n_idx]) |*ix| ix.* = rng.uintLessThan(usize, len);
    if (zarrow.computeDatumTake(.{ .array = arr }, indices[0..n_idx])) |taken| {
        var out = taken;
        defer out.release();
        if (out.asArray()) |a| expectValid("take", a, n_idx);
    } else |_| {}

    var pred_builder = try zarrow.BooleanBuilder.init(gpa, len);
    defer pred_builder.deinit();
    var expected_rows: usize = 0;
    const drop_nulls = rng.boolean();
    for (0..len) |_| switch (rng.uintLessThan(u8, 3)) {
        0 => try pred_builder.append(false),
        1 => {
            try pred_builder.append(true);
            expected_rows += 1;
        },
        else => {
            try pred_builder.appendNull();
            if (!drop_nulls) expected_rows += 1;
        },
    };
    var pred = try pred_builder.finish();
    defer pred.release();
    if (zarrow.computeDatumFilter(.{ .array = arr }, .{ .array = pred }, .{ .drop_nulls = drop_nulls })) |filtered| {
        var out = filtered;
        defer out.release();
        if (out.asArray()) |a| expectValid("filter", a, expected_rows);
    } else |_| {}

    if (zarrow.concatArrayRefs(gpa, arr.data().data_type, &.{ arr, sliced })) |joined| {
        var out = joined;
        defer out.release();
        expectValid("concat", out, len + sub_len);
    } else |_| {}

    // dictionaryEncode, and a concat of two parts encoded apart (different dictionaries: unified).
    const value_type = &arr.data().data_type;
    var encoded = zarrow.dictionaryEncode(gpa, arr, value_type) catch return;
    defer encoded.release();
    expectValid("dictionaryEncode", encoded, len);
    var encoded_slice = zarrow.dictionaryEncode(gpa, sliced, value_type) catch return;
    defer encoded_slice.release();
    if (zarrow.concatArrayRefs(gpa, encoded.data().data_type, &.{ encoded, encoded_slice })) |joined| {
        var out = joined;
        defer out.release();
        expectValid("dictionary concat", out, len + sub_len);
    } else |_| {}
}

/// Record batch and table datums over a batch whose columns passed validateFull (the table
/// chunks every column at one random cut): take (with null indices), filter, drop_null, slice,
/// column selection, sort_indices on every sortable column and select_k_unstable. Every output
/// column must validate with the expected row count, drop_null must leave no null, and sort
/// results must be permutations (select_k: distinct in-range rows, `min(k, n)` of them).
fn exerciseTabular(gpa: Allocator, batch: *const zarrow.RecordBatch, rng: std.Random) !void {
    const compute = zarrow.compute;
    const n = batch.numRows();
    var ctx = compute.ExecContext.init(gpa, kernelRegistry());
    var rb = rb: {
        var copy = try batch.slice(0, n);
        break :rb compute.Datum.fromRecordBatch(gpa, copy) catch |e| {
            copy.deinit();
            std.debug.panic("fuzz-invariant: a batch of valid columns was rejected as a datum: {s}", .{@errorName(e)});
        };
    };
    defer rb.release();
    const cut = rng.uintAtMost(usize, n);
    const cols = try gpa.alloc(zarrow.ChunkedArray, batch.numColumns());
    var made: usize = 0;
    defer {
        for (cols[0..made]) |*c| c.release();
        gpa.free(cols);
    }
    for (batch.columns) |c| {
        var head = try c.slice(0, cut);
        defer head.release();
        var tail = try c.slice(cut, n - cut);
        defer tail.release();
        cols[made] = try zarrow.ChunkedArray.init(gpa, c.data().data_type, &.{ head, tail });
        made += 1;
    }
    var schema_ref = batch.schema_ref.retain();
    var table = zarrow.Table.init(gpa, schema_ref, cols) catch |e| {
        schema_ref.release();
        return e;
    };
    var t = compute.Datum.fromTable(gpa, table) catch |e| {
        table.deinit();
        std.debug.panic("fuzz-invariant: a table of valid columns was rejected as a datum: {s}", .{@errorName(e)});
    };
    defer t.release();

    var idx: [16]?usize = undefined;
    // No rows: no index (16 null rows of a type of huge width would be a huge, if legitimate,
    // output; with rows, every row width is backed by the input).
    const n_idx = if (n == 0) 0 else rng.uintAtMost(usize, idx.len);
    for (idx[0..n_idx]) |*ix| ix.* = if (rng.uintLessThan(u8, 8) == 0) null else rng.uintLessThan(usize, n);
    var pred_builder = try zarrow.BooleanBuilder.init(gpa, n);
    defer pred_builder.deinit();
    var kept: usize = 0;
    for (0..n) |_| if (rng.boolean()) {
        try pred_builder.append(true);
        kept += 1;
    } else try pred_builder.append(false);
    var pred = try pred_builder.finish();
    defer pred.release();
    const off = rng.uintAtMost(usize, n);
    const sub = rng.uintAtMost(usize, n - off);

    var keys: std.ArrayList(compute.SortKey) = .empty;
    defer keys.deinit(gpa);
    for (batch.schema().fields, batch.columns) |f, c| if (zarrow.compute.kernels.sort.isSortable(c.data().data_type)) {
        try keys.append(gpa, .{ .name = f.name, .order = if (rng.boolean()) .ascending else .descending, .null_placement = if (rng.boolean()) .at_start else .at_end });
    };

    for ([_]compute.Datum{ rb, t }) |d| {
        // A non-nullable field never gets a null row from a null index: InvalidInput.
        if (compute.datumTakeNullable(d, idx[0..n_idx])) |out| {
            var o = out;
            defer o.release();
            expectTabular("take", o, d.kind(), n_idx, false);
        } else |e| if (e != error.InvalidInput) std.debug.panic("fuzz-invariant: tabular take failed: {s}", .{@errorName(e)});
        var filtered = compute.datumFilter(d, .{ .array = pred }, .{}) catch |e| std.debug.panic("fuzz-invariant: tabular filter failed: {s}", .{@errorName(e)});
        expectTabular("filter", filtered, d.kind(), kept, false);
        filtered.release();
        var dropped = ctx.invoke("drop_null", .vector, &.{d}, .none) catch |e| std.debug.panic("fuzz-invariant: tabular drop_null failed: {s}", .{@errorName(e)});
        expectTabular("drop_null", dropped, d.kind(), dropped.length(), true);
        dropped.release();
        var sliced = compute.datumSlice(d, off, sub) catch |e| std.debug.panic("fuzz-invariant: tabular slice failed: {s}", .{@errorName(e)});
        expectTabular("slice", sliced, d.kind(), sub, false);
        sliced.release();
        if (batch.numColumns() > 0) {
            const name = batch.schema().fields[batch.numColumns() - 1].name;
            var picked = compute.datumSelectColumns(d, &.{ name, name }) catch |e| std.debug.panic("fuzz-invariant: tabular column selection failed: {s}", .{@errorName(e)});
            expectTabular("select", picked, d.kind(), n, false);
            picked.release();
        }
        if (keys.items.len == 0) continue;
        // A name several fields share is ambiguous (InvalidOptions, as Arrow).
        var sorted = ctx.invoke("sort_indices", .vector, &.{d}, .{ .sort = .{ .sort_keys = keys.items } }) catch |e| {
            if (e == error.InvalidOptions) continue;
            std.debug.panic("fuzz-invariant: tabular sort_indices failed: {s}", .{@errorName(e)});
        };
        expectPermutation("tabular sort_indices", sorted.array, n, false);
        sorted.release();
        const k = rng.uintAtMost(usize, n + 2);
        var top = ctx.invoke("select_k_unstable", .vector, &.{d}, .{ .select_k = .{ .k = k, .sort_keys = keys.items } }) catch |e| std.debug.panic("fuzz-invariant: tabular select_k_unstable failed: {s}", .{@errorName(e)});
        expectSelection("tabular select_k_unstable", top.array, n, @min(k, n));
        top.release();
    }
}

/// A record batch or table datum of `kind` with `rows` rows whose every column part passes
/// validateFull (and, for `no_nulls`, holds no null).
fn expectTabular(op: []const u8, d: zarrow.compute.Datum, kind: zarrow.compute.DatumKind, rows: usize, no_nulls: bool) void {
    if (d.kind() != kind or d.length() != rows) std.debug.panic("fuzz-invariant: tabular {s} gave a {s} of {d} rows, expected {d}", .{ op, @tagName(d.kind()), d.length(), rows });
    const check = struct {
        fn part(o: []const u8, a: ArrayRef, nn: bool) void {
            a.data().validateFull() catch |e| std.debug.panic("fuzz-invariant: tabular {s} column failed validateFull: {s}", .{ o, @errorName(e) });
            if (nn and (a.data().hasNulls() or (a.data().data_type == .null and a.data().length > 0))) std.debug.panic("fuzz-invariant: tabular {s} left a null", .{o});
        }
    };
    switch (d) {
        .record_batch => |b| for (b.get().columns) |c| {
            if (c.data().length != rows) std.debug.panic("fuzz-invariant: tabular {s} column length differs", .{op});
            check.part(op, c, no_nulls);
        },
        .table => |tb| for (tb.get().columns) |c| {
            if (c.len() != rows) std.debug.panic("fuzz-invariant: tabular {s} column length differs", .{op});
            for (c.chunks()) |part| check.part(op, part, no_nulls);
        },
        else => unreachable,
    }
}

/// A uint64 index array of `k` distinct rows of `[0, n)`.
fn expectSelection(op: []const u8, arr: ArrayRef, n: usize, k: usize) void {
    expectValid(op, arr, k);
    const vals = (zarrow.PrimitiveArray(u64){ .data = arr.data() }).values() catch std.debug.panic("fuzz-invariant: {s} output is not uint64", .{op});
    const seen = std.heap.page_allocator.alloc(bool, n) catch return;
    defer std.heap.page_allocator.free(seen);
    @memset(seen, false);
    for (vals) |v| {
        if (v >= n or seen[v]) std.debug.panic("fuzz-invariant: {s} gave row {d} of {d} twice or out of range", .{ op, v, n });
        seen[v] = true;
    }
}

fn expectValid(op: []const u8, arr: ArrayRef, expected_len: usize) void {
    arr.data().validateFull() catch |e| std.debug.panic("fuzz-invariant: {s} of a valid {s} array failed validateFull: {s}", .{ op, @tagName(arr.data().data_type.id()), @errorName(e) });
    if (arr.data().length != expected_len)
        std.debug.panic("fuzz-invariant: {s} returned {d} rows, expected {d}", .{ op, arr.data().length, expected_len });
}

/// Every kernel family, registered once per process (page allocator: outside the per-input cap).
var kernel_registry: ?zarrow.compute.FunctionRegistry = null;

fn kernelRegistry() *const zarrow.compute.FunctionRegistry {
    if (kernel_registry == null) {
        var registry = zarrow.compute.FunctionRegistry.init(std.heap.page_allocator);
        zarrow.compute.registerAll(&registry) catch |e| std.debug.panic("registerAll failed: {s}", .{@errorName(e)});
        kernel_registry = registry;
    }
    return &kernel_registry.?;
}

/// add / less / sum / min_max (numeric, also with operands of another numeric type: implicit
/// casts), min / max_element_wise, is_nan / is_finite, aggregate options, and_kleene /
/// and_not_kleene / any (bool), less and the string kernels (Unicode case mapping and predicates,
/// trims, slices, replace_slice, zero_fill), temporal extraction, rounding and interval
/// differences (dates, timestamps, times), comparisons of temporal, decimal and fixed-size
/// binary arrays, binary_join of list<string-like>, cast (numbers, strings, binaries, dates,
/// timestamps, times, durations, dictionaries), is_null and count (any type), is_in and a group
/// by (Grouper, hash_count_distinct, hash_sum) on every key type, decimal arithmetic, casts,
/// aggregates and grouped aggregates with options, and comparisons across layouts (a string /
/// binary array against itself in another layout, a timestamp against a finer unit, a decimal
/// against itself at another scale: every valid row must compare equal), on a decoded array that
/// passed validateFull and on a slice of it: the call must succeed (or fail with an expected error
/// such as InvalidCast) and an array output must validate with the input length (one row per
/// group for a grouped aggregate).
fn exerciseKernels(gpa: Allocator, arr: ArrayRef, sliced: ArrayRef) void {
    var ctx = zarrow.compute.ExecContext.initWithConfig(gpa, kernelRegistry(), .{ .overflow_mode = .wrapping });
    for ([_]ArrayRef{ arr, sliced }) |a| {
        const d: zarrow.compute.Datum = .{ .array = a };
        const len = a.data().length;
        const binary = [_]zarrow.compute.Datum{ d, d };
        // is_null / count reject unions and run-end encoded arrays (NoMatchingKernel).
        expectKernel(&ctx, "is_null", .scalar, &.{d}, len, true);
        expectKernel(&ctx, "count", .aggregate, &.{d}, null, true);
        expectCall(&ctx, "count", .aggregate, &.{d}, .{ .count = .{ .mode = .only_null } }, null, &.{error.NoMatchingKernel});
        expectCall(&ctx, "is_null", .scalar, &.{d}, .{ .null_options = .{ .nan_is_null = true } }, len, &.{error.NoMatchingKernel});
        const no_skip: zarrow.compute.Options = .{ .scalar_aggregate = .{ .skip_nulls = false, .min_count = 3 } };
        switch (a.data().data_type) {
            .int8, .int16, .int32, .int64, .uint8, .uint16, .uint32, .uint64, .float, .double => {
                expectKernel(&ctx, "add", .scalar, &binary, len, false);
                expectKernel(&ctx, "less", .scalar, &binary, len, false);
                expectKernel(&ctx, "sum", .aggregate, &.{d}, null, false);
                expectKernel(&ctx, "min_max", .aggregate, &.{d}, null, false);
                expectCall(&ctx, "mean", .aggregate, &.{d}, no_skip, null, &.{});
                expectCall(&ctx, "min_max", .aggregate, &.{d}, no_skip, null, &.{});
                expectKernel(&ctx, "is_nan", .scalar, &.{d}, len, false);
                expectKernel(&ctx, "is_finite", .scalar, &.{d}, len, false);
                expectCall(&ctx, "max_element_wise", .scalar, &binary, .none, len, &.{});
                expectCall(&ctx, "min_element_wise", .scalar, &.{ d, d, d }, .{ .element_wise_aggregate = .{ .skip_nulls = false } }, len, &.{});
                // Implicit casts to the common numeric type: a uint64 past int64 or an integer
                // past 2^53 does not cast safely.
                const wide: zarrow.compute.Datum = .{ .scalar = zarrow.compute.Scalar.init(.{ .int64 = {} }, .{ .i64 = 3 }) };
                const real: zarrow.compute.Datum = .{ .scalar = zarrow.compute.Scalar.init(.{ .double = {} }, .{ .f64 = 0.5 }) };
                expectCall(&ctx, "add", .scalar, &.{ d, wide }, .none, len, &.{error.InvalidCast});
                expectCall(&ctx, "less", .scalar, &.{ real, d }, .none, len, &.{error.InvalidCast});
                expectCall(&ctx, "max_element_wise", .scalar, &.{ d, wide }, .none, len, &.{error.InvalidCast});
            },
            .bool => {
                expectKernel(&ctx, "and_kleene", .scalar, &binary, len, false);
                expectKernel(&ctx, "and_not_kleene", .scalar, &binary, len, false);
                expectKernel(&ctx, "less", .scalar, &binary, len, false);
                expectKernel(&ctx, "any", .aggregate, &.{d}, null, false);
                expectCall(&ctx, "all", .aggregate, &.{d}, no_skip, null, &.{});
            },
            .string, .large_string, .string_view => {
                expectKernel(&ctx, "less", .scalar, &binary, len, false);
                // Mixed layouts compare bytes in place: the same values must compare equal.
                expectSameAfterCast(&ctx, a, if (a.data().data_type == .string_view) .{ .large_string = {} } else .{ .string_view = {} }, true);
                expectSameAfterCast(&ctx, a, .{ .binary = {} }, true);
                // validateFull passed, so the values are UTF-8: the utf8 kernels must succeed.
                expectCall(&ctx, "utf8_upper", .scalar, &.{d}, .none, len, &.{});
                inline for (.{ "utf8_lower", "utf8_swapcase", "utf8_title", "utf8_capitalize", "ascii_title", "utf8_is_lower", "utf8_is_title", "utf8_is_alnum", "utf8_is_printable", "ascii_is_upper", "utf8_is_numeric" }) |name| {
                    expectCall(&ctx, name, .scalar, &.{d}, .none, len, &.{});
                }
                expectCall(&ctx, "utf8_trim", .scalar, &.{d}, .{ .trim = .{ .characters = "a\u{e9} " } }, len, &.{});
                expectCall(&ctx, "utf8_replace_slice", .scalar, &.{d}, .{ .replace_slice = .{ .start = -2, .stop = 7, .replacement = "\u{df}" } }, len, &.{});
                expectCall(&ctx, "utf8_zero_fill", .scalar, &.{d}, .{ .zero_fill = .{ .width = 5 } }, len, &.{});
                expectCall(&ctx, "min_element_wise", .scalar, &binary, .none, len, &.{});
                expectCall(&ctx, "utf8_slice_codeunits", .scalar, &.{d}, .{ .slice = .{ .start = -3, .stop = 1, .step = -1 } }, len, &.{});
                expectCall(&ctx, "binary_join_element_wise", .scalar, &binary, .none, len, &.{});
                expectCall(&ctx, "utf8_split_whitespace", .scalar, &.{d}, .none, len, &.{});
                expectCall(&ctx, "replace_substring", .scalar, &.{d}, .{ .replace_substring = .{ .pattern = "a", .replacement = "\u{e9}\u{e9}" } }, len, &.{});
                // The regex kernels (src/compute/string_regex.zig) on valid UTF-8.
                expectCall(&ctx, "match_substring_regex", .scalar, &.{d}, .{ .match_substring = .{ .pattern = "(?i)\u{e9}|[a-z]+\\d" } }, len, &.{});
                expectCall(&ctx, "count_substring_regex", .scalar, &.{d}, .{ .match_substring = .{ .pattern = "\\w+|$" } }, len, &.{});
                expectCall(&ctx, "find_substring_regex", .scalar, &.{d}, .{ .match_substring = .{ .pattern = "\\pL\\b", .ignore_case = true } }, len, &.{});
                expectCall(&ctx, "replace_substring_regex", .scalar, &.{d}, .{ .replace_substring = .{ .pattern = "(?P<a>[a-z])(\\d)?", .replacement = "<\\2\\1>" } }, len, &.{});
                expectCall(&ctx, "replace_substring_regex", .scalar, &.{d}, .{ .replace_substring = .{ .pattern = "\\s*", .replacement = "_", .max_replacements = 3 } }, len, &.{});
                expectCall(&ctx, "split_pattern_regex", .scalar, &.{d}, .{ .split_pattern = .{ .pattern = "[,;]+|\\s" } }, len, &.{});
                expectCall(&ctx, "extract_regex", .scalar, &.{d}, .{ .extract_regex = .{ .pattern = "(?P<x>\\pL+)(?P<y>\\d*)" } }, len, &.{});
                expectCall(&ctx, "extract_regex_span", .scalar, &.{d}, .{ .extract_regex = .{ .pattern = "(?P<x>.)(?P<y>.)?" } }, len, &.{});
                expectCall(&ctx, "match_like", .scalar, &.{d}, .{ .match_substring = .{ .pattern = "%a_\\%%", .ignore_case = true } }, len, &.{});
                expectCall(&ctx, "match_substring", .scalar, &.{d}, .{ .match_substring = .{ .pattern = "\u{c9}A", .ignore_case = true } }, len, &.{});
                expectCall(&ctx, "cast", .scalar, &.{d}, cast(.{ .large_string = {} }, true), len, &.{});
                expectCall(&ctx, "cast", .scalar, &.{d}, cast(.{ .binary_view = {} }, true), len, &.{});
                // utf8_normalize (round 20): every form succeeds on valid UTF-8 and is idempotent.
                for ([_]zarrow.compute.Utf8NormalizeOptions.Form{ .nfc, .nfd, .nfkc, .nfkd }) |form| expectNormalized(&ctx, d, len, form);
                expectCall(&ctx, "cast", .scalar, &.{d}, cast(.{ .int32 = {} }, true), len, &.{error.InvalidCast});
                expectCall(&ctx, "cast", .scalar, &.{d}, cast(.{ .double = {} }, true), len, &.{error.InvalidCast});
            },
            .binary, .large_binary, .binary_view => {
                expectCall(&ctx, "cast", .scalar, &.{d}, cast(.{ .string = {} }, true), len, &.{ error.InvalidCast, error.Overflow });
                expectCall(&ctx, "binary_replace_slice", .scalar, &.{d}, .{ .replace_slice = .{ .start = 1, .stop = -1, .replacement = "\xff" } }, len, &.{});
                // Latin-1 regex: any byte is a character.
                expectCall(&ctx, "count_substring_regex", .scalar, &.{d}, .{ .match_substring = .{ .pattern = "\\C\\x00?" } }, len, &.{});
                expectCall(&ctx, "replace_substring_regex", .scalar, &.{d}, .{ .replace_substring = .{ .pattern = "[\\x80-\\xff]+", .replacement = "\xff" } }, len, &.{});
                expectCall(&ctx, "extract_regex", .scalar, &.{d}, .{ .extract_regex = .{ .pattern = "(?P<hi>[^\\x00-\\x7f])" } }, len, &.{});
                expectKernel(&ctx, "greater", .scalar, &binary, len, false);
                expectSameAfterCast(&ctx, a, if (a.data().data_type == .binary_view) .{ .binary = {} } else .{ .binary_view = {} }, true);
            },
            .fixed_size_binary, .duration => {
                expectKernel(&ctx, "less_equal", .scalar, &binary, len, false);
                expectCall(&ctx, "max_element_wise", .scalar, &binary, .none, len, &.{});
            },
            .decimal32, .decimal64, .decimal128, .decimal256 => exerciseDecimal(&ctx, a),
            .time32, .time64 => |t| if ((a.data().data_type == .time32) == (t.unit == .second or t.unit == .millisecond)) {
                expectKernel(&ctx, "hour", .scalar, &.{d}, len, false);
                expectKernel(&ctx, "subsecond", .scalar, &.{d}, len, false);
                expectKernel(&ctx, "milliseconds_between", .scalar, &binary, len, false);
                expectKernel(&ctx, "day_time_interval_between", .scalar, &binary, len, false);
                expectCall(&ctx, "floor_temporal", .scalar, &.{d}, .{ .round_temporal = .{ .multiple = 7, .unit = .minute } }, len, &.{});
                // A time is not bounded to a day by validateFull: a finer unit may overflow when safe.
                expectCall(&ctx, "cast", .scalar, &.{d}, cast(.{ .time64 = .{ .unit = .nanosecond } }, true), len, &.{error.InvalidCast});
                expectCall(&ctx, "cast", .scalar, &.{d}, cast(.{ .time32 = .{ .unit = .second } }, false), len, &.{});
            },
            .list, .large_list => |l| switch (l.value_field.data_type.*) {
                // binary_join: a null item gives a null row; every valid row joins.
                inline .string, .large_string, .binary, .large_binary => |_, tag| {
                    const sep = zarrow.compute.Scalar.init(@unionInit(DataType, @tagName(tag), {}), if (tag == .string or tag == .large_string) .{ .string = ", " } else .{ .binary = "\x00" });
                    expectCall(&ctx, "binary_join", .scalar, &.{ d, .{ .scalar = sep } }, .none, len, &.{});
                },
                else => {},
            },
            .date32, .date64, .timestamp => {
                // An unknown timezone name, or an ambiguous / nonexistent local result of a
                // rounding, is InvalidInput; casts may overflow when safe.
                const tz_ok = [_]anyerror{error.InvalidInput};
                expectCall(&ctx, "us_year", .scalar, &.{d}, .none, len, &tz_ok);
                expectCall(&ctx, "month_day_nano_interval_between", .scalar, &binary, .none, len, &tz_ok);
                expectCall(&ctx, "day_time_interval_between", .scalar, &binary, .none, len, &tz_ok);
                expectKernel(&ctx, "greater_equal", .scalar, &binary, len, false);
                expectCall(&ctx, "max_element_wise", .scalar, &binary, .none, len, &.{});
                expectCall(&ctx, "year", .scalar, &.{d}, .none, len, &tz_ok);
                expectCall(&ctx, "iso_calendar", .scalar, &.{d}, .none, len, &tz_ok);
                expectCall(&ctx, "floor_temporal", .scalar, &.{d}, .{ .round_temporal = .{ .multiple = 3, .unit = .week, .calendar_based_origin = true } }, len, &tz_ok);
                expectCall(&ctx, "round_temporal", .scalar, &.{d}, .{ .round_temporal = .{ .multiple = 2, .unit = .month } }, len, &tz_ok);
                expectCall(&ctx, "cast", .scalar, &.{d}, cast(.{ .date32 = {} }, false), len, &tz_ok);
                expectCall(&ctx, "cast", .scalar, &.{d}, cast(.{ .timestamp = .{ .unit = .nanosecond, .timezone = null } }, true), len, &.{error.InvalidCast});
                if (a.data().data_type == .timestamp) expectCall(&ctx, "cast", .scalar, &.{d}, cast(.{ .time64 = .{ .unit = .microsecond } }, false), len, &tz_ok);
                // Temporal text (round 20): an unknown timezone name, or %Z without one, is InvalidInput.
                expectCall(&ctx, "strftime", .scalar, &.{d}, .{ .strftime = .{ .format = "%c %G-W%V %j %Ez %Z %S" } }, len, &.{error.InvalidInput});
                expectCall(&ctx, "cast", .scalar, &.{d}, cast(.{ .string_view = {} }, true), len, &.{error.InvalidInput});
                // Against itself in a finer unit (the common unit): equal unless the cast overflows.
                switch (a.data().data_type) {
                    .timestamp => |ts| if (ts.unit != .nanosecond) expectSameAfterCast(&ctx, a, .{ .timestamp = .{ .unit = .nanosecond, .timezone = ts.timezone } }, false),
                    .date32 => expectSameAfterCast(&ctx, a, .{ .date64 = {} }, false),
                    else => {},
                }
            },
            .dictionary => |dict| {
                expectCall(&ctx, "cast", .scalar, &.{d}, cast(dict.value_type.*, true), len, &.{error.UnsupportedType});
                // Re-encoded with 16-bit indices: more distinct values than fit is InvalidCast.
                const re: DataType = .{ .dictionary = .{ .index_type = .{ .bit_width = 16, .signed = true }, .value_type = dict.value_type } };
                expectCall(&ctx, "cast", .scalar, &.{d}, cast(re, true), len, &.{ error.UnsupportedType, error.InvalidCast });
                // Comparisons decode dictionaries first (Arrow's DispatchBest).
                expectCall(&ctx, "equal", .scalar, &binary, .none, len, &.{ error.NoMatchingKernel, error.UnsupportedType });
            },
            else => {},
        }
        switch (a.data().data_type) {
            .bool, .int8, .int16, .int32, .int64, .uint8, .uint16, .uint32, .uint64, .float, .double => {
                expectCall(&ctx, "cast", .scalar, &.{d}, cast(.{ .int8 = {} }, false), len, &.{});
                expectCall(&ctx, "cast", .scalar, &.{d}, cast(.{ .int64 = {} }, true), len, &.{error.InvalidCast});
                expectCall(&ctx, "cast", .scalar, &.{d}, cast(.{ .string_view = {} }, true), len, &.{});
            },
            else => {},
        }
        if (zarrow.compute.kernels.hash_aggregate.isKeyType(a.data().data_type)) {
            expectCall(&ctx, "is_in", .scalar, &.{d}, .{ .set_lookup = .{ .value_set = .{ .array = sliced } } }, len, &.{});
        }
        // A dictionary is looked up decoded (against its own dictionary) and grouped by index.
        if (a.data().data_type == .dictionary and zarrow.compute.kernels.hash_aggregate.isGroupKeyType(a.data().data_type)) {
            expectCall(&ctx, "is_in", .scalar, &.{d}, .{ .set_lookup = .{ .value_set = .{ .array = a.data().dictionary.? } } }, len, &.{});
            expectCall(&ctx, "index_in", .scalar, &.{d}, .{ .set_lookup = .{ .value_set = .{ .array = a.data().dictionary.? }, .null_matching_behavior = .skip } }, len, &.{});
        }
        if (zarrow.compute.kernels.hash_aggregate.isGroupKeyType(a.data().data_type)) exerciseGroupBy(&ctx, a);
        exerciseRound16(&ctx, a);
        exerciseRound18(&ctx, a);
        exerciseRounds21To23(&ctx, a);
    }
}

/// Round 21 structural kernels on any decoded array: make_struct of the array with itself and a
/// scalar; list_value_length, list_element (index 0: an empty valid row is InvalidInput),
/// list_flatten (also recursive), list_parent_indices and list_slice (as the input layout and as
/// fixed_size_list) on the list layouts; struct_field of child 0 through structs and unions;
/// map_lookup of a zero / "a" key in maps with an integer or string key; binary_join of the
/// list layouts of every string / binary item layout; the fixed_size_binary string kernels. Each succeeds (or fails with a listed
/// error) and every array output validates (one row per input row for the scalar functions).
fn exerciseRound21(ctx: *zarrow.compute.ExecContext, a: ArrayRef) void {
    const d: zarrow.compute.Datum = .{ .array = a };
    const len = a.data().length;
    const one: zarrow.compute.Datum = .{ .scalar = zarrow.compute.Scalar.init(.{ .int32 = {} }, .{ .i32 = 1 }) };
    expectCall(ctx, "make_struct", .scalar, &.{ d, d, one }, .none, len, &.{});
    switch (a.data().data_type) {
        .list, .large_list, .list_view, .large_list_view, .fixed_size_list => {
            expectCall(ctx, "list_value_length", .scalar, &.{d}, .none, len, &.{});
            const zero: zarrow.compute.Datum = .{ .scalar = zarrow.compute.Scalar.init(.{ .int64 = {} }, .{ .i64 = 0 }) };
            expectCall(ctx, "list_element", .scalar, &.{ d, zero }, .none, len, &.{error.InvalidInput});
            expectAnyLength(ctx, "list_flatten", .vector, &.{d}, .none);
            expectAnyLength(ctx, "list_flatten", .vector, &.{d}, .{ .list_flatten = .{ .recursive = true } });
            expectAnyLength(ctx, "list_parent_indices", .vector, &.{d}, .none);
            expectCall(ctx, "list_slice", .scalar, &.{d}, .{ .list_slice = .{ .start = 1, .stop = 4, .step = 2 } }, len, &.{});
            expectCall(ctx, "list_slice", .scalar, &.{d}, .{ .list_slice = .{ .start = 0, .stop = 3, .return_fixed_size_list = true } }, len, &.{});
            const item = switch (a.data().data_type) {
                .list, .large_list => |l| l.value_field.data_type.*,
                .list_view, .large_list_view => |l| l.value_field.data_type.*,
                else => DataType{ .null = {} },
            };
            switch (item) {
                inline .string, .large_string, .string_view, .binary, .large_binary, .binary_view => |_, tag| {
                    const text = tag == .string or tag == .large_string or tag == .string_view;
                    const sep = zarrow.compute.Scalar.init(item, if (text) .{ .string = "-" } else .{ .binary = "\x00" });
                    expectCall(ctx, "binary_join", .scalar, &.{ d, .{ .scalar = sep } }, .none, len, &.{});
                },
                else => {},
            }
        },
        .struct_, .sparse_union, .dense_union => {
            const n = switch (a.data().data_type) {
                .struct_ => |st| st.fields.len,
                .sparse_union, .dense_union => |u| u.fields.len,
                else => unreachable,
            };
            if (n > 0) expectCall(ctx, "struct_field", .scalar, &.{d}, .{ .struct_field = .{ .indices = &.{0} } }, len, &.{});
        },
        .map => |m| {
            const key: ?zarrow.compute.Scalar = switch (m.key_field.data_type.*) {
                .int32 => zarrow.compute.Scalar.init(.{ .int32 = {} }, .{ .i32 = 0 }),
                .int64 => zarrow.compute.Scalar.init(.{ .int64 = {} }, .{ .i64 = 0 }),
                .string => zarrow.compute.Scalar.init(.{ .string = {} }, .{ .string = "a" }),
                else => null,
            };
            if (key) |k| for ([_]zarrow.compute.MapLookupOptions.Occurrence{ .first, .last, .all }) |occ| {
                expectCall(ctx, "map_lookup", .scalar, &.{d}, .{ .map_lookup = .{ .query_key = k, .occurrence = occ } }, len, &.{});
            };
        },
        .fixed_size_binary => {
            // As the binary layouts; a zero-width result is UnsupportedType (no fixed_size_binary[0]).
            expectCall(ctx, "binary_length", .scalar, &.{d}, .none, len, &.{});
            expectCall(ctx, "count_substring", .scalar, &.{d}, .{ .match_substring = .{ .pattern = "\x00", .ignore_case = true } }, len, &.{});
            expectCall(ctx, "find_substring", .scalar, &.{d}, .{ .match_substring = .{ .pattern = "a" } }, len, &.{});
            expectCall(ctx, "binary_slice", .scalar, &.{d}, .{ .slice = .{ .start = -3, .stop = 1, .step = -1 } }, len, &.{error.UnsupportedType});
            expectCall(ctx, "binary_replace_slice", .scalar, &.{d}, .{ .replace_slice = .{ .start = 1, .stop = 2, .replacement = "\xff" } }, len, &.{});
        },
        .string, .large_string, .string_view => {
            // A byte slice that splits a code point is InvalidInput on a string layout.
            expectCall(ctx, "binary_replace_slice", .scalar, &.{d}, .{ .replace_slice = .{ .start = 1, .stop = -1, .replacement = "\u{e9}" } }, len, &.{error.InvalidInput});
        },
        else => {},
    }
}

/// `name` must succeed (or fail with NoMatchingKernel) and an array output must validate.
fn expectAnyLength(ctx: *zarrow.compute.ExecContext, name: []const u8, kind: zarrow.compute.FunctionKind, args: []const zarrow.compute.Datum, options: zarrow.compute.Options) void {
    var out = ctx.invoke(name, kind, args, options) catch |e| std.debug.panic("fuzz-invariant: kernel {s} failed on a valid {s} array: {s}", .{ name, @tagName(args[0].dataType().id()), @errorName(e) });
    defer out.release();
    if (out.asArray()) |arr| arr.data().validateFull() catch |e| std.debug.panic("fuzz-invariant: {s} output failed validateFull: {s}", .{ name, @errorName(e) });
}

/// Round 18 kernels on any decoded array: min / max / min_max and hash_min_max of the ordered
/// non-numeric types, scatter and inverse_permutation of the reversed rows (their outputs must be
/// valid, the inverse a permutation), replace_with_mask (a mask of every third row, the array as
/// its own replacements), cumulative_sum / cumulative_max / pairwise_diff / winsorize / indices_nonzero
/// on numbers, float16 casts and promotion, and nested casts (list layouts among themselves, a
/// struct to its fields reversed, a map to a list of its entries): each succeeds or fails with a
/// listed error, and every array output validates with the expected length.
fn exerciseRound18(ctx: *zarrow.compute.ExecContext, a: ArrayRef) void {
    const gpa = std.heap.page_allocator;
    const d: zarrow.compute.Datum = .{ .array = a };
    const len = a.data().length;
    const dt = a.data().data_type;
    if (zarrow.compute.kernels.aggregate_basic.isOrderedType(dt)) {
        // UnsupportedType: min_max of an empty fixed_size_binary wider than 16 MiB.
        expectCall(ctx, "min_max", .aggregate, &.{d}, .none, null, &.{error.UnsupportedType});
        expectCall(ctx, "max", .aggregate, &.{d}, .{ .scalar_aggregate = .{ .skip_nulls = false } }, null, &.{});
        var ids_b = zarrow.UInt32Builder.init(gpa, len) catch return;
        defer ids_b.deinit();
        for (0..len) |i| ids_b.append(@intCast(i % 3)) catch return;
        var ids = ids_b.finish() catch return;
        defer ids.release();
        expectCall(ctx, "hash_min_max", .hash_aggregate, &.{ d, .{ .array = ids } }, .none, @min(len, 3), &.{});
    }
    // The reversed rows as int64 indices.
    var rev_b = zarrow.Int64Builder.init(gpa, len) catch return;
    defer rev_b.deinit();
    for (0..len) |i| rev_b.append(@intCast(len - 1 - i)) catch return;
    var rev = rev_b.finish() catch return;
    defer rev.release();
    expectCall(ctx, "scatter", .vector, &.{ d, .{ .array = rev } }, .none, len, &.{});
    var inv = ctx.invoke("inverse_permutation", .vector, &.{.{ .array = rev }}, .none) catch |e| std.debug.panic("fuzz-invariant: inverse_permutation failed: {s}", .{@errorName(e)});
    defer inv.release();
    expectValid("inverse_permutation", inv.array, len);
    for ((zarrow.PrimitiveArray(i64){ .data = inv.array.data() }).values() catch unreachable, 0..) |v, i| {
        if (v != @as(i64, @intCast(len - 1 - i))) std.debug.panic("fuzz-invariant: inverse_permutation of the reversed rows gave {d} at {d}", .{ v, i });
    }
    var mask_b = zarrow.BooleanBuilder.init(gpa, len) catch return;
    defer mask_b.deinit();
    for (0..len) |i| (if (i % 5 == 4) mask_b.appendNull() else mask_b.append(i % 3 == 0)) catch return;
    var mask = mask_b.finish() catch return;
    defer mask.release();
    expectCall(ctx, "replace_with_mask", .vector, &.{ d, .{ .array = mask }, d }, .none, len, &.{ error.NoMatchingKernel, error.UnsupportedType });
    switch (dt) {
        .int8, .int16, .int32, .int64, .uint8, .uint16, .uint32, .uint64, .float, .double => {
            expectCall(ctx, "cumulative_sum", .vector, &.{d}, .{ .cumulative = .{ .skip_nulls = true } }, len, &.{});
            expectCall(ctx, "cumulative_max", .vector, &.{d}, .none, len, &.{});
            expectCall(ctx, "cumulative_prod_checked", .vector, &.{d}, .none, len, &.{error.Overflow});
            expectCall(ctx, "cumulative_mean", .vector, &.{d}, .{ .cumulative = .{ .skip_nulls = true } }, len, &.{});
            expectCall(ctx, "pairwise_diff", .vector, &.{d}, .{ .pairwise = .{ .period = -2 } }, len, &.{});
            expectCall(ctx, "winsorize", .vector, &.{d}, .{ .winsorize = .{ .lower_limit = 0.1, .upper_limit = 0.8 } }, len, &.{});
            expectAnyLen(ctx, "indices_nonzero", &.{d}, .none, &.{});
        },
        .bool, .decimal32, .decimal128, .decimal256 => expectAnyLen(ctx, "indices_nonzero", &.{d}, .none, &.{}),
        .half_float => {
            expectCall(ctx, "cast", .scalar, &.{d}, cast(.{ .float = {} }, true), len, &.{});
            expectCall(ctx, "cast", .scalar, &.{d}, cast(.{ .string = {} }, true), len, &.{});
            expectCall(ctx, "add", .scalar, &.{ d, d }, .none, len, &.{});
            expectCall(ctx, "less", .scalar, &.{ d, d }, .none, len, &.{});
            expectKernel(ctx, "is_nan", .scalar, &.{d}, len, false);
        },
        .date32, .date64, .timestamp, .time32, .time64, .duration => expectCall(ctx, "pairwise_diff_checked", .vector, &.{d}, .none, len, &.{ error.Overflow, error.NotImplemented }),
        else => {},
    }
    exerciseNestedCasts(ctx, a);
}

/// Nested casts on a decoded list-like, struct or map array (see exerciseRound18).
fn exerciseNestedCasts(ctx: *zarrow.compute.ExecContext, a: ArrayRef) void {
    const d: zarrow.compute.Datum = .{ .array = a };
    const len = a.data().length;
    // A child type the cast may not support, a list size it does not match, a map to a fixed
    // size list, offsets past 32 bits, or a non-nullable target field over nulls.
    const allowed = [_]anyerror{ error.UnsupportedType, error.InvalidCast, error.Overflow, error.NotImplemented };
    switch (a.data().data_type) {
        inline .list, .large_list, .list_view, .large_list_view, .fixed_size_list => |l| {
            const f = l.value_field;
            for ([_]DataType{
                .{ .list = .{ .value_field = f } },
                .{ .large_list = .{ .value_field = f } },
                .{ .list_view = .{ .value_field = f } },
                .{ .large_list_view = .{ .value_field = f } },
                .{ .fixed_size_list = .{ .value_field = f, .list_size = 2 } },
            }) |to| expectCall(ctx, "cast", .scalar, &.{d}, cast(to, true), len, &allowed);
        },
        .map => |m| {
            const fields = [_]Field{ m.key_field, m.item_field };
            const entries = DataType{ .struct_ = .{ .fields = &fields } };
            expectCall(ctx, "cast", .scalar, &.{d}, cast(.{ .large_list = .{ .value_field = .{ .name = "e", .data_type = &entries } } }, true), len, &allowed);
            expectCall(ctx, "cast", .scalar, &.{d}, cast(.{ .map = .{ .key_field = m.key_field, .item_field = m.item_field, .keys_sorted = true } }, true), len, &allowed);
        },
        .struct_ => |st| {
            if (st.fields.len > 8) return;
            var reversed: [8]Field = undefined;
            for (reversed[0..st.fields.len], 0..) |*r, i| r.* = st.fields[st.fields.len - 1 - i];
            expectCall(ctx, "cast", .scalar, &.{d}, cast(.{ .struct_ = .{ .fields = reversed[0..st.fields.len] } }, true), len, &allowed);
        },
        else => {},
    }
}

/// Round 16 kernels on any decoded array: math and rounding (numbers, decimals), temporal
/// arithmetic, unique / value_counts, the multiplexers and null replacement, the new aggregates
/// and the sorts. A sort result must be a permutation of the rows and rank values in [1, len].
fn exerciseRound16(ctx: *zarrow.compute.ExecContext, a: ArrayRef) void {
    const d: zarrow.compute.Datum = .{ .array = a };
    const len = a.data().length;
    const dt = a.data().data_type;
    const sort = zarrow.compute.kernels.sort;
    const hash = zarrow.compute.kernels.hash_aggregate;
    switch (dt) {
        .int8, .int16, .int32, .int64, .uint8, .uint16, .uint32, .uint64, .float, .double => {
            // Integers go to float64 by a safe cast: past 2^53 it is InvalidCast (as Arrow).
            expectCall(ctx, "sqrt", .scalar, &.{d}, .none, len, &.{error.InvalidCast});
            expectCall(ctx, "power", .scalar, &.{ d, d }, .none, len, &.{error.InvalidInput});
            expectCall(ctx, "round", .scalar, &.{d}, .{ .round = .{ .ndigits = -1, .round_mode = .half_to_odd } }, len, &.{error.Overflow});
            expectCall(ctx, "round_to_multiple", .scalar, &.{d}, .{ .round_to_multiple = .{ .multiple = zarrow.compute.Scalar.init(.{ .int8 = {} }, .{ .i8 = 3 }), .round_mode = .up } }, len, &.{ error.Overflow, error.InvalidCast });
            expectCall(ctx, "variance", .aggregate, &.{d}, .{ .variance = .{ .ddof = 1 } }, null, &.{});
            expectCall(ctx, "kurtosis", .aggregate, &.{d}, .{ .skew = .{ .biased = false } }, null, &.{});
            expectCall(ctx, "product", .aggregate, &.{d}, .none, null, &.{});
            expectAnyLen(ctx, "mode", &.{d}, .{ .mode = .{ .n = 3 } }, &.{});
            const q = [_]f64{ 0, 0.3, 1 };
            expectCall(ctx, "quantile", .vector, &.{d}, .{ .quantile = .{ .q = &q, .interpolation = .nearest } }, q.len, &.{});
        },
        .decimal128, .decimal256 => {
            expectCall(ctx, "round", .scalar, &.{d}, .{ .round = .{ .ndigits = 1 } }, len, &.{ error.Overflow, error.InvalidOptions });
            expectCall(ctx, "floor", .scalar, &.{d}, .none, len, &.{ error.Overflow, error.InvalidOptions });
            expectCall(ctx, "min_max", .aggregate, &.{d}, .none, null, &.{});
            expectCall(ctx, "stddev", .aggregate, &.{d}, .none, null, &.{});
        },
        .timestamp => {
            expectCall(ctx, "subtract_checked", .scalar, &.{ d, d }, .none, len, &.{ error.Overflow, error.InvalidInput });
            const dur = zarrow.compute.Scalar.init(.{ .duration = .{ .unit = .second } }, .{ .duration = 3 });
            expectCall(ctx, "add", .scalar, &.{ d, .{ .scalar = dur } }, .none, len, &.{error.InvalidCast});
        },
        else => {},
    }
    if (hash.isGroupKeyType(dt)) {
        expectAnyLen(ctx, "unique", &.{d}, .none, &.{});
        expectValueCounts(ctx, d, len);
        if (dt != .dictionary) expectCall(ctx, "count_distinct", .aggregate, &.{d}, .{ .count = .{ .mode = .all } }, null, &.{});
        // dictionary_encode decodes back to the same rows.
        var enc = ctx.invoke("dictionary_encode", .vector, &.{d}, .none) catch |e| std.debug.panic("fuzz-invariant: dictionary_encode of a valid {s} array failed: {s}", .{ @tagName(dt.id()), @errorName(e) });
        defer enc.release();
        expectValid("dictionary_encode", enc.array, len);
    }
    // Multiplexers and null replacement take every layout the take kernels handle.
    expectCall(ctx, "coalesce", .scalar, &.{ d, d }, .none, len, &.{ error.UnsupportedType, error.NoMatchingKernel });
    expectAnyLen(ctx, "drop_null", &.{d}, .none, &.{});
    expectCall(ctx, "fill_null_backward", .vector, &.{d}, .none, len, &.{});
    if (sort.isSortable(dt)) {
        for ([_]zarrow.compute.Options{ .none, .{ .sort = .{ .order = .descending, .null_placement = .at_start } } }) |o| {
            var out = ctx.invoke("sort_indices", .vector, &.{ d, d }, o) catch |e| std.debug.panic("fuzz-invariant: sort_indices of a valid {s} array failed: {s}", .{ @tagName(dt.id()), @errorName(e) });
            defer out.release();
            expectPermutation("sort_indices", out.array, len, false);
        }
        var r = ctx.invoke("rank", .vector, &.{d}, .{ .rank = .{ .tiebreaker = .dense } }) catch |e| std.debug.panic("fuzz-invariant: rank of a valid {s} array failed: {s}", .{ @tagName(dt.id()), @errorName(e) });
        defer r.release();
        expectPermutation("rank", r.array, len, true);
        expectCall(ctx, "select_k_unstable", .vector, &.{d}, .{ .select_k = .{ .k = 5 } }, @min(len, 5), &.{});
        for ([_]zarrow.compute.SortNullPlacement{ .at_end, .at_start }) |placement| exerciseSelections(ctx, d, len, placement);
    }
}

/// select_k_unstable must give the prefix of the stable sort, partition_nth_indices a
/// permutation with the sort's row at the pivot and the sort's first rows before it,
/// rank_quantile values in (0, 1], and search_sorted of the rows in the sorted rows positions in
/// [0, len].
fn exerciseSelections(ctx: *zarrow.compute.ExecContext, d: zarrow.compute.Datum, len: usize, placement: zarrow.compute.SortNullPlacement) void {
    const gpa = std.heap.page_allocator;
    const keys = [_]zarrow.compute.SortKey{.{ .null_placement = placement }};
    var sorted = ctx.invoke("sort_indices", .vector, &.{d}, .{ .sort = .{ .sort_keys = &keys } }) catch |e| std.debug.panic("fuzz-invariant: sort_indices failed: {s}", .{@errorName(e)});
    defer sorted.release();
    const order = (zarrow.PrimitiveArray(u64){ .data = sorted.array.data() }).values() catch unreachable;
    const k = len / 2 + 1;
    var top = ctx.invoke("select_k_unstable", .vector, &.{d}, .{ .select_k = .{ .k = k, .sort_keys = &keys } }) catch |e| std.debug.panic("fuzz-invariant: select_k_unstable failed: {s}", .{@errorName(e)});
    defer top.release();
    expectValid("select_k_unstable", top.array, @min(k, len));
    const got = (zarrow.PrimitiveArray(u64){ .data = top.array.data() }).values() catch unreachable;
    if (!std.mem.eql(u64, got, order[0..@min(k, len)])) std.debug.panic("fuzz-invariant: select_k_unstable is not the prefix of the sort", .{});
    const pivot = len / 3;
    var part = ctx.invoke("partition_nth_indices", .vector, &.{d}, .{ .partition_nth = .{ .pivot = pivot, .null_placement = placement } }) catch |e| std.debug.panic("fuzz-invariant: partition_nth_indices failed: {s}", .{@errorName(e)});
    defer part.release();
    expectPermutation("partition_nth_indices", part.array, len, false);
    const rows = (zarrow.PrimitiveArray(u64){ .data = part.array.data() }).values() catch unreachable;
    if (pivot < len) {
        if (rows[pivot] != order[pivot]) std.debug.panic("fuzz-invariant: partition_nth_indices put row {d} at the pivot, the sort row {d}", .{ rows[pivot], order[pivot] });
        const before = gpa.alloc(bool, len) catch return;
        defer gpa.free(before);
        @memset(before, false);
        for (order[0..pivot]) |r| before[r] = true;
        for (rows[0..pivot]) |r| if (!before[r]) std.debug.panic("fuzz-invariant: partition_nth_indices put row {d} before the pivot", .{r});
    }
    var q = ctx.invoke("rank_quantile", .vector, &.{d}, .{ .rank_quantile = .{ .null_placement = placement } }) catch |e| std.debug.panic("fuzz-invariant: rank_quantile failed: {s}", .{@errorName(e)});
    defer q.release();
    expectValid("rank_quantile", q.array, len);
    for ((zarrow.PrimitiveArray(f64){ .data = q.array.data() }).values() catch unreachable) |v| {
        if (!(v > 0 and v <= 1)) std.debug.panic("fuzz-invariant: rank_quantile gave {d}", .{v});
    }
    if (placement != .at_end or d.dataType() == .struct_) return; // search_sorted has no struct kernel
    var ascending = zarrow.computeDatumTake(d, @as([]const usize, @ptrCast(order))) catch return;
    defer ascending.release();
    var pos = ctx.invoke("search_sorted", .vector, &.{ ascending, d }, .{ .search_sorted = .{ .side = .right } }) catch |e| std.debug.panic("fuzz-invariant: search_sorted failed: {s}", .{@errorName(e)});
    defer pos.release();
    expectValid("search_sorted", pos.array, len);
    const at = (zarrow.PrimitiveArray(u64){ .data = pos.array.data() }).values() catch unreachable;
    for (at, 0..) |v, i| if (!pos.array.data().isNull(i) and v > len) std.debug.panic("fuzz-invariant: search_sorted gave {d} of {d}", .{ v, len });
}

/// value_counts must validate, its counts add up to the rows, and (a dictionary) its values keep
/// the input's dictionary.
fn expectValueCounts(ctx: *zarrow.compute.ExecContext, d: zarrow.compute.Datum, len: usize) void {
    var out = ctx.invoke("value_counts", .vector, &.{d}, .none) catch |e| std.debug.panic("fuzz-invariant: value_counts failed on a valid {s} array: {s}", .{ @tagName(d.dataType().id()), @errorName(e) });
    defer out.release();
    expectValid("value_counts", out.array, out.array.data().length);
    const children = out.array.data().children;
    const counts = (zarrow.PrimitiveArray(i64){ .data = children[1].data() }).values() catch std.debug.panic("fuzz-invariant: value_counts counts are not int64", .{});
    var total: usize = 0;
    for (counts) |c| total += @intCast(c);
    if (total != len) std.debug.panic("fuzz-invariant: value_counts counts add up to {d}, expected {d}", .{ total, len });
    if (d.dataType() == .dictionary and children[0].data().dictionary.?.node != d.array.data().dictionary.?.node)
        std.debug.panic("fuzz-invariant: value_counts of a dictionary changed the dictionary", .{});
}

/// Invoke vector function `name`: an error outside `allowed` is a finding, the output array
/// must validate (any length).
fn expectAnyLen(ctx: *zarrow.compute.ExecContext, name: []const u8, args: []const zarrow.compute.Datum, options: zarrow.compute.Options, allowed: []const anyerror) void {
    var out = ctx.invoke(name, .vector, args, options) catch |e| {
        for (allowed) |ok| if (e == ok) return;
        std.debug.panic("fuzz-invariant: kernel {s} failed on a valid {s} array: {s}", .{ name, @tagName(args[0].dataType().id()), @errorName(e) });
    };
    defer out.release();
    const a = out.asArray() orelse std.debug.panic("fuzz-invariant: {s} did not return an array", .{name});
    expectValid(name, a, a.data().length);
}

/// A uint64 index array of `len` rows: a permutation of `[0, len)` (or, for `ranks`, values in
/// `[1, len]`).
fn expectPermutation(op: []const u8, arr: ArrayRef, len: usize, ranks: bool) void {
    expectValid(op, arr, len);
    const vals = (zarrow.PrimitiveArray(u64){ .data = arr.data() }).values() catch std.debug.panic("fuzz-invariant: {s} output is not uint64", .{op});
    var seen = std.heap.page_allocator.alloc(bool, len) catch return;
    defer std.heap.page_allocator.free(seen);
    @memset(seen, false);
    for (vals) |v| {
        if (ranks) {
            if (v < 1 or v > len) std.debug.panic("fuzz-invariant: {s} gave rank {d} of {d} rows", .{ op, v, len });
            continue;
        }
        if (v >= len or seen[v]) std.debug.panic("fuzz-invariant: {s} is not a permutation (row {d} of {d})", .{ op, v, len });
        seen[v] = true;
    }
}

/// Group the rows of `a` by their own values: the group ids must validate with one row per input
/// row, and the grouped aggregates must give one row per group.
fn exerciseGroupBy(ctx: *zarrow.compute.ExecContext, a: ArrayRef) void {
    const hash = zarrow.compute.kernels.hash_aggregate;
    var grouper = hash.Grouper.init(ctx.allocator, &.{a.data().data_type}) catch |e| std.debug.panic("fuzz-invariant: Grouper.init failed: {s}", .{@errorName(e)});
    defer grouper.deinit();
    var ids = grouper.consume(&.{a}) catch |e| std.debug.panic("fuzz-invariant: Grouper.consume failed on a valid {s} array: {s}", .{ @tagName(a.data().data_type.id()), @errorName(e) });
    defer ids.release();
    expectValid("Grouper.consume", ids, a.data().length);
    const groups = grouper.numGroups();
    const keys = grouper.uniques() catch |e| std.debug.panic("fuzz-invariant: Grouper.uniques failed: {s}", .{@errorName(e)});
    defer {
        for (keys) |*k| k.release();
        ctx.allocator.free(keys);
    }
    expectValid("Grouper.uniques", keys[0], groups);
    const args = [_]zarrow.compute.Datum{ .{ .array = a }, .{ .array = ids } };
    expectCall(ctx, "hash_count_distinct", .hash_aggregate, &args, .none, groups, &.{});
    expectCall(ctx, "hash_first_last", .hash_aggregate, &args, .{ .scalar_aggregate = .{ .skip_nulls = false } }, groups, &.{});
    expectCall(ctx, "hash_list", .hash_aggregate, &args, .none, groups, &.{});
    switch (a.data().data_type) {
        .int8, .int16, .int32, .int64, .uint8, .uint16, .uint32, .uint64, .float, .double => {
            expectCall(ctx, "hash_sum", .hash_aggregate, &args, .none, groups, &.{});
            expectCall(ctx, "hash_variance", .hash_aggregate, &args, .none, groups, &.{});
            expectCall(ctx, "hash_product", .hash_aggregate, &args, .none, groups, &.{});
            expectCall(ctx, "hash_tdigest", .hash_aggregate, &args, .{ .tdigest = .{ .q = &.{ 0, 0.5 }, .delta = 10, .buffer_size = 5 } }, groups, &.{});
            expectCall(ctx, "hash_approximate_median", .hash_aggregate, &args, .none, groups, &.{});
            expectTDigest(ctx, args[0]);
        },
        else => {},
    }
    switch (a.data().data_type) {
        .string, .large_string, .binary, .large_binary, .fixed_size_binary, .int8, .int16, .int32, .int64, .uint8, .uint16, .uint32, .uint64 => exercisePivot(ctx, a, ids, groups),
        else => {},
    }
}

/// Decimal kernels on a decoded decimal array (any stored value, precision from the schema):
/// arithmetic with itself and an integer, negate / abs / sign, comparisons at another scale, casts
/// (a text round trip must give the values back), aggregates and grouped aggregates.
fn exerciseDecimal(ctx: *zarrow.compute.ExecContext, a: ArrayRef) void {
    const d: zarrow.compute.Datum = .{ .array = a };
    const len = a.data().length;
    const dt = a.data().data_type;
    const binary = [_]zarrow.compute.Datum{ d, d };
    const p = switch (dt) {
        .decimal32, .decimal64, .decimal128, .decimal256 => |x| x,
        else => unreachable,
    };
    expectKernel(ctx, "less_equal", .scalar, &binary, len, false);
    // A result precision past the width is Overflow, a negative scale NotImplemented, a zero
    // divisor DivideByZero; the checked variants also report values past the precision.
    const arith_ok = [_]anyerror{ error.Overflow, error.NotImplemented, error.DivideByZero };
    inline for (.{ "add", "subtract", "multiply", "divide", "add_checked", "subtract_checked", "multiply_checked", "divide_checked" }) |name| {
        expectCall(ctx, name, .scalar, &binary, .none, len, &arith_ok);
    }
    const three: zarrow.compute.Datum = .{ .scalar = zarrow.compute.Scalar.init(.{ .int16 = {} }, .{ .i16 = 3 }) };
    expectCall(ctx, "multiply", .scalar, &.{ d, three }, .none, len, &arith_ok);
    expectCall(ctx, "less", .scalar, &.{ three, d }, .none, len, &.{});
    const half: zarrow.compute.Datum = .{ .scalar = zarrow.compute.Scalar.init(.{ .double = {} }, .{ .f64 = 0.5 }) };
    expectCall(ctx, "add", .scalar, &.{ d, half }, .none, len, &.{});
    if (dt == .decimal128 or dt == .decimal256) {
        inline for (.{ "negate", "abs", "sign" }) |name| expectCall(ctx, name, .scalar, &.{d}, .none, len, &.{});
        inline for (.{ "negate_checked", "abs_checked" }) |name| expectCall(ctx, name, .scalar, &.{d}, .none, len, &.{error.Overflow});
        inline for (.{ "sum", "mean", "min", "max" }) |name| {
            expectCall(ctx, name, .aggregate, &.{d}, .none, null, &.{});
            expectCall(ctx, name, .aggregate, &.{d}, .{ .scalar_aggregate = .{ .skip_nulls = false, .min_count = 0 } }, null, &.{});
        }
        expectCall(ctx, "max_element_wise", .scalar, &binary, .none, len, &.{});
        exerciseGroupedDecimal(ctx, a);
    }
    // The same values at a scale one larger (a wider type when the precision is full): equal
    // unless the stored values exceed what the wider scale holds.
    const wider: DataType = .{ .decimal256 = .{ .precision = @min(76, p.precision + 1), .scale = p.scale +| 1 } };
    expectSameAfterCast(ctx, a, wider, false);
    expectCall(ctx, "cast", .scalar, &.{d}, cast(.{ .string = {} }, true), len, &.{});
    expectCall(ctx, "cast", .scalar, &.{d}, cast(.{ .double = {} }, true), len, &.{});
    expectCall(ctx, "cast", .scalar, &.{d}, cast(.{ .float = {} }, true), len, &.{});
    expectCall(ctx, "cast", .scalar, &.{d}, cast(.{ .int64 = {} }, false), len, &.{});
    expectCall(ctx, "cast", .scalar, &.{d}, cast(.{ .int8 = {} }, true), len, &.{error.InvalidCast});
    expectCall(ctx, "cast", .scalar, &.{d}, cast(.{ .decimal128 = .{ .precision = 38, .scale = p.scale -| 1 } }, true), len, &.{error.InvalidCast});
    expectCall(ctx, "cast", .scalar, &.{d}, cast(.{ .decimal32 = .{ .precision = 9, .scale = p.scale } }, false), len, &.{});
    // Text round trip (unsafe back: no precision check): the stored values come back, unless
    // the text has an exponent past the width's largest precision (InvalidCast).
    var text = ctx.invoke("cast", .scalar, &.{d}, cast(.{ .large_string = {} }, true)) catch |e| std.debug.panic("fuzz-invariant: decimal to text failed: {s}", .{@errorName(e)});
    defer text.release();
    var back = ctx.invoke("cast", .scalar, &.{text}, cast(dt, false)) catch |e| {
        if (e == error.InvalidCast and p.scale < 0) return;
        std.debug.panic("fuzz-invariant: decimal text round trip failed: {s}", .{@errorName(e)});
    };
    defer back.release();
    expectAllEqual(ctx, d, back);
}

/// hash_sum / hash_mean / hash_min / hash_max (with options) of decimal values in 3 groups.
fn exerciseGroupedDecimal(ctx: *zarrow.compute.ExecContext, a: ArrayRef) void {
    const len = a.data().length;
    var b = zarrow.UInt32Builder.init(ctx.allocator, len) catch return;
    defer b.deinit();
    for (0..len) |i| b.append(@intCast(i % 3)) catch return;
    var ids = b.finish() catch return;
    defer ids.release();
    const groups: usize = @min(len, 3);
    const args = [_]zarrow.compute.Datum{ .{ .array = a }, .{ .array = ids } };
    inline for (.{ "hash_sum", "hash_mean", "hash_min", "hash_max" }) |name| {
        expectCall(ctx, name, .hash_aggregate, &args, .none, groups, &.{});
        expectCall(ctx, name, .hash_aggregate, &args, .{ .scalar_aggregate = .{ .skip_nulls = false, .min_count = 2 } }, groups, &.{});
    }
    expectCall(ctx, "hash_count", .hash_aggregate, &args, .{ .count = .{ .mode = .only_null } }, groups, &.{});
    // Decimal products past the precision wrap here (overflow_mode wrapping); 3 groups.
    if (a.data().data_type == .decimal128 or a.data().data_type == .decimal256) {
        // A negative scale (or one past the type's digits) has no product (NotImplemented).
        expectCall(ctx, "hash_product", .hash_aggregate, &args, .none, groups, &.{error.NotImplemented});
        expectCall(ctx, "product", .aggregate, &.{args[0]}, .{ .scalar_aggregate = .{ .min_count = 0 } }, null, &.{error.NotImplemented});
        expectCall(ctx, "hash_tdigest", .hash_aggregate, &args, .{ .tdigest = .{ .q = &.{ 0.5, 1 }, .delta = 11, .buffer_size = 3 } }, groups, &.{});
        expectTDigest(ctx, args[0]);
    }
}

/// tdigest at q 0, 0.25, 0.5, 0.75 and 1 with a small digest (delta 12, buffer 7) and with the
/// defaults: the quantiles must not decrease, and approximate_median must lie between the
/// first and the last (both within 1e-9 of the value range: interpolation rounds).
fn expectTDigest(ctx: *zarrow.compute.ExecContext, d: zarrow.compute.Datum) void {
    const qs = [_]f64{ 0, 0.25, 0.5, 0.75, 1 };
    for ([_]zarrow.compute.TDigestOptions{ .{ .q = &qs, .delta = 12, .buffer_size = 7 }, .{ .q = &qs } }) |o| {
        var out = ctx.invoke("tdigest", .aggregate, &.{d}, .{ .tdigest = o }) catch |e| std.debug.panic("fuzz-invariant: tdigest failed: {s}", .{@errorName(e)});
        defer out.release();
        expectValid("tdigest", out.array, qs.len);
        if (out.array.data().hasNulls()) continue;
        const r = (zarrow.PrimitiveArray(f64){ .data = out.array.data() }).values() catch unreachable;
        // Values past the float64 range (decimal256, a negative scale) give infinities, and
        // centroids mixing them NaN, as in Arrow: no order to check then.
        const finite = for (r) |x| {
            if (!std.math.isFinite(x)) break false;
        } else true;
        if (!finite) continue;
        // Interpolation rounds: quantiles may step back by a rounding error of the value range.
        const tol = 1e-9 * (@abs(r[0]) + @abs(r[r.len - 1]));
        for (r[0 .. r.len - 1], r[1..]) |x, y| if (!(x <= y + tol)) std.debug.panic("fuzz-invariant: tdigest quantiles decrease: {d} then {d} in {any}", .{ x, y, r });
        var m = ctx.invoke("approximate_median", .aggregate, &.{d}, .{ .scalar_aggregate = .{ .min_count = 0 } }) catch |e| std.debug.panic("fuzz-invariant: approximate_median failed: {s}", .{@errorName(e)});
        defer m.release();
        if (m.scalar.isNull()) continue;
        const v = m.scalar.value.f64;
        if (!(v >= r[0] - tol and v <= r[r.len - 1] + tol)) std.debug.panic("fuzz-invariant: approximate_median {d} outside [{d}, {d}]", .{ v, r[0], r[r.len - 1] });
    }
}

/// pivot_wider and hash_pivot_wider with the array as keys and as values: duplicates, null keys
/// and names that do not parse as the key type are errors, anything else must validate.
fn exercisePivot(ctx: *zarrow.compute.ExecContext, a: ArrayRef, ids: ArrayRef, groups: usize) void {
    // Without rows, a value type of huge width (from the schema) would make huge null fields.
    if (a.data().length == 0) return;
    const d: zarrow.compute.Datum = .{ .array = a };
    const names = [_][]const u8{ "0", "1", "a", "-7" };
    const ok = [_]anyerror{ error.InvalidInput, error.InvalidOptions };
    for ([_]zarrow.compute.UnexpectedKeyBehavior{ .ignore, .raise }) |b| {
        const o: zarrow.compute.Options = .{ .pivot_wider = .{ .key_names = &names, .unexpected_key_behavior = b } };
        var out = ctx.invoke("pivot_wider", .aggregate, &.{ d, d }, o) catch |e| {
            if (e == error.InvalidInput or e == error.InvalidOptions) continue;
            std.debug.panic("fuzz-invariant: pivot_wider failed: {s}", .{@errorName(e)});
        };
        out.release();
        expectCall(ctx, "hash_pivot_wider", .hash_aggregate, &.{ d, d, .{ .array = ids } }, o, groups, &ok);
    }
}

/// `a` against itself cast to `to` (unsafe when `safe` is false: InvalidCast is then not
/// expected; a safe cast may fail with InvalidCast / Overflow): `equal` must hold on every valid
/// row, and `less` must succeed.
fn expectSameAfterCast(ctx: *zarrow.compute.ExecContext, a: ArrayRef, to: DataType, safe: bool) void {
    var other = ctx.invoke("cast", .scalar, &.{.{ .array = a }}, cast(to, true)) catch |e| {
        if (!safe and (e == error.InvalidCast or e == error.Overflow)) return;
        if (safe and e == error.InvalidCast) return;
        std.debug.panic("fuzz-invariant: cast of a valid {s} array to {s} failed: {s}", .{ @tagName(a.data().data_type.id()), @tagName(to.id()), @errorName(e) });
    };
    defer other.release();
    expectAllEqual(ctx, .{ .array = a }, other);
    expectCall(ctx, "less", .scalar, &.{ .{ .array = a }, other }, .none, a.data().length, &.{});
}

/// `equal(x, y)` must be true on every valid row.
fn expectAllEqual(ctx: *zarrow.compute.ExecContext, x: zarrow.compute.Datum, y: zarrow.compute.Datum) void {
    var eq = ctx.invoke("equal", .scalar, &.{ x, y }, .none) catch |e| std.debug.panic("fuzz-invariant: equal across {s} / {s} failed: {s}", .{ @tagName(x.dataType().id()), @tagName(y.dataType().id()), @errorName(e) });
    defer eq.release();
    const data = eq.array.data();
    for (0..data.length) |i| {
        if (data.isNull(i)) continue;
        const bit = data.offset + i;
        if ((data.buffers[1].data[bit / 8] >> @intCast(bit % 8)) & 1 == 0)
            std.debug.panic("fuzz-invariant: row {d} of a {s} array differs from itself as {s}", .{ i, @tagName(x.dataType().id()), @tagName(y.dataType().id()) });
    }
}

fn cast(to: DataType, safe: bool) zarrow.compute.Options {
    return .{ .cast = .{ .to_type = to, .safe = safe } };
}

fn expectKernel(ctx: *zarrow.compute.ExecContext, name: []const u8, kind: zarrow.compute.FunctionKind, args: []const zarrow.compute.Datum, expected_len: ?usize, unsupported_ok: bool) void {
    expectCall(ctx, name, kind, args, .none, expected_len, if (unsupported_ok) &.{error.NoMatchingKernel} else &.{});
}

/// Invoke `name`; an error outside `allowed` is a finding, and an array output must validate with
/// `expected_len` rows.
/// `utf8_normalize` of `d` in `form` must validate with `len` rows and stay the same when
/// normalized again.
fn expectNormalized(ctx: *zarrow.compute.ExecContext, d: zarrow.compute.Datum, len: usize, form: zarrow.compute.Utf8NormalizeOptions.Form) void {
    const opts: zarrow.compute.Options = .{ .utf8_normalize = .{ .form = form } };
    var once = ctx.invoke("utf8_normalize", .scalar, &.{d}, opts) catch |e| std.debug.panic("fuzz-invariant: utf8_normalize failed on valid UTF-8: {s}", .{@errorName(e)});
    defer once.release();
    expectValid("utf8_normalize", once.array, len);
    var twice = ctx.invoke("utf8_normalize", .scalar, &.{once}, opts) catch |e| std.debug.panic("fuzz-invariant: utf8_normalize failed on its output: {s}", .{@errorName(e)});
    defer twice.release();
    for (0..len) |i| {
        if (once.array.data().isNull(i)) continue;
        const x = zarrow.compute.kernels.string_normalize.rowAt(once.array, i);
        const y = zarrow.compute.kernels.string_normalize.rowAt(twice.array, i);
        if (!std.mem.eql(u8, x, y)) std.debug.panic("fuzz-invariant: utf8_normalize is not idempotent", .{});
    }
}

fn expectCall(ctx: *zarrow.compute.ExecContext, name: []const u8, kind: zarrow.compute.FunctionKind, args: []const zarrow.compute.Datum, options: zarrow.compute.Options, expected_len: ?usize, allowed: []const anyerror) void {
    var out = ctx.invoke(name, kind, args, options) catch |e| {
        for (allowed) |ok| if (e == ok) return;
        std.debug.panic("fuzz-invariant: kernel {s} failed on a valid {s} array: {s}", .{ name, @tagName(args[0].dataType().id()), @errorName(e) });
    };
    defer out.release();
    if (out.asArray()) |a| expectValid(name, a, expected_len orelse std.debug.panic("fuzz-invariant: {s} returned an array", .{name}));
}

// ---------------------------------------------------------------------------------------------
// regex: the regex compiler and matcher (src/regex/) on a pattern and a text from the input.
//
// Input: a flag byte (bit 0 Latin-1, bit 1 case-insensitive, bit 2 literal), the pattern length
// (u16, little-endian), the pattern, then the text. A pattern that does not compile is a pass;
// one that compiles must stay inside the program limits, and every search must agree between
// the DFA path (`find`), the Pike VM and `captures`, with spans inside the text and groups
// inside their match.

/// Longest text the regex target matches (matching is O(text x program)).
const regex_max_text = 4096;

fn regexTarget(gpa: Allocator, bytes: []const u8, fb: *Feedback) !void {
    if (bytes.len < 3) return error.EndOfStream;
    const flags = bytes[0];
    const plen = @min(std.mem.readInt(u16, bytes[1..3], .little), bytes.len - 3);
    const pattern = bytes[3..][0..plen];
    const rest = bytes[3 + plen ..];
    const text = rest[0..@min(rest.len, regex_max_text)];
    const Regex = zarrow.regex.Regex;
    var diag: zarrow.regex.Diagnostic = .{};
    const re = Regex.compile(gpa, pattern, .{ .latin1 = flags & 1 != 0, .case_insensitive = flags & 2 != 0, .literal = flags & 4 != 0 }, &diag) catch |e| {
        if (e != error.OutOfMemory and diag.message.len == 0) std.debug.panic("fuzz-invariant: regex error {s} without a message", .{@errorName(e)});
        fb.types = @as(u64, 1) << @intCast(@intFromError(e) % 64);
        return e;
    };
    defer re.deinit();
    if (re.forward.insts.len > zarrow.regex.prog.max_insts or re.backward.insts.len > zarrow.regex.prog.max_insts)
        std.debug.panic("fuzz-invariant: regex program over the instruction limit", .{});
    const groups = try gpa.alloc(?zarrow.regex.Span, re.groupCount() + 1);
    defer gpa.free(groups);
    var slots: [2]usize = undefined;
    const any = try re.isMatch(text);
    if (any != try re.pike.search(&re.forward, text, 0, false, &slots)) std.debug.panic("fuzz-invariant: regex isMatch and the Pike VM disagree", .{});
    var pos: usize = 0;
    var n: u32 = 0;
    while (pos <= text.len and n < 64) : (n += 1) {
        const m = try re.find(text, pos) orelse {
            if (n == 0 and any) std.debug.panic("fuzz-invariant: regex isMatch without a match", .{});
            break;
        };
        if (m.start < pos or m.end < m.start or m.end > text.len) std.debug.panic("fuzz-invariant: regex span out of range", .{});
        if (!try re.pike.search(&re.forward, text, pos, false, &slots) or slots[0] != m.start or slots[1] != m.end)
            std.debug.panic("fuzz-invariant: regex DFA span [{d}, {d}) and Pike VM span differ", .{ m.start, m.end });
        if (!try re.captures(text, pos, groups) or groups[0].?.start != m.start or groups[0].?.end != m.end)
            std.debug.panic("fuzz-invariant: regex captures disagree with find", .{});
        for (groups[1..]) |g| if (g) |sp| {
            if (sp.start > sp.end or sp.start < m.start or sp.end > m.end) std.debug.panic("fuzz-invariant: regex group outside its match", .{});
        };
        pos = if (m.end > pos) m.end else pos + 1;
    }
    fb.batches = n;
    fb.types |= @as(u64, 1) << @intCast(re.groupCount() % 16);
    if (re.literal != null) fb.types |= 1 << 16;
    if (re.dfa_failures > 0) fb.types |= 1 << 17;
    if (any) fb.types |= 1 << 18;
    fb.types |= @as(u64, flags & 7) << 19;
}

// ---------------------------------------------------------------------------------------------
// integration-json: the Arrow integration JSON reader (src/integration/json.zig) on any bytes.
//
// What it accepts must pass validateFull, write back with the JSON writer (only nesting past the
// writer's depth limit may fail), read back equal, and, when zarrow's IPC stream writer and
// reader take it, come back from IPC equal too.

fn integrationJsonTarget(gpa: Allocator, bytes: []const u8, fb: *Feedback) !void {
    const integration = zarrow.integration;
    var doc = try integration.json.read(gpa, bytes);
    defer doc.deinit();
    fb.batches = @intCast(@min(doc.batches.len, std.math.maxInt(u32)));
    for (doc.batches) |*b| for (b.columns) |c| {
        if (!fb.noteArray(c)) std.debug.panic("fuzz-invariant: integration JSON column fails validateFull: {s}", .{@errorName(fb.invalid.?)});
    };

    var out: std.Io.Writer.Allocating = .init(gpa);
    defer out.deinit();
    integration.json.write(gpa, &out.writer, doc.schema().*, doc.batches) catch |e| switch (e) {
        error.NestingTooDeep => return,
        else => std.debug.panic("fuzz-invariant: integration JSON write of an accepted document failed: {s}", .{@errorName(e)}),
    };
    var again = integration.json.read(gpa, out.written()) catch |e|
        std.debug.panic("fuzz-invariant: integration JSON written by zarrow does not read back: {s}", .{@errorName(e)});
    defer again.deinit();
    var r: integration.golden.Result = .{};
    if (!integration.golden.same(&r, "JSON round trip", &doc, &again)) std.debug.panic("fuzz-invariant: {s}", .{r.message()});

    // The IPC writers may refuse an input the JSON reader accepts (an index type the writer
    // cannot number, say); what they write must read back equal.
    const ipc = integration.golden.writeIpc(gpa, &doc, .stream) catch return;
    defer gpa.free(ipc);
    var back = integration.golden.readStream(gpa, ipc) catch return;
    defer back.deinit();
    if (!integration.golden.same(&r, "IPC round trip", &doc, &back)) std.debug.panic("fuzz-invariant: {s}", .{r.message()});
}

// ---------------------------------------------------------------------------------------------
// temporal: strptime, strftime and the temporal <-> text casts (src/compute/temporal_text.zig)
// on a format and a text from the input.
//
// Input: a flag byte (bits 0-1 unit, bit 2 string_view rows, bits 3-5 timezone, bit 6 TZif), a
// u16 format length, the format, then the text (first 4 KiB): its lines are the strptime / cast
// rows and its 8-byte words the timestamps. Every call must succeed or fail with its documented
// error; every array result must pass validateFull with one row per input row, and a naive
// timestamp cast to text and back must give its values back. The zoned timestamps also go
// through the timezone functions (assume_timezone with every choice, local_timestamp, is_dst,
// rounding, fields, casts). With bit 6 the text is also read as a TZif file by the timezone
// reader (src/compute/tz.zig): it must reject it or give intervals that hold their instant.

const temporal_max_text = 4096;
const temporal_zones = [_]?[]const u8{ null, "UTC", "+05:30", "America/New_York", "Australia/Lord_Howe", "Africa/Casablanca", null, "Pacific/Apia" };

fn temporalTarget(gpa: Allocator, bytes: []const u8, fb: *Feedback) !void {
    if (bytes.len < 3) return error.EndOfStream;
    const compute = zarrow.compute;
    const flags = bytes[0];
    const flen = @min(std.mem.readInt(u16, bytes[1..3], .little), bytes.len - 3);
    const format = bytes[3..][0..flen];
    const rest = bytes[3 + flen ..];
    const text = rest[0..@min(rest.len, temporal_max_text)];
    const unit: zarrow.TimeUnit = @enumFromInt(flags & 3);
    const zone = temporal_zones[(flags >> 3) & 7];
    var ctx = compute.ExecContext.init(gpa, kernelRegistry());

    // Rows: the lines of the text.
    var rows: ArrayRef = blk: {
        var b = try zarrow.StringBuilder.init(gpa, 16, text.len);
        defer b.deinit();
        var lines = std.mem.splitScalar(u8, text, '\n');
        while (lines.next()) |line| try b.append(line);
        var s = try b.finish();
        if (flags & 4 == 0) break :blk s;
        defer s.release();
        const v = try ctx.invokeScalar("cast", &.{.{ .array = s }}, .{ .cast = .{ .to_type = .{ .string_view = {} } } });
        break :blk v.array;
    };
    defer rows.release();
    const n_rows = rows.data().length;
    fb.batches = @intCast(n_rows);
    const rd: compute.Datum = .{ .array = rows };
    const fmt_ok = format.len <= compute.kernels.temporal_text.max_format_len and std.mem.indexOfScalar(u8, format, 0) == null;
    expectCall(&ctx, "strptime", .scalar, &.{rd}, .{ .strptime = .{ .format = format, .unit = unit, .error_is_null = true } }, n_rows, if (fmt_ok) &.{} else &.{error.InvalidOptions});
    const targets = [_]DataType{
        .{ .timestamp = .{ .unit = unit, .timezone = null } }, .{ .timestamp = .{ .unit = unit, .timezone = "UTC" } },
        .{ .date32 = {} },                                     .{ .date64 = {} },
        .{ .time32 = .{ .unit = .millisecond } },              .{ .time64 = .{ .unit = .nanosecond } },
        .{ .duration = .{ .unit = unit } },
    };
    for (targets) |t| expectCall(&ctx, "cast", .scalar, &.{rd}, .{ .cast = .{ .to_type = t } }, n_rows, &.{error.InvalidCast});

    // Timestamps: the 8-byte words of the text.
    const n_words = text.len / 8;
    var words = try zarrow.Int64Builder.init(gpa, n_words);
    defer words.deinit();
    for (0..n_words) |i| try words.append(std.mem.readInt(i64, text[i * 8 ..][0..8], .little));
    var ints = try words.finish();
    defer ints.release();
    const ts_type: DataType = .{ .timestamp = .{ .unit = unit, .timezone = zone } };
    var ts = try ctx.invokeScalar("cast", &.{.{ .array = ints }}, .{ .cast = .{ .to_type = ts_type } });
    defer ts.release();
    const names_zone = std.mem.indexOf(u8, format, "%z") != null or std.mem.indexOf(u8, format, "%Z") != null;
    const allowed: []const anyerror = if (zone == null and names_zone) &.{ error.InvalidOptions, error.InvalidInput } else &.{error.InvalidOptions};
    expectCall(&ctx, "strftime", .scalar, &.{ts}, .{ .strftime = .{ .format = format } }, n_words, allowed);
    expectCall(&ctx, "strftime", .scalar, &.{ts}, .none, n_words, &.{});
    var text_out = ctx.invokeScalar("cast", &.{ts}, .{ .cast = .{ .to_type = .{ .string = {} } } }) catch |e| std.debug.panic("fuzz-invariant: timestamp cast to text failed: {s}", .{@errorName(e)});
    defer text_out.release();
    expectValid("cast timestamp to text", text_out.array, n_words);
    // Text back to the timestamp (rows outside the years 0000..9999 fail: no comparison then).
    if (zone == null) if (ctx.invokeScalar("cast", &.{text_out}, .{ .cast = .{ .to_type = ts_type } })) |back| {
        var b = back;
        defer b.release();
        const before = ts.array.data().buffers[1].typedSlice(i64) catch unreachable;
        const after = b.array.data().buffers[1].typedSlice(i64) catch unreachable;
        if (!std.mem.eql(i64, before[0..n_words], after[0..n_words])) std.debug.panic("fuzz-invariant: timestamp text round trip changed a value", .{});
    } else |e| if (e != error.InvalidCast) std.debug.panic("fuzz-invariant: text to timestamp failed: {s}", .{@errorName(e)});
    // Timezone functions on the zoned timestamps and on the naive ones read as local time.
    if (zone != null) {
        expectCall(&ctx, "local_timestamp", .scalar, &.{ts}, .none, n_words, &.{});
        expectCall(&ctx, "is_dst", .scalar, &.{ts}, .none, n_words, &.{});
        expectCall(&ctx, "hour", .scalar, &.{ts}, .none, n_words, &.{});
        expectCall(&ctx, "iso_calendar", .scalar, &.{ts}, .none, n_words, &.{});
        expectCall(&ctx, "days_between", .scalar, &.{ ts, ts }, .none, n_words, &.{});
        expectCall(&ctx, "cast", .scalar, &.{ts}, .{ .cast = .{ .to_type = .{ .date32 = {} }, .safe = false } }, n_words, &.{});
        // An ambiguous or nonexistent local result of a rounding is InvalidInput (Arrow).
        expectCall(&ctx, "floor_temporal", .scalar, &.{ts}, .{ .round_temporal = .{ .multiple = 2, .unit = .hour } }, n_words, &.{error.InvalidInput});
        expectCall(&ctx, "round_temporal", .scalar, &.{ts}, .{ .round_temporal = .{ .unit = .day } }, n_words, &.{error.InvalidInput});
        const naive: DataType = .{ .timestamp = .{ .unit = unit, .timezone = null } };
        var local = try ctx.invokeScalar("cast", &.{.{ .array = ints }}, .{ .cast = .{ .to_type = naive } });
        defer local.release();
        const z = zone.?;
        expectCall(&ctx, "assume_timezone", .scalar, &.{local}, .{ .assume_timezone = .{ .timezone = z, .ambiguous = .earliest, .nonexistent = .latest } }, n_words, &.{});
        expectCall(&ctx, "assume_timezone", .scalar, &.{local}, .{ .assume_timezone = .{ .timezone = z, .ambiguous = .latest, .nonexistent = .earliest } }, n_words, &.{});
        expectCall(&ctx, "assume_timezone", .scalar, &.{local}, .{ .assume_timezone = .{ .timezone = z } }, n_words, &.{error.InvalidInput});
    }
    if (flags & 64 != 0) tzif: {
        const tz = compute.kernels.tz;
        var rules = tz.parseTzif(gpa, rest[0..@min(rest.len, tz.max_tzif_len)]) catch |e| {
            if (e == error.OutOfMemory) return e;
            break :tzif;
        };
        defer rules.deinit(gpa);
        const zone_rules: tz.Zone = .{ .rules = &rules };
        var cursor = tz.Cursor.init(&zone_rules);
        for (0..n_words) |i| {
            const t = std.mem.readInt(i64, text[i * 8 ..][0..8], .little) >> 4;
            const sp = zone_rules.span(t);
            if (t < sp.begin or t >= sp.end) std.debug.panic("fuzz-invariant: TZif interval does not hold its instant", .{});
            if (@abs(sp.info.offset) > 93_599 or sp.info.abbr.len > tz.max_abbr_len) std.debug.panic("fuzz-invariant: TZif local time type out of range", .{});
            if (cursor.info(t).offset != sp.info.offset) std.debug.panic("fuzz-invariant: TZif cursor and lookup disagree", .{});
            switch (zone_rules.local(t)) {
                .unique => |off| if (zone_rules.info(t - off).offset != off) std.debug.panic("fuzz-invariant: TZif local time does not map back", .{}),
                .ambiguous, .gap => {},
            }
        }
        fb.types |= 1 << 63;
    }
    fb.types |= @as(u64, 1) << @intCast(flags % 63);
}

// ---------------------------------------------------------------------------------------------
// array-layout: hand-built ArrayData from bytes (structure from the input, contents from a PRNG).

const Input = struct {
    bytes: []const u8,
    pos: usize = 0,

    fn byte(in: *Input) u8 {
        if (in.pos >= in.bytes.len) return 0;
        defer in.pos += 1;
        return in.bytes[in.pos];
    }

    fn below(in: *Input, n: usize) usize {
        if (n == 0) return 0;
        return (@as(usize, in.byte()) | @as(usize, in.byte()) << 8) % n;
    }
};

const int_types = [_]zarrow.IntType{
    .{ .bit_width = 8, .signed = true },  .{ .bit_width = 8, .signed = false },
    .{ .bit_width = 16, .signed = true }, .{ .bit_width = 16, .signed = false },
    .{ .bit_width = 32, .signed = true }, .{ .bit_width = 32, .signed = false },
    .{ .bit_width = 64, .signed = true }, .{ .bit_width = 64, .signed = false },
};

fn intDataType(t: zarrow.IntType) DataType {
    return switch (t.bit_width) {
        8 => if (t.signed) .{ .int8 = {} } else .{ .uint8 = {} },
        16 => if (t.signed) .{ .int16 = {} } else .{ .uint16 = {} },
        32 => if (t.signed) .{ .int32 = {} } else .{ .uint32 = {} },
        else => if (t.signed) .{ .int64 = {} } else .{ .uint64 = {} },
    };
}

fn genType(in: *Input, a: Allocator, depth: u32) Allocator.Error!*const DataType {
    const dt = try a.create(DataType);
    const unit: zarrow.TimeUnit = @enumFromInt(in.byte() % 4);
    const leaf_kinds = 30;
    const pick = in.byte() % @as(u8, if (depth >= 3) leaf_kinds else leaf_kinds + 12);
    dt.* = switch (pick) {
        0 => .{ .null = {} },
        1 => .{ .bool = {} },
        2...9 => intDataType(int_types[pick - 2]),
        10 => .{ .half_float = {} },
        11 => .{ .float = {} },
        12 => .{ .double = {} },
        13 => .{ .string = {} },
        14 => .{ .binary = {} },
        15 => .{ .large_string = {} },
        16 => .{ .large_binary = {} },
        17 => .{ .string_view = {} },
        18 => .{ .binary_view = {} },
        19 => .{ .fixed_size_binary = .{ .byte_width = @as(i32, in.byte() % 20) - 2 } },
        20 => .{ .date32 = {} },
        21 => .{ .date64 = {} },
        22 => .{ .timestamp = .{ .unit = unit, .timezone = null } },
        23 => .{ .time32 = .{ .unit = unit } },
        24 => .{ .time64 = .{ .unit = unit } },
        25 => .{ .duration = .{ .unit = unit } },
        26 => .{ .interval_month_day_nano = .{ .unit = .month_day_nano } },
        27 => .{ .decimal128 = .{ .precision = in.byte() % 40, .scale = in.byte() % 40 } },
        28 => .{ .decimal256 = .{ .precision = in.byte() % 80, .scale = in.byte() % 80 } },
        29 => .{ .decimal32 = .{ .precision = in.byte() % 12, .scale = in.byte() % 12 } },
        30 => .{ .list = .{ .value_field = try genField(in, a, depth) } },
        31 => .{ .large_list = .{ .value_field = try genField(in, a, depth) } },
        32 => .{ .list_view = .{ .value_field = try genField(in, a, depth) } },
        33 => .{ .large_list_view = .{ .value_field = try genField(in, a, depth) } },
        34 => .{ .fixed_size_list = .{ .value_field = try genField(in, a, depth), .list_size = @as(i32, in.byte() % 6) - 1 } },
        35 => .{ .struct_ = .{ .fields = try genFields(in, a, depth, in.byte() % 4) } },
        36, 37 => blk: {
            const fields = try genFields(in, a, depth, 1 + in.byte() % 3);
            const ids = try a.alloc(i8, fields.len);
            for (ids, 0..) |*id, i| id.* = if (in.byte() % 4 == 0) @bitCast(in.byte()) else @intCast(i);
            const u: zarrow.UnionType = .{ .type_ids = ids, .fields = fields, .mode = if (pick == 36) .sparse else .dense };
            break :blk if (pick == 36) .{ .sparse_union = u } else .{ .dense_union = u };
        },
        38 => .{ .dictionary = .{ .index_type = int_types[in.byte() % 8], .value_type = try genType(in, a, depth + 1) } },
        39 => .{ .map = .{ .key_field = try genField(in, a, depth), .item_field = try genField(in, a, depth) } },
        40 => .{ .run_end_encoded = .{ .run_end_type = int_types[in.byte() % 8], .value_type = try genType(in, a, depth + 1) } },
        else => .{ .extension = .{ .name = "fuzz.ext", .storage_type = try genType(in, a, depth + 1) } },
    };
    return dt;
}

fn genField(in: *Input, a: Allocator, depth: u32) Allocator.Error!Field {
    return .{ .name = "f", .data_type = try genType(in, a, depth + 1), .nullable = in.byte() % 2 == 0 };
}

fn genFields(in: *Input, a: Allocator, depth: u32, n: usize) Allocator.Error![]const Field {
    const fields = try a.alloc(Field, n);
    for (fields) |*f| f.* = try genField(in, a, depth);
    return fields;
}

/// Child types the layout expects (what validateLayout checks children against).
fn childTypes(a: Allocator, dt: DataType) Allocator.Error![]const *const DataType {
    const one = struct {
        fn of(al: Allocator, t: *const DataType) Allocator.Error![]const *const DataType {
            const s = try al.alloc(*const DataType, 1);
            s[0] = t;
            return s;
        }
    }.of;
    return switch (dt) {
        .list, .large_list => |l| one(a, l.value_field.data_type),
        .list_view, .large_list_view => |l| one(a, l.value_field.data_type),
        .fixed_size_list => |l| one(a, l.value_field.data_type),
        .struct_ => |s| fieldTypes(a, s.fields),
        .sparse_union, .dense_union => |u| fieldTypes(a, u.fields),
        .map => |m| blk: {
            const fields = try a.alloc(Field, 2);
            fields[0] = m.key_field;
            fields[1] = m.item_field;
            const entries = try a.create(DataType);
            entries.* = .{ .struct_ = .{ .fields = fields } };
            break :blk one(a, entries);
        },
        .run_end_encoded => |r| blk: {
            const s = try a.alloc(*const DataType, 2);
            const ends = try a.create(DataType);
            ends.* = intDataType(r.run_end_type);
            s[0] = ends;
            s[1] = r.value_type;
            break :blk s;
        },
        .extension => |e| childTypes(a, e.storage_type.*),
        else => &.{},
    };
}

fn fieldTypes(a: Allocator, fields: []const Field) Allocator.Error![]const *const DataType {
    const s = try a.alloc(*const DataType, fields.len);
    for (s, fields) |*t, f| t.* = f.data_type;
    return s;
}

fn bufferCount(dt: DataType) usize {
    return switch (dt) {
        .null, .run_end_encoded => 0,
        .struct_, .fixed_size_list, .sparse_union => 1,
        .string, .binary, .large_string, .large_binary, .list_view, .large_list_view => 3,
        .string_view, .binary_view => 4,
        .extension => |e| bufferCount(e.storage_type.*),
        else => 2,
    };
}

fn genBuffer(in: *Input, rng: std.Random, a: Allocator, total: usize) Allocator.Error!zarrow.SharedBuffer {
    const mode = in.byte() % 8;
    const n = switch (mode) {
        0 => return zarrow.SharedBuffer.empty,
        1 => in.below(257),
        else => (total + 1) * 32,
    };
    const buf = try a.alignedAlloc(u8, .fromByteUnits(64), n);
    switch (mode) {
        1 => rng.bytes(buf),
        2 => @memset(buf, 0),
        3 => @memset(buf, 0xff),
        4, 5 => { // ascending offsets / run ends (i32 or i64) with small steps
            const step = in.byte() % 4;
            var v: i64 = 0;
            if (mode == 4) {
                for (std.mem.bytesAsSlice(i32, buf)) |*x| {
                    x.* = @intCast(v);
                    v += rng.uintAtMost(u8, step);
                }
            } else for (std.mem.bytesAsSlice(i64, buf)) |*x| {
                x.* = v;
                v += rng.uintAtMost(u8, step);
            }
        },
        6 => for (buf) |*x| { // small ids (union type ids, dictionary indices)
            x.* = rng.uintLessThan(u8, 4);
        },
        else => rng.bytes(buf),
    }
    return zarrow.SharedBuffer.fromSlice(buf);
}

fn genArray(in: *Input, rng: std.Random, a: Allocator, gpa: Allocator, dt: *const DataType, depth: u32) Allocator.Error!ArrayRef {
    const length = in.below(65);
    const offset = in.below(17);
    const total = length + offset;
    const null_count: ?usize = switch (in.byte() % 4) {
        0 => null,
        1 => 0,
        2 => length,
        else => in.below(70),
    };
    var n_buffers = bufferCount(dt.*);
    if (in.byte() % 8 == 0) n_buffers = in.below(5);
    const buffers = try a.alloc(zarrow.SharedBuffer, n_buffers);
    for (buffers) |*b| b.* = try genBuffer(in, rng, a, total);

    var child_types = try childTypes(a, dt.*);
    if (in.byte() % 16 == 0) child_types = child_types[0 .. child_types.len - @min(child_types.len, 1)];
    const children = try a.alloc(ArrayRef, if (depth < 4) child_types.len else 0);
    var made: usize = 0;
    defer for (children[0..made]) |*c| c.release();
    for (children) |*c| {
        c.* = try genArray(in, rng, a, gpa, child_types[made], depth + 1);
        made += 1;
    }
    var dictionary: ?ArrayRef = null;
    defer if (dictionary) |*d| d.release();
    if (dt.* == .dictionary and depth < 4 and in.byte() % 8 != 0)
        dictionary = try genArray(in, rng, a, gpa, dt.dictionary.value_type, depth + 1);

    return ArrayRef.fromBorrowed(gpa, .{
        .data_type = dt.*,
        .length = length,
        .offset = offset,
        .null_count = null_count,
        .buffers = buffers,
        .children = children,
        .dictionary = dictionary,
    });
}

fn arrayLayout(gpa: Allocator, bytes: []const u8, fb: *Feedback) !void {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    var prng = std.Random.DefaultPrng.init(std.hash.Wyhash.hash(2, bytes));
    var in: Input = .{ .bytes = bytes };
    const dt = try genType(&in, a, 0);
    var arr = try genArray(&in, prng.random(), a, gpa, dt, 0);
    defer arr.release();
    try arr.data().validateLayout();
    const valid = fb.noteArray(arr);
    // The type tree comes straight from the input: only the root type is a useful signal here.
    fb.types = @as(u64, 1) << @intCast(@intFromEnum(dt.id()) % 64);
    if (!valid) return;
    try exerciseCompute(gpa, arr, prng.random());
    // The array as the one column of a record batch.
    const fields = [_]Field{.{ .name = "c", .data_type = dt }};
    var batch = try zarrow.RecordBatch.initBorrowed(gpa, .{ .fields = &fields }, &.{arr});
    defer batch.deinit();
    try exerciseTabular(gpa, &batch, prng.random());
}

// ---------------------------------------------------------------------------------------------
// c-abi: ArrowSchema/ArrowArray trees built from bytes, imported through the C Data Interface.
//
// The C ABI trusts producer pointers and buffer sizes by contract, so every buffer points into
// one 64 KiB arena whose integers are all in 0..255 (or -1), and lengths stay below 256: any
// size the importer derives fits. Everything else (formats, counts, lengths, flags, metadata,
// tree shape) is fuzzed.

const arena_len = 72 << 10;

const formats = [_][]const u8{
    "n",    "b",       "c",          "C",   "s",   "S",   "i",    "I",    "l",   "L",   "e",   "f",   "g",
    "u",    "U",       "z",          "Z",   "vu",  "vz",  "tdD",  "tdm",  "tts", "ttm", "ttu", "ttn", "tDs",
    "tDm",  "tDu",     "tDn",        "tiM", "tiD", "tin", "+l",   "+L",   "+vl", "+vL", "+r",  "+m",  "+s",
    "tss:", "tsu:UTC", "tsn:+01:00", "w:",  "d:",  "+w:", "+us:", "+ud:",
};

fn genFormat(in: *Input, a: Allocator, n_children: usize) Allocator.Error![:0]const u8 {
    const pick = in.byte() % (formats.len + 1);
    if (pick == formats.len) { // raw garbage
        const s = try a.allocSentinel(u8, in.below(9), 0);
        for (s) |*c| c.* = in.byte() | 1;
        return s;
    }
    const f = formats[pick];
    if (std.mem.eql(u8, f, "w:")) return std.fmt.allocPrintSentinel(a, "w:{d}", .{@as(i32, in.byte() % 66) - 1}, 0);
    if (std.mem.eql(u8, f, "+w:")) return std.fmt.allocPrintSentinel(a, "+w:{d}", .{@as(i32, @as(i8, @bitCast(in.byte())))}, 0);
    if (std.mem.eql(u8, f, "d:")) {
        const widths = [_]u32{ 32, 64, 128, 256, 0, 7 };
        return std.fmt.allocPrintSentinel(a, "d:{d},{d},{d}", .{ in.byte() % 80, @as(i8, @bitCast(in.byte())), widths[in.byte() % widths.len] }, 0);
    }
    if (std.mem.startsWith(u8, f, "+u")) {
        var list: std.ArrayList(u8) = .empty;
        try list.appendSlice(a, f);
        for (0..n_children) |i| {
            if (i > 0) try list.append(a, ',');
            const id: i32 = if (in.byte() % 4 == 0) @as(i8, @bitCast(in.byte())) else @intCast(i);
            try list.print(a, "{d}", .{id});
        }
        return list.toOwnedSliceSentinel(a, 0);
    }
    return a.dupeSentinel(u8, f, 0);
}

fn expectedChildren(fmt: []const u8, in: *Input) usize {
    if (std.mem.eql(u8, fmt, "+r")) return 2;
    if (std.mem.eql(u8, fmt, "+s") or std.mem.startsWith(u8, fmt, "+u")) return in.below(4);
    if (fmt.len > 1 and fmt[0] == '+') return 1;
    return 0;
}

fn expectedBuffers(fmt: []const u8) usize {
    if (std.mem.eql(u8, fmt, "n") or std.mem.eql(u8, fmt, "+r")) return 0;
    if (std.mem.eql(u8, fmt, "+s") or std.mem.startsWith(u8, fmt, "+w") or std.mem.startsWith(u8, fmt, "+us")) return 1;
    if (std.mem.eql(u8, fmt, "vu") or std.mem.eql(u8, fmt, "vz")) return 4;
    if (std.mem.eql(u8, fmt, "u") or std.mem.eql(u8, fmt, "U") or std.mem.eql(u8, fmt, "z") or
        std.mem.eql(u8, fmt, "Z") or std.mem.startsWith(u8, fmt, "+v")) return 3;
    return 2;
}

fn releaseSchema(s: ?*zarrow.ArrowSchema) callconv(.c) void {
    s.?.release = null;
}

fn releaseArray(arr: ?*zarrow.ArrowArray) callconv(.c) void {
    arr.?.release = null;
}

fn genMetadata(in: *Input, a: Allocator) Allocator.Error![*c]const u8 {
    if (in.byte() % 4 != 0) return null;
    var blob: std.ArrayList(u8) = .empty;
    const keys = [_][]const u8{ "ARROW:extension:name", "ARROW:extension:metadata", "k", "" };
    const n = in.below(4);
    try blob.appendSlice(a, std.mem.asBytes(&std.mem.nativeToLittle(i32, @intCast(n))));
    for (0..n) |_| {
        const key = keys[in.byte() % keys.len];
        const value = in.bytes[@min(in.pos, in.bytes.len)..@min(in.pos + in.below(8), in.bytes.len)];
        try blob.appendSlice(a, std.mem.asBytes(&std.mem.nativeToLittle(i32, @intCast(key.len))));
        try blob.appendSlice(a, key);
        try blob.appendSlice(a, std.mem.asBytes(&std.mem.nativeToLittle(i32, @intCast(value.len))));
        try blob.appendSlice(a, value);
    }
    return blob.items.ptr;
}

fn genCSchema(in: *Input, a: Allocator, depth: u32, root: bool) Allocator.Error!*zarrow.ArrowSchema {
    const s = try a.create(zarrow.ArrowSchema);
    var n_children: usize = 0;
    const fmt: [:0]const u8 = if (root and in.byte() % 16 != 0) "+s" else blk: {
        const pos = in.pos;
        const probe = try genFormat(in, a, 0);
        n_children = if (depth < 4) expectedChildren(probe, in) else 0;
        if (std.mem.startsWith(u8, probe, "+u")) { // regenerate with matching id count
            in.pos = pos;
            break :blk try genFormat(in, a, n_children);
        }
        break :blk probe;
    };
    if (root) n_children = in.below(4);
    if (in.byte() % 16 == 0) n_children = in.below(4);
    const children = try a.alloc(?*zarrow.ArrowSchema, n_children);
    for (children) |*c| c.* = try genCSchema(in, a, depth + 1, false);
    const is_int = fmt.len == 1 and std.mem.indexOfScalar(u8, "cCsSiIlL", fmt[0]) != null;
    s.* = .{
        .format = fmt.ptr,
        .name = if (in.byte() % 8 == 0) null else "f",
        .metadata = try genMetadata(in, a),
        .flags = switch (in.byte() % 4) {
            0 => 0,
            1 => 2,
            2 => in.byte() % 8,
            else => @as(i8, @bitCast(in.byte())),
        },
        .n_children = if (in.byte() % 32 == 0) -1 else @intCast(n_children),
        .children = children.ptr,
        .dictionary = if (is_int and depth < 4 and in.byte() % 4 == 0) try genCSchema(in, a, depth + 1, false) else null,
        .release = releaseSchema,
        .private_data = null,
    };
    return s;
}

fn genCArray(in: *Input, a: Allocator, mem: []align(64) const u8, schema: *const zarrow.ArrowSchema) Allocator.Error!*zarrow.ArrowArray {
    const arr = try a.create(zarrow.ArrowArray);
    const fmt = std.mem.span(schema.format);
    var n_buffers = expectedBuffers(fmt);
    if (in.byte() % 8 == 0) n_buffers = in.below(6);
    const buffers = try a.alloc(?*const anyopaque, n_buffers);
    for (buffers) |*b| b.* = if (in.byte() % 32 == 0) null else @ptrCast(mem[64 * in.below(16) ..].ptr);
    const schema_children: usize = @intCast(@max(schema.n_children, 0));
    var n_children = schema_children;
    if (in.byte() % 16 == 0) n_children = in.below(4);
    const children = try a.alloc(?*zarrow.ArrowArray, n_children);
    for (children, 0..) |*c, i| c.* = if (i < schema_children)
        try genCArray(in, a, mem, schema.children[i].?)
    else
        try genCArray(in, a, mem, schema);
    const length: i64 = if (in.byte() % 32 == 0) -1 else @intCast(in.below(201));
    arr.* = .{
        .length = length,
        .null_count = switch (in.byte() % 4) {
            0 => -1,
            1 => 0,
            2 => length,
            else => @intCast(in.below(210)),
        },
        .offset = if (in.byte() % 32 == 0) -1 else @intCast(in.below(56)),
        .n_buffers = @intCast(n_buffers),
        .n_children = @intCast(n_children),
        .buffers = buffers.ptr,
        .children = children.ptr,
        .dictionary = if (schema.dictionary) |d| try genCArray(in, a, mem, d) else null,
        .release = releaseArray,
        .private_data = null,
    };
    return arr;
}

fn cAbi(gpa: Allocator, bytes: []const u8, fb: *Feedback) !void {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    var in: Input = .{ .bytes = bytes };

    const mem = try a.alignedAlloc(u8, .fromByteUnits(64), arena_len);
    switch (if (bytes.len == 0) 1 else in.byte() % 3) {
        0 => for (mem, 0..) |*b, i| { // ints read at any width are 0..255
            b.* = if (i % 8 == 0) bytes[i / 8 % bytes.len] else 0;
        },
        1 => @memset(mem, 0),
        else => @memset(mem, 0xff), // all -1: must be rejected as negative
    }

    const c_schema = try genCSchema(&in, a, 0, true);
    const c_array = try genCArray(&in, a, mem, c_schema);
    var schema = try zarrow.importSchemaFromC(gpa, c_schema);
    defer schema.deinit();
    const top: DataType = .{ .struct_ = .{ .fields = schema.schema.fields } };
    var arr = try zarrow.importArrayFromC(gpa, &top, c_array);
    defer arr.release();
    fb.batches = 1;
    if (!fb.noteArray(arr)) return;
    var prng = std.Random.DefaultPrng.init(std.hash.Wyhash.hash(3, bytes));
    try exerciseCompute(gpa, arr, prng.random());

    // Round trip what we accepted: export must produce something import accepts again.
    var exported = try zarrow.exportArrayToC(gpa, arr);
    var again = zarrow.importArrayFromC(gpa, &top, &exported) catch |e|
        std.debug.panic("fuzz-invariant: re-import of an exported valid array failed: {s}", .{@errorName(e)});
    defer again.release();
    expectValid("c-abi round trip", again, arr.data().length);
    var exported_schema = try zarrow.exportSchemaToC(gpa, schema.schema);
    var schema_again = zarrow.importSchemaFromC(gpa, &exported_schema) catch |e|
        std.debug.panic("fuzz-invariant: re-import of an exported schema failed: {s}", .{@errorName(e)});
    schema_again.deinit();
}

// ---------------------------------------------------------------------------------------------
// Mutation engine (libFuzzer-style mutators; corpus grows on new Feedback signatures).

const tokens = [_][]const u8{
    "ARROW1",           "ARROW1\x00\x00",   "\xff\xff\xff\xff", "\xff\xff\xff\xff\x00\x00\x00\x00",
    "\x04\x00",         "\x03\x00",         "\x00\x00\x00\x00", "\x01\x00\x00\x00",
    "\x08\x00\x0c\x00", "\x0c\x00\x00\x00", "\x10\x00\x00\x00", "\x00\x00\x00\x00\x00\x00\x00\x00",
};
const interesting = [_]u64{ 0, 1, 2, 4, 8, 16, 32, 64, 127, 128, 255, 256, 0x7fff, 0x8000, 0xffff, 0x10000, 0x7fffffff, 0x80000000, 0xffffffff, 0x7fffffffffffffff, 0x8000000000000000, std.math.maxInt(u64) };

fn mutate(rng: std.Random, gpa: Allocator, buf: *std.ArrayList(u8), corpus: []const []const u8, max_len: usize) !void {
    const rounds = 1 + rng.uintLessThan(usize, 5);
    for (0..rounds) |_| {
        const len = buf.items.len;
        switch (rng.uintLessThan(u8, 11)) {
            0 => if (len > 0) { // erase a range
                const at = rng.uintLessThan(usize, len);
                const n = 1 + rng.uintLessThan(usize, @min(len - at, 64));
                buf.replaceRangeAssumeCapacity(at, n, &.{});
            },
            1 => if (len < max_len) { // insert random bytes
                const at = rng.uintAtMost(usize, len);
                var tmp: [16]u8 = undefined;
                const n = 1 + rng.uintLessThan(usize, @min(tmp.len, max_len - len));
                rng.bytes(tmp[0..n]);
                try buf.insertSlice(gpa, at, tmp[0..n]);
            },
            2 => if (len > 0) { // flip a bit
                buf.items[rng.uintLessThan(usize, len)] ^= @as(u8, 1) << rng.int(u3);
            },
            3 => if (len > 0) { // random byte
                buf.items[rng.uintLessThan(usize, len)] = rng.int(u8);
            },
            4, 5 => if (len > 0) { // arithmetic / interesting value at width 1/2/4/8
                const width = @as(usize, 1) << rng.int(u2);
                if (len < width) continue;
                const at = rng.uintAtMost(usize, len - width);
                var v: u64 = 0;
                @memcpy(std.mem.asBytes(&v)[0..width], buf.items[at..][0..width]);
                v = if (rng.boolean())
                    interesting[rng.uintLessThan(usize, interesting.len)]
                else
                    v +% @as(u64, @bitCast(@as(i64, rng.intRangeAtMost(i8, -35, 35))));
                if (rng.boolean()) v = @byteSwap(v) >> @intCast(64 - 8 * width);
                @memcpy(buf.items[at..][0..width], std.mem.asBytes(&v)[0..width]);
            },
            6 => if (len > 1) { // copy a chunk over another place
                const from = rng.uintLessThan(usize, len);
                const to = rng.uintLessThan(usize, len);
                const n = 1 + rng.uintLessThan(usize, @min(len - @max(from, to), 128));
                std.mem.copyForwards(u8, buf.items[to..][0..n], buf.items[from..][0..n]);
            },
            7, 8 => { // dictionary token, overwrite or insert
                const tok = tokens[rng.uintLessThan(usize, tokens.len)];
                const at = rng.uintAtMost(usize, len);
                if (rng.boolean() and at + tok.len <= len) {
                    @memcpy(buf.items[at..][0..tok.len], tok);
                } else if (len + tok.len <= max_len) try buf.insertSlice(gpa, at, tok);
            },
            9 => if (corpus.len > 0) { // splice with another corpus entry
                const other = corpus[rng.uintLessThan(usize, corpus.len)];
                const cut = rng.uintAtMost(usize, len);
                const from = rng.uintAtMost(usize, other.len);
                buf.shrinkRetainingCapacity(cut);
                try buf.appendSlice(gpa, other[from..@min(other.len, from + (max_len - cut))]);
            },
            else => if (len > 0) { // repeat a byte
                const at = rng.uintLessThan(usize, len);
                const n = rng.uintLessThan(usize, @min(64, max_len - @min(max_len, len)) + 1);
                try buf.appendNTimes(gpa, buf.items[at], n);
            },
        }
    }
}

fn hexName(bytes: []const u8) [16]u8 {
    var digest: [20]u8 = undefined;
    std.crypto.hash.Sha1.hash(bytes, &digest, .{});
    return std.fmt.bytesToHex(digest[0..8].*, .lower);
}

// ---------------------------------------------------------------------------------------------
// CLI.

/// Path of the input being replayed, printed by the panic handler so a crash names its file.
var current_path: []const u8 = "";
/// Executions so far; the watchdog calls a hang when it stops moving for `hang_seconds`.
var progress = std.atomic.Value(u64).init(0);
var hang_seconds: u64 = 10;

fn watchdog(io: std.Io) void {
    var last = progress.load(.monotonic);
    var still: u64 = 0;
    while (true) {
        io.sleep(.fromNanoseconds(1_000_000_000), .awake) catch return;
        const now = progress.load(.monotonic);
        still = if (now == last) still + 1 else 0;
        last = now;
        if (still >= hang_seconds) {
            std.debug.print("fuzz-hang: one input ran for over {d}s: {s}\n", .{ hang_seconds, if (current_path.len > 0) current_path else "<out>/current.bin" });
            std.process.exit(77);
        }
    }
}
pub const panic = std.debug.FullPanic(struct {
    fn f(msg: []const u8, ra: ?usize) noreturn {
        if (current_path.len > 0) std.debug.print("fuzz: crashing input: {s}\n", .{current_path});
        std.debug.defaultPanic(msg, ra);
    }
}.f);

fn usage() noreturn {
    std.debug.print(
        \\usage: zarrow-fuzz <target> replay <file|dir>...
        \\       zarrow-fuzz <target> run <iters|secs s> --corpus <dir>... --out <dir> [--seed N] [--max-len N] [--alloc-cap N] [--timeout S]
        \\       zarrow-fuzz gen-seeds <corpus-root>
        \\targets: ipc-stream ipc-stream-zero-copy ipc-file array-layout compute c-abi
        \\
    , .{});
    std.process.exit(2);
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;
    const arena = init.arena.allocator();
    const argv = try init.minimal.args.toSlice(arena);
    var args: std.ArrayList([]const u8) = .empty;
    var i: usize = 0;
    while (i < argv.len) : (i += 1) {
        if (std.mem.eql(u8, argv[i], "--timeout") and i + 1 < argv.len) {
            i += 1;
            hang_seconds = std.fmt.parseInt(u64, argv[i], 10) catch usage();
        } else try args.append(arena, argv[i]);
    }
    if (args.items.len < 3) usage();
    if (std.mem.eql(u8, args.items[1], "gen-seeds")) return genSeeds(gpa, io, args.items[2]);
    const target = std.meta.stringToEnum(Target, args.items[1]) orelse usage();
    (try std.Thread.spawn(.{}, watchdog, .{io})).detach();
    if (std.mem.eql(u8, args.items[2], "replay")) return replay(gpa, io, target, args.items[3..]);
    if (std.mem.eql(u8, args.items[2], "run")) return run(gpa, io, target, args.items[3..]);
    usage();
}

/// Files directly under `path` (or `path` itself), sorted for a stable order.
fn listInputs(a: Allocator, io: std.Io, path: []const u8, out: *std.ArrayList([]const u8)) !void {
    var dir = std.Io.Dir.cwd().openDir(io, path, .{ .iterate = true }) catch |e| switch (e) {
        error.NotDir => return out.append(a, path),
        else => return e,
    };
    defer dir.close(io);
    const start = out.items.len;
    var it = dir.iterate();
    while (try it.next(io)) |entry| {
        if (entry.kind != .file) continue;
        if (std.mem.endsWith(u8, entry.name, ".md") or std.mem.eql(u8, entry.name, "NOTICE")) continue;
        try out.append(a, try std.fs.path.join(a, &.{ path, entry.name }));
    }
    std.mem.sort([]const u8, out.items[start..], {}, struct {
        fn lt(_: void, x: []const u8, y: []const u8) bool {
            return std.mem.order(u8, x, y) == .lt;
        }
    }.lt);
}

fn replay(gpa: Allocator, io: std.Io, target: Target, paths: []const []const u8) !void {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    var files: std.ArrayList([]const u8) = .empty;
    for (paths) |p| try listInputs(a, io, p, &files);
    var ok: usize = 0;
    var errors: usize = 0;
    explain = files.items.len == 1;
    for (files.items) |path| {
        const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(max_input_len)) catch |e| {
            std.debug.print("skip {s}: {s}\n", .{ path, @errorName(e) });
            continue;
        };
        defer gpa.free(bytes);
        current_path = path;
        const fb = runOne(target, gpa, bytes);
        _ = progress.fetchAdd(1, .monotonic);
        current_path = "";
        if (fb.err != null) errors += 1 else ok += 1;
        if (files.items.len == 1) std.debug.print("{s}: {s}, {d} batches, validateFull: {s}\n", .{
            path,
            if (fb.err) |e| @errorName(e) else "ok",
            fb.batches,
            if (fb.invalid) |e| @errorName(e) else "ok",
        });
    }
    std.debug.print("{s}: replayed {d} inputs ({d} decoded, {d} rejected), no crash\n", .{ @tagName(target), files.items.len, ok, errors });
}

fn run(gpa: Allocator, io: std.Io, target: Target, args: []const []const u8) !void {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    if (args.len < 1) usage();
    const budget = args[0];
    const by_time = std.mem.endsWith(u8, budget, "s");
    const limit = std.fmt.parseInt(u64, if (by_time) budget[0 .. budget.len - 1] else budget, 10) catch usage();
    var out_dir: ?[]const u8 = null;
    var seed: u64 = @truncate(@as(u96, @bitCast(std.Io.Timestamp.now(io, .real).toNanoseconds())));
    var max_len: usize = 64 << 10;
    var corpus_paths: std.ArrayList([]const u8) = .empty;
    var i: usize = 1;
    while (i + 1 < args.len) : (i += 2) {
        const v = args[i + 1];
        if (std.mem.eql(u8, args[i], "--corpus")) {
            listInputs(a, io, v, &corpus_paths) catch |e| std.debug.print("corpus {s}: {s}\n", .{ v, @errorName(e) });
        } else if (std.mem.eql(u8, args[i], "--out")) {
            out_dir = v;
        } else if (std.mem.eql(u8, args[i], "--seed")) {
            seed = std.fmt.parseInt(u64, v, 10) catch usage();
        } else if (std.mem.eql(u8, args[i], "--max-len")) {
            max_len = @min(std.fmt.parseInt(usize, v, 10) catch usage(), max_input_len);
        } else if (std.mem.eql(u8, args[i], "--alloc-cap")) {
            alloc_cap = std.fmt.parseInt(usize, v, 10) catch usage();
        } else usage();
    }
    const out = out_dir orelse usage();
    const cwd = std.Io.Dir.cwd();
    const grow_dir = try std.fs.path.join(a, &.{ out, "corpus" });
    try cwd.createDirPath(io, grow_dir);
    try listInputs(a, io, grow_dir, &corpus_paths);
    const current = try cwd.createFile(io, try std.fs.path.join(a, &.{ out, "current.bin" }), .{ .truncate = true });
    defer current.close(io);
    var out_handle = try cwd.openDir(io, out, .{});
    defer out_handle.close(io);

    // Load the corpus, skipping inputs that crashed before (tools/fuzz.py marks <out>/skip/<hash>).
    var corpus: std.ArrayList([]const u8) = .empty;
    var seen: std.AutoHashMapUnmanaged(u64, void) = .empty;
    for (corpus_paths.items) |path| {
        const bytes = cwd.readFileAlloc(io, path, a, .limited(max_len)) catch continue;
        const hex = hexName(bytes);
        const crash_name = try std.fmt.allocPrint(a, "skip/{s}", .{&hex});
        if (out_handle.statFile(io, crash_name, .{})) |_| {
            std.debug.print("skipping known crasher {s}\n", .{path});
            continue;
        } else |_| {}
        try current.writePositionalAll(io, bytes, 0);
        try current.setLength(io, bytes.len);
        const fb = runOne(target, gpa, bytes);
        _ = progress.fetchAdd(1, .monotonic);
        try seen.put(a, fb.sig(), {});
        try corpus.append(a, bytes);
    }
    if (corpus.items.len == 0) try corpus.append(a, "");
    std.debug.print("{s}: seed {d}, {d} corpus inputs, {d} signatures\n", .{ @tagName(target), seed, corpus.items.len, seen.count() });

    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(gpa);
    const start = std.Io.Timestamp.now(io, .awake);
    var iters: u64 = 0;
    while (true) : (iters += 1) {
        if (iters % 256 == 0) {
            const elapsed: u64 = @intCast(start.durationTo(std.Io.Timestamp.now(io, .awake)).toSeconds());
            if ((by_time and elapsed >= limit) or (!by_time and iters >= limit)) break;
            if (iters % (1 << 14) == 0 and iters > 0)
                std.debug.print("#{d} {d}s corpus {d} signatures {d}\n", .{ iters, elapsed, corpus.items.len, seen.count() });
        } else if (!by_time and iters >= limit) break;

        buf.clearRetainingCapacity();
        try buf.appendSlice(gpa, corpus.items[rng.uintLessThan(usize, corpus.items.len)]);
        try mutate(rng, gpa, &buf, corpus.items, max_len);
        try current.writePositionalAll(io, buf.items, 0);
        try current.setLength(io, buf.items.len);
        const fb = runOne(target, gpa, buf.items);
        _ = progress.fetchAdd(1, .monotonic);
        const gop = try seen.getOrPut(a, fb.sig());
        // ponytail: fixed corpus cap; signature-based growth has no coverage to prune with.
        if (gop.found_existing or corpus.items.len >= max_corpus) continue;
        const copy = try a.dupe(u8, buf.items);
        try corpus.append(a, copy);
        const hex = hexName(copy);
        const name = try std.fmt.allocPrint(a, "{s}.bin", .{&hex});
        var dir = try cwd.openDir(io, grow_dir, .{});
        defer dir.close(io);
        try dir.writeFile(io, .{ .sub_path = name, .data = copy });
    }
    std.debug.print("{s}: done, {d} execs, corpus {d}, {d} signatures\n", .{ @tagName(target), iters, corpus.items.len, seen.count() });
}

// ---------------------------------------------------------------------------------------------
// Seed generation: re-encode every decodable IPC stream in <root>/ipc-stream with zarrow's own
// writers (plain, lz4, zstd; stream and file; plus a sliced copy of each batch) so mutations
// start from rich, zarrow-produced inputs for every type the readers support.

fn genSeeds(gpa: Allocator, io: std.Io, root: []const u8) !void {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const cwd = std.Io.Dir.cwd();
    const stream_dir = try std.fs.path.join(a, &.{ root, "ipc-stream" });
    const file_dir = try std.fs.path.join(a, &.{ root, "ipc-file" });
    try cwd.createDirPath(io, file_dir);
    var inputs: std.ArrayList([]const u8) = .empty;
    try listInputs(a, io, stream_dir, &inputs);
    var written: usize = 0;
    for (inputs.items) |path| {
        // Only valid sources: skip earlier output and crash reproducers.
        const base = std.fs.path.basename(path);
        if (for ([_][]const u8{ "gen-", "regress-", "at-stream-", "at-tensor-" }) |p| {
            if (std.mem.startsWith(u8, base, p)) break true;
        } else false) continue;
        const bytes = cwd.readFileAlloc(io, path, a, .limited(max_input_len)) catch continue;
        if (runOne(.@"ipc-stream", gpa, bytes).err != null) continue;
        for ([_]?zarrow.IpcBodyCompressionCodec{ null, .lz4_frame, .zstd }) |codec| {
            for ([_]bool{ false, true }) |as_file| {
                const encoded = (if (as_file) transcode(gpa, bytes, codec, true) else transcode(gpa, bytes, codec, false)) catch continue;
                defer gpa.free(encoded);
                const check: Target = if (as_file) .@"ipc-file" else .@"ipc-stream";
                if (runOne(check, gpa, encoded).err != null or encoded.len > 16 << 10) continue;
                const hex = hexName(encoded);
                const name = try std.fmt.allocPrint(a, "{s}/gen-{s}-{s}.bin", .{
                    if (as_file) file_dir else stream_dir,
                    if (codec) |c| @tagName(c) else "plain",
                    hex[0..12],
                });
                try cwd.writeFile(io, .{ .sub_path = name, .data = encoded });
                written += 1;
            }
        }
    }
    std.debug.print("gen-seeds: wrote {d} seeds from {d} inputs\n", .{ written, inputs.items.len });
}

fn transcode(gpa: Allocator, bytes: []const u8, codec: ?zarrow.IpcBodyCompressionCodec, comptime as_file: bool) ![]u8 {
    var fixed: std.Io.Reader = .fixed(bytes);
    var reader = zarrow.IpcStreamReader.init(gpa, &fixed);
    defer reader.deinit();
    const schema = try reader.readSchema();
    var out: std.Io.Writer.Allocating = .init(gpa);
    errdefer out.deinit();
    const options: zarrow.IpcStreamWriterOptions = .{ .body_compression = codec };
    var writer = if (as_file)
        try zarrow.IpcFileWriter.initWithOptions(gpa, &out.writer, options)
    else
        zarrow.IpcStreamWriter.initWithOptions(gpa, &out.writer, options);
    defer writer.deinit();
    try writer.writeSchema(schema);
    while (try reader.nextRecordBatch()) |batch_value| {
        var batch = batch_value;
        defer batch.deinit();
        try writer.writeRecordBatch(batch);
        if (batch.numRows() > 2 and !as_file) { // sliced batches exercise non-zero offsets
            var sliced = try batch.slice(1, batch.numRows() - 2);
            defer sliced.deinit();
            try writer.writeRecordBatch(sliced);
        }
    }
    try writer.writeEnd();
    return out.toOwnedSlice();
}

// ---------------------------------------------------------------------------------------------
// std.testing.fuzz entry points: `zig build fuzz-test --fuzz=100K` (coverage-guided, Linux).
// Under plain `zig build test` each runs once on its embedded seed and on empty input.

fn smithSeed(comptime path: []const u8) []const u8 {
    const data = @embedFile(path);
    return comptime &(std.mem.toBytes(std.mem.nativeToLittle(u32, data.len)) ++ data.*);
}

fn fuzzOne(target: Target, smith: *std.testing.Smith) anyerror!void {
    const S = struct {
        var buf: [64 << 10]u8 = undefined;
    };
    const n = smith.slice(&S.buf);
    _ = runOne(target, std.testing.allocator, S.buf[0..n]);
}

test "fuzz ipc-stream" {
    try std.testing.fuzz(Target.@"ipc-stream", fuzzOne, .{ .corpus = &.{smithSeed("corpus/ipc-stream/zarrow-canonical.bin")} });
}

test "fuzz ipc-stream-zero-copy" {
    try std.testing.fuzz(Target.@"ipc-stream-zero-copy", fuzzOne, .{ .corpus = &.{smithSeed("corpus/ipc-stream/zarrow-canonical.bin")} });
}

test "fuzz ipc-file" {
    try std.testing.fuzz(Target.@"ipc-file", fuzzOne, .{ .corpus = &.{smithSeed("corpus/ipc-file/zarrow-canonical.bin")} });
}

test "fuzz compute" {
    try std.testing.fuzz(Target.compute, fuzzOne, .{ .corpus = &.{smithSeed("corpus/ipc-stream/zarrow-canonical.bin")} });
}

test "fuzz array-layout" {
    try std.testing.fuzz(Target.@"array-layout", fuzzOne, .{});
}

test "fuzz c-abi" {
    try std.testing.fuzz(Target.@"c-abi", fuzzOne, .{});
}

test "fuzz regex" {
    try std.testing.fuzz(Target.regex, fuzzOne, .{ .corpus = &.{smithSeed("corpus/regex/named-groups.bin")} });
}

test "fuzz integration-json" {
    try std.testing.fuzz(Target.@"integration-json", fuzzOne, .{ .corpus = &.{smithSeed("corpus/integration-json/zarrow-nested.json")} });
}

test "fuzz temporal" {
    try std.testing.fuzz(Target.temporal, fuzzOne, .{ .corpus = &.{smithSeed("corpus/temporal/iso.bin")} });
}

/// The decoded-array kernels of rounds 21, 23, 24 and 25.
fn exerciseRounds21To23(ctx: *zarrow.compute.ExecContext, a: ArrayRef) void {
    exerciseRound21(ctx, a);
    exerciseRound23(ctx, a);
    exerciseRound24(ctx, a);
    exerciseRound25(ctx, a);
}

/// Round 25 kernels on any decoded array: binary_join_element_wise of a binary / large_binary
/// column with itself (the column as separator, every null handling), and binary_repeat with an
/// int32 count array (0..3, every fifth null) on any column (other types: NoMatchingKernel; a
/// row past the offset range: Overflow). Each output validates with the input's length.
fn exerciseRound25(ctx: *zarrow.compute.ExecContext, a: ArrayRef) void {
    const d: zarrow.compute.Datum = .{ .array = a };
    const len = a.data().length;
    switch (a.data().data_type) {
        .binary, .large_binary => {
            expectCall(ctx, "binary_join_element_wise", .scalar, &.{ d, d, d }, .none, len, &.{error.Overflow});
            for ([_]zarrow.compute.JoinNullHandling{ .emit_null, .skip, .replace }) |h| {
                expectCall(ctx, "binary_join_element_wise", .scalar, &.{ d, d }, .{ .join = .{ .null_handling = h, .null_replacement = "\xff" } }, len, &.{error.Overflow});
            }
        },
        else => {},
    }
    var cb = zarrow.Int32Builder.init(std.heap.page_allocator, len) catch return;
    defer cb.deinit();
    for (0..len) |i| (if (i % 5 == 4) cb.appendNull() else cb.append(@intCast(i % 4))) catch return;
    var counts = cb.finish() catch return;
    defer counts.release();
    expectCall(ctx, "binary_repeat", .scalar, &.{ d, .{ .array = counts } }, .none, len, &.{ error.NoMatchingKernel, error.Overflow });
}

/// Round 24 kernels on any decoded array: is_null, is_valid and true_unless_null (logical nulls,
/// also of unions and run-end encoded arrays) and count in every mode, which must agree with
/// is_null; and the arithmetic / math / rounding smoke on null-type arguments (the array itself
/// when it is of the null type, and a null-type array of its length): a null-type result of the
/// same length. Nothing may fail.
fn exerciseRound24(ctx: *zarrow.compute.ExecContext, a: ArrayRef) void {
    const d: zarrow.compute.Datum = .{ .array = a };
    const len = a.data().length;
    expectCall(ctx, "is_valid", .scalar, &.{d}, .none, len, &.{});
    expectCall(ctx, "true_unless_null", .scalar, &.{d}, .none, len, &.{});
    var nulls: usize = 0;
    if (ctx.invoke("is_null", .scalar, &.{d}, .none)) |out| {
        var o = out;
        defer o.release();
        expectValid("is_null", o.array, len);
        const bits = o.array.data();
        for (0..len) |i| nulls += @intFromBool((zarrow.BooleanArray{ .data = bits }).value(i) catch false);
    } else |err| std.debug.panic("fuzz-invariant: is_null failed on a valid {s} array: {s}", .{ @tagName(a.data().data_type.id()), @errorName(err) });
    for ([_]zarrow.compute.CountMode{ .only_valid, .only_null, .all }) |mode| {
        var out = ctx.invoke("count", .aggregate, &.{d}, .{ .count = .{ .mode = mode } }) catch |err|
            std.debug.panic("fuzz-invariant: count failed on a valid {s} array: {s}", .{ @tagName(a.data().data_type.id()), @errorName(err) });
        defer out.release();
        const want: usize = switch (mode) {
            .only_valid => len - nulls,
            .only_null => nulls,
            .all => len,
        };
        if (out.scalar.value.i64 != @as(i64, @intCast(want))) std.debug.panic("fuzz-invariant: count {s} is {d}, is_null says {d}", .{ @tagName(mode), out.scalar.value.i64, want });
    }
    var made = zarrow.compute.datumBuildNullLikeWithAllocator(std.heap.page_allocator, .{ .null = {} }, len) catch return;
    defer made.release();
    const inputs = [_]zarrow.compute.Datum{ made, d };
    for (inputs[0..if (a.data().data_type == .null) 2 else 1]) |n| {
        inline for (.{ "add", "subtract_checked", "divide", "power", "atan2", "shift_left", "bit_wise_and", "round_binary" }) |name| {
            expectNullResult(ctx, name, .scalar, &.{ n, n }, len);
        }
        inline for (.{ "abs", "negate_checked", "sign", "sqrt", "ln_checked", "ceil", "round", "bit_wise_not" }) |name| {
            expectNullResult(ctx, name, .scalar, &.{n}, len);
        }
        expectNullResult(ctx, "pairwise_diff", .vector, &.{n}, len);
    }
}

/// `name` on null-type arguments returns a valid null-type array of `len` rows.
fn expectNullResult(ctx: *zarrow.compute.ExecContext, name: []const u8, kind: zarrow.compute.FunctionKind, args: []const zarrow.compute.Datum, len: usize) void {
    var out = ctx.invoke(name, kind, args, .none) catch |err| std.debug.panic("fuzz-invariant: {s} failed on null-type arguments: {s}", .{ name, @errorName(err) });
    defer out.release();
    if (out.dataType() != .null) std.debug.panic("fuzz-invariant: {s} of null-type arguments is not of the null type", .{name});
    expectValid(name, out.array, len);
}

/// Round 23 kernels on any decoded array: take of the reversed rows with every third index null,
/// filter of the even rows, run_end_encode (unsupported types: NoMatchingKernel) and
/// run_end_decode back to as many rows, dictionary_decode, and modulo / remainder (+ checked) and
/// hypot of an integer or float column with itself. Each succeeds (or fails with a listed error)
/// and every array output validates with the expected length.
fn exerciseRound23(ctx: *zarrow.compute.ExecContext, a: ArrayRef) void {
    const gpa = std.heap.page_allocator;
    const d: zarrow.compute.Datum = .{ .array = a };
    const len = a.data().length;
    var ib = zarrow.Int64Builder.init(gpa, len) catch return;
    defer ib.deinit();
    var pb = zarrow.BooleanBuilder.init(gpa, len) catch return;
    defer pb.deinit();
    for (0..len) |i| {
        (if (i % 3 == 2) ib.appendNull() else ib.append(@intCast(len - 1 - i))) catch return;
        pb.append(i % 2 == 0) catch return;
    }
    var idx = ib.finish() catch return;
    defer idx.release();
    var pred = pb.finish() catch return;
    defer pred.release();
    expectCall(ctx, "take", .vector, &.{ d, .{ .array = idx } }, .none, len, &.{});
    expectCall(ctx, "filter", .vector, &.{ d, .{ .array = pred } }, .none, (len + 1) / 2, &.{});
    expectCall(ctx, "dictionary_decode", .vector, &.{d}, .none, len, &.{});
    if (ctx.invoke("run_end_encode", .vector, &.{d}, .none)) |enc| {
        var e = enc;
        defer e.release();
        expectValid("run_end_encode", e.array, len);
        expectCall(ctx, "run_end_decode", .vector, &.{e}, .none, len, &.{});
    } else |err| if (err != error.NoMatchingKernel) std.debug.panic("fuzz-invariant: run_end_encode failed on a valid {s} array: {s}", .{ @tagName(a.data().data_type.id()), @errorName(err) });
    switch (a.data().data_type) {
        .int8, .int16, .int32, .int64, .uint8, .uint16, .uint32, .uint64, .float, .double => {
            inline for (.{ "modulo", "modulo_checked", "remainder", "remainder_checked" }) |name| {
                expectCall(ctx, name, .scalar, &.{ d, d }, .none, len, &.{ error.DivideByZero, error.Overflow });
            }
            // InvalidCast: an integer that float64 cannot hold exactly (Arrow's safe implicit cast).
            expectCall(ctx, "hypot", .scalar, &.{ d, d }, .none, len, &.{error.InvalidCast});
        },
        else => {},
    }
}
