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
            msg.deinit();
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
        t.deinit();
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
        for (0..batch.numColumns()) |i| {
            const col = batch.column(i).*;
            if (fb.noteArray(col)) try exerciseCompute(gpa, col, prng.random());
        }
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

/// add / less / sum / min_max (numeric), and_kleene / any (bool), less and string kernels
/// (strings), temporal extraction (dates, timestamps), cast (numbers, strings, binaries, dates,
/// timestamps, dictionaries), is_null and count (any type) on a decoded array that passed
/// validateFull and on a slice of it: the call must succeed (or fail with an expected error such as
/// InvalidCast) and an array output must validate with the input length.
fn exerciseKernels(gpa: Allocator, arr: ArrayRef, sliced: ArrayRef) void {
    var ctx = zarrow.compute.ExecContext.initWithConfig(gpa, kernelRegistry(), .{ .overflow_mode = .wrapping });
    for ([_]ArrayRef{ arr, sliced }) |a| {
        const d: zarrow.compute.Datum = .{ .array = a };
        const len = a.data().length;
        const binary = [_]zarrow.compute.Datum{ d, d };
        // is_null / count reject unions and run-end encoded arrays (NoMatchingKernel).
        expectKernel(&ctx, "is_null", .scalar, &.{d}, len, true);
        expectKernel(&ctx, "count", .aggregate, &.{d}, null, true);
        switch (a.data().data_type) {
            .int8, .int16, .int32, .int64, .uint8, .uint16, .uint32, .uint64, .float, .double => {
                expectKernel(&ctx, "add", .scalar, &binary, len, false);
                expectKernel(&ctx, "less", .scalar, &binary, len, false);
                expectKernel(&ctx, "sum", .aggregate, &.{d}, null, false);
                expectKernel(&ctx, "min_max", .aggregate, &.{d}, null, false);
            },
            .bool => {
                expectKernel(&ctx, "and_kleene", .scalar, &binary, len, false);
                expectKernel(&ctx, "less", .scalar, &binary, len, false);
                expectKernel(&ctx, "any", .aggregate, &.{d}, null, false);
            },
            .string, .large_string, .string_view => {
                expectKernel(&ctx, "less", .scalar, &binary, len, false);
                // validateFull passed, so the values are UTF-8: the utf8 kernels must succeed.
                expectCall(&ctx, "utf8_upper", .scalar, &.{d}, .none, len, &.{});
                expectCall(&ctx, "utf8_slice_codeunits", .scalar, &.{d}, .{ .slice = .{ .start = -3, .stop = 1, .step = -1 } }, len, &.{});
                expectCall(&ctx, "binary_join_element_wise", .scalar, &binary, .none, len, &.{});
                expectCall(&ctx, "cast", .scalar, &.{d}, cast(.{ .large_string = {} }, true), len, &.{});
                expectCall(&ctx, "cast", .scalar, &.{d}, cast(.{ .binary_view = {} }, true), len, &.{});
                expectCall(&ctx, "cast", .scalar, &.{d}, cast(.{ .int32 = {} }, true), len, &.{error.InvalidCast});
                expectCall(&ctx, "cast", .scalar, &.{d}, cast(.{ .double = {} }, true), len, &.{error.InvalidCast});
            },
            .binary, .large_binary, .binary_view => expectCall(&ctx, "cast", .scalar, &.{d}, cast(.{ .string = {} }, true), len, &.{ error.InvalidCast, error.Overflow }),
            .date32, .date64, .timestamp => {
                // A timestamp with a non-UTC timezone is NotImplemented; casts may overflow when safe.
                const tz_ok = [_]anyerror{error.NotImplemented};
                expectCall(&ctx, "year", .scalar, &.{d}, .none, len, &tz_ok);
                expectCall(&ctx, "iso_calendar", .scalar, &.{d}, .none, len, &tz_ok);
                expectCall(&ctx, "cast", .scalar, &.{d}, cast(.{ .date32 = {} }, false), len, &.{});
                expectCall(&ctx, "cast", .scalar, &.{d}, cast(.{ .timestamp = .{ .unit = .nanosecond, .timezone = null } }, true), len, &.{error.InvalidCast});
            },
            .dictionary => |dict| expectCall(&ctx, "cast", .scalar, &.{d}, cast(dict.value_type.*, true), len, &.{error.UnsupportedType}),
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
fn expectCall(ctx: *zarrow.compute.ExecContext, name: []const u8, kind: zarrow.compute.FunctionKind, args: []const zarrow.compute.Datum, options: zarrow.compute.Options, expected_len: ?usize, allowed: []const anyerror) void {
    var out = ctx.invoke(name, kind, args, options) catch |e| {
        for (allowed) |ok| if (e == ok) return;
        std.debug.panic("fuzz-invariant: kernel {s} failed on a valid {s} array: {s}", .{ name, @tagName(args[0].dataType().id()), @errorName(e) });
    };
    defer out.release();
    if (out.asArray()) |a| expectValid(name, a, expected_len orelse std.debug.panic("fuzz-invariant: {s} returned an array", .{name}));
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
    if (valid) try exerciseCompute(gpa, arr, prng.random());
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
