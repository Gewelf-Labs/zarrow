//! Sorts and partitions (Arrow C++ "Sorts and partitions"): sort_indices, array_sort_indices,
//! partition_nth_indices, select_k_unstable and rank, vector functions returning uint64 row
//! indices (rank: 1-based ranks). `sortIndicesRecordBatch` / `sortIndicesTable` sort the rows
//! of a record batch or table by named columns.
//!
//! - sort_indices(col0, col1, ...): arrays or chunked arrays of equal length, compared in
//!   argument order; `SortOptions.sort_keys` gives one order per column (empty: every column in
//!   `order`), `null_placement` places nulls. array_sort_indices takes one column (`order`,
//!   `null_placement`). Both are stable: equal rows keep their input order.
//! - Value order (Arrow C++): numbers, dates, timestamps, times, durations and decimals by value
//!   (-0.0 equals 0.0), bool false before true, string / binary layouts and fixed_size_binary by
//!   bytes; dictionaries by their decoded values. NaN sits between the values and the nulls in
//!   both orders: `at_end` gives values, NaNs, nulls; `at_start` nulls, NaNs, values.
//! - partition_nth_indices (`PartitionNthOptions`: pivot <= length, else `InvalidOptions`): every
//!   row before position `pivot` sorts before or equal to the row at `pivot` and every row after
//!   it after or equal (the output is a full stable sort, which is one such partition).
//! - select_k_unstable (`SelectKOptions`: k, sort_keys): the first `min(k, length)` rows of the
//!   sort (nulls and NaN always last, as Arrow); ties keep input order.
//! - rank (`RankOptions`: order, null_placement, tiebreaker min / max / first / dense): ranks of
//!   the sort, equal values (nulls with nulls, NaN with NaN) tied.
//!
//! A single integer, temporal, decimal32 / decimal64, bool or float column sorts with a stable
//! LSD radix sort on order-preserving keys; other columns and several keys with a comparison
//! sort that breaks ties by row index. Every column passes `validateLayout`; at most 2^32 - 1
//! rows (`InvalidInput` past it).
const std = @import("std");
const core = @import("core.zig");
const dense = @import("dense.zig");
const bitmap = @import("../bitmap.zig");
const concat = @import("../concat_array_refs.zig");
const datatype = @import("../datatype.zig");

const ArrayData = @import("../array/array_data.zig").ArrayData;
const PrimitiveArray = @import("../array/primitive_array.zig").PrimitiveArray;
const OwnedBuffer = @import("../buffer.zig").OwnedBuffer;
const ArrayRef = core.ArrayRef;
const DataType = core.DataType;
const Datum = core.Datum;
const ExecContext = core.ExecContext;
const FunctionRegistry = core.FunctionRegistry;
const KernelError = core.KernelError;
const Options = core.Options;
const SortOrder = core.SortOrder;
const SortNullPlacement = core.SortNullPlacement;
const RecordBatch = @import("../record_batch.zig").RecordBatch;
const Table = @import("../table.zig").Table;

const uint64_type = DataType{ .uint64 = {} };

/// Register sort_indices, array_sort_indices, partition_nth_indices, select_k_unstable and rank.
pub fn register(registry: *FunctionRegistry) KernelError!void {
    try registry.registerVectorKernel("sort_indices", .{
        .signature = .{ .arity = 1, .variadic = true, .type_check = sortable, .options_check = sortOptions, .result_type_fn = dense.resultTypeOf(uint64_type) },
        .exec = sortIndicesExec,
    });
    try registry.registerVectorKernel("array_sort_indices", .{
        .signature = .{ .arity = 1, .type_check = sortable, .options_check = sortOptions, .result_type_fn = dense.resultTypeOf(uint64_type) },
        .exec = sortIndicesExec,
    });
    try registry.registerVectorKernel("partition_nth_indices", .{
        .signature = .{ .arity = 1, .type_check = sortable, .options_check = partitionOptions, .result_type_fn = dense.resultTypeOf(uint64_type) },
        .exec = partitionExec,
    });
    try registry.registerVectorKernel("select_k_unstable", .{
        .signature = .{ .arity = 1, .variadic = true, .type_check = sortable, .options_check = selectKOptions, .result_type_fn = dense.resultTypeOf(uint64_type) },
        .exec = selectKExec,
    });
    try registry.registerVectorKernel("rank", .{
        .signature = .{ .arity = 1, .type_check = sortable, .options_check = rankOptions, .result_type_fn = dense.resultTypeOf(uint64_type) },
        .exec = rankExec,
    });
}

fn sortOptions(options: Options) bool {
    return options == .none or options == .sort;
}

fn partitionOptions(options: Options) bool {
    return options == .partition_nth;
}

fn selectKOptions(options: Options) bool {
    return options == .select_k;
}

fn rankOptions(options: Options) bool {
    return options == .none or options == .rank;
}

/// Whether a column of `dt` can be sorted.
pub fn isSortable(dt: DataType) bool {
    return switch (dt) {
        .int8, .int16, .int32, .int64, .uint8, .uint16, .uint32, .uint64, .float, .double, .bool => true,
        .date32, .date64, .timestamp, .time32, .time64, .duration => true,
        .decimal32, .decimal64, .decimal128, .decimal256 => true,
        .string, .large_string, .string_view, .binary, .large_binary, .binary_view, .fixed_size_binary => true,
        .dictionary => |d| isSortable(d.value_type.*) and d.value_type.* != .dictionary,
        else => false,
    };
}

fn sortable(args: []const Datum) bool {
    if (args.len == 0) return false;
    for (args) |a| if (a == .scalar or !isSortable(a.dataType())) return false;
    return true;
}

// ------------------------------------------------------------------------------ columns

/// Where a row of a column sorts relative to the values: its class decides first.
const Class = enum(u2) { value, nan, null };

/// The typed rows of one sort column.
const Values = union(enum) {
    i8: []const i8,
    i16: []const i16,
    i32: []const i32,
    i64: []const i64,
    u8: []const u8,
    u16: []const u16,
    u32: []const u32,
    u64: []const u64,
    f32: []const f32,
    f64: []const f64,
    i128: []const i128,
    i256: []const i256,
    boolean: struct { bits: []const u8, offset: usize },
    bytes: dense.StringOperand,
    fixed: struct { data: []const u8, width: usize, offset: usize },
};

/// One sort column: the rows of all chunks as one validated array.
const Column = struct {
    arr: ArrayRef,
    values: Values,
    order: SortOrder,

    fn deinit(self: *Column) void {
        self.arr.release();
    }

    fn init(ctx: *ExecContext, datum: Datum, order: SortOrder) KernelError!Column {
        var arr = try asArray(ctx, datum);
        errdefer arr.release();
        const data = arr.data();
        data.validateLayout() catch return error.InvalidInput;
        return .{ .arr = arr, .values = try valuesOf(arr), .order = order };
    }

    inline fn class(self: *const Column, i: usize) Class {
        if (self.arr.data().isNull(i)) return .null;
        return switch (self.values) {
            .f32 => |v| if (std.math.isNan(v[i])) .nan else .value,
            .f64 => |v| if (std.math.isNan(v[i])) .nan else .value,
            else => .value,
        };
    }

    /// Compare the values of two `value` rows (ascending; -0.0 equals 0.0).
    inline fn compareValues(self: *const Column, i: usize, j: usize) std.math.Order {
        return switch (self.values) {
            .boolean => |b| std.math.order(@intFromBool(bitmap.bitIsSet(b.bits, b.offset + i)), @intFromBool(bitmap.bitIsSet(b.bits, b.offset + j))),
            .bytes => |op| switch (op) {
                inline else => |r| std.mem.order(u8, r.get(i), r.get(j)),
            },
            .fixed => |f| std.mem.order(u8, f.data[(f.offset + i) * f.width ..][0..f.width], f.data[(f.offset + j) * f.width ..][0..f.width]),
            inline else => |v| std.math.order(v[i], v[j]),
        };
    }

    /// Compare two rows with the column's order and `placement` (classes are not reversed by a
    /// descending order).
    inline fn compare(self: *const Column, i: usize, j: usize, placement: SortNullPlacement) std.math.Order {
        const ci = self.class(i);
        const cj = self.class(j);
        if (ci != cj) {
            const a: u2 = @intFromEnum(ci);
            const b: u2 = @intFromEnum(cj);
            return if (placement == .at_end) std.math.order(a, b) else std.math.order(b, a);
        }
        if (ci != .value) return .eq;
        const o = self.compareValues(i, j);
        return if (self.order == .ascending) o else o.invert();
    }
};

/// The column as one array: a chunked column is concatenated, a dictionary decoded.
fn asArray(ctx: *ExecContext, datum: Datum) KernelError!ArrayRef {
    if (datum.dataType() == .dictionary) {
        var decoded = try ctx.invoke("cast", .scalar, &.{datum}, .{ .cast = .{ .to_type = datum.dataType().dictionary.value_type.* } });
        defer decoded.release();
        return asArray(ctx, decoded);
    }
    switch (datum) {
        .array => |a| return a.retain(),
        .chunked => |c| {
            if (c.numChunks() == 1) return c.chunks()[0].retain();
            if (c.numChunks() == 0) return dense.nullArray(ctx.allocator, c.dataType(), 0);
            return concat.concatArrayRefs(ctx.allocator, c.dataType(), c.chunks()) catch |err| switch (err) {
                error.OutOfMemory => error.OutOfMemory,
                else => error.InvalidInput,
            };
        },
        .scalar => return error.InvalidInput,
    }
}

fn valuesOf(arr: ArrayRef) KernelError!Values {
    const data = arr.data();
    return switch (data.data_type) {
        .bool => .{ .boolean = .{ .bits = data.buffers[1].data, .offset = data.offset } },
        .string, .large_string, .string_view, .binary, .large_binary, .binary_view => .{ .bytes = try dense.StringOperand.init(.{ .array = arr }) },
        .fixed_size_binary => |f| blk: {
            const w: usize = @intCast(f.byte_width);
            if (w == 0 or data.buffers[1].data.len / w < data.offset + data.length) return error.InvalidInput;
            break :blk .{ .fixed = .{ .data = data.buffers[1].data, .width = w, .offset = data.offset } };
        },
        .int8 => .{ .i8 = try typed(i8, data) },
        .int16 => .{ .i16 = try typed(i16, data) },
        .int32 => .{ .i32 = try typed(i32, data) },
        .int64 => .{ .i64 = try typed(i64, data) },
        .uint8 => .{ .u8 = try typed(u8, data) },
        .uint16 => .{ .u16 = try typed(u16, data) },
        .uint32 => .{ .u32 = try typed(u32, data) },
        .uint64 => .{ .u64 = try typed(u64, data) },
        .float => .{ .f32 = try typed(f32, data) },
        .double => .{ .f64 = try typed(f64, data) },
        .date32, .time32, .decimal32 => .{ .i32 = try typed(i32, data) },
        .date64, .timestamp, .time64, .duration, .decimal64 => .{ .i64 = try typed(i64, data) },
        .decimal128 => .{ .i128 = try typed(i128, data) },
        .decimal256 => .{ .i256 = try typed(i256, data) },
        else => error.UnsupportedType,
    };
}

fn typed(comptime T: type, data: *const ArrayData) KernelError![]const T {
    return (PrimitiveArray(T){ .data = data }).values() catch error.InvalidInput;
}

/// The sort columns of one call.
const Columns = struct {
    allocator: std.mem.Allocator,
    cols: []Column,
    len: usize,
    placement: SortNullPlacement,

    fn init(ctx: *ExecContext, args: []const Datum, orders: []const SortOrder, placement: SortNullPlacement) KernelError!Columns {
        const len = try core.inferNaryExecLen(args);
        if (len > std.math.maxInt(u32)) return error.InvalidInput;
        const cols = ctx.allocator.alloc(Column, args.len) catch return error.OutOfMemory;
        var made: usize = 0;
        errdefer {
            for (cols[0..made]) |*c| c.deinit();
            ctx.allocator.free(cols);
        }
        for (args, orders, cols) |a, o, *c| {
            c.* = try Column.init(ctx, a, o);
            made += 1;
            if (c.arr.data().length != len) return error.InvalidInput;
        }
        return .{ .allocator = ctx.allocator, .cols = cols, .len = len, .placement = placement };
    }

    fn deinit(self: *Columns) void {
        for (self.cols) |*c| c.deinit();
        self.allocator.free(self.cols);
    }

    fn compare(self: *const Columns, i: usize, j: usize) std.math.Order {
        for (self.cols) |*c| {
            const o = c.compare(i, j, self.placement);
            if (o != .eq) return o;
        }
        return .eq;
    }

    fn lessThan(self: *const Columns, a: u32, b: u32) bool {
        return switch (self.compare(a, b)) {
            .lt => true,
            .gt => false,
            .eq => a < b,
        };
    }

    /// The stable sort permutation of the rows.
    fn sort(self: *const Columns) KernelError![]u32 {
        const idx = self.allocator.alloc(u32, self.len) catch return error.OutOfMemory;
        errdefer self.allocator.free(idx);
        for (idx, 0..) |*d, i| d.* = @intCast(i);
        if (self.cols.len == 1 and try radixSort(self.allocator, &self.cols[0], self.placement, idx)) return idx;
        std.mem.sort(u32, idx, self, lessThan);
        return idx;
    }
};

// ------------------------------------------------------------------------------ radix sort

/// The order-preserving unsigned key of a value (floats: -0.0 as 0.0; NaN never reaches here).
inline fn radixKey(comptime K: type, v: anytype) K {
    const T = @TypeOf(v);
    if (T == bool) return @intFromBool(v);
    if (@typeInfo(T) == .float) {
        const U = @Int(.unsigned, @bitSizeOf(T));
        const x: T = if (v == 0) 0 else v;
        const bits: U = @bitCast(x);
        const sign: U = @as(U, 1) << (@bitSizeOf(U) - 1);
        return @intCast(if (bits & sign != 0) ~bits else bits | sign);
    }
    const U = @Int(.unsigned, @bitSizeOf(T));
    const u: U = @bitCast(v);
    if (@typeInfo(T).int.signedness == .signed) return @intCast(u ^ (@as(U, 1) << (@bitSizeOf(U) - 1)));
    return @intCast(u);
}

/// Sort `idx` (the identity) by one column with an LSD radix sort when the column has a radix
/// key; false when it has none (the caller uses the comparison sort).
fn radixSort(allocator: std.mem.Allocator, col: *const Column, placement: SortNullPlacement, idx: []u32) KernelError!bool {
    switch (col.values) {
        inline .i8, .i16, .i32, .i64, .u8, .u16, .u32, .u64, .f32, .f64 => |v| {
            const T = @typeInfo(@TypeOf(v)).pointer.child;
            const K = if (@bitSizeOf(T) <= 32) u32 else u64;
            try radixWith(K, allocator, col, placement, idx, struct {
                fn key(c: *const Column, i: usize) K {
                    return radixKey(K, @field(c.values, @typeName(T))[i]);
                }
            }.key);
            return true;
        },
        .boolean => {
            try radixWith(u32, allocator, col, placement, idx, struct {
                fn key(c: *const Column, i: usize) u32 {
                    return @intFromBool(bitmap.bitIsSet(c.values.boolean.bits, c.values.boolean.offset + i));
                }
            }.key);
            return true;
        },
        else => return false,
    }
}

fn radixWith(comptime K: type, allocator: std.mem.Allocator, col: *const Column, placement: SortNullPlacement, idx: []u32, comptime keyFn: fn (*const Column, usize) K) KernelError!void {
    const n = idx.len;
    // Rows by class: values first into `vals`, NaN and null rows kept in input order.
    var vals: std.ArrayList(u32) = .empty;
    defer vals.deinit(allocator);
    var nans: std.ArrayList(u32) = .empty;
    defer nans.deinit(allocator);
    var nulls: std.ArrayList(u32) = .empty;
    defer nulls.deinit(allocator);
    vals.ensureTotalCapacity(allocator, n) catch return error.OutOfMemory;
    for (0..n) |i| switch (col.class(i)) {
        .value => vals.appendAssumeCapacity(@intCast(i)),
        .nan => nans.append(allocator, @intCast(i)) catch return error.OutOfMemory,
        .null => nulls.append(allocator, @intCast(i)) catch return error.OutOfMemory,
    };
    const m = vals.items.len;
    const keys = allocator.alloc(K, m) catch return error.OutOfMemory;
    defer allocator.free(keys);
    const desc = col.order == .descending;
    var diff: K = 0;
    for (keys, vals.items) |*k, i| {
        const x = keyFn(col, i);
        k.* = if (desc) ~x else x;
    }
    if (m > 0) for (keys) |k| {
        diff |= k ^ keys[0];
    };
    const tmp_keys = allocator.alloc(K, m) catch return error.OutOfMemory;
    defer allocator.free(tmp_keys);
    const tmp_idx = allocator.alloc(u32, m) catch return error.OutOfMemory;
    defer allocator.free(tmp_idx);
    var src_k = keys;
    var src_i = vals.items;
    var dst_k = tmp_keys;
    var dst_i = tmp_idx;
    var shift: u7 = 0;
    while (shift < @bitSizeOf(K)) : (shift += 8) {
        // A digit every key shares needs no pass.
        if ((diff >> @intCast(shift)) & 0xFF == 0) continue;
        var counts: [257]usize = undefined;
        @memset(&counts, 0);
        for (src_k) |k| counts[@as(usize, @intCast((k >> @intCast(shift)) & 0xFF)) + 1] += 1;
        for (1..257) |d| counts[d] += counts[d - 1];
        for (src_k, src_i) |k, i| {
            const d: usize = @intCast((k >> @intCast(shift)) & 0xFF);
            dst_k[counts[d]] = k;
            dst_i[counts[d]] = i;
            counts[d] += 1;
        }
        std.mem.swap([]K, &src_k, &dst_k);
        std.mem.swap([]u32, &src_i, &dst_i);
    }
    var pos: usize = 0;
    if (placement == .at_start) {
        @memcpy(idx[pos..][0..nulls.items.len], nulls.items);
        pos += nulls.items.len;
        @memcpy(idx[pos..][0..nans.items.len], nans.items);
        pos += nans.items.len;
    }
    @memcpy(idx[pos..][0..m], src_i);
    pos += m;
    if (placement == .at_end) {
        @memcpy(idx[pos..][0..nans.items.len], nans.items);
        pos += nans.items.len;
        @memcpy(idx[pos..][0..nulls.items.len], nulls.items);
    }
}

// ------------------------------------------------------------------------------ kernels

/// The order of every argument: `sort_keys` (one per argument) or `order` for all.
fn ordersOf(allocator: std.mem.Allocator, n: usize, keys: []const core.SortKey, order: SortOrder) KernelError![]SortOrder {
    if (keys.len != 0 and keys.len != n) return error.InvalidOptions;
    const out = allocator.alloc(SortOrder, n) catch return error.OutOfMemory;
    for (out, 0..) |*o, i| o.* = if (keys.len == 0) order else keys[i].order;
    return out;
}

/// A uint64 array of the indices `idx` (the first `n`).
fn indexArray(allocator: std.mem.Allocator, idx: []const u32) KernelError!ArrayRef {
    var buf = OwnedBuffer.initUninit(allocator, idx.len * 8) catch return error.OutOfMemory;
    for (buf.typedSlice(u64)[0..idx.len], idx) |*d, i| d.* = i;
    return dense.finishFixed(allocator, uint64_type, idx.len, &buf, null);
}

fn sortIndicesExec(ctx: *ExecContext, args: []const Datum, options: Options) KernelError!Datum {
    const o: core.SortOptions = if (options == .sort) options.sort else .{};
    const orders = try ordersOf(ctx.allocator, args.len, o.sort_keys, o.order);
    defer ctx.allocator.free(orders);
    var cols = try Columns.init(ctx, args, orders, o.null_placement);
    defer cols.deinit();
    const idx = try cols.sort();
    defer ctx.allocator.free(idx);
    return .{ .array = try indexArray(ctx.allocator, idx) };
}

fn partitionExec(ctx: *ExecContext, args: []const Datum, options: Options) KernelError!Datum {
    const o = options.partition_nth;
    var cols = try Columns.init(ctx, args, &.{.ascending}, o.null_placement);
    defer cols.deinit();
    if (o.pivot > cols.len) return error.InvalidOptions;
    // ponytail: a full stable sort is one valid partition; an nth_element selection would
    // be O(n) instead of O(n log n).
    const idx = try cols.sort();
    defer ctx.allocator.free(idx);
    return .{ .array = try indexArray(ctx.allocator, idx) };
}

fn selectKExec(ctx: *ExecContext, args: []const Datum, options: Options) KernelError!Datum {
    const o = options.select_k;
    const orders = try ordersOf(ctx.allocator, args.len, o.sort_keys, .ascending);
    defer ctx.allocator.free(orders);
    var cols = try Columns.init(ctx, args, orders, .at_end);
    defer cols.deinit();
    // ponytail: sorts every row, then keeps k; a bounded heap would be O(n log k).
    const idx = try cols.sort();
    defer ctx.allocator.free(idx);
    return .{ .array = try indexArray(ctx.allocator, idx[0..@min(o.k, idx.len)]) };
}

fn rankExec(ctx: *ExecContext, args: []const Datum, options: Options) KernelError!Datum {
    const o: core.RankOptions = if (options == .rank) options.rank else .{};
    var cols = try Columns.init(ctx, args, &.{o.order}, o.null_placement);
    defer cols.deinit();
    const idx = try cols.sort();
    defer ctx.allocator.free(idx);
    const n = idx.len;
    var buf = OwnedBuffer.initUninit(ctx.allocator, n * 8) catch return error.OutOfMemory;
    errdefer buf.deinit();
    const ranks = buf.typedSlice(u64)[0..n];
    var start: usize = 0;
    var dense_rank: u64 = 0;
    while (start < n) {
        var end = start + 1;
        while (end < n and cols.compare(idx[start], idx[end]) == .eq) end += 1;
        dense_rank += 1;
        for (idx[start..end], start..) |row, pos| ranks[row] = switch (o.tiebreaker) {
            .min => start + 1,
            .max => end,
            .first => pos + 1,
            .dense => dense_rank,
        };
        start = end;
    }
    return .{ .array = try dense.finishFixed(ctx.allocator, uint64_type, n, &buf, null) };
}

/// Sort the rows of `batch` by the columns `keys` names (each with its order): uint64 row
/// indices, as `sort_indices` on those columns.
pub fn sortIndicesRecordBatch(ctx: *ExecContext, batch: *const RecordBatch, keys: []const core.SortKey, null_placement: SortNullPlacement) KernelError!ArrayRef {
    if (keys.len == 0) return error.InvalidOptions;
    const args = ctx.allocator.alloc(Datum, keys.len) catch return error.OutOfMemory;
    defer ctx.allocator.free(args);
    for (keys, args) |k, *a| a.* = .{ .array = (batch.columnByName(k.name) orelse return error.InvalidOptions).* };
    var out = try ctx.invokeVector("sort_indices", args, .{ .sort = .{ .sort_keys = keys, .null_placement = null_placement } });
    return out.asArray() orelse {
        out.release();
        return error.InvalidInput;
    };
}

/// `sortIndicesRecordBatch` for a table (chunked columns).
pub fn sortIndicesTable(ctx: *ExecContext, table: *const Table, keys: []const core.SortKey, null_placement: SortNullPlacement) KernelError!ArrayRef {
    if (keys.len == 0) return error.InvalidOptions;
    const args = ctx.allocator.alloc(Datum, keys.len) catch return error.OutOfMemory;
    defer ctx.allocator.free(args);
    for (keys, args) |k, *a| a.* = .{ .chunked = (table.columnByName(k.name) orelse return error.InvalidOptions).* };
    var out = try ctx.invokeVector("sort_indices", args, .{ .sort = .{ .sort_keys = keys, .null_placement = null_placement } });
    return out.asArray() orelse {
        out.release();
        return error.InvalidInput;
    };
}
