//! Sorts and partitions (Arrow C++ "Sorts and partitions"): sort_indices, array_sort_indices,
//! partition_nth_indices, select_k_unstable, rank, rank_quantile, rank_normal and search_sorted,
//! vector functions returning uint64 row indices (rank: 1-based ranks; rank_quantile /
//! rank_normal: float64). `sortIndicesRecordBatch` / `sortIndicesTable` sort the rows of a record
//! batch or table by named columns, as sort_indices does on a record batch or table datum.
//!
//! - sort_indices(col0, col1, ...): arrays or chunked arrays of equal length, compared in
//!   argument order, or one record batch / table whose columns `SortOptions.sort_keys` names (at
//!   least one key, else `InvalidOptions`, as Arrow). Each `SortKey` has its order and null
//!   placement; without keys every column uses `order` and nulls go last. `SortOptions.
//!   null_placement`, when set, overrides every key's (Arrow 26). array_sort_indices takes one
//!   column. Both are stable: equal rows keep their input order.
//! - Value order (Arrow C++): numbers, dates, timestamps, times, durations and decimals by value
//!   (-0.0 equals 0.0), bool false before true, string / binary layouts and fixed_size_binary by
//!   bytes; dictionaries by their decoded values; a struct by its fields in order (Arrow flattens
//!   a struct key: a null struct row is null in every field). NaN sits between the values and
//!   the nulls in both orders: `at_end` gives values, NaNs, nulls; `at_start` nulls, NaNs,
//!   values. Arrow 26 has no separate NaN placement, so neither do the options.
//! - partition_nth_indices (`PartitionNthOptions`: pivot <= length, else `InvalidOptions`;
//!   `null_placement`): the row a stable ascending sort puts at `pivot` is there, every row
//!   before it sorts before it and every row after it after it (rows are ordered with ties
//!   broken by row index, so that row is unique). One radix column: an MSD radix selection over
//!   packed (key, row) integers and one partition pass, O(n); other columns: quickselect
//!   (O(n) average, a sort of the remaining range past 2 log2(n) rounds).
//! - select_k_unstable (`SelectKOptions`: k, sort_keys with order and null placement): the first
//!   `min(k, length)` rows of the stable sort, from a bounded max-heap of k rows, O(n log k);
//!   ties keep input order (Arrow's order among ties is unspecified).
//! - rank (`RankOptions`: order, null_placement, tiebreaker min / max / first / dense): ranks of
//!   the sort, equal values (nulls with nulls, NaN with NaN) tied.
//! - rank_quantile / rank_normal (`RankQuantileOptions`: order, null_placement): (rows before +
//!   half the rows equal) / rows, nulls included, ties as rank; rank_normal maps it through the
//!   standard normal quantile function (Wichura AS 241).
//! - search_sorted(values, needles) (`SearchSortedOptions.side`): per needle the leftmost /
//!   rightmost insertion point in the ascending `values` (array or chunked; its nulls all at the
//!   start or the end, else `InvalidInput`; Arrow leaves that case unspecified), searched among
//!   the non-null values; values and needles of one sortable type (no struct); a null needle
//!   gives null, a scalar needle a scalar.
//!
//! A single integer, temporal, decimal32 / decimal64, bool or float column sorts with a stable
//! LSD radix sort on order-preserving keys; other columns and several keys with a comparison
//! sort that breaks ties by row index. Every column passes `validateLayout` (a dictionary also
//! its index bounds, when it is decoded); at most 2^32 - 1 rows (`InvalidInput` past it).
const std = @import("std");
const core = @import("core.zig");
const dense = @import("dense.zig");
const bitmap = @import("../bitmap.zig");
const concat = @import("../concat_array_refs.zig");
const datatype = @import("../datatype.zig");

const ArrayData = @import("../array/array_data.zig").ArrayData;
const PrimitiveArray = @import("../array/primitive_array.zig").PrimitiveArray;
const StructArray = @import("../array/struct_array.zig").StructArray;
const OwnedBuffer = @import("../buffer.zig").OwnedBuffer;
const SharedBuffer = @import("../buffer.zig").SharedBuffer;
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
const float64_type = DataType{ .double = {} };

/// Register sort_indices, array_sort_indices, partition_nth_indices, select_k_unstable, rank,
/// rank_quantile, rank_normal and search_sorted.
pub fn register(registry: *FunctionRegistry) KernelError!void {
    try registry.registerVectorKernel("sort_indices", .{
        .signature = .{ .arity = 1, .variadic = true, .type_check = sortable, .options_check = sortOptions, .result_type_fn = dense.resultTypeOf(uint64_type), .tabular = true },
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
        .signature = .{ .arity = 1, .variadic = true, .type_check = sortable, .options_check = selectKOptions, .result_type_fn = dense.resultTypeOf(uint64_type), .tabular = true },
        .exec = selectKExec,
    });
    try registry.registerVectorKernel("rank", .{
        .signature = .{ .arity = 1, .type_check = sortable, .options_check = rankOptions, .result_type_fn = dense.resultTypeOf(uint64_type) },
        .exec = rankExec,
    });
    inline for (.{ "rank_quantile", "rank_normal" }, .{ false, true }) |name, normal| {
        try registry.registerVectorKernel(name, .{
            .signature = .{ .arity = 1, .type_check = sortable, .options_check = rankQuantileOptions, .result_type_fn = dense.resultTypeOf(float64_type) },
            .exec = RankQuantile(normal).exec,
        });
    }
    try registry.registerVectorKernel("search_sorted", .{
        .signature = .{ .arity = 2, .type_check = searchSortedArgs, .options_check = searchSortedOptions, .result_type_fn = dense.resultTypeOf(uint64_type) },
        .exec = searchSortedExec,
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

fn rankQuantileOptions(options: Options) bool {
    return options == .none or options == .rank_quantile;
}

/// Whether a column of `dt` can be sorted.
pub fn isSortable(dt: DataType) bool {
    return switch (dt) {
        .int8, .int16, .int32, .int64, .uint8, .uint16, .uint32, .uint64, .half_float, .float, .double, .bool => true,
        .date32, .date64, .timestamp, .time32, .time64, .duration => true,
        .decimal32, .decimal64, .decimal128, .decimal256 => true,
        .string, .large_string, .string_view, .binary, .large_binary, .binary_view, .fixed_size_binary => true,
        .dictionary => |d| isSortable(d.value_type.*) and d.value_type.* != .dictionary and d.value_type.* != .struct_,
        // A struct sorts by its fields in order (Arrow flattens a struct sort key).
        .struct_ => |st| blk: {
            for (st.fields) |f| if (!isSortable(f.data_type.*)) break :blk false;
            break :blk st.fields.len > 0;
        },
        else => false,
    };
}

fn sortable(args: []const Datum) bool {
    if (args.len == 0) return false;
    // A record batch or table: its key columns are checked when the keys are resolved.
    if (args.len == 1 and args[0].isTabular()) return true;
    for (args) |a| if (a == .scalar or a.isTabular() or !isSortable(a.dataType())) return false;
    return true;
}

// ------------------------------------------------------------------------------ columns

/// Where a row of a column sorts relative to the values: its class decides first.
const Class = enum(u2) { value, nan, null };

/// The typed rows of one sort column (also the ordered min / max kernels).
pub const Values = union(enum) {
    i8: []const i8,
    i16: []const i16,
    i32: []const i32,
    i64: []const i64,
    u8: []const u8,
    u16: []const u16,
    u32: []const u32,
    u64: []const u64,
    f16: []const f16,
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
    placement: SortNullPlacement,

    fn deinit(self: *Column) void {
        self.arr.release();
    }

    fn init(ctx: *ExecContext, key: Key) KernelError!Column {
        var arr = try asArray(ctx, key.datum);
        errdefer arr.release();
        return initArray(arr, key);
    }

    /// A column over `arr` (taken on success; the caller still owns it on error).
    fn initArray(arr: ArrayRef, key: Key) KernelError!Column {
        arr.data().validateLayout() catch return error.InvalidInput;
        return .{ .arr = arr, .values = try valuesOf(arr), .order = key.order, .placement = key.placement };
    }

    inline fn class(self: *const Column, i: usize) Class {
        if (self.arr.data().isNull(i)) return .null;
        return switch (self.values) {
            .f16 => |v| if (std.math.isNan(v[i])) .nan else .value,
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

    /// Compare two rows with the column's order and null placement (classes are not reversed
    /// by a descending order).
    inline fn compare(self: *const Column, i: usize, j: usize) std.math.Order {
        const ci = self.class(i);
        const cj = self.class(j);
        if (ci != cj) {
            const a: u2 = @intFromEnum(ci);
            const b: u2 = @intFromEnum(cj);
            return if (self.placement == .at_end) std.math.order(a, b) else std.math.order(b, a);
        }
        if (ci != .value) return .eq;
        const o = self.compareValues(i, j);
        return if (self.order == .ascending) o else o.invert();
    }
};

/// `leaf` (a sortable column without children) with the null rows of its struct ancestors as
/// nulls: its own buffers under a new validity bitmap, the leaf's validity AND `parent_valid`
/// (bit `i` for row `i`).
fn withParentNulls(allocator: std.mem.Allocator, leaf: ArrayRef, parent_valid: []const u8) KernelError!ArrayRef {
    const d = leaf.data();
    d.validateLayout() catch return error.InvalidInput;
    if (d.children.len != 0 or d.dictionary != null or d.buffers.len == 0) return error.InvalidInput;
    const end = d.offset + d.length;
    var bits = OwnedBuffer.init(allocator, bitmap.byteLength(end)) catch return error.OutOfMemory;
    var nulls: usize = 0;
    for (0..d.length) |i| {
        if (bitmap.bitIsSet(parent_valid, i) and !d.isNull(i)) bitmap.setBit(bits.data, d.offset + i) else nulls += 1;
    }
    const buffers = allocator.alloc(SharedBuffer, d.buffers.len) catch {
        bits.deinit();
        return error.OutOfMemory;
    };
    defer allocator.free(buffers);
    buffers[0] = bits.toShared(bitmap.byteLength(end)) catch {
        bits.deinit();
        return error.OutOfMemory;
    };
    for (buffers[1..], d.buffers[1..]) |*b, src| b.* = src.retain();
    return ArrayRef.fromOwnedInlineBuffers(allocator, .{
        .data_type = d.data_type,
        .length = d.length,
        .offset = d.offset,
        .null_count = nulls,
        .buffers = buffers,
    }) catch {
        for (buffers) |*b| b.release();
        return error.OutOfMemory;
    };
}

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
        .scalar, .record_batch, .table => return error.InvalidInput,
    }
}

/// The typed rows of `arr` (validated by the caller); `UnsupportedType` for an unsortable type.
pub fn valuesOf(arr: ArrayRef) KernelError!Values {
    const data = arr.data();
    return switch (data.data_type) {
        .bool => .{ .boolean = .{ .bits = data.buffers[1].data, .offset = data.offset } },
        .string, .large_string, .string_view, .binary, .large_binary, .binary_view => .{ .bytes = try dense.StringOperand.init(.{ .array = arr }) },
        .fixed_size_binary => |f| blk: {
            const w: usize = @intCast(f.byte_width);
            // Width 0: every value is empty (all rows compare equal).
            if (w != 0 and data.buffers[1].data.len / w < data.offset + data.length) return error.InvalidInput;
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
        .half_float => .{ .f16 = try typed(f16, data) },
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

    /// The columns of `keys`; a struct key becomes one column per field, recursively (Arrow
    /// flattens it: a null struct row is null in every field, each field takes the key's order
    /// and null placement).
    fn init(ctx: *ExecContext, keys: []const Key) KernelError!Columns {
        if (keys.len == 0) return error.InvalidArity;
        const len = keys[0].datum.length();
        if (len > std.math.maxInt(u32)) return error.InvalidInput;
        var cols: std.ArrayList(Column) = .empty;
        errdefer {
            for (cols.items) |*c| c.deinit();
            cols.deinit(ctx.allocator);
        }
        for (keys) |k| {
            if (k.datum != .array and k.datum != .chunked) return error.InvalidInput;
            try addColumns(ctx, &cols, try asArray(ctx, k.datum), k, len, null);
        }
        return .{ .allocator = ctx.allocator, .cols = cols.toOwnedSlice(ctx.allocator) catch return error.OutOfMemory, .len = len };
    }

    /// Append the column(s) of `arr` (taken, also on error) to `cols`. `parent_valid`: the rows
    /// (bit `i` for row `i`) where every struct above `arr` is valid, null at the top.
    fn addColumns(ctx: *ExecContext, cols: *std.ArrayList(Column), arr: ArrayRef, key: Key, len: usize, parent_valid: ?[]const u8) KernelError!void {
        var owned = arr;
        const data = arr.data();
        if (data.data_type == .dictionary) {
            // A dictionary field of a struct: sorted by its decoded values too.
            defer owned.release();
            return addColumns(ctx, cols, try asArray(ctx, .{ .array = arr }), key, len, parent_valid);
        }
        if (data.data_type != .struct_) {
            errdefer owned.release();
            if (data.length != len) return error.InvalidInput;
            if (parent_valid) |pv| {
                const masked = try withParentNulls(ctx.allocator, arr, pv);
                owned.release();
                owned = masked;
            }
            cols.ensureUnusedCapacity(ctx.allocator, 1) catch return error.OutOfMemory;
            cols.appendAssumeCapacity(try Column.initArray(owned, key));
            return;
        }
        defer owned.release();
        if (data.length != len) return error.InvalidInput;
        data.validateLayout() catch return error.InvalidInput;
        // The rows valid in this struct and every struct above it.
        var valid: ?[]u8 = null;
        defer if (valid) |v| ctx.allocator.free(v);
        if (parent_valid != null or data.hasNulls()) {
            const v = ctx.allocator.alloc(u8, bitmap.byteLength(len)) catch return error.OutOfMemory;
            @memset(v, 0);
            for (0..len) |i| if ((parent_valid == null or bitmap.bitIsSet(parent_valid.?, i)) and !data.isNull(i)) bitmap.setBit(v, i);
            valid = v;
        }
        for (0..data.children.len) |j| {
            const field = (StructArray{ .data = data }).field(j) catch return error.InvalidInput;
            try addColumns(ctx, cols, field, key, len, valid);
        }
    }

    fn deinit(self: *Columns) void {
        for (self.cols) |*c| c.deinit();
        self.allocator.free(self.cols);
    }

    fn compare(self: *const Columns, i: usize, j: usize) std.math.Order {
        for (self.cols) |*c| {
            const o = c.compare(i, j);
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
        if (self.cols.len == 1 and try radixSort(self.allocator, &self.cols[0], idx)) return idx;
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

/// Call `body.run(K, keyFn, float)` with the column's order-preserving radix key (K: u32 or u64;
/// `float`: the column may hold NaN, which has no key): numbers, temporal types, decimal32 /
/// decimal64 and bool. False when the column has none (the caller compares rows instead).
fn withRadixKey(col: *const Column, body: anytype) KernelError!bool {
    switch (col.values) {
        inline .i8, .i16, .i32, .i64, .u8, .u16, .u32, .u64, .f16, .f32, .f64 => |v| {
            const T = @typeInfo(@TypeOf(v)).pointer.child;
            const K = if (@bitSizeOf(T) <= 32) u32 else u64;
            try body.run(K, struct {
                fn key(c: *const Column, i: usize) K {
                    return radixKey(K, @field(c.values, @typeName(T))[i]);
                }
            }.key, @typeInfo(T) == .float);
            return true;
        },
        .boolean => {
            try body.run(u32, struct {
                fn key(c: *const Column, i: usize) u32 {
                    return @intFromBool(bitmap.bitIsSet(c.values.boolean.bits, c.values.boolean.offset + i));
                }
            }.key, false);
            return true;
        },
        else => return false,
    }
}

/// Whether row `i` (valid) of a float column is NaN.
inline fn isNanAt(col: *const Column, i: usize) bool {
    return switch (col.values) {
        .f16 => |v| std.math.isNan(v[i]),
        .f32 => |v| std.math.isNan(v[i]),
        .f64 => |v| std.math.isNan(v[i]),
        else => false,
    };
}

/// The rows of one column by class, each list in row order; `limit` caps the NaN and null lists
/// (a selection of `limit` rows needs no more).
const Split = struct {
    vals: std.ArrayList(u32) = .empty,
    nans: std.ArrayList(u32) = .empty,
    nulls: std.ArrayList(u32) = .empty,

    fn init(allocator: std.mem.Allocator, col: *const Column, comptime float: bool, limit: usize) KernelError!Split {
        var s: Split = .{};
        errdefer s.deinit(allocator);
        const data = col.arr.data();
        const n = data.length;
        s.vals.ensureTotalCapacity(allocator, n) catch return error.OutOfMemory;
        const bits = dense.validityBits(data);
        for (0..n) |i| {
            const row: u32 = @intCast(i);
            if (bits) |b| if (!bitmap.bitIsSet(b, data.offset + i)) {
                if (s.nulls.items.len < limit) s.nulls.append(allocator, row) catch return error.OutOfMemory;
                continue;
            };
            if (float and isNanAt(col, i)) {
                if (s.nans.items.len < limit) s.nans.append(allocator, row) catch return error.OutOfMemory;
                continue;
            }
            s.vals.appendAssumeCapacity(row);
        }
        return s;
    }

    fn deinit(self: *Split, allocator: std.mem.Allocator) void {
        self.vals.deinit(allocator);
        self.nans.deinit(allocator);
        self.nulls.deinit(allocator);
    }

    /// Where the value rows start in the column's order.
    fn valueStart(self: *const Split, placement: SortNullPlacement) usize {
        return if (placement == .at_start) self.nulls.items.len + self.nans.items.len else 0;
    }

    /// Fill `out` with the rows in the column's order: the nulls, the NaNs and `values` (the
    /// value rows, already ordered), or `values`, NaNs and nulls (`at_end`), truncated.
    fn assemble(self: *const Split, placement: SortNullPlacement, values: []const u32, out: []u32) void {
        const parts = if (placement == .at_start)
            [3][]const u32{ self.nulls.items, self.nans.items, values }
        else
            [3][]const u32{ values, self.nans.items, self.nulls.items };
        var pos: usize = 0;
        for (parts) |p| {
            const take = @min(p.len, out.len - pos);
            @memcpy(out[pos..][0..take], p[0..take]);
            pos += take;
        }
    }
};

/// A radix key with its row in the low 32 bits: integers compare like (key, row) pairs.
fn Packed(comptime K: type) type {
    return if (K == u32) u64 else u128;
}

inline fn pack(comptime K: type, key: K, desc: bool, row: u32) Packed(K) {
    return (@as(Packed(K), if (desc) ~key else key) << 32) | row;
}

fn packedLess(comptime P: type) fn (void, P, P) bool {
    return struct {
        fn less(_: void, a: P, b: P) bool {
            return a < b;
        }
    }.less;
}

/// Sort `idx` (the identity) by one column with an LSD radix sort when the column has a radix
/// key; false when it has none (the caller uses the comparison sort).
fn radixSort(allocator: std.mem.Allocator, col: *const Column, idx: []u32) KernelError!bool {
    return withRadixKey(col, struct {
        allocator: std.mem.Allocator,
        col: *const Column,
        idx: []u32,

        fn run(self: @This(), comptime K: type, comptime keyFn: fn (*const Column, usize) K, comptime float: bool) KernelError!void {
            return radixWith(K, float, self.allocator, self.col, self.idx, keyFn);
        }
    }{ .allocator = allocator, .col = col, .idx = idx });
}

fn radixWith(comptime K: type, comptime float: bool, allocator: std.mem.Allocator, col: *const Column, idx: []u32, comptime keyFn: fn (*const Column, usize) K) KernelError!void {
    var split = try Split.init(allocator, col, float, idx.len);
    defer split.deinit(allocator);
    const m = split.vals.items.len;
    const keys = allocator.alloc(K, m) catch return error.OutOfMemory;
    defer allocator.free(keys);
    const desc = col.order == .descending;
    var diff: K = 0;
    for (keys, split.vals.items) |*k, i| {
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
    var src_i = split.vals.items;
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
    split.assemble(col.placement, src_i, idx);
}

// ------------------------------------------------------------------------------ selection

/// Reorder `items` so that `items[nth]` is the element a sort would put there, every element
/// before it is not greater and every one after it not less (nth_element): quickselect with a
/// median-of-three pivot and Hoare partitioning, O(n) on average; past 2 log2(n) rounds the
/// remaining range is sorted instead, so the worst case stays O(n log n). `lessThan` must be a
/// strict total order (ties broken, as by row index).
fn nthElement(comptime T: type, items: []T, nth: usize, context: anytype, comptime lessThan: fn (@TypeOf(context), T, T) bool) void {
    std.debug.assert(nth < items.len);
    var lo: usize = 0;
    var hi: usize = items.len;
    var budget: usize = 2 * (std.math.log2_int(usize, items.len) + 1);
    while (hi - lo > 16) {
        if (budget == 0) return std.sort.pdq(T, items[lo..hi], context, lessThan);
        budget -= 1;
        const mid = lo + (hi - lo) / 2;
        // Median of three at `mid`, the smallest at `lo` and the largest at `hi - 1`: both scans
        // below stop inside the range.
        if (lessThan(context, items[mid], items[lo])) std.mem.swap(T, &items[mid], &items[lo]);
        if (lessThan(context, items[hi - 1], items[mid])) std.mem.swap(T, &items[hi - 1], &items[mid]);
        if (lessThan(context, items[mid], items[lo])) std.mem.swap(T, &items[mid], &items[lo]);
        const pivot = items[mid];
        var i = lo;
        var j = hi - 1;
        while (true) {
            while (lessThan(context, items[i], pivot)) i += 1;
            while (lessThan(context, pivot, items[j])) j -= 1;
            if (i >= j) break;
            std.mem.swap(T, &items[i], &items[j]);
            i += 1;
            j -= 1;
        }
        // [lo, j] holds no element greater than the pivot, (j, hi) none smaller.
        if (nth <= j) hi = j + 1 else lo = j + 1;
    }
    std.sort.insertion(T, items[lo..hi], context, lessThan);
}

/// Keep the `heap.len` smallest items offered (`offer`) in a max-heap of capacity `heap.len`.
fn BoundedHeap(comptime T: type, comptime Context: type, comptime lessThan: fn (Context, T, T) bool) type {
    return struct {
        heap: []T,
        len: usize = 0,
        context: Context,

        const Self = @This();

        inline fn offer(self: *Self, item: T) void {
            if (self.len < self.heap.len) {
                // Sift up.
                var i = self.len;
                self.len += 1;
                while (i > 0) {
                    const parent = (i - 1) / 2;
                    if (!lessThan(self.context, self.heap[parent], item)) break;
                    self.heap[i] = self.heap[parent];
                    i = parent;
                }
                self.heap[i] = item;
            } else if (self.heap.len > 0 and lessThan(self.context, item, self.heap[0])) {
                // Replace the largest and sift down.
                var i: usize = 0;
                while (true) {
                    var child = 2 * i + 1;
                    if (child >= self.len) break;
                    if (child + 1 < self.len and lessThan(self.context, self.heap[child], self.heap[child + 1])) child += 1;
                    if (!lessThan(self.context, item, self.heap[child])) break;
                    self.heap[i] = self.heap[child];
                    i = child;
                }
                self.heap[i] = item;
            }
        }

        /// The kept items, smallest first.
        fn sorted(self: *Self) []T {
            std.sort.pdq(T, self.heap[0..self.len], self.context, lessThan);
            return self.heap[0..self.len];
        }
    };
}

/// partition_nth_indices of one radix column: the value rows packed with their row into distinct
/// integers (one pass that also sets the NaN and null rows aside), the pivot value found by an
/// MSD radix selection, then one branch-free pass that puts every smaller value row before it and
/// every larger one after it. False when the column has no radix key.
fn partitionRadix(allocator: std.mem.Allocator, col: *const Column, idx: []u32, pivot: usize) KernelError!bool {
    return withRadixKey(col, struct {
        allocator: std.mem.Allocator,
        col: *const Column,
        idx: []u32,
        pivot: usize,

        fn run(self: @This(), comptime K: type, comptime keyFn: fn (*const Column, usize) K, comptime float: bool) KernelError!void {
            const P = Packed(K);
            const n = self.idx.len;
            var split: Split = .{};
            defer split.deinit(self.allocator);
            const keys = self.allocator.alloc(P, n) catch return error.OutOfMemory;
            defer self.allocator.free(keys);
            const data = self.col.arr.data();
            const bits = dense.validityBits(data);
            const desc = self.col.order == .descending;
            var m: usize = 0;
            for (0..n) |i| {
                const row: u32 = @intCast(i);
                if (bits) |b| if (!bitmap.bitIsSet(b, data.offset + i)) {
                    split.nulls.append(self.allocator, row) catch return error.OutOfMemory;
                    continue;
                };
                if (float and isNanAt(self.col, i)) {
                    split.nans.append(self.allocator, row) catch return error.OutOfMemory;
                    continue;
                }
                keys[m] = pack(K, keyFn(self.col, i), desc, row);
                m += 1;
            }
            const start = split.valueStart(self.col.placement);
            const values = self.idx[start..][0..m];
            if (self.pivot >= start and self.pivot - start < m) {
                const nth = self.pivot - start;
                const at = try radixNth(K, self.allocator, keys[0..m], nth);
                var lo: usize = 0;
                var hi: usize = m - 1;
                for (keys[0..m]) |k| {
                    // Branch free: the row goes to both ends, the end it belongs to advances.
                    values[lo] = @truncate(k);
                    values[hi] = @truncate(k);
                    lo += @intFromBool(k < at);
                    hi -= @intFromBool(k > at);
                }
                values[nth] = @truncate(at);
            } else for (values, keys[0..m]) |*v, k| {
                v.* = @truncate(k);
            }
            const tail = if (self.col.placement == .at_start)
                [2][]const u32{ split.nulls.items, split.nans.items }
            else
                [2][]const u32{ split.nans.items, split.nulls.items };
            var pos: usize = if (self.col.placement == .at_start) 0 else m;
            for (tail) |part| {
                @memcpy(self.idx[pos..][0..part.len], part);
                pos += part.len;
            }
        }
    }{ .allocator = allocator, .col = col, .idx = idx, .pivot = pivot });
}

/// The `nth` smallest of `keys` (distinct packed keys of radix key type K): an MSD radix
/// selection, one 256-bucket histogram per byte over the candidates that share the bytes above
/// (a byte every candidate shares costs one pass and no copy), O(n) for any input.
fn radixNth(comptime K: type, allocator: std.mem.Allocator, keys: []const Packed(K), nth: usize) KernelError!Packed(K) {
    const P = Packed(K);
    var cand: []const P = keys;
    var scratch: []P = &.{};
    defer allocator.free(scratch);
    var rank = nth;
    var shift: usize = @bitSizeOf(K) + 32 - 8;
    while (cand.len > 32) : (shift -= 8) {
        var counts: [256]usize = undefined;
        @memset(&counts, 0);
        for (cand) |k| counts[@as(u8, @truncate(k >> @intCast(shift)))] += 1;
        var b: usize = 0;
        while (rank >= counts[b]) : (b += 1) rank -= counts[b];
        if (counts[b] != cand.len) {
            // Keep the candidates of bucket `b` (in place once they live in `scratch`).
            if (scratch.len == 0) scratch = allocator.alloc(P, counts[b]) catch return error.OutOfMemory;
            var w: usize = 0;
            for (cand) |k| if (@as(u8, @truncate(k >> @intCast(shift))) == b) {
                scratch[w] = k;
                w += 1;
            };
            cand = scratch[0..w];
        }
        if (shift == 0) break;
    }
    var small: [32]P = undefined;
    @memcpy(small[0..cand.len], cand);
    std.sort.insertion(P, small[0..cand.len], {}, packedLess(P));
    return small[rank];
}

/// select_k_unstable of one radix column: one pass offering every value row (packed with its
/// row) to a bounded max-heap, the NaN and null rows kept in order up to `out.len`. False when
/// the column has no radix key.
fn selectRadix(allocator: std.mem.Allocator, col: *const Column, out: []u32) KernelError!bool {
    return withRadixKey(col, struct {
        allocator: std.mem.Allocator,
        col: *const Column,
        out: []u32,

        fn run(self: @This(), comptime K: type, comptime keyFn: fn (*const Column, usize) K, comptime float: bool) KernelError!void {
            const P = Packed(K);
            const k = self.out.len;
            var split: Split = .{};
            defer split.deinit(self.allocator);
            const storage = self.allocator.alloc(P, k) catch return error.OutOfMemory;
            defer self.allocator.free(storage);
            var heap: BoundedHeap(P, void, packedLess(P)) = .{ .heap = storage, .context = {} };
            const data = self.col.arr.data();
            const bits = dense.validityBits(data);
            const desc = self.col.order == .descending;
            for (0..data.length) |i| {
                const row: u32 = @intCast(i);
                if (bits) |b| if (!bitmap.bitIsSet(b, data.offset + i)) {
                    if (split.nulls.items.len < k) split.nulls.append(self.allocator, row) catch return error.OutOfMemory;
                    continue;
                };
                if (float and isNanAt(self.col, i)) {
                    if (split.nans.items.len < k) split.nans.append(self.allocator, row) catch return error.OutOfMemory;
                    continue;
                }
                heap.offer(pack(K, keyFn(self.col, i), desc, row));
            }
            const best = heap.sorted();
            const rows = self.allocator.alloc(u32, best.len) catch return error.OutOfMemory;
            defer self.allocator.free(rows);
            for (rows, best) |*r, p| r.* = @truncate(p);
            split.assemble(self.col.placement, rows, self.out);
        }
    }{ .allocator = allocator, .col = col, .out = out });
}

// ------------------------------------------------------------------------------ kernels

/// One sort column: an array or chunked datum (borrowed) with its order and null placement.
const Key = struct {
    datum: Datum,
    order: SortOrder = .ascending,
    placement: SortNullPlacement = .at_end,
};

/// The sort columns of a call: the array / chunked arguments (one `sort_keys` entry each, or
/// `order` for all when there are none), or the columns of one record batch / table argument
/// that `sort_keys` names (at least one key; Arrow: "Must specify one or more sort keys"; a name
/// matching no field or several is `InvalidOptions`, as Arrow's FieldRef).
/// `placement`, when set, overrides every key's null placement. The caller frees the slice.
fn keysOf(allocator: std.mem.Allocator, args: []const Datum, sort_keys: []const core.SortKey, order: SortOrder, placement: ?SortNullPlacement) KernelError![]Key {
    if (args.len == 1 and args[0].isTabular()) {
        if (sort_keys.len == 0) return error.InvalidOptions;
        const keys = allocator.alloc(Key, sort_keys.len) catch return error.OutOfMemory;
        errdefer allocator.free(keys);
        const fields = switch (args[0]) {
            .record_batch => |b| b.get().schema().fields,
            .table => |t| t.get().schema().fields,
            else => unreachable,
        };
        for (sort_keys, keys) |sk, *k| {
            // The one field of that name (none, or several: Arrow's FieldRef error).
            var at: ?usize = null;
            for (fields, 0..) |f, i| if (std.mem.eql(u8, f.name, sk.name)) {
                if (at != null) return error.InvalidOptions;
                at = i;
            };
            const i = at orelse return error.InvalidOptions;
            const datum: Datum = switch (args[0]) {
                .record_batch => |b| .{ .array = b.get().columns[i] },
                .table => |t| .{ .chunked = t.get().columns[i] },
                else => unreachable,
            };
            k.* = .{ .datum = datum, .order = sk.order, .placement = placement orelse sk.null_placement };
        }
        return keys;
    }
    if (sort_keys.len != 0 and sort_keys.len != args.len) return error.InvalidOptions;
    const keys = allocator.alloc(Key, args.len) catch return error.OutOfMemory;
    for (args, keys, 0..) |a, *k, i| {
        if (a != .array and a != .chunked) {
            allocator.free(keys);
            return error.InvalidInput;
        }
        k.* = if (sort_keys.len == 0)
            .{ .datum = a, .order = order, .placement = placement orelse .at_end }
        else
            .{ .datum = a, .order = sort_keys[i].order, .placement = placement orelse sort_keys[i].null_placement };
    }
    return keys;
}

/// A uint64 array of the indices `idx` (the first `n`).
fn indexArray(allocator: std.mem.Allocator, idx: []const u32) KernelError!ArrayRef {
    var buf = OwnedBuffer.initUninit(allocator, idx.len * 8) catch return error.OutOfMemory;
    for (buf.typedSlice(u64)[0..idx.len], idx) |*d, i| d.* = i;
    return dense.finishFixed(allocator, uint64_type, idx.len, &buf, null);
}

fn sortIndicesExec(ctx: *ExecContext, args: []const Datum, options: Options) KernelError!Datum {
    const o: core.SortOptions = if (options == .sort) options.sort else .{};
    const keys = try keysOf(ctx.allocator, args, o.sort_keys, o.order, o.null_placement);
    defer ctx.allocator.free(keys);
    var cols = try Columns.init(ctx, keys);
    defer cols.deinit();
    const idx = try cols.sort();
    defer ctx.allocator.free(idx);
    return .{ .array = try indexArray(ctx.allocator, idx) };
}

fn partitionExec(ctx: *ExecContext, args: []const Datum, options: Options) KernelError!Datum {
    const o = options.partition_nth;
    var cols = try Columns.init(ctx, &.{.{ .datum = args[0], .placement = o.null_placement }});
    defer cols.deinit();
    if (o.pivot > cols.len) return error.InvalidOptions;
    const idx = ctx.allocator.alloc(u32, cols.len) catch return error.OutOfMemory;
    defer ctx.allocator.free(idx);
    if (!(cols.cols.len == 1 and try partitionRadix(ctx.allocator, &cols.cols[0], idx, o.pivot))) {
        for (idx, 0..) |*d, i| d.* = @intCast(i);
        if (o.pivot < idx.len) nthElement(u32, idx, o.pivot, &cols, Columns.lessThan);
    }
    return .{ .array = try indexArray(ctx.allocator, idx) };
}

fn selectKExec(ctx: *ExecContext, args: []const Datum, options: Options) KernelError!Datum {
    const o = options.select_k;
    const keys = try keysOf(ctx.allocator, args, o.sort_keys, .ascending, null);
    defer ctx.allocator.free(keys);
    var cols = try Columns.init(ctx, keys);
    defer cols.deinit();
    const out = ctx.allocator.alloc(u32, @min(o.k, cols.len)) catch return error.OutOfMemory;
    defer ctx.allocator.free(out);
    if (!(cols.cols.len == 1 and try selectRadix(ctx.allocator, &cols.cols[0], out))) {
        var heap: BoundedHeap(u32, *const Columns, Columns.lessThan) = .{ .heap = out, .context = &cols };
        for (0..cols.len) |i| heap.offer(@intCast(i));
        _ = heap.sorted();
    }
    return .{ .array = try indexArray(ctx.allocator, out) };
}

fn rankExec(ctx: *ExecContext, args: []const Datum, options: Options) KernelError!Datum {
    const o: core.RankOptions = if (options == .rank) options.rank else .{};
    var cols = try Columns.init(ctx, &.{.{ .datum = args[0], .order = o.order, .placement = o.null_placement }});
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

/// rank_quantile (`normal` false) and rank_normal: per row (rows sorting before it + half the
/// rows equal to it, itself included) / all rows, nulls and NaN counted as values that tie with
/// each other (Arrow); rank_normal maps it through the standard normal quantile function.
fn RankQuantile(comptime normal: bool) type {
    return struct {
        fn exec(ctx: *ExecContext, args: []const Datum, options: Options) KernelError!Datum {
            const o: core.RankQuantileOptions = if (options == .rank_quantile) options.rank_quantile else .{};
            var cols = try Columns.init(ctx, &.{.{ .datum = args[0], .order = o.order, .placement = o.null_placement }});
            defer cols.deinit();
            const idx = try cols.sort();
            defer ctx.allocator.free(idx);
            const n = idx.len;
            var buf = OwnedBuffer.initUninit(ctx.allocator, n * 8) catch return error.OutOfMemory;
            const out = buf.typedSlice(f64)[0..n];
            const total: f64 = @floatFromInt(n);
            var start: usize = 0;
            while (start < n) {
                var end = start + 1;
                while (end < n and cols.compare(idx[start], idx[end]) == .eq) end += 1;
                const less: f64 = @floatFromInt(start);
                const equal: f64 = @floatFromInt(end - start);
                const q = (less + 0.5 * equal) / total;
                const v = if (normal) normalQuantile(q) else q;
                for (idx[start..end]) |row| out[row] = v;
                start = end;
            }
            return .{ .array = try dense.finishFixed(ctx.allocator, float64_type, n, &buf, null) };
        }
    };
}

/// The standard normal quantile function (inverse CDF) of `p` in (0, 1): Wichura's algorithm
/// AS 241 (PPND16), about 1e-16 relative accuracy.
fn normalQuantile(p: f64) f64 {
    const q = p - 0.5;
    if (@abs(q) <= 0.425) {
        const r = 0.180625 - q * q;
        return q * (((((((r * 2509.0809287301226727 + 33430.575583588128105) * r + 67265.770927008700853) * r + 45921.953931549871457) * r + 13731.693765509461125) * r + 1971.5909503065514427) * r + 133.14166789178437745) * r + 3.387132872796366608) /
            (((((((r * 5226.495278852545925 + 28729.085735721942674) * r + 39307.89580009271061) * r + 21213.794301586595867) * r + 5394.1960214247511077) * r + 687.1870074920579083) * r + 42.313330701600911252) * r + 1.0);
    }
    var r = @sqrt(-@log(if (q < 0) p else 1 - p));
    var v: f64 = undefined;
    if (r <= 5) {
        r -= 1.6;
        v = (((((((r * 7.7454501427834140764e-4 + 0.0227238449892691845833) * r + 0.24178072517745061177) * r + 1.27045825245236838258) * r + 3.64784832476320460504) * r + 5.7694972214606914055) * r + 4.6303378461565452959) * r + 1.42343711074968357734) /
            (((((((r * 1.05075007164441684324e-9 + 5.475938084995344946e-4) * r + 0.0151986665636164571966) * r + 0.14810397642748007459) * r + 0.68976733498510000455) * r + 1.6763848301838038494) * r + 2.05319162663775882187) * r + 1.0);
    } else {
        r -= 5;
        v = (((((((r * 2.01033439929228813265e-7 + 2.71155556874348757815e-5) * r + 0.0012426609473880784386) * r + 0.026532189526576123093) * r + 0.29656057182850489123) * r + 1.7848265399172913358) * r + 5.4637849111641143699) * r + 6.6579046435011037772) /
            (((((((r * 2.04426310338993978564e-15 + 1.4215117583164458887e-7) * r + 1.8463183175100546818e-5) * r + 7.868691311456132591e-4) * r + 0.0148753612908506148525) * r + 0.13692988092273580531) * r + 0.59983220655588793769) * r + 1.0);
    }
    return if (q < 0) -v else v;
}

/// Compare row `i` of `a` with row `j` of `b` (columns of one type, ascending, neither row
/// null): values before NaN, -0.0 equal to 0.0.
fn crossCompare(a: *const Column, i: usize, b: *const Column, j: usize) std.math.Order {
    const ca = a.class(i);
    const cb = b.class(j);
    if (ca != cb) return std.math.order(@intFromEnum(ca), @intFromEnum(cb));
    if (ca != .value) return .eq;
    return valueOrder(&a.values, i, &b.values, j);
}

/// Compare row `i` of `a` with row `j` of `b` (rows of one type, neither null; floats by `<`).
pub fn valueOrder(a: *const Values, i: usize, b: *const Values, j: usize) std.math.Order {
    return switch (a.*) {
        .boolean => |x| std.math.order(@intFromBool(bitmap.bitIsSet(x.bits, x.offset + i)), @intFromBool(bitmap.bitIsSet(b.boolean.bits, b.boolean.offset + j))),
        .bytes => |op| switch (op) {
            inline else => |ra| switch (b.bytes) {
                inline else => |rb| std.mem.order(u8, ra.get(i), rb.get(j)),
            },
        },
        .fixed => |f| std.mem.order(u8, f.data[(f.offset + i) * f.width ..][0..f.width], b.fixed.data[(b.fixed.offset + j) * f.width ..][0..f.width]),
        inline else => |v, tag| std.math.order(v[i], @field(b.*, @tagName(tag))[j]),
    };
}

fn searchSortedArgs(args: []const Datum) bool {
    return args.len == 2 and args[0] != .scalar and args[0].dataType() != .struct_ and isSortable(args[0].dataType()) and args[0].dataType().eql(args[1].dataType());
}

fn searchSortedOptions(options: Options) bool {
    return options == .none or options == .search_sorted;
}

/// search_sorted(values, needles): per needle the insertion point in `values` (ascending, its
/// nulls all at the start or all at the end, else `InvalidInput`) that keeps it sorted, found
/// by binary search among the non-null values: `left` the first such position, `right` the last.
/// A null needle gives null; a scalar needle a scalar. uint64 positions in [0, len].
fn searchSortedExec(ctx: *ExecContext, args: []const Datum, options: Options) KernelError!Datum {
    if (args.len != 2) return error.InvalidArity;
    const right = options == .search_sorted and options.search_sorted.side == .right;
    var values = try Column.init(ctx, .{ .datum = args[0] });
    defer values.deinit();
    var needle_arr = switch (args[1]) {
        .scalar => |s| try dense.chunkArray(ctx.allocator, .{ .scalar = s }),
        .array, .chunked => try asArray(ctx, args[1]),
        .record_batch, .table => return error.InvalidInput,
    };
    defer needle_arr.release();
    var needles = try Column.init(ctx, .{ .datum = .{ .array = needle_arr } });
    defer needles.deinit();
    if (std.meta.activeTag(values.values) != std.meta.activeTag(needles.values)) return error.InvalidInput;
    if (values.values == .fixed and values.values.fixed.width != needles.values.fixed.width) return error.InvalidInput;

    const vdata = values.arr.data();
    const n = vdata.length;
    // Trailing nulls first: values that are all null count as nulls at the end (Arrow).
    var trail: usize = 0;
    while (trail < n and vdata.isNull(n - 1 - trail)) trail += 1;
    var lead: usize = 0;
    while (lead < n - trail and vdata.isNull(lead)) lead += 1;
    for (lead..n - trail) |i| if (vdata.isNull(i)) return error.InvalidInput;

    const ndata = needles.arr.data();
    const m = ndata.length;
    var buf = OwnedBuffer.initUninit(ctx.allocator, m * 8) catch return error.OutOfMemory;
    var validity: ?OwnedBuffer = null;
    if (ndata.hasNulls()) {
        var v = dense.bitsBuffer(ctx.allocator, m) catch |err| {
            buf.deinit();
            return err;
        };
        @memset(v.data[0..bitmap.byteLength(m)], 0xFF);
        validity = v;
    }
    for (buf.typedSlice(u64)[0..m], 0..) |*d, j| {
        if (ndata.isNull(j)) {
            bitmap.clearBit(validity.?.data, j);
            d.* = 0;
            continue;
        }
        // First position in [lead, n - trail) whose value is not before (left) / after (right)
        // the needle.
        var lo = lead;
        var hi = n - trail;
        while (lo < hi) {
            const mid = lo + (hi - lo) / 2;
            const o = crossCompare(&values, mid, &needles, j);
            if (o == .lt or (right and o == .eq)) lo = mid + 1 else hi = mid;
        }
        d.* = lo;
    }
    var out = try dense.finishFixed(ctx.allocator, uint64_type, m, &buf, validity);
    if (args[1] != .scalar) return .{ .array = out };
    defer out.release();
    return .{ .scalar = try dense.scalarAt0(out) };
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
