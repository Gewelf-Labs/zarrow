//! Grouped aggregation (Arrow C++ "Grouped aggregations", the `hash_*` functions). A `Grouper`
//! maps the rows of one or more key columns to dense uint32 group ids (first-seen order; a null
//! key is a group of its own, as in Arrow); the `hash_*` kernels (`FunctionKind.hash_aggregate`)
//! take a value array and the group id array and return one row per group (as many groups as the
//! largest id plus one); `groupBy` runs both and returns a record batch of the key columns
//! followed by one column per aggregate.
//!
//! Functions: hash_sum, hash_mean, hash_min, hash_max, hash_min_max (int8..int64, uint8..uint64,
//! float32/64, the output types and NaN / overflow rules of sum / mean / min / max / min_max;
//! hash_sum, hash_mean, hash_min and hash_max also on decimal128 / decimal256 with the decimal
//! rules of aggregate_basic.zig); hash_count for every type with a validity bitmap; hash_count_all
//! (rows per group; one argument, the group ids, like Arrow's nullary hash_count_all); hash_any,
//! hash_all (bool); hash_count_distinct and hash_distinct (`list<T>`, first-seen order) for every
//! key type.
//!
//! Options, as Arrow's hash kernels (`none` gives the defaults): `ScalarAggregateOptions` for
//! hash_sum, hash_mean, hash_min, hash_max, hash_min_max, hash_any and hash_all (a group with a
//! null row and `skip_nulls` false, or fewer valid rows than `min_count` (not for the min / max
//! kernels, as Arrow), is null; with `min_count` 0 an empty group sums to 0 and its float mean
//! is NaN, min / max stay null; hash_any / hash_all without `skip_nulls` use Kleene logic) and `CountOptions` for hash_count,
//! hash_count_distinct and hash_distinct (`only_valid` counts / lists the non-null values,
//! `only_null` the nulls, `all` both; a null is one distinct value). `groupBy` hands each
//! `Aggregate.options` to its function.
//! Each kernel has an `AggregateLifecycle`: `update` takes (values, group ids) batches whose ids
//! may grow, `merge` adds another state over the same group ids, `finalize` returns the groups
//! seen so far. A one-shot call (`invoke`) requires every group id to be below its number of rows
//! (dense ids from a `Grouper` always are), so its state is bounded by the input size; a
//! lifecycle `update` accepts any id (its state grows to the largest id).
//!
//! Key types (Grouper, KeyMap, the distinct functions and set_lookup.zig): int8..int64,
//! uint8..uint64, float32/64, bool, date32, date64, timestamp, time32, time64, duration and the
//! string / binary layouts. Floats compare by their bits, as pyarrow 26 hashes them (NaN
//! with another sign or payload is another value, -0.0 differs from 0.0).
const std = @import("std");
const bitmap = @import("../bitmap.zig");
const core = @import("core.zig");
const datatype = @import("../datatype.zig");
const aggregate_basic = @import("aggregate_basic.zig");
const decimal = @import("decimal.zig");
const dense = @import("dense.zig");
const is_null = @import("is_null.zig");
const concat = @import("../concat_array_refs.zig");
const RecordBatch = @import("../record_batch.zig").RecordBatch;
const schema_mod = @import("../schema.zig");
const sort = @import("sort.zig");

const ArrayData = @import("../array/array_data.zig").ArrayData;
const PrimitiveArray = @import("../array/primitive_array.zig").PrimitiveArray;
const OwnedBuffer = @import("../buffer.zig").OwnedBuffer;
const SharedBuffer = @import("../buffer.zig").SharedBuffer;
const ArrayRef = core.ArrayRef;
const DataType = core.DataType;
const Datum = core.Datum;
const ExecChunkValue = core.ExecChunkValue;
const ExecContext = core.ExecContext;
const FunctionRegistry = core.FunctionRegistry;
const KernelError = core.KernelError;
const Options = core.Options;
const TypeCheckFn = core.TypeCheckFn;

const uint32_type = DataType{ .uint32 = {} };
const int64_type = DataType{ .int64 = {} };
const bool_type = DataType{ .bool = {} };

// ------------------------------------------------------------------------------ key maps

/// Open-addressing table from the bits of a fixed-width key to its id (linear probing, at most
/// half full, Fibonacci hashing of the key folded on itself): a lookup is one multiply and
/// usually one compare, several times cheaper than `std.HashMap` on these keys.
// ponytail: unkeyed hash, so crafted keys can degrade probing to linear scans (slow, not unsafe);
// seed the multiplier per Grouper if adversarial group-bys matter.
const FixedTable = struct {
    keys: []u64 = &.{},
    /// `missing` marks an empty slot.
    ids: []u32 = &.{},
    len: usize = 0,
    shift: u6 = 63,

    fn deinit(self: *FixedTable, allocator: std.mem.Allocator) void {
        allocator.free(self.keys);
        allocator.free(self.ids);
    }

    inline fn slot(self: *const FixedTable, k: u64) usize {
        return @intCast(((k ^ (k >> 32)) *% 0x9E37_79B9_7F4A_7C15) >> self.shift);
    }

    fn get(self: *const FixedTable, k: u64) ?u32 {
        if (self.ids.len == 0) return null;
        const mask = self.ids.len - 1;
        var i = self.slot(k);
        while (true) : (i = (i + 1) & mask) {
            const id = self.ids[i];
            if (id == missing) return null;
            if (self.keys[i] == k) return id;
        }
    }

    /// The id of `k`, inserting `next_id` for a new key (`added` tells which happened).
    fn getOrPut(self: *FixedTable, allocator: std.mem.Allocator, k: u64, next_id: u32, added: *bool) error{OutOfMemory}!u32 {
        if ((self.len + 1) * 2 > self.ids.len) try self.grow(allocator);
        const mask = self.ids.len - 1;
        var i = self.slot(k);
        while (true) : (i = (i + 1) & mask) {
            const id = self.ids[i];
            if (id == missing) break;
            if (self.keys[i] == k) {
                added.* = false;
                return id;
            }
        }
        self.keys[i] = k;
        self.ids[i] = next_id;
        self.len += 1;
        added.* = true;
        return next_id;
    }

    fn grow(self: *FixedTable, allocator: std.mem.Allocator) error{OutOfMemory}!void {
        const cap = @max(16, self.ids.len * 2);
        const keys = try allocator.alloc(u64, cap);
        errdefer allocator.free(keys);
        const ids = try allocator.alloc(u32, cap);
        @memset(ids, missing);
        var bigger: FixedTable = .{ .keys = keys, .ids = ids, .len = self.len, .shift = @intCast(@as(usize, 64) - std.math.log2_int(usize, cap)) };
        for (self.ids, self.keys) |id, k| {
            if (id == missing) continue;
            var i = bigger.slot(k);
            while (bigger.ids[i] != missing) i = (i + 1) & (cap - 1);
            bigger.keys[i] = k;
            bigger.ids[i] = id;
        }
        self.deinit(allocator);
        self.* = bigger;
    }
};

/// How the rows of a key type are read (`fixed_bytes`: the value bytes of a wider fixed-width
/// type, hashed as a byte string).
const KeyKind = enum { b8, b16, b32, b64, f16, f32, f64, boolean, bytes, fixed_bytes };

fn keyKind(dt: DataType) ?KeyKind {
    return switch (dt) {
        .int8, .uint8 => .b8,
        .int16, .uint16 => .b16,
        .int32, .uint32, .date32, .time32, .decimal32, .interval_months => .b32,
        .int64, .uint64, .date64, .timestamp, .time64, .duration, .decimal64, .interval_day_time => .b64,
        .half_float => .f16,
        .float => .f32,
        .double => .f64,
        .bool => .boolean,
        .string, .large_string, .string_view, .binary, .large_binary, .binary_view => .bytes,
        .decimal128, .decimal256, .interval_month_day_nano => .fixed_bytes,
        .fixed_size_binary => |f| if (f.byte_width > 0) .fixed_bytes else null,
        else => null,
    };
}

/// Return whether `dt` can be a key (Grouper, KeyMap, is_in / index_in, hash_count_distinct).
pub fn isKeyType(dt: DataType) bool {
    return keyKind(dt) != null;
}

/// A key type, or a dictionary of one: the `Grouper` (group-by keys, unique, value_counts,
/// hash_count_distinct, hash_distinct) groups a dictionary by its index (a null index and an
/// index to a null entry are two groups, as Arrow) and keeps the dictionary.
pub fn isGroupKeyType(dt: DataType) bool {
    return switch (dt) {
        .dictionary => |d| isKeyType(d.value_type.*),
        else => isKeyType(dt),
    };
}

/// Check a dictionary array before its indices or dictionary are read: the dictionary passes
/// `validateLayout` and has the declared value type, and every non-null index is below its
/// length (`InvalidInput` otherwise; `validateLayout` of the array checks neither).
pub fn validateDictionary(data: *const ArrayData) KernelError!void {
    const d = data.data_type.dictionary;
    const dict = data.dictionary orelse return error.InvalidInput;
    dict.data().validateLayout() catch return error.InvalidInput;
    if (!dict.data().data_type.eql(d.value_type.*)) return error.InvalidInput;
    data.validateDictionaryIndices(d, data.offset, data.offset + data.length) catch return error.InvalidInput;
}

/// Whether two validated dictionaries of a key type hold the same values in the same order
/// (floats by bits, as the key hashing).
pub fn sameDictionary(allocator: std.mem.Allocator, a: ArrayRef, b: ArrayRef) KernelError!bool {
    if (a.node == b.node) return true;
    const n = a.data().length;
    if (n != b.data().length or !a.data().data_type.eql(b.data().data_type)) return false;
    // Equal arrays get equal first-seen ids, row by row.
    var map: KeyMap = .{};
    defer map.deinit(allocator);
    const ids = allocator.alloc(u32, 2 * n) catch return error.OutOfMemory;
    defer allocator.free(ids);
    try map.map(allocator, a, ids[0..n], .insert);
    try map.map(allocator, b, ids[n..], .insert);
    return std.mem.eql(u32, ids[0..n], ids[n..]);
}

/// Id that `KeyMap.map` gives to a row it does not find (lookup mode).
pub const missing: u32 = std.math.maxInt(u32);

/// Dense ids (first-seen order) for the distinct values of one key column; null rows share one
/// id. Byte keys are borrowed from the arrays passed to `map`, which must outlive the map.
pub const KeyMap = struct {
    fixed: FixedTable = .{},
    /// Direct id table over a narrow range of fixed-width keys (`freeze`), for lookups.
    dense: ?struct { min: u64, ids: []u32 } = null,
    bytes: std.StringHashMapUnmanaged(u32) = .empty,
    null_id: ?u32 = null,
    count: u32 = 0,
    /// Compare floats by value with every NaN one value (mode, index: pyarrow 26 folds there).
    fold_floats: bool = false,

    pub const Mode = enum { insert, lookup };

    pub fn deinit(self: *KeyMap, allocator: std.mem.Allocator) void {
        self.fixed.deinit(allocator);
        self.bytes.deinit(allocator);
        if (self.dense) |d| allocator.free(d.ids);
    }

    /// Prepare for lookups only: when the fixed-width keys span fewer than 2^16 values (and at
    /// most 64 per key), index them directly instead of hashing. No `insert` after this.
    pub fn freeze(self: *KeyMap, allocator: std.mem.Allocator) KernelError!void {
        if (self.fixed.len == 0) return;
        var lo: u64 = std.math.maxInt(u64);
        var hi: u64 = 0;
        for (self.fixed.ids, self.fixed.keys) |id, k| if (id != missing) {
            lo = @min(lo, k);
            hi = @max(hi, k);
        };
        const span = hi - lo;
        if (span >= 1 << 16 or span > 64 * self.fixed.len) return;
        const ids = allocator.alloc(u32, @intCast(span + 1)) catch return error.OutOfMemory;
        @memset(ids, missing);
        for (self.fixed.ids, self.fixed.keys) |id, k| if (id != missing) {
            ids[@intCast(k - lo)] = id;
        };
        self.dense = .{ .min = lo, .ids = ids };
    }

    inline fn lookupFixed(self: *const KeyMap, key: u64) u32 {
        if (self.dense) |d| {
            const off = key -% d.min;
            return if (off < d.ids.len) d.ids[@intCast(off)] else missing;
        }
        return self.fixed.get(key) orelse missing;
    }

    /// Set `ids[i]` to the id of row `i` of `arr` (a key type that passed `validateLayout`):
    /// `insert` gives a new value the next id, `lookup` gives it `missing`.
    pub fn map(self: *KeyMap, allocator: std.mem.Allocator, arr: ArrayRef, ids: []u32, comptime mode: Mode) KernelError!void {
        const data = arr.data();
        if (ids.len != data.length) return error.InvalidInput;
        const bits = dense.validityBits(data);
        // A dictionary maps its indices (the caller checked them, see `validateDictionary`).
        const key_type = if (data.data_type == .dictionary) dense.indexType(data.data_type.dictionary) else data.data_type;
        switch (keyKind(key_type) orelse return error.UnsupportedType) {
            inline .b8, .b16, .b32, .b64, .f16, .f32, .f64 => |k| {
                const U = switch (k) {
                    .b8 => u8,
                    .b16, .f16 => u16,
                    .b32, .f32 => u32,
                    else => u64,
                };
                const vals = (PrimitiveArray(U){ .data = data }).values() catch return error.InvalidInput;
                // Lookups in a frozen narrow key range read the direct table: no hashing, and the
                // table lives in locals rather than behind `self`.
                if (mode == .lookup and k != .f16 and k != .f32 and k != .f64) if (self.dense) |d| {
                    const null_id = self.null_id orelse missing;
                    for (vals, ids, 0..) |v, *id, i| {
                        const off = @as(u64, v) -% d.min;
                        id.* = if (off < d.ids.len) d.ids[@intCast(off)] else missing;
                        if (bits) |b| if (!bitmap.bitIsSet(b, data.offset + i)) {
                            id.* = null_id;
                        };
                    }
                    return;
                };
                for (vals, ids, 0..) |v, *id, i| {
                    if (bits) |b| if (!bitmap.bitIsSet(b, data.offset + i)) {
                        id.* = try self.nullId(mode);
                        continue;
                    };
                    // Floats by bits, as pyarrow 26 hashing (`count_distinct([0.0, -0.0, nan, -nan])` is 4).
                    const key: u64 = if (self.fold_floats) foldFloat(k, v) else v;
                    id.* = if (mode == .insert) try self.put(allocator, &self.fixed, key) else self.lookupFixed(key);
                }
            },
            .boolean => for (ids, 0..) |*id, i| {
                if (bits) |b| if (!bitmap.bitIsSet(b, data.offset + i)) {
                    id.* = try self.nullId(mode);
                    continue;
                };
                const key: u64 = @intFromBool(bitmap.bitIsSet(data.buffers[1].data, data.offset + i));
                id.* = if (mode == .insert) try self.put(allocator, &self.fixed, key) else self.lookupFixed(key);
            },
            .fixed_bytes => {
                // validateLayout bounds the values buffer by (offset + length) * width.
                const w = (data.data_type.bitWidth() orelse return error.UnsupportedType) / 8;
                const vals = data.buffers[1].data;
                if (vals.len / w < data.offset + data.length) return error.InvalidInput;
                for (ids, 0..) |*id, i| {
                    if (bits) |b| if (!bitmap.bitIsSet(b, data.offset + i)) {
                        id.* = try self.nullId(mode);
                        continue;
                    };
                    const s = vals[(data.offset + i) * w ..][0..w];
                    id.* = if (mode == .insert) try self.put(allocator, &self.bytes, s) else self.bytes.get(s) orelse missing;
                }
            },
            .bytes => switch (try dense.StringOperand.init(.{ .array = arr })) {
                inline else => |r| for (ids, 0..) |*id, i| {
                    if (bits) |b| if (!bitmap.bitIsSet(b, data.offset + i)) {
                        id.* = try self.nullId(mode);
                        continue;
                    };
                    const s = r.get(i);
                    id.* = if (mode == .insert) try self.put(allocator, &self.bytes, s) else self.bytes.get(s) orelse missing;
                },
            },
        }
    }

    fn nullId(self: *KeyMap, comptime mode: Mode) KernelError!u32 {
        if (self.null_id) |id| return id;
        if (mode == .lookup) return missing;
        const id = try self.next();
        self.null_id = id;
        return id;
    }

    fn next(self: *KeyMap) KernelError!u32 {
        if (self.count == missing) return error.Overflow;
        defer self.count += 1;
        return self.count;
    }

    fn put(self: *KeyMap, allocator: std.mem.Allocator, m: anytype, key: anytype) KernelError!u32 {
        if (self.count == missing) return error.Overflow;
        if (@TypeOf(m) == *FixedTable) {
            var added = false;
            const id = m.getOrPut(allocator, key, self.count, &added) catch return error.OutOfMemory;
            if (added) self.count += 1;
            return id;
        }
        const gop = m.getOrPut(allocator, key) catch return error.OutOfMemory;
        if (!gop.found_existing) {
            gop.value_ptr.* = self.count;
            self.count += 1;
        }
        return gop.value_ptr.*;
    }
};

// ------------------------------------------------------------------------------ grouper

/// Assigns dense group ids to rows of one or more key columns, batch after batch (Arrow
/// `Grouper`). The key arrays of every batch are retained until `deinit` (byte keys are borrowed
/// from them, and `uniques` takes the first row of each group from them).
// ponytail: keeps every input batch alive; copy the unique keys out if long streams matter.
pub const Grouper = struct {
    allocator: std.mem.Allocator,
    key_types: []DataType,
    columns: []KeyMap,
    /// Column `j + 1` combines (id so far, column id) pairs into the next id.
    levels: []KeyMap,
    batches: std.ArrayList([]ArrayRef) = .empty,
    /// First (batch, row) of every group, in group id order.
    firsts: std.ArrayList(struct { batch: usize, row: usize }) = .empty,

    /// A grouper for keys of `key_types` (at least one, each `isGroupKeyType`).
    pub fn init(allocator: std.mem.Allocator, key_types: []const DataType) KernelError!Grouper {
        if (key_types.len == 0) return error.InvalidArity;
        for (key_types) |t| if (!isGroupKeyType(t)) return error.UnsupportedType;
        const types = allocator.dupe(DataType, key_types) catch return error.OutOfMemory;
        errdefer allocator.free(types);
        const columns = allocator.alloc(KeyMap, key_types.len) catch return error.OutOfMemory;
        errdefer allocator.free(columns);
        const levels = allocator.alloc(KeyMap, key_types.len - 1) catch return error.OutOfMemory;
        for (columns) |*c| c.* = .{};
        for (levels) |*l| l.* = .{};
        return .{ .allocator = allocator, .key_types = types, .columns = columns, .levels = levels };
    }

    pub fn deinit(self: *Grouper) void {
        for (self.columns) |*c| c.deinit(self.allocator);
        for (self.levels) |*l| l.deinit(self.allocator);
        for (self.batches.items) |keys| {
            for (keys) |*k| k.release();
            self.allocator.free(keys);
        }
        self.batches.deinit(self.allocator);
        self.firsts.deinit(self.allocator);
        self.allocator.free(self.levels);
        self.allocator.free(self.columns);
        self.allocator.free(self.key_types);
    }

    /// Number of groups seen so far.
    pub fn numGroups(self: *const Grouper) usize {
        return self.firsts.items.len;
    }

    /// Return the group id (uint32 array, no nulls) of every row of `keys` (one array per key
    /// column, equal lengths, the types given to `init`). A dictionary key column must keep the
    /// dictionary of its first batch (equal values; Arrow: "Unifying differing dictionaries" is
    /// `NotImplemented`).
    pub fn consume(self: *Grouper, keys: []const ArrayRef) KernelError!ArrayRef {
        if (keys.len != self.columns.len) return error.InvalidArity;
        const len = keys[0].data().length;
        for (keys, self.key_types, 0..) |k, t, j| {
            if (!k.data().data_type.eql(t)) return error.InvalidInput;
            if (k.data().length != len) return error.InvalidInput;
            k.data().validateLayout() catch return error.InvalidInput;
            if (t != .dictionary) continue;
            try validateDictionary(k.data());
            if (self.batches.items.len > 0 and !try sameDictionary(self.allocator, self.batches.items[0][j].data().dictionary.?, k.data().dictionary.?))
                return error.NotImplemented;
        }
        const held = self.allocator.alloc(ArrayRef, keys.len) catch return error.OutOfMemory;
        for (keys, held) |k, *h| h.* = k.retain();
        self.batches.append(self.allocator, held) catch {
            for (held) |*h| h.release();
            self.allocator.free(held);
            return error.OutOfMemory;
        };
        const batch = self.batches.items.len - 1;

        var out = OwnedBuffer.initUninit(self.allocator, len * 4) catch return error.OutOfMemory;
        errdefer out.deinit();
        const ids = out.typedSlice(u32)[0..len];
        try self.columns[0].map(self.allocator, keys[0], ids, .insert);
        if (keys.len > 1) {
            const col = self.allocator.alloc(u32, len) catch return error.OutOfMemory;
            defer self.allocator.free(col);
            for (keys[1..], self.columns[1..], self.levels) |k, *m, *level| {
                try m.map(self.allocator, k, col, .insert);
                for (ids, col) |*id, c| id.* = try level.put(self.allocator, &level.fixed, (@as(u64, id.*) << 32) | c);
            }
        }
        // Ids are handed out in row order, so a new group is one whose id is the next one.
        for (ids, 0..) |id, row| if (id == self.firsts.items.len) {
            self.firsts.append(self.allocator, .{ .batch = batch, .row = row }) catch return error.OutOfMemory;
        };
        return dense.finishFixed(self.allocator, uint32_type, len, &out, null);
    }

    /// Return the key columns of the groups (one row per group, in group id order); the caller
    /// releases every array and frees the slice.
    pub fn uniques(self: *const Grouper) KernelError![]ArrayRef {
        const allocator = self.allocator;
        const n = self.firsts.items.len;
        const starts = allocator.alloc(usize, self.batches.items.len) catch return error.OutOfMemory;
        defer allocator.free(starts);
        var total: usize = 0;
        for (self.batches.items, starts) |keys, *s| {
            s.* = total;
            total += keys[0].data().length;
        }
        const indices = allocator.alloc(usize, n) catch return error.OutOfMemory;
        defer allocator.free(indices);
        for (self.firsts.items, indices) |f, *ix| ix.* = starts[f.batch] + f.row;

        const out = allocator.alloc(ArrayRef, self.columns.len) catch return error.OutOfMemory;
        var made: usize = 0;
        errdefer {
            for (out[0..made]) |*o| o.release();
            allocator.free(out);
        }
        const parts = allocator.alloc(ArrayRef, self.batches.items.len) catch return error.OutOfMemory;
        defer allocator.free(parts);
        for (self.key_types, 0..) |t, j| {
            if (parts.len == 0) {
                out[j] = try dense.nullArray(allocator, t, 0);
                made += 1;
                continue;
            }
            for (self.batches.items, parts) |keys, *p| p.* = keys[j];
            var column = if (parts.len == 1) parts[0].retain() else concat.concatArrayRefs(allocator, t, parts) catch |err| return switch (err) {
                error.OutOfMemory => error.OutOfMemory,
                else => error.InvalidInput,
            };
            defer column.release();
            out[j] = try takeArray(allocator, column, indices);
            made += 1;
        }
        return out;
    }
};

// ------------------------------------------------------------------------------ kernels

/// Register every `hash_*` function (kind `hash_aggregate`, uint32 group ids).
pub fn register(registry: *FunctionRegistry) KernelError!void {
    inline for (dense.numeric_types) |T| {
        const tag = comptime std.meta.activeTag(dense.dataTypeOf(T));
        try add(registry, "hash_sum", withIds(tag), aggregateOptions, Grouped(Sum(T, .sum)));
        try add(registry, "hash_mean", withIds(tag), aggregateOptions, Grouped(Sum(T, .mean)));
        try add(registry, "hash_min", withIds(tag), aggregateOptions, Grouped(MinMax(T, .min)));
        try add(registry, "hash_max", withIds(tag), aggregateOptions, Grouped(MinMax(T, .max)));
        try add(registry, "hash_min_max", withIds(tag), aggregateOptions, Grouped(MinMax(T, .min_max)));
    }
    inline for (.{ i128, i256 }) |T| {
        const check = comptime withIds(if (T == i128) .decimal128 else .decimal256);
        try add(registry, "hash_sum", check, aggregateOptions, Grouped(DecimalSum(T, .sum)));
        try add(registry, "hash_mean", check, aggregateOptions, Grouped(DecimalSum(T, .mean)));
        try add(registry, "hash_min", check, aggregateOptions, Grouped(DecimalMinMax(T, .min)));
        try add(registry, "hash_max", check, aggregateOptions, Grouped(DecimalMinMax(T, .max)));
    }
    try add(registry, "hash_count", countArgs, countOptions, Grouped(Count(.count)));
    try add(registry, "hash_count_all", idsOnly, dense.noOptions, Grouped(Count(.count_all)));
    try add(registry, "hash_any", withIds(.bool), aggregateOptions, Grouped(AnyAll(.any)));
    try add(registry, "hash_all", withIds(.bool), aggregateOptions, Grouped(AnyAll(.all)));
    try add(registry, "hash_count_distinct", distinctArgs, countOptions, Distinct(.count));
    try add(registry, "hash_distinct", distinctArgs, countOptions, Distinct(.list));
    try registerMore(registry);
}

/// `none` (Arrow defaults) or `ScalarAggregateOptions`.
fn aggregateOptions(options: Options) bool {
    return options == .none or options == .scalar_aggregate;
}

/// `none` (only_valid) or `CountOptions`.
fn countOptions(options: Options) bool {
    return options == .none or options == .count;
}

fn scalarOptions(options: Options) core.ScalarAggregateOptions {
    return if (options == .scalar_aggregate) options.scalar_aggregate else .{};
}

fn countMode(options: Options) core.CountMode {
    return if (options == .count) options.count.mode else .only_valid;
}

/// Whether the options make a group null: a null row with `skip_nulls` false, or fewer valid
/// rows than `min_count` (Arrow C++).
fn nullByOptions(o: core.ScalarAggregateOptions, count: u64, nulls: u64) bool {
    return (!o.skip_nulls and nulls > 0) or count < o.min_count;
}

pub fn add(registry: *FunctionRegistry, comptime name: []const u8, comptime check: TypeCheckFn, comptime options_check: core.OptionsCheckFn, comptime A: type) KernelError!void {
    const alloc = comptime aggregate_basic.hasFn(A, "resultTypeAlloc");
    try registry.registerKernel(name, .hash_aggregate, .{
        .signature = .{
            .arity = if (A.ids_only) 1 else 2,
            .type_check = check,
            .options_check = options_check,
            .result_type_fn = if (alloc) null else A.resultType,
            .result_type_alloc_fn = if (alloc) A.resultTypeAlloc else null,
        },
        .exec = A.exec,
        .aggregate_lifecycle = .{ .init = A.init, .update = A.update, .merge = A.merge, .finalize = A.finalize, .deinit = A.deinit },
    });
}

fn withIds(comptime tag: std.meta.Tag(DataType)) TypeCheckFn {
    return &struct {
        fn check(args: []const Datum) bool {
            return args.len == 2 and std.meta.activeTag(args[0].dataType()) == tag and args[1].dataType() == .uint32;
        }
    }.check;
}

fn countArgs(args: []const Datum) bool {
    return args.len == 2 and is_null.hasValidityBitmap(args[0..1]) and args[1].dataType() == .uint32;
}

fn idsOnly(args: []const Datum) bool {
    return args.len == 1 and args[0].dataType() == .uint32;
}

fn distinctArgs(args: []const Datum) bool {
    return args.len == 2 and isGroupKeyType(args[0].dataType()) and args[1].dataType() == .uint32;
}

/// Group ids seen so far by a lifecycle state: their count bounds the ids (see the file doc).
pub const Seen = struct {
    rows: u64 = 0,
    /// Groups: the largest id plus one.
    groups: usize = 0,
};

/// Call `sink.piece(values, ids, groups)` for every aligned piece of `args` (values then group
/// ids, or the ids alone; arrays or chunked arrays): every chunk passes `validateLayout` and the
/// ids have no null. With `bounded`, every id must be below the number of rows of `args`.
pub fn forEachPiece(allocator: std.mem.Allocator, args: []const Datum, comptime ids_only: bool, comptime bounded: bool, seen: *Seen, sink: anytype) KernelError!void {
    if (args.len != @as(usize, if (ids_only) 1 else 2)) return error.InvalidArity;
    for (args) |a| if (a == .scalar) return error.InvalidInput;
    var it = try core.NaryExecChunkIterator.init(allocator, args);
    defer it.deinit();
    const limit = it.total_len;
    while (try it.next()) |next| {
        var chunk = next;
        defer chunk.deinit();
        for (chunk.values) |v| v.array.data().validateLayout() catch return error.InvalidInput;
        const id_data = chunk.values[chunk.values.len - 1].array.data();
        if (id_data.hasNulls()) return error.InvalidInput;
        const ids = (PrimitiveArray(u32){ .data = id_data }).values() catch return error.InvalidInput;
        seen.rows += ids.len;
        var top: u32 = 0;
        for (ids) |id| top = @max(top, id);
        if (ids.len > 0) {
            if (bounded and top >= limit) return error.InvalidInput;
            seen.groups = @max(seen.groups, @as(usize, top) + 1);
        }
        try sink.piece(if (ids_only) null else chunk.values[0].array, ids, seen.groups);
    }
}

/// Exec and lifecycle callbacks around an `Impl` with a per-group `Group` (its default is the
/// empty group), `out_type`, `update(groups, values, ids)`, `mergeGroup` and `finalize`.
fn Grouped(comptime Impl: type) type {
    const Group = Impl.Group;
    return struct {
        const ids_only = Impl.ids_only;

        const State = struct {
            allocator: std.mem.Allocator,
            groups: std.ArrayList(Group) = .empty,
            seen: Seen = .{},
            /// Type of the values (null for the ids-only functions or before any update).
            value_type: ?DataType = null,

            fn grow(self: *State, n: usize) KernelError!void {
                if (n > self.groups.items.len) self.groups.appendNTimes(self.allocator, .{}, n - self.groups.items.len) catch return error.OutOfMemory;
            }

            pub fn piece(self: *State, values: ?ArrayRef, ids: []const u32, groups: usize) KernelError!void {
                try self.grow(groups);
                if (values) |v| self.value_type = v.data().data_type;
                try Impl.update(self.groups.items, if (values) |v| v.data() else null, ids);
            }
        };

        fn resultType(args: []const Datum, _: Options) KernelError!DataType {
            if (@hasDecl(Impl, "resultType")) return Impl.resultType(args);
            if (comptime aggregate_basic.hasFn(Impl, "resultTypeAlloc")) return error.NotImplemented;
            return Impl.out_type;
        }

        pub const resultTypeAlloc = if (!aggregate_basic.hasFn(Impl, "resultTypeAlloc")) {} else struct {
            fn f(allocator: std.mem.Allocator, args: []const Datum, _: Options) KernelError!DataType {
                return Impl.resultTypeAlloc(allocator, args);
            }
        }.f;

        /// Aggregate the values of every group: one row per group id.
        fn exec(ctx: *ExecContext, args: []const Datum, options: Options) KernelError!Datum {
            var state: State = .{ .allocator = ctx.allocator };
            defer state.groups.deinit(ctx.allocator);
            if (!ids_only and args.len > 0) state.value_type = args[0].dataType();
            try forEachPiece(ctx.allocator, args, ids_only, true, &state.seen, &state);
            return .{ .array = try Impl.finalize(ctx, state.groups.items, state.value_type, options) };
        }

        fn init(ctx: *ExecContext, _: Options) KernelError!*anyopaque {
            const state = ctx.allocator.create(State) catch return error.OutOfMemory;
            state.* = .{ .allocator = ctx.allocator };
            return state;
        }

        fn update(_: *ExecContext, state: *anyopaque, args: []const Datum, _: Options) KernelError!void {
            const s = cast(state);
            try forEachPiece(s.allocator, args, ids_only, false, &s.seen, s);
        }

        fn merge(_: *ExecContext, state: *anyopaque, other: *anyopaque, _: Options) KernelError!void {
            const s = cast(state);
            const o = cast(other);
            try s.grow(o.groups.items.len);
            for (s.groups.items[0..o.groups.items.len], o.groups.items) |*a, b| Impl.mergeGroup(a, b);
            s.seen.rows += o.seen.rows;
            s.seen.groups = @max(s.seen.groups, o.seen.groups);
            if (s.value_type == null) s.value_type = o.value_type;
        }

        fn finalize(ctx: *ExecContext, state: *anyopaque, options: Options) KernelError!Datum {
            const s = cast(state);
            return .{ .array = try Impl.finalize(ctx, s.groups.items, s.value_type, options) };
        }

        fn deinit(ctx: *ExecContext, state: *anyopaque) void {
            const s = cast(state);
            s.groups.deinit(s.allocator);
            ctx.allocator.destroy(s);
        }

        fn cast(state: *anyopaque) *State {
            return @ptrCast(@alignCast(state));
        }
    };
}

/// Typed values of a numeric chunk.
fn typedValues(comptime T: type, data: *const ArrayData) KernelError![]const T {
    return (PrimitiveArray(T){ .data = data }).values() catch error.InvalidInput;
}

/// Run `f(group, value)` for every valid row and count the null rows of every group.
inline fn eachValid(comptime T: type, data: *const ArrayData, ids: []const u32, groups: anytype, comptime f: anytype) KernelError!void {
    const vals = try typedValues(T, data);
    if (dense.validityBits(data)) |bits| {
        for (vals, ids, 0..) |v, id, i| {
            if (bitmap.bitIsSet(bits, data.offset + i)) f(&groups[id], v) else groups[id].nulls += 1;
        }
    } else {
        for (vals, ids) |v, id| f(&groups[id], v);
    }
}

/// Fixed-width output of `n` rows: `fill(values, validity bits)` sets every value and clears the
/// validity bit of a null group (all bits start set); the bitmap is dropped when nothing is null.
fn fixedOutput(comptime O: type, allocator: std.mem.Allocator, out_type: DataType, n: usize, ctx: anytype, comptime fill: anytype) KernelError!ArrayRef {
    var vals = OwnedBuffer.initUninit(allocator, if (O == bool) bitmap.byteLength(n) else n * @sizeOf(O)) catch return error.OutOfMemory;
    var validity = dense.bitsBuffer(allocator, n) catch |err| {
        vals.deinit();
        return err;
    };
    @memset(validity.data[0..bitmap.byteLength(n)], 0xFF);
    if (O == bool) @memset(vals.data[0..bitmap.byteLength(n)], 0);
    fill(ctx, if (O == bool) vals.data[0..bitmap.byteLength(n)] else vals.typedSlice(O)[0..n], validity.data) catch |err| {
        vals.deinit();
        validity.deinit();
        return err;
    };
    var v: ?OwnedBuffer = validity;
    if (bitmap.countUnsetBit(validity.data, n) == 0) {
        validity.deinit();
        v = null;
    }
    // Both take ownership of the buffers, also on error.
    if (O == bool) return dense.boolArray(allocator, n, vals, v);
    return dense.finishFixed(allocator, out_type, n, &vals, v);
}

fn Sum(comptime T: type, comptime which: enum { sum, mean }) type {
    const is_float = @typeInfo(T) == .float;
    const signed = !is_float and @typeInfo(T).int.signedness == .signed;
    const Exact = if (is_float) f64 else if (signed) i128 else u128;
    const Out = if (which == .mean or is_float) f64 else if (signed) i64 else u64;
    return struct {
        const ids_only = false;
        const Group = struct { sum: Exact = 0, count: u64 = 0, nulls: u64 = 0 };
        const out_type = dense.dataTypeOf(Out);

        fn update(groups: []Group, data: ?*const ArrayData, ids: []const u32) KernelError!void {
            try eachValid(T, data.?, ids, groups, struct {
                inline fn f(g: *Group, v: T) void {
                    g.sum += if (is_float) @as(f64, @floatCast(v)) else v;
                    g.count += 1;
                }
            }.f);
        }

        fn mergeGroup(a: *Group, b: Group) void {
            a.sum += b.sum;
            a.count += b.count;
            a.nulls += b.nulls;
        }

        fn finalize(ctx: *ExecContext, groups: []const Group, _: ?DataType, options: Options) KernelError!ArrayRef {
            return fixedOutput(Out, ctx.allocator, out_type, groups.len, .{ ctx, groups, scalarOptions(options) }, struct {
                fn fill(c: anytype, out: []Out, valid: []u8) KernelError!void {
                    for (c[1], out, 0..) |g, *o, i| {
                        if (nullByOptions(c[2], g.count, g.nulls)) {
                            bitmap.clearBit(valid, i);
                            o.* = 0;
                            continue;
                        }
                        // min_count 0: an empty group sums to 0, its mean is NaN (0 / 0).
                        o.* = if (which == .mean)
                            (if (is_float) g.sum else @as(f64, @floatFromInt(g.sum))) / @as(f64, @floatFromInt(g.count))
                        else if (is_float)
                            g.sum
                        else switch (c[0].overflowMode()) {
                            .checked => std.math.cast(Out, g.sum) orelse return error.Overflow,
                            .wrapping => @truncate(g.sum),
                            .saturating => @intCast(std.math.clamp(g.sum, std.math.minInt(Out), std.math.maxInt(Out))),
                        };
                    }
                }
            }.fill);
        }
    };
}

fn MinMax(comptime T: type, comptime which: enum { min, max, min_max }) type {
    const is_float = @typeInfo(T) == .float;
    const min_start: T = if (is_float) std.math.nan(T) else std.math.maxInt(T);
    const max_start: T = if (is_float) std.math.nan(T) else std.math.minInt(T);
    return struct {
        const ids_only = false;
        const Group = struct { min: T = min_start, max: T = max_start, count: u64 = 0, nulls: u64 = 0 };
        const elem_type = dense.dataTypeOf(T);
        const fields = [_]datatype.Field{
            .{ .name = "min", .data_type = &elem_type },
            .{ .name = "max", .data_type = &elem_type },
        };
        const struct_type = DataType{ .struct_ = .{ .fields = &fields } };
        const out_type = if (which == .min_max) struct_type else elem_type;

        fn update(groups: []Group, data: ?*const ArrayData, ids: []const u32) KernelError!void {
            try eachValid(T, data.?, ids, groups, struct {
                inline fn f(g: *Group, v: T) void {
                    g.min = @min(g.min, v);
                    g.max = @max(g.max, v);
                    g.count += 1;
                }
            }.f);
        }

        fn mergeGroup(a: *Group, b: Group) void {
            a.min = @min(a.min, b.min);
            a.max = @max(a.max, b.max);
            a.count += b.count;
            a.nulls += b.nulls;
        }

        fn side(ctx: *ExecContext, groups: []const Group, opts: core.ScalarAggregateOptions, comptime is_min: bool) KernelError!ArrayRef {
            return fixedOutput(T, ctx.allocator, elem_type, groups.len, .{ groups, opts }, struct {
                fn fill(c: anytype, out: []T, valid: []u8) KernelError!void {
                    for (c[0], out, 0..) |g, *o, i| {
                        o.* = if (is_min) g.min else g.max;
                        if (g.count == 0 or nullBySkipNulls(c[1], g.nulls)) {
                            bitmap.clearBit(valid, i);
                            o.* = 0;
                        }
                    }
                }
            }.fill);
        }

        fn finalize(ctx: *ExecContext, groups: []const Group, _: ?DataType, options: Options) KernelError!ArrayRef {
            const o = scalarOptions(options);
            if (which != .min_max) return side(ctx, groups, o, which == .min);
            var mins = try side(ctx, groups, o, true);
            var maxs = side(ctx, groups, o, false) catch |err| {
                mins.release();
                return err;
            };
            var children = [2]ArrayRef{ mins, maxs };
            const no_validity = [1]SharedBuffer{.empty};
            return ArrayRef.fromOwnedInline(ctx.allocator, .{
                .data_type = struct_type,
                .length = groups.len,
                .null_count = 0,
                .buffers = &no_validity,
                .children = &children,
            }) catch {
                mins.release();
                maxs.release();
                return error.OutOfMemory;
            };
        }
    };
}

fn Count(comptime which: enum { count, count_all }) type {
    return struct {
        const ids_only = which == .count_all;
        /// Valid rows and all rows of the group.
        const Group = struct { valid: u64 = 0, rows: u64 = 0 };
        const out_type = int64_type;

        fn update(groups: []Group, data: ?*const ArrayData, ids: []const u32) KernelError!void {
            for (ids) |id| groups[id].rows += 1;
            if (which == .count_all) return;
            const d = data.?;
            if (datatype.storageDataType(d.data_type) == .null) return;
            if (dense.validityBits(d)) |bits| {
                for (ids, 0..) |id, i| groups[id].valid += @intFromBool(bitmap.bitIsSet(bits, d.offset + i));
            } else for (ids) |id| groups[id].valid += 1;
        }

        fn mergeGroup(a: *Group, b: Group) void {
            a.valid += b.valid;
            a.rows += b.rows;
        }

        fn finalize(ctx: *ExecContext, groups: []const Group, _: ?DataType, options: Options) KernelError!ArrayRef {
            const mode: core.CountMode = if (which == .count_all) .all else countMode(options);
            return fixedOutput(i64, ctx.allocator, out_type, groups.len, .{ groups, mode }, struct {
                fn fill(c: anytype, out: []i64, _: []u8) KernelError!void {
                    for (c[0], out) |g, *o| {
                        const n = switch (c[1]) {
                            .only_valid => g.valid,
                            .only_null => g.rows - g.valid,
                            .all => g.rows,
                        };
                        o.* = std.math.cast(i64, n) orelse return error.Overflow;
                    }
                }
            }.fill);
        }
    };
}

fn AnyAll(comptime which: enum { any, all }) type {
    return struct {
        const ids_only = false;
        const Group = struct { any_true: bool = false, any_false: bool = false, count: u64 = 0, nulls: u64 = 0 };
        const out_type = bool_type;

        fn update(groups: []Group, data: ?*const ArrayData, ids: []const u32) KernelError!void {
            const d = data.?;
            const bits = dense.validityBits(d);
            for (ids, 0..) |id, i| {
                if (bits) |b| if (!bitmap.bitIsSet(b, d.offset + i)) {
                    groups[id].nulls += 1;
                    continue;
                };
                const g = &groups[id];
                if (bitmap.bitIsSet(d.buffers[1].data, d.offset + i)) g.any_true = true else g.any_false = true;
                g.count += 1;
            }
        }

        fn mergeGroup(a: *Group, b: Group) void {
            a.any_true = a.any_true or b.any_true;
            a.any_false = a.any_false or b.any_false;
            a.count += b.count;
            a.nulls += b.nulls;
        }

        fn finalize(ctx: *ExecContext, groups: []const Group, _: ?DataType, options: Options) KernelError!ArrayRef {
            return fixedOutput(bool, ctx.allocator, out_type, groups.len, .{ groups, scalarOptions(options) }, struct {
                fn fill(c: anytype, out: []u8, valid: []u8) KernelError!void {
                    const o: core.ScalarAggregateOptions = c[1];
                    for (c[0], 0..) |g, i| {
                        // Kleene without skip_nulls: a null row decides unless a true (any) /
                        // false (all) row already did.
                        const decided = if (which == .any) g.any_true else g.any_false;
                        if ((!o.skip_nulls and !decided and g.nulls > 0) or g.count < o.min_count) {
                            bitmap.clearBit(valid, i);
                            continue;
                        }
                        if (if (which == .any) g.any_true else !g.any_false) bitmap.setBit(out, i);
                    }
                }
            }.fill);
        }
    };
}

/// hash_count_distinct / hash_distinct: a `Grouper` over (group id, value) pairs; `CountOptions`
/// picks the pairs whose value is valid (`only_valid`, the default), null (`only_null`) or both.
fn Distinct(comptime which: enum { count, list }) type {
    return struct {
        const ids_only = false;

        const State = struct {
            allocator: std.mem.Allocator,
            pairs: ?Grouper = null,
            seen: Seen = .{},
            value_type: ?DataType = null,

            fn deinit(self: *State) void {
                if (self.pairs) |*p| p.deinit();
            }

            pub fn piece(self: *State, vals: ?ArrayRef, ids: []const u32, _: usize) KernelError!void {
                const v = vals.?;
                if (self.pairs == null) {
                    self.pairs = try Grouper.init(self.allocator, &.{ uint32_type, v.data().data_type });
                    self.value_type = v.data().data_type;
                }
                // The ids as an array of their own (the chunk may be a slice of a larger one).
                var id_buf = OwnedBuffer.initUninit(self.allocator, ids.len * 4) catch return error.OutOfMemory;
                @memcpy(id_buf.typedSlice(u32)[0..ids.len], ids);
                var id_arr = try dense.finishFixed(self.allocator, uint32_type, ids.len, &id_buf, null);
                defer id_arr.release();
                var pair_ids = try self.pairs.?.consume(&.{ id_arr, v });
                pair_ids.release();
            }
        };

        fn resultType(args: []const Datum, _: Options) KernelError!DataType {
            if (which == .count) return int64_type;
            if (args.len == 0) return error.InvalidArity;
            return listOver(switch (args[0]) {
                .array => |*a| &a.data().data_type,
                .chunked => |*c| &c.node.data_type,
                .scalar => |*s| &s.data_type,
                .record_batch, .table => return error.InvalidInput,
            });
        }

        fn exec(ctx: *ExecContext, args: []const Datum, options: Options) KernelError!Datum {
            var state: State = .{ .allocator = ctx.allocator };
            defer state.deinit();
            try forEachPiece(ctx.allocator, args, false, true, &state.seen, &state);
            return .{ .array = try finish(ctx, &state, args[0].dataType(), countMode(options)) };
        }

        fn init(ctx: *ExecContext, _: Options) KernelError!*anyopaque {
            const state = ctx.allocator.create(State) catch return error.OutOfMemory;
            state.* = .{ .allocator = ctx.allocator };
            return state;
        }

        fn update(_: *ExecContext, state: *anyopaque, args: []const Datum, _: Options) KernelError!void {
            const s = cast(state);
            try forEachPiece(s.allocator, args, false, false, &s.seen, s);
        }

        /// Add the pairs of `other` (same group ids): its distinct pairs are consumed again.
        fn merge(_: *ExecContext, state: *anyopaque, other: *anyopaque, _: Options) KernelError!void {
            const s = cast(state);
            const o = cast(other);
            s.seen.rows += o.seen.rows;
            s.seen.groups = @max(s.seen.groups, o.seen.groups);
            const op = &(o.pairs orelse return);
            const keys = try op.uniques();
            defer {
                for (keys) |*k| k.release();
                s.allocator.free(keys);
            }
            if (s.pairs == null) {
                s.pairs = try Grouper.init(s.allocator, op.key_types);
                s.value_type = o.value_type;
            }
            var ids = try s.pairs.?.consume(keys);
            ids.release();
        }

        fn finalize(ctx: *ExecContext, state: *anyopaque, options: Options) KernelError!Datum {
            const s = cast(state);
            return .{ .array = try finish(ctx, s, s.value_type orelse .{ .null = {} }, countMode(options)) };
        }

        fn deinit(ctx: *ExecContext, state: *anyopaque) void {
            const s = cast(state);
            s.deinit();
            ctx.allocator.destroy(s);
        }

        fn cast(state: *anyopaque) *State {
            return @ptrCast(@alignCast(state));
        }

        /// Whether a distinct pair with this value validity is kept under `mode`.
        inline fn kept(mode: core.CountMode, is_null_value: bool) bool {
            return switch (mode) {
                .only_valid => !is_null_value,
                .only_null => is_null_value,
                .all => true,
            };
        }

        /// Count (or list) the distinct values of every group that `mode` keeps.
        fn finish(ctx: *ExecContext, s: *State, value_type: DataType, mode: core.CountMode) KernelError!ArrayRef {
            const allocator = ctx.allocator;
            const n = s.seen.groups;
            const counts = allocator.alloc(usize, n + 1) catch return error.OutOfMemory;
            defer allocator.free(counts);
            @memset(counts, 0);
            var keys: []ArrayRef = &.{};
            defer {
                for (keys) |*k| k.release();
                allocator.free(keys);
            }
            if (s.pairs) |*p| keys = try p.uniques();
            const group_of: []const u32 = if (keys.len > 0) try typedValues(u32, keys[0].data()) else &.{};
            for (group_of, 0..) |g, i| {
                if (kept(mode, keys[1].data().isNull(i))) counts[g + 1] += 1;
            }
            if (which == .count) {
                return fixedOutput(i64, allocator, int64_type, n, counts[1..], struct {
                    fn fill(cs: []const usize, out: []i64, _: []u8) KernelError!void {
                        for (cs, out) |c, *o| o.* = @intCast(c);
                    }
                }.fill);
            }
            // list<T>: the pairs grouped by group id, in first-seen order (a stable counting sort).
            for (1..n + 1) |g| counts[g] += counts[g - 1];
            var offsets = OwnedBuffer.initUninit(allocator, (n + 1) * 4) catch return error.OutOfMemory;
            defer offsets.deinit();
            for (offsets.typedSlice(i32)[0 .. n + 1], counts) |*o, c| o.* = std.math.cast(i32, c) orelse return error.Overflow;
            const order = allocator.alloc(usize, counts[n]) catch return error.OutOfMemory;
            defer allocator.free(order);
            const fill = allocator.dupe(usize, counts[0..n]) catch return error.OutOfMemory;
            defer allocator.free(fill);
            for (group_of, 0..) |g, i| {
                if (!kept(mode, keys[1].data().isNull(i))) continue;
                order[fill[g]] = i;
                fill[g] += 1;
            }
            var child = if (keys.len > 0)
                try takeArray(allocator, keys[1], order)
            else
                try dense.nullArray(allocator, value_type, 0);
            errdefer child.release();
            var buffers = [2]SharedBuffer{ .empty, .empty };
            buffers[1] = offsets.toShared((n + 1) * 4) catch return error.OutOfMemory;
            errdefer buffers[1].release();
            var children = [1]ArrayRef{child};
            return ArrayRef.fromOwnedInline(allocator, .{
                // The value field points at the child's own type, which lives as long as the list.
                .data_type = listOver(&child.data().data_type),
                .length = n,
                .null_count = 0,
                .buffers = &buffers,
                .children = &children,
            }) catch error.OutOfMemory;
        }
    };
}

/// hash_sum / hash_mean of decimal128 / decimal256: exact sums (`aggregate_basic` rules).
fn DecimalSum(comptime T: type, comptime which: enum { sum, mean }) type {
    const Exact = if (T == i128) i256 else i512;
    const max_p: u8 = if (T == i128) 38 else 76;
    return struct {
        const ids_only = false;
        const Group = struct { sum: Exact = 0, count: u64 = 0, nulls: u64 = 0 };

        fn outType(dt: DataType) KernelError!DataType {
            const p = try decimal.checkedParams(dt);
            if (which == .mean) return dt;
            const out: decimal.Params = .{ .precision = max_p, .scale = p.scale };
            return if (T == i128) .{ .decimal128 = out } else .{ .decimal256 = out };
        }

        pub fn resultType(args: []const Datum) KernelError!DataType {
            if (args.len != 2) return error.InvalidArity;
            return outType(args[0].dataType());
        }

        fn update(groups: []Group, data: ?*const ArrayData, ids: []const u32) KernelError!void {
            _ = try decimal.checkedParams(data.?.data_type);
            try eachValid(T, data.?, ids, groups, struct {
                inline fn f(g: *Group, v: T) void {
                    g.sum += v;
                    g.count += 1;
                }
            }.f);
        }

        fn mergeGroup(a: *Group, b: Group) void {
            a.sum += b.sum;
            a.count += b.count;
            a.nulls += b.nulls;
        }

        fn finalize(ctx: *ExecContext, groups: []const Group, value_type: ?DataType, options: Options) KernelError!ArrayRef {
            const out_type = try outType(value_type orelse return error.InvalidInput);
            return fixedOutput(T, ctx.allocator, out_type, groups.len, .{ ctx, groups, scalarOptions(options) }, struct {
                fn fill(c: anytype, out: []T, valid: []u8) KernelError!void {
                    for (c[1], out, 0..) |g, *o, i| {
                        o.* = 0;
                        if (nullByOptions(c[2], g.count, g.nulls) or (which == .mean and g.count == 0)) {
                            bitmap.clearBit(valid, i);
                            continue;
                        }
                        o.* = if (which == .mean)
                            @intCast(aggregate_basic.roundedMean(Exact, g.sum, g.count))
                        else
                            try aggregate_basic.decimalSumValue(T, Exact, c[0], g.sum);
                    }
                }
            }.fill);
        }
    };
}

/// hash_min / hash_max of decimal128 / decimal256.
fn DecimalMinMax(comptime T: type, comptime which: enum { min, max }) type {
    const start: T = if (which == .min) std.math.maxInt(T) else std.math.minInt(T);
    return struct {
        const ids_only = false;
        const Group = struct { v: T = start, count: u64 = 0, nulls: u64 = 0 };

        pub fn resultType(args: []const Datum) KernelError!DataType {
            if (args.len != 2) return error.InvalidArity;
            _ = try decimal.checkedParams(args[0].dataType());
            return args[0].dataType();
        }

        fn update(groups: []Group, data: ?*const ArrayData, ids: []const u32) KernelError!void {
            _ = try decimal.checkedParams(data.?.data_type);
            try eachValid(T, data.?, ids, groups, struct {
                inline fn f(g: *Group, v: T) void {
                    g.v = if (which == .min) @min(g.v, v) else @max(g.v, v);
                    g.count += 1;
                }
            }.f);
        }

        fn mergeGroup(a: *Group, b: Group) void {
            a.v = if (which == .min) @min(a.v, b.v) else @max(a.v, b.v);
            a.count += b.count;
            a.nulls += b.nulls;
        }

        fn finalize(ctx: *ExecContext, groups: []const Group, value_type: ?DataType, options: Options) KernelError!ArrayRef {
            const out_type = value_type orelse return error.InvalidInput;
            return fixedOutput(T, ctx.allocator, out_type, groups.len, .{ groups, scalarOptions(options) }, struct {
                fn fill(c: anytype, out: []T, valid: []u8) KernelError!void {
                    for (c[0], out, 0..) |g, *o, i| {
                        o.* = g.v;
                        if (g.count == 0 or nullBySkipNulls(c[1], g.nulls)) {
                            bitmap.clearBit(valid, i);
                            o.* = 0;
                        }
                    }
                }
            }.fill);
        }
    };
}

fn listOver(value_type: *const DataType) DataType {
    return .{ .list = .{ .value_field = .{ .name = "item", .data_type = value_type } } };
}

/// Rows `indices` of `arr` as a new array (an empty one of its type for no index).
fn takeArray(allocator: std.mem.Allocator, arr: ArrayRef, indices: []const usize) KernelError!ArrayRef {
    // No rows: an empty array of the type (a dictionary keeps its dictionary through the take).
    if (indices.len == 0 and arr.data().data_type != .dictionary) return dense.nullArray(allocator, arr.data().data_type, 0);
    var taken = try core.datumTake(.{ .array = arr }, indices);
    return switch (taken) {
        .array => |a| a,
        else => {
            taken.release();
            return error.InvalidInput;
        },
    };
}

// ------------------------------------------------------------------------------ group by

/// A key column of `groupBy`.
pub const Key = struct { name: []const u8, array: ArrayRef };

/// One aggregate of `groupBy`: a `hash_*` function over `input` (null for hash_count_all).
pub const Aggregate = struct {
    function: []const u8,
    input: ?ArrayRef = null,
    name: []const u8,
    options: Options = .none,
};

/// Group the rows of `keys` and aggregate each input per group (Arrow `Table.group_by(keys)
/// .aggregate(...)`): a record batch of the key columns (one row per group, first-seen order)
/// then one column per aggregate. Every key and input has the same number of rows.
pub fn groupBy(ctx: *ExecContext, keys: []const Key, aggregates: []const Aggregate) KernelError!RecordBatch {
    const allocator = ctx.allocator;
    if (keys.len == 0) return error.InvalidArity;
    const types = allocator.alloc(DataType, keys.len) catch return error.OutOfMemory;
    defer allocator.free(types);
    const arrays = allocator.alloc(ArrayRef, keys.len) catch return error.OutOfMemory;
    defer allocator.free(arrays);
    for (keys, types, arrays) |k, *t, *a| {
        t.* = k.array.data().data_type;
        a.* = k.array;
    }
    var grouper = try Grouper.init(allocator, types);
    defer grouper.deinit();
    var ids = try grouper.consume(arrays);
    defer ids.release();

    const columns = allocator.alloc(ArrayRef, keys.len + aggregates.len) catch return error.OutOfMemory;
    var made: usize = 0;
    var owned = true;
    defer if (owned) {
        for (columns[0..made]) |*c| c.release();
        allocator.free(columns);
    };
    const uniques = try grouper.uniques();
    defer allocator.free(uniques);
    @memcpy(columns[0..keys.len], uniques);
    made = keys.len;
    for (aggregates) |agg| {
        const id_datum: Datum = .{ .array = ids };
        var out = if (agg.input) |input| blk: {
            if (input.data().length != ids.data().length) return error.InvalidInput;
            break :blk try ctx.invoke(agg.function, .hash_aggregate, &.{ .{ .array = input }, id_datum }, agg.options);
        } else try ctx.invoke(agg.function, .hash_aggregate, &.{id_datum}, agg.options);
        columns[made] = switch (out) {
            .array => |a| a,
            else => {
                out.release();
                return error.InvalidInput;
            },
        };
        made += 1;
    }

    const fields = allocator.alloc(datatype.Field, columns.len) catch return error.OutOfMemory;
    defer allocator.free(fields);
    for (fields, columns, 0..) |*f, c, i| f.* = .{ .name = if (i < keys.len) keys[i].name else aggregates[i - keys.len].name, .data_type = &c.data().data_type };
    var schema = schema_mod.SchemaRef.fromBorrowed(allocator, .{ .fields = fields }) catch return error.OutOfMemory;
    const batch = RecordBatch.initPrevalidatedOwned(allocator, schema, columns) catch {
        schema.release();
        return error.InvalidInput;
    };
    owned = false;
    return batch;
}

// ------------------------------------------------------------------------------ round 16 kernels

const stats = @import("aggregate_stats.zig");

/// Register hash_product, hash_variance, hash_stddev, hash_skew, hash_kurtosis, hash_first,
/// hash_last, hash_first_last, hash_one, hash_list and the decimal hash_min_max.
fn registerMore(registry: *FunctionRegistry) KernelError!void {
    inline for (dense.numeric_types) |T| {
        const tag = comptime std.meta.activeTag(dense.dataTypeOf(T));
        try add(registry, "hash_product", withIds(tag), aggregateOptions, Grouped(Product(T)));
    }
    inline for (stats.moment_types) |pair| {
        const T = pair[0];
        const tag = pair[1];
        const dec = comptime (tag == .decimal32 or tag == .decimal64 or tag == .decimal128 or tag == .decimal256);
        try add(registry, "hash_variance", withIds(tag), varianceOptions, Grouped(GroupMoments(T, dec, .variance)));
        try add(registry, "hash_stddev", withIds(tag), varianceOptions, Grouped(GroupMoments(T, dec, .stddev)));
        try add(registry, "hash_skew", withIds(tag), skewOptions, Grouped(GroupMoments(T, dec, .skew)));
        try add(registry, "hash_kurtosis", withIds(tag), skewOptions, Grouped(GroupMoments(T, dec, .kurtosis)));
    }
    inline for (.{ i128, i256 }) |T| {
        const check = comptime withIds(if (T == i128) .decimal128 else .decimal256);
        try add(registry, "hash_min_max", check, aggregateOptions, Grouped(DecimalMinMaxPair(T)));
        try add(registry, "hash_product", check, aggregateOptions, Grouped(DecimalProductGroups(T)));
    }
    try add(registry, "hash_first", anyWithIds, aggregateOptions, Pick(.first));
    try add(registry, "hash_last", anyWithIds, aggregateOptions, Pick(.last));
    try add(registry, "hash_first_last", anyWithIds, aggregateOptions, Pick(.first_last));
    try add(registry, "hash_one", anyWithIds, dense.noOptions, Pick(.one));
    try add(registry, "hash_list", anyWithIds, dense.noOptions, Pick(.list));
    try add(registry, "hash_min", orderedWithIds, aggregateOptions, OrderedGroups(.min));
    try add(registry, "hash_max", orderedWithIds, aggregateOptions, OrderedGroups(.max));
    try add(registry, "hash_min_max", orderedWithIds, aggregateOptions, OrderedGroups(.min_max));
}

fn varianceOptions(options: Options) bool {
    return options == .none or options == .variance;
}

fn skewOptions(options: Options) bool {
    return options == .none or options == .skew;
}

/// Values of any type the take kernels handle, then the group ids.
fn anyWithIds(args: []const Datum) bool {
    return args.len == 2 and args[1].dataType() == .uint32 and args[0].dataType() != .null;
}

fn Product(comptime T: type) type {
    const is_float = @typeInfo(T) == .float;
    const signed = !is_float and @typeInfo(T).int.signedness == .signed;
    const Out = if (is_float) f64 else if (signed) i64 else u64;
    return struct {
        const ids_only = false;
        const Group = struct { prod: Out = 1, count: u64 = 0, nulls: u64 = 0, zero: bool = false, neg: bool = false, overflow: bool = false };
        const out_type = dense.dataTypeOf(Out);

        inline fn mul(g: *Group, v: Out) void {
            if (is_float) {
                g.prod *= v;
                return;
            }
            if (v == 0) g.zero = true;
            if (signed and v < 0) g.neg = !g.neg;
            const r = @mulWithOverflow(g.prod, v);
            g.prod = r[0];
            g.overflow = g.overflow or r[1] != 0;
        }

        fn update(groups: []Group, data: ?*const ArrayData, ids: []const u32) KernelError!void {
            try eachValid(T, data.?, ids, groups, struct {
                inline fn f(g: *Group, v: T) void {
                    mul(g, if (is_float) @floatCast(v) else v);
                    g.count += 1;
                }
            }.f);
        }

        fn mergeGroup(a: *Group, b: Group) void {
            a.count += b.count;
            a.nulls += b.nulls;
            if (is_float) {
                a.prod *= b.prod;
                return;
            }
            a.zero = a.zero or b.zero;
            if (b.neg) a.neg = !a.neg;
            const r = @mulWithOverflow(a.prod, b.prod);
            a.prod = r[0];
            a.overflow = a.overflow or b.overflow or r[1] != 0;
        }

        fn finalize(ctx: *ExecContext, groups: []const Group, _: ?DataType, options: Options) KernelError!ArrayRef {
            return fixedOutput(Out, ctx.allocator, out_type, groups.len, .{ ctx, groups, scalarOptions(options) }, struct {
                fn fill(c: anytype, out: []Out, valid: []u8) KernelError!void {
                    for (c[1], out, 0..) |g, *o, i| {
                        o.* = 0;
                        if (nullByOptions(c[2], g.count, g.nulls)) {
                            bitmap.clearBit(valid, i);
                            continue;
                        }
                        o.* = g.prod;
                        if (!is_float and g.zero) o.* = 0;
                        if (!is_float and !g.zero and g.overflow) switch (c[0].overflowMode()) {
                            .checked => return error.Overflow,
                            .wrapping => {},
                            .saturating => o.* = if (g.neg) std.math.minInt(Out) else std.math.maxInt(Out),
                        };
                    }
                }
            }.fill);
        }
    };
}

fn GroupMoments(comptime T: type, comptime is_dec: bool, comptime kind: stats.MomentKind) type {
    return struct {
        const ids_only = false;
        const Group = struct { m: stats.MomentSums = .{}, nulls: u64 = 0 };
        const out_type = DataType{ .double = {} };

        fn update(groups: []Group, data: ?*const ArrayData, ids: []const u32) KernelError!void {
            const d = data.?;
            const scale: i32 = if (is_dec) (try decimal.checkedParams(d.data_type)).scale else 0;
            const vals = try typedValues(T, d);
            const bits = dense.validityBits(d);
            for (vals, ids, 0..) |v, id, i| {
                if (bits) |b| if (!bitmap.bitIsSet(b, d.offset + i)) {
                    groups[id].nulls += 1;
                    continue;
                };
                const x: f64 = if (is_dec) decimal.toFloatValue(f64, v, scale) else if (@typeInfo(T) == .float) @floatCast(v) else @floatFromInt(v);
                groups[id].m = stats.MomentSums.merge(groups[id].m, stats.MomentSums.of(x));
            }
        }

        fn mergeGroup(a: *Group, b: Group) void {
            a.m = stats.MomentSums.merge(a.m, b.m);
            a.nulls += b.nulls;
        }

        fn finalize(ctx: *ExecContext, groups: []const Group, _: ?DataType, options: Options) KernelError!ArrayRef {
            if (options == .variance and options.variance.ddof < 0) return error.InvalidOptions;
            return fixedOutput(f64, ctx.allocator, out_type, groups.len, .{ groups, options }, struct {
                fn fill(c: anytype, out: []f64, valid: []u8) KernelError!void {
                    for (c[0], out, 0..) |g, *o, i| {
                        o.* = stats.momentResult(kind, g.m, g.nulls, c[1]) orelse {
                            bitmap.clearBit(valid, i);
                            o.* = 0;
                            continue;
                        };
                    }
                }
            }.fill);
        }
    };
}

/// hash_min_max of decimal128 / decimal256: `struct<min, max>` of the input type, stored with
/// the array (`dense.structArray`).
fn DecimalMinMaxPair(comptime T: type) type {
    return struct {
        const ids_only = false;
        const Group = struct { min: T = std.math.maxInt(T), max: T = std.math.minInt(T), count: u64 = 0, nulls: u64 = 0 };

        pub fn resultTypeAlloc(allocator: std.mem.Allocator, args: []const Datum) KernelError!DataType {
            if (args.len != 2) return error.InvalidArity;
            const dt = args[0].dataType();
            _ = try decimal.checkedParams(dt);
            return dense.structTypeAlloc(allocator, &aggregate_basic.min_max_names, &.{ dt, dt });
        }

        fn update(groups: []Group, data: ?*const ArrayData, ids: []const u32) KernelError!void {
            _ = try decimal.checkedParams(data.?.data_type);
            try eachValid(T, data.?, ids, groups, struct {
                inline fn f(g: *Group, v: T) void {
                    g.min = @min(g.min, v);
                    g.max = @max(g.max, v);
                    g.count += 1;
                }
            }.f);
        }

        fn mergeGroup(a: *Group, b: Group) void {
            a.min = @min(a.min, b.min);
            a.max = @max(a.max, b.max);
            a.count += b.count;
            a.nulls += b.nulls;
        }

        fn side(ctx: *ExecContext, dt: DataType, groups: []const Group, opts: core.ScalarAggregateOptions, comptime is_min: bool) KernelError!ArrayRef {
            return fixedOutput(T, ctx.allocator, dt, groups.len, .{ groups, opts }, struct {
                fn fill(c: anytype, out: []T, valid: []u8) KernelError!void {
                    for (c[0], out, 0..) |g, *o, i| {
                        o.* = if (is_min) g.min else g.max;
                        if (g.count == 0 or nullBySkipNulls(c[1], g.nulls)) {
                            bitmap.clearBit(valid, i);
                            o.* = 0;
                        }
                    }
                }
            }.fill);
        }

        fn finalize(ctx: *ExecContext, groups: []const Group, value_type: ?DataType, options: Options) KernelError!ArrayRef {
            const dt = value_type orelse return error.InvalidInput;
            const o = scalarOptions(options);
            const mins = try side(ctx, dt, groups, o, true);
            const maxs = side(ctx, dt, groups, o, false) catch |err| {
                var m = mins;
                m.release();
                return err;
            };
            return dense.structArray(ctx.allocator, &aggregate_basic.min_max_names, &.{ mins, maxs }, groups.len, null);
        }
    };
}

const first_last_names = [_][]const u8{ "first", "last" };

/// hash_first, hash_last, hash_first_last, hash_one (the first non-null value of the group;
/// Arrow returns an arbitrary one) and hash_list (every value of the group, nulls included, in
/// row order: `list<T>`). The value batches are retained; each group keeps the global row of its
/// first / last (valid) value, then one take builds the output.
fn Pick(comptime which: enum { first, last, first_last, one, list }) type {
    return struct {
        const ids_only = false;

        const Rows = struct {
            first_valid: ?u64 = null,
            last_valid: ?u64 = null,
            first_any: ?u64 = null,
            last_any: ?u64 = null,
            count: u64 = 0,
        };

        const State = struct {
            allocator: std.mem.Allocator,
            batches: std.ArrayList(ArrayRef) = .empty,
            total: u64 = 0,
            groups: std.ArrayList(Rows) = .empty,
            /// hash_list: (group, global row) of every row in order.
            pairs: std.ArrayList(struct { group: u32, row: u64 }) = .empty,
            seen: Seen = .{},
            value_type: ?DataType = null,

            fn deinit(self: *State) void {
                for (self.batches.items) |*b| b.release();
                self.batches.deinit(self.allocator);
                self.groups.deinit(self.allocator);
                self.pairs.deinit(self.allocator);
            }

            pub fn piece(self: *State, vals: ?ArrayRef, ids: []const u32, groups: usize) KernelError!void {
                const v = vals.?;
                if (self.value_type == null) self.value_type = v.data().data_type;
                if (groups > self.groups.items.len) self.groups.appendNTimes(self.allocator, .{}, groups - self.groups.items.len) catch return error.OutOfMemory;
                self.batches.append(self.allocator, v.retain()) catch return error.OutOfMemory;
                const d = v.data();
                const base = self.total;
                self.total += d.length;
                if (which == .list) {
                    self.pairs.ensureUnusedCapacity(self.allocator, ids.len) catch return error.OutOfMemory;
                    for (ids, 0..) |id, i| self.pairs.appendAssumeCapacity(.{ .group = id, .row = base + i });
                    return;
                }
                for (ids, 0..) |id, i| {
                    const g = &self.groups.items[id];
                    const row = base + i;
                    if (g.first_any == null) g.first_any = row;
                    g.last_any = row;
                    if (!d.isNull(i)) {
                        if (g.first_valid == null) g.first_valid = row;
                        g.last_valid = row;
                        g.count += 1;
                    }
                }
            }
        };

        fn resultType(args: []const Datum, _: Options) KernelError!DataType {
            if (args.len != 2) return error.InvalidArity;
            if (which == .first_last) return error.NotImplemented;
            if (which == .list) return listOver(switch (args[0]) {
                .array => |*a| &a.data().data_type,
                .chunked => |*c| &c.node.data_type,
                .scalar => |*s| &s.data_type,
                .record_batch, .table => return error.InvalidInput,
            });
            return args[0].dataType();
        }

        pub const resultTypeAlloc = if (which != .first_last) {} else struct {
            fn f(allocator: std.mem.Allocator, args: []const Datum, _: Options) KernelError!DataType {
                if (args.len != 2) return error.InvalidArity;
                return dense.structTypeAlloc(allocator, &first_last_names, &.{ args[0].dataType(), args[0].dataType() });
            }
        }.f;

        fn exec(ctx: *ExecContext, args: []const Datum, options: Options) KernelError!Datum {
            var state: State = .{ .allocator = ctx.allocator };
            defer state.deinit();
            if (args.len > 0) state.value_type = args[0].dataType();
            try forEachPiece(ctx.allocator, args, false, true, &state.seen, &state);
            return .{ .array = try finish(ctx, &state, options) };
        }

        fn init(ctx: *ExecContext, _: Options) KernelError!*anyopaque {
            const state = ctx.allocator.create(State) catch return error.OutOfMemory;
            state.* = .{ .allocator = ctx.allocator };
            return state;
        }

        fn update(_: *ExecContext, state: *anyopaque, args: []const Datum, _: Options) KernelError!void {
            const s = cast(state);
            try forEachPiece(s.allocator, args, false, false, &s.seen, s);
        }

        /// Append the batches of `other` (same group ids) after this state's rows.
        fn merge(_: *ExecContext, state: *anyopaque, other: *anyopaque, _: Options) KernelError!void {
            const s = cast(state);
            const o = cast(other);
            const base = s.total;
            for (o.batches.items) |b| s.batches.append(s.allocator, b.retain()) catch return error.OutOfMemory;
            s.total += o.total;
            if (o.groups.items.len > s.groups.items.len) s.groups.appendNTimes(s.allocator, .{}, o.groups.items.len - s.groups.items.len) catch return error.OutOfMemory;
            for (s.groups.items[0..o.groups.items.len], o.groups.items) |*a, b| {
                if (a.first_any == null) a.first_any = if (b.first_any) |r| base + r else null;
                if (a.first_valid == null) a.first_valid = if (b.first_valid) |r| base + r else null;
                if (b.last_any) |r| a.last_any = base + r;
                if (b.last_valid) |r| a.last_valid = base + r;
                a.count += b.count;
            }
            s.pairs.ensureUnusedCapacity(s.allocator, o.pairs.items.len) catch return error.OutOfMemory;
            for (o.pairs.items) |p| s.pairs.appendAssumeCapacity(.{ .group = p.group, .row = base + p.row });
            s.seen.rows += o.seen.rows;
            s.seen.groups = @max(s.seen.groups, o.seen.groups);
            if (s.value_type == null) s.value_type = o.value_type;
        }

        fn finalize(ctx: *ExecContext, state: *anyopaque, options: Options) KernelError!Datum {
            return .{ .array = try finish(ctx, cast(state), options) };
        }

        fn deinit(ctx: *ExecContext, state: *anyopaque) void {
            const s = cast(state);
            s.deinit();
            ctx.allocator.destroy(s);
        }

        fn cast(state: *anyopaque) *State {
            return @ptrCast(@alignCast(state));
        }

        /// The row of group `g` for `first` (or `last`), null when none qualifies. `min_count`
        /// is not applied: pyarrow 26 group_by ignores it for hash_first / hash_last.
        fn rowOf(g: Rows, comptime is_first: bool, o: core.ScalarAggregateOptions) ?usize {
            const r = if (o.skip_nulls) (if (is_first) g.first_valid else g.last_valid) else (if (is_first) g.first_any else g.last_any);
            return if (r) |x| @intCast(x) else null;
        }

        fn finish(ctx: *ExecContext, s: *State, options: Options) KernelError!ArrayRef {
            const allocator = ctx.allocator;
            const n = s.seen.groups;
            if (s.groups.items.len < n) s.groups.appendNTimes(allocator, .{}, n - s.groups.items.len) catch return error.OutOfMemory;
            var all = try concatBatches(allocator, s.value_type, s.batches.items);
            defer all.release();
            if (which == .list) return listOf(allocator, s, all, n);
            const o: core.ScalarAggregateOptions = if (which == .one) .{} else scalarOptions(options);
            const indices = allocator.alloc(?usize, n) catch return error.OutOfMemory;
            defer allocator.free(indices);
            for (indices, s.groups.items[0..n]) |*ix, g| ix.* = rowOf(g, which != .last, o);
            if (n == 0) return all.slice(0, 0) catch error.OutOfMemory;
            const firsts = try takeNullableArray(all, indices);
            if (which != .first_last) return firsts;
            for (indices, s.groups.items[0..n]) |*ix, g| ix.* = rowOf(g, false, o);
            const lasts = takeNullableArray(all, indices) catch |err| {
                var f = firsts;
                f.release();
                return err;
            };
            return dense.structArray(allocator, &first_last_names, &.{ firsts, lasts }, n, null);
        }

        /// hash_list: the rows of every group in row order (a stable counting sort).
        fn listOf(allocator: std.mem.Allocator, s: *const State, all: ArrayRef, n: usize) KernelError!ArrayRef {
            const counts = allocator.alloc(usize, n + 1) catch return error.OutOfMemory;
            defer allocator.free(counts);
            @memset(counts, 0);
            for (s.pairs.items) |p| counts[p.group + 1] += 1;
            for (1..n + 1) |g| counts[g] += counts[g - 1];
            var offsets = OwnedBuffer.initUninit(allocator, (n + 1) * 4) catch return error.OutOfMemory;
            defer offsets.deinit();
            for (offsets.typedSlice(i32)[0 .. n + 1], counts) |*o, c| o.* = std.math.cast(i32, c) orelse return error.Overflow;
            const order = allocator.alloc(usize, counts[n]) catch return error.OutOfMemory;
            defer allocator.free(order);
            const fill = allocator.dupe(usize, counts[0..n]) catch return error.OutOfMemory;
            defer allocator.free(fill);
            for (s.pairs.items) |p| {
                order[fill[p.group]] = @intCast(p.row);
                fill[p.group] += 1;
            }
            var child = try takeArray(allocator, all, order);
            errdefer child.release();
            var buffers = [2]SharedBuffer{ .empty, .empty };
            buffers[1] = offsets.toShared((n + 1) * 4) catch return error.OutOfMemory;
            errdefer buffers[1].release();
            var children = [1]ArrayRef{child};
            return ArrayRef.fromOwnedInline(allocator, .{
                .data_type = listOver(&child.data().data_type),
                .length = n,
                .null_count = 0,
                .buffers = &buffers,
                .children = &children,
            }) catch error.OutOfMemory;
        }
    };
}

/// hash_product of decimal128 / decimal256 (`ScalarAggregateOptions`): the input type, each
/// group's product stepped row by row as Arrow's grouped product (`stats.DecimalProduct`).
fn DecimalProductGroups(comptime T: type) type {
    const Acc = stats.DecimalProduct(T);
    const Scale = stats.DecimalScale(T);
    return struct {
        const ids_only = false;
        const Group = struct { acc: Acc = .{}, count: u64 = 0, nulls: u64 = 0, params: ?decimal.Params = null };

        pub fn resultType(args: []const Datum) KernelError!DataType {
            if (args.len != 2) return error.InvalidArity;
            _ = try Scale.of(args[0].dataType());
            return args[0].dataType();
        }

        fn typeOf(p: decimal.Params) DataType {
            return if (T == i128) .{ .decimal128 = p } else .{ .decimal256 = p };
        }

        fn update(groups: []Group, data: ?*const ArrayData, ids: []const u32) KernelError!void {
            const d = data.?;
            const s = try Scale.of(d.data_type);
            const vals = try typedValues(T, d);
            const bits = dense.validityBits(d);
            for (vals, ids, 0..) |v, id, i| {
                const g = &groups[id];
                g.params = s.params;
                if (bits) |b| if (!bitmap.bitIsSet(b, d.offset + i)) {
                    g.nulls += 1;
                    continue;
                };
                g.acc.mul(v, &s);
                g.count += 1;
            }
        }

        fn mergeGroup(a: *Group, b: Group) void {
            a.count += b.count;
            a.nulls += b.nulls;
            const p = a.params orelse b.params orelse return;
            a.params = p;
            // The parameters were checked by `update`.
            const s = Scale.of(typeOf(p)) catch unreachable;
            a.acc.merge(b.acc, &s);
        }

        fn finalize(ctx: *ExecContext, groups: []const Group, value_type: ?DataType, options: Options) KernelError!ArrayRef {
            const dt = value_type orelse return error.InvalidInput;
            const s = try Scale.of(dt);
            return fixedOutput(T, ctx.allocator, dt, groups.len, .{ ctx, groups, scalarOptions(options), &s }, struct {
                fn fill(c: anytype, out: []T, valid: []u8) KernelError!void {
                    for (c[1], out, 0..) |g, *o, i| {
                        o.* = 0;
                        if (nullByOptions(c[2], g.count, g.nulls)) {
                            bitmap.clearBit(valid, i);
                            continue;
                        }
                        o.* = try g.acc.result(c[0].overflowMode(), c[3]);
                    }
                }
            }.fill);
        }
    };
}

// ------------------------------------------------------------------------------ round 18 kernels

/// Values of an ordered non-numeric type (`aggregate_basic.isOrderedType`), then the group ids.
fn orderedWithIds(args: []const Datum) bool {
    return args.len == 2 and args[1].dataType() == .uint32 and aggregate_basic.isOrderedType(args[0].dataType());
}

/// The batches as one array of `dt` (none: an empty array).
fn concatBatches(allocator: std.mem.Allocator, dt: ?DataType, batches: []const ArrayRef) KernelError!ArrayRef {
    const t = dt orelse return error.InvalidInput;
    if (batches.len == 1) return batches[0].retain();
    if (batches.len == 0) return dense.nullArray(allocator, t, 0);
    return concat.concatArrayRefs(allocator, t, batches) catch |err| switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        else => error.InvalidInput,
    };
}

/// `datumTakeNullable` of an array, as an array.
fn takeNullableArray(arr: ArrayRef, indices: []const ?usize) KernelError!ArrayRef {
    var taken = try core.datumTakeNullable(.{ .array = arr }, indices);
    return switch (taken) {
        .array => |a| a,
        else => {
            taken.release();
            return error.InvalidInput;
        },
    };
}

/// hash_min, hash_max and hash_min_max of the ordered non-numeric types (bool, temporal,
/// decimal32 / decimal64, string / binary layouts, fixed_size_binary, null): the value
/// batches are retained with their typed rows, each group keeps the (batch, row) of its
/// smallest and largest value, then one take per output builds the result. hash_min_max gives
/// `struct<min, max>` of the input type stored with the array. Options as hash_min.
fn OrderedGroups(comptime which: enum { min, max, min_max }) type {
    return struct {
        const ids_only = false;

        const Loc = struct { batch: u32, row: usize };
        const Best = struct { lo: ?Loc = null, hi: ?Loc = null, count: u64 = 0, nulls: u64 = 0 };

        const State = struct {
            allocator: std.mem.Allocator,
            batches: std.ArrayList(ArrayRef) = .empty,
            /// Typed rows of every batch (unused for the null type).
            values: std.ArrayList(sort.Values) = .empty,
            groups: std.ArrayList(Best) = .empty,
            seen: Seen = .{},
            value_type: ?DataType = null,

            fn deinit(self: *State) void {
                for (self.batches.items) |*b| b.release();
                self.batches.deinit(self.allocator);
                self.values.deinit(self.allocator);
                self.groups.deinit(self.allocator);
            }

            fn order(self: *const State, a: Loc, b: Loc) std.math.Order {
                return sort.valueOrder(&self.values.items[a.batch], a.row, &self.values.items[b.batch], b.row);
            }

            fn offer(self: *const State, g: *Best, at: Loc) void {
                if (g.lo == null or (which != .max and self.order(at, g.lo.?) == .lt)) g.lo = at;
                if (g.hi == null or (which != .min and self.order(at, g.hi.?) == .gt)) g.hi = at;
            }

            /// Retain `v` as a new batch; its index.
            fn addBatch(self: *State, v: ArrayRef) KernelError!u32 {
                const d = v.data();
                if (self.value_type) |t| if (!t.eql(d.data_type)) return error.InvalidInput;
                if (self.value_type == null) self.value_type = d.data_type;
                if (self.batches.items.len >= std.math.maxInt(u32)) return error.InvalidInput;
                const vals: sort.Values = if (d.data_type == .null) .{ .u8 = &.{} } else try sort.valuesOf(v);
                self.values.append(self.allocator, vals) catch return error.OutOfMemory;
                self.batches.append(self.allocator, v.retain()) catch {
                    _ = self.values.pop();
                    return error.OutOfMemory;
                };
                return @intCast(self.batches.items.len - 1);
            }

            pub fn piece(self: *State, vals: ?ArrayRef, ids: []const u32, groups: usize) KernelError!void {
                const v = vals.?;
                if (groups > self.groups.items.len) self.groups.appendNTimes(self.allocator, .{}, groups - self.groups.items.len) catch return error.OutOfMemory;
                const b = try self.addBatch(v);
                const d = v.data();
                if (d.data_type == .null) {
                    for (ids) |id| self.groups.items[id].nulls += 1;
                    return;
                }
                for (ids, 0..) |id, i| {
                    const g = &self.groups.items[id];
                    if (d.isNull(i)) {
                        g.nulls += 1;
                        continue;
                    }
                    g.count += 1;
                    self.offer(g, .{ .batch = b, .row = i });
                }
            }
        };

        fn resultType(args: []const Datum, _: Options) KernelError!DataType {
            if (args.len != 2) return error.InvalidArity;
            if (which == .min_max) return error.NotImplemented;
            return args[0].dataType();
        }

        pub const resultTypeAlloc = if (which != .min_max) {} else struct {
            fn f(allocator: std.mem.Allocator, args: []const Datum, _: Options) KernelError!DataType {
                if (args.len != 2) return error.InvalidArity;
                return dense.structTypeAlloc(allocator, &aggregate_basic.min_max_names, &.{ args[0].dataType(), args[0].dataType() });
            }
        }.f;

        fn exec(ctx: *ExecContext, args: []const Datum, options: Options) KernelError!Datum {
            var state: State = .{ .allocator = ctx.allocator };
            defer state.deinit();
            try forEachPiece(ctx.allocator, args, false, true, &state.seen, &state);
            if (state.value_type == null and args.len > 0) state.value_type = args[0].dataType();
            return .{ .array = try finish(ctx, &state, options) };
        }

        fn init(ctx: *ExecContext, _: Options) KernelError!*anyopaque {
            const state = ctx.allocator.create(State) catch return error.OutOfMemory;
            state.* = .{ .allocator = ctx.allocator };
            return state;
        }

        fn update(_: *ExecContext, state: *anyopaque, args: []const Datum, _: Options) KernelError!void {
            const s = cast(state);
            try forEachPiece(s.allocator, args, false, false, &s.seen, s);
        }

        /// Append the batches of `other` (same group ids) and offer its picks to every group.
        fn merge(_: *ExecContext, state: *anyopaque, other: *anyopaque, _: Options) KernelError!void {
            const s = cast(state);
            const o = cast(other);
            const base: u32 = @intCast(s.batches.items.len);
            for (o.batches.items) |b| _ = try s.addBatch(b);
            if (o.groups.items.len > s.groups.items.len) s.groups.appendNTimes(s.allocator, .{}, o.groups.items.len - s.groups.items.len) catch return error.OutOfMemory;
            for (s.groups.items[0..o.groups.items.len], o.groups.items) |*a, b| {
                inline for (.{ "lo", "hi" }) |f| if (@field(b, f)) |at| s.offer(a, .{ .batch = base + at.batch, .row = at.row });
                a.count += b.count;
                a.nulls += b.nulls;
            }
            s.seen.rows += o.seen.rows;
            s.seen.groups = @max(s.seen.groups, o.seen.groups);
        }

        fn finalize(ctx: *ExecContext, state: *anyopaque, options: Options) KernelError!Datum {
            return .{ .array = try finish(ctx, cast(state), options) };
        }

        fn deinit(ctx: *ExecContext, state: *anyopaque) void {
            const s = cast(state);
            s.deinit();
            ctx.allocator.destroy(s);
        }

        fn cast(state: *anyopaque) *State {
            return @ptrCast(@alignCast(state));
        }

        fn finish(ctx: *ExecContext, s: *State, options: Options) KernelError!ArrayRef {
            const allocator = ctx.allocator;
            const n = s.seen.groups;
            if (s.groups.items.len < n) s.groups.appendNTimes(allocator, .{}, n - s.groups.items.len) catch return error.OutOfMemory;
            var all = try concatBatches(allocator, s.value_type, s.batches.items);
            defer all.release();
            // Global row of the first row of every batch.
            const bases = allocator.alloc(usize, s.batches.items.len) catch return error.OutOfMemory;
            defer allocator.free(bases);
            var total: usize = 0;
            for (bases, s.batches.items) |*b, batch| {
                b.* = total;
                total += batch.data().length;
            }
            const o = scalarOptions(options);
            const indices = allocator.alloc(?usize, n) catch return error.OutOfMemory;
            defer allocator.free(indices);
            var parts: [2]ArrayRef = undefined;
            var made: usize = 0;
            errdefer for (parts[0..made]) |*p| p.release();
            inline for (.{ "lo", "hi" }, 0..) |f, k| {
                if ((which == .min and k == 1) or (which == .max and k == 0)) continue;
                for (indices, s.groups.items[0..n]) |*ix, g| {
                    const at = @field(g, f);
                    ix.* = if (at == null or nullBySkipNulls(o, g.nulls)) null else bases[at.?.batch] + at.?.row;
                }
                parts[made] = if (n == 0) all.slice(0, 0) catch return error.OutOfMemory else try takeNullableArray(all, indices);
                made += 1;
            }
            if (which != .min_max) return parts[0];
            made = 0;
            return dense.structArray(allocator, &aggregate_basic.min_max_names, &parts, n, null);
        }
    };
}

/// `nullByOptions` of hash_min / hash_max / hash_min_max: `min_count` is not applied, as Arrow's
/// grouped min / max (pyarrow 26 `group_by` max with `min_count=2` on [1, null] gives 1).
fn nullBySkipNulls(o: core.ScalarAggregateOptions, nulls: u64) bool {
    return !o.skip_nulls and nulls > 0;
}

/// A float key compared by value: every NaN one key and -0.0 equal to 0.0 (see `KeyMap.fold_floats`).
fn foldFloat(comptime k: anytype, v: anytype) u64 {
    const F = switch (k) {
        .f16 => f16,
        .f32 => f32,
        .f64 => f64,
        else => return v,
    };
    const x: F = @bitCast(v);
    if (std.math.isNan(x)) return @as(@TypeOf(v), @bitCast(std.math.nan(F)));
    if (x == 0) return 0;
    return v;
}
