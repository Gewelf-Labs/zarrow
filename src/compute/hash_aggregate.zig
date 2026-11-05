//! Grouped aggregation (Arrow C++ "Grouped aggregations", the `hash_*` functions). A `Grouper`
//! maps the rows of one or more key columns to dense uint32 group ids (first-seen order; a null
//! key is a group of its own, as in Arrow); the `hash_*` kernels (`FunctionKind.hash_aggregate`)
//! take a value array and the group id array and return one row per group (as many groups as the
//! largest id plus one); `groupBy` runs both and returns a record batch of the key columns
//! followed by one column per aggregate.
//!
//! Functions: hash_sum, hash_mean, hash_min, hash_max, hash_min_max (int8..int64, uint8..uint64,
//! float32/64, the output types and NaN / overflow rules of sum / mean / min / max / min_max);
//! hash_count (valid rows) for every type with a validity bitmap; hash_count_all (rows per group;
//! one argument, the group ids, like Arrow's nullary hash_count_all); hash_any, hash_all (bool);
//! hash_count_distinct and hash_distinct (`list<T>`, first-seen order) for every key type. Nulls
//! are skipped and a group with no valid value is null (Arrow's `ScalarAggregateOptions` /
//! `CountOptions` defaults: skip_nulls, min_count 1, only_valid; the options are not exposed).
//! Each kernel has an `AggregateLifecycle`: `update` takes (values, group ids) batches whose ids
//! may grow, `merge` adds another state over the same group ids, `finalize` returns the groups
//! seen so far. A one-shot call (`invoke`) requires every group id to be below its number of rows
//! (dense ids from a `Grouper` always are), so its state is bounded by the input size; a
//! lifecycle `update` accepts any id (its state grows to the largest id).
//!
//! Key types (Grouper, KeyMap, the distinct functions and set_lookup.zig): int8..int64,
//! uint8..uint64, float32/64, bool, date32, date64, timestamp, time32, time64, duration and the
//! string / binary layouts. Floats compare by bits after folding every NaN into one (NaN matches
//! NaN, -0.0 differs from 0.0, as in Arrow).
const std = @import("std");
const bitmap = @import("../bitmap.zig");
const core = @import("core.zig");
const datatype = @import("../datatype.zig");
const dense = @import("dense.zig");
const is_null = @import("is_null.zig");
const concat = @import("../concat_array_refs.zig");
const RecordBatch = @import("../record_batch.zig").RecordBatch;
const schema_mod = @import("../schema.zig");

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

/// How the rows of a key type are read.
const KeyKind = enum { b8, b16, b32, b64, f32, f64, boolean, bytes };

fn keyKind(dt: DataType) ?KeyKind {
    return switch (dt) {
        .int8, .uint8 => .b8,
        .int16, .uint16 => .b16,
        .int32, .uint32, .date32, .time32 => .b32,
        .int64, .uint64, .date64, .timestamp, .time64, .duration => .b64,
        .float => .f32,
        .double => .f64,
        .bool => .boolean,
        .string, .large_string, .string_view, .binary, .large_binary, .binary_view => .bytes,
        else => null,
    };
}

/// Return whether `dt` can be a key (Grouper, KeyMap, is_in / index_in, hash_count_distinct).
pub fn isKeyType(dt: DataType) bool {
    return keyKind(dt) != null;
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
        switch (keyKind(data.data_type) orelse return error.UnsupportedType) {
            inline .b8, .b16, .b32, .b64, .f32, .f64 => |k| {
                const U = switch (k) {
                    .b8 => u8,
                    .b16 => u16,
                    .b32, .f32 => u32,
                    else => u64,
                };
                const vals = (PrimitiveArray(U){ .data = data }).values() catch return error.InvalidInput;
                for (vals, ids, 0..) |v, *id, i| {
                    if (bits) |b| if (!bitmap.bitIsSet(b, data.offset + i)) {
                        id.* = try self.nullId(mode);
                        continue;
                    };
                    var key: u64 = v;
                    if (k == .f32 and std.math.isNan(@as(f32, @bitCast(v)))) key = 0x7FC0_0000;
                    if (k == .f64 and std.math.isNan(@as(f64, @bitCast(v)))) key = 0x7FF8_0000_0000_0000;
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

    /// A grouper for keys of `key_types` (at least one, each `isKeyType`).
    pub fn init(allocator: std.mem.Allocator, key_types: []const DataType) KernelError!Grouper {
        if (key_types.len == 0) return error.InvalidArity;
        for (key_types) |t| if (!isKeyType(t)) return error.UnsupportedType;
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
    /// column, equal lengths, the types given to `init`).
    pub fn consume(self: *Grouper, keys: []const ArrayRef) KernelError!ArrayRef {
        if (keys.len != self.columns.len) return error.InvalidArity;
        const len = keys[0].data().length;
        for (keys, self.key_types) |k, t| {
            if (!k.data().data_type.eql(t)) return error.InvalidInput;
            if (k.data().length != len) return error.InvalidInput;
            k.data().validateLayout() catch return error.InvalidInput;
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
        try add(registry, "hash_sum", withIds(tag), Grouped(Sum(T, .sum)));
        try add(registry, "hash_mean", withIds(tag), Grouped(Sum(T, .mean)));
        try add(registry, "hash_min", withIds(tag), Grouped(MinMax(T, .min)));
        try add(registry, "hash_max", withIds(tag), Grouped(MinMax(T, .max)));
        try add(registry, "hash_min_max", withIds(tag), Grouped(MinMax(T, .min_max)));
    }
    try add(registry, "hash_count", countArgs, Grouped(Count(.count)));
    try add(registry, "hash_count_all", idsOnly, Grouped(Count(.count_all)));
    try add(registry, "hash_any", withIds(.bool), Grouped(AnyAll(.any)));
    try add(registry, "hash_all", withIds(.bool), Grouped(AnyAll(.all)));
    try add(registry, "hash_count_distinct", distinctArgs, Distinct(.count));
    try add(registry, "hash_distinct", distinctArgs, Distinct(.list));
}

fn add(registry: *FunctionRegistry, comptime name: []const u8, comptime check: TypeCheckFn, comptime A: type) KernelError!void {
    try registry.registerKernel(name, .hash_aggregate, .{
        .signature = .{ .arity = if (A.ids_only) 1 else 2, .type_check = check, .options_check = dense.noOptions, .result_type_fn = A.resultType },
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
    return args.len == 2 and isKeyType(args[0].dataType()) and args[1].dataType() == .uint32;
}

/// Group ids seen so far by a lifecycle state: their count bounds the ids (see the file doc).
const Seen = struct {
    rows: u64 = 0,
    /// Groups: the largest id plus one.
    groups: usize = 0,
};

/// Call `sink.piece(values, ids, groups)` for every aligned piece of `args` (values then group
/// ids, or the ids alone; arrays or chunked arrays): every chunk passes `validateLayout` and the
/// ids have no null. With `bounded`, every id must be below the number of rows of `args`.
fn forEachPiece(allocator: std.mem.Allocator, args: []const Datum, comptime ids_only: bool, comptime bounded: bool, seen: *Seen, sink: anytype) KernelError!void {
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

            fn grow(self: *State, n: usize) KernelError!void {
                if (n > self.groups.items.len) self.groups.appendNTimes(self.allocator, .{}, n - self.groups.items.len) catch return error.OutOfMemory;
            }

            pub fn piece(self: *State, values: ?ArrayRef, ids: []const u32, groups: usize) KernelError!void {
                try self.grow(groups);
                try Impl.update(self.groups.items, if (values) |v| v.data() else null, ids);
            }
        };

        fn resultType(_: []const Datum, _: Options) KernelError!DataType {
            return Impl.out_type;
        }

        /// Aggregate the values of every group: one row per group id.
        fn exec(ctx: *ExecContext, args: []const Datum, _: Options) KernelError!Datum {
            var state: State = .{ .allocator = ctx.allocator };
            defer state.groups.deinit(ctx.allocator);
            try forEachPiece(ctx.allocator, args, ids_only, true, &state.seen, &state);
            return .{ .array = try Impl.finalize(ctx, state.groups.items) };
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
        }

        fn finalize(ctx: *ExecContext, state: *anyopaque, _: Options) KernelError!Datum {
            return .{ .array = try Impl.finalize(ctx, cast(state).groups.items) };
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

/// Run `f(group, value)` for every valid row.
inline fn eachValid(comptime T: type, data: *const ArrayData, ids: []const u32, groups: anytype, comptime f: anytype) KernelError!void {
    const vals = try typedValues(T, data);
    if (dense.validityBits(data)) |bits| {
        for (vals, ids, 0..) |v, id, i| if (bitmap.bitIsSet(bits, data.offset + i)) f(&groups[id], v);
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
        const Group = struct { sum: Exact = 0, count: u64 = 0 };
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
        }

        fn finalize(ctx: *ExecContext, groups: []const Group) KernelError!ArrayRef {
            return fixedOutput(Out, ctx.allocator, out_type, groups.len, .{ ctx, groups }, struct {
                fn fill(c: anytype, out: []Out, valid: []u8) KernelError!void {
                    for (c[1], out, 0..) |g, *o, i| {
                        if (g.count == 0) {
                            bitmap.clearBit(valid, i);
                            o.* = 0;
                            continue;
                        }
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
        const Group = struct { min: T = min_start, max: T = max_start, count: u64 = 0 };
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
        }

        fn side(ctx: *ExecContext, groups: []const Group, comptime is_min: bool) KernelError!ArrayRef {
            return fixedOutput(T, ctx.allocator, elem_type, groups.len, groups, struct {
                fn fill(gs: []const Group, out: []T, valid: []u8) KernelError!void {
                    for (gs, out, 0..) |g, *o, i| {
                        o.* = if (is_min) g.min else g.max;
                        if (g.count == 0) {
                            bitmap.clearBit(valid, i);
                            o.* = 0;
                        }
                    }
                }
            }.fill);
        }

        fn finalize(ctx: *ExecContext, groups: []const Group) KernelError!ArrayRef {
            if (which != .min_max) return side(ctx, groups, which == .min);
            var mins = try side(ctx, groups, true);
            var maxs = side(ctx, groups, false) catch |err| {
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
        const Group = struct { n: u64 = 0 };
        const out_type = int64_type;

        fn update(groups: []Group, data: ?*const ArrayData, ids: []const u32) KernelError!void {
            if (which == .count_all) {
                for (ids) |id| groups[id].n += 1;
                return;
            }
            const d = data.?;
            if (datatype.storageDataType(d.data_type) == .null) return;
            if (dense.validityBits(d)) |bits| {
                for (ids, 0..) |id, i| groups[id].n += @intFromBool(bitmap.bitIsSet(bits, d.offset + i));
            } else for (ids) |id| groups[id].n += 1;
        }

        fn mergeGroup(a: *Group, b: Group) void {
            a.n += b.n;
        }

        fn finalize(ctx: *ExecContext, groups: []const Group) KernelError!ArrayRef {
            return fixedOutput(i64, ctx.allocator, out_type, groups.len, groups, struct {
                fn fill(gs: []const Group, out: []i64, _: []u8) KernelError!void {
                    for (gs, out) |g, *o| o.* = std.math.cast(i64, g.n) orelse return error.Overflow;
                }
            }.fill);
        }
    };
}

fn AnyAll(comptime which: enum { any, all }) type {
    return struct {
        const ids_only = false;
        const Group = struct { any_true: bool = false, any_false: bool = false, count: u64 = 0 };
        const out_type = bool_type;

        fn update(groups: []Group, data: ?*const ArrayData, ids: []const u32) KernelError!void {
            const d = data.?;
            const bits = dense.validityBits(d);
            for (ids, 0..) |id, i| {
                if (bits) |b| if (!bitmap.bitIsSet(b, d.offset + i)) continue;
                const g = &groups[id];
                if (bitmap.bitIsSet(d.buffers[1].data, d.offset + i)) g.any_true = true else g.any_false = true;
                g.count += 1;
            }
        }

        fn mergeGroup(a: *Group, b: Group) void {
            a.any_true = a.any_true or b.any_true;
            a.any_false = a.any_false or b.any_false;
            a.count += b.count;
        }

        fn finalize(ctx: *ExecContext, groups: []const Group) KernelError!ArrayRef {
            return fixedOutput(bool, ctx.allocator, out_type, groups.len, groups, struct {
                fn fill(gs: []const Group, out: []u8, valid: []u8) KernelError!void {
                    for (gs, 0..) |g, i| {
                        if (g.count == 0) {
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

/// hash_count_distinct / hash_distinct: a `Grouper` over (group id, value) pairs; pairs whose
/// value is null are left out (Arrow's `CountOptions` default `only_valid`).
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
            });
        }

        fn exec(ctx: *ExecContext, args: []const Datum, _: Options) KernelError!Datum {
            var state: State = .{ .allocator = ctx.allocator };
            defer state.deinit();
            try forEachPiece(ctx.allocator, args, false, true, &state.seen, &state);
            return .{ .array = try finish(ctx, &state, args[0].dataType()) };
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

        fn finalize(ctx: *ExecContext, state: *anyopaque, _: Options) KernelError!Datum {
            const s = cast(state);
            return .{ .array = try finish(ctx, s, s.value_type orelse .{ .null = {} }) };
        }

        fn deinit(ctx: *ExecContext, state: *anyopaque) void {
            const s = cast(state);
            s.deinit();
            ctx.allocator.destroy(s);
        }

        fn cast(state: *anyopaque) *State {
            return @ptrCast(@alignCast(state));
        }

        /// Count (or list) the non-null distinct values of every group.
        fn finish(ctx: *ExecContext, s: *State, value_type: DataType) KernelError!ArrayRef {
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
                if (!keys[1].data().isNull(i)) counts[g + 1] += 1;
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
                if (keys[1].data().isNull(i)) continue;
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

fn listOver(value_type: *const DataType) DataType {
    return .{ .list = .{ .value_field = .{ .name = "item", .data_type = value_type } } };
}

/// Rows `indices` of `arr` as a new array (an empty one of its type for no index).
fn takeArray(allocator: std.mem.Allocator, arr: ArrayRef, indices: []const usize) KernelError!ArrayRef {
    if (indices.len == 0) return dense.nullArray(allocator, arr.data().data_type, 0);
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
