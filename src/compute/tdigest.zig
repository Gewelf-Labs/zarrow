//! Approximate quantiles (Arrow C++ "Aggregations" / "Grouped aggregations"): tdigest and
//! approximate_median (aggregates), hash_tdigest and hash_approximate_median (grouped), over
//! int8..int64, uint8..uint64, float32/64 and decimal128 / decimal256 (as float64).
//!
//! `TDigest` is Arrow's merging t-digest (arrow/util/tdigest.cc): values are buffered
//! (`buffer_size`), sorted and merged into at most about `delta` centroids under the arcsine
//! scale function k1; digests merge with a k-way merge of their centroids; a quantile
//! interpolates between the centroids around it (exact at the minimum and maximum). The kernels
//! consume the way Arrow's executor does (each chunk, or `update` batch, into a digest of its
//! own that is then merged into the state's), so the centroids, and the results, follow
//! pyarrow 26 (within 1e-9 relative on the tests; the arcsine / sine of another libm may move a
//! centroid boundary).
//!
//! - tdigest (`TDigestOptions`: q, delta, buffer_size, skip_nulls, min_count): float64 array of
//!   one quantile per `q`, all null when there is no value, fewer valid rows than `min_count`, or
//!   a null row without `skip_nulls`. NaN values are skipped but count as valid rows (Arrow).
//! - approximate_median (`ScalarAggregateOptions`): float64 scalar, tdigest at q 0.5 with
//!   delta 100 and buffer_size 500.
//! - hash_tdigest: `fixed_size_list<double>[q.len]` per group, a group without result holds
//!   null quantiles (the list itself is valid, as Arrow); hash_approximate_median: float64.
//!
//! Options are bounded so they cannot force a huge allocation: every q in [0, 1] (Arrow gives
//! NaN outside; here `InvalidOptions`), at most 65536 quantiles, delta <= 2^20, buffer_size <=
//! 2^24; a digest never holds more than its values anyway.
const std = @import("std");
const bitmap = @import("../bitmap.zig");
const core = @import("core.zig");
const decimal = @import("decimal.zig");
const dense = @import("dense.zig");
const basic = @import("aggregate_basic.zig");
const hash = @import("hash_aggregate.zig");

const ArrayData = @import("../array/array_data.zig").ArrayData;
const OwnedBuffer = @import("../buffer.zig").OwnedBuffer;
const SharedBuffer = @import("../buffer.zig").SharedBuffer;
const ArrayRef = core.ArrayRef;
const DataType = core.DataType;
const Datum = core.Datum;
const ExecContext = core.ExecContext;
const FunctionRegistry = core.FunctionRegistry;
const KernelError = core.KernelError;
const Options = core.Options;
const Scalar = core.Scalar;

const f64_type = DataType{ .double = {} };

pub const max_quantiles = 1 << 16;
pub const max_delta = 1 << 20;
pub const max_buffer_size = 1 << 24;

// ------------------------------------------------------------------------------ t-digest

pub const Centroid = struct {
    mean: f64,
    weight: f64,

    fn merge(self: *Centroid, other: Centroid) void {
        self.weight += other.weight;
        self.mean += (other.mean - self.mean) * other.weight / self.weight;
    }
};

/// Feeds sorted centroids into a digest under the k1 scale function: a centroid joins the last
/// one while the weight so far stays within the limit, else it starts a new one.
const Merger = struct {
    delta_norm: f64,
    total_weight: f64 = 0,
    weight_so_far: f64 = 0,
    weight_limit: f64 = -1,

    fn init(delta: u32) Merger {
        return .{ .delta_norm = @as(f64, @floatFromInt(delta)) / (2 * std.math.pi) };
    }

    fn k(self: *const Merger, q: f64) f64 {
        return self.delta_norm * std.math.asin(2 * q - 1);
    }

    fn quantileOf(self: *const Merger, kv: f64) f64 {
        return (@sin(kv / self.delta_norm) + 1) / 2;
    }

    fn add(self: *Merger, allocator: std.mem.Allocator, out: *std.ArrayList(Centroid), c: Centroid) KernelError!void {
        const weight = self.weight_so_far + c.weight;
        if (weight <= self.weight_limit) {
            out.items[out.items.len - 1].merge(c);
        } else {
            const quantile = self.weight_so_far / self.total_weight;
            const next = self.total_weight * self.quantileOf(self.k(quantile) + 1);
            // The limit grows strictly until the last centroid.
            self.weight_limit = if (next <= self.weight_limit) self.total_weight else next;
            out.append(allocator, c) catch return error.OutOfMemory;
        }
        self.weight_so_far = weight;
    }
};

/// Arrow's merging t-digest (arrow/util/tdigest.cc).
pub const TDigest = struct {
    delta: u32,
    buffer_size: usize,
    input: std.ArrayList(f64) = .empty,
    /// Ping-pong centroid buffers: `current` holds the digest.
    digests: [2]std.ArrayList(Centroid) = .{ .empty, .empty },
    current: u1 = 0,
    total_weight: f64 = 0,
    min: f64 = std.math.floatMax(f64),
    max: f64 = -std.math.floatMax(f64),

    pub fn init(delta: u32, buffer_size: u32) TDigest {
        return .{ .delta = @max(delta, 10), .buffer_size = @max(buffer_size, 1) };
    }

    pub fn deinit(self: *TDigest, allocator: std.mem.Allocator) void {
        self.input.deinit(allocator);
        for (&self.digests) |*d| d.deinit(allocator);
    }

    /// No value added.
    pub fn isEmpty(self: *const TDigest) bool {
        return self.input.items.len == 0 and self.total_weight == 0;
    }

    /// Add a value (NaN is skipped).
    pub fn add(self: *TDigest, allocator: std.mem.Allocator, v: f64) KernelError!void {
        if (std.math.isNan(v)) return;
        if (self.input.items.len >= self.buffer_size) try self.flush(allocator);
        self.input.append(allocator, v) catch return error.OutOfMemory;
    }

    /// Merge the buffered values into the centroids.
    pub fn flush(self: *TDigest, allocator: std.mem.Allocator) KernelError!void {
        const input = self.input.items;
        if (input.len == 0) return;
        self.total_weight += @floatFromInt(input.len);
        std.sort.pdq(f64, input, {}, std.sort.asc(f64));
        self.min = @min(self.min, input[0]);
        self.max = @max(self.max, input[input.len - 1]);
        var merger = Merger.init(self.delta);
        merger.total_weight = self.total_weight;
        const td = self.digests[self.current].items;
        const out = &self.digests[1 - self.current];
        out.clearRetainingCapacity();
        var i: usize = 0;
        var j: usize = 0;
        while (i < td.len and j < input.len) {
            if (td[i].mean < input[j]) {
                try merger.add(allocator, out, td[i]);
                i += 1;
            } else {
                try merger.add(allocator, out, .{ .mean = input[j], .weight = 1 });
                j += 1;
            }
        }
        for (td[i..]) |c| try merger.add(allocator, out, c);
        for (input[j..]) |v| try merger.add(allocator, out, .{ .mean = v, .weight = 1 });
        self.input.clearRetainingCapacity();
        self.current = 1 - self.current;
    }

    /// Merge `other` into this digest (both are flushed first): the centroids of both, by mean
    /// (this digest's first on a tie), through the merger.
    pub fn merge(self: *TDigest, allocator: std.mem.Allocator, other: *TDigest) KernelError!void {
        try self.flush(allocator);
        try other.flush(allocator);
        const a = self.digests[self.current].items;
        const b = other.digests[other.current].items;
        if (b.len > 0) {
            self.total_weight += other.total_weight;
            self.min = @min(self.min, other.min);
            self.max = @max(self.max, other.max);
        }
        var merger = Merger.init(self.delta);
        merger.total_weight = self.total_weight;
        const out = &self.digests[1 - self.current];
        out.clearRetainingCapacity();
        var i: usize = 0;
        var j: usize = 0;
        while (i < a.len and j < b.len) {
            if (b[j].mean < a[i].mean) {
                try merger.add(allocator, out, b[j]);
                j += 1;
            } else {
                try merger.add(allocator, out, a[i]);
                i += 1;
            }
        }
        for (a[i..]) |c| try merger.add(allocator, out, c);
        for (b[j..]) |c| try merger.add(allocator, out, c);
        self.current = 1 - self.current;
    }

    /// The approximate `q` quantile (flushes first); NaN when empty.
    pub fn quantile(self: *TDigest, allocator: std.mem.Allocator, q: f64) KernelError!f64 {
        try self.flush(allocator);
        const td = self.digests[self.current].items;
        if (q < 0 or q > 1 or td.len == 0) return std.math.nan(f64);
        const index = q * self.total_weight;
        if (index <= 1) return self.min;
        if (index >= self.total_weight - 1) return self.max;
        // The centroid holding the index.
        var ci: usize = 0;
        var weight_sum: f64 = 0;
        while (ci < td.len) : (ci += 1) {
            weight_sum += td[ci].weight;
            if (index <= weight_sum) break;
        }
        if (ci == td.len) return self.max;
        // Deviation of the index from the centroid's center.
        var diff = index + td[ci].weight / 2 - weight_sum;
        if (td[ci].weight == 1 and @abs(diff) < 0.5) return td[ci].mean;
        var left = ci;
        var right = ci;
        if (diff > 0) {
            if (right == td.len - 1) return lerp(td[right].mean, self.max, diff / (td[right].weight / 2));
            right += 1;
        } else {
            if (left == 0) return lerp(self.min, td[0].mean, index / (td[0].weight / 2));
            left -= 1;
            diff += td[left].weight / 2 + td[right].weight / 2;
        }
        diff /= td[left].weight / 2 + td[right].weight / 2;
        return lerp(td[left].mean, td[right].mean, diff);
    }
};

fn lerp(a: f64, b: f64, t: f64) f64 {
    return a + t * (b - a);
}

// ------------------------------------------------------------------------------ options

/// The digest parameters of a call: `TDigestOptions` (tdigest), or `ScalarAggregateOptions`
/// with q 0.5, delta 100, buffer_size 500 (approximate_median, as Arrow).
const Params = struct {
    q: []const f64,
    delta: u32,
    buffer_size: u32,
    skip_nulls: bool,
    min_count: u32,

    fn of(options: Options, comptime median: bool) Params {
        if (median) {
            const o: core.ScalarAggregateOptions = if (options == .scalar_aggregate) options.scalar_aggregate else .{};
            return .{ .q = &.{0.5}, .delta = 100, .buffer_size = 500, .skip_nulls = o.skip_nulls, .min_count = o.min_count };
        }
        const o: core.TDigestOptions = if (options == .tdigest) options.tdigest else .{};
        return .{ .q = o.q, .delta = o.delta, .buffer_size = o.buffer_size, .skip_nulls = o.skip_nulls, .min_count = o.min_count };
    }
};

fn tdigestOptions(options: Options) bool {
    if (options == .none) return true;
    if (options != .tdigest) return false;
    const o = options.tdigest;
    if (o.q.len > max_quantiles or o.delta > max_delta or o.buffer_size > max_buffer_size) return false;
    for (o.q) |q| if (!(q >= 0 and q <= 1)) return false;
    return true;
}

fn medianOptions(options: Options) bool {
    return options == .none or options == .scalar_aggregate;
}

inline fn toDouble(comptime T: type, v: T, scale: i32) f64 {
    if (T == i128 or T == i256) return decimal.toFloatValue(f64, v, scale);
    return if (@typeInfo(T) == .float) @floatCast(v) else @floatFromInt(v);
}

fn scaleOf(dt: DataType) i32 {
    return switch (dt) {
        .decimal128, .decimal256 => |d| d.scale,
        else => 0,
    };
}

// ------------------------------------------------------------------------------ aggregates

/// tdigest (`median` false) and approximate_median over values of `T`.
fn Digest(comptime T: type, comptime median: bool) type {
    return struct {
        pub const State = struct {
            allocator: std.mem.Allocator,
            digest: TDigest,
            count: u64 = 0,
            all_valid: bool = true,
            skip_nulls: bool,
        };
        pub const out_type = f64_type;

        pub fn initState(ctx: *ExecContext, options: Options) KernelError!State {
            const p = Params.of(options, median);
            return .{ .allocator = ctx.allocator, .digest = TDigest.init(p.delta, p.buffer_size), .skip_nulls = p.skip_nulls };
        }

        pub fn deinitState(state: *State) void {
            state.digest.deinit(state.allocator);
        }

        /// One chunk: a digest of its own, merged into the state's (Arrow's executor gives every
        /// batch a fresh state and merges it), unless the chunk is empty.
        pub fn update(state: *State, data: *const ArrayData) KernelError!void {
            if (data.length == 0 or !state.all_valid) return;
            if (!state.skip_nulls and data.hasNulls()) {
                state.all_valid = false;
                return;
            }
            var batch = TDigest.init(state.digest.delta, @intCast(state.digest.buffer_size));
            defer batch.deinit(state.allocator);
            const vals = try basic.values(T, data);
            const bits = dense.validityBits(data);
            const scale = scaleOf(data.data_type);
            for (vals, 0..) |v, i| {
                if (bits) |b| if (!bitmap.bitIsSet(b, data.offset + i)) continue;
                try batch.add(state.allocator, toDouble(T, v, scale));
                state.count += 1;
            }
            try state.digest.merge(state.allocator, &batch);
        }

        pub fn updateScalar(state: *State, s: Scalar) KernelError!void {
            if (!state.all_valid) return;
            if (s.isNull()) {
                if (!state.skip_nulls) state.all_valid = false;
                return;
            }
            var batch = TDigest.init(state.digest.delta, @intCast(state.digest.buffer_size));
            defer batch.deinit(state.allocator);
            const x = if (T == i128 or T == i256)
                decimal.toFloatValue(f64, try decimal.scalarInt(T, s), (try decimal.checkedParams(s.data_type)).scale)
            else
                toDouble(T, try dense.scalarAs(T, s), 0);
            try batch.add(state.allocator, x);
            state.count += 1;
            try state.digest.merge(state.allocator, &batch);
        }

        pub fn merge(state: *State, other: *State) KernelError!void {
            if (!state.all_valid or !other.all_valid) {
                state.all_valid = false;
                return;
            }
            try state.digest.merge(state.allocator, &other.digest);
            state.count += other.count;
        }

        pub fn finalize(ctx: *ExecContext, state: *State, options: Options) KernelError!Datum {
            const p = Params.of(options, median);
            const none = state.digest.isEmpty() or !state.all_valid or state.count < p.min_count;
            if (median) {
                if (none) return .{ .scalar = Scalar.init(f64_type, .null) };
                return .{ .scalar = Scalar.init(f64_type, .{ .f64 = try state.digest.quantile(state.allocator, 0.5) }) };
            }
            const n = p.q.len;
            var buf = OwnedBuffer.initUninit(ctx.allocator, n * 8) catch return error.OutOfMemory;
            const out = buf.typedSlice(f64)[0..n];
            var validity: ?OwnedBuffer = null;
            if (none) {
                @memset(out, 0);
                var v = dense.bitsBuffer(ctx.allocator, n) catch |err| {
                    buf.deinit();
                    return err;
                };
                @memset(v.data[0..bitmap.byteLength(n)], 0);
                validity = v;
            } else for (out, p.q) |*o, q| {
                o.* = state.digest.quantile(state.allocator, q) catch |err| {
                    buf.deinit();
                    return err;
                };
            }
            return .{ .array = try dense.finishFixed(ctx.allocator, f64_type, n, &buf, validity) };
        }
    };
}

// ------------------------------------------------------------------------------ grouped

/// hash_tdigest (`median` false) and hash_approximate_median over values of `T`: one digest per
/// group, values added in row order (Arrow's grouped t-digest; no per-batch merge).
fn GroupedDigest(comptime T: type, comptime median: bool) type {
    return struct {
        pub const ids_only = false;

        const Group = struct {
            digest: TDigest,
            count: u64 = 0,
            has_null: bool = false,
        };

        const State = struct {
            allocator: std.mem.Allocator,
            groups: std.ArrayList(Group) = .empty,
            seen: hash.Seen = .{},
            delta: u32,
            buffer_size: u32,

            fn init(allocator: std.mem.Allocator, options: Options) State {
                const p = Params.of(options, median);
                return .{ .allocator = allocator, .delta = p.delta, .buffer_size = p.buffer_size };
            }

            fn deinit(self: *State) void {
                for (self.groups.items) |*g| g.digest.deinit(self.allocator);
                self.groups.deinit(self.allocator);
            }

            fn grow(self: *State, n: usize) KernelError!void {
                if (n <= self.groups.items.len) return;
                self.groups.ensureTotalCapacity(self.allocator, n) catch return error.OutOfMemory;
                while (self.groups.items.len < n) self.groups.appendAssumeCapacity(.{ .digest = TDigest.init(self.delta, self.buffer_size) });
            }

            pub fn piece(self: *State, values: ?ArrayRef, ids: []const u32, groups: usize) KernelError!void {
                try self.grow(groups);
                const data = values.?.data();
                const vals = try basic.values(T, data);
                const bits = dense.validityBits(data);
                const scale = scaleOf(data.data_type);
                for (vals, ids, 0..) |v, id, i| {
                    const g = &self.groups.items[id];
                    if (bits) |b| if (!bitmap.bitIsSet(b, data.offset + i)) {
                        g.has_null = true;
                        continue;
                    };
                    try g.digest.add(self.allocator, toDouble(T, v, scale));
                    g.count += 1;
                }
            }
        };

        pub fn resultType(_: []const Datum, options: Options) KernelError!DataType {
            if (median) return f64_type;
            const n = Params.of(options, false).q.len;
            return .{ .fixed_size_list = .{ .value_field = .{ .name = "item", .data_type = &f64_type }, .list_size = @intCast(n) } };
        }

        pub fn exec(ctx: *ExecContext, args: []const Datum, options: Options) KernelError!Datum {
            var state = State.init(ctx.allocator, options);
            defer state.deinit();
            try hash.forEachPiece(ctx.allocator, args, false, true, &state.seen, &state);
            return .{ .array = try finish(ctx, &state, options) };
        }

        pub fn init(ctx: *ExecContext, options: Options) KernelError!*anyopaque {
            const state = ctx.allocator.create(State) catch return error.OutOfMemory;
            state.* = State.init(ctx.allocator, options);
            return state;
        }

        pub fn update(_: *ExecContext, state: *anyopaque, args: []const Datum, _: Options) KernelError!void {
            const s = cast(state);
            try hash.forEachPiece(s.allocator, args, false, false, &s.seen, s);
        }

        pub fn merge(_: *ExecContext, state: *anyopaque, other: *anyopaque, _: Options) KernelError!void {
            const s = cast(state);
            const o = cast(other);
            try s.grow(o.groups.items.len);
            for (s.groups.items[0..o.groups.items.len], o.groups.items) |*a, *b| {
                try a.digest.merge(s.allocator, &b.digest);
                a.count += b.count;
                a.has_null = a.has_null or b.has_null;
            }
            s.seen.rows += o.seen.rows;
            s.seen.groups = @max(s.seen.groups, o.seen.groups);
        }

        pub fn finalize(ctx: *ExecContext, state: *anyopaque, options: Options) KernelError!Datum {
            return .{ .array = try finish(ctx, cast(state), options) };
        }

        pub fn deinit(ctx: *ExecContext, state: *anyopaque) void {
            const s = cast(state);
            s.deinit();
            ctx.allocator.destroy(s);
        }

        fn cast(state: *anyopaque) *State {
            return @ptrCast(@alignCast(state));
        }

        /// One row per group: the quantiles of a group with a result, nulls otherwise.
        fn finish(ctx: *ExecContext, s: *State, options: Options) KernelError!ArrayRef {
            const allocator = ctx.allocator;
            const p = Params.of(options, median);
            const groups = s.seen.groups;
            try s.grow(groups);
            const width = p.q.len;
            const n = std.math.mul(usize, groups, width) catch return error.Overflow;
            var buf = OwnedBuffer.initUninit(allocator, n * 8) catch return error.OutOfMemory;
            const out = buf.typedSlice(f64)[0..n];
            var validity = dense.bitsBuffer(allocator, n) catch |err| {
                buf.deinit();
                return err;
            };
            @memset(validity.data[0..bitmap.byteLength(n)], 0xFF);
            for (s.groups.items[0..groups], 0..) |*g, i| {
                const row = out[i * width ..][0..width];
                const ok = !g.digest.isEmpty() and g.count >= p.min_count and (p.skip_nulls or !g.has_null);
                if (!ok) {
                    @memset(row, 0);
                    for (0..width) |j| bitmap.clearBit(validity.data, i * width + j);
                    continue;
                }
                for (row, p.q) |*o, q| {
                    o.* = g.digest.quantile(allocator, q) catch |err| {
                        buf.deinit();
                        validity.deinit();
                        return err;
                    };
                }
            }
            var child = try dense.finishFixed(allocator, f64_type, n, &buf, validity);
            if (median) return child;
            errdefer child.release();
            var buffers = [1]SharedBuffer{.empty};
            var children = [1]ArrayRef{child};
            return ArrayRef.fromOwnedInline(allocator, .{
                // The value field points at the child's own type, which lives as long as the list.
                .data_type = .{ .fixed_size_list = .{ .value_field = .{ .name = "item", .data_type = &child.data().data_type }, .list_size = @intCast(width) } },
                .length = groups,
                .null_count = 0,
                .buffers = &buffers,
                .children = &children,
            }) catch error.OutOfMemory;
        }
    };
}

/// Register tdigest, approximate_median, hash_tdigest and hash_approximate_median.
pub fn register(registry: *FunctionRegistry) KernelError!void {
    inline for (dense.numeric_types ++ [_]type{ i128, i256 }) |T| {
        const tag = comptime if (T == i128) .decimal128 else if (T == i256) .decimal256 else std.meta.activeTag(dense.dataTypeOf(T));
        try basic.add(registry, "tdigest", 1, comptime dense.argsOfType(tag, 1), tdigestOptions, Digest(T, false));
        try basic.add(registry, "approximate_median", 1, comptime dense.argsOfType(tag, 1), medianOptions, Digest(T, true));
        const with_ids = comptime withIds(tag);
        try hash.add(registry, "hash_tdigest", with_ids, tdigestOptions, GroupedDigest(T, false));
        try hash.add(registry, "hash_approximate_median", with_ids, medianOptions, GroupedDigest(T, true));
    }
}

fn withIds(comptime tag: std.meta.Tag(DataType)) core.TypeCheckFn {
    return &struct {
        fn check(args: []const Datum) bool {
            return args.len == 2 and std.meta.activeTag(args[0].dataType()) == tag and args[1].dataType() == .uint32;
        }
    }.check;
}
