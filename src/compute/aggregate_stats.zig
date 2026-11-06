//! More scalar aggregates (Arrow C++ "Aggregations") and the vector aggregates mode and quantile:
//!
//! - product (`ScalarAggregateOptions`): int -> int64, uint -> uint64, float -> float64; an empty
//!   product is 1 with `min_count = 0`, null otherwise. Integer overflow follows
//!   `ExecConfig.overflow_mode` like sum (`checked`: `Overflow`, `wrapping`: the low 64 bits as
//!   Arrow C++, `saturating`: clamped with the sign of the exact product).
//! - variance, stddev (`VarianceOptions`: ddof, skip_nulls, min_count) and skew, kurtosis
//!   (`SkewOptions`: skip_nulls, biased, min_count) on numbers and decimal128 / decimal256, as
//!   float64: count, mean and central moments per chunk (two passes), merged across chunks and
//!   states with the pairwise update formulas (Chan, Pébay). A NaN makes the result NaN;
//!   `count <= ddof`, fewer valid rows than `min_count` or a null row without `skip_nulls` gives
//!   null; unbiased skew needs 3 rows, unbiased kurtosis 4 (else null); constant data gives NaN
//!   skew and kurtosis (0 / 0), as Arrow.
//! - first, last, first_last (`ScalarAggregateOptions`): the first / last non-null value (with
//!   `skip_nulls = false` the first / last row, which may be null); null with fewer valid rows
//!   than `min_count`. Numbers, bool, dates, timestamps, times, durations, decimals and the
//!   string / binary layouts. first_last is `struct<first, last>`.
//! - count_distinct (`CountOptions`): distinct values of a key type (hash_aggregate.zig); a null
//!   is one value (`only_null` counts 1 when there is one, `all` adds it).
//! - index (`IndexOptions`): int64 position of the first row equal to the value (cast to the
//!   input type), -1 when absent; a null or NaN value is never found (as Arrow). Key types.
//! - mode (vector, `ModeOptions`): `struct<mode: T, count: int64>`, the `n` most common values by
//!   count, ties by value (NaN last); empty when a null row and no `skip_nulls`, or fewer valid
//!   rows than `min_count`. Numbers, bool and decimals.
//! - quantile (vector, `QuantileOptions`): one row per `q` (each in [0, 1]); nulls and NaN are
//!   ignored; `linear` and `midpoint` give float64 (Arrow's formulas: `f * hi + (1 - f) * lo`,
//!   `lo / 2 + hi / 2`), `lower`, `higher` and `nearest` (ties to the even index) the input
//!   type; null rows when no value is left, fewer than `min_count`, or a null row without
//!   `skip_nulls`. Numbers and decimals.
//!
//! approximate_median and tdigest are not provided (their results depend on Arrow's T-Digest
//! merging; see the support matrix).
const std = @import("std");
const bitmap = @import("../bitmap.zig");
const buffer = @import("../buffer.zig");
const core = @import("core.zig");
const datatype = @import("../datatype.zig");
const decimal = @import("decimal.zig");
const dense = @import("dense.zig");
const basic = @import("aggregate_basic.zig");
const hash = @import("hash_aggregate.zig");

const ArrayData = @import("../array/array_data.zig").ArrayData;
const PrimitiveArray = @import("../array/primitive_array.zig").PrimitiveArray;
const ArrayRef = core.ArrayRef;
const DataType = core.DataType;
const Datum = core.Datum;
const ExecContext = core.ExecContext;
const FunctionRegistry = core.FunctionRegistry;
const KernelError = core.KernelError;
const Options = core.Options;
const Scalar = core.Scalar;
const OwnedBuffer = buffer.OwnedBuffer;

const int64_type = DataType{ .int64 = {} };
const f64_type = DataType{ .double = {} };

/// Register product, variance, stddev, skew, kurtosis, first, last, first_last,
/// count_distinct, index (aggregates), mode and quantile (vector).
pub fn register(registry: *FunctionRegistry) KernelError!void {
    inline for (dense.numeric_types) |T| {
        const check = comptime dense.argsOfType(std.meta.activeTag(dense.dataTypeOf(T)), 1);
        try basic.add(registry, "product", 1, check, basic.scalarAggregateOptions, Product(T));
    }
    inline for (.{ i128, i256 }) |T| {
        const check = comptime dense.argsOfType(if (T == i128) .decimal128 else .decimal256, 1);
        try basic.add(registry, "product", 1, check, basic.scalarAggregateOptions, ProductDecimal(T));
    }
    inline for (moment_types) |pair| {
        const T = pair[0];
        const check = comptime dense.argsOfType(pair[1], 1);
        const dec = comptime (pair[1] == .decimal32 or pair[1] == .decimal64 or pair[1] == .decimal128 or pair[1] == .decimal256);
        try basic.add(registry, "variance", 1, check, varianceOptions, Moments(T, dec, .variance));
        try basic.add(registry, "stddev", 1, check, varianceOptions, Moments(T, dec, .stddev));
        try basic.add(registry, "skew", 1, check, skewOptions, Moments(T, dec, .skew));
        try basic.add(registry, "kurtosis", 1, check, skewOptions, Moments(T, dec, .kurtosis));
    }
    try basic.add(registry, "first", 1, firstLastArg, basic.scalarAggregateOptions, FirstLast(.first));
    try basic.add(registry, "last", 1, firstLastArg, basic.scalarAggregateOptions, FirstLast(.last));
    try basic.add(registry, "first_last", 1, firstLastArg, basic.scalarAggregateOptions, FirstLast(.first_last));
    try basic.add(registry, "count_distinct", 1, keyArg, basic.countOptions, CountDistinct);
    const I = basic.Aggregate(Index);
    try registry.registerAggregateKernel("index", .{
        .signature = .{ .arity = 1, .type_check = keyArg, .options_check = indexOptions, .result_type_fn = dense.resultTypeOf(int64_type) },
        .exec = indexExec,
        .aggregate_lifecycle = .{ .init = I.init, .update = I.update, .merge = I.merge, .finalize = I.finalize, .deinit = I.deinit },
    });
    try registry.registerVectorKernel("mode", .{
        .signature = .{ .arity = 1, .type_check = modeArg, .options_check = modeOptions, .result_type_alloc_fn = modeType },
        .exec = mode,
    });
    try registry.registerVectorKernel("quantile", .{
        .signature = .{ .arity = 1, .type_check = quantileArg, .options_check = quantileOptions, .result_type_fn = quantileType },
        .exec = quantile,
    });
}

/// Storage and type tag of every input of variance, stddev, skew and kurtosis (and their hash_
/// forms): the numbers and decimal32..decimal256.
pub const moment_types = blk: {
    var out: [dense.numeric_types.len + 4]struct { type, std.meta.Tag(DataType) } = undefined;
    for (dense.numeric_types, 0..) |T, i| out[i] = .{ T, std.meta.activeTag(dense.dataTypeOf(T)) };
    out[dense.numeric_types.len] = .{ i32, .decimal32 };
    out[dense.numeric_types.len + 1] = .{ i64, .decimal64 };
    out[dense.numeric_types.len + 2] = .{ i128, .decimal128 };
    out[dense.numeric_types.len + 3] = .{ i256, .decimal256 };
    break :blk out;
};

fn varianceOptions(options: Options) bool {
    return options == .none or options == .variance;
}

fn skewOptions(options: Options) bool {
    return options == .none or options == .skew;
}

fn indexOptions(options: Options) bool {
    return options == .index;
}

fn modeOptions(options: Options) bool {
    return options == .none or options == .mode;
}

fn quantileOptions(options: Options) bool {
    return options == .none or options == .quantile;
}

fn keyArg(args: []const Datum) bool {
    return args.len == 1 and hash.isKeyType(args[0].dataType());
}

fn isNumber(dt: DataType) bool {
    return switch (dt) {
        .int8, .int16, .int32, .int64, .uint8, .uint16, .uint32, .uint64, .float, .double => true,
        else => false,
    };
}

fn isDecimal(dt: DataType) bool {
    return switch (dt) {
        .decimal32, .decimal64, .decimal128, .decimal256 => true,
        else => false,
    };
}

fn firstLastArg(args: []const Datum) bool {
    if (args.len != 1) return false;
    return switch (args[0].dataType()) {
        .fixed_size_binary, .interval_months, .interval_day_time, .interval_month_day_nano => false,
        else => |dt| hash.isKeyType(dt),
    };
}

fn modeArg(args: []const Datum) bool {
    return args.len == 1 and (isNumber(args[0].dataType()) or isDecimal(args[0].dataType()) or args[0].dataType() == .bool);
}

fn quantileArg(args: []const Datum) bool {
    return args.len == 1 and (isNumber(args[0].dataType()) or isDecimal(args[0].dataType()));
}

fn nullScalar(dt: DataType) Datum {
    return .{ .scalar = Scalar.init(dt, .null) };
}

// ------------------------------------------------------------------------------ product

fn Product(comptime T: type) type {
    const is_float = @typeInfo(T) == .float;
    const signed = !is_float and @typeInfo(T).int.signedness == .signed;
    const Out = if (is_float) f64 else if (signed) i64 else u64;
    return struct {
        pub const State = struct {
            prod: Out = 1,
            count: u64 = 0,
            nulls: u64 = 0,
            /// Integers: a factor was zero, the exact product is negative, a step overflowed.
            zero: bool = false,
            neg: bool = false,
            overflow: bool = false,
        };
        pub const out_type = dense.dataTypeOf(Out);

        inline fn mul(state: *State, v: Out) void {
            if (is_float) {
                state.prod *= v;
                return;
            }
            if (v == 0) state.zero = true;
            if (signed and v < 0) state.neg = !state.neg;
            const r = @mulWithOverflow(state.prod, v);
            state.prod = r[0];
            state.overflow = state.overflow or r[1] != 0;
        }

        pub fn update(state: *State, data: *const ArrayData) KernelError!void {
            const vals = try basic.values(T, data);
            const bits = dense.validityBits(data);
            var count: u64 = 0;
            for (vals, 0..) |v, i| {
                if (bits) |b| if (!bitmap.bitIsSet(b, data.offset + i)) continue;
                mul(state, if (is_float) @floatCast(v) else v);
                count += 1;
            }
            state.count += count;
            state.nulls += data.length - count;
        }

        pub fn updateScalar(state: *State, s: Scalar) KernelError!void {
            if (s.isNull()) {
                state.nulls += 1;
                return;
            }
            const v = try dense.scalarAs(T, s);
            mul(state, if (is_float) @floatCast(v) else v);
            state.count += 1;
        }

        pub fn merge(state: *State, other: *const State) void {
            state.count += other.count;
            state.nulls += other.nulls;
            if (!is_float) {
                state.zero = state.zero or other.zero;
                if (other.neg) state.neg = !state.neg;
                state.overflow = state.overflow or other.overflow;
                const r = @mulWithOverflow(state.prod, other.prod);
                state.prod = r[0];
                state.overflow = state.overflow or r[1] != 0;
            } else state.prod *= other.prod;
        }

        pub fn finalize(ctx: *ExecContext, state: *const State, options: Options) KernelError!Datum {
            if (basic.nullByOptions(basic.aggregateOptions(options), state.count, state.nulls)) return nullScalar(out_type);
            var result = state.prod;
            if (!is_float and state.zero) result = 0;
            if (!is_float and !state.zero and state.overflow) switch (ctx.overflowMode()) {
                .checked => return error.Overflow,
                .wrapping => {},
                .saturating => result = if (state.neg) std.math.minInt(Out) else std.math.maxInt(Out),
            };
            return .{ .scalar = Scalar.init(out_type, dense.scalarValueOf(Out, result)) };
        }
    };
}

// ------------------------------------------------------------------------------ moments

/// Count, mean and central moment sums (M2, M3, M4) of a set of values.
pub const MomentSums = struct {
    n: u64 = 0,
    mean: f64 = 0,
    m2: f64 = 0,
    m3: f64 = 0,
    m4: f64 = 0,

    /// Combine two sets (Chan et al. / Pébay pairwise formulas).
    pub fn merge(a: MomentSums, b: MomentSums) MomentSums {
        if (a.n == 0) return b;
        if (b.n == 0) return a;
        const na: f64 = @floatFromInt(a.n);
        const nb: f64 = @floatFromInt(b.n);
        const n = na + nb;
        const d = b.mean - a.mean;
        const d2 = d * d;
        return .{
            .n = a.n + b.n,
            .mean = a.mean + d * nb / n,
            .m2 = a.m2 + b.m2 + d2 * na * nb / n,
            .m3 = a.m3 + b.m3 + d2 * d * na * nb * (na - nb) / (n * n) + 3 * d * (na * b.m2 - nb * a.m2) / n,
            .m4 = a.m4 + b.m4 + d2 * d2 * na * nb * (na * na - na * nb + nb * nb) / (n * n * n) + 6 * d2 * (na * na * b.m2 + nb * nb * a.m2) / (n * n) + 4 * d * (na * b.m3 - nb * a.m3) / n,
        };
    }

    /// The moments of the valid rows of a chunk read as f64 (two passes: the mean, then the
    /// central sums; `level` 4 also sums the third and fourth powers).
    pub fn ofChunk(comptime T: type, comptime is_dec: bool, data: *const ArrayData, scale: i32, comptime level: u8) KernelError!MomentSums {
        const vals = try basic.values(T, data);
        const bits = dense.validityBits(data);
        const is_float = @typeInfo(T) == .float;
        var sum: f64 = 0;
        var n: u64 = 0;
        if (!is_dec) {
            const zero: @Vector(8, T) = @splat(0);
            var acc: @Vector(8, f64) = @splat(0);
            var i: usize = 0;
            while (i < vals.len) : (i += 8) {
                const mask = dense.laneMask(data, bits, i);
                const v = @select(T, mask, basic.lanes(T, vals, i), zero);
                acc += if (is_float) @floatCast(v) else @floatFromInt(v);
                n += basic.popCount(mask);
            }
            sum = @reduce(.Add, acc);
        } else for (vals, 0..) |v, i| {
            if (bits) |b| if (!bitmap.bitIsSet(b, data.offset + i)) continue;
            sum += decimal.toFloatValue(f64, v, scale);
            n += 1;
        }
        if (n == 0) return .{};
        const mean = sum / @as(f64, @floatFromInt(n));
        var out: MomentSums = .{ .n = n, .mean = mean };
        if (!is_dec) {
            const zero: @Vector(8, f64) = @splat(0);
            const mv: @Vector(8, f64) = @splat(mean);
            var s2: @Vector(8, f64) = @splat(0);
            var s3: @Vector(8, f64) = @splat(0);
            var s4: @Vector(8, f64) = @splat(0);
            var i: usize = 0;
            while (i < vals.len) : (i += 8) {
                const mask = dense.laneMask(data, bits, i);
                const raw = basic.lanes(T, vals, i);
                const x: @Vector(8, f64) = if (is_float) @floatCast(raw) else @floatFromInt(raw);
                const d = @select(f64, mask, x - mv, zero);
                const dd = d * d;
                s2 += dd;
                if (level == 4) {
                    s3 += dd * d;
                    s4 += dd * dd;
                }
            }
            out.m2 = @reduce(.Add, s2);
            out.m3 = @reduce(.Add, s3);
            out.m4 = @reduce(.Add, s4);
        } else for (vals, 0..) |v, i| {
            if (bits) |b| if (!bitmap.bitIsSet(b, data.offset + i)) continue;
            const d = decimal.toFloatValue(f64, v, scale) - mean;
            out.m2 += d * d;
            out.m3 += d * d * d;
            out.m4 += d * d * d * d;
        }
        return out;
    }

    /// The moments of one value.
    pub fn of(x: f64) MomentSums {
        return .{ .n = 1, .mean = x };
    }
};

pub const MomentKind = enum { variance, stddev, skew, kurtosis };

/// The result of `kind` over `m` (null when the options or the count rule it out).
pub fn momentResult(comptime kind: MomentKind, m: MomentSums, nulls: u64, options: Options) ?f64 {
    const skip_nulls, const min_count = switch (options) {
        .variance => |o| .{ o.skip_nulls, o.min_count },
        .skew => |o| .{ o.skip_nulls, o.min_count },
        else => .{ true, @as(u32, 0) },
    };
    if ((!skip_nulls and nulls > 0) or m.n < min_count or m.n == 0) return null;
    const n: f64 = @floatFromInt(m.n);
    switch (kind) {
        .variance, .stddev => {
            const ddof: i64 = if (options == .variance) options.variance.ddof else 0;
            if (@as(i64, @intCast(m.n)) <= ddof) return null;
            const v = m.m2 / (n - @as(f64, @floatFromInt(ddof)));
            return if (kind == .stddev) @sqrt(v) else v;
        },
        .skew => {
            const biased = if (options == .skew) options.skew.biased else true;
            const g1 = @sqrt(n) * m.m3 / std.math.pow(f64, m.m2, 1.5);
            if (biased) return g1;
            if (m.n < 3) return null;
            return g1 * @sqrt(n * (n - 1)) / (n - 2);
        },
        .kurtosis => {
            const biased = if (options == .skew) options.skew.biased else true;
            const g2 = n * m.m4 / (m.m2 * m.m2) - 3;
            if (biased) return g2;
            if (m.n < 4) return null;
            return ((n + 1) * g2 + 6) * (n - 1) / ((n - 2) * (n - 3));
        },
    }
}

fn Moments(comptime T: type, comptime is_dec: bool, comptime kind: MomentKind) type {
    const level: u8 = if (kind == .skew or kind == .kurtosis) 4 else 2;
    return struct {
        pub const State = struct { m: MomentSums = .{}, nulls: u64 = 0 };
        pub const out_type = f64_type;

        pub fn update(state: *State, data: *const ArrayData) KernelError!void {
            const scale: i32 = if (is_dec) (try decimal.checkedParams(data.data_type)).scale else 0;
            const m = try MomentSums.ofChunk(T, is_dec, data, scale, level);
            state.m = MomentSums.merge(state.m, m);
            state.nulls += data.length - m.n;
        }

        pub fn updateScalar(state: *State, s: Scalar) KernelError!void {
            if (s.isNull()) {
                state.nulls += 1;
                return;
            }
            const x: f64 = if (is_dec)
                decimal.toFloatValue(f64, try decimal.scalarInt(i256, s), (try decimal.checkedParams(s.data_type)).scale)
            else if (@typeInfo(T) == .float) @floatCast(try dense.scalarAs(T, s)) else @floatFromInt(try dense.scalarAs(T, s));
            state.m = MomentSums.merge(state.m, MomentSums.of(x));
        }

        pub fn merge(state: *State, other: *const State) void {
            state.m = MomentSums.merge(state.m, other.m);
            state.nulls += other.nulls;
        }

        pub fn finalize(_: *ExecContext, state: *const State, options: Options) KernelError!Datum {
            if (options == .variance and options.variance.ddof < 0) return error.InvalidOptions;
            const r = momentResult(kind, state.m, state.nulls, options) orelse return nullScalar(out_type);
            return .{ .scalar = Scalar.init(out_type, .{ .f64 = r }) };
        }
    };
}

// ------------------------------------------------------------------------------ first / last

const first_last_names = [_][]const u8{ "first", "last" };

fn FirstLast(comptime which: enum { first, last, first_last }) type {
    return struct {
        pub const update_takes_ref = true;

        /// One-row slices of the input holding the first / last valid row and the first / last
        /// row; the state owns them.
        pub const State = struct {
            allocator: std.mem.Allocator,
            first_valid: ?ArrayRef = null,
            last_valid: ?ArrayRef = null,
            first_any: ?ArrayRef = null,
            last_any: ?ArrayRef = null,
            count: u64 = 0,
            nulls: u64 = 0,
            dt: DataType = .{ .null = {} },
        };

        pub const resultType = if (which == .first_last) {} else struct {
            fn f(args: []const Datum, _: Options) KernelError!DataType {
                if (args.len != 1) return error.InvalidArity;
                return args[0].dataType();
            }
        }.f;

        pub const resultTypeAlloc = if (which != .first_last) {} else struct {
            fn f(allocator: std.mem.Allocator, args: []const Datum, _: Options) KernelError!DataType {
                if (args.len != 1) return error.InvalidArity;
                return dense.structTypeAlloc(allocator, &first_last_names, &.{ args[0].dataType(), args[0].dataType() });
            }
        }.f;

        pub fn initState(ctx: *ExecContext, _: Options) KernelError!State {
            return .{ .allocator = ctx.allocator };
        }

        pub fn deinitState(state: *State) void {
            inline for (.{ "first_valid", "last_valid", "first_any", "last_any" }) |f| {
                if (@field(state, f)) |*a| a.release();
            }
        }

        fn put(slot: *?ArrayRef, arr: ArrayRef, row: usize) KernelError!void {
            const one = arr.slice(row, 1) catch return error.OutOfMemory;
            if (slot.*) |*old| old.release();
            slot.* = one;
        }

        pub fn update(state: *State, arr: ArrayRef) KernelError!void {
            const data = arr.data();
            state.dt = data.data_type;
            if (data.length == 0) return;
            if (state.first_any == null) try put(&state.first_any, arr, 0);
            try put(&state.last_any, arr, data.length - 1);
            var valid: u64 = 0;
            var first: ?usize = null;
            var last: ?usize = null;
            for (0..data.length) |i| if (!data.isNull(i)) {
                if (first == null) first = i;
                last = i;
                valid += 1;
            };
            if (first) |f| if (state.first_valid == null) try put(&state.first_valid, arr, f);
            if (last) |l| try put(&state.last_valid, arr, l);
            state.count += valid;
            state.nulls += data.length - valid;
        }

        pub fn updateScalar(state: *State, s: Scalar) KernelError!void {
            var arr = try dense.chunkArray(state.allocator, .{ .scalar = s });
            defer arr.release();
            try update(state, arr);
        }

        pub fn merge(state: *State, other: *const State) void {
            inline for (.{ "first_valid", "first_any" }) |f| {
                if (@field(state, f) == null) if (@field(other, f)) |a| {
                    @field(state, f) = a.retain();
                };
            }
            inline for (.{ "last_valid", "last_any" }) |f| {
                if (@field(other, f)) |a| {
                    if (@field(state, f)) |*old| old.release();
                    @field(state, f) = a.retain();
                }
            }
            state.count += other.count;
            state.nulls += other.nulls;
            if (state.dt == .null) state.dt = other.dt;
        }

        /// The row picked for `first` / `last`, or null.
        fn pick(state: *const State, comptime is_first: bool, o: core.ScalarAggregateOptions) ?ArrayRef {
            if (state.count < o.min_count) return null;
            if (o.skip_nulls) return if (is_first) state.first_valid else state.last_valid;
            return if (is_first) state.first_any else state.last_any;
        }

        pub fn finalize(ctx: *ExecContext, state: *const State, options: Options) KernelError!Datum {
            const o = basic.aggregateOptions(options);
            if (which != .first_last) {
                const row = pick(state, which == .first, o) orelse return nullScalar(state.dt);
                return .{ .scalar = try dense.scalarAt0(row) };
            }
            var parts: [2]ArrayRef = undefined;
            for (&parts, [_]bool{ true, false }, 0..) |*p, is_first, k| {
                const row = if (is_first) pick(state, true, o) else pick(state, false, o);
                p.* = if (row) |r| r.retain() else dense.nullArray(ctx.allocator, state.dt, 1) catch |err| {
                    for (parts[0..k]) |*q| q.release();
                    return err;
                };
            }
            var payload = try dense.structArray(ctx.allocator, &first_last_names, &parts, 1, null);
            defer payload.release();
            return .{ .scalar = try Scalar.initNested(payload.data().data_type, payload) };
        }
    };
}

// ------------------------------------------------------------------------------ count_distinct

const CountDistinct = struct {
    pub const update_takes_ref = true;
    pub const State = struct {
        allocator: std.mem.Allocator,
        grouper: ?hash.Grouper = null,
        saw_null: bool = false,
    };
    pub const out_type = int64_type;

    pub fn initState(ctx: *ExecContext, _: Options) KernelError!State {
        return .{ .allocator = ctx.allocator };
    }

    pub fn deinitState(state: *State) void {
        if (state.grouper) |*g| g.deinit();
    }

    pub fn update(state: *State, arr: ArrayRef) KernelError!void {
        if (state.grouper == null) state.grouper = try hash.Grouper.init(state.allocator, &.{arr.data().data_type});
        var ids = try state.grouper.?.consume(&.{arr});
        ids.release();
        if (arr.data().hasNulls()) state.saw_null = true;
    }

    pub fn updateScalar(state: *State, s: Scalar) KernelError!void {
        var arr = try dense.chunkArray(state.allocator, .{ .scalar = s });
        defer arr.release();
        try update(state, arr);
    }

    pub fn merge(state: *State, other: *const State) KernelError!void {
        state.saw_null = state.saw_null or other.saw_null;
        const og = &(other.grouper orelse return);
        const keys = try og.uniques();
        defer {
            for (keys) |*k| k.release();
            og.allocator.free(keys);
        }
        try update(state, keys[0]);
    }

    pub fn finalize(_: *ExecContext, state: *const State, options: Options) KernelError!Datum {
        const groups: u64 = if (state.grouper) |g| g.numGroups() else 0;
        const has_null: u64 = @intFromBool(state.saw_null);
        const n = switch (if (options == .count) options.count.mode else .only_valid) {
            .only_valid => groups - has_null,
            .only_null => has_null,
            .all => groups,
        };
        return .{ .scalar = Scalar.init(out_type, .{ .i64 = @intCast(n) }) };
    }
};

// ------------------------------------------------------------------------------ index

const Index = struct {
    pub const update_takes_ref = true;
    pub const State = struct {
        allocator: std.mem.Allocator,
        /// The value as a one-row array (its key is id 0 of `map`), null when it is never
        /// found (a null or NaN value).
        target: ?ArrayRef = null,
        map: hash.KeyMap = .{},
        found: ?u64 = null,
        rows: u64 = 0,
    };
    pub const out_type = int64_type;

    pub fn initState(ctx: *ExecContext, options: Options) KernelError!State {
        if (options != .index) return error.InvalidOptions;
        var state: State = .{ .allocator = ctx.allocator };
        const v = options.index.value;
        const nan = switch (v.value) {
            .f32 => |x| std.math.isNan(x),
            .f64 => |x| std.math.isNan(x),
            else => false,
        };
        if (v.isNull() or nan or !hash.isKeyType(v.data_type)) return state;
        state.target = try dense.chunkArray(ctx.allocator, .{ .scalar = v });
        errdefer state.target.?.release();
        var id: [1]u32 = undefined;
        try state.map.map(ctx.allocator, state.target.?, &id, .insert);
        return state;
    }

    pub fn deinitState(state: *State) void {
        state.map.deinit(state.allocator);
        if (state.target) |*t| t.release();
    }

    pub fn update(state: *State, arr: ArrayRef) KernelError!void {
        defer state.rows += arr.data().length;
        if (state.found != null) return;
        const target = state.target orelse return;
        if (!target.data().data_type.eql(arr.data().data_type)) return error.InvalidOptions;
        const ids = state.allocator.alloc(u32, arr.data().length) catch return error.OutOfMemory;
        defer state.allocator.free(ids);
        try state.map.map(state.allocator, arr, ids, .lookup);
        for (ids, 0..) |id, i| if (id == 0 and !arr.data().isNull(i)) {
            state.found = state.rows + i;
            return;
        };
    }

    pub fn updateScalar(state: *State, s: Scalar) KernelError!void {
        var arr = try dense.chunkArray(state.allocator, .{ .scalar = s });
        defer arr.release();
        try update(state, arr);
    }

    pub fn merge(state: *State, other: *const State) void {
        if (state.found == null) if (other.found) |f| {
            state.found = state.rows + f;
        };
        state.rows += other.rows;
    }

    pub fn finalize(_: *ExecContext, state: *const State, _: Options) KernelError!Datum {
        return .{ .scalar = Scalar.init(out_type, .{ .i64 = if (state.found) |f| @intCast(f) else -1 }) };
    }
};

/// index: the value is cast to the input type first (safe; a value the cast rejects is never
/// found, so the result is -1). A lifecycle session takes the value as it is.
fn indexExec(ctx: *ExecContext, args: []const Datum, options: Options) KernelError!Datum {
    if (args.len != 1) return error.InvalidArity;
    const v = options.index.value;
    const to = args[0].dataType();
    if (v.data_type.eql(to) or v.isNull()) return basic.Aggregate(Index).exec(ctx, args, options);
    var cast = ctx.invoke("cast", .scalar, &.{.{ .scalar = v }}, .{ .cast = .{ .to_type = to } }) catch |err| switch (err) {
        error.InvalidCast, error.UnsupportedType, error.NoMatchingKernel, error.NotImplemented => return .{ .scalar = Scalar.init(int64_type, .{ .i64 = -1 }) },
        else => return err,
    };
    defer cast.release();
    const s = cast.asScalar() orelse return error.InvalidOptions;
    return basic.Aggregate(Index).exec(ctx, args, .{ .index = .{ .value = s } });
}

// ------------------------------------------------------------------------------ mode

const mode_names = [_][]const u8{ "mode", "count" };

fn modeType(allocator: std.mem.Allocator, args: []const Datum, _: Options) KernelError!DataType {
    return dense.structTypeAlloc(allocator, &mode_names, &.{ args[0].dataType(), int64_type });
}

/// The input of a vector aggregate as array parts (a scalar as one row); the caller releases.
fn partsOf(allocator: std.mem.Allocator, datum: Datum) KernelError![]ArrayRef {
    const src: []const ArrayRef = switch (datum) {
        .array => |*a| a[0..1],
        .chunked => |c| c.chunks(),
        .scalar => |s| {
            const one = allocator.alloc(ArrayRef, 1) catch return error.OutOfMemory;
            one[0] = dense.chunkArray(allocator, .{ .scalar = s }) catch |err| {
                allocator.free(one);
                return err;
            };
            return one;
        },
        .record_batch, .table => return error.InvalidInput,
    };
    const out = allocator.alloc(ArrayRef, src.len) catch return error.OutOfMemory;
    for (out, src) |*o, s| o.* = s.retain();
    return out;
}

fn releaseParts(allocator: std.mem.Allocator, parts: []ArrayRef) void {
    for (parts) |*p| p.release();
    allocator.free(parts);
}

fn mode(ctx: *ExecContext, args: []const Datum, options: Options) KernelError!Datum {
    if (args.len != 1) return error.InvalidArity;
    const o: core.ModeOptions = if (options == .mode) options.mode else .{};
    if (o.n < 1) return error.InvalidOptions;
    const allocator = ctx.allocator;
    const dt = args[0].dataType();
    const parts = try partsOf(allocator, args[0]);
    defer releaseParts(allocator, parts);
    var grouper = try hash.Grouper.init(allocator, &.{dt});
    defer grouper.deinit();
    var counts: std.ArrayList(i64) = .empty;
    defer counts.deinit(allocator);
    var null_group: ?u32 = null;
    var valid: u64 = 0;
    var nulls: u64 = 0;
    for (parts) |p| {
        var ids = try grouper.consume(&.{p});
        defer ids.release();
        const id_vals = (PrimitiveArray(u32){ .data = ids.data() }).values() catch return error.InvalidInput;
        counts.appendNTimes(allocator, 0, grouper.numGroups() - counts.items.len) catch return error.OutOfMemory;
        for (id_vals, 0..) |id, i| {
            counts.items[id] += 1;
            if (p.data().isNull(i)) {
                null_group = id;
                nulls += 1;
            } else valid += 1;
        }
    }
    var keys = blk: {
        const u = try grouper.uniques();
        defer allocator.free(u);
        break :blk u[0];
    };
    defer keys.release();
    // Groups other than the null one, by count (descending) then value.
    var order: std.ArrayList(u32) = .empty;
    defer order.deinit(allocator);
    const empty = (!o.skip_nulls and nulls > 0) or valid < o.min_count;
    if (!empty) for (0..counts.items.len) |g| if (null_group == null or g != null_group.?) order.append(allocator, @intCast(g)) catch return error.OutOfMemory;
    try sortGroups(keys.data(), counts.items, order.items);
    const n: usize = @min(order.items.len, std.math.cast(usize, o.n) orelse std.math.maxInt(usize));
    const picks = allocator.alloc(usize, n) catch return error.OutOfMemory;
    defer allocator.free(picks);
    var cbuf = OwnedBuffer.initUninit(allocator, n * 8) catch return error.OutOfMemory;
    for (picks, cbuf.typedSlice(i64)[0..n], order.items[0..n]) |*p, *c, g| {
        p.* = g;
        c.* = counts.items[g];
    }
    var count_arr = try dense.finishFixed(allocator, int64_type, n, &cbuf, null);
    const mode_arr = takeRows(allocator, keys, picks) catch |err| {
        count_arr.release();
        return err;
    };
    return .{ .array = try dense.structArray(allocator, &mode_names, &.{ mode_arr, count_arr }, n, null) };
}

/// Sort group ids by count (descending), then value (ascending, NaN last).
fn sortGroups(keys: *const ArrayData, counts: []const i64, order: []u32) KernelError!void {
    switch (keys.data_type) {
        inline .int8, .int16, .int32, .int64, .uint8, .uint16, .uint32, .uint64, .float, .double, .decimal32, .decimal64, .decimal128, .decimal256 => |_, tag| {
            const T = switch (tag) {
                .int8 => i8,
                .int16 => i16,
                .int32, .decimal32 => i32,
                .int64, .decimal64 => i64,
                .uint8 => u8,
                .uint16 => u16,
                .uint32 => u32,
                .uint64 => u64,
                .float => f32,
                .double => f64,
                .decimal128 => i128,
                else => i256,
            };
            const vals = (PrimitiveArray(T){ .data = keys }).values() catch return error.InvalidInput;
            const Ctx = struct {
                vals: []const T,
                counts: []const i64,
                fn less(c: @This(), a: u32, b: u32) bool {
                    if (c.counts[a] != c.counts[b]) return c.counts[a] > c.counts[b];
                    return valueLess(T, c.vals[a], c.vals[b]);
                }
            };
            std.mem.sort(u32, order, Ctx{ .vals = vals, .counts = counts }, Ctx.less);
        },
        .bool => {
            const Ctx = struct {
                keys: *const ArrayData,
                counts: []const i64,
                fn less(c: @This(), a: u32, b: u32) bool {
                    if (c.counts[a] != c.counts[b]) return c.counts[a] > c.counts[b];
                    const x = bitmap.bitIsSet(c.keys.buffers[1].data, c.keys.offset + a);
                    const y = bitmap.bitIsSet(c.keys.buffers[1].data, c.keys.offset + b);
                    return !x and y;
                }
            };
            std.mem.sort(u32, order, Ctx{ .keys = keys, .counts = counts }, Ctx.less);
        },
        else => return error.UnsupportedType,
    }
}

/// `a < b` with NaN after every number.
inline fn valueLess(comptime T: type, a: T, b: T) bool {
    if (@typeInfo(T) == .float) {
        if (std.math.isNan(a)) return false;
        if (std.math.isNan(b)) return true;
    }
    return a < b;
}

/// Rows `indices` of `arr` as a new array.
fn takeRows(allocator: std.mem.Allocator, arr: ArrayRef, indices: []const usize) KernelError!ArrayRef {
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

// ------------------------------------------------------------------------------ quantile

fn quantileType(args: []const Datum, options: Options) KernelError!DataType {
    if (args.len != 1) return error.InvalidArity;
    const interp = if (options == .quantile) options.quantile.interpolation else .linear;
    return if (interp == .linear or interp == .midpoint) f64_type else args[0].dataType();
}

fn quantile(ctx: *ExecContext, args: []const Datum, options: Options) KernelError!Datum {
    if (args.len != 1) return error.InvalidArity;
    const o: core.QuantileOptions = if (options == .quantile) options.quantile else .{};
    for (o.q) |q| if (!(q >= 0 and q <= 1)) return error.InvalidOptions;
    const dt = args[0].dataType();
    const parts = try partsOf(ctx.allocator, args[0]);
    defer releaseParts(ctx.allocator, parts);
    for (parts) |p| p.data().validateLayout() catch return error.InvalidInput;
    switch (dt) {
        inline .int8, .int16, .int32, .int64, .uint8, .uint16, .uint32, .uint64, .float, .double, .decimal32, .decimal64, .decimal128, .decimal256 => |_, tag| {
            const T = switch (tag) {
                .int8 => i8,
                .int16 => i16,
                .int32, .decimal32 => i32,
                .int64, .decimal64 => i64,
                .uint8 => u8,
                .uint16 => u16,
                .uint32 => u32,
                .uint64 => u64,
                .float => f32,
                .double => f64,
                .decimal128 => i128,
                else => i256,
            };
            return .{ .array = try Quantile(T).run(ctx, dt, parts, o) };
        },
        else => return error.UnsupportedType,
    }
}

fn Quantile(comptime T: type) type {
    const is_float = @typeInfo(T) == .float;
    return struct {
        fn run(ctx: *ExecContext, dt: DataType, parts: []const ArrayRef, o: core.QuantileOptions) KernelError!ArrayRef {
            const allocator = ctx.allocator;
            var vals: std.ArrayList(T) = .empty;
            defer vals.deinit(allocator);
            var nulls: u64 = 0;
            for (parts) |p| {
                const d = p.data();
                const src = (PrimitiveArray(T){ .data = d }).values() catch return error.InvalidInput;
                for (src, 0..) |v, i| {
                    if (d.isNull(i)) {
                        nulls += 1;
                        continue;
                    }
                    if (is_float and std.math.isNan(v)) continue;
                    vals.append(allocator, v) catch return error.OutOfMemory;
                }
            }
            const none_left = vals.items.len == 0 or vals.items.len < o.min_count or (!o.skip_nulls and nulls > 0);
            std.mem.sort(T, vals.items, {}, std.sort.asc(T));
            const scale: i32 = if (T == i128 or T == i256 or dt == .decimal32 or dt == .decimal64) @intCast(decimal.scaleOf(dt)) else 0;
            const n = o.q.len;
            const is_dec = switch (dt) {
                .decimal32, .decimal64, .decimal128, .decimal256 => true,
                else => false,
            };
            // No value left: every quantile is null.
            var v: ?OwnedBuffer = null;
            if (none_left and n > 0) {
                v = try dense.bitsBuffer(allocator, n);
                @memset(v.?.data[0..bitmap.byteLength(n)], 0);
            }
            errdefer if (v) |*b| b.deinit();
            if (o.interpolation == .linear or o.interpolation == .midpoint) {
                var out = OwnedBuffer.init(allocator, n * 8) catch return error.OutOfMemory;
                if (!none_left) for (out.typedSlice(f64)[0..n], o.q) |*d, q| {
                    d.* = interpolated(vals.items, q, o.interpolation == .linear, is_dec, scale);
                };
                return dense.finishFixed(allocator, f64_type, n, &out, v);
            }
            var out = OwnedBuffer.init(allocator, n * @sizeOf(T)) catch return error.OutOfMemory;
            if (!none_left) for (out.typedSlice(T)[0..n], o.q) |*d, q| {
                d.* = vals.items[pointIndex(vals.items.len, q, o.interpolation)];
            };
            return dense.finishFixed(allocator, dt, n, &out, v);
        }

        fn toF64(x: T, is_dec: bool, scale: i32) f64 {
            if (is_float) return @floatCast(x);
            if (is_dec) return decimal.toFloatValue(f64, x, scale);
            return @floatFromInt(x);
        }

        /// Arrow's `GetQuantileByInterp` over the sorted values.
        fn interpolated(sorted: []const T, q: f64, linear: bool, is_dec: bool, scale: i32) f64 {
            const index = @as(f64, @floatFromInt(sorted.len - 1)) * q;
            const lower: usize = @intFromFloat(index);
            const fraction = index - @as(f64, @floatFromInt(lower));
            const lo = toF64(sorted[lower], is_dec, scale);
            if (fraction == 0) return lo;
            const hi = toF64(sorted[lower + 1], is_dec, scale);
            return if (linear) fraction * hi + (1 - fraction) * lo else lo / 2 + hi / 2;
        }
    };
}

/// Arrow's `GetQuantileAtDataPoint`: the index of the `lower`, `higher` or `nearest` point (ties
/// to the even index).
fn pointIndex(len: usize, q: f64, interp: core.QuantileInterpolation) usize {
    const index = @as(f64, @floatFromInt(len - 1)) * q;
    const lower: usize = @intFromFloat(index);
    const fraction = index - @as(f64, @floatFromInt(lower));
    return switch (interp) {
        .lower => lower,
        .higher => if (fraction > 0) lower + 1 else lower,
        else => if (fraction < 0.5) lower else if (fraction > 0.5) lower + 1 else lower + (lower & 1),
    };
}

// ------------------------------------------------------------------------------ decimal product

/// The scale arithmetic of one decimal type for `DecimalProduct`.
pub fn DecimalScale(comptime T: type) type {
    const W = if (T == i128) i256 else i512;
    return struct {
        /// 10^scale, wide and at the storage width.
        div: W,
        narrow_div: T,
        /// 10^precision - 1, the largest value the type holds.
        limit: W,
        params: decimal.Params,

        /// `InvalidInput` for a precision past the type's; `NotImplemented` for a scale outside
        /// [0, the precision max] (Arrow's decimal product is undefined there: it scales 1 by it).
        pub fn of(dt: DataType) KernelError!@This() {
            const p = try decimal.checkedParams(dt);
            if (p.scale < 0 or p.scale > decimal.maxPrecision(dt)) return error.NotImplemented;
            var div: W = 1;
            for (0..@intCast(p.scale)) |_| div *= 10;
            var limit: W = 1;
            for (0..p.precision) |_| limit *= 10;
            return .{ .div = div, .narrow_div = @intCast(div), .limit = limit - 1, .params = p };
        }
    };
}

/// Divide by `d` (10^scale), rounding half away from zero (Arrow `ReduceScaleBy(_, true)`).
fn rescale(comptime I: type, v: I, d: I) I {
    if (d == 1) return v;
    const q = @divTrunc(v, d);
    const r = @rem(v, d);
    if (@abs(r) >= @abs(@divTrunc(d, 2))) return if (v < 0) q - 1 else q + 1;
    return q;
}

/// A running decimal product as Arrow C++ computes it (`MultiplyTraits` for decimals): every
/// step multiplies and rescales to the input scale, rounding half away from zero; the empty
/// product is 1. `exact` follows the true steps while they fit the precision (`overflow` once
/// one does not); `arrow` repeats Arrow's arithmetic, which wraps at the storage width (the
/// `wrapping` result).
pub fn DecimalProduct(comptime T: type) type {
    const W = if (T == i128) i256 else i512;
    const Scale = DecimalScale(T);
    return struct {
        exact: T = 0,
        arrow: T = 0,
        started: bool = false,
        zero: bool = false,
        neg: bool = false,
        overflow: bool = false,

        const Self = @This();

        fn step(self: *Self, ve: T, va: T, s: *const Scale) void {
            if (!self.overflow) {
                const w = rescale(W, @as(W, self.exact) * ve, s.div);
                if (w > s.limit or w < -s.limit) self.overflow = true else self.exact = @intCast(w);
            }
            const a = if (self.started) self.arrow else s.narrow_div;
            self.arrow = rescale(T, a *% va, s.narrow_div);
        }

        /// Multiply by one value.
        pub fn mul(self: *Self, v: T, s: *const Scale) void {
            if (v == 0) self.zero = true;
            if (v < 0) self.neg = !self.neg;
            if (!self.started) {
                // 1 * v rescaled is v (Arrow's own steps may wrap: `arrow` repeats them).
                self.arrow = rescale(T, s.narrow_div *% v, s.narrow_div);
                self.exact = v;
                self.started = true;
                return;
            }
            self.step(v, v, s);
        }

        /// Multiply by another product (Arrow merges states with one more step).
        pub fn merge(self: *Self, other: Self, s: *const Scale) void {
            if (!other.started) {
                if (self.started) self.arrow = rescale(T, self.arrow *% s.narrow_div, s.narrow_div);
                return;
            }
            self.zero = self.zero or other.zero;
            if (other.neg) self.neg = !self.neg;
            if (!self.started) {
                self.exact = other.exact;
                self.arrow = rescale(T, s.narrow_div *% other.arrow, s.narrow_div);
                self.overflow = other.overflow;
                self.started = true;
                return;
            }
            self.overflow = self.overflow or other.overflow;
            self.step(other.exact, other.arrow, s);
        }

        /// The product under `mode`: past the precision `checked` is `Overflow`, `wrapping`
        /// gives Arrow's value, `saturating` the largest value of the product's sign.
        pub fn result(self: *const Self, overflow_mode: core.OverflowMode, s: *const Scale) KernelError!T {
            if (!self.started) return s.narrow_div;
            if (self.zero) return 0;
            if (overflow_mode == .wrapping) return self.arrow;
            if (!self.overflow) return self.exact;
            if (overflow_mode == .checked) return error.Overflow;
            const lim: T = @intCast(s.limit);
            return if (self.neg) -lim else lim;
        }
    };
}

/// product of decimal128 / decimal256 (`ScalarAggregateOptions`): the input type, each chunk a
/// product of its own merged into the state's (Arrow's executor), overflow per `overflow_mode`.
fn ProductDecimal(comptime T: type) type {
    const Acc = DecimalProduct(T);
    const Scale = DecimalScale(T);
    return struct {
        pub const State = struct {
            dt: DataType = .{ .null = {} },
            acc: Acc = .{},
            count: u64 = 0,
            nulls: u64 = 0,
        };

        pub fn resultType(args: []const Datum, _: Options) KernelError!DataType {
            if (args.len != 1) return error.InvalidArity;
            return args[0].dataType();
        }

        pub fn update(state: *State, data: *const ArrayData) KernelError!void {
            state.dt = data.data_type;
            const s = try Scale.of(data.data_type);
            if (data.length == 0) return;
            const vals = try basic.values(T, data);
            const bits = dense.validityBits(data);
            var chunk: Acc = .{};
            var count: u64 = 0;
            for (vals, 0..) |v, i| {
                if (bits) |b| if (!bitmap.bitIsSet(b, data.offset + i)) continue;
                chunk.mul(v, &s);
                count += 1;
            }
            state.count += count;
            state.nulls += data.length - count;
            state.acc.merge(chunk, &s);
        }

        pub fn updateScalar(state: *State, sc: Scalar) KernelError!void {
            state.dt = sc.data_type;
            const s = try Scale.of(sc.data_type);
            var chunk: Acc = .{};
            if (sc.isNull()) state.nulls += 1 else {
                chunk.mul(try decimal.scalarInt(T, sc), &s);
                state.count += 1;
            }
            state.acc.merge(chunk, &s);
        }

        pub fn merge(state: *State, other: *const State) KernelError!void {
            if (state.dt == .null) state.dt = other.dt;
            state.count += other.count;
            state.nulls += other.nulls;
            if (other.dt == .null) return;
            const s = try Scale.of(state.dt);
            state.acc.merge(other.acc, &s);
        }

        pub fn finalize(ctx: *ExecContext, state: *const State, options: Options) KernelError!Datum {
            if (basic.nullByOptions(basic.aggregateOptions(options), state.count, state.nulls)) return nullScalar(state.dt);
            const s = try Scale.of(state.dt);
            const v = try state.acc.result(ctx.overflowMode(), &s);
            return .{ .scalar = Scalar.init(state.dt, if (T == i128) .{ .decimal128 = v } else .{ .decimal256 = v }) };
        }
    };
}
