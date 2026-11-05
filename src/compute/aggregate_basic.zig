//! Basic scalar aggregates (Arrow C++ "Aggregations"): sum, mean, min, max, min_max, count,
//! count_all, any and all. Every function takes one array, chunked array or scalar and returns a
//! scalar. Each kernel also has an `AggregateLifecycle`, so a chunked input or a stream of batches
//! can be consumed with `beginAggregate` / `update` / `merge` / `finalize`.
//!
//! sum, mean, min, max, min_max, any and all take `ScalarAggregateOptions` (Arrow defaults when
//! `none`: `skip_nulls = true`, `min_count = 1`): the result is null when a row is null and
//! `skip_nulls` is false, or when fewer than `min_count` rows are valid; any / all with
//! `skip_nulls = false` use Kleene logic instead (any: true if a row is true, else null if a row
//! is null; all: false if a row is false, else null if a row is null). count takes `CountOptions`
//! (`only_valid`, `only_null`, `all`).
//!
//! - sum: int -> int64, uint -> uint64, float -> float64. Integer sums are exact (128-bit) until
//!   finalize, which applies `ExecConfig.overflow_mode` once: `checked` returns `Overflow` when the
//!   total does not fit, `wrapping` keeps the low 64 bits (Arrow C++ behaviour), `saturating` clamps.
//!   0 when no row is valid and `min_count` is 0.
//! - mean: float64 of the exact sum over the valid count; NaN when no row is valid and
//!   `min_count` is 0.
//! - min / max / min_max: the input type; NaN is ignored unless every valid value is NaN; null (both
//!   fields null for min_max) when no row is valid, whatever `min_count`. min_max returns a struct
//!   scalar `{min, max}`.
//! - count: valid, null or all rows (`CountOptions.mode`); count_all: all rows of its one
//!   argument, or 0 when called with no argument (Arrow's nullary calling convention: no batch, no
//!   rows). Both int64, never null.
//! - any / all: bool input; with no valid row (and `min_count` 0) any is false and all is true.
const std = @import("std");
const bitmap = @import("../bitmap.zig");
const buffer = @import("../buffer.zig");
const core = @import("core.zig");
const datatype = @import("../datatype.zig");
const dense = @import("dense.zig");
const is_null = @import("is_null.zig");
const primitive_array = @import("../array/primitive_array.zig");

const ArrayData = @import("../array/array_data.zig").ArrayData;
const ArrayRef = core.ArrayRef;
const DataType = core.DataType;
const Datum = core.Datum;
const ExecContext = core.ExecContext;
const FunctionRegistry = core.FunctionRegistry;
const KernelError = core.KernelError;
const Options = core.Options;
const Scalar = core.Scalar;
const TypeCheckFn = core.TypeCheckFn;
const Bits = dense.Bits;

/// Register sum, mean, min, max, min_max, count, count_all, any and all.
pub fn register(registry: *FunctionRegistry) KernelError!void {
    inline for (dense.numeric_types) |T| {
        const check = comptime dense.argsOfType(std.meta.activeTag(dense.dataTypeOf(T)), 1);
        try add(registry, "sum", 1, check, scalarAggregateOptions, Sum(T, .sum));
        try add(registry, "mean", 1, check, scalarAggregateOptions, Sum(T, .mean));
        try add(registry, "min", 1, check, scalarAggregateOptions, MinMax(T, .min));
        try add(registry, "max", 1, check, scalarAggregateOptions, MinMax(T, .max));
        try add(registry, "min_max", 1, check, scalarAggregateOptions, MinMax(T, .min_max));
    }
    try add(registry, "count", 1, is_null.hasValidityBitmap, countOptions, Count(.count));
    try add(registry, "count_all", 1, is_null.hasValidityBitmap, dense.noOptions, Count(.count_all));
    try add(registry, "count_all", 0, noArgs, dense.noOptions, Count(.count_all));
    try add(registry, "any", 1, dense.argsOfType(.bool, 1), scalarAggregateOptions, AnyAll(.any));
    try add(registry, "all", 1, dense.argsOfType(.bool, 1), scalarAggregateOptions, AnyAll(.all));
}

fn noArgs(args: []const Datum) bool {
    return args.len == 0;
}

/// `none` (Arrow defaults) or `ScalarAggregateOptions`.
fn scalarAggregateOptions(options: Options) bool {
    return options == .none or options == .scalar_aggregate;
}

/// `none` (only_valid) or `CountOptions`.
fn countOptions(options: Options) bool {
    return options == .none or options == .count;
}

fn aggregateOptions(options: Options) core.ScalarAggregateOptions {
    return if (options == .scalar_aggregate) options.scalar_aggregate else .{};
}

/// Whether the options make the result null: a null row with `skip_nulls` false, or fewer valid
/// rows than `min_count` (Arrow C++).
fn nullByOptions(o: core.ScalarAggregateOptions, count: u64, nulls: u64) bool {
    return (!o.skip_nulls and nulls > 0) or count < o.min_count;
}

fn add(registry: *FunctionRegistry, comptime name: []const u8, comptime arity: usize, comptime check: TypeCheckFn, comptime options_check: core.OptionsCheckFn, comptime Impl: type) KernelError!void {
    const A = Aggregate(Impl);
    try registry.registerAggregateKernel(name, .{
        .signature = .{ .arity = arity, .type_check = check, .options_check = options_check, .result_type_fn = dense.resultTypeOf(Impl.out_type) },
        .exec = A.exec,
        .aggregate_lifecycle = .{ .init = A.init, .update = A.update, .merge = A.merge, .finalize = A.finalize, .deinit = A.deinit },
    });
}

/// Exec and lifecycle callbacks around an `Impl` with `State`, `update` (one validated array),
/// `updateScalar`, `merge` and `finalize`.
fn Aggregate(comptime Impl: type) type {
    const State = Impl.State;
    return struct {
        fn consume(state: *State, datum: Datum) KernelError!void {
            switch (datum) {
                .array => |arr| try consumeArray(state, arr.data()),
                .chunked => |chunked| for (chunked.chunks()) |chunk| try consumeArray(state, chunk.data()),
                .scalar => |s| try Impl.updateScalar(state, s),
            }
        }

        fn consumeArray(state: *State, data: *const ArrayData) KernelError!void {
            data.validateLayout() catch return error.InvalidInput;
            try Impl.update(state, data);
        }

        /// Aggregate one array, chunked array or scalar into a scalar (count_all: also none).
        fn exec(ctx: *ExecContext, args: []const Datum, options: Options) KernelError!Datum {
            if (args.len > 1) return error.InvalidArity;
            var state: State = .{};
            if (args.len == 1) try consume(&state, args[0]);
            return Impl.finalize(ctx, &state, options);
        }

        fn init(ctx: *ExecContext, _: Options) KernelError!*anyopaque {
            const state = ctx.allocator.create(State) catch return error.OutOfMemory;
            state.* = .{};
            return state;
        }

        fn update(_: *ExecContext, state: *anyopaque, args: []const Datum, _: Options) KernelError!void {
            if (args.len > 1) return error.InvalidArity;
            if (args.len == 1) try consume(cast(state), args[0]);
        }

        fn merge(_: *ExecContext, state: *anyopaque, other: *anyopaque, _: Options) KernelError!void {
            Impl.merge(cast(state), cast(other));
        }

        fn finalize(ctx: *ExecContext, state: *anyopaque, options: Options) KernelError!Datum {
            return Impl.finalize(ctx, cast(state), options);
        }

        fn deinit(ctx: *ExecContext, state: *anyopaque) void {
            ctx.allocator.destroy(cast(state));
        }

        fn cast(state: *anyopaque) *State {
            return @ptrCast(@alignCast(state));
        }
    };
}

fn nullScalar(dt: DataType) Datum {
    return .{ .scalar = Scalar.init(dt, .null) };
}

fn values(comptime T: type, data: *const ArrayData) KernelError![]const T {
    return (primitive_array.PrimitiveArray(T){ .data = data }).values() catch error.InvalidInput;
}

/// Load 8 values starting at `i` (zero past the end).
inline fn lanes(comptime T: type, vals: []const T, i: usize) @Vector(8, T) {
    if (i + 8 <= vals.len) return vals[i..][0..8].*;
    var tmp: [8]T = @splat(0);
    @memcpy(tmp[0 .. vals.len - i], vals[i..]);
    return tmp;
}

inline fn popCount(mask: @Vector(8, bool)) u64 {
    return @popCount(@as(u8, @bitCast(mask)));
}

fn Sum(comptime T: type, comptime which: enum { sum, mean }) type {
    const is_float = @typeInfo(T) == .float;
    const signed = !is_float and @typeInfo(T).int.signedness == .signed;
    // Exact running sum: integers are summed without loss until finalize.
    const Exact = if (is_float) f64 else if (signed) i128 else u128;
    const Out = if (which == .mean or is_float) f64 else if (signed) i64 else u64;
    return struct {
        const State = struct { sum: Exact = 0, count: u64 = 0, nulls: u64 = 0 };
        const out_type = dense.dataTypeOf(Out);

        fn update(state: *State, data: *const ArrayData) KernelError!void {
            const vals = try values(T, data);
            const bits = dense.validityBits(data);
            const before = state.count;
            defer state.nulls += data.length - (state.count - before);
            // Blocks of 2^31 rows: the 64-bit lane sums of 32-bit halves below cannot overflow.
            var start: usize = 0;
            while (start < vals.len) {
                const end = start + @min(vals.len - start, @as(usize, 1) << 31);
                if (is_float) sumFloat(state, data, bits, vals, start, end) else sumInt(state, data, bits, vals, start, end);
                start = end;
            }
        }

        fn sumFloat(state: *State, data: *const ArrayData, bits: ?[]const u8, vals: []const T, start: usize, end: usize) void {
            const zero: @Vector(8, T) = @splat(0);
            var acc: @Vector(8, f64) = @splat(0);
            var count: u64 = 0;
            var i = start;
            while (i < end) : (i += 8) {
                const mask = dense.laneMask(data, bits, i);
                acc += @floatCast(@select(T, mask, lanes(T, vals, i), zero));
                count += popCount(mask);
            }
            state.sum += @reduce(.Add, acc);
            state.count += count;
        }

        fn sumInt(state: *State, data: *const ArrayData, bits: ?[]const u8, vals: []const T, start: usize, end: usize) void {
            // value = hi * 2^32 + lo with lo the unsigned low 32 bits: both halves sum in 64-bit lanes.
            const W = if (signed) i64 else u64;
            const zero: @Vector(8, T) = @splat(0);
            const low_mask: @Vector(8, u64) = @splat(0xFFFF_FFFF);
            const shift: @Vector(8, u6) = @splat(32);
            var lo: @Vector(8, u64) = @splat(0);
            var hi: @Vector(8, W) = @splat(0);
            var count: u64 = 0;
            var i = start;
            while (i < end) : (i += 8) {
                const mask = dense.laneMask(data, bits, i);
                const v: @Vector(8, W) = @intCast(@select(T, mask, lanes(T, vals, i), zero));
                lo += @as(@Vector(8, u64), @bitCast(v)) & low_mask;
                hi += v >> shift;
                count += popCount(mask);
            }
            state.sum += (@as(Exact, @reduce(.Add, hi)) << 32) + @reduce(.Add, lo);
            state.count += count;
        }

        fn updateScalar(state: *State, s: Scalar) KernelError!void {
            if (s.isNull()) {
                state.nulls += 1;
                return;
            }
            const v = try dense.scalarAs(T, s);
            state.sum += if (is_float) @as(f64, @floatCast(v)) else v;
            state.count += 1;
        }

        fn merge(state: *State, other: *const State) void {
            state.sum += other.sum;
            state.count += other.count;
            state.nulls += other.nulls;
        }

        fn finalize(ctx: *ExecContext, state: *const State, options: Options) KernelError!Datum {
            if (nullByOptions(aggregateOptions(options), state.count, state.nulls)) return nullScalar(out_type);
            const result: Out = if (which == .mean)
                (if (is_float) state.sum else @as(f64, @floatFromInt(state.sum))) / @as(f64, @floatFromInt(state.count))
            else if (is_float)
                state.sum
            else switch (ctx.overflowMode()) {
                .checked => std.math.cast(Out, state.sum) orelse return error.Overflow,
                .wrapping => @truncate(state.sum),
                .saturating => @intCast(std.math.clamp(state.sum, std.math.minInt(Out), std.math.maxInt(Out))),
            };
            return .{ .scalar = Scalar.init(out_type, dense.scalarValueOf(Out, result)) };
        }
    };
}

fn MinMax(comptime T: type, comptime which: enum { min, max, min_max }) type {
    const is_float = @typeInfo(T) == .float;
    // NaN start values: @min / @max return the non-NaN operand, so NaN rows are skipped and an
    // all-NaN input stays NaN (Arrow C++).
    const min_start: T = if (is_float) std.math.nan(T) else std.math.maxInt(T);
    const max_start: T = if (is_float) std.math.nan(T) else std.math.minInt(T);
    return struct {
        const State = struct { min: T = min_start, max: T = max_start, count: u64 = 0, nulls: u64 = 0 };
        const elem_type = dense.dataTypeOf(T);
        const fields = [_]datatype.Field{
            .{ .name = "min", .data_type = &elem_type },
            .{ .name = "max", .data_type = &elem_type },
        };
        const struct_type = DataType{ .struct_ = .{ .fields = &fields } };
        const out_type = if (which == .min_max) struct_type else elem_type;

        fn update(state: *State, data: *const ArrayData) KernelError!void {
            const vals = try values(T, data);
            const bits = dense.validityBits(data);
            var lo: @Vector(8, T) = @splat(min_start);
            var hi: @Vector(8, T) = @splat(max_start);
            var count: u64 = 0;
            var i: usize = 0;
            while (i < vals.len) : (i += 8) {
                const mask = dense.laneMask(data, bits, i);
                const v = lanes(T, vals, i);
                lo = @min(lo, @select(T, mask, v, lo));
                hi = @max(hi, @select(T, mask, v, hi));
                count += popCount(mask);
            }
            state.min = @min(state.min, @reduce(.Min, lo));
            state.max = @max(state.max, @reduce(.Max, hi));
            state.count += count;
            state.nulls += data.length - count;
        }

        fn updateScalar(state: *State, s: Scalar) KernelError!void {
            if (s.isNull()) {
                state.nulls += 1;
                return;
            }
            const v = try dense.scalarAs(T, s);
            state.min = @min(state.min, v);
            state.max = @max(state.max, v);
            state.count += 1;
        }

        fn merge(state: *State, other: *const State) void {
            state.min = @min(state.min, other.min);
            state.max = @max(state.max, other.max);
            state.count += other.count;
            state.nulls += other.nulls;
        }

        fn finalize(ctx: *ExecContext, state: *const State, options: Options) KernelError!Datum {
            const valid = state.count != 0 and !nullByOptions(aggregateOptions(options), state.count, state.nulls);
            if (which != .min_max) {
                if (!valid) return nullScalar(out_type);
                return .{ .scalar = Scalar.init(out_type, dense.scalarValueOf(T, if (which == .min) state.min else state.max)) };
            }
            var mins = try oneValue(ctx.allocator, valid, state.min);
            var maxs = oneValue(ctx.allocator, valid, state.max) catch |err| {
                mins.release();
                return err;
            };
            var children = [2]ArrayRef{ mins, maxs };
            const no_validity = [1]buffer.SharedBuffer{.empty};
            var payload = ArrayRef.fromOwnedInline(ctx.allocator, .{
                .data_type = struct_type,
                .length = 1,
                .null_count = 0,
                .buffers = &no_validity,
                .children = &children,
            }) catch {
                mins.release();
                maxs.release();
                return error.OutOfMemory;
            };
            defer payload.release();
            return .{ .scalar = try Scalar.initNested(struct_type, payload) };
        }

        fn oneValue(allocator: std.mem.Allocator, valid: bool, v: T) KernelError!ArrayRef {
            var builder = primitive_array.PrimitiveBuilder(T, elem_type).init(allocator, 1) catch return error.OutOfMemory;
            defer builder.deinit();
            (if (valid) builder.append(v) else builder.appendNull()) catch return error.OutOfMemory;
            return builder.finish() catch error.OutOfMemory;
        }
    };
}

fn Count(comptime which: enum { count, count_all }) type {
    return struct {
        const State = struct { valid: u64 = 0, total: u64 = 0 };
        const out_type = DataType{ .int64 = {} };

        fn update(state: *State, data: *const ArrayData) KernelError!void {
            const nulls = if (datatype.storageDataType(data.data_type) == .null)
                data.length
            else if (dense.validityBits(data)) |bits|
                bitmap.countUnsetBitsInRange(bits, data.offset, data.offset + data.length)
            else
                0;
            state.total += data.length;
            state.valid += data.length - nulls;
        }

        fn updateScalar(state: *State, s: Scalar) KernelError!void {
            state.total += 1;
            if (!s.isNull()) state.valid += 1;
        }

        fn merge(state: *State, other: *const State) void {
            state.valid += other.valid;
            state.total += other.total;
        }

        fn finalize(_: *ExecContext, state: *const State, options: Options) KernelError!Datum {
            const mode: core.CountMode = if (which == .count_all) .all else if (options == .count) options.count.mode else .only_valid;
            const n = switch (mode) {
                .only_valid => state.valid,
                .only_null => state.total - state.valid,
                .all => state.total,
            };
            return .{ .scalar = Scalar.init(out_type, .{ .i64 = std.math.cast(i64, n) orelse return error.Overflow }) };
        }
    };
}

fn AnyAll(comptime which: enum { any, all }) type {
    return struct {
        const State = struct { any_true: bool = false, any_false: bool = false, count: u64 = 0, nulls: u64 = 0 };
        const out_type = DataType{ .bool = {} };

        fn update(state: *State, data: *const ArrayData) KernelError!void {
            const vals = Bits.values(data);
            const valid = Bits.validity(data);
            const n = bitmap.byteLength(data.length);
            var trues: u8 = 0;
            var falses: u8 = 0;
            var count: u64 = 0;
            for (0..n) |j| {
                var m = valid.byte(j);
                if (j + 1 == n and data.length % 8 != 0) m &= (@as(u8, 1) << @as(u3, @intCast(data.length % 8))) - 1;
                const v = vals.byte(j);
                trues |= v & m;
                falses |= ~v & m;
                count += @popCount(m);
            }
            state.any_true = state.any_true or trues != 0;
            state.any_false = state.any_false or falses != 0;
            state.count += count;
            state.nulls += data.length - count;
        }

        fn updateScalar(state: *State, s: Scalar) KernelError!void {
            if (s.isNull()) {
                state.nulls += 1;
                return;
            }
            if (try dense.scalarAs(bool, s)) state.any_true = true else state.any_false = true;
            state.count += 1;
        }

        fn merge(state: *State, other: *const State) void {
            state.any_true = state.any_true or other.any_true;
            state.any_false = state.any_false or other.any_false;
            state.count += other.count;
            state.nulls += other.nulls;
        }

        fn finalize(_: *ExecContext, state: *const State, options: Options) KernelError!Datum {
            const o = aggregateOptions(options);
            const result = if (which == .any) state.any_true else !state.any_false;
            // Kleene: a null row decides unless a true (any) / false (all) row already did.
            const decided = if (which == .any) state.any_true else state.any_false;
            if ((!o.skip_nulls and !decided and state.nulls > 0) or state.count < o.min_count) return nullScalar(out_type);
            return .{ .scalar = Scalar.init(out_type, .{ .bool = result }) };
        }
    };
}
