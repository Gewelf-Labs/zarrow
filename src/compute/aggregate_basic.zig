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
//! - decimal128 / decimal256 (as Arrow; decimal32 / decimal64 have no aggregate kernels there):
//!   sum gives the widest precision of the width (38 / 76) at the input scale, exact in 256 / 512
//!   bits until finalize, which applies `overflow_mode` (`checked`: `Overflow` past the precision,
//!   `wrapping`: the low 128 / 256 bits as Arrow C++, `saturating`: clamp to the precision); mean
//!   gives the input type, rounded half away from zero (null with no valid row whatever
//!   `min_count`, as Arrow); min / max give the input type; min_max a `struct<min, max>` of the
//!   input type, stored with its payload array (`dense.structArray`; the result type comes from
//!   `resolveResultTypeAlloc`).
const std = @import("std");
const bitmap = @import("../bitmap.zig");
const buffer = @import("../buffer.zig");
const core = @import("core.zig");
const datatype = @import("../datatype.zig");
const decimal = @import("decimal.zig");
const dense = @import("dense.zig");
const is_null = @import("is_null.zig");
const primitive_array = @import("../array/primitive_array.zig");
const sort = @import("sort.zig");

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
    inline for (.{ i128, i256 }) |T| {
        const check = comptime dense.argsOfType(if (T == i128) .decimal128 else .decimal256, 1);
        try add(registry, "sum", 1, check, scalarAggregateOptions, DecimalSum(T, .sum));
        try add(registry, "mean", 1, check, scalarAggregateOptions, DecimalSum(T, .mean));
        try add(registry, "min", 1, check, scalarAggregateOptions, DecimalMinMax(T, .min));
        try add(registry, "max", 1, check, scalarAggregateOptions, DecimalMinMax(T, .max));
        try add(registry, "min_max", 1, check, scalarAggregateOptions, DecimalMinMax(T, .min_max));
    }
    try add(registry, "min", 1, orderedArg, scalarAggregateOptions, OrderedMinMax(.min));
    try add(registry, "max", 1, orderedArg, scalarAggregateOptions, OrderedMinMax(.max));
    try add(registry, "min_max", 1, orderedArg, scalarAggregateOptions, OrderedMinMax(.min_max));
}

fn noArgs(args: []const Datum) bool {
    return args.len == 0;
}

/// `none` (Arrow defaults) or `ScalarAggregateOptions`.
pub fn scalarAggregateOptions(options: Options) bool {
    return options == .none or options == .scalar_aggregate;
}

/// `none` (only_valid) or `CountOptions`.
pub fn countOptions(options: Options) bool {
    return options == .none or options == .count;
}

pub fn aggregateOptions(options: Options) core.ScalarAggregateOptions {
    return if (options == .scalar_aggregate) options.scalar_aggregate else .{};
}

/// Whether the options make the result null: a null row with `skip_nulls` false, or fewer valid
/// rows than `min_count` (Arrow C++).
pub fn nullByOptions(o: core.ScalarAggregateOptions, count: u64, nulls: u64) bool {
    return (!o.skip_nulls and nulls > 0) or count < o.min_count;
}

/// Whether `Impl` declares a function `name` (a `void` declaration stands for none).
pub fn hasFn(comptime Impl: type, comptime name: []const u8) bool {
    if (!@hasDecl(Impl, name)) return false;
    return @TypeOf(@field(Impl, name)) != void;
}

/// Register `Impl` (see `Aggregate`) as an aggregate kernel of `name`. A result type that needs
/// storage comes from `Impl.resultTypeAlloc`, a plain one from `Impl.resultType` or `out_type`.
pub fn add(registry: *FunctionRegistry, comptime name: []const u8, comptime arity: usize, comptime check: TypeCheckFn, comptime options_check: core.OptionsCheckFn, comptime Impl: type) KernelError!void {
    const A = Aggregate(Impl);
    const alloc = comptime hasFn(Impl, "resultTypeAlloc");
    const plain = comptime hasFn(Impl, "resultType");
    try registry.registerAggregateKernel(name, .{
        .signature = .{
            .arity = arity,
            .type_check = check,
            .options_check = options_check,
            .result_type_fn = if (alloc) null else if (plain) Impl.resultType else dense.resultTypeOf(Impl.out_type),
            .result_type_alloc_fn = if (alloc) Impl.resultTypeAlloc else null,
        },
        .exec = A.exec,
        .aggregate_lifecycle = .{ .init = A.init, .update = A.update, .merge = A.merge, .finalize = A.finalize, .deinit = A.deinit },
    });
}

/// Exec and lifecycle callbacks around an `Impl` with `State`, `update` (one validated array:
/// its `ArrayData`, or its `ArrayRef` when `Impl` declares `update_takes_ref`), `updateScalar`, `merge` (may
/// fail) and `finalize`. A state that owns memory is built by `Impl.initState(ctx, options)` and
/// freed by `Impl.deinitState`.
pub fn Aggregate(comptime Impl: type) type {
    const State = Impl.State;
    const takes_ref = @hasDecl(Impl, "update_takes_ref");
    return struct {
        fn consume(state: *State, datum: Datum) KernelError!void {
            switch (datum) {
                .array => |arr| try consumeArray(state, arr),
                .chunked => |chunked| for (chunked.chunks()) |chunk| try consumeArray(state, chunk),
                .scalar => |s| try Impl.updateScalar(state, s),
                .record_batch, .table => return error.InvalidInput,
            }
        }

        fn consumeArray(state: *State, arr: ArrayRef) KernelError!void {
            arr.data().validateLayout() catch return error.InvalidInput;
            try Impl.update(state, if (takes_ref) arr else arr.data());
        }

        fn newState(ctx: *ExecContext, options: Options) KernelError!State {
            return if (@hasDecl(Impl, "initState")) Impl.initState(ctx, options) else .{};
        }

        fn freeState(state: *State) void {
            if (@hasDecl(Impl, "deinitState")) Impl.deinitState(state);
        }

        /// Aggregate one array, chunked array or scalar into a scalar (count_all: also none).
        pub fn exec(ctx: *ExecContext, args: []const Datum, options: Options) KernelError!Datum {
            if (args.len > 1) return error.InvalidArity;
            var state = try newState(ctx, options);
            defer freeState(&state);
            // Decimal states keep their input type (the result type depends on it).
            if (@hasField(State, "dt") and args.len == 1) state.dt = args[0].dataType();
            if (args.len == 1) try consume(&state, args[0]);
            return Impl.finalize(ctx, &state, options);
        }

        pub fn init(ctx: *ExecContext, options: Options) KernelError!*anyopaque {
            const state = ctx.allocator.create(State) catch return error.OutOfMemory;
            state.* = newState(ctx, options) catch |err| {
                ctx.allocator.destroy(state);
                return err;
            };
            return state;
        }

        pub fn update(_: *ExecContext, state: *anyopaque, args: []const Datum, _: Options) KernelError!void {
            if (args.len > 1) return error.InvalidArity;
            if (args.len == 1) try consume(cast(state), args[0]);
        }

        pub fn merge(_: *ExecContext, state: *anyopaque, other: *anyopaque, _: Options) KernelError!void {
            const r = Impl.merge(cast(state), cast(other));
            if (@TypeOf(r) != void) try r;
        }

        pub fn finalize(ctx: *ExecContext, state: *anyopaque, options: Options) KernelError!Datum {
            return Impl.finalize(ctx, cast(state), options);
        }

        pub fn deinit(ctx: *ExecContext, state: *anyopaque) void {
            freeState(cast(state));
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

pub fn values(comptime T: type, data: *const ArrayData) KernelError![]const T {
    return (primitive_array.PrimitiveArray(T){ .data = data }).values() catch error.InvalidInput;
}

/// Load 8 values starting at `i` (zero past the end).
pub inline fn lanes(comptime T: type, vals: []const T, i: usize) @Vector(8, T) {
    if (i + 8 <= vals.len) return vals[i..][0..8].*;
    var tmp: [8]T = @splat(0);
    @memcpy(tmp[0 .. vals.len - i], vals[i..]);
    return tmp;
}

pub inline fn popCount(mask: @Vector(8, bool)) u64 {
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

// ------------------------------------------------------------------------------ decimals

/// The decimal type with the tag of the storage `T` (decimal128 / decimal256) and `p`.
fn decimalType(comptime T: type, p: decimal.Params) DataType {
    return if (T == i128) .{ .decimal128 = p } else .{ .decimal256 = p };
}

fn decimalScalar(comptime T: type, dt: DataType, v: T) Scalar {
    return Scalar.init(dt, @unionInit(core.ScalarValue, if (T == i128) "decimal128" else "decimal256", v));
}

/// `sum / count` rounded half away from zero (Arrow's decimal mean).
pub fn roundedMean(comptime Exact: type, sum: Exact, count: u64) Exact {
    const n: Exact = @intCast(count);
    const q = @divTrunc(sum, n);
    const r = @rem(sum, n);
    if (2 * @abs(r) >= @abs(n)) return if (sum < 0) q - 1 else q + 1;
    return q;
}

/// The exact sum of a decimal group as the result storage `T` per `ExecConfig.overflow_mode`:
/// `checked` rejects a sum past `10^max_p`, `wrapping` keeps the low bits (Arrow C++),
/// `saturating` clamps to the precision.
pub fn decimalSumValue(comptime T: type, comptime Exact: type, ctx: *ExecContext, sum: Exact) KernelError!T {
    const max_p: u8 = if (T == i128) 38 else 76;
    const limit: Exact = @intCast(decimal.pow10(i256, max_p));
    return switch (ctx.overflowMode()) {
        .checked => if (@abs(sum) < @abs(limit)) @intCast(sum) else error.Overflow,
        .wrapping => @truncate(sum),
        .saturating => @intCast(std.math.clamp(sum, -limit + 1, limit - 1)),
    };
}

fn DecimalSum(comptime T: type, comptime which: enum { sum, mean }) type {
    // Exact: 2^64 rows of |v| < 2^(bits - 1) need 64 more bits.
    const Exact = if (T == i128) i256 else i512;
    const max_p: u8 = if (T == i128) 38 else 76;
    return struct {
        const State = struct { sum: Exact = 0, count: u64 = 0, nulls: u64 = 0, dt: DataType = .{ .null = {} } };

        fn outType(dt: DataType) KernelError!DataType {
            const p = try decimal.checkedParams(dt);
            return if (which == .mean) dt else decimalType(T, .{ .precision = max_p, .scale = p.scale });
        }

        pub fn resultType(args: []const Datum, _: Options) KernelError!DataType {
            if (args.len != 1) return error.InvalidArity;
            return outType(args[0].dataType());
        }

        fn update(state: *State, data: *const ArrayData) KernelError!void {
            _ = try decimal.checkedParams(data.data_type);
            state.dt = data.data_type;
            const vals = try values(T, data);
            var sum: Exact = 0;
            var count: u64 = 0;
            if (dense.validityBits(data)) |bits| {
                for (vals, 0..) |v, i| if (bitmap.bitIsSet(bits, data.offset + i)) {
                    sum += v;
                    count += 1;
                };
            } else {
                for (vals) |v| sum += v;
                count = vals.len;
            }
            state.sum += sum;
            state.count += count;
            state.nulls += data.length - count;
        }

        fn updateScalar(state: *State, s: Scalar) KernelError!void {
            _ = try decimal.checkedParams(s.data_type);
            state.dt = s.data_type;
            if (s.isNull()) {
                state.nulls += 1;
                return;
            }
            state.sum += try decimal.scalarInt(T, s);
            state.count += 1;
        }

        fn merge(state: *State, other: *const State) void {
            state.sum += other.sum;
            state.count += other.count;
            state.nulls += other.nulls;
            if (state.dt == .null) state.dt = other.dt;
        }

        fn finalize(ctx: *ExecContext, state: *const State, options: Options) KernelError!Datum {
            // A lifecycle finalized before any update has no input type.
            if (state.dt == .null) return nullScalar(.{ .null = {} });
            const out_type = try outType(state.dt);
            if (nullByOptions(aggregateOptions(options), state.count, state.nulls)) return nullScalar(out_type);
            if (which == .mean) {
                if (state.count == 0) return nullScalar(out_type);
                return .{ .scalar = decimalScalar(T, out_type, @intCast(roundedMean(Exact, state.sum, state.count))) };
            }
            return .{ .scalar = decimalScalar(T, out_type, try decimalSumValue(T, Exact, ctx, state.sum)) };
        }
    };
}

/// Names of the fields of a min_max result.
pub const min_max_names = [_][]const u8{ "min", "max" };

fn DecimalMinMax(comptime T: type, comptime which: enum { min, max, min_max }) type {
    return struct {
        const State = struct { min: T = std.math.maxInt(T), max: T = std.math.minInt(T), count: u64 = 0, nulls: u64 = 0, dt: DataType = .{ .null = {} } };

        pub const resultType = if (which == .min_max) {} else struct {
            fn f(args: []const Datum, _: Options) KernelError!DataType {
                if (args.len != 1) return error.InvalidArity;
                _ = try decimal.checkedParams(args[0].dataType());
                return args[0].dataType();
            }
        }.f;

        /// min_max: `struct<min: T, max: T>` over the input's decimal type.
        pub const resultTypeAlloc = if (which != .min_max) {} else struct {
            fn f(allocator: std.mem.Allocator, args: []const Datum, _: Options) KernelError!DataType {
                if (args.len != 1) return error.InvalidArity;
                const dt = args[0].dataType();
                _ = try decimal.checkedParams(dt);
                return dense.structTypeAlloc(allocator, &min_max_names, &.{ dt, dt });
            }
        }.f;

        fn update(state: *State, data: *const ArrayData) KernelError!void {
            _ = try decimal.checkedParams(data.data_type);
            state.dt = data.data_type;
            const vals = try values(T, data);
            var lo = state.min;
            var hi = state.max;
            var count: u64 = 0;
            if (dense.validityBits(data)) |bits| {
                for (vals, 0..) |v, i| if (bitmap.bitIsSet(bits, data.offset + i)) {
                    lo = @min(lo, v);
                    hi = @max(hi, v);
                    count += 1;
                };
            } else {
                for (vals) |v| {
                    lo = @min(lo, v);
                    hi = @max(hi, v);
                }
                count = vals.len;
            }
            state.min = lo;
            state.max = hi;
            state.count += count;
            state.nulls += data.length - count;
        }

        fn updateScalar(state: *State, s: Scalar) KernelError!void {
            _ = try decimal.checkedParams(s.data_type);
            state.dt = s.data_type;
            if (s.isNull()) {
                state.nulls += 1;
                return;
            }
            const v = try decimal.scalarInt(T, s);
            state.min = @min(state.min, v);
            state.max = @max(state.max, v);
            state.count += 1;
        }

        fn merge(state: *State, other: *const State) void {
            state.min = @min(state.min, other.min);
            state.max = @max(state.max, other.max);
            state.count += other.count;
            state.nulls += other.nulls;
            if (state.dt == .null) state.dt = other.dt;
        }

        fn finalize(ctx: *ExecContext, state: *const State, options: Options) KernelError!Datum {
            if (state.dt == .null) return nullScalar(.{ .null = {} });
            const valid = state.count != 0 and !nullByOptions(aggregateOptions(options), state.count, state.nulls);
            if (which != .min_max) {
                if (!valid) return nullScalar(state.dt);
                return .{ .scalar = decimalScalar(T, state.dt, if (which == .min) state.min else state.max) };
            }
            var lo = try oneDecimal(T, ctx.allocator, state.dt, if (valid) state.min else null);
            var hi = oneDecimal(T, ctx.allocator, state.dt, if (valid) state.max else null) catch |err| {
                lo.release();
                return err;
            };
            _ = &hi;
            var payload = try dense.structArray(ctx.allocator, &min_max_names, &.{ lo, hi }, 1, null);
            defer payload.release();
            return .{ .scalar = try Scalar.initNested(payload.data().data_type, payload) };
        }
    };
}

/// A one-row decimal array of type `dt` holding `v` (null when `v` is null).
pub fn oneDecimal(comptime T: type, allocator: std.mem.Allocator, dt: DataType, v: ?T) KernelError!ArrayRef {
    var buf = buffer.OwnedBuffer.init(allocator, @sizeOf(T)) catch return error.OutOfMemory;
    var validity: ?buffer.OwnedBuffer = null;
    if (v) |x| {
        buf.typedSlice(T)[0] = x;
    } else {
        validity = dense.bitsBuffer(allocator, 1) catch |err| {
            buf.deinit();
            return err;
        };
        validity.?.data[0] = 0;
    }
    return dense.finishFixed(allocator, dt, 1, &buf, validity);
}

// ------------------------------------------------------------------------------ ordered types

/// Whether min / max / min_max take `dt` through `OrderedMinMax`: the null type, bool, the
/// temporal types, decimal32 / decimal64 and the string / binary layouts with
/// fixed_size_binary (pyarrow 26 has all but duration, the views and decimal32 / decimal64).
pub fn isOrderedType(dt: DataType) bool {
    return switch (dt) {
        .null, .bool, .date32, .date64, .timestamp, .time32, .time64, .duration, .decimal32, .decimal64 => true,
        .string, .large_string, .string_view, .binary, .large_binary, .binary_view, .fixed_size_binary => true,
        else => false,
    };
}

fn orderedArg(args: []const Datum) bool {
    return args.len == 1 and isOrderedType(args[0].dataType());
}

/// The rows of the smallest and largest valid value of a chunk and its valid row count.
pub const Extremes = struct { lo: usize = 0, hi: usize = 0, count: u64 = 0 };

/// Scan the valid rows of `data` (typed rows `vals`).
pub fn extremesOf(data: *const ArrayData, vals: *const sort.Values) Extremes {
    return switch (vals.*) {
        .bytes => |op| switch (op) {
            inline else => |r| scanRows(data, r),
        },
        .fixed => |f| scanRows(data, FixedRows{ .data = f.data, .width = f.width, .offset = f.offset }),
        .boolean => |b| scanRows(data, BoolRows{ .bits = b.bits, .offset = b.offset }),
        inline .i32, .i64 => |s| scanRows(data, SliceRows(@TypeOf(s[0])){ .vals = s }),
        // isOrderedType admits no other storage.
        else => .{},
    };
}

const FixedRows = struct {
    data: []const u8,
    width: usize,
    offset: usize,
    inline fn get(self: FixedRows, i: usize) []const u8 {
        return self.data[(self.offset + i) * self.width ..][0..self.width];
    }
};

const BoolRows = struct {
    bits: []const u8,
    offset: usize,
    inline fn get(self: BoolRows, i: usize) u1 {
        return @intFromBool(bitmap.bitIsSet(self.bits, self.offset + i));
    }
};

fn SliceRows(comptime T: type) type {
    return struct {
        vals: []const T,
        inline fn get(self: @This(), i: usize) T {
            return self.vals[i];
        }
    };
}

/// The first 8 bytes of `s` as a big-endian integer, zero padded: integer order is byte order
/// up to a tie (equal keys: compare the rest).
inline fn prefixKey(s: []const u8) u64 {
    const n = s.len;
    if (n >= 8) return std.mem.readInt(u64, s[0..8], .big);
    // Short rows without a copy: two overlapping loads land every byte at its place.
    if (n >= 4) {
        const first: u64 = std.mem.readInt(u32, s[0..4], .big);
        const last: u64 = std.mem.readInt(u32, s[n - 4 ..][0..4], .big);
        return (first << 32) | (last << @intCast(64 - 8 * n));
    }
    if (n == 0) return 0;
    const b0: u64 = s[0];
    const bm: u64 = s[n / 2];
    const bl: u64 = s[n - 1];
    return (b0 << 56) | (bm << @intCast(56 - 8 * (n / 2))) | (bl << @intCast(56 - 8 * (n - 1)));
}

/// Byte order of `a` before `b`, 8 bytes at a time.
fn bytesLess(a: []const u8, b: []const u8) bool {
    var i: usize = 0;
    while (true) : (i += 8) {
        const ka = prefixKey(a[i..]);
        const kb = prefixKey(b[i..]);
        if (ka != kb) return ka < kb;
        // Equal padded keys: one ran out (the shorter is first) or both go on.
        if (a.len - i <= 8 or b.len - i <= 8) return a.len < b.len;
    }
}

/// The padded key of bytes 8..16 (0 for a row of at most 8 bytes).
inline fn restKey(s: []const u8) u64 {
    return if (s.len > 8) prefixKey(s[8..]) else 0;
}

/// Order of two byte rows whose padded 16-byte prefixes are equal: a row of at most 16 bytes
/// is the shorter one, else the bytes from 16 on decide.
inline fn tailLess(a: []const u8, b: []const u8) bool {
    if (a.len <= 16 or b.len <= 16) return a.len < b.len;
    return bytesLess(a[16..], b[16..]);
}

fn scanRows(data: *const ArrayData, rows: anytype) Extremes {
    if (@TypeOf(rows.get(0)) == []const u8) return scanBytes(data, rows);
    const bits = dense.validityBits(data);
    var i: usize = 0;
    if (bits) |b| {
        while (i < data.length and !bitmap.bitIsSet(b, data.offset + i)) i += 1;
    }
    if (i == data.length) return .{};
    var out: Extremes = .{ .lo = i, .hi = i, .count = 1 };
    var lo = rows.get(i);
    var hi = lo;
    for (i + 1..data.length) |j| {
        if (bits) |b| if (!bitmap.bitIsSet(b, data.offset + j)) continue;
        const x = rows.get(j);
        // A value below the minimum cannot be above the maximum.
        if (x < lo) {
            lo = x;
            out.lo = j;
        } else if (hi < x) {
            hi = x;
            out.hi = j;
        }
        out.count += 1;
    }
    return out;
}

/// Byte rows in blocks: the 8-byte prefix key of every row goes to a stack buffer with the
/// block's smallest and largest key, then only the rows of the extreme keys are compared
/// further (the next 8 bytes, then the rest).
fn scanBytes(data: *const ArrayData, rows: anytype) Extremes {
    const bits = dense.validityBits(data);
    const n = data.length;
    const Best = struct {
        row: ?usize = null,
        k: u64 = 0,
        k2: u64 = 0,
        v: []const u8 = &.{},
    };
    var lo: Best = .{};
    var hi: Best = .{};
    var count: u64 = 0;
    var keys: [1024]u64 = undefined;
    var start: usize = 0;
    while (start < n) : (start += keys.len) {
        const m = @min(keys.len, n - start);
        var block_lo: u64 = std.math.maxInt(u64);
        var block_hi: u64 = 0;
        var valid: usize = m;
        if (bits) |b| {
            valid = 0;
            for (keys[0..m], start..) |*k, j| {
                k.* = prefixKey(rows.get(j));
                if (bitmap.bitIsSet(b, data.offset + j)) {
                    block_lo = @min(block_lo, k.*);
                    block_hi = @max(block_hi, k.*);
                    valid += 1;
                }
            }
        } else for (keys[0..m], start..) |*k, j| {
            k.* = prefixKey(rows.get(j));
            block_lo = @min(block_lo, k.*);
            block_hi = @max(block_hi, k.*);
        }
        if (valid == 0) continue;
        count += valid;
        if (lo.row == null or block_lo < lo.k) lo = .{ .k = block_lo };
        if (hi.row == null or block_hi > hi.k) hi = .{ .k = block_hi };
        for (keys[0..m], start..) |k, j| {
            if (k != lo.k and k != hi.k) continue;
            if (bits) |b| if (!bitmap.bitIsSet(b, data.offset + j)) continue;
            const x = rows.get(j);
            const k2 = restKey(x);
            if (k == lo.k and (lo.row == null or k2 < lo.k2 or (k2 == lo.k2 and tailLess(x, lo.v)))) lo = .{ .row = j, .k = k, .k2 = k2, .v = x };
            if (k == hi.k and (hi.row == null or k2 > hi.k2 or (k2 == hi.k2 and tailLess(hi.v, x)))) hi = .{ .row = j, .k = k, .k2 = k2, .v = x };
        }
    }
    if (count == 0) return .{};
    return .{ .lo = lo.row.?, .hi = hi.row.?, .count = count };
}

/// min / max / min_max of the ordered non-numeric types (`isOrderedType`): bool (false < true),
/// dates, timestamps, times and durations and decimal32 / decimal64 by value, the string /
/// binary layouts and fixed_size_binary by bytes. The state keeps one-row slices of the input
/// holding the smallest and largest value so far; the options rules are those of `MinMax`
/// (null with no valid row whatever `min_count`; min_max gives `struct<min, max>` of the input
/// type, stored with its payload array).
fn OrderedMinMax(comptime which: enum { min, max, min_max }) type {
    return struct {
        pub const update_takes_ref = true;

        pub const State = struct {
            allocator: std.mem.Allocator,
            min: ?ArrayRef = null,
            max: ?ArrayRef = null,
            /// min_max: a one-row slice of the first null input row, the null field of a null
            /// result without a new value (a fixed_size_binary may be wide).
            null_row: ?ArrayRef = null,
            count: u64 = 0,
            nulls: u64 = 0,
            dt: DataType = .{ .null = {} },
        };

        pub const resultType = if (which == .min_max) {} else struct {
            fn f(args: []const Datum, _: Options) KernelError!DataType {
                if (args.len != 1) return error.InvalidArity;
                return args[0].dataType();
            }
        }.f;

        pub const resultTypeAlloc = if (which != .min_max) {} else struct {
            fn f(allocator: std.mem.Allocator, args: []const Datum, _: Options) KernelError!DataType {
                if (args.len != 1) return error.InvalidArity;
                const dt = args[0].dataType();
                return dense.structTypeAlloc(allocator, &min_max_names, &.{ dt, dt });
            }
        }.f;

        pub fn initState(ctx: *ExecContext, _: Options) KernelError!State {
            return .{ .allocator = ctx.allocator };
        }

        pub fn deinitState(state: *State) void {
            if (state.min) |*a| a.release();
            if (state.max) |*a| a.release();
            if (state.null_row) |*a| a.release();
        }

        /// Keep row `row` of `arr` in `slot` when the slot is empty or the row orders `want`
        /// (.lt: smaller, .gt: larger) against the value held.
        fn offer(slot: *?ArrayRef, arr: ArrayRef, vals: *const sort.Values, row: usize, want: std.math.Order) KernelError!void {
            if (slot.*) |held| {
                const hv = try sort.valuesOf(held);
                if (sort.valueOrder(vals, row, &hv, 0) != want) return;
            }
            const one = arr.slice(row, 1) catch return error.OutOfMemory;
            if (slot.*) |*old| old.release();
            slot.* = one;
        }

        pub fn update(state: *State, arr: ArrayRef) KernelError!void {
            const data = arr.data();
            if (!isOrderedType(data.data_type)) return error.InvalidInput;
            if (state.dt != .null and !state.dt.eql(data.data_type)) return error.InvalidInput;
            state.dt = data.data_type;
            if (data.data_type == .null) {
                state.nulls += data.length;
                return;
            }
            const vals = try sort.valuesOf(arr);
            const ext = extremesOf(data, &vals);
            state.nulls += data.length - ext.count;
            if (which == .min_max and state.null_row == null and ext.count < data.length) {
                var i: usize = 0;
                while (!data.isNull(i)) i += 1;
                state.null_row = arr.slice(i, 1) catch return error.OutOfMemory;
            }
            if (ext.count == 0) return;
            state.count += ext.count;
            if (which != .max) try offer(&state.min, arr, &vals, ext.lo, .lt);
            if (which != .min) try offer(&state.max, arr, &vals, ext.hi, .gt);
        }

        pub fn updateScalar(state: *State, s: Scalar) KernelError!void {
            var arr = try dense.chunkArray(state.allocator, .{ .scalar = s });
            defer arr.release();
            arr.data().validateLayout() catch return error.InvalidInput;
            try update(state, arr);
        }

        pub fn merge(state: *State, other: *const State) KernelError!void {
            if (state.dt != .null and other.dt != .null and !state.dt.eql(other.dt)) return error.InvalidInput;
            if (other.min) |a| try offer(&state.min, a, &(try sort.valuesOf(a)), 0, .lt);
            if (other.max) |a| try offer(&state.max, a, &(try sort.valuesOf(a)), 0, .gt);
            if (state.null_row == null) if (other.null_row) |a| {
                state.null_row = a.retain();
            };
            state.count += other.count;
            state.nulls += other.nulls;
            if (state.dt == .null) state.dt = other.dt;
        }

        pub fn finalize(ctx: *ExecContext, state: *const State, options: Options) KernelError!Datum {
            const valid = state.count != 0 and !nullByOptions(aggregateOptions(options), state.count, state.nulls);
            if (which != .min_max) {
                const held = if (which == .min) state.min else state.max;
                if (!valid or held == null) return nullScalar(state.dt);
                return .{ .scalar = try dense.scalarAt0(held.?) };
            }
            // An empty input has no row to lend its buffers to the null fields: a value of a
            // fixed_size_binary past 16 MiB is not allocated from nothing (hostile schema).
            if (state.null_row == null and state.dt == .fixed_size_binary and state.dt.fixed_size_binary.byte_width > 1 << 24 and !(valid and state.min != null)) return error.UnsupportedType;
            var parts: [2]ArrayRef = undefined;
            for (&parts, [_]?ArrayRef{ state.min, state.max }, 0..) |*p, held, k| {
                p.* = if (valid and held != null) held.?.retain() else if (state.null_row) |r| r.retain() else dense.nullArray(ctx.allocator, state.dt, 1) catch |err| {
                    for (parts[0..k]) |*q| q.release();
                    return err;
                };
            }
            var payload = try dense.structArray(ctx.allocator, &min_max_names, &parts, 1, null);
            defer payload.release();
            return .{ .scalar = try Scalar.initNested(payload.data().data_type, payload) };
        }
    };
}
