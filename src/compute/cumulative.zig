//! Cumulative, statistical and pairwise functions (Arrow C++ "Cumulative functions", "Pairwise
//! functions", winsorize): vector functions keeping the input length.
//!
//! - cumulative_sum, cumulative_prod (+ `_checked`), cumulative_min, cumulative_max and
//!   cumulative_mean (`CumulativeOptions`) on int8..int64, uint8..uint64, float32 and float64
//!   (pyarrow 26 has no other input type). Row `i` is the operation over rows `0..i` and the
//!   start value: the result has the input type (cumulative_mean: float64, the start ignored, as
//!   Arrow). A chunked input carries the running value across its chunks and keeps them; a scalar
//!   gives a one-row array (Arrow). `skip_nulls = false` (the default): every row from the first
//!   null on is null; `skip_nulls = true`: a null row is null and the accumulation goes on.
//!   Integer overflow wraps (Arrow C++) or saturates when `ExecConfig.overflow_mode` is
//!   `saturating`; the `_checked` forms return `Overflow`. Floats never overflow. min / max skip
//!   NaN (an all-NaN prefix gives the identity, inf / -inf, as pyarrow 26) and keep the later of
//!   two equal values (so -0.0 then 0.0 gives 0.0 for max). `CumulativeOptions.start` is a
//!   numeric scalar cast safely to the input type (`InvalidCast` when it does not fit,
//!   `InvalidOptions` for a non-numeric scalar; Arrow parses a string start, zarrow refuses it).
//! - pairwise_diff / pairwise_diff_checked (`PairwiseOptions.period`): row `i` is
//!   `x[i] - x[i - period]` through subtract / subtract_checked, so the types and overflow rules
//!   are subtract's (numbers keep their type, temporal types give durations, decimal128(p, s)
//!   gives decimal128(p + 1, s)); the first `period` rows (the last `-period` for a negative
//!   period) are null. A chunked input is concatenated first (Arrow rejects it) and gives a
//!   chunked result of one chunk.
//! - winsorize (`WinsorizeOptions`, required as in Arrow: 0 <= lower_limit <= upper_limit <= 1,
//!   else `InvalidOptions`) on numbers and decimals: values below the `lower_limit` quantile or
//!   above the `upper_limit` quantile are replaced by it; the quantiles are quantile's `nearest`
//!   interpolation over the valid non-NaN values (pyarrow 26 results); NaN and nulls stay.
//! - random (`RandomOptions`: length, initializer, seed): `length` float64 values in [0, 1) from
//!   std.Random.DefaultPrng seeded with `seed` (the sequence differs from Arrow's, which seeds
//!   its own generator); `initializer = system` is `NotImplemented` (no I/O source of
//!   randomness in the exec context). Nullary (no argument), as Arrow.
//!
//! Every input chunk passes `validateLayout` before its buffers are read.
const std = @import("std");
const bitmap = @import("../bitmap.zig");
const core = @import("core.zig");
const dense = @import("dense.zig");
const concat = @import("../concat_array_refs.zig");
const datatype = @import("../datatype.zig");

const ArrayData = @import("../array/array_data.zig").ArrayData;
const PrimitiveArray = @import("../array/primitive_array.zig").PrimitiveArray;
const OwnedBuffer = @import("../buffer.zig").OwnedBuffer;
const ArrayRef = core.ArrayRef;
const ChunkedArray = core.ChunkedArray;
const DataType = core.DataType;
const Datum = core.Datum;
const ExecContext = core.ExecContext;
const FunctionRegistry = core.FunctionRegistry;
const KernelError = core.KernelError;
const Options = core.Options;
const Scalar = core.Scalar;

const Op = enum { sum, sum_checked, prod, prod_checked, min, max, mean };

/// Register the cumulative functions, pairwise_diff (+ `_checked`), winsorize and random.
pub fn register(registry: *FunctionRegistry) KernelError!void {
    inline for (comptime std.meta.tags(Op)) |op| {
        const name = switch (op) {
            .sum => "cumulative_sum",
            .sum_checked => "cumulative_sum_checked",
            .prod => "cumulative_prod",
            .prod_checked => "cumulative_prod_checked",
            .min => "cumulative_min",
            .max => "cumulative_max",
            .mean => "cumulative_mean",
        };
        try registry.registerVectorKernel(name, .{
            .signature = .{ .arity = 1, .type_check = numericArg, .options_check = cumulativeOptions, .result_type_fn = Cumulative(op).resultType },
            .exec = Cumulative(op).exec,
        });
    }
    inline for (.{ "pairwise_diff", "pairwise_diff_checked" }) |name| {
        try registry.registerVectorKernel(name, .{
            .signature = .{ .arity = 1, .type_check = pairwiseArg, .options_check = pairwiseOptions },
            .exec = Pairwise(std.mem.endsWith(u8, name, "_checked")).exec,
        });
    }
    try registry.registerVectorKernel("winsorize", .{
        .signature = .{ .arity = 1, .type_check = winsorizeArg, .options_check = winsorizeOptions, .result_type_fn = sameType },
        .exec = winsorize,
    });
    try registry.registerScalarKernel("random", .{
        .signature = .{ .arity = 0, .type_check = noArgs, .options_check = randomOptions, .result_type_fn = dense.resultTypeOf(.{ .double = {} }) },
        .exec = random,
    });
}

fn isNumeric(dt: DataType) bool {
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

fn numericArg(args: []const Datum) bool {
    return args.len == 1 and !args[0].isTabular() and isNumeric(args[0].dataType());
}

fn pairwiseArg(args: []const Datum) bool {
    if (args.len != 1 or args[0].isTabular()) return false;
    return switch (args[0].dataType()) {
        .date32, .date64, .timestamp, .time32, .time64, .duration => true,
        else => |dt| isNumeric(dt) or isDecimal(dt),
    };
}

fn winsorizeArg(args: []const Datum) bool {
    return args.len == 1 and !args[0].isTabular() and (isNumeric(args[0].dataType()) or isDecimal(args[0].dataType()));
}

fn noArgs(args: []const Datum) bool {
    return args.len == 0;
}

fn cumulativeOptions(options: Options) bool {
    return options == .none or options == .cumulative;
}

fn pairwiseOptions(options: Options) bool {
    return options == .none or options == .pairwise;
}

fn winsorizeOptions(options: Options) bool {
    return options == .winsorize;
}

fn randomOptions(options: Options) bool {
    return options == .random;
}

fn sameType(args: []const Datum, _: Options) KernelError!DataType {
    if (args.len != 1) return error.InvalidArity;
    return args[0].dataType();
}

/// The array chunks of an array, chunked array or scalar (a one-row array, owned by `owned`).
/// `datum` must outlive the returned slice.
fn partsOf(allocator: std.mem.Allocator, datum: *const Datum, owned: *?ArrayRef) KernelError![]const ArrayRef {
    const parts: []const ArrayRef = switch (datum.*) {
        .array => |*a| a[0..1],
        .chunked => |c| c.chunks(),
        .scalar => |s| blk: {
            owned.* = try dense.chunkArray(allocator, .{ .scalar = s });
            break :blk @as(*const [1]ArrayRef, &owned.*.?)[0..1];
        },
        .record_batch, .table => return error.InvalidInput,
    };
    for (parts) |p| p.data().validateLayout() catch return error.InvalidInput;
    return parts;
}

/// `outs` (taken over) as a datum shaped like `like`: a chunked array for a chunked input,
/// else the one array.
fn shaped(allocator: std.mem.Allocator, like: Datum, out_type: DataType, outs: []ArrayRef) KernelError!Datum {
    if (like != .chunked) return .{ .array = outs[0] };
    defer for (outs) |*o| o.release();
    const c = ChunkedArray.init(allocator, out_type, outs) catch |err| return switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        else => error.InvalidInput,
    };
    return .{ .chunked = c };
}

// ------------------------------------------------------------------------------ cumulative

fn Cumulative(comptime op: Op) type {
    return struct {
        fn resultType(args: []const Datum, _: Options) KernelError!DataType {
            if (args.len != 1) return error.InvalidArity;
            return if (op == .mean) .{ .double = {} } else args[0].dataType();
        }

        fn exec(ctx: *ExecContext, args: []const Datum, options: Options) KernelError!Datum {
            if (args.len != 1) return error.InvalidArity;
            const o: core.CumulativeOptions = if (options == .cumulative) options.cumulative else .{};
            inline for (dense.numeric_types) |T| {
                if (std.meta.activeTag(args[0].dataType()) == comptime std.meta.activeTag(dense.dataTypeOf(T))) {
                    return run(T, ctx, args[0], o);
                }
            }
            return error.NoMatchingKernel;
        }

        const Mode = enum { wrap, saturate, check };

        fn run(comptime T: type, ctx: *ExecContext, datum: Datum, o: core.CumulativeOptions) KernelError!Datum {
            const mode: Mode = switch (op) {
                .sum_checked, .prod_checked => .check,
                else => if (ctx.overflowMode() == .saturating) .saturate else .wrap,
            };
            return switch (mode) {
                inline else => |m| runMode(T, m, ctx, datum, o),
            };
        }

        /// Running state across the chunks of one call.
        fn State(comptime T: type) type {
            return struct { acc: T, mean_sum: f64 = 0, mean_n: f64 = 0, poisoned: bool = false };
        }

        fn runMode(comptime T: type, comptime mode: Mode, ctx: *ExecContext, datum: Datum, o: core.CumulativeOptions) KernelError!Datum {
            const Out = if (op == .mean) f64 else T;
            var st: State(T) = .{ .acc = if (op == .sum or op == .sum_checked or op == .mean) 0 else identity(T) };
            if (op != .mean) if (try startValue(T, ctx, o.start)) |s| {
                st.acc = s;
            };
            var owned: ?ArrayRef = null;
            defer if (owned) |*a| a.release();
            const parts = try partsOf(ctx.allocator, &datum, &owned);
            const outs = ctx.allocator.alloc(ArrayRef, parts.len) catch return error.OutOfMemory;
            var made: usize = 0;
            defer {
                for (outs[0..made]) |*a| a.release();
                ctx.allocator.free(outs);
            }
            const out_type = dense.dataTypeOf(Out);
            for (parts) |p| {
                const d = p.data();
                var buf = OwnedBuffer.initUninit(ctx.allocator, d.length * @sizeOf(Out)) catch return error.OutOfMemory;
                const validity = fill(T, Out, mode, ctx.allocator, d, buf.typedSlice(Out)[0..d.length], &st, o.skip_nulls) catch |err| {
                    buf.deinit();
                    return err;
                };
                // finishFixed takes both buffers over, also on error.
                outs[made] = try dense.finishFixed(ctx.allocator, out_type, d.length, &buf, validity);
                made += 1;
            }
            const n = made;
            made = 0;
            return shaped(ctx.allocator, datum, out_type, outs[0..n]);
        }

        /// Write the running values of one chunk into `ys`; its validity (null: no null row).
        fn fill(comptime T: type, comptime Out: type, comptime mode: Mode, allocator: std.mem.Allocator, d: *const ArrayData, ys: []Out, st: *State(T), skip_nulls: bool) KernelError!?OwnedBuffer {
            const xs = (PrimitiveArray(T){ .data = d }).values() catch return error.InvalidInput;
            const n = d.length;
            const bits = dense.validityBits(d);
            if (bits == null and !st.poisoned) {
                // No null: a tight loop.
                for (ys, xs) |*y, x| y.* = try step(T, Out, mode, st, x);
                return null;
            }
            var validity = try dense.bitsBuffer(allocator, n);
            errdefer validity.deinit();
            const vb = validity.data[0..bitmap.byteLength(n)];
            @memset(vb, 0);
            if (!skip_nulls) {
                // The rows before the first null (none once poisoned) are valid, the rest null.
                var first: usize = 0;
                if (!st.poisoned) {
                    while (first < n and bitmap.bitIsSet(bits.?, d.offset + first)) first += 1;
                }
                for (ys[0..first], xs[0..first]) |*y, x| y.* = try step(T, Out, mode, st, x);
                @memset(ys[first..], 0);
                bitmap.setBits(vb, 0, first);
                if (first < n) st.poisoned = true;
            } else {
                // A null row is null and leaves the running value as it is.
                const b = bits.?;
                if (op == .mean) {
                    for (ys, xs, 0..) |*y, x, i| {
                        y.* = if (bitmap.bitIsSet(b, d.offset + i)) try step(T, Out, mode, st, x) else 0;
                    }
                } else {
                    // Branch-free: a null row adds the operation's identity.
                    const id = identity(T);
                    for (ys, xs, 0..) |*y, x, i| y.* = try step(T, Out, mode, st, if (bitmap.bitIsSet(b, d.offset + i)) x else id);
                }
                for (vb, 0..) |*o, j| o.* = bitmap.byteAtBit(b, d.offset + j * 8);
                if (n % 8 != 0) vb[vb.len - 1] &= (@as(u8, 1) << @intCast(n % 8)) - 1;
            }
            if (bitmap.countUnsetBit(vb, n) == 0) {
                validity.deinit();
                return null;
            }
            return validity;
        }

        /// The value that leaves the running value unchanged (-0.0 for a float sum keeps a -0.0).
        fn identity(comptime T: type) T {
            const is_float = @typeInfo(T) == .float;
            return switch (op) {
                .sum, .sum_checked, .mean => if (is_float) -0.0 else 0,
                .prod, .prod_checked => 1,
                .min => if (is_float) std.math.inf(T) else std.math.maxInt(T),
                .max => if (is_float) -std.math.inf(T) else std.math.minInt(T),
            };
        }

        inline fn step(comptime T: type, comptime Out: type, comptime mode: Mode, st: *State(T), x: T) KernelError!Out {
            const is_float = @typeInfo(T) == .float;
            const acc = &st.acc;
            switch (op) {
                .mean => {
                    st.mean_sum += if (is_float) @as(f64, @floatCast(x)) else @as(f64, @floatFromInt(x));
                    st.mean_n += 1;
                    return st.mean_sum / st.mean_n;
                },
                .sum, .sum_checked => acc.* = if (is_float) acc.* + x else switch (mode) {
                    .wrap => acc.* +% x,
                    .saturate => acc.* +| x,
                    .check => blk: {
                        const r = @addWithOverflow(acc.*, x);
                        if (r[1] != 0) return error.Overflow;
                        break :blk r[0];
                    },
                },
                .prod, .prod_checked => acc.* = if (is_float) acc.* * x else switch (mode) {
                    .wrap => acc.* *% x,
                    .saturate => acc.* *| x,
                    .check => blk: {
                        const r = @mulWithOverflow(acc.*, x);
                        if (r[1] != 0) return error.Overflow;
                        break :blk r[0];
                    },
                },
                // The later of two equal values wins and NaN never does (pyarrow 26).
                .min => if (!(is_float and std.math.isNan(x)) and (x <= acc.* or (is_float and std.math.isNan(acc.*)))) {
                    acc.* = x;
                },
                .max => if (!(is_float and std.math.isNan(x)) and (x >= acc.* or (is_float and std.math.isNan(acc.*)))) {
                    acc.* = x;
                },
            }
            return if (op == .mean) unreachable else acc.*;
        }
    };
}

/// `CumulativeOptions.start` as `T` (null: none or a null scalar). A numeric scalar is cast
/// safely (`InvalidCast` when it does not fit); another type is `InvalidOptions`.
fn startValue(comptime T: type, ctx: *ExecContext, start: ?Scalar) KernelError!?T {
    const s = start orelse return null;
    if (!isNumeric(s.data_type)) return error.InvalidOptions;
    if (s.isNull()) return null;
    var cast = try ctx.invoke("cast", .scalar, &.{.{ .scalar = s }}, .{ .cast = .{ .to_type = dense.dataTypeOf(T) } });
    defer cast.release();
    if (cast != .scalar or cast.scalar.isNull()) return error.InvalidOptions;
    return dense.scalarAs(T, cast.scalar) catch error.InvalidOptions;
}

// ------------------------------------------------------------------------------ pairwise

fn Pairwise(comptime checked: bool) type {
    return struct {
        fn exec(ctx: *ExecContext, args: []const Datum, options: Options) KernelError!Datum {
            if (args.len != 1) return error.InvalidArity;
            const period: i64 = if (options == .pairwise) options.pairwise.period else 1;
            var owned: ?ArrayRef = null;
            defer if (owned) |*a| a.release();
            const parts = try partsOf(ctx.allocator, &args[0], &owned);
            var whole = if (parts.len == 1) parts[0].retain() else if (parts.len == 0) try dense.nullArray(ctx.allocator, args[0].dataType(), 0) else concat.concatArrayRefs(ctx.allocator, args[0].dataType(), parts) catch |err| return switch (err) {
                error.OutOfMemory => error.OutOfMemory,
                else => error.InvalidInput,
            };
            defer whole.release();
            const n = whole.data().length;
            const p: usize = @intCast(@min(@abs(period), n));
            // Output rows that have a partner, and their partners (row i - period).
            var rows = whole.slice(if (period >= 0) p else 0, n - p) catch return error.OutOfMemory;
            defer rows.release();
            var partners = whole.slice(if (period >= 0) 0 else p, n - p) catch return error.OutOfMemory;
            defer partners.release();
            var diff = try ctx.invoke(if (checked) "subtract_checked" else "subtract", .scalar, &.{ .{ .array = rows }, .{ .array = partners } }, Options.noneValue());
            defer diff.release();
            if (diff != .array) return error.InvalidInput;
            const dt = diff.array.data().data_type;
            var nulls = try dense.nullArray(ctx.allocator, dt, p);
            defer nulls.release();
            const pieces: [2]ArrayRef = if (period >= 0) .{ nulls, diff.array } else .{ diff.array, nulls };
            var one = [1]ArrayRef{concat.concatArrayRefs(ctx.allocator, dt, &pieces) catch |err| return switch (err) {
                error.OutOfMemory => error.OutOfMemory,
                else => error.InvalidInput,
            }};
            return shaped(ctx.allocator, args[0], dt, &one);
        }
    };
}

// ------------------------------------------------------------------------------ winsorize

fn winsorize(ctx: *ExecContext, args: []const Datum, options: Options) KernelError!Datum {
    if (args.len != 1) return error.InvalidArity;
    const o = options.winsorize;
    if (!(o.lower_limit >= 0 and o.lower_limit <= o.upper_limit and o.upper_limit <= 1)) return error.InvalidOptions;
    var owned: ?ArrayRef = null;
    defer if (owned) |*a| a.release();
    const parts = try partsOf(ctx.allocator, &args[0], &owned);
    const input: Datum = switch (args[0]) {
        .scalar => .{ .array = owned.? },
        else => args[0],
    };
    const limits = [2]f64{ o.lower_limit, o.upper_limit };
    var q = try ctx.invokeVector("quantile", &.{input}, .{ .quantile = .{ .q = &limits, .interpolation = .nearest } });
    defer q.release();
    if (q != .array) return error.InvalidInput;
    const qd = q.array.data();
    const dt = args[0].dataType();
    const outs = ctx.allocator.alloc(ArrayRef, parts.len) catch return error.OutOfMemory;
    var made: usize = 0;
    defer {
        for (outs[0..made]) |*a| a.release();
        ctx.allocator.free(outs);
    }
    inline for (dense.numeric_types ++ [_]type{ i32, i64, i128, i256 }, 0..) |T, k| {
        const tag: std.meta.Tag(DataType) = comptime if (k < dense.numeric_types.len) std.meta.activeTag(dense.dataTypeOf(T)) else switch (T) {
            i32 => .decimal32,
            i64 => .decimal64,
            i128 => .decimal128,
            else => .decimal256,
        };
        if (std.meta.activeTag(dt) == tag) {
            // No value to take a quantile of (all null or NaN): the input as is.
            const bounds: ?[2]T = if (qd.length == 2 and !qd.isNull(0) and !qd.isNull(1)) blk: {
                const qs = (PrimitiveArray(T){ .data = qd }).values() catch return error.InvalidInput;
                break :blk .{ qs[0], qs[1] };
            } else null;
            for (parts) |p| {
                outs[made] = try clamp(T, ctx.allocator, p, bounds);
                made += 1;
            }
            const n = made;
            made = 0;
            return shaped(ctx.allocator, args[0], dt, outs[0..n]);
        }
    }
    return error.NoMatchingKernel;
}

/// `p` with every value outside `[lo, hi]` replaced by the bound (NaN compares false and stays).
fn clamp(comptime T: type, allocator: std.mem.Allocator, p: ArrayRef, bounds: ?[2]T) KernelError!ArrayRef {
    const b = bounds orelse return p.retain();
    const d = p.data();
    const xs = (PrimitiveArray(T){ .data = d }).values() catch return error.InvalidInput;
    var out = try dense.DenseBinaryOutput(T).initValidity(allocator, d.length, d, null, d.data_type);
    errdefer out.deinit();
    for (out.values(), xs) |*y, x| y.* = if (x < b[0]) b[0] else if (x > b[1]) b[1] else x;
    return out.finish() catch error.OutOfMemory;
}

// ------------------------------------------------------------------------------ random

fn random(ctx: *ExecContext, args: []const Datum, options: Options) KernelError!Datum {
    if (args.len != 0) return error.InvalidArity;
    const o = options.random;
    if (o.initializer == .system) return error.NotImplemented;
    var prng = std.Random.DefaultPrng.init(o.seed);
    const rng = prng.random();
    var buf = OwnedBuffer.initUninit(ctx.allocator, o.length * @sizeOf(f64)) catch return error.OutOfMemory;
    for (buf.typedSlice(f64)[0..o.length]) |*y| y.* = rng.float(f64);
    return .{ .array = try dense.finishFixed(ctx.allocator, .{ .double = {} }, o.length, &buf, null) };
}
