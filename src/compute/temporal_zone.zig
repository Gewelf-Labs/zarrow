//! Timezone handling (Arrow C++ "Timezone handling"): assume_timezone, local_timestamp and
//! is_dst, and the localization every other temporal function applies to a timestamp in a
//! timezone other than UTC (Arrow reads such a value as its local time).
//!
//! - assume_timezone (`AssumeTimezoneOptions`: `timezone`, `ambiguous`, `nonexistent`): a naive
//!   timestamp read as local time of `timezone` becomes `timestamp(unit, timezone)`. An ambiguous
//!   local time (a fall-back hour) takes the earlier or later instant, or is `InvalidInput`
//!   (`raise`, the default); a nonexistent one (a spring-forward gap) takes the instant the gap
//!   ends at (`latest`) or one unit before it (`earliest`), or is `InvalidInput`. A zoned input
//!   is `InvalidInput` (Arrow: "Timestamps already have a timezone").
//! - local_timestamp: a timestamp as naive local time (a naive input comes back as is).
//! - is_dst: whether daylight saving time is in effect (bool); a naive input is `InvalidInput`.
//!
//! Localization (`localize`) shifts each value by the offset of its instant (wrapping at the i64
//! edges like Arrow's arithmetic); `unlocalize` maps a local result back (rounding), an ambiguous
//! or nonexistent local time being `InvalidInput`, as Arrow ("Local time is ambiguous").
const std = @import("std");
const bitmap = @import("../bitmap.zig");
const core = @import("core.zig");
const dense = @import("dense.zig");
const tz = @import("tz.zig");

const ArrayData = @import("../array/array_data.zig").ArrayData;
const PrimitiveArray = @import("../array/primitive_array.zig").PrimitiveArray;
const ArrayRef = core.ArrayRef;
const DataType = core.DataType;
const Datum = core.Datum;
const ExecChunkValue = core.ExecChunkValue;
const ExecContext = core.ExecContext;
const FunctionRegistry = core.FunctionRegistry;
const KernelError = core.KernelError;
const Options = core.Options;
const AssumeTimezoneOptions = core.AssumeTimezoneOptions;

pub fn register(registry: *FunctionRegistry) KernelError!void {
    try registry.registerScalarKernel("assume_timezone", .{
        .signature = .{ .arity = 1, .type_check = dense.argsOfType(.timestamp, 1), .options_check = assumeOptions, .result_type_fn = assumeType },
        .exec = assumeExec,
    });
    try registry.registerScalarKernel("local_timestamp", .{
        .signature = .{ .arity = 1, .type_check = dense.argsOfType(.timestamp, 1), .options_check = dense.noOptions, .result_type_fn = naiveType },
        .exec = localExec,
    });
    try registry.registerScalarKernel("is_dst", .{
        .signature = .{ .arity = 1, .type_check = dense.argsOfType(.timestamp, 1), .options_check = dense.noOptions, .result_type_fn = dense.resultTypeOf(.{ .bool = {} }) },
        .exec = isDstExec,
    });
}

fn assumeOptions(options: Options) bool {
    return options == .assume_timezone;
}

fn assumeType(args: []const Datum, options: Options) KernelError!DataType {
    if (args.len != 1) return error.InvalidArity;
    return .{ .timestamp = .{ .unit = args[0].dataType().timestamp.unit, .timezone = options.assume_timezone.timezone } };
}

fn naiveType(args: []const Datum, _: Options) KernelError!DataType {
    if (args.len != 1) return error.InvalidArity;
    return .{ .timestamp = .{ .unit = args[0].dataType().timestamp.unit, .timezone = null } };
}

fn perSecond(dt: DataType) i64 {
    return @import("temporal_text.zig").perSecond(dt.timestamp.unit);
}

/// The timezone a timestamp type's values read in, or null when they read as UTC (naive, `UTC`,
/// `Etc/UTC`, `+00:00`, or not a timestamp).
pub fn zoneName(dt: DataType) ?[]const u8 {
    return switch (dt) {
        .timestamp => |t| if (t.timezone) |z| (if (tz.isUtc(z)) null else z) else null,
        else => null,
    };
}

/// Resolve `name` with the context's I/O (system zoneinfo) or the bundled table.
pub fn resolve(ctx: *const ExecContext, name: []const u8) KernelError!tz.Zone {
    return tz.resolve(ctx.allocator, ctx.config.io, name);
}

/// What the chunk functions below get through a custom options payload.
const State = struct {
    zone: *const tz.Zone,
    per: i64,
    out_type: DataType,
    assume: AssumeTimezoneOptions = .{ .timezone = "" },
};

fn run(ctx: *ExecContext, d: Datum, state: *const State, comptime chunk: dense.ChunkOptionsFn) KernelError!Datum {
    return dense.execElementwiseWith(ctx, &.{d}, .{ .custom = .{ .tag = "temporal_zone", .payload = state } }, state.out_type, true, chunk);
}

fn stateOf(options: Options) *const State {
    return @ptrCast(@alignCast(options.custom.payload.?));
}

/// `d` (a zoned timestamp datum) as naive local time; the caller releases the result.
pub fn localize(ctx: *ExecContext, d: Datum, name: []const u8) KernelError!Datum {
    var zone = try resolve(ctx, name);
    defer zone.deinit(ctx.allocator);
    const dt = d.dataType();
    const state: State = .{ .zone = &zone, .per = perSecond(dt), .out_type = .{ .timestamp = .{ .unit = dt.timestamp.unit, .timezone = null } } };
    return run(ctx, d, &state, localizeChunk);
}

fn localizeChunk(ctx: *ExecContext, values: []const ExecChunkValue, len: usize, options: Options) KernelError!ArrayRef {
    const state = stateOf(options);
    var arr = try dense.chunkArray(ctx.allocator, values[0]);
    defer arr.release();
    const data = arr.data();
    const vals = (PrimitiveArray(i64){ .data = data }).values() catch return error.InvalidInput;
    var out = try dense.DenseBinaryOutput(i64).initValidity(ctx.allocator, len, data, null, state.out_type);
    errdefer out.deinit();
    var cursor = tz.Cursor.init(state.zone);
    for (vals, out.values()) |v, *o| o.* = v +% @as(i64, cursor.info(@divFloor(v, state.per)).offset) *% state.per;
    return out.finish() catch error.OutOfMemory;
}

/// Naive local values `d` mapped back to UTC as `out_type` (a timestamp in zone `name`); an
/// ambiguous or nonexistent local time in a valid row is `InvalidInput`.
pub fn unlocalize(ctx: *ExecContext, d: Datum, name: []const u8, out_type: DataType) KernelError!Datum {
    var zone = try resolve(ctx, name);
    defer zone.deinit(ctx.allocator);
    const state: State = .{ .zone = &zone, .per = perSecond(out_type), .out_type = out_type };
    return run(ctx, d, &state, toUtcChunk);
}

fn toUtcChunk(ctx: *ExecContext, values: []const ExecChunkValue, len: usize, options: Options) KernelError!ArrayRef {
    const state = stateOf(options);
    var arr = try dense.chunkArray(ctx.allocator, values[0]);
    defer arr.release();
    const data = arr.data();
    const vals = (PrimitiveArray(i64){ .data = data }).values() catch return error.InvalidInput;
    var out = try dense.DenseBinaryOutput(i64).initValidity(ctx.allocator, len, data, null, state.out_type);
    errdefer out.deinit();
    const bits = dense.validityBits(data);
    var cursor = tz.Cursor.init(state.zone);
    const a = state.assume;
    for (vals, out.values(), 0..) |v, *o, i| {
        o.* = 0;
        if (bits) |b| if (!bitmap.bitIsSet(b, data.offset + i)) continue;
        o.* = switch (cursor.local(@divFloor(v, state.per))) {
            .unique => |off| v -% @as(i64, off) *% state.per,
            .ambiguous => |offs| switch (a.ambiguous) {
                .raise => return error.InvalidInput,
                .earliest => v -% @as(i64, offs[0]) *% state.per,
                .latest => v -% @as(i64, offs[1]) *% state.per,
            },
            .gap => |end| switch (a.nonexistent) {
                .raise => return error.InvalidInput,
                .earliest => end *% state.per -% 1,
                .latest => end *% state.per,
            },
        };
    }
    return out.finish() catch error.OutOfMemory;
}

fn assumeExec(ctx: *ExecContext, args: []const Datum, options: Options) KernelError!Datum {
    if (args.len != 1) return error.InvalidArity;
    if (args[0].dataType().timestamp.timezone != null) return error.InvalidInput;
    const o = options.assume_timezone;
    var zone = try resolve(ctx, o.timezone);
    defer zone.deinit(ctx.allocator);
    const state: State = .{ .zone = &zone, .per = perSecond(args[0].dataType()), .out_type = try assumeType(args, options), .assume = o };
    return run(ctx, args[0], &state, toUtcChunk);
}

fn localExec(ctx: *ExecContext, args: []const Datum, _: Options) KernelError!Datum {
    if (args.len != 1) return error.InvalidArity;
    const dt = args[0].dataType();
    const name = dt.timestamp.timezone orelse return args[0].retain();
    var zone = try resolve(ctx, name);
    defer zone.deinit(ctx.allocator);
    const state: State = .{ .zone = &zone, .per = perSecond(dt), .out_type = .{ .timestamp = .{ .unit = dt.timestamp.unit, .timezone = null } } };
    return run(ctx, args[0], &state, localizeChunk);
}

fn isDstExec(ctx: *ExecContext, args: []const Datum, _: Options) KernelError!Datum {
    if (args.len != 1) return error.InvalidArity;
    const dt = args[0].dataType();
    const name = dt.timestamp.timezone orelse return error.InvalidInput;
    var zone = try resolve(ctx, name);
    defer zone.deinit(ctx.allocator);
    const state: State = .{ .zone = &zone, .per = perSecond(dt), .out_type = .{ .bool = {} } };
    return run(ctx, args[0], &state, isDstChunk);
}

fn isDstChunk(ctx: *ExecContext, values: []const ExecChunkValue, len: usize, options: Options) KernelError!ArrayRef {
    const state = stateOf(options);
    var arr = try dense.chunkArray(ctx.allocator, values[0]);
    defer arr.release();
    const data = arr.data();
    const vals = (PrimitiveArray(i64){ .data = data }).values() catch return error.InvalidInput;
    var out = try dense.DenseBinaryOutput(bool).initValidity(ctx.allocator, len, data, null, state.out_type);
    errdefer out.deinit();
    const bits = out.values();
    @memset(bits, 0);
    var cursor = tz.Cursor.init(state.zone);
    for (vals, 0..) |v, i| {
        if (cursor.info(@divFloor(v, state.per)).is_dst) bits[i / 8] |= @as(u8, 1) << @intCast(i % 8);
    }
    return out.finish() catch error.OutOfMemory;
}
