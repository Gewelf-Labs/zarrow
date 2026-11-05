pub const std = @import("std");
pub const datatype = @import("../datatype.zig");
pub const array_mod = @import("../array/array.zig");
pub const core = @import("core.zig");

pub const DataType = core.DataType;
pub const ArrayRef = core.ArrayRef;
pub const ChunkedArray = core.ChunkedArray;
pub const Scalar = core.Scalar;
pub const Datum = core.Datum;
pub const FunctionRegistry = core.FunctionRegistry;
pub const ExecContext = core.ExecContext;
pub const KernelSignature = core.KernelSignature;
pub const KernelError = core.KernelError;
pub const Options = core.Options;
pub const OptionsTag = core.OptionsTag;
pub const SortOptions = core.SortOptions;
pub const SortOrder = core.SortOrder;
pub const SortNullPlacement = core.SortNullPlacement;
pub const FilterOptions = core.FilterOptions;
pub const UnaryExecChunkIterator = core.UnaryExecChunkIterator;
pub const BinaryExecChunkIterator = core.BinaryExecChunkIterator;
pub const NaryExecChunkIterator = core.NaryExecChunkIterator;

pub const intCastOrInvalidCast = core.intCastOrInvalidCast;
pub const arithmeticDivI64 = core.arithmeticDivI64;
pub const hasArity = core.hasArity;
pub const unaryArray = core.unaryArray;
pub const unaryChunked = core.unaryChunked;
pub const unaryScalar = core.unaryScalar;
pub const sameDataTypes = core.sameDataTypes;
pub const unaryNumeric = core.unaryNumeric;
pub const binarySameNumeric = core.binarySameNumeric;
pub const allNumeric = core.allNumeric;
pub const unaryNullPropagates = core.unaryNullPropagates;
pub const binaryNullPropagates = core.binaryNullPropagates;
pub const naryNullPropagates = core.naryNullPropagates;
pub const inferBinaryExecLen = core.inferBinaryExecLen;
pub const inferNaryExecLen = core.inferNaryExecLen;
pub const datumListValueAt = core.datumListValueAt;
pub const datumStructField = core.datumStructField;
pub const datumBuildEmptyLike = core.datumBuildEmptyLike;
pub const datumBuildEmptyLikeWithAllocator = core.datumBuildEmptyLikeWithAllocator;
pub const datumBuildNullLike = core.datumBuildNullLike;
pub const datumBuildNullLikeWithAllocator = core.datumBuildNullLikeWithAllocator;
pub const datumSliceEmpty = core.datumSliceEmpty;
pub const datumSelect = core.datumSelect;
pub const datumSelectNullable = core.datumSelectNullable;
pub const datumFilter = core.datumFilter;
pub const chunkedResolveLogicalIndices = core.chunkedResolveLogicalIndices;
pub const datumTake = core.datumTake;
pub const datumFilterSelectionIndices = core.datumFilterSelectionIndices;
pub const datumFilterChunkAware = core.datumFilterChunkAware;

pub fn isInt32Datum(args: []const Datum) bool {
    return args.len == 1 and args[0].dataType() == .int32;
}

pub fn isTwoInt32(args: []const Datum) bool {
    return args.len == 2 and args[0].dataType() == .int32 and args[1].dataType() == .int32;
}

pub fn isInt64Scalar(args: []const Datum) bool {
    return args.len == 1 and args[0].isScalar() and args[0].scalar.data_type == .int64;
}

pub fn isTwoInt64Scalars(args: []const Datum) bool {
    return args.len == 2 and args[0].isScalar() and args[1].isScalar() and args[0].scalar.data_type == .int64 and args[1].scalar.data_type == .int64;
}

pub fn allInt32Datums(args: []const Datum) bool {
    for (args) |arg| {
        if (!arg.dataType().eql(.{ .int32 = {} })) return false;
    }
    return true;
}

pub fn passthroughInt32Kernel(ctx: *ExecContext, args: []const Datum, options: Options) KernelError!Datum {
    _ = ctx;
    _ = options;
    return args[0].retain();
}

pub fn exactArityMarkerKernel(ctx: *ExecContext, args: []const Datum, options: Options) KernelError!Datum {
    _ = ctx;
    _ = args;
    _ = options;
    return Datum.fromScalar(.{
        .data_type = .{ .int32 = {} },
        .value = .{ .i32 = 303 },
    });
}

pub fn rangeArityMarkerKernel(ctx: *ExecContext, args: []const Datum, options: Options) KernelError!Datum {
    _ = ctx;
    _ = args;
    _ = options;
    return Datum.fromScalar(.{
        .data_type = .{ .int32 = {} },
        .value = .{ .i32 = 202 },
    });
}

pub fn atLeastArityMarkerKernel(ctx: *ExecContext, args: []const Datum, options: Options) KernelError!Datum {
    _ = ctx;
    _ = args;
    _ = options;
    return Datum.fromScalar(.{
        .data_type = .{ .int32 = {} },
        .value = .{ .i32 = 101 },
    });
}

pub fn countLenAggregateKernel(ctx: *ExecContext, args: []const Datum, options: Options) KernelError!Datum {
    _ = ctx;
    _ = options;
    if (args.len != 1) return error.InvalidArity;

    const count: usize = switch (args[0]) {
        .array => |arr| arr.data().length,
        .chunked => |chunks| chunks.len(),
        .scalar => return error.InvalidInput,
    };

    return Datum.fromScalar(Scalar.init(.{ .int64 = {} }, .{ .i64 = @intCast(count) }));
}

pub fn onlyCastOptions(options: Options) bool {
    return switch (options) {
        .cast => true,
        else => false,
    };
}

pub fn onlyArithmeticOptions(options: Options) bool {
    return switch (options) {
        .arithmetic => true,
        else => false,
    };
}

pub fn onlySortOptions(options: Options) bool {
    return switch (options) {
        .sort => true,
        else => false,
    };
}

pub fn firstArgResultType(args: []const Datum, options: Options) KernelError!DataType {
    _ = options;
    if (args.len == 0) return error.InvalidInput;
    return args[0].dataType();
}

pub fn castResultType(args: []const Datum, options: Options) KernelError!DataType {
    if (args.len != 1) return error.InvalidArity;
    return switch (options) {
        .cast => |cast_opts| cast_opts.to_type orelse args[0].dataType(),
        else => error.InvalidOptions,
    };
}

pub fn castI64ToI32ResultType(args: []const Datum, options: Options) KernelError!DataType {
    if (args.len != 1) return error.InvalidArity;
    if (!args[0].dataType().eql(.{ .int64 = {} })) return error.InvalidCast;
    return switch (options) {
        .cast => |cast_opts| blk: {
            if (cast_opts.to_type) |to_type| {
                if (!to_type.eql(.{ .int32 = {} })) return error.InvalidCast;
            }
            break :blk DataType{ .int32 = {} };
        },
        else => error.InvalidOptions,
    };
}

pub fn castI64ToI32Kernel(ctx: *ExecContext, args: []const Datum, options: Options) KernelError!Datum {
    _ = ctx;
    if (args.len != 1) return error.InvalidArity;
    if (!isInt64Scalar(args)) return error.InvalidInput;

    const cast_opts = switch (options) {
        .cast => |o| o,
        else => return error.InvalidOptions,
    };
    if (cast_opts.to_type) |to_type| {
        if (!to_type.eql(.{ .int32 = {} })) return error.InvalidCast;
    }

    const out = try intCastOrInvalidCast(i32, args[0].scalar.value.i64);
    return Datum.fromScalar(.{
        .data_type = .{ .int32 = {} },
        .value = .{ .i32 = out },
    });
}

pub fn divI64ScalarResultType(args: []const Datum, options: Options) KernelError!DataType {
    if (args.len != 2) return error.InvalidArity;
    _ = switch (options) {
        .arithmetic => {},
        else => return error.InvalidOptions,
    };
    if (!isTwoInt64Scalars(args)) return error.InvalidInput;
    return .{ .int64 = {} };
}

pub fn divI64ScalarKernel(ctx: *ExecContext, args: []const Datum, options: Options) KernelError!Datum {
    _ = ctx;
    if (args.len != 2) return error.InvalidArity;
    if (!isTwoInt64Scalars(args)) return error.InvalidInput;
    const arithmetic_opts = switch (options) {
        .arithmetic => |o| o,
        else => return error.InvalidOptions,
    };
    const lhs = args[0].scalar.value.i64;
    const rhs = args[1].scalar.value.i64;
    const out = try arithmeticDivI64(lhs, rhs, arithmetic_opts);
    return Datum.fromScalar(.{
        .data_type = .{ .int64 = {} },
        .value = .{ .i64 = out },
    });
}

pub fn countAggregateResultType(args: []const Datum, options: Options) KernelError!DataType {
    _ = args;
    _ = options;
    return .{ .int64 = {} };
}

pub const CountAggState = struct {
    count: usize,
};

pub fn countLifecycleInit(ctx: *ExecContext, options: Options) KernelError!*anyopaque {
    _ = options;
    const state = ctx.allocator.create(CountAggState) catch return error.OutOfMemory;
    state.* = .{ .count = 0 };
    return state;
}

pub fn countLifecycleUpdate(ctx: *ExecContext, state_ptr: *anyopaque, args: []const Datum, options: Options) KernelError!void {
    _ = ctx;
    _ = options;
    if (!unaryArray(args)) return error.InvalidInput;
    const state: *CountAggState = @ptrCast(@alignCast(state_ptr));
    const count = args[0].array.data().length;
    state.count = std.math.add(usize, state.count, count) catch return error.Overflow;
}

pub fn countLifecycleMerge(ctx: *ExecContext, state_ptr: *anyopaque, other_ptr: *anyopaque, options: Options) KernelError!void {
    _ = ctx;
    _ = options;
    const state: *CountAggState = @ptrCast(@alignCast(state_ptr));
    const other: *CountAggState = @ptrCast(@alignCast(other_ptr));
    state.count = std.math.add(usize, state.count, other.count) catch return error.Overflow;
}

pub fn countLifecycleFinalize(ctx: *ExecContext, state_ptr: *anyopaque, options: Options) KernelError!Datum {
    _ = ctx;
    _ = options;
    const state: *CountAggState = @ptrCast(@alignCast(state_ptr));
    return Datum.fromScalar(.{
        .data_type = .{ .int64 = {} },
        .value = .{ .i64 = @intCast(state.count) },
    });
}

pub fn countLifecycleDeinit(ctx: *ExecContext, state_ptr: *anyopaque) void {
    const state: *CountAggState = @ptrCast(@alignCast(state_ptr));
    ctx.allocator.destroy(state);
}

pub fn makeInt32Array(allocator: std.mem.Allocator, values: []const ?i32) !ArrayRef {
    const int32_builder = @import("../array/array.zig").Int32Builder;
    var builder = try int32_builder.init(allocator, values.len);
    defer builder.deinit();

    for (values) |v| {
        if (v) |value| {
            try builder.append(value);
        } else {
            try builder.appendNull();
        }
    }
    return builder.finish();
}

pub fn makeBoolArray(allocator: std.mem.Allocator, values: []const ?bool) !ArrayRef {
    var builder = try array_mod.BooleanBuilder.init(allocator, values.len);
    defer builder.deinit();

    for (values) |v| {
        if (v) |value| {
            try builder.append(value);
        } else {
            try builder.appendNull();
        }
    }
    return builder.finish();
}

pub fn collectInt32ValuesFromDatum(allocator: std.mem.Allocator, datum: Datum) ![]?i32 {
    return switch (datum) {
        .array => |arr| blk: {
            if (arr.data().data_type != .int32) break :blk error.InvalidInput;
            var out = try allocator.alloc(?i32, arr.data().length);
            const view = array_mod.Int32Array{ .data = arr.data() };
            var i: usize = 0;
            while (i < out.len) : (i += 1) {
                out[i] = if (view.isNull(i)) null else try view.value(i);
            }
            break :blk out;
        },
        .chunked => |chunks| blk: {
            if (!chunks.dataType().eql(.{ .int32 = {} })) break :blk error.InvalidInput;
            var out = try allocator.alloc(?i32, chunks.len());
            var out_index: usize = 0;
            var chunk_index: usize = 0;
            while (chunk_index < chunks.numChunks()) : (chunk_index += 1) {
                const chunk = chunks.chunk(chunk_index);
                const view = array_mod.Int32Array{ .data = chunk.data() };
                var i: usize = 0;
                while (i < view.len()) : (i += 1) {
                    out[out_index] = if (view.isNull(i)) null else try view.value(i);
                    out_index += 1;
                }
            }
            break :blk out;
        },
        .scalar => |s| blk: {
            if (s.data_type != .int32) break :blk error.InvalidInput;
            var out = try allocator.alloc(?i32, 1);
            out[0] = if (s.isNull()) null else s.value.i32;
            break :blk out;
        },
    };
}

// ------------------------------------------------------------------ kernel family test helpers

pub const kernels = @import("kernels.zig");
pub const dense = @import("dense.zig");
pub const bitmap = @import("../bitmap.zig");
pub const SharedBuffer = @import("../buffer.zig").SharedBuffer;
pub const none = Options.noneValue();

/// Registry holding every kernel family (`registerAll`).
pub fn kernelRegistry(allocator: std.mem.Allocator) !FunctionRegistry {
    var registry = FunctionRegistry.init(allocator);
    errdefer registry.deinit();
    try kernels.registerAll(&registry);
    return registry;
}

/// Array of the numeric or bool type `T` from optionals.
pub fn makeArray(comptime T: type, allocator: std.mem.Allocator, values: []const ?T) !ArrayRef {
    if (T == bool) return makeBoolArray(allocator, values);
    var b = try array_mod.PrimitiveBuilder(T, dense.dataTypeOf(T)).init(allocator, values.len);
    defer b.deinit();
    for (values) |v| if (v) |x| try b.append(x) else try b.appendNull();
    return b.finish();
}

/// Borrowed primitive array over `values` with an optional validity bitmap (null rows keep the
/// given values, such as a zero divisor or minInt, where a builder would store zero).
pub fn borrowedArray(comptime T: type, allocator: std.mem.Allocator, values: []const T, validity: ?[]const u8) !ArrayRef {
    var nulls: usize = 0;
    if (validity) |v| for (0..values.len) |i| {
        if (!bitmap.bitIsSet(v, i)) nulls += 1;
    };
    const bufs = [_]SharedBuffer{ if (validity) |v| .fromSlice(v) else .empty, .fromSlice(std.mem.sliceAsBytes(values)) };
    return ArrayRef.fromBorrowed(allocator, .{ .data_type = dense.dataTypeOf(T), .length = values.len, .null_count = nulls, .buffers = &bufs });
}

/// Random value of `T`: full range, with edge values (0, 1, -1, min, max, NaN, inf) mixed in.
pub fn randomValue(comptime T: type, rng: std.Random) T {
    if (T == bool) return rng.boolean();
    const edge = rng.uintLessThan(u8, 16) == 0;
    if (@typeInfo(T) == .float) {
        if (edge) return switch (rng.uintLessThan(u8, 5)) {
            0 => 0,
            1 => std.math.nan(T),
            2 => std.math.inf(T),
            3 => -std.math.inf(T),
            else => -0.0,
        };
        return @floatCast((rng.float(f64) - 0.5) * 2000.0);
    }
    if (edge) return switch (rng.uintLessThan(u8, 5)) {
        0 => 0,
        1 => 1,
        2 => std.math.maxInt(T),
        3 => std.math.minInt(T),
        else => if (@typeInfo(T).int.signedness == .signed) -1 else 2,
    };
    return rng.int(T);
}

/// Options of `randomArray`.
pub const RandomOptions = struct {
    /// Keep integers in a range where add / subtract / multiply cannot overflow.
    small: bool = false,
    /// Avoid zero in valid rows.
    nonzero: bool = false,
};

/// Random array of `len + off` rows (about `null_pct`% null) sliced to the last `len` rows, so
/// the result has offset `off`.
pub fn randomArray(comptime T: type, allocator: std.mem.Allocator, rng: std.Random, len: usize, off: usize, null_pct: u8, opts: RandomOptions) !ArrayRef {
    var values: std.ArrayList(?T) = .empty;
    defer values.deinit(allocator);
    for (0..len + off) |_| {
        if (rng.uintLessThan(u8, 100) < null_pct) {
            try values.append(allocator, null);
            continue;
        }
        var v = randomValue(T, rng);
        if (T != bool and @typeInfo(T) == .int and opts.small) {
            v = v >> (@typeInfo(T).int.bits / 2 + 1);
        }
        if (T != bool and opts.nonzero and v == 0) v = 1;
        try values.append(allocator, v);
    }
    var full = try makeArray(T, allocator, values.items);
    defer full.release();
    return full.slice(off, len);
}

/// Random utf8 values of type `tag` (string / large_string / string_view), some longer than 12
/// bytes so string_view stores them out of line; sliced like `randomArray`.
pub fn randomStrings(comptime tag: std.meta.Tag(DataType), allocator: std.mem.Allocator, rng: std.Random, len: usize, off: usize, null_pct: u8) !ArrayRef {
    const Builder = switch (tag) {
        .string => array_mod.StringBuilder,
        .large_string => array_mod.LargeStringBuilder,
        .string_view => array_mod.StringViewBuilder,
        else => @compileError("not a string type"),
    };
    var b = try Builder.init(allocator, len + off, (len + off) * 20);
    defer b.deinit();
    const alphabet = "abz";
    var buf: [24]u8 = undefined;
    for (0..len + off) |_| {
        if (rng.uintLessThan(u8, 100) < null_pct) {
            try b.appendNull();
            continue;
        }
        const n = if (rng.boolean()) rng.uintAtMost(usize, 4) else rng.uintAtMost(usize, 20);
        for (buf[0..n]) |*c| c.* = alphabet[rng.uintLessThan(usize, alphabet.len)];
        try b.append(buf[0..n]);
    }
    var full = try b.finish();
    defer full.release();
    return full.slice(off, len);
}

/// Bytes of string row `i` (string / large_string / string_view), null for a null row.
pub fn stringAt(data: *const array_mod.ArrayData, i: usize) !?[]const u8 {
    if (data.isNull(i)) return null;
    return switch (data.data_type) {
        .string => try (array_mod.StringArray{ .data = data }).value(i),
        .large_string => try (array_mod.LargeStringArray{ .data = data }).value(i),
        .string_view => try (array_mod.StringViewArray{ .data = data }).value(i),
        else => error.InvalidInput,
    };
}

fn collectArray(comptime T: type, data: *const array_mod.ArrayData, out: []?T) !void {
    for (out, 0..) |*o, i| {
        if (data.isNull(i)) {
            o.* = null;
        } else if (T == bool) {
            o.* = try (array_mod.BooleanArray{ .data = data }).value(i);
        } else {
            o.* = (try (array_mod.PrimitiveArray(T){ .data = data }).values())[i];
        }
    }
}

/// Rows of an array, chunked or scalar datum of `T` (numeric or bool) as optionals; caller frees.
pub fn collect(comptime T: type, allocator: std.mem.Allocator, datum: Datum) ![]?T {
    switch (datum) {
        .array => |arr| {
            const out = try allocator.alloc(?T, arr.data().length);
            errdefer allocator.free(out);
            try collectArray(T, arr.data(), out);
            return out;
        },
        .chunked => |chunked| {
            const out = try allocator.alloc(?T, chunked.len());
            errdefer allocator.free(out);
            var pos: usize = 0;
            for (chunked.chunks()) |chunk| {
                try collectArray(T, chunk.data(), out[pos..][0..chunk.data().length]);
                pos += chunk.data().length;
            }
            return out;
        },
        .scalar => |s| {
            const out = try allocator.alloc(?T, 1);
            errdefer allocator.free(out);
            // i128 rows are month_day_nano intervals (no plain i128 scalar payload).
            out[0] = if (s.isNull()) null else if (comptime T == i128) s.value.interval_month_day_nano else try dense.scalarAs(T, s);
            return out;
        },
    }
}

/// Element-wise equality of optional rows; NaN equals NaN.
pub fn expectRows(comptime T: type, expected: []const ?T, actual: []const ?T) !void {
    try std.testing.expectEqual(expected.len, actual.len);
    for (expected, actual, 0..) |e, a, i| {
        const same = if (e == null or a == null)
            e == null and a == null
        else if (T != bool and @typeInfo(T) == .float)
            (std.math.isNan(e.?) and std.math.isNan(a.?)) or e.? == a.?
        else
            e.? == a.?;
        if (!same) {
            std.debug.print("row {d}: expected {any}, got {any}\n", .{ i, e, a });
            return error.TestExpectedEqual;
        }
    }
}

/// Invoke `name` and compare the result rows with `expected`; the output must validate.
pub fn expectInvoke(comptime T: type, ctx: *ExecContext, kind: core.FunctionKind, name: []const u8, args: []const Datum, expected: []const ?T) !void {
    const allocator = std.testing.allocator;
    var out = try ctx.invoke(name, kind, args, none);
    defer out.release();
    switch (out) {
        .array => |arr| try arr.data().validateFull(),
        .chunked => |chunked| for (chunked.chunks()) |c| try c.data().validateFull(),
        .scalar => {},
    }
    const rows = try collect(T, allocator, out);
    defer allocator.free(rows);
    try expectRows(T, expected, rows);
}

/// Chunked array of `arr` cut before each index in `cuts` (ascending, inside `(0, len)`).
pub fn chunkedAt(allocator: std.mem.Allocator, arr: ArrayRef, cuts: []const usize) !ChunkedArray {
    var parts: std.ArrayList(ArrayRef) = .empty;
    defer {
        for (parts.items) |*p| p.release();
        parts.deinit(allocator);
    }
    var start: usize = 0;
    for (cuts) |c| {
        try parts.append(allocator, try arr.slice(start, c - start));
        start = c;
    }
    try parts.append(allocator, try arr.slice(start, arr.data().length - start));
    return ChunkedArray.init(allocator, arr.data().data_type, parts.items);
}

/// Scalar of the Zig primitive `v` (or a null one).
pub fn scalarOf(comptime T: type, v: ?T) Scalar {
    return Scalar.init(dense.dataTypeOf(T), if (v) |x| dense.scalarValueOf(T, x) else .null);
}

/// Primitive array of logical type `dt` (date32, timestamp, ...) stored as `T`, from optionals.
pub fn makeTyped(comptime T: type, comptime dt: DataType, allocator: std.mem.Allocator, values: []const ?T) !ArrayRef {
    var b = try array_mod.PrimitiveBuilder(T, dt).init(allocator, values.len);
    defer b.deinit();
    for (values) |v| if (v) |x| try b.append(x) else try b.appendNull();
    return b.finish();
}

/// String or binary array of layout `tag` (string, large_string, string_view, binary, large_binary,
/// binary_view) from optional byte strings.
pub fn makeStrings(comptime tag: std.meta.Tag(DataType), allocator: std.mem.Allocator, values: []const ?[]const u8) !ArrayRef {
    const Builder = switch (tag) {
        .string => array_mod.StringBuilder,
        .large_string => array_mod.LargeStringBuilder,
        .string_view => array_mod.StringViewBuilder,
        .binary => array_mod.BinaryBuilder,
        .large_binary => array_mod.LargeBinaryBuilder,
        .binary_view => array_mod.BinaryViewBuilder,
        else => @compileError("not a string or binary type"),
    };
    var b = try Builder.init(allocator, values.len, 64);
    defer b.deinit();
    for (values) |v| if (v) |x| try b.append(x) else try b.appendNull();
    return b.finish();
}

/// Bytes of row `i` of a string or binary array of any layout, null for a null row.
pub fn bytesAt(data: *const array_mod.ArrayData, i: usize) !?[]const u8 {
    if (data.isNull(i)) return null;
    return switch (data.data_type) {
        .binary => try (array_mod.BinaryArray{ .data = data }).value(i),
        .large_binary => try (array_mod.LargeBinaryArray{ .data = data }).value(i),
        .binary_view => try (array_mod.BinaryViewArray{ .data = data }).value(i),
        else => stringAt(data, i),
    };
}

/// Rows of a string / binary datum (array, chunked or scalar) as optional slices borrowed from
/// the datum; every array part must pass validateFull. Caller frees the outer slice.
pub fn collectStrings(allocator: std.mem.Allocator, datum: Datum) ![]?[]const u8 {
    switch (datum) {
        .scalar => |s| {
            const out = try allocator.alloc(?[]const u8, 1);
            out[0] = switch (s.value) {
                .null => null,
                .string, .binary => |b| b,
                else => return error.TestUnexpectedResult,
            };
            return out;
        },
        .array => |arr| return collectStringParts(allocator, &.{arr}),
        .chunked => |chunked| return collectStringParts(allocator, chunked.chunks()),
    }
}

fn collectStringParts(allocator: std.mem.Allocator, parts: []const ArrayRef) ![]?[]const u8 {
    var total: usize = 0;
    for (parts) |p| total += p.data().length;
    const out = try allocator.alloc(?[]const u8, total);
    errdefer allocator.free(out);
    var pos: usize = 0;
    for (parts) |p| {
        try p.data().validateFull();
        for (0..p.data().length) |i| out[pos + i] = try bytesAt(p.data(), i);
        pos += p.data().length;
    }
    return out;
}

/// Element-wise equality of optional byte strings.
pub fn expectStrings(expected: []const ?[]const u8, actual: []const ?[]const u8) !void {
    try std.testing.expectEqual(expected.len, actual.len);
    for (expected, actual, 0..) |e, a, i| {
        const same = if (e == null or a == null) e == null and a == null else std.mem.eql(u8, e.?, a.?);
        if (!same) {
            std.debug.print("row {d}: expected {?s}, got {?s}\n", .{ i, e, a });
            return error.TestExpectedEqual;
        }
    }
}

/// Invoke scalar function `name` with `options` and compare the string rows with `expected`.
pub fn expectInvokeStrings(ctx: *ExecContext, name: []const u8, args: []const Datum, options: Options, expected: []const ?[]const u8) !void {
    const allocator = std.testing.allocator;
    var out = try ctx.invokeScalar(name, args, options);
    defer out.release();
    const rows = try collectStrings(allocator, out);
    defer allocator.free(rows);
    try expectStrings(expected, rows);
}

/// Invoke scalar function `name` with `options` and compare the rows (of storage `T`) with `expected`.
pub fn expectInvokeWith(comptime T: type, ctx: *ExecContext, name: []const u8, args: []const Datum, options: Options, expected: []const ?T) !void {
    const allocator = std.testing.allocator;
    var out = try ctx.invokeScalar(name, args, options);
    defer out.release();
    switch (out) {
        .array => |arr| try arr.data().validateFull(),
        .chunked => |chunked| for (chunked.chunks()) |c| try c.data().validateFull(),
        .scalar => {},
    }
    const rows = try collect(T, allocator, out);
    defer allocator.free(rows);
    try expectRows(T, expected, rows);
}
