//! Associative transforms (Arrow C++ "Associative transforms"): unique, value_counts and
//! dictionary_encode, vector functions over an array, chunked array or scalar of a key type
//! (numbers, bool, dates, timestamps, times, durations, intervals, decimals, fixed_size_binary
//! and the string / binary layouts; unique and value_counts also take the null type).
//!
//! The rows are hashed by the `Grouper` of hash_aggregate.zig: distinct values come in
//! first-seen order, a null is one value of its own, floats compare by bits after folding every
//! NaN into one (NaN is one value, -0.0 and 0.0 are two, as Arrow).
//!
//! - unique: the distinct values (an array of the input type).
//! - value_counts: `struct<values: T, counts: int64>`, one row per distinct value (the struct
//!   type is stored with the array, see `dense.structArray`).
//! - dictionary_encode (`DictionaryEncodeOptions`): `dictionary<int32, T>`; `mask` (default)
//!   keeps a null row null and leaves nulls out of the dictionary, `encode` gives the null an
//!   entry of the dictionary. A chunked input gives a chunked output whose chunks share one
//!   dictionary.
//!
//! Every input chunk passes `validateLayout` (in `Grouper.consume`).
const std = @import("std");
const core = @import("core.zig");
const dense = @import("dense.zig");
const hash = @import("hash_aggregate.zig");
const bitmap = @import("../bitmap.zig");

const ArrayRef = core.ArrayRef;
const ChunkedArray = core.ChunkedArray;
const DataType = core.DataType;
const Datum = core.Datum;
const ExecContext = core.ExecContext;
const FunctionRegistry = core.FunctionRegistry;
const KernelError = core.KernelError;
const Options = core.Options;
const OwnedBuffer = @import("../buffer.zig").OwnedBuffer;
const SharedBuffer = @import("../buffer.zig").SharedBuffer;
const PrimitiveArray = @import("../array/primitive_array.zig").PrimitiveArray;

const int64_type = DataType{ .int64 = {} };

/// Register unique, value_counts and dictionary_encode.
pub fn register(registry: *FunctionRegistry) KernelError!void {
    try registry.registerVectorKernel("unique", .{
        .signature = .{ .arity = 1, .type_check = keyOrNull, .options_check = dense.noOptions },
        .exec = unique,
    });
    try registry.registerVectorKernel("value_counts", .{
        .signature = .{ .arity = 1, .type_check = keyOrNull, .options_check = dense.noOptions, .result_type_alloc_fn = valueCountsType },
        .exec = valueCounts,
    });
    try registry.registerVectorKernel("dictionary_encode", .{
        .signature = .{ .arity = 1, .type_check = keyOnly, .options_check = encodeOptions, .result_type_alloc_fn = dictionaryType },
        .exec = dictionaryEncode,
    });
}

fn keyOrNull(args: []const Datum) bool {
    return args.len == 1 and (hash.isKeyType(args[0].dataType()) or args[0].dataType() == .null);
}

fn keyOnly(args: []const Datum) bool {
    return args.len == 1 and hash.isKeyType(args[0].dataType());
}

fn encodeOptions(options: Options) bool {
    return options == .none or options == .dictionary_encode;
}

const value_counts_names = [_][]const u8{ "values", "counts" };

fn valueCountsType(allocator: std.mem.Allocator, args: []const Datum, _: Options) KernelError!DataType {
    return dense.structTypeAlloc(allocator, &value_counts_names, &.{ args[0].dataType(), int64_type });
}

fn dictionaryType(allocator: std.mem.Allocator, args: []const Datum, _: Options) KernelError!DataType {
    const value_type = allocator.create(DataType) catch return error.OutOfMemory;
    value_type.* = args[0].dataType();
    return .{ .dictionary = .{ .index_type = .{ .bit_width = 32, .signed = true }, .value_type = value_type } };
}

/// The input as array parts (a scalar becomes a one-row array); the caller releases them.
fn partsOf(ctx: *ExecContext, datum: Datum) KernelError![]ArrayRef {
    const allocator = ctx.allocator;
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
    };
    const out = allocator.alloc(ArrayRef, src.len) catch return error.OutOfMemory;
    for (out, src) |*o, s| o.* = s.retain();
    return out;
}

fn releaseParts(allocator: std.mem.Allocator, parts: []ArrayRef) void {
    for (parts) |*p| p.release();
    allocator.free(parts);
}

/// Distinct values of `parts` and the group id of every row (per part).
const Grouped = struct {
    grouper: hash.Grouper,
    ids: []ArrayRef,

    fn init(allocator: std.mem.Allocator, dt: DataType, parts: []const ArrayRef) KernelError!Grouped {
        var grouper = try hash.Grouper.init(allocator, &.{dt});
        errdefer grouper.deinit();
        const ids = allocator.alloc(ArrayRef, parts.len) catch return error.OutOfMemory;
        var made: usize = 0;
        errdefer {
            for (ids[0..made]) |*i| i.release();
            allocator.free(ids);
        }
        for (parts, ids) |p, *i| {
            if (!p.data().data_type.eql(dt)) return error.InvalidInput;
            i.* = try grouper.consume(&.{p});
            made += 1;
        }
        return .{ .grouper = grouper, .ids = ids };
    }

    fn deinit(self: *Grouped, allocator: std.mem.Allocator) void {
        for (self.ids) |*i| i.release();
        allocator.free(self.ids);
        self.grouper.deinit();
    }

    /// The distinct values (one row per group, first-seen order).
    fn values(self: *const Grouped, allocator: std.mem.Allocator) KernelError!ArrayRef {
        const keys = try self.grouper.uniques();
        defer allocator.free(keys);
        return keys[0];
    }

    fn idsOf(part: ArrayRef) KernelError![]const u32 {
        return (PrimitiveArray(u32){ .data = part.data() }).values() catch error.InvalidInput;
    }
};

/// Total rows of the null-type input (unique / value_counts of the null type).
fn nullRows(parts: []const ArrayRef) usize {
    var n: usize = 0;
    for (parts) |p| n += p.data().length;
    return n;
}

fn unique(ctx: *ExecContext, args: []const Datum, _: Options) KernelError!Datum {
    if (args.len != 1) return error.InvalidArity;
    const dt = args[0].dataType();
    const parts = try partsOf(ctx, args[0]);
    defer releaseParts(ctx.allocator, parts);
    if (dt == .null) return .{ .array = try dense.nullArray(ctx.allocator, dt, @min(nullRows(parts), 1)) };
    var g = try Grouped.init(ctx.allocator, dt, parts);
    defer g.deinit(ctx.allocator);
    return .{ .array = try g.values(ctx.allocator) };
}

fn valueCounts(ctx: *ExecContext, args: []const Datum, _: Options) KernelError!Datum {
    if (args.len != 1) return error.InvalidArity;
    const allocator = ctx.allocator;
    const dt = args[0].dataType();
    const parts = try partsOf(ctx, args[0]);
    defer releaseParts(allocator, parts);
    var values: ArrayRef = undefined;
    var counts_buf: OwnedBuffer = undefined;
    var n: usize = 0;
    if (dt == .null) {
        const rows = nullRows(parts);
        n = @min(rows, 1);
        values = try dense.nullArray(allocator, dt, n);
        errdefer values.release();
        counts_buf = OwnedBuffer.initUninit(allocator, n * 8) catch return error.OutOfMemory;
        if (n == 1) counts_buf.typedSlice(i64)[0] = @intCast(rows);
    } else {
        var g = try Grouped.init(allocator, dt, parts);
        defer g.deinit(allocator);
        n = g.grouper.numGroups();
        values = try g.values(allocator);
        errdefer values.release();
        counts_buf = OwnedBuffer.init(allocator, n * 8) catch return error.OutOfMemory;
        const counts = counts_buf.typedSlice(i64)[0..n];
        for (g.ids) |part| for (try Grouped.idsOf(part)) |id| {
            counts[id] += 1;
        };
    }
    const counts = dense.finishFixed(allocator, int64_type, n, &counts_buf, null) catch |err| {
        values.release();
        return err;
    };
    return .{ .array = try dense.structArray(allocator, &value_counts_names, &.{ values, counts }, n, null) };
}

fn dictionaryEncode(ctx: *ExecContext, args: []const Datum, options: Options) KernelError!Datum {
    if (args.len != 1) return error.InvalidArity;
    const allocator = ctx.allocator;
    const encode_nulls = options == .dictionary_encode and options.dictionary_encode.null_encoding == .encode;
    const dt = args[0].dataType();
    const parts = try partsOf(ctx, args[0]);
    defer releaseParts(allocator, parts);
    var g = try Grouped.init(allocator, dt, parts);
    defer g.deinit(allocator);
    // int32 indices: more distinct values than that is Overflow.
    if (g.grouper.numGroups() > @as(usize, std.math.maxInt(i32)) + 1) return error.Overflow;

    // The group of the null rows, which `mask` leaves out of the dictionary.
    var null_group: ?u32 = null;
    if (!encode_nulls) for (parts, g.ids) |p, ids| {
        if (!p.data().hasNulls()) continue;
        const bits = dense.validityBits(p.data()) orelse continue;
        for (try Grouped.idsOf(ids), 0..) |id, i| if (!bitmap.bitIsSet(bits, p.data().offset + i)) {
            null_group = id;
            break;
        };
        if (null_group != null) break;
    };
    var all = try g.values(allocator);
    defer all.release();
    var dict = if (null_group) |ng| blk: {
        const keep = allocator.alloc(usize, g.grouper.numGroups() - 1) catch return error.OutOfMemory;
        defer allocator.free(keep);
        var k: usize = 0;
        for (0..g.grouper.numGroups()) |id| if (id != ng) {
            keep[k] = id;
            k += 1;
        };
        break :blk try takeRows(allocator, all, keep);
    } else all.retain();
    defer dict.release();
    const out_type = DataType{ .dictionary = .{ .index_type = .{ .bit_width = 32, .signed = true }, .value_type = &dict.data().data_type } };

    const outs = allocator.alloc(ArrayRef, parts.len) catch return error.OutOfMemory;
    var made: usize = 0;
    defer {
        for (outs[0..made]) |*o| o.release();
        allocator.free(outs);
    }
    for (parts, g.ids) |p, ids| {
        const len = p.data().length;
        var idx = OwnedBuffer.initUninit(allocator, len * 4) catch return error.OutOfMemory;
        errdefer idx.deinit();
        for (idx.typedSlice(i32)[0..len], try Grouped.idsOf(ids)) |*d, id| {
            d.* = if (null_group) |ng| (if (id == ng) 0 else if (id > ng) @intCast(id - 1) else @intCast(id)) else @intCast(id);
        }
        var validity: ?OwnedBuffer = null;
        errdefer if (validity) |*v| v.deinit();
        if (null_group != null and p.data().hasNulls()) {
            if (dense.validityBits(p.data())) |bits| {
                var v = try dense.bitsBuffer(allocator, len);
                @memset(v.data[0..bitmap.byteLength(len)], 0);
                for (0..len) |i| if (bitmap.bitIsSet(bits, p.data().offset + i)) bitmap.setBit(v.data, i);
                validity = v;
            }
        }
        var buffers: [2]SharedBuffer = .{ .empty, .empty };
        var null_count: usize = 0;
        if (validity) |*v| {
            null_count = bitmap.countUnsetBit(v.data, len);
            buffers[0] = v.toShared(bitmap.byteLength(len)) catch return error.OutOfMemory;
            validity = null;
        }
        errdefer buffers[0].release();
        buffers[1] = idx.toShared(len * 4) catch return error.OutOfMemory;
        errdefer buffers[1].release();
        outs[made] = ArrayRef.fromOwnedInlineBuffers(allocator, .{
            .data_type = out_type,
            .length = len,
            .null_count = null_count,
            .buffers = &buffers,
            .dictionary = dict.retain(),
        }) catch {
            var d = dict;
            d.release();
            return error.OutOfMemory;
        };
        made += 1;
    }
    if (args[0] != .chunked) return .{ .array = outs[0].retain() };
    const chunked = ChunkedArray.init(allocator, out_type, outs[0..made]) catch |err| return switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        else => error.InvalidInput,
    };
    return .{ .chunked = chunked };
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
