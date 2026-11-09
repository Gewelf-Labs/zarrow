//! Run-end encoding (Arrow C++ "Run-end encoding"): run_end_encode and run_end_decode, vector
//! functions over an array or a chunked array (chunk by chunk, as Arrow: runs do not cross chunk
//! boundaries and empty chunks are dropped, as pyarrow 26).
//!
//! - run_end_encode (`RunEndEncodeOptions.run_end_type`: int16, int32 (default) or int64):
//!   `run_end_encoded<run_ends: I, values: T>` where each run is a stretch of equal consecutive
//!   rows; nulls form runs of their own (`[1, 1, 2, null, null]` gives run ends `[2, 3, 5]` and
//!   values `[1, 2, null]`). Rows compare by value bits (floats bitwise, as Arrow: NaNs with the
//!   same bits are one run, 0.0 and -0.0 two). Types: null, bool, numbers (float16 too), dates,
//!   times, timestamps, durations, intervals, decimals, fixed_size_binary and the string / binary
//!   layouts, also string_view / binary_view (pyarrow 26 has no view kernel), and the nested
//!   types list, large_list, list_view, large_list_view, fixed_size_list, struct and map, whose
//!   rows compare by `integration/compare.zig`'s logical equality (a null row equals a null row,
//!   a struct of null fields is not a null struct, item nulls equal item nulls; float items by
//!   bits, so a NaN item with the same bits is one run where Arrow, comparing by value, makes
//!   two). Dictionaries, unions and extension types are `NoMatchingKernel`. A
//!   chunk longer than the run end type holds is `InvalidInput` (Arrow: "Cannot run-end encode
//!   Arrays with more elements than the run end type can hold").
//! - run_end_decode: the plain values of a `run_end_encoded` array (the value type), also of a
//!   sliced one.
//!
//! Every input chunk passes `validateLayout` (for run_end_decode also its values child) before any
//! buffer is read.
const std = @import("std");
const core = @import("core.zig");
const dense = @import("dense.zig");
const bitmap = @import("../bitmap.zig");
const PrimitiveArray = @import("../array/primitive_array.zig").PrimitiveArray;
const ArrayData = @import("../array/array_data.zig").ArrayData;
const buffer = @import("../buffer.zig");
const compare = @import("../integration/compare.zig");

const ArrayRef = core.ArrayRef;
const ChunkedArray = core.ChunkedArray;
const DataType = core.DataType;
const Datum = core.Datum;
const ExecContext = core.ExecContext;
const FunctionRegistry = core.FunctionRegistry;
const KernelError = core.KernelError;
const Options = core.Options;

/// Register run_end_encode and run_end_decode.
pub fn register(registry: *FunctionRegistry) KernelError!void {
    try registry.registerVectorKernel("run_end_encode", .{
        .signature = .{ .arity = 1, .type_check = encodable, .options_check = encodeOptions, .result_type_alloc_fn = encodedType },
        .exec = runEndEncode,
    });
    try registry.registerVectorKernel("run_end_decode", .{
        .signature = .{ .arity = 1, .type_check = decodable, .options_check = dense.noOptions, .result_type_fn = decodedType },
        .exec = runEndDecode,
    });
}

fn arrayLike(args: []const Datum) bool {
    return args.len == 1 and (args[0] == .array or args[0] == .chunked);
}

/// Whether run_end_encode takes the type (rows of these layouts compare by value).
pub fn isEncodable(dt: DataType) bool {
    return switch (dt) {
        .null, .bool, .int8, .int16, .int32, .int64, .uint8, .uint16, .uint32, .uint64 => true,
        .half_float, .float, .double, .date32, .date64, .time32, .time64, .timestamp, .duration => true,
        .interval_months, .interval_day_time, .interval_month_day_nano => true,
        .decimal32, .decimal64, .decimal128, .decimal256, .fixed_size_binary => true,
        .string, .large_string, .string_view, .binary, .large_binary, .binary_view => true,
        .list, .large_list, .list_view, .large_list_view, .fixed_size_list, .struct_, .map => true,
        else => false,
    };
}

fn encodable(args: []const Datum) bool {
    return arrayLike(args) and isEncodable(args[0].dataType());
}

fn decodable(args: []const Datum) bool {
    return arrayLike(args) and args[0].dataType() == .run_end_encoded;
}

fn encodeOptions(options: Options) bool {
    return options == .none or options == .run_end_encode;
}

fn runEndBits(options: Options) u8 {
    if (options != .run_end_encode) return 32;
    return switch (options.run_end_encode.run_end_type) {
        .int16 => 16,
        .int32 => 32,
        .int64 => 64,
    };
}

fn encodedType(allocator: std.mem.Allocator, args: []const Datum, options: Options) KernelError!DataType {
    const value_type = allocator.create(DataType) catch return error.OutOfMemory;
    value_type.* = args[0].dataType();
    return .{ .run_end_encoded = .{ .run_end_type = .{ .bit_width = runEndBits(options), .signed = true }, .value_type = value_type } };
}

fn decodedType(args: []const Datum, _: Options) KernelError!DataType {
    if (args.len != 1) return error.InvalidArity;
    return args[0].dataType().run_end_encoded.value_type.*;
}

/// Apply `f` to every non-empty chunk; an array gives an array, a chunked array a chunked array
/// (one empty chunk when no chunk has rows, so a chunk owns the result type).
fn perChunk(ctx: *ExecContext, input: Datum, comptime f: anytype, extra: anytype) KernelError!Datum {
    switch (input) {
        .array => |a| return .{ .array = try f(ctx.allocator, a, extra) },
        .chunked => |c| {
            const outs = ctx.allocator.alloc(ArrayRef, @max(c.numChunks(), 1)) catch return error.OutOfMemory;
            var made: usize = 0;
            defer {
                for (outs[0..made]) |*o| o.release();
                ctx.allocator.free(outs);
            }
            for (c.chunks()) |part| {
                if (part.data().length == 0) continue;
                outs[made] = try f(ctx.allocator, part, extra);
                made += 1;
            }
            if (made == 0) {
                var src = if (c.numChunks() > 0) c.chunks()[0].retain() else try dense.nullArray(ctx.allocator, c.dataType(), 0);
                defer src.release();
                outs[0] = try f(ctx.allocator, src, extra);
                made = 1;
            }
            const out = ChunkedArray.init(ctx.allocator, outs[0].data().data_type, outs[0..made]) catch |err| return switch (err) {
                error.OutOfMemory => error.OutOfMemory,
                else => error.InvalidInput,
            };
            return .{ .chunked = out };
        },
        else => return error.InvalidInput,
    }
}

/// run_end_encode: see the file comment.
pub fn runEndEncode(ctx: *ExecContext, args: []const Datum, options: Options) KernelError!Datum {
    if (args.len != 1) return error.InvalidArity;
    if (!isEncodable(args[0].dataType())) return error.UnsupportedType;
    return perChunk(ctx, args[0], encodeChunk, runEndBits(options));
}

fn encodeChunk(allocator: std.mem.Allocator, arr: ArrayRef, bits: u8) KernelError!ArrayRef {
    const d = arr.data();
    d.validateLayout() catch return error.InvalidInput;
    var starts: std.ArrayList(usize) = .empty;
    defer starts.deinit(allocator);
    try runStarts(allocator, arr, &starts);
    var taken = try core.datumTake(.{ .array = arr }, starts.items);
    var values = taken.asArray() orelse {
        taken.release();
        return error.InvalidInput;
    };
    defer values.release();
    return switch (bits) {
        16 => assemble(i16, allocator, d.length, starts.items, values),
        32 => assemble(i32, allocator, d.length, starts.items, values),
        else => assemble(i64, allocator, d.length, starts.items, values),
    };
}

/// A `run_end_encoded` array of `len` rows whose runs start at `starts` over `values` (retained).
fn assemble(comptime T: type, allocator: std.mem.Allocator, len: usize, starts: []const usize, values: ArrayRef) KernelError!ArrayRef {
    if (len > std.math.maxInt(T)) return error.InvalidInput;
    const runs = starts.len;
    var out_ends = buffer.OwnedBuffer.initUninit(allocator, @max(runs, 1) * @sizeOf(T)) catch return error.OutOfMemory;
    errdefer out_ends.deinit();
    for (out_ends.typedSlice(T)[0..runs], 0..) |*e, k| e.* = @intCast(if (k + 1 < runs) starts[k + 1] else len);
    const children = allocator.alloc(ArrayRef, 2) catch return error.OutOfMemory;
    errdefer allocator.free(children);
    var ends_buffers: [2]buffer.SharedBuffer = .{ buffer.SharedBuffer.empty, out_ends.toShared(runs * @sizeOf(T)) catch return error.OutOfMemory };
    children[0] = ArrayRef.fromOwnedInlineBuffers(allocator, .{
        .data_type = dense.dataTypeOf(T),
        .length = runs,
        .null_count = 0,
        .buffers = ends_buffers[0..],
    }) catch {
        ends_buffers[1].release();
        return error.OutOfMemory;
    };
    children[1] = values.retain();
    const out_type: DataType = .{ .run_end_encoded = .{
        .run_end_type = .{ .bit_width = @bitSizeOf(T), .signed = true },
        .value_type = &children[1].data().data_type,
    } };
    return ArrayRef.fromOwnedInlineBuffers(allocator, .{
        .data_type = out_type,
        .length = len,
        .null_count = 0,
        .buffers = &.{},
        .children = children,
    }) catch {
        for (children) |*c| c.release();
        return error.OutOfMemory;
    };
}

/// The rows of `d` (validated) where a run starts: the first row and every row that differs
/// from the one before (validity first, then the value).
fn runStarts(allocator: std.mem.Allocator, arr: ArrayRef, starts: *std.ArrayList(usize)) KernelError!void {
    const d = arr.data();
    if (d.length == 0) return;
    starts.append(allocator, 0) catch return error.OutOfMemory;
    if (d.data_type == .null) return;
    const valid = dense.validityBits(d);
    const Ctx = struct {
        allocator: std.mem.Allocator,
        d: *const ArrayData,
        valid: ?[]const u8,
        starts: *std.ArrayList(usize),

        fn run(self: @This(), rows: anytype) KernelError!void {
            for (1..self.d.length) |i| {
                const same = if (self.valid) |v| blk: {
                    const a = bitmap.bitIsSet(v, self.d.offset + i - 1);
                    const b = bitmap.bitIsSet(v, self.d.offset + i);
                    break :blk a == b and (!a or rows.eql(i - 1, i));
                } else rows.eql(i - 1, i);
                if (!same) self.starts.append(self.allocator, i) catch return error.OutOfMemory;
            }
        }
    };
    const c: Ctx = .{ .allocator = allocator, .d = d, .valid = valid, .starts = starts };
    switch (d.data_type) {
        .bool => return c.run(BoolRows{ .bits = d.buffers[1].data, .offset = d.offset }),
        .list, .large_list, .list_view, .large_list_view, .fixed_size_list, .struct_, .map => return c.run(NestedRows{ .d = d }),
        .string, .large_string, .string_view, .binary, .large_binary, .binary_view => switch (try dense.StringOperand.init(.{ .array = arr })) {
            inline else => |op| return c.run(BytesRows(@TypeOf(op)){ .op = op }),
        },
        else => {},
    }
    const width = (d.data_type.bitWidth() orelse return error.UnsupportedType) / 8;
    switch (width) {
        inline 1, 2, 4, 8 => |w| {
            const xs = (PrimitiveArray(@Int(.unsigned, w * 8)){ .data = d }).values() catch return error.InvalidInput;
            return c.run(WordRows(@Int(.unsigned, w * 8)){ .xs = xs });
        },
        else => {
            if (width == 0) return error.UnsupportedType;
            const start = d.offset * width;
            if (d.buffers.len < 2 or d.buffers[1].data.len < start + d.length * width) return error.InvalidInput;
            return c.run(FixedRows{ .bytes = d.buffers[1].data[start..][0 .. d.length * width], .width = width });
        },
    }
}

const BoolRows = struct {
    bits: []const u8,
    offset: usize,
    inline fn eql(self: BoolRows, i: usize, j: usize) bool {
        return bitmap.bitIsSet(self.bits, self.offset + i) == bitmap.bitIsSet(self.bits, self.offset + j);
    }
};

fn WordRows(comptime U: type) type {
    return struct {
        xs: []const U,
        inline fn eql(self: @This(), i: usize, j: usize) bool {
            return self.xs[i] == self.xs[j];
        }
    };
}

const FixedRows = struct {
    bytes: []const u8,
    width: usize,
    inline fn eql(self: FixedRows, i: usize, j: usize) bool {
        return std.mem.eql(u8, self.bytes[i * self.width ..][0..self.width], self.bytes[j * self.width ..][0..self.width]);
    }
};

/// Rows of a nested array compared by `integration/compare.zig` (Arrow's logical equality: the
/// validity, then the values of valid rows, nested nulls equal to nested nulls; a malformed child
/// compares unequal).
const NestedRows = struct {
    d: *const ArrayData,
    fn eql(self: NestedRows, i: usize, j: usize) bool {
        return compare.rowsEqual(self.d, i, self.d, j);
    }
};

fn BytesRows(comptime Op: type) type {
    return struct {
        op: Op,
        inline fn eql(self: @This(), i: usize, j: usize) bool {
            return std.mem.eql(u8, self.op.get(i), self.op.get(j));
        }
    };
}

/// run_end_decode: see the file comment.
pub fn runEndDecode(ctx: *ExecContext, args: []const Datum, _: Options) KernelError!Datum {
    if (args.len != 1) return error.InvalidArity;
    if (args[0].dataType() != .run_end_encoded) return error.UnsupportedType;
    if (args[0] == .chunked and args[0].chunked.numChunks() == 0) {
        const out = ChunkedArray.init(ctx.allocator, args[0].dataType().run_end_encoded.value_type.*, &.{}) catch return error.OutOfMemory;
        return .{ .chunked = out };
    }
    return perChunk(ctx, args[0], decodeChunk, {});
}

fn decodeChunk(allocator: std.mem.Allocator, arr: ArrayRef, _: void) KernelError!ArrayRef {
    const d = arr.data();
    d.validateLayout() catch return error.InvalidInput;
    d.children[1].data().validateLayout() catch return error.InvalidInput;
    return switch (d.data_type.run_end_encoded.run_end_type.bit_width) {
        16 => decodeWith(i16, allocator, d),
        32 => decodeWith(i32, allocator, d),
        64 => decodeWith(i64, allocator, d),
        else => error.InvalidInput,
    };
}

/// Take the values child at the run of every row `offset .. offset + length` (one merge walk
/// over the run ends, which `validateLayout` checked to rise and to cover every row).
fn decodeWith(comptime T: type, allocator: std.mem.Allocator, d: *const ArrayData) KernelError!ArrayRef {
    const ends = (PrimitiveArray(T){ .data = d.children[0].data() }).values() catch return error.InvalidInput;
    const phys = allocator.alloc(usize, d.length) catch return error.OutOfMemory;
    defer allocator.free(phys);
    var k: usize = 0;
    for (phys, d.offset..) |*p, pos| {
        while (k < ends.len and ends[k] <= pos) k += 1;
        if (k == ends.len) return error.InvalidInput;
        p.* = k;
    }
    var taken = try core.datumTake(.{ .array = d.children[1] }, phys);
    return taken.asArray() orelse {
        taken.release();
        return error.InvalidInput;
    };
}
