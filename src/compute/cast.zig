//! The `cast` function (Arrow C++ "Conversions"): `Options.cast` names the target type
//! (`to_type`, required); a cast is safe when both `CastOptions.safe` and `ExecConfig.safe_cast`
//! are true. Null rows stay null; casting to the input type returns the input.
//!
//! Supported conversions:
//! - between int8..int64, uint8..uint64, float32, float64 and bool. Safe: an integer out of the
//!   target range, a float with a fraction or out of the integer range (or NaN), and an integer
//!   beyond 2^24 / 2^53 cast to float32 / float64 are `InvalidCast`. Unsafe: integers wrap (two's
//!   complement truncation), floats truncate toward zero and saturate (NaN gives 0; Arrow C++
//!   leaves that case undefined). float64 to float32 rounds (overflow gives inf) in both modes.
//!   bool to number gives 0 / 1, number to bool is `x != 0`.
//! - numbers and bool to string / large_string / string_view: integers in plain decimal, floats in
//!   the shortest digits that round-trip (Python `repr` layout without a trailing `.0`: exponent
//!   form below 1e-4 and from 1e16, `inf`, `-inf`, `nan`), bool as `true` / `false`.
//! - string / large_string / string_view to numbers and bool: an optional sign and decimal digits
//!   (floats: also a fraction, an exponent, `inf`, `infinity`, `nan`, any case; bool: `true`,
//!   `false`, `1`, `0`, any case). Text that does not parse, or does not fit, is `InvalidCast` in
//!   both modes, as in Arrow C++.
//! - among string, large_string, string_view, binary, large_binary and binary_view. A string
//!   target is checked for UTF-8 in both modes (Arrow C++ skips the check when unsafe). Offsets
//!   past the 32-bit range are `Overflow`.
//! - date32 <-> date64, timestamp unit changes, timestamp -> date32 / date64 (floored to the day),
//!   date32 / date64 -> timestamp, time32 / time64 unit changes, timestamp -> time32 / time64 (the
//!   time of day), duration unit changes. Safe: losing a fraction (date64 to date32, a finer to a
//!   coarser unit) or overflowing is `InvalidCast`; unsafe truncates toward zero and wraps. A
//!   timezone change keeps the instants (values unchanged); a timestamp in a timezone other than
//!   UTC to a date or time is `NotImplemented` (Arrow uses the local time, no timezone database).
//! - int32 <-> date32 / time32 and int64 <-> date64 / timestamp / time64 / duration keep the
//!   stored values (Arrow's zero-copy casts; other integer widths are not cast, as in Arrow).
//! - dictionary to its value type (decoded with take), then on to any target above; a value
//!   array to `dictionary<I, value type>` for every integer index type `I` through
//!   `dictionaryEncode` (more distinct values than `I` holds is `InvalidCast`); a dictionary to a
//!   dictionary of another index or value type (the values are cast first). The target's
//!   `value_type` must outlive the result.
//! - the null type to any type above (all rows null).
//! Other pairs are `UnsupportedType`.
const std = @import("std");
const bitmap = @import("../bitmap.zig");
const core = @import("core.zig");
const datatype = @import("../datatype.zig");
const dense = @import("dense.zig");
const concat = @import("../concat_array_refs.zig");

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
const TypeTag = std.meta.Tag(DataType);

/// Register `cast` (one kernel for every input type; the pair is checked at execution).
pub fn register(registry: *FunctionRegistry) KernelError!void {
    try registry.registerScalarKernel("cast", .{
        .signature = .{
            .arity = 1,
            .options_check = castOptions,
            .result_type_fn = resultType,
        },
        .exec = exec,
    });
}

fn castOptions(options: Options) bool {
    return options == .cast and options.cast.to_type != null;
}

fn resultType(_: []const Datum, options: Options) KernelError!DataType {
    return options.cast.to_type.?;
}

/// Cast each row to `options.cast.to_type` (array, chunked or scalar).
fn exec(ctx: *ExecContext, args: []const Datum, options: Options) KernelError!Datum {
    if (args.len != 1) return error.InvalidArity;
    if (args[0].dataType().eql(options.cast.to_type.?)) return args[0].retain();
    return dense.execElementwiseWith(ctx, args, options, options.cast.to_type.?, true, chunk);
}

fn chunk(ctx: *ExecContext, values: []const ExecChunkValue, _: usize, options: Options) KernelError!ArrayRef {
    var arr = try dense.chunkArray(ctx.allocator, values[0]);
    defer arr.release();
    return castArray(ctx, arr, options.cast.to_type.?, options.cast.safe and ctx.safeCastEnabled());
}

const Kind = enum { null, boolean, number, string, binary, temporal, dictionary, other };

fn kindOf(dt: DataType) Kind {
    return switch (dt) {
        .null => .null,
        .bool => .boolean,
        .int8, .int16, .int32, .int64, .uint8, .uint16, .uint32, .uint64, .float, .double => .number,
        .string, .large_string, .string_view => .string,
        .binary, .large_binary, .binary_view => .binary,
        .date32, .date64, .timestamp, .time32, .time64, .duration => .temporal,
        .dictionary => .dictionary,
        else => .other,
    };
}

/// Cast a whole array that passed `validateLayout` (or was built from a scalar).
fn castArray(ctx: *ExecContext, arr: ArrayRef, to: DataType, safe: bool) KernelError!ArrayRef {
    const data = arr.data();
    const from = data.data_type;
    if (from.eql(to)) return arr.retain();
    const fk = kindOf(from);
    const tk = kindOf(to);
    if (fk == .null and tk != .dictionary and tk != .other) return dense.nullArray(ctx.allocator, to, data.length);
    if (fk == .dictionary) {
        var decoded = try decode(ctx.allocator, arr);
        defer decoded.release();
        if (tk == .dictionary) return encode(ctx, decoded, to, safe, true);
        return castArray(ctx, decoded, to, safe);
    }
    if (tk == .dictionary) return encode(ctx, arr, to, safe, false);
    if (sameStorage(from, to)) return reinterpret(ctx, data, to);
    if ((fk == .number or fk == .boolean) and (tk == .number or tk == .boolean)) return numeric(ctx, data, to, safe);
    if ((fk == .number or fk == .boolean) and tk == .string) return toText(ctx, data, to);
    if (fk == .string and (tk == .number or tk == .boolean)) return fromText(ctx, arr, to);
    if ((fk == .string or fk == .binary) and (tk == .string or tk == .binary)) return bytesCast(ctx, arr, to);
    if (fk == .temporal and tk == .temporal) {
        if (!temporalPair(from, to)) return error.UnsupportedType;
        if (from == .timestamp and (to == .date32 or to == .date64 or to == .time32 or to == .time64)) {
            if (from.timestamp.timezone) |tz| if (!@import("temporal.zig").isUtc(tz)) return error.NotImplemented;
        }
        return temporal(ctx, data, to, safe);
    }
    return error.UnsupportedType;
}

/// int32 with date32 / time32, int64 with date64 / timestamp / time64 / duration (either way):
/// the stored values are kept (Arrow's zero-copy casts).
fn sameStorage(from: DataType, to: DataType) bool {
    const S = struct {
        fn width(dt: DataType) ?u8 {
            return switch (dt) {
                .date32, .time32 => 32,
                .date64, .timestamp, .time64, .duration => 64,
                else => null,
            };
        }
    };
    if (from == .int32 or from == .int64) {
        const w = S.width(to) orelse return false;
        return w == @as(u8, if (from == .int32) 32 else 64);
    }
    if (to == .int32 or to == .int64) {
        const w = S.width(from) orelse return false;
        return w == @as(u8, if (to == .int32) 32 else 64);
    }
    return false;
}

/// The array over the same buffers with type `to` (no copy).
fn reinterpret(ctx: *ExecContext, data: *const ArrayData, to: DataType) KernelError!ArrayRef {
    var layout = data.*;
    layout.data_type = to;
    return ArrayRef.fromBorrowed(ctx.allocator, layout) catch error.OutOfMemory;
}

/// The temporal pairs Arrow casts: dates and timestamps among themselves, times among
/// themselves, a timestamp to a time, durations among themselves.
fn temporalPair(from: DataType, to: DataType) bool {
    const Group = enum { calendar, time, duration };
    const S = struct {
        fn group(dt: DataType) Group {
            return switch (dt) {
                .time32, .time64 => .time,
                .duration => .duration,
                else => .calendar,
            };
        }
    };
    const f = S.group(from);
    const t = S.group(to);
    return f == t or (from == .timestamp and t == .time);
}

// ------------------------------------------------------------------------------ dictionary

/// Return the values of a dictionary array (`datumTakeNullable` over the checked indices).
fn decode(allocator: std.mem.Allocator, arr: ArrayRef) KernelError!ArrayRef {
    const data = arr.data();
    const dict = data.dictionary orelse return error.InvalidInput;
    // validateLayout checks neither the dictionary nor the index values.
    dict.data().validateLayout() catch return error.InvalidInput;
    data.validateDictionaryIndices(data.data_type.dictionary, data.offset, data.offset + data.length) catch return error.InvalidInput;
    const indices = allocator.alloc(?usize, data.length) catch return error.OutOfMemory;
    defer allocator.free(indices);
    const it = data.data_type.dictionary.index_type;
    switch (it.bit_width) {
        inline 8, 16, 32, 64 => |w| {
            if (it.signed) try readIndices(@Int(.signed, w), data, indices) else try readIndices(@Int(.unsigned, w), data, indices);
        },
        else => return error.InvalidInput,
    }
    var out = try core.datumTakeNullable(.{ .array = dict }, indices);
    errdefer out.release();
    return switch (out) {
        .array => |a| a,
        else => error.InvalidInput,
    };
}

fn readIndices(comptime I: type, data: *const ArrayData, out: []?usize) KernelError!void {
    // Not bytesAsSlice: an IPC index buffer may carry a trailing partial element (fuzz finding,
    // fuzz/corpus/array-layout/regress-d0767e18.bin); the typed accessor ignores it.
    const raw = (data.typedBufferAt(1, I) catch return error.InvalidInput)[data.offset..][0..data.length];
    for (out, raw, 0..) |*o, x, i| o.* = if (data.isNull(i)) null else @intCast(x);
}

/// Dictionary-encode `arr` when the target is `dictionary<I, type of arr>` (or, with
/// `cast_values`, of a type `arr` casts to first: the values of a dictionary input).
fn encode(ctx: *ExecContext, arr: ArrayRef, to: DataType, safe: bool, cast_values: bool) KernelError!ArrayRef {
    const d = to.dictionary;
    if (d.ordered) return error.UnsupportedType;
    switch (d.index_type.bit_width) {
        8, 16, 32, 64 => {},
        else => return error.UnsupportedType,
    }
    if (!d.value_type.eql(arr.data().data_type)) {
        if (!cast_values) return error.UnsupportedType;
        var values = try castArray(ctx, arr, d.value_type.*, safe);
        defer values.release();
        return encode(ctx, values, to, safe, false);
    }
    var encoded = concat.dictionaryEncode(ctx.allocator, arr, d.value_type) catch |err| return switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        error.UnsupportedType => error.UnsupportedType,
        else => error.InvalidInput,
    };
    if (d.index_type.bit_width == 32 and d.index_type.signed) return encoded;
    defer encoded.release();
    return switch (d.index_type.bit_width) {
        inline 8, 16, 32, 64 => |w| if (d.index_type.signed) reindex(@Int(.signed, w), ctx, encoded, to) else reindex(@Int(.unsigned, w), ctx, encoded, to),
        else => unreachable,
    };
}

/// The int32-indexed dictionary array `encoded` with its indices stored as `I` (`InvalidCast`
/// when an index does not fit, in both modes, as Arrow).
fn reindex(comptime I: type, ctx: *ExecContext, encoded: ArrayRef, to: DataType) KernelError!ArrayRef {
    const data = encoded.data();
    // dictionaryEncode output: offset 0, a validity buffer and an int32 index buffer.
    if (data.offset != 0 or data.buffers.len != 2) return error.InvalidInput;
    const src = (data.typedBufferAt(1, i32) catch return error.InvalidInput)[0..data.length];
    var buf = @import("../buffer.zig").OwnedBuffer.init(ctx.allocator, data.length * @sizeOf(I)) catch return error.OutOfMemory;
    for (buf.typedSlice(I)[0..data.length], src, 0..) |*d, x, i| {
        // Null rows hold 0 (dictionaryEncode), which every index type holds.
        d.* = std.math.cast(I, x) orelse {
            if (data.isNull(i)) continue;
            buf.deinit();
            return error.InvalidCast;
        };
    }
    var shared = buf.toShared(data.length * @sizeOf(I)) catch {
        buf.deinit();
        return error.OutOfMemory;
    };
    defer shared.release();
    var layout = data.*;
    layout.data_type = to;
    const bufs = [2]@import("../buffer.zig").SharedBuffer{ data.buffers[0], shared };
    layout.buffers = &bufs;
    return ArrayRef.fromBorrowed(ctx.allocator, layout) catch error.OutOfMemory;
}

// ------------------------------------------------------------------------------ numbers

fn isSigned(comptime T: type) bool {
    return @typeInfo(T) == .int and @typeInfo(T).int.signedness == .signed;
}

/// Whether every `F` converts to `T` exactly (no check needed in safe mode).
fn exact(comptime F: type, comptime T: type) bool {
    const fi = @typeInfo(F);
    const ti = @typeInfo(T);
    if (fi == .float) return ti == .float and @bitSizeOf(T) >= @bitSizeOf(F) or F == f64 and T == f32;
    if (ti == .float) return fi.int.bits - @intFromBool(isSigned(F)) <= std.math.floatMantissaBits(T) + 1;
    return std.math.minInt(F) >= std.math.minInt(T) and std.math.maxInt(F) <= std.math.maxInt(T);
}

/// Convert one value: wrap integers, truncate and saturate floats (defined for every input).
inline fn convert(comptime F: type, comptime T: type, x: F) T {
    if (@typeInfo(F) == .float) {
        if (@typeInfo(T) == .float) return @floatCast(x);
        const t = @trunc(x);
        if (inIntRange(T, t)) return @intFromFloat(t);
        return if (x > 0) std.math.maxInt(T) else if (x < 0) std.math.minInt(T) else 0;
    }
    if (@typeInfo(T) == .float) return @floatFromInt(x);
    if (comptime exact(F, T)) return x;
    const U = @Int(.unsigned, @bitSizeOf(T));
    return @bitCast(@as(U, @truncate(@as(u128, @bitCast(@as(i128, x))))));
}

/// Whether the truncated float `t` fits `T` (false for NaN).
inline fn inIntRange(comptime T: type, t: anytype) bool {
    const F = @TypeOf(t);
    const max_excl: F = comptime std.math.pow(F, 2, @bitSizeOf(T) - (if (isSigned(T)) 1 else 0));
    const min_incl: F = if (comptime isSigned(T)) -max_excl else 0;
    return t >= min_incl and t < max_excl;
}

/// Whether `convert(F, T, x)` loses information (the safe-mode check).
inline fn lossy(comptime F: type, comptime T: type, x: F) bool {
    if (@typeInfo(F) == .float) {
        if (@typeInfo(T) == .float) return false;
        const t = @trunc(x);
        return !inIntRange(T, t) or t != x;
    }
    if (@typeInfo(T) == .float) {
        const limit: i128 = @as(i128, 1) << (std.math.floatMantissaBits(T) + 1);
        return @abs(@as(i128, x)) > limit;
    }
    return @as(i128, x) < std.math.minInt(T) or @as(i128, x) > std.math.maxInt(T);
}

fn numeric(ctx: *ExecContext, data: *const ArrayData, to: DataType, safe: bool) KernelError!ArrayRef {
    if (data.data_type == .bool) {
        inline for (dense.numeric_types) |T| if (std.meta.activeTag(to) == comptime std.meta.activeTag(dense.dataTypeOf(T))) return fromBool(T, ctx, data);
        return error.UnsupportedType;
    }
    inline for (dense.numeric_types) |F| if (std.meta.activeTag(data.data_type) == comptime std.meta.activeTag(dense.dataTypeOf(F))) {
        const src = (PrimitiveArray(F){ .data = data }).values() catch return error.InvalidInput;
        if (to == .bool) return toBool(F, ctx, data, src);
        inline for (dense.numeric_types) |T| if (std.meta.activeTag(to) == comptime std.meta.activeTag(dense.dataTypeOf(T))) {
            return numToNum(F, T, ctx, data, src, safe);
        };
    };
    return error.UnsupportedType;
}

fn numToNum(comptime F: type, comptime T: type, ctx: *ExecContext, data: *const ArrayData, src: []const F, safe: bool) KernelError!ArrayRef {
    var out = try dense.DenseBinaryOutput(T).initValidity(ctx.allocator, data.length, data, null, dense.dataTypeOf(T));
    errdefer out.deinit();
    const dst = out.values();
    if (comptime exact(F, T)) {
        for (dst, src) |*d, x| d.* = convert(F, T, x);
    } else {
        // One dense pass with a branch-free loss flag; rows are checked one by one (null rows may
        // hold anything) only when some row lost information.
        var any: u8 = 0;
        for (dst, src) |*d, x| {
            d.* = convert(F, T, x);
            any |= @intFromBool(lossy(F, T, x));
        }
        if (safe and any != 0) for (src, 0..) |x, i| {
            if (lossy(F, T, x) and out.isValid(i)) return error.InvalidCast;
        };
    }
    return out.finish() catch error.OutOfMemory;
}

fn fromBool(comptime T: type, ctx: *ExecContext, data: *const ArrayData) KernelError!ArrayRef {
    var out = try dense.DenseBinaryOutput(T).initValidity(ctx.allocator, data.length, data, null, dense.dataTypeOf(T));
    errdefer out.deinit();
    for (out.values(), 0..) |*d, i| d.* = if (bitmap.bitIsSet(data.buffers[1].data, data.offset + i)) 1 else 0;
    return out.finish() catch error.OutOfMemory;
}

fn toBool(comptime F: type, ctx: *ExecContext, data: *const ArrayData, src: []const F) KernelError!ArrayRef {
    var out = try dense.DenseBinaryOutput(bool).initValidity(ctx.allocator, data.length, data, null, .{ .bool = {} });
    errdefer out.deinit();
    const bits = out.values();
    @memset(bits, 0);
    for (src, 0..) |x, i| bits[i / 8] |= @as(u8, @intFromBool(x != 0)) << @intCast(i % 8);
    return out.finish() catch error.OutOfMemory;
}

// ------------------------------------------------------------------------------ to text

fn toText(ctx: *ExecContext, data: *const ArrayData, to: DataType) KernelError!ArrayRef {
    const len = data.length;
    const bits = dense.validityBits(data);
    if (data.data_type == .bool) {
        var out = try dense.VarBinaryOutput.init(ctx.allocator, to, len, try dense.combinedValidity(ctx.allocator, len, &.{data}), len * 5);
        errdefer out.deinit();
        for (0..len) |i| {
            if (bits) |b| if (!bitmap.bitIsSet(b, data.offset + i)) {
                try out.commit(0);
                continue;
            };
            try out.append(if (bitmap.bitIsSet(data.buffers[1].data, data.offset + i)) "true" else "false");
        }
        return out.finish();
    }
    inline for (dense.numeric_types) |F| if (std.meta.activeTag(data.data_type) == comptime std.meta.activeTag(dense.dataTypeOf(F))) {
        const src = (PrimitiveArray(F){ .data = data }).values() catch return error.InvalidInput;
        if (@typeInfo(F) == .int) switch (to) {
            .string => return intText(F, i32, ctx, data, src, to),
            .large_string => return intText(F, i64, ctx, data, src, to),
            else => {},
        };
        // Views (and floats): pass 1 sums the lengths, pass 2 appends row by row.
        var total: usize = len * 8;
        if (@typeInfo(F) == .int) {
            total = 0;
            for (src, 0..) |x, i| {
                if (bits) |b| if (!bitmap.bitIsSet(b, data.offset + i)) continue;
                total += intLen(x);
            }
        }
        var out = try dense.VarBinaryOutput.init(ctx.allocator, to, len, try dense.combinedValidity(ctx.allocator, len, &.{data}), total);
        errdefer out.deinit();
        for (src, 0..) |x, i| {
            if (bits) |b| if (!bitmap.bitIsSet(b, data.offset + i)) {
                try out.commit(0);
                continue;
            };
            if (@typeInfo(F) == .int) {
                const n = intLen(x);
                writeInt(try out.reserve(n), x);
                try out.commit(n);
            } else {
                var buf: [64]u8 = undefined;
                try out.append(formatFloat(&buf, x));
            }
        }
        return out.finish();
    };
    return error.UnsupportedType;
}

/// Integers to string / large_string: pass 1 writes the offsets (the digit count of each valid
/// row), pass 2 writes the digits in place.
fn intText(comptime F: type, comptime O: type, ctx: *ExecContext, data: *const ArrayData, src: []const F, to: DataType) KernelError!ArrayRef {
    const len = data.length;
    const bits = dense.validityBits(data);
    var out = try dense.VarBinaryOutput.init(ctx.allocator, to, len, try dense.combinedValidity(ctx.allocator, len, &.{data}), 0);
    errdefer out.deinit();
    const offsets = out.index.typedSlice(O)[0 .. len + 1];
    var total: usize = 0;
    for (src, offsets[1..], 0..) |x, *o, i| {
        const valid = if (bits) |b| bitmap.bitIsSet(b, data.offset + i) else true;
        total += intLen(x) * @intFromBool(valid);
        o.* = @truncate(@as(i64, @intCast(@min(total, std.math.maxInt(i64)))));
    }
    if (total > std.math.maxInt(O)) return error.Overflow;
    const bytes = try out.reserve(total);
    for (src, offsets[0..len], offsets[1..]) |x, a, b| {
        if (a != b) writeInt(bytes[@intCast(a)..@intCast(b)], x);
    }
    out.pos = total;
    out.row = len;
    return out.finish();
}

/// Number of characters of the decimal form of `x` (with its minus sign): branch-free compares
/// against the powers of ten.
fn intLen(x: anytype) usize {
    const m: u64 = @intCast(@abs(@as(i128, x)));
    var n: usize = @as(usize, @intFromBool(x < 0)) + 1;
    comptime var p: u64 = 10;
    inline while (true) {
        n += @intFromBool(m >= p);
        if (p > std.math.maxInt(u64) / 10) break;
        p *= 10;
    }
    return n;
}

const digit_pairs = blk: {
    var pairs: [200]u8 = undefined;
    for (0..100) |i| {
        pairs[2 * i] = '0' + i / 10;
        pairs[2 * i + 1] = '0' + i % 10;
    }
    break :blk pairs;
};

/// Write the decimal form of `x` into `dst` (exactly `intLen(x)` bytes), two digits at a time.
fn writeInt(dst: []u8, x: anytype) void {
    var m: u64 = @intCast(@abs(@as(i128, x)));
    var i = dst.len;
    while (m >= 100) {
        const r: usize = @intCast(m % 100);
        m /= 100;
        i -= 2;
        dst[i..][0..2].* = digit_pairs[2 * r ..][0..2].*;
    }
    if (m >= 10) {
        i -= 2;
        dst[i..][0..2].* = digit_pairs[2 * m ..][0..2].*;
    } else {
        i -= 1;
        dst[i] = @intCast('0' + m);
    }
    if (x < 0) dst[0] = '-';
}

/// Shortest round-trip form of `x` in Python `repr` layout, without a trailing `.0`.
pub fn formatFloat(buf: *[64]u8, x: anytype) []const u8 {
    if (std.math.isNan(x)) return "nan";
    if (std.math.isInf(x)) return if (x > 0) "inf" else "-inf";
    var tmp: [64]u8 = undefined;
    // Scientific shortest digits: "[-]d[.ddd]e[-]x".
    const sci = std.fmt.float.render(&tmp, x, .{}) catch unreachable;
    const neg = sci[0] == '-';
    const e_at = std.mem.indexOfScalar(u8, sci, 'e').?;
    const exp = std.fmt.parseInt(i32, sci[e_at + 1 ..], 10) catch unreachable;
    var digits_buf: [32]u8 = undefined;
    var n: usize = 0;
    for (sci[@intFromBool(neg)..e_at]) |c| if (c != '.') {
        digits_buf[n] = c;
        n += 1;
    };
    const digits = digits_buf[0..n];
    var w: usize = 0;
    if (neg) {
        buf[0] = '-';
        w = 1;
    }
    if (exp < -4 or exp >= 16) {
        buf[w] = digits[0];
        w += 1;
        if (digits.len > 1) {
            buf[w] = '.';
            @memcpy(buf[w + 1 ..][0 .. digits.len - 1], digits[1..]);
            w += digits.len;
        }
        const s = std.fmt.bufPrint(buf[w..], "e{c}{d:0>2}", .{ @as(u8, if (exp < 0) '-' else '+'), @abs(exp) }) catch unreachable;
        return buf[0 .. w + s.len];
    }
    if (exp < 0) {
        const zeros: usize = @intCast(-exp - 1);
        @memcpy(buf[w..][0..2], "0.");
        @memset(buf[w + 2 ..][0..zeros], '0');
        @memcpy(buf[w + 2 + zeros ..][0..digits.len], digits);
        return buf[0 .. w + 2 + zeros + digits.len];
    }
    const int_len: usize = @intCast(exp + 1);
    if (digits.len <= int_len) {
        @memcpy(buf[w..][0..digits.len], digits);
        @memset(buf[w + digits.len ..][0 .. int_len - digits.len], '0');
        return buf[0 .. w + int_len];
    }
    @memcpy(buf[w..][0..int_len], digits[0..int_len]);
    buf[w + int_len] = '.';
    @memcpy(buf[w + int_len + 1 ..][0 .. digits.len - int_len], digits[int_len..]);
    return buf[0 .. w + digits.len + 1];
}

// ------------------------------------------------------------------------------ from text

fn fromText(ctx: *ExecContext, arr: ArrayRef, to: DataType) KernelError!ArrayRef {
    const data = arr.data();
    const rows = try dense.StringOperand.init(.{ .array = arr });
    const bits = dense.validityBits(data);
    if (to == .bool) {
        var out = try dense.DenseBinaryOutput(bool).initValidity(ctx.allocator, data.length, data, null, to);
        errdefer out.deinit();
        const dst = out.values();
        @memset(dst, 0);
        switch (rows) {
            inline else => |r| for (0..data.length) |i| {
                if (bits) |b| if (!bitmap.bitIsSet(b, data.offset + i)) continue;
                if (try parseBool(r.get(i))) dst[i / 8] |= @as(u8, 1) << @intCast(i % 8);
            },
        }
        return out.finish() catch error.OutOfMemory;
    }
    inline for (dense.numeric_types) |T| if (std.meta.activeTag(to) == comptime std.meta.activeTag(dense.dataTypeOf(T))) {
        var out = try dense.DenseBinaryOutput(T).initValidity(ctx.allocator, data.length, data, null, to);
        errdefer out.deinit();
        const dst = out.values();
        switch (rows) {
            inline else => |r| for (dst, 0..) |*d, i| {
                d.* = 0;
                if (bits) |b| if (!bitmap.bitIsSet(b, data.offset + i)) continue;
                d.* = try parseNumber(T, r.get(i));
            },
        }
        return out.finish() catch error.OutOfMemory;
    };
    return error.UnsupportedType;
}

fn parseBool(s: []const u8) KernelError!bool {
    if (std.ascii.eqlIgnoreCase(s, "true") or std.mem.eql(u8, s, "1")) return true;
    if (std.ascii.eqlIgnoreCase(s, "false") or std.mem.eql(u8, s, "0")) return false;
    return error.InvalidCast;
}

/// Parse an optional sign and decimal digits (integers), or a decimal float with an optional
/// exponent, `inf`, `infinity` or `nan`; anything else (spaces, `_`, hex) is `InvalidCast`.
pub fn parseNumber(comptime T: type, s: []const u8) KernelError!T {
    const body = if (s.len > 0 and (s[0] == '-' or s[0] == '+')) s[1..] else s;
    if (body.len == 0) return error.InvalidCast;
    if (@typeInfo(T) == .int) {
        for (body) |c| if (!std.ascii.isDigit(c)) return error.InvalidCast;
        return std.fmt.parseInt(T, s, 10) catch error.InvalidCast;
    }
    const special = std.ascii.eqlIgnoreCase(body, "inf") or std.ascii.eqlIgnoreCase(body, "infinity") or std.ascii.eqlIgnoreCase(body, "nan");
    if (!special) for (body) |c| switch (c) {
        '0'...'9', '.', 'e', 'E', '+', '-' => {},
        else => return error.InvalidCast,
    };
    return std.fmt.parseFloat(T, s) catch error.InvalidCast;
}

// ------------------------------------------------------------------------------ bytes

fn bytesCast(ctx: *ExecContext, arr: ArrayRef, to: DataType) KernelError!ArrayRef {
    const data = arr.data();
    const check_utf8 = kindOf(data.data_type) == .binary and kindOf(to) == .string;
    const same_layout = switch (data.data_type) {
        .string, .binary => to == .string or to == .binary,
        .large_string, .large_binary => to == .large_string or to == .large_binary,
        else => to == .string_view or to == .binary_view,
    };
    var out = if (same_layout) blk: {
        var retyped = data.*;
        retyped.data_type = to;
        break :blk ArrayRef.fromBorrowed(ctx.allocator, retyped) catch return error.OutOfMemory;
    } else try copyBytes(ctx, arr, to);
    errdefer out.release();
    // The UTF-8 check of validateFull (non-null rows only), in both modes.
    if (check_utf8) out.data().validateFull() catch |err| return if (err == error.InvalidUtf8) error.InvalidCast else error.InvalidInput;
    return out;
}

fn copyBytes(ctx: *ExecContext, arr: ArrayRef, to: DataType) KernelError!ArrayRef {
    const data = arr.data();
    const len = data.length;
    const rows = try dense.StringOperand.init(.{ .array = arr });
    var out = try dense.VarBinaryOutput.init(ctx.allocator, to, len, try dense.combinedValidity(ctx.allocator, len, &.{data}), dense.totalBytes(rows, len));
    errdefer out.deinit();
    const bits = dense.validityBits(data);
    switch (rows) {
        inline else => |r| for (0..len) |i| {
            if (bits) |b| if (!bitmap.bitIsSet(b, data.offset + i)) {
                try out.commit(0);
                continue;
            };
            try out.append(r.get(i));
        },
    }
    return out.finish();
}

// ------------------------------------------------------------------------------ temporal

/// Nanoseconds per stored unit (date32: a day, date64: a millisecond).
fn tickNs(dt: DataType) i64 {
    const unit = switch (dt) {
        .date32 => return 86_400_000_000_000,
        .date64 => return 1_000_000,
        .timestamp => |ts| ts.unit,
        .time32, .time64 => |t| t.unit,
        .duration => |d| d.unit,
        else => unreachable,
    };
    return switch (unit) {
        .second => 1_000_000_000,
        .millisecond => 1_000_000,
        .microsecond => 1_000,
        .nanosecond => 1,
    };
}

/// One temporal conversion: `v / div` (floored or truncated), then `* mul`.
const Plan = struct {
    div: i64 = 1,
    floor: bool = false,
    /// A remainder of `v / div` is a loss (safe mode fails).
    exact_div: bool = true,
    mul: i64 = 1,
    /// Nonzero: first take `v mod day` (a timestamp to a time of day, in input ticks).
    day: i64 = 0,
};

fn plan(from: DataType, to: DataType) Plan {
    const day_ns: i64 = 86_400_000_000_000;
    const f = tickNs(from);
    if (from == .timestamp and (to == .date32 or to == .date64)) {
        // To a date: floor to the day (dropping the time of day is not a loss), then to ms for date64.
        return .{ .div = @divExact(day_ns, f), .floor = true, .exact_div = false, .mul = if (to == .date64) 86_400_000 else 1 };
    }
    const t = tickNs(to);
    const day: i64 = if (from == .timestamp and (to == .time32 or to == .time64)) @divExact(day_ns, f) else 0;
    if (t > f) return .{ .div = @divExact(t, f), .floor = to == .date32, .day = day };
    return .{ .mul = @divExact(f, t), .day = day };
}

fn temporal(ctx: *ExecContext, data: *const ArrayData, to: DataType, safe: bool) KernelError!ArrayRef {
    const p = plan(data.data_type, to);
    const from32 = data.data_type == .date32 or data.data_type == .time32;
    const to32 = to == .date32 or to == .time32;
    if (from32) return if (to32) temporalRun(i32, i32, ctx, data, to, p, safe) else temporalRun(i32, i64, ctx, data, to, p, safe);
    return if (to32) temporalRun(i64, i32, ctx, data, to, p, safe) else temporalRun(i64, i64, ctx, data, to, p, safe);
}

fn temporalRun(comptime F: type, comptime T: type, ctx: *ExecContext, data: *const ArrayData, to: DataType, p: Plan, safe: bool) KernelError!ArrayRef {
    const src = (PrimitiveArray(F){ .data = data }).values() catch return error.InvalidInput;
    var out = try dense.DenseBinaryOutput(T).initValidity(ctx.allocator, data.length, data, null, to);
    errdefer out.deinit();
    for (out.values(), src, 0..) |*d, x, i| {
        const v: i64 = if (p.day != 0) @mod(@as(i64, x), p.day) else x;
        var lost = false;
        var q = v;
        if (p.div != 1) {
            q = if (p.floor) @divFloor(v, p.div) else @divTrunc(v, p.div);
            lost = p.exact_div and @rem(v, p.div) != 0;
        }
        const m = @mulWithOverflow(q, p.mul);
        lost = lost or m[1] != 0;
        d.* = @truncate(m[0]);
        lost = lost or (T == i32 and std.math.cast(i32, m[0]) == null);
        if (safe and lost and out.isValid(i)) return error.InvalidCast;
    }
    return out.finish() catch error.OutOfMemory;
}
