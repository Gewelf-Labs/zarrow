//! Structural transforms (Arrow C++ "Structural transforms"): list_element, list_flatten,
//! list_parent_indices, list_slice, list_value_length, make_struct, struct_field and map_lookup.
//!
//! The list functions take list, large_list, list_view, large_list_view and fixed_size_list
//! (Arrow's kernels; map is not a list for them, as in Arrow):
//!
//! - list_value_length: the length of each row (int32; int64 for large_list / large_list_view).
//! - list_element(list, index): row item `index` (a scalar of any integer type; a null or
//!   negative index, or one past a valid row, is `InvalidInput`, Arrow's IndexError / Invalid; an
//!   index array is `NotImplemented`, as Arrow).
//! - list_flatten (vector, `ListFlattenOptions`): the items of the valid rows in row order; with
//!   `recursive`, again while the result is a list (a map stops it). A scalar gives an array.
//! - list_parent_indices (vector): for each flattened item the int64 row it came from (a chunked
//!   input counts rows from the start of the first chunk; a scalar is `NotImplemented`, as Arrow).
//! - list_slice (`ListSliceOptions`): positions `start, start + step, ... < stop` of each row,
//!   as the input's list layout or (`return_fixed_size_list`) a fixed_size_list padded with nulls.
//!
//! make_struct (`MakeStructOptions`) zips its arguments (arrays, chunked arrays or scalars, a
//! scalar repeated) into a struct; a field declared not nullable must get no null.
//! struct_field (`StructFieldOptions`) follows a path of child indices or names through struct,
//! sparse_union and dense_union levels: a null struct row, or a union row of another child, is a
//! null row of the result. map_lookup (`MapLookupOptions`) finds a key in each map row and gives
//! the item of its first or last occurrence, or a list of every occurrence (a null map, an empty
//! map or a missing key gives null); keys compare by value (bytes), floating-point keys are
//! `UnsupportedType` and the query must have the key type exactly, as Arrow.
//!
//! Every input chunk, and the child read through it, passes `validateLayout` before a buffer is
//! read. Items are gathered with the take kernels, so every child type works. zarrow has no
//! scalars of list_view, large_list_view, map or union types, so a scalar result of those types
//! is `UnsupportedType` and map_lookup takes arrays and chunked arrays only.
const std = @import("std");
const bitmap = @import("../bitmap.zig");
const core = @import("core.zig");
const datatype = @import("../datatype.zig");
const dense = @import("dense.zig");
const cast = @import("cast.zig");

const ArrayData = @import("../array/array_data.zig").ArrayData;
const StructArray = @import("../array/struct_array.zig").StructArray;
const ArrayRef = core.ArrayRef;
const ChunkedArray = core.ChunkedArray;
const DataType = core.DataType;
const Datum = core.Datum;
const ExecChunkValue = core.ExecChunkValue;
const ExecContext = core.ExecContext;
const Field = datatype.Field;
const FunctionRegistry = core.FunctionRegistry;
const KernelError = core.KernelError;
const ListRows = cast.ListRows;
const Options = core.Options;
const OwnedBuffer = @import("../buffer.zig").OwnedBuffer;
const SharedBuffer = @import("../buffer.zig").SharedBuffer;
const Scalar = core.Scalar;
const TypeTag = std.meta.Tag(DataType);

const list_tags = [_]TypeTag{ .list, .large_list, .list_view, .large_list_view, .fixed_size_list };
const record_tags = [_]TypeTag{ .struct_, .sparse_union, .dense_union };

/// Register the structural functions.
pub fn register(registry: *FunctionRegistry) KernelError!void {
    inline for (list_tags) |tag| {
        const wide = tag == .large_list or tag == .large_list_view;
        try registry.registerScalarKernel("list_value_length", .{
            .signature = .{ .arity = 1, .type_check = dense.argsOfType(tag, 1), .options_check = dense.noOptions, .result_type_fn = dense.resultTypeOf(if (wide) .{ .int64 = {} } else .{ .int32 = {} }) },
            .exec = valueLength,
        });
        try registry.registerScalarKernel("list_element", .{
            .signature = .{ .arity = 2, .type_check = elementArgs(tag), .options_check = dense.noOptions, .result_type_fn = valueTypeOfArg },
            .exec = listElement,
        });
        try registry.registerScalarKernel("list_slice", .{
            .signature = .{ .arity = 1, .type_check = dense.argsOfType(tag, 1), .options_check = sliceOptions, .result_type_fn = sliceResultType },
            .exec = listSlice,
        });
        try registry.registerVectorKernel("list_flatten", .{
            .signature = .{ .arity = 1, .type_check = dense.argsOfType(tag, 1), .options_check = flattenOptions, .result_type_fn = flattenResultType },
            .exec = listFlatten,
        });
        try registry.registerVectorKernel("list_parent_indices", .{
            .signature = .{ .arity = 1, .type_check = dense.argsOfType(tag, 1), .options_check = dense.noOptions, .result_type_fn = dense.resultTypeOf(.{ .int64 = {} }) },
            .exec = parentIndices,
        });
    }
    try registry.registerScalarKernel("make_struct", .{
        .signature = .{ .arity = 0, .variadic = true, .options_check = makeStructOptions, .result_type_alloc_fn = makeStructType },
        .exec = makeStruct,
    });
    inline for (record_tags) |tag| {
        try registry.registerScalarKernel("struct_field", .{
            .signature = .{ .arity = 1, .type_check = dense.argsOfType(tag, 1), .options_check = fieldOptions, .result_type_fn = fieldResultType },
            .exec = structField,
        });
    }
    try registry.registerScalarKernel("map_lookup", .{
        .signature = .{ .arity = 1, .type_check = dense.argsOfType(.map, 1), .options_check = lookupOptions, .result_type_fn = lookupResultType },
        .exec = mapLookup,
    });
}

// ------------------------------------------------------------------------------ helpers

/// The item field of a list-like type.
fn valueField(dt: DataType) KernelError!Field {
    return switch (dt) {
        .list, .large_list => |l| l.value_field,
        .list_view, .large_list_view => |l| l.value_field,
        .fixed_size_list => |l| l.value_field,
        else => error.UnsupportedType,
    };
}

fn isList(dt: DataType) bool {
    return switch (dt) {
        .list, .large_list, .list_view, .large_list_view, .fixed_size_list => true,
        else => false,
    };
}

/// The array of an exec chunk value and its single child (both passed `validateLayout`). The
/// caller releases the array.
fn listOperand(allocator: std.mem.Allocator, v: ExecChunkValue) KernelError!ArrayRef {
    var arr = try dense.chunkArray(allocator, v);
    errdefer arr.release();
    try checkList(arr.data());
    return arr;
}

fn checkList(d: *const ArrayData) KernelError!void {
    d.validateLayout() catch return error.InvalidInput;
    if (d.children.len != 1) return error.InvalidInput;
    // validateLayout of a list bounds its offsets by the child length; the child's own buffers
    // are checked here.
    d.children[0].data().validateLayout() catch return error.InvalidInput;
}

/// Rows `picks` of `src` (a null pick gives a null row) with the take kernels.
fn take(src: ArrayRef, picks: anytype) KernelError!ArrayRef {
    if (picks.len == 0) return src.slice(0, 0) catch error.OutOfMemory;
    const nullable = comptime @typeInfo(std.meta.Elem(@TypeOf(picks))) == .optional;
    var out = if (nullable) try core.datumTakeNullable(.{ .array = src }, picks) else try core.datumTake(.{ .array = src }, picks);
    if (out != .array) {
        out.release();
        return error.InvalidInput;
    }
    return out.array;
}

fn chunkedOf(allocator: std.mem.Allocator, dt: DataType, outs: []const ArrayRef) KernelError!Datum {
    const chunked = ChunkedArray.init(allocator, dt, outs) catch |err| return switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        else => error.InvalidInput,
    };
    return .{ .chunked = chunked };
}

fn releaseAll(allocator: std.mem.Allocator, outs: *std.ArrayList(ArrayRef)) void {
    for (outs.items) |*o| o.release();
    outs.deinit(allocator);
}

fn validityOf(allocator: std.mem.Allocator, d: *const ArrayData) KernelError!?OwnedBuffer {
    return dense.combinedValidity(allocator, d.length, &.{d}) catch |err| switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        else => error.InvalidInput,
    };
}

// ------------------------------------------------------------------------------ list_value_length

fn valueLength(ctx: *ExecContext, args: []const Datum, options: Options) KernelError!Datum {
    const dt = args[0].dataType();
    const wide = dt == .large_list or dt == .large_list_view;
    const out_type: DataType = if (wide) .{ .int64 = {} } else .{ .int32 = {} };
    return dense.execElementwiseWith(ctx, args, options, out_type, true, valueLengthChunk);
}

fn valueLengthChunk(ctx: *ExecContext, values: []const ExecChunkValue, len: usize, _: Options) KernelError!ArrayRef {
    var arr = try dense.chunkArray(ctx.allocator, values[0]);
    defer arr.release();
    const d = arr.data();
    d.validateLayout() catch return error.InvalidInput;
    const rows = try ListRows.of(d);
    const wide = d.data_type == .large_list or d.data_type == .large_list_view;
    const out_type: DataType = if (wide) .{ .int64 = {} } else .{ .int32 = {} };
    var buf = OwnedBuffer.initUninit(ctx.allocator, len * @as(usize, if (wide) 8 else 4)) catch return error.OutOfMemory;
    if (wide) fillLengths(i64, buf.typedSlice(i64)[0..len], rows) else fillLengths(i32, buf.typedSlice(i32)[0..len], rows);
    const validity = validityOf(ctx.allocator, d) catch |err| {
        buf.deinit();
        return err;
    };
    return dense.finishFixed(ctx.allocator, out_type, len, &buf, validity);
}

/// Row lengths (a null row's slot is hidden by the validity bitmap).
fn fillLengths(comptime O: type, out: []O, rows: ListRows) void {
    switch (rows) {
        inline .off32, .off64 => |o| for (out, o[0..out.len], o[1 .. out.len + 1]) |*v, lo, hi| {
            v.* = @intCast(hi - lo);
        },
        inline .view32, .view64 => |vw| for (out, vw.s[0..out.len]) |*v, s| {
            v.* = @intCast(s);
        },
        .fixed => |f| @memset(out, @intCast(f.size)),
    }
}

// ------------------------------------------------------------------------------ list_element

fn elementArgs(comptime tag: TypeTag) core.TypeCheckFn {
    return &struct {
        fn check(args: []const Datum) bool {
            return args.len == 2 and std.meta.activeTag(args[0].dataType()) == tag and isInteger(args[1].dataType());
        }
    }.check;
}

fn isInteger(dt: DataType) bool {
    return switch (dt) {
        .int8, .int16, .int32, .int64, .uint8, .uint16, .uint32, .uint64 => true,
        else => false,
    };
}

fn valueTypeOfArg(args: []const Datum, _: Options) KernelError!DataType {
    return (try valueField(args[0].dataType())).data_type.*;
}

/// The index scalar of list_element: not null, not negative (Arrow's errors).
fn indexOf(s: Scalar) KernelError!u64 {
    return switch (s.value) {
        inline .i8, .i16, .i32, .i64, .u8, .u16, .u32, .u64 => |x| if (x < 0) error.InvalidInput else @intCast(x),
        else => error.InvalidInput,
    };
}

fn listElement(ctx: *ExecContext, args: []const Datum, options: Options) KernelError!Datum {
    // Arrow: "list_element not yet implemented for arrays of list indices".
    if (args[1] != .scalar) return error.NotImplemented;
    _ = try indexOf(args[1].scalar);
    return dense.execElementwiseWith(ctx, args, options, try valueTypeOfArg(args, options), false, elementChunk);
}

fn elementChunk(ctx: *ExecContext, values: []const ExecChunkValue, len: usize, _: Options) KernelError!ArrayRef {
    const index = try indexOf(values[1].scalar);
    var arr = try listOperand(ctx.allocator, values[0]);
    defer arr.release();
    const d = arr.data();
    const rows = try ListRows.of(d);
    const picks = ctx.allocator.alloc(?usize, len) catch return error.OutOfMemory;
    defer ctx.allocator.free(picks);
    for (picks, 0..) |*p, i| {
        if (d.isNull(i)) {
            p.* = null;
            continue;
        }
        const sp = rows.span(i);
        if (index >= sp[1]) return error.InvalidInput;
        p.* = sp[0] + @as(usize, @intCast(index));
    }
    return take(d.children[0], picks);
}

// ------------------------------------------------------------------------------ list_flatten

fn flattenOptions(options: Options) bool {
    return options == .none or options == .list_flatten;
}

fn recursiveOf(options: Options) bool {
    return options == .list_flatten and options.list_flatten.recursive;
}

fn flattenedType(dt: DataType, recursive: bool) KernelError!DataType {
    var t = (try valueField(dt)).data_type.*;
    while (recursive and isList(t)) t = (try valueField(t)).data_type.*;
    return t;
}

fn flattenResultType(args: []const Datum, options: Options) KernelError!DataType {
    return flattenedType(args[0].dataType(), recursiveOf(options));
}

fn listFlatten(ctx: *ExecContext, args: []const Datum, options: Options) KernelError!Datum {
    const recursive = recursiveOf(options);
    switch (args[0]) {
        .array => |a| return .{ .array = try flattenAll(ctx.allocator, a, recursive) },
        // A list scalar flattens to an array of its items (Arrow).
        .scalar => |s| {
            var a = try dense.chunkArray(ctx.allocator, .{ .scalar = s });
            defer a.release();
            return .{ .array = try flattenAll(ctx.allocator, a, recursive) };
        },
        .chunked => |c| {
            var outs: std.ArrayList(ArrayRef) = .empty;
            defer releaseAll(ctx.allocator, &outs);
            for (c.chunks()) |chunk| {
                var out = try flattenAll(ctx.allocator, chunk, recursive);
                outs.append(ctx.allocator, out) catch {
                    out.release();
                    return error.OutOfMemory;
                };
            }
            return chunkedOf(ctx.allocator, try flattenedType(c.dataType(), recursive), outs.items);
        },
        else => return error.InvalidInput,
    }
}

fn flattenAll(allocator: std.mem.Allocator, a: ArrayRef, recursive: bool) KernelError!ArrayRef {
    var cur = try flattenOnce(allocator, a);
    while (recursive and isList(cur.data().data_type)) {
        const next = flattenOnce(allocator, cur) catch |err| {
            cur.release();
            return err;
        };
        cur.release();
        cur = next;
    }
    return cur;
}

/// The items of the valid rows of one list array: a slice of the child when they are stored back
/// to back in row order, else gathered.
fn flattenOnce(allocator: std.mem.Allocator, a: ArrayRef) KernelError!ArrayRef {
    const d = a.data();
    try checkList(d);
    const child = d.children[0];
    const rows = try ListRows.of(d);
    const n = d.length;
    contiguous: {
        const w = rows.contiguous(n) orelse break :contiguous;
        if (d.hasNulls()) {
            // A null row must hold no items (fixed_size_list null rows always do).
            if (rows == .fixed) break :contiguous;
            for (0..n) |i| if (d.isNull(i) and rows.span(i)[1] != 0) break :contiguous;
        }
        return child.slice(w[0], w[1] - w[0]) catch error.OutOfMemory;
    }
    var total: usize = 0;
    for (0..n) |i| {
        if (!d.isNull(i)) total += rows.span(i)[1];
    }
    const picks = allocator.alloc(usize, total) catch return error.OutOfMemory;
    defer allocator.free(picks);
    var at: usize = 0;
    for (0..n) |i| {
        if (d.isNull(i)) continue;
        const sp = rows.span(i);
        for (picks[at..][0..sp[1]], sp[0]..) |*p, r| p.* = r;
        at += sp[1];
    }
    return take(child, @as([]const usize, picks));
}

// ------------------------------------------------------------------------------ list_parent_indices

fn parentIndices(ctx: *ExecContext, args: []const Datum, _: Options) KernelError!Datum {
    switch (args[0]) {
        .array => |a| return .{ .array = try parentsOf(ctx.allocator, a, 0) },
        .chunked => |c| {
            var outs: std.ArrayList(ArrayRef) = .empty;
            defer releaseAll(ctx.allocator, &outs);
            var base: usize = 0;
            for (c.chunks()) |chunk| {
                var out = try parentsOf(ctx.allocator, chunk, base);
                outs.append(ctx.allocator, out) catch {
                    out.release();
                    return error.OutOfMemory;
                };
                base += chunk.data().length;
            }
            return chunkedOf(ctx.allocator, .{ .int64 = {} }, outs.items);
        },
        // Arrow: "Unsupported input type for function 'list_parent_indices'".
        else => return error.NotImplemented,
    }
}

fn parentsOf(allocator: std.mem.Allocator, a: ArrayRef, base: usize) KernelError!ArrayRef {
    const d = a.data();
    try checkList(d);
    const rows = try ListRows.of(d);
    var total: usize = 0;
    for (0..d.length) |i| {
        if (!d.isNull(i)) total += rows.span(i)[1];
    }
    var buf = OwnedBuffer.initUninit(allocator, total * 8) catch return error.OutOfMemory;
    const out = buf.typedSlice(i64)[0..total];
    var at: usize = 0;
    for (0..d.length) |i| {
        if (d.isNull(i)) continue;
        const n = rows.span(i)[1];
        @memset(out[at..][0..n], @intCast(base + i));
        at += n;
    }
    return dense.finishFixed(allocator, .{ .int64 = {} }, total, &buf, null);
}

// ------------------------------------------------------------------------------ list_slice

/// Arrow's checks: `start >= 0`, `stop` not below `start`, `step >= 1`.
fn sliceOptions(options: Options) bool {
    if (options != .list_slice) return false;
    const o = options.list_slice;
    if (o.start < 0 or o.step < 1) return false;
    if (o.stop) |stop| if (stop < o.start) return false;
    return true;
}

fn sliceResultType(args: []const Datum, options: Options) KernelError!DataType {
    return sliceType(args[0].dataType(), options.list_slice);
}

/// The result type of list_slice: a fixed_size_list of `ceil((stop - start) / step)` items, or
/// the input's list layout (list for a fixed_size_list input).
fn sliceType(dt: DataType, o: core.ListSliceOptions) KernelError!DataType {
    const field = try valueField(dt);
    if (o.return_fixed_size_list orelse (dt == .fixed_size_list)) {
        // Arrow: "Unable to produce FixedSizeListArray from non-FixedSizeListArray without `stop`".
        const stop = o.stop orelse if (dt == .fixed_size_list) dt.fixed_size_list.list_size else return error.InvalidOptions;
        const span: u64 = @intCast(@max(stop - o.start, 0));
        const step: u64 = @intCast(o.step);
        const size = std.math.cast(i32, span / step + @intFromBool(span % step != 0)) orelse return error.InvalidOptions;
        return .{ .fixed_size_list = .{ .value_field = field, .list_size = size } };
    }
    return if (dt == .fixed_size_list) .{ .list = .{ .value_field = field } } else dt;
}

fn listSlice(ctx: *ExecContext, args: []const Datum, options: Options) KernelError!Datum {
    return dense.execElementwiseWith(ctx, args, options, try sliceType(args[0].dataType(), options.list_slice), false, sliceChunk);
}

fn sliceChunk(ctx: *ExecContext, values: []const ExecChunkValue, len: usize, options: Options) KernelError!ArrayRef {
    const allocator = ctx.allocator;
    const o = options.list_slice;
    var arr = try listOperand(allocator, values[0]);
    defer arr.release();
    const d = arr.data();
    const rows = try ListRows.of(d);
    const out_type = try sliceType(d.data_type, o);
    const start: u64 = @intCast(o.start);
    const step: u64 = @intCast(o.step);
    const stop: u64 = if (o.stop) |s| @intCast(s) else std.math.maxInt(u64);
    var picks: std.ArrayList(?usize) = .empty;
    defer picks.deinit(allocator);

    if (out_type == .fixed_size_list) {
        const size: usize = @intCast(out_type.fixed_size_list.list_size);
        picks.ensureTotalCapacity(allocator, std.math.mul(usize, size, len) catch return error.Overflow) catch return error.OutOfMemory;
        for (0..len) |i| {
            if (d.isNull(i)) {
                picks.appendNTimesAssumeCapacity(null, size);
                continue;
            }
            const sp = rows.span(i);
            for (0..size) |k| {
                const pos = start +| (k *| step);
                picks.appendAssumeCapacity(if (pos < sp[1]) sp[0] + @as(usize, @intCast(pos)) else null);
            }
        }
        var child = try take(d.children[0], picks.items);
        const validity = validityOf(allocator, d) catch |err| {
            child.release();
            return err;
        };
        return cast.nestedArray(allocator, out_type, len, validity, &.{}, &.{}, child);
    }

    const wide = out_type == .large_list or out_type == .large_list_view;
    const view = out_type == .list_view or out_type == .large_list_view;
    const ow: usize = if (wide) 8 else 4;
    var offsets = OwnedBuffer.init(allocator, (len + 1) * ow) catch return error.OutOfMemory;
    var sizes = OwnedBuffer.init(allocator, if (view) len * ow else 0) catch {
        offsets.deinit();
        return error.OutOfMemory;
    };
    var buffers_owned = true;
    defer if (buffers_owned) {
        offsets.deinit();
        sizes.deinit();
    };
    for (0..len + 1) |i| {
        const at = picks.items.len;
        if (wide) offsets.typedSlice(i64)[i] = @intCast(at) else offsets.typedSlice(i32)[i] = std.math.cast(i32, at) orelse return error.Overflow;
        if (i == len) break;
        if (!d.isNull(i)) {
            const sp = rows.span(i);
            const end = @min(@as(u64, sp[1]), stop);
            var pos = start;
            while (pos < end) : (pos +|= step) picks.append(allocator, sp[0] + @as(usize, @intCast(pos))) catch return error.OutOfMemory;
        }
        if (view) {
            const n = picks.items.len - at;
            if (wide) sizes.typedSlice(i64)[i] = @intCast(n) else sizes.typedSlice(i32)[i] = @intCast(n);
        }
    }
    var child = try take(d.children[0], picks.items);
    const validity = validityOf(allocator, d) catch |err| {
        child.release();
        return err;
    };
    buffers_owned = false;
    if (view) return cast.nestedArray(allocator, out_type, len, validity, &.{ offsets, sizes }, &.{ len * ow, len * ow }, child);
    sizes.deinit();
    return cast.nestedArray(allocator, out_type, len, validity, &.{offsets}, &.{(len + 1) * ow}, child);
}

// ------------------------------------------------------------------------------ make_struct

fn makeStructOptions(options: Options) bool {
    return options == .none or options == .make_struct;
}

/// The field of argument `i` (`names` holds the default names, "0", "1", ...).
fn fieldOf(o: core.MakeStructOptions, i: usize, names: []const u8, dt: *const DataType) Field {
    return .{
        .name = if (o.field_names.len > 0) o.field_names[i] else defaultName(names, i),
        .data_type = dt,
        .nullable = if (o.field_nullability.len > 0) o.field_nullability[i] else true,
        .metadata = if (o.field_metadata.len > 0) o.field_metadata[i] else null,
    };
}

/// pyarrow's default field names: "0", "1", ..., stored 20 bytes apart in `names`.
fn defaultNames(allocator: std.mem.Allocator, n: usize) KernelError![]u8 {
    const buf = allocator.alloc(u8, n * 20) catch return error.OutOfMemory;
    for (0..n) |i| _ = std.fmt.bufPrint(buf[i * 20 ..][0..20], "{d}", .{i}) catch unreachable;
    return buf;
}

fn defaultName(names: []const u8, i: usize) []const u8 {
    const slot = names[i * 20 ..][0..20];
    var n: usize = 1;
    while (n < 20 and std.ascii.isDigit(slot[n])) n += 1;
    return slot[0..n];
}

/// One entry per argument, or none (Arrow: "make_struct() was passed N arguments but ...").
fn makeStructOf(args_len: usize, options: Options) KernelError!core.MakeStructOptions {
    const o: core.MakeStructOptions = if (options == .make_struct) options.make_struct else .{};
    for ([_]usize{ o.field_names.len, o.field_nullability.len, o.field_metadata.len }) |n| {
        if (n != 0 and n != args_len) return error.InvalidOptions;
    }
    return o;
}

fn makeStructType(allocator: std.mem.Allocator, args: []const Datum, options: Options) KernelError!DataType {
    const o = try makeStructOf(args.len, options);
    const names = try defaultNames(allocator, args.len);
    const types = allocator.alloc(DataType, args.len) catch return error.OutOfMemory;
    const fields = allocator.alloc(Field, args.len) catch return error.OutOfMemory;
    for (fields, types, args, 0..) |*f, *t, a, i| {
        t.* = a.dataType();
        f.* = fieldOf(o, i, names, t);
    }
    return .{ .struct_ = .{ .fields = fields } };
}

fn makeStruct(ctx: *ExecContext, args: []const Datum, options: Options) KernelError!Datum {
    const allocator = ctx.allocator;
    const o = try makeStructOf(args.len, options);
    const names = try defaultNames(allocator, args.len);
    defer allocator.free(names);
    const types = allocator.alloc(DataType, args.len) catch return error.OutOfMemory;
    defer allocator.free(types);
    const fields = allocator.alloc(Field, args.len) catch return error.OutOfMemory;
    defer allocator.free(fields);
    var any_chunked = false;
    var all_scalar = true;
    for (fields, types, args, 0..) |*f, *t, a, i| {
        t.* = a.dataType();
        f.* = fieldOf(o, i, names, t);
        if (a == .chunked) any_chunked = true;
        if (a != .scalar) all_scalar = false;
        if (a.isTabular()) return error.InvalidInput;
    }
    // No argument: an empty struct<> (Arrow).
    if (args.len == 0) return .{ .array = try dense.structArrayFields(allocator, &.{}, &.{}, 0, null) };
    const len = try core.inferNaryExecLen(args);
    if (!any_chunked) {
        const values = allocator.alloc(ExecChunkValue, args.len) catch return error.OutOfMemory;
        defer allocator.free(values);
        for (values, args) |*v, a| v.* = switch (a) {
            .array => |arr| .{ .array = arr },
            .scalar => |s| .{ .scalar = s },
            else => unreachable,
        };
        var out = try structChunk(allocator, fields, values, len);
        if (!all_scalar) return .{ .array = out };
        defer out.release();
        return .{ .scalar = try dense.scalarAt0(out) };
    }
    var it = try core.NaryExecChunkIterator.init(allocator, args);
    defer it.deinit();
    var outs: std.ArrayList(ArrayRef) = .empty;
    defer releaseAll(allocator, &outs);
    while (try it.next()) |next| {
        var chunk = next;
        defer chunk.deinit();
        var out = try structChunk(allocator, fields, chunk.values, chunk.len);
        outs.append(allocator, out) catch {
            out.release();
            return error.OutOfMemory;
        };
    }
    if (outs.items.len == 0) {
        // No chunk: the type still needs an owner, so give the result one empty chunk.
        const empty = allocator.alloc(ExecChunkValue, args.len) catch return error.OutOfMemory;
        defer allocator.free(empty);
        for (empty, types) |*v, t| v.* = .{ .scalar = Scalar.init(t, .null) };
        try outs.append(allocator, try structChunk(allocator, fields, empty, 0));
    }
    return chunkedOf(allocator, outs.items[0].data().data_type, outs.items);
}

/// The struct of `len` rows over the arguments of one chunk (a scalar repeated).
fn structChunk(allocator: std.mem.Allocator, fields: []const Field, values: []const ExecChunkValue, len: usize) KernelError!ArrayRef {
    const kids = allocator.alloc(ArrayRef, values.len) catch return error.OutOfMemory;
    defer allocator.free(kids);
    var made: usize = 0;
    errdefer for (kids[0..made]) |*k| k.release();
    for (kids, values, fields) |*k, v, f| {
        k.* = switch (v) {
            .array => |a| blk: {
                a.data().validateLayout() catch return error.InvalidInput;
                break :blk a.retain();
            },
            .scalar => |s| try repeatScalar(allocator, s, len),
        };
        made += 1;
        // Arrow: "Output field ... does not allow nulls but the corresponding argument was not
        // entirely valid".
        if (!f.nullable and k.data().hasNulls()) return error.InvalidInput;
    }
    made = 0;
    return dense.structArrayFields(allocator, fields, kids, len, null);
}

/// `len` copies of a scalar.
fn repeatScalar(allocator: std.mem.Allocator, s: Scalar, len: usize) KernelError!ArrayRef {
    var one = if (s.isNull() and s.payload == null) try dense.nullArray(allocator, s.data_type, 1) else try dense.chunkArray(allocator, .{ .scalar = s });
    defer one.release();
    if (len == 1) return one.retain();
    const zeros = allocator.alloc(usize, len) catch return error.OutOfMemory;
    defer allocator.free(zeros);
    @memset(zeros, 0);
    return take(one, @as([]const usize, zeros));
}

// ------------------------------------------------------------------------------ struct_field

fn fieldOptions(options: Options) bool {
    return options == .struct_field;
}

fn pathLen(o: core.StructFieldOptions) usize {
    return if (o.names.len > 0) o.names.len else o.indices.len;
}

/// The child at level `level` of the path in a struct or union type (Arrow: "out-of-bounds field
/// reference", "No match for FieldRef", "Multiple matches for FieldRef").
fn childIndex(dt: DataType, o: core.StructFieldOptions, level: usize) KernelError!usize {
    const fields = switch (datatype.storageDataType(dt)) {
        .struct_ => |st| st.fields,
        .sparse_union, .dense_union => |u| u.fields,
        else => return error.InvalidOptions,
    };
    if (o.names.len > 0) {
        var found: ?usize = null;
        for (fields, 0..) |f, k| if (std.mem.eql(u8, f.name, o.names[level])) {
            if (found != null) return error.InvalidOptions;
            found = k;
        };
        return found orelse error.InvalidOptions;
    }
    const idx = o.indices[level];
    if (idx < 0 or idx >= fields.len) return error.InvalidOptions;
    return @intCast(idx);
}

fn fieldType(dt: DataType, o: core.StructFieldOptions) KernelError!DataType {
    var t = dt;
    for (0..pathLen(o)) |level| {
        const k = try childIndex(t, o, level);
        t = switch (datatype.storageDataType(t)) {
            .struct_ => |st| st.fields[k].data_type.*,
            .sparse_union, .dense_union => |u| u.fields[k].data_type.*,
            else => unreachable,
        };
    }
    return t;
}

fn fieldResultType(args: []const Datum, options: Options) KernelError!DataType {
    return fieldType(args[0].dataType(), options.struct_field);
}

fn structField(ctx: *ExecContext, args: []const Datum, options: Options) KernelError!Datum {
    return dense.execElementwiseWith(ctx, args, options, try fieldType(args[0].dataType(), options.struct_field), false, fieldChunk);
}

fn fieldChunk(ctx: *ExecContext, values: []const ExecChunkValue, _: usize, options: Options) KernelError!ArrayRef {
    const allocator = ctx.allocator;
    const o = options.struct_field;
    var cur = try dense.chunkArray(allocator, values[0]);
    errdefer cur.release();
    for (0..pathLen(o)) |level| {
        const d = cur.data();
        d.validateLayout() catch return error.InvalidInput;
        const k = try childIndex(d.data_type, o, level);
        if (k >= d.children.len) return error.InvalidInput;
        const next = switch (datatype.storageDataType(d.data_type)) {
            .struct_ => try structChild(allocator, d, k),
            .sparse_union => |u| try sparseChild(allocator, d, u.type_ids[k], k),
            .dense_union => |u| try denseChild(allocator, d, u.type_ids[k], k),
            else => unreachable,
        };
        cur.release();
        cur = next;
    }
    return cur;
}

/// Field `k` of a struct array, null where the struct row is null.
fn structChild(allocator: std.mem.Allocator, d: *const ArrayData, k: usize) KernelError!ArrayRef {
    var child = (StructArray{ .data = d }).field(k) catch return error.InvalidInput;
    defer child.release();
    if (!d.hasNulls()) return child.retain();
    const keep = try bitsBuffer(allocator, d.length);
    defer allocator.free(keep);
    bitmap.copyBits(keep, 0, d.buffers[0].data, d.offset, d.length);
    return maskRows(allocator, child, keep);
}

/// Child `k` of a sparse union, null where a row holds another child.
fn sparseChild(allocator: std.mem.Allocator, d: *const ArrayData, code: i8, k: usize) KernelError!ArrayRef {
    const ids = (d.buffers[0].typedSlice(i8) catch return error.InvalidInput)[d.offset..][0..d.length];
    const src = d.children[k];
    if (src.data().length < d.offset + d.length) return error.InvalidInput;
    var child = src.slice(d.offset, d.length) catch return error.OutOfMemory;
    defer child.release();
    const keep = try bitsBuffer(allocator, d.length);
    defer allocator.free(keep);
    var all = true;
    for (ids, 0..) |id, i| {
        if (id == code) bitmap.setBit(keep, i) else all = false;
    }
    if (all) return child.retain();
    return maskRows(allocator, child, keep);
}

/// Child `k` of a dense union gathered row by row, null where a row holds another child.
fn denseChild(allocator: std.mem.Allocator, d: *const ArrayData, code: i8, k: usize) KernelError!ArrayRef {
    const ids = (d.buffers[0].typedSlice(i8) catch return error.InvalidInput)[d.offset..][0..d.length];
    const offs = (d.buffers[1].typedSlice(i32) catch return error.InvalidInput)[d.offset..][0..d.length];
    const picks = allocator.alloc(?usize, d.length) catch return error.OutOfMemory;
    defer allocator.free(picks);
    // validateLayout bounds the offset of every row by the length of its child.
    for (picks, ids, offs) |*p, id, off| p.* = if (id == code) @intCast(off) else null;
    return take(d.children[k], picks);
}

fn bitsBuffer(allocator: std.mem.Allocator, len: usize) KernelError![]u8 {
    const b = allocator.alloc(u8, bitmap.byteLength(len)) catch return error.OutOfMemory;
    @memset(b, 0);
    return b;
}

/// `arr` with row `i` made null where bit `i` of `keep` is clear: the validity bitmap ANDed
/// (buffers and children shared), or a gather for the layouts without one.
fn maskRows(allocator: std.mem.Allocator, arr: ArrayRef, keep: []const u8) KernelError!ArrayRef {
    const d = arr.data();
    const n = d.length;
    switch (datatype.storageDataType(d.data_type)) {
        .null => return arr.retain(),
        .sparse_union, .dense_union, .run_end_encoded => {
            const picks = allocator.alloc(?usize, n) catch return error.OutOfMemory;
            defer allocator.free(picks);
            for (picks, 0..) |*p, i| p.* = if (bitmap.bitIsSet(keep, i)) i else null;
            return take(arr, picks);
        },
        else => {},
    }
    if (d.buffers.len == 0) return error.InvalidInput;
    const end = d.offset + n;
    const bytes = bitmap.byteLength(end);
    var bits = OwnedBuffer.init(allocator, bytes) catch return error.OutOfMemory;
    bitmap.copyBits(bits.data, d.offset, keep, 0, n);
    if (d.validity()) |v| {
        for (bits.data[d.offset / 8 .. bytes], v.data[d.offset / 8 .. bytes]) |*b, x| b.* &= x;
    }
    const nulls = bitmap.countUnsetBitsInRange(bits.data, d.offset, end);
    var shared = bits.toShared(bytes) catch {
        bits.deinit();
        return error.OutOfMemory;
    };
    // A slice is a new node sharing every buffer, child and owner of `arr`; its validity buffer
    // is replaced.
    var out = arr.slice(0, n) catch {
        shared.release();
        return error.OutOfMemory;
    };
    const buffers: []SharedBuffer = @constCast(out.node.data.buffers);
    buffers[0].release();
    buffers[0] = shared;
    out.node.data.null_count = nulls;
    return out;
}

// ------------------------------------------------------------------------------ map_lookup

/// A non-null query key (Arrow: "map_lookup: query_key can't be null").
fn lookupOptions(options: Options) bool {
    return options == .map_lookup and !options.map_lookup.query_key.isNull();
}

/// The item type (first / last) or a list of it (all). The query must have the key type and
/// the key type must not be floating point (Arrow's TypeErrors).
fn lookupType(dt: DataType, o: core.MapLookupOptions) KernelError!DataType {
    if (dt != .map) return error.UnsupportedType;
    const m = dt.map;
    const key = m.key_field.data_type.*;
    if (!key.eql(o.query_key.data_type)) return error.UnsupportedType;
    if (!comparableKey(key)) return error.UnsupportedType;
    if (o.occurrence != .all) return m.item_field.data_type.*;
    return .{ .list = .{ .value_field = .{ .name = "item", .data_type = m.item_field.data_type } } };
}

fn comparableKey(dt: DataType) bool {
    return switch (dt) {
        .bool, .int8, .int16, .int32, .int64, .uint8, .uint16, .uint32, .uint64 => true,
        .date32, .date64, .timestamp, .time32, .time64, .duration => true,
        .interval_months, .interval_day_time, .interval_month_day_nano => true,
        .decimal32, .decimal64, .decimal128, .decimal256, .fixed_size_binary => true,
        .string, .large_string, .string_view, .binary, .large_binary, .binary_view => true,
        else => false,
    };
}

fn lookupResultType(args: []const Datum, options: Options) KernelError!DataType {
    return lookupType(args[0].dataType(), options.map_lookup);
}

fn mapLookup(ctx: *ExecContext, args: []const Datum, options: Options) KernelError!Datum {
    return dense.execElementwiseWith(ctx, args, options, try lookupType(args[0].dataType(), options.map_lookup), false, lookupChunk);
}

/// The bytes of one key row: a fixed-width value, a bool as one byte, or string / binary bytes.
const Keys = union(enum) {
    fixed: struct { bytes: []const u8, width: usize, base: usize },
    bits: struct { data: []const u8, base: usize },
    var_bytes: dense.StringOperand,

    fn init(arr: ArrayRef) KernelError!Keys {
        const d = arr.data();
        return switch (d.data_type) {
            .bool => .{ .bits = .{ .data = d.buffers[1].data, .base = d.offset } },
            .string, .large_string, .string_view, .binary, .large_binary, .binary_view => .{ .var_bytes = try dense.StringOperand.init(.{ .array = arr }) },
            .fixed_size_binary => |f| .{ .fixed = .{ .bytes = d.buffers[1].data, .width = std.math.cast(usize, f.byte_width) orelse return error.InvalidInput, .base = d.offset } },
            else => .{ .fixed = .{ .bytes = d.buffers[1].data, .width = (d.data_type.bitWidth() orelse return error.UnsupportedType) / 8, .base = d.offset } },
        };
    }

    inline fn get(self: Keys, i: usize) []const u8 {
        return switch (self) {
            .fixed => |f| f.bytes[(f.base + i) * f.width ..][0..f.width],
            .bits => |b| if (bitmap.bitIsSet(b.data, b.base + i)) "\x01" else "\x00",
            .var_bytes => |op| switch (op) {
                inline else => |r| r.get(i),
            },
        };
    }
};

/// The bytes of the query scalar, as `Keys.get` reads a key row of its type.
fn queryBytes(s: Scalar, buf: *[32]u8) KernelError![]const u8 {
    return switch (s.value) {
        .string, .binary => |b| b,
        .bool => |b| if (b) "\x01" else "\x00",
        inline .i8, .i16, .i32, .i64, .u8, .u16, .u32, .u64, .date32, .date64, .time32, .time64, .timestamp, .duration, .interval_months, .interval_day_time, .interval_month_day_nano, .decimal32, .decimal64, .decimal128, .decimal256 => |x| blk: {
            const bytes = std.mem.asBytes(&x);
            @memcpy(buf[0..bytes.len], bytes);
            break :blk buf[0..bytes.len];
        },
        else => error.UnsupportedType,
    };
}

fn lookupChunk(ctx: *ExecContext, values: []const ExecChunkValue, len: usize, options: Options) KernelError!ArrayRef {
    const allocator = ctx.allocator;
    const o = options.map_lookup;
    var arr = try listOperand(allocator, values[0]);
    defer arr.release();
    const d = arr.data();
    const entries = d.children[0].data();
    if (entries.children.len != 2) return error.InvalidInput;
    var keys = (StructArray{ .data = entries }).field(0) catch return error.InvalidInput;
    defer keys.release();
    var items = (StructArray{ .data = entries }).field(1) catch return error.InvalidInput;
    defer items.release();
    keys.data().validateLayout() catch return error.InvalidInput;
    items.data().validateLayout() catch return error.InvalidInput;
    if (!keys.data().data_type.eql(o.query_key.data_type)) return error.UnsupportedType;
    var qbuf: [32]u8 = undefined;
    const query = try queryBytes(o.query_key, &qbuf);
    const k = try Keys.init(keys);
    if (k == .fixed and k.fixed.width != query.len) return error.InvalidInput;
    const rows = try ListRows.of(d);

    if (o.occurrence != .all) {
        const picks = allocator.alloc(?usize, len) catch return error.OutOfMemory;
        defer allocator.free(picks);
        for (picks, 0..) |*p, i| {
            p.* = null;
            if (d.isNull(i)) continue;
            const sp = rows.span(i);
            for (0..sp[1]) |j| {
                const r = if (o.occurrence == .first) sp[0] + j else sp[0] + sp[1] - 1 - j;
                if (std.mem.eql(u8, k.get(r), query)) {
                    p.* = r;
                    break;
                }
            }
        }
        return take(items, picks);
    }

    // all: a list of the items of every match; a row without one is null.
    var matches: std.ArrayList(usize) = .empty;
    defer matches.deinit(allocator);
    var offsets = OwnedBuffer.init(allocator, (len + 1) * 4) catch return error.OutOfMemory;
    var validity = OwnedBuffer.init(allocator, bitmap.byteLength(len)) catch {
        offsets.deinit();
        return error.OutOfMemory;
    };
    var owned = true;
    defer if (owned) {
        offsets.deinit();
        validity.deinit();
    };
    const offs = offsets.typedSlice(i32);
    offs[0] = 0;
    for (0..len) |i| {
        if (!d.isNull(i)) {
            const sp = rows.span(i);
            const before = matches.items.len;
            for (sp[0]..sp[0] + sp[1]) |r| {
                if (std.mem.eql(u8, k.get(r), query)) matches.append(allocator, r) catch return error.OutOfMemory;
            }
            if (matches.items.len > before) bitmap.setBit(validity.data, i);
        }
        offs[i + 1] = std.math.cast(i32, matches.items.len) orelse return error.Overflow;
    }
    const out_type = try lookupType(d.data_type, o);
    const child = try take(items, @as([]const usize, matches.items));
    owned = false;
    return cast.nestedArray(allocator, out_type, len, validity, &.{offsets}, &.{(len + 1) * 4}, child);
}
