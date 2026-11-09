//! Selecting, multiplexing and null replacement (Arrow C++ "Selecting / multiplexing", "Replace
//! functions", "Selections"): coalesce, if_else, case_when, choose (scalar functions),
//! drop_null, fill_null_forward, fill_null_backward, replace_with_mask, scatter,
//! inverse_permutation and indices_nonzero (vector functions).
//!
//! - coalesce(v0, v1, ...): per row the first valid value (null when none is).
//! - if_else(cond, a, b): `a` where cond is true, `b` where false, null where cond is null (a
//!   null-type cond is all null).
//! - case_when(cond, v0, ..., vn[, else]): `cond` is a struct of bool fields (one per value;
//!   a struct row may not be null: `InvalidInput`, as Arrow); per row the value of the first
//!   true field (a null field counts as false), else the `else` value or null.
//! - choose(index, v0, v1, ...): per row the value `index` selects (any integer type, bool as
//!   0 / 1 and the null type, cast to int64 as Arrow's implicit cast; null gives null; a valid
//!   index outside `[0, n)` is `InvalidInput`, Arrow's IndexError).
//! - if_else and choose whose values are all of the null type give a null-type result, without
//!   looking at the index (pyarrow 26: choose([5], null, null) is [null]).
//!
//! The value arguments (array, chunked or scalar) are cast to their common type first: numbers
//! to `commonNumericType`, temporal types to `commonTemporalType`, string / binary layouts to
//! binary when one is binary and to the large layout when one is large, decimals to the
//! precision and scale that hold every value (decimal256 past 38 digits), a null-type argument
//! to the type of the others; any other mix is `NoMatchingKernel`. Every layout the take kernels
//! handle is a valid value type (also nested and view layouts, which Arrow lacks for some of
//! these functions). Chunked inputs give a chunked result.
//!
//! - drop_null: the valid rows of an array or chunked array (scalars are `NoMatchingKernel`, as
//!   Arrow); fill_null_forward / fill_null_backward: each null row takes the previous / next
//!   valid value (across chunk boundaries, as Arrow), leading / trailing nulls stay null; a
//!   scalar comes back as is. Every type.
//!
//! - replace_with_mask(values, mask, replacements), scatter(values, indices) (`ScatterOptions`)
//!   and inverse_permutation(indices) (`InversePermutationOptions`): see their functions below;
//!   indices are bounds-checked before any write and option-driven output lengths are capped.
//!
//! - take / array_take(values, indices) (`TakeOptions`) and filter / array_filter(values,
//!   predicate) (`FilterOptions`): see `takeExec` and `filterExec`.
//!
//! Inputs pass `validateLayout` before any buffer is read.
const std = @import("std");
const core = @import("core.zig");
const dense = @import("dense.zig");
const registry_mod = @import("registry.zig");
const bitmap = @import("../bitmap.zig");
const concat = @import("../concat_array_refs.zig");

const ArrayRef = core.ArrayRef;
const ChunkedArray = core.ChunkedArray;
const DataType = core.DataType;
const Datum = core.Datum;
const ExecChunkValue = core.ExecChunkValue;
const ExecContext = core.ExecContext;
const FunctionRegistry = core.FunctionRegistry;
const KernelError = core.KernelError;
const Options = core.Options;
const Scalar = core.Scalar;
const PrimitiveArray = @import("../array/primitive_array.zig").PrimitiveArray;
const ArrayData = @import("../array/array_data.zig").ArrayData;
const OwnedBuffer = @import("../buffer.zig").OwnedBuffer;

const Kind = enum { coalesce, if_else, case_when, choose };

/// Register coalesce, if_else, case_when, choose, drop_null, fill_null_forward and
/// fill_null_backward.
pub fn register(registry: *FunctionRegistry) KernelError!void {
    inline for (comptime std.meta.tags(Kind)) |kind| {
        try registry.registerScalarKernel(@tagName(kind), .{
            .signature = .{
                .arity = switch (kind) {
                    .coalesce => 1,
                    .if_else => 3,
                    .case_when, .choose => 2,
                },
                .variadic = kind != .if_else,
                .type_check = Select(kind).check,
                .options_check = dense.noOptions,
                .result_type_fn = Select(kind).resultType,
            },
            .exec = Select(kind).exec,
        });
    }
    try registry.registerVectorKernel("drop_null", .{
        .signature = .{ .arity = 1, .type_check = notScalar, .options_check = dense.noOptions, .tabular = true },
        .exec = dropNull,
    });
    inline for (.{ "fill_null_forward", "fill_null_backward" }) |name| {
        try registry.registerVectorKernel(name, .{
            .signature = .{ .arity = 1, .options_check = dense.noOptions },
            .exec = FillNull(std.mem.eql(u8, name, "fill_null_forward")).exec,
        });
    }
    try registry.registerVectorKernel("replace_with_mask", .{
        .signature = .{ .arity = 3, .type_check = replaceArgs, .options_check = dense.noOptions },
        .exec = replaceWithMask,
    });
    try registry.registerVectorKernel("scatter", .{
        .signature = .{ .arity = 2, .type_check = scatterArgs, .options_check = scatterOptions },
        .exec = scatterExec,
    });
    try registry.registerVectorKernel("indices_nonzero", .{
        .signature = .{ .arity = 1, .type_check = nonzeroArg, .options_check = dense.noOptions, .result_type_fn = dense.resultTypeOf(.{ .uint64 = {} }) },
        .exec = indicesNonzero,
    });
    try registry.registerVectorKernel("inverse_permutation", .{
        .signature = .{ .arity = 1, .type_check = inverseArgs, .options_check = inverseOptions, .result_type_fn = inverseType },
        .exec = inverseExec,
    });
    // take and filter (Arrow's meta functions) also take chunked, record batch and table values;
    // array_take takes arrays only and array_filter arrays and chunked arrays, as Arrow.
    inline for (.{ "take", "array_take" }) |name| {
        const meta = comptime std.mem.eql(u8, name, "take");
        try registry.registerVectorKernel(name, .{
            .signature = .{ .arity = 2, .type_check = Selector(meta).takeArgs, .options_check = takeOptions, .result_type_fn = valuesType, .tabular = meta },
            .exec = takeExec,
        });
    }
    inline for (.{ "filter", "array_filter" }) |name| {
        const meta = comptime std.mem.eql(u8, name, "filter");
        try registry.registerVectorKernel(name, .{
            .signature = .{ .arity = 2, .type_check = Selector(meta).filterArgs, .options_check = filterOptions, .result_type_fn = valuesType, .tabular = meta },
            .exec = filterExec,
        });
    }
}

fn notScalar(args: []const Datum) bool {
    return args.len == 1 and args[0] != .scalar;
}

// ------------------------------------------------------------------------------ common type

fn isBinaryLike(dt: DataType) bool {
    return switch (dt) {
        .string, .large_string, .binary, .large_binary => true,
        else => false,
    };
}

fn isDecimal(dt: DataType) bool {
    return switch (dt) {
        .decimal32, .decimal64, .decimal128, .decimal256 => true,
        else => false,
    };
}

/// The common type of two different value types (null when they do not mix).
fn pairType(a: DataType, b: DataType) ?DataType {
    const pair = [2]Datum{ .{ .scalar = Scalar.init(a, .null) }, .{ .scalar = Scalar.init(b, .null) } };
    if (registry_mod.commonNumericType(&pair)) |t| return t;
    if (registry_mod.commonTemporalType(&pair)) |t| return t;
    if (isBinaryLike(a) and isBinaryLike(b)) {
        const bin = a == .binary or a == .large_binary or b == .binary or b == .large_binary;
        const large = a == .large_string or a == .large_binary or b == .large_string or b == .large_binary;
        return if (bin) (if (large) .{ .large_binary = {} } else .{ .binary = {} }) else if (large) .{ .large_string = {} } else .{ .string = {} };
    }
    if (isDecimal(a) and isDecimal(b)) {
        const pa = switch (a) {
            .decimal32, .decimal64, .decimal128, .decimal256 => |p| p,
            else => unreachable,
        };
        const pb = switch (b) {
            .decimal32, .decimal64, .decimal128, .decimal256 => |p| p,
            else => unreachable,
        };
        const scale = @max(pa.scale, pb.scale);
        const digits = @max(@as(i64, pa.precision) - pa.scale, @as(i64, pb.precision) - pb.scale);
        const precision = digits + scale;
        if (precision < 1 or precision > 76) return null;
        const p: @TypeOf(pa) = .{ .precision = @intCast(precision), .scale = scale };
        return if (a == .decimal256 or b == .decimal256 or precision > 38) .{ .decimal256 = p } else .{ .decimal128 = p };
    }
    return null;
}

/// The common type of the value arguments, or null when they do not mix.
pub fn commonValueType(values: []const Datum) ?DataType {
    var acc: ?DataType = null;
    for (values) |v| {
        const t = v.dataType();
        if (t == .null) continue;
        if (acc) |a| {
            if (!a.eql(t)) acc = pairType(a, t) orelse return null;
        } else acc = t;
    }
    return acc orelse if (values.len > 0) DataType{ .null = {} } else null;
}

// ------------------------------------------------------------------------------ multiplexing

const Plan = struct {
    out_type: DataType,
    /// case_when: the last value is the `else` value.
    has_else: bool = false,
};

fn isInteger(dt: DataType) bool {
    return switch (dt) {
        .int8, .int16, .int32, .int64, .uint8, .uint16, .uint32, .uint64 => true,
        else => false,
    };
}

fn Select(comptime kind: Kind) type {
    const first: usize = if (kind == .coalesce) 0 else 1;
    return struct {
        fn check(args: []const Datum) bool {
            if (args.len <= first) return false;
            switch (kind) {
                .coalesce => {},
                .if_else => if (args.len != 3 or (args[0].dataType() != .bool and args[0].dataType() != .null)) return false,
                // A bool index is 0 / 1 (Arrow's implicit cast to an integer index).
                .choose => if (!isInteger(args[0].dataType()) and args[0].dataType() != .bool and args[0].dataType() != .null) return false,
                .case_when => {
                    const st = switch (args[0].dataType()) {
                        .struct_ => |st| st,
                        else => return false,
                    };
                    for (st.fields) |f| if (f.data_type.* != .bool) return false;
                    const n = args.len - 1;
                    if (n != st.fields.len and n != st.fields.len + 1) return false;
                },
            }
            return commonValueType(args[first..]) != null;
        }

        fn resultType(args: []const Datum, _: Options) KernelError!DataType {
            if (args.len <= first) return error.InvalidArity;
            return commonValueType(args[first..]) orelse error.NoMatchingKernel;
        }

        fn exec(ctx: *ExecContext, args: []const Datum, _: Options) KernelError!Datum {
            if (args.len <= first) return error.InvalidArity;
            const out_type = commonValueType(args[first..]) orelse return error.NoMatchingKernel;
            // Null-type values: a null-type result (Arrow's null kernels; no index is checked).
            if (out_type == .null and (kind == .if_else or kind == .choose)) return registry_mod.null_type_kernel.exec(ctx, args, .none);
            const call = ctx.allocator.alloc(Datum, args.len) catch return error.OutOfMemory;
            defer ctx.allocator.free(call);
            const owned = ctx.allocator.alloc(bool, args.len) catch return error.OutOfMemory;
            defer ctx.allocator.free(owned);
            @memset(owned, false);
            defer for (call, owned) |*c, o| if (o) c.release();
            for (args, call, owned, 0..) |a, *c, *o, i| {
                c.* = a;
                const target: ?DataType = if (i >= first) out_type else switch (kind) {
                    .choose => if (a.dataType() != .int64) DataType{ .int64 = {} } else null,
                    .if_else => if (a.dataType() == .null) DataType{ .bool = {} } else null,
                    else => null,
                };
                if (target) |t| if (!a.dataType().eql(t)) {
                    c.* = try ctx.invoke("cast", .scalar, &.{a}, .{ .cast = .{ .to_type = t } });
                    o.* = true;
                };
            }
            var plan: Plan = .{ .out_type = out_type };
            if (kind == .case_when) plan.has_else = args.len - 1 == args[0].dataType().struct_.fields.len + 1;
            return dense.execElementwiseWith(ctx, call, .{ .custom = .{ .tag = @tagName(kind), .payload = &plan } }, out_type, false, chunk);
        }

        /// Whether value `v` (an array chunk or a broadcast scalar) is valid at row `i`.
        inline fn validAt(v: ExecChunkValue, i: usize) bool {
            return switch (v) {
                .scalar => |s| !s.isNull(),
                .array => |arr| !arr.data().isNull(i),
            };
        }

        fn chunk(ctx: *ExecContext, values: []const ExecChunkValue, len: usize, options: Options) KernelError!ArrayRef {
            const allocator = ctx.allocator;
            const plan: *const Plan = @ptrCast(@alignCast(options.custom.payload.?));
            const vals = values[first..];
            const choice = allocator.alloc(?u32, len) catch return error.OutOfMemory;
            defer allocator.free(choice);
            try choose(allocator, values, plan, choice);

            // Every value as an array (a scalar as one row), then one take over all of them.
            const pieces = allocator.alloc(ArrayRef, vals.len) catch return error.OutOfMemory;
            var made: usize = 0;
            defer {
                for (pieces[0..made]) |*p| p.release();
                allocator.free(pieces);
            }
            const base = allocator.alloc(usize, vals.len) catch return error.OutOfMemory;
            defer allocator.free(base);
            var total: usize = 0;
            for (vals, pieces, base) |v, *p, *b| {
                p.* = try dense.chunkArray(allocator, v);
                made += 1;
                b.* = total;
                total += p.data().length;
            }
            var combined = if (pieces.len == 1) pieces[0].retain() else concat.concatArrayRefs(allocator, plan.out_type, pieces) catch |err| return switch (err) {
                error.OutOfMemory => error.OutOfMemory,
                else => error.InvalidInput,
            };
            defer combined.release();
            const indices = allocator.alloc(?usize, len) catch return error.OutOfMemory;
            defer allocator.free(indices);
            for (indices, choice, 0..) |*ix, c, i| ix.* = if (c) |k| base[k] + (if (vals[k] == .scalar) 0 else i) else null;
            if (len == 0) return combined.slice(0, 0) catch error.OutOfMemory;
            var taken = try core.datumTakeNullable(.{ .array = combined }, indices);
            return switch (taken) {
                .array => |a| a,
                else => {
                    taken.release();
                    return error.InvalidInput;
                },
            };
        }

        /// The value index of every row (null: a null row).
        fn choose(allocator: std.mem.Allocator, values: []const ExecChunkValue, plan: *const Plan, out: []?u32) KernelError!void {
            const vals = values[first..];
            switch (kind) {
                .coalesce => for (out, 0..) |*o, i| {
                    o.* = null;
                    for (vals, 0..) |v, k| if (validAt(v, i)) {
                        o.* = @intCast(k);
                        break;
                    };
                },
                .if_else => {
                    var cond = try dense.chunkArray(allocator, values[0]);
                    defer cond.release();
                    const c = cond.data();
                    const row0 = values[0] == .scalar;
                    for (out, 0..) |*o, i| {
                        const r = if (row0) 0 else i;
                        o.* = if (c.isNull(r)) null else if (bitmap.bitIsSet(c.buffers[1].data, c.offset + r)) 0 else 1;
                    }
                },
                .case_when => {
                    var cond = try dense.chunkArray(allocator, values[0]);
                    defer cond.release();
                    const c = cond.data();
                    const row0 = values[0] == .scalar;
                    const fields = c.children.len;
                    for (c.children) |child| child.data().validateLayout() catch return error.InvalidInput;
                    for (out, 0..) |*o, i| {
                        const r = if (row0) 0 else i;
                        if (c.isNull(r)) return error.InvalidInput;
                        o.* = if (plan.has_else) @intCast(fields) else null;
                        for (c.children, 0..) |child, k| {
                            const d = child.data();
                            // A child is row-aligned with its parent or in parent coordinates.
                            const row = if (d.length == c.length) r else c.offset + r;
                            if (row >= d.length) return error.InvalidInput;
                            if (!d.isNull(row) and bitmap.bitIsSet(d.buffers[1].data, d.offset + row)) {
                                o.* = @intCast(k);
                                break;
                            }
                        }
                    }
                },
                .choose => {
                    const n: i64 = @intCast(vals.len);
                    switch (values[0]) {
                        .scalar => |s| {
                            const k: ?i64 = if (s.isNull()) null else try dense.scalarAs(i64, s);
                            if (k) |x| if (x < 0 or x >= n) return error.InvalidInput;
                            for (out) |*o| o.* = if (k) |x| @intCast(x) else null;
                        },
                        .array => |arr| {
                            const d = arr.data();
                            const ks = (PrimitiveArray(i64){ .data = d }).values() catch return error.InvalidInput;
                            for (out, ks, 0..) |*o, k, i| {
                                if (d.isNull(i)) {
                                    o.* = null;
                                    continue;
                                }
                                if (k < 0 or k >= n) return error.InvalidInput;
                                o.* = @intCast(k);
                            }
                        },
                    }
                },
            }
        }
    };
}

// ------------------------------------------------------------------------------ null removal

fn dropNull(ctx: *ExecContext, args: []const Datum, _: Options) KernelError!Datum {
    if (args.len != 1) return error.InvalidArity;
    switch (args[0]) {
        .array => |arr| return .{ .array = try dropArray(ctx.allocator, arr) },
        .chunked => |c| {
            const outs = ctx.allocator.alloc(ArrayRef, c.numChunks()) catch return error.OutOfMemory;
            var made: usize = 0;
            defer {
                for (outs[0..made]) |*o| o.release();
                ctx.allocator.free(outs);
            }
            for (c.chunks()) |part| {
                outs[made] = try dropArray(ctx.allocator, part);
                made += 1;
            }
            const out = ChunkedArray.init(ctx.allocator, c.dataType(), outs) catch |err| return switch (err) {
                error.OutOfMemory => error.OutOfMemory,
                else => error.InvalidInput,
            };
            return .{ .chunked = out };
        },
        .scalar => return error.NoMatchingKernel,
        .record_batch, .table => return dropNullTabular(ctx, args[0]),
    }
}

/// The valid rows of `arr` (the array itself when none is null).
fn dropArray(allocator: std.mem.Allocator, arr: ArrayRef) KernelError!ArrayRef {
    const data = arr.data();
    data.validateLayout() catch return error.InvalidInput;
    if (!data.hasNulls() and datatypeHasNoNulls(data)) return arr.retain();
    var keep: std.ArrayList(usize) = .empty;
    defer keep.deinit(allocator);
    for (0..data.length) |i| if (!data.isNull(i)) keep.append(allocator, i) catch return error.OutOfMemory;
    if (keep.items.len == data.length) return arr.retain();
    var taken = try core.datumTake(.{ .array = arr }, keep.items);
    return switch (taken) {
        .array => |a| a,
        else => {
            taken.release();
            return error.InvalidInput;
        },
    };
}

/// drop_null of a record batch or table: the rows with no null in any column (Arrow
/// `DropNull` on a `RecordBatch` / `Table`); the datum itself when no row has one.
fn dropNullTabular(ctx: *ExecContext, datum: Datum) KernelError!Datum {
    const n = datum.length();
    // ponytail: one byte per row; a bitmap AND of the validity buffers would be 8x smaller.
    const drop = ctx.allocator.alloc(bool, n) catch return error.OutOfMemory;
    defer ctx.allocator.free(drop);
    @memset(drop, false);
    switch (datum) {
        .record_batch => |b| for (b.get().columns) |col| markNulls(col.data(), drop),
        .table => |t| for (t.get().columns) |col| {
            var pos: usize = 0;
            for (col.chunks()) |chunk| {
                const len = chunk.data().length;
                markNulls(chunk.data(), drop[pos..][0..len]);
                pos += len;
            }
        },
        else => return error.InvalidInput,
    }
    var keep: std.ArrayList(usize) = .empty;
    defer keep.deinit(ctx.allocator);
    for (drop, 0..) |d, i| if (!d) keep.append(ctx.allocator, i) catch return error.OutOfMemory;
    if (keep.items.len == n) return datum.retain();
    return core.datumTake(datum, keep.items);
}

/// Mark the null rows of one validated column part.
fn markNulls(data: *const ArrayData, drop: []bool) void {
    if (!datatypeHasNoNulls(data)) return @memset(drop, true);
    if (!data.hasNulls()) return;
    for (drop, 0..) |*d, i| d.* = d.* or data.isNull(i);
}

/// The null type has no validity bitmap but every row is null.
fn datatypeHasNoNulls(data: *const ArrayData) bool {
    return !(data.data_type == .null and data.length > 0);
}

fn FillNull(comptime forward: bool) type {
    return struct {
        fn exec(ctx: *ExecContext, args: []const Datum, _: Options) KernelError!Datum {
            if (args.len != 1) return error.InvalidArity;
            const parts: []const ArrayRef = switch (args[0]) {
                .scalar => return args[0].retain(),
                .array => |*a| a[0..1],
                .chunked => |c| c.chunks(),
                .record_batch, .table => return error.InvalidInput,
            };
            var total: usize = 0;
            var any_null = false;
            for (parts) |p| {
                p.data().validateLayout() catch return error.InvalidInput;
                total += p.data().length;
                any_null = any_null or p.data().hasNulls() or !datatypeHasNoNulls(p.data());
            }
            if (!any_null) return args[0].retain();
            // The logical row each row takes (null: no valid row before / after it).
            const indices = ctx.allocator.alloc(?usize, total) catch return error.OutOfMemory;
            defer ctx.allocator.free(indices);
            var last: ?usize = null;
            if (forward) {
                var pos: usize = 0;
                for (parts) |p| for (0..p.data().length) |i| {
                    if (!p.data().isNull(i)) last = pos;
                    indices[pos] = last;
                    pos += 1;
                };
            } else {
                var pos = total;
                var k = parts.len;
                while (k > 0) {
                    k -= 1;
                    const d = parts[k].data();
                    var i = d.length;
                    while (i > 0) {
                        i -= 1;
                        pos -= 1;
                        if (!d.isNull(i)) last = pos;
                        indices[pos] = last;
                    }
                }
            }
            return core.datumTakeNullable(args[0], indices);
        }
    };
}

// ------------------------------------------------------------------------------ replace_with_mask, scatter, inverse_permutation

fn isSignedInt(dt: DataType) bool {
    return switch (dt) {
        .int8, .int16, .int32, .int64 => true,
        else => false,
    };
}

fn replaceArgs(args: []const Datum) bool {
    return args.len == 3 and args[0] != .scalar and !args[0].isTabular() and args[1].dataType() == .bool and
        !args[1].isTabular() and !args[2].isTabular() and args[2].dataType().eql(args[0].dataType());
}

fn scatterArgs(args: []const Datum) bool {
    return args.len == 2 and args[0] != .scalar and !args[0].isTabular() and args[1] != .scalar and !args[1].isTabular() and isSignedInt(args[1].dataType());
}

fn inverseArgs(args: []const Datum) bool {
    return args.len == 1 and args[0] != .scalar and !args[0].isTabular() and isSignedInt(args[0].dataType());
}

fn scatterOptions(options: Options) bool {
    return options == .none or options == .scatter;
}

fn inverseOptions(options: Options) bool {
    return options == .none or options == .inverse_permutation;
}

fn inverseType(args: []const Datum, options: Options) KernelError!DataType {
    if (args.len != 1) return error.InvalidArity;
    if (options == .inverse_permutation) if (options.inverse_permutation.output_type) |t| return t;
    return args[0].dataType();
}

/// The rows of an array or chunked datum as one validated array (chunks concatenated).
fn wholeArray(allocator: std.mem.Allocator, datum: Datum) KernelError!ArrayRef {
    const parts: []const ArrayRef = switch (datum) {
        .array => |*a| a[0..1],
        .chunked => |c| c.chunks(),
        else => return error.InvalidInput,
    };
    for (parts) |p| p.data().validateLayout() catch return error.InvalidInput;
    if (parts.len == 1) return parts[0].retain();
    if (parts.len == 0) return dense.nullArray(allocator, datum.dataType(), 0);
    return concat.concatArrayRefs(allocator, datum.dataType(), parts) catch |err| switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        else => error.InvalidInput,
    };
}

/// `arr` as a datum of the kind of `like`: a chunked array of one chunk when `like` is chunked.
fn likeInput(allocator: std.mem.Allocator, like: Datum, arr: ArrayRef) KernelError!Datum {
    if (like != .chunked) return .{ .array = arr };
    var a = arr;
    defer a.release();
    const out = ChunkedArray.init(allocator, a.data().data_type, &.{a}) catch |err| return switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        else => error.InvalidInput,
    };
    return .{ .chunked = out };
}

/// One mask row: null, false or true.
const MaskRow = enum(u2) { null, false, true };

const Mask = union(enum) {
    scalar: MaskRow,
    array: struct { values: []const u8, validity: ?[]const u8, offset: usize },

    inline fn at(self: Mask, i: usize) MaskRow {
        return switch (self) {
            .scalar => |m| m,
            .array => |a| if (a.validity) |v| (if (!bitmap.bitIsSet(v, a.offset + i)) .null else if (bitmap.bitIsSet(a.values, a.offset + i)) .true else .false) else if (bitmap.bitIsSet(a.values, a.offset + i)) .true else .false,
        };
    }
};

/// replace_with_mask(values, mask, replacements): each row whose mask is true takes the next
/// replacement (an array, consumed in order, or a scalar), a null mask row gives null, a false
/// one keeps the value (Arrow C++). The mask is a bool array of the values' length or a bool
/// scalar; an array of replacements needs at least as many rows as true mask rows
/// (`InvalidInput`, Arrow's Invalid); a chunked mask or replacement is `InvalidInput`, as Arrow.
/// Every layout the take kernels handle (Arrow lacks nested and view layouts here).
fn replaceWithMask(ctx: *ExecContext, args: []const Datum, _: Options) KernelError!Datum {
    if (args.len != 3) return error.InvalidArity;
    if (args[1] == .chunked or args[2] == .chunked) return error.InvalidInput;
    const allocator = ctx.allocator;
    var values = try wholeArray(allocator, args[0]);
    defer values.release();
    const vd = values.data();
    const n = vd.length;
    var trues: usize = 0;
    const mask: Mask = switch (args[1]) {
        .scalar => |s| blk: {
            const m: MaskRow = if (s.isNull()) .null else if (try dense.scalarAs(bool, s)) .true else .false;
            if (m == .true) trues = n;
            break :blk .{ .scalar = m };
        },
        .array => |a| blk: {
            const md = a.data();
            md.validateLayout() catch return error.InvalidInput;
            if (md.length != n) return error.InvalidInput;
            const m: Mask = .{ .array = .{ .values = md.buffers[1].data, .validity = dense.validityBits(md), .offset = md.offset } };
            const vals = dense.Bits.values(md);
            const valid = dense.Bits.validity(md);
            for (0..bitmap.byteLength(n)) |j| {
                var t = vals.byte(j) & valid.byte(j);
                if (j * 8 + 8 > n) t &= (@as(u8, 1) << @intCast(n - j * 8)) - 1;
                trues += @popCount(t);
            }
            break :blk m;
        },
        else => return error.InvalidInput,
    };
    var repl = try dense.chunkArray(allocator, switch (args[2]) {
        .array => |a| .{ .array = a },
        .scalar => |s| .{ .scalar = s },
        else => return error.InvalidInput,
    });
    defer repl.release();
    const rd = repl.data();
    rd.validateLayout() catch return error.InvalidInput;
    const broadcast = args[2] == .scalar;
    if (!broadcast and rd.length < trues) return error.InvalidInput;
    if (!rd.data_type.eql(vd.data_type)) return error.InvalidInput;
    const out = try replaceFast(allocator, vd, mask, rd, broadcast) orelse try replaceTake(allocator, values, mask, repl, broadcast);
    return likeInput(allocator, args[0], out);
}

/// Fixed-width values of 1..32 bytes (numbers, temporal types, decimals, half floats): one pass
/// writing the values and the validity; null when the type takes the generic path.
fn replaceFast(allocator: std.mem.Allocator, vd: *const ArrayData, mask: Mask, rd: *const ArrayData, broadcast: bool) KernelError!?ArrayRef {
    switch (vd.data_type) {
        .int8, .int16, .int32, .int64, .uint8, .uint16, .uint32, .uint64, .half_float, .float, .double => {},
        .date32, .date64, .timestamp, .time32, .time64, .duration => {},
        .interval_months, .interval_day_time, .interval_month_day_nano => {},
        .decimal32, .decimal64, .decimal128, .decimal256 => {},
        else => return null,
    }
    const bits = vd.data_type.bitWidth() orelse return null;
    // 16- and 32-byte values as byte arrays: no alignment beyond the buffer's own.
    inline for (.{ u8, u16, u32, u64, [16]u8, [32]u8 }) |U| if (bits == @bitSizeOf(U)) {
        return try replaceFixed(U, allocator, vd, mask, rd, broadcast);
    };
    return null;
}

fn replaceFixed(comptime U: type, allocator: std.mem.Allocator, vd: *const ArrayData, mask: Mask, rd: *const ArrayData, broadcast: bool) KernelError!ArrayRef {
    const n = vd.length;
    const src = (PrimitiveArray(U){ .data = vd }).values() catch return error.InvalidInput;
    const rep = (PrimitiveArray(U){ .data = rd }).values() catch return error.InvalidInput;
    const rep_valid = dense.validityBits(rd);
    var buf = OwnedBuffer.initUninit(allocator, n * @sizeOf(U)) catch return error.OutOfMemory;
    var validity = dense.bitsBuffer(allocator, n) catch |err| {
        buf.deinit();
        return err;
    };
    const out = buf.typedSlice(U)[0..n];
    const vbits = validity.data[0..bitmap.byteLength(n)];
    // Mask and value bits 8 rows at a time (a scalar mask is a splat).
    const mvals: dense.Bits, const mvalid: dense.Bits = switch (mask) {
        .scalar => |m| .{ dense.Bits.of(m == .true), dense.Bits.of(m != .null) },
        .array => |a| .{ .{ .bytes = a.values, .offset = a.offset }, if (a.validity) |v| .{ .bytes = v, .offset = a.offset } else .{} },
    };
    const svalid = dense.Bits.validity(vd);
    // Replacement row `k` clamped to the last one: rows past it are never selected.
    const last = if (rep.len == 0) 0 else rep.len - 1;
    var k: usize = 0;
    for (vbits, 0..) |*vb, j| {
        const mv = mvals.byte(j);
        const mok = mvalid.byte(j);
        const take = mv & mok;
        var ob = ~mv & mok & svalid.byte(j);
        const base = j * 8;
        const cnt = @min(8, n - base);
        if (take == 0 or rep.len == 0) {
            // (No replacement row means no true row: `trues` bounds the replacements.)
            @memcpy(out[base..][0..cnt], src[base..][0..cnt]);
        } else for (0..cnt) |r| {
            const i = base + r;
            const t = (take >> @intCast(r)) & 1;
            const ri = if (broadcast) 0 else @min(k, last);
            out[i] = if (t != 0) rep[ri] else src[i];
            const rv: u8 = if (rep_valid) |b| @intFromBool(bitmap.bitIsSet(b, rd.offset + ri)) else 1;
            ob |= (t & rv) << @intCast(r);
            k += t;
        }
        vb.* = ob;
    }
    if (n % 8 != 0) vbits[vbits.len - 1] &= (@as(u8, 1) << @intCast(n % 8)) - 1;
    var v: ?OwnedBuffer = validity;
    if (bitmap.countUnsetBit(validity.data, n) == 0) {
        validity.deinit();
        v = null;
    }
    return dense.finishFixed(allocator, vd.data_type, n, &buf, v);
}

/// Every other layout: one take over the values followed by the replacements.
fn replaceTake(allocator: std.mem.Allocator, values: ArrayRef, mask: Mask, repl: ArrayRef, broadcast: bool) KernelError!ArrayRef {
    const n = values.data().length;
    var both = concat.concatArrayRefs(allocator, values.data().data_type, &.{ values, repl }) catch |err| return switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        else => error.InvalidInput,
    };
    defer both.release();
    const indices = allocator.alloc(?usize, n) catch return error.OutOfMemory;
    defer allocator.free(indices);
    var k: usize = 0;
    for (indices, 0..) |*ix, i| ix.* = switch (mask.at(i)) {
        .null => null,
        .false => i,
        .true => blk: {
            defer k += 1;
            break :blk n + (if (broadcast) 0 else k);
        },
    };
    var taken = try core.datumTakeNullable(.{ .array = both }, indices);
    return switch (taken) {
        .array => |a| a,
        else => {
            taken.release();
            return error.InvalidInput;
        },
    };
}

/// Output rows of scatter / inverse_permutation: `max_index + 1`, or `n` when negative.
/// Capped at max(n, 2^24) rows so an option cannot force an allocation the input does not
/// justify (`InvalidOptions`).
fn outputLength(max_index: i64, n: usize) KernelError!usize {
    if (max_index < 0) return n;
    const want = std.math.cast(usize, max_index) orelse return error.InvalidOptions;
    if (want >= @max(n, @as(usize, 1) << 24)) return error.InvalidOptions;
    return want + 1;
}

/// Call `sink.put(position, index)` for every valid row of a signed integer index datum
/// (array or chunked; every chunk passes `validateLayout`).
fn eachIndex(datum: Datum, sink: anytype) KernelError!void {
    const parts: []const ArrayRef = switch (datum) {
        .array => |*a| a[0..1],
        .chunked => |c| c.chunks(),
        else => return error.InvalidInput,
    };
    var pos: usize = 0;
    for (parts) |p| {
        const d = p.data();
        d.validateLayout() catch return error.InvalidInput;
        switch (d.data_type) {
            inline .int8, .int16, .int32, .int64 => |_, tag| {
                const I = switch (tag) {
                    .int8 => i8,
                    .int16 => i16,
                    .int32 => i32,
                    else => i64,
                };
                const xs = (PrimitiveArray(I){ .data = d }).values() catch return error.InvalidInput;
                const bits = dense.validityBits(d);
                for (xs, 0..) |x, i| {
                    if (bits) |b| if (!bitmap.bitIsSet(b, d.offset + i)) continue;
                    try sink.put(pos + i, x);
                }
            },
            else => return error.InvalidInput,
        }
        pos += d.length;
    }
}

/// scatter(values, indices): output row `indices[i]` takes `values[i]` (a later row wins on a
/// repeated index; a null index drops its row; rows no index names are null), Arrow C++. The
/// indices are signed integers of the values' length, each in `[0, length)` (`InvalidInput`,
/// Arrow's IndexError); `ScatterOptions.max_index` sets the length (see `outputLength`). Every
/// layout the take kernels handle; a chunked input gives a chunked output.
fn scatterExec(ctx: *ExecContext, args: []const Datum, options: Options) KernelError!Datum {
    if (args.len != 2) return error.InvalidArity;
    const n = args[1].length();
    if (args[0].length() != n) return error.InvalidInput;
    const max_index: i64 = if (options == .scatter) options.scatter.max_index else -1;
    const out_len = try outputLength(max_index, n);
    const src = ctx.allocator.alloc(?usize, out_len) catch return error.OutOfMemory;
    defer ctx.allocator.free(src);
    @memset(src, null);
    const Sink = struct {
        src: []?usize,
        fn put(self: @This(), pos: usize, idx: anytype) KernelError!void {
            const at = std.math.cast(usize, idx) orelse return error.InvalidInput;
            if (at >= self.src.len) return error.InvalidInput;
            self.src[at] = pos;
        }
    };
    try eachIndex(args[1], Sink{ .src = src });
    switch (args[0]) {
        .array => |a| a.data().validateLayout() catch return error.InvalidInput,
        .chunked => |c| for (c.chunks()) |p| p.data().validateLayout() catch return error.InvalidInput,
        else => return error.InvalidInput,
    }
    return core.datumTakeNullable(args[0], src);
}

/// inverse_permutation(indices): output row `indices[i]` is `i` (a later row wins; null
/// indices are skipped; rows no index names are null), Arrow C++. Signed integer indices,
/// each in `[0, length)` (`InvalidInput`); the output type (`InversePermutationOptions.
/// output_type`, else the index type) is a signed integer type holding the input length
/// (another type is `InvalidOptions`, a length past its maximum `InvalidInput`, as Arrow).
fn inverseExec(ctx: *ExecContext, args: []const Datum, options: Options) KernelError!Datum {
    if (args.len != 1) return error.InvalidArity;
    const out_type = try inverseType(args, options);
    const n = args[0].length();
    const max_index: i64 = if (options == .inverse_permutation) options.inverse_permutation.max_index else -1;
    const out_len = try outputLength(max_index, n);
    const arr = switch (out_type) {
        inline .int8, .int16, .int32, .int64 => |_, tag| blk: {
            const O = switch (tag) {
                .int8 => i8,
                .int16 => i16,
                .int32 => i32,
                else => i64,
            };
            if (n > std.math.maxInt(O)) return error.InvalidInput;
            break :blk try inverseInto(O, ctx.allocator, args[0], out_type, out_len);
        },
        else => return error.InvalidOptions,
    };
    return likeInput(ctx.allocator, args[0], arr);
}

fn inverseInto(comptime O: type, allocator: std.mem.Allocator, indices: Datum, out_type: DataType, out_len: usize) KernelError!ArrayRef {
    var buf = OwnedBuffer.init(allocator, out_len * @sizeOf(O)) catch return error.OutOfMemory;
    var validity = dense.bitsBuffer(allocator, out_len) catch |err| {
        buf.deinit();
        return err;
    };
    @memset(validity.data[0..bitmap.byteLength(out_len)], 0);
    const Sink = struct {
        out: []O,
        valid: []u8,
        fn put(self: @This(), pos: usize, idx: anytype) KernelError!void {
            const at = std.math.cast(usize, idx) orelse return error.InvalidInput;
            if (at >= self.out.len) return error.InvalidInput;
            self.out[at] = @intCast(pos);
            bitmap.setBit(self.valid, at);
        }
    };
    eachIndex(indices, Sink{ .out = buf.typedSlice(O)[0..out_len], .valid = validity.data }) catch |err| {
        buf.deinit();
        validity.deinit();
        return err;
    };
    var v: ?OwnedBuffer = validity;
    if (bitmap.countUnsetBit(validity.data, out_len) == 0) {
        validity.deinit();
        v = null;
    }
    return dense.finishFixed(allocator, out_type, out_len, &buf, v);
}

// ------------------------------------------------------------------------------ indices_nonzero

fn nonzeroArg(args: []const Datum) bool {
    if (args.len != 1 or args[0].isTabular()) return false;
    return switch (args[0].dataType()) {
        .bool, .int8, .int16, .int32, .int64, .uint8, .uint16, .uint32, .uint64, .float, .double => true,
        .decimal32, .decimal64, .decimal128, .decimal256 => true,
        else => false,
    };
}

/// indices_nonzero: the uint64 positions of the valid rows that are not zero (bool: true; a
/// float NaN is nonzero, -0.0 is zero) across every chunk, as one array (Arrow C++). Numbers,
/// bool and decimals (pyarrow 26 has no decimal32 / decimal64 kernel; zarrow takes them too).
fn indicesNonzero(ctx: *ExecContext, args: []const Datum, _: Options) KernelError!Datum {
    if (args.len != 1) return error.InvalidArity;
    var scalar_arr: ?ArrayRef = null;
    defer if (scalar_arr) |*a| a.release();
    const parts: []const ArrayRef = switch (args[0]) {
        .array => |*a| a[0..1],
        .chunked => |c| c.chunks(),
        .scalar => |s| blk: {
            scalar_arr = try dense.chunkArray(ctx.allocator, .{ .scalar = s });
            break :blk @as(*const [1]ArrayRef, &scalar_arr.?)[0..1];
        },
        else => return error.InvalidInput,
    };
    var total: usize = 0;
    for (parts) |p| {
        p.data().validateLayout() catch return error.InvalidInput;
        total += try nonzeroIn(p.data(), null, 0);
    }
    // One slot of slack: the branch-free writer stores past the last index.
    var buf = OwnedBuffer.initUninit(ctx.allocator, (total + 1) * 8) catch return error.OutOfMemory;
    const out = buf.typedSlice(u64)[0 .. total + 1];
    var k: usize = 0;
    var base: usize = 0;
    for (parts) |p| {
        k += nonzeroIn(p.data(), out[k..], base) catch unreachable;
        base += p.data().length;
    }
    return .{ .array = try dense.finishFixed(ctx.allocator, .{ .uint64 = {} }, total, &buf, null) };
}

/// Count the nonzero valid rows of `d`, writing `base + row` of each into `out` when given
/// (`out` holds the count plus one).
fn nonzeroIn(d: *const ArrayData, out: ?[]u64, base: usize) KernelError!usize {
    const bits = dense.validityBits(d);
    if (d.data_type == .bool) {
        const vals = dense.Bits.values(d);
        const valid = dense.Bits.validity(d);
        var k: usize = 0;
        for (0..bitmap.byteLength(d.length)) |j| {
            var m = vals.byte(j) & valid.byte(j);
            if (j + 1 == bitmap.byteLength(d.length) and d.length % 8 != 0) m &= (@as(u8, 1) << @as(u3, @intCast(d.length % 8))) - 1;
            if (out) |o| {
                while (m != 0) : (m &= m - 1) {
                    o[k] = base + j * 8 + @ctz(m);
                    k += 1;
                }
            } else k += @popCount(m);
        }
        return k;
    }
    inline for (.{ i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, i128, i256 }) |T| {
        const match = if (T == i128) d.data_type == .decimal128 else if (T == i256) d.data_type == .decimal256 else std.meta.activeTag(d.data_type) == (comptime std.meta.activeTag(dense.dataTypeOf(T))) or
            (T == i32 and d.data_type == .decimal32) or (T == i64 and d.data_type == .decimal64);
        if (match) {
            const xs = (PrimitiveArray(T){ .data = d }).values() catch return error.InvalidInput;
            var k: usize = 0;
            if (bits) |b| {
                if (out) |o| {
                    for (xs, 0..) |x, i| {
                        o[k] = base + i;
                        k += @intFromBool(x != 0 and bitmap.bitIsSet(b, d.offset + i));
                    }
                } else for (xs, 0..) |x, i| {
                    k += @intFromBool(x != 0 and bitmap.bitIsSet(b, d.offset + i));
                }
            } else if (out) |o| {
                // Branch-free: every row writes its index, only a nonzero one advances.
                for (xs, 0..) |x, i| {
                    o[k] = base + i;
                    k += @intFromBool(x != 0);
                }
            } else for (xs) |x| {
                k += @intFromBool(x != 0);
            }
            return k;
        }
    }
    return error.InvalidInput;
}

// ------------------------------------------------------------------------------ take, filter

fn Selector(comptime meta: bool) type {
    return struct {
        /// take: values of any datum kind but a scalar, indices an integer array or chunked
        /// array; array_take: both arrays.
        fn takeArgs(args: []const Datum) bool {
            if (args.len != 2 or !isIntegerType(args[1].dataType())) return false;
            if (!meta) return args[0] == .array and args[1] == .array;
            return args[0] != .scalar and (args[1] == .array or args[1] == .chunked);
        }

        /// filter: values of any datum kind but a scalar; array_filter: an array or chunked
        /// array. The predicate is a bool array or chunked array (a scalar is not, as Arrow).
        fn filterArgs(args: []const Datum) bool {
            if (args.len != 2 or args[1].dataType() != .bool) return false;
            if (args[1] != .array and args[1] != .chunked) return false;
            return args[0] == .array or args[0] == .chunked or (meta and args[0].isTabular());
        }
    };
}

fn isIntegerType(dt: DataType) bool {
    return switch (dt) {
        .int8, .int16, .int32, .int64, .uint8, .uint16, .uint32, .uint64 => true,
        else => false,
    };
}

fn takeOptions(options: Options) bool {
    return options == .none or options == .take;
}

fn filterOptions(options: Options) bool {
    return options == .none or options == .filter;
}

fn valuesType(args: []const Datum, _: Options) KernelError!DataType {
    if (args.len != 2) return error.InvalidArity;
    return args[0].dataType();
}

/// Validate every chunk of an array or chunked datum (record batch and table columns were
/// checked when the batch or table was built).
fn validateParts(datum: Datum) KernelError!void {
    switch (datum) {
        .array => |a| a.data().validateLayout() catch return error.InvalidInput,
        .chunked => |c| for (c.chunks()) |p| p.data().validateLayout() catch return error.InvalidInput,
        else => {},
    }
}

/// take(values, indices): row `indices[i]` of `values` for every `i` (Arrow C++). Indices of any
/// integer type; a null index gives a null row; a negative index or one at or past the values'
/// length is `InvalidInput` (Arrow's IndexError), also with `TakeOptions.boundscheck` false
/// (Arrow then skips the check; zarrow never reads out of bounds). Chunked values give a chunked
/// result, record batches and tables their own kind (`take` only); chunked indices give a
/// chunked result. Every layout the take kernels handle.
pub fn takeExec(ctx: *ExecContext, args: []const Datum, _: Options) KernelError!Datum {
    if (args.len != 2) return error.InvalidArity;
    try validateParts(args[0]);
    const n = args[0].length();
    const parts: []const ArrayRef = switch (args[1]) {
        .array => |*a| a[0..1],
        .chunked => |c| c.chunks(),
        else => return error.InvalidInput,
    };
    const total = args[1].length();
    const none_idx = std.math.maxInt(usize); // a null index
    const dense_idx = ctx.allocator.alloc(usize, total) catch return error.OutOfMemory;
    defer ctx.allocator.free(dense_idx);
    var any_null = false;
    var pos: usize = 0;
    for (parts) |p| {
        const d = p.data();
        d.validateLayout() catch return error.InvalidInput;
        const out = dense_idx[pos..][0..d.length];
        switch (d.data_type) {
            inline .int8, .int16, .int32, .int64, .uint8, .uint16, .uint32, .uint64 => |_, tag| {
                const I = switch (tag) {
                    .int8 => i8,
                    .int16 => i16,
                    .int32 => i32,
                    .int64 => i64,
                    .uint8 => u8,
                    .uint16 => u16,
                    .uint32 => u32,
                    else => u64,
                };
                const xs = (PrimitiveArray(I){ .data = d }).values() catch return error.InvalidInput;
                const bits = dense.validityBits(d);
                for (out, xs, 0..) |*o, x, i| {
                    if (bits) |b| if (!bitmap.bitIsSet(b, d.offset + i)) {
                        any_null = true;
                        o.* = none_idx;
                        continue;
                    };
                    const at = std.math.cast(usize, x) orelse return error.InvalidInput;
                    if (at >= n) return error.InvalidInput;
                    o.* = at;
                }
            },
            else => return error.InvalidInput,
        }
        pos += d.length;
    }
    const out = if (any_null) blk: {
        const nullable = ctx.allocator.alloc(?usize, total) catch return error.OutOfMemory;
        defer ctx.allocator.free(nullable);
        for (nullable, dense_idx) |*o, x| o.* = if (x == none_idx) null else x;
        break :blk try core.datumTakeNullable(args[0], nullable);
    } else try core.datumTake(args[0], dense_idx);
    if (args[1] == .chunked and out == .array) return likeInput(ctx.allocator, args[1], out.array);
    return out;
}

/// filter(values, predicate): the rows where the bool predicate (of the values' length,
/// `InvalidInput` otherwise) is true; a null predicate row drops the row, or gives a null row with
/// `FilterOptions.drop_nulls` false (Arrow's `emit_null`). Array values and predicate give an
/// array, a chunked argument a chunked result, record batches and tables their own kind
/// (`filter` only). Every layout the take kernels handle.
pub fn filterExec(ctx: *ExecContext, args: []const Datum, options: Options) KernelError!Datum {
    if (args.len != 2) return error.InvalidArity;
    try validateParts(args[0]);
    try validateParts(args[1]);
    if (args[1].length() != args[0].length()) return error.InvalidInput;
    const opts: core.FilterOptions = if (options == .filter) options.filter else .{};
    if (args[0] == .array and args[1] == .array) return core.datumFilter(args[0], args[1], opts);
    const out = try core.datumFilterChunkAware(args[0], args[1], opts);
    if (args[1] == .chunked and out == .array) return likeInput(ctx.allocator, args[1], out.array);
    return out;
}
