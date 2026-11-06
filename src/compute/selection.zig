//! Selecting, multiplexing and null replacement (Arrow C++ "Selecting / multiplexing", "Replace
//! functions", "Selections"): coalesce, if_else, case_when, choose (scalar functions),
//! drop_null, fill_null_forward and fill_null_backward (vector functions).
//!
//! - coalesce(v0, v1, ...): per row the first valid value (null when none is).
//! - if_else(cond, a, b): `a` where cond is true, `b` where false, null where cond is null.
//! - case_when(cond, v0, ..., vn[, else]): `cond` is a struct of bool fields (one per value;
//!   a struct row may not be null: `InvalidInput`, as Arrow); per row the value of the first
//!   true field (a null field counts as false), else the `else` value or null.
//! - choose(index, v0, v1, ...): per row the value `index` selects (any integer type; null gives
//!   null; a valid index outside `[0, n)` is `InvalidInput`, Arrow's IndexError).
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
                .if_else => if (args.len != 3 or args[0].dataType() != .bool) return false,
                .choose => if (!isInteger(args[0].dataType())) return false,
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
            const call = ctx.allocator.alloc(Datum, args.len) catch return error.OutOfMemory;
            defer ctx.allocator.free(call);
            const owned = ctx.allocator.alloc(bool, args.len) catch return error.OutOfMemory;
            defer ctx.allocator.free(owned);
            @memset(owned, false);
            defer for (call, owned) |*c, o| if (o) c.release();
            for (args, call, owned, 0..) |a, *c, *o, i| {
                c.* = a;
                const target: ?DataType = if (i >= first) out_type else if (kind == .choose and a.dataType() != .int64) DataType{ .int64 = {} } else null;
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
