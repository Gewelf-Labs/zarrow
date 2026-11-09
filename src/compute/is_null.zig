//! Categorization kernels (Arrow C++ "Categorizations"): is_null and is_valid return a bool that
//! is never null; true_unless_null returns true where the input is valid and null elsewhere.
//! Every type is accepted. The nulls are Arrow's logical nulls: the top-level validity bitmap,
//! every row for the null type, and for the types without one the child's (`logicalValidity`: a
//! union row is null when its selected child's row is, a run-end encoded row when its run's value
//! is). is_null takes `NullOptions`: with `nan_is_null` a float NaN is null too (other types
//! ignore it).
//!
//! is_nan, is_finite and is_inf on float16 / float32 / float64 (IEEE 754), the integer types, the decimal
//! types and durations (every value finite, none NaN or infinite) and the null type (all null), as in
//! Arrow C++ and pyarrow 26; null handling is intersection.
const std = @import("std");
const bitmap = @import("../bitmap.zig");
const core = @import("core.zig");
const datatype = @import("../datatype.zig");
const dense = @import("dense.zig");

const ArrayRef = core.ArrayRef;
const DataType = core.DataType;
const Datum = core.Datum;
const ExecChunkValue = core.ExecChunkValue;
const ExecContext = core.ExecContext;
const FunctionRegistry = core.FunctionRegistry;
const KernelError = core.KernelError;
const Options = core.Options;
const Bits = dense.Bits;
const ArrayData = @import("../array/array_data.zig").ArrayData;
const OwnedBuffer = @import("../buffer.zig").OwnedBuffer;
const PrimitiveArray = @import("../array/primitive_array.zig").PrimitiveArray;
const concat = @import("../concat_array_refs.zig");

const Op = enum { is_null, is_valid, true_unless_null };
const FloatOp = enum { is_nan, is_finite, is_inf };
const bool_type = DataType{ .bool = {} };
const TypeTag = std.meta.Tag(DataType);

/// Register is_null, is_valid and true_unless_null.
pub fn register(registry: *FunctionRegistry) KernelError!void {
    inline for (comptime std.meta.tags(Op)) |op| {
        try registry.registerScalarKernel(@tagName(op), .{
            .signature = .{
                .arity = 1,
                .type_check = oneArg,
                .options_check = if (op == .is_null) nullOptions else dense.noOptions,
                .result_type_fn = dense.resultTypeOf(bool_type),
            },
            .exec = Categorize(op).exec,
        });
    }
    inline for (comptime std.meta.tags(FloatOp)) |op| {
        inline for (dense.numeric_types) |T| try registerFloatOp(registry, op, comptime std.meta.activeTag(dense.dataTypeOf(T)));
        inline for (.{ TypeTag.half_float, TypeTag.decimal32, TypeTag.decimal64, TypeTag.decimal128, TypeTag.decimal256, TypeTag.duration, TypeTag.null }) |tag| try registerFloatOp(registry, op, tag);
    }
}

fn registerFloatOp(registry: *FunctionRegistry, comptime op: FloatOp, comptime tag: TypeTag) KernelError!void {
    try registry.registerScalarKernel(@tagName(op), .{
        .signature = .{ .arity = 1, .type_check = dense.argsOfType(tag, 1), .options_check = dense.noOptions, .result_type_fn = dense.resultTypeOf(bool_type) },
        .exec = FloatCheck(op).exec,
    });
}

/// `none` or `NullOptions`.
fn nullOptions(options: Options) bool {
    return options == .none or options == .null_options;
}

/// Set bit `i` of `bits` for every row `i < len` of the float operand `v` that is NaN.
fn orNanBits(comptime T: type, bits: []u8, v: ExecChunkValue, len: usize) KernelError!void {
    switch (try dense.operand(T, v)) {
        inline else => |x| for (0..len) |i| {
            if (std.math.isNan(dense.at(x, i))) bits[i / 8] |= @as(u8, 1) << @intCast(i % 8);
        },
    }
}

fn FloatCheck(comptime op: FloatOp) type {
    return struct {
        /// Test each row of a numeric, decimal or null operand (array, chunked or scalar).
        fn exec(ctx: *ExecContext, args: []const Datum, _: Options) KernelError!Datum {
            return dense.execElementwise(ctx, args, bool_type, true, chunk);
        }

        fn test1(x: anytype) bool {
            return switch (op) {
                .is_nan => std.math.isNan(x),
                .is_finite => std.math.isFinite(x),
                .is_inf => std.math.isInf(x),
            };
        }

        fn chunk(ctx: *ExecContext, values: []const ExecChunkValue, len: usize) KernelError!ArrayRef {
            const dt = values[0].dataType();
            if (dt == .null) return dense.nullArray(ctx.allocator, bool_type, len);
            var out = try dense.DenseBinaryOutput(bool).initValidity(ctx.allocator, len, dense.arrayData(values[0]), null, bool_type);
            errdefer out.deinit();
            const bits = out.values();
            @memset(bits, 0);
            switch (dt) {
                inline .half_float, .float, .double => |_, tag| switch (try dense.operand(if (tag == .half_float) f16 else if (tag == .float) f32 else f64, values[0])) {
                    inline else => |x| for (0..len) |i| {
                        if (test1(dense.at(x, i))) bits[i / 8] |= @as(u8, 1) << @intCast(i % 8);
                    },
                },
                // Integers and decimals: every value is finite, none is NaN or infinite.
                else => if (op == .is_finite) for (0..len) |i| {
                    bits[i / 8] |= @as(u8, 1) << @intCast(i % 8);
                },
            }
            return out.finish() catch error.OutOfMemory;
        }
    };
}

/// Accept one argument whose nulls are its top-level validity (not a union or run-end encoded array).
pub fn hasValidityBitmap(args: []const Datum) bool {
    if (args.len != 1) return false;
    return switch (datatype.storageDataType(args[0].dataType())) {
        .sparse_union, .dense_union, .run_end_encoded => false,
        else => true,
    };
}

/// Accept one argument of any type.
pub fn oneArg(args: []const Datum) bool {
    return args.len == 1;
}

/// The logical validity of the rows of `data` (passed `validateLayout`) as packed bits, bit `i`
/// for row `i`: the top-level validity, none for the null type, and for a union or a run-end
/// encoded array that of its children (a union row is valid when its selected child's row is,
/// a run-end encoded row when its run's value is; recursively). The caller frees the buffer.
pub fn logicalValidity(allocator: std.mem.Allocator, data: *const ArrayData) KernelError!OwnedBuffer {
    try validateChildren(data);
    var out = try dense.bitsBuffer(allocator, data.length);
    errdefer out.deinit();
    @memset(out.data, 0);
    for (0..data.length) |i| if (try rowValid(data, i)) bitmap.setBit(out.data, i);
    return out;
}

/// `validateLayout` on the children of a union or run-end encoded array (recursively): their
/// validity is read by `rowValid`.
fn validateChildren(data: *const ArrayData) KernelError!void {
    switch (datatype.storageDataType(data.data_type)) {
        .sparse_union, .dense_union, .run_end_encoded => for (data.children) |c| {
            c.data().validateLayout() catch return error.InvalidInput;
            try validateChildren(c.data());
        },
        else => {},
    }
}

/// Whether logical row `i` of `data` is valid (see `logicalValidity`).
// ponytail: a binary search over the run ends per row; walk the runs if REE validity gets hot.
fn rowValid(data: *const ArrayData, i: usize) KernelError!bool {
    const st = datatype.storageDataType(data.data_type);
    switch (st) {
        .null => return false,
        .sparse_union, .dense_union => |u| {
            const pos = data.offset + i;
            const ids = data.typedBufferAt(0, i8) catch return error.InvalidInput;
            if (pos >= ids.len) return error.InvalidInput;
            const table = concat.unionChildTable(u) catch return error.InvalidInput;
            const k = table[@as(u8, @bitCast(ids[pos]))];
            if (k >= data.children.len) return error.InvalidInput;
            const child = data.children[k].data();
            const row = if (st == .dense_union) blk: {
                const offsets = data.typedBufferAt(1, i32) catch return error.InvalidInput;
                if (pos >= offsets.len) return error.InvalidInput;
                break :blk std.math.cast(usize, offsets[pos]) orelse return error.InvalidInput;
            } else pos;
            if (row >= child.length) return error.InvalidInput;
            return rowValid(child, row);
        },
        .run_end_encoded => |r| {
            if (data.children.len != 2) return error.InvalidInput;
            const ends = data.children[0].data();
            const k = switch (r.run_end_type.bit_width) {
                inline 16, 32, 64 => |w| blk: {
                    const vals = (PrimitiveArray(@Int(.signed, w)){ .data = ends }).values() catch return error.InvalidInput;
                    break :blk concat.findRun(@Int(.signed, w), vals, data.offset + i) catch return error.InvalidInput;
                },
                else => return error.InvalidInput,
            };
            const values = data.children[1].data();
            if (k >= values.length) return error.InvalidInput;
            return rowValid(values, k);
        },
        else => return !data.isNull(i),
    }
}

/// The logical null rows of `data` (passed `validateLayout`; see `logicalValidity`).
pub fn logicalNullCount(data: *const ArrayData) KernelError!usize {
    try validateChildren(data);
    var nulls: usize = 0;
    for (0..data.length) |i| nulls += @intFromBool(!try rowValid(data, i));
    return nulls;
}

/// Whether the nulls of `dt` are its top-level validity (not a union or run-end encoded type).
pub fn plainValidity(dt: DataType) bool {
    return switch (datatype.storageDataType(dt)) {
        .sparse_union, .dense_union, .run_end_encoded => false,
        else => true,
    };
}

fn Categorize(comptime op: Op) type {
    return struct {
        /// Categorize each row of the operand (array, chunked or scalar) by its null-ness.
        fn exec(ctx: *ExecContext, args: []const Datum, options: Options) KernelError!Datum {
            return dense.execElementwiseWith(ctx, args, options, bool_type, false, chunk);
        }

        fn f(in: [1]u8) u8 {
            return switch (op) {
                .is_null => ~in[0],
                .is_valid => in[0],
                .true_unless_null => 0xFF,
            };
        }

        fn copy(in: [1]u8) u8 {
            return in[0];
        }

        fn chunk(ctx: *ExecContext, values: []const ExecChunkValue, len: usize, options: Options) KernelError!ArrayRef {
            var logical: ?OwnedBuffer = null;
            defer if (logical) |*l| l.deinit();
            const valid = switch (values[0]) {
                .array => |arr| if (plainValidity(arr.data().data_type)) Bits.validity(arr.data()) else blk: {
                    logical = try logicalValidity(ctx.allocator, arr.data());
                    break :blk Bits{ .bytes = logical.?.data };
                },
                .scalar => |s| Bits.of(!s.isNull()),
            };
            const n = bitmap.byteLength(len);
            // boolArray takes both buffers over, so no errdefer may free them past this point.
            var values_buf = try dense.bitsBuffer(ctx.allocator, len);
            dense.mapBits(1, values_buf.data[0..n], .{valid}, len, f);
            if (op == .is_null and options == .null_options and options.null_options.nan_is_null) {
                // A NaN in a null row is already true; a null scalar has no value to look at.
                const v = values[0];
                if (!(v == .scalar and v.scalar.isNull())) switch (v.dataType()) {
                    .float => orNanBits(f32, values_buf.data[0..n], v, len) catch |err| {
                        values_buf.deinit();
                        return err;
                    },
                    .half_float => orNanBits(f16, values_buf.data[0..n], v, len) catch |err| {
                        values_buf.deinit();
                        return err;
                    },
                    .double => orNanBits(f64, values_buf.data[0..n], v, len) catch |err| {
                        values_buf.deinit();
                        return err;
                    },
                    else => {},
                };
            }
            var validity: ?@TypeOf(values_buf) = null;
            if (op == .true_unless_null and (valid.bytes != null or valid.splat != 0xFF)) {
                var v = dense.bitsBuffer(ctx.allocator, len) catch |err| {
                    values_buf.deinit();
                    return err;
                };
                dense.mapBits(1, v.data[0..n], .{valid}, len, copy);
                validity = v;
            }
            return dense.boolArray(ctx.allocator, len, values_buf, validity);
        }
    };
}
