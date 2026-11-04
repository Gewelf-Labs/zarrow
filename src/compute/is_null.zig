//! Null categorization kernels (Arrow C++ "Categorizations"): is_null and is_valid return a bool
//! that is never null; true_unless_null returns true where the input is valid and null elsewhere.
//! Every type with a top-level validity bitmap is accepted (and the null type, whose rows are all
//! null); unions and run-end encoded arrays, whose nulls live in their children, are not. NaN is not
//! null (Arrow's `NullOptions.nan_is_null` is not supported).
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

const Op = enum { is_null, is_valid, true_unless_null };
const bool_type = DataType{ .bool = {} };

/// Register is_null, is_valid and true_unless_null.
pub fn register(registry: *FunctionRegistry) KernelError!void {
    inline for (comptime std.meta.tags(Op)) |op| {
        try registry.registerScalarKernel(@tagName(op), .{
            .signature = .{
                .arity = 1,
                .type_check = hasValidityBitmap,
                .options_check = dense.noOptions,
                .result_type_fn = dense.resultTypeOf(bool_type),
            },
            .exec = Categorize(op).exec,
        });
    }
}

/// Accept one argument whose nulls are its top-level validity (not a union or run-end encoded array).
pub fn hasValidityBitmap(args: []const Datum) bool {
    if (args.len != 1) return false;
    return switch (datatype.storageDataType(args[0].dataType())) {
        .sparse_union, .dense_union, .run_end_encoded => false,
        else => true,
    };
}

fn Categorize(comptime op: Op) type {
    return struct {
        /// Categorize each row of the operand (array, chunked or scalar) by its null-ness.
        fn exec(ctx: *ExecContext, args: []const Datum, _: Options) KernelError!Datum {
            return dense.execElementwise(ctx, args, bool_type, false, chunk);
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

        fn chunk(ctx: *ExecContext, values: []const ExecChunkValue, len: usize) KernelError!ArrayRef {
            const valid = switch (values[0]) {
                .array => |arr| Bits.validity(arr.data()),
                .scalar => |s| Bits.of(!s.isNull()),
            };
            const n = bitmap.byteLength(len);
            // boolArray takes both buffers over, so no errdefer may free them past this point.
            var values_buf = try dense.bitsBuffer(ctx.allocator, len);
            dense.mapBits(1, values_buf.data[0..n], .{valid}, len, f);
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
