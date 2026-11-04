//! Logical kernels on bool operands (Arrow C++ "Logical functions"): and, or, xor, and_not and
//! invert propagate nulls (intersection); and_kleene and or_kleene use Kleene three-valued logic
//! (`false and null` is false, `true or null` is true, otherwise a null operand gives null).
const std = @import("std");
const bitmap = @import("../bitmap.zig");
const core = @import("core.zig");
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
const OwnedBuffer = @import("../buffer.zig").OwnedBuffer;

const Op = enum { @"and", @"or", xor, and_not, invert, and_kleene, or_kleene };
const bool_type = DataType{ .bool = {} };

/// Register and, or, xor, and_not, invert, and_kleene and or_kleene.
pub fn register(registry: *FunctionRegistry) KernelError!void {
    inline for (comptime std.meta.tags(Op)) |op| {
        const arity: usize = if (op == .invert) 1 else 2;
        try registry.registerScalarKernel(@tagName(op), .{
            .signature = .{
                .arity = arity,
                .type_check = dense.argsOfType(.bool, arity),
                .options_check = dense.noOptions,
                .result_type_fn = dense.resultTypeOf(bool_type),
            },
            .exec = Logical(op).exec,
        });
    }
}

fn Logical(comptime op: Op) type {
    const kleene = op == .and_kleene or op == .or_kleene;
    return struct {
        /// Apply the logical operation element-wise (array, chunked or scalar operands).
        fn exec(ctx: *ExecContext, args: []const Datum, _: Options) KernelError!Datum {
            return dense.execElementwise(ctx, args, bool_type, !kleene, chunk);
        }

        fn valueBits(v: ExecChunkValue) KernelError!Bits {
            return switch (v) {
                .array => |arr| Bits.values(arr.data()),
                .scalar => |s| Bits.of(!s.isNull() and try dense.scalarAs(bool, s)),
            };
        }

        fn validBits(v: ExecChunkValue) Bits {
            return switch (v) {
                .array => |arr| Bits.validity(arr.data()),
                .scalar => |s| Bits.of(!s.isNull()),
            };
        }

        fn value(in: [2]u8) u8 {
            return switch (op) {
                .@"and", .and_kleene => in[0] & in[1],
                .@"or", .or_kleene => in[0] | in[1],
                .xor => in[0] ^ in[1],
                .and_not => in[0] & ~in[1],
                .invert => ~in[0],
            };
        }

        /// Compute the Kleene validity from (a, valid a, b, valid b): both known, or one side decides alone.
        fn kleeneValid(in: [4]u8) u8 {
            const a = in[0];
            const va = in[1];
            const b = in[2];
            const vb = in[3];
            return if (op == .and_kleene)
                (va & vb) | (va & ~a) | (vb & ~b)
            else
                (va & vb) | (va & a) | (vb & b);
        }

        fn chunk(ctx: *ExecContext, values: []const ExecChunkValue, len: usize) KernelError!ArrayRef {
            if (op == .invert) {
                var out = try dense.DenseBinaryOutput(bool).initValidity(ctx.allocator, len, dense.arrayData(values[0]), null, bool_type);
                errdefer out.deinit();
                dense.mapBits(2, out.values(), .{ try valueBits(values[0]), Bits.of(true) }, len, value);
                return out.finish() catch error.OutOfMemory;
            }
            const a = try valueBits(values[0]);
            const b = try valueBits(values[1]);
            if (!kleene) {
                var out = try dense.DenseBinaryOutput(bool).initValidity(ctx.allocator, len, dense.arrayData(values[0]), dense.arrayData(values[1]), bool_type);
                errdefer out.deinit();
                dense.mapBits(2, out.values(), .{ a, b }, len, value);
                return out.finish() catch error.OutOfMemory;
            }
            // boolArray takes both buffers over, so no errdefer may free them past this point.
            var values_buf = try dense.bitsBuffer(ctx.allocator, len);
            dense.mapBits(2, values_buf.data[0..bitmap.byteLength(len)], .{ a, b }, len, value);
            const va = validBits(values[0]);
            const vb = validBits(values[1]);
            var validity: ?OwnedBuffer = null;
            if (va.bytes != null or vb.bytes != null or va.splat != 0xFF or vb.splat != 0xFF) {
                var v = dense.bitsBuffer(ctx.allocator, len) catch |err| {
                    values_buf.deinit();
                    return err;
                };
                dense.mapBits(4, v.data[0..bitmap.byteLength(len)], .{ a, va, b, vb }, len, kleeneValid);
                validity = v;
            }
            return dense.boolArray(ctx.allocator, len, values_buf, validity);
        }
    };
}
