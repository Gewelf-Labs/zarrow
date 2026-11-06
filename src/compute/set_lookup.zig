//! Set lookup (Arrow C++ "Containment tests"): is_in (bool) and index_in (int32) with
//! `SetLookupOptions`. The value set (an array or chunked array) is hashed once per call into a
//! `KeyMap` (hash_aggregate.zig), then every input row is looked up. A value set of another type
//! is cast to the input's type (safe cast); when that cast fails, the input is cast to the value
//! set's type instead (as pyarrow 26: an int64 value set [1, 300] against int8 rows, or a float
//! value set [1.5] against int32 rows); when neither cast exists the call is `InvalidOptions`. index_in returns the position of the
//! first occurrence in the value set. Inputs: every key type of hash_aggregate.zig (numbers, bool,
//! dates, timestamps, times, durations, string and binary layouts); floats match by bits after
//! folding every NaN into one (NaN matches NaN, -0.0 does not match 0.0, as in Arrow).
//!
//! Nulls follow `null_matching_behavior` (Arrow C++): `match` (a null input matches a null of
//! the value set: is_in true, index_in its position; is_in false / index_in null without one),
//! `skip` (nulls never match: is_in false, index_in null), `emit_null` (a null input gives null)
//! and `inconclusive` (like emit_null; also null for a value not found when the value set holds a
//! null).
const std = @import("std");
const bitmap = @import("../bitmap.zig");
const core = @import("core.zig");
const dense = @import("dense.zig");
const hash = @import("hash_aggregate.zig");

const ArrayRef = core.ArrayRef;
const DataType = core.DataType;
const Datum = core.Datum;
const ExecChunkValue = core.ExecChunkValue;
const ExecContext = core.ExecContext;
const FunctionRegistry = core.FunctionRegistry;
const KernelError = core.KernelError;
const Options = core.Options;
const SetLookupOptions = core.SetLookupOptions;

const bool_type = DataType{ .bool = {} };
const int32_type = DataType{ .int32 = {} };

/// The lookup functions, by Arrow name.
const Op = enum { is_in, index_in };

/// Register is_in and index_in for every key type.
pub fn register(registry: *FunctionRegistry) KernelError!void {
    inline for (comptime std.meta.tags(Op)) |op| {
        try registry.registerScalarKernel(@tagName(op), .{
            .signature = .{ .arity = 1, .type_check = keyArg, .options_check = lookupOptions, .result_type_fn = dense.resultTypeOf(if (op == .is_in) bool_type else int32_type) },
            .exec = Lookup(op).exec,
        });
        try registry.setImplicitCast(@tagName(op), .scalar, .decode);
    }
}

fn keyArg(args: []const Datum) bool {
    return args.len == 1 and hash.isKeyType(args[0].dataType());
}

/// `SetLookupOptions` whose value set is an array or a chunked array.
fn lookupOptions(options: Options) bool {
    return options == .set_lookup and options.set_lookup.value_set != .scalar;
}

/// The hashed value set of one call.
const Set = struct {
    map: hash.KeyMap = .{},
    /// Value set position of the first row of every id.
    positions: std.ArrayList(i32) = .empty,
    /// Id of the null of the value set (null when it has none or nulls are skipped).
    null_id: ?u32 = null,
    has_null: bool = false,
    behavior: core.NullMatchingBehavior,

    fn deinit(self: *Set, allocator: std.mem.Allocator) void {
        self.map.deinit(allocator);
        self.positions.deinit(allocator);
    }

    fn build(self: *Set, allocator: std.mem.Allocator, value_set: Datum) KernelError!void {
        const parts: []const ArrayRef = switch (value_set) {
            .array => |*a| a[0..1],
            .chunked => |c| c.chunks(),
            .scalar, .record_batch, .table => return error.InvalidOptions,
        };
        var row: usize = 0;
        var ids: std.ArrayList(u32) = .empty;
        defer ids.deinit(allocator);
        for (parts) |part| {
            part.data().validateLayout() catch return error.InvalidInput;
            ids.resize(allocator, part.data().length) catch return error.OutOfMemory;
            try self.map.map(allocator, part, ids.items, .insert);
            for (ids.items) |id| {
                if (id == self.positions.items.len) self.positions.append(allocator, std.math.cast(i32, row) orelse return error.Overflow) catch return error.OutOfMemory;
                row += 1;
            }
        }
        try self.map.freeze(allocator);
        self.has_null = self.map.null_id != null;
        if (self.behavior != .skip) self.null_id = self.map.null_id;
        // A null input is looked up through `null_id` below, never through the map.
        self.map.null_id = null;
    }
};

fn Lookup(comptime op: Op) type {
    return struct {
        /// Look up each row (array, chunked or scalar) in the value set.
        fn exec(ctx: *ExecContext, args: []const Datum, options: Options) KernelError!Datum {
            if (args.len != 1) return error.InvalidArity;
            const o = options.set_lookup;
            const in_type = args[0].dataType();
            const set_type = o.value_set.dataType();
            if (set_type.eql(in_type)) return run(ctx, args[0], o.value_set, o.null_matching_behavior);
            if (ctx.invoke("cast", .scalar, &.{o.value_set}, .{ .cast = .{ .to_type = in_type } })) |cast_set| {
                var vs = cast_set;
                defer vs.release();
                return run(ctx, args[0], vs, o.null_matching_behavior);
            } else |err| if (!castFailed(err)) return err;
            if (!hash.isKeyType(set_type)) return error.InvalidOptions;
            var input = ctx.invoke("cast", .scalar, &.{args[0]}, .{ .cast = .{ .to_type = set_type } }) catch |err| {
                return if (castFailed(err)) error.InvalidOptions else err;
            };
            defer input.release();
            return run(ctx, input, o.value_set, o.null_matching_behavior);
        }

        fn castFailed(err: KernelError) bool {
            return switch (err) {
                error.InvalidCast, error.UnsupportedType, error.NoMatchingKernel, error.NotImplemented => true,
                else => false,
            };
        }

        fn run(ctx: *ExecContext, input: Datum, value_set: Datum, behavior: core.NullMatchingBehavior) KernelError!Datum {
            if (value_set == .scalar) return error.InvalidOptions;
            var set: Set = .{ .behavior = behavior };
            defer set.deinit(ctx.allocator);
            try set.build(ctx.allocator, value_set);
            // The chunks read the prebuilt set through the custom options payload.
            const inner: Options = .{ .custom = .{ .tag = "set_lookup", .payload = &set } };
            return dense.execElementwiseWith(ctx, &.{input}, inner, if (op == .is_in) bool_type else int32_type, false, chunk);
        }

        fn chunk(ctx: *ExecContext, values: []const ExecChunkValue, len: usize, options: Options) KernelError!ArrayRef {
            const set: *const Set = @ptrCast(@alignCast(options.custom.payload.?));
            var arr = try dense.chunkArray(ctx.allocator, values[0]);
            defer arr.release();
            const data = arr.data();
            const ids = ctx.allocator.alloc(u32, len) catch return error.OutOfMemory;
            defer ctx.allocator.free(ids);
            var m = set.map; // a lookup reads the map only
            try m.map(ctx.allocator, arr, ids, .lookup);

            var validity = try dense.bitsBuffer(ctx.allocator, len);
            @memset(validity.data[0..bitmap.byteLength(len)], 0xFF);
            var out = dense.DenseBinaryOutput(if (op == .is_in) bool else i32).initValidity(ctx.allocator, len, null, null, if (op == .is_in) bool_type else int32_type) catch |err| {
                validity.deinit();
                return err;
            };
            out.validity = validity;
            errdefer out.deinit();
            const dst = out.values();
            if (op == .is_in) @memset(dst, 0);
            const bits = dense.validityBits(data);
            const valid_bits = out.validity.?.data;
            for (ids, 0..) |found, i| {
                const is_null = if (bits) |b| !bitmap.bitIsSet(b, data.offset + i) else false;
                // The matching id, or null when the row is null.
                const id: ?u32 = if (!is_null) found else switch (set.behavior) {
                    .match => set.null_id orelse hash.missing,
                    .skip => hash.missing,
                    .emit_null, .inconclusive => null,
                };
                const hit = id != null and id.? != hash.missing;
                if (op == .is_in) {
                    if (id == null or (!hit and set.behavior == .inconclusive and set.has_null)) {
                        bitmap.clearBit(valid_bits, i);
                    } else if (hit) bitmap.setBit(dst, i);
                } else if (hit) {
                    dst[i] = set.positions.items[id.?];
                } else {
                    dst[i] = 0;
                    bitmap.clearBit(valid_bits, i);
                }
            }
            if (bitmap.countUnsetBit(valid_bits, len) == 0) {
                out.validity.?.deinit();
                out.validity = null;
            }
            return out.finish() catch error.OutOfMemory;
        }
    };
}
