//! pivot_wider (aggregate) and hash_pivot_wider (grouped), Arrow C++ "Aggregations":
//! pivot_wider(keys, values) gives a struct with one field per `PivotWiderOptions.key_names`
//! entry, of the values' type; each field holds the non-null value whose key is that name (null
//! when there is none). hash_pivot_wider(keys, values, group_ids) gives one such struct row per
//! group.
//!
//! - Keys: string / large_string / binary / large_binary, fixed_size_binary and int8..uint64;
//!   the key names are matched as the key type (Arrow casts them: an integer name must parse, a
//!   fixed_size_binary name must have the width; else `InvalidOptions`). Values: any type the
//!   take kernels handle.
//! - A null key is `InvalidInput` (Arrow: "pivot key name cannot be null"); a key outside
//!   `key_names` is skipped (`ignore`) or `InvalidInput` (`raise`); a null value is skipped; two
//!   non-null values for one key (of one group) are `InvalidInput` (Arrow: "more than one
//!   non-null value").
//! - Options: key names distinct (else `InvalidOptions`), at most 65536 of them; a grouped call
//!   holds at most 2^28 (group, key) slots, and fixed-width fields of at most 4 GiB in all
//!   (`InvalidInput` past either).
//!
//! The value batches are retained (update / merge lifecycle), each (group, key) keeps the global
//! row of its value, and one take per key builds the fields. Every input chunk passes
//! `validateLayout` before it is read.
const std = @import("std");
const bitmap = @import("../bitmap.zig");
const concat = @import("../concat_array_refs.zig");
const core = @import("core.zig");
const dense = @import("dense.zig");
const hash = @import("hash_aggregate.zig");

const ArrayData = @import("../array/array_data.zig").ArrayData;
const PrimitiveArray = @import("../array/primitive_array.zig").PrimitiveArray;
const ArrayRef = core.ArrayRef;
const DataType = core.DataType;
const Datum = core.Datum;
const ExecContext = core.ExecContext;
const FunctionRegistry = core.FunctionRegistry;
const KernelError = core.KernelError;
const Options = core.Options;

pub const max_keys = 1 << 16;
pub const max_slots = 1 << 28;
pub const max_output_bytes = 1 << 32;

fn IntOf(comptime tag: std.meta.Tag(DataType)) type {
    return switch (tag) {
        .int8 => i8,
        .int16 => i16,
        .int32 => i32,
        .int64 => i64,
        .uint8 => u8,
        .uint16 => u16,
        .uint32 => u32,
        else => u64,
    };
}

fn isKeyType(dt: DataType) bool {
    return switch (dt) {
        .string, .large_string, .binary, .large_binary, .fixed_size_binary => true,
        .int8, .int16, .int32, .int64, .uint8, .uint16, .uint32, .uint64 => true,
        else => false,
    };
}

fn optionsOf(options: Options) core.PivotWiderOptions {
    return if (options == .pivot_wider) options.pivot_wider else .{};
}

fn pivotOptions(options: Options) bool {
    if (options == .none) return true;
    if (options != .pivot_wider) return false;
    const names = options.pivot_wider.key_names;
    if (names.len > max_keys) return false;
    for (names, 0..) |a, i| for (names[i + 1 ..]) |b| {
        if (std.mem.eql(u8, a, b)) return false;
    };
    return true;
}

fn resultTypeAlloc(allocator: std.mem.Allocator, args: []const Datum, options: Options) KernelError!DataType {
    if (args.len < 2) return error.InvalidArity;
    const names = optionsOf(options).key_names;
    const types = allocator.alloc(DataType, names.len) catch return error.OutOfMemory;
    @memset(types, args[1].dataType());
    return dense.structTypeAlloc(allocator, names, types);
}

/// The key names as the key type: the index of the name a key row matches.
const KeyIndex = struct {
    bytes: std.StringHashMapUnmanaged(u32) = .empty,
    ints: std.AutoHashMapUnmanaged(u64, u32) = .empty,

    fn init(allocator: std.mem.Allocator, key_type: DataType, names: []const []const u8) KernelError!KeyIndex {
        var self: KeyIndex = .{};
        errdefer self.deinit(allocator);
        for (names, 0..) |name, i| {
            const id: u32 = @intCast(i);
            const fresh = switch (key_type) {
                .string, .large_string, .binary, .large_binary => try put(allocator, &self.bytes, name, id),
                .fixed_size_binary => |f| blk: {
                    if (name.len != @as(usize, @intCast(f.byte_width))) return error.InvalidOptions;
                    break :blk try put(allocator, &self.bytes, name, id);
                },
                inline .int8, .int16, .int32, .int64, .uint8, .uint16, .uint32, .uint64 => |_, tag| blk: {
                    const T = IntOf(tag);
                    const v = std.fmt.parseInt(T, name, 10) catch return error.InvalidOptions;
                    break :blk try put(allocator, &self.ints, bitsOf(T, v), id);
                },
                else => return error.UnsupportedType,
            };
            // Two names of one key value (an integer written twice): Arrow's duplicate key name.
            if (!fresh) return error.InvalidOptions;
        }
        return self;
    }

    fn put(allocator: std.mem.Allocator, table: anytype, key: anytype, id: u32) KernelError!bool {
        const gop = table.getOrPut(allocator, key) catch return error.OutOfMemory;
        if (gop.found_existing) return false;
        gop.value_ptr.* = id;
        return true;
    }

    fn deinit(self: *KeyIndex, allocator: std.mem.Allocator) void {
        self.bytes.deinit(allocator);
        self.ints.deinit(allocator);
    }

    inline fn bitsOf(comptime T: type, v: T) u64 {
        return if (@typeInfo(T).int.signedness == .signed) @bitCast(@as(i64, v)) else v;
    }

    /// Set `out[i]` to the name index of key row `i` (null: not a key name). A null key row is
    /// `InvalidInput`.
    fn map(self: *const KeyIndex, key_arr: ArrayRef, out: []?u32) KernelError!void {
        const keys = key_arr.data();
        if (keys.hasNulls()) return error.InvalidInput;
        switch (keys.data_type) {
            .string, .large_string, .binary, .large_binary => switch (try dense.StringOperand.init(.{ .array = key_arr })) {
                inline else => |r| for (out, 0..) |*o, i| {
                    o.* = self.bytes.get(r.get(i));
                },
            },
            .fixed_size_binary => |f| {
                const w: usize = @intCast(f.byte_width);
                const vals = keys.buffers[1].data;
                if (w == 0 or vals.len / w < keys.offset + keys.length) return error.InvalidInput;
                for (out, 0..) |*o, i| o.* = self.bytes.get(vals[(keys.offset + i) * w ..][0..w]);
            },
            inline .int8, .int16, .int32, .int64, .uint8, .uint16, .uint32, .uint64 => |_, tag| {
                const T = IntOf(tag);
                const vals = (PrimitiveArray(T){ .data = keys }).values() catch return error.InvalidInput;
                for (out, vals) |*o, v| o.* = self.ints.get(bitsOf(T, v));
            },
            else => return error.UnsupportedType,
        }
    }
};

/// pivot_wider (`grouped` false: arguments keys and values, one group) and hash_pivot_wider
/// (arguments keys, values and uint32 group ids).
fn Pivot(comptime grouped: bool) type {
    return struct {
        const State = struct {
            allocator: std.mem.Allocator,
            names: ?KeyIndex = null,
            key_type: ?DataType = null,
            value_type: ?DataType = null,
            /// The value pieces, retained; a slot row counts across them in order.
            batches: std.ArrayList(ArrayRef) = .empty,
            total: u64 = 0,
            /// Global row of the value of (group, key) at `group * width + key`.
            slots: std.ArrayList(?u64) = .empty,
            groups: usize = 0,
            width: usize,
            raise: bool,

            fn init(allocator: std.mem.Allocator, options: Options) State {
                const o = optionsOf(options);
                return .{ .allocator = allocator, .width = o.key_names.len, .raise = o.unexpected_key_behavior == .raise };
            }

            fn deinit(self: *State) void {
                if (self.names) |*n| n.deinit(self.allocator);
                for (self.batches.items) |*b| b.release();
                self.batches.deinit(self.allocator);
                self.slots.deinit(self.allocator);
            }

            fn grow(self: *State, groups: usize) KernelError!void {
                if (groups <= self.groups) return;
                const n = std.math.mul(usize, groups, self.width) catch return error.InvalidInput;
                if (n > max_slots) return error.InvalidInput;
                self.slots.appendNTimes(self.allocator, null, n - self.slots.items.len) catch return error.OutOfMemory;
                self.groups = groups;
            }

            /// The key index of `key_type` (the first piece sets it; later pieces must match).
            fn namesFor(self: *State, key_type: DataType, options: Options) KernelError!*const KeyIndex {
                if (self.names) |*n| {
                    if (!self.key_type.?.eql(key_type)) return error.InvalidInput;
                    return n;
                }
                self.names = try KeyIndex.init(self.allocator, key_type, optionsOf(options).key_names);
                self.key_type = key_type;
                return &self.names.?;
            }

            fn piece(self: *State, keys: ArrayRef, values: ArrayRef, ids: []const u32, options: Options) KernelError!void {
                const vd = values.data();
                if (self.value_type) |t| {
                    if (!t.eql(vd.data_type)) return error.InvalidInput;
                } else self.value_type = vd.data_type;
                const names = try self.namesFor(keys.data().data_type, options);
                const len = vd.length;
                const key_ids = self.allocator.alloc(?u32, len) catch return error.OutOfMemory;
                defer self.allocator.free(key_ids);
                try names.map(keys, key_ids);
                self.batches.append(self.allocator, values.retain()) catch return error.OutOfMemory;
                const base = self.total;
                self.total += len;
                for (key_ids, 0..) |key, i| {
                    const k = key orelse {
                        if (self.raise) return error.InvalidInput;
                        continue;
                    };
                    if (vd.isNull(i)) continue;
                    const slot = &self.slots.items[@as(usize, if (grouped) ids[i] else 0) * self.width + k];
                    if (slot.* != null) return error.InvalidInput;
                    slot.* = base + i;
                }
            }

            /// Feed every aligned piece of `args`; `bounded`: every group id below the rows.
            fn consume(self: *State, args: []const Datum, options: Options, comptime bounded: bool) KernelError!void {
                if (args.len != @as(usize, if (grouped) 3 else 2)) return error.InvalidArity;
                for (args) |a| if (a != .array and a != .chunked) return error.InvalidInput;
                if (!grouped) try self.grow(1);
                var it = try core.NaryExecChunkIterator.init(self.allocator, args);
                defer it.deinit();
                const limit = it.total_len;
                while (try it.next()) |next| {
                    var chunk = next;
                    defer chunk.deinit();
                    for (chunk.values) |v| v.array.data().validateLayout() catch return error.InvalidInput;
                    var ids: []const u32 = &.{};
                    if (grouped) {
                        const id_data = chunk.values[2].array.data();
                        if (id_data.hasNulls()) return error.InvalidInput;
                        ids = (PrimitiveArray(u32){ .data = id_data }).values() catch return error.InvalidInput;
                        var top: u32 = 0;
                        for (ids) |id| top = @max(top, id);
                        if (ids.len > 0) {
                            if (bounded and top >= limit) return error.InvalidInput;
                            try self.grow(@as(usize, top) + 1);
                        }
                    }
                    try self.piece(chunk.values[0].array, chunk.values[1].array, ids, options);
                }
            }
        };

        fn exec(ctx: *ExecContext, args: []const Datum, options: Options) KernelError!Datum {
            var state = State.init(ctx.allocator, options);
            defer state.deinit();
            if (args.len >= 2) state.value_type = args[1].dataType();
            try state.consume(args, options, true);
            return finish(ctx, &state, options);
        }

        fn init(ctx: *ExecContext, options: Options) KernelError!*anyopaque {
            const state = ctx.allocator.create(State) catch return error.OutOfMemory;
            state.* = State.init(ctx.allocator, options);
            return state;
        }

        fn update(_: *ExecContext, state: *anyopaque, args: []const Datum, options: Options) KernelError!void {
            try cast(state).consume(args, options, false);
        }

        /// Add the slots of `other` (same group ids) after this state's rows: a (group, key)
        /// set in both is `InvalidInput`.
        fn merge(_: *ExecContext, state: *anyopaque, other: *anyopaque, options: Options) KernelError!void {
            const s = cast(state);
            const o = cast(other);
            if (o.key_type) |t| _ = try s.namesFor(t, options);
            if (o.value_type) |t| {
                if (s.value_type) |st| {
                    if (!st.eql(t)) return error.InvalidInput;
                } else s.value_type = t;
            }
            try s.grow(o.groups);
            const base = s.total;
            for (s.slots.items[0..o.slots.items.len], o.slots.items) |*a, b| {
                const row = b orelse continue;
                if (a.* != null) return error.InvalidInput;
                a.* = base + row;
            }
            s.batches.ensureUnusedCapacity(s.allocator, o.batches.items.len) catch return error.OutOfMemory;
            for (o.batches.items) |b| s.batches.appendAssumeCapacity(b.retain());
            s.total += o.total;
        }

        fn finalize(ctx: *ExecContext, state: *anyopaque, options: Options) KernelError!Datum {
            return finish(ctx, cast(state), options);
        }

        fn deinit(ctx: *ExecContext, state: *anyopaque) void {
            const s = cast(state);
            s.deinit();
            ctx.allocator.destroy(s);
        }

        fn cast(state: *anyopaque) *State {
            return @ptrCast(@alignCast(state));
        }

        /// One struct row per group (a struct scalar for pivot_wider): field `k` takes the value
        /// row of (group, k), null when unset.
        fn finish(ctx: *ExecContext, s: *State, options: Options) KernelError!Datum {
            const allocator = ctx.allocator;
            const value_type = s.value_type orelse return error.InvalidInput;
            const n = s.groups;
            // Every (group, key) slot takes the width of a fixed-width value, null or not: a type
            // parameter, not the input, sets it, so the output is bounded here.
            const width: u64 = switch (value_type) {
                .fixed_size_binary => |f| @intCast(@max(f.byte_width, 0)),
                else => (value_type.bitWidth() orelse 8) / 8,
            };
            const bytes = std.math.mul(u64, @as(u64, n) * s.width, width) catch return error.InvalidInput;
            if (bytes > max_output_bytes) return error.InvalidInput;
            var all = if (s.batches.items.len == 1)
                s.batches.items[0].retain()
            else if (s.batches.items.len == 0)
                try dense.nullArray(allocator, value_type, 0)
            else
                concat.concatArrayRefs(allocator, value_type, s.batches.items) catch |err| return switch (err) {
                    error.OutOfMemory => error.OutOfMemory,
                    else => error.InvalidInput,
                };
            defer all.release();
            const children = allocator.alloc(ArrayRef, s.width) catch return error.OutOfMemory;
            defer allocator.free(children);
            var made: usize = 0;
            errdefer for (children[0..made]) |*c| c.release();
            const indices = allocator.alloc(?usize, n) catch return error.OutOfMemory;
            defer allocator.free(indices);
            for (children, 0..) |*c, k| {
                for (indices, 0..) |*ix, g| ix.* = if (s.slots.items[g * s.width + k]) |r| @intCast(r) else null;
                var taken = try core.datumTakeNullable(.{ .array = all }, indices);
                c.* = taken.asArray() orelse {
                    taken.release();
                    return error.InvalidInput;
                };
                made += 1;
            }
            made = 0; // structArray takes the children over, also on error.
            var out = try dense.structArray(allocator, optionsOf(options).key_names, children, n, null);
            if (grouped) return .{ .array = out };
            defer out.release();
            return .{ .scalar = try dense.scalarAt0(out) };
        }
    };
}

fn scalarArgs(args: []const Datum) bool {
    return args.len == 2 and isKeyType(args[0].dataType());
}

fn groupedArgs(args: []const Datum) bool {
    return args.len == 3 and isKeyType(args[0].dataType()) and args[2].dataType() == .uint32;
}

/// Register pivot_wider and hash_pivot_wider.
pub fn register(registry: *FunctionRegistry) KernelError!void {
    inline for (.{ false, true }) |grouped| {
        const P = Pivot(grouped);
        try registry.registerKernel(if (grouped) "hash_pivot_wider" else "pivot_wider", if (grouped) .hash_aggregate else .aggregate, .{
            .signature = .{ .arity = if (grouped) 3 else 2, .type_check = if (grouped) groupedArgs else scalarArgs, .options_check = pivotOptions, .result_type_alloc_fn = resultTypeAlloc },
            .exec = P.exec,
            .aggregate_lifecycle = .{ .init = P.init, .update = P.update, .merge = P.merge, .finalize = P.finalize, .deinit = P.deinit },
        });
    }
}
