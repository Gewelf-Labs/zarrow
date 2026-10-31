const std = @import("std");
const bitmap = @import("../bitmap.zig");
const buffer = @import("../buffer.zig");
const array_utils = @import("array_utils.zig");
const array_ref = @import("array_ref.zig");
const builder_state = @import("builder_state.zig");
const datatype = @import("../datatype.zig");
const array_data = @import("array_data.zig");

// Generic fixed-width primitive array views/builders.

const SharedBuffer = buffer.SharedBuffer;
const OwnedBuffer = buffer.OwnedBuffer;
const DataType = datatype.DataType;
const ArrayData = array_data.ArrayData;
const AccessorError = array_data.AccessorError;
const ArrayRef = array_ref.ArrayRef;
const BuilderState = builder_state.BuilderState;

/// Generic array view for fixed-width primitive types.
pub fn PrimitiveArray(comptime T: type) type {
    return struct {
        data: *const ArrayData,

        const Self = @This();

        pub const Error = AccessorError;

        /// Return the logical length.
        pub fn len(self: Self) usize {
            return self.data.length;
        }

        /// Check whether the element at index is null.
        pub fn isNull(self: Self, i: usize) bool {
            return self.data.isNull(i);
        }

        pub fn values(self: Self) Error![]const T {
            // Primitive arrays require [validity] and [values] buffers.
            const raw = try self.data.typedBufferAt(1, T);
            const start = self.data.offset;
            const end = try self.data.logicalEnd();
            if (end > raw.len) return error.BufferTooSmall;
            return raw[start..end];
        }

        /// Return the logical value view at the requested index.
        pub fn value(self: Self, i: usize) Error!T {
            if (i >= self.data.length) return error.IndexOutOfBounds;
            return (try self.values())[i];
        }
    };
}

/// Generic builder for fixed-width primitive arrays.
/// The caller must specify the data type and handle validity bits as needed.
pub fn PrimitiveBuilder(comptime T: type, comptime dtype: DataType) type {
    return struct {
        allocator: std.mem.Allocator,
        values: OwnedBuffer,
        validity: ?OwnedBuffer = null,
        buffers: [2]SharedBuffer = undefined,
        len: usize = 0,
        null_count: usize = 0,
        state: BuilderState = .ready,

        const Self = @This();

        const Error = error{ AlreadyFinished, NotFinished };

        const TYPE: DataType = dtype;

        /// Initialize and return a new instance.
        pub fn init(allocator: std.mem.Allocator, capacity: usize) !Self {
            return .{
                .allocator = allocator,
                .values = try OwnedBuffer.init(allocator, capacity * @sizeOf(T)),
            };
        }

        /// Release resources owned by this instance.
        pub fn deinit(self: *Self) void {
            self.values.deinit();
            if (self.validity) |*valid| valid.deinit();
        }

        /// Reset state while retaining reusable capacity when possible.
        pub fn reset(self: *Self) Error!void {
            if (self.state != .finished) return Error.NotFinished;
            self.len = 0;
            self.null_count = 0;
            self.state = .ready;
        }

        /// Clear state and release reusable buffers when required.
        pub fn clear(self: *Self) Error!void {
            if (self.state != .finished) return Error.NotFinished;
            self.values.deinit();
            if (self.validity) |*valid| valid.deinit();
            self.validity = null;
            self.len = 0;
            self.null_count = 0;
            self.state = .ready;
        }

        fn ensureValuesCapacity(self: *Self, new_len: usize) !void {
            const capacity = self.values.len() / @sizeOf(T);
            if (new_len <= capacity) return;
            try self.values.resize(new_len * @sizeOf(T));
        }

        /// Append one logical value into the builder.
        pub fn append(self: *Self, value: T) !void {
            if (self.state == .finished) return Error.AlreadyFinished;
            const next_len = self.len + 1;
            try self.ensureValuesCapacity(next_len);
            const slice = std.mem.bytesAsSlice(T, self.values.data);
            slice[self.len] = value;
            try array_utils.setValidBit(&self.validity, self.len);
            self.len = next_len;
        }

        /// Ensure capacity for `additional` more values without further reallocation.
        pub fn reserve(self: *Self, additional: usize) !void {
            try self.ensureValuesCapacity(self.len + additional);
            if (self.validity) |*valid| try array_utils.ensureBitmapCapacity(valid, self.len + additional);
        }

        /// Append non-null values in bulk (one memcpy when no validity bitmap exists yet).
        pub fn appendSlice(self: *Self, items: []const T) !void {
            if (self.state == .finished) return Error.AlreadyFinished;
            try self.reserve(items.len);
            const slice = std.mem.bytesAsSlice(T, self.values.data);
            @memcpy(slice[self.len..][0..items.len], items);
            if (self.validity) |*valid| {
                for (self.len..self.len + items.len) |i| bitmap.setBit(valid.data, i);
            }
            self.len += items.len;
        }

        /// Append a null entry into the builder.
        pub fn appendNull(self: *Self) !void {
            if (self.state == .finished) return Error.AlreadyFinished;
            const next_len = self.len + 1;
            try self.ensureValuesCapacity(next_len);
            try array_utils.ensureValidityForNull(self.allocator, &self.validity, &self.null_count, next_len);
            self.len = next_len;
        }

        /// Finalize builder state and return an immutable array reference.
        pub fn finish(self: *Self) !ArrayRef {
            if (self.state == .finished) return Error.AlreadyFinished;
            const validity_buf = try array_utils.takeValidity(&self.validity, self.len);
            self.buffers[0] = validity_buf;
            self.buffers[1] = try self.values.toShared(self.len * @sizeOf(T));

            const data = ArrayData{
                .data_type = TYPE,
                .length = self.len,
                .null_count = self.null_count,
                .buffers = self.buffers[0..],
            };

            self.state = .finished;
            return ArrayRef.fromOwnedInlineBuffers(self.allocator, data);
        }

        /// Finalize output and then reset builder state for reuse.
        pub fn finishReset(self: *Self) !ArrayRef {
            const finished_len = self.len;
            const finished_ref = try self.finish();
            try self.reset();
            // Batches built in a loop are usually the same size: start the next one at that capacity
            // (one allocation instead of ~log2(len) regrowths). On OOM it just grows lazily.
            if (finished_len > 0) {
                if (OwnedBuffer.init(self.allocator, finished_len * @sizeOf(T))) |buf| self.values = buf else |_| {}
            }
            return finished_ref;
        }
    };
}

test "primitive builder appends values and nulls" {
    var builder = try PrimitiveBuilder(i32, DataType{ .int32 = {} }).init(std.testing.allocator, 2);
    defer builder.deinit();

    try builder.append(10);
    try builder.appendNull();
    try builder.append(30);

    var array_handle = try builder.finish();
    defer array_handle.release();
    const built = PrimitiveArray(i32){ .data = array_handle.data() };
    try std.testing.expectEqual(@as(usize, 3), built.len());
    try std.testing.expect(built.isNull(1));
    try std.testing.expectEqual(@as(i32, 30), try built.value(2));
    try std.testing.expectError(error.IndexOutOfBounds, built.value(3));
}

test "primitive builder appendSlice matches per-element append" {
    var builder = try PrimitiveBuilder(i32, DataType{ .int32 = {} }).init(std.testing.allocator, 0);
    defer builder.deinit();

    try builder.appendSlice(&.{ 1, 2 });
    try builder.appendNull();
    try builder.appendSlice(&.{ 4, 5, 6 });

    var array_handle = try builder.finish();
    defer array_handle.release();
    const built = PrimitiveArray(i32){ .data = array_handle.data() };
    try std.testing.expectEqual(@as(usize, 6), built.len());
    try std.testing.expect(built.isNull(2));
    try std.testing.expect(!built.isNull(3) and !built.isNull(5));
    try std.testing.expectEqual(@as(i32, 6), try built.value(5));
    try std.testing.expectEqual(@as(usize, 1), array_handle.data().null_count.?);
}

test "primitive array values returns error for missing value buffer" {
    const buffers = [_]SharedBuffer{SharedBuffer.empty};
    const data = ArrayData{
        .data_type = DataType{ .int32 = {} },
        .length = 1,
        .null_count = 0,
        .buffers = &buffers,
    };
    const view = PrimitiveArray(i32){ .data = &data };
    try std.testing.expectError(error.InvalidBufferCount, view.values());
}
