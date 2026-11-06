const std = @import("std");
const chunked_array_mod = @import("../chunked_array.zig");
const core = @import("core.zig");

const Datum = core.Datum;
const KernelError = core.KernelError;

pub fn mapArrayReadError(err: anyerror) KernelError {
    return switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        else => error.InvalidInput,
    };
}

pub fn mapChunkedError(err: chunked_array_mod.Error) KernelError {
    return switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        error.LengthOverflow => error.Overflow,
        else => error.InvalidInput,
    };
}

pub fn mapConcatError(err: anyerror) KernelError {
    return switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        error.UnsupportedType => error.UnsupportedType,
        else => error.InvalidInput,
    };
}

pub fn inferDatumAllocator(datum: Datum) ?std.mem.Allocator {
    return switch (datum) {
        .array => |arr| arr.node.allocator,
        .chunked => |chunks| chunks.node.allocator,
        .scalar => |s| if (s.payload) |payload| payload.node.allocator else null,
        .record_batch => |b| b.node.allocator,
        .table => |t| t.node.allocator,
    };
}

pub fn inferDatumsAllocator(datums: []const Datum) std.mem.Allocator {
    for (datums) |datum| {
        if (inferDatumAllocator(datum)) |allocator| return allocator;
    }
    return std.heap.page_allocator;
}

/// A record batch datum owning `batch` (deinitialized on error).
pub fn ownBatch(allocator: std.mem.Allocator, batch: core.RecordBatch) KernelError!Datum {
    var b = batch;
    const shared = core.SharedRecordBatch.create(allocator, b) catch |err| {
        b.deinit();
        return err;
    };
    return .{ .record_batch = shared };
}

/// A table datum owning `table` (deinitialized on error).
pub fn ownTable(allocator: std.mem.Allocator, table: core.Table) KernelError!Datum {
    var t = table;
    const shared = core.SharedTable.create(allocator, t) catch |err| {
        t.deinit();
        return err;
    };
    return .{ .table = shared };
}

pub fn mapTabularError(err: anyerror) KernelError {
    return switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        else => error.InvalidInput,
    };
}
