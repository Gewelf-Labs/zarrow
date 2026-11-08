const std = @import("std");
const datatype = @import("../datatype.zig");

pub const DataType = datatype.DataType;
pub const IntType = datatype.IntType;

pub const BufferRegion = struct {
    offset: usize,
    length: usize,

    pub fn bytes(self: BufferRegion, body: []const u8) []const u8 {
        return body[self.offset .. self.offset + self.length];
    }
};

pub const TensorDim = struct {
    size: i64,
    name: ?[]const u8,
};

pub const TensorMetadata = struct {
    value_type: DataType,
    shape: []const TensorDim,
    strides: ?[]const i64,
    data: BufferRegion,
};

/// Byte width of a tensor value type: the fixed-width numeric types Arrow C++ supports for
/// tensors (`is_tensor_supported`: integers, half_float, float, double); null for any other.
pub fn tensorByteWidth(dt: DataType) ?usize {
    return switch (dt) {
        .int8, .uint8 => 1,
        .int16, .uint16, .half_float => 2,
        .int32, .uint32, .float => 4,
        .int64, .uint64, .double => 8,
        else => null,
    };
}

pub const TensorError = error{InvalidTensor};

/// Arrow C++'s `CheckTensorValidity` / `CheckTensorStridesValidity` against a body of
/// `body_len` bytes: a supported value type, sizes >= 0, dimension names absent or one per
/// dimension, strides one per dimension and >= 0 (row-major when absent), the data region inside
/// the body and the farthest element (sum of (size - 1) * stride, plus one value) inside the
/// region, every product and sum checked for overflow. A tensor with a zero-size dimension holds
/// no element and needs no data.
pub fn validateTensor(meta: TensorMetadata, body_len: usize) TensorError!void {
    const width = tensorByteWidth(meta.value_type) orelse return error.InvalidTensor;
    const end = std.math.add(usize, meta.data.offset, meta.data.length) catch return error.InvalidTensor;
    if (end > body_len) return error.InvalidTensor;
    var empty = false;
    for (meta.shape) |dim| {
        if (dim.size < 0) return error.InvalidTensor;
        if (dim.size == 0) empty = true;
    }
    if (meta.strides) |strides| if (strides.len != meta.shape.len) return error.InvalidTensor;
    if (empty) return;
    // The largest byte offset of an element (row-major strides when none are given).
    var largest: u64 = 0;
    if (meta.strides) |strides| {
        for (meta.shape, strides) |dim, stride| {
            if (stride < 0) return error.InvalidTensor;
            const step = std.math.mul(u64, @intCast(dim.size - 1), @intCast(stride)) catch return error.InvalidTensor;
            largest = std.math.add(u64, largest, step) catch return error.InvalidTensor;
        }
    } else {
        var count: u64 = 1;
        for (meta.shape) |dim| count = std.math.mul(u64, count, @intCast(dim.size)) catch return error.InvalidTensor;
        largest = std.math.mul(u64, count - 1, width) catch return error.InvalidTensor;
    }
    const need = std.math.add(u64, largest, width) catch return error.InvalidTensor;
    if (need > meta.data.length) return error.InvalidTensor;
}

pub const SparseMatrixAxis = enum {
    row,
    column,
};

pub const SparseTensorIndexMetadata = union(enum) {
    coo: struct {
        indices_type: IntType,
        indices_strides: ?[]const i64,
        indices: BufferRegion,
        is_canonical: bool,
    },
    csx: struct {
        compressed_axis: SparseMatrixAxis,
        indptr_type: IntType,
        indptr: BufferRegion,
        indices_type: IntType,
        indices: BufferRegion,
    },
    csf: struct {
        indptr_type: IntType,
        indptr_buffers: []const BufferRegion,
        indices_type: IntType,
        indices_buffers: []const BufferRegion,
        axis_order: []const i32,
    },
};

pub const SparseTensorMetadata = struct {
    value_type: DataType,
    shape: []const TensorDim,
    non_zero_length: usize,
    sparse_index: SparseTensorIndexMetadata,
    data: BufferRegion,
};

pub const TensorLikeMetadata = union(enum) {
    tensor: TensorMetadata,
    sparse_tensor: SparseTensorMetadata,
};
