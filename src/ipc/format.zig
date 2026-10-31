const std = @import("std");

pub const StreamError = error{
    InvalidMessage,
    UnsupportedType,
    SchemaNotRead,
    InvalidMetadata,
    InvalidBody,
};

pub const Alignment: usize = 8;
pub const ContinuationMarker: u32 = 0xFFFF_FFFF;

pub fn paddedLen(len: usize) usize {
    return len + padLen(len);
}

pub fn padLen(len: usize) usize {
    const rem = len % Alignment;
    return if (rem == 0) 0 else Alignment - rem;
}

pub fn writeMessageLength(writer: *std.Io.Writer, len: u32) std.Io.Writer.Error!void {
    try writeInt(writer, u32, ContinuationMarker);
    try writeInt(writer, u32, len);
}

pub fn readMessageLength(reader: *std.Io.Reader) std.Io.Reader.Error!?u32 {
    const first = try readInt(reader, u32);
    if (first == 0) return null;
    if (first == ContinuationMarker) {
        const len = try readInt(reader, u32);
        if (len == 0) return null;
        return len;
    }
    return first;
}

pub fn writePadding(writer: *std.Io.Writer, pad_len: usize) std.Io.Writer.Error!void {
    try writer.splatByteAll(0, pad_len);
}

pub fn skipPadding(reader: *std.Io.Reader, pad_len: usize) std.Io.Reader.Error!void {
    try reader.discardAll(pad_len);
}

pub fn writeInt(writer: *std.Io.Writer, comptime T: type, value: T) std.Io.Writer.Error!void {
    try writer.writeInt(T, value, .little);
}

pub fn readInt(reader: *std.Io.Reader, comptime T: type) std.Io.Reader.Error!T {
    // takeInt asserts the reader buffer can hold T; readSliceAll works with unbuffered readers too.
    var b: [@sizeOf(T)]u8 = undefined;
    try reader.readSliceAll(&b);
    return std.mem.readInt(T, &b, .little);
}
