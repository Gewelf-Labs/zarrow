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

/// Metadata length (null at end of stream) and the padding after the metadata that 8-aligns
/// prefix + metadata. Pre-0.15 streams have a 4-byte prefix (no continuation marker).
pub fn readMessageLength(reader: *std.Io.Reader) std.Io.Reader.Error!?struct { len: u32, pad: usize } {
    const first = try readInt(reader, u32);
    if (first == 0) return null;
    if (first == ContinuationMarker) {
        const len = try readInt(reader, u32);
        if (len == 0) return null;
        return .{ .len = len, .pad = padLen(len) };
    }
    return .{ .len = first, .pad = padLen(@as(usize, first) + 4) };
}

pub fn writePadding(writer: *std.Io.Writer, pad_len: usize) std.Io.Writer.Error!void {
    try writer.splatByteAll(0, pad_len);
}

pub fn skipPadding(reader: *std.Io.Reader, pad_len: usize) std.Io.Reader.Error!void {
    try reader.discardAll(pad_len);
}

/// Skip up to `pad_len` zero bytes after a message body whose length is not a multiple of 8.
/// pyarrow's `write_tensor` writes such a body without padding, so the next message (or the end
/// of the stream) may start right after it; a writer that pads writes zeros.
pub fn skipBodyPadding(reader: *std.Io.Reader, pad_len: usize) void {
    for (0..pad_len) |_| {
        const b = reader.peekByte() catch return;
        if (b != 0) return;
        reader.toss(1);
    }
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
