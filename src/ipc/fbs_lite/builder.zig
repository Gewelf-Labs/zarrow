const std = @import("std");
const fb = @import("fbs_runtime");
const types = @import("types.zig");
const verify = @import("verify.zig");

/// Unpack errors; `InvalidMetadata` when the bytes fail `verify` (untrusted input never reaches
/// the unchecked generated readers).
pub const PackError = types.PackError || verify.Error;

pub fn unpackMessage(allocator: std.mem.Allocator, metadata: []const u8) PackError!types.MessageT {
    try verify.verifyMessage(metadata);
    const root = types.Message.GetRootAs(@constCast(metadata), 0);
    const opts: types.PackOptions = .{ .allocator = allocator };
    return try types.MessageT.Unpack(root, opts);
}

pub fn unpackFooter(allocator: std.mem.Allocator, footer_bytes: []const u8) PackError!types.FooterT {
    try verify.verifyFooter(footer_bytes);
    const root = types.Footer.GetRootAs(@constCast(footer_bytes), 0);
    const opts: types.PackOptions = .{ .allocator = allocator };
    return try types.FooterT.Unpack(root, opts);
}

pub fn packMessageBytes(allocator: std.mem.Allocator, msg: types.MessageT) (PackError || error{OutOfMemory})![]u8 {
    var builder = fb.Builder.init(allocator);
    defer builder.deinitAll();

    const opts: types.PackOptions = .{ .allocator = allocator };
    const msg_off = try types.MessageT.Pack(msg, &builder, opts);
    try types.Message.FinishBuffer(&builder, msg_off);
    const bytes = try builder.finishedBytes();
    return try allocator.dupe(u8, bytes);
}

pub fn packFooterBytes(allocator: std.mem.Allocator, footer: types.FooterT) (PackError || error{OutOfMemory})![]u8 {
    var builder = fb.Builder.init(allocator);
    defer builder.deinitAll();

    const opts: types.PackOptions = .{ .allocator = allocator };
    const footer_off = try types.FooterT.Pack(footer, &builder, opts);
    try types.Footer.FinishBuffer(&builder, footer_off);
    const bytes = try builder.finishedBytes();
    return try allocator.dupe(u8, bytes);
}
