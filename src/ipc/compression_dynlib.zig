const std = @import("std");

pub const ZstdSymbols = struct {
    compress_bound: *const fn (usize) callconv(.c) usize,
    compress: *const fn (?*anyopaque, usize, ?*const anyopaque, usize, c_int) callconv(.c) usize,
    decompress: *const fn (?*anyopaque, usize, ?*const anyopaque, usize) callconv(.c) usize,
    is_error: *const fn (usize) callconv(.c) c_uint,
    // Reusable contexts: the one-shot calls above allocate and free a context on every call.
    create_cctx: *const fn () callconv(.c) ?*anyopaque,
    free_cctx: *const fn (?*anyopaque) callconv(.c) usize,
    compress_cctx: *const fn (?*anyopaque, ?*anyopaque, usize, ?*const anyopaque, usize, c_int) callconv(.c) usize,
    create_dctx: *const fn () callconv(.c) ?*anyopaque,
    free_dctx: *const fn (?*anyopaque) callconv(.c) usize,
    decompress_dctx: *const fn (?*anyopaque, ?*anyopaque, usize, ?*const anyopaque, usize) callconv(.c) usize,
};

pub const Lz4Symbols = struct {
    create_decompression_context: *const fn (*?*anyopaque, c_uint) callconv(.c) usize,
    free_decompression_context: *const fn (?*anyopaque) callconv(.c) usize,
    reset_decompression_context: *const fn (?*anyopaque) callconv(.c) void,
    decompress: *const fn (?*anyopaque, ?*anyopaque, *usize, ?*const anyopaque, *usize, ?*const anyopaque) callconv(.c) usize,
    is_error: *const fn (usize) callconv(.c) c_uint,
    compress_frame_bound: *const fn (usize, ?*const anyopaque) callconv(.c) usize,
    compress_frame: *const fn (?*anyopaque, usize, ?*const anyopaque, usize, ?*const anyopaque) callconv(.c) usize,
};

extern fn ZSTD_compressBound(src_size: usize) callconv(.c) usize;
extern fn ZSTD_compress(dst: ?*anyopaque, dst_capacity: usize, src: ?*const anyopaque, src_size: usize, compression_level: c_int) callconv(.c) usize;
extern fn ZSTD_decompress(dst: ?*anyopaque, dst_capacity: usize, src: ?*const anyopaque, compressed_size: usize) callconv(.c) usize;
extern fn ZSTD_isError(code: usize) callconv(.c) c_uint;
extern fn ZSTD_createCCtx() callconv(.c) ?*anyopaque;
extern fn ZSTD_freeCCtx(cctx: ?*anyopaque) callconv(.c) usize;
extern fn ZSTD_compressCCtx(cctx: ?*anyopaque, dst: ?*anyopaque, dst_capacity: usize, src: ?*const anyopaque, src_size: usize, compression_level: c_int) callconv(.c) usize;
extern fn ZSTD_createDCtx() callconv(.c) ?*anyopaque;
extern fn ZSTD_freeDCtx(dctx: ?*anyopaque) callconv(.c) usize;
extern fn ZSTD_decompressDCtx(dctx: ?*anyopaque, dst: ?*anyopaque, dst_capacity: usize, src: ?*const anyopaque, src_size: usize) callconv(.c) usize;

extern fn LZ4F_createDecompressionContext(ctx_ptr: *?*anyopaque, version: c_uint) callconv(.c) usize;
extern fn LZ4F_freeDecompressionContext(ctx: ?*anyopaque) callconv(.c) usize;
extern fn LZ4F_resetDecompressionContext(ctx: ?*anyopaque) callconv(.c) void;
extern fn LZ4F_decompress(
    ctx: ?*anyopaque,
    dst: ?*anyopaque,
    dst_size_ptr: *usize,
    src: ?*const anyopaque,
    src_size_ptr: *usize,
    options_ptr: ?*const anyopaque,
) callconv(.c) usize;
extern fn LZ4F_isError(code: usize) callconv(.c) c_uint;
extern fn LZ4F_compressFrameBound(src_size: usize, prefs_ptr: ?*const anyopaque) callconv(.c) usize;
extern fn LZ4F_compressFrame(
    dst: ?*anyopaque,
    dst_capacity: usize,
    src: ?*const anyopaque,
    src_size: usize,
    prefs_ptr: ?*const anyopaque,
) callconv(.c) usize;

var linked_zstd_symbols: ZstdSymbols = .{
    .compress_bound = ZSTD_compressBound,
    .compress = ZSTD_compress,
    .decompress = ZSTD_decompress,
    .is_error = ZSTD_isError,
    .create_cctx = ZSTD_createCCtx,
    .free_cctx = ZSTD_freeCCtx,
    .compress_cctx = ZSTD_compressCCtx,
    .create_dctx = ZSTD_createDCtx,
    .free_dctx = ZSTD_freeDCtx,
    .decompress_dctx = ZSTD_decompressDCtx,
};

var linked_lz4_symbols: Lz4Symbols = .{
    .create_decompression_context = LZ4F_createDecompressionContext,
    .free_decompression_context = LZ4F_freeDecompressionContext,
    .reset_decompression_context = LZ4F_resetDecompressionContext,
    .decompress = LZ4F_decompress,
    .is_error = LZ4F_isError,
    .compress_frame_bound = LZ4F_compressFrameBound,
    .compress_frame = LZ4F_compressFrame,
};

pub fn loadZstdSymbols() !*const ZstdSymbols {
    return &linked_zstd_symbols;
}

pub fn loadLz4Symbols() !*const Lz4Symbols {
    return &linked_lz4_symbols;
}

/// One spare context of each kind, shared by every writer and reader: streams are often short
/// (one writer or reader per message), and a fresh context per stream costs its allocation plus
/// first-touch page faults of its workspace. A stream takes the spare (or creates a context) on
/// first use and puts it back on deinit; a second spare is freed.
/// ponytail: one slot per kind; a small pool if many short streams compress concurrently.
var spare_zstd_cctx: std.atomic.Value(?*anyopaque) = .init(null);
var spare_zstd_dctx: std.atomic.Value(?*anyopaque) = .init(null);
var spare_lz4_dctx: std.atomic.Value(?*anyopaque) = .init(null);

pub fn takeZstdCCtx() ?*anyopaque {
    return spare_zstd_cctx.swap(null, .acquire) orelse linked_zstd_symbols.create_cctx();
}

pub fn putZstdCCtx(cctx: *anyopaque) void {
    if (spare_zstd_cctx.cmpxchgStrong(null, cctx, .release, .monotonic) != null) _ = linked_zstd_symbols.free_cctx(cctx);
}

pub fn takeZstdDCtx() ?*anyopaque {
    return spare_zstd_dctx.swap(null, .acquire) orelse linked_zstd_symbols.create_dctx();
}

pub fn putZstdDCtx(dctx: *anyopaque) void {
    if (spare_zstd_dctx.cmpxchgStrong(null, dctx, .release, .monotonic) != null) _ = linked_zstd_symbols.free_dctx(dctx);
}

/// An LZ4 frame decompression context; callers reset it before each frame.
pub fn takeLz4DCtx() ?*anyopaque {
    if (spare_lz4_dctx.swap(null, .acquire)) |dctx| return dctx;
    var created: ?*anyopaque = null;
    const rc = linked_lz4_symbols.create_decompression_context(&created, 100); // LZ4F_VERSION
    if (linked_lz4_symbols.is_error(rc) != 0) return null;
    return created;
}

pub fn putLz4DCtx(dctx: *anyopaque) void {
    if (spare_lz4_dctx.cmpxchgStrong(null, dctx, .release, .monotonic) != null) _ = linked_lz4_symbols.free_decompression_context(dctx);
}

fn repeatStr(comptime s: []const u8, comptime n: usize) *const [s.len * n]u8 {
    comptime {
        @setEvalBranchQuota(n * (s.len + 16));
        var buf: [s.len * n]u8 = undefined;
        for (0..n) |i| @memcpy(buf[i * s.len ..][0..s.len], s);
        const final = buf;
        return &final;
    }
}
test "zstd c api round-trip via compression symbol table" {
    const allocator = std.testing.allocator;
    const syms = try loadZstdSymbols();
    const input = comptime repeatStr("zstd-c-api-roundtrip-0123456789abcdef\n", 256);

    const bound = syms.*.compress_bound(input.len);
    try std.testing.expect(bound > 0);
    try std.testing.expectEqual(@as(c_uint, 0), syms.*.is_error(bound));

    const compressed = try allocator.alloc(u8, bound);
    defer allocator.free(compressed);
    const written = syms.*.compress(
        @ptrCast(compressed.ptr),
        compressed.len,
        @ptrCast(input.ptr),
        input.len,
        3,
    );
    try std.testing.expectEqual(@as(c_uint, 0), syms.*.is_error(written));
    try std.testing.expect(written > 0 and written <= compressed.len);

    const decoded = try allocator.alloc(u8, input.len);
    defer allocator.free(decoded);
    const decoded_len = syms.*.decompress(
        @ptrCast(decoded.ptr),
        decoded.len,
        @ptrCast(compressed.ptr),
        written,
    );
    try std.testing.expectEqual(@as(c_uint, 0), syms.*.is_error(decoded_len));
    try std.testing.expectEqual(input.len, decoded_len);
    try std.testing.expectEqualSlices(u8, input, decoded[0..decoded_len]);
}

test "lz4 frame c api round-trip via compression symbol table" {
    const allocator = std.testing.allocator;
    const syms = try loadLz4Symbols();
    const input = comptime repeatStr("lz4-frame-c-api-roundtrip-0123456789abcdef\n", 256);

    const bound = syms.*.compress_frame_bound(input.len, null);
    try std.testing.expect(bound > 0);
    try std.testing.expectEqual(@as(c_uint, 0), syms.*.is_error(bound));

    const compressed = try allocator.alloc(u8, bound);
    defer allocator.free(compressed);
    const written = syms.*.compress_frame(
        @ptrCast(compressed.ptr),
        compressed.len,
        @ptrCast(input.ptr),
        input.len,
        null,
    );
    try std.testing.expectEqual(@as(c_uint, 0), syms.*.is_error(written));
    try std.testing.expect(written > 0 and written <= compressed.len);

    var dctx: ?*anyopaque = null;
    const create_rc = syms.*.create_decompression_context(&dctx, 100); // LZ4F_VERSION
    try std.testing.expectEqual(@as(c_uint, 0), syms.*.is_error(create_rc));
    try std.testing.expect(dctx != null);
    defer _ = syms.*.free_decompression_context(dctx);

    const decoded = try allocator.alloc(u8, input.len);
    defer allocator.free(decoded);

    var src_pos: usize = 0;
    var dst_pos: usize = 0;
    while (true) {
        var src_size = written - src_pos;
        var dst_size = decoded.len - dst_pos;
        const rc = syms.*.decompress(
            dctx,
            if (dst_size == 0) null else @ptrCast(decoded.ptr + dst_pos),
            &dst_size,
            if (src_size == 0) null else @ptrCast(compressed.ptr + src_pos),
            &src_size,
            null,
        );
        try std.testing.expectEqual(@as(c_uint, 0), syms.*.is_error(rc));

        src_pos += src_size;
        dst_pos += dst_size;

        if (rc == 0) break;
        try std.testing.expect(!(src_size == 0 and dst_size == 0));
        try std.testing.expect(src_pos <= written);
        try std.testing.expect(dst_pos <= decoded.len);
    }

    try std.testing.expectEqual(written, src_pos);
    try std.testing.expectEqual(input.len, dst_pos);
    try std.testing.expectEqualSlices(u8, input, decoded[0..dst_pos]);
}
