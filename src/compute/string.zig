//! String kernels (Arrow C++ "String transforms", "String trimming", "String slicing",
//! "String joining", "String predicates", "Containment tests") on string, large_string and
//! string_view; the output keeps the input layout (string in, string out; lengths are int32 for
//! string / string_view and int64 for large_string). Null handling is intersection, except
//! binary_join_element_wise, which follows `JoinOptions`.
//!
//! Scope against Arrow C++ (no Unicode tables in the Zig standard library):
//! - utf8_upper / utf8_lower map ASCII letters only and leave other code points unchanged
//!   (Arrow maps every cased code point through utf8proc).
//! - utf8_is_alpha / utf8_is_digit treat non-ASCII code points as neither letters nor digits;
//!   utf8_is_space and utf8_*trim_whitespace use the Unicode White_Space property.
//! - match_substring / starts_with / ends_with take a literal pattern; `ignore_case` folds ASCII
//!   letters only.
//! The utf8_* kernels that decode code points (upper / lower, trim_whitespace, slice_codeunits,
//! reverse, is_alpha / is_digit / is_space) reject a non-null row that is not valid UTF-8 with
//! `InvalidInput`, so they never emit invalid UTF-8; utf8_length counts the bytes that do not
//! continue a sequence, like Arrow, without validating.
const std = @import("std");
const bitmap = @import("../bitmap.zig");
const core = @import("core.zig");
const datatype = @import("../datatype.zig");
const dense = @import("dense.zig");

const ArrayData = @import("../array/array_data.zig").ArrayData;
const ArrayRef = core.ArrayRef;
const DataType = core.DataType;
const Datum = core.Datum;
const ExecChunkValue = core.ExecChunkValue;
const ExecContext = core.ExecContext;
const FunctionRegistry = core.FunctionRegistry;
const KernelError = core.KernelError;
const Options = core.Options;
const TypeTag = std.meta.Tag(DataType);

const string_tags = [_]TypeTag{ .string, .large_string, .string_view };
const binary_tags = [_]TypeTag{ .binary, .large_binary, .binary_view };
const bool_type = DataType{ .bool = {} };

/// String to string functions.
const Transform = enum {
    ascii_upper,
    ascii_lower,
    utf8_upper,
    utf8_lower,
    ascii_trim,
    ascii_ltrim,
    ascii_rtrim,
    ascii_trim_whitespace,
    ascii_ltrim_whitespace,
    ascii_rtrim_whitespace,
    utf8_trim_whitespace,
    utf8_ltrim_whitespace,
    utf8_rtrim_whitespace,
    utf8_slice_codeunits,
    utf8_reverse,
};

/// String to bool functions.
const Predicate = enum {
    starts_with,
    ends_with,
    match_substring,
    string_is_ascii,
    ascii_is_alpha,
    ascii_is_decimal,
    ascii_is_space,
    utf8_is_alpha,
    utf8_is_digit,
    utf8_is_space,
};

/// String to length functions.
const Length = enum { utf8_length, binary_length };

/// Register every string function for string, large_string and string_view (binary_length also
/// for binary, large_binary and binary_view).
pub fn register(registry: *FunctionRegistry) KernelError!void {
    inline for (string_tags) |tag| {
        inline for (comptime std.meta.tags(Transform)) |t| {
            try registry.registerScalarKernel(@tagName(t), unaryKernel(tag, transformOptions(t), dense.resultTypeOf(@unionInit(DataType, @tagName(tag), {})), Transformer(t).exec));
        }
        inline for (comptime std.meta.tags(Predicate)) |p| {
            try registry.registerScalarKernel(@tagName(p), unaryKernel(tag, predicateOptions(p), dense.resultTypeOf(bool_type), Predicates(p).exec));
        }
        try registry.registerScalarKernel("utf8_length", unaryKernel(tag, dense.noOptions, dense.resultTypeOf(lengthType(tag)), Lengths(.utf8_length).exec));
        try registry.registerScalarKernel("binary_join_element_wise", .{
            .signature = .{
                .arity = 1,
                .variadic = true,
                .type_check = allOfType(tag),
                .options_check = joinOptions,
                .result_type_fn = dense.resultTypeOf(@unionInit(DataType, @tagName(tag), {})),
            },
            .exec = join,
        });
    }
    inline for (string_tags ++ binary_tags) |tag| {
        try registry.registerScalarKernel("binary_length", unaryKernel(tag, dense.noOptions, dense.resultTypeOf(lengthType(tag)), Lengths(.binary_length).exec));
    }
}

fn unaryKernel(comptime tag: TypeTag, comptime options_check: core.OptionsCheckFn, comptime result: core.ResultTypeFn, comptime exec: core.KernelExecFn) core.Kernel {
    return .{
        .signature = .{ .arity = 1, .type_check = dense.argsOfType(tag, 1), .options_check = options_check, .result_type_fn = result },
        .exec = exec,
    };
}

/// int64 lengths for the large layouts, int32 otherwise (Arrow C++).
fn lengthType(comptime tag: TypeTag) DataType {
    return if (tag == .large_string or tag == .large_binary) .{ .int64 = {} } else .{ .int32 = {} };
}

fn allOfType(comptime tag: TypeTag) *const fn ([]const Datum) bool {
    return &struct {
        fn check(args: []const Datum) bool {
            if (args.len == 0) return false;
            for (args) |arg| if (std.meta.activeTag(arg.dataType()) != tag) return false;
            return true;
        }
    }.check;
}

fn transformOptions(comptime t: Transform) core.OptionsCheckFn {
    return switch (t) {
        .ascii_trim, .ascii_ltrim, .ascii_rtrim => &struct {
            /// TrimOptions whose characters are ASCII (trimming other bytes could split a code point).
            fn check(options: Options) bool {
                if (options != .trim) return false;
                for (options.trim.characters) |c| if (c >= 0x80) return false;
                return true;
            }
        }.check,
        .utf8_slice_codeunits => &struct {
            fn check(options: Options) bool {
                return options == .slice and options.slice.step != 0;
            }
        }.check,
        else => dense.noOptions,
    };
}

fn predicateOptions(comptime p: Predicate) core.OptionsCheckFn {
    return switch (p) {
        .starts_with, .ends_with, .match_substring => &struct {
            fn check(options: Options) bool {
                return options == .match_substring;
            }
        }.check,
        else => dense.noOptions,
    };
}

/// Accept `none` (emit_null) or `JoinOptions` with a UTF-8 `null_replacement`.
fn joinOptions(options: Options) bool {
    return switch (options) {
        .none => true,
        .join => |o| std.unicode.utf8ValidateSlice(o.null_replacement),
        else => false,
    };
}

// ------------------------------------------------------------------------------ code points

/// Return whether `s` is valid UTF-8 (all-ASCII strings take a vector fast path).
fn validUtf8(s: []const u8) bool {
    return isAscii(s) or std.unicode.utf8ValidateSlice(s);
}

fn isAscii(s: []const u8) bool {
    const L = 16;
    var acc: @Vector(L, u8) = @splat(0);
    var i: usize = 0;
    while (i + L <= s.len) : (i += L) acc |= s[i..][0..L].*;
    var tail: u8 = 0;
    for (s[i..]) |c| tail |= c;
    return (@reduce(.Or, acc) | tail) < 0x80;
}

/// Byte length of the code point starting with `first` (valid UTF-8 only).
inline fn cpLen(first: u8) usize {
    return if (first < 0x80) 1 else if (first < 0xE0) 2 else if (first < 0xF0) 3 else 4;
}

/// Unicode White_Space (the property utf8proc reports as space for Arrow's utf8 kernels).
fn isUnicodeSpace(cp: u21) bool {
    return switch (cp) {
        0x09...0x0D, 0x20, 0x85, 0xA0, 0x1680, 0x2000...0x200A, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000 => true,
        else => false,
    };
}

/// Code point starting at `s[i]` (valid UTF-8 only).
fn decodeAt(s: []const u8, i: usize) u21 {
    return std.unicode.utf8Decode(s[i..][0..cpLen(s[i])]) catch unreachable;
}

/// Start of the code point ending at `s[end - 1]` (valid UTF-8 only).
fn prevStart(s: []const u8, end: usize) usize {
    var i = end - 1;
    while (s[i] & 0xC0 == 0x80) i -= 1;
    return i;
}

// ------------------------------------------------------------------------------ transforms

fn upper(c: u8) u8 {
    return if (c -% 'a' < 26) c - 32 else c;
}

fn lower(c: u8) u8 {
    return if (c -% 'A' < 26) c + 32 else c;
}

/// `dst[i] = f(src[i])` for an ASCII case map `f` (upper or lower), 32 bytes per step.
fn mapBytes(comptime f: fn (u8) u8, dst: []u8, src: []const u8) void {
    const L = 32;
    const V = @Vector(L, u8);
    const first: V = @splat(if (f == upper) 'a' else 'A');
    const flip: V = @splat(32);
    const n26: V = @splat(26);
    var i: usize = 0;
    while (i + L <= src.len) : (i += L) {
        const v: V = src[i..][0..L].*;
        dst[i..][0..L].* = @select(u8, (v -% first) < n26, v ^ flip, v);
    }
    for (dst[i..src.len], src[i..]) |*d, c| d.* = f(c);
}

fn Transformer(comptime t: Transform) type {
    const case_map: ?fn (u8) u8 = switch (t) {
        .ascii_upper, .utf8_upper => upper,
        .ascii_lower, .utf8_lower => lower,
        else => null,
    };
    const validates = std.mem.startsWith(u8, @tagName(t), "utf8_");
    return struct {
        /// Transform each row (array, chunked or scalar).
        fn exec(ctx: *ExecContext, args: []const Datum, options: Options) KernelError!Datum {
            return dense.execElementwiseWith(ctx, args, options, args[0].dataType(), true, chunk);
        }

        fn chunk(ctx: *ExecContext, values: []const ExecChunkValue, len: usize, options: Options) KernelError!ArrayRef {
            var arr = try dense.chunkArray(ctx.allocator, values[0]);
            defer arr.release();
            const data = arr.data();
            const rows = try dense.StringOperand.init(.{ .array = arr });
            var out = try dense.VarBinaryOutput.init(ctx.allocator, data.data_type, len, try dense.combinedValidity(ctx.allocator, len, &.{data}), dense.totalBytes(rows, len));
            errdefer out.deinit();

            // Case mapping keeps every length: map the whole value window of an offsets layout at
            // once (ascii_*: always; utf8_*: when the window is ASCII, hence valid UTF-8).
            if (case_map) |f| switch (rows) {
                inline .offsets32, .offsets64 => |o| {
                    const window = o.data[@intCast(o.offsets[0])..@intCast(o.offsets[len])];
                    if (!validates or isAscii(window)) {
                        mapBytes(f, try out.reserve(window.len), window);
                        try out.bulkOffsets(o.offsets);
                        return out.finish();
                    }
                },
                else => {},
            };

            var scratch: std.ArrayList(usize) = .empty;
            defer scratch.deinit(ctx.allocator);
            const bits = dense.validityBits(data);
            switch (rows) {
                inline else => |r| for (0..len) |i| {
                    if (bits) |b| if (!bitmap.bitIsSet(b, data.offset + i)) {
                        try out.commit(0);
                        continue;
                    };
                    const src = r.get(i);
                    if (validates and !validUtf8(src)) return error.InvalidInput;
                    try out.commit(try apply(ctx.allocator, try out.reserve(src.len), src, options, &scratch));
                },
            }
            return out.finish();
        }

        /// Write the transform of `src` into `dst` (never longer than `src`); return its length.
        fn apply(allocator: std.mem.Allocator, dst: []u8, src: []const u8, options: Options, scratch: *std.ArrayList(usize)) KernelError!usize {
            if (case_map) |f| {
                mapBytes(f, dst, src);
                return src.len;
            }
            var lo: usize = 0;
            var hi: usize = src.len;
            switch (t) {
                .ascii_trim, .ascii_ltrim, .ascii_rtrim => {
                    var set: [256]bool = @splat(false);
                    for (options.trim.characters) |c| set[c] = true;
                    if (t != .ascii_rtrim) while (lo < hi and set[src[lo]]) {
                        lo += 1;
                    };
                    if (t != .ascii_ltrim) while (hi > lo and set[src[hi - 1]]) {
                        hi -= 1;
                    };
                },
                .ascii_trim_whitespace, .ascii_ltrim_whitespace, .ascii_rtrim_whitespace => {
                    if (t != .ascii_rtrim_whitespace) while (lo < hi and std.ascii.isWhitespace(src[lo])) {
                        lo += 1;
                    };
                    if (t != .ascii_ltrim_whitespace) while (hi > lo and std.ascii.isWhitespace(src[hi - 1])) {
                        hi -= 1;
                    };
                },
                .utf8_trim_whitespace, .utf8_ltrim_whitespace, .utf8_rtrim_whitespace => {
                    if (t != .utf8_rtrim_whitespace) while (lo < hi and isUnicodeSpace(decodeAt(src, lo))) {
                        lo += cpLen(src[lo]);
                    };
                    if (t != .utf8_ltrim_whitespace) while (hi > lo) {
                        const s = prevStart(src, hi);
                        if (!isUnicodeSpace(decodeAt(src, s))) break;
                        hi = s;
                    };
                },
                .utf8_reverse => {
                    var i: usize = 0;
                    while (i < src.len) {
                        const n = cpLen(src[i]);
                        @memcpy(dst[src.len - i - n ..][0..n], src[i..][0..n]);
                        i += n;
                    }
                    return src.len;
                },
                .utf8_slice_codeunits => return slice(allocator, dst, src, options.slice, scratch),
                else => unreachable,
            }
            @memcpy(dst[0 .. hi - lo], src[lo..hi]);
            return hi - lo;
        }
    };
}

/// Python `s[start:stop:step]` over the code points of valid UTF-8 `src` (Arrow C++ semantics).
fn slice(allocator: std.mem.Allocator, dst: []u8, src: []const u8, o: core.SliceOptions, starts: *std.ArrayList(usize)) KernelError!usize {
    starts.clearRetainingCapacity();
    var i: usize = 0;
    while (i < src.len) : (i += cpLen(src[i])) starts.append(allocator, i) catch return error.OutOfMemory;
    starts.append(allocator, src.len) catch return error.OutOfMemory;
    const n: i64 = @intCast(starts.items.len - 1);
    const lower_bound: i64 = if (o.step > 0) 0 else -1;
    const upper_bound: i64 = if (o.step > 0) n else n - 1;
    const clamp = struct {
        fn f(x: i64, len: i64, lo: i64, hi: i64) i64 {
            const y = if (x < 0) x + len else x;
            return @max(lo, @min(hi, y));
        }
    }.f;
    var cur = clamp(o.start, n, lower_bound, upper_bound);
    const stop = clamp(o.stop, n, lower_bound, upper_bound);
    var w: usize = 0;
    while (if (o.step > 0) cur < stop else cur > stop) {
        const k: usize = @intCast(cur);
        const cp = src[starts.items[k]..starts.items[k + 1]];
        @memcpy(dst[w..][0..cp.len], cp);
        w += cp.len;
        cur = std.math.add(i64, cur, o.step) catch break;
    }
    return w;
}

// ------------------------------------------------------------------------------ predicates

fn indexOfIgnoreCase(haystack: []const u8, needle: []const u8) bool {
    if (needle.len > haystack.len) return false;
    for (0..haystack.len - needle.len + 1) |i| {
        if (std.ascii.eqlIgnoreCase(haystack[i..][0..needle.len], needle)) return true;
    }
    return false;
}

fn Predicates(comptime p: Predicate) type {
    return struct {
        /// Test each row (array, chunked or scalar).
        fn exec(ctx: *ExecContext, args: []const Datum, options: Options) KernelError!Datum {
            return dense.execElementwiseWith(ctx, args, options, bool_type, true, chunk);
        }

        fn chunk(ctx: *ExecContext, values: []const ExecChunkValue, len: usize, options: Options) KernelError!ArrayRef {
            var arr = try dense.chunkArray(ctx.allocator, values[0]);
            defer arr.release();
            const data = arr.data();
            var out = try dense.DenseBinaryOutput(bool).initValidity(ctx.allocator, len, data, null, bool_type);
            errdefer out.deinit();
            const out_bits = out.values();
            @memset(out_bits, 0);
            const bits = dense.validityBits(data);
            switch (try dense.StringOperand.init(.{ .array = arr })) {
                inline else => |r| for (0..len) |i| {
                    if (bits) |b| if (!bitmap.bitIsSet(b, data.offset + i)) continue;
                    if (try test1(r.get(i), options)) out_bits[i / 8] |= @as(u8, 1) << @intCast(i % 8);
                },
            }
            return out.finish() catch error.OutOfMemory;
        }

        fn test1(s: []const u8, options: Options) KernelError!bool {
            switch (p) {
                .starts_with, .ends_with, .match_substring => {
                    const o = options.match_substring;
                    return switch (p) {
                        .starts_with => if (o.ignore_case) std.ascii.startsWithIgnoreCase(s, o.pattern) else std.mem.startsWith(u8, s, o.pattern),
                        .ends_with => if (o.ignore_case) std.ascii.endsWithIgnoreCase(s, o.pattern) else std.mem.endsWith(u8, s, o.pattern),
                        else => if (o.ignore_case) indexOfIgnoreCase(s, o.pattern) else std.mem.indexOf(u8, s, o.pattern) != null,
                    };
                },
                .string_is_ascii => return isAscii(s),
                .ascii_is_alpha, .ascii_is_decimal, .ascii_is_space => {
                    if (s.len == 0) return false;
                    for (s) |c| if (!switch (p) {
                        .ascii_is_alpha => std.ascii.isAlphabetic(c),
                        .ascii_is_decimal => std.ascii.isDigit(c),
                        else => std.ascii.isWhitespace(c),
                    }) return false;
                    return true;
                },
                .utf8_is_alpha, .utf8_is_digit, .utf8_is_space => {
                    if (!validUtf8(s)) return error.InvalidInput;
                    if (s.len == 0) return false;
                    var i: usize = 0;
                    while (i < s.len) : (i += cpLen(s[i])) {
                        const cp = decodeAt(s, i);
                        const ok = switch (p) {
                            .utf8_is_alpha => cp < 0x80 and std.ascii.isAlphabetic(@intCast(cp)),
                            .utf8_is_digit => cp < 0x80 and std.ascii.isDigit(@intCast(cp)),
                            else => isUnicodeSpace(cp),
                        };
                        if (!ok) return false;
                    }
                    return true;
                },
            }
        }
    };
}

// ------------------------------------------------------------------------------ lengths

fn Lengths(comptime l: Length) type {
    return struct {
        /// Return the length of each row in code points (utf8_length) or bytes (binary_length).
        fn exec(ctx: *ExecContext, args: []const Datum, options: Options) KernelError!Datum {
            const tag = std.meta.activeTag(args[0].dataType());
            const out_type: DataType = if (tag == .large_string or tag == .large_binary) .{ .int64 = {} } else .{ .int32 = {} };
            return dense.execElementwiseWith(ctx, args, options, out_type, true, chunk);
        }

        fn chunk(ctx: *ExecContext, values: []const ExecChunkValue, len: usize, _: Options) KernelError!ArrayRef {
            var arr = try dense.chunkArray(ctx.allocator, values[0]);
            defer arr.release();
            const data = arr.data();
            return switch (datatype.storageDataType(data.data_type)) {
                .large_string, .large_binary => run(i64, ctx, arr, len),
                else => run(i32, ctx, arr, len),
            };
        }

        fn run(comptime O: type, ctx: *ExecContext, arr: ArrayRef, len: usize) KernelError!ArrayRef {
            var out = try dense.DenseBinaryOutput(O).initValidity(ctx.allocator, len, arr.data(), null, dense.dataTypeOf(O));
            errdefer out.deinit();
            switch (try dense.StringOperand.init(.{ .array = arr })) {
                inline else => |r| for (out.values(), 0..) |*o, i| {
                    const s = r.get(i);
                    o.* = @intCast(if (l == .binary_length) s.len else s.len - countContinuation(s));
                },
            }
            return out.finish() catch error.OutOfMemory;
        }

        fn countContinuation(s: []const u8) usize {
            var n: usize = 0;
            for (s) |c| n += @intFromBool(c & 0xC0 == 0x80);
            return n;
        }
    };
}

// ------------------------------------------------------------------------------ join

/// Join the string arguments of each row with the last argument as the separator.
fn join(ctx: *ExecContext, args: []const Datum, options: Options) KernelError!Datum {
    return dense.execElementwiseWith(ctx, args, options, args[0].dataType(), false, joinChunk);
}

fn joinChunk(ctx: *ExecContext, values: []const ExecChunkValue, len: usize, options: Options) KernelError!ArrayRef {
    const o: core.JoinOptions = if (options == .join) options.join else .{};
    const k = values.len - 1;
    const ops = ctx.allocator.alloc(dense.StringOperand, values.len) catch return error.OutOfMemory;
    defer ctx.allocator.free(ops);
    for (values, ops) |v, *op| op.* = if (v == .scalar and v.scalar.isNull()) .{ .scalar = .{ .bytes = "" } } else try dense.StringOperand.init(v);

    // Pass 1: the validity and the byte total of every row.
    var validity = dense.bitsBuffer(ctx.allocator, len) catch return error.OutOfMemory;
    var validity_owned = true;
    defer if (validity_owned) validity.deinit();
    @memset(validity.data[0..bitmap.byteLength(len)], 0);
    var total: usize = 0;
    for (0..len) |i| {
        if (values[k].isNullAt(i)) continue;
        var n: usize = 0;
        var parts: usize = 0;
        var valid = true;
        for (values[0..k], ops[0..k]) |v, op| {
            if (v.isNullAt(i)) switch (o.null_handling) {
                .emit_null => valid = false,
                .skip => continue,
                .replace => {
                    n += o.null_replacement.len;
                    parts += 1;
                    continue;
                },
            };
            n += rowBytes(op, i).len;
            parts += 1;
        }
        if (!valid) continue;
        bitmap.setBit(validity.data, i);
        total += n + rowBytes(ops[k], i).len * (parts -| 1);
    }

    // Pass 2: fill.
    validity_owned = false;
    var out = try dense.VarBinaryOutput.init(ctx.allocator, values[0].dataType(), len, validity, total);
    errdefer out.deinit();
    for (0..len) |i| {
        if (!bitmap.bitIsSet(out.validity.?.data, i)) {
            try out.commit(0);
            continue;
        }
        const sep = rowBytes(ops[k], i);
        var w: usize = 0;
        var first = true;
        for (values[0..k], ops[0..k]) |v, op| {
            const part = if (!v.isNullAt(i)) rowBytes(op, i) else if (o.null_handling == .replace) o.null_replacement else continue;
            if (!first) {
                @memcpy((try out.reserve(w + sep.len))[w..], sep);
                w += sep.len;
            }
            first = false;
            @memcpy((try out.reserve(w + part.len))[w..], part);
            w += part.len;
        }
        try out.commit(w);
    }
    return out.finish();
}

fn rowBytes(op: dense.StringOperand, i: usize) []const u8 {
    return switch (op) {
        inline else => |r| r.get(i),
    };
}
