//! String kernels (Arrow C++ "String transforms", "String trimming", "String slicing",
//! "String joining", "String predicates", "Containment tests") on string, large_string and
//! string_view; the output keeps the input layout (string in, string out; lengths are int32 for
//! string / string_view and int64 for large_string). Null handling is intersection, except
//! binary_join_element_wise, which follows `JoinOptions`.
//!
//! Unicode (tables in `unicode_tables.zig`, Unicode 17.0.0, the version of pyarrow 26's utf8proc):
//! - utf8_upper / utf8_lower / utf8_swapcase / utf8_capitalize / utf8_title map every code point
//!   through its simple (one to one) case mapping, like Arrow's utf8proc kernels (U+00DF uppercases
//!   to U+1E9E there and here); special casing (ß to SS, final sigma, ligatures) is not applied,
//!   as in Arrow. A row may change its byte length (at most 3/2 of it).
//! - utf8_is_* use the general category with Arrow's definitions: alpha L*, alnum L* N*, decimal
//!   Nd, digit Nd No, numeric Nd Nl No, printable anything but Cc Cf Cs Co Cn Zs Zl Zp (U+0020
//!   is printable), lower / upper / title over cased code points (Lu Ll Lt or a case mapping);
//!   utf8_is_space, utf8_*trim_whitespace and utf8_split_whitespace use Unicode White_Space plus
//!   U+001C..U+001F (the set Arrow gets from utf8proc).
//! - The ascii_* functions look at bytes: a byte >= 0x80 is no letter, digit, space or printable.
//! - match_substring / starts_with / ends_with take a literal pattern; `ignore_case` matches
//!   through Unicode simple case folding (the regex engine, as Arrow uses RE2).
//! The utf8_* kernels that decode code points reject a non-null row that is not valid UTF-8 with
//! `InvalidInput`, so they never emit invalid UTF-8; utf8_length counts the bytes that do not
//! continue a sequence, like Arrow, without validating.
const std = @import("std");
const bitmap = @import("../bitmap.zig");
const core = @import("core.zig");
const datatype = @import("../datatype.zig");
const dense = @import("dense.zig");
const unicode = @import("unicode_tables.zig");
const string_regex = @import("string_regex.zig");

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
pub const Transform = enum {
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
    ascii_swapcase,
    utf8_swapcase,
    ascii_capitalize,
    utf8_capitalize,
    ascii_title,
    utf8_title,
    ascii_reverse,
    utf8_trim,
    utf8_ltrim,
    utf8_rtrim,
};

/// String to bool functions.
const Predicate = enum {
    starts_with,
    ends_with,
    match_substring,
    string_is_ascii,
    ascii_is_alnum,
    ascii_is_alpha,
    ascii_is_decimal,
    ascii_is_lower,
    ascii_is_printable,
    ascii_is_space,
    ascii_is_title,
    ascii_is_upper,
    utf8_is_alnum,
    utf8_is_alpha,
    utf8_is_decimal,
    utf8_is_digit,
    utf8_is_lower,
    utf8_is_numeric,
    utf8_is_printable,
    utf8_is_space,
    utf8_is_title,
    utf8_is_upper,
};

/// String to length functions.
const Length = enum { utf8_length, binary_length };

/// Register every string function for string, large_string and string_view (binary_length also
/// for binary, large_binary and binary_view).
pub fn register(registry: *FunctionRegistry) KernelError!void {
    @setEvalBranchQuota(10_000);
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
    // The containment tests also take the binary layouts (Arrow; Latin-1 with `ignore_case`).
    inline for (binary_tags) |tag| {
        inline for (.{ Predicate.starts_with, Predicate.ends_with, Predicate.match_substring }) |p| {
            try registry.registerScalarKernel(@tagName(p), unaryKernel(tag, predicateOptions(p), dense.resultTypeOf(bool_type), Predicates(p).exec));
        }
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
        .utf8_trim, .utf8_ltrim, .utf8_rtrim => &struct {
            /// TrimOptions whose characters are UTF-8 (each code point is one character).
            fn check(options: Options) bool {
                return options == .trim and std.unicode.utf8ValidateSlice(options.trim.characters);
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
pub fn validUtf8(s: []const u8) bool {
    return isAscii(s) or std.unicode.utf8ValidateSlice(s);
}

/// Return whether every byte of `s` is ASCII (16 bytes at a time).
pub fn isAscii(s: []const u8) bool {
    const L = 16;
    var acc: @Vector(L, u8) = @splat(0);
    var i: usize = 0;
    while (i + L <= s.len) : (i += L) acc |= s[i..][0..L].*;
    var tail: u8 = 0;
    for (s[i..]) |c| tail |= c;
    return (@reduce(.Or, acc) | tail) < 0x80;
}

/// Byte length of the code point starting with `first` (valid UTF-8 only).
pub inline fn cpLen(first: u8) usize {
    return if (first < 0x80) 1 else if (first < 0xE0) 2 else if (first < 0xF0) 3 else 4;
}

/// Unicode White_Space plus U+001C..U+001F: the code points Arrow's utf8 kernels treat as space
/// (checked against pyarrow 26).
pub fn isUnicodeSpace(cp: u21) bool {
    return switch (cp) {
        0x09...0x0D, 0x1C...0x20, 0x85, 0xA0, 0x1680, 0x2000...0x200A, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000 => true,
        else => false,
    };
}

/// Code point starting at `s[i]` (valid UTF-8 only).
pub fn decodeAt(s: []const u8, i: usize) u21 {
    return std.unicode.utf8Decode(s[i..][0..cpLen(s[i])]) catch unreachable;
}

/// Start of the code point ending at `s[end - 1]` (valid UTF-8 only).
fn prevStart(s: []const u8, end: usize) usize {
    var i = end - 1;
    while (s[i] & 0xC0 == 0x80) i -= 1;
    return i;
}

/// A decoded code point and its byte length.
pub const Decoded = struct { cp: u21, len: u3 };

/// Decode the code point starting at `s[i]`, or return null when `s[i..]` does not start with a
/// well-formed UTF-8 sequence (overlong forms, surrogates and values past U+10FFFF are not).
pub inline fn decodeStrict(s: []const u8, i: usize) ?Decoded {
    const c0 = s[i];
    if (c0 < 0x80) return .{ .cp = c0, .len = 1 };
    if (c0 < 0xC2 or c0 > 0xF4) return null;
    const n: u3 = if (c0 < 0xE0) 2 else if (c0 < 0xF0) 3 else 4;
    if (s.len - i < n) return null;
    const c1 = s[i + 1];
    const lo: u8 = switch (c0) {
        0xE0 => 0xA0,
        0xF0 => 0x90,
        else => 0x80,
    };
    const hi: u8 = switch (c0) {
        0xED => 0x9F,
        0xF4 => 0x8F,
        else => 0xBF,
    };
    if (c1 < lo or c1 > hi) return null;
    var cp: u21 = @as(u21, c0 & (@as(u8, 0x7F) >> n)) << 6 | (c1 & 0x3F);
    for (s[i + 2 .. i + n]) |c| {
        if (c & 0xC0 != 0x80) return null;
        cp = cp << 6 | (c & 0x3F);
    }
    return .{ .cp = cp, .len = n };
}

/// Write the UTF-8 encoding of the scalar value `cp` at the start of `dst`; return its length.
pub inline fn encodeCp(cp: u21, dst: []u8) usize {
    if (cp < 0x80) {
        dst[0] = @intCast(cp);
        return 1;
    }
    if (cp < 0x800) {
        dst[0..2].* = .{ @intCast(0xC0 | (cp >> 6)), @intCast(0x80 | (cp & 0x3F)) };
        return 2;
    }
    if (cp < 0x10000) {
        dst[0..3].* = .{ @intCast(0xE0 | (cp >> 12)), @intCast(0x80 | ((cp >> 6) & 0x3F)), @intCast(0x80 | (cp & 0x3F)) };
        return 3;
    }
    dst[0..4].* = .{ @intCast(0xF0 | (cp >> 18)), @intCast(0x80 | ((cp >> 12) & 0x3F)), @intCast(0x80 | ((cp >> 6) & 0x3F)), @intCast(0x80 | (cp & 0x3F)) };
    return 4;
}

// ------------------------------------------------------------------------------ Unicode tables

pub const Category = unicode.Category;

/// Index of the entry of a bucketed table holding `cp`: the last one whose start is at or
/// before `cp` (0 when none), searched between the bucket's entry and the next bucket's.
inline fn findEntry(entries: anytype, buckets: []const u16, comptime start: anytype, cp: u21) usize {
    var lo: usize = undefined;
    var hi: usize = undefined;
    if (cp < unicode.bucket_limit) {
        const b = cp >> unicode.bucket_shift;
        lo = buckets[b];
        hi = buckets[b + 1];
    } else {
        lo = buckets[buckets.len - 1];
        hi = entries.len - 1;
    }
    while (lo < hi) {
        const mid = (lo + hi + 1) / 2;
        if (start(entries[mid]) <= cp) lo = mid else hi = mid - 1;
    }
    return lo;
}

fn runStart(run: u32) u21 {
    return @intCast(run >> 5);
}

fn rangeFirst(r: unicode.CaseRange) u21 {
    return r.first;
}

/// General category of `cp` (`Cn` when unassigned).
pub fn category(cp: u21) Category {
    const run = unicode.category_runs[findEntry(&unicode.category_runs, &unicode.category_buckets, runStart, cp)];
    return @enumFromInt(@as(u5, @truncate(run)));
}

fn mapCase(comptime ranges: []const unicode.CaseRange, comptime buckets: []const u16, cp: u21) u21 {
    if (cp >= unicode.bucket_limit) return cp; // the generator checks that no mapping starts there
    const r = ranges[findEntry(ranges, buckets, rangeFirst, cp)];
    if (cp < r.first or cp > r.last or (r.stride2 == 1 and (cp - r.first) & 1 == 1)) return cp;
    return @intCast(@as(i32, cp) + r.delta);
}

/// Simple uppercase of `cp` (U+00DF gives U+1E9E, as in Arrow C++).
pub inline fn toUpper(cp: u21) u21 {
    if (cp < 0x80) return upper(@intCast(cp));
    return mapCase(&unicode.upper_ranges, &unicode.upper_buckets, cp);
}

/// Simple lowercase of `cp`.
pub inline fn toLower(cp: u21) u21 {
    if (cp < 0x80) return lower(@intCast(cp));
    return mapCase(&unicode.lower_ranges, &unicode.lower_buckets, cp);
}

/// Cased code point (Arrow: category Lu, Ll or Lt, or a case mapping to another code point).
pub fn isCased(cp: u21) bool {
    if (cp < 0x80) return std.ascii.isAlphabetic(@intCast(cp));
    return switch (category(cp)) {
        .Lu, .Ll, .Lt => true,
        else => toUpper(cp) != cp or toLower(cp) != cp,
    };
}

/// Lowercase code point (Arrow: Ll, or an uppercase mapping and no lowercase one; never Lt).
pub fn isLowerCp(cp: u21) bool {
    if (cp < 0x80) return cp -% 'a' < 26;
    return switch (category(cp)) {
        .Ll => true,
        .Lt => false,
        else => toUpper(cp) != cp and toLower(cp) == cp,
    };
}

/// Uppercase code point (Arrow: Lu, or a lowercase mapping and no uppercase one; never Lt).
pub fn isUpperCp(cp: u21) bool {
    if (cp < 0x80) return cp -% 'A' < 26;
    return switch (category(cp)) {
        .Lu => true,
        .Lt => false,
        else => toUpper(cp) == cp and toLower(cp) != cp,
    };
}

/// Per code point classes of the utf8_is_* predicates (Arrow C++ definitions, checked against
/// pyarrow 26 over every code point): alpha L*, alnum L* N*, decimal Nd, digit Nd No (utf8proc
/// has no Numeric_Type), numeric Nd Nl No, printable not Cc Cf Cs Co Cn Zs Zl Zp (U+0020 is).
pub const CpClass = enum { alpha, alnum, decimal, digit, numeric, printable };

pub fn hasClass(comptime class: CpClass, cp: u21) bool {
    if (cp < 0x80) {
        const c: u8 = @intCast(cp);
        return switch (class) {
            .alpha => std.ascii.isAlphabetic(c),
            .alnum => std.ascii.isAlphanumeric(c),
            .decimal, .digit, .numeric => std.ascii.isDigit(c),
            .printable => c >= 0x20 and c < 0x7F,
        };
    }
    const c = category(cp);
    return switch (class) {
        .alpha => switch (c) {
            .Lu, .Ll, .Lt, .Lm, .Lo => true,
            else => false,
        },
        .alnum => switch (c) {
            .Lu, .Ll, .Lt, .Lm, .Lo, .Nd, .Nl, .No => true,
            else => false,
        },
        .decimal => c == .Nd,
        .digit => c == .Nd or c == .No,
        .numeric => c == .Nd or c == .Nl or c == .No,
        .printable => switch (c) {
            .Cc, .Cf, .Cs, .Co, .Cn, .Zs, .Zl, .Zp => false,
            else => true,
        },
    };
}

// ------------------------------------------------------------------------------ transforms

pub fn upper(c: u8) u8 {
    return if (c -% 'a' < 26) c - 32 else c;
}

pub fn lower(c: u8) u8 {
    return if (c -% 'A' < 26) c + 32 else c;
}

fn swap(c: u8) u8 {
    return if ((c | 32) -% 'a' < 26) c ^ 32 else c;
}

/// `dst[i] = f(src[i])` for an ASCII case map `f` (upper, lower or swap), 32 bytes per step.
fn mapBytes(comptime f: fn (u8) u8, dst: []u8, src: []const u8) void {
    const L = 32;
    const V = @Vector(L, u8);
    const first: V = @splat(if (f == upper) 'a' else 'A');
    const flip: V = @splat(32);
    const n26: V = @splat(26);
    var i: usize = 0;
    while (i + L <= src.len) : (i += L) {
        const v: V = src[i..][0..L].*;
        const hit = if (f == swap) ((v | flip) -% @as(V, @splat('a'))) < n26 else (v -% first) < n26;
        dst[i..][0..L].* = @select(u8, hit, v ^ flip, v);
    }
    for (dst[i..src.len], src[i..]) |*d, c| d.* = f(c);
}

fn Transformer(comptime t: Transform) type {
    const case_map: ?fn (u8) u8 = switch (t) {
        .ascii_upper, .utf8_upper => upper,
        .ascii_lower, .utf8_lower => lower,
        .ascii_swapcase, .utf8_swapcase => swap,
        else => null,
    };
    const validates = std.mem.startsWith(u8, @tagName(t), "utf8_");
    // Unicode case mapping decodes, validates and maps in one pass; a row may change length.
    const unicode_case = switch (t) {
        .utf8_upper, .utf8_lower, .utf8_swapcase, .utf8_capitalize, .utf8_title => true,
        else => false,
    };
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
            // Unicode case mapping may grow a row: start with its worst case, so no reserve reallocates.
            const total = dense.totalBytes(rows, len);
            var out = try dense.VarBinaryOutput.init(ctx.allocator, data.data_type, len, try dense.combinedValidity(ctx.allocator, len, &.{data}), if (unicode_case) caseMapCapacity(total) else total);
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

            const bits = dense.validityBits(data);
            // Unicode case mapping of an offsets layout: every row into one slot of the window's
            // worst-case size, offsets written directly (no per-row reserve / commit).
            if (unicode_case) switch (rows) {
                inline .offsets32, .offsets64 => |o| {
                    const O = @TypeOf(o.offsets[0]);
                    const base: usize = @intCast(o.offsets[0]);
                    const window = o.data[base..@intCast(o.offsets[len])];
                    const dst = try out.reserve(caseMapCapacity(window.len));
                    const offs = out.offsetSlots(O);
                    // upper / lower of a chunk without nulls: the whole window as one stream
                    // when every row starts on a code point (each row must be UTF-8 on its own;
                    // otherwise, or on bad UTF-8, the per-row loop below reports it). Row
                    // offsets move by the length changes recorded on the way.
                    if ((t == .utf8_upper or t == .utf8_lower) and bits == null) stream: {
                        var cut = false;
                        for (o.offsets[0..len]) |off| {
                            const at = @as(usize, @intCast(off)) - base;
                            cut = cut or (at < window.len and window[at] & 0xC0 == 0x80);
                        }
                        if (cut) break :stream;
                        var growth: Growths = .{ .allocator = ctx.allocator };
                        defer growth.list.deinit(ctx.allocator);
                        const n = caseMapImpl(t, dst, window, &growth) catch |err| switch (err) {
                            error.InvalidInput => break :stream,
                            error.OutOfMemory => return error.OutOfMemory,
                        };
                        var k: usize = 0;
                        var delta: isize = 0;
                        for (offs[1..], o.offsets[1 .. len + 1]) |*d, off| {
                            const b = @as(usize, @intCast(off)) - base;
                            while (k < growth.list.items.len and growth.list.items[k].pos < b) : (k += 1) delta += growth.list.items[k].delta;
                            d.* = std.math.cast(O, @as(isize, @intCast(b)) + delta) orelse return error.Overflow;
                        }
                        out.closeRows(n);
                        return out.finish();
                    }
                    var w: usize = 0;
                    for (0..len) |i| {
                        const valid = if (bits) |b| bitmap.bitIsSet(b, data.offset + i) else true;
                        if (valid) {
                            const src = window[@as(usize, @intCast(o.offsets[i])) - base .. @as(usize, @intCast(o.offsets[i + 1])) - base];
                            w += caseMapUtf8(t, dst[w..], src) orelse return error.InvalidInput;
                        }
                        offs[i + 1] = std.math.cast(O, w) orelse return error.Overflow;
                    }
                    out.closeRows(w);
                    return out.finish();
                },
                else => {},
            };

            var scratch: std.ArrayList(usize) = .empty;
            defer scratch.deinit(ctx.allocator);
            switch (rows) {
                inline else => |r| for (0..len) |i| {
                    if (bits) |b| if (!bitmap.bitIsSet(b, data.offset + i)) {
                        try out.commit(0);
                        continue;
                    };
                    const src = r.get(i);
                    if (unicode_case) {
                        try out.commit(caseMapUtf8(t, try out.reserve(caseMapCapacity(src.len)), src) orelse return error.InvalidInput);
                        continue;
                    }
                    if (validates and !validUtf8(src)) return error.InvalidInput;
                    // Arrow: ascii_reverse of a non-ASCII row is an error.
                    if (t == .ascii_reverse and !isAscii(src)) return error.InvalidInput;
                    try out.commit(try apply(ctx.allocator, try out.reserve(src.len), src, options, &scratch));
                },
            }
            return out.finish();
        }

        /// Write the transform of `src` into `dst` (never longer than `src`); return its length.
        fn apply(allocator: std.mem.Allocator, dst: []u8, src: []const u8, options: Options, scratch: *std.ArrayList(usize)) KernelError!usize {
            if (case_map) |f| if (!unicode_case) {
                mapBytes(f, dst, src);
                return src.len;
            };
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
                .utf8_trim, .utf8_ltrim, .utf8_rtrim => {
                    const set = options.trim.characters;
                    if (t != .utf8_rtrim) while (lo < hi and inSet(set, src[lo..][0..cpLen(src[lo])])) {
                        lo += cpLen(src[lo]);
                    };
                    if (t != .utf8_ltrim) while (hi > lo) {
                        const s = prevStart(src, hi);
                        if (!inSet(set, src[s..hi])) break;
                        hi = s;
                    };
                },
                .ascii_reverse => {
                    for (src, 0..) |c, k| dst[src.len - 1 - k] = c;
                    return src.len;
                },
                .ascii_title => {
                    var next_upper = true;
                    for (src, dst[0..src.len]) |c, *d| {
                        const letter = std.ascii.isAlphabetic(c);
                        d.* = if (!letter) c else if (next_upper) upper(c) else lower(c);
                        next_upper = !letter;
                    }
                    return src.len;
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
                .utf8_slice_codeunits => return slice(false, allocator, dst, src, options.slice, scratch),
                .ascii_capitalize, .utf8_capitalize => {
                    if (src.len == 0) return 0;
                    dst[0] = upper(src[0]);
                    mapBytes(lower, dst[1..src.len], src[1..]);
                    return src.len;
                },
                else => unreachable,
            }
            @memcpy(dst[0 .. hi - lo], src[lo..hi]);
            return hi - lo;
        }
    };
}

/// Whether the code point `cp` (its UTF-8 bytes) is one of the code points of the UTF-8 `set`.
fn inSet(set: []const u8, cp: []const u8) bool {
    var i: usize = 0;
    while (i < set.len) : (i += cpLen(set[i])) {
        if (std.mem.eql(u8, set[i..][0..cpLen(set[i])], cp)) return true;
    }
    return false;
}

/// Bytes `caseMapUtf8` may write for `n` input bytes: a mapping grows a code point from 2 to at
/// most 3 bytes (the generator checks it), plus 3 bytes of slack for its 4-byte stores.
pub fn caseMapCapacity(n: usize) usize {
    return n + n / 2 + 3;
}

/// UTF-8 encodings of the simple upper / lower case of U+0000..U+07FF (the 1- and 2-byte code
/// points: ASCII, Latin, Greek, Cyrillic, Hebrew, Arabic, ...), built at compile time from the
/// range tables: the bytes in the low 24 bits (in order), the length in the high 8 (8 KiB each).
fn twoByteMap(comptime f: anytype) [0x800]u32 {
    @setEvalBranchQuota(1_000_000);
    var table: [0x800]u32 = undefined;
    for (&table, 0..) |*e, cp| {
        var buf: [4]u8 = @splat(0);
        const n = encodeCp(f(@intCast(cp)), &buf);
        e.* = std.mem.readInt(u32, &buf, .little) | @as(u32, @intCast(n)) << 24;
    }
    return table;
}

const upper2 = twoByteMap(toUpper);
const lower2 = twoByteMap(toLower);

/// A code point whose case mapping changes the UTF-8 length: its input byte offset and the
/// output bytes minus the input bytes (-2 .. +1).
pub const Growth = struct { pos: usize, delta: i8 };

/// Positions where a case mapping changed the length (rare: U+00DF, U+0131, U+212A, ...).
pub const Growths = struct {
    list: std.ArrayList(Growth) = .empty,
    allocator: std.mem.Allocator,

    fn note(self: *Growths, pos: usize, delta: isize) error{OutOfMemory}!void {
        try self.list.append(self.allocator, .{ .pos = pos, .delta = @intCast(delta) });
    }
};

/// Case-map `src` code point by code point into `dst` (at least `caseMapCapacity(src.len)`
/// bytes) for one of the utf8 case functions; return the bytes written, or null when `src` is
/// not UTF-8. capitalize: the first code point to upper, the rest to lower; title: a cased code
/// point to upper after an uncased one (or at the start), to lower after a cased one; swapcase:
/// upper to lower and lower to upper (Arrow C++ rules).
pub fn caseMapUtf8(comptime t: Transform, dst: []u8, src: []const u8) ?usize {
    return caseMapImpl(t, dst, src, {}) catch null;
}

/// `caseMapUtf8`; with a `*Growths` as `growth`, also every code point whose mapping changes
/// the length (to rebuild row offsets after mapping many rows as one stream).
fn caseMapImpl(comptime t: Transform, dst: []u8, src: []const u8, growth: anytype) error{ InvalidInput, OutOfMemory }!usize {
    const track = @TypeOf(growth) != void;
    var r: usize = 0;
    var w: usize = 0;
    var next_upper = true;
    while (r < src.len) {
        const c = src[r];
        if (t == .utf8_upper or t == .utf8_lower) {
            const table = if (t == .utf8_upper) &upper2 else &lower2;
            // Four bytes at a time when they are four ASCII bytes or two 2-byte sequences: the
            // loop is bound by the latency of finding the next code point, so fewer steps win.
            if (src.len - r >= 4) {
                const word = std.mem.readInt(u32, src[r..][0..4], .little);
                if (word & 0x8080_8080 == 0) {
                    const v: @Vector(4, u8) = src[r..][0..4].*;
                    const first: @Vector(4, u8) = @splat(if (t == .utf8_upper) 'a' else 'A');
                    const hit = (v -% first) < @as(@Vector(4, u8), @splat(26));
                    dst[w..][0..4].* = @select(u8, hit, v ^ @as(@Vector(4, u8), @splat(32)), v);
                    r += 4;
                    w += 4;
                    continue;
                }
                // Lead bytes C2..DF at 0 and 2, continuation bytes at 1 and 3.
                if (word & 0xC0E0_C0E0 == 0x80C0_80C0 and c >= 0xC2 and src[r + 2] >= 0xC2) {
                    const e0 = table[(word & 0x1F) << 6 | (word >> 8 & 0x3F)];
                    const e1 = table[(word >> 16 & 0x1F) << 6 | (word >> 24 & 0x3F)];
                    std.mem.writeInt(u32, dst[w..][0..4], e0, .little);
                    w += e0 >> 24;
                    std.mem.writeInt(u32, dst[w..][0..4], e1, .little);
                    w += e1 >> 24;
                    if (track and (e0 >> 24 != 2 or e1 >> 24 != 2)) {
                        if (e0 >> 24 != 2) try growth.note(r, @as(isize, e0 >> 24) - 2);
                        if (e1 >> 24 != 2) try growth.note(r + 2, @as(isize, e1 >> 24) - 2);
                    }
                    r += 4;
                    continue;
                }
            }
            // An ASCII byte or a well-formed 2-byte sequence, without a branch between the two
            // (text mixes them): one table load, one 4-byte store (`dst` has slack).
            const c1: u8 = if (r + 1 < src.len) src[r + 1] else 0;
            const ascii = c < 0x80;
            if (ascii or (c >= 0xC2 and c < 0xE0 and c1 & 0xC0 == 0x80)) {
                const cp: u32 = if (ascii) c else @as(u32, c & 0x1F) << 6 | (c1 & 0x3F);
                const e = table[cp];
                std.mem.writeInt(u32, dst[w..][0..4], e, .little);
                w += e >> 24;
                const n = 2 - @as(usize, @intFromBool(ascii));
                if (track and e >> 24 != n) try growth.note(r, @as(isize, e >> 24) - @as(isize, @intCast(n)));
                r += n;
                continue;
            }
        }
        const d = decodeStrict(src, r) orelse return error.InvalidInput;
        const at = r;
        r += d.len;
        const cp = d.cp;
        const m: u21 = switch (t) {
            .utf8_upper => toUpper(cp),
            .utf8_lower => toLower(cp),
            .utf8_swapcase => if (isUpperCp(cp)) toLower(cp) else if (isLowerCp(cp)) toUpper(cp) else cp,
            .utf8_capitalize => if (r == d.len) toUpper(cp) else toLower(cp),
            .utf8_title => blk: {
                if (!isCased(cp)) {
                    next_upper = true;
                    break :blk cp;
                }
                defer next_upper = false;
                break :blk if (next_upper) toUpper(cp) else toLower(cp);
            },
            else => unreachable,
        };
        const n = encodeCp(m, dst[w..]);
        w += n;
        if (track and n != d.len) try growth.note(at, @as(isize, @intCast(n)) - d.len);
    }
    return w;
}

/// Python `s[start:stop:step]` over the code points of valid UTF-8 `src` (Arrow C++ semantics),
/// or over its bytes when `bytes` (binary_slice; `starts` is then unused).
pub fn slice(comptime bytes: bool, allocator: std.mem.Allocator, dst: []u8, src: []const u8, o: core.SliceOptions, starts: *std.ArrayList(usize)) KernelError!usize {
    var n: i64 = @intCast(src.len);
    if (!bytes) {
        starts.clearRetainingCapacity();
        var i: usize = 0;
        while (i < src.len) : (i += cpLen(src[i])) starts.append(allocator, i) catch return error.OutOfMemory;
        starts.append(allocator, src.len) catch return error.OutOfMemory;
        n = @intCast(starts.items.len - 1);
    }
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
        const cp = if (bytes) src[k..][0..1] else src[starts.items[k]..starts.items[k + 1]];
        @memcpy(dst[w..][0..cp.len], cp);
        w += cp.len;
        cur = std.math.add(i64, cur, o.step) catch break;
    }
    return w;
}

// ------------------------------------------------------------------------------ predicates

fn nextAsciiLetter(s: []const u8, i: *usize) ?u21 {
    if (i.* >= s.len) return null;
    defer i.* += 1;
    return if (std.ascii.isAlphabetic(s[i.*])) s[i.*] else 0;
}

fn nextCodePoint(s: []const u8, i: *usize) ?u21 {
    if (i.* >= s.len) return null;
    defer i.* += cpLen(s[i.*]);
    return decodeAt(s, i.*);
}

/// is_lower / is_upper / is_title over the code points `next` yields (0: an uncased byte of the
/// ascii_* functions). lower: some cased code point and every cased one lowercase; upper: the
/// same with uppercase; title: some cased code point, every lowercase one right after a cased
/// one and every other cased one right after an uncased one (Arrow C++ rules).
fn casePredicate(comptime p: Predicate, s: []const u8, comptime next: fn ([]const u8, *usize) ?u21) bool {
    var i: usize = 0;
    var any_cased = false;
    var prev_cased = false;
    while (next(s, &i)) |cp| {
        if (cp == 0 or !isCased(cp)) {
            prev_cased = false;
            continue;
        }
        any_cased = true;
        const ok = switch (p) {
            .ascii_is_lower, .utf8_is_lower => isLowerCp(cp),
            .ascii_is_upper, .utf8_is_upper => isUpperCp(cp),
            else => isLowerCp(cp) == prev_cased,
        };
        if (!ok) return false;
        prev_cased = true;
    }
    return any_cased;
}

fn Predicates(comptime p: Predicate) type {
    return struct {
        /// Test each row (array, chunked or scalar).
        fn exec(ctx: *ExecContext, args: []const Datum, options: Options) KernelError!Datum {
            return dense.execElementwiseWith(ctx, args, options, bool_type, true, chunk);
        }

        fn chunk(ctx: *ExecContext, values: []const ExecChunkValue, len: usize, options: Options) KernelError!ArrayRef {
            if (comptime (p == .starts_with or p == .ends_with or p == .match_substring)) {
                // Case-insensitive: Unicode simple case folding through the regex engine (RE2
                // in Arrow).
                const o = options.match_substring;
                if (o.ignore_case) {
                    const which: @TypeOf(.contains) = switch (p) {
                        .starts_with => .starts,
                        .ends_with => .ends,
                        else => .contains,
                    };
                    var m = try string_regex.literalMatcher(ctx.allocator, which, o.pattern, values[0].dataType());
                    defer m.deinit();
                    return string_regex.boolRows(ctx, values[0], len, &m);
                }
            }
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
                        .starts_with => std.mem.startsWith(u8, s, o.pattern),
                        .ends_with => std.mem.endsWith(u8, s, o.pattern),
                        else => std.mem.indexOf(u8, s, o.pattern) != null,
                    };
                },
                .string_is_ascii => return isAscii(s),
                // A byte that is no ASCII letter is uncased (stands as 0).
                .ascii_is_lower, .ascii_is_upper, .ascii_is_title => return casePredicate(p, s, nextAsciiLetter),
                .utf8_is_lower, .utf8_is_upper, .utf8_is_title => {
                    if (!validUtf8(s)) return error.InvalidInput;
                    return casePredicate(p, s, nextCodePoint);
                },
                .ascii_is_alnum, .ascii_is_alpha, .ascii_is_decimal, .ascii_is_printable, .ascii_is_space => {
                    // Arrow: an empty row is printable, nothing else.
                    if (s.len == 0) return p == .ascii_is_printable;
                    for (s) |c| if (!switch (p) {
                        .ascii_is_alnum => std.ascii.isAlphanumeric(c),
                        .ascii_is_alpha => std.ascii.isAlphabetic(c),
                        .ascii_is_decimal => std.ascii.isDigit(c),
                        .ascii_is_printable => c >= 0x20 and c < 0x7F,
                        else => std.ascii.isWhitespace(c),
                    }) return false;
                    return true;
                },
                else => {
                    if (!validUtf8(s)) return error.InvalidInput;
                    if (s.len == 0) return p == .utf8_is_printable;
                    var i: usize = 0;
                    while (i < s.len) : (i += cpLen(s[i])) {
                        const cp = decodeAt(s, i);
                        const ok = switch (p) {
                            .utf8_is_alnum => hasClass(.alnum, cp),
                            .utf8_is_alpha => hasClass(.alpha, cp),
                            .utf8_is_decimal => hasClass(.decimal, cp),
                            .utf8_is_digit => hasClass(.digit, cp),
                            .utf8_is_numeric => hasClass(.numeric, cp),
                            .utf8_is_printable => hasClass(.printable, cp),
                            .utf8_is_space => isUnicodeSpace(cp),
                            else => unreachable,
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
