//! String transforms that change lengths or split rows (Arrow C++ "String transforms", "String
//! padding", "String splitting", "String slicing", "Containment tests"): replace_substring,
//! split_pattern, ascii_split_whitespace, utf8_split_whitespace, ascii_lpad / rpad / center,
//! utf8_lpad / rpad / center, utf8_zero_fill, count_substring, find_substring, binary_slice,
//! binary_repeat, binary_reverse, utf8_replace_slice, binary_replace_slice and binary_join. A
//! string output keeps the input layout; a split gives `list<T>` of the input layout `T`; counts
//! and positions are int32 (int64 for the large layouts). Null handling is intersection
//! (binary_join: a null list, a null separator or a null list item gives a null row, as in Arrow).
//!
//! utf8_replace_slice / binary_replace_slice (`ReplaceSliceOptions`) replace the code points /
//! bytes `[start, stop)`: a negative position counts from the end, both are clamped to the row and
//! a stop before the start inserts at the start (Arrow C++). utf8_zero_fill (`ZeroFillOptions`)
//! left-pads to `width` code points with one padding code point, after a leading `+` or `-`.
//! binary_join joins the items of a `list<T>` / `large_list<T>` row (T string, large_string,
//! binary or large_binary) with a separator of type T (array or scalar).
//!
//! Scope against Arrow C++: patterns are literal (no regex); `ignore_case` folds ASCII letters
//! only (Arrow folds through RE2). For a string layout, the pattern, replacement and padding of an
//! option must be valid UTF-8 (ASCII for the ascii_* pad functions), so a valid row stays valid;
//! the utf8_* functions reject a non-null row that is not UTF-8 with `InvalidInput`. Layouts:
//! replace_substring, split_pattern, count_substring, find_substring and binary_repeat take the
//! string and binary layouts (string_view too, which Arrow does not register for split_pattern);
//! the pad functions and the whitespace splits the string layouts; binary_slice and
//! binary_reverse the binary layouts only (as in Arrow). binary_repeat takes an int64 count
//! (Arrow: any integer type). replace_substring and the forward splits of the offsets layouts
//! work over the whole value window of a chunk at once (no per-row output bookkeeping).
const std = @import("std");
const bitmap = @import("../bitmap.zig");
const core = @import("core.zig");
const datatype = @import("../datatype.zig");
const dense = @import("dense.zig");
const string = @import("string.zig");

const ArrayData = @import("../array/array_data.zig").ArrayData;
const ArrayRef = core.ArrayRef;
const DataType = core.DataType;
const Datum = core.Datum;
const ExecChunkValue = core.ExecChunkValue;
const ExecContext = core.ExecContext;
const FunctionRegistry = core.FunctionRegistry;
const KernelError = core.KernelError;
const Options = core.Options;
const OwnedBuffer = @import("../buffer.zig").OwnedBuffer;
const SharedBuffer = @import("../buffer.zig").SharedBuffer;
const TypeTag = std.meta.Tag(DataType);

const string_tags = [_]TypeTag{ .string, .large_string, .string_view };
const binary_tags = [_]TypeTag{ .binary, .large_binary, .binary_view };
const all_tags = string_tags ++ binary_tags;

/// Pad functions, by Arrow name.
const Pad = enum { ascii_lpad, ascii_rpad, ascii_center, utf8_lpad, utf8_rpad, utf8_center };
const offset_tags = [_]TypeTag{ .string, .large_string, .binary, .large_binary };

/// Register every function of the family for the layouts it supports.
pub fn register(registry: *FunctionRegistry) KernelError!void {
    inline for (all_tags) |tag| {
        const text = comptime isText(tag);
        try registry.registerScalarKernel("replace_substring", unary(tag, replaceOptions(text), sameType(tag), Transform(.replace).exec));
        try registry.registerScalarKernel("split_pattern", unary(tag, splitPatternOptions(text), dense.resultTypeOf(listOf(tag)), Splitter(.pattern).exec));
        try registry.registerScalarKernel("count_substring", unary(tag, matchOptions, dense.resultTypeOf(indexType(tag)), Search(.count).exec));
        try registry.registerScalarKernel("find_substring", unary(tag, matchOptions, dense.resultTypeOf(indexType(tag)), Search(.find).exec));
        try registry.registerScalarKernel("binary_repeat", .{
            .signature = .{ .arity = 2, .type_check = repeatArgs(tag), .options_check = dense.noOptions, .result_type_fn = sameType(tag) },
            .exec = repeat,
        });
    }
    inline for (string_tags) |tag| {
        try registry.registerScalarKernel("utf8_replace_slice", unary(tag, replaceSliceOptions(true), sameType(tag), Transform(.replace_slice_utf8).exec));
        try registry.registerScalarKernel("utf8_zero_fill", unary(tag, zeroFillOptions, sameType(tag), Transform(.zero_fill).exec));
        try registry.registerScalarKernel("ascii_split_whitespace", unary(tag, splitOptions, dense.resultTypeOf(listOf(tag)), Splitter(.ascii_whitespace).exec));
        try registry.registerScalarKernel("utf8_split_whitespace", unary(tag, splitOptions, dense.resultTypeOf(listOf(tag)), Splitter(.utf8_whitespace).exec));
        inline for (comptime std.meta.tags(Pad)) |p| {
            try registry.registerScalarKernel(@tagName(p), unary(tag, padOptions(p), sameType(tag), Transform(.{ .pad = p }).exec));
        }
    }
    // binary_join: list<T> / large_list<T> with a T separator.
    inline for (offset_tags) |tag| {
        inline for (.{ TypeTag.list, TypeTag.large_list }) |list_tag| {
            try registry.registerScalarKernel("binary_join", .{
                .signature = .{ .arity = 2, .type_check = joinArgs(list_tag, tag), .options_check = dense.noOptions, .result_type_fn = sameType(tag) },
                .exec = joinList,
            });
        }
    }
    inline for (binary_tags) |tag| {
        // Byte positions could split a code point, so the string layouts are not registered.
        try registry.registerScalarKernel("binary_replace_slice", unary(tag, replaceSliceOptions(false), sameType(tag), Transform(.replace_slice_bytes).exec));
        try registry.registerScalarKernel("binary_slice", unary(tag, sliceOptions, sameType(tag), Transform(.slice).exec));
        try registry.registerScalarKernel("binary_reverse", unary(tag, dense.noOptions, sameType(tag), Transform(.reverse).exec));
    }
}

fn unary(comptime tag: TypeTag, comptime options_check: core.OptionsCheckFn, comptime result: core.ResultTypeFn, comptime exec: core.KernelExecFn) core.Kernel {
    return .{
        .signature = .{ .arity = 1, .type_check = dense.argsOfType(tag, 1), .options_check = options_check, .result_type_fn = result },
        .exec = exec,
    };
}

fn isText(comptime tag: TypeTag) bool {
    return tag == .string or tag == .large_string or tag == .string_view;
}

fn sameType(comptime tag: TypeTag) core.ResultTypeFn {
    return dense.resultTypeOf(@unionInit(DataType, @tagName(tag), {}));
}

/// int64 for the large layouts, int32 otherwise (Arrow C++).
fn indexType(comptime tag: TypeTag) DataType {
    return if (tag == .large_string or tag == .large_binary) .{ .int64 = {} } else .{ .int32 = {} };
}

/// `list<T>` of the layout `T` (Arrow: split of large_string is list<large_string>).
pub fn listOf(comptime tag: TypeTag) DataType {
    const S = struct {
        const value: DataType = @unionInit(DataType, @tagName(tag), {});
    };
    return .{ .list = .{ .value_field = .{ .name = "item", .data_type = &S.value } } };
}

/// `listOf` the layout of `dt`.
fn listTypeFor(dt: DataType) KernelError!DataType {
    return switch (std.meta.activeTag(dt)) {
        inline .string, .large_string, .string_view, .binary, .large_binary, .binary_view => |tag| listOf(tag),
        else => error.UnsupportedType,
    };
}

fn repeatArgs(comptime tag: TypeTag) core.TypeCheckFn {
    return &struct {
        fn check(args: []const Datum) bool {
            return args.len == 2 and std.meta.activeTag(args[0].dataType()) == tag and args[1].dataType() == .int64;
        }
    }.check;
}

// ------------------------------------------------------------------------------ options

fn replaceOptions(comptime text: bool) core.OptionsCheckFn {
    return &struct {
        /// An empty pattern only with a bound (Arrow loops forever without one); UTF-8 for text.
        fn check(options: Options) bool {
            if (options != .replace_substring) return false;
            const o = options.replace_substring;
            if (o.pattern.len == 0 and o.max_replacements == null) return false;
            return !text or (std.unicode.utf8ValidateSlice(o.pattern) and std.unicode.utf8ValidateSlice(o.replacement));
        }
    }.check;
}

fn splitPatternOptions(comptime text: bool) core.OptionsCheckFn {
    return &struct {
        /// A non-empty separator (Arrow: "Empty separator"); UTF-8 for text.
        fn check(options: Options) bool {
            if (options != .split_pattern) return false;
            const p = options.split_pattern.pattern;
            return p.len > 0 and (!text or std.unicode.utf8ValidateSlice(p));
        }
    }.check;
}

/// `none` (split everywhere) or `SplitOptions`.
fn splitOptions(options: Options) bool {
    return options == .none or options == .split;
}

fn matchOptions(options: Options) bool {
    return options == .match_substring;
}

fn sliceOptions(options: Options) bool {
    return options == .slice and options.slice.step != 0;
}

fn replaceSliceOptions(comptime text: bool) core.OptionsCheckFn {
    return &struct {
        /// A UTF-8 replacement for text, so a valid row stays valid.
        fn check(options: Options) bool {
            return options == .replace_slice and (!text or std.unicode.utf8ValidateSlice(options.replace_slice.replacement));
        }
    }.check;
}

/// One code point of padding and a width that fits an offset.
fn zeroFillOptions(options: Options) bool {
    if (options != .zero_fill) return false;
    const o = options.zero_fill;
    if (o.width > max_width or o.padding.len == 0) return false;
    const n = std.unicode.utf8ByteSequenceLength(o.padding[0]) catch return false;
    return n == o.padding.len and std.unicode.utf8ValidateSlice(o.padding);
}

/// `list_tag` of `tag` items, then a `tag` separator.
fn joinArgs(comptime list_tag: TypeTag, comptime tag: TypeTag) core.TypeCheckFn {
    return &struct {
        fn check(args: []const Datum) bool {
            if (args.len != 2 or std.meta.activeTag(args[0].dataType()) != list_tag or std.meta.activeTag(args[1].dataType()) != tag) return false;
            const item = switch (args[0].dataType()) {
                .list, .large_list => |l| l.value_field.data_type.*,
                else => return false,
            };
            return std.meta.activeTag(item) == tag;
        }
    }.check;
}

/// Widths past `maxInt(i32)` cannot be stored in a 32-bit offset layout and are rejected for all.
const max_width: usize = std.math.maxInt(i32);

fn padOptions(comptime p: Pad) core.OptionsCheckFn {
    return &struct {
        /// One ASCII byte (ascii_*) or one code point (utf8_*) of padding.
        fn check(options: Options) bool {
            if (options != .pad) return false;
            const o = options.pad;
            if (o.width > max_width or o.padding.len == 0) return false;
            if (comptime isAsciiPad(p)) return o.padding.len == 1 and o.padding[0] < 0x80;
            const n = std.unicode.utf8ByteSequenceLength(o.padding[0]) catch return false;
            return n == o.padding.len and std.unicode.utf8ValidateSlice(o.padding);
        }
    }.check;
}

fn isAsciiPad(comptime p: Pad) bool {
    return p == .ascii_lpad or p == .ascii_rpad or p == .ascii_center;
}

// ------------------------------------------------------------------------------ row driver

/// Call `row(state, &out, bytes)` for every valid row of `values[0]` (it commits exactly one row
/// of `out`); null rows commit nothing. With `utf8`, a valid row that is not UTF-8 is `InvalidInput`.
fn mapRows(ctx: *ExecContext, value: ExecChunkValue, len: usize, out_type: DataType, comptime utf8: bool, state: anytype, comptime row: anytype) KernelError!ArrayRef {
    var arr = try dense.chunkArray(ctx.allocator, value);
    defer arr.release();
    const data = arr.data();
    const rows = try dense.StringOperand.init(.{ .array = arr });
    var out = try dense.VarBinaryOutput.init(ctx.allocator, out_type, len, try dense.combinedValidity(ctx.allocator, len, &.{data}), dense.totalBytes(rows, len));
    errdefer out.deinit();
    const bits = dense.validityBits(data);
    switch (rows) {
        inline else => |r| for (0..len) |i| {
            if (bits) |b| if (!bitmap.bitIsSet(b, data.offset + i)) {
                try out.commit(0);
                continue;
            };
            const src = r.get(i);
            if (utf8 and !string.validUtf8(src)) return error.InvalidInput;
            try row(&state, &out, src);
        },
    }
    return out.finish();
}

// ------------------------------------------------------------------------------ transforms

const TransformKind = union(enum) { replace, slice, reverse, pad: Pad, replace_slice_utf8, replace_slice_bytes, zero_fill };

fn Transform(comptime kind: TransformKind) type {
    return struct {
        /// Transform each row (array, chunked or scalar).
        fn exec(ctx: *ExecContext, args: []const Datum, options: Options) KernelError!Datum {
            return dense.execElementwiseWith(ctx, args, options, args[0].dataType(), true, chunk);
        }

        fn chunk(ctx: *ExecContext, values: []const ExecChunkValue, len: usize, options: Options) KernelError!ArrayRef {
            const utf8 = comptime switch (kind) {
                .pad => |p| !isAsciiPad(p),
                .replace_slice_utf8, .zero_fill => true,
                else => false,
            };
            if (kind == .replace) if (try replaceWindow(ctx, values[0], len, options.replace_substring)) |arr| return arr;
            var scratch: std.ArrayList(usize) = .empty;
            defer scratch.deinit(ctx.allocator);
            const state: State = .{ .allocator = ctx.allocator, .options = options, .scratch = &scratch };
            return mapRows(ctx, values[0], len, values[0].dataType(), utf8, state, row);
        }

        const State = struct { allocator: std.mem.Allocator, options: Options, scratch: *std.ArrayList(usize) };

        fn row(s: *const State, out: *dense.VarBinaryOutput, src: []const u8) KernelError!void {
            switch (kind) {
                .replace => {
                    const o = &s.options.replace_substring;
                    const limit = o.max_replacements orelse std.math.maxInt(usize);
                    const r = o.replacement;
                    if (o.pattern.len == 0) {
                        // Arrow inserts the replacement `limit` times before a non-empty row.
                        const n = if (src.len == 0 or r.len == 0) 0 else limit;
                        const size = std.math.add(usize, src.len, std.math.mul(usize, n, r.len) catch return error.Overflow) catch return error.Overflow;
                        const dst = try out.reserve(size);
                        for (0..n) |k| @memcpy(dst[k * r.len ..][0..r.len], r);
                        @memcpy(dst[n * r.len ..], src);
                        return out.commit(size);
                    }
                    // One pass into a slot that fits every possible match replaced (a shift bounds
                    // the match count without a division).
                    const most = @min(limit, src.len >> std.math.log2_int(usize, o.pattern.len));
                    const cap = std.math.add(usize, src.len, std.math.mul(usize, most, r.len -| o.pattern.len) catch return error.Overflow) catch return error.Overflow;
                    const dst = try out.reserve(cap);
                    var w: usize = 0;
                    var i: usize = 0;
                    var n: usize = 0;
                    while (n < limit) : (n += 1) {
                        const pos = std.mem.indexOfPos(u8, src, i, o.pattern) orelse break;
                        dense.copyBytes(dst[w..], src[i..pos]);
                        w += pos - i;
                        dense.copyBytes(dst[w..], r);
                        w += r.len;
                        i = pos + o.pattern.len;
                    }
                    dense.copyBytes(dst[w..], src[i..]);
                    try out.commit(w + src.len - i);
                },
                .slice => try out.commit(try string.slice(true, s.allocator, try out.reserve(src.len), src, s.options.slice, s.scratch)),
                .reverse => {
                    const dst = try out.reserve(src.len);
                    for (src, 0..) |c, k| dst[src.len - 1 - k] = c;
                    try out.commit(src.len);
                },
                .replace_slice_utf8, .replace_slice_bytes => {
                    const o = &s.options.replace_slice;
                    // Positions in code points (utf8) or bytes, mapped to byte offsets.
                    var n: usize = src.len;
                    if (kind == .replace_slice_utf8) {
                        s.scratch.clearRetainingCapacity();
                        var i: usize = 0;
                        while (i < src.len) : (i += string.cpLen(src[i])) s.scratch.append(s.allocator, i) catch return error.OutOfMemory;
                        s.scratch.append(s.allocator, src.len) catch return error.OutOfMemory;
                        n = s.scratch.items.len - 1;
                    }
                    const begin = clampPosition(o.start, n);
                    const end = @max(begin, clampPosition(o.stop, n));
                    const lo = if (kind == .replace_slice_utf8) s.scratch.items[begin] else begin;
                    const hi = if (kind == .replace_slice_utf8) s.scratch.items[end] else end;
                    const size = std.math.add(usize, lo + (src.len - hi), o.replacement.len) catch return error.Overflow;
                    const dst = try out.reserve(size);
                    @memcpy(dst[0..lo], src[0..lo]);
                    @memcpy(dst[lo..][0..o.replacement.len], o.replacement);
                    @memcpy(dst[lo + o.replacement.len ..][0 .. src.len - hi], src[hi..]);
                    try out.commit(size);
                },
                .zero_fill => {
                    const o = &s.options.zero_fill;
                    const n = std.unicode.utf8CountCodepoints(src) catch unreachable;
                    const total = o.width -| n;
                    const sign: usize = if (src.len > 0 and (src[0] == '+' or src[0] == '-')) 1 else 0;
                    const size = src.len + total * o.padding.len;
                    const dst = try out.reserve(size);
                    @memcpy(dst[0..sign], src[0..sign]);
                    var w: usize = sign;
                    for (0..total) |_| {
                        @memcpy(dst[w..][0..o.padding.len], o.padding);
                        w += o.padding.len;
                    }
                    @memcpy(dst[w..][0 .. src.len - sign], src[sign..]);
                    try out.commit(size);
                },
                .pad => |p| {
                    const o = &s.options.pad;
                    const n = if (comptime isAsciiPad(p)) src.len else std.unicode.utf8CountCodepoints(src) catch unreachable;
                    const total = o.width -| n;
                    const left = switch (p) {
                        .ascii_lpad, .utf8_lpad => total,
                        .ascii_rpad, .utf8_rpad => 0,
                        else => if (o.lean_left_on_odd_padding) total / 2 else total - total / 2,
                    };
                    const size = src.len + total * o.padding.len;
                    const dst = try out.reserve(size);
                    var w: usize = 0;
                    for (0..left) |_| {
                        @memcpy(dst[w..][0..o.padding.len], o.padding);
                        w += o.padding.len;
                    }
                    @memcpy(dst[w..][0..src.len], src);
                    w += src.len;
                    for (0..total - left) |_| {
                        @memcpy(dst[w..][0..o.padding.len], o.padding);
                        w += o.padding.len;
                    }
                    try out.commit(size);
                },
            }
        }
    };
}

/// A replace_slice position in `[0, n]`: negative counts from the end (Arrow C++).
fn clampPosition(pos: i64, n: usize) usize {
    const len: i64 = @intCast(n);
    const p = if (pos < 0) @max(len +| pos, 0) else @min(pos, len);
    return @intCast(p);
}

// ------------------------------------------------------------------------------ binary_join

/// Join the items of each list row with the separator of the row.
fn joinList(ctx: *ExecContext, args: []const Datum, options: Options) KernelError!Datum {
    return dense.execElementwiseWith(ctx, args, options, args[1].dataType(), true, joinListChunk);
}

fn joinListChunk(ctx: *ExecContext, values: []const ExecChunkValue, len: usize, _: Options) KernelError!ArrayRef {
    var lists = try dense.chunkArray(ctx.allocator, values[0]);
    defer lists.release();
    const ld = lists.data();
    if (ld.children.len != 1) return error.InvalidInput;
    const child = ld.children[0].data();
    // validateLayout of the list bounds its offsets by the child length; the child's own buffers
    // are checked here, and the child must have the separator's layout.
    child.validateLayout() catch return error.InvalidInput;
    const out_type = values[1].dataType();
    if (std.meta.activeTag(datatype.storageDataType(child.data_type)) != std.meta.activeTag(datatype.storageDataType(out_type))) return error.InvalidInput;
    var child_ref = ld.children[0].retain();
    defer child_ref.release();
    const items = try dense.StringOperand.init(.{ .array = child_ref });
    const seps = try dense.StringOperand.init(values[1]);
    return switch (ld.data_type) {
        .list => joinListRows(i32, ctx, ld, items, seps, values[1], len, out_type),
        .large_list => joinListRows(i64, ctx, ld, items, seps, values[1], len, out_type),
        else => error.UnsupportedType,
    };
}

fn joinListRows(comptime O: type, ctx: *ExecContext, ld: *const ArrayData, items: dense.StringOperand, seps: dense.StringOperand, sep_value: ExecChunkValue, len: usize, out_type: DataType) KernelError!ArrayRef {
    const all = ld.buffers[1].typedSlice(O) catch return error.InvalidInput;
    const offsets = all[ld.offset..][0 .. len + 1];
    const child = ld.children[0].data();
    // Pass 1: the validity and the byte total of every row.
    var validity = dense.bitsBuffer(ctx.allocator, len) catch return error.OutOfMemory;
    var validity_owned = true;
    defer if (validity_owned) validity.deinit();
    @memset(validity.data[0..bitmap.byteLength(len)], 0);
    var total: usize = 0;
    for (0..len) |i| {
        if (ld.isNull(i) or sep_value.isNullAt(i)) continue;
        const lo: usize = @intCast(offsets[i]);
        const hi: usize = @intCast(offsets[i + 1]);
        var n: usize = 0;
        var valid = true;
        for (lo..hi) |j| {
            if (child.isNull(j)) {
                valid = false;
                break;
            }
            n += rowOf(items, j).len;
        }
        if (!valid) continue;
        bitmap.setBit(validity.data, i);
        total += n + rowOf(seps, i).len * ((hi - lo) -| 1);
    }
    // Pass 2: fill.
    validity_owned = false;
    var out = try dense.VarBinaryOutput.init(ctx.allocator, out_type, len, validity, total);
    errdefer out.deinit();
    for (0..len) |i| {
        if (!bitmap.bitIsSet(out.validity.?.data, i)) {
            try out.commit(0);
            continue;
        }
        const sep = rowOf(seps, i);
        var w: usize = 0;
        for (@as(usize, @intCast(offsets[i]))..@as(usize, @intCast(offsets[i + 1]))) |j| {
            if (j > offsets[i]) {
                @memcpy((try out.reserve(w + sep.len))[w..], sep);
                w += sep.len;
            }
            const part = rowOf(items, j);
            @memcpy((try out.reserve(w + part.len))[w..], part);
            w += part.len;
        }
        try out.commit(w);
    }
    return out.finish();
}

fn rowOf(op: dense.StringOperand, i: usize) []const u8 {
    return switch (op) {
        inline else => |r| r.get(i),
    };
}

/// replace_substring over the whole value window of an offsets layout at once: the matches are
/// searched in the window (long haystacks take the vector paths of `std.mem.indexOfPos`) and
/// mapped to their rows; a match crossing a row end is not one. Null when the per-row path
/// must run instead (string_view layouts, an empty pattern, a replacement that could more than
/// double the data).
fn replaceWindow(ctx: *ExecContext, value: ExecChunkValue, len: usize, o: core.ReplaceSubstringOptions) KernelError!?ArrayRef {
    if (o.pattern.len == 0) return null;
    var arr = try dense.chunkArray(ctx.allocator, value);
    defer arr.release();
    const data = arr.data();
    return switch (try dense.StringOperand.init(.{ .array = arr })) {
        inline .offsets32, .offsets64 => |r| try replaceRows(ctx.allocator, data, r, len, o),
        else => null,
    };
}

fn replaceRows(allocator: std.mem.Allocator, data: *const @import("../array/array_data.zig").ArrayData, r: anytype, len: usize, o: core.ReplaceSubstringOptions) KernelError!?ArrayRef {
    const offs = r.offsets;
    const base: usize = @intCast(offs[0]);
    const window = r.data[base..@intCast(offs[len])];
    const grow = o.replacement.len -| o.pattern.len;
    const worst = std.math.add(usize, window.len, std.math.mul(usize, window.len >> std.math.log2_int(usize, o.pattern.len), grow) catch return null) catch return null;
    if (worst > 2 * window.len + (1 << 20)) return null;
    const limit = o.max_replacements orelse std.math.maxInt(usize);

    var out = try dense.VarBinaryOutput.init(allocator, data.data_type, len, try dense.combinedValidity(allocator, len, &.{data}), worst);
    errdefer out.deinit();
    const new_offs = allocator.alloc(i64, len + 1) catch return error.OutOfMemory;
    defer allocator.free(new_offs);
    const dst = try out.reserve(worst);
    var w: usize = 0; // bytes written
    var copied: usize = 0; // window bytes handled
    var pos: usize = 0; // search start
    var delta: i64 = 0; // output minus input bytes so far
    var row: usize = 0;
    var count: usize = 0; // replacements in `row`
    new_offs[0] = 0;
    const rowEnd = struct {
        inline fn f(of: anytype, b: usize, i: usize) usize {
            return @as(usize, @intCast(of[i + 1])) - b;
        }
    }.f;
    while (row < len) {
        const m = (if (o.pattern.len == 1) nextByte(window, pos, o.pattern[0]) else std.mem.indexOfPos(u8, window, pos, o.pattern)) orelse break;
        // Close the rows that end at or before the match.
        while (row < len and m >= rowEnd(offs, base, row)) : (row += 1) {
            new_offs[row + 1] = @as(i64, @intCast(rowEnd(offs, base, row))) + delta;
            count = 0;
        }
        if (row == len) break;
        const end = rowEnd(offs, base, row);
        if (m + o.pattern.len > end) {
            pos = m + 1; // crosses the row end: not a match, an overlapping one may start later
            continue;
        }
        if (count >= limit) {
            pos = end;
            continue;
        }
        dense.copyBytes(dst[w..], window[copied..m]);
        w += m - copied;
        dense.copyBytes(dst[w..], o.replacement);
        w += o.replacement.len;
        copied = m + o.pattern.len;
        pos = copied;
        delta += @as(i64, @intCast(o.replacement.len)) - @as(i64, @intCast(o.pattern.len));
        count += 1;
    }
    @memcpy(dst[w..][0 .. window.len - copied], window[copied..]);
    while (row < len) : (row += 1) new_offs[row + 1] = @as(i64, @intCast(rowEnd(offs, base, row))) + delta;
    try out.bulkOffsets(new_offs);
    return try out.finish();
}

// ------------------------------------------------------------------------------ repeat

/// Repeat each row `count` times (binary_repeat); a negative count in a valid row is `InvalidInput`.
fn repeat(ctx: *ExecContext, args: []const Datum, options: Options) KernelError!Datum {
    return dense.execElementwiseWith(ctx, args, options, args[0].dataType(), true, repeatChunk);
}

fn repeatChunk(ctx: *ExecContext, values: []const ExecChunkValue, len: usize, _: Options) KernelError!ArrayRef {
    const out_type = values[0].dataType();
    const rows = try dense.StringOperand.init(values[0]);
    const counts = try dense.operand(i64, values[1]);
    const validity = try dense.combinedValidity(ctx.allocator, len, &.{ dense.arrayData(values[0]), dense.arrayData(values[1]) });
    var out = try dense.VarBinaryOutput.init(ctx.allocator, out_type, len, validity, 0);
    errdefer out.deinit();
    // One row longer than a 32-bit offset (or view) can address is `Overflow` before allocating.
    const row_limit: usize = switch (datatype.storageDataType(out_type)) {
        .large_string, .large_binary => std.math.maxInt(usize),
        else => std.math.maxInt(i32),
    };
    switch (rows) {
        inline else => |r| switch (counts) {
            inline else => |c| for (0..len) |i| {
                if (out.validity) |v| if (!bitmap.bitIsSet(v.data, i)) {
                    try out.commit(0);
                    continue;
                };
                const src = r.get(i);
                const n = std.math.cast(usize, dense.at(c, i)) orelse return error.InvalidInput;
                const size = std.math.mul(usize, src.len, n) catch return error.Overflow;
                if (size > row_limit) return error.Overflow;
                const dst = try out.reserve(size);
                if (src.len != 0) for (0..n) |k| @memcpy(dst[k * src.len ..][0..src.len], src);
                try out.commit(size);
            },
        },
    }
    return out.finish();
}

// ------------------------------------------------------------------------------ search

const ones: u64 = 0x0101_0101_0101_0101;
const highs: u64 = ones << 7;

/// The 8 bytes of `h` from `i` (`i < h.len`) as a little-endian word; bytes past the end are 0
/// (callers reject a hit there by position).
inline fn word(h: []const u8, i: usize) u64 {
    if (i + 8 <= h.len) return std.mem.readInt(u64, h[i..][0..8], .little);
    if (h.len >= 8) return std.mem.readInt(u64, h[h.len - 8 ..][0..8], .little) >> @intCast((i + 8 - h.len) * 8);
    var w: u64 = 0;
    for (h[i..], 0..) |c, k| w |= @as(u64, c) << @intCast(k * 8);
    return w;
}

/// Position of the first `c` in `h` at or after `start`, 8 bytes at a time (no call per match,
/// unlike `std.mem.indexOfScalarPos`, which pays its vector setup on every call).
inline fn nextByte(h: []const u8, start: usize, c: u8) ?usize {
    const pattern = ones *% c;
    var i = start;
    while (i < h.len) : (i += 8) {
        const x = word(h, i) ^ pattern;
        const hits = (x -% ones) & ~x & highs; // the lowest set byte is the first match
        if (hits != 0) {
            const at = i + @ctz(hits) / 8;
            return if (at < h.len) at else null;
        }
    }
    return null;
}

/// Position of the first ASCII whitespace byte of `h` at or after `start` (`h.len` when none):
/// bytes below 0x21 are found 8 at a time, then checked.
fn nextSpace(h: []const u8, start: usize) usize {
    var i = start;
    while (i < h.len) {
        const x = word(h, i);
        const hits = (x -% ones *% 0x21) & ~x & highs; // bytes < 0x21 (the lowest one is exact)
        if (hits == 0) {
            i += 8;
            continue;
        }
        const at = i + @ctz(hits) / 8;
        if (at >= h.len) return h.len;
        if (std.ascii.isWhitespace(h[at])) return at;
        i = at + 1;
    }
    return h.len;
}

/// Position of `needle` in `haystack` at or after `start`, folding ASCII letters when `fold`.
fn indexOf(haystack: []const u8, start: usize, needle: []const u8, fold: bool) ?usize {
    if (!fold) return std.mem.indexOfPos(u8, haystack, start, needle);
    if (start > haystack.len or needle.len > haystack.len - start) return null;
    for (start..haystack.len - needle.len + 1) |i| {
        if (std.ascii.eqlIgnoreCase(haystack[i..][0..needle.len], needle)) return i;
    }
    return null;
}

fn Search(comptime which: enum { count, find }) type {
    return struct {
        /// Count the non-overlapping matches (count_substring) or return the byte index of the
        /// first one, -1 when absent (find_substring), of each row.
        fn exec(ctx: *ExecContext, args: []const Datum, options: Options) KernelError!Datum {
            const tag = std.meta.activeTag(args[0].dataType());
            const out_type: DataType = if (tag == .large_string or tag == .large_binary) .{ .int64 = {} } else .{ .int32 = {} };
            return dense.execElementwiseWith(ctx, args, options, out_type, true, chunk);
        }

        fn chunk(ctx: *ExecContext, values: []const ExecChunkValue, len: usize, options: Options) KernelError!ArrayRef {
            return switch (datatype.storageDataType(values[0].dataType())) {
                .large_string, .large_binary => run(i64, ctx, values[0], len, options.match_substring),
                else => run(i32, ctx, values[0], len, options.match_substring),
            };
        }

        fn run(comptime O: type, ctx: *ExecContext, value: ExecChunkValue, len: usize, o: core.MatchSubstringOptions) KernelError!ArrayRef {
            var arr = try dense.chunkArray(ctx.allocator, value);
            defer arr.release();
            var out = try dense.DenseBinaryOutput(O).initValidity(ctx.allocator, len, arr.data(), null, dense.dataTypeOf(O));
            errdefer out.deinit();
            switch (try dense.StringOperand.init(.{ .array = arr })) {
                inline else => |r| for (out.values(), 0..) |*dst, i| {
                    const s = r.get(i);
                    if (which == .find) {
                        dst.* = if (indexOf(s, 0, o.pattern, o.ignore_case)) |p| std.math.cast(O, p) orelse return error.Overflow else -1;
                        continue;
                    }
                    // An empty pattern matches at every position, the end included (Arrow: len + 1).
                    var n: usize = 0;
                    var start: usize = 0;
                    while (indexOf(s, start, o.pattern, o.ignore_case)) |p| {
                        n += 1;
                        start = p + @max(o.pattern.len, 1);
                        if (start > s.len) break;
                    }
                    dst.* = std.math.cast(O, n) orelse return error.Overflow;
                },
            }
            return out.finish() catch error.OutOfMemory;
        }
    };
}

// ------------------------------------------------------------------------------ split

const SplitKind = enum { pattern, ascii_whitespace, utf8_whitespace };

/// A separator `[start, end)` inside a row.
const Sep = struct { start: usize, end: usize };

fn Splitter(comptime kind: SplitKind) type {
    return struct {
        /// Split each row into a list of parts (array, chunked or scalar).
        fn exec(ctx: *ExecContext, args: []const Datum, options: Options) KernelError!Datum {
            return dense.execElementwiseWith(ctx, args, options, try listTypeFor(args[0].dataType()), true, chunk);
        }

        fn limits(options: Options) struct { max: usize, reverse: bool } {
            return switch (options) {
                .split_pattern => |o| .{ .max = o.max_splits orelse std.math.maxInt(usize), .reverse = o.reverse },
                .split => |o| .{ .max = o.max_splits orelse std.math.maxInt(usize), .reverse = o.reverse },
                else => .{ .max = std.math.maxInt(usize), .reverse = false },
            };
        }

        /// Fill `seps` with the separators used to split `s`, left to right.
        fn separators(allocator: std.mem.Allocator, seps: *std.ArrayList(Sep), s: []const u8, options: Options) KernelError!void {
            seps.clearRetainingCapacity();
            const l = limits(options);
            switch (kind) {
                .pattern => {
                    const p = options.split_pattern.pattern;
                    if (!l.reverse) {
                        var i: usize = 0;
                        while (seps.items.len < l.max) {
                            const pos = std.mem.indexOfPos(u8, s, i, p) orelse break;
                            seps.append(allocator, .{ .start = pos, .end = pos + p.len }) catch return error.OutOfMemory;
                            i = pos + p.len;
                        }
                    } else {
                        // Search from the right, like Arrow (matters for self-overlapping patterns).
                        var end = s.len;
                        while (seps.items.len < l.max) {
                            const pos = std.mem.lastIndexOf(u8, s[0..end], p) orelse break;
                            seps.append(allocator, .{ .start = pos, .end = pos + p.len }) catch return error.OutOfMemory;
                            end = pos;
                        }
                        std.mem.reverse(Sep, seps.items);
                    }
                },
                .ascii_whitespace, .utf8_whitespace => {
                    // Runs of whitespace are the same from either end: find all, keep the first
                    // or last `max`. Leading and trailing runs give empty parts (Arrow C++).
                    var i: usize = 0;
                    if (kind == .ascii_whitespace) {
                        while (true) {
                            const begin = nextSpace(s, i);
                            if (begin == s.len) break;
                            var stop = begin + 1;
                            while (stop < s.len and std.ascii.isWhitespace(s[stop])) stop += 1;
                            seps.append(allocator, .{ .start = begin, .end = stop }) catch return error.OutOfMemory;
                            i = stop;
                        }
                    }
                    while (kind == .utf8_whitespace and i < s.len) {
                        const n = spaceLen(s, i);
                        if (n == 0) {
                            i += if (kind == .utf8_whitespace) string.cpLen(s[i]) else 1;
                            continue;
                        }
                        const start = i;
                        while (i < s.len) {
                            const m = spaceLen(s, i);
                            if (m == 0) break;
                            i += m;
                        }
                        seps.append(allocator, .{ .start = start, .end = i }) catch return error.OutOfMemory;
                    }
                    if (seps.items.len > l.max) {
                        if (l.reverse) {
                            const drop = seps.items.len - l.max;
                            std.mem.copyForwards(Sep, seps.items[0..l.max], seps.items[drop..]);
                        }
                        seps.shrinkRetainingCapacity(l.max);
                    }
                },
            }
        }

        /// Next separator `[start, end)` of `s` at or after `from`, null when there is none.
        inline fn nextSep(s: []const u8, from: usize, options: Options) ?Sep {
            switch (kind) {
                .pattern => {
                    const p = options.split_pattern.pattern;
                    const at = std.mem.indexOfPos(u8, s, from, p) orelse return null;
                    return .{ .start = at, .end = at + p.len };
                },
                .ascii_whitespace => {
                    const at = nextSpace(s, from);
                    if (at == s.len) return null;
                    var stop = at + 1;
                    while (stop < s.len and std.ascii.isWhitespace(s[stop])) stop += 1;
                    return .{ .start = at, .end = stop };
                },
                .utf8_whitespace => {
                    var i = from;
                    while (i < s.len) {
                        const n = spaceLen(s, i);
                        if (n == 0) {
                            i += string.cpLen(s[i]);
                            continue;
                        }
                        var stop = i + n;
                        while (stop < s.len) {
                            const m = spaceLen(s, stop);
                            if (m == 0) break;
                            stop += m;
                        }
                        return .{ .start = i, .end = stop };
                    }
                    return null;
                },
            }
        }

        /// The list array of an offsets layout, the parts written in one pass: the child bytes
        /// are at most the input window, the child offsets grow as parts are found.
        fn splitRows(allocator: std.mem.Allocator, data: *const @import("../array/array_data.zig").ArrayData, r: anytype, bits: ?[]const u8, len: usize, options: Options) KernelError!ArrayRef {
            const O = std.meta.Child(@TypeOf(r.offsets));
            const max = limits(options).max;
            const window_len: usize = @intCast(r.offsets[len] - r.offsets[0]);
            var child_data = OwnedBuffer.initUninit(allocator, window_len) catch return error.OutOfMemory;
            defer child_data.deinit();
            var child_offs = OwnedBuffer.initUninit(allocator, (2 * len + 1) * @sizeOf(O)) catch return error.OutOfMemory;
            defer child_offs.deinit();
            var list_offs = OwnedBuffer.initUninit(allocator, (len + 1) * 4) catch return error.OutOfMemory;
            defer list_offs.deinit();
            const lo = list_offs.typedSlice(i32)[0 .. len + 1];
            lo[0] = 0;
            var parts: usize = 0;
            var w: usize = 0;
            child_offs.typedSlice(O)[0] = 0;
            for (0..len) |i| {
                if (bits == null or bitmap.bitIsSet(bits.?, data.offset + i)) {
                    const s = r.get(i);
                    if (kind == .utf8_whitespace and !string.validUtf8(s)) return error.InvalidInput;
                    var start: usize = 0;
                    var splits: usize = 0;
                    while (true) {
                        const sep: ?Sep = if (splits < max) nextSep(s, start, options) else null;
                        const stop = if (sep) |x| x.start else s.len;
                        // Room for this part's end offset.
                        if ((parts + 2) * @sizeOf(O) > child_offs.data.len) child_offs.resize(child_offs.data.len * 2) catch return error.OutOfMemory;
                        dense.copyBytes(child_data.data[w..], s[start..stop]);
                        w += stop - start;
                        parts += 1;
                        child_offs.typedSlice(O)[parts] = std.math.cast(O, w) orelse return error.Overflow;
                        const x = sep orelse break;
                        start = x.end;
                        splits += 1;
                    }
                }
                lo[i + 1] = std.math.cast(i32, parts) orelse return error.Overflow;
            }
            var child_bufs: [3]SharedBuffer = .{ .empty, .empty, .empty };
            errdefer for (&child_bufs) |*b| b.release();
            child_bufs[1] = child_offs.toShared((parts + 1) * @sizeOf(O)) catch return error.OutOfMemory;
            child_bufs[2] = child_data.toShared(w) catch return error.OutOfMemory;
            var child = ArrayRef.fromOwnedInlineBuffers(allocator, .{ .data_type = data.data_type, .length = parts, .null_count = 0, .buffers = &child_bufs }) catch return error.OutOfMemory;
            child_bufs = .{ .empty, .empty, .empty };
            errdefer child.release();

            var validity = try dense.combinedValidity(allocator, len, &.{data});
            errdefer if (validity) |*v| v.deinit();
            var buffers: [2]SharedBuffer = .{ .empty, .empty };
            var null_count: usize = 0;
            errdefer for (&buffers) |*b| b.release();
            if (validity) |*v| {
                null_count = bitmap.countUnsetBit(v.data, len);
                buffers[0] = v.toShared(bitmap.byteLength(len)) catch return error.OutOfMemory;
                validity = null;
            }
            buffers[1] = list_offs.toShared((len + 1) * 4) catch return error.OutOfMemory;
            var children = [1]ArrayRef{child};
            return ArrayRef.fromOwnedInline(allocator, .{
                .data_type = try listTypeFor(data.data_type),
                .length = len,
                .null_count = null_count,
                .buffers = &buffers,
                .children = &children,
            }) catch error.OutOfMemory;
        }

        /// Byte length of the whitespace code point at `s[i]`, 0 when it is not whitespace.
        fn spaceLen(s: []const u8, i: usize) usize {
            if (kind == .ascii_whitespace) return @intFromBool(std.ascii.isWhitespace(s[i]));
            const n = string.cpLen(s[i]);
            return if (string.isUnicodeSpace(string.decodeAt(s, i))) n else 0;
        }

        fn chunk(ctx: *ExecContext, values: []const ExecChunkValue, len: usize, options: Options) KernelError!ArrayRef {
            const allocator = ctx.allocator;
            var arr = try dense.chunkArray(allocator, values[0]);
            defer arr.release();
            const data = arr.data();
            const rows = try dense.StringOperand.init(.{ .array = arr });
            const bits = dense.validityBits(data);
            const l = limits(options);
            // Offsets layouts split in one pass straight into the child (`reverse` searches from
            // the end: the generic path below).
            if (!l.reverse) switch (rows) {
                inline .offsets32, .offsets64 => |r| return splitRows(allocator, data, r, bits, len, options),
                else => {},
            };
            var seps: std.ArrayList(Sep) = .empty;
            defer seps.deinit(allocator);

            // The parts of every row (borrowed from the input) and the list offsets; null rows are
            // empty lists under the null bit.
            var offsets = OwnedBuffer.initUninit(allocator, (len + 1) * 4) catch return error.OutOfMemory;
            defer offsets.deinit();
            const offs = offsets.typedSlice(i32)[0 .. len + 1];
            offs[0] = 0;
            var parts: std.ArrayList([]const u8) = .empty;
            defer parts.deinit(allocator);
            switch (rows) {
                inline else => |r| for (0..len) |i| {
                    if (bits == null or bitmap.bitIsSet(bits.?, data.offset + i)) {
                        const s = r.get(i);
                        if (kind == .utf8_whitespace and !string.validUtf8(s)) return error.InvalidInput;
                        try separators(allocator, &seps, s, options);
                        parts.ensureUnusedCapacity(allocator, seps.items.len + 1) catch return error.OutOfMemory;
                        var start: usize = 0;
                        for (seps.items) |sep| {
                            parts.appendAssumeCapacity(s[start..sep.start]);
                            start = sep.end;
                        }
                        parts.appendAssumeCapacity(s[start..]);
                    }
                    offs[i + 1] = std.math.cast(i32, parts.items.len) orelse return error.Overflow;
                },
            }

            // The parts as an array of the input layout.
            var child = try dense.VarBinaryOutput.init(allocator, data.data_type, parts.items.len, null, dense.totalBytes(rows, len));
            errdefer child.deinit();
            for (parts.items) |part| try child.append(part);
            var values_arr = try child.finish();
            errdefer values_arr.release();

            var validity = try dense.combinedValidity(allocator, len, &.{data});
            errdefer if (validity) |*v| v.deinit();
            var buffers: [2]SharedBuffer = .{ .empty, .empty };
            var null_count: usize = 0;
            errdefer for (&buffers) |*b| b.release();
            if (validity) |*v| {
                null_count = bitmap.countUnsetBit(v.data, len);
                buffers[0] = v.toShared(bitmap.byteLength(len)) catch return error.OutOfMemory;
                validity = null;
            }
            buffers[1] = offsets.toShared((len + 1) * 4) catch return error.OutOfMemory;
            var children = [1]ArrayRef{values_arr};
            return ArrayRef.fromOwnedInline(allocator, .{
                .data_type = try listTypeFor(data.data_type),
                .length = len,
                .null_count = null_count,
                .buffers = &buffers,
                .children = &children,
            }) catch error.OutOfMemory;
        }
    };
}
