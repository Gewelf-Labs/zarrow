//! utf8_normalize (Arrow C++ "String transforms"): Unicode normalization forms NFC, NFD, NFKC
//! and NFKD (`Utf8NormalizeOptions.form`, NFC by default) on string, large_string and string_view
//! (pyarrow 26 has no string_view kernel); the result has the input type, nulls stay null.
//!
//! The algorithm is UAX #15's over the Unicode 17.0.0 tables of `unicode_props.zig`
//! (`tools/gen_unicode_tables.py`): full canonical or compatibility decomposition (Hangul
//! syllables algorithmically), canonical ordering by combining class, and for NFC / NFKC
//! canonical composition (primary composites only, Hangul algorithmically). A row that passes
//! the quick check of its form (no code point that the form changes, combining classes in
//! order) is copied as is; an input whose rows all pass (always, for ASCII) comes back
//! unchanged without a copy. A row that is not UTF-8 is `InvalidInput` (Arrow: "Invalid UTF-8
//! string"). Bounds: a code point decomposes to at most `max_decomposition` (18) code points;
//! the decomposition buffer of a row is sized by that bound and checked, and offsets past the
//! output layout are `Overflow`.
const std = @import("std");
const bitmap = @import("../bitmap.zig");
const core = @import("core.zig");
const dense = @import("dense.zig");
const string = @import("string.zig");
const props = @import("unicode_props.zig");

const ArrayRef = core.ArrayRef;
const DataType = core.DataType;
const Datum = core.Datum;
const ExecChunkValue = core.ExecChunkValue;
const ExecContext = core.ExecContext;
const FunctionRegistry = core.FunctionRegistry;
const KernelError = core.KernelError;
const Options = core.Options;
const TypeTag = std.meta.Tag(DataType);

/// A normalization form (Arrow `Utf8NormalizeOptions::Form`).
pub const Form = core.Utf8NormalizeOptions.Form;

pub fn register(registry: *FunctionRegistry) KernelError!void {
    inline for (.{ TypeTag.string, TypeTag.large_string, TypeTag.string_view }) |tag| {
        try registry.registerScalarKernel("utf8_normalize", .{
            .signature = .{ .arity = 1, .type_check = dense.argsOfType(tag, 1), .options_check = normalizeOptions, .result_type_fn = dense.resultTypeOf(@unionInit(DataType, @tagName(tag), {})) },
            .exec = exec,
        });
    }
}

fn normalizeOptions(options: Options) bool {
    return options == .none or options == .utf8_normalize;
}

const ccc_mask: u16 = 0xFF;
const has_canon: u16 = 1 << 8;
const has_compat: u16 = 1 << 9;
const excluded: u16 = 1 << 10;
const second: u16 = 1 << 11;

/// The properties of `cp` (see `unicode_props.zig`).
inline fn prop(cp: u21) u16 {
    if (cp >= props.props_limit) return 0;
    const block: usize = props.props_index[cp >> props.props_shift];
    return props.props_blocks[block << props.props_shift | (cp & ((1 << props.props_shift) - 1))];
}

/// Properties that make the quick check of `form` fail (No or Maybe).
fn quickMask(form: Form) u16 {
    return switch (form) {
        .nfd => has_canon,
        .nfkd => has_canon | has_compat,
        .nfc => excluded | second,
        .nfkc => has_compat | excluded | second,
    };
}

/// Whether `row` (valid UTF-8) is already in `form`: no code point of `mask`, combining classes
/// in order. Null when `row` is not UTF-8.
fn quickCheck(row: []const u8, mask: u16) ?bool {
    var i: usize = 0;
    var last: u16 = 0;
    while (i < row.len) {
        const c0 = row[i];
        if (c0 < 0x80) {
            i += 1;
            last = 0;
            continue;
        }
        var cp: u21 = undefined;
        // Two-byte sequences (Latin, Greek, Cyrillic, ...) inline; the rest through decodeStrict.
        if (c0 >= 0xC2 and c0 < 0xE0 and i + 1 < row.len and row[i + 1] & 0xC0 == 0x80) {
            cp = (@as(u21, c0 & 0x1F) << 6) | (row[i + 1] & 0x3F);
            i += 2;
        } else {
            const d = string.decodeStrict(row, i) orelse return null;
            cp = d.cp;
            i += d.len;
        }
        const p = prop(cp);
        const ccc = p & ccc_mask;
        // Out of form: the caller normalizes the row, which checks the rest of its UTF-8.
        if (p & mask != 0 or (ccc != 0 and last > ccc)) return false;
        last = ccc;
    }
    return true;
}

const hangul_s: u21 = 0xAC00;
const hangul_l: u21 = 0x1100;
const hangul_v: u21 = 0x1161;
const hangul_t: u21 = 0x11A7;
const hangul_v_count = 21;
const hangul_t_count = 28;
const hangul_count = 19 * hangul_v_count * hangul_t_count;

/// The full decomposition of `cp` for `compat` (null: `cp` itself).
fn decomposition(cp: u21, compat: bool) ?[]const u21 {
    var lo: usize = 0;
    var hi: usize = props.decompositions.len;
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        const e: props.Decomposition = @bitCast(props.decompositions[mid]);
        if (e.cp < cp) {
            lo = mid + 1;
        } else if (e.cp > cp) {
            hi = mid;
        } else {
            const off: usize = if (compat) e.compat_off else e.canon_off;
            const len: usize = if (compat) e.compat_len else e.canon_len;
            return props.decomposition_data[off..][0..len];
        }
    }
    return null;
}

/// The primary composite of `a` and `b`, or null.
fn compose(a: u21, b: u21) ?u21 {
    if (a >= hangul_l and a < hangul_l + 19 and b >= hangul_v and b < hangul_v + hangul_v_count)
        return hangul_s + ((a - hangul_l) * hangul_v_count + (b - hangul_v)) * hangul_t_count;
    if (a >= hangul_s and a < hangul_s + hangul_count and (a - hangul_s) % hangul_t_count == 0 and b > hangul_t and b < hangul_t + hangul_t_count)
        return a + (b - hangul_t);
    const key = @as(u64, a) << 21 | b;
    const keys = &props.composition_keys;
    var lo: usize = 0;
    var hi: usize = keys.len;
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        if (keys[mid] < key) lo = mid + 1 else hi = mid;
    }
    return if (lo < keys.len and keys[lo] == key) props.composition_values[lo] else null;
}

/// Normalize `row` (UTF-8) into `cps` (cleared first). `InvalidInput` when `row` is not UTF-8.
pub fn normalizeRow(allocator: std.mem.Allocator, row: []const u8, form: Form, cps: *std.ArrayList(u21)) KernelError!void {
    cps.clearRetainingCapacity();
    // A code point (at least one byte) decomposes to at most max_decomposition code points.
    cps.ensureTotalCapacity(allocator, row.len * props.max_decomposition) catch return error.OutOfMemory;
    const compat = form == .nfkc or form == .nfkd;
    // Decompose; every code point grows to at most max_decomposition code points.
    var i: usize = 0;
    while (i < row.len) {
        const d = string.decodeStrict(row, i) orelse return error.InvalidInput;
        i += d.len;
        const cp = d.cp;
        if (cp >= hangul_s and cp < hangul_s + hangul_count) {
            const s = cp - hangul_s;
            cps.appendSliceAssumeCapacity(&.{ hangul_l + s / (hangul_v_count * hangul_t_count), hangul_v + s % (hangul_v_count * hangul_t_count) / hangul_t_count });
            if (s % hangul_t_count != 0) cps.appendAssumeCapacity(hangul_t + s % hangul_t_count);
            continue;
        }
        const p = prop(cp);
        if (p & (if (compat) has_canon | has_compat else has_canon) != 0) {
            if (decomposition(cp, compat)) |seq| {
                // The cap the capacity above relies on (the generator checks it for every entry).
                if (seq.len > props.max_decomposition) return error.InvalidInput;
                cps.appendSliceAssumeCapacity(seq);
                continue;
            }
        }
        cps.appendAssumeCapacity(cp);
    }
    const buf = cps.items;
    // Canonical ordering: a stable insertion sort of every run of non-starters by class.
    var k: usize = 1;
    while (k < buf.len) : (k += 1) {
        const c = buf[k];
        const cc = prop(c) & ccc_mask;
        if (cc == 0) continue;
        var j = k;
        while (j > 0) {
            const pc = prop(buf[j - 1]) & ccc_mask;
            if (pc <= cc) break;
            buf[j] = buf[j - 1];
            j -= 1;
        }
        buf[j] = c;
    }
    if (form == .nfd or form == .nfkd or buf.len < 2) return;
    // Canonical composition (the UAX #15 sample algorithm).
    var starter: usize = 0;
    var last_class: u16 = prop(buf[0]) & ccc_mask;
    if (last_class != 0) last_class = 256; // a leading non-starter composes with nothing
    var out: usize = 1;
    for (buf[1..]) |c| {
        const pc = prop(c);
        const cc = pc & ccc_mask;
        // Only a code point that is the second of some composite can combine.
        if (pc & second != 0 and (last_class < cc or last_class == 0)) if (compose(buf[starter], c)) |comp| {
            buf[starter] = comp;
            continue;
        };
        if (cc == 0) starter = out;
        last_class = cc;
        buf[out] = c;
        out += 1;
    }
    cps.shrinkRetainingCapacity(out);
}

fn exec(ctx: *ExecContext, args: []const Datum, options: Options) KernelError!Datum {
    if (args.len != 1) return error.InvalidArity;
    return dense.execElementwiseWith(ctx, args, options, args[0].dataType(), true, chunk);
}

fn chunk(ctx: *ExecContext, values: []const ExecChunkValue, len: usize, options: Options) KernelError!ArrayRef {
    const form: Form = if (options == .utf8_normalize) options.utf8_normalize.form else .nfc;
    var arr = try dense.chunkArray(ctx.allocator, values[0]);
    defer arr.release();
    const data = arr.data();
    const rows = try dense.StringOperand.init(.{ .array = arr });
    const bits = dense.validityBits(data);
    const mask = quickMask(form);
    // An input whose rows are all in the form comes back as is (ASCII: one scan of the bytes).
    const unchanged = switch (rows) {
        inline .offsets32, .offsets64 => |o| string.isAscii(o.data[@intCast(o.offsets[0])..@intCast(o.offsets[len])]) or
            (allInForm(o, bits, data.offset, len, mask) orelse return error.InvalidInput),
        inline else => |o| allInForm(o, bits, data.offset, len, mask) orelse return error.InvalidInput,
    };
    if (unchanged) return arr.retain();
    var out = try dense.VarBinaryOutput.init(ctx.allocator, data.data_type, len, try dense.combinedValidity(ctx.allocator, len, &.{data}), dense.totalBytes(rows, len));
    errdefer out.deinit();
    var cps: std.ArrayList(u21) = .empty;
    defer cps.deinit(ctx.allocator);
    switch (rows) {
        inline else => |r| for (0..len) |i| {
            if (bits) |b| if (!bitmap.bitIsSet(b, data.offset + i)) {
                try out.commit(0);
                continue;
            };
            const row = r.get(i);
            if ((quickCheck(row, mask) orelse return error.InvalidInput)) {
                try out.append(row);
                continue;
            }
            try normalizeRow(ctx.allocator, row, form, &cps);
            const slot = try out.reserve(cps.items.len * 4);
            var w: usize = 0;
            for (cps.items) |c| w += std.unicode.utf8Encode(c, slot[w..]) catch return error.InvalidInput;
            try out.commit(w);
        },
    }
    return out.finish();
}

/// Whether every valid row passes the quick check (null: a row is not UTF-8).
fn allInForm(rows: anytype, bits: ?[]const u8, offset: usize, len: usize, mask: u16) ?bool {
    for (0..len) |i| {
        if (bits) |b| if (!bitmap.bitIsSet(b, offset + i)) continue;
        const row = rows.get(i);
        // ASCII is in every form.
        var ascii = true;
        for (row) |c| ascii = ascii and c < 0x80;
        if (ascii) continue;
        // The first row out of form decides (the row pass checks UTF-8 of the rest).
        if (!(quickCheck(row, mask) orelse return null)) return false;
    }
    return true;
}

/// Bytes of row `i` of a string-layout array that passed `validateLayout` (for tests and fuzzing).
pub fn rowAt(arr: ArrayRef, i: usize) []const u8 {
    const rows = dense.StringOperand.init(.{ .array = arr }) catch return "";
    return switch (rows) {
        inline else => |r| r.get(i),
    };
}
