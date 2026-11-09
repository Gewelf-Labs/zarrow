//! run_end_encode and run_end_decode. Expected values are pyarrow 26 results.
const std = @import("std");
const common = @import("test_common.zig");

const ArrayRef = common.ArrayRef;
const DataType = common.DataType;
const Datum = common.Datum;
const ExecContext = common.ExecContext;
const Options = common.Options;
const none = common.none;

fn opts(t: @FieldType(common.core.RunEndEncodeOptions, "run_end_type")) Options {
    return .{ .run_end_encode = .{ .run_end_type = t } };
}

/// Check one run_end_encoded chunk: its run ends and (through `T`) its values.
fn expectRuns(comptime T: type, comptime E: type, arr: ArrayRef, ends: []const E, values: []const ?T) !void {
    const allocator = std.testing.allocator;
    const d = arr.data();
    try d.validateFull();
    try std.testing.expectEqual(@as(u8, @bitSizeOf(E)), d.data_type.run_end_encoded.run_end_type.bit_width);
    const got = try (common.array_mod.PrimitiveArray(E){ .data = d.children[0].data() }).values();
    try std.testing.expectEqualSlices(E, ends, got);
    const rows = try common.collect(T, allocator, .{ .array = d.children[1] });
    defer allocator.free(rows);
    try common.expectRows(T, values, rows);
}

/// Encode, check the runs, decode and check the rows again.
fn roundTrip(comptime T: type, ctx: *ExecContext, input: ArrayRef, ends: []const i32, values: []const ?T, rows: []const ?T) !void {
    var enc = try ctx.invoke("run_end_encode", .vector, &.{.{ .array = input }}, none);
    defer enc.release();
    try expectRuns(T, i32, enc.array, ends, values);
    try common.expectInvoke(T, ctx, .vector, "run_end_decode", &.{enc}, rows);
}

/// Encode a string / binary array, check the run ends, decode and compare with the input.
fn bytesRoundTrip(ctx: *ExecContext, input: ArrayRef, ends: []const i32) !void {
    const allocator = std.testing.allocator;
    var enc = try ctx.invoke("run_end_encode", .vector, &.{.{ .array = input }}, none);
    defer enc.release();
    try enc.array.data().validateFull();
    try std.testing.expectEqualSlices(i32, ends, try (common.array_mod.PrimitiveArray(i32){ .data = enc.array.data().children[0].data() }).values());
    var dec = try ctx.invoke("run_end_decode", .vector, &.{enc}, none);
    defer dec.release();
    const got = try common.collectStrings(allocator, dec);
    defer allocator.free(got);
    const want = try common.collectStrings(allocator, .{ .array = input });
    defer allocator.free(want);
    try common.expectStrings(want, got);
}

test "run_end_encode and run_end_decode hand-written values, nulls, sliced, every run end type (pyarrow 26 results)" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);

    var a = try common.makeArray(i64, allocator, &.{ 1, 1, 2, null, null });
    defer a.release();
    try roundTrip(i64, &ctx, a, &.{ 2, 3, 5 }, &.{ 1, 2, null }, &.{ 1, 1, 2, null, null });
    // Sliced input: [1, 2, null].
    var s = try a.slice(1, 3);
    defer s.release();
    try roundTrip(i64, &ctx, s, &.{ 1, 2, 3 }, &.{ 1, 2, null }, &.{ 1, 2, null });
    inline for (.{ .{ .int16, i16 }, .{ .int64, i64 } }) |pair| {
        var enc = try ctx.invoke("run_end_encode", .vector, &.{.{ .array = a }}, opts(pair[0]));
        defer enc.release();
        try expectRuns(i64, pair[1], enc.array, &.{ 2, 3, 5 }, &.{ 1, 2, null });
        var arena = std.heap.ArenaAllocator.init(allocator);
        defer arena.deinit();
        const rt = try registry.resolveResultTypeAlloc(arena.allocator(), "run_end_encode", .vector, &.{.{ .array = a }}, opts(pair[0]));
        try std.testing.expect(rt.eql(enc.array.data().data_type));
    }

    // Floats compare by bits: NaN with the same bits is one run, 0.0 and -0.0 are two.
    var f = try common.makeArray(f64, allocator, &.{ std.math.nan(f64), std.math.nan(f64), 0.0, -0.0, null });
    defer f.release();
    var fe = try ctx.invoke("run_end_encode", .vector, &.{.{ .array = f }}, none);
    defer fe.release();
    try std.testing.expectEqualSlices(i32, &.{ 2, 3, 4, 5 }, try (common.array_mod.PrimitiveArray(i32){ .data = fe.array.data().children[0].data() }).values());

    var b = try common.makeArray(bool, allocator, &.{ true, true, false, null });
    defer b.release();
    try roundTrip(bool, &ctx, b, &.{ 2, 3, 4 }, &.{ true, false, null }, &.{ true, true, false, null });
    var h = try common.makeArray(f16, allocator, &.{ 1.5, 1.5 });
    defer h.release();
    try roundTrip(f16, &ctx, h, &.{2}, &.{1.5}, &.{ 1.5, 1.5 });
    var ts = try common.makeTyped(i64, .{ .timestamp = .{ .unit = .second, .timezone = null } }, allocator, &.{ 5, 5, 5, 6 });
    defer ts.release();
    try roundTrip(i64, &ctx, ts, &.{ 3, 4 }, &.{ 5, 6 }, &.{ 5, 5, 5, 6 });
    var dec = try common.makeTyped(i128, .{ .decimal128 = .{ .precision = 5, .scale = 1 } }, allocator, &.{ 15, 15, null, 20 });
    defer dec.release();
    try roundTrip(i128, &ctx, dec, &.{ 2, 3, 4 }, &.{ 15, null, 20 }, &.{ 15, 15, null, 20 });

    // Strings of every layout, binaries.
    inline for (.{ .string, .large_string, .string_view, .large_binary }) |tag| {
        var st = try common.makeStrings(tag, allocator, &.{ "ab", "ab", null, "a long string past twelve bytes", "a long string past twelve bytes", "c" });
        defer st.release();
        var tail = try st.slice(1, 5);
        defer tail.release();
        try bytesRoundTrip(&ctx, st, &.{ 2, 3, 5, 6 });
        try bytesRoundTrip(&ctx, tail, &.{ 1, 2, 4, 5 });
    }

    // fixed_size_binary.
    var fsb_b = try common.array_mod.FixedSizeBinaryBuilder.init(allocator, 2, 4);
    defer fsb_b.deinit();
    try fsb_b.append("ab");
    try fsb_b.append("ab");
    try fsb_b.appendNull();
    try fsb_b.append("cd");
    var fsb = try fsb_b.finish();
    defer fsb.release();
    var fsb_e = try ctx.invoke("run_end_encode", .vector, &.{.{ .array = fsb }}, none);
    defer fsb_e.release();
    try fsb_e.array.data().validateFull();
    try std.testing.expectEqualSlices(i32, &.{ 2, 3, 4 }, try (common.array_mod.PrimitiveArray(i32){ .data = fsb_e.array.data().children[0].data() }).values());

    // The null type: one run.
    var nulls = try common.dense.nullArray(allocator, .{ .null = {} }, 2);
    defer nulls.release();
    var ne = try ctx.invoke("run_end_encode", .vector, &.{.{ .array = nulls }}, none);
    defer ne.release();
    try std.testing.expectEqual(@as(usize, 1), ne.array.data().children[1].data().length);
    try std.testing.expectEqual(@as(usize, 2), ne.array.data().length);
}

test "run_end_decode of a sliced run_end_encoded array and a sliced values child (pyarrow 26 results)" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);

    var a = try common.makeArray(i32, allocator, &.{ 1, 1, 2, null, null, 7, 7, 7 });
    defer a.release();
    var enc = try ctx.invoke("run_end_encode", .vector, &.{.{ .array = a }}, none);
    defer enc.release();
    // Slices that start and end inside runs.
    var mid = try enc.array.slice(1, 3);
    defer mid.release();
    try common.expectInvoke(i32, &ctx, .vector, "run_end_decode", &.{.{ .array = mid }}, &.{ 1, 2, null });
    var tail = try enc.array.slice(4, 4);
    defer tail.release();
    try common.expectInvoke(i32, &ctx, .vector, "run_end_decode", &.{.{ .array = tail }}, &.{ null, 7, 7, 7 });
    var empty = try enc.array.slice(8, 0);
    defer empty.release();
    try common.expectInvoke(i32, &ctx, .vector, "run_end_decode", &.{.{ .array = empty }}, &.{});

    // A hand-built run_end_encoded array over a sliced values child.
    var vals = try common.makeArray(i32, allocator, &.{ 99, 4, null, 5 });
    defer vals.release();
    var vals_tail = try vals.slice(1, 3);
    defer vals_tail.release();
    const vt = vals_tail.data().data_type;
    var rb = try common.array_mod.RunEndEncodedBuilder.init(allocator, .{ .bit_width = 16, .signed = true }, &vt, 3);
    defer rb.deinit();
    for ([_]i64{ 2, 3, 6 }) |e| try rb.appendRunEnd(e);
    var ree = try rb.finish(vals_tail);
    defer ree.release();
    try common.expectInvoke(i32, &ctx, .vector, "run_end_decode", &.{.{ .array = ree }}, &.{ 4, 4, null, 5, 5, 5 });
    try std.testing.expect((try registry.resolveResultType("run_end_decode", .vector, &.{.{ .array = ree }}, none)) == .int32);
}

test "run_end_encode and run_end_decode on chunked inputs: per chunk, empty chunks dropped (pyarrow 26 results)" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);

    var a = try common.makeArray(i64, allocator, &.{ 1, 1, 1, 2 });
    defer a.release();
    var parts = [_]ArrayRef{ try a.slice(0, 2), try a.slice(2, 2), try a.slice(4, 0) };
    defer for (&parts) |*p| p.release();
    var c = try common.ChunkedArray.init(allocator, a.data().data_type, &parts);
    defer c.release();
    var enc = try ctx.invoke("run_end_encode", .vector, &.{.{ .chunked = c }}, none);
    defer enc.release();
    try std.testing.expectEqual(@as(usize, 2), enc.chunked.numChunks());
    try expectRuns(i64, i32, enc.chunked.chunks()[0], &.{2}, &.{1});
    try expectRuns(i64, i32, enc.chunked.chunks()[1], &.{ 1, 2 }, &.{ 1, 2 });
    try common.expectInvoke(i64, &ctx, .vector, "run_end_decode", &.{enc}, &.{ 1, 1, 1, 2 });
    var dec = try ctx.invoke("run_end_decode", .vector, &.{enc}, none);
    defer dec.release();
    try std.testing.expectEqual(@as(usize, 2), dec.chunked.numChunks());

    // Only empty chunks: one empty chunk; an empty array: an empty run_end_encoded array.
    var only_empty = try common.ChunkedArray.init(allocator, a.data().data_type, parts[2..3]);
    defer only_empty.release();
    var ee = try ctx.invoke("run_end_encode", .vector, &.{.{ .chunked = only_empty }}, none);
    defer ee.release();
    try std.testing.expectEqual(@as(usize, 0), ee.chunked.len());
    try std.testing.expect(ee.chunked.dataType() == .run_end_encoded);
    var e1 = try ctx.invoke("run_end_encode", .vector, &.{.{ .array = parts[2] }}, none);
    defer e1.release();
    try expectRuns(i64, i32, e1.array, &.{}, &.{});
    try common.expectInvoke(i64, &ctx, .vector, "run_end_decode", &.{e1}, &.{});
}

test "run_end_encode and run_end_decode errors: run end overflow, types, scalars, options and malformed input" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);

    // More rows than int16 run ends hold (Arrow: "Cannot run-end encode Arrays with more
    // elements than the run end type can hold: 32767"), also with a single run.
    const big = try allocator.alloc(?i32, 32768);
    defer allocator.free(big);
    @memset(big, 0);
    var long = try common.makeArray(i32, allocator, big);
    defer long.release();
    try std.testing.expectError(error.InvalidInput, ctx.invoke("run_end_encode", .vector, &.{.{ .array = long }}, opts(.int16)));
    var fits = try long.slice(0, 32767);
    defer fits.release();
    var ok = try ctx.invoke("run_end_encode", .vector, &.{.{ .array = fits }}, opts(.int16));
    ok.release();

    // Types Arrow has no kernel for here: dictionaries, run_end_encoded, nested types; scalars.
    var a = try common.makeArray(i32, allocator, &.{ 1, 1 });
    defer a.release();
    var enc = try ctx.invoke("run_end_encode", .vector, &.{.{ .array = a }}, none);
    defer enc.release();
    try std.testing.expectError(error.NoMatchingKernel, ctx.invoke("run_end_encode", .vector, &.{enc}, none));
    try std.testing.expectError(error.NoMatchingKernel, ctx.invoke("run_end_decode", .vector, &.{.{ .array = a }}, none));
    try std.testing.expectError(error.NoMatchingKernel, ctx.invoke("run_end_encode", .vector, &.{.{ .scalar = common.scalarOf(i32, 1) }}, none));
    var strs = try common.makeStrings(.string, allocator, &.{ "x", "y" });
    defer strs.release();
    var dict = try common.makeDictionary(allocator, strs, .{ .bit_width = 32, .signed = true }, &.{ 0, 0, 1 });
    defer dict.release();
    try std.testing.expectError(error.NoMatchingKernel, ctx.invoke("run_end_encode", .vector, &.{.{ .array = dict }}, none));
    var st = try common.dense.structArray(allocator, &.{"a"}, &.{a.retain()}, 2, null);
    defer st.release();
    try std.testing.expectError(error.NoMatchingKernel, ctx.invoke("run_end_encode", .vector, &.{.{ .array = st }}, none));
    try std.testing.expectError(error.InvalidOptions, ctx.invoke("run_end_encode", .vector, &.{.{ .array = a }}, .{ .take = .{} }));
    try std.testing.expectError(error.InvalidArity, ctx.invoke("run_end_decode", .vector, &.{ enc, enc }, none));

    // A run_end_encoded array whose run ends stop short of its length fails validateLayout.
    const vt = a.data().data_type;
    var rb = try common.array_mod.RunEndEncodedBuilder.init(allocator, .{ .bit_width = 32, .signed = true }, &vt, 2);
    defer rb.deinit();
    try rb.appendRunEnd(1);
    try rb.appendRunEnd(2);
    var ree = try rb.finish(a);
    defer ree.release();
    const d = ree.data();
    var bad_data = d.*;
    bad_data.length = 5;
    var bad = try ArrayRef.fromBorrowed(allocator, bad_data);
    defer bad.release();
    try std.testing.expectError(error.InvalidInput, ctx.invoke("run_end_decode", .vector, &.{.{ .array = bad }}, none));
}
