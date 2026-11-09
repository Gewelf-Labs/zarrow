//! Math, bit-wise and shift kernels. Expected values are pyarrow 26 results.
const std = @import("std");
const common = @import("test_common.zig");
const math = @import("math.zig");

const ArrayRef = common.ArrayRef;
const DataType = common.DataType;
const Datum = common.Datum;
const ExecContext = common.ExecContext;
const none = common.none;

const nan = std.math.nan(f64);
const inf = std.math.inf(f64);

// Generated with pyarrow 26 (pc.<op> on float64 and float32 arrays of `math_inputs`).
const math_inputs = [_]?f64{ 4.0, -1.0, 0.0, 0.5, -0.5, 1.0, 2.5, 100.0, -0.0, nan, inf, -inf, null };
const math_expected = .{
    .{ "sqrt", [_]?f64{ 2.0, nan, 0.0, 0.7071067811865476, nan, 1.0, 1.5811388300841898, 10.0, -0.0, nan, inf, nan, null }, [_]?f32{ 2.0, nan, 0.0, 0.7071067690849304, nan, 1.0, 1.5811388492584229, 10.0, -0.0, nan, inf, nan, null } },
    .{ "exp", [_]?f64{ 54.598150033144236, 0.36787944117144233, 1.0, 1.6487212707001282, 0.6065306597126334, 2.718281828459045, 12.182493960703473, 2.6881171418161356e+43, 1.0, nan, inf, 0.0, null }, [_]?f32{ 54.598148345947266, 0.3678794503211975, 1.0, 1.6487212181091309, 0.6065306663513184, 2.7182817459106445, 12.182494163513184, inf, 1.0, nan, inf, 0.0, null } },
    .{ "expm1", [_]?f64{ 53.598150033144236, -0.6321205588285577, 0.0, 0.6487212707001282, -0.39346934028736663, 1.718281828459045, 11.182493960703473, 2.6881171418161356e+43, -0.0, nan, inf, -1.0, null }, [_]?f32{ 53.598148345947266, -0.6321205496788025, 0.0, 0.6487212181091309, -0.39346933364868164, 1.7182817459106445, 11.182494163513184, inf, -0.0, nan, inf, -1.0, null } },
    .{ "ln", [_]?f64{ 1.3862943611198906, nan, -inf, -0.6931471805599453, nan, 0.0, 0.9162907318741551, 4.605170185988092, -inf, nan, inf, nan, null }, [_]?f32{ 1.3862943649291992, nan, -inf, -0.6931471824645996, nan, 0.0, 0.9162907600402832, 4.605170249938965, -inf, nan, inf, nan, null } },
    .{ "log10", [_]?f64{ 0.6020599913279624, nan, -inf, -0.3010299956639812, nan, 0.0, 0.3979400086720376, 2.0, -inf, nan, inf, nan, null }, [_]?f32{ 0.6020600199699402, nan, -inf, -0.3010300099849701, nan, 0.0, 0.3979400098323822, 2.0, -inf, nan, inf, nan, null } },
    .{ "log2", [_]?f64{ 2.0, nan, -inf, -1.0, nan, 0.0, 1.3219280948873624, 6.643856189774724, -inf, nan, inf, nan, null }, [_]?f32{ 2.0, nan, -inf, -1.0, nan, 0.0, 1.3219281435012817, 6.643856048583984, -inf, nan, inf, nan, null } },
    .{ "log1p", [_]?f64{ 1.6094379124341003, -inf, 0.0, 0.4054651081081644, -0.6931471805599453, 0.6931471805599453, 1.252762968495368, 4.61512051684126, -0.0, nan, inf, nan, null }, [_]?f32{ 1.6094379425048828, -inf, 0.0, 0.40546509623527527, -0.6931471824645996, 0.6931471824645996, 1.2527629137039185, 4.6151204109191895, -0.0, nan, inf, nan, null } },
    .{ "sin", [_]?f64{ -0.7568024953079282, -0.8414709848078965, 0.0, 0.479425538604203, -0.479425538604203, 0.8414709848078965, 0.5984721441039564, -0.5063656411097588, -0.0, nan, nan, nan, null }, [_]?f32{ -0.756802499294281, -0.8414709568023682, 0.0, 0.4794255495071411, -0.4794255495071411, 0.8414709568023682, 0.5984721183776855, -0.5063656568527222, -0.0, nan, nan, nan, null } },
    .{ "cos", [_]?f64{ -0.6536436208636119, 0.5403023058681398, 1.0, 0.8775825618903728, 0.8775825618903728, 0.5403023058681398, -0.8011436155469337, 0.8623188722876839, 1.0, nan, nan, nan, null }, [_]?f32{ -0.6536436080932617, 0.5403022766113281, 1.0, 0.8775825500488281, 0.8775825500488281, 0.5403022766113281, -0.8011435866355896, 0.8623188734054565, 1.0, nan, nan, nan, null } },
    .{ "tan", [_]?f64{ 1.1578212823495777, -1.5574077246549023, 0.0, 0.5463024898437905, -0.5463024898437905, 1.5574077246549023, -0.7470222972386602, -0.5872139151569291, -0.0, nan, nan, nan, null }, [_]?f32{ 1.1578212976455688, -1.5574077367782593, 0.0, 0.5463024973869324, -0.5463024973869324, 1.5574077367782593, -0.747022271156311, -0.587213933467865, -0.0, nan, nan, nan, null } },
    .{ "asin", [_]?f64{ nan, -1.5707963267948966, 0.0, 0.5235987755982989, -0.5235987755982989, 1.5707963267948966, nan, nan, -0.0, nan, nan, nan, null }, [_]?f32{ nan, -1.5707963705062866, 0.0, 0.5235987901687622, -0.5235987901687622, 1.5707963705062866, nan, nan, -0.0, nan, nan, nan, null } },
    .{ "acos", [_]?f64{ nan, 3.141592653589793, 1.5707963267948966, 1.0471975511965979, 2.0943951023931957, 0.0, nan, nan, 1.5707963267948966, nan, nan, nan, null }, [_]?f32{ nan, 3.1415927410125732, 1.5707963705062866, 1.0471975803375244, 2.094395160675049, 0.0, nan, nan, 1.5707963705062866, nan, nan, nan, null } },
    .{ "atan", [_]?f64{ 1.3258176636680326, -0.7853981633974483, 0.0, 0.4636476090008061, -0.4636476090008061, 0.7853981633974483, 1.1902899496825317, 1.5607966601082315, -0.0, nan, 1.5707963267948966, -1.5707963267948966, null }, [_]?f32{ 1.3258177042007446, -0.7853981852531433, 0.0, 0.46364760398864746, -0.46364760398864746, 0.7853981852531433, 1.1902899742126465, 1.5607966184616089, -0.0, nan, 1.5707963705062866, -1.5707963705062866, null } },
    .{ "sinh", [_]?f64{ 27.289917197127753, -1.1752011936438014, 0.0, 0.5210953054937474, -0.5210953054937474, 1.1752011936438014, 6.0502044810397875, 1.3440585709080678e+43, -0.0, nan, inf, -inf, null }, [_]?f32{ 27.2899169921875, -1.175201177597046, 0.0, 0.5210952758789062, -0.5210952758789062, 1.175201177597046, 6.050204277038574, inf, -0.0, nan, inf, -inf, null } },
    .{ "cosh", [_]?f64{ 27.308232836016487, 1.5430806348152437, 1.0, 1.1276259652063807, 1.1276259652063807, 1.5430806348152437, 6.132289479663687, 1.3440585709080678e+43, 1.0, nan, inf, inf, null }, [_]?f32{ 27.3082332611084, 1.5430806875228882, 1.0, 1.1276259422302246, 1.1276259422302246, 1.5430806875228882, 6.132289409637451, inf, 1.0, nan, inf, inf, null } },
    .{ "tanh", [_]?f64{ 0.999329299739067, -0.7615941559557649, 0.0, 0.46211715726000974, -0.46211715726000974, 0.7615941559557649, 0.9866142981514303, 1.0, -0.0, nan, 1.0, -1.0, null }, [_]?f32{ 0.9993293285369873, -0.7615941762924194, 0.0, 0.46211716532707214, -0.46211716532707214, 0.7615941762924194, 0.9866142868995667, 1.0, -0.0, nan, 1.0, -1.0, null } },
    .{ "asinh", [_]?f64{ 2.0947125472611012, -0.881373587019543, 0.0, 0.48121182505960347, -0.48121182505960347, 0.881373587019543, 1.6472311463710958, 5.298342365610589, -0.0, nan, inf, -inf, null }, [_]?f32{ 2.094712495803833, -0.8813735842704773, 0.0, 0.4812118411064148, -0.4812118411064148, 0.8813735842704773, 1.647231101989746, 5.298342227935791, -0.0, nan, inf, -inf, null } },
    .{ "acosh", [_]?f64{ 2.0634370688955608, nan, nan, nan, nan, 0.0, 1.5667992369724109, 5.298292365610485, nan, nan, inf, nan, null }, [_]?f32{ 2.063436985015869, nan, nan, nan, nan, 0.0, 1.566799283027649, 5.29829216003418, nan, nan, inf, nan, null } },
    .{ "atanh", [_]?f64{ nan, -inf, 0.0, 0.5493061443340549, -0.5493061443340549, inf, nan, nan, -0.0, nan, nan, nan, null }, [_]?f32{ nan, -inf, 0.0, 0.5493061542510986, -0.5493061542510986, inf, nan, nan, -0.0, nan, nan, nan, null } },
};

/// Equal, both NaN, or at most `ulps` units in the last place apart (the C library Arrow calls
/// and Zig's std.math may differ in the last bit of a transcendental function).
fn close(comptime T: type, e: T, a: T, ulps: u32) bool {
    if (std.math.isNan(e) or std.math.isNan(a)) return std.math.isNan(e) and std.math.isNan(a);
    if (e == a) return true;
    if (std.math.isInf(e) or std.math.isInf(a)) return false;
    if (std.math.signbit(e) != std.math.signbit(a)) return false;
    const U = @Int(.unsigned, @bitSizeOf(T));
    const x: U = @bitCast(@abs(e));
    const y: U = @bitCast(@abs(a));
    return (if (x > y) x - y else y - x) <= ulps;
}

fn expectClose(comptime T: type, name: []const u8, expected: []const ?T, actual: []const ?T) !void {
    try std.testing.expectEqual(expected.len, actual.len);
    for (expected, actual, 0..) |e, a, i| {
        const ok = if (e == null or a == null) e == null and a == null else close(T, e.?, a.?, 2);
        if (!ok) {
            std.debug.print("{s} row {d}: expected {any}, got {any}\n", .{ name, i, e, a });
            return error.TestExpectedEqual;
        }
    }
}

fn invokeRows(comptime T: type, ctx: *ExecContext, name: []const u8, args: []const Datum) ![]?T {
    var out = try ctx.invokeScalar(name, args, none);
    defer out.release();
    switch (out) {
        .array => |arr| try arr.data().validateFull(),
        .chunked => |c| for (c.chunks()) |part| try part.data().validateFull(),
        .scalar => {},
        .record_batch, .table => return error.InvalidInput,
    }
    try std.testing.expect(std.meta.activeTag(out.dataType()) == std.meta.activeTag(common.dense.dataTypeOf(T)));
    return common.collect(T, std.testing.allocator, out);
}

test "math unary functions on float64 and float32 match pyarrow 26 (NaN, infinities, zeros, nulls)" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    var f64s = try common.makeArray(f64, allocator, &math_inputs);
    defer f64s.release();
    var in32: [math_inputs.len]?f32 = undefined;
    for (&in32, math_inputs) |*d, s| d.* = if (s) |x| @floatCast(x) else null;
    var f32s = try common.makeArray(f32, allocator, &in32);
    defer f32s.release();
    inline for (math_expected) |case| {
        const e64 = case[1];
        const e32 = case[2];
        const got = try invokeRows(f64, &ctx, case[0], &.{.{ .array = f64s }});
        defer allocator.free(got);
        try expectClose(f64, case[0], &e64, got);
        const got32 = try invokeRows(f32, &ctx, case[0], &.{.{ .array = f32s }});
        defer allocator.free(got32);
        try expectClose(f32, case[0], &e32, got32);
    }
}

test "math binary functions, integer power, bit-wise and shifts match pyarrow 26" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);

    var a = try common.makeArray(f64, allocator, &.{ 8.0, 0.0, -1.0, 8.0, 2.0, -8.0, 0.0, null, 3.0 });
    defer a.release();
    var b = try common.makeArray(f64, allocator, &.{ 2.0, 2.0, 2.0, 1.0, 0.5, 1.0 / 3.0, -1.0, 1.0, null });
    defer b.release();
    const ab = [_]Datum{ .{ .array = a }, .{ .array = b } };
    const cases = .{
        .{ "logb", [_]?f64{ 3.0, -inf, nan, inf, -1.0, nan, nan, null, null } },
        .{ "atan2", [_]?f64{ 1.3258176636680326, 0.0, -0.4636476090008061, 1.446441332248135, 1.3258176636680326, -1.5291537476963082, 3.141592653589793, null, null } },
        .{ "power", [_]?f64{ 64.0, 0.0, 1.0, 8.0, 1.4142135623730951, nan, inf, null, null } },
        // A float power never fails, also when checked.
        .{ "power_checked", [_]?f64{ 64.0, 0.0, 1.0, 8.0, 1.4142135623730951, nan, inf, null, null } },
    };
    inline for (cases) |case| {
        const expected = case[1];
        const got = try invokeRows(f64, &ctx, case[0], &ab);
        defer allocator.free(got);
        try expectClose(f64, case[0], &expected, got);
    }

    // Integer power wraps; checked overflows; 0^0 = 1.
    var base = try common.makeArray(i32, allocator, &.{ 0, 1, -1, 2, -2, 3, 3, null });
    defer base.release();
    var exps = try common.makeArray(i32, allocator, &.{ 0, 5, 3, 0, 3, 19, 20, 4 });
    defer exps.release();
    const be = [_]Datum{ .{ .array = base }, .{ .array = exps } };
    try common.expectInvoke(i32, &ctx, .scalar, "power", &be, &.{ 1, 1, -1, 1, -8, 1162261467, -808182895, null });
    try std.testing.expectError(error.Overflow, ctx.invokeScalar("power_checked", &be, none));
    var u = try common.makeArray(u8, allocator, &.{ 3, 2 });
    defer u.release();
    try common.expectInvoke(u8, &ctx, .scalar, "power", &.{ .{ .array = u }, .{ .scalar = common.scalarOf(u8, 7) } }, &.{ 139, 128 });
    // (-2)^7 fits int8: no false overflow from the squaring.
    var m2 = try common.makeArray(i8, allocator, &.{ -2, null });
    defer m2.release();
    try common.expectInvoke(i8, &ctx, .scalar, "power_checked", &.{ .{ .array = m2 }, .{ .scalar = common.scalarOf(i8, 7) } }, &.{ -128, null });
    try std.testing.expectError(error.Overflow, ctx.invokeScalar("power_checked", &.{ .{ .array = m2 }, .{ .scalar = common.scalarOf(i8, 8) } }, none));
    try std.testing.expectError(error.Overflow, ctx.invokeScalar("power_checked", &.{ .{ .scalar = common.scalarOf(i8, 2) }, .{ .scalar = common.scalarOf(i8, 7) } }, none));

    // Bit-wise and shifts; out-of-range shift amounts keep the value.
    var x = try common.makeArray(i8, allocator, &.{ 1, 1, 1, -1, 5, -8, -8, null });
    defer x.release();
    var s = try common.makeArray(i8, allocator, &.{ 0, 7, 8, 1, -1, 1, 9, 1 });
    defer s.release();
    const xs = [_]Datum{ .{ .array = x }, .{ .array = s } };
    try common.expectInvoke(i8, &ctx, .scalar, "shift_left", &xs, &.{ 1, 1, 1, -2, 5, -16, -8, null });
    try common.expectInvoke(i8, &ctx, .scalar, "shift_right", &xs, &.{ 1, 1, 1, -1, 5, -4, -8, null });
    try common.expectInvoke(i8, &ctx, .scalar, "bit_wise_and", &xs, &.{ 0, 1, 0, 1, 5, 0, 8, null });
    try common.expectInvoke(i8, &ctx, .scalar, "bit_wise_or", &xs, &.{ 1, 7, 9, -1, -1, -7, -7, null });
    try common.expectInvoke(i8, &ctx, .scalar, "bit_wise_xor", &xs, &.{ 1, 6, 9, -2, -6, -7, -15, null });
    try common.expectInvoke(i8, &ctx, .scalar, "bit_wise_not", &.{.{ .array = x }}, &.{ -2, -2, -2, 0, -6, 7, 7, null });
    var big = try common.makeArray(i8, allocator, &.{64});
    defer big.release();
    try common.expectInvoke(i8, &ctx, .scalar, "shift_left_checked", &.{ .{ .array = big }, .{ .scalar = common.scalarOf(i8, 1) } }, &.{-128});
    var w = try common.makeArray(u32, allocator, &.{ 1, 1 });
    defer w.release();
    var ws = try common.makeArray(u32, allocator, &.{ 31, 32 });
    defer ws.release();
    try common.expectInvoke(u32, &ctx, .scalar, "shift_left", &.{ .{ .array = w }, .{ .array = ws } }, &.{ 2147483648, 1 });
    try common.expectInvoke(u32, &ctx, .scalar, "shift_right", &.{ .{ .array = w }, .{ .array = ws } }, &.{ 0, 1 });
}

test "math checked variants: domain errors only on valid rows, NaN passes (pyarrow 26)" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    const Case = struct { name: []const u8, bad: f64, ok: f64 };
    const cases = [_]Case{
        .{ .name = "sqrt_checked", .bad = -1, .ok = -0.0 },
        .{ .name = "ln_checked", .bad = 0, .ok = inf },
        .{ .name = "ln_checked", .bad = -1, .ok = 1 },
        .{ .name = "log10_checked", .bad = -0.0, .ok = 10 },
        .{ .name = "log2_checked", .bad = -3, .ok = 8 },
        .{ .name = "log1p_checked", .bad = -1, .ok = -0.5 },
        .{ .name = "log1p_checked", .bad = -2, .ok = 0 },
        .{ .name = "sin_checked", .bad = inf, .ok = 1 },
        .{ .name = "cos_checked", .bad = -inf, .ok = 1 },
        .{ .name = "tan_checked", .bad = -inf, .ok = 1 },
        .{ .name = "asin_checked", .bad = 1.5, .ok = 1 },
        .{ .name = "acos_checked", .bad = -1.5, .ok = -1 },
        .{ .name = "acosh_checked", .bad = 0.5, .ok = 1 },
        .{ .name = "atanh_checked", .bad = 1, .ok = 0.5 },
        .{ .name = "atanh_checked", .bad = -2, .ok = -0.5 },
    };
    for (cases) |c| {
        var bad = try common.makeArray(f64, allocator, &.{ c.ok, c.bad });
        defer bad.release();
        try std.testing.expectError(error.InvalidInput, ctx.invokeScalar(c.name, &.{.{ .array = bad }}, none));
        try std.testing.expectError(error.InvalidInput, ctx.invokeScalar(c.name, &.{.{ .scalar = common.scalarOf(f64, c.bad) }}, none));
        // The same value in a null row, NaN and the domain's edge pass.
        const vals = [_]f64{ c.ok, c.bad, nan };
        var masked = try common.borrowedArray(f64, allocator, &vals, &.{0b101});
        defer masked.release();
        var out = try ctx.invokeScalar(c.name, &.{.{ .array = masked }}, none);
        defer out.release();
        try std.testing.expectEqual(@as(usize, 1), out.array.data().null_count.?);
        var unchecked = try ctx.invokeScalar(c.name[0 .. c.name.len - "_checked".len], &.{.{ .array = bad }}, none);
        unchecked.release();
    }
    var lb = try common.makeArray(f64, allocator, &.{ 8, 8 });
    defer lb.release();
    var lbase = try common.makeArray(f64, allocator, &.{ 1, 2 });
    defer lbase.release();
    try common.expectInvoke(f64, &ctx, .scalar, "logb_checked", &.{ .{ .array = lb }, .{ .array = lbase } }, &.{ inf, 3.0 });
    inline for (.{ 0.0, -2.0 }) |bad_base| {
        try std.testing.expectError(error.InvalidInput, ctx.invokeScalar("logb_checked", &.{ .{ .array = lb }, .{ .scalar = common.scalarOf(f64, bad_base) } }, none));
        try std.testing.expectError(error.InvalidInput, ctx.invokeScalar("logb_checked", &.{ .{ .scalar = common.scalarOf(f64, bad_base) }, .{ .array = lbase } }, none));
    }
    // Integer power: a negative exponent fails in both variants wherever the exponent is valid.
    var e = try common.makeArray(i32, allocator, &.{ -1, null });
    defer e.release();
    var nb = try common.makeArray(i32, allocator, &.{ null, 2 });
    defer nb.release();
    try std.testing.expectError(error.InvalidInput, ctx.invokeScalar("power", &.{ .{ .array = nb }, .{ .array = e } }, none));
    try std.testing.expectError(error.InvalidInput, ctx.invokeScalar("power_checked", &.{ .{ .array = nb }, .{ .scalar = common.scalarOf(i32, -2) } }, none));
    // Checked shifts reject amounts outside [0, digits) in valid rows only.
    var one = try common.makeArray(i8, allocator, &.{ 1, 1 });
    defer one.release();
    inline for (.{ "shift_left_checked", "shift_right_checked" }) |name| {
        inline for (.{ 7, 8, -1 }) |amount| {
            try std.testing.expectError(error.InvalidInput, ctx.invokeScalar(name, &.{ .{ .array = one }, .{ .scalar = common.scalarOf(i8, amount) } }, none));
        }
        const amounts = [_]i8{ 7, 2 };
        var masked = try common.borrowedArray(i8, allocator, &amounts, &.{0b10});
        defer masked.release();
        try common.expectInvoke(i8, &ctx, .scalar, name, &.{ .{ .array = one }, .{ .array = masked } }, &.{ null, if (comptime std.mem.eql(u8, name, "shift_left_checked")) 4 else 0 });
    }
}

/// Wrapping integer power by repeated multiplication.
fn naivePower(comptime T: type, x: T, e: T) T {
    var r: T = 1;
    var k: T = 0;
    while (k < e) : (k += 1) r *%= x;
    return r;
}

test "math oracle matches a per-element loop on random sliced inputs with nulls, chunked and scalar operands" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    var prng = std.Random.DefaultPrng.init(0x3A77);
    const rng = prng.random();
    inline for (.{ f32, f64 }) |T| {
        var a = try common.randomArray(T, allocator, rng, 333, 5, 20, .{});
        defer a.release();
        const xs = try common.collect(T, allocator, .{ .array = a });
        defer allocator.free(xs);
        var chunked = try common.chunkedAt(allocator, a, &.{ 3, 100, 101 });
        defer chunked.release();
        inline for (comptime std.meta.tags(math.Unary)) |op| {
            const expected = try allocator.alloc(?T, xs.len);
            defer allocator.free(expected);
            for (expected, xs) |*d, x| d.* = if (x) |v| math.unaryValue(T, op, v) else null;
            const got = try invokeRows(T, &ctx, @tagName(op), &.{.{ .array = a }});
            defer allocator.free(got);
            try common.expectRows(T, expected, got);
            const got_chunked = try invokeRows(T, &ctx, @tagName(op), &.{.{ .chunked = chunked }});
            defer allocator.free(got_chunked);
            try common.expectRows(T, expected, got_chunked);
            const one = try invokeRows(T, &ctx, @tagName(op), &.{.{ .scalar = common.scalarOf(T, 0.25) }});
            defer allocator.free(one);
            try common.expectRows(T, &.{math.unaryValue(T, op, 0.25)}, one);
        }
    }
    inline for (.{ i8, i16, i32, i64, u8, u16, u32, u64 }) |T| {
        var a = try common.randomArray(T, allocator, rng, 257, 3, 15, .{});
        defer a.release();
        var b = try common.randomArray(T, allocator, rng, 257, 9, 15, .{});
        defer b.release();
        const xs = try common.collect(T, allocator, .{ .array = a });
        defer allocator.free(xs);
        const expected = try allocator.alloc(?T, xs.len);
        defer allocator.free(expected);
        const digits = @typeInfo(T).int.bits - @intFromBool(@typeInfo(T).int.signedness == .signed);
        inline for (.{ "bit_wise_and", "bit_wise_or", "bit_wise_xor", "shift_left", "shift_right", "power" }) |name| {
            // Small exponents keep the naive power loop short.
            var e = if (comptime std.mem.eql(u8, name, "power")) try common.randomArray(T, allocator, rng, 257, 1, 15, .{ .small = true }) else b.retain();
            defer e.release();
            const es = try common.collect(T, allocator, .{ .array = e });
            defer allocator.free(es);
            for (expected, xs, es) |*d, x, y| {
                d.* = if (x == null or y == null) null else blk: {
                    const p = x.?;
                    var q = y.?;
                    if (comptime std.mem.eql(u8, name, "power")) {
                        q = @intCast(@mod(@as(i128, q), 70));
                    }
                    const S = std.math.Log2Int(T);
                    const U = @Int(.unsigned, @bitSizeOf(T));
                    break :blk if (comptime std.mem.eql(u8, name, "bit_wise_and")) p & q else if (comptime std.mem.eql(u8, name, "bit_wise_or")) p | q else if (comptime std.mem.eql(u8, name, "bit_wise_xor")) p ^ q else if (comptime std.mem.eql(u8, name, "power")) naivePower(T, p, q) else if (q < 0 or q >= digits) p else if (comptime std.mem.eql(u8, name, "shift_left")) @as(T, @bitCast(@as(U, @bitCast(p)) << @as(S, @intCast(q)))) else p >> @as(S, @intCast(q));
                };
            }
            // power: the exponents reduced modulo 70 (non-negative) are the second argument.
            var exps = e.retain();
            if (comptime std.mem.eql(u8, name, "power")) {
                exps.release();
                const red = try allocator.alloc(?T, es.len);
                defer allocator.free(red);
                for (red, es) |*d, v| d.* = if (v) |q| @intCast(@mod(@as(i128, q), 70)) else null;
                exps = try common.makeArray(T, allocator, red);
            }
            defer exps.release();
            const got = try invokeRows(T, &ctx, name, &.{ .{ .array = a }, .{ .array = exps } });
            defer allocator.free(got);
            try common.expectRows(T, expected, got);
        }
    }
}

test "math implicit casts: integers and decimals to float64, power and bit-wise common types (pyarrow 26)" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    var i = try common.makeArray(i32, allocator, &.{ 4, -1, 0, null });
    defer i.release();
    try common.expectInvoke(f64, &ctx, .scalar, "sqrt", &.{.{ .array = i }}, &.{ 2.0, nan, 0.0, null });
    try common.expectInvoke(f64, &ctx, .scalar, "ceil", &.{.{ .array = i }}, &.{ 4.0, -1.0, 0.0, null });
    try common.expectInvoke(f64, &ctx, .scalar, "atan2", &.{ .{ .array = i }, .{ .array = i } }, &.{ 0.7853981633974483, -2.356194490192345, 0.0, null });
    try common.expectInvoke(f64, &ctx, .scalar, "logb", &.{ .{ .array = i }, .{ .array = i } }, &.{ 1.0, nan, nan, null });
    try std.testing.expectEqual(DataType{ .double = {} }, try registry.resolveResultType("ln", .scalar, &.{.{ .array = i }}, none));
    const dec = try common.makeTyped(i128, .{ .decimal128 = .{ .precision = 5, .scale = 2 } }, allocator, &.{ 150, -225 });
    var d = dec;
    defer d.release();
    try common.expectInvoke(f64, &ctx, .scalar, "sqrt", &.{.{ .array = d }}, &.{ 1.224744871391589, nan });
    try common.expectInvoke(f64, &ctx, .scalar, "power", &.{ .{ .array = d }, .{ .scalar = common.scalarOf(i64, 2) } }, &.{ 2.25, 5.0625 });
    var f32s = try common.makeArray(f32, allocator, &.{2.0});
    defer f32s.release();
    try common.expectInvoke(f32, &ctx, .scalar, "power", &.{ .{ .array = f32s }, .{ .scalar = common.scalarOf(i32, 3) } }, &.{8.0});
    try common.expectInvoke(f64, &ctx, .scalar, "atan2", &.{ .{ .array = f32s }, .{ .scalar = common.scalarOf(i32, 3) } }, &.{0.5880026035475675});
    var i8s = try common.makeArray(i8, allocator, &.{2});
    defer i8s.release();
    try common.expectInvoke(i64, &ctx, .scalar, "power", &.{ .{ .array = i8s }, .{ .scalar = common.scalarOf(u64, 3) } }, &.{8});
    var u8s = try common.makeArray(u8, allocator, &.{12});
    defer u8s.release();
    try common.expectInvoke(i16, &ctx, .scalar, "bit_wise_and", &.{ .{ .array = u8s }, .{ .scalar = common.scalarOf(i8, 10) } }, &.{8});
    try common.expectInvoke(i32, &ctx, .scalar, "shift_left", &.{ .{ .array = i8s }, .{ .scalar = common.scalarOf(i32, 2) } }, &.{8});
}

test "math scalars, empty and all-null inputs, errors: types, options, arity and malformed input" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);
    var empty = try common.makeArray(f64, allocator, &.{});
    defer empty.release();
    try common.expectInvoke(f64, &ctx, .scalar, "sqrt_checked", &.{.{ .array = empty }}, &.{});
    var nulls = try common.makeArray(f64, allocator, &.{ null, null });
    defer nulls.release();
    try common.expectInvoke(f64, &ctx, .scalar, "ln_checked", &.{.{ .array = nulls }}, &.{ null, null });
    try common.expectInvoke(f64, &ctx, .scalar, "sqrt", &.{.{ .scalar = common.scalarOf(f64, null) }}, &.{null});
    try common.expectInvoke(i32, &ctx, .scalar, "power", &.{ .{ .scalar = common.scalarOf(i32, 3) }, .{ .scalar = common.scalarOf(i32, 4) } }, &.{81});
    try common.expectInvoke(f64, &ctx, .scalar, "logb", &.{ .{ .array = nulls }, .{ .scalar = common.scalarOf(f64, null) } }, &.{ null, null });

    var b = try common.makeArray(bool, allocator, &.{true});
    defer b.release();
    try std.testing.expectError(error.NoMatchingKernel, ctx.invokeScalar("sqrt", &.{.{ .array = b }}, none));
    var f = try common.makeArray(f64, allocator, &.{1.0});
    defer f.release();
    try std.testing.expectError(error.NoMatchingKernel, ctx.invokeScalar("bit_wise_not", &.{.{ .array = f }}, none));
    try std.testing.expectError(error.NoMatchingKernel, ctx.invokeScalar("shift_left", &.{ .{ .array = f }, .{ .array = f } }, none));
    try std.testing.expectError(error.InvalidOptions, ctx.invokeScalar("sqrt", &.{.{ .array = f }}, .{ .arithmetic = .{} }));
    try std.testing.expectError(error.InvalidArity, ctx.invokeScalar("atan2", &.{.{ .array = f }}, none));
    var two = try common.makeArray(f64, allocator, &.{ 1.0, 2.0 });
    defer two.release();
    try std.testing.expectError(error.InvalidInput, ctx.invokeScalar("power", &.{ .{ .array = f }, .{ .array = two } }, none));
    const short = [_]f64{1};
    const bufs = [_]common.SharedBuffer{ .empty, .fromSlice(std.mem.sliceAsBytes(&short)) };
    var malformed = try ArrayRef.fromBorrowed(allocator, .{ .data_type = .{ .double = {} }, .length = 2, .null_count = 0, .buffers = &bufs });
    defer malformed.release();
    try std.testing.expectError(error.InvalidInput, ctx.invokeScalar("exp", &.{.{ .array = malformed }}, none));
    const bad = common.Scalar.init(.{ .double = {} }, .{ .i64 = 1 });
    try std.testing.expectError(error.InvalidInput, ctx.invokeScalar("cos", &.{.{ .scalar = bad }}, none));
}

test "hypot: no intermediate overflow, infinities over NaN, integer and mixed operands, nulls, sliced, chunked, scalars (pyarrow 26 results)" {
    const allocator = std.testing.allocator;
    var registry = try common.kernelRegistry(allocator);
    defer registry.deinit();
    var ctx = ExecContext.init(allocator, &registry);

    // pc.hypot([inf, nan, -inf, 1e308, None, nan], [nan, -inf, 1, 1e308, 1, 1])
    var x = try common.makeArray(f64, allocator, &.{ inf, nan, -inf, 1e308, null, nan, 3 });
    defer x.release();
    var y = try common.makeArray(f64, allocator, &.{ nan, -inf, 1, 1e308, 1, 1, -4 });
    defer y.release();
    try common.expectInvoke(f64, &ctx, .scalar, "hypot", &.{ .{ .array = x }, .{ .array = y } }, &.{ inf, inf, inf, 1.4142135623730951e308, null, nan, 5 });
    // pc.hypot([3, 4], [4, 3]) on int64 -> [5.0, 5.0] double; int8 with float32 is double.
    var a = try common.makeArray(i64, allocator, &.{ 3, 4 });
    defer a.release();
    var b = try common.makeArray(i64, allocator, &.{ 4, 3 });
    defer b.release();
    try common.expectInvoke(f64, &ctx, .scalar, "hypot", &.{ .{ .array = a }, .{ .array = b } }, &.{ 5, 5 });
    var a8 = try common.makeArray(i8, allocator, &.{3});
    defer a8.release();
    var f4 = try common.makeArray(f32, allocator, &.{ 4, 3 });
    defer f4.release();
    try common.expectInvoke(f64, &ctx, .scalar, "hypot", &.{ .{ .array = a8 }, .{ .scalar = common.scalarOf(f32, 4) } }, &.{5});
    try common.expectInvoke(f32, &ctx, .scalar, "hypot", &.{ .{ .array = f4 }, .{ .scalar = common.scalarOf(f32, 3) } }, &.{ 5, 4.2426405 });
    try common.expectInvoke(f64, &ctx, .scalar, "hypot", &.{ .{ .scalar = common.scalarOf(f64, 3) }, .{ .scalar = common.scalarOf(i64, 4) } }, &.{5});
    // Sliced and chunked operands.
    var xs = try x.slice(5, 2);
    defer xs.release();
    var ys = try y.slice(5, 2);
    defer ys.release();
    try common.expectInvoke(f64, &ctx, .scalar, "hypot", &.{ .{ .array = xs }, .{ .array = ys } }, &.{ nan, 5 });
    var cx = try common.chunkedAt(allocator, x, &.{ 2, 5 });
    defer cx.release();
    var out = try ctx.invokeScalar("hypot", &.{ .{ .chunked = cx }, .{ .array = y } }, none);
    defer out.release();
    try std.testing.expect(out == .chunked);
    const got = try common.collect(f64, allocator, out);
    defer allocator.free(got);
    try common.expectRows(f64, &.{ inf, inf, inf, 1.4142135623730951e308, null, nan, 5 }, got);
    // float16 is cast to float32 as for atan2 (pyarrow 26 has no float16 kernel); no string kernel.
    var h = try common.makeArray(f16, allocator, &.{3});
    defer h.release();
    try common.expectInvoke(f32, &ctx, .scalar, "hypot", &.{ .{ .array = h }, .{ .array = h } }, &.{4.2426405});
    var s = try common.makeStrings(.string, allocator, &.{"3"});
    defer s.release();
    try std.testing.expectError(error.NoMatchingKernel, ctx.invokeScalar("hypot", &.{ .{ .array = s }, .{ .array = s } }, none));
}
