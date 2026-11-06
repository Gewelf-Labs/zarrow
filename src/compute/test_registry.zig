const std = @import("std");
const datatype = @import("../datatype.zig");
const array_mod = @import("../array/array.zig");
const common = @import("test_common.zig");

const DataType = common.DataType;
const ArrayRef = common.ArrayRef;
const ChunkedArray = common.ChunkedArray;
const Scalar = common.Scalar;
const Datum = common.Datum;
const FunctionRegistry = common.FunctionRegistry;
const ExecContext = common.ExecContext;
const KernelSignature = common.KernelSignature;
const KernelError = common.KernelError;
const Options = common.Options;
const OptionsTag = common.OptionsTag;
const SortOptions = common.SortOptions;
const SortOrder = common.SortOrder;
const SortNullPlacement = common.SortNullPlacement;
const FilterOptions = common.FilterOptions;
const UnaryExecChunkIterator = common.UnaryExecChunkIterator;
const BinaryExecChunkIterator = common.BinaryExecChunkIterator;
const NaryExecChunkIterator = common.NaryExecChunkIterator;

const intCastOrInvalidCast = common.intCastOrInvalidCast;
const arithmeticDivI64 = common.arithmeticDivI64;
const hasArity = common.hasArity;
const unaryArray = common.unaryArray;
const unaryChunked = common.unaryChunked;
const unaryScalar = common.unaryScalar;
const sameDataTypes = common.sameDataTypes;
const unaryNumeric = common.unaryNumeric;
const binarySameNumeric = common.binarySameNumeric;
const allNumeric = common.allNumeric;
const unaryNullPropagates = common.unaryNullPropagates;
const binaryNullPropagates = common.binaryNullPropagates;
const naryNullPropagates = common.naryNullPropagates;
const inferBinaryExecLen = common.inferBinaryExecLen;
const inferNaryExecLen = common.inferNaryExecLen;
const datumListValueAt = common.datumListValueAt;
const datumStructField = common.datumStructField;
const datumBuildEmptyLike = common.datumBuildEmptyLike;
const datumBuildEmptyLikeWithAllocator = common.datumBuildEmptyLikeWithAllocator;
const datumBuildNullLike = common.datumBuildNullLike;
const datumBuildNullLikeWithAllocator = common.datumBuildNullLikeWithAllocator;
const datumSliceEmpty = common.datumSliceEmpty;
const datumSelect = common.datumSelect;
const datumSelectNullable = common.datumSelectNullable;
const datumFilter = common.datumFilter;
const chunkedResolveLogicalIndices = common.chunkedResolveLogicalIndices;
const datumTake = common.datumTake;
const datumFilterSelectionIndices = common.datumFilterSelectionIndices;
const datumFilterChunkAware = common.datumFilterChunkAware;

const isInt32Datum = common.isInt32Datum;
const isTwoInt32 = common.isTwoInt32;
const isInt64Scalar = common.isInt64Scalar;
const isTwoInt64Scalars = common.isTwoInt64Scalars;
const allInt32Datums = common.allInt32Datums;
const passthroughInt32Kernel = common.passthroughInt32Kernel;
const exactArityMarkerKernel = common.exactArityMarkerKernel;
const rangeArityMarkerKernel = common.rangeArityMarkerKernel;
const atLeastArityMarkerKernel = common.atLeastArityMarkerKernel;
const countLenAggregateKernel = common.countLenAggregateKernel;
const onlyCastOptions = common.onlyCastOptions;
const onlyArithmeticOptions = common.onlyArithmeticOptions;
const onlySortOptions = common.onlySortOptions;
const firstArgResultType = common.firstArgResultType;
const castResultType = common.castResultType;
const castI64ToI32ResultType = common.castI64ToI32ResultType;
const castI64ToI32Kernel = common.castI64ToI32Kernel;
const divI64ScalarResultType = common.divI64ScalarResultType;
const divI64ScalarKernel = common.divI64ScalarKernel;
const countAggregateResultType = common.countAggregateResultType;
const countLifecycleInit = common.countLifecycleInit;
const countLifecycleUpdate = common.countLifecycleUpdate;
const countLifecycleMerge = common.countLifecycleMerge;
const countLifecycleFinalize = common.countLifecycleFinalize;
const countLifecycleDeinit = common.countLifecycleDeinit;
const makeInt32Array = common.makeInt32Array;
const makeBoolArray = common.makeBoolArray;
const collectInt32ValuesFromDatum = common.collectInt32ValuesFromDatum;

test "compute registry registers and invokes scalar kernel" {
    const allocator = std.testing.allocator;
    var registry = FunctionRegistry.init(allocator);
    defer registry.deinit();

    try registry.registerScalarKernel("identity", .{
        .signature = .{
            .arity = 1,
            .type_check = isInt32Datum,
        },
        .exec = passthroughInt32Kernel,
    });

    const int32_builder = @import("../array/array.zig").Int32Builder;
    const int32_array = @import("../array/array.zig").Int32Array;

    var builder = try int32_builder.init(allocator, 3);
    defer builder.deinit();
    try builder.append(7);
    try builder.append(8);
    try builder.append(9);

    var arr = try builder.finish();
    defer arr.release();
    const args = [_]Datum{
        .{ .array = arr.retain() },
    };
    defer {
        var d = args[0];
        d.release();
    }

    var ctx = ExecContext.init(allocator, &registry);
    var out = try ctx.invokeScalar("identity", args[0..], Options.noneValue());
    defer out.release();

    try std.testing.expect(out == .array);
    const view = int32_array{ .data = out.array.data() };
    try std.testing.expectEqual(@as(usize, 3), view.len());
    try std.testing.expectEqual(@as(i32, 7), try view.value(0));
    try std.testing.expectEqual(@as(i32, 9), try view.value(2));
}

test "compute registry reports function and arity errors" {
    const allocator = std.testing.allocator;
    var registry = FunctionRegistry.init(allocator);
    defer registry.deinit();

    try registry.registerScalarKernel("identity", .{
        .signature = .{
            .arity = 1,
            .type_check = isInt32Datum,
        },
        .exec = passthroughInt32Kernel,
    });

    var ctx = ExecContext.init(allocator, &registry);
    try std.testing.expectError(
        error.FunctionNotFound,
        ctx.invoke("missing", .scalar, &[_]Datum{}, Options.noneValue()),
    );
    try std.testing.expectError(
        error.InvalidArity,
        ctx.invoke("identity", .scalar, &[_]Datum{}, Options.noneValue()),
    );
}

test "compute registry keeps separate indices per function kind" {
    const allocator = std.testing.allocator;
    var registry = FunctionRegistry.init(allocator);
    defer registry.deinit();

    try registry.registerScalarKernel("same_name", .{
        .signature = .{ .arity = 1, .type_check = isInt32Datum },
        .exec = passthroughInt32Kernel,
    });
    try registry.registerVectorKernel("same_name", .{
        .signature = .{ .arity = 1, .type_check = isInt32Datum },
        .exec = passthroughInt32Kernel,
    });

    try std.testing.expect(registry.findFunction("same_name", .scalar) != null);
    try std.testing.expect(registry.findFunction("same_name", .vector) != null);
    try std.testing.expect(registry.findFunction("same_name", .aggregate) == null);
}

test "compute registry helper APIs and function metadata" {
    const allocator = std.testing.allocator;
    var registry = FunctionRegistry.init(allocator);
    defer registry.deinit();

    try registry.registerScalarKernel("identity", .{
        .signature = KernelSignature.unary(isInt32Datum),
        .exec = passthroughInt32Kernel,
    });
    try registry.registerVectorKernel("identity", .{
        .signature = KernelSignature.unary(isInt32Datum),
        .exec = passthroughInt32Kernel,
    });

    try std.testing.expect(registry.containsFunction("identity", .scalar));
    try std.testing.expect(registry.containsFunction("identity", .vector));
    try std.testing.expect(!registry.containsFunction("identity", .aggregate));
    try std.testing.expectEqual(@as(usize, 2), registry.functionCount());
    try std.testing.expectEqual(@as(usize, 1), registry.kernelCount("identity", .scalar));
    try std.testing.expectEqual(@as(usize, 1), registry.kernelCount("identity", .vector));
    try std.testing.expectEqual(@as(usize, 0), registry.kernelCount("identity", .aggregate));

    const f0 = registry.functionAt(0).?;
    try std.testing.expect(f0.kernelCount() > 0);
    try std.testing.expect(f0.kernelsSlice().len > 0);
}

test "compute invoke helpers cover vector and aggregate kernels" {
    const allocator = std.testing.allocator;
    const int32_builder = @import("../array/array.zig").Int32Builder;
    const int32_array = @import("../array/array.zig").Int32Array;
    var registry = FunctionRegistry.init(allocator);
    defer registry.deinit();

    try registry.registerVectorKernel("vec_identity", .{
        .signature = KernelSignature.unary(isInt32Datum),
        .exec = passthroughInt32Kernel,
    });
    try registry.registerAggregateKernel("count_len", .{
        .signature = KernelSignature.unary(unaryArray),
        .exec = countLenAggregateKernel,
    });

    var builder = try int32_builder.init(allocator, 4);
    defer builder.deinit();
    try builder.append(10);
    try builder.append(20);
    try builder.append(30);
    try builder.append(40);
    var arr = try builder.finish();
    defer arr.release();

    const args = [_]Datum{Datum.fromArray(arr.retain())};
    defer {
        var d = args[0];
        d.release();
    }

    var ctx = ExecContext.init(allocator, &registry);
    var vec_out = try ctx.invokeVector("vec_identity", args[0..], Options.noneValue());
    defer vec_out.release();
    try std.testing.expect(vec_out.isArray());
    const vec_view = int32_array{ .data = vec_out.array.data() };
    try std.testing.expectEqual(@as(usize, 4), vec_view.len());
    try std.testing.expectEqual(@as(i32, 10), try vec_view.value(0));
    try std.testing.expectEqual(@as(i32, 40), try vec_view.value(3));

    var agg_out = try ctx.invokeAggregate("count_len", args[0..], Options.noneValue());
    defer agg_out.release();
    try std.testing.expect(agg_out.isScalar());
    try std.testing.expectEqual(@as(i64, 4), agg_out.scalar.value.i64);
}

test "compute registry functionAt out of range returns null" {
    const allocator = std.testing.allocator;
    var registry = FunctionRegistry.init(allocator);
    defer registry.deinit();

    try registry.registerScalarKernel("identity", .{
        .signature = KernelSignature.unary(isInt32Datum),
        .exec = passthroughInt32Kernel,
    });
    try std.testing.expect(registry.functionAt(0) != null);
    try std.testing.expect(registry.functionAt(1) == null);
    try std.testing.expect(registry.functionAt(99) == null);
}

test "compute resolveKernel prefers exact then range then at_least arity" {
    const allocator = std.testing.allocator;
    var registry = FunctionRegistry.init(allocator);
    defer registry.deinit();

    try registry.registerScalarKernel("pick_by_arity", .{
        .signature = .{
            .arity = 2,
            .variadic = true,
            .type_check = allInt32Datums,
        },
        .exec = atLeastArityMarkerKernel,
    });
    try registry.registerScalarKernel("pick_by_arity", .{
        .signature = .{
            .arity = 2,
            .variadic = true,
            .max_arity = 4,
            .type_check = allInt32Datums,
        },
        .exec = rangeArityMarkerKernel,
    });
    try registry.registerScalarKernel("pick_by_arity", .{
        .signature = .{
            .arity = 3,
            .type_check = allInt32Datums,
        },
        .exec = exactArityMarkerKernel,
    });

    var ctx = ExecContext.init(allocator, &registry);
    const scalar_i32 = Datum.fromScalar(.{
        .data_type = .{ .int32 = {} },
        .value = .{ .i32 = 1 },
    });

    const args3 = [_]Datum{ scalar_i32, scalar_i32, scalar_i32 };
    var out3 = try ctx.invokeScalar("pick_by_arity", args3[0..], Options.noneValue());
    defer out3.release();
    try std.testing.expect(out3.isScalar());
    try std.testing.expectEqual(@as(i32, 303), out3.scalar.value.i32);

    const args4 = [_]Datum{ scalar_i32, scalar_i32, scalar_i32, scalar_i32 };
    var out4 = try ctx.invokeScalar("pick_by_arity", args4[0..], Options.noneValue());
    defer out4.release();
    try std.testing.expect(out4.isScalar());
    try std.testing.expectEqual(@as(i32, 202), out4.scalar.value.i32);

    const args5 = [_]Datum{ scalar_i32, scalar_i32, scalar_i32, scalar_i32, scalar_i32 };
    var out5 = try ctx.invokeScalar("pick_by_arity", args5[0..], Options.noneValue());
    defer out5.release();
    try std.testing.expect(out5.isScalar());
    try std.testing.expectEqual(@as(i32, 101), out5.scalar.value.i32);
}

test "compute explainResolveKernelFailure prefers minimum arity diagnostic in mixed models" {
    const allocator = std.testing.allocator;
    var registry = FunctionRegistry.init(allocator);
    defer registry.deinit();

    try registry.registerScalarKernel("mixed_arity", .{
        .signature = .{
            .arity = 3,
            .variadic = true,
            .type_check = allInt32Datums,
        },
        .exec = atLeastArityMarkerKernel,
    });
    try registry.registerScalarKernel("mixed_arity", .{
        .signature = .{
            .arity = 5,
            .variadic = true,
            .max_arity = 7,
            .type_check = allInt32Datums,
        },
        .exec = rangeArityMarkerKernel,
    });

    const two_i32 = [_]Datum{
        Datum.fromScalar(.{ .data_type = .{ .int32 = {} }, .value = .{ .i32 = 1 } }),
        Datum.fromScalar(.{ .data_type = .{ .int32 = {} }, .value = .{ .i32 = 2 } }),
    };
    try std.testing.expectError(
        error.InvalidArity,
        registry.resolveKernel("mixed_arity", .scalar, two_i32[0..], Options.noneValue()),
    );
    const reason = registry.explainResolveKernelFailure("mixed_arity", .scalar, two_i32[0..], Options.noneValue());
    try std.testing.expect(std.mem.eql(u8, reason, "no kernel matched minimum arity"));
}

test "compute registerKernel rejects invalid arity model combinations" {
    const allocator = std.testing.allocator;
    var registry = FunctionRegistry.init(allocator);
    defer registry.deinit();

    try std.testing.expectError(
        error.InvalidInput,
        registry.registerScalarKernel("invalid_sig_non_variadic_max", .{
            .signature = .{
                .arity = 2,
                .variadic = false,
                .max_arity = 3,
                .type_check = allInt32Datums,
            },
            .exec = exactArityMarkerKernel,
        }),
    );

    try std.testing.expectError(
        error.InvalidInput,
        registry.registerScalarKernel("invalid_sig_reversed_range", .{
            .signature = .{
                .arity = 4,
                .variadic = true,
                .max_arity = 3,
                .type_check = allInt32Datums,
            },
            .exec = rangeArityMarkerKernel,
        }),
    );
}

test "compute aggregate lifecycle session supports init/update/merge/finalize" {
    const allocator = std.testing.allocator;
    const int32_builder = @import("../array/array.zig").Int32Builder;
    var registry = FunctionRegistry.init(allocator);
    defer registry.deinit();

    try registry.registerAggregateKernel("count_stateful", .{
        .signature = .{
            .arity = 1,
            .type_check = unaryArray,
            .result_type_fn = countAggregateResultType,
        },
        .exec = countLenAggregateKernel,
        .aggregate_lifecycle = .{
            .init = countLifecycleInit,
            .update = countLifecycleUpdate,
            .merge = countLifecycleMerge,
            .finalize = countLifecycleFinalize,
            .deinit = countLifecycleDeinit,
        },
    });

    var b1 = try int32_builder.init(allocator, 3);
    defer b1.deinit();
    try b1.append(1);
    try b1.append(2);
    try b1.append(3);
    var a1 = try b1.finish();
    defer a1.release();

    var b2 = try int32_builder.init(allocator, 2);
    defer b2.deinit();
    try b2.append(10);
    try b2.append(20);
    var a2 = try b2.finish();
    defer a2.release();

    const args1 = [_]Datum{Datum.fromArray(a1.retain())};
    defer {
        var d = args1[0];
        d.release();
    }
    const args2 = [_]Datum{Datum.fromArray(a2.retain())};
    defer {
        var d = args2[0];
        d.release();
    }

    var ctx = ExecContext.init(allocator, &registry);
    var s1 = try ctx.beginAggregate("count_stateful", args1[0..], Options.noneValue());
    defer s1.deinit();
    var s2 = try ctx.beginAggregate("count_stateful", args2[0..], Options.noneValue());
    defer s2.deinit();

    try s1.update(args1[0..]);
    try s2.update(args2[0..]);
    try s1.merge(&s2);

    var out = try s1.finalize();
    defer out.release();
    try std.testing.expect(out.isScalar());
    try std.testing.expectEqual(@as(i64, 5), out.scalar.value.i64);
}

test "compute beginAggregate returns MissingLifecycle when not provided" {
    const allocator = std.testing.allocator;
    const int32_builder = @import("../array/array.zig").Int32Builder;
    var registry = FunctionRegistry.init(allocator);
    defer registry.deinit();

    try registry.registerAggregateKernel("count_stateless_only", .{
        .signature = KernelSignature.unary(unaryArray),
        .exec = countLenAggregateKernel,
    });

    var b = try int32_builder.init(allocator, 1);
    defer b.deinit();
    try b.append(1);
    var arr = try b.finish();
    defer arr.release();

    const args = [_]Datum{Datum.fromArray(arr.retain())};
    defer {
        var d = args[0];
        d.release();
    }

    var ctx = ExecContext.init(allocator, &registry);
    try std.testing.expectError(
        error.MissingLifecycle,
        ctx.beginAggregate("count_stateless_only", args[0..], Options.noneValue()),
    );
}

test "registerAll registers every kernel family function name with its kind" {
    const allocator = std.testing.allocator;
    var registry = FunctionRegistry.init(allocator);
    defer registry.deinit();
    try common.kernels.registerAll(&registry);

    const string_names = [_][]const u8{
        "ascii_upper",            "ascii_lower",            "utf8_upper",           "utf8_lower",
        "ascii_trim",             "ascii_ltrim",            "ascii_rtrim",          "ascii_trim_whitespace",
        "ascii_ltrim_whitespace", "ascii_rtrim_whitespace", "utf8_trim_whitespace", "utf8_ltrim_whitespace",
        "utf8_rtrim_whitespace",  "utf8_slice_codeunits",   "utf8_reverse",         "starts_with",
        "ends_with",              "match_substring",        "string_is_ascii",      "ascii_is_alpha",
        "ascii_is_decimal",       "ascii_is_space",         "utf8_is_alpha",        "utf8_is_digit",
        "utf8_is_space",          "utf8_length",            "binary_length",        "binary_join_element_wise",
        "ascii_swapcase",         "utf8_swapcase",          "ascii_capitalize",     "utf8_capitalize",
        "ascii_title",            "utf8_title",             "ascii_reverse",        "utf8_trim",
        "utf8_ltrim",             "utf8_rtrim",             "ascii_is_alnum",       "ascii_is_lower",
        "ascii_is_printable",     "ascii_is_title",         "ascii_is_upper",       "utf8_is_alnum",
        "utf8_is_decimal",        "utf8_is_lower",          "utf8_is_numeric",      "utf8_is_printable",
        "utf8_is_title",          "utf8_is_upper",
    };
    const string_transform_names = [_][]const u8{
        "replace_substring", "split_pattern", "ascii_split_whitespace", "utf8_split_whitespace",
        "ascii_lpad",        "ascii_rpad",    "ascii_center",           "utf8_lpad",
        "utf8_rpad",         "utf8_center",   "count_substring",        "find_substring",
        "binary_slice",      "binary_repeat", "binary_reverse",         "utf8_replace_slice",
        "utf8_zero_fill",    "binary_join",   "binary_replace_slice",
    };
    const temporal_names = [_][]const u8{
        "year",        "month",       "day",        "day_of_week",    "day_of_year", "hour",         "minute",  "second",
        "millisecond", "microsecond", "nanosecond", "iso_year",       "iso_week",    "iso_calendar", "quarter", "is_leap_year",
        "week",        "us_week",     "subsecond",  "year_month_day", "us_year",
    };
    const temporal_round_names = [_][]const u8{
        "floor_temporal",       "ceil_temporal",          "round_temporal",            "years_between",
        "quarters_between",     "month_interval_between", "weeks_between",             "days_between",
        "hours_between",        "minutes_between",        "seconds_between",           "milliseconds_between",
        "microseconds_between", "nanoseconds_between",    "day_time_interval_between", "month_day_nano_interval_between",
    };
    const math_names = [_][]const u8{
        "sqrt",       "sqrt_checked",       "exp",         "expm1",               "ln",            "ln_checked",   "log10",         "log10_checked",
        "log2",       "log2_checked",       "log1p",       "log1p_checked",       "sin",           "sin_checked",  "cos",           "cos_checked",
        "tan",        "tan_checked",        "asin",        "asin_checked",        "acos",          "acos_checked", "atan",          "sinh",
        "cosh",       "tanh",               "asinh",       "acosh",               "acosh_checked", "atanh",        "atanh_checked", "atan2",
        "logb",       "logb_checked",       "power",       "power_checked",       "bit_wise_and",  "bit_wise_or",  "bit_wise_xor",  "bit_wise_not",
        "shift_left", "shift_left_checked", "shift_right", "shift_right_checked",
    };
    const scalar_names = [_][]const u8{
        "add",            "add_checked",    "subtract",  "subtract_checked", "multiply",         "multiply_checked",
        "divide",         "divide_checked", "negate",    "negate_checked",   "abs",              "abs_checked",
        "equal",          "not_equal",      "less",      "less_equal",       "greater",          "greater_equal",
        "and",            "or",             "xor",       "and_not",          "invert",           "and_kleene",
        "or_kleene",      "is_null",        "is_valid",  "true_unless_null", "cast",             "is_in",
        "index_in",       "is_nan",         "is_finite", "is_inf",           "min_element_wise", "max_element_wise",
        "and_not_kleene", "sign",
    } ++ string_names ++ temporal_names ++ string_transform_names ++ temporal_round_names ++ math_names ++ [_][]const u8{
        "round",  "round_to_multiple", "round_binary", "ceil", "floor", "trunc", "coalesce", "if_else", "case_when", "choose",
        "random",
    };
    const vector_names = [_][]const u8{
        "unique",                 "value_counts",    "dictionary_encode",       "drop_null",             "fill_null_forward",   "fill_null_backward", "mode",
        "quantile",               "sort_indices",    "array_sort_indices",      "partition_nth_indices", "select_k_unstable",   "rank",               "rank_quantile",
        "rank_normal",            "search_sorted",   "replace_with_mask",       "scatter",               "inverse_permutation", "indices_nonzero",    "cumulative_sum",
        "cumulative_sum_checked", "cumulative_prod", "cumulative_prod_checked", "cumulative_min",        "cumulative_max",      "cumulative_mean",    "pairwise_diff",
        "pairwise_diff_checked",  "winsorize",
    };
    const aggregate_names = [_][]const u8{
        "sum",                "mean",        "min",  "max",      "min_max", "count", "count_all",  "any",            "all",   "product",
        "variance",           "stddev",      "skew", "kurtosis", "first",   "last",  "first_last", "count_distinct", "index", "tdigest",
        "approximate_median", "pivot_wider",
    };
    const hash_names = [_][]const u8{
        "hash_sum",   "hash_mean",           "hash_min",        "hash_max",     "hash_min_max",  "hash_count",   "hash_count_all",          "hash_any",
        "hash_all",   "hash_count_distinct", "hash_distinct",   "hash_product", "hash_variance", "hash_stddev",  "hash_skew",               "hash_kurtosis",
        "hash_first", "hash_last",           "hash_first_last", "hash_one",     "hash_list",     "hash_tdigest", "hash_approximate_median", "hash_pivot_wider",
    };
    for (scalar_names) |name| try std.testing.expect(registry.containsFunction(name, .scalar));
    for (aggregate_names) |name| {
        try std.testing.expect(registry.containsFunction(name, .aggregate));
        // Every aggregate kernel also carries the init / update / merge / finalize lifecycle.
        for (registry.findFunction(name, .aggregate).?.kernelsSlice()) |kernel| try std.testing.expect(kernel.supportsAggregateLifecycle());
    }
    for (hash_names) |name| {
        try std.testing.expect(registry.containsFunction(name, .hash_aggregate));
        for (registry.findFunction(name, .hash_aggregate).?.kernelsSlice()) |kernel| try std.testing.expect(kernel.supportsAggregateLifecycle());
    }
    for (vector_names) |name| try std.testing.expect(registry.containsFunction(name, .vector));
    try std.testing.expectEqual(scalar_names.len + aggregate_names.len + hash_names.len + vector_names.len, registry.functionCount());

    // One kernel per input type: 10 numeric types and one for every decimal / integer pair;
    // negate_checked has no unsigned kernels and adds decimal128 / decimal256; comparisons add
    // bool, the string and binary layouts, six temporal types, four decimal types,
    // fixed_size_binary, mixed byte layouts and mixed decimals; min / max_element_wise numbers,
    // temporal, strings, binaries, decimals and fixed_size_binary.
    // add also has the six temporal patterns (timestamp + duration, ...), negate_checked duration.
    try std.testing.expectEqual(@as(usize, 17), registry.kernelCount("add", .scalar));
    try std.testing.expectEqual(@as(usize, 9), registry.kernelCount("negate_checked", .scalar));
    try std.testing.expectEqual(@as(usize, 12), registry.kernelCount("sign", .scalar));
    try std.testing.expectEqual(@as(usize, 30), registry.kernelCount("less", .scalar));
    try std.testing.expectEqual(@as(usize, 27), registry.kernelCount("min_element_wise", .scalar));
    try std.testing.expectEqual(@as(usize, 16), registry.kernelCount("is_nan", .scalar));
    try std.testing.expectEqual(@as(usize, 2), registry.kernelCount("count_all", .aggregate));
    // Implicit numeric casts: binary arithmetic and the comparisons, not negate / abs.
    try std.testing.expectEqual(common.core.ImplicitCast.numeric, registry.findFunction("add_checked", .scalar).?.implicit_cast);
    try std.testing.expectEqual(common.core.ImplicitCast.comparison, registry.findFunction("greater", .scalar).?.implicit_cast);
    try std.testing.expectEqual(common.core.ImplicitCast.element_wise, registry.findFunction("max_element_wise", .scalar).?.implicit_cast);
    try std.testing.expectEqual(common.core.ImplicitCast.none, registry.findFunction("negate", .scalar).?.implicit_cast);
    try std.testing.expectEqual(@as(usize, 1), registry.kernelCount("and_kleene", .scalar));
    // min_max: ten numeric types, decimal128 / decimal256 and one kernel for the ordered
    // non-numeric types (bool, temporal, decimal32 / decimal64, string / binary layouts).
    try std.testing.expectEqual(@as(usize, 13), registry.kernelCount("min_max", .aggregate));
    // One cast kernel for every pair; string kernels per string layout (binary_length also for the
    // binary layouts); temporal date parts for date32 / date64 / timestamp, time parts without date32.
    try std.testing.expectEqual(@as(usize, 1), registry.kernelCount("cast", .scalar));
    try std.testing.expectEqual(@as(usize, 3), registry.kernelCount("utf8_upper", .scalar));
    try std.testing.expectEqual(@as(usize, 6), registry.kernelCount("binary_length", .scalar));
    try std.testing.expectEqual(@as(usize, 3), registry.kernelCount("year", .scalar));
    try std.testing.expectEqual(@as(usize, 4), registry.kernelCount("hour", .scalar));
    try std.testing.expectEqual(@as(usize, 3), registry.kernelCount("subsecond", .scalar));
    try std.testing.expectEqual(@as(usize, 5), registry.kernelCount("hours_between", .scalar));
    try std.testing.expectEqual(@as(usize, 3), registry.kernelCount("years_between", .scalar));
    try std.testing.expectEqual(@as(usize, 8), registry.kernelCount("binary_join", .scalar));
    try std.testing.expectEqual(@as(usize, 6), registry.kernelCount("replace_substring", .scalar));
    try std.testing.expectEqual(@as(usize, 3), registry.kernelCount("binary_slice", .scalar));
    try std.testing.expectEqual(@as(usize, 5), registry.kernelCount("floor_temporal", .scalar));
    try std.testing.expectEqual(@as(usize, 1), registry.kernelCount("is_in", .scalar));
    try std.testing.expectEqual(@as(usize, 12), registry.kernelCount("hash_sum", .hash_aggregate));
    try std.testing.expectEqual(@as(usize, 12), registry.kernelCount("sum", .aggregate));
    // Kinds are separate namespaces: no scalar "sum".
    try std.testing.expect(!registry.containsFunction("sum", .scalar));
}

test "ExecContext.invoke add works end to end after registerAll" {
    const allocator = std.testing.allocator;
    var registry = FunctionRegistry.init(allocator);
    defer registry.deinit();
    try common.kernels.registerAll(&registry);
    var ctx = ExecContext.init(allocator, &registry);

    var lhs = try common.makeInt32Array(allocator, &.{ 1, null, 3, 2147483647 });
    defer lhs.release();
    var rhs = try common.makeInt32Array(allocator, &.{ 10, 20, null, 1 });
    defer rhs.release();
    const args = [_]Datum{ .{ .array = lhs }, .{ .array = rhs } };
    var out = try ctx.invoke("add", .scalar, &args, Options.noneValue());
    defer out.release();
    const values = try common.collectInt32ValuesFromDatum(allocator, out);
    defer allocator.free(values);
    try std.testing.expectEqualSlices(?i32, &.{ 11, null, null, -2147483648 }, values);
    try std.testing.expectError(error.Overflow, ctx.invoke("add_checked", .scalar, &args, Options.noneValue()));

    try std.testing.expect((try registry.resolveResultType("add", .scalar, &args, Options.noneValue())) == .int32);
    try std.testing.expect((try registry.resolveResultType("less", .scalar, &args, Options.noneValue())) == .bool);
    try std.testing.expect((try registry.resolveResultType("sum", .aggregate, args[0..1], Options.noneValue())) == .int64);
    try std.testing.expect((try registry.resolveResultType("mean", .aggregate, args[0..1], Options.noneValue())) == .double);
    try std.testing.expectError(error.FunctionNotFound, ctx.invoke("no_such_function", .scalar, &args, Options.noneValue()));

    var total = try ctx.invokeAggregate("sum", &.{.{ .array = out.array }}, Options.noneValue());
    defer total.release();
    try std.testing.expectEqual(@as(i64, 11 - 2147483648), total.scalar.value.i64);
}
