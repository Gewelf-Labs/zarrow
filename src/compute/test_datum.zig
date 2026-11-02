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
const datumTakeNullable = @import("core.zig").datumTakeNullable;

test "compute datum chunked variant retain and release" {
    const allocator = std.testing.allocator;
    const int32_builder = @import("../array/array.zig").Int32Builder;

    var builder = try int32_builder.init(allocator, 2);
    defer builder.deinit();
    try builder.append(1);
    try builder.append(2);
    var arr = try builder.finish();
    defer arr.release();

    var chunked = try ChunkedArray.fromSingle(allocator, arr);
    defer chunked.release();

    var datum = Datum{ .chunked = chunked.retain() };
    defer datum.release();

    try std.testing.expect(datum.dataType().eql(.{ .int32 = {} }));
    try std.testing.expectEqual(@as(usize, 2), datum.chunked.len());
}

test "compute datum helpers and shared type predicates" {
    const scalar_i32 = Scalar.init(.{ .int32 = {} }, .{ .i32 = 42 });
    const scalar_i64 = Scalar.init(.{ .int64 = {} }, .{ .i64 = 42 });

    const args_numeric = [_]Datum{
        Datum.fromScalar(scalar_i32),
        Datum.fromScalar(scalar_i64),
    };

    try std.testing.expect(hasArity(args_numeric[0..], 2));
    try std.testing.expect(!sameDataTypes(args_numeric[0..]));
    try std.testing.expect(allNumeric(args_numeric[0..]));
    try std.testing.expect(!binarySameNumeric(args_numeric[0..]));

    const args_same_numeric = [_]Datum{
        Datum.fromScalar(scalar_i32),
        Datum.fromScalar(scalar_i32),
    };
    try std.testing.expect(sameDataTypes(args_same_numeric[0..]));
    try std.testing.expect(binarySameNumeric(args_same_numeric[0..]));
    try std.testing.expect(unaryScalar(args_same_numeric[0..1]));
}

test "compute datum accessors and signature helpers" {
    const allocator = std.testing.allocator;
    const int32_builder = @import("../array/array.zig").Int32Builder;
    var builder = try int32_builder.init(allocator, 2);
    defer builder.deinit();
    try builder.append(1);
    try builder.append(2);
    var arr = try builder.finish();
    defer arr.release();

    var chunked = try ChunkedArray.fromSingle(allocator, arr);
    defer chunked.release();

    var datum_arr = Datum.fromArray(arr.retain());
    defer datum_arr.release();
    try std.testing.expect(datum_arr.isArray());
    try std.testing.expect(!datum_arr.isChunked());
    try std.testing.expect(!datum_arr.isScalar());
    try std.testing.expect(datum_arr.asArray() != null);
    try std.testing.expect(datum_arr.asChunked() == null);
    try std.testing.expect(datum_arr.asScalar() == null);

    var datum_chunked = Datum.fromChunked(chunked.retain());
    defer datum_chunked.release();
    try std.testing.expect(datum_chunked.isChunked());
    try std.testing.expect(datum_chunked.asChunked() != null);
    try std.testing.expectEqual(@as(usize, 2), datum_chunked.asChunked().?.len());

    const scalar = Scalar.init(.{ .int32 = {} }, .{ .i32 = 7 });
    var datum_scalar = Datum.fromScalar(scalar);
    defer datum_scalar.release();
    try std.testing.expect(datum_scalar.isScalar());
    try std.testing.expectEqual(@as(i32, 7), datum_scalar.asScalar().?.value.i32);

    const args_bad = [_]Datum{ Datum.fromScalar(Scalar.init(.{ .int32 = {} }, .{ .i32 = 1 })), Datum.fromScalar(Scalar.init(.{ .int64 = {} }, .{ .i64 = 2 })) };
    const args_ok = [_]Datum{ Datum.fromScalar(scalar), Datum.fromScalar(scalar) };
    const sig_any = KernelSignature.any(2);
    const sig_binary_int32 = KernelSignature.binary(isTwoInt32);
    try std.testing.expect(sig_any.matches(args_bad[0..]));
    try std.testing.expect(!sig_binary_int32.matches(args_bad[0..]));
    try std.testing.expect(sig_binary_int32.matches(args_ok[0..]));
    const inferred_binary = try sig_binary_int32.inferResultType(args_ok[0..], Options.noneValue());
    try std.testing.expect(inferred_binary.eql(.{ .int32 = {} }));
}

test "compute kernel signature supports exact at_least and range arity" {
    const scalar_i32 = Datum.fromScalar(.{
        .data_type = .{ .int32 = {} },
        .value = .{ .i32 = 1 },
    });
    const one = [_]Datum{scalar_i32};
    const two = [_]Datum{ scalar_i32, scalar_i32 };
    const three = [_]Datum{ scalar_i32, scalar_i32, scalar_i32 };
    const four = [_]Datum{ scalar_i32, scalar_i32, scalar_i32, scalar_i32 };
    const five = [_]Datum{ scalar_i32, scalar_i32, scalar_i32, scalar_i32, scalar_i32 };

    const sig_exact = KernelSignature.any(2);
    const sig_at_least = KernelSignature.atLeast(2);
    const sig_range = KernelSignature.range(2, 4);

    try std.testing.expect(!sig_exact.matches(one[0..]));
    try std.testing.expect(sig_exact.matches(two[0..]));
    try std.testing.expect(!sig_exact.matches(three[0..]));

    try std.testing.expect(!sig_at_least.matches(one[0..]));
    try std.testing.expect(sig_at_least.matches(two[0..]));
    try std.testing.expect(sig_at_least.matches(five[0..]));

    try std.testing.expect(!sig_range.matches(one[0..]));
    try std.testing.expect(sig_range.matches(two[0..]));
    try std.testing.expect(sig_range.matches(three[0..]));
    try std.testing.expect(sig_range.matches(four[0..]));
    try std.testing.expect(!sig_range.matches(five[0..]));
}

test "compute scalar value temporal and decimal coverage" {
    const scalars = [_]Scalar{
        .{ .data_type = .{ .date32 = {} }, .value = .{ .date32 = 18_630 } },
        .{ .data_type = .{ .date64 = {} }, .value = .{ .date64 = 1_609_545_600_000 } },
        .{ .data_type = .{ .time32 = .{ .unit = .millisecond } }, .value = .{ .time32 = 1234 } },
        .{ .data_type = .{ .time64 = .{ .unit = .nanosecond } }, .value = .{ .time64 = 99_000 } },
        .{ .data_type = .{ .timestamp = .{ .unit = .microsecond, .timezone = "UTC" } }, .value = .{ .timestamp = 1_700_000_000_123_456 } },
        .{ .data_type = .{ .duration = .{ .unit = .nanosecond } }, .value = .{ .duration = 42 } },
        .{ .data_type = .{ .interval_months = .{ .unit = .months } }, .value = .{ .interval_months = 7 } },
        .{ .data_type = .{ .interval_day_time = .{ .unit = .day_time } }, .value = .{ .interval_day_time = -43_200_000 } },
        .{ .data_type = .{ .interval_month_day_nano = .{ .unit = .month_day_nano } }, .value = .{ .interval_month_day_nano = -123_456_789_012_345_678 } },
        .{ .data_type = .{ .decimal32 = .{ .precision = 9, .scale = 2 } }, .value = .{ .decimal32 = 123_45 } },
        .{ .data_type = .{ .decimal64 = .{ .precision = 18, .scale = 4 } }, .value = .{ .decimal64 = -9_876_543_210 } },
        .{ .data_type = .{ .decimal128 = .{ .precision = 38, .scale = 10 } }, .value = .{ .decimal128 = -987_654_321_098_765_432 } },
        .{ .data_type = .{ .decimal256 = .{ .precision = 76, .scale = 20 } }, .value = .{ .decimal256 = -987_654_321_098_765_432_109_876_543_210 } },
        .{ .data_type = .{ .string = {} }, .value = .{ .string = "borrowed-string" } },
        .{ .data_type = .{ .binary = {} }, .value = .{ .binary = "borrowed-binary" } },
    };

    try std.testing.expectEqual(@as(usize, 15), scalars.len);
    try std.testing.expect(scalars[0].data_type == .date32);
    try std.testing.expect(scalars[12].data_type == .decimal256);
    try std.testing.expectEqual(@as(i32, 18_630), scalars[0].value.date32);
    try std.testing.expectEqual(@as(i256, -987_654_321_098_765_432_109_876_543_210), scalars[12].value.decimal256);
    try std.testing.expectEqualStrings("borrowed-string", scalars[13].value.string);
    try std.testing.expectEqualStrings("borrowed-binary", scalars[14].value.binary);
}

test "compute exec context config and scalar payload duplication" {
    const allocator = std.testing.allocator;
    var registry = FunctionRegistry.init(allocator);
    defer registry.deinit();

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();

    var ctx = ExecContext.initWithConfig(allocator, &registry, .{
        .safe_cast = false,
        .overflow_mode = .wrapping,
        .threads = 0,
        .arena_allocator = arena.allocator(),
    });
    try std.testing.expect(!ctx.safeCastEnabled());
    try std.testing.expect(ctx.overflowMode() == .wrapping);
    try std.testing.expectEqual(@as(usize, 1), ctx.threads());

    const s = try ctx.dupScalarString("hello");
    const b = try ctx.dupScalarBinary("world");
    try std.testing.expectEqualStrings("hello", s);
    try std.testing.expectEqualStrings("world", b);
}

test "compute nested scalar payload is retained across datum and exec chunk lifecycle" {
    const allocator = std.testing.allocator;

    const item_type = DataType{ .int32 = {} };
    const item_field = datatype.Field{
        .name = "item",
        .data_type = &item_type,
        .nullable = true,
    };

    var values = try makeInt32Array(allocator, &[_]?i32{ 10, 20 });
    defer values.release();

    var list_builder = try array_mod.ListBuilder.init(allocator, 1, item_field);
    defer list_builder.deinit();
    try list_builder.appendLen(2);
    var list_array = try list_builder.finish(values);
    defer list_array.release();

    var scalar = try Scalar.initNested(list_array.data().data_type, list_array);
    defer scalar.release();
    try std.testing.expect(scalar.value == .list);
    try std.testing.expect(scalar.payload != null);

    var datum = Datum.fromScalar(scalar.retain());
    defer datum.release();

    var iter_input = datum.retain();
    defer iter_input.release();
    var iter = UnaryExecChunkIterator.init(iter_input);
    var chunk = (try iter.next()).?;
    defer chunk.deinit();
    try std.testing.expectEqual(@as(usize, 1), chunk.len);
    try std.testing.expect(chunk.values == .scalar);
    try std.testing.expect(chunk.values.scalar.value == .list);
    try std.testing.expect(chunk.values.scalar.payload != null);
}

test "compute datum list extraction supports array chunked and scalar inputs" {
    const allocator = std.testing.allocator;
    const int32_array = @import("../array/array.zig").Int32Array;

    const item_type = DataType{ .int32 = {} };
    const item_field = datatype.Field{
        .name = "item",
        .data_type = &item_type,
        .nullable = true,
    };

    var values = try makeInt32Array(allocator, &[_]?i32{ 1, 2, 3 });
    defer values.release();

    var list_builder = try array_mod.ListBuilder.init(allocator, 3, item_field);
    defer list_builder.deinit();
    try list_builder.appendLen(2); // [1, 2]
    try list_builder.appendNull(); // null
    try list_builder.appendLen(1); // [3]
    var list_array = try list_builder.finish(values);
    defer list_array.release();

    var list_slice = try list_array.slice(1, 2); // [null, [3]]
    defer list_slice.release();
    var array_datum = Datum.fromArray(list_slice.retain());
    defer array_datum.release();
    var out_array = try datumListValueAt(array_datum, 1);
    defer out_array.release();
    const out_array_view = int32_array{ .data = out_array.data() };
    try std.testing.expectEqual(@as(usize, 1), out_array_view.len());
    try std.testing.expectEqual(@as(i32, 3), try out_array_view.value(0));

    var list_chunk0 = try list_array.slice(0, 1);
    defer list_chunk0.release();
    var list_chunk1 = try list_array.slice(1, 2);
    defer list_chunk1.release();
    var chunked = try ChunkedArray.init(allocator, list_array.data().data_type, &[_]ArrayRef{ list_chunk0, list_chunk1 });
    defer chunked.release();
    var chunked_datum = Datum.fromChunked(chunked.retain());
    defer chunked_datum.release();
    var out_chunked = try datumListValueAt(chunked_datum, 2);
    defer out_chunked.release();
    const out_chunked_view = int32_array{ .data = out_chunked.data() };
    try std.testing.expectEqual(@as(usize, 1), out_chunked_view.len());
    try std.testing.expectEqual(@as(i32, 3), try out_chunked_view.value(0));

    var scalar_payload = try list_array.slice(2, 1);
    defer scalar_payload.release();
    var scalar_datum = Datum.fromScalar(try Scalar.initNested(list_array.data().data_type, scalar_payload));
    defer scalar_datum.release();
    var out_scalar = try datumListValueAt(scalar_datum, 1234);
    defer out_scalar.release();
    const out_scalar_view = int32_array{ .data = out_scalar.data() };
    try std.testing.expectEqual(@as(usize, 1), out_scalar_view.len());
    try std.testing.expectEqual(@as(i32, 3), try out_scalar_view.value(0));

    var null_payload = try list_array.slice(1, 1);
    defer null_payload.release();
    var null_scalar_datum = Datum.fromScalar(try Scalar.initNested(list_array.data().data_type, null_payload));
    defer null_scalar_datum.release();
    try std.testing.expectError(error.InvalidInput, datumListValueAt(null_scalar_datum, 0));
}

test "compute datumStructField supports scalar struct bool fields" {
    const allocator = std.testing.allocator;

    const bool_type = DataType{ .bool = {} };
    const fields = [_]datatype.Field{
        .{ .name = "a", .data_type = &bool_type, .nullable = true },
        .{ .name = "b", .data_type = &bool_type, .nullable = true },
    };

    var child_a = try makeBoolArray(allocator, &[_]?bool{true});
    defer child_a.release();
    var child_b = try makeBoolArray(allocator, &[_]?bool{false});
    defer child_b.release();

    var struct_builder = array_mod.StructBuilder.init(allocator, fields[0..]);
    defer struct_builder.deinit();
    try struct_builder.appendValid();
    var struct_array = try struct_builder.finish(&[_]ArrayRef{ child_a, child_b });
    defer struct_array.release();

    var struct_scalar_datum = Datum.fromScalar(try Scalar.initNested(struct_array.data().data_type, struct_array));
    defer struct_scalar_datum.release();

    var field0 = try datumStructField(struct_scalar_datum, 0);
    defer field0.release();
    try std.testing.expect(field0.isScalar());
    try std.testing.expect(field0.scalar.data_type == .bool);
    try std.testing.expectEqual(true, field0.scalar.value.bool);

    var field1 = try datumStructField(struct_scalar_datum, 1);
    defer field1.release();
    try std.testing.expect(field1.isScalar());
    try std.testing.expectEqual(false, field1.scalar.value.bool);

    var struct_array_datum = Datum.fromArray(struct_array.retain());
    defer struct_array_datum.release();
    var field0_array = try datumStructField(struct_array_datum, 0);
    defer field0_array.release();
    try std.testing.expect(field0_array.isArray());
    const bool_view = array_mod.BooleanArray{ .data = field0_array.array.data() };
    try std.testing.expectEqual(@as(usize, 1), bool_view.len());
    try std.testing.expectEqual(true, try bool_view.value(0));

    var null_child_a = try makeBoolArray(allocator, &[_]?bool{true});
    defer null_child_a.release();
    var null_child_b = try makeBoolArray(allocator, &[_]?bool{false});
    defer null_child_b.release();
    var null_struct_builder = array_mod.StructBuilder.init(allocator, fields[0..]);
    defer null_struct_builder.deinit();
    try null_struct_builder.appendNull();
    var null_struct_array = try null_struct_builder.finish(&[_]ArrayRef{ null_child_a, null_child_b });
    defer null_struct_array.release();

    var null_struct_scalar = Datum.fromScalar(try Scalar.initNested(null_struct_array.data().data_type, null_struct_array));
    defer null_struct_scalar.release();
    var null_field = try datumStructField(null_struct_scalar, 0);
    defer null_field.release();
    try std.testing.expect(null_field.isScalar());
    try std.testing.expect(null_field.scalar.isNull());
}

test "compute datumBuildEmptyLike and datumSliceEmpty preserve nested layout" {
    const allocator = std.testing.allocator;

    const int32_type = DataType{ .int32 = {} };
    const bool_type = DataType{ .bool = {} };
    const list_field = datatype.Field{
        .name = "item",
        .data_type = &int32_type,
        .nullable = true,
    };
    const struct_fields = [_]datatype.Field{
        .{ .name = "x", .data_type = &int32_type, .nullable = true },
        .{ .name = "flag", .data_type = &bool_type, .nullable = true },
    };

    const list_type = DataType{ .list = .{ .value_field = list_field } };
    const fixed_size_list_type = DataType{ .fixed_size_list = .{ .list_size = 2, .value_field = list_field } };
    const struct_type = DataType{ .struct_ = .{ .fields = struct_fields[0..] } };

    var empty_list = try datumBuildEmptyLike(list_type);
    defer empty_list.release();
    try std.testing.expect(empty_list.isArray());
    try std.testing.expectEqual(@as(usize, 0), empty_list.array.data().length);
    try std.testing.expect(empty_list.array.data().data_type.eql(list_type));
    try empty_list.array.data().validateLayout();

    var empty_fsl = try datumBuildEmptyLike(fixed_size_list_type);
    defer empty_fsl.release();
    try std.testing.expect(empty_fsl.isArray());
    try std.testing.expectEqual(@as(usize, 0), empty_fsl.array.data().length);
    try std.testing.expect(empty_fsl.array.data().data_type.eql(fixed_size_list_type));
    try empty_fsl.array.data().validateLayout();

    var empty_struct = try datumBuildEmptyLike(struct_type);
    defer empty_struct.release();
    try std.testing.expect(empty_struct.isArray());
    try std.testing.expectEqual(@as(usize, 0), empty_struct.array.data().length);
    try std.testing.expect(empty_struct.array.data().data_type.eql(struct_type));
    try empty_struct.array.data().validateLayout();

    var values = try makeInt32Array(allocator, &[_]?i32{ 1, 2 });
    defer values.release();
    var list_builder = try array_mod.ListBuilder.init(allocator, 1, list_field);
    defer list_builder.deinit();
    try list_builder.appendLen(2);
    var list_one = try list_builder.finish(values);
    defer list_one.release();

    var scalar_list = Datum.fromScalar(try Scalar.initNested(list_type, list_one));
    defer scalar_list.release();
    var sliced_empty = try datumSliceEmpty(scalar_list);
    defer sliced_empty.release();
    try std.testing.expect(sliced_empty.isArray());
    try std.testing.expectEqual(@as(usize, 0), sliced_empty.array.data().length);
    try std.testing.expect(sliced_empty.array.data().data_type.eql(list_type));
    try sliced_empty.array.data().validateLayout();
}

test "compute chunkedResolveLogicalIndices maps chunk-local coordinates and validates bounds" {
    const allocator = std.testing.allocator;

    var c0 = try makeInt32Array(allocator, &[_]?i32{ 10, 20 });
    defer c0.release();
    var c1 = try makeInt32Array(allocator, &[_]?i32{ 30, 40, 50 });
    defer c1.release();
    var chunks = try ChunkedArray.init(allocator, .{ .int32 = {} }, &[_]ArrayRef{ c0, c1 });
    defer chunks.release();

    const mapped = try chunkedResolveLogicalIndices(allocator, chunks, &[_]usize{ 0, 1, 2, 4 });
    defer allocator.free(mapped);

    try std.testing.expectEqual(@as(usize, 4), mapped.len);
    try std.testing.expectEqual(@as(usize, 0), mapped[0].chunk_index);
    try std.testing.expectEqual(@as(usize, 0), mapped[0].index_in_chunk);
    try std.testing.expectEqual(@as(usize, 0), mapped[1].chunk_index);
    try std.testing.expectEqual(@as(usize, 1), mapped[1].index_in_chunk);
    try std.testing.expectEqual(@as(usize, 1), mapped[2].chunk_index);
    try std.testing.expectEqual(@as(usize, 0), mapped[2].index_in_chunk);
    try std.testing.expectEqual(@as(usize, 1), mapped[3].chunk_index);
    try std.testing.expectEqual(@as(usize, 2), mapped[3].index_in_chunk);

    try std.testing.expectError(
        error.InvalidInput,
        chunkedResolveLogicalIndices(allocator, chunks, &[_]usize{5}),
    );
}

test "compute datumTake keeps chunked output and matches array logical result on misaligned boundaries" {
    const allocator = std.testing.allocator;

    var base = try makeInt32Array(allocator, &[_]?i32{ 10, 20, 30, 40, 50 });
    defer base.release();

    var c0 = try makeInt32Array(allocator, &[_]?i32{ 10, 20 });
    defer c0.release();
    var c1 = try makeInt32Array(allocator, &[_]?i32{30});
    defer c1.release();
    var c2 = try makeInt32Array(allocator, &[_]?i32{ 40, 50 });
    defer c2.release();
    var chunked = try ChunkedArray.init(allocator, .{ .int32 = {} }, &[_]ArrayRef{ c0, c1, c2 });
    defer chunked.release();

    var array_datum = Datum.fromArray(base.retain());
    defer array_datum.release();
    var chunked_datum = Datum.fromChunked(chunked.retain());
    defer chunked_datum.release();

    const indices = [_]usize{ 1, 2, 4, 0 };

    var out_array = try datumTake(array_datum, indices[0..]);
    defer out_array.release();
    var out_chunked = try datumTake(chunked_datum, indices[0..]);
    defer out_chunked.release();

    try std.testing.expect(out_array.isArray());
    try std.testing.expect(out_chunked.isChunked());
    try std.testing.expectEqual(@as(usize, 4), out_chunked.chunked.len());

    const array_values = try collectInt32ValuesFromDatum(allocator, out_array);
    defer allocator.free(array_values);
    const chunked_values = try collectInt32ValuesFromDatum(allocator, out_chunked);
    defer allocator.free(chunked_values);

    try std.testing.expectEqual(@as(usize, array_values.len), chunked_values.len);
    var i: usize = 0;
    while (i < array_values.len) : (i += 1) {
        try std.testing.expectEqual(array_values[i], chunked_values[i]);
    }
    try std.testing.expectEqual(@as(?i32, 20), chunked_values[0]);
    try std.testing.expectEqual(@as(?i32, 30), chunked_values[1]);
    try std.testing.expectEqual(@as(?i32, 50), chunked_values[2]);
    try std.testing.expectEqual(@as(?i32, 10), chunked_values[3]);
}

test "compute datumTake direct gather matches slice+concat path for int32 and bool" {
    const allocator = std.testing.allocator;

    // Array datums hit the direct gather kernel; single-chunk chunked datums use the existing slice+concat path.
    var ints = try makeInt32Array(allocator, &[_]?i32{ 10, null, 30, 40, null, 60 });
    defer ints.release();
    var bools = try makeBoolArray(allocator, &[_]?bool{ true, null, false, true, true, null });
    defer bools.release();
    const indices = [_]?usize{ 5, 0, 3, 3, null, 1, 2, 0, 4 };

    var int_chunked = try ChunkedArray.init(allocator, .{ .int32 = {} }, &[_]ArrayRef{ints});
    defer int_chunked.release();
    var int_fast = try datumTakeNullable(Datum.fromArray(ints), indices[0..]);
    defer int_fast.release();
    var int_slow = try datumTakeNullable(Datum.fromChunked(int_chunked), indices[0..]);
    defer int_slow.release();
    try std.testing.expect(int_fast.isArray());
    try std.testing.expectEqual(@as(?usize, 3), int_fast.array.data().null_count);
    const fast_values = try collectInt32ValuesFromDatum(allocator, int_fast);
    defer allocator.free(fast_values);
    const slow_values = try collectInt32ValuesFromDatum(allocator, int_slow);
    defer allocator.free(slow_values);
    try std.testing.expectEqualSlices(?i32, slow_values, fast_values);
    try std.testing.expectEqualSlices(?i32, &[_]?i32{ 60, 10, 40, 40, null, null, 30, 10, null }, fast_values);

    var bool_fast = try datumTakeNullable(Datum.fromArray(bools), indices[0..]);
    defer bool_fast.release();
    const fast_bools = array_mod.BooleanArray{ .data = bool_fast.array.data() };
    const expected_bools = [_]?bool{ null, true, true, true, null, null, false, true, true };
    for (expected_bools, 0..) |expected, i| {
        try std.testing.expectEqual(expected == null, fast_bools.isNull(i));
        if (expected) |e| try std.testing.expectEqual(e, try fast_bools.value(i));
    }

    // Out-of-range index is rejected, not read.
    try std.testing.expectError(error.InvalidInput, datumTakeNullable(Datum.fromArray(ints), &[_]?usize{6}));
}

test "compute datumFilterSelectionIndices and datumFilterChunkAware stay consistent for array and chunked" {
    const allocator = std.testing.allocator;

    var values_array = try makeInt32Array(allocator, &[_]?i32{ 1, 2, 3, 4, 5 });
    defer values_array.release();
    var pred_array = try makeBoolArray(allocator, &[_]?bool{ true, false, null, true, true });
    defer pred_array.release();

    var v0 = try makeInt32Array(allocator, &[_]?i32{ 1, 2 });
    defer v0.release();
    var v1 = try makeInt32Array(allocator, &[_]?i32{ 3, 4, 5 });
    defer v1.release();
    var values_chunked = try ChunkedArray.init(allocator, .{ .int32 = {} }, &[_]ArrayRef{ v0, v1 });
    defer values_chunked.release();

    var p0 = try makeBoolArray(allocator, &[_]?bool{true});
    defer p0.release();
    var p1 = try makeBoolArray(allocator, &[_]?bool{ false, null, true });
    defer p1.release();
    var p2 = try makeBoolArray(allocator, &[_]?bool{true});
    defer p2.release();
    var pred_chunked = try ChunkedArray.init(allocator, .{ .bool = {} }, &[_]ArrayRef{ p0, p1, p2 });
    defer pred_chunked.release();

    var pred_array_datum = Datum.fromArray(pred_array.retain());
    defer pred_array_datum.release();
    const selections = try datumFilterSelectionIndices(allocator, pred_array_datum, 5, .{ .drop_nulls = false });
    defer allocator.free(selections);
    try std.testing.expectEqual(@as(usize, 4), selections.len);
    try std.testing.expectEqual(@as(?usize, 0), selections[0]);
    try std.testing.expectEqual(@as(?usize, null), selections[1]);
    try std.testing.expectEqual(@as(?usize, 3), selections[2]);
    try std.testing.expectEqual(@as(?usize, 4), selections[3]);

    var array_datum = Datum.fromArray(values_array.retain());
    defer array_datum.release();
    var chunked_datum = Datum.fromChunked(values_chunked.retain());
    defer chunked_datum.release();
    var pred_chunked_datum = Datum.fromChunked(pred_chunked.retain());
    defer pred_chunked_datum.release();

    var filtered_array = try datumFilterChunkAware(array_datum, pred_array_datum, .{ .drop_nulls = false });
    defer filtered_array.release();
    var filtered_chunked = try datumFilterChunkAware(chunked_datum, pred_chunked_datum, .{ .drop_nulls = false });
    defer filtered_chunked.release();

    try std.testing.expect(filtered_array.isArray());
    try std.testing.expect(filtered_chunked.isChunked());

    const expected = try collectInt32ValuesFromDatum(allocator, filtered_array);
    defer allocator.free(expected);
    const actual = try collectInt32ValuesFromDatum(allocator, filtered_chunked);
    defer allocator.free(actual);

    try std.testing.expectEqual(@as(usize, expected.len), actual.len);
    var i: usize = 0;
    while (i < expected.len) : (i += 1) {
        try std.testing.expectEqual(expected[i], actual[i]);
    }
}

test "compute datumFilter keeps compatibility array output shape for chunked inputs" {
    const allocator = std.testing.allocator;

    var v0 = try makeInt32Array(allocator, &[_]?i32{ 1, 2 });
    defer v0.release();
    var v1 = try makeInt32Array(allocator, &[_]?i32{ 3, 4 });
    defer v1.release();
    var values_chunked = try ChunkedArray.init(allocator, .{ .int32 = {} }, &[_]ArrayRef{ v0, v1 });
    defer values_chunked.release();

    var p0 = try makeBoolArray(allocator, &[_]?bool{ true, false, true, true });
    defer p0.release();
    var values_datum = Datum.fromChunked(values_chunked.retain());
    defer values_datum.release();
    var pred_datum = Datum.fromArray(p0.retain());
    defer pred_datum.release();

    var filtered = try datumFilter(values_datum, pred_datum, .{ .drop_nulls = true });
    defer filtered.release();
    try std.testing.expect(filtered.isArray());
}

test "compute datumFilter supports scalar array chunked and fixed_size_list null alignment" {
    const allocator = std.testing.allocator;
    const int32_array = array_mod.Int32Array;

    const int32_type = DataType{ .int32 = {} };
    const list_field = datatype.Field{
        .name = "item",
        .data_type = &int32_type,
        .nullable = true,
    };
    const list_type = DataType{ .list = .{ .value_field = list_field } };

    var scalar_values_child = try makeInt32Array(allocator, &[_]?i32{ 7, 8 });
    defer scalar_values_child.release();
    var scalar_list_builder = try array_mod.ListBuilder.init(allocator, 1, list_field);
    defer scalar_list_builder.deinit();
    try scalar_list_builder.appendLen(2);
    var scalar_list_payload = try scalar_list_builder.finish(scalar_values_child);
    defer scalar_list_payload.release();

    var scalar_datum = Datum.fromScalar(try Scalar.initNested(list_type, scalar_list_payload));
    defer scalar_datum.release();

    var pred_array = try makeBoolArray(allocator, &[_]?bool{ true, false, true });
    defer pred_array.release();
    var pred_datum = Datum.fromArray(pred_array.retain());
    defer pred_datum.release();

    var filtered_scalar = try datumFilter(scalar_datum, pred_datum, .{ .drop_nulls = true });
    defer filtered_scalar.release();
    try std.testing.expect(filtered_scalar.isArray());
    try std.testing.expectEqual(@as(usize, 2), filtered_scalar.array.data().length);
    try filtered_scalar.array.data().validateLayout();
    const filtered_scalar_list = array_mod.ListArray{ .data = filtered_scalar.array.data() };
    var fs0 = try filtered_scalar_list.value(0);
    defer fs0.release();
    const fs0_values = int32_array{ .data = fs0.data() };
    try std.testing.expectEqual(@as(i32, 7), try fs0_values.value(0));
    try std.testing.expectEqual(@as(i32, 8), try fs0_values.value(1));

    var list_values = try makeInt32Array(allocator, &[_]?i32{ 1, 2, 3 });
    defer list_values.release();
    var list_builder = try array_mod.ListBuilder.init(allocator, 3, list_field);
    defer list_builder.deinit();
    try list_builder.appendLen(1);
    try list_builder.appendLen(1);
    try list_builder.appendLen(1);
    var list_array = try list_builder.finish(list_values);
    defer list_array.release();

    var pred_chunk0 = try makeBoolArray(allocator, &[_]?bool{ true, null });
    defer pred_chunk0.release();
    var pred_chunk1 = try makeBoolArray(allocator, &[_]?bool{false});
    defer pred_chunk1.release();
    var pred_chunked = try ChunkedArray.init(allocator, .{ .bool = {} }, &[_]ArrayRef{ pred_chunk0, pred_chunk1 });
    defer pred_chunked.release();

    var list_datum = Datum.fromArray(list_array.retain());
    defer list_datum.release();
    var pred_chunked_datum = Datum.fromChunked(pred_chunked.retain());
    defer pred_chunked_datum.release();

    var filtered_drop_nulls = try datumFilter(list_datum, pred_chunked_datum, .{ .drop_nulls = true });
    defer filtered_drop_nulls.release();
    try std.testing.expectEqual(@as(usize, 1), filtered_drop_nulls.array.data().length);

    var filtered_keep_nulls = try datumFilter(list_datum, pred_chunked_datum, .{ .drop_nulls = false });
    defer filtered_keep_nulls.release();
    try std.testing.expectEqual(@as(usize, 2), filtered_keep_nulls.array.data().length);
    try filtered_keep_nulls.array.data().validateLayout();
    const filtered_keep_view = array_mod.ListArray{ .data = filtered_keep_nulls.array.data() };
    try std.testing.expect(!filtered_keep_view.isNull(0));
    try std.testing.expect(filtered_keep_view.isNull(1));

    var fsl_values = try makeInt32Array(allocator, &[_]?i32{ 10, 11, 20, 21, 30, 31 });
    defer fsl_values.release();
    var fsl_builder = try array_mod.FixedSizeListBuilder.init(allocator, list_field, 2);
    defer fsl_builder.deinit();
    try fsl_builder.appendValid();
    try fsl_builder.appendNull();
    try fsl_builder.appendValid();
    var fsl_array = try fsl_builder.finish(fsl_values);
    defer fsl_array.release();

    var fsl_pred = try makeBoolArray(allocator, &[_]?bool{ false, true, true });
    defer fsl_pred.release();
    var fsl_datum = Datum.fromArray(fsl_array.retain());
    defer fsl_datum.release();
    var fsl_pred_datum = Datum.fromArray(fsl_pred.retain());
    defer fsl_pred_datum.release();

    var filtered_fsl = try datumFilter(fsl_datum, fsl_pred_datum, .{ .drop_nulls = true });
    defer filtered_fsl.release();
    try std.testing.expect(filtered_fsl.isArray());
    try std.testing.expectEqual(@as(usize, 2), filtered_fsl.array.data().length);
    try filtered_fsl.array.data().validateLayout();

    const fsl_view = array_mod.FixedSizeListArray{ .data = filtered_fsl.array.data() };
    try std.testing.expect(fsl_view.isNull(0));
    try std.testing.expectEqual(@as(usize, 4), (try fsl_view.valuesRef()).data().length);
}

test "compute datumSelect supports mixed candidate datum forms" {
    const allocator = std.testing.allocator;
    const int32_array = array_mod.Int32Array;

    var a = try makeInt32Array(allocator, &[_]?i32{ 10, 20, 30 });
    defer a.release();
    var c0 = try makeInt32Array(allocator, &[_]?i32{100});
    defer c0.release();
    var c1 = try makeInt32Array(allocator, &[_]?i32{ 200, 300 });
    defer c1.release();
    var chunked = try ChunkedArray.init(allocator, .{ .int32 = {} }, &[_]ArrayRef{ c0, c1 });
    defer chunked.release();

    const values = [_]Datum{
        Datum.fromArray(a.retain()),
        Datum.fromScalar(Scalar.init(.{ .int32 = {} }, .{ .i32 = 99 })),
        Datum.fromChunked(chunked.retain()),
    };
    defer {
        var i: usize = 0;
        while (i < values.len) : (i += 1) {
            var d = values[i];
            d.release();
        }
    }

    var out = try datumSelect(&[_]usize{ 0, 1, 2 }, values[0..]);
    defer out.release();
    try std.testing.expect(out.isArray());
    try std.testing.expectEqual(@as(usize, 3), out.array.data().length);
    const view = int32_array{ .data = out.array.data() };
    try std.testing.expectEqual(@as(i32, 10), try view.value(0));
    try std.testing.expectEqual(@as(i32, 99), try view.value(1));
    try std.testing.expectEqual(@as(i32, 300), try view.value(2));
}

test "compute datumSelectNullable and datumBuildNullLikeWithAllocator support null output rows" {
    const allocator = std.testing.allocator;
    const int32_array = array_mod.Int32Array;
    const int32_type = DataType{ .int32 = {} };

    var base = try makeInt32Array(allocator, &[_]?i32{ 10, 20, 30 });
    defer base.release();

    const values = [_]Datum{
        Datum.fromArray(base.retain()),
        Datum.fromScalar(Scalar.init(.{ .int32 = {} }, .{ .i32 = 88 })),
    };
    defer {
        var i: usize = 0;
        while (i < values.len) : (i += 1) {
            var d = values[i];
            d.release();
        }
    }

    var out = try datumSelectNullable(&[_]?usize{ 0, null, 1 }, values[0..]);
    defer out.release();
    try std.testing.expect(out.isArray());
    try std.testing.expectEqual(@as(usize, 3), out.array.data().length);
    try out.array.data().validateLayout();
    const view = int32_array{ .data = out.array.data() };
    try std.testing.expectEqual(@as(i32, 10), try view.value(0));
    try std.testing.expect(view.isNull(1));
    try std.testing.expectEqual(@as(i32, 88), try view.value(2));

    var nulls = try datumBuildNullLikeWithAllocator(allocator, .{ .int32 = {} }, 3);
    defer nulls.release();
    try std.testing.expect(nulls.isArray());
    try std.testing.expectEqual(@as(usize, 3), nulls.array.data().length);
    const null_view = int32_array{ .data = nulls.array.data() };
    try std.testing.expect(null_view.isNull(0));
    try std.testing.expect(null_view.isNull(1));
    try std.testing.expect(null_view.isNull(2));

    const item_field = datatype.Field{
        .name = "item",
        .data_type = &int32_type,
        .nullable = true,
    };
    const fixed_size_list_type = DataType{
        .fixed_size_list = .{
            .list_size = 2,
            .value_field = item_field,
        },
    };
    var fsl_nulls = try datumBuildNullLikeWithAllocator(allocator, fixed_size_list_type, 3);
    defer fsl_nulls.release();
    try std.testing.expect(fsl_nulls.isArray());
    try std.testing.expectEqual(@as(usize, 3), fsl_nulls.array.data().length);
    try fsl_nulls.array.data().validateLayout();
    const fsl_view = array_mod.FixedSizeListArray{ .data = fsl_nulls.array.data() };
    try std.testing.expect(fsl_view.isNull(0));
    try std.testing.expect(fsl_view.isNull(1));
    try std.testing.expect(fsl_view.isNull(2));
    try std.testing.expectEqual(@as(usize, 6), (try fsl_view.valuesRef()).data().length);
}

test "compute datumSelectNullable emits fixed_size_list null rows without child mismatch" {
    const allocator = std.testing.allocator;
    const int32_type = DataType{ .int32 = {} };
    const int32_array = array_mod.Int32Array;
    const item_field = datatype.Field{
        .name = "item",
        .data_type = &int32_type,
        .nullable = true,
    };

    var values0 = try makeInt32Array(allocator, &[_]?i32{ 1, 2, 3, 4, 5, 6 });
    defer values0.release();
    var builder0 = try array_mod.FixedSizeListBuilder.init(allocator, item_field, 2);
    defer builder0.deinit();
    try builder0.appendValid();
    try builder0.appendValid();
    try builder0.appendValid();
    var fsl0 = try builder0.finish(values0);
    defer fsl0.release();

    var values1 = try makeInt32Array(allocator, &[_]?i32{ 10, 11, 20, 21, 30, 31 });
    defer values1.release();
    var builder1 = try array_mod.FixedSizeListBuilder.init(allocator, item_field, 2);
    defer builder1.deinit();
    try builder1.appendValid();
    try builder1.appendValid();
    try builder1.appendValid();
    var fsl1 = try builder1.finish(values1);
    defer fsl1.release();

    const values = [_]Datum{
        Datum.fromArray(fsl0.retain()),
        Datum.fromArray(fsl1.retain()),
    };
    defer {
        var i: usize = 0;
        while (i < values.len) : (i += 1) {
            var d = values[i];
            d.release();
        }
    }

    var out = try datumSelectNullable(&[_]?usize{ 0, null, 1 }, values[0..]);
    defer out.release();
    try std.testing.expect(out.isArray());
    try std.testing.expectEqual(@as(usize, 3), out.array.data().length);
    try out.array.data().validateLayout();

    const fsl_view = array_mod.FixedSizeListArray{ .data = out.array.data() };
    try std.testing.expect(!fsl_view.isNull(0));
    try std.testing.expect(fsl_view.isNull(1));
    try std.testing.expect(!fsl_view.isNull(2));
    try std.testing.expectEqual(@as(usize, 6), (try fsl_view.valuesRef()).data().length);

    const child = int32_array{ .data = (try fsl_view.valuesRef()).data() };
    try std.testing.expectEqual(@as(i32, 1), try child.value(0));
    try std.testing.expectEqual(@as(i32, 2), try child.value(1));
    try std.testing.expect(child.isNull(2));
    try std.testing.expect(child.isNull(3));
    try std.testing.expectEqual(@as(i32, 30), try child.value(4));
    try std.testing.expectEqual(@as(i32, 31), try child.value(5));
}

test "compute datumFilter emits fixed_size_list null rows when predicate nulls are kept" {
    const allocator = std.testing.allocator;
    const int32_type = DataType{ .int32 = {} };
    const int32_array = array_mod.Int32Array;
    const item_field = datatype.Field{
        .name = "item",
        .data_type = &int32_type,
        .nullable = true,
    };

    var values = try makeInt32Array(allocator, &[_]?i32{ 1, 2, 3, 4, 5, 6 });
    defer values.release();
    var builder = try array_mod.FixedSizeListBuilder.init(allocator, item_field, 2);
    defer builder.deinit();
    try builder.appendValid();
    try builder.appendValid();
    try builder.appendValid();
    var fsl = try builder.finish(values);
    defer fsl.release();

    var pred = try makeBoolArray(allocator, &[_]?bool{ true, null, false });
    defer pred.release();

    var in_datum = Datum.fromArray(fsl.retain());
    defer in_datum.release();
    var pred_datum = Datum.fromArray(pred.retain());
    defer pred_datum.release();

    var out = try datumFilter(in_datum, pred_datum, .{ .drop_nulls = false });
    defer out.release();
    try std.testing.expect(out.isArray());
    try std.testing.expectEqual(@as(usize, 2), out.array.data().length);
    try out.array.data().validateLayout();

    const fsl_view = array_mod.FixedSizeListArray{ .data = out.array.data() };
    try std.testing.expect(!fsl_view.isNull(0));
    try std.testing.expect(fsl_view.isNull(1));
    try std.testing.expectEqual(@as(usize, 4), (try fsl_view.valuesRef()).data().length);

    const child = int32_array{ .data = (try fsl_view.valuesRef()).data() };
    try std.testing.expectEqual(@as(i32, 1), try child.value(0));
    try std.testing.expectEqual(@as(i32, 2), try child.value(1));
    try std.testing.expect(child.isNull(2));
    try std.testing.expect(child.isNull(3));
}

test "compute datumTake direct gather for string and list on sliced input with null indices" {
    const allocator = std.testing.allocator;

    const words = [_]?[]const u8{ "skip", "a", null, "ccc", "", "eeeee", null, "g" };
    var sb = try array_mod.StringBuilder.init(allocator, 0, 0);
    defer sb.deinit();
    for (words) |w| if (w) |v| try sb.append(v) else try sb.appendNull();
    var full_strs = try sb.finish();
    defer full_strs.release();
    var strs = try full_strs.slice(1, words.len - 1);
    defer strs.release();

    const indices = [_]?usize{ 6, 0, null, 1, 3, 3, 5, 2, 4 };
    var taken = try datumTakeNullable(Datum.fromArray(strs), indices[0..]);
    defer taken.release();
    try taken.array.data().validateLayout();
    const out = array_mod.StringArray{ .data = taken.array.data() };
    try std.testing.expectEqual(indices.len, out.len());
    for (indices, 0..) |maybe, i| {
        const expected: ?[]const u8 = if (maybe) |idx| words[idx + 1] else null;
        try std.testing.expectEqual(expected == null, out.isNull(i));
        if (expected) |e| try std.testing.expectEqualStrings(e, try out.value(i));
    }
    try std.testing.expectError(error.InvalidInput, datumTakeNullable(Datum.fromArray(strs), &[_]?usize{7}));

    // list<int32>: rows [9], null, [], [2, null, 4], [5]; sliced to drop row 0.
    const rows = [_]?[]const ?i32{ &.{9}, null, &.{}, &.{ 2, null, 4 }, &.{5} };
    const item_type = DataType{ .int32 = {} };
    const item_field = datatype.Field{ .name = "item", .data_type = &item_type, .nullable = true };
    var lb = try array_mod.ListBuilder.init(allocator, 0, item_field);
    defer lb.deinit();
    var vb = try array_mod.Int32Builder.init(allocator, 0);
    defer vb.deinit();
    for (rows) |row| {
        const items = row orelse {
            try lb.appendNull();
            continue;
        };
        for (items) |item| if (item) |v| try vb.append(v) else try vb.appendNull();
        try lb.appendLen(items.len);
    }
    var child = try vb.finish();
    defer child.release();
    var full_list = try lb.finish(child);
    defer full_list.release();
    var lists = try full_list.slice(1, rows.len - 1);
    defer lists.release();

    const list_indices = [_]?usize{ 2, 1, null, 0, 3, 2 };
    var taken_list = try datumTakeNullable(Datum.fromArray(lists), list_indices[0..]);
    defer taken_list.release();
    try taken_list.array.data().validateLayout();
    const out_list = array_mod.ListArray{ .data = taken_list.array.data() };
    for (list_indices, 0..) |maybe, i| {
        const expected: ?[]const ?i32 = if (maybe) |idx| rows[idx + 1] else null;
        try std.testing.expectEqual(expected == null, out_list.isNull(i));
        const items = expected orelse continue;
        var value = try out_list.value(i);
        defer value.release();
        const ints = array_mod.Int32Array{ .data = value.data() };
        try std.testing.expectEqual(items.len, ints.len());
        for (items, 0..) |item, k| {
            try std.testing.expectEqual(item == null, ints.isNull(k));
            if (item) |v| try std.testing.expectEqual(v, try ints.value(k));
        }
    }
}

test "compute datumFilterSelectionIndices word path matches per-row decisions on sliced predicate" {
    const allocator = std.testing.allocator;

    var vals: [29]?bool = undefined;
    for (&vals, 0..) |*v, i| v.* = if (i % 5 == 3) null else (i * 7) % 3 != 0;
    var full = try makeBoolArray(allocator, &vals);
    defer full.release();
    var pred = try full.slice(3, 23);
    defer pred.release();

    inline for (.{ true, false }) |drop| {
        // logical_len 21 < predicate length 23: only the first 21 rows are read.
        const got = try datumFilterSelectionIndices(allocator, Datum.fromArray(pred), 21, .{ .drop_nulls = drop });
        defer allocator.free(got);
        var expected: std.ArrayList(?usize) = .empty;
        defer expected.deinit(allocator);
        for (0..21) |i| {
            if (vals[i + 3]) |keep| {
                if (keep) try expected.append(allocator, i);
            } else if (!drop) try expected.append(allocator, null);
        }
        try std.testing.expectEqualSlices(?usize, expected.items, got);
    }
    try std.testing.expectError(error.InvalidInput, datumFilterSelectionIndices(allocator, Datum.fromArray(pred), 24, .{}));
}

test "compute dense take and u32 filter kernels match the nullable path" {
    const allocator = std.testing.allocator;
    const SharedBuffer = @import("../buffer.zig").SharedBuffer;

    // int32 with nulls; 10 indices cover a full validity byte and a tail.
    var ints = try makeInt32Array(allocator, &[_]?i32{ 10, null, 30, 40, null, 60, 70, 80, 90 });
    defer ints.release();
    const idx = [_]usize{ 8, 0, 4, 4, 1, 2, 7, 3, 5, 6 };
    var nidx: [idx.len]?usize = undefined;
    for (idx, &nidx) |i, *n| n.* = i;
    var dense = try datumTake(Datum.fromArray(ints), idx[0..]);
    defer dense.release();
    var nullable = try datumTakeNullable(Datum.fromArray(ints), nidx[0..]);
    defer nullable.release();
    const dense_values = try collectInt32ValuesFromDatum(allocator, dense);
    defer allocator.free(dense_values);
    const nullable_values = try collectInt32ValuesFromDatum(allocator, nullable);
    defer allocator.free(nullable_values);
    try std.testing.expectEqualSlices(?i32, nullable_values, dense_values);
    try std.testing.expectEqual(@as(?usize, 3), dense.array.data().null_count);
    try std.testing.expectError(error.InvalidInput, datumTake(Datum.fromArray(ints), &[_]usize{ 0, 9 }));

    // Boundary change: a null string slot that holds bytes keeps them (like arrow-rs); the row stays null.
    const offsets = [_]i32{ 0, 1, 3, 6 };
    const validity = [_]u8{0b101};
    const bufs = [_]SharedBuffer{
        SharedBuffer.fromSlice(validity[0..]),
        SharedBuffer.fromSlice(std.mem.sliceAsBytes(offsets[0..])),
        SharedBuffer.fromSlice("abbccc"),
    };
    var strs = try ArrayRef.fromBorrowed(allocator, .{ .data_type = .{ .string = {} }, .length = 3, .null_count = 1, .buffers = bufs[0..] });
    defer strs.release();
    var taken = try datumTake(Datum.fromArray(strs), &[_]usize{ 1, 2, 1 });
    defer taken.release();
    try taken.array.data().validateLayout();
    const out = array_mod.StringArray{ .data = taken.array.data() };
    try std.testing.expect(out.isNull(0) and !out.isNull(1) and out.isNull(2));
    try std.testing.expectEqualStrings("bb", try out.value(0));
    try std.testing.expectEqualStrings("ccc", try out.value(1));
    try std.testing.expectEqual(@as(?usize, 2), taken.array.data().null_count);

    // A validity bitmap shorter than offset + length bits is rejected, not read past.
    const zeros: [10]i32 = @splat(0);
    const short_bufs = [_]SharedBuffer{
        SharedBuffer.fromSlice(validity[0..]),
        SharedBuffer.fromSlice(std.mem.sliceAsBytes(zeros[0..])),
        SharedBuffer.fromSlice(""),
    };
    var short = try ArrayRef.fromBorrowed(allocator, .{ .data_type = .{ .string = {} }, .length = 9, .null_count = 1, .buffers = short_bufs[0..] });
    defer short.release();
    try std.testing.expectError(error.InvalidInput, datumTake(Datum.fromArray(short), &[_]usize{0}));

    // datumFilter (dense u32 selections) equals take of the nullable selection indices.
    const words = [_]?[]const u8{ "a", null, "ccc", "", "eeeee", null, "g", "hh", "i", "jjj", "k" };
    var sb = try array_mod.StringBuilder.init(allocator, 0, 0);
    defer sb.deinit();
    for (words) |w| if (w) |v| try sb.append(v) else try sb.appendNull();
    var word_ref = try sb.finish();
    defer word_ref.release();
    var pred = try makeBoolArray(allocator, &[_]?bool{ true, true, null, false, true, true, null, true, false, true, true });
    defer pred.release();
    var filtered = try datumFilter(Datum.fromArray(word_ref), Datum.fromArray(pred), .{});
    defer filtered.release();
    const sel = try datumFilterSelectionIndices(allocator, Datum.fromArray(pred), words.len, .{});
    defer allocator.free(sel);
    var expected = try datumTakeNullable(Datum.fromArray(word_ref), sel);
    defer expected.release();
    const got = array_mod.StringArray{ .data = filtered.array.data() };
    const want = array_mod.StringArray{ .data = expected.array.data() };
    try std.testing.expectEqual(want.len(), got.len());
    for (0..want.len()) |i| {
        try std.testing.expectEqual(want.isNull(i), got.isNull(i));
        if (!want.isNull(i)) try std.testing.expectEqualStrings(try want.value(i), try got.value(i));
    }
}

const ArrayData = array_mod.ArrayData;
const concatArrayRefs = @import("../concat_array_refs.zig").concatArrayRefs;

/// Per-row oracle for the take/concat kernels: logical row `i` of `a` equals row `j` of `b`.
fn rowsEqual(a: *const ArrayData, i: usize, b: *const ArrayData, j: usize) bool {
    const dt = if (a.data_type == .extension) a.data_type.extension.storage_type.* else a.data_type;
    if (dt == .null) return b.data_type == .null; // a sliced null array has an unknown null_count
    if (a.isNull(i) != b.isNull(j)) return false;
    if (a.isNull(i)) return true;
    const ai = a.offset + i;
    const bj = b.offset + j;
    switch (dt) {
        .int32, .fixed_size_binary => {
            const w: usize = if (dt == .int32) 4 else @intCast(dt.fixed_size_binary.byte_width);
            return std.mem.eql(u8, a.buffers[1].data[ai * w ..][0..w], b.buffers[1].data[bj * w ..][0..w]);
        },
        .string => {
            const x = (array_mod.StringArray{ .data = a }).value(i) catch return false;
            const y = (array_mod.StringArray{ .data = b }).value(j) catch return false;
            return std.mem.eql(u8, x, y);
        },
        .dictionary => {
            const x: usize = @intCast((a.buffers[1].typedSlice(i32) catch return false)[ai]);
            const y: usize = @intCast((b.buffers[1].typedSlice(i32) catch return false)[bj]);
            return rowsEqual(a.dictionary.?.data(), x, b.dictionary.?.data(), y);
        },
        .list, .map, .list_view, .fixed_size_list => {
            var ra: [2]usize = undefined;
            var rb: [2]usize = undefined;
            for ([_]*const ArrayData{ a, b }, [_]usize{ ai, bj }, [_]*[2]usize{ &ra, &rb }) |d, k, r| {
                if (dt == .fixed_size_list) {
                    const size: usize = @intCast(dt.fixed_size_list.list_size);
                    r.* = .{ k * size, k * size + size };
                } else {
                    const o = d.buffers[1].typedSlice(i32) catch return false;
                    const lo: usize = @intCast(o[k]);
                    r.* = if (dt == .list_view) .{ lo, lo + @as(usize, @intCast((d.buffers[2].typedSlice(i32) catch return false)[k])) } else .{ lo, @intCast(o[k + 1]) };
                }
            }
            if (ra[1] - ra[0] != rb[1] - rb[0]) return false;
            for (0..ra[1] - ra[0]) |k| {
                if (!rowsEqual(a.children[0].data(), ra[0] + k, b.children[0].data(), rb[0] + k)) return false;
            }
            return true;
        },
        .struct_, .sparse_union => {
            if (dt == .sparse_union and a.buffers[0].data[ai] != b.buffers[0].data[bj]) return false;
            for (a.children, b.children) |ca, cb| {
                const x = if (ca.data().length == a.length) i else ai;
                const y = if (cb.data().length == b.length) j else bj;
                if (!rowsEqual(ca.data(), x, cb.data(), y)) return false;
            }
            return true;
        },
        else => return false,
    }
}

fn expectRowsMatchSource(out: *const ArrayData, src: *const ArrayData, indices: []const ?usize) !void {
    try out.validateFull();
    try std.testing.expectEqual(indices.len, out.length);
    for (indices, 0..) |maybe, i| {
        if (maybe) |idx| {
            try std.testing.expect(rowsEqual(out, i, src, idx));
        } else if (out.data_type != .sparse_union) try std.testing.expect(out.isNull(i));
    }
}

/// Six-row arrays of every layout with a take kernel, each with nulls or null children.
const NestedSources = struct {
    arrays: [13]ArrayRef,
    count: usize = 0,

    fn add(self: *NestedSources, arr: ArrayRef) void {
        self.arrays[self.count] = arr;
        self.count += 1;
    }

    fn deinit(self: *NestedSources) void {
        for (self.arrays[0..self.count]) |*arr| arr.release();
    }
};

const nested_int32_type = DataType{ .int32 = {} };
const nested_string_type = DataType{ .string = {} };
const nested_int_field = datatype.Field{ .name = "a", .data_type = &nested_int32_type, .nullable = true };
const nested_str_field = datatype.Field{ .name = "b", .data_type = &nested_string_type, .nullable = true };
const nested_pair_fields = [_]datatype.Field{ nested_int_field, nested_str_field };
const nested_union_ids = [_]i8{ 5, 7 };

fn makeStrings(allocator: std.mem.Allocator, values: []const ?[]const u8) !ArrayRef {
    var b = try array_mod.StringBuilder.init(allocator, 0, 0);
    defer b.deinit();
    for (values) |v| if (v) |s| try b.append(s) else try b.appendNull();
    return b.finish();
}

fn makePairStruct(allocator: std.mem.Allocator, valid: []const bool, ints: []const ?i32, strs: []const ?[]const u8) !ArrayRef {
    var a = try makeInt32Array(allocator, ints);
    defer a.release();
    var b = try makeStrings(allocator, strs);
    defer b.release();
    var sb = array_mod.StructBuilder.init(allocator, &nested_pair_fields);
    defer sb.deinit();
    for (valid) |v| if (v) try sb.appendValid() else try sb.appendNull();
    return sb.finish(&.{ a, b });
}

fn buildNestedSources(allocator: std.mem.Allocator) !NestedSources {
    var out: NestedSources = .{ .arrays = undefined };
    errdefer out.deinit();
    const ints = [_]?i32{ 1, 2, 3, null, 5, 6 };
    const strs = [_]?[]const u8{ "x", "yy", "ccc", "d", null, "ffffffffffffffffff" };

    // struct{a: int32, b: string}, row 1 null.
    const pair = try makePairStruct(allocator, &.{ true, false, true, true, true, true }, &ints, &strs);
    out.add(pair);

    // The same struct as rows 1..5 with children in parent coordinates (offset 1, children of 6 rows).
    var view = pair.data().*;
    view.offset = 1;
    view.length = 5;
    view.null_count = null;
    out.add(try ArrayRef.fromBorrowed(allocator, view));

    // dictionary<int32, string>.
    const dict_type = DataType{ .string = {} };
    var dict = try makeStrings(allocator, &.{ "a", "bb", "ccc" });
    defer dict.release();
    var db = try array_mod.DictionaryBuilder.init(allocator, .{ .bit_width = 32, .signed = true }, &dict_type, 6);
    defer db.deinit();
    for ([_]?i64{ 0, null, 2, 1, 0, 2 }) |v| if (v) |x| try db.appendIndex(x) else try db.appendNull();
    out.add(try db.finish(dict));

    // fixed_size_list<int32, 2>, row 2 null.
    var fvals = try makeInt32Array(allocator, &.{ 0, 1, 2, null, 4, 5, 6, 7, 8, 9, 10, 11 });
    defer fvals.release();
    var fb = try array_mod.FixedSizeListBuilder.init(allocator, nested_int_field, 2);
    defer fb.deinit();
    for ([_]bool{ true, true, false, true, true, true }) |v| if (v) try fb.appendValid() else try fb.appendNull();
    out.add(try fb.finish(fvals));

    // fixed_size_list<string, 2>: not fixed-width, so the child is taken with expanded indices.
    var fstrs = try makeStrings(allocator, &.{ "a", null, "cc", "d", "", "f", "g", "hh", null, "j", "k", "l" });
    defer fstrs.release();
    var fsb = try array_mod.FixedSizeListBuilder.init(allocator, nested_str_field, 2);
    defer fsb.deinit();
    for ([_]bool{ true, false, true, true, true, true }) |v| if (v) try fsb.appendValid() else try fsb.appendNull();
    out.add(try fsb.finish(fstrs));

    // fixed_size_binary(3).
    var bb = try array_mod.FixedSizeBinaryBuilder.init(allocator, 3, 6);
    defer bb.deinit();
    for ([_]?[]const u8{ "abc", null, "def", "ghi", "jkl", null }) |v| if (v) |s| try bb.append(s) else try bb.appendNull();
    out.add(try bb.finish());

    // extension over int32 storage.
    var ext_storage = try makeInt32Array(allocator, &ints);
    defer ext_storage.release();
    var eb = try array_mod.ExtensionBuilder.init(allocator, .{ .name = "test.ext", .storage_type = &nested_int32_type });
    defer eb.deinit();
    out.add(try eb.finish(ext_storage));

    // map<int32, string>: rows of 2, 0, null, 1, 3, 2 entries.
    var entries = try makePairStruct(allocator, &.{ true, true, true, true, true, true, true, true }, &.{ 1, 2, 3, 4, 5, 6, 7, 8 }, &.{ "k1", "k2", "k3", "k4", "k5", "k6", "k7", "k8" });
    defer entries.release();
    var mb = try array_mod.MapBuilder.init(allocator, 6, nested_int_field, nested_str_field, false);
    defer mb.deinit();
    for ([_]?usize{ 2, 0, null, 1, 3, 2 }) |v| if (v) |n| try mb.appendLen(n) else try mb.appendNull();
    out.add(try mb.finish(entries));

    // list_view<int32>.
    var lvals = try makeInt32Array(allocator, &.{ 1, 2, null, 4, 5, 6, 7 });
    defer lvals.release();
    var lb = try array_mod.ListViewBuilder.init(allocator, 6, nested_int_field);
    defer lb.deinit();
    for ([_]?usize{ 1, 2, null, 0, 3, 1 }) |v| if (v) |n| try lb.appendLen(n) else try lb.appendNull();
    out.add(try lb.finish(lvals));

    // sparse union<a: int32 = 5, b: string = 7>.
    var ub = try array_mod.SparseUnionBuilder.init(allocator, .{ .type_ids = &nested_union_ids, .fields = &nested_pair_fields, .mode = .sparse }, 6);
    defer ub.deinit();
    for ([_]i8{ 5, 7, 7, 5, 5, 7 }) |id| try ub.appendTypeId(id);
    var uints = try makeInt32Array(allocator, &ints);
    defer uints.release();
    var ustrs = try makeStrings(allocator, &strs);
    defer ustrs.release();
    out.add(try ub.finish(&.{ uints, ustrs }));

    // struct without fields: rows are validity only.
    var eb0 = array_mod.StructBuilder.init(allocator, &.{});
    defer eb0.deinit();
    for ([_]bool{ true, false, true, true, false, true }) |v| if (v) try eb0.appendValid() else try eb0.appendNull();
    out.add(try eb0.finish(&.{}));

    // null type.
    var nb = try array_mod.NullBuilder.init(allocator, 6);
    try nb.appendNulls(6);
    out.add(try nb.finish());

    // list<struct{a, b}>: nested windows through list -> struct.
    var lsb = try array_mod.ListBuilder.init(allocator, 6, .{ .name = "item", .data_type = &pair.data().data_type, .nullable = true });
    defer lsb.deinit();
    for ([_]?usize{ 2, null, 0, 3, 1, 0 }) |v| if (v) |n| try lsb.appendLen(n) else try lsb.appendNull();
    out.add(try lsb.finish(pair));
    return out;
}

test "compute datumTake nested kernels match the per-row oracle on sliced inputs with null indices" {
    const allocator = std.testing.allocator;
    var sources = try buildNestedSources(allocator);
    defer sources.deinit();

    for (sources.arrays[0..sources.count]) |full| {
        var sliced = try full.slice(1, full.data().length - 1);
        defer sliced.release();
        for ([_]ArrayRef{ full, sliced }) |src| {
            const last = src.data().length - 1;
            const indices = [_]?usize{ last, null, 0, 2, 2, 1, last - 1, null, 0 };
            var taken = try datumTakeNullable(Datum.fromArray(src), &indices);
            defer taken.release();
            try expectRowsMatchSource(taken.array.data(), src.data(), &indices);

            const dense = [_]usize{ 1, last, 0, 0, 2 };
            var taken_dense = try datumTake(Datum.fromArray(src), &dense);
            defer taken_dense.release();
            try expectRowsMatchSource(taken_dense.array.data(), src.data(), &[_]?usize{ 1, last, 0, 0, 2 });
        }
        try std.testing.expectError(error.InvalidInput, datumTakeNullable(Datum.fromArray(sliced), &[_]?usize{sliced.data().length}));
    }
}

test "concatArrayRefs N-ary nested path matches the source rows for sliced parts" {
    const allocator = std.testing.allocator;
    var sources = try buildNestedSources(allocator);
    defer sources.deinit();

    for (sources.arrays[0..sources.count]) |full| {
        const len = full.data().length;
        const cuts = [_][2]usize{ .{ 0, 2 }, .{ 2, 1 }, .{ 3, 0 }, .{ 3, len - 3 }, .{ 1, len - 1 } };
        var parts: [cuts.len]ArrayRef = undefined;
        for (cuts, &parts) |cut, *part| part.* = try full.slice(cut[0], cut[1]);
        defer for (&parts) |*part| part.release();

        var joined = concatArrayRefs(allocator, full.data().data_type, &parts) catch |err| {
            // Unions and list views have no concat (unchanged); every other layout must concat.
            const tag = full.data().data_type;
            try std.testing.expect(err == error.UnsupportedType and (tag == .sparse_union or tag == .list_view));
            continue;
        };
        defer joined.release();
        try joined.data().validateFull();
        try std.testing.expectEqual(2 * len - 1, joined.data().length);
        var row: usize = 0;
        for (cuts) |cut| {
            for (cut[0]..cut[0] + cut[1]) |src_row| {
                try std.testing.expect(rowsEqual(joined.data(), row, full.data(), src_row));
                row += 1;
            }
        }
    }
}

fn makeSequenceChunks(allocator: std.mem.Allocator, chunks: []ArrayRef, rows_per_chunk: usize) !void {
    const values = try allocator.alloc(?i32, rows_per_chunk);
    defer allocator.free(values);
    for (chunks, 0..) |*chunk, c| {
        for (values, 0..) |*v, k| {
            const row = c * rows_per_chunk + k;
            v.* = if (row % 7 == 3) null else @as(i32, @intCast(row));
        }
        chunk.* = try makeInt32Array(allocator, values);
    }
}

test "compute chunked take and filter run the array kernels per chunk and match the flat result" {
    const allocator = std.testing.allocator;
    var parts: [3]ArrayRef = undefined;
    try makeSequenceChunks(allocator, &parts, 100);
    defer for (&parts) |*p| p.release();
    var chunked = try ChunkedArray.init(allocator, .{ .int32 = {} }, &parts);
    defer chunked.release();
    var flat = try concatArrayRefs(allocator, .{ .int32 = {} }, &parts);
    defer flat.release();

    // Sorted (one run per chunk: per-chunk kernels) and scattered (concat first) indices, with nulls.
    var sorted: [120]?usize = undefined;
    var scattered: [120]?usize = undefined;
    for (&sorted, &scattered, 0..) |*s, *r, i| {
        s.* = if (i % 13 == 5) null else i * 2 + i / 50;
        r.* = if (i % 11 == 4) null else (i * 7919) % 300;
    }
    for ([_][]const ?usize{ &sorted, &scattered }) |indices| {
        var taken = try datumTakeNullable(Datum.fromChunked(chunked), indices);
        defer taken.release();
        try std.testing.expect(taken.isChunked());
        var out = try concatArrayRefs(allocator, .{ .int32 = {} }, taken.chunked.chunks());
        defer out.release();
        try expectRowsMatchSource(out.data(), flat.data(), indices);
    }

    var pred_values: [300]?bool = undefined;
    for (&pred_values, 0..) |*p, i| p.* = if (i % 5 == 0) null else i % 3 != 0;
    var pred = try makeBoolArray(allocator, &pred_values);
    defer pred.release();
    for ([_]bool{ true, false }) |drop_nulls| {
        var expected = try datumFilter(Datum.fromArray(flat), Datum.fromArray(pred), .{ .drop_nulls = drop_nulls });
        defer expected.release();
        var actual = try datumFilter(Datum.fromChunked(chunked), Datum.fromArray(pred), .{ .drop_nulls = drop_nulls });
        defer actual.release();
        try actual.array.data().validateFull();
        try std.testing.expectEqual(expected.array.data().length, actual.array.data().length);
        for (0..expected.array.data().length) |i| try std.testing.expect(rowsEqual(actual.array.data(), i, expected.array.data(), i));
    }

    // Chunks with different dictionaries do not concat: the scattered take falls back to per-run pieces.
    const dict_type = DataType{ .string = {} };
    var dicts = [_]ArrayRef{ try makeStrings(allocator, &.{ "a", "b" }), try makeStrings(allocator, &.{ "c", "d", "e" }) };
    defer for (&dicts) |*d| d.release();
    var dict_chunks: [2]ArrayRef = undefined;
    for (&dict_chunks, dicts, 0..) |*chunk, d, c| {
        var db = try array_mod.DictionaryBuilder.init(allocator, .{ .bit_width = 32, .signed = true }, &dict_type, 4);
        defer db.deinit();
        for (0..4) |k| if (k == 2) try db.appendNull() else try db.appendIndex(@intCast((k + c) % 2));
        chunk.* = try db.finish(d);
    }
    defer for (&dict_chunks) |*c| c.release();
    var dict_chunked = try ChunkedArray.init(allocator, dict_chunks[0].data().data_type, &dict_chunks);
    defer dict_chunked.release();
    const dict_indices = [_]?usize{ 7, 0, null, 5, 1, 4, 3 };
    var dict_taken = try datumTakeNullable(Datum.fromChunked(dict_chunked), &dict_indices);
    defer dict_taken.release();
    var row: usize = 0;
    for (dict_taken.chunked.chunks()) |piece| {
        try piece.data().validateFull();
        for (0..piece.data().length) |k| {
            const want = dict_indices[row] orelse {
                try std.testing.expect(piece.data().isNull(k));
                row += 1;
                continue;
            };
            try std.testing.expect(rowsEqual(piece.data(), k, dict_chunks[want / 4].data(), want % 4));
            row += 1;
        }
    }
    try std.testing.expectEqual(dict_indices.len, row);
}
