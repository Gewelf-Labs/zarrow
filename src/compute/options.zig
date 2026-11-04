const std = @import("std");
const core = @import("core.zig");

const DataType = core.DataType;

/// Well-known options payload tags used by compute kernels.
pub const OptionsTag = enum {
    /// Kernel does not require options.
    none,
    /// Type conversion / cast behavior.
    cast,
    /// Arithmetic behavior (overflow, divide-by-zero, etc.).
    arithmetic,
    /// Filter behavior.
    filter,
    /// Sort behavior (order, null placement, NaN handling, stability).
    sort,
    /// day_of_week numbering.
    day_of_week,
    /// Substring pattern of starts_with / ends_with / match_substring.
    match_substring,
    /// Code point slice of utf8_slice_codeunits.
    slice,
    /// Null handling of binary_join_element_wise.
    join,
    /// Characters removed by ascii_trim / ascii_ltrim / ascii_rtrim.
    trim,
    /// Escape hatch for downstream custom kernels.
    custom,
};

/// Common options for cast-like kernels.
pub const CastOptions = struct {
    /// If true, conversion must fail on lossy/overflowing cast.
    safe: bool = true,
    /// Optional target type override used by cast kernels.
    to_type: ?DataType = null,
};

/// Common options for arithmetic kernels.
pub const ArithmeticOptions = struct {
    /// If true, overflow should produce an error instead of wrapping.
    check_overflow: bool = true,
    /// If true, division by zero should return an error.
    divide_by_zero_is_error: bool = true,
};

/// Common options for filter-like kernels.
pub const FilterOptions = struct {
    /// If true, nulls in predicate are treated as false and dropped.
    drop_nulls: bool = true,
};

/// Sort direction for sort-like kernels.
pub const SortOrder = enum {
    ascending,
    descending,
};

/// Null placement strategy for sort-like kernels.
pub const SortNullPlacement = enum {
    at_start,
    at_end,
};

/// NaN placement strategy for floating-point sort-like kernels.
pub const SortNaNPlacement = enum {
    at_start,
    at_end,
};

/// Common options for sort-like kernels.
pub const SortOptions = struct {
    /// Ordering direction.
    order: SortOrder = .ascending,
    /// Null placement behavior.
    null_placement: SortNullPlacement = .at_end,
    /// Optional NaN placement policy for floating-point inputs.
    nan_placement: ?SortNaNPlacement = null,
    /// If true, equal-key ordering should be stable.
    stable: bool = false,
};

/// Options of day_of_week (Arrow `DayOfWeekOptions`).
pub const DayOfWeekOptions = struct {
    /// Number days from 0 (true) or from 1 (false).
    count_from_zero: bool = true,
    /// First day of the week: 1 = Monday ... 7 = Sunday (ISO numbering).
    week_start: u32 = 1,
};

/// Options of starts_with, ends_with and match_substring (Arrow `MatchSubstringOptions`).
pub const MatchSubstringOptions = struct {
    /// Literal bytes to look for (no regex).
    pattern: []const u8,
    /// Compare ASCII letters case-insensitively (other bytes compare exactly).
    ignore_case: bool = false,
};

/// Options of utf8_slice_codeunits (Arrow `SliceOptions`): Python slice semantics over code
/// points; negative `start` / `stop` count from the end.
pub const SliceOptions = struct {
    start: i64,
    stop: i64 = std.math.maxInt(i64),
    /// Must not be 0.
    step: i64 = 1,
};

/// What binary_join_element_wise does with a null value (Arrow `JoinOptions::NullHandlingBehavior`).
pub const JoinNullHandling = enum {
    /// The row is null.
    emit_null,
    /// Null values are left out of the row.
    skip,
    /// Null values are replaced by `null_replacement`.
    replace,
};

/// Options of binary_join_element_wise (Arrow `JoinOptions`).
pub const JoinOptions = struct {
    null_handling: JoinNullHandling = .emit_null,
    null_replacement: []const u8 = "",
};

/// Options of ascii_trim, ascii_ltrim and ascii_rtrim (Arrow `TrimOptions`).
pub const TrimOptions = struct {
    /// Bytes to remove (each byte is one character; non-ASCII bytes are rejected).
    characters: []const u8,
};

/// Custom untyped options hook for downstream extension kernels.
pub const CustomOptions = struct {
    /// Caller-defined discriminator.
    tag: []const u8,
    /// Optional opaque payload owned by caller.
    payload: ?*const anyopaque = null,
};

/// Type-safe options payload passed to all kernel signatures and executors.
pub const Options = union(OptionsTag) {
    none: void,
    cast: CastOptions,
    arithmetic: ArithmeticOptions,
    filter: FilterOptions,
    sort: SortOptions,
    day_of_week: DayOfWeekOptions,
    match_substring: MatchSubstringOptions,
    slice: SliceOptions,
    join: JoinOptions,
    trim: TrimOptions,
    custom: CustomOptions,

    pub fn noneValue() Options {
        return .{ .none = {} };
    }

    pub fn tag(self: Options) OptionsTag {
        return std.meta.activeTag(self);
    }
};
