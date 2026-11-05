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
    /// Pattern and replacement of replace_substring.
    replace_substring,
    /// Separator and limits of split_pattern.
    split_pattern,
    /// Limits of ascii_split_whitespace / utf8_split_whitespace.
    split,
    /// Width and padding of the ascii_* / utf8_* pad functions.
    pad,
    /// Multiple and unit of floor_temporal / ceil_temporal / round_temporal.
    round_temporal,
    /// Week numbering of week.
    week,
    /// Value set of is_in / index_in.
    set_lookup,
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

/// Options of replace_substring (Arrow `ReplaceSubstringOptions`).
pub const ReplaceSubstringOptions = struct {
    /// Literal bytes to replace (no regex). Empty only with a `max_replacements` bound.
    pattern: []const u8,
    replacement: []const u8,
    /// Replace at most this many matches per row from the left; null replaces all.
    max_replacements: ?usize = null,
};

/// Options of split_pattern (Arrow `SplitPatternOptions`).
pub const SplitPatternOptions = struct {
    /// Literal separator; must not be empty.
    pattern: []const u8,
    /// Split at most this many times per row; null splits at every separator.
    max_splits: ?usize = null,
    /// Count `max_splits` from the end of the row.
    reverse: bool = false,
};

/// Options of ascii_split_whitespace and utf8_split_whitespace (Arrow `SplitOptions`).
pub const SplitOptions = struct {
    max_splits: ?usize = null,
    reverse: bool = false,
};

/// Options of ascii_lpad / rpad / center and utf8_lpad / rpad / center (Arrow `PadOptions`).
pub const PadOptions = struct {
    /// Output length in bytes (ascii_*) or code points (utf8_*); longer rows are kept as is.
    width: usize,
    /// One ASCII byte (ascii_*) or one code point (utf8_*).
    padding: []const u8 = " ",
    /// center with an odd amount of padding puts the extra character on the right.
    lean_left_on_odd_padding: bool = true,
};

/// Rounding unit of the *_temporal functions (Arrow `CalendarUnit`).
pub const CalendarUnit = enum { nanosecond, microsecond, millisecond, second, minute, hour, day, week, month, quarter, year };

/// Options of floor_temporal, ceil_temporal and round_temporal (Arrow `RoundTemporalOptions`).
pub const RoundTemporalOptions = struct {
    /// Number of units to round to; must be > 0.
    multiple: i64 = 1,
    unit: CalendarUnit = .day,
    /// Weeks start on Monday (true) or Sunday (false).
    week_starts_monday: bool = true,
    /// ceil_temporal of a value already on a boundary moves to the next boundary.
    ceil_is_strictly_greater: bool = false,
    /// Count multiples from the start of the next larger unit instead of from 1970-01-01.
    calendar_based_origin: bool = false,
};

/// Options of week (Arrow `WeekOptions`; iso_week and us_week are fixed instances).
pub const WeekOptions = struct {
    week_starts_monday: bool = true,
    /// Days before week 1 of their year are week 0 (instead of the last week of the previous year).
    count_from_zero: bool = false,
    /// Week 1 starts on the first week start day of the year (instead of holding its 4th day).
    first_week_is_fully_in_year: bool = false,
};

/// How is_in / index_in treat nulls (Arrow `SetLookupOptions::NullMatchingBehavior`).
pub const NullMatchingBehavior = enum {
    /// A null input matches a null in the value set.
    match,
    /// A null input never matches (Arrow's `skip_nulls = true`).
    skip,
    /// A null input gives a null output.
    emit_null,
    /// Like emit_null; a value not found is also null when the value set holds a null.
    inconclusive,
};

/// Options of is_in and index_in (Arrow `SetLookupOptions`). `value_set` is borrowed: an array or
/// chunked array of the input's type, kept alive by the caller for the call.
pub const SetLookupOptions = struct {
    value_set: core.Datum,
    null_matching_behavior: NullMatchingBehavior = .match,
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
    replace_substring: ReplaceSubstringOptions,
    split_pattern: SplitPatternOptions,
    split: SplitOptions,
    pad: PadOptions,
    round_temporal: RoundTemporalOptions,
    week: WeekOptions,
    set_lookup: SetLookupOptions,
    custom: CustomOptions,

    pub fn noneValue() Options {
        return .{ .none = {} };
    }

    pub fn tag(self: Options) OptionsTag {
        return std.meta.activeTag(self);
    }
};
