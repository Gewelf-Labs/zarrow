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
    /// Sort behavior (order, sort keys, null placement).
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
    /// NaN handling of is_null.
    null_options,
    /// Null handling and minimum count of sum, mean, min, max, min_max, any and all.
    scalar_aggregate,
    /// Which rows count does count.
    count,
    /// Null handling of min_element_wise / max_element_wise.
    element_wise_aggregate,
    /// Slice and replacement of utf8_replace_slice / binary_replace_slice.
    replace_slice,
    /// Width and padding of utf8_zero_fill.
    zero_fill,
    /// Digits and mode of round (also ceil / floor / trunc take none).
    round,
    /// Multiple and mode of round_to_multiple.
    round_to_multiple,
    /// Mode of round_binary.
    round_binary,
    /// Number of modes of mode.
    mode,
    /// Delta degrees of freedom of variance / stddev (and their hash_ forms).
    variance,
    /// Bias correction of skew / kurtosis.
    skew,
    /// Quantiles and interpolation of quantile.
    quantile,
    /// Value searched by index.
    index,
    /// Pivot of partition_nth_indices.
    partition_nth,
    /// k and sort keys of select_k_unstable.
    select_k,
    /// Order, null placement and tie breaking of rank.
    rank,
    /// Order and null placement of rank_quantile / rank_normal.
    rank_quantile,
    /// Side of search_sorted.
    search_sorted,
    /// Quantiles, compression and null handling of tdigest / hash_tdigest.
    tdigest,
    /// Keys and unexpected-key behavior of pivot_wider / hash_pivot_wider.
    pivot_wider,
    /// Null encoding of dictionary_encode.
    dictionary_encode,
    /// Output length of scatter.
    scatter,
    /// Output length and type of inverse_permutation.
    inverse_permutation,
    /// Start value and null handling of the cumulative_* functions.
    cumulative,
    /// Period of pairwise_diff.
    pairwise,
    /// Quantile limits of winsorize.
    winsorize,
    /// Length and seed of random.
    random,
    /// Pattern of extract_regex / extract_regex_span.
    extract_regex,
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

/// One sort key of sort_indices / select_k_unstable (Arrow `SortKey`): a column by position
/// (the array arguments) or by name (a record batch or table argument).
pub const SortKey = struct {
    /// Column name (record batch / table input); unused for positional arguments.
    name: []const u8 = "",
    order: SortOrder = .ascending,
    /// Where the nulls of this key go (Arrow 26 places NaN next to the nulls: between the
    /// values and the nulls; it has no separate NaN placement).
    null_placement: SortNullPlacement = .at_end,
};

/// Common options for sort-like kernels (Arrow `SortOptions` and `ArraySortOptions`).
pub const SortOptions = struct {
    /// Ordering direction (array_sort_indices, and every column when `sort_keys` is empty).
    order: SortOrder = .ascending,
    /// One key per sort column, in priority order (sort_indices); empty: every argument with
    /// `order` (an array or chunked argument; a record batch or table needs keys).
    sort_keys: []const SortKey = &.{},
    /// Null placement of every key when set (Arrow's `SortOptions.null_placement`, which
    /// overrides the keys'); null: each key's own (`at_end` without keys).
    null_placement: ?SortNullPlacement = null,
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
    /// Literal bytes (starts_with, ends_with, match_substring, count_substring, find_substring),
    /// an RE2 pattern (the `*_regex` functions) or a SQL LIKE pattern (match_like).
    pattern: []const u8,
    /// Match case-insensitively through simple case folding (Unicode for the string layouts,
    /// Latin-1 for the binary layouts), as RE2 does for Arrow.
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
    /// Literal bytes to replace (replace_substring; empty only with a `max_replacements` bound)
    /// or an RE2 pattern (replace_substring_regex, whose replacement may use `\0`..`\9`).
    pattern: []const u8,
    replacement: []const u8,
    /// Replace at most this many matches per row from the left; null replaces all.
    max_replacements: ?usize = null,
};

/// Options of split_pattern (Arrow `SplitPatternOptions`).
pub const SplitPatternOptions = struct {
    /// Literal separator (split_pattern; must not be empty) or an RE2 pattern
    /// (split_pattern_regex).
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

/// Options of is_null (Arrow `NullOptions`).
pub const NullOptions = struct {
    /// A floating-point NaN counts as null.
    nan_is_null: bool = false,
};

/// Options of sum, mean, min, max, min_max, any and all (Arrow `ScalarAggregateOptions`).
pub const ScalarAggregateOptions = struct {
    /// Skip null rows (false: any null row makes the result null; any / all use Kleene logic).
    skip_nulls: bool = true,
    /// Fewer valid rows than this give a null result.
    min_count: u32 = 1,
};

/// Rows counted by count (Arrow `CountOptions::CountMode`).
pub const CountMode = enum { only_valid, only_null, all };

/// Options of count (Arrow `CountOptions`).
pub const CountOptions = struct {
    mode: CountMode = .only_valid,
};

/// Options of min_element_wise and max_element_wise (Arrow `ElementWiseAggregateOptions`).
pub const ElementWiseAggregateOptions = struct {
    /// Ignore null operands of a row (false: a null operand makes the row null).
    skip_nulls: bool = true,
};

/// Options of utf8_replace_slice (code points) and binary_replace_slice (bytes) (Arrow
/// `ReplaceSliceOptions`): `[start, stop)` is replaced; negative positions count from the end.
pub const ReplaceSliceOptions = struct {
    start: i64,
    stop: i64,
    replacement: []const u8,
};

/// Options of utf8_zero_fill (Arrow `ZeroFillOptions`).
pub const ZeroFillOptions = struct {
    /// Output length in code points; longer rows are kept as is.
    width: usize,
    /// One code point inserted after a leading `+` or `-`.
    padding: []const u8 = "0",
};

/// Rounding mode (Arrow `RoundMode`): `down` toward -inf, `up` toward +inf, `towards_zero`,
/// `towards_infinity` (away from zero); the `half_*` modes round to the nearest and break ties
/// that way (`half_to_even` / `half_to_odd`: to the even / odd neighbour).
pub const RoundMode = enum { down, up, towards_zero, towards_infinity, half_down, half_up, half_towards_zero, half_towards_infinity, half_to_even, half_to_odd };

/// Options of round (Arrow `RoundOptions`).
pub const RoundOptions = struct {
    /// Digits after the decimal point (negative: to tens, hundreds, ...).
    ndigits: i64 = 0,
    round_mode: RoundMode = .half_to_even,
};

/// Options of round_to_multiple (Arrow `RoundToMultipleOptions`). `multiple` must be positive;
/// it is cast to the input type (safe cast).
pub const RoundToMultipleOptions = struct {
    multiple: core.Scalar = core.Scalar.init(.{ .double = {} }, .{ .f64 = 1.0 }),
    round_mode: RoundMode = .half_to_even,
};

/// Options of round_binary (Arrow `RoundBinaryOptions`); the digits are the second argument.
pub const RoundBinaryOptions = struct {
    round_mode: RoundMode = .half_to_even,
};

/// Options of mode (Arrow `ModeOptions`).
pub const ModeOptions = struct {
    /// Number of most common values to return (at least 1).
    n: i64 = 1,
    skip_nulls: bool = true,
    min_count: u32 = 0,
};

/// Options of variance / stddev (Arrow `VarianceOptions`).
pub const VarianceOptions = struct {
    /// Delta degrees of freedom: the divisor is `count - ddof` (at least 0).
    ddof: i32 = 0,
    skip_nulls: bool = true,
    min_count: u32 = 0,
};

/// Options of skew / kurtosis (Arrow `SkewOptions`).
pub const SkewOptions = struct {
    skip_nulls: bool = true,
    /// false: the sample (bias-corrected) estimator.
    biased: bool = true,
    min_count: u32 = 0,
};

/// Interpolation of quantile between two data points (Arrow `QuantileOptions::Interpolation`).
pub const QuantileInterpolation = enum { linear, lower, higher, nearest, midpoint };

/// Options of quantile (Arrow `QuantileOptions`).
pub const QuantileOptions = struct {
    /// Quantiles in [0, 1].
    q: []const f64 = &.{0.5},
    interpolation: QuantileInterpolation = .linear,
    skip_nulls: bool = true,
    min_count: u32 = 0,
};

/// Options of index (Arrow `IndexOptions`): the value to find, of the input's type.
pub const IndexOptions = struct {
    value: core.Scalar,
};

/// Options of partition_nth_indices (Arrow `PartitionNthOptions`).
pub const PartitionNthOptions = struct {
    /// The row at this sorted position is in place; at most the input length.
    pivot: usize,
    null_placement: SortNullPlacement = .at_end,
};

/// Options of select_k_unstable (Arrow `SelectKOptions`).
pub const SelectKOptions = struct {
    k: usize,
    /// One key per argument (empty: one ascending key).
    sort_keys: []const SortKey = &.{},
};

/// Tie breaking of rank (Arrow `RankOptions::Tiebreaker`).
pub const RankTiebreaker = enum {
    /// Equal values get the lowest rank of the run.
    min,
    /// Equal values get the highest rank of the run.
    max,
    /// Equal values get ranks in input order.
    first,
    /// Like min, but the ranks of the runs are consecutive.
    dense,
};

/// Options of rank (Arrow `RankOptions`).
pub const RankOptions = struct {
    order: SortOrder = .ascending,
    null_placement: SortNullPlacement = .at_end,
    tiebreaker: RankTiebreaker = .first,
};

/// Options of rank_quantile and rank_normal (Arrow `RankQuantileOptions`).
pub const RankQuantileOptions = struct {
    order: SortOrder = .ascending,
    null_placement: SortNullPlacement = .at_end,
};

/// Options of search_sorted (Arrow `SearchSortedOptions`).
pub const SearchSortedOptions = struct {
    /// `left`: the first position the needle can be inserted at (lower bound); `right`: the
    /// last (upper bound).
    side: enum { left, right } = .left,
};

/// Options of tdigest and hash_tdigest (Arrow `TDigestOptions`).
pub const TDigestOptions = struct {
    /// Quantiles, each in [0, 1] (at most 65536 of them).
    q: []const f64 = &.{0.5},
    /// Compression: about `delta` centroids (at least 10 are used; at most 2^20).
    delta: u32 = 100,
    /// Values buffered before they are merged into the digest (at least 1 is used; at most 2^24).
    buffer_size: u32 = 500,
    skip_nulls: bool = true,
    /// Fewer valid rows than this give null quantiles.
    min_count: u32 = 0,
};

/// What pivot_wider does with a key that is not in `key_names` (Arrow
/// `PivotWiderOptions::UnexpectedKeyBehavior`).
pub const UnexpectedKeyBehavior = enum { ignore, raise };

/// Options of pivot_wider and hash_pivot_wider (Arrow `PivotWiderOptions`).
pub const PivotWiderOptions = struct {
    /// The output columns, one per key value (distinct).
    key_names: []const []const u8 = &.{},
    unexpected_key_behavior: UnexpectedKeyBehavior = .ignore,
};

/// Options of dictionary_encode (Arrow `DictionaryEncodeOptions`).
pub const DictionaryEncodeOptions = struct {
    /// `mask`: a null row stays null; `encode`: nulls become one dictionary entry.
    null_encoding: enum { mask, encode } = .mask,
};

/// Options of scatter (Arrow `ScatterOptions`).
pub const ScatterOptions = struct {
    /// The largest index allowed; the output has `max_index + 1` rows (negative: as many rows as
    /// indices). Capped: more than max(indices, 2^24) output rows is `InvalidOptions`.
    max_index: i64 = -1,
};

/// Options of inverse_permutation (Arrow `InversePermutationOptions`).
pub const InversePermutationOptions = struct {
    /// As `ScatterOptions.max_index`.
    max_index: i64 = -1,
    /// A signed integer type holding the input length (null: the type of the indices).
    output_type: ?DataType = null,
};

/// Options of cumulative_sum, cumulative_prod, cumulative_min, cumulative_max (and their
/// `_checked` forms) and cumulative_mean (Arrow `CumulativeOptions`).
pub const CumulativeOptions = struct {
    /// Starting value, a numeric scalar cast (safely) to the input type (null or a null scalar:
    /// the operation's identity). cumulative_mean ignores it, as Arrow.
    start: ?core.Scalar = null,
    /// false: every row from the first null on is null; true: a null row is null and the
    /// accumulation goes on.
    skip_nulls: bool = false,
};

/// Options of pairwise_diff / pairwise_diff_checked (Arrow `PairwiseOptions`).
pub const PairwiseOptions = struct {
    /// Row `i` is `x[i] - x[i - period]` (a negative period looks ahead); rows without a
    /// partner are null.
    period: i64 = 1,
};

/// Options of winsorize (Arrow `WinsorizeOptions`): both limits in [0, 1], lower <= upper.
pub const WinsorizeOptions = struct {
    lower_limit: f64 = 0,
    upper_limit: f64 = 1,
};

/// Options of extract_regex and extract_regex_span (Arrow `ExtractRegexOptions` and
/// `ExtractRegexSpanOptions`).
pub const ExtractRegexOptions = struct {
    /// RE2 syntax; every group must be named, one struct field per group.
    pattern: []const u8,
};

/// Options of random (Arrow `RandomOptions`; the length is the batch length Arrow passes).
pub const RandomOptions = struct {
    /// Number of values.
    length: usize = 0,
    /// `seed`: a deterministic generator seeded with `seed`; `system`: `NotImplemented` (the
    /// exec context has no I/O source of randomness; draw a seed and pass it instead).
    initializer: enum { seed, system } = .seed,
    seed: u64 = 0,
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
    null_options: NullOptions,
    scalar_aggregate: ScalarAggregateOptions,
    count: CountOptions,
    element_wise_aggregate: ElementWiseAggregateOptions,
    replace_slice: ReplaceSliceOptions,
    zero_fill: ZeroFillOptions,
    round: RoundOptions,
    round_to_multiple: RoundToMultipleOptions,
    round_binary: RoundBinaryOptions,
    mode: ModeOptions,
    variance: VarianceOptions,
    skew: SkewOptions,
    quantile: QuantileOptions,
    index: IndexOptions,
    partition_nth: PartitionNthOptions,
    select_k: SelectKOptions,
    rank: RankOptions,
    rank_quantile: RankQuantileOptions,
    search_sorted: SearchSortedOptions,
    tdigest: TDigestOptions,
    pivot_wider: PivotWiderOptions,
    dictionary_encode: DictionaryEncodeOptions,
    scatter: ScatterOptions,
    inverse_permutation: InversePermutationOptions,
    cumulative: CumulativeOptions,
    pairwise: PairwiseOptions,
    winsorize: WinsorizeOptions,
    random: RandomOptions,
    extract_regex: ExtractRegexOptions,
    custom: CustomOptions,

    pub fn noneValue() Options {
        return .{ .none = {} };
    }

    pub fn tag(self: Options) OptionsTag {
        return std.meta.activeTag(self);
    }
};
