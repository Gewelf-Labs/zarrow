# Sorts and partitions: function notes

Up: [function matrix](matrix.md) · [compute support matrix](../../compute-support-matrix.md)

Arrow C++'s "Sorts and partitions" vector functions, registered by `src/compute/sort.zig:73`.
Shared rules: key types (`isSortable`, `src/compute/sort.zig:127`) are numbers, temporal types and
decimals by value (-0.0 equals 0.0), bool, string / binary layouts (views included, which Arrow 26
does not sort) and fixed_size_binary by bytes, dictionaries by decoded value (validated indices;
chunked dictionary columns too, which Arrow 26 does not sort), and struct columns by their fields in
order, a null struct row null in every field (Arrow flattens a struct key; a struct array argument
sorts by all its fields, where Arrow wants keys naming them). Arrow 26 has no separate NaN placement
and neither do the options: NaN sits between the values and the nulls (`at_end`: values, NaNs,
nulls; `at_start`: nulls, NaNs, values). Ties keep input order (stable). At most 2^32 - 1 rows
(`InvalidInput` past it). Index results are uint64. Algorithms: one radix-able column (integer,
temporal, decimal32 / decimal64, bool, float) sorts with an LSD radix sort; other columns and several
keys use comparison sorts broken by row index. Evidence: `src/compute/test_sort.zig:"sort oracle: random sliced inputs with nulls give a stable sorted permutation for every type and both orders"`, `src/compute/test_tabular.zig:"struct sort keys sort by their fields, a null struct row null in every field (pyarrow 26 results)"`.

## Summary

| Function | Arity | Input types | Output type | Options |
|---|---|---|---|---|
| [`sort_indices`](#sort_indices) | Varargs | sortable arrays / chunked, or one record batch / table | uint64 | `SortOptions` |
| [`array_sort_indices`](#array_sort_indices) | Unary | one sortable array or chunked | uint64 | `SortOptions` |
| [`rank`](#rank) | Unary | sortable | uint64 | `RankOptions` |
| [`rank_normal`](#rank_normal) | Unary | sortable | float64 | `RankQuantileOptions` |
| [`rank_quantile`](#rank_quantile) | Unary | sortable | float64 | `RankQuantileOptions` |
| [`partition_nth_indices`](#partition_nth_indices) | Unary | sortable or null type | uint64 | `PartitionNthOptions` |
| [`select_k_unstable`](#select_k_unstable) | Varargs | sortable arrays / chunked, or one record batch / table | uint64 | `SelectKOptions` |
| [`search_sorted`](#search_sorted) | Binary | ascending sortable values; needles of the same type | uint64 | `SearchSortedOptions` |

## sort_indices

The permutation that sorts the input, stable. Array or chunked columns are compared in argument
order; one record batch or table is sorted by the columns `SortOptions.sort_keys` names (at least
one key, else `InvalidOptions`, as Arrow). Each `SortKey` has `name`, `order` (default ascending)
and `null_placement` (default at_end); `SortOptions.null_placement`, when set, overrides every
key's (Arrow 26). Without keys every argument uses `SortOptions.order`. Also sorts float16 columns,
which pyarrow 26 cannot sort. Null-type columns tie every row (alone they give 0 .. n-1, as pyarrow
26; another key decides).

Evidence: `src/compute/sort.zig:74`, `src/compute/sort.zig:913`, `src/compute/sort.zig:1137`, `src/compute/options.zig:172`, `src/compute/test_sort.zig:"sort_indices, array_sort_indices and rank hand-written values: NaN, nulls and both orders (pyarrow 26 results)"`, `src/compute/test_sort.zig:"sort_indices on several keys, record batches and tables; select_k_unstable and partition_nth_indices (pyarrow 26 results)"`, `src/compute/test_tabular.zig:"sort_indices and select_k_unstable on record batches and tables with per-key null placement (pyarrow 26 results)"`, `src/compute/test_sort.zig:"sort_indices and array_sort_indices of the null type: every row ties, 0 .. n-1; sliced, chunked, with another key (pyarrow 26 results)"`, `src/compute/test_sort.zig:"sort oracle: random sliced inputs with nulls give a stable sorted permutation for every type and both orders"`

## array_sort_indices

As `sort_indices` for a single array or chunked array, ordered by `SortOptions.order` with the
placement of `SortOptions.null_placement` (at_end when unset). Null-type input gives 0 .. n-1.

Evidence: `src/compute/sort.zig:78`, `src/compute/sort.zig:913`, `src/compute/test_sort.zig:"sort_indices, array_sort_indices and rank hand-written values: NaN, nulls and both orders (pyarrow 26 results)"`, `src/compute/test_sort.zig:"sort_indices and array_sort_indices of the null type: every row ties, 0 .. n-1; sliced, chunked, with another key (pyarrow 26 results)"`, `src/compute/test_sort.zig:"sort oracle: random sliced inputs with nulls give a stable sorted permutation for every type and both orders"`

## rank

The 1-based rank of each row. `RankOptions`: `order` (ascending), `null_placement` (at_end) and
`tiebreaker` (default `first`): `min` and `max` give equal values the lowest or highest rank of
their run, `first` ranks ties in input order, `dense` is like `min` with consecutive run ranks.

Evidence: `src/compute/sort.zig:90`, `src/compute/sort.zig:958`, `src/compute/test_sort.zig:"sort_indices, array_sort_indices and rank hand-written values: NaN, nulls and both orders (pyarrow 26 results)"`, `src/compute/test_sort.zig:"sort errors: options, types, scalars, lengths and malformed input"`

## rank_normal

`rank_quantile` mapped through the normal quantile function (Wichura AS 241,
`src/compute/sort.zig:1018`); within 1e-12 of pyarrow 26. Same options and null handling as
`rank_quantile`.

Evidence: `src/compute/sort.zig:94`, `src/compute/sort.zig:1018`, `src/compute/test_sort.zig:"rank_quantile, rank_normal and search_sorted hand-written values (pyarrow 26 results)"`

## rank_quantile

(rows before + half the equal rows) / rows, as float64. Nulls are counted and tied with each other.
`RankQuantileOptions`: `order` (ascending), `null_placement` (at_end).

Evidence: `src/compute/sort.zig:94`, `src/compute/sort.zig:988`, `src/compute/options.zig:456`, `src/compute/test_sort.zig:"rank_quantile, rank_normal and search_sorted hand-written values (pyarrow 26 results)"`

## partition_nth_indices

A permutation whose row at sorted position `pivot` is in place, with no larger row before it and no
smaller one after. `PartitionNthOptions.pivot` (required, at most the length, else
`InvalidOptions`) and `null_placement` (at_end). A null-type column gives 0 .. n-1 for any pivot,
as pyarrow 26. One radix column partitions with an MSD radix selection over packed (key, row)
integers and one branch-free pass (O(n)); other columns use quickselect (O(n) average, a sort of the
remaining range past 2 log2(n) rounds).

Evidence: `src/compute/sort.zig:82`, `src/compute/sort.zig:924`, `src/compute/sort.zig:713`, `src/compute/test_sort.zig:"sort_indices on several keys, record batches and tables; select_k_unstable and partition_nth_indices (pyarrow 26 results)"`, `src/compute/test_sort.zig:"partition_nth_indices and select_k_unstable oracle: selections agree with the stable sort for radix and comparison columns, both orders and null placements"`, `src/compute/test_sort.zig:"partition_nth_indices of the null type: 0 .. n-1 for any pivot; sliced, chunked and empty (pyarrow 26 results)"`

## select_k_unstable

The indices of the first `k` rows in sort order. `SelectKOptions`: `k` (required) and `sort_keys`
(empty: one ascending key), also naming columns of a record batch or table. zarrow returns the
stable sort's first k rows (Arrow's order among ties is unspecified). One radix column uses a
bounded max-heap of packed keys (O(n log k)); other inputs the same bounded heap with comparisons.

Evidence: `src/compute/sort.zig:86`, `src/compute/sort.zig:666`, `src/compute/sort.zig:812`, `src/compute/test_sort.zig:"sort_indices on several keys, record batches and tables; select_k_unstable and partition_nth_indices (pyarrow 26 results)"`, `src/compute/test_sort.zig:"partition_nth_indices and select_k_unstable oracle: selections agree with the stable sort for radix and comparison columns, both orders and null placements"`, `src/compute/test_tabular.zig:"sort_indices and select_k_unstable on record batches and tables with per-key null placement (pyarrow 26 results)"`

## search_sorted

Insertion positions of the needles among the non-null values of the ascending values.
`SearchSortedOptions.side` `left` (default, lower bound) or `right` (upper bound). A null needle
gives null; a scalar needle gives a scalar. Values with a null between two values are
`InvalidInput` (Arrow leaves the position unspecified).

Evidence: `src/compute/sort.zig:100`, `src/compute/sort.zig:1075`, `src/compute/options.zig:462`, `src/compute/test_sort.zig:"rank_quantile, rank_normal and search_sorted hand-written values (pyarrow 26 results)"`, `src/compute/test_sort.zig:"sort errors: options, types, scalars, lengths and malformed input"`
