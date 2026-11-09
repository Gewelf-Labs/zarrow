# Containment: function notes

Up: [function matrix](matrix.md) · [compute support matrix](../../compute-support-matrix.md)

Arrow C++'s "Containment tests": literal substring tests (`src/compute/string.zig:107`,
`src/compute/string_transform.zig:665`) and set lookups (`src/compute/set_lookup.zig:40`). All take
array, chunked or scalar inputs with null intersection (set lookups follow `null_matching_behavior`
instead). The substring functions take a literal `MatchSubstringOptions.pattern`; `ignore_case`
(default false) matches through the simple case folding orbits, Unicode for the string layouts and
ASCII for the binary layouts, via the regex engine as Arrow does through RE2. Regex forms are in
[Regular expressions](../regular-expressions/notes.md).
Evidence: `src/compute/string_regex.zig:201`, `src/compute/test_string_regex.zig:"case-insensitive literal kernels fold through Unicode simple case folding (pyarrow 26 results)"`, `src/compute/test_string.zig:"string oracle matches a per-row loop on random sliced inputs with nulls for every layout"`, `src/compute/test_set_lookup.zig:"set lookup on chunked inputs with misaligned chunks equals the array result"`

## Summary

| Function | Arity | Input types | Output type | Options |
|---|---|---|---|---|
| [`is_in`](#is_in) | Unary | key types (see below) | bool | `SetLookupOptions` |
| [`index_in`](#index_in) | Unary | key types (see below) | int32 | `SetLookupOptions` |
| [`match_substring`](#match_substring) | Unary | string-like, binary-like | bool | `MatchSubstringOptions` |
| [`starts_with`](#starts_with) | Unary | string-like, binary-like | bool | `MatchSubstringOptions` |
| [`ends_with`](#ends_with) | Unary | string-like, binary-like | bool | `MatchSubstringOptions` |
| [`count_substring`](#count_substring) | Unary | string-like, binary-like, fixed_size_binary | int32 / int64 | `MatchSubstringOptions` |
| [`find_substring`](#find_substring) | Unary | string-like, binary-like, fixed_size_binary | int32 / int64 | `MatchSubstringOptions` |

"string-like" is string, large_string, string_view; "binary-like" is binary, large_binary,
binary_view; int64 is for the large layouts.

## is_in

Whether each row occurs in `SetLookupOptions.value_set` (an array or chunked array; a scalar is
`InvalidOptions`), hashed once per call. Key types: numbers, bool, dates, timestamps, times,
durations, intervals, decimals, fixed_size_binary and the string / binary layouts; floats compare by
bits with one NaN; decimals, fixed_size_binary and intervals compare their value bytes.

- `null_matching_behavior` (default `match`): `match`, `skip`, `emit_null`, `inconclusive`. The
  deprecated `skip_nulls` maps to `skip`.
- A value set of another type is cast to the input type (safe cast); when that fails the input is
  cast to the value set's type (pyarrow 26: an int64 value set [1, 300] against int8 rows, a float
  value set against int32 rows, a string value set against integers). No cast either way is
  `InvalidOptions`.
- A dictionary input is decoded first (implicit cast mode `decode`) and its values are looked up.
- A null-type input is all null rows looked up under `null_matching_behavior`, cast to the value
  set's type (int8 for a null-type value set): `is_in(null, value_set = nulls)` is all true, false
  with `skip`, null with `emit_null` or `inconclusive` (pyarrow 26).

Evidence: `src/compute/set_lookup.zig:37`, `src/compute/set_lookup.zig:100`, `src/compute/options.zig:308`, `src/compute/test_set_lookup.zig:"set lookup hand-written values for every null matching behavior (Arrow C++ results)"`, `src/compute/test_set_lookup.zig:"set lookup casts a value set of another type and takes decimal, fixed_size_binary and interval keys (pyarrow 26 results)"`, `src/compute/test_set_lookup.zig:"is_in and index_in on a null-type input: every null matching behavior, null-type and typed value sets, sliced, chunked and scalar (pyarrow 26 results)"`, `src/compute/test_dictionary.zig:"dictionary inputs to is_in, index_in, comparisons and sort_indices are decoded (pyarrow 26 results)"`

## index_in

As `is_in`, returning the int32 position in the value set of the first occurrence of the row's
value, or null when absent. For a null-type input with a null-containing value set and `match` every
row is 0 (pyarrow 26); `skip`, `emit_null` and `inconclusive` give null.

Evidence: `src/compute/set_lookup.zig:37`, `src/compute/set_lookup.zig:100`, `src/compute/test_set_lookup.zig:"set lookup oracle matches a linear search on random sliced inputs with nulls"`, `src/compute/test_set_lookup.zig:"set lookup on scalars, empty value sets, empty and all-null inputs"`, `src/compute/test_set_lookup.zig:"set lookup errors: options, value set type, unsupported types and malformed input"`

## match_substring

True when the literal pattern occurs anywhere in the row. Registered on the string and binary layouts
(as Arrow, plus the view layouts).

Evidence: `src/compute/string.zig:80`, `src/compute/string.zig:873`, `src/compute/test_string.zig:"string hand-written values: case mapping, trims, slices, reverse, lengths, predicates and join on every string layout"`, `src/compute/test_string_regex.zig:"case-insensitive literal kernels fold through Unicode simple case folding (pyarrow 26 results)"`

## starts_with

True when the row begins with the literal pattern. Same layouts and `ignore_case` handling as
`match_substring`.

Evidence: `src/compute/string.zig:78`, `src/compute/string.zig:873`, `src/compute/test_string.zig:"string hand-written values: case mapping, trims, slices, reverse, lengths, predicates and join on every string layout"`

## ends_with

True when the row ends with the literal pattern. Same layouts and `ignore_case` handling as
`match_substring`.

Evidence: `src/compute/string.zig:79`, `src/compute/string.zig:873`, `src/compute/test_string.zig:"string hand-written values: case mapping, trims, slices, reverse, lengths, predicates and join on every string layout"`

## count_substring

Number of non-overlapping occurrences of the literal pattern (int32, int64 for the large layouts). An
empty pattern matches at every position including the end: byte length + 1, as Arrow. Takes the
string and binary layouts and fixed_size_binary (int32, `ignore_case` too, as Arrow).

Evidence: `src/compute/string_transform.zig:69`, `src/compute/string_transform.zig:665`, `src/compute/string_transform.zig:1006`, `src/compute/test_string_transform.zig:"string transform hand-written values: replace, split, pad, count, find, slice, repeat, reverse, case on every layout"`, `src/compute/test_string_transform.zig:"string transform fixed_size_binary: binary_length, binary_slice, binary_replace_slice, count_substring, find_substring on nulls, sliced, chunked and scalar inputs (pyarrow 26 results)"`

## find_substring

Byte index of the first occurrence of the literal pattern, -1 when absent (int32, int64 for the large
layouts). Same layouts as `count_substring`, fixed_size_binary included.

Evidence: `src/compute/string_transform.zig:70`, `src/compute/string_transform.zig:665`, `src/compute/string_transform.zig:1010`, `src/compute/test_string_transform.zig:"string transform oracle matches a per-row loop on random sliced inputs with nulls for every layout"`, `src/compute/test_string_transform.zig:"string transform fixed_size_binary: binary_length, binary_slice, binary_replace_slice, count_substring, find_substring on nulls, sliced, chunked and scalar inputs (pyarrow 26 results)"`
