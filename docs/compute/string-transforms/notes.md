# String transforms: function notes

Up: [function matrix](matrix.md) · [compute support matrix](../../compute-support-matrix.md)

Arrow C++'s "String transforms", "String padding", "String trimming", "String splitting", "String
slicing" and "String joining" categories, spread over `src/compute/string.zig:107`,
`src/compute/string_transform.zig:64` and `src/compute/string_normalize.zig:35`. All functions take
array, chunked or scalar inputs, use null intersection (except the two join functions, see below),
keep the input layout for a string or binary result (lengths and positions are int32, int64 for the
large layouts), and the `utf8_*` functions reject a non-null row that is not UTF-8 with
`InvalidInput`. For a string layout, a pattern, replacement or padding in the options must be UTF-8
(else `InvalidOptions`), so a valid row stays valid. Regex forms are in
[Regular expressions](../regular-expressions/notes.md); count_substring and find_substring in
[Containment](../containment/notes.md).
Evidence: `src/compute/string_transform.zig:278`, `src/compute/test_string_transform.zig:"string transform oracle matches a per-row loop on random sliced inputs with nulls for every layout"`, `src/compute/test_string_transform.zig:"string transform chunked inputs with misaligned chunks give a chunked result equal to the array result"`, `src/compute/test_string_transform.zig:"string transform scalars, empty and all-null inputs"`

## Summary

| Function | Arity | Input types | Output type | Options |
|---|---|---|---|---|
| [`ascii_capitalize`](#ascii_capitalize) | Unary | string-like | input type | none |
| [`ascii_lower`](#ascii_lower) | Unary | string-like | input type | none |
| [`ascii_reverse`](#ascii_reverse) | Unary | string-like | input type | none |
| [`ascii_swapcase`](#ascii_swapcase) | Unary | string-like | input type | none |
| [`ascii_title`](#ascii_title) | Unary | string-like | input type | none |
| [`ascii_upper`](#ascii_upper) | Unary | string-like | input type | none |
| [`utf8_capitalize`](#utf8_capitalize) | Unary | string-like | input type | none |
| [`utf8_lower`](#utf8_lower) | Unary | string-like | input type | none |
| [`utf8_reverse`](#utf8_reverse) | Unary | string-like | input type | none |
| [`utf8_swapcase`](#utf8_swapcase) | Unary | string-like | input type | none |
| [`utf8_title`](#utf8_title) | Unary | string-like | input type | none |
| [`utf8_upper`](#utf8_upper) | Unary | string-like | input type | none |
| [`utf8_normalize`](#utf8_normalize) | Unary | string-like | input type | `Utf8NormalizeOptions` |
| [`binary_length`](#binary_length) | Unary | string-like, binary-like, fixed_size_binary | int32 / int64 | none |
| [`utf8_length`](#utf8_length) | Unary | string-like | int32 / int64 | none |
| [`binary_repeat`](#binary_repeat) | Binary | string-like or binary-like, integer | input type | none |
| [`binary_reverse`](#binary_reverse) | Unary | binary-like | input type | none |
| [`binary_replace_slice`](#binary_replace_slice) | Unary | string-like, binary-like, fixed_size_binary | input type | `ReplaceSliceOptions` |
| [`utf8_replace_slice`](#utf8_replace_slice) | Unary | string-like | input type | `ReplaceSliceOptions` |
| [`replace_substring`](#replace_substring) | Unary | string-like, binary-like | input type | `ReplaceSubstringOptions` |
| [`binary_slice`](#binary_slice) | Unary | binary-like, fixed_size_binary | input type | `SliceOptions` |
| [`utf8_slice_codeunits`](#utf8_slice_codeunits) | Unary | string-like | input type | `SliceOptions` |
| [`utf8_zero_fill`](#utf8_zero_fill) | Unary | string-like | input type | `ZeroFillOptions` |
| [`ascii_center`](#ascii_center) | Unary | string-like | input type | `PadOptions` |
| [`ascii_lpad`](#ascii_lpad) | Unary | string-like | input type | `PadOptions` |
| [`ascii_rpad`](#ascii_rpad) | Unary | string-like | input type | `PadOptions` |
| [`utf8_center`](#utf8_center) | Unary | string-like | input type | `PadOptions` |
| [`utf8_lpad`](#utf8_lpad) | Unary | string-like | input type | `PadOptions` |
| [`utf8_rpad`](#utf8_rpad) | Unary | string-like | input type | `PadOptions` |
| [`ascii_ltrim`](#ascii_ltrim) | Unary | string-like | input type | `TrimOptions` |
| [`ascii_rtrim`](#ascii_rtrim) | Unary | string-like | input type | `TrimOptions` |
| [`ascii_trim`](#ascii_trim) | Unary | string-like | input type | `TrimOptions` |
| [`ascii_ltrim_whitespace`](#ascii_ltrim_whitespace) | Unary | string-like | input type | none |
| [`ascii_rtrim_whitespace`](#ascii_rtrim_whitespace) | Unary | string-like | input type | none |
| [`ascii_trim_whitespace`](#ascii_trim_whitespace) | Unary | string-like | input type | none |
| [`utf8_ltrim`](#utf8_ltrim) | Unary | string-like | input type | `TrimOptions` |
| [`utf8_rtrim`](#utf8_rtrim) | Unary | string-like | input type | `TrimOptions` |
| [`utf8_trim`](#utf8_trim) | Unary | string-like | input type | `TrimOptions` |
| [`utf8_ltrim_whitespace`](#utf8_ltrim_whitespace) | Unary | string-like | input type | none |
| [`utf8_rtrim_whitespace`](#utf8_rtrim_whitespace) | Unary | string-like | input type | none |
| [`utf8_trim_whitespace`](#utf8_trim_whitespace) | Unary | string-like | input type | none |
| [`ascii_split_whitespace`](#ascii_split_whitespace) | Unary | string-like | `list<T>` | `SplitOptions` |
| [`utf8_split_whitespace`](#utf8_split_whitespace) | Unary | string-like | `list<T>` | `SplitOptions` |
| [`split_pattern`](#split_pattern) | Unary | string-like, binary-like | `list<T>` | `SplitPatternOptions` |
| [`binary_join`](#binary_join) | Binary | list-like of T, T | T | none |
| [`binary_join_element_wise`](#binary_join_element_wise) | Varargs | string-like, binary, large_binary | input type | `JoinOptions` |

"string-like" is string, large_string, string_view; "binary-like" is binary, large_binary,
binary_view; `T` is the input layout.

## ascii_upper

Maps ASCII letters to uppercase; every other byte is copied. Shares the ASCII case kernel with
`ascii_lower` and `ascii_swapcase` (SIMD byte mapping, no UTF-8 validation).

Evidence: `src/compute/string.zig:49`, `src/compute/string.zig:464`, `src/compute/test_string.zig:"string hand-written values: case mapping, trims, slices, reverse, lengths, predicates and join on every string layout"`

## ascii_lower

As `ascii_upper`, to lowercase.

Evidence: `src/compute/string.zig:50`, `src/compute/string.zig:464`, `src/compute/test_string.zig:"string hand-written values: case mapping, trims, slices, reverse, lengths, predicates and join on every string layout"`

## ascii_swapcase

As `ascii_upper`, swapping the case of each ASCII letter.

Evidence: `src/compute/string.zig:64`, `src/compute/string.zig:464`, `src/compute/test_string.zig:"string hand-written values: case mapping, trims, slices, reverse, lengths, predicates and join on every string layout"`

## ascii_capitalize

Uppercases the first byte if it is an ASCII letter and lowercases the other ASCII letters.

Evidence: `src/compute/string.zig:66`, `src/compute/string.zig:464`, `src/compute/test_string.zig:"string Unicode case mapping and predicates on every case class (Arrow C++ results)"`

## ascii_title

Uppercases each ASCII letter that follows a non-letter and lowercases the others.

Evidence: `src/compute/string.zig:68`, `src/compute/string.zig:464`, `src/compute/test_string.zig:"string Unicode case mapping and predicates on every case class (Arrow C++ results)"`

## ascii_reverse

Reverses the bytes of the row. A non-null row with a byte >= 0x80 is `InvalidInput`, as in Arrow.

Evidence: `src/compute/string.zig:70`, `src/compute/string.zig:629`, `src/compute/test_string.zig:"string errors: invalid UTF-8 rows, options, types, arity and malformed input"`, `src/compute/test_string.zig:"string hand-written values: case mapping, trims, slices, reverse, lengths, predicates and join on every string layout"`

## utf8_upper

Maps every code point through its simple (one to one) uppercase mapping, Unicode 17.0.0 tables
generated from UnicodeData.txt by `tools/gen_unicode_tables.py`. U+00DF uppercases to U+1E9E, as in
Arrow's utf8proc; no special casing (no U+00DF to SS, final sigma or ligature expansion), as Arrow. A row
may change its byte length (at most 3/2 of it). Checked against pyarrow 26 over every code point.
Not UTF-8: `InvalidInput`.

Evidence: `src/compute/string.zig:51`, `src/compute/string.zig:717`, `src/compute/test_string.zig:"string Unicode case mapping and predicates on every case class (Arrow C++ results)"`

## utf8_lower

As `utf8_upper`, through the simple lowercase mapping.

Evidence: `src/compute/string.zig:52`, `src/compute/string.zig:717`, `src/compute/test_string.zig:"string Unicode case mapping and predicates on every case class (Arrow C++ results)"`

## utf8_swapcase

As `utf8_upper`, lowercasing uppercase code points and uppercasing lowercase ones.

Evidence: `src/compute/string.zig:65`, `src/compute/string.zig:717`, `src/compute/test_string.zig:"string Unicode case mapping and predicates on every case class (Arrow C++ results)"`

## utf8_capitalize

As `utf8_upper` for the first code point, lowercase for the rest (simple mappings).

Evidence: `src/compute/string.zig:67`, `src/compute/string.zig:717`, `src/compute/test_string.zig:"string Unicode case mapping and predicates on every case class (Arrow C++ results)"`

## utf8_title

Uppercases each cased code point that follows an uncased one and lowercases the others (simple
mappings, as Arrow).

Evidence: `src/compute/string.zig:69`, `src/compute/string.zig:717`, `src/compute/test_string.zig:"string Unicode case mapping and predicates on every case class (Arrow C++ results)"`

## utf8_reverse

Reverses the code points of the row (not grapheme clusters, as Arrow). Not UTF-8: `InvalidInput`.

Evidence: `src/compute/string.zig:63`, `src/compute/string.zig:642`, `src/compute/test_string.zig:"string hand-written values: case mapping, trims, slices, reverse, lengths, predicates and join on every string layout"`

## utf8_normalize

Unicode normalization (UAX #15) with `Utf8NormalizeOptions.form`: `nfc` (default), `nfd`, `nfkc`,
`nfkd`, over Unicode 17.0.0 decomposition, composition and combining class tables (generated with
CompositionExclusions.txt; Hangul algorithmically). A row that already passes the quick check of its
form is copied, and an input whose rows all pass (always for ASCII) comes back without a copy. A row
that is not UTF-8 is `InvalidInput`; offsets past the output layout are `Overflow`. Equal to Python's
`unicodedata` and to pyarrow 26 on every code point in every form and on 4,000 mixed sequences.
string_view is also registered (pyarrow 26 has no string_view kernel).

Evidence: `src/compute/string_normalize.zig:37`, `src/compute/string_normalize.zig:148`, `src/compute/test_string_normalize.zig:"utf8_normalize matches Python unicodedata and pyarrow 26 on every code point and on mixed sequences (tools/normalize_corpus.py)"`, `src/compute/test_string_normalize.zig:"utf8_normalize on nulls, sliced, chunked and scalar inputs, every string layout, unchanged inputs and invalid UTF-8 (pyarrow 26 results)"`

## binary_length

Byte length of each row: int32, int64 for large_string and large_binary. Takes the string, binary
and fixed_size_binary layouts (the last gives the width, int32, as Arrow).

Evidence: `src/compute/string.zig:125`, `src/compute/string_transform.zig:1168`, `src/compute/test_string.zig:"string hand-written values: case mapping, trims, slices, reverse, lengths, predicates and join on every string layout"`, `src/compute/test_string_transform.zig:"string transform fixed_size_binary: binary_length, binary_slice, binary_replace_slice, count_substring, find_substring on nulls, sliced, chunked and scalar inputs (pyarrow 26 results)"`

## utf8_length

Number of code points of each row (int32, int64 for large_string). Like Arrow it counts the bytes
that do not continue a UTF-8 sequence and does not validate, so an invalid row is not an error.

Evidence: `src/compute/string.zig:116`, `src/compute/string.zig:969`, `src/compute/test_string.zig:"string hand-written values: case mapping, trims, slices, reverse, lengths, predicates and join on every string layout"`

## binary_repeat

Repeats each row `count` times. The count argument is an array or scalar of any integer type, cast
to int64; a negative count in a valid row is `InvalidInput` (Arrow: "Repeat count must be a
non-negative integer"), a result too large for the layout is an error too. A dictionary input is
decoded first (implicit cast mode `decode`), so dictionary<string> gives string, as pyarrow 26. Takes
the string and binary layouts; nulls in either argument give null.

Evidence: `src/compute/string_transform.zig:71`, `src/compute/string_transform.zig:577`, `src/compute/test_string_transform.zig:"binary_repeat with counts of every integer type: nulls, negative counts, sliced, chunked, scalars and dictionaries (pyarrow 26 results)"`, `src/compute/test_string_transform.zig:"string transform errors: options, invalid UTF-8 rows, negative and huge repeat counts, types and malformed input"`

## binary_reverse

Reverses the bytes of each row; binary layouts only (as Arrow).

Evidence: `src/compute/string_transform.zig:103`, `src/compute/string_transform.zig:278`, `src/compute/test_string_transform.zig:"string transform hand-written values: replace, split, pad, count, find, slice, repeat, reverse, case on every layout"`

## binary_replace_slice

Replaces the bytes `[start, stop)` of each row with `ReplaceSliceOptions.replacement` (start, stop
required). A negative position counts from the end, both are clamped to the row, and a stop before
the start inserts at the start (Arrow C++). Registered on the binary layouts, the string layouts (as
Arrow) and fixed_size_binary (the result is a fixed_size_binary of the width the options give a row
of the input width, as pyarrow 26).

Deviations: on a string layout the replacement must be UTF-8, and a row whose result is not UTF-8 (a
split code point) is `InvalidInput`, where Arrow writes the invalid row. A fixed_size_binary result
width of 0 is `UnsupportedType`, where pyarrow gives fixed_size_binary[0] (zarrow's `validateLayout`
needs a positive width).

Evidence: `src/compute/string_transform.zig:98`, `src/compute/string_transform.zig:410`, `src/compute/string_transform.zig:1018`, `src/compute/test_string_transform.zig:"string transform binary_join on list_view and view layouts, binary_replace_slice on the string layouts (pyarrow 26 results)"`, `src/compute/test_string_transform.zig:"string transform fixed_size_binary: binary_length, binary_slice, binary_replace_slice, count_substring, find_substring on nulls, sliced, chunked and scalar inputs (pyarrow 26 results)"`

## utf8_replace_slice

As `binary_replace_slice`, with positions in code points; string layouts only. The replacement must
be UTF-8; a row that is not UTF-8 is `InvalidInput`.

Evidence: `src/compute/string_transform.zig:79`, `src/compute/string_transform.zig:410`, `src/compute/test_string_transform.zig:"string transform replace_slice, zero_fill and binary_join on lists (Arrow C++ results)"`

## replace_substring

Replaces non-overlapping occurrences of the literal `ReplaceSubstringOptions.pattern` with
`replacement`, at most `max_replacements` (default unbounded). An empty pattern needs a bound
(Arrow loops forever without one), else `InvalidOptions`. Takes the string and binary layouts; for a
string layout pattern and replacement must be UTF-8. The offsets layouts are processed over the whole
value window of a chunk at once.

Evidence: `src/compute/string_transform.zig:67`, `src/compute/string_transform.zig:504`, `src/compute/test_string_transform.zig:"string transform hand-written values: replace, split, pad, count, find, slice, repeat, reverse, case on every layout"`, `src/compute/test_string_transform.zig:"string transform errors: options, invalid UTF-8 rows, negative and huge repeat counts, types and malformed input"`

## binary_slice

Python slice semantics over bytes (`SliceOptions`: `start` required, `stop` default end, `step`
default 1, a step of 0 is `InvalidOptions`). Binary layouts (as Arrow) and fixed_size_binary (result
width from the options, 0 is `UnsupportedType` as for `binary_replace_slice`).

Evidence: `src/compute/string_transform.zig:102`, `src/compute/string.zig:804`, `src/compute/string_transform.zig:1014`, `src/compute/test_string_transform.zig:"string transform fixed_size_binary: binary_length, binary_slice, binary_replace_slice, count_substring, find_substring on nulls, sliced, chunked and scalar inputs (pyarrow 26 results)"`

## utf8_slice_codeunits

Python slice semantics over code points (`SliceOptions`, same defaults and step check as
`binary_slice`). Not UTF-8: `InvalidInput`.

Evidence: `src/compute/string.zig:62`, `src/compute/string.zig:804`, `src/compute/test_string.zig:"string hand-written values: case mapping, trims, slices, reverse, lengths, predicates and join on every string layout"`

## utf8_zero_fill

Left-pads each row to `ZeroFillOptions.width` code points with `padding` (default `"0"`, exactly one
code point), inserted after a leading `+` or `-`. A width past `maxInt(i32)` is `InvalidOptions`.

Evidence: `src/compute/string_transform.zig:80`, `src/compute/string_transform.zig:204`, `src/compute/test_string_transform.zig:"string transform replace_slice, zero_fill and binary_join on lists (Arrow C++ results)"`

## ascii_center

Pads both sides to `PadOptions.width` bytes with `padding` (default `" "`, exactly one ASCII byte);
with odd padding the extra byte goes right when `lean_left_on_odd_padding` (default true), as Arrow.
A row at least `width` long is unchanged. A width past `maxInt(i32)` or other padding is
`InvalidOptions`. The three ascii pad functions share one kernel.

Evidence: `src/compute/string_transform.zig:61`, `src/compute/string_transform.zig:230`, `src/compute/test_string_transform.zig:"string transform hand-written values: replace, split, pad, count, find, slice, repeat, reverse, case on every layout"`

## ascii_lpad

As `ascii_center`, padding on the left only.

Evidence: `src/compute/string_transform.zig:61`, `src/compute/string_transform.zig:230`, `src/compute/test_string_transform.zig:"string transform hand-written values: replace, split, pad, count, find, slice, repeat, reverse, case on every layout"`

## ascii_rpad

As `ascii_center`, padding on the right only.

Evidence: `src/compute/string_transform.zig:61`, `src/compute/string_transform.zig:230`, `src/compute/test_string_transform.zig:"string transform hand-written values: replace, split, pad, count, find, slice, repeat, reverse, case on every layout"`

## utf8_center

As `ascii_center` with widths in code points and a padding of exactly one UTF-8 code point. Not
UTF-8: `InvalidInput`.

Evidence: `src/compute/string_transform.zig:61`, `src/compute/string_transform.zig:230`, `src/compute/test_string_transform.zig:"string transform hand-written values: replace, split, pad, count, find, slice, repeat, reverse, case on every layout"`

## utf8_lpad

As `utf8_center`, padding on the left only.

Evidence: `src/compute/string_transform.zig:61`, `src/compute/string_transform.zig:230`, `src/compute/test_string_transform.zig:"string transform hand-written values: replace, split, pad, count, find, slice, repeat, reverse, case on every layout"`

## utf8_rpad

As `utf8_center`, padding on the right only.

Evidence: `src/compute/string_transform.zig:61`, `src/compute/string_transform.zig:230`, `src/compute/test_string_transform.zig:"string transform hand-written values: replace, split, pad, count, find, slice, repeat, reverse, case on every layout"`

## ascii_trim

Removes the bytes of `TrimOptions.characters` (required) from both ends. Every character must be
ASCII (a byte >= 0x80 could split a code point), else `InvalidOptions`.

Evidence: `src/compute/string.zig:53`, `src/compute/string.zig:157`, `src/compute/test_string.zig:"string hand-written values: case mapping, trims, slices, reverse, lengths, predicates and join on every string layout"`, `src/compute/test_string.zig:"string errors: invalid UTF-8 rows, options, types, arity and malformed input"`

## ascii_ltrim

As `ascii_trim`, left end only.

Evidence: `src/compute/string.zig:54`, `src/compute/string.zig:157`, `src/compute/test_string.zig:"string hand-written values: case mapping, trims, slices, reverse, lengths, predicates and join on every string layout"`

## ascii_rtrim

As `ascii_trim`, right end only.

Evidence: `src/compute/string.zig:55`, `src/compute/string.zig:157`, `src/compute/test_string.zig:"string hand-written values: case mapping, trims, slices, reverse, lengths, predicates and join on every string layout"`

## ascii_trim_whitespace

Removes ASCII whitespace bytes from both ends; no options.

Evidence: `src/compute/string.zig:56`, `src/compute/string.zig:464`, `src/compute/test_string.zig:"string hand-written values: case mapping, trims, slices, reverse, lengths, predicates and join on every string layout"`

## ascii_ltrim_whitespace

As `ascii_trim_whitespace`, left end only.

Evidence: `src/compute/string.zig:57`, `src/compute/string.zig:464`, `src/compute/test_string.zig:"string hand-written values: case mapping, trims, slices, reverse, lengths, predicates and join on every string layout"`

## ascii_rtrim_whitespace

As `ascii_trim_whitespace`, right end only.

Evidence: `src/compute/string.zig:58`, `src/compute/string.zig:464`, `src/compute/test_string.zig:"string hand-written values: case mapping, trims, slices, reverse, lengths, predicates and join on every string layout"`

## utf8_trim

Removes the code points of `TrimOptions.characters` (required, must be UTF-8, each code point one
character) from both ends. Not UTF-8: `InvalidInput`.

Evidence: `src/compute/string.zig:71`, `src/compute/string.zig:157`, `src/compute/test_string.zig:"string hand-written values: case mapping, trims, slices, reverse, lengths, predicates and join on every string layout"`

## utf8_ltrim

As `utf8_trim`, left end only.

Evidence: `src/compute/string.zig:72`, `src/compute/string.zig:157`, `src/compute/test_string.zig:"string hand-written values: case mapping, trims, slices, reverse, lengths, predicates and join on every string layout"`

## utf8_rtrim

As `utf8_trim`, right end only.

Evidence: `src/compute/string.zig:73`, `src/compute/string.zig:157`, `src/compute/test_string.zig:"string hand-written values: case mapping, trims, slices, reverse, lengths, predicates and join on every string layout"`

## utf8_trim_whitespace

Removes Unicode White_Space code points plus U+001C..U+001F (the utf8proc set Arrow uses) from both
ends. Not UTF-8: `InvalidInput`.

Evidence: `src/compute/string.zig:59`, `src/compute/string.zig:245`, `src/compute/test_string.zig:"string Unicode case mapping and predicates on every case class (Arrow C++ results)"`

## utf8_ltrim_whitespace

As `utf8_trim_whitespace`, left end only.

Evidence: `src/compute/string.zig:60`, `src/compute/string.zig:245`, `src/compute/test_string.zig:"string hand-written values: case mapping, trims, slices, reverse, lengths, predicates and join on every string layout"`

## utf8_rtrim_whitespace

As `utf8_trim_whitespace`, right end only.

Evidence: `src/compute/string.zig:61`, `src/compute/string.zig:245`, `src/compute/test_string.zig:"string hand-written values: case mapping, trims, slices, reverse, lengths, predicates and join on every string layout"`

## ascii_split_whitespace

Splits each row on runs of ASCII whitespace into a `list<T>` of the input layout. `SplitOptions`
(or no options): `max_splits` (default unbounded), `reverse` (default false, splits from the end when
bounded). Leading and trailing separators give empty parts, as Arrow.

Evidence: `src/compute/string_transform.zig:81`, `src/compute/string_transform.zig:725`, `src/compute/test_string_transform.zig:"string transform hand-written values: replace, split, pad, count, find, slice, repeat, reverse, case on every layout"`

## utf8_split_whitespace

As `ascii_split_whitespace`, splitting on Unicode White_Space plus U+001C..U+001F. Not UTF-8:
`InvalidInput`.

Evidence: `src/compute/string_transform.zig:82`, `src/compute/string_transform.zig:725`, `src/compute/test_string_transform.zig:"string transform hand-written values: replace, split, pad, count, find, slice, repeat, reverse, case on every layout"`

## split_pattern

Splits each row on the literal `SplitPatternOptions.pattern` into a `list<T>` (`max_splits` default
unbounded, `reverse` default false). An empty pattern is `InvalidOptions` (Arrow: "Empty
separator"); for a string layout the pattern must be UTF-8. Takes the string and binary layouts,
string_view included (which Arrow does not register).

Evidence: `src/compute/string_transform.zig:68`, `src/compute/string_transform.zig:170`, `src/compute/test_string_transform.zig:"string transform hand-written values: replace, split, pad, count, find, slice, repeat, reverse, case on every layout"`, `src/compute/test_string_transform.zig:"string transform errors: options, invalid UTF-8 rows, negative and huge repeat counts, types and malformed input"`

## binary_join

Joins the items of each list row with a separator (array or scalar) of the item type T. The list may
be `list`, `large_list`, `list_view` or `large_list_view` and T any string or binary layout, views
included (a superset of Arrow's list / large_list of string, large_string, binary, large_binary);
list_view rows may be out of order or overlap. A null list, a null separator or a null item gives a
null row, as in Arrow. No options.

Evidence: `src/compute/string_transform.zig:90`, `src/compute/string_transform.zig:419`, `src/compute/string_transform.zig:441`, `src/compute/test_string_transform.zig:"string transform replace_slice, zero_fill and binary_join on lists (Arrow C++ results)"`, `src/compute/test_string_transform.zig:"string transform binary_join on list_view and view layouts, binary_replace_slice on the string layouts (pyarrow 26 results)"`

## binary_join_element_wise

Joins one or more value arguments row by row with the last argument as separator (array, chunked or
scalar). `JoinOptions.null_handling`: `emit_null` (default; also with no options), `skip`, or
`replace` with `null_replacement` (default ""). On string, large_string and string_view the
replacement must be UTF-8; on binary and large_binary any bytes are allowed (as Arrow; binary_view is
not registered).

Evidence: `src/compute/string.zig:117`, `src/compute/string.zig:1011`, `src/compute/test_string.zig:"binary_join_element_wise on binary and large_binary: nulls, null handling, sliced, chunked and scalar separators (pyarrow 26 results)"`, `src/compute/test_string.zig:"string hand-written values: case mapping, trims, slices, reverse, lengths, predicates and join on every string layout"`
