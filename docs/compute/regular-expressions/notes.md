# Regular expressions: function notes

Up: [function matrix](matrix.md) · [compute support matrix](../../compute-support-matrix.md)

Arrow C++'s regex functions, registered in `src/compute/string_regex.zig:63` over zarrow's own
linear-time engine (`src/regex/`: parser, Thompson NFA, lazy DFA forward for the match end and over a
reverse program for the start, Pike VM for groups; no backtracking) with RE2 syntax and leftmost-first
semantics. All eight take string, large_string, binary and large_binary (as Arrow) plus string_view
and binary_view (not registered by Arrow); string layouts are UTF-8 (a character is a code point),
binary layouts Latin-1 (a character is a byte). Array, chunked and scalar inputs, null intersection;
every input chunk passes `validateLayout` and a pattern that does not compile is `InvalidOptions`.
Checked against pyarrow 26 on a generated corpus of 33 605 cases (`tools/regex_corpus.py`) with no
unexplained difference.
Evidence: `src/compute/string_regex.zig:129`, `src/regex/regex.zig:36`, `src/compute/test_string_regex.zig:"string regex kernels match pyarrow 26 on the generated corpus (tools/regex_corpus.py)"`, `src/compute/test_string_regex.zig:"string regex sliced and chunked inputs with nulls equal the per-row results"`, `src/compute/test_string_regex.zig:"string regex scalars, empty and all-null inputs"`

Pattern syntax (shared by every function): literals, `.`, classes with ranges and negation, POSIX
`[:alpha:]` classes, `\d \s \w` and negations (ASCII, as RE2), `\pL` / `\p{Lu}` / `\PL` / `\p{^L}`
general categories and script classes such as `\p{Greek}` (Unicode 17.0.0), `^ $ \A \z`, `\b \B`,
groups, `(?:)`, `(?P<name>)` / `(?<name>)`, alternation, greedy and lazy `* + ? {n} {n,} {n,m}`
(n, m <= 1000, nested counts multiplied <= 1000), flags `(?imsU)` / `(?i-s:...)`, escapes `\x41
\x{263a} \123 \Q..\E \C`. Backreferences and lookaround are `InvalidOptions` (RE2 rejects them too).
`(?i)` and `ignore_case` fold through the simple case folding orbits (equal to every RE2 orbit of
pyarrow 26; Latin-1 mode folds ASCII only); the case pairs added in Unicode 16 / 17 fold here and not
in pyarrow 26's RE2. Compile limits: 64 KiB pattern, nesting 200 groups, 100 000 program
instructions, 2 000 000 instructions x capture slots; the DFA cache holds 4096 states / 4 MiB, then
the Pike VM takes over. A result Arrow would emit with invalid UTF-8 (a match inside a code point
through `\C` or an empty `\B` match) is `InvalidInput` for a string row.
Evidence: `src/regex/parse.zig:90`, `src/regex/prog.zig:62`, `src/regex/test_regex.zig:"regex literals, classes, anchors and quantifiers (RE2 results)"`, `src/regex/test_regex.zig:"regex UTF-8, Latin-1, Unicode classes and case folding (RE2 results)"`, `src/regex/test_regex.zig:"regex errors: syntax, unsupported constructs and limits"`, `src/regex/test_regex.zig:"regex matching is linear: no blow-up on patterns that backtrack exponentially"`, `src/regex/test_regex.zig:"regex case folding orbits equal RE2's (pyarrow 26, tools/regex_fold_check.py)"`

## Summary

| Function | Arity | Input types | Output type | Options |
|---|---|---|---|---|
| [`match_substring_regex`](#match_substring_regex) | Unary | string-like, binary-like | bool | `MatchSubstringOptions` |
| [`match_like`](#match_like) | Unary | string-like, binary-like | bool | `MatchSubstringOptions` |
| [`count_substring_regex`](#count_substring_regex) | Unary | string-like, binary-like, fixed_size_binary | int32 / int64 | `MatchSubstringOptions` |
| [`find_substring_regex`](#find_substring_regex) | Unary | string-like, binary-like, fixed_size_binary | int32 / int64 | `MatchSubstringOptions` |
| [`replace_substring_regex`](#replace_substring_regex) | Unary | string-like, binary-like | input type | `ReplaceSubstringOptions` |
| [`split_pattern_regex`](#split_pattern_regex) | Unary | string-like, binary-like | `list<T>` | `SplitPatternOptions` |
| [`extract_regex`](#extract_regex) | Unary | string-like, binary-like | struct of T | `ExtractRegexOptions` |
| [`extract_regex_span`](#extract_regex_span) | Unary | string-like, binary-like | struct of `fixed_size_list<int32, 2>` | `ExtractRegexOptions` |

"string-like" is string, large_string, string_view; "binary-like" is binary, large_binary,
binary_view; int64 and the int64 span lists are for the large layouts.

## match_substring_regex

True when the pattern (`MatchSubstringOptions.pattern`, `ignore_case` default false) matches anywhere
in the row (unanchored search).

Evidence: `src/compute/string_regex.zig:65`, `src/compute/string_regex.zig:150`, `src/compute/test_string_regex.zig:"string regex hand-written values on every layout: match, like, count, find, replace, split, extract (pyarrow 26 results)"`

## match_like

SQL LIKE: the pattern is translated as Arrow does (`%` any run, `_` one character, `\` escapes the
next one) and matched against the whole row; `ignore_case` applies. As in Arrow, case-sensitive
patterns of the forms `%x%`, `x%`, `%x` run as plain substring / prefix / suffix tests, so a
backslash there is literal.

Evidence: `src/compute/string_regex.zig:66`, `src/compute/string_regex.zig:222`, `src/compute/test_string_regex.zig:"string regex hand-written values on every layout: match, like, count, find, replace, split, extract (pyarrow 26 results)"`

## count_substring_regex

Number of matches per row (int32, int64 for the large layouts). As Arrow, the row is consumed as RE2
`FindAndConsume`: each search sees only the rest of the row (no left context, so `^` and `\b` can
match again after a match), and an empty match counts and advances one byte. Also takes
fixed_size_binary (Latin-1 rows, int32 result, as Arrow).

Evidence: `src/compute/string_regex.zig:67`, `src/compute/string_regex.zig:279`, `src/compute/string_regex.zig:79`, `src/compute/test_string_regex.zig:"count_substring_regex and find_substring_regex on fixed_size_binary: nulls, sliced, chunked and scalars (pyarrow 26 results)"`, `src/compute/test_string_regex.zig:"string regex hand-written values on every layout: match, like, count, find, replace, split, extract (pyarrow 26 results)"`

## find_substring_regex

Byte index of the start of the first match, -1 when there is none (int32, int64 for the large
layouts). Also takes fixed_size_binary (int32), as Arrow.

Evidence: `src/compute/string_regex.zig:68`, `src/compute/string_regex.zig:292`, `src/compute/test_string_regex.zig:"count_substring_regex and find_substring_regex on fixed_size_binary: nulls, sliced, chunked and scalars (pyarrow 26 results)"`

## replace_substring_regex

`ReplaceSubstringOptions` (`pattern`, `replacement`, `max_replacements` default unbounded).

- Without a bound (or on an empty row) it is RE2 `GlobalReplace`: `\0`..`\9` and `\\` in the
  replacement, which is checked against the group count (`InvalidOptions`); an empty match right
  after the previous match is skipped (one code point in UTF-8 mode).
- With `max_replacements` the row is consumed as `FindAndConsume` and each match is replaced by
  matching the pattern again inside it (RE2 `Replace`, as Arrow); when that fails (`\Ba` on the
  isolated match) the result is `InvalidInput`, as Arrow's "replacement failed".
- A valid string row whose result would not be UTF-8 is `InvalidInput` (Arrow emits the bytes).

Evidence: `src/compute/string_regex.zig:69`, `src/compute/string_regex.zig:454`, `src/compute/string_regex.zig:478`, `src/compute/test_string_regex.zig:"string regex errors: patterns, replacements, unnamed groups, reverse and empty separators, types, malformed input"`, `src/compute/test_string_regex.zig:"string regex hand-written values on every layout: match, like, count, find, replace, split, extract (pyarrow 26 results)"`

## split_pattern_regex

Splits each row at the matches into a `list<T>` of the input layout (`SplitPatternOptions`:
`max_splits` default unbounded). `reverse = true` is `NotImplemented`, as Arrow. Deviations: an empty
separator match without `max_splits` makes the row `InvalidInput` (Arrow loops forever), and a
separator that cuts a code point of a string row is `InvalidInput` (Arrow would emit invalid UTF-8).

Evidence: `src/compute/string_regex.zig:70`, `src/compute/string_regex.zig:510`, `src/compute/test_string_regex.zig:"string regex errors: patterns, replacements, unnamed groups, reverse and empty separators, types, malformed input"`, `src/compute/test_string_regex.zig:"string regex hand-written values on every layout: match, like, count, find, replace, split, extract (pyarrow 26 results)"`

## extract_regex

`ExtractRegexOptions.pattern` must have only named groups (`InvalidOptions` otherwise). The result is
a struct with one field of the input layout per group; a row without a match is a null struct row,
and a group that did not take part is "" (as Arrow).

Evidence: `src/compute/string_regex.zig:72`, `src/compute/string_regex.zig:558`, `src/compute/test_string_regex.zig:"string regex hand-written values on every layout: match, like, count, find, replace, split, extract (pyarrow 26 results)"`, `src/compute/test_string_regex.zig:"string regex errors: patterns, replacements, unnamed groups, reverse and empty separators, types, malformed input"`

## extract_regex_span

As `extract_regex`, but each group field is `fixed_size_list<int32, 2>` holding [start, length] in
bytes (int64 for the large layouts), null for a group that did not take part.

Evidence: `src/compute/string_regex.zig:72`, `src/compute/string_regex.zig:558`, `src/compute/test_string_regex.zig:"string regex kernels match pyarrow 26 on the generated corpus (tools/regex_corpus.py)"`
