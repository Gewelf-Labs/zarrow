# String predicates: function notes

Up: [function matrix](matrix.md) · [compute support matrix](../../compute-support-matrix.md)

Arrow C++'s "String predicates": unary functions from string, large_string or string_view to bool,
registered once per layout in `src/compute/string.zig:107`. Every predicate takes array, chunked or
scalar inputs (a chunked input gives a chunked result), keeps null rows null (intersection) and has
no options. The `ascii_*` predicates look at bytes (a byte >= 0x80 is no letter, digit, space or
printable character); the `utf8_*` predicates decode code points over the Unicode 17.0.0 tables of
`src/compute/unicode_tables.zig:14` and reject a non-null row that is not UTF-8 with `InvalidInput`
(Arrow stops at the first failing code point and may not report a later bad byte). Each utf8_*
predicate was checked against pyarrow 26 over every code point and on mixed rows.
Evidence: `src/compute/string.zig:873`, `src/compute/test_string.zig:"string chunked inputs with misaligned chunks give a chunked result equal to the array result"`, `src/compute/test_string.zig:"string scalars, empty and all-null inputs"`, `src/compute/test_string.zig:"string errors: invalid UTF-8 rows, options, types, arity and malformed input"`

## Summary

| Function | Arity | Input types | Output type | Options |
|---|---|---|---|---|
| [`string_is_ascii`](#string_is_ascii) | Unary | string-like | bool | none |
| [`ascii_is_alnum`](#ascii_is_alnum) | Unary | string-like | bool | none |
| [`ascii_is_alpha`](#ascii_is_alpha) | Unary | string-like | bool | none |
| [`ascii_is_decimal`](#ascii_is_decimal) | Unary | string-like | bool | none |
| [`ascii_is_lower`](#ascii_is_lower) | Unary | string-like | bool | none |
| [`ascii_is_printable`](#ascii_is_printable) | Unary | string-like | bool | none |
| [`ascii_is_space`](#ascii_is_space) | Unary | string-like | bool | none |
| [`ascii_is_title`](#ascii_is_title) | Unary | string-like | bool | none |
| [`ascii_is_upper`](#ascii_is_upper) | Unary | string-like | bool | none |
| [`utf8_is_alnum`](#utf8_is_alnum) | Unary | string-like | bool | none |
| [`utf8_is_alpha`](#utf8_is_alpha) | Unary | string-like | bool | none |
| [`utf8_is_decimal`](#utf8_is_decimal) | Unary | string-like | bool | none |
| [`utf8_is_digit`](#utf8_is_digit) | Unary | string-like | bool | none |
| [`utf8_is_lower`](#utf8_is_lower) | Unary | string-like | bool | none |
| [`utf8_is_numeric`](#utf8_is_numeric) | Unary | string-like | bool | none |
| [`utf8_is_printable`](#utf8_is_printable) | Unary | string-like | bool | none |
| [`utf8_is_space`](#utf8_is_space) | Unary | string-like | bool | none |
| [`utf8_is_title`](#utf8_is_title) | Unary | string-like | bool | none |
| [`utf8_is_upper`](#utf8_is_upper) | Unary | string-like | bool | none |

"string-like" is string, large_string and string_view (no binary layouts).

## string_is_ascii

True when every byte of the row is below 0x80; an empty row is true. Never validates UTF-8, so it
never fails on a row.

Evidence: `src/compute/string.zig:81`, `src/compute/string.zig:228`, `src/compute/test_string.zig:"string hand-written values: case mapping, trims, slices, reverse, lengths, predicates and join on every string layout"`, `src/compute/test_string.zig:"string oracle matches a per-row loop on random sliced inputs with nulls for every layout"`

## ascii_is_alnum

True for a non-empty row whose bytes are all ASCII letters or digits; an empty row is false.

Evidence: `src/compute/string.zig:82`, `src/compute/string.zig:873`, `src/compute/test_string.zig:"string Unicode case mapping and predicates on every case class (Arrow C++ results)"`

## ascii_is_alpha

True for a non-empty row of ASCII letters only; an empty row is false.

Evidence: `src/compute/string.zig:83`, `src/compute/string.zig:873`, `src/compute/test_string.zig:"string Unicode case mapping and predicates on every case class (Arrow C++ results)"`

## ascii_is_decimal

True for a non-empty row of ASCII digits `0`..`9` only; an empty row is false.

Evidence: `src/compute/string.zig:84`, `src/compute/string.zig:873`, `src/compute/test_string.zig:"string Unicode case mapping and predicates on every case class (Arrow C++ results)"`

## ascii_is_lower

True when the row has at least one ASCII letter and every ASCII letter is lowercase; other bytes
(digits, punctuation, bytes >= 0x80) are uncased and ignored. A row without letters is false.

Evidence: `src/compute/string.zig:85`, `src/compute/string.zig:852`, `src/compute/test_string.zig:"string Unicode case mapping and predicates on every case class (Arrow C++ results)"`

## ascii_is_printable

True when every byte is in 0x20..0x7E. An empty row is true (the only ascii_* predicate that is,
as in Arrow).

Evidence: `src/compute/string.zig:86`, `src/compute/string.zig:873`, `src/compute/test_string.zig:"string Unicode case mapping and predicates on every case class (Arrow C++ results)"`

## ascii_is_space

True for a non-empty row of ASCII whitespace bytes only (Zig's `std.ascii.isWhitespace`: space, tab,
newline, vertical tab, form feed, carriage return); an empty row is false.

Evidence: `src/compute/string.zig:87`, `src/compute/string.zig:873`, `src/compute/test_string.zig:"string Unicode case mapping and predicates on every case class (Arrow C++ results)"`

## ascii_is_title

Python `istitle` rules over ASCII letters: at least one letter, every uppercase letter follows an
uncased byte and every lowercase letter follows a cased one.

Evidence: `src/compute/string.zig:88`, `src/compute/string.zig:852`, `src/compute/test_string.zig:"string Unicode case mapping and predicates on every case class (Arrow C++ results)"`

## ascii_is_upper

True when the row has at least one ASCII letter and every ASCII letter is uppercase.

Evidence: `src/compute/string.zig:89`, `src/compute/string.zig:852`, `src/compute/test_string.zig:"string Unicode case mapping and predicates on every case class (Arrow C++ results)"`

## utf8_is_alnum

True for a non-empty row whose code points all have a general category L* or N*. An empty row is
false; a row that is not UTF-8 is `InvalidInput`.

Evidence: `src/compute/string.zig:90`, `src/compute/string.zig:404`, `src/compute/test_string.zig:"string Unicode case mapping and predicates on every case class (Arrow C++ results)"`, `src/compute/test_string.zig:"unicode tables: bucketed lookups equal a sequential scan for every code point"`

## utf8_is_alpha

True for a non-empty row whose code points are all L* (letters). Empty: false.

Evidence: `src/compute/string.zig:91`, `src/compute/string.zig:404`, `src/compute/test_string.zig:"string Unicode case mapping and predicates on every case class (Arrow C++ results)"`

## utf8_is_decimal

True for a non-empty row whose code points are all Nd (decimal digits). Empty: false.

Evidence: `src/compute/string.zig:92`, `src/compute/string.zig:404`, `src/compute/test_string.zig:"string Unicode case mapping and predicates on every case class (Arrow C++ results)"`

## utf8_is_digit

True for a non-empty row whose code points are all Nd or No (Arrow's definition, so superscript
digits count). Empty: false.

Evidence: `src/compute/string.zig:93`, `src/compute/string.zig:404`, `src/compute/test_string.zig:"string Unicode case mapping and predicates on every case class (Arrow C++ results)"`

## utf8_is_lower

True when the row has at least one cased code point (Lu, Ll, Lt or one with a case mapping) and
every cased code point is lowercase. Not UTF-8: `InvalidInput`.

Evidence: `src/compute/string.zig:94`, `src/compute/string.zig:852`, `src/compute/string.zig:380`, `src/compute/test_string.zig:"string Unicode case mapping and predicates on every case class (Arrow C++ results)"`

## utf8_is_numeric

True for a non-empty row whose code points are all Nd, Nl or No. Empty: false.

Evidence: `src/compute/string.zig:95`, `src/compute/string.zig:404`, `src/compute/test_string.zig:"string Unicode case mapping and predicates on every case class (Arrow C++ results)"`

## utf8_is_printable

True when no code point is in Cc, Cf, Cs, Co, Cn, Zs, Zl or Zp, except U+0020, which is printable.
An empty row is true, as in Arrow.

Evidence: `src/compute/string.zig:96`, `src/compute/string.zig:404`, `src/compute/test_string.zig:"string Unicode case mapping and predicates on every case class (Arrow C++ results)"`

## utf8_is_space

True for a non-empty row whose code points are all Unicode White_Space or U+001C..U+001F (the set
Arrow gets from utf8proc). Empty: false.

Evidence: `src/compute/string.zig:97`, `src/compute/string.zig:245`, `src/compute/test_string.zig:"string Unicode case mapping and predicates on every case class (Arrow C++ results)"`

## utf8_is_title

Python `istitle` rules over cased code points: at least one cased code point, every lowercase one
right after a cased one and every other cased one right after an uncased one.

Evidence: `src/compute/string.zig:98`, `src/compute/string.zig:852`, `src/compute/test_string.zig:"string Unicode case mapping and predicates on every case class (Arrow C++ results)"`

## utf8_is_upper

True when the row has at least one cased code point and every cased one is uppercase.

Evidence: `src/compute/string.zig:99`, `src/compute/string.zig:390`, `src/compute/test_string.zig:"string Unicode case mapping and predicates on every case class (Arrow C++ results)"`
