# Associative transforms: function notes

Up: [function matrix](matrix.md) · [compute support matrix](../../compute-support-matrix.md)

Hash-based vector functions over distinct values, as Arrow's associative transforms. They accept array, chunked or scalar inputs of every key type (numbers, float16, bool, temporal, intervals, decimals, fixed_size_binary, string and binary layouts); a null is one value; floats compare by their bits, as pyarrow 26 hashes them: NaN with another sign or payload is another value, -0.0 and 0.0 two (`count_distinct([0.0, -0.0, nan, -nan])` is 4) (pyarrow 26 results; `src/compute/test_associative.zig:"hash-based kernels compare floats by bits: NaN sign and payload, signed zeros; nulls, sliced, chunked (pyarrow 26 results)"`). Every dictionary passes `validateDictionary` (layout, value type, index bounds) before an index is read.

## Summary

| Function | Arity | Input types | Output type | Options |
|---|---|---|---|---|
| [`dictionary_encode`](#dictionary_encode) | Unary | every key type, null, dictionary | `dictionary<int32, T>` | `DictionaryEncodeOptions` |
| [`dictionary_decode`](#dictionary_decode) | Unary | dictionary (other types pass through) | the dictionary value type | none |
| [`unique`](#unique) | Unary | every key type, null, dictionary | input type | none |
| [`value_counts`](#value_counts) | Unary | every key type, null, dictionary | `struct<values: T, counts: int64>` | none |

## dictionary_encode

`dictionary_encode` returns int32 indices into a dictionary of the distinct values in first-seen order.

- Options: `DictionaryEncodeOptions.null_encoding`: `mask` (default) keeps nulls out of the dictionary (null index), `encode` gives null an entry.
- A chunked input gives chunks sharing one dictionary.
- A dictionary input is returned as it is.
- The null type gives a dictionary of one null, every index null (`mask`) or 0 (`encode`), as pyarrow 26.
- The `dictionaryEncode` helper stays for fixed-width and string / binary arrays.

Evidence: `src/compute/associative.zig:65`, `src/compute/associative.zig:56`, `src/compute/test_associative.zig:"unique, value_counts and dictionary_encode hand-written values (pyarrow 26 results)"`, `src/compute/test_associative.zig:"associative oracle on random sliced inputs with nulls: unique and counts equal a linear scan, dictionary_encode decodes back"`, `src/compute/test_associative.zig:"associative edge cases and errors: empty, scalars, unsupported types, options and malformed input"`

## dictionary_decode

`dictionary_decode` (Arrow's meta function) returns the plain values of a dictionary array or chunked array.

- Each chunk is decoded with its own dictionary and passes `validateLayout` and `validateDictionary`.
- A null index and an index to a null entry both give null.
- Any other type, also an extension type over a dictionary, is returned as is (pyarrow 26 does the same).
- A scalar is `NoMatchingKernel` (Arrow's TypeError "Expected an Array or a Chunked Array").

Evidence: `src/compute/associative.zig:69`, `src/compute/associative.zig:56`, `src/compute/test_associative.zig:"dictionary_decode: nulls, null entries, sliced, chunked, non-dictionary and extension inputs returned as is (pyarrow 26 results)"`

## unique

`unique` returns the distinct values in first-seen order.

- Dictionary inputs work on the indices and keep the dictionary: the result has the input dictionary type and unused entries stay; a null index and an index to a null entry are two values. A direct table over the index values is used when the dictionary is at most 4 times the rows plus 4096, the hash grouper otherwise.
- Chunks with different dictionaries are unified first (the first dictionary, then the new values of the next ones; index type kept); dictionaries with null entries are `InvalidInput` (Arrow's "Cannot yet unify dictionaries with nulls").
- Also the null type.

Evidence: `src/compute/associative.zig:57`, `src/compute/associative.zig:56`, `src/compute/test_associative.zig:"unique, value_counts and dictionary_encode hand-written values (pyarrow 26 results)"`, `src/compute/test_associative.zig:"associative oracle on random sliced inputs with nulls: unique and counts equal a linear scan, dictionary_encode decodes back"`, `src/compute/test_associative.zig:"associative edge cases and errors: empty, scalars, unsupported types, options and malformed input"`

## value_counts

`value_counts` returns each distinct value with its count as `struct<values: T, counts: int64>` (the struct type stored with the array), values in first-seen order. Dictionary and chunked inputs follow the rules of [`unique`](#unique). Also the null type.

Evidence: `src/compute/associative.zig:61`, `src/compute/associative.zig:56`, `src/compute/test_associative.zig:"unique, value_counts and dictionary_encode hand-written values (pyarrow 26 results)"`, `src/compute/test_associative.zig:"associative oracle on random sliced inputs with nulls: unique and counts equal a linear scan, dictionary_encode decodes back"`, `src/compute/test_associative.zig:"associative edge cases and errors: empty, scalars, unsupported types, options and malformed input"`
