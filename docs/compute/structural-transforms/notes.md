# Structural transforms: function notes

Up: [function matrix](matrix.md) · [compute support matrix](../../compute-support-matrix.md)

Arrow C++'s "Structural transforms", registered by `src/compute/structural.zig:60` as array-producing
kernels, as Arrow C++. The list functions accept list, large_list, list_view, large_list_view and
fixed_size_list (a map is not a list for them, as Arrow). Every input chunk and the child read through
it pass `validateLayout`; items are gathered with the take kernels, so every child type works.
Checked against pyarrow 26 on nulls, sliced, chunked (a chunked result) and scalar inputs, and against
a per-row loop on random sliced inputs. zarrow has no scalars of list_view, large_list_view, map or
union types. Evidence: `src/compute/test_structural.zig:"structural oracle matches a per-row loop on random sliced inputs with nulls for every layout"`, `src/compute/test_structural.zig:"structural chunked inputs with misaligned chunks give a chunked result equal to the array result"`, `src/compute/test_structural.zig:"structural scalars, empty and all-null inputs"`.

## Summary

| Function | Arity | Input types | Output type | Options |
|---|---|---|---|---|
| [`list_element`](#list_element) | Binary | list layouts; integer index scalar | list value type | none |
| [`list_slice`](#list_slice) | Unary | list layouts | list layout or fixed_size_list | `ListSliceOptions` |
| [`list_value_length`](#list_value_length) | Unary | list layouts | int32 (int64 for large layouts) | none |
| [`make_struct`](#make_struct) | Varargs | any | struct | `MakeStructOptions` |
| [`struct_field`](#struct_field) | Unary | struct, sparse_union, dense_union | field type | `StructFieldOptions` |
| [`map_lookup`](#map_lookup) | Unary | map | item type, or list of items | `MapLookupOptions` |
| [`list_flatten`](#list_flatten) | Unary | list layouts | list value type | `ListFlattenOptions` |
| [`list_parent_indices`](#list_parent_indices) | Unary | list layouts | int64 | none |

## list_element

The item at a given position of each list row. The index is a scalar of any integer type; a null
or negative index, or one at or past the length of a valid row, is `InvalidInput` (Arrow's
Invalid). An index array is `NotImplemented`, as Arrow. A null list row gives null.

Deviation: on list_view / large_list_view zarrow returns the item, where pyarrow 26 rejects every
index ("should be in [0, -2)").

Evidence: `src/compute/structural.zig:67`, `src/compute/structural.zig:236`, `src/compute/test_structural.zig:"structural list functions: list_value_length, list_element, list_flatten, list_parent_indices, list_slice on every list layout, sliced (pyarrow 26 results)"`, `src/compute/test_structural.zig:"structural errors: indices, options, types, arity and malformed input"`, `src/compute/test_structural.zig:"structural oracle matches a per-row loop on random sliced inputs with nulls for every layout"`

## list_slice

A sub-list of each row from `start` up to `stop` by `step`. Options (`src/compute/options.zig:592`):
`start` (required, `>= 0`), `stop` (default: the end of each row; `>= start`), `step` (default 1,
`>= 1`), `return_fixed_size_list` (default null). Output layout: the input's list layout; a
fixed_size_list input gives list with `false`; with `true` a fixed_size_list of
`ceil((stop - start) / step)` items padded with nulls, which needs `stop` unless the input is
already fixed_size_list. Violating any of these checks is an options error, as Arrow.

Evidence: `src/compute/structural.zig:71`, `src/compute/structural.zig:429`, `src/compute/options.zig:592`, `src/compute/test_structural.zig:"structural list functions: list_value_length, list_element, list_flatten, list_parent_indices, list_slice on every list layout, sliced (pyarrow 26 results)"`, `src/compute/test_structural.zig:"structural errors: indices, options, types, arity and malformed input"`

## list_value_length

The number of items in each list row: int32, or int64 for large_list and large_list_view. A null
row gives null.

Evidence: `src/compute/structural.zig:63`, `src/compute/structural.zig:170`, `src/compute/test_structural.zig:"structural list functions: list_value_length, list_element, list_flatten, list_parent_indices, list_slice on every list layout, sliced (pyarrow 26 results)"`, `src/compute/test_structural.zig:"structural oracle matches a per-row loop on random sliced inputs with nulls for every layout"`, `src/compute/test_structural.zig:"structural scalars, empty and all-null inputs"`

## make_struct

A struct whose fields are the arguments. `MakeStructOptions` `field_names`, `field_nullability`,
`field_metadata`: one entry per argument, or empty for pyarrow's defaults (names "0", "1", ...,
nullable, no metadata). Scalars are repeated over the array length; all scalars give a struct
scalar. A field declared not nullable with a null argument row is `InvalidInput` (Arrow's Invalid).

Evidence: `src/compute/structural.zig:84`, `src/compute/structural.zig:559`, `src/compute/options.zig:607`, `src/compute/test_structural.zig:"structural make_struct, struct_field and map_lookup hand-written values on structs, unions and maps (pyarrow 26 results)"`, `src/compute/test_structural.zig:"structural scalars, empty and all-null inputs"`

## struct_field

Extracts a nested field by a path of child indices or of names (`StructFieldOptions.indices` /
`names`), through struct, sparse_union and dense_union levels. A null struct row, or a union row
that holds another child, is a null result row. An empty path gives the input. An out-of-range
index, or a missing or ambiguous name, is `InvalidOptions`.

Evidence: `src/compute/structural.zig:89`, `src/compute/structural.zig:698`, `src/compute/options.zig:616`, `src/compute/test_structural.zig:"structural make_struct, struct_field and map_lookup hand-written values on structs, unions and maps (pyarrow 26 results)"`, `src/compute/test_structural.zig:"structural errors: indices, options, types, arity and malformed input"`

## map_lookup

Looks up `MapLookupOptions.query_key` in each map row. `occurrence` `first` (default) or `last`
gives the item, `all` a list of every matching item. A null map, an empty map or a missing key
gives null. Keys compare by value over integer, bool, temporal, interval, decimal,
fixed_size_binary and string / binary keys. A null query is `InvalidOptions`; a query of another
type than the key, and floating-point keys, are `UnsupportedType` (Arrow's TypeError).

Deviation: zarrow has no map scalars, so map_lookup takes arrays and chunked arrays only, and a
scalar result of a list_view, map or union type is `UnsupportedType`.

Evidence: `src/compute/structural.zig:94`, `src/compute/structural.zig:844`, `src/compute/options.zig:622`, `src/compute/test_structural.zig:"structural make_struct, struct_field and map_lookup hand-written values on structs, unions and maps (pyarrow 26 results)"`, `src/compute/test_structural.zig:"structural errors: indices, options, types, arity and malformed input"`, `src/compute/test_structural.zig:"structural chunked inputs with misaligned chunks give a chunked result equal to the array result"`

## list_flatten

Vector function: the items of all valid list rows, in order. `ListFlattenOptions.recursive`
(default false) flattens until the result is no list; a map stops it, as Arrow. A list scalar gives
an array. Rows stored back to back come back as a slice of the child (no copy).

Evidence: `src/compute/structural.zig:75`, `src/compute/structural.zig:283`, `src/compute/options.zig:585`, `src/compute/test_structural.zig:"structural list functions: list_value_length, list_element, list_flatten, list_parent_indices, list_slice on every list layout, sliced (pyarrow 26 results)"`, `src/compute/test_structural.zig:"structural chunked inputs with misaligned chunks give a chunked result equal to the array result"`

## list_parent_indices

Vector function: for each item of the flattened lists, the row index of its list. A chunked input
counts rows from the start of its first chunk, as pyarrow 26. A scalar is `NotImplemented`, as
Arrow.

Evidence: `src/compute/structural.zig:79`, `src/compute/structural.zig:357`, `src/compute/test_structural.zig:"structural list functions: list_value_length, list_element, list_flatten, list_parent_indices, list_slice on every list layout, sliced (pyarrow 26 results)"`, `src/compute/test_structural.zig:"structural chunked inputs with misaligned chunks give a chunked result equal to the array result"`, `src/compute/test_structural.zig:"structural errors: indices, options, types, arity and malformed input"`
