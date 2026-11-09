# Logical: function notes

Up: [function matrix](matrix.md) · [compute support matrix](../../compute-support-matrix.md)

The Arrow C++ "Logical functions" in `logical.zig`, on bool operands only (no implicit casts; a
non-bool operand is an error). Each takes bool arrays, chunked arrays or scalars at any bit offset
and returns bool. The plain functions propagate nulls (intersection); the `_kleene` functions use
Kleene three-valued logic, where a known operand can decide the row alone. Truth tables are checked
against Arrow. Evidence: `src/compute/logical.zig:25`,
`src/compute/test_logical.zig:"logical hand-written truth tables including Kleene logic (Arrow C++ results)"`,
`src/compute/test_logical.zig:"logical errors: non-bool operands, arity, options, lengths and malformed scalars"`.

## Summary

| Function | Arity | Input types | Output type | Options |
|---|---|---|---|---|
| [`and`](#and) | Binary | bool | bool | none |
| [`and_kleene`](#and_kleene) | Binary | bool | bool | none |
| [`and_not`](#and_not) | Binary | bool | bool | none |
| [`and_not_kleene`](#and_not_kleene) | Binary | bool | bool | none |
| [`invert`](#invert) | Unary | bool | bool | none |
| [`or`](#or) | Binary | bool | bool | none |
| [`or_kleene`](#or_kleene) | Binary | bool | bool | none |
| [`xor`](#xor) | Binary | bool | bool | none |

## and

`a and b`; a null in either operand gives null.

Evidence: `src/compute/logical.zig:40`,
`src/compute/test_logical.zig:"logical hand-written truth tables including Kleene logic (Arrow C++ results)"`,
`src/compute/test_logical.zig:"logical oracle matches a per-element loop on random inputs at unaligned bit offsets"`,
`src/compute/test_logical.zig:"logical chunked inputs with misaligned chunks give a chunked result equal to the array result"`

## and_kleene

`a and b` in Kleene logic: `false and null` is false (either side), `true and null` and
`null and null` are null. A null scalar operand keeps the deciding side of the other operand.

Evidence: `src/compute/logical.zig:73`,
`src/compute/test_logical.zig:"logical hand-written truth tables including Kleene logic (Arrow C++ results)"`,
`src/compute/test_logical.zig:"logical scalar operands: Kleene keeps the deciding side of a null scalar, scalar-scalar gives a scalar"`

## and_not

`a and not b`; a null in either operand gives null.

Evidence: `src/compute/logical.zig:40`,
`src/compute/test_logical.zig:"logical hand-written truth tables including Kleene logic (Arrow C++ results)"`

## and_not_kleene

`a and not b` in Kleene logic: a false `a` or a true `b` decides the row alone (false), so
`null and not true` and `false and not null` are false; otherwise a null operand gives null.

Evidence: `src/compute/logical.zig:73`,
`src/compute/test_logical.zig:"logical hand-written truth tables including Kleene logic (Arrow C++ results)"`,
`src/compute/test_logical.zig:"logical empty and all-null inputs"`

## invert

`not a`; null stays null.

Evidence: `src/compute/logical.zig:40`,
`src/compute/test_logical.zig:"logical hand-written truth tables including Kleene logic (Arrow C++ results)"`,
`src/compute/test_logical.zig:"logical oracle matches a per-element loop on random inputs at unaligned bit offsets"`

## or

`a or b`; a null in either operand gives null.

Evidence: `src/compute/logical.zig:40`,
`src/compute/test_logical.zig:"logical hand-written truth tables including Kleene logic (Arrow C++ results)"`

## or_kleene

`a or b` in Kleene logic: `true or null` is true (either side), `false or null` and `null or null`
are null.

Evidence: `src/compute/logical.zig:73`,
`src/compute/test_logical.zig:"logical hand-written truth tables including Kleene logic (Arrow C++ results)"`,
`src/compute/test_logical.zig:"logical scalar operands: Kleene keeps the deciding side of a null scalar, scalar-scalar gives a scalar"`

## xor

`a xor b` (true when exactly one is true); a null in either operand gives null.

Evidence: `src/compute/logical.zig:40`,
`src/compute/test_logical.zig:"logical hand-written truth tables including Kleene logic (Arrow C++ results)"`
