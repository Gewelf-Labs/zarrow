#!/usr/bin/env python3
"""Write the canonical extension fixtures read by src/extension/test_canonical.zig, or check a
zarrow-written stream of canonical extension types (`check <in.arrows>`).

    python -P tools/interop/pyarrow_canonical_extensions.py write src/extension/testdata
    python -P tools/interop/pyarrow_canonical_extensions.py check .interop-fixtures/zarrow_canonical_ext.arrows

pyarrow (26) registers arrow.uuid, arrow.bool8, arrow.json, arrow.fixed_shape_tensor and
arrow.opaque; arrow.variable_shape_tensor, arrow.timestamp_with_offset and arrow.parquet.variant
are written as their storage with the extension field metadata. pyarrow lives in the user site,
so run this with `python -P` (isolated mode hides it).
"""
import sys
from pathlib import Path

import pyarrow as pa


def ext_field(name, storage, ext_name, ext_md=b"", nullable=True):
    return pa.field(name, storage, nullable=nullable, metadata={b"ARROW:extension:name": ext_name, b"ARROW:extension:metadata": ext_md})


def write(out: Path) -> None:
    assert pa.__version__.startswith("26."), pa.__version__
    uuid = pa.ExtensionArray.from_storage(pa.uuid(), pa.array([bytes(range(16)), None, b"\xff" * 16], pa.binary(16)))
    bool8 = pa.ExtensionArray.from_storage(pa.bool8(), pa.array([1, None, 0], pa.int8()))
    js = pa.ExtensionArray.from_storage(pa.json_(pa.string()), pa.array(['{"a": [1, 2.5e3]}', None, '"x"']))
    tensor_type = pa.fixed_shape_tensor(pa.int32(), [2, 3], dim_names=["r", "c"], permutation=[1, 0])
    tensor = pa.ExtensionArray.from_storage(tensor_type, pa.array([list(range(6)), None, [6] * 6], pa.list_(pa.int32(), 6)))
    opaque = pa.ExtensionArray.from_storage(pa.opaque(pa.int32(), "geometry", "postgis"), pa.array([1, 2, None], pa.int32()))
    vst_storage = pa.array(
        [{"data": [1, 2, 3, 4], "shape": [2, 2]}, None, {"data": [], "shape": [0, 2]}],
        pa.struct([("data", pa.list_(pa.int32())), ("shape", pa.list_(pa.int32(), 2))]))
    two_storage = pa.array(
        [{"timestamp": 1_700_000_000_000_000, "offset_minutes": 60}, {"timestamp": 0, "offset_minutes": -330}, {"timestamp": 5, "offset_minutes": 0}],
        pa.struct([pa.field("timestamp", pa.timestamp("us", "UTC"), nullable=False), pa.field("offset_minutes", pa.int16(), nullable=False)]))
    variant_storage = pa.array(
        [{"metadata": b"\x01\x00", "value": b"\x0c\x01"}, {"metadata": b"\x01\x00", "value": None}, None],
        pa.struct([pa.field("metadata", pa.binary(), nullable=False), pa.field("value", pa.binary())]))
    fields = [
        pa.field("uuid", pa.uuid()), pa.field("bool8", pa.bool8()), pa.field("json", js.type),
        pa.field("tensor", tensor_type), pa.field("opaque", opaque.type),
        ext_field("vst", vst_storage.type, b"arrow.variable_shape_tensor", b'{"uniform_shape": [null, 2], "dim_names": ["n", "k"]}'),
        ext_field("two", two_storage.type, b"arrow.timestamp_with_offset"),
        ext_field("variant", variant_storage.type, b"arrow.parquet.variant"),
    ]
    arrays = [uuid, bool8, js, tensor, opaque, vst_storage, two_storage, variant_storage]
    batch = pa.RecordBatch.from_arrays(arrays, schema=pa.schema(fields))
    out.mkdir(parents=True, exist_ok=True)
    with pa.ipc.new_stream(out / "pyarrow_canonical.arrows", batch.schema) as w:
        w.write_batch(batch)
    # A uuid over 8-byte storage: a canonical name with the wrong storage type.
    bad = pa.RecordBatch.from_arrays([pa.array([b"12345678"], pa.binary(8))], schema=pa.schema([ext_field("u", pa.binary(8), b"arrow.uuid")]))
    with pa.ipc.new_stream(out / "pyarrow_canonical_bad_uuid.arrows", bad.schema) as w:
        w.write_batch(bad)
    # A fixed_shape_tensor whose shape product is not the list size.
    bad = pa.RecordBatch.from_arrays([pa.array([[1, 2, 3]], pa.list_(pa.int32(), 3))],
                                     schema=pa.schema([ext_field("t", pa.list_(pa.int32(), 3), b"arrow.fixed_shape_tensor", b'{"shape": [2, 2]}')]))
    with pa.ipc.new_stream(out / "pyarrow_canonical_bad_tensor.arrows", bad.schema) as w:
        w.write_batch(bad)
    # Values that break the constraints validateFull checks: a tensor row whose data length is
    # not its shape's product, an offset of 2000 minutes, a JSON value that is not JSON.
    bad_vst = pa.array([{"data": [1, 2, 3], "shape": [2, 2]}], vst_storage.type)
    bad_two = pa.array([{"timestamp": 0, "offset_minutes": 2000}], two_storage.type)
    bad_json = pa.ExtensionArray.from_storage(pa.json_(pa.string()), pa.array(["{oops}"]))
    bad = pa.RecordBatch.from_arrays([bad_vst, bad_two, bad_json], schema=pa.schema([
        ext_field("vst", vst_storage.type, b"arrow.variable_shape_tensor", b'{"uniform_shape": [null, 2]}'),
        ext_field("two", two_storage.type, b"arrow.timestamp_with_offset"), pa.field("json", bad_json.type)]))
    with pa.ipc.new_stream(out / "pyarrow_canonical_bad_values.arrows", bad.schema) as w:
        w.write_batch(bad)
    print("wrote", out)


def check(path: Path) -> None:
    with pa.ipc.open_stream(path) as r:
        t = r.read_all()
    s = t.schema
    want = {"uuid": pa.UuidType, "bool8": pa.Bool8Type, "json": pa.JsonType, "tensor": pa.FixedShapeTensorType, "opaque": pa.OpaqueType}
    for name, cls in want.items():
        if not isinstance(s.field(name).type, cls):
            raise SystemExit("%s: %r is not %s" % (name, s.field(name).type, cls.__name__))
    if t.column("uuid").to_pylist()[0].bytes != bytes(range(16)):
        raise SystemExit("uuid values")
    if t.column("bool8").to_pylist() != [True, None, False]:
        raise SystemExit("bool8 values: %r" % t.column("bool8").to_pylist())
    if t.column("json").to_pylist() != ['{"a": [1, 2.5e3]}', None, '"x"']:
        raise SystemExit("json values")
    tt = s.field("tensor").type
    if tt.shape != [2, 3] or tt.dim_names != ["r", "c"] or tt.permutation != [1, 0]:
        raise SystemExit("tensor type %r" % tt)
    ot = s.field("opaque").type
    if ot.type_name != "geometry" or ot.vendor_name != "postgis":
        raise SystemExit("opaque type %r" % ot)
    t.validate(full=True)
    print("ok: pyarrow reads zarrow's canonical extension types")


if __name__ == "__main__":
    {"write": write, "check": check}[sys.argv[1]](Path(sys.argv[2]))
