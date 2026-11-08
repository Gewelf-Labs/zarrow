#!/usr/bin/env python3
"""Write src/ipc/testdata/pyarrow_tensors.bin: tensor messages written by pyarrow's
`ipc.write_tensor`, one after the other, read by the zarrow test
src/ipc/stream_reader.zig:"ipc reader reads pyarrow 26 tensor messages and writes them back".

Tensors: int32[2, 3] row-major with dimension names; float64[3, 4] column-major; int16[2, 2, 2]
from a strided view (pyarrow writes it contiguous); uint8[0, 5]; a 0-dimensional int64; and
float16[5].

pyarrow lives in the user site, so run it with `python -P`:
    python -P tools/interop/pyarrow_tensors.py src/ipc/testdata/pyarrow_tensors.bin
"""
import sys

import numpy as np
import pyarrow as pa


def main() -> None:
    if not pa.__version__.startswith("26."):
        sys.exit(f"pyarrow {pa.__version__}: expected 26.x")
    tensors = [
        pa.Tensor.from_numpy(np.arange(6, dtype=np.int32).reshape(2, 3), dim_names=["rows", "cols"]),
        pa.Tensor.from_numpy(np.asfortranarray(np.arange(12, dtype=np.float64).reshape(3, 4))),
        pa.Tensor.from_numpy(np.arange(24, dtype=np.int16).reshape(2, 3, 4)[:, ::2, 1:3]),
        pa.Tensor.from_numpy(np.zeros((0, 5), dtype=np.uint8)),
        pa.Tensor.from_numpy(np.array(7, dtype=np.int64)),
        pa.Tensor.from_numpy(np.arange(5, dtype=np.float16)),
    ]
    sink = pa.BufferOutputStream()
    for t in tensors:
        pa.ipc.write_tensor(t, sink)
    with open(sys.argv[1], "wb") as f:
        f.write(sink.getvalue().to_pybytes())


if __name__ == "__main__":
    main()
