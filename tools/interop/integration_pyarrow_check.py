#!/usr/bin/env python3
"""Cross-check zarrow's integration runner against pyarrow on the apache/arrow-testing gold files
(docs/integration-suite.md). Run with `python -P` (pyarrow) and pass the paths as arguments:

  python -P tools/interop/integration_pyarrow_check.py <integration-runner exe> <arrow-testing dir> <scratch dir>

For every case (minus the known rejections; big-endian gold files included, which pyarrow and zarrow
both read byte-swapped to little-endian) it checks that
  1. `--json-to-arrow` of the gold JSON gives a file pyarrow reads equal to the gold arrow_file
     (schema with metadata, and data),
  2. `--arrow-to-json` of the gold arrow_file gives JSON equal to the gold JSON after a
     normalising parse (key order, number spelling and the values of null slots ignored),
  3. `--stream-to-file` of the gold stream and `--file-to-stream` of the gold file give IPC
     pyarrow reads equal to the gold data.
Exit 1 on any difference. The gold files are data only: nothing from them is imported or run.
"""
import gzip
import json
import os
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc

KNOWN_REJECTIONS: set[str] = set()  # cases zarrow rejects on purpose (none today)


def run(exe, *args, stdin=None):
    return subprocess.run([exe, *args], input=stdin, capture_output=True, check=False)


def read_ipc(path, stream):
    with pa.memory_map(str(path)) as src:
        reader = ipc.open_stream(src) if stream else ipc.open_file(src)
        return reader.read_all()


def same_table(a, b):
    if not a.schema.equals(b.schema, check_metadata=True):
        return f"schema differs:\n{a.schema}\nvs\n{b.schema}"
    if not a.equals(b):
        return "data differs"
    return None


def as_int(v):
    return int(v) if isinstance(v, str) else v


def norm_value(v):
    if isinstance(v, str):
        try:
            return int(v)
        except ValueError:
            return v
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return v
    if isinstance(v, dict):
        return {k: as_int(x) for k, x in v.items()}
    return v


def norm_column(col, path):
    """A comparable form of a FieldData: values of null slots dropped."""
    out = {"count": col["count"]}
    validity = col.get("VALIDITY")
    valid = [bool(x) for x in validity] if validity is not None else None
    if validity is not None and "TYPE_ID" in col:
        validity = valid = None  # pre-1.0 unions had a validity bitmap; the format dropped it
    if validity is not None:
        out["VALIDITY"] = [int(bool(x)) for x in validity]

    def keep(i):
        return valid is None or valid[i]

    if "DATA" in col:
        out["DATA"] = [norm_value(v) if keep(i) else None for i, v in enumerate(col["DATA"])]
    if "OFFSET" in col and "VIEWS" not in col:
        offs = [as_int(x) for x in col["OFFSET"]]
        if "DATA" in col:
            # binary / string: the lengths of valid values (DATA holds the values themselves)
            out["LENGTHS"] = [offs[i + 1] - offs[i] if keep(i) else None for i in range(len(offs) - 1)] if offs else []
        else:
            base = offs[0] if offs else 0
            out["OFFSET"] = [o - base for o in offs] if "SIZE" not in col else offs
    if "SIZE" in col:
        out["SIZE"] = [as_int(x) if keep(i) else None for i, x in enumerate(col["SIZE"])]
    if "TYPE_ID" in col:
        out["TYPE_ID"] = col["TYPE_ID"]
    if "VIEWS" in col:
        out["VIEWS"] = [v if keep(i) else None for i, v in enumerate(col["VIEWS"])]
        out["VARIADIC_DATA_BUFFERS"] = col["VARIADIC_DATA_BUFFERS"]
    children = col.get("children") or []
    out["children"] = [norm_column(c, path + "/" + c.get("name", "?")) for c in children]
    return out


def norm_meta(m):
    # Arrow compares key-value metadata as a set of pairs.
    return sorted((kv["key"], kv["value"]) for kv in (m or []))


def norm_field(f):
    dictionary = f.get("dictionary")
    return {
        "name": f["name"],
        "nullable": f["nullable"],
        "type": f["type"] if f["type"].get("name") != "decimal" else {**{"bitWidth": 128}, **f["type"]},
        "children": [norm_field(c) for c in f.get("children") or []],
        "dictionary": {k: v for k, v in dictionary.items() if k != "id"} if dictionary else None,
        "metadata": norm_meta(f.get("metadata")),
    }


def norm_doc(d):
    return {
        "schema": {"fields": [norm_field(f) for f in d["schema"]["fields"]], "metadata": norm_meta(d["schema"].get("metadata"))},
        # Dictionary ids are an IPC detail: a writer may share one dictionary between fields or
        # give each its own (Arrow C++ does the latter for nested dictionaries), so the distinct
        # dictionaries are compared, not their ids.
        "dictionaries": sorted({json.dumps([x["data"]["count"], [norm_column(c, "dict") for c in x["data"]["columns"]]], sort_keys=True) for x in d.get("dictionaries") or []}),
        "batches": [(b["count"], [norm_column(c, c.get("name", "?")) for c in b["columns"]]) for b in d["batches"]],
    }


def first_difference(a, b, path="$"):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
        a, b = float(a), float(b)  # 445 and 445.0 are one JSON number
    if type(a) is not type(b):
        return f"{path}: {a!r} vs {b!r}"[:400]
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                return f"{path}.{k}: present on one side only"
            d = first_difference(a[k], b[k], f"{path}.{k}")
            if d:
                return d
        return None
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return f"{path}: length {len(a)} vs {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            d = first_difference(x, y, f"{path}[{i}]")
            if d:
                return d
        return None
    if isinstance(a, float):
        # float32 values: Arrow C++ (and zarrow) write the double of the float, the gold files
        # a short decimal that rounds to it.
        return None if a == b or abs(a - b) <= 1e-6 * max(1.0, abs(a)) else f"{path}: {a!r} vs {b!r}"
    return None if a == b else f"{path}: {a!r} vs {b!r}"[:400]


def main():
    exe, root, scratch = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    scratch.mkdir(parents=True, exist_ok=True)
    base = root / "data" / "arrow-ipc-stream" / "integration"
    failures = 0
    checked = 0
    for d in sorted(p for p in base.iterdir() if p.is_dir()):
        for gz in sorted(d.glob("*.json.gz")):
            case = gz.name[: -len(".json.gz")]
            if case in KNOWN_REJECTIONS:
                continue
            checked += 1
            tag = f"{d.name}/{case}"
            text = gzip.decompress(gz.read_bytes())
            json_path = scratch / f"{d.name}-{case}.json"
            json_path.write_bytes(text)
            gold_file = read_ipc(d / f"{case}.arrow_file", stream=False)
            problems = []

            out_file = scratch / f"{d.name}-{case}.zarrow.arrow_file"
            r = run(exe, "--integration", f"--json={json_path}", f"--arrow={out_file}", "--mode=JSON_TO_ARROW")
            if r.returncode != 0:
                problems.append(f"json-to-arrow failed: {r.stderr.decode(errors='replace').strip()}")
            else:
                diff = same_table(read_ipc(out_file, stream=False), gold_file)
                if diff:
                    problems.append(f"json-to-arrow: {diff}")

            out_json = scratch / f"{d.name}-{case}.zarrow.json"
            r = run(exe, "--integration", f"--json={out_json}", f"--arrow={d / (case + '.arrow_file')}", "--mode=ARROW_TO_JSON")
            if r.returncode != 0:
                problems.append(f"arrow-to-json failed: {r.stderr.decode(errors='replace').strip()}")
            else:
                ours = json.loads(out_json.read_bytes())
                gold = json.loads(text)
                diff = first_difference(norm_doc(ours), norm_doc(gold))
                if diff:
                    problems.append(f"arrow-to-json vs gold JSON: {diff}")

            r = run(exe, "--integration", f"--json={json_path}", f"--arrow={d / (case + '.arrow_file')}", "--mode=VALIDATE")
            if r.returncode != 0:
                problems.append(f"validate failed: {r.stderr.decode(errors='replace').strip()}")

            for flag, src, stream_in in (("--stream-to-file", f"{case}.stream", True), ("--file-to-stream", f"{case}.arrow_file", False)):
                out = scratch / f"{d.name}-{case}{flag}.out"
                r = run(exe, flag, str(d / src), str(out))
                if r.returncode != 0:
                    problems.append(f"{flag} failed: {r.stderr.decode(errors='replace').strip()}")
                    continue
                diff = same_table(read_ipc(out, stream=stream_in is False), read_ipc(d / src, stream=stream_in))
                if diff:
                    problems.append(f"{flag}: {diff}")

            if problems:
                failures += 1
                print(f"FAIL {tag}")
                for p in problems:
                    print(f"    {p}")
            else:
                print(f"ok   {tag}")
    print(f"\n{checked} cases checked against pyarrow {pa.__version__}: {failures} with differences")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
