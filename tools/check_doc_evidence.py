#!/usr/bin/env python3
"""Check that every compute / compatibility claim in the docs cites evidence that exists.

Rule (docs/compute-support-matrix.md, "Evidence rule"): a table row whose status is `yes` or
`partial` must cite at least one source reference and one test reference, and every reference in
the checked docs must resolve:

- source reference  `path/to/file.zig:LINE`   -> the file exists and has at least LINE lines
- test reference    `path/to/file.zig:"test name"` -> the file contains  test "test name"
- build step        `build.zig:"step-name"`    -> build.zig mentions the step name

Usage: python -I tools/check_doc_evidence.py [doc ...]   (default: the compute docs)
Exit 1 on any missing evidence.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DOCS = ["docs/compute-support-matrix.md", "docs/canonical-extensions.md", "docs/compute-api.md"]
STATUS_NEEDING_EVIDENCE = ("| yes", "| partial")

SRC_REF = re.compile(r"`([\w./-]+\.(?:zig|py|json)):(\d+)`")
TEST_REF = re.compile(r"`([\w./-]+\.zig):\"([^\"`]+)\"`")


def line_count(path: Path) -> int:
    with path.open("rb") as f:
        return sum(1 for _ in f)


def check_doc(doc: Path) -> list[str]:
    problems = []
    text = doc.read_text(encoding="utf-8")
    for lineno, line in enumerate(text.splitlines(), 1):
        where = f"{doc.relative_to(ROOT)}:{lineno}"
        srcs = SRC_REF.findall(line)
        tests = TEST_REF.findall(line)
        for path, num in srcs:
            p = ROOT / path
            if not p.is_file():
                problems.append(f"{where}: missing file {path}")
            elif int(num) > line_count(p):
                problems.append(f"{where}: {path} has fewer than {num} lines")
        for path, name in tests:
            p = ROOT / path
            if not p.is_file():
                problems.append(f"{where}: missing file {path}")
                continue
            content = p.read_text(encoding="utf-8", errors="replace")
            needle = f'test "{name}"' if path != "build.zig" else name
            if needle not in content:
                problems.append(f'{where}: {path} has no test "{name}"')
        is_claim_row = line.startswith("|") and any(s in line for s in STATUS_NEEDING_EVIDENCE)
        if is_claim_row and doc.name != "compute-api.md":
            if not srcs:
                problems.append(f"{where}: claim row cites no source reference")
            if not tests:
                problems.append(f"{where}: claim row cites no test reference")
    return problems


def main(argv: list[str]) -> int:
    docs = [ROOT / d for d in (argv or DEFAULT_DOCS)]
    problems = [p for d in docs for p in check_doc(d)]
    for p in problems:
        print(p)
    print(f"{'FAIL' if problems else 'OK'}: {len(problems)} problem(s) in {len(docs)} doc(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
