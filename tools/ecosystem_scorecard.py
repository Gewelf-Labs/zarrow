#!/usr/bin/env python3
"""Count the yes / partial / no claim rows of the evidence-format docs.

A claim row is a Markdown table row whose Status cell starts with `yes`, `partial` or `no`. The Status
column is found by its header name; tables whose header also has the type-coverage dimension columns
(Build, Validate, IPC, C ABI, Compute) are counted per dimension as well. Ratios are yes / total per
column; partial is shown on its own and nothing is averaged across columns or docs.

Usage: python -I tools/ecosystem_scorecard.py [doc ...]   (default: DOCS)
Exit 1 when a doc has no claim rows or a Status / dimension cell is not yes, partial or no.
"""
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = (
    "docs/compute-support-matrix.md",
    "docs/canonical-extensions.md",
    "docs/type-coverage.md",
    "docs/interop-c-abi.md",
    "docs/interop-arrow-go.md",
    "docs/integration-suite.md",
)
STATUS = "Status"
DIMENSIONS = ("Build", "Validate", "IPC", "C ABI", "Compute")
STATUS_WORD = re.compile(r"(yes|partial|no)\b")
SEPARATOR = re.compile(r"\|\s*:?-{3,}")


def split_row(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def count(text: str) -> tuple[dict[str, Counter], list[str]]:
    """Counters per counted column (Status first, then dimensions) and a list of bad cells."""
    counts: dict[str, Counter] = {}
    errors = []
    lines = text.splitlines()
    columns: dict[str, int] = {}
    for i, line in enumerate(lines):
        if not line.startswith("|"):
            columns = {}
            continue
        if i + 1 < len(lines) and SEPARATOR.match(lines[i + 1]):  # header row of a new table
            header = split_row(line)
            columns = {name: header.index(name) for name in (STATUS, *DIMENSIONS) if name in header}
            if STATUS not in columns:
                columns = {}
            continue
        if not columns or SEPARATOR.match(line):
            continue
        cells = split_row(line)
        for name, idx in columns.items():
            cell = cells[idx] if idx < len(cells) else ""
            m = STATUS_WORD.match(cell)
            if not m:
                errors.append(f"line {i + 1}: {name} cell {cell!r} is not yes / partial / no")
                continue
            counts.setdefault(name, Counter())[m.group(1)] += 1
    return counts, errors


def ratio(c: Counter) -> str:
    total = sum(c.values())
    return f"{c['yes'] / total:.2f}" if total else "n/a"


def main(argv: list[str]) -> int:
    docs = argv or list(DOCS)
    rows = ["| Doc | Column | yes | partial | no | total | yes ratio |", "|---|---|---|---|---|---|---|"]
    problems = []
    for doc in docs:
        counts, errors = count((ROOT / doc).read_text(encoding="utf-8"))
        problems += [f"{doc}: {e}" for e in errors]
        if not counts.get(STATUS):
            problems.append(f"{doc}: no claim rows")
        for name, c in counts.items():
            rows.append(f"| {doc} | {name} | {c['yes']} | {c['partial']} | {c['no']} | {sum(c.values())} | {ratio(c)} |")
    print("\n".join(rows))
    for p in problems:
        print(p, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
