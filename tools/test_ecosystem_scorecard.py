#!/usr/bin/env python3
"""Checks for ecosystem_scorecard.count. Run: python -I tools/test_ecosystem_scorecard.py"""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("ecosystem_scorecard", Path(__file__).with_name("ecosystem_scorecard.py"))
scorecard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scorecard)

SAMPLE = """\
# Doc

| Type | Build | Validate | IPC | C ABI | Compute | Status | Source | Tests |
|---|---|---|---|---|---|---|---|---|
| A | yes | yes | yes | yes | yes | yes | `a.zig:1` | `a.zig:"t"` |
| B | yes | partial | no | partial (format only) | yes | partial | `b.zig:1` | `b.zig:"t"` |

Text between tables resets the header.

| Shared | Source | Tests |
|---|---|---|
| ignored yes | `c.zig:1` | `c.zig:"t"` |

| Capability | Status | Source |
|:---|---|---|
| C | no (missing) | |
"""

counts, errors = scorecard.count(SAMPLE)
assert errors == [], errors
assert counts["Status"] == {"yes": 1, "partial": 1, "no": 1}, counts["Status"]
assert counts["Build"] == {"yes": 2}, counts["Build"]
assert counts["IPC"] == {"yes": 1, "no": 1}, counts["IPC"]
assert counts["C ABI"] == {"yes": 1, "partial": 1}, counts["C ABI"]
assert scorecard.ratio(counts["Validate"]) == "0.50"

bad = SAMPLE.replace("| partial (format only) |", "| maybe |").replace("| no (missing) |", "| nope |")
counts, errors = scorecard.count(bad)
assert len(errors) == 2, errors
assert "maybe" in errors[0] and "C ABI" in errors[0], errors
assert "nope" in errors[1], errors  # "nope" must not pass as "no"

counts, errors = scorecard.count("| a | b |\n|---|---|\n| yes | yes |\n")
assert counts == {} and errors == []  # no Status column: not a claim table

assert scorecard.main(list(scorecard.DOCS)) == 0
print("OK: test_ecosystem_scorecard")
