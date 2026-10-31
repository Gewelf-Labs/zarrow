#!/usr/bin/env python3
"""Markdown triage report for a fuzz output dir written by tools/fuzz.py (stdlib only).

  python tools/fuzz_triage.py [.fuzz-out] > triage.md

Per target: each unique crash signature with its count, kind, smallest reproducer, the first 10
non-std stack frames and the first frame inside src/ as the suggested place to look.
"""
import json
import sys
from pathlib import Path


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else ".fuzz-out")
    indexes = sorted(out.glob("*/index.json"))
    print(f"# Fuzz triage: {out}\n")
    if not indexes:
        print("No findings.")
        return 0
    for index_path in indexes:
        target = index_path.parent.name
        index = json.loads(index_path.read_text())
        total = sum(e["count"] for e in index.values())
        print(f"## {target}: {len(index)} unique signatures, {total} crashes\n")
        for n, (sig, e) in enumerate(sorted(index.items(), key=lambda kv: -kv[1]["count"]), 1):
            frames = e.get("frames") or []
            src = next((f for f in frames if f[0].startswith("src/")), None)
            repro = index_path.parent / f"{e['smallest']}.bin"
            print(f"### {target} #{n}: {e['kind']} x{e['count']}\n")
            print(f"- message: `{e['message'] or '(none)'}`")
            print(f"- signature: `{sig}`")
            print(f"- smallest reproducer: `{repro}` ({e['size']} bytes); trace in `{repro.with_suffix('.txt')}`")
            print(f"- suggested location: `{src[0]}:{src[1]}` in `{src[2]}`" if src else "- suggested location: (no src/ frame)")
            print(f"- replay: `zig build fuzz -- {target} replay {repro}`")
            if frames:
                print("\n```")
                for path, line, fn in frames[:10]:
                    print(f"{path}:{line} {fn}")
                print("```")
            print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
