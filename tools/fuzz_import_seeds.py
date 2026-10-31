#!/usr/bin/env python3
"""Copy apache/arrow-testing IPC fuzz reproducers and integration fixtures into fuzz/corpus.

  python -I tools/fuzz_import_seeds.py <arrow-testing checkout> [--max-size 32768]

Stdlib only; the checkout is read as data, nothing in it is executed. Files are deduped by
content hash against what each corpus dir already holds, files over --max-size are skipped, and
each dir gets a NOTICE stating origin and license. Re-running is idempotent.
"""
import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# (corpus dir, source dir under data/, glob, name prefix)
INTEGRATION = ["cpp-21.0.0", "2.0.0-compression", "4.0.0-shareddict", "1.0.0-bigendian", "0.14.1"]
SOURCES = [("ipc-stream", "arrow-ipc-stream", "*", "at-stream-"),
           ("ipc-stream", "arrow-ipc-tensor-stream", "*", "at-tensor-"),
           ("ipc-file", "arrow-ipc-file", "*", "at-file-")]
for v in INTEGRATION:
    SOURCES += [("ipc-stream", f"arrow-ipc-stream/integration/{v}", "*.stream", f"at-{v}-"),
                ("ipc-file", f"arrow-ipc-stream/integration/{v}", "*.arrow_file", f"at-{v}-")]
SKIP_SUFFIXES = (".md", ".tmp", ".gz", ".json", ".txt")

NOTICE = """Files named at-* in this directory are copied unmodified from apache/arrow-testing
(https://github.com/apache/arrow-testing, revision {rev}), directories
data/arrow-ipc-stream, data/arrow-ipc-file and data/arrow-ipc-tensor-stream.
They are licensed under the Apache License, Version 2.0
(https://www.apache.org/licenses/LICENSE-2.0); see LICENSE.txt in that repository.

at-stream-*, at-file-*, at-tensor-*: ClusterFuzz reproducers that once crashed or misled the
Arrow C++ IPC readers (mostly invalid input). at-<arrow version>-*: valid integration fixtures
written by that Arrow version (incl. compression, shared dictionaries, big-endian).

Regenerate with: python -I tools/fuzz_import_seeds.py <arrow-testing checkout>
Other files: gen-* come from `zig build fuzz-gen-seeds`, regress-* are fixed zarrow findings.
"""


def revision(testing):
    git = testing / ".git"
    try:
        head = (git / "HEAD").read_text().strip()
        if head.startswith("ref: "):
            ref = head[5:]
            p = git / ref
            if p.exists():
                return p.read_text().strip()
            for line in (git / "packed-refs").read_text().splitlines():
                if line.endswith(" " + ref):
                    return line.split()[0]
        return head
    except OSError:
        return "unknown"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arrow_testing", type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "fuzz" / "corpus")
    ap.add_argument("--max-size", type=int, default=32 << 10)
    args = ap.parse_args()
    data = args.arrow_testing / "data"
    if not data.is_dir():
        sys.exit(f"{data} is not a directory")
    seen = {}
    copied = skipped = 0
    for corpus, sub, pattern, prefix in SOURCES:
        dst = args.out / corpus
        dst.mkdir(parents=True, exist_ok=True)
        if corpus not in seen:
            seen[corpus] = {hashlib.sha1(f.read_bytes()).digest() for f in dst.iterdir() if f.is_file()}
        for src in sorted((data / sub).glob(pattern)):
            if not src.is_file() or src.name.endswith(SKIP_SUFFIXES) or src.name.startswith("."):
                continue
            if src.stat().st_size > args.max_size:
                skipped += 1
                continue
            blob = src.read_bytes()
            h = hashlib.sha1(blob).digest()
            if h in seen[corpus]:
                continue
            seen[corpus].add(h)
            (dst / (prefix + src.name)).write_bytes(blob)
            copied += 1
    rev = revision(args.arrow_testing)
    for corpus in seen:
        (args.out / corpus / "NOTICE").write_text(NOTICE.format(rev=rev), encoding="utf-8")
    total = sum(f.stat().st_size for d in seen for f in (args.out / d).iterdir() if f.is_file())
    print(f"copied {copied} files, skipped {skipped} over {args.max_size} bytes; corpus now {total} bytes")


if __name__ == "__main__":
    main()
