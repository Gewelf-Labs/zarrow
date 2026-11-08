#!/usr/bin/env python3
"""Fuzz supervisor for zarrow (stdlib only). See docs/fuzzing.md.

  python tools/fuzz.py run --targets all --seconds 600 --out .fuzz-out
  python tools/fuzz.py replay --target ipc-stream --out .fuzz-out <file|dir>...

`run` loops the harness (`zarrow-fuzz <target> run`) in chunks; `replay` runs every input in its
own process. A non-zero exit / signal is a crash; the harness watchdog (exit 77, one input over
--timeout s) or a child outliving its chunk is a hang; RSS over --rss-mb is an OOM. A crash in
`run` leaves its input in <out>/<target>/current.bin.
Each finding is saved once per signature (kind + panic message + top 3 frames) as
<out>/<target>/<sha1-16>.bin + .txt, with counts in <out>/<target>/index.json.
"""
import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

TARGETS = ["ipc-stream", "ipc-stream-zero-copy", "ipc-file", "array-layout", "compute", "c-abi", "regex", "temporal"]
CORPUS = {"ipc-stream-zero-copy": "ipc-stream", "compute": "ipc-stream"}
ROOT = Path(__file__).resolve().parent.parent
EXE = ROOT / "zig-out" / "bin" / ("zarrow-fuzz.exe" if os.name == "nt" else "zarrow-fuzz")
FRAME = re.compile(r"^(?P<path>.+?):(?P<line>\d+)(?::\d+)?: 0x[0-9a-f]+ in (?P<fn>[^ ]+)")
SKIP_FN = {"charge", "alloc", "resize", "remap", "free", "expectValid", "f"}


def frames(trace):
    """Stack frames outside the std library, as (repo-relative path, line, fn)."""
    out = []
    for line in trace.splitlines():
        m = FRAME.match(line.strip())
        if not m:
            continue
        path = m["path"].replace("\\", "/")
        if "/lib/std/" in path or "/lib/compiler" in path or "/libc/" in path or path.startswith("???"):
            continue
        for anchor in ("/src/", "/fuzz/", "/tools/"):
            if anchor in path:
                path = path[path.rindex(anchor) + 1:]
                break
        out.append((path, int(m["line"]), re.sub(r"__(func|anon)_\d+", "", m["fn"])))
    return out


def crash_frames(trace):
    """Frames of the panic's own stack; for leaks (whose stack is just the harness) the error
    return context instead, which names the failing path that skipped its cleanup."""
    stack = frames(trace.split("stack trace:", 1)[-1])
    if any(p.startswith("src/") for p, _, _ in stack):
        return stack
    return frames(trace)


def signature(kind, trace):
    msg = ""
    for line in trace.splitlines():
        if " panic: " in line or line.startswith(("panic: ", "Segmentation fault", "General protection",
                                                    "Illegal instruction", "Bus error", "fuzz-hang")):
            msg = line.split("panic: ", 1)[-1]
            break
    msg = re.sub(r"0x[0-9a-f]+|\d+", "#", msg)[:120]
    top = [f"{fn}@{p}" for p, _, fn in crash_frames(trace) if not (p.startswith("fuzz/") and fn in SKIP_FN)][:3]
    return f"{kind} | {msg} | " + " < ".join(top), msg


def save(out, target, kind, data, trace):
    tdir = out / target
    tdir.mkdir(parents=True, exist_ok=True)
    index_path = tdir / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    sig, msg = signature(kind, trace)
    h = hashlib.sha1(data).hexdigest()[:16]
    (tdir / "skip").mkdir(exist_ok=True)
    (tdir / "skip" / h).touch()  # the harness skips this input when it reloads the corpus
    entry = index.setdefault(sig, {"kind": kind, "message": msg, "count": 0, "smallest": None, "size": None})
    entry["count"] += 1
    if entry["smallest"] is None or len(data) < entry["size"]:
        if entry["smallest"]:
            for ext in (".bin", ".txt"):
                (tdir / (entry["smallest"] + ext)).unlink(missing_ok=True)
        (tdir / f"{h}.bin").write_bytes(data)
        (tdir / f"{h}.txt").write_text(f"signature: {sig}\n\n{trace}", encoding="utf-8")
        entry.update(smallest=h, size=len(data), frames=crash_frames(trace)[:10])
    index_path.write_text(json.dumps(index, indent=1))
    new = entry["count"] == 1
    print(f"[{target}] {'NEW ' if new else ''}{kind}: {sig} -> {tdir / (entry['smallest'] + '.bin')}", flush=True)
    return new


def rss_bytes(pid):
    try:
        if sys.platform.startswith("linux"):
            for line in open(f"/proc/{pid}/status"):
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
        elif os.name == "nt":
            import ctypes
            from ctypes import wintypes

            class PMC(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                    (n, ctypes.c_size_t) for n in ("Peak", "WorkingSetSize", "a", "b", "c", "d", "e", "f")]

            h = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED_INFORMATION
            pmc = PMC()
            pmc.cb = ctypes.sizeof(PMC)
            ok = ctypes.windll.psapi.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb)
            ctypes.windll.kernel32.CloseHandle(h)
            return pmc.WorkingSetSize if ok else 0
    except (OSError, ValueError, AttributeError):
        pass
    return 0


def supervise(cmd, args, wall):
    """Runs cmd; returns (kind or None, stderr text). Kills on wall-time overrun / RSS cap."""
    log = args.out / "child.log"
    with open(log, "wb") as err:
        p = subprocess.Popen(cmd + ["--timeout", str(int(args.timeout))], stdout=subprocess.DEVNULL, stderr=err, cwd=ROOT)
        start = time.monotonic()
        kind = None
        while p.poll() is None:
            time.sleep(0.25)
            if time.monotonic() - start > wall:
                kind = "hang"
            elif rss_bytes(p.pid) > args.rss_mb << 20:
                kind = "oom-rss"
            if kind:
                p.kill()
                p.wait()
                break
    text = log.read_text(encoding="utf-8", errors="replace")
    if kind is None and p.returncode != 0:
        kind = "hang" if p.returncode == 77 else "oom" if "panic: fuzz-oom" in text else \
            "leak" if "panic: fuzz-leak" in text else "crash"
    return kind, text


def build(args):
    if args.no_build:
        return
    cmd = ["zig", "build", "fuzz-install", f"-Doptimize={args.optimize}", "-j2"] + args.zig_args
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def cmd_run(args):
    targets = TARGETS if args.targets == "all" else args.targets.split(",")
    build(args)
    for target in targets:
        tdir = args.out / target
        tdir.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + args.seconds
        execs_note = 0
        while (left := deadline - time.monotonic()) > 1:
            chunk = int(min(args.chunk, left))
            seed = random.randrange(1 << 62)
            corpus = ROOT / "fuzz" / "corpus" / CORPUS.get(target, target)
            cmd = [str(args.exe), target, "run", f"{chunk}s", "--corpus", str(corpus), "--out", str(tdir),
                   "--seed", str(seed), "--max-len", str(args.max_len)]
            kind, text = supervise(cmd, args, wall=chunk + 30 + 2 * args.timeout)
            execs_note += 1
            if kind:
                data = (tdir / "current.bin").read_bytes() if (tdir / "current.bin").exists() else b""
                save(args.out, target, kind, data, text)
            else:
                print(f"[{target}] " + (text.strip().splitlines() or ["(no output)"])[-1], flush=True)
    return 0


def cmd_replay(args):
    build(args)
    files = []
    for p in map(Path, args.paths):
        files += sorted(f for f in p.iterdir() if f.is_file()) if p.is_dir() else [p]
    findings = 0
    for f in files:
        if f.stat().st_size > 1 << 20:
            continue
        kind, text = supervise([str(args.exe), args.target, "replay", str(f.resolve())], args, wall=2 * args.timeout + 5)
        if kind:
            findings += 1
            save(args.out, args.target, kind, f.read_bytes(), f"input: {f}\n" + text)
    print(f"[{args.target}] replayed {len(files)} files, {findings} findings", flush=True)
    return 1 if findings and args.fail else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / ".fuzz-out")
    ap.add_argument("--exe", type=Path, default=EXE)
    ap.add_argument("--optimize", default="Debug", help="Debug or ReleaseSafe: safety checks are our sanitizer")
    ap.add_argument("--no-build", action="store_true")
    ap.add_argument("--timeout", type=float, default=10.0, help="seconds without progress = hang")
    ap.add_argument("--rss-mb", type=int, default=2048)
    ap.add_argument("--zig-args", nargs="*", default=[])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--targets", default="all")
    r.add_argument("--seconds", type=int, default=600, help="per target")
    r.add_argument("--chunk", type=int, default=120, help="seconds per child process")
    r.add_argument("--max-len", type=int, default=64 << 10)
    p = sub.add_parser("replay")
    p.add_argument("--target", required=True, choices=TARGETS)
    p.add_argument("--fail", action="store_true", help="exit 1 if anything crashed")
    p.add_argument("paths", nargs="+")
    args = ap.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    return cmd_run(args) if args.cmd == "run" else cmd_replay(args)


if __name__ == "__main__":
    sys.exit(main())
