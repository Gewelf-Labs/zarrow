#!/bin/sh
# Entrypoint of fuzz/Dockerfile: the full Linux campaign against the checkout mounted at /work.
#   docker run --rm -v "$PWD:/work" zarrow-fuzz [seconds per target] [coverage-guided iterations]
# 1. replay the committed corpus, 2. mutation campaign on every target (ReleaseSafe keeps all
# safety checks), 3. Zig's coverage-guided fuzzer, 4. triage report. Exit 1 if anything was found.
set -eu
SECS=${1:-300}
ITERS=${2:-1M}
zig version
zig build fuzz-corpus -j2
# The harness writes every input to <out>/current.bin before running it; on the bind mount that
# costs ~50x in exec/s, so the campaign writes to the volume and the results are copied back.
rm -rf /cache/fuzz-out
python3 tools/fuzz.py --optimize ReleaseSafe --out /cache/fuzz-out run --targets all --seconds "$SECS"
# ReleaseSafe: Zig 0.16 cannot build a Debug --fuzz test runner; 0.17 accepts either.
zig build fuzz-test --fuzz="$ITERS" -Doptimize=ReleaseSafe -j2 \
    || echo "fuzz-test --fuzz failed: a coverage-guided finding (see above) or an unsupported host"
rm -rf .fuzz-out && cp -r /cache/fuzz-out .fuzz-out
python3 tools/fuzz_triage.py .fuzz-out
if ls .fuzz-out/*/index.json >/dev/null 2>&1; then
    echo "findings in .fuzz-out: see the triage report above"
    exit 1
fi
