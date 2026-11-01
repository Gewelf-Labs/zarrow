#!/bin/sh
# Build the Linux fuzzing image and run the campaign against this checkout; results in .fuzz-out.
#   sh tools/fuzz_docker.sh [seconds per target] [coverage-guided iterations]     e.g. 600 2M
#   ZIG_VERSION=0.16.0 sh tools/fuzz_docker.sh                                     other compiler
set -eu
cd "$(dirname "$0")/.."
ZIG_VERSION=${ZIG_VERSION:-0.17.0}
docker build -q -f fuzz/Dockerfile --build-arg "ZIG_VERSION=$ZIG_VERSION" -t "zarrow-fuzz:$ZIG_VERSION" fuzz
# Zig caches persist in a named volume per compiler (the checkout mount cannot hold them).
docker run --rm -v "$PWD:/work" -v "zarrow-fuzz-cache-$ZIG_VERSION:/cache" "zarrow-fuzz:$ZIG_VERSION" "$@"
