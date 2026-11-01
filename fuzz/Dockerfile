# Linux fuzzing host for zarrow. Zig's coverage-guided fuzzer (`zig build fuzz-test --fuzz`) is
# Linux-only, and Linux also builds the vendored C codecs with UBSan. The image holds only the
# toolchain; the checkout is mounted at /work at run time, so findings and corpus growth land in
# the host's .fuzz-out. Zig's caches live in the /cache volume: Zig renames cache directories,
# which a bind mount of a Windows checkout refuses (AccessDenied).
#
#   docker build -f fuzz/Dockerfile -t zarrow-fuzz fuzz
#   docker run --rm -v "$PWD:/work" -v zarrow-fuzz-cache:/cache zarrow-fuzz [seconds per target] [coverage-guided iterations]
# or simply: sh tools/fuzz_docker.sh 600 2M      (see docs/fuzzing.md)
#
# ZIG_VERSION may be 0.16.0 or 0.17.0; the tarball URL and sha256 come from ziglang.org's index.
ARG ZIG_VERSION=0.17.0
FROM debian:bookworm-slim
ARG ZIG_VERSION
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl xz-utils python3 \
    && rm -rf /var/lib/apt/lists/*
RUN set -eux; cd /tmp; \
    curl -fsSL https://ziglang.org/download/index.json -o index.json; \
    python3 -c "import json;v=json.load(open('index.json'))['$ZIG_VERSION']['x86_64-linux'];print(v['tarball']);print(v['shasum'])" > zig.txt; \
    curl -fsSL "$(sed -n 1p zig.txt)" -o zig.tar.xz; \
    echo "$(sed -n 2p zig.txt)  zig.tar.xz" | sha256sum -c -; \
    mkdir -p /opt/zig && tar -xJf zig.tar.xz -C /opt/zig --strip-components=1; \
    rm -f zig.tar.xz index.json zig.txt
ENV PATH=/opt/zig:$PATH \
    ZIG_LOCAL_CACHE_DIR=/cache/local \
    ZIG_GLOBAL_CACHE_DIR=/cache/global
VOLUME /cache
WORKDIR /work
ENTRYPOINT ["sh", "fuzz/docker_entry.sh"]
