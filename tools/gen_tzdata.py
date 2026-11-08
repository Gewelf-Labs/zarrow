#!/usr/bin/env python3
"""Generate src/compute/tzdata/tzdata.bin, the timezone table zarrow bundles for Windows and for
systems without /usr/share/zoneinfo.

Input: the `tzdata` wheel from PyPI, which carries the zic output (TZif v2+ files) of one IANA
tzdata release. Download it into its own empty directory; this script only reads it as a zip
archive (nothing in it is imported or executed):

    mkdir dl && python -I -m pip download --no-deps --only-binary=:all: tzdata==2026.6 -d dl
    python -I tools/gen_tzdata.py dl/tzdata-2026.6-py2.py3-none-any.whl src/compute/tzdata/tzdata.bin

Output layout (little-endian; read by `src/compute/tz.zig`, which checks every bound):

    "ZTZB"  u8 version_len  version (the IANA release, e.g. "2026f")  u32 zone_count
    zone_count x (u8 name_len, name, u32 offset, u32 length)     sorted by name
    TZif blobs at those offsets: each zone file with its version 1 data block dropped (the
    first header's counts set to zero), identical blobs stored once
"""
from __future__ import annotations

import re
import struct
import sys
import zipfile

MAX_FILE = 256 << 10


def strip_v1(data: bytes) -> bytes:
    """The TZif file without its 32-bit data block (the first header keeps its magic only)."""
    if len(data) < 44 or data[:4] != b"TZif" or data[4] not in b"234":
        raise ValueError("not a TZif v2+ file")
    isut, isstd, leap, timecnt, typecnt, charcnt = struct.unpack(">6I", data[20:44])
    v1 = timecnt * 5 + typecnt * 6 + charcnt + leap * 8 + isstd + isut
    rest = data[44 + v1:]
    if len(rest) < 44 or rest[:4] != b"TZif":
        raise ValueError("missing version 2 header")
    return data[:20] + bytes(24) + rest


def main() -> None:
    wheel, out_path = sys.argv[1], sys.argv[2]
    with zipfile.ZipFile(wheel) as z:
        init = z.read("tzdata/__init__.py").decode("utf-8")
        m = re.search(r'^IANA_VERSION = "(\d{4}[a-z])"$', init, re.M)
        if not m:
            sys.exit("no IANA_VERSION in tzdata/__init__.py")
        version = m.group(1).encode()
        zones = {}
        for info in z.infolist():
            name = info.filename
            if not name.startswith("tzdata/zoneinfo/") or name.endswith("/") or "." in name.rsplit("/", 1)[1]:
                continue
            zone = name[len("tzdata/zoneinfo/"):]
            if info.file_size > MAX_FILE or not re.fullmatch(r"[A-Za-z0-9_+\-]+(/[A-Za-z0-9_+\-]+)*", zone) or len(zone) > 255:
                sys.exit(f"unexpected zone file {name!r}")
            data = z.read(info)
            if not data.startswith(b"TZif"):
                continue  # tzdata.zi / zones list and the like
            zones[zone] = strip_v1(data)
    names = sorted(zones)
    header = b"ZTZB" + bytes([len(version)]) + version + struct.pack("<I", len(names))
    index_len = sum(1 + len(n) + 8 for n in names)
    blobs: dict[bytes, int] = {}
    body = bytearray()
    base = len(header) + index_len
    index = bytearray()
    for n in names:
        blob = zones[n]
        if blob not in blobs:
            blobs[blob] = base + len(body)
            body += blob
        index += bytes([len(n)]) + n.encode() + struct.pack("<II", blobs[blob], len(blob))
    with open(out_path, "wb") as f:
        f.write(header + index + body)
    print(f"tzdata {version.decode()}: {len(names)} zones, {len(blobs)} distinct, {len(header) + len(index) + len(body)} bytes")


if __name__ == "__main__":
    main()
