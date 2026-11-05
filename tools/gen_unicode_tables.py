#!/usr/bin/env python3
"""Generate src/compute/unicode_tables.zig from a Unicode Character Database UnicodeData.txt.

Usage: python -I tools/gen_unicode_tables.py <path/to/UnicodeData.txt> [<unicode version>]

The input is treated as untrusted data: every field is parsed strictly and range-checked, and the
script fails on anything unexpected. The output is deterministic (same input, same bytes).

Tables (see the header of the generated file):
- the general category of every code point, as runs `start << 5 | category`;
- the simple (one code point to one) uppercase and lowercase mappings, as ranges of code points
  sharing one delta, with stride 1 or 2 (alternating upper / lower pairs);
- for each table, a bucket index over code points below 0x20000 (128 code points per bucket):
  the entry holding the first code point of each bucket, so a lookup binary-searches only
  between two neighbouring bucket entries.

Arrow C++ maps case through utf8proc with simple mappings only. Its one difference from
UnicodeData.txt (checked against pyarrow 26 over every code point) is U+00DF, whose uppercase is
U+1E9E; this script applies it. Special casing (SpecialCasing.txt: one code point to several, or
context such as the final sigma) is out of scope, as in Arrow.
"""
import re
import sys
from pathlib import Path

CATEGORIES = [
    "Cn", "Lu", "Ll", "Lt", "Lm", "Lo", "Mn", "Mc", "Me", "Nd", "Nl", "No", "Pc", "Pd", "Ps",
    "Pe", "Pi", "Pf", "Po", "Sm", "Sc", "Sk", "So", "Zs", "Zl", "Zp", "Cc", "Cf", "Cs", "Co",
]
MAX_CP = 0x10FFFF
BUCKET_SHIFT = 7
BUCKET_LIMIT = 0x20000
# Arrow C++ (utf8proc) uppercase that differs from UnicodeData.txt.
UPPER_OVERRIDES = {0xDF: 0x1E9E}
HEX = re.compile(r"[0-9A-F]{4,6}")


def fail(msg: str) -> None:
    raise SystemExit(f"gen_unicode_tables: {msg}")


def parse_cp(field: str, lineno: int) -> int:
    if not HEX.fullmatch(field):
        fail(f"line {lineno}: bad code point {field!r}")
    cp = int(field, 16)
    if cp > MAX_CP:
        fail(f"line {lineno}: code point {field} out of range")
    return cp


def parse(path: Path):
    cat = [0] * (MAX_CP + 1)  # Cn by default
    upper: dict[int, int] = {}
    lower: dict[int, int] = {}
    first = None
    prev = -1
    raw = path.read_bytes()
    if len(raw) > 16 * 1024 * 1024:
        fail("input larger than 16 MiB")
    for lineno, line in enumerate(raw.decode("ascii").splitlines(), 1):
        fields = line.split(";")
        if len(fields) != 15:
            fail(f"line {lineno}: expected 15 fields")
        cp = parse_cp(fields[0], lineno)
        if cp <= prev:
            fail(f"line {lineno}: code points not increasing")
        prev = cp
        if fields[2] not in CATEGORIES or fields[2] == "Cn":
            fail(f"line {lineno}: bad category {fields[2]!r}")
        c = CATEGORIES.index(fields[2])
        name = fields[1]
        if name.endswith(", First>"):
            if first is not None:
                fail(f"line {lineno}: nested range")
            first = (cp, c)
            continue
        lo = cp
        if name.endswith(", Last>"):
            if first is None or first[1] != c:
                fail(f"line {lineno}: range end without matching start")
            lo = first[0]
            first = None
        elif first is not None:
            fail(f"line {lineno}: range start without end")
        for x in range(lo, cp + 1):
            cat[x] = c
        for field, table in ((fields[12], upper), (fields[13], lower)):
            if field:
                table[cp] = parse_cp(field, lineno)
    if first is not None:
        fail("unterminated range")
    upper.update(UPPER_OVERRIDES)
    return cat, upper, lower


def utf8_len(cp: int) -> int:
    return 1 if cp < 0x80 else 2 if cp < 0x800 else 3 if cp < 0x10000 else 4


def check_mapping(table: dict[int, int], what: str) -> None:
    for src, dst in table.items():
        if 0xD800 <= src <= 0xDFFF or 0xD800 <= dst <= 0xDFFF:
            fail(f"{what} mapping touches a surrogate: {src:X} -> {dst:X}")
        # The kernels reserve 3/2 of the input bytes for a case-mapped row.
        if 2 * utf8_len(dst) > 3 * utf8_len(src):
            fail(f"{what} mapping {src:X} -> {dst:X} grows past 3/2 of its UTF-8 length")
        if src < 0x80 and dst >= 0x80:
            fail(f"{what} mapping {src:X} -> {dst:X} leaves ASCII (the kernels' fast path assumes not)")
        if src >= BUCKET_LIMIT:
            fail(f"{what} mapping {src:X} past the bucket index")


def category_runs(cat: list[int]) -> list[tuple[int, int]]:
    runs = []
    for cp, c in enumerate(cat):
        if not runs or runs[-1][1] != c:
            runs.append((cp, c))
    return runs


def case_ranges(table: dict[int, int]) -> list[tuple[int, int, int, int]]:
    """(first, last, stride, delta): every stride-th code point of [first, last] maps by delta."""
    out: list[list[int]] = []
    for cp in sorted(table):
        delta = table[cp] - cp
        if out:
            r = out[-1]
            gap = cp - r[1]
            # A stride-2 step must skip an unmapped code point, so ranges never interleave.
            fits = gap == 1 or (gap == 2 and cp - 1 not in table)
            if r[3] == delta and fits and (r[0] == r[1] or gap == r[2]):
                r[1] = cp
                r[2] = gap
                continue
        out.append([cp, cp, 1, delta])
    return [tuple(r) for r in out]


def buckets(starts: list[int]) -> list[int]:
    """For every bucket b (and one past the last), the index of the last entry starting at or
    before the bucket's first code point (0 when none)."""
    out = []
    k = 0
    for b in range((BUCKET_LIMIT >> BUCKET_SHIFT) + 1):
        start = b << BUCKET_SHIFT
        while k + 1 < len(starts) and starts[k + 1] <= start:
            k += 1
        out.append(k)
    return out


def lookup(starts: list[int], bucket: list[int], cp: int) -> int:
    """The kernels' search: the last entry starting at or before `cp`."""
    if cp < BUCKET_LIMIT:
        b = cp >> BUCKET_SHIFT
        lo, hi = bucket[b], bucket[b + 1]
    else:
        lo, hi = bucket[-1], len(starts) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if starts[mid] <= cp:
            lo = mid
        else:
            hi = mid - 1
    return lo


def verify(cat, upper, lower, runs, ups, lows) -> None:
    """Decode the tables back through the kernels' search and compare with the parsed data for
    every code point."""
    starts = [s for s, _ in runs]
    cb = buckets(starts)
    for cp in range(MAX_CP + 1):
        if runs[lookup(starts, cb, cp)][1] != cat[cp]:
            fail(f"category lookup disagrees at {cp:X}")
    for ranges, table in ((ups, upper), (lows, lower)):
        firsts = [r[0] for r in ranges]
        rb = buckets(firsts)
        for cp in range(BUCKET_LIMIT):
            first, last, stride, delta = ranges[lookup(firsts, rb, cp)]
            hit = first <= cp <= last and (cp - first) % stride == 0
            if (cp + delta if hit else cp) != table.get(cp, cp):
                fail(f"case lookup disagrees at {cp:X}")


def emit_ints(values: list[str], per_line: int) -> str:
    """Rows of `per_line` values, columns padded like `zig fmt` (to the widest value of the
    column, no trailing space), so the output is already formatted."""
    rows = [values[i:i + per_line] for i in range(0, len(values), per_line)]
    widths = [max(len(r[c]) for r in rows if c < len(r)) for c in range(per_line)]
    lines = []
    for r in rows:
        cells = [(v + ",").ljust(widths[c] + 2) for c, v in enumerate(r)]
        lines.append(("    " + "".join(cells)).rstrip())
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    if len(argv) not in (1, 2):
        print(__doc__.splitlines()[2], file=sys.stderr)
        return 2
    version = argv[1] if len(argv) == 2 else "17.0.0"
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        fail(f"bad version {version!r}")
    cat, upper, lower = parse(Path(argv[0]))
    check_mapping(upper, "upper")
    check_mapping(lower, "lower")
    runs = category_runs(cat)
    ups = case_ranges(upper)
    lows = case_ranges(lower)
    verify(cat, upper, lower, runs, ups, lows)
    for r in ups + lows:
        if not -(1 << 20) <= r[3] < (1 << 20):
            fail("delta does not fit in 21 bits")
    if len(runs) >= 1 << 16 or len(ups) >= 1 << 16 or len(lows) >= 1 << 16:
        fail("table too long for u16 bucket indices")

    out = []
    out.append(f"""//! Unicode {version} tables of the utf8_* kernels: general categories and simple case mappings.
//! Generated by `python -I tools/gen_unicode_tables.py UnicodeData.txt {version}` from the Unicode
//! Character Database file UnicodeData.txt (Unicode {version}). Do not edit; regenerate.
//!
//! The uppercase of U+00DF is U+1E9E, as in Arrow C++ (utf8proc); every other mapping is the
//! simple mapping of UnicodeData.txt. Special casing (one code point to several) is not included.
//! Every mapping stays a Unicode scalar value, keeps ASCII in ASCII and at most grows a code
//! point's UTF-8 length from 2 to 3 bytes (the generator checks all three).

/// Unicode version of the tables.
pub const unicode_version = "{version}";

/// General category (UnicodeData.txt field 2); `Cn` (unassigned) is 0.
pub const Category = enum(u5) {{ {", ".join(CATEGORIES)} }};

/// Every `stride`-th code point of `[first, last]` maps to itself plus `delta`.
pub const CaseRange = packed struct(u64) {{
    first: u21,
    last: u21,
    /// 0: stride 1, 1: stride 2.
    stride2: u1,
    delta: i21,
}};

/// Bucket index: code points below `bucket_limit` in buckets of `1 << bucket_shift`.
pub const bucket_shift = {BUCKET_SHIFT};
pub const bucket_limit = 0x{BUCKET_LIMIT:X};
""")
    out.append(f"/// {len(runs)} runs `start << 5 | category`, sorted; run k spans [start k, start k+1).")
    out.append(f"pub const category_runs = [{len(runs)}]u32{{")
    out.append(emit_ints([f"0x{(s << 5) | c:07X}" for s, c in runs], 8))
    out.append("};")
    for name, ranges in (("upper", ups), ("lower", lows)):
        out.append(f"\n/// {len(ranges)} simple {name}case ranges, sorted and disjoint.")
        out.append(f"pub const {name}_ranges = [{len(ranges)}]CaseRange{{")
        for first, last, stride, delta in ranges:
            out.append(f"    .{{ .first = 0x{first:X}, .last = 0x{last:X}, .stride2 = {1 if stride == 2 else 0}, .delta = {delta} }},")
        out.append("};")
    for name, starts in (("category", [s for s, _ in runs]), ("upper", [r[0] for r in ups]), ("lower", [r[0] for r in lows])):
        b = buckets(starts)
        out.append(f"\n/// For bucket b (and one past the last), the last {name} entry starting at or before b << bucket_shift.")
        out.append(f"pub const {name}_buckets = [{len(b)}]u16{{")
        out.append(emit_ints([str(x) for x in b], 16))
        out.append("};")
    target = Path(__file__).resolve().parent.parent / "src" / "compute" / "unicode_tables.zig"
    target.write_bytes(("\n".join(out) + "\n").encode("ascii"))
    print(f"wrote {target} ({len(runs)} category runs, {len(ups)} upper and {len(lows)} lower ranges)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
