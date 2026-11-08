#!/usr/bin/env python3
"""Generate src/compute/unicode_tables.zig (and src/compute/unicode_props.zig) from the Unicode
Character Database.

Usage: python -I tools/gen_unicode_tables.py <UnicodeData.txt> [<version> [<CompositionExclusions.txt> <Scripts.txt>]]

Download the UCD files into their own empty directory first; they are only read as text.

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



# ------------------------------------------------------------------ normalization and scripts
#
# With CompositionExclusions.txt and Scripts.txt (same UCD version) the script also writes
# src/compute/unicode_props.zig: the tables of utf8_normalize (canonical combining classes,
# full canonical and compatibility decompositions, canonical compositions, quick-check flags)
# and the script runs of the regex `\p{Script}` classes.

NORM_SHIFT = 7
NORM_LIMIT = 0x30000
# A code point's full compatibility decomposition (NFKD) has at most this many code points
# (U+FDFA); the kernels size their buffers by it.
MAX_DECOMP = 18
# Hangul syllables decompose algorithmically (UAX #15, section 16).
HANGUL_S = range(0xAC00, 0xD7A4)


def parse_norm(path: Path):
    """Canonical combining classes and decomposition mappings of UnicodeData.txt."""
    ccc: dict[int, int] = {}
    canon: dict[int, list[int]] = {}
    compat: dict[int, list[int]] = {}
    raw = path.read_bytes()
    for lineno, line in enumerate(raw.decode("ascii").splitlines(), 1):
        fields = line.split(";")
        cp = parse_cp(fields[0], lineno)
        if not re.fullmatch(r"\d{1,3}", fields[3]) or int(fields[3]) > 254:
            fail(f"line {lineno}: bad combining class {fields[3]!r}")
        if int(fields[3]):
            if fields[1].endswith(("First>", "Last>")):
                fail(f"line {lineno}: combining class on a range")
            ccc[cp] = int(fields[3])
        d = fields[5]
        if not d:
            continue
        tag = None
        if d.startswith("<"):
            m = re.fullmatch(r"<([a-zA-Z]+)> (.*)", d)
            if not m:
                fail(f"line {lineno}: bad decomposition tag")
            tag, d = m.group(1), m.group(2)
        cps = [parse_cp(x, lineno) for x in d.split(" ")]
        if not 1 <= len(cps) <= MAX_DECOMP:
            fail(f"line {lineno}: decomposition of {len(cps)} code points")
        (compat if tag else canon)[cp] = cps
    return ccc, canon, compat


def parse_exclusions(path: Path) -> set[int]:
    out = set()
    raw = path.read_bytes()
    if len(raw) > 1024 * 1024:
        fail("CompositionExclusions.txt larger than 1 MiB")
    for lineno, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        body = line.split("#", 1)[0].strip()
        if body:
            out.add(parse_cp(body, lineno))
    return out


def parse_scripts(path: Path) -> tuple[list[str], list[int]]:
    """Script names (index 0: Unknown, then sorted) and the script of every code point."""
    raw = path.read_bytes()
    if len(raw) > 4 * 1024 * 1024:
        fail("Scripts.txt larger than 4 MiB")
    entries = []
    for lineno, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        body = line.split("#", 1)[0].strip()
        if not body:
            continue
        m = re.fullmatch(r"([0-9A-F]{4,6})(?:\.\.([0-9A-F]{4,6}))?\s*;\s*([A-Za-z_]+)", body)
        if not m:
            fail(f"Scripts.txt line {lineno}: unexpected {body!r}")
        lo = parse_cp(m.group(1), lineno)
        hi = parse_cp(m.group(2), lineno) if m.group(2) else lo
        if hi < lo:
            fail(f"Scripts.txt line {lineno}: empty range")
        entries.append((lo, hi, m.group(3)))
    names = ["Unknown"] + sorted({e[2] for e in entries} - {"Unknown"})
    if len(names) > 255:
        fail("too many scripts")
    script = [0] * (MAX_CP + 1)
    for lo, hi, name in entries:
        for cp in range(lo, hi + 1):
            if script[cp]:
                fail(f"Scripts.txt: {cp:X} listed twice")
            script[cp] = names.index(name)
    return names, script


def full_decomp(cp: int, canon, compat, compat_too: bool) -> list[int]:
    if 0xAC00 <= cp <= 0xD7A3:
        fail("a Hangul syllable has a decomposition field")
    d = canon.get(cp) or (compat.get(cp) if compat_too else None)
    if d is None:
        return [cp]
    out = []
    for x in d:
        out += full_decomp(x, canon, compat, compat_too)
    return out


def norm_tables(ccc, canon, compat, exclusions):
    # Full_Composition_Exclusion: listed exclusions, singletons and non-starter decompositions.
    full_excl = set(exclusions)
    for cp, d in canon.items():
        if len(d) == 1 or ccc.get(cp, 0) or ccc.get(d[0], 0):
            full_excl.add(cp)
    pairs = {}
    for cp, d in canon.items():
        if cp in full_excl:
            continue
        if len(d) != 2:
            fail(f"primary composite {cp:X} does not decompose to a pair")
        pairs[(d[0] << 21) | d[1]] = cp
    seconds = {k & ((1 << 21) - 1) for k in pairs}
    # Hangul vowel and trailing consonant jamo compose algorithmically with the code point before.
    seconds |= set(range(0x1161, 0x1176)) | set(range(0x11A8, 0x11C3))
    decomp = []  # (cp, canonical full, compatibility full)
    for cp in sorted(set(canon) | set(compat)):
        nfd = full_decomp(cp, canon, compat, False)
        nfkd = full_decomp(cp, canon, compat, True)
        if len(nfkd) > MAX_DECOMP or len(nfd) > 4:
            fail(f"decomposition of {cp:X} too long")
        decomp.append((cp, nfd if cp in canon else [cp], nfkd))
    props = [0] * NORM_LIMIT
    for cp in range(MAX_CP + 1):
        v = ccc.get(cp, 0)
        if cp in canon or cp in HANGUL_S:
            v |= 1 << 8
        # Bit 9: the compatibility decomposition differs from the canonical one (also through a
        # canonical mapping to a compatibility character, U+0385 -> U+00A8 U+0301).
        if (cp in canon or cp in compat) and full_decomp(cp, canon, compat, True) != full_decomp(cp, canon, compat, False):
            v |= 1 << 9
        if cp in full_excl and cp in canon:
            v |= 1 << 10
        if cp in seconds:
            v |= 1 << 11
        if cp >= NORM_LIMIT:
            if v:
                fail(f"normalization property past 0x{NORM_LIMIT:X} at {cp:X}")
            continue
        props[cp] = v
    blocks: list[tuple[int, ...]] = []
    index = []
    for b in range(NORM_LIMIT >> NORM_SHIFT):
        blk = tuple(props[b << NORM_SHIFT:(b + 1) << NORM_SHIFT])
        if blk not in blocks:
            blocks.append(blk)
        index.append(blocks.index(blk))
    if len(blocks) > 256:
        fail("too many property blocks")
    return pairs, decomp, index, blocks


def emit_props(version: str, ccc, canon, compat, exclusions, names, script) -> None:
    pairs, decomp, index, blocks = norm_tables(ccc, canon, compat, exclusions)
    data: list[int] = []
    entries = []
    for cp, nfd, nfkd in decomp:
        off = len(data)
        data += nfd
        koff = off if nfkd == nfd else len(data)
        if nfkd != nfd:
            data += nfkd
        entries.append((cp, off, len(nfd), koff, len(nfkd)))
    if len(data) >= 1 << 16:
        fail("decomposition data past 16-bit offsets")
    runs = category_runs(script)
    out = [f"""//! Unicode {version} normalization and script tables (utf8_normalize, regex `\\p{{Script}}`).
//! Generated by `python -I tools/gen_unicode_tables.py UnicodeData.txt {version}
//! CompositionExclusions.txt Scripts.txt` from the Unicode Character Database (Unicode {version}).
//! Do not edit; regenerate.
//!
//! Hangul syllables are not listed: the kernels decompose and compose them algorithmically.

/// Unicode version of the tables.
pub const unicode_version = "{version}";

/// Most code points one code point decomposes to (NFKD of U+FDFA); buffers are sized by it.
pub const max_decomposition = {MAX_DECOMP};

/// Properties of a code point: canonical combining class (bits 0-7), a canonical decomposition
/// (bit 8, Hangul syllables included), a compatibility decomposition that differs from the
/// canonical one (bit 9), excluded from
/// composition (bit 10: Full_Composition_Exclusion of a canonically decomposable code point),
/// second code point of a primary composite or a Hangul vowel / trailing jamo (bit 11). Code points from `props_limit` on have none.
pub const props_shift = {NORM_SHIFT};
pub const props_limit = 0x{NORM_LIMIT:X};

/// Block of every `1 << props_shift` code points below `props_limit`.
pub const props_index = [{len(index)}]u8{{"""]
    out.append(emit_ints([str(x) for x in index], 24))
    out.append("};")
    out.append(f"\n/// {len(blocks)} distinct blocks of `1 << props_shift` properties.")
    out.append(f"pub const props_blocks = [{len(blocks) << NORM_SHIFT}]u16{{")
    out.append(emit_ints([str(x) for blk in blocks for x in blk], 24))
    out.append("};")
    out.append(f"""
/// A decomposable code point (a `u64` of `decompositions`): its full canonical decomposition
/// (the code point itself when it has only a compatibility one) and its full compatibility
/// decomposition, as `[offset, offset + len)` of `decomposition_data`.
pub const Decomposition = packed struct(u64) {{ cp: u21, canon_off: u16, canon_len: u5, compat_off: u16, compat_len: u5, _: u1 = 0 }};

/// {len(entries)} decompositions (`Decomposition` bits), sorted by code point.
pub const decompositions = [{len(entries)}]u64{{""")
    out.append(emit_ints([f"0x{cp | off << 21 | n << 37 | koff << 42 | kn << 58:X}" for cp, off, n, koff, kn in entries], 6))
    out.append("};")
    out.append(f"\npub const decomposition_data = [{len(data)}]u21{{")
    out.append(emit_ints([f"0x{x:X}" for x in data], 12))
    out.append("};")
    keys = sorted(pairs)
    out.append(f"\n/// {len(keys)} canonical compositions: `first << 21 | second`, sorted, and the composite.")
    out.append(f"pub const composition_keys = [{len(keys)}]u64{{")
    out.append(emit_ints([f"0x{k:X}" for k in keys], 6))
    out.append("};")
    out.append(f"pub const composition_values = [{len(keys)}]u21{{")
    out.append(emit_ints([f"0x{pairs[k]:X}" for k in keys], 12))
    out.append("};")
    out.append(f"\n/// Script names (Scripts.txt); index 0 is Unknown (unassigned and unlisted code points).")
    out.append(f"pub const script_names = [{len(names)}][]const u8{{")
    out.append(emit_ints([f'"{n}"' for n in names], 6))
    out.append("};")
    out.append(f"\n/// {len(runs)} runs `start << 8 | script`, sorted; run k spans [start k, start k+1).")
    out.append(f"pub const script_runs = [{len(runs)}]u32{{")
    out.append(emit_ints([f"0x{(s << 8) | c:08X}" for s, c in runs], 8))
    out.append("};")
    target = Path(__file__).resolve().parent.parent / "src" / "compute" / "unicode_props.zig"
    target.write_bytes(("\n".join(out) + "\n").encode("ascii"))
    print(f"wrote {target} ({len(entries)} decompositions, {len(keys)} compositions, {len(blocks)} property blocks, {len(runs)} script runs)")


def main(argv: list[str]) -> int:
    if len(argv) not in (1, 2, 4):
        print(__doc__.splitlines()[2], file=sys.stderr)
        return 2
    version = argv[1] if len(argv) >= 2 else "17.0.0"
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
    if len(argv) == 4:
        ccc, canon, compat = parse_norm(Path(argv[0]))
        names, script = parse_scripts(Path(argv[3]))
        emit_props(version, ccc, canon, compat, parse_exclusions(Path(argv[2])), names, script)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
