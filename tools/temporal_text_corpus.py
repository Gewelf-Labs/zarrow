#!/usr/bin/env python3
"""Generate the temporal text corpus checked by src/compute/test_temporal_text.zig.

Runs strftime, strptime and the casts between the temporal types and string through pyarrow
and writes one line per case (text fields hex-encoded, `E` for an error):

    F <TAB> type <TAB> format-hex <TAB> value <TAB> result-hex|E    strftime
    C <TAB> type <TAB> value <TAB> result-hex|E                      cast to string
    P <TAB> type <TAB> text-hex <TAB> stored value|E                 cast from string
    S <TAB> unit <TAB> format-hex <TAB> text-hex <TAB> value|E       strptime

`type` is `ts:<unit>:<timezone>` (an empty timezone for a naive timestamp), `d32`, `d64`,
`t32:<unit>`, `t64:<unit>` or `dur:<unit>`; units are s, ms, us, ns.

pyarrow lives in the user site, so run it with `python -P` (isolated mode hides it):
    python -P tools/temporal_text_corpus.py src/compute/testdata/temporal_text_corpus.tsv
"""
from __future__ import annotations

import random
import sys

import pyarrow as pa
import pyarrow.compute as pc

EXPECTED_PYARROW = "26."
UNITS = ["s", "ms", "us", "ns"]
PER = {"s": 1, "ms": 1000, "us": 10**6, "ns": 10**9}
ZONES = ["", "UTC", "+05:30", "-0800", "+00:00", "-23:59"] + sys.argv[2:]


def arrow_type(code: str) -> pa.DataType:
    parts = code.split(":", 2)
    if parts[0] == "ts":
        return pa.timestamp(parts[1], parts[2] or None)
    if parts[0] == "d32":
        return pa.date32()
    if parts[0] == "d64":
        return pa.date64()
    if parts[0] == "t32":
        return pa.time32(parts[1])
    if parts[0] == "t64":
        return pa.time64(parts[1])
    return pa.duration(parts[1])


def hx(s: str) -> str:
    return s.encode("utf-8").hex()


def random_seconds(rng: random.Random) -> int:
    r = rng.random()
    if r < 0.6:  # years 1900..2100
        return rng.randrange(-2208988800, 4102444800)
    if r < 0.8:  # years 1..9999
        return rng.randrange(-62135596800, 253402300800)
    return rng.randrange(-1096193779200, 971890963200)  # years -32767..32767


def value_for(code: str, rng: random.Random) -> int:
    parts = code.split(":", 2)
    if parts[0] == "d32":
        return random_seconds(rng) // 86400
    if parts[0] == "d64":
        return random_seconds(rng) // 86400 * 86400000 + (rng.randrange(86400000) if rng.random() < 0.2 else 0)
    if parts[0] in ("t32", "t64"):
        per = PER[parts[1]]
        r = rng.random()
        if r < 0.1:
            return rng.choice([-1, 86400 * per, 86400 * per - 1, 0])
        return rng.randrange(86400 * per)
    if parts[0] == "dur":
        return rng.randrange(-2**40, 2**40)
    per = PER[parts[1]]
    lo, hi = -(2**63), 2**63 - 1
    v = random_seconds(rng) * per + rng.randrange(per)
    return min(max(v, lo), hi)


SPECIFIERS = list("aAbhBcCdDeFgGHIjmMnprRStTuUVwWxXyYzZ%") + [
    "Ec", "EC", "Ex", "EX", "Ey", "EY", "Ez", "Od", "Oe", "OH", "OI", "Om", "OM", "OS", "Ou",
    "OU", "OV", "Ow", "OW", "Oy", "Oz"]
FORMATS = ["%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S%z", "%d/%m/%Y %I:%M %p", "%A %B %e %Y", "%G-W%V-%u",
           "%j %U %W %w", "%C %y %g", "%F %T %Z", "%c", "%x %X %r", "plain text", "%%Y", "%Ez|%Oz",
           "%", "abc%", "%Q", "%Ea", "%OY", "%E", "%5Y", "%-d", "%s", "%f", "%i", "%k", "%l", "%P",
           "%E%", "%Oc", "%%z"] + ["%" + s for s in SPECIFIERS]


def strftime_cases(rng: random.Random, out: list[str]) -> None:
    codes = [f"ts:{u}:{z}" for u in UNITS for z in ZONES] + ["d32", "d64", "t32:s", "t32:ms", "t64:us", "t64:ns"]
    for code in codes:
        t = arrow_type(code)
        values = [value_for(code, rng) for _ in range(12)]
        for fmt in FORMATS:
            vs = values if fmt.startswith("%Y-%m-%dT") or fmt == "%c" else values[:3]
            for v in vs:  # one call per value: an error fails the whole call
                try:
                    r = pc.strftime(pa.array([v], t), format=fmt).to_pylist()[0]
                    out.append(f"F\t{code}\t{hx(fmt)}\t{v}\t{hx(r)}")
                except Exception:  # noqa: BLE001 - every pyarrow error is an expected E
                    out.append(f"F\t{code}\t{hx(fmt)}\t{v}\tE")


def cast_to_string_cases(rng: random.Random, out: list[str]) -> None:
    codes = [f"ts:{u}:{z}" for u in UNITS for z in ZONES] + ["d32", "d64"] + \
        [f"t32:{u}" for u in ["s", "ms"]] + [f"t64:{u}" for u in ["us", "ns"]] + [f"dur:{u}" for u in UNITS]
    for code in codes:
        t = arrow_type(code)
        values = [value_for(code, rng) for _ in range(40)]
        if code.startswith("ts:s") or code == "d32" or code == "d64":
            scale = 1 if code == "d32" else (86400000 if code == "d64" else 1)
            edge = [11248737, -12687428, 11248738, -12687429] if code == "d32" else \
                [971890963199 * (1000 if code == "d64" else 1), -1096193779200 * (1000 if code == "d64" else 1),
                 971890963200 * (1000 if code == "d64" else 1), -1096193779201 * (1000 if code == "d64" else 1)]
            values += [e * (scale if code == "d32" else 1) for e in edge]
        for v in values:
            try:
                r = pa.array([v], t).cast(pa.string()).to_pylist()[0]
                out.append(f"C\t{code}\t{v}\t{hx(r)}")
            except Exception:  # noqa: BLE001
                out.append(f"C\t{code}\t{v}\tE")


def mutate(rng: random.Random, s: str) -> str:
    choice = rng.randrange(6)
    if not s:
        return s
    i = rng.randrange(len(s))
    if choice == 0:
        return s[:i] + s[i + 1:]
    if choice == 1:
        return s[:i] + rng.choice("0123456789 T:-+.Zz/x") + s[i:]
    if choice == 2:
        return s[:i] + rng.choice("0123456789 T:-+.Zz") + s[i + 1:]
    if choice == 3:
        return s + rng.choice(["Z", "+01", "-0130", "+05:30", "+24:00", ".5", " ", "x"])
    if choice == 4:
        return s[:rng.randrange(len(s) + 1)]
    return s


def iso_texts(rng: random.Random) -> list[str]:
    texts = []
    for _ in range(300):
        secs = random_seconds(rng) if rng.random() < 0.5 else rng.randrange(-2208988800, 4102444800)
        tp = pa.array([secs], pa.timestamp("s")).cast(pa.string()).to_pylist()[0]
        if tp.startswith("<"):
            continue
        date, time = tp.split(" ")
        sep = rng.choice([" ", "T"])
        form = rng.randrange(5)
        if form == 0:
            s = date
        elif form == 1:
            s = date + sep + time[:2]
        elif form == 2:
            s = date + sep + time[:5]
        elif form == 3:
            s = date + sep + time
        else:
            s = date + sep + time + "." + "".join(rng.choice("0123456789") for _ in range(rng.randint(0, 10)))
        if rng.random() < 0.5:
            s += rng.choice(["Z", "+01", "-11", "+0130", "-0945", "+05:30", "-12:00", "+23:59", "+24", "-99:00"])
        if rng.random() < 0.3:
            s = mutate(rng, s)
        texts.append(s)
    texts += ["1970-01-01", "2021-02-29", "2020-02-29T23:59:59.999999999Z", "0000-01-01", "9999-12-31T23:59:59",
              "2262-04-11T23:47:16.854775807", "2262-04-11T23:47:16.854775808", "1677-09-21T00:12:43.145224192",
              "1677-09-21T00:12:43.145224192Z", "1970-01-01T01Z", "1970-01-01T01:02Z", "1970-01-01T00:00:00+00:00",
              "1970-01-01Z", " 1970-01-01", "1970-1-1", "19700101", "1970-01-01T24:00:00", "1970-01-01T23:59:60",
              "1970-01-01T01:02:03.", "1970-01-01T01:02:03,5", "1970-01-01t01:02:03", "", "x", "1970-01-01T01:02:03.000Z"]
    return texts


def time_texts(rng: random.Random) -> list[str]:
    texts = []
    for _ in range(80):
        s = f"{rng.randrange(25):02d}:{rng.randrange(61):02d}"
        if rng.random() < 0.7:
            s += f":{rng.randrange(61):02d}"
            if rng.random() < 0.5:
                s += "." + "".join(rng.choice("0123456789") for _ in range(rng.randint(0, 10)))
        if rng.random() < 0.2:
            s = mutate(rng, s)
        texts.append(s)
    return texts


def cast_from_string_cases(rng: random.Random, out: list[str]) -> None:
    texts = iso_texts(rng)
    codes = [f"ts:{u}:" for u in UNITS] + [f"ts:{u}:UTC" for u in UNITS] + ["ts:ms:+05:30", "d32", "d64"]
    for code in codes:
        t = arrow_type(code)
        for s in texts:
            try:
                r = pa.array([s], pa.string()).cast(t)
                v = r.cast(pa.int64() if t.bit_width == 64 else pa.int32()).to_pylist()[0]
                out.append(f"P\t{code}\t{hx(s)}\t{v}")
            except Exception:  # noqa: BLE001
                out.append(f"P\t{code}\t{hx(s)}\tE")
    # pyarrow 26 has no string to time or duration cast: expected values from Arrow's ParseTime
    # and integer parsing are in the Zig test.


STRP_FORMATS = ["%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%d/%m/%Y", "%m/%d/%y %I:%M", "%D %T", "%Y%m%d",
                "%Y %j", "%C%y-%m-%d", "%Y-%m-%d%n%R", "%e.%m.%Y %H", "%Y %U %w", "%y", "%Y %W", "%+4Y-%2m-%2d",
                "%Y%%%m", "%H:%M:%S", "%Y-%m-%d %H:%M:%S%z", "%Y %b", "%a %Y", "%Y %p", "%Y %Z", "%s", "%F",
                "%Y %Od", "%Y%", "x %Y", "%Y\t%m"]


def strptime_cases(rng: random.Random, out: list[str]) -> None:
    for fmt in STRP_FORMATS:
        texts = []
        for _ in range(40):
            secs = random_seconds(rng) if rng.random() < 0.3 else rng.randrange(-2208988800, 4102444800)
            try:
                s = pc.strftime(pa.array([secs], pa.timestamp("s", "UTC")), format=fmt.replace("%+4Y", "%Y").replace("%2", "%")).to_pylist()[0]
            except Exception:  # noqa: BLE001
                s = "2009-02-13 23:31:31"
            if rng.random() < 0.4:
                s = mutate(rng, s)
            texts.append(s)
        texts += ["2009-02-30 00:00:00", "2009-00-10 00:00:00", "2009 0", "2009 00", "-2009", "+2009", "99999",
                  "2009-02-13 23:59:60", "2009-02-13 24:00:00", "1 2 3", "", "2009"]
        for unit in UNITS if fmt in ("%Y-%m-%d %H:%M:%S", "%y") else ["s", "ns"]:
            for s in texts:
                try:
                    r = pc.strptime(pa.array([s]), format=fmt, unit=unit)
                    v = r.cast(pa.int64()).to_pylist()[0]
                    out.append(f"S\t{unit}\t{hx(fmt)}\t{hx(s)}\t{v}")
                except Exception:  # noqa: BLE001
                    out.append(f"S\t{unit}\t{hx(fmt)}\t{hx(s)}\tE")


def main() -> None:
    if not pa.__version__.startswith(EXPECTED_PYARROW):
        sys.exit(f"pyarrow {pa.__version__}: expected {EXPECTED_PYARROW}x")
    rng = random.Random(20)
    out = [f"# Generated by tools/temporal_text_corpus.py with pyarrow {pa.__version__}; do not edit."]
    strftime_cases(rng, out)
    cast_to_string_cases(rng, out)
    cast_from_string_cases(rng, out)
    strptime_cases(rng, out)
    with open(sys.argv[1], "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
