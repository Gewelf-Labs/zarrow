#!/usr/bin/env python3
"""Generate the regex compute corpus checked by src/compute/test_string_regex.zig.

Runs every pattern of a hand-written list plus seeded random patterns over a set of texts
through pyarrow's RE2-backed kernels (match_substring_regex, find_substring_regex,
count_substring_regex, replace_substring_regex with and without max_replacements,
split_pattern_regex, extract_regex_span, match_like) and writes one line per case:

    P <TAB> type <TAB> pattern-hex        (a pattern; `type` is `s` string or `b` binary)
    T <TAB> text-hex                      (a text of the current pattern)
    fn <TAB> option <TAB> result          (a case on the current text)
 Results: `E` (the call fails), `H` (Arrow loops
forever: split with an empty separator), `1`/`0`, an integer, hex bytes, comma-separated hex
parts or `start:length` spans (`-` for a group that did not take part, `N` for a null row).

pyarrow lives in the user site, so run it with `python -P` (isolated mode hides it):
    python -P tools/regex_corpus.py src/compute/testdata/regex_corpus.tsv
"""
from __future__ import annotations

import random
import sys

import pyarrow as pa
import pyarrow.compute as pc

EXPECTED_PYARROW = "26."
BS = chr(92)  # backslash

ALPHABET = ["a", "b", "c", "x", "_", "0", "9", " ", "\n", ".", "-", "é", "É", "ß", "ẞ", "σ",
            "Σ", "ς", "ſ", "s", "S", "k", "K", "\u212a", "İ", "ı", "i", "I", "日", "😀", "ǅ", "ǆ"]

TEXTS = ["", "a", "abc", "aaa", "abcabc", "xaxbx", "a b  c", "foo bar_baz 42", "line1\nline2\n",
         "héllo wörld", "ÉCOLE école", "Straße STRASSE ẞ", "ΣΊΣΥΦΟΣ σίσυφος ς", "ſ s S",
         "K k \u212a", "İ ı i I", "日本語テキスト", "emoji 😀 x", "ǅungla", "a1b22c333",
         "--x--", "x.y.z", "tab\tsep", "AbCaBc", "aXbXc"]

HAND = [
    "a", "abc", "", "a*", "a+", "a?", "a*?", "a+?", "a??", "a{2}", "a{2,}", "a{1,2}", "a{1,2}?",
    "a{,2}", "a{", "x{2}y", ".", ".*", ".+", "(?s).", "(?s).*", "^", "$", "^a", "a$", "^$",
    "(?m)^", "(?m)$", "(?m)^l", "(?m)\\d$", "\\A.", ".\\z", "\\b", "\\B", "\\bb", "\\Bb", "\\w+",
    "\\W+", "\\d+", "\\D+", "\\s+", "\\S+", "[abc]", "[^abc]", "[a-c]+", "[^a-c]+", "[[:alpha:]]+",
    "[[:^alpha:]]+", "[[:digit:][:space:]]+", "[[:punct:]]", "[[:upper:]]", "[[:word:]]+",
    "[\\d.]+", "[\\w\\s]+", "[^\\w]", "[]a]", "[a-]", "[-a]", "\\pL+", "\\p{L}", "\\PL+",
    "\\p{Lu}", "\\p{Ll}+", "\\p{^Ll}+", "\\pN+", "\\p{Greek}", "\\pZ", "\\pP", "\\pS", "\\pC",
    "\\pM", "[\\pL\\d]+", "[^\\pL]", "a|b", "ab|a", "a|ab", "(?:ab)+", "(?:a|b)*c", "(?P<x>a)",
    "(?P<x>a)(?P<y>b)?", "(?P<x>a|b)+", "(?P<x>a*)(?P<y>a*)", "(?P<x>a*?)(?P<y>a*)",
    "(?P<w>\\w+) (?P<n>\\d+)", "(?P<x>(?P<y>a)b)", "(?<x>a)", "(?i)a", "(?i)é", "(?i)ß",
    "(?i)ẞ", "(?i)σ", "(?i)ς", "(?i)s", "(?i)k", "(?i)i", "(?i)İ", "(?i)ı", "(?i)ǅ", "(?i)école",
    "(?i)[a-z]+", "(?i)[^a-z]+", "(?i)\\w+", "(?i)\\W", "(?i)\\p{Lu}+", "(?i)straße",
    "(?i:a)b", "a(?i)b", "(?i)a(?-i)b", "(?U)a+", "(?U)a+?", "(?m:^a)", "(?s:.)", "é", "[é]",
    "[^é]", "[α-ω]+", "日+", "😀", "[😀-😁]", ".😀", "\\x61", "\\x{e9}", "\\x{1F600}", "\\141",
    "\\0", "\\Q.*\\E", "\\Q.*\\E+", "\\.", "\\-", "\\/", "\\n", "\\t", "\\C", "\\C\\C",
    "x*", "(a*)*", "(a|b)*", "(?:a*)+b", "(a+|b+)*c", "((a)|b)+", "a{0}", "(?:){3}", "(?:a?){2}",
    "[a-c]{2,3}?", "\\d{2,}", "\\b\\w", "\\w\\b", "^\\w+$", "^.*$", "(?m)^.*$", "\\s*", "\\s*\\S",
    "_", "\\.", "x\\.y", "-+", "\\p{Han}", "\\p{Any}", "[[:ascii:]]+", "[[:^ascii:]]+",
    # Invalid or unsupported.
    "(a", "a)", "[a", "[z-a]", "*a", "a**", "a++", "a{1001}", "a{2,1}", "(a{100}){100}", "\\1",
    "(?=a)", "(?!a)", "(?<=a)", "(?<!a)", "(?P=x)", "\\8", "\\q", "\\", "(?P<x>a)(?P<x>b)",
    "(?P<>a)", "(?x)a", "(?i-)a", "[[:foo:]]", "\\x{110000}", "\\pX", "\\p{Cn}", "[\\b]", "\\Z",
    "x{2}{3}", "(?#c)", "\\z+", "(?P<1a>x)", "(?P<a-b>x)", "(?i)(?-i)a", "a{1000}", "(?:a{20}){50}",
]

LIKE = ["", "%", "%%", "_", "a", "a%", "%a", "%a%", "a_c", "%b%", "_b%", "%\\%%", "a\\_c", "a\\",
        "%a\\", "%a\\b%", "\\%", "abc", "ABC", "%é%", "_é%", "%.%", "%[%", "%(%", "a%c", "%x%y%",
        "%\n%", "%ß", "Σ%", "%ς", "line_\n%", "%s%"]



def hexs(b: bytes) -> str:
    return b.hex()


def rand_atom(rng: random.Random, depth: int) -> str:
    r = rng.random()
    if r < 0.30:
        c = rng.choice(ALPHABET)
        return "\\" + c if c in ".^$*+?()[]{}|\\" else ("\\n" if c == "\n" else c)
    if r < 0.45:
        return rng.choice([".", "\\d", "\\w", "\\s", "\\W", "\\D", "\\pL", "\\p{Lu}", "\\b", "\\B",
                           "^", "$", "[a-c]", "[^a ]", "[[:alpha:]]", "[é-ſ]", "[\\pL\\d]"])
    if r < 0.6 and depth < 3:
        inner = rand_pattern(rng, depth + 1)
        kind = rng.choice(["(?:%s)", "(?P<g%d>%s)", "(?i:%s)", "(?s:%s)", "(?m:%s)"])
        if "%d" in kind:
            rand_atom.group += 1
            return kind % (rand_atom.group, inner)
        return kind % inner
    return rng.choice(ALPHABET[:12]).replace(".", "\\.").replace("\n", "\\n")


rand_atom.group = 0


def rand_pattern(rng: random.Random, depth: int = 0) -> str:
    parts = []
    for _ in range(rng.randint(1, 4)):
        a = rand_atom(rng, depth)
        q = rng.random()
        if q < 0.35 and a not in ("^", "$", "\\b", "\\B"):
            a += rng.choice(["*", "+", "?", "{2}", "{1,3}", "{0,}", "{2,}"])
            if rng.random() < 0.3:
                a += "?"
        parts.append(a)
    p = "".join(parts)
    if depth == 0 and rng.random() < 0.15:
        p = p + "|" + rand_pattern(rng, 1)
    if depth == 0 and rng.random() < 0.15:
        p = rng.choice(["(?i)", "(?m)", "(?s)", "(?U)"]) + p
    return p


def call(f, *args, **kw):
    try:
        return f(*args, **kw)
    except Exception:  # noqa: BLE001 - any Arrow error is an "E" case
        return None


def one(f, arr_type, text, *args, **kw):
    arr = pa.array([text], arr_type)
    r = call(f, arr, *args, **kw)
    if r is None:
        return None
    if pa.types.is_string(r.type) or pa.types.is_binary(r.type):
        return r.cast(pa.binary())[0].as_py()  # raw bytes: a string result may be invalid UTF-8
    return r[0].as_py()


def enc(text, binary: bool) -> bytes:
    return text if isinstance(text, bytes) else text.encode("utf-8")


def spans_of(v) -> str:
    if v is None:
        return "N"
    return ",".join("-" if s is None else "%d:%d" % (s[0], s[1]) for s in v.values()) or "{}"


def split_sim(pattern: str, data: bytes, max_splits: int | None, arr_type) -> str:
    """Arrow's split loop (FindAndConsume over the rest of the row), with RE2 matching."""
    sep = "(?P<zzsep>" + pattern + ")"
    parts = []
    i = 0
    splits = 0
    while max_splits is None or splits < max_splits:
        piece = data[i:]
        if arr_type == pa.string():
            piece = piece.decode("utf-8")
        v = one(pc.extract_regex_span, arr_type, piece, sep)
        if v is None:
            break
        s, n = v["zzsep"]
        if n == 0 and max_splits is None:
            return "H"
        parts.append(data[i:i + s])
        i += s + n
        splits += 1
    parts.append(data[i:])
    return ",".join(hexs(p) for p in parts)


def main() -> None:
    assert pa.__version__.startswith(EXPECTED_PYARROW), pa.__version__
    out_path = sys.argv[1]
    rng = random.Random(19)
    patterns = list(HAND)
    seen = set(patterns)
    while len(patterns) < len(HAND) + 150:
        p = rand_pattern(rng)
        if p not in seen:
            seen.add(p)
            patterns.append(p)
    lines = []
    for pi, pat in enumerate(patterns):
        texts = rng.sample(TEXTS, 8) if pi < len(HAND) else rng.sample(TEXTS, 3) + ["".join(rng.choice(ALPHABET) for _ in range(rng.randint(0, 10))) for _ in range(3)]
        valid = call(pc.match_substring_regex, pa.array(["x"]), pat) is not None
        can_split = call(pc.extract_regex_span, pa.array(["x"]), "(?P<zzsep>" + pat + ")") is not None and "\\C" not in pat
        named_ok = call(pc.extract_regex_span, pa.array(["x"]), pat) is not None
        groups = "(" in pat
        for ty, arr_type in (("s", pa.string()), ("b", pa.binary())):
            if ty == "b" and pi % 3 != 0:
                continue
            lines.append("P\t%s\t%s" % (ty, hexs(pat.encode("utf-8"))))
            if not valid:
                lines.append("T\t")
                lines.append("match\t0\tE")
                continue
            for text in texts:
                data = text.encode("utf-8")
                t = data if ty == "b" else text
                lines.append("T\t" + hexs(data))

                def put(fn, opt, res):
                    lines.append("\t".join([fn, opt, res]))

                for ic in ((False, True) if pi % 2 == 0 else (False,)):
                    m = one(pc.match_substring_regex, arr_type, t, pat, ignore_case=ic)
                    put("match", "1" if ic else "0", "E" if m is None else str(int(m)))
                f = one(pc.find_substring_regex, arr_type, t, pat)
                put("find", "0", "E" if f is None else str(f))
                c = one(pc.count_substring_regex, arr_type, t, pat)
                put("count", "0", "E" if c is None else str(c))
                reps = [("<" + BS + "0>", None), ("<" + BS + "0>", 2), ("[" + BS + "2" + BS + "1]" if groups else "-", None)]
                if pi % 4 == 0:
                    reps += [(BS + BS, 1), ("", None)]
                for rep, mx in reps:
                    r = one(pc.replace_substring_regex, arr_type, t, pat, rep, max_replacements=mx)
                    opt = "%d:%s" % (-1 if mx is None else mx, hexs(rep.encode()))
                    put("replace", opt, "E" if r is None else hexs(enc(r, ty == "b")))
                if can_split:
                    for mx in (None, 1):
                        sim = split_sim(pat, data, mx, arr_type)
                        if sim != "H":
                            r = call(pc.split_pattern_regex, pa.array([t], arr_type), pat, max_splits=mx)
                            got = "E" if r is None else ",".join(hexs(x) for x in r.cast(pa.list_(pa.binary()))[0].as_py())
                            if got != "E" and got != sim:
                                raise SystemExit("split simulation disagrees: %r %r %r %r" % (pat, text, got, sim))
                        put("split", str(-1 if mx is None else mx), sim)
                if named_ok or "(" not in pat:
                    v = one(pc.extract_regex_span, arr_type, t, pat)
                    put("span", "0", spans_of(v))
                elif pi % 5 == 0:
                    put("span", "0", "E")
    for pat in LIKE:
        for ty, arr_type in (("s", pa.string()), ("b", pa.binary())):
            lines.append("P\t%s\t%s" % (ty, hexs(pat.encode())))
            for text in TEXTS + ["a%c", "a_c", "a" + BS + "b", "a" + BS, "%", "_"]:
                data = text.encode("utf-8")
                t = data if ty == "b" else text
                lines.append("T\t" + hexs(data))
                for ic in (False, True):
                    m = one(pc.match_like, arr_type, t, pat, ignore_case=ic)
                    lines.append("like\t%d\t%s" % (ic, "E" if m is None else str(int(m))))
    with open(out_path, "w", encoding="ascii", newline="\n") as f:
        f.write("# Generated by tools/regex_corpus.py with pyarrow %s; do not edit.\n" % pa.__version__)
        f.write("\n".join(lines) + "\n")
    print(len(patterns), "patterns,", len(lines), "cases")


if __name__ == "__main__":
    main()
