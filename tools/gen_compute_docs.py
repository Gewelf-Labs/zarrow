#!/usr/bin/env python3
"""Generate the per-function matrix page of every compute group: docs/compute/<group>/matrix.md.

The overview is docs/compute-support-matrix.md (one row per group). Each group folder holds
matrix.md (generated here: one row per registered function with its evidence) and notes.md
(hand-written, Arrow-style function notes). Evidence per function:

- source: the first line of the family file naming the function, and the family's `register`;
- tests:  every `test "..."` block in src/compute/*.zig whose title or body names the function
  (a quoted string, or a word in the title for a name with an underscore).

A function with no such test is `partial`, as CLAUDE.md requires.
Usage: python -I tools/gen_compute_docs.py [--check]   (--check: exit 1 if a page is stale)
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COMPUTE = ROOT / "src/compute"
MAX_TESTS = 12

S, V, A, H = "scalar", "vector", "aggregate", "hash_aggregate"


def fns(kind, names):
    return [(n, kind) for n in names.split()]


def checked(names):
    return " ".join(f"{n} {n}_checked" for n in names.split())


# (slug, title, family files searched in order, functions). Every function compute.registerAll registers.
GROUPS = [
    ("aggregations", "Aggregations", "aggregate_basic aggregate_stats tdigest pivot",
     fns(A, "all any approximate_median count count_all count_distinct first first_last index kurtosis "
            "last max mean min min_max pivot_wider product skew stddev sum tdigest variance")
     + fns(V, "mode quantile")),
    ("grouped-aggregations", "Grouped aggregations", "hash_aggregate pivot tdigest",
     fns(H, "hash_all hash_any hash_approximate_median hash_count hash_count_all hash_count_distinct "
            "hash_distinct hash_first hash_first_last hash_kurtosis hash_last hash_list hash_max hash_mean "
            "hash_min hash_min_max hash_one hash_pivot_wider hash_product hash_skew hash_stddev hash_sum "
            "hash_tdigest hash_variance")),
    ("arithmetic", "Arithmetic", "arithmetic decimal math",
     fns(S, checked("add subtract multiply divide modulo remainder negate abs power") + " sign")),
    ("bit-wise", "Bit-wise", "math arithmetic",
     fns(S, "bit_wise_and bit_wise_not bit_wise_or bit_wise_xor " + checked("shift_left shift_right"))),
    ("rounding-math", "Rounding, logarithmic, trigonometric, hyperbolic", "round math",
     fns(S, "ceil floor trunc round round_binary round_to_multiple exp expm1 "
            + checked("sqrt ln log2 log10 log1p logb sin cos tan asin acos atanh acosh")
            + " atan atan2 sinh cosh tanh asinh hypot")),
    ("comparisons", "Comparisons", "comparison",
     fns(S, "equal not_equal less less_equal greater greater_equal max_element_wise min_element_wise")),
    ("logical", "Logical", "logical",
     fns(S, "and and_kleene and_not and_not_kleene invert or or_kleene xor")),
    ("string-predicates", "String predicates", "string",
     fns(S, "string_is_ascii ascii_is_alnum ascii_is_alpha ascii_is_decimal ascii_is_lower ascii_is_printable "
            "ascii_is_space ascii_is_title ascii_is_upper utf8_is_alnum utf8_is_alpha utf8_is_decimal "
            "utf8_is_digit utf8_is_lower utf8_is_numeric utf8_is_printable utf8_is_space utf8_is_title "
            "utf8_is_upper")),
    ("string-transforms", "String transforms, padding, trimming, splitting, slicing, joining",
     "string string_transform string_normalize",
     fns(S, "ascii_capitalize ascii_lower ascii_reverse ascii_swapcase ascii_title ascii_upper "
            "utf8_capitalize utf8_lower utf8_reverse utf8_swapcase utf8_title utf8_upper utf8_normalize "
            "binary_length utf8_length binary_repeat binary_reverse binary_replace_slice utf8_replace_slice "
            "replace_substring binary_slice utf8_slice_codeunits utf8_zero_fill "
            "ascii_center ascii_lpad ascii_rpad utf8_center utf8_lpad utf8_rpad "
            "ascii_ltrim ascii_rtrim ascii_trim ascii_ltrim_whitespace ascii_rtrim_whitespace "
            "ascii_trim_whitespace utf8_ltrim utf8_rtrim utf8_trim utf8_ltrim_whitespace "
            "utf8_rtrim_whitespace utf8_trim_whitespace ascii_split_whitespace utf8_split_whitespace "
            "split_pattern binary_join binary_join_element_wise")),
    ("regular-expressions", "Regular expressions", "string_regex",
     fns(S, "match_substring_regex match_like count_substring_regex find_substring_regex "
            "replace_substring_regex split_pattern_regex extract_regex extract_regex_span")),
    ("containment", "Containment", "set_lookup string",
     fns(S, "is_in index_in match_substring starts_with ends_with count_substring find_substring")),
    ("categorizations", "Categorizations", "is_null math",
     fns(S, "is_null is_valid is_nan is_finite is_inf true_unless_null")),
    ("selecting-multiplexing", "Selecting / multiplexing", "selection",
     fns(S, "if_else case_when coalesce choose")),
    ("structural-transforms", "Structural transforms", "structural",
     fns(S, "list_element list_slice list_value_length make_struct struct_field map_lookup")
     + fns(V, "list_flatten list_parent_indices")),
    ("conversions", "Conversions", "cast temporal_text",
     fns(S, "cast strptime strftime")),
    ("temporal", "Temporal", "temporal temporal_arith temporal_round temporal_zone",
     fns(S, "year month day day_of_week day_of_year hour minute second millisecond microsecond nanosecond "
            "subsecond quarter week us_week us_year iso_week iso_year iso_calendar year_month_day "
            "is_leap_year is_dst years_between quarters_between month_interval_between weeks_between "
            "days_between hours_between minutes_between seconds_between milliseconds_between "
            "microseconds_between nanoseconds_between day_time_interval_between "
            "month_day_nano_interval_between ceil_temporal floor_temporal round_temporal "
            "assume_timezone local_timestamp")),
    ("random", "Random", "math", fns(S, "random")),
    ("cumulative-pairwise", "Cumulative, statistical, pairwise", "cumulative",
     fns(V, "cumulative_max cumulative_mean cumulative_min " + checked("cumulative_prod cumulative_sum")
            + " winsorize " + checked("pairwise_diff"))),
    ("associative", "Associative transforms", "associative",
     fns(V, "dictionary_encode dictionary_decode unique value_counts")),
    ("run-end-encoding", "Run-end encoding", "run_end", fns(V, "run_end_encode run_end_decode")),
    ("selections", "Selections", "selection",
     fns(V, "take array_take filter array_filter drop_null inverse_permutation scatter")),
    ("sorts-partitions", "Sorts and partitions", "sort",
     fns(V, "sort_indices array_sort_indices rank rank_normal rank_quantile partition_nth_indices "
            "select_k_unstable search_sorted")),
    ("replace", "Replace functions", "selection",
     fns(V, "fill_null_forward fill_null_backward replace_with_mask")),
    ("indices", "Indices", "selection", fns(V, "indices_nonzero")),
]

TEST_BLOCK = re.compile(r'^test "([^"]+)" \{\n(.*?)^\}', re.M | re.S)


def load_tests():
    out = []
    for p in sorted(COMPUTE.glob("*.zig")):
        text = p.read_text(encoding="utf-8").replace("\r\n", "\n")
        for m in TEST_BLOCK.finditer(text):
            out.append((f"src/compute/{p.name}", m.group(1), m.group(2)))
    return out


def find_line(path, pattern):
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if re.search(pattern, line):
            return i
    return None


def evidence(name, families, tests):
    word = re.compile(rf"(?<![\w.]){re.escape(name)}(?!\w)")
    lit = re.compile(rf'"{re.escape(name)}"')
    srcs = []
    for fam in families:
        p = COMPUTE / f"{fam}.zig"
        hit = (find_line(p, lit.pattern) or find_line(p, rf"\.{re.escape(name)}\b")
               or find_line(p, rf"\b{re.escape(name)}\b"))
        if hit:
            srcs.append(f"`src/compute/{fam}.zig:{hit}`")
            break
    reg = find_line(COMPUTE / f"{families[0]}.zig", r"^pub fn register\(")
    if reg:
        srcs.append(f"`src/compute/{families[0]}.zig:{reg}`")
    # A title word counts only for names no English sentence uses (`sort_indices`, not `and`).
    hits = [(f, t) for f, t, body in tests if ("_" in name and word.search(t)) or lit.search(body)]
    return srcs, hits


def page(title, families, functions, tests):
    fam = families.split()
    rows, counts = [], {"yes": 0, "partial": 0}
    for name, kind in functions:
        srcs, hits = evidence(name, fam, tests)
        status = "yes" if hits else "partial"
        counts[status] += 1
        shown = [f'`{f}:"{t}"`' for f, t in hits[:MAX_TESTS]]
        more = f" (+{len(hits) - MAX_TESTS} more)" if len(hits) > MAX_TESTS else ""
        tcell = ", ".join(shown) + more if shown else "missing: no test names this function"
        rows.append(f"| [`{name}`](notes.md#{name}) | {kind} | {status} | {', '.join(srcs)} | {tcell} |")
    head = [
        f"# {title}: function matrix",
        "",
        "Generated by `tools/gen_compute_docs.py`; do not edit by hand. Up: the",
        "[compute support matrix](../../compute-support-matrix.md). Function details: [notes.md](notes.md).",
        "",
        "Status per function: **yes** when a test in `src/compute/` names it (as a quoted name, or in its",
        "title for a name with an underscore), **partial** when it is registered but no test names it. Source cites the",
        "first line of the family file that names the function and the family's `register` function.",
        "",
        f"{counts['yes']} yes, {counts['partial']} partial, {len(functions)} functions.",
        "",
        "| Function | Kind | Status | Source | Tests |",
        "|---|---|---|---|---|",
    ]
    return "\n".join(head + rows) + "\n", counts


def main(argv):
    check = "--check" in argv
    tests = load_tests()
    stale, seen, total = [], set(), {"yes": 0, "partial": 0}
    for slug, title, families, functions in GROUPS:
        for n, _ in functions:
            assert n not in seen, f"{n} listed twice"
            seen.add(n)
        text, counts = page(title, families, functions, tests)
        for k in total:
            total[k] += counts[k]
        out = ROOT / "docs/compute" / slug / "matrix.md"
        old = out.read_text(encoding="utf-8").replace("\r\n", "\n") if out.exists() else None
        if old != text:
            if check:
                stale.append(str(out.relative_to(ROOT)))
            else:
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(text, encoding="utf-8", newline="\n")
    print(f"{len(seen)} functions in {len(GROUPS)} groups: {total['yes']} yes, {total['partial']} partial")
    for s in stale:
        print(f"stale: {s} (run python -I tools/gen_compute_docs.py)")
    return 1 if stale else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
