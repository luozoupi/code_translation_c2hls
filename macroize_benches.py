"""Macroize HLSFactory polybench dimensions so cosim can run on smaller arrays.

Each polybench bench has a testbench.cpp main() that declares scalar bounds like:
    int ni = 60; int nj = 70; int nk = 80;
followed by array decls like `double C[60 + 0][70 + 0];` and kernel calls.

We:
  1. Parse the testbench's `int <var> = <int>;` lines (inside main only) to learn
     the (lower_var, value) pairs. Convention: macro name = upper(var).
  2. Substitute literal occurrences across header, gold, plain, hls_baseline,
     testbench:
       - `[<v> + 0]` -> `[<MACRO> + 0]`  (array dims, polybench style)
       - `[<v>]`     -> `[<MACRO>]`      (catch-all dim)
       - `= <v>;`    -> `= <MACRO>;`     (const int / int initializer)
  3. Prepend `#ifndef <MACRO> / #define <MACRO> <v> / #endif` guards to the
     header so the original size is the default, but cosim can override
     via `-D<MACRO>=8` cflags.

Idempotent: skips files that already define the macro guards.

NEVER runs while a matrix_sweep is in-flight — call only when sweeps are quiesced.

Usage:
    python3 macroize_benches.py                 # all hlsfactory_* benches
    python3 macroize_benches.py --bench gemm    # one bench
    python3 macroize_benches.py --dry-run       # print plan, write nothing
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent
BENCH_ROOT = REPO / "benchmarks"

# Allowed scalar names — restrict to polybench convention to avoid accidental
# capture of unrelated `int foo = 7;` in testbenches.
ALLOWED_VARS = {
    "m", "n",
    "ni", "nj", "nk", "nl", "nm",
    "nr", "nq", "np",
    "nx", "ny",
    "tmax", "tsteps",
}


def _parse_testbench_dims(tb_text: str) -> dict[str, int]:
    """Extract `int <var> = <value>;` pairs from the testbench main()."""
    # Find the int declarations between `int main(` and the first array decl.
    # Polybench testbenches always declare scalar dims at the very top of main().
    m = re.search(r"int\s+main\s*\([^)]*\)\s*\{", tb_text)
    if not m:
        return {}
    body = tb_text[m.end():]
    dims: dict[str, int] = {}
    for var, val in re.findall(r"\bint\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(\d+)\s*;", body):
        if var.lower() in ALLOWED_VARS and var.lower() not in dims:
            dims[var.lower()] = int(val)
    return dims


def _substitute(text: str, var: str, value: int, macro: str) -> tuple[str, int]:
    """Replace `value` with `macro` in dim brackets and `= value;` contexts."""
    n_total = 0
    # 1) `[ <value> + 0]` -> `[ <macro> + 0]`   (polybench dim form)
    pat1 = re.compile(rf"(\[\s*){value}(\s*\+\s*0\s*\])")
    text, n = pat1.subn(rf"\g<1>{macro}\g<2>", text)
    n_total += n
    # 2) `[<value>]` -> `[<macro>]`             (plain dim form)
    pat2 = re.compile(rf"(\[\s*){value}(\s*\])")
    text, n = pat2.subn(rf"\g<1>{macro}\g<2>", text)
    n_total += n
    # 3) `<var> = <value>;` -> `<var> = <macro>;` for const int / int decls
    #    Only when the variable name matches (so we don't catch unrelated literals).
    pat3 = re.compile(rf"(\b{re.escape(var)}\s*=\s*){value}(\s*[;,)])")
    text, n = pat3.subn(rf"\g<1>{macro}\g<2>", text)
    n_total += n
    return text, n_total


def _ensure_header_guards(header_text: str, dims_macro: list[tuple[str, int]]) -> tuple[str, bool]:
    """Prepend `#ifndef MACRO / #define MACRO val / #endif` for each dim, if not present."""
    block_lines = ["// >>> c2hls auto-macro guards (do not edit between markers)"]
    added = False
    for macro, val in dims_macro:
        if re.search(rf"^\s*#\s*define\s+{macro}\b", header_text, flags=re.MULTILINE):
            continue
        block_lines.append(f"#ifndef {macro}")
        block_lines.append(f"#define {macro} {val}")
        block_lines.append(f"#endif")
        added = True
    block_lines.append("// <<< c2hls auto-macro guards")
    if not added:
        return header_text, False
    # Insert after `#pragma once` if present, else at top
    block = "\n".join(block_lines) + "\n"
    m = re.search(r"^#pragma\s+once\s*\n", header_text, flags=re.MULTILINE)
    if m:
        idx = m.end()
        return header_text[:idx] + block + header_text[idx:], True
    return block + header_text, True


def _process_bench(bench_dir: Path, dry_run: bool) -> dict:
    meta_path = bench_dir / "metadata.json"
    if not meta_path.exists():
        return {"bench": bench_dir.name, "skipped": "no-metadata"}
    meta = json.loads(meta_path.read_text())

    tb_file = bench_dir / (meta.get("testbench_file") or "testbench.cpp")
    if not tb_file.exists():
        return {"bench": bench_dir.name, "skipped": "no-testbench"}

    dims_lower = _parse_testbench_dims(tb_file.read_text())
    if not dims_lower:
        return {"bench": bench_dir.name, "skipped": "no-dims-parsed"}

    # (var, macro, value) triples — use upper case macros
    triples = [(v, v.upper(), val) for v, val in dims_lower.items()]
    dims_macro = [(macro, val) for _, macro, val in triples]

    # Files to substitute in: header + every .cpp in the bench dir
    header_file = bench_dir / (meta.get("header_file") or f"{bench_dir.name.replace('hlsfactory_','')}.h")
    targets = [header_file]
    for p in sorted(bench_dir.glob("*.cpp")):
        targets.append(p)

    report = {"bench": bench_dir.name, "dims": dims_lower, "files": {}}
    for f in targets:
        if not f.exists():
            continue
        original = f.read_text()
        text = original
        per_file = {}
        for var, macro, val in triples:
            text, n = _substitute(text, var, val, macro)
            if n:
                per_file[macro] = n
        if f == header_file:
            text, added = _ensure_header_guards(text, dims_macro)
            if added:
                per_file["_guards_added"] = True
        if text != original:
            report["files"][f.name] = per_file
            if not dry_run:
                f.write_text(text)

    # Update metadata with cosim_size_overrides defaults (cap each dim at 8)
    overrides = {macro: min(val, 8) for _, macro, val in triples}
    if meta.get("cosim_size_overrides") != overrides:
        report["meta_updated"] = True
        if not dry_run:
            meta["cosim_size_overrides"] = overrides
            meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bench", help="single bench name (without hlsfactory_ prefix), or full dirname")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if args.bench:
        name = args.bench if args.bench.startswith("hlsfactory_") else f"hlsfactory_{args.bench}"
        bench_dirs = [BENCH_ROOT / name]
    else:
        bench_dirs = sorted(BENCH_ROOT.glob("hlsfactory_*"))

    reports = []
    for bd in bench_dirs:
        if not bd.is_dir():
            continue
        rep = _process_bench(bd, args.dry_run)
        reports.append(rep)
        if "skipped" in rep:
            print(f"SKIP {bd.name}: {rep['skipped']}")
        else:
            n_files = len(rep["files"])
            dims = ", ".join(f"{k.upper()}={v}" for k, v in rep["dims"].items())
            tag = " [dry-run]" if args.dry_run else ""
            print(f"OK   {bd.name}: dims={{{dims}}} files_touched={n_files}{tag}")
    print()
    print(f"Total benches processed: {sum(1 for r in reports if 'skipped' not in r)}")


if __name__ == "__main__":
    main()
