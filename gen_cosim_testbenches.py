"""Generate authentic cosim testbenches that diff candidate-vs-gold outputs.

The csim testbench in each polybench bench dumps arrays to stderr; csim only
checks that the dump matches the gold trace produced separately. For cosim
we want a SELF-CONTAINED PASS/FAIL: rerun the gold kernel inside the testbench
on identical inputs and diff element-wise against the candidate's output.

This script emits, per bench:
  - testbench_cosim.cpp : the new cosim driver
  - gold_kernel_for_cosim.cpp : a thin shim that defines kernel_<bench>_gold
    by #include'ing the original gold kernel with macro renaming.
  - metadata.json fields:
      cosim_testbench_file : "testbench_cosim.cpp"
      cosim_support_files  : ["gold_kernel_for_cosim.cpp"]

The cosim testbench:
  - duplicates EVERY array decl in main() with a `_gold` variant
  - calls init_array twice (once on primary, once on _gold) for identical state
  - calls kernel_<bench>(primary) then kernel_<bench>_gold(gold)
  - diffs every array passed to print_array within REL_TOL=1e-4, ABS_TOL=1e-6
  - prints `PASS` (exit 0) or `FAIL: ...` (exit 1)

NEVER runs while a matrix_sweep is in-flight.

Usage:
    python3 gen_cosim_testbenches.py [--bench gemm] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent
BENCH_ROOT = REPO / "benchmarks"

# Per-tolerance: polybench small kernels are FP-double; 1e-4 relative is
# plenty for FP reduction-order divergence yet still flags real bugs.
REL_TOL = 1e-4
ABS_TOL = 1e-6


def _parse_main_body(tb_text: str) -> tuple[str, str]:
    """Return (preamble, main_body) where preamble is everything before main()
    and main_body is the body of main() with braces stripped."""
    m = re.search(r"int\s+main\s*\([^)]*\)\s*\{", tb_text)
    if not m:
        raise ValueError("no main() found")
    body_start = m.end()
    # Walk to matching close brace
    depth = 1
    i = body_start
    while i < len(tb_text) and depth > 0:
        c = tb_text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        i += 1
    if depth != 0:
        raise ValueError("unbalanced main() braces")
    body = tb_text[body_start:i - 1]
    preamble = tb_text[:m.start()]
    return preamble, body


def _array_decls(body: str) -> list[tuple[str, str, str, str]]:
    """Find `<type> <name>[<dims>];` declarations where type is double/float/int/long.
    Returns list of (name, dims, raw, elem_type)."""
    out = []
    pat = r"\b(double|float|int|long|unsigned\s+int|unsigned\s+long)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(\[[^;]+\])\s*;"
    for m in re.finditer(pat, body):
        elem_type = m.group(1).strip()
        name = m.group(2)
        dims = m.group(3).strip()
        out.append((name, dims, m.group(0), elem_type))
    return out


def _scalar_decls(body: str) -> list[tuple[str, str, str]]:
    """Find `double <name>;` scalar declarations (no brackets, no assignment)."""
    out = []
    for m in re.finditer(r"\bdouble\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*;", body):
        out.append((m.group(1), "scalar", m.group(0)))
    return out


def _find_call(body: str, fn_name: str) -> tuple[int, int, str] | None:
    """Find a call to `fn_name(...)` in body. Returns (start, end, call_text)."""
    m = re.search(rf"\b{re.escape(fn_name)}\s*\(", body)
    if not m:
        return None
    start = m.start()
    depth = 0
    i = m.end() - 1
    while i < len(body):
        c = body[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                # consume optional `;`
                j = i + 1
                while j < len(body) and body[j] in " \t\n":
                    j += 1
                if j < len(body) and body[j] == ";":
                    return start, j + 1, body[start:j + 1]
                return start, i + 1, body[start:i + 1]
        i += 1
    return None


def _parse_call_args(call_text: str) -> list[str]:
    """Split args of fn_name(arg1, arg2, ...) respecting paren depth."""
    open_i = call_text.index("(")
    close_i = call_text.rfind(")")
    inside = call_text[open_i + 1:close_i]
    args = []
    depth = 0
    cur = []
    for c in inside:
        if c == "(":
            depth += 1
            cur.append(c)
        elif c == ")":
            depth -= 1
            cur.append(c)
        elif c == "," and depth == 0:
            args.append("".join(cur).strip())
            cur = []
        else:
            cur.append(c)
    if cur:
        args.append("".join(cur).strip())
    return args


def _gold_arg(arg: str, array_names: set[str]) -> str:
    """Rewrite a call arg to use _gold version if it references an array name.
    Handles: `name`, `&name`, `name[...]` (rare in testbench calls)."""
    # `&scalar` — pass same scalar (kernel won't modify scalars in HLS)
    if arg.startswith("&"):
        return arg
    # bare name
    if arg in array_names:
        return f"{arg}_gold"
    return arg


def _emit_cosim_testbench(bench_dir: Path, meta: dict) -> str | None:
    tb_file = bench_dir / (meta.get("testbench_file") or "testbench.cpp")
    if not tb_file.exists():
        return None
    tb_text = tb_file.read_text()
    try:
        preamble, body = _parse_main_body(tb_text)
    except ValueError as e:
        print(f"  ! parse error: {e}")
        return None

    kernel_top = meta["kernel_top"]
    gold_kernel = f"{kernel_top}_gold"

    # Find init/kernel/print calls
    init_call = _find_call(body, "init_array")
    kernel_call = _find_call(body, kernel_top)
    print_call = _find_call(body, "print_array")
    if not (init_call and kernel_call and print_call):
        print(f"  ! missing init/kernel/print call (init={bool(init_call)}, "
              f"kernel={bool(kernel_call)}, print={bool(print_call)})")
        return None

    # Array decls in main()
    arrays = _array_decls(body)
    array_names = {n for n, _, _, _ in arrays}
    array_type = {n: t for n, _, _, t in arrays}
    if not arrays:
        print(f"  ! no array decls found")
        return None

    # Build inserted text after each section
    # 1) Duplicate array decls with _gold suffix
    dup_decls = []
    for name, dims, raw, elem_type in arrays:
        dup_decls.append(f"  {elem_type} {name}_gold{dims};")
    dup_decl_block = "\n".join(dup_decls)

    # 2) Second init_array call on _gold args
    init_args = _parse_call_args(init_call[2])
    init_gold_args = [_gold_arg(a, array_names) for a in init_args]
    init_gold_call = f"init_array({', '.join(init_gold_args)});"

    # 3) Second kernel call on _gold args with gold kernel name
    kernel_args = _parse_call_args(kernel_call[2])
    kernel_gold_args = [_gold_arg(a, array_names) for a in kernel_args]
    kernel_gold_call = f"{gold_kernel}({', '.join(kernel_gold_args)});"

    # 4) Replace print_array with diff loop. Use the arrays in the print call
    #    as the "output" set. Print's first args are usually scalar bounds.
    print_args = _parse_call_args(print_call[2])
    output_arrays = [a for a in print_args if a in array_names]
    if not output_arrays:
        print(f"  ! print_array has no array args ({print_args})")
        return None

    # Diff snippet: walk each output array linearly. We get total element count
    # from sizeof(arr)/sizeof(arr[0][0]) for 2D, /sizeof(arr[0]) for 1D — but
    # simpler: use sizeof(arr)/sizeof(double) since base type is always double.
    diff_lines = ["  int __mismatches = 0;",
                  f"  const double __rel_tol = {REL_TOL};",
                  f"  const double __abs_tol = {ABS_TOL};"]
    for out in output_arrays:
        et = array_type.get(out, "double")
        # Integer arrays: exact match required (no tolerance)
        is_int = "int" in et or "long" in et
        diff_lines.append(f"  {{")
        diff_lines.append(f"    const {et}* __cand = (const {et}*)&{out}[0];")
        diff_lines.append(f"    const {et}* __gold = (const {et}*)&{out}_gold[0];")
        diff_lines.append(f"    size_t __n = sizeof({out}) / sizeof({et});")
        diff_lines.append(f"    for (size_t __k = 0; __k < __n; __k++) {{")
        if is_int:
            diff_lines.append(f"      if (__cand[__k] != __gold[__k]) {{")
            diff_lines.append(f"        if (__mismatches < 8) printf(\"diff {out}[%zu]: cand=%lld gold=%lld\\n\", __k, (long long)__cand[__k], (long long)__gold[__k]);")
            diff_lines.append(f"        __mismatches++;")
            diff_lines.append(f"      }}")
        else:
            diff_lines.append(f"      double a = (double)__cand[__k], b = (double)__gold[__k];")
            diff_lines.append(f"      double d = a - b; if (d < 0) d = -d;")
            diff_lines.append(f"      double r = (b < 0 ? -b : b); if (r < 1.0) r = 1.0;")
            diff_lines.append(f"      if (d > __abs_tol && d / r > __rel_tol) {{")
            diff_lines.append(f"        if (__mismatches < 8) printf(\"diff {out}[%zu]: cand=%.6g gold=%.6g delta=%.3g\\n\", __k, a, b, d);")
            diff_lines.append(f"        __mismatches++;")
            diff_lines.append(f"      }}")
        diff_lines.append(f"    }}")
        diff_lines.append(f"  }}")
    diff_lines.append(f"  if (__mismatches > 0) {{ printf(\"FAIL: %d mismatches\\n\", __mismatches); return 1; }}")
    diff_lines.append(f"  printf(\"PASS\\n\");")
    diff_block = "\n".join(diff_lines)

    # Now stitch the new body together
    parts = []
    # Everything from start of body through end of init_call
    parts.append(body[:init_call[1]])
    parts.append("\n\n  // === cosim: duplicate arrays + reinit gold copies ===\n")
    parts.append(dup_decl_block + "\n")
    parts.append("  " + init_gold_call + "\n")
    # Between init_call end and kernel_call start (whitespace/comments)
    parts.append(body[init_call[1]:kernel_call[1]])
    parts.append("\n\n  // === cosim: run gold reference kernel ===\n")
    parts.append("  " + kernel_gold_call + "\n")
    # Between kernel_call end and print_call start
    parts.append(body[kernel_call[1]:print_call[0]])
    parts.append("\n  // === cosim: diff candidate vs gold ===\n")
    parts.append(diff_block + "\n")
    # Skip the print_call entirely, keep tail after it (the `return 0;`)
    parts.append(body[print_call[1]:])

    new_body = "".join(parts)

    # Forward-declare the gold kernel using the same signature as kernel_top.
    # We extract the signature from the bench header verbatim.
    header_file = bench_dir / (meta.get("header_file") or "")
    if not header_file.exists() or not header_file.is_file():
        print(f"  ! header not found: {header_file}")
        return None
    header_text = header_file.read_text()
    # Find `void kernel_<bench>(...);` (possibly multi-line)
    sig_m = re.search(rf"void\s+{re.escape(kernel_top)}\s*\(([^;]+?)\)\s*;",
                      header_text, flags=re.DOTALL)
    if not sig_m:
        print(f"  ! could not find kernel signature in header")
        return None
    sig_args = sig_m.group(1)
    gold_fwd = f"extern \"C\" void {gold_kernel}({sig_args});"

    # Compose the final file
    out = []
    out.append("// AUTO-GENERATED by gen_cosim_testbenches.py — do not edit.\n")
    out.append("// Authentic cosim driver: runs candidate + gold kernels on\n")
    out.append("// identical inputs and exits 0 (PASS) / 1 (FAIL).\n\n")
    out.append(preamble)
    out.append(f"\n// Forward declaration of gold reference kernel (defined in gold_kernel_for_cosim.cpp)\n")
    out.append(gold_fwd + "\n\n")
    out.append("int main(int argc, char** argv)\n{\n")
    out.append(new_body)
    out.append("}\n")
    return "".join(out)


def _emit_gold_shim(bench_dir: Path, meta: dict) -> str | None:
    """Emit a translation unit that defines kernel_<bench>_gold by including
    the gold source with the symbol renamed via macro."""
    kernel_top = meta["kernel_top"]
    gold_kernel = f"{kernel_top}_gold"
    gold_src_name = meta.get("gold_hls_source_file") or "gold_hls_source.cpp"
    out = [
        "// AUTO-GENERATED by gen_cosim_testbenches.py — do not edit.\n",
        "// Defines the gold reference kernel under a renamed symbol so the\n",
        "// cosim testbench can call both candidate and gold side-by-side.\n",
        "\n",
        f"#define {kernel_top} {gold_kernel}\n",
        f"#include \"{gold_src_name}\"\n",
        f"#undef {kernel_top}\n",
    ]
    return "".join(out)


def _process_bench(bench_dir: Path, dry_run: bool) -> dict:
    meta_path = bench_dir / "metadata.json"
    if not meta_path.exists():
        return {"bench": bench_dir.name, "skipped": "no-metadata"}
    meta = json.loads(meta_path.read_text())

    cosim_tb = _emit_cosim_testbench(bench_dir, meta)
    if cosim_tb is None:
        return {"bench": bench_dir.name, "skipped": "gen-failed"}
    gold_shim = _emit_gold_shim(bench_dir, meta)
    if gold_shim is None:
        return {"bench": bench_dir.name, "skipped": "gold-shim-failed"}

    cosim_tb_path = bench_dir / "testbench_cosim.cpp"
    gold_shim_path = bench_dir / "gold_kernel_for_cosim.cpp"

    if not dry_run:
        cosim_tb_path.write_text(cosim_tb)
        gold_shim_path.write_text(gold_shim)
        meta["cosim_testbench_file"] = "testbench_cosim.cpp"
        meta["cosim_support_files"] = ["gold_kernel_for_cosim.cpp"]
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    return {"bench": bench_dir.name, "ok": True,
            "tb_bytes": len(cosim_tb), "shim_bytes": len(gold_shim)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bench", help="single bench name (without hlsfactory_ prefix)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if args.bench:
        name = args.bench if args.bench.startswith("hlsfactory_") else f"hlsfactory_{args.bench}"
        bench_dirs = [BENCH_ROOT / name]
    else:
        bench_dirs = sorted(BENCH_ROOT.glob("hlsfactory_*"))

    n_ok = n_skip = 0
    for bd in bench_dirs:
        if not bd.is_dir():
            continue
        r = _process_bench(bd, args.dry_run)
        if "skipped" in r:
            n_skip += 1
            print(f"SKIP {bd.name}: {r['skipped']}")
        else:
            n_ok += 1
            tag = " [dry-run]" if args.dry_run else ""
            print(f"OK   {bd.name}: tb={r['tb_bytes']}B shim={r['shim_bytes']}B{tag}")
    print()
    print(f"OK={n_ok}  SKIP={n_skip}")


if __name__ == "__main__":
    main()
