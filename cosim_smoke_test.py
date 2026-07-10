"""Smoke test for the macroized cosim pipeline.

After macroize_benches.py + gen_cosim_testbenches.py have been applied,
this verifies the full chain for ONE bench (default: gemm):

  1. csim at FULL size using the original testbench (catches macro bugs:
     if our header/gold/testbench substitution broke the default size,
     csim against gold would FAIL — same numerical output as before is
     required to PASS).
  2. csim at SHRUNK size using -D overrides on header (sanity check
     that the macros propagate to dim brackets and loop bounds).
  3. cosim at SHRUNK size using the gold-vs-candidate testbench
     (the authentic comparison: gold and candidate are the SAME source
     here, so it must PASS by construction; if it FAILS we've broken
     the wiring).

Run from the repo root:
    python3 cosim_smoke_test.py --bench gemm

Requires VITIS_SETTINGS env var pointing at the Vitis 2023.2 settings64.sh.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from hls_eval import run_csim, run_cosim, DEFAULT_PART, DEFAULT_CLOCK_NS


def _load_bench_assets(bench_dir: Path) -> dict:
    meta = json.loads((bench_dir / "metadata.json").read_text())
    header_file = bench_dir / meta["header_file"]
    gold_file = bench_dir / meta["gold_hls_source_file"]
    tb_file = bench_dir / meta["testbench_file"]

    cosim_tb_name = meta.get("cosim_testbench_file")
    cosim_extra_names = meta.get("cosim_support_files") or []
    cosim_overrides = meta.get("cosim_size_overrides") or {}

    assets = {
        "meta": meta,
        "kernel_top": meta["kernel_top"],
        "header_name": meta["header_file"],
        "header_code": header_file.read_text(),
        "gold_code": gold_file.read_text(),
        "testbench_code": tb_file.read_text(),
        "cosim_overrides": cosim_overrides,
    }
    if cosim_tb_name and (bench_dir / cosim_tb_name).exists():
        assets["cosim_testbench_code"] = (bench_dir / cosim_tb_name).read_text()
    else:
        assets["cosim_testbench_code"] = None
    cosim_extras = []
    for name in cosim_extra_names:
        p = bench_dir / name
        if p.exists():
            cosim_extras.append({"path": name, "content": p.read_text()})
    # Gold source needs to be materialized too (cosim shim #includes it),
    # but it must NOT be added as a separate TCL compile unit — the shim's
    # #include pulls it in, so a -tb add would create a duplicate definition.
    gold_src = meta.get("gold_hls_source_file")
    if cosim_extras and gold_src and (bench_dir / gold_src).exists():
        cosim_extras.append({"path": gold_src, "content": gold_file.read_text(), "compile": False})
    assets["cosim_extra_files"] = cosim_extras
    return assets


def _report(prefix: str, result: dict, key: str = "passed"):
    ok = result.get(key)
    err = (result.get("error") or "").splitlines()[0:2]
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {prefix}: success={result.get('success')} {key}={ok}")
    if not ok and err:
        print(f"         err: {' | '.join(err)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bench", default="gemm")
    p.add_argument("--skip-full", action="store_true", help="skip the full-size csim sanity")
    args = p.parse_args()

    bench_name = args.bench if args.bench.startswith("hlsfactory_") else f"hlsfactory_{args.bench}"
    bench_dir = REPO / "benchmarks" / bench_name
    if not bench_dir.is_dir():
        sys.exit(f"no such bench: {bench_dir}")

    A = _load_bench_assets(bench_dir)
    print(f"=== {bench_name} ===")
    print(f"  kernel_top={A['kernel_top']}  overrides={A['cosim_overrides']}")
    print(f"  cosim_tb={'YES' if A['cosim_testbench_code'] else 'NO (falls back to plain tb)'}")
    print(f"  cosim_extra_files={[f['path'] for f in A['cosim_extra_files']]}")
    print()

    common = dict(
        header_code=A["header_code"],
        header_name=A["header_name"],
        top_function=A["kernel_top"],
        part=DEFAULT_PART,
        clock_ns=DEFAULT_CLOCK_NS,
    )

    # 1) csim @ FULL — gold-vs-self (no overrides)
    if not args.skip_full:
        print("[1] csim @ full size (gold vs original testbench):")
        r = run_csim(A["gold_code"], A["testbench_code"], **common)
        _report("csim-full", r)
        print()

    # 2) csim @ SHRUNK
    if A["cosim_overrides"]:
        from hls_eval import _cflags_clause
        print(f"[2] csim @ shrunk size (overrides via -D macros):")
        # csim doesn't take size_overrides; we use cosim path which does.
        # But run_cosim also runs csynth — too slow for smoke. Just skip [2]
        # and rely on [3] cosim to exercise the override path.
        print("    (skipped — covered by [3])")
        print()

    # 3) cosim @ SHRUNK — must PASS by construction (gold-vs-self)
    if A["cosim_testbench_code"]:
        print("[3] cosim @ shrunk (gold-vs-gold using cosim testbench):")
        r = run_cosim(
            A["gold_code"],
            A["cosim_testbench_code"],
            extra_files=A["cosim_extra_files"],
            size_overrides=A["cosim_overrides"] or None,
            **common,
        )
        _report("cosim-shrunk-gold-vs-self", r)
        print()
    else:
        print("[3] cosim — SKIP (no cosim_testbench_file in metadata)")


if __name__ == "__main__":
    main()
