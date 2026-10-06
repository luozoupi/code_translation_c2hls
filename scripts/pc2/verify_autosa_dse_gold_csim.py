#!/usr/bin/env python3
"""Verify gold csim for one benchmarks_autosa_dse package via hls_eval path."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from hls_eval import run_csim  # noqa: E402


def main() -> int:
    bench = Path(sys.argv[1] if len(sys.argv) > 1 else REPO / "benchmarks_autosa_dse/autosa_mm_rank1")
    meta = json.loads((bench / "metadata.json").read_text(encoding="utf-8"))
    support = meta.get("support_files") or []
    print(f"bench={bench}")
    print(f"support_files={support}")
    if any(Path(s).name == "kernel_kernel_modules.cpp" for s in support):
        print("ERROR: modules still listed in support_files", file=sys.stderr)
        return 2

    gold = (bench / (meta.get("gold_hls_baseline_file") or "hls_baseline.cpp")).read_text(encoding="utf-8")
    header = (bench / (meta.get("header_file") or "kernel.h")).read_text(encoding="utf-8")
    tb = (bench / (meta.get("testbench_file") or "testbench.cpp")).read_text(encoding="utf-8")
    extra = []
    for rel in support:
        p = bench / rel
        if p.is_file():
            extra.append({"path": rel, "content": p.read_text(encoding="utf-8")})

    work = REPO / "c2hls_tmp" / f"verify_gold_csim_{bench.name}"
    if work.exists():
        import shutil

        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)

    result = run_csim(
        hls_code=gold,
        testbench_code=tb,
        header_code=header,
        header_name=meta.get("header_file") or "kernel.h",
        top_function=meta.get("hls_top") or "kernel0",
        part=meta.get("target_part") or "xcu280-fsvh2892-2L-e",
        clock_ns=float(meta.get("target_clock_ns") or 3.33),
        work_dir=str(work),
        extra_files=extra,
    )
    out = {
        "bench": bench.name,
        "success": result.get("success"),
        "passed": result.get("passed"),
        "error": result.get("error"),
        "work_dir": result.get("work_dir"),
        "support_files": support,
    }
    (work / "verify_summary.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
