#!/usr/bin/env python3
"""Enumerate, HLS-validate, and rank compact PE×SIMD autosa_mm candidates.

No LLM. Writes ranking.jsonl plus selected.cpp / selected_report.json for rank-1.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from compact_pe_rank import rank_candidates
from compact_pe_search import enumerate_mm_mesh_recipes, enumerate_mm_recipes
from compact_pe_validate import validate_candidate

DEFAULT_PART = "xcu280-fsvh2892-2L-e"
DEFAULT_CLOCK_NS = 3.33


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stamp", required=True)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--header", required=True, type=Path)
    p.add_argument("--testbench", required=True, type=Path)
    args = p.parse_args(argv)

    out_root = args.out
    out_root.mkdir(parents=True, exist_ok=True)

    header_code = args.header.read_text(encoding="utf-8")
    testbench_code = args.testbench.read_text(encoding="utf-8")
    part = os.environ.get("C2HLS_PART", DEFAULT_PART)
    clock_ns = float(os.environ.get("C2HLS_CLOCK_NS", str(DEFAULT_CLOCK_NS)))

    recs = list(enumerate_mm_recipes()) + list(enumerate_mm_mesh_recipes())
    rows = []
    for rec in recs:
        rows.append(
            validate_candidate(
                rec,
                out_root,
                header_code,
                testbench_code,
                part=part,
                clock_ns=clock_ns,
            )
        )

    ranked = rank_candidates(rows)
    ranking_path = out_root / "ranking.jsonl"
    with ranking_path.open("w", encoding="utf-8") as fh:
        for row in ranked:
            fh.write(json.dumps(row) + "\n")

    if ranked:
        top = ranked[0]
        src = out_root / str(top["cand_id"]) / "kernel.cpp"
        shutil.copyfile(src, out_root / "selected.cpp")
        (out_root / "selected_report.json").write_text(
            json.dumps(top, indent=2) + "\n", encoding="utf-8"
        )

    print(
        f"stamp={args.stamp} out={out_root} candidates={len(rows)} ranked={len(ranked)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
