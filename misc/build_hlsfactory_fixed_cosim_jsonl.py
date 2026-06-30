#!/usr/bin/env python3
"""Build fixed benchmarks_cosim JSONL after csynth/csim base + PC2 full cosim.

Step 1 (PC2): start_hlsfactory_cosim_baseline_jsonl.sh
  -> misc/hlsfactory_cosim_baseline_u280_<stamp>.jsonl (hls_synth + sw_run)

Step 2 (PC2): start_baseline_fullsize_cosim.sh (already run for fixed corpus)

Step 3 (any host): this script merges rtl_sim from the fixed cosim campaign.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from export_schema_jsonl import validate_jsonl  # noqa: E402
from export_pc2_baseline_cosim_jsonl import (  # noqa: E402
    FIXED_COSIM_RUN,
    export_baseline_with_cosim,
)

PC2 = REPO / "artifacts" / "pc2"
DEFAULT_STAMP = "20260616_benchmarks"
DEFAULT_BASELINE_JSONL = REPO / "misc" / f"hlsfactory_cosim_baseline_u280_{DEFAULT_STAMP}.jsonl"
DEFAULT_OUTPUT = REPO / "misc" / f"hlsfactory_baseline_u280_{DEFAULT_STAMP}_full_cosim.jsonl"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return records


def _baseline_coverage(records: list[dict[str, Any]]) -> dict[str, set[str]]:
    by_group: dict[str, set[str]] = {}
    for rec in records:
        group = "/".join(rec.get("problem", {}).get("group_path") or ["unknown"])
        by_group.setdefault(group, set()).add(rec.get("report_type", ""))
    return by_group


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-jsonl", type=Path, default=DEFAULT_BASELINE_JSONL)
    parser.add_argument("--cosim-run-root", type=Path, default=FIXED_COSIM_RUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--allow-incomplete-cosim",
        action="store_true",
        help="Emit JSONL even if some manifest cells lack cosim_result.json",
    )
    parser.add_argument(
        "--allow-incomplete-baseline",
        action="store_true",
        help="Skip baseline hls_synth coverage check",
    )
    args = parser.parse_args()

    if not args.baseline_jsonl.is_file():
        raise SystemExit(
            f"missing baseline JSONL: {args.baseline_jsonl}\n"
            "Run scripts/pc2/start_hlsfactory_cosim_baseline_jsonl.sh on PC2 first."
        )

    baseline_records = _read_jsonl(args.baseline_jsonl)
    coverage = _baseline_coverage(baseline_records)
    missing_synth = sorted(g for g, types in coverage.items() if "hls_synth" not in types)
    if missing_synth and not args.allow_incomplete_baseline:
        raise SystemExit(f"baseline JSONL missing hls_synth for: {', '.join(missing_synth)}")

    records, meta = export_baseline_with_cosim(
        args.baseline_jsonl,
        args.cosim_run_root,
        allow_fixed_corpus=True,
    )

    if meta.get("missing_cosim") and not args.allow_incomplete_cosim:
        missing = ", ".join(meta["missing_cosim"])
        raise SystemExit(
            f"fixed cosim campaign incomplete ({len(meta['missing_cosim'])} missing): {missing}\n"
            "Re-submit missing cells or pass --allow-incomplete-cosim."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    validation = validate_jsonl(args.output, verbose=True)
    by_type: dict[str, int] = {}
    rtl_by_status: dict[str, int] = {}
    for rec in records:
        rt = rec.get("report_type", "")
        by_type[rt] = by_type.get(rt, 0) + 1
        if rt == "rtl_sim":
            s = rec.get("rtl_sim", {}).get("status", "")
            rtl_by_status[s] = rtl_by_status.get(s, 0) + 1

    summary = {
        **meta,
        "output": str(args.output),
        "baseline_coverage_groups": len(coverage),
        "by_report_type": by_type,
        "rtl_sim_by_status": rtl_by_status,
        "validation": validation,
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0 if validation.get("invalid", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
