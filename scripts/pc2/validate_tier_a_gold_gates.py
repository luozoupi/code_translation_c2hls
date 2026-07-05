#!/usr/bin/env python3
"""Run validate_gold_reference for tier_A_ready benches (no LLM)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PC2 = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(PC2) not in sys.path:
    sys.path.insert(0, str(PC2))

from c2hls import _load_benchmark_inputs, validate_gold_reference  # noqa: E402
from tier_a_gold_gate_audit import TIER_A_READY_ROOT, iter_tier_a_benches  # noqa: E402
from tier_a_flash_lib import apply_bench_synth_timeout_from_meta  # noqa: E402


def _validate_one(bench: str, *, corpus_root: Path) -> dict:
    bench_dir = corpus_root / bench
    inputs = _load_benchmark_inputs(str(bench_dir))
    apply_bench_synth_timeout_from_meta(inputs.get("meta") or {})
    ref = validate_gold_reference(inputs)
    return {
        "bench": bench,
        "gold_pass": bool(ref.get("benchmark_ready")),
        "invalid_reason": ref.get("invalid_reason") or "",
        "top_function": ref.get("top_function") or inputs["meta"].get("hls_top", ""),
        "synthesis": (ref.get("synthesis") or {}).get("status"),
        "csim": (ref.get("csim") or {}).get("status"),
        "reference_validation": ref,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", action="append", dest="benches", default=[])
    parser.add_argument("--all", action="store_true", help="Validate every tier_A_ready bench")
    parser.add_argument(
        "--corpus-root",
        type=Path,
        default=TIER_A_READY_ROOT,
        help="tier_A_ready root (default: repo corpus)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory for per-bench JSON + matrix.json",
    )
    args = parser.parse_args()

    if args.all:
        names = iter_tier_a_benches(args.corpus_root)
    elif args.benches:
        names = list(args.benches)
    else:
        parser.error("pass --bench NAME and/or --all")

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for bench in names:
        print(f"[gold] validating {bench}...", flush=True)
        row = _validate_one(bench, corpus_root=args.corpus_root)
        rows.append(row)
        (args.out / f"{bench}.json").write_text(
            json.dumps(row, indent=2) + "\n",
            encoding="utf-8",
        )
        status = "PASS" if row["gold_pass"] else "FAIL"
        print(f"[gold] {bench}: {status}", flush=True)
        if not row["gold_pass"]:
            print(f"       {row['invalid_reason'][:200]}", flush=True)

    passed = sum(1 for r in rows if r["gold_pass"])
    matrix = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "corpus_root": str(args.corpus_root.resolve()),
        "total": len(rows),
        "gold_pass": passed,
        "gold_fail": len(rows) - passed,
        "rows": [
            {
                "bench": r["bench"],
                "gold_pass": r["gold_pass"],
                "synthesis": r["synthesis"],
                "csim": r["csim"],
                "invalid_reason": r["invalid_reason"],
            }
            for r in rows
        ],
    }
    (args.out / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n", encoding="utf-8")
    print(f"\n[gold] matrix: {passed}/{len(rows)} pass -> {args.out / 'matrix.json'}", flush=True)
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
