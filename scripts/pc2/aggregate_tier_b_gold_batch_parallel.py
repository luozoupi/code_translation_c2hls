#!/usr/bin/env python3
"""Aggregate tier_B gold-gate batch_parallel campaign results into matrix.json."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _load_row(cell_dir: Path) -> dict:
    for name in ("gold_gate_results.json", "reference_validation.json"):
        path = cell_dir / name
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if name == "reference_validation.json":
            ref = payload
            synth = (ref.get("synthesis") or {}) if isinstance(ref.get("synthesis"), dict) else {}
            csim = (ref.get("csim") or {}) if isinstance(ref.get("csim"), dict) else {}
            return {
                "bench": cell_dir.parent.name,
                "gold_pass": bool(ref.get("benchmark_ready")),
                "synthesis": synth.get("status"),
                "csim": csim.get("status"),
                "invalid_reason": ref.get("invalid_reason") or "",
                "top_function": ref.get("top_function") or "",
                "cell_dir": str(cell_dir),
            }
        return {
            "bench": payload.get("benchmark") or cell_dir.parent.name,
            "gold_pass": bool(payload.get("gold_pass") or payload.get("success")),
            "synthesis": payload.get("synthesis"),
            "csim": payload.get("csim"),
            "invalid_reason": payload.get("invalid_reason") or "",
            "top_function": payload.get("top_function") or "",
            "cell_dir": str(cell_dir),
        }
    return {
        "bench": cell_dir.parent.name,
        "gold_pass": False,
        "synthesis": None,
        "csim": None,
        "invalid_reason": "no gold_gate_results.json or reference_validation.json",
        "top_function": "",
        "cell_dir": str(cell_dir),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_root", type=Path, help="batch_parallel campaign root")
    parser.add_argument(
        "--variant",
        default="tier_b_machsuite",
        help="Variant subdirectory under variants/",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default: <campaign>/reports/gold_gate)",
    )
    args = parser.parse_args()

    variant_root = args.campaign_root / "variants" / args.variant
    if not variant_root.is_dir():
        print(f"missing variant root: {variant_root}", file=sys.stderr)
        return 1

    rows = []
    for cell_dir in sorted(variant_root.glob("*/gold_gate")):
        if cell_dir.is_dir():
            rows.append(_load_row(cell_dir))

    out_dir = args.out or (args.campaign_root / "reports" / "gold_gate")
    out_dir.mkdir(parents=True, exist_ok=True)

    passed = sum(1 for r in rows if r["gold_pass"])
    matrix = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "campaign_root": str(args.campaign_root.resolve()),
        "variant": args.variant,
        "total": len(rows),
        "gold_pass": passed,
        "gold_fail": len(rows) - passed,
        "rows": rows,
    }
    (out_dir / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n", encoding="utf-8")

    for row in rows:
        bench = row["bench"]
        (out_dir / f"{bench}.json").write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")

    print(f"matrix: {passed}/{len(rows)} pass -> {out_dir / 'matrix.json'}")
    for row in rows:
        if not row["gold_pass"]:
            reason = (row["invalid_reason"] or "").replace("\n", " ")[:160]
            print(f"  FAIL {row['bench']}: synth={row['synthesis']} csim={row['csim']} — {reason}")
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
