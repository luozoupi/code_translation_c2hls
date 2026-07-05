#!/usr/bin/env python3
"""Merge per-bench JSON from a tier_a_gold_verify_* array run into matrix.json."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} artifacts/pc2/tier_a_gold_verify_STAMP", file=sys.stderr)
        return 1

    out = Path(sys.argv[1])
    rows = []
    for path in sorted(out.glob("*.json")):
        if path.name == "matrix.json":
            continue
        row = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "bench": row.get("bench", path.stem),
                "gold_pass": bool(row.get("gold_pass")),
                "synthesis": row.get("synthesis"),
                "csim": row.get("csim"),
                "invalid_reason": row.get("invalid_reason", ""),
            }
        )

    passed = sum(1 for r in rows if r["gold_pass"])
    matrix = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(out.resolve()),
        "total": len(rows),
        "gold_pass": passed,
        "gold_fail": len(rows) - passed,
        "rows": rows,
    }
    (out / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n", encoding="utf-8")
    print(f"matrix: {passed}/{len(rows)} pass -> {out / 'matrix.json'}")
    for r in rows:
        if not r["gold_pass"]:
            reason = (r["invalid_reason"] or "").replace("\n", " ")[:140]
            print(f"  FAIL {r['bench']}: synth={r['synthesis']} csim={r['csim']} — {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
