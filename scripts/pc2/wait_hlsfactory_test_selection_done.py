#!/usr/bin/env python3
"""Poll until HLSFactory test selection is done (not cosim).

Selection = campaign.selection_complete OR enough benches have
``{bench}_dataflow_candidate_ranking.json``.

Usage:
  .venv/bin/python scripts/pc2/wait_hlsfactory_test_selection_done.py \\
    --manifest PATH [--min-ranked N] [--poll-sec 120] [--max-wait-sec SEC]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _campaign_selection_stats(campaign_root: Path) -> dict[str, Any]:
    camp = _load_json(campaign_root / "campaign.json")
    if camp.get("selection_complete") is True:
        return {
            "campaign_root": str(campaign_root),
            "selection_complete": True,
            "ranked_benches": [],
            "n_ranked": -1,
        }
    ranked: list[str] = []
    variants = campaign_root / "variants"
    if variants.is_dir():
        for cell in sorted(variants.glob("*/*/*")):
            if not cell.is_dir():
                continue
            bench = cell.parent.name
            if (cell / f"{bench}_dataflow_candidate_ranking.json").is_file():
                ranked.append(bench)
    return {
        "campaign_root": str(campaign_root),
        "selection_complete": False,
        "ranked_benches": sorted(set(ranked)),
        "n_ranked": len(set(ranked)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Test sequence / arm manifest JSON")
    parser.add_argument(
        "--min-ranked",
        type=int,
        default=1,
        help="Min benches with dataflow ranking per campaign (ignored if selection_complete)",
    )
    parser.add_argument("--poll-sec", type=int, default=120)
    parser.add_argument("--max-wait-sec", type=int, default=0, help="0 = wait forever")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        print(f"ERROR: missing manifest {manifest_path}", file=sys.stderr)
        return 2

    t0 = time.time()
    while True:
        doc = _load_json(manifest_path)
        flavors = doc.get("flavors") or {}
        if not isinstance(flavors, dict) or not flavors:
            print(f"ERROR: no flavors in {manifest_path}", file=sys.stderr)
            return 2

        all_ok = True
        lines = []
        for flavor, meta in sorted(flavors.items()):
            if not isinstance(meta, dict):
                all_ok = False
                lines.append(f"{flavor}: bad_meta")
                continue
            root = Path(str(meta.get("campaign_root") or ""))
            if not root.is_dir():
                all_ok = False
                lines.append(f"{flavor}: missing_campaign")
                continue
            st = _campaign_selection_stats(root)
            if st["selection_complete"]:
                lines.append(f"{flavor}: selection_complete")
                continue
            n = int(st["n_ranked"])
            if n >= args.min_ranked:
                lines.append(f"{flavor}: ranked={n}")
                continue
            all_ok = False
            lines.append(f"{flavor}: ranked={n}/{args.min_ranked}")

        msg = " | ".join(lines)
        print(f"[selection] {msg}", flush=True)
        if all_ok:
            print("selection done for all flavors", flush=True)
            return 0

        if args.max_wait_sec > 0 and (time.time() - t0) >= args.max_wait_sec:
            print("ERROR: timed out waiting for selection", file=sys.stderr)
            return 1
        time.sleep(max(1, args.poll_sec))


if __name__ == "__main__":
    raise SystemExit(main())
