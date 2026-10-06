#!/usr/bin/env python3
"""Compare Multistep aav_n vs msss on autosa_mm by csynth worst-case latency."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]


def _latency_worst(report: dict[str, Any] | None) -> int | None:
    if not isinstance(report, dict):
        return None
    lat = report.get("latency_cycles_worst")
    if lat is None:
        lat = report.get("latency_cycles")
    try:
        return int(lat) if lat is not None else None
    except (TypeError, ValueError):
        return None


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def collect_variant(campaign_root: Path, variant: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cell_root = campaign_root / "variants" / variant
    if not cell_root.is_dir():
        return rows
    for bench_dir in sorted(cell_root.glob("autosa_mm_r*")):
        if not bench_dir.is_dir():
            continue
        result_path = next(bench_dir.glob("**/autosa_mm_r*_multistep_results.json"), None)
        if result_path is None:
            result_path = next(bench_dir.rglob("*_multistep_results.json"), None)
        payload = _load_json(result_path) if result_path else None
        lat = _latency_worst((payload or {}).get("final_report"))
        rows.append(
            {
                "bench": bench_dir.name,
                "campaign": str(campaign_root),
                "variant": variant,
                "result_path": str(result_path) if result_path else "",
                "success": bool(payload and payload.get("success") and lat is not None),
                "latency_cycles_worst": lat,
            }
        )
    return rows


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [int(r["latency_cycles_worst"]) for r in rows if r.get("latency_cycles_worst") is not None]
    failed = [r["bench"] for r in rows if r.get("latency_cycles_worst") is None]
    out: dict[str, Any] = {
        "n": len(rows),
        "n_ok": len(ok),
        "failed": failed,
        "latencies": ok,
    }
    if ok:
        out["min"] = min(ok)
        out["max"] = max(ok)
        out["median"] = statistics.median(ok)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aav-n", required=True, type=Path)
    parser.add_argument("--msss", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    aav_rows = collect_variant(args.aav_n, "autosa_ms_aav_n")
    msss_rows = collect_variant(args.msss, "autosa_ms_msss")
    doc = {
        "metric": "csynth Worst-caseLatency (latency_cycles_worst)",
        "aav_n": {"campaign": str(args.aav_n), **_summary(aav_rows), "runs": aav_rows},
        "msss": {"campaign": str(args.msss), **_summary(msss_rows), "runs": msss_rows},
    }
    text = json.dumps(doc, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
