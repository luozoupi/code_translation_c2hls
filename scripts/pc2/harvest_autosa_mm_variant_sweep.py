#!/usr/bin/env python3
"""Harvest autosa_mm variant-sweep QoR into sweep_qor.csv.

Quote latency min/max (latency_cycles / latency_cycles_worst) and DSP per stage.
Never emit an interval field.

Usage:
  python scripts/pc2/harvest_autosa_mm_variant_sweep.py \\
    --root artifacts/pc2/autosa_mm_variant_sweep_YYYYMMDD
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "pc2"))

from autosa_mm_variant_sweep import (  # noqa: E402
    CSV_FIELDS,
    find_variant_cell,
    load_cells,
    qor_from_report,
    stage_report_paths,
)

STAGES = ("flash", "dse", "stream", "enf", "oneshot", "mem")


def harvest_cell(cell: dict[str, Any], *, campaign_root: Path | None = None) -> list[dict[str, Any]]:
    """Parse stage reports under one sweep cell. Safe on fake JSON trees."""
    root = Path(campaign_root or cell.get("campaign_root") or "")
    variant_cell = find_variant_cell(root) if root.is_dir() else None
    rows: list[dict[str, Any]] = []
    reports = stage_report_paths(variant_cell) if variant_cell is not None else {}
    family = cell.get("family") or ""
    if family == "oneshot":
        wanted = ["oneshot"]
    elif family == "mem":
        wanted = ["mem"]
    else:
        wanted = [s for s in STAGES if s not in {"oneshot", "mem"}]
    for stage in wanted:
        path = reports.get(stage)
        if path is None or not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        qor = qor_from_report(data)
        if qor is None:
            continue
        rows.append(
            {
                "family": family,
                "skill": cell.get("skill") or "",
                "floor": int(cell.get("floor") or 0),
                "dse": cell.get("dse") or "",
                "stream": int(cell.get("stream") or 0),
                "enf": int(cell.get("enf") or 0),
                "mem": int(cell.get("mem") or 0),
                "rep": int(cell.get("rep") or 0),
                "stage": stage,
                "latency_cycles": qor["latency_cycles"],
                "latency_cycles_worst": qor["latency_cycles_worst"],
                "dsp": qor["dsp"],
                "bram": qor["bram"],
                "lut": qor["lut"],
                "ff": qor["ff"],
                "csim": qor["csim"],
                "campaign_root": str(root),
            }
        )
    return rows


def harvest_sweep(root: Path) -> list[dict[str, Any]]:
    cells_path = root / "cells.jsonl"
    cells = load_cells(cells_path) if cells_path.is_file() else []
    rows: list[dict[str, Any]] = []
    for cell in cells:
        rows.extend(harvest_cell(cell))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(CSV_FIELDS), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Sweep root containing cells.jsonl",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="CSV path (default: ROOT/sweep_qor.csv)",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    rows = harvest_sweep(root)
    out = args.out.resolve() if args.out else root / "sweep_qor.csv"
    write_csv(out, rows)
    print(f"rows={len(rows)} csv={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
