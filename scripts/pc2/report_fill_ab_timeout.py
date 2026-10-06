#!/usr/bin/env python3
"""Write annotated fill CSVs for csynth_full and cosim_small."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from fill_ab_timeout_lib import FILL_ROOT, SCOPE, TIMEOUT_REASON, job_result_path  # noqa: E402

FIELDS = [
    "bench",
    "model",
    "metric_kind",
    "problem_size_note",
    "gold_cycles_or_latency",
    "flash_cycles_or_latency",
    "speedup_gold_over_flash",
    "gold_status",
    "flash_status",
    "timeout_reason",
    "source_gold",
    "source_flash",
    "included_in_geomean",
]


def _num(v):
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _geomean(vals: list[float]) -> float | None:
    vals = [v for v in vals if v is not None and v > 0]
    if not vals:
        return None
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def _load_side(model_tag: str, bench: str, metric_kind: str, side: str) -> dict:
    job = {
        "metric_kind": metric_kind,
        "model_tag": model_tag,
        "bench": bench,
        "side": side,
    }
    path = job_result_path(job)
    if not path.exists():
        return {"status": "missing", "path": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(model_tag: str, metric_kind: str) -> Path:
    out_dir = FILL_ROOT / "csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_tag}_fill_{metric_kind}.csv"
    original = {
        "deepseek_v4_flash_aav_n": "deepseek_v4_flash_aav_n_flash_cosim_speedup_vs_gold.csv",
        "devstral2_aav_n": "devstral2_aav_n_flash_cosim_speedup_vs_gold.csv",
    }[model_tag]

    rows = []
    speedups = []
    for bench in SCOPE[model_tag]:
        gold = _load_side(model_tag, bench, metric_kind, "gold")
        flash = _load_side(model_tag, bench, metric_kind, "flash")
        if metric_kind == "csynth_full":
            g = _num(gold.get("latency_cycles"))
            f = _num(flash.get("latency_cycles"))
            size_note = (gold.get("job") or flash.get("job") or {}).get("problem_size_note") or "full"
        else:
            g = _num(gold.get("kernel_runtime_cycles"))
            f = _num(flash.get("kernel_runtime_cycles"))
            size_note = (gold.get("job") or flash.get("job") or {}).get("problem_size_note") or "small"
        sp = (g / f) if (g and f and f > 0) else None
        included = sp is not None and sp > 0
        if included:
            speedups.append(sp)
        flash_src = ""
        bench_dir = FILL_ROOT / ("benches_full" if metric_kind == "csynth_full" else "benches_small") / model_tag / bench
        src_json = bench_dir / "flash_kernel_source.json"
        if src_json.exists():
            flash_src = json.loads(src_json.read_text()).get("note", "")
        rows.append(
            {
                "bench": bench,
                "model": model_tag,
                "metric_kind": metric_kind,
                "problem_size_note": size_note,
                "gold_cycles_or_latency": "" if g is None else str(int(g) if g == int(g) else g),
                "flash_cycles_or_latency": "" if f is None else str(int(f) if f == int(f) else f),
                "speedup_gold_over_flash": "" if sp is None else f"{sp:.6f}".rstrip("0").rstrip("."),
                "gold_status": gold.get("status", "missing"),
                "flash_status": flash.get("status", "missing"),
                "timeout_reason": TIMEOUT_REASON[(model_tag, bench)],
                "source_gold": "hls_baseline_cosim.cpp (copy)",
                "source_flash": flash_src,
                "included_in_geomean": str(included),
            }
        )

    gm = _geomean(speedups)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
        w.writerow(
            {
                "bench": "__GEOMEAN__",
                "model": model_tag,
                "metric_kind": metric_kind,
                "problem_size_note": "",
                "gold_cycles_or_latency": "",
                "flash_cycles_or_latency": "",
                "speedup_gold_over_flash": "" if gm is None else f"{gm:.6f}".rstrip("0").rstrip("."),
                "gold_status": "",
                "flash_status": "",
                "timeout_reason": "",
                "source_gold": "",
                "source_flash": f"n={len(speedups)} included fill benches only",
                "included_in_geomean": "True" if gm is not None else "False",
            }
        )
        f.write(f"metric_kind={metric_kind}\n")
        if metric_kind == "csynth_full":
            f.write(
                "note=full-size average csynth latency (gold vs flash); "
                "used because gold cosim and/or flash cosim timed out or was missing "
                "on the original full-size run\n"
            )
        else:
            f.write(
                "note=reduced-size cosim cycles (gold vs flash); "
                "used because gold cosim and/or flash cosim timed out or was missing "
                "on the original full-size run\n"
            )
        f.write(f"original_csv=artifacts/pc2/reports/hlsfactory_flash_cosim_vs_gold_latrag_off/{original}\n")
        f.write("copies_only=benchmarks_cosim untouched; see fill_ab_20260730/benches_{full,small}\n")
    return out_path


def main() -> int:
    paths = []
    for model_tag in SCOPE:
        for metric_kind in ("csynth_full", "cosim_small"):
            paths.append(_write_csv(model_tag, metric_kind))
    for p in paths:
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
