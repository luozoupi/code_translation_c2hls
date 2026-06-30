#!/usr/bin/env python3
"""Export multistep csynth speedup tables (per-step progression + selected)."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import median

REPO = Path(__file__).resolve().parents[2]
DEFAULT_BASELINE_JSONL = REPO / "misc/hlsfactory_baseline_u280_20260616_benchmarks_full_cosim.jsonl"

OPT_STEPS = ["tiling", "pipeline", "unroll", "doublebuffer", "coalescing"]


def norm_bench(name: str) -> str:
    s = name.removeprefix("hlsfactory_")
    return re.sub(r"[-_]", "", s.lower())


def display_bench(nb: str) -> str:
    aliases = {
        "fdtd2d": "fdtd-2d",
        "floydwarshall": "floyd-warshall",
        "heat3d": "heat-3d",
        "jacobi1d": "jacobi-1d",
        "jacobi2d": "jacobi-2d",
        "seidel2d": "seidel-2d",
    }
    return aliases.get(nb, nb)


def load_bench_jsonl(path: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("report_type") != "hls_synth":
            continue
        gp = rec.get("problem", {}).get("group_path") or []
        if not gp:
            continue
        pe = rec["hls_synth"]["PerformanceEstimates"]["SummaryOfOverallLatency"]
        out[norm_bench(gp[0])] = int(pe["Worst-caseLatency"])
    return out


def geomean(xs: list[float]) -> float | None:
    xs = [x for x in xs if x and x > 0]
    if not xs:
        return None
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def collect(stamp: str, bench_jsonl: dict[str, int], *, variant: str = "aav_n") -> tuple[list[dict], list[dict], list[dict]]:
    root = REPO / "artifacts/pc2" / f"multistep_fixed_cosim_{variant}_{stamp}"
    if not root.is_dir():
        matches = sorted(REPO.glob(f"artifacts/pc2/multistep_fixed_cosim_{variant}_*{stamp}*"))
        if not matches:
            matches = sorted(REPO.glob(f"artifacts/pc2/multistep_fixed_cosim_*_{stamp}"))
        root = matches[0] if matches else root

    per_bench: list[dict] = []
    progression_rows: list[dict] = []
    aggregate: list[dict] = []

    sp_gt: list[float] = []
    sp_pb: list[float] = []
    ok = fail = 0

    for f in sorted(root.glob("hlsfactory_*/*/*_multistep_results.json")):
        bench_dir = f.parts[-3]
        nb = norm_bench(bench_dir)
        r = json.loads(f.read_text(encoding="utf-8"))
        if not r.get("success"):
            fail += 1
            per_bench.append({"bench": display_bench(nb), "status": "fail", "error": r.get("error", "")})
            continue
        ok += 1
        pb = (r.get("baseline_report") or {}).get("latency_cycles")
        gt = (r.get("ground_truth_report") or {}).get("latency_cycles")
        sel = (r.get("final_report") or {}).get("latency_cycles")
        bb = bench_jsonl.get(nb)
        row = {
            "bench": display_bench(nb),
            "status": "ok",
            "phase_b_cycles": pb,
            "selected_cycles": sel,
            "gt_cycles": gt,
            "benchmark_cycles": bb,
            "speedup_vs_gt": gt / sel if gt and sel else None,
            "speedup_vs_phase_b": pb / sel if pb and sel else None,
            "speedup_vs_benchmark": bb / sel if bb and sel else None,
        }
        flow_path = f.with_name(f.name.replace("_multistep_results.json", "_flow_manifest.json"))
        if flow_path.is_file():
            flow = json.loads(flow_path.read_text())
            row["selected_from"] = flow.get("selected_from")
            lat = flow.get("latency_cycles") or {}
            for step in OPT_STEPS:
                row[f"{step}_cycles"] = lat.get(step)
                if pb and lat.get(step):
                    row[f"{step}_speedup_vs_phase_b"] = pb / lat[step]
                if gt and lat.get(step):
                    row[f"{step}_speedup_vs_gt"] = gt / lat[step]
            progression_rows.append({"bench": display_bench(nb), **{k: row.get(k) for k in row if k.endswith("_cycles")}})
        if row.get("speedup_vs_gt"):
            sp_gt.append(row["speedup_vs_gt"])
        if row.get("speedup_vs_phase_b"):
            sp_pb.append(row["speedup_vs_phase_b"])
        per_bench.append(row)

    aggregate.append(
        {
            "variant": variant,
            "ok": ok,
            "fail": fail,
            "gmean_speedup_vs_gt": geomean(sp_gt),
            "gmean_speedup_vs_phase_b": geomean(sp_pb),
            "median_speedup_vs_gt": median(sp_gt) if sp_gt else None,
            "median_speedup_vs_phase_b": median(sp_pb) if sp_pb else None,
        }
    )
    return per_bench, progression_rows, aggregate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stamp", default="", required=True)
    parser.add_argument("--variant", default="aav_n")
    parser.add_argument("--baseline-jsonl", default=str(DEFAULT_BASELINE_JSONL))
    parser.add_argument("--out-dir", default="")
    args = parser.parse_args()

    bench_jsonl = load_bench_jsonl(Path(args.baseline_jsonl))
    per_bench, progression, aggregate = collect(args.stamp, bench_jsonl, variant=args.variant)

    out_dir = Path(args.out_dir) if args.out_dir else REPO / "artifacts/pc2/analysis" / args.stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    if per_bench:
        keys = sorted({k for row in per_bench for k in row})
        with (out_dir / "csynth_speedup_per_bench.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(per_bench)

    if progression:
        keys = sorted({k for row in progression for k in row})
        with (out_dir / "csynth_speedup_step_progression.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(progression)

    if aggregate:
        keys = sorted({k for row in aggregate for k in row})
        with (out_dir / "csynth_speedup_aggregate.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(aggregate)

    print(json.dumps({"out_dir": str(out_dir), "benches_ok": len([r for r in per_bench if r.get("status") == "ok"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
