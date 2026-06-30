#!/usr/bin/env python3
"""Export flash csynth speedup tables (CSV) for a fixed-cosim pipelined stamp."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import median

REPO = Path(__file__).resolve().parents[2]
DEFAULT_STAMP = "20260628_fixed_cosim_flash_r2_pipelined"
DEFAULT_BASELINE_JSONL = REPO / "misc/hlsfactory_baseline_u280_20260616_benchmarks_full_cosim.jsonl"

VARIANTS = [
    ("nav_o", "No avoids (old, 55)"),
    ("aav_n", "All+avoids (new, 90)"),
    ("nav_n", "No avoids (new, 73)"),
    ("noskills", "Noskills"),
    ("aav_o", "All+avoids (old, 55)"),
]

# Skill A/B pairs (variant_a, variant_b, label)
H2H_PAIRS = [
    ("aav_n", "nav_n", "All+avoids (new) vs No avoids (new)"),
    ("aav_n", "noskills", "All+avoids (new) vs Noskills"),
    ("nav_n", "noskills", "No avoids (new) vs Noskills"),
    ("aav_n", "aav_o", "All+avoids new vs old skills"),
    ("nav_n", "nav_o", "No avoids new vs old skills"),
    ("aav_o", "nav_o", "All+avoids (old) vs No avoids (old)"),
    ("aav_n", "nav_o", "All+avoids (new) vs No avoids (old)"),
    ("nav_n", "aav_o", "No avoids (new) vs All+avoids (old)"),
]


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


def sel_lat(r: dict) -> int | None:
    sel = r.get("final_report")
    if isinstance(sel, dict):
        v = sel.get("latency_cycles")
        return int(v) if v is not None else None
    return int(sel) if sel is not None else None


def geomean(xs: list[float]) -> float | None:
    xs = [x for x in xs if x and x > 0]
    if not xs:
        return None
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def collect(stamp: str, bench_jsonl: dict[str, int]) -> tuple[dict, list[dict]]:
    per_variant: dict[str, dict[str, dict]] = {k: {} for k, _ in VARIANTS}
    aggregate: list[dict] = []

    for key, label in VARIANTS:
        root = REPO / "artifacts/pc2" / f"flash_fixed_cosim_{key}_{stamp}"
        sp_b, sp_g, sp_p = [], [], []
        ok = fail = 0
        for f in sorted(root.glob("hlsfactory_*/*/*_multistep_results.json")):
            bench_dir = f.parts[-3]
            nb = norm_bench(bench_dir)
            r = json.loads(f.read_text(encoding="utf-8"))
            if not r.get("success"):
                fail += 1
                per_variant[key][nb] = {
                    "bench": display_bench(nb),
                    "status": "fail",
                    "error": r.get("error", ""),
                }
                continue
            ok += 1
            sel = sel_lat(r)
            gt = (r.get("ground_truth_report") or {}).get("latency_cycles")
            pb = (r.get("baseline_report") or {}).get("latency_cycles")
            bb = bench_jsonl.get(nb)
            sb = bb / sel if bb and sel else None
            sg = gt / sel if gt and sel else None
            sp = pb / sel if pb and sel else None
            if sb:
                sp_b.append(sb)
            if sg:
                sp_g.append(sg)
            if sp:
                sp_p.append(sp)
            flow_path = f.with_name(f.name.replace("_multistep_results.json", "_flow_manifest.json"))
            selected_from = ""
            if flow_path.is_file():
                selected_from = json.loads(flow_path.read_text()).get("selected_from") or ""
            per_variant[key][nb] = {
                "bench": display_bench(nb),
                "status": "ok",
                "selected_cycles": sel,
                "benchmark_cycles": bb,
                "gt_cycles": int(gt) if gt else None,
                "phase_b_cycles": int(pb) if pb else None,
                "speedup_vs_benchmark": sb,
                "speedup_vs_gt": sg,
                "speedup_vs_phase_b": sp,
                "selected_from": selected_from,
            }
        aggregate.append(
            {
                "variant": key,
                "label": label,
                "ok": ok,
                "fail": fail,
                "gmean_speedup_vs_benchmark": geomean(sp_b),
                "gmean_speedup_vs_gt": geomean(sp_g),
                "gmean_speedup_vs_phase_b": geomean(sp_p),
                "median_speedup_vs_benchmark": median(sp_b) if sp_b else None,
                "median_speedup_vs_gt": median(sp_g) if sp_g else None,
                "median_speedup_vs_phase_b": median(sp_p) if sp_p else None,
                "benches_faster_than_gt": sum(1 for x in sp_g if x > 1),
                "benches_with_gt_speedup": len(sp_g),
            }
        )
    return per_variant, aggregate


def h2h_compare(per_variant: dict[str, dict[str, dict]], key_a: str, key_b: str) -> dict:
    wins_a = wins_b = ties = skipped = 0
    for nb in sorted(set(per_variant[key_a]) | set(per_variant[key_b])):
        a = per_variant[key_a].get(nb, {})
        b = per_variant[key_b].get(nb, {})
        if a.get("status") != "ok" or b.get("status") != "ok":
            skipped += 1
            continue
        la, lb = a.get("selected_cycles"), b.get("selected_cycles")
        if not la or not lb:
            skipped += 1
            continue
        if lb < la * 0.999:
            wins_b += 1
        elif la < lb * 0.999:
            wins_a += 1
        else:
            ties += 1
    compared = wins_a + wins_b + ties
    return {
        "variant_a": key_a,
        "variant_b": key_b,
        "wins_a": wins_a,
        "wins_b": wins_b,
        "ties": ties,
        "skipped": skipped,
        "compared": compared,
        "a_win_pct": (100.0 * wins_a / compared) if compared else None,
        "b_win_pct": (100.0 * wins_b / compared) if compared else None,
    }


def write_per_bench_csv(path: Path, per_variant: dict[str, dict[str, dict]], bench_jsonl: dict[str, int]) -> None:
    fieldnames = [
        "bench",
        "benchmark_cycles",
        "gt_cycles",
    ]
    for key, label in VARIANTS:
        fieldnames.extend(
            [
                f"{key}_status",
                f"{key}_selected_cycles",
                f"{key}_speedup_vs_benchmark",
                f"{key}_speedup_vs_gt",
                f"{key}_speedup_vs_phase_b",
                f"{key}_selected_from",
            ]
        )
    rows = []
    for nb in sorted(bench_jsonl):
        row = {
            "bench": display_bench(nb),
            "benchmark_cycles": bench_jsonl[nb],
            "gt_cycles": bench_jsonl[nb],
        }
        for key, _ in VARIANTS:
            d = per_variant[key].get(nb, {})
            row[f"{key}_status"] = d.get("status", "missing")
            row[f"{key}_selected_cycles"] = d.get("selected_cycles", "")
            row[f"{key}_speedup_vs_benchmark"] = (
                f"{d['speedup_vs_benchmark']:.6f}" if d.get("speedup_vs_benchmark") is not None else ""
            )
            row[f"{key}_speedup_vs_gt"] = (
                f"{d['speedup_vs_gt']:.6f}" if d.get("speedup_vs_gt") is not None else ""
            )
            row[f"{key}_speedup_vs_phase_b"] = (
                f"{d['speedup_vs_phase_b']:.6f}" if d.get("speedup_vs_phase_b") is not None else ""
            )
            row[f"{key}_selected_from"] = d.get("selected_from", d.get("error", ""))
        rows.append(row)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def write_aggregate_csv(path: Path, aggregate: list[dict]) -> None:
    fieldnames = list(aggregate[0].keys()) if aggregate else []
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in aggregate:
            w.writerow(row)


def write_h2h_csv(path: Path, h2h_rows: list[dict]) -> None:
    fieldnames = [
        "comparison",
        "variant_a",
        "variant_b",
        "wins_a",
        "wins_b",
        "ties",
        "skipped",
        "compared",
        "a_win_pct",
        "b_win_pct",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(h2h_rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stamp", default=DEFAULT_STAMP)
    parser.add_argument("--baseline-jsonl", type=Path, default=DEFAULT_BASELINE_JSONL)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Default: artifacts/pc2/analysis/<stamp>/",
    )
    args = parser.parse_args()

    out_dir = args.out_dir or (REPO / "artifacts/pc2/analysis" / args.stamp)
    out_dir.mkdir(parents=True, exist_ok=True)

    bench_jsonl = load_bench_jsonl(args.baseline_jsonl)
    per_variant, aggregate = collect(args.stamp, bench_jsonl)

    per_bench_path = out_dir / "csynth_speedup_per_bench.csv"
    aggregate_path = out_dir / "csynth_speedup_aggregate.csv"
    h2h_path = out_dir / "csynth_speedup_h2h.csv"

    write_per_bench_csv(per_bench_path, per_variant, bench_jsonl)
    write_aggregate_csv(aggregate_path, aggregate)

    h2h_rows = []
    for key_a, key_b, label in H2H_PAIRS:
        row = h2h_compare(per_variant, key_a, key_b)
        row["comparison"] = label
        h2h_rows.append(row)
    write_h2h_csv(h2h_path, h2h_rows)

    print(f"wrote {per_bench_path}")
    print(f"wrote {aggregate_path}")
    print(f"wrote {h2h_path}")
    print()
    print("Head-to-head (lower selected csynth cycles wins):")
    for row in h2h_rows:
        print(
            f"  {row['comparison']}: {row['variant_a']} {row['wins_a']} - {row['wins_b']} {row['variant_b']}"
            f" (ties {row['ties']}, skipped {row['skipped']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
