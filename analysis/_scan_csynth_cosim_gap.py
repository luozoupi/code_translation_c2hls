"""For each cell with PASSING cosim, compare csynth latency_cycles to cosim
latency_cycles_avg. Flag cells whose ratio falls outside [0.85, 1.15].

Direction note: cosim is the truth (cycle-accurate RTL sim). csynth is the
HLS scheduler's static estimate. A gap typically arises because:

  - Loop bounds the scheduler couldn't fold (runtime-bound loops)
  - II misses revealed only at cosim time (memory port conflicts in real RTL)
  - Dynamic memory contention (m_axi adapter latency variance)
  - Conditional dataflow not captured in static analysis

Direction of suspicion:
  - cosim > csynth by a lot: scheduler was optimistic. Likely legit.
  - csynth > cosim by a lot: scheduler was pessimistic, or — more concerning —
    cosim is shortcutting (gold testbench reset midway, sim ends early, etc.)
    which can indicate the kernel is skipping work / hacking the bench.

For each flagged cell:
  - Print bench, setup, csynth, cosim, ratio
  - Print latency vs gold ratio (so we can see if cosim cycles are also far below gold — bench-hacking signal)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent  # repo root (script lives in analysis/)

THRESHOLD = 0.15  # 15%

SWEEPS = [
    ("PHASE-9 (EXT)", ROOT / "results_matrix_u280_fullcosim_extended"),
    ("PHASE-8 (OLD+OFF)", ROOT / "results_matrix_u280_fullcosim"),
]


def _gold_latency_cycles(j: dict) -> Optional[float]:
    """Pull gold cosim cycles if available; fall back to gold csynth."""
    gt = j.get("ground_truth_report") or {}
    # Prefer gold cosim if present
    gt_cosim = gt.get("cosim") or {}
    gt_cosim_meas = gt_cosim.get("measured") or {}
    gtc = gt_cosim_meas.get("latency_cycles_avg")
    if gtc:
        return float(gtc)
    # Else gold csynth
    gtl = gt.get("latency_cycles")
    if gtl:
        return float(gtl)
    # Try comparison.gold/ground_truth keys for older schema
    cmp = j.get("comparison") or {}
    for k in ("ground_truth", "gold"):
        gg = cmp.get(k) or {}
        if "latency_cycles" in gg:
            return float(gg["latency_cycles"])
    return None


def main() -> int:
    print(f"Csynth-vs-cosim gap scan; threshold = ±{int(THRESHOLD*100)}%")
    print(f"Direction: ratio = cosim / csynth. >1 = cosim slower than estimate, <1 = cosim faster than estimate (suspicious)\n")

    grand_flagged = []
    for label, sweep_dir in SWEEPS:
        if not sweep_dir.is_dir():
            continue
        print("=" * 110)
        print(f"  {label}: {sweep_dir.name}")
        print("=" * 110)
        rows = []
        for bench_dir in sorted(sweep_dir.iterdir()):
            if not bench_dir.is_dir():
                continue
            bench = bench_dir.name
            for cell_dir in sorted(bench_dir.iterdir()):
                if not cell_dir.is_dir():
                    continue
                rj = cell_dir / f"{bench}_results.json"
                if not rj.exists():
                    continue
                try:
                    j = json.loads(rj.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                cosim = (j.get("cosim") or {})
                measured = cosim.get("measured") or {}
                cosim_cyc = measured.get("latency_cycles_avg")
                synth = j.get("synth_report") or {}
                csynth_cyc = synth.get("latency_cycles")
                gold_cyc = _gold_latency_cycles(j)
                if cosim_cyc is None or csynth_cyc is None or csynth_cyc == 0:
                    continue
                ratio = cosim_cyc / csynth_cyc
                gold_ratio = (cosim_cyc / gold_cyc) if gold_cyc else None
                rows.append((bench, cell_dir.name, int(csynth_cyc), int(cosim_cyc),
                             ratio, gold_ratio, gold_cyc))
        # Print all rows for reference, flag big gaps
        print(f"\n  {'bench':<28}{'cell':<28}{'csynth':>10}{'cosim':>12}{'cosim/csynth':>15}{'cosim/gold':>15}  flag")
        print("  " + "-" * 110)
        for bench, cell, csynth, cosim, ratio, gold_ratio, gold_cyc in sorted(rows, key=lambda r: -abs(1 - r[4])):
            flag = ""
            if abs(1 - ratio) >= THRESHOLD:
                flag = "*** GAP > 15%"
                if ratio < 1 - THRESHOLD:
                    flag += " (cosim FASTER than csynth)"
                else:
                    flag += " (cosim slower than csynth)"
                # bench-hacking secondary signal: cosim << gold
                if gold_ratio is not None and gold_ratio < 0.2:
                    flag += "  +++ cosim << gold (BENCH-HACKING SUSPECT)"
                grand_flagged.append((label, bench, cell, csynth, cosim, ratio, gold_ratio))
            gold_str = f"{gold_ratio:.3f}" if gold_ratio is not None else "(no gold)"
            print(f"  {bench:<28}{cell:<28}{csynth:>10}{cosim:>12}{ratio:>15.3f}{gold_str:>15}  {flag}")

    print()
    print("=" * 110)
    print(f"GRAND SUMMARY — flagged cells (|cosim/csynth - 1| >= {int(THRESHOLD*100)}%)")
    print("=" * 110)
    if not grand_flagged:
        print("  (none — all cells within ±15%)")
    else:
        for label, bench, cell, csynth, cosim, ratio, gold_ratio in grand_flagged:
            gold_str = f"{gold_ratio:.3f}" if gold_ratio is not None else "(no gold)"
            print(f"  {label}  {bench:<28} {cell:<26} csynth={csynth} cosim={cosim} ratio={ratio:.3f} cosim/gold={gold_str}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
