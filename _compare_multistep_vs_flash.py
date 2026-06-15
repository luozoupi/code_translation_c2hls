"""Join phase-10 multistep+OLD-skills cells with phase-8 flash+OLD-skills
cells (and phase-8 flash+OFF as a baseline column). Print per-bench cosim
cycle comparison + aggregate stats.

Phase 10 cosim lives at:
  matrix.json[i].summary.steps[-1].cosim.measured.latency_cycles_avg
  (fallback: matrix.json[i].summary.baseline_cosim.measured.latency_cycles_avg)

Phase 8 cosim lives at:
  results_matrix_u280_fullcosim/<bench>/sonnet__flash__skills/<bench>_results.json
    .cosim.measured.latency_cycles_avg
  (analogous for sonnet__flash__noskills)
"""
from __future__ import annotations

import json
from math import exp, log
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PHASE10 = ROOT / "results_matrix_u280_multistep_old_skills" / "matrix.json"
PHASE8_DIR = ROOT / "results_matrix_u280_fullcosim"


def _final_cosim_multistep(summary: dict) -> int | None:
    steps = summary.get("steps") or []
    if steps and isinstance(steps[-1], dict):
        m = ((steps[-1].get("cosim") or {}).get("measured") or {})
        if m.get("latency_cycles_avg"):
            return m["latency_cycles_avg"]
    base = (summary.get("baseline_cosim") or {}).get("measured") or {}
    return base.get("latency_cycles_avg")


def _final_csynth_multistep(summary: dict) -> int | None:
    final = summary.get("final_report") or {}
    if final.get("latency_cycles"):
        return final["latency_cycles"]
    steps = summary.get("steps") or []
    if steps and isinstance(steps[-1], dict):
        sr = steps[-1].get("synth_report") or {}
        if sr.get("latency_cycles"):
            return sr["latency_cycles"]
    return None


def _flash_cell(bench: str, setup: str) -> dict | None:
    p = PHASE8_DIR / bench / f"sonnet__flash__{setup}" / f"{bench}_results.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _flash_cosim(j: dict) -> int | None:
    m = (j.get("cosim") or {}).get("measured") or {}
    return m.get("latency_cycles_avg")


def _flash_csynth(j: dict) -> int | None:
    return (j.get("synth_report") or {}).get("latency_cycles")


def main() -> int:
    multi = json.loads(PHASE10.read_text())
    print(f"Phase 10 entries: {len(multi)}")

    print()
    print("=" * 130)
    print("  Per-bench cosim comparison (cycles)")
    print(f"  {'bench':<28}{'flash_OFF':>14}{'flash_OLD':>14}{'multi_OLD':>14}   {'multi/flash_OLD':>16}{'multi/flash_OFF':>16}  flag")
    print("-" * 130)

    ratios_multi_vs_flash_old = []
    ratios_multi_vs_flash_off = []
    for e in sorted(multi, key=lambda x: x["bench"]):
        bench = e["bench"]
        if e.get("status") != "ok":
            continue
        s = e.get("summary") or {}
        m_cosim = _final_cosim_multistep(s)
        f_old = _flash_cell(bench, "skills")
        f_off = _flash_cell(bench, "noskills")
        old_cosim = _flash_cosim(f_old) if f_old else None
        off_cosim = _flash_cosim(f_off) if f_off else None

        if m_cosim and old_cosim:
            r1 = m_cosim / old_cosim
            ratios_multi_vs_flash_old.append(r1)
        else:
            r1 = None
        if m_cosim and off_cosim:
            r2 = m_cosim / off_cosim
            ratios_multi_vs_flash_off.append(r2)
        else:
            r2 = None

        flag = ""
        if r1 is not None:
            if r1 > 1.5: flag += "MULTI WORSE"
            elif r1 < 0.66: flag += "MULTI BETTER"

        off_s = f"{off_cosim:,}" if off_cosim else "—"
        old_s = f"{old_cosim:,}" if old_cosim else "—"
        m_s = f"{m_cosim:,}" if m_cosim else "—"
        r1_s = f"{r1:.2f}x" if r1 else "—"
        r2_s = f"{r2:.2f}x" if r2 else "—"
        print(f"  {bench:<28}{off_s:>14}{old_s:>14}{m_s:>14}   {r1_s:>16}{r2_s:>16}  {flag}")

    def geomean(xs):
        if not xs: return None
        return exp(sum(log(x) for x in xs) / len(xs))

    def amean(xs):
        if not xs: return None
        return sum(xs) / len(xs)

    g1 = geomean(ratios_multi_vs_flash_old)
    g2 = geomean(ratios_multi_vs_flash_off)
    a1 = amean(ratios_multi_vs_flash_old)
    a2 = amean(ratios_multi_vs_flash_off)
    n1 = len(ratios_multi_vs_flash_old)
    n2 = len(ratios_multi_vs_flash_off)

    print()
    print("=" * 80)
    print("  Aggregate (cosim cycle ratio, multistep / flash; >1 means multistep slower)")
    print("=" * 80)
    print(f"  multi vs flash_OLD:  n={n1}  geomean={g1:.3f}x  mean={a1:.3f}x   "
          f"wins(>0.95)={sum(1 for r in ratios_multi_vs_flash_old if r<0.95)}  "
          f"losses(>1.05)={sum(1 for r in ratios_multi_vs_flash_old if r>1.05)}")
    print(f"  multi vs flash_OFF:  n={n2}  geomean={g2:.3f}x  mean={a2:.3f}x   "
          f"wins(<0.95)={sum(1 for r in ratios_multi_vs_flash_off if r<0.95)}  "
          f"losses(>1.05)={sum(1 for r in ratios_multi_vs_flash_off if r>1.05)}")


if __name__ == "__main__":
    raise SystemExit(main())
