"""All-setup data table — comparison across:
  - Direct Vitis (gold)         : ground_truth_report from phase 8 cells
  - Flash + skills OFF          : phase 8 noskills cells
  - Flash + skills OLD (base)   : phase 8 skills cells
  - Flash + skills EXT (ext)    : phase 9 cells
  - Multistep + skills OLD      : phase 10 matrix.json

For each bench: csynth + cosim cycle counts + cosim/gold ratio + skill traces.

Outputs both a human-readable table and a CSV.
"""
from __future__ import annotations

import csv
import json
from math import exp, log
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # repo root (script lives in analysis/)

PHASE8 = ROOT / "results_matrix_u280_fullcosim"
PHASE9 = ROOT / "results_matrix_u280_fullcosim_extended"
PHASE10_MATRIX = ROOT / "results_matrix_u280_multistep_old_skills" / "matrix.json"


def _read_json(p: Path) -> dict | None:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _flash_cell(bench: str, setup: str, root: Path) -> dict | None:
    return _read_json(root / bench / f"sonnet__flash__{setup}" / f"{bench}_results.json")


def _cosim_cycles(j: dict | None) -> int | None:
    if not j: return None
    m = (j.get("cosim") or {}).get("measured") or {}
    return m.get("latency_cycles_avg")


def _csynth_cycles(j: dict | None) -> int | None:
    if not j: return None
    return (j.get("synth_report") or {}).get("latency_cycles")


def _gold_cosim(j: dict | None) -> int | None:
    """Direct Vitis (gold) cosim cycles — from reference_validation.cosim
    (ground_truth_report itself only has csynth fields; cosim is under
    reference_validation)."""
    if not j: return None
    rv = j.get("reference_validation") or {}
    m = (rv.get("cosim") or {}).get("measured") or {}
    return m.get("latency_cycles_avg")


def _gold_csynth(j: dict | None) -> int | None:
    if not j: return None
    return (j.get("ground_truth_report") or {}).get("latency_cycles")


def _flash_skills_log(j: dict | None) -> dict | None:
    if not j: return None
    sl = j.get("skills_log")
    return sl if sl and sl.get("enabled") else None


def _multi_cell(bench: str, matrix: list[dict]) -> dict | None:
    for e in matrix:
        if e.get("bench") == bench and e.get("status") == "ok":
            return e
    return None


def _multi_final_cosim(e: dict | None) -> int | None:
    if not e: return None
    steps = (e.get("summary") or {}).get("steps") or []
    if steps and isinstance(steps[-1], dict):
        m = ((steps[-1].get("cosim") or {}).get("measured") or {})
        if m.get("latency_cycles_avg"):
            return m["latency_cycles_avg"]
    base = ((e.get("summary") or {}).get("baseline_cosim") or {}).get("measured") or {}
    return base.get("latency_cycles_avg")


def _multi_final_csynth(e: dict | None) -> int | None:
    if not e: return None
    final = (e.get("summary") or {}).get("final_report") or {}
    return final.get("latency_cycles")


def _multi_skills_log(e: dict | None) -> dict | None:
    if not e: return None
    sl = (e.get("summary") or {}).get("skills_log")
    return sl if sl and sl.get("enabled") else None


def _backfilled_skills(bench: str, setup: str, root: Path) -> dict | None:
    p = root / bench / f"sonnet__flash__{setup}" / f"{bench}_skills_log.backfilled.json"
    return _read_json(p)


def _short_ids(ids: list[str], max_n: int = 4) -> str:
    if not ids: return "—"
    short = [i.replace("prompt-", "p-").replace("hls-", "")[:14] for i in ids[:max_n]]
    extra = f" (+{len(ids)-max_n})" if len(ids) > max_n else ""
    return ",".join(short) + extra


def _ratio_or_dash(num, denom) -> str:
    if num is None or denom is None or denom == 0:
        return "—"
    return f"{num/denom:.2f}x"


def main():
    multi_matrix = json.loads(PHASE10_MATRIX.read_text())
    benches = sorted({d.name for d in PHASE8.iterdir() if d.is_dir() and d.name.startswith("hlsfactory_")})
    print(f"Building all-setup summary across {len(benches)} benches\n")

    rows = []
    for bench in benches:
        f_off = _flash_cell(bench, "noskills", PHASE8)
        f_old = _flash_cell(bench, "skills", PHASE8)
        f_ext = _flash_cell(bench, "skills", PHASE9)
        m_cell = _multi_cell(bench, multi_matrix)

        # Gold (direct Vitis) — same regardless of setup, pull from OFF cell
        gold_cs = _gold_csynth(f_off)
        gold_co = _gold_cosim(f_off)
        # Flash setups
        off_cs = _csynth_cycles(f_off); off_co = _cosim_cycles(f_off)
        old_cs = _csynth_cycles(f_old); old_co = _cosim_cycles(f_old)
        ext_cs = _csynth_cycles(f_ext); ext_co = _cosim_cycles(f_ext)
        m_cs = _multi_final_csynth(m_cell); m_co = _multi_final_cosim(m_cell)

        # Skill traces
        # OFF backfilled — should be enabled=False (no skills applied)
        # OLD: backfilled if pre-feature; native if post
        old_sl = _flash_skills_log(f_old) or _backfilled_skills(bench, "skills", PHASE8)
        ext_sl = _flash_skills_log(f_ext) or _backfilled_skills(bench, "skills", PHASE9)
        multi_sl = _multi_skills_log(m_cell)

        rows.append({
            "bench": bench,
            "gold_cs": gold_cs, "gold_co": gold_co,
            "off_cs": off_cs, "off_co": off_co,
            "old_cs": old_cs, "old_co": old_co,
            "ext_cs": ext_cs, "ext_co": ext_co,
            "multi_cs": m_cs, "multi_co": m_co,
            "old_skills": (old_sl or {}).get("unique_skill_ids") or [],
            "ext_skills": (ext_sl or {}).get("unique_skill_ids") or [],
            "multi_skills": (multi_sl or {}).get("unique_skill_ids") or [],
        })

    # ============ TABLE 1: cosim cycles per setup ============
    print("=" * 145)
    print("  TABLE 1 — cosim cycles per setup (raw)")
    print("=" * 145)
    print(f"  {'bench':<28}{'GOLD':>13}{'OFF':>13}{'OLD':>13}{'EXT':>13}{'MULTI':>13}")
    print("  " + "-" * 90)
    def cfmt(n): return f"{n:,}" if n is not None else "—"
    for r in rows:
        print(f"  {r['bench']:<28}{cfmt(r['gold_co']):>13}{cfmt(r['off_co']):>13}"
              f"{cfmt(r['old_co']):>13}{cfmt(r['ext_co']):>13}{cfmt(r['multi_co']):>13}")

    # ============ TABLE 2: ratios vs GOLD (Direct Vitis) ============
    print()
    print("=" * 110)
    print("  TABLE 2 — cosim ratio vs GOLD (Direct Vitis cosim). <1.0 = agent faster than direct Vitis. n/a if missing")
    print("=" * 110)
    print(f"  {'bench':<28}{'OFF/gold':>12}{'OLD/gold':>12}{'EXT/gold':>12}{'MULTI/gold':>13}   {'best':<10}")
    print("  " + "-" * 80)
    for r in rows:
        g = r["gold_co"]
        ratios = {
            "OFF": r["off_co"]/g if r["off_co"] and g else None,
            "OLD": r["old_co"]/g if r["old_co"] and g else None,
            "EXT": r["ext_co"]/g if r["ext_co"] and g else None,
            "MULTI": r["multi_co"]/g if r["multi_co"] and g else None,
        }
        non_null = {k: v for k, v in ratios.items() if v is not None}
        best = min(non_null, key=non_null.get) if non_null else "—"
        def rs(v): return f"{v:.2f}x" if v else "—"
        print(f"  {r['bench']:<28}{rs(ratios['OFF']):>12}{rs(ratios['OLD']):>12}"
              f"{rs(ratios['EXT']):>12}{rs(ratios['MULTI']):>13}   {best:<10}")

    # Aggregate
    def geomean(xs):
        xs = [x for x in xs if x and x > 0]
        if not xs: return None
        return exp(sum(log(x) for x in xs) / len(xs))

    print()
    print("=" * 80)
    print("  Aggregate cosim/gold ratio (geomean; lower = faster than direct Vitis)")
    print("=" * 80)
    for setup, key in (("Flash OFF", "off_co"), ("Flash OLD", "old_co"),
                      ("Flash EXT", "ext_co"), ("Multistep OLD", "multi_co")):
        rs = [r[key]/r["gold_co"] for r in rows if r[key] and r["gold_co"]]
        g = geomean(rs)
        n = len(rs)
        wins = sum(1 for r in rs if r < 0.95) if rs else 0
        losses = sum(1 for r in rs if r > 1.05) if rs else 0
        g_str = f"{g:.3f}x" if g is not None else "—"
        print(f"  {setup:<20}  n={n:>3}  geomean={g_str:>7}   wins(faster)={wins:>3}  losses(slower)={losses:>3}")

    # ============ TABLE 3: skill traces per setup ============
    print()
    print("=" * 140)
    print("  TABLE 3 — Skill traces (which skill_ids were retrieved per setup; OFF has none by construction)")
    print("=" * 140)
    print(f"  {'bench':<25}{'OLD skills (base)':<46}{'EXT skills (+ext)':<46}{'MULTI skills (base)':<46}")
    print("  " + "-" * 130)
    for r in rows:
        print(f"  {r['bench']:<25}{_short_ids(r['old_skills']):<46}{_short_ids(r['ext_skills']):<46}{_short_ids(r['multi_skills']):<46}")

    # ============ CSV export ============
    csv_out = ROOT / "all_setups_summary.csv"
    with csv_out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bench",
                    "gold_csynth", "gold_cosim",
                    "off_csynth", "off_cosim", "off_cosim_over_gold",
                    "old_csynth", "old_cosim", "old_cosim_over_gold",
                    "ext_csynth", "ext_cosim", "ext_cosim_over_gold",
                    "multi_csynth", "multi_cosim", "multi_cosim_over_gold",
                    "old_skill_ids", "ext_skill_ids", "multi_skill_ids"])
        for r in rows:
            def rg(c): return c/r["gold_co"] if c and r["gold_co"] else None
            w.writerow([r["bench"], r["gold_cs"], r["gold_co"],
                        r["off_cs"], r["off_co"], rg(r["off_co"]),
                        r["old_cs"], r["old_co"], rg(r["old_co"]),
                        r["ext_cs"], r["ext_co"], rg(r["ext_co"]),
                        r["multi_cs"], r["multi_co"], rg(r["multi_co"]),
                        "|".join(r["old_skills"]),
                        "|".join(r["ext_skills"]),
                        "|".join(r["multi_skills"])])
    print(f"\nCSV written: {csv_out}")


if __name__ == "__main__":
    main()
