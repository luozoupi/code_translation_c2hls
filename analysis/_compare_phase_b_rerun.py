"""Compare Phase B rerun cosim cycles (from cosim_phase_b_rerun_*/cosim_result.json)
against the final-saved cosim cycles in *_results.json.

Tells you whether re-running with extended timeout produced numbers that
should replace the current matrix headline, or whether the original
quality_repair pivot was the better design.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

CELLS = [
    ("symm/OFF",        "results_matrix_u280_fullcosim/hlsfactory_symm/sonnet__flash__noskills",         "hlsfactory_symm"),
    ("symm/EXT",        "results_matrix_u280_fullcosim_extended/hlsfactory_symm/sonnet__flash__skills",  "hlsfactory_symm"),
    ("floyd-wars./OFF", "results_matrix_u280_fullcosim/hlsfactory_floyd-warshall/sonnet__flash__noskills", "hlsfactory_floyd-warshall"),
]

print(f"{'cell':<20}{'phase_b_status':<18}{'wall_h':>8}{'phase_b_cosim':>16}{'final_cosim':>14}{'ratio_B/final':>16}  verdict")
print("-" * 120)
for label, cell_dir, bench in CELLS:
    cell = Path(cell_dir)
    # Final saved cosim
    rj = cell / f"{bench}_results.json"
    final_cosim = None
    if rj.exists():
        try:
            j = json.loads(rj.read_text())
            meas = (j.get("cosim") or {}).get("measured") or {}
            final_cosim = meas.get("latency_cycles_avg")
        except (OSError, json.JSONDecodeError):
            pass

    # Latest Phase B rerun
    rerun_dirs = sorted(cell.glob("cosim_phase_b_rerun_*"))
    if not rerun_dirs:
        print(f"{label:<20}{'NOT RUN':<18}{'':>8}{'-':>16}{(final_cosim or '-'):>14}{'':>16}  Run _rerun_phase_b_extended_timeout.sh on WSL first")
        continue
    latest = rerun_dirs[-1]
    cr = latest / "cosim_result.json"
    if not cr.exists():
        print(f"{label:<20}{'INCOMPLETE':<18}{'':>8}{'-':>16}{(final_cosim or '-'):>14}{'':>16}  rerun.log may have errored")
        continue
    cdata = json.loads(cr.read_text())
    passed = cdata.get("passed")
    err = (cdata.get("error") or "").strip()
    wall = cdata.get("wall_time_s") or 0
    meas = cdata.get("measured") or {}
    pb_cyc = meas.get("latency_cycles_avg")
    if not passed and "timeout" in err.lower():
        status = "STILL_TIMED_OUT"
    elif not passed:
        status = "FAILED"
    elif pb_cyc is None:
        status = "PASSED_NO_CYCLES"
    else:
        status = "PASSED"

    if pb_cyc and final_cosim:
        ratio = pb_cyc / final_cosim
        if ratio < 0.95:
            verdict = f"Phase B FASTER by {(1-ratio)*100:.1f}% — UPDATE matrix"
        elif ratio > 1.05:
            verdict = f"Phase B SLOWER by {(ratio-1)*100:.1f}% — repair was correct"
        else:
            verdict = "essentially identical — no change needed"
    else:
        verdict = "(no comparison possible)"

    pb_str = f"{pb_cyc:,}" if pb_cyc else "-"
    final_str = f"{final_cosim:,}" if final_cosim else "-"
    ratio_str = f"{(pb_cyc/final_cosim):.2f}x" if pb_cyc and final_cosim else "-"
    wall_str = f"{wall/3600:.2f}" if wall else "-"
    print(f"{label:<20}{status:<18}{wall_str:>8}{pb_str:>16}{final_str:>14}{ratio_str:>16}  {verdict}")
