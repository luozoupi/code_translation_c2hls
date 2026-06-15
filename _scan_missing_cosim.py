"""Scan the most-recent sweep result dirs for cells whose agentic (generated)
cosim cycles are MISSING and classify the root cause.

NOTE: A cell whose Phase B cosim timed out but whose quality_repair turn then
recovered with a working (less-aggressive) design will NOT show up here — the
final results.json reflects the successful repair attempt. To find those cases,
run _scan_cosim_walltimes.py and look for invocations near 14400s; cells with
the literal timeout invocation that nevertheless ended with cosim cycles
present are silent timeout-pivoted cells.

Cosim cycles are at: results_json.cosim.measured.latency_cycles_avg

Failure taxonomy:
  - 'no_cosim_object'  : cosim key absent
  - 'unsupported'      : cosim.supported is False (TB lacks reference/dump)
  - 'csim_fail'        : csim never passed → cosim never attempted
  - 'cosim_timeout'    : cosim ran but hit HLS_COSIM_TIMEOUT
  - 'cosim_rtl_fail'   : cosim ran but RTL elab/sim failed
  - 'cosim_status_bad' : cosim.status != 'passed' for other reason
  - 'no_cycles'        : cosim.passed=True but measured.latency_cycles_avg null
  - 'ok'               : cycles present

For each missing cell, also pulls the last few lines of the cosim error and
the matrix_run.log tail (if present) to aid debugging.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def _classify(j: dict) -> tuple[str, str]:
    """Return (category, short_evidence)."""
    cosim = j.get("cosim") or {}
    csim = j.get("csim") or {}
    if not cosim:
        return "no_cosim_object", "no cosim key in results.json"

    csim_passed = csim.get("passed") if isinstance(csim, dict) else None
    if csim_passed is False:
        err = (csim.get("error") or "")[:140]
        return "csim_fail", f"csim failed: {err}"

    if cosim.get("supported") is False:
        return "unsupported", "cosim.supported=False (TB lacks reference/dump)"

    measured = cosim.get("measured") or {}
    cycles = measured.get("latency_cycles_avg") if isinstance(measured, dict) else None
    status = cosim.get("status") or ""
    error = (cosim.get("error") or "").strip()
    error_short = error.replace("\n", " ")[:200]

    if cosim.get("passed") and cycles is not None:
        return "ok", f"cycles={cycles}"

    if "timeout" in error.lower() or "timed out" in error.lower():
        return "cosim_timeout", error_short
    if "elaborat" in error.lower() or "compile" in error.lower() or "xsim" in error.lower():
        return "cosim_rtl_fail", error_short
    if cycles is None and cosim.get("passed"):
        return "no_cycles", "passed but no latency_cycles_avg in measured"
    if status and status != "passed":
        return "cosim_status_bad", f"status={status}; err={error_short}"
    if error:
        return "cosim_other_fail", error_short
    return "unknown", f"status={status}, ran={cosim.get('ran')}, supported={cosim.get('supported')}"


def _tail_log(cell_dir: Path, n_lines: int = 8) -> str:
    log = cell_dir / "matrix_run.log"
    if not log.exists():
        return "(no matrix_run.log)"
    try:
        lines = log.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return "(unreadable)"
    return "\n      ".join(lines[-n_lines:]) if lines else "(empty)"


def main() -> int:
    sweep_dirs = [
        ("PHASE-9 (EXT)", ROOT / "results_matrix_u280_fullcosim_extended"),
        ("PHASE-8 (OLD+OFF)", ROOT / "results_matrix_u280_fullcosim"),
    ]
    grand_summary: dict[str, list[str]] = {}

    for label, sweep_dir in sweep_dirs:
        if not sweep_dir.is_dir():
            print(f"[skip] {label}: dir not found at {sweep_dir}")
            continue
        print(f"\n{'='*100}")
        print(f"  {label}: {sweep_dir.name}")
        print(f"{'='*100}")
        rows = []
        by_cat: dict[str, list[str]] = {}
        for bench_dir in sorted(sweep_dir.iterdir()):
            if not bench_dir.is_dir():
                continue
            bench = bench_dir.name
            for cell_dir in sorted(bench_dir.iterdir()):
                if not cell_dir.is_dir():
                    continue
                rj = cell_dir / f"{bench}_results.json"
                if not rj.exists():
                    rows.append((bench, cell_dir.name, "no_results_json", "results.json missing"))
                    continue
                try:
                    j = json.loads(rj.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as e:
                    rows.append((bench, cell_dir.name, "parse_error", str(e)[:160]))
                    continue
                cat, evid = _classify(j)
                rows.append((bench, cell_dir.name, cat, evid))
                by_cat.setdefault(cat, []).append(f"{bench}/{cell_dir.name}")

        print(f"\nCategory summary ({sum(len(v) for v in by_cat.values())} cells):")
        for cat in sorted(by_cat):
            print(f"  {cat:<22} {len(by_cat[cat]):>3} cells")

        print(f"\nNon-OK cells with detail:")
        print(f"{'bench':<28}{'cell':<28}{'category':<22}evidence")
        print("-" * 130)
        for bench, cell, cat, evid in rows:
            if cat == "ok":
                continue
            cell_dir = sweep_dir / bench / cell
            tail = _tail_log(cell_dir, 4)
            print(f"{bench:<28}{cell:<28}{cat:<22}{evid[:55]}")
            print(f"{'':>56}log_tail:")
            for ln in tail.split("\n"):
                print(f"{'':>58}{ln[:110]}")
            print()

        grand_summary[label] = [f"{b}/{c} [{cat}]" for b, c, cat, _ in rows if cat != "ok"]

    print("\n" + "=" * 100)
    print("GRAND SUMMARY — cells with NO cosim cycles")
    print("=" * 100)
    for label, items in grand_summary.items():
        print(f"\n{label}: {len(items)} cells")
        for it in items:
            print(f"  {it}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
