#!/usr/bin/env python3
"""Merge 10800s HLSFactory direct-cosim supplement into prior direct baseline."""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from export_schema_jsonl import validate_jsonl


BASE_SUMMARY = Path("artifacts/hlsfactory_direct_reference_merged_cosim7200_20260531.summary.json")
BASE_JSONL = Path("artifacts/hlsfactory_direct_reference_merged_cosim7200_20260531.jsonl")
SUPP_SUMMARY = Path("artifacts/hlsfactory_direct_reference_hlsfactory_direct_cosim10800_remaining_20260601.summary.json")
SUPP_JSONL = Path("artifacts/hlsfactory_direct_reference_hlsfactory_direct_cosim10800_remaining_20260601.jsonl")
AGENT_SUMMARY = Path("artifacts/agentic_no_streamcluster_hlsfactory_flash_sonnet46_cosim_skill_onoff_20260528_cosim1800.summary.json")

OUT_SUMMARY = Path("artifacts/hlsfactory_direct_reference_merged_cosim10800_20260601.summary.json")
OUT_JSONL = Path("artifacts/hlsfactory_direct_reference_merged_cosim10800_20260601.jsonl")
OUT_COMPARISON_CSV = Path("artifacts/hlsfactory_flash_agentic_vs_direct_merged_cosim10800_20260601.csv")
OUT_COMPARISON_SUMMARY = Path("artifacts/hlsfactory_flash_agentic_vs_direct_merged_cosim10800_20260601.summary.json")
OUT_COSIM_CSV = Path("artifacts/hlsfactory_flash_agentic_vs_direct_cosim_merged_10800_20260601.csv")
OUT_COSIM_SUMMARY = Path("artifacts/hlsfactory_flash_agentic_vs_direct_cosim_merged_10800_20260601.summary.json")


def _bench_from_record(rec: dict) -> str:
    problem = rec.get("problem") or {}
    group_path = problem.get("group_path")
    if isinstance(group_path, list) and group_path:
        return str(group_path[-1])
    if isinstance(group_path, str) and group_path:
        return group_path.split("/")[-1]
    return str(problem.get("name") or problem.get("benchmark") or "")


def _gmean(values: list[float]) -> float | None:
    xs = [x for x in values if x and x > 0]
    if not xs:
        return None
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    base = json.loads(BASE_SUMMARY.read_text())
    supp = json.loads(SUPP_SUMMARY.read_text())
    agent = json.loads(AGENT_SUMMARY.read_text())

    base_rows = {row["bench"]: dict(row) for row in base.get("rows", [])}
    updates: list[dict] = []
    for row in supp.get("rows", []):
        bench = row["bench"]
        base_row = base_rows.get(bench)
        if not base_row:
            base_rows[bench] = row
            updates.append({"bench": bench, "action": "added_supplement_row", "cosim_status": row.get("cosim_status")})
            continue
        if base_row.get("cosim_status") != "pass" and row.get("cosim_status") == "pass":
            for key in ("cosim_status", "cosim_runtime_seconds", "cosim_cycles", "cosim_error", "cosim_work_dir"):
                base_row[key] = row.get(key)
            base_row["cosim_supplement"] = "hlsfactory_direct_cosim10800_remaining_20260601"
            updates.append({"bench": bench, "action": "timeout_to_pass", "cosim_cycles": row.get("cosim_cycles")})
        elif base_row.get("cosim_status") == "timeout" and row.get("cosim_status") == "timeout":
            base_row["cosim_supplement"] = "timeout_at_10800s"
            base_row["cosim_10800_runtime_seconds"] = row.get("cosim_runtime_seconds")

    merged_rows = [base_rows[name] for name in sorted(base_rows)]

    supplement_rtl = {}
    for rec in _read_jsonl(SUPP_JSONL):
        bench = _bench_from_record(rec)
        if rec.get("report_type") == "rtl_sim" and (rec.get("rtl_sim") or {}).get("status") == "pass":
            supplement_rtl[bench] = rec

    updated_benches = {item["bench"] for item in updates if item["action"] == "timeout_to_pass"}
    merged_records = []
    for rec in _read_jsonl(BASE_JSONL):
        bench = _bench_from_record(rec)
        if rec.get("report_type") == "rtl_sim" and bench in updated_benches and bench in supplement_rtl:
            merged_records.append(supplement_rtl.pop(bench))
        else:
            merged_records.append(rec)
    merged_records.extend(supplement_rtl[bench] for bench in sorted(supplement_rtl) if bench in updated_benches)
    OUT_JSONL.write_text("\n".join(json.dumps(rec) for rec in merged_records) + ("\n" if merged_records else ""))

    merged_summary = dict(base)
    merged_summary["rows"] = merged_rows
    merged_summary["jsonl"] = str(OUT_JSONL)
    merged_summary["jsonl_records"] = len(merged_records)
    merged_summary["merge_10800"] = {
        "base_summary": str(BASE_SUMMARY),
        "supplement_summary": str(SUPP_SUMMARY),
        "policy": "prefer 10800s cosim pass over previous timeout; otherwise annotate timeout_at_10800s",
        "updates": updates,
    }
    OUT_SUMMARY.write_text(json.dumps(merged_summary, indent=2) + "\n")

    direct_by = {row["bench"]: row for row in merged_rows}
    comparison_rows = []
    for row in agent.get("rows", []):
        cur = row.get("current") or {}
        best = cur.get("best") or {}
        bench = row["bench"]
        direct = direct_by.get(bench, {})
        direct_report = direct.get("report") or {}
        direct_cycles = direct_report.get("latency_cycles")
        best_cycles = best.get("cycles")
        speedup = (
            direct_cycles / best_cycles
            if isinstance(direct_cycles, (int, float)) and isinstance(best_cycles, (int, float)) and best_cycles
            else None
        )
        best_step = best.get("step")
        step_record = next((s for s in cur.get("step_cycles") or [] if s.get("step") == best_step), {})
        comparison_rows.append({
            "bench": bench,
            "model": row.get("model"),
            "model_id": row.get("model_id"),
            "skill_mode": row.get("skill_mode"),
            "agent_success": bool(cur.get("success")),
            "best_step": best_step or "",
            "best_cycles": best_cycles if best_cycles is not None else "",
            "baseline_cycles_agent": cur.get("baseline_cycles") if cur.get("baseline_cycles") is not None else "",
            "direct_synth_status": direct.get("synth_status", ""),
            "direct_csim_status": direct.get("csim_status", ""),
            "direct_cosim_status": direct.get("cosim_status", ""),
            "direct_cycles": direct_cycles if direct_cycles is not None else "",
            "direct_cosim_cycles": direct.get("cosim_cycles") if direct.get("cosim_cycles") is not None else "",
            "direct_cosim_supplement": direct.get("cosim_supplement", ""),
            "speedup_vs_direct_csynth": round(speedup, 6) if speedup is not None else "",
            "agent_step_cosim": step_record.get("cosim", ""),
            "agent_step_cosim_cycles": step_record.get("cosim_cycles") if step_record.get("cosim_cycles") is not None else "",
            "steps_attempted": cur.get("steps_attempted", ""),
            "steps_success": cur.get("steps_success", ""),
            "elapsed_sec": cur.get("elapsed_sec", ""),
            "error": cur.get("error") or "",
        })
    _write_csv(OUT_COMPARISON_CSV, comparison_rows)

    comparison_summary = {}
    for mode in sorted({row["skill_mode"] for row in comparison_rows}):
        mode_rows = [row for row in comparison_rows if row["skill_mode"] == mode]
        speeds = [float(row["speedup_vs_direct_csynth"]) for row in mode_rows if row["speedup_vs_direct_csynth"] != ""]
        comparison_summary[mode] = {
            "rows": len(mode_rows),
            "success": sum(row["agent_success"] for row in mode_rows),
            "fail": sum(not row["agent_success"] for row in mode_rows),
            "flash_selected": sum(row["best_step"] == "flash" for row in mode_rows),
            "baseline_selected": sum(row["best_step"] == "baseline" for row in mode_rows),
            "agent_cosim_pass": sum(str(row["agent_step_cosim"]).lower() == "true" for row in mode_rows),
            "agent_cosim_fail": sum(str(row["agent_step_cosim"]).lower() == "false" for row in mode_rows),
            "geomean_speedup_vs_direct_csynth": _gmean(speeds),
            "median_speedup_vs_direct_csynth": statistics.median(speeds) if speeds else None,
            "mean_speedup_vs_direct_csynth": statistics.mean(speeds) if speeds else None,
            "min_speedup_vs_direct_csynth": min(speeds) if speeds else None,
            "max_speedup_vs_direct_csynth": max(speeds) if speeds else None,
        }
    comparison_summary["direct_merged"] = {
        "rows": len(merged_rows),
        "synth": dict(Counter(row.get("synth_status") for row in merged_rows)),
        "csim": dict(Counter(row.get("csim_status") for row in merged_rows)),
        "cosim": dict(Counter(row.get("cosim_status") for row in merged_rows)),
        "supplement_updates": updates,
    }
    OUT_COMPARISON_SUMMARY.write_text(json.dumps(comparison_summary, indent=2) + "\n")

    cosim_rows = []
    for row in comparison_rows:
        try:
            agent_cycles = float(row["agent_step_cosim_cycles"])
            direct_cosim_cycles = float(row["direct_cosim_cycles"])
        except (TypeError, ValueError):
            continue
        if agent_cycles <= 0 or direct_cosim_cycles <= 0:
            continue
        cosim_rows.append({
            "bench": row["bench"],
            "skill_mode": row["skill_mode"],
            "best_step": row["best_step"],
            "agent_cosim_cycles": int(agent_cycles),
            "direct_cosim_cycles": int(direct_cosim_cycles),
            "cosim_speedup": round(direct_cosim_cycles / agent_cycles, 6),
            "direct_cosim_supplement": row.get("direct_cosim_supplement", ""),
            "agent_csynth_cycles": row["best_cycles"],
            "direct_csynth_cycles": row["direct_cycles"],
        })
    _write_csv(OUT_COSIM_CSV, cosim_rows)

    cosim_summary = {}
    for mode in sorted({row["skill_mode"] for row in cosim_rows}):
        mode_rows = [row for row in cosim_rows if row["skill_mode"] == mode]
        speeds = [float(row["cosim_speedup"]) for row in mode_rows]
        cosim_summary[mode] = {
            "available_pairs": len(mode_rows),
            "geomean_cosim_speedup": _gmean(speeds),
            "median_cosim_speedup": statistics.median(speeds) if speeds else None,
            "mean_cosim_speedup": statistics.mean(speeds) if speeds else None,
            "min_cosim_speedup": min(speeds) if speeds else None,
            "max_cosim_speedup": max(speeds) if speeds else None,
            "supplement_pairs": sum(bool(row.get("direct_cosim_supplement")) for row in mode_rows),
            "top": sorted(mode_rows, key=lambda item: item["cosim_speedup"], reverse=True)[:8],
            "bottom": sorted(mode_rows, key=lambda item: item["cosim_speedup"])[:8],
        }
    OUT_COSIM_SUMMARY.write_text(json.dumps(cosim_summary, indent=2) + "\n")

    validation = validate_jsonl(OUT_JSONL)
    print(json.dumps({
        "merged_summary": str(OUT_SUMMARY),
        "merged_jsonl": str(OUT_JSONL),
        "comparison_csv": str(OUT_COMPARISON_CSV),
        "comparison_summary": str(OUT_COMPARISON_SUMMARY),
        "cosim_csv": str(OUT_COSIM_CSV),
        "cosim_summary": str(OUT_COSIM_SUMMARY),
        "jsonl_validation": validation,
        "direct_merged": comparison_summary["direct_merged"],
        "cosim_modes": {
            key: {
                "available_pairs": val["available_pairs"],
                "geomean_cosim_speedup": val["geomean_cosim_speedup"],
                "median_cosim_speedup": val["median_cosim_speedup"],
                "supplement_pairs": val["supplement_pairs"],
            }
            for key, val in cosim_summary.items()
        },
    }, indent=2))
    return 0 if validation.get("invalid") == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
