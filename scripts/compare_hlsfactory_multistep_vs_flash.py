#!/usr/bin/env python3
"""Compare a HLSFactory multistep sweep summary against the flash baseline CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
DEFAULT_FLASH = REPO / "artifacts" / "hlsfactory_flash_agentic_vs_direct_merged_cosim10800_supplemented_skills_20260610.csv"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _num(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _ratio(new_value: Any, old_value: Any) -> float | None:
    new = _num(new_value)
    old = _num(old_value)
    if new is None or old in (None, 0):
        return None
    return old / new


def _cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _compact_skill_prompt(prompt: dict[str, Any] | None) -> str:
    if not isinstance(prompt, dict):
        return ""
    parts: list[str] = []
    reason = prompt.get("reason")
    if reason:
        parts.append(f"reason={reason}")
    injected = prompt.get("injected_skill_ids") or []
    matched = prompt.get("matched_skill_ids") or []
    if injected:
        parts.append("injected=" + "|".join(str(x) for x in injected))
    if matched:
        parts.append("matched=" + "|".join(str(x) for x in matched[:5]))
    return ";".join(parts)


def _summarize_multistep_row(row: dict[str, Any]) -> dict[str, Any]:
    cur = row.get("current") or {}
    best = cur.get("best") or {}
    steps = cur.get("step_cycles") or []
    final_step = steps[-1] if steps else {}
    skill_ids = [step.get("skill_id") for step in steps if step.get("skill_id")]
    skill_prompts = [
        _compact_skill_prompt(step.get("skill_prompt"))
        for step in steps
        if _compact_skill_prompt(step.get("skill_prompt"))
    ]
    return {
        "bench": row.get("bench"),
        "model": row.get("model"),
        "model_id": row.get("model_id"),
        "skill_mode": row.get("skill_mode"),
        "multistep_success": cur.get("success"),
        "multistep_steps_success": cur.get("steps_success"),
        "multistep_steps_attempted": cur.get("steps_attempted"),
        "multistep_best_step": best.get("step"),
        "multistep_best_cycles": best.get("cycles"),
        "multistep_baseline_cycles": cur.get("baseline_cycles"),
        "multistep_final_step": final_step.get("step"),
        "multistep_final_cosim": final_step.get("cosim"),
        "multistep_final_cosim_cycles": final_step.get("cosim_cycles"),
        "multistep_llm_calls": (cur.get("llm_usage") or {}).get("calls"),
        "multistep_total_tokens": (cur.get("llm_usage") or {}).get("total_tokens"),
        "multistep_skill_ids": "|".join(str(x) for x in skill_ids),
        "multistep_skill_prompt_trace": " || ".join(skill_prompts),
        "multistep_error": cur.get("error"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--multistep-summary", type=Path, required=True)
    parser.add_argument("--flash-csv", type=Path, default=DEFAULT_FLASH)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args()

    multistep = json.loads(args.multistep_summary.read_text())
    flash_rows = _read_csv(args.flash_csv)
    flash_by_key = {
        (row.get("bench"), row.get("skill_mode")): row
        for row in flash_rows
    }

    rows: list[dict[str, Any]] = []
    for raw in multistep.get("rows") or []:
        ms = _summarize_multistep_row(raw)
        flash = flash_by_key.get((ms["bench"], ms["skill_mode"])) or {}
        out = dict(ms)
        out.update({
            "flash_success": flash.get("agent_success"),
            "flash_best_step": flash.get("best_step"),
            "flash_best_cycles": flash.get("best_cycles"),
            "flash_agent_cosim": flash.get("agent_step_cosim"),
            "flash_agent_cosim_cycles": flash.get("agent_step_cosim_cycles"),
            "flash_direct_synth_status": flash.get("direct_synth_status"),
            "flash_direct_csim_status": flash.get("direct_csim_status"),
            "flash_direct_cosim_status": flash.get("direct_cosim_status"),
            "flash_direct_cycles": flash.get("direct_cycles"),
            "flash_direct_cosim_cycles": flash.get("direct_cosim_cycles"),
            "flash_skill_trace_techniques": flash.get("skill_trace_techniques"),
            "flash_skill_trace_total_tokens": flash.get("skill_trace_total_tokens"),
            "csynth_speedup_flash_over_multistep": _ratio(ms.get("multistep_best_cycles"), flash.get("best_cycles")),
            "cosim_speedup_flash_over_multistep": _ratio(ms.get("multistep_final_cosim_cycles"), flash.get("agent_step_cosim_cycles")),
        })
        rows.append({key: _cell(value) for key, value in out.items()})

    _write_csv(args.out_csv, rows)

    completed = len(rows)
    success = sum(1 for row in rows if str(row.get("multistep_success")).lower() == "true")
    cosim = sum(1 for row in rows if row.get("multistep_final_cosim_cycles") not in ("", None))
    lines = [
        "# HLSFactory Multistep vs Flash Comparison",
        "",
        f"- multistep summary: `{args.multistep_summary}`",
        f"- flash comparison table: `{args.flash_csv}`",
        f"- output csv: `{args.out_csv}`",
        f"- rows: {completed}",
        f"- multistep successes: {success}",
        f"- multistep rows with final cosim cycles: {cosim}",
        "",
        "| bench | skill | multistep status | multistep best | multistep cycles | flash best | flash cycles | multistep cosim | flash cosim | skill ids |",
        "|---|---|---|---|---:|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('bench')} | {row.get('skill_mode')} | {row.get('multistep_success')} | "
            f"{row.get('multistep_best_step') or '-'} | {row.get('multistep_best_cycles') or '-'} | "
            f"{row.get('flash_best_step') or '-'} | {row.get('flash_best_cycles') or '-'} | "
            f"{row.get('multistep_final_cosim_cycles') or '-'} | {row.get('flash_agent_cosim_cycles') or '-'} | "
            f"{row.get('multistep_skill_ids') or '-'} |"
        )
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(lines) + "\n")
    print(args.out_csv)
    print(args.out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
