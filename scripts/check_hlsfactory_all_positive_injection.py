#!/usr/bin/env python3
"""Summarize all-positive skill injection coverage for a running sweep."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
DEFAULT_STAMP = "hlsfactory_flash_sonnet46_all_positive_20260630"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        return {"_load_error": str(exc)}


def _skill_sets() -> tuple[set[str], set[str]]:
    data = _load_json(REPO / "skills" / "skills.json")
    positive: set[str] = set()
    avoid: set[str] = set()
    for skill in data.get("skills") or []:
        sid = skill.get("id")
        if not sid:
            continue
        if (
            skill.get("confidence") == "avoid"
            or skill.get("kind") == "avoid_rule"
            or sid.startswith("avoid-")
            or sid.startswith("hls-avoid-")
        ):
            avoid.add(sid)
        else:
            positive.add(sid)
    return positive, avoid


def _tmux_sessions() -> list[str]:
    try:
        proc = subprocess.run(
            ["tmux", "list-sessions"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except FileNotFoundError:
        return []
    if proc.returncode != 0:
        return []
    return [line.split(":", 1)[0] for line in proc.stdout.splitlines() if line.strip()]


def summarize(stamp: str, exclude: set[str]) -> dict[str, Any]:
    positive, avoid = _skill_sets()
    results_root = REPO / "results_sweeps" / f"agentic_no_streamcluster_{stamp}"
    log_path = REPO / "artifacts" / f"{stamp}.queue.log"
    if not log_path.exists():
        log_path = REPO / "artifacts" / "hlsfactory_flash_sonnet46_all_positive_20260630.queue.log"
    summary_path = REPO / "artifacts" / f"agentic_no_streamcluster_{stamp}.summary.json"
    jsonl_path = REPO / "artifacts" / f"agentic_no_streamcluster_{stamp}.jsonl"

    bench_root = REPO / "benchmarks_external" / "HLSFactory" / "polybench_float_small"
    expected = [
        path.parent.name
        for path in sorted(bench_root.glob("*/metadata.json"))
        if path.parent.name not in exclude
    ]

    rows = []
    bad_prompt_records = []
    no_prompt_steps = []
    prompt_records = 0
    for path in sorted(results_root.glob("*/*_multistep_results.json")):
        data = _load_json(path)
        bench = data.get("benchmark") or path.name.replace("_multistep_results.json", "")
        step_rows = []
        for step in data.get("steps") or []:
            prompts = []
            if step.get("skill_prompt"):
                prompts.append(("step", step["skill_prompt"]))
            for idx, attempt in enumerate(step.get("candidate_attempts") or []):
                if attempt.get("skill_prompt"):
                    prompts.append((f"attempt{idx}", attempt["skill_prompt"]))
            if not prompts:
                no_prompt_steps.append({
                    "benchmark": bench,
                    "step": step.get("step_name"),
                    "success": step.get("success"),
                    "error": step.get("error"),
                    "stage": step.get("stage"),
                })
            for where, prompt in prompts:
                prompt_records += 1
                ids = set(prompt.get("injected_skill_ids") or [])
                row = {
                    "benchmark": bench,
                    "where": where,
                    "prompt_scope": prompt.get("prompt_scope"),
                    "prompt_mode": prompt.get("prompt_mode"),
                    "injected_count": len(ids),
                    "missing_positive": sorted(positive - ids),
                    "avoid_injected": sorted(ids & avoid),
                    "avoid_skill_ids": prompt.get("avoid_skill_ids") or [],
                }
                step_rows.append(row)
                if (
                    row["prompt_scope"] != "all_positive"
                    or row["prompt_mode"] != "action_only"
                    or row["missing_positive"]
                    or row["avoid_injected"]
                    or row["avoid_skill_ids"]
                ):
                    bad_prompt_records.append(row)
        rows.append({
            "benchmark": bench,
            "success": data.get("success"),
            "steps": len(data.get("steps") or []),
            "step_prompt_checks": step_rows,
            "path": str(path),
        })

    done_lines = []
    if log_path.exists():
        done_lines = [
            line for line in log_path.read_text(errors="replace").splitlines()
            if line.startswith("DONE bench=")
        ]

    sessions = _tmux_sessions()
    return {
        "checked_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "stamp": stamp,
        "tmux_sessions": sessions,
        "sweep_session_running": "hlsfactory_flash_allpos_20260630" in sessions,
        "results_root": str(results_root),
        "log_path": str(log_path),
        "summary_exists": summary_path.exists(),
        "jsonl_exists": jsonl_path.exists(),
        "expected_benchmarks": len(expected),
        "completed_results": len(rows),
        "done_lines": len(done_lines),
        "positive_skill_count": len(positive),
        "avoid_skill_count": len(avoid),
        "prompt_records_checked": prompt_records,
        "bad_prompt_records": bad_prompt_records,
        "step_records_without_prompt": no_prompt_steps,
        "all_prompt_records_ok": not bad_prompt_records,
        "completed": rows,
    }


def write_report(summary: dict[str, Any], out_prefix: Path) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = Path(str(out_prefix) + ".json")
    md_path = Path(str(out_prefix) + ".md")
    json_path.write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        "# HLSFactory All-Positive Injection Status",
        "",
        f"checked_at: `{summary['checked_at']}`",
        f"stamp: `{summary['stamp']}`",
        f"session_running: `{summary['sweep_session_running']}`",
        f"completed_results: `{summary['completed_results']}/{summary['expected_benchmarks']}`",
        f"done_lines: `{summary['done_lines']}`",
        f"prompt_records_checked: `{summary['prompt_records_checked']}`",
        f"positive_skill_count: `{summary['positive_skill_count']}`",
        f"avoid_skill_count: `{summary['avoid_skill_count']}`",
        f"all_prompt_records_ok: `{summary['all_prompt_records_ok']}`",
        f"bad_prompt_records: `{len(summary['bad_prompt_records'])}`",
        f"step_records_without_prompt: `{len(summary['step_records_without_prompt'])}`",
        "",
        "| benchmark | success | prompt records | status |",
        "|---|---:|---:|---|",
    ]
    bad_by_bench = {row["benchmark"] for row in summary["bad_prompt_records"]}
    no_prompt_by_bench = {row["benchmark"] for row in summary["step_records_without_prompt"]}
    for row in summary["completed"]:
        bench = row["benchmark"]
        prompt_count = len(row["step_prompt_checks"])
        if bench in bad_by_bench:
            status = "bad_prompt"
        elif prompt_count:
            status = "42/42 positive, 0 avoid"
        elif bench in no_prompt_by_bench:
            status = "no prompt metadata"
        else:
            status = "-"
        lines.append(f"| {bench} | {row['success']} | {prompt_count} | {status} |")
    md_path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stamp", default=DEFAULT_STAMP)
    parser.add_argument("--exclude", default="hlsfactory_doitgen")
    parser.add_argument(
        "--out-prefix",
        default=str(REPO / "artifacts" / "hlsfactory_flash_sonnet46_all_positive_20260630.injection_status"),
    )
    args = parser.parse_args()
    summary = summarize(args.stamp, {item.strip() for item in args.exclude.split(",") if item.strip()})
    write_report(summary, Path(args.out_prefix))
    print(json.dumps({
        key: summary[key]
        for key in [
            "checked_at",
            "sweep_session_running",
            "completed_results",
            "expected_benchmarks",
            "prompt_records_checked",
            "all_prompt_records_ok",
        ]
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
