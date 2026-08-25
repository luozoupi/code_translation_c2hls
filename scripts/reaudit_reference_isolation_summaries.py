#!/usr/bin/env python3
"""Re-audit persisted sweep transcripts with current isolation rules."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from reference_isolation import audit_history_file  # noqa: E402


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Reference-Isolation Transcript Re-audit",
        "",
        f"Status: **{report['status'].upper()}**",
        "",
        "| Benchmark | Strategy | Skill mode | Audit | Allowed generated-metric collisions |",
        "|---|---|---|---|---:|",
    ]
    for row in report["rows"]:
        lines.append(
            f"| {row['benchmark']} | {row['strategy']} | {row['skill_mode']} | "
            f"{'PASS' if row['audit']['passed'] else 'FAIL'} | "
            f"{row['audit'].get('allowed_controller_metric_match_count', 0)} |"
        )
    lines.extend([
        "",
        "Allowed collisions are exact values independently present in generated-candidate Vitis reports. "
        "The report stores hashes and counts, not the values.",
    ])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for summary_path in args.summary:
        summary = json.loads(summary_path.read_text())
        for summary_row in summary.get("rows") or []:
            current = summary_row.get("current") or {}
            result_path = Path(str(current.get("json") or ""))
            benchmark = str(summary_row.get("bench") or "")
            benchmark_dir = Path(str(summary_row.get("bench_dir") or ""))
            history_path = result_path.parent / f"{benchmark}_history.json"
            controller_data: dict[str, Any] = {}
            try:
                controller_data = json.loads(result_path.read_text())
                audit = audit_history_file(
                    history_path,
                    benchmark_dir=benchmark_dir,
                    reference_data=controller_data.get("reference_validation"),
                    controller_data=controller_data,
                )
            except (OSError, json.JSONDecodeError) as exc:
                audit = {
                    "schema_version": "c2hls.reference-isolation-audit.v1",
                    "passed": False,
                    "finding_count": 0,
                    "finding_counts": {},
                    "findings": [],
                    "error": f"re-audit input unavailable: {type(exc).__name__}",
                }
            strategy = str(controller_data.get("phase") or "")
            if not strategy:
                stamp = str(summary.get("stamp") or "")
                strategy = "dynamic" if "_dynamic_" in stamp else "flash"
            elif strategy == "multistep":
                strategy = "dynamic"
            skill_mode = str(summary_row.get("skill_mode") or "")
            row = {
                "summary": str(summary_path.resolve()),
                "benchmark": benchmark,
                "strategy": strategy,
                "skill_mode": skill_mode,
                "result": str(result_path),
                "history": str(history_path),
                "original_audit": current.get("reference_isolation_audit"),
                "audit": audit,
            }
            rows.append(row)
            if not audit.get("passed"):
                failures.append(f"{benchmark}/{strategy}/{skill_mode}")

    report = {
        "schema_version": "c2hls.reference-isolation-reaudit.v1",
        "status": "passed" if not failures else "failed",
        "summary_count": len(args.summary),
        "arm_count": len(rows),
        "passed_count": sum(bool(row["audit"].get("passed")) for row in rows),
        "failure_count": len(failures),
        "failures": failures,
        "rows": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "smoke_reaudit.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    (args.output_dir / "smoke_reaudit.md").write_text(_markdown(report))
    print(json.dumps({key: report[key] for key in (
        "status", "summary_count", "arm_count", "passed_count", "failure_count"
    )}, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
