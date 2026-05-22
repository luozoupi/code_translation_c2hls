#!/usr/bin/env python3
"""Validate schema-1.0 JSONL records for benchmark-quality semantics.

`export_schema_jsonl.validate_jsonl` checks the envelope. This script checks
payload meaning: pass records must contain usable metrics, timing/resource
numbers must be plausible, and optional external-dataset quality checks flag
schema-valid synth results that look like non-compute stubs.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from export_schema_jsonl import validate_jsonl  # noqa: E402


RESOURCE_KEYS = {
    "bram": ("BRAM_18K", "BRAM", "bram"),
    "dsp": ("DSP", "dsp"),
    "ff": ("FF", "ff"),
    "lut": ("LUT", "lut"),
    "uram": ("URAM", "uram"),
}
INIT_LOOP_RE = re.compile(r"^(?:init|initialize|zero|clear|reset)(?:_|$)", re.IGNORECASE)


@dataclass
class Issue:
    line: int
    severity: str
    code: str
    message: str
    problem: str = ""
    report_type: str = ""


def _problem_key(record: dict[str, Any]) -> str:
    problem = record.get("problem") or {}
    group = problem.get("group_path") or []
    return "/".join(group) if isinstance(group, list) else ""


def _num(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip().replace(",", "")
        if not text or text in {"-", "?", "NA", "N/A"}:
            return None
        match = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", text)
        if not match:
            return None
        try:
            return float(match.group(0))
        except ValueError:
            return None
    return None


def _nested(obj: dict[str, Any], *keys: str) -> Any:
    cur: Any = obj
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _latency_cycles(payload: dict[str, Any]) -> float | None:
    summary = _nested(payload, "PerformanceEstimates", "SummaryOfOverallLatency") or {}
    for key in ("Worst-caseLatency", "Average-caseLatency", "Best-caseLatency"):
        value = _num(summary.get(key))
        if value is not None:
            return value
    return None


def _interval_cycles(payload: dict[str, Any]) -> float | None:
    summary = _nested(payload, "PerformanceEstimates", "SummaryOfOverallLatency") or {}
    for key in ("Interval-max", "Interval-min"):
        value = _num(summary.get(key))
        if value is not None:
            return value
    return None


def _estimated_clock_ns(payload: dict[str, Any]) -> float | None:
    return _num(_nested(payload, "PerformanceEstimates", "SummaryOfTimingAnalysis", "EstimatedClockPeriod"))


def _target_clock_ns(record: dict[str, Any], payload: dict[str, Any]) -> float | None:
    run_clock = _num((record.get("run") or {}).get("clock_ns"))
    if run_clock is not None:
        return run_clock
    return _num(_nested(payload, "UserAssignments", "TargetClockPeriod"))


def _resource(resources: dict[str, Any], logical: str) -> float | None:
    for key in RESOURCE_KEYS[logical]:
        value = _num(resources.get(key))
        if value is not None:
            return value
    return None


def _selected_attempt(record: dict[str, Any]) -> dict[str, Any] | None:
    meta = ((record.get("implementation") or {}).get("origin_meta") or {})
    attempts = meta.get("candidate_attempts")
    if not isinstance(attempts, list):
        return None
    selected_candidate = meta.get("selected_candidate_index")
    selected_attempt = meta.get("selected_attempt_index")
    for attempt in attempts:
        if not isinstance(attempt, dict):
            continue
        if (
            selected_candidate is not None
            and selected_attempt is not None
            and attempt.get("candidate_index") == selected_candidate
            and attempt.get("attempt_index") == selected_attempt
        ):
            return attempt
    for attempt in attempts:
        if isinstance(attempt, dict) and selected_candidate is not None and attempt.get("candidate_index") == selected_candidate:
            return attempt
    return None


def _substantive_loop_count(record: dict[str, Any], top_latency: float | None) -> int | None:
    attempt = _selected_attempt(record)
    if not attempt:
        return None
    scopes = _nested(attempt, "report", "feedback", "scopes")
    if not isinstance(scopes, list):
        return None
    count = 0
    min_loop_latency = max(8.0, (top_latency or 0.0) * 0.02)
    for scope in scopes:
        if not isinstance(scope, dict) or scope.get("kind") != "loop":
            continue
        name = str(scope.get("name") or "")
        if INIT_LOOP_RE.search(name):
            continue
        trip = _num(scope.get("trip_count"))
        lat = _num(scope.get("latency_cycles"))
        if (trip is not None and trip > 1) and (lat is None or lat >= min_loop_latency):
            count += 1
    return count


def _severity_for_suspicious(args: argparse.Namespace) -> str:
    return "error" if args.quality_profile == "external" or args.strict_suspicious else "warn"


def _add(issues: list[Issue], line: int, severity: str, code: str, message: str, record: dict[str, Any]) -> None:
    issues.append(Issue(
        line=line,
        severity=severity,
        code=code,
        message=message,
        problem=_problem_key(record),
        report_type=str(record.get("report_type") or ""),
    ))


def _check_hls_synth(record: dict[str, Any], line: int, args: argparse.Namespace, issues: list[Issue]) -> None:
    payload = record.get("hls_synth") or {}
    status = payload.get("status")
    if status != "pass":
        return

    latency = _latency_cycles(payload)
    interval = _interval_cycles(payload)
    if latency is None or latency <= 0:
        _add(issues, line, "error", "missing_latency", "passing hls_synth record lacks positive latency cycles", record)
    if interval is None or interval <= 0:
        _add(issues, line, "error", "missing_interval", "passing hls_synth record lacks positive interval cycles", record)

    estimated = _estimated_clock_ns(payload)
    target = _target_clock_ns(record, payload)
    if estimated is None:
        _add(issues, line, "error", "missing_estimated_clock", "passing hls_synth record lacks estimated clock period", record)
    elif target is not None and estimated > target + args.clock_tolerance_ns:
        _add(
            issues,
            line,
            "error",
            "timing_not_clean",
            f"estimated clock {estimated:g} ns exceeds target {target:g} ns",
            record,
        )

    resources = _nested(payload, "AreaEstimates", "Resources") or {}
    available = _nested(payload, "AreaEstimates", "AvailableResources") or {}
    for logical in RESOURCE_KEYS:
        used = _resource(resources, logical)
        cap = _resource(available, logical)
        if used is None:
            _add(issues, line, "error", f"missing_{logical}", f"passing hls_synth record lacks {logical.upper()} resource count", record)
            continue
        if used < 0:
            _add(issues, line, "error", f"negative_{logical}", f"{logical.upper()} resource count is negative", record)
        if cap is not None and used > cap:
            _add(
                issues,
                line,
                "error",
                f"{logical}_over_capacity",
                f"{logical.upper()} usage {used:g} exceeds available {cap:g}",
                record,
            )

    if latency is not None and args.min_latency_cycles and latency < args.min_latency_cycles:
        _add(
            issues,
            line,
            _severity_for_suspicious(args),
            "suspicious_low_latency",
            f"latency {latency:g} cycles is below configured floor {args.min_latency_cycles:g}",
            record,
        )

    dsp = _resource(resources, "dsp")
    if latency is not None and dsp == 0 and latency <= args.zero_dsp_low_latency_cycles:
        _add(
            issues,
            line,
            _severity_for_suspicious(args),
            "suspicious_zero_dsp_low_latency",
            f"latency {latency:g} cycles with DSP=0 is suspicious for benchmark-quality external results",
            record,
        )

    substantive_loops = _substantive_loop_count(record, latency)
    if substantive_loops == 0:
        _add(
            issues,
            line,
            _severity_for_suspicious(args),
            "no_substantive_compute_loop",
            "selected attempt feedback has loop scopes but none beyond initialization/reset-style loops",
            record,
        )


def _check_rtl_sim(record: dict[str, Any], line: int, _args: argparse.Namespace, issues: list[Issue]) -> None:
    payload = record.get("rtl_sim") or {}
    status = payload.get("status")
    if status != "pass":
        return
    for key in ("kernel_runtime_cycles", "kernel_runtime_us", "kernel_clock_freq_mhz"):
        value = _num(payload.get(key))
        if value is None or value <= 0:
            _add(issues, line, "error", f"missing_{key}", f"passing rtl_sim record lacks positive {key}", record)


def validate_semantics(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    schema = validate_jsonl(path, verbose=False)
    issues: list[Issue] = []
    if schema.get("invalid"):
        for item in schema.get("errors", []):
            for err in item.get("errors", []):
                issues.append(Issue(
                    line=int(item.get("line") or 0),
                    severity="error",
                    code="schema_invalid",
                    message=str(err),
                ))
        return {
            "path": str(path),
            "schema": schema,
            "semantic": {"errors": len(issues), "warnings": 0, "issues": [asdict(i) for i in issues]},
        }

    for line_no, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        rt = record.get("report_type")
        if rt == "hls_synth":
            _check_hls_synth(record, line_no, args, issues)
        elif rt == "rtl_sim":
            _check_rtl_sim(record, line_no, args, issues)

    errors = sum(1 for issue in issues if issue.severity == "error")
    warnings = sum(1 for issue in issues if issue.severity == "warn")
    return {
        "path": str(path),
        "schema": schema,
        "semantic": {
            "errors": errors,
            "warnings": warnings,
            "issues": [asdict(issue) for issue in issues],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", type=Path, nargs="+")
    parser.add_argument("--quality-profile", choices=("default", "external"), default="default",
                        help="external promotes suspicious external-dataset patterns to errors.")
    parser.add_argument("--strict-suspicious", action="store_true",
                        help="Promote suspicious-record warnings to errors.")
    parser.add_argument("--min-latency-cycles", type=float, default=0.0,
                        help="Optional floor for passing hls_synth latency.")
    parser.add_argument("--zero-dsp-low-latency-cycles", type=float, default=512.0)
    parser.add_argument("--clock-tolerance-ns", type=float, default=1e-3)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    summaries = [validate_semantics(path, args) for path in args.jsonl]
    if args.json:
        print(json.dumps(summaries, indent=2))
    else:
        for summary in summaries:
            semantic = summary["semantic"]
            schema = summary["schema"]
            print(
                f"{summary['path']}: schema_invalid={schema.get('invalid')} "
                f"semantic_errors={semantic['errors']} semantic_warnings={semantic['warnings']}"
            )
            for issue in semantic["issues"][:50]:
                print(
                    f"  {issue['severity'].upper()} line {issue['line']} "
                    f"{issue['code']} {issue.get('problem') or '-'}: {issue['message']}"
                )
            if len(semantic["issues"]) > 50:
                print(f"  ... {len(semantic['issues']) - 50} more issues")

    failed = any(item["schema"].get("invalid") or item["semantic"]["errors"] for item in summaries)
    if args.quality_profile == "external":
        failed = failed or any(item["semantic"]["warnings"] for item in summaries)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
