#!/usr/bin/env python3
"""Stage durable kernel sources for the strongest Qwen agentic results.

The cycle comparison report points at sweep summaries whose selected kernel
sources still live in temporary Vitis directories.  This script resolves those
sources, verifies their hashes and synthesis-cycle attribution, and copies the
small source files into a durable COSIM queue before scratch relocation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
DEFAULT_COMPARISON = (
    REPO / "artifacts" / "analysis" / "hlsfactory_cycle_setup_comparison_20260725.json"
)
DEFAULT_OUTPUT = (
    REPO / "artifacts" / "qwen_cosim_enrichment" / "strong_gains_20260725_staged"
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cycles(value: Any) -> int | None:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _summary_row_candidates(
    summary: dict[str, Any], gain: dict[str, Any]
) -> list[dict[str, Any]]:
    benchmark = gain["benchmark"]
    expected_cycles = _cycles(gain.get("skilled_cycles"))
    selected_step = gain.get("selected_step_name")
    rows = [
        row
        for row in summary.get("rows", [])
        if isinstance(row, dict) and row.get("bench") == benchmark
    ]
    if len(rows) <= 1:
        return rows

    exact_skill = [row for row in rows if row.get("skill_mode") == gain.get("skill_mode")]
    if exact_skill:
        rows = exact_skill
    if len(rows) <= 1:
        return rows

    cycle_matches: list[dict[str, Any]] = []
    for row in rows:
        for step in (row.get("current") or {}).get("step_cycles") or []:
            if (
                step.get("step") == selected_step
                and _cycles(step.get("cycles")) == expected_cycles
            ):
                cycle_matches.append(row)
                break
    return cycle_matches or rows


def _step_candidates(
    result: dict[str, Any], step_name: str, expected_cycles: int | None
) -> list[dict[str, Any]]:
    steps = [
        step
        for step in result.get("steps") or []
        if isinstance(step, dict)
        and step.get("step_name") == step_name
        and step.get("success")
    ]
    exact = [
        step
        for step in steps
        if _cycles((step.get("report") or {}).get("latency_cycles")) == expected_cycles
    ]
    return exact or steps


def _kernel_paths(step: dict[str, Any]) -> list[Path]:
    paths: list[Path] = []
    for container_name in ("report", "csim", "cosim"):
        work_dir = (step.get(container_name) or {}).get("work_dir")
        if work_dir:
            paths.append(Path(work_dir) / "kernel.cpp")
    for attempt in step.get("attempt_results") or []:
        if not isinstance(attempt, dict):
            continue
        for container_name in ("report", "csim", "cosim"):
            work_dir = (attempt.get(container_name) or {}).get("work_dir")
            if work_dir:
                paths.append(Path(work_dir) / "kernel.cpp")
    return paths


def _resolve_case(gain: dict[str, Any], index: int, output_dir: Path) -> dict[str, Any]:
    source_summary = Path(gain["source_path"]).resolve()
    summary = _read_json(source_summary)
    summary_rows = _summary_row_candidates(summary, gain)
    if len(summary_rows) != 1:
        raise RuntimeError(
            f"{gain['benchmark']}: expected one matching row in {source_summary}, "
            f"found {len(summary_rows)}"
        )
    summary_row = summary_rows[0]
    result_path = Path((summary_row.get("current") or {}).get("json") or "").resolve()
    if not result_path.is_file():
        raise FileNotFoundError(f"selected result JSON is unavailable: {result_path}")
    result = _read_json(result_path)
    expected_cycles = _cycles(gain.get("skilled_cycles"))
    steps = _step_candidates(
        result, str(gain.get("selected_step_name") or ""), expected_cycles
    )
    if len(steps) != 1:
        raise RuntimeError(
            f"{gain['benchmark']}: expected one selected step in {result_path}, "
            f"found {len(steps)}"
        )
    selected_step = steps[0]
    report_cycles = _cycles((selected_step.get("report") or {}).get("latency_cycles"))
    if report_cycles != expected_cycles:
        raise RuntimeError(
            f"{gain['benchmark']}: selected report cycles {report_cycles} do not "
            f"match comparison cycles {expected_cycles}"
        )

    available_sources = [path for path in _kernel_paths(selected_step) if path.is_file()]
    if not available_sources:
        raise FileNotFoundError(
            f"{gain['benchmark']}: no selected kernel.cpp remains for {result_path}"
        )
    source_kernel = available_sources[0]
    source_hash = _sha256(source_kernel)
    for alternate in available_sources[1:]:
        if _sha256(alternate) != source_hash:
            raise RuntimeError(
                f"{gain['benchmark']}: selected step work directories contain "
                "different kernel sources"
            )

    training = str(gain.get("training") or "unknown")
    strategy = str(gain.get("strategy") or "unknown")
    skill_mode = str(gain.get("skill_mode") or "unknown")
    benchmark = str(gain["benchmark"])
    case_id = (
        f"{index:02d}_{_slug(training)}_{_slug(strategy)}_"
        f"{_slug(skill_mode)}_{_slug(benchmark.removeprefix('hlsfactory_'))}"
    )
    case_dir = output_dir / "cases" / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    staged_kernel = case_dir / "kernel.cpp"
    shutil.copy2(source_kernel, staged_kernel)
    if _sha256(staged_kernel) != source_hash:
        raise RuntimeError(f"staged source hash mismatch: {staged_kernel}")

    bench_dir = Path(summary_row["bench_dir"]).resolve()
    metadata_path = bench_dir / "metadata.json"
    metadata = _read_json(metadata_path)
    return {
        "schema_version": "c2hls.qwen_cosim_staged_case.v1",
        "case_id": case_id,
        "priority": index,
        "model": gain.get("model"),
        "training": training,
        "strategy": strategy,
        "benchmark": benchmark,
        "skill_mode": skill_mode,
        "selected_step_name": gain.get("selected_step_name"),
        "csynth_cycles": expected_cycles,
        "reference_cycles": _cycles(gain.get("reference_cycles")),
        "reference_source_kind": gain.get("reference_source_kind"),
        "speedup_vs_skillless": gain.get("speedup_vs_skillless"),
        "speedup_vs_one_shot": gain.get("speedup_vs_one_shot"),
        "speedup_vs_reference": gain.get("speedup_vs_reference"),
        "skill_exposure_confirmed": gain.get("skill_exposure_confirmed"),
        "skill_injected_count": gain.get("skill_injected_count"),
        "reference_isolation_audit_passed": gain.get(
            "reference_isolation_audit_passed"
        ),
        "supports_cosim": bool(metadata.get("supports_cosim")),
        "benchmark_dir": str(bench_dir),
        "metadata_path": str(metadata_path),
        "source_summary": str(source_summary),
        "source_result": str(result_path),
        "source_kernel": str(source_kernel),
        "staged_kernel": str(staged_kernel.resolve()),
        "kernel_sha256": source_hash,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    comparison = _read_json(args.comparison.resolve())
    gains = [
        row
        for row in comparison.get("strong_direct_one_shot_gains") or []
        if isinstance(row, dict) and str(row.get("model") or "").startswith("qwen")
    ]
    if args.limit > 0:
        gains = gains[: args.limit]
    if not gains:
        raise RuntimeError("comparison contains no strong Qwen gains")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = [
        _resolve_case(gain, index, output_dir)
        for index, gain in enumerate(gains, start=1)
    ]
    manifest = {
        "schema_version": "c2hls.qwen_cosim_stage_manifest.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "comparison": str(args.comparison.resolve()),
        "selection": "strong_direct_one_shot_gains where model starts with qwen",
        "case_count": len(cases),
        "cosim_supported_count": sum(case["supports_cosim"] for case in cases),
        "unique_kernel_count": len(
            {(case["benchmark"], case["kernel_sha256"]) for case in cases}
        ),
        "cases": cases,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_jsonl = output_dir / "manifest.jsonl"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    with manifest_jsonl.open("w") as handle:
        for case in cases:
            handle.write(json.dumps(case, sort_keys=True) + "\n")

    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "manifest_jsonl": str(manifest_jsonl),
                "case_count": len(cases),
                "cosim_supported_count": manifest["cosim_supported_count"],
                "unique_kernel_count": manifest["unique_kernel_count"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
