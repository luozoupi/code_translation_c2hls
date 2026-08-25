#!/usr/bin/env python3
"""Run a resumable, bounded multi-benchmark QoR OFAT campaign."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "scripts" / "run_saved_qor_step_smoke.py"
sys.path.insert(0, str(REPO))

from qor_design_space import STEP_PREFERRED_KINDS, discover_qor_knobs  # noqa: E402


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _completed_output(path: Path, source_sha256: str) -> bool:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        payload.get("schema_version") == "c2hls.saved-qor-step-smoke.v1"
        and (payload.get("source_result") or {}).get("sha256") == source_sha256
        and (payload.get("design_sweep") or {}).get("attempted") is True
    )


def _case_command(
    case: dict[str, Any],
    config: dict[str, Any],
    output: Path,
) -> list[str]:
    benchmarks_root = Path(config["benchmarks_root"])
    benchmark = str(case["benchmark"])
    command = [
        sys.executable,
        str(RUNNER),
        "--result-json",
        str(case["result_path"]),
        "--bench-dir",
        str(benchmarks_root / benchmark),
        "--origin-step",
        str(case["origin_step"]),
        "--shape-registry",
        str(config["shape_registry"]),
        "--output",
        str(output),
        "--part",
        str(config.get("part", "xcu280-fsvh2892-2L-e")),
        "--clock-ns",
        str(config.get("clock_ns", 3.33)),
        "--vitis-version",
        str(config.get("vitis_version", "2023.2")),
        "--max-knobs",
        str(case.get("max_knobs", 1)),
        "--max-candidates",
        str(case["max_candidates"]),
        "--factor-values",
        str(case.get("factor_values", "1,2,4,8,16")),
        "--ii-values",
        str(case.get("ii_values", "1,2,4,8")),
        "--tile-values",
        str(case.get("tile_values", "4,8,16,32,64")),
    ]
    if case.get("expected_knob_kind"):
        command.extend(["--expected-knob-kind", str(case["expected_knob_kind"])])
    if case.get("expected_knob_name"):
        command.extend(["--expected-knob-name", str(case["expected_knob_name"])])
    return command


def _values(raw: Any) -> tuple[int, ...]:
    return tuple(
        int(item.strip())
        for item in str(raw).split(",")
        if item.strip()
    )


def _preflight_case(case: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    source_path = Path(case["result_path"])
    benchmark_dir = Path(config["benchmarks_root"]) / str(case["benchmark"])
    errors = []
    result: dict[str, Any] = {}
    try:
        result = json.loads(source_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"source_result_unreadable: {exc}")
    if not benchmark_dir.is_dir():
        errors.append(f"benchmark_dir_missing: {benchmark_dir}")
    if result:
        if result.get("benchmark") != case.get("benchmark"):
            errors.append("benchmark_identity_mismatch")
        if (result.get("csim") or {}).get("passed") is not True:
            errors.append("parent_csim_not_passed")
        if not result.get("final_report"):
            errors.append("parent_csynth_report_missing")
    origin = str(case["origin_step"]).lower().replace("double_buffer", "doublebuffer")
    knobs = discover_qor_knobs(
        str(result.get("hls_code") or ""),
        factor_values=_values(case.get("factor_values", "1,2,4,8,16")),
        ii_values=_values(case.get("ii_values", "1,2,4,8")),
        tile_values=_values(case.get("tile_values", "4,8,16,32,64")),
        stream_depth_values=_values(case.get("factor_values", "1,2,4,8,16")),
        max_knobs=int(case.get("max_knobs", 1)),
        preferred_kinds=STEP_PREFERRED_KINDS.get(origin, ()),
    )
    first = knobs[0] if knobs else None
    if first is None:
        errors.append("no_qor_knob_selected")
    elif first.kind != case.get("expected_knob_kind"):
        errors.append(
            f"knob_kind_mismatch: expected {case.get('expected_knob_kind')}, "
            f"got {first.kind}"
        )
    elif first.name != case.get("expected_knob_name"):
        errors.append(
            f"knob_name_mismatch: expected {case.get('expected_knob_name')}, "
            f"got {first.name}"
        )
    return {
        "case_id": case["case_id"],
        "benchmark": case["benchmark"],
        "passed": not errors,
        "errors": errors,
        "source_sha256": _sha256(source_path) if source_path.is_file() else None,
        "selected_knob": first.public() if first is not None else None,
        "requested_candidate_count": int(case["max_candidates"]),
        "discoverable_candidate_count": (
            len(first.candidate_values) if first is not None else 0
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    cases = list(config.get("cases") or [])
    workers = max(1, min(2, int(args.workers)))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "logs").mkdir(parents=True, exist_ok=True)
    args.scratch_root.mkdir(parents=True, exist_ok=True)
    preflight = {
        "schema_version": "c2hls.qor-ofat-campaign-preflight.v1",
        "campaign_id": config.get("campaign_id"),
        "reference_blind": config.get("reference_blind") is True,
        "model_calls": config.get("model_calls"),
        "cosim": config.get("cosim"),
        "cases": [_preflight_case(case, config) for case in cases],
    }
    preflight["passed"] = bool(preflight["cases"]) and all(
        case["passed"] for case in preflight["cases"]
    )
    preflight["requested_candidate_count"] = sum(
        case["requested_candidate_count"] for case in preflight["cases"]
    )
    _atomic_json(args.output_dir / "preflight.json", preflight)
    if args.preflight_only or not preflight["passed"]:
        print(json.dumps({
            "preflight": str(args.output_dir / "preflight.json"),
            "passed": preflight["passed"],
            "case_count": len(preflight["cases"]),
            "requested_candidate_count": preflight["requested_candidate_count"],
        }))
        return 0 if preflight["passed"] else 1
    state_path = args.output_dir / "campaign_state.json"
    lock = threading.Lock()
    state: dict[str, Any] = {
        "schema_version": "c2hls.qor-ofat-campaign-state.v1",
        "campaign_id": config.get("campaign_id"),
        "config_path": str(args.config.resolve()),
        "config_sha256": _sha256(args.config),
        "started_at": _now(),
        "updated_at": _now(),
        "workers": workers,
        "reference_blind": True,
        "model_calls": 0,
        "cosim": False,
        "cases": {},
    }

    def save() -> None:
        state["updated_at"] = _now()
        _atomic_json(state_path, state)

    def run_case(case: dict[str, Any]) -> dict[str, Any]:
        case_id = str(case["case_id"])
        source_path = Path(case["result_path"])
        source_sha256 = _sha256(source_path)
        output = args.output_dir / "cases" / f"{case_id}.json"
        log_path = args.output_dir / "logs" / f"{case_id}.log"
        if _completed_output(output, source_sha256):
            return {
                "case_id": case_id,
                "status": "complete",
                "resumed": True,
                "output": str(output),
                "source_sha256": source_sha256,
            }

        scratch = args.scratch_root / case_id
        env = dict(os.environ)
        env.update({
            "C2HLS_TMP_ROOT": str(scratch / "work"),
            "C2HLS_VITIS_USER_HOME": str(scratch / "vitis_home"),
            "C2HLS_VITIS_SETTINGS": str(config["vitis_settings"]),
            "C2HLS_PART": str(config.get("part", "xcu280-fsvh2892-2L-e")),
            "C2HLS_CLOCK_NS": str(config.get("clock_ns", 3.33)),
            "C2HLS_CSIM_TIMEOUT": str(config.get("csim_timeout_s", 600)),
            "C2HLS_SYNTH_TIMEOUT": str(config.get("synth_timeout_s", 1200)),
        })
        command = _case_command(case, config, output)
        started_at = _now()
        with log_path.open("w") as log_handle:
            completed = subprocess.run(
                command,
                cwd=REPO,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        return {
            "case_id": case_id,
            "status": "complete" if completed.returncode == 0 else "failed",
            "resumed": False,
            "returncode": completed.returncode,
            "started_at": started_at,
            "finished_at": _now(),
            "output": str(output),
            "log": str(log_path),
            "scratch": str(scratch),
            "source_sha256": source_sha256,
            "command": command,
        }

    save()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_cases = {executor.submit(run_case, case): case for case in cases}
        for future in concurrent.futures.as_completed(future_cases):
            case = future_cases[future]
            case_id = str(case["case_id"])
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "case_id": case_id,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "finished_at": _now(),
                }
            with lock:
                state["cases"][case_id] = result
                save()
            print(json.dumps({
                "case_id": case_id,
                "status": result["status"],
                "output": result.get("output"),
            }), flush=True)

    statuses = [item.get("status") for item in state["cases"].values()]
    state["status"] = "complete" if statuses and all(
        status == "complete" for status in statuses
    ) else "failed"
    state["finished_at"] = _now()
    save()
    return 0 if state["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
