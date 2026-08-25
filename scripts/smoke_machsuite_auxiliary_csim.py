#!/usr/bin/env python3
"""Run a real Vitis CSim smoke for MachSuite auxiliary testbench files."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import c2hls
from hls_eval import run_csim


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _top_function(benchmark_dir: Path, metadata: dict) -> str:
    configured = str(
        metadata.get("top_function") or metadata.get("top") or ""
    ).strip()
    if configured:
        return configured
    top_path = benchmark_dir / "top.txt"
    if top_path.exists():
        top = top_path.read_text(encoding="utf-8").strip()
        if top:
            return top
    raise ValueError(f"no top function configured for {benchmark_dir.name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmarks-dir",
        type=Path,
        default=REPO_ROOT / "benchmarks_external" / "hls_eval",
    )
    parser.add_argument(
        "--benchmarks",
        default="hlseval_machsuite_aes_aes",
        help="Comma-separated benchmark directory names.",
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        default=None,
        help="Retain Vitis work here; otherwise use a temporary directory.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--part", default="xcu280-fsvh2892-2L-e")
    parser.add_argument("--clock-ns", type=float, default=3.33)
    args = parser.parse_args()

    if args.work_root is None:
        work_root = Path(tempfile.mkdtemp(prefix="c2hls_machsuite_aux_csim_"))
    else:
        work_root = args.work_root.resolve()
        work_root.mkdir(parents=True, exist_ok=True)
    os.environ["C2HLS_TMP_ROOT"] = str(work_root)
    os.environ["C2HLS_VITIS_USER_HOME"] = str(work_root / "vitis_user_home")

    results = []
    for benchmark in [
        item.strip() for item in args.benchmarks.split(",") if item.strip()
    ]:
        benchmark_dir = (args.benchmarks_dir / benchmark).resolve()
        inputs = c2hls._load_benchmark_inputs(str(benchmark_dir))
        top_function = _top_function(benchmark_dir, inputs["meta"])
        work_dir = work_root / benchmark
        work_dir.mkdir(parents=True, exist_ok=True)
        result = run_csim(
            inputs["ground_truth_code"],
            inputs["testbench_code"],
            inputs["header_code"],
            header_name=inputs["header_name"],
            top_function=top_function,
            part=args.part,
            clock_ns=args.clock_ns,
            work_dir=str(work_dir),
            extra_files=inputs["extra_files"],
        )
        tcl = (work_dir / "run_csim.tcl").read_text(encoding="utf-8")
        auxiliary_paths = [
            str(item["path"])
            for item in inputs["extra_files"]
            if str(item.get("path") or "")
        ]
        staged_paths = [
            path
            for path in auxiliary_paths
            if f"{{{path}}}" in tcl
        ]
        required_runtime_paths = [
            path
            for path in auxiliary_paths
            if Path(path).name.lower()
            not in {
                "hls_eval_config.toml",
                "kernel_description.md",
                "metadata.json",
                "top.txt",
            }
        ]
        results.append(
            {
                "benchmark": benchmark,
                "top_function": top_function,
                "success": result.get("success") is True,
                "passed": result.get("passed") is True,
                "error": result.get("error") or "",
                "auxiliary_file_count": len(auxiliary_paths),
                "required_runtime_file_count": len(required_runtime_paths),
                "required_runtime_files": required_runtime_paths,
                "staged_auxiliary_file_count": len(staged_paths),
                "staged_auxiliary_files": staged_paths,
                "work_dir": str(work_dir),
            }
        )

    payload = {
        "schema_version": "c2hls.machsuite-auxiliary-csim-smoke.v1",
        "created_at": _utc_now(),
        "vitis": {
            "part": args.part,
            "clock_ns": args.clock_ns,
        },
        "work_root": str(work_root),
        "success": all(
            item["success"]
            and set(item["staged_auxiliary_files"])
            == set(item["required_runtime_files"])
            for item in results
        ),
        "results": results,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if payload["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
