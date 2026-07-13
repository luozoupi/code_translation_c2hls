#!/usr/bin/env python3
"""Run one fingerprinted HPCA 2027 matched-baseline matrix cell.

The matrix expander supplies one benchmark, one model, one method, and one
base seed through environment variables.  This runner intentionally refuses
multi-cell invocations so each process has an isolated model seed, transcript,
fingerprint, and result artifact.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Sequence

from evaluation_repro import apply_evaluation_profile
from paper_baselines import (
    SUPPORTED_METHODS,
    load_public_benchmark_inputs,
    run_baseline_case,
)


REPO = Path(__file__).resolve().parent


def _single_env(name: str) -> str:
    values = [item.strip() for item in os.getenv(name, "").split(",") if item.strip()]
    if len(values) != 1:
        raise ValueError(f"{name} must select exactly one value, got {values!r}")
    return values[0]


def _safe_label(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "cell"


def _benchmark_dir(name: str) -> Path:
    explicit_root = os.getenv("C2HLS_SWEEP_BENCHMARKS_DIR")
    root = Path(explicit_root).expanduser() if explicit_root else REPO / "benchmarks"
    candidate = root / name
    if not candidate.is_dir():
        raise FileNotFoundError(f"benchmark directory not found: {candidate}")
    return candidate


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run one reference-blind best-of-five or pragma-only paper baseline"
    )
    parser.add_argument("--method", choices=SUPPORTED_METHODS)
    parser.add_argument("--benchmark")
    parser.add_argument("--model")
    parser.add_argument("--base-seed", type=int)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)

    method = args.method or os.getenv("C2HLS_BASELINE_METHOD") or os.getenv(
        "C2HLS_STRATEGY"
    )
    if method not in SUPPORTED_METHODS:
        raise ValueError(
            f"C2HLS_BASELINE_METHOD must be one of {SUPPORTED_METHODS!r}, got {method!r}"
        )
    benchmark = args.benchmark or _single_env("C2HLS_SWEEP_BENCHES")
    model_id = args.model or _single_env("C2HLS_SWEEP_MODELS")
    base_seed = (
        args.base_seed
        if args.base_seed is not None
        else int(os.getenv("C2HLS_LLM_SEED", "0"))
    )
    profile = apply_evaluation_profile()
    if not profile.get("reference_blind"):
        raise ValueError("paper baseline runner is reference-blind only")

    benchmark_dir = _benchmark_dir(benchmark)
    inputs = load_public_benchmark_inputs(benchmark_dir)
    stamp = os.getenv("C2HLS_SWEEP_STAMP") or "manual"
    output_dir = args.output_dir or (
        Path(os.getenv("C2HLS_BASELINE_OUTPUT_ROOT", REPO / "artifacts" / "hpca2027_baselines"))
        / _safe_label(stamp)
    )
    output_dir = output_dir.resolve()
    result_path = output_dir / f"{_safe_label(benchmark)}_results.json"
    model_label = os.getenv("C2HLS_MODEL_LABEL") or _safe_label(model_id)
    resume = str(os.getenv("C2HLS_SWEEP_RESUME", "1")).lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    result = run_baseline_case(
        repo=REPO,
        inputs=inputs,
        method=method,
        model_id=model_id,
        model_label=model_label,
        base_seed=base_seed,
        profile=profile,
        output_dir=output_dir,
        result_path=result_path,
        resume=resume,
    )
    print(
        json.dumps(
            {
                "benchmark": benchmark,
                "method": method,
                "model": model_id,
                "success": bool(result.get("success")),
                "result": str(result_path),
                "transcript": (result.get("run") or {}).get("transcript_file"),
                "reference_isolation_audit": (result.get("run") or {}).get(
                    "reference_isolation_audit_path"
                ),
                "llm_calls": (result.get("llm_usage") or {}).get("calls"),
                "selection_synthesis_evaluations": (
                    result.get("synthesis_evaluations") or {}
                ).get("count"),
                "selected_winner_cosim_count": result.get(
                    "selected_winner_cosim_count"
                ),
                "total_synthesis_calls": result.get("total_synthesis_calls"),
                "run_fingerprint_sha256": (
                    result.get("run_fingerprint") or {}
                ).get("sha256"),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
