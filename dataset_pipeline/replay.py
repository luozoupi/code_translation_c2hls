"""Replay an already-completed multistep run into v2.0 trajectory records.

Use this to upgrade existing results (saved to results/ or
results_multistep/ as `<bench>_multistep_results.json`) without re-running
Vitis. The v1.0 jsonl artifact (e.g. requested_agentic_hwemu_rerun_*.jsonl)
is kept intact; this module produces a v2.0 jsonl alongside.

Usage:
    from dataset_pipeline.replay import replay_existing_results
    summary = replay_existing_results(
        results_dirs=["/home/luo00466/code_translation-c2hls/results_multistep"],
        output_jsonl="/tmp/c2hls_trajectory_v2.jsonl",
        run_meta=RunMeta(vitis_version="2023.2",
                         device="xcu280-fsvh2892-2L-e",
                         clock_ns=3.33),
    )
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .recorder import record_step_outcome
from .schema import RunMeta


def _load_results_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logging.warning("could not load %s: %s", path, exc)
        return None


def _suite_for_benchmark(benchmark: str) -> str:
    """Best-effort: the existing dataset_jsonl uses 'rodinia_hls' as the
    suite tag for the rodinia kernels. We use the same naming so v2 records
    join cleanly with the v1 reference set."""
    if benchmark in {
        "knn", "lud", "pathfinder", "hotspot", "kmeans", "lavaMD", "nw",
        "srad", "StreamCluster", "lc_dilate", "lc_gicov", "lc_mgvf",
        "cfd_flux", "cfd_step_factor",
    }:
        return "rodinia_hls"
    if benchmark in {
        "aes", "fft", "gemm_ncubed", "md_knn", "sort_merge", "spmv_crs",
        "stencil2D", "viterbi",
    }:
        return "ml4accel"
    return "unknown"


def _group_path_for(benchmark: str) -> List[str]:
    # Mirror the existing v1.0 records ([cfd, cfd_step_factor]; [pathfinder]
    # for top-level kernels).
    if benchmark.startswith("cfd_"):
        return ["cfd", benchmark]
    if benchmark.startswith("lc_"):
        return ["leukocyte", benchmark]
    return [benchmark]


def replay_existing_results(
    *,
    results_dirs: List[str],
    output_jsonl: str,
    run_meta: RunMeta,
    origin_version: str = "",
) -> Dict[str, Any]:
    """Walk results_dirs for `*_multistep_results.json` files, convert each
    step into v2.0 records, and write to output_jsonl. Returns a summary
    dict (kernels processed, records emitted)."""
    out_path = Path(output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    kernels: List[str] = []
    skipped: List[str] = []

    with out_path.open("w", encoding="utf-8") as out:
        for results_dir in results_dirs:
            base = Path(results_dir)
            if not base.is_dir():
                logging.warning("results dir missing: %s", base)
                continue
            for path in sorted(base.glob("*/*_multistep_results.json")):
                payload = _load_results_json(path)
                if payload is None:
                    skipped.append(str(path))
                    continue
                bench = payload.get("benchmark") or path.parent.name
                kernels.append(bench)

                # Synthesize parent_report progression: baseline → step 0 →
                # step 1 → … so step_effect classification gets real history.
                steps = payload.get("steps") or []
                parent_report = payload.get("baseline_report") or {}

                origin_v = origin_version or _detect_origin_version(payload)

                for idx, step in enumerate(steps):
                    new_report = step.get("report") or step.get("rejected_report") or {}
                    rtl_sim = step.get("rtl_sim") or step.get("hw_emu")
                    rtl_payload = None
                    if rtl_sim:
                        rtl_payload = {
                            "target": "vitis.hw_emu",
                            "device": run_meta.device,
                            "status": rtl_sim.get("status",
                                                  "pass" if rtl_sim.get("passed") else "fail"),
                            "kernel_runtime_cycles": rtl_sim.get("kernel_runtime_cycles"),
                            "kernel_runtime_us": rtl_sim.get("kernel_runtime_us"),
                            "kernel_clock_freq_mhz": rtl_sim.get("kernel_clock_freq_mhz"),
                            "error": rtl_sim.get("error"),
                        }
                    recs = record_step_outcome(
                        step_result=step,
                        suite=_suite_for_benchmark(bench),
                        group_path=_group_path_for(bench),
                        variant_index=idx,
                        variant_name=step.get("step_name", f"step_{idx}"),
                        run_meta=run_meta,
                        parent_report=parent_report,
                        origin="c2hls_orchestrator",
                        origin_version=origin_v,
                        multistep=True,
                        rtl_sim_payload=rtl_payload,
                    )
                    for rec in recs:
                        out.write(json.dumps(rec, separators=(",", ":")))
                        out.write("\n")
                        written += 1
                    if step.get("success"):
                        # Successful steps move the parent forward; no-op /
                        # regressed steps don't (the orchestrator reverts).
                        if new_report:
                            parent_report = new_report

    return {
        "kernels": kernels,
        "kernel_count": len(set(kernels)),
        "records_written": written,
        "skipped_files": skipped,
        "output": output_jsonl,
    }


def _detect_origin_version(payload: Dict[str, Any]) -> str:
    """Reach into the existing payload for a model id, falling back to
    blanks if not present."""
    run = payload.get("run") or {}
    model = run.get("origin_version") or run.get("model")
    if model:
        return model
    # Some saved payloads carry it under top-level keys.
    return payload.get("model", "")
