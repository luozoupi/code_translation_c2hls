"""Csynth latency speedup (baseline vs final) for Fir batch_parallel dashboard."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts" / "explorer"))

from batch_parallel.dispatch import cell_dir, model_cell_tag, setup_tag_for_campaign
from benchmark_cosim_baseline import load_benchmark_cosim_baseline
from metrics import (
    bench_cosim_metrics_from_multistep_doc,
    bench_csynth_latency_from_multistep,
    bench_run_issues_from_multistep_doc,
    bench_speedup_from_multistep,
    geomean,
    geomean_cosim_speedup_from_benches,
    mean_latency_from_benches,
)

_COSIM_BASELINE_MAP: dict[str, int] | None = None


def _cosim_baseline_map() -> dict[str, int]:
    global _COSIM_BASELINE_MAP
    if _COSIM_BASELINE_MAP is None:
        _COSIM_BASELINE_MAP = load_benchmark_cosim_baseline()
    return _COSIM_BASELINE_MAP


def multistep_results_path(
    campaign_root: Path,
    *,
    bench: str,
    model_id: str,
    setup_tag: str,
) -> Path:
    tag = model_cell_tag(model_id)
    return cell_dir(campaign_root, bench, tag, setup_tag) / f"{bench}_multistep_results.json"


def compute_speedup_summary(
    campaign_root: Path,
    *,
    benches: list[str],
    model_id: str,
    setup_tag: str,
) -> dict[str, Any]:
    per_bench: dict[str, dict[str, float | None]] = {}
    per_bench_latency: dict[str, dict[str, int | None]] = {}
    per_bench_cosim: dict[str, dict[str, Any]] = {}
    per_bench_issues: dict[str, list[str]] = {}
    buckets: dict[str, list[float]] = {"best": [], "avg": [], "worst": []}
    latency_benches: dict[str, dict[str, Any]] = {}
    cosim_benches: dict[str, dict[str, Any]] = {}
    baseline_map = _cosim_baseline_map()

    for bench in benches:
        path = multistep_results_path(
            campaign_root, bench=bench, model_id=model_id, setup_tag=setup_tag
        )
        speedup = bench_speedup_from_multistep(path)
        if speedup is not None:
            per_bench[bench] = speedup
            for kind in ("best", "avg", "worst"):
                value = speedup.get(kind)
                if value is not None and value > 0:
                    buckets[kind].append(value)
        latency = bench_csynth_latency_from_multistep(path)
        if latency is not None:
            per_bench_latency[bench] = latency
            latency_benches[bench] = {"status": "ok", "latency": latency}
        cosim_metrics = None
        if path.is_file():
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                doc = None
            if isinstance(doc, dict):
                short = bench.removeprefix("hlsfactory_")
                cosim_metrics = bench_cosim_metrics_from_multistep_doc(
                    doc,
                    baseline_map=baseline_map,
                    bench_short_name=short,
                )
                issues = bench_run_issues_from_multistep_doc(doc)
                if issues:
                    per_bench_issues[bench] = issues
        if cosim_metrics:
            per_bench_cosim[bench] = cosim_metrics
            cosim_benches[bench] = {"status": "ok", "cosim": cosim_metrics}

    return {
        "n": len(per_bench),
        "best_geomean": geomean(buckets["best"]),
        "avg_geomean": geomean(buckets["avg"]),
        "worst_geomean": geomean(buckets["worst"]),
        "latency_mean": mean_latency_from_benches(latency_benches),
        "cosim_speedup_geomean": geomean_cosim_speedup_from_benches(cosim_benches),
        "per_bench": per_bench,
        "per_bench_latency": per_bench_latency,
        "per_bench_cosim": per_bench_cosim,
        "per_bench_issues": per_bench_issues,
    }
