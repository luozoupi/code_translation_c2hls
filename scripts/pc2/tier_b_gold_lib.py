"""Tier B ready corpus helpers for gold-gate (csynth + csim) validation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
TIER_B_READY_ROOT = REPO / "related_work/benchmarks/HLSFactory_benchmarks/tier_B_ready"
DEFAULT_SYNTH_TIMEOUT_S = 3600
DEFAULT_CSIM_TIMEOUT_S = 600


def list_tier_b_benches(
    root: Path = TIER_B_READY_ROOT,
    *,
    dataset: str | None = None,
) -> list[str]:
    names: list[str] = []
    if not root.is_dir():
        return names
    for meta_path in sorted(root.glob("*/metadata.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if dataset and str(meta.get("dataset") or "") != dataset:
            continue
        bench_dir = meta_path.parent
        if not (bench_dir / "plain.cpp").is_file():
            continue
        if not (bench_dir / "hls_baseline.cpp").is_file():
            continue
        names.append(str(meta.get("benchmark") or bench_dir.name))
    return names


def resolve_tier_b_benches(
    requested: list[str],
    root: Path = TIER_B_READY_ROOT,
) -> list[tuple[str, Path]]:
    if not root.is_dir():
        raise FileNotFoundError(f"tier_B_ready root missing: {root}")
    available: dict[str, Path] = {}
    for meta_path in sorted(root.glob("*/metadata.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        name = str(meta.get("benchmark") or meta_path.parent.name)
        available[name] = meta_path.parent
    missing = [name for name in requested if name not in available]
    if missing:
        raise ValueError(f"unknown tier_B_ready benchmark(s): {missing}")
    return [(name, available[name]) for name in requested]


def apply_bench_synth_timeout_from_meta(meta: dict[str, Any]) -> int:
    timeout = meta.get("synth_timeout_s")
    if timeout is not None:
        os.environ["C2HLS_SYNTH_TIMEOUT"] = str(int(timeout))
        return int(timeout)
    os.environ.setdefault("C2HLS_SYNTH_TIMEOUT", str(DEFAULT_SYNTH_TIMEOUT_S))
    return int(os.environ["C2HLS_SYNTH_TIMEOUT"])


def configure_tier_b_gold_env() -> None:
    import sys

    from c2hls_paths import apply_runtime_defaults
    from c2hls_temp import configure_temp_env

    scripts_root = Path(__file__).resolve().parents[1]
    if str(scripts_root) not in sys.path:
        sys.path.insert(0, str(scripts_root))

    apply_runtime_defaults(profile="sweep")
    configure_temp_env(create=True)

    os.environ.setdefault("C2HLS_RECORD_FLOW", "1")
    os.environ.setdefault("C2HLS_RUN_COSIM", "0")
    os.environ.setdefault("C2HLS_REFERENCE_COSIM", "0")
    os.environ.setdefault("C2HLS_PART", "xcu280-fsvh2892-2L-e")
    os.environ.setdefault("C2HLS_CLOCK_NS", "3.33")
    os.environ.setdefault("C2HLS_SYNTH_TIMEOUT", str(DEFAULT_SYNTH_TIMEOUT_S))
    os.environ.setdefault("C2HLS_CSIM_TIMEOUT", str(DEFAULT_CSIM_TIMEOUT_S))
