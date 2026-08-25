#!/usr/bin/env python3
"""Expand the C2HLS agentic RL experiment matrix into row-level files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = (
    REPO
    / "artifacts"
    / "experiment_matrix"
    / "c2hls_agentic_rl_experiment_matrix_20260707.json"
)


FRAMEWORK_SKILLS = {
    "direct_zero_shot": ["none"],
    "agentic_flash": ["none", "selective_positive", "curated_positive", "all_positive"],
    "agentic_multistep": ["none", "selective_positive", "curated_positive", "all_positive"],
}


LOCAL_PROVIDERS = {"local_vllm", "local_openai_compatible"}


def _is_local_provider(provider: str) -> bool:
    return provider in LOCAL_PROVIDERS


def _scope_rows(provider: str, training_state: str) -> list[tuple[str, str, bool]]:
    if not _is_local_provider(provider):
        return [("full", "full28", False)]
    if training_state == "base_no_adapter":
        return [("smoke", "smoke9", False), ("full", "full28", True)]
    return [("smoke", "smoke9", False), ("full", "full28", True)]


def _priority(framework: str, provider: str, training_state: str, phase: str) -> int:
    if framework == "reference":
        return 0
    if framework == "direct_zero_shot" and not _is_local_provider(provider):
        return 1
    if framework == "direct_zero_shot" and training_state == "base_no_adapter" and phase == "smoke":
        return 1
    if framework == "agentic_flash" and not _is_local_provider(provider):
        return 2
    if framework == "direct_zero_shot":
        return 2 if phase == "smoke" else 4
    if framework == "agentic_flash":
        return 2 if training_state == "base_no_adapter" and phase == "smoke" else 3
    if framework == "agentic_multistep":
        return 2 if not _is_local_provider(provider) else 4
    return 5


def _runner(framework: str, provider: str) -> str:
    if framework == "reference":
        return "run_hlsfactory_direct_reference.py / Vitis reference flow"
    if framework == "direct_zero_shot" and _is_local_provider(provider):
        return "scripts/run_vllm_corpus_eval.py + scripts/run_vllm_vitis_smoke.py + scripts/run_vllm_cosim_compare.py"
    if framework == "direct_zero_shot":
        return "direct C2HLS translation runner; verify current script before launch"
    return "run_agentic_sweep.py"


def _status(model_key: str, framework: str, skill_mode: str) -> str:
    if model_key != "anthropic_sonnet46":
        return "planned"
    if framework == "agentic_flash" and skill_mode in {"none", "selective_positive"}:
        return "historical_anchor_available"
    if framework == "agentic_flash" and skill_mode == "all_positive":
        return "historical_partial_27_rows"
    if framework == "agentic_multistep" and skill_mode in {"none", "selective_positive"}:
        return "historical_anchor_available"
    return "planned"


def expand(matrix: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        {
            "row_id": "R0__reference__hlsfactory_gold__none__full28",
            "cell_id": "R0",
            "framework": "reference",
            "model_key": "hlsfactory_gold",
            "provider": "reference",
            "model_id": "hlsfactory_gold",
            "training_state": "none",
            "skill_mode": "none",
            "phase": "full",
            "benchmark_scope": "full28",
            "requires_promotion": False,
            "eval_depth": "csim+csynth+cosim",
            "priority": 0,
            "runner": _runner("reference", "reference"),
            "status": "required_control",
        }
    ]
    models = matrix["models"]
    for model_key, meta in models.items():
        provider = meta["provider"]
        for training_state in meta["training_states"]:
            for framework, skill_modes in FRAMEWORK_SKILLS.items():
                for skill_mode in skill_modes:
                    for phase, scope, gated in _scope_rows(provider, training_state):
                        if not _is_local_provider(provider) and phase != "full":
                            continue
                        if not _is_local_provider(provider) and training_state != "base_api":
                            continue
                        rows.append(
                            {
                                "row_id": "__".join(
                                    [
                                        framework,
                                        model_key,
                                        training_state,
                                        skill_mode,
                                        scope,
                                    ]
                                ),
                                "cell_id": (
                                    "D_COMM"
                                    if framework == "direct_zero_shot" and not _is_local_provider(provider)
                                    else "D_LOCAL_BASE"
                                    if framework == "direct_zero_shot" and training_state == "base_no_adapter"
                                    else "D_LOCAL_SFT"
                                    if framework == "direct_zero_shot"
                                    else "F_COMM"
                                    if framework == "agentic_flash" and not _is_local_provider(provider)
                                    else "F_LOCAL_BASE"
                                    if framework == "agentic_flash" and training_state == "base_no_adapter"
                                    else "F_LOCAL_SFT"
                                    if framework == "agentic_flash"
                                    else "M_COMM"
                                    if not _is_local_provider(provider)
                                    else "M_LOCAL_BASE"
                                    if training_state == "base_no_adapter"
                                    else "M_LOCAL_SFT"
                                ),
                                "framework": framework,
                                "model_key": model_key,
                                "provider": provider,
                                "model_id": meta["model_id"],
                                "training_state": training_state,
                                "skill_mode": skill_mode,
                                "phase": phase,
                                "benchmark_scope": scope,
                                "requires_promotion": gated,
                                "eval_depth": "csim+csynth+cosim",
                                "priority": _priority(framework, provider, training_state, phase),
                                "runner": _runner(framework, provider),
                                "status": _status(model_key, framework, skill_mode),
                            }
                        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--out-jsonl", type=Path)
    parser.add_argument("--out-csv", type=Path)
    args = parser.parse_args()

    matrix = json.loads(args.matrix.read_text())
    rows = expand(matrix)
    out_jsonl = args.out_jsonl or args.matrix.with_suffix(".expanded.jsonl")
    out_csv = args.out_csv or args.matrix.with_suffix(".expanded.csv")
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with out_jsonl.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    fieldnames = list(rows[0].keys())
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps({"rows": len(rows), "jsonl": str(out_jsonl), "csv": str(out_csv)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
