#!/usr/bin/env python3
"""Run or replay a versioned C2HLS setup tournament."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import c2hls
from setup_router import (
    CORRECTED_VERSION,
    registry_by_id,
    resolve_policy_setups,
    select_tournament_winner,
)


@contextmanager
def _environment(updates: dict[str, str]) -> Iterator[None]:
    previous = {name: os.environ.get(name) for name in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _setup_environment(setup, args: argparse.Namespace) -> dict[str, str]:
    skillless = setup.skill_scope == "skillless"
    return {
        "C2HLS_STRATEGY": (
            "flash" if setup.strategy == "flash" else "dynamic"
        ),
        "C2HLS_SKILL_MODE": "skillless" if skillless else "default",
        "C2HLS_FORCE_SKILL_PROMPTS": "1",
        "C2HLS_SKILL_PROMPT_SCOPE": setup.skill_scope,
        "C2HLS_SKILL_PROMPT_MODE": setup.prompt_mode,
        "C2HLS_SKILL_LIBRARY_FROZEN": "1",
        "C2HLS_SKILL_LIBRARY_PATH": str(args.skill_library.resolve()),
        "C2HLS_SKILL_LIBRARY_PERSIST": "0",
        "C2HLS_SKILL_UPDATE_STATS": "0",
        "C2HLS_PHASE_B_SEED_MANIFEST": str(
            args.phase_b_manifest.resolve()
        ),
        "C2HLS_SKILL_USAGE_DECLARATION": "1",
        "C2HLS_CANDIDATES_PER_STEP": "1",
        "C2HLS_ATTEMPTS_PER_CANDIDATE": "1",
        "C2HLS_EXHAUSTIVE_CANDIDATE_ATTEMPTS": "0",
        "C2HLS_SKILL_EXHAUSTIVE_MAX_CANDIDATES": "5",
        "C2HLS_LLM_CANDIDATE_BUDGET": "5",
        "C2HLS_SYNTHESIS_EVAL_BUDGET": "5",
        "C2HLS_REFERENCE_BLIND": "1",
        "C2HLS_ORACLE_MODE": "0",
        "C2HLS_GT_COMPARISON_IN_CONTROL": "0",
        "C2HLS_REFERENCE_METRICS_IN_PROMPTS": "0",
        "C2HLS_REFERENCE_CODE_IN_PROMPTS": "0",
        "C2HLS_COSIM_REQUIRED": "0",
        "C2HLS_COSIM_SELECTED_ONLY": "0",
        "C2HLS_FORCE_SELECTED_COSIM": "0",
        "C2HLS_REFERENCE_COSIM": "0",
        "C2HLS_HW_EMU_FINAL": "0",
        "C2HLS_VITIS_VERSION": args.vitis_version,
        "C2HLS_FLOW_TARGET": "vitis",
        "C2HLS_PART": args.part,
        "C2HLS_CLOCK_NS": str(args.clock_ns),
    }


def _load_predictions(
    path: Path | None,
) -> tuple[list[str], dict[str, object]]:
    if path is None:
        return [], {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = (
        payload.get("ranked_setup_ids")
        if isinstance(payload, dict)
        else payload
    )
    if not isinstance(values, list):
        raise ValueError("prediction file must contain ranked_setup_ids")
    metadata = {}
    if isinstance(payload, dict):
        nested = payload.get("routing_metadata")
        if isinstance(nested, dict):
            metadata.update(nested)
        for key in (
            "recommended_candidate_budget",
            "candidate_budget",
            "committee_disagreement",
            "ood_score",
        ):
            if key in payload:
                metadata[key] = payload[key]
    return [str(value) for value in values], metadata


def _run_live(args: argparse.Namespace) -> dict:
    predictions, prediction_metadata = _load_predictions(args.predictions)
    requested_setup_ids = [
        item.strip()
        for item in str(getattr(args, "setup_ids", "") or "").split(",")
        if item.strip()
    ]
    if requested_setup_ids:
        registry = registry_by_id(args.registry_version)
        unknown = [
            setup_id
            for setup_id in requested_setup_ids
            if setup_id not in registry
        ]
        if unknown:
            raise ValueError(
                "unknown setup id(s): " + ", ".join(sorted(unknown))
            )
        setups = [registry[setup_id] for setup_id in requested_setup_ids]
    else:
        setups = resolve_policy_setups(
            policy=args.policy,
            predicted_setup_ids=predictions,
            version=args.registry_version,
            prediction_metadata=prediction_metadata,
        )
    if not setups:
        raise ValueError("tournament policy selected no setups")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates = []
    for index, setup in enumerate(setups):
        setup_dir = args.output_dir / setup.setup_id.replace(":", "__")
        setup_dir.mkdir(parents=True, exist_ok=True)
        with _environment(_setup_environment(setup, args)):
            result = c2hls.run_benchmark_multistep(
                str(args.benchmark_dir),
                output_dir=str(setup_dir),
                gpt_model=args.model,
                turns_limitation=args.turns,
            )
        result["candidate_index"] = index
        result["setup_id"] = setup.setup_id
        result["setup_fingerprint"] = setup.fingerprint
        result["setup"] = setup.to_record()
        result["result_path"] = str(
            setup_dir
            / f"{result.get('benchmark', args.benchmark_dir.name)}"
            "_multistep_results.json"
        )
        candidates.append(result)
        (setup_dir / "tournament_measurement.json").write_text(
            json.dumps(result, indent=2, default=str) + "\n",
            encoding="utf-8",
        )

    outcome = select_tournament_winner(candidates)
    outcome.update(
        {
            "policy": args.policy,
            "registry_version": args.registry_version,
            "benchmark_dir": str(args.benchmark_dir.resolve()),
            "model": args.model,
            "reference_blind": True,
            "requested_setup_ids": requested_setup_ids,
            "predicted_setup_ids": predictions,
            "prediction_metadata": prediction_metadata,
            "evaluated_setup_ids": [setup.setup_id for setup in setups],
        }
    )
    winner = outcome.get("winner")
    if isinstance(winner, dict):
        code = winner.get("hls_code") or winner.get("code")
        if code:
            (args.output_dir / "winner_kernel.cpp").write_text(
                str(code),
                encoding="utf-8",
            )
    return outcome


def _replay(args: argparse.Namespace) -> dict:
    candidates = []
    with args.measurements_jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                candidates.append(json.loads(line))
    return select_tournament_winner(candidates)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--measurements-jsonl", type=Path)
    parser.add_argument(
        "--policy",
        choices=(
            "exhaustive",
            "advisory",
            "learned_top_k",
            "adaptive_diverse_top_k",
        ),
        default="exhaustive",
    )
    parser.add_argument("--predictions", type=Path)
    parser.add_argument(
        "--setup-ids",
        default="",
        help=(
            "Optional comma-separated exact setup IDs. When set, these "
            "replace policy-based setup selection."
        ),
    )
    parser.add_argument("--registry-version", default=CORRECTED_VERSION)
    parser.add_argument("--phase-b-manifest", type=Path)
    parser.add_argument(
        "--skill-library",
        type=Path,
        default=REPO_ROOT / "skill_v2" / "skills.json",
    )
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--turns", type=int, default=3)
    parser.add_argument("--vitis-version", default="2023.2")
    parser.add_argument("--part", default="xcu280-fsvh2892-2L-e")
    parser.add_argument("--clock-ns", type=float, default=3.33)
    args = parser.parse_args()
    if args.measurements_jsonl is None:
        if args.benchmark_dir is None or args.phase_b_manifest is None:
            parser.error(
                "live mode requires --benchmark-dir and --phase-b-manifest"
            )
    return args


if __name__ == "__main__":
    arguments = parse_args()
    result = (
        _replay(arguments)
        if arguments.measurements_jsonl is not None
        else _run_live(arguments)
    )
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    destination = arguments.output_dir / "tournament_result.json"
    destination.write_text(
        json.dumps(result, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "success": result.get("success"),
                "winner": (
                    (result.get("winner") or {}).get("setup_id")
                    if isinstance(result.get("winner"), dict)
                    else None
                ),
                "output": str(destination),
            },
            sort_keys=True,
        )
    )
