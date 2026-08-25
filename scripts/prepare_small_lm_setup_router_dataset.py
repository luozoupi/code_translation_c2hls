#!/usr/bin/env python3
"""Build leakage-safe pairwise SFT data for a small setup-routing LM."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_setup_router import _load, _sha256
from scripts.train_strengthened_setup_router import (
    _first_wins,
    _ranking_outcomes,
)


SCHEMA_VERSION = "c2hls.small-lm-setup-router-pairs.v1"
PHASE_B_FEATURES = (
    "phase_b_latency_cycles",
    "phase_b_interval",
    "phase_b_estimated_clock_period_ns",
    "phase_b_requested_clock_period_ns",
    "phase_b_slack_ns",
    "phase_b_bram",
    "phase_b_dsp",
    "phase_b_ff",
    "phase_b_lut",
    "phase_b_uram",
    "phase_b_bottleneck_count",
    "phase_b_pipeline_bottlenecks",
    "phase_b_memory_bottlenecks",
    "phase_b_dataflow_bottlenecks",
    "phase_b_recurrence_bottlenecks",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def _source_code(problem: str, benchmark_root: Path) -> tuple[str, Path]:
    benchmark_dir = benchmark_root / f"hlsfactory_{problem}"
    for filename in ("plain.cpp", "hls_baseline.cpp"):
        path = benchmark_dir / filename
        if path.is_file():
            return path.read_text(encoding="utf-8"), path
    raise FileNotFoundError(f"source code not found for {problem}")


def _setup_description(record: dict[str, Any]) -> dict[str, Any]:
    setup = record["setup"]
    return {
        "setup_id": setup["setup_id"],
        "strategy": setup["strategy"],
        "skill_scope": setup["skill_scope"],
        "prompt_mode": setup["prompt_mode"],
        "router_version": setup["router_version"],
        "candidate_policy": setup["candidate_policy"],
    }


def _prompt(
    *,
    source: str,
    phase_b: dict[str, Any],
    first: dict[str, Any],
    second: dict[str, Any],
) -> str:
    return "\n".join(
        [
            "You route an HLS optimization workflow.",
            "Choose which setup is more likely to produce a valid kernel "
            "with lower Vitis synthesis latency.",
            "Use only the source and frozen Phase-B evidence below.",
            "Do not assume reference cycles or candidate outcomes.",
            "Reply with exactly A or B.",
            "",
            "SOURCE_C:",
            source,
            "",
            f"PHASE_B: {_canonical_json(phase_b)}",
            f"SETUP_A: {_canonical_json(_setup_description(first))}",
            f"SETUP_B: {_canonical_json(_setup_description(second))}",
            "ANSWER:",
        ]
    )


def build(args: argparse.Namespace) -> dict[str, Any]:
    records = _ranking_outcomes(_load(args.corpus))
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["problem"])].append(record)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_manifest = {}
    skipped_pairs: dict[str, int] = defaultdict(int)
    for problem, outcomes in sorted(grouped.items()):
        split_values = {str(record["split"]) for record in outcomes}
        if len(split_values) != 1:
            raise ValueError(f"mixed lineage split for {problem}")
        split = next(iter(split_values))
        source, source_path = _source_code(problem, args.benchmark_root)
        source_manifest[problem] = {
            "path": str(source_path.resolve()),
            "sha256": _sha256(source_path),
        }
        mandatory = next(
            record
            for record in outcomes
            if record["setup"]["setup_id"].endswith(
                ":multistep:skillless"
            )
        )
        phase_b = {
            name: mandatory["features"].get(name)
            for name in PHASE_B_FEATURES
        }
        ordered = sorted(
            outcomes,
            key=lambda record: record["setup"]["setup_id"],
        )
        for first_index, first in enumerate(ordered):
            for second in ordered[first_index + 1 :]:
                first_wins = _first_wins(first, second)
                if first_wins is None:
                    skipped_pairs[split] += 1
                    continue
                winner = "A" if first_wins else "B"
                for orientation, (
                    setup_a,
                    setup_b,
                    answer,
                ) in enumerate(
                    (
                        (first, second, winner),
                        (
                            second,
                            first,
                            "B" if winner == "A" else "A",
                        ),
                    )
                ):
                    identity = {
                        "problem": problem,
                        "split": split,
                        "setup_a": setup_a["setup"]["setup_id"],
                        "setup_b": setup_b["setup"]["setup_id"],
                        "orientation": orientation,
                    }
                    output_records[split].append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "record_id": hashlib.sha256(
                                _canonical_json(identity).encode("utf-8")
                            ).hexdigest(),
                            **identity,
                            "benchmark_lineage": setup_a[
                                "benchmark_lineage"
                            ],
                            "prompt": _prompt(
                                source=source,
                                phase_b=phase_b,
                                first=setup_a,
                                second=setup_b,
                            ),
                            "completion": answer,
                            "label_source": (
                                "validity_first_then_lower_final_cycles"
                            ),
                        }
                    )

    artifact_hashes = {}
    for split in ("train", "validation", "test"):
        path = args.output_dir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for record in output_records[split]:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        artifact_hashes[path.name] = {
            "sha256": _sha256(path),
            "records": len(output_records[split]),
        }

    split_lineages = {
        split: sorted(
            {
                record["benchmark_lineage"]
                for record in output_records[split]
            }
        )
        for split in ("train", "validation", "test")
    }
    if (
        set(split_lineages["train"]) & set(split_lineages["validation"])
        or set(split_lineages["train"]) & set(split_lineages["test"])
        or set(split_lineages["validation"]) & set(split_lineages["test"])
    ):
        raise ValueError("lineage leakage in small-LM router corpus")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "corpus": {
            "path": str(args.corpus.resolve()),
            "sha256": _sha256(args.corpus),
        },
        "benchmark_root": str(args.benchmark_root.resolve()),
        "artifacts": artifact_hashes,
        "split_lineages": split_lineages,
        "source_manifest": source_manifest,
        "skipped_unordered_pairs": dict(skipped_pairs),
        "methodology": {
            "pair_orientations": 2,
            "reference_metrics_in_prompt": False,
            "candidate_outcomes_in_prompt": False,
            "canonical_phase_b_setup": "multistep skillless",
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    manifest = build(parse_args())
    print(
        json.dumps(
            {
                "artifacts": manifest["artifacts"],
                "lineage_counts": {
                    split: len(lineages)
                    for split, lineages in manifest[
                        "split_lineages"
                    ].items()
                },
                "skipped_unordered_pairs": manifest[
                    "skipped_unordered_pairs"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
