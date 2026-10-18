#!/usr/bin/env python3
"""Filter agentic chat records to directly attributable optimizer turns.

The historical corpus contains translation, optimization, and repair turns in
the same conversations.  Saved step metadata alone is not sufficient because
older exports associated step results with assistant ordinals.  This filter
keeps an example only when the user message immediately before the target is
an exact optimization-step prompt and that parsed step agrees with the saved
result metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = REPO_ROOT / "artifacts" / "rl_corpus" / "agentic_trl_chat_v1"
DEFAULT_OUTPUT_DIR = Path(
    "/mnt/data2/luo00466/c2hls_sft/source/orchestrator_optimization_v1"
)
SPLITS = ("train", "val", "test")
OPTIMIZATION_STEPS = (
    "flash",
    "tiling",
    "pipeline",
    "unroll",
    "doublebuffer",
    "coalescing",
)
STEP_PROMPT_RE = re.compile(
    r"^\[Step: (?P<step>" + "|".join(OPTIMIZATION_STEPS) + r")\](?:\s|$)"
)


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            yield value


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            count += 1
    return count


def optimization_step_for_record(record: dict[str, Any]) -> tuple[str | None, str]:
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return None, "invalid_messages"
    target = messages[-1]
    prompt = messages[-2]
    if not isinstance(target, dict) or target.get("role") != "assistant":
        return None, "target_not_assistant"
    if not isinstance(prompt, dict) or prompt.get("role") != "user":
        return None, "preceding_message_not_user"
    content = prompt.get("content")
    if not isinstance(content, str):
        return None, "prompt_not_text"
    match = STEP_PROMPT_RE.match(content)
    if not match:
        return None, "not_direct_optimization_prompt"
    step = match.group("step")
    metadata = record.get("metadata") or {}
    if metadata.get("step") != step:
        return None, "step_metadata_mismatch"
    return step, "kept"


def filter_split(
    records: Iterable[dict[str, Any]],
    *,
    required_quality: str = "validated_positive",
) -> tuple[list[dict[str, Any]], Counter[str]]:
    kept: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    seen: set[tuple[str, str, str, str]] = set()

    for record in records:
        if record.get("quality_label") != required_quality:
            rejected["quality"] += 1
            continue
        metadata = record.get("metadata") or {}
        if metadata.get("synth_passed") is not True:
            rejected["synth_not_passed"] += 1
            continue
        if metadata.get("csim_passed") is not True:
            rejected["csim_not_passed"] += 1
            continue
        step, reason = optimization_step_for_record(record)
        if step is None:
            rejected[reason] += 1
            continue

        messages = record["messages"]
        prompt_text = str(messages[-2].get("content") or "")
        target_text = str(messages[-1].get("content") or "")
        if not target_text.strip():
            rejected["empty_target"] += 1
            continue
        key = (
            str(record.get("benchmark") or ""),
            step,
            hashlib.sha256(prompt_text.encode()).hexdigest(),
            hashlib.sha256(target_text.encode()).hexdigest(),
        )
        if key in seen:
            rejected["exact_duplicate"] += 1
            continue
        seen.add(key)

        projected = dict(record)
        projected["target_agent_role"] = "orchestrator"
        projected_metadata = dict(metadata)
        projected_metadata["target_agent_role"] = "orchestrator"
        projected_metadata["attribution_method"] = (
            "direct_step_prompt_and_matching_result_metadata"
        )
        projected["metadata"] = projected_metadata
        kept.append(projected)

    return kept, rejected


def build_role_dataset(
    input_dir: Path,
    output_dir: Path,
    *,
    required_quality: str = "validated_positive",
) -> dict[str, Any]:
    split_records: dict[str, list[dict[str, Any]]] = {}
    rejected_by_split: dict[str, dict[str, int]] = {}
    benchmark_sets: dict[str, set[str]] = {}
    step_counts: Counter[str] = Counter()
    skill_counts: Counter[str] = Counter()

    for split in SPLITS:
        input_path = input_dir / f"{split}.jsonl"
        if not input_path.is_file():
            raise FileNotFoundError(f"missing input split: {input_path}")
        records, rejected = filter_split(
            _read_jsonl(input_path), required_quality=required_quality
        )
        if not records:
            raise RuntimeError(f"role filter produced no {split} records")
        split_records[split] = records
        rejected_by_split[split] = dict(sorted(rejected.items()))
        benchmark_sets[split] = {
            str(record.get("benchmark") or "") for record in records
        }
        for record in records:
            metadata = record.get("metadata") or {}
            step_counts[str(metadata.get("step"))] += 1
            skill_state = (
                "skills_injected"
                if metadata.get("injected_skill_ids")
                else "no_skills_injected"
            )
            skill_counts[skill_state] += 1

    overlaps: dict[str, list[str]] = {}
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            overlap = sorted(benchmark_sets[left] & benchmark_sets[right])
            if overlap:
                overlaps[f"{left}_{right}"] = overlap
    if overlaps:
        raise RuntimeError(f"benchmark leakage across splits: {overlaps}")

    output_dir.mkdir(parents=True, exist_ok=True)
    counts = {
        split: _write_jsonl(output_dir / f"{split}.jsonl", records)
        for split, records in split_records.items()
    }
    manifest = {
        "schema_version": "c2hls.role_filtered_sft.v1",
        "target_agent_role": "orchestrator",
        "attribution_method": "direct_step_prompt_and_matching_result_metadata",
        "required_quality": required_quality,
        "required_gates": {"synth_passed": True, "csim_passed": True},
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "counts": counts,
        "benchmarks": {
            split: sorted(values) for split, values in benchmark_sets.items()
        },
        "benchmark_overlap": overlaps,
        "counts_by_step": dict(sorted(step_counts.items())),
        "counts_by_skill_state": dict(sorted(skill_counts.items())),
        "rejected_by_split": rejected_by_split,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--quality", default="validated_positive")
    args = parser.parse_args()
    manifest = build_role_dataset(
        args.input_dir,
        args.output_dir,
        required_quality=args.quality,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
