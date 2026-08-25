#!/usr/bin/env python3
"""Prepare compact agentic C2HLS SFT splits.

Input is the corpus produced by build_agentic_sft_corpus.py.  This script
filters for high-confidence rows, optionally deduplicates repeated code targets,
and writes train/val/test JSONL files that keep OpenAI-style `messages` plus
metadata/provenance for later audit.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    REPO_ROOT / "artifacts" / "rl_corpus" / "agentic_sft_compact_positive.jsonl"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "rl_corpus" / "agentic_sft_v1"


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")
            count += 1
    return count


def _target_text(record: dict[str, Any]) -> str:
    messages = record.get("messages") or []
    if not messages:
        return ""
    return str(messages[-1].get("content", ""))


def _keep_record(
    record: dict[str, Any],
    *,
    qualities: set[str],
    suites: set[str],
    max_target_chars: int,
    max_prompt_chars: int,
) -> bool:
    if qualities and record.get("quality_label") not in qualities:
        return False
    if suites and record.get("suite") not in suites:
        return False
    messages = record.get("messages") or []
    if len(messages) < 2:
        return False
    target = _target_text(record)
    if not target:
        return False
    prompt_chars = sum(len(str(m.get("content", ""))) for m in messages[:-1])
    if max_prompt_chars and prompt_chars > max_prompt_chars:
        return False
    if max_target_chars and len(target) > max_target_chars:
        return False
    return True


def _project_record(record: dict[str, Any]) -> dict[str, Any]:
    metadata = record.get("metadata") or {}
    return {
        "messages": record["messages"],
        "benchmark": record.get("benchmark"),
        "suite": record.get("suite"),
        "split": record.get("split"),
        "quality_label": record.get("quality_label"),
        "source_history": record.get("source_history"),
        "source_result": record.get("source_result"),
        "model_teacher": record.get("model"),
        "assistant_turn_index": record.get("assistant_turn_index"),
        "metadata": {
            "code_sha256": metadata.get("code_sha256"),
            "code_chars": metadata.get("code_chars"),
            "prompt_chars": sum(
                len(str(m.get("content", ""))) for m in record["messages"][:-1]
            ),
            "target_chars": len(_target_text(record)),
            "synth_passed": metadata.get("synth_passed"),
            "csim_passed": metadata.get("csim_passed"),
            "step": metadata.get("step"),
            "latency_cycles": metadata.get("latency_cycles"),
            "latency_ns": metadata.get("latency_ns"),
            "bram": metadata.get("bram"),
            "dsp": metadata.get("dsp"),
            "ff": metadata.get("ff"),
            "lut": metadata.get("lut"),
            "injected_skill_count": len(metadata.get("injected_skill_ids") or []),
        },
    }


def build_dataset(
    input_path: Path,
    output_dir: Path,
    *,
    qualities: set[str],
    suites: set[str],
    dedupe: bool,
    max_target_chars: int,
    max_prompt_chars: int,
) -> dict[str, Any]:
    seen: set[tuple[str, str, str]] = set()
    splits: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    skipped = Counter()

    for raw in _read_jsonl(input_path):
        if not _keep_record(
            raw,
            qualities=qualities,
            suites=suites,
            max_target_chars=max_target_chars,
            max_prompt_chars=max_prompt_chars,
        ):
            skipped["filter"] += 1
            continue
        projected = _project_record(raw)
        split = projected.get("split") or "train"
        if split not in splits:
            skipped["unknown_split"] += 1
            continue
        if dedupe:
            key = (
                str(projected.get("benchmark")),
                str(projected["metadata"].get("code_sha256")),
                split,
            )
            if key in seen:
                skipped["dedupe"] += 1
                continue
            seen.add(key)
        splits[split].append(projected)

    counts = {}
    for split, records in splits.items():
        counts[split] = _write_jsonl(output_dir / f"{split}.jsonl", records)

    manifest = {
        "schema_version": "agentic_sft_v1_manifest",
        "input": str(input_path),
        "output_dir": str(output_dir),
        "qualities": sorted(qualities),
        "suites": sorted(suites),
        "dedupe": dedupe,
        "max_target_chars": max_target_chars,
        "max_prompt_chars": max_prompt_chars,
        "counts": counts,
        "skipped": dict(skipped),
        "files": {split: str(output_dir / f"{split}.jsonl") for split in splits},
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--quality",
        action="append",
        default=["validated_positive"],
        help="Quality label to keep; repeatable. Default: validated_positive.",
    )
    parser.add_argument(
        "--suite",
        action="append",
        default=["hlsfactory"],
        help="Suite to keep; repeatable. Default: hlsfactory.",
    )
    parser.add_argument("--no-dedupe", action="store_true")
    parser.add_argument("--max-target-chars", type=int, default=80000)
    parser.add_argument("--max-prompt-chars", type=int, default=90000)
    args = parser.parse_args()

    manifest = build_dataset(
        Path(args.input),
        Path(args.output_dir),
        qualities=set(args.quality or []),
        suites=set(args.suite or []),
        dedupe=not args.no_dedupe,
        max_target_chars=args.max_target_chars,
        max_prompt_chars=args.max_prompt_chars,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
