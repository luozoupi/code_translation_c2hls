#!/usr/bin/env python3
"""Export normalized, model-tokenized C2HLS trajectories for TRL SFT.

The source chat histories contain framework events represented as system
messages and can contain several assistant turns before the supervised target.
Those histories are useful training context, but they do not satisfy the chat
grammar expected by Qwen and Gemma tokenizers.  This exporter converts every
record to one canonical system/user/assistant exchange, renders it with the
target model's own chat template, and writes exact completion-only token masks.

The Parquet output is intentionally pretokenized.  SFTTrainer must be invoked
with completion_only_loss=True so prompt and framework-event tokens are never
used as labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = REPO_ROOT / "artifacts" / "rl_corpus" / "agentic_trl_chat_v1"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "artifacts" / "rl_corpus" / "agentic_model_sft_v2"
DEFAULT_QWEN_CHECKPOINT = Path(
    "/mnt/data/vllm_models/hub/models--Qwen--Qwen3.6-27B/"
    "snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
)
DEFAULT_GEMMA_CHECKPOINT = Path(
    "/mnt/data/vllm_models/hub/models--google--gemma-4-31B-it/"
    "snapshots/3548789868c5356dbf307c98e6f609007b82b3eb"
)
TRUNCATION_MARKER = "\n\n<TRUNCATED_TRAJECTORY_CONTEXT>\n\n"
TARGET_SENTINEL = "C2HLS_TARGET_BOUNDARY_7f6b11d997e84d0e"


@dataclass(frozen=True)
class ModelProfile:
    name: str
    checkpoint: Path
    template_kwargs: dict[str, Any]


@dataclass(frozen=True)
class CanonicalTrajectory:
    prompt_messages: list[dict[str, str]]
    target: str
    raw_roles: list[str]
    leading_system_count: int
    had_original_user: bool


@dataclass(frozen=True)
class EncodedTrajectory:
    input_ids: list[int]
    completion_mask: list[int]
    prompt_tokens: int
    completion_tokens: int
    original_tokens: int
    original_prompt_tokens: int
    prompt_tokens_dropped: int
    context_truncated: bool
    rendered_text: str


MODEL_PROFILES = {
    "qwen3.6-27b": ModelProfile(
        name="qwen3.6-27b",
        checkpoint=DEFAULT_QWEN_CHECKPOINT,
        template_kwargs={"enable_thinking": False},
    ),
    "gemma-4-31b-it": ModelProfile(
        name="gemma-4-31b-it",
        checkpoint=DEFAULT_GEMMA_CHECKPOINT,
        template_kwargs={"enable_thinking": False},
    ),
}


class RecordRejected(ValueError):
    """Raised when a source row cannot produce a valid SFT example."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(detail or reason)
        self.reason = reason
        self.detail = detail


def _read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open() as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RecordRejected("invalid_json", f"line {line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise RecordRejected("invalid_record", f"line {line_number} is not an object")
            yield line_number, value


def _content_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def _role_heading(role: str, ordinal: int) -> str:
    labels = {
        "assistant": "PRIOR ASSISTANT RESPONSE",
        "developer": "DEVELOPER EVENT",
        "system": "SYSTEM EVENT",
        "tool": "TOOL RESULT",
        "user": "USER REQUEST",
    }
    return f"[{labels.get(role, role.upper())} {ordinal}]"


def canonicalize_messages(messages: Any) -> CanonicalTrajectory:
    if not isinstance(messages, list) or len(messages) < 2:
        raise RecordRejected("invalid_messages", "expected at least one prompt message and one target")

    normalized: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            raise RecordRejected("invalid_message", "message is not an object")
        role = str(message.get("role") or "").strip().lower()
        content = _content_text(message.get("content")).strip()
        if not role:
            raise RecordRejected("missing_role")
        normalized.append({"role": role, "content": content})

    target_message = normalized[-1]
    if target_message["role"] != "assistant":
        raise RecordRejected("target_not_assistant")
    if not target_message["content"]:
        raise RecordRejected("empty_target")
    if TARGET_SENTINEL in target_message["content"]:
        raise RecordRejected("target_contains_boundary_sentinel")

    context = normalized[:-1]
    leading_system: list[str] = []
    context_start = 0
    for message in context:
        if message["role"] not in {"system", "developer"}:
            break
        if message["content"]:
            leading_system.append(message["content"])
        context_start += 1

    role_ordinals: Counter[str] = Counter()
    transcript_blocks: list[str] = []
    for message in context[context_start:]:
        role = message["role"]
        role_ordinals[role] += 1
        transcript_blocks.append(
            f"{_role_heading(role, role_ordinals[role])}\n{message['content']}"
        )

    transcript = "\n\n".join(transcript_blocks)
    if not transcript:
        transcript = "[NO NON-SYSTEM TRAJECTORY EVENTS RECORDED]"
    user_content = (
        "The following is the chronological context from a C2HLS agent run. "
        "Treat labeled prior responses and framework events as context.\n\n"
        f"{transcript}\n\n"
        "Continue the trajectory with the next assistant response."
    )

    prompt_messages: list[dict[str, str]] = []
    if leading_system:
        prompt_messages.append({"role": "system", "content": "\n\n".join(leading_system)})
    prompt_messages.append({"role": "user", "content": user_content})

    return CanonicalTrajectory(
        prompt_messages=prompt_messages,
        target=target_message["content"],
        raw_roles=[message["role"] for message in normalized],
        leading_system_count=context_start,
        had_original_user=any(message["role"] == "user" for message in context),
    )


def _render_text(tokenizer: Any, messages: list[dict[str, str]], template_kwargs: dict[str, Any]) -> str:
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        **template_kwargs,
    )
    if not isinstance(rendered, str):
        raise RecordRejected("chat_template_did_not_render_text")
    return rendered


def _flatten_tokenizer_value(value: Any) -> list[Any]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list):
        raise RecordRejected("unexpected_tokenizer_output", type(value).__name__)
    return value


def _tokenize_with_boundary(tokenizer: Any, text: str, boundary: int) -> tuple[list[int], int]:
    try:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        input_ids = [int(value) for value in _flatten_tokenizer_value(encoded["input_ids"])]
        offsets = _flatten_tokenizer_value(encoded["offset_mapping"])
        if len(input_ids) != len(offsets):
            raise RecordRejected("offset_length_mismatch")
        completion_start = next(
            (
                index
                for index, offset in enumerate(offsets)
                if isinstance(offset, (list, tuple))
                and len(offset) == 2
                and int(offset[1]) > boundary
            ),
            None,
        )
        if completion_start is None:
            raise RecordRejected("completion_boundary_not_tokenized")
        return input_ids, completion_start
    except (NotImplementedError, TypeError, ValueError, KeyError):
        full_ids = [int(value) for value in tokenizer.encode(text, add_special_tokens=False)]
        prefix_ids = [int(value) for value in tokenizer.encode(text[:boundary], add_special_tokens=False)]
        common = 0
        for left, right in zip(full_ids, prefix_ids):
            if left != right:
                break
            common += 1
        if common < max(0, len(prefix_ids) - 1):
            raise RecordRejected(
                "unstable_completion_boundary",
                f"only {common}/{len(prefix_ids)} prompt tokens match",
            )
        return full_ids, common


def render_and_encode(
    tokenizer: Any,
    trajectory: CanonicalTrajectory,
    *,
    template_kwargs: dict[str, Any],
    max_length: int,
    min_prompt_tokens: int,
    head_fraction: float,
) -> EncodedTrajectory:
    sentinel_messages = trajectory.prompt_messages + [
        {"role": "assistant", "content": TARGET_SENTINEL}
    ]
    sentinel_text = _render_text(tokenizer, sentinel_messages, template_kwargs)
    if sentinel_text.count(TARGET_SENTINEL) != 1:
        raise RecordRejected("chat_template_boundary_ambiguous")
    prompt_text, completion_suffix = sentinel_text.split(TARGET_SENTINEL)

    full_messages = trajectory.prompt_messages + [
        {"role": "assistant", "content": trajectory.target}
    ]
    rendered_text = _render_text(tokenizer, full_messages, template_kwargs)
    expected_text = prompt_text + trajectory.target + completion_suffix
    if rendered_text != expected_text:
        raise RecordRejected(
            "target_render_mismatch",
            "the chat template transformed target content",
        )

    original_ids, completion_start = _tokenize_with_boundary(
        tokenizer,
        rendered_text,
        len(prompt_text),
    )
    if completion_start <= 0 or completion_start >= len(original_ids):
        raise RecordRejected("invalid_completion_token_boundary")

    original_prompt_ids = original_ids[:completion_start]
    completion_ids = original_ids[completion_start:]
    if len(completion_ids) + min_prompt_tokens > max_length:
        raise RecordRejected(
            "completion_exceeds_budget",
            f"completion={len(completion_ids)}, max={max_length}, min_prompt={min_prompt_tokens}",
        )

    context_truncated = len(original_ids) > max_length
    prompt_tokens_dropped = 0
    if context_truncated:
        marker_ids = [
            int(value)
            for value in tokenizer.encode(TRUNCATION_MARKER, add_special_tokens=False)
        ]
        prompt_budget = max_length - len(completion_ids)
        retained_budget = prompt_budget - len(marker_ids)
        if retained_budget < min_prompt_tokens:
            raise RecordRejected("insufficient_prompt_budget_after_marker")
        head_tokens = min(retained_budget, max(1, math.floor(retained_budget * head_fraction)))
        tail_tokens = retained_budget - head_tokens
        retained_prompt = original_prompt_ids[:head_tokens]
        if tail_tokens:
            retained_prompt += marker_ids + original_prompt_ids[-tail_tokens:]
        else:
            retained_prompt += marker_ids
        prompt_tokens_dropped = len(original_prompt_ids) - head_tokens - tail_tokens
        input_ids = retained_prompt + completion_ids
        prompt_tokens = len(retained_prompt)
    else:
        input_ids = original_ids
        prompt_tokens = completion_start

    if len(input_ids) > max_length:
        raise RecordRejected("max_length_enforcement_failed")
    completion_mask = [0] * prompt_tokens + [1] * len(completion_ids)
    if len(completion_mask) != len(input_ids) or not any(completion_mask):
        raise RecordRejected("invalid_completion_mask")

    return EncodedTrajectory(
        input_ids=input_ids,
        completion_mask=completion_mask,
        prompt_tokens=prompt_tokens,
        completion_tokens=len(completion_ids),
        original_tokens=len(original_ids),
        original_prompt_tokens=len(original_prompt_ids),
        prompt_tokens_dropped=prompt_tokens_dropped,
        context_truncated=context_truncated,
        rendered_text=rendered_text,
    )


def _record_id(profile: str, record: dict[str, Any], line_number: int) -> str:
    metadata = record.get("metadata") or {}
    payload = "\0".join(
        [
            profile,
            str(record.get("source_history") or ""),
            str(record.get("assistant_turn_index") or ""),
            str(metadata.get("code_sha256") or ""),
            str(line_number),
        ]
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def project_record(
    tokenizer: Any,
    profile: ModelProfile,
    record: dict[str, Any],
    *,
    line_number: int,
    max_length: int,
    min_prompt_tokens: int,
    head_fraction: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    trajectory = canonicalize_messages(record.get("messages"))
    encoded = render_and_encode(
        tokenizer,
        trajectory,
        template_kwargs=profile.template_kwargs,
        max_length=max_length,
        min_prompt_tokens=min_prompt_tokens,
        head_fraction=head_fraction,
    )
    metadata = dict(record.get("metadata") or {})
    record_id = _record_id(profile.name, record, line_number)
    output = {
        "record_id": record_id,
        "input_ids": encoded.input_ids,
        "completion_mask": encoded.completion_mask,
        "benchmark": record.get("benchmark"),
        "suite": record.get("suite"),
        "split": record.get("split"),
        "quality_label": record.get("quality_label"),
        "model_teacher": record.get("model_teacher"),
        "assistant_turn_index": record.get("assistant_turn_index"),
        "target_agent_role": record.get("target_agent_role")
        or metadata.get("target_agent_role"),
        "source_history": record.get("source_history"),
        "source_result": record.get("source_result"),
        "source_line": line_number,
        "token_count": len(encoded.input_ids),
        "prompt_token_count": encoded.prompt_tokens,
        "completion_token_count": encoded.completion_tokens,
        "original_token_count": encoded.original_tokens,
        "original_prompt_token_count": encoded.original_prompt_tokens,
        "prompt_tokens_dropped": encoded.prompt_tokens_dropped,
        "context_truncated": encoded.context_truncated,
        "raw_message_count": len(trajectory.raw_roles),
        "raw_roles_json": json.dumps(trajectory.raw_roles, separators=(",", ":")),
        "leading_system_count": trajectory.leading_system_count,
        "had_original_user": trajectory.had_original_user,
        "target_sha256": hashlib.sha256(trajectory.target.encode()).hexdigest(),
        "metadata_json": json.dumps(metadata, sort_keys=True, separators=(",", ":")),
    }
    sample = {
        "record_id": record_id,
        "benchmark": record.get("benchmark"),
        "split": record.get("split"),
        "raw_roles": trajectory.raw_roles,
        "context_truncated": encoded.context_truncated,
        "original_token_count": encoded.original_tokens,
        "token_count": len(encoded.input_ids),
        "prompt_token_count": encoded.prompt_tokens,
        "completion_token_count": encoded.completion_tokens,
        "rendered_text": tokenizer.decode(encoded.input_ids, skip_special_tokens=False),
    }
    return output, sample


def _parquet_schema(pa: Any) -> Any:
    return pa.schema(
        [
            ("record_id", pa.string()),
            ("input_ids", pa.list_(pa.int32())),
            ("completion_mask", pa.list_(pa.uint8())),
            ("benchmark", pa.string()),
            ("suite", pa.string()),
            ("split", pa.string()),
            ("quality_label", pa.string()),
            ("model_teacher", pa.string()),
            ("assistant_turn_index", pa.int32()),
            ("target_agent_role", pa.string()),
            ("source_history", pa.string()),
            ("source_result", pa.string()),
            ("source_line", pa.int32()),
            ("token_count", pa.int32()),
            ("prompt_token_count", pa.int32()),
            ("completion_token_count", pa.int32()),
            ("original_token_count", pa.int32()),
            ("original_prompt_token_count", pa.int32()),
            ("prompt_tokens_dropped", pa.int32()),
            ("context_truncated", pa.bool_()),
            ("raw_message_count", pa.int32()),
            ("raw_roles_json", pa.string()),
            ("leading_system_count", pa.int32()),
            ("had_original_user", pa.bool_()),
            ("target_sha256", pa.string()),
            ("metadata_json", pa.string()),
        ]
    )


def _percentiles(values: list[int]) -> dict[str, int | None]:
    if not values:
        return {key: None for key in ("min", "p50", "p90", "p95", "p99", "max")}
    ordered = sorted(values)

    def pick(q: float) -> int:
        return ordered[min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1)]

    return {
        "min": ordered[0],
        "p50": pick(0.50),
        "p90": pick(0.90),
        "p95": pick(0.95),
        "p99": pick(0.99),
        "max": ordered[-1],
    }


def _write_readme(output_dir: Path, manifest: dict[str, Any]) -> None:
    files = manifest["files"]
    text = f"""# Model-aware C2HLS SFT export

Profile: `{manifest['model_profile']}`  
Checkpoint: `{manifest['checkpoint']}`  
Maximum sequence length: `{manifest['max_length']}`

The Parquet files are already tokenized with the checkpoint's own chat
template. Each row contains `input_ids` and a same-length `completion_mask`.
The mask is zero for system/user/framework context and one for the supervised
assistant target and its closing turn token.

```python
from datasets import load_dataset
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

dataset = load_dataset("parquet", data_files={{
    "train": r"{files.get('train', '')}",
    "validation": r"{files.get('val', '')}",
}})
tokenizer = AutoTokenizer.from_pretrained(
    r"{manifest['checkpoint']}", local_files_only=True
)

args = SFTConfig(
    output_dir="/mnt/data2/luo00466/c2hls_sft/checkpoints/example",
    max_length={manifest['max_length']},
    completion_only_loss=True,
    assistant_only_loss=False,
    packing=False,
)

# Instantiate the conditional-generation model explicitly, then pass it here.
trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    processing_class=tokenizer,
)
```

Do not enable `assistant_only_loss`; the official templates do not expose the
generation mask expected by that TRL option. Completion-only supervision is
already encoded in `completion_mask`.
"""
    (output_dir / "README.md").write_text(text)


def build_export(
    input_dir: Path,
    output_dir: Path,
    *,
    profile: ModelProfile,
    tokenizer: Any,
    splits: list[str],
    max_length: int,
    min_prompt_tokens: int,
    head_fraction: float,
    batch_size: int,
    sample_count: int,
    limit_per_split: int,
) -> dict[str, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required for the model-aware export") from exc

    if not 0.0 < head_fraction < 1.0:
        raise ValueError("head_fraction must be between 0 and 1")
    if max_length <= min_prompt_tokens:
        raise ValueError("max_length must be greater than min_prompt_tokens")

    output_dir.mkdir(parents=True, exist_ok=True)
    schema = _parquet_schema(pa)
    files: dict[str, str] = {}
    counts: Counter[str] = Counter()
    skipped: Counter[str] = Counter()
    role_patterns: Counter[tuple[str, ...]] = Counter()
    token_lengths: list[int] = []
    original_lengths: list[int] = []
    completion_lengths: list[int] = []
    sample_rows: list[dict[str, Any]] = []
    rejection_rows: list[dict[str, Any]] = []

    for split in splits:
        input_path = input_dir / f"{split}.jsonl"
        if not input_path.exists():
            raise FileNotFoundError(input_path)
        output_path = output_dir / f"{split}.parquet"
        temp_path = output_path.with_suffix(".parquet.tmp")
        writer = pq.ParquetWriter(temp_path, schema, compression="zstd")
        batch: list[dict[str, Any]] = []
        split_count = 0
        try:
            for line_number, record in _read_jsonl(input_path):
                if limit_per_split and split_count >= limit_per_split:
                    break
                try:
                    projected, sample = project_record(
                        tokenizer,
                        profile,
                        record,
                        line_number=line_number,
                        max_length=max_length,
                        min_prompt_tokens=min_prompt_tokens,
                        head_fraction=head_fraction,
                    )
                except RecordRejected as exc:
                    skipped[exc.reason] += 1
                    if len(rejection_rows) < 1000:
                        rejection_rows.append(
                            {
                                "split": split,
                                "source_line": line_number,
                                "benchmark": record.get("benchmark"),
                                "reason": exc.reason,
                                "detail": exc.detail,
                            }
                        )
                    continue

                batch.append(projected)
                split_count += 1
                counts[split] += 1
                counts["context_truncated"] += int(projected["context_truncated"])
                counts["without_original_user"] += int(not projected["had_original_user"])
                metadata = json.loads(projected["metadata_json"])
                counts["cosim_observed"] += int(bool(metadata.get("cosim_observed")))
                counts["cosim_passed"] += int(metadata.get("cosim_passed") is True)
                role_patterns[tuple(json.loads(projected["raw_roles_json"]))] += 1
                token_lengths.append(projected["token_count"])
                original_lengths.append(projected["original_token_count"])
                completion_lengths.append(projected["completion_token_count"])
                if len([row for row in sample_rows if row["split"] == split]) < sample_count:
                    sample_rows.append(sample)

                if len(batch) >= batch_size:
                    writer.write_table(pa.Table.from_pylist(batch, schema=schema))
                    batch.clear()
            if batch:
                writer.write_table(pa.Table.from_pylist(batch, schema=schema))
            if split_count == 0:
                writer.write_table(pa.Table.from_pylist([], schema=schema))
        finally:
            writer.close()
        temp_path.replace(output_path)
        files[split] = str(output_path)

    samples_path = output_dir / "samples.jsonl"
    with samples_path.open("w") as f:
        for sample in sample_rows:
            f.write(json.dumps(sample, sort_keys=True) + "\n")
    files["samples"] = str(samples_path)

    rejected_path = output_dir / "rejected.jsonl"
    with rejected_path.open("w") as f:
        for rejection in rejection_rows:
            f.write(json.dumps(rejection, sort_keys=True) + "\n")
    files["rejected"] = str(rejected_path)

    config_path = profile.checkpoint / "config.json"
    try:
        model_config = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError):
        model_config = {}
    chat_template = str(getattr(tokenizer, "chat_template", "") or "")
    manifest = {
        "schema_version": "agentic_model_sft.v2_manifest",
        "model_profile": profile.name,
        "checkpoint": str(profile.checkpoint),
        "model_type": model_config.get("model_type"),
        "tokenizer_class": type(tokenizer).__name__,
        "chat_template_sha256": hashlib.sha256(chat_template.encode()).hexdigest(),
        "template_kwargs": profile.template_kwargs,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "files": files,
        "splits": splits,
        "max_length": max_length,
        "min_prompt_tokens": min_prompt_tokens,
        "head_fraction_when_truncated": head_fraction,
        "truncation_marker": TRUNCATION_MARKER.strip(),
        "counts": dict(counts),
        "skipped": dict(skipped),
        "lengths": {
            "exported_tokens": _percentiles(token_lengths),
            "original_tokens": _percentiles(original_lengths),
            "completion_tokens": _percentiles(completion_lengths),
        },
        "top_raw_role_patterns": [
            {"roles": list(roles), "count": count}
            for roles, count in role_patterns.most_common(20)
        ],
        "trl_contract": {
            "format": "pretokenized_input_ids_with_completion_mask",
            "completion_only_loss": True,
            "assistant_only_loss": False,
            "packing": False,
            "processing_class": "tokenizer_not_multimodal_processor",
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    files["manifest"] = str(manifest_path)
    _write_readme(output_dir, manifest)
    files["readme"] = str(output_dir / "README.md")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def _load_tokenizer(checkpoint: Path) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers is required for the model-aware export") from exc
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        local_files_only=True,
        trust_remote_code=False,
    )
    if not getattr(tokenizer, "chat_template", None):
        raise RuntimeError(f"checkpoint has no chat template: {checkpoint}")
    return tokenizer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--profile", choices=sorted(MODEL_PROFILES), required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--split", action="append", choices=["train", "val", "test"])
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--min-prompt-tokens", type=int, default=256)
    parser.add_argument("--head-fraction", type=float, default=0.40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sample-count", type=int, default=2)
    parser.add_argument("--limit-per-split", type=int, default=0)
    args = parser.parse_args()

    base_profile = MODEL_PROFILES[args.profile]
    profile = ModelProfile(
        name=base_profile.name,
        checkpoint=(args.checkpoint or base_profile.checkpoint).resolve(),
        template_kwargs=base_profile.template_kwargs,
    )
    output_dir = args.output_dir or (
        DEFAULT_OUTPUT_ROOT / f"{profile.name}_max{args.max_length}"
    )
    tokenizer = _load_tokenizer(profile.checkpoint)
    manifest = build_export(
        args.input_dir.resolve(),
        output_dir.resolve(),
        profile=profile,
        tokenizer=tokenizer,
        splits=args.split or ["train", "val", "test"],
        max_length=args.max_length,
        min_prompt_tokens=args.min_prompt_tokens,
        head_fraction=args.head_fraction,
        batch_size=args.batch_size,
        sample_count=args.sample_count,
        limit_per_split=args.limit_per_split,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
