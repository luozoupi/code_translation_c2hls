#!/usr/bin/env python3
"""Export agentic C2HLS histories as TRL-ready chat-template JSONL.

The input records come from build_agentic_sft_corpus.py.  This exporter keeps
OpenAI-style `messages` for TRL conversational SFT and also writes a rendered
Gemma-style `text` field for simple dataset_text_field training.  Vitis labels
include csynth, csim, and exact cosim labels when they can be recovered from
the companion result JSON or schema JSONL backfills.
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
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "rl_corpus" / "agentic_trl_chat_v1"
DEFAULT_COSIM_JSONLS = [
    REPO_ROOT / "artifacts" / "hlsfactory_multistep_sonnet46_skill_on_website_revstyle_combined_20260615.jsonl",
    REPO_ROOT / "artifacts" / "hlsfactory_multistep_best_cosim10800_final_20260614.referencekeyfix.schema.jsonl",
    REPO_ROOT / "artifacts" / "hlsfactory_multistep_sonnet46_no_skills_20260615_cosim_backfill_20260615.schema.jsonl",
]


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


def _as_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        low = value.strip().lower()
        if low in {"pass", "passed", "success", "true", "ok"}:
            return True
        if low in {"fail", "failed", "false", "error", "timeout", "timed_out"}:
            return False
    return None


def _as_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(str(value).replace(",", "")))
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _norm_path(value: Any) -> str | None:
    if not value:
        return None
    text = str(value)
    try:
        return str(Path(text).resolve())
    except OSError:
        return text


def _bench_key(value: Any) -> str:
    raw = str(value or "")
    raw = raw.removeprefix("hlsfactory_")
    raw = raw.replace("-", "_").replace("/", "_")
    return f"hlsfactory_{raw}" if raw else ""


def _step_key(value: Any) -> str:
    return str(value or "").strip().lower()


def _cosim_payload(raw: Any, *, source: str) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        passed = _as_bool(raw)
        if passed is None:
            return None
        return {
            "cosim_observed": True,
            "cosim_passed": passed,
            "cosim_status": "pass" if passed else "fail",
            "cosim_source": source,
        }
    status = raw.get("status")
    passed = _as_bool(raw.get("passed"))
    success = _as_bool(raw.get("success"))
    if passed is None:
        passed = _as_bool(status)
    if passed is None and success is not None and str(status or "").lower() not in {"timeout"}:
        passed = success
    observed = bool(raw.get("ran", True)) or passed is not None or status is not None
    if not observed:
        return None
    return {
        "cosim_observed": True,
        "cosim_passed": passed,
        "cosim_status": status or ("pass" if passed else "fail" if passed is False else None),
        "cosim_supported": raw.get("supported"),
        "cosim_ran": raw.get("ran"),
        "cosim_cycles": _as_int(raw.get("kernel_runtime_cycles")),
        "cosim_runtime_us": _as_float(raw.get("kernel_runtime_us")),
        "cosim_clock_freq_mhz": _as_float(raw.get("kernel_clock_freq_mhz")),
        "cosim_error": raw.get("error"),
        "cosim_source": source,
    }


def _collect_step_results(result: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("generated_step_history", "optimization_history", "turn_history", "steps"):
        value = result.get(key)
        if isinstance(value, list) and value:
            return [item for item in value if isinstance(item, dict)]
    return []


def _load_json(path: str | None, cache: dict[str, Any]) -> Any:
    norm = _norm_path(path)
    if not norm:
        return None
    if norm not in cache:
        try:
            cache[norm] = json.loads(Path(norm).read_text())
        except (OSError, json.JSONDecodeError):
            cache[norm] = None
    return cache[norm]


def _source_result_cosim(record: dict[str, Any], cache: dict[str, Any]) -> dict[str, Any] | None:
    result = _load_json(record.get("source_result"), cache)
    if not isinstance(result, dict):
        return None
    metadata = record.get("metadata") or {}
    assistant_index = record.get("assistant_turn_index")
    step_name = _step_key(metadata.get("step"))
    steps = _collect_step_results(result)
    candidates: list[dict[str, Any]] = []
    if isinstance(assistant_index, int) and 0 <= assistant_index < len(steps):
        candidates.append(steps[assistant_index])
    if step_name:
        candidates.extend(
            step for step in steps
            if _step_key(step.get("step_name") or step.get("step") or step.get("phase")) == step_name
        )
    if step_name == "baseline" and isinstance(result.get("baseline_cosim"), dict):
        payload = _cosim_payload(result.get("baseline_cosim"), source="source_result.baseline_cosim")
        if payload:
            return payload
    for step in candidates:
        payload = _cosim_payload(step.get("cosim"), source="source_result.step")
        if payload:
            return payload
    payload = _cosim_payload(result.get("cosim"), source="source_result.final")
    return payload


def _schema_cosim_index(paths: list[Path]) -> dict[tuple[str, str], dict[str, Any]]:
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for path in paths:
        if not path.exists():
            continue
        for record in _read_jsonl(path):
            if record.get("report_type") != "rtl_sim":
                continue
            problem = record.get("problem") or {}
            impl = record.get("implementation") or {}
            origin_meta = impl.get("origin_meta") or {}
            source_result = _norm_path(origin_meta.get("source_result_json"))
            step = _step_key(origin_meta.get("step") or origin_meta.get("selected_step"))
            if not source_result or not step:
                continue
            payload = _cosim_payload(record.get("rtl_sim"), source=f"schema_jsonl:{path.name}")
            if payload:
                payload["cosim_schema_jsonl"] = str(path)
                payload["cosim_benchmark"] = _bench_key("/".join(problem.get("group_path") or []))
                index[(source_result, step)] = payload
    return index


def _schema_cosim(record: dict[str, Any], index: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any] | None:
    source_result = _norm_path(record.get("source_result"))
    step = _step_key((record.get("metadata") or {}).get("step"))
    if not source_result or not step:
        return None
    return index.get((source_result, step))


def _target_text(record: dict[str, Any]) -> str:
    messages = record.get("messages") or []
    if not messages:
        return ""
    return str(messages[-1].get("content", ""))


def _prompt_chars(messages: list[dict[str, Any]]) -> int:
    return sum(len(str(message.get("content", ""))) for message in messages[:-1])


def _keep_record(
    record: dict[str, Any],
    *,
    qualities: set[str],
    suites: set[str],
    max_prompt_chars: int,
    max_target_chars: int,
) -> bool:
    if qualities and record.get("quality_label") not in qualities:
        return False
    if suites and record.get("suite") not in suites:
        return False
    messages = record.get("messages") or []
    if len(messages) < 2 or not _target_text(record):
        return False
    if max_prompt_chars and _prompt_chars(messages) > max_prompt_chars:
        return False
    if max_target_chars and len(_target_text(record)) > max_target_chars:
        return False
    return True


def _render_gemma(messages: list[dict[str, Any]], *, add_generation_prompt: bool = False) -> str:
    turns: list[tuple[str, str]] = []
    user_buffer: list[str] = []
    for message in messages:
        role = message.get("role", "user")
        content = str(message.get("content", ""))
        if role == "assistant":
            if user_buffer:
                turns.append(("user", "\n\n".join(user_buffer)))
                user_buffer = []
            turns.append(("model", content))
        else:
            prefix = "System" if role == "system" else "Tool" if role == "tool" else "User"
            user_buffer.append(f"{prefix}:\n{content}")
    if user_buffer:
        turns.append(("user", "\n\n".join(user_buffer)))
    rendered = "".join(f"<start_of_turn>{role}\n{content}<end_of_turn>\n" for role, content in turns)
    if add_generation_prompt:
        rendered += "<start_of_turn>model\n"
    return rendered


def _reward_label(metadata: dict[str, Any]) -> dict[str, Any]:
    synth = _as_bool(metadata.get("synth_passed"))
    csim = _as_bool(metadata.get("csim_passed"))
    cosim_observed = bool(metadata.get("cosim_observed"))
    cosim = _as_bool(metadata.get("cosim_passed"))
    if synth is False:
        tier = 0
        reward = -1.0
    elif synth is True and csim is not True:
        tier = 1
        reward = 0.25
    elif csim is True and cosim_observed and cosim is not True:
        tier = 2
        reward = 0.55
    elif csim is True and cosim is True:
        tier = 3
        reward = 1.0
    elif csim is True:
        tier = 2
        reward = 0.75
    else:
        tier = 0
        reward = 0.0
    return {
        "correctness_tier": tier,
        "reward_scalar": reward,
        "reward_policy": "synth<csim<cosim; speed/resources kept as separate metrics",
    }


def _project_record(record: dict[str, Any], cosim: dict[str, Any] | None) -> dict[str, Any]:
    messages = record["messages"]
    metadata = dict(record.get("metadata") or {})
    metadata.update(
        {
            "prompt_chars": _prompt_chars(messages),
            "target_chars": len(_target_text(record)),
            "cosim_observed": False,
            "cosim_passed": None,
            "cosim_status": None,
            "cosim_source": None,
        }
    )
    if cosim:
        metadata.update(cosim)
    metadata.update(_reward_label(metadata))
    return {
        "messages": messages,
        "text": _render_gemma(messages),
        "prompt": _render_gemma(messages[:-1], add_generation_prompt=True),
        "completion": _target_text(record) + "<end_of_turn>\n",
        "benchmark": record.get("benchmark"),
        "suite": record.get("suite"),
        "split": record.get("split"),
        "quality_label": record.get("quality_label"),
        "model_teacher": record.get("model"),
        "assistant_turn_index": record.get("assistant_turn_index"),
        "source_history": record.get("source_history"),
        "source_result": record.get("source_result"),
        "metadata": metadata,
    }


def build_export(
    input_path: Path,
    output_dir: Path,
    *,
    qualities: set[str],
    suites: set[str],
    dedupe: bool,
    max_prompt_chars: int,
    max_target_chars: int,
    cosim_jsonls: list[Path],
) -> dict[str, Any]:
    cosim_index = _schema_cosim_index(cosim_jsonls)
    result_cache: dict[str, Any] = {}
    seen: set[tuple[str, str, str]] = set()
    splits: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    skipped = Counter()
    cosim_sources = Counter()

    for raw in _read_jsonl(input_path):
        if not _keep_record(
            raw,
            qualities=qualities,
            suites=suites,
            max_prompt_chars=max_prompt_chars,
            max_target_chars=max_target_chars,
        ):
            skipped["filter"] += 1
            continue
        split = raw.get("split") or "train"
        if split not in splits:
            skipped["unknown_split"] += 1
            continue
        metadata = raw.get("metadata") or {}
        if dedupe:
            key = (str(raw.get("benchmark")), str(metadata.get("code_sha256")), split)
            if key in seen:
                skipped["dedupe"] += 1
                continue
            seen.add(key)
        cosim = _source_result_cosim(raw, result_cache) or _schema_cosim(raw, cosim_index)
        projected = _project_record(raw, cosim)
        cosim_sources[projected["metadata"].get("cosim_source") or "missing"] += 1
        splits[split].append(projected)

    all_records = [record for records in splits.values() for record in records]
    counts = {split: _write_jsonl(output_dir / f"{split}.jsonl", records) for split, records in splits.items()}
    cosim_observed_records = [
        record for record in all_records
        if record["metadata"].get("cosim_observed")
    ]
    cosim_passed_records = [
        record for record in cosim_observed_records
        if record["metadata"].get("cosim_passed") is True
    ]
    counts["cosim_observed"] = _write_jsonl(
        output_dir / "cosim_observed.jsonl",
        cosim_observed_records,
    )
    counts["cosim_passed"] = _write_jsonl(
        output_dir / "cosim_passed.jsonl",
        cosim_passed_records,
    )
    label_counts = Counter(
        (
            record["metadata"].get("synth_passed"),
            record["metadata"].get("csim_passed"),
            record["metadata"].get("cosim_observed"),
            record["metadata"].get("cosim_passed"),
            record["metadata"].get("correctness_tier"),
        )
        for record in all_records
    )
    manifest = {
        "schema_version": "agentic_trl_chat_v1_manifest",
        "input": str(input_path),
        "output_dir": str(output_dir),
        "files": {
            **{split: str(output_dir / f"{split}.jsonl") for split in splits},
            "cosim_observed": str(output_dir / "cosim_observed.jsonl"),
            "cosim_passed": str(output_dir / "cosim_passed.jsonl"),
        },
        "format": {
            "messages": "OpenAI-style chat messages for TRL conversational SFT",
            "text": "Gemma-style rendered chat text with <start_of_turn>/<end_of_turn>",
            "prompt_completion": "prompt is rendered without target; completion is assistant target plus <end_of_turn>",
        },
        "filters": {
            "qualities": sorted(qualities),
            "suites": sorted(suites),
            "dedupe": dedupe,
            "max_prompt_chars": max_prompt_chars,
            "max_target_chars": max_target_chars,
        },
        "counts": counts,
        "skipped": dict(skipped),
        "cosim_sources": dict(cosim_sources),
        "label_counts": {"|".join(map(str, key)): value for key, value in sorted(label_counts.items())},
        "cosim_jsonls": [str(path) for path in cosim_jsonls],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--quality", action="append", default=["validated_positive"])
    parser.add_argument("--suite", action="append", default=["hlsfactory"])
    parser.add_argument("--no-dedupe", action="store_true")
    parser.add_argument("--max-prompt-chars", type=int, default=90000)
    parser.add_argument("--max-target-chars", type=int, default=80000)
    parser.add_argument("--cosim-jsonl", type=Path, action="append", default=DEFAULT_COSIM_JSONLS)
    args = parser.parse_args()
    manifest = build_export(
        args.input,
        args.output_dir,
        qualities=set(args.quality or []),
        suites=set(args.suite or []),
        dedupe=not args.no_dedupe,
        max_prompt_chars=args.max_prompt_chars,
        max_target_chars=args.max_target_chars,
        cosim_jsonls=args.cosim_jsonl,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
