#!/usr/bin/env python3
"""Build role-attributed SFT rows from audited C2HLS agent trajectories.

The HPCA evaluation profile records the exact messages sent for every model
call, the corresponding assistant response, the calling agent, and a
same-index candidate validation event.  This builder joins those records and
keeps only responses that passed CSim and CSynth while meeting timing and
resource constraints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_ROLES = ("translator", "synthesis", "orchestrator")
FIXED_VAL = {"hlsfactory_gramschmidt"}
FIXED_TEST = {
    "hlsfactory_durbin",
    "hlsfactory_floyd_warshall",
    "hlsfactory_gemm",
    "hlsfactory_trmm",
}
CODE_FENCE_RE = re.compile(r"```(?:cpp|c\+\+|cc|c)?\s*(.*?)```", re.DOTALL | re.I)
TMP_PATH_RE = re.compile(
    r"(?:/mnt/data/[^\s`'\"),\]}]+)?/tmp(?:_[^\s/`'\"),\]}]+)?/"
    r"[^\s`'\"),\]}]+"
)
HOME_PATH_RE = re.compile(r"/home/luo00466")


class SourceRejected(ValueError):
    """Raised when a source run cannot satisfy the attribution contract."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(detail or reason)
        self.reason = reason
        self.detail = detail


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise SourceRejected("document_not_object", str(path))
    return value


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_prompt_hash(messages: list[dict[str, str]]) -> str:
    payload = json.dumps(
        messages,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return _sha256_text(payload)


def _sanitize_text(text: str) -> str:
    return HOME_PATH_RE.sub("<HOME>", TMP_PATH_RE.sub("<HLS_TMP>", text))


def _normalize_message(message: dict[str, Any], *, sanitize: bool) -> dict[str, str]:
    role = str(message.get("role") or "").strip().lower()
    content = message.get("content", "")
    if not isinstance(content, str):
        content = json.dumps(content, sort_keys=True, ensure_ascii=False)
    if not role:
        raise SourceRejected("message_missing_role")
    return {
        "role": role,
        "content": _sanitize_text(content) if sanitize else content,
    }


def _contains_code(text: str) -> bool:
    if CODE_FENCE_RE.search(text):
        return True
    return "#include" in text and ("#pragma HLS" in text or 'extern "C"' in text)


def _split_for_benchmark(benchmark: str) -> str:
    if benchmark in FIXED_VAL:
        return "val"
    if benchmark in FIXED_TEST:
        return "test"
    return "train"


def _history_path(result_path: Path, benchmark: str) -> Path:
    return result_path.with_name(f"{benchmark}_history.json")


def _llm_events(result: dict[str, Any]) -> list[dict[str, Any]]:
    usage = result.get("llm_usage")
    if not isinstance(usage, dict):
        usage = (result.get("run") or {}).get("llm_usage")
    events = usage.get("events") if isinstance(usage, dict) else None
    if not isinstance(events, list):
        raise SourceRejected("missing_llm_events")
    return [event for event in events if isinstance(event, dict)]


def _candidate_events(result: dict[str, Any]) -> list[dict[str, Any]]:
    evaluations = result.get("synthesis_evaluations")
    if not isinstance(evaluations, dict):
        evaluations = (result.get("run") or {}).get("synthesis_evaluations")
    if not isinstance(evaluations, dict):
        raise SourceRejected("missing_candidate_evaluations")
    if evaluations.get("complete_candidate_event_stream") is not True:
        raise SourceRejected("incomplete_candidate_event_stream")
    events = evaluations.get("events")
    if not isinstance(events, list):
        raise SourceRejected("missing_candidate_events")
    return [event for event in events if isinstance(event, dict)]


def _group_call_messages(history: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    messages = history.get("messages")
    if not isinstance(messages, list):
        raise SourceRejected("missing_call_messages")
    grouped: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for message in messages:
        if not isinstance(message, dict):
            continue
        try:
            call_index = int(message["call_index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SourceRejected("invalid_call_index") from exc
        grouped[call_index].append(message)
    for call_messages in grouped.values():
        call_messages.sort(key=lambda item: int(item.get("message_index", 0)))
    return dict(grouped)


def _assistant_outputs(history: dict[str, Any]) -> list[str]:
    event_history = history.get("event_history")
    if not isinstance(event_history, list):
        raise SourceRejected("missing_event_history")
    outputs: list[str] = []
    for event in event_history:
        if not isinstance(event, dict) or event.get("role") != "assistant":
            continue
        content = event.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content, sort_keys=True, ensure_ascii=False)
        outputs.append(content)
    return outputs


def _candidate_passes(event: dict[str, Any]) -> tuple[bool, str]:
    checks = (
        ("correctness_status", event.get("correctness_status") == "passed"),
        ("synthesis_status", event.get("synthesis_status") == "passed"),
        ("timing_met", event.get("timing_met") is True),
        ("resource_fit", event.get("resource_fit") is True),
        ("failure_class", event.get("failure_class") in (None, "")),
    )
    for name, passed in checks:
        if not passed:
            return False, f"candidate_{name}"
    return True, ""


def _source_paths(results_root: Path) -> Iterable[Path]:
    if results_root.is_file():
        yield results_root
        return
    yield from sorted(results_root.rglob("*_multistep_results.json"))


def records_from_result(
    result_path: Path,
    *,
    allowed_roles: set[str],
) -> tuple[list[dict[str, Any]], Counter[str]]:
    rejected: Counter[str] = Counter()
    result = _load_json(result_path)
    benchmark = str(result.get("benchmark") or "").strip()
    if not benchmark:
        raise SourceRejected("missing_benchmark")
    history_path = _history_path(result_path, benchmark)
    if not history_path.is_file():
        raise SourceRejected("missing_history", str(history_path))
    history = _load_json(history_path)

    llm_events = _llm_events(result)
    candidate_events = _candidate_events(result)
    call_messages = _group_call_messages(history)
    assistant_outputs = _assistant_outputs(history)
    if len(llm_events) != len(assistant_outputs):
        raise SourceRejected(
            "llm_output_count_mismatch",
            f"{len(llm_events)} LLM events versus {len(assistant_outputs)} outputs",
        )

    candidates_by_index: dict[int, dict[str, Any]] = {}
    for candidate in candidate_events:
        try:
            index = int(candidate["candidate_evaluation_index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SourceRejected("invalid_candidate_index") from exc
        if index in candidates_by_index:
            raise SourceRejected("duplicate_candidate_index", str(index))
        candidates_by_index[index] = candidate

    run = result.get("run") if isinstance(result.get("run"), dict) else {}
    fingerprint = run.get("run_fingerprint") or result.get("run_fingerprint") or {}
    fingerprint_sha = fingerprint.get("sha256") if isinstance(fingerprint, dict) else None
    split = _split_for_benchmark(benchmark)
    records: list[dict[str, Any]] = []

    for ordinal, (llm_event, target) in enumerate(zip(llm_events, assistant_outputs)):
        role = str(llm_event.get("agent") or "").strip()
        if role not in allowed_roles:
            rejected[f"role_{role or 'missing'}"] += 1
            continue
        try:
            call_index = int(llm_event.get("candidate_evaluation_index"))
        except (TypeError, ValueError) as exc:
            rejected["missing_candidate_join_index"] += 1
            continue
        candidate = candidates_by_index.get(call_index)
        if candidate is None:
            rejected["candidate_event_missing"] += 1
            continue
        passed, reason = _candidate_passes(candidate)
        if not passed:
            rejected[reason] += 1
            continue
        if not target.strip():
            rejected["empty_target"] += 1
            continue
        if not _contains_code(target):
            rejected["target_without_code"] += 1
            continue

        raw_call_messages = call_messages.get(ordinal)
        if not raw_call_messages:
            rejected["call_messages_missing"] += 1
            continue
        prompt_messages = [
            _normalize_message(message, sanitize=False) for message in raw_call_messages
        ]
        expected_prompt_hash = str(llm_event.get("prompt_sha256") or "")
        observed_prompt_hash = _canonical_prompt_hash(prompt_messages)
        if not expected_prompt_hash or observed_prompt_hash != expected_prompt_hash:
            rejected["prompt_hash_mismatch"] += 1
            continue

        sanitized_prompt = [
            _normalize_message(message, sanitize=True) for message in raw_call_messages
        ]
        sanitized_target = _sanitize_text(target)
        messages = sanitized_prompt + [
            {"role": "assistant", "content": sanitized_target}
        ]
        target_hash = _sha256_text(sanitized_target)
        record_id = _sha256_text(
            "\0".join(
                [
                    str(result_path.resolve()),
                    benchmark,
                    role,
                    str(call_index),
                    target_hash,
                ]
            )
        )
        record = {
            "schema_version": "c2hls.role_attributed_sft.v1",
            "record_id": record_id,
            "messages": messages,
            "benchmark": benchmark,
            "suite": "hlsfactory",
            "split": split,
            "quality_label": "validated_positive",
            "model_teacher": llm_event.get("model"),
            "assistant_turn_index": ordinal,
            "target_agent_role": role,
            "source_history": str(history_path.resolve()),
            "source_result": str(result_path.resolve()),
            "metadata": {
                "target_agent_role": role,
                "agent_call_ordinal": ordinal,
                "candidate_evaluation_index": call_index,
                "attribution": "exact_call_transcript_and_candidate_index_join",
                "source_prompt_sha256": expected_prompt_hash,
                "sanitized_prompt_sha256": _canonical_prompt_hash(sanitized_prompt),
                "target_sha256": target_hash,
                "candidate_code_sha256": candidate.get("code_sha256"),
                "candidate_postprocessed_by_framework": True,
                "correctness_status": candidate.get("correctness_status"),
                "csim_passed": True,
                "synthesis_status": candidate.get("synthesis_status"),
                "csynth_passed": True,
                "timing_met": candidate.get("timing_met"),
                "resource_fit": candidate.get("resource_fit"),
                "latency_cycles": candidate.get("synthesized_latency_cycles"),
                "latency_source": candidate.get("latency_source"),
                "report_sha256": candidate.get("report_sha256"),
                "cosim_observed": False,
                "cosim_passed": None,
                "strategy": (fingerprint.get("payload") or {})
                .get("search", {})
                .get("strategy")
                if isinstance(fingerprint, dict)
                else None,
                "skill_mode": run.get("skill_mode"),
                "run_fingerprint_sha256": fingerprint_sha,
                "teacher_provider": llm_event.get("provider"),
                "teacher_model_revision": llm_event.get("model_revision"),
            },
        }
        records.append(record)
    return records, rejected


def build_dataset(
    results_roots: list[Path],
    output_dir: Path,
    *,
    allowed_roles: set[str],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records_by_split: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    rejected: Counter[str] = Counter()
    source_files: list[dict[str, Any]] = []
    seen: set[str] = set()

    result_paths: list[Path] = []
    for root in results_roots:
        result_paths.extend(_source_paths(root))
    for result_path in sorted(set(path.resolve() for path in result_paths)):
        try:
            records, local_rejected = records_from_result(
                result_path,
                allowed_roles=allowed_roles,
            )
        except (OSError, json.JSONDecodeError, SourceRejected) as exc:
            reason = exc.reason if isinstance(exc, SourceRejected) else "source_read_error"
            rejected[reason] += 1
            continue
        rejected.update(local_rejected)
        source_files.append(
            {
                "path": str(result_path),
                "bytes": result_path.stat().st_size,
                "sha256": _sha256_file(result_path),
            }
        )
        for record in records:
            dedupe_key = _sha256_text(
                json.dumps(
                    {
                        "benchmark": record["benchmark"],
                        "role": record["target_agent_role"],
                        "messages": record["messages"],
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                )
            )
            if dedupe_key in seen:
                rejected["exact_duplicate"] += 1
                continue
            seen.add(dedupe_key)
            records_by_split[record["split"]].append(record)

    files: dict[str, str] = {}
    counts_by_split: Counter[str] = Counter()
    counts_by_role: Counter[str] = Counter()
    counts_by_split_role: Counter[str] = Counter()
    benchmarks_by_split: dict[str, list[str]] = {}
    for split in ("train", "val", "test"):
        rows = sorted(
            records_by_split[split],
            key=lambda row: (
                row["benchmark"],
                row["target_agent_role"],
                row["record_id"],
            ),
        )
        path = output_dir / f"{split}.jsonl"
        with path.open("w") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
        files[split] = str(path)
        counts_by_split[split] = len(rows)
        benchmarks_by_split[split] = sorted({row["benchmark"] for row in rows})
        for row in rows:
            role = row["target_agent_role"]
            counts_by_role[role] += 1
            counts_by_split_role[f"{split}:{role}"] += 1

    split_sets = {name: set(values) for name, values in benchmarks_by_split.items()}
    overlap = sorted(
        (split_sets["train"] & split_sets["val"])
        | (split_sets["train"] & split_sets["test"])
        | (split_sets["val"] & split_sets["test"])
    )
    if overlap:
        raise RuntimeError(f"benchmark split leakage detected: {overlap}")

    manifest = {
        "schema_version": "c2hls.role_attributed_sft_manifest.v1",
        "output_dir": str(output_dir.resolve()),
        "results_roots": [str(path.resolve()) for path in results_roots],
        "allowed_roles": sorted(allowed_roles),
        "role_scope": {
            "trained": sorted(allowed_roles),
            "dormant_untrained": ["feedback", "quality_repair"],
            "claim": "all LLM-active roles in the reference-blind HPCA profile",
        },
        "quality_filter": {
            "correctness_status": "passed",
            "csim_required": True,
            "synthesis_status": "passed",
            "csynth_required": True,
            "timing_met": True,
            "resource_fit": True,
            "failure_class": None,
            "cosim_required": False,
            "postprocessed_validation": True,
        },
        "split_contract": {
            "val": sorted(FIXED_VAL),
            "test": sorted(FIXED_TEST),
            "train": "all remaining benchmarks",
            "benchmark_overlap": overlap,
        },
        "counts": {
            "total": sum(counts_by_split.values()),
            "by_split": dict(counts_by_split),
            "by_role": dict(counts_by_role),
            "by_split_role": dict(counts_by_split_role),
            "rejected": dict(rejected),
            "source_result_files": len(source_files),
        },
        "benchmarks_by_split": benchmarks_by_split,
        "files": files,
        "source_files": source_files,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    manifest["files"]["manifest"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        action="append",
        required=True,
        help="Root containing *_multistep_results.json files; repeatable.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--role",
        action="append",
        choices=sorted(DEFAULT_ROLES),
        help="Role to include; defaults to every LLM-active reference-blind role.",
    )
    args = parser.parse_args()
    manifest = build_dataset(
        [path.resolve() for path in args.results_root],
        args.output_dir.resolve(),
        allowed_roles=set(args.role or DEFAULT_ROLES),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
