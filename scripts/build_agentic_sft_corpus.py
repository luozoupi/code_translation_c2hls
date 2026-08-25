#!/usr/bin/env python3
"""Build SFT-ready records from saved C2HLS agent trajectories.

This complements export_rl_corpus.py, which emits leak-free ground-truth
translation rows. Here we mine actual agent runs:

  - results_sweeps/**/<bench>_history.json from the agentic framework
  - optional external result histories, e.g. code_translation_c2hls/results

Each assistant code turn becomes one OpenAI-chat-style record containing only
the context available before that assistant response plus the assistant target.
Synthesis/csim labels are attached as metadata so trainers can filter positives
or build repair/preference datasets later.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_SWEEPS = REPO_ROOT / "results_sweeps"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "rl_corpus" / "agentic_sft_all.jsonl"
DEFAULT_POSITIVE_OUTPUT = (
    REPO_ROOT / "artifacts" / "rl_corpus" / "agentic_sft_positive.jsonl"
)
DEFAULT_COMPACT_OUTPUT = (
    REPO_ROOT / "artifacts" / "rl_corpus" / "agentic_sft_compact_all.jsonl"
)
DEFAULT_COMPACT_POSITIVE_OUTPUT = (
    REPO_ROOT / "artifacts" / "rl_corpus" / "agentic_sft_compact_positive.jsonl"
)

FIXED_VAL = {"StreamCluster", "viterbi"}
FIXED_TEST = {"nw", "spmv_crs"}

CODE_FENCE_RE = re.compile(r"```(?:cpp|c\\+\\+|cc|c)?\\s*(.*?)```", re.DOTALL | re.I)
TMP_PATH_RE = re.compile(r"/mnt/data/[^\\s`'\"),\\]}]+/tmp/[^\\s`'\"),\\]}]+")
HOME_PATH_RE = re.compile(r"/home/luo00466")


@dataclass
class HistoryDoc:
    path: Path
    messages: list[dict[str, Any]]
    model: Optional[str]
    usage: dict[str, Any]


def _json_load(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _stable_split(benchmark: str) -> str:
    if benchmark in FIXED_VAL:
        return "val"
    if benchmark in FIXED_TEST:
        return "test"
    bucket = int(hashlib.sha256(benchmark.encode()).hexdigest()[:8], 16) % 10
    if bucket == 8:
        return "val"
    if bucket == 9:
        return "test"
    return "train"


def _sanitize_text(text: str) -> str:
    text = TMP_PATH_RE.sub("<HLS_TMP>", text)
    text = HOME_PATH_RE.sub("<HOME>", text)
    return text


def _sanitize_message(message: dict[str, Any]) -> dict[str, str]:
    role = str(message.get("role", "user"))
    content = message.get("content", "")
    if not isinstance(content, str):
        content = json.dumps(content, sort_keys=True)
    return {"role": role, "content": _sanitize_text(content)}


def _extract_code(text: str) -> Optional[str]:
    match = CODE_FENCE_RE.search(text)
    if match:
        return match.group(1).strip()
    # Fallback for rare assistant messages that return raw source.
    if "#include" in text and ("#pragma HLS" in text or "extern \"C\"" in text):
        return text.strip()
    return None


def _as_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        low = value.lower()
        if low in {"pass", "passed", "success", "true", "ok"}:
            return True
        if low in {"fail", "failed", "false", "error"}:
            return False
    return None


def _as_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> Optional[int]:
    value = _as_float(value)
    return int(value) if value is not None else None


def _bench_from_history_path(path: Path) -> str:
    name = path.name
    if name.endswith("_history.json"):
        return name[: -len("_history.json")]
    return path.stem


def _suite_from_path(path: Path) -> str:
    parts = path.parts
    if "results_sweeps" in parts:
        bench = _bench_from_history_path(path)
        if bench.startswith("hlsfactory_"):
            return "hlsfactory"
        if bench.startswith("hls_eval_") or "hlseval" in str(path).lower():
            return "hlseval"
        return "agentic"
    if "code_translation_c2hls" in str(path):
        return "portable_branch"
    return "unknown"


def _load_history(path: Path) -> Optional[HistoryDoc]:
    try:
        raw = _json_load(path)
    except (json.JSONDecodeError, OSError):
        return None

    if isinstance(raw, list):
        messages = raw
        model = None
        usage: dict[str, Any] = {}
    elif isinstance(raw, dict):
        messages = raw.get("messages") or raw.get("history")
        model = raw.get("model") or raw.get("model_synthesis") or raw.get("model_translator")
        usage = raw.get("llm_usage") or {}
    else:
        return None

    if not isinstance(messages, list):
        return None
    normalized = [m for m in messages if isinstance(m, dict) and "role" in m]
    if not normalized:
        return None
    return HistoryDoc(path=path, messages=normalized, model=model, usage=usage)


def _companion_result_path(history_path: Path, benchmark: str) -> Optional[Path]:
    candidates = [
        history_path.with_name(f"{benchmark}_multistep_results.json"),
        history_path.with_name(f"{benchmark}_results.json"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _collect_step_results(result: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("generated_step_history", "optimization_history", "turn_history", "steps"):
        value = result.get(key)
        if isinstance(value, list) and value:
            return [v for v in value if isinstance(v, dict)]
    return []


def _metrics_from_step(step: Optional[dict[str, Any]]) -> dict[str, Any]:
    if not step:
        return {}
    report = step.get("report") or step.get("synth_report") or {}
    csim = step.get("csim")
    if not isinstance(csim, dict):
        csim = {"passed": _as_bool(csim)}
    skill_prompt = step.get("skill_prompt") or {}
    return {
        "step": step.get("step_name") or step.get("step") or step.get("phase"),
        "synth_passed": _as_bool(step.get("success")),
        "csim_passed": _as_bool(
            csim.get("passed", csim.get("success", csim.get("status")))
        ),
        "latency_cycles": _as_int(report.get("latency_cycles")),
        "latency_ns": _as_float(report.get("latency_ns")),
        "interval": _as_int(report.get("interval")),
        "bram": _as_int(report.get("bram")),
        "dsp": _as_int(report.get("dsp")),
        "ff": _as_int(report.get("ff")),
        "lut": _as_int(report.get("lut")),
        "uram": _as_int(report.get("uram")),
        "fmax_mhz": _as_float(report.get("fmax_mhz")),
        "skill_prompt_scope": skill_prompt.get("prompt_scope"),
        "skill_prompt_mode": skill_prompt.get("prompt_mode"),
        "injected_skill_ids": skill_prompt.get("injected_skill_ids") or [],
        "avoid_skill_ids": skill_prompt.get("avoid_skill_ids") or [],
    }


def _result_level_metrics(result: Optional[dict[str, Any]]) -> dict[str, Any]:
    if not result:
        return {}
    synth = result.get("synth_report") or result.get("final_report") or {}
    csim = result.get("csim") or {}
    if not isinstance(csim, dict):
        csim = {"passed": _as_bool(csim)}
    return {
        "synth_passed": _as_bool(result.get("success")),
        "csim_passed": _as_bool(csim.get("passed", csim.get("success", csim.get("status")))),
        "latency_cycles": _as_int(synth.get("latency_cycles")),
        "latency_ns": _as_float(synth.get("latency_ns")),
        "interval": _as_int(synth.get("interval")),
        "bram": _as_int(synth.get("bram")),
        "dsp": _as_int(synth.get("dsp")),
        "ff": _as_int(synth.get("ff")),
        "lut": _as_int(synth.get("lut")),
        "uram": _as_int(synth.get("uram")),
        "fmax_mhz": _as_float(synth.get("fmax_mhz")),
    }


def _lookahead_status(messages: list[dict[str, Any]], assistant_index: int) -> dict[str, Any]:
    synth: Optional[bool] = None
    csim: Optional[bool] = None
    phase: Optional[str] = None
    snippets: list[str] = []
    for message in messages[assistant_index + 1 :]:
        role = message.get("role")
        if role in {"assistant", "user"}:
            break
        content = message.get("content", "")
        if not isinstance(content, str):
            continue
        snippets.append(content[:400])
        low = content.lower()
        if "synthesis success" in low or "synth success" in low:
            synth = True
        if "synthesis fail" in low or "synth fail" in low:
            synth = False
        if "csim" in low or "c-simulation" in low:
            if "passed" in low or "success" in low:
                csim = True
            if "failed" in low or "fail" in low:
                csim = False
        phase_match = re.search(r"\[(Phase [A-Z]|Step: [^\]]+)\]", content)
        if phase_match:
            phase = phase_match.group(1)
    return {
        "lookahead_synth_passed": synth,
        "lookahead_csim_passed": csim,
        "lookahead_phase": phase,
        "lookahead_status_excerpt": _sanitize_text("\\n".join(snippets))[:1200],
    }


def _quality_label(metrics: dict[str, Any], code: Optional[str]) -> str:
    if not code:
        return "no_code"
    synth = metrics.get("synth_passed")
    csim = metrics.get("csim_passed")
    if synth is False or csim is False:
        return "negative"
    if synth is True and csim is True:
        return "validated_positive"
    if synth is True:
        return "synth_positive"
    return "uncertain"


def _iter_history_paths(roots: Iterable[Path]) -> Iterable[Path]:
    seen: set[Path] = set()
    for root in roots:
        if root.is_file() and root.name.endswith("_history.json"):
            paths = [root]
        elif root.exists():
            paths = sorted(root.rglob("*_history.json"))
        else:
            paths = []
        for path in paths:
            real = path.resolve()
            if real in seen:
                continue
            seen.add(real)
            yield path


def build_records(roots: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for history_path in _iter_history_paths(roots):
        history = _load_history(history_path)
        if history is None:
            continue
        benchmark = _bench_from_history_path(history_path)
        suite = _suite_from_path(history_path)
        result_path = _companion_result_path(history_path, benchmark)
        result = None
        steps: list[dict[str, Any]] = []
        if result_path:
            try:
                result = _json_load(result_path)
                if isinstance(result, dict):
                    steps = _collect_step_results(result)
            except (json.JSONDecodeError, OSError):
                result = None

        assistant_ordinal = 0
        for i, message in enumerate(history.messages):
            if message.get("role") != "assistant":
                continue
            content = message.get("content", "")
            if not isinstance(content, str):
                continue
            code = _extract_code(content)
            step = steps[assistant_ordinal] if assistant_ordinal < len(steps) else None
            step_metrics = _metrics_from_step(step)
            lookahead = _lookahead_status(history.messages, i)
            result_metrics = _result_level_metrics(result if isinstance(result, dict) else None)

            metrics = {k: v for k, v in result_metrics.items() if v is not None}
            metrics.update({k: v for k, v in step_metrics.items() if v is not None})
            if metrics.get("synth_passed") is None:
                metrics["synth_passed"] = lookahead["lookahead_synth_passed"]
            if metrics.get("csim_passed") is None:
                metrics["csim_passed"] = lookahead["lookahead_csim_passed"]

            messages = [_sanitize_message(m) for m in history.messages[: i + 1]]
            label = _quality_label(metrics, code)
            record = {
                "schema_version": "agentic_sft.v1",
                "benchmark": benchmark,
                "suite": suite,
                "split": _stable_split(benchmark),
                "source": "agentic_history",
                "source_history": str(history_path),
                "source_result": str(result_path) if result_path else None,
                "model": history.model,
                "assistant_turn_index": assistant_ordinal,
                "quality_label": label,
                "messages": messages,
                "metadata": {
                    "code_sha256": _sha256(code or ""),
                    "code_chars": len(code or ""),
                    "message_count": len(messages),
                    "llm_usage": history.usage,
                    **metrics,
                    **lookahead,
                },
            }
            records.append(record)
            assistant_ordinal += 1
    return records


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")
            count += 1
    return count


def _trim_text(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    head = max(0, limit // 2)
    tail = max(0, limit - head)
    return text[:head] + "\n\n<TRUNCATED_CONTEXT>\n\n" + text[-tail:]


def _compact_record(
    record: dict[str, Any],
    *,
    max_context_chars: int,
    max_message_chars: int,
) -> dict[str, Any]:
    compact = dict(record)
    messages = list(record["messages"])
    if not messages:
        return compact

    target = messages[-1]
    context = messages[:-1]
    trimmed_context = [
        {
            "role": m.get("role", "user"),
            "content": _trim_text(str(m.get("content", "")), max_message_chars),
        }
        for m in context
    ]

    kept_reversed: list[dict[str, str]] = []
    total = 0
    for message in reversed(trimmed_context):
        size = len(message["content"])
        if kept_reversed and total + size > max_context_chars:
            continue
        kept_reversed.append(message)
        total += size

    kept = list(reversed(kept_reversed))
    if trimmed_context and trimmed_context[0].get("role") == "system" and (
        not kept or kept[0] != trimmed_context[0]
    ):
        kept.insert(0, trimmed_context[0])

    compact["messages"] = kept + [target]
    metadata = dict(record.get("metadata") or {})
    metadata["compact"] = True
    metadata["raw_message_count"] = len(messages)
    metadata["compact_message_count"] = len(compact["messages"])
    metadata["max_context_chars"] = max_context_chars
    metadata["max_message_chars"] = max_message_chars
    compact["metadata"] = metadata
    return compact


def _manifest(
    records: list[dict[str, Any]],
    output: Optional[Path],
    positive_output: Optional[Path],
    compact_output: Path,
    compact_positive_output: Path,
) -> dict[str, Any]:
    by_quality = Counter(r["quality_label"] for r in records)
    by_split = Counter(r["split"] for r in records)
    by_suite = Counter(r["suite"] for r in records)
    by_model = Counter(r.get("model") or "unknown" for r in records)
    positives = [
        r for r in records
        if r["quality_label"] in {"validated_positive", "synth_positive"}
    ]
    return {
        "schema_version": "agentic_sft_manifest.v1",
        "record_count": len(records),
        "positive_record_count": len(positives),
        "counts_by_quality": dict(sorted(by_quality.items())),
        "counts_by_split": dict(sorted(by_split.items())),
        "counts_by_suite": dict(sorted(by_suite.items())),
        "counts_by_model": dict(sorted(by_model.items())),
        "output_files": {
            "all": str(output) if output else None,
            "positive": str(positive_output) if positive_output else None,
            "compact_all": str(compact_output),
            "compact_positive": str(compact_positive_output),
        },
        "positive_policy": [
            "validated_positive: code present, synth passed, csim passed",
            "synth_positive: code present, synth passed, csim missing/unknown",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--history-root",
        action="append",
        default=[str(DEFAULT_RESULTS_SWEEPS)],
        help="Root to scan for *_history.json; can be passed more than once.",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--positive-output", default=str(DEFAULT_POSITIVE_OUTPUT))
    parser.add_argument("--compact-output", default=str(DEFAULT_COMPACT_OUTPUT))
    parser.add_argument(
        "--compact-positive-output",
        default=str(DEFAULT_COMPACT_POSITIVE_OUTPUT),
    )
    parser.add_argument("--max-context-chars", type=int, default=60000)
    parser.add_argument("--max-message-chars", type=int, default=16000)
    parser.add_argument(
        "--skip-raw",
        action="store_true",
        help="Do not write full-context raw JSONLs; useful when disk is tight.",
    )
    parser.add_argument(
        "--manifest",
        default=str(REPO_ROOT / "artifacts" / "rl_corpus" / "agentic_sft_manifest.json"),
    )
    args = parser.parse_args()

    roots = [Path(p) for p in args.history_root]
    output = Path(args.output)
    positive_output = Path(args.positive_output)
    compact_output = Path(args.compact_output)
    compact_positive_output = Path(args.compact_positive_output)
    manifest_path = Path(args.manifest)

    records = build_records(roots)
    positives = [
        r for r in records
        if r["quality_label"] in {"validated_positive", "synth_positive"}
    ]

    compact_records = [
        _compact_record(
            r,
            max_context_chars=args.max_context_chars,
            max_message_chars=args.max_message_chars,
        )
        for r in records
    ]
    compact_positives = [
        r for r in compact_records
        if r["quality_label"] in {"validated_positive", "synth_positive"}
    ]
    if args.skip_raw:
        all_count = 0
        positive_count = 0
        manifest_output = None
        manifest_positive_output = None
    else:
        all_count = _write_jsonl(output, records)
        positive_count = _write_jsonl(positive_output, positives)
        manifest_output = output
        manifest_positive_output = positive_output
    compact_count = _write_jsonl(compact_output, compact_records)
    compact_positive_count = _write_jsonl(compact_positive_output, compact_positives)
    manifest = _manifest(
        records,
        manifest_output,
        manifest_positive_output,
        compact_output,
        compact_positive_output,
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    if args.skip_raw:
        print("skipped raw full-context outputs")
    else:
        print(f"wrote {all_count} records to {output}")
        print(f"wrote {positive_count} positive records to {positive_output}")
    print(f"wrote {compact_count} compact records to {compact_output}")
    print(f"wrote {compact_positive_count} compact positive records to {compact_positive_output}")
    print(f"wrote manifest to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
