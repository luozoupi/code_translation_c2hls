#!/usr/bin/env python3
"""Run a small C2HLS corpus generation eval against a local vLLM endpoint."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "artifacts" / "rl_corpus" / "agentic_sft_v1" / "val.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "vllm_baseline"
DEFAULT_MODEL = "google/gemma-4-31B-it"
DEFAULT_BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
DEFAULT_API_KEY = os.environ.get("OPENAI_API_KEY", "EMPTY")


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as f:
        for row_index, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            record["_row_index"] = row_index
            yield record


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _messages_text_chars(messages: list[dict[str, Any]]) -> int:
    return sum(len(str(message.get("content", ""))) for message in messages)


def _normalize_chat_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    system_parts: list[str] = []
    non_system: list[dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role", "user"))
        content = str(message.get("content", ""))
        if role == "system":
            system_parts.append(content)
        else:
            non_system.append({**message, "role": role, "content": content})
    if not system_parts:
        return non_system
    return [{"role": "system", "content": "\n\n".join(system_parts)}] + non_system


def _largest_fenced_block(text: str) -> str:
    blocks = re.findall(r"```(?:[A-Za-z0-9_+.-]+)?\s*\n(.*?)```", text, flags=re.DOTALL)
    if not blocks:
        return ""
    return max(blocks, key=len).strip()


def _signals(text: str) -> dict[str, Any]:
    largest_block = _largest_fenced_block(text)
    return {
        "response_chars": len(text),
        "has_fenced_code": "```" in text,
        "fenced_block_count": text.count("```") // 2,
        "largest_fenced_block_chars": len(largest_block),
        "has_include": "#include" in text,
        "has_pragma_hls": "#pragma HLS" in text,
        "has_extern_c": 'extern "C"' in text,
        "has_ap_int_or_hls_type": any(token in text for token in ("ap_int", "ap_uint", "hls::")),
    }


def _select_records(
    records: list[dict[str, Any]],
    *,
    limit: int,
    seed: int,
    sample: str,
    unique_benchmarks: bool,
    max_prompt_chars: int,
    row_indices: set[int],
    benchmarks: set[str],
    exclude_benchmarks: set[str],
) -> list[dict[str, Any]]:
    candidates = []
    for record in records:
        if row_indices and int(record.get("_row_index", -1)) not in row_indices:
            continue
        if benchmarks and str(record.get("benchmark") or "") not in benchmarks:
            continue
        if str(record.get("benchmark") or "") in exclude_benchmarks:
            continue
        messages = record.get("messages") or []
        if len(messages) < 2:
            continue
        prompt_chars = _messages_text_chars(messages[:-1])
        if max_prompt_chars and prompt_chars > max_prompt_chars:
            continue
        candidates.append(record)

    if sample == "random":
        rng = random.Random(seed)
        candidates = candidates[:]
        rng.shuffle(candidates)

    selected = []
    seen_benchmarks: set[str] = set()
    for record in candidates:
        benchmark = str(record.get("benchmark") or "")
        if unique_benchmarks and benchmark in seen_benchmarks:
            continue
        if benchmark:
            seen_benchmarks.add(benchmark)
        selected.append(record)
        if limit and len(selected) >= limit:
            break
    return selected


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    temporary.replace(path)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as output:
        output.write(json.dumps(payload, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def _result_benchmark(result: dict[str, Any]) -> str:
    return str((result.get("record") or {}).get("benchmark") or "")


def _load_latest_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    latest: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    with path.open() as source:
        for line in source:
            line = line.strip()
            if not line:
                continue
            result = json.loads(line)
            benchmark = _result_benchmark(result)
            if not benchmark:
                continue
            if benchmark not in latest:
                order.append(benchmark)
            latest[benchmark] = result
    return [latest[benchmark] for benchmark in order]


def _summary_payload(
    *,
    args: argparse.Namespace,
    input_path: Path,
    output_jsonl: Path,
    records: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    results: list[dict[str, Any]],
    state: str,
    current_benchmark: str | None = None,
) -> dict[str, Any]:
    status_counts = Counter(str(result.get("status")) for result in results)
    finish_counts = Counter(
        str(result.get("finish_reason"))
        for result in results
        if result.get("finish_reason")
    )
    signal_counts = Counter()
    response_chars = []
    for result in results:
        signals = result.get("signals") or {}
        for key, value in signals.items():
            if isinstance(value, bool) and value:
                signal_counts[key] += 1
        if "response_chars" in signals:
            response_chars.append(signals["response_chars"])
    completed_benchmarks = {
        str((result.get("record") or {}).get("benchmark") or "")
        for result in results
    }
    return {
        "schema_version": "vllm_corpus_eval_v1_summary",
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "state": state,
        "current_benchmark": current_benchmark,
        "input": str(input_path),
        "output_jsonl": str(output_jsonl),
        "model": args.model,
        "base_url": args.base_url.rstrip("/"),
        "selection": {
            "records_available": len(records),
            "records_selected": len(selected),
            "records_completed": len(results),
            "records_succeeded": sum(
                1 for result in results if result.get("status") == "ok"
            ),
            "records_pending": max(0, len(selected) - len(results)),
            "limit": args.limit,
            "seed": args.seed,
            "sample": args.sample,
            "unique_benchmarks": args.unique_benchmarks,
            "max_prompt_chars": args.max_prompt_chars,
            "row_indices": args.row_index or [],
            "benchmarks": args.benchmark or [],
            "exclude_benchmarks": args.exclude_benchmark or [],
        },
        "request": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "timeout": args.timeout,
            "retries": args.retries,
        },
        "status_counts": dict(status_counts),
        "finish_reason_counts": dict(finish_counts),
        "signal_counts": dict(signal_counts),
        "response_chars": {
            "min": min(response_chars) if response_chars else None,
            "max": max(response_chars) if response_chars else None,
            "mean": (
                sum(response_chars) / len(response_chars)
                if response_chars
                else None
            ),
        },
        "benchmarks": [
            str(record.get("benchmark") or "") for record in selected
        ],
        "completed_benchmarks": sorted(completed_benchmarks),
    }


def _post_chat_completion(
    *,
    endpoint: str,
    api_key: str,
    payload: dict[str, Any],
    timeout: int,
) -> dict[str, Any]:
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    return json.loads(body)


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = list(_read_jsonl(input_path))
    selected = _select_records(
        records,
        limit=args.limit,
        seed=args.seed,
        sample=args.sample,
        unique_benchmarks=args.unique_benchmarks,
        max_prompt_chars=args.max_prompt_chars,
        row_indices=set(args.row_index or []),
        benchmarks=set(args.benchmark or []),
        exclude_benchmarks=set(args.exclude_benchmark or []),
    )

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{args.run_name}_{timestamp}" if args.run_name else f"vllm_eval_{timestamp}"
    output_jsonl = (
        Path(args.output_jsonl)
        if args.output_jsonl
        else output_dir / f"{stem}.jsonl"
    )
    summary_path = (
        Path(args.summary_path)
        if args.summary_path
        else output_dir / f"{stem}.summary.json"
    )
    heartbeat_path = (
        Path(args.heartbeat_path)
        if args.heartbeat_path
        else summary_path.with_suffix(".heartbeat.json")
    )
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    heartbeat_path.parent.mkdir(parents=True, exist_ok=True)

    if args.resume:
        results = _load_latest_results(output_jsonl)
    else:
        output_jsonl.write_text("")
        results = []

    selected_benchmarks = {
        str(record.get("benchmark") or "") for record in selected
    }
    results = [
        result
        for result in results
        if _result_benchmark(result) in selected_benchmarks
    ]
    latest_by_benchmark = {
        _result_benchmark(result): result for result in results
    }
    endpoint = args.base_url.rstrip("/") + "/chat/completions"

    def checkpoint(state: str, current_benchmark: str | None = None) -> dict[str, Any]:
        latest_results = [
            latest_by_benchmark[benchmark]
            for benchmark in (
                str(record.get("benchmark") or "") for record in selected
            )
            if benchmark in latest_by_benchmark
        ]
        summary = _summary_payload(
            args=args,
            input_path=input_path,
            output_jsonl=output_jsonl,
            records=records,
            selected=selected,
            results=latest_results,
            state=state,
            current_benchmark=current_benchmark,
        )
        _write_json(summary_path, summary)
        _write_json(
            heartbeat_path,
            {
                "schema_version": "vllm_corpus_eval_v1_heartbeat",
                "updated_at": dt.datetime.now().isoformat(timespec="seconds"),
                "pid": os.getpid(),
                "state": state,
                "model": args.model,
                "base_url": args.base_url.rstrip("/"),
                "current_benchmark": current_benchmark,
                "records_selected": len(selected),
                "records_completed": len(latest_results),
                "records_succeeded": sum(
                    1
                    for result in latest_results
                    if result.get("status") == "ok"
                ),
                "output_jsonl": str(output_jsonl),
                "summary": str(summary_path),
            },
        )
        return summary

    checkpoint("running")
    consecutive_errors = 0
    state = "complete"
    for ordinal, record in enumerate(selected):
        benchmark = str(record.get("benchmark") or "")
        existing = latest_by_benchmark.get(benchmark)
        if existing and existing.get("status") == "ok":
            print(
                json.dumps(
                    {
                        "ordinal": ordinal,
                        "benchmark": benchmark,
                        "status": "resume_skip",
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            continue

        checkpoint("running", benchmark)
        messages = record["messages"]
        prompt_messages = _normalize_chat_messages(messages[:-1])
        target_text = str(messages[-1].get("content", ""))
        payload = {
            "model": args.model,
            "messages": prompt_messages,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
        }
        if args.top_p is not None:
            payload["top_p"] = args.top_p
        if "qwen" in args.model.lower():
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        started = time.time()
        result: dict[str, Any] = {
            "schema_version": "vllm_corpus_eval_v1",
            "ordinal": ordinal,
            "input": str(input_path),
            "record": {
                "row_index": record.get("_row_index"),
                "benchmark": record.get("benchmark"),
                "suite": record.get("suite"),
                "split": record.get("split"),
                "quality_label": record.get("quality_label"),
                "source_history": record.get("source_history"),
                "source_result": record.get("source_result"),
                "model_teacher": record.get("model_teacher"),
                "metadata": record.get("metadata"),
            },
            "request": {
                "model": args.model,
                "base_url": args.base_url.rstrip("/"),
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_tokens": args.max_tokens,
                "prompt_chars": _messages_text_chars(prompt_messages),
                "prompt_sha256": _sha256(json.dumps(prompt_messages, sort_keys=True)),
                "target_chars": len(target_text),
                "target_sha256": _sha256(target_text),
            },
        }

        attempts: list[dict[str, Any]] = []
        for attempt in range(1, args.retries + 2):
            attempt_started = time.time()
            try:
                response = _post_chat_completion(
                    endpoint=endpoint,
                    api_key=args.api_key,
                    payload=payload,
                    timeout=args.timeout,
                )
                choice = (response.get("choices") or [{}])[0]
                content = str((choice.get("message") or {}).get("content", ""))
                attempts.append(
                    {
                        "attempt": attempt,
                        "status": "ok",
                        "elapsed_sec": round(time.time() - attempt_started, 3),
                    }
                )
                result.update(
                    {
                        "status": "ok",
                        "elapsed_sec": time.time() - started,
                        "finish_reason": choice.get("finish_reason"),
                        "usage": response.get("usage"),
                        "response": {
                            "content": content,
                            "content_sha256": _sha256(content),
                        },
                        "signals": _signals(content),
                    }
                )
                break
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                error = {
                    "code": exc.code,
                    "reason": str(exc.reason),
                    "body": body[:4000],
                }
                attempts.append(
                    {
                        "attempt": attempt,
                        "status": "http_error",
                        "elapsed_sec": round(time.time() - attempt_started, 3),
                        "error": error,
                    }
                )
                result.update(
                    {
                        "status": "http_error",
                        "elapsed_sec": time.time() - started,
                        "error": error,
                    }
                )
            except Exception as exc:  # noqa: BLE001 - preserve exact audit failure
                error = {"type": type(exc).__name__, "message": str(exc)}
                attempts.append(
                    {
                        "attempt": attempt,
                        "status": "error",
                        "elapsed_sec": round(time.time() - attempt_started, 3),
                        "error": error,
                    }
                )
                result.update(
                    {
                        "status": "error",
                        "elapsed_sec": time.time() - started,
                        "error": error,
                    }
                )
            if attempt <= args.retries:
                time.sleep(args.retry_backoff_seconds * attempt)

        result["attempts"] = attempts
        result["attempt_count"] = len(attempts)
        _append_jsonl(output_jsonl, result)
        latest_by_benchmark[benchmark] = result
        checkpoint("running")
        print(
            json.dumps(
                {
                    "ordinal": ordinal,
                    "benchmark": record.get("benchmark"),
                    "status": result["status"],
                    "elapsed_sec": round(result["elapsed_sec"], 3),
                    "finish_reason": result.get("finish_reason"),
                    "response_chars": (result.get("signals") or {}).get("response_chars"),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if result.get("status") == "ok":
            consecutive_errors = 0
        else:
            consecutive_errors += 1
            if consecutive_errors >= args.max_consecutive_errors:
                state = "aborted_consecutive_errors"
                break
        if args.inter_request_delay_seconds:
            time.sleep(args.inter_request_delay_seconds)

    summary = checkpoint(state)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--run-name", default="gemma4_31b_agentic_sft_val_smoke")
    parser.add_argument(
        "--output-jsonl",
        default="",
        help="Fixed checkpoint JSONL path. Overrides timestamped output naming.",
    )
    parser.add_argument(
        "--summary-path",
        default="",
        help="Fixed atomic summary path. Defaults beside the output JSONL.",
    )
    parser.add_argument(
        "--heartbeat-path",
        default="",
        help="Fixed atomic heartbeat path. Defaults beside the summary.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample", choices=["first", "random"], default="random")
    parser.add_argument(
        "--row-index",
        type=int,
        action="append",
        default=[],
        help="Input JSONL row index to evaluate; repeatable.",
    )
    parser.add_argument(
        "--benchmark",
        action="append",
        default=[],
        help="Benchmark name to evaluate; repeatable.",
    )
    parser.add_argument(
        "--exclude-benchmark",
        action="append",
        default=[],
        help="Benchmark name to exclude; repeatable.",
    )
    parser.add_argument(
        "--unique-benchmarks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sample at most one record per benchmark. Default: true.",
    )
    parser.add_argument("--max-prompt-chars", type=int, default=40000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--retry-backoff-seconds", type=float, default=5.0)
    parser.add_argument("--max-consecutive-errors", type=int, default=3)
    parser.add_argument("--inter-request-delay-seconds", type=float, default=0.0)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Resume successful benchmarks from an existing fixed output JSONL.",
    )
    run_eval(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
