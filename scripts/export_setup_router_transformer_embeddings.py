#!/usr/bin/env python3
"""Export deduplicated frozen-transformer embeddings for router inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_setup_router_corpus import _load_input, _phase_b_code


SCHEMA_VERSION = "c2hls.setup-router-transformer-embeddings.v1"
DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"
SOURCE_INSTRUCTION = (
    "Encode this plain C or C++ kernel for predicting which HLS agent "
    "strategy and skill-routing setup will produce the lowest valid "
    "Vitis latency."
)
PHASE_B_INSTRUCTION = (
    "Encode this functionally correct Phase-B HLS C or C++ kernel for "
    "predicting which agent strategy and skill-routing setup will "
    "produce the lowest valid Vitis latency."
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _instruct(instruction: str, code: str) -> str:
    return f"Instruct: {instruction}\nQuery:\n{code}"


def _last_token_pool(
    last_hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    left_padded = bool(
        attention_mask[:, -1].sum().item() == attention_mask.shape[0]
    )
    if left_padded:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    row_indices = torch.arange(
        last_hidden_states.shape[0],
        device=last_hidden_states.device,
    )
    return last_hidden_states[row_indices, sequence_lengths]


def _source_text(
    benchmarks_dir: Path,
    benchmark: str,
) -> tuple[str, str]:
    plain, header = _load_input(benchmarks_dir, benchmark)
    combined = (
        "[PLAIN_KERNEL]\n"
        f"{plain.rstrip()}\n"
        "[KERNEL_HEADER]\n"
        f"{header.rstrip()}\n"
    )
    return combined, _text_sha256(combined)


def _embedding_requests(
    records: list[dict[str, Any]],
    benchmarks_dir: Path,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, str]]]:
    requests: dict[str, dict[str, Any]] = {}
    source_cache: dict[str, tuple[str, str]] = {}
    phase_cache: dict[tuple[str, str], str] = {}
    record_map: list[dict[str, str]] = []
    seen_records: set[str] = set()

    for record in records:
        benchmark = str(record["benchmark"])
        provenance = record.get("provenance") or {}
        record_id = str(provenance.get("dedup_key_sha256") or "")
        if not record_id:
            raise ValueError(f"{benchmark}: missing dedup_key_sha256")
        if record_id in seen_records:
            raise ValueError(f"duplicate corpus record id: {record_id}")
        seen_records.add(record_id)

        if benchmark not in source_cache:
            source_cache[benchmark] = _source_text(
                benchmarks_dir,
                benchmark,
            )
        source_code, source_hash = source_cache[benchmark]
        source_key = f"source:{source_hash}"
        requests.setdefault(
            source_key,
            {
                "key": source_key,
                "kind": "source",
                "benchmark": benchmark,
                "code_sha256": source_hash,
                "text": _instruct(SOURCE_INSTRUCTION, source_code),
            },
        )

        expected_phase_hash = str(
            provenance.get("phase_b_code_sha256") or ""
        )
        result_path = Path(str(provenance.get("source_result_path") or ""))
        explicit_phase_path = Path(
            str(provenance.get("phase_b_code_path") or "")
        )
        cache_key = (
            benchmark,
            str(explicit_phase_path)
            if explicit_phase_path.is_file()
            else str(result_path),
        )
        if cache_key not in phase_cache:
            if explicit_phase_path.is_file():
                phase_cache[cache_key] = explicit_phase_path.read_text(
                    encoding="utf-8"
                )
            else:
                history_path = result_path.with_name(
                    f"{benchmark}_history.json"
                )
                if not history_path.is_file():
                    raise FileNotFoundError(
                        f"{benchmark}: missing Phase-B history "
                        f"{history_path}"
                    )
                phase_cache[cache_key] = _phase_b_code(history_path)
        phase_code = phase_cache[cache_key]
        phase_hash = _text_sha256(phase_code)
        if not phase_code:
            raise ValueError(f"{benchmark}: empty Phase-B code")
        if phase_hash != expected_phase_hash:
            raise ValueError(
                f"{benchmark}: Phase-B hash mismatch: "
                f"{phase_hash} != {expected_phase_hash}"
            )
        phase_key = f"phase_b:{phase_hash}"
        requests.setdefault(
            phase_key,
            {
                "key": phase_key,
                "kind": "phase_b",
                "benchmark": benchmark,
                "code_sha256": phase_hash,
                "text": _instruct(PHASE_B_INSTRUCTION, phase_code),
            },
        )
        record_map.append(
            {
                "record_id": record_id,
                "benchmark": benchmark,
                "benchmark_lineage": str(record["benchmark_lineage"]),
                "split": str(record["split"]),
                "source_embedding_key": source_key,
                "phase_b_embedding_key": phase_key,
            }
        )
    return requests, record_map


def _token_lengths(tokenizer: Any, texts: list[str]) -> list[int]:
    lengths = []
    for text in texts:
        encoded = tokenizer(
            text,
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=False,
        )
        lengths.append(len(encoded["input_ids"]))
    return lengths


def _encode(
    *,
    model: Any,
    tokenizer: Any,
    texts: list[str],
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    vectors = []
    for start in range(0, len(texts), batch_size):
        batch = tokenizer(
            texts[start : start + batch_size],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.inference_mode():
            output = model(**batch)
            pooled = _last_token_pool(
                output.last_hidden_state,
                batch["attention_mask"],
            )
            pooled = F.normalize(pooled.float(), p=2, dim=1)
        vectors.append(pooled.cpu().numpy().astype(np.float32))
    return np.concatenate(vectors, axis=0)


def export(args: argparse.Namespace) -> dict[str, Any]:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    records = _load_records(args.corpus)
    requests, record_map = _embedding_requests(
        records,
        args.benchmarks_dir,
    )
    ordered = [requests[key] for key in sorted(requests)]
    reused_manifest = None
    if args.reuse_embedding_dir is not None:
        reuse_dir = args.reuse_embedding_dir
        archive = np.load(
            reuse_dir / "embeddings.npz",
            allow_pickle=False,
        )
        reused_vectors = {
            str(key): vector
            for key, vector in zip(
                archive["keys"],
                archive["vectors"],
                strict=True,
            )
        }
        reused_index = {}
        with (reuse_dir / "embedding_index.jsonl").open(
            encoding="utf-8"
        ) as handle:
            for line in handle:
                if line.strip():
                    item = json.loads(line)
                    reused_index[str(item["embedding_key"])] = item
        requested_keys = [str(item["key"]) for item in ordered]
        missing = sorted(set(requested_keys) - set(reused_vectors))
        if missing:
            raise ValueError(
                "reuse embedding directory lacks requested keys: "
                + ", ".join(missing[:10])
            )
        vectors = np.asarray(
            [reused_vectors[key] for key in requested_keys],
            dtype=np.float32,
        )
        token_lengths = [
            int((reused_index.get(key) or {}).get("token_count") or 0)
            for key in requested_keys
        ]
        reused_manifest = json.loads(
            (reuse_dir / "manifest.json").read_text(encoding="utf-8")
        )
    else:
        texts = [str(item["text"]) for item in ordered]
        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            local_files_only=args.local_files_only,
            padding_side="left",
        )
        token_lengths = _token_lengths(tokenizer, texts)
        device = torch.device(args.device)
        dtype = (
            torch.bfloat16
            if device.type == "cuda" and torch.cuda.is_bf16_supported()
            else torch.float32
        )
        model = AutoModel.from_pretrained(
            args.model,
            local_files_only=args.local_files_only,
            dtype=dtype,
        )
        model.eval()
        model.to(device)
        vectors = _encode(
            model=model,
            tokenizer=tokenizer,
            texts=texts,
            device=device,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
    if vectors.shape[0] != len(ordered):
        raise RuntimeError("embedding row count mismatch")
    if not np.isfinite(vectors).all():
        raise RuntimeError("non-finite transformer embeddings")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    vectors_path = args.output_dir / "embeddings.npz"
    np.savez_compressed(
        vectors_path,
        keys=np.asarray([item["key"] for item in ordered]),
        vectors=vectors,
    )
    index_path = args.output_dir / "embedding_index.jsonl"
    with index_path.open("w", encoding="utf-8") as handle:
        for index, (item, token_count) in enumerate(
            zip(ordered, token_lengths, strict=True)
        ):
            handle.write(
                json.dumps(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "embedding_index": index,
                        "embedding_key": item["key"],
                        "kind": item["kind"],
                        "benchmark_example": item["benchmark"],
                        "code_sha256": item["code_sha256"],
                        "token_count": token_count,
                        "truncated": token_count > args.max_length,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    record_map_path = args.output_dir / "record_embedding_map.jsonl"
    with record_map_path.open("w", encoding="utf-8") as handle:
        for item in record_map:
            handle.write(json.dumps(item, sort_keys=True) + "\n")

    config = getattr(locals().get("model"), "config", None)
    model_name = (
        str(reused_manifest.get("model"))
        if reused_manifest is not None
        else args.model
    )
    model_revision = (
        reused_manifest.get("model_revision")
        if reused_manifest is not None
        else getattr(config, "_commit_hash", None)
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "source_corpus": str(args.corpus.resolve()),
        "source_corpus_sha256": _sha256(args.corpus),
        "benchmarks_dir": str(args.benchmarks_dir.resolve()),
        "model": model_name,
        "model_revision": model_revision,
        "model_hidden_size": int(vectors.shape[1]),
        "pooling": "last_non_padding_token_then_l2_normalize",
        "matryoshka_supported_dimensions": [32, int(vectors.shape[1])],
        "max_length": args.max_length,
        "request_count": len(ordered),
        "source_embedding_count": sum(
            item["kind"] == "source" for item in ordered
        ),
        "phase_b_embedding_count": sum(
            item["kind"] == "phase_b" for item in ordered
        ),
        "record_count": len(record_map),
        "max_observed_tokens": max(token_lengths, default=0),
        "truncated_request_count": sum(
            value > args.max_length for value in token_lengths
        ),
        "instructions": {
            "source": SOURCE_INSTRUCTION,
            "phase_b": PHASE_B_INSTRUCTION,
        },
        "reuse": (
            {
                "source_dir": str(args.reuse_embedding_dir.resolve()),
                "source_manifest_sha256": _sha256(
                    args.reuse_embedding_dir / "manifest.json"
                ),
                "all_requested_keys_reused": True,
            }
            if args.reuse_embedding_dir is not None
            else None
        ),
        "artifacts": {
            path.name: {
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in (vectors_path, index_path, record_map_path)
        },
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--benchmarks-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument(
        "--reuse-embedding-dir",
        type=Path,
        help=(
            "Reuse vectors when every source/Phase-B embedding key already "
            "exists; only the record map and manifest are regenerated."
        ),
    )
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.max_length < 128:
        parser.error("--max-length must be at least 128")
    if args.reuse_embedding_dir is not None:
        required = (
            "embeddings.npz",
            "embedding_index.jsonl",
            "manifest.json",
        )
        missing = [
            name
            for name in required
            if not (args.reuse_embedding_dir / name).is_file()
        ]
        if missing:
            parser.error(
                "--reuse-embedding-dir is incomplete: "
                + ", ".join(missing)
            )
    return args


if __name__ == "__main__":
    result = export(parse_args())
    print(
        json.dumps(
            {
                "model": result["model"],
                "requests": result["request_count"],
                "records": result["record_count"],
                "truncated": result["truncated_request_count"],
            },
            sort_keys=True,
        )
    )
