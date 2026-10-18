#!/usr/bin/env python3
"""Serve a local PEFT adapter through a minimal OpenAI-compatible API.

This is a single-request-at-a-time inference server intended for controlled
C2HLS agent experiments. It loads a 4-bit base checkpoint, attaches a LoRA
adapter, and exposes `/v1/models` and `/v1/chat/completions` on localhost.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import platform
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_CHECKPOINT = Path(
    "/mnt/data/vllm_models/hub/models--Qwen--Qwen3.6-27B/"
    "snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
)
DEFAULT_ADAPTER = Path(
    "/mnt/data2/luo00466/c2hls_sft/runs/"
    "qwen3.6-27b_qlora_r16_smoke10_1024_20260716/adapter"
)
TRUNCATION_MARKER = "\n\n<C2HLS_SERVER_CONTEXT_TRUNCATED>\n\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--model-id", default="qwen3.6-27b-c2hls-sft-smoke10")
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30101)
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--min-free-gib", type=float, default=80.0)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--log-file", type=Path)
    return parser.parse_args()


def _configure_process(args: argparse.Namespace) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if args.log_file:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(args.log_file))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
        force=True,
    )


def normalize_messages(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise ValueError("messages must be a non-empty list")
    messages: list[dict[str, str]] = []
    for index, message in enumerate(value):
        if not isinstance(message, dict):
            raise ValueError(f"message {index} is not an object")
        role = str(message.get("role") or "").strip()
        content = message.get("content")
        if role not in {"system", "developer", "user", "assistant", "tool"}:
            raise ValueError(f"message {index} has unsupported role {role!r}")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False, sort_keys=True)
        messages.append({"role": role, "content": content})
    return messages


def cap_input_ids(
    input_ids: list[int],
    *,
    max_tokens: int,
    marker_ids: list[int],
    head_fraction: float = 0.40,
) -> tuple[list[int], int]:
    """Cap a prompt while preserving both system-prefix and latest context."""
    if max_tokens <= 0 or len(input_ids) <= max_tokens:
        return input_ids, 0
    retained = max_tokens - len(marker_ids)
    if retained < 2:
        raise ValueError("max input budget is too small for the truncation marker")
    head = max(1, int(retained * head_fraction))
    tail = retained - head
    capped = input_ids[:head] + marker_ids
    if tail:
        capped += input_ids[-tail:]
    return capped, len(input_ids) - head - tail


def _gpu_snapshot(torch: Any) -> dict[str, Any]:
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    return {
        "name": torch.cuda.get_device_name(0),
        "free_gib": free_bytes / 2**30,
        "total_gib": total_bytes / 2**30,
        "allocated_gib": torch.cuda.memory_allocated(0) / 2**30,
        "reserved_gib": torch.cuda.memory_reserved(0) / 2**30,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(0) / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(0) / 2**30,
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


class PeftChatServer:
    def __init__(self, args: argparse.Namespace) -> None:
        import bitsandbytes
        import peft
        import torch
        import transformers
        from peft import PeftModel
        from transformers import (
            AutoModelForImageTextToText,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError(
                f"expected exactly one visible CUDA device, found {torch.cuda.device_count()}"
            )
        gpu_before = _gpu_snapshot(torch)
        if gpu_before["free_gib"] < args.min_free_gib:
            raise RuntimeError(
                f"GPU has only {gpu_before['free_gib']:.1f} GiB free; "
                f"required {args.min_free_gib:.1f} GiB"
            )
        if not args.checkpoint.is_dir():
            raise FileNotFoundError(f"base checkpoint does not exist: {args.checkpoint}")
        if not (args.adapter / "adapter_config.json").is_file():
            raise FileNotFoundError(f"PEFT adapter is incomplete: {args.adapter}")

        self.args = args
        self.torch = torch
        self.lock = asyncio.Lock()
        self.requests = 0
        self.requests_with_truncation = 0
        self.prompt_tokens_dropped_total = 0
        self.max_prompt_tokens_seen = 0
        self.max_prompt_tokens_dropped = 0
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.adapter,
            local_files_only=True,
            trust_remote_code=False,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.marker_ids = self.tokenizer.encode(
            TRUNCATION_MARKER, add_special_tokens=False
        )

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        load_started = time.monotonic()
        base_model = AutoModelForImageTextToText.from_pretrained(
            args.checkpoint,
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
            quantization_config=quantization_config,
            device_map={"": 0},
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )
        self.model = PeftModel.from_pretrained(
            base_model,
            args.adapter,
            is_trainable=False,
            local_files_only=True,
        )
        self.model.eval()
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = True
        if hasattr(self.model.config, "text_config"):
            self.model.config.text_config.use_cache = True
        self.load_seconds = time.monotonic() - load_started
        self.versions = {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "bitsandbytes": bitsandbytes.__version__,
        }
        self.gpu_before = gpu_before
        self.gpu_after_load = _gpu_snapshot(torch)
        logging.info(
            "Loaded model=%s adapter=%s on GPU %d in %.2fs",
            args.checkpoint,
            args.adapter,
            args.gpu,
            self.load_seconds,
        )
        if args.manifest:
            self.write_manifest(args.manifest, status="ready")

    def manifest(self, *, status: str) -> dict[str, Any]:
        adapter_weights = self.args.adapter / "adapter_model.safetensors"
        return {
            "schema_version": "c2hls.peft_openai_server.v1",
            "status": status,
            "started_at": self.started_at,
            "model_id": self.args.model_id,
            "checkpoint": str(self.args.checkpoint),
            "adapter": str(self.args.adapter),
            "adapter_bytes": adapter_weights.stat().st_size if adapter_weights.exists() else None,
            "physical_gpu": self.args.gpu,
            "host": self.args.host,
            "port": self.args.port,
            "base_url": f"http://{self.args.host}:{self.args.port}/v1",
            "max_input_tokens": self.args.max_input_tokens,
            "max_new_tokens": self.args.max_new_tokens,
            "requests_completed": self.requests,
            "context_audit": {
                "requests_with_truncation": self.requests_with_truncation,
                "prompt_tokens_dropped_total": self.prompt_tokens_dropped_total,
                "max_prompt_tokens_seen": self.max_prompt_tokens_seen,
                "max_prompt_tokens_dropped": self.max_prompt_tokens_dropped,
            },
            "model_load_seconds": self.load_seconds,
            "versions": self.versions,
            "gpu_before_load": self.gpu_before,
            "gpu_after_load": self.gpu_after_load,
            "gpu_current": _gpu_snapshot(self.torch),
        }

    def write_manifest(self, path: Path, *, status: str) -> None:
        _write_json(path, self.manifest(status=status))

    def _generate(self, payload: dict[str, Any]) -> dict[str, Any]:
        messages = normalize_messages(payload.get("messages"))
        requested_model = str(payload.get("model") or self.args.model_id)
        if requested_model != self.args.model_id:
            raise ValueError(
                f"unknown model {requested_model!r}; available model is {self.args.model_id!r}"
            )
        max_new_tokens = int(
            payload.get("max_tokens")
            or payload.get("max_completion_tokens")
            or self.args.max_new_tokens
        )
        max_new_tokens = max(1, min(max_new_tokens, self.args.max_new_tokens))
        temperature = float(payload.get("temperature") or 0.0)
        top_p = float(payload.get("top_p") or 1.0)

        rendered = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        input_ids = self.tokenizer.encode(rendered, add_special_tokens=False)
        input_ids, dropped = cap_input_ids(
            input_ids,
            max_tokens=self.args.max_input_tokens,
            marker_ids=self.marker_ids,
        )
        input_tensor = self.torch.tensor(
            [input_ids], dtype=self.torch.long, device="cuda:0"
        )
        attention_mask = self.torch.ones_like(input_tensor)
        generation_kwargs: dict[str, Any] = {
            "input_ids": input_tensor,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,
        }
        if temperature > 0:
            generation_kwargs.update({"temperature": temperature, "top_p": top_p})

        started = time.monotonic()
        with self.torch.inference_mode():
            generated = self.model.generate(**generation_kwargs)
        elapsed = time.monotonic() - started
        new_ids = generated[0, input_tensor.shape[1] :].tolist()
        content = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        for stop in payload.get("stop") or []:
            if isinstance(stop, str) and stop in content:
                content = content.split(stop, 1)[0]
        finish_reason = "length" if len(new_ids) >= max_new_tokens else "stop"
        self.requests += 1
        self.max_prompt_tokens_seen = max(self.max_prompt_tokens_seen, len(input_ids))
        self.max_prompt_tokens_dropped = max(self.max_prompt_tokens_dropped, dropped)
        self.prompt_tokens_dropped_total += dropped
        if dropped:
            self.requests_with_truncation += 1
        logging.info(
            "Completed request=%d prompt_tokens=%d dropped=%d completion_tokens=%d",
            self.requests,
            len(input_ids),
            dropped,
            len(new_ids),
        )
        if self.args.manifest:
            self.write_manifest(self.args.manifest, status="ready")
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.args.model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": len(input_ids),
                "completion_tokens": len(new_ids),
                "total_tokens": len(input_ids) + len(new_ids),
                "prompt_tokens_dropped": dropped,
            },
            "c2hls_server": {"elapsed_sec": elapsed},
        }

    async def chat_completions(self, request: Any) -> Any:
        from aiohttp import web

        try:
            payload = await request.json()
            if payload.get("stream"):
                raise ValueError("streaming responses are not supported")
            async with self.lock:
                response = await asyncio.to_thread(self._generate, payload)
            return web.json_response(response)
        except ValueError as exc:
            return web.json_response(
                {"error": {"message": str(exc), "type": "invalid_request_error"}},
                status=400,
            )
        except Exception as exc:  # noqa: BLE001 - preserve server diagnostics
            logging.error("Generation failed: %s\n%s", exc, traceback.format_exc())
            return web.json_response(
                {"error": {"message": str(exc), "type": type(exc).__name__}},
                status=500,
            )

    async def models(self, request: Any) -> Any:
        del request
        from aiohttp import web

        return web.json_response(
            {
                "object": "list",
                "data": [
                    {
                        "id": self.args.model_id,
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "c2hls-local-peft",
                    }
                ],
            }
        )

    async def health(self, request: Any) -> Any:
        del request
        from aiohttp import web

        return web.json_response(
            {
                "status": "ok",
                "model": self.args.model_id,
                "requests_completed": self.requests,
                "requests_with_truncation": self.requests_with_truncation,
                "prompt_tokens_dropped_total": self.prompt_tokens_dropped_total,
            }
        )


def main() -> int:
    args = _parse_args()
    _configure_process(args)
    from aiohttp import web

    server = PeftChatServer(args)
    app = web.Application(client_max_size=16 * 1024**2)
    app.router.add_get("/healthz", server.health)
    app.router.add_get("/v1/models", server.models)
    app.router.add_post("/v1/chat/completions", server.chat_completions)
    logging.info("Serving %s at http://%s:%d/v1", args.model_id, args.host, args.port)
    web.run_app(app, host=args.host, port=args.port, print=None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
