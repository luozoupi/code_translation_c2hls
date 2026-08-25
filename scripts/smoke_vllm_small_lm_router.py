#!/usr/bin/env python3
"""Load a setup-router LoRA through vLLM and run one constrained decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


def _first_validation_prompt(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record["orientation"] == 0:
                return record
    raise ValueError("validation dataset has no canonical pair")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--validation-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    record = _first_validation_prompt(args.validation_jsonl)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=True,
    )
    allowed = [
        tokenizer.encode(value, add_special_tokens=False)[0]
        for value in ("A", "B")
    ]
    engine = LLM(
        model=str(args.model),
        enable_lora=True,
        max_model_len=2048,
        gpu_memory_utilization=0.25,
        dtype="bfloat16",
        trust_remote_code=True,
        enforce_eager=True,
    )
    parameters = SamplingParams(
        temperature=0,
        max_tokens=1,
        allowed_token_ids=allowed,
    )
    base = engine.generate(
        [record["prompt"]],
        parameters,
        use_tqdm=False,
    )[0].outputs[0].text.strip()
    adapted = engine.generate(
        [record["prompt"]],
        parameters,
        lora_request=LoRARequest(
            "c2hls_setup_router",
            1,
            str(args.adapter),
        ),
        use_tqdm=False,
    )[0].outputs[0].text.strip()
    result = {
        "problem": record["problem"],
        "setup_a": record["setup_a"],
        "setup_b": record["setup_b"],
        "expected": record["completion"],
        "base": base,
        "adapter": adapted,
        "adapter_loaded": adapted in {"A", "B"},
        "model": str(args.model.resolve()),
        "adapter_path": str(args.adapter.resolve()),
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
