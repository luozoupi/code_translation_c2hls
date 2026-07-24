#!/usr/bin/env python3
"""LoRA / QLoRA SFT for C→HLS chat data (TRL SFTTrainer).

Devstral serve checkpoints on disk are often FineGrained FP8 (inference-only).
Those cannot be trained with BitsAndBytes QLoRA. This script:
  - detects FP8 configs and dequantizes to bf16 for LoRA training
  - otherwise uses BitsAndBytes 4-bit QLoRA when --no-4bit is not set

Mistral/Devstral chat templates reject a trailing assistant message unless
`continue_final_message=True`. We therefore render prompt/completion *strings*
before handing data to TRL (avoids the serving-oriented validator error).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    FineGrainedFP8Config,
)
from trl import SFTConfig, SFTTrainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="HF id or local path")
    p.add_argument(
        "--init-adapter",
        type=Path,
        default=None,
        help="Optional prior LoRA adapter to merge before training a NEW LoRA "
        "(does not modify the prior adapter files).",
    )
    p.add_argument("--train-file", type=Path, required=True)
    p.add_argument("--val-file", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-seq-length", type=int, default=8192)
    p.add_argument("--num-train-epochs", type=float, default=2.0)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--per-device-train-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--logging-steps", type=int, default=5)
    p.add_argument("--save-steps", type=int, default=50)
    p.add_argument("--eval-steps", type=int, default=50)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument(
        "--target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated LoRA modules (Dense Llama/Mistral-style).",
    )
    p.add_argument("--load-in-4bit", action="store_true", default=True)
    p.add_argument("--no-4bit", action="store_true", help="Disable BitsAndBytes 4-bit.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--attn-implementation",
        default="sdpa",
        choices=["sdpa", "eager", "flash_attention_2"],
    )
    args = p.parse_args()
    if args.init_adapter is None and os.environ.get("INIT_ADAPTER"):
        args.init_adapter = Path(os.environ["INIT_ADAPTER"])
    return args


def _model_is_fp8(model_path: str) -> bool:
    try:
        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        q = getattr(cfg, "quantization_config", None)
        if q is None:
            return False
        if isinstance(q, dict):
            return str(q.get("quant_method", "")).lower() == "fp8"
        return str(getattr(q, "quant_method", "")).lower() == "fp8"
    except Exception:
        p = Path(model_path) / "config.json"
        if p.exists():
            raw = json.loads(p.read_text())
            q = raw.get("quantization_config") or {}
            return str(q.get("quant_method", "")).lower() == "fp8"
        return False


DEFAULT_SYSTEM = (
    "You are an expert in Xilinx Vitis HLS. Convert the given plain C/C++ kernel "
    "into synthesizable, high-performance Vitis HLS. Preserve correctness; prefer "
    "lower latency when possible. Return a complete HLS source in a single "
    "```cpp code fence."
)


def _as_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                parts.append(str(block["text"]))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content)


def _sanitize_prompt_messages(messages: list[dict]) -> list[dict]:
    """Collapse agentic multi-turn history to Mistral-legal [system, user]."""
    systems = [_as_text(m["content"]) for m in messages if m.get("role") == "system" and m.get("content")]
    users = [_as_text(m["content"]) for m in messages if m.get("role") == "user" and m.get("content")]
    system = systems[0] if systems else DEFAULT_SYSTEM
    if not users:
        raise ValueError("SFT example needs at least one user message")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": users[-1]},
    ]


def _messages_to_prompt_completion(tokenizer, messages: list[dict]) -> dict[str, str]:
    """Render chat into plain prompt/completion strings (Mistral-safe)."""
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("SFT example must end with an assistant message")
    assistant = {"role": "assistant", "content": _as_text(messages[-1].get("content", ""))}
    prompt_msgs = _sanitize_prompt_messages(messages[:-1] if messages[:-1] else messages)
    # If original prompt already system+user only, sanitize is a no-op-ish collapse.
    prompt = tokenizer.apply_chat_template(
        prompt_msgs,
        tokenize=False,
        add_generation_prompt=True,
    )
    full = tokenizer.apply_chat_template(
        prompt_msgs + [assistant],
        tokenize=False,
        continue_final_message=True,
    )
    # Align prefix (template whitespace / special-token quirks).
    i = 0
    for a, b in zip(prompt, full):
        if a != b:
            break
        i += 1
    prompt = full[:i]
    completion = full[i:]
    if not completion:
        completion = assistant["content"]
    return {"prompt": prompt, "completion": completion}


def _prepare_prompt_completion_dataset(ds, tokenizer):
    def _map(example):
        if "prompt" in example and "completion" in example and isinstance(example["prompt"], str):
            return {"prompt": example["prompt"], "completion": example["completion"]}
        messages = example.get("messages")
        if not messages:
            raise ValueError("Example needs messages or string prompt/completion")
        return _messages_to_prompt_completion(tokenizer, messages)

    cols = ds["train"].column_names
    return ds.map(_map, remove_columns=cols)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    is_fp8 = _model_is_fp8(args.model)
    use_4bit = args.load_in_4bit and not args.no_4bit and not is_fp8
    if is_fp8:
        print(
            "Detected FineGrained FP8 checkpoint — dequantizing to bf16 for LoRA "
            "(BitsAndBytes QLoRA is incompatible with this serve pack)."
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: dict = {
        "trust_remote_code": True,
        "attn_implementation": args.attn_implementation,
        "device_map": "auto",
        "dtype": torch.bfloat16,
    }
    if is_fp8:
        load_kwargs["quantization_config"] = FineGrainedFP8Config(dequantize=True)
    elif use_4bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    model.config.use_cache = False

    if args.init_adapter and Path(args.init_adapter).exists():
        print(f"Merging prior adapter (read-only) from {args.init_adapter}")
        model = PeftModel.from_pretrained(model, str(args.init_adapter))
        model = model.merge_and_unload()
        print("Prior adapter merged into base weights; training a NEW LoRA on top.")

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[m.strip() for m in args.target_modules.split(",") if m.strip()],
    )

    data_files = {"train": str(args.train_file)}
    if args.val_file and args.val_file.exists():
        data_files["validation"] = str(args.val_file)
    ds = load_dataset("json", data_files=data_files)
    print("Rendering Mistral-safe prompt/completion strings…")
    ds = _prepare_prompt_completion_dataset(ds, tokenizer)
    print("Sample prompt tail:", repr(ds["train"][0]["prompt"][-120:]))
    print("Sample completion head:", repr(ds["train"][0]["completion"][:120]))

    training_args = SFTConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy="steps" if "validation" in ds else "no",
        eval_steps=args.eval_steps if "validation" in ds else None,
        warmup_ratio=args.warmup_ratio,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=args.max_seq_length,
        packing=False,
        report_to=os.environ.get("REPORT_TO", "none"),
        seed=args.seed,
        save_total_limit=2,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit" if use_4bit else "adamw_torch",
        completion_only_loss=True,
        # Avoid TRL 1.8 chunked_nll patch bug with accelerate/device_map partials on Devstral.
        loss_type="nll",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds.get("validation"),
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(args.output_dir / "adapter"))
    tokenizer.save_pretrained(str(args.output_dir / "adapter"))
    print(f"Saved LoRA adapter to {args.output_dir / 'adapter'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
