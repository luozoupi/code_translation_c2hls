#!/usr/bin/env python3
"""Offline DPO on top of an SFT LoRA adapter (TRL DPOTrainer).

Expects prepared DPO JSONL with either:
  - conversational: prompt/chosen/rejected as message lists
  - plain strings

FP8 serve checkpoints are dequantized to bf16 (same policy as train_sft.py).
Agentic multi-turn prompts are sanitized to Mistral-legal [system, user] before
templating (Mistral rejects system-after-assistant histories).
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
from trl import DPOConfig, DPOTrainer

DEFAULT_SYSTEM = (
    "You are an expert in Xilinx Vitis HLS. Convert the given plain C/C++ kernel "
    "into synthesizable, high-performance Vitis HLS for the AMD Alveo U280 "
    "(xcu280-fsvh2892-2L-e). Preserve correctness; prefer lower latency when "
    "possible. Return a complete HLS source in a single ```cpp code fence."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="Base HF id or local path")
    p.add_argument(
        "--sft-adapter",
        type=Path,
        default=None,
        help="Optional SFT LoRA adapter dir to continue from",
    )
    p.add_argument("--train-file", type=Path, required=True)
    p.add_argument("--val-file", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-length", type=int, default=4096,
                   help="DPO needs chosen+rejected forwards; 4096 is safer than 8192 on 4×H100 bf16.")
    p.add_argument("--num-train-epochs", type=float, default=1.0)
    p.add_argument("--learning-rate", type=float, default=5e-6)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--per-device-train-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--logging-steps", type=int, default=5)
    p.add_argument("--save-steps", type=int, default=50)
    p.add_argument("--eval-steps", type=int, default=50)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument(
        "--target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    p.add_argument("--no-4bit", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--attn-implementation",
        default="sdpa",
        choices=["sdpa", "eager", "flash_attention_2"],
    )
    return p.parse_args()


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


def _as_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # rare multimodal-ish content blocks
        parts = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                parts.append(str(block["text"]))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content)


def _sanitize_prompt_messages(prompt) -> list[dict]:
    """Collapse agentic histories to a Mistral-legal [system, user] pair."""
    if isinstance(prompt, str):
        return [
            {"role": "system", "content": DEFAULT_SYSTEM},
            {"role": "user", "content": prompt},
        ]
    systems = [_as_text(m["content"]) for m in prompt if m.get("role") == "system" and m.get("content")]
    users = [_as_text(m["content"]) for m in prompt if m.get("role") == "user" and m.get("content")]
    system = systems[0] if systems else DEFAULT_SYSTEM
    if not users:
        # fall back: dump non-assistant text
        blob = "\n\n".join(
            _as_text(m.get("content", ""))
            for m in prompt
            if m.get("role") != "assistant" and m.get("content")
        )
        users = [blob or "Convert the kernel to Vitis HLS."]
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": users[-1]},
    ]


def _assistant_messages(side) -> list[dict]:
    if isinstance(side, str):
        text = side.strip()
        if not text.startswith("```"):
            text = f"```cpp\n{text}\n```"
        return [{"role": "assistant", "content": text}]
    if isinstance(side, list) and side:
        last = side[-1]
        return [{"role": "assistant", "content": _as_text(last.get("content", ""))}]
    raise ValueError("chosen/rejected must be string or message list")


def _common_prefix_len(a: str, b: str) -> int:
    i = 0
    for x, y in zip(a, b):
        if x != y:
            break
        i += 1
    return i


def _render_dpo_row(tokenizer, row: dict) -> dict:
    prompt_msgs = _sanitize_prompt_messages(row["prompt"])
    chosen_msgs = _assistant_messages(row["chosen"])
    rejected_msgs = _assistant_messages(row["rejected"])

    prompt = tokenizer.apply_chat_template(
        prompt_msgs,
        tokenize=False,
        add_generation_prompt=True,
    )
    chosen_full = tokenizer.apply_chat_template(
        prompt_msgs + chosen_msgs,
        tokenize=False,
        continue_final_message=True,
    )
    rejected_full = tokenizer.apply_chat_template(
        prompt_msgs + rejected_msgs,
        tokenize=False,
        continue_final_message=True,
    )
    # Align prompts to the common prefix with each completion (template quirks).
    i_c = _common_prefix_len(prompt, chosen_full)
    i_r = _common_prefix_len(prompt, rejected_full)
    # Use the shorter shared prefix so both completions are consistent with one prompt.
    i = min(i_c, i_r)
    prompt = chosen_full[:i] if i_c <= i_r else rejected_full[:i]
    # Recompute against aligned prompt
    i_c = _common_prefix_len(prompt, chosen_full)
    i_r = _common_prefix_len(prompt, rejected_full)
    chosen = chosen_full[i_c:]
    rejected = rejected_full[i_r:]
    if not chosen:
        chosen = chosen_msgs[0]["content"]
    if not rejected:
        rejected = rejected_msgs[0]["content"]
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


def _prepare_dpo_dataset(ds, tokenizer):
    def _map(example):
        return _render_dpo_row(tokenizer, example)

    # Keep only rendered fields for TRL.
    cols = ds["train"].column_names
    return ds.map(_map, remove_columns=cols)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    is_fp8 = _model_is_fp8(args.model)
    use_4bit = (not args.no_4bit) and (not is_fp8)
    if is_fp8:
        print(
            "Detected FineGrained FP8 checkpoint — dequantizing to bf16 for LoRA/DPO."
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

    if args.sft_adapter and args.sft_adapter.exists():
        model = PeftModel.from_pretrained(model, str(args.sft_adapter))
        model = model.merge_and_unload()
        print(f"Merged SFT adapter from {args.sft_adapter}")

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
    print("Rendering Mistral-safe DPO prompt/chosen/rejected strings…")
    ds = _prepare_dpo_dataset(ds, tokenizer)
    print("Sample prompt tail:", repr(ds["train"][0]["prompt"][-100:]))
    print("Sample chosen head:", repr(ds["train"][0]["chosen"][:100]))

    training_args = DPOConfig(
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
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        beta=args.beta,
        max_length=args.max_length,
        # Avoid keeping a live ref forward during training (major VRAM saver on 123B).
        precompute_ref_log_probs=True,
        report_to=os.environ.get("REPORT_TO", "none"),
        seed=args.seed,
        save_total_limit=2,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit" if use_4bit else "adamw_torch",
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
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
    print(f"Saved DPO LoRA adapter to {args.output_dir / 'adapter'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
