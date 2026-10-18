#!/usr/bin/env python3
"""Run local completion-only QLoRA on the model-aware C2HLS corpus.

The current implementation supports the local Qwen3.6-27B checkpoint. It uses
the conditional-generation wrapper, freezes the vision tower, and injects LoRA
only into language-model projections. The source Parquet is already tokenized;
shorter sequences are made by trimming prompt tokens while preserving the
complete assistant target and its completion mask. Training can be restricted
to one assistant turn and evaluated on a benchmark-disjoint held-out split.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import math
import os
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = Path(
    "/mnt/data/vllm_models/hub/models--Qwen--Qwen3.6-27B/"
    "snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
)
DEFAULT_DATASET_DIR = Path(
    "/mnt/data2/luo00466/c2hls_sft/corpus/qwen3.6-27b_max8192"
)
DEFAULT_OUTPUT_ROOT = Path("/mnt/data2/luo00466/c2hls_sft/runs")
SMOKE_TRUNCATION_MARKER = "\n\n<SMOKE_PROMPT_TRUNCATED>\n\n"

LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_a",
    "in_proj_b",
    "out_proj",
]


class SmokeRecordRejected(ValueError):
    pass


def cap_pretokenized_record(
    record: dict[str, Any],
    *,
    max_length: int,
    min_prompt_tokens: int,
    marker_ids: list[int],
    head_fraction: float = 0.40,
) -> dict[str, Any]:
    input_ids = [int(value) for value in record["input_ids"]]
    completion_mask = [int(value) for value in record["completion_mask"]]
    if len(input_ids) != len(completion_mask):
        raise SmokeRecordRejected("input IDs and completion mask have different lengths")
    try:
        completion_start = completion_mask.index(1)
    except ValueError as exc:
        raise SmokeRecordRejected("record has no completion tokens") from exc
    if completion_start == 0 or any(completion_mask[index] != 1 for index in range(completion_start, len(input_ids))):
        raise SmokeRecordRejected("completion mask is not one contiguous suffix")

    prompt_ids = input_ids[:completion_start]
    completion_ids = input_ids[completion_start:]
    if len(completion_ids) + min_prompt_tokens > max_length:
        raise SmokeRecordRejected(
            f"completion has {len(completion_ids)} tokens and cannot fit the smoke budget"
        )

    prompt_budget = max_length - len(completion_ids)
    if len(prompt_ids) > prompt_budget:
        retained_budget = prompt_budget - len(marker_ids)
        if retained_budget < min_prompt_tokens:
            raise SmokeRecordRejected("truncation marker leaves too little prompt context")
        head_tokens = max(1, math.floor(retained_budget * head_fraction))
        tail_tokens = retained_budget - head_tokens
        prompt_ids = prompt_ids[:head_tokens] + marker_ids
        if tail_tokens:
            prompt_ids += input_ids[completion_start - tail_tokens : completion_start]

    capped_ids = prompt_ids + completion_ids
    capped_mask = [0] * len(prompt_ids) + [1] * len(completion_ids)
    output = dict(record)
    output.update(
        {
            "input_ids": capped_ids,
            "completion_mask": capped_mask,
            "token_count": len(capped_ids),
            "prompt_token_count": len(prompt_ids),
            "completion_token_count": len(completion_ids),
            "smoke_prompt_truncated": len(capped_ids) < len(input_ids),
            "smoke_original_token_count": len(input_ids),
        }
    )
    return output


def _parse_args() -> argparse.Namespace:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / f"qwen3.6-27b_qlora_r16_smoke10_{timestamp}",
    )
    parser.add_argument("--gpu", type=int, required=True, help="Physical GPU index before CUDA remapping.")
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--max-train-samples", type=int, default=32)
    parser.add_argument("--max-eval-samples", type=int, default=32)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="val")
    parser.add_argument(
        "--assistant-turn-index",
        type=int,
        help="Train and evaluate only this assistant turn (0 is initial translation).",
    )
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--min-prompt-tokens", type=int, default=256)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, help="Defaults to two times the LoRA rank.")
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lr-scheduler-type", default="constant")
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-free-gib", type=float, default=80.0)
    parser.add_argument("--load-only", action="store_true")
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Evaluate an untrained, zero-initialized LoRA without optimizer updates.",
    )
    return parser.parse_args()


def _configure_process(args: argparse.Namespace) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["WANDB_DISABLED"] = "true"
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(args.output_dir / "run.log"),
        ],
        force=True,
    )


def _gpu_snapshot(torch: Any) -> dict[str, Any]:
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    return {
        "visible_index": 0,
        "name": torch.cuda.get_device_name(0),
        "free_gib": free_bytes / 2**30,
        "total_gib": total_bytes / 2**30,
        "allocated_gib": torch.cuda.memory_allocated(0) / 2**30,
        "reserved_gib": torch.cuda.memory_reserved(0) / 2**30,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(0) / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(0) / 2**30,
    }


def _patch_irrelevant_broken_gptq_dispatch() -> bool:
    """Disable GPTQ dispatch when GPTQModel exists without its Optimum dependency."""
    if importlib.util.find_spec("gptqmodel") is None or importlib.util.find_spec("optimum") is not None:
        return False
    import peft.tuners.lora.gptq as peft_gptq

    peft_gptq.is_gptqmodel_available = lambda: False
    return True


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def _prepare_dataset(
    args: argparse.Namespace,
    tokenizer: Any,
    *,
    split: str,
    max_samples: int,
    shuffle_seed: int,
) -> tuple[Any, dict[str, Any]]:
    from datasets import load_dataset

    split_path = args.dataset_dir / f"{split}.parquet"
    if not split_path.is_file():
        raise FileNotFoundError(f"dataset split does not exist: {split_path}")
    dataset = load_dataset(
        "parquet",
        data_files={split: str(split_path)},
        split=split,
    )
    source_rows = len(dataset)
    if args.assistant_turn_index is not None:
        if "assistant_turn_index" not in dataset.column_names:
            raise RuntimeError(
                "assistant-turn filtering requested but the dataset has no "
                "assistant_turn_index column"
            )
        dataset = dataset.filter(
            lambda record: int(record["assistant_turn_index"])
            == args.assistant_turn_index,
            desc=f"Filtering {split} assistant turn {args.assistant_turn_index}",
        )
    filtered_rows = len(dataset)
    marker_ids = tokenizer.encode(SMOKE_TRUNCATION_MARKER, add_special_tokens=False)
    eligible = dataset.filter(
        lambda record: int(record["token_count"]) <= args.sequence_length
        or int(record["completion_token_count"])
        + args.min_prompt_tokens
        + len(marker_ids)
        <= args.sequence_length,
        desc=f"Filtering {split} sequence-compatible completions",
    )
    if not len(eligible):
        raise RuntimeError(
            f"no {split} examples can fit the requested sequence length"
        )
    selected = eligible.shuffle(seed=shuffle_seed)
    if max_samples > 0:
        selected = selected.select(range(min(max_samples, len(selected))))
    selected_ids = list(selected["record_id"])
    selected = selected.map(
        lambda record: cap_pretokenized_record(
            record,
            max_length=args.sequence_length,
            min_prompt_tokens=args.min_prompt_tokens,
            marker_ids=marker_ids,
        ),
        desc=f"Capping {split} prompts without truncating completions",
    )
    lengths = list(selected["token_count"])
    completion_lengths = list(selected["completion_token_count"])
    details = {
        "split": split,
        "source_rows": source_rows,
        "assistant_turn_index": args.assistant_turn_index,
        "turn_filtered_rows": filtered_rows,
        "eligible_rows": len(eligible),
        "selected_rows": len(selected),
        "selected_record_ids": selected_ids,
        "benchmarks": sorted(set(selected["benchmark"])),
        "token_min": min(lengths),
        "token_max": max(lengths),
        "completion_token_min": min(completion_lengths),
        "completion_token_max": max(completion_lengths),
        "truncated_rows": sum(bool(value) for value in selected["smoke_prompt_truncated"]),
    }
    return selected, details


def _run(args: argparse.Namespace) -> dict[str, Any]:
    import bitsandbytes
    import peft
    import torch
    import transformers
    import trl
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForImageTextToText,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainerCallback,
        set_seed,
    )
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"expected exactly one visible CUDA device, found {torch.cuda.device_count()}"
        )
    if args.eval_only and args.no_eval:
        raise ValueError("--eval-only cannot be combined with --no-eval")
    start_gpu = _gpu_snapshot(torch)
    if start_gpu["free_gib"] < args.min_free_gib:
        raise RuntimeError(
            f"GPU has only {start_gpu['free_gib']:.1f} GiB free; required {args.min_free_gib:.1f} GiB"
        )

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint,
        local_files_only=True,
        trust_remote_code=False,
    )
    train_dataset, train_details = _prepare_dataset(
        args,
        tokenizer,
        split=args.train_split,
        max_samples=args.max_train_samples,
        shuffle_seed=args.seed,
    )
    eval_dataset = None
    eval_details = None
    if not args.no_eval:
        eval_dataset, eval_details = _prepare_dataset(
            args,
            tokenizer,
            split=args.eval_split,
            max_samples=args.max_eval_samples,
            shuffle_seed=args.seed + 1,
        )
    benchmark_overlap = sorted(
        set(train_details["benchmarks"])
        & set(eval_details["benchmarks"] if eval_details else [])
    )
    if benchmark_overlap:
        logging.warning(
            "Training and evaluation benchmark sets overlap: %s",
            benchmark_overlap,
        )
    logging.info("Selected training dataset: %s", json.dumps(train_details, sort_keys=True))
    if eval_details:
        logging.info("Selected evaluation dataset: %s", json.dumps(eval_details, sort_keys=True))

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    logging.info("Loading 4-bit Qwen checkpoint on physical GPU %d", args.gpu)
    load_started = time.monotonic()
    model = AutoModelForImageTextToText.from_pretrained(
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
    load_seconds = time.monotonic() - load_started
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = False

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    patched_gptq_dispatch = _patch_irrelevant_broken_gptq_dispatch()
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha or 2 * args.rank,
        lora_dropout=args.dropout,
        target_modules=LORA_TARGETS,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    visual_trainable = [name for name in trainable_names if ".visual." in name]
    if visual_trainable:
        raise RuntimeError(f"LoRA unexpectedly targeted the vision tower: {visual_trainable[:5]}")
    outside_language = [name for name in trainable_names if ".language_model." not in name]
    if outside_language:
        raise RuntimeError(f"trainable parameters exist outside the language model: {outside_language[:5]}")
    trainable_parameters, total_parameters = model.get_nb_trainable_parameters()
    model.print_trainable_parameters()
    after_load_gpu = _gpu_snapshot(torch)

    base_manifest = {
        "schema_version": "c2hls.agentic_qlora.v2",
        "status": (
            "model_loaded"
            if args.load_only
            else "evaluating_base" if args.eval_only else "training"
        ),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "physical_gpu": args.gpu,
        "checkpoint": str(args.checkpoint),
        "dataset_dir": str(args.dataset_dir),
        "output_dir": str(args.output_dir),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "trl": trl.__version__,
            "peft": peft.__version__,
            "bitsandbytes": bitsandbytes.__version__,
        },
        "qlora": {
            "rank": args.rank,
            "alpha": args.alpha or 2 * args.rank,
            "dropout": args.dropout,
            "quant_type": "nf4",
            "double_quant": True,
            "compute_dtype": "bfloat16",
            "targets": LORA_TARGETS,
            "target_module_count": sum("lora_A" in name for name in trainable_names),
            "trainable_parameters": trainable_parameters,
            "total_parameters": total_parameters,
            "trainable_fraction": trainable_parameters / total_parameters,
        },
        "training": {
            "max_steps": args.max_steps,
            "sequence_length": args.sequence_length,
            "min_prompt_tokens": args.min_prompt_tokens,
            "learning_rate": args.learning_rate,
            "lr_scheduler_type": args.lr_scheduler_type,
            "warmup_ratio": args.warmup_ratio,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "gradient_checkpointing": True,
            "completion_only_loss": True,
            "assistant_only_loss": False,
            "packing": False,
            "optimizer": "paged_adamw_8bit",
            "seed": args.seed,
            "assistant_turn_index": args.assistant_turn_index,
            "held_out_evaluation": eval_dataset is not None,
        },
        "dataset": {
            "train": train_details,
            "eval": eval_details,
            "benchmark_overlap": benchmark_overlap,
        },
        "gptq_dispatch_disabled": patched_gptq_dispatch,
        "model_load_seconds": load_seconds,
        "gpu_before_load": start_gpu,
        "gpu_after_load": after_load_gpu,
    }
    _write_json(args.output_dir / "run_manifest.json", base_manifest)
    if args.load_only:
        base_manifest["status"] = "load_only_complete"
        base_manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        _write_json(args.output_dir / "run_manifest.json", base_manifest)
        return base_manifest

    step_metrics_path = args.output_dir / "step_metrics.jsonl"

    class StepMetricsCallback(TrainerCallback):
        def on_log(self, training_args, state, control, logs=None, **kwargs):
            del training_args, control, kwargs
            row = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "step": state.global_step,
                "gpu": _gpu_snapshot(torch),
                **(logs or {}),
            }
            with step_metrics_path.open("a") as f:
                f.write(json.dumps(row, sort_keys=True, default=str) + "\n")

    training_args = SFTConfig(
        output_dir=str(args.output_dir / "trainer"),
        do_train=not args.eval_only,
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        optim="paged_adamw_8bit",
        bf16=True,
        fp16=False,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_strategy="steps",
        logging_steps=1,
        logging_first_step=True,
        save_strategy="no",
        eval_strategy="no",
        do_eval=eval_dataset is not None,
        report_to="none",
        max_length=args.sequence_length,
        completion_only_loss=True,
        assistant_only_loss=False,
        packing=False,
        shuffle_dataset=False,
        remove_unused_columns=True,
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        seed=args.seed,
        data_seed=args.seed,
        disable_tqdm=False,
        skip_memory_metrics=False,
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=[StepMetricsCallback()],
    )
    if args.eval_only:
        eval_started = time.monotonic()
        eval_metrics = trainer.evaluate(metric_key_prefix="eval")
        eval_seconds = time.monotonic() - eval_started
        trainer.save_metrics("eval", eval_metrics)
        base_manifest.update(
            {
                "status": "eval_only_complete",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "eval_seconds": eval_seconds,
                "eval_metrics": eval_metrics,
                "gpu_after_eval": _gpu_snapshot(torch),
            }
        )
        _write_json(args.output_dir / "run_manifest.json", base_manifest)
        return base_manifest

    train_started = time.monotonic()
    train_result = trainer.train()
    train_seconds = time.monotonic() - train_started
    eval_metrics = None
    eval_seconds = None
    if eval_dataset is not None:
        eval_started = time.monotonic()
        eval_metrics = trainer.evaluate(metric_key_prefix="eval")
        eval_seconds = time.monotonic() - eval_started
        trainer.save_metrics("eval", eval_metrics)
    adapter_dir = args.output_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(adapter_dir)
    trainer.save_state()
    trainer.save_metrics("train", train_result.metrics)

    base_manifest.update(
        {
            "status": "complete",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "train_seconds": train_seconds,
            "train_metrics": train_result.metrics,
            "eval_seconds": eval_seconds,
            "eval_metrics": eval_metrics,
            "adapter_dir": str(adapter_dir),
            "gpu_after_train": _gpu_snapshot(torch),
        }
    )
    _write_json(args.output_dir / "run_manifest.json", base_manifest)
    return base_manifest


def main() -> int:
    args = _parse_args()
    _configure_process(args)
    logging.info("Starting QLoRA smoke with arguments: %s", vars(args))
    try:
        result = _run(args)
    except Exception as exc:
        failure = {
            "schema_version": "c2hls.agentic_qlora_smoke_failure.v1",
            "status": "failed",
            "failed_at": datetime.now(timezone.utc).isoformat(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "arguments": vars(args),
        }
        _write_json(args.output_dir / "failure.json", failure)
        logging.exception("QLoRA smoke failed")
        return 1
    logging.info("QLoRA smoke finished with status %s", result["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
