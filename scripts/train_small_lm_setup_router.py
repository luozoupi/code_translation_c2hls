#!/usr/bin/env python3
"""LoRA-fine-tune and evaluate a small causal LM as a setup router."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_setup_router import _best_fixed_setup, _load, _sha256
from scripts.train_strengthened_setup_router import (
    _ranking_metrics_from_scores,
    _ranking_outcomes,
    _record_id,
)


SCHEMA_VERSION = "c2hls.small-lm-setup-router-lora.v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


class CompletionDataset(Dataset):
    def __init__(
        self,
        records: list[dict[str, Any]],
        tokenizer: Any,
        *,
        max_length: int,
    ):
        self.examples = []
        for record in records:
            prompt_ids = tokenizer(
                record["prompt"],
                add_special_tokens=True,
                truncation=True,
                max_length=max_length - 1,
            )["input_ids"]
            answer_ids = tokenizer.encode(
                record["completion"],
                add_special_tokens=False,
            )
            if len(answer_ids) != 1:
                raise ValueError(
                    f"router answer is not one token: {answer_ids}"
                )
            input_ids = [
                *prompt_ids,
                answer_ids[0],
            ]
            labels = [
                *([-100] * len(prompt_ids)),
                answer_ids[0],
            ]
            self.examples.append(
                {
                    "input_ids": input_ids,
                    "labels": labels,
                    "record_id": record["record_id"],
                }
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.examples[index]


class CompletionCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(
        self,
        examples: list[dict[str, Any]],
    ) -> dict[str, torch.Tensor]:
        maximum = max(len(example["input_ids"]) for example in examples)
        input_ids = []
        attention_mask = []
        labels = []
        for example in examples:
            padding = maximum - len(example["input_ids"])
            input_ids.append(
                example["input_ids"]
                + [self.pad_token_id] * padding
            )
            attention_mask.append(
                [1] * len(example["input_ids"]) + [0] * padding
            )
            labels.append(example["labels"] + [-100] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(
                attention_mask,
                dtype=torch.long,
            ),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def _pair_margins(
    model: torch.nn.Module,
    tokenizer: Any,
    records: list[dict[str, Any]],
    *,
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    selected = [
        record for record in records if record["orientation"] == 0
    ]
    token_a = tokenizer.encode("A", add_special_tokens=False)
    token_b = tokenizer.encode("B", add_special_tokens=False)
    if len(token_a) != 1 or len(token_b) != 1:
        raise ValueError("A and B must each tokenize to one token")
    output = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(selected), batch_size):
            batch = selected[start : start + batch_size]
            encoded = tokenizer(
                [record["prompt"] for record in batch],
                add_special_tokens=True,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {
                name: value.to(device)
                for name, value in encoded.items()
            }
            logits = model(**encoded).logits
            positions = encoded["attention_mask"].sum(dim=1) - 1
            rows = torch.arange(len(batch), device=device)
            next_logits = logits[rows, positions]
            margins = (
                next_logits[:, token_a[0]]
                - next_logits[:, token_b[0]]
            ).float().cpu().numpy()
            for record, margin in zip(batch, margins, strict=True):
                output.append(
                    {
                        "record_id": record["record_id"],
                        "problem": record["problem"],
                        "split": record["split"],
                        "setup_a": record["setup_a"],
                        "setup_b": record["setup_b"],
                        "expected": record["completion"],
                        "predicted": "A" if margin >= 0 else "B",
                        "margin_a_minus_b": float(margin),
                    }
                )
    return output


def _router_metrics(
    pair_predictions: list[dict[str, Any]],
    outcomes: list[dict[str, Any]],
    *,
    split: str,
    best_fixed_setup: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    setup_utilities: dict[str, dict[str, float]] = {}
    correct = 0
    for prediction in pair_predictions:
        problem = prediction["problem"]
        utilities = setup_utilities.setdefault(problem, {})
        first = prediction["setup_a"]
        second = prediction["setup_b"]
        margin = float(prediction["margin_a_minus_b"])
        utilities[first] = utilities.get(first, 0.0) + margin
        utilities[second] = utilities.get(second, 0.0) - margin
        correct += int(
            prediction["predicted"] == prediction["expected"]
        )

    score_map = {}
    for record in outcomes:
        if record["split"] != split:
            continue
        utility = setup_utilities.get(record["problem"], {}).get(
            record["setup"]["setup_id"],
            0.0,
        )
        score_map[_record_id(record)] = -float(utility)
    metrics, router_predictions = _ranking_metrics_from_scores(
        outcomes,
        score_map,
        split=split,
        best_fixed_setup=best_fixed_setup,
    )
    metrics["pairwise_accuracy"] = (
        correct / len(pair_predictions) if pair_predictions else None
    )
    metrics["pairwise_comparisons"] = len(pair_predictions)
    return metrics, router_predictions


def _evaluate(
    model: torch.nn.Module,
    tokenizer: Any,
    dataset_dir: Path,
    outcomes: list[dict[str, Any]],
    *,
    best_fixed_setup: str,
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metrics = {}
    predictions = []
    for split in ("validation", "test"):
        records = _read_jsonl(dataset_dir / f"{split}.jsonl")
        pair_predictions = _pair_margins(
            model,
            tokenizer,
            records,
            max_length=max_length,
            batch_size=batch_size,
            device=device,
        )
        split_metrics, router_predictions = _router_metrics(
            pair_predictions,
            outcomes,
            split=split,
            best_fixed_setup=best_fixed_setup,
        )
        metrics[split] = split_metrics
        predictions.extend(
            {"split": split, "kind": "pair", **record}
            for record in pair_predictions
        )
        predictions.extend(
            {"split": split, "kind": "setup", **record}
            for record in router_predictions
        )
    return metrics, predictions


def _train_lora(
    model: torch.nn.Module,
    tokenizer: Any,
    records: list[dict[str, Any]],
    args: argparse.Namespace,
    *,
    device: torch.device,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[torch.nn.Module, list[dict[str, Any]]]:
    # This BF16 model does not use GPTQ. An incompatible optional GPTQ
    # installation must not block PEFT's ordinary torch Linear dispatcher.
    import peft.tuners.lora.gptq as peft_gptq

    peft_gptq.is_gptqmodel_available = lambda: False
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, config)
    model.config.use_cache = False
    dataset = CompletionDataset(
        records,
        tokenizer,
        max_length=args.max_length,
    )
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=CompletionCollator(tokenizer.pad_token_id),
        num_workers=0,
        pin_memory=True,
    )
    parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    update_steps = math.ceil(
        len(loader) / args.gradient_accumulation
    ) * args.epochs
    warmup_steps = max(1, int(update_steps * args.warmup_ratio))

    def learning_rate_multiplier(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        remaining = max(update_steps - warmup_steps, 1)
        return max((update_steps - step) / remaining, 0.0)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        learning_rate_multiplier,
    )
    logs = []
    token_a = tokenizer.encode("A", add_special_tokens=False)
    token_b = tokenizer.encode("B", add_special_tokens=False)
    if len(token_a) != 1 or len(token_b) != 1:
        raise ValueError("A and B must each tokenize to one token")
    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    model.train()
    for epoch in range(args.epochs):
        rolling_loss = 0.0
        for batch_index, batch in enumerate(loader):
            batch = {
                name: value.to(device, non_blocking=True)
                for name, value in batch.items()
            }
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
            )
            answer_mask = batch["labels"].ne(-100)
            answer_positions = answer_mask.to(torch.int64).argmax(dim=1)
            prompt_positions = answer_positions - 1
            rows = torch.arange(
                len(answer_positions),
                device=device,
            )
            answer_ids = batch["labels"][rows, answer_positions]
            targets = answer_ids.eq(token_a[0]).to(torch.float32)
            decision_logits = outputs.logits[rows, prompt_positions]
            margins = (
                decision_logits[:, token_a[0]]
                - decision_logits[:, token_b[0]]
            ).to(torch.float32)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                margins,
                targets,
            )
            (loss / args.gradient_accumulation).backward()
            rolling_loss += float(loss.detach())
            should_update = (
                (batch_index + 1) % args.gradient_accumulation == 0
                or batch_index + 1 == len(loader)
            )
            if not should_update:
                continue
            torch.nn.utils.clip_grad_norm_(
                parameters,
                args.max_grad_norm,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            if (
                global_step == 1
                or global_step % args.log_every == 0
                or global_step == update_steps
            ):
                log_record = {
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "update_steps": update_steps,
                    "loss": rolling_loss
                    / min(
                        args.gradient_accumulation,
                        batch_index + 1,
                    ),
                    "learning_rate": scheduler.get_last_lr()[0],
                }
                logs.append(log_record)
                if progress_callback is not None:
                    progress_callback(log_record)
            rolling_loss = 0.0
    model.save_pretrained(args.adapter_dir)
    tokenizer.save_pretrained(args.adapter_dir)
    return model, logs


def train(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the small-LM router run")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device)

    outcomes = _ranking_outcomes(_load(args.corpus))
    train_ranking = [
        record
        for record in outcomes
        if record["split"] == "train" and record["labels"]["valid"]
    ]
    best_fixed, _ = _best_fixed_setup(train_ranking)
    base_metrics, base_predictions = _evaluate(
        model,
        tokenizer,
        args.dataset_dir,
        outcomes,
        best_fixed_setup=best_fixed,
        max_length=args.max_length,
        batch_size=args.eval_batch_size,
        device=device,
    )

    train_records = _read_jsonl(args.dataset_dir / "train.jsonl")
    model, training_logs = _train_lora(
        model,
        tokenizer,
        train_records,
        args,
        device=device,
    )
    sft_metrics, sft_predictions = _evaluate(
        model,
        tokenizer,
        args.dataset_dir,
        outcomes,
        best_fixed_setup=best_fixed,
        max_length=args.max_length,
        batch_size=args.eval_batch_size,
        device=device,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    compact_config = {
        "model": str(args.model.resolve()),
        "model_config_sha256": _sha256(args.model / "config.json"),
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "max_length": args.max_length,
        "seed": args.seed,
        "visible_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "loss_objective": "binary_a_minus_b_margin",
    }
    output = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "configuration": compact_config,
        "corpus": {
            "path": str(args.corpus.resolve()),
            "sha256": _sha256(args.corpus),
        },
        "dataset_manifest": json.loads(
            (args.dataset_dir / "manifest.json").read_text(
                encoding="utf-8"
            )
        ),
        "adapter": {
            "path": str(args.adapter_dir.resolve()),
            "adapter_config_sha256": _sha256(
                args.adapter_dir / "adapter_config.json"
            ),
        },
        "base": base_metrics,
        "lora_sft": sft_metrics,
        "training_logs": training_logs,
        "methodology": {
            "selection_split": "validation",
            "test_reuse_caveat": (
                "historical test lineages were exposed by prior router "
                "studies; this is architecture comparison only"
            ),
            "reference_metrics_as_inputs": False,
            "candidate_outcomes_as_inputs": False,
            "pairwise_decoding": "next-token A versus B logit margin",
        },
    }
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "predictions.jsonl").open(
        "w",
        encoding="utf-8",
    ) as handle:
        for model_name, predictions in (
            ("base", base_predictions),
            ("lora_sft", sft_predictions),
        ):
            for prediction in predictions:
                handle.write(
                    json.dumps(
                        {"model": model_name, **prediction},
                        sort_keys=True,
                    )
                    + "\n"
                )
    (args.output_dir / "training_log.jsonl").write_text(
        "".join(
            json.dumps(record, sort_keys=True) + "\n"
            for record in training_logs
        ),
        encoding="utf-8",
    )
    report_path = args.output_dir / "report.md"
    report_path.write_text(
        "\n".join(
            [
                "# Qwen3-0.6B LoRA Setup Router",
                "",
                "| model | split | pair accuracy | within 5% | geomean regret |",
                "|---|---|---:|---:|---:|",
                *[
                    (
                        f"| {model_name} | {split} | "
                        f"{metrics[split]['pairwise_accuracy']:.3f} | "
                        f"{metrics[split]['learned_top_k']['within_5pct_coverage']:.3f} | "
                        f"{metrics[split]['learned_top_k']['geomean_regret']:.3f} |"
                    )
                    for model_name, metrics in (
                        ("base", base_metrics),
                        ("LoRA SFT", sft_metrics),
                    )
                    for split in ("validation", "test")
                ],
                "",
                "The historical test set is not pristine. Promotion must "
                "use corrected-v2 outcomes and an unexposed confirmation set.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=46)
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    output = train(parse_args())
    print(
        json.dumps(
            {
                "base": output["base"],
                "lora_sft": output["lora_sft"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
