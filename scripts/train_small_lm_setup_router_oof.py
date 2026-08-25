#!/usr/bin/env python3
"""Evaluate a prompted and LoRA-tuned small LM with corrected-v2 OOF folds."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_setup_router_topk_architectures import (
    DISPLAY_NAMES,
    _aggregate,
    _write_csv,
    evaluate_topk_scores,
)
from scripts.prepare_small_lm_setup_router_dataset import (
    PHASE_B_FEATURES,
    _canonical_json,
    _prompt,
    _source_code,
)
from scripts.train_setup_router import _load, _sha256
from scripts.train_small_lm_setup_router import (
    _pair_margins,
    _train_lora,
)
from scripts.train_strengthened_setup_router import (
    _canonicalize_phase_b,
    _first_wins,
    _outer_fold_records,
    _outcome_groups,
    _record_id,
)


SCHEMA_VERSION = "c2hls.small-lm-setup-router-oof.v1"
BASE_ROUTER = "qwen3_06b_prompted_base_oof"
LORA_ROUTER = "qwen3_06b_pairwise_lora_oof"
DISPLAY_NAMES.update(
    {
        BASE_ROUTER: "Qwen3-0.6B prompted base",
        LORA_ROUTER: "Qwen3-0.6B LoRA pairwise",
    }
)
DEFAULT_MODEL = Path(
    "/mnt/data/vllm_models/hub/models--Qwen--Qwen3-0.6B/"
    "snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
)
DEFAULT_CORPUS = (
    REPO_ROOT
    / "artifacts"
    / "setup_router"
    / "corpus_corrected_v2"
    / "setup_router_outcomes.jsonl"
)
DEFAULT_BENCHMARK_ROOT = (
    REPO_ROOT
    / "benchmarks_external"
    / "HLSFactory"
    / "polybench_float_small"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "artifacts"
    / "setup_router"
    / "qwen3_06b_lora_oof_corrected_v2_20260728"
)
DEFAULT_ADAPTER_ROOT = Path(
    "/mnt/data2/luo00466/c2hls_rl/setup_router/"
    "qwen3_06b_lora_oof_corrected_v2_20260728"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _heartbeat(
    args: argparse.Namespace,
    *,
    fold: int,
    phase: str,
    progress: dict[str, Any] | None = None,
) -> None:
    payload = {
        "schema_version": f"{SCHEMA_VERSION}.heartbeat",
        "updated_at": _utc_now(),
        "pid": os.getpid(),
        "state": "running",
        "fold": fold,
        "folds_total": 5,
        "phase": phase,
    }
    if progress:
        payload["progress"] = progress
    _write_json(args.output_dir / "heartbeat.json", payload)
    print(json.dumps(payload, sort_keys=True), flush=True)


def _pair_records(
    records: list[dict[str, Any]],
    *,
    benchmark_root: Path,
    split: str,
    both_orientations: bool,
) -> list[dict[str, Any]]:
    output = []
    for problem, outcomes in sorted(
        _outcome_groups(records, split=split).items()
    ):
        source, _ = _source_code(problem, benchmark_root)
        mandatory = next(
            record
            for record in outcomes
            if str(record["setup"]["setup_id"]).endswith(
                ":multistep:skillless"
            )
        )
        phase_b = {
            name: mandatory["features"].get(name)
            for name in PHASE_B_FEATURES
        }
        ordered = sorted(
            outcomes,
            key=lambda record: record["setup"]["setup_id"],
        )
        for first_index, first in enumerate(ordered):
            for second in ordered[first_index + 1 :]:
                first_wins = _first_wins(first, second)
                if first_wins is None:
                    continue
                winner = "A" if first_wins else "B"
                orientations = [
                    (first, second, winner, 0),
                ]
                if both_orientations:
                    orientations.append(
                        (
                            second,
                            first,
                            "B" if winner == "A" else "A",
                            1,
                        )
                    )
                for setup_a, setup_b, answer, orientation in orientations:
                    identity = {
                        "problem": problem,
                        "split": split,
                        "setup_a": setup_a["setup"]["setup_id"],
                        "setup_b": setup_b["setup"]["setup_id"],
                        "orientation": orientation,
                    }
                    output.append(
                        {
                            "record_id": hashlib.sha256(
                                _canonical_json(identity).encode("utf-8")
                            ).hexdigest(),
                            **identity,
                            "prompt": _prompt(
                                source=source,
                                phase_b=phase_b,
                                first=setup_a,
                                second=setup_b,
                            ),
                            "completion": answer,
                        }
                    )
    return output


def _scores_from_pair_margins(
    pair_predictions: list[dict[str, Any]],
    outcomes: list[dict[str, Any]],
) -> dict[str, float]:
    utilities: dict[str, dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    for prediction in pair_predictions:
        problem = str(prediction["problem"])
        first = str(prediction["setup_a"])
        second = str(prediction["setup_b"])
        margin = float(prediction["margin_a_minus_b"])
        # Bounded margins retain confidence without allowing one pair to
        # dominate the complete ten-setup tournament.
        preference = math.tanh(margin / 2.0)
        utilities[problem][first] += preference
        utilities[problem][second] -= preference

    scores = {}
    for record in outcomes:
        if record["split"] != "validation":
            continue
        utility = utilities[str(record["problem"])].get(
            str(record["setup"]["setup_id"]),
            0.0,
        )
        scores[_record_id(record)] = -float(utility)
    return scores


def _pair_accuracy(predictions: list[dict[str, Any]]) -> float:
    if not predictions:
        return 0.0
    return sum(
        prediction["predicted"] == prediction["expected"]
        for prediction in predictions
    ) / len(predictions)


def _load_model(args: argparse.Namespace) -> tuple[Any, torch.nn.Module]:
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
    ).to(torch.device("cuda:0"))
    return tokenizer, model


def _run_fold(
    args: argparse.Namespace,
    *,
    fold: int,
    held_out: set[str],
    canonical_records: list[dict[str, Any]],
) -> dict[str, Any]:
    fold_records = _outer_fold_records(canonical_records, held_out)
    _heartbeat(args, fold=fold, phase="build_pairs")
    train_pairs = _pair_records(
        fold_records,
        benchmark_root=args.benchmark_root,
        split="train",
        both_orientations=True,
    )
    validation_pairs = _pair_records(
        fold_records,
        benchmark_root=args.benchmark_root,
        split="validation",
        both_orientations=False,
    )
    _heartbeat(args, fold=fold, phase="load_base_model")
    tokenizer, model = _load_model(args)
    device = torch.device("cuda:0")
    _heartbeat(args, fold=fold, phase="base_pairwise_evaluation")
    base_pairs = _pair_margins(
        model,
        tokenizer,
        validation_pairs,
        max_length=args.max_length,
        batch_size=args.eval_batch_size,
        device=device,
    )
    base_scores = _scores_from_pair_margins(base_pairs, fold_records)

    adapter_dir = args.adapter_root / f"fold_{fold}"
    fold_configuration = {
        **vars(args),
        "adapter_dir": adapter_dir,
        "seed": args.seed + fold,
    }
    fold_args = argparse.Namespace(**fold_configuration)
    _heartbeat(
        args,
        fold=fold,
        phase="lora_training",
        progress={"global_step": 0},
    )

    def training_progress(record: dict[str, Any]) -> None:
        _heartbeat(
            args,
            fold=fold,
            phase="lora_training",
            progress=record,
        )

    model, training_logs = _train_lora(
        model,
        tokenizer,
        train_pairs,
        fold_args,
        device=device,
        progress_callback=training_progress,
    )
    _heartbeat(args, fold=fold, phase="lora_pairwise_evaluation")
    lora_pairs = _pair_margins(
        model,
        tokenizer,
        validation_pairs,
        max_length=args.max_length,
        batch_size=args.eval_batch_size,
        device=device,
    )
    lora_scores = _scores_from_pair_margins(lora_pairs, fold_records)
    output = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "fold": fold,
        "held_out_problems": sorted(held_out),
        "training_pairs": len(train_pairs),
        "validation_pairs": len(validation_pairs),
        "base_pair_accuracy": _pair_accuracy(base_pairs),
        "lora_pair_accuracy": _pair_accuracy(lora_pairs),
        "base_scores": base_scores,
        "lora_scores": lora_scores,
        "training_logs": training_logs,
        "adapter_dir": str(adapter_dir.resolve()),
    }
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return output


def _metric_lookup(
    aggregate: list[dict[str, Any]],
    router: str,
    top_k: int,
) -> dict[str, Any]:
    return next(
        row
        for row in aggregate
        if row["router"] == router
        and row["protocol"] == "raw_predicted"
        and row["top_k"] == top_k
    )


def _report(
    aggregate: list[dict[str, Any]],
    *,
    pair_accuracy: dict[str, float],
) -> str:
    lines = [
        "# Qwen3-0.6B Corrected-v2 OOF Setup Router",
        "",
        "- Protocol: identical five-fold benchmark-grouped OOF evaluation over 19 development lineages.",
        "- Inputs: source code, frozen Phase-B evidence, and two candidate setup descriptions.",
        "- Labels and candidate outcomes are unavailable at inference.",
        "- Fine-tuning: fold-specific LoRA; no held-out lineage is used by its predicting adapter.",
        "",
        "| router | pair accuracy | exact@1 | exact@3 | exact@5 | within5@1 | within5@3 | within5@5 | regret@1 | regret@3 | regret@5 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for router in (BASE_ROUTER, LORA_ROUTER):
        metrics = [
            _metric_lookup(aggregate, router, top_k)
            for top_k in (1, 3, 5)
        ]
        lines.append(
            "| "
            f"{DISPLAY_NAMES[router]} | "
            f"{pair_accuracy[router]:.3f} | "
            + " | ".join(
                f"{metric['top_k_exact_accuracy']:.3f}"
                for metric in metrics
            )
            + " | "
            + " | ".join(
                f"{metric['within_5pct_coverage']:.3f}"
                for metric in metrics
            )
            + " | "
            + " | ".join(
                f"{metric['geomean_regret']:.3f}"
                for metric in metrics
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Candidate savings versus the ten-setup exhaustive tournament are 90%, 70%, and 50% for Top-1, Top-3, and Top-5.",
            "",
            "These are development OOF architecture estimates, not final held-out test results. The four fixed test lineages remain untouched by this run.",
        ]
    )
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    canonical_records = _canonicalize_phase_b(_load(args.corpus))
    train_problems = sorted(
        {
            str(record["problem"])
            for record in canonical_records
            if record["split"] == "train"
        }
    )
    splitter = GroupKFold(n_splits=5)
    held_out_folds = [
        {train_problems[index] for index in held_out_indices}
        for _, held_out_indices in splitter.split(
            np.zeros(len(train_problems)),
            groups=np.asarray(train_problems),
        )
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.adapter_root.mkdir(parents=True, exist_ok=True)

    fold_outputs = []
    for fold, held_out in enumerate(held_out_folds):
        checkpoint = args.output_dir / f"fold_{fold}.json"
        if args.resume and checkpoint.is_file():
            fold_output = json.loads(checkpoint.read_text(encoding="utf-8"))
        else:
            _heartbeat(args, fold=fold, phase="starting_fold")
            fold_output = _run_fold(
                args,
                fold=fold,
                held_out=held_out,
                canonical_records=canonical_records,
            )
            _write_json(checkpoint, fold_output)
        fold_outputs.append(fold_output)
        _write_json(
            args.output_dir / "heartbeat.json",
            {
                "schema_version": f"{SCHEMA_VERSION}.heartbeat",
                "updated_at": _utc_now(),
                "pid": os.getpid(),
                "state": "running",
                "folds_completed": len(fold_outputs),
                "folds_total": len(held_out_folds),
                "last_fold": fold,
            },
        )

    selection_rows = []
    ranking_rows = []
    weighted_pair_correct = {BASE_ROUTER: 0.0, LORA_ROUTER: 0.0}
    total_pairs = 0
    for fold_output, held_out in zip(
        fold_outputs,
        held_out_folds,
        strict=True,
    ):
        fold = int(fold_output["fold"])
        fold_records = _outer_fold_records(canonical_records, held_out)
        pair_count = int(fold_output["validation_pairs"])
        total_pairs += pair_count
        weighted_pair_correct[BASE_ROUTER] += (
            float(fold_output["base_pair_accuracy"]) * pair_count
        )
        weighted_pair_correct[LORA_ROUTER] += (
            float(fold_output["lora_pair_accuracy"]) * pair_count
        )
        for router, score_key in (
            (BASE_ROUTER, "base_scores"),
            (LORA_ROUTER, "lora_scores"),
        ):
            selected, rankings = evaluate_topk_scores(
                fold_records,
                {
                    str(record_id): float(score)
                    for record_id, score in fold_output[score_key].items()
                },
                split="validation",
                router=router,
                fold=fold,
            )
            selection_rows.extend(selected)
            ranking_rows.extend(rankings)

    aggregate = _aggregate(selection_rows)
    pair_accuracy = {
        router: weighted_pair_correct[router] / total_pairs
        for router in (BASE_ROUTER, LORA_ROUTER)
    }
    _write_csv(args.output_dir / "topk_metrics.csv", aggregate)
    _write_csv(
        args.output_dir / "per_benchmark_topk.csv",
        selection_rows,
    )
    _write_csv(
        args.output_dir / "oof_setup_rankings.csv",
        ranking_rows,
    )
    (args.output_dir / "report.md").write_text(
        _report(aggregate, pair_accuracy=pair_accuracy),
        encoding="utf-8",
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "configuration": {
            "model": str(args.model.resolve()),
            "model_config_sha256": _sha256(args.model / "config.json"),
            "max_length": args.max_length,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "seed": args.seed,
            "visible_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "pair_aggregation": "sum_tanh_half_margin",
        },
        "methodology": {
            "protocol": "five-fold benchmark-grouped out-of-fold",
            "development_lineages": len(train_problems),
            "reference_metrics_as_inputs": False,
            "post_candidate_features_as_inputs": False,
            "fixed_final_test_lineages_used": False,
            "top_k_values": [1, 3, 5],
        },
        "corpus": {
            "path": str(args.corpus.resolve()),
            "sha256": _sha256(args.corpus),
        },
        "folds": [
            {
                key: fold_output[key]
                for key in (
                    "fold",
                    "held_out_problems",
                    "training_pairs",
                    "validation_pairs",
                    "base_pair_accuracy",
                    "lora_pair_accuracy",
                    "adapter_dir",
                )
            }
            for fold_output in fold_outputs
        ],
        "pair_accuracy": pair_accuracy,
        "metrics": aggregate,
    }
    _write_json(args.output_dir / "metrics.json", summary)
    _write_json(
        args.output_dir / "heartbeat.json",
        {
            "schema_version": f"{SCHEMA_VERSION}.heartbeat",
            "updated_at": _utc_now(),
            "pid": os.getpid(),
            "state": "complete",
            "folds_completed": len(held_out_folds),
            "folds_total": len(held_out_folds),
        },
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=DEFAULT_BENCHMARK_ROOT,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--adapter-root",
        type=Path,
        default=DEFAULT_ADAPTER_ROOT,
    )
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=46)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> int:
    summary = run(parse_args())
    print(
        json.dumps(
            {
                "output_dir": str(DEFAULT_OUTPUT),
                "pair_accuracy": summary["pair_accuracy"],
                "metrics": summary["metrics"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
