#!/usr/bin/env python3
"""Summarize tree, neural, and small-LM setup-router comparisons."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_setup_router import _sha256


SCHEMA_VERSION = "c2hls.setup-router-architecture-comparison.v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _learned(metrics: dict[str, Any]) -> dict[str, Any]:
    return metrics["learned_top_k"]


def build(args: argparse.Namespace) -> dict[str, Any]:
    strengthened = json.loads(
        args.strengthened.read_text(encoding="utf-8")
    )
    neural = json.loads(args.neural.read_text(encoding="utf-8"))
    small_lm = json.loads(args.small_lm.read_text(encoding="utf-8"))

    outer_rows = []
    for name, metrics in (
        (
            "adaptive_extratrees_committee",
            strengthened["outer_grouped_evaluation"]["policies"][
                "committee_disagreement_adaptive"
            ],
        ),
        (
            "embedding_retrieval_k1",
            strengthened["outer_grouped_evaluation"]["policies"][
                "retrieval_source_k1"
            ],
        ),
        *[
            (name, metrics)
            for name, metrics in neural["outer_grouped_evaluation"][
                "variants"
            ].items()
        ],
    ):
        outer_rows.append(
            {
                "router": name,
                "protocol": "five_fold_grouped_development",
                "benchmarks": metrics["benchmark_count"],
                "candidate_count": metrics["candidate_count"],
                "candidate_savings": metrics[
                    "candidate_savings_vs_exhaustive"
                ],
                "validity": metrics["validity"],
                "within_5pct": metrics["within_5pct_coverage"],
                "oracle_coverage": metrics["oracle_coverage"],
                "geomean_regret": metrics["geomean_regret"],
            }
        )

    selected_neural = neural["selection"]["selected_neural"]
    confirmation_metrics = (
        (
            "absolute_extratrees",
            strengthened["held_out_test"]["absolute_structured"],
        ),
        (
            "adaptive_extratrees_committee",
            strengthened["held_out_test"][
                "committee_disagreement_adaptive"
            ],
        ),
        (
            selected_neural,
            neural["historical_confirmation"]["selected_neural"],
        ),
        ("qwen3_06b_base", small_lm["base"]["test"]),
        ("qwen3_06b_lora_margin", small_lm["lora_sft"]["test"]),
    )
    confirmation_rows = []
    for name, metrics in confirmation_metrics:
        learned = _learned(metrics)
        confirmation_rows.append(
            {
                "router": name,
                "protocol": "historical_exposed_confirmation",
                "benchmarks": metrics["benchmark_count"],
                "candidate_count": learned["candidate_count"],
                "candidate_savings": learned[
                    "candidate_savings_vs_exhaustive"
                ],
                "validity": learned["validity"],
                "within_5pct": learned["within_5pct_coverage"],
                "oracle_coverage": learned["oracle_coverage"],
                "geomean_regret": learned["geomean_regret"],
                "pairwise_accuracy": metrics.get(
                    "pairwise_accuracy"
                ),
            }
        )

    output = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "source_artifacts": {
            "strengthened": {
                "path": str(args.strengthened.resolve()),
                "sha256": _sha256(args.strengthened),
            },
            "neural": {
                "path": str(args.neural.resolve()),
                "sha256": _sha256(args.neural),
            },
            "small_lm": {
                "path": str(args.small_lm.resolve()),
                "sha256": _sha256(args.small_lm),
            },
        },
        "outer_grouped_development": outer_rows,
        "historical_confirmation": confirmation_rows,
        "small_lm_validation": {
            "base": small_lm["base"]["validation"],
            "lora_sft": small_lm["lora_sft"]["validation"],
        },
        "recommendation": {
            "selected_router": "adaptive_extratrees_committee",
            "deployment_status": "advisory",
            "ranknet_role": (
                "three-candidate low-cost challenger for corrected-v2"
            ),
            "small_lm_role": (
                "research ablation; vLLM-compatible but not promoted"
            ),
            "reason": (
                "the adaptive committee has the strongest leakage-safe "
                "grouped result; RankNet is more candidate-efficient but "
                "weaker, and Qwen LoRA does not improve validation "
                "tournament quality over the base model"
            ),
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for name, rows in (
        ("outer_grouped_development.csv", outer_rows),
        ("historical_confirmation.csv", confirmation_rows),
    ):
        with (args.output_dir / name).open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(rows[0]),
            )
            writer.writeheader()
            writer.writerows(rows)

    names = [row["router"] for row in outer_rows]
    within = [row["within_5pct"] for row in outer_rows]
    regret = [row["geomean_regret"] for row in outer_rows]
    positions = np.arange(len(names))
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    axes[0].bar(positions, within, color="#2f855a")
    axes[0].set_ylabel("Grouped within-5% coverage")
    axes[0].set_ylim(0, 1)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(positions, regret, color="#2b6cb0")
    axes[1].axhline(1, color="black", linestyle="--", linewidth=1)
    axes[1].set_ylabel("Grouped geomean regret")
    axes[1].set_xticks(positions, names, rotation=28, ha="right")
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("Setup-router architectures: grouped development")
    fig.tight_layout()
    fig.savefig(
        args.output_dir / "outer_grouped_development.png",
        dpi=180,
    )
    plt.close(fig)

    best_outer = max(
        outer_rows,
        key=lambda row: (
            row["within_5pct"],
            -row["geomean_regret"],
        ),
    )
    qwen_base_validation = _learned(
        small_lm["base"]["validation"]
    )
    qwen_sft_validation = _learned(
        small_lm["lora_sft"]["validation"]
    )
    (args.output_dir / "report.md").write_text(
        "\n".join(
            [
                "# Setup Router Architecture Comparison",
                "",
                "## Grouped Development",
                "",
                "| router | candidates | within 5% | geomean regret |",
                "|---|---:|---:|---:|",
                *[
                    (
                        f"| {row['router']} | "
                        f"{row['candidate_count']:.2f} | "
                        f"{row['within_5pct']:.3f} | "
                        f"{row['geomean_regret']:.3f} |"
                    )
                    for row in sorted(
                        outer_rows,
                        key=lambda row: (
                            -row["within_5pct"],
                            row["geomean_regret"],
                        ),
                    )
                ],
                "",
                (
                    f"Best grouped router: `{best_outer['router']}`. "
                    "Small-LM results are excluded from this table because "
                    "five-fold LoRA retraining was not run."
                ),
                "",
                "## Qwen3-0.6B Validation",
                "",
                "| model | within 5% | geomean regret |",
                "|---|---:|---:|",
                (
                    "| base | "
                    f"{qwen_base_validation['within_5pct_coverage']:.3f} | "
                    f"{qwen_base_validation['geomean_regret']:.3f} |"
                ),
                (
                    "| LoRA binary margin | "
                    f"{qwen_sft_validation['within_5pct_coverage']:.3f} | "
                    f"{qwen_sft_validation['geomean_regret']:.3f} |"
                ),
                "",
                "Keep the adaptive ExtraTrees committee advisory. Carry "
                "RankNet into corrected-v2 confirmation as the efficient "
                "three-candidate challenger. Do not promote the current "
                "small-LM LoRA.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strengthened", type=Path, required=True)
    parser.add_argument("--neural", type=Path, required=True)
    parser.add_argument("--small-lm", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    output = build(parse_args())
    print(json.dumps(output["recommendation"], sort_keys=True))


if __name__ == "__main__":
    main()
