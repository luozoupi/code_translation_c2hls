#!/usr/bin/env python3
"""Compare compact neural setup routers on leakage-safe benchmark splits."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_hybrid_setup_router import (
    Variant,
    _load_embeddings,
    _variant_records,
)
from scripts.train_setup_router import (
    _best_fixed_setup,
    _feature_schema,
    _load,
    _matrix,
    _preprocessor,
    _sha256,
    _valid_probability,
)
from scripts.train_strengthened_setup_router import (
    _aggregate_outer_metrics,
    _canonicalize_phase_b,
    _first_wins,
    _outer_fold_records,
    _outer_selection_key,
    _outcome_groups,
    _pair_features,
    _pair_training_records,
    _ranking_metrics_from_scores,
    _ranking_outcomes,
    _record_id,
)


SCHEMA_VERSION = "c2hls.neural-setup-router-ablation.v1"


@dataclass(frozen=True)
class NeuralVariant:
    name: str
    kind: str
    hidden_layers: tuple[int, ...]


VARIANTS = (
    NeuralVariant("mlp_shallow_hybrid32", "direct", (32,)),
    NeuralVariant("mlp_deep_hybrid32", "direct", (128, 64, 32)),
    NeuralVariant("mlp_pairwise_hybrid32", "pairwise", (64, 32)),
    NeuralVariant("ranknet_hybrid32", "ranknet", (96, 48)),
)


class RankNet(torch.nn.Module):
    """A shared candidate scorer trained from benchmark-local preferences."""

    def __init__(self, input_size: int, hidden_layers: tuple[int, ...]):
        super().__init__()
        layers: list[torch.nn.Module] = []
        current = input_size
        for width in hidden_layers:
            layers.extend(
                [
                    torch.nn.Linear(current, width),
                    torch.nn.GELU(),
                    torch.nn.LayerNorm(width),
                ]
            )
            current = width
        layers.append(torch.nn.Linear(current, 1))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).squeeze(-1)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_hybrid_records(
    corpus: Path,
    embedding_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base_records = _load(corpus)
    embeddings, record_map, embedding_manifest = _load_embeddings(
        embedding_dir
    )
    hybrid = _variant_records(
        base_records,
        Variant("hybrid32", 32, True, True, True),
        embeddings,
        record_map,
    )
    canonical_structured = _canonicalize_phase_b(base_records)
    canonical_hybrid = _canonicalize_phase_b(hybrid)
    phase_b_log_by_id = {
        _record_id(record): float(
            record["features"]["phase_b_log_latency_cycles"]
        )
        for record in canonical_structured
    }
    canonical_hybrid = [
        {
            **record,
            "_phase_b_log_latency_cycles_for_target": (
                phase_b_log_by_id[_record_id(record)]
            ),
        }
        for record in canonical_hybrid
    ]
    return canonical_hybrid, embedding_manifest


def _mlp_pipeline(
    categorical: list[str],
    numeric: list[str],
    names: list[str],
    *,
    hidden_layers: tuple[int, ...],
    classifier: bool,
) -> Pipeline:
    model: MLPClassifier | MLPRegressor
    common = {
        "hidden_layer_sizes": hidden_layers,
        "activation": "relu",
        "solver": "lbfgs",
        "alpha": 0.01,
        "max_iter": 1000,
        "random_state": 46,
    }
    if classifier:
        model = MLPClassifier(**common)
    else:
        model = MLPRegressor(**common)
    return Pipeline(
        [
            ("features", _preprocessor(categorical, numeric, names)),
            ("scale", StandardScaler()),
            ("model", model),
        ]
    )


def _fit_direct_mlp(
    records: list[dict[str, Any]],
    hidden_layers: tuple[int, ...],
) -> tuple[dict[str, float], dict[str, Any]]:
    names, categorical, numeric = _feature_schema(records)
    feasibility = [
        record
        for record in records
        if record["split"] == "train"
        and record["eligibility"]["feasibility_model"]
    ]
    ranking = [
        record
        for record in _ranking_outcomes(records, split="train")
        if record["labels"]["valid"]
    ]
    classifier = _mlp_pipeline(
        categorical,
        numeric,
        names,
        hidden_layers=hidden_layers,
        classifier=True,
    )
    regressor = _mlp_pipeline(
        categorical,
        numeric,
        names,
        hidden_layers=hidden_layers,
        classifier=False,
    )
    targets = np.asarray(
        [
            float(record["labels"]["log_latency_cycles"])
            - float(record["_phase_b_log_latency_cycles_for_target"])
            for record in ranking
        ],
        dtype=np.float64,
    )
    target_mean = float(np.mean(targets))
    target_scale = float(np.std(targets)) or 1.0
    normalized_targets = (targets - target_mean) / target_scale
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        classifier.fit(
            _matrix(feasibility, names),
            [record["labels"]["valid"] for record in feasibility],
        )
        regressor.fit(
            _matrix(ranking, names),
            normalized_targets,
        )

    outcomes = _ranking_outcomes(records)
    valid_probabilities = _valid_probability(
        classifier,
        _matrix(outcomes, names),
    )
    normalized_predictions = regressor.predict(
        _matrix(outcomes, names)
    )
    predictions = (
        normalized_predictions * target_scale + target_mean
    )
    scores = {
        _record_id(record): (
            math.exp(float(prediction))
            / max(float(valid_probability), 0.05)
        )
        for record, prediction, valid_probability in zip(
            outcomes,
            predictions,
            valid_probabilities,
            strict=True,
        )
    }
    return scores, {
        "kind": "direct_mlp",
        "classifier": classifier,
        "regressor": regressor,
        "target_mean": target_mean,
        "target_scale": target_scale,
        "feature_names": names,
        "hidden_layers": hidden_layers,
        "feasibility_records": len(feasibility),
        "ranking_records": len(ranking),
    }


def _fit_pairwise_mlp(
    records: list[dict[str, Any]],
    hidden_layers: tuple[int, ...],
) -> tuple[dict[str, float], dict[str, Any]]:
    pairs = _pair_training_records(records)
    names, categorical, numeric = _feature_schema(pairs)
    classifier = _mlp_pipeline(
        categorical,
        numeric,
        names,
        hidden_layers=hidden_layers,
        classifier=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        classifier.fit(
            _matrix(pairs, names),
            [pair["label"] for pair in pairs],
        )

    scores = {}
    for outcomes in _outcome_groups(records).values():
        ordered = sorted(
            outcomes,
            key=lambda record: record["setup"]["setup_id"],
        )
        comparisons = []
        owners = []
        for first in ordered:
            for second in ordered:
                if first is second:
                    continue
                comparisons.append(
                    {"features": _pair_features(first, second)}
                )
                owners.append(_record_id(first))
        probabilities = _valid_probability(
            classifier,
            _matrix(comparisons, names),
        )
        wins: dict[str, list[float]] = defaultdict(list)
        for owner, probability in zip(
            owners,
            probabilities,
            strict=True,
        ):
            wins[owner].append(float(probability))
        for record in ordered:
            scores[_record_id(record)] = 1.0 - float(
                np.mean(wins[_record_id(record)])
            )
    return scores, {
        "kind": "pairwise_mlp",
        "classifier": classifier,
        "feature_names": names,
        "hidden_layers": hidden_layers,
        "pair_records": len(pairs),
    }


def _ranknet_pairs(
    outcomes: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[tuple[int, int]]]:
    ordered = sorted(
        outcomes,
        key=lambda record: (
            record["problem"],
            record["setup"]["setup_id"],
        ),
    )
    indices = {_record_id(record): index for index, record in enumerate(ordered)}
    pairs = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in ordered:
        grouped[str(record["problem"])].append(record)
    for problem_records in grouped.values():
        for first_index, first in enumerate(problem_records):
            for second in problem_records[first_index + 1 :]:
                first_wins = _first_wins(first, second)
                if first_wins is None:
                    continue
                winner, loser = (
                    (first, second)
                    if first_wins
                    else (second, first)
                )
                pairs.append(
                    (indices[_record_id(winner)], indices[_record_id(loser)])
                )
    return ordered, pairs


def _fit_ranknet(
    records: list[dict[str, Any]],
    hidden_layers: tuple[int, ...],
    *,
    epochs: int,
) -> tuple[dict[str, float], dict[str, Any]]:
    names, categorical, numeric = _feature_schema(records)
    train_outcomes, pairs = _ranknet_pairs(
        _ranking_outcomes(records, split="train")
    )
    preprocessor = _preprocessor(categorical, numeric, names)
    train_values = preprocessor.fit_transform(
        _matrix(train_outcomes, names)
    )
    scaler = StandardScaler()
    train_values = scaler.fit_transform(train_values).astype(np.float32)
    winner_indices = torch.tensor(
        [pair[0] for pair in pairs],
        dtype=torch.long,
    )
    loser_indices = torch.tensor(
        [pair[1] for pair in pairs],
        dtype=torch.long,
    )
    values = torch.from_numpy(train_values)

    torch.manual_seed(46)
    model = RankNet(train_values.shape[1], hidden_layers)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-3,
        weight_decay=1e-3,
    )
    model.train()
    final_loss = None
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        candidate_scores = model(values)
        loss = torch.nn.functional.softplus(
            candidate_scores[winner_indices]
            - candidate_scores[loser_indices]
        ).mean()
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach())

    outcomes = _ranking_outcomes(records)
    inference_values = scaler.transform(
        preprocessor.transform(_matrix(outcomes, names))
    ).astype(np.float32)
    model.eval()
    with torch.no_grad():
        predictions = model(
            torch.from_numpy(inference_values)
        ).numpy()
    scores = {
        _record_id(record): float(prediction)
        for record, prediction in zip(
            outcomes,
            predictions,
            strict=True,
        )
    }
    return scores, {
        "kind": "ranknet",
        "preprocessor": preprocessor,
        "scaler": scaler,
        "state_dict": {
            name: value.detach().cpu().numpy()
            for name, value in model.state_dict().items()
        },
        "input_size": train_values.shape[1],
        "hidden_layers": hidden_layers,
        "feature_names": names,
        "pair_records": len(pairs),
        "epochs": epochs,
        "final_training_loss": final_loss,
    }


def _fit_variant(
    records: list[dict[str, Any]],
    variant: NeuralVariant,
    *,
    ranknet_epochs: int,
) -> tuple[dict[str, float], dict[str, Any]]:
    if variant.kind == "direct":
        return _fit_direct_mlp(records, variant.hidden_layers)
    if variant.kind == "pairwise":
        return _fit_pairwise_mlp(records, variant.hidden_layers)
    if variant.kind == "ranknet":
        return _fit_ranknet(
            records,
            variant.hidden_layers,
            epochs=ranknet_epochs,
        )
    raise ValueError(f"unknown neural variant kind: {variant.kind}")


def _outer_grouped_evaluation(
    records: list[dict[str, Any]],
    *,
    ranknet_epochs: int,
) -> dict[str, Any]:
    train_problems = sorted(
        {
            str(record["problem"])
            for record in records
            if record["split"] == "train"
        }
    )
    splitter = GroupKFold(n_splits=5)
    fold_held_out = [
        {
            train_problems[index]
            for index in held_out_indices
        }
        for _, held_out_indices in splitter.split(
            np.zeros(len(train_problems)),
            groups=np.asarray(train_problems),
        )
    ]
    fold_metrics: dict[str, list[dict[str, Any]]] = defaultdict(list)
    manifest = []
    for fold, held_out in enumerate(fold_held_out):
        fold_records = _outer_fold_records(records, held_out)
        train_ranking = [
            record
            for record in _ranking_outcomes(
                fold_records,
                split="train",
            )
            if record["labels"]["valid"]
        ]
        best_fixed, _ = _best_fixed_setup(train_ranking)
        manifest.append(
            {
                "fold": fold,
                "held_out_problems": sorted(held_out),
            }
        )
        for variant in VARIANTS:
            scores, _ = _fit_variant(
                fold_records,
                variant,
                ranknet_epochs=ranknet_epochs,
            )
            metrics, _ = _ranking_metrics_from_scores(
                fold_records,
                scores,
                split="validation",
                best_fixed_setup=best_fixed,
            )
            fold_metrics[variant.name].append(metrics)
    return {
        "fold_manifest": manifest,
        "variants": {
            name: _aggregate_outer_metrics(values)
            for name, values in fold_metrics.items()
        },
    }


def _plot(rows: list[dict[str, Any]], destination: Path) -> None:
    names = [row["variant"] for row in rows]
    within = [row["outer_within_5pct"] for row in rows]
    regret = [row["outer_geomean_regret"] for row in rows]
    positions = np.arange(len(names))
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    axes[0].bar(positions, within, color="#2f855a")
    axes[0].set_ylabel("Grouped within-5% coverage")
    axes[0].set_ylim(0, 1)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(positions, regret, color="#2b6cb0")
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[1].set_ylabel("Grouped geomean regret")
    axes[1].set_xticks(positions, names, rotation=25, ha="right")
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("Compact neural setup-router comparison")
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def train(args: argparse.Namespace) -> dict[str, Any]:
    records, embedding_manifest = _canonical_hybrid_records(
        args.corpus,
        args.embedding_dir,
    )
    train_ranking = [
        record
        for record in _ranking_outcomes(records, split="train")
        if record["labels"]["valid"]
    ]
    best_fixed, _ = _best_fixed_setup(train_ranking)

    outer = _outer_grouped_evaluation(
        records,
        ranknet_epochs=args.ranknet_epochs,
    )
    selected = min(
        outer["variants"],
        key=lambda name: _outer_selection_key(
            outer["variants"][name],
            name,
        ),
    )

    models = {}
    validation = {}
    for variant in VARIANTS:
        scores, model = _fit_variant(
            records,
            variant,
            ranknet_epochs=args.ranknet_epochs,
        )
        models[variant.name] = model
        metrics, _ = _ranking_metrics_from_scores(
            records,
            scores,
            split="validation",
            best_fixed_setup=best_fixed,
        )
        validation[variant.name] = metrics

    selected_variant = next(
        variant for variant in VARIANTS if variant.name == selected
    )
    selected_scores, selected_model = _fit_variant(
        records,
        selected_variant,
        ranknet_epochs=args.ranknet_epochs,
    )
    models[selected] = selected_model
    test_metrics, test_predictions = _ranking_metrics_from_scores(
        records,
        selected_scores,
        split="test",
        best_fixed_setup=best_fixed,
    )

    strengthened_metrics = json.loads(
        args.strengthened_metrics.read_text(encoding="utf-8")
    )
    reference = {
        "absolute_structured": strengthened_metrics["held_out_test"][
            "absolute_structured"
        ],
        "adaptive_committee": strengthened_metrics["held_out_test"][
            "committee_disagreement_adaptive"
        ],
    }
    rows = [
        {
            "variant": name,
            "outer_within_5pct": values["within_5pct_coverage"],
            "outer_oracle_coverage": values["oracle_coverage"],
            "outer_geomean_regret": values["geomean_regret"],
            "outer_validity": values["validity"],
            "candidate_count": values["candidate_count"],
        }
        for name, values in sorted(outer["variants"].items())
    ]
    output = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "methodology": {
            "selection": "five-fold benchmark-grouped development evaluation",
            "test_not_used_for_selection": True,
            "test_reuse_caveat": (
                "historical confirmation kernels were exposed by prior "
                "router studies and are not a pristine final test"
            ),
            "canonical_phase_b_context": True,
            "reference_metrics_as_inputs": False,
            "post_candidate_features_as_inputs": False,
            "candidate_policy": (
                "mandatory multistep skillless plus two predicted setups"
            ),
        },
        "corpus": {
            "path": str(args.corpus.resolve()),
            "sha256": _sha256(args.corpus),
        },
        "embedding": embedding_manifest,
        "outer_grouped_evaluation": outer,
        "validation": validation,
        "selection": {"selected_neural": selected},
        "historical_confirmation": {
            "selected_neural": test_metrics,
            **reference,
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    model_path = args.output_dir / "neural_router_models.joblib"
    joblib.dump(
        {
            "schema_version": SCHEMA_VERSION,
            "selected_neural": selected,
            "models": models,
            "embedding_manifest": embedding_manifest,
        },
        model_path,
    )
    csv_path = args.output_dir / "outer_grouped_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    prediction_path = args.output_dir / "test_predictions.jsonl"
    with prediction_path.open("w", encoding="utf-8") as handle:
        for row in test_predictions:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    chart_path = args.output_dir / "outer_grouped_comparison.png"
    _plot(rows, chart_path)

    selected_outer = outer["variants"][selected]
    neural_learned = test_metrics["learned_top_k"]
    committee_learned = reference["adaptive_committee"]["learned_top_k"]
    report_path = args.output_dir / "report.md"
    report_path.write_text(
        "\n".join(
            [
                "# Compact Neural Setup Routers",
                "",
                f"- Selected without test access: `{selected}`",
                (
                    "- Selected grouped development result: "
                    f"{selected_outer['within_5pct_coverage']:.3f} within "
                    f"5%, {selected_outer['geomean_regret']:.3f} "
                    "geomean regret."
                ),
                "",
                "## Historical Confirmation",
                "",
                "| router | candidates | within 5% | geomean regret |",
                "|---|---:|---:|---:|",
                (
                    "| selected neural | "
                    f"{neural_learned['candidate_count']:.1f} | "
                    f"{neural_learned['within_5pct_coverage']:.3f} | "
                    f"{neural_learned['geomean_regret']:.3f} |"
                ),
                (
                    "| adaptive ExtraTrees committee | "
                    f"{committee_learned['candidate_count']:.1f} | "
                    f"{committee_learned['within_5pct_coverage']:.3f} | "
                    f"{committee_learned['geomean_regret']:.3f} |"
                ),
                "",
                "The confirmation set is historically exposed. Use these "
                "results for architecture screening only.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--strengthened-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ranknet-epochs", type=int, default=250)
    return parser.parse_args()


def main() -> None:
    output = train(parse_args())
    print(
        json.dumps(
            {
                "selection": output["selection"],
                "outer_grouped_evaluation": output[
                    "outer_grouped_evaluation"
                ]["variants"],
                "historical_confirmation": output[
                    "historical_confirmation"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
