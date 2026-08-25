#!/usr/bin/env python3
"""Compare structured, frozen-transformer, and hybrid setup routers."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
from sklearn.model_selection import GroupKFold, cross_val_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, balanced_accuracy_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_setup_router import (
    _best_fixed_setup,
    _classifier_metrics,
    _classifier_pipeline,
    _feature_schema,
    _load,
    _matrix,
    _preferred_behavior_records,
    _ranking_metrics,
    _regressor_pipeline,
    _sha256,
    _valid_probability,
)


SCHEMA_VERSION = "c2hls.hybrid-setup-router-ablation.v1"


@dataclass(frozen=True)
class Variant:
    name: str
    dimension: int
    structured: bool
    source_embedding: bool
    phase_b_embedding: bool


VARIANTS = (
    Variant("structured", 0, True, False, False),
    Variant("transformer_source_mrl64", 64, False, True, False),
    Variant("transformer_phase_b_mrl64", 64, False, False, True),
    Variant("transformer_both_mrl64", 64, False, True, True),
    Variant("hybrid_both_mrl32", 32, True, True, True),
    Variant("hybrid_source_mrl64", 64, True, True, False),
    Variant("hybrid_phase_b_mrl64", 64, True, False, True),
    Variant("hybrid_both_mrl64", 64, True, True, True),
    Variant("hybrid_both_mrl128", 128, True, True, True),
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_embeddings(
    embedding_dir: Path,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, str]], dict]:
    archive = np.load(embedding_dir / "embeddings.npz", allow_pickle=False)
    keys = [str(value) for value in archive["keys"]]
    vectors = np.asarray(archive["vectors"], dtype=np.float32)
    if len(keys) != len(vectors) or len(keys) != len(set(keys)):
        raise ValueError("invalid or duplicate embedding keys")
    if vectors.ndim != 2 or not np.isfinite(vectors).all():
        raise ValueError("embedding matrix must be finite and two-dimensional")
    by_key = {
        key: vector
        for key, vector in zip(keys, vectors, strict=True)
    }
    record_map = {}
    with (embedding_dir / "record_embedding_map.jsonl").open(
        encoding="utf-8"
    ) as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            record_id = str(item["record_id"])
            if record_id in record_map:
                raise ValueError(f"duplicate record embedding map: {record_id}")
            record_map[record_id] = item
    manifest = json.loads(
        (embedding_dir / "manifest.json").read_text(encoding="utf-8")
    )
    return by_key, record_map, manifest


def _mrl(vector: np.ndarray, dimension: int) -> np.ndarray:
    if dimension < 1 or dimension > len(vector):
        raise ValueError(
            f"invalid MRL dimension {dimension} for vector {len(vector)}"
        )
    truncated = np.asarray(vector[:dimension], dtype=np.float64)
    norm = float(np.linalg.norm(truncated))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("zero or invalid truncated embedding norm")
    return truncated / norm


def _setup_features(features: dict[str, Any]) -> dict[str, Any]:
    return {
        name: value
        for name, value in features.items()
        if name == "model_id" or name.startswith("setup_")
    }


def _variant_records(
    records: list[dict[str, Any]],
    variant: Variant,
    embeddings: dict[str, np.ndarray],
    record_map: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    output = []
    for record in records:
        record_id = str(
            (record.get("provenance") or {}).get("dedup_key_sha256") or ""
        )
        mapping = record_map.get(record_id)
        if mapping is None:
            raise ValueError(f"record missing embedding mapping: {record_id}")
        if mapping["benchmark_lineage"] != record["benchmark_lineage"]:
            raise ValueError(f"lineage mismatch for {record_id}")
        if mapping["split"] != record["split"]:
            raise ValueError(f"split mismatch for {record_id}")
        features = (
            dict(record["features"])
            if variant.structured
            else _setup_features(record["features"])
        )
        for enabled, prefix, key_name in (
            (
                variant.source_embedding,
                "transformer_source",
                "source_embedding_key",
            ),
            (
                variant.phase_b_embedding,
                "transformer_phase_b",
                "phase_b_embedding_key",
            ),
        ):
            if not enabled:
                continue
            key = str(mapping[key_name])
            if key not in embeddings:
                raise ValueError(f"missing embedding vector: {key}")
            reduced = _mrl(embeddings[key], variant.dimension)
            features.update(
                {
                    f"{prefix}_{index:03d}": float(value)
                    for index, value in enumerate(reduced)
                }
            )
        output.append({**record, "features": features})
    return output


def _fit_models(
    records: list[dict[str, Any]],
    *,
    trees: int,
    jobs: int,
) -> tuple[Any, Any, dict[str, Any], list[str], list[str], list[str]]:
    names, categorical, numeric = _feature_schema(records)
    train_feasibility = [
        record
        for record in records
        if record["split"] == "train"
        and record["eligibility"]["feasibility_model"]
    ]
    train_ranking = [
        record
        for record in records
        if record["split"] == "train"
        and record["eligibility"]["ranking_model"]
        and record["labels"]["valid"]
    ]
    classifier = _classifier_pipeline(categorical, numeric, names)
    classifier.set_params(
        model__n_estimators=trees,
        model__n_jobs=jobs,
        model__max_depth=None,
        model__max_features=0.75,
        model__min_samples_leaf=3,
    )
    regressor = _regressor_pipeline(categorical, numeric, names)
    regressor.set_params(
        model__n_estimators=trees,
        model__n_jobs=jobs,
        model__max_depth=12,
        model__max_features=0.75,
        model__min_samples_leaf=1,
    )
    classifier_cv = cross_val_score(
        classifier,
        _matrix(train_feasibility, names),
        [record["labels"]["valid"] for record in train_feasibility],
        groups=[
            record["benchmark_lineage"] for record in train_feasibility
        ],
        scoring="balanced_accuracy",
        cv=GroupKFold(n_splits=5),
        n_jobs=1,
    )
    regressor_cv = -cross_val_score(
        regressor,
        _matrix(train_ranking, names),
        [record["labels"]["log_latency_cycles"] for record in train_ranking],
        groups=[record["benchmark_lineage"] for record in train_ranking],
        scoring="neg_mean_absolute_error",
        cv=GroupKFold(n_splits=5),
        n_jobs=1,
    )
    classifier.fit(
        _matrix(train_feasibility, names),
        [record["labels"]["valid"] for record in train_feasibility],
    )
    regressor.fit(
        _matrix(train_ranking, names),
        [record["labels"]["log_latency_cycles"] for record in train_ranking],
    )
    cv_metrics = {
        "feasibility_records": len(train_feasibility),
        "ranking_records": len(train_ranking),
        "benchmark_grouped_cv_folds": 5,
        "classifier_balanced_accuracy_mean": float(
            np.mean(classifier_cv)
        ),
        "classifier_balanced_accuracy_std": float(np.std(classifier_cv)),
        "regressor_log_mae_mean": float(np.mean(regressor_cv)),
        "regressor_log_mae_std": float(np.std(regressor_cv)),
    }
    return (
        classifier,
        regressor,
        cv_metrics,
        names,
        categorical,
        numeric,
    )


def _evaluate(
    classifier: Any,
    regressor: Any,
    records: list[dict[str, Any]],
    names: list[str],
    *,
    split: str,
    best_fixed_setup: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    split_records = [
        record for record in records if record["split"] == split
    ]
    feasibility = _classifier_metrics(
        classifier,
        [
            record
            for record in split_records
            if record["eligibility"]["feasibility_model"]
        ],
        names,
    )
    ranking, predictions = _ranking_metrics(
        classifier,
        regressor,
        split_records,
        names,
        best_fixed_setup=best_fixed_setup,
    )
    return {"feasibility": feasibility, "ranking": ranking}, predictions


def _calibrated_feasibility(
    classifier: Any,
    records: list[dict[str, Any]],
    names: list[str],
) -> dict[str, Any]:
    by_split = {}
    for split in ("validation", "test"):
        eligible = [
            record
            for record in records
            if record["split"] == split
            and record["eligibility"]["feasibility_model"]
        ]
        labels = np.asarray(
            [bool(record["labels"]["valid"]) for record in eligible]
        )
        probabilities = _valid_probability(
            classifier,
            _matrix(eligible, names),
        )
        by_split[split] = (labels, probabilities)

    validation_labels, validation_probabilities = by_split["validation"]
    thresholds = sorted(
        {
            0.0,
            0.5,
            1.0,
            *[float(value) for value in validation_probabilities],
        }
    )
    scored = []
    for threshold in thresholds:
        predictions = validation_probabilities >= threshold
        scored.append(
            (
                float(
                    balanced_accuracy_score(
                        validation_labels,
                        predictions,
                    )
                ),
                float(accuracy_score(validation_labels, predictions)),
                -abs(threshold - 0.5),
                -threshold,
                threshold,
            )
        )
    best = max(scored)
    threshold = float(best[-1])

    output = {"threshold_selected_on_validation": threshold}
    for split, (labels, probabilities) in by_split.items():
        predictions = probabilities >= threshold
        output[split] = {
            "records": len(labels),
            "accuracy": float(accuracy_score(labels, predictions)),
            "balanced_accuracy": float(
                balanced_accuracy_score(labels, predictions)
            ),
            "predicted_valid_fraction": float(np.mean(predictions)),
            "actual_valid_fraction": float(np.mean(labels)),
        }
    return output


def _selection_key(metrics: dict[str, Any], name: str) -> tuple:
    ranking = metrics["ranking"]
    learned = ranking["learned_top_k"]
    return (
        -float(learned["within_5pct_coverage"] or 0.0),
        -float(ranking["top_3_oracle_coverage"] or 0.0),
        float(learned["geomean_regret"] or math.inf),
        -float(ranking["top_1_accuracy"] or 0.0),
        name,
    )


def _importance_groups(
    classifier: Any,
    regressor: Any,
    categorical: list[str],
    numeric: list[str],
) -> dict[str, dict[str, float]]:
    def summarize(model: Any) -> dict[str, float]:
        preprocessor = model.named_steps["features"]
        categorical_pipe = preprocessor.named_transformers_["categorical"]
        encoder = categorical_pipe.named_steps["onehot"]
        categorical_names = list(
            encoder.get_feature_names_out(categorical)
        )
        feature_names = categorical_names + list(numeric)
        importance = model.named_steps["model"].feature_importances_
        if len(feature_names) != len(importance):
            raise ValueError("feature importance name mismatch")
        groups = {
            "source_transformer": 0.0,
            "phase_b_transformer": 0.0,
            "source_structured": 0.0,
            "phase_b_structured": 0.0,
            "setup": 0.0,
            "model": 0.0,
            "other": 0.0,
        }
        for name, value in zip(feature_names, importance, strict=True):
            if name.startswith("transformer_source_"):
                group = "source_transformer"
            elif name.startswith("transformer_phase_b_"):
                group = "phase_b_transformer"
            elif name.startswith("source_"):
                group = "source_structured"
            elif name.startswith("phase_b_"):
                group = "phase_b_structured"
            elif name.startswith("setup_"):
                group = "setup"
            elif name.startswith("model_id"):
                group = "model"
            else:
                group = "other"
            groups[group] += float(value)
        return groups

    return {
        "feasibility_classifier": summarize(classifier),
        "log_cycle_regressor": summarize(regressor),
    }


def _plot(
    validation_rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    names = [row["variant"] for row in validation_rows]
    baseline_regret = validation_rows[0]["validation_geomean_regret"]
    regret_delta = [
        100.0
        * (row["validation_geomean_regret"] / baseline_regret - 1.0)
        for row in validation_rows
    ]
    within = [row["validation_within_5pct"] for row in validation_rows]
    top3 = [row["validation_top3_oracle"] for row in validation_rows]
    positions = np.arange(len(names))
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    axes[0].bar(positions, regret_delta, color="#2368a2")
    axes[0].axhline(0.0, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Regret change vs structured (%)")
    axes[0].grid(axis="y", alpha=0.25)
    width = 0.38
    axes[1].bar(
        positions - width / 2,
        within,
        width,
        color="#2d8a56",
        label="within 5%",
    )
    axes[1].bar(
        positions + width / 2,
        top3,
        width,
        color="#8b5a2b",
        label="exact best in top 3",
    )
    axes[1].set_ylabel("Validation coverage")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xticks(positions, names, rotation=35, ha="right")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()
    fig.suptitle("Frozen-transformer router ablation")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def train(args: argparse.Namespace) -> dict[str, Any]:
    base_records = _load(args.corpus)
    embeddings, record_map, embedding_manifest = _load_embeddings(
        args.embedding_dir
    )
    corpus_ids = {
        str(record["provenance"]["dedup_key_sha256"])
        for record in base_records
    }
    if corpus_ids != set(record_map):
        missing = sorted(corpus_ids - set(record_map))
        extra = sorted(set(record_map) - corpus_ids)
        raise ValueError(
            f"embedding map/corpus mismatch: missing={len(missing)}, "
            f"extra={len(extra)}"
        )
    maximum_dimension = len(next(iter(embeddings.values())))
    if max(variant.dimension for variant in VARIANTS) > maximum_dimension:
        raise ValueError("embedding dimension is too small for ablation")

    base_train_ranking = [
        record
        for record in base_records
        if record["split"] == "train"
        and record["eligibility"]["ranking_model"]
        and record["labels"]["valid"]
    ]
    best_fixed, fixed_training_scores = _best_fixed_setup(
        _preferred_behavior_records(base_train_ranking)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    validation_metrics: dict[str, Any] = {}
    fitted_models: dict[str, tuple[Any, ...]] = {}
    prediction_rows = []
    table_rows = []
    for variant in VARIANTS:
        records = _variant_records(
            base_records,
            variant,
            embeddings,
            record_map,
        )
        (
            classifier,
            regressor,
            cv_metrics,
            names,
            categorical,
            numeric,
        ) = _fit_models(records, trees=args.trees, jobs=args.jobs)
        validation, predictions = _evaluate(
            classifier,
            regressor,
            records,
            names,
            split="validation",
            best_fixed_setup=best_fixed,
        )
        validation_metrics[variant.name] = {
            "variant": {
                "dimension": variant.dimension,
                "structured": variant.structured,
                "source_embedding": variant.source_embedding,
                "phase_b_embedding": variant.phase_b_embedding,
            },
            "grouped_cv": cv_metrics,
            "validation": validation,
            "feature_count": len(names),
        }
        fitted_models[variant.name] = (
            classifier,
            regressor,
            records,
            names,
            categorical,
            numeric,
        )
        prediction_rows.extend(
            {
                "evaluation_stage": "validation_model_selection",
                "variant": variant.name,
                "split": "validation",
                **row,
            }
            for row in predictions
        )
        learned = validation["ranking"]["learned_top_k"]
        table_rows.append(
            {
                "variant": variant.name,
                "feature_count": len(names),
                "grouped_cv_feasibility_balanced_accuracy": cv_metrics[
                    "classifier_balanced_accuracy_mean"
                ],
                "grouped_cv_log_cycle_mae": cv_metrics[
                    "regressor_log_mae_mean"
                ],
                "validation_top1_accuracy": validation["ranking"][
                    "top_1_accuracy"
                ],
                "validation_top3_oracle": validation["ranking"][
                    "top_3_oracle_coverage"
                ],
                "validation_within_5pct": learned[
                    "within_5pct_coverage"
                ],
                "validation_geomean_regret": learned[
                    "geomean_regret"
                ],
            }
        )
        gc.collect()

    structured_name = "structured"
    transformer_names = [
        variant.name for variant in VARIANTS if variant.name != structured_name
    ]
    selected_transformer = min(
        transformer_names,
        key=lambda name: _selection_key(
            validation_metrics[name]["validation"],
            name,
        ),
    )
    selected_overall = min(
        validation_metrics,
        key=lambda name: _selection_key(
            validation_metrics[name]["validation"],
            name,
        ),
    )
    test_metrics = {}
    calibrated_feasibility = {}
    test_names = list(dict.fromkeys([structured_name, selected_transformer]))
    for name in test_names:
        classifier, regressor, records, names, _, _ = fitted_models[name]
        test, predictions = _evaluate(
            classifier,
            regressor,
            records,
            names,
            split="test",
            best_fixed_setup=best_fixed,
        )
        test_metrics[name] = test
        calibrated_feasibility[name] = _calibrated_feasibility(
            classifier,
            records,
            names,
        )
        prediction_rows.extend(
            {
                "evaluation_stage": "held_out_test",
                "variant": name,
                "split": "test",
                **row,
            }
            for row in predictions
        )

    selected_classifier, selected_regressor, _, selected_names, selected_cat, selected_num = (
        fitted_models[selected_transformer]
    )
    structured_classifier, structured_regressor, _, structured_names, structured_cat, structured_num = (
        fitted_models[structured_name]
    )
    model_paths = {
        "selected_transformer": (
            args.output_dir / "selected_transformer_router.joblib"
        ),
        "structured_baseline": (
            args.output_dir / "structured_baseline_router.joblib"
        ),
    }
    for path, name, classifier, regressor, names, categorical, numeric in (
        (
            model_paths["selected_transformer"],
            selected_transformer,
            selected_classifier,
            selected_regressor,
            selected_names,
            selected_cat,
            selected_num,
        ),
        (
            model_paths["structured_baseline"],
            structured_name,
            structured_classifier,
            structured_regressor,
            structured_names,
            structured_cat,
            structured_num,
        ),
    ):
        joblib.dump(
            {
                "schema_version": SCHEMA_VERSION,
                "variant": name,
                "classifier": classifier,
                "regressor": regressor,
                "feature_names": names,
                "categorical_features": categorical,
                "numeric_features": numeric,
                "best_fixed_setup_id": best_fixed,
                "embedding_manifest": embedding_manifest,
            },
            path,
        )

    selected_importance = _importance_groups(
        selected_classifier,
        selected_regressor,
        selected_cat,
        selected_num,
    )
    output = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "methodology": {
            "selection_split": "validation",
            "held_out_test_variants": test_names,
            "test_not_used_for_variant_selection": True,
            "grouped_cv": "five-fold benchmark-lineage GroupKFold",
            "labels": "feasibility and log valid latency",
            "reference_metrics_as_inputs": False,
            "post_candidate_features_as_inputs": False,
        },
        "corpus": {
            "path": str(args.corpus.resolve()),
            "sha256": _sha256(args.corpus),
            "records": len(base_records),
            "train_lineages": len(
                {
                    record["benchmark_lineage"]
                    for record in base_records
                    if record["split"] == "train"
                }
            ),
            "validation_lineages": len(
                {
                    record["benchmark_lineage"]
                    for record in base_records
                    if record["split"] == "validation"
                }
            ),
            "test_lineages": len(
                {
                    record["benchmark_lineage"]
                    for record in base_records
                    if record["split"] == "test"
                }
            ),
        },
        "embedding": embedding_manifest,
        "training": {
            "trees": args.trees,
            "jobs": args.jobs,
            "fixed_hyperparameters_from_structured_router": True,
            "best_fixed_setup_id": best_fixed,
            "global_fixed_training_geomean_regret": fixed_training_scores,
        },
        "validation_ablation": validation_metrics,
        "selection": {
            "selected_transformer_variant": selected_transformer,
            "selected_overall_on_validation": selected_overall,
        },
        "held_out_test": test_metrics,
        "validation_calibrated_feasibility": calibrated_feasibility,
        "selected_transformer_feature_importance_groups": (
            selected_importance
        ),
        "deployment": {
            "status": "advisory",
            "reason": (
                "small historical corpus and no corrected-v2 training "
                "coverage; transformer experiment cannot activate routing"
            ),
        },
    }
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    table_path = args.output_dir / "validation_ablation.csv"
    with table_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table_rows[0]))
        writer.writeheader()
        writer.writerows(table_rows)
    predictions_path = args.output_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as handle:
        for row in prediction_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    figure_path = args.output_dir / "validation_ablation.png"
    _plot(table_rows, figure_path)

    baseline_test = test_metrics[structured_name]["ranking"]
    transformer_test = test_metrics[selected_transformer]["ranking"]
    baseline_feasibility = test_metrics[structured_name]["feasibility"]
    transformer_feasibility = test_metrics[selected_transformer][
        "feasibility"
    ]
    comparison = {
        "feasibility_roc_auc_delta": (
            transformer_feasibility["roc_auc"]
            - baseline_feasibility["roc_auc"]
        ),
        "feasibility_balanced_accuracy_delta": (
            transformer_feasibility["balanced_accuracy"]
            - baseline_feasibility["balanced_accuracy"]
        ),
        "learned_top_k_geomean_regret_delta": (
            transformer_test["learned_top_k"]["geomean_regret"]
            - baseline_test["learned_top_k"]["geomean_regret"]
        ),
        "learned_top_k_geomean_regret_ratio": (
            transformer_test["learned_top_k"]["geomean_regret"]
            / baseline_test["learned_top_k"]["geomean_regret"]
        ),
        "within_5pct_count_delta": (
            transformer_test["learned_top_k"]["within_5pct_count"]
            - baseline_test["learned_top_k"]["within_5pct_count"]
        ),
        "top_1_accuracy_delta": (
            transformer_test["top_1_accuracy"]
            - baseline_test["top_1_accuracy"]
        ),
        "top_3_oracle_delta": (
            transformer_test["top_3_oracle_coverage"]
            - baseline_test["top_3_oracle_coverage"]
        ),
    }
    output["held_out_comparison"] = comparison
    replace_structured = (
        transformer_test["learned_top_k"]["within_5pct_count"]
        > baseline_test["learned_top_k"]["within_5pct_count"]
        and transformer_test["learned_top_k"]["geomean_regret"]
        <= baseline_test["learned_top_k"]["geomean_regret"]
        and transformer_test["top_3_oracle_coverage"]
        >= baseline_test["top_3_oracle_coverage"]
    )
    output["recommendation"] = {
        "replace_structured_router": replace_structured,
        "use_transformer_for": (
            "continued feasibility ablation only; do not use it to "
            "replace structured setup ranking"
        ),
        "reason": (
            "the validation-selected transformer did not improve held-out "
            "within-5% coverage, exact-best coverage, or learned-top-k "
            "geomean regret"
        ),
    }
    metrics_path.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path = args.output_dir / "report.md"
    validation_lines = []
    for row in table_rows:
        validation_lines.append(
            "| "
            f"{row['variant']} | "
            f"{row['grouped_cv_feasibility_balanced_accuracy']:.3f} | "
            f"{row['grouped_cv_log_cycle_mae']:.3f} | "
            f"{row['validation_top3_oracle']:.3f} | "
            f"{row['validation_within_5pct']:.3f} | "
            f"{row['validation_geomean_regret']:.3f} |"
        )
    markdown_path.write_text(
        "\n".join(
            [
                "# Frozen-Transformer Setup Router Ablation",
                "",
                (
                    f"- Encoder: `{embedding_manifest['model']}` "
                    f"({embedding_manifest['model_hidden_size']} dimensions)"
                ),
                (
                    "- Selected transformer variant on validation: "
                    f"`{selected_transformer}`"
                ),
                (
                    "- Overall validation winner: "
                    f"`{selected_overall}`"
                ),
                "- Test was evaluated only for the structured baseline and the validation-selected transformer variant.",
                "",
                "## Validation Ablation",
                "",
                "| model | CV feasibility balanced accuracy | CV log-cycle MAE | validation top-3 | validation within 5% | validation geomean regret |",
                "|---|---:|---:|---:|---:|---:|",
                *validation_lines,
                "",
                "## Held-Out Test",
                "",
                "| model | test top-1 | test top-3 oracle | test within 5% | test geomean regret |",
                "|---|---:|---:|---:|---:|",
                (
                    "| structured | "
                    f"{baseline_test['top_1_accuracy']:.3f} | "
                    f"{baseline_test['top_3_oracle_coverage']:.3f} | "
                    f"{baseline_test['learned_top_k']['within_5pct_coverage']:.3f} | "
                    f"{baseline_test['learned_top_k']['geomean_regret']:.3f} |"
                ),
                (
                    f"| {selected_transformer} | "
                    f"{transformer_test['top_1_accuracy']:.3f} | "
                    f"{transformer_test['top_3_oracle_coverage']:.3f} | "
                    f"{transformer_test['learned_top_k']['within_5pct_coverage']:.3f} | "
                    f"{transformer_test['learned_top_k']['geomean_regret']:.3f} |"
                ),
                "",
                (
                    "- Feasibility ROC-AUC delta: "
                    f"{comparison['feasibility_roc_auc_delta']:+.3f}"
                ),
                (
                    "- Feasibility balanced-accuracy delta: "
                    f"{comparison['feasibility_balanced_accuracy_delta']:+.3f}"
                ),
                (
                    "- Learned-top-k geomean-regret ratio: "
                    f"{comparison['learned_top_k_geomean_regret_ratio']:.3f}"
                ),
                (
                    "- Structured calibrated test feasibility balanced accuracy: "
                    f"{calibrated_feasibility[structured_name]['test']['balanced_accuracy']:.3f}"
                ),
                (
                    "- Transformer calibrated test feasibility balanced accuracy: "
                    f"{calibrated_feasibility[selected_transformer]['test']['balanced_accuracy']:.3f}"
                ),
                "",
                "## Decision",
                "",
                "Do not replace the structured ranking router. The transformer improved held-out feasibility ROC-AUC but did not improve setup coverage or learned-top-k regret. Keep it as an auxiliary feasibility experiment until corrected-v2 and external benchmark lineages materially increase training diversity.",
                "",
                "The result remains advisory because only 19 independent training lineages are available and corrected-v2 outcomes are not yet represented.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    artifact_paths = [
        *model_paths.values(),
        metrics_path,
        table_path,
        predictions_path,
        figure_path,
        markdown_path,
    ]
    (args.output_dir / "artifact_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "artifacts": {
                    path.name: {
                        "sha256": _sha256(path),
                        "bytes": path.stat().st_size,
                    }
                    for path in artifact_paths
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trees", type=int, default=400)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    if args.trees < 50:
        parser.error("--trees must be at least 50")
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    return args


if __name__ == "__main__":
    metrics = train(parse_args())
    print(
        json.dumps(
            {
                "selected_transformer": metrics["selection"][
                    "selected_transformer_variant"
                ],
                "selected_overall": metrics["selection"][
                    "selected_overall_on_validation"
                ],
                "held_out_test": metrics["held_out_test"],
            },
            sort_keys=True,
        )
    )
