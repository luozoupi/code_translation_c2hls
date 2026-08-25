#!/usr/bin/env python3
"""Train and evaluate ExtraTrees models for C2HLS setup routing."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import geometric_mean
from typing import Any

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    mean_absolute_error,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


MODEL_SCHEMA_VERSION = "c2hls.learned-setup-router.v1"
MANDATORY_BASELINE_SUFFIX = ":multistep:skillless"
INVALID_REGRET_PENALTY = 100.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _feature_schema(
    records: list[dict[str, Any]],
) -> tuple[list[str], list[str], list[str]]:
    names = sorted(
        {name for record in records for name in record["features"]}
    )
    categorical = [
        name
        for name in names
        if any(
            isinstance(record["features"].get(name), str)
            for record in records
        )
    ]
    numeric = [name for name in names if name not in categorical]
    leaked = [
        name
        for name in names
        if name.lower().startswith(
            ("reference_", "ground_truth_", "gold_")
        )
        or any(
            token in name.lower()
            for token in (
                "reference_cycles",
                "ground_truth",
                "final_report",
                "selected_latency",
                "winner",
            )
        )
    ]
    if leaked:
        raise ValueError(f"forbidden router input features: {leaked}")
    return names, categorical, numeric


def _matrix(
    records: list[dict[str, Any]],
    names: list[str],
) -> list[list[Any]]:
    return [
        [record["features"].get(name) for name in names]
        for record in records
    ]


def _preprocessor(
    categorical: list[str],
    numeric: list[str],
    all_names: list[str],
) -> ColumnTransformer:
    categorical_indices = [all_names.index(name) for name in categorical]
    numeric_indices = [all_names.index(name) for name in numeric]
    return ColumnTransformer(
        [
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "impute",
                            SimpleImputer(
                                strategy="constant",
                                fill_value="missing",
                            ),
                        ),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                categorical_indices,
            ),
            (
                "numeric",
                SimpleImputer(strategy="median"),
                numeric_indices,
            ),
        ],
        remainder="drop",
    )


def _classifier_pipeline(
    categorical: list[str],
    numeric: list[str],
    names: list[str],
) -> Pipeline:
    return Pipeline(
        [
            ("features", _preprocessor(categorical, numeric, names)),
            (
                "model",
                ExtraTreesClassifier(
                    n_estimators=400,
                    random_state=46,
                    n_jobs=-1,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def _regressor_pipeline(
    categorical: list[str],
    numeric: list[str],
    names: list[str],
) -> Pipeline:
    return Pipeline(
        [
            ("features", _preprocessor(categorical, numeric, names)),
            (
                "model",
                ExtraTreesRegressor(
                    n_estimators=400,
                    random_state=46,
                    n_jobs=-1,
                    criterion="squared_error",
                ),
            ),
        ]
    )


def _valid_probability(model: Pipeline, matrix: list[list[Any]]) -> np.ndarray:
    probabilities = model.predict_proba(matrix)
    classes = list(model.named_steps["model"].classes_)
    if True not in classes and 1 not in classes:
        return np.zeros(len(matrix), dtype=float)
    positive_index = classes.index(True) if True in classes else classes.index(1)
    return probabilities[:, positive_index]


def _classifier_metrics(
    model: Pipeline,
    records: list[dict[str, Any]],
    names: list[str],
) -> dict[str, Any]:
    labels = np.asarray(
        [bool(record["labels"]["valid"]) for record in records]
    )
    probabilities = _valid_probability(model, _matrix(records, names))
    predictions = probabilities >= 0.5
    metrics = {
        "records": len(records),
        "class_counts": dict(Counter(str(value) for value in labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, predictions)
        ),
        "roc_auc": None,
    }
    if len(set(labels)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(labels, probabilities))
    return metrics


def _best_fixed_setup(
    train_outcomes: list[dict[str, Any]],
) -> tuple[str, dict[str, float]]:
    by_problem: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in train_outcomes:
        by_problem[record["problem"]].append(record)
    regrets: dict[str, list[float]] = defaultdict(list)
    for records in by_problem.values():
        valid = [record for record in records if record["labels"]["valid"]]
        best = min(
            record["labels"]["latency_cycles"] for record in valid
        )
        for record in valid:
            regrets[record["setup"]["setup_id"]].append(
                record["labels"]["latency_cycles"] / best
            )
    scores = {
        setup_id: geometric_mean(values)
        for setup_id, values in regrets.items()
    }
    return min(scores, key=lambda setup_id: (scores[setup_id], setup_id)), scores


def _preferred_behavior_records(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    corrected_present = any(
        record["features"].get("setup_behavior_version") == "corrected_v2"
        for record in records
    )
    preferred = "corrected_v2" if corrected_present else "legacy_v1"
    return [
        record
        for record in records
        if record["features"].get("setup_behavior_version") == preferred
    ]


def _percentile95(values: list[float]) -> float | None:
    return float(np.percentile(values, 95)) if values else None


def _ranking_metrics(
    classifier: Pipeline,
    regressor: Pipeline,
    records: list[dict[str, Any]],
    names: list[str],
    *,
    best_fixed_setup: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    outcomes = _preferred_behavior_records([
        record
        for record in records
        if record["record_kind"] == "setup_outcome"
        and record["eligibility"]["ranking_model"]
    ])
    probabilities = _valid_probability(
        classifier, _matrix(outcomes, names)
    )
    predicted_log_cycles = regressor.predict(_matrix(outcomes, names))
    scored = []
    for record, probability, log_cycles in zip(
        outcomes,
        probabilities,
        predicted_log_cycles,
        strict=True,
    ):
        predicted_cycles = math.exp(float(log_cycles))
        risk_adjusted = predicted_cycles / max(float(probability), 0.05)
        scored.append(
            {
                "problem": record["problem"],
                "setup_id": record["setup"]["setup_id"],
                "setup_fingerprint": record["setup"]["setup_fingerprint"],
                "predicted_valid_probability": float(probability),
                "predicted_log_cycles": float(log_cycles),
                "predicted_cycles": predicted_cycles,
                "predicted_valid_latency_score": risk_adjusted,
                "actual_valid": record["labels"]["valid"],
                "actual_cycles": record["labels"]["latency_cycles"],
                "actual_regret": record["labels"]["regret"],
                "actual_is_best": record["labels"]["is_best_setup"],
            }
        )
    by_problem: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        by_problem[row["problem"]].append(row)

    top1_accuracy = 0
    top3_oracle = 0
    top3_valid_flags: list[bool] = []
    top1_regrets: list[float] = []
    learned_regrets: list[float] = []
    learned_within5 = 0
    learned_valid = 0
    fixed_regrets: list[float] = []
    prediction_rows: list[dict[str, Any]] = []
    for problem, rows in sorted(by_problem.items()):
        predicted = sorted(
            rows,
            key=lambda row: (
                row["predicted_valid_latency_score"],
                row["setup_fingerprint"],
            ),
        )
        actual_valid = [
            row for row in rows if row["actual_valid"]
        ]
        actual_best = min(
            actual_valid,
            key=lambda row: (
                row["actual_cycles"],
                row["setup_fingerprint"],
            ),
        )
        top1 = predicted[0]
        top3 = predicted[:3]
        top1_accuracy += int(top1["setup_id"] == actual_best["setup_id"])
        top3_oracle += int(
            actual_best["setup_id"] in {row["setup_id"] for row in top3}
        )
        top3_valid_flags.extend(row["actual_valid"] for row in top3)
        top1_regrets.append(
            float(top1["actual_regret"])
            if top1["actual_valid"]
            and top1["actual_regret"] is not None
            else INVALID_REGRET_PENALTY
        )

        mandatory = next(
            (
                row
                for row in rows
                if row["setup_id"].endswith(MANDATORY_BASELINE_SUFFIX)
            ),
            None,
        )
        learned_candidates = [mandatory] if mandatory is not None else []
        learned_candidates.extend(
            row
            for row in predicted
            if mandatory is None or row["setup_id"] != mandatory["setup_id"]
        )
        deduped = list(
            {row["setup_id"]: row for row in learned_candidates}.values()
        )[:3]
        valid_learned = [
            row for row in deduped if row["actual_valid"]
        ]
        learned_winner = (
            min(
                valid_learned,
                key=lambda row: (
                    row["actual_cycles"],
                    row["setup_fingerprint"],
                ),
            )
            if valid_learned
            else None
        )
        if learned_winner is not None:
            learned_valid += 1
            regret = float(learned_winner["actual_regret"])
            learned_regrets.append(regret)
            learned_within5 += int(regret <= 1.05)

        fixed = next(
            row for row in rows if row["setup_id"] == best_fixed_setup
        )
        fixed_regrets.append(
            float(fixed["actual_regret"])
            if fixed["actual_valid"]
            and fixed["actual_regret"] is not None
            else INVALID_REGRET_PENALTY
        )
        for rank, row in enumerate(predicted, start=1):
            prediction_rows.append(
                {
                    **row,
                    "predicted_rank": rank,
                    "actual_best_setup_id": actual_best["setup_id"],
                    "learned_top_k_evaluated": row["setup_id"]
                    in {item["setup_id"] for item in deduped},
                    "learned_top_k_winner": bool(
                        learned_winner
                        and row["setup_id"] == learned_winner["setup_id"]
                    ),
                }
            )

    count = len(by_problem)
    metrics = {
        "benchmark_count": count,
        "top_1_accuracy": top1_accuracy / count if count else None,
        "top_3_oracle_coverage": top3_oracle / count if count else None,
        "top_3_actual_validity": (
            sum(top3_valid_flags) / len(top3_valid_flags)
            if top3_valid_flags else None
        ),
        "top_1_geomean_regret": (
            geometric_mean(top1_regrets) if top1_regrets else None
        ),
        "top_1_p95_regret": _percentile95(top1_regrets),
        "learned_top_k": {
            "candidate_count": 3,
            "candidate_savings_vs_exhaustive": 0.7,
            "validity": learned_valid / count if count else None,
            "within_5pct_coverage": (
                learned_within5 / count if count else None
            ),
            "within_5pct_count": learned_within5,
            "geomean_regret": (
                geometric_mean(learned_regrets)
                if learned_regrets else None
            ),
            "p95_regret": _percentile95(learned_regrets),
        },
        "global_best_fixed": {
            "setup_id": best_fixed_setup,
            "geomean_regret": (
                geometric_mean(fixed_regrets) if fixed_regrets else None
            ),
            "p95_regret": _percentile95(fixed_regrets),
            "candidate_count": 1,
            "candidate_savings_vs_exhaustive": 0.9,
        },
        "exhaustive": {
            "geomean_regret": 1.0,
            "p95_regret": 1.0,
            "candidate_count": 10,
            "candidate_savings_vs_exhaustive": 0.0,
            "validity": 1.0,
        },
    }
    return metrics, prediction_rows


def train(args: argparse.Namespace) -> dict[str, Any]:
    records = _load(args.corpus)
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
    if len(
        {record["benchmark_lineage"] for record in train_feasibility}
    ) < 5:
        raise ValueError("fewer than five train lineages")

    classifier_search = GridSearchCV(
        _classifier_pipeline(categorical, numeric, names),
        {
            "model__max_depth": [None, 12],
            "model__min_samples_leaf": [1, 3],
            "model__max_features": ["sqrt", 0.75],
        },
        scoring="balanced_accuracy",
        cv=GroupKFold(n_splits=5),
        n_jobs=-1,
        refit=True,
    )
    classifier_search.fit(
        _matrix(train_feasibility, names),
        [record["labels"]["valid"] for record in train_feasibility],
        groups=[
            record["benchmark_lineage"] for record in train_feasibility
        ],
    )
    regressor_search = GridSearchCV(
        _regressor_pipeline(categorical, numeric, names),
        {
            "model__max_depth": [None, 12],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": [0.75, 1.0],
        },
        scoring="neg_mean_absolute_error",
        cv=GroupKFold(n_splits=5),
        n_jobs=-1,
        refit=True,
    )
    regressor_search.fit(
        _matrix(train_ranking, names),
        [record["labels"]["log_latency_cycles"] for record in train_ranking],
        groups=[record["benchmark_lineage"] for record in train_ranking],
    )
    classifier = classifier_search.best_estimator_
    regressor = regressor_search.best_estimator_
    best_fixed, fixed_training_scores = _best_fixed_setup(
        _preferred_behavior_records(train_ranking)
    )

    output: dict[str, Any] = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "training": {
            "feasibility_records": len(train_feasibility),
            "ranking_records": len(train_ranking),
            "benchmark_grouped_cv_folds": 5,
            "classifier_best_params": classifier_search.best_params_,
            "classifier_best_cv_balanced_accuracy": float(
                classifier_search.best_score_
            ),
            "regressor_best_params": regressor_search.best_params_,
            "regressor_best_cv_log_mae": float(
                -regressor_search.best_score_
            ),
            "global_fixed_training_geomean_regret": fixed_training_scores,
        },
        "feature_schema": {
            "feature_names": names,
            "categorical": categorical,
            "numeric": numeric,
            "allowed_sources": [
                "plain source",
                "frozen Phase-B CSim/CSynth",
                "versioned setup definition",
            ],
        },
        "evaluation": {},
    }
    all_predictions = []
    for split in ("validation", "test"):
        split_records = [
            record for record in records if record["split"] == split
        ]
        classification = _classifier_metrics(
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
            best_fixed_setup=best_fixed,
        )
        output["evaluation"][split] = {
            "feasibility": classification,
            "ranking": ranking,
        }
        all_predictions.extend(
            {"split": split, **prediction} for prediction in predictions
        )

    test_ranking = output["evaluation"]["test"]["ranking"]
    learned = test_ranking["learned_top_k"]
    fixed = test_ranking["global_best_fixed"]
    threshold_checks = {
        "held_out_top3_validity_100pct": (
            test_ranking["top_3_actual_validity"] == 1.0
        ),
        "three_of_four_within_5pct": (
            learned["within_5pct_count"] >= 3
        ),
        "geomean_regret_at_most_1_15": (
            learned["geomean_regret"] is not None
            and learned["geomean_regret"] <= 1.15
        ),
        "no_worse_than_best_fixed": (
            learned["geomean_regret"] is not None
            and learned["geomean_regret"] <= fixed["geomean_regret"]
        ),
    }
    corrected_training_present = any(
        record["features"].get("setup_behavior_version") == "corrected_v2"
        and record["split"] == "train"
        for record in records
    )
    output["deployment"] = {
        "status": (
            "active"
            if all(threshold_checks.values()) and corrected_training_present
            else "advisory"
        ),
        "threshold_checks": threshold_checks,
        "corrected_v2_training_coverage": corrected_training_present,
        "reason": (
            "held-out thresholds and corrected-v2 coverage passed"
            if all(threshold_checks.values()) and corrected_training_present
            else "router remains advisory until every held-out threshold "
            "passes and corrected-v2 outcomes are represented in training"
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "setup_router_extratrees.joblib"
    joblib.dump(
        {
            "schema_version": MODEL_SCHEMA_VERSION,
            "classifier": classifier,
            "regressor": regressor,
            "feature_names": names,
            "categorical_features": categorical,
            "numeric_features": numeric,
            "best_fixed_setup_id": best_fixed,
        },
        model_path,
    )
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    validation_ranking = output["evaluation"]["validation"]["ranking"]
    test_ranking = output["evaluation"]["test"]["ranking"]
    markdown_path = args.output_dir / "metrics.md"
    markdown_path.write_text(
        "\n".join(
            [
                "# Learned Setup Router",
                "",
                f"- Deployment status: **{output['deployment']['status']}**",
                (
                    "- Grouped-CV feasibility balanced accuracy: "
                    f"{output['training']['classifier_best_cv_balanced_accuracy']:.3f}"
                ),
                (
                    "- Grouped-CV log-cycle MAE: "
                    f"{output['training']['regressor_best_cv_log_mae']:.3f}"
                ),
                "",
                "| split | top-1 accuracy | top-3 oracle | learned top-k within 5% | learned top-k geomean regret |",
                "|---|---:|---:|---:|---:|",
                (
                    "| validation | "
                    f"{validation_ranking['top_1_accuracy']:.3f} | "
                    f"{validation_ranking['top_3_oracle_coverage']:.3f} | "
                    f"{validation_ranking['learned_top_k']['within_5pct_coverage']:.3f} | "
                    f"{validation_ranking['learned_top_k']['geomean_regret']:.3f} |"
                ),
                (
                    "| test | "
                    f"{test_ranking['top_1_accuracy']:.3f} | "
                    f"{test_ranking['top_3_oracle_coverage']:.3f} | "
                    f"{test_ranking['learned_top_k']['within_5pct_coverage']:.3f} | "
                    f"{test_ranking['learned_top_k']['geomean_regret']:.3f} |"
                ),
                "",
                output["deployment"]["reason"] + ".",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    schema_path = args.output_dir / "feature_schema.json"
    schema_path.write_text(
        json.dumps(output["feature_schema"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    predictions_path = args.output_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as handle:
        for row in all_predictions:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    ood_path = args.output_dir / "ood_evaluation_manifest.json"
    ood_path.write_text(
        json.dumps(
            {
                "schema_version": MODEL_SCHEMA_VERSION,
                "status": "pending_corrected_runs",
                "datasets": {
                    "HLS-Eval": "non-overlapping lineages only",
                    "MachSuite": "adapted compatible kernels only",
                    "HLSPilot": "compatible non-PolyBench lineages only",
                },
                "polybench_duplicate_policy": (
                    "same lineage; excluded from external generalization"
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    artifacts = {
        path.name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
        for path in (
            model_path,
            metrics_path,
            markdown_path,
            schema_path,
            predictions_path,
            ood_path,
        )
    }
    (args.output_dir / "artifact_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": MODEL_SCHEMA_VERSION,
                "source_corpus": str(args.corpus.resolve()),
                "source_corpus_sha256": _sha256(args.corpus),
                "artifacts": artifacts,
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
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    metrics = train(arguments)
    print(
        json.dumps(
            {
                "deployment": metrics["deployment"]["status"],
                "validation": metrics["evaluation"]["validation"]["ranking"],
                "test": metrics["evaluation"]["test"]["ranking"],
            },
            sort_keys=True,
        )
    )
