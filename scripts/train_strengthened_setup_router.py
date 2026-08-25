#!/usr/bin/env python3
"""Train and compare stronger small-data setup-routing objectives."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import geometric_mean
from typing import Any

import joblib
import matplotlib
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold, cross_val_score

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
    MANDATORY_BASELINE_SUFFIX,
    _best_fixed_setup,
    _classifier_pipeline,
    _feature_schema,
    _load,
    _matrix,
    _preferred_behavior_records,
    _regressor_pipeline,
    _sha256,
    _valid_probability,
)


SCHEMA_VERSION = "c2hls.strengthened-setup-router.v1"
INFERENCE_BUNDLE_SCHEMA_VERSION = (
    "c2hls.strengthened-setup-router-inference-bundle.v1"
)
INVALID_REGRET_PENALTY = 100.0
RETRIEVAL_INVALID_LOG_COST = math.log(20.0)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _record_id(record: dict[str, Any]) -> str:
    value = str(
        (record.get("provenance") or {}).get("dedup_key_sha256") or ""
    )
    if not value:
        raise ValueError("router record lacks dedup_key_sha256")
    return value


def _is_setup_feature(name: str) -> bool:
    return name == "model_id" or name.startswith("setup_")


def _canonicalize_phase_b(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Use one mandatory-baseline context for every setup in a benchmark."""

    baseline_context: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        if record["record_kind"] != "setup_outcome":
            continue
        setup_id = str(record["setup"]["setup_id"])
        if not setup_id.endswith(MANDATORY_BASELINE_SUFFIX):
            continue
        key = (
            str(record["problem"]),
            str(record["setup"]["behavior_version"]),
        )
        context = {
            name: value
            for name, value in record["features"].items()
            if not _is_setup_feature(name)
        }
        if key in baseline_context and baseline_context[key] != context:
            raise ValueError(f"ambiguous mandatory baseline context: {key}")
        baseline_context[key] = context

    output = []
    for record in records:
        key = (
            str(record["problem"]),
            str(record["setup"]["behavior_version"]),
        )
        if key not in baseline_context:
            raise ValueError(f"missing mandatory baseline context: {key}")
        setup_features = {
            name: value
            for name, value in record["features"].items()
            if _is_setup_feature(name)
        }
        output.append(
            {
                **record,
                "features": {
                    **baseline_context[key],
                    **setup_features,
                },
            }
        )
    return output


def _ranking_outcomes(
    records: list[dict[str, Any]],
    *,
    split: str | None = None,
) -> list[dict[str, Any]]:
    selected = [
        record
        for record in records
        if record["record_kind"] == "setup_outcome"
        and record["eligibility"]["ranking_model"]
        and (split is None or record["split"] == split)
    ]
    return _preferred_behavior_records(selected)


def _fit_direct(
    records: list[dict[str, Any]],
    *,
    target: str,
    trees: int,
    jobs: int,
    compute_cv: bool = True,
) -> tuple[dict[str, float], dict[str, Any], dict[str, Any]]:
    names, categorical, numeric = _feature_schema(records)
    train_feasibility = [
        record
        for record in records
        if record["split"] == "train"
        and record["eligibility"]["feasibility_model"]
    ]
    train_ranking = [
        record
        for record in _ranking_outcomes(records, split="train")
        if record["labels"]["valid"]
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
    if target == "absolute_log_cycles":
        targets = [
            float(record["labels"]["log_latency_cycles"])
            for record in train_ranking
        ]
    elif target == "relative_log_cycles":
        targets = [
            float(record["labels"]["log_latency_cycles"])
            - float(
                record.get("_phase_b_log_latency_cycles_for_target")
                or record["features"]["phase_b_log_latency_cycles"]
            )
            for record in train_ranking
        ]
    elif target == "log_regret":
        targets = [
            math.log(float(record["labels"]["regret"]))
            for record in train_ranking
        ]
    else:
        raise ValueError(f"unknown direct target: {target}")

    classifier_cv = (
        cross_val_score(
            classifier,
            _matrix(train_feasibility, names),
            [record["labels"]["valid"] for record in train_feasibility],
            groups=[
                record["benchmark_lineage"]
                for record in train_feasibility
            ],
            scoring="balanced_accuracy",
            cv=GroupKFold(n_splits=5),
            n_jobs=1,
        )
        if compute_cv
        else np.asarray([], dtype=float)
    )
    regressor_cv = (
        -cross_val_score(
            regressor,
            _matrix(train_ranking, names),
            targets,
            groups=[
                record["benchmark_lineage"] for record in train_ranking
            ],
            scoring="neg_mean_absolute_error",
            cv=GroupKFold(n_splits=5),
            n_jobs=1,
        )
        if compute_cv
        else np.asarray([], dtype=float)
    )
    classifier.fit(
        _matrix(train_feasibility, names),
        [record["labels"]["valid"] for record in train_feasibility],
    )
    regressor.fit(_matrix(train_ranking, names), targets)

    outcomes = _ranking_outcomes(records)
    probabilities = _valid_probability(
        classifier,
        _matrix(outcomes, names),
    )
    predictions = regressor.predict(_matrix(outcomes, names))
    scores = {
        _record_id(record): (
            math.exp(float(prediction)) / max(float(probability), 0.05)
        )
        for record, probability, prediction in zip(
            outcomes,
            probabilities,
            predictions,
            strict=True,
        )
    }
    training = {
        "target": target,
        "feature_count": len(names),
        "feasibility_records": len(train_feasibility),
        "ranking_records": len(train_ranking),
        "grouped_cv_feasibility_balanced_accuracy": (
            float(np.mean(classifier_cv)) if compute_cv else None
        ),
        "grouped_cv_feasibility_balanced_accuracy_std": (
            float(np.std(classifier_cv)) if compute_cv else None
        ),
        "grouped_cv_regression_log_mae": (
            float(np.mean(regressor_cv)) if compute_cv else None
        ),
        "grouped_cv_regression_log_mae_std": (
            float(np.std(regressor_cv)) if compute_cv else None
        ),
    }
    model = {
        "kind": "direct",
        "target": target,
        "classifier": classifier,
        "regressor": regressor,
        "feature_names": names,
        "categorical_features": categorical,
        "numeric_features": numeric,
    }
    return scores, training, model


def _within5_training_weights(
    records: list[dict[str, Any]],
) -> np.ndarray:
    positive_counts: dict[str, int] = defaultdict(int)
    for record in records:
        if record["labels"].get("within_5pct_of_best") is True:
            positive_counts[str(record["setup"]["setup_id"])] += 1
    maximum = max(positive_counts.values(), default=1)
    weights = []
    for record in records:
        labels = record["labels"]
        if labels.get("within_5pct_of_best") is True:
            count = positive_counts[str(record["setup"]["setup_id"])]
            weights.append(math.sqrt(maximum / max(count, 1)))
            continue
        regret = labels.get("regret")
        if labels.get("valid") is True and isinstance(
            regret, (int, float)
        ) and float(regret) <= 1.25:
            weights.append(2.0)
        elif labels.get("valid") is not True:
            weights.append(1.5)
        else:
            weights.append(1.0)
    return np.asarray(weights, dtype=float)


def _fit_within5(
    records: list[dict[str, Any]],
    *,
    trees: int,
    jobs: int,
    compute_cv: bool = True,
) -> tuple[dict[str, float], dict[str, Any], dict[str, Any]]:
    train = _ranking_outcomes(records, split="train")
    names, categorical, numeric = _feature_schema(train)
    labels = [
        bool(record["labels"].get("within_5pct_of_best"))
        and bool(record["labels"].get("valid"))
        for record in train
    ]
    groups = [record["benchmark_lineage"] for record in train]
    model = _classifier_pipeline(categorical, numeric, names)
    model.set_params(
        model__n_estimators=trees,
        model__n_jobs=jobs,
        model__max_depth=12,
        model__max_features=0.75,
        model__min_samples_leaf=2,
        model__class_weight="balanced",
    )
    cv_scores = (
        cross_val_score(
            model,
            _matrix(train, names),
            labels,
            groups=groups,
            scoring="balanced_accuracy",
            cv=GroupKFold(n_splits=5),
            n_jobs=1,
        )
        if compute_cv
        else np.asarray([], dtype=float)
    )
    weights = _within5_training_weights(train)
    model.fit(
        _matrix(train, names),
        labels,
        model__sample_weight=weights,
    )
    outcomes = _ranking_outcomes(records)
    probabilities = _valid_probability(
        model,
        _matrix(outcomes, names),
    )
    scores = {
        _record_id(record): 1.0 - float(probability)
        for record, probability in zip(
            outcomes,
            probabilities,
            strict=True,
        )
    }
    training = {
        "target": "within_5pct_of_best",
        "feature_count": len(names),
        "ranking_records": len(train),
        "positive_records": int(sum(labels)),
        "hard_negative_weight": 2.0,
        "invalid_weight": 1.5,
        "rare_positive_setup_weighting": "sqrt_inverse_frequency",
        "grouped_cv_balanced_accuracy": (
            float(np.mean(cv_scores)) if compute_cv else None
        ),
        "grouped_cv_balanced_accuracy_std": (
            float(np.std(cv_scores)) if compute_cv else None
        ),
    }
    bundle = {
        "kind": "within5_classifier",
        "classifier": model,
        "feature_names": names,
        "categorical_features": categorical,
        "numeric_features": numeric,
    }
    return scores, training, bundle


def _context_features(record: dict[str, Any]) -> dict[str, Any]:
    return {
        name: value
        for name, value in record["features"].items()
        if not _is_setup_feature(name)
    }


def _pair_features(
    first: dict[str, Any],
    second: dict[str, Any],
) -> dict[str, Any]:
    return {
        **_context_features(first),
        "first_setup_id": str(first["setup"]["setup_id"]),
        "second_setup_id": str(second["setup"]["setup_id"]),
        "first_setup_strategy": str(first["setup"]["strategy"]),
        "second_setup_strategy": str(second["setup"]["strategy"]),
        "first_setup_skill_scope": str(first["setup"]["skill_scope"]),
        "second_setup_skill_scope": str(second["setup"]["skill_scope"]),
        "first_setup_router_version": float(
            first["setup"]["router_version"]
        ),
        "second_setup_router_version": float(
            second["setup"]["router_version"]
        ),
    }


def _first_wins(
    first: dict[str, Any],
    second: dict[str, Any],
) -> bool | None:
    first_valid = first["labels"]["valid"] is True
    second_valid = second["labels"]["valid"] is True
    if first_valid != second_valid:
        return first_valid
    if not first_valid:
        return None
    first_cycles = float(first["labels"]["latency_cycles"])
    second_cycles = float(second["labels"]["latency_cycles"])
    if first_cycles == second_cycles:
        return None
    return first_cycles < second_cycles


def _outcome_groups(
    records: list[dict[str, Any]],
    *,
    split: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in _ranking_outcomes(records, split=split):
        grouped[str(record["problem"])].append(record)
    return dict(grouped)


def _pair_training_records(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    pairs = []
    for problem, outcomes in _outcome_groups(
        records,
        split="train",
    ).items():
        ordered = sorted(
            outcomes,
            key=lambda record: record["setup"]["setup_id"],
        )
        for first_index, first in enumerate(ordered):
            for second in ordered[first_index + 1 :]:
                label = _first_wins(first, second)
                if label is None:
                    continue
                pairs.append(
                    {
                        "features": _pair_features(first, second),
                        "label": bool(label),
                        "benchmark_lineage": first[
                            "benchmark_lineage"
                        ],
                        "problem": problem,
                    }
                )
                pairs.append(
                    {
                        "features": _pair_features(second, first),
                        "label": not bool(label),
                        "benchmark_lineage": first[
                            "benchmark_lineage"
                        ],
                        "problem": problem,
                    }
                )
    return pairs


def _fit_pairwise(
    records: list[dict[str, Any]],
    *,
    trees: int,
    jobs: int,
    compute_cv: bool = True,
) -> tuple[dict[str, float], dict[str, Any], dict[str, Any]]:
    pairs = _pair_training_records(records)
    names, categorical, numeric = _feature_schema(pairs)
    model = _classifier_pipeline(categorical, numeric, names)
    model.set_params(
        model__n_estimators=trees,
        model__n_jobs=jobs,
        model__max_depth=12,
        model__max_features=0.75,
        model__min_samples_leaf=2,
        model__class_weight="balanced",
    )
    labels = [pair["label"] for pair in pairs]
    groups = [pair["benchmark_lineage"] for pair in pairs]
    cv_scores = (
        cross_val_score(
            model,
            _matrix(pairs, names),
            labels,
            groups=groups,
            scoring="balanced_accuracy",
            cv=GroupKFold(n_splits=5),
            n_jobs=1,
        )
        if compute_cv
        else np.asarray([], dtype=float)
    )
    model.fit(_matrix(pairs, names), labels)

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
            model,
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
            record_id = _record_id(record)
            scores[record_id] = 1.0 - float(np.mean(wins[record_id]))

    training = {
        "feature_count": len(names),
        "pair_records": len(pairs),
        "train_lineages": len(set(groups)),
        "grouped_cv_pairwise_balanced_accuracy": (
            float(np.mean(cv_scores)) if compute_cv else None
        ),
        "grouped_cv_pairwise_balanced_accuracy_std": (
            float(np.std(cv_scores)) if compute_cv else None
        ),
    }
    bundle = {
        "kind": "pairwise",
        "classifier": model,
        "feature_names": names,
        "categorical_features": categorical,
        "numeric_features": numeric,
    }
    return scores, training, bundle


def _embedding_vector(
    record: dict[str, Any],
    *,
    include_phase_b: bool,
) -> np.ndarray:
    names = _embedding_feature_names(
        record,
        include_phase_b=include_phase_b,
    )
    vector = np.asarray(
        [float(record["features"][name]) for name in names],
        dtype=np.float64,
    )
    norm = float(np.linalg.norm(vector))
    if not names or not math.isfinite(norm) or norm <= 0:
        raise ValueError("invalid retrieval embedding")
    return vector / norm


def _embedding_feature_names(
    record: dict[str, Any],
    *,
    include_phase_b: bool,
) -> list[str]:
    return sorted(
        name
        for name in record["features"]
        if name.startswith("transformer_source_")
        or (
            include_phase_b
            and name.startswith("transformer_phase_b_")
        )
    )


def _build_retrieval_index(
    records: list[dict[str, Any]],
    *,
    include_phase_b: bool,
) -> dict[str, Any]:
    """Package the training-only neighbors and outcomes needed at inference."""

    groups = _outcome_groups(records)
    train_problems = sorted(
        problem
        for problem, outcomes in groups.items()
        if outcomes[0]["split"] == "train"
    )
    if not train_problems:
        raise ValueError("retrieval index has no training lineages")

    feature_names = _embedding_feature_names(
        groups[train_problems[0]][0],
        include_phase_b=include_phase_b,
    )
    if not feature_names:
        raise ValueError("retrieval index has no embedding features")

    train_vectors = {}
    costs: dict[str, dict[str, float]] = {}
    setup_costs: dict[str, list[float]] = defaultdict(list)
    for problem in train_problems:
        problem_names = _embedding_feature_names(
            groups[problem][0],
            include_phase_b=include_phase_b,
        )
        if problem_names != feature_names:
            raise ValueError(
                f"inconsistent retrieval feature schema for {problem}"
            )
        train_vectors[problem] = _embedding_vector(
            groups[problem][0],
            include_phase_b=include_phase_b,
        ).astype(np.float32)
        costs[problem] = {}
        for record in groups[problem]:
            setup_id = str(record["setup"]["setup_id"])
            cost = (
                math.log(float(record["labels"]["regret"]))
                if record["labels"]["valid"]
                else RETRIEVAL_INVALID_LOG_COST
            )
            costs[problem][setup_id] = cost
            setup_costs[setup_id].append(cost)

    setup_ids = sorted(setup_costs)
    for problem in train_problems:
        missing = set(setup_ids) - set(costs[problem])
        if missing:
            raise ValueError(
                f"incomplete retrieval outcomes for {problem}: "
                f"{sorted(missing)}"
            )
    return {
        "include_phase_b": include_phase_b,
        "feature_names": feature_names,
        "train_lineages": train_problems,
        "train_vectors": train_vectors,
        "setup_costs": costs,
        "setup_priors": {
            setup_id: float(np.mean(values))
            for setup_id, values in setup_costs.items()
        },
        "setup_ids": setup_ids,
        "temperature": 0.1,
        "global_prior_weight": 0.5,
        "invalid_log_cost": RETRIEVAL_INVALID_LOG_COST,
    }


def _retrieval_scores(
    records: list[dict[str, Any]],
    *,
    neighbors: int,
    include_phase_b: bool,
) -> tuple[dict[str, float], dict[str, Any]]:
    groups = _outcome_groups(records)
    index = _build_retrieval_index(
        records,
        include_phase_b=include_phase_b,
    )
    train_problems = index["train_lineages"]
    vectors = {
        problem: _embedding_vector(
            groups[problem][0],
            include_phase_b=include_phase_b,
        )
        for problem in groups
    }
    costs = index["setup_costs"]
    priors = index["setup_priors"]

    scores = {}
    neighbor_audit = {}
    for problem, outcomes in groups.items():
        candidates = [
            train_problem
            for train_problem in train_problems
            if train_problem != problem
        ]
        similarities = sorted(
            (
                (
                    float(vectors[problem] @ vectors[train_problem]),
                    train_problem,
                )
                for train_problem in candidates
            ),
            key=lambda item: (-item[0], item[1]),
        )[:neighbors]
        if not similarities:
            raise ValueError(f"no retrieval neighbors for {problem}")
        raw = np.asarray(
            [item[0] for item in similarities],
            dtype=np.float64,
        )
        weights = np.exp(
            (raw - np.max(raw)) / float(index["temperature"])
        )
        weights = weights / np.sum(weights)
        neighbor_audit[problem] = [
            {
                "problem": neighbor,
                "cosine_similarity": similarity,
                "weight": float(weight),
            }
            for (similarity, neighbor), weight in zip(
                similarities,
                weights,
                strict=True,
            )
        ]
        for record in outcomes:
            setup_id = str(record["setup"]["setup_id"])
            observed = sum(
                float(weight) * costs[neighbor][setup_id]
                for (_, neighbor), weight in zip(
                    similarities,
                    weights,
                    strict=True,
                )
            )
            scores[_record_id(record)] = (
                observed
                + float(index["global_prior_weight"])
                * priors[setup_id]
            ) / (1.0 + float(index["global_prior_weight"]))
    audit = {
        "neighbors": neighbors,
        "include_phase_b": include_phase_b,
        "temperature": 0.1,
        "global_prior_weight": 0.5,
        "invalid_log_cost": RETRIEVAL_INVALID_LOG_COST,
        "neighbor_audit": neighbor_audit,
    }
    return scores, audit


def _validate_inference_bundle(bundle: dict[str, Any]) -> None:
    if bundle.get("schema_version") != INFERENCE_BUNDLE_SCHEMA_VERSION:
        raise ValueError("unexpected strengthened-router bundle schema")
    selected = str(bundle.get("selected_strengthened") or "")
    if not selected:
        raise ValueError("inference bundle has no selected policy")

    committee = bundle.get("committee_policy") or {}
    required_members = (
        list(committee.get("members") or [])
        if selected.startswith("committee_")
        else [selected]
    )
    models = bundle.get("models") or {}
    retrieval_indexes = bundle.get("retrieval_indexes") or {}
    missing = [
        member
        for member in required_members
        if (
            member not in retrieval_indexes
            if member.startswith("retrieval_")
            else member not in models
        )
    ]
    if missing:
        raise ValueError(
            f"inference bundle is missing policy members: {missing}"
        )
    if selected.startswith("committee_"):
        maximum_budget = int(
            committee.get("maximum_candidate_budget") or 0
        )
        if maximum_budget < 3 or maximum_budget > 8:
            raise ValueError(
                "committee candidate budget must be between three and eight"
            )
        if committee.get("mandatory_setup") != "multistep skillless":
            raise ValueError("committee mandatory baseline is missing")

    for name, index in retrieval_indexes.items():
        if not index.get("train_vectors"):
            raise ValueError(f"empty retrieval index: {name}")
        if not index.get("setup_costs") or not index.get("setup_priors"):
            raise ValueError(f"incomplete retrieval outcomes: {name}")
        dimensions = {
            len(vector) for vector in index["train_vectors"].values()
        }
        if dimensions != {len(index.get("feature_names") or [])}:
            raise ValueError(f"retrieval dimensions do not match: {name}")


def _build_inference_bundle(
    *,
    selected_strengthened: str,
    models: dict[str, Any],
    training: dict[str, Any],
    canonical_transformer: list[dict[str, Any]],
    embedding_manifest: dict[str, Any],
) -> dict[str, Any]:
    committee = training["committee_policies"]
    policy_members = (
        list(committee["members"])
        if selected_strengthened.startswith("committee_")
        else [selected_strengthened]
    )
    model_names = {
        "absolute_structured",
        "relative_structured_canonical",
        "pairwise_structured_canonical",
        *(
            member
            for member in policy_members
            if not member.startswith("retrieval_")
        ),
    }
    retrieval_names = sorted(
        member
        for member in policy_members
        if member.startswith("retrieval_")
    )
    retrieval_indexes = {}
    for name in retrieval_names:
        metadata = training[name]
        index = _build_retrieval_index(
            canonical_transformer,
            include_phase_b=bool(metadata["include_phase_b"]),
        )
        index["neighbors"] = int(metadata["neighbors"])
        retrieval_indexes[name] = index

    bundle = {
        "schema_version": INFERENCE_BUNDLE_SCHEMA_VERSION,
        "training_schema_version": SCHEMA_VERSION,
        "selected_strengthened": selected_strengthened,
        "models": {
            name: models[name]
            for name in sorted(model_names)
            if name in models
        },
        "retrieval_indexes": retrieval_indexes,
        "retrieval_training": {
            name: training[name] for name in retrieval_names
        },
        "consensus": training[
            "consensus_relative_pairwise_retrieval"
        ],
        "committee_policy": committee,
        "embedding_manifest": embedding_manifest,
        "canonical_phase_b_context": True,
    }
    _validate_inference_bundle(bundle)
    return bundle


def _rank_consensus(
    records: list[dict[str, Any]],
    score_maps: list[dict[str, float]],
) -> dict[str, float]:
    output = {}
    for outcomes in _outcome_groups(records).values():
        rank_sum = defaultdict(float)
        for score_map in score_maps:
            ordered = sorted(
                outcomes,
                key=lambda record: (
                    score_map[_record_id(record)],
                    record["setup"]["setup_fingerprint"],
                ),
            )
            denominator = max(len(ordered) - 1, 1)
            for rank, record in enumerate(ordered):
                rank_sum[_record_id(record)] += rank / denominator
        for record in outcomes:
            output[_record_id(record)] = (
                rank_sum[_record_id(record)] / len(score_maps)
            )
    return output


def _p95(values: list[float]) -> float | None:
    return float(np.percentile(values, 95)) if values else None


def _ranking_metrics_from_scores(
    records: list[dict[str, Any]],
    scores: dict[str, float],
    *,
    split: str,
    best_fixed_setup: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    groups = _outcome_groups(records, split=split)
    top1_accuracy = 0
    top1_valid = 0
    raw_top3_oracle = 0
    raw_top3_valid: list[bool] = []
    learned_oracle = 0
    learned_valid = 0
    learned_within5 = 0
    top1_regrets = []
    learned_regrets = []
    fixed_regrets = []
    prediction_rows = []

    for problem, outcomes in sorted(groups.items()):
        predicted = sorted(
            outcomes,
            key=lambda record: (
                scores[_record_id(record)],
                record["setup"]["setup_fingerprint"],
            ),
        )
        valid = [
            record for record in outcomes if record["labels"]["valid"]
        ]
        actual_best = min(
            valid,
            key=lambda record: (
                record["labels"]["latency_cycles"],
                record["setup"]["setup_fingerprint"],
            ),
        )
        top1 = predicted[0]
        raw_top3 = predicted[:3]
        top1_accuracy += int(
            top1["setup"]["setup_id"]
            == actual_best["setup"]["setup_id"]
        )
        top1_valid += int(top1["labels"]["valid"])
        top1_regrets.append(
            float(top1["labels"]["regret"])
            if top1["labels"]["valid"]
            else INVALID_REGRET_PENALTY
        )
        raw_top3_oracle += int(
            actual_best["setup"]["setup_id"]
            in {record["setup"]["setup_id"] for record in raw_top3}
        )
        raw_top3_valid.extend(
            bool(record["labels"]["valid"]) for record in raw_top3
        )

        mandatory = next(
            (
                record
                for record in outcomes
                if record["setup"]["setup_id"].endswith(
                    MANDATORY_BASELINE_SUFFIX
                )
            ),
            None,
        )
        learned = [mandatory] if mandatory is not None else []
        learned.extend(
            record
            for record in predicted
            if mandatory is None
            or record["setup"]["setup_id"]
            != mandatory["setup"]["setup_id"]
        )
        learned = list(
            {
                record["setup"]["setup_id"]: record
                for record in learned
            }.values()
        )[:3]
        learned_ids = {
            record["setup"]["setup_id"] for record in learned
        }
        learned_oracle += int(
            actual_best["setup"]["setup_id"] in learned_ids
        )
        valid_learned = [
            record for record in learned if record["labels"]["valid"]
        ]
        learned_winner = (
            min(
                valid_learned,
                key=lambda record: (
                    record["labels"]["latency_cycles"],
                    record["setup"]["setup_fingerprint"],
                ),
            )
            if valid_learned
            else None
        )
        if learned_winner is not None:
            learned_valid += 1
            regret = float(learned_winner["labels"]["regret"])
            learned_regrets.append(regret)
            learned_within5 += int(regret <= 1.05)
        else:
            learned_regrets.append(INVALID_REGRET_PENALTY)

        fixed = next(
            record
            for record in outcomes
            if record["setup"]["setup_id"] == best_fixed_setup
        )
        fixed_regrets.append(
            float(fixed["labels"]["regret"])
            if fixed["labels"]["valid"]
            else INVALID_REGRET_PENALTY
        )
        for rank, record in enumerate(predicted, start=1):
            prediction_rows.append(
                {
                    "split": split,
                    "problem": problem,
                    "setup_id": record["setup"]["setup_id"],
                    "predicted_rank": rank,
                    "score": scores[_record_id(record)],
                    "actual_valid": record["labels"]["valid"],
                    "actual_cycles": record["labels"]["latency_cycles"],
                    "actual_regret": record["labels"]["regret"],
                    "actual_best_setup_id": actual_best["setup"][
                        "setup_id"
                    ],
                    "raw_top3": rank <= 3,
                    "learned_top_k": record["setup"]["setup_id"]
                    in learned_ids,
                    "learned_top_k_winner": bool(
                        learned_winner is not None
                        and record["setup"]["setup_id"]
                        == learned_winner["setup"]["setup_id"]
                    ),
                }
            )

    count = len(groups)
    metrics = {
        "benchmark_count": count,
        "top_1_accuracy": top1_accuracy / count,
        "top_1_validity": top1_valid / count,
        "top_1_geomean_regret": geometric_mean(top1_regrets),
        "top_1_p95_regret": _p95(top1_regrets),
        "raw_top_3_oracle_coverage": raw_top3_oracle / count,
        "raw_top_3_actual_validity": (
            sum(raw_top3_valid) / len(raw_top3_valid)
        ),
        "learned_top_k": {
            "candidate_count": 3,
            "candidate_savings_vs_exhaustive": 0.7,
            "validity": learned_valid / count,
            "oracle_coverage": learned_oracle / count,
            "within_5pct_count": learned_within5,
            "within_5pct_coverage": learned_within5 / count,
            "geomean_regret": geometric_mean(learned_regrets),
            "p95_regret": _p95(learned_regrets),
        },
        "global_best_fixed": {
            "setup_id": best_fixed_setup,
            "geomean_regret": geometric_mean(fixed_regrets),
            "p95_regret": _p95(fixed_regrets),
        },
        "exhaustive": {
            "candidate_count": 10,
            "geomean_regret": 1.0,
            "validity": 1.0,
        },
    }
    return metrics, prediction_rows


def _diverse_voted_setups(
    *,
    alternatives_by_id: dict[str, dict[str, Any]],
    voted: list[str],
    count: int,
) -> list[str]:
    selected = list(voted[: min(2, count)])
    remaining = [
        setup_id for setup_id in voted if setup_id not in set(selected)
    ]
    while len(selected) < count and remaining:
        strategies = {
            alternatives_by_id[setup_id]["setup"]["strategy"]
            for setup_id in selected
        }
        scopes = {
            alternatives_by_id[setup_id]["setup"]["skill_scope"]
            for setup_id in selected
        }
        ranks = {setup_id: rank for rank, setup_id in enumerate(remaining)}
        chosen = min(
            remaining,
            key=lambda setup_id: (
                -(
                    alternatives_by_id[setup_id]["setup"]["skill_scope"]
                    not in scopes
                ),
                -(
                    alternatives_by_id[setup_id]["setup"]["strategy"]
                    not in strategies
                ),
                ranks[setup_id],
                alternatives_by_id[setup_id]["setup"][
                    "setup_fingerprint"
                ],
            ),
        )
        selected.append(chosen)
        remaining.remove(chosen)
    return selected


def _committee_candidate_budget(
    top_alternatives: list[str],
    *,
    adaptive: bool,
    maximum_budget: int,
) -> int:
    if not adaptive:
        return maximum_budget
    distinct = len(set(top_alternatives))
    if distinct <= 1:
        return 3
    if distinct == 2:
        return min(5, maximum_budget)
    return maximum_budget


def _committee_policy_metrics(
    records: list[dict[str, Any]],
    committee_scores: dict[str, dict[str, float]],
    *,
    split: str,
    best_fixed_setup: str,
    adaptive: bool,
    maximum_budget: int = 5,
    diverse: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    groups = _outcome_groups(records, split=split)
    top1_accuracy = 0
    top1_valid = 0
    raw_top3_oracle = 0
    raw_top3_valid: list[bool] = []
    learned_oracle = 0
    learned_valid = 0
    learned_within5 = 0
    candidate_counts = []
    top1_regrets = []
    learned_regrets = []
    fixed_regrets = []
    prediction_rows = []

    for problem, outcomes in sorted(groups.items()):
        mandatory = next(
            record
            for record in outcomes
            if record["setup"]["setup_id"].endswith(
                MANDATORY_BASELINE_SUFFIX
            )
        )
        mandatory_id = mandatory["setup"]["setup_id"]
        rankings = {}
        votes: dict[str, float] = defaultdict(float)
        top_alternatives = []
        for model_name, score_map in committee_scores.items():
            ordered = sorted(
                outcomes,
                key=lambda record: (
                    score_map[_record_id(record)],
                    record["setup"]["setup_fingerprint"],
                ),
            )
            rankings[model_name] = ordered
            alternatives = [
                record
                for record in ordered
                if record["setup"]["setup_id"] != mandatory_id
            ]
            top_alternatives.append(alternatives[0]["setup"]["setup_id"])
            for rank, record in enumerate(alternatives):
                votes[record["setup"]["setup_id"]] += 1.0 / (rank + 1.0)

        budget = _committee_candidate_budget(
            top_alternatives,
            adaptive=adaptive,
            maximum_budget=maximum_budget,
        )
        alternatives_by_id = {
            record["setup"]["setup_id"]: record
            for record in outcomes
            if record["setup"]["setup_id"] != mandatory_id
        }
        voted = sorted(
            alternatives_by_id,
            key=lambda setup_id: (
                -votes[setup_id],
                alternatives_by_id[setup_id]["setup"][
                    "setup_fingerprint"
                ],
            ),
        )
        selected_alternatives = (
            _diverse_voted_setups(
                alternatives_by_id=alternatives_by_id,
                voted=voted,
                count=budget - 1,
            )
            if diverse
            else voted[: budget - 1]
        )
        learned = [mandatory] + [
            alternatives_by_id[setup_id]
            for setup_id in selected_alternatives
        ]
        learned_ids = {
            record["setup"]["setup_id"] for record in learned
        }
        candidate_counts.append(len(learned))

        valid = [
            record for record in outcomes if record["labels"]["valid"]
        ]
        actual_best = min(
            valid,
            key=lambda record: (
                record["labels"]["latency_cycles"],
                record["setup"]["setup_fingerprint"],
            ),
        )
        consensus_order = [mandatory] + [
            alternatives_by_id[setup_id] for setup_id in voted
        ]
        top1 = consensus_order[0]
        raw_top3 = consensus_order[:3]
        top1_accuracy += int(
            top1["setup"]["setup_id"]
            == actual_best["setup"]["setup_id"]
        )
        top1_valid += int(top1["labels"]["valid"])
        top1_regrets.append(
            float(top1["labels"]["regret"])
            if top1["labels"]["valid"]
            else INVALID_REGRET_PENALTY
        )
        raw_top3_oracle += int(
            actual_best["setup"]["setup_id"]
            in {record["setup"]["setup_id"] for record in raw_top3}
        )
        raw_top3_valid.extend(
            bool(record["labels"]["valid"]) for record in raw_top3
        )
        learned_oracle += int(
            actual_best["setup"]["setup_id"] in learned_ids
        )
        valid_learned = [
            record for record in learned if record["labels"]["valid"]
        ]
        learned_winner = (
            min(
                valid_learned,
                key=lambda record: (
                    record["labels"]["latency_cycles"],
                    record["setup"]["setup_fingerprint"],
                ),
            )
            if valid_learned
            else None
        )
        if learned_winner is not None:
            learned_valid += 1
            regret = float(learned_winner["labels"]["regret"])
            learned_regrets.append(regret)
            learned_within5 += int(regret <= 1.05)
        else:
            learned_regrets.append(INVALID_REGRET_PENALTY)
        fixed = next(
            record
            for record in outcomes
            if record["setup"]["setup_id"] == best_fixed_setup
        )
        fixed_regrets.append(
            float(fixed["labels"]["regret"])
            if fixed["labels"]["valid"]
            else INVALID_REGRET_PENALTY
        )
        consensus_rank = {
            record["setup"]["setup_id"]: rank
            for rank, record in enumerate(consensus_order, start=1)
        }
        for record in outcomes:
            setup_id = record["setup"]["setup_id"]
            prediction_rows.append(
                {
                    "split": split,
                    "problem": problem,
                    "setup_id": setup_id,
                    "predicted_rank": consensus_rank[setup_id],
                    "committee_vote": votes.get(setup_id, 0.0),
                    "actual_valid": record["labels"]["valid"],
                    "actual_cycles": record["labels"]["latency_cycles"],
                    "actual_regret": record["labels"]["regret"],
                    "actual_best_setup_id": actual_best["setup"][
                        "setup_id"
                    ],
                    "learned_top_k": setup_id in learned_ids,
                    "learned_top_k_winner": bool(
                        learned_winner is not None
                        and setup_id
                        == learned_winner["setup"]["setup_id"]
                    ),
                    "adaptive_budget": budget,
                    "committee_top_alternatives": top_alternatives,
                }
            )

    count = len(groups)
    mean_candidates = float(np.mean(candidate_counts))
    metrics = {
        "benchmark_count": count,
        "top_1_accuracy": top1_accuracy / count,
        "top_1_validity": top1_valid / count,
        "top_1_geomean_regret": geometric_mean(top1_regrets),
        "top_1_p95_regret": _p95(top1_regrets),
        "raw_top_3_oracle_coverage": raw_top3_oracle / count,
        "raw_top_3_actual_validity": (
            sum(raw_top3_valid) / len(raw_top3_valid)
        ),
        "learned_top_k": {
            "candidate_count": mean_candidates,
            "candidate_count_min": min(candidate_counts),
            "candidate_count_max": max(candidate_counts),
            "candidate_savings_vs_exhaustive": 1.0
            - mean_candidates / 10.0,
            "validity": learned_valid / count,
            "oracle_coverage": learned_oracle / count,
            "within_5pct_count": learned_within5,
            "within_5pct_coverage": learned_within5 / count,
            "geomean_regret": geometric_mean(learned_regrets),
            "p95_regret": _p95(learned_regrets),
        },
        "global_best_fixed": {
            "setup_id": best_fixed_setup,
            "geomean_regret": geometric_mean(fixed_regrets),
            "p95_regret": _p95(fixed_regrets),
        },
        "exhaustive": {
            "candidate_count": 10,
            "geomean_regret": 1.0,
            "validity": 1.0,
        },
        "committee": {
            "members": sorted(committee_scores),
            "adaptive": adaptive,
            "maximum_candidate_budget": maximum_budget,
            "mode_diversity": diverse,
            "budget_rule": (
                "3 if unanimous, 5 if two top choices, otherwise maximum"
                if adaptive
                else "fixed maximum"
            ),
        },
    }
    return metrics, prediction_rows


def _selection_key(metrics: dict[str, Any], name: str) -> tuple:
    learned = metrics["learned_top_k"]
    return (
        -float(learned["validity"]),
        -float(learned["within_5pct_coverage"]),
        -float(learned["oracle_coverage"]),
        -float(metrics["raw_top_3_oracle_coverage"]),
        float(learned["geomean_regret"]),
        -float(learned["candidate_savings_vs_exhaustive"]),
        -float(metrics["top_1_accuracy"]),
        name,
    )


def _outer_fold_records(
    records: list[dict[str, Any]],
    held_out_problems: set[str],
) -> list[dict[str, Any]]:
    return [
        {
            **record,
            "split": (
                "validation"
                if record["problem"] in held_out_problems
                else "train"
            ),
        }
        for record in records
        if record["split"] == "train"
    ]


def _aggregate_outer_metrics(
    fold_metrics: list[dict[str, Any]],
) -> dict[str, Any]:
    total = sum(item["benchmark_count"] for item in fold_metrics)
    learned = [item["learned_top_k"] for item in fold_metrics]
    weighted_log_regret = sum(
        item["benchmark_count"]
        * math.log(item["learned_top_k"]["geomean_regret"])
        for item in fold_metrics
    )
    mean_candidates = sum(
        item["benchmark_count"]
        * item["learned_top_k"]["candidate_count"]
        for item in fold_metrics
    ) / total
    return {
        "benchmark_count": total,
        "folds": len(fold_metrics),
        "validity": sum(
            item["benchmark_count"] * item["learned_top_k"]["validity"]
            for item in fold_metrics
        )
        / total,
        "oracle_coverage": sum(
            item["benchmark_count"]
            * item["learned_top_k"]["oracle_coverage"]
            for item in fold_metrics
        )
        / total,
        "within_5pct_count": int(
            round(
                sum(
                    item["benchmark_count"]
                    * item["learned_top_k"]["within_5pct_coverage"]
                    for item in fold_metrics
                )
            )
        ),
        "within_5pct_coverage": sum(
            item["benchmark_count"]
            * item["learned_top_k"]["within_5pct_coverage"]
            for item in fold_metrics
        )
        / total,
        "geomean_regret": math.exp(weighted_log_regret / total),
        "candidate_count": mean_candidates,
        "candidate_savings_vs_exhaustive": 1.0
        - mean_candidates / 10.0,
        "fold_metrics": fold_metrics,
    }


def _outer_grouped_policy_evaluation(
    *,
    canonical_structured: list[dict[str, Any]],
    canonical_transformer: list[dict[str, Any]],
    canonical_hybrid: list[dict[str, Any]],
    trees: int,
    jobs: int,
) -> dict[str, Any]:
    train_problems = sorted(
        {
            str(record["problem"])
            for record in canonical_structured
            if record["split"] == "train"
        }
    )
    splitter = GroupKFold(n_splits=5)
    fold_results: dict[str, list[dict[str, Any]]] = defaultdict(list)
    fold_manifest = []
    for fold_index, (train_indices, held_out_indices) in enumerate(
        splitter.split(
            np.zeros(len(train_problems)),
            groups=np.asarray(train_problems),
        )
    ):
        del train_indices
        held_out = {
            train_problems[index] for index in held_out_indices
        }
        fold_structured = _outer_fold_records(
            canonical_structured,
            held_out,
        )
        fold_transformer = _outer_fold_records(
            canonical_transformer,
            held_out,
        )
        fold_hybrid = _outer_fold_records(canonical_hybrid, held_out)
        train_ranking = [
            record
            for record in _ranking_outcomes(
                fold_structured,
                split="train",
            )
            if record["labels"]["valid"]
        ]
        best_fixed, _ = _best_fixed_setup(train_ranking)
        log_regret_scores, _, _ = _fit_direct(
            fold_hybrid,
            target="log_regret",
            trees=trees,
            jobs=jobs,
            compute_cv=False,
        )
        within5_scores, _, _ = _fit_within5(
            fold_hybrid,
            trees=trees,
            jobs=jobs,
            compute_cv=False,
        )
        pairwise_scores, _, _ = _fit_pairwise(
            fold_hybrid,
            trees=trees,
            jobs=jobs,
            compute_cv=False,
        )
        retrieval_scores, _ = _retrieval_scores(
            fold_transformer,
            neighbors=1,
            include_phase_b=False,
        )
        committee_scores = {
            "log_regret_hybrid32_canonical": log_regret_scores,
            "within5_hybrid32_canonical": within5_scores,
            "pairwise_hybrid32_canonical": pairwise_scores,
            "retrieval_source_k1": retrieval_scores,
        }
        retrieval_metrics, _ = _ranking_metrics_from_scores(
            fold_structured,
            retrieval_scores,
            split="validation",
            best_fixed_setup=best_fixed,
        )
        committee_metrics_v1, _ = _committee_policy_metrics(
            fold_structured,
            committee_scores,
            split="validation",
            best_fixed_setup=best_fixed,
            adaptive=True,
            maximum_budget=5,
            diverse=False,
        )
        committee_metrics_v2, _ = _committee_policy_metrics(
            fold_structured,
            committee_scores,
            split="validation",
            best_fixed_setup=best_fixed,
            adaptive=True,
            maximum_budget=8,
            diverse=True,
        )
        fold_results["retrieval_source_k1"].append(retrieval_metrics)
        fold_results["committee_disagreement_adaptive"].append(
            committee_metrics_v1
        )
        fold_results["committee_regret_within5_adaptive_v2"].append(
            committee_metrics_v2
        )
        fold_manifest.append(
            {
                "fold": fold_index,
                "held_out_problems": sorted(held_out),
                "best_fixed_setup": best_fixed,
            }
        )
    return {
        "schema": "five-fold outer benchmark-grouped confirmation",
        "trees_per_outer_model": trees,
        "fold_manifest": fold_manifest,
        "policies": {
            name: _aggregate_outer_metrics(metrics)
            for name, metrics in fold_results.items()
        },
    }


def _outer_selection_key(metrics: dict[str, Any], name: str) -> tuple:
    return (
        -float(metrics["validity"]),
        -float(metrics["within_5pct_coverage"]),
        -float(metrics["oracle_coverage"]),
        float(metrics["geomean_regret"]),
        -float(metrics["candidate_savings_vs_exhaustive"]),
        name,
    )


def _plot(rows: list[dict[str, Any]], destination: Path) -> None:
    names = [row["variant"] for row in rows]
    within = [row["validation_within_5pct"] for row in rows]
    oracle = [row["validation_learned_oracle"] for row in rows]
    regret = [row["validation_geomean_regret"] for row in rows]
    positions = np.arange(len(rows))
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    width = 0.38
    axes[0].bar(
        positions - width / 2,
        within,
        width,
        label="within 5%",
        color="#2d8a56",
    )
    axes[0].bar(
        positions + width / 2,
        oracle,
        width,
        label="oracle in learned top-k",
        color="#8b5a2b",
    )
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Validation coverage")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(positions, regret, color="#2368a2")
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[1].set_ylabel("Validation geomean regret")
    axes[1].set_xticks(positions, names, rotation=35, ha="right")
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("Strengthened setup-router objectives")
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def train(args: argparse.Namespace) -> dict[str, Any]:
    base_records = _load(args.corpus)
    embeddings, record_map, embedding_manifest = _load_embeddings(
        args.embedding_dir
    )
    transformer64 = _variant_records(
        base_records,
        Variant("transformer64", 64, False, True, True),
        embeddings,
        record_map,
    )
    hybrid32 = _variant_records(
        base_records,
        Variant("hybrid32", 32, True, True, True),
        embeddings,
        record_map,
    )
    canonical_structured = _canonicalize_phase_b(base_records)
    canonical_transformer = _canonicalize_phase_b(transformer64)
    canonical_hybrid = _canonicalize_phase_b(hybrid32)
    phase_b_log_by_id = {
        _record_id(record): float(
            record["features"]["phase_b_log_latency_cycles"]
        )
        for record in canonical_structured
    }
    canonical_transformer = [
        {
            **record,
            "_phase_b_log_latency_cycles_for_target": (
                phase_b_log_by_id[_record_id(record)]
            ),
        }
        for record in canonical_transformer
    ]
    canonical_hybrid = [
        {
            **record,
            "_phase_b_log_latency_cycles_for_target": (
                phase_b_log_by_id[_record_id(record)]
            ),
        }
        for record in canonical_hybrid
    ]

    train_ranking = [
        record
        for record in _ranking_outcomes(base_records, split="train")
        if record["labels"]["valid"]
    ]
    best_fixed, fixed_scores = _best_fixed_setup(train_ranking)

    scores: dict[str, dict[str, float]] = {}
    training: dict[str, Any] = {}
    models: dict[str, Any] = {}
    for name, records, target in (
        ("absolute_structured", base_records, "absolute_log_cycles"),
        (
            "relative_structured_canonical",
            canonical_structured,
            "relative_log_cycles",
        ),
        (
            "relative_transformer64_canonical",
            canonical_transformer,
            "relative_log_cycles",
        ),
        (
            "relative_hybrid32_canonical",
            canonical_hybrid,
            "relative_log_cycles",
        ),
        (
            "log_regret_hybrid32_canonical",
            canonical_hybrid,
            "log_regret",
        ),
    ):
        scores[name], training[name], models[name] = _fit_direct(
            records,
            target=target,
            trees=args.trees,
            jobs=args.jobs,
        )

    scores["within5_hybrid32_canonical"], training[
        "within5_hybrid32_canonical"
    ], models["within5_hybrid32_canonical"] = _fit_within5(
        canonical_hybrid,
        trees=args.trees,
        jobs=args.jobs,
    )

    for name, records in (
        ("pairwise_structured_canonical", canonical_structured),
        ("pairwise_transformer64_canonical", canonical_transformer),
        ("pairwise_hybrid32_canonical", canonical_hybrid),
    ):
        scores[name], training[name], models[name] = _fit_pairwise(
            records,
            trees=args.trees,
            jobs=args.jobs,
        )

    retrieval_names = []
    for include_phase_b, suffix in (
        (False, "source"),
        (True, "source_phase_b"),
    ):
        for neighbors in (1, 3, 5):
            name = f"retrieval_{suffix}_k{neighbors}"
            scores[name], training[name] = _retrieval_scores(
                canonical_transformer,
                neighbors=neighbors,
                include_phase_b=include_phase_b,
            )
            retrieval_names.append(name)

    scores["consensus_relative_pairwise_retrieval"] = _rank_consensus(
        canonical_structured,
        [
            scores["relative_structured_canonical"],
            scores["pairwise_structured_canonical"],
            scores["retrieval_source_phase_b_k3"],
        ],
    )
    training["consensus_relative_pairwise_retrieval"] = {
        "components": [
            "relative_structured_canonical",
            "pairwise_structured_canonical",
            "retrieval_source_phase_b_k3",
        ],
        "aggregation": "equal_weight_normalized_rank",
    }
    committee_scores = {
        name: scores[name]
        for name in (
            "log_regret_hybrid32_canonical",
            "within5_hybrid32_canonical",
            "pairwise_hybrid32_canonical",
            "retrieval_source_k1",
        )
    }
    committee_policies = {
        "committee_top5": {
            "adaptive": False,
            "maximum_budget": 5,
            "diverse": False,
        },
        "committee_disagreement_adaptive": {
            "adaptive": True,
            "maximum_budget": 5,
            "diverse": False,
        },
        "committee_regret_within5_adaptive_v2": {
            "adaptive": True,
            "maximum_budget": 8,
            "diverse": True,
        },
    }
    training["committee_policies"] = {
        "members": sorted(committee_scores),
        "vote": "reciprocal_rank",
        "mandatory_setup": "multistep skillless",
        "maximum_candidate_budget": 8,
        "policies": committee_policies,
        "uncertainty_budget_rule": (
            "3 candidates when committee top choice is unanimous, 5 when "
            "two alternatives are proposed, otherwise 8"
        ),
        "mode_diversity": (
            "retain the two highest reciprocal-rank alternatives, then "
            "prefer unseen skill scopes and strategies"
        ),
    }

    validation_metrics = {}
    validation_predictions = []
    table_rows = []
    evaluation_records = canonical_structured
    for name, score_map in scores.items():
        metrics, predictions = _ranking_metrics_from_scores(
            evaluation_records,
            score_map,
            split="validation",
            best_fixed_setup=best_fixed,
        )
        validation_metrics[name] = metrics
        validation_predictions.extend(
            {"variant": name, **row} for row in predictions
        )
        table_rows.append(
            {
                "variant": name,
                "validation_top1_accuracy": metrics["top_1_accuracy"],
                "validation_top1_validity": metrics["top_1_validity"],
                "validation_raw_top3_oracle": metrics[
                    "raw_top_3_oracle_coverage"
                ],
                "validation_learned_oracle": metrics["learned_top_k"][
                    "oracle_coverage"
                ],
                "validation_within_5pct": metrics["learned_top_k"][
                    "within_5pct_coverage"
                ],
                "validation_geomean_regret": metrics["learned_top_k"][
                    "geomean_regret"
                ],
            }
        )
    for name, policy_spec in committee_policies.items():
        metrics, predictions = _committee_policy_metrics(
            evaluation_records,
            committee_scores,
            split="validation",
            best_fixed_setup=best_fixed,
            **policy_spec,
        )
        validation_metrics[name] = metrics
        validation_predictions.extend(
            {"variant": name, **row} for row in predictions
        )
        table_rows.append(
            {
                "variant": name,
                "validation_top1_accuracy": metrics["top_1_accuracy"],
                "validation_top1_validity": metrics["top_1_validity"],
                "validation_raw_top3_oracle": metrics[
                    "raw_top_3_oracle_coverage"
                ],
                "validation_learned_oracle": metrics["learned_top_k"][
                    "oracle_coverage"
                ],
                "validation_within_5pct": metrics["learned_top_k"][
                    "within_5pct_coverage"
                ],
                "validation_geomean_regret": metrics["learned_top_k"][
                    "geomean_regret"
                ],
            }
        )

    outer_grouped_evaluation = _outer_grouped_policy_evaluation(
        canonical_structured=canonical_structured,
        canonical_transformer=canonical_transformer,
        canonical_hybrid=canonical_hybrid,
        trees=args.outer_cv_trees,
        jobs=args.jobs,
    )
    outer_policy_metrics = outer_grouped_evaluation["policies"]
    selected_strengthened = min(
        outer_policy_metrics,
        key=lambda name: _outer_selection_key(
            outer_policy_metrics[name],
            name,
        ),
    )
    selected_overall = min(
        ("absolute_structured", selected_strengthened),
        key=lambda name: _selection_key(validation_metrics[name], name),
    )
    test_metrics = {}
    test_predictions = []
    for name in ("absolute_structured", selected_strengthened):
        if name in committee_policies:
            metrics, predictions = _committee_policy_metrics(
                evaluation_records,
                committee_scores,
                split="test",
                best_fixed_setup=best_fixed,
                **committee_policies[name],
            )
        else:
            metrics, predictions = _ranking_metrics_from_scores(
                evaluation_records,
                scores[name],
                split="test",
                best_fixed_setup=best_fixed,
            )
        test_metrics[name] = metrics
        test_predictions.extend(
            {"variant": name, **row} for row in predictions
        )

    baseline_test = test_metrics["absolute_structured"]
    strengthened_test = test_metrics[selected_strengthened]
    replace_baseline = (
        strengthened_test["learned_top_k"]["validity"]
        >= baseline_test["learned_top_k"]["validity"]
        and strengthened_test["learned_top_k"]["within_5pct_count"]
        > baseline_test["learned_top_k"]["within_5pct_count"]
        and strengthened_test["learned_top_k"]["geomean_regret"]
        <= baseline_test["learned_top_k"]["geomean_regret"]
    )
    corrected_training_present = any(
        record["features"].get("setup_behavior_version") == "corrected_v2"
        and record["split"] == "train"
        for record in base_records
    )
    selected_outer = outer_policy_metrics[selected_strengthened]
    threshold_checks = {
        "candidate_budget_at_most_8": (
            strengthened_test["learned_top_k"]["candidate_count_max"]
            if "candidate_count_max"
            in strengthened_test["learned_top_k"]
            else strengthened_test["learned_top_k"]["candidate_count"]
        )
        <= 8,
        "held_out_candidate_validity_100pct": (
            strengthened_test["learned_top_k"]["validity"] == 1.0
        ),
        "three_of_four_within_5pct": (
            strengthened_test["learned_top_k"]["within_5pct_count"] >= 3
        ),
        "geomean_regret_at_most_1_15": (
            strengthened_test["learned_top_k"]["geomean_regret"] <= 1.15
        ),
        "no_worse_than_best_fixed": (
            strengthened_test["learned_top_k"]["geomean_regret"]
            <= strengthened_test["global_best_fixed"]["geomean_regret"]
        ),
        "strict_gain_over_absolute_router": replace_baseline,
        "outer_grouped_within_5pct_at_least_75pct": (
            selected_outer["within_5pct_coverage"] >= 0.75
        ),
        "outer_grouped_geomean_regret_at_most_1_15": (
            selected_outer["geomean_regret"] <= 1.15
        ),
        "corrected_v2_training_coverage": corrected_training_present,
        "fresh_unexposed_confirmation_set": False,
    }
    numerical_thresholds_pass = all(
        threshold_checks[name]
        for name in (
            "candidate_budget_at_most_8",
            "held_out_candidate_validity_100pct",
            "three_of_four_within_5pct",
            "geomean_regret_at_most_1_15",
            "no_worse_than_best_fixed",
            "strict_gain_over_absolute_router",
            "outer_grouped_within_5pct_at_least_75pct",
            "outer_grouped_geomean_regret_at_most_1_15",
        )
    )
    output = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "methodology": {
            "selection_split": "validation",
            "test_variants": [
                "absolute_structured",
                selected_strengthened,
            ],
            "test_not_used_for_selection": True,
            "test_reuse_caveat": (
                "the four historical test kernels were exposed by earlier "
                "router studies; treat this as confirmation, not a pristine "
                "new final test"
            ),
            "canonical_phase_b_context": True,
            "reference_metrics_as_inputs": False,
            "post_candidate_features_as_inputs": False,
            "mandatory_candidate": "multistep skillless",
            "finalist_selection": (
                "five-fold outer benchmark-grouped evaluation over the "
                "19 development lineages"
            ),
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
        "training": training,
        "validation": validation_metrics,
        "outer_grouped_evaluation": outer_grouped_evaluation,
        "selection": {
            "selected_strengthened": selected_strengthened,
            "selected_overall": selected_overall,
        },
        "held_out_test": test_metrics,
        "comparison": {
            "within_5pct_count_delta": (
                strengthened_test["learned_top_k"]["within_5pct_count"]
                - baseline_test["learned_top_k"]["within_5pct_count"]
            ),
            "learned_oracle_delta": (
                strengthened_test["learned_top_k"]["oracle_coverage"]
                - baseline_test["learned_top_k"]["oracle_coverage"]
            ),
            "geomean_regret_delta": (
                strengthened_test["learned_top_k"]["geomean_regret"]
                - baseline_test["learned_top_k"]["geomean_regret"]
            ),
            "geomean_regret_ratio": (
                strengthened_test["learned_top_k"]["geomean_regret"]
                / baseline_test["learned_top_k"]["geomean_regret"]
            ),
        },
        "recommendation": {
            "replace_absolute_structured_router": replace_baseline,
            "numerical_thresholds_pass": numerical_thresholds_pass,
            "promote_to_corrected_v2_confirmation": (
                numerical_thresholds_pass
            ),
            "deployment_status": (
                "active"
                if all(threshold_checks.values())
                else "advisory"
            ),
            "threshold_checks": threshold_checks,
            "reason": (
                "numerical routing thresholds pass, but activation still "
                "requires corrected-v2 training coverage and a fresh "
                "unexposed confirmation set"
                if numerical_thresholds_pass
                else "use the strengthened router as the advisory challenger, "
                "but do not activate it while one or more numerical routing "
                "thresholds remain unmet"
            ),
        },
        "best_fixed_training_geomean_regret": fixed_scores,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "strengthened_router_models.joblib"
    inference_bundle = _build_inference_bundle(
        selected_strengthened=selected_strengthened,
        models=models,
        training=training,
        canonical_transformer=canonical_transformer,
        embedding_manifest=embedding_manifest,
    )
    joblib.dump(inference_bundle, model_path)
    _validate_inference_bundle(joblib.load(model_path))
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
        for row in validation_predictions + test_predictions:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    figure_path = args.output_dir / "validation_ablation.png"
    _plot(table_rows, figure_path)

    baseline_learned = baseline_test["learned_top_k"]
    strengthened_learned = strengthened_test["learned_top_k"]
    report_path = args.output_dir / "report.md"
    report_path.write_text(
        "\n".join(
            [
                "# Strengthened Setup Router",
                "",
                f"- Validation-selected strengthened router: `{selected_strengthened}`",
                f"- Overall validation winner: `{selected_overall}`",
                "- Every setup was scored from the same canonical Phase-B context.",
                "- Historical test reuse is confirmation only; it is not a new pristine final test.",
                "",
                "## Outer Grouped Evaluation",
                "",
                "| policy | lineages | candidates | within 5% | geomean regret |",
                "|---|---:|---:|---:|---:|",
                (
                    "| retrieval source k1 | "
                    f"{outer_policy_metrics['retrieval_source_k1']['benchmark_count']} | "
                    f"{outer_policy_metrics['retrieval_source_k1']['candidate_count']:.2f} | "
                    f"{outer_policy_metrics['retrieval_source_k1']['within_5pct_coverage']:.3f} | "
                    f"{outer_policy_metrics['retrieval_source_k1']['geomean_regret']:.3f} |"
                ),
                (
                    "| adaptive committee v1 | "
                    f"{outer_policy_metrics['committee_disagreement_adaptive']['benchmark_count']} | "
                    f"{outer_policy_metrics['committee_disagreement_adaptive']['candidate_count']:.2f} | "
                    f"{outer_policy_metrics['committee_disagreement_adaptive']['within_5pct_coverage']:.3f} | "
                    f"{outer_policy_metrics['committee_disagreement_adaptive']['geomean_regret']:.3f} |"
                ),
                (
                    "| regret/within-5 adaptive v2 | "
                    f"{outer_policy_metrics['committee_regret_within5_adaptive_v2']['benchmark_count']} | "
                    f"{outer_policy_metrics['committee_regret_within5_adaptive_v2']['candidate_count']:.2f} | "
                    f"{outer_policy_metrics['committee_regret_within5_adaptive_v2']['within_5pct_coverage']:.3f} | "
                    f"{outer_policy_metrics['committee_regret_within5_adaptive_v2']['geomean_regret']:.3f} |"
                ),
                "",
                "## Held-Out Comparison",
                "",
                "| model | candidates | savings | learned oracle | within 5% | geomean regret |",
                "|---|---:|---:|---:|---:|---:|",
                (
                    "| absolute structured | "
                    f"{baseline_learned['candidate_count']:.1f} | "
                    f"{baseline_learned['candidate_savings_vs_exhaustive']:.3f} | "
                    f"{baseline_learned['oracle_coverage']:.3f} | "
                    f"{baseline_learned['within_5pct_coverage']:.3f} | "
                    f"{baseline_learned['geomean_regret']:.3f} |"
                ),
                (
                    f"| {selected_strengthened} | "
                    f"{strengthened_learned['candidate_count']:.1f} | "
                    f"{strengthened_learned['candidate_savings_vs_exhaustive']:.3f} | "
                    f"{strengthened_learned['oracle_coverage']:.3f} | "
                    f"{strengthened_learned['within_5pct_coverage']:.3f} | "
                    f"{strengthened_learned['geomean_regret']:.3f} |"
                ),
                "",
                "## Decision",
                "",
                (
                    "Promote the adaptive committee to corrected-v2 confirmation, but do not activate it yet."
                    if numerical_thresholds_pass
                    else "Use the strengthened router as the advisory challenger; do not activate it yet."
                ),
                (
                    " Activation still requires corrected-v2 coverage and "
                    "a fresh unexposed confirmation set."
                    if numerical_thresholds_pass
                    else " One or more numerical routing thresholds remain "
                    "unmet; see the machine-readable threshold checks."
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    artifact_paths = [
        model_path,
        metrics_path,
        table_path,
        predictions_path,
        figure_path,
        report_path,
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
    parser.add_argument("--outer-cv-trees", type=int, default=100)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    if args.trees < 50:
        parser.error("--trees must be at least 50")
    if args.outer_cv_trees < 50:
        parser.error("--outer-cv-trees must be at least 50")
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    return args


if __name__ == "__main__":
    result = train(parse_args())
    print(
        json.dumps(
            {
                "selection": result["selection"],
                "held_out_test": result["held_out_test"],
                "recommendation": result["recommendation"],
            },
            sort_keys=True,
        )
    )
