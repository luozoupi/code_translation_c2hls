#!/usr/bin/env python3
"""Compare setup-router models on held-out external benchmark lineages."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_setup_router import (
    _classifier_metrics,
    _ranking_metrics,
)


SCHEMA_VERSION = "c2hls.external-setup-router-evaluation.v1"


def _load_records(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _evaluate_model(
    model_path: Path,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    bundle = joblib.load(model_path)
    names = list(bundle["feature_names"])
    output: dict[str, Any] = {
        "model_path": str(model_path.resolve()),
        "schema_version": bundle.get("schema_version"),
        "splits": {},
    }
    for split in ("validation", "test"):
        split_records = [
            record
            for record in records
            if record["split"] == split
        ]
        feasibility_records = [
            record
            for record in split_records
            if record["eligibility"]["feasibility_model"]
        ]
        ranking, predictions = _ranking_metrics(
            bundle["classifier"],
            bundle["regressor"],
            split_records,
            names,
            best_fixed_setup=bundle["best_fixed_setup_id"],
        )
        output["splits"][split] = {
            "lineages": sorted(
                {
                    record["benchmark_lineage"]
                    for record in split_records
                }
            ),
            "feasibility": _classifier_metrics(
                bundle["classifier"],
                feasibility_records,
                names,
            ),
            "ranking": ranking,
            "predictions": predictions,
        }
    return output


def run(args: argparse.Namespace) -> dict[str, Any]:
    all_records = _load_records(args.corpus)
    records = [
        record
        for record in all_records
        if str(record.get("benchmark_lineage") or "").startswith(
            args.lineage_prefix
        )
        and record.get("record_kind") == "setup_outcome"
        and (record.get("features") or {}).get(
            "setup_behavior_version"
        )
        == "corrected_v2"
    ]
    if not records:
        raise ValueError(
            f"no records matched external lineage prefix "
            f"{args.lineage_prefix!r}"
        )
    models = {
        name: _evaluate_model(path, records)
        for name, path in (
            ("base", args.base_model),
            ("augmented", args.augmented_model),
        )
    }
    output = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "corpus": str(args.corpus.resolve()),
        "lineage_prefix": args.lineage_prefix,
        "external_lineages": sorted(
            {record["benchmark_lineage"] for record in records}
        ),
        "models": models,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "external_router_comparison.json"
    json_path.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    rows = []
    for model_name, model in models.items():
        for split, result in model["splits"].items():
            ranking = result["ranking"]
            learned = ranking["learned_top_k"]
            rows.append(
                (
                    model_name,
                    split,
                    len(result["lineages"]),
                    ranking["top_1_accuracy"],
                    ranking["top_3_oracle_coverage"],
                    learned["within_5pct_coverage"],
                    learned["geomean_regret"],
                    learned["candidate_count"],
                )
            )
    markdown = [
        "# External Setup Router Comparison",
        "",
        (
            "Only corrected-v2 HLS-Eval/MachSuite lineages are included. "
            "The test lineages are unavailable to both training runs."
        ),
        "",
        "| model | split | lineages | top-1 | top-3 oracle | within 5% | geomean regret | candidates |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        markdown.append(
            "| {} | {} | {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.1f} |".format(
                *row
            )
        )
    (args.output_dir / "external_router_comparison.md").write_text(
        "\n".join(markdown) + "\n",
        encoding="utf-8",
    )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--augmented-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lineage-prefix", default="machsuite:")
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(
        json.dumps(
            {
                "schema_version": result["schema_version"],
                "external_lineages": len(result["external_lineages"]),
            },
            sort_keys=True,
        )
    )
