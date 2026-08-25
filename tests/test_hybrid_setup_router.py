from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts.export_setup_router_transformer_embeddings import (
    _last_token_pool,
)
from scripts.train_hybrid_setup_router import (
    Variant,
    _mrl,
    _selection_key,
    _variant_records,
)


def _record() -> dict:
    return {
        "benchmark": "hlsfactory_demo",
        "benchmark_lineage": "demo:one",
        "split": "train",
        "features": {
            "model_id": "teacher",
            "setup_strategy": "flash",
            "setup_router_version": 2.0,
            "source_loop_count": 3.0,
            "phase_b_latency_cycles": 100.0,
        },
        "provenance": {"dedup_key_sha256": "record-one"},
    }


def test_last_token_pool_handles_left_and_right_padding() -> None:
    hidden = torch.tensor(
        [
            [[1.0], [2.0], [3.0]],
            [[4.0], [5.0], [6.0]],
        ]
    )
    left_mask = torch.tensor([[0, 1, 1], [1, 1, 1]])
    right_mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    assert _last_token_pool(hidden, left_mask).tolist() == [[3.0], [6.0]]
    assert _last_token_pool(hidden, right_mask).tolist() == [[2.0], [6.0]]


def test_mrl_truncates_and_renormalizes() -> None:
    reduced = _mrl(np.asarray([3.0, 4.0, 9.0]), 2)
    np.testing.assert_allclose(reduced, [0.6, 0.8])
    assert np.linalg.norm(reduced) == pytest.approx(1.0)


def test_transformer_only_keeps_setup_features() -> None:
    variant = Variant(
        "transformer_both",
        2,
        structured=False,
        source_embedding=True,
        phase_b_embedding=True,
    )
    records = _variant_records(
        [_record()],
        variant,
        {
            "source:key": np.asarray([3.0, 4.0, 0.0]),
            "phase:key": np.asarray([0.0, 5.0, 12.0]),
        },
        {
            "record-one": {
                "benchmark_lineage": "demo:one",
                "split": "train",
                "source_embedding_key": "source:key",
                "phase_b_embedding_key": "phase:key",
            }
        },
    )
    features = records[0]["features"]
    assert features["model_id"] == "teacher"
    assert features["setup_strategy"] == "flash"
    assert "source_loop_count" not in features
    assert "phase_b_latency_cycles" not in features
    assert features["transformer_source_000"] == pytest.approx(0.6)
    assert features["transformer_source_001"] == pytest.approx(0.8)
    assert features["transformer_phase_b_000"] == pytest.approx(0.0)
    assert features["transformer_phase_b_001"] == pytest.approx(1.0)


def test_embedding_join_rejects_split_drift() -> None:
    variant = Variant("hybrid", 2, True, True, False)
    with pytest.raises(ValueError, match="split mismatch"):
        _variant_records(
            [_record()],
            variant,
            {"source:key": np.asarray([1.0, 0.0])},
            {
                "record-one": {
                    "benchmark_lineage": "demo:one",
                    "split": "test",
                    "source_embedding_key": "source:key",
                    "phase_b_embedding_key": "phase:key",
                }
            },
        )


def test_validation_selection_prefers_top3_before_tiny_regret_change() -> None:
    stronger_coverage = {
        "ranking": {
            "top_1_accuracy": 0.0,
            "top_3_oracle_coverage": 0.5,
            "learned_top_k": {
                "within_5pct_coverage": 0.5,
                "geomean_regret": 1.442,
            },
        }
    }
    tiny_regret_gain = {
        "ranking": {
            "top_1_accuracy": 0.0,
            "top_3_oracle_coverage": 0.0,
            "learned_top_k": {
                "within_5pct_coverage": 0.5,
                "geomean_regret": 1.441,
            },
        }
    }
    assert _selection_key(
        stronger_coverage,
        "structured",
    ) < _selection_key(tiny_regret_gain, "hybrid")
