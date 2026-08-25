import importlib.util
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "train_setup_router.py"
)
SPEC = importlib.util.spec_from_file_location("train_setup_router", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_router_feature_schema_rejects_reference_metrics() -> None:
    records = [{"features": {"source_loop_count": 2.0, "reference_cycles": 1}}]

    with pytest.raises(ValueError, match="forbidden router input"):
        MODULE._feature_schema(records)


def test_router_feature_schema_allows_source_array_references() -> None:
    records = [
        {
            "features": {
                "source_array_reference_count": 4.0,
                "phase_b_latency_cycles": 100.0,
                "setup_strategy": "flash",
            }
        }
    ]

    names, categorical, numeric = MODULE._feature_schema(records)
    assert "source_array_reference_count" in names
    assert categorical == ["setup_strategy"]
    assert "phase_b_latency_cycles" in numeric
