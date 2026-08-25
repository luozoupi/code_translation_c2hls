from __future__ import annotations

import copy
import hashlib
import json
import os
from functools import reduce
from operator import mul
from pathlib import Path
from unittest import mock

import c2hls
import hls_eval
import pytest


REPO = Path(__file__).resolve().parents[1]
REGISTRY = REPO / "configs" / "hlsfactory_output_shapes.json"
SUITE = REPO / "configs" / "hlsfactory_development_suite.json"
EXTERNAL = Path(
    "/home/luo00466/code_translation-c2hls/benchmarks_external/"
    "HLSFactory/polybench_float_small"
)


def _size(shape: list[int]) -> int:
    return reduce(mul, shape, 1)


def _dump(name: str, count: int) -> str:
    return (
        f"begin dump: {name}\n"
        + " ".join("0.0" for _ in range(count))
        + f"\nend dump: {name}\n"
    )


def test_pinned_development_suite_and_shape_registry_cover_exactly_28() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    suite = json.loads(SUITE.read_text(encoding="utf-8"))
    kernels = suite["kernels"]

    assert suite["kernel_count"] == 28
    assert len(kernels) == len(set(kernels)) == 28
    assert set(kernels) == set(registry["benchmarks"])
    for benchmark, entry in registry["benchmarks"].items():
        assert len(entry["testbench_sha256"]) == 64, benchmark
        assert entry["outputs"], benchmark
        for name, output in entry["outputs"].items():
            assert _size(output["shape"]) > 0, (benchmark, name)
            assert _size(output["logical_shape"]) > 0, (benchmark, name)
            assert output["kind"] in {"float", "integer"}
            assert output["layout"]


def test_registry_is_bound_to_the_public_testbench_bytes() -> None:
    assert EXTERNAL.is_dir(), "pinned HLSFactory development inputs are unavailable"
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    for benchmark, entry in registry["benchmarks"].items():
        testbench = EXTERNAL / benchmark / "testbench.cpp"
        assert testbench.is_file(), benchmark
        assert hashlib.sha256(testbench.read_bytes()).hexdigest() == entry[
            "testbench_sha256"
        ], benchmark


def test_real_golden_preparation_uses_matrix_shape_not_flat_value_count() -> None:
    testbench = (EXTERNAL / "hlsfactory_2mm" / "testbench.cpp").read_text(
        encoding="utf-8"
    )
    native = {"success": True, "output": _dump("D", 40 * 80), "stage": "run"}
    inputs = {
        "meta": {"benchmark": "hlsfactory_2mm", "source_repo": "HLSFactory"},
        "c_code": "void kernel_2mm() {}",
        "testbench_code": testbench,
        "header_code": "",
        "header_name": "2mm.h",
        "extra_files": [],
    }
    with mock.patch.object(c2hls, "run_native_testbench", return_value=native):
        prepared = c2hls._prepare_independent_golden(inputs)

    assert prepared["success"]
    assert prepared["specs"]["D"]["shape"] == [40, 80]
    assert prepared["provenance"]["outputs"]["D"]["declared_shape"] == [40, 80]
    assert prepared["provenance"]["outputs"]["D"]["logical_shape"] == [40, 80]
    assert prepared["provenance"]["shape_registry"]["testbench_sha256"]


def test_same_count_wrong_dimensional_override_is_rejected_by_preparation() -> None:
    testbench = (EXTERNAL / "hlsfactory_2mm" / "testbench.cpp").read_text(
        encoding="utf-8"
    )
    native = {"success": True, "output": _dump("D", 40 * 80), "stage": "run"}
    inputs = {
        "meta": {
            "benchmark": "hlsfactory_2mm",
            "source_repo": "HLSFactory",
            # Same element count, wrong dimensional contract.
            "golden_output_specs": {"D": {"shape": [80, 40], "kind": "float"}},
        },
        "c_code": "void kernel_2mm() {}",
        "testbench_code": testbench,
        "header_code": "",
        "header_name": "2mm.h",
        "extra_files": [],
    }
    with mock.patch.object(c2hls, "run_native_testbench", return_value=native):
        prepared = c2hls._prepare_independent_golden(copy.deepcopy(inputs))

    assert not prepared["success"]
    assert "shape conflicts" in prepared["error"]


def test_changed_testbench_or_unregistered_kernel_fails_closed() -> None:
    testbench = (EXTERNAL / "hlsfactory_2mm" / "testbench.cpp").read_text(
        encoding="utf-8"
    )
    with mock.patch.object(
        c2hls,
        "run_native_testbench",
        return_value={"success": True, "output": _dump("D", 40 * 80)},
    ):
        changed = c2hls._prepare_independent_golden(
            {
                "meta": {"benchmark": "hlsfactory_2mm", "source_repo": "HLSFactory"},
                "c_code": "",
                "testbench_code": testbench + "\n// changed",
                "header_code": "",
            }
        )
        missing = c2hls._prepare_independent_golden(
            {
                "meta": {"benchmark": "hlsfactory_unknown", "source_repo": "HLSFactory"},
                "c_code": "",
                "testbench_code": testbench,
                "header_code": "",
            }
        )
    assert not changed["success"]
    assert "does not match audited shape contract" in changed["error"]
    assert not missing["success"]
    assert "no authoritative HLSFactory shape entry" in missing["error"]


def test_explicit_shape_registry_override_remains_hash_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    benchmark = "hlsfactory_2mm"
    testbench = (
        EXTERNAL / benchmark / "testbench.cpp"
    ).read_text(encoding="utf-8") + "\n// audited variant\n"
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    registry["benchmarks"][benchmark]["testbench_sha256"] = hashlib.sha256(
        testbench.encode("utf-8")
    ).hexdigest()
    override = tmp_path / "shape_registry.json"
    override.write_text(json.dumps(registry), encoding="utf-8")
    monkeypatch.setenv("C2HLS_HLSFACTORY_SHAPE_REGISTRY", str(override))

    with mock.patch.object(
        c2hls,
        "run_native_testbench",
        return_value={"success": True, "output": _dump("D", 40 * 80)},
    ):
        prepared = c2hls._prepare_independent_golden(
            {
                "meta": {"benchmark": benchmark, "source_repo": "HLSFactory"},
                "c_code": "",
                "testbench_code": testbench,
                "header_code": "",
            }
        )

    assert prepared["success"]
    assert prepared["provenance"]["shape_registry"]["path"] == str(override)


@pytest.mark.skipif(
    os.getenv("C2HLS_RUN_HLSFACTORY_GOLDEN_INTEGRATION") != "1",
    reason="set C2HLS_RUN_HLSFACTORY_GOLDEN_INTEGRATION=1 for all 28 native oracles",
)
@pytest.mark.parametrize(
    "benchmark",
    sorted(json.loads(REGISTRY.read_text(encoding="utf-8"))["benchmarks"]),
)
def test_all_28_public_testbenches_execute_and_match_shape_registry(
    benchmark: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    work_dir = tmp_path / benchmark
    monkeypatch.setattr(hls_eval, "make_tempdir", lambda prefix: str(work_dir))
    inputs = c2hls._load_benchmark_inputs(str(EXTERNAL / benchmark))

    prepared = c2hls._prepare_independent_golden(inputs)

    assert prepared["success"], prepared.get("error")
    declarations = json.loads(REGISTRY.read_text(encoding="utf-8"))[
        "benchmarks"
    ][benchmark]["outputs"]
    assert set(prepared["specs"]) == set(declarations)
    for name, declaration in declarations.items():
        assert prepared["specs"][name]["shape"] == declaration["shape"]
        assert prepared["provenance"]["outputs"][name]["count"] == _size(
            declaration["shape"]
        )
