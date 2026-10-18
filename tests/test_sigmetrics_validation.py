from pathlib import Path
from unittest.mock import patch
import json

import pytest
import c2hls
import hls_eval
from hlsfactory_registry import resolve_shape_registry, registry_identity
from paper_baselines import load_public_benchmark_inputs

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "benchmarks_external/HLSFactory/polybench_float_small/hlsfactory_atax"
IMPORTED = Path("/home/luo00466/code_translation-c2hls-enhanced-fc27133/benchmarks/hlsfactory_atax")
DUMP = "==BEGIN DUMP_ARRAYS==\nbegin dump: y\n1.0 2.0\nend dump: y\n==END DUMP_ARRAYS==\n"
SPECS = {"y": {"shape": [2], "kind": "float", "atol": 1e-6, "rtol": 1e-5}}


@pytest.mark.parametrize("bench,registry", [
    (DEFAULT, "hlsfactory_output_shapes.json"),
    (IMPORTED, "hlsfactory_output_shapes_fc27133.json"),
])
def test_exact_harness_selects_its_registry(bench, registry, monkeypatch):
    monkeypatch.delenv("C2HLS_HLSFACTORY_SHAPE_REGISTRY", raising=False)
    meta = json.loads((bench / "metadata.json").read_text())
    tb = (bench / "testbench.cpp").read_text()
    assert resolve_shape_registry(meta, tb).name == registry
    assert registry_identity(meta, tb)["testbench_sha256"]


def test_explicit_wrong_registry_does_not_fall_back(monkeypatch):
    monkeypatch.setenv("C2HLS_HLSFACTORY_SHAPE_REGISTRY", str(ROOT / "configs/hlsfactory_output_shapes.json"))
    meta = json.loads((IMPORTED / "metadata.json").read_text())
    with pytest.raises(ValueError, match="does not match audited shape contract"):
        resolve_shape_registry(meta, (IMPORTED / "testbench.cpp").read_text())


@pytest.mark.parametrize("bench", [DEFAULT, IMPORTED])
def test_public_baseline_preserves_benchmark_id_and_uses_contract(bench, monkeypatch):
    monkeypatch.delenv("C2HLS_HLSFACTORY_SHAPE_REGISTRY", raising=False)
    name = "y"
    count = 124
    output = "begin dump: y\n" + "0.0 " * count + "\nend dump: y\n"
    with patch.object(c2hls, "run_native_testbench", return_value={"success": True, "output": output}):
        inputs = load_public_benchmark_inputs(bench, environ={"C2HLS_PART": "xcu280-fsvh2892-2L-e", "C2HLS_CLOCK_NS": "3.33"})
    assert inputs.benchmark == "hlsfactory_atax"
    assert inputs.independent_golden_specs[name]["shape"] == [124]
    assert inputs.independent_golden_provenance["status"] == "passed"


def check(log, stage="cosim"):
    return hls_eval._apply_independent_golden(
        {"success": True, "passed": True, "log": log},
        golden_output_text=DUMP, golden_output_specs=SPECS, stage=stage)


def test_shared_rtl_path_accepts_both_stages_but_csim_stays_strict():
    log = "Starting C TB testing\n" + DUMP + "Starting C post checking\n" + DUMP
    assert check(log)["passed"]
    assert check(log)["correctness"]["rtl_postcheck_dump_count"] == 1
    assert not check(log, "csim")["passed"]


@pytest.mark.parametrize("corrupt_first", [True, False])
def test_shared_rtl_path_checks_every_dump(corrupt_first):
    bad = DUMP.replace("2.0", "99.0")
    first, last = (bad, DUMP) if corrupt_first else (DUMP, bad)
    result = check(first + "Starting C post checking\n" + last)
    assert not result["passed"] and not result["success"]


@pytest.mark.parametrize("log", [DUMP, "Starting C post checking\n", DUMP + "Starting C post checking\n" + DUMP.split("==END")[0]])
def test_missing_or_incomplete_rtl_postcheck_fails(log):
    assert not check(log)["passed"]


@pytest.mark.parametrize("summary", [None, {}, {"ran": False, "passed": True, "kernel_runtime_cycles": 12},
    {"ran": True, "passed": True}, {"ran": True, "passed": True, "success": False, "kernel_runtime_cycles": 12}])
def test_not_executed_or_failed_measurement_cannot_pass(summary):
    assert not c2hls._selected_cosim_measurement_ok(summary)
