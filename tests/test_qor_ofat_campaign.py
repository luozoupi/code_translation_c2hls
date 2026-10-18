from __future__ import annotations

import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import analyze_qor_ofat_campaign as analyze  # noqa: E402
import run_qor_ofat_campaign as campaign  # noqa: E402


def _metrics(cycles: int, *, bram: int = 4) -> dict:
    return {
        "latency_cycles": cycles,
        "latency_cycles_worst": cycles,
        "interval": cycles + 1,
        "estimated_clock_period_ns": 3.1,
        "achieved_pipeline_ii_max": 1,
        "dsp": 2,
        "bram": bram,
        "lut": 20,
        "ff": 30,
        "uram": 0,
    }


def test_campaign_preflight_binds_exact_typed_knob(tmp_path: Path) -> None:
    benchmark = "hlsfactory_test"
    benchmark_dir = tmp_path / "benches" / benchmark
    benchmark_dir.mkdir(parents=True)
    source = tmp_path / "source.json"
    source.write_text(json.dumps({
        "benchmark": benchmark,
        "success": True,
        "hls_code": "void workload() {\n#pragma HLS PIPELINE II=1\n}\n",
        "csim": {"passed": True},
        "final_report": _metrics(100),
    }))
    case = {
        "case_id": "test_pipeline",
        "benchmark": benchmark,
        "result_path": str(source),
        "origin_step": "pipeline",
        "expected_knob_kind": "pipeline_ii",
        "expected_knob_name": "pipeline_ii@L2",
        "max_knobs": 1,
        "max_candidates": 3,
    }
    result = campaign._preflight_case(
        case, {"benchmarks_root": str(tmp_path / "benches")}
    )
    assert result["passed"] is True
    assert result["selected_knob"]["kind"] == "pipeline_ii"
    assert result["discoverable_candidate_count"] == 3


def test_campaign_preflight_validates_every_striped_gemm_lane(tmp_path: Path) -> None:
    benchmark = "hlsfactory_gemm"
    benchmark_dir = tmp_path / "benches" / benchmark
    benchmark_dir.mkdir(parents=True)
    (benchmark_dir / "gemm.h").write_text("#define NK 80\n")
    hls_code = """\
#include "gemm.h"
void kernel_gemm() {
double local_A[80][80];
double local_B[80][80];
#pragma HLS ARRAY_PARTITION variable=local_A cyclic factor=2 dim=2
#pragma HLS ARRAY_PARTITION variable=local_B cyclic factor=2 dim=1
// Compute: C = alpha * A * B + beta * C
for (int i = 0; i < 80; ++i) {
  for (int j = 0; j < 80; ++j) {
#pragma HLS PIPELINE II=1
    double sum = 0.0;
    for (int k = 0; k < 80; ++k) {
#pragma HLS UNROLL factor=2
      sum += local_A[i][k] * local_B[k][j];
    }
  }
}
// Store C
}
"""
    source = tmp_path / "source.json"
    source.write_text(json.dumps({
        "benchmark": benchmark,
        "hls_code": hls_code,
        "csim": {"passed": True},
        "final_report": _metrics(100),
    }))
    case = {
        "case_id": "gemm_striped_compute_lanes",
        "benchmark": benchmark,
        "result_path": str(source),
        "origin_step": "unroll",
        "expected_knob_kind": "compute_lanes",
        "expected_knob_name": "GEMM_LANES",
        "gemm_linked_lanes_parent": True,
        "gemm_linked_lanes_architecture": "striped_accumulators",
        "parent_lanes": 2,
        "factor_values": "1,2,4,8,16",
        "max_knobs": 1,
        "max_candidates": 4,
    }
    config = {"benchmarks_root": str(tmp_path / "benches")}
    valid = campaign._preflight_case(case, config)
    assert valid["passed"] is True
    assert valid["linked_control"]["domain_validation"]["nk"] == 80

    case["factor_values"] = "1,2,3,4"
    invalid = campaign._preflight_case(case, config)
    assert invalid["passed"] is False
    assert any("do not divide NK=80" in error for error in invalid["errors"])


def test_completed_output_requires_matching_source_hash(tmp_path: Path) -> None:
    output = tmp_path / "case.json"
    output.write_text(json.dumps({
        "schema_version": "c2hls.saved-qor-step-smoke.v1",
        "source_result": {"sha256": "abc"},
        "design_sweep": {"attempted": True},
    }))
    assert campaign._completed_output(output, "abc") is True
    assert campaign._completed_output(output, "different") is False


def test_analysis_exports_parent_normalized_parameter_effects(tmp_path: Path) -> None:
    case_path = tmp_path / "atax_pipeline_ii.json"
    case_path.write_text(json.dumps({
        "schema_version": "c2hls.saved-qor-step-smoke.v1",
        "benchmark": "hlsfactory_atax",
        "source_result": {"sha256": "source"},
        "design_sweep": {
            "discovered_knobs": [{
                "knob_id": "ii",
                "kind": "pipeline_ii",
                "name": "pipeline_ii@L2",
                "current_value": 1,
                "current_label": "1",
            }],
            "parent": {
                "candidate_id": "frozen_parent",
                "status": "feasible",
                "feasible": True,
                "csim": {"status": "passed"},
                "metrics": _metrics(100, bram=4),
            },
            "candidates": [{
                "candidate_id": "ii2",
                "status": "feasible",
                "feasible": True,
                "csim": {"status": "passed"},
                "pareto_frontier": True,
                "changed_knobs": [{"to": 2}],
                "metrics": _metrics(80, bram=6),
            }],
        },
    }))
    payload, rows = analyze._measurement_rows(case_path)
    candidate = next(row for row in rows if not row["is_parent"])
    assert candidate["schema_version"] == "c2hls.qor-ofat-measurement.v1"
    assert candidate["parameter_label"] == "2"
    assert candidate["latency_cycles_worst_ratio"] == 0.8
    assert candidate["top_level_transaction_interval_cycles"] == 81
    assert candidate["bram_ratio"] == 1.5
    assert candidate["mean_resource_ratio"] is not None
    summary = analyze._summary(
        {case_path.stem: payload}, [(case_path.stem, rows)]
    )
    assert summary["cycle_improvement_case_count"] == 1
    assert summary["resource_tie_improvement_case_count"] == 0
    assert "not a loop pipeline II" in summary["metric_definitions"]["interval"]
    report = analyze._report(summary)
    assert "## Metric definitions" in report
    assert "Top-level transaction interval" in report
    assert "## Controlled-run audit" in report


def test_analysis_resource_tie_break_matches_qor_selector() -> None:
    parent = {
        "candidate_id": "frozen_parent",
        "latency_cycles_worst": 100.0,
        "dsp": 8.0,
        "bram": 4.0,
        "lut": 20.0,
        "ff": 30.0,
        "uram": 0.0,
    }
    candidate = {
        **parent,
        "candidate_id": "lower-resource",
        "dsp": 4.0,
    }
    assert min((parent, candidate), key=analyze._selection_key) is candidate
