"""Tests for hybrid DATAFLOW contract checking."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataflow_contract_check import (
    CONTRACT_BREACH_SCHEMA,
    ContractBreach,
    extract_contract_json_block,
    hybrid_contract_check,
    merge_contract_reports,
    parse_llm_contract_report,
    static_contract_check,
)

ARTIFACTS = (
    Path(__file__).resolve().parents[1]
    / "artifacts/pc2/flash_all_new_skills_avoids_global_20260623_024548"
)


def _load(name: str) -> str:
    path = (
        ARTIFACTS
        / name
        / "devstral2__flash__all_new_skills_avoids_global"
        / f"{name}_dataflow.cpp"
    )
    return path.read_text(encoding="utf-8")


def test_static_detects_2mm_multi_writer_and_port_rw():
    report = static_contract_check(_load("hlsfactory_2mm"), top_function="kernel_2mm")
    rule_ids = {b.rule_id for b in report.breaches}
    assert not report.passed
    assert "local-buffer-multi-writer" in rule_ids
    assert "m_axi-port-concurrent-rw" in rule_ids
    symbols = {b.symbol for b in report.breaches}
    assert "local_D" in symbols or "D" in symbols


def test_static_detects_gemver_tile_loop_m_axi():
    report = static_contract_check(_load("hlsfactory_gemver"), top_function="kernel_gemver")
    rule_ids = {b.rule_id for b in report.breaches}
    assert not report.passed
    assert "tile-loop-m_axi-in-dataflow" in rule_ids


def test_breach_schema_roundtrip():
    breach = ContractBreach(
        rule_id="local-buffer-fanout",
        severity="error",
        symbol="local_tmp",
        tasks=["compute_fused_task", "store_tmp_task"],
        location="DATAFLOW region",
        message="fan-out",
        fix_skill_id="hls-dataflow-merge-parallel-consumers",
        source="static",
    )
    raw = breach.to_dict()
    restored = ContractBreach.from_dict(raw)
    assert restored is not None
    assert restored.rule_id == breach.rule_id
    assert restored.tasks == breach.tasks


def test_parse_llm_contract_json():
    text = """
```json
{
  "schema": "dataflow_contract_breach_v1",
  "passed": false,
  "breaches": [
    {
      "rule_id": "m_axi-bundle-multi-reader",
      "severity": "error",
      "symbol": "gmem4",
      "tasks": ["load_D_task", "compute_D_task"],
      "location": "DATAFLOW",
      "message": "two readers on gmem4",
      "fix_skill_id": "hls-distinct-gmem-bundle-per-port",
      "source": "llm"
    }
  ]
}
```
"""
    data = extract_contract_json_block(text)
    assert data.get("schema") == CONTRACT_BREACH_SCHEMA
    report = parse_llm_contract_report(text)
    assert not report.passed
    assert len(report.breaches) == 1
    assert report.breaches[0].source == "llm"


def test_merge_static_and_llm_dedupes():
    static = static_contract_check("void k() {}", top_function="k")
    llm = parse_llm_contract_report(
        "```json\n"
        + json.dumps(
            {
                "schema": CONTRACT_BREACH_SCHEMA,
                "passed": False,
                "breaches": [
                    {
                        "rule_id": "dataflow-min-tasks",
                        "severity": "error",
                        "symbol": None,
                        "tasks": [],
                        "location": "top",
                        "message": "too few tasks",
                        "fix_skill_id": None,
                        "source": "llm",
                    }
                ],
            }
        )
        + "\n```"
    )
    merged = merge_contract_reports(static, llm)
    assert not merged.passed
    assert merged.static_count >= 1
    assert merged.llm_count == 1


def test_valid_minimal_dataflow_kernel_passes_static():
    code = """
static void load_task(double A[8], double local_A[8]) {
#pragma HLS INLINE off
    load: for (int i = 0; i < 8; i++) local_A[i] = A[i];
}
static void compute_task(double local_A[8], double local_B[8]) {
#pragma HLS INLINE off
    compute: for (int i = 0; i < 8; i++) local_B[i] = local_A[i] * 2.0;
}
static void store_task(double B[8], double local_B[8]) {
#pragma HLS INLINE off
    store: for (int i = 0; i < 8; i++) B[i] = local_B[i];
}
extern "C" void kernel_test(double A[8], double B[8]) {
#pragma HLS INTERFACE m_axi port=A bundle=gmem0
#pragma HLS INTERFACE m_axi port=B bundle=gmem1
    double local_A[8];
    double local_B[8];
#pragma HLS DATAFLOW
    load_task(A, local_A);
    compute_task(local_A, local_B);
    store_task(B, local_B);
}
"""
    report = static_contract_check(code, top_function="kernel_test")
    assert report.passed, [b.message for b in report.breaches]


if __name__ == "__main__":
    test_static_detects_2mm_multi_writer_and_port_rw()
    test_static_detects_gemver_tile_loop_m_axi()
    test_breach_schema_roundtrip()
    test_parse_llm_contract_json()
    test_merge_static_and_llm_dedupes()
    test_valid_minimal_dataflow_kernel_passes_static()
    print("test_dataflow_contract_check: ok")
