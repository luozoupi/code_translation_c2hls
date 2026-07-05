"""Tests for post-flash pragma optimization helpers."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import post_flash_pragma_opt as ppo
from c2hls_paths import VITIS_PRAGMAS_CURATED_MD


def test_pragma_guide_loads():
    text = ppo.load_pragma_guide(max_chars=5000)
    assert "Vitis HLS 2023.2" in text
    assert "#pragma HLS pipeline" in text.lower() or "pipeline" in text.lower()
    assert VITIS_PRAGMAS_CURATED_MD.is_file()


def test_prompts_require_core_pragmas():
    prompts = ppo.prompt_text_for_docs()
    system = prompts["system"].lower()
    for needle in ("pipeline", "unroll", "array_partition", "quality over quantity"):
        assert needle in system, needle
    assert "vitis_hls_2023_2_pragmas_curated.md" in prompts["pragma_guide_path"]


def test_artifact_paths_by_source():
    cell = Path("/tmp/cell")
    flash = ppo.artifact_paths(cell, "hlsfactory_gemm", "flash_final")
    df = ppo.artifact_paths(cell, "hlsfactory_gemm", "dataflow")
    assert flash["kernel"].name == "hlsfactory_gemm_pragma_opt.cpp"
    assert df["kernel"].name == "hlsfactory_gemm_dataflow_pragma_opt.cpp"


def test_summarize_synth_report_empty():
    assert "no prior" in ppo.summarize_synth_report(None)


if __name__ == "__main__":
    test_pragma_guide_loads()
    test_prompts_require_core_pragmas()
    test_artifact_paths_by_source()
    test_summarize_synth_report_empty()
    print("test_post_flash_pragma_opt: ok")
