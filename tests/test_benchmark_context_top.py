"""Tests for metadata-driven top function guidance (HLSFactory vs Rodinia)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from c2hls import _build_benchmark_context, resolve_metadata_top, top_function_policy_hint
import prompt_c2hls


def test_resolve_metadata_top_prefers_translated_hls_top():
    meta = {
        "translated_hls_top": "kernel_gemm",
        "hls_top": "workload",
        "kernel_top": "kernel_gemm",
    }
    assert resolve_metadata_top(meta) == "kernel_gemm"


def test_hlsfactory_policy_uses_kernel_not_workload():
    meta = {
        "benchmark": "hlsfactory_gesummv",
        "source_repo": "HLSFactory",
        "translated_hls_top": "kernel_gesummv",
        "kernel_top": "kernel_gesummv",
    }
    policy = top_function_policy_hint(meta)
    assert "HLSFactory" in policy
    assert "kernel_gesummv" in policy
    assert "Never rename to `workload`" in policy


def test_rodinia_policy_uses_workload():
    meta = {
        "benchmark": "nw",
        "source_repo": "rodinia-hls",
        "translated_hls_top": "workload",
        "kernel_top": "needwun",
    }
    policy = top_function_policy_hint(meta)
    assert "Rodinia/MachSuite" in policy
    assert "`workload`" in policy
    assert "needwun" in policy
    assert "static" in policy


def test_benchmark_context_includes_metadata_top():
    meta = {
        "benchmark": "hlsfactory_gemm",
        "source_repo": "HLSFactory",
        "translated_hls_top": "kernel_gemm",
        "kernel_top": "kernel_gemm",
    }
    ctx = _build_benchmark_context(meta, "gemm.h", 'extern "C" void kernel_gemm();', "")
    assert "kernel_gemm" in ctx
    assert "translated_hls_top" in ctx
    assert "HLSFactory" in ctx


def test_phase_b_prompt_references_metadata_top():
    prompt = prompt_c2hls.q_translate_c_to_hls.format(
        c_code="void foo() {}",
        header_code="#pragma once",
        benchmark_context="- Required HLS top function (metadata.json `translated_hls_top`): `kernel_gemm`.",
    )
    assert "metadata.json" in prompt
    assert "kernel_*` for HLSFactory" in prompt
    assert "`workload` for Rodinia/MachSuite" in prompt
    assert "{_TOP_FUNCTION_REQUIREMENT}" not in prompt


if __name__ == "__main__":
    test_resolve_metadata_top_prefers_translated_hls_top()
    test_hlsfactory_policy_uses_kernel_not_workload()
    test_rodinia_policy_uses_workload()
    test_benchmark_context_includes_metadata_top()
    test_phase_b_prompt_references_metadata_top()
    print("test_benchmark_context_top: ok")
