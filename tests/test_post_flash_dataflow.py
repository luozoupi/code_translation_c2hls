"""Tests for post-flash DATAFLOW helpers."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import post_flash_dataflow as pfd


def test_extract_kernel_block():
    text = """
```kernel
void kernel_top() {}
```
"""
    code = pfd.extract_kernel_block(text)
    assert "kernel_top" in code


def test_extract_kernel_block_labeled_include():
    text = '```kernel\n#include "gemm.h"\nvoid kernel_gemm() {}\n```'
    code = pfd.extract_kernel_block(text)
    assert code.startswith('#include "gemm.h"')
    assert "kernel_gemm" in code
    assert not code.lower().startswith("kernel\n")


def test_sanitize_kernel_source():
    assert pfd.sanitize_kernel_source("kernel\n#include \"x.h\"\n") == '#include "x.h"'


def test_extract_kernel_block_generic_fence():
    text = "```cpp\nvoid foo() {}\n```"
    assert "foo" in pfd.extract_kernel_block(text)


def test_prepare_recovered_kernel_strips_workload_and_moves_pragmas():
    trisolv_tail = '''
extern "C" {
void workload(double L[N + 0][N + 0], double x[N + 0], double b[N + 0]) {
#pragma HLS INTERFACE m_axi port=L offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=return bundle=control
    kernel_trisolv(L, x, b);
}
}'''
    head = '''#include "trisolv.h"
extern "C" {
void kernel_trisolv(double L[N + 0][N + 0], double x[N + 0], double b[N + 0])
{
    kernel_trisolv_body();
}
}
'''
    code, meta = pfd.prepare_recovered_kernel_for_validate(head + trisolv_tail, "kernel_trisolv")
    assert meta["stripped_workload"] is True
    assert meta["moved_interface_pragmas"] is True
    assert "void workload" not in code
    assert "#pragma HLS INTERFACE m_axi port=L" in code


def test_dataflow_prompt_covers_hls_failure_modes():
    for policy in pfd.PROMPT_POLICIES:
        prompts = pfd.prompt_text_for_docs(policy)
        system = prompts["system"].lower()
        repair = prompts["repair_user"].lower()
        combined = (system + repair).lower()
        for needle in (
            "hls 200-1013",
            "hls 200-979",
            "fused read task",
            "fused write task",
            "exactly one writer",
            "pre-output checklist",
            "undeclared loop",
            "#pragma hls dataflow",
            "gmem0",
            "never omit",
            "name: for",
        ):
            assert needle in combined, (policy, needle)
        assert "200-1013" in repair
        assert "200-979" in repair
        assert "keep `#pragma hls dataflow`" in repair
        assert "gmem0" in repair


def test_prompt_policy_system_skills_layout():
    bundle = pfd.compose_dataflow_prompts("system_skills")
    assert "[skill " in bundle.system_prompt
    assert "{skills_block}" not in bundle.initial_user_template
    user = pfd.format_dataflow_initial_user(
        bundle,
        benchmark_context="- bench",
        header_name="k.h",
        header_code="// h",
        kernel_code="// k",
    )
    assert "[skill " not in user
    assert len(bundle.system_prompt) > len(user)


def test_prompt_policy_user_skills_layout():
    bundle = pfd.compose_dataflow_prompts("user_skills")
    assert "[skill " not in bundle.system_prompt
    user = pfd.format_dataflow_initial_user(
        bundle,
        benchmark_context="- bench",
        header_name="k.h",
        header_code="// h",
        kernel_code="// k",
    )
    assert "[skill " in user
    assert "## Task — DATAFLOW refactor" in user
    assert len(user) > 5000


def test_format_results_root_name_includes_policy():
    name = pfd.format_results_root_name(
        "20260704_120000",
        results_suffix="fanout_fix",
        prompt_policy="user_skills",
    )
    assert name == "post_flash_dataflow_results_20260704_120000_pp-user_skills_fanout_fix"
    assert pfd.kernel_bundle_dir_name("user_skills") == "kernel_bundle_pp-user_skills"


def test_resolve_prompt_policy_rejects_unknown():
    try:
        pfd.resolve_prompt_policy("bogus")
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "bogus" in str(exc)


def test_dataflow_skills_header_mandates_dataflow():
    block, _meta = pfd.build_dataflow_skills_prompt_block()
    low = block.lower()
    assert "always emit `#pragma hls dataflow`" in low
    assert "never omit dataflow" in low


def test_dataflow_prompt_is_kernel_blind():
    # Base templates only — skill catalog may reference generic patterns (e.g. gemm).
    combined = (pfd._SYSTEM + pfd._INITIAL_USER_LEGACY).lower()
    for forbidden in ("atax", "bicg", "durbin", "kernel_gemm", "kernel_atax", "hlsfactory_"):
        assert forbidden not in combined, forbidden


def test_dataflow_skills_load_from_flash_overlay():
    from c2hls_paths import FLASH_NO_RMW_M_AXI_SKILL_ENTRIES_JSON

    errors = pfd.validate_dataflow_skill_entries(FLASH_NO_RMW_M_AXI_SKILL_ENTRIES_JSON)
    assert errors == [], errors
    block, meta = pfd.build_dataflow_skills_prompt_block()
    assert meta["skill_count"] == 33
    assert "hls-distinct-gmem-bundle-per-port" in meta["skill_ids"]
    assert "avoid-dataflow-without-stream-or-overlap-proof" in block


def test_prompt_text_includes_contract_prompts():
    prompts = pfd.prompt_text_for_docs("user_skills")
    assert "contract_audit_system" in prompts
    assert "dataflow_contract_breach_v1" in prompts["contract_audit_system"]
    assert "contract_fix_user" in prompts
    assert "Contract breaches" in prompts["contract_fix_user"]


if __name__ == "__main__":
    test_extract_kernel_block()
    test_extract_kernel_block_labeled_include()
    test_sanitize_kernel_source()
    test_extract_kernel_block_generic_fence()
    test_prepare_recovered_kernel_strips_workload_and_moves_pragmas()
    test_dataflow_prompt_covers_hls_failure_modes()
    test_prompt_policy_system_skills_layout()
    test_prompt_policy_user_skills_layout()
    test_format_results_root_name_includes_policy()
    test_resolve_prompt_policy_rejects_unknown()
    test_dataflow_skills_header_mandates_dataflow()
    test_dataflow_prompt_is_kernel_blind()
    test_dataflow_skills_load_from_flash_overlay()
    test_prompt_text_includes_contract_prompts()
    print("test_post_flash_dataflow: ok")
