"""Tests for the post-flash DSE step (multi-PE GEMM nest rewrite)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import post_flash_dse as pfd
from skill_library import _coerce_skill_entry, render_skill_for_prompt, render_skill_set_for_prompt_full


REQUIRED_SKILL_IDS = (
    "hls-dse-gemm-multi-pe-latency-hiding",
    "hls-dse-gemm-simd-k-adder-tree",
    "hls-dse-partition-match-pe-simd",
    "avoid-dse-k-recurrence-on-single-c",
    "hls-dse-pick-pe-simd-for-kernel",
)


def test_dse_skills_file_is_valid_schema_11():
    path = pfd.resolve_dse_skills_path()
    assert path.is_file(), path
    errors = pfd.validate_dse_skill_entries(path)
    assert errors == [], errors
    data = json.loads(path.read_text(encoding="utf-8"))
    ids = [entry["id"] for entry in data["skills"]]
    for sid in REQUIRED_SKILL_IDS:
        assert sid in ids, sid
    for entry in data["skills"]:
        skill = _coerce_skill_entry(entry)
        assert skill is not None, entry.get("id")
        assert skill.pattern.strip()
        assert skill.strategy.strip()
        assert skill.template.strip()
        assert skill.required_steps
        assert skill.guards


def test_dse_skills_require_pe_simd_and_reject_k_recurrence():
    block, meta = pfd.build_dse_skills_prompt_block()
    assert meta["skill_count"] == 5
    assert set(REQUIRED_SKILL_IDS) <= set(meta["skill_ids"])
    low = block.lower()
    for needle in (
        "pe 16",
        "simd 4",
        "pipeline ii=1",
        "compute_j",
        "crow[p][j]",
        "array_partition",
        "avoid-dse-k-recurrence-on-single-c",
        "local_b[j][k0 + s]",
        "do not emit autosa",
        "32x8",
    ):
        assert needle in low, needle
    assert "crow[j] += local_a[i][k]" in low


def test_full_skill_render_keeps_pe_template_not_truncated():
    skills = pfd.load_dse_skills()
    pe = next(sk for sk in skills if sk.id == "hls-dse-gemm-multi-pe-latency-hiding")
    truncated = render_skill_for_prompt(pe)
    full = render_skill_set_for_prompt_full([pe])
    assert truncated.count("\n") < full.count("\n")
    assert "compute_i0:" in full
    assert "pe_mac:" in full
    assert "simd_k:" in full
    assert "DSP ~= PE*SIMD*5" in full or "dsp ~= pe*simd*5" in full.lower()
    # Compact render must not be what DSE injects.
    assert "compute_i0:" not in truncated or truncated.count("compute_k0") == 0 or "..." in truncated


def test_prompts_require_nest_rewrite_not_pragma_only():
    prompts = pfd.prompt_text_for_docs()
    system = prompts["system"].lower()
    user = prompts["initial_user"].lower()
    for needle in (
        "dse",
        "multi-pe",
        "simd",
        "latency hiding",
        "keep the exact top-level",
        "do not emit autosa",
        "pipeline the independent j",
    ):
        assert needle in system, needle
    assert "```kernel" in prompts["system"] or "```kernel" in prompts["initial_user"]
    assert "pragma-only" in system or "not pragma-only" in system
    assert "hls-dse-gemm-multi-pe-latency-hiding" in user
    assert prompts["skills_path"].endswith("post_flash_dse_pe_skill_entries.json")


def test_artifact_paths_are_dse_named():
    cell = Path("/tmp/cell")
    paths = pfd.artifact_paths(cell, "autosa_mm")
    assert paths["kernel"].name == "autosa_mm_dse.cpp"
    assert paths["result"].name == "autosa_mm_dse_result.json"
    assert paths["report"].name == "autosa_mm_dse_report.json"
    assert paths["history"].name == "autosa_mm_dse_history.json"


def test_enabled_default_off(monkeypatch):
    monkeypatch.delenv("C2HLS_POST_FLASH_DSE", raising=False)
    monkeypatch.delenv("C2HLS_DSE_CHAIN_FLASH", raising=False)
    assert pfd.dse_enabled() is False
    assert pfd.chain_after_flash() is False


def test_chain_after_flash_follows_enabled(monkeypatch):
    monkeypatch.setenv("C2HLS_POST_FLASH_DSE", "1")
    monkeypatch.delenv("C2HLS_DSE_CHAIN_FLASH", raising=False)
    assert pfd.dse_enabled() is True
    assert pfd.chain_after_flash() is True
    monkeypatch.setenv("C2HLS_DSE_CHAIN_FLASH", "0")
    assert pfd.chain_after_flash() is False


def test_architecture_ok_requires_multi_pe_dsp():
    assert pfd.architecture_ok({"dsp": 3}) is False
    assert pfd.architecture_ok({"dsp": 6}) is False
    assert pfd.architecture_ok({"dsp": 32}) is True
    assert pfd.architecture_ok({"dsp": 320}) is True
    assert pfd.architecture_ok({}) is False
    assert pfd.architecture_ok({"dsp": 352}, "autosa_mm_getting_started") is False
    assert pfd.architecture_ok({"dsp": 640}, "autosa_mm_getting_started") is True
    assert pfd.architecture_ok({"dsp": 64}, "autosa_mm_int16") is True
    assert pfd.architecture_ok({"dsp": 96}, "autosa_mm_catapult") is True
    assert pfd.architecture_ok({"dsp": 352}, "autosa_mm_32x8") is False
    assert pfd.architecture_ok({"dsp": 1280}, "autosa_mm_32x8") is True


def test_maybe_chain_dse_skipped_when_disabled(monkeypatch):
    monkeypatch.delenv("C2HLS_POST_FLASH_DSE", raising=False)
    monkeypatch.delenv("C2HLS_DSE_CHAIN_FLASH", raising=False)
    called = {"n": 0}

    def boom(**_kwargs):
        called["n"] += 1
        raise AssertionError("must not run")

    monkeypatch.setattr(pfd, "run_dse_for_cell", boom)
    out = pfd.maybe_chain_dse(
        bench="autosa_mm",
        bench_dir=Path("/tmp/bench"),
        cell_dir=Path("/tmp/cell"),
        orchestrator=object(),
        source_role="flash_final",
    )
    assert out is None
    assert called["n"] == 0


def test_discover_cells_walks_selected_cpp(tmp_path):
    cell = tmp_path / "variants" / "autosa_nav_n" / "autosa_mm" / "deepseek__flash"
    cell.mkdir(parents=True)
    (cell / "autosa_mm_selected.cpp").write_text("void autosa_mm() {}\n", encoding="utf-8")
    cells = pfd.discover_dse_cells(tmp_path)
    assert len(cells) == 1
    assert cells[0]["bench"] == "autosa_mm"
    assert Path(cells[0]["cell_dir"]) == cell


def test_promote_dse_as_selected_preserves_flash_seed(tmp_path):
    bench = "autosa_mm"
    cell = tmp_path
    flash = "// flash selected\n"
    dse = "// dse selected\n"
    (cell / f"{bench}_selected.cpp").write_text(flash, encoding="utf-8")
    (cell / f"{bench}_selected_report.json").write_text(
        json.dumps({"latency_cycles": 149082, "dsp": 6}) + "\n", encoding="utf-8"
    )
    (cell / f"{bench}_flow_manifest.json").write_text(
        json.dumps({
            "selected_from": "flash_opt",
            "latency_cycles": {"selected": 149082},
            "files": {"selected": f"{bench}_selected.cpp"},
        })
        + "\n",
        encoding="utf-8",
    )
    promotion = pfd.promote_dse_as_selected(
        cell_dir=cell,
        bench=bench,
        code=dse,
        report={"latency_cycles": 5000, "dsp": 320},
        result_payload={"success": True, "latency_cycles": 5000, "dsp": 320},
    )
    assert promotion["selected_stage"] == "dse"
    assert (cell / f"{bench}_selected.cpp").read_text(encoding="utf-8") == dse
    assert (cell / f"{bench}_flash_seed.cpp").read_text(encoding="utf-8") == flash
    manifest = json.loads((cell / f"{bench}_flow_manifest.json").read_text())
    assert manifest["selected_from"] == "dse"
    assert manifest["latency_cycles"]["selected"] == 5000
    assert manifest["latency_cycles"]["dse"] == 5000


def test_should_promote_requires_better_latency_and_dsp():
    assert pfd.should_promote_dse(
        success=True,
        latency_cycles=5000,
        baseline_latency=149082,
        dsp=320,
    )
    assert not pfd.should_promote_dse(
        success=True,
        latency_cycles=149000,
        baseline_latency=149082,
        dsp=6,
    )
    assert not pfd.should_promote_dse(
        success=True,
        latency_cycles=200000,
        baseline_latency=149082,
        dsp=320,
    )
    assert not pfd.should_promote_dse(
        success=False,
        latency_cycles=5000,
        baseline_latency=149082,
        dsp=320,
    )


def test_run_dse_for_cell_writes_artifacts_and_promotes(tmp_path, monkeypatch):
    bench = "autosa_mm"
    cell = tmp_path / "cell"
    cell.mkdir()
    bench_dir = tmp_path / "bench"
    bench_dir.mkdir()
    seed = 'extern "C" void autosa_mm() { /* flash */ }\n'
    (cell / f"{bench}_selected.cpp").write_text(seed, encoding="utf-8")
    (cell / f"{bench}_selected_report.json").write_text(
        json.dumps({"latency_cycles": 149082, "dsp": 6}) + "\n", encoding="utf-8"
    )
    rewritten = (
        'extern "C" void autosa_mm() {\n'
        "#define PE 16\n#define SIMD 4\n"
        "  compute_j: for (int j = 0; j < 64; ++j) {\n"
        "#pragma HLS PIPELINE II=1\n"
        "  }\n}\n"
    )

    class Orch:
        gpt_model = "fake"
        part = "xcu280-fsvh2892-2L-e"
        clock_ns = 3.33

        def _call_llm(self, messages, max_tokens=None):
            self.last_max_tokens = max_tokens
            return "```kernel\n" + rewritten + "```\n"

    fake_inputs = {
        "meta": {
            "part": "xcu280-fsvh2892-2L-e",
            "clock_ns": 3.33,
            "translated_hls_top": "autosa_mm",
            "hls_top": "autosa_mm",
        },
        "header_code": "#define I 64\n",
        "header_name": "kernel.h",
        "testbench_code": "int main(){return 0;}\n",
        "extra_files": [],
        "benchmark_context": "- autosa_mm",
    }

    monkeypatch.setattr(
        "c2hls._load_benchmark_inputs",
        lambda _bench_dir: fake_inputs,
    )
    monkeypatch.setattr("c2hls.compile_check_cpp", lambda *a, **k: (True, ""))
    monkeypatch.setattr(
        "c2hls._run_synth_csim_cosim",
        lambda *a, **k: {
            "synth": {
                "success": True,
                "report": {"latency_cycles": 4228, "dsp": 320, "lut": 10, "ff": 10, "bram": 1},
            },
            "csim": {"passed": True},
            "cosim": None,
        },
    )

    outcome = pfd.run_dse_for_cell(
        bench=bench,
        bench_dir=bench_dir,
        cell_dir=cell,
        orchestrator=Orch(),
        skip_existing=True,
    )
    assert outcome.success
    paths = pfd.artifact_paths(cell, bench)
    assert paths["kernel"].is_file()
    assert "PE 16" in paths["kernel"].read_text(encoding="utf-8")
    result = json.loads(paths["result"].read_text(encoding="utf-8"))
    assert result["success"] is True
    assert result["dsp"] == 320
    assert result["promoted"] is True
    assert (cell / f"{bench}_selected.cpp").read_text(encoding="utf-8").strip() == rewritten.strip()
    assert (cell / f"{bench}_flash_seed.cpp").read_text(encoding="utf-8") == seed


def test_dse_max_tokens_default_is_above_flash_empty_budget(monkeypatch):
    monkeypatch.delenv("C2HLS_DSE_MAX_TOKENS", raising=False)
    monkeypatch.delenv("C2HLS_FLASH_MAX_TOKENS", raising=False)
    monkeypatch.delenv("C2HLS_LLM_MAX_TOKENS", raising=False)
    assert pfd.dse_max_tokens() >= 16384
    monkeypatch.setenv("C2HLS_DSE_MAX_TOKENS", "4096")
    assert pfd.dse_max_tokens() == 8192
    monkeypatch.setenv("C2HLS_DSE_MAX_TOKENS", "65536")
    assert pfd.dse_max_tokens() == 65536


def test_repair_prompt_omits_skill_dump():
    prompts = pfd.prompt_text_for_docs()
    repair = prompts["repair_user"].lower()
    assert "```kernel" in repair
    assert "hls-dse-gemm-multi-pe-latency-hiding" not in repair
    assert "required steps:" not in repair
    assert "do not spend the token budget" in repair


def test_call_dse_llm_retries_empty_and_passes_max_tokens(monkeypatch):
    monkeypatch.delenv("C2HLS_DSE_MAX_TOKENS", raising=False)
    monkeypatch.delenv("C2HLS_FLASH_MAX_TOKENS", raising=False)
    monkeypatch.delenv("C2HLS_LLM_MAX_TOKENS", raising=False)
    monkeypatch.setenv("C2HLS_LLM_EMPTY_RETRIES", "3")
    calls = []

    class Orch:
        def _call_llm(self, messages, max_tokens=None):
            calls.append(max_tokens)
            if len(calls) < 2:
                return ""
            return "```kernel\nint ok() { return 1; }\n```"

    reply = pfd.call_dse_llm(Orch(), [{"role": "user", "content": "go"}], purpose="dse")
    assert "int ok" in reply
    assert len(calls) == 2
    assert calls[0] == pfd.dse_max_tokens()
    assert calls[0] >= 16384


def test_baseline_source_role_and_prompt(monkeypatch):
    monkeypatch.delenv("C2HLS_DSE_SOURCE_ROLE", raising=False)
    assert pfd.dse_source_role() == "flash_final"
    monkeypatch.setenv("C2HLS_DSE_SOURCE_ROLE", "baseline")
    assert pfd.dse_source_role() == "baseline"
    user = pfd.format_dse_initial_user(
        skills_block="",
        benchmark_context="ctx",
        header_name="kernel.h",
        header_code="// h",
        kernel_code="// k",
        synth_summary="(none)",
        source_role="baseline",
    )
    low = user.lower()
    assert "flash was **not** run" in user.lower() or "flash was not run" in low
    assert "baseline kernel" in low
    assert "flash already passed" not in low
    assert "flash kernel (seed)" not in low
    assert "Flash already produced" not in pfd._SYSTEM_BASELINE
    assert "after flash" not in pfd._SYSTEM_BASELINE
    assert "flash skipped" in pfd._SYSTEM_BASELINE.lower()


def test_run_dse_for_cell_from_baseline_uses_hls_baseline(tmp_path, monkeypatch):
    bench = "autosa_mm"
    cell = tmp_path / "cell"
    cell.mkdir()
    bench_dir = tmp_path / "bench"
    bench_dir.mkdir()
    baseline = 'extern "C" void autosa_mm() { /* naive ijk */ }\n'
    (bench_dir / "hls_baseline.cpp").write_text(baseline, encoding="utf-8")
    (cell / f"{bench}_selected.cpp").write_text(
        'extern "C" void autosa_mm() { /* must not be the seed */ }\n',
        encoding="utf-8",
    )
    rewritten = (
        'extern "C" void autosa_mm() {\n'
        "#define PE 16\n#define SIMD 4\n"
        "  compute_j: for (int j = 0; j < 64; ++j) {\n"
        "#pragma HLS PIPELINE II=1\n"
        "  }\n}\n"
    )
    captured = {}

    class Orch:
        gpt_model = "fake"
        part = "xcu280-fsvh2892-2L-e"
        clock_ns = 3.33

        def _call_llm(self, messages, max_tokens=None):
            captured["messages"] = messages
            return "```kernel\n" + rewritten + "```\n"

    fake_inputs = {
        "meta": {
            "part": "xcu280-fsvh2892-2L-e",
            "clock_ns": 3.33,
            "translated_hls_top": "autosa_mm",
            "hls_top": "autosa_mm",
        },
        "header_code": "#define I 64\n",
        "header_name": "kernel.h",
        "testbench_code": "int main(){return 0;}\n",
        "extra_files": [],
        "benchmark_context": "- autosa_mm",
    }
    monkeypatch.setattr("c2hls._load_benchmark_inputs", lambda _bench_dir: fake_inputs)
    monkeypatch.setattr("c2hls.compile_check_cpp", lambda *a, **k: (True, ""))
    monkeypatch.setattr(
        "c2hls._run_synth_csim_cosim",
        lambda *a, **k: {
            "synth": {
                "success": True,
                "report": {"latency_cycles": 13160, "dsp": 320, "lut": 10, "ff": 10, "bram": 1},
            },
            "csim": {"passed": True},
            "cosim": None,
        },
    )

    outcome = pfd.run_dse_for_cell(
        bench=bench,
        bench_dir=bench_dir,
        cell_dir=cell,
        orchestrator=Orch(),
        source_role="baseline",
        skip_existing=True,
    )
    assert outcome.success
    user = captured["messages"][1]["content"]
    assert "naive ijk" in user
    assert "must not be the seed" not in user
    assert "Baseline kernel (seed, no flash)" in user
    assert "Flash was **not** run" in user
    result = json.loads(pfd.artifact_paths(cell, bench)["result"].read_text(encoding="utf-8"))
    assert result["source_role"] == "baseline"
    assert result["source_kernel"] == "hls_baseline.cpp"
    assert result["source_kernel_role"] == "baseline"


def test_resolve_baseline_kernel_prefers_cell_then_hls_baseline(tmp_path):
    cell = tmp_path / "cell"
    bench = tmp_path / "bench"
    cell.mkdir()
    bench.mkdir()
    (bench / "plain.cpp").write_text("plain\n", encoding="utf-8")
    (bench / "hls_baseline.cpp").write_text("hls_baseline\n", encoding="utf-8")
    assert pfd.resolve_baseline_kernel(cell, "autosa_mm", bench).name == "hls_baseline.cpp"
    (cell / "autosa_mm_baseline.cpp").write_text("cell_baseline\n", encoding="utf-8")
    assert pfd.resolve_baseline_kernel(cell, "autosa_mm", bench).name == "autosa_mm_baseline.cpp"


def test_run_dse_rejects_unknown_source_role(tmp_path):
    outcome = pfd.run_dse_for_cell(
        bench="autosa_mm",
        bench_dir=tmp_path,
        cell_dir=tmp_path,
        orchestrator=object(),
        source_role="flash_opt",
        skip_existing=True,
    )
    assert outcome.success is False
    assert "flash-final or baseline" in outcome.error


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
