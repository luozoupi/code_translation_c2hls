"""Tests for the post-DSE stream/I/O step (PE modules + hls::stream DATAFLOW)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import post_flash_stream as pfs
from skill_library import _coerce_skill_entry, render_skill_for_prompt, render_skill_set_for_prompt_full


REQUIRED_SKILL_IDS = (
    "hls-stream-pe-array-dataflow",
    "hls-stream-io-overlap-dram",
    "hls-stream-b-systolic-forward",
    "avoid-stream-shared-local-abc",
    "hls-stream-match-pe-pack-to-data-t",
    "hls-stream-tile-loop-around-dataflow",
)


def test_stream_skills_file_is_valid_schema_11():
    path = pfs.resolve_stream_skills_path()
    assert path.is_file(), path
    errors = pfs.validate_stream_skill_entries(path)
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


def test_stream_skills_require_pe_streams_and_reject_shared_locals():
    block, meta = pfs.build_stream_skills_prompt_block()
    assert meta["skill_count"] == 6
    assert set(REQUIRED_SKILL_IDS) <= set(meta["skill_ids"])
    low = block.lower()
    for needle in (
        "hls::stream",
        "dataflow",
        "pe_num=16",
        "simd 4",
        "ap_uint<128>",
        "fifo_a",
        "fifo_b",
        "systolic",
        "avoid-stream-shared-local-abc",
        "do not emit autosa",
        "local_a[i][k]",
        "struct-of-floats",
        "ram_2p",
        "mux_case_0",
        "do not complete-partition crow",
        "pe_kj",
        "1024",
        "do not use loop_flatten off",
        "do not dependence",
    ):
        assert needle in low, needle
    assert "struct vec4" not in block or "avoid" in low
    assert "ap_uint<256>" in block
    assert "32x8" in low or "simd=8" in low



def test_full_skill_render_keeps_pe_stream_template_not_truncated():
    skills = pfs.load_stream_skills()
    pe = next(sk for sk in skills if sk.id == "hls-stream-pe-array-dataflow")
    truncated = render_skill_for_prompt(pe)
    full = render_skill_set_for_prompt_full([pe])
    assert truncated.count("\n") < full.count("\n")
    assert "ap_uint<128>" in full
    assert "unpack4(" in full or "unpack4" in full
    assert "load_A(" in full
    assert "fifo_B[" in full or "fifo_b[" in full.lower()
    assert "#pragma HLS DATAFLOW" in full
    assert "pe_kj:" in full
    assert "#pragma HLS LOOP_FLATTEN off" not in full
    assert "#pragma HLS DEPENDENCE variable=Crow inter false" not in full
    assert "BIND_STORAGE" in full
    assert "ram_2p" in full.lower()
    assert "fifo_C.write" in full
    assert "ARRAY_PARTITION variable=Crow complete" not in full
    assert truncated.count("ap_uint<128>") == 0 or "..." in truncated


def test_prompts_require_stream_rewrite_not_dataflow_on_lcst():
    prompts = pfs.prompt_text_for_docs()
    system = prompts["system"].lower()
    user = prompts["initial_user"].lower()
    for needle in (
        "stream",
        "dataflow",
        "hls::stream",
        "keep the exact top-level",
        "do not emit autosa",
        "pe modules",
        "ap_uint<128>",
        "ram_2p",
        "crow",
        "flatten",
        "1024",
        "loop_flatten off",
    ):
        assert needle in system, needle
    assert "```kernel" in prompts["system"] or "```kernel" in prompts["initial_user"]
    assert "hls-stream-pe-array-dataflow" in user
    assert prompts["skills_path"].endswith("post_flash_stream_pe_io_skill_entries.json")
    repair = prompts["repair_user"].lower()
    assert "hls-stream-pe-array-dataflow" not in repair
    assert "do not spend the token budget" in repair
    assert "handshake" in repair
    assert "chapter" in repair
    assert "pipo" in repair
    assert "whole matrix" in repair


def test_artifact_paths_are_stream_named():
    cell = Path("/tmp/cell")
    paths = pfs.artifact_paths(cell, "autosa_mm")
    assert paths["kernel"].name == "autosa_mm_stream.cpp"
    assert paths["result"].name == "autosa_mm_stream_result.json"
    assert paths["report"].name == "autosa_mm_stream_report.json"
    assert paths["history"].name == "autosa_mm_stream_history.json"


def test_enabled_default_off(monkeypatch):
    monkeypatch.delenv("C2HLS_POST_FLASH_STREAM", raising=False)
    monkeypatch.delenv("C2HLS_STREAM_CHAIN_FLASH", raising=False)
    assert pfs.stream_enabled() is False
    assert pfs.chain_after_flash() is False


def test_chain_after_flash_follows_enabled(monkeypatch):
    monkeypatch.setenv("C2HLS_POST_FLASH_STREAM", "1")
    monkeypatch.delenv("C2HLS_STREAM_CHAIN_FLASH", raising=False)
    assert pfs.stream_enabled() is True
    assert pfs.chain_after_flash() is True
    monkeypatch.setenv("C2HLS_STREAM_CHAIN_FLASH", "0")
    assert pfs.chain_after_flash() is False


def test_architecture_ok_requires_packed_streams_dsp_and_ii1():
    dse_lcst = (
        "#define PE 16\n"
        "data_t local_A[I][K];\n"
        "extern \"C\" void autosa_mm() { compute_j: for (int j=0;j<64;++j) {} }\n"
    )
    float_struct = (
        '#include <hls_stream.h>\n'
        "#pragma HLS DATAFLOW\n"
        "struct vec4 { data_t v0, v1, v2, v3; };\n"
        "hls::stream<vec4> fifo_A[16];\n"
        "static void mm_pe(hls::stream<vec4> &fifo_A, hls::stream<vec4> &fifo_B_in);\n"
    )
    packed = (
        '#include <ap_int.h>\n'
        '#include <hls_stream.h>\n'
        "typedef ap_uint<128> vec4_bits;\n"
        "#pragma HLS DATAFLOW\n"
        "hls::stream<vec4_bits> fifo_A[16];\n"
        "static void mm_pe(hls::stream<vec4_bits> &fifo_A);\n"
        "compute_j: for (int j = 0; j < 64; ++j) {}\n"
    )
    assert pfs.architecture_ok(dse_lcst, {"dsp": 352}) is False
    assert pfs.architecture_ok(float_struct, {"dsp": 320}) is False
    assert pfs.architecture_ok(packed, {"dsp": 80}) is False
    assert pfs.architecture_ok(packed, {"dsp": 32}) is False
    assert pfs.architecture_ok(
        packed,
        {
            "dsp": 320,
            "feedback": {
                "scopes": [{"name": "pe_k0_compute_j", "pipeline_ii": 4}],
            },
        },
    ) is False
    assert pfs.architecture_ok(
        packed,
        {
            "dsp": 320,
            "latency_cycles": 4285,
            "interval": 4173,
            "feedback": {
                "scopes": [{"name": "pe_k0_compute_j", "pipeline_ii": 1}],
            },
        },
    ) is True
    packed_crow_complete = packed + (
        "#pragma HLS ARRAY_PARTITION variable=Crow complete\n"
    )
    assert pfs.architecture_ok(
        packed_crow_complete,
        {
            "dsp": 320,
            "feedback": {
                "scopes": [{"name": "compute_j", "pipeline_ii": 1}],
            },
        },
    ) is False
    packed_flatten_off = packed + "#pragma HLS LOOP_FLATTEN off\n"
    assert pfs.architecture_ok(
        packed_flatten_off,
        {
            "dsp": 320,
            "feedback": {
                "scopes": [{"name": "pe_kj", "pipeline_ii": 1}],
            },
        },
    ) is False
    assert pfs.architecture_ok("", {"dsp": 320}) is False
    int16_ok = {
        "dsp": 64,
        "latency_cycles": 4285,
        "interval": 4173,
        "feedback": {"scopes": [{"name": "pe_kj", "pipeline_ii": 1}]},
    }
    assert pfs.architecture_ok(packed, int16_ok) is False  # default floor 200
    assert pfs.architecture_ok(packed, int16_ok, "autosa_mm_int16") is True
    assert pfs.architecture_ok(
        packed,
        {"dsp": 352, "feedback": {"scopes": [{"name": "pe_kj", "pipeline_ii": 1}]}},
        "autosa_mm_getting_started",
    ) is False
    assert pfs.architecture_ok(
        packed,
        {"dsp": 640, "feedback": {"scopes": [{"name": "pe_kj", "pipeline_ii": 1}]}},
        "autosa_mm_getting_started",
    ) is False  # DATAFLOW not wrapped by i-tile loop
    wrapped = packed.replace(
        "#pragma HLS DATAFLOW\n",
        "for (int i0 = 0; i0 < I; i0 += PE_NUM) {\n#pragma HLS DATAFLOW\n",
    )
    gs_ok = {
        "dsp": 640,
        "latency_cycles": 4285,
        "interval": 4173,
        "feedback": {"scopes": [{"name": "pe_kj", "pipeline_ii": 1}]},
    }
    assert pfs.architecture_ok(wrapped, gs_ok, "autosa_mm_getting_started") is True
    inner_tile = wrapped + (
        "void mm_pe(hls::stream<vec4_bits> &fifo_A) {\n"
        "  for (int tile = 0; tile < ROWS_PER_PE; ++tile) {\n"
        "    pe_kj: for (int t = 0; t < 1024; ++t) {}\n"
        "  }\n"
        "}\n"
    )
    assert pfs.architecture_ok(inner_tile, gs_ok, "autosa_mm_getting_started") is False
    assert pfs.architecture_ok(wrapped, gs_ok, "autosa_mm_intel") is True
    wrapped256 = wrapped.replace("ap_uint<128>", "ap_uint<256>")
    x8_ok = {
        "dsp": 1280,
        "latency_cycles": 1200,
        "interval": 1100,
        "feedback": {"scopes": [{"name": "pe_kj", "pipeline_ii": 1}]},
    }
    assert pfs.architecture_ok(packed, x8_ok, "autosa_mm_32x8") is False
    assert pfs.architecture_ok(wrapped256, x8_ok, "autosa_mm_32x8") is True
    assert pfs.architecture_ok(
        wrapped256,
        {"dsp": 320, "latency_cycles": 1200, "interval": 1100,
         "feedback": {"scopes": [{"name": "pe_kj", "pipeline_ii": 1}]}},
        "autosa_mm_32x8",
    ) is False


def test_architecture_ok_requires_dram_compute_overlap():
    packed = (
        '#include <ap_int.h>\n'
        '#include <hls_stream.h>\n'
        "typedef ap_uint<128> vec4_bits;\n"
        "#pragma HLS DATAFLOW\n"
        "hls::stream<vec4_bits> fifo_A[16];\n"
        "static void mm_pe(hls::stream<vec4_bits> &fifo_A);\n"
        "compute_j: for (int j = 0; j < 64; ++j) {}\n"
    )
    lcst = {
        "dsp": 320,
        "latency_cycles": 12893,
        "interval": 4553,
        "feedback": {"scopes": [{"name": "pe_kj", "pipeline_ii": 1}]},
    }
    assert pfs.architecture_ok(packed, lcst) is False
    miss = pfs.architecture_miss_message(packed, lcst)
    assert "12893" in miss
    assert "4553" in miss
    assert "pragma missing" not in miss.lower()


def test_maybe_chain_stream_skipped_when_disabled(monkeypatch):
    monkeypatch.delenv("C2HLS_POST_FLASH_STREAM", raising=False)
    monkeypatch.delenv("C2HLS_STREAM_CHAIN_FLASH", raising=False)
    called = {"n": 0}

    def boom(**_kwargs):
        called["n"] += 1
        raise AssertionError("must not run")

    monkeypatch.setattr(pfs, "run_stream_for_cell", boom)
    out = pfs.maybe_chain_stream(
        bench="autosa_mm",
        bench_dir=Path("/tmp/bench"),
        cell_dir=Path("/tmp/cell"),
        orchestrator=object(),
        source_role="flash_final",
    )
    assert out is None
    assert called["n"] == 0


def test_resolve_stream_source_prefers_dse_kernel(tmp_path):
    bench = "autosa_mm"
    cell = tmp_path
    (cell / f"{bench}_selected.cpp").write_text("// selected leftover\n", encoding="utf-8")
    (cell / f"{bench}_selected_report.json").write_text(
        json.dumps({"latency_cycles": 13160, "dsp": 352}) + "\n", encoding="utf-8"
    )
    (cell / f"{bench}_dse.cpp").write_text("// dse seed\n", encoding="utf-8")
    (cell / f"{bench}_dse_result.json").write_text(
        json.dumps({"success": True, "latency_cycles": 13160, "dsp": 352}) + "\n",
        encoding="utf-8",
    )
    (cell / f"{bench}_dse_report.json").write_text(
        json.dumps({"latency_cycles": 13160, "dsp": 352}) + "\n", encoding="utf-8"
    )
    path, role, report = pfs.resolve_stream_source_kernel(cell, bench)
    assert path == cell / f"{bench}_dse.cpp"
    assert role == "dse"
    assert report["latency_cycles"] == 13160


def test_promote_stream_as_selected_preserves_dse_kernel(tmp_path):
    bench = "autosa_mm"
    cell = tmp_path
    dse = "// dse selected\n"
    streamed = "// stream selected\n"
    (cell / f"{bench}_selected.cpp").write_text(dse, encoding="utf-8")
    (cell / f"{bench}_dse.cpp").write_text(dse, encoding="utf-8")
    (cell / f"{bench}_selected_report.json").write_text(
        json.dumps({"latency_cycles": 13160, "dsp": 352}) + "\n", encoding="utf-8"
    )
    (cell / f"{bench}_flow_manifest.json").write_text(
        json.dumps({
            "selected_from": "dse",
            "latency_cycles": {"selected": 13160, "dse": 13160},
            "files": {"selected": f"{bench}_selected.cpp", "dse": f"{bench}_dse.cpp"},
        })
        + "\n",
        encoding="utf-8",
    )
    promotion = pfs.promote_stream_as_selected(
        cell_dir=cell,
        bench=bench,
        code=streamed,
        report={"latency_cycles": 5000, "dsp": 320},
        result_payload={"success": True, "latency_cycles": 5000, "dsp": 320},
    )
    assert promotion["selected_stage"] == "stream"
    assert (cell / f"{bench}_selected.cpp").read_text(encoding="utf-8") == streamed
    assert (cell / f"{bench}_dse.cpp").read_text(encoding="utf-8") == dse
    manifest = json.loads((cell / f"{bench}_flow_manifest.json").read_text())
    assert manifest["selected_from"] == "stream"
    assert manifest["latency_cycles"]["selected"] == 5000
    assert manifest["latency_cycles"]["stream"] == 5000
    assert manifest["latency_cycles"]["dse"] == 13160


def test_should_promote_requires_better_latency_and_streams():
    assert pfs.should_promote_stream(
        success=True,
        latency_cycles=5000,
        baseline_latency=13160,
        dsp=320,
        code="#pragma HLS DATAFLOW\nhls::stream<float> s;\n",
    )
    assert not pfs.should_promote_stream(
        success=True,
        latency_cycles=13100,
        baseline_latency=13160,
        dsp=6,
        code="#pragma HLS DATAFLOW\nhls::stream<float> s;\n",
    )
    assert not pfs.should_promote_stream(
        success=True,
        latency_cycles=20000,
        baseline_latency=13160,
        dsp=320,
        code="#pragma HLS DATAFLOW\nhls::stream<float> s;\n",
    )
    assert not pfs.should_promote_stream(
        success=True,
        latency_cycles=5000,
        baseline_latency=13160,
        dsp=320,
        code="data_t local_A[I][K];\n",
    )


def test_run_stream_for_cell_writes_artifacts_and_promotes(tmp_path, monkeypatch):
    bench = "autosa_mm"
    cell = tmp_path / "cell"
    cell.mkdir()
    bench_dir = tmp_path / "bench"
    bench_dir.mkdir()
    dse = 'extern "C" void autosa_mm() { /* dse lcst */ }\n'
    (cell / f"{bench}_dse.cpp").write_text(dse, encoding="utf-8")
    (cell / f"{bench}_dse_result.json").write_text(
        json.dumps({"success": True, "latency_cycles": 13160, "dsp": 352}) + "\n",
        encoding="utf-8",
    )
    (cell / f"{bench}_dse_report.json").write_text(
        json.dumps({"latency_cycles": 13160, "dsp": 352}) + "\n", encoding="utf-8"
    )
    (cell / f"{bench}_selected.cpp").write_text(dse, encoding="utf-8")
    rewritten = (
        '#include <ap_int.h>\n'
        '#include <hls_stream.h>\n'
        "typedef ap_uint<128> vec4_bits;\n"
        'extern "C" void autosa_mm() {\n'
        "#pragma HLS DATAFLOW\n"
        "  hls::stream<vec4_bits> fifo_A[16];\n"
        "  compute_j: for (int j = 0; j < 64; ++j) {}\n"
        "}\n"
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
        "header_code": "#define I 64\n#define J 64\n#define K 64\n",
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
                "report": {
                    "latency_cycles": 5000,
                    "interval": 5000,
                    "dsp": 320,
                    "lut": 10,
                    "ff": 10,
                    "bram": 1,
                },
            },
            "csim": {"passed": True},
            "cosim": None,
        },
    )

    outcome = pfs.run_stream_for_cell(
        bench=bench,
        bench_dir=bench_dir,
        cell_dir=cell,
        orchestrator=Orch(),
        skip_existing=True,
    )
    assert outcome.success
    paths = pfs.artifact_paths(cell, bench)
    assert paths["kernel"].is_file()
    text = paths["kernel"].read_text(encoding="utf-8")
    assert "DATAFLOW" in text
    assert "hls::stream" in text
    result = json.loads(paths["result"].read_text(encoding="utf-8"))
    assert result["success"] is True
    assert result["dsp"] == 320
    assert result["promoted"] is True
    assert result["source_kernel_role"] == "dse"
    assert (cell / f"{bench}_selected.cpp").read_text(encoding="utf-8").strip() == rewritten.strip()
    assert (cell / f"{bench}_dse.cpp").read_text(encoding="utf-8") == dse


def test_stream_max_tokens_default_is_above_flash_empty_budget(monkeypatch):
    monkeypatch.delenv("C2HLS_STREAM_MAX_TOKENS", raising=False)
    monkeypatch.delenv("C2HLS_DSE_MAX_TOKENS", raising=False)
    monkeypatch.delenv("C2HLS_FLASH_MAX_TOKENS", raising=False)
    monkeypatch.delenv("C2HLS_LLM_MAX_TOKENS", raising=False)
    assert pfs.stream_max_tokens() >= 16384
    monkeypatch.setenv("C2HLS_STREAM_MAX_TOKENS", "4096")
    assert pfs.stream_max_tokens() == 8192
    monkeypatch.setenv("C2HLS_STREAM_MAX_TOKENS", "65536")
    assert pfs.stream_max_tokens() == 65536


def test_stream_64_cube_keeps_authored_trips():
    header = "#define I 64\n#define J 64\n#define K 64\n"
    rec, size = pfs.stream_recipe_for_header("autosa_mm", header)
    assert size == (64, 64, 64)
    assert rec.pe == 16
    assert rec.simd == 4
    assert rec.pe_kj == 1024
    assert rec.i_tiles == 1
    assert pfs.problem_size_override_block(64, 64, 64, rec) == ""
    plain = pfs.format_recipe_prompt("autosa_mm", step="stream")
    sized = pfs.format_recipe_prompt("autosa_mm", step="stream", i=64, j=64, k=64)
    assert plain == sized
    block, _meta = pfs.build_stream_skills_prompt_block()
    assert pfs.specialize_skill_block_for_problem(block, rec, 64, 64, 64) == block


def test_problem_size_comes_from_header_not_constant_64():
    header = (
        "//#define I 64\n//#define J 64\n//#define K 64\n"
        "#ifdef __cplusplus\n"
        "#define I 1024\n#define J 1024\n#define K 1024\n"
        "#endif\n"
    )
    rec, size = pfs.stream_recipe_for_header("autosa_mm", header, {"ijk": 64})
    assert size == (1024, 1024, 1024)
    assert rec.pe_kj == (1024 // 4) * 1024
    assert rec.i_tiles == 1024 // 16
    assert rec.pe_kj != 1024
    prompt = pfs.format_recipe_prompt(
        "autosa_mm", step="stream", i=size[0], j=size[1], k=size[2]
    )
    assert "Problem size is I=1024 J=1024 K=1024 from kernel.h" in prompt
    assert "Do not assume a 64^3 matrix" in prompt
    assert "pe_kj trip = (K/SIMD)*J = 262144" in prompt
    assert "i_tiles = I/PE_NUM = 64" in prompt
    assert "pe_kj trip = (K/SIMD)*J = 1024" not in prompt
    override = pfs.problem_size_override_block(*size, rec)
    assert "64^3 illustration" in override
    assert "262144" in override
    miss = pfs.architecture_miss_message("nope", {"dsp": 1}, "autosa_mm", rec)
    assert "262144" in miss
    assert "of 1024 beats" not in miss


def test_n1024_bench_header_drives_stream_size():
    header_path = (
        Path(__file__).resolve().parents[1]
        / "artifacts/pc2/autosa_mm_ijk_benches/n1024/autosa_mm/kernel.h"
    )
    header = header_path.read_text(encoding="utf-8")
    meta = {"ijk": 1024, "csim_timeout_s": 86400, "synth_timeout_s": 86400}
    rec, size = pfs.stream_recipe_for_header("autosa_mm", header, meta)
    assert size == (1024, 1024, 1024)
    assert rec.pe_kj == 262144
    block, _meta = pfs.build_stream_skills_prompt_block()
    specialized = pfs.specialize_skill_block_for_problem(block, rec, *size)
    assert "J=64" not in specialized
    assert "depth=64" not in specialized
    assert "262144" in specialized
    assert "1024" in block


def test_resolve_stream_bench_dir_prefers_ready_root(tmp_path, monkeypatch):
    bench = tmp_path / "autosa_mm"
    bench.mkdir()
    (bench / "metadata.json").write_text(
        '{"benchmark": "autosa_mm", "ijk": 1024}\n', encoding="utf-8"
    )
    (bench / "kernel.h").write_text(
        "#define I 1024\n#define J 1024\n#define K 1024\n", encoding="utf-8"
    )
    monkeypatch.setenv("C2HLS_AUTOSA_READY_ROOT", str(tmp_path))
    found = pfs.resolve_stream_bench_dir("autosa_mm")
    assert found == bench
    assert pfs.problem_size_from_bench(
        (found / "kernel.h").read_text(encoding="utf-8"),
        {"ijk": 64},
    ) == (1024, 1024, 1024)


def test_default_stream_bench_stays_live_64(monkeypatch):
    monkeypatch.delenv("C2HLS_AUTOSA_READY_ROOT", raising=False)
    found = pfs.resolve_stream_bench_dir("autosa_mm")
    text = (found / "kernel.h").read_text(encoding="utf-8")
    assert "#define I 64" in text
    assert "#define I 1024" not in text


def test_stream_timeouts_follow_bench_metadata(monkeypatch):
    import os

    import hls_eval

    prev = hls_eval.CSIM_TIMEOUT
    monkeypatch.setenv("C2HLS_CSIM_TIMEOUT", "180")
    monkeypatch.setenv("C2HLS_SYNTH_TIMEOUT", "1200")
    hls_eval.CSIM_TIMEOUT = 180
    try:
        pfs.apply_stream_bench_timeouts(
            {"csim_timeout_s": 86400, "synth_timeout_s": 86400}
        )
        assert hls_eval.CSIM_TIMEOUT == 86400
        assert os.environ["C2HLS_SYNTH_TIMEOUT"] == "86400"
        assert os.environ["C2HLS_CSIM_TIMEOUT"] == "86400"
    finally:
        hls_eval.CSIM_TIMEOUT = prev


def test_problem_size_metadata_ijk_when_header_missing():
    assert pfs.problem_size_from_bench("", {"ijk": 1024}) == (1024, 1024, 1024)


def test_n1024_seed_pe16_simd32_not_locked_16x4(monkeypatch):
    """Seed PE 16 × SIMD 32 on the n1024 header must not keep the 64³ 16×4 trips."""
    monkeypatch.delenv("C2HLS_PE_RECIPE", raising=False)
    monkeypatch.delenv("C2HLS_STREAM_PE", raising=False)
    monkeypatch.delenv("C2HLS_STREAM_SIMD", raising=False)
    header_path = (
        Path(__file__).resolve().parents[1]
        / "artifacts/pc2/autosa_mm_ijk_benches/n1024/autosa_mm/kernel.h"
    )
    header = header_path.read_text(encoding="utf-8")
    seed = "const int PE = 16;\nconst int SIMD = 32;\n"
    rec, size = pfs.stream_recipe_for_header("autosa_mm", header, seed_code=seed)
    assert size == (1024, 1024, 1024)
    assert rec.pe == 16
    assert rec.simd == 32
    assert rec.pe_kj == (1024 // 32) * 1024
    assert rec.pe_kj == 32768
    assert rec.i_tiles == 1024 // 16
    assert rec.i_tiles == 64
    prompt = pfs.format_recipe_prompt(
        "autosa_mm",
        step="stream",
        i=size[0],
        j=size[1],
        k=size[2],
        recipe=rec,
    )
    override = pfs.problem_size_override_block(*size, rec)
    block, _meta = pfs.build_stream_skills_prompt_block()
    specialized = pfs.specialize_skill_block_for_problem(block, rec, *size)
    system = pfs.specialize_skill_block_for_problem(pfs._SYSTEM, rec, *size)
    full = system + prompt + override + specialized
    assert "\n- PE_NUM=16\n" in prompt
    assert "\n- SIMD=32\n" in prompt
    assert "\n- SIMD=4\n" not in prompt
    assert "PE=16 SIMD=4" not in prompt
    assert "pe_kj trip = (K/SIMD)*J = 32768" in prompt
    assert "i_tiles = I/PE_NUM = 64" in prompt
    assert "pe_kj=1024" not in full
    assert "pe_kj trip = (K/SIMD)*J = 1024" not in full
    assert "#define SIMD 4" not in full
    assert "#define SIMD 32" in specialized
    assert "J=64" not in full
    assert "depth=64" not in full
    assert "32768" in full


def test_n1024_onchip_tile_seed_does_not_replay_full_b(monkeypatch):
    """TI=TJ=TK=64 seed must stream one 64×64 B tile, not (I/PE)×(K/SIMD)×J."""
    monkeypatch.delenv("C2HLS_PE_RECIPE", raising=False)
    monkeypatch.delenv("C2HLS_STREAM_PE", raising=False)
    monkeypatch.delenv("C2HLS_STREAM_SIMD", raising=False)
    monkeypatch.delenv("C2HLS_STREAM_TI", raising=False)
    monkeypatch.delenv("C2HLS_STREAM_TJ", raising=False)
    monkeypatch.delenv("C2HLS_STREAM_TK", raising=False)
    root = Path(__file__).resolve().parents[1]
    header = (
        root / "artifacts/pc2/autosa_mm_ijk_benches/n1024/autosa_mm/kernel.h"
    ).read_text(encoding="utf-8")
    seed = (
        root
        / "artifacts/pc2/autosa_mm_ijk_sweep_20260923/n1024/cells"
        / "dse2_90_f1_r09_camp/variants/autosa_aav_n_90/autosa_mm"
        / "deepseek-v4-flash__flash__autosa__aav_n_90/dse_v2/pe16_simd32/autosa_mm.cpp"
    ).read_text(encoding="utf-8")
    rec, size = pfs.stream_recipe_for_header("autosa_mm", header, seed_code=seed)
    assert size == (1024, 1024, 1024)
    assert rec.pe == 16
    assert rec.simd == 32
    assert rec.ti == 64
    assert rec.tj == 64
    assert rec.tk == 64
    assert rec.load_trip == 64 * 64
    assert rec.compute_trip == (64 // 32) * 64
    assert rec.store_trip == 16 * 64
    assert rec.outer_tiles == (1024 // 64) ** 2
    prompt = pfs.format_recipe_prompt(
        "autosa_mm",
        step="stream",
        i=size[0],
        j=size[1],
        k=size[2],
        recipe=rec,
    )
    override = pfs.problem_size_override_block(*size, rec)
    block, _meta = pfs.build_stream_skills_prompt_block()
    specialized = pfs.specialize_skill_block_for_problem(block, rec, *size)
    system = pfs.specialize_skill_block_for_problem(pfs._SYSTEM, rec, *size)
    repair = pfs.specialize_skill_block_for_problem(
        pfs._REPAIR_USER.format(
            recipe_block=prompt + override,
            stage="architecture",
            error="dry-run",
            header_name="kernel.h",
            header_code="",
            kernel_code="",
        ),
        rec,
        *size,
    )
    full = system + prompt + override + specialized + repair
    assert "TI=64" in prompt
    assert "TJ=64" in prompt
    assert "TK=64" in prompt
    assert "4096" in prompt
    assert "pe_kj trip = (K/SIMD)*J = 32768" not in full
    assert pfs.prompt_instructs_full_matrix_b_replay(full) is False


def test_named_pe_recipe_ignores_seed_tiles(monkeypatch):
    monkeypatch.setenv("C2HLS_PE_RECIPE", "autosa_mm_32x8")
    monkeypatch.delenv("C2HLS_STREAM_PE", raising=False)
    monkeypatch.delenv("C2HLS_STREAM_SIMD", raising=False)
    monkeypatch.delenv("C2HLS_STREAM_TI", raising=False)
    monkeypatch.delenv("C2HLS_STREAM_TJ", raising=False)
    monkeypatch.delenv("C2HLS_STREAM_TK", raising=False)
    header = "#define I 1024\n#define J 1024\n#define K 1024\n"
    seed = (
        "const int PE = 16;\nconst int SIMD = 32;\n"
        "const int TI = 64;\nconst int TJ = 64;\nconst int TK = 64;\n"
    )
    rec, size = pfs.stream_recipe_for_header("autosa_mm", header, seed_code=seed)
    assert size == (1024, 1024, 1024)
    assert rec.pe == 32
    assert rec.simd == 8
    assert rec.ti == 0
    assert rec.tj == 0
    assert rec.tk == 0
    assert rec.pe_kj == (1024 // 8) * 1024


def test_stream_tile_env_overrides_seed(monkeypatch):
    monkeypatch.delenv("C2HLS_PE_RECIPE", raising=False)
    monkeypatch.delenv("C2HLS_STREAM_PE", raising=False)
    monkeypatch.delenv("C2HLS_STREAM_SIMD", raising=False)
    monkeypatch.setenv("C2HLS_STREAM_TI", "32")
    monkeypatch.setenv("C2HLS_STREAM_TJ", "32")
    monkeypatch.setenv("C2HLS_STREAM_TK", "32")
    header = "#define I 1024\n#define J 1024\n#define K 1024\n"
    seed = (
        "const int PE = 16;\nconst int SIMD = 32;\n"
        "const int TI = 64;\nconst int TJ = 64;\nconst int TK = 64;\n"
    )
    rec, _size = pfs.stream_recipe_for_header("autosa_mm", header, seed_code=seed)
    assert rec.pe == 16
    assert rec.simd == 32
    assert rec.ti == 32
    assert rec.tj == 32
    assert rec.tk == 32
    assert rec.load_trip == 32 * 32


def test_stream_numeric_env_overrides_seed_pe_simd(monkeypatch):
    monkeypatch.delenv("C2HLS_PE_RECIPE", raising=False)
    monkeypatch.setenv("C2HLS_STREAM_PE", "32")
    monkeypatch.setenv("C2HLS_STREAM_SIMD", "8")
    header = "#define I 1024\n#define J 1024\n#define K 1024\n"
    seed = "const int PE = 16;\nconst int SIMD = 32;\n"
    rec, size = pfs.stream_recipe_for_header(
        "autosa_mm", header, seed_code=seed
    )
    assert size == (1024, 1024, 1024)
    assert rec.pe == 32
    assert rec.simd == 8
    assert rec.pe_kj == (1024 // 8) * 1024
    assert rec.i_tiles == 1024 // 32


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
