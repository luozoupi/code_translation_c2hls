from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from c2hls_rag import (  # noqa: E402
    chunk_text,
    load_index,
    rag_config_from_env,
    retrieve,
    retrieve_for_stage,
    should_inject,
)

FIX = REPO / "tests" / "fixtures" / "rag_ug1399_mini"


def test_chunk_text_overlap():
    text = "x" * 2500
    chunks = chunk_text(text, chunk_size=1000, overlap=200)
    assert len(chunks) >= 2
    assert all(len(c) <= 1000 for c in chunks)
    assert chunks[0][-200:] == chunks[1][:200]


def test_chunk_text_invalid_overlap_raises():
    with pytest.raises(ValueError, match="overlap"):
        chunk_text("abc", chunk_size=100, overlap=100)
    with pytest.raises(ValueError, match="chunk_size"):
        chunk_text("abc", chunk_size=0, overlap=0)


def test_should_inject_matrix():
    assert should_inject(None, "flashopt") is False
    assert should_inject("flashopt", "flashopt") is True
    assert should_inject("flashopt", "repair") is False
    assert should_inject("repair", "repair") is True
    assert should_inject("both", "flashopt") and should_inject("both", "repair")
    assert should_inject("everywhere", "dataflow") is True
    assert should_inject("flashopt", "dataflow") is False


def test_should_inject_everywhere_unknown_stage_false():
    assert should_inject("everywhere", "unknown") is False
    assert should_inject("everywhere", "synth") is False


def test_retrieve_prefers_relevant_chunk():
    idx = load_index(FIX)
    block = retrieve(idx, "DATAFLOW single producer consumer array", top_k=1)
    assert "DATAFLOW" in block
    assert "Retrieved HLS documentation" in block


def test_retrieve_skips_zero_score_when_top_k_large():
    idx = load_index(FIX)
    block = retrieve(idx, "DATAFLOW", top_k=4)
    assert "DATAFLOW" in block
    assert "PIPELINE" not in block


def test_rag_config_off_by_default(monkeypatch):
    monkeypatch.delenv("C2HLS_RAG", raising=False)
    monkeypatch.delenv("C2HLS_RAG_MODE", raising=False)
    cfg = rag_config_from_env()
    assert cfg.enabled is False
    assert cfg.mode is None


def test_rag_config_default_mode_flashopt(monkeypatch):
    monkeypatch.setenv("C2HLS_RAG", "1")
    monkeypatch.delenv("C2HLS_RAG_MODE", raising=False)
    cfg = rag_config_from_env()
    assert cfg.enabled is True
    assert cfg.mode == "flashopt"


def test_rag_mode_without_rag_raises(monkeypatch):
    monkeypatch.delenv("C2HLS_RAG", raising=False)
    monkeypatch.setenv("C2HLS_RAG_MODE", "flashopt")
    with pytest.raises(ValueError, match="not enabled"):
        rag_config_from_env()


def test_rag_invalid_mode_raises(monkeypatch):
    monkeypatch.setenv("C2HLS_RAG", "1")
    monkeypatch.setenv("C2HLS_RAG_MODE", "bogus")
    with pytest.raises(ValueError, match="invalid C2HLS_RAG_MODE"):
        rag_config_from_env()


def test_rag_config_kwargs_override(monkeypatch):
    monkeypatch.delenv("C2HLS_RAG", raising=False)
    monkeypatch.delenv("C2HLS_RAG_MODE", raising=False)
    cfg = rag_config_from_env(enabled=True, mode="repair")
    assert cfg.enabled is True
    assert cfg.mode == "repair"


def test_retrieve_for_stage_gating(monkeypatch):
    monkeypatch.setenv("C2HLS_RAG", "1")
    monkeypatch.setenv("C2HLS_RAG_MODE", "flashopt")
    monkeypatch.setenv("C2HLS_RAG_CORPUS", str(FIX))
    cfg = rag_config_from_env()
    query = "DATAFLOW single producer consumer"
    flash_block = retrieve_for_stage(cfg, "flashopt", query)
    repair_block = retrieve_for_stage(cfg, "repair", query)
    assert flash_block
    assert "DATAFLOW" in flash_block
    assert repair_block == ""


def test_flashopt_rag_injection_sites_in_c2hls_source():
    src = (REPO / "c2hls.py").read_text(encoding="utf-8")
    assert "def _rag_append" in src
    assert "_rag_append(" in src
    assert '"flashopt"' in src
    assert "curate_for_flash" in src
    assert "_optimization_step_initial_codegen" in src
    curate_idx = src.index("def curate_for_flash")
    opt_idx = src.index("def _optimization_step_initial_codegen")
    rag_in_curate = src.find("_rag_append(", curate_idx, opt_idx)
    rag_in_opt = src.find("_rag_append(", opt_idx)
    assert rag_in_curate != -1
    assert rag_in_opt != -1
    assert '"flashopt"' in src[rag_in_curate:rag_in_curate + 200]
    assert '"flashopt"' in src[rag_in_opt:rag_in_opt + 200]


def test_repair_rag_injection_sites_in_c2hls_source():
    src = (REPO / "c2hls.py").read_text(encoding="utf-8")
    assert src.count("_rag_append(") >= 6  # flashopt + repair sites
    assert '"repair"' in src
    assert "_scrape_enabled_for" in src
    quality_idx = src.index("hls_quality_repair.format")
    synth_fix_idx = src.index("hls_synthesis_fix.format")
    quality_region = src[quality_idx:quality_idx + 1200]
    assert "_rag_append(" in quality_region
    assert '"repair"' in quality_region
    assert "prompt" in quality_region
    synth_region = src[synth_fix_idx:synth_fix_idx + 4000]
    assert "_rag_append(" in synth_region
    assert '"repair"' in synth_region
    assert "fix_prompt" in synth_region

    opt_single_idx = src.index("def _optimization_step_attempt_single")
    opt_single_end = src.index("\n    def ", opt_single_idx + 1)
    opt_single_region = src[opt_single_idx:opt_single_end]
    assert '_rag_append(' in opt_single_region
    assert '"repair"' in opt_single_region
    assert opt_single_region.count('_rag_append(') >= 3

    pipelined_repair_idx = src.index("def _pipelined_opt_step_repair_codegen")
    pipelined_repair_end = src.index("\n    def ", pipelined_repair_idx + 1)
    pipelined_repair_region = src[pipelined_repair_idx:pipelined_repair_end]
    assert '_rag_append(' in pipelined_repair_region
    assert '"repair"' in pipelined_repair_region


def test_repair_mode_not_flashopt(monkeypatch):
    monkeypatch.setenv("C2HLS_RAG", "1")
    monkeypatch.setenv("C2HLS_RAG_MODE", "repair")
    monkeypatch.setenv("C2HLS_RAG_CORPUS", str(FIX))
    cfg = rag_config_from_env()
    assert retrieve_for_stage(cfg, "repair", "PIPELINE II")
    assert retrieve_for_stage(cfg, "flashopt", "PIPELINE") == ""


def test_rag_append_appends_when_enabled(monkeypatch):
    monkeypatch.setenv("C2HLS_RAG", "1")
    monkeypatch.setenv("C2HLS_RAG_MODE", "flashopt")
    monkeypatch.setenv("C2HLS_RAG_CORPUS", str(FIX))
    try:
        from c2hls import _rag_append
    except ImportError:
        pytest.skip("c2hls import failed (e.g. missing openai)")

    out = _rag_append("base prompt", "flashopt", "DATAFLOW single producer consumer")
    assert out.startswith("base prompt")
    assert "Retrieved HLS documentation" in out
    assert "DATAFLOW" in out


def test_cli_rag_wiring_in_c2hls_source():
    src = (REPO / "c2hls.py").read_text(encoding="utf-8")
    for flag in ('"--rag"', '"--rag-mode"', '"--rag-corpus"', '"--rag-top-k"', '"--scrape"', '"--scrape-corpus"', '"--rag2"'):
        assert flag in src
    assert 'parser.error("--rag-mode requires --rag or --rag2")' in src
    assert 'parser.error("--scrape requires --rag")' in src
    assert 'parser.error("--rag2 is incompatible with --scrape")' in src
    assert 'os.environ["C2HLS_RAG"] = "1"' in src
    assert 'os.environ["C2HLS_RAG2"] = "1"' in src
    assert "rag_config_from_env(" in src
    assert "if not args.scrape and not args.rag2:" in src
    assert "get_index(cfg)" in src


def test_cli_rag_mode_requires_rag_flag():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--rag", action="store_true")
    parser.add_argument(
        "--rag-mode",
        choices=["flashopt", "repair", "both", "everywhere"],
        default=None,
    )
    args = parser.parse_args(["--rag-mode", "repair"])
    with pytest.raises(SystemExit):
        if args.rag_mode and not args.rag:
            parser.error("--rag-mode requires --rag")


def test_everywhere_injects_dataflow(monkeypatch):
    monkeypatch.setenv("C2HLS_RAG", "1")
    monkeypatch.setenv("C2HLS_RAG_MODE", "everywhere")
    monkeypatch.setenv("C2HLS_RAG_CORPUS", str(FIX))
    cfg = rag_config_from_env()
    block = retrieve_for_stage(cfg, "dataflow", "m_axi bundle gmem")
    assert block
    assert "Retrieved HLS documentation" in block


def test_dataflow_format_appends_rag(monkeypatch):
    monkeypatch.setenv("C2HLS_RAG", "1")
    monkeypatch.setenv("C2HLS_RAG_MODE", "everywhere")
    monkeypatch.setenv("C2HLS_RAG_CORPUS", str(FIX))
    import post_flash_dataflow as pfd

    bundle = pfd.compose_dataflow_prompts("system_skills")
    out = pfd.format_dataflow_initial_user(
        bundle,
        benchmark_context="ctx",
        header_name="kernel.h",
        header_code="/* hdr */",
        kernel_code="void k() { m_axi bundle gmem; }",
    )
    assert "Retrieved HLS documentation" in out
    assert "m_axi" in out.lower() or "INTERFACE" in out

    repair = pfd.format_dataflow_repair_user(
        bundle,
        stage="csynth",
        error="HLS 200-1013 m_axi bundle gmem conflict",
        benchmark_context="ctx",
        header_name="kernel.h",
        header_code="/* hdr */",
        kernel_code="void k() {}",
    )
    assert "Retrieved HLS documentation" in repair


def test_dataflow_rag_wiring_in_source():
    pfd_src = (REPO / "post_flash_dataflow.py").read_text(encoding="utf-8")
    assert "def _append_dataflow_rag" in pfd_src
    assert '"dataflow"' in pfd_src
    assert "_append_dataflow_rag(" in pfd_src
    assert "_prepare_dataflow_scrape" in pfd_src
    assert "_scrape_enabled_for_dataflow" in pfd_src

    runner_src = (REPO / "scripts/pc2/run_post_flash_dataflow.py").read_text(encoding="utf-8")
    for flag in ('"--rag"', '"--rag-mode"', '"--rag-corpus"', '"--rag-top-k"', '"--scrape"', '"--scrape-corpus"', '"--rag2"'):
        assert flag in runner_src
    assert 'parser.error("--rag-mode requires --rag or --rag2")' in runner_src
    assert 'parser.error("--scrape requires --rag")' in runner_src
    assert 'os.environ["C2HLS_RAG"] = "1"' in runner_src
    assert "if not args.scrape and not args.rag2:" in runner_src


def test_rag_append_skips_bm25_when_scrape_enabled(monkeypatch):
    monkeypatch.setenv("C2HLS_RAG", "1")
    monkeypatch.setenv("C2HLS_RAG_MODE", "flashopt")
    monkeypatch.setenv("C2HLS_RAG_SCRAPE", "1")
    monkeypatch.setenv("C2HLS_RAG_CORPUS", str(FIX))
    try:
        from c2hls import _rag_append
    except ImportError:
        pytest.skip("c2hls import failed (e.g. missing openai)")

    out = _rag_append("base prompt", "flashopt", "DATAFLOW single producer consumer")
    assert out == "base prompt"


def test_build_ug1399_rag_index_smoke(tmp_path: Path):
    source = tmp_path / "ug_mini.txt"
    source.write_text(
        "PIPELINE pragma II=1. DATAFLOW single reader writer. ",
        encoding="utf-8",
    )
    out = tmp_path / "rag_out"
    script = REPO / "scripts" / "build_ug1399_rag_index.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--source", str(source), "--out", str(out)],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert (out / "chunks.jsonl").is_file()
    assert (out / "index_meta.json").is_file()

    idx = load_index(out)
    block = retrieve(idx, "DATAFLOW", top_k=1)
    assert "DATAFLOW" in block
