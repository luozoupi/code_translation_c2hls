"""Tests for c2hls RAG2 dual-policy structured retrieval."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures"
OPT_DIR = FIXTURE_ROOT / "rag2_opt"
REPAIR_DIR = FIXTURE_ROOT / "rag2_repair"


def test_extract_hls_error_ids():
    from c2hls_rag2 import extract_hls_error_ids

    text = "ERROR: [HLS 200-880] II Violation\nWARNING: [HLS 207-5570] unknown"
    ids = extract_hls_error_ids(text)
    assert "HLS 200-880" in ids
    assert "HLS 207-5570" in ids


def test_extract_bottleneck_tags():
    from c2hls_rag2 import extract_bottleneck_tags

    report = "Loop L1: Target II=1, Final II=3 due to carried dependence"
    code = "#pragma HLS PIPELINE II=1\n#pragma HLS ARRAY_PARTITION variable=A complete"
    tags = extract_bottleneck_tags(report, code)
    assert "ii" in tags or "initiation" in " ".join(tags)
    assert any("pipeline" in t for t in tags)
    assert any("partition" in t or "array_partition" in t for t in tags)


def test_policy_for_stage():
    from c2hls_rag2 import policy_for_stage

    assert policy_for_stage("flashopt") == "opt"
    assert policy_for_stage("dataflow") == "opt"
    assert policy_for_stage("repair") == "repair"
    assert policy_for_stage("dataflow_repair") == "repair"


def test_build_repair_query_includes_error_ids():
    from c2hls_rag2 import build_repair_query

    q = build_repair_query(
        code="#pragma HLS PIPELINE II=1\nvoid k(){}",
        error="ERROR: [HLS 200-880] II Violation carried dependence",
    )
    assert "HLS 200-880" in q
    assert q.count("HLS 200-880") >= 2  # boost


def test_rag2_config_rejects_scrape():
    from c2hls_rag2 import rag2_config_from_env

    os.environ["C2HLS_RAG2"] = "1"
    os.environ["C2HLS_RAG_SCRAPE"] = "1"
    try:
        with pytest.raises(ValueError, match="scrape"):
            rag2_config_from_env()
    finally:
        os.environ.pop("C2HLS_RAG2", None)
        os.environ.pop("C2HLS_RAG_SCRAPE", None)


def test_format_rag2_block_and_truncate():
    from c2hls_rag2 import format_rag2_block

    chunks = [
        {"id": "c0", "text": "AAA " * 50, "source": "ug1399.pdf"},
        {"id": "c1", "text": "BBB " * 50, "source": "ug902.pdf"},
    ]
    block = format_rag2_block("opt", chunks, max_chars=120)
    assert block.startswith("## RAG2 (opt)")
    assert "chunk c0" in block
    assert len(block) <= 120 + 20  # small slack for headers if implementation soft-caps


def test_retrieve_rag2_opt_and_repair(monkeypatch):
    from c2hls_rag2 import Rag2Config, retrieve_rag2

    cfg = Rag2Config(
        enabled=True,
        mode="everywhere",
        opt_corpus_dir=OPT_DIR,
        repair_corpus_dir=REPAIR_DIR,
        top_k=2,
        max_chars=4000,
    )
    opt_block = retrieve_rag2(
        cfg,
        policy="opt",
        query="pipeline initiation interval PIPELINE",
    )
    assert "## RAG2 (opt)" in opt_block
    assert "PIPELINE" in opt_block or "pipelining" in opt_block.lower()

    repair_block = retrieve_rag2(
        cfg,
        policy="repair",
        query="HLS 200-880 HLS 200-880 II Violation dependence",
    )
    assert "## RAG2 (repair)" in repair_block
    assert "200-880" in repair_block or "dependence" in repair_block.lower()


def test_hybrid_fallback_when_deterministic_empty():
    from c2hls_rag2 import Rag2Config, retrieve_rag2

    cfg = Rag2Config(
        enabled=True,
        mode="everywhere",
        opt_corpus_dir=OPT_DIR,
        repair_corpus_dir=REPAIR_DIR,
        top_k=2,
        max_chars=4000,
    )

    def llm_call(messages):
        return '{"keywords": ["DATAFLOW", "canonical"]}'

    block = retrieve_rag2(
        cfg,
        policy="opt",
        query="",  # empty → fallback
        llm_call=llm_call,
        analysis_kind="latency",
    )
    assert "## RAG2 (opt)" in block
    assert "DATAFLOW" in block or "dataflow" in block.lower()
