from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
FIX = REPO / "tests" / "fixtures" / "rag_scrape_mini"

from c2hls_rag import (  # noqa: E402
    rag_config_from_env,
    resolve_scrape_corpus,
)
from c2hls_rag_scrape import (  # noqa: E402
    extract_text_cached,
    format_scrape_block,
    parse_keywords_json,
    prepare_scrape_block,
    scrape_keywords,
)


def test_parse_keywords_json_fenced():
    raw = 'Here:\n```json\n{"keywords": ["DATAFLOW", "HLS 200-979"]}\n```\n'
    assert parse_keywords_json(raw) == ["DATAFLOW", "HLS 200-979"]


def test_parse_keywords_json_raw_object():
    assert parse_keywords_json('{"keywords": ["PIPELINE"]}') == ["PIPELINE"]


def test_parse_keywords_json_invalid_returns_empty():
    assert parse_keywords_json("no json here") == []


def test_scrape_keywords_finds_hits():
    block = scrape_keywords(
        ["DATAFLOW", "200-979"],
        corpus_paths=[FIX / "doc_a.txt", FIX / "doc_b.txt"],
        max_hits_per_keyword=2,
        max_total_chars=4000,
        context_chars=80,
    )
    assert "DATAFLOW" in block
    assert "Retrieved HLS documentation" in block or "Scraped" in block
    assert "PIPELINE" not in block or "DATAFLOW" in block  # at least dataflow hit


def test_scrape_respects_max_keywords():
    kws = [f"kw{i}" for i in range(20)]
    # should not throw; truncates internally
    scrape_keywords(kws, corpus_paths=[FIX / "doc_b.txt"], max_keywords=5)


def test_extract_text_cached_txt(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text("hello PIPELINE", encoding="utf-8")
    cache = tmp_path / "cache"
    t1 = extract_text_cached(p, cache_dir=cache)
    t2 = extract_text_cached(p, cache_dir=cache)
    assert "PIPELINE" in t1 and t1 == t2


def test_prepare_scrape_block_with_fake_llm(tmp_path):
    def fake_llm(messages):
        return '{"keywords": ["DATAFLOW"]}'

    block, kws = prepare_scrape_block(
        llm_call=fake_llm,
        analysis_kind="repair",
        code="void f(){}",
        errors="ERROR: [HLS 200-979] DATAFLOW",
        corpus_paths=[FIX / "doc_a.txt"],
        cache_dir=tmp_path / "cache",
    )
    assert kws == ["DATAFLOW"]
    assert "DATAFLOW" in block


def test_rag_config_scrape_env(monkeypatch, tmp_path):
    monkeypatch.setenv("C2HLS_RAG", "1")
    monkeypatch.setenv("C2HLS_RAG_SCRAPE", "1")
    p = tmp_path / "d.txt"
    p.write_text("x", encoding="utf-8")
    monkeypatch.setenv("C2HLS_RAG_SCRAPE_CORPUS", str(p))
    cfg = rag_config_from_env()
    assert cfg.scrape_enabled
    assert cfg.scrape_corpus_paths[0] == p.resolve()


def test_resolve_scrape_corpus_skips_missing(tmp_path):
    existing = tmp_path / "ok.txt"
    existing.write_text("x", encoding="utf-8")
    missing = tmp_path / "missing.txt"
    paths = resolve_scrape_corpus(f"{existing}:{missing}")
    assert paths == (existing.resolve(),)


def test_resolve_scrape_corpus_expands_directory(tmp_path):
    d = tmp_path / "docs"
    d.mkdir()
    a = d / "a.pdf"
    b = d / "b.txt"
    (d / "skip.bin").write_bytes(b"x")
    a.write_bytes(b"%PDF")
    b.write_text("hi", encoding="utf-8")
    paths = resolve_scrape_corpus(str(d))
    assert paths == (a.resolve(), b.resolve())


def test_scrape_cli_flags_in_c2hls_source():
    src = (REPO / "c2hls.py").read_text(encoding="utf-8")
    assert '"--scrape"' in src
    assert '"--scrape-corpus"' in src
    assert 'parser.error("--scrape requires --rag")' in src
    assert 'os.environ["C2HLS_RAG_SCRAPE"] = "1"' in src
    assert "if not args.scrape and not args.rag2:" in src
    assert "get_index(cfg)" in src


def test_scrape_helpers_in_c2hls_source():
    src = (REPO / "c2hls.py").read_text(encoding="utf-8")
    for name in (
        "_scrape_docs_for_repair",
        "_scrape_docs_for_latency",
        "_prepend_scrape",
        "_scrape_enabled_for",
    ):
        assert f"def {name}" in src
