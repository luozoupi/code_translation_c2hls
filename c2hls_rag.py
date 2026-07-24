"""UG1399 RAG retrieval for c2hls (BM25, opt-in via env/CLI)."""

from __future__ import annotations

import json
import logging
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

REPO = Path(__file__).resolve().parent
DEFAULT_CORPUS = REPO / "artifacts" / "rag" / "ug1399"
DEFAULT_TOP_K = 4
CHUNK_SIZE = 1000
OVERLAP = 200

RAG_MODES = ("flashopt", "repair", "both", "everywhere")

BM25_K1 = 1.5
BM25_B = 0.75

_index_cache: dict[Path, "RagIndex"] = {}


@dataclass(frozen=True)
class RagConfig:
    enabled: bool
    mode: str | None
    corpus_dir: Path = DEFAULT_CORPUS
    top_k: int = DEFAULT_TOP_K
    scrape_enabled: bool = False
    scrape_corpus_paths: tuple[Path, ...] = ()


@dataclass
class RagIndex:
    meta: dict
    chunks: list[dict]
    avgdl: float
    doc_freqs: list[dict[str, int]]
    doc_lens: list[int]
    idf: dict[str, float]


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def chunk_text(
    text: str,
    *,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = OVERLAP,
) -> list[str]:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(f"overlap must satisfy 0 <= overlap < chunk_size, got {overlap}")

    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start += chunk_size - overlap
    return chunks


def _build_bm25(chunks: list[dict]) -> tuple[float, list[dict[str, int]], list[int], dict[str, float]]:
    n = len(chunks)
    doc_freqs: list[dict[str, int]] = []
    doc_lens: list[int] = []
    df: dict[str, int] = {}

    for chunk in chunks:
        tokens = tokenize(chunk["text"])
        doc_lens.append(len(tokens))
        tf: dict[str, int] = {}
        for tok in tokens:
            tf[tok] = tf.get(tok, 0) + 1
        doc_freqs.append(tf)
        for tok in tf:
            df[tok] = df.get(tok, 0) + 1

    avgdl = sum(doc_lens) / n if n else 0.0
    idf: dict[str, float] = {}
    for tok, freq in df.items():
        idf[tok] = math.log((n - freq + 0.5) / (freq + 0.5) + 1.0)

    return avgdl, doc_freqs, doc_lens, idf


def load_index(corpus_dir: Path | str) -> RagIndex:
    path = Path(corpus_dir).resolve()
    if path in _index_cache:
        return _index_cache[path]

    meta_path = path / "index_meta.json"
    chunks_path = path / "chunks.jsonl"
    build_hint = (
        "Build the index with: "
        "python3 scripts/build_ug1399_rag_index.py --source <UG1399> --out <dir>"
    )
    if not meta_path.is_file():
        raise FileNotFoundError(f"RAG index meta not found: {meta_path}. {build_hint}")
    if not chunks_path.is_file():
        raise FileNotFoundError(f"RAG chunks not found: {chunks_path}. {build_hint}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    chunks: list[dict] = []
    for line in chunks_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            chunks.append(json.loads(line))

    avgdl, doc_freqs, doc_lens, idf = _build_bm25(chunks)
    index = RagIndex(
        meta=meta,
        chunks=chunks,
        avgdl=avgdl,
        doc_freqs=doc_freqs,
        doc_lens=doc_lens,
        idf=idf,
    )
    _index_cache[path] = index
    return index


def _query_term_freqs(query: str) -> dict[str, int]:
    tf: dict[str, int] = {}
    for tok in tokenize(query):
        tf[tok] = tf.get(tok, 0) + 1
    return tf


def _bm25_score(index: RagIndex, doc_idx: int, query_tf: dict[str, int]) -> float:
    tf_map = index.doc_freqs[doc_idx]
    dl = index.doc_lens[doc_idx]
    score = 0.0
    for tok, qf in query_tf.items():
        if tok not in index.idf:
            continue
        freq = tf_map.get(tok, 0)
        if freq == 0:
            continue
        idf = index.idf[tok]
        denom = freq + BM25_K1 * (1.0 - BM25_B + BM25_B * dl / index.avgdl)
        score += idf * (freq * (BM25_K1 + 1.0)) / denom * qf
    return score


def _rank_chunks(index: RagIndex, query: str, top_k: int) -> list[dict]:
    query_tf = _query_term_freqs(query)
    if not query_tf or not index.chunks:
        return []

    scored = [
        (i, _bm25_score(index, i, query_tf))
        for i in range(len(index.chunks))
    ]
    scored.sort(key=lambda x: x[1], reverse=True)

    ranked: list[dict] = []
    seen_positive = False
    for i, score in scored:
        if seen_positive and score <= 0:
            continue
        if score > 0:
            seen_positive = True
        ranked.append(index.chunks[i])
        if len(ranked) >= top_k:
            break
    return ranked


def format_rag_block(chunks: list[dict]) -> str:
    if not chunks:
        return ""
    lines = ["## Retrieved HLS documentation (UG1399)"]
    for chunk in chunks:
        lines.append(f"### chunk {chunk['id']}")
        lines.append(chunk["text"])
    return "\n".join(lines)


def rank_chunks(index: RagIndex, query: str, *, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """Public BM25 ranking over index chunks (positive scores preferred)."""
    return _rank_chunks(index, query, top_k)


def retrieve(index: RagIndex, query: str, *, top_k: int = DEFAULT_TOP_K) -> str:
    ranked = rank_chunks(index, query, top_k=top_k)
    return format_rag_block(ranked)


def should_inject(mode: str | None, stage: str) -> bool:
    if mode is None:
        return False
    if mode == "flashopt":
        return stage == "flashopt"
    if mode == "repair":
        return stage == "repair"
    if mode == "both":
        return stage in ("flashopt", "repair")
    if mode == "everywhere":
        return stage in ("flashopt", "repair", "dataflow")
    return False


def _truthy_env(name: str) -> bool:
    val = os.environ.get(name, "")
    return val.strip().lower() in ("1", "true", "yes", "on")


_SCRAPE_SOURCE_SUFFIXES = {".pdf", ".html", ".htm", ".txt"}


def _expand_scrape_path(path: Path) -> list[Path]:
    """Resolve one path: a file, or a directory of PDF/HTML/TXT sources."""
    path = path.expanduser()
    if path.is_file():
        return [path.resolve()]
    if path.is_dir():
        found = sorted(
            p.resolve()
            for p in path.iterdir()
            if p.is_file() and p.suffix.lower() in _SCRAPE_SOURCE_SUFFIXES
        )
        if not found:
            logging.warning("RAG scrape corpus dir has no PDF/HTML/TXT: %s", path)
        return found
    logging.warning("RAG scrape corpus path not found (skipped): %s", path)
    return []


def resolve_scrape_corpus(raw: str | None) -> tuple[Path, ...]:
    """Split colon/comma-separated paths; keep existing files (dirs expand to docs)."""
    if not raw or not str(raw).strip():
        return ()
    existing: list[Path] = []
    seen: set[Path] = set()
    for part in re.split(r"[:,]", str(raw)):
        part = part.strip()
        if not part:
            continue
        for path in _expand_scrape_path(Path(part)):
            if path not in seen:
                seen.add(path)
                existing.append(path)
    return tuple(existing)


def rag_config_from_env(
    *,
    enabled: bool | None = None,
    mode: str | None = None,
    corpus_dir: Path | str | None = None,
    top_k: int | None = None,
    scrape_enabled: bool | None = None,
    scrape_corpus: str | None = None,
    scrape_corpus_paths: tuple[Path, ...] | None = None,
) -> RagConfig:
    env_enabled = _truthy_env("C2HLS_RAG")
    mode_raw = os.environ.get("C2HLS_RAG_MODE")
    env_mode = mode_raw.strip().lower() if mode_raw else None

    resolved_enabled = env_enabled if enabled is None else enabled
    resolved_mode = env_mode if mode is None else (mode.strip().lower() if mode else None)

    if resolved_mode and not resolved_enabled:
        raise ValueError("C2HLS_RAG_MODE set but C2HLS_RAG is not enabled")

    if resolved_enabled and resolved_mode is None:
        resolved_mode = "flashopt"
    elif resolved_enabled and resolved_mode not in RAG_MODES:
        invalid = mode_raw if mode is None else mode
        raise ValueError(f"invalid C2HLS_RAG_MODE: {invalid!r}")

    corpus_raw = os.environ.get("C2HLS_RAG_CORPUS")
    env_corpus = Path(corpus_raw) if corpus_raw else DEFAULT_CORPUS
    resolved_corpus = env_corpus if corpus_dir is None else Path(corpus_dir)

    top_k_raw = os.environ.get("C2HLS_RAG_TOP_K")
    env_top_k = int(top_k_raw) if top_k_raw else DEFAULT_TOP_K
    resolved_top_k = env_top_k if top_k is None else top_k

    env_scrape = _truthy_env("C2HLS_RAG_SCRAPE")
    resolved_scrape_enabled = env_scrape if scrape_enabled is None else scrape_enabled

    if scrape_corpus_paths is not None:
        resolved_scrape_paths = scrape_corpus_paths
    elif scrape_corpus is not None:
        resolved_scrape_paths = resolve_scrape_corpus(scrape_corpus)
    else:
        resolved_scrape_paths = resolve_scrape_corpus(
            os.environ.get("C2HLS_RAG_SCRAPE_CORPUS"),
        )

    return RagConfig(
        enabled=resolved_enabled,
        mode=resolved_mode,
        corpus_dir=resolved_corpus,
        top_k=resolved_top_k,
        scrape_enabled=resolved_scrape_enabled,
        scrape_corpus_paths=resolved_scrape_paths,
    )


def get_index(cfg: RagConfig) -> RagIndex:
    return load_index(cfg.corpus_dir)


def retrieve_for_stage(cfg: RagConfig, stage: str, query: str) -> str:
    if not cfg.enabled or not should_inject(cfg.mode, stage):
        return ""
    index = get_index(cfg)
    return retrieve(index, query, top_k=cfg.top_k)
