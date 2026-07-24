"""RAG2: dual-policy structured BM25 retrieval for HLS opt vs repair."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

from c2hls_rag import (
    DEFAULT_TOP_K,
    RAG_MODES,
    load_index,
    rank_chunks,
    should_inject,
    tokenize,
    _truthy_env,
)

REPO = Path(__file__).resolve().parent
DEFAULT_OPT_CORPUS = REPO / "artifacts" / "rag" / "rag2_opt"
DEFAULT_REPAIR_CORPUS = REPO / "artifacts" / "rag" / "rag2_repair"
DEFAULT_MAX_CHARS = 4000

Policy = Literal["opt", "repair"]

_HLS_ID_RE = re.compile(
    r"(?:\[)?HLS\s+(\d+-\d+)(?:\])?",
    re.IGNORECASE,
)

_BOTTLENECK_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("ii", re.compile(r"\bii\b|initiation\s+interval", re.I)),
    ("pipeline", re.compile(r"pipeline|pipelin", re.I)),
    ("dependence", re.compile(r"dependenc|carried\s+depend", re.I)),
    ("partition", re.compile(r"array_partition|partition", re.I)),
    ("dataflow", re.compile(r"dataflow", re.I)),
    ("burst", re.compile(r"\bburst\b", re.I)),
    ("m_axi", re.compile(r"m_axi|axi", re.I)),
    ("unroll", re.compile(r"\bunroll\b", re.I)),
    ("bound", re.compile(r"\bbound\b|loop_tripcount", re.I)),
]

_PRAGMA_RE = re.compile(
    r"#\s*pragma\s+HLS\s+(\w+)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Rag2Config:
    enabled: bool
    mode: str | None
    opt_corpus_dir: Path = DEFAULT_OPT_CORPUS
    repair_corpus_dir: Path = DEFAULT_REPAIR_CORPUS
    top_k: int = DEFAULT_TOP_K
    max_chars: int = DEFAULT_MAX_CHARS


def policy_for_stage(stage: str) -> Policy:
    if stage in ("repair", "dataflow_repair"):
        return "repair"
    return "opt"


def extract_hls_error_ids(text: str) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for match in _HLS_ID_RE.finditer(text or ""):
        eid = f"HLS {match.group(1)}"
        key = eid.upper()
        if key not in seen:
            seen.add(key)
            found.append(eid)
    return found


def extract_bottleneck_tags(latency_report: str, code: str) -> list[str]:
    blob = f"{latency_report or ''}\n{code or ''}"
    tags: list[str] = []
    for name, pattern in _BOTTLENECK_PATTERNS:
        if pattern.search(blob):
            tags.append(name)
    return tags


def _pragma_names(code: str) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for match in _PRAGMA_RE.finditer(code or ""):
        name = match.group(1).upper()
        if name not in seen:
            seen.add(name)
            names.append(name)
    return names


def build_opt_query(*, code: str, latency_report: str = "") -> str:
    tags = extract_bottleneck_tags(latency_report, code)
    pragmas = _pragma_names(code)
    head = (code or "")[:800]
    parts = tags + pragmas
    if head.strip():
        parts.append(head)
    return " ".join(parts).strip()


def build_repair_query(*, code: str, error: str = "") -> str:
    ids = extract_hls_error_ids(error)
    # Lexical boost: repeat each ID.
    boosted = []
    for eid in ids:
        boosted.extend([eid, eid])
    pragmas = _pragma_names(f"{error}\n{code}")
    err_head = (error or "")[:1500]
    code_head = (code or "")[:800]
    parts = boosted + pragmas
    if err_head.strip():
        parts.append(err_head)
    if code_head.strip():
        parts.append(code_head)
    return " ".join(parts).strip()


def format_rag2_block(
    policy: Policy,
    chunks: list[dict],
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> str:
    if not chunks:
        return ""
    lines = [f"## RAG2 ({policy})", ""]
    body_parts: list[str] = []
    total = len("\n".join(lines))
    for chunk in chunks:
        cid = chunk.get("id", "?")
        source = chunk.get("source", "")
        text = chunk.get("text", "")
        header = f"### chunk {cid}" + (f" ({source})" if source else "")
        piece = f"{header}\n{text}"
        # Soft-cap: stop adding chunks once we'd exceed max_chars.
        candidate_len = total + len(piece) + 2
        if body_parts and candidate_len > max_chars:
            break
        body_parts.append(piece)
        total = candidate_len
        if total >= max_chars:
            break
    block = "\n".join(lines + body_parts).rstrip()
    if len(block) > max_chars:
        block = block[:max_chars].rstrip()
    return block


def _corpus_for_policy(cfg: Rag2Config, policy: Policy) -> Path:
    return cfg.opt_corpus_dir if policy == "opt" else cfg.repair_corpus_dir


def _query_has_tokens(query: str) -> bool:
    return bool(tokenize(query or ""))


def _llm_keyword_query(
    llm_call: Callable,
    *,
    analysis_kind: str,
    code: str,
    errors: str = "",
    latency_report: str = "",
) -> str:
    from c2hls_rag_scrape import (
        KEYWORD_ANALYSIS_LATENCY,
        KEYWORD_ANALYSIS_REPAIR,
        parse_keywords_json,
    )

    system = (
        KEYWORD_ANALYSIS_REPAIR
        if analysis_kind == "repair"
        else KEYWORD_ANALYSIS_LATENCY
    )
    user_parts = [f"## HLS code\n{(code or '')[:8000]}"]
    if analysis_kind == "repair":
        user_parts.append(f"## Errors\n{(errors or '')[:4000]}")
    else:
        user_parts.append(f"## Latency report\n{(latency_report or '')[:4000]}")
    try:
        raw = llm_call(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": "\n\n".join(user_parts)},
            ]
        )
    except Exception as exc:  # pragma: no cover
        logging.warning("RAG2 keyword fallback LLM failed: %s", exc)
        return ""
    kws = parse_keywords_json(raw if isinstance(raw, str) else str(raw))
    return " ".join(kws).strip()


def retrieve_rag2(
    cfg: Rag2Config,
    *,
    policy: Policy,
    query: str,
    llm_call: Callable | None = None,
    analysis_kind: str | None = None,
    code: str = "",
    errors: str = "",
    latency_report: str = "",
) -> str:
    if not cfg.enabled:
        return ""

    q = (query or "").strip()
    if not _query_has_tokens(q) and llm_call is not None:
        kind = analysis_kind or ("repair" if policy == "repair" else "latency")
        q = _llm_keyword_query(
            llm_call,
            analysis_kind=kind,
            code=code,
            errors=errors,
            latency_report=latency_report,
        )
    if not _query_has_tokens(q):
        return ""

    index = load_index(_corpus_for_policy(cfg, policy))
    ranked = rank_chunks(index, q, top_k=cfg.top_k)
    return format_rag2_block(policy, ranked, max_chars=cfg.max_chars)


def rag2_config_from_env(
    *,
    enabled: bool | None = None,
    mode: str | None = None,
    opt_corpus_dir: Path | str | None = None,
    repair_corpus_dir: Path | str | None = None,
    top_k: int | None = None,
    max_chars: int | None = None,
    allow_scrape: bool = False,
) -> Rag2Config:
    env_enabled = _truthy_env("C2HLS_RAG2")
    resolved_enabled = env_enabled if enabled is None else enabled

    if resolved_enabled and _truthy_env("C2HLS_RAG_SCRAPE") and not allow_scrape:
        raise ValueError("RAG2 cannot be combined with scrape (C2HLS_RAG_SCRAPE)")

    mode_raw = os.environ.get("C2HLS_RAG_MODE")
    env_mode = mode_raw.strip().lower() if mode_raw else None
    resolved_mode = env_mode if mode is None else (mode.strip().lower() if mode else None)

    if resolved_enabled and resolved_mode is None:
        resolved_mode = "both"
    elif resolved_enabled and resolved_mode not in RAG_MODES:
        raise ValueError(f"invalid C2HLS_RAG_MODE for RAG2: {resolved_mode!r}")

    opt_raw = os.environ.get("C2HLS_RAG2_OPT_CORPUS")
    repair_raw = os.environ.get("C2HLS_RAG2_REPAIR_CORPUS")
    resolved_opt = Path(opt_corpus_dir) if opt_corpus_dir is not None else (
        Path(opt_raw) if opt_raw else DEFAULT_OPT_CORPUS
    )
    resolved_repair = Path(repair_corpus_dir) if repair_corpus_dir is not None else (
        Path(repair_raw) if repair_raw else DEFAULT_REPAIR_CORPUS
    )

    top_k_raw = os.environ.get("C2HLS_RAG_TOP_K")
    env_top_k = int(top_k_raw) if top_k_raw else DEFAULT_TOP_K
    resolved_top_k = env_top_k if top_k is None else top_k

    max_raw = os.environ.get("C2HLS_RAG2_MAX_CHARS")
    env_max = int(max_raw) if max_raw else DEFAULT_MAX_CHARS
    resolved_max = env_max if max_chars is None else max_chars

    return Rag2Config(
        enabled=resolved_enabled,
        mode=resolved_mode,
        opt_corpus_dir=resolved_opt,
        repair_corpus_dir=resolved_repair,
        top_k=resolved_top_k,
        max_chars=resolved_max,
    )


def rag2_enabled_for_stage(stage: str, cfg: Rag2Config | None = None) -> bool:
    cfg = cfg or rag2_config_from_env()
    return bool(cfg.enabled and should_inject(cfg.mode, stage))


def retrieve_for_stage_rag2(
    cfg: Rag2Config,
    stage: str,
    *,
    code: str = "",
    error: str = "",
    latency_report: str = "",
    llm_call: Callable | None = None,
    dataflow_repair: bool = False,
) -> str:
    """Build query + retrieve for a pipeline stage."""
    if not cfg.enabled:
        return ""
    inject_stage = "dataflow" if stage == "dataflow" else stage
    if dataflow_repair:
        if not should_inject(cfg.mode, "dataflow"):
            return ""
        policy: Policy = "repair"
    else:
        if not should_inject(cfg.mode, inject_stage):
            return ""
        policy = policy_for_stage(inject_stage if inject_stage != "dataflow" else "dataflow")
        if stage == "dataflow" and not dataflow_repair:
            policy = "opt"

    if policy == "repair":
        query = build_repair_query(code=code, error=error)
        return retrieve_rag2(
            cfg,
            policy="repair",
            query=query,
            llm_call=llm_call,
            analysis_kind="repair",
            code=code,
            errors=error,
        )

    query = build_opt_query(code=code, latency_report=latency_report)
    return retrieve_rag2(
        cfg,
        policy="opt",
        query=query,
        llm_call=llm_call,
        analysis_kind="latency",
        code=code,
        latency_report=latency_report,
    )
