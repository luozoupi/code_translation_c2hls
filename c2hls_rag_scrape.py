"""Keyword-driven PDF/HTML/TXT scrape for c2hls RAG (--rag --scrape)."""

from __future__ import annotations

import hashlib
import json
import re
from html.parser import HTMLParser
from pathlib import Path

MAX_KEYWORDS = 8
MAX_HITS_PER_KEYWORD = 3
MAX_TOTAL_CHARS = 6000
CONTEXT_CHARS = 200
MAX_CODE_CHARS = 8000

KEYWORD_ANALYSIS_REPAIR = """You are diagnosing an HLS failure. Do NOT write or rewrite kernel code.

Given the current HLS code, errors/warnings, and brief context, propose search keywords
to look up in Vitis HLS documentation (error IDs, pragma names, constraint phrases).

Return ONLY a JSON object:
{"keywords": ["...", "..."]}
Max 8 keywords. Prefer HLS error codes and pragma names when present.
"""

KEYWORD_ANALYSIS_LATENCY = """You are analyzing HLS performance before an optimization rewrite. Do NOT write kernel code.

Given the HLS code and latency/loop synthesis summary, propose documentation search keywords
(pragmas, patterns, bottlenecks) useful for the next optimization/dataflow step.

Return ONLY a JSON object:
{"keywords": ["...", "..."]}
Max 8 keywords.
"""


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self._parts.append(data)

    def get_text(self) -> str:
        return re.sub(r"\s+", " ", " ".join(self._parts)).strip()


def _cache_path(path: Path, cache_dir: Path) -> Path:
    mtime = path.stat().st_mtime
    digest = hashlib.sha256(f"{path}:{mtime}".encode()).hexdigest()
    suffix = path.suffix.lower().lstrip(".") or "txt"
    return cache_dir / f"{digest}.{suffix}.txt"


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError(
            "PDF source requires the optional 'pypdf' package. "
            "Install with: pip install pypdf"
        ) from exc

    reader = PdfReader(str(path))
    pages: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            pages.append(text)
    return "\n".join(pages)


def _read_html(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="replace")
    parser = _HTMLTextExtractor()
    parser.feed(raw)
    return parser.get_text()


def _read_source(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _read_pdf(path)
    if suffix in (".html", ".htm"):
        return _read_html(path)
    return path.read_text(encoding="utf-8", errors="replace")


def extract_text_cached(path: Path, *, cache_dir: Path) -> str:
    """txt/html/pdf → plain text; cache under cache_dir / sha256(path+mtime).{ext}.txt"""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = _cache_path(path, cache_dir)
    if cached.is_file():
        return cached.read_text(encoding="utf-8", errors="replace")

    text = _read_source(path)
    cached.write_text(text, encoding="utf-8")
    return text


def parse_keywords_json(llm_text: str) -> list[str]:
    """Extract {"keywords": [...]} from raw LLM text (fenced or bare). Cap length/strip empties."""
    text = llm_text.strip()

    fenced = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()

    match = re.search(
        r'\{[^{}]*"keywords"\s*:\s*\[[^\]]*\][^{}]*\}',
        text,
        re.DOTALL,
    )
    if not match:
        return []

    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []

    raw_keywords = obj.get("keywords", [])
    if not isinstance(raw_keywords, list):
        return []

    keywords: list[str] = []
    for item in raw_keywords:
        if not isinstance(item, str):
            continue
        kw = item.strip()
        if kw:
            keywords.append(kw)
        if len(keywords) >= MAX_KEYWORDS:
            break
    return keywords


def format_scrape_block(hits: list[dict]) -> str:
    """
    hits: {source, keyword, excerpt}
    Header: ## Scraped HLS documentation (keyword RAG)
    """
    if not hits:
        return ""

    lines = ["## Scraped HLS documentation (keyword RAG)", ""]
    for hit in hits:
        source = hit.get("source", "")
        keyword = hit.get("keyword", "")
        excerpt = hit.get("excerpt", "")
        lines.append(f"**{keyword}** in `{source}`:")
        lines.append(excerpt)
        lines.append("")
    return "\n".join(lines).rstrip()


def _find_keyword_hits(
    text: str,
    *,
    keyword: str,
    source: str,
    context_chars: int,
    max_hits: int,
) -> list[dict]:
    hits: list[dict] = []
    lower_text = text.lower()
    lower_kw = keyword.lower()
    start = 0

    while len(hits) < max_hits:
        idx = lower_text.find(lower_kw, start)
        if idx == -1:
            break
        excerpt_start = max(0, idx - context_chars)
        excerpt_end = min(len(text), idx + len(keyword) + context_chars)
        hits.append(
            {
                "source": source,
                "keyword": keyword,
                "excerpt": text[excerpt_start:excerpt_end].strip(),
            }
        )
        start = idx + max(1, len(lower_kw))

    return hits


def scrape_keywords(
    keywords: list[str],
    *,
    corpus_paths: list[Path],
    max_keywords: int = MAX_KEYWORDS,
    max_hits_per_keyword: int = MAX_HITS_PER_KEYWORD,
    max_total_chars: int = MAX_TOTAL_CHARS,
    context_chars: int = CONTEXT_CHARS,
    cache_dir: Path | None = None,
) -> str:
    """Case-insensitive search; return format_scrape_block(...) or ""."""
    cleaned: list[str] = []
    for kw in keywords:
        if not isinstance(kw, str):
            continue
        s = kw.strip()
        if s:
            cleaned.append(s)
        if len(cleaned) >= max_keywords:
            break

    if not cleaned or not corpus_paths:
        return ""

    docs: list[tuple[str, str]] = []
    for corpus_path in corpus_paths:
        if not corpus_path.is_file():
            continue
        if cache_dir is not None:
            text = extract_text_cached(corpus_path, cache_dir=cache_dir)
        else:
            text = _read_source(corpus_path)
        docs.append((corpus_path.name, text))

    if not docs:
        return ""

    hits: list[dict] = []
    total_chars = 0

    for keyword in cleaned:
        keyword_hits = 0
        for source_name, text in docs:
            for hit in _find_keyword_hits(
                text,
                keyword=keyword,
                source=source_name,
                context_chars=context_chars,
                max_hits=max_hits_per_keyword - keyword_hits,
            ):
                excerpt_len = len(hit["excerpt"])
                if total_chars + excerpt_len > max_total_chars:
                    return format_scrape_block(hits)
                hits.append(hit)
                total_chars += excerpt_len
                keyword_hits += 1
                if keyword_hits >= max_hits_per_keyword:
                    break
            if keyword_hits >= max_hits_per_keyword:
                break

    return format_scrape_block(hits)


def _truncate_code(code: str, max_chars: int = MAX_CODE_CHARS) -> str:
    if len(code) <= max_chars:
        return code
    return code[:max_chars] + "\n... [truncated]"


def _build_analysis_user_message(
    *,
    analysis_kind: str,
    code: str,
    errors: str = "",
    latency_report: str = "",
) -> str:
    truncated = _truncate_code(code)
    if analysis_kind == "repair":
        parts = ["## HLS code", truncated]
        if errors.strip():
            parts.extend(["", "## Errors / warnings", errors.strip()])
        return "\n".join(parts)

    if analysis_kind == "latency":
        parts = ["## HLS code", truncated]
        report = latency_report.strip() or "(no latency report)"
        parts.extend(["", "## Latency / loop synthesis summary", report])
        return "\n".join(parts)

    raise ValueError(f"unknown analysis_kind: {analysis_kind!r}")


def prepare_scrape_block(
    *,
    llm_call,
    analysis_kind: str,
    code: str,
    errors: str = "",
    latency_report: str = "",
    corpus_paths: list[Path],
    cache_dir: Path,
) -> tuple[str, list[str]]:
    """
    1) Build analysis user message from code/errors/latency (truncate code to 8k).
    2) llm_call([{system or user analysis}, {user content}])
    3) parse_keywords_json; if empty, return ("", [])
    4) scrape_keywords → (block, keywords)
    """
    if analysis_kind == "repair":
        system_prompt = KEYWORD_ANALYSIS_REPAIR
    elif analysis_kind == "latency":
        system_prompt = KEYWORD_ANALYSIS_LATENCY
    else:
        raise ValueError(f"unknown analysis_kind: {analysis_kind!r}")

    user_message = _build_analysis_user_message(
        analysis_kind=analysis_kind,
        code=code,
        errors=errors,
        latency_report=latency_report,
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    raw = llm_call(messages)
    keywords = parse_keywords_json(raw)
    if not keywords:
        return ("", [])

    block = scrape_keywords(
        keywords,
        corpus_paths=corpus_paths,
        cache_dir=cache_dir,
    )
    return (block, keywords)
