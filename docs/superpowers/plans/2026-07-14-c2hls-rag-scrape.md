# c2hls RAG scrape (`--rag --scrape`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `--rag --scrape`: analysis LLM emits keywords from code+errors or latency report → deterministic PDF/HTML scrape → prepend excerpts before skills on repair / first flash / first dataflow action prompts. BM25 `--rag` alone stays unchanged.

**Architecture:** New `c2hls_rag_scrape.py` owns text-cache extraction, keyword search, caps, and formatting. Keyword analysis uses a dedicated prompt + JSON parse (not codegen). Orchestrator/dataflow call `prepare_scrape_context(...)` before action LLM calls when scrape is enabled for the stage. CLI adds `--scrape` / `--scrape-corpus` requiring `--rag`.

**Tech Stack:** Python 3, existing pytest, optional `pypdf` for PDF text (same as index builder), reuse `c2hls_rag.should_inject` / `RagConfig` extended with `scrape` fields.

**Spec:** `docs/superpowers/specs/2026-07-14-c2hls-rag-scrape-addendum.md`  
**Parent RAG (done):** `c2hls_rag.py`, BM25 `--rag` wiring in `c2hls.py` / `post_flash_dataflow.py`

**Commits:** Only if the user explicitly asks; plan steps mark commit as optional.

---

## File map

| Path | Responsibility |
|------|----------------|
| `c2hls_rag_scrape.py` | Extract/cache doc text; scrape by keywords; format scrape block; parse keyword JSON; `ScrapeConfig` / `prepare_scrape_block` |
| `prompt_c2hls.py` (or small section in scrape module) | Analysis prompt templates: repair keywords vs latency keywords |
| `c2hls_rag.py` | Extend `RagConfig` with `scrape_enabled`, `scrape_corpus_paths`; env `C2HLS_RAG_SCRAPE`; `rag_config_from_env` kwargs |
| `c2hls.py` | `--scrape`, `--scrape-corpus`; wire Phase B repair / flash pre-gen / flash repair |
| `post_flash_dataflow.py` | Prepend scrape before skills on initial + repair when scrape+everywhere |
| `scripts/pc2/run_post_flash_dataflow.py` | Pass `--scrape` / env |
| `tests/fixtures/rag_scrape_mini/` | Tiny `.txt` “docs” for offline scrape tests (no PDF required in CI) |
| `tests/test_c2hls_rag_scrape.py` | Unit tests for scraper, parse, gating, CLI rules |
| `artifacts/rag/README.md` | Document `--scrape` + default PDF paths |

---

### Task 1: Scraper core + fixtures (TDD)

**Files:**
- Create: `c2hls_rag_scrape.py`
- Create: `tests/fixtures/rag_scrape_mini/doc_a.txt`, `doc_b.txt`
- Create: `tests/test_c2hls_rag_scrape.py`

- [ ] **Step 1: Write fixtures**

`tests/fixtures/rag_scrape_mini/doc_a.txt`:
```
DATAFLOW allows task-level parallelism. Arrays between processes must have a single producer and a single consumer. HLS 200-979 fails when a variable is written in more than one process.
```

`tests/fixtures/rag_scrape_mini/doc_b.txt`:
```
PIPELINE II=1 reduces initiation interval. Use #pragma HLS PIPELINE on the innermost loop.
```

- [ ] **Step 2: Write failing tests**

```python
# tests/test_c2hls_rag_scrape.py
from __future__ import annotations
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
FIX = REPO / "tests" / "fixtures" / "rag_scrape_mini"

from c2hls_rag_scrape import (  # noqa: E402
    extract_text_cached,
    format_scrape_block,
    parse_keywords_json,
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
```

- [ ] **Step 3: Run pytest — expect fail (import)**

```bash
cd /scratch/hpc-prf-llmfpga/asa582/projects/c2hls
python3 -m pytest tests/test_c2hls_rag_scrape.py -v
```

Expected: `ModuleNotFoundError: c2hls_rag_scrape`

- [ ] **Step 4: Implement `c2hls_rag_scrape.py` (minimal)**

API surface:

```python
MAX_KEYWORDS = 8
MAX_HITS_PER_KEYWORD = 3
MAX_TOTAL_CHARS = 6000
CONTEXT_CHARS = 200

def parse_keywords_json(llm_text: str) -> list[str]:
    """Extract {"keywords": [...]} from raw LLM text (fenced or bare). Cap length/strip empties."""

def extract_text_cached(path: Path, *, cache_dir: Path) -> str:
    """txt/html/pdf → plain text; cache under cache_dir / sha256(path+mtime).pdf.txt"""

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

def format_scrape_block(hits: list[dict]) -> str:
    """
    hits: {source, keyword, excerpt}
    Header: ## Scraped HLS documentation (keyword RAG)
    """
```

PDF: try `pypdf` like `build_ug1399_rag_index.py`; HTML strip tags; `.txt` read utf-8.

- [ ] **Step 5: Run tests — expect pass**

```bash
python3 -m pytest tests/test_c2hls_rag_scrape.py -v
```

- [ ] **Step 6: Commit** (only if user asks)

---

### Task 2: Analysis prompts + `prepare_scrape_block` orchestration helper

**Files:**
- Modify: `c2hls_rag_scrape.py` (add prompts + prepare helper)
- Modify: `tests/test_c2hls_rag_scrape.py`

- [ ] **Step 1: Add prompt constants**

```python
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
```

- [ ] **Step 2: Add `prepare_scrape_block`**

```python
def prepare_scrape_block(
    *,
    llm_call,  # Callable[[list[dict]], str] — messages in, text out
    analysis_kind: str,  # "repair" | "latency"
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
```

- [ ] **Step 3: Test with fake llm_call**

```python
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
```

- [ ] **Step 4: pytest pass**

- [ ] **Step 5: Commit** (optional)

---

### Task 3: Extend `RagConfig` + CLI `--scrape`

**Files:**
- Modify: `c2hls_rag.py` (`RagConfig`, `rag_config_from_env`)
- Modify: `c2hls.py` argparse
- Modify: `scripts/pc2/run_post_flash_dataflow.py`
- Modify: `tests/test_c2hls_rag.py` or `tests/test_c2hls_rag_scrape.py`

- [ ] **Step 1: Extend config**

```python
@dataclass(frozen=True)
class RagConfig:
    enabled: bool
    mode: Optional[RagMode]
    corpus_dir: Path          # BM25 index
    top_k: int
    scrape_enabled: bool = False
    scrape_corpus_paths: tuple[Path, ...] = ()
```

Env:
- `C2HLS_RAG_SCRAPE=1` → scrape_enabled  
- `C2HLS_RAG_SCRAPE_CORPUS` → colon/comma-separated paths  

Default scrape corpus (if scrape enabled and paths empty):  
`Path(os.environ.get("C2HLS_CHATHLS_KNOWLEDGE_REPO", "")) / ...` **or** explicit default:

```python
DEFAULT_SCRAPE_CORPUS = (
    # Prefer env C2HLS_SCRAPE_CORPUS; else empty and require --scrape-corpus
)
```

**v1 rule:** if `--scrape` and no corpus paths → `parser.error` listing example ChatHLS knowledge_repo PDF path (do not hard-depend on ChatHLS checkout existing).

Helper:

```python
def resolve_scrape_corpus(raw: str | None) -> tuple[Path, ...]:
    # split on : or , ; expanduser; keep existing files only; warn on missing
```

- [ ] **Step 2: CLI in `c2hls.py`**

```python
parser.add_argument("--scrape", action="store_true",
    help="With --rag: analysis→keyword PDF/doc scrape before action prompts (see scrape addendum).")
parser.add_argument("--scrape-corpus", type=str, default=None,
    help="Colon/comma-separated PDF/HTML/TXT paths for --scrape.")
```

After parse:
```python
if args.scrape and not args.rag:
    parser.error("--scrape requires --rag")
if args.rag and args.scrape:
    os.environ["C2HLS_RAG_SCRAPE"] = "1"
    if args.scrape_corpus:
        os.environ["C2HLS_RAG_SCRAPE_CORPUS"] = args.scrape_corpus
    cfg = rag_config_from_env(...)
    if not cfg.scrape_corpus_paths:
        parser.error("--scrape requires --scrape-corpus with existing files")
# BM25 get_index only when scrape is False (parent behavior)
if args.rag and not args.scrape:
    get_index(cfg)
```

**v1 inject policy when both would apply:** if `scrape_enabled` and stage should_inject → use scrape path; **do not** also BM25-append for that call (avoid double docs). Keep `_rag_append` BM25 for `--rag` without scrape.

- [ ] **Step 3: Mirror flags in `run_post_flash_dataflow.py`**

- [ ] **Step 4: Tests**

```python
def test_scrape_requires_rag(monkeypatch):
    # argparse logic unit test or source+rag_config
    ...

def test_rag_config_scrape_env(monkeypatch, tmp_path):
    monkeypatch.setenv("C2HLS_RAG", "1")
    monkeypatch.setenv("C2HLS_RAG_SCRAPE", "1")
    p = tmp_path / "d.txt"; p.write_text("x")
    monkeypatch.setenv("C2HLS_RAG_SCRAPE_CORPUS", str(p))
    cfg = rag_config_from_env()
    assert cfg.scrape_enabled and cfg.scrape_corpus_paths[0] == p
```

- [ ] **Step 5: Commit** (optional)

---

### Task 4: Wire Phase B + flash repair (error → scrape → fix)

**Files:**
- Modify: `c2hls.py`

- [ ] **Step 1: Add helper next to `_rag_append`**

```python
def _scrape_docs_for_repair(orch, *, code: str, error: str) -> str:
    from c2hls_rag import rag_config_from_env, should_inject
    from c2hls_rag_scrape import prepare_scrape_block
    cfg = rag_config_from_env()
    if not (cfg.enabled and cfg.scrape_enabled and should_inject(cfg.mode, "repair")):
        return ""
    cache = Path(os.environ.get("C2HLS_TMP_ROOT", str(REPO / "c2hls_tmp"))) / "rag_scrape_cache"
    def llm_call(messages):
        return orch._call_llm(messages, max_tokens=400)  # or orchestrator method used for short calls
    block, kws = prepare_scrape_block(
        llm_call=llm_call,
        analysis_kind="repair",
        code=code,
        errors=error,
        corpus_paths=list(cfg.scrape_corpus_paths),
        cache_dir=cache,
    )
    if kws:
        logging.info("RAG scrape keywords=%s chars=%d", kws, len(block))
    return block


def _prepend_scrape(prompt: str, scrape_block: str) -> str:
    if not scrape_block:
        return prompt
    return scrape_block.rstrip() + "\n\n" + prompt.lstrip()
```

- [ ] **Step 2: At each repair `_rag_append(..., "repair", ...)` site**  
  When scrape enabled, **replace** BM25 append with scrape prepend:

```python
        if _scrape_enabled_for("repair"):
            scrape = _scrape_docs_for_repair(orch, code=..., error=...)
            fix_prompt = _prepend_scrape(fix_prompt, scrape)
        else:
            fix_prompt = _rag_append(fix_prompt, "repair", ...)
```

Minimum sites: Phase B synthesis repair loop, quality repair, `_optimization_step_attempt_single` repairs, pipelined repair helpers (same set already using `_rag_append` for repair).

- [ ] **Step 3: Source/unit test** that scrape helper names exist; optional monkeypatch of `prepare_scrape_block`.

- [ ] **Step 4: Commit** (optional)

---

### Task 5: Wire first flash generation (latency analysis → scrape → flash)

**Files:**
- Modify: `c2hls.py` — `_optimization_step_initial_codegen` when `step_name == "flash"` (or flash entry)

- [ ] **Step 1: Before building flash user prompt / after report_str available**

```python
        scrape_block = ""
        if step_name == "flash":
            scrape_block = _scrape_docs_for_latency(
                self,
                code=self.hls_code or "",
                latency_report=report_str,
            )
        # after skills/extra_blocks assembled:
        if scrape_block:
            prompt = _prepend_scrape(prompt, scrape_block)
        elif not scrape_mode:
            prompt = _rag_append(prompt, "flashopt", ...)  # existing BM25
```

`_scrape_docs_for_latency` mirrors repair helper with `analysis_kind="latency"` and `should_inject(..., "flashopt")`.

- [ ] **Step 2: Test** source contains latency scrape for flash; gating test via config.

- [ ] **Step 3: Commit** (optional)

---

### Task 6: Wire dataflow initial + repair

**Files:**
- Modify: `post_flash_dataflow.py`
- Modify: `scripts/pc2/run_post_flash_dataflow.py` (already CLI in Task 3)

- [ ] **Step 1: Before `format_dataflow_initial_user` LLM call in the cell runner**  
  (find where initial user is built ~line 745)

```python
    scrape = ""
    if scrape_enabled_for_dataflow():
        scrape = prepare_scrape_block(... analysis_kind="latency",
            code=kernel_code,
            latency_report=flash_latency_summary,  # from synth_report if available
            ...)
    user = format_dataflow_initial_user(...)
    user = prepend_scrape(user, scrape)  # scrape before skills already inside template — prepend to whole user msg
```

If latency summary is missing, pass `format_report_summary` or `"(no latency report)"`.

- [ ] **Step 2: On dataflow repair** (`format_dataflow_repair_user`): error-driven `prepare_scrape_block` then prepend.

- [ ] **Step 3: When `--rag --scrape`, skip `_append_dataflow_rag` BM25 path** (v1 scrape replaces BM25 for dataflow).

- [ ] **Step 4: Tests** with fake llm + fixture corpus in `test_c2hls_rag_scrape.py` or post_flash tests.

- [ ] **Step 5: Commit** (optional)

---

### Task 7: Docs + verification

**Files:**
- Modify: `artifacts/rag/README.md`
- Update addendum status line to “implemented” only after tasks pass (optional)

- [ ] **Step 1: Document**

```bash
export CORPUS=/path/to/ChatHLS-ACL-26/src/knowledge_repo/ug1399-vitis-hls-en-us-2024.1.pdf

# BM25 only (parent)
python3 c2hls.py ... --rag

# Scrape path
python3 c2hls.py ... --rag --scrape --scrape-corpus "$CORPUS" --rag-mode both

python3 scripts/pc2/run_post_flash_dataflow.py ... --rag --scrape --scrape-corpus "$CORPUS" --rag-mode everywhere
```

- [ ] **Step 2: Run**

```bash
python3 -m pytest tests/test_c2hls_rag_scrape.py tests/test_c2hls_rag.py -v --tb=short
```

- [ ] **Step 3: Confirm** no PC2 `start_*.sh` enables scrape by default (`grep -E 'RAG_SCRAPE|--scrape' scripts/pc2/start*.sh` → empty).

---

## Spec coverage

| Addendum requirement | Task |
|----------------------|------|
| `--scrape` requires `--rag` | 3 |
| analysis → keywords → scrape → act | 2, 4–6 |
| Phase B initial unchanged; repair scrape | 4 |
| Flash first gen from latency analysis | 5 |
| Flash/Phase B repair scrape | 4 |
| Dataflow first gen + repair | 6 |
| Scrape before skills | 4–6 (`_prepend_scrape`) |
| Deterministic scraper + caps | 1 |
| BM25 alone unchanged | 3 |
| Soft fail empty scrape | 1–2 |
| No default campaign enable | 7 |

## Placeholder scan

No TBD left for v1 behavior; PDF default path is explicitly “must pass `--scrape-corpus`” to avoid silent ChatHLS coupling.

---

## Execution handoff

Plan saved to `docs/superpowers/plans/2026-07-14-c2hls-rag-scrape.md`.

**Two execution options:**

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks  
2. **Inline Execution** — this session with checkpoints  

Which approach?
