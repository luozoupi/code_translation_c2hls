# Addendum: LLM keyword analysis + PDF/doc scrape (`--rag --scrape`)

**Date:** 2026-07-14  
**Parent:** `docs/superpowers/specs/2026-07-14-c2hls-ug1399-rag-design.md`  
**Status:** Design only (approved intent; not implemented yet)

## Intent

Extend opt-in RAG with an **agent-driven scrape path**:

1. **Analysis prompt** (not the optimization/codegen prompt): LLM sees code + errors/warnings **or** latency/loop analysis and returns **keywords only**.  
2. **Fixed Python scraper** searches configured PDF/HTML/docs for those keywords.  
3. **Action prompt** (repair / flash / dataflow): scraped excerpts are **prepended ahead of skills**, then the LLM writes or fixes code.

This is **not** classic “same response emits kernel + keywords” (chicken-and-egg). It is **analysis → scrape → act**.

BM25 over `artifacts/rag/ug1399` (parent spec) remains available as `--rag` without `--scrape`. Scrape requires **both** `--rag` and `--scrape`.

## CLI

| Flag | Meaning |
|------|---------|
| `--rag` | Enable RAG family (parent BM25 and/or scrape) |
| `--scrape` | Enable keyword-analysis + PDF/doc scrape path (**requires `--rag`**) |
| `--rag-mode …` | Same injection stages as parent (`flashopt`, `repair`, `both`, `everywhere`) |
| `--scrape-corpus DIR` or file list | Optional override; default: ChatHLS-style knowledge PDFs path or c2hls-configured list (UG1399 primary; bug DBs optional for repair) |

Env mirrors: `C2HLS_RAG`, `C2HLS_RAG_SCRAPE=1`, etc.

Rules:

- `--scrape` without `--rag` → error  
- No `--rag` → neither BM25 nor scrape  
- `--rag` alone → parent BM25 behavior (unchanged)  
- `--rag --scrape` → scrape path for stages allowed by `--rag-mode` (BM25 may be skipped or used as fallback — **v1: scrape only when `--scrape` set**)

## Shared scrape pattern

```
analysis_input = {code, errors/warnings} OR {code, latency/loop report}
     │
     ▼
LLM analysis prompt  →  {"keywords": ["...", ...]}   (≤ N keywords, hard-parse/retry)
     │
     ▼
scrape_pdfs_or_html(keywords)  →  excerpts (≤ K hits, ≤ C chars)
     │
     ▼
action prompt = [scraped docs] + [skills] + [task/context/code/errors]
     │
     ▼
LLM codegen / repair
```

Analysis prompt responsibilities: diagnose what matters (pragmas, HLS error IDs, patterns).  
**Does not** emit optimized kernels.  
Action prompt responsibilities: apply skills + scraped doc grounding to produce/fix code.

## Stage wiring

### Phase B
- **Initial translate:** unchanged (no scrape required).  
- **Repair turns** (`--rag --scrape`, mode includes `repair`):  
  errors/warnings + code → keywords → scrape → fix prompt (scrape **before** skills).

### Flash
- **Before first flash generation** (`flashopt` / `both` / `everywhere`):  
  after Phase B success, **latency/loop analysis + code** → keywords → scrape → flash codegen (scrape + skills).  
- **Flash repair turns:** same as Phase B repair (error-driven analysis → scrape → fix).

### Dataflow (`everywhere`, or dataflow entry when scrape enabled for that campaign)
- **Before first dataflow generation:**  
  flash_selected **code + latency analysis** → keywords → scrape → dataflow codegen with scrape + **no_RMW dataflow skills**.  
- **Dataflow repair:** error-driven analysis → scrape → fix (same pattern).

Skills remain the optimization recipe layer; scrape is **documentation grounding** only.

## Scraper (implementation sketch; later)

- Deterministic Python (no LLM): keyword search over pre-extracted text cache of PDFs/HTML (extract-once, reuse).  
- Caps: max keywords, max hits per keyword, max total chars, case-insensitive match, context window around hit.  
- Default corpus (configurable): prefer `ug1399-*.pdf`; optionally include bug/pragma databases for repair stages.  
- Fail soft on scrape miss (empty block + warning); do not fail the whole run unless configured.

## Relationship to parent BM25 RAG

| Mode | Behavior |
|------|----------|
| `--rag` | BM25 chunk retrieve (parent spec) |
| `--rag --scrape` | Analysis + keyword scrape (this addendum); v1 preferred over BM25 for those stages |
| off | unchanged |

Do not merge ChatHLS code; may **point at** ChatHLS `knowledge_repo` PDFs as corpus sources via path config.

## Non-goals (this addendum)

- Embedding / vector search for scrape  
- Mixing keyword JSON into the same message as full kernel codegen  
- Enabling scrape by default in PC2 campaigns  
- Committing UG1399/bug PDFs into c2hls git  

## Success criteria

1. `--scrape` alone errors; `--rag --scrape` runs analysis→scrape→act for allowed stages.  
2. Phase B initial translate unchanged; Phase B/flash/dataflow repairs can use scrape.  
3. First flash gen and first dataflow gen can receive scrape hits from **pre-action** analysis (latency/code), not from a post-hoc sidecar on the codegen response.  
4. Scraped text is prepended before skills in action prompts.  
5. Without flags, behavior matches today / parent BM25-only when `--rag` alone.

## Next

Implementation plan: `docs/superpowers/plans/2026-07-14-c2hls-rag-scrape.md`.
