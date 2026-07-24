# c2hls UG1399 RAG (opt-in) — Design

**Date:** 2026-07-14  
**Status:** Approved for planning (pending user review of this spec)  
**Scope:** c2hls only. ChatHLS is an external inspiration (UG1399 chunking scheme); do not merge ChatHLS code, skills, or grafted local ChatHLS tree into this feature.

## Goal

Add optional Retrieval-Augmented Generation over a **bundled/cached UG1399 (Vitis HLS User Guide) index** so LLM prompts in c2hls can be grounded in vendor documentation. RAG is **off by default**. When enabled, mode selects which pipeline stages receive retrieved context.

## Non-goals (v1)

- Shipping the UG1399 PDF/HTML inside git (operator builds the index once).
- Fine-tuned HLSFixer/HLSTuner models.
- Replacing the skills library; RAG is **additive** to skills.
- Treating ChatHLS `skills.json` / local ChatHLS grafts as part of this design.
- Embedding-model dependency for v1 (upgrade path only).

## CLI and env

| Flag / env | Meaning |
|------------|---------|
| `--rag` / `C2HLS_RAG=1` | Enable RAG. **Absent → RAG off.** |
| `--rag-mode {flashopt,repair,both,everywhere}` / `C2HLS_RAG_MODE` | Injection scope. **Default when `--rag` is set: `flashopt`.** |
| `--rag-corpus DIR` / `C2HLS_RAG_CORPUS` | Index directory. **Default:** `<repo>/artifacts/rag/ug1399` |
| `--rag-top-k N` / `C2HLS_RAG_TOP_K` | Number of chunks. **Default: 4** |

Rules:

- `--rag-mode` without `--rag` → error (or ignore with warning; prefer **error**).
- `--rag` with missing/invalid index → **hard fail** with message to run the build script.
- PC2 wrappers may pass the same via env for batch jobs.

## Mode → injection points

| Mode | Injection |
|------|-----------|
| `flashopt` | Flash skill curation and flash rewrite / optimize prompts |
| `repair` | Compile / csim / csynth / cosim / quality-repair prompts |
| `both` | `flashopt` + `repair` |
| `everywhere` | `both` + post-flash dataflow generate/repair (and any other HLS edit LLM calls in-tree) |

Retrieved text is appended as a distinct prompt section, e.g.:

```markdown
## Retrieved HLS documentation (UG1399)
...chunks...
```

It must not replace skill blocks or failure ERROR extracts.

## Corpus layout (bundled cache)

Default path: `artifacts/rag/ug1399/`

```
artifacts/rag/ug1399/
  index_meta.json      # source hash, chunk_size=1000, overlap=200, engine=bm25, built_at
  chunks.jsonl         # one JSON object per chunk: {id, text, start, end, ...}
  bm25_index/          # or single pickle/json sidecar — implementation detail
```

Build once:

```bash
python3 scripts/build_ug1399_rag_index.py \
  --source /path/to/UG1399.pdf|html|txt \
  --out artifacts/rag/ug1399
```

Chunking (inspired by ChatHLS paper Appendix A.1, implemented in c2hls):

- chunk length **1000**
- overlap **200**

## Runtime architecture

New module: `c2hls_rag.py` (repo root, alongside other c2hls libs).

Public API (sketch):

- `rag_enabled_from_args_env(...) -> bool`
- `rag_mode_from_args_env(...) -> Literal["flashopt","repair","both","everywhere"] | None`
- `load_index(corpus_dir) -> RagIndex` (cached per process)
- `retrieve(query: str, *, top_k: int) -> str`  # formatted prompt block or empty
- `should_inject(mode, stage: str) -> bool` where `stage` ∈ `{flashopt, repair, dataflow, ...}`

**v1 engine: BM25** (or equivalent lexical retrieval) over chunk texts — no sentence-transformers required.

**v2 path:** same API; `index_meta.json` `engine` field may become `faiss_minilm` later without changing CLI.

### Query construction

- **flashopt:** kernel excerpt + bottleneck / skill tags / brief task description  
- **repair:** kernel excerpt + stage + ERROR lines / failure summary  
- **dataflow (everywhere):** current kernel + contract/synth errors as applicable  

Cap query and retrieved block size so prompts stay bounded (implementation plan can set char limits).

## Integration map (c2hls)

1. `c2hls.py` argparse: add `--rag`, `--rag-mode`, `--rag-corpus`, `--rag-top-k`; plumb into orchestrator / env.  
2. Flash path: when `should_inject(..., "flashopt")`, append RAG block to skill-curation / flash rewrite prompts.  
3. Repair path: when `should_inject(..., "repair")`, append RAG block next to existing failure context.  
4. Dataflow (`post_flash_dataflow.py` / related): only if mode is `everywhere`.  
5. PC2 batch scripts: optional env passthrough; **default campaigns remain RAG-off** unless explicitly set.

## Failure / observability

- Hard fail if `--rag` and corpus unreadable or meta missing.  
- Log: mode, corpus path, top-k, retrieved chunk ids (and optionally scores) at INFO.  
- Optional artifact: write retrieved block sidecar next to step transcripts when record-flow is on (nice-to-have; not required for v1).

## Testing

- Unit: chunker (1000/200), BM25 retrieve ordering, `should_inject` matrix for all modes.  
- Unit: CLI — no `--rag` → off; `--rag` → mode defaults `flashopt`; `--rag-mode` without `--rag` → error.  
- Smoke: tiny fixture corpus (not full UG1399) under `tests/fixtures/rag_ug1399_mini/`.

## Success criteria

1. Default behavior unchanged (no `--rag` ⇒ identical prompts to today).  
2. `--rag` alone injects UG1399 context into **flashopt** only.  
3. `--rag --rag-mode repair|both|everywhere` match the injection table.  
4. Index built by script into `artifacts/rag/ug1399` and consumed by default.  
5. Skills remain primary optimization recipes; RAG is documentation grounding only.

## Self-review checklist

- [x] No “TBD” placeholders for mode semantics or CLI defaults  
- [x] ChatHLS vs c2hls ownership stated  
- [x] Corpus build vs runtime separated  
- [x] v1 BM25 / v2 embeddings path noted without blocking v1  
- [x] User review of this file before implementation plan  

## Related addendum

Keyword-analysis + PDF/doc scrape path (`--rag --scrape`): see  
`docs/superpowers/specs/2026-07-14-c2hls-rag-scrape-addendum.md`.
