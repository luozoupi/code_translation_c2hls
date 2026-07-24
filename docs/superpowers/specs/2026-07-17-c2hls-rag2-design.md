# c2hls RAG2 — Dual-Policy Structured Retrieval

**Date:** 2026-07-17  
**Status:** Approved; implemented (2026-07-17)  
**Scope:** c2hls only. Additive to existing `--rag` (BM25) and `--rag --scrape` (keyword scrape). Does not modify the in-flight DeepSeek U280 sequence.

## Goal

Ship **RAG2**: stage-conditioned retrieval with two policies (optimization vs repair), structured deterministic queries (hybrid LLM-keyword fallback), curated corpora per policy, and small precise prompt blocks — as a third opt-in method for ablation against scrape/BM25 and skills-only.

## Decisions (locked)

| Decision | Choice |
|----------|--------|
| Deployment | **A** — separate opt-in; scrape/BM25 unchanged |
| Corpora | **1** — opt: UG1399±ug902; repair: UG1399 + bug/pragma bug DBs |
| Queries | **2** — hybrid: deterministic first; LLM keywords only if empty |
| Architecture | **A** — dual BM25 indexes + policy router |
| v1 surface | **2** — library + CLI + `rag2_skills` / `rag2_ns` flavors |

## Non-goals (v1)

- Replacing scrape/BM25 or changing running DeepSeek sequence jobs.
- Embedding / FAISS engines.
- Hand-curated error→excerpt map as primary repair path (BM25 + error-ID boost instead).
- Merging ChatHLS code; may reuse PDF paths already under `artifacts/rag/knowledge_repo/`.
- Changing skills retrieval; skills remain a separate prompt layer.

## CLI and env

| Flag / env | Meaning |
|------------|---------|
| `--rag2` / `C2HLS_RAG2=1` | Enable RAG2. **Absent → RAG2 off.** |
| `--rag-mode {flashopt,repair,both,everywhere}` / `C2HLS_RAG_MODE` | Same stage scope as existing RAG. Default when RAG2 on and mode unset: `everywhere` for ChatHLS flavors; library default `both` if only `--rag2` and no mode. |
| `--rag2-opt-corpus DIR` / `C2HLS_RAG2_OPT_CORPUS` | Opt index dir. Default: `artifacts/rag/rag2_opt` |
| `--rag2-repair-corpus DIR` / `C2HLS_RAG2_REPAIR_CORPUS` | Repair index dir. Default: `artifacts/rag/rag2_repair` |
| `--rag-top-k N` / `C2HLS_RAG_TOP_K` | Shared top-k (default **4**). |
| `--rag2-max-chars N` / `C2HLS_RAG2_MAX_CHARS` | Max formatted block chars (default **4000**). |

Mutual exclusion:

- `--rag2` + `--scrape` (or `C2HLS_RAG_SCRAPE=1`) → **hard error** at CLI/config resolve time.
- `--rag2` may coexist with `C2HLS_RAG=1` only as a convenience alias for mode/top-k env reuse; when `C2HLS_RAG2=1`, injection uses **RAG2 only** (skip BM25 `retrieve_for_stage` and scrape at wired call sites).
- `--rag2` without indexes → **hard fail** with build-script hint.

## Policies and stages

| Policy | Stages (via `should_inject`) | Corpus |
|--------|------------------------------|--------|
| **opt** | `flashopt`, `dataflow` (initial generate) | `rag2_opt` |
| **repair** | `repair`, and dataflow repair turns | `rag2_repair` |

Mapping:

- `flashopt` / dataflow first-gen → policy `opt`
- compile/csim/csynth/cosim/quality repair + dataflow repair → policy `repair`

## Corpus layout

Sources (already vendored):

```
artifacts/rag/knowledge_repo/
  ug1399-vitis-hls-en-us-2024.1.pdf
  ug902-vivado-high-level-synthesis.pdf
  bug_database.pdf
  pragma_bug_database.pdf
```

Indexes:

```
artifacts/rag/rag2_opt/
  index_meta.json    # policy=opt, sources=[ug1399, ug902], engine=bm25, chunk_size=1000, overlap=200
  chunks.jsonl       # {id, text, source, ...}

artifacts/rag/rag2_repair/
  index_meta.json    # policy=repair, sources=[ug1399, bug_database, pragma_bug_database], engine=bm25, ...
  chunks.jsonl
```

Build once:

```bash
python3 scripts/build_rag2_indexes.py \
  --knowledge-repo artifacts/rag/knowledge_repo \
  --opt-out artifacts/rag/rag2_opt \
  --repair-out artifacts/rag/rag2_repair
```

Chunking: same as UG1399 RAG (1000 / 200). Each chunk records `source` basename for citation in the prompt block.

**Opt sources:** ug1399 + ug902.  
**Repair sources:** ug1399 + bug_database + pragma_bug_database (ug902 omitted from repair v1 to reduce interface-noise on error fixes; ug1399 covers HLS errors).

## Runtime architecture

New module: `c2hls_rag2.py` (repo root).

Public API:

- `rag2_enabled_from_env(...) -> bool`
- `rag2_config_from_env(...) -> Rag2Config`
- `policy_for_stage(stage: str) -> Literal["opt","repair"]`
- `build_opt_query(*, code: str, latency_report: str) -> str`
- `build_repair_query(*, code: str, error: str) -> str`
- `extract_hls_error_ids(text: str) -> list[str]`
- `extract_bottleneck_tags(latency_report: str, code: str) -> list[str]`
- `retrieve_rag2(cfg, *, policy, query, llm_call=None) -> str`  
  — BM25 retrieve; if deterministic query empty and `llm_call` provided, run keyword analysis once, rebuild query, retrieve; format + truncate.
- `format_rag2_block(policy, chunks) -> str`

Reuse BM25 primitives from `c2hls_rag.py` (`load_index`, `retrieve` / `_rank_chunks`, `tokenize`) rather than forking scoring math. Prefer importing `load_index` + ranking helpers; if helpers are private, export a thin `rank_chunks(index, query, top_k)` from `c2hls_rag.py` without changing BM25 defaults.

### Query construction

**Opt (deterministic):**

- Bottleneck tags from latency/loop text: e.g. `II`, `initiation interval`, `pipeline`, `dependence`, `partition`, `dataflow`, `burst`, `m_axi`, `unroll`, `bound`.
- From code: pragma names (`PIPELINE`, `ARRAY_PARTITION`, `DATAFLOW`, `INTERFACE`, …) and a short kernel head (≤800 chars).
- Query string = space-joined tags + pragma tokens + truncated code head.

**Repair (deterministic):**

- Regex HLS IDs: `HLS\s+\d+-\d+`, bracket forms `[HLS …]`.
- Pragma names appearing in error + nearby code window (≤1500 chars of error, ≤800 of code).
- Query = error IDs (repeated once for boost) + pragma tokens + error excerpt head.

**Hybrid fallback:** if deterministic query has no tokens (or only whitespace after tokenize), and an `llm_call` is available, call the existing scrape analysis prompts (`KEYWORD_ANALYSIS_LATENCY` / `KEYWORD_ANALYSIS_REPAIR` in `c2hls_rag_scrape.py`), parse `{"keywords":[...]}`, join as query. If still empty → return `""` (no injection).

### Repair scoring boost

Before ranking, if error IDs were extracted, append each ID twice to the query string (lexical boost). No separate IDF hack required for v1.

### Prompt block

```markdown
## RAG2 (opt)
### chunk c12 (ug1399-vitis-hls-en-us-2024.1.pdf)
...
```

or `## RAG2 (repair)`. Truncate total block to `max_chars` on chunk boundaries when possible.

Injection placement (match scrape precedence for consistency with skills ordering):

- **Prepend** RAG2 block ahead of skills/task on flashopt, repair, and dataflow action prompts (same sites that currently call scrape or `_rag_append`).
- When RAG2 enabled for a stage, **do not** also append BM25 or scrape for that call.

## Integration map

| Site | Change |
|------|--------|
| `c2hls.py` | `--rag2` CLI; `_rag2_block_for_*` helpers; prefer RAG2 over scrape/BM25 at flash/repair injection points |
| `post_flash_dataflow.py` | Same for dataflow generate/repair |
| `scripts/pc2/run_post_flash_dataflow.py` | Pass-through `--rag2` and corpus env |
| `scripts/pc2/start_chathls_deepseek_one.sh` | Flavors `rag2_skills`, `rag2_ns` |
| Flash/dataflow batch wrappers | Export `C2HLS_RAG2=1`, clear scrape; mode `everywhere` |

### Flavors

| Flavor | RAG2 | Skills | Notes |
|--------|------|--------|-------|
| `rag2_skills` | on, mode `everywhere` | on (90-skills flash / no_RMW dataflow) | Opt+repair indexes |
| `rag2_ns` | on, mode `everywhere` | off | Same corpora |
| existing `rag_skills` / `skills` / `rag_ns` | unchanged | — | scrape or off |

Artifact prefixes: `batch_parallel_chathls_fd_ds_rag2` and `..._rag2_ns` (DeepSeek); mirror non-DeepSeek starters if a shared flavor switch already exists — extend `start_chathls_deepseek_one.sh` and any parallel `start_chathls_*` flavor matrix that already has `rag_skills`.

## Error handling

- Missing index → fail fast with `scripts/build_rag2_indexes.py` hint.
- Empty retrieval → silent no-op (prompt unchanged).
- LLM keyword fallback failure / bad JSON → treat as empty keywords; no crash.
- Scrape+RAG2 both set → config error before any LLM call.

## Testing

| Test file | Coverage |
|-----------|----------|
| `tests/test_c2hls_rag2.py` | error-ID extract; bottleneck tags; policy_for_stage; mutual exclusion with scrape; format+truncate; retrieve over tiny fixture indexes; hybrid fallback when deterministic empty |
| Optional extend | CLI argparse rejects `--rag2 --scrape` |

Fixture: tiny `chunks.jsonl` + `index_meta.json` under `tests/fixtures/rag2_{opt,repair}/` (hand-written, no PDF dependency in CI).

## Success criteria

1. With `--rag2` and built indexes, flashopt/repair/dataflow prompts can include `## RAG2 (opt|repair)` blocks from the correct corpus.
2. `--rag2 --scrape` fails closed.
3. Existing scrape/BM25 tests still pass; running DeepSeek sequence env untouched.
4. `rag2_skills` / `rag2_ns` flavors start campaigns with `C2HLS_RAG2=1` and scrape off.
5. Unit tests green without network or GPU.

## Out of scope follow-ups (not v1)

- Switch live DeepSeek sequence to RAG2.
- Embeddings / hybrid dense retrieval.
- Auto-build indexes inside campaign start (operator builds once).
- ug902 in repair corpus.
