# analysis_tools/

The current, actively-used analysis/reporting scripts for the enhanced-framework
Opus 4.8 campaign. Everything else (the ~80 older one-off analysis/validation/debug
scripts) was archived under `../_archive/scripts/` during the 2026-07-09 cleanup.

**Run these from the project root** (`code_translation_c2hls/`), e.g.:

```bash
python3 analysis_tools/_emit_all_arms_schema.py     # -> schema_records.jsonl (523 records, canonical v1.0)
python3 analysis_tools/_compare_ms_ablation.py      # multistep curated vs all-positive skill-breadth ablation
python3 analysis_tools/_summary_table.py            # per-arm speedup-vs-gold summary
python3 analysis_tools/_success_rates.py            # cosim-verified success rates per arm
python3 analysis_tools/_count_tokens.py             # per-sweep token totals from llm_usage
```

Notes:
- `_emit_all_arms_schema.py` locates the project root via `__file__` (works from
  either `analysis_tools/` or root), and drives the canonical emitter
  `analysis/_emit_schema_records.py`. Reads the active `results_matrix_u280_ENH_*_OPUS`
  arms + the `*_symmrerun_OPUS` dirs; writes `schema_records.jsonl` at the root.
- The other four are **cwd-relative** (they `glob("results_matrix_*")` and
  `open("gold_reports_*.json")`), so they only work when invoked from the project root.
- Validate the emitted JSONL with `python3 scripts/validate_jsonl_semantics.py schema_records.jsonl`.
