# PC2 Flash Test Matrix — LLM-Curated Skills

15 PC2 flash runs: **5 variants × 3 curation waves**, isolated from legacy and deterministic new-matrix families.

---

## Overview

| Item | Value |
|------|-------|
| Matrix family | `flash_llm_curated_skills` |
| Skills file | `skills_ii_target_miss_solutions_added.json` (73 skills, packaged-only) |
| Waves (sequential) | `bottleneck` → `warnings` → `combined` |
| Variants per wave | 5 |
| Walltime | 12h GPU + 12h compute (`PC2_FORCE_WALLTIME`) |
| Watch interval | 60s (`PC2_WATCH_INTERVAL_SEC`) |
| Queue gate | Start each wave only when `squeue -u $USER` is empty |

---

## Curation waves

| Wave | `C2HLS_SKILL_CURATION_FOCUS` | First LLM call emphasizes |
|------|------------------------------|---------------------------|
| A | `bottleneck` | Synthesis bottlenecks and scope data |
| B | `warnings` | HLS warnings, errors, rejected pragmas |
| C | `combined` | A + B + load-compute-store (tiling, pipeline, double buffer, coalescing) |

---

## Five variants (each wave)

| # | Variant key | Session ID | Sector | Curation | Skill injection |
|---|-------------|------------|--------|----------|-----------------|
| 1 | `noskills` | `flash_curated_noskills` | n/a | **skipped** | none |
| 2 | `all_avoids_json` | `flash_curated_all_avoids_json` | A (`json_only`) | LLM picks catalog IDs | curated subset + avoids |
| 3 | `all_avoids_llm` | `flash_curated_all_avoids_llm` | B (`json_plus_llm`) | catalog + LLM snippets | curated + free-form guidance |
| 4 | `no_avoids_json` | `flash_curated_no_avoids_json` | A (`json_only`) | catalog only | positive skills only |
| 5 | `no_avoids_llm` | `flash_curated_no_avoids_llm` | B (`json_plus_llm`) | catalog + LLM knowledge | curated + free-form guidance |

Noskills uses packaged JSON env for parity but `C2HLS_FORCE_SKILL_PROMPTS=0` and `C2HLS_SKILL_CURATION_ENABLED=0`.

---

## Artifact paths

Pattern: `artifacts/pc2/{artifact_prefix}_{focus}_{stamp}/`

| Variant | Artifact prefix |
|---------|-----------------|
| noskills | `flash_curated_noskills` |
| all_avoids_json | `flash_curated_all_avoids_json` |
| all_avoids_llm | `flash_curated_all_avoids_llm` |
| no_avoids_json | `flash_curated_no_avoids_json` |
| no_avoids_llm | `flash_curated_no_avoids_llm` |

Example (stamp `20260622_120000`, wave bottleneck):

- `artifacts/pc2/flash_curated_noskills_bottleneck_20260622_120000/`
- `artifacts/pc2/flash_curated_all_avoids_json_bottleneck_20260622_120000/`
- …

Per-bench cell: `{out_root}/{bench}/{model_tag}__{setup_tag}/`  
Curation audit: `{cell}/skill_curation.json`

---

## Environment variables (curated variants)

| Variable | Curated variants | Noskills |
|----------|------------------|----------|
| `C2HLS_PACKAGED_SKILLS_JSON` | `skills_ii_target_miss_solutions_added.json` | same |
| `C2HLS_PACKAGED_SKILLS_ONLY` | `1` | `1` |
| `C2HLS_SKILL_CURATION_ENABLED` | `1` | `0` |
| `C2HLS_SKILL_CURATION_FOCUS` | `bottleneck` / `warnings` / `combined` | unset |
| `C2HLS_SKILL_CURATION_SECTOR` | `json_only` or `json_plus_llm` | unset |
| `C2HLS_SKILL_CURATION_INCLUDE_AVOIDS` | `1` (all_avoids_*) or `0` (no_avoids_*) | unset |
| `C2HLS_SKILL_PROMPT_MODE` | `llm_curated` | unset |
| `C2HLS_FORCE_SKILL_PROMPTS` | `1` | `0` |

Optional: `C2HLS_CURATION_MODEL` (defaults to codegen model).

---

## Launchers

```bash
# Dry-run all 15 configs (no Slurm)
./scripts/pc2/start_curated_skills_matrix.sh --dry-run

# Full sequential matrix (waits for empty squeue between waves)
./scripts/pc2/start_curated_skills_matrix.sh --auto-stop-on-complete

# Single wave only
./scripts/pc2/start_curated_skills_flash_wave.sh --focus bottleneck --stamp 20260622_120000 --auto-stop-on-complete

# Single variant (login node / debug)
python3 scripts/pc2/run_flash_curated_skills_batch.py --pc2 --variant all_avoids_json --focus bottleneck --dry-run
```

Resume: batch skips benches with existing `{bench}_multistep_results.json`.

Monitor: `tail -f artifacts/pc2/sessions/flash_curated_*/watch.log`

---

## 15-run checklist

| Wave | focus | Variants | Stamp (example) |
|------|-------|----------|-----------------|
| 1 | bottleneck | 5 | `{STAMP}_` + focus in dirname |
| 2 | warnings | 5 | same `{STAMP}` base |
| 3 | combined | 5 | same `{STAMP}` base |

---

## Related families (do not mix artifacts)

| Family | Doc |
|--------|-----|
| Legacy flash | `artifacts/pc2/flash_comparison_20260620.md` |
| New deterministic matrix | `artifacts/pc2/flash_new_skills_test_matrix.md` |
| **LLM-curated (this)** | this file |
