# Team reference artifacts (not our PC2 runs)

Files here came from the UMN team machine or team matrix exports.
They may contain paths like `/home/luo00466/...` or `/mnt/e/courses/...`.
Do **not** merge these into our Devstral baseline JSONL without relabeling.

| Path | What it is |
|------|------------|
| `hlsfactory_direct_reference_merged_cosim3600_20260531(2).jsonl` | Team naive-baseline csynth/csim/cosim JSONL (`run_hlsfactory_direct_reference.py`, luo00466 paths) |
| `u280__all_models_schema.jsonl` | Team U280 matrix export (Sonnet/Opus, multistep + flash) |
| `hlsfactory_baseline_plus_ai_u280_20260616.jsonl` | Accidental merge of our baseline + team `u280__all_models_schema` |
| `sonnet__flash__skills/` | Copy of team matrix gemm cell (`results_matrix_u280_fullcosim`, `/mnt/e/...` paths) |
| `team_gemm_functional_baseline/` | Team functional m_axi gemm kernel used for csynth comparison |

Our PC2 artifacts live in `misc/` (baseline JSONL, Devstral flash JSONL) and `artifacts/pc2/`.
