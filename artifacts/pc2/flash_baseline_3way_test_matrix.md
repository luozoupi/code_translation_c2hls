# Baseline 3-way flash test (hlsfactory_*)

| Field | Value |
|-------|-------|
| Orchestrator | `scripts/pc2/start_baseline_skills_3way_test.sh` |
| Skills file | `skills_ii_target_miss_solutions_added.json` (80 skills, baseline-first) |
| Benches | All 28 `hlsfactory_*` |
| Validation | **csim + csynth only** (`C2HLS_RUN_COSIM=0`) |
| Model | `mistralai/Devstral-2-123B-Instruct-2512` (default) |

## Variants (parallel — 3 independent sessions)

Each variant has its own GPU (`gpu_h100`) + Vitis (`normal`) Slurm jobs.

| Variant key | Session | Artifact prefix |
|-------------|---------|-----------------|
| `noskills_new` | `flash_noskills_new` | `flash_noskills_new_<stamp>` |
| `all_new_skills_avoids_global` | `flash_all_new_skills_avoids_global` | `flash_all_new_skills_avoids_global_<stamp>` |
| `all_new_skills_no_avoids_global` | `flash_all_new_skills_no_avoids_global` | `flash_all_new_skills_no_avoids_global_<stamp>` |

## Monitor

```bash
tail -f artifacts/pc2/sessions/flash_noskills_new/watch.log
tail -f artifacts/pc2/sessions/flash_all_new_skills_avoids_global/watch.log
tail -f artifacts/pc2/sessions/flash_all_new_skills_no_avoids_global/watch.log
```

## Stamp

`20260622_215520` (85-skill library with regression fixes)

## Artifact paths

- `artifacts/pc2/flash_noskills_new_20260622_215520/`
- `artifacts/pc2/flash_all_new_skills_avoids_global_20260622_215520/`
- `artifacts/pc2/flash_all_new_skills_no_avoids_global_20260622_215520/`

## Orchestrator log

`artifacts/pc2/sessions/baseline_3way_orchestrator_20260622_215520.log`
