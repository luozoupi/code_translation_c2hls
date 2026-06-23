# PC2 Flash Test Matrix — Legacy vs New Skills

This document tracks **two independent** flash experiment families. They must not share artifact directories, session IDs, or skills files.

---

## Families at a glance

| Family | Matrix ID | Skills file | Status |
|--------|-----------|-------------|--------|
| **Legacy** | `flash_legacy` | `skills.json` (+ mutable `skills/skills.json` merge) | Completed runs under `flash_*_20260620_*` |
| **New (ii-target-miss)** | `flash_new_skills_ii_target_miss` | frozen `(73skills).json` / `(90skills).json` | Runs under `flash_*_new_*` |

---

## Skills files

| File | Path | Skills | Used by |
|------|------|--------|---------|
| Legacy base | `hls_full_optimization_skills_schema_1_1_package/skills.json` | 55 | Legacy `flash_noskills`, `flash_skills`, `flash_all_skills_*` |
| Legacy extension | `hls_full_optimization_skills_schema_1_1_package/skills_extension.json` | 3 | **Not** loaded at runtime (analysis only) |
| **New library (73)** | `.../skills_ii_target_miss_solutions_added(73skills).json` | **73** (39 high, 18 medium, 16 avoid; **57** positive injected, U280) | No avoids (new) best stamp `20260621_075846`; Jun-21 matrix |
| **New library (90)** | `.../skills_ii_target_miss_solutions_added(90skills).json` | **90** (48 high, 18 medium, 24 avoid; **66** positive / **90** all injected) | All+avoids (new) best stamp `20260623_024548`; Jun-23+ runs |

New runs set (in batch script only — legacy scripts never set these):

- `C2HLS_PACKAGED_SKILLS_JSON` → `(73skills).json` or `(90skills).json` per variant (see `flash_new_skills_lib.skills_json_for_variant`)
- `C2HLS_PACKAGED_SKILLS_ONLY=1` → no merge with `skills/skills.json` or bootstrap prompts

---

## Legacy variants (do not re-use for new skills)

| Label | Session ID | Artifact prefix | Skills in prompt |
|-------|------------|-----------------|------------------|
| Noskills | `flash_noskills` | `flash_noskills_<stamp>` | None |
| Bn skills (2+2) | `flash_skills` | `flash_skills_<stamp>` | Top bottleneck: 2 positive + 2 avoid |
| All+avoids global | `flash_all_skills_avoids_global` | `flash_all_skills_avoids_global_<stamp>` | All 55 skills + avoids |
| All no-avoids global | `flash_all_skills_no_avoids_global` | `flash_all_skills_no_avoids_global_<stamp>` | All 55 positive only |

Reference results: `artifacts/pc2/flash_comparison_20260620.md` (legacy) · `artifacts/pc2/flash_comparison_20260621.md` (legacy vs new)

Launcher: `scripts/pc2/start_dual_flash_sessions.sh`, `start_dual_global_skills_flash_sessions.sh`

---

## New variants (`skills_ii_target_miss_solutions_added(73|90skills).json`)

| Label | Variant key | Session ID | Artifact prefix | Cell tag | How skills reach the agent |
|-------|-------------|------------|-----------------|----------|----------------------------|
| **Noskills_new** | `noskills_new` | `flash_noskills_new` | `flash_noskills_new_<stamp>` | `flash__noskills_new` | **None** (`C2HLS_FORCE_SKILL_PROMPTS=0`) |
| **Bn_skills_new_2_2** | `bn_skills_new_2_2` | `flash_bn_skills_new_2_2` | `flash_bn_skills_new_2_2_<stamp>` | `flash__bn_skills_new_2_2` | Bottleneck: **2** positive + **2** avoid |
| **Bn_skills_new_4_2** | `bn_skills_new_4_2` | `flash_bn_skills_new_4_2` | `flash_bn_skills_new_4_2_<stamp>` | `flash__bn_skills_new_4_2` | Bottleneck: **4** positive + **2** avoid |
| **Bn_skills_new_6_2** | `bn_skills_new_6_2` | `flash_bn_skills_new_6_2` | `flash_bn_skills_new_6_2_<stamp>` | `flash__bn_skills_new_6_2` | Bottleneck: **6** positive + **2** avoid |
| **flash_all_new_skills_avoids_global** | `all_new_skills_avoids_global` | `flash_all_new_skills_avoids_global` | `flash_all_new_skills_avoids_global_<stamp>` | `flash__all_new_skills_avoids_global` | Global: all positive + all avoid (`(90skills).json`, 90 injected) |
| **flash_all_new_skills_no_avoids_global** | `all_new_skills_no_avoids_global` | `flash_all_new_skills_no_avoids_global` | `flash_all_new_skills_no_avoids_global_<stamp>` | `flash__all_new_skills_no_avoids_global` | Global: positive only (`(73skills).json`, 57 injected) |

Bottleneck selection: match skills to **top bottleneck kind** from synthesis feedback; rank by library confidence/pass-rate.

Global selection: inject entire applicable library (filtered by Vitis version / FPGA); avoids appended when mode includes them.

---

## How to run (new matrix only)

```bash
# List variants
python3 scripts/pc2/run_flash_new_skills_batch.py --pc2 --list-variants

# Dry-run one variant
python3 scripts/pc2/run_flash_new_skills_batch.py --pc2 --variant bn_skills_new_4_2 --dry-run

# Start all 6 sessions (shared stamp, separate session IDs / artifact trees)
./scripts/pc2/start_new_skills_flash_sessions.sh --dry-run
./scripts/pc2/start_new_skills_flash_sessions.sh --auto-stop-on-complete

# Subset only
./scripts/pc2/start_new_skills_flash_sessions.sh --variants bn_skills_new_2_2,all_new_skills_no_avoids_global
```

Watch logs: `artifacts/pc2/sessions/<session_id>/watch.log`

Each run writes `manifest.json` + `matrix.json` under its artifact prefix with `matrix_family: flash_new_skills_ii_target_miss`.

---

## Run log (new matrix)

**Active stamp:** `20260621_020847` — submitted 2026-06-21, auto-stop on worker success. Compute manually nudged after 30m watch interval delayed submit.

| Stamp | Variant | Session | Status | Artifact root | Notes |
|-------|---------|---------|--------|---------------|-------|
| `20260621_020847` | `noskills_new` | `flash_noskills_new` | done (27/28) | `flash_noskills_new_20260621_020847` | |
| `20260621_020847` | `bn_skills_new_2_2` | `flash_bn_skills_new_2_2` | done (27/28) | `flash_bn_skills_new_2_2_20260621_020847` | |
| `20260621_020847` | `bn_skills_new_4_2` | `flash_bn_skills_new_4_2` | done (27/28) | `flash_bn_skills_new_4_2_20260621_020847` | |
| `20260621_020847` | `bn_skills_new_6_2` | `flash_bn_skills_new_6_2` | done (27/28) | `flash_bn_skills_new_6_2_20260621_020847` | resubmit 12h + resume |
| `20260621_020847` | `all_new_skills_avoids_global` | `flash_all_new_skills_avoids_global` | done (27/28) | `flash_all_new_skills_avoids_global_20260621_020847` | resubmit 12h + resume |
| `20260621_020847` | `all_new_skills_no_avoids_global` | `flash_all_new_skills_no_avoids_global` | done (27/28) | `flash_all_new_skills_no_avoids_global_20260621_020847` | |

---

## Run log — round 2 (rerun, fresh stamp)

Launcher: `scripts/pc2/start_flash_rerun_round2.sh --auto-stop-on-complete`

- **Watch interval:** 60s (`PC2_WATCH_INTERVAL_SEC=60`, not 30min)
- **Compute walltime:** 12h (`PC2_FORCE_WALLTIME`)
- **Prior runs preserved:** `*_20260620_*`, `*_20260621_020847` — new artifacts use a new `<stamp>` only

| Stamp | Scope | Status |
|-------|-------|--------|
| `20260621_075846` | legacy 4 + new 6 (10 sessions) | submitted |

---

## Isolation checklist

- [x] Separate session IDs (`flash_*_new*` vs `flash_noskills` / `flash_skills` / `flash_all_skills_*`)
- [x] Separate artifact prefixes (`flash_*_new*` / `flash_bn_skills_new_*`)
- [x] Separate cell setup tags (`flash__*_new*`)
- [x] New skills file loaded only when new batch scripts set env vars
- [x] Legacy batch scripts unchanged; default `make_default_library()` still uses `skills.json`
- [x] Results comparison doc for new matrix → `artifacts/pc2/flash_comparison_20260621.md`
