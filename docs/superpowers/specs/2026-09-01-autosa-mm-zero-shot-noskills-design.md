# Design: autosa_mm zero-shot vs no-skills vs with-skills ablation

**Date:** 2026-09-01  
**Status:** Approved (approach 1: `--flavor` on `start_autosa_mm_flow.sh`)  
**Scope:** Two new mm-flow arms. Do **not** rerun with-skills. Do not overwrite `20260830_mmflow`, locked `autosa_mm` / `autosa_mm_32x8`, family A/B/C artifacts, or the Aug 18 slide (4285 vs 4228).

## Goal

Measure DeepSeek-v4-flash on `autosa_mm` under three prompt regimes, same hardware lock as with-skills:

| Arm | LLM regime | Flow |
|-----|------------|------|
| **zero-shot** | HLS-engineer prompt only. One rewrite, then csim+csynth. | Flash only. No Phase B as an LLM job. No flash repair loop. No DSE. No stream. |
| **no-skills** | Same flow as with-skills, but **no skills JSON** at flash, DSE, or stream. | flash → DSE → stream. Keep PE-recipe text. Keep synth-report repairs. |
| **with-skills** | Frozen. Do not relaunch. | `scripts/pc2/start_autosa_mm_flow.sh` default (`20260830_mmflow`). |

Compare on **latency min, latency max, DSP** only. Do not use interval. Do not say “DSE” on slides.

## Locked hardware / model

- Model: DeepSeek-v4-flash
- Part: U280 `xcu280-fsvh2892-2L-e`
- Clock: 3.33 ns
- Cosim: off
- Seed: `related_work/benchmarks/autosa_ready/autosa_mm` (`plain.cpp`)
- `C2HLS_PHASEB_FROM_GOLD=0`

## Decisions (locked)

| Decision | Choice |
|----------|--------|
| Launcher | **1:** `--flavor zero_shot\|noskills` on `start_autosa_mm_flow.sh`. Default (omit / `--flavor skills`) is unchanged with-skills. |
| Not | Two new start scripts. Not HLSFactory `zero_shot_cosim` (wrong corpus; those paths still repair). |
| Zero-shot prompt | Existing `C2HLS_FLASH_OPT_PROMPT_MODE=zero_shot` + `Instruction_c2hls_zero_shot` / `q_optimize_zero_shot_direct`. Skip Phase B (`C2HLS_SKIP_PHASE_B=1`) so flash seeds from `plain.cpp`. `C2HLS_TURNS=1`. Failed csim is the result. |
| No-skills strip | 90-skill pack / `C2HLS_PACKAGED_SKILLS_JSON`, `C2HLS_FLASH_SKILL_ENTRIES_JSON` (no_RMW overlay), `post_flash_dse_pe_skill_entries.json`, `post_flash_stream_pe_io_skill_entries.json`. |
| No-skills keep | `format_recipe_prompt` PE-recipe block. DSE/stream **system** architecture text (flow, not a skills file). Repair rounds from synth/csim/architecture gates. |
| Worker variants | `autosa_zero_shot` and `autosa_noskills` so `configure_autosa_campaign_env` does not re-inject `aav_n_gf` skills. |
| Artifacts | Separate prefixes. Never write into `batch_parallel_autosa_mm_flow_aav_n_gf_20260830_mmflow`. |
| Jobs | Do not submit without `--endpoint-url`. |

## Flavor table

| Flavor | Variant | Job prefix | Artifact prefix | DSE | Stream | Skills JSON | PE recipe | Turns |
|--------|---------|------------|-----------------|-----|--------|-------------|-----------|-------|
| `skills` (default) | `autosa_aav_n_gf` | `mmflow` | `batch_parallel_autosa_mm_flow_aav_n_gf` | on | on | gemm_flatten_v1 + no_RMW + DSE/stream JSON | locked 16×4 (or `--pe-recipe autosa_mm_32x8`) | 4 |
| `noskills` | `autosa_noskills` | `mmns` | `batch_parallel_autosa_mm_flow_noskills` | on | on | none | same recipe as skills | 4 |
| `zero_shot` | `autosa_zero_shot` | `mmzs` | `batch_parallel_autosa_mm_flow_zero_shot` | off | off | none | omit (`--pe-recipe` rejected) | 1 |

## Env the flavors must set

**zero_shot**

- `C2HLS_SKIP_PHASE_B=1`
- `C2HLS_FLASH_OPT_PROMPT_MODE=zero_shot`
- `C2HLS_SKILL_MODE=skill_off`, `C2HLS_FORCE_SKILL_PROMPTS=0`
- unset `C2HLS_PACKAGED_SKILLS_JSON`, `C2HLS_FLASH_SKILL_ENTRIES_JSON`, `C2HLS_SKILL_PROMPT_MODE`, DSE/stream skill JSON
- `C2HLS_POST_FLASH_DSE=0`, `C2HLS_POST_FLASH_STREAM=0`
- `C2HLS_TURNS=1`
- `C2HLS_POST_FLASH_NO_SKILLS=1` (belt: if DSE were chained, inject nothing)

**noskills**

- `C2HLS_SKILL_MODE=skill_off`, `C2HLS_FORCE_SKILL_PROMPTS=0`
- unset the same skill JSON keys
- `C2HLS_POST_FLASH_NO_SKILLS=1` → `build_dse_skills_prompt_block` / `build_stream_skills_prompt_block` return empty (`skill_count=0`)
- `C2HLS_POST_FLASH_DSE=1`, `C2HLS_POST_FLASH_STREAM=1` (same as default)
- PE recipe still in the DSE/stream user prompt

**skills (default)**

- Unchanged: `C2HLS_SKILL_PROMPT_MODE=all_skills_avoids_global`, gemm_flatten_v1 pack, flash no_RMW overlay, DSE/stream skill JSON loaded.

## Out of scope

- Relaunching with-skills
- Changing the zero-shot prompt templates beyond wiring them for mm-flow
- HLSFactory / ChatHLS / MachSuite corpora
- Cosim
- Git commit unless the user asks
- Slide edits
