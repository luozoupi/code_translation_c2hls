# autosa_mm tests catalog (knobs, results, LLM history index)

Generated: 2026-09-11T00:06:10.215079+00:00

This file is the **grep-able map**. Full LLM transcripts are **not** copied here (too large).
Each call is indexed in [`2026-09-10-autosa-mm-llm-calls.jsonl`](2026-09-10-autosa-mm-llm-calls.jsonl)
with `file` + `msg_index` pointing at the campaign `*_history.json`.

Machine tables: [`2026-09-10-autosa-mm-tests-catalog.json`](2026-09-10-autosa-mm-tests-catalog.json),
[`.csv`](2026-09-10-autosa-mm-tests-catalog.csv).
Pytest knobs: [`2026-09-10-autosa-mm-pytest-knobs.json`](2026-09-10-autosa-mm-pytest-knobs.json).
Regenerate: `python scripts/pc2/export_autosa_mm_tests_catalog.py`.

**Metrics:** latency min / max + DSP. Do not quote interval as the result.
**Two leaderboards (never mix):** iso-compute ~320 DSP vs AutoSA rank-1 **4228 / 320**; spend-DSP champion **940 / 5344** (`pe16_20260904_131622`).

## 1. Knob legend

| Knob | Env / campaign field | Aim | Default |
|---|---|---|---|
| 3-stage flow | `C2HLS_AUTOSA_FLOW=1` + DSE + stream | flash then compute-rewrite then hide-load-store | off except `start_autosa_mm_flow.sh` |
| Enforcement | `C2HLS_ENFORCEMENT=1` | ping-pong + DATAFLOW judged on code+csynth | off; flow sets 0 |
| Keep-flash | `keep_flash` + overlay skills | wrap must not drop flash LANES / jack latency > flash×1.10 | enforcement launcher |
| Skip-flash | `C2HLS_SKIP_FLASH=1` `--seed-flash DIR` | start enforcement from a frozen flash_opt (no flash LLM) | off |
| Explicit ping-pong | flow-gates | require `buf[2]` + `t&1`, B loaded once outside; DATAFLOW+arrays-inside is fail | after 101836 |
| Skills pack | `C2HLS_PACKAGED_SKILLS_JSON` | 90-skill GEMM flatten vs distilled on-chip vs keep-flash overlay | gemm_flatten_v1 |
| Prompt mode | `C2HLS_SKILL_PROMPT_MODE` | how skills are injected | `all_skills_avoids_global` |
| DSP floor | `C2HLS_FLASH_MIN_DSP` | reject flash if DSP below cutoff | off |
| PE_BLK | `C2HLS_FLASH_PE_BLK` 16/32/64 | pin PE width; spend-chip | off |
| Tile ping-pong | `C2HLS_FLASH_TILE_PP=1` | in-GEMM tile PP in **flash** (not enforcement) | off |
| ROW_UF | `C2HLS_FLASH_ROW_UF` | unroll I in one tile | off |
| On-chip pack | `C2HLS_FLASH_ONCHIP=1` | distilled 940-class skills, not 90 dump | off |
| Zero-shot | `C2HLS_MM_FLOW_FLAVOR=zero_shot` | no skills | off |
| No-skills | `C2HLS_MM_FLOW_FLAVOR=noskills` | 3-stage without 90-skill dump | off |
| Skip phase B | `C2HLS_SKIP_PHASE_B=1` | start from gold/plain, skip translator | enforcement seed path |
| DSE / stream | `C2HLS_POST_FLASH_DSE` / `_STREAM` | post-flash systolic stages (say compute rewrite / hide load-store on slides) | flow on; flash-* launchers off |

## 2. Where each LLM call lives

| Stage | File under the variant cell | What it contains |
|---|---|---|
| Phase B + flash (+ enforcement repairs) | `autosa_mm_history.json` → `messages[]` | Full chat: system, user, assistant. `llm_usage.calls` / tokens. |
| Flash skill injection | `autosa_mm_flash_skills.json` | Which skill IDs were stuffed into the flash prompt (`injected_prompt_text`). |
| Compute rewrite | `autosa_mm_dse_history.json` | Separate 3-message DSE call. |
| Hide load/store | `autosa_mm_stream_history.json` | Separate stream call. |
| Enforcement rounds | `autosa_mm_enforcement.json` → `attempts[]` | Per-round judge_before/after, latency, pass/fail. Repair *code* is in history.json, not duplicated here. |
| Orchestrator checkpoint | `pipelined/orchestrator_state.json` | Resume state; often duplicates history. Huge. Prefer `*_history.json`. |
| Drain log | `flow/gpu_drain.log` | Skip-flash accepted, 502s, round progress. |

Index row example: `jq 'select(.campaign|test("101836"))' docs/pc2/2026-09-10-autosa-mm-llm-calls.jsonl`

## 3. Result table (one row per campaign cell)

| Family | Stamp / name | Status | Flash min/max / DSP (before wrap) | After (enf or selected) min/max / DSP | Enf applied | LLM msgs | Frozen |
|---|---|---|---|---|---|---:|:---:|
| mmflow_32x8 | `batch_parallel_autosa_mm_32x8_flow_aav_n_gf_20260830_mm32x8` | complete | 12456/12456 / 320 | 4583/4583 / 1280 |  | 15 | yes |
| flash_onchip | `batch_parallel_autosa_mm_block_sparse_flash_onchip_t7_20260908_010347_autosa_mm_block_sparse` | complete | 937/937 / 5344 | 937/937 / 5344 |  | 7 |  |
| other_kernel_seed_synth | `batch_parallel_autosa_mm_block_sparse_seed_synth_20260908_124628_autosa_mm_block_sparse` | complete | — | — |  | 0 |  |
| flash_onchip | `batch_parallel_autosa_mm_catapult_flash_onchip_t7_20260908_010347_autosa_mm_catapult` | complete | 744/744 / 6144 | 744/744 / 6144 |  | 11 |  |
| other_kernel_seed_synth | `batch_parallel_autosa_mm_catapult_seed_synth_20260908_124628_autosa_mm_catapult` | complete | — | — |  | 0 |  |
| enforcement_pingpong | `batch_parallel_autosa_mm_enf_aav_n_gf_20260829_095725` | aborted | — | — |  | 0 |  |
| enforcement_pingpong | `batch_parallel_autosa_mm_enf_aav_n_gf_20260829_123043` | complete | 12893/12893 / 320 (opt file overwritten) | 12893/12893 / 320 | True | 9 | yes |
| enforcement_pingpong | `batch_parallel_autosa_mm_enf_aav_n_gf_20260909_085740` | complete | 4808/4808 / 318 (opt file overwritten) | 12642/12642 / 318 | True | 10 | yes |
| enforcement_pingpong | `batch_parallel_autosa_mm_enf_aav_n_gf_20260909_101836` | complete | 5214/5214 / 320 (opt file overwritten) | 5542/5542 / 320 | True | 43 | yes |
| enforcement_pingpong | `batch_parallel_autosa_mm_enf_aav_n_gf_20260909_114519` | complete | 41128/41128 / 40 (opt file overwritten) | 39116/39116 / 42 | True | 9 | yes |
| enforcement_pingpong | `batch_parallel_autosa_mm_enf_aav_n_gf_20260909_121934` | complete | 41126/41126 / 40 (opt file overwritten) | 34233/68113 / 40 | True | 14 | yes |
| enforcement_pingpong | `batch_parallel_autosa_mm_enf_aav_n_gf_20260909_175131` | complete | 139484/139484 / 10 | 139484/139484 / 10 |  | 32 | yes |
| enforcement_pingpong | `batch_parallel_autosa_mm_enf_aav_n_gf_20260910_234214` | running | 139484/139484 / 10 | — |  | 0 |  |
| flash_dataflow_prompt | `batch_parallel_autosa_mm_flash_dataflow_20260904_083918` | complete | 82025/82025 / 20 | 82025/82025 / 20 |  | 8 |  |
| flash_dsp_floor | `batch_parallel_autosa_mm_flash_dsp1000_20260903_121153` | complete | 13549/13549 / 768 | 13549/13549 / 768 |  | 10 |  |
| flash_dsp_floor | `batch_parallel_autosa_mm_flash_dsp300_20260903_113114` | running | — | — |  | 0 |  |
| flash_dsp_floor | `batch_parallel_autosa_mm_flash_dsp300_20260903_113128` | complete | 15708/15708 / 384 | 15708/15708 / 384 |  | 7 |  |
| flash_dsp_floor | `batch_parallel_autosa_mm_flash_dsp500_20260904_085931` | complete | 9423/9423 / 1344 | 9423/9423 / 1344 |  | 9 |  |
| flash_compute_ii | `batch_parallel_autosa_mm_flash_dsp500_computeii_20260904_120524` | complete | 1261/1261 / 2672 | 1261/1261 / 2672 |  | 8 |  |
| flash_compute_ii | `batch_parallel_autosa_mm_flash_dsp500_computeii_20260904_131523` | aborted | — | — |  | 0 |  |
| flash_pe_blk | `batch_parallel_autosa_mm_flash_dsp500_pe16_20260904_131622` | complete | 940/940 / 5344 | 940/940 / 5344 |  | 11 | yes |
| flash_tile_pp | `batch_parallel_autosa_mm_flash_dsp500_pe16_tilepp_20260905_001958` | running | — | — |  | 0 |  |
| flash_tile_pp | `batch_parallel_autosa_mm_flash_dsp500_pe16_tilepp_20260905_002017` | complete | 3817/3817 / 5120 | 3817/3817 / 5120 |  | 8 |  |
| flash_pe_blk | `batch_parallel_autosa_mm_flash_dsp500_pe32_20260904_133157` | complete | 9991/9991 / 782 | 9991/9991 / 782 |  | 10 |  |
| flash_pe_blk | `batch_parallel_autosa_mm_flash_dsp500_pe64_20260904_134954` | complete | 1733/1733 / 1272 | 1733/1733 / 1272 |  | 9 |  |
| flash_wide_io | `batch_parallel_autosa_mm_flash_dsp500_wideio_20260904_100437` | complete | 2760/2760 / 636 | 2760/2760 / 636 |  | 9 |  |
| flash_onchip | `batch_parallel_autosa_mm_flash_onchip_t7_20260906_104715` | running | — | — |  | 0 |  |
| flash_onchip | `batch_parallel_autosa_mm_flash_onchip_t7_20260906_104733` | complete | 871/871 / 5088 | 871/871 / 5088 |  | 9 |  |
| flash_onchip | `batch_parallel_autosa_mm_flash_onchip_t7_20260906_123340` | complete | 1002/1002 / 5344 | 1002/1002 / 5344 |  | 7 |  |
| flash_row_uf | `batch_parallel_autosa_mm_flash_rowuf64_20260904_012045` | running | — | — |  | 0 |  |
| flash_row_uf | `batch_parallel_autosa_mm_flash_rowuf64_20260904_012102` | complete | 12456/12456 / 320 | 12456/12456 / 320 |  | 7 |  |
| mmflow_3stage | `batch_parallel_autosa_mm_flow_aav_n_gf_20260830_mmflow` | complete | 139484/139484 / 10 | 4292/4292 / 320 |  | 16 | yes |
| mmflow_3stage | `batch_parallel_autosa_mm_flow_aav_n_gf_task5_mmflow_dryrun` | running | — | — |  | 0 |  |
| ablation_noskills | `batch_parallel_autosa_mm_flow_noskills_20260901_mmns` | complete | — | — |  | 0 |  |
| ablation_noskills | `batch_parallel_autosa_mm_flow_noskills_20260902_mmns` | complete | 1056894/1056894 / 11 | 4216/4216 / 320 |  | 20 |  |
| ablation_zero_shot | `batch_parallel_autosa_mm_flow_zero_shot_20260901_mmzs` | aborted | — | — |  | 0 |  |
| ablation_zero_shot | `batch_parallel_autosa_mm_flow_zero_shot_20260901_mmzs2` | complete | — | — |  | 0 |  |
| ablation_zero_shot | `batch_parallel_autosa_mm_flow_zero_shot_20260902_mmzs` | complete | 40454/42758 / 320 | 40454/42758 / 320 |  | 4 |  |
| gap_flash | `batch_parallel_autosa_mm_gap_20260818_mm_gap_flash` | complete | — | — |  | 0 |  |
| gap_flash | `batch_parallel_autosa_mm_gap_20260818_mm_gap_flash_abi` | complete | 1056932/1056932 / 3 | 1056932/1056932 / 3 |  | 8 |  |
| gap_flash | `batch_parallel_autosa_mm_gap_ds_v4f_20260818_mm_gap_flash_ds_v4f` | complete | 149082/149082 / 6 | 4285/4285 / 320 |  | 14 |  |
| gap_flash | `batch_parallel_autosa_mm_gap_ds_v4f_aav_n_20260819_mm_gap_flash_ds_v4f_aav_n` | complete | 73897/73897 / 20 | 73897/73897 / 20 |  | 8 |  |
| gap_flash | `batch_parallel_autosa_mm_gap_ds_v4f_aav_n_gf_20260823_mm_gap_flash_ds_v4f_aav_n_gf` | complete | 24745/24745 / 80 | 4299/4299 / 352 |  | 14 |  |
| flash_onchip | `batch_parallel_autosa_mm_getting_started_flash_onchip_t7_20260908_010347_autosa_mm_getting_started` | complete | 900/900 / 5088 | 900/900 / 5088 |  | 7 |  |
| other_kernel_seed_synth | `batch_parallel_autosa_mm_getting_started_seed_synth_20260908_124628_autosa_mm_getting_started` | complete | — | — |  | 0 |  |
| flash_onchip | `batch_parallel_autosa_mm_hbm_flash_onchip_t7_20260908_010347_autosa_mm_hbm` | complete | 938/938 / 5344 | 938/938 / 5344 |  | 10 |  |
| other_kernel_seed_synth | `batch_parallel_autosa_mm_hbm_seed_synth_20260908_124628_autosa_mm_hbm` | complete | — | — |  | 0 |  |
| flash_onchip | `batch_parallel_autosa_mm_hcl_flash_onchip_t7_20260907_dryrun` | running | — | — |  | 0 |  |
| flash_onchip | `batch_parallel_autosa_mm_hcl_flash_onchip_t7_20260908_002240` | complete | 906/906 / 5344 | 906/906 / 5344 |  | 7 |  |
| flash_onchip | `batch_parallel_autosa_mm_hcl_flash_onchip_t7_20260908_004300` | complete | 938/938 / 5344 | 938/938 / 5344 |  | 7 |  |
| flash_onchip | `batch_parallel_autosa_mm_hcl_intel_flash_onchip_t7_20260908_010347_autosa_mm_hcl_intel` | complete | 1032/1032 / 5120 | 1032/1032 / 5120 |  | 7 |  |
| flash_onchip | `batch_parallel_autosa_mm_hcl_intel_flash_onchip_t7_20260908_045607_autosa_mm_hcl_intel` | complete | 1002/1002 / 5344 | 1002/1002 / 5344 |  | 7 |  |
| other_kernel_seed_synth | `batch_parallel_autosa_mm_hcl_intel_seed_synth_20260908_124628_autosa_mm_hcl_intel` | complete | — | — |  | 0 |  |
| other_kernel_seed_synth | `batch_parallel_autosa_mm_hcl_seed_synth_20260908_124628_autosa_mm_hcl` | complete | — | — |  | 0 |  |
| other_kernel_seed_synth | `batch_parallel_autosa_mm_hcl_seed_synth_20260908_dryseed_hcl` | running | — | — |  | 0 |  |
| flash_onchip | `batch_parallel_autosa_mm_int16_flash_onchip_t7_20260908_010347_autosa_mm_int16` | complete | 863/863 / 1024 | 863/863 / 1024 |  | 7 |  |
| flash_onchip | `batch_parallel_autosa_mm_int16_flash_onchip_t7_20260908_045550` | running | — | — |  | 0 |  |
| flash_onchip | `batch_parallel_autosa_mm_int16_flash_onchip_t7_20260908_045607_autosa_mm_int16` | complete | 381/381 / 8192 | 381/381 / 8192 |  | 8 |  |
| other_kernel_seed_synth | `batch_parallel_autosa_mm_int16_seed_synth_20260908_124628_autosa_mm_int16` | complete | — | — |  | 0 |  |
| flash_onchip | `batch_parallel_autosa_mm_intel_flash_onchip_t7_20260908_010347_autosa_mm_intel` | complete | 938/938 / 5344 | 938/938 / 5344 |  | 7 |  |
| other_kernel_seed_synth | `batch_parallel_autosa_mm_intel_seed_synth_20260908_124628_autosa_mm_intel` | complete | — | — |  | 0 |  |
| wave1_other_kernels | `batch_parallel_autosa_wave1_aav_n_gf_20260825_062035` | running | — | — |  | 0 |  |
| wave1_other_kernels | `batch_parallel_autosa_wave1_aav_n_gf_20260825_wave1_aav_n_gf` | complete | 32916/32916 / 24 | 8351/8351 / 96 |  | 74 |  |
| wave1_other_kernels | `batch_parallel_autosa_wave1_aav_n_gf_20260825_wave1_aav_n_gf` | complete | 12588/12588 / 352 | 4525/4525 / 640 |  | 74 |  |
| wave1_other_kernels | `batch_parallel_autosa_wave1_aav_n_gf_20260825_wave1_aav_n_gf` | complete | 33357/33357 / 40 | 4294/4294 / 320 |  | 74 |  |
| wave1_other_kernels | `batch_parallel_autosa_wave1_aav_n_gf_20260825_wave1_aav_n_gf` | complete | 35085/35085 / 48 | 4292/4292 / 320 |  | 74 |  |
| wave1_other_kernels | `batch_parallel_autosa_wave1_aav_n_gf_20260825_wave1_aav_n_gf` | complete | 4509/4509 / 64 | 4280/4280 / 64 |  | 74 |  |
| wave1_other_kernels | `batch_parallel_autosa_wave1_aav_n_gf_20260825_wave1_aav_n_gf` | complete | 29457/29457 / 64 | 4525/4525 / 640 |  | 74 |  |
| family_c_io_mesh | `compact_pe_io_search` | unknown | — | — |  | 0 |  |
| family_c_io_mesh | `compact_pe_io_search_20260831_io` | unknown | — | — |  | 0 |  |
| family_c_io_mesh | `compact_pe_io_search_20260831_io2` | unknown | — | — |  | 0 | yes |
| family_b_pack | `compact_pe_pack_search` | unknown | — | — |  | 0 |  |
| family_b_pack | `compact_pe_pack_search_20260831_pack` | unknown | — | — |  | 0 |  |
| family_a_pe_search | `compact_pe_search` | unknown | — | — |  | 0 |  |
| family_a_pe_search | `compact_pe_search_20260830_pesearch` | unknown | — | — |  | 0 |  |
| family_a_pe_search | `compact_pe_search_20260831_mesh` | unknown | — | — |  | 0 |  |
| manual_handwritten | `manual_mmflow_pe_pp` | manual | — | 4688/4688 / 352 |  | 0 |  |
| manual_handwritten | `manual_pe16_tile_pp` | manual | — | 1074/1074 / 5344 |  | 0 |  |
| manual_handwritten | `manual_mm_lcst_tile_pp2` | manual | — | 4662/4662 / 352 |  | 0 |  |
| manual_handwritten | `manual_rank1_shaped` | manual | — | — |  | 0 |  |

## 4. Campaign details

### `batch_parallel_autosa_mm_32x8_flow_aav_n_gf_20260830_mm32x8`

- **Family / aim:** mmflow_32x8 — Same 3-stage flow with locked 32x8 PE recipe.
- **Status:** complete  **Frozen:** True
- **Created / done:** 2026-08-30T13:10:07.301833+00:00 / 2026-08-30T13:24:07.901449+00:00
- **Path:** `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_32x8_flow_aav_n_gf_20260830_mm32x8`
- **Knobs:** `{"autosa_flow": true, "benches": "autosa_mm", "dse": true, "enforcement": false, "latency_opt": false, "model": "deepseek-v4-flash", "pe_recipe": "autosa_mm_32x8", "post_flash_dse": true, "post_flash_stream": true, "skill_prompt_mode": "all_skills_avoids_global", "skills_pack": "gemm_flatten_v1", "stream": true, "synth_timeout": 3600, "turns": 4, "variant": "autosa_aav_n_gf", "workflow": "autosa_flash"}`
- **Cell:** `deepseek-v4-flash__flash__autosa__aav_n_gf`
  - flash: **12456–12456** cycles, DSP **320**
  - dse: **9782–9782** cycles, DSP **1344**
  - stream: **4583–4583** cycles, DSP **1280**
  - selected: **4583–4583** cycles, DSP **1280**
  - flash skills injected=129 routed=None pack_sha=847494f44a9b4a6e `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_32x8_flow_aav_n_gf_20260830_mm32x8/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_flash_skills.json`
  - code flash_opt: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_32x8_flow_aav_n_gf_20260830_mm32x8/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_flash_opt.cpp`
  - code selected: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_32x8_flow_aav_n_gf_20260830_mm32x8/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_selected.cpp`
  - code dse: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_32x8_flow_aav_n_gf_20260830_mm32x8/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_dse.cpp`
  - code stream: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_32x8_flow_aav_n_gf_20260830_mm32x8/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_stream.cpp`
- **LLM flash_combined:** 9 msgs, calls=0, tokens=0 `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_32x8_flow_aav_n_gf_20260830_mm32x8/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_history.json`
- **LLM dse:** 3 msgs, calls=None, tokens=None `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_32x8_flow_aav_n_gf_20260830_mm32x8/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_dse_history.json`
- **LLM stream:** 3 msgs, calls=None, tokens=None `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_32x8_flow_aav_n_gf_20260830_mm32x8/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_stream_history.json`

### `batch_parallel_autosa_mm_block_sparse_flash_onchip_t7_20260908_010347_autosa_mm_block_sparse`

- **Family / aim:** flash_onchip — Flash-only: distilled on-chip GEMM pack (940-class), not 90-skills.
- **Status:** complete  **Frozen:** False
- **Created / done:** 2026-09-08T01:04:38.760932+00:00 / 2026-09-08T01:21:49.178536+00:00
- **Path:** `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_block_sparse_flash_onchip_t7_20260908_010347_autosa_mm_block_sparse`
- **Knobs:** `{"autosa_flow": true, "benches": "autosa_mm_block_sparse", "dse": "0", "enforcement": false, "flash_min_dsp": 500, "flash_onchip": 1, "flash_pe_blk": 16, "flash_skill_bin": "onchip", "model": "deepseek-v4-flash", "packaged_skills_json": "/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/hls_full_optimization_skills_schema_1_1_package/flash_onchip_wide_gemm_skill_entries.json", "post_flash_dse": false, "post_flash_stream": false, "stream": "0", "synth_timeout": 14400, "turns": 7, "variant": "autosa_onchip_gemm", "workflow": "autosa_flash"}`
- **Cell:** `deepseek-v4-flash__flash__autosa__onchip_gemm`
  - flash: **937–937** cycles, DSP **5344**
  - selected: **937–937** cycles, DSP **5344**
  - flash skills injected=9 routed=None pack_sha=5c7ee088b5820199 `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_block_sparse_flash_onchip_t7_20260908_010347_autosa_mm_block_sparse/variants/autosa_onchip_gemm/autosa_mm_block_sparse/deepseek-v4-flash__flash__autosa__onchip_gemm/autosa_mm_block_sparse_flash_skills.json`
  - code flash_opt: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_block_sparse_flash_onchip_t7_20260908_010347_autosa_mm_block_sparse/variants/autosa_onchip_gemm/autosa_mm_block_sparse/deepseek-v4-flash__flash__autosa__onchip_gemm/autosa_mm_block_sparse_flash_opt.cpp`
  - code selected: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_block_sparse_flash_onchip_t7_20260908_010347_autosa_mm_block_sparse/variants/autosa_onchip_gemm/autosa_mm_block_sparse/deepseek-v4-flash__flash__autosa__onchip_gemm/autosa_mm_block_sparse_selected.cpp`
- **LLM flash_combined:** 7 msgs, calls=0, tokens=0 `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_block_sparse_flash_onchip_t7_20260908_010347_autosa_mm_block_sparse/variants/autosa_onchip_gemm/autosa_mm_block_sparse/deepseek-v4-flash__flash__autosa__onchip_gemm/autosa_mm_block_sparse_history.json`

### `batch_parallel_autosa_mm_block_sparse_seed_synth_20260908_124628_autosa_mm_block_sparse`

- **Family / aim:** other_kernel_seed_synth — Seed csynth of other AutoSA mm-family kernels (no LLM).
- **Status:** complete  **Frozen:** False
- **Created / done:** 2026-09-08T12:46:33.134623+00:00 / 2026-09-08T12:48:51.883130+00:00
- **Path:** `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_block_sparse_seed_synth_20260908_124628_autosa_mm_block_sparse`
- **Knobs:** `{"autosa_flow": true, "benches": "autosa_mm_block_sparse", "dse": "0", "enforcement": false, "model": "none", "post_flash_dse": false, "post_flash_stream": false, "skip_phase_b": true, "stream": "0", "synth_timeout": 14400, "turns": 1, "variant": "autosa_gold", "workflow": "autosa_gold"}`
- **Cell:** `batch_parallel_autosa_mm_block_sparse_seed_synth_20260908_124628_autosa_mm_block_sparse`

### `batch_parallel_autosa_mm_catapult_flash_onchip_t7_20260908_010347_autosa_mm_catapult`

- **Family / aim:** flash_onchip — Flash-only: distilled on-chip GEMM pack (940-class), not 90-skills.
- **Status:** complete  **Frozen:** False
- **Created / done:** 2026-09-08T01:04:31.300724+00:00 / 2026-09-08T01:19:51.850191+00:00
- **Path:** `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_catapult_flash_onchip_t7_20260908_010347_autosa_mm_catapult`
- **Knobs:** `{"autosa_flow": true, "benches": "autosa_mm_catapult", "dse": "0", "enforcement": false, "flash_min_dsp": 500, "flash_onchip": 1, "flash_pe_blk": 32, "flash_skill_bin": "onchip", "model": "deepseek-v4-flash", "packaged_skills_json": "/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/hls_full_optimization_skills_schema_1_1_package/flash_onchip_wide_gemm_skill_entries.json", "post_flash_dse": false, "post_flash_stream": false, "stream": "0", "synth_timeout": 14400, "turns": 7, "variant": "autosa_onchip_gemm", "workflow": "autosa_flash"}`
- **Cell:** `deepseek-v4-flash__flash__autosa__onchip_gemm`
  - flash: **744–744** cycles, DSP **6144**
  - selected: **744–744** cycles, DSP **6144**
  - flash skills injected=9 routed=None pack_sha=5c7ee088b5820199 `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_catapult_flash_onchip_t7_20260908_010347_autosa_mm_catapult/variants/autosa_onchip_gemm/autosa_mm_catapult/deepseek-v4-flash__flash__autosa__onchip_gemm/autosa_mm_catapult_flash_skills.json`
  - code flash_opt: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_catapult_flash_onchip_t7_20260908_010347_autosa_mm_catapult/variants/autosa_onchip_gemm/autosa_mm_catapult/deepseek-v4-flash__flash__autosa__onchip_gemm/autosa_mm_catapult_flash_opt.cpp`
  - code selected: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_catapult_flash_onchip_t7_20260908_010347_autosa_mm_catapult/variants/autosa_onchip_gemm/autosa_mm_catapult/deepseek-v4-flash__flash__autosa__onchip_gemm/autosa_mm_catapult_selected.cpp`
- **LLM flash_combined:** 11 msgs, calls=0, tokens=0 `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_catapult_flash_onchip_t7_20260908_010347_autosa_mm_catapult/variants/autosa_onchip_gemm/autosa_mm_catapult/deepseek-v4-flash__flash__autosa__onchip_gemm/autosa_mm_catapult_history.json`

### `batch_parallel_autosa_mm_catapult_seed_synth_20260908_124628_autosa_mm_catapult`

- **Family / aim:** other_kernel_seed_synth — Seed csynth of other AutoSA mm-family kernels (no LLM).
- **Status:** complete  **Frozen:** False
- **Created / done:** 2026-09-08T12:46:30.960917+00:00 / 2026-09-08T12:48:51.997068+00:00
- **Path:** `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_catapult_seed_synth_20260908_124628_autosa_mm_catapult`
- **Knobs:** `{"autosa_flow": true, "benches": "autosa_mm_catapult", "dse": "0", "enforcement": false, "model": "none", "post_flash_dse": false, "post_flash_stream": false, "skip_phase_b": true, "stream": "0", "synth_timeout": 14400, "turns": 1, "variant": "autosa_gold", "workflow": "autosa_gold"}`
- **Cell:** `batch_parallel_autosa_mm_catapult_seed_synth_20260908_124628_autosa_mm_catapult`

### `batch_parallel_autosa_mm_enf_aav_n_gf_20260829_095725`

- **Family / aim:** enforcement_pingpong — Force explicit ping-pong + DATAFLOW on a flash kernel; judge code+csynth; keep-flash must not jack latency.
- **Status:** aborted  **Frozen:** False
- **Created / done:** 2026-08-29T09:57:26.109934+00:00 / None
- **Path:** `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_enf_aav_n_gf_20260829_095725`
- **Knobs:** `{"benches": "autosa_mm", "dse": false, "enforcement": true, "enforcement_rounds": 20, "latency_opt": false, "model": "deepseek-v4-flash", "skill_prompt_mode": "all_skills_avoids_global", "skills_pack": "gemm_flatten_v1", "stream": false, "turns": 4, "variant": "autosa_aav_n_gf", "workflow": "autosa_flash"}`

### `batch_parallel_autosa_mm_enf_aav_n_gf_20260829_123043`

- **Family / aim:** enforcement_pingpong — Force explicit ping-pong + DATAFLOW on a flash kernel; judge code+csynth; keep-flash must not jack latency.
- **Status:** complete  **Frozen:** True
- **Created / done:** 2026-08-29T12:30:43.609655+00:00 / 2026-08-29T12:36:57.533153+00:00
- **Path:** `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_enf_aav_n_gf_20260829_123043`
- **Knobs:** `{"benches": "autosa_mm", "dse": false, "enforcement": true, "enforcement_rounds": 20, "latency_opt": false, "model": "deepseek-v4-flash", "skill_prompt_mode": "all_skills_avoids_global", "skills_pack": "gemm_flatten_v1", "stream": false, "synth_timeout": 3600, "turns": 4, "variant": "autosa_aav_n_gf", "workflow": "autosa_flash"}`
- **Cell:** `deepseek-v4-flash__flash__autosa__aav_n_gf`
  - flash: **12893–12893** cycles, DSP **320**
  - selected: **12893–12893** cycles, DSP **320**
  - enforcement: applied=True passed=True rounds=1/20 overlap=True reason=already ping-pong DATAFLOW
  - flash skills injected=129 routed=None pack_sha=847494f44a9b4a6e `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_enf_aav_n_gf_20260829_123043/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_flash_skills.json`
  - code flash_opt: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_enf_aav_n_gf_20260829_123043/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_flash_opt.cpp`
  - code selected: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_enf_aav_n_gf_20260829_123043/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_selected.cpp`
- **LLM flash_combined:** 9 msgs, calls=2, tokens=64531 `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_enf_aav_n_gf_20260829_123043/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_history.json`

### `batch_parallel_autosa_mm_enf_aav_n_gf_20260909_085740`

- **Family / aim:** enforcement_pingpong — Force explicit ping-pong + DATAFLOW on a flash kernel; judge code+csynth; keep-flash must not jack latency.
- **Status:** complete  **Frozen:** True
- **Created / done:** 2026-09-09T08:58:13.150189+00:00 / 2026-09-09T09:10:19.568598+00:00
- **Path:** `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_enf_aav_n_gf_20260909_085740`
- **Knobs:** `{"benches": "autosa_mm", "dse": false, "enforcement": true, "enforcement_rounds": 20, "latency_opt": false, "model": "deepseek-v4-flash", "note": "New LLM C2HLS_ENFORCEMENT with csynth overlap judge (tile loop + arrays inside + latency \u2248 max(rooms)). Does not overwrite 12893 syntax-judge campaign.", "overlap_judge": true, "packaged_skills_json": "/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/hls_full_optimization_skills_schema_1_1_package/skills_ii_target_miss_solutions_added(90skills)_gemm_flatten_v1.json", "skill_prompt_mode": "all_skills_avoids_global", "skills_pack": "gemm_flatten_v1", "stream": false, "synth_timeout": 14400, "turns": 4, "variant": "autosa_aav_n_gf", "workflow": "autosa_flash"}`
- **Note:** New LLM C2HLS_ENFORCEMENT with csynth overlap judge (tile loop + arrays inside + latency ≈ max(rooms)). Does not overwrite 12893 syntax-judge campaign.
- **Cell:** `deepseek-v4-flash__flash__autosa__aav_n_gf`
  - flash: **12642–12642** cycles, DSP **318**
  - selected: **12642–12642** cycles, DSP **318**
  - enforcement: applied=True passed=True rounds=1/20 overlap=True reason=tile DATAFLOW parent 8467 is below 0.85 of N*sum 12678 (max=2122, N=2); rooms overlap
  - flash skills injected=129 routed=None pack_sha=13f1aa29df4d2494 `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_enf_aav_n_gf_20260909_085740/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_flash_skills.json`
  - code flash_opt: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_enf_aav_n_gf_20260909_085740/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_flash_opt.cpp`
  - code selected: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_enf_aav_n_gf_20260909_085740/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_selected.cpp`
- **LLM flash_combined:** 10 msgs, calls=2, tokens=80085 `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_enf_aav_n_gf_20260909_085740/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_history.json`

### `batch_parallel_autosa_mm_enf_aav_n_gf_20260909_101836`

- **Family / aim:** enforcement_pingpong — Force explicit ping-pong + DATAFLOW on a flash kernel; judge code+csynth; keep-flash must not jack latency. keep-flash gate: wrap max latency <= flash*1.10; do not drop flash LANES.
- **Status:** complete  **Frozen:** True
- **Created / done:** 2026-09-09T10:18:36.801441+00:00 / 2026-09-09T11:33:19.740048+00:00
- **Path:** `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_enf_aav_n_gf_20260909_101836`
- **Knobs:** `{"benches": "autosa_mm", "dse": false, "enforcement": true, "enforcement_rounds": 20, "keep_flash": true, "latency_opt": false, "model": "deepseek-v4-flash", "note": "Keep-flash enforcement: wrap LANES=16 flash load/compute with tile DATAFLOW. Reject scalar AXI walks and kernel latency > flash x 1.10. Does not overwrite 12893 or 20260909_085740.", "overlap_judge": true, "packaged_skills_json": "/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/hls_full_optimization_skills_schema_1_1_package/skills_ii_target_miss_solutions_added(90skills)_gemm_flatten_v1.json", "skill_prompt_mode": "all_skills_avoids_global", "skills_pack": "gemm_flatten_v1", "stream": false, "synth_timeout": 14400, "turns": 4, "variant": "autosa_aav_n_gf", "workflow": "autosa_flash"}`
- **Note:** Keep-flash enforcement: wrap LANES=16 flash load/compute with tile DATAFLOW. Reject scalar AXI walks and kernel latency > flash x 1.10. Does not overwrite 12893 or 20260909_085740.
- **Cell:** `deepseek-v4-flash__flash__autosa__aav_n_gf`
  - flash: **5542–5542** cycles, DSP **320**
  - selected: **5542–5542** cycles, DSP **320**
  - enforcement: applied=True passed=True rounds=16/20 overlap=True reason=tile DATAFLOW parent 3035 ≈ 2*max(2503)=5006 (not 2*sum=6066); load/compute/store overlap
  - flash skills injected=129 routed=None pack_sha=13f1aa29df4d2494 `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_enf_aav_n_gf_20260909_101836/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_flash_skills.json`
  - code flash_opt: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_enf_aav_n_gf_20260909_101836/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_flash_opt.cpp`
  - code selected: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_enf_aav_n_gf_20260909_101836/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_selected.cpp`
- **LLM flash_combined:** 43 msgs, calls=33, tokens=2566549 `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_enf_aav_n_gf_20260909_101836/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_history.json`

### `batch_parallel_autosa_mm_enf_aav_n_gf_20260909_114519`

- **Family / aim:** enforcement_pingpong — Force explicit ping-pong + DATAFLOW on a flash kernel; judge code+csynth; keep-flash must not jack latency. keep-flash gate: wrap max latency <= flash*1.10; do not drop flash LANES.
- **Status:** complete  **Frozen:** True
- **Created / done:** 2026-09-09T11:45:19.501429+00:00 / 2026-09-09T12:01:33.764320+00:00
- **Path:** `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_enf_aav_n_gf_20260909_114519`
- **Knobs:** `{"benches": "autosa_mm", "dse": false, "enforcement": true, "enforcement_rounds": 20, "keep_flash": true, "latency_opt": false, "model": "deepseek-v4-flash", "note": "Keep-flash enforcement: wrap LANES=16 flash load/compute with tile DATAFLOW. Reject scalar AXI walks and kernel latency > flash x 1.10. Does not overwrite 12893 or 20260909_085740.", "overlap_judge": true, "packaged_skills_json": "/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/hls_full_optimization_skills_schema_1_1_package/skills_ii_target_miss_solutions_added(90skills)_gemm_flatten_v1.json", "skill_prompt_mode": "all_skills_avoids_global", "skills_pack": "gemm_flatten_v1", "stream": false, "synth_timeout": 14400, "turns": 4, "variant": "autosa_aav_n_gf", "workflow": "autosa_flash"}`
- **Note:** Keep-flash enforcement: wrap LANES=16 flash load/compute with tile DATAFLOW. Reject scalar AXI walks and kernel latency > flash x 1.10. Does not overwrite 12893 or 20260909_085740.
- **Cell:** `deepseek-v4-flash__flash__autosa__aav_n_gf`
  - flash: **39116–39116** cycles, DSP **42**
  - selected: **39116–39116** cycles, DSP **42**
  - enforcement: applied=True passed=True rounds=1/20 overlap=True reason=tile DATAFLOW parent 34945 ≈ 2*max(16409)=32818 (not 2*sum=45572); load/compute/store overlap
  - flash skills injected=129 routed=None pack_sha=847494f44a9b4a6e `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_enf_aav_n_gf_20260909_114519/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_flash_skills.json`
  - code flash_opt: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_enf_aav_n_gf_20260909_114519/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_flash_opt.cpp`
  - code selected: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_enf_aav_n_gf_20260909_114519/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_selected.cpp`
- **LLM flash_combined:** 9 msgs, calls=2, tokens=88369 `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_enf_aav_n_gf_20260909_114519/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_history.json`

### `batch_parallel_autosa_mm_enf_aav_n_gf_20260909_121934`

- **Family / aim:** enforcement_pingpong — Force explicit ping-pong + DATAFLOW on a flash kernel; judge code+csynth; keep-flash must not jack latency. keep-flash gate: wrap max latency <= flash*1.10; do not drop flash LANES.
- **Status:** complete  **Frozen:** True
- **Created / done:** 2026-09-09T12:19:35.084264+00:00 / 2026-09-09T12:36:46.734931+00:00
- **Path:** `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_enf_aav_n_gf_20260909_121934`
- **Knobs:** `{"benches": "autosa_mm", "dse": false, "enforcement": true, "enforcement_rounds": 20, "keep_flash": true, "latency_opt": false, "model": "deepseek-v4-flash", "note": "Keep-flash enforcement: wrap LANES=16 flash load/compute with tile DATAFLOW. Reject scalar AXI walks and kernel latency > flash x 1.10. Does not overwrite 12893 or 20260909_085740.", "overlap_judge": true, "packaged_skills_json": "/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/hls_full_optimization_skills_schema_1_1_package/skills_ii_target_miss_solutions_added(90skills)_gemm_flatten_v1.json", "skill_prompt_mode": "all_skills_avoids_global", "skills_pack": "gemm_flatten_v1", "stream": false, "synth_timeout": 14400, "turns": 4, "variant": "autosa_aav_n_gf", "workflow": "autosa_flash"}`
- **Note:** Keep-flash enforcement: wrap LANES=16 flash load/compute with tile DATAFLOW. Reject scalar AXI walks and kernel latency > flash x 1.10. Does not overwrite 12893 or 20260909_085740.
- **Cell:** `deepseek-v4-flash__flash__autosa__aav_n_gf`
  - flash: **34233–68113** cycles, DSP **40**
  - selected: **34233–68113** cycles, DSP **40**
  - enforcement: applied=True passed=True rounds=2/20 overlap=True reason=tile DATAFLOW parent 18 ≈ 2*max(16531)=33062 (not 2*sum=33876); load/compute/store overlap
  - flash skills injected=129 routed=None pack_sha=847494f44a9b4a6e `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_enf_aav_n_gf_20260909_121934/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_flash_skills.json`
  - code flash_opt: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_enf_aav_n_gf_20260909_121934/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_flash_opt.cpp`
  - code selected: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_enf_aav_n_gf_20260909_121934/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_selected.cpp`
- **LLM flash_combined:** 14 msgs, calls=4, tokens=183985 `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_enf_aav_n_gf_20260909_121934/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_history.json`

### `batch_parallel_autosa_mm_enf_aav_n_gf_20260909_175131`

- **Family / aim:** enforcement_pingpong — Force explicit ping-pong + DATAFLOW on a flash kernel; judge code+csynth; keep-flash must not jack latency. skip-flash: reuse a frozen flash_opt, no flash LLM. keep-flash gate: wrap max latency <= flash*1.10; do not drop flash LANES. seed=deepseek-v4-flash__flash__autosa__aav_n_gf
- **Status:** complete  **Frozen:** True
- **Created / done:** 2026-09-09T17:51:31.846956+00:00 / 2026-09-09T19:19:20.187712+00:00
- **Path:** `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_enf_aav_n_gf_20260909_175131`
- **Knobs:** `{"benches": "autosa_mm", "dse": false, "enforcement": true, "enforcement_rounds": 20, "flash_seed_dir": "/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flow_aav_n_gf_20260830_mmflow/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf", "keep_flash": true, "latency_opt": false, "model": "deepseek-v4-flash", "note": "Enforcement-only on seeded mmflow flash 139484/10. No flash LLM. Does not overwrite 123043, 085740, 101836, 114519, 121934, or mmflow.", "overlap_judge": true, "packaged_skills_json": "/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/hls_full_optimization_skills_schema_1_1_package/skills_ii_target_miss_solutions_added(90skills)_gemm_flatten_v1.json", "skill_prompt_mode": "all_skills_avoids_global", "skills_pack": "gemm_flatten_v1", "skip_flash": true, "skip_phase_b": true, "stream": false, "synth_timeout": 14400, "turns": 4, "variant": "autosa_aav_n_gf", "workflow": "autosa_flash"}`
- **Note:** Enforcement-only on seeded mmflow flash 139484/10. No flash LLM. Does not overwrite 123043, 085740, 101836, 114519, 121934, or mmflow.
- **Cell:** `deepseek-v4-flash__flash__autosa__aav_n_gf`
  - flash: **139484–139484** cycles, DSP **10**
  - selected: **139484–139484** cycles, DSP **10**
  - code flash_opt: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_enf_aav_n_gf_20260909_175131/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_flash_opt.cpp`
  - code selected: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_enf_aav_n_gf_20260909_175131/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_selected.cpp`
- **LLM flash_combined:** 32 msgs, calls=25, tokens=1298073 `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_enf_aav_n_gf_20260909_175131/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_history.json`

### `batch_parallel_autosa_mm_enf_aav_n_gf_20260910_234214`

- **Family / aim:** enforcement_pingpong — Force explicit ping-pong + DATAFLOW on a flash kernel; judge code+csynth; keep-flash must not jack latency. skip-flash: reuse a frozen flash_opt, no flash LLM. keep-flash gate: wrap max latency <= flash*1.10; do not drop flash LANES. seed=deepseek-v4-flash__flash__autosa__aav_n_gf
- **Status:** running  **Frozen:** False
- **Created / done:** 2026-09-10T23:42:14.949614+00:00 / None
- **Path:** `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_enf_aav_n_gf_20260910_234214`
- **Knobs:** `{"benches": "autosa_mm", "dse": false, "enforcement": true, "enforcement_rounds": 20, "flash_seed_dir": "/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flow_aav_n_gf_20260830_mmflow/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf", "keep_flash": true, "latency_opt": false, "model": "deepseek-v4-flash", "note": "Enforcement-only on seeded mmflow flash 139484/10. No flash LLM. Does not overwrite 123043, 085740, 101836, 114519, 121934, or mmflow.", "overlap_judge": true, "packaged_skills_json": "/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/hls_full_optimization_skills_schema_1_1_package/skills_ii_target_miss_solutions_added(90skills)_gemm_flatten_v1.json", "skill_prompt_mode": "all_skills_avoids_global", "skills_pack": "gemm_flatten_v1", "skip_flash": true, "skip_phase_b": true, "stream": false, "synth_timeout": 14400, "turns": 4, "variant": "autosa_aav_n_gf", "workflow": "autosa_flash"}`
- **Note:** Enforcement-only on seeded mmflow flash 139484/10. No flash LLM. Does not overwrite 123043, 085740, 101836, 114519, 121934, or mmflow.
- **Cell:** `deepseek-v4-flash__flash__autosa__aav_n_gf`
  - flash: **139484–139484** cycles, DSP **10**
  - code flash_opt: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_enf_aav_n_gf_20260910_234214/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_flash_opt.cpp`

### `batch_parallel_autosa_mm_flash_dataflow_20260904_083918`

- **Family / aim:** flash_dataflow_prompt — Flash-only: DATAFLOW allowed when simple; no systolic stream stage.
- **Status:** complete  **Frozen:** False
- **Created / done:** 2026-09-04T08:39:18.264747+00:00 / 2026-09-04T08:48:25.985354+00:00
- **Path:** `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flash_dataflow_20260904_083918`
- **Knobs:** `{"autosa_flow": true, "benches": "autosa_mm", "dse": false, "enforcement": false, "latency_opt": false, "mm_flow_flavor": "skills", "model": "deepseek-v4-flash", "post_flash_dse": false, "post_flash_stream": false, "skill_prompt_mode": "all_skills_avoids_global", "skills_pack": "gemm_flatten_v1", "skip_phase_b": false, "stream": false, "synth_timeout": 7200, "turns": 7, "variant": "autosa_aav_n_gf", "workflow": "autosa_flash"}`
- **Cell:** `deepseek-v4-flash__flash__autosa__aav_n_gf`
  - flash: **82025–82025** cycles, DSP **20**
  - selected: **82025–82025** cycles, DSP **20**
  - flash skills injected=129 routed=None pack_sha=a4f373cd21848e1a `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flash_dataflow_20260904_083918/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_flash_skills.json`
  - code flash_opt: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flash_dataflow_20260904_083918/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_flash_opt.cpp`
  - code selected: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flash_dataflow_20260904_083918/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_selected.cpp`
- **LLM flash_combined:** 8 msgs, calls=0, tokens=0 `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flash_dataflow_20260904_083918/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_history.json`

### `batch_parallel_autosa_mm_flash_dsp1000_20260903_121153`

- **Family / aim:** flash_dsp_floor — Flash-only: reject csynth if DSP below cutoff; repair toward more MACs.
- **Status:** complete  **Frozen:** False
- **Created / done:** 2026-09-03T12:11:54.226772+00:00 / 2026-09-03T12:30:02.217907+00:00
- **Path:** `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flash_dsp1000_20260903_121153`
- **Knobs:** `{"autosa_flow": true, "benches": "autosa_mm", "dse": false, "enforcement": false, "flash_min_dsp": 1000, "latency_opt": false, "mm_flow_flavor": "skills", "model": "deepseek-v4-flash", "post_flash_dse": false, "post_flash_stream": false, "skill_prompt_mode": "all_skills_avoids_global", "skills_pack": "gemm_flatten_v1", "skip_phase_b": false, "stream": false, "synth_timeout": 3600, "turns": 7, "variant": "autosa_aav_n_gf", "workflow": "autosa_flash"}`
- **Cell:** `deepseek-v4-flash__flash__autosa__aav_n_gf`
  - flash: **13549–13549** cycles, DSP **768**
  - selected: **13549–13549** cycles, DSP **768**
  - flash skills injected=129 routed=None pack_sha=847494f44a9b4a6e `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flash_dsp1000_20260903_121153/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_flash_skills.json`
  - code flash_opt: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flash_dsp1000_20260903_121153/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_flash_opt.cpp`
  - code selected: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flash_dsp1000_20260903_121153/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_selected.cpp`
- **LLM flash_combined:** 10 msgs, calls=0, tokens=0 `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flash_dsp1000_20260903_121153/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_history.json`

### `batch_parallel_autosa_mm_flash_dsp300_20260903_113114`

- **Family / aim:** flash_dsp_floor — Flash-only: reject csynth if DSP below cutoff; repair toward more MACs.
- **Status:** running  **Frozen:** False
- **Created / done:** 2026-09-03T11:31:15.083827+00:00 / None
- **Path:** `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flash_dsp300_20260903_113114`
- **Knobs:** `{"autosa_flow": true, "benches": "autosa_mm", "dse": false, "enforcement": false, "flash_min_dsp": 300, "latency_opt": false, "mm_flow_flavor": "skills", "model": "deepseek-v4-flash", "post_flash_dse": false, "post_flash_stream": false, "skill_prompt_mode": "all_skills_avoids_global", "skills_pack": "gemm_flatten_v1", "skip_phase_b": false, "stream": false, "synth_timeout": 3600, "turns": 7, "variant": "autosa_aav_n_gf", "workflow": "autosa_flash"}`

### `batch_parallel_autosa_mm_flash_dsp300_20260903_113128`

- **Family / aim:** flash_dsp_floor — Flash-only: reject csynth if DSP below cutoff; repair toward more MACs.
- **Status:** complete  **Frozen:** False
- **Created / done:** 2026-09-03T11:31:28.808870+00:00 / 2026-09-03T11:43:16.699373+00:00
- **Path:** `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flash_dsp300_20260903_113128`
- **Knobs:** `{"autosa_flow": true, "benches": "autosa_mm", "dse": false, "enforcement": false, "flash_min_dsp": 300, "latency_opt": false, "mm_flow_flavor": "skills", "model": "deepseek-v4-flash", "post_flash_dse": false, "post_flash_stream": false, "skill_prompt_mode": "all_skills_avoids_global", "skills_pack": "gemm_flatten_v1", "skip_phase_b": false, "stream": false, "synth_timeout": 3600, "turns": 7, "variant": "autosa_aav_n_gf", "workflow": "autosa_flash"}`
- **Cell:** `deepseek-v4-flash__flash__autosa__aav_n_gf`
  - flash: **15708–15708** cycles, DSP **384**
  - selected: **15708–15708** cycles, DSP **384**
  - flash skills injected=129 routed=None pack_sha=847494f44a9b4a6e `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flash_dsp300_20260903_113128/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_flash_skills.json`
  - code flash_opt: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flash_dsp300_20260903_113128/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_flash_opt.cpp`
  - code selected: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flash_dsp300_20260903_113128/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_selected.cpp`
- **LLM flash_combined:** 7 msgs, calls=0, tokens=0 `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flash_dsp300_20260903_113128/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_history.json`

### `batch_parallel_autosa_mm_flash_dsp500_20260904_085931`

- **Family / aim:** flash_dsp_floor — Flash-only: reject csynth if DSP below cutoff; repair toward more MACs.
- **Status:** complete  **Frozen:** False
- **Created / done:** 2026-09-04T08:59:31.575166+00:00 / 2026-09-04T09:07:06.742322+00:00
- **Path:** `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flash_dsp500_20260904_085931`
- **Knobs:** `{"autosa_flow": true, "benches": "autosa_mm", "dse": false, "enforcement": false, "flash_min_dsp": 500, "latency_opt": false, "mm_flow_flavor": "skills", "model": "deepseek-v4-flash", "post_flash_dse": false, "post_flash_stream": false, "skill_prompt_mode": "all_skills_avoids_global", "skills_pack": "gemm_flatten_v1", "skip_phase_b": false, "stream": false, "synth_timeout": 7200, "turns": 7, "variant": "autosa_aav_n_gf", "workflow": "autosa_flash"}`
- **Cell:** `deepseek-v4-flash__flash__autosa__aav_n_gf`
  - flash: **9423–9423** cycles, DSP **1344**
  - selected: **9423–9423** cycles, DSP **1344**
  - flash skills injected=129 routed=None pack_sha=a4f373cd21848e1a `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flash_dsp500_20260904_085931/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_flash_skills.json`
  - code flash_opt: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flash_dsp500_20260904_085931/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_flash_opt.cpp`
  - code selected: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flash_dsp500_20260904_085931/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_selected.cpp`
- **LLM flash_combined:** 9 msgs, calls=0, tokens=0 `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flash_dsp500_20260904_085931/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_history.json`

### `batch_parallel_autosa_mm_flash_dsp500_computeii_20260904_120524`

- **Family / aim:** flash_compute_ii — Flash-only: compute-II pressure on DSP-floor kernels.
- **Status:** complete  **Frozen:** False
- **Created / done:** 2026-09-04T12:05:24.245048+00:00 / 2026-09-04T12:16:22.722275+00:00
- **Path:** `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flash_dsp500_computeii_20260904_120524`
- **Knobs:** `{"autosa_flow": true, "benches": "autosa_mm", "dse": false, "enforcement": false, "flash_min_dsp": 500, "latency_opt": false, "mm_flow_flavor": "skills", "model": "deepseek-v4-flash", "post_flash_dse": false, "post_flash_stream": false, "skill_prompt_mode": "all_skills_avoids_global", "skills_pack": "gemm_flatten_v1", "skip_phase_b": false, "stream": false, "synth_timeout": 7200, "turns": 7, "variant": "autosa_aav_n_gf", "workflow": "autosa_flash"}`
- **Cell:** `deepseek-v4-flash__flash__autosa__aav_n_gf`
  - flash: **1261–1261** cycles, DSP **2672**
  - selected: **1261–1261** cycles, DSP **2672**
  - flash skills injected=129 routed=None pack_sha=add0fb4b775ae467 `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flash_dsp500_computeii_20260904_120524/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_flash_skills.json`
  - code flash_opt: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flash_dsp500_computeii_20260904_120524/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_flash_opt.cpp`
  - code selected: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flash_dsp500_computeii_20260904_120524/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_selected.cpp`
- **LLM flash_combined:** 8 msgs, calls=0, tokens=0 `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flash_dsp500_computeii_20260904_120524/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_history.json`

### `batch_parallel_autosa_mm_flash_dsp500_computeii_20260904_131523`

- **Family / aim:** flash_compute_ii — Flash-only: compute-II pressure on DSP-floor kernels.
- **Status:** aborted  **Frozen:** False
- **Created / done:** 2026-09-04T13:15:23.333730+00:00 / None
- **Path:** `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flash_dsp500_computeii_20260904_131523`
- **Knobs:** `{"autosa_flow": true, "benches": "autosa_mm", "dse": false, "enforcement": false, "flash_min_dsp": 500, "flash_pe_blk": 16, "latency_opt": false, "mm_flow_flavor": "skills", "model": "deepseek-v4-flash", "post_flash_dse": false, "post_flash_stream": false, "skill_prompt_mode": "all_skills_avoids_global", "skills_pack": "gemm_flatten_v1", "skip_phase_b": false, "stream": false, "synth_timeout": 14400, "turns": 7, "variant": "autosa_aav_n_gf", "workflow": "autosa_flash"}`

### `batch_parallel_autosa_mm_flash_dsp500_pe16_20260904_131622`

- **Family / aim:** flash_pe_blk — Flash-only: pin PE_BLK 16/32/64 and spend U280 DSP.
- **Status:** complete  **Frozen:** True
- **Created / done:** 2026-09-04T13:16:22.808812+00:00 / 2026-09-04T13:33:15.448453+00:00
- **Path:** `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flash_dsp500_pe16_20260904_131622`
- **Knobs:** `{"autosa_flow": true, "benches": "autosa_mm", "dse": false, "enforcement": false, "flash_min_dsp": 500, "flash_pe_blk": 16, "latency_opt": false, "mm_flow_flavor": "skills", "model": "deepseek-v4-flash", "post_flash_dse": false, "post_flash_stream": false, "skill_prompt_mode": "all_skills_avoids_global", "skills_pack": "gemm_flatten_v1", "skip_phase_b": false, "stream": false, "synth_timeout": 14400, "turns": 7, "variant": "autosa_aav_n_gf", "workflow": "autosa_flash"}`
- **Cell:** `deepseek-v4-flash__flash__autosa__aav_n_gf`
  - flash: **940–940** cycles, DSP **5344**
  - selected: **940–940** cycles, DSP **5344**
  - flash skills injected=129 routed=None pack_sha=1270f94f2b83fdab `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flash_dsp500_pe16_20260904_131622/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_flash_skills.json`
  - code flash_opt: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flash_dsp500_pe16_20260904_131622/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_flash_opt.cpp`
  - code selected: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flash_dsp500_pe16_20260904_131622/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_selected.cpp`
- **LLM flash_combined:** 11 msgs, calls=0, tokens=0 `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flash_dsp500_pe16_20260904_131622/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_history.json`

### `batch_parallel_autosa_mm_flash_dsp500_pe16_tilepp_20260905_001958`

- **Family / aim:** flash_tile_pp — Flash-only: PE_BLK=16 plus in-GEMM tile ping-pong prompt.
- **Status:** running  **Frozen:** False
- **Created / done:** 2026-09-05T00:19:58.457532+00:00 / None
- **Path:** `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flash_dsp500_pe16_tilepp_20260905_001958`
- **Knobs:** `{"autosa_flow": true, "benches": "autosa_mm", "dse": false, "enforcement": false, "flash_min_dsp": 500, "flash_pe_blk": 16, "flash_tile_pp": 1, "latency_opt": false, "mm_flow_flavor": "skills", "model": "deepseek-v4-flash", "post_flash_dse": false, "post_flash_stream": false, "skill_prompt_mode": "all_skills_avoids_global", "skills_pack": "gemm_flatten_v1", "skip_phase_b": false, "stream": false, "synth_timeout": 14400, "turns": 7, "variant": "autosa_aav_n_gf", "workflow": "autosa_flash"}`

### `batch_parallel_autosa_mm_flash_dsp500_pe16_tilepp_20260905_002017`

- **Family / aim:** flash_tile_pp — Flash-only: PE_BLK=16 plus in-GEMM tile ping-pong prompt.
- **Status:** complete  **Frozen:** False
- **Created / done:** 2026-09-05T00:20:17.375861+00:00 / 2026-09-05T00:31:36.285041+00:00
- **Path:** `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flash_dsp500_pe16_tilepp_20260905_002017`
- **Knobs:** `{"autosa_flow": true, "benches": "autosa_mm", "dse": false, "enforcement": false, "flash_min_dsp": 500, "flash_pe_blk": 16, "flash_tile_pp": 1, "latency_opt": false, "mm_flow_flavor": "skills", "model": "deepseek-v4-flash", "post_flash_dse": false, "post_flash_stream": false, "skill_prompt_mode": "all_skills_avoids_global", "skills_pack": "gemm_flatten_v1", "skip_phase_b": false, "stream": false, "synth_timeout": 14400, "turns": 7, "variant": "autosa_aav_n_gf", "workflow": "autosa_flash"}`
- **Cell:** `deepseek-v4-flash__flash__autosa__aav_n_gf`
  - flash: **3817–3817** cycles, DSP **5120**
  - selected: **3817–3817** cycles, DSP **5120**
  - flash skills injected=129 routed=None pack_sha=13f1aa29df4d2494 `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flash_dsp500_pe16_tilepp_20260905_002017/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_flash_skills.json`
  - code flash_opt: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flash_dsp500_pe16_tilepp_20260905_002017/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_flash_opt.cpp`
  - code selected: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flash_dsp500_pe16_tilepp_20260905_002017/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_selected.cpp`
- **LLM flash_combined:** 8 msgs, calls=0, tokens=0 `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flash_dsp500_pe16_tilepp_20260905_002017/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_history.json`

### `batch_parallel_autosa_mm_flash_dsp500_pe32_20260904_133157`

- **Family / aim:** flash_pe_blk — Flash-only: pin PE_BLK 16/32/64 and spend U280 DSP.
- **Status:** complete  **Frozen:** False
- **Created / done:** 2026-09-04T13:31:57.489500+00:00 / 2026-09-04T14:43:32.163294+00:00
- **Path:** `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flash_dsp500_pe32_20260904_133157`
- **Knobs:** `{"autosa_flow": true, "benches": "autosa_mm", "dse": false, "enforcement": false, "flash_min_dsp": 500, "flash_pe_blk": 32, "latency_opt": false, "mm_flow_flavor": "skills", "model": "deepseek-v4-flash", "post_flash_dse": false, "post_flash_stream": false, "skill_prompt_mode": "all_skills_avoids_global", "skills_pack": "gemm_flatten_v1", "skip_phase_b": false, "stream": false, "synth_timeout": 14400, "turns": 7, "variant": "autosa_aav_n_gf", "workflow": "autosa_flash"}`
- **Cell:** `deepseek-v4-flash__flash__autosa__aav_n_gf`
  - flash: **9991–9991** cycles, DSP **782**
  - selected: **9991–9991** cycles, DSP **782**
  - flash skills injected=129 routed=None pack_sha=1270f94f2b83fdab `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flash_dsp500_pe32_20260904_133157/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_flash_skills.json`
  - code flash_opt: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flash_dsp500_pe32_20260904_133157/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_flash_opt.cpp`
  - code selected: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flash_dsp500_pe32_20260904_133157/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_selected.cpp`
- **LLM flash_combined:** 10 msgs, calls=0, tokens=0 `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flash_dsp500_pe32_20260904_133157/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_history.json`

### `batch_parallel_autosa_mm_flash_dsp500_pe64_20260904_134954`

- **Family / aim:** flash_pe_blk — Flash-only: pin PE_BLK 16/32/64 and spend U280 DSP.
- **Status:** complete  **Frozen:** False
- **Created / done:** 2026-09-04T13:49:55.088629+00:00 / 2026-09-04T14:00:31.143223+00:00
- **Path:** `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flash_dsp500_pe64_20260904_134954`
- **Knobs:** `{"autosa_flow": true, "benches": "autosa_mm", "dse": false, "enforcement": false, "flash_min_dsp": 500, "flash_pe_blk": 64, "latency_opt": false, "mm_flow_flavor": "skills", "model": "deepseek-v4-flash", "post_flash_dse": false, "post_flash_stream": false, "skill_prompt_mode": "all_skills_avoids_global", "skills_pack": "gemm_flatten_v1", "skip_phase_b": false, "stream": false, "synth_timeout": 14400, "turns": 7, "variant": "autosa_aav_n_gf", "workflow": "autosa_flash"}`
- **Cell:** `deepseek-v4-flash__flash__autosa__aav_n_gf`
  - flash: **1733–1733** cycles, DSP **1272**
  - selected: **1733–1733** cycles, DSP **1272**
  - flash skills injected=129 routed=None pack_sha=1270f94f2b83fdab `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flash_dsp500_pe64_20260904_134954/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_flash_skills.json`
  - code flash_opt: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flash_dsp500_pe64_20260904_134954/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_flash_opt.cpp`
  - code selected: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flash_dsp500_pe64_20260904_134954/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_selected.cpp`
- **LLM flash_combined:** 9 msgs, calls=0, tokens=0 `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flash_dsp500_pe64_20260904_134954/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_history.json`

### `batch_parallel_autosa_mm_flash_dsp500_wideio_20260904_100437`

- **Family / aim:** flash_wide_io — Flash-only: wide AXI / LANES prompt (opt-in).
- **Status:** complete  **Frozen:** False
- **Created / done:** 2026-09-04T10:04:38.277508+00:00 / 2026-09-04T11:11:22.518275+00:00
- **Path:** `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flash_dsp500_wideio_20260904_100437`
- **Knobs:** `{"autosa_flow": true, "benches": "autosa_mm", "dse": false, "enforcement": false, "flash_min_dsp": 500, "latency_opt": false, "mm_flow_flavor": "skills", "model": "deepseek-v4-flash", "post_flash_dse": false, "post_flash_stream": false, "skill_prompt_mode": "all_skills_avoids_global", "skills_pack": "gemm_flatten_v1", "skip_phase_b": false, "stream": false, "synth_timeout": 7200, "turns": 7, "variant": "autosa_aav_n_gf", "workflow": "autosa_flash"}`
- **Cell:** `deepseek-v4-flash__flash__autosa__aav_n_gf`
  - flash: **2760–2760** cycles, DSP **636**
  - selected: **2760–2760** cycles, DSP **636**
  - flash skills injected=129 routed=None pack_sha=d4cc487941c47dff `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flash_dsp500_wideio_20260904_100437/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_flash_skills.json`
  - code flash_opt: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flash_dsp500_wideio_20260904_100437/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_flash_opt.cpp`
  - code selected: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flash_dsp500_wideio_20260904_100437/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_selected.cpp`
- **LLM flash_combined:** 9 msgs, calls=0, tokens=0 `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flash_dsp500_wideio_20260904_100437/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_history.json`

### `batch_parallel_autosa_mm_flash_onchip_t7_20260906_104715`

- **Family / aim:** flash_onchip — Flash-only: distilled on-chip GEMM pack (940-class), not 90-skills.
- **Status:** running  **Frozen:** False
- **Created / done:** 2026-09-06T10:47:15.589350+00:00 / None
- **Path:** `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flash_onchip_t7_20260906_104715`
- **Knobs:** `{"autosa_flow": true, "benches": "autosa_mm", "dse": false, "enforcement": false, "flash_min_dsp": 500, "flash_onchip": 1, "flash_pe_blk": 16, "latency_opt": false, "mm_flow_flavor": "skills", "model": "deepseek-v4-flash", "packaged_skills_json": "/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/hls_full_optimization_skills_schema_1_1_package/flash_onchip_wide_gemm_skill_entries.json", "post_flash_dse": false, "post_flash_stream": false, "skill_prompt_mode": "all_skills_avoids_global", "skills_pack": "gemm_flatten_v1", "skip_phase_b": false, "stream": false, "synth_timeout": 14400, "turns": 7, "variant": "autosa_onchip_gemm", "workflow": "autosa_flash"}`

### `batch_parallel_autosa_mm_flash_onchip_t7_20260906_104733`

- **Family / aim:** flash_onchip — Flash-only: distilled on-chip GEMM pack (940-class), not 90-skills.
- **Status:** complete  **Frozen:** False
- **Created / done:** 2026-09-06T10:47:33.891438+00:00 / 2026-09-06T11:03:51.824987+00:00
- **Path:** `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flash_onchip_t7_20260906_104733`
- **Knobs:** `{"autosa_flow": true, "benches": "autosa_mm", "dse": false, "enforcement": false, "flash_min_dsp": 500, "flash_onchip": 1, "flash_pe_blk": 16, "latency_opt": false, "mm_flow_flavor": "skills", "model": "deepseek-v4-flash", "packaged_skills_json": "/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/hls_full_optimization_skills_schema_1_1_package/flash_onchip_wide_gemm_skill_entries.json", "post_flash_dse": false, "post_flash_stream": false, "skill_prompt_mode": "all_skills_avoids_global", "skills_pack": "gemm_flatten_v1", "skip_phase_b": false, "stream": false, "synth_timeout": 14400, "turns": 7, "variant": "autosa_onchip_gemm", "workflow": "autosa_flash"}`
- **Cell:** `deepseek-v4-flash__flash__autosa__onchip_gemm`
  - flash: **871–871** cycles, DSP **5088**
  - selected: **871–871** cycles, DSP **5088**
  - flash skills injected=8 routed=None pack_sha=a7302183c2c35e52 `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flash_onchip_t7_20260906_104733/variants/autosa_onchip_gemm/autosa_mm/deepseek-v4-flash__flash__autosa__onchip_gemm/autosa_mm_flash_skills.json`
  - code flash_opt: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flash_onchip_t7_20260906_104733/variants/autosa_onchip_gemm/autosa_mm/deepseek-v4-flash__flash__autosa__onchip_gemm/autosa_mm_flash_opt.cpp`
  - code selected: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flash_onchip_t7_20260906_104733/variants/autosa_onchip_gemm/autosa_mm/deepseek-v4-flash__flash__autosa__onchip_gemm/autosa_mm_selected.cpp`
- **LLM flash_combined:** 9 msgs, calls=0, tokens=0 `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flash_onchip_t7_20260906_104733/variants/autosa_onchip_gemm/autosa_mm/deepseek-v4-flash__flash__autosa__onchip_gemm/autosa_mm_history.json`

### `batch_parallel_autosa_mm_flash_onchip_t7_20260906_123340`

- **Family / aim:** flash_onchip — Flash-only: distilled on-chip GEMM pack (940-class), not 90-skills.
- **Status:** complete  **Frozen:** False
- **Created / done:** 2026-09-06T12:33:40.958569+00:00 / 2026-09-06T12:45:12.095710+00:00
- **Path:** `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flash_onchip_t7_20260906_123340`
- **Knobs:** `{"autosa_flow": true, "benches": "autosa_mm", "dse": false, "enforcement": false, "flash_min_dsp": 500, "flash_onchip": 1, "flash_pe_blk": 16, "latency_opt": false, "mm_flow_flavor": "skills", "model": "deepseek-v4-flash", "packaged_skills_json": "/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/hls_full_optimization_skills_schema_1_1_package/flash_onchip_wide_gemm_skill_entries.json", "post_flash_dse": false, "post_flash_stream": false, "skill_prompt_mode": "all_skills_avoids_global", "skills_pack": "gemm_flatten_v1", "skip_phase_b": false, "stream": false, "synth_timeout": 14400, "turns": 7, "variant": "autosa_onchip_gemm", "workflow": "autosa_flash"}`
- **Cell:** `deepseek-v4-flash__flash__autosa__onchip_gemm`
  - flash: **1002–1002** cycles, DSP **5344**
  - selected: **1002–1002** cycles, DSP **5344**
  - flash skills injected=8 routed=None pack_sha=a7302183c2c35e52 `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flash_onchip_t7_20260906_123340/variants/autosa_onchip_gemm/autosa_mm/deepseek-v4-flash__flash__autosa__onchip_gemm/autosa_mm_flash_skills.json`
  - code flash_opt: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flash_onchip_t7_20260906_123340/variants/autosa_onchip_gemm/autosa_mm/deepseek-v4-flash__flash__autosa__onchip_gemm/autosa_mm_flash_opt.cpp`
  - code selected: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flash_onchip_t7_20260906_123340/variants/autosa_onchip_gemm/autosa_mm/deepseek-v4-flash__flash__autosa__onchip_gemm/autosa_mm_selected.cpp`
- **LLM flash_combined:** 7 msgs, calls=0, tokens=0 `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flash_onchip_t7_20260906_123340/variants/autosa_onchip_gemm/autosa_mm/deepseek-v4-flash__flash__autosa__onchip_gemm/autosa_mm_history.json`

### `batch_parallel_autosa_mm_flash_rowuf64_20260904_012045`

- **Family / aim:** flash_row_uf — Flash-only: unroll I in one tile (ROW_UF).
- **Status:** running  **Frozen:** False
- **Created / done:** 2026-09-04T01:20:45.505951+00:00 / None
- **Path:** `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flash_rowuf64_20260904_012045`
- **Knobs:** `{"autosa_flow": true, "benches": "autosa_mm", "dse": false, "enforcement": false, "flash_row_uf": 64, "latency_opt": false, "mm_flow_flavor": "skills", "model": "deepseek-v4-flash", "post_flash_dse": false, "post_flash_stream": false, "skill_prompt_mode": "all_skills_avoids_global", "skills_pack": "gemm_flatten_v1", "skip_phase_b": false, "stream": false, "synth_timeout": 3600, "turns": 7, "variant": "autosa_aav_n_gf", "workflow": "autosa_flash"}`

### `batch_parallel_autosa_mm_flash_rowuf64_20260904_012102`

- **Family / aim:** flash_row_uf — Flash-only: unroll I in one tile (ROW_UF).
- **Status:** complete  **Frozen:** False
- **Created / done:** 2026-09-04T01:21:02.875823+00:00 / 2026-09-04T01:29:55.250067+00:00
- **Path:** `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flash_rowuf64_20260904_012102`
- **Knobs:** `{"autosa_flow": true, "benches": "autosa_mm", "dse": false, "enforcement": false, "flash_row_uf": 64, "latency_opt": false, "mm_flow_flavor": "skills", "model": "deepseek-v4-flash", "post_flash_dse": false, "post_flash_stream": false, "skill_prompt_mode": "all_skills_avoids_global", "skills_pack": "gemm_flatten_v1", "skip_phase_b": false, "stream": false, "synth_timeout": 7200, "turns": 7, "variant": "autosa_aav_n_gf", "workflow": "autosa_flash"}`
- **Cell:** `deepseek-v4-flash__flash__autosa__aav_n_gf`
  - flash: **12456–12456** cycles, DSP **320**
  - selected: **12456–12456** cycles, DSP **320**
  - flash skills injected=129 routed=None pack_sha=847494f44a9b4a6e `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flash_rowuf64_20260904_012102/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_flash_skills.json`
  - code flash_opt: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flash_rowuf64_20260904_012102/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_flash_opt.cpp`
  - code selected: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flash_rowuf64_20260904_012102/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_selected.cpp`
- **LLM flash_combined:** 7 msgs, calls=0, tokens=0 `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flash_rowuf64_20260904_012102/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_history.json`

### `batch_parallel_autosa_mm_flow_aav_n_gf_20260830_mmflow`

- **Family / aim:** mmflow_3stage — Flash then compute-rewrite then hide-load-store stream. Frozen slide column.
- **Status:** complete  **Frozen:** True
- **Created / done:** 2026-08-30T11:18:19.777642+00:00 / 2026-08-30T11:31:31.264378+00:00
- **Path:** `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flow_aav_n_gf_20260830_mmflow`
- **Knobs:** `{"autosa_flow": true, "benches": "autosa_mm", "dse": true, "enforcement": false, "latency_opt": false, "model": "deepseek-v4-flash", "post_flash_dse": true, "post_flash_stream": true, "skill_prompt_mode": "all_skills_avoids_global", "skills_pack": "gemm_flatten_v1", "stream": true, "synth_timeout": 3600, "turns": 4, "variant": "autosa_aav_n_gf", "workflow": "autosa_flash"}`
- **Cell:** `deepseek-v4-flash__flash__autosa__aav_n_gf`
  - flash: **139484–139484** cycles, DSP **10**
  - dse: **13160–13160** cycles, DSP **352**
  - stream: **4292–4292** cycles, DSP **320**
  - selected: **4292–4292** cycles, DSP **320**
  - flash skills injected=129 routed=None pack_sha=847494f44a9b4a6e `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flow_aav_n_gf_20260830_mmflow/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_flash_skills.json`
  - code flash_opt: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flow_aav_n_gf_20260830_mmflow/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_flash_opt.cpp`
  - code selected: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flow_aav_n_gf_20260830_mmflow/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_selected.cpp`
  - code dse: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flow_aav_n_gf_20260830_mmflow/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_dse.cpp`
  - code stream: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flow_aav_n_gf_20260830_mmflow/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_stream.cpp`
- **LLM flash_combined:** 10 msgs, calls=0, tokens=0 `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flow_aav_n_gf_20260830_mmflow/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_history.json`
- **LLM dse:** 3 msgs, calls=None, tokens=None `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flow_aav_n_gf_20260830_mmflow/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_dse_history.json`
- **LLM stream:** 3 msgs, calls=None, tokens=None `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flow_aav_n_gf_20260830_mmflow/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_stream_history.json`

### `batch_parallel_autosa_mm_flow_aav_n_gf_task5_mmflow_dryrun`

- **Family / aim:** mmflow_3stage — Flash then compute-rewrite then hide-load-store stream. Frozen slide column.
- **Status:** running  **Frozen:** False
- **Created / done:** 2026-08-30T10:24:01.110371+00:00 / None
- **Path:** `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flow_aav_n_gf_task5_mmflow_dryrun`
- **Knobs:** `{"autosa_flow": true, "benches": "autosa_mm", "dse": true, "enforcement": false, "latency_opt": false, "model": "deepseek-v4-flash", "post_flash_dse": true, "post_flash_stream": true, "skill_prompt_mode": "all_skills_avoids_global", "skills_pack": "gemm_flatten_v1", "stream": true, "synth_timeout": 3600, "turns": 4, "variant": "autosa_aav_n_gf", "workflow": "autosa_flash"}`

### `batch_parallel_autosa_mm_flow_noskills_20260901_mmns`

- **Family / aim:** ablation_noskills — 3-stage flow without the 90-skill dump (JSON PE recipe still on for stream).
- **Status:** complete  **Frozen:** False
- **Created / done:** 2026-09-01T10:35:54.818858+00:00 / 2026-09-01T10:40:12.690874+00:00
- **Path:** `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flow_noskills_20260901_mmns`
- **Knobs:** `{"autosa_flow": true, "benches": "autosa_mm", "dse": true, "enforcement": false, "latency_opt": false, "mm_flow_flavor": "noskills", "model": "deepseek-v4-flash", "post_flash_dse": true, "post_flash_stream": true, "skills_pack": "none", "skip_phase_b": false, "stream": true, "synth_timeout": 3600, "turns": 4, "variant": "autosa_noskills", "workflow": "autosa_flash"}`
- **Cell:** `batch_parallel_autosa_mm_flow_noskills_20260901_mmns`

### `batch_parallel_autosa_mm_flow_noskills_20260902_mmns`

- **Family / aim:** ablation_noskills — 3-stage flow without the 90-skill dump (JSON PE recipe still on for stream).
- **Status:** complete  **Frozen:** False
- **Created / done:** 2026-09-02T08:26:22.552946+00:00 / 2026-09-02T09:07:08.516133+00:00
- **Path:** `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flow_noskills_20260902_mmns`
- **Knobs:** `{"autosa_flow": true, "benches": "autosa_mm", "dse": true, "enforcement": false, "latency_opt": false, "mm_flow_flavor": "noskills", "model": "deepseek-v4-flash", "post_flash_dse": true, "post_flash_stream": true, "skills_pack": "none", "skip_phase_b": false, "stream": true, "synth_timeout": 3600, "turns": 4, "variant": "autosa_noskills", "workflow": "autosa_flash"}`
- **Cell:** `deepseek-v4-flash__flash__autosa__noskills`
  - flash: **1056894–1056894** cycles, DSP **11**
  - dse: **25080–25080** cycles, DSP **384**
  - stream: **4216–4216** cycles, DSP **320**
  - selected: **4216–4216** cycles, DSP **320**
  - flash skills injected=0 routed=None pack_sha=3fab1320923be793 `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flow_noskills_20260902_mmns/variants/autosa_noskills/autosa_mm/deepseek-v4-flash__flash__autosa__noskills/autosa_mm_flash_skills.json`
  - code flash_opt: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flow_noskills_20260902_mmns/variants/autosa_noskills/autosa_mm/deepseek-v4-flash__flash__autosa__noskills/autosa_mm_flash_opt.cpp`
  - code selected: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flow_noskills_20260902_mmns/variants/autosa_noskills/autosa_mm/deepseek-v4-flash__flash__autosa__noskills/autosa_mm_selected.cpp`
  - code dse: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flow_noskills_20260902_mmns/variants/autosa_noskills/autosa_mm/deepseek-v4-flash__flash__autosa__noskills/autosa_mm_dse.cpp`
  - code stream: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flow_noskills_20260902_mmns/variants/autosa_noskills/autosa_mm/deepseek-v4-flash__flash__autosa__noskills/autosa_mm_stream.cpp`
- **LLM flash_combined:** 8 msgs, calls=0, tokens=0 `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flow_noskills_20260902_mmns/variants/autosa_noskills/autosa_mm/deepseek-v4-flash__flash__autosa__noskills/autosa_mm_history.json`
- **LLM dse:** 3 msgs, calls=None, tokens=None `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flow_noskills_20260902_mmns/variants/autosa_noskills/autosa_mm/deepseek-v4-flash__flash__autosa__noskills/autosa_mm_dse_history.json`
- **LLM stream:** 9 msgs, calls=None, tokens=None `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flow_noskills_20260902_mmns/variants/autosa_noskills/autosa_mm/deepseek-v4-flash__flash__autosa__noskills/autosa_mm_stream_history.json`

### `batch_parallel_autosa_mm_flow_zero_shot_20260901_mmzs`

- **Family / aim:** ablation_zero_shot — No skills, no PE recipe: what flash does with a generic prompt only.
- **Status:** aborted  **Frozen:** False
- **Created / done:** 2026-09-01T10:20:35.987880+00:00 / 2026-09-01T10:25:02.533339+00:00
- **Path:** `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flow_zero_shot_20260901_mmzs`
- **Knobs:** `{"autosa_flow": true, "benches": "autosa_mm", "dse": false, "enforcement": false, "latency_opt": false, "mm_flow_flavor": "zero_shot", "model": "deepseek-v4-flash", "post_flash_dse": false, "post_flash_stream": false, "skills_pack": "none", "skip_phase_b": true, "stream": false, "synth_timeout": 3600, "turns": 1, "variant": "autosa_zero_shot", "workflow": "autosa_flash"}`
- **Cell:** `batch_parallel_autosa_mm_flow_zero_shot_20260901_mmzs`

### `batch_parallel_autosa_mm_flow_zero_shot_20260901_mmzs2`

- **Family / aim:** ablation_zero_shot — No skills, no PE recipe: what flash does with a generic prompt only.
- **Status:** complete  **Frozen:** False
- **Created / done:** 2026-09-01T10:31:27.962478+00:00 / 2026-09-01T10:34:46.287287+00:00
- **Path:** `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flow_zero_shot_20260901_mmzs2`
- **Knobs:** `{"autosa_flow": true, "benches": "autosa_mm", "dse": false, "enforcement": false, "latency_opt": false, "mm_flow_flavor": "zero_shot", "model": "deepseek-v4-flash", "post_flash_dse": false, "post_flash_stream": false, "skills_pack": "none", "skip_phase_b": true, "stream": false, "synth_timeout": 3600, "turns": 1, "variant": "autosa_zero_shot", "workflow": "autosa_flash"}`
- **Cell:** `batch_parallel_autosa_mm_flow_zero_shot_20260901_mmzs2`

### `batch_parallel_autosa_mm_flow_zero_shot_20260902_mmzs`

- **Family / aim:** ablation_zero_shot — No skills, no PE recipe: what flash does with a generic prompt only.
- **Status:** complete  **Frozen:** False
- **Created / done:** 2026-09-02T08:16:30.461238+00:00 / 2026-09-02T08:24:35.233825+00:00
- **Path:** `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flow_zero_shot_20260902_mmzs`
- **Knobs:** `{"autosa_flow": true, "benches": "autosa_mm", "dse": false, "enforcement": false, "latency_opt": false, "mm_flow_flavor": "zero_shot", "model": "deepseek-v4-flash", "post_flash_dse": false, "post_flash_stream": false, "skills_pack": "none", "skip_phase_b": true, "stream": false, "synth_timeout": 3600, "turns": 1, "variant": "autosa_zero_shot", "workflow": "autosa_flash"}`
- **Cell:** `deepseek-v4-flash__flash__autosa__zero_shot`
  - flash: **40454–42758** cycles, DSP **320**
  - selected: **40454–42758** cycles, DSP **320**
  - flash skills injected=0 routed=None pack_sha=3fab1320923be793 `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flow_zero_shot_20260902_mmzs/variants/autosa_zero_shot/autosa_mm/deepseek-v4-flash__flash__autosa__zero_shot/autosa_mm_flash_skills.json`
  - code flash_opt: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flow_zero_shot_20260902_mmzs/variants/autosa_zero_shot/autosa_mm/deepseek-v4-flash__flash__autosa__zero_shot/autosa_mm_flash_opt.cpp`
  - code selected: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flow_zero_shot_20260902_mmzs/variants/autosa_zero_shot/autosa_mm/deepseek-v4-flash__flash__autosa__zero_shot/autosa_mm_selected.cpp`
- **LLM flash_combined:** 4 msgs, calls=0, tokens=0 `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flow_zero_shot_20260902_mmzs/variants/autosa_zero_shot/autosa_mm/deepseek-v4-flash__flash__autosa__zero_shot/autosa_mm_history.json`

### `batch_parallel_autosa_mm_gap_20260818_mm_gap_flash`

- **Family / aim:** gap_flash — Early gap-vs-rank-1 flash (Aug 18–23). Some ABI/model variants.
- **Status:** complete  **Frozen:** False
- **Created / done:** 2026-08-18T05:02:31.098164+00:00 / 2026-08-18T05:09:22.171121+00:00
- **Path:** `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_gap_20260818_mm_gap_flash`
- **Knobs:** `{"benches": "autosa_mm", "model": "mistralai/Devstral-2-123B-Instruct-2512", "turns": 4, "variant": "autosa_nav_n", "workflow": "autosa_flash"}`
- **Cell:** `batch_parallel_autosa_mm_gap_20260818_mm_gap_flash`

### `batch_parallel_autosa_mm_gap_20260818_mm_gap_flash_abi`

- **Family / aim:** gap_flash — Early gap-vs-rank-1 flash (Aug 18–23). Some ABI/model variants.
- **Status:** complete  **Frozen:** False
- **Created / done:** 2026-08-18T05:58:43.094041+00:00 / 2026-08-18T06:02:17.545773+00:00
- **Path:** `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_gap_20260818_mm_gap_flash_abi`
- **Knobs:** `{"benches": "autosa_mm", "model": "mistralai/Devstral-2-123B-Instruct-2512", "turns": 4, "variant": "autosa_nav_n", "workflow": "autosa_flash"}`
- **Cell:** `devstral2__flash__autosa__nav_n`
  - flash: **1056932–1056932** cycles, DSP **3**
  - selected: **1056932–1056932** cycles, DSP **3**
  - flash skills injected=84 routed=None pack_sha=022f4d57f3ef126d `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_gap_20260818_mm_gap_flash_abi/variants/autosa_nav_n/autosa_mm/devstral2__flash__autosa__nav_n/autosa_mm_flash_skills.json`
  - code flash_opt: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_gap_20260818_mm_gap_flash_abi/variants/autosa_nav_n/autosa_mm/devstral2__flash__autosa__nav_n/autosa_mm_flash_opt.cpp`
  - code selected: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_gap_20260818_mm_gap_flash_abi/variants/autosa_nav_n/autosa_mm/devstral2__flash__autosa__nav_n/autosa_mm_selected.cpp`
- **LLM flash_combined:** 8 msgs, calls=0, tokens=0 `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_gap_20260818_mm_gap_flash_abi/variants/autosa_nav_n/autosa_mm/devstral2__flash__autosa__nav_n/autosa_mm_history.json`

### `batch_parallel_autosa_mm_gap_ds_v4f_20260818_mm_gap_flash_ds_v4f`

- **Family / aim:** gap_flash — Early gap-vs-rank-1 flash (Aug 18–23). Some ABI/model variants.
- **Status:** complete  **Frozen:** False
- **Created / done:** 2026-08-18T07:57:17.059153+00:00 / 2026-08-18T08:01:34.993664+00:00
- **Path:** `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_gap_ds_v4f_20260818_mm_gap_flash_ds_v4f`
- **Knobs:** `{"benches": "autosa_mm", "latency_opt": false, "model": "deepseek-v4-flash", "turns": 4, "variant": "autosa_nav_n", "workflow": "autosa_flash"}`
- **Cell:** `deepseek-v4-flash__flash__autosa__nav_n`
  - flash: **149082–149082** cycles, DSP **6**
  - dse: **13160–13160** cycles, DSP **352**
  - stream: **4285–4285** cycles, DSP **320**
  - selected: **4285–4285** cycles, DSP **320**
  - flash skills injected=84 routed=None pack_sha=022f4d57f3ef126d `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_gap_ds_v4f_20260818_mm_gap_flash_ds_v4f/variants/autosa_nav_n/autosa_mm/deepseek-v4-flash__flash__autosa__nav_n/autosa_mm_flash_skills.json`
  - code flash_opt: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_gap_ds_v4f_20260818_mm_gap_flash_ds_v4f/variants/autosa_nav_n/autosa_mm/deepseek-v4-flash__flash__autosa__nav_n/autosa_mm_flash_opt.cpp`
  - code selected: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_gap_ds_v4f_20260818_mm_gap_flash_ds_v4f/variants/autosa_nav_n/autosa_mm/deepseek-v4-flash__flash__autosa__nav_n/autosa_mm_selected.cpp`
  - code dse: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_gap_ds_v4f_20260818_mm_gap_flash_ds_v4f/variants/autosa_nav_n/autosa_mm/deepseek-v4-flash__flash__autosa__nav_n/autosa_mm_dse.cpp`
  - code stream: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_gap_ds_v4f_20260818_mm_gap_flash_ds_v4f/variants/autosa_nav_n/autosa_mm/deepseek-v4-flash__flash__autosa__nav_n/autosa_mm_stream.cpp`
- **LLM flash_combined:** 8 msgs, calls=0, tokens=0 `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_gap_ds_v4f_20260818_mm_gap_flash_ds_v4f/variants/autosa_nav_n/autosa_mm/deepseek-v4-flash__flash__autosa__nav_n/autosa_mm_history.json`
- **LLM dse:** 3 msgs, calls=None, tokens=None `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_gap_ds_v4f_20260818_mm_gap_flash_ds_v4f/variants/autosa_nav_n/autosa_mm/deepseek-v4-flash__flash__autosa__nav_n/autosa_mm_dse_history.json`
- **LLM stream:** 3 msgs, calls=None, tokens=None `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_gap_ds_v4f_20260818_mm_gap_flash_ds_v4f/variants/autosa_nav_n/autosa_mm/deepseek-v4-flash__flash__autosa__nav_n/autosa_mm_stream_history.json`

### `batch_parallel_autosa_mm_gap_ds_v4f_aav_n_20260819_mm_gap_flash_ds_v4f_aav_n`

- **Family / aim:** gap_flash — Early gap-vs-rank-1 flash (Aug 18–23). Some ABI/model variants.
- **Status:** complete  **Frozen:** False
- **Created / done:** 2026-08-19T18:39:43.434317+00:00 / 2026-08-19T22:49:34.278484+00:00
- **Path:** `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_gap_ds_v4f_aav_n_20260819_mm_gap_flash_ds_v4f_aav_n`
- **Knobs:** `{"benches": "autosa_mm", "dse": false, "latency_opt": false, "model": "deepseek-v4-flash", "skill_prompt_mode": "all_skills_avoids_global", "stream": false, "turns": 4, "variant": "autosa_aav_n", "workflow": "autosa_flash"}`
- **Cell:** `deepseek-v4-flash__flash__autosa__aav_n`
  - flash: **73897–73897** cycles, DSP **20**
  - selected: **73897–73897** cycles, DSP **20**
  - flash skills injected=123 routed=None pack_sha=022f4d57f3ef126d `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_gap_ds_v4f_aav_n_20260819_mm_gap_flash_ds_v4f_aav_n/variants/autosa_aav_n/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n/autosa_mm_flash_skills.json`
  - code flash_opt: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_gap_ds_v4f_aav_n_20260819_mm_gap_flash_ds_v4f_aav_n/variants/autosa_aav_n/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n/autosa_mm_flash_opt.cpp`
  - code selected: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_gap_ds_v4f_aav_n_20260819_mm_gap_flash_ds_v4f_aav_n/variants/autosa_aav_n/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n/autosa_mm_selected.cpp`
- **LLM flash_combined:** 8 msgs, calls=0, tokens=0 `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_gap_ds_v4f_aav_n_20260819_mm_gap_flash_ds_v4f_aav_n/variants/autosa_aav_n/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n/autosa_mm_history.json`

### `batch_parallel_autosa_mm_gap_ds_v4f_aav_n_gf_20260823_mm_gap_flash_ds_v4f_aav_n_gf`

- **Family / aim:** gap_flash — Early gap-vs-rank-1 flash (Aug 18–23). Some ABI/model variants.
- **Status:** complete  **Frozen:** False
- **Created / done:** 2026-08-23T08:25:43.749740+00:00 / 2026-08-23T08:29:18.349903+00:00
- **Path:** `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_gap_ds_v4f_aav_n_gf_20260823_mm_gap_flash_ds_v4f_aav_n_gf`
- **Knobs:** `{"benches": "autosa_mm", "dse": true, "latency_opt": false, "model": "deepseek-v4-flash", "skill_prompt_mode": "all_skills_avoids_global", "skills_pack": "gemm_flatten_v1", "stream": true, "turns": 4, "variant": "autosa_aav_n_gf", "workflow": "autosa_flash"}`
- **Cell:** `deepseek-v4-flash__flash__autosa__aav_n_gf`
  - flash: **24745–24745** cycles, DSP **80**
  - dse: **13160–13160** cycles, DSP **352**
  - stream: **4299–4299** cycles, DSP **352**
  - selected: **4299–4299** cycles, DSP **352**
  - flash skills injected=129 routed=None pack_sha=847494f44a9b4a6e `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_gap_ds_v4f_aav_n_gf_20260823_mm_gap_flash_ds_v4f_aav_n_gf/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_flash_skills.json`
  - code flash_opt: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_gap_ds_v4f_aav_n_gf_20260823_mm_gap_flash_ds_v4f_aav_n_gf/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_flash_opt.cpp`
  - code selected: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_gap_ds_v4f_aav_n_gf_20260823_mm_gap_flash_ds_v4f_aav_n_gf/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_selected.cpp`
  - code dse: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_gap_ds_v4f_aav_n_gf_20260823_mm_gap_flash_ds_v4f_aav_n_gf/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_dse.cpp`
  - code stream: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_gap_ds_v4f_aav_n_gf_20260823_mm_gap_flash_ds_v4f_aav_n_gf/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_stream.cpp`
- **LLM flash_combined:** 8 msgs, calls=0, tokens=0 `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_gap_ds_v4f_aav_n_gf_20260823_mm_gap_flash_ds_v4f_aav_n_gf/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_history.json`
- **LLM dse:** 3 msgs, calls=None, tokens=None `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_gap_ds_v4f_aav_n_gf_20260823_mm_gap_flash_ds_v4f_aav_n_gf/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_dse_history.json`
- **LLM stream:** 3 msgs, calls=None, tokens=None `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_gap_ds_v4f_aav_n_gf_20260823_mm_gap_flash_ds_v4f_aav_n_gf/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_stream_history.json`

### `batch_parallel_autosa_mm_getting_started_flash_onchip_t7_20260908_010347_autosa_mm_getting_started`

- **Family / aim:** flash_onchip — Flash-only: distilled on-chip GEMM pack (940-class), not 90-skills.
- **Status:** complete  **Frozen:** False
- **Created / done:** 2026-09-08T01:04:35.313815+00:00 / 2026-09-08T01:28:03.979924+00:00
- **Path:** `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_getting_started_flash_onchip_t7_20260908_010347_autosa_mm_getting_started`
- **Knobs:** `{"autosa_flow": true, "benches": "autosa_mm_getting_started", "dse": "0", "enforcement": false, "flash_min_dsp": 500, "flash_onchip": 1, "flash_pe_blk": 16, "flash_skill_bin": "onchip", "model": "deepseek-v4-flash", "packaged_skills_json": "/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/hls_full_optimization_skills_schema_1_1_package/flash_onchip_wide_gemm_skill_entries.json", "post_flash_dse": false, "post_flash_stream": false, "stream": "0", "synth_timeout": 14400, "turns": 7, "variant": "autosa_onchip_gemm", "workflow": "autosa_flash"}`
- **Cell:** `deepseek-v4-flash__flash__autosa__onchip_gemm`
  - flash: **900–900** cycles, DSP **5088**
  - selected: **900–900** cycles, DSP **5088**
  - flash skills injected=9 routed=None pack_sha=5c7ee088b5820199 `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_getting_started_flash_onchip_t7_20260908_010347_autosa_mm_getting_started/variants/autosa_onchip_gemm/autosa_mm_getting_started/deepseek-v4-flash__flash__autosa__onchip_gemm/autosa_mm_getting_started_flash_skills.json`
  - code flash_opt: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_getting_started_flash_onchip_t7_20260908_010347_autosa_mm_getting_started/variants/autosa_onchip_gemm/autosa_mm_getting_started/deepseek-v4-flash__flash__autosa__onchip_gemm/autosa_mm_getting_started_flash_opt.cpp`
  - code selected: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_getting_started_flash_onchip_t7_20260908_010347_autosa_mm_getting_started/variants/autosa_onchip_gemm/autosa_mm_getting_started/deepseek-v4-flash__flash__autosa__onchip_gemm/autosa_mm_getting_started_selected.cpp`
- **LLM flash_combined:** 7 msgs, calls=0, tokens=0 `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_getting_started_flash_onchip_t7_20260908_010347_autosa_mm_getting_started/variants/autosa_onchip_gemm/autosa_mm_getting_started/deepseek-v4-flash__flash__autosa__onchip_gemm/autosa_mm_getting_started_history.json`

### `batch_parallel_autosa_mm_getting_started_seed_synth_20260908_124628_autosa_mm_getting_started`

- **Family / aim:** other_kernel_seed_synth — Seed csynth of other AutoSA mm-family kernels (no LLM).
- **Status:** complete  **Frozen:** False
- **Created / done:** 2026-09-08T12:46:32.398672+00:00 / 2026-09-08T12:48:51.858126+00:00
- **Path:** `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_getting_started_seed_synth_20260908_124628_autosa_mm_getting_started`
- **Knobs:** `{"autosa_flow": true, "benches": "autosa_mm_getting_started", "dse": "0", "enforcement": false, "model": "none", "post_flash_dse": false, "post_flash_stream": false, "skip_phase_b": true, "stream": "0", "synth_timeout": 14400, "turns": 1, "variant": "autosa_gold", "workflow": "autosa_gold"}`
- **Cell:** `batch_parallel_autosa_mm_getting_started_seed_synth_20260908_124628_autosa_mm_getting_started`

### `batch_parallel_autosa_mm_hbm_flash_onchip_t7_20260908_010347_autosa_mm_hbm`

- **Family / aim:** flash_onchip — Flash-only: distilled on-chip GEMM pack (940-class), not 90-skills.
- **Status:** complete  **Frozen:** False
- **Created / done:** 2026-09-08T01:04:46.662183+00:00 / 2026-09-08T01:18:51.541260+00:00
- **Path:** `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_hbm_flash_onchip_t7_20260908_010347_autosa_mm_hbm`
- **Knobs:** `{"autosa_flow": true, "benches": "autosa_mm_hbm", "dse": "0", "enforcement": false, "flash_min_dsp": 500, "flash_onchip": 1, "flash_pe_blk": 16, "flash_skill_bin": "onchip", "model": "deepseek-v4-flash", "packaged_skills_json": "/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/hls_full_optimization_skills_schema_1_1_package/flash_onchip_wide_gemm_skill_entries.json", "post_flash_dse": false, "post_flash_stream": false, "stream": "0", "synth_timeout": 14400, "turns": 7, "variant": "autosa_onchip_gemm", "workflow": "autosa_flash"}`
- **Cell:** `deepseek-v4-flash__flash__autosa__onchip_gemm`
  - flash: **938–938** cycles, DSP **5344**
  - selected: **938–938** cycles, DSP **5344**
  - flash skills injected=9 routed=None pack_sha=5c7ee088b5820199 `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_hbm_flash_onchip_t7_20260908_010347_autosa_mm_hbm/variants/autosa_onchip_gemm/autosa_mm_hbm/deepseek-v4-flash__flash__autosa__onchip_gemm/autosa_mm_hbm_flash_skills.json`
  - code flash_opt: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_hbm_flash_onchip_t7_20260908_010347_autosa_mm_hbm/variants/autosa_onchip_gemm/autosa_mm_hbm/deepseek-v4-flash__flash__autosa__onchip_gemm/autosa_mm_hbm_flash_opt.cpp`
  - code selected: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_hbm_flash_onchip_t7_20260908_010347_autosa_mm_hbm/variants/autosa_onchip_gemm/autosa_mm_hbm/deepseek-v4-flash__flash__autosa__onchip_gemm/autosa_mm_hbm_selected.cpp`
- **LLM flash_combined:** 10 msgs, calls=0, tokens=0 `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_hbm_flash_onchip_t7_20260908_010347_autosa_mm_hbm/variants/autosa_onchip_gemm/autosa_mm_hbm/deepseek-v4-flash__flash__autosa__onchip_gemm/autosa_mm_hbm_history.json`

### `batch_parallel_autosa_mm_hbm_seed_synth_20260908_124628_autosa_mm_hbm`

- **Family / aim:** other_kernel_seed_synth — Seed csynth of other AutoSA mm-family kernels (no LLM).
- **Status:** complete  **Frozen:** False
- **Created / done:** 2026-09-08T12:46:34.862265+00:00 / 2026-09-08T12:48:51.863179+00:00
- **Path:** `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_hbm_seed_synth_20260908_124628_autosa_mm_hbm`
- **Knobs:** `{"autosa_flow": true, "benches": "autosa_mm_hbm", "dse": "0", "enforcement": false, "model": "none", "post_flash_dse": false, "post_flash_stream": false, "skip_phase_b": true, "stream": "0", "synth_timeout": 14400, "turns": 1, "variant": "autosa_gold", "workflow": "autosa_gold"}`
- **Cell:** `batch_parallel_autosa_mm_hbm_seed_synth_20260908_124628_autosa_mm_hbm`

### `batch_parallel_autosa_mm_hcl_flash_onchip_t7_20260907_dryrun`

- **Family / aim:** flash_onchip — Flash-only: distilled on-chip GEMM pack (940-class), not 90-skills.
- **Status:** running  **Frozen:** False
- **Created / done:** 2026-09-07T14:45:28.740899+00:00 / None
- **Path:** `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_hcl_flash_onchip_t7_20260907_dryrun`
- **Knobs:** `{"autosa_flow": true, "benches": "autosa_mm_hcl", "dse": "0", "enforcement": false, "flash_min_dsp": 500, "flash_onchip": 1, "flash_pe_blk": 16, "flash_skill_bin": "onchip", "model": "deepseek-v4-flash", "packaged_skills_json": "/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/hls_full_optimization_skills_schema_1_1_package/flash_onchip_wide_gemm_skill_entries.json", "post_flash_dse": false, "post_flash_stream": false, "stream": "0", "synth_timeout": 14400, "turns": 7, "variant": "autosa_onchip_gemm", "workflow": "autosa_flash"}`

### `batch_parallel_autosa_mm_hcl_flash_onchip_t7_20260908_002240`

- **Family / aim:** flash_onchip — Flash-only: distilled on-chip GEMM pack (940-class), not 90-skills.
- **Status:** complete  **Frozen:** False
- **Created / done:** 2026-09-08T00:22:41.512527+00:00 / 2026-09-08T00:36:03.196789+00:00
- **Path:** `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_hcl_flash_onchip_t7_20260908_002240`
- **Knobs:** `{"autosa_flow": true, "benches": "autosa_mm_hcl", "dse": "0", "enforcement": false, "flash_min_dsp": 500, "flash_onchip": 1, "flash_pe_blk": 16, "flash_skill_bin": "onchip", "model": "deepseek-v4-flash", "packaged_skills_json": "/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/hls_full_optimization_skills_schema_1_1_package/flash_onchip_wide_gemm_skill_entries.json", "post_flash_dse": false, "post_flash_stream": false, "stream": "0", "synth_timeout": 14400, "turns": 7, "variant": "autosa_onchip_gemm", "workflow": "autosa_flash"}`
- **Cell:** `deepseek-v4-flash__flash__autosa__onchip_gemm`
  - flash: **906–906** cycles, DSP **5344**
  - selected: **906–906** cycles, DSP **5344**
  - flash skills injected=8 routed=None pack_sha=cd663db975c92279 `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_hcl_flash_onchip_t7_20260908_002240/variants/autosa_onchip_gemm/autosa_mm_hcl/deepseek-v4-flash__flash__autosa__onchip_gemm/autosa_mm_hcl_flash_skills.json`
  - code flash_opt: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_hcl_flash_onchip_t7_20260908_002240/variants/autosa_onchip_gemm/autosa_mm_hcl/deepseek-v4-flash__flash__autosa__onchip_gemm/autosa_mm_hcl_flash_opt.cpp`
  - code selected: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_hcl_flash_onchip_t7_20260908_002240/variants/autosa_onchip_gemm/autosa_mm_hcl/deepseek-v4-flash__flash__autosa__onchip_gemm/autosa_mm_hcl_selected.cpp`
- **LLM flash_combined:** 7 msgs, calls=0, tokens=0 `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_hcl_flash_onchip_t7_20260908_002240/variants/autosa_onchip_gemm/autosa_mm_hcl/deepseek-v4-flash__flash__autosa__onchip_gemm/autosa_mm_hcl_history.json`

### `batch_parallel_autosa_mm_hcl_flash_onchip_t7_20260908_004300`

- **Family / aim:** flash_onchip — Flash-only: distilled on-chip GEMM pack (940-class), not 90-skills.
- **Status:** complete  **Frozen:** False
- **Created / done:** 2026-09-08T00:43:00.542168+00:00 / 2026-09-08T00:58:06.522302+00:00
- **Path:** `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_hcl_flash_onchip_t7_20260908_004300`
- **Knobs:** `{"autosa_flow": true, "benches": "autosa_mm_hcl", "dse": "0", "enforcement": false, "flash_min_dsp": 500, "flash_onchip": 1, "flash_pe_blk": 16, "flash_skill_bin": "onchip", "model": "deepseek-v4-flash", "packaged_skills_json": "/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/hls_full_optimization_skills_schema_1_1_package/flash_onchip_wide_gemm_skill_entries.json", "post_flash_dse": false, "post_flash_stream": false, "stream": "0", "synth_timeout": 14400, "turns": 7, "variant": "autosa_onchip_gemm", "workflow": "autosa_flash"}`
- **Cell:** `deepseek-v4-flash__flash__autosa__onchip_gemm`
  - flash: **938–938** cycles, DSP **5344**
  - selected: **938–938** cycles, DSP **5344**
  - flash skills injected=9 routed=None pack_sha=5c7ee088b5820199 `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_hcl_flash_onchip_t7_20260908_004300/variants/autosa_onchip_gemm/autosa_mm_hcl/deepseek-v4-flash__flash__autosa__onchip_gemm/autosa_mm_hcl_flash_skills.json`
  - code flash_opt: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_hcl_flash_onchip_t7_20260908_004300/variants/autosa_onchip_gemm/autosa_mm_hcl/deepseek-v4-flash__flash__autosa__onchip_gemm/autosa_mm_hcl_flash_opt.cpp`
  - code selected: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_hcl_flash_onchip_t7_20260908_004300/variants/autosa_onchip_gemm/autosa_mm_hcl/deepseek-v4-flash__flash__autosa__onchip_gemm/autosa_mm_hcl_selected.cpp`
- **LLM flash_combined:** 7 msgs, calls=0, tokens=0 `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_hcl_flash_onchip_t7_20260908_004300/variants/autosa_onchip_gemm/autosa_mm_hcl/deepseek-v4-flash__flash__autosa__onchip_gemm/autosa_mm_hcl_history.json`

### `batch_parallel_autosa_mm_hcl_intel_flash_onchip_t7_20260908_010347_autosa_mm_hcl_intel`

- **Family / aim:** flash_onchip — Flash-only: distilled on-chip GEMM pack (940-class), not 90-skills.
- **Status:** complete  **Frozen:** False
- **Created / done:** 2026-09-08T01:04:27.554756+00:00 / 2026-09-08T01:19:28.251931+00:00
- **Path:** `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_hcl_intel_flash_onchip_t7_20260908_010347_autosa_mm_hcl_intel`
- **Knobs:** `{"autosa_flow": true, "benches": "autosa_mm_hcl_intel", "dse": "0", "enforcement": false, "flash_min_dsp": 500, "flash_onchip": 1, "flash_pe_blk": 16, "flash_skill_bin": "onchip", "model": "deepseek-v4-flash", "packaged_skills_json": "/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/hls_full_optimization_skills_schema_1_1_package/flash_onchip_wide_gemm_skill_entries.json", "post_flash_dse": false, "post_flash_stream": false, "stream": "0", "synth_timeout": 14400, "turns": 7, "variant": "autosa_onchip_gemm", "workflow": "autosa_flash"}`
- **Cell:** `deepseek-v4-flash__flash__autosa__onchip_gemm`
  - flash: **1032–1032** cycles, DSP **5120**
  - selected: **1032–1032** cycles, DSP **5120**
  - flash skills injected=9 routed=None pack_sha=5c7ee088b5820199 `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_hcl_intel_flash_onchip_t7_20260908_010347_autosa_mm_hcl_intel/variants/autosa_onchip_gemm/autosa_mm_hcl_intel/deepseek-v4-flash__flash__autosa__onchip_gemm/autosa_mm_hcl_intel_flash_skills.json`
  - code flash_opt: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_hcl_intel_flash_onchip_t7_20260908_010347_autosa_mm_hcl_intel/variants/autosa_onchip_gemm/autosa_mm_hcl_intel/deepseek-v4-flash__flash__autosa__onchip_gemm/autosa_mm_hcl_intel_flash_opt.cpp`
  - code selected: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_hcl_intel_flash_onchip_t7_20260908_010347_autosa_mm_hcl_intel/variants/autosa_onchip_gemm/autosa_mm_hcl_intel/deepseek-v4-flash__flash__autosa__onchip_gemm/autosa_mm_hcl_intel_selected.cpp`
- **LLM flash_combined:** 7 msgs, calls=0, tokens=0 `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_hcl_intel_flash_onchip_t7_20260908_010347_autosa_mm_hcl_intel/variants/autosa_onchip_gemm/autosa_mm_hcl_intel/deepseek-v4-flash__flash__autosa__onchip_gemm/autosa_mm_hcl_intel_history.json`

### `batch_parallel_autosa_mm_hcl_intel_flash_onchip_t7_20260908_045607_autosa_mm_hcl_intel`

- **Family / aim:** flash_onchip — Flash-only: distilled on-chip GEMM pack (940-class), not 90-skills.
- **Status:** complete  **Frozen:** False
- **Created / done:** 2026-09-08T04:56:32.521315+00:00 / 2026-09-08T05:08:35.908257+00:00
- **Path:** `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_hcl_intel_flash_onchip_t7_20260908_045607_autosa_mm_hcl_intel`
- **Knobs:** `{"autosa_flow": true, "benches": "autosa_mm_hcl_intel", "dse": "0", "enforcement": false, "flash_max_dsp": 9024, "flash_min_dsp": 5000, "flash_onchip": 1, "flash_pe_blk": 16, "flash_skill_bin": "onchip", "model": "deepseek-v4-flash", "packaged_skills_json": "/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/hls_full_optimization_skills_schema_1_1_package/flash_onchip_wide_gemm_skill_entries.json", "post_flash_dse": false, "post_flash_stream": false, "stream": "0", "synth_timeout": 14400, "turns": 7, "variant": "autosa_onchip_gemm", "workflow": "autosa_flash"}`
- **Cell:** `deepseek-v4-flash__flash__autosa__onchip_gemm`
  - flash: **1002–1002** cycles, DSP **5344**
  - selected: **1002–1002** cycles, DSP **5344**
  - flash skills injected=12 routed=None pack_sha=2cf2d92d9723a0f5 `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_hcl_intel_flash_onchip_t7_20260908_045607_autosa_mm_hcl_intel/variants/autosa_onchip_gemm/autosa_mm_hcl_intel/deepseek-v4-flash__flash__autosa__onchip_gemm/autosa_mm_hcl_intel_flash_skills.json`
  - code flash_opt: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_hcl_intel_flash_onchip_t7_20260908_045607_autosa_mm_hcl_intel/variants/autosa_onchip_gemm/autosa_mm_hcl_intel/deepseek-v4-flash__flash__autosa__onchip_gemm/autosa_mm_hcl_intel_flash_opt.cpp`
  - code selected: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_hcl_intel_flash_onchip_t7_20260908_045607_autosa_mm_hcl_intel/variants/autosa_onchip_gemm/autosa_mm_hcl_intel/deepseek-v4-flash__flash__autosa__onchip_gemm/autosa_mm_hcl_intel_selected.cpp`
- **LLM flash_combined:** 7 msgs, calls=0, tokens=0 `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_hcl_intel_flash_onchip_t7_20260908_045607_autosa_mm_hcl_intel/variants/autosa_onchip_gemm/autosa_mm_hcl_intel/deepseek-v4-flash__flash__autosa__onchip_gemm/autosa_mm_hcl_intel_history.json`

### `batch_parallel_autosa_mm_hcl_intel_seed_synth_20260908_124628_autosa_mm_hcl_intel`

- **Family / aim:** other_kernel_seed_synth — Seed csynth of other AutoSA mm-family kernels (no LLM).
- **Status:** complete  **Frozen:** False
- **Created / done:** 2026-09-08T12:46:29.649120+00:00 / 2026-09-08T12:48:51.972627+00:00
- **Path:** `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_hcl_intel_seed_synth_20260908_124628_autosa_mm_hcl_intel`
- **Knobs:** `{"autosa_flow": true, "benches": "autosa_mm_hcl_intel", "dse": "0", "enforcement": false, "model": "none", "post_flash_dse": false, "post_flash_stream": false, "skip_phase_b": true, "stream": "0", "synth_timeout": 14400, "turns": 1, "variant": "autosa_gold", "workflow": "autosa_gold"}`
- **Cell:** `batch_parallel_autosa_mm_hcl_intel_seed_synth_20260908_124628_autosa_mm_hcl_intel`

### `batch_parallel_autosa_mm_hcl_seed_synth_20260908_124628_autosa_mm_hcl`

- **Family / aim:** other_kernel_seed_synth — Seed csynth of other AutoSA mm-family kernels (no LLM).
- **Status:** complete  **Frozen:** False
- **Created / done:** 2026-09-08T12:46:29.214700+00:00 / 2026-09-08T12:48:51.428415+00:00
- **Path:** `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_hcl_seed_synth_20260908_124628_autosa_mm_hcl`
- **Knobs:** `{"autosa_flow": true, "benches": "autosa_mm_hcl", "dse": "0", "enforcement": false, "model": "none", "post_flash_dse": false, "post_flash_stream": false, "skip_phase_b": true, "stream": "0", "synth_timeout": 14400, "turns": 1, "variant": "autosa_gold", "workflow": "autosa_gold"}`
- **Cell:** `batch_parallel_autosa_mm_hcl_seed_synth_20260908_124628_autosa_mm_hcl`

### `batch_parallel_autosa_mm_hcl_seed_synth_20260908_dryseed_hcl`

- **Family / aim:** other_kernel_seed_synth — Seed csynth of other AutoSA mm-family kernels (no LLM).
- **Status:** running  **Frozen:** False
- **Created / done:** 2026-09-08T12:46:03.328919+00:00 / None
- **Path:** `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_hcl_seed_synth_20260908_dryseed_hcl`
- **Knobs:** `{"autosa_flow": true, "benches": "autosa_mm_hcl", "dse": "0", "enforcement": false, "model": "none", "post_flash_dse": false, "post_flash_stream": false, "skip_phase_b": true, "stream": "0", "synth_timeout": 14400, "turns": 1, "variant": "autosa_gold", "workflow": "autosa_gold"}`

### `batch_parallel_autosa_mm_int16_flash_onchip_t7_20260908_010347_autosa_mm_int16`

- **Family / aim:** flash_onchip — Flash-only: distilled on-chip GEMM pack (940-class), not 90-skills.
- **Status:** complete  **Frozen:** False
- **Created / done:** 2026-09-08T01:04:30.297484+00:00 / 2026-09-08T01:21:58.618318+00:00
- **Path:** `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_int16_flash_onchip_t7_20260908_010347_autosa_mm_int16`
- **Knobs:** `{"autosa_flow": true, "benches": "autosa_mm_int16", "dse": "0", "enforcement": false, "flash_min_dsp": 500, "flash_onchip": 1, "flash_pe_blk": 16, "flash_skill_bin": "onchip", "model": "deepseek-v4-flash", "packaged_skills_json": "/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/hls_full_optimization_skills_schema_1_1_package/flash_onchip_wide_gemm_skill_entries.json", "post_flash_dse": false, "post_flash_stream": false, "stream": "0", "synth_timeout": 14400, "turns": 7, "variant": "autosa_onchip_gemm", "workflow": "autosa_flash"}`
- **Cell:** `deepseek-v4-flash__flash__autosa__onchip_gemm`
  - flash: **863–863** cycles, DSP **1024**
  - selected: **863–863** cycles, DSP **1024**
  - flash skills injected=9 routed=None pack_sha=5c7ee088b5820199 `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_int16_flash_onchip_t7_20260908_010347_autosa_mm_int16/variants/autosa_onchip_gemm/autosa_mm_int16/deepseek-v4-flash__flash__autosa__onchip_gemm/autosa_mm_int16_flash_skills.json`
  - code flash_opt: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_int16_flash_onchip_t7_20260908_010347_autosa_mm_int16/variants/autosa_onchip_gemm/autosa_mm_int16/deepseek-v4-flash__flash__autosa__onchip_gemm/autosa_mm_int16_flash_opt.cpp`
  - code selected: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_int16_flash_onchip_t7_20260908_010347_autosa_mm_int16/variants/autosa_onchip_gemm/autosa_mm_int16/deepseek-v4-flash__flash__autosa__onchip_gemm/autosa_mm_int16_selected.cpp`
- **LLM flash_combined:** 7 msgs, calls=0, tokens=0 `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_int16_flash_onchip_t7_20260908_010347_autosa_mm_int16/variants/autosa_onchip_gemm/autosa_mm_int16/deepseek-v4-flash__flash__autosa__onchip_gemm/autosa_mm_int16_history.json`

### `batch_parallel_autosa_mm_int16_flash_onchip_t7_20260908_045550`

- **Family / aim:** flash_onchip — Flash-only: distilled on-chip GEMM pack (940-class), not 90-skills.
- **Status:** running  **Frozen:** False
- **Created / done:** 2026-09-08T04:55:50.367465+00:00 / None
- **Path:** `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_int16_flash_onchip_t7_20260908_045550`
- **Knobs:** `{"autosa_flow": true, "benches": "autosa_mm_int16", "dse": "0", "enforcement": false, "flash_max_dsp": 9024, "flash_min_dsp": 5000, "flash_onchip": 1, "flash_pe_blk": 16, "flash_row_uf": 8, "flash_skill_bin": "onchip", "model": "deepseek-v4-flash", "packaged_skills_json": "/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/hls_full_optimization_skills_schema_1_1_package/flash_onchip_wide_gemm_skill_entries.json", "post_flash_dse": false, "post_flash_stream": false, "stream": "0", "synth_timeout": 14400, "turns": 7, "variant": "autosa_onchip_gemm", "workflow": "autosa_flash"}`

### `batch_parallel_autosa_mm_int16_flash_onchip_t7_20260908_045607_autosa_mm_int16`

- **Family / aim:** flash_onchip — Flash-only: distilled on-chip GEMM pack (940-class), not 90-skills.
- **Status:** complete  **Frozen:** False
- **Created / done:** 2026-09-08T04:56:32.980843+00:00 / 2026-09-08T05:40:40.769290+00:00
- **Path:** `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_int16_flash_onchip_t7_20260908_045607_autosa_mm_int16`
- **Knobs:** `{"autosa_flow": true, "benches": "autosa_mm_int16", "dse": "0", "enforcement": false, "flash_max_dsp": 9024, "flash_min_dsp": 5000, "flash_onchip": 1, "flash_pe_blk": 16, "flash_row_uf": 8, "flash_skill_bin": "onchip", "model": "deepseek-v4-flash", "packaged_skills_json": "/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/hls_full_optimization_skills_schema_1_1_package/flash_onchip_wide_gemm_skill_entries.json", "post_flash_dse": false, "post_flash_stream": false, "stream": "0", "synth_timeout": 14400, "turns": 7, "variant": "autosa_onchip_gemm", "workflow": "autosa_flash"}`
- **Cell:** `deepseek-v4-flash__flash__autosa__onchip_gemm`
  - flash: **381–381** cycles, DSP **8192**
  - selected: **381–381** cycles, DSP **8192**
  - flash skills injected=12 routed=None pack_sha=2cf2d92d9723a0f5 `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_int16_flash_onchip_t7_20260908_045607_autosa_mm_int16/variants/autosa_onchip_gemm/autosa_mm_int16/deepseek-v4-flash__flash__autosa__onchip_gemm/autosa_mm_int16_flash_skills.json`
  - code flash_opt: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_int16_flash_onchip_t7_20260908_045607_autosa_mm_int16/variants/autosa_onchip_gemm/autosa_mm_int16/deepseek-v4-flash__flash__autosa__onchip_gemm/autosa_mm_int16_flash_opt.cpp`
  - code selected: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_int16_flash_onchip_t7_20260908_045607_autosa_mm_int16/variants/autosa_onchip_gemm/autosa_mm_int16/deepseek-v4-flash__flash__autosa__onchip_gemm/autosa_mm_int16_selected.cpp`
- **LLM flash_combined:** 8 msgs, calls=0, tokens=0 `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_int16_flash_onchip_t7_20260908_045607_autosa_mm_int16/variants/autosa_onchip_gemm/autosa_mm_int16/deepseek-v4-flash__flash__autosa__onchip_gemm/autosa_mm_int16_history.json`

### `batch_parallel_autosa_mm_int16_seed_synth_20260908_124628_autosa_mm_int16`

- **Family / aim:** other_kernel_seed_synth — Seed csynth of other AutoSA mm-family kernels (no LLM).
- **Status:** complete  **Frozen:** False
- **Created / done:** 2026-09-08T12:46:30.529506+00:00 / 2026-09-08T12:48:51.913832+00:00
- **Path:** `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_int16_seed_synth_20260908_124628_autosa_mm_int16`
- **Knobs:** `{"autosa_flow": true, "benches": "autosa_mm_int16", "dse": "0", "enforcement": false, "model": "none", "post_flash_dse": false, "post_flash_stream": false, "skip_phase_b": true, "stream": "0", "synth_timeout": 14400, "turns": 1, "variant": "autosa_gold", "workflow": "autosa_gold"}`
- **Cell:** `batch_parallel_autosa_mm_int16_seed_synth_20260908_124628_autosa_mm_int16`

### `batch_parallel_autosa_mm_intel_flash_onchip_t7_20260908_010347_autosa_mm_intel`

- **Family / aim:** flash_onchip — Flash-only: distilled on-chip GEMM pack (940-class), not 90-skills.
- **Status:** complete  **Frozen:** False
- **Created / done:** 2026-09-08T01:04:34.114219+00:00 / 2026-09-08T01:17:31.211681+00:00
- **Path:** `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_intel_flash_onchip_t7_20260908_010347_autosa_mm_intel`
- **Knobs:** `{"autosa_flow": true, "benches": "autosa_mm_intel", "dse": "0", "enforcement": false, "flash_min_dsp": 500, "flash_onchip": 1, "flash_pe_blk": 16, "flash_skill_bin": "onchip", "model": "deepseek-v4-flash", "packaged_skills_json": "/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/hls_full_optimization_skills_schema_1_1_package/flash_onchip_wide_gemm_skill_entries.json", "post_flash_dse": false, "post_flash_stream": false, "stream": "0", "synth_timeout": 14400, "turns": 7, "variant": "autosa_onchip_gemm", "workflow": "autosa_flash"}`
- **Cell:** `deepseek-v4-flash__flash__autosa__onchip_gemm`
  - flash: **938–938** cycles, DSP **5344**
  - selected: **938–938** cycles, DSP **5344**
  - flash skills injected=9 routed=None pack_sha=5c7ee088b5820199 `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_intel_flash_onchip_t7_20260908_010347_autosa_mm_intel/variants/autosa_onchip_gemm/autosa_mm_intel/deepseek-v4-flash__flash__autosa__onchip_gemm/autosa_mm_intel_flash_skills.json`
  - code flash_opt: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_intel_flash_onchip_t7_20260908_010347_autosa_mm_intel/variants/autosa_onchip_gemm/autosa_mm_intel/deepseek-v4-flash__flash__autosa__onchip_gemm/autosa_mm_intel_flash_opt.cpp`
  - code selected: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_intel_flash_onchip_t7_20260908_010347_autosa_mm_intel/variants/autosa_onchip_gemm/autosa_mm_intel/deepseek-v4-flash__flash__autosa__onchip_gemm/autosa_mm_intel_selected.cpp`
- **LLM flash_combined:** 7 msgs, calls=0, tokens=0 `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_intel_flash_onchip_t7_20260908_010347_autosa_mm_intel/variants/autosa_onchip_gemm/autosa_mm_intel/deepseek-v4-flash__flash__autosa__onchip_gemm/autosa_mm_intel_history.json`

### `batch_parallel_autosa_mm_intel_seed_synth_20260908_124628_autosa_mm_intel`

- **Family / aim:** other_kernel_seed_synth — Seed csynth of other AutoSA mm-family kernels (no LLM).
- **Status:** complete  **Frozen:** False
- **Created / done:** 2026-09-08T12:46:31.436689+00:00 / 2026-09-08T12:48:51.934001+00:00
- **Path:** `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_intel_seed_synth_20260908_124628_autosa_mm_intel`
- **Knobs:** `{"autosa_flow": true, "benches": "autosa_mm_intel", "dse": "0", "enforcement": false, "model": "none", "post_flash_dse": false, "post_flash_stream": false, "skip_phase_b": true, "stream": "0", "synth_timeout": 14400, "turns": 1, "variant": "autosa_gold", "workflow": "autosa_gold"}`
- **Cell:** `batch_parallel_autosa_mm_intel_seed_synth_20260908_124628_autosa_mm_intel`

### `batch_parallel_autosa_wave1_aav_n_gf_20260825_062035`

- **Family / aim:** wave1_other_kernels — Other AutoSA kernels toward 2% of their rank-1.
- **Status:** running  **Frozen:** False
- **Created / done:** 2026-08-25T06:20:36.035236+00:00 / None
- **Path:** `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_wave1_aav_n_gf_20260825_062035`
- **Knobs:** `{"benches": "autosa_mm_hcl,autosa_mm_hcl_intel,autosa_mm_intel,autosa_mm_int16,autosa_mm_catapult,autosa_mm_getting_started", "dse": true, "latency_opt": false, "model": "deepseek-v4-flash", "skill_prompt_mode": "all_skills_avoids_global", "skills_pack": "gemm_flatten_v1", "stream": true, "turns": 4, "variant": "autosa_aav_n_gf", "workflow": "autosa_flash"}`

### `batch_parallel_autosa_wave1_aav_n_gf_20260825_wave1_aav_n_gf`

- **Family / aim:** wave1_other_kernels — Other AutoSA kernels toward 2% of their rank-1.
- **Status:** complete  **Frozen:** False
- **Created / done:** 2026-08-25T06:27:04.793962+00:00 / 2026-08-25T08:55:09.041654+00:00
- **Path:** `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_wave1_aav_n_gf_20260825_wave1_aav_n_gf`
- **Knobs:** `{"benches": "autosa_mm_hcl,autosa_mm_hcl_intel,autosa_mm_intel,autosa_mm_int16,autosa_mm_catapult,autosa_mm_getting_started", "dse": true, "latency_opt": false, "model": "deepseek-v4-flash", "skill_prompt_mode": "all_skills_avoids_global", "skills_pack": "gemm_flatten_v1", "stream": true, "turns": 4, "variant": "autosa_aav_n_gf", "workflow": "autosa_flash"}`
- **Cell:** `deepseek-v4-flash__flash__autosa__aav_n_gf`
  - dse: **17688–17688** cycles, DSP **96**
  - stream: **8351–8351** cycles, DSP **96**
  - selected: **8351–8351** cycles, DSP **96**
  - code selected: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_wave1_aav_n_gf_20260825_wave1_aav_n_gf/variants/autosa_aav_n_gf/autosa_mm_catapult/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_catapult_selected.cpp`
  - code dse: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_wave1_aav_n_gf_20260825_wave1_aav_n_gf/variants/autosa_aav_n_gf/autosa_mm_catapult/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_catapult_dse.cpp`
  - code stream: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_wave1_aav_n_gf_20260825_wave1_aav_n_gf/variants/autosa_aav_n_gf/autosa_mm_catapult/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_catapult_stream.cpp`
- **Cell:** `deepseek-v4-flash__flash__autosa__aav_n_gf`
  - flash: **12588–12588** cycles, DSP **352**
  - dse: **10619–10619** cycles, DSP **704**
  - stream: **4525–4525** cycles, DSP **640**
  - selected: **4525–4525** cycles, DSP **640**
  - flash skills injected=129 routed=None pack_sha=847494f44a9b4a6e `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_wave1_aav_n_gf_20260825_wave1_aav_n_gf/variants/autosa_aav_n_gf/autosa_mm_getting_started/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_getting_started_flash_skills.json`
  - code flash_opt: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_wave1_aav_n_gf_20260825_wave1_aav_n_gf/variants/autosa_aav_n_gf/autosa_mm_getting_started/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_getting_started_flash_opt.cpp`
  - code selected: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_wave1_aav_n_gf_20260825_wave1_aav_n_gf/variants/autosa_aav_n_gf/autosa_mm_getting_started/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_getting_started_selected.cpp`
  - code dse: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_wave1_aav_n_gf_20260825_wave1_aav_n_gf/variants/autosa_aav_n_gf/autosa_mm_getting_started/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_getting_started_dse.cpp`
  - code stream: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_wave1_aav_n_gf_20260825_wave1_aav_n_gf/variants/autosa_aav_n_gf/autosa_mm_getting_started/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_getting_started_stream.cpp`
- **Cell:** `deepseek-v4-flash__flash__autosa__aav_n_gf`
  - dse: **12770–12770** cycles, DSP **384**
  - stream: **4294–4294** cycles, DSP **320**
  - selected: **4294–4294** cycles, DSP **320**
  - code selected: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_wave1_aav_n_gf_20260825_wave1_aav_n_gf/variants/autosa_aav_n_gf/autosa_mm_hcl/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_hcl_selected.cpp`
  - code dse: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_wave1_aav_n_gf_20260825_wave1_aav_n_gf/variants/autosa_aav_n_gf/autosa_mm_hcl/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_hcl_dse.cpp`
  - code stream: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_wave1_aav_n_gf_20260825_wave1_aav_n_gf/variants/autosa_aav_n_gf/autosa_mm_hcl/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_hcl_stream.cpp`
- **Cell:** `deepseek-v4-flash__flash__autosa__aav_n_gf`
  - flash: **35085–35085** cycles, DSP **48**
  - dse: **17485–17485** cycles, DSP **352**
  - stream: **4292–4292** cycles, DSP **320**
  - selected: **4292–4292** cycles, DSP **320**
  - flash skills injected=129 routed=None pack_sha=847494f44a9b4a6e `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_wave1_aav_n_gf_20260825_wave1_aav_n_gf/variants/autosa_aav_n_gf/autosa_mm_hcl_intel/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_hcl_intel_flash_skills.json`
  - code flash_opt: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_wave1_aav_n_gf_20260825_wave1_aav_n_gf/variants/autosa_aav_n_gf/autosa_mm_hcl_intel/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_hcl_intel_flash_opt.cpp`
  - code selected: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_wave1_aav_n_gf_20260825_wave1_aav_n_gf/variants/autosa_aav_n_gf/autosa_mm_hcl_intel/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_hcl_intel_selected.cpp`
  - code dse: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_wave1_aav_n_gf_20260825_wave1_aav_n_gf/variants/autosa_aav_n_gf/autosa_mm_hcl_intel/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_hcl_intel_dse.cpp`
  - code stream: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_wave1_aav_n_gf_20260825_wave1_aav_n_gf/variants/autosa_aav_n_gf/autosa_mm_hcl_intel/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_hcl_intel_stream.cpp`
- **Cell:** `deepseek-v4-flash__flash__autosa__aav_n_gf`
  - flash: **4509–4509** cycles, DSP **64**
  - dse: **8626–8626** cycles, DSP **64**
  - stream: **4280–4280** cycles, DSP **64**
  - selected: **4280–4280** cycles, DSP **64**
  - flash skills injected=129 routed=None pack_sha=847494f44a9b4a6e `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_wave1_aav_n_gf_20260825_wave1_aav_n_gf/variants/autosa_aav_n_gf/autosa_mm_int16/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_int16_flash_skills.json`
  - code flash_opt: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_wave1_aav_n_gf_20260825_wave1_aav_n_gf/variants/autosa_aav_n_gf/autosa_mm_int16/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_int16_flash_opt.cpp`
  - code selected: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_wave1_aav_n_gf_20260825_wave1_aav_n_gf/variants/autosa_aav_n_gf/autosa_mm_int16/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_int16_selected.cpp`
  - code dse: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_wave1_aav_n_gf_20260825_wave1_aav_n_gf/variants/autosa_aav_n_gf/autosa_mm_int16/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_int16_dse.cpp`
  - code stream: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_wave1_aav_n_gf_20260825_wave1_aav_n_gf/variants/autosa_aav_n_gf/autosa_mm_int16/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_int16_stream.cpp`
- **Cell:** `deepseek-v4-flash__flash__autosa__aav_n_gf`
  - flash: **29457–29457** cycles, DSP **64**
  - dse: **10750–10750** cycles, DSP **704**
  - stream: **4525–4525** cycles, DSP **640**
  - selected: **4525–4525** cycles, DSP **640**
  - flash skills injected=129 routed=None pack_sha=847494f44a9b4a6e `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_wave1_aav_n_gf_20260825_wave1_aav_n_gf/variants/autosa_aav_n_gf/autosa_mm_intel/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_intel_flash_skills.json`
  - code flash_opt: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_wave1_aav_n_gf_20260825_wave1_aav_n_gf/variants/autosa_aav_n_gf/autosa_mm_intel/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_intel_flash_opt.cpp`
  - code selected: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_wave1_aav_n_gf_20260825_wave1_aav_n_gf/variants/autosa_aav_n_gf/autosa_mm_intel/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_intel_selected.cpp`
  - code dse: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_wave1_aav_n_gf_20260825_wave1_aav_n_gf/variants/autosa_aav_n_gf/autosa_mm_intel/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_intel_dse.cpp`
  - code stream: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_wave1_aav_n_gf_20260825_wave1_aav_n_gf/variants/autosa_aav_n_gf/autosa_mm_intel/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_intel_stream.cpp`
- **LLM flash_combined:** 3 msgs, calls=None, tokens=None `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_wave1_aav_n_gf_20260825_wave1_aav_n_gf/variants/autosa_aav_n_gf/autosa_mm_catapult/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_catapult_stream_history.json`
- **LLM dse:** 3 msgs, calls=None, tokens=None `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_wave1_aav_n_gf_20260825_wave1_aav_n_gf/variants/autosa_aav_n_gf/autosa_mm_catapult/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_catapult_dse_history.json`
- **LLM stream:** 3 msgs, calls=None, tokens=None `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_wave1_aav_n_gf_20260825_wave1_aav_n_gf/variants/autosa_aav_n_gf/autosa_mm_catapult/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_catapult_stream_history.json`
- **LLM flash_combined:** 3 msgs, calls=None, tokens=None `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_wave1_aav_n_gf_20260825_wave1_aav_n_gf/variants/autosa_aav_n_gf/autosa_mm_getting_started/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_getting_started_stream_history.json`
- **LLM dse:** 3 msgs, calls=None, tokens=None `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_wave1_aav_n_gf_20260825_wave1_aav_n_gf/variants/autosa_aav_n_gf/autosa_mm_getting_started/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_getting_started_dse_history.json`
- **LLM stream:** 3 msgs, calls=None, tokens=None `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_wave1_aav_n_gf_20260825_wave1_aav_n_gf/variants/autosa_aav_n_gf/autosa_mm_getting_started/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_getting_started_stream_history.json`
- **LLM flash_combined:** 7 msgs, calls=None, tokens=None `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_wave1_aav_n_gf_20260825_wave1_aav_n_gf/variants/autosa_aav_n_gf/autosa_mm_hcl/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_hcl_stream_history.json`
- **LLM dse:** 3 msgs, calls=None, tokens=None `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_wave1_aav_n_gf_20260825_wave1_aav_n_gf/variants/autosa_aav_n_gf/autosa_mm_hcl/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_hcl_dse_history.json`
- **LLM stream:** 7 msgs, calls=None, tokens=None `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_wave1_aav_n_gf_20260825_wave1_aav_n_gf/variants/autosa_aav_n_gf/autosa_mm_hcl/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_hcl_stream_history.json`
- **LLM flash_combined:** 9 msgs, calls=None, tokens=None `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_wave1_aav_n_gf_20260825_wave1_aav_n_gf/variants/autosa_aav_n_gf/autosa_mm_hcl_intel/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_hcl_intel_overlap_history.json`
- **LLM dse:** 3 msgs, calls=None, tokens=None `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_wave1_aav_n_gf_20260825_wave1_aav_n_gf/variants/autosa_aav_n_gf/autosa_mm_hcl_intel/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_hcl_intel_dse_history.json`
- **LLM stream:** 3 msgs, calls=None, tokens=None `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_wave1_aav_n_gf_20260825_wave1_aav_n_gf/variants/autosa_aav_n_gf/autosa_mm_hcl_intel/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_hcl_intel_stream_history.json`
- **LLM flash_combined:** 3 msgs, calls=None, tokens=None `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_wave1_aav_n_gf_20260825_wave1_aav_n_gf/variants/autosa_aav_n_gf/autosa_mm_int16/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_int16_stream_history.json`
- **LLM dse:** 3 msgs, calls=None, tokens=None `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_wave1_aav_n_gf_20260825_wave1_aav_n_gf/variants/autosa_aav_n_gf/autosa_mm_int16/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_int16_dse_history.json`
- **LLM stream:** 3 msgs, calls=None, tokens=None `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_wave1_aav_n_gf_20260825_wave1_aav_n_gf/variants/autosa_aav_n_gf/autosa_mm_int16/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_int16_stream_history.json`
- **LLM flash_combined:** 9 msgs, calls=None, tokens=None `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_wave1_aav_n_gf_20260825_wave1_aav_n_gf/variants/autosa_aav_n_gf/autosa_mm_intel/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_intel_overlap_history.json`
- **LLM dse:** 3 msgs, calls=None, tokens=None `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_wave1_aav_n_gf_20260825_wave1_aav_n_gf/variants/autosa_aav_n_gf/autosa_mm_intel/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_intel_dse_history.json`
- **LLM stream:** 3 msgs, calls=None, tokens=None `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_wave1_aav_n_gf_20260825_wave1_aav_n_gf/variants/autosa_aav_n_gf/autosa_mm_intel/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_intel_stream_history.json`

### `compact_pe_io_search`

- **Family / aim:** family_c_io_mesh — IO-mesh search (family C). First campaign frozen.
- **Status:** unknown  **Frozen:** False
- **Created / done:** None / None
- **Path:** `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/compact_pe_io_search`
- **Knobs:** `{"benches": "", "model": null, "turns": null, "variant": null, "workflow": null}`

### `compact_pe_io_search_20260831_io`

- **Family / aim:** family_c_io_mesh — IO-mesh search (family C). First campaign frozen.
- **Status:** unknown  **Frozen:** False
- **Created / done:** None / None
- **Path:** `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/compact_pe_io_search_20260831_io`
- **Knobs:** `{"benches": "", "model": null, "turns": null, "variant": null, "workflow": null}`
- **Cell:** `compact_pe_io_search_20260831_io`

### `compact_pe_io_search_20260831_io2`

- **Family / aim:** family_c_io_mesh — IO-mesh search (family C). First campaign frozen.
- **Status:** unknown  **Frozen:** True
- **Created / done:** None / None
- **Path:** `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/compact_pe_io_search_20260831_io2`
- **Knobs:** `{"benches": "", "model": null, "turns": null, "variant": null, "workflow": null}`
- **Cell:** `compact_pe_io_search_20260831_io2`

### `compact_pe_pack_search`

- **Family / aim:** family_b_pack — Packing search (family B).
- **Status:** unknown  **Frozen:** False
- **Created / done:** None / None
- **Path:** `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/compact_pe_pack_search`
- **Knobs:** `{"benches": "", "model": null, "turns": null, "variant": null, "workflow": null}`

### `compact_pe_pack_search_20260831_pack`

- **Family / aim:** family_b_pack — Packing search (family B).
- **Status:** unknown  **Frozen:** False
- **Created / done:** None / None
- **Path:** `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/compact_pe_pack_search_20260831_pack`
- **Knobs:** `{"benches": "", "model": null, "turns": null, "variant": null, "workflow": null}`
- **Cell:** `compact_pe_pack_search_20260831_pack`

### `compact_pe_search`

- **Family / aim:** family_a_pe_search — Packed PE-count search (family A).
- **Status:** unknown  **Frozen:** False
- **Created / done:** None / None
- **Path:** `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/compact_pe_search`
- **Knobs:** `{"benches": "", "model": null, "turns": null, "variant": null, "workflow": null}`

### `compact_pe_search_20260830_pesearch`

- **Family / aim:** family_a_pe_search — Packed PE-count search (family A).
- **Status:** unknown  **Frozen:** False
- **Created / done:** None / None
- **Path:** `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/compact_pe_search_20260830_pesearch`
- **Knobs:** `{"benches": "", "model": null, "turns": null, "variant": null, "workflow": null}`
- **Cell:** `compact_pe_search_20260830_pesearch`

### `compact_pe_search_20260831_mesh`

- **Family / aim:** family_a_pe_search — Packed PE-count search (family A).
- **Status:** unknown  **Frozen:** False
- **Created / done:** None / None
- **Path:** `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/compact_pe_search_20260831_mesh`
- **Knobs:** `{"benches": "", "model": null, "turns": null, "variant": null, "workflow": null}`
- **Cell:** `compact_pe_search_20260831_mesh`

### `manual_mmflow_pe_pp`

- **Family / aim:** manual_handwritten — Hand-written kernels for ping-pong / PE16 reference (no LLM).
- **Status:** manual  **Frozen:** False
- **Created / done:** None / None
- **Path:** `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/manual_mmflow_pe_pp`
- **Knobs:** `{"benches": "", "model": null, "turns": null, "variant": null, "workflow": null}`
- **Cell:** `manual_mmflow_pe_pp`

### `manual_pe16_tile_pp`

- **Family / aim:** manual_handwritten — Hand-written kernels for ping-pong / PE16 reference (no LLM).
- **Status:** manual  **Frozen:** False
- **Created / done:** None / None
- **Path:** `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/manual_pe16_tile_pp`
- **Knobs:** `{"benches": "", "model": null, "turns": null, "variant": null, "workflow": null}`
- **Cell:** `manual_pe16_tile_pp`

### `manual_mm_lcst_tile_pp2`

- **Family / aim:** manual_handwritten — Hand-written kernels for ping-pong / PE16 reference (no LLM).
- **Status:** manual  **Frozen:** False
- **Created / done:** None / None
- **Path:** `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/manual_mm_lcst_tile_pp2`
- **Knobs:** `{"benches": "", "model": null, "turns": null, "variant": null, "workflow": null}`
- **Cell:** `manual_mm_lcst_tile_pp2`

### `manual_rank1_shaped`

- **Family / aim:** manual_handwritten — Hand-written kernels for ping-pong / PE16 reference (no LLM).
- **Status:** manual  **Frozen:** False
- **Created / done:** None / None
- **Path:** `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/manual_rank1_shaped`
- **Knobs:** `{"benches": "", "model": null, "turns": null, "variant": null, "workflow": null}`
- **Cell:** `manual_rank1_shaped`
  - code stream: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/manual_rank1_shaped/autosa_mm_rank1_io_stream.cpp`

## 5. Pytest files that lock these knobs

These are **unit tests** (no Vitis, no LLM except mocked). They exist so a campaign knob cannot silently re-contaminate flash.

| File | #tests | Aim | Env knobs |
|---|---:|---|---|
| `tests/test_flash_cosim_repair_multiloop_smoke.py` | 9 | Smoke tests for multi-loop cosim repair (offline, no LLM/Vitis). |  |
| `tests/test_flash_dataflow_ranking.py` | 3 | Generic flash may use DATAFLOW when it is simple; it must not mandate GEMM ping-pong. |  |
| `tests/test_flash_defer_cosim.py` | 0 | Tests for C2HLS_FLASH_DEFER_COSIM support in batch_parallel_bench. | C2HLS_FLASH_DEFER_COSIM |
| `tests/test_flash_df_candidate_rank.py` | 9 | Unit tests for flash/dataflow candidate ranking. |  |
| `tests/test_flash_dsp_floor.py` | 10 | Flash DSP floor: reject csynth DSP below cutoff and tell the LLM why. | C2HLS_FLASH_MAX_DSP, C2HLS_FLASH_MIN_DSP |
| `tests/test_flash_enforcement.py` | 44 | Flash overlap enforcement: ping-pong + DATAFLOW judged on code and csynth. | C2HLS_AUTOSA_FLOW, C2HLS_ENFORCEMENT, C2HLS_ENFORCEMENT_ROUNDS |
| `tests/test_flash_fused_ab.py` | 8 | On-chip flash must reject fused A+B loads (run-1 / 871 failure mode). | C2HLS_FLASH_ONCHIP |
| `tests/test_flash_generic_uncontaminated.py` | 4 | Default flash 90-skills and FLASH MODE must stay kernel-independent. AutoSA mm recipes (LANES=16 / PE_BLK / 64x64 ping-pong) belong in gemm_family, systolic_io, onchip, or C2HLS_FLASH_* knobs — not in the generic dump or q_optimize_flash. Frozen mmflow flash 139484/10 used these payloads. | C2HLS_FLASH_ |
| `tests/test_flash_gmem_bundles.py` | 0 | Flash flow must assign distinct gmemN bundles per m_axi pointer port. |  |
| `tests/test_flash_no_RMW_m_axi_skills.py` | 0 | Tests for flash standalone overlay on packaged 90-skills base. | C2HLS_FLASH_SKILL_ENTRIES_JSON, C2HLS_PACKAGED_SKILLS_JSON, C2HLS_PACKAGED_SKILLS_ONLY |
| `tests/test_flash_onchip_gemm.py` | 7 | Distilled on-chip GEMM flash pack (940-class), not the 90-skill dump. | C2HLS_FLASH_MIN_DSP, C2HLS_FLASH_ONCHIP, C2HLS_FLASH_ONLY, C2HLS_FLASH_PE_BLK, C2HLS_FLASH_SKILL_ENTRIES_JSON, C2HLS_FORCE_SKILL_PROMPTS, C2HLS_PACKAGED_SKILLS_JSON, C2HLS_PACKAGED_SKILLS_ONLY, C2HLS_ |
| `tests/test_flash_pe_blk.py` | 5 | Flash PE_BLK prompt knob: 16 / 32 / 64 on the full U280 DSP budget. | C2HLS_FLASH_ONLY, C2HLS_FLASH_PE_BLK |
| `tests/test_flash_pipelined_finalize_chain.py` | 0 | Flash pipelined finalize should chain dse + stream + pragma_opt + latency_opt like c2hls.py. |  |
| `tests/test_flash_pipelined_queue.py` | 0 | Tests for pipelined flash queue. |  |
| `tests/test_flash_row_uf.py` | 4 | Flash ROW_UF prompt knob: tell the model to cover I in one tile. | C2HLS_FLASH_ONLY, C2HLS_FLASH_ROW_UF |
| `tests/test_flash_skill_bins.py` | 15 | Curated flash skill bins for AutoSA kernels other than frozen autosa_mm. | C2HLS_FLASH_K_TILE, C2HLS_FLASH_MAX_DSP, C2HLS_FLASH_MIN_DSP, C2HLS_FLASH_ONCHIP, C2HLS_FLASH_ONCHIP_TILE, C2HLS_FLASH_ONLY, C2HLS_FLASH_OPT_PROMPT_MODE, C2HLS_FLASH_PE_BLK, C2HLS_FLASH_RETRY_TAG, C2H |
| `tests/test_flash_skip_seed.py` | 8 | Skip-flash seed: enforcement starts from an existing flash_opt, not a new LLM. | C2HLS_AUTOSA_FLOW, C2HLS_ENFORCEMENT, C2HLS_ENFORCEMENT_ROUNDS, C2HLS_FLASH_SEED_DIR, C2HLS_SKIP_FLASH, C2HLS_SKIP_PHASE_B |
| `tests/test_flash_tile_pp.py` | 6 | Flash in-GEMM tile ping-pong knob (C2HLS_FLASH_TILE_PP). | C2HLS_FLASH_ONLY, C2HLS_FLASH_PE_BLK, C2HLS_FLASH_TILE_PP |
| `tests/test_flash_wide_io.py` | 5 | Wide AXI is opt-in. Default 90-skills stay generic LCST. |  |
| `tests/test_autosa_flow_gates.py` | 22 |  |  |
| `tests/test_autosa_mm_flow_flavors.py` | 0 | autosa_mm flow flavors: zero-shot and no-skills vs frozen with-skills. | C2HLS_CPP_CONTINUATIONS, C2HLS_DSE_CHAIN_FLASH, C2HLS_DSE_SKILL_ENTRIES_JSON, C2HLS_FLASH_MAX_TOKENS, C2HLS_FLASH_OPT_PROMPT_MODE, C2HLS_FLASH_SKILL_ENTRIES_JSON, C2HLS_FORCE_SKILL_PROMPTS, C2HLS_MM_F |
| `tests/test_post_flash_dataflow.py` | 17 | Tests for post-flash DATAFLOW helpers. |  |
| `tests/test_post_flash_dse.py` | 20 | Tests for the post-flash DSE step (multi-PE GEMM nest rewrite). | C2HLS_DSE_CHAIN_FLASH, C2HLS_DSE_MAX_TOKENS, C2HLS_DSE_SOURCE_ROLE, C2HLS_FLASH_MAX_TOKENS, C2HLS_LLM_EMPTY_RETRIES, C2HLS_LLM_MAX_TOKENS, C2HLS_POST_FLASH_DSE |
| `tests/test_post_flash_latency_opt.py` | 28 |  | C2HLS_LATENCY_OPT_CHAIN_FLASH, C2HLS_LATENCY_OPT_REPAIR_ROUNDS, C2HLS_LATENCY_OPT_ROUNDS, C2HLS_POST_FLASH_LATENCY_OPT |
| `tests/test_post_flash_mem_parallel.py` | 10 | Tests for post-flash memory parallelism helpers. |  |
| `tests/test_post_flash_overlap.py` | 5 | Overlap ablation: PE array + fuse + ping-pong DATAFLOW, not the stream FIFO pack. | C2HLS_MM_MATRIX_ROOT |
| `tests/test_post_flash_pe_recipe.py` | 11 | Per-kernel PE recipes for AutoSA 64^3 GEMM DSE/stream. | C2HLS_PE_RECIPE |
| `tests/test_post_flash_pragma_opt.py` | 4 | Tests for post-flash pragma optimization helpers. |  |
| `tests/test_post_flash_stream.py` | 15 | Tests for the post-DSE stream/I/O step (PE modules + hls::stream DATAFLOW). | C2HLS_DSE_MAX_TOKENS, C2HLS_FLASH_MAX_TOKENS, C2HLS_LLM_MAX_TOKENS, C2HLS_POST_FLASH_STREAM, C2HLS_STREAM_CHAIN_FLASH, C2HLS_STREAM_MAX_TOKENS |
| `tests/test_autosa_seed_synth.py` | 5 | Gold-gate seed synth path for AutoSA-ready kernels (no LLM, no flash). | C2HLS_FLASH_ONCHIP, C2HLS_FLASH_ONLY, C2HLS_PACKAGED_SKILLS_JSON, C2HLS_REFERENCE_ONLY, C2HLS_TMP_ROOT |

## 6. Launchers (how a campaign is supposed to be started)

All under `scripts/pc2/`. Unset `BATCH_PARALLEL_STAMP` and flash knobs you do not want before launch.

| Script | Family |
|---|---|
| `start_autosa_mm_flow.sh` | mmflow 3-stage (skills / noskills / zero_shot via flavor) |
| `start_autosa_mm_enforcement.sh` | enforcement; `--seed-flash DIR` skip-flash |
| `start_autosa_flash_enforcement.sh` | same, generic prefix |
| `start_autosa_mm_flash_dsp_floor.sh` | C2HLS_FLASH_MIN_DSP |
| `start_autosa_mm_flash_pe_blk.sh` | C2HLS_FLASH_PE_BLK |
| `start_autosa_mm_flash_tile_pp.sh` | PE16 + TILE_PP |
| `start_autosa_mm_flash_row_uf.sh` | ROW_UF |
| `start_autosa_mm_flash_onchip.sh` | distilled on-chip pack |
| `start_autosa_mm_flash_dataflow.sh` | flash-only DATAFLOW prompt |
| `start_autosa_mm_gap_flash_deepseek_aav_n_gf.sh` | early gap flash |
| `start_autosa_mm_pe_search.sh / pack / io` | family A/B/C |

Do not overwrite frozen stamps listed in `FROZEN` in the exporter.

