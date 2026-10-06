# Plan: other AutoSA kernels after autosa_mm (spend-DSP primary)

**Date:** 2026-09-07  
**Branch:** `c2hls_enhanced_l_pc2_api_layout` (dirty; do not commit unless asked)  
**Does not overwrite:** Aug 18 / Sep 2 slide briefs, Sep 6 mm handoff, frozen `20260830_mmflow` / 32x8 / family A/B/C first campaigns / mm champion stamps.

Speak as compute rewrite / hide load-store. Never “DSE” on slides. Metrics: **latency min, latency max, DSP**. Never interval or HLS estimated clock.

## Status this session

| Check | Result |
|---|---|
| squeue | AutoSA SIMD16 jobs only: `2887969` R, `2887967_0` R, `2887970_0` PD, `2887971` PD. No c2hls flash job. |
| Flash LLM | `http://login5:18092/v1` **down** (connection refused on login1-6). No campaign submitted. |
| mm | Done. Do not relaunch as the mm study. |

## Two leaderboards (do not mix)

1. **Iso-compute** (wave1 2% vs rank-1) — already measured for six GEMMs; intel / getting_started still miss. Not the primary this round.
2. **Spend DSP** (mm champion **940 / 5344**, campaign `..._pe16_20260904_131622`). Primary for every remaining kernel that can legally take about 5000 DSP.

Fused A+B **871 / 5088** (cosim 1224) is a failure mode. Do not put 871 on a 4228 slide. Do not retarget fused A+B.

## Skill bins (transfer test, not filenames)

Do **not** dump the 90-skill pack. It is tagged kernel-independent but contaminated with PE_BLK / 64x64 GEMM ping-pong (`hls-baseline-load-compute-store-gate` in the three 90 JSON files).

| Bin | Pack JSON | Variant | What it tests |
|---|---|---|---|
| zero-shot | none | `autosa_zero_shot` | HLS-engineer prompt, no skills JSON |
| generic HLS | `flash_generic_hls_skill_entries.json` | `autosa_generic_hls` | pipeline / partition / AXI / LCST / no RMW; no PE_BLK, no 64x64 ping-pong |
| GEMM-family | `flash_gemm_family_skill_entries.json` | `autosa_gemm_family` | B is JxK, gold from zero, write-once C, no fused A+B, keep kernel.h ABI |
| systolic I/O | `flash_systolic_io_skill_entries.json` | `autosa_systolic_io` | hide load-store with PE streams + DATAFLOW; not spend-chip |
| spend-DSP on-chip | `flash_onchip_wide_gemm_skill_entries.json` (8 skills) | `autosa_onchip_gemm` | 940-class LCST, 512-bit, affine C, PE_BLK 16/32/64 |

Gate: `C2HLS_FLASH_SKILL_BIN`. Spend-DSP also sets `C2HLS_FLASH_ONCHIP=1`.

Launcher (any kernel except frozen mm stamps):

```bash
./scripts/pc2/start_autosa_kernel_flash.sh --kernel autosa_mm_hcl --pack onchip --endpoint-url http://login5:18092/v1
```

mm-only distilled launcher (unchanged): `scripts/pc2/start_autosa_mm_flash_onchip.sh`.

## Inventory (plain.cpp seed, ABI from kernel.h)

Working-set “fits” = A+B+C (or cin/w/cout) under about 8 MB BRAM-comfortable. U280 DSP budget **9024**. Float MAC about 5 DSP; uint16 about 1; uint32 about 3.

| Kernel | Size / dtype | Rank-1 (csynth) | Wave1 agent | Fits on-chip? | 5000+ DSP? |
|---|---|---|---|---|---|
| `autosa_mm` | 64^3 float | 4228 / 320 | frozen 4292 / 320 stream | yes 48 KB | **yes** — champion 940 / 5344. **Skip.** |
| `autosa_mm_hcl` | 64^3 float | 4230 / 320 | 4294 / 320 (2% pass) | yes | **yes** — **first** spend-DSP target |
| `autosa_mm_hcl_intel` | 64^3 float | 4226 / 320 | 4292 / 320 (2% pass) | yes | **yes** |
| `autosa_mm_int16` | 64^3 uint16 | 4219 / 64 | 4280 / 64 (2% pass) | yes | **yes with extra row unroll** (1 DSP/MAC; PE_BLK=16 times K=64 is about 1024). PE_BLK=64 about 4096; ROW_UF>=5 times PE_BLK=16 times 64 reaches 5000. |
| `autosa_mm_catapult` | 64^3 uint32, `I_P/J_P/K_P` | 8286 / 96 | 8351 / 96 (2% pass) | yes | **yes at PE_BLK=32** (about 3 DSP/MAC; PE_BLK=16 about 3072) |
| `autosa_mm_intel` | 64^3 float | 4178 / 640 | 4525 / 640 (miss 8.3%) | yes | **yes** |
| `autosa_mm_getting_started` | 64^3 float | 2194 / 640 | 4525 / 640 (miss about 2x; rank-1 is two DATAFLOW tiles) | yes | **yes** |
| `autosa_mm_block_sparse` | 64^3 float (plain is dense; sparsity macros unused) | none | — | yes | **yes** |
| `autosa_mm_hbm` | 64^3 float | none | — | yes | **yes** (user order: after CNN/LU) |
| `autosa_cnn` | O,I,R,C=16 K=3 float | 18554 / 160 | — | yes about 46 KB | **yes** if O times I times 9 MACs unrolled (cap at 9024) |
| `autosa_lu` | N=32 float | none | — | yes about 12 KB | **no** — k is sequential (div + rank-1 update). Ablation + highest legal DSP. |
| `autosa_dnn_ops` | FC 16x16 float (`#define FC`) | none | — | yes about 1 KB | **no** — full unroll <= 256 MACs times about 5 = **1280**. Ablation + that ceiling. |
| `autosa_large_mm` | 208x512x256 int | 524403 / 156 | — | yes about 1.1 MB | **yes tiled** (K=256 times PE_BLK=16 int about 4096; add a row group to clear 5000) |
| `autosa_large_mm_intel` | 1040x1024x1024 float | 13631621 / 400 | — | **no** (about 12.7 MB) | **yes on a tile**; do not complete-partition full matrices |
| `autosa_large_mm_int16` | 1024^3 uint16 | none | — | borderline (about 6.4 MB) | **yes on a tile** |
| `autosa_large_mm_int8` | 1056x1024x1024 int8 | none | — | yes about 3.2 MB | **yes tiled** (workspace fits; still tile K for II=1) |
| `autosa_large_mm_block_sparse` | 1024^3 float | none | — | **no** (about 12.6 MB) | tile |
| `autosa_large_cnn` | O=640 I=512 R,C=56 K=3 float | 144607317 (5.0 ns XML; re-csynth 3.33 before scoring) | — | **no** | tile; CNN PE pack later |
| `autosa_large_ttm` | 264x256x256x256 float | export failed | — | **no** (about 69 MB A) | tile |
| `autosa_large_ttmc` | 128^5 float | export failed | — | **no** | tile |
| `autosa_large_mttkrp` | 256x336x256x256 float | 50337174 (5.0 ns XML) | — | **no** | tile |

Gold: C = A·B from **zero** where the kernel is GEMM. B is **JxK** on mm-family. Seed is **plain.cpp**, not a stripped AutoSA netlist.

## Run protocol (one kernel, one flash job)

Primary pack = **onchip** (940-class). Ladder already learned on mm; first hcl run ships the champion recipe, then one failure-mode fix per retry:

1. min DSP 500 (or inventory `min_dsp_onchip` if the kernel cannot take 5000)
2. 512-bit I/O, three bundles
3. write-once affine C (nested i / j += LANES or PE_BLK; no linearized g, no RMW)
4. PE_BLK=16 (32 for catapult; extra ROW_UF for int16)

Failure modes to reject between runs: fused A+B in one loop, II>1, DSP far below 5000 (when legal), DATAFLOW ping-pong on a full workspace that already fits, AutoSA netlist clone, RMW, narrow AXI.

**10 runs per kernel**, real csynth. Cosim off until a champion. New campaign stamp every submit. Unset `BATCH_PARALLEL_ARTIFACT_PREFIX` first.

Order: **hcl → hcl_intel → int16 → catapult → intel → getting_started → block_sparse → cnn → lu → hbm → remaining large_***.

Ablation (zero-shot / generic / GEMM-family / systolic I/O / onchip) after the spend-DSP onchip arm has a legal csynth, still **one flash job at a time**.

## Launch (when login5:18092 is up)

```bash
# First GEMM variant, spend-DSP, flash-only. Does not touch mmflow / pe16 champion.
./scripts/pc2/start_autosa_kernel_flash.sh \
  --kernel autosa_mm_hcl --pack onchip \
  --endpoint-url http://login5:18092/v1
```

Dry-run (no LLM): add `--dry-run`.

## Tests

`.venv/bin/python -m pytest tests/test_flash_skill_bins.py tests/test_flash_onchip_gemm.py tests/test_flash_pe_blk.py tests/test_flash_dsp_floor.py tests/test_flash_wide_io.py`

## Blockers

1. **DeepSeek-v4-flash endpoint down** — cannot start hcl until `curl http://login5:18092/v1/models` succeeds. Do not overlap two flash jobs; do not collide with AutoSA SIMD16 jobs.
2. Frozen mm campaigns stay read-only.
