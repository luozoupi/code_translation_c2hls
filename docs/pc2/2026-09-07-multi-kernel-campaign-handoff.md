# Handoff: multi-kernel campaign after `autosa_mm`

**Date:** 2026-09-07
**Source:** Cursor Agent chat on Otus, 2026-09-07 (workspace disconnects; no jobs launched)
**Purpose:** continue this work in a **new** chat on a better connection. Paste this file (or `@` it) as the starting context. For the completed `autosa_mm` chronology, also read `docs/pc2/2026-09-06-autosa-mm-gap-chat-handoff.md`.

**Repo:** `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls`
**Branch:** `c2hls_enhanced_l_pc2_api_layout` (often dirty; **do not commit unless asked**)
**Transcript (read-only, not resumable):**
`/pc2/users/h/haqc2/.cursor/projects/scratch-hpc-prf-llmfpga-asa582-projects-c2hls/agent-transcripts/e9f5a8c7-d6e5-4607-b99a-e2cc0a5e5391/`

**Status of this campaign as of 2026-09-07 07:22 PT:** **not started.** Inventory, skill/template curation, 10-run protocol, and new HLS jobs were requested. Two background agents died on workspace disconnect before writing any new `docs/pc2/` plan, before editing skill JSON, and before submitting a campaign. Frozen mm artifacts were not touched.

---

## 0. How to pick this up in a new chat

> Continue c2hls after `autosa_mm`. Read `docs/pc2/2026-09-07-multi-kernel-campaign-handoff.md` and `docs/pc2/2026-09-06-autosa-mm-gap-chat-handoff.md`. mm iso-compute is locked (4216–4292 vs 4228 at 320 DSP). Spend-DSP champion is **940 / 5344**, campaign `..._pe16_20260904_131622`. Now run the **same ablation** + the **~960-cycle high-DSP (5000+)** flash flow on the **other GEMM variants and remaining kernels**. Curate skills/templates (do not dump 90 skills; do not clone `kernel0`). **10 runs per kernel**, fix between runs. Do not overwrite frozen mmflow / 32×8 / family C / Aug 18–Sep 2 briefs. Metrics: **latency min/max + DSP only**. One flash job at a time on `http://login5:18092/v1`. If a kernel cannot take 5000 DSP, document why and still run ablation + highest legal DSP.

Then:

1. `squeue -u $USER` — do not collide with an existing flash codegen job.
2. Confirm the flash endpoint: `curl -sS -m 3 http://login5:18092/v1/models`.
3. Open the mm champion cpp + csynth report before cloning the flow onto another kernel.
4. Write campaign results to a **new** dated file; do not edit frozen slide briefs.

**Pytest:** `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/.venv/bin/python -m pytest`
**Model:** DeepSeek-v4-flash. **Do not overlap two flash codegen jobs on that endpoint.**

---

## 1. What the supervisor study already closed (`autosa_mm` only)

Protocol: `autosa_mm`, I=J=K=64 float. ABI `extern "C" void autosa_mm(data_t A[I][K], B[J][K], C[I][J])`; **B is J×K**. Gold C=A·B from **zero**. Seed is `plain.cpp`, not stripped AutoSA `kernel0`. Device U280 `xcu280-fsvh2892-2L-e`, Vitis HLS 2023.2, 3.33 ns.

On slides: **flash / multi-PE (compute rewrite) / stream**. **Never say “DSE”.** Never quote **interval** or HLS **estimated clock**.

**Two leaderboards — never mix.**

### Leaderboard 1 — iso-compute (16 PE × SIMD 4 ≈ 320 DSP)

| Design | Latency | DSP |
|---|---:|---:|
| AutoSA rank-1 | **4228** | **320** |
| Aug 18 stream (`nav_n`) | **4285** | 320 |
| Frozen mmflow stream | **4292** | 320 |
| No-skills stream (`20260902_mmns`) | **4216** | 320 |
| Zero-shot flash (`20260902_mmzs`) | **40454–42758** | 320 |
| mmflow flash (90-skill) | **139484** | **10** |
| Compute rewrite | **13160** | 352 |
| Enforcement ping-pong DATAFLOW | **12893** | 320 |
| 32×8 stream | **4583** | 1280 |

Four-column ablation Fang asked for:

| Col | Given | Cycles | DSP |
|---|---|---:|---:|
| 1 | No expertise (zero-shot) | 40454–42758 | 320 |
| 2 | Generic HLS 90-skill flash | 139484 | 10 |
| 3 | Skills + ping-pong/DATAFLOW enforcement | 12893 | 320 |
| 4a | Systolic rewrite, with skills | 4292 | 320 |
| 4b | Same PE recipe, skills JSON empty | 4216 | 320 |

Column 2 is **slower than column 1**. Column 3 has 320 DSP and a DATAFLOW shape; one-run finish is still load+compute+store (overlap gate **fails**). Column 4 is hide load/store on a 16×4 array. **PE recipe + stream skeleton > 90-skill dump.** Reverse-engineering `kernel0` is **not** the method.

Wave1 2% gate (`batch_parallel_autosa_wave1_aav_n_gf_20260825_wave1_aav_n_gf`; `docs/pc2/2026-09-02-wave1-vs-rank1.md`):

| Bench | Agent | Rank-1 | DSP | Gate |
|---|---:|---:|---:|---|
| `autosa_mm_hcl` | 4294 | 4230 | 320 | pass 1.51% |
| `autosa_mm_hcl_intel` | 4292 | 4226 | 320 | pass 1.56% |
| `autosa_mm_int16` | 4280 | 4219 | 64 | pass 1.45% |
| `autosa_mm_catapult` | 8351 | 8286 | 96 | pass 0.78% |
| `autosa_mm_intel` | **4525** | 4178 | 640 | **fail 8.3%** |
| `autosa_mm_getting_started` | **4525** | 2194 | 640 | **fail ~2×** |

CNN / LU / HBM / large_* were **not** that round. Cand-9 efficiency (**1846–2161 / ~1296 DSP**) is **not** closed.

### Leaderboard 2 — spend DSP (~960-cycle / 5000+ DSP)

Flash-only, one knob per step. This is the **flow to clone**, not a fair beat of 4228.

| One change | Campaign stamp | Csynth | DSP |
|---|---|---:|---:|
| DSP floor 500 | `…_dsp500_20260904_085931` | 9423 | 1344 |
| + 512-bit I/O | `…_wideio_20260904_100437` | 2760 | 636 |
| + write-once affine C | `…_computeii_20260904_120524` | 1261 | 2672 |
| + PE_BLK=16 | **`…_pe16_20260904_131622`** | **940** | **5344** |
| Legal 2-tile ping-pong | `manual_pe16_tile_pp` | 1074 | 5344 |
| LLM tile-pp | `…_tilepp_20260905_002017` | 3817 | 5120 |

940 equation: `max(load_A 259, load_B 258) + compute 340 + store 260`. Type **no** (not DATAFLOW). Two **independent** load **modules**, started together. Nested `i` / `j += LANES`. AXI `latency=32`. Cosim **PASS 1071**.

**Champion code:**
`artifacts/pc2/batch_parallel_autosa_mm_flash_dsp500_pe16_20260904_131622/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_selected.cpp`

**Champion csynth:**
`c2hls_tmp/batch_parallel_autosa_mm_flash_dsp500_pe16_20260904_131622/autosa_mm/hls_synth__flash_synth/hls_proj/sol1/syn/report/autosa_mm_csynth.rpt`

Ping-pong was **worse**: splits the 256-trip compute, pays depth 84 twice. Ping-pong arrays must be **inside** the DATAFLOW loop (outside → HLS 214-397, wrong RTL).

### Distilled 8-skill pack (not a 940 ablation step)

| Item | Path / value |
|---|---|
| Pack | `hls_full_optimization_skills_schema_1_1_package/flash_onchip_wide_gemm_skill_entries.json` |
| Launcher | `scripts/pc2/start_autosa_mm_flash_onchip.sh` |
| Gate | `C2HLS_FLASH_ONCHIP` |
| Tests | `tests/test_flash_onchip_gemm.py` |
| Guidance | `autosa_flow_gates.py` → `flash_onchip_initial_guidance()` |

Eight skills: `hls-onchip-stage-when-fits`, `hls-onchip-axi-512-three-bundles`, `hls-onchip-write-once-affine-c`, `hls-onchip-flatten-c-output-groups`, `hls-onchip-full-k-unroll-adder-tree`, `hls-onchip-partition-match-unroll`, `avoid-onchip-dataflow-when-io-matches-compute`, `avoid-onchip-kernel0-systolic`.

Mandatory guidance: ignore FLASH item 5; no DATAFLOW / `kernel0`; LCST ≈ `max(load_A,load_B)+compute+store`.

| Run | Csynth | Cosim | I/O shape |
|---|---|---|---|
| 7-turn `…onchip_t7_20260906_104733` | **871 / 5088** | **PASS 1224** | **Fused** `load_ab_blocks` linearized `t` |
| Repro 7-turn `…onchip_t7_20260906_123340` | **1002 / 5344** | not run | Split load_a/load_b, still linearized `t` |

**871 csynth → 1224 cosim is a failure mode, not a better champion.** Fusion stalls if **either** gmem0/gmem1 RVALID/ARREADY is 0; store waits BVALID inside the loop; linearized `t` → ~132k FF in load vs champion ~540. Csynth priced lockstep II=1; RTL did not. Fusion is **model invention**, not instructed. The pack said `max(load_A,load_B)` and “overlapping A/B loads”; it did **not** say nested `i` / `j += LANES` as a required step, and it never said “do not put A and B in one loop.” Flatten-C teaches linearized `tile`. Do **not** put 871/1224 on the 4228 slide. Do **not** call 871 a 940 ablation step.

Cosim TB must **return 1** on mismatch (official AutoSA TB often returns 0). Gold-from-zero examples: `artifacts/pc2/manual_pe16_tile_pp/testbench_cosim.cpp`, `artifacts/pc2/onchip_t7_cosim/testbench_cosim.cpp`.

Arm B (`TURNS=1`) **never successfully launched**. Nested rewrite of 871 (keep compute, champion nested loads) **not done**.

---

## 2. Task that disconnected (do this next)

User (2026-09-07): mm is done. Focus on **the other GEMM variants and the rest of the kernels**.

1. Same **ablation of each kernel as mm** (zero-shot / generic skills / enforcement-or-equivalent / systolic-or-spend-DSP champion).
2. **Curate** skills, templates, guidance, launchers — anything else the mm flow used.
3. Follow the mm **~960-cycle / 5000+ DSP** generation flow (champion 940 class), **not** the 320-DSP iso-compute target as primary.
4. **10 runs per kernel**, fix issues and problems **between** runs.
5. **Get the results** (real csynth: latency min/max + DSP + module table).

Primary leaderboard for this campaign: **spend DSP, 5000+**. Keep ablation columns so Fang can compare kernels the same way as mm. If a kernel cannot take 5000 DSP (tiny `int16`, tiny problem size, sparse, LU, etc.), document why and still run ablation + the highest **legal** DSP design.

### Methodological stance (do not drift)

- Generic vs application-specific is a **transfer test**, not a filename.
- Bins: **generic HLS / GEMM family / systolic-instance / spend-DSP on-chip**.
- The 90-skill dump is tagged kernel-independent but **contaminated** with PE_BLK / 64×64 GEMM ping-pong inside `hls-baseline-load-compute-store-gate`.
- Reverse-engineering AutoSA then asking the LLM to emit `kernel0` **cannot be the method**.
- Do not mix the two leaderboards in one table.

### Skill taxonomy to curate (transfer, not filename)

| Bin | Contents | mm effect | Next-kernel job |
|---|---|---|---|
| **Generic HLS** | INTERFACE, PIPELINE, UNROLL, PARTITION, burst, LCST, avoid illegal DATAFLOW. Must actually be kernel-independent. | Legal kernel. mmflow flash 139484/10. Can lose to zero-shot. Does not build 16×4. | Strip PE_BLK / 64×64 / GEMM ping-pong from anything labelled generic. |
| **GEMM-family** | Flatten i/k/j, write-once C, no `g>>k`, PE recipe, Crow partition on PE dim. | Compute rewrite 13160/352. | Generalize names; do not hard-code 64×64 where the kernel differs. |
| **Systolic I/O** (AutoSA **plan**, not netlist) | Streams, `ap_uint<128>`, A per PE, B forward, Crow ram_2p, flatten `pe_kj`, fuse init/drain, `mm_pe`. | 4292 / 4216 at 320 DSP. | Ablation arm only unless the kernel is on the iso-compute slide. |
| **Ping-pong PE** | `buf[2]`, DATAFLOW load(t+1)/compute(t)/store(t−1). | 12893; 8678. Not rank-1. | Not the 5000+ champion path. |
| **On-chip spend-DSP** | Stage A,B; 512-bit three bundles; write-once; PE_BLK=16; full-K tree; no DATAFLOW when compute≈I/O; **independent nested A/B loads (not fused)**. | 940 / 5344. | Clone this pack; add an explicit “do not fuse A and B into one loop” skill; prefer nested `i` / `j += LANES` over linearized `tile`. |

Packs stay separate. Stream skills in flash do not invent PEs.

### Known pack gaps to fix before/during the 10-run loop

- Pack never forbids fused `load_ab`. Add that as a required step / avoid-rule.
- Flatten-C linearized `tile` caused huge load FF. Champion I/O is nested.
- `avoid-onchip-dataflow` currently says “four sequential pipelines” **and** overlapping A/B loads. Make it: two **independent** load **functions/modules** (separate stall domains), then compute, then store.
- Instance constants (64×64, PE_BLK=16, LANES=16) belong in guards / kernel overlays, not in a “generic” pack.
- Non-GEMM (cnn, lu, dnn_ops, ttm, mttkrp): do **not** paste the GEMM 8-skill pack unchanged. Extract a truly generic HLS pack first, then a kernel-family overlay.

---

## 3. Kernel inventory (from `scripts/prepare_autosa_ready.py` + `scripts/pc2/autosa_rank1_u280_targets.json`)

Materialize seeds with `scripts/prepare_autosa_ready.py` into `related_work/benchmarks/autosa_ready/` if that tree is missing.

Wave1 launcher benches (`scripts/pc2/batch_parallel_autosa_wave1_aav_n_gf.json`):
`autosa_mm_hcl`, `autosa_mm_hcl_intel`, `autosa_mm_intel`, `autosa_mm_int16`, `autosa_mm_catapult`, `autosa_mm_getting_started`.

| Kernel | Wave | Rank-1 cycles | Rank-1 DSP | Notes |
|---|---|---:|---:|---|
| `autosa_mm` | done | 4228 | 320 | **Skip spend-DSP re-run unless asked.** Champion 940/5344 already exists. |
| `autosa_mm_hcl` | 1 | 4230 | 320 | Start here for GEMM transfer. Iso-compute already in 2%. |
| `autosa_mm_hcl_intel` | 1 | 4226 | 320 | Same. |
| `autosa_mm_int16` | 1 | 4219 | 64 | `unsigned short`. May not reach 5000 DSP legally — document. |
| `autosa_mm_catapult` | 1 | 8286 | 96 | `unsigned int`; macros `I_P,J_P,K_P`. |
| `autosa_mm_intel` | 1 | 4178 | 640 | Iso-compute **miss 2%**. PE recipe 32×4. |
| `autosa_mm_getting_started` | 1 | 2194 | 640 | Iso-compute **~2×**. Rank-1 is two DATAFLOW tiles. |
| `autosa_mm_hbm` | 1 | — | — | No rank-1 package yet. |
| `autosa_mm_block_sparse` | 2 | — | — | Sparse; no rank-1 package. |
| `autosa_large_mm` | 2 | 524403 | 156 | Working set may **not** fit on-chip. Do not force stage-when-fits. |
| `autosa_large_mm_intel` | 2 | 13631621 | 400 | Same. |
| `autosa_large_mm_int16` | 2 | — | — | |
| `autosa_large_mm_int8` | 2 | — | — | |
| `autosa_large_mm_block_sparse` | 2 | — | — | |
| `autosa_cnn` | 3 | 18554 | 160 | Not GEMM. Needs CNN PE/stream skills. |
| `autosa_lu` | 3 | — | — | No rank-1 package. |
| `autosa_dnn_ops` | 3 | — | — | No rank-1 package. |
| `autosa_large_cnn` | 3 | 144607317 | — | DSE XML at 5.0 ns; re-csynth at 3.33 ns before scoring. |
| `autosa_large_mttkrp` | 3 | 50337174 | — | Same 5.0 ns caveat. |
| `autosa_large_ttm` | 3 | — | — | Rank-1 export failed. |
| `autosa_large_ttmc` | 3 | — | — | Rank-1 export failed. |

**Run order:** remaining small GEMM variants first (hcl → hcl_intel → int16 → catapult → intel → getting_started → hbm), then large GEMM, then cnn / lu / dnn_ops / ttm / mttkrp. **One kernel at a time on the flash endpoint.**

Verify per-kernel ABI, `data_t`, B layout, and gold (C from zero unless the spec differs) **before** the first flash turn.

---

## 4. Ten-run protocol (per kernel)

Clone mm spend-DSP, not mmflow three-stage, as the **primary** generation:

```
plain.cpp
  → flash-only
  → distilled on-chip / family pack (not the 90-skill dump)
  → C2HLS_FLASH_MIN_DSP (start 500; target 5000+ when legal)
  → C2HLS_FLASH_PE_BLK=16 (or documented exception)
  → C2HLS_TURNS=7 unless a kernel needs a written exception
  → C2HLS_FLASH_ONLY=1 (do not chain compute/stream on this arm)
```

For **ablation columns** (separate campaigns, new stamps):

1. Zero-shot (empty / no skill JSON, no repair, or the mm `autosa_zero_shot` variant).
2. Generic HLS only (cleaned generic pack — **after** curation).
3. Enforcement / ping-pong DATAFLOW if the kernel is GEMM-family (optional; mm showed this is not the champion).
4. Spend-DSP on-chip champion (this campaign’s primary).
5. If Fang still wants iso-compute vs that kernel’s rank-1: three-stage compute rewrite + stream at the **rank-1 DSP**, never mixed into the 5000+ table.

Between each of the ~10 flash generations:

- Parse csynth **Latency min/max**, **DSP**, and the **module table** (load / compute / store depths).
- Fail and repair skills/guidance if: fused A+B in one loop, II>1 on the hot compute, DSP << 5000 when the size fits, DATAFLOW ping-pong on a fit-on-chip GEMM, `kernel0` clone, C RMW / `g>>k`, narrow AXI, linearized load `t` with huge FF.
- Persist a **new** campaign stamp each run. `unset BATCH_PARALLEL_ARTIFACT_PREFIX` first. Dedicated launchers must **assign** the prefix.
- Do not clobber frozen trees listed in §6.

Launchers to clone (mm-only today — generalize per kernel, do not point them at mmflow stamps):

```bash
# Distilled on-chip pack (940-class). Arm C default TURNS=7.
./scripts/pc2/start_autosa_mm_flash_onchip.sh --endpoint-url http://login5:18092/v1

# Dry-run first
./scripts/pc2/start_autosa_mm_flash_onchip.sh --dry-run
```

Existing mm-only knobs (`C2HLS_FLASH_MIN_DSP`, `C2HLS_FLASH_PE_BLK`, `C2HLS_FLASH_ONLY`, `C2HLS_PACKAGED_SKILLS_ONLY`) live in `scripts/pc2/start_autosa_mm_flash_onchip.sh` and `autosa_flash_lib.py`. A multi-kernel launcher should take a **bench name**, assign a **new** `BATCH_PARALLEL_ARTIFACT_PREFIX`, and must not inherit a stale prefix from the shell.

Synth timeout often 14400 s. `C2HLS_MAX_TOKENS` 65536 after Sep 1 length failures.

**Cosim:** off until a kernel has a spend-DSP champion worth checking. Then use a TB that returns 1 on mismatch.

---

## 5. What to write (new files only)

| File | When |
|---|---|
| This file | Already the campaign handoff. Update a “status” section if you actually launch; or write a sibling dated results doc. |
| `docs/pc2/2026-09-07-multi-kernel-results.md` (or later date) | Per-kernel 10-run tables, ablation columns, DSP, module depths, cosim if any, failures and what changed, still-running vs done. |
| Generalized skill JSON | e.g. `flash_onchip_wide_gemm_skill_entries.json` overlay **or** a new `flash_onchip_gemm_family_*.json` — do not mix into the 90-skill dump. |
| Truly generic HLS pack | New file. Tests like `tests/test_flash_onchip_gemm.py`. |
| Per-family launchers | `scripts/pc2/start_autosa_<bench>_flash_onchip.sh` or one launcher with `BENCH=`. |

Do **not** overwrite:

- `docs/pc2/2026-08-18-autosa-mm-agent-vs-rank1-slide-brief.md`
- `docs/pc2/2026-09-02-supervisor-remaining-slide-brief.md`
- `docs/pc2/2026-09-02-wave1-vs-rank1.md`
- `docs/pc2/2026-09-02-why-320-dsp.md`
- `docs/pc2/2026-09-06-autosa-mm-gap-chat-handoff.md` (append a pointer here if useful; do not rewrite the mm chronology)

---

## 6. Do not overwrite (frozen artifacts)

- `artifacts/pc2/batch_parallel_autosa_mm_flow_aav_n_gf_20260830_mmflow`
- `artifacts/pc2/batch_parallel_autosa_mm_32x8_flow_aav_n_gf_20260830_mm32x8`
- Locked `autosa_mm` PE recipe / `autosa_mm_32x8`
- Family A/B packed-search first campaigns
- Family C first IO-mesh campaign (`compact_pe_io_search_20260831_io2`)
- mm champion `batch_parallel_autosa_mm_flash_dsp500_pe16_20260904_131622`
- Aug 18 / Sep 2 slide briefs listed in §5

New experiments **write a new campaign stamp**.

---

## 7. Queue / endpoint note from the dying agent

Around 2026-09-07 07:14 PT the inventory agent saw Slurm jobs **2887969**, **2887967_0**, **2887970_0**, **2887971**. Those may already be gone. **Re-check `squeue`.** Do not submit a second flash codegen job if `login5:18092` is already serving one.

---

## 8. Suggested first commands

```bash
cd /scratch/hpc-prf-llmfpga/asa582/projects/c2hls
squeue -u "$USER"
curl -sS -m 3 http://login5:18092/v1/models | head

# mm champion (do not overwrite; read before cloning)
less artifacts/pc2/batch_parallel_autosa_mm_flash_dsp500_pe16_20260904_131622/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_selected.cpp
less c2hls_tmp/batch_parallel_autosa_mm_flash_dsp500_pe16_20260904_131622/autosa_mm/hls_synth__flash_synth/hls_proj/sol1/syn/report/autosa_mm_csynth.rpt

# on-chip pack + tests
less hls_full_optimization_skills_schema_1_1_package/flash_onchip_wide_gemm_skill_entries.json
.venv/bin/python -m pytest tests/test_flash_onchip_gemm.py -q

# seeds
ls related_work/benchmarks/autosa_ready
# if missing:
# python3 scripts/prepare_autosa_ready.py
```

**Do not commit unless asked.**
