# Handoff: c2hls vs AutoSA rank-1 on `autosa_mm`

**Date:** 2026-09-06  
**Source:** local Cursor Agent chat on Otus, 2026-08-17 → 2026-09-06  
**Purpose:** continue this work in a **new** chat. Cursor local Agent history does not follow you to another laptop. Paste this file (or `@` it) as the starting context.

**Repo:** `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls`  
**Branch:** `c2hls_enhanced_l_pc2_api_layout` (often dirty; **do not commit unless asked**)  
**Transcript (read-only logs, not a resumable chat):**  
`/pc2/users/h/haqc2/.cursor/projects/scratch-hpc-prf-llmfpga-asa582-projects-c2hls/agent-transcripts/c967e877-78fe-413e-a23f-23445c808cb4/`

---

## 0. Read this first

### Current champion (flash, spend-the-chip)

| Field | Value |
|---|---|
| Latency min–max | **940–940** |
| DSP / BRAM / FF / LUT | **5344** / 90 / 765008 / 346116 |
| Cosim | **PASS, 1071 cycles** |
| Pipeline type | **no** (not DATAFLOW) |
| Equation | `940 ≈ max(load_A 259, load_B 258) + compute 340 + store_C 260` |
| Compute | trip 256, II=1, depth 84, `PE_BLK=16`, 16-wide AXI lanes |
| Campaign | `batch_parallel_autosa_mm_flash_dsp500_pe16_20260904_131622` |

**Code:**  
`/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flash_dsp500_pe16_20260904_131622/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_selected.cpp`

**Csynth report:**  
`/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/c2hls_tmp/batch_parallel_autosa_mm_flash_dsp500_pe16_20260904_131622/autosa_mm/hls_synth__flash_synth/hls_proj/sol1/syn/report/autosa_mm_csynth.rpt`

**JSON (same numbers):**  
`.../autosa_mm_flash_opt_report.json` in that variant folder (`latency_cycles: 940`, `dsp: 5344`).

**Cosim JSON:**  
`/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/manual_pe16_tile_pp/cosim_pe16.json`

**Do not claim 940 “beats AutoSA fairly.”** Rank-1 is **4228 / 320 DSP**. 940 uses **~17× DSP**. Two comparisons must stay separate (see §2).

### Locked iso-compute slide (same 16×4 as AutoSA)

| Design | Latency | DSP |
|---|---:|---:|
| AutoSA rank-1 | **4228** | **320** |
| Aug 18 stream (first locked slide) | **4285** | 320 |
| Frozen mmflow stream | **4292** | 320 |
| No-skills stream | **4216** | 320 |

On slides say **flash / multi-PE (compute rewrite) / stream**. **Never say “DSE”.**

### Do not overwrite

- `artifacts/pc2/batch_parallel_autosa_mm_flow_aav_n_gf_20260830_mmflow`
- `artifacts/pc2/batch_parallel_autosa_mm_32x8_flow_aav_n_gf_20260830_mm32x8`
- Locked `autosa_mm` PE recipe / `autosa_mm_32x8`
- Family A/B packed-search first campaigns
- Family C first IO-mesh campaign (`compact_pe_io_search_20260831_io2`)
- `docs/pc2/2026-08-18-autosa-mm-agent-vs-rank1-slide-brief.md` (4285 vs 4228)

New flash experiments **write a new campaign stamp**. Dedicated launchers **assign** `BATCH_PARALLEL_ARTIFACT_PREFIX`; they must not inherit a stale prefix from the shell (`unset` first; tile-pp launcher already does this).

---

## 1. How to pick this up in a new chat

Start the new Agent with something like:

> Continue c2hls `autosa_mm` gap vs AutoSA rank-1. Read `docs/pc2/2026-09-06-autosa-mm-gap-chat-handoff.md`. Champion is flash PE_BLK=16 at **940 / 5344 DSP**, campaign `..._pe16_20260904_131622`. Frozen `20260830_mmflow` is 4292/320. Compare **latency min/max + DSP only**. Do not overwrite mmflow / 32×8 / family C first campaign. Do not call multi-PE “DSE” on slides.

Then open the champion cpp + csynth report before proposing new runs.

**Pytest:** `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/.venv/bin/python -m pytest`  
**One LLM endpoint:** `http://login5:18092/v1`, model `deepseek-v4-flash`. Do not overlap two flash codegen jobs.

---

## 2. Problem statement (what the supervisor asked)

**Original request (2026-08-17):** pick **one** representative design; keep architecture/code fixed; run aggressive DSE; quantify how much of the gap vs the **reference** is DSE vs missing expertise / missed pragmas / code transforms / the LLM.

**Reference is AutoSA best rank, not ChatHLS.**

Seed is **plain C** `autosa_mm`, not stripped AutoSA `kernel0`.  
Trap: an older `plainseed_stream` run generated AutoSA rank-1 and stripped pragmas. That is already a systolic netlist. **This study does not use that seed.**

Later the supervisor added:

1. Flash looked weak (II=4, low DSP). Later stages sit on flash — why didn’t repair catch II=4?
2. Stream should equal **DATAFLOW + ping-pong**. If that lived in multi-PE, stream might be a naming artifact.
3. Enforcement: make ping-pong + DATAFLOW real, judged on **code and csynth**.
4. Other AutoSA mm-family benches within **2%** of their rank-1 cycles.
5. What AutoSA automates (space-time, tiles, L1/L2 IO, packing) — clone the **plan**, not the compiler.
6. Ablation: zero-shot vs no-skills vs with-skills.
7. Why 320 DSP (iso-compute, not filling U280).
8. Flash itself must stop under-using DSP; fix the **dominant** latency module.

---

## 3. Locked comparison rules

| Rule | Detail |
|---|---|
| Metrics | **Latency min**, **latency max**, **DSP** |
| Never on slides | **Interval**, HLS **estimated clock** |
| Device | Alveo U280 `xcu280-fsvh2892-2L-e` |
| Csynth clock | **3.33 ns** requested |
| Tool | Vitis HLS **2023.2** |
| DSP budget | **9024** on the full device. **Not** limited to 1 SLR (3008). The 940 kernel is 177% of one SLR DSP and 59% of the chip — that is allowed. |
| Gold | `C = A·B` from **zero**. Kernel must not treat DRAM `C` as an input to accumulate. |
| ABI | `extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J])` — **B is J×K**, not K×J |
| Sizes | `I=J=K=64`, `typedef float data_t` in `related_work/benchmarks/autosa_ready/autosa_mm/kernel.h` |
| Cosim | Off for most campaigns. On for PE=16 vs manual ping-pong (Sep 4). |

**Two leaderboards (do not mix):**

1. **Iso-compute 16 PE × SIMD 4 ≈ 320 DSP** vs rank-1 **4228**. Fair architecture-class comparison.
2. **Spend DSP** (PE_BLK=16 flash **940 / 5344**, packed 128 PE **1362 / 5120**). Faster finish time, not AutoSA-efficient.

---

## 4. Directory map

| Role | Path |
|---|---|
| Repo root | `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls` |
| Seed kernel | `related_work/benchmarks/autosa_ready/autosa_mm/plain.cpp` (HLS baseline same text: `hls_baseline.cpp`) |
| Header / TB | `related_work/benchmarks/autosa_ready/autosa_mm/{kernel.h,testbench.cpp}` |
| Campaign artifacts | `artifacts/pc2/<campaign>/` |
| HLS work dirs | `c2hls_tmp/<campaign>/` |
| PC2 launchers | `scripts/pc2/start_autosa_mm_*.sh` |
| Skill JSON | `hls_full_optimization_skills_schema_1_1_package/` |
| PE recipe given to multi-PE | `post_flash_pe_recipe.py` |
| AutoSA repo | `/scratch/hpc-prf-llmfpga/asa582/projects/AutoSA` |
| AutoSA paper | `AutoSA/3431920.3439292.pdf` |
| Rank-1 exhaustive cand 5/9 | `AutoSA/artifacts/dse/campaigns/20260827_032231_mm_exhaustive/` |
| Slide briefs | `docs/pc2/2026-08-18-*.md`, `docs/pc2/2026-09-02-*.md` |

### How to read any campaign

```
artifacts/pc2/<campaign>/
  campaign.json                          # status, flash_min_dsp, flash_pe_blk, job ids
  variants/<pack>/autosa_mm/<model>__flash__autosa__<pack>/
    autosa_mm_selected.cpp               # promoted kernel
    autosa_mm_flash_opt.cpp              # flash output
    autosa_mm_flash_opt_report.json      # latency_cycles, dsp  ← use this
    autosa_mm_dse.cpp / *_dse_result.json
    autosa_mm_stream.cpp / *_stream_result.json

c2hls_tmp/<campaign>/autosa_mm/
  hls_synth__flash_synth/hls_proj/sol1/syn/report/autosa_mm_csynth.rpt
  hls_synth__dse_flash_final_a0_synth/...
  hls_synth__stream_flash_final_a0_synth/...
```

**Ignore** `autosa_mm_multistep_results.json` / `phase_b_report.json` when they show **2445313 / 50 DSP** — that is the **naive baseline**, not flash.

In the csynth report, copy **Latency min/max** from the top-level summary table. Copy **DSP** from Utilization **Total**. Do not quote Interval. Do not quote Estimated clock.

Pack tag in the path:

| Tag | Meaning |
|---|---|
| `autosa_nav_n` | 90-skill II-miss pack |
| `autosa_aav_n` | 90-skill + no-RMW m_axi overlay |
| `autosa_aav_n_gf` | gemm_flatten_v1 90-skill + no-RMW (default for later mm) |
| `autosa_zero_shot` / `autosa_noskills` | ablation variants |

---

## 5. Method: agent pipeline

```
plain.cpp
   │
   ▼
[1] Flash          LLM rewrite + pragmas + 90-skill pack
                   csim + csynth (cosim usually off)
   │
   ▼
[2] Multi-PE       post-flash compute rewrite
                   PE×SIMD recipe (default 16×4) via post_flash_pe_recipe.py
                   historically called “DSE” in code; slides say multi-PE / compute rewrite
   │
   ▼
[3] Stream         DATAFLOW + hls::stream PE/IO to hide load/store
```

**What each stage is for**

| Stage | Job | Typical failure |
|---|---|---|
| Flash | Legal, TB-correct HLS kernel with interfaces | Triple loop, II=4 on k, almost no DSP |
| Multi-PE | On-chip PE array; compute nest II=1 | Compute becomes ~4816 but LCST still serial → ~13k |
| Stream | Overlap DRAM with PEs | Broken I/O, missing flatten, single gmem bundle |

**Flash is not the PE array.** Measuring flash vs AutoSA 4228 is the wrong comparison unless you are specifically studying flash quality.

**Supervisor’s later claim:** stream ≈ DATAFLOW + ping-pong on the multi-PE array. If that is done in multi-PE, the third stage may be naming. Tests in §6.E and §6.K showed “pragma DATAFLOW present” ≠ “one-call latency = max(modules)”.

### Skill files

All under `hls_full_optimization_skills_schema_1_1_package/`:

| Use | File |
|---|---|
| Flash 90-skill | `skills_ii_target_miss_solutions_added(90skills).json` |
| + no-RMW m_axi | `flash_no_RMW_m_axi_skill_entries.json` |
| gemm flatten v1 | `skills_ii_target_miss_solutions_added(90skills)_gemm_flatten_v1.json` |
| Multi-PE | `post_flash_dse_pe_skill_entries.json` |
| Stream | `post_flash_stream_pe_io_skill_entries.json` |
| Overlap / ping-pong PE | `post_flash_overlap_pe_skill_entries.json` |

Flash historically **downranked DATAFLOW**: skill `hls-baseline-load-compute-store-gate` (split load/compute/store first), prompt item 5 (“simple pipelining often beats dataflow”), and avoid-rules in `all_skills_avoids_global`. Sep 4 turned those brakes off. The PE=16 champion still won **without** DATAFLOW because compute (340) was already in the same ballpark as I/O (259).

### Environment knobs (flash-only campaigns)

| Env | Role |
|---|---|
| `C2HLS_FLASH_ONLY=1` | Do not chain multi-PE/stream |
| `C2HLS_FLASH_MIN_DSP` | Reject csynth below this DSP; LLM retry with reason |
| `C2HLS_FLASH_PE_BLK` | 16 / 32 / 64 compute lanes |
| `C2HLS_FLASH_TILE_PP=1` | Prompt for in-GEMM tile ping-pong |
| `C2HLS_FLASH_ROW_UF` | Force row unroll (64 was tried) |
| `C2HLS_FLASH_MAX_TOKENS` | 65536 after Sep 1 length failures |
| `C2HLS_ENFORCEMENT` / `_ROUNDS` | Ping-pong+DATAFLOW judge loop |
| `C2HLS_POST_FLASH_DSE` / `_STREAM` | Chain stages (frozen mmflow uses these) |
| `BATCH_PARALLEL_ARTIFACT_PREFIX` | Campaign name prefix — **assign, do not inherit** |

LLM: `http://login5:18092/v1`. Typical: `--endpoint-url http://login5:18092/v1`.  
`C2HLS_TURNS` often 7 for DSP-gated flash. Synth timeout often 14400 s.

---

## 6. Tests (chronological, with paths and results)

All artifact paths are under  
`/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/`  
unless written as absolute.

Csynth tmp:  
`/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/c2hls_tmp/<campaign>/`

---

### 6.A Devstral flash (2026-08-17) — ABI then weak flash

**Method:** first c2hls flash on plain `autosa_mm`. Devstral-2, `nav_n`, csim+csynth, cosim/lat-opt/RAG off.

| Campaign | Result |
|---|---|
| `batch_parallel_autosa_mm_gap_20260818_mm_gap_flash` | ABI mismatch (`extern "C"` vs TB). Fixed and rerun. |
| `batch_parallel_autosa_mm_gap_20260818_mm_gap_flash_abi` | Flash **1056932 / 3 DSP**. Selected CPP: **no array partition**. Naive baseline **2445313 / 50 DSP**. |

Selected: `.../devstral2__flash__autosa__nav_n/autosa_mm_selected.cpp`

---

### 6.B DeepSeek three-stage gap (2026-08-18) — locked 4285 vs 4228

**Campaign:** `batch_parallel_autosa_mm_gap_ds_v4f_20260818_mm_gap_flash_ds_v4f`  
**Pack:** `autosa_nav_n`  
**Model:** `deepseek-v4-flash`  
**Doc:** `docs/pc2/2026-08-18-autosa-mm-agent-vs-rank1-slide-brief.md`

| Stage | File stem | Latency | DSP | Notes |
|---|---|---:|---:|---|
| Flash | `autosa_mm_selected.cpp` / flash_opt | **149082** | 6 | Triple loop; k-recurrence; **II=4** |
| Multi-PE | `autosa_mm_dse.cpp` | **13160** | 352 | PE=16 SIMD=4; compute ~4816 II=1; LCST serial |
| Stream flatten k×j | `autosa_mm_stream.cpp` | **4285** | 320 | **+57 cycles vs 4228 (1.3%)** |

JSON: `post_flash_dse_summary_20260818_090615.json` (13160), `post_flash_stream_summary_20260818_134448.json` (4285).

Supervisor after slides: flash is a bad base (II=4, low DSP); how many agent rounds; caps?

---

### 6.C Skill overlays + gemm flatten (2026-08-19–24)

| Campaign | Pack | Flash | After |
|---|---|---|---|
| `batch_parallel_autosa_mm_gap_ds_v4f_aav_n_20260819_mm_gap_flash_ds_v4f_aav_n` | `aav_n` (II-miss + no-RMW) | **73897 / 20 DSP** | Flash-only isolate whether II moves |
| `batch_parallel_autosa_mm_gap_ds_v4f_aav_n_gf_20260823_mm_gap_flash_ds_v4f_aav_n_gf` | `aav_n_gf` flatten k,i,j trip 16384 | **24745 / 80 DSP** | Multi-PE **13160 / 352**; stream **4299 / 352** (retry 5737/320) |

Same story: compute rewrite gets you to ~13k; stream closes most of the rank-1 gap at 320 DSP.

---

### 6.D Wave1 other AutoSA mm variants (2026-08-25)

**Goal:** agent latency ≤ rank-1 × **1.02**.  
**Campaign:** `batch_parallel_autosa_wave1_aav_n_gf_20260825_wave1_aav_n_gf`  
**Doc:** `docs/pc2/2026-09-02-wave1-vs-rank1.md`

| Bench | Rank-1 | Agent selected | DSP | 2% gate |
|---|---:|---:|---:|---|
| `autosa_mm` (later frozen as mmflow) | 4228 | 4292 / 4216 noskills | 320 | pass |
| `autosa_mm_hcl` | 4230 | **4294** | 320 | **pass 1.51%** |
| `autosa_mm_hcl_intel` | 4226 | **4292** | 320 | **pass 1.56%** |
| `autosa_mm_int16` | 4219 | **4280** | 64 | **pass 1.45%** |
| `autosa_mm_catapult` | 8286 | **8351** | 96 | **pass 0.78%** |
| `autosa_mm_intel` | 4178 | **4525** | 640 | **fail 8.3%** (ceiling 4261) |
| `autosa_mm_getting_started` | 2194 | **4525** | 640 | **fail ~2×** (rank-1 is two DATAFLOW tiles) |

Retry Slurm **2789605** did not close intel / getting_started. Intel compute-rewrite LLM timed out; stream still ran.  
Not this round: cnn / lu / large_* / HBM. P&R later.

---

### 6.E Enforcement: ping-pong + DATAFLOW (2026-08-26–29)

**Supervisor:** if DATAFLOW + ping-pong were correct on multi-PE, that should match stream.

**Method:** `--enforcement` / `--enforcement_rounds 20`. LLM judge: (1) code intends ping-pong+DATAFLOW; (2) csynth shows it. Overlap judged via interval vs latency — **do not put that interval on slides**.

| Campaign | Status | Result | Lesson |
|---|---|---|---|
| `batch_parallel_autosa_flash_enf_aav_n_gf_20260826_105836` | aborted | mm flash **41147 / 40 DSP** | `C2HLS_ENFORCEMENT` often **never reached the synth worker**. Kernel had *less* parallelism, no real ping-pong. |
| `batch_parallel_autosa_mm_enf_aav_n_gf_20260829_123043` | complete | **12893–12893 / 320 DSP**, interval 4553 | Judge: ping-pong DATAFLOW **present**. One-call finish is still load+compute+store. 85% interval/latency gate too rigid. |

Csynth (enforcement step):  
`c2hls_tmp/batch_parallel_autosa_mm_enf_aav_n_gf_20260829_123043/autosa_mm/hls_synth__step_synth/hls_proj/sol1/syn/report/autosa_mm_csynth.rpt`

**DATAFLOW does not make latency = max(modules) by itself.** Rank-1 has interval ≈ latency because of a real systolic IO schedule (K-tiles, B ping/pong, C drain).

Launcher: `scripts/pc2/start_autosa_mm_enforcement.sh`

Manual coarse ping-pong on mmflow multi-PE (not AutoSA IO):  
`artifacts/pc2/manual_mmflow_pe_pp/`

| File | Csynth | DSP | csim |
|---|---:|---:|---|
| `autosa_mm_pe_pp.cpp` | 9640 | 352 | pass |
| `autosa_mm_pe_pp_plus.cpp` | 4688 | 352 | pass |

Aug 23 overlap seed (not mmflow): **8678 / 320**, architecture gate fail. Coarse tile overlap ≠ systolic hide-load/store.

---

### 6.F Frozen three-stage mmflow (2026-08-30) — do not overwrite

**Campaign:** `batch_parallel_autosa_mm_flow_aav_n_gf_20260830_mmflow`  
**Launcher family:** `scripts/pc2/start_autosa_mm_flow.sh`  
**Pack:** `aav_n_gf`

| Stage | Latency | DSP | tmp synth dir |
|---|---:|---:|---|
| Flash | 139484 | 10 | `hls_synth__flash_synth` |
| Multi-PE | **13160** | 352 | `hls_synth__dse_flash_final_a0_synth` |
| Stream | **4292** | 320 | `hls_synth__stream_flash_final_a0_synth` |

**+1.5% vs 4228.** This is the frozen with-skills column for later ablations.

**32×8 overlay** (more silicon, not the locked slide):  
`batch_parallel_autosa_mm_32x8_flow_aav_n_gf_20260830_mm32x8`

| Stage | Latency | DSP |
|---|---:|---:|
| Multi-PE | 9782 | 1344 |
| Stream | **4583** | 1280 |

**Worse than 16×4 stream 4292.** Inner loop `pe_kj=512` × two tiles (`I/PE=2`). Extra workers do not hide extra I/O. Do not relaunch. Memo: `docs/pc2/2026-09-02-why-320-dsp.md`.

---

### 6.G Compact PE / packed mesh / family C (2026-08-30–31)

**Question:** what is AutoSA automating, and can we search that space without calling `./autosa`?

Missing vs rank-1 / cand 9, in supervisor language: **latency hiding, space-time PEs, real I/O double buffering, packing** — not “streamify.” Rank-1/cand 9 have L3→L2→L1, K-tiles, B ping/pong, C systolic drain. Packed-mesh agent kernels often **replay full B** `LAT_I` times in one BRAM.

#### AutoSA exhaustive (cross-validated; latency min/max only)

`/scratch/hpc-prf-llmfpga/asa582/projects/AutoSA/artifacts/dse/campaigns/20260827_032231_mm_exhaustive/default_cap/validation/mm/`

| Candidate | `--sa-sizes` | Latency min–max | Notes |
|---|---|---|---|
| **#9** | `space_time[4]; array_part[64,32,32]; latency[8,4]; simd[8]` | **1846–2161** | ~1296 DSP; K=32 B tile + ping/pong |
| **#5** | `space_time[4]; array_part[64,64,32]; latency[4,8]; simd[8]` | **2033–2656** | |

Both used `--local-reduce --reduce-op=+ --simd-touch-space --host-serialize --hls`.

#### Family B packed mesh

**Campaign:** `compact_pe_pack_search_20260831_pack`  
**Launcher:** `scripts/pc2/start_autosa_mm_pack_search.sh`  
**Selected:** `pack32x4_simd8` — `selected_report.json`

| Point | PEs | SIMD | Latency | DSP |
|---|---:|---:|---:|---:|
| **pack32x4_simd8** | 128 | 8 | **1362** | **5120** |
| Several 8/16 PE packs | | | 4434 | | |

1362 beats cand 9 on **latency** by spending **4× PEs** so `LAT_I` drops. That is DSP as a substitute for AutoSA’s IO stack. 32-PE analog stays ~4434 because of B replay.

#### Family C IO-mesh (io4 + io5)

**Campaign:** `compact_pe_io_search_20260831_io2`  
**Launcher:** `scripts/pc2/start_autosa_mm_io_search.sh`  
**Spec:** `docs/superpowers/specs/2026-08-31-compact-io-mesh-search-design.md`

| Point | PEs | Latency | DSP |
|---|---:|---:|---:|
| `io4_8x4_s8_k32_j64` (iso-DSP 32-PE) | 32 | **2245** | **1280** — **misses cand 9 ≤1846** |
| `io4_16x8_s8_k32_j64` (selected) | 128 | **1571** | **5120** — faster only by oversubscribe |

Do **not** mix family C into the 4228/320 slide. Do not overwrite this first campaign.

PE-count overlay: stock AutoSA search wanted ~190–210 PEs; a prune for small `autosa_tests` kernels had effectively capped ~32. Bounds were discussed toward `[8,512]`. SIMD prune during search is what actually caps `n_pe`.

---

### 6.H Zero-shot / no-skills / with-skills (2026-09-01–02)

**Doc (table source):** `artifacts/pc2/autosa_mm_ablation_20260901/comparison.md`  
**Slide brief:** `docs/pc2/2026-09-02-supervisor-remaining-slide-brief.md`

Sep 1 died on **client `max_tokens=16384`** (`finish_reason=length`). Sep 2 used **65536** + 8 continuations. vLLM on login5 already accepted 65536.

| Arm | Campaign | Method | Flash | Multi-PE | Stream |
|---|---|---|---|---|---|
| **Zero-shot** | `batch_parallel_autosa_mm_flow_zero_shot_20260902_mmzs` | HLS-engineer prompt; **PE recipe on, PE skills JSON off** | **40454–42758 / 320** | — | — |
| **No-skills** | `batch_parallel_autosa_mm_flow_noskills_20260902_mmns` | Full flow, `skill_count=0` | **1056894 / 11** | **25080 / 384** | **4216 / 320** |
| **With-skills** | frozen `20260830_mmflow` | 90-skill + flatten | **139484 / 10** | **13160 / 352** | **4292 / 320** |

**Corrections the user locked**

- Zero-shot **had the PE recipe**, not “no PE recipe.”
- Generic flash skills can **lose to zero-shot** (139k vs 40k) by emitting a legal kernel with **no PE array**.
- No-skills **stream 4216** slightly **beats** with-skills 4292 and rank-1 4228 at 320 DSP. The PE recipe + stream skeleton matter more than the 90-skill dump.

**Four-column supervisor figure**

| Col | Point | Latency | DSP |
|---|---|---|---|
| 1 | Zero-shot | 40454–42758 | 320 |
| 2 | Generic flash skills (mmflow flash) | 139484 | 10 |
| 3 | Enforcement | 12893 | 320 |
| 4a/4b | Systolic stream | 4292 / **4216** | 320 |

---

### 6.I Multi-PE directly on baseline (2026-09-03)

**Question:** is flash required for the ~13k multi-PE number?

**Campaign:** `dse_from_baseline_20260903_083901`  
**Launcher:** `scripts/pc2/start_autosa_mm_dse_from_baseline.sh`  
**Seed:** `related_work/benchmarks/autosa_ready/autosa_mm/hls_baseline.cpp` (same text as `plain.cpp`)  
**Does not touch mmflow.**

| Arm | Latency | DSP |
|---|---:|---:|
| Multi-PE on baseline | **13104** | **320** |
| Flash then multi-PE (mmflow) | 13160 | 352 |

Flash is **not** what made multi-PE ~13k. The PE recipe is. JSON: `.../autosa_mm_dse_result.json`.

---

### 6.J Flash DSP gates and I/O / II prompts (2026-09-03–04) — path to 940

**Method:** reject csynth below a DSP floor and tell the LLM why; then prompt the **dominant** latency module (load/store, then compute II). Flash-only (`C2HLS_FLASH_ONLY=1`). Launchers under `scripts/pc2/start_autosa_mm_flash_*.sh`.

| Campaign | Method | Latency | DSP | Notes |
|---|---|---:|---:|---|
| `batch_parallel_autosa_mm_flash_dsp300_20260903_113128` | min DSP 300 | **15708** | 384 | More DSP ≠ much less latency; I/O still dominates |
| `batch_parallel_autosa_mm_flash_dsp1000_20260903_121153` | min DSP 1000 | **13549** | 768 | Report: `row_group` + `load_B_rows_load_B_cols` sum to kernel latency |
| `batch_parallel_autosa_mm_flash_rowuf64_20260904_012102` | force row UF 64 | **12456** | 320 | Same 12456 as 32×8 flash; UF 16 vs 64 not the real bottleneck |
| `batch_parallel_autosa_mm_flash_dataflow_20260904_083918` | stop DATAFLOW downrank | **82025** | 20 | DATAFLOW without DSP/I/O work made it worse |
| `batch_parallel_autosa_mm_flash_dsp500_20260904_085931` | min DSP 500 | **9423** | 1344 | First DSP-500 flash; still LCST |
| `batch_parallel_autosa_mm_flash_dsp500_wideio_20260904_100437` | 512-bit coalesced A/B/C | **2760** | 636 | Load/store ~258; compute **2099 because II=4** |
| `batch_parallel_autosa_mm_flash_dsp500_computeii_20260904_120524` | affine i/j, write-once C, no RMW | **1261** | 2672 | Interval 602, report type dataflow; still `max(loads)+compute+store` in spirit |
| **`batch_parallel_autosa_mm_flash_dsp500_pe16_20260904_131622`** | **PE_BLK=16** | **940** | **5344** | **Champion** |
| `batch_parallel_autosa_mm_flash_dsp500_pe32_20260904_133157` | PE_BLK=32 | **9991** | 782 | Model did not actually emit a 32-wide II=1 array |
| `batch_parallel_autosa_mm_flash_dsp500_pe64_20260904_134954` | PE_BLK=64 | **1733** | 1272 | Better than 32 run, worse than PE=16 |

Aborted sibling: `..._computeii_20260904_131523` (started PE=16 then aborted; use **131622**).

**Why wide I/O dropped load/store from ~4k to ~258:** 64×64 floats, 16 lanes = 512-bit AXI / 32-bit float (`LANES = 16`). Burst 64, outstanding 16. Bundles `gmem0/1/2`.

**Why compute was II=4 at 2760:** linearized `g` with `i = g>>3`, `jbase = (g&7)<<3`, RMW `old_c = C_loc[...]`. HLS cannot prove write-once → II=4, DSP only 636 (shared multipliers). Fix: nested `row`, `j0 += PE_BLK`, assign `C_local[row][j0+p] = dot` (no load of C from DRAM). Then II=1: compute ≈ (256−1)+84 = 340 at PE=16.

**PE=16 csynth module table (quote this):**

| Instance | Cycles |
|---|---:|
| `Pipeline_load_A_rows_load_A_cols` | 259 |
| `Pipeline_load_B_rows_load_B_cols` | 258 |
| `Pipeline_compute_blocks` | 340 |
| `Pipeline_store_C_rows_store_C_cols` | 260 |
| **Kernel** | **940** (type **no**) |

Load A and B overlap; then compute; then store. Clock target 3.33 ns, estimated 2.431 ns (ignore estimate). DSP 5344 = 59% device / 177% one SLR.

Cosim of this kernel: **PASS, 1071** (`artifacts/pc2/manual_pe16_tile_pp/cosim_pe16.json`, work dir `c2hls_tmp/cosim_pe16_nopong`).

Launcher for PE_BLK:  
`C2HLS_FLASH_PE_BLK=16 ./scripts/pc2/start_autosa_mm_flash_pe_blk.sh --endpoint-url http://login5:18092/v1`  
Prefix `mmpe16` / `batch_parallel_autosa_mm_flash_dsp500_pe16`.

---

### 6.K Tile ping-pong on PE=16 (2026-09-04–05) — did not beat 940

**Intent:** hide I/O so one-call latency → `max(load, compute, store) ≈ 340` instead of `max(loadA,loadB)+compute+store ≈ 940`.

#### LLM flash with `C2HLS_FLASH_TILE_PP=1`

**Launcher:** `scripts/pc2/start_autosa_mm_flash_tile_pp.sh`  
Unsets stale `BATCH_PARALLEL_ARTIFACT_PREFIX`. Prefix `mmpe16pp`. PE_BLK=16, min DSP 500, flash-only.  
**Test:** `tests/test_flash_tile_pp.py`

| Campaign | Status | Latency | DSP |
|---|---|---:|---:|
| `..._pe16_tilepp_20260905_001958` | running/aborted sibling | — | — |
| **`..._pe16_tilepp_20260905_002017`** | complete | **3817** | **5120** |

Interval 3674 (do not compare with this). Model streamed 4 j-tiles, **reloaded full A each tile**, compute not overlapped. Worse than 940.

Selected:  
`artifacts/pc2/batch_parallel_autosa_mm_flash_dsp500_pe16_tilepp_20260905_002017/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_selected.cpp`

#### Manual 2-tile ping-pong

**Dir:** `artifacts/pc2/manual_pe16_tile_pp/`  
**Kernel:** `autosa_mm_tile_pp.cpp`  
**TB:** `testbench_cosim.cpp` (returns 1 on mismatch)  
**Csynth tmp:** `c2hls_tmp/manual_pe16_tile_pp/`

| Version | Csynth min–max | DSP | Cosim |
|---|---|---|---|
| Broken: `B_pp[2]` / `C_pp[2]` **outside** DATAFLOW loop | **321–1509** | 5344 | **FAIL, 4096 errors** (every C). HLS **214-397** localized arrays and dropped ping-pong writes. RTL 1584 |
| **Fixed:** declare `B_buf` / `C_buf` **inside** the DATAFLOW tile loop | **1074–1074** | **5344** | **PASS, 1269**. Host g++ pass. Zero 214-397 |

`csynth_summary.json`: latency 1074, dsp 5344, bram 90, ff 1019388, lut 523071.  
`cosim_tilepp.json`: pass, 1269 cycles, work dir `c2hls_tmp/cosim_pe16_tilepp`.

**Why 1074 > 940 with the same 16 PEs / 5344 DSP**

Same MAC tree, same II=1 depth 84. Two j-tiles of 32 columns:

- Load A still **serial** (~299), not in the DATAFLOW tile loop.
- Per tile: load B ~172, compute 212 (`128+84`), store ~169.
- DATAFLOW II ~213; first-tile fill ~555; two tiles `(2−1)×213 + 555 ≈ 768` → parent ~771.
- Splitting compute **pays pipeline depth 84 twice** (424 vs 340).
- Hiding B/store cannot beat one 256-trip compute of 340.

**Legal HLS ping-pong:** ping-pong arrays must be declared **inside** the DATAFLOW-region loop. Outside + enable flags → 214-397 and silent wrong RTL.

No further ping-pong unless asked. Champion remains **940**.

---

## 7. One-page number sheet

| Design | Latency min–max | DSP | Where |
|---|---|---|---|
| Naive triple loop | 2445313 | 50 | every `hls_synth__ref_baseline_synth` |
| AutoSA rank-1 | **4228–4228** | **320** | AutoSA `20260709_full21_dse`, resynth 3.33 ns |
| Aug 18 stream (locked slide) | **4285** | 320 | `..._mm_gap_flash_ds_v4f` |
| Frozen mmflow stream | 4292 | 320 | `20260830_mmflow` |
| No-skills stream | **4216** | 320 | `20260902_mmns` |
| Zero-shot | 40454–42758 | 320 | `20260902_mmzs` |
| Enforcement DATAFLOW | 12893 | 320 | `20260829_123043` |
| Multi-PE (flash or baseline) | 13160 / **13104** | 352 / 320 | mmflow / `dse_from_baseline_20260903_083901` |
| AutoSA exhaustive #9 | **1846–2161** | ~1296 | `20260827_032231_mm_exhaustive` cand 9 |
| Packed 128 PE | **1362** | 5120 | `compact_pe_pack_search_20260831_pack` |
| Family C iso-32PE | 2245 | 1280 | `io4_8x4_s8_k32_j64` |
| Family C selected | 1571 | 5120 | `io4_16x8_s8_k32_j64` |
| **Flash PE=16** | **940–940** | **5344** | **`..._pe16_20260904_131622`** |
| Manual 2-tile pp fixed | 1074 | 5344 | `manual_pe16_tile_pp` |
| LLM tile-pp | 3817 | 5120 | `..._tilepp_20260905_002017` |

---

## 8. Launchers (copy-paste)

From repo root. Always pass the LLM endpoint. Do not overlap two flash codegen jobs.

```bash
# Frozen-style three-stage flow (do NOT point this at mmflow stamps)
./scripts/pc2/start_autosa_mm_flow.sh --endpoint-url http://login5:18092/v1

# Flash-only DSP floor
C2HLS_FLASH_MIN_DSP=500 ./scripts/pc2/start_autosa_mm_flash_dsp_floor.sh --endpoint-url http://login5:18092/v1

# Flash-only PE_BLK
C2HLS_FLASH_PE_BLK=16 ./scripts/pc2/start_autosa_mm_flash_pe_blk.sh --endpoint-url http://login5:18092/v1

# Flash-only tile ping-pong (unsets stale prefix)
./scripts/pc2/start_autosa_mm_flash_tile_pp.sh --endpoint-url http://login5:18092/v1

# Multi-PE on baseline, skip flash
./scripts/pc2/start_autosa_mm_dse_from_baseline.sh --submit --endpoint-url http://login5:18092/v1

# Dry-run first
./scripts/pc2/start_autosa_mm_flash_pe_blk.sh --dry-run
```

`--dry-run` is supported on several of these. Check `squeue` after submit. Campaign `campaign.json` → `campaign_status` (`complete` / `aborted` / `running`).

---

## 9. Related docs (already written)

| File | Use |
|---|---|
| `docs/pc2/2026-08-18-autosa-mm-agent-vs-rank1-slide-brief.md` | Locked 4285 vs 4228 talk track |
| `docs/pc2/2026-09-02-supervisor-remaining-slide-brief.md` | Ablation / 320 DSP / wave1 leftovers |
| `docs/pc2/2026-09-02-why-320-dsp.md` | Why not fill the chip on the iso-compute slide |
| `docs/pc2/2026-09-02-wave1-vs-rank1.md` | Other benches 2% gate |
| `docs/pc2/2026-08-30-autosa-agentic-flow-one-pager.md` | AutoSA-like agent flow |
| `artifacts/pc2/autosa_mm_ablation_20260901/comparison.md` | Zero-shot / no-skills table |
| `docs/superpowers/specs/2026-08-31-compact-io-mesh-search-design.md` | Family C |
| `docs/superpowers/specs/2026-09-01-autosa-mm-zero-shot-noskills-design.md` | Ablation spec |
| `docs/superpowers/plans/2026-09-05-rank1-shaped-multipe-stream.md` | **Planned** rank-1-shaped 16×4 tiles (A 16×32, B 64×32, C 16×64), not the 940 champion. Target dir `artifacts/pc2/manual_rank1_shaped/`. Do not confuse with PE=16 flash. |

---

## 10. Uncommitted / dirty (as of this chat)

Unless the user committed later:

- Tile ping-pong prompt / `C2HLS_FLASH_TILE_PP` knob / skill entries / `tests/test_flash_tile_pp.py`
- Manual kernel tree `artifacts/pc2/manual_pe16_tile_pp/`
- Possible prompt edits in `prompt_c2hls.py`, `autosa_flow_gates.py`, `c2hls.py` extra_blocks
- LCST skill edits in the three 90-skill packs (DATAFLOW downrank / wide I/O / affine C)

**Do not commit unless asked.**

---

## 11. What is still open

1. **Iso-compute story (320 DSP)** is essentially closed: 4285 / 4292 / 4216 vs 4228. Wave1 intel and getting_started still miss 2%.
2. **Efficiency vs cand 9 (1846 / ~1296 DSP)** is **not** closed. Need K-tile + L1/L2 IO + B ping-pong, not more PEs. Family C 2245/1280 missed; 1571/5120 cheated with DSP.
3. **940** is the flash latency champion on this kernel. It is a wide PE + wide AXI LCST design, not AutoSA.
4. **Ping-pong of j-tiles** on this PE=16 kernel made latency **worse** (1074 / 3817). Next I/O lever, if any, is overlapping **load A** (or not splitting the 256-trip compute). Do not retry the broken “arrays outside DATAFLOW” pattern.
5. Planned rank-1-shaped manual 16×4 (`docs/superpowers/plans/2026-09-05-rank1-shaped-multipe-stream.md`) is a **different** experiment: match AutoSA’s **tile sizes**, not the 940 array.
6. P&R / bitstream / extra HBM: later. This whole chat is **csynth** (+ two cosims).

---

## 12. Suggested first commands in a new session

```bash
# Champion report
less /scratch/hpc-prf-llmfpga/asa582/projects/c2hls/c2hls_tmp/batch_parallel_autosa_mm_flash_dsp500_pe16_20260904_131622/autosa_mm/hls_synth__flash_synth/hls_proj/sol1/syn/report/autosa_mm_csynth.rpt

# Champion kernel
less /scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flash_dsp500_pe16_20260904_131622/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_selected.cpp

# Frozen iso-compute stream number
python3 -c "import json; p='/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_mm_flow_aav_n_gf_20260830_mmflow/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_stream_result.json'; print(json.load(open(p)))"
```
