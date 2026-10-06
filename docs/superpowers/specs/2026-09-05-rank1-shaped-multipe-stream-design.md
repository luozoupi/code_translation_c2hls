# Design: rank-1-shaped multi-PE then stream (manual + LLM-style)

**Date:** 2026-09-05  
**Status:** Approved in chat (approach B: this session writes kernels; no cluster LLM job)  
**Do not commit** unless the user asks.

## Goal

Give the supervisor a fair shape comparison: the agent kernels use **the same tiling and ping-pong as AutoSA rank-1**, not AutoSA’s 1600-line `kernel0` and not candidate 9. Measure whether matching rank-1’s I×K tiles (without FIFOs, then with FIFOs) closes the remaining gap vs **4228 / 320**.

This is **not** a relaunch of frozen mmflow. It is not candidate 9.

## Locked

- ABI: `autosa_mm(A[I][K], B[J][K], C[I][J])`, `I=J=K=64`, `data_t` float, B is **J×K**.
- TB: official `related_work/benchmarks/autosa_ready/autosa_mm/testbench.cpp` (gold from **zero**; kernel must **not** load C from DRAM).
- Hardware: U280 `xcu280-fsvh2892-2L-e`, 3.33 ns. Cosim off.
- Metrics: **latency min, latency max, DSP**. Do not quote interval on slides.
- Iso-compute: **16 PEs × SIMD 4**.
- Rank-1 reference: `benchmarks_autosa_dse/autosa_mm_rank1/hls_baseline.cpp`, `{array_part[16,64,32]; latency[1,32]; simd[4]}`, **4228 / 320**.
- Frozen: `20260830_mmflow`, recipes, family A/B/C, Aug 18 / Sep 2 slide briefs. Do not overwrite.
- LLM-style files are **hand-written in frozen-dse/stream style**. They are not a DeepSeek sample. Slides must say that.

## What we copy from rank-1

| Item | Rank-1 | This work |
|------|--------|-----------|
| PE array | 16 PEs, 1D, SIMD 4 | Same |
| I tiles | `c0 = 0..3`, 16 rows | `ti = 0..3` |
| K tiles | `c2 = 0..1`, 32 k | `tk = 0..1` |
| J | not coarse-tiled; `local_C[64]` | `Crow[16][64]` |
| On-chip this step | A 16×32, B 64×32, C 16×64 | Same sizes |
| B replay | full B once per I-tile (serialize 4×) | `load_B` of the 64×32 K-slab inside the I-loop |
| K reduction | init K0, accumulate K1, drain on last k-group of K1 | `if (tk == 0) write else +=`; store only `tk == 1` |
| Ping-pong / DATAFLOW | L1/L2 ping-pong | Manual: function DATAFLOW. LLM-style: fat nest + DATAFLOW on the tile loop if HLS accepts it |
| FIFOs | PE chain B, per-PE A/C streams | **Stage 1 none.** Stage 2 B chain PE0→PE15, per-PE A in and C drain |

Not copied in v1: `A_t16` / `B_t16` host-serialize, 512-bit bursts, AutoSA module names, candidate 9 J×K mesh.

## Four kernels

Folder: `artifacts/pc2/manual_rank1_shaped/`  
Header: copy or include `related_work/benchmarks/autosa_ready/autosa_mm/kernel.h`  
TB: copy official testbench.

| File | Author | FIFOs | Style |
|------|--------|-------|-------|
| `autosa_mm_rank1_pe.cpp` | Manual gold | No | `load_A_tile` / `load_B_ktile` / `compute_tile` / `store_C_tile`. Ping-pong buffers declared inside DATAFLOW (same HLS 214-397 workaround as `manual_mmflow_pe_pp`). |
| `autosa_mm_rank1_pe_llm.cpp` | LLM-style | No | One `autosa_mm` like frozen `autosa_mm_dse.cpp`: nested `ti`/`tk`/`k0`/`j`/`p`, partitions, no helper story. Same tiles. |
| `autosa_mm_rank1_stream.cpp` | Manual gold | Yes | Same tiles. `hls::stream` A per PE, B forwarded PE→PE, C drain per PE after K1. |
| `autosa_mm_rank1_stream_llm.cpp` | LLM-style | Yes | Frozen `autosa_mm_stream.cpp` shape, but I×K tiles and B 64×32 replay, not all-of-B once with a single K pass. |

## Compute nest (both no-FIFO kernels)

```
Crow[p][j] starts unset; first K-tile writes, second adds.
for ti in 0..3:                 // i = 16*ti .. 16*ti+15
  for tk in 0..1:               // k = 32*tk .. 32*tk+31
    load A[16][32], load B[64][32]
    for k0 in {0,4,...,28}:
      for j in 0..63:
        unroll p = 0..15:
          partial = A[p][k0:4] · B[j][k0:4]
          Crow[p][j] = (k0==0 && tk==0) ? partial : Crow[p][j] + partial
    if tk == 1: store C rows 16×64
```

`Crow` lives across the two K-tiles of one I-tile. B is **not** the full 64×64 resident across I-tiles.

## Stream nest (both FIFO kernels)

Same `ti`/`tk`. Each PE holds `Crow[64]`. Each II=1 beat: read A_t4, read B_t4, MAC SIMD 4 into `Crow[j]`, write B_t4 to the next PE. Drain `Crow[j]` to C FIFO only on the last k-group of `tk == 1`. Top DATAFLOW: load_A ∥ load_B ∥ 16 PEs ∥ store_C.

## Verification

Reuse the `hls_eval` pattern in `artifacts/pc2/manual_mmflow_pe_pp/run_csim_csynth.py`.

Per kernel: csim must print `Passed!`. Then csynth. Write `csim_csynth_<name>.json` with `latency_cycles`, `latency_cycles_worst`, `dsp`.

Quote vs:

| Baseline | Cycles | DSP |
|----------|-------:|----:|
| AutoSA rank-1 | 4228 | 320 |
| Frozen mmflow stream (old shape) | 4292 | 320 |
| Frozen mmflow multi-PE (old shape) | 13160 | 352 |
| Manual I-tile ping-pong (all of B) | 9640 | 352 |

Work dirs: `c2hls_tmp/manual_rank1_shaped_*`. Do not write under `20260830_mmflow`.

## Out of scope

- Calling AutoSA or emitting `kernel0`
- Candidate 9 geometry
- Real DeepSeek job / new mm-flow campaign
- Wide DRAM packing / host-serialize
- Cosim, P&R, bitstream
- Editing Aug 18 or Sep 2 slide briefs until numbers exist
- Git commit unless asked
- Saying “DSE” on slides (say compute rewrite / hide load-store / rank-1-shaped tiles)

## Success

Four kernels csim-pass. Four csynth numbers. A short table: old LLM shape vs rank-1-shaped vs AutoSA rank-1, same 16×4.
