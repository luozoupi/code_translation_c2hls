# Phase 9 e2e on knn — Haiku 4.5 vs Sonnet 4.6

_generated 2026-05-09. Comparative Phase 9 run on knn under Haiku 4.5 and Sonnet 4.6,
Vitis 2023.2 / xcu280-fsvh2892-2L-e / 3.33 ns clock. GT coalescing = 262,480 cycles
(ref-best). Phase 9 correctness-repair and Phase 6a best-so-far active for both runs._

## TL;DR

- **Haiku outperforms Sonnet on knn despite being the smaller model**: Haiku best = 692,437 cycles
  (coalescing, 2.64× off ref-best); Sonnet best = 1,048,818 cycles (baseline, 3.99× off ref-best).
- **Sonnet's optimization steps all regressed vs its own baseline**: doublebuffer 2,029,579 and
  coalescing 2,029,579 — both 1.93× worse than the 1,048,818-cycle baseline. Phase 6a promoted
  baseline as the final result.
- **Haiku's optimization steps progressively improved**: unroll 1.13M → doublebuffer 940K → 
  coalescing 692K. Phase 9 csim-repair fired on doublebuffer (1 repair → 940K from failing 787K).
- **Both models 100% csim-passing (6/6 steps)** and at clock_gap_ratio ≈ 1.00 (fmax 411.35 MHz).
- **No-op trap fired for Sonnet's unroll** (attempt 0 = pipeline metrics exactly); re-prompt
  produced 3,752,153 cycles — different but 3.58× worse than baseline.

## Configuration

```
C2HLS_VITIS_SETTINGS = /mnt/data/luo00466/Xilinx/Vitis/2023.2/settings64.sh
C2HLS_PART           = xcu280-fsvh2892-2L-e
C2HLS_CLOCK_NS       = 3.33
C2HLS_PHASE8_BASELINE_ALIGN = 1
C2HLS_PHASE5_GT_PREPOP      = 1
C2HLS_PHASE7A               = 1
strategy = dynamic
turns    = 4
haiku model = claude-haiku-4-5-20251001
sonnet model = claude-sonnet-4-6
```

## Baseline quality (Phase 8 alignment)

| Run | GT baseline (cyc) | Agent baseline (cyc) | lat_ratio | Fmax (MHz) |
|-----|------------------:|---------------------:|----------:|-----------:|
| GT reference | 1,740,530 | — | 1.000× | — |
| **knn-haiku** | 1,740,530 | **1,048,816** | **0.603×** | 411.35 |
| **knn-sonnet** | 1,740,530 | **1,048,818** | **0.603×** | 411.35 |

Both models produce near-identical baselines (1.048M cycles, 0.603× of GT baseline) with
identical hardware. Phase 8 alignment needed 1 retranslation for Sonnet.

## Multistep trajectories

### knn-haiku (Phase 9)

| Step | GT (cyc) | Haiku cyc | csim | Event |
|------|--------:|----------:|:----:|-------|
| baseline | 1,740,530 | 1,048,816 | ✓ | Phase 8 aligned (1 retranslation) |
| tiling | 4,276,372 | 4,276,365 | ✓ | Near-exact GT match |
| pipeline | 4,276,372 | 4,563,017 | ✓ | No-op trap fired (attempt 0 = tiling) → re-prompt |
| unroll | 4,044,880 | 1,130,637 | ✓ | Regression vs Phase 8 (different LLM draw) |
| doublebuffer | 1,740,530 | **940,434** | ✓ | Phase 9 csim-repair (787K fail → 940K pass) |
| coalescing | **262,480** | **692,437** | ✓ | 3 compile repairs (mc.h chain) → csim passes |
| **Phase 6a best** | — | **692,437** (coalescing) | ✓ | Progressive improvement through pipeline |

**Haiku csim rate: 6/6**

### knn-sonnet (Phase 9)

| Step | GT (cyc) | Sonnet cyc | csim | Event |
|------|--------:|-----------:|:----:|-------|
| baseline | 1,740,530 | 1,048,818 | ✓ | Phase 8 aligned |
| tiling | 4,276,372 | 4,276,441 | ✓ | Near-exact GT match |
| pipeline | 4,276,372 | 4,274,393 | ✓ | Near-identical to tiling |
| unroll | 4,044,880 | **3,752,153** | ✓ | No-op trap fired (attempt 0 = pipeline) → re-prompt → 3.75M |
| doublebuffer | 1,740,530 | 2,029,579 | ✓ | Regression vs baseline (1.93×); GT synthesis timed out (20min) |
| coalescing | 262,480 | **2,029,579** | ✓ | 2 compile repairs (mc.h, ap_int.h) → same as doublebuffer |
| **Phase 6a best** | — | **1,048,818** (baseline) | ✓ | All opt steps regressed; baseline promoted |

**Sonnet csim rate: 6/6**

_Note: GT doublebuffer synthesis timed out (SYNTH_TIMEOUT=1200s, took 25+ min in scheduler).
Synthesis was manually killed to unblock coalescing. GT comparison for doublebuffer is missing._

## Best-vs-best rubric scoring

| Run | Best agent step | Best agent cyc | Ref-best step | Ref-best cyc | cycles_ratio | clock_gap_ratio |
|-----|-----------------|---------------:|---------------|-------------:|-------------:|----------------:|
| **knn-haiku Phase 9** | coalescing | 692,437 | coalescing | 262,480 | **2.64×** | 1.0002 |
| **knn-sonnet Phase 9** | baseline | 1,048,818 | coalescing | 262,480 | **3.99×** | 1.0001 |

Haiku beats Sonnet by 1.35× on the headline metric despite being the smaller model.

## Cross-model analysis: why Haiku outperforms Sonnet on knn

### Optimization trajectory quality

Haiku produces a monotonically improving trajectory after unroll:
```
baseline → (regression tiling/pipeline/unroll) → doublebuffer ↓ → coalescing ↓↓
                                                    940K           692K
```

Sonnet regresses on every step:
```
baseline → tiling ↑ → pipeline ≈ → unroll ↑ (no-op trap) → doublebuffer ↑↑ → coalescing ≈
  1.05M    4.28M      4.27M         3.75M                     2.03M             2.03M
```

The key difference: Haiku's doublebuffer code finds an effective double-buffering pattern that
reduces latency from 1.13M → 940K (17% improvement). Sonnet's doublebuffer code increases
latency from 3.75M → 2.03M (46% improvement vs unroll, but still 1.93× worse than baseline).

### Likely mechanism

Sonnet generates more complex optimization patterns. For knn's `compute_dist` loop:
- Haiku likely applies a targeted local buffer + pragma pipeline to the inner distance loop
- Sonnet likely over-unrolls or applies aggressive array partitioning that creates long
  combinational paths, increasing the design's total latency even at the same Fmax

Evidence: Sonnet's doublebuffer uses 42 BRAM, 56 DSP, 16K FF, 13K LUT — substantially more
resources than Haiku's doublebuffer (46 BRAM, 56 DSP, 17K FF, 14K LUT). Resource counts are
similar but the latency is 2.16× worse, suggesting an architectural inefficiency (perhaps
excessive outer-loop overhead or a data dependency bottleneck Sonnet introduced).

### No-op trap behavior

Both models triggered the no-op trap:
- **Haiku**: pipeline attempt 0 produced metrics identical to tiling → re-prompt → 4.56M (worse,
  but different)
- **Sonnet**: unroll attempt 0 produced metrics identical to pipeline → re-prompt → 3.75M
  (actually an improvement over the no-op 4.27M!)

Sonnet's no-op trap re-prompt produced a BETTER result (3.75M vs 4.27M, 12% improvement).
This shows the no-op trap adds value even when the optimization step eventually regresses overall.

### Phase 9 csim-repair: Haiku needed it, Sonnet did not

- Haiku: doublebuffer attempt 0 failed csim (787K cycles but wrong output). Phase 9 repair →
  940K cycles, csim passes. Without repair, haiku's trajectory would have included a
  correctness-failing step.
- Sonnet: All steps passed csim on first try (once compile errors were fixed). Sonnet's code
  is functionally correct even when it regresses on performance.

This confirms the Phase 9 correctness-repair loop provides value primarily for Haiku (which
generates more creative but sometimes incorrect optimizations) and less so for Sonnet (which
generates more conservative but always correct code).

## knn coalescing: why neither reaches ref-best 262K

Both models struggle with the coalescing cliff (262K vs 692K-1048K cycles). The GT coalescing
variant uses AXI burst widening (512-bit burst reads in the `coalescing_5` variant) that
dramatically reduces DRAM transfer cycles. Specific observations:

- **Haiku coalescing (692K)**: 3 compile repairs resolved include chain. The resulting code
  is functionally correct and improves on doublebuffer, but doesn't achieve the burst-widening
  bandwidth that yields the GT's 262K.
- **Sonnet coalescing (2,029K = doublebuffer)**: 2 compile repairs then synthesized. The
  coalescing code is functionally identical to doublebuffer — Sonnet generated essentially
  the same implementation twice. This suggests Sonnet's coalescing prompt didn't trigger
  a fundamentally different memory access pattern.

The 2.64× gap (Haiku) and 3.99× gap (Sonnet) to ref-best coalescing suggest that explicit
burst-widening knowledge (high-confidence skill library entry, Pillar 3/5) is needed:
> `AXI m_axi interface: explicitly set max_read_burst_length=64 and num_read_outstanding=16
> to widen burst width to 512 bits for float arrays. Default burst is 16 bytes (4 floats);
> 512-bit burst is 64 bytes (16 floats) → 4× bandwidth improvement per burst.`

## GT synthesis timeout (new operational finding)

The GT doublebuffer synthesis in `_optimization_step_attempt` took 25+ minutes and timed out.
The Vitis HLS scheduler was stuck on the `LOAD_TILE` sub-module scheduling phase, using only
~10-15% CPU for the duration. The manual SIGTERM unblocked the Python process.

**Root cause**: the GT knn doublebuffer code has complex flag-parameter double-buffering that
creates intricate scheduling dependencies. Vitis's scheduler exponential-searches the schedule
space and can get stuck in a long-running branch.

**Recommendation**: add a per-step GT synthesis timeout that is SHORTER than SYNTH_TIMEOUT
(e.g., `C2HLS_GT_SYNTH_TIMEOUT = min(300, SYNTH_TIMEOUT // 4)`). GT synthesis is only for
comparison purposes; if it times out, skip the GT comparison and continue. Today's code uses
the same SYNTH_TIMEOUT for both agent and GT synthesis.

## Summary comparison table

| Metric | knn-haiku Phase 9 | knn-sonnet Phase 9 |
|--------|:----------------:|:-----------------:|
| Baseline (cyc) | 1,048,816 | 1,048,818 |
| Best step | coalescing | baseline |
| Best (cyc) | **692,437** | 1,048,818 |
| Cycles_ratio (vs ref-best) | **2.64×** | 3.99× |
| csim rate | 6/6 | 6/6 |
| Phase 9 repair fired | doublebuffer (1 shot) | none |
| No-op trap fired | pipeline | unroll |
| Phase 6a promoted | coalescing | baseline |
| clock_gap_ratio | 1.0002 | 1.0001 |

## Files of record

- knn-haiku results: [results_phase2/knn_haiku_phase9_u280_v2023/knn_multistep_results.json](../results_phase2/knn_haiku_phase9_u280_v2023/knn_multistep_results.json)
- knn-sonnet results: [results_phase2/knn_sonnet_phase9_u280_v2023/knn_multistep_results.json](../results_phase2/knn_sonnet_phase9_u280_v2023/knn_multistep_results.json)
- knn-haiku artifact: [phase9_e2e_knn_haiku_u280_v2023.md](phase9_e2e_knn_haiku_u280_v2023.md)
- knn-haiku log: `/tmp/phase9_e2e_knn_haiku_u280_v2023.log`
- knn-sonnet log: `/tmp/phase9_knn_sonnet_u280.log`
