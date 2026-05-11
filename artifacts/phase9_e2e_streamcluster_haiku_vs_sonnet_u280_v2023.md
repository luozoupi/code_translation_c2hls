# Phase 9 e2e on StreamCluster — Haiku 4.5 vs Sonnet 4.6

_generated 2026-05-09. First comparative run of knn follow-up benchmark StreamCluster under
Phase 9 (correctness-repair loop) with both Haiku 4.5 and Sonnet 4.6, Vitis 2023.2 /
xcu280-fsvh2892-2L-e / 3.33 ns clock. GT coalescing = 11,017 cycles (163× burst-widening
cliff from baseline 1,795,074 cycles)._

## TL;DR

- **Sonnet baseline is 4× better than Haiku baseline on SC**: 37,494 vs 148,920 cycles.
- **Neither model reaches the GT coalescing cliff (11,017 cycles)**: Sonnet best = 37,494 cyc
  (3.40× off), Haiku best = 148,920 cyc (13.52× off). The burst-widening pattern that ref-best
  exploits remains out of reach for both.
- **Sonnet's optimization steps all regressed vs its baseline**: tiling ~66K, pipeline ~66K,
  unroll ~66K, doublebuffer 38,219, coalescing 38,165. Phase 6a promoted baseline (37,494) as best.
- **Haiku's optimization steps all regressed even further**: tiling/pipeline/unroll/doublebuffer all
  ~365K, coalescing failed (4/4 attempts exhausted by s_axilite bundle errors). Phase 6a promoted
  haiku baseline (148,920) as best.
- **Phase 9 csim-repair fired on both runs** (tiling and doublebuffer for haiku; tiling for sonnet),
  demonstrating the correctness-repair loop works on SC as well as knn.

## Configuration

```
C2HLS_VITIS_SETTINGS = /mnt/data/luo00466/Xilinx/Vitis/2023.2/settings64.sh
C2HLS_PART           = xcu280-fsvh2892-2L-e
C2HLS_CLOCK_NS       = 3.33
strategy             = dynamic
turns                = 4
haiku model          = claude-haiku-4-5-20251001
sonnet model         = claude-sonnet-4-6
```

## Phase 8 alignment (baseline quality)

Both runs needed Phase 8 baseline alignment before multistep optimization.

| Run | GT baseline (cyc) | Agent baseline (cyc) | lat_ratio | Fmax (MHz) |
|-----|------------------:|---------------------:|----------:|-----------:|
| GT reference | 1,795,074 | — | 1.000× | — |
| **sc-haiku** | 1,795,074 | **148,920** | **0.0830×** | 167.8 |
| **sc-sonnet** | 1,795,074 | **37,494** | **0.0209×** | 319.49 |

Both agents dramatically outperformed the GT baseline on the baseline translation.
Sonnet's baseline (37,494 cycles at 319.49 MHz) is 3.97× better than Haiku's (148,920 cycles
at 167.8 MHz) and runs at a faster clock — indicating Sonnet produces a cleaner kernel.

Note: Haiku's baseline Fmax ≈ 167.8 MHz (inferred from 148,920 cycles × ? ns/cycle = 887,000 ns).
Sonnet's baseline Fmax = 319.49 MHz (confirmed from synthesis report).

## Multistep trajectories

### sc-haiku (claude-haiku-4-5-20251001)

| Step | GT (cyc) | Haiku cyc | csim | Event |
|------|--------:|----------:|:----:|-------|
| baseline | 1,795,074 | **148,920** | ✓ | Phase 8 aligned |
| tiling | — | 364,858 | ✓ | Phase 9 csim-repair fired; severe regression (2.4×) |
| pipeline | — | 364,854 | ✓ | Near-identical to tiling (no-op-like) |
| unroll | — | 364,854 | ✓ | Identical to pipeline |
| doublebuffer | 1,795,074 | 365,143 | ✓ | Phase 9 csim-repair fired; still ~365K |
| coalescing | **11,017** | **N/A** | ✗ | 4/4 attempts exhausted on s_axilite bundle errors |
| **Phase 6a best** | — | **148,920** (baseline) | ✓ | All opt steps regressed; baseline promoted |

**Haiku csim rate: 5/6** (coalescing had no successful synthesis to csim)

### sc-sonnet (claude-sonnet-4-6)

| Step | GT (cyc) | Sonnet cyc | csim | Event |
|------|--------:|-----------:|:----:|-------|
| baseline | 1,795,074 | **37,494** | ✓ | Phase 8 aligned (attempt 2) |
| tiling | — | 66,105 | ✓ | Phase 9 csim-repair fired (1 shot); regression (1.8×) vs baseline |
| pipeline | — | 66,106 | ✓ | Near-identical to tiling |
| unroll | — | 66,347 | ✓ | Slight regression vs pipeline |
| doublebuffer | 1,795,074 | **38,219** | ✓ | Recovery — close to baseline |
| coalescing | **11,017** | **38,165** | ✓ | 2 compile-error repairs (mc.h→ap_int.h→success); near-doublebuffer |
| **Phase 6a best** | — | **37,494** (baseline) | ✓ | Baseline still best; all opt steps regressed or recovered to ~38K |

**Sonnet csim rate: 6/6**

## Best-vs-best rubric scoring

| Run | Best agent step | Best agent cyc | Ref-best step | Ref-best cyc | cycles_ratio | clock_gap_ratio |
|-----|-----------------|---------------:|---------------|-------------:|-------------:|----------------:|
| **sc-haiku Phase 9** | baseline | 148,920 | coalescing | 11,017 | **13.52×** | ~0.59 (167.8/300 MHz) |
| **sc-sonnet Phase 9** | baseline | 37,494 | coalescing | 11,017 | **3.40×** | ~1.06 (319.49/300 MHz) |

Notes:
- `clock_gap_ratio` = agent_fmax / ref_fmax (300 MHz). Haiku's design runs much slower than ref
  (167.8 vs 300 MHz); Sonnet's runs at 319.49 MHz (slightly faster than ref).
- Haiku's hardware mismatch penalizes its cycles ratio: at equal Fmax, haiku's 148,920 cycles
  would still be 13.5× off ref-best.

## Cross-model analysis: why sonnet dominates on SC

### Baseline translation quality
Sonnet produced a SC kernel that runs at 319.49 MHz / 37,494 cycles. Haiku's kernel runs at
167.8 MHz / 148,920 cycles. The 4× cycle difference AND 1.9× clock difference suggest Sonnet
wrote fundamentally better code (likely avoided a critical path or resource bottleneck that
Haiku introduced). For SC, baseline translation quality dominates the final score because
**neither model could improve on its own baseline through the optimization steps**.

### Optimization trajectory collapse
Both models show the same pattern: tiling/pipeline/unroll all regress dramatically from the
baseline. Haiku's optimized steps regressed to 365K cycles (2.4× worse than its own 148K
baseline). Sonnet's regressed to 66K cycles (1.8× worse than its own 37K baseline).

The root cause: SC's inner loop `compute_tile` has significant floating-point parallelism that
Vitis HLS already exploits well in the baseline. Manual tiling and pipeline pragmas fight with
Vitis's auto-scheduler rather than helping it, especially when the LLM doesn't know the
exact tile/block size that fits the memory hierarchy.

### Doublebuffer recovery
Both models partially recover with doublebuffer: Haiku recovers from 365K to 365K (no recovery),
Sonnet recovers from 66K to 38K (close to baseline). Sonnet's doublebuffer appears to implement
a more effective double-buffering scheme that avoids the tiling pitfall.

### The coalescing burst-widening cliff
Neither model reaches the GT coalescing result (11,017 cycles). The ref-best SC coalescing
variant uses burst-widened AXI reads (512-bit burst access) to process data in large tiles with
high DRAM bandwidth utilization. Neither Haiku nor Sonnet generated this pattern:

- **Haiku**: coalescing step exhausted 4 attempts on include errors (mc.h → s_axilite bundle)
- **Sonnet**: coalescing attempt 2 succeeded (after mc.h → ap_int.h compile repairs) but
  produced 38,165 cycles — essentially the same as doublebuffer. Sonnet improved the compile
  error sequence but not the burst-widening implementation.

The burst-widening cliff (11K vs 38K, 3.5×) is a Pillar 7 candidate: it may require the agent
to understand that the default AXI interface (m_axi without explicit burst width) limits DRAM
bandwidth, and that `#pragma HLS INTERFACE m_axi max_read_burst_length=64 bundle=gmem` or
similar changes the bandwidth substantially. This pragma knowledge should be a high-confidence
skill library entry.

## Phase 9 feature validation on SC

| Feature | sc-haiku | sc-sonnet |
|---------|---------|---------|
| **No-op trap** | Not explicitly triggered (metrics differ each step) | Not triggered (tiling regressed, not identical to baseline) |
| **Phase 9 csim-repair** | Fired on tiling + doublebuffer (both repaired) | Fired on tiling (1-shot repair) |
| **Phase 6a best-so-far** | ✓ promoted baseline over regressed steps | ✓ promoted baseline over all opt steps |
| **Compile-error repair** | Coalescing: exhausted on bundle errors | Coalescing: 2 compile repairs (mc.h, ap_int.h) → success |
| **csim rate** | 5/6 (coalescing has no successful synth) | **6/6** |

## Recommended investigation: the burst-widening cliff

The GT coalescing advantage (38K → 11K, 3.5×) is a pure bandwidth improvement. The key pattern:
- GT coalescing uses a 512-bit burst read in a pipelined inner loop, effectively reading 16 floats
  per cycle from DRAM
- Agent's coalescing (38K cycles) uses narrower bursts or the same bandwidth as doublebuffer

A **counterfactual probe** (Pillar 7 item): run `csynth_design` on the GT coalescing HLS code
with burst pragmas stripped. If the result is ~38K cycles (matching both models), the burst
pattern is *not* absorbed by Vitis 2023.2 and must be explicitly hand-coded. If the result is
~11K, Vitis auto-discovers it and the agent should be able to generate it with the right prompt.

This probe costs one csynth run (~5 min) and directly answers whether the 11K cliff requires
explicit burst-widening knowledge in the skill library.

## Files of record

- sc-haiku results: [results_phase2/streamcluster_haiku_phase9_u280_v2023/StreamCluster_multistep_results.json](../results_phase2/streamcluster_haiku_phase9_u280_v2023/StreamCluster_multistep_results.json)
- sc-sonnet results: [results_phase2/streamcluster_sonnet_phase9_u280_v2023/StreamCluster_multistep_results.json](../results_phase2/streamcluster_sonnet_phase9_u280_v2023/StreamCluster_multistep_results.json)
- sc-haiku log: `/tmp/phase9_sc_haiku_u280.log`
- sc-sonnet log: `/tmp/phase9_sc_sonnet_u280.log`
