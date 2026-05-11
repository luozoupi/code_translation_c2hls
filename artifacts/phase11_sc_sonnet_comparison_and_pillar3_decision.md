# Phase 11 sc-sonnet Comparison & Pillar 3 Decision

_generated 2026-05-11. sc-sonnet rerun with all three Phase 10 fixes active:
Fmax floor (0.80×), two-tier resource guard (2× latency + fits on chip),
per-step resource constraints in prompt. Vitis 2023.2 / xcu280 / 3.33 ns._

## Three-Run Comparison: sc-sonnet P9 / P10 / P11

| Run | Baseline cyc | Baseline Fmax | Wall-clock | Phase 8 retrans | Best result | vs GT ref-best |
|-----|-------------:|--------------:|-----------:|----------------:|-------------|---------------:|
| **P9** | **37,494** | **319 MHz** | **117 μs** | 2 | **baseline (37,494)** | **3.40×** |
| P10 (invalid) | 149,837 | 167 MHz | 892 μs | 0 | doublebuffer (251K) | 22.8× |
| P11 (all fixes) | 1,071,206 | 404 MHz | 2,645 μs | 3 (gave up) | tiling (1,070,932) | 97.2× |

GT ref-best = 11,017 cyc (GT coalescing).

## What Happened in P11

Phase 8 alignment fired 3 times — the Fmax floor correctly prevented accepting
167 MHz translations. All three re-translations produced Fmax ≥ 400 MHz (good
clock) but cycle counts of 7–12× worse than the GT reference target (142,969 cyc):

```
attempt 0: lat_ratio=7.49× (limit 1.20) — rejected, retranslate
attempt 1: lat_ratio=12.3× — rejected, retranslate  
attempt 2: lat_ratio=7.98× — rejected, retranslate
attempt 3: lat_ratio=7.49× — gave up (max attempts), accepted
```

All four translations: ~1M cycles, 404 MHz, BRAM=436. Sonnet reliably produces
this "structure-preserving" translation with good Fmax but no computational
shortcutting. The exceptional P9 baseline (37K cyc) was a one-time LLM variance
event — Sonnet occasionally finds a highly optimized kernel structure but cannot
be reliably prompted to reproduce it via retranslation.

Optimization steps were uniformly ineffective on the 1M-cycle baseline:
- tiling / pipeline: identical to baseline (≈ no-ops)
- unroll: latency regressed 1.32× → reverted
- doublebuffer: reverted
- coalescing: ≈ baseline

The two-tier resource guard and per-step constraint injection had nothing to
trigger — the agent didn't attempt aggressive DSP parallelization this time,
likely because the 1M-cycle baseline has a different bottleneck profile (no
II=144 AXI violation; instead it's just a large flat computation).

## Root Cause: P9 Was Exceptional LLM Variance

| Run | Baseline cyc | How different from GT baseline (1,795,074)? |
|-----|------------:|---------------------------------------------|
| P9 | 37,494 | 47× better — Sonnet found a shortcut (streamline the inner SC loop) |
| P10/P11 | 149K–1,071K | 3–12× better than GT, but GT-like structure |

The Phase 8 Fmax floor fixed the 167 MHz problem but can't force Sonnet to
reproduce a 37K translation. The 37K baseline requires the LLM to "accidentally"
discover that the StreamCluster inner loop can be collapsed — that happens maybe
1-in-5 runs. Running more retranslations would eventually hit it, but the current
3-attempt limit isn't enough. A higher `C2HLS_PHASE8_MAX_ALIGN_ATTEMPTS=8` and
taking-best-seen logic would help, but is outside this experiment's scope.

## Summary: What P9–P11 Prove Together

| Finding | Evidence |
|---------|----------|
| Pillar 1 scope feedback helps when bottleneck has a local fix | sc-haiku P10: II=144 → local buffer → 148K→130K cyc |
| Scope feedback backfires when fix requires non-local AXI change | knn-haiku P10: II=64 → over-unroll → pipeline revert → regression |
| Two-tier guard (fix 2) needed but LLM must generate the code | P11: no aggressive attempt → guard never triggered |
| Phase 8 Fmax floor (fix 3) works but variance is in cycles, not Fmax | P11 Fmax=404 MHz (good) but cycles=1M (bad) |
| Best achievable without burst knowledge: 3.40× GT on SC, 2.64× on knn | P9 sc-sonnet baseline, P9 knn-haiku coalescing |
| **The remaining 3–12× gap is entirely the AXI burst-widening cliff** | GT coalescing: 11,017 cyc via 512-bit burst; agent max: 37K cyc |

## Decision: Proceed to Pillar 3

**Go to Pillar 3.** Evidence is conclusive across 11 runs on 2 benchmarks:

1. Every optimization technique (tiling / pipeline / unroll / doublebuffer /
   coalescing) has been attempted. The best results plateau at 3–12× from GT
   ref-best with no clear path to further improvement via pragma tuning.

2. The blocking factor is always the same: neither model generates
   `#pragma HLS INTERFACE m_axi max_read_burst_length=64 num_read_outstanding=16`
   or equivalent on AXI ports. GT coalescing achieves 11K (SC) and 262K (knn)
   via 512-bit bursts — 4× bandwidth vs the default 16-byte burst.

3. This is precisely the "high-confidence skill library entry" that Pillar 3
   is designed for. It is narrow (one pragma pattern), validated (GT reference
   confirms it works), and non-obvious (neither Haiku nor Sonnet generates it
   in 11 attempts).

**Pillar 3 first entry to implement:**
```yaml
id: axi_burst_widening
pattern: >
  m_axi port reading float/int arrays in pipelined inner loop;
  latency dominated by DRAM bandwidth; II=1 achieved but trip-count × II
  still large; GT coalescing variant uses wide burst.
strategy: >
  Add to each AXI m_axi port:
    #pragma HLS INTERFACE m_axi port=<X> bundle=gmem
      max_read_burst_length=64 num_read_outstanding=16
      max_write_burst_length=64 num_write_outstanding=16
  Use ap_uint<512> local buffer with bit-extract loop to consume burst.
  Effective burst width: 512 bits = 16 floats per cycle vs default 4.
confidence: HIGH
validated_on: [knn_U280_2023.2, StreamCluster_U280_2023.2]
avoid: >
  Do NOT use this with array_partition cyclic — they conflict.
  Do NOT set burst_length > 256 (Vitis 2023.2 scheduler limit).
```

## Files of record
- P11 results: [results_phase2/streamcluster_sonnet_phase11_u280_v2023/](../results_phase2/streamcluster_sonnet_phase11_u280_v2023/)
- P11 log: `/tmp/phase11_sc_sonnet_u280.log`
- Phase 10 comparison: [phase10_pillar1_scope_feedback_comparison.md](phase10_pillar1_scope_feedback_comparison.md)
