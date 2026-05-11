# Pathfinder full-metrics comparison (cycles + wall time)

_generated 2026-05-08. Correcting an earlier under-stated gap claim._

## TL;DR — the gap to philip's reference is 29x in cycles, 46x in time

| Comparison axis | Our Run 4 (best) | Philip's coalescing | Ratio |
|-----------------|-----------------:|--------------------:|------:|
| **Latency (cycles)** | 639,710 | 22,090 | **29.0× slower** |
| **Latency (wall time)** | 3,420 us | 73.6 us | **46.5× slower** |
| Cycle-period (request) | 4.00 ns | 3.33 ns | 1.20× wider |
| Achieved Fmax | 187 MHz | 411 MHz | 0.46× slower clock |
| FPGA target | xc7a100t-csg324-1 | xcu280-fsvh2892-2L-e | datacenter vs Artix-7 |

The clock + FPGA gap accounts for only ~1.6× of the time-gap; the
remaining **~28× cycle gap is purely algorithmic** — none of our runs
produced a `coalescing` rewrite that captured philip's 48× burst-widening
win.

## Full per-run table (both metrics)

| Run | LLM | Strategy / Phase | baseline_cyc | baseline (ms) | best_cyc | best (ms) | Fmax | survived |
|-----|-----|------------------|-------------:|---------------|---------:|-----------|-----:|:--------:|
| 1 | Haiku | dynamic | 2,375,633 | 12.70 | 2,322,488 | 12.19 | 190 | 2/5 |
| 2 | Sonnet | dynamic | 2,112,720 | 11.30 | 1,626,264 | 8.70 | 187 | 3/5 |
| 3 | Sonnet | dynamic + Phase 5, strict 1.10× | 2,111,697 | 11.29 | 1,626,264 | 8.70 | 187 | 4/5 |
| **4 (B)** | **Sonnet** | **dynamic + Phase 5, global 1.25×** | 2,110,674 | 11.29 | **639,710** | **3.42** | 187 | 5/5 |
| C | Sonnet | dynamic + Phase 5, per-step | 2,111,697 | 11.29 | 1,132,615 | 6.06 | 187 | 5/5 |
| D | Sonnet | Phase 6 forward_eval + per-step | 2,111,697 | 11.29 | 1,098,904 | 5.88 | 187 | 5/5 |

## Philip's reference (full)

Vitis 2023.2 / U280 / 3.33 ns clock target:

| variant | cycles | wall time | est_period_ns | Fmax≈ | comment |
|---------|-------:|----------:|--------------:|------:|---------|
| baseline | 2,113,742 | 7.045 ms | 2.545 | 393 MHz | |
| tiling | 3,159,646 | **10.531 ms** | 2.433 | 411 | reference *itself* regresses (+50%) on tiling — the canonical enabling-step pattern |
| unroll | 2,126,430 | 7.087 ms | 2.433 | 411 | matches baseline |
| doublebuffer | 1,056,310 | 3.521 ms | 2.433 | 411 | 50% improvement |
| **coalescing** | **22,090** | **73.6 us** | 2.433 | 411 | **48× win on top of doublebuffer (the headline gain)** |

## What this tells us about each run's trajectory shape

| Run | Reached doublebuffer-equivalent? | Reached coalescing-equivalent? |
|-----|:-:|:-:|
| 1 (Haiku) | ❌ (2.32M cyc, ~baseline) | ❌ |
| 2 (Sonnet) | partial (1.63M cyc, mid-way) | ❌ |
| 3 (Phase 5 strict) | partial (1.63M cyc) | ❌ |
| **4 (Phase 5 1.25×)** | **✓ (640k cyc, near doublebuffer ref's 1.06M)** | ❌ — never widened the burst |
| C (per-step) | ✓ (1.13M cyc) | ❌ |
| D (forward_eval) | ✓ (1.10M cyc, via best-so-far promotion) | ❌ |

**The plateau is real**: every Phase 5/6 improvement got us *to* roughly
doublebuffer-class performance (~1M cycles) but **none of them landed a
coalescing rewrite that crossed the burst-widening cliff** (the
1,056,310 → 22,090 jump is 48×). That's where the remaining 28× cycle
gap lives.

Phase 7a's [`burst.xml` parser](../hls_feedback.py) is now in place and
*does* show that earlier coalescing attempts widened bursts to 512-bit
(`Sequential read of 1048576 x 32bit widened by 16`) — but only on
*some* runs and the LLM's overall coalescing rewrite still didn't
collapse latency the way philip's reference does. Diagnosing that
specifically is the next high-value action: if bursts widen but
latency stays high, the issue is the inner-loop unroll on the widened
data path, not the burst widening itself.

## Why the wall-time gap (46×) is wider than the cycle gap (29×)

Two compounding effects, both on our side:

1. **Clock period**: 4 ns request vs 3.33 ns reference → 1.20× per cycle.
2. **Achieved Fmax**: ours synth'd to ~187 MHz vs reference's ~411 MHz
   → another 0.46× factor. Combined: 1.20 × 2.20 ≈ **2.6× time per cycle**.

So the cycle-side comparison is the apples-to-apples one:
- Cycles: **29× slower**, purely algorithmic — *the agent has not landed
  the coalescing optimization*.
- Wall time: **46× slower**, of which ~28× is algorithmic and ~1.6× is
  toolchain (clock period + actual Fmax). The toolchain piece is partly
  the FPGA target (Artix-7 100T vs Alveo U280) and partly that our
  implementations have more control overhead, dropping the achieved
  clock from 411 MHz to 187 MHz.

## Implications for the framework

1. **Reporting conclusion**: every artifact going forward shows both
   `latency_cycles` and `latency_ns` side-by-side. Wall-time gap can be
   misleading because clock-period and Fmax differences amplify the
   apparent gap; cycle-count gap is the agent-attributable signal.

2. **Real next-step focus**: closing the coalescing gap. The agent needs
   a kernel that:
   - widens AXI bursts to 512-bit (✓ already happens — Phase 7a confirms it)
   - **AND** unrolls the *post-widening* inner loop by 16× to consume
     16 × 32-bit words per clock (this is what philip's reference does
     and what our runs have been missing)
   - **AND** uses dataflow at the kernel boundary so the wide reads
     overlap with compute

3. **Phase 7a → Phase 7a-tail (the recommended next change from last
   round)** becomes more obviously valuable: when the FeedbackAgent's
   `compose_with_llm` retries a regressed `coalescing` step with the
   `static_extras` block in its prompt, it will see "bursts widened ✓
   but latency_cycles still 1M+" and can suggest the missing
   post-widening unroll explicitly.

## Summary check on artifact correctness

I went back and audited every artifact written this round:

| Artifact | Showed cycles? | Showed wall time? | Now corrected? |
|----------|:-:|:-:|:-:|
| phase2_e2e_knn_final | only ns | ✓ | needs both |
| phase2_e2e_cross_kernel_summary | mixed | ✓ | needs both |
| phase5_compounding_pathfinder | only ns | ✓ | needs both |
| phase5_per_step_thresholds_pathfinder | only ns | ✓ | needs both |
| phase6_forward_eval_pathfinder | only ns | ✓ | needs both |
| **pathfinder_full_metrics_comparison** (this) | **✓** | **✓** | (canonical) |

I'll treat this as the canonical pathfinder comparison artifact going
forward. Earlier ones remain as-written for trace integrity.
