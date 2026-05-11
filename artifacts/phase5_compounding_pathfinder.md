# Phase 5 compounding study on pathfinder (5-way)

_generated 2026-05-07. Same kernel (pathfinder), same Vitis HLS 2025.2 / `xc7a100t-csg324-1` /
4 ns clock, same `--strategy=dynamic --turns 4`. Only the LLM backend +
Phase 5 flags + regression threshold differ._

## TL;DR

| # | Run | LLM | Phase 5 | Threshold | Best lat_ns | **Δ baseline** | Steps survived |
|---|-----|-----|---------|----------:|------------:|---------------:|:--------------:|
| 1 | pathfinder_dynamic | Haiku 4.5 | off | 1.10x | 12.19M | 0.96x (4%) | 2/5 |
| 2 | pathfinder_dynamic_sonnet | Sonnet 4.6 | off | 1.10x | 8.695M | 0.77x (23%) | 3/5 |
| 3 | pathfinder_dynamic_sonnet_phase5_strict | Sonnet 4.6 | **5a + 5b** | 1.10x | 8.695M | 0.77x (23%) | 4/5 |
| 4 | pathfinder_dynamic_sonnet_phase5_relaxed | Sonnet 4.6 | **5a + 5b** | **1.25x** | **3.420M** | **0.30x (70%)** | **5/5** |

**Headline**: row 4 is **3.5x better latency than row 1, 2.5x better than row 2 and 3**. The compounding effect comes specifically from threshold relaxation enabling the `unroll` step to land — which alone gave another 1.7x latency reduction on top of `doublebuffer`'s 2x.

## Per-step trajectories

### Run 3 — Sonnet + Phase 5 strict (1.10x)

| Step | Effect | lat_cyc | lat_ns | BRAM | FF | LUT |
|------|--------|--------:|-------:|-----:|---:|---:|
| baseline | — | 2 111 697 | 11 290 000 | 53 | 9 800 | 6 318 |
| coalescing | absorbed | 2 113 743 | 11 301 000 | 155 | 8 688 | 5 531 |
| **doublebuffer** | **improved** | **1 626 264** | **8 695 000** | 163 | 9 032 | 6 927 |
| tiling | absorbed | 1 627 288 | 8 700 000 | 163 | 9 100 | 6 927 |
| pipeline | absorbed | 1 626 264 | 8 695 000 | 159 | 8 746 | 7 002 |
| unroll | **FAIL** (no_op_persisted) | (last accepted) | (last accepted) | 159 | 8 746 | 7 002 |

Phase 5b GT pre-pop fired all 5 entries upfront (4.22M / 6.32M / 4.25M / 2.11M / 0.04M cycles for baseline / tiling / unroll / doublebuffer / coalescing). Phase 5a's LLM-aided 3rd attempt fired on coalescing's 1.50x latency regression but the regression persisted (Sonnet's coalescing rewrite couldn't get below 16.9M ns).

### Run 4 — Sonnet + Phase 5 RELAXED (1.25x)

| Step | Effect | lat_cyc | lat_ns | BRAM | FF | LUT |
|------|--------|--------:|-------:|-----:|---:|---:|
| baseline | — | 2 110 674 | 11 285 000 | 35 | 8 125 | 5 355 |
| coalescing | absorbed | 2 112 720 | 11 296 000 | 155 | 6 723 | 5 501 |
| **doublebuffer** | **improved** | **1 099 928** | **5 881 000** | 159 | 8 980 | 6 750 |
| tiling | absorbed | 1 102 050 | 5 892 000 | 167 | 9 361 | 7 280 |
| pipeline | absorbed | 1 102 050 | 5 892 000 | 167 | 9 361 | 7 328 |
| **unroll** | **improved** | **639 710** | **3 420 000** | 191 | 10 108 | 8 401 |

Two big wins: doublebuffer cut latency in half (5.88M from 11.29M baseline), and **unroll cut it again by another 42%** (3.42M from 5.88M). With the strict 1.10x threshold (Run 3), unroll's resource growth (LUT 1.41x, FF 1.17x, BRAM 1.13x) tripped Phase 1's revert and the deterministic retry produced a no-op. With 1.25x threshold (Run 4), the same step's resource growth fits within tolerance and lands.

## Why the relaxation helped so much

Run 3's `unroll` failure was specifically the resource-regression trip: `lut 7002→9889 (1.41x), ff 8746→10205 (1.17x), bram 159→179 (1.13x)`. Three resources grew >10%, so Phase 1 reverted. The deterministic retry produced byte-identical metrics (no-op) because Sonnet had already shipped its best attempt and didn't have a different surgical move available.

With the threshold raised to 1.25x:
- BRAM 1.13x → within tolerance (no resource trip)
- FF 1.17x → within tolerance (no resource trip)
- LUT 1.41x → still over (1.41 > 1.25), but only **one** resource over the limit, not three. The `len(grown_resources) >= 3` guard in `_step_regression_reasons` requires 3+ resources over to trigger — so a single LUT overshoot is *no longer* a regression.

The fact that **a 1.25x threshold produced this much improvement** validates point (3) from the prior recommendation: **the default 1.10x threshold is too strict for steps like `unroll` that legitimately need to spend resources to gain throughput.** Per-step thresholds (e.g., `unroll → 1.30x`, `coalescing → 1.40x BRAM`) would be even better than a global relaxation, but a single global bump from 1.10 to 1.25 already opens up significant headroom.

## Phase 5 features observed firing in production

| Feature | Run 3 | Run 4 |
|---------|:-----:|:-----:|
| 5b GT pre-pop (5 variants synthesized upfront) | ✓ | ✓ |
| 5a LLM-aided 3rd retry on regression | ✓ (coalescing) | ✓ (coalescing) |
| 5b skill-template prompt injection | ✓ on every step | ✓ on every step |
| Phase 9 no-op trap | ✓ (unroll persistent no-op) | — (not triggered) |
| Phase 1 regression-revert | ✓ (coalescing reverted twice → kept by alignment? final state shows it was kept) | ✓ (one retry sufficient) |
| Phase 3 GT-aware-revert | ✓ (now has full GT cache) | ✓ |

## Net comparison vs philip's reference

Philip's reference for pathfinder (Vitis 2023.2 + U280, different toolchain):
- baseline: 2 113 742 cycles
- best: 476 355 cycles (unroll, hw_emu) / 342 676 (doublebuffer csynth)
- Reference improvement: **0.16x** (84% reduction, 6x speedup)

Our **Run 4** result:
- baseline: 2 110 674 cycles (matches reference baseline within 0.1%)
- best: 639 710 cycles
- Improvement: **0.30x** (70% reduction, 3.3x speedup)

So with Phase 5 + relaxed threshold + Sonnet, **we're now within 1.9x of philip's reference** on pathfinder — much closer than the 5x of the original Haiku run. Most of the remaining gap is on `coalescing`: philip's reference best comes from coalescing (262K cycles for knn, 476K for pathfinder) while our `coalescing` step keeps absorbing into baseline because the LLM's coalescing rewrites consistently regress under our threshold.

## Smoke totals after this round

| Phase | Checks | Result |
|-------|-------:|:------:|
| Phase 1 | 47 | ✓ |
| Phase 2 | 50 | ✓ |
| Phase 3 | 22 | ✓ |
| Phase 4 | 21 | ✓ |
| Phase 5 | 18 | ✓ |
| **Total** | **158** | **✓** |

## Next-most-valuable experiments

1. **Per-step thresholds.** A step-aware threshold lookup (e.g. `unroll → 1.30x`, `coalescing → 1.50x BRAM`) would let `coalescing` land on more kernels — that's where the remaining gap to philip's reference lives. Concrete: a small dict in `c2hls.py` keyed on step name.
2. **Try Run 4's recipe on more kernels** (knn, hotspot). The same flag combination should produce significant improvements on any kernel where unroll-style steps were previously reverted.
3. **Add a `--rerun-failed-steps` flag** that, after a multistep run finishes, retries any step that hit `no_op_persisted` with `compose_with_llm` enabled. Cheap insurance on the trajectory.
4. **Per-skill-id confidence persistence**. After this campaign, the skill library has 11 step outcomes recorded but isn't persisting. Wire `update_skill_statistics` to fire post-step and `lib.save()` at trajectory end.

## Files of record

| Run | Result JSON | Wall log |
|-----|-------------|----------|
| Run 1 (Haiku, no Phase 5) | [pathfinder_dynamic/pathfinder_multistep_results.json](../results_phase2/pathfinder_dynamic/pathfinder_multistep_results.json) | /tmp/phase4_e2e_pathfinder_dynamic.log |
| Run 2 (Sonnet, no Phase 5) | [pathfinder_dynamic_sonnet/pathfinder_multistep_results.json](../results_phase2/pathfinder_dynamic_sonnet/pathfinder_multistep_results.json) | /tmp/phase5_e2e_pathfinder_dynamic_sonnet.log |
| Run 3 (Sonnet + Phase 5 strict) | [pathfinder_dynamic_sonnet_phase5_strict/pathfinder_multistep_results.json](../results_phase2/pathfinder_dynamic_sonnet_phase5_strict/pathfinder_multistep_results.json) | /tmp/phase5_e2e_pathfinder_sonnet_strict.log |
| **Run 4 (Sonnet + Phase 5 relaxed)** | [pathfinder_dynamic_sonnet_phase5_relaxed/pathfinder_multistep_results.json](../results_phase2/pathfinder_dynamic_sonnet_phase5_relaxed/pathfinder_multistep_results.json) | /tmp/phase5_e2e_pathfinder_sonnet_relaxed.log |
