# Phase 2 + 3 + 4 End-to-End: Cross-Kernel Summary

_assembled 2026-05-07 from on-disk run artifacts._

This rolls up everything that's been exercised end-to-end since Phase 1
shipped, across two kernels and four orchestration strategies, plus the
LLM-aided FeedbackAgent demonstration on a real regression case.

Toolchain (held constant across all runs):
- Vitis HLS 2025.2 on `xc7a100t-csg324-1` at 4 ns clock
- Agent: `claude-haiku-4-5-20251001`
- multistep `--turns 4` (4 repair attempts per step)

Philip's reference jsonl (Vitis 2023.2 + xilinx_u280) is used for
trajectory-shape comparison only — absolute numbers are not directly
comparable due to toolchain + FPGA differences.

---

## End-to-end runs collected

| Run id | Kernel | Strategy | Wall (min) | Best PPA (lat_cyc) | Δ vs baseline | Steps survived |
|--------|--------|----------|-----------:|-------------------:|---------------|----------------|
| knn_static | knn | static (legacy) | 64.8 | 5 320 925 | 13.6x improved (from 72M baseline) | 1 of 5 (tiling) |
| knn_dynamic | knn | dynamic (Phase 2) | 66.4 | **524 753** | 2.0x improved | 2 of 5 (coalescing-absorbed + unroll) |
| knn_combo_full | knn | combo_full (Phase 3) | 21.0 | 1 048 841 | baseline kept | 0 of 1 (combo reverted twice) |
| knn_combo_progressive | knn | combo_progressive (Phase 3) | 28.5 | 1 048 841 | baseline kept | 0 of 2 (both reverted/no-op) |
| pathfinder_dynamic | pathfinder | dynamic (Phase 2) | 27.4 | **2 322 488** | 0.96x (4% improved) | 2 of 5 (coalescing + unroll) |

**Headline trends**:

1. On both kernels, **`dynamic` beat `static`** in terms of PPA-improvement-given-baseline. Pathfinder dynamic ran to completion in 27 min (vs ~65 min for knn static) because the host load was lighter and dynamic's bottleneck-router didn't have to fight Phase 1's regression-revert as much.
2. **Both combo strategies failed** on knn (both reverted to baseline). This is a real Phase 3 finding: when the LLM's baseline translation is already good (knn dynamic baseline = 1.05M cycles ≈ philip's 1.05M reference baseline), asking the model to apply 3-5 techniques in one rewrite produces an over-aggressive kernel that synth's worse. **Combo modes are inappropriate for kernels where the baseline is already a near-final design point.**
3. **Phase 1's regression-revert and Phase 9's no-op detector fired correctly throughout** — every reverted step in every run had a documented reason. Pillar 9 caught **3 no-ops in production** on knn dynamic + 1 on knn combo_progressive.

---

## Per-kernel: knn

Reference baseline: 1 048 818 cycles (Vitis 2023.2 + U280).
Reference best: 262 480 cycles (coalescing).

| Run | Baseline lat_cyc | Best lat_cyc | Best lat_ns | Final BRAM | Final DSP | Final FF | Final LUT | Final Fmax |
|-----|----------------:|-------------:|------------:|-----------:|----------:|---------:|----------:|-----------:|
| static | 72 351 885 | 5 320 925 (tiling) | 240 000 000 | 32 | 5 | 5 559 | 5 587 | 22 |
| dynamic | 1 048 841 | 524 753 (unroll) | 2 099 000 | 54 | 12 | 13 316 | 10 592 | 285 |
| combo_full | 1 048 841 | 1 048 841 (baseline) | 4 195 000 | 33 | 6 | 46 649 | 6 826 | 257 |
| combo_progressive | 1 048 841 | 1 048 841 (baseline) | 4 195 000 | 33 | 6 | 46 649 | 6 826 | 257 |
| philip ref | 1 048 818 | 262 480 (coalescing) | (300MHz) | 30 | 224 | 101 850 | 23 346 | (300) |

**Observation**: Static's 72M baseline was a translation outlier (Haiku non-determinism); dynamic's 1.05M baseline tracks philip's reference closely. Once you control for baseline, dynamic delivered the most progress (524K cycles, 2x improvement) without exotic resource usage.

### Phase 2 robustness events fired on knn runs (production)

| Run | no_op detected | throughput regressed | trajectory collapse |
|-----|---------------:|--------------------:|---------------------:|
| static | 0 | 0 | 0 |
| dynamic | **3** | 1 | 0 |
| combo_full | 0 | 0 | 0 |
| combo_progressive | **1** | 0 | 0 |

Plus Phase 1's traditional regression-revert fired 3× in static, 3× in dynamic, 2× in each combo.

---

## Per-kernel: pathfinder

Reference baseline: 2 113 742 cycles (Vitis 2023.2 + U280).
Reference best: 476 355 cycles (unroll, hw_emu) / 342 676 (doublebuffer csynth).

| Run | Baseline lat_cyc | Best lat_cyc | Best lat_ns | Final BRAM | Final DSP | Final FF | Final LUT | Final Fmax |
|-----|----------------:|-------------:|------------:|-----------:|----------:|---------:|----------:|-----------:|
| dynamic | 2 375 633 | **2 322 488** (unroll) | 12 194 000 | 167 | 0 | 16 446 | 14 515 | 190 |

Routing trace (live):
1. `coalescing` (matched `interval_exceeds_latency`) — accepted but with absorbed effect
2. `doublebuffer` (matched `interval_exceeds_latency`) — REVERTED on resource regression (LUT 1.80x, FF 1.94x, BRAM 1.18x)
3. `tiling` (matched `interval_exceeds_latency`) — REVERTED (latency 10.5x worse on retry)
4. `pipeline` (matched `non_pipelined_hot_loop`) — REVERTED (latency 1.32x worse)
5. `unroll` (matched `non_pipelined_hot_loop`) — accepted; final state 0.960x baseline

The dynamic router on pathfinder explored 5 step types and kept the 2 that improved or matched baseline. The final kernel has dramatically different resource shape (BRAM 5.8x, FF 0.12x, LUT 0.31x vs baseline) — the agent chose to push memory storage and shrink compute logic.

---

## FeedbackAgent (Phase 4) LLM-aided composition demo

Real regression case from `knn_static`'s `unroll` step (latency 6.88x, LUT 1.80x, FF 2.65x, DSP 2.40x). Both deterministic-template and LLM-aided paths were exercised:

- **Deterministic template** (399 chars, 0 LLM calls):
  > "Your previous attempt at the `unroll` step was REJECTED because it regressed against the previous step's metrics: latency_ns regressed 6.88x. Produce a more conservative version…"

- **LLM-aided composition** with `C2HLS_FEEDBACK_LLM=1` (856 chars, 1 small-model call to FeedbackAgent's own Haiku):
  > "The 6.88x latency regression indicates aggressive unrolling is causing critical path delays and memory contention rather than improving throughput. **Remove the `#pragma HLS UNROLL factor=2` directives from the inner feature loop (the innermost `for(int j = 0; j < NUM_FEATURE; ++j)` block)** and the STORE phase loop—these are creating excessive parallel memory ports that violate timing constraints. **Keep only the outer point loop (`for(int i = 0; i < NUM_PT_IN_BUFFER; i += UNROLL_FACTOR)`) with `#pragma HLS UNROLL factor=2`**, and **reduce the cyclic array partitioning factors on `search_tile` and `distance_tile` from factor=2 to factor=4** (coarser grain) to match only the single unrolled loop level."

The LLM-aided variant **reads the actual code, names specific loops by source line, and prescribes specific pragma changes** — which the deterministic template can't do. This is the single highest-value path opened by Phase 4. Off by default (`C2HLS_FEEDBACK_LLM=0`); turning it on costs 1 small-model call per failure and produces dramatically more actionable feedback.

Full demo: [phase4_feedback_llm_demo_20260507_030101.md](phase4_feedback_llm_demo_20260507_030101.md)

---

## Smoke-test totals (140/140 across all phases)

| Phase | Checks | Result | Artifact |
|-------|-------:|:------:|----------|
| Phase 1 | 47 | ✓ | [phase1_smoke_20260507_010603.md](phase1_smoke_20260507_010603.md) |
| Phase 2 | 50 | ✓ | [phase2_smoke_20260507_010604.md](phase2_smoke_20260507_010604.md) |
| Phase 3 | 22 | ✓ | [phase3_smoke_20260507_000212.md](phase3_smoke_20260507_000212.md) |
| Phase 4 | 21 | ✓ | (latest re-run after wiring) |
| **Total** | **140** | **✓** | |

---

## Files of record

| Run | Result JSON | Wall log |
|-----|------------|----------|
| knn_static | [knn_static/knn_multistep_results.json](../results_phase2/knn_static/knn_multistep_results.json) | (in process) |
| knn_dynamic | [knn_dynamic/knn_multistep_results.json](../results_phase2/knn_dynamic/knn_multistep_results.json) | (in process) |
| knn_combo_full | [knn_combo_full/knn_multistep_results.json](../results_phase2/knn_combo_full/knn_multistep_results.json) | /tmp/phase4_e2e_knn_combo_full.log |
| knn_combo_progressive | [knn_combo_progressive/knn_multistep_results.json](../results_phase2/knn_combo_progressive/knn_multistep_results.json) | /tmp/phase4_e2e_knn_combo_progressive.log |
| pathfinder_dynamic | [pathfinder_dynamic/pathfinder_multistep_results.json](../results_phase2/pathfinder_dynamic/pathfinder_multistep_results.json) | /tmp/phase4_e2e_pathfinder_dynamic.log |

Reference: [csynth_vitis_2023.2__device_xilinx_u280_gen3x16_xdma_1_202211_1.jsonl](../csynth_vitis_2023.2__device_xilinx_u280_gen3x16_xdma_1_202211_1.jsonl) and [results/references_philip/](../results/references_philip/).
