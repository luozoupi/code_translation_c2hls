# Phase 10 — Pillar 1 Scope Feedback: Comparison with Phase 9

_generated 2026-05-09. First run after wiring Pillar 1 per-loop bottleneck data into every
optimization prompt (format_report_summary now includes top-6 per-loop bottleneck records) and
adding a baseline-vs-current scope diff block injected into each optimization step prompt.
Vitis 2023.2 / xcu280-fsvh2892-2L-e / 3.33 ns clock._

## What Changed (Phase 9 → Phase 10)

Two new signals added to every optimization step prompt:

1. **Per-loop bottlenecks in `{synth_report}`** (`hls_eval.py:format_report_summary`): The
   existing 12 top-level scalars now include a `per_loop_bottlenecks` section showing the top-6
   Pillar 1 records: scope_id, kind (ii_target_miss / interval_exceeds_latency / pipeline_blocked),
   evidence (e.g. "II=144"), and source location. Previously the LLM only saw total latency/BRAM/DSP.

2. **Baseline-vs-current scope diff** (`c2hls.py:_render_baseline_scope_diff`): Each step prompt
   receives an injected block comparing the baseline's per-loop bottlenecks against the current
   step's bottlenecks — showing which loops regressed (NEW bottleneck), which resolved (RESOLVED ✓),
   and the latency ratio vs baseline. This tells the agent exactly what its prior steps broke.

## Results Summary

| Run | Phase | Baseline (cyc) | Best step | Best (cyc) | vs GT ref-best | Δ vs Phase 9 |
|-----|-------|---------------:|-----------|----------:|---------------:|--------------|
| knn-haiku | P9 | 1,048,816 | coalescing | **692,437** | **2.64×** | — |
| knn-haiku | P10 | 1,048,816 | doublebuffer | 1,756,923 | 6.69× | **↑ WORSE** |
| knn-sonnet | P9 | 1,048,818 | baseline | 1,048,818 | 3.99× | — |
| knn-sonnet | P10 | 1,048,753 | doublebuffer | 2,027,529 | 7.72× | ≈ same |
| sc-haiku | P9 | 148,920 | baseline | 148,920 | 13.52× | — |
| sc-haiku | P10 | 148,920 | doublebuffer | **130,627** | **11.86×** | **↓ BETTER** |
| sc-sonnet | P9 | 37,494 | baseline | 37,494 | 3.46× | — |
| sc-sonnet | P10 | 1,754,532 | baseline | 1,754,532 | 159× | **INVALID** (rerun launched) |

GT ref-best: knn=262,480 cyc (coalescing), SC=11,017 cyc (coalescing).

## sc-haiku Phase 10: Major Win — Scope Feedback Worked

**Phase 9**: baseline=148,920 cyc; all 5 optimization steps regressed to ~365K cycles.
**Phase 10**: baseline=148,920 cyc; steps progressively improve; best=doublebuffer=130,627 cyc.

### Step-by-step trajectory

| Step | P9 cyc | P10 cyc | Δ | Key bottleneck in P10 |
|------|-------:|--------:|---|------------------------|
| baseline | 148,920 | 148,920 | = | II=144 on VITIS_LOOP_43_1 (AXI bus dependency) |
| tiling | 364,858 | **239,266** | **↓ 34%** | II=6 on coord_tile load loop (port conflict) |
| pipeline | 364,854 | 240,226 | ↓ 34% | II=6 same loop |
| unroll | 364,854 | 258,161 | ↓ 29% | II=42 on distance loop (partial regression) |
| doublebuffer | 365,143 | **130,627** | **↓ 64%** | II=24 on 2 parallel loops (double-buffer pattern) |
| coalescing | N/A | 133,219 | ↓ 63% vs P9 | Same II=24 pattern |
| **Phase 6a best** | **148,920** (baseline) | **130,627** (doublebuffer) | **↓ 12.3%** | First time any step beat the sc-haiku baseline |

### Mechanism

The baseline scope diff exposed a critical bottleneck the agent had never seen before:
```
BASELINE bottleneck: VITIS_LOOP_43_1 — II=144 (AXI bus carried-dependence on gmem_addr_read)
```
With this information, Haiku's tiling step introduced a local `coord_tile` buffer to stage DRAM
reads off the critical path — reducing the AXI-constrained inner loop's II from 144→6, dropping
latency from 365K→239K (P9 tiling had no such understanding and still produced 365K).

Doublebuffer then added a double-buffering scheme over the local buffer, reducing the II further
across 2 parallel loops, achieving 130,627 cycles — 12.3% better than the 148,920 baseline.
The scope diff's "TARGET: eliminate NEW bottlenecks" instruction guided subsequent steps to
fix what tiling introduced (port conflict on coord_tile) rather than blindly applying a pragma.

## knn-haiku Phase 10: Regression — Scope Feedback Backfired

**Phase 9**: coalescing=692,437 (progressive: unroll=1.13M → doublebuffer=940K → coalescing=692K).
**Phase 10**: pipeline reverted; doublebuffer=1,756,923 (worse than P9's 940K baseline).

### What happened

The scope diff for the pipeline step showed:
```
BASELINE vs `pipeline` per-loop diff:
  workload/compute_dist loop: NEW bottleneck → II=64, AXI carried-dep on gmem_addr_read ← 
  introduced by tiling
```

Haiku's pipeline response: aggressively unrolled/parallelized the inner distance loop to fix II=64,
producing a design with LUT 7,513→24,402 (3.25×), FF 9,217→45,854 (4.97×), DSP 14→112 (8×).
The resource regression guard (`pipeline` resource limits: LUT 1.20×, FF 1.50×, DSP 1.20×)
correctly detected 3 resources over limit and reverted after 2 attempts.

After pipeline revert, subsequent steps built on the un-improved tiling foundation:
- unroll=3,358,861 (was 1,130,637 in P9 — 3× worse)
- doublebuffer=1,756,923 (was 940,434 in P9 — 1.9× worse)
- coalescing=1,830,723 (was 692,437 in P9 — 2.6× worse)

**Root cause**: For knn, the bottleneck is a data-dependent AXI distance-loop dependency that
cannot be fixed by local buffering or simple unroll. The scope diff correctly identified II=64
but the prescribed solution (local buffer + unroll) over-parallelized instead of leaving the
inner loop to Vitis's auto-pipeliner. The regression guard protected correctness but the
cascade effect on subsequent steps cost the progressive improvement P9 had achieved.

## knn-sonnet Phase 10: No Change

Both phases produce trajectories that regress on every step vs baseline:
```
P9: tiling=4.28M → pipeline=4.27M → unroll=3.75M → doublebuffer=2.03M → coalescing=2.03M
P10: tiling=4.28M → pipeline=4.27M → unroll=3.75M → doublebuffer=2.03M → coalescing=2.03M
```
The scope feedback provided no new signal to break the Sonnet regression pattern on knn.
Sonnet's bottleneck on knn (AXI burst-width limitation) requires skill library knowledge
(explicit `max_read_burst_length=64` pragma), not per-loop II feedback.

## sc-sonnet Phase 10: INVALID

Phase 8 alignment accepted a 1,754,532-cycle baseline (vs P9's 37,494 cycles) because this
translation is within 1.20× of the GT reference (1,795,074 cycles). The Phase 8 alignment
check prevents 1.20×-worse translations but does not force the LLM toward high-quality
optimized translations. A rerun was launched; results pending.

All optimization step latencies show `None` (variable-length loops from the new structure),
confirming the baseline code structure changed fundamentally due to LLM variance.

## Analysis: When Scope Feedback Helps vs Hurts

| Bottleneck type | SC (haiku) | knn (haiku) |
|----------------|-----------|------------|
| **Root cause** | AXI bus dependency II=144 | AXI burst-width II=64 (data-dep distance) |
| **Local buffer fix** | ✓ Effective (II 144→6) | ✗ Over-parallelizes (8× DSP blowup) |
| **Scope feedback effect** | ✓ Guided agent to correct fix | ✗ Prompted over-aggressive response |
| **Result** | Win: 148K→130K cycles | Loss: pipeline revert → cascade regression |

**Key insight**: The scope feedback works when the bottleneck has a local solution (stage DRAM
reads into a local buffer, reducing memory port conflicts). It backfires when the bottleneck
requires a non-local solution (burst-widening the AXI interface width, which requires a pragma
that changes the entire memory interface topology, not just the inner loop).

The skill library (Pillar 3) should encode this distinction:
- `ii_target_miss` with `gmem_addr_read` carried-dep + simple array → `array_partition + local buffer`
- `ii_target_miss` with `gmem_addr_read` carried-dep + no-local-solution → `AXI max_read_burst_length=64`

## Recommendations

1. **Re-run sc-sonnet P10** — current result is LLM-variance noise (baseline 1.75M vs 37K).
   A re-run is live; track whether Sonnet also shows scope-feedback benefit on SC.

2. **Per-step resource limit tuning for pipeline** — knn-haiku's pipeline revert was triggered
   by DSP 1.20× limit. knn's inner distance loop legitimately uses more DSP when vectorized.
   Consider relaxing `pipeline` DSP limit to 2.0× (matching the `doublebuffer` limit) so the
   agent can explore pipelined distance computation without triggering a hard revert.

3. **Skill library entry for AXI burst-widening** (Pillar 3/5 priority):
   ```
   Pattern: ii_target_miss on loop reading from m_axi port (gmem_addr_read carried dep)
   Strategy: Add #pragma HLS INTERFACE m_axi max_read_burst_length=64 num_read_outstanding=16
   Confidence: HIGH on U280 xcu280-fsvh2892-2L-e, Vitis 2023.2
   Avoid: array_partition + local buffer (wrong tool for AXI bandwidth bottleneck)
   ```

4. **Phase 8 alignment floor** — Sonnet sometimes produces dramatically better baselines than
   GT reference (37K vs 1.79M on SC). Phase 8 currently only checks "is our code within 1.20×
   of GT?" — it should also check "is this within 1.20× of our own previous best baseline
   (if recorded)?" to prevent variance noise from wiping out a good prior translation.

## sc-sonnet P10 Rerun Results (completed 2026-05-09 23:xx)

The rerun also produced a variance-afflicted baseline (149,837 cycles at 167.87 MHz = **892.6 μs
wall-clock**). For reference, P9's baseline was 37,494 cycles at 319.49 MHz = **117.4 μs**. Phase 8
accepted the rerun baseline because its cycle count (149,837) is within 1.20× of the GT baseline
cycle count (142,969 cycles at 411.35 MHz), despite a 7.6× wall-clock gap to P9's best translation.

### Rerun step trajectory

| Step | Cycles | Fmax (MHz) | Wall-clock (μs) | vs baseline | Note |
|------|-------:|-----------:|----------------:|------------:|------|
| baseline | 149,837 | 167.87 | 892.6 | 1.00× | LLM variance; P9 was 117.4 μs |
| tiling (attempt 0) | 67,646 | 411.35 | 164.4 | **0.18×** | DSP 12→356 (29.67×) → resource guard BLOCKED |
| tiling (accepted) | 390,471 | 167.87 | 2,326.0 | 2.61× | Conservative retry, severe regression |
| pipeline | — | — | — | — | Both attempts: DSP 25.50× → REVERTED |
| unroll | 211,271 | 167.87 | 1,258.5 | 1.41× | Regression on top of tiling |
| **doublebuffer** | **251,272** | **402.09** | **624.9** | **0.70×** | DSP=360; fmax jump saves wall-clock |
| coalescing | 252,555 | 402.09 | 628.1 | 0.70× | ≈ doublebuffer |
| **Phase6a best** | **251,272** | **402.09** | **624.9** | **0.70×** | Best wall-clock despite high resource cost |

_Phase 9 P9 comparison_: P9's best result was baseline=37,494 cyc at 319.49 MHz = **117.4 μs** (5.3×
better wall-clock than P10 rerun doublebuffer at 624.9 μs). The 5.3× gap is entirely from LLM
variance in baseline translation, not from the scope feedback.

### Key finding: Sonnet's response to scope feedback

Sonnet sees the `II=144` bottleneck from the baseline scope diff and responds with **massive DSP
parallelization** (unrolling the float distance loop by 16–32×):
- Tiling attempt 0: 67,646 cycles — fast (0.18× baseline wall-clock!) — but DSP 29.67× → blocked
- Pipeline: similar (25.50× DSP) → blocked → reverted
- Doublebuffer: DSP=360 (30× growth) accepted only because the doublebuffer resource ceiling
  is more lenient; achieves 0.70× baseline wall-clock but with 30× resource cost

The fundamental issue: Sonnet's parallelization strategy works for Fmax improvement (167→402 MHz)
but requires far more resources than the tiling/pipeline per-step limits allow. If the pipeline
resource limits were relaxed to match the doublebuffer limits (DSP 30×), tiling attempt 0
at 67,646 cycles and 402 MHz would have been accepted — giving ~164 μs (comparable to P9's 117 μs).

### Contrast with sc-haiku P10

| Model | II=144 response | DSP growth | Resource guard outcome | Best wall-clock |
|-------|----------------|-----------|----------------------|-----------------|
| **Haiku** | Local `coord_tile` buffer | 1× (conservative) | Passes all steps | 130,627 cyc / 435.4 μs |
| **Sonnet** | Unroll + DSP parallelization | 30× (aggressive) | Blocked at tiling/pipeline; passes doublebuffer | 251,272 cyc / 624.9 μs |

Haiku's local-buffer approach is more effective per-resource. Sonnet's approach would be better in
absolute terms (67K cycles in tiling attempt 0 ≈ 164 μs vs Haiku's doublebuffer 435 μs) **if the
resource guard permitted it**.

### sc-sonnet P10 Conclusion: LLM Variance Dominates

Both sc-sonnet P10 runs (invalid: 1,754,532-cycle baseline; rerun: 149,837-cycle baseline) show
that Sonnet's baseline translation quality for StreamCluster is highly variable. With a 7.6×
wall-clock gap between P9's best translation and P10's rerun translation, no conclusion can be
drawn about whether the scope feedback helps or hurts Sonnet on SC. The scope feedback effect is
**masked by baseline variance**.

**Required fix**: Phase 8 alignment currently accepts any translation within 1.20× of GT cycle
count. It should additionally require `Fmax ≥ 0.80 × ref_fmax` to catch translations like the
P10 rerun (167.87 MHz vs GT's 411.35 MHz). A design that matches GT cycle count but runs at
40% of GT Fmax is structurally different and should trigger realignment.

## Summary Across All Phase 10 Runs

| Run | Phase | Baseline μs | Best μs | Best step | Scope feedback effect |
|-----|-------|------------:|--------:|-----------|----------------------|
| sc-haiku | P9 | 892.6 | 892.6 | baseline | — |
| **sc-haiku** | **P10** | **892.6** | **435.4** | **doublebuffer** | **✓ Helped: local buffer resolved II=144** |
| sc-sonnet | P9 | 117.4 | 117.4 | baseline | — |
| sc-sonnet | P10 | 892.6 | 624.9 | doublebuffer | ✗ Masked by LLM variance; rerun needed |
| knn-haiku | P9 | 3,492.6 | 2,305.6 | coalescing | — |
| knn-haiku | P10 | 3,492.6 | 5,849.7 | doublebuffer | ✗ Pipeline over-parallelized → revert → cascade |
| knn-sonnet | P9 | 3,492.6 | 3,492.6 | baseline | — |
| knn-sonnet | P10 | 3,492.3 | 6,749.9 | doublebuffer | ✗ No change (same failure mode) |

_Wall-clock = latency_cycles / fmax_MHz (in μs)._

## Open Issues

1. **Phase 8 Fmax floor**: Add `Fmax ≥ 0.80 × ref_fmax` to baseline alignment acceptance to
   prevent accepting structurally inferior translations that happen to match GT cycle count.

2. **Per-step DSP ceiling for tiling/pipeline**: The 1.20–1.30× DSP limits block Sonnet's
   aggressive parallelization even when it achieves better latency. Consider a two-tier check:
   accept if *either* (a) within resource limits, or (b) latency improves by >2× with no
   resource exceeding absolute device capacity.

3. **Skill library entry for II=144 AXI bus dependency**: Two valid strategies exist:
   - Local buffer (conservative, works within resource limits) → Pillar 3 HIGH confidence
   - DSP parallelization (aggressive, blocked by step limits today) → Pillar 3 MEDIUM confidence
     with guidance: "only apply at doublebuffer/coalescing step where DSP ceiling is relaxed"

## Files of record

- sc-haiku P10: [results_phase2/streamcluster_haiku_phase10_u280_v2023/](../results_phase2/streamcluster_haiku_phase10_u280_v2023/)
- knn-haiku P10: [results_phase2/knn_haiku_phase10_u280_v2023/](../results_phase2/knn_haiku_phase10_u280_v2023/)
- knn-sonnet P10: [results_phase2/knn_sonnet_phase10_u280_v2023/](../results_phase2/knn_sonnet_phase10_u280_v2023/)
- sc-sonnet P10 (invalid): [results_phase2/streamcluster_sonnet_phase10_u280_v2023/](../results_phase2/streamcluster_sonnet_phase10_u280_v2023/)
- sc-sonnet P10 rerun log: `/tmp/phase10_sc_sonnet_u280_rerun.log`
- knn-haiku P10 log: `/tmp/phase10_knn_haiku_u280.log`
- knn-sonnet P10 log: `/tmp/phase10_knn_sonnet_u280.log`
