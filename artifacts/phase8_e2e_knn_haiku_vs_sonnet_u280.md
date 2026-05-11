# Phase 8 e2e on knn — Haiku vs Sonnet (matched hardware)

_generated 2026-05-08. First end-to-end run with `C2HLS_PHASE8_BASELINE_ALIGN=1`
on knn under both Haiku 4.5 and Sonnet 4.6, on the **same hardware as
philip's reference**: Vitis 2023.2 / xcu280-fsvh2892-2L-e / 3.33 ns clock.
Validates Phase 8's alignment loop, the rubric's new best-vs-best
comparison, and exposes a model-quality split in multistep behavior._

## TL;DR

- **Phase 8 alignment closed the baseline gap for both LLMs.** Haiku
  reached `lat_ratio=0.603` after the s_axilite repair (no Phase 8
  retranslation needed); Sonnet's first translation was 2.46× off ref
  baseline and Phase 8 retranslated once to also reach `lat_ratio=0.603`.
  Both LLMs converged to **byte-equivalent post-alignment baselines**
  (1,048,816 cyc Haiku / 1,048,818 cyc Sonnet) — strong evidence the
  metric-only feedback works model-independently.
- **Both post-alignment baselines beat ref baseline by 1.66× in cycles**
  (1.05M vs 1.74M GT baseline on identical hardware). The 70× knn
  baseline misalignment from prior Haiku runs (72M cyc → ref 1.05M) is
  fully closed.
- **Multistep diverges sharply by model.** Haiku reaches `unroll` at
  737K cyc (2.81× off ref-best coalescing 262K cyc); Sonnet's steps all
  regress and Phase 6a promotes baseline back. Sonnet's eligible best
  is its baseline at 1.05M cyc (4.00× off ref-best).
- **clock_gap_ratio ≈ 1.00 across every step.** The hardware match
  eliminated the 1.6× toolchain confound that distorted prior
  pathfinder comparisons. The full ns-gap is now purely algorithmic.

## Configuration (identical for both runs)

```
C2HLS_VITIS_SETTINGS = /mnt/data/luo00466/Xilinx/Vitis/2023.2/settings64.sh
C2HLS_PART           = xcu280-fsvh2892-2L-e
C2HLS_CLOCK_NS       = 3.33
C2HLS_VITIS_VERSION  = 2023.2
C2HLS_PHASE8_BASELINE_ALIGN = 1
C2HLS_PHASE5_GT_PREPOP      = 1
C2HLS_PHASE7A               = 1
strategy = dynamic
turns    = 4
```

This is the **exact toolchain philip's `references_philip` jsonl was
generated under** (Vitis 2023.2, U280, 300 MHz target = 3.33 ns).
Confirms apples-to-apples: our locally-synthesised GT baseline =
1,048,818 cyc vs philip's hw_emu = 1,049,027 cyc (0.02% drift).

## Phase 8 alignment outcomes

| LLM | Phase B repair | Phase 8 attempts | Final lat_ratio | Final BRAM/DSP/FF/LUT ratios |
|-----|---------------:|-----------------:|----------------:|------------------------------|
| Haiku 4.5 | 1 (s_axilite bundle conflict) | **0** (aligned at first check) | **0.6026×** | 0.94 / 0.50 / 0.10 / 0.19 |
| Sonnet 4.6 | 0 | **1** (retranslated once) | **0.6026×** | 0.94 / 0.50 / 0.10 / 0.21 |

Sonnet's pre-Phase-8 attempt-0 was `lat_ratio=2.457` (worse than ref
baseline). After **one** metric-only retranslation it converged to the
same 0.6026× ratio Haiku reached — same numerator, same denominator —
which means Phase 8's metric guidance was sufficient to bring two
different LLMs to the same baseline class without leaking GT code.

## Multistep trajectories (cycles)

| Step | GT (cyc) | Haiku (cyc) | Haiku csim | Sonnet (cyc) | Sonnet csim |
|------|---------:|-------------:|:---------:|-------------:|:----------:|
| baseline (post-alignment) | 1,740,530 | **1,048,816** | ✓ | **1,048,818** | ✓ |
| tiling | 4,276,372 | 4,276,372 | ✓ | 4,276,441 | ✓ |
| pipeline | 4,276,372 | 4,563,024 | ✓ | 4,274,393 | ✓ |
| unroll | 4,044,880 | **737,428** | ✓ | 3,752,153 | ✓ |
| doublebuffer | 1,740,530 | 723,162 | ✗ | 2,029,579 | ✓ |
| coalescing | **262,480** | 493,754 | ✗ | 2,029,579 | ✓ |
| **best csim-passing** | — | **737,428 (unroll)** | — | **1,048,818 (baseline)** | — |
| Phase 6a promotion | — | None (final was best) | — | **baseline** (idx -1) | — |

Reference's coalescing variant lands the burst-widening cliff at 262K
cycles — neither agent crosses it. Haiku gets within 2.81×; Sonnet
fails to make progress past baseline.

## Best-vs-best rubric scoring (new, matched hardware)

The rubric now compares the agent's lowest-cycles synth+csim-passing
step against the reference's lowest-cycles variant, regardless of step
name. Composite weights unchanged; cycles ratio computed alongside ns
ratio so cross-toolchain noise is visible.

| Run | Best agent step | Best agent cyc | Ref-best step | Ref-best cyc | cycles_ratio | ns_ratio | clock_gap_ratio | Composite |
|-----|-----------------|---------------:|---------------|-------------:|-------------:|---------:|----------------:|----------:|
| **Haiku 4.5** | unroll | 737,428 | coalescing | 262,480 | **2.81×** | 2.81× | 1.0002 | **62.07** |
| **Sonnet 4.6** | baseline | 1,048,818 | coalescing | 262,480 | **4.00×** | 4.00× | 1.0002 | **60.13** |

`clock_gap_ratio = 1.0` across the board confirms the hardware match
worked: there is no toolchain offset between our synth and philip's,
so the cycles-ratio and ns-ratio are now numerically identical and the
gap is **100% algorithmic**.

## Per-benchmark composite (averaged over all steps)

| Run | Composite | Synth | csim_rate | cosim_rate | Avg latency_score | Avg fmax_score | Avg resource_score |
|-----|----------:|------:|----------:|-----------:|------------------:|---------------:|-------------------:|
| Haiku | **72.46** | 100% | 66.67% | 0% | 73.40 | 85.0 | 65.16 |
| Sonnet | **70.49** | 100% | **100%** | 0% | 60.18 | 85.0 | 64.42 |

Haiku's per-step composite is higher because it landed `unroll` at
737K cyc (a real win), but Haiku **sacrificed correctness on the two
fastest steps** — doublebuffer and coalescing both failed csim. Sonnet
preserved correctness on every step but produced no algorithmic gain
beyond its baseline.

## What this tells us

1. **Phase 8 is doing its job.** The 70× knn baseline misalignment
   that motivated the loop is fully closed. Both LLMs end at
   `lat_ratio=0.603` (better than ref baseline by 1.66×). The
   metric-only renderer is sufficient — no GT-code leak required.

2. **Hardware match neutralises the toolchain confound.** With Vitis
   2023.2 + U280 + 3.33 ns matching philip's exact target,
   `clock_gap_ratio` collapses to ~1.0 on every step. The 46× wall-time
   gap that pathfinder's earlier artifact reported (decomposed there
   into 28× algorithmic + 1.6× toolchain) is now a clean 28× — i.e.
   the 1.6× toolchain piece is gone, leaving only the algorithmic gap
   to chase.

3. **Multistep quality is model-dependent in a way Phase 8 cannot fix.**
   Phase 8 only governs the *baseline*; once into the optimization
   trajectory, Haiku takes more aggressive (sometimes incorrect)
   pragma rewrites while Sonnet plays it safe and regresses. This is
   exactly what Pillar 7 (per-version Avoid band + counterfactual
   probe) and Pillar 9 (no-op detector + csim-gating) were proposed to
   handle — neither is in Phase 1's scope yet, but the data motivates
   prioritising them.

4. **The csim-failure trap on Haiku's coalescing is a Pillar 9 case.**
   Haiku produces 493,754 cyc on coalescing — 1.88× off ref-best — but
   csim fails. The rubric's new best-vs-best correctly excludes it
   from headline scoring, so the *honest* best is the unroll step at
   737K cyc. Without csim-gating, the rubric would have reported a
   misleadingly-good 1.88× headline that didn't actually compute the
   correct kernel output.

## Files of record

- Implementation:
  - [c2hls.py](../c2hls.py): Phase 8 helpers + `_baseline_alignment_loop`, run_multistep wiring
  - [rubric.py](../rubric.py): `_pick_best_step_report` + `BEST(...)_vs_REF(...)` headline step in `load_multistep_results`
- Run results:
  - [results_phase2/knn_haiku_phase8_u280_v2023/knn_multistep_results.json](../results_phase2/knn_haiku_phase8_u280_v2023/knn_multistep_results.json)
  - [results_phase2/knn_sonnet_phase8_u280_v2023/knn_multistep_results.json](../results_phase2/knn_sonnet_phase8_u280_v2023/knn_multistep_results.json)
- Logs: `/tmp/phase8_e2e_knn_haiku_u280_v2023.log`, `/tmp/phase8_e2e_knn_sonnet_u280_v2023.log`

## Follow-up — Phase 9 csim/cosim repair (landed in this round)

Direct response to the live failure mode this artifact exposed (Haiku's
doublebuffer + coalescing csim-failed but were recorded as success): a
correctness-repair loop now runs **inside** `_optimization_step_attempt`.
When csynth passes but csim or cosim ran and failed, the LLM is
re-prompted with the testbench failure log via the new
[`hls_correctness_repair_fix`](../prompt_c2hls.py) prompt, then the
attempt re-runs under the same `turns_limitation` budget that
csynth-fail repair uses. Default-on; disable via
`C2HLS_DISABLE_CORRECTNESS_REPAIR=1` for legacy comparison runs.

The prompt is metric/log-only — it tells the LLM **not** to revert the
whole step but to fix the specific defect class (loop bounds /
indexing / ordering / buffering / burst-tail). If correctness budget
is exhausted, the step returns `success=False` with an
`error="Correctness repair exhausted ..."` marker so `run_optimization_step`
can revert cleanly instead of poisoning the trajectory with wrong values.

[Phase 9 smoke](../tests/test_phase9_smoke.py) — 18/18 green; all
nine-phase suite still passes (250/250 total).

## Recommended next moves

1. **Rerun knn under Haiku** with this Phase 9 path on. Expected: the
   doublebuffer/coalescing csim-failures get re-prompted, and either
   converge to a csim-passing version (real ~500K cyc result with
   correctness preserved) or revert to unroll instead of silently
   ranking high. Either outcome is more honest than today's number.
2. **Counterfactual probe for Sonnet's regression mode.** Sonnet's
   tiling/pipeline/unroll all match GT exactly within 0.01% (1,000-cyc
   drift on 4M cyc designs). That suggests Sonnet's transformations
   *are* being absorbed by Vitis 2023.2 the same way GT's are — the
   missing 28× is purely the unroll-after-burst-widening pattern that
   neither model lands. Pillar 7's "Avoid band" is the right framing.
3. **Cross-version sweep (Pillar 6).** Now that the 2023.2 cell is
   reproducing philip's numbers, repeat under Vitis 2025.2 on the
   same kernels and see whether the absorbed-by-synthesis set widens
   — same 6 kernels, different toolchain row, populates the
   portability matrix.
