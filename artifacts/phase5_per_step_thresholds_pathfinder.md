# Per-step regression thresholds on pathfinder (Run C)

_generated 2026-05-07. Same kernel + toolchain + Sonnet + Phase 5 flags as
Run B, but with the **new per-step `STEP_REGRESSION_THRESHOLDS` lookup**
(no global override; latency/resource limits calibrated against philip's
rodinia-hls reference per-step ratios)._

## TL;DR

**Per-step thresholds successfully unlocked the `coalescing` step** that
all prior runs failed to land — but loose tiling/pipeline thresholds
(designed to tolerate enabling regressions) let the post-coalescing
trajectory drift worse than coalescing's mid-trajectory peak.

| Run | Best mid-trajectory | Final lat_ns | Δ baseline (final) | Coalescing? |
|-----|-------------------:|-------------:|-------------------:|:-----------:|
| Run 1 (Haiku, no Phase 5) | 12.19M | 12.19M | 0.96x (4%) | ❌ |
| Run 2 (Sonnet, no Phase 5) | 8.69M | 8.69M | 0.77x (23%) | ❌ |
| Run 3 (Sonnet, Phase 5, strict 1.10x) | 8.69M | 8.69M | 0.77x (23%) | ❌ |
| Run 4 (Sonnet, Phase 5, global 1.25x) | 3.42M | **3.42M** | **0.30x (70%)** | ❌ (absorbed) |
| **Run C (Sonnet, Phase 5, per-step)** | **6.06M (after coalescing)** | 6.39M | 0.57x (43%) | **✓** |

## Per-step trajectory (Run C)

| Step | Effect | lat_ns | BRAM | FF | LUT | Notes |
|------|--------|-------:|-----:|---:|---:|-------|
| baseline | — | 11 290 000 | 41 | 9 077 | 5 658 | (Phase B output) |
| **coalescing** | **improved** | **6 056 000** | 147 | 10 412 | 5 747 | **47% improvement, the big win** |
| doublebuffer | absorbed | 6 061 000 | 151 | 12 034 | 6 264 | matched coalescing |
| tiling | accepted (drift) | 14 645 000 | 153 | 5 993 | 7 755 | latency 2.4x worse, but tiling threshold is 5x — accepted |
| pipeline | absorbed-on-drift | 14 645 000 | 163 | 6 968 | 8 076 | matched tiling |
| unroll | partially recovered | **6 391 000** | 183 | 8 718 | 8 128 | recovered 56% but not back to coalescing's peak |

## What happened

Phase 5b's pre-populated GT cache fired correctly (5 entries: 4.22M / 6.32M /
4.25M / 2.11M / 0.04M cycles for baseline / tiling / unroll / doublebuffer /
coalescing). Phase 5a's LLM-aided retry fired on coalescing's 1.50x
latency-regression (`limit 1.20x for step 'coalescing'` per the new
per-step threshold) — the LLM-aided 3rd attempt or the alignment-aware
keep let coalescing land at 6.056M ns. **Per-step thresholds did exactly
what they were designed to do for coalescing.**

The trouble started later. With the per-step latency tolerance for
`tiling` set to **5.0x** (intentionally loose so tiling-as-enabler can
land on kernels where it regresses 4x like knn), Sonnet's tiling rewrite
on top of the doublebuffer'd code made things 2.4x slower — *but stayed
within the 5x tolerance and was accepted*. Pipeline didn't fix it (no-op
on top of tiling). Unroll then partially recovered but didn't return all
the way to coalescing's peak.

## The cause + the obvious Phase 6 fix

The trajectory peaked at **6.056M ns (after step 1, coalescing)** and
the orchestrator returned the *final* state (6.391M ns) instead of the
peak. The c2hls multistep loop today writes `final_report = self.synth_report`
at the end — there's no "best-so-far" tracking that snapshots the
peak-PPA mid-trajectory and promotes it as the final answer.

This is the natural Phase 6 fix:

```python
# In run_multistep, after each accepted step:
score = quality_score(self.synth_report)
if best is None or score < best["score"]:
    best = {"score": score,
            "code": self.hls_code,
            "report": dict(self.synth_report),
            "step_index": idx}
# At end of multistep:
results["final_report"] = best["report"]
results["final_hls_code"] = best["code"]
results["best_step"] = best["step_index"]
```

Concretely on Run C: the orchestrator would have returned the
post-coalescing state (6.056M ns) instead of the post-unroll state
(6.391M ns). Combined with what we already have, that's a **5%
improvement** on Run C without any rerun.

More importantly, this reframes loose per-step thresholds from "danger,
let bad steps in" to "exploration mode" — every step gets a chance to
land, but the orchestrator only commits the best one observed.

## Comparison to Run B (the previous best, 3.42M ns)

Run B used a *global* 1.25x threshold that was loose enough to let
unroll's resource growth (LUT 1.41x) land *but tight enough* to prevent
tiling/pipeline drift. The per-step approach loosened tiling specifically
to 5x — which is right for kernels like knn (where tiling reference
ratio is 4.08x) but too loose for pathfinder (where tiling reference
ratio is 1.50x).

**The takeaway**: per-step *thresholds* alone aren't enough — they need
either:
- (a) **kernel-aware tightening** of tiling/doublebuffer thresholds based
  on the GT trajectory's actual ratios at each step, OR
- (b) **best-so-far tracking** so loose thresholds become exploration not
  drift.

Option (b) is much simpler and more general. Phase 6 candidate.

## Phase 5 features observed firing in Run C (production)

| Feature | Fired? | Notes |
|---------|:------:|-------|
| 5b GT pre-pop (5 entries upfront) | ✓ | All 5 pre-cached: 4.22M / 6.32M / 4.25M / 2.11M / 0.04M cycles |
| 5a LLM-aided 3rd retry on regression | ✓ | Fired on coalescing |
| 5b skill-template prompt injection | ✓ | Every step |
| Per-step regression threshold | ✓ | Fired with `latency_ns regressed 1.50x (limit 1.20x for step 'coalescing')` |
| Phase 9 no-op trap | — | Not triggered this run |
| Phase 1 regression-revert | ✓ | Coalescing reverted twice; recovered via LLM-aided retry |
| Phase 3 GT-aware-revert (alignment) | ✓ | GT cache populated by 5b made this finally usable |

## Cross-run summary (5 runs total on pathfinder)

| Strategy combo | Final lat_ns | Δ baseline | BRAM | FF | LUT | Steps survived |
|---------------|-------------:|-----------:|-----:|---:|---:|:--------------:|
| Haiku, no Phase 5, 1.10x | 12.19M | 0.96x | 167 | 16 446 | 14 515 | 2/5 |
| Sonnet, no Phase 5, 1.10x | 8.69M | 0.77x | 39 | 7 938 | 6 704 | 3/5 |
| Sonnet, Phase 5, 1.10x | 8.69M | 0.77x | 159 | 8 746 | 7 002 | 4/5 |
| Sonnet, Phase 5, **1.25x global** | **3.42M** | **0.30x** | 191 | 10 108 | 8 401 | 5/5 |
| Sonnet, Phase 5, **per-step** | 6.39M (mid: 6.06M) | 0.57x | 183 | 8 718 | 8 128 | 5/5 |

## Smoke totals

| Phase | Checks | Result |
|-------|-------:|:------:|
| Phase 1 | 47 | ✓ |
| Phase 2 | 50 | ✓ |
| Phase 3 | 22 | ✓ |
| Phase 4 | 21 | ✓ |
| Phase 5 (incl. per-step thresholds) | 24 | ✓ |
| **Total** | **164** | **✓** |

## Files of record

- Run C result: [results_phase2/pathfinder_dynamic_sonnet_phase5_perstep/](../results_phase2/pathfinder_dynamic_sonnet_phase5_perstep/)
- Run C log: `/tmp/phase5_e2e_pathfinder_sonnet_perstep.log`
- Per-step thresholds dict: [c2hls.py:STEP_REGRESSION_THRESHOLDS](../c2hls.py)
- Phase 5 smoke (24/24): [phase5_smoke_20260507_203515.md](phase5_smoke_20260507_203515.md)
