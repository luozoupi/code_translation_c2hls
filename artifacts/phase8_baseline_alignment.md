# Phase 8: baseline alignment + cycle-aware rubric

_generated 2026-05-08. Two related changes: (a) opt-in alignment loop
between Phase B and Phase C; (b) `latency_cycles` ratio + score added
to [rubric.py](../rubric.py) so cross-toolchain comparisons can
separate algorithmic gap from clock/Fmax gap._

## TL;DR

- **Phase 8** adds an opt-in baseline-alignment loop. After Phase B
  succeeds, if our baseline is more than 1.20× the reference baseline
  in cycles (or any single resource is more than 2.00× over), the
  orchestrator re-translates with **metric-only** feedback — never
  the reference HLS code — and re-synths. Up to 3 attempts.
- **Rubric extension** adds `latency_cycles_ratio`, `latency_cycles_score`,
  and `clock_gap_ratio` to every `StepScore`. The headline composite
  still uses `latency_ns` (preserves comparability with prior runs),
  but the cycle-axis is now visible everywhere — surfacing
  algorithmic-vs-toolchain attribution that was previously lost in
  the wall-time number.

## Why this matters

The data motivates both changes:

1. **knn_static**'s 72M-cycle baseline (vs philip's 1.05M reference =
   70× worse) compounded through every optimization step. The
   trajectory only recovered to 5.32M — never close to the reference's
   262K. With Phase 8 enabled, the alignment loop would have caught
   the misalignment and re-translated *before* burning multistep
   budget on a poisoned starting point.

2. **Pathfinder Run 4** (the ostensibly "best" 0.30× baseline run) at
   640K cycles is actually *better than reference doublebuffer* in
   cycles (640K vs 1.06M = 0.61×). The 46× wall-time gap to reference
   coalescing is mostly toolchain (clock 1.20× + Fmax 2.20× = 1.6×)
   plus a real 28× algorithmic gap. The rubric used to show only
   "47× wall-time slower" — now it shows "29× cycles slower, 1.6×
   clock-toolchain factor." That decomposition is what surfaces *what
   the agent actually has to fix*.

## Phase 8 — baseline alignment loop

### Wiring

The loop runs *between* Phase B (translate + synth + repair) and Phase C
(compare against gold). Sketched flow:

```
run_phase_a → run_phase_b → [Phase 8a alignment loop] → run_phase_c → multistep
```

Pseudo-code:

```python
for attempt in range(C2HLS_PHASE8_MAX_ATTEMPTS):
    gap = _compute_baseline_gap(self.synth_report, reference_report,
                                  latency_tolerance=lat_tol,
                                  resource_tolerance=res_tol)
    if gap.within_tolerance:
        return  # done — proceed to Phase C
    guidance = _render_baseline_alignment_guidance(gap, attempt)
    new_code = self.translator.retranslate_with_guidance(guidance)
    self.hls_code = new_code
    self.synthesis.synthesize_with_repair()
```

### Env flags (default off)

| Variable | Default | Purpose |
|----------|--------:|---------|
| `C2HLS_PHASE8_BASELINE_ALIGN` | `0` | Master switch. Set to `1` to enable. |
| `C2HLS_PHASE8_BASELINE_LATENCY_TOL` | `1.20` | Re-translate if `cycles_ratio` exceeds this. |
| `C2HLS_PHASE8_BASELINE_RESOURCE_TOL` | `2.00` | Re-translate if any resource ratio exceeds this. |
| `C2HLS_PHASE8_MAX_ATTEMPTS` | `3` | Cap on retranslation attempts. |

### Crucial constraint: no gold-code leak

The translator agent should still be solving the C-to-HLS problem on
its own. The alignment guidance is **metric-only**:

- `latency_cycles` ratio (ours / ref) with arrow rendering
- per-resource ratios (BRAM, DSP, FF, LUT)
- Fmax comparison
- Per-loop diagnostics from our *own* report's `feedback["scopes"]`
  (which Pillar 1's parser already attached)
- Generic "likely cause" hints (helper functions not inlined,
  conservative loop scheduling, missing `extern "C"`, AXI bundle
  mistakes)

The renderer explicitly tells the LLM:

> "Do NOT add optimization pragmas (PIPELINE / UNROLL / DATAFLOW /
> array_partition) — those belong to the multistep optimization phase
> that runs AFTER this alignment. Keep this translation conservative:
> just AXI INTERFACE pragmas + the minimal kernel structure."

That keeps the alignment scope narrow and prevents Phase 8 from
swallowing what should be Phase 5/6's work.

### Outcome surfaced in results

Every `*_multistep_results.json` now includes a `baseline_alignment`
block with attempt history (per-attempt `latency_ratio`, `resource_ratios`,
`within_tolerance`) and the final `aligned` flag.

## Rubric extension — `latency_cycles` is now first-class

[rubric.py:StepScore](../rubric.py) gains three fields:

| Field | Meaning |
|-------|---------|
| `latency_cycles_ratio` | `gen_latency_cycles / gt_latency_cycles` — pure algorithmic comparison, independent of clock/Fmax. |
| `latency_cycles_score` | 0–100 score from the cycles ratio (same shape as the existing ns-based score). |
| `clock_gap_ratio` | `latency_ratio / latency_cycles_ratio` — how much of the ns gap is *not* explained by the cycle gap. >1 means our toolchain is slower per cycle. |

Composite weights are **unchanged** — the headline `latency_score` is
still ns-based for backward comparability with prior runs. The new
fields are surfaced for visibility, not folded into the composite.

### Decomposition demo (real Run 4 data)

| Comparison | latency_ns_ratio | latency_cycles_ratio | clock_gap_ratio |
|------------|----------------:|---------------------:|----------------:|
| Run 4 vs ref **coalescing** | 46.45× | 28.96× | 1.60× |
| Run 4 vs ref **doublebuffer** | 0.97× | **0.61× (we beat ref!)** | 1.60× |

Reading row 2: against the doublebuffer reference variant, **our run
uses fewer cycles** (640K vs 1.06M = 0.61×), but the wall-time barely
beats it (0.97×) because our clock-period is 1.60× wider per cycle
(4 ns request → 5.34 ns achieved on Artix-7 vs ref's 3.33 ns request →
2.43 ns achieved on U280). That decomposition was previously hidden
behind a single "0.97× wall time" number.

## Phase 8 smoke (27/27)

| Check group | Count | Notes |
|-------------|------:|-------|
| Gap detection — matched baseline | 2 | within_tolerance=True; lat_ratio≈1.0 |
| Gap detection — 70× worse baseline (knn-static-style) | 3 | lat_ratio≈68.6×; latency_over=True |
| Gap detection — resource over-budget | 2 | BRAM 6.7× flagged correctly |
| Guidance renderer | 5 | non-empty + mentions latency multiplier + no GT-code leak + tells LLM not to add optimization pragmas + empty when aligned |
| TranslatorAgent.retranslate_with_guidance signature | 4 | reuses canonical translate template; appends BASELINE ALIGNMENT FEEDBACK; extracts cpp; logs as Phase 8 |
| Orchestrator wiring | 9 | env-flag-gated; respects all three tolerance/attempt envs; calls retranslate; re-synths after; runs before Phase C; surfaces in results |
| Disabled by default (no env flag) | 2 | enabled=False; attempts=0 |

## All-phase smokes

| Phase | Checks | Result |
|-------|-------:|:------:|
| Phase 1 | 47 | ✓ |
| Phase 2 | 50 | ✓ |
| Phase 3 | 22 | ✓ |
| Phase 4 | 21 | ✓ |
| Phase 5 | 24 | ✓ |
| Phase 6 | 18 | ✓ |
| Phase 7 | 23 | ✓ |
| **Phase 8** | **27** | **✓** |
| **Total** | **232** | **✓** |

## Files of record

- Implementation:
  - [c2hls.py](../c2hls.py): `_compute_baseline_gap`, `_render_baseline_alignment_guidance`, `TranslatorAgent.retranslate_with_guidance`, `Orchestrator._baseline_alignment_loop`, wired into `run_multistep`
  - [rubric.py](../rubric.py): `latency_cycles_ratio`, `latency_cycles_score`, `clock_gap_ratio` fields + computation
- Smoke: [tests/test_phase8_smoke.py](../tests/test_phase8_smoke.py) — 27/27
- Canonical full-metrics comparison: [pathfinder_full_metrics_comparison.md](pathfinder_full_metrics_comparison.md)

## What this enables next

Three follow-ons that benefit directly from Phase 8 + the cycle-aware rubric:

1. **Run an actual e2e with `C2HLS_PHASE8_BASELINE_ALIGN=1` on a kernel
   where the LLM previously produced a misaligned baseline** (e.g.
   knn under Haiku where baseline was 72M cycles vs ref 1.05M).
   Expected: 1–2 retranslation attempts converge to a baseline within
   tolerance, then multistep starts from a known-good foundation.
2. **Use `clock_gap_ratio` to flag when toolchain limitations dominate
   the wall-time gap** — gives the agent a clean signal to not chase
   PPA improvements that the toolchain can't deliver.
3. **Per-loop scope diagnostics in the alignment guidance.** Right now
   the guidance is whole-design metrics. The Pillar 1 `feedback["scopes"]`
   already contains per-loop II / Slack / Trip-count — surfacing those
   into the alignment retranslation prompt would let the LLM target
   *specific* loops it's falling short on.

(3) is the next-most-valuable change, similar to last round's "plumb
static_extras into compose_with_llm" recommendation. Both are
prompt-side enrichments that turn data we already have into signal the
LLM can act on.
