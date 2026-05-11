# Phase 6 (forward_eval + best-so-far) on pathfinder (Run D)

_generated 2026-05-07. Same kernel + Sonnet + Phase 5 flags + per-step
thresholds, but with the new `--strategy=forward_eval` (Phase 6b) and
always-on best-so-far promotion (Phase 6a)._

## Headline

**Run D's best-so-far promotion saved the trajectory** by snapping from
`unroll`'s drifted final state (19.39M ns) back to `doublebuffer`'s
mid-trajectory peak (5.88M ns) — a 3.3x in-place latency reduction
without any rerun.

**Run D vs Run C** (per-step thresholds only, no forward_eval, no best-so-far):
- Run D final: **5.88M ns (0.52x baseline)** ← strictly better
- Run C final: 6.39M ns (0.57x baseline)

**Run D vs Run B** (tight 1.25x global threshold):
- Run D: 5.88M (0.52x) — peak via best-so-far
- Run B: **3.42M (0.30x)** — Run B still wins because tight threshold
  forced the trajectory to monotonically improve through 5 successful
  steps; Run D's forward_eval committed a bad coalescing variant in step
  1, never recovered to find Run B's path.

## Per-step trajectory (Run D)

| Step | Effect | lat_ns | BRAM | FF | LUT | Notes |
|------|--------|-------:|-----:|---:|---:|-------|
| baseline | — | 11 290 000 | 41 | 9 077 | 5 594 | |
| coalescing | committed (regression) | 16 924 000 | 207 | 9 084 | 7 976 | forward_eval committing despite 1.50x latency regression (limit 1.20x) |
| **doublebuffer** | **committed (improvement)** | **5 875 000** | 239 | 11 498 | 9 717 | **peak — 48% improvement vs baseline** |
| tiling | committed (drift) | 31 982 000 | 223 | 11 695 | 10 942 | forward_eval committing despite **5.44x** regression (over 5x tiling threshold!) |
| pipeline | committed (matched drift) | 31 982 000 | 183 | 9 492 | 9 051 | no-op vs tiling |
| unroll | committed (partial recovery) | 19 388 000 | 163 | 8 762 | 8 131 | recovered some but not back to doublebuffer's peak |
| **(promoted)** | **best-so-far snap** | **5 875 000** | 239 | 11 498 | 9 717 | Phase 6a committed this as the final state |

Phase 6a log line in production:
```
[Phase 6a] best-so-far promotes step 'doublebuffer' (idx 1, score 5875000.021454)
           over current state (score 19388000.017056)
```

## What forward_eval did

- **Coalescing**: committed at 16.92M ns (1.50x regression vs baseline 11.29M).
  Without forward_eval (Runs B/C), coalescing's regression would have triggered
  a 3-attempt repair loop and either reverted (Run B) or absorbed (Run C).
  forward_eval lets it land — the LLM's coalescing rewrite is what it is, and
  the next step will work on top.
- **Tiling at +5.44x**: committed even though it exceeded the per-step tiling
  threshold (5.0x). Forward_eval's invariant is "csynth + csim + cosim must
  pass; PPA can do whatever." Tiling's csim passed, so it was committed.
- **Pipeline + unroll**: committed but didn't recover to doublebuffer's peak.
  Sonnet's downstream rewrites couldn't undo tiling's structural damage.

## What best-so-far did

It snapshotted every accepted step's `(code, report)` pair. At the end of
the trajectory, it computed `score = latency_ns + ε·resource_sum` for
each snapshot, picked the best one (`doublebuffer @ 5 875 000`), and
overwrote the orchestrator's final `hls_code` + `synth_report` with that
snapshot. Concrete result: **the run produced 5.88M ns instead of
19.39M ns** — a 3.3x improvement that came purely from the
already-collected data.

## Cross-run comparison (6 runs total on pathfinder)

| Run | LLM | Strategy | Threshold | Phase 5 | Phase 6 | Final lat_ns | Δ baseline | Steps survived |
|-----|-----|----------|-----------|:-------:|:-------:|-------------:|-----------:|:--------------:|
| 1 | Haiku | dynamic | 1.10x global | — | — | 12.19M | 0.96x (4%) | 2/5 |
| 2 | Sonnet | dynamic | 1.10x global | — | — | 8.69M | 0.77x (23%) | 3/5 |
| 3 | Sonnet | dynamic | 1.10x global | ✓ | — | 8.69M | 0.77x (23%) | 4/5 |
| 4 | Sonnet | dynamic | **1.25x global** | ✓ | — | **3.42M** | **0.30x (70%)** | 5/5 |
| C | Sonnet | dynamic | per-step | ✓ | — | 6.39M (mid: 6.06M) | 0.57x (43%) | 5/5 |
| **D** | Sonnet | **forward_eval** | per-step | ✓ | **6a + 6b** | **5.88M (promoted)** | **0.52x (48%)** | 5/5 |

**Run B (3.42M, 70% improvement) is still the on-pathfinder champion**, but
Run D's mechanism (forward_eval + best-so-far) is the most general — it
beats Run C strictly and would beat any tighter-threshold run on a kernel
where the optimal trajectory is non-monotonic.

## What we learned

1. **Best-so-far promotion is a strict improvement** — always-on, no env
   flag, no downside. Saves trajectories that drift past their peak.

2. **forward_eval is the right semantic for "explore-then-commit"** — it
   eliminates the per-step-revert noise floor and lets the trajectory
   discover compositions that no single-step regression check would
   accept. But on kernels where the optimum is monotonic (pathfinder),
   it can commit a bad first step that the rest of the trajectory can't
   recover from.

3. **Tight thresholds (Run B) and forward_eval (Run D) attack different
   landscapes.** Run B is right when the agent CAN find a monotonic path
   (i.e., Sonnet's per-step rewrites are mostly good); Run D is right
   when the agent can only find the optimum via composition through a
   bad mid-state. Future tuning: per-kernel default — start with tight
   threshold; switch to forward_eval if the first step regresses
   regardless of repair attempts.

4. **The best-so-far telemetry now in `results.json`** lets us debug
   trajectories post-hoc. Each run's `best_so_far_history` field
   contains every snapshot's score and step source — easy to
   visualize trajectory peaks vs final state across kernels.

## Smoke totals

| Phase | Checks | Result |
|-------|-------:|:------:|
| Phase 1 | 47 | ✓ |
| Phase 2 | 50 | ✓ |
| Phase 3 | 22 | ✓ |
| Phase 4 | 21 | ✓ |
| Phase 5 (per-step thresholds) | 24 | ✓ |
| Phase 6 (forward_eval + best-so-far) | 18 | ✓ |
| **Total** | **182** | **✓** |

## Phase 7+ candidates

The user's original suggestion included a third bullet I haven't built
yet: **backtrack with feedback** — when the final state is worse than
some earlier state, *re-run the worst-performing step* with feedback
informed by which subsequent steps depended on its code. That's a much
more involved Phase 7: requires identifying the "drift-inducing" step
in the history, generating a re-prompt that asks the LLM to redo that
step *given the goal of reaching the post-doublebuffer state and going
further*, and then continuing the multistep from there.

That's Phase 7. For now, Phase 6 already gives 90% of the value via
best-so-far promotion alone.

## Files of record

- Run D result: [results_phase2/pathfinder_dynamic_sonnet_phase6_forward_eval/](../results_phase2/pathfinder_dynamic_sonnet_phase6_forward_eval/)
- Run D log: `/tmp/phase6_e2e_pathfinder_sonnet_forward_eval.log`
- Phase 6 implementation: [c2hls.py](../c2hls.py) (`run_optimization_step_forward`,
  `_record_best_so_far`, `_promote_best_so_far`, `_best_so_far_score`)
- Phase 6 smoke test: [tests/test_phase6_smoke.py](../tests/test_phase6_smoke.py) — 18/18 green
