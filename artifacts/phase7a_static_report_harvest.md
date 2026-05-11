# Phase 7a: static report harvest (burst.xml, fe/be_messages.xml, design_size)

_generated 2026-05-08. Three new parsers added to
[hls_feedback.py](../hls_feedback.py); attached to every synth output
under `report["feedback"]["static_extras"]` when `work_dir` is provided._

## Why this matters: what we couldn't see before

After every Vitis HLS `csynth_design`, four reports get written that
we've been ignoring:

| File | What it tells us |
|------|------------------|
| `<wd>/hls_proj/sol1/.autopilot/db/burst.xml` | Per-AXI-access burst inference outcome — passed / widened (32→512-bit) / **failed** |
| `<wd>/hls_proj/sol1/.autopilot/db/fe_messages.xml` | Front-end (clang) diagnostics — including **`PRAGMA_INVALID`** (silently rejected pragmas) |
| `<wd>/hls_proj/sol1/.autopilot/db/be_messages.xml` | Back-end (scheduler) diagnostics — pipeline / dataflow / dependence rejections |
| `<wd>/hls_proj/sol1/syn/report/csynth_design_size.rpt` | Per-phase instruction-count growth (compile → unroll → array-partition → performance → HW) |

The **single most important new signal** is `PRAGMA_INVALID` — when the
LLM emits a pragma but Vitis silently ignores it, the orchestrator
currently has no way to know. Now we do.

## What's wired

- **`parse_burst_info(work_dir)`** — returns `passed/widened/failed/summary` lists with bundle, var, direction, length, width, source line.
- **`parse_diagnostic_messages(work_dir)`** — returns counts of warnings/errors/info plus a focused `rejected_pragmas` list with msg_id, severity, source location, body.
- **`parse_design_size_report(work_dir)`** — returns per-phase + per-step instruction counts plus `compile_to_hw_growth` and `max_phase_growth` ratios.
- **`attach_feedback(report, work_dir=...)`** — automatic invocation. Every `run_hls_synthesis` output now carries `report["feedback"]["static_extras"]`.
- **`render_static_extras_for_prompt(extras)`** — compact human-readable block for FeedbackAgent's `compose_with_llm` to feed the LLM.
- ElementTree quirk handled: Vitis emits both XML files with unbound namespace prefixes (`<VitisHLS:BurstInfo>`, `<xilinx:hls_fe_msgs>`); we strip them before parsing.

## Demonstration on a real recent synth

From `/tmp/hls_synth_j91eflub` (a recent coalescing-step output on pathfinder):

```
AXI burst inference: passed=6, widened=3, failed=0
  WIDENED: Jout on bundle gmem (write) — Sequential write of 1024 x 32bit words has been widened by 16: 64 x 512bit words
  WIDENED: J on bundle gmem (read) — Sequential read of 1048576 x 32bit words has been widened by 16: 65536 x 512bit words
  WIDENED: J on bundle gmem (read) — Sequential read of 1024 x 32bit words has been widened by 16: 64 x 512bit words
Pragmas silently rejected by Vitis: 32
  207-6973 [WARNING] at ./support/common/mars_wide_bus.h:26:24 — the 'self/all' option to 'Inline' pragma is not supported and will be ignored
  207-6973 [WARNING] at ./support/common/mars_wide_bus.h:46:24 — ...
  ...
Design size growth across compilation phases: 1.00x end-to-end
```

**Concrete diagnostic this gives us**: in Run D's coalescing step (which regressed 1.50x latency), the *bursts widened correctly* (3 widened to 512-bit) — so the wide-bus mechanism worked. The LLM could see this and stop blaming `coalescing` for the regression. Instead, the cause is somewhere else (probably control overhead from the new load/compute/store split). That's a surgical, kernel-level diagnostic that the deterministic regression-template can't produce.

Separately: 32 pragmas in the `mars_wide_bus.h` support header are silently rejected (the `'self/all'` option to `Inline` was deprecated in 2025.2). The agent now knows this is a *toolchain* issue, not a kernel-side issue — and won't waste retries trying to "fix" `mars_wide_bus.h`.

## Smoke totals (with Phase 7a wiring)

| Phase | Checks | Result |
|-------|-------:|:------:|
| Phase 1 | 47 | ✓ |
| Phase 2 | 50 | ✓ |
| Phase 3 | 22 | ✓ |
| Phase 4 | 21 | ✓ |
| Phase 5 | 24 | ✓ |
| Phase 6 | 18 | ✓ |
| **Phase 7** | **23** | **✓** |
| **Total** | **205** | **✓** |

Phase 7 smoke covers: synthetic XML fixtures for all three parsers, the namespace-prefix workaround, attach_feedback's `work_dir=...` path, `attach_feedback()` without a work_dir leaving `static_extras` absent, the prompt-renderer producing actionable text, and an *optional* real-on-disk sample test that auto-skips when no `/tmp/hls_synth_*` exists.

## What it cost

- Code: ~280 new lines in [hls_feedback.py](../hls_feedback.py); ~10-line touch in [hls_eval.py](../hls_eval.py) (`run_hls_synthesis` now passes `work_dir=` to `attach_feedback`).
- Synth-time overhead: **zero**. All four files are already on disk after every csynth — we're just reading them.
- Storage in saved results: ~3-15KB extra JSON per step in `*_multistep_results.json` (negligible).
- API/LLM overhead: zero by default. The new signals only enter the LLM's context if FeedbackAgent's `compose_with_llm` is invoked (Phase 5a).

## What's next

Three obvious follow-ons, in priority order:

1. **Plumb `static_extras` into `compose_with_llm`'s `bottleneck_record`** (Phase 7a-tail, ~30 min). Currently `static_extras` lands in the report dict but the FeedbackAgent's LLM-aided retry doesn't consciously pull it. Patching that means *every* regression retry now includes the burst/pragma diagnostics. Highest ROI of any small change in this round.
2. **Phase 7b — hw_emu profiling** (your earlier suggestion): bump the auto-injected `xrt.ini` to enable `[Profile] kernel_profile=true`, `data_transfer_trace=fine`, `stall_trace=all`. Parse the resulting `profile_summary.csv` for stall classes (memory-port stall, dataflow-channel-empty, etc.). This is the dynamic-execution equivalent of NCU stall reasons. Adds ~20% to hw_emu wall time.
3. **Per-step `static_extras` diff in the trajectory** (Phase 7c): when step N regresses against step N-1, automatically diff their `static_extras` fields. New `BURST_VERBOSE_FAILED` records or new `PRAGMA_INVALID` messages between the two runs *exactly* identify the drift cause. Feed the diff into FeedbackAgent's bottleneck_record.

Recommendation: do (1) next — it's the smallest change with the highest immediate impact on retry quality. (2) and (3) can follow.
