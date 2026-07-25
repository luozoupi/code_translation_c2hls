# Multistep pragma rules, cosim top-3 fallback, latency records

**Date:** 2026-07-25  
**Status:** Approved

## Goals

1. Teach agents to keep every `#pragma HLS …` on **one line** (no `\` continuations).
2. Ban **`data_width=`** on `#pragma HLS INTERFACE` (use `max_widen_bitwidth` when widening).
3. Final cosim: rank successful **pre** and **post** lat-opt kernels by csynth latency; try **best → next1 → next2**; first PASS becomes `selected`; if all three fail → cosim failure (`C2HLS_COSIM_REQUIRED` soft/hard unchanged).
4. Record clear **pre / post lat-opt csynth latency** per phase in `*_multistep_results.json`.

## Cosim candidate pool

For each successful phase (`phase_b` + opt steps), include when files/reports exist:

- `pre_lat_opt`: `{bench}_multistep_{phase}.cpp` + `_report.json`
- `post_lat_opt`: `{bench}_multistep_{phase}_latency_opt.cpp` + successful lat-opt result/report

Sort ascending by `latency_cycles`; ties: later phase, then post over pre. Attempt at most **3**.

## Recording

Per phase in results / `latency_table`:

- `pre_lat_opt_csynth_latency`
- `post_lat_opt_csynth_latency` (null if lat-opt skipped/failed)
- `lat_opt_ran`, `lat_opt_improved`

Plus `cosim_fallback`: ranked candidates, attempts, pass/fail, selected id.
