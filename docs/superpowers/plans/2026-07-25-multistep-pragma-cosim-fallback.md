# Multistep pragma / cosim fallback Implementation Plan

> **For agentic workers:** Implement task-by-task. Steps use checkbox syntax.

**Goal:** Prompt guards for one-line pragmas / no INTERFACE `data_width`; top-3 cosim fallback over pre+post lat-opt; latency_table + cosim_fallback in results.

**Architecture:** Prompt strings in `prompt_c2hls.py` + `post_flash_latency_opt.py`; ranking/fallback helpers + `_run_cosim` rewrite in `multistep_batch_parallel_bench.py`; stash pre/post in `_maybe_run_latency_opt` and emit in `_finalize_success`.

**Tech Stack:** Python 3, existing batch_parallel multistep session, pytest.

---

### Task 1: Prompts
- [ ] Update `_TOP_INTERFACE_REQUIREMENT`, `Instruction_c2hls`, `Instruction_c2hls_multistep`
- [ ] Update `_PLAN_SYSTEM` / `_MODIFY_SYSTEM` in `post_flash_latency_opt.py`

### Task 2: Ranking + cosim fallback
- [ ] Add `rank_cosim_candidates` / attempt loop in `multistep_batch_parallel_bench.py`
- [ ] Cap at 3; soft/hard via `C2HLS_COSIM_REQUIRED` (no repair after top-3 exhaust)

### Task 3: Latency records
- [ ] Stash pre/post in `_maybe_run_latency_opt`; `latency_table` + `cosim_fallback` on finalize

### Task 4: Tests
- [ ] Unit-test ranking order and top-3 selection without Vitis
