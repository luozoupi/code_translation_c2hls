# Flash_selected latency-opt + gemm_flatten FP-II skill

> **For agentic workers:** Implement task-by-task. Steps use checkbox syntax.

**Goal:** Chain latency-opt (and pragma_opt) after batch_parallel flash finalize; backfill gf98 flash_final; add FP serial-acc avoid skill to `gemm_flatten_v1.json`.

**Architecture:** Mirror `c2hls.py` flash-success hooks inside `FlashPipelinedBenchSession._finalize_success`. Backfill via existing `start_post_flash_latency_opt.sh --submit --source flash_final`.

**Tech Stack:** Python, pytest, existing `post_flash_latency_opt` / `post_flash_pragma_opt`.

---

### Task 1: Hook in `_finalize_success`
- [ ] After saving multistep results, call `maybe_chain_pragma_opt` then `maybe_chain_latency_opt` with `source_role="flash_final"`, using `self.bench_dir` / `self.cell_dir` / orchestrator; warn on exception.

### Task 2: Unit test
- [ ] Test that `_finalize_success` invokes both chain helpers when mocked (env not required if helpers are mocked at call site).

### Task 3: Skill JSON
- [ ] Add `hls-avoid-serial-fp-acc-under-full-k-unroll` + patch GEMM flatten success/II guards in `gemm_flatten_v1.json`; bump `skill_count` / `change_summary`.

### Task 4: Backfill gf98
- [ ] Dry-run then submit `flash_final` latency-opt on gf98 matrix (at least `chathls_kernel_2mm`; prefer all 16 cells missing flash latency_opt).
