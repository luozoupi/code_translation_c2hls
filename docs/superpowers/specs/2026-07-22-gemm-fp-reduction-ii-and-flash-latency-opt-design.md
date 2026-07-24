# Design: GEMM FP-reduction II avoid + flash_selected latency-opt

**Date:** 2026-07-22  
**Status:** approved (skill target: `gemm_flatten_v1.json`; flash hook included)  
**Evidence:** `chathls_kernel_2mm` flash selected ~826569 cycles; `add_gemm` Final II=250 on trip≈3200 from serial FP `sum` under full `k` UNROLL with shared 1×dmul/1×dadd. Campaign used `skills_ii_target_miss_solutions_added(90skills)_gemm_flatten_v1.json`. Latency-opt never ran on flash path in batch_parallel (only after successful dataflow).

## Goals

1. Teach the active packaged skill pack that **full `k` UNROLL with a serial FP accumulator is not success** when achieved II stays huge.
2. Run **latency-opt on flash_selected** in the ChatHLS / batch_parallel flash finalize path (same as dataflow chain), so benches that fail dataflow still get latency-opt.

In scope: backfill gf98 flash_final latency-opt (especially `chathls_kernel_2mm`) via `run_post_flash_latency_opt.py --source flash_final`.

---

## Part A — Skill pack (`gemm_flatten_v1.json` only)

**File:** `hls_full_optimization_skills_schema_1_1_package/skills_ii_target_miss_solutions_added(90skills)_gemm_flatten_v1.json`  
Do **not** edit `skills/skills.json` or the base `skills_ii_target_miss_solutions_added(90skills).json` in this change.

### A1. Add avoid skill

**id:** `hls-avoid-serial-fp-acc-under-full-k-unroll`  
**kind:** `avoid_rule`  
**confidence:** `avoid`  
**origin:** `u280_chathls_kernel_2mm_ii250_20260722`

**pattern:** Static-bound GEMM/matmul (or 2mm/3mm phase) on locals with `#pragma HLS PIPELINE` on `(i,j)` (often + `LOOP_FLATTEN`) and `#pragma HLS UNROLL` on reduction `k`, but csynth shows **Final II ≫ 1** (e.g. tens–hundreds) on that compute loop due to a **loop-carried FP dependence on a single accumulator** and/or **resource-limited** dmul/dadd (often 1 instance), so latency ≈ trip × II remains catastrophic.

**strategy:** Do not treat “`k` fully unrolled” or “outer flattened” as optimized if achieved II is huge. Rewrite the reduction: lane/partial sums or a tree (see `hls-unroll-reduction-partial-sums` / `ii-reduction-lane-partial-tree`), ensure enough FP operators for the chosen parallelism, or fall back to **pipeline-on-`k`** with modest outer/`j` unroll (ChatHLS-style). Re-synth until Final II is near the legal target (prefer II≈1 when resources allow).

**required_steps:**
1. Detect full `k` UNROLL under `(i,j)` pipeline with serial `sum += a*b` (float/double).
2. Read csynth: Final II, dependence on accumulator, Binding/operation sharing for dmul/dadd.
3. If Final II ≫ 1 from FP reduction / resource limit, reject the design as optimized.
4. Route to partial-sum/tree reduction + matching resources, or pipeline-on-`k` fallback.
5. Re-synth and confirm Final II improved; do not stop at flatten-only success.

**guards:**
- Still prefer flatten of sequential outer `i` when that is the bug (`hls-avoid-outer-sequential-gemm-after-inner-pipeline`).
- Preserve numerical tolerance expectations when reassociating FP reductions.
- Device util ≤ 100% after allocating more operators / wider trees.

**bottleneck_kinds:** `ii_target_miss`, `latency_high`, `pipeline_body_latency`, `resource_limited`, `compute_bound`  
**tags:** `avoid`, `gemm`, `2mm`, `3mm`, `fp-reduction`, `unroll`, `ii`, `flash-mandatory`, `chathls-ready`  
**template:** short BAD (serial `sum` + full `k` UNROLL under `(i,j)` PIPELINE) vs GOOD sketch (partial/`tree` or pipeline-on-`k`).

### A2. Patch existing positive skills

Update these entries in the same JSON:

| id | change |
|----|--------|
| `hls-gemm-static-nest-flatten-and-k-partition` | Success check: achieved **Final II ≈ 1** (or best legal), not only “outer not sequential”. Guard: full `k` UNROLL requires parallel reduction structure **or** enough FP units — serial accumulator alone is insufficient. Strategy/required_steps: if II stays high after flatten+unroll, apply A1 avoid / partial sums / pipeline-on-`k`. Soften template comment to show serial `sum` as **illegal when II blows up**. |
| `hls-pipeline-hot-loop-achieve-ii` | Same II success criterion for GEMM path; link A1 when FP reduction blocks II. |
| `hls-pipeline-unroll-small-inner-loops` | Guard: for FP reductions under outer PIPELINE, do not complete-unroll `k` with a single accumulator; use tree/partials or pipeline-on-`k`. Adjust GEMM template comment accordingly. |
| `hls-avoid-outer-sequential-gemm-after-inner-pipeline` | After routing to flatten+k-unroll, note: if Final II ≫ 1 from serial FP acc, continue with A1 (do not stop). |
| `hls-avoid-pipeline-innermost-only-goal-on-gemm` | Clarify: total cycles require both flatten **and** low achieved II on the flattened region. |

### A3. Metadata

- Append bullets to `change_summary` describing A1–A2.
- Set `skill_count` to `len(skills)` after the add (currently 98 → 99).
- Update `saved_at`.
- No change required to default `accepted_skill_counts` when campaigns set `C2HLS_PACKAGED_SKILLS_JSON` (env override accepts any count). Optionally extend `accepted_skill_counts` in `tier_a_flash_lib.py` if default path ever points at this file without override — only if tests require it.

---

## Part B — Flash_selected latency-opt

### Problem

`maybe_chain_latency_opt(..., source_role="flash_final")` exists in `c2hls.py` multistep success, and dataflow success chains in `post_flash_dataflow.py`. ChatHLS batch_parallel flash uses `scripts/pc2/flash_pipelined_bench.py` → `_finalize_success`, which **never** calls the chain. Result in gf98: only `*_dataflow_latency_opt.*`; flash-only / dataflow-fail benches (e.g. `kernel_2mm`) get no latency-opt.

### Fix

In `FlashPipelinedBenchSession._finalize_success`, after saving multistep/selected artifacts:

1. Call `maybe_chain_pragma_opt(..., source_role="flash_final", skip_existing=True)` (mirror `c2hls.py`).
2. Call `maybe_chain_latency_opt(..., source_role="flash_final", skip_existing=True)`.

Use existing env gates: `C2HLS_POST_FLASH_LATENCY_OPT`, `C2HLS_LATENCY_OPT_CHAIN_FLASH` (launcher already sets both to 1 with `--latency-opt`). Swallow exceptions with warnings (same as `c2hls.py`).

Source kernel remains `{bench}_selected.cpp` / flash base via `resolve_latency_source_kernel`. Promotion behavior unchanged (`promote_latency_opt_as_selected`).

### Tests

- Unit/integration: finalize success path invokes latency_opt when env enabled (mock `maybe_chain_latency_opt` or lightweight env+spy).
- Existing `tests/test_post_flash_latency_opt.py` remain green.
- Skill JSON: loadable JSON; new skill id present; `skill_count` matches `len(skills)`.

### Docs

- Short note in latency-opt design or README that batch_parallel flash finalize chains latency-opt.
- Do not invent a second skill pack path.

---

## Non-goals

- Editing base `90skills.json` or repo-root `skills/skills.json`.
- Automatically re-running latency-opt on completed gf98 cells (manual/scripted follow-up).
- Changing ChatHLS-ACL-26 itself.

## Success criteria

1. Packaged `gemm_flatten_v1.json` contains the new avoid skill and patched GEMM flatten success criteria around Final II / serial FP accumulator.
2. With `C2HLS_POST_FLASH_LATENCY_OPT=1` and `C2HLS_LATENCY_OPT_CHAIN_FLASH=1`, a flash finalize success produces `{bench}_latency_opt_*` (or skip-existing no-op) without requiring dataflow success.
3. Tests for chain hook and skill metadata pass.
