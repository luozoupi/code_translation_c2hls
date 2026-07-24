# Design: HLSFactory + MachSuite DeepSeek U280 Dual-Track (c2hls RAG2+skills ∥ ChatHLS hybrid)

**Date:** 2026-07-18  
**Status:** approved  
**Plan:** `docs/superpowers/plans/2026-07-18-hlsfactory-machsuite-deepseek-u280-dual-track.md`  
**Approach:** A — dual-track adapters  
**Goal:** Apples-to-apples DeepSeek + U280 (3.33 ns) comparison on the **same 46 kernels** from `c2hls/benchmarks/{hlsfactory_*,machsuite_*}`:
1. **c2hls** flash→dataflow with **RAG2+skills**
2. **ChatHLS** hybrid (HLSTuner + HLSFixer + DeepSeek API)

## Problem

Prior DeepSeek U280 work compared only the 16 `chathls_*` / `benchmark_optimization` benches. User now wants:

- c2hls RAG2+skills on **all** `hlsfactory_*` (28) and `machsuite_*` (18) under `c2hls/benchmarks`
- ChatHLS hybrid DeepSeek on the **same kernels** (not the existing 16 ChatHLS-only suite)

ChatHLS hybrid today only reads `benchmark/benchmark_optimization/<name>/{kernel}.cpp` + `kernel_info.txt` + `run_hls.tcl`. It cannot consume c2hls bench dirs as-is.

## Decisions (locked)

| Item | Choice |
|------|--------|
| Bench set | All **46**: 28 `hlsfactory_*` + 18 `machsuite_*` from `c2hls/benchmarks/` |
| ChatHLS naming | **Keep c2hls prefixes** (`hlsfactory_atax`, `machsuite_aes_table`, …) under `benchmark_optimization/` |
| Port depth | **Full auto**: generate labeled kernel, `kernel_info.txt`, `run_hls.tcl` |
| Kernel dtype / semantics | **Preserve c2hls source** (e.g. `double`, tops from `metadata.json`) — do **not** rewrite to ChatHLS `ap_fixed` PolyBench style |
| c2hls stack | DeepSeek (`deepseek-chat`) + **RAG2+skills** + U280 3.33 ns |
| ChatHLS stack | Existing **hybrid** (GPU HLSFixer/HLSTuner + DeepSeek API) + U280 3.33 ns |
| Launch order | **Port → smoke → parallel full submit** of both tracks |
| DeepSeek proxy | Shared login-node queue proxy, **`workers=1`** (same pattern as chathls DeepSeek campaigns) |
| Peak gate | Reuse Beijing peak pause for c2hls codegen; ChatHLS hybrid follows its existing API usage (no new peak logic required for v1 unless proxy is shared under load) |
| Existing 16 ChatHLS benches | **Untouched** (`atax`, `gemm`, … remain as-is) |

## Non-goals

- Porting HLSTuner/HLSFixer into c2hls
- Converting c2hls kernels to `ap_fixed` to match old ChatHLS PolyBench ports
- Overwriting or replacing the current 16 `benchmark_optimization` kernels
- Devstral runs for this campaign
- Perfect cosim parity on every MachSuite edge case in v1 (failure_policy ignore / report skips)

## Bench inventory

**HLSFactory (28):**  
`2mm`, `3mm`, `atax`, `bicg`, `cholesky`, `correlation`, `covariance`, `doitgen`, `durbin`, `fdtd-2d`, `floyd-warshall`, `gemm`, `gemver`, `gesummv`, `gramschmidt`, `heat-3d`, `jacobi-1d`, `jacobi-2d`, `lu`, `ludcmp`, `mvt`, `nussinov`, `seidel-2d`, `symm`, `syr2k`, `syrk`, `trisolv`, `trmm`  
→ dir names `hlsfactory_<name>`.

**MachSuite (18):** as in `scripts/pc2/machsuite_18_benches.txt`  
→ dir names `machsuite_<name>`.

Source of truth: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/benchmarks/<prefixed_name>/`.

## Architecture

```
c2hls/benchmarks/{hlsfactory_*,machsuite_*}
        │
        ▼
  port_c2hls_bench_to_chathls.py   (exporter)
        │
        ├─► ChatHLS-ACL-26/benchmark/benchmark_optimization/<prefixed>/
        │         {top}.cpp  kernel_info.txt  run_hls.tcl  (+ headers if needed)
        │
        └─► (c2hls track uses original benchmarks/ + ready corpora as today)

login DeepSeek queue proxy (workers=1)
        │
        ├─► c2hls batch_parallel RAG2+skills (machsuite + hlsfactory campaigns)
        │         drain ──HTTP──► proxy
        │         Vitis on normal nodes
        │
        └─► ChatHLS hybrid batch parallel (46-bench list)
                  GPU HF queue (HLSFixer/Tuner)
                  CPU array ──DeepSeek──► proxy
```

## Component design

### 1. ChatHLS port exporter (`c2hls`)

**New script** (name TBD): `scripts/pc2/export_c2hls_bench_to_chathls.py`

For each bench dir under `benchmarks/{hlsfactory_*,machsuite_*}`:

1. Read `metadata.json` → `hls_top` / `kernel_top`, baseline file (`hls_baseline.cpp`), headers.
2. Build a **single translation unit** suitable for ChatHLS:
   - Prefer inlining / `#include` of required headers into one `.cpp` named after the ChatHLS convention: **`<short_or_top>.cpp`** where the file basename matches what `run_hls.tcl` `add_files` / `set_top` expect.
   - **Top function name:** keep c2hls HLS top (e.g. `kernel_atax`) unless ChatHLS hybrid hard-requires filename==top; if so, document mapping in a sidecar `port_manifest.json`.
3. **Loop label injection:** scan the top function body; assign `L1:`, `L2:`, … to each `for`/`while` that is a candidate HLS loop (skip trivial one-liners only if necessary). Labels must be stable across re-exports (deterministic order).
4. Emit `kernel_info.txt`:
   - Line 1: top function name
   - Subsequent lines: `L#,loop,<line>` and array entries derived from parameters / local arrays (best-effort AST or regex+line numbers). Quality bar: HLSTuner can address labeled loops; incomplete array rows are acceptable if loops are complete.
5. Emit `run_hls.tcl` templated like ChatHLS benches, with placeholders patched by `prepare_u280_bench.sh` to U280 / 3.33 ns (default template may still say Zynq/10 ns; prepare step rewrites).
6. Write `port_manifest.json` per bench: source path, top, label count, warnings.

**Output root:**  
`/scratch/hpc-prf-llmfpga/asa582/projects/test-chathls/ChatHLS-ACL-26/benchmark/benchmark_optimization/<prefixed_name>/`

**Idempotent:** re-run overwrites generated files; never deletes unrelated existing 16 benches.

**Validation gate before full hybrid:**
- Export all 46
- Unit test: label count ≥ 1 for benches with loops; TCL parses; top symbol exists in cpp
- Smoke: ChatHLS hybrid on **2** benches (`hlsfactory_atax`, `machsuite_gemm_ncubed`) must start csynth

### 2. c2hls RAG2+skills DeepSeek U280 campaigns

Reuse patterns from `start_chathls_deepseek_one.sh --flavor rag2_skills` + external LLM proxy.

**Two configs** (or one combined — prefer **two** for job_prefix / watcher clarity):

| Campaign | Config base | Workflow | Nodes guidance |
|----------|-------------|----------|----------------|
| MachSuite 18 | clone `batch_parallel_machsuite_flash_dataflow.json` | `tier_b_flash` + streaming dataflow watcher | combined-HLS style or existing machsuite layout; `model=deepseek-chat`; `gpu_policy=always_on`; external_llm |
| HLSFactory 28 | clone `batch_parallel_full_aav_n.json` | flash workflow with post flash→dataflow if available; else flash+cosim then export | same DeepSeek/U280/RAG2 env |

**New starters** (names TBD):
- `start_machsuite_deepseek_rag2_skills_u280.sh`
- `start_hlsfactory_deepseek_rag2_skills_u280.sh`

Env (both): `C2HLS_RAG2=1`, skills on, `C2HLS_PART=xcu280-fsvh2892-2L-e`, `C2HLS_CLOCK_NS=3.33`, `C2HLS_CHATHLS_NOSKILLS` unset, `C2HLS_DEEPSEEK_PEAK_PAUSE=1`, `C2HLS_DEEPSEEK_SKIP_PEAK` only if operator overrides.

**Corpus note:** MachSuite batch_parallel today uses `tier_B_ready`. HLSFactory full config uses the standard flash corpus resolution for `hlsfactory_*`. Exporter reads `benchmarks/`; c2hls campaigns continue to use whatever ready/corpus path those workflows already use (must remain the same kernel sources).

### 3. ChatHLS hybrid 46-bench submit

1. New list file: `scripts/pc2/c2hls_port_46_benches.txt` (prefixed names).
2. Extend / wrap `submit_chathls_hybrid_batch_parallel.sh` to accept `CHATHLS_BENCH_LIST` pointing at that file (already supported if present — verify; else add).
3. Array size = 46; same U280 env (`setup_chathls_u280_env.sh`), DeepSeek API via login proxy or direct key as today’s hybrid.
4. Session id prefix e.g. `hybrid-u280-c2hlsport-...` for artifact isolation.

### 4. Parallel launch orchestration

New optional umbrella (c2hls or ChatHLS scripts/pc2):

1. Ensure DeepSeek proxy up (reuse `c2hls_deepseek_proxy.sh` or ChatHLS proxy; **one** proxy, document port).
2. Run exporter + validation.
3. Smoke (2 ChatHLS + 2 c2hls dry or short).
4. Submit **in parallel**:
   - c2hls machsuite deepseek rag2_skills
   - c2hls hlsfactory deepseek rag2_skills
   - ChatHLS hybrid 46
5. Write a small `dual_track_state.json` with campaign roots / job ids.

Vitis CPU capacity: both tracks will compete on `normal` — acceptable; do not over-subscribe beyond site norms (prefer combined-HLS / existing machsuite node counts rather than 46+46 exclusive nodes).

### 5. Compare / reporting

After completion (or partial):

- Collect ChatHLS `final_latency_csynth.csv` / `final_resources_csynth.csv` for the 46-port session
- Collect c2hls flash_selected (+ dataflow where OK) latency/resources per prefixed bench
- Emit md under `c2hls/docs/pc2/` analogous to `2026-07-18-deepseek-u280-c2hls-vs-chathls.md`, keyed by prefixed bench name

## Risks & mitigations

| Risk | Mitigation |
|------|------------|
| Loop label injection misses nested/MachSuite macros | Deterministic visitor + per-bench `port_manifest` warnings; manual fix list for failures |
| ChatHLS assumes filename == top (`atax`/`atax`) | Sidecar mapping; adjust TCL `set_top` / `add_files` to match emitted names |
| Multi-file MachSuite (`support.c`, data files) | v1: kernel-only synth path like ChatHLS (csynth of top); document cosim gaps; prefer gold/baseline already used by c2hls flash |
| DeepSeek queue saturation (3 campaigns) | `workers=1`; peak pause on c2hls; expect long wall-clock |
| HLSFactory flash path lacks streaming dataflow | Prefer enabling machsuite-style watcher where workflow supports it; else report flash-only for that subset |
| Drain/`C2HLS_CHATHLS_NOSKILLS` env bug | RAG2+skills leaves noskills **off**; export `C2HLS_RAG2` into helper jobs explicitly |

## Success criteria

1. 46 ChatHLS dirs exist with cpp + kernel_info + run_hls.tcl; exporter idempotent.
2. Smoke hybrid csynth starts on ≥2 ported benches.
3. c2hls RAG2+skills DeepSeek U280 campaigns submitted for machsuite 18 and hlsfactory 28.
4. ChatHLS hybrid array submitted for all 46 prefixed benches.
5. Compare artifact listing latency (and resources where present) c2hls vs ChatHLS per prefixed bench.

## Implementation phases (for plan)

1. Exporter + tests (label injection, 1 hlsfactory + 1 machsuite golden fixture)
2. Export all 46 into ChatHLS tree + bench list
3. c2hls DeepSeek RAG2+skills configs/starters (machsuite + hlsfactory)
4. ChatHLS hybrid list + submit wrapper
5. Umbrella parallel launcher + smoke
6. Compare report script

## Defaults for previously open points

| Topic | Default for v1 |
|-------|----------------|
| `.cpp` basename vs top | File = `<top>.cpp` (e.g. `kernel_atax.cpp`); dir = prefixed bench id; TCL `add_files`/`set_top` use top name |
| HLSFactory dataflow | **Yes** — attach ChatHLS/machsuite-style streaming flash→dataflow watcher when workflow allows; if blocked, flash+cosim and note in compare |
| Shared DeepSeek proxy | Use **c2hls** `c2hls_deepseek_proxy.sh` on a free port (prefer 18092 if free); point ChatHLS hybrid `OPENAI_BASE_URL` at the same proxy for the dual run |
