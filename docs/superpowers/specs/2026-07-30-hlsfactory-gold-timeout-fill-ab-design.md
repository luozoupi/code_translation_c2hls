# HLSFactory gold/flash timeout fill (A+B)

Date: 2026-07-30  
Status: draft for review

## Problem

Shared full-size gold cosim (`misc/hlsfactory_baseline_u280_20260616_benchmarks_full_cosim.jsonl`) timed out (12h) for several PolyBench sizes. Some flash cosims are also missing. That blocks speedup rows / geomean inclusion in:

- `artifacts/pc2/reports/hlsfactory_flash_cosim_vs_gold_latrag_off/deepseek_v4_flash_aav_n_flash_cosim_speedup_vs_gold.csv`
- `artifacts/pc2/reports/hlsfactory_flash_cosim_vs_gold_latrag_off/devstral2_aav_n_flash_cosim_speedup_vs_gold.csv`

## Goals

Run **A and B in parallel** on **copied** benches only (never modify `benchmarks_cosim/` or original campaign trees):

| Track | Metric | Problem size | Both sides |
|-------|--------|--------------|------------|
| **B** | average **csynth** latency | **full** (same macros as today) | gold + flash |
| **A** | **cosim** `kernel_runtime_cycles` | **reduced** macros in copy | gold + flash |

Produce **new** annotated CSVs (do not overwrite existing report CSVs). Every fill row and footer must state whether the number is `csynth_full` or `cosim_small`.

## Non-goals

- No new LLM / flash campaign.
- No silent merge of fills into the original geomean without a `metric_kind` / footer note.
- No edits to existing headers/testbenches under `benchmarks_cosim/`.

## Models and kernels

| Model | Flash source (existing) | Gold source |
|-------|-------------------------|-------------|
| deepseek-v4-flash aav_n | `batch_parallel_hlsfactory_ds_v4f_skills_20260725_084654_v4f_skills_ab/variants/aav_n/<bench>/*_selected.cpp` | `benchmarks_cosim/<bench>/hls_baseline_cosim.cpp` |
| Devstral-2 aav_n | `batch_parallel_20260701_bp_full_aav_n_p28/variants/aav_n/<bench>/` selected/flash_opt when present; else sister `flash_fixed_cosim_aav_n_20260628_fixed_cosim_flash_r2_pipelined` for incomplete p28 cells | same gold |

## Benches

### deepseek-v4-flash aav_n

| Bench | Why |
|-------|-----|
| `hlsfactory_fdtd-2d` | gold cosim timeout (flash cosim OK) |
| `hlsfactory_heat-3d` | gold cosim timeout (flash cosim OK) |
| `hlsfactory_seidel-2d` | gold cosim timeout (flash cosim OK) |
| `hlsfactory_syr2k` | gold cosim timeout (flash cosim OK) |
| `hlsfactory_jacobi-1d` | gold OK, **flash cosim missing** |
| `hlsfactory_ludcmp` | gold OK, **flash cosim missing** |

### Devstral-2 aav_n

| Bench | Why |
|-------|-----|
| `hlsfactory_fdtd-2d` | gold + flash incomplete / missing |
| `hlsfactory_heat-3d` | gold + flash incomplete / missing |
| `hlsfactory_seidel-2d` | gold + flash incomplete / missing |
| `hlsfactory_syr2k` | gold cosim timeout (flash cosim OK at full size) |

## Copy layout

Root:

`artifacts/pc2/reports/hlsfactory_flash_cosim_vs_gold_latrag_off/fill_ab_20260730/`

```
fill_ab_20260730/
  benches_full/<model_tag>/<bench>/     # full macros; B
  benches_small/<model_tag>/<bench>/    # reduced macros; A
  work/{csynth_full,cosim_small}/...
  csv/
    deepseek_v4_flash_aav_n_fill_csynth_full.csv
    deepseek_v4_flash_aav_n_fill_cosim_small.csv
    devstral2_aav_n_fill_csynth_full.csv
    devstral2_aav_n_fill_cosim_small.csv
  README.md
```

`model_tag`: `deepseek_v4_flash_aav_n` | `devstral2_aav_n`.

Each copy is a full bench directory clone (`*.h`, gold, flash kernel as `flash_kernel.cpp`, `testbench_cosim.cpp`, `metadata.json`, …). Flash code is copied in; gold remains `hls_baseline_cosim.cpp`.

## Reduced sizes (A only)

| Bench | Full | Small |
|-------|------|-------|
| fdtd-2d | TMAX=40, NX=60, NY=80 | TMAX=10, NX=20, NY=24 |
| heat-3d | N=20, TSTEPS=40 | N=8, TSTEPS=10 |
| seidel-2d | N=120, TSTEPS=40 | N=40, TSTEPS=10 |
| syr2k | N=80, M=60 | N=24, M=20 |
| jacobi-1d | N=120, TSTEPS=40 | N=40, TSTEPS=10 |
| ludcmp | N=120 | N=40 |

Only headers in **`benches_small`** are changed. `benches_full` keeps original macros.

Update m_axi `depth=` in copied gold/flash sources if present so they match array sizes in the small copy (copy-only).

## Flows

### B — full csynth

For each `(model, bench)`:

1. Csynth gold kernel in `benches_full/...` → `latency_cycles` (avg).
2. Csynth flash kernel in same tree (or sibling work dir sharing header) → `latency_cycles`.
3. `speedup = gold_csynth_avg / flash_csynth_avg` when both finite.

Reuse existing c2hls / `hls_eval` synth helpers and PC2 Vitis env; part/clock match campaign (U280, 3.33 ns) when available.

### A — small cosim

For each `(model, bench)`:

1. Cosim gold on `benches_small/...`.
2. Cosim flash on same small tree.
3. `speedup = gold_cosim_cycles / flash_cosim_cycles` when both pass.

Cosim knobs: `C2HLS_COSIM_TRACE_LEVEL=none`, `C2HLS_COSIM_XELAB_MT_OFF=1`, generous but finite timeout (e.g. 2–6 h for small).

Run A and B jobs **in parallel** (Slurm array or background processes).

## CSV schema

Shared columns:

```
bench,model,metric_kind,problem_size_note,
gold_cycles_or_latency,flash_cycles_or_latency,speedup_gold_over_flash,
gold_status,flash_status,timeout_reason,source_gold,source_flash,included_in_geomean
```

- `metric_kind`: exactly `csynth_full` or `cosim_small` (never ambiguous).
- `problem_size_note`: e.g. `full: N=120 TSTEPS=40` or `small: N=40 TSTEPS=10`.
- `timeout_reason`: short tag, e.g. `gold_cosim_timeout`, `flash_cosim_missing`, `gold_and_flash_missing`.

### Footer (required on every fill CSV)

Must include both:

1. **Which metric**: `metric_kind=csynth_full` (full problem size average csynth latency) **or** `metric_kind=cosim_small` (reduced problem size cosim cycles).
2. **Why**: these benches had **gold cosim and/or flash cosim timeout/missing** on the full-size campaign; fills use the method named in `metric_kind`, not mixed into the original full-size cosim geomean without explicit labeling.

Example footer lines:

```
metric_kind=csynth_full
note=full-size average csynth latency (gold vs flash); used because gold cosim and/or flash cosim timed out or was missing on the original full-size run
original_csv=.../deepseek_v4_flash_aav_n_flash_cosim_speedup_vs_gold.csv
```

```
metric_kind=cosim_small
note=reduced-size cosim cycles (gold vs flash); used because gold cosim and/or flash cosim timed out or was missing on the original full-size run
small_sizes=see problem_size_note column / README
original_csv=...
```

## Geomean policy

- Fill CSVs may compute a **fill-only** geomean over successful A or B rows.
- Do **not** overwrite `__GEOMEAN__` in the original CSVs in this pass.
- Optional later: a merged “reported” geomean that cites `metric_kind` per filled bench — out of scope unless requested.

## Success criteria

1. Original `benchmarks_cosim/**` and campaign artifact sources unchanged (copies only).
2. Four fill CSVs exist with `metric_kind` on every data row and footer.
3. A and B attempted for every bench in scope; failures recorded with status/error, not silent blanks without reason.
4. Notes distinguish **cosim_small** vs **csynth_full** explicitly.

## Open implementation details (OK to decide during plan)

- Exact Slurm partition / mem / time for A vs B.
- Whether flash and gold share one HLS project with two tops or two separate work dirs (prefer two work dirs for isolation).
