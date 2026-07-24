# ChatHLS U280 full-size cosim — high-memory tooling retry (2026-07-21)

## Goal

Re-run **ChatHLS native** U280 full-size cosim **only** for benches that previously failed due to tooling/memory aborts (OOM / `Killed` / xelab·xsim `SIGSEGV` / missing xsim snapshot), **not** for resource-over-budget fails.

Original batch (unchanged): `artifacts/pc2/u280_compare_cosim_20260719_072401/`  
Original ChatHLS cosim: `.../flash_cosim/20260719_072401_chathls_u280/` (8/14 pass under 32G).

## Classification

| Class | Benches | Action |
|-------|---------|--------|
| **Resource overuse = keep FAIL** | `matmul` (DSP ≫ budget), `kernel_3mm` (FF ≫ budget + SIGSEGV) | **Not retried** |
| **Tooling/memory retry** | `gemm`, `gemm_ncubed`, `kernel_symm`, `kernel_syrk` | Retried @ **256G** / 16 CPUs |
| **Already PASS** | atax, bicg, covariance, gemm_blocked, gesummv, mvt, syr2k, kernel_2mm | Untouched |

## New batch

| Field | Value |
|-------|-------|
| batch_root | `artifacts/pc2/u280_compare_cosim_20260721_011927/` |
| run_root | `.../flash_cosim/20260721_011927_chathls_u280_memretry/` |
| job_id | `2030824` (array 0–3) |
| mem / cpus | `PC2_COSIM_MEM=256G`, `PC2_COSIM_CPUS=16` |
| walltime / timeout | `7-00:00:00` / `C2HLS_COSIM_TIMEOUT=604800` |
| kernel | selected from `hybrid-u280-split-20260717-001649` via `final_latency_csynth.csv` |
| launcher | `scripts/pc2/start_chathls_u280_cosim_mem_retry.sh` |
| DS/GLM campaigns | **not** launched |

## How to report when jobs finish

```bash
python3 scripts/pc2/report_chathls_u280_cosim_mem_retry.py \
  --run-root artifacts/pc2/u280_compare_cosim_20260721_011927/flash_cosim/20260721_011927_chathls_u280_memretry \
  --write-md docs/pc2/2026-07-21-chathls-u280-cosim-memretry-results.md
```

## Outcomes (fill after array completes)

| Bench | passed | cycles | tooling flags | first error | notes |
|-------|--------|--------|---------------|-------------|-------|
| gemm | pending | — | — | — | was Killed @ BIND (32G) |
| gemm_ncubed | pending | — | — | — | was xelab SIGSEGV |
| kernel_symm | pending | — | — | — | was xsim Killed / missing snapshot |
| kernel_syrk | pending | — | — | — | was COSIM FAIL after C TB smoke PASS |

## Pass-rate accounting (ChatHLS native full-size)

- Original @ 32G: **8 / 14** pass (resource fails + tooling fails both counted fail).
- After this retry: `pass = 8 + (# of retry benches that now pass)`.
- `matmul` and `kernel_3mm` remain **resource fails** even if tooling would also abort.
