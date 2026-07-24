# ChatHLS U280 full-size cosim — SIGSEGV + high-mem retry #2 (2026-07-21)

## Goal

Retry the three remaining tooling/hard failures with:

1. **More RAM** on `largemem` (`1024G`) — especially `kernel_symm`, which was Slurm-killed at ~256G during xelab.
2. **xelab SIGSEGV mitigation** for `gemm_ncubed`: `cosim_design -setup` → inject `xelab -mt off` into `run_xsim.sh` → run `sim.sh`, plus `ulimit -s unlimited`.

`gemm` left alone (still running on prior job `2030824_0`).

## Classification (after memretry @ 256G)

| Bench | Prior @ 256G | Retry #2 action |
|-------|--------------|-----------------|
| gemm_ncubed | xelab `SIGSEGV` (~20G RSS — not OOM) | `-mt off` + 1024G |
| kernel_symm | OOM kill at xelab (`Killed`, MaxRSS≈256G) | 1024G largemem |
| kernel_syrk | clean `COSIM 212-4 FAIL` (~22G) | same mitigations (may stay functional fail) |

## Mitigations (code)

- `hls_eval.py`: `C2HLS_COSIM_XELAB_MT_OFF`, `C2HLS_COSIM_EXTRA_ARGS`, unlimited stack in `_run_vitis_cmd`
- `cosim_array.sbatch.sh`: `ulimit -s unlimited`
- Launcher: `scripts/pc2/start_chathls_u280_cosim_mem_retry2.sh`

## Batch

| Field | Value |
|-------|-------|
| batch_root | `artifacts/pc2/u280_compare_cosim_20260721_060802/` |
| run_root | `.../flash_cosim/20260721_060802_chathls_u280_memretry2/` |
| job_id | `2031402` (array 0–2) |
| mem / partition | `1024G` / `largemem` |
| env | `C2HLS_COSIM_XELAB_MT_OFF=1`, `C2HLS_COSIM_EXTRA_ARGS=-disable_deadlock_detection` |
| left alone | `gemm` (still on `2030824_0`) |

## Report

```bash
python3 scripts/pc2/report_chathls_u280_cosim_mem_retry.py \
  --run-root <RUN_ROOT> \
  --write-md docs/pc2/2026-07-21-chathls-u280-cosim-memretry2-results.md
```

## kernel_syrk zip error (2031402_2) → zipfix relaunch

**What happened:** After `xelab -mt off` built the snapshot, `xsim` died on `run all`
(`Simulator command interrupted` — same root as the earlier “COSIM 212-4 FAIL”).
Post-sim `df_record_move` then ran `zip process.zip` on an **empty** `glob` →
`zip error: Nothing to do! (process.zip)` (misleading surface error).

**Fix in `hls_eval.py`:** guard empty zip; `catch {df_record_move}`; strip
`-autoloadwcfg`; `ulimit -s` inside `run_xsim.sh`; unset `DISPLAY` for `sim.sh`.

**Relaunch:** job `2031425` — `artifacts/pc2/u280_compare_cosim_20260721_063227/flash_cosim/20260721_063227_chathls_u280_syrk_zipfix/`
