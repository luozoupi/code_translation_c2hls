# HLSFactory Gold Timeout Fill A+B Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** For deepseek/devstral aav_n benches missing gold and/or flash full-size cosim, run parallel **csynth_full** and **cosim_small** on bench **copies** and emit annotated fill CSVs.

**Architecture:** Prepare `fill_ab_20260730/` with full+small bench copies; manifest of jobs; Slurm array runs `hls_eval.run_hls_synthesis` / `run_cosim`; reporter writes four CSVs with `metric_kind` and timeout notes.

**Tech Stack:** Python 3, hls_eval, Vitis HLS (PC2), Slurm

**Spec:** `docs/superpowers/specs/2026-07-30-hlsfactory-gold-timeout-fill-ab-design.md`

---

### Task 1: Prepare copies + job manifest

**Files:**
- Create: `scripts/pc2/fill_ab_timeout_lib.py`
- Create: `scripts/pc2/prepare_fill_ab_timeout.py`

- [ ] Scaffold `artifacts/pc2/reports/hlsfactory_flash_cosim_vs_gold_latrag_off/fill_ab_20260730/`
- [ ] Copy benches; apply small macros only under `benches_small/`
- [ ] Copy gold + flash kernels; write `jobs.jsonl` manifest

### Task 2: Runner + Slurm submit

**Files:**
- Create: `scripts/pc2/run_fill_ab_one.py`
- Create: `scripts/pc2/submit_fill_ab_timeout.sh`
- Create: `scripts/pc2/fill_ab_array.sbatch.sh`

- [ ] Run one job (csynth or cosim, gold or flash) to JSON under `work/`
- [ ] Submit A and B arrays in parallel

### Task 3: Report CSVs

**Files:**
- Create: `scripts/pc2/report_fill_ab_timeout.py`

- [ ] Emit four fill CSVs with `metric_kind` and footer notes
- [ ] Do not overwrite original speedup CSVs

### Task 4: Verify

- [ ] Confirm originals under `benchmarks_cosim/` unchanged
- [ ] Spot-check one csynth_full and one cosim_small result JSON
