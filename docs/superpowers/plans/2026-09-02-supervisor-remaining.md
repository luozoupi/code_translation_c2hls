# Supervisor remaining asks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Document completed mm ablations as the four-column supervisor figure, re-measure ping-pong on the frozen multi-PE kernel, retry the two wave1 kernels still outside 2% of AutoSA, and explain 320 DSP.

**Architecture:** No new LLM stages. Reuse `flash_enforcement` numbers, `post_flash_overlap.py`, and `start_autosa_wave1_dse_stream_retry.sh`. Add `--mm-only` so overlap does not re-run wave1. New slide brief dated 2026-09-02; do not edit the Aug 18 4285-vs-4228 file.

**Tech Stack:** bash launchers, pytest, Vitis HLS 2023.2 on PC2 `normal`, DeepSeek-v4-flash at `http://login5:18092/v1`.

**Do not commit. Do not overwrite `20260830_mmflow`, recipes, family A/B/C, or Aug 18 slides.**

---

### Task 1: `--mm-only` on overlap launcher

**Files:**
- Modify: `scripts/pc2/start_autosa_pe_overlap.sh`
- Modify: `tests/test_post_flash_overlap.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_post_flash_overlap.py`:

```python
def test_overlap_launcher_supports_mm_only():
    text = (
        Path(__file__).resolve().parents[1]
        / "scripts/pc2/start_autosa_pe_overlap.sh"
    ).read_text(encoding="utf-8")
    assert "--mm-only" in text
    assert "20260830_mmflow" in text or "C2HLS_MM_MATRIX_ROOT" in text
```

- [ ] **Step 2: Run test, expect fail**

Run: `.venv/bin/python -m pytest tests/test_post_flash_overlap.py::test_overlap_launcher_supports_mm_only -v`

Expected: FAIL `assert '--mm-only' in text`

- [ ] **Step 3: Implement `--mm-only`**

In `scripts/pc2/start_autosa_pe_overlap.sh`:

- Default `MM_ROOT` stays the Aug 23 gap campaign (do not delete). Add a comment that frozen mmflow is `artifacts/pc2/batch_parallel_autosa_mm_flow_aav_n_gf_20260830_mmflow`.
- Parse `--mm-only`. When set, `JOBS_FILE` contains only the mm line.
- Keep `--mm-root`, `--wave1-root`, `--dry-run`, `--submit`, `--endpoint-url`.

- [ ] **Step 4: Re-run pytest**

Run: `.venv/bin/python -m pytest tests/test_post_flash_overlap.py tests/test_autosa_mm_flow_flavors.py -q`

Expected: PASS

---

### Task 2: Submit mm-only overlap on frozen mmflow

**Files:** none in git; writes `artifacts/pc2/post_flash_overlap/`

- [ ] **Step 1: Confirm seed**

`artifacts/pc2/batch_parallel_autosa_mm_flow_aav_n_gf_20260830_mmflow/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_dse.cpp` exists.

- [ ] **Step 2: Dry-run**

```bash
./scripts/pc2/start_autosa_pe_overlap.sh --dry-run --mm-only \
  --mm-root artifacts/pc2/batch_parallel_autosa_mm_flow_aav_n_gf_20260830_mmflow \
  --endpoint-url http://login5:18092/v1
```

Expected: one cell `autosa_mm`, seed=dse, no wave1 benches.

- [ ] **Step 3: Submit**

Same command without `--dry-run`, with `--submit`. Record job id under `artifacts/pc2/post_flash_overlap/slurm_job_*`. Endpoint `http://login5:18092/v1`. `C2HLS_STREAM_MAX_TOKENS=65536`.

Do not overwrite `*_selected.cpp`.

---

### Task 3: Wave1 2% retry (intel + getting_started)

**Files:** none in git; writes into `artifacts/pc2/batch_parallel_autosa_wave1_aav_n_gf_20260825_wave1_aav_n_gf`

- [ ] **Step 1: Dry-run**

```bash
./scripts/pc2/start_autosa_wave1_dse_stream_retry.sh --dry-run \
  --benches autosa_mm_intel,autosa_mm_getting_started \
  --endpoint-url http://login5:18092/v1
```

- [ ] **Step 2: Submit** (drop `--dry-run`, keep `--submit` default)

Force is already on in the launcher. Do not pass a benches CSV on the sbatch `--export` line (comma split). Stamp a new `slurm_job_wave1_retry_*`. Do not start a new flash campaign.

---

### Task 4: Comparison + DSP memo + wave1 table

**Files:**
- Modify: `artifacts/pc2/autosa_mm_ablation_20260901/comparison.md`
- Create: `docs/pc2/2026-09-02-why-320-dsp.md`
- Create: `docs/pc2/2026-09-02-wave1-vs-rank1.md`

- [ ] **Step 1: Extend comparison.md** with four columns, overlap 8678, 32×8 4583/1280, pointer to new overlap job. Metric lock unchanged.

- [ ] **Step 2: Why 320 DSP memo** — rank-1 iso-compute 16×4; U280 not filled; 32×8 4583/1280 does not beat 4228; family C iso-DSP 32PE is 2245/1280 vs cand 9 ≤1846 (selected 1571/5120 oversubscribes); P&R/HBM later.

- [ ] **Step 3: Wave1 table** with rank-1, agent, err%, pass/fail, retry status.

---

### Task 5: Slide brief + one-pager

**Files:**
- Create: `docs/pc2/2026-09-02-supervisor-remaining-slide-brief.md`
- Modify: `docs/pc2/2026-08-30-autosa-agentic-flow-one-pager.md`

- [ ] **Step 1: New slide brief** — four columns, pack waterfall (flash skills can slow), ping-pong ≠ hide-load-store, 320 DSP one slide, wave1 2%. Say hide load/store, not DSE. Quote 4216 / 4292 / 40454 / 12893. Keep Aug 18 file untouched.

- [ ] **Step 2: Update one-pager** numbers; keep “enforcement-only 12893 vs 4553 is not a pass.”

---

### Task 6: Canvas

**Files:**
- Create: `/pc2/users/h/haqc2/.cursor/projects/scratch-hpc-prf-llmfpga-asa582-projects-c2hls/canvases/supervisor-remaining-20260902.canvas.tsx`
- Modify: `/pc2/users/h/haqc2/.cursor/projects/scratch-hpc-prf-llmfpga-asa582-projects-c2hls/canvases/pe-stream-skill-catalog.canvas.tsx`

- [ ] **Step 1:** Four-column bar (cycles + DSP), pack waterfall, wave1 vs rank-1, 32×8 vs 16×4. Import only `cursor/canvas`. Real numbers, no placeholders.

- [ ] **Step 2:** Skill catalog: add noskills 4216, zero-shot 40454, overlap 8678, note pack-level not per-skill.

---

## Self-review

- Spec columns 1–4 → Tasks 4–5 (table) + Task 2 (overlap A/B) + existing enforcement.
- 2% benches → Task 3 (intel, getting_started) + Task 4 table (the four that already pass).
- 320 DSP → Task 4 memo, no 32×8 rerun.
- Slides → Task 5 new file, Aug 18 untouched.
- No TBD. No commit.
