# HLSFactory v4-flash Test1/2/3 (lat-opt / RAG2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run three sequential HLSFactory DeepSeek-v4-flash campaigns (lat_opt-only → rag2+lat_opt → rag2-only), each with concurrent skills/noskills/bare, pipeline flash→(lat_opt)→ranked async cosim + immediate dataflow→(lat_opt)→ranked async cosim, without blocking the next stage/test on cosim.

**Architecture:** Reuse batch_parallel flash (`C2HLS_FLASH_DEFER_COSIM=1`), existing `post_flash_latency_opt` / `post_flash_dataflow`, per-bench proxies, and parallel dataflow Slurm jobs. Add a shared flash/dataflow **candidate ranking + ranked cosim** helper (lifted from multistep). Rewrite the HLSFactory post waiter so dataflow starts from csynth rank-1 **before** flash cosim finishes, dataflow runs with cosim deferred, then ranked cosim is submitted async. Add a test-mode launcher + sequential test1→2→3 orchestrator that advances when selection (not cosim) completes.

**Tech Stack:** bash Slurm launchers, Python 3 (c2hls / post_flash_*), DeepSeek queue proxies, Vitis HLS 2023.2.

---

## Spec lock (do not drift)

| Test | RAG2 | Lat-opt both sides |
|------|------|--------------------|
| 1 `lat_opt` | off | on |
| 2 `rag2_lat` | on | on |
| 3 `rag2` | on | off |

- Flavors concurrent per test; tests sequential; next test not blocked by prior cosim.
- Pool = `{seed} ∪ lat_opt outputs` when lat-opt on; rank by ascending max csynth latency; seed can win.
- Cosim walks rank; all fail → cosim fail. Dataflow seeds from **csynth rank-1**, not cosim winner.
- Repair unchanged: flash MAX_REPAIR/TURNS; dataflow contract 4 + repair 4; lat-opt N=3 R=3.
- Packs/prompts as last triple; U280 @ 3.33 ns; per-bench proxies; 1 Slurm job/bench non-exclusive.

## File map

| File | Responsibility |
|------|----------------|
| `scripts/pc2/flash_df_candidate_rank.py` | Collect/rank flash+lat_opt and dataflow+lat_opt candidates; write ranking JSON; promote rank-1 pointers |
| `scripts/pc2/run_ranked_cosim_cell.py` | Walk ranked candidates; run cosim until pass or exhausted |
| `scripts/pc2/wait_hlsfactory_flash_lat_dataflow.sh` | New waiter: flash complete → (lat_opt already chained in finalize) → rank → async flash cosim + parallel dataflow (RUN_COSIM=0) → after dataflow cell: lat_opt → rank → async df cosim |
| `scripts/pc2/start_hlsfactory_deepseek_flash_dataflow_one.sh` | Add `--test lat_opt\|rag2_lat\|rag2`; toggle RAG2/lat_opt; point post watcher at new waiter |
| `scripts/pc2/start_hlsfactory_v4f_test_sequence.sh` | Launch test1 triple → wait selection done → test2 → test3; trail cosims |
| `tests/test_flash_df_candidate_rank.py` | Ranking / seed-wins / lat_opt-only-if-better |

Reuse without rewrite: `start_hlsfactory_parallel_dataflow.sh`, `start_hlsfactory_per_bench_proxies.sh`, `post_flash_latency_opt.py`, `multistep_batch_parallel_bench.rank_cosim_candidates` patterns, `run_campaign_selected_cosim.sh` submit primitives.

---

### Task 1: Candidate ranking library

**Files:**
- Create: `scripts/pc2/flash_df_candidate_rank.py`
- Create: `tests/test_flash_df_candidate_rank.py`

- [x] **Step 1: Write failing tests for flash-side pool ranking**

```python
# tests/test_flash_df_candidate_rank.py
from pathlib import Path
import json
from flash_df_candidate_rank import collect_flash_side_candidates, rank_candidates

def _write(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")

def test_flash_seed_wins_when_lat_opt_worse(tmp_path: Path):
    bench = "hlsfactory_gemm"
    cell = tmp_path
    _write(cell / f"{bench}_selected.cpp", "int selected;")
    _write(cell / f"{bench}_selected_report.json", json.dumps({"latency_cycles": 100}))
    _write(cell / f"{bench}_latency_opt.cpp", "int lat;")
    _write(cell / f"{bench}_latency_opt_report.json", json.dumps({"latency_cycles": 200}))
    _write(cell / f"{bench}_latency_opt_result.json", json.dumps({"success": True, "latency_cycles": 200}))
    ranked = rank_candidates(collect_flash_side_candidates(cell, bench))
    assert ranked[0]["id"] == "flash:seed"
    assert ranked[0]["latency_cycles"] == 100.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /scratch/hpc-prf-llmfpga/asa582/projects/c2hls && .venv/bin/python -m pytest tests/test_flash_df_candidate_rank.py::test_flash_seed_wins_when_lat_opt_worse -v`

Expected: FAIL (module missing)

- [x] **Step 3: Implement collector + ranker**

```python
# scripts/pc2/flash_df_candidate_rank.py (core API)
def _latency(report: dict | None) -> float | None: ...
def lat_opt_improved_vs_seed(seed_lat, post_lat) -> bool:  # strict <
def collect_flash_side_candidates(cell_dir: Path, bench: str) -> list[dict]:
    # seed: {bench}_selected.cpp + _selected_report.json (or final)
    # lat_opt: {bench}_latency_opt.cpp if result.success and report latency present
    # Always include seed when present; include lat_opt even if worse (ranking decides)
def collect_dataflow_side_candidates(cell_dir: Path, bench: str) -> list[dict]:
    # seed: {bench}_dataflow.cpp + report
    # lat_opt: {bench}_dataflow_latency_opt.cpp + result/report
def rank_candidates(cands: list[dict]) -> list[dict]:
    # sort by latency_cycles ascending; ties prefer seed over lat_opt (stable)
def write_ranking(cell_dir, bench, side: str, ranked: list) -> Path:
    # {bench}_{side}_candidate_ranking.json
def promote_rank1_for_downstream(cell_dir, bench, ranked, *, side: str) -> Path:
    # flash side: ensure {bench}_selected.cpp is rank-1 code (copy if needed)
    # dataflow side: ensure {bench}_dataflow_selected.cpp or keep resolve_selected_kernel happy
```

**Important:** Include **all** validated candidates in the pool (seed always; lat_opt if success+latency). Ranking picks min latency; seed can win. Cosim walks full ranked list.

- [x] **Step 4: Run tests to pass**

Run: `.venv/bin/python -m pytest tests/test_flash_df_candidate_rank.py -v`

- [ ] **Step 5: Commit** (only if user asked; otherwise skip)

---

### Task 2: Ranked cosim runner

**Files:**
- Create: `scripts/pc2/run_ranked_cosim_cell.py`
- Modify: reuse `run_campaign_selected_cosim.sh` submit pattern / `hls_eval` cosim helpers already used by campaign cosim

- [x] **Step 1: CLI that reads ranking JSON and tries cosim in order**

```bash
.venv/bin/python scripts/pc2/run_ranked_cosim_cell.py \
  --cell-dir CELL --bench BENCH --side flash|dataflow \
  --campaign-root ROOT
```

Behavior:
1. Load `{bench}_{side}_candidate_ranking.json` (or build via collector).
2. For each candidate in rank order: stage kernel as cosim input, run full-size cosim (`C2HLS_COSIM_XELAB_MT_OFF=1`).
3. On first pass: write `{bench}_{side}_cosim_opt_result.json` with winner id + latency; exit 0.
4. If all fail: write fail result; exit 1.

- [x] **Step 2: Slurm wrapper for one cell** (optional thin bash used by waiter)

`scripts/pc2/run_ranked_cosim_bench.sh` — sets env, calls the Python CLI.

---

### Task 3: Post waiter — dataflow before cosim

**Files:**
- Create: `scripts/pc2/wait_hlsfactory_flash_lat_dataflow.sh` (prefer new file; keep old waiter intact)
- Modify: `scripts/pc2/start_hlsfactory_parallel_dataflow.sh` EXPORT to allow `C2HLS_RUN_COSIM=0` and post-df lat_opt + ranked cosim hook
- Modify: `scripts/pc2/run_hlsfactory_dataflow_bench.sh` — after dataflow success, if lat_opt on run chain; then rank; submit ranked cosim job; do not wait

Pipeline per campaign:

```
wait campaign_status complete
export gate + matrix.json + flash_selected export
for each ready cell (or after all flash):
   rank flash-side candidates
   promote rank-1 → selected
   sbatch ranked cosim (flash)   # fire-and-forget
start_hlsfactory_parallel_dataflow.sh with:
  C2HLS_RUN_COSIM=0
  C2HLS_POST_FLASH_LATENCY_OPT / CHAIN_DATAFLOW from test mode
  after each bench script: rank df side; sbatch ranked cosim (dataflow)
```

- [x] **Step 1: Implement waiter script** calling existing parallel dataflow + new ranking/cosim
- [x] **Step 2: Smoke dry-run path** (`--dry-run` prints actions)

---

### Task 4: One-flavor launcher `--test` modes

**Files:**
- Modify: `scripts/pc2/start_hlsfactory_deepseek_flash_dataflow_one.sh`

- [x] **Step 1: Add `--test lat_opt|rag2_lat|rag2`**

| `--test` | env |
|----------|-----|
| `lat_opt` | `RAG2=0`, `POST_FLASH_LATENCY_OPT=1`, `LATENCY_OPT_CHAIN_FLASH=1`, `LATENCY_OPT_CHAIN_DATAFLOW=1` |
| `rag2_lat` | `RAG2=1` + corpora paths + lat_opt chains on |
| `rag2` | `RAG2=1` + corpora, lat_opt **off**, chains unset |

Artifact prefixes include test tag, e.g. `batch_parallel_hlsfactory_ds_v4f_skills_lat_opt`.

- [x] **Step 2: Point post watcher at `wait_hlsfactory_flash_lat_dataflow.sh`**
- [x] **Step 3: Keep flavor skills/noskills/bare wiring from last triple**
- [x] **Step 4: Ensure `C2HLS_CLOCK_NS=3.33` and flash_pipelined fallback already 3.33**

---

### Task 5: Sequential test orchestrator

**Files:**
- Create: `scripts/pc2/start_hlsfactory_v4f_test_sequence.sh`
- Create: `scripts/pc2/wait_hlsfactory_test_selection_done.py` (or bash)

- [x] **Step 1: For test in lat_opt, rag2_lat, rag2:**
  - start 3 flavor one.sh with same stamp suffix `STAMP_testN`
  - start/reuse proxy port bases (skills/noskills/bare + per-bench for post)
  - poll until all 3 campaigns have **dataflow-side ranking written** (or dataflow done + ranking) for ≥ threshold benches
  - **do not** wait for cosim job completion
  - proceed to next test

- [x] **Step 2: Manifest** under `artifacts/pc2/hlsfactory_ds_v4f_tests_<STAMP>/` listing campaign roots, job ids, test modes

---

### Task 6: Verification

- [x] Unit tests for ranking (seed wins / lat_opt wins / dataflow side)
- [x] Dry-run launcher prints correct env for each `--test` × flavor
- [ ] Manual: one bench smoke on `lat_opt` skills if queue allows (optional)

---

## Spec coverage check

| Requirement | Task |
|-------------|------|
| Test1/2/3 RAG2×lat_opt matrix | 4, 5 |
| skills/noskills/bare concurrent | 4, 5 (reuse triple pattern) |
| flash→lat_opt→rank; seed in pool | 1, 3, 4 |
| async cosim_flash_opt ranked | 2, 3 |
| immediate dataflow on rank-1 | 3 |
| dataflow→lat_opt→rank→async cosim | 2, 3, 4 |
| next test not blocked by cosim | 5 |
| per-bench proxies, 1 job/bench | reuse parallel dataflow |
| repair rounds unchanged | env defaults in one.sh |

## Out of scope

- Changing lat-opt plan/modify prompts or per-scope BRAM fields
- Replacing DeepSeek with another model
- Force-push / git commits unless user asks
