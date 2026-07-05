# Tier A Flash batch_parallel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run tier_A_ready flash campaigns with one always-on GPU for LLM codegen and ~12 parallel Vitis workers (csynth+csim, no RTL cosim), replacing serialized `run_tier_a_flash_smoke.sh` for production sweeps.

**Architecture:** Extend existing `batch_parallel` queue/coordinator/watch/gpu_drain with `workflow: tier_a_flash` — reference gold gates as parallel `synth` jobs, Phase B/flash as alternating `codegen`/`synth` with inline csim on synth workers.

**Tech Stack:** Python 3, SQLite queue, Slurm (gpu_h100 + normal), Vitis HLS, existing `C2HLSOrchestrator` pipelined APIs, `tier_a_flash_lib`.

**Design spec:** `docs/superpowers/specs/2026-07-01-tier-a-flash-batch-parallel-design.md`

---

## File map

| File | Action | Responsibility |
|------|--------|----------------|
| `scripts/pc2/batch_parallel_tier_a_lib.py` | Create | Bench resolve, env, cell paths, workflow helpers |
| `scripts/pc2/tier_a_batch_parallel_bench.py` | Create | `TierABatchParallelBenchSession`, `execute_job` |
| `scripts/pc2/batch_parallel_queue.py` | Modify | Workflow-aware `seed_bench` |
| `scripts/pc2/batch_parallel_config.py` | Modify | `workflow`, `corpus` in pilot; load from JSON |
| `scripts/pc2/batch_parallel_gpu_drain.py` | Modify | Workflow branch for tier_a |
| `scripts/pc2/batch_parallel_worker.py` | Modify | Workflow branch for tier_a |
| `scripts/pc2/start_batch_parallel_campaign.sh` | Modify | Pass reference seed for tier_a workflow |
| `scripts/pc2/start_tier_a_batch_parallel.sh` | Create | User-facing submit wrapper |
| `scripts/pc2/batch_parallel_tier_a_flash.json` | Create | Default campaign config (4×3 synth, always_on) |
| `scripts/pc2/batch_parallel_tier_a_forgebench10.json` | Create | 10 forgebench list preset |
| `tests/test_tier_a_batch_parallel_bench.py` | Create | Session followup unit tests |
| `tests/test_batch_parallel_tier_a_seed.py` | Create | Queue seeding tests |

---

### Task 1: Campaign config and workflow field

**Files:**
- Modify: `scripts/pc2/batch_parallel_config.py`
- Create: `scripts/pc2/batch_parallel_tier_a_flash.json`
- Create: `scripts/pc2/batch_parallel_tier_a_forgebench10.json`

- [ ] **Step 1: Add `workflow` and `corpus` to `BatchParallelConfig`**

In `scripts/pc2/batch_parallel_config.py`, extend the dataclass:

```python
@dataclass
class BatchParallelConfig:
    ...
    pilot_workflow: str = "flash"  # flash | tier_a_flash
    pilot_corpus: str = ""         # tier_A_ready when tier_a_flash
```

In `load_config()` / JSON merge, read `pilot.workflow` and `pilot.corpus` into these fields. Include in `to_dict()`.

- [ ] **Step 2: Create base tier_A config**

Create `scripts/pc2/batch_parallel_tier_a_flash.json`:

```json
{
  "synth_nodes_per_variant": 4,
  "synth_workers_per_node": 3,
  "cosim_nodes_per_variant": 0,
  "cosim_workers_per_node": 0,
  "worker_cpus": 8,
  "worker_mem_gb": 32,
  "gpu_policy": "always_on",
  "max_inflight_benches": 10,
  "bench_order": "listed",
  "bench_seeding": "short_first_waves",
  "poll_sec": 2.0,
  "coordinator_poll_sec": 15.0,
  "pilot": {
    "variant": "tier_a_90",
    "workflow": "tier_a_flash",
    "corpus": "tier_A_ready",
    "benches": ["spector_hls_dct", "hp_fft_n256__UF1"],
    "failure_policy": "ignore",
    "model": "mistralai/Devstral-2-123B-Instruct-2512",
    "turns": 4
  }
}
```

- [ ] **Step 3: Create forgebench-10 preset**

Create `scripts/pc2/batch_parallel_tier_a_forgebench10.json` — same as above but `benches` = 10 forgebench names from tier_A 25 list.

- [ ] **Step 4: Verify config loads**

Run:

```bash
cd /scratch/hpc-prf-llmfpga/asa582/projects/c2hls
python3 -c "
import sys; sys.path.insert(0,'scripts/pc2')
import os; os.environ['BATCH_PARALLEL_CONFIG']='scripts/pc2/batch_parallel_tier_a_flash.json'
from batch_parallel_config import load_config
c=load_config()
assert c.pilot_workflow=='tier_a_flash'
assert c.cosim_nodes_per_variant==0
assert c.synth_slots_per_variant==12
assert c.gpu_policy=='always_on'
print('ok', c.synth_slots_per_variant)
"
```

Expected: `ok 12`

---

### Task 2: tier_a lib helpers

**Files:**
- Create: `scripts/pc2/batch_parallel_tier_a_lib.py`
- Test: `tests/test_batch_parallel_tier_a_lib.py`

- [ ] **Step 1: Write failing test for bench resolution**

Create `tests/test_batch_parallel_tier_a_lib.py`:

```python
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "pc2"))

from batch_parallel_tier_a_lib import resolve_tier_a_bench_map, TIER_A_VARIANT, SETUP_TAG


class TierALibTests(unittest.TestCase):
    def test_resolve_dct(self) -> None:
        m = resolve_tier_a_bench_map(["spector_hls_dct"])
        self.assertIn("spector_hls_dct", m)
        self.assertTrue((m["spector_hls_dct"] / "plain.cpp").is_file())

    def test_constants(self) -> None:
        self.assertEqual(TIER_A_VARIANT, "tier_a_90")
        self.assertIn("tier_a", SETUP_TAG)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python3 -m pytest tests/test_batch_parallel_tier_a_lib.py -v
```

Expected: FAIL — module not found

- [ ] **Step 3: Implement `batch_parallel_tier_a_lib.py`**

```python
"""Tier A batch_parallel helpers."""

from __future__ import annotations

import os
from pathlib import Path

from tier_a_flash_lib import (
    SETUP_TAG,
    configure_tier_a_flash_90skills_env,
    resolve_tier_a_benches,
)

TIER_A_VARIANT = "tier_a_90"
WORKFLOW_TIER_A_FLASH = "tier_a_flash"


def resolve_tier_a_bench_map(benches: list[str]) -> dict[str, Path]:
    return {name: path for name, path in resolve_tier_a_benches(benches)}


def configure_tier_a_campaign_env() -> None:
    configure_tier_a_flash_90skills_env()


def cell_dir(cell_root: Path, bench: str, model_tag: str) -> Path:
    return cell_root / bench / f"{model_tag}__{SETUP_TAG}"


def workflow_from_campaign(campaign: dict) -> str:
    pilot = (campaign.get("config") or {}).get("pilot") or campaign.get("pilot") or {}
    return str(pilot.get("workflow") or "flash")


def is_tier_a_workflow(campaign: dict) -> bool:
    return workflow_from_campaign(campaign) == WORKFLOW_TIER_A_FLASH


def benches_from_env_or_config(cfg) -> list[str]:
    raw = os.getenv("C2HLS_TIER_A_FLASH_BENCHES", "").strip()
    if raw:
        return [b.strip() for b in raw.split(",") if b.strip()]
    return list(cfg.pilot_benches)
```

- [ ] **Step 4: Run test**

```bash
python3 -m pytest tests/test_batch_parallel_tier_a_lib.py -v
```

Expected: PASS

---

### Task 3: Queue reference seeding

**Files:**
- Modify: `scripts/pc2/batch_parallel_queue.py`
- Test: `tests/test_batch_parallel_tier_a_seed.py`

- [ ] **Step 1: Write failing seed test**

```python
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "pc2"))

from batch_parallel_queue import BatchParallelQueue


class TierASeedTests(unittest.TestCase):
    def test_seed_reference_job(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            q = BatchParallelQueue(Path(td) / "q.db")
            q.register_benches("tier_a_90", ["spector_hls_dct"])
            q.seed_bench("tier_a_90", "spector_hls_dct", initial_kind="synth", initial_phase="reference")
            with q._conn() as conn:
                row = conn.execute(
                    "SELECT kind, phase, stage FROM jobs WHERE bench=?",
                    ("spector_hls_dct",),
                ).fetchone()
            self.assertEqual(row["kind"], "synth")
            self.assertEqual(row["phase"], "reference")
            self.assertEqual(row["stage"], "gold_gate")
```

- [ ] **Step 2: Run test — expect FAIL** (unexpected keyword `initial_kind`)

- [ ] **Step 3: Extend `seed_bench` signature**

In `batch_parallel_queue.py`:

```python
def seed_bench(
    self,
    variant: str,
    bench: str,
    *,
    initial_kind: str = "codegen",
    initial_phase: str = "phase_b",
    initial_stage: str = "",
) -> None:
    ...
    stage = initial_stage or ("gold_gate" if initial_phase == "reference" else "translate")
    conn.execute(
        """
        INSERT INTO jobs(variant, bench, kind, phase, attempt, stage, meta_json, status, created_at)
        VALUES (?, ?, ?, ?, 0, ?, '{}', 'pending', ?)
        """,
        (variant, bench, initial_kind, initial_phase, stage, time.time()),
    )
```

Default args preserve Rodinia behavior (`codegen`, `phase_b`, `translate`).

- [ ] **Step 4: Run seed test — expect PASS**

---

### Task 4: TierABatchParallelBenchSession core

**Files:**
- Create: `scripts/pc2/tier_a_batch_parallel_bench.py`
- Test: `tests/test_tier_a_batch_parallel_bench.py`

- [ ] **Step 1: Write test — reference pass enqueues codegen**

```python
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "pc2"))

from batch_parallel_queue import BatchParallelJob
from tier_a_batch_parallel_bench import TierABatchParallelBenchSession


class TierABenchSessionTests(unittest.TestCase):
    def test_reference_pass_enqueues_phase_b_codegen(self) -> None:
        with patch.object(TierABatchParallelBenchSession, "__init__", lambda self, **kw: None):
            session = TierABatchParallelBenchSession(
                variant_key="tier_a_90", bench="spector_hls_dct",
                bench_dir=Path("/tmp"), cell_dir=Path("/tmp/cell"),
                model_id="m", turns=4,
            )
        session.variant_key = "tier_a_90"
        session.bench = "spector_hls_dct"
        session.cell_dir = Path("/tmp/cell")
        session.inputs = {"meta": {}}
        job = BatchParallelJob(
            id=1, variant="tier_a_90", bench="spector_hls_dct",
            kind="synth", phase="reference", attempt=0, stage="gold_gate", meta={},
        )
        ref_ok = {"benchmark_ready": True, "invalid_reason": "", "report": {"latency_ns": 1}}
        with patch("tier_a_batch_parallel_bench.validate_gold_reference", return_value=ref_ok):
            followups = session._run_reference_synth(job)
        self.assertEqual(followups[0]["kind"], "codegen")
        self.assertEqual(followups[0]["phase"], "phase_b")
        self.assertEqual(followups[0]["stage"], "translate")

    def test_phase_b_synth_success_no_cosim_followup(self) -> None:
        with patch.object(TierABatchParallelBenchSession, "__init__", lambda self, **kw: None):
            session = TierABatchParallelBenchSession(
                variant_key="tier_a_90", bench="spector_hls_dct",
                bench_dir=Path("/tmp"), cell_dir=Path("/tmp/cell"),
                model_id="m", turns=4,
            )
        session.bench = "spector_hls_dct"
        job = BatchParallelJob(
            id=2, variant="tier_a_90", bench="spector_hls_dct",
            kind="synth", phase="phase_b", attempt=0, stage="synth", meta={},
        )
        mock_orch = MagicMock()
        mock_orch.hls_code = "code"
        mock_orch.header_code = ""
        mock_orch.header_name = "kernel.h"
        mock_orch.translated_hls_top = "top"
        mock_orch.part = "xcu280-fsvh2892-2L-e"
        mock_orch.clock_ns = 3.33
        mock_orch.extra_files = []
        mock_orch.testbench_code = "tb"
        mock_orch.turns_limitation = 4
        mock_orch.turn_results = []
        mock_orch.synthesis.revert_threshold = 3
        mock_orch.synthesis._should_revert.return_value = False
        mock_orch.synthesis._record_best.return_value = {}
        mock_orch._pipelined_ctx = {}
        session.orchestrator = mock_orch
        outcome = {
            "synth": {"success": True, "report": {"latency_ns": 1}},
            "csim": {"ran": True, "passed": True},
            "cosim": None,
        }
        with patch("c2hls.compile_check_cpp", return_value=(True, "")):
            with patch.object(session, "_synth_csim_only", return_value=outcome):
                followups = session._run_synth_phase_b(job)
        self.assertEqual(followups[0]["kind"], "codegen")
        self.assertEqual(followups[0]["phase"], "flash")
        kinds = [f["kind"] for f in followups]
        self.assertNotIn("cosim", kinds)
```

- [ ] **Step 2: Run tests — expect FAIL**

- [ ] **Step 3: Implement session skeleton**

Create `scripts/pc2/tier_a_batch_parallel_bench.py` with:

- `validate_gold_reference` import from `c2hls`
- `TierABatchParallelBenchSession(BatchParallelBenchSession)`
- `_synth_csim_only()` — copy pattern from `batch_parallel_bench._synth_only` but `run_csim_check=True`
- `_run_reference_synth(job)` — validate, write `cell_dir/reference_validation.json`, return codegen or finalize
- `_run_synth_phase_b(job)` / `_run_synth_flash(job)` — extracted from parent `_run_synth` without cosim followup; handle csim fail → codegen repair
- Override `_run_synth(job)` to dispatch on `job.phase`
- Override `_ensure_orchestrator()` — load `reference_validation.json`; skip `validate_gold_reference` in parent init; raise if not `benchmark_ready`

Reference validation write:

```python
def _save_reference_validation(self, payload: dict) -> None:
    path = self.cell_dir / "reference_validation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    self.reference_validation = payload
```

For `_ensure_orchestrator`, subclass `FlashPipelinedBenchSession` init path: after loading inputs, read reference file before Phase A.

- [ ] **Step 4: Implement `execute_job` wrapper** (same signature as `batch_parallel_bench.execute_job` but uses `TierABatchParallelBenchSession`)

- [ ] **Step 5: Run unit tests — expect PASS**

```bash
python3 -m pytest tests/test_tier_a_batch_parallel_bench.py -v
```

---

### Task 5: Wire GPU drain and synth worker

**Files:**
- Modify: `scripts/pc2/batch_parallel_gpu_drain.py`
- Modify: `scripts/pc2/batch_parallel_worker.py`

- [ ] **Step 1: Add shared dispatch helper**

At top of both files, pattern:

```python
from batch_parallel_tier_a_lib import (
    TIER_A_VARIANT,
    configure_tier_a_campaign_env,
    is_tier_a_workflow,
    resolve_tier_a_bench_map,
    cell_dir as tier_a_cell_dir,
)
from batch_parallel_tier_a_bench import execute_job as tier_a_execute_job
from batch_parallel_bench import execute_job as rodinia_execute_job
```

In drain `bench_dir_for`:

```python
campaign = load_campaign(campaign_root)
if is_tier_a_workflow(campaign):
    m = resolve_tier_a_bench_map(benches_order)
    ...
```

Replace `configure_fixed_cosim_flash_env` + `VARIANTS.get` with:

```python
if is_tier_a_workflow(campaign):
    configure_tier_a_campaign_env()
    cell = tier_a_cell_dir(cell_root / job.variant, job.bench, model_tag)
    tier_a_execute_job(...)
else:
    ...  # existing rodinia path
```

- [ ] **Step 2: Mirror changes in `batch_parallel_worker.py`** for synth role only (cosim workers not submitted when nodes=0).

- [ ] **Step 3: Manual import smoke**

```bash
python3 -c "
import sys; sys.path.insert(0,'scripts/pc2')
from batch_parallel_gpu_drain import main
print('import ok')
"
```

---

### Task 6: Campaign start script integration

**Files:**
- Modify: `scripts/pc2/start_batch_parallel_campaign.sh`
- Create: `scripts/pc2/start_tier_a_batch_parallel.sh`

- [ ] **Step 1: Workflow-aware seed in campaign init Python block**

After `load_config()`, branch:

```python
wf = cfg.pilot_workflow
if wf == "tier_a_flash":
    seed_kw = dict(initial_kind="synth", initial_phase="reference", initial_stage="gold_gate")
else:
    seed_kw = {}
for bench in benches[:cfg.max_inflight_benches]:
    queue.seed_bench(variant, bench, **seed_kw)
# deferred benches: register only (existing loop)
```

Replace single `seed_initial_wave` call if needed, or extend `seed_initial_wave` to accept `**seed_kw`.

- [ ] **Step 2: Create `start_tier_a_batch_parallel.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${BATCH_PARALLEL_CONFIG:-${SCRIPT_DIR}/batch_parallel_tier_a_flash.json}"
STAMP="${BATCH_PARALLEL_STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
BENCHES="${C2HLS_TIER_A_FLASH_BENCHES:-}"
export BATCH_PARALLEL_CONFIG="${CONFIG}"
export BATCH_PARALLEL_STAMP="${STAMP}"
export C2HLS_TIER_A_FLASH_BENCHES="${BENCHES}"
exec "${SCRIPT_DIR}/start_batch_parallel_campaign.sh" --config "${CONFIG}" --stamp "${STAMP}" "$@"
```

`chmod +x` the script.

- [ ] **Step 3: Dry-run**

```bash
./scripts/pc2/start_tier_a_batch_parallel.sh --dry-run --stamp test_tier_a_bp_dry
```

Expected output includes:
- `synth: 4 nodes x 3 workers`
- `cosim: 0 nodes x 0 workers`
- seeded benches with reference jobs (verify via sqlite query in dry-run campaign dir)

```bash
python3 -c "
import sqlite3, sys
db=sys.argv[1]
c=sqlite3.connect(db)
for row in c.execute('SELECT bench,kind,phase FROM jobs'):
    print(row)
" artifacts/pc2/batch_parallel_test_tier_a_bp_dry/queue.db
```

Expected rows: `kind=synth phase=reference` for seeded benches.

---

### Task 7: Finalize artifacts and results JSON

**Files:**
- Modify: `scripts/pc2/tier_a_batch_parallel_bench.py`

- [ ] **Step 1: Reuse pipelined finalize from `FlashPipelinedBenchSession`**

On `finalize` success, call `_finalize_success()` inherited from parent — ensure it writes `{bench}_multistep_results.json` compatible with smoke.

- [ ] **Step 2: Add `reference_validation` to results payload** in finalize path (read from cell file).

- [ ] **Step 3: Write `matrix.json` at campaign root** (optional v1: per-variant summary JSON updated by coordinator on completion — match smoke `matrix.json` row schema from `run_tier_a_flash_smoke_batch.py`).

---

### Task 8: End-to-end validation

- [ ] **Step 1: Run all new unit tests**

```bash
python3 -m pytest tests/test_batch_parallel_tier_a_lib.py \
  tests/test_batch_parallel_tier_a_seed.py \
  tests/test_tier_a_batch_parallel_bench.py \
  tests/test_batch_parallel_bench.py -v
```

Expected: all PASS (existing Rodinia tests unchanged).

- [ ] **Step 2: PC2 pilot — 2 benches**

```bash
export C2HLS_TIER_A_FLASH_BENCHES="spector_hls_dct,forgebench_attention_op_p1"
./scripts/pc2/start_tier_a_batch_parallel.sh \
  --config scripts/pc2/batch_parallel_tier_a_flash.json \
  --stamp pilot_tier_a_bp_2
```

Verify:
- `flow/events.jsonl` shows parallel `synth_start` for both benches before any `codegen_start`
- `gpu_mode` stays `up` in `campaign.json`
- `spector_hls_dct` reaches flash; forgebench gold csim passes (fixed testbench)

- [ ] **Step 3: Replace deferred forgebench-10 submit**

Update `scripts/pc2/deferred_tier_a_forgebench_10_submit.sh` to call `start_tier_a_batch_parallel.sh` with `batch_parallel_tier_a_forgebench10.json` instead of serialized smoke.

---

## Spec coverage checklist

| Spec requirement | Task |
|------------------|------|
| 1 GPU always_on | Task 1 config, Task 6 |
| 4×3 synth slots | Task 1 |
| cosim_nodes=0 | Task 1, existing variant.sh |
| Parallel reference gate | Task 3, 4, 6 |
| csynth+csim on synth | Task 4 |
| tier_a_flash_lib env | Task 2, 5 |
| Repair turns=4 | Inherited pipelined paths |
| Artifact compat | Task 7 |
| start_tier_a_batch_parallel.sh UX | Task 6 |

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-01-tier-a-flash-batch-parallel.md`.

**Two execution options:**

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks  
2. **Inline Execution** — implement tasks sequentially in this session with checkpoints

Which approach do you want?
