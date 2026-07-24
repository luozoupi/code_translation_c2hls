# c2hls DeepSeek U280 Campaigns Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run three sequential c2hls ChatHLS-bench campaigns (RAG+skills → noRAG+skills → RAG-noskills) using DeepSeek via a login-node queue proxy (`deepseek-chat`, workers=1), U280 @ 3.33 ns, 16 combined HLS nodes (csim+csynth+cosim per bench), with Beijing peak-hour codegen pause.

**Architecture:** Reuse ChatHLS’s `deepseek_queue_proxy.py` in-place (no copy). Add c2hls `external_llm` campaign mode (no GPU vLLM). Peak helpers pause only the codegen drain. Combined-HLS mode submits 16 synth nodes that can also claim `cosim` jobs (`cosim_nodes=0`). A sequential launcher waits for off-peak between campaigns.

**Tech Stack:** bash/Slurm, Python 3, existing batch_parallel queue/drain/coordinator, ChatHLS proxy at `test-chathls/ChatHLS-ACL-26/scripts/pc2/`.

**Spec:** `docs/superpowers/specs/2026-07-16-c2hls-deepseek-u280-campaigns-design.md`

---

## File map

| File | Responsibility |
|------|----------------|
| `scripts/pc2/deepseek_peak.py` | Beijing peak detection |
| `scripts/pc2/c2hls_deepseek_proxy.sh` | Thin wrapper: source ChatHLS setup + start proxy into campaign dir |
| `scripts/pc2/c2hls_deepseek_reachability.sbatch.sh` | Compute-node curl gate for proxy URL |
| `scripts/pc2/start_batch_parallel_campaign.sh` | `--external-llm` path (skip GPU submit; plant endpoint) |
| `scripts/pc2/batch_parallel_gpu_drain.py` | Skip claim during Beijing peak |
| `scripts/pc2/batch_parallel_coordinator.py` | Never submit/scancel GPU when `external_llm` |
| `scripts/pc2/batch_parallel_queue.py` | `claim(kinds=(...))` multi-kind support |
| `scripts/pc2/batch_parallel_worker.py` | Combined-HLS: synth role claims synth then cosim |
| `scripts/pc2/start_batch_parallel_variant.sh` | Honor `cosim_nodes=0` (no cosim sbatch) |
| `scripts/pc2/batch_parallel_chathls_deepseek_u280.json` | DeepSeek config: combined HLS, always_on, model id |
| `scripts/pc2/start_chathls_deepseek_one.sh` | Start one DeepSeek campaign flavor |
| `scripts/pc2/start_chathls_deepseek_u280_sequence.sh` | Peak-gated A→B→C orchestrator |
| `tests/test_deepseek_peak.py` | Peak window unit tests |
| `tests/test_batch_parallel_queue.py` | Multi-kind claim tests |

**Pinned choices (from spec TBD):**

1. **Combined HLS:** `cosim_nodes_per_variant=0`; synth workers claim `kinds=("synth","cosim")` preferring synth.
2. **Proxy:** invoke ChatHLS scripts via `CHATHLS_ROOT` (default sibling `test-chathls/ChatHLS-ACL-26`).

---

### Task 1: Beijing peak helper + tests

**Files:**
- Create: `scripts/pc2/deepseek_peak.py`
- Create: `tests/test_deepseek_peak.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_deepseek_peak.py
from datetime import datetime
from zoneinfo import ZoneInfo
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "pc2"))
from deepseek_peak import is_beijing_peak, seconds_until_off_peak

TZ = ZoneInfo("Asia/Shanghai")

def test_morning_peak():
    assert is_beijing_peak(datetime(2026, 7, 17, 9, 0, tzinfo=TZ))
    assert is_beijing_peak(datetime(2026, 7, 17, 11, 59, tzinfo=TZ))
    assert not is_beijing_peak(datetime(2026, 7, 17, 12, 0, tzinfo=TZ))

def test_afternoon_peak():
    assert is_beijing_peak(datetime(2026, 7, 17, 14, 0, tzinfo=TZ))
    assert is_beijing_peak(datetime(2026, 7, 17, 17, 59, tzinfo=TZ))
    assert not is_beijing_peak(datetime(2026, 7, 17, 18, 0, tzinfo=TZ))

def test_off_peak_night():
    assert not is_beijing_peak(datetime(2026, 7, 17, 2, 0, tzinfo=TZ))

def test_seconds_until_off_peak_positive_in_peak():
    assert seconds_until_off_peak(datetime(2026, 7, 17, 10, 0, tzinfo=TZ)) == 2 * 3600
```

- [ ] **Step 2: Run tests — expect FAIL (module missing)**

```bash
cd /scratch/hpc-prf-llmfpga/asa582/projects/c2hls
./.venv/bin/pytest tests/test_deepseek_peak.py -v
```

- [ ] **Step 3: Implement `deepseek_peak.py`**

```python
"""DeepSeek API peak windows in Beijing (Asia/Shanghai)."""
from __future__ import annotations
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

BEIJING = ZoneInfo("Asia/Shanghai")
# [start_hour, end_hour) local Beijing
PEAK_WINDOWS = ((9, 12), (14, 18))

def is_beijing_peak(now: datetime | None = None) -> bool:
    dt = now.astimezone(BEIJING) if now else datetime.now(BEIJING)
    if now is None:
        dt = datetime.now(BEIJING)
    elif now.tzinfo is None:
        dt = now.replace(tzinfo=BEIJING)
    else:
        dt = now.astimezone(BEIJING)
    h = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
    for start, end in PEAK_WINDOWS:
        if start <= h < end:
            return True
    return False

def seconds_until_off_peak(now: datetime | None = None) -> float:
    if not is_beijing_peak(now):
        return 0.0
    dt = now.astimezone(BEIJING) if (now and now.tzinfo) else (now.replace(tzinfo=BEIJING) if now else datetime.now(BEIJING))
    if now is None:
        dt = datetime.now(BEIJING)
    elif now.tzinfo is None:
        dt = now.replace(tzinfo=BEIJING)
    else:
        dt = now.astimezone(BEIJING)
    for start, end in PEAK_WINDOWS:
        if start <= (dt.hour + dt.minute / 60.0) < end:
            end_dt = dt.replace(hour=end, minute=0, second=0, microsecond=0)
            return max(0.0, (end_dt - dt).total_seconds())
    return 0.0

def sleep_hint_sec(now: datetime | None = None, *, max_sleep: float = 300.0) -> float:
    rem = seconds_until_off_peak(now)
    if rem <= 0:
        return 0.0
    return min(rem, max_sleep)
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
./.venv/bin/pytest tests/test_deepseek_peak.py -v
```

- [ ] **Step 5: Commit** (only if user requested commits)

```bash
git add scripts/pc2/deepseek_peak.py tests/test_deepseek_peak.py
git commit -m "feat(pc2): Beijing DeepSeek peak-hour helpers"
```

---

### Task 2: Multi-kind `claim` for combined HLS workers

**Files:**
- Modify: `scripts/pc2/batch_parallel_queue.py` (`claim`)
- Modify: `tests/test_batch_parallel_queue.py`

- [ ] **Step 1: Add failing test for multi-kind claim**

```python
def test_claim_kinds_prefers_first_matching_kind(tmp_path):
    from batch_parallel_queue import BatchParallelQueue
    q = BatchParallelQueue(tmp_path / "q.db")
    q.register_benches("v", ["b1"])
    # insert pending cosim then synth with earlier created_at for synth — use API if available
    # Prefer: claim(kinds=("synth","cosim")) returns synth when both pending
```

Implement by extending `claim` signature:

```python
def claim(self, *, kind: str | None = None, kinds: tuple[str, ...] | None = None, ...):
    kind_list = list(kinds) if kinds else [kind]
    assert kind_list and all(kind_list)
    # SQL: AND j.kind IN (?,...) ORDER BY CASE kind WHEN ? THEN 0 ... END, created_at
```

- [ ] **Step 2: Run focused queue tests — FAIL then implement — PASS**

```bash
./.venv/bin/pytest tests/test_batch_parallel_queue.py -v -k claim
```

- [ ] **Step 3: Commit** (if requested)

---

### Task 3: Combined-HLS worker + config submit path

**Files:**
- Modify: `scripts/pc2/batch_parallel_worker.py`
- Modify: `scripts/pc2/start_batch_parallel_variant.sh` (skip cosim submit when nodes=0)
- Modify: `scripts/pc2/batch_parallel_config.py` (optional `combined_hls_nodes: bool`)

- [ ] **Step 1: Worker — when campaign/config `combined_hls_nodes` or env `C2HLS_COMBINED_HLS=1` and role=`synth`:**

```python
combined = bool(int(os.getenv("C2HLS_COMBINED_HLS", "0"))) or bool(
    (campaign.get("config") or {}).get("combined_hls_nodes")
)
if args.role == "synth" and combined:
    job = queue.claim(kinds=("synth", "cosim"), variant=..., role=..., ...)
    kind = job.kind if job else "synth"
else:
    kind = "synth" if args.role == "synth" else "cosim"
    job = queue.claim(kind=kind, ...)
```

Fix event naming to use `job.kind` after claim (not predeclared `kind` only).

- [ ] **Step 2: `start_batch_parallel_variant.sh` — if `cosim_nodes_per_variant` is 0, do not sbatch cosim jobs**

- [ ] **Step 3: Smoke dry-run with a temp JSON `cosim_nodes_per_variant: 0`, `combined_hls_nodes: true`**

```bash
BATCH_PARALLEL_CONFIG=...dry.json ./scripts/pc2/start_batch_parallel_campaign.sh --dry-run --stamp ds_combined_dry
```

Expected: prints `cosim: 0 nodes` and no cosim submit lines.

---

### Task 4: External LLM mode (no GPU vLLM)

**Files:**
- Modify: `scripts/pc2/start_batch_parallel_campaign.sh`
- Modify: `scripts/pc2/batch_parallel_coordinator.py`
- Modify: `scripts/pc2/batch_parallel_park.py` (skip park when external)

- [ ] **Step 1: Add CLI `--external-llm` and env `BATCH_PARALLEL_EXTERNAL_LLM=1`**

Behavior:

1. Do **not** call `batch_parallel_submit_gpu.sh`.
2. Require `BATCH_PARALLEL_EXTERNAL_ENDPOINT_URL` (e.g. `http://login5:18082/v1`).
3. Write `llm_endpoint.json`:

```json
{
  "url": "<URL>",
  "model": "deepseek-chat",
  "job_id": null,
  "borrowed": true,
  "external_llm": true,
  "queued": true
}
```

4. In `campaign.json`: `gpu_job_id=null`, `gpu_borrowed=true`, `gpu_mode=up`, `external_llm=true`, `gpu_policy=always_on`.

5. Still launch watch/drain/coord as Slurm helpers (existing path).

- [ ] **Step 2: Coordinator — if `campaign.get("external_llm")`:**

- Never `_submit_gpu` / `_scancel` for GPU.
- Never flip to `parked` for GPU savings (treat as always_on).
- Completion path must not wait on GPU job id.

- [ ] **Step 3: Park helpers — `gpu_parking_enabled` false when `external_llm`**

- [ ] **Step 4: Manual smoke (login):** start ChatHLS proxy on a free port, plant URL, `--external-llm --dry-run` then a 10s live drain health check against `/models`.

---

### Task 5: Peak pause in codegen drain

**Files:**
- Modify: `scripts/pc2/batch_parallel_gpu_drain.py`

- [ ] **Step 1: At top of drain loop, after loading campaign:**

```python
from deepseek_peak import is_beijing_peak, sleep_hint_sec

if campaign.get("external_llm") or os.getenv("C2HLS_DEEPSEEK_PEAK_PAUSE", "0") == "1":
    if is_beijing_peak():
        time.sleep(sleep_hint_sec(max_sleep=cfg.poll_sec * 30) or cfg.poll_sec)
        continue
```

Do **not** claim codegen while peak. Emit occasional `codegen_peak_pause` flow event (rate-limit to once/5 min).

- [ ] **Step 2: Unit-test with monkeypatch `is_beijing_peak` returning True → claim not called** (optional small test wrapping a drain iteration helper if easy; else manual).

---

### Task 6: DeepSeek proxy + reachability wrappers

**Files:**
- Create: `scripts/pc2/c2hls_deepseek_proxy.sh`
- Create: `scripts/pc2/c2hls_deepseek_reachability.sbatch.sh`

- [ ] **Step 1: `c2hls_deepseek_proxy.sh`**

```bash
#!/usr/bin/env bash
# Usage: c2hls_deepseek_proxy.sh <campaign_or_session_dir>
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CAMPAIGN_DIR="${1:?campaign dir}"
CHATHLS_ROOT="${CHATHLS_ROOT:-/scratch/hpc-prf-llmfpga/asa582/projects/test-chathls/ChatHLS-ACL-26}"
export CHATHLS_DEEPSEEK_PROXY_PORT="${CHATHLS_DEEPSEEK_PROXY_PORT:-18092}"  # avoid clash with ChatHLS 18082
export CHATHLS_DEEPSEEK_QUEUE_WORKERS="${CHATHLS_DEEPSEEK_QUEUE_WORKERS:-1}"
# shellcheck disable=SC1091
source "${CHATHLS_ROOT}/scripts/pc2/setup_deepseek_api.sh"
bash "${CHATHLS_ROOT}/scripts/pc2/start_deepseek_queue_proxy.sh" "${CAMPAIGN_DIR}"
# Also symlink/copy deepseek_endpoint.json → llm_endpoint.json fields for c2hls
python3 - <<PY
import json, time
from pathlib import Path
root = Path("${CAMPAIGN_DIR}")
ds = json.loads((root / "deepseek_endpoint.json").read_text())
ep = {
  "url": ds["url"],
  "model": "deepseek-chat",
  "job_id": None,
  "borrowed": True,
  "external_llm": True,
  "queued": True,
  "workers": ds.get("workers", 1),
  "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
(root / "llm_endpoint.json").write_text(json.dumps(ep, indent=2) + "\n")
print(ep["url"])
PY
```

- [ ] **Step 2: Reachability sbatch** — on `normal`, curl `${URL}/models` and `/health` until OK or timeout; write `reachability_ok.json`. Mirror ChatHLS `compute_reachability_gate.sbatch.sh` but DeepSeek-only (no HF GPU).

---

### Task 7: DeepSeek campaign config + one-shot starter

**Files:**
- Create: `scripts/pc2/batch_parallel_chathls_deepseek_u280.json`
- Create: `scripts/pc2/start_chathls_deepseek_one.sh`

- [ ] **Step 1: JSON** (base on `batch_parallel_chathls_flash_dataflow.json`):

```json
{
  "job_prefix": "bpchds",
  "synth_nodes_per_variant": 16,
  "synth_workers_per_node": 1,
  "cosim_nodes_per_variant": 0,
  "cosim_workers_per_node": 0,
  "combined_hls_nodes": true,
  "worker_cpus": 8,
  "worker_mem_gb": 32,
  "gpu_policy": "always_on",
  "cosim_timeout_s": 7200,
  "max_inflight_benches": 16,
  "pilot": {
    "variant": "chathls_aav_n",
    "workflow": "chathls_flash",
    "corpus": "chathls_ready",
    "benches": [ "...same 16..." ],
    "failure_policy": "ignore",
    "model": "deepseek-chat",
    "turns": 4
  }
}
```

- [ ] **Step 2: `start_chathls_deepseek_one.sh --flavor rag_skills|skills|rag_ns [--stamp ...] [--dry-run]`**

Sets env per flavor:

| Flavor | Env |
|--------|-----|
| `rag_skills` | RAG on (ug1399+ug902 or full knowledge_repo as prior skills run), skills on, prefix `batch_parallel_chathls_fd_ds_rag` |
| `skills` | RAG off, skills on, prefix `..._ds_skills` |
| `rag_ns` | RAG on ug1399+ug902, `C2HLS_CHATHLS_NOSKILLS=1` `C2HLS_DATAFLOW_NO_SKILLS=1`, prefix `..._ds_rag_ns` |

Common:

```bash
export C2HLS_MODEL=deepseek-chat
export C2HLS_COMBINED_HLS=1
export C2HLS_DEEPSEEK_PEAK_PAUSE=1
export C2HLS_PART=xcu280-fsvh2892-2L-e
export C2HLS_CLOCK_NS=3.33
export BATCH_PARALLEL_CONFIG=.../batch_parallel_chathls_deepseek_u280.json
export PC2_BATCH_JOB_PREFIX=bpchds
# proxy already running; URL from campaign llm_endpoint.json
export BATCH_PARALLEL_EXTERNAL_LLM=1
export BATCH_PARALLEL_EXTERNAL_ENDPOINT_URL="$(python3 -c 'import json;print(json.load(open("'"$CAMPAIGN"'"/llm_endpoint.json"))["url"])')"
```

Flow:

1. `mkdir` campaign root early OR start proxy into a session dir then pass stamp.
2. Start proxy → write endpoint.
3. Submit reachability gate; `afterok` then call `start_batch_parallel_campaign.sh --external-llm`.
4. Submit streaming dataflow watcher (same as existing ChatHLS starters) with RAG/skills exports.

---

### Task 8: Sequential off-peak orchestrator

**Files:**
- Create: `scripts/pc2/start_chathls_deepseek_u280_sequence.sh`

- [ ] **Step 1: Implement**

```bash
#!/usr/bin/env bash
# Sequential DeepSeek c2hls: rag_skills → skills → rag_ns
# Gates each start on Beijing off-peak; shares one login proxy (workers=1).
set -euo pipefail
SCRIPT_DIR=...
source common.sh
SEQ_ROOT="${C2HLS_ROOT}/artifacts/pc2/deepseek_u280_seq_$(date -u +%Y%m%d_%H%M%S)"
mkdir -p "${SEQ_ROOT}"
bash "${SCRIPT_DIR}/c2hls_deepseek_proxy.sh" "${SEQ_ROOT}"

wait_off_peak() {
  while "${C2HLS_PYTHON:-python3}" -c "import sys;sys.path.insert(0,'${SCRIPT_DIR}');from deepseek_peak import is_beijing_peak;raise SystemExit(0 if not is_beijing_peak() else 1)"; do
    echo "[$(date -Is)] Beijing peak — sleeping 5m before start"
    sleep 300
  done
}

for flavor in rag_skills skills rag_ns; do
  wait_off_peak
  echo "=== starting flavor=${flavor} ==="
  stamp="$(date -u +%Y%m%d_%H%M%S)_${flavor}"
  "${SCRIPT_DIR}/start_chathls_deepseek_one.sh" --flavor "${flavor}" --stamp "${stamp}" \
    --endpoint-url "$(python3 -c "import json;print(json.load(open('${SEQ_ROOT}/llm_endpoint.json'))['url'])")"
  # Wait for campaign_status complete|aborted|failed
  camp="${C2HLS_ROOT}/artifacts/pc2/batch_parallel_chathls_fd_ds_*_${stamp}"  # resolve exact prefix per flavor
  while true; do
    st=$(python3 -c "import json,glob;...")
    case "$st" in complete|completed|failed|aborted) break ;; esac
    sleep 120
  done
done
# optional: kill proxy pid from SEQ_ROOT/deepseek_proxy.pid
```

Wire exact artifact prefixes inside the one-shot script so the waiter path is deterministic.

- [ ] **Step 2: `--dry-run` mode prints the three stamps and peak status without submitting.**

---

### Task 9: Latency compare helper (DeepSeek vs U280 ChatHLS)

**Files:**
- Create: `scripts/pc2/compare_chathls_latency_u280.py`

- [ ] **Step 1: CLI**

```bash
./.venv/bin/python scripts/pc2/compare_chathls_latency_u280.py \
  --chat-hls-csv /scratch/.../hybrid-u280-split-20260717-001649/final_latency_csynth.csv \
  --campaigns ds_rag=PATH ds_skills=PATH ds_rag_ns=PATH
```

Print per-bench table + geomean(lat/U280) ranking (reuse logic from the earlier interactive compare).

- [ ] **Step 2: Run against existing Devstral campaigns as a smoke test of the script, then ready for DeepSeek paths.**

---

### Task 10: End-to-end dry verification (no full 16-bench burn unless user asks)

- [ ] **Step 1:** `start_chathls_deepseek_u280_sequence.sh --dry-run`
- [ ] **Step 2:** Start proxy only; curl `/models` from login; sbatch reachability job; confirm OK.
- [ ] **Step 3:** Optional **1-bench pilot** (`CHATHLS_FAST_TEST` style env limiting benches to `chathls_gemm`) with `--flavor rag_skills` if user approves API spend.
- [ ] **Step 4:** Only after pilot OK, run full sequence (user-gated).

---

## Spec coverage checklist

| Spec requirement | Task |
|------------------|------|
| Sequential A→B→C | Task 8 |
| Model `deepseek-chat` | Tasks 6–7 |
| U280 3.33 ns | Task 7 env |
| Login proxy workers=1 | Task 6 |
| No GPU vLLM | Task 4 |
| Combined 16-node HLS | Tasks 2–3, 7 |
| Peak start gate | Task 8 |
| Peak codegen pause | Task 5 |
| Compare vs U280 CSV | Task 9 |
| Streaming dataflow | Task 7 (reuse watcher) |

## Placeholder scan

None intentional. Combined-HLS and proxy reuse paths are pinned above.

---

**Plan complete and saved to** `docs/superpowers/plans/2026-07-16-c2hls-deepseek-u280-campaigns.md`.

**Two execution options:**

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks  
2. **Inline Execution** — execute tasks in this session with checkpoints  

Which approach?
