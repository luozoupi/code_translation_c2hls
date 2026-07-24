# HLSFactory + MachSuite DeepSeek U280 Dual-Track Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port all 46 `hlsfactory_*` + `machsuite_*` benches from `c2hls/benchmarks` into ChatHLS hybrid form, then run parallel DeepSeek U280 campaigns: c2hls RAG2+skills and ChatHLS hybrid on the same kernels.

**Architecture:** Exporter writes prefixed dirs under ChatHLS `benchmark_optimization/`. c2hls gets two DeepSeek external_llm batch_parallel starters (machsuite flash→dataflow, hlsfactory flash→dataflow). ChatHLS hybrid submit honors a 46-bench list. Shared login DeepSeek proxy; umbrella launcher smokes then submits both tracks.

**Tech Stack:** bash, Python 3, pytest, Slurm, Vitis HLS, DeepSeek OpenAI-compatible proxy, existing c2hls batch_parallel + ChatHLS hybrid PC2 scripts.

**Spec:** `docs/superpowers/specs/2026-07-18-hlsfactory-machsuite-deepseek-u280-dual-track-design.md`

---

## File map

| Path | Role |
|------|------|
| `c2hls/scripts/pc2/export_c2hls_bench_to_chathls.py` | Port one/all benches → ChatHLS layout |
| `c2hls/scripts/pc2/c2hls_port_loop_labels.py` | Deterministic loop label injection + kernel_info lines |
| `c2hls/tests/test_export_c2hls_bench_to_chathls.py` | Exporter + label tests |
| `c2hls/scripts/pc2/hlsfactory_28_benches.txt` | Bench list |
| `c2hls/scripts/pc2/c2hls_port_46_benches.txt` | Combined 46 list (also copied/symlinked for ChatHLS) |
| `c2hls/scripts/pc2/batch_parallel_machsuite_deepseek_u280.json` | MachSuite DeepSeek config |
| `c2hls/scripts/pc2/batch_parallel_hlsfactory_deepseek_u280.json` | HLSFactory DeepSeek config |
| `c2hls/scripts/pc2/start_machsuite_deepseek_rag2_skills_u280.sh` | MachSuite starter |
| `c2hls/scripts/pc2/start_hlsfactory_deepseek_rag2_skills_u280.sh` | HLSFactory starter |
| `c2hls/scripts/pc2/wait_hlsfactory_flash_then_dataflow.sh` | Thin clone of machsuite waiter for hlsfactory campaign |
| `c2hls/scripts/pc2/start_c2hls_chathls_dual_track_u280.sh` | Umbrella: proxy + export + smoke + parallel submit |
| `ChatHLS-ACL-26/scripts/pc2/c2hls_port_46_benches.txt` | Bench list for hybrid |
| `ChatHLS-ACL-26/scripts/pc2/submit_chathls_hybrid_batch_parallel.sh` | Honor pre-set `CHATHLS_BENCH_LIST` / optional external proxy URL |
| `ChatHLS-ACL-26/scripts/pc2/submit_chathls_hybrid_c2hls_port_u280.sh` | Wrapper: 46 list + session prefix + shared proxy |
| `c2hls/scripts/pc2/compare_c2hls_chathls_port_u280.py` | Paired latency/resources report |
| `c2hls/docs/pc2/2026-07-18-hlsfactory-machsuite-deepseek-dual-track.md` | Generated compare (after runs) |

---

### Task 1: Loop-label helper + failing tests

**Files:**
- Create: `scripts/pc2/c2hls_port_loop_labels.py`
- Create: `tests/test_export_c2hls_bench_to_chathls.py`

- [ ] **Step 1: Write failing tests for label injection**

```python
# tests/test_export_c2hls_bench_to_chathls.py
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "pc2"))

from c2hls_port_loop_labels import inject_loop_labels, build_kernel_info


SAMPLE = '''
void kernel_atax(double A[38][42], double x[42], double y[42], double tmp[38]) {
  int i, j;
  for (i = 0; i < 42; i++)
    y[i] = 0;
  for (i = 0; i < 38; i++) {
    tmp[i] = 0.0;
    for (j = 0; j < 42; j++)
      tmp[i] = tmp[i] + A[i][j] * x[j];
    for (j = 0; j < 42; j++)
      y[j] = y[j] + A[i][j] * tmp[i];
  }
}
'''


def test_inject_loop_labels_is_deterministic_and_numbered():
    out1, n1 = inject_loop_labels(SAMPLE, top="kernel_atax")
    out2, n2 = inject_loop_labels(SAMPLE, top="kernel_atax")
    assert n1 == 4 and n2 == 4
    assert out1 == out2
    assert "L1:" in out1 and "L4:" in out1
    assert out1.index("L1:") < out1.index("L2:")


def test_build_kernel_info_lists_loops_and_top():
    labeled, _ = inject_loop_labels(SAMPLE, top="kernel_atax")
    info = build_kernel_info(labeled, top="kernel_atax")
    lines = info.strip().splitlines()
    assert lines[0] == "kernel_atax"
    assert any(l.startswith("L1,loop,") for l in lines)
    assert sum(1 for l in lines if ",loop," in l) == 4
```

- [ ] **Step 2: Run tests — expect fail (module missing)**

```bash
cd /scratch/hpc-prf-llmfpga/asa582/projects/c2hls
./.venv/bin/python -m pytest tests/test_export_c2hls_bench_to_chathls.py -v
```

Expected: `ModuleNotFoundError` or import error for `c2hls_port_loop_labels`.

- [ ] **Step 3: Implement `c2hls_port_loop_labels.py`**

```python
# scripts/pc2/c2hls_port_loop_labels.py
"""Deterministic HLS loop label injection for ChatHLS kernel_info.txt."""
from __future__ import annotations

import re
from typing import Tuple

_FOR_WHILE = re.compile(r"^(\s*)(for\s*\(|while\s*\()")


def inject_loop_labels(source: str, *, top: str) -> Tuple[str, int]:
    """Insert L1:, L2:, ... before for/while lines inside the top function body.

    Only labels loops that are not already labeled (line does not already match
    ``^\\s*L\\d+:``). Returns (new_source, label_count).
    """
    lines = source.splitlines(keepends=True)
    # Find function start: loose match on top name
    start = None
    for i, line in enumerate(lines):
        if re.search(rf"\b{re.escape(top)}\s*\(", line):
            start = i
            break
    if start is None:
        return source, 0

    # Brace depth from first { after start
    body_start = None
    depth = 0
    for i in range(start, len(lines)):
        depth += lines[i].count("{") - lines[i].count("}")
        if "{" in lines[i] and body_start is None:
            body_start = i
            break
    if body_start is None:
        return source, 0

    out: list[str] = []
    n = 0
    depth = 0
    in_body = False
    for i, line in enumerate(lines):
        if i == body_start:
            in_body = True
        if in_body:
            depth += line.count("{") - line.count("}")
        already = bool(re.match(r"^\s*L\d+:", line))
        if in_body and depth >= 1 and not already and _FOR_WHILE.match(line):
            n += 1
            indent = re.match(r"^(\s*)", line).group(1)
            out.append(f"{indent}L{n}: {line.lstrip()}")
        else:
            out.append(line)
        if in_body and i > body_start and depth <= 0:
            in_body = False
    return "".join(out), n


def build_kernel_info(labeled_source: str, *, top: str) -> str:
    """Build ChatHLS kernel_info.txt from labeled source."""
    rows = [top]
    for i, line in enumerate(labeled_source.splitlines(), start=1):
        m = re.match(r"^\s*(L\d+):\s*(for|while)\b", line)
        if m:
            rows.append(f"{m.group(1)},loop,{i}")
    # Best-effort array params from top signature
    sig = re.search(rf"{re.escape(top)}\s*\((.*?)\)\s*\{{", labeled_source, re.S)
    if sig:
        for part in sig.group(1).split(","):
            part = part.strip()
            am = re.search(r"\b([A-Za-z_]\w*)\s*(\[|$)", part)
            if am and "[" in part:
                name = am.group(1)
                # attach to first loop line if any, else line 0
                loop_line = next(
                    (r.split(",")[-1] for r in rows[1:] if r.startswith("L")),
                    "0",
                )
                rows.append(f"{name},array,{loop_line}")
    return "\n".join(rows) + "\n"
```

- [ ] **Step 4: Re-run tests — expect pass**

```bash
./.venv/bin/python -m pytest tests/test_export_c2hls_bench_to_chathls.py -v
```

Expected: both tests PASS.

- [ ] **Step 5: Commit (only if user requested commits)**

```bash
git add scripts/pc2/c2hls_port_loop_labels.py tests/test_export_c2hls_bench_to_chathls.py
git commit -m "$(cat <<'EOF'
Add deterministic loop-label helper for ChatHLS bench port.

EOF
)"
```

---

### Task 2: Exporter script + integration test on real benches

**Files:**
- Create: `scripts/pc2/export_c2hls_bench_to_chathls.py`
- Modify: `tests/test_export_c2hls_bench_to_chathls.py`

- [ ] **Step 1: Add failing integration test**

```python
def test_export_hlsfactory_atax_writes_chathls_layout(tmp_path):
    from export_c2hls_bench_to_chathls import export_bench

    src = REPO / "benchmarks" / "hlsfactory_atax"
    out_root = tmp_path / "benchmark_optimization"
    manifest = export_bench(src, out_root)
    dest = out_root / "hlsfactory_atax"
    assert dest.is_dir()
    top = manifest["top"]
    assert (dest / f"{top}.cpp").is_file()
    assert (dest / "kernel_info.txt").is_file()
    assert (dest / "run_hls.tcl").is_file()
    assert (dest / "port_manifest.json").is_file()
    info = (dest / "kernel_info.txt").read_text().splitlines()
    assert info[0] == top
    assert any(",loop," in line for line in info)
```

- [ ] **Step 2: Run test — expect fail**

```bash
./.venv/bin/python -m pytest tests/test_export_c2hls_bench_to_chathls.py::test_export_hlsfactory_atax_writes_chathls_layout -v
```

- [ ] **Step 3: Implement exporter**

Key behaviors in `export_c2hls_bench_to_chathls.py`:

- CLI: `--bench-dir`, `--out-root`, `--all-prefixed` (glob `benchmarks/hlsfactory_*` + `machsuite_*`)
- Read `metadata.json` → `hls_top` or `kernel_top`, baseline `hls_baseline.cpp`, headers
- Concatenate needed headers + baseline into one TU; call `inject_loop_labels`; write `<top>.cpp`
- `build_kernel_info` → `kernel_info.txt`
- Write `run_hls.tcl`:

```tcl
open_project -reset test_proj
add_files {TOP}.cpp
set_top {TOP}
open_solution -reset solution
set_part {xczu7ev-ffvc1156-2-e}
create_clock -period 10 -name default
csynth_design
exit
```

- Write `port_manifest.json` with source, top, label_count, warnings
- Dest dir name = source dir basename (prefixed)

- [ ] **Step 4: Tests pass**

```bash
./.venv/bin/python -m pytest tests/test_export_c2hls_bench_to_chathls.py -v
```

- [ ] **Step 5: Dry-export one machsuite bench**

```bash
./.venv/bin/python scripts/pc2/export_c2hls_bench_to_chathls.py \
  --bench-dir benchmarks/machsuite_gemm_ncubed \
  --out-root /tmp/chathls_port_smoke/benchmark_optimization
ls /tmp/chathls_port_smoke/benchmark_optimization/machsuite_gemm_ncubed
```

Expected: `kernel_*.cpp` or top-named cpp, `kernel_info.txt`, `run_hls.tcl`, `port_manifest.json`.

---

### Task 3: Export all 46 into ChatHLS tree + bench lists

**Files:**
- Create: `scripts/pc2/hlsfactory_28_benches.txt`
- Create: `scripts/pc2/c2hls_port_46_benches.txt`
- Create: `../test-chathls/ChatHLS-ACL-26/scripts/pc2/c2hls_port_46_benches.txt` (same content)

- [ ] **Step 1: Write list files**

```bash
cd /scratch/hpc-prf-llmfpga/asa582/projects/c2hls
ls -1 benchmarks | rg '^hlsfactory_' | sort > scripts/pc2/hlsfactory_28_benches.txt
cat scripts/pc2/machsuite_18_benches.txt scripts/pc2/hlsfactory_28_benches.txt | sort > scripts/pc2/c2hls_port_46_benches.txt
wc -l scripts/pc2/c2hls_port_46_benches.txt   # expect 46
cp scripts/pc2/c2hls_port_46_benches.txt \
  /scratch/hpc-prf-llmfpga/asa582/projects/test-chathls/ChatHLS-ACL-26/scripts/pc2/c2hls_port_46_benches.txt
```

- [ ] **Step 2: Export all 46**

```bash
CHATHLS_OPT=/scratch/hpc-prf-llmfpga/asa582/projects/test-chathls/ChatHLS-ACL-26/benchmark/benchmark_optimization
./.venv/bin/python scripts/pc2/export_c2hls_bench_to_chathls.py \
  --all-prefixed \
  --benchmarks-root benchmarks \
  --out-root "${CHATHLS_OPT}"
# verify no clobber of classic atax
test -f "${CHATHLS_OPT}/atax/atax.cpp"
test -f "${CHATHLS_OPT}/hlsfactory_atax/kernel_info.txt"
ls -d "${CHATHLS_OPT}"/hlsfactory_* "${CHATHLS_OPT}"/machsuite_* | wc -l   # 46
```

- [ ] **Step 3: Summarize label coverage**

```bash
./.venv/bin/python - <<'PY'
import json
from pathlib import Path
root=Path("/scratch/hpc-prf-llmfpga/asa582/projects/test-chathls/ChatHLS-ACL-26/benchmark/benchmark_optimization")
bad=[]
for p in sorted(root.glob("hlsfactory_*/port_manifest.json"))+sorted(root.glob("machsuite_*/port_manifest.json")):
    m=json.loads(p.read_text())
    if m.get("label_count",0)<1:
        bad.append(p.parent.name)
print("zero_label", bad)
PY
```

Expected: empty or a short known list documented in port notes; fix exporter if many zeros.

---

### Task 4: ChatHLS hybrid submit honors external bench list + shared proxy

**Files:**
- Modify: `test-chathls/ChatHLS-ACL-26/scripts/pc2/submit_chathls_hybrid_batch_parallel.sh`
- Create: `test-chathls/ChatHLS-ACL-26/scripts/pc2/submit_chathls_hybrid_c2hls_port_u280.sh`

- [ ] **Step 1: Patch bench list selection**

In `submit_chathls_hybrid_batch_parallel.sh`, replace hardcoded list assignment with:

```bash
# Prefer caller-provided list (e.g. c2hls_port_46_benches.txt); else default 16.
if [[ -n "${CHATHLS_BENCH_LIST:-}" && -f "${CHATHLS_BENCH_LIST}" ]]; then
  BENCH_LIST="${CHATHLS_BENCH_LIST}"
else
  BENCH_LIST="${REPO_ROOT}/scripts/pc2/chathls_benchmarks.txt"
fi
if [[ -n "${CHATHLS_FAST_TEST:-}" ]]; then
  printf '%s\n' gemm kernel_2mm covariance mobilenet > "${SESSION_DIR}/fast_benches.txt"
  BENCH_LIST="${SESSION_DIR}/fast_benches.txt"
fi
export CHATHLS_BENCH_LIST="${BENCH_LIST}"
```

- [ ] **Step 2: Optional shared proxy skip**

If `CHATHLS_EXTERNAL_DEEPSEEK_ENDPOINT` is set, skip starting a new proxy and write/use that URL in session `llm` env the same way `start_deepseek_queue_proxy.sh` would (set `OPENAI_BASE_URL` / session endpoint file). Minimal approach: document that wrapper exports `OPENAI_BASE_URL` before submit and make `start_deepseek_queue_proxy.sh` a no-op when `CHATHLS_SKIP_DEEPSEEK_PROXY=1`.

```bash
# at top of proxy start section in submit script:
if [[ "${CHATHLS_SKIP_DEEPSEEK_PROXY:-0}" == "1" ]]; then
  echo "skipping DeepSeek proxy start; using OPENAI_BASE_URL=${OPENAI_BASE_URL:-unset}"
else
  bash "${REPO_ROOT}/scripts/pc2/start_deepseek_queue_proxy.sh" "${SESSION_DIR}"
fi
```

- [ ] **Step 3: Create wrapper**

```bash
#!/usr/bin/env bash
# submit_chathls_hybrid_c2hls_port_u280.sh
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/pc2/setup_chathls_u280_env.sh"
export CHATHLS_BENCH_LIST="${CHATHLS_BENCH_LIST:-${REPO_ROOT}/scripts/pc2/c2hls_port_46_benches.txt}"
export CHATHLS_SESSION_DIR="${CHATHLS_SESSION_DIR:-${REPO_ROOT}/artifacts/pc2/sessions/hybrid-u280-c2hlsport-$(date +%Y%m%d-%H%M%S)}"
# Optional: CHATHLS_SKIP_DEEPSEEK_PROXY=1 OPENAI_BASE_URL=http://login:18092/v1
exec bash "${REPO_ROOT}/scripts/pc2/submit_chathls_hybrid_batch_parallel.sh"
```

Make executable.

- [ ] **Step 4: Sanity — list maps to 46**

```bash
wc -l /scratch/hpc-prf-llmfpga/asa582/projects/test-chathls/ChatHLS-ACL-26/scripts/pc2/c2hls_port_46_benches.txt
# dry: do not sbatch yet — just bash -n the scripts
bash -n .../submit_chathls_hybrid_batch_parallel.sh
bash -n .../submit_chathls_hybrid_c2hls_port_u280.sh
```

---

### Task 5: c2hls MachSuite DeepSeek RAG2+skills U280 config + starter

**Files:**
- Create: `scripts/pc2/batch_parallel_machsuite_deepseek_u280.json`
- Create: `scripts/pc2/start_machsuite_deepseek_rag2_skills_u280.sh`

- [ ] **Step 1: Write JSON config**

Copy from `batch_parallel_machsuite_flash_dataflow.json`, then set:

```json
{
  "job_prefix": "bpmds",
  "combined_hls_nodes": true,
  "synth_nodes_per_variant": 18,
  "synth_workers_per_node": 1,
  "cosim_nodes_per_variant": 0,
  "cosim_workers_per_node": 0,
  "gpu_policy": "always_on",
  "cosim_timeout_s": 7200,
  "max_inflight_benches": 18,
  "pilot": {
    "variant": "tier_b_aav_n",
    "workflow": "tier_b_flash",
    "corpus": "tier_B_ready",
    "benches": ["/* same 18 machsuite_* */"],
    "failure_policy": "ignore",
    "model": "deepseek-chat",
    "turns": 4
  }
}
```

(Keep worker_cpus/mem from original.)

- [ ] **Step 2: Write starter** (pattern from `start_chathls_deepseek_one.sh` rag2_skills + machsuite post watcher)

Required exports:

```bash
export BATCH_PARALLEL_CONFIG="${SCRIPT_DIR}/batch_parallel_machsuite_deepseek_u280.json"
export BATCH_PARALLEL_VARIANT="tier_b_aav_n"
export BATCH_PARALLEL_ARTIFACT_PREFIX="batch_parallel_machsuite_ds_rag2"
export PC2_BATCH_JOB_PREFIX="bpmds"
export C2HLS_MODEL=deepseek-chat
export BATCH_PARALLEL_EXTERNAL_MODEL=deepseek-chat
export C2HLS_COMBINED_HLS=1
export C2HLS_RAG2=1
export C2HLS_RAG=0
export C2HLS_RAG_ENABLE=0
export C2HLS_RAG_SCRAPE=0
export C2HLS_PART=xcu280-fsvh2892-2L-e
export C2HLS_CLOCK_NS=3.33
export C2HLS_DEEPSEEK_PEAK_PAUSE=1
# RAG2 corpus paths same as start_chathls_deepseek_one.sh rag2_skills
```

Call `start_batch_parallel_campaign.sh --external-llm --external-endpoint-url "$URL"` (same flags as deepseek one-shot).

Submit post watcher:

```bash
--wrap="bash ${SCRIPT_DIR}/wait_machsuite_flash_then_dataflow.sh --campaign-root ${CAMPAIGN_ROOT} >> ${WATCH_LOG} 2>&1"
```

Support `--dry-run` and `--endpoint-url`.

- [ ] **Step 3: Dry-run**

```bash
./scripts/pc2/start_machsuite_deepseek_rag2_skills_u280.sh --dry-run --endpoint-url http://127.0.0.1:18092/v1
```

Expected: campaign root created, no Slurm jobs (or dry path of start_batch_parallel).

---

### Task 6: c2hls HLSFactory DeepSeek RAG2+skills U280 config + starter

**Files:**
- Create: `scripts/pc2/batch_parallel_hlsfactory_deepseek_u280.json`
- Create: `scripts/pc2/wait_hlsfactory_flash_then_dataflow.sh`
- Create: `scripts/pc2/start_hlsfactory_deepseek_rag2_skills_u280.sh`

- [ ] **Step 1: JSON** — clone `batch_parallel_full_aav_n.json` benches; set `model=deepseek-chat`, `combined_hls_nodes=true`, `synth_nodes_per_variant=28`, `cosim_nodes=0`, `gpu_policy=always_on`, `job_prefix=bphfds`, `workflow=flash`.

- [ ] **Step 2: Waiter** — copy `wait_machsuite_flash_then_dataflow.sh` to `wait_hlsfactory_flash_then_dataflow.sh`; only change log strings / any machsuite-only filters if present (keep export_flash_selected_bundle + post_flash_dataflow path).

- [ ] **Step 3: Starter** — same DeepSeek/RAG2/U280 env as Task 5; artifact prefix `batch_parallel_hlsfactory_ds_rag2`; post job runs `wait_hlsfactory_flash_then_dataflow.sh`.

- [ ] **Step 4: Dry-run**

```bash
./scripts/pc2/start_hlsfactory_deepseek_rag2_skills_u280.sh --dry-run --endpoint-url http://127.0.0.1:18092/v1
```

---

### Task 7: Umbrella dual-track launcher

**Files:**
- Create: `scripts/pc2/start_c2hls_chathls_dual_track_u280.sh`

- [ ] **Step 1: Implement launcher**

```bash
#!/usr/bin/env bash
# Port 46 benches, start shared DeepSeek proxy, smoke, then parallel submit:
#   - c2hls machsuite RAG2+skills DeepSeek U280
#   - c2hls hlsfactory RAG2+skills DeepSeek U280
#   - ChatHLS hybrid c2hls-port 46
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

DRY_RUN=0
SKIP_EXPORT=0
SKIP_CHATHLS=0
SKIP_C2HLS=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --skip-export) SKIP_EXPORT=1; shift ;;
    --skip-chathls) SKIP_CHATHLS=1; shift ;;
    --skip-c2hls) SKIP_C2HLS=1; shift ;;
    *) echo "unknown: $1" >&2; exit 2 ;;
  esac
done

SEQ_ROOT="${C2HLS_ROOT}/artifacts/pc2/dual_track_u280_$(date -u +%Y%m%d_%H%M%S)"
mkdir -p "${SEQ_ROOT}"
CHATHLS_ROOT="${CHATHLS_ROOT:-/scratch/hpc-prf-llmfpga/asa582/projects/test-chathls/ChatHLS-ACL-26}"

if [[ "${SKIP_EXPORT}" -eq 0 ]]; then
  "${C2HLS_PYTHON:-python3}" "${SCRIPT_DIR}/export_c2hls_bench_to_chathls.py" \
    --all-prefixed --benchmarks-root "${C2HLS_ROOT}/benchmarks" \
    --out-root "${CHATHLS_ROOT}/benchmark/benchmark_optimization"
fi

# Shared proxy
"${SCRIPT_DIR}/c2hls_deepseek_proxy.sh" "${SEQ_ROOT}"
URL="$("${C2HLS_PYTHON:-python3}" -c "import json;print(json.load(open('${SEQ_ROOT}/llm_endpoint.json'))['url'])")"

# Smoke export presence
test -f "${CHATHLS_ROOT}/benchmark/benchmark_optimization/hlsfactory_atax/kernel_info.txt"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "[dry-run] would start machsuite+hlsfactory+chathls with endpoint ${URL}"
  echo "${URL}" > "${SEQ_ROOT}/endpoint.url"
  exit 0
fi

# Parallel submits
if [[ "${SKIP_C2HLS}" -eq 0 ]]; then
  nohup "${SCRIPT_DIR}/start_machsuite_deepseek_rag2_skills_u280.sh" --endpoint-url "${URL}" \
    > "${SEQ_ROOT}/machsuite_launch.log" 2>&1 &
  echo $! > "${SEQ_ROOT}/machsuite_launcher.pid"
  nohup "${SCRIPT_DIR}/start_hlsfactory_deepseek_rag2_skills_u280.sh" --endpoint-url "${URL}" \
    > "${SEQ_ROOT}/hlsfactory_launch.log" 2>&1 &
  echo $! > "${SEQ_ROOT}/hlsfactory_launcher.pid"
fi

if [[ "${SKIP_CHATHLS}" -eq 0 ]]; then
  export CHATHLS_SKIP_DEEPSEEK_PROXY=1
  export OPENAI_BASE_URL="${URL}"
  export CHATHLS_SESSION_DIR="${CHATHLS_ROOT}/artifacts/pc2/sessions/hybrid-u280-c2hlsport-$(date +%Y%m%d-%H%M%S)"
  nohup bash "${CHATHLS_ROOT}/scripts/pc2/submit_chathls_hybrid_c2hls_port_u280.sh" \
    > "${SEQ_ROOT}/chathls_launch.log" 2>&1 &
  echo $! > "${SEQ_ROOT}/chathls_launcher.pid"
  echo "${CHATHLS_SESSION_DIR}" > "${SEQ_ROOT}/chathls_session_dir.txt"
fi

"${C2HLS_PYTHON:-python3}" - <<PY
import json, time
from pathlib import Path
p=Path("${SEQ_ROOT}")/"dual_track_state.json"
p.write_text(json.dumps({
  "seq_root": "${SEQ_ROOT}",
  "endpoint_url": "${URL}",
  "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}, indent=2)+"\n")
PY
echo "dual_track seq_root=${SEQ_ROOT}"
```

- [ ] **Step 2: Dry-run umbrella**

```bash
./scripts/pc2/start_c2hls_chathls_dual_track_u280.sh --dry-run --skip-export
```

(Use without `--skip-export` once exporter is ready; dry-run still starts proxy — if that is undesired, gate proxy behind `DRY_RUN=0` in a follow-up tweak.)

**Prefer:** in dry-run, skip real proxy and print intended actions only.

---

### Task 8: Compare helper

**Files:**
- Create: `scripts/pc2/compare_c2hls_chathls_port_u280.py`

- [ ] **Step 1: Implement CLI**

```bash
./.venv/bin/python scripts/pc2/compare_c2hls_chathls_port_u280.py \
  --chathls-latency-csv PATH/final_latency_csynth.csv \
  --chathls-resources-csv PATH/final_resources_csynth.csv \
  --c2hls-machsuite-campaign PATH \
  --c2hls-hlsfactory-campaign PATH \
  --out docs/pc2/2026-07-18-hlsfactory-machsuite-deepseek-dual-track.md
```

Reuse latency extraction patterns from `compare_chathls_latency_u280.py` (selected_report / flash_selected synth_report). Key benches by prefixed names. Emit markdown tables: latency, ratio, LUT/DSP.

- [ ] **Step 2: Unit-test ratio formatting on tiny fixtures** (optional small test with tmp json/csv).

---

### Task 9: Live launch (operator)

- [ ] **Step 1: Confirm proxy port free / API key present**

```bash
ss -ltn | rg '18092|18082' || true
bash -c 'source /scratch/hpc-prf-llmfpga/asa582/projects/test-chathls/ChatHLS-ACL-26/scripts/pc2/setup_deepseek_api.sh && echo ok_len=${#OPENAI_API_KEY}'
```

- [ ] **Step 2: Export 46 (if not done)**

```bash
./.venv/bin/python scripts/pc2/export_c2hls_bench_to_chathls.py --all-prefixed \
  --benchmarks-root benchmarks \
  --out-root /scratch/hpc-prf-llmfpga/asa582/projects/test-chathls/ChatHLS-ACL-26/benchmark/benchmark_optimization
```

- [ ] **Step 3: Smoke ChatHLS one bench (optional short)**

```bash
# single-bench hybrid smoke if a one-bench helper exists; else rely on array afterok gate
```

- [ ] **Step 4: Full dual-track launch**

```bash
nohup ./scripts/pc2/start_c2hls_chathls_dual_track_u280.sh \
  > artifacts/pc2/dual_track_u280_launch_$(date -u +%Y%m%d_%H%M%S).log 2>&1 &
```

- [ ] **Step 5: Verify jobs**

```bash
squeue -u "$USER" | rg 'bpmds|bphfds|chathls|hybrid|hf'
cat artifacts/pc2/dual_track_u280_*/dual_track_state.json
```

---

## Spec coverage checklist

| Spec requirement | Task |
|------------------|------|
| Exporter + labels + kernel_info + TCL | 1–2 |
| Keep prefixes; preserve dtypes | 2–3 |
| Export all 46; leave old 16 intact | 3 |
| ChatHLS hybrid 46 list + U280 | 4 |
| c2hls machsuite DeepSeek RAG2+skills | 5 |
| c2hls hlsfactory DeepSeek RAG2+skills + dataflow waiter | 6 |
| Shared proxy; parallel launch | 7 |
| Compare report | 8 |
| Live submit | 9 |

## Placeholder / consistency self-check

- No TBD steps; proxy skip flag names consistent: `CHATHLS_SKIP_DEEPSEEK_PROXY`, `OPENAI_BASE_URL`
- Job prefixes: `bpmds` (machsuite), `bphfds` (hlsfactory)
- Model id everywhere: `deepseek-chat`
- Part/clock: `xcu280-fsvh2892-2L-e` / `3.33`
