# ChatHLS Hybrid Backend + Devstral Escalation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `hybrid` ChatHLS LLM backend (local HLSFixer/HLSTuner + Devstral/OpenAI-compatible for transform and `debug_multi`), plus PC2 Slurm scripts to smoke-test `benchmark_optimization/gemm` on U280 with Vitis 2023.2.

**Architecture:** Extend `LLMAdapter` so each call site chooses HF vs HTTP by role when `llm_backend=hybrid`. Serve Devstral with vLLM on 4×H100; run ChatHLS+Vitis on a separate GPU compute job that reads `llm_endpoint.json`. Patch FPGA part only in a run-local gemm copy.

**Tech Stack:** Python 3.11, Hugging Face `transformers`/`torch`, OpenAI-compatible HTTP (vLLM), Slurm, Vitis HLS 2023.2, Alveo U280 (`xcu280-fsvh2892-2L-e`).

**Spec:** `/pc2/users/h/haqc2/docs/superpowers/specs/2026-07-10-chathls-hybrid-devstral-design.md`

**Roots:**
- ChatHLS: `/scratch/hpc-prf-llmfpga/asa582/projects/test-chathls/ChatHLS-ACL-26`
- Models HF: `/scratch/hpc-prf-llmfpga/asa582/projects/test-chathls/models`
- Devstral: `/scratch/hpc-prf-llmfpga/asa582/projects/devstral2`
- c2hls PC2 patterns: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/scripts/pc2`

**Commits:** Do **not** auto-commit unless the user explicitly asks. Skip commit steps or stop and ask.

---

## File map

| Path | Responsibility |
|------|----------------|
| `ChatHLS-ACL-26/src/chathls/config.py` | `hybrid` backend resolution; general model + env loading |
| `ChatHLS-ACL-26/src/chathls/adapter.py` | Per-role HF vs HTTP routing; API key optional |
| `ChatHLS-ACL-26/src/chathls/cli.py` | `--llm-backend hybrid` |
| `ChatHLS-ACL-26/tests/test_hybrid_routing.py` | Unit tests with mocked invoke paths |
| `ChatHLS-ACL-26/scripts/prepare_u280_gemm.sh` | Copy gemm + patch `set_part` to U280 |
| `ChatHLS-ACL-26/scripts/pc2/gpu_serve_devstral.sbatch.sh` | 4×H100 vLLM Devstral + write endpoint JSON |
| `ChatHLS-ACL-26/scripts/pc2/compute_gemm_hybrid.sbatch.sh` | Wait for endpoint; run hybrid gemm smoke |
| `ChatHLS-ACL-26/scripts/pc2/submit_hybrid_gemm_smoke.sh` | Submit GPU + compute with dependency |
| `ChatHLS-ACL-26/docs/HYBRID_BACKEND.md` | Operator docs for hybrid + PC2 |

---

### Task 1: Failing unit test for hybrid routing

**Files:**
- Create: `ChatHLS-ACL-26/tests/test_hybrid_routing.py`
- Create: `ChatHLS-ACL-26/tests/__init__.py` (empty)

- [ ] **Step 1: Add test file that expects hybrid role routing**

```python
# tests/test_hybrid_routing.py
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from chathls.adapter import LLMAdapter
from chathls.config import ChatHLSConfig
from chathls.models import StageResult


def _hybrid_adapter(tmp_path: Path) -> LLMAdapter:
    fixer = tmp_path / "ChatHLS-HLSFixer"
    tuner = tmp_path / "ChatHLS-HLSTuner"
    fixer.mkdir()
    tuner.mkdir()
    (fixer / "config.json").write_text("{}")
    (tuner / "config.json").write_text("{}")
    cfg = ChatHLSConfig.from_repo_root(
        tmp_path,
        llm_backend="hybrid",
        api_key="EMPTY",
        base_url="http://127.0.0.1:8000/v1",
        transform_model="mistralai/Devstral-2-123B-Instruct-2512",
        debug_analysis_model=str(fixer),
        debug_modify_model=str(fixer),
        optimize_analysis_model=str(tuner),
        optimize_modify_model=str(tuner),
    )
    return LLMAdapter.from_config(cfg)


def test_hybrid_debug_uses_hf_for_first_attempt(tmp_path: Path) -> None:
    adapter = _hybrid_adapter(tmp_path)
    failure = StageResult(stage="csyn", passed=False, details=["ERROR: bad pragma"])
    with patch.object(adapter, "_invoke_huggingface", return_value="fix it") as hf, patch.object(
        adapter, "_invoke_api", return_value="should not call"
    ) as api:
        # analysis returns text; modify returns code fence
        hf.side_effect = ["suggestion", "```cpp\nint x;\n```"]
        out = adapter.debug("int x;", failure, "int x;")
    assert "int x;" in out
    assert hf.call_count == 2
    assert api.call_count == 0


def test_hybrid_debug_multi_uses_api_then_hf_modify(tmp_path: Path) -> None:
    adapter = _hybrid_adapter(tmp_path)
    failure = StageResult(stage="csyn", passed=False, details=["ERROR: unknown"])
    with patch.object(adapter, "_invoke_huggingface", return_value="```cpp\nint y;\n```") as hf, patch.object(
        adapter, "_invoke_api", return_value="api suggestion"
    ) as api:
        out = adapter.debug_multi("int x;", failure)
    assert "int y;" in out
    # 3 analysis + 1 score via API
    assert api.call_count == 4
    # final modify via HF
    assert hf.call_count == 1


def test_hybrid_allows_empty_api_key(tmp_path: Path) -> None:
    adapter = _hybrid_adapter(tmp_path)
    assert adapter.api_key in ("", "EMPTY")
    assert adapter.base_url.endswith("/v1")
```

- [ ] **Step 2: Run test and confirm it fails (hybrid not implemented)**

```bash
cd /scratch/hpc-prf-llmfpga/asa582/projects/test-chathls/ChatHLS-ACL-26
PYTHONPATH=src python -m pytest tests/test_hybrid_routing.py -v
```

Expected: FAIL (import/config/`hybrid` not accepted, or routing wrong).

---

### Task 2: Config — `hybrid` backend + env wiring

**Files:**
- Modify: `ChatHLS-ACL-26/src/chathls/config.py`

- [ ] **Step 1: Add general-model defaults and hybrid resolution**

In `config.py`, add near the other defaults:

```python
DEFAULT_GENERAL_MODEL = os.environ.get(
    "CHATHLS_GENERAL_MODEL",
    "mistralai/Devstral-2-123B-Instruct-2512",
).strip() or "mistralai/Devstral-2-123B-Instruct-2512"
```

Change `from_repo_root` so when `resolved_llm_backend == "hybrid"`:

```python
if resolved_llm_backend == "hybrid":
    resolved_transform = (
        transform_model
        if transform_model != DEFAULT_TRANSFORM_MODEL
        else DEFAULT_GENERAL_MODEL
    )
    resolved_debug_analysis = debug_analysis_model or hf_debug
    resolved_debug_modify = debug_modify_model or hf_debug
    resolved_debug_score = debug_score_model or DEFAULT_GENERAL_MODEL
    resolved_optimize_analysis = optimize_analysis_model or hf_optimize
    resolved_optimize_modify = optimize_modify_model or hf_optimize
    # multi-end analysis agents use general model
    resolved_debug_analysis_model_0 = debug_analysis_model_0 or DEFAULT_GENERAL_MODEL
    resolved_debug_analysis_model_1 = debug_analysis_model_1 or DEFAULT_GENERAL_MODEL
    resolved_debug_analysis_model_2 = debug_analysis_model_2 or DEFAULT_GENERAL_MODEL
elif resolved_llm_backend in ("hf", "agent"):
    # existing hf/agent block unchanged
    ...
else:
    # existing api block unchanged
    ...
```

Also resolve API credentials from env when callers pass defaults:

```python
resolved_api_key = (
    api_key
    or os.environ.get("CHATHLS_API_KEY", "").strip()
    or os.environ.get("OPENAI_API_KEY", "").strip()
    or DEFAULT_OPENAI_API_KEY
)
resolved_base_url = (
    base_url
    or os.environ.get("CHATHLS_API_BASE", "").strip()
    or os.environ.get("OPENAI_BASE_URL", "").strip()
    or os.environ.get("OPENAI_API_BASE", "").strip()
    or DEFAULT_OPENAI_API_BASE
)
```

Pass `api_key=resolved_api_key`, `base_url=resolved_base_url.rstrip("/")` into the dataclass.

Keep `DEFAULT_VITIS_VERSION` / env default path such that `CHATHLS_VITIS_VERSION=2023.2` is the PC2 operator default (document in scripts; do not hard-break Otus users who still use 2021.2 for rodinia).

- [ ] **Step 2: Re-run unit tests** — still expected FAIL until adapter/CLI updated.

---

### Task 3: Adapter — hybrid invoke routing + optional API key

**Files:**
- Modify: `ChatHLS-ACL-26/src/chathls/adapter.py`

- [ ] **Step 1: Allow empty/`EMPTY` key when base_url is set**

Replace `from_config` guard:

```python
@classmethod
def from_config(cls, config: ChatHLSConfig) -> "LLMAdapter":
    needs_http = config.llm_backend in ("api", "hybrid", "agent")
    if config.llm_backend == "api" and (not config.base_url):
        raise RuntimeError("Missing API base URL (CHATHLS_API_BASE / OPENAI_BASE_URL)")
    if config.llm_backend == "hybrid" and (not config.base_url):
        raise RuntimeError("hybrid backend requires CHATHLS_API_BASE or OPENAI_BASE_URL")
    if config.llm_backend == "api" and not config.api_key:
        # allow EMPTY for local vLLM
        pass
    if config.llm_backend not in ("hf", "agent", "hybrid", "api"):
        raise RuntimeError(f"Unknown llm_backend: {config.llm_backend}")
    ...
```

- [ ] **Step 2: Add transport helper**

```python
def _model_uses_http(self, model: str) -> bool:
    if self.llm_backend == "api":
        return True
    if self.llm_backend == "hf":
        return False
    if self.llm_backend != "hybrid":
        return False
    # hybrid: local fine-tune dirs / hub ids for Fixer/Tuner stay on HF
    name = model.lower()
    if "hlsfixer" in name or "hlstuner" in name:
        return False
    if Path(model).is_dir():
        return False
    return True

def _invoke_llm(self, messages: list[dict[str, str]], model: str) -> str:
    if self.llm_backend == "agent":
        raise RuntimeError("Use optimize/debug paths for agent backend, not _invoke_llm")
    if self.llm_backend == "hybrid":
        if self._model_uses_http(model):
            return self._invoke_api(messages, model)
        return self._invoke_huggingface(messages, model)
    if self.llm_backend == "hf":
        return self._invoke_huggingface(messages, model)
    return self._invoke_api(messages, model)
```

In `_invoke_api`, use `self.api_key or "EMPTY"` in the Authorization header.

- [ ] **Step 3: Ensure `debug_multi` models are general for analysis/score**

No change to `debug_multi` control flow if config already sets `debug_analysis_model_{0,1,2}` and `debug_score_model` to the general model under hybrid (Task 2). First-attempt `debug()` keeps Fixer for both analysis and modify.

- [ ] **Step 4: Run unit tests**

```bash
cd /scratch/hpc-prf-llmfpga/asa582/projects/test-chathls/ChatHLS-ACL-26
PYTHONPATH=src python -m pytest tests/test_hybrid_routing.py -v
```

Expected: PASS.

---

### Task 4: CLI — accept `hybrid`

**Files:**
- Modify: `ChatHLS-ACL-26/src/chathls/cli.py`

- [ ] **Step 1: Extend choices**

```python
parser.add_argument(
    "--llm-backend",
    choices=["api", "hf", "agent", "hybrid"],
    ...
    help="LLM backend: api, hf, agent, or hybrid (HF Fixer/Tuner + HTTP general model)",
)
```

Same for `--analysis-backend` choices.

- [ ] **Step 2: Smoke import**

```bash
PYTHONPATH=src python -m chathls --help | grep hybrid
```

Expected: help text lists `hybrid`.

---

### Task 5: U280 gemm prepare script

**Files:**
- Create: `ChatHLS-ACL-26/scripts/prepare_u280_gemm.sh`

- [ ] **Step 1: Write script**

```bash
#!/usr/bin/env bash
# Copy benchmark_optimization/gemm and force Alveo U280 part.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${REPO_ROOT}/benchmark/benchmark_optimization/gemm"
OUT="${1:-${REPO_ROOT}/artifacts/u280_gemm}"
PART="${CHATHLS_FPGA_PART:-xcu280-fsvh2892-2L-e}"

rm -rf "${OUT}"
mkdir -p "$(dirname "${OUT}")"
cp -a "${SRC}" "${OUT}"
python3 - <<PY
from pathlib import Path
import re
tcl = Path("${OUT}") / "run_hls.tcl"
text = tcl.read_text()
text2, n = re.subn(
    r"set_part\s*\{[^}]*\}",
    "set_part {${PART}}",
    text,
    count=1,
)
if n != 1:
    raise SystemExit(f"failed to patch set_part in {tcl}")
tcl.write_text(text2)
print(f"prepared {tcl} with set_part {{{'${PART}'}}}")
PY
```

- [ ] **Step 2: Run and verify**

```bash
bash scripts/prepare_u280_gemm.sh
grep set_part artifacts/u280_gemm/run_hls.tcl
```

Expected: `set_part {xcu280-fsvh2892-2L-e}`

---

### Task 6: GPU serve sbatch (Devstral)

**Files:**
- Create: `ChatHLS-ACL-26/scripts/pc2/gpu_serve_devstral.sbatch.sh`

- [ ] **Step 1: Write sbatch that mirrors `devstral2/serve_fp8_single_node.slurm` and writes endpoint JSON**

```bash
#!/bin/bash
#SBATCH -J chathls-devstral
#SBATCH -A hpc-prf-llmfpga
#SBATCH -p gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH -t 8:00:00
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err

set -euo pipefail
SESSION_DIR="${CHATHLS_SESSION_DIR:?set CHATHLS_SESSION_DIR}"
DEVSTRAL_ROOT="${DEVSTRAL_ROOT:-/scratch/hpc-prf-llmfpga/asa582/projects/devstral2}"
MODEL_PATH="${SERVE_MODEL_PATH:-${DEVSTRAL_ROOT}/models/Devstral-2-123B-Instruct-2512}"
SERVED_NAME="${CHATHLS_GENERAL_MODEL:-mistralai/Devstral-2-123B-Instruct-2512}"
PORT="${CHATHLS_LLM_PORT:-8000}"
TP="${TENSOR_PARALLEL_SIZE:-4}"
ENDPOINT_FILE="${SESSION_DIR}/llm_endpoint.json"

mkdir -p "${SESSION_DIR}"
cd "${DEVSTRAL_ROOT}"
# shellcheck disable=SC1091
source ./load_gpu_modules.sh
# shellcheck disable=SC1091
source ./ensure_venv.sh

HOST="$(hostname -s)"
vllm serve "${MODEL_PATH}" \
  --tensor-parallel-size "${TP}" \
  --tool-call-parser mistral \
  --enable-auto-tool-choice \
  --served-model-name "${SERVED_NAME}" \
  --host 0.0.0.0 \
  --port "${PORT}" &
SERVE_PID=$!

for _ in $(seq 1 180); do
  if curl -sf --max-time 5 "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then
    break
  fi
  sleep 10
done

python3 - <<PY
import json
from pathlib import Path
Path("${ENDPOINT_FILE}").write_text(json.dumps({
    "url": "http://${HOST}:${PORT}/v1",
    "model": "${SERVED_NAME}",
    "job_id": "${SLURM_JOB_ID}",
    "host": "${HOST}",
    "port": ${PORT},
}, indent=2) + "\n")
print("wrote ${ENDPOINT_FILE}")
PY

wait "${SERVE_PID}"
```

- [ ] **Step 2: `chmod +x` the script**

```bash
chmod +x scripts/pc2/gpu_serve_devstral.sbatch.sh
```

---

### Task 7: Compute sbatch + submit helper

**Files:**
- Create: `ChatHLS-ACL-26/scripts/pc2/compute_gemm_hybrid.sbatch.sh`
- Create: `ChatHLS-ACL-26/scripts/pc2/submit_hybrid_gemm_smoke.sh`

- [ ] **Step 1: Compute job**

```bash
#!/bin/bash
#SBATCH -J chathls-gemm-hyb
#SBATCH -A hpc-prf-llmfpga
#SBATCH -p gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err

set -euo pipefail
REPO_ROOT="${CHATHLS_ROOT:-${SLURM_SUBMIT_DIR}}"
PROJECT_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
SESSION_DIR="${CHATHLS_SESSION_DIR:?}"
ENDPOINT_FILE="${SESSION_DIR}/llm_endpoint.json"
export CHATHLS_VITIS_VERSION="${CHATHLS_VITIS_VERSION:-2023.2}"
export CHATHLS_FPGA_PART="${CHATHLS_FPGA_PART:-xcu280-fsvh2892-2L-e}"
export CHATHLS_LLM_BACKEND=hybrid
export CHATHLS_MODELS_DIR="${CHATHLS_MODELS_DIR:-${PROJECT_ROOT}/models}"
export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/hf_cache}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export CHATHLS_API_KEY="${CHATHLS_API_KEY:-EMPTY}"

cd "${REPO_ROOT}"
module purge 2>/dev/null || true
module load lang Python/3.11.5-GCCcore-13.2.0 2>/dev/null || true
# Prefer c2hls PC2 Vitis 2023.2 + U280 setup when available
if [[ -f /scratch/hpc-prf-llmfpga/asa582/projects/c2hls/scripts/pc2/setup_vitis_env.sh ]]; then
  # shellcheck disable=SC1091
  source /scratch/hpc-prf-llmfpga/asa582/projects/c2hls/scripts/pc2/common.sh
  # shellcheck disable=SC1091
  source /scratch/hpc-prf-llmfpga/asa582/projects/c2hls/scripts/pc2/setup_vitis_env.sh
  pc2_setup_vitis_env
fi
# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/setup_vitis_hls.sh"
setup_vitis_hls || exit 1

# Wait up to 3h for Devstral endpoint
deadline=$((SECONDS + 10800))
while (( SECONDS < deadline )); do
  if [[ -f "${ENDPOINT_FILE}" ]]; then
    export OPENAI_BASE_URL="$(python3 -c "import json;print(json.load(open('${ENDPOINT_FILE}'))['url'])")"
    export CHATHLS_API_BASE="${OPENAI_BASE_URL}"
    if curl -sf --max-time 5 "${OPENAI_BASE_URL}/models" >/dev/null; then
      break
    fi
  fi
  sleep 30
done
[[ -n "${OPENAI_BASE_URL:-}" ]] || { echo "LLM endpoint not ready"; exit 2; }

if [[ ! -d "${REPO_ROOT}/.venv" ]]; then
  python -m venv "${REPO_ROOT}/.venv"
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.venv/bin/activate"
  pip install -r requirements.txt
else
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.venv/bin/activate"
fi

bash "${REPO_ROOT}/scripts/prepare_u280_gemm.sh" "${SESSION_DIR}/u280_gemm"

./run_chathls.sh \
  --repo-root "${REPO_ROOT}" \
  --llm-backend hybrid \
  --project-dir "${SESSION_DIR}/u280_gemm" \
  --kernel-name gemm \
  --top-function gemm \
  --source-file gemm.cpp \
  --run-name opt-gemm-hybrid-u280 \
  --max-optimization-rounds "${CHATHLS_MAX_OPTIMIZATION_ROUNDS:-1}" \
  --max-debug-attempts "${CHATHLS_MAX_DEBUG_ATTEMPTS:-2}" \
  --timeout "${CHATHLS_TIMEOUT:-7200}"
```

- [ ] **Step 2: Submit helper**

```bash
#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SESSION_DIR="${CHATHLS_SESSION_DIR:-${REPO_ROOT}/artifacts/pc2/sessions/hybrid-gemm-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "${SESSION_DIR}"
export CHATHLS_SESSION_DIR="${SESSION_DIR}"
export CHATHLS_ROOT="${REPO_ROOT}"

GPU_JOB=$(sbatch --parsable \
  --export=ALL,CHATHLS_SESSION_DIR,CHATHLS_ROOT,DEVSTRAL_ROOT \
  --output="${SESSION_DIR}/slurm-gpu-%j.out" \
  --error="${SESSION_DIR}/slurm-gpu-%j.err" \
  "${REPO_ROOT}/scripts/pc2/gpu_serve_devstral.sbatch.sh")

COMPUTE_JOB=$(sbatch --parsable \
  --dependency=after:${GPU_JOB} \
  --export=ALL,CHATHLS_SESSION_DIR,CHATHLS_ROOT,CHATHLS_VITIS_VERSION=2023.2,CHATHLS_FPGA_PART=xcu280-fsvh2892-2L-e \
  --output="${SESSION_DIR}/slurm-compute-%j.out" \
  --error="${SESSION_DIR}/slurm-compute-%j.err" \
  "${REPO_ROOT}/scripts/pc2/compute_gemm_hybrid.sbatch.sh")

echo "session=${SESSION_DIR}"
echo "gpu_job=${GPU_JOB}"
echo "compute_job=${COMPUTE_JOB}"
echo "${GPU_JOB}" > "${SESSION_DIR}/gpu_job_id"
echo "${COMPUTE_JOB}" > "${SESSION_DIR}/compute_job_id"
```

- [ ] **Step 3: chmod +x**

```bash
chmod +x scripts/pc2/compute_gemm_hybrid.sbatch.sh scripts/pc2/submit_hybrid_gemm_smoke.sh scripts/prepare_u280_gemm.sh
```

---

### Task 8: Operator docs

**Files:**
- Create: `ChatHLS-ACL-26/docs/HYBRID_BACKEND.md`
- Modify: `ChatHLS-ACL-26/README.md` (short pointer only)

- [ ] **Step 1: Write `docs/HYBRID_BACKEND.md`** covering routing table, env vars, `bash scripts/pc2/submit_hybrid_gemm_smoke.sh`, Vitis 2023.2, U280, and commercial-key alternative (`CHATHLS_API_KEY` + hosted base).

- [ ] **Step 2: Add a README blurb** under Configuration pointing to `docs/HYBRID_BACKEND.md`.

---

### Task 9: Verification checklist (before claiming done)

- [ ] **Step 1: Unit tests pass**

```bash
cd /scratch/hpc-prf-llmfpga/asa582/projects/test-chathls/ChatHLS-ACL-26
PYTHONPATH=src python -m pytest tests/test_hybrid_routing.py -v
```

- [ ] **Step 2: Prepare gemm U280 copy and grep part**

```bash
bash scripts/prepare_u280_gemm.sh /tmp/u280_gemm_check
grep 'xcu280' /tmp/u280_gemm_check/run_hls.tcl
```

- [ ] **Step 3: Submit smoke only when user asks**

```bash
bash scripts/pc2/submit_hybrid_gemm_smoke.sh
```

Then monitor `CHATHLS_SESSION_DIR` logs until `runs/opt-gemm-hybrid-u280-*/summary.json` exists. Do **not** submit Slurm jobs without explicit user approval in the execution session.

---

## Spec coverage self-check

| Spec requirement | Task |
|------------------|------|
| Hybrid routing Fixer/Tuner local | 2, 3 |
| Transform + debug_multi via HTTP/Devstral | 2, 3 |
| API key optional / EMPTY | 3 |
| Commercial key still works | 3, 8 |
| PC2 GPU+compute split | 6, 7 |
| Vitis 2023.2 | 7 |
| U280 gemm smoke | 5, 7, 9 |
| No permanent suite TCL rewrite | 5 |
| Preserve api/hf/agent | 2, 3, 4 |

## Placeholder scan

None intentional. If `ensure_venv.sh` / `load_gpu_modules.sh` names differ under `devstral2/`, adjust Task 6 to the actual scripts present (`ensure_venv.sh`, `load_gpu_modules.sh` confirmed on disk).
