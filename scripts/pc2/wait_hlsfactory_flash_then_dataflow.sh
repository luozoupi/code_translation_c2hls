#!/usr/bin/env bash
# Wait for HLSFactory flash batch_parallel campaign, then:
#   1) write matrix.json from variant cells
#   2) export flash_selected_bundle
#   3) run post-flash dataflow with cosim + multi-round repairs
#      - external_llm / DeepSeek campaigns: use campaign endpoint (NO gpu_h100 vLLM)
#      - otherwise: supervised session, prefer --borrow-gpu
#   4) export dataflow_selected_bundle
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/setup_vitis_env.sh"
pc2_setup_vitis_env
cd "${C2HLS_ROOT}"

CAMPAIGN_ROOT=""
POLL_SEC="${POST_FLASH_POLL_SEC:-120}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --campaign-root) shift; CAMPAIGN_ROOT="$1"; shift ;;
    --poll-sec) shift; POLL_SEC="$1"; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "${CAMPAIGN_ROOT}" ]]; then
  echo "ERROR: --campaign-root required" >&2
  exit 2
fi

PY="${C2HLS_PYTHON:-${C2HLS_ROOT}/.venv/bin/python}"
if [[ ! -x "${PY}" ]]; then
  PY=python3
fi

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] waiting for HLSFactory campaign complete: ${CAMPAIGN_ROOT}"
while true; do
  status="$("${PY}" - <<PY
import json
from pathlib import Path
p = Path("${CAMPAIGN_ROOT}") / "campaign.json"
if not p.is_file():
    print("missing")
else:
    print(json.loads(p.read_text()).get("campaign_status", "unknown"))
PY
)"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] campaign_status=${status}"
  case "${status}" in
    complete|completed|failed|aborted) break ;;
  esac
  sleep "${POLL_SEC}"
done

if [[ "${status}" != "complete" && "${status}" != "completed" ]]; then
  echo "campaign ended with status=${status}; still exporting whatever cells exist"
fi

# Refuse hollow dataflow: wait until enough benches have a resolvable selected kernel.
MIN_EXPORTABLE="${POST_FLASH_MIN_EXPORTABLE:-1}"
EXPORT_WAIT_SEC="${POST_FLASH_EXPORT_WAIT_SEC:-7200}"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] gating on exportable flash kernels (min=${MIN_EXPORTABLE})"
if ! "${PY}" "${SCRIPT_DIR}/post_flash_export_gate.py" \
  --campaign-root "${CAMPAIGN_ROOT}" \
  --min-exportable "${MIN_EXPORTABLE}" \
  --poll-sec "${POLL_SEC}" \
  --max-wait-sec "${EXPORT_WAIT_SEC}"; then
  echo "ERROR: export gate failed; not starting post-flash dataflow" >&2
  exit 3
fi

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] building matrix.json"
"${PY}" - <<PY
import json
from pathlib import Path
from datetime import datetime, timezone

root = Path("${CAMPAIGN_ROOT}")
variant_root = root / "variants"
# One preferred cell per bench (deepseek over leftover failed-model cells).
best = {}
for cell in sorted(variant_root.glob("*/*/*")):
    if not cell.is_dir():
        continue
    bench = cell.parent.name
    if not (cell / f"{bench}_multistep_results.json").is_file() and not any(cell.glob("*_final.cpp")) and not any(cell.glob("*_selected.cpp")):
        if not (cell / "reference_validation.json").is_file() and not (cell / "pipelined").is_dir():
            continue
    score = (0 if "deepseek" in cell.name else 1, 0 if "devstral" not in cell.name else 2, cell.name)
    prev = best.get(bench)
    if prev is None or score < prev[0]:
        best[bench] = (score, {
            "bench": bench,
            "cell_dir": str(cell.resolve()),
            "status": "unknown",
            "model": cell.name,
            "variant": cell.parent.parent.name,
        })
rows = [best[b][1] for b in sorted(best)]
matrix = {
    "schema": "batch_parallel_matrix_v1",
    "campaign_root": str(root.resolve()),
    "created_at": datetime.now(timezone.utc).isoformat(),
    "rows": rows,
}
(root / "matrix.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
(root / "reports").mkdir(parents=True, exist_ok=True)
(root / "reports" / "matrix_meta.json").write_text(json.dumps(matrix, indent=2) + "\n", encoding="utf-8")
print(f"wrote matrix.json with {len(rows)} cells")
PY

FLASH_BUNDLE="${C2HLS_ROOT}/artifacts/pc2/flash_selected_bundle/$(basename "${CAMPAIGN_ROOT}")"
DATAFLOW_BUNDLE="${C2HLS_ROOT}/artifacts/pc2/dataflow_selected_bundle/$(basename "${CAMPAIGN_ROOT}")"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] exporting flash_selected -> ${FLASH_BUNDLE}"
"${PY}" "${C2HLS_ROOT}/scripts/pc2/export_flash_selected_bundle.py" --pc2 \
  --matrix-root "${CAMPAIGN_ROOT}" \
  --out-root "${C2HLS_ROOT}/artifacts/pc2/flash_selected_bundle"

# Cosim flash-selected only (individual sbatch jobs, no arrays). Pin xelab -mt off.
export C2HLS_COSIM_XELAB_MT_OFF="${C2HLS_COSIM_XELAB_MT_OFF:-1}"
export C2HLS_FLASH_COSIM_KERNEL=selected
export C2HLS_FLASH_COSIM_FULL_SIZE="${C2HLS_FLASH_COSIM_FULL_SIZE:-1}"
export C2HLS_COSIM_BENCHMARKS_ROOT="${C2HLS_COSIM_BENCHMARKS_ROOT:-${C2HLS_ROOT}/benchmarks_cosim}"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] cosim flash-selected (MT_OFF=${C2HLS_COSIM_XELAB_MT_OFF})"
bash "${SCRIPT_DIR}/run_campaign_selected_cosim.sh" --campaign-root "${CAMPAIGN_ROOT}" --force

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] starting post-flash dataflow PARALLEL (one exclusive Slurm job per remaining bench)"
export C2HLS_POST_FLASH_MATRIX_ROOT="${CAMPAIGN_ROOT}"
export C2HLS_RUN_COSIM=1
export C2HLS_COSIM_XELAB_MT_OFF="${C2HLS_COSIM_XELAB_MT_OFF:-1}"
export C2HLS_DATAFLOW_REPAIR_ROUNDS="${C2HLS_DATAFLOW_REPAIR_ROUNDS:-4}"
export C2HLS_DATAFLOW_CONTRACT_ROUNDS="${C2HLS_DATAFLOW_CONTRACT_ROUNDS:-4}"
export C2HLS_POST_FLASH_RESULTS_SUFFIX="${C2HLS_POST_FLASH_RESULTS_SUFFIX:-hlsfactory_cosim_repairs}"
export C2HLS_DATAFLOW_EXCLUSIVE="${C2HLS_DATAFLOW_EXCLUSIVE:-1}"
export C2HLS_DATAFLOW_WORKER_CPUS="${C2HLS_DATAFLOW_WORKER_CPUS:-16}"
export C2HLS_DATAFLOW_WORKER_MEM_GB="${C2HLS_DATAFLOW_WORKER_MEM_GB:-64}"

# Prompt policy: skills arm uses system_skills; noskills/bare keep DATAFLOW_NO_SKILLS / BARE envs.
PROMPT_POLICY="${C2HLS_POST_FLASH_PROMPT_POLICY:-system_skills}"
export C2HLS_POST_FLASH_PROMPT_POLICY="${PROMPT_POLICY}"

# Prefer campaign DeepSeek / external_llm endpoint. Never spawn an open-weight
# gpu_h100 vLLM when the flash campaign already used an external API.
ep_url="${BATCH_PARALLEL_EXTERNAL_ENDPOINT_URL:-}"
ep_model="${BATCH_PARALLEL_EXTERNAL_MODEL:-${C2HLS_MODEL:-deepseek-chat}}"
if [[ -z "${ep_url}" && -f "${CAMPAIGN_ROOT}/llm_endpoint.json" ]]; then
  ep_url="$("${PY}" -c "import json;print(json.load(open('${CAMPAIGN_ROOT}/llm_endpoint.json')).get('url',''))")"
  ep_model="$("${PY}" -c "import json;d=json.load(open('${CAMPAIGN_ROOT}/llm_endpoint.json'));print(d.get('model') or '${ep_model}')")"
fi
if [[ -n "${ep_url}" ]]; then
  export OPENAI_BASE_URL="${ep_url}"
  export CHATHLS_API_BASE="${OPENAI_BASE_URL}"
  export C2HLS_MODEL="${ep_model}"
  export OPENAI_API_KEY="${OPENAI_API_KEY:-${CHATHLS_API_KEY:-EMPTY}}"
  export CHATHLS_API_KEY="${CHATHLS_API_KEY:-${OPENAI_API_KEY}}"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] external_llm dataflow via ${OPENAI_BASE_URL} model=${C2HLS_MODEL}"
  if ! curl -sf --max-time 10 "${OPENAI_BASE_URL}/models" >/dev/null; then
    echo "ERROR: external endpoint not reachable: ${OPENAI_BASE_URL}" >&2
    exit 4
  fi
fi

# One exclusive Slurm job per remaining bench (max Vitis parallelism). Export job
# depends on all df jobs and writes dataflow_selected_bundle.
PREFIX="${PC2_BATCH_JOB_PREFIX:-bphfpdf}"
bash "${SCRIPT_DIR}/start_hlsfactory_parallel_dataflow.sh" \
  --campaign-root "${CAMPAIGN_ROOT}" \
  --remaining \
  --job-prefix "${PREFIX}" \
  --no-cancel-post

LAUNCH_JSON="${CAMPAIGN_ROOT}/flow/parallel_dataflow/launch.json"
EXPORT_JOB="$("${PY}" -c "import json;from pathlib import Path;p=Path('${LAUNCH_JSON}');
print(json.load(open(p)).get('export_job_id','') if p.is_file() else '')")"
if [[ -z "${EXPORT_JOB}" ]]; then
  echo "ERROR: parallel dataflow launch did not record export_job_id" >&2
  exit 5
fi

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] waiting for parallel export job ${EXPORT_JOB}"
while true; do
  st="$(squeue -j "${EXPORT_JOB}" -h -o '%T' 2>/dev/null || true)"
  if [[ -z "${st}" ]]; then
    break
  fi
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] export_job=${EXPORT_JOB} state=${st}"
  sleep "${POLL_SEC}"
done

# Also keep a stable pointer next to flash_selected naming.
if [[ -d "${FLASH_BUNDLE}" ]]; then
  ln -sfn "${FLASH_BUNDLE}" "${CAMPAIGN_ROOT}/flash_selected"
fi
if [[ -d "${DATAFLOW_BUNDLE}" ]]; then
  ln -sfn "${DATAFLOW_BUNDLE}" "${CAMPAIGN_ROOT}/dataflow_selected"
fi

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] done"
echo "flash_selected=${FLASH_BUNDLE}"
echo "dataflow_selected=${DATAFLOW_BUNDLE}"
