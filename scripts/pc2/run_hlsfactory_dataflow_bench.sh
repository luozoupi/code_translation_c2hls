#!/usr/bin/env bash
# One-bench HLSFactory post-flash dataflow worker (parallel Slurm jobs).
# Uses campaign llm_endpoint.json (DeepSeek queue proxy); does not spawn gpu_h100.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/setup_vitis_env.sh"
pc2_setup_vitis_env
cd "${C2HLS_ROOT}"

CAMPAIGN_ROOT="${BATCH_PARALLEL_CAMPAIGN_ROOT:?set BATCH_PARALLEL_CAMPAIGN_ROOT}"
BENCH="${1:?usage: $0 <bench>}"
PY="${C2HLS_PYTHON:-${C2HLS_ROOT}/.venv/bin/python}"
[[ -x "${PY}" ]] || PY=python3

export C2HLS_RUN_COSIM="${C2HLS_RUN_COSIM:-1}"
export C2HLS_REFERENCE_COSIM="${C2HLS_REFERENCE_COSIM:-1}"
export C2HLS_COSIM_REQUIRED="${C2HLS_COSIM_REQUIRED:-0}"
export C2HLS_COSIM_XELAB_MT_OFF="${C2HLS_COSIM_XELAB_MT_OFF:-1}"
export C2HLS_DATAFLOW_REPAIR_ROUNDS="${C2HLS_DATAFLOW_REPAIR_ROUNDS:-4}"
export C2HLS_DATAFLOW_CONTRACT_ROUNDS="${C2HLS_DATAFLOW_CONTRACT_ROUNDS:-4}"
export C2HLS_POST_FLASH_RESULTS_SUFFIX="${C2HLS_POST_FLASH_RESULTS_SUFFIX:-hlsfactory_cosim_repairs}"
export C2HLS_POST_FLASH_MATRIX_ROOT="${CAMPAIGN_ROOT}"
export C2HLS_TMP_RUN="${C2HLS_TMP_RUN:-$(basename "${CAMPAIGN_ROOT}")}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-${CHATHLS_API_KEY:-EMPTY}}"
export CHATHLS_API_KEY="${CHATHLS_API_KEY:-${OPENAI_API_KEY}}"

PROMPT_POLICY="${C2HLS_POST_FLASH_PROMPT_POLICY:-system_skills}"

# Prefer per-bench / per-job endpoint (OPENAI_BASE_URL already exported by sbatch).
# Fall back to campaign llm_endpoint.json (legacy shared proxy).
if [[ -n "${OPENAI_BASE_URL:-}" && "${OPENAI_BASE_URL}" != "EMPTY" ]]; then
  export OPENAI_BASE_URL="${OPENAI_BASE_URL%/}"
  export CHATHLS_API_BASE="${OPENAI_BASE_URL}"
  pc2_log "dataflow bench=${BENCH} using injected endpoint ${OPENAI_BASE_URL}"
  deadline=$((SECONDS + 300))
  while (( SECONDS < deadline )); do
    if curl -sf --max-time 10 "${OPENAI_BASE_URL}/models" >/dev/null 2>&1; then
      pc2_log "endpoint ready ${OPENAI_BASE_URL} model=${C2HLS_MODEL:-unset}"
      break
    fi
    sleep 5
  done
  if ! curl -sf --max-time 10 "${OPENAI_BASE_URL}/models" >/dev/null 2>&1; then
    pc2_log "ERROR: injected endpoint not reachable: ${OPENAI_BASE_URL}"
    exit 2
  fi
else
  EP="${CAMPAIGN_ROOT}/llm_endpoint.json"
  deadline=$((SECONDS + ${C2HLS_ENDPOINT_WAIT_SEC:-7200}))
  pc2_log "dataflow bench=${BENCH} waiting for ${EP}"
  while (( SECONDS < deadline )); do
    if [[ -f "${EP}" ]]; then
      OPENAI_BASE_URL="$("${PY}" -c "import json;print(json.load(open('${EP}'))['url'].rstrip('/'))")"
      export OPENAI_BASE_URL
      export CHATHLS_API_BASE="${OPENAI_BASE_URL}"
      MODEL="$("${PY}" -c "import json;print(json.load(open('${EP}')).get('model') or '')" 2>/dev/null || true)"
      if [[ -n "${MODEL}" ]]; then
        export C2HLS_MODEL="${MODEL}"
      fi
      if curl -sf --max-time 10 "${OPENAI_BASE_URL}/models" >/dev/null 2>&1; then
        pc2_log "endpoint ready ${OPENAI_BASE_URL} model=${C2HLS_MODEL:-unset}"
        break
      fi
    fi
    sleep 20
  done
  if [[ -z "${OPENAI_BASE_URL:-}" ]]; then
    pc2_log "ERROR: timed out waiting for LLM endpoint"
    exit 2
  fi
fi

# Streaming DF may start before campaign-complete matrix.json exists.
# Upsert this bench's preferred cell so the planner finds 1 cell.
CELL_DIR="$("${PY}" - <<PY
import json
from pathlib import Path
from datetime import datetime, timezone

root = Path("${CAMPAIGN_ROOT}")
bench = "${BENCH}"
best = None
best_score = None
for cell in sorted((root / "variants").glob(f"*/{bench}/*")):
    if not cell.is_dir() or "failed" in cell.name:
        continue
    if not (
        (cell / f"{bench}_multistep_results.json").is_file()
        or any(cell.glob("*_final.cpp"))
        or any(cell.glob("*_selected.cpp"))
        or any(cell.glob("*_flash_seed.cpp"))
        or any(cell.glob("*_latency_opt*.cpp"))
    ):
        continue
    score = (
        0 if "claude" in cell.name or "haiku" in cell.name or "sonnet" in cell.name else (1 if "deepseek" in cell.name else 2),
        0 if "devstral" not in cell.name else 3,
        cell.name,
    )
    if best is None or score < best_score:
        best = cell
        best_score = score
if best is None:
    raise SystemExit(0)
row = {
    "bench": bench,
    "cell_dir": str(best.resolve()),
    "status": "unknown",
    "model": best.name,
    "variant": best.parent.parent.name,
}
mj = root / "matrix.json"
rows = []
if mj.is_file():
    try:
        rows = json.loads(mj.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            rows = []
    except Exception:
        rows = []
out = []
replaced = False
for r in rows:
    if isinstance(r, dict) and r.get("bench") == bench:
        out.append(row)
        replaced = True
    elif isinstance(r, dict):
        out.append(r)
if not replaced:
    out.append(row)
mj.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
meta = {
    "schema": "batch_parallel_matrix_v1",
    "campaign_root": str(root.resolve()),
    "updated_at": datetime.now(timezone.utc).isoformat(),
    "rows": out,
}
(root / "reports").mkdir(parents=True, exist_ok=True)
(root / "reports" / "matrix_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
print(str(best.resolve()))
PY
)"
if [[ -z "${CELL_DIR}" || ! -d "${CELL_DIR}" ]]; then
  pc2_log "ERROR: no selectable flash cell for ${BENCH} (cannot build matrix row)"
  exit 3
fi
pc2_log "matrix row ready ${BENCH} cell=${CELL_DIR}"

pc2_log "START dataflow ${BENCH}"
set +e
"${PY}" "${SCRIPT_DIR}/run_post_flash_dataflow.py" --pc2 \
  --matrix-root "${CAMPAIGN_ROOT}" \
  --benches "${BENCH}" \
  --force \
  --results-suffix "${C2HLS_POST_FLASH_RESULTS_SUFFIX}" \
  --prompt-policy "${PROMPT_POLICY}" \
  --contract-turns "${C2HLS_DATAFLOW_CONTRACT_ROUNDS}"
rc=$?
set -e
pc2_log "DONE dataflow ${BENCH} rc=${rc}"

# After dataflow (+ optional lat_opt chain): rank candidates and fire async ranked cosim.
if [[ "${rc}" -eq 0 && "${C2HLS_RANKED_COSIM_AFTER_DATAFLOW:-0}" == "1" ]]; then
  if [[ -n "${CELL_DIR}" && -d "${CELL_DIR}" ]]; then
    pc2_log "rank dataflow candidates ${BENCH}"
    "${PY}" - <<PY
import sys
from pathlib import Path
sys.path.insert(0, "${SCRIPT_DIR}")
from flash_df_candidate_rank import rank_and_promote
rank_and_promote(Path("${CELL_DIR}"), "${BENCH}", side="dataflow")
print("ok")
PY
    FLOW_DIR="${CAMPAIGN_ROOT}/flow/dataflow_ranked_cosim"
    mkdir -p "${FLOW_DIR}/slurm"
    short="${BENCH#hlsfactory_}"
    PREFIX="${PC2_BATCH_JOB_PREFIX:-bphfrc}"
    JOB_ID="$(
      sbatch --parsable \
        --chdir="${C2HLS_ROOT}" \
        --job-name="${PREFIX}-drc-${short}" \
        --output="${FLOW_DIR}/slurm/${BENCH}-%j.out" \
        --error="${FLOW_DIR}/slurm/${BENCH}-%j.err" \
        --account="${PC2_SLURM_ACCOUNT:-hpc-prf-llmfpga}" \
        --partition="${PC2_COMPUTE_PARTITION}" \
        --cpus-per-task="${PC2_COSIM_CPUS:-8}" \
        --mem="${PC2_COSIM_MEM:-32G}" \
        --time="${PC2_COSIM_WALLTIME:-48:00:00}" \
        --export="ALL,BATCH_PARALLEL_CAMPAIGN_ROOT=${CAMPAIGN_ROOT},C2HLS_RANKED_COSIM_CELL_DIR=${CELL_DIR},C2HLS_RANKED_COSIM_BENCH=${BENCH},C2HLS_RANKED_COSIM_SIDE=dataflow,C2HLS_RANKED_COSIM_FORCE=1,C2HLS_COSIM_XELAB_MT_OFF=1,C2HLS_FLASH_COSIM_FULL_SIZE=1,C2HLS_COSIM_TIMEOUT=${C2HLS_COSIM_TIMEOUT:-43200},C2HLS_COSIM_TRACE_LEVEL=${C2HLS_COSIM_TRACE_LEVEL:-none},C2HLS_COSIM_BENCHMARKS_ROOT=${C2HLS_COSIM_BENCHMARKS_ROOT:-${C2HLS_ROOT}/benchmarks_cosim}" \
        --wrap="bash ${SCRIPT_DIR}/run_ranked_cosim_bench.sh"
    )"
    JOB_ID="${JOB_ID%%;*}"
    echo "${JOB_ID} ${BENCH}" >> "${FLOW_DIR}/job_ids.txt"
    pc2_log "submitted dataflow ranked cosim ${BENCH} job=${JOB_ID} (async)"
  else
    pc2_log "WARNING: no cell_dir for ${BENCH}; skip ranked cosim"
  fi
fi

exit "${rc}"
