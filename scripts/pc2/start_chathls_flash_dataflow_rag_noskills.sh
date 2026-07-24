#!/usr/bin/env bash
# ChatHLS flash+streaming-dataflow: RAG scrape only (Vitis PDFs), NO skills files.
#
# Flash: skill_off (no 90-skills JSON).
# Dataflow: no no_RMW skill overlay (C2HLS_DATAFLOW_NO_SKILLS=1).
# RAG: --rag --scrape over ug1399 + ug902 (Vitis HLS PDFs only by default).
#
# Usage:
#   ./scripts/pc2/start_chathls_flash_dataflow_rag_noskills.sh --dry-run
#   ./scripts/pc2/start_chathls_flash_dataflow_rag_noskills.sh --stamp 20260716_chathls_rag_ns
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

STAMP="${BATCH_PARALLEL_STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stamp) shift; STAMP="$1"; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --borrow-gpu)
      echo "ERROR: this campaign requires --no-borrow-gpu (GPU borrow off)" >&2
      exit 2
      ;;
    --no-borrow-gpu) shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

READY_ROOT="${C2HLS_ROOT}/related_work/benchmarks/HLSFactory_benchmarks/chathls_ready"
if [[ ! -d "${READY_ROOT}/chathls_gemm" ]]; then
  echo "preparing chathls_ready corpus..."
  "${C2HLS_PYTHON:-python3}" "${C2HLS_ROOT}/scripts/prepare_chathls_ready.py" \
    --output-root "${READY_ROOT}"
fi

export BATCH_PARALLEL_CONFIG="${BATCH_PARALLEL_CONFIG:-${SCRIPT_DIR}/batch_parallel_chathls_flash_dataflow.json}"
export BATCH_PARALLEL_VARIANT="${BATCH_PARALLEL_VARIANT:-chathls_aav_n}"
export BATCH_PARALLEL_ARTIFACT_PREFIX="${BATCH_PARALLEL_ARTIFACT_PREFIX:-batch_parallel_chathls_fd_rag_ns}"
export PC2_BATCH_JOB_PREFIX="${PC2_BATCH_JOB_PREFIX:-bpchrn}"
export PC2_FORCE_WALLTIME="${PC2_BATCH_PARALLEL_WALLTIME:-48:00:00}"
export C2HLS_RUN_COSIM=1
export C2HLS_REFERENCE_COSIM=1
export C2HLS_COSIM_REQUIRED=0
export C2HLS_SYNTH_TIMEOUT="${C2HLS_SYNTH_TIMEOUT:-3600}"
export C2HLS_CSIM_TIMEOUT="${C2HLS_CSIM_TIMEOUT:-600}"
# Cap pathological gold/flash cosim (syr2k xsim burned ~11h at 12h cap).
export C2HLS_COSIM_TIMEOUT="${C2HLS_COSIM_TIMEOUT:-7200}"
export C2HLS_DATAFLOW_REPAIR_ROUNDS="${C2HLS_DATAFLOW_REPAIR_ROUNDS:-4}"
export C2HLS_DATAFLOW_CONTRACT_ROUNDS="${C2HLS_DATAFLOW_CONTRACT_ROUNDS:-4}"
export C2HLS_DATAFLOW_MAX_PARALLEL="${C2HLS_DATAFLOW_MAX_PARALLEL:-16}"

# No skills (flash + dataflow).
export C2HLS_CHATHLS_NOSKILLS=1
export C2HLS_DATAFLOW_NO_SKILLS=1
unset C2HLS_DATAFLOW_SKILL_ENTRIES_JSON || true
unset C2HLS_FLASH_SKILL_ENTRIES_JSON || true

# RAG scrape: Vitis PDFs only (ug1399 + ug902), not bug databases.
KR="${C2HLS_ROOT}/artifacts/rag/knowledge_repo"
DEFAULT_VITIS_CORPUS="${KR}/ug1399-vitis-hls-en-us-2024.1.pdf:${KR}/ug902-vivado-high-level-synthesis.pdf"
export C2HLS_RAG=1
export C2HLS_RAG_ENABLE=1
export C2HLS_RAG_MODE="${C2HLS_RAG_MODE:-everywhere}"
export C2HLS_RAG_SCRAPE=1
export C2HLS_RAG_SCRAPE_CORPUS="${C2HLS_RAG_SCRAPE_CORPUS:-${DEFAULT_VITIS_CORPUS}}"
export C2HLS_TMP_RUN="${BATCH_PARALLEL_ARTIFACT_PREFIX}_${STAMP}"

EXTRA_ARGS=(--no-borrow-gpu)
if [[ "${DRY_RUN}" -eq 1 ]]; then
  EXTRA_ARGS+=(--dry-run)
fi

echo "=== ChatHLS flash+streaming-dataflow RAG-ONLY (no skills) ==="
echo "stamp=${STAMP}"
echo "config=${BATCH_PARALLEL_CONFIG}"
echo "variant=${BATCH_PARALLEL_VARIANT}"
echo "gpu_borrow=off park_policy=on park_grace_s=5400"
echo "flash_skills=OFF dataflow_skills=OFF cosim=on-if-possible"
echo "streaming=per-bench flash→dataflow max_parallel=${C2HLS_DATAFLOW_MAX_PARALLEL}"
echo "rag=on scrape=on mode=${C2HLS_RAG_MODE} corpus=${C2HLS_RAG_SCRAPE_CORPUS}"
echo "tmp_run=${C2HLS_TMP_RUN}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  exec env BATCH_PARALLEL_STAMP="${STAMP}" \
    "${SCRIPT_DIR}/start_batch_parallel_campaign.sh" \
    --stamp "${STAMP}" \
    "${EXTRA_ARGS[@]}"
fi

env BATCH_PARALLEL_STAMP="${STAMP}" \
  "${SCRIPT_DIR}/start_batch_parallel_campaign.sh" \
  --stamp "${STAMP}" \
  "${EXTRA_ARGS[@]}"

CAMPAIGN_ROOT="${C2HLS_ROOT}/artifacts/pc2/${BATCH_PARALLEL_ARTIFACT_PREFIX}_${STAMP}"
WATCH_LOG="${CAMPAIGN_ROOT}/flow/stream_dataflow_watcher.log"
mkdir -p "${CAMPAIGN_ROOT}/flow"

GPU_JOB_ID="$(
  "${C2HLS_PYTHON:-python3}" - <<PY
import json
from pathlib import Path
p = Path("${CAMPAIGN_ROOT}") / "campaign.json"
print((json.loads(p.read_text()).get("gpu_job_id") or "") if p.is_file() else "")
PY
)"
DEP_ARGS=()
if [[ -n "${GPU_JOB_ID}" ]]; then
  # Start after GPU allocation begins (not after GPU ends).
  DEP_ARGS=(--dependency="after:${GPU_JOB_ID}")
fi

POST_JOB="$(
  sbatch --parsable \
    --chdir="${C2HLS_ROOT}" \
    --job-name="bpchrn-post" \
    --output="${CAMPAIGN_ROOT}/flow/post_watcher-%j.out" \
    --error="${CAMPAIGN_ROOT}/flow/post_watcher-%j.err" \
    --account="${PC2_SLURM_ACCOUNT:-hpc-prf-llmfpga}" \
    --partition="${PC2_COMPUTE_PARTITION}" \
    --cpus-per-task=4 \
    --mem=16G \
    --time=72:00:00 \
    "${DEP_ARGS[@]}" \
    --export=ALL,C2HLS_RAG=1,C2HLS_RAG_ENABLE=1,C2HLS_RAG_MODE=${C2HLS_RAG_MODE},C2HLS_RAG_SCRAPE=1,C2HLS_RAG_SCRAPE_CORPUS=${C2HLS_RAG_SCRAPE_CORPUS},C2HLS_TMP_RUN=${C2HLS_TMP_RUN},C2HLS_CHATHLS_NOSKILLS=1,C2HLS_DATAFLOW_NO_SKILLS=1,C2HLS_ENDPOINT_WAIT_SEC=${C2HLS_ENDPOINT_WAIT_SEC:-172800} \
    --wrap="bash ${SCRIPT_DIR}/wait_chathls_flash_stream_dataflow.sh --campaign-root ${CAMPAIGN_ROOT} --max-parallel ${C2HLS_DATAFLOW_MAX_PARALLEL} >> ${WATCH_LOG} 2>&1"
)"

"${C2HLS_PYTHON:-python3}" - <<PY
import json
from pathlib import Path
p = Path("${CAMPAIGN_ROOT}") / "campaign.json"
doc = json.loads(p.read_text()) if p.is_file() else {}
doc["post_watcher_job_id"] = "${POST_JOB}"
p.write_text(json.dumps(doc, indent=2) + "\n")
PY

echo "submitted streaming flash→dataflow watcher job ${POST_JOB} (dependency after:${GPU_JOB_ID:-none})"
echo "campaign=${CAMPAIGN_ROOT}"
echo "watch: tail -f ${CAMPAIGN_ROOT}/flow/watch.log"
echo "post:  tail -f ${WATCH_LOG}"
