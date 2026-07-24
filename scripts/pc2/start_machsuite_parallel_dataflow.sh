#!/usr/bin/env bash
# Parallel MachSuite post-flash dataflow on an existing flash-complete campaign.
#
# Replaces the serialized post_flash session with:
#   1) shared campaign GPU (bpmachfd-gpu-*)
#   2) one Slurm compute job per bench (bpmachfd-df-*)
#   3) export job after all dataflow jobs (bpmachfd-df-export)
#
# Usage:
#   ./scripts/pc2/start_machsuite_parallel_dataflow.sh
#   ./scripts/pc2/start_machsuite_parallel_dataflow.sh --dry-run
#   ./scripts/pc2/start_machsuite_parallel_dataflow.sh \
#       --campaign-root artifacts/pc2/batch_parallel_machsuite_fd_20260710_machsuite_flash_dataflow \
#       --benches machsuite_nw,machsuite_md_knn
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

CAMPAIGN_ROOT="${C2HLS_ROOT}/artifacts/pc2/batch_parallel_machsuite_fd_20260710_machsuite_flash_dataflow"
# Remaining flash-done benches from partial restart (bfs_bulk already succeeded).
DEFAULT_BENCHES="machsuite_aes_tableless,machsuite_bfs_queue,machsuite_fft_transpose,machsuite_gemm_ncubed,machsuite_md_knn,machsuite_nw,machsuite_sort_merge,machsuite_sort_radix,machsuite_spmv_crs,machsuite_spmv_ellpack,machsuite_stencil2D,machsuite_stencil3D"
BENCHES="${DEFAULT_BENCHES}"
DRY_RUN=0
CANCEL_SESSION_ID="post_flash_dataflow_20260714_003234"
WALLTIME="${PC2_FORCE_WALLTIME:-72:00:00}"
WORKER_CPUS="${C2HLS_DATAFLOW_WORKER_CPUS:-8}"
WORKER_MEM_GB="${C2HLS_DATAFLOW_WORKER_MEM_GB:-32}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --campaign-root) shift; CAMPAIGN_ROOT="$1"; shift ;;
    --benches) shift; BENCHES="$1"; shift ;;
    --cancel-session-id) shift; CANCEL_SESSION_ID="$1"; shift ;;
    --no-cancel-session) CANCEL_SESSION_ID=""; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --walltime) shift; WALLTIME="$1"; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

# Resolve relative campaign paths.
if [[ "${CAMPAIGN_ROOT}" != /* ]]; then
  CAMPAIGN_ROOT="${C2HLS_ROOT}/${CAMPAIGN_ROOT}"
fi

export BATCH_PARALLEL_CAMPAIGN_ROOT="${CAMPAIGN_ROOT}"
export BATCH_PARALLEL_CONFIG="${BATCH_PARALLEL_CONFIG:-${SCRIPT_DIR}/batch_parallel_machsuite_flash_dataflow.json}"
export BATCH_PARALLEL_VARIANT="${BATCH_PARALLEL_VARIANT:-tier_b_aav_n}"
export PC2_BATCH_JOB_PREFIX="${PC2_BATCH_JOB_PREFIX:-bpmachfd}"
export PC2_FORCE_WALLTIME="${WALLTIME}"
export C2HLS_RUN_COSIM=1
export C2HLS_REFERENCE_COSIM=1
export C2HLS_COSIM_REQUIRED=0
export C2HLS_DATAFLOW_REPAIR_ROUNDS="${C2HLS_DATAFLOW_REPAIR_ROUNDS:-4}"
export C2HLS_DATAFLOW_CONTRACT_ROUNDS="${C2HLS_DATAFLOW_CONTRACT_ROUNDS:-4}"
export C2HLS_POST_FLASH_RESULTS_SUFFIX="${C2HLS_POST_FLASH_RESULTS_SUFFIX:-machsuite_stream_cosim_repairs}"

IFS=',' read -r -a BENCH_ARR <<< "${BENCHES}"
BENCH_ARR=("${BENCH_ARR[@]// /}")

FLOW_DIR="${CAMPAIGN_ROOT}/flow"
DF_DIR="${FLOW_DIR}/parallel_dataflow"
mkdir -p "${DF_DIR}/logs" "${FLOW_DIR}"

echo "=== MachSuite parallel dataflow (batch_parallel) ==="
echo "campaign=${CAMPAIGN_ROOT}"
echo "benches=${#BENCH_ARR[@]}: ${BENCHES}"
echo "walltime=${WALLTIME} worker=${WORKER_CPUS}c/${WORKER_MEM_GB}G"
echo "cancel_session=${CANCEL_SESSION_ID:-<none>}"
echo "dry_run=${DRY_RUN}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "(dry-run) would stop session ${CANCEL_SESSION_ID}, submit GPU + ${#BENCH_ARR[@]} df jobs + export"
  exit 0
fi

# ---------------------------------------------------------------------------
# 1) Stop serialized post_flash session (ONLY those two jobs + its watcher)
# ---------------------------------------------------------------------------
if [[ -n "${CANCEL_SESSION_ID}" ]]; then
  echo "[1/4] stop session ${CANCEL_SESSION_ID}"
  "${SCRIPT_DIR}/stop_session.sh" --session-id "${CANCEL_SESSION_ID}" || true
else
  echo "[1/4] skip session cancel"
fi

# ---------------------------------------------------------------------------
# 2) Shared campaign GPU
# ---------------------------------------------------------------------------
echo "[2/4] submit campaign GPU"
# Drop any stale endpoint; do not cancel unrelated bpchfd/bpautfd GPUs.
rm -f "${CAMPAIGN_ROOT}/llm_endpoint.json"
GPU_JOB="$(
  BATCH_PARALLEL_CAMPAIGN_ROOT="${CAMPAIGN_ROOT}" \
    PC2_BATCH_JOB_PREFIX="${PC2_BATCH_JOB_PREFIX}" \
    PC2_FORCE_WALLTIME="${WALLTIME}" \
    "${SCRIPT_DIR}/batch_parallel_submit_gpu.sh"
)"
GPU_JOB="${GPU_JOB%%;*}"
echo "  gpu_job=${GPU_JOB}"

# Keep campaign.json gpu pointer fresh (flash campaign already complete).
"${C2HLS_PYTHON:-python3}" - <<PY
import json
from pathlib import Path
p = Path("${CAMPAIGN_ROOT}") / "campaign.json"
doc = json.loads(p.read_text()) if p.is_file() else {}
doc["gpu_job_id"] = "${GPU_JOB}"
doc["gpu_mode"] = "up"
doc["dataflow_parallel"] = {
    "scheme": "one_slurm_job_per_bench",
    "gpu_job_id": "${GPU_JOB}",
    "benches": [b for b in """${BENCHES}""".split(",") if b.strip()],
}
p.write_text(json.dumps(doc, indent=2) + "\n")
PY

# ---------------------------------------------------------------------------
# 3) One compute job per bench
# ---------------------------------------------------------------------------
echo "[3/4] submit ${#BENCH_ARR[@]} dataflow jobs"
DF_JOBS=()
JOB_LIST="${DF_DIR}/jobs.jsonl"
: > "${JOB_LIST}"
for bench in "${BENCH_ARR[@]}"; do
  [[ -n "${bench}" ]] || continue
  short="${bench#machsuite_}"
  log_out="${DF_DIR}/logs/${bench}.%j.out"
  log_err="${DF_DIR}/logs/${bench}.%j.err"
  job_id="$(
    sbatch --parsable \
      --chdir="${C2HLS_ROOT}" \
      --job-name="${PC2_BATCH_JOB_PREFIX}-df-${short}" \
      --output="${log_out}" \
      --error="${log_err}" \
      --account="${PC2_SLURM_ACCOUNT:-hpc-prf-llmfpga}" \
      --partition="${PC2_COMPUTE_PARTITION}" \
      --cpus-per-task="${WORKER_CPUS}" \
      --mem="${WORKER_MEM_GB}G" \
      --time="${WALLTIME}" \
      --dependency="after:${GPU_JOB}" \
      --export=ALL,BATCH_PARALLEL_CAMPAIGN_ROOT="${CAMPAIGN_ROOT}",C2HLS_RUN_COSIM=1,C2HLS_REFERENCE_COSIM=1,C2HLS_COSIM_REQUIRED=0,C2HLS_DATAFLOW_REPAIR_ROUNDS="${C2HLS_DATAFLOW_REPAIR_ROUNDS}",C2HLS_DATAFLOW_CONTRACT_ROUNDS="${C2HLS_DATAFLOW_CONTRACT_ROUNDS}",C2HLS_POST_FLASH_RESULTS_SUFFIX="${C2HLS_POST_FLASH_RESULTS_SUFFIX}",C2HLS_POST_FLASH_MATRIX_ROOT="${CAMPAIGN_ROOT}" \
      --wrap="bash ${SCRIPT_DIR}/run_machsuite_dataflow_bench.sh ${bench}"
  )"
  job_id="${job_id%%;*}"
  DF_JOBS+=("${job_id}")
  echo "{\"bench\":\"${bench}\",\"job_id\":\"${job_id}\"}" >> "${JOB_LIST}"
  echo "  ${bench} -> ${job_id}"
done

dep_csv="$(IFS=,; echo "${DF_JOBS[*]}")"

# ---------------------------------------------------------------------------
# 4) Export after all dataflow jobs finish (success or fail)
# ---------------------------------------------------------------------------
echo "[4/4] submit export job after ${#DF_JOBS[@]} df jobs"
EXPORT_JOB="$(
  sbatch --parsable \
    --chdir="${C2HLS_ROOT}" \
    --job-name="${PC2_BATCH_JOB_PREFIX}-df-export" \
    --output="${DF_DIR}/export-%j.out" \
    --error="${DF_DIR}/export-%j.err" \
    --account="${PC2_SLURM_ACCOUNT:-hpc-prf-llmfpga}" \
    --partition="${PC2_COMPUTE_PARTITION}" \
    --cpus-per-task=2 \
    --mem=8G \
    --time=4:00:00 \
    --dependency="afterany:${dep_csv}" \
    --wrap="bash ${SCRIPT_DIR}/wait_machsuite_dataflow_export.sh --campaign-root ${CAMPAIGN_ROOT}"
)"
EXPORT_JOB="${EXPORT_JOB%%;*}"
echo "  export_job=${EXPORT_JOB}"

cat > "${DF_DIR}/launch.json" <<EOF
{
  "scheme": "one_slurm_job_per_bench",
  "campaign_root": "${CAMPAIGN_ROOT}",
  "gpu_job_id": "${GPU_JOB}",
  "export_job_id": "${EXPORT_JOB}",
  "dataflow_job_ids": $(printf '%s\n' "${DF_JOBS[@]}" | "${C2HLS_PYTHON:-python3}" -c 'import json,sys; print(json.dumps([l.strip() for l in sys.stdin if l.strip()]))'),
  "benches": $(printf '%s\n' "${BENCH_ARR[@]}" | "${C2HLS_PYTHON:-python3}" -c 'import json,sys; print(json.dumps([l.strip() for l in sys.stdin if l.strip()]))')
}
EOF

echo
echo "gpu=${GPU_JOB}"
echo "dataflow_jobs=${dep_csv}"
echo "export=${EXPORT_JOB}"
echo "launch=${DF_DIR}/launch.json"
echo "squeue: squeue -u \$USER | rg 'bpmachfd'"
echo "logs:   ls ${DF_DIR}/logs"
