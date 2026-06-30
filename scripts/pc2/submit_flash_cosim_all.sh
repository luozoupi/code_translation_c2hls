#!/usr/bin/env bash
# Build manifest (if needed), verify env, submit all cosim jobs as Slurm array.
# Use --individual to submit one sbatch per cell instead of an array job.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

STAMP="${C2HLS_FLASH_COSIM_STAMP:-$(date +%Y%m%d_%H%M%S)}"
export C2HLS_FLASH_COSIM_STAMP="${STAMP}"
INDIVIDUAL=0
SKIP_VERIFY=0
MAX_PARALLEL="${PC2_COSIM_ARRAY_MAX_PARALLEL:-40}"
ARTIFACT_GLOB="${C2HLS_FLASH_COSIM_ARTIFACT_GLOB:-flash_*}"
ARTIFACT_FILTER=""
FULL_SIZE=0
KERNEL_SOURCE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stamp) STAMP="$2"; export C2HLS_FLASH_COSIM_STAMP="${STAMP}"; shift 2 ;;
    --individual) INDIVIDUAL=1; shift ;;
    --skip-verify) SKIP_VERIFY=1; shift ;;
    --max-parallel) MAX_PARALLEL="$2"; shift 2 ;;
    --artifact-glob) ARTIFACT_GLOB="$2"; shift 2 ;;
    --artifact) ARTIFACT_FILTER="${ARTIFACT_FILTER} $2"; shift 2 ;;
    --full-size) FULL_SIZE=1; export C2HLS_FLASH_COSIM_FULL_SIZE=1; shift ;;
    --kernel-source)
      KERNEL_SOURCE="$2"
      export C2HLS_FLASH_COSIM_KERNEL="${KERNEL_SOURCE}"
      shift 2
      ;;
    -h|--help)
      echo "usage: $0 [--stamp STAMP] [--individual] [--skip-verify] [--max-parallel N]"
      echo "          [--artifact-glob GLOB] [--artifact BASENAME]... [--full-size]"
      echo "          [--kernel-source selected|phase_b|flash_opt]"
      exit 0
      ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

COSIM_ROOT="${C2HLS_FLASH_COSIM_ROOT:-${C2HLS_ROOT}/artifacts/pc2/flash_cosim}"
RUN_ROOT="${COSIM_ROOT}/${STAMP}"
export C2HLS_FLASH_COSIM_RUN_ROOT="${RUN_ROOT}"
mkdir -p "${RUN_ROOT}/submissions" "${COSIM_ROOT}/slurm"

if [[ ! -f "${RUN_ROOT}/manifest.json" ]]; then
  pc2_log "building cosim manifest stamp=${STAMP} glob=${ARTIFACT_GLOB} full_size=${FULL_SIZE} kernel=${KERNEL_SOURCE:-selected}"
  BUILD_ARGS=(--stamp "${STAMP}" --artifact-glob "${ARTIFACT_GLOB}")
  if [[ "${FULL_SIZE}" -eq 1 ]]; then
    BUILD_ARGS+=(--full-size)
  fi
  if [[ -n "${KERNEL_SOURCE}" ]]; then
    BUILD_ARGS+=(--kernel-source "${KERNEL_SOURCE}")
  fi
  for art in ${ARTIFACT_FILTER}; do
    [[ -n "${art}" ]] && BUILD_ARGS+=(--artifact "${art}")
  done
  python3 "${SCRIPT_DIR}/build_flash_cosim_manifest.py" "${BUILD_ARGS[@]}"
fi

if [[ "${SKIP_VERIFY}" -eq 0 ]]; then
  "${SCRIPT_DIR}/verify_flash_cosim.sh" "${RUN_ROOT}"
fi

CELL_COUNT="$(python3 - <<'PY' "${RUN_ROOT}/manifest.json"
import json, sys
from pathlib import Path
print(len(json.loads(Path(sys.argv[1]).read_text()).get("cells", [])))
PY
)"
if [[ "${CELL_COUNT}" -le 0 ]]; then
  echo "ERROR: manifest has zero cells" >&2
  exit 2
fi
LAST_INDEX=$((CELL_COUNT - 1))

WALLTIME="${PC2_COSIM_WALLTIME:-4:00:00}"
if [[ "${FULL_SIZE}" -eq 1 ]]; then
  WALLTIME="${PC2_COSIM_WALLTIME:-13:00:00}"
  export C2HLS_COSIM_TIMEOUT="${C2HLS_COSIM_TIMEOUT:-43200}"
fi
CPUS="${PC2_COSIM_CPUS:-8}"
MEM="${PC2_COSIM_MEM:-32G}"
PARTITION="${PC2_COSIM_PARTITION:-${PC2_COMPUTE_PARTITION:-normal}}"

if [[ "${INDIVIDUAL}" -eq 1 ]]; then
  pc2_log "submitting ${CELL_COUNT} individual cosim jobs stamp=${STAMP}"
  python3 - <<'PY' "${RUN_ROOT}/manifest.json" "${SCRIPT_DIR}/submit_flash_cosim_one.sh" "${RUN_ROOT}"
import json, subprocess, sys
from pathlib import Path
manifest = Path(sys.argv[1])
submit = sys.argv[2]
run_root = sys.argv[3]
for cell in json.loads(manifest.read_text()).get("cells", []):
    subprocess.run([submit, cell["cell_id"], run_root], check=True)
PY
  echo "submitted ${CELL_COUNT} individual jobs; log: ${RUN_ROOT}/submissions/individual_jobs.log"
  exit 0
fi

SBATCH_ARGS=(
  --job-name="c2hls-cosim-${STAMP}"
  --partition="${PARTITION}"
  --cpus-per-task="${CPUS}"
  --mem="${MEM}"
  --time="${WALLTIME}"
  --array="0-${LAST_INDEX}%${MAX_PARALLEL}"
  --export=ALL,C2HLS_ROOT,C2HLS_SITE,C2HLS_FLASH_COSIM_RUN_ROOT,C2HLS_FLASH_COSIM_STAMP,C2HLS_COSIM_TIMEOUT,C2HLS_FLASH_COSIM_FULL_SIZE
)

if [[ -n "${PC2_SLURM_ACCOUNT:-}" ]]; then
  SBATCH_ARGS+=(--account="${PC2_SLURM_ACCOUNT}")
fi

JOB_ID="$(sbatch "${SBATCH_ARGS[@]}" "${SCRIPT_DIR}/cosim_array.sbatch.sh" | awk '{print $NF}')"
{
  echo "stamp=${STAMP}"
  echo "run_root=${RUN_ROOT}"
  echo "job_id=${JOB_ID}"
  echo "cell_count=${CELL_COUNT}"
  echo "array=0-${LAST_INDEX}%${MAX_PARALLEL}"
  echo "submitted_at=$(date -Is)"
} > "${RUN_ROOT}/submissions/array_job.txt"

pc2_log "submitted cosim array job_id=${JOB_ID} cells=${CELL_COUNT} run_root=${RUN_ROOT}"
echo "array job_id=${JOB_ID}"
echo "run_root=${RUN_ROOT}"
