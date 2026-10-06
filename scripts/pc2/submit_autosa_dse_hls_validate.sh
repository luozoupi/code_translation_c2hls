#!/usr/bin/env bash
# Submit one Slurm job per exported AutoSA kernel (max parallelism = kernel count).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

STAMP="${AUTOSA_DSE_HLS_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${C2HLS_ROOT}/artifacts/pc2/autosa_dse_hls_validate_${STAMP}"
SOURCES_ROOT="${AUTOSA_DSE_SOURCES_ROOT:-${C2HLS_ROOT}/AutoSA_sources}"
KERNELS="${AUTOSA_DSE_KERNELS:-}"
MANIFEST="${RUN_ROOT}/manifest.json"
SBATCH_SCRIPT="${SCRIPT_DIR}/autosa_dse_hls_validate.sbatch.sh"

mkdir -p "${RUN_ROOT}" artifacts/pc2/autosa_dse_hls_validate "${RUN_ROOT}/slurm"

MANIFEST_ARGS=(--sources-root "${SOURCES_ROOT}" --out "${MANIFEST}")
if [[ -n "${KERNELS}" ]]; then
  MANIFEST_ARGS+=(--kernels "${KERNELS}")
fi
"${C2HLS_PYTHON:-python3}" "${SCRIPT_DIR}/build_autosa_dse_hls_manifest.py" "${MANIFEST_ARGS[@]}"

JOB_IDS=()
while IFS=$'\t' read -r kernel_id walltime cosim_timeout; do
  jid="$(
    AUTOSA_DSE_KERNEL_ID="${kernel_id}" \
    AUTOSA_DSE_HLS_RUN_ROOT="${RUN_ROOT}" \
    AUTOSA_DSE_SOURCES_ROOT="${SOURCES_ROOT}" \
    AUTOSA_DSE_COSIM_TIMEOUT="${cosim_timeout}" \
    sbatch \
      --job-name="autosa-hls-${kernel_id}" \
      --time="${walltime}" \
      --output="${RUN_ROOT}/slurm/%x-%j.out" \
      --error="${RUN_ROOT}/slurm/%x-%j.err" \
      --export=ALL,AUTOSA_DSE_KERNEL_ID,AUTOSA_DSE_HLS_RUN_ROOT,AUTOSA_DSE_SOURCES_ROOT,AUTOSA_DSE_COSIM_TIMEOUT \
      "${SBATCH_SCRIPT}" \
      | awk '{print $4}'
  )"
  echo "submitted kernel=${kernel_id} job=${jid} walltime=${walltime} cosim_timeout=${cosim_timeout}"
  JOB_IDS+=("${jid}")
done < <(
  "${C2HLS_PYTHON:-python3}" - "${MANIFEST}" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
for k in m["kernels"]:
    print(f"{k['kernel_id']}\t{k['slurm_walltime']}\t{k['cosim_timeout_s']}")
PY
)

{
  echo "stamp=${STAMP}"
  echo "run_root=${RUN_ROOT}"
  echo "manifest=${MANIFEST}"
  echo "kernels=${KERNELS:-all}"
  echo "job_ids=${JOB_IDS[*]}"
} > "${RUN_ROOT}/submit.txt"

echo "submitted ${#JOB_IDS[@]} jobs -> ${RUN_ROOT}"
