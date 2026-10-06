#!/usr/bin/env bash
# Write Slurm scripts for DSE v4 configs.
# This script does not start a proxy and does not submit a job.
# start_dse_v4_one.sh starts one proxy and submits one config.
#
# Usage:
#   ./scripts/pc2/write_dse_v4_sbatch.sh --out-dir /path/to/dse_v4_jobs
#   ./scripts/pc2/write_dse_v4_sbatch.sh --out-dir DIR --config st0_c1 \
#       --endpoint-url http://login:18140/v1
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

OUT_DIR=""
ONLY_CONFIG=""
ENDPOINT_URL=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --out-dir) shift; OUT_DIR="$1"; shift ;;
    --config) shift; ONLY_CONFIG="$1"; shift ;;
    --endpoint-url) shift; ENDPOINT_URL="$1"; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "${OUT_DIR}" ]]; then
  echo "ERROR: --out-dir is required" >&2
  exit 2
fi

case "${OUT_DIR}" in
  *autosa_mm_variant_sweep_20260918*|*20260919*)
    echo "ERROR: refusing to write into ${OUT_DIR}" >&2
    exit 2
    ;;
esac

mkdir -p "${OUT_DIR}/runs"
PY="${C2HLS_PYTHON:-python3}"
READY="${C2HLS_ROOT}/artifacts/pc2/autosa_mm_ijk_benches/n1024"
mapfile -t IDS < <("${PY}" - <<'PY'
import post_flash_dse_v4 as v4
for row in v4.load_v4_configs():
    print(row["id"])
PY
)

if [[ -z "${ONLY_CONFIG}" && -n "${ENDPOINT_URL}" ]]; then
  echo "ERROR: --endpoint-url requires --config" >&2
  exit 2
fi
if [[ -n "${ENDPOINT_URL}" ]]; then
  if [[ "${ENDPOINT_URL}" == *api.deepseek.com* ]]; then
    echo "ERROR: endpoint must be the login-node proxy, not the public DeepSeek URL" >&2
    exit 2
  fi
  if [[ ! "${ENDPOINT_URL}" =~ ^https?://[A-Za-z0-9._-]+:[0-9]+/v1$ ]]; then
    echo "ERROR: endpoint must look like http://host:port/v1" >&2
    exit 2
  fi
fi
if [[ -n "${ONLY_CONFIG}" ]]; then
  found=0
  for config_id in "${IDS[@]}"; do
    if [[ "${config_id}" == "${ONLY_CONFIG}" ]]; then
      found=1
      break
    fi
  done
  if [[ "${found}" -ne 1 ]]; then
    echo "ERROR: unknown config ${ONLY_CONFIG}" >&2
    exit 2
  fi
  IDS=("${ONLY_CONFIG}")
elif [[ "${#IDS[@]}" -ne 30 ]]; then
  echo "ERROR: expected 30 configs, found ${#IDS[@]}" >&2
  exit 2
fi

for config_id in "${IDS[@]}"; do
  job="${OUT_DIR}/${config_id}.sbatch.sh"
  endpoint_exports=""
  if [[ -n "${ENDPOINT_URL}" ]]; then
    endpoint_exports="export OPENAI_BASE_URL=${ENDPOINT_URL}
export CHATHLS_API_BASE=${ENDPOINT_URL}
export C2HLS_MODEL=deepseek-v4-flash
export C2HLS_DSE_MODEL=deepseek-v4-flash"
  fi
  cat > "${job}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=${config_id}
#SBATCH --partition=normal
#SBATCH --qos=cont
#SBATCH --account=hpc-prf-llmfpga
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --chdir=${C2HLS_ROOT}
#SBATCH --output=${OUT_DIR}/${config_id}-%j.out
#SBATCH --error=${OUT_DIR}/${config_id}-%j.err

# Written for a later submit. One dedicated proxy per job must already
# export OPENAI_BASE_URL. This file is not submitted by the writer.
set -euo pipefail
cd "${C2HLS_ROOT}"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

${endpoint_exports}
export C2HLS_DSE_V4=1
export C2HLS_DSE_V4_CONFIG=${config_id}
export C2HLS_DSE_V4_REPAIR_ROUNDS=12
export C2HLS_RUN_COSIM=0
export C2HLS_FOCUS_GROUP=0
export C2HLS_SYNTH_TIMEOUT=86400
export C2HLS_CSIM_TIMEOUT=86400
export C2HLS_AUTOSA_READY_ROOT=${READY}
export C2HLS_MODEL="\${C2HLS_MODEL:-deepseek-v4-flash}"
unset C2HLS_DSE_V3 || true
unset C2HLS_DSE_V3_HARNESS || true
unset C2HLS_DSE_SOURCE_KERNEL || true

PY="\${C2HLS_PYTHON:-python3}"
"\${PY}" "${C2HLS_ROOT}/scripts/pc2/run_dse_v4_one.py" --pc2 \\
  --config ${config_id} \\
  --out-dir "${OUT_DIR}/runs/${config_id}"
EOF
  chmod +x "${job}"
done

{
  echo "configs=${#IDS[@]}"
  echo "submitted=no"
  echo "cpus=8"
  echo "mem=64G"
  echo "time=12:00:00"
  echo "synth_timeout_s=86400"
  echo "csim_timeout_s=86400"
  echo "ready=${READY}"
} > "${OUT_DIR}/manifest.txt"

echo "wrote ${#IDS[@]} scripts under ${OUT_DIR}"
echo "submitted=no"
