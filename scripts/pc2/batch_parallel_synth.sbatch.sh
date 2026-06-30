#!/usr/bin/env bash
#SBATCH --job-name=bp-synth
#SBATCH --output=slurm-bp-synth-%j.out
#SBATCH --error=slurm-bp-synth-%j.err

set -euo pipefail

_REPO_ROOT="${C2HLS_ROOT:-${SLURM_SUBMIT_DIR:?missing SLURM_SUBMIT_DIR}}"
SCRIPT_DIR="${_REPO_ROOT}/scripts/pc2"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/setup_vitis_env.sh"
pc2_setup_vitis_env
cd "${C2HLS_ROOT}"

export C2HLS_RUN_COSIM=0
export C2HLS_COSIM_TIMEOUT="${C2HLS_COSIM_TIMEOUT:-43200}"

exec "${C2HLS_PYTHON:-python3}" "${SCRIPT_DIR}/batch_parallel_node_runner.py" \
  --campaign-root "${BATCH_PARALLEL_CAMPAIGN_ROOT}" \
  --variant "${BATCH_PARALLEL_VARIANT}" \
  --role synth \
  --node-index "${BATCH_PARALLEL_NODE_INDEX}"
