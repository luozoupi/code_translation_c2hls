#!/usr/bin/env bash
#SBATCH --job-name=tier-a-csim-tcl
#SBATCH --output=slurm-tier-a-csim-tcl-%j.out
#SBATCH --error=slurm-tier-a-csim-tcl-%j.err

set -euo pipefail

_REPO_ROOT="${C2HLS_ROOT:-${SLURM_SUBMIT_DIR:?}}"
SCRIPT_DIR="${_REPO_ROOT}/scripts/pc2"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/setup_vitis_env.sh"
pc2_setup_vitis_env

BENCH_DIR="${TIER_A_BENCH_DIR:?}"
cd "${BENCH_DIR}"
echo "cwd=$(pwd)"
vitis-run --tcl --input_file dataset_hls_csim.tcl
echo "dataset_hls_csim.tcl exit=$?"
