#!/usr/bin/env bash
#SBATCH --job-name=hlsf-cosim-bl
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=artifacts/pc2/baseline_cosim/slurm/cosim-baseline-jsonl-%j.out
#SBATCH --error=artifacts/pc2/baseline_cosim/slurm/cosim-baseline-jsonl-%j.err

set -euo pipefail

_REPO_ROOT="${C2HLS_ROOT:-${SLURM_SUBMIT_DIR:?missing SLURM_SUBMIT_DIR}}"
_SCRIPT_DIR="${_REPO_ROOT}/scripts/pc2"
# shellcheck disable=SC1091
source "${_SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"
mkdir -p artifacts/pc2/baseline_cosim/slurm c2hls_tmp

export C2HLS_SITE=pc2
# shellcheck disable=SC1091
source "${C2HLS_ROOT}/scripts/setup_emu_env.sh"

STAMP="${C2HLS_HLSFACTORY_BASELINE_STAMP:-20260616_benchmarks}"
OUT_JSONL="${C2HLS_HLSFACTORY_BASELINE_JSONL:-${C2HLS_ROOT}/misc/hlsfactory_cosim_baseline_u280_${STAMP}.jsonl}"

export C2HLS_HLSFACTORY_BASELINE_CORPUS=benchmarks_cosim
export C2HLS_HLSFACTORY_BASELINE_STAMP="${STAMP}"
export C2HLS_HLSFACTORY_BASELINE_JSONL="${OUT_JSONL}"
export C2HLS_HLSFACTORY_BASELINE_CSIM=1
export C2HLS_HLSFACTORY_BASELINE_COSIM=0

pc2_log "hlsfactory cosim baseline jsonl stamp=${STAMP} out=${OUT_JSONL}"

"${C2HLS_PYTHON:-python3}" "${C2HLS_ROOT}/misc/generate_hlsfactory_baseline_jsonl.py"

pc2_log "finished cosim baseline jsonl -> ${OUT_JSONL}"
