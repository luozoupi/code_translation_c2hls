#!/usr/bin/env bash
#SBATCH --job-name=tier-a-gold
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=slurm-tier-a-gold-%j.out
#SBATCH --error=slurm-tier-a-gold-%j.err

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
export C2HLS_REFERENCE_COSIM=0
export C2HLS_SYNTH_TIMEOUT="${C2HLS_SYNTH_TIMEOUT:-3600}"

PY="${C2HLS_PYTHON:-python3}"
BENCH="${TIER_A_GOLD_BENCH:?set TIER_A_GOLD_BENCH}"
STAMP="${TIER_A_GOLD_STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
OUT="${C2HLS_ROOT}/artifacts/pc2/tier_a_gold_verify_${STAMP}"

exec "${PY}" "${SCRIPT_DIR}/validate_tier_a_gold_gates.py" \
  --bench "${BENCH}" \
  --out "${OUT}"
