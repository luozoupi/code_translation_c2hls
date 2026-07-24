#!/bin/bash
#SBATCH -J c2hls-devstral-sft-combined
#SBATCH -p gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH -A hpc-prf-llmfpga
#SBATCH -t 24:00:00
#SBATCH -o /scratch/hpc-prf-llmfpga/asa582/projects/c2hls/rl/logs/devstral-sft-combined-%j.out
#SBATCH -e /scratch/hpc-prf-llmfpga/asa582/projects/c2hls/rl/logs/devstral-sft-combined-%j.err
#
# NON-DESTRUCTIVE: trains a NEW adapter from sft_combined (~4.5k).
# Does NOT overwrite outputs/devstral_sft_2024406 or the DPO adapter.
# Continues from the first SFT adapter (merge then new LoRA) with a lower LR
# so the first checkpoint remains the safety baseline until held-out eval.

set -euo pipefail

RL_ROOT="${RL_ROOT:-/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/rl}"
DEVSTRAL_ROOT="${DEVSTRAL_ROOT:-/scratch/hpc-prf-llmfpga/asa582/projects/devstral2}"
MODEL_PATH="${MODEL_PATH:-${DEVSTRAL_ROOT}/models/Devstral-2-123B-Instruct-2512}"
PRIOR_SFT="${PRIOR_SFT_ADAPTER:-${RL_ROOT}/outputs/devstral_sft_2024406/adapter}"
OUT_DIR="${OUT_DIR:-${RL_ROOT}/outputs/devstral_sft_combined_${SLURM_JOB_ID}}"
VENV="${TRL_VENV:-${RL_ROOT}/.venv-trl}"
TRAIN="${TRAIN_FILE:-${RL_ROOT}/prepared/sft_combined/train.jsonl}"
VAL="${VAL_FILE:-${RL_ROOT}/prepared/sft_combined/val.jsonl}"

mkdir -p "${RL_ROOT}/logs" "${OUT_DIR}"
cd "${RL_ROOT}"

# shellcheck disable=SC1091
source "${DEVSTRAL_ROOT}/load_gpu_modules.sh"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

if [[ ! -f "${TRAIN}" ]]; then
  echo "Missing ${TRAIN}; run merge_sft_corpora.py first"
  exit 1
fi
if [[ ! -d "${PRIOR_SFT}" ]]; then
  echo "Missing prior SFT adapter ${PRIOR_SFT}"
  exit 1
fi

# Reuse train_dpo's merge path: load base, merge prior SFT, then SFTTrainer with new LoRA
# via a tiny wrapper flag on train_sft — for now call train_sft with --init-adapter if present,
# else a one-liner merge helper.
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

python - <<'PY' || true
# ensure train_sft supports --init-adapter; if not, we patch below at submit time
import inspect, pathlib, sys
sys.path.insert(0, "scripts")
import train_sft
print("train_sft has init-adapter:", "--init-adapter" in open("scripts/train_sft.py").read())
PY

# Prefer dedicated flag; fall back to env INIT_ADAPTER consumed inside train_sft if added.
INIT_ARGS=()
if grep -q 'init-adapter\|init_adapter\|INIT_ADAPTER' scripts/train_sft.py; then
  INIT_ARGS+=(--init-adapter "${PRIOR_SFT}")
else
  export INIT_ADAPTER="${PRIOR_SFT}"
fi

python scripts/train_sft.py \
  --model "${MODEL_PATH}" \
  "${INIT_ARGS[@]}" \
  --train-file "${TRAIN}" \
  --val-file "${VAL}" \
  --output-dir "${OUT_DIR}" \
  --max-seq-length "${MAX_SEQ_LENGTH:-8192}" \
  --num-train-epochs "${NUM_EPOCHS:-1}" \
  --learning-rate "${LR:-2e-5}" \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps "${GRAD_ACCUM:-16}" \
  --lora-r "${LORA_R:-16}" \
  --lora-alpha "${LORA_ALPHA:-32}"

echo "Combined SFT adapter (new): ${OUT_DIR}/adapter"
echo "Prior SFT untouched: ${PRIOR_SFT}"
echo "${OUT_DIR}/adapter" > "${OUT_DIR}/adapter.path"
