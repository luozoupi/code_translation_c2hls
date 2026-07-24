#!/bin/bash
#SBATCH -J c2hls-glm-dpo
#SBATCH -p gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH -A hpc-prf-llmfpga
#SBATCH -t 12:00:00
#SBATCH -o /scratch/hpc-prf-llmfpga/asa582/projects/c2hls/rl/logs/glm-dpo-%j.out
#SBATCH -e /scratch/hpc-prf-llmfpga/asa582/projects/c2hls/rl/logs/glm-dpo-%j.err
#
# EXPERIMENTAL offline DPO for a trainable GLM dense checkpoint.
#   MODEL_PATH=... SFT_ADAPTER=.../adapter sbatch c2hls/rl/slurm/train_glm_dpo.slurm

set -euo pipefail

RL_ROOT="${RL_ROOT:-/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/rl}"
MODEL_PATH="${MODEL_PATH:-}"
OUT_DIR="${OUT_DIR:-${RL_ROOT}/outputs/glm_dpo_${SLURM_JOB_ID}}"
VENV="${TRL_VENV:-${RL_ROOT}/.venv-trl}"
SFT_ADAPTER="${SFT_ADAPTER:-}"

mkdir -p "${RL_ROOT}/logs" "${OUT_DIR}"
cd "${RL_ROOT}"

if [[ -z "${MODEL_PATH}" ]]; then
  echo "Set MODEL_PATH to a trainable GLM dense checkpoint."
  exit 2
fi

if [[ -f /scratch/hpc-prf-llmfpga/asa582/projects/devstral2/load_gpu_modules.sh ]]; then
  # shellcheck disable=SC1091
  source /scratch/hpc-prf-llmfpga/asa582/projects/devstral2/load_gpu_modules.sh
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

if [[ ! -f prepared/dpo/train.jsonl ]]; then
  python scripts/prepare_trl_datasets.py
fi

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

EXTRA=()
if [[ -n "${SFT_ADAPTER}" ]]; then
  EXTRA+=(--sft-adapter "${SFT_ADAPTER}")
fi

python scripts/train_dpo.py \
  --model "${MODEL_PATH}" \
  "${EXTRA[@]}" \
  --train-file prepared/dpo/train.jsonl \
  --val-file prepared/dpo/val.jsonl \
  --output-dir "${OUT_DIR}" \
  --max-length "${MAX_SEQ_LENGTH:-8192}" \
  --max-prompt-length "${MAX_PROMPT_LENGTH:-4096}" \
  --num-train-epochs "${NUM_EPOCHS:-1}" \
  --learning-rate "${LR:-5e-6}" \
  --beta "${DPO_BETA:-0.1}" \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps "${GRAD_ACCUM:-16}" \
  --attn-implementation "${ATTN_IMPL:-sdpa}" \
  --lora-r "${LORA_R:-16}" \
  --lora-alpha "${LORA_ALPHA:-32}"

echo "DPO adapter: ${OUT_DIR}/adapter"
echo "${OUT_DIR}/adapter" > "${OUT_DIR}/adapter.path"
