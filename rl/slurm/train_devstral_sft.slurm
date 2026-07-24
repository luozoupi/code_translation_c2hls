#!/bin/bash
#SBATCH -J c2hls-devstral-sft
#SBATCH -p gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH -A hpc-prf-llmfpga
#SBATCH -t 24:00:00
#SBATCH -o /scratch/hpc-prf-llmfpga/asa582/projects/c2hls/rl/logs/devstral-sft-%j.out
#SBATCH -e /scratch/hpc-prf-llmfpga/asa582/projects/c2hls/rl/logs/devstral-sft-%j.err
#
# QLoRA/LoRA SFT on Devstral-2-123B using cosim-verified rl_dataset.
# FP8 serve weights are auto-dequantized to bf16 inside train_sft.py.
# Walltime 24h. Submit: sbatch c2hls/rl/slurm/train_devstral_sft.slurm

set -euo pipefail

RL_ROOT="${RL_ROOT:-/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/rl}"
DEVSTRAL_ROOT="${DEVSTRAL_ROOT:-/scratch/hpc-prf-llmfpga/asa582/projects/devstral2}"
MODEL_PATH="${MODEL_PATH:-${DEVSTRAL_ROOT}/models/Devstral-2-123B-Instruct-2512}"
OUT_DIR="${OUT_DIR:-${RL_ROOT}/outputs/devstral_sft_${SLURM_JOB_ID}}"
VENV="${TRL_VENV:-${RL_ROOT}/.venv-trl}"

mkdir -p "${RL_ROOT}/logs" "${OUT_DIR}"
cd "${RL_ROOT}"

# shellcheck disable=SC1091
source "${DEVSTRAL_ROOT}/load_gpu_modules.sh"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "Missing model at ${MODEL_PATH}"
  exit 1
fi
if [[ ! -f prepared/sft/train.jsonl ]]; then
  python scripts/prepare_trl_datasets.py
fi

export TOKENIZERS_PARALLELISM=false
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

python scripts/train_sft.py \
  --model "${MODEL_PATH}" \
  --train-file prepared/sft/train.jsonl \
  --val-file prepared/sft/val.jsonl \
  --output-dir "${OUT_DIR}" \
  --max-seq-length "${MAX_SEQ_LENGTH:-8192}" \
  --num-train-epochs "${NUM_EPOCHS:-2}" \
  --learning-rate "${LR:-1e-4}" \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps "${GRAD_ACCUM:-16}" \
  --lora-r "${LORA_R:-16}" \
  --lora-alpha "${LORA_ALPHA:-32}"

echo "SFT adapter: ${OUT_DIR}/adapter"
echo "${OUT_DIR}/adapter" > "${OUT_DIR}/adapter.path"
