#!/bin/bash
#SBATCH -J c2hls-glm-sft
#SBATCH -p gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH -A hpc-prf-llmfpga
#SBATCH -t 12:00:00
#SBATCH -o /scratch/hpc-prf-llmfpga/asa582/projects/c2hls/rl/logs/glm-sft-%j.out
#SBATCH -e /scratch/hpc-prf-llmfpga/asa582/projects/c2hls/rl/logs/glm-sft-%j.err
#
# EXPERIMENTAL: QLoRA SFT against a *trainable* GLM checkpoint.
#
# Important:
#   - Your serve stack uses zai-org/GLM-4.7-FP8 (358B MoE) for *inference*.
#   - Offline SFT of that FP8 MoE with TRL/bitsandbytes is usually NOT practical.
#   - Point MODEL_PATH at a smaller dense GLM (e.g. GLM-4-9B / GLM-4-32B) if available,
#     or keep Devstral as the fine-tune target and only *evaluate* with GLM-4.7 serve.
#
#   MODEL_PATH=/path/to/GLM-4-9B-Chat sbatch c2hls/rl/slurm/train_glm_sft.slurm

set -euo pipefail

RL_ROOT="${RL_ROOT:-/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/rl}"
GLM_ROOT="${GLM_ROOT:-/scratch/hpc-prf-llmfpga/asa582/projects/glm4.7}"
MODEL_PATH="${MODEL_PATH:-}"
OUT_DIR="${OUT_DIR:-${RL_ROOT}/outputs/glm_sft_${SLURM_JOB_ID}}"
VENV="${TRL_VENV:-${RL_ROOT}/.venv-trl}"

mkdir -p "${RL_ROOT}/logs" "${OUT_DIR}"
cd "${RL_ROOT}"

if [[ -z "${MODEL_PATH}" ]]; then
  echo "Set MODEL_PATH to a trainable GLM dense checkpoint (not GLM-4.7-FP8 serve weights)."
  echo "Example: MODEL_PATH=/scratch/.../models/GLM-4-9B-Chat sbatch $0"
  exit 2
fi

# Prefer Devstral CUDA/Python for TRL+bnb; override if your GLM train stack differs.
if [[ -f /scratch/hpc-prf-llmfpga/asa582/projects/devstral2/load_gpu_modules.sh ]]; then
  # shellcheck disable=SC1091
  source /scratch/hpc-prf-llmfpga/asa582/projects/devstral2/load_gpu_modules.sh
elif [[ -f "${GLM_ROOT}/load_gpu_modules.sh" ]]; then
  # shellcheck disable=SC1091
  source "${GLM_ROOT}/load_gpu_modules.sh"
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

if [[ ! -f prepared/sft/train.jsonl ]]; then
  python scripts/prepare_trl_datasets.py
fi

export TOKENIZERS_PARALLELISM=false
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
  --attn-implementation "${ATTN_IMPL:-sdpa}" \
  --lora-r "${LORA_R:-16}" \
  --lora-alpha "${LORA_ALPHA:-32}"

echo "SFT adapter: ${OUT_DIR}/adapter"
echo "${OUT_DIR}/adapter" > "${OUT_DIR}/adapter.path"
