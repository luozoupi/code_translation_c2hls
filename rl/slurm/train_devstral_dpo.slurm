#!/bin/bash
#SBATCH -J c2hls-devstral-dpo
#SBATCH -p gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH -A hpc-prf-llmfpga
#SBATCH -t 24:00:00
#SBATCH -o /scratch/hpc-prf-llmfpga/asa582/projects/c2hls/rl/logs/devstral-dpo-%j.out
#SBATCH -e /scratch/hpc-prf-llmfpga/asa582/projects/c2hls/rl/logs/devstral-dpo-%j.err
#
# Offline DPO after SFT. Set SFT_ADAPTER to the SFT output adapter dir.
#   SFT_ADAPTER=/.../outputs/devstral_sft_JOBID/adapter \
#     sbatch c2hls/rl/slurm/train_devstral_dpo.slurm

set -euo pipefail

RL_ROOT="${RL_ROOT:-/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/rl}"
DEVSTRAL_ROOT="${DEVSTRAL_ROOT:-/scratch/hpc-prf-llmfpga/asa582/projects/devstral2}"
MODEL_PATH="${MODEL_PATH:-${DEVSTRAL_ROOT}/models/Devstral-2-123B-Instruct-2512}"
OUT_DIR="${OUT_DIR:-${RL_ROOT}/outputs/devstral_dpo_${SLURM_JOB_ID}}"
VENV="${TRL_VENV:-${RL_ROOT}/.venv-trl}"
SFT_ADAPTER="${SFT_ADAPTER:-}"

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
# Prefer hybrid latency pairs (cosim if pass else csynth). Fallback to rl_dataset DPO.
DPO_TRAIN="${DPO_TRAIN:-${RL_ROOT}/prepared/dpo_hybrid/train.jsonl}"
DPO_VAL="${DPO_VAL:-${RL_ROOT}/prepared/dpo_hybrid/val.jsonl}"
if [[ ! -f "${DPO_TRAIN}" ]]; then
  python scripts/build_hybrid_latency_dpo.py
fi
if [[ ! -f "${DPO_TRAIN}" ]]; then
  python scripts/prepare_trl_datasets.py
  DPO_TRAIN="${RL_ROOT}/prepared/dpo/train.jsonl"
  DPO_VAL="${RL_ROOT}/prepared/dpo/val.jsonl"
fi

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

EXTRA=()
if [[ -n "${SFT_ADAPTER}" ]]; then
  EXTRA+=(--sft-adapter "${SFT_ADAPTER}")
fi

python scripts/train_dpo.py \
  --model "${MODEL_PATH}" \
  "${EXTRA[@]}" \
  --train-file "${DPO_TRAIN}" \
  --val-file "${DPO_VAL}" \
  --output-dir "${OUT_DIR}" \
  --max-length "${MAX_SEQ_LENGTH:-4096}" \
  --num-train-epochs "${NUM_EPOCHS:-1}" \
  --learning-rate "${LR:-5e-6}" \
  --beta "${DPO_BETA:-0.1}" \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps "${GRAD_ACCUM:-16}" \
  --lora-r "${LORA_R:-16}" \
  --lora-alpha "${LORA_ALPHA:-32}"

echo "DPO adapter: ${OUT_DIR}/adapter"
echo "${OUT_DIR}/adapter" > "${OUT_DIR}/adapter.path"
