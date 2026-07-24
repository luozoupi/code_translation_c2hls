#!/usr/bin/env bash
# Create a dedicated TRL training venv (do not reuse vLLM serve envs).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${TRL_VENV:-${ROOT}/.venv-trl}"

# Prefer Devstral-style CUDA 12.6 + Python 3.11 for bitsandbytes/TRL.
# Override with: source /path/to/load_gpu_modules.sh before calling this.
if [[ -z "${CUDA_HOME:-}" ]]; then
  if [[ -f /scratch/hpc-prf-llmfpga/asa582/projects/devstral2/load_gpu_modules.sh ]]; then
    # shellcheck disable=SC1091
    source /scratch/hpc-prf-llmfpga/asa582/projects/devstral2/load_gpu_modules.sh
  else
    module purge 2>/dev/null || true
    module load lang system CUDA/12.6.0 Python/3.11.5-GCCcore-13.2.0 || true
  fi
fi

python3 -m venv "${VENV}"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
pip install -U pip setuptools wheel
pip install -r "${ROOT}/requirements-trl.txt"
python -c "import torch, trl, peft, transformers; print('torch', torch.__version__, 'trl', trl.__version__, 'cuda', torch.cuda.is_available())"
echo "Activate with: source ${VENV}/bin/activate"
