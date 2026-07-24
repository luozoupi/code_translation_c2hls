#!/bin/bash
#SBATCH -J c2hls-trl-venv
#SBATCH -p gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -A hpc-prf-llmfpga
#SBATCH -t 1:00:00
#SBATCH -o /scratch/hpc-prf-llmfpga/asa582/projects/c2hls/rl/logs/trl-venv-%j.out
#SBATCH -e /scratch/hpc-prf-llmfpga/asa582/projects/c2hls/rl/logs/trl-venv-%j.err
#
# One-shot: create rl/.venv-trl with TRL/PEFT/torch for offline SFT+DPO.

set -euo pipefail
RL_ROOT="${RL_ROOT:-/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/rl}"
mkdir -p "${RL_ROOT}/logs"
cd "${RL_ROOT}"
bash scripts/setup_trl_venv.sh
source .venv-trl/bin/activate
python -c "import torch, trl, peft, transformers; print('OK', 'torch', torch.__version__, 'trl', trl.__version__, 'cuda', torch.cuda.is_available(), 'gpus', torch.cuda.device_count())"
echo "Venv ready at ${RL_ROOT}/.venv-trl"
