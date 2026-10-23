#!/bin/bash
#SBATCH --account=hpc-prf-llmfpga
#SBATCH --partition=normal
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=06:00:00
cd /scratch/hpc-prf-llmfpga/asa582/projects/workspaces/c2hls_claude_code/study/bench_hard
python3 verify_hard.py --jobs 32
