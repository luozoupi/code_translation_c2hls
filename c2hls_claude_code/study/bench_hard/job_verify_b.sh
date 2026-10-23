#!/bin/bash
#SBATCH --account=hpc-prf-llmfpga
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
cd /scratch/hpc-prf-llmfpga/asa582/projects/workspaces/c2hls_claude_code/study/bench_hard
python3 verify_hard.py --jobs 8 --csv verify_hard_b.csv
