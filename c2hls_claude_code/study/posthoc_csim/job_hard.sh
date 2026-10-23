#!/bin/bash
#SBATCH --account=hpc-prf-llmfpga
#SBATCH --partition=normal
#SBATCH --job-name=posthoc_hard
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/hpc-prf-llmfpga/asa582/projects/workspaces/c2hls_claude_code/study/posthoc_csim/hard-%j.out
python3 /scratch/hpc-prf-llmfpga/asa582/projects/workspaces/c2hls_claude_code/study/posthoc_csim/hard.py
