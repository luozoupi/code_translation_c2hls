#!/bin/bash
#SBATCH --account=def-zhenman
#SBATCH --job-name=posthoc_hard
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/asa582/workspaces/c2hls_claude_code/study/posthoc_csim/hard-%j.out
python3 /scratch/asa582/workspaces/c2hls_claude_code/study/posthoc_csim/hard.py
