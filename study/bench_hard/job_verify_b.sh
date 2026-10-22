#!/bin/bash
#SBATCH --account=def-zhenman
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
cd /scratch/asa582/workspaces/c2hls_claude_code/study/bench_hard
python3 verify_hard.py --jobs 8 --csv verify_hard_b.csv
