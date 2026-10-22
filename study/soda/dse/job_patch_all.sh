#!/bin/bash
#SBATCH --account=def-zhenman
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:00:00
# patch_soda.py over every designs/<cfg>, 16 in parallel; one line per config in patch_all.log
cd /scratch/asa582/workspaces/c2hls_claude_code/study/soda/dse
ls designs | xargs -P 16 -I{} python3 patch_soda.py {} > patch_all.log 2>&1
echo "done $(grep -c . patch_all.log)" >> patch_all.log
