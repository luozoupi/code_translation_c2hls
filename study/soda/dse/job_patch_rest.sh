#!/bin/bash
#SBATCH --account=def-zhenman
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
# patch_soda.py over every designs/<cfg>, 16 in parallel; one line per config in patch_rest.log
cd /scratch/asa582/workspaces/c2hls_claude_code/study/soda/dse
(python3 patch_soda.py contrast_u16 & python3 patch_soda.py contrast_u16_cr & wait) > patch_rest.log 2>&1
echo "done $(grep -c . patch_rest.log)" >> patch_rest.log
