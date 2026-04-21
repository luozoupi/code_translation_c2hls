#!/usr/bin/env bash
source /mnt/data/luo00466/Xilinx/2025.2/Vitis/settings64.sh 2>/dev/null || true
eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate py310_2 2>/dev/null || true
cd /home/luo00466/code_translation-c2hls || exit 1
mkdir -p logs
log="logs/rerun_kmeans.log"
echo "=== $(date -Is)  kmeans  rerun after signature-comment fix ===" > "$log"
python c2hls.py --bench kmeans --model claude-haiku-4-5-20251001 --turns 3 >> "$log" 2>&1
echo "=== $(date -Is)  kmeans  exit=$? ===" >> "$log"
