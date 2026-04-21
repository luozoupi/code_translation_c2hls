#!/usr/bin/env bash
# Rerun the 3 most-infeasible-on-Artix benchmarks on U50 @ 3.33 ns.
# Uses Fix A: variants[-1] as GT, no g++ preflight.
source /mnt/data/luo00466/Xilinx/2025.2/Vitis/settings64.sh 2>/dev/null || true
eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate py310_2 2>/dev/null || true
cd /home/luo00466/code_translation-c2hls || exit 1

MODEL="${MODEL:-claude-haiku-4-5-20251001}"
TURNS="${TURNS:-3}"

mkdir -p logs
for b in nw lud kmeans; do
    log="logs/rerun_${b}.log"
    echo "=== $(date -Is)  $b  model=$MODEL turns=$TURNS ===" | tee "$log"
    python c2hls.py --bench "$b" --model "$MODEL" --turns "$TURNS" >> "$log" 2>&1
    echo "=== $(date -Is)  $b  exit=$? ===" | tee -a "$log"
done
echo "=== $(date -Is)  all 3 done ==="
