#!/usr/bin/env bash
# Rerun the 3 most-infeasible-on-Artix benchmarks on U50 @ 3.33 ns.
# Uses Fix A: variants[-1] as GT, no g++ preflight.
# shellcheck disable=SC1091
source "$(dirname "$0")/scripts/bootstrap_site.sh" "$@"
source "$(dirname "$0")/scripts/source_local_env.sh"
cd "${C2HLS_ROOT}" || exit 1
if [[ "${C2HLS_SITE:-team}" == "pc2" ]]; then
  if [[ -n "${C2HLS_VITIS_SETTINGS:-}" && -f "${C2HLS_VITIS_SETTINGS}" ]]; then
    # shellcheck disable=SC1090
    source "${C2HLS_VITIS_SETTINGS}" 2>/dev/null || true
  fi
else
  # shellcheck disable=SC1091
  source /mnt/data/luo00466/Xilinx/2025.2/Vitis/settings64.sh 2>/dev/null || true
fi
eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate py310_2 2>/dev/null || true

MODEL="${MODEL:-claude-haiku-4-5-20251001}"
TURNS="${TURNS:-3}"
PC2_FLAG=""
if [[ "${C2HLS_SITE:-team}" == "pc2" ]]; then
  PC2_FLAG="--pc2"
fi

mkdir -p logs
for b in nw lud kmeans; do
    log="logs/rerun_${b}.log"
    echo "=== $(date -Is)  $b  model=$MODEL turns=$TURNS ===" | tee "$log"
    python c2hls.py ${PC2_FLAG} --bench "$b" --model "$MODEL" --turns "$TURNS" >> "$log" 2>&1
    echo "=== $(date -Is)  $b  exit=$? ===" | tee -a "$log"
done
echo "=== $(date -Is)  all 3 done ==="
