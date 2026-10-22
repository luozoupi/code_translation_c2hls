#!/bin/bash
# resubmit_sd_heat3d_otus.sh [--dry-run] : on Otus, submit sd_heat3d_ds, the DeepSeek SODA heat3d run pending on Fir on
# 2026-10-22 (Fir job 66475350: the 12 h tier after a 6 h TIMEOUT, 192G). Same --export as on Fir; only the paths,
# the include dir (native $XILINX_HLS/include) and the proxy URL (controlled/ds_proxy_url.txt, start_ds_proxy_otus.sh) change.
set -eo pipefail
CC=$(dirname "$(dirname "$(readlink -f "$0")")")
[ "${CC_CLUSTER:-}" = fir ] && [ "${1:-}" != --dry-run ] && { echo "this is Fir: only --dry-run here"; exit 1; }
URL=$(cat "$CC/controlled/ds_proxy_url.txt")
[ "${1:-}" = --dry-run ] || curl -s -m 5 -o /dev/null "${URL%/v1}/" || { echo "DeepSeek proxy $URL not reachable: run otus/start_ds_proxy_otus.sh"; exit 1; }
B=$CC/study/bench_soda
EXP="ALL,CONTROLLED=1,C2HLS_THINKING=enabled,C2HLS_FLASH_MAX_TOKENS=262144,CTRL_MIN_MAX_TOKENS=262144"
EXP+=",OUT_ROOT=$CC/study/bench_soda_runs/runs,BENCH_ROOT=$B,BENCH=heat3d,SEED_CODE=$B/heat3d/hls_baseline.cpp"
EXP+=",FLAVOR=skills,REFINED=1,DSP_WINDOW=1320:8121,FLASH_REFINED_TOTAL_MACS=2768240640,FLASH_REFINED_DSP_PER_MAC=1"
EXP+=",FLASH_REFINED_MEM_CYCLES=2097152,MODEL=deepseek-v4-flash,CLAUDE_CODE_SHIM_URL=$URL,TAG=sd_heat3d_ds"
CMD=(sbatch --parsable --time=12:00:00 --job-name=sd_heat3d_ds --mem=192G --export="$EXP" otus/sbatch_flash_cc_otus.sh)
cd "$CC"; mkdir -p logs
if [ "${1:-}" = --dry-run ]; then echo "(cd $CC && ${CMD[*]})"; exit 0; fi
J=$("${CMD[@]}"); echo "sd_heat3d_ds $J 12:00:00" | tee -a "$CC/study/bench_soda_runs/ds_submitted_otus.txt"
