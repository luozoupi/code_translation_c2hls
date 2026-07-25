#!/usr/bin/env bash
# 1-bench-per-corpus multistep pilot (DeepSeek v4-flash + RAG2 + lat-opt).
#
# Benches:
#   chathls  -> chathls_gesummv
#   tier_A   -> spector_hls_fir
#   tier_B   -> machsuite_gemm_ncubed
#
# Usage:
#   ./scripts/pc2/start_all_multistep_pilot_triple.sh [--dry-run] [--stamp STAMP]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

DRY_RUN=0
STAMP="${BATCH_PARALLEL_STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
# Default deliberately ignores a stale C2HLS_MODEL=deepseek-chat in the shell.
MODEL="deepseek-v4-flash"
MODEL_SET=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --stamp) shift; STAMP="$1"; shift ;;
    --model) shift; MODEL="$1"; MODEL_SET=1; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

export C2HLS_MODEL="${MODEL}"
export DEEPSEEK_PROXY_MODEL="${MODEL}"
export BATCH_PARALLEL_EXTERNAL_MODEL="${MODEL}"

SEQ_ROOT="${C2HLS_ROOT}/artifacts/pc2/multistep_pilot_triple_${STAMP}"
mkdir -p "${SEQ_ROOT}"/{proxy_chathls,proxy_tier_a,proxy_tier_b}

declare -A PORTS=(
  [chathls]=18094
  [tier_a]=18095
  [tier_b]=18096
)
declare -A PROXY_DIRS=(
  [chathls]="${SEQ_ROOT}/proxy_chathls"
  [tier_a]="${SEQ_ROOT}/proxy_tier_a"
  [tier_b]="${SEQ_ROOT}/proxy_tier_b"
)

echo "=== Multistep pilot triple model=${MODEL} stamp=${STAMP} ==="
echo "seq_root=${SEQ_ROOT}"
echo "benches: chathls_gesummv | spector_hls_fir | machsuite_gemm_ncubed"

# Stop any stale proxies on these ports (old deepseek-chat listeners).
for port in 18094 18095 18096; do
  pids="$(ss -ltnp 2>/dev/null | awk -v p=":${port}" '$4 ~ p {print}' | sed -n 's/.*pid=\([0-9]*\).*/\1/p' | sort -u || true)"
  for pid in ${pids}; do
    echo "stopping stale listener pid=${pid} on :${port}"
    kill "${pid}" 2>/dev/null || true
  done
done
sleep 1

if [[ "${DRY_RUN}" -eq 1 ]]; then
  for corpus in chathls tier_a tier_b; do
    port="${PORTS[$corpus]}"
    url="http://127.0.0.1:${port}/v1"
    echo "[dry-run] would start proxy ${corpus} on :${port} model=${MODEL}"
    "${SCRIPT_DIR}/start_multistep_deepseek_rag2_one.sh" \
      --corpus "${corpus}" \
      --pilot \
      --model "${MODEL}" \
      --stamp "${STAMP}_${corpus}" \
      --endpoint-url "${url}" \
      --dry-run
  done
  echo "dry-run ok"
  echo "seq_root=${SEQ_ROOT}"
  exit 0
fi

if [[ "${OPENAI_API_KEY:-}" == "EMPTY" || "${OPENAI_API_KEY:-}" == "empty" ]]; then
  unset OPENAI_API_KEY || true
fi

for corpus in chathls tier_a tier_b; do
  port="${PORTS[$corpus]}"
  pdir="${PROXY_DIRS[$corpus]}"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] starting DeepSeek proxy ${corpus} on :${port} model=${MODEL}"
  CHATHLS_DEEPSEEK_PROXY_PORT="${port}" \
  CHATHLS_DEEPSEEK_QUEUE_WORKERS=1 \
  DEEPSEEK_PROXY_MODEL="${MODEL}" \
  C2HLS_MODEL="${MODEL}" \
    "${SCRIPT_DIR}/c2hls_deepseek_proxy.sh" "${pdir}"
  url="$("${C2HLS_PYTHON:-python3}" -c "import json; print(json.load(open('${pdir}/llm_endpoint.json'))['url'])")"
  model_written="$("${C2HLS_PYTHON:-python3}" -c "import json; print(json.load(open('${pdir}/llm_endpoint.json')).get('model',''))")"
  echo "proxy ${corpus}: ${url} model=${model_written}"
  echo "${url}" > "${SEQ_ROOT}/endpoint_${corpus}.txt"
done

for corpus in chathls tier_a tier_b; do
  pdir="${PROXY_DIRS[$corpus]}"
  url="$("${C2HLS_PYTHON:-python3}" -c "import json; print(json.load(open('${pdir}/llm_endpoint.json'))['url'])")"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] submitting ${corpus} pilot"
  "${SCRIPT_DIR}/start_multistep_deepseek_rag2_one.sh" \
    --corpus "${corpus}" \
    --pilot \
    --model "${MODEL}" \
    --stamp "${STAMP}_${corpus}" \
    --endpoint-url "${url}"
  case "${corpus}" in
    chathls) prefix="batch_parallel_chathls_ms_pilot_v4f_lat" ;;
    tier_a) prefix="batch_parallel_tier_a_ms_pilot_v4f_lat" ;;
    tier_b) prefix="batch_parallel_tier_b_ms_pilot_v4f_lat" ;;
  esac
  camp="${C2HLS_ROOT}/artifacts/pc2/${prefix}_${STAMP}_${corpus}"
  echo "${camp}" > "${SEQ_ROOT}/campaign_${corpus}.txt"
  ln -sfn "${camp}" "${SEQ_ROOT}/campaign_${corpus}"
done

{
  echo "stamp=${STAMP}"
  echo "model=${MODEL}"
  echo "seq_root=${SEQ_ROOT}"
  echo "chathls_bench=chathls_gesummv"
  echo "tier_a_bench=spector_hls_fir"
  echo "tier_b_bench=machsuite_gemm_ncubed"
  for corpus in chathls tier_a tier_b; do
    echo "${corpus}_port=${PORTS[$corpus]}"
    echo "${corpus}_proxy=${PROXY_DIRS[$corpus]}"
    echo "${corpus}_campaign=$(cat "${SEQ_ROOT}/campaign_${corpus}.txt")"
  done
} | tee "${SEQ_ROOT}/launch_summary.txt"

echo "=== launched ==="
cat "${SEQ_ROOT}/launch_summary.txt"
