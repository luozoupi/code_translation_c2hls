#!/usr/bin/env bash
# Start 3 dedicated DeepSeek queue proxies + 3 multistep campaigns in parallel:
#   chathls_ready (16) @ :18094
#   tier_A_ready  (54) @ :18095
#   tier_B_ready  (18) @ :18096
#
# Usage:
#   ./scripts/pc2/start_all_multistep_deepseek_triple.sh [--dry-run] [--stamp STAMP]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

DRY_RUN=0
STAMP="${BATCH_PARALLEL_STAMP:-$(date -u +%Y%m%d_%H%M%S)}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --stamp) shift; STAMP="$1"; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

SEQ_ROOT="${C2HLS_ROOT}/artifacts/pc2/multistep_deepseek_triple_${STAMP}"
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

echo "=== Multistep DeepSeek triple launch stamp=${STAMP} ==="
echo "seq_root=${SEQ_ROOT}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  for corpus in chathls tier_a tier_b; do
    port="${PORTS[$corpus]}"
    url="http://127.0.0.1:${port}/v1"
    echo "[dry-run] would start proxy ${corpus} on :${port}"
    echo "[dry-run] would submit ${corpus} campaign endpoint=${url}"
    "${SCRIPT_DIR}/start_multistep_deepseek_rag2_one.sh" \
      --corpus "${corpus}" \
      --stamp "${STAMP}_${corpus}" \
      --endpoint-url "${url}" \
      --dry-run
  done
  echo "dry-run ok"
  echo "seq_root=${SEQ_ROOT}"
  exit 0
fi

# Ensure OPENAI_API_KEY is available for proxies (unset EMPTY placeholders).
if [[ "${OPENAI_API_KEY:-}" == "EMPTY" || "${OPENAI_API_KEY:-}" == "empty" ]]; then
  unset OPENAI_API_KEY || true
fi

for corpus in chathls tier_a tier_b; do
  port="${PORTS[$corpus]}"
  pdir="${PROXY_DIRS[$corpus]}"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] starting DeepSeek proxy ${corpus} on :${port} -> ${pdir}"
  CHATHLS_DEEPSEEK_PROXY_PORT="${port}" \
  CHATHLS_DEEPSEEK_QUEUE_WORKERS=1 \
    "${SCRIPT_DIR}/c2hls_deepseek_proxy.sh" "${pdir}"
  url="$("${C2HLS_PYTHON:-python3}" -c "import json; print(json.load(open('${pdir}/llm_endpoint.json'))['url'])")"
  echo "proxy ${corpus}: ${url}"
  echo "${url}" > "${SEQ_ROOT}/endpoint_${corpus}.txt"
done

CAMPAIGN_ROOTS=()
for corpus in chathls tier_a tier_b; do
  pdir="${PROXY_DIRS[$corpus]}"
  url="$("${C2HLS_PYTHON:-python3}" -c "import json; print(json.load(open('${pdir}/llm_endpoint.json'))['url'])")"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] submitting ${corpus} campaign"
  "${SCRIPT_DIR}/start_multistep_deepseek_rag2_one.sh" \
    --corpus "${corpus}" \
    --stamp "${STAMP}_${corpus}" \
    --endpoint-url "${url}"
  # Reconstruct campaign root from the one-shot naming convention.
  case "${corpus}" in
    chathls) prefix="batch_parallel_chathls_ms_ds_rag2_lat" ;;
    tier_a) prefix="batch_parallel_tier_a_ms_ds_rag2_lat" ;;
    tier_b) prefix="batch_parallel_tier_b_ms_ds_rag2_lat" ;;
  esac
  camp="${C2HLS_ROOT}/artifacts/pc2/${prefix}_${STAMP}_${corpus}"
  CAMPAIGN_ROOTS+=("${camp}")
  echo "${camp}" > "${SEQ_ROOT}/campaign_${corpus}.txt"
  ln -sfn "${camp}" "${SEQ_ROOT}/campaign_${corpus}"
done

{
  echo "stamp=${STAMP}"
  echo "seq_root=${SEQ_ROOT}"
  for corpus in chathls tier_a tier_b; do
    echo "${corpus}_port=${PORTS[$corpus]}"
    echo "${corpus}_proxy=${PROXY_DIRS[$corpus]}"
    echo "${corpus}_campaign=$(cat "${SEQ_ROOT}/campaign_${corpus}.txt")"
  done
} | tee "${SEQ_ROOT}/launch_summary.txt"

echo "=== launched ==="
cat "${SEQ_ROOT}/launch_summary.txt"
