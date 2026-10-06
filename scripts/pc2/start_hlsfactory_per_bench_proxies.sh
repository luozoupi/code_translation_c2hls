#!/usr/bin/env bash
# Start one DeepSeek queue proxy per bench on login node (unique ports).
#
# Usage:
#   ./scripts/pc2/start_hlsfactory_per_bench_proxies.sh \
#       --proxy-root <dir> --benches hlsfactory_gemm,hlsfactory_atax \
#       --base-port 18200
#
# Writes:
#   <proxy-root>/<bench>/llm_endpoint.json
#   <proxy-root>/port_map.json   {bench: {port,url,pid_file,...}}
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

PROXY_ROOT=""
BENCHES=""
BASE_PORT="${C2HLS_PER_BENCH_PROXY_BASE_PORT:-18200}"
LOGIN_HOST="${CHATHLS_LOGIN_HOST:-$(hostname -s)}"
MODEL="${DEEPSEEK_PROXY_MODEL:-deepseek-v4-flash}"
# Never inherit a non-DeepSeek C2HLS_MODEL into the proxy (breaks DeepSeek API).
case "${MODEL}" in
  deepseek-*) ;;
  *)
    echo "WARNING: forcing DeepSeek model (was MODEL=${MODEL})" >&2
    MODEL="deepseek-v4-flash"
    ;;
esac
WORKERS="${CHATHLS_DEEPSEEK_QUEUE_WORKERS:-1}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --proxy-root) shift; PROXY_ROOT="$1"; shift ;;
    --benches) shift; BENCHES="$1"; shift ;;
    --base-port) shift; BASE_PORT="$1"; shift ;;
    --login-host) shift; LOGIN_HOST="$1"; shift ;;
    --workers) shift; WORKERS="$1"; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "${PROXY_ROOT}" || -z "${BENCHES}" ]]; then
  echo "ERROR: --proxy-root and --benches required" >&2
  exit 2
fi
if [[ "${PROXY_ROOT}" != /* ]]; then
  PROXY_ROOT="${C2HLS_ROOT}/${PROXY_ROOT}"
fi
mkdir -p "${PROXY_ROOT}"

IFS=',' read -r -a BENCH_ARR <<< "${BENCHES}"
BENCH_ARR=("${BENCH_ARR[@]// /}")

# Load DeepSeek key once (avoid EMPTY placeholder).
if [[ "${OPENAI_API_KEY:-}" == "EMPTY" || "${OPENAI_API_KEY:-}" == "empty" ]]; then
  unset OPENAI_API_KEY || true
fi
CHATHLS_ROOT="${CHATHLS_ROOT:-/scratch/hpc-prf-llmfpga/asa582/projects/test-chathls/ChatHLS-ACL-26}"
# shellcheck disable=SC1091
source "${CHATHLS_ROOT}/scripts/pc2/setup_deepseek_api.sh"
if [[ -z "${OPENAI_API_KEY:-}" || "${OPENAI_API_KEY}" == "EMPTY" || ${#OPENAI_API_KEY} -lt 20 ]]; then
  echo "ERROR: OPENAI_API_KEY missing/invalid after setup_deepseek_api.sh" >&2
  exit 1
fi

MAP_JSON="${PROXY_ROOT}/port_map.json"
TMP_MAP="$(mktemp)"
echo '{' > "${TMP_MAP}"
first=1
port="${BASE_PORT}"
idx=0

for bench in "${BENCH_ARR[@]}"; do
  [[ -n "${bench}" ]] || continue
  # skip used ports and the shared login5:18092 / ChatHLS 18082 proxies
  while ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE ":${port}$" \
     || [[ "${port}" == "18082" || "${port}" == "18092" ]]; do
    port=$((port + 1))
  done
  dir="${PROXY_ROOT}/${bench}"
  mkdir -p "${dir}"
  echo "[$((idx+1))/${#BENCH_ARR[@]}] proxy ${bench} -> :${port}"
  DEEPSEEK_PROXY_MODEL="${MODEL}" \
    CHATHLS_DEEPSEEK_PROXY_PORT="${port}" \
    CHATHLS_DEEPSEEK_QUEUE_WORKERS="${WORKERS}" \
    CHATHLS_LOGIN_HOST="${LOGIN_HOST}" \
    OPENAI_API_KEY="${OPENAI_API_KEY}" \
    "${SCRIPT_DIR}/c2hls_deepseek_proxy.sh" "${dir}" >/dev/null
  url="http://${LOGIN_HOST}:${port}/v1"
  # force login host in endpoint (compute nodes cannot use 127.0.0.1)
  python3 - <<PY
import json
from pathlib import Path
p = Path("${dir}") / "llm_endpoint.json"
d = json.loads(p.read_text()) if p.is_file() else {}
d.update({"url": "${url}", "host": "${LOGIN_HOST}", "port": ${port}, "bench": "${bench}", "workers": ${WORKERS}})
p.write_text(json.dumps(d, indent=2) + "\n")
PY
  if [[ "${first}" -eq 0 ]]; then
    echo ',' >> "${TMP_MAP}"
  fi
  first=0
  printf '  "%s": {"port": %s, "url": "%s", "dir": "%s"}' \
    "${bench}" "${port}" "${url}" "${dir}" >> "${TMP_MAP}"
  # health
  if ! curl -sf --max-time 5 "http://127.0.0.1:${port}/v1/models" >/dev/null \
     && ! curl -sf --max-time 5 "http://127.0.0.1:${port}/health" >/dev/null; then
    echo "WARNING: proxy health check weak for ${bench} :${port}" >&2
  fi
  port=$((port + 1))
  idx=$((idx + 1))
done

echo >> "${TMP_MAP}"
echo '}' >> "${TMP_MAP}"
mv "${TMP_MAP}" "${MAP_JSON}"
echo "wrote ${MAP_JSON} (${idx} proxies, login=${LOGIN_HOST})"
