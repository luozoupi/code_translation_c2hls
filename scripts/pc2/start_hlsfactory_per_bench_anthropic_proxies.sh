#!/usr/bin/env bash
# Start one Anthropic queue proxy per bench on the login node (unique ports).
# High per-proxy workers for max LLM parallelism alongside one Vitis job/bench.
#
# Usage:
#   ./scripts/pc2/start_hlsfactory_per_bench_anthropic_proxies.sh \
#       --proxy-root <dir> --benches hlsfactory_2mm,hlsfactory_3mm \
#       --base-port 18300 [--workers 16]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

PROXY_ROOT=""
BENCHES=""
BASE_PORT="${C2HLS_PER_BENCH_ANTHROPIC_PROXY_BASE_PORT:-18300}"
LOGIN_HOST="${CHATHLS_LOGIN_HOST:-$(hostname -s)}"
MODEL="${ANTHROPIC_PROXY_MODEL:-${C2HLS_MODEL:-claude-sonnet-5}}"
WORKERS="${C2HLS_ANTHROPIC_QUEUE_WORKERS:-16}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --proxy-root) shift; PROXY_ROOT="$1"; shift ;;
    --benches) shift; BENCHES="$1"; shift ;;
    --base-port) shift; BASE_PORT="$1"; shift ;;
    --login-host) shift; LOGIN_HOST="$1"; shift ;;
    --workers) shift; WORKERS="$1"; shift ;;
    --model) shift; MODEL="$1"; shift ;;
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

# Map Claude_API → ANTHROPIC_API_KEY (avoid sourcing interactive bashrc).
if [[ -z "${ANTHROPIC_API_KEY:-}" || "${ANTHROPIC_API_KEY}" == "EMPTY" ]]; then
  if [[ -z "${Claude_API:-}" ]]; then
    line="$(grep -E '^[[:space:]]*Claude_API=' "${HOME}/.bashrc" 2>/dev/null | tail -1 || true)"
    if [[ -n "${line}" ]]; then
      val="${line#*Claude_API=}"
      val="${val%\"}"; val="${val#\"}"
      val="${val%\'}"; val="${val#\'}"
      export Claude_API="${val}"
    fi
  fi
  if [[ -n "${Claude_API:-}" ]]; then
    export ANTHROPIC_API_KEY="${Claude_API}"
  fi
fi
if [[ -z "${ANTHROPIC_API_KEY:-}" || ${#ANTHROPIC_API_KEY} -lt 20 ]]; then
  echo "ERROR: ANTHROPIC_API_KEY / Claude_API missing" >&2
  exit 1
fi
export OPENAI_API_KEY="${ANTHROPIC_API_KEY}"

PY="${C2HLS_PYTHON:-${C2HLS_ROOT}/.venv/bin/python}"
[[ -x "${PY}" ]] || PY=python3

IFS=',' read -r -a BENCH_ARR <<< "${BENCHES}"
BENCH_ARR=("${BENCH_ARR[@]// /}")

MAP_JSON="${PROXY_ROOT}/port_map.json"
TMP_MAP="$(mktemp)"
echo '{' > "${TMP_MAP}"
first=1
port="${BASE_PORT}"

for bench in "${BENCH_ARR[@]}"; do
  [[ -n "${bench}" ]] || continue
  while ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE ":${port}$"; do
    port=$((port + 1))
  done
  sess="${PROXY_ROOT}/${bench}"
  mkdir -p "${sess}"
  export C2HLS_ANTHROPIC_PROXY_PORT="${port}"
  export C2HLS_ANTHROPIC_QUEUE_WORKERS="${WORKERS}"
  export ANTHROPIC_PROXY_MODEL="${MODEL}"
  export C2HLS_MODEL="${MODEL}"
  url="$(
    C2HLS_ANTHROPIC_PROXY_PORT="${port}" \
      C2HLS_ANTHROPIC_QUEUE_WORKERS="${WORKERS}" \
      ANTHROPIC_PROXY_MODEL="${MODEL}" \
      C2HLS_MODEL="${MODEL}" \
      bash "${SCRIPT_DIR}/c2hls_anthropic_proxy.sh" "${sess}"
  )"
  # Rewrite host to login hostname if proxy bound 0.0.0.0 (compute must reach login).
  url="$("${PY}" -c "import json;from pathlib import Path;p=Path('${sess}')/'llm_endpoint.json';d=json.loads(p.read_text());d['url']=f'http://${LOGIN_HOST}:${port}/v1';d['host']='${LOGIN_HOST}';p.write_text(json.dumps(d,indent=2)+'\n');print(d['url'])")"
  if [[ "${first}" -eq 0 ]]; then
    echo ',' >> "${TMP_MAP}"
  fi
  first=0
  printf '  "%s": {"port": %s, "url": "%s", "workers": %s, "model": "%s"}' \
    "${bench}" "${port}" "${url}" "${WORKERS}" "${MODEL}" >> "${TMP_MAP}"
  echo "per-bench anthropic proxy ${bench} -> ${url} workers=${WORKERS}"
  port=$((port + 1))
done

echo >> "${TMP_MAP}"
echo '}' >> "${TMP_MAP}"
mv "${TMP_MAP}" "${MAP_JSON}"
echo "wrote ${MAP_JSON}"
