#!/usr/bin/env bash
#SBATCH --job-name=c2hls-ds-reach
#SBATCH -A hpc-prf-llmfpga
#SBATCH -p normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH -t 01:00:00
#SBATCH -o slurm-c2hls-ds-reach-%j.out
#SBATCH -e slurm-c2hls-ds-reach-%j.err
#
# Compute-node reachability gate for the DeepSeek login-node queue proxy
# started by c2hls_deepseek_proxy.sh. Confirms the proxy URL is reachable
# FROM a compute node (login-node internet access does not guarantee compute
# nodes can reach the same host:port) before an external_llm campaign starts.
#
# The -A/-p defaults above match common.sh's PC2_SLURM_ACCOUNT/PC2_COMPUTE_PARTITION
# defaults; override at submit time if needed, e.g.:
#   sbatch -A "${PC2_SLURM_ACCOUNT:-hpc-prf-llmfpga}" -p "${PC2_COMPUTE_PARTITION:-normal}" \
#     scripts/pc2/c2hls_deepseek_reachability.sbatch.sh
#
# Usage:
#   CAMPAIGN_ROOT=<dir> sbatch scripts/pc2/c2hls_deepseek_reachability.sbatch.sh
#
# Env:
#   CAMPAIGN_ROOT / C2HLS_DEEPSEEK_CAMPAIGN_DIR   dir containing llm_endpoint.json
#                                                  or deepseek_endpoint.json (required, one of)
#   C2HLS_DEEPSEEK_REACH_TIMEOUT_SEC               overall timeout in seconds (default 1800)
#   C2HLS_DEEPSEEK_REACH_POLL_SEC                  poll interval in seconds (default 15)
#
# On success writes "${CAMPAIGN_DIR}/reachability_ok.json" ({ok, url, host, timestamp})
# and exits 0. On timeout exits non-zero without writing that file.

set -euo pipefail

CAMPAIGN_DIR="${CAMPAIGN_ROOT:-${C2HLS_DEEPSEEK_CAMPAIGN_DIR:-}}"
if [[ -z "${CAMPAIGN_DIR}" ]]; then
  echo "c2hls_deepseek_reachability: set CAMPAIGN_ROOT or C2HLS_DEEPSEEK_CAMPAIGN_DIR" >&2
  exit 1
fi

TIMEOUT_SEC="${C2HLS_DEEPSEEK_REACH_TIMEOUT_SEC:-1800}"
POLL_SEC="${C2HLS_DEEPSEEK_REACH_POLL_SEC:-15}"
DEADLINE=$((SECONDS + TIMEOUT_SEC))

echo "c2hls_deepseek_reachability: gate on $(hostname -s) campaign=${CAMPAIGN_DIR} timeout=${TIMEOUT_SEC}s"

load_url() {
  local f
  for f in "${CAMPAIGN_DIR}/llm_endpoint.json" "${CAMPAIGN_DIR}/deepseek_endpoint.json"; do
    if [[ -f "${f}" ]]; then
      python3 -c "
import json
from pathlib import Path
try:
    doc = json.loads(Path('${f}').read_text())
except Exception:
    raise SystemExit(1)
url = str(doc.get('url', '')).rstrip('/')
if not url:
    raise SystemExit(1)
print(url)
" 2>/dev/null && return 0
    fi
  done
  return 1
}

while (( SECONDS < DEADLINE )); do
  URL="$(load_url || true)"
  if [[ -n "${URL}" ]]; then
    if curl -sf --max-time 5 "${URL}/models" >/dev/null 2>&1 || curl -sf --max-time 5 "${URL}/health" >/dev/null 2>&1; then
      python3 - "${CAMPAIGN_DIR}" "${URL}" "$(hostname -s)" <<'PY'
import json
import sys
import time
from pathlib import Path

root, url, host = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
doc = {
    "ok": True,
    "url": url,
    "host": host,
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
(root / "reachability_ok.json").write_text(json.dumps(doc, indent=2) + "\n")
print(json.dumps(doc, indent=2))
PY
      echo "REACHABILITY OK url=${URL}"
      exit 0
    fi
    echo "$(date -Is) endpoint present (${URL}) but not reachable yet; retrying..."
  else
    echo "$(date -Is) waiting for llm_endpoint.json/deepseek_endpoint.json in ${CAMPAIGN_DIR}"
  fi
  sleep "${POLL_SEC}"
done

echo "c2hls_deepseek_reachability: FAILED within ${TIMEOUT_SEC}s" >&2
exit 2
