#!/usr/bin/env bash
# Launch all three HLSFactory DeepSeek-v4-flash flash+dataflow flavors in parallel.
# Each flavor gets an independent DeepSeek queue proxy (workers=1) on its own port.
# Compute: 28 combined-HLS nodes per flavor (all benches start immediately).
# No Slurm arrays. No Beijing peak pause.
#
# Usage:
#   ./scripts/pc2/start_hlsfactory_deepseek_flash_dataflow_triple.sh [--dry-run] [--stamp STAMP]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

STAMP="${BATCH_PARALLEL_STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stamp) shift; STAMP="$1"; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

TRIPLE_ROOT="${C2HLS_ROOT}/artifacts/pc2/hlsfactory_ds_v4f_triple_${STAMP}"
mkdir -p "${TRIPLE_ROOT}"

# flavor -> port
declare -A PORTS=(
  [skills]=18092
  [noskills]=18093
  [bare]=18094
)

MANIFEST="${TRIPLE_ROOT}/triple_manifest.json"
echo "{}" > "${MANIFEST}"

echo "=== HLSFactory DeepSeek-v4-flash triple launch stamp=${STAMP} dry_run=${DRY_RUN} ==="

for FLAVOR in skills noskills bare; do
  PORT="${PORTS[${FLAVOR}]}"
  PROXY_DIR="${TRIPLE_ROOT}/proxy_${FLAVOR}"
  mkdir -p "${PROXY_DIR}"

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    ENDPOINT_URL="http://127.0.0.1:${PORT}/v1"
    echo "[dry-run] flavor=${FLAVOR} would start proxy on :${PORT}"
  else
    echo "starting DeepSeek queue proxy flavor=${FLAVOR} port=${PORT} workers=1"
    DEEPSEEK_PROXY_MODEL=deepseek-v4-flash \
      C2HLS_MODEL=deepseek-v4-flash \
      CHATHLS_DEEPSEEK_PROXY_PORT="${PORT}" \
      CHATHLS_DEEPSEEK_QUEUE_WORKERS=1 \
      "${SCRIPT_DIR}/c2hls_deepseek_proxy.sh" "${PROXY_DIR}"
    ENDPOINT_URL="$(
      "${C2HLS_PYTHON:-python3}" -c "import json;print(json.load(open('${PROXY_DIR}/llm_endpoint.json'))['url'])"
    )"
    echo "proxy ready flavor=${FLAVOR} url=${ENDPOINT_URL}"
  fi

  ONE_ARGS=(--flavor "${FLAVOR}" --stamp "${STAMP}" --endpoint-url "${ENDPOINT_URL}")
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    ONE_ARGS+=(--dry-run)
  fi

  echo "launching campaign flavor=${FLAVOR}"
  "${SCRIPT_DIR}/start_hlsfactory_deepseek_flash_dataflow_one.sh" "${ONE_ARGS[@]}"

  case "${FLAVOR}" in
    skills) PREFIX="batch_parallel_hlsfactory_ds_v4f_skills" ;;
    noskills) PREFIX="batch_parallel_hlsfactory_ds_v4f_noskills" ;;
    bare) PREFIX="batch_parallel_hlsfactory_ds_v4f_bare" ;;
  esac
  CAMPAIGN_ROOT="${C2HLS_ROOT}/artifacts/pc2/${PREFIX}_${STAMP}"

  "${C2HLS_PYTHON:-python3}" - <<PY
import json
from pathlib import Path
p = Path("${MANIFEST}")
doc = json.loads(p.read_text())
doc["stamp"] = "${STAMP}"
doc.setdefault("flavors", {})["${FLAVOR}"] = {
    "port": int("${PORT}"),
    "proxy_dir": "${PROXY_DIR}",
    "endpoint_url": "${ENDPOINT_URL}",
    "campaign_root": "${CAMPAIGN_ROOT}",
    "queue_workers": 1,
}
p.write_text(json.dumps(doc, indent=2) + "\n")
PY
done

echo "=== triple launched ==="
echo "manifest=${MANIFEST}"
cat "${MANIFEST}"
