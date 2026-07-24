#!/usr/bin/env bash
# After parallel bpmachfd-df-* jobs finish, export flash_selected + dataflow_selected.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

CAMPAIGN_ROOT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --campaign-root) shift; CAMPAIGN_ROOT="$1"; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
[[ -n "${CAMPAIGN_ROOT}" ]] || { echo "ERROR: --campaign-root required" >&2; exit 2; }

PY="${C2HLS_PYTHON:-${C2HLS_ROOT}/.venv/bin/python}"
[[ -x "${PY}" ]] || PY=python3

FLASH_BUNDLE="${C2HLS_ROOT}/artifacts/pc2/flash_selected_bundle/$(basename "${CAMPAIGN_ROOT}")"
DATAFLOW_BUNDLE="${C2HLS_ROOT}/artifacts/pc2/dataflow_selected_bundle/$(basename "${CAMPAIGN_ROOT}")"

pc2_log "exporting flash_selected -> ${FLASH_BUNDLE}"
"${PY}" "${SCRIPT_DIR}/export_flash_selected_bundle.py" --pc2 \
  --matrix-root "${CAMPAIGN_ROOT}" \
  --out-root "${C2HLS_ROOT}/artifacts/pc2/flash_selected_bundle" \
  || true

pc2_log "exporting dataflow_selected -> ${DATAFLOW_BUNDLE}"
mkdir -p "${DATAFLOW_BUNDLE}"
"${PY}" "${SCRIPT_DIR}/export_post_flash_dataflow_csynth_bundle.py" \
  --matrix-root "${CAMPAIGN_ROOT}" \
  --flash-bundle-root "${FLASH_BUNDLE}" \
  --kernel-bundle "${DATAFLOW_BUNDLE}" \
  --force \
  || true

ln -sfn "${FLASH_BUNDLE}" "${CAMPAIGN_ROOT}/flash_selected"
ln -sfn "${DATAFLOW_BUNDLE}" "${CAMPAIGN_ROOT}/dataflow_selected"
pc2_log "export done"
echo "flash_selected=${FLASH_BUNDLE}"
echo "dataflow_selected=${DATAFLOW_BUNDLE}"
