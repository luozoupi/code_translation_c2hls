#!/usr/bin/env bash
# Wait for multistep pipelined matrix completion, then cosim all roles + JSONL export.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

PY="${C2HLS_PYTHON:-python3}"
MULTISTEP_STAMP="${1:-20260628_fixed_cosim_multistep_full_pipelined}"
POLL_SEC="${PC2_PIPELINE_POLL_SEC:-120}"
LOG="${C2HLS_ROOT}/artifacts/pc2/pipelines/wait_multistep_then_postprocess.log"

STAMP_SUFFIX="${MULTISTEP_STAMP}"
if [[ "${STAMP_SUFFIX}" != *_pipelined ]]; then
  STAMP_SUFFIX="${MULTISTEP_STAMP}_pipelined"
fi

ARTIFACT_DIR="${C2HLS_ROOT}/artifacts/pc2/multistep_fixed_cosim_aav_n_${STAMP_SUFFIX}"
MATRIX="${ARTIFACT_DIR}/matrix.json"
DATE_PREFIX="$(printf '%s' "${MULTISTEP_STAMP}" | grep -oE '[0-9]{8}' | head -1)"
COSIM_STAMP="fixed_cosim_multistep_${DATE_PREFIX}"
JSONL_OUT="${C2HLS_ROOT}/misc/hlsfactory_fixed_cosim_multistep_u280_${DATE_PREFIX}.jsonl"

mkdir -p "$(dirname "${LOG}")"
exec >> "${LOG}" 2>&1

plog() { printf '[%s] %s\n' "$(date -Is)" "$*"; }

plog "waiting for multistep matrix ${MATRIX}"
while [[ ! -f "${MATRIX}" ]]; do
  sleep "${POLL_SEC}"
done

expected="$(python3 - <<'PY' "${MATRIX}"
import json, sys
rows = json.loads(open(sys.argv[1]).read())
print(len(rows))
PY
)"

while true; do
  done_count="$(python3 - <<'PY' "${MATRIX}"
import json, sys
rows = json.loads(open(sys.argv[1]).read())
print(sum(1 for r in rows if r.get("status") == "ok"))
PY
)"
  plog "matrix progress ok=${done_count}/${expected}"
  if [[ "${done_count}" -ge "${expected}" ]] && [[ "${expected}" -gt 0 ]]; then
    break
  fi
  sleep "${POLL_SEC}"
done

plog "multistep complete; submitting cosim stamp=${COSIM_STAMP}"
export C2HLS_MULTISTEP_COSIM_STAMP="${COSIM_STAMP}"
export C2HLS_MULTISTEP_COSIM_ARTIFACT_GLOB="multistep_fixed_cosim_aav_n_${STAMP_SUFFIX}"
export PC2_COSIM_WALLTIME="${PC2_COSIM_PIPELINE_WALLTIME:-13:00:00}"
export C2HLS_COSIM_TIMEOUT="${C2HLS_COSIM_TIMEOUT:-43200}"
"${SCRIPT_DIR}/submit_multistep_cosim_all.sh" \
  --stamp "${COSIM_STAMP}" \
  --artifact-glob "multistep_fixed_cosim_aav_n_${STAMP_SUFFIX}"

plog "waiting for cosim + exporting JSONL"
"${SCRIPT_DIR}/wait_multistep_cosim_export_jsonl.sh" \
  --multistep-stamp "${STAMP_SUFFIX}" \
  --cosim-stamp "${COSIM_STAMP}" \
  --output "${JSONL_OUT}"

plog "exporting multistep speedup CSV"
"${PY}" "${SCRIPT_DIR}/export_multistep_csynth_speedup_csv.py" \
  --stamp "${STAMP_SUFFIX}" \
  --out-dir "${C2HLS_ROOT}/artifacts/pc2/analysis/${STAMP_SUFFIX}"

plog "postprocess complete"
