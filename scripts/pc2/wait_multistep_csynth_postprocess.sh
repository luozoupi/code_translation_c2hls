#!/usr/bin/env bash
# Wait for pipelined multistep matrix completion, then export JSONL + CSV + MD summary.
#
# Usage:
#   ./scripts/pc2/wait_multistep_csynth_postprocess.sh \
#     --variant nav_n \
#     --stamp 20260629_fixed_cosim_multistep_nav_n_pipelined
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

PY="${C2HLS_PYTHON:-python3}"
VARIANT="aav_n"
MULTISTEP_STAMP=""
JSONL_OUT=""
BASELINE_JSONL="${C2HLS_ROOT}/misc/hlsfactory_baseline_u280_20260616_benchmarks_full_cosim.jsonl"
POLL_SEC="${PC2_PIPELINE_POLL_SEC:-120}"
LOG="${C2HLS_ROOT}/artifacts/pc2/pipelines/wait_multistep_csynth_postprocess.log"
EXPECTED_BENCHES=28

while [[ $# -gt 0 ]]; do
  case "$1" in
    --variant) VARIANT="$2"; shift 2 ;;
    --stamp) MULTISTEP_STAMP="$2"; shift 2 ;;
    --output) JSONL_OUT="$2"; shift 2 ;;
    --baseline-jsonl) BASELINE_JSONL="$2"; shift 2 ;;
    --expected-benches) EXPECTED_BENCHES="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,10p' "$0"
      exit 0
      ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "${MULTISTEP_STAMP}" ]]; then
  echo "ERROR: --stamp required" >&2
  exit 2
fi

STAMP_SUFFIX="${MULTISTEP_STAMP}"
if [[ "${STAMP_SUFFIX}" != *_pipelined ]]; then
  STAMP_SUFFIX="${MULTISTEP_STAMP}_pipelined"
fi

ARTIFACT_DIR="${C2HLS_ROOT}/artifacts/pc2/multistep_fixed_cosim_${VARIANT}_${STAMP_SUFFIX}"
MATRIX="${ARTIFACT_DIR}/matrix.json"
DATE_PREFIX="$(printf '%s' "${MULTISTEP_STAMP}" | grep -oE '[0-9]{8}' | head -1)"
JSONL_OUT="${JSONL_OUT:-${C2HLS_ROOT}/misc/hlsfactory_fixed_cosim_multistep_${VARIANT}_u280_${DATE_PREFIX}.jsonl}"
ANALYSIS_DIR="${C2HLS_ROOT}/artifacts/pc2/analysis/${STAMP_SUFFIX}"
SUMMARY_MD="${ANALYSIS_DIR}/summary.md"

mkdir -p "$(dirname "${LOG}")" "$(dirname "${JSONL_OUT}")" "${ANALYSIS_DIR}"

plog() { printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "${LOG}"; }

plog "waiting for multistep matrix variant=${VARIANT} path=${MATRIX} expected=${EXPECTED_BENCHES}"

while [[ ! -f "${MATRIX}" ]]; do
  sleep "${POLL_SEC}"
done

while true; do
  read -r ok_count fail_count row_count <<<"$("${PY}" - <<PY "${MATRIX}"
import json, sys
rows = json.loads(open(sys.argv[1]).read())
ok = sum(1 for r in rows if r.get("status") == "ok")
fail = sum(1 for r in rows if r.get("status") == "fail")
print(ok, fail, len(rows))
PY
)"
  plog "matrix progress rows=${row_count}/${EXPECTED_BENCHES} ok=${ok_count} fail=${fail_count}"
  if [[ "${row_count}" -ge "${EXPECTED_BENCHES}" ]]; then
    plog "matrix complete rows=${row_count} ok=${ok_count} fail=${fail_count}"
    break
  fi
  sleep "${POLL_SEC}"
done

plog "export JSONL -> ${JSONL_OUT}"
"${PY}" "${C2HLS_ROOT}/misc/export_pc2_fixed_cosim_multistep_jsonl.py" \
  --baseline-jsonl "${BASELINE_JSONL}" \
  --variant "${VARIANT}" \
  --multistep-stamp "${STAMP_SUFFIX}" \
  --output "${JSONL_OUT}" | tee -a "${LOG}"

plog "export csynth speedup CSV"
"${PY}" "${SCRIPT_DIR}/export_multistep_csynth_speedup_csv.py" \
  --variant "${VARIANT}" \
  --stamp "${STAMP_SUFFIX}" \
  --baseline-jsonl "${BASELINE_JSONL}" \
  --out-dir "${ANALYSIS_DIR}" | tee -a "${LOG}"

plog "write summary markdown -> ${SUMMARY_MD}"
"${PY}" "${SCRIPT_DIR}/write_multistep_summary_md.py" \
  --variant "${VARIANT}" \
  --stamp "${STAMP_SUFFIX}" \
  --baseline-jsonl "${BASELINE_JSONL}" \
  --output "${SUMMARY_MD}" | tee -a "${LOG}"

plog "postprocess complete jsonl=${JSONL_OUT} summary=${SUMMARY_MD}"
