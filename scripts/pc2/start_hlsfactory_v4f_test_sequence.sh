#!/usr/bin/env bash
# Sequential HLSFactory DeepSeek-v4-flash tests:
#   1) lat_opt only
#   2) rag2 + lat_opt
#   3) rag2 only
# Within each test: skills / noskills / bare concurrent.
# Next test gated on prior *selection* (not cosim).
#
# Usage:
#   ./scripts/pc2/start_hlsfactory_v4f_test_sequence.sh [--dry-run] [--stamp STAMP]
#   ./scripts/pc2/start_hlsfactory_v4f_test_sequence.sh --only lat_opt|rag2_lat|rag2
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

STAMP_BASE="${BATCH_PARALLEL_STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
DRY_RUN=0
ONLY=""
MIN_RANKED="${C2HLS_TEST_MIN_RANKED:-1}"
POLL_SEC="${C2HLS_TEST_SELECTION_POLL_SEC:-180}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stamp) shift; STAMP_BASE="$1"; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --only) shift; ONLY="$1"; shift ;;
    --min-ranked) shift; MIN_RANKED="$1"; shift ;;
    --poll-sec) shift; POLL_SEC="$1"; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

case "${ONLY}" in
  ""|lat_opt|rag2_lat|rag2) ;;
  *)
    echo "ERROR: --only must be lat_opt|rag2_lat|rag2" >&2
    exit 2
    ;;
esac

SEQ_ROOT="${C2HLS_ROOT}/artifacts/pc2/hlsfactory_ds_v4f_tests_${STAMP_BASE}"
mkdir -p "${SEQ_ROOT}"
MANIFEST="${SEQ_ROOT}/sequence_manifest.json"

declare -A PORTS=(
  [skills]=18092
  [noskills]=18093
  [bare]=18094
)

TESTS=(lat_opt rag2_lat rag2)
if [[ -n "${ONLY}" ]]; then
  TESTS=("${ONLY}")
fi

PY="${C2HLS_PYTHON:-${C2HLS_ROOT}/.venv/bin/python}"
[[ -x "${PY}" ]] || PY=python3

echo "{}" > "${MANIFEST}"
"${PY}" - <<PY
import json
from pathlib import Path
from datetime import datetime, timezone
p = Path("${MANIFEST}")
doc = {
    "schema": "hlsfactory_v4f_test_sequence_v1",
    "stamp_base": "${STAMP_BASE}",
    "seq_root": "${SEQ_ROOT}",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "dry_run": bool(int("${DRY_RUN}")),
    "tests": {},
}
p.write_text(json.dumps(doc, indent=2) + "\n")
PY

echo "=== HLSFactory v4f test sequence stamp_base=${STAMP_BASE} dry_run=${DRY_RUN} ==="
echo "seq_root=${SEQ_ROOT}"

# Start / reuse flavor proxies once for the whole sequence.
declare -A ENDPOINTS=()
for FLAVOR in skills noskills bare; do
  PORT="${PORTS[${FLAVOR}]}"
  PROXY_DIR="${SEQ_ROOT}/proxy_${FLAVOR}"
  mkdir -p "${PROXY_DIR}"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    ENDPOINTS["${FLAVOR}"]="http://127.0.0.1:${PORT}/v1"
    echo "[dry-run] flavor=${FLAVOR} proxy :${PORT}"
  else
    if [[ -f "${PROXY_DIR}/llm_endpoint.json" ]] && curl -sf --max-time 5 "$(
      "${PY}" -c "import json;print(json.load(open('${PROXY_DIR}/llm_endpoint.json'))['url'])"
    )/models" >/dev/null 2>&1; then
      ENDPOINTS["${FLAVOR}"]="$(
        "${PY}" -c "import json;print(json.load(open('${PROXY_DIR}/llm_endpoint.json'))['url'])"
      )"
      echo "reusing proxy flavor=${FLAVOR} url=${ENDPOINTS[${FLAVOR}]}"
    else
      echo "starting DeepSeek queue proxy flavor=${FLAVOR} port=${PORT} workers=1"
      DEEPSEEK_PROXY_MODEL=deepseek-v4-flash \
        C2HLS_MODEL=deepseek-v4-flash \
        CHATHLS_DEEPSEEK_PROXY_PORT="${PORT}" \
        CHATHLS_DEEPSEEK_QUEUE_WORKERS=1 \
        "${SCRIPT_DIR}/c2hls_deepseek_proxy.sh" "${PROXY_DIR}"
      ENDPOINTS["${FLAVOR}"]="$(
        "${PY}" -c "import json;print(json.load(open('${PROXY_DIR}/llm_endpoint.json'))['url'])"
      )"
      echo "proxy ready flavor=${FLAVOR} url=${ENDPOINTS[${FLAVOR}]}"
    fi
  fi
done

launch_test() {
  local TEST="$1"
  local STAMP="${STAMP_BASE}_${TEST}"
  local ARM_MANIFEST="${SEQ_ROOT}/${TEST}_manifest.json"
  echo "{}" > "${ARM_MANIFEST}"

  echo
  echo "=== launching test=${TEST} stamp=${STAMP} ==="
  for FLAVOR in skills noskills bare; do
    local EP="${ENDPOINTS[${FLAVOR}]}"
    local ONE_ARGS=(--flavor "${FLAVOR}" --stamp "${STAMP}" --test "${TEST}" --endpoint-url "${EP}")
    if [[ "${DRY_RUN}" -eq 1 ]]; then
      ONE_ARGS+=(--dry-run)
    fi
    echo "launch flavor=${FLAVOR} test=${TEST}"
    "${SCRIPT_DIR}/start_hlsfactory_deepseek_flash_dataflow_one.sh" "${ONE_ARGS[@]}"

    local PREFIX
    case "${FLAVOR}" in
      skills) PREFIX="batch_parallel_hlsfactory_ds_v4f_skills_${TEST}" ;;
      noskills) PREFIX="batch_parallel_hlsfactory_ds_v4f_noskills_${TEST}" ;;
      bare) PREFIX="batch_parallel_hlsfactory_ds_v4f_bare_${TEST}" ;;
    esac
    local CAMPAIGN_ROOT="${C2HLS_ROOT}/artifacts/pc2/${PREFIX}_${STAMP}"

    "${PY}" - <<PY
import json
from pathlib import Path
arm = Path("${ARM_MANIFEST}")
doc = json.loads(arm.read_text()) if arm.is_file() else {}
doc["test"] = "${TEST}"
doc["stamp"] = "${STAMP}"
doc.setdefault("flavors", {})["${FLAVOR}"] = {
    "port": int("${PORTS[${FLAVOR}]}"),
    "endpoint_url": "${EP}",
    "campaign_root": "${CAMPAIGN_ROOT}",
    "proxy_dir": "${SEQ_ROOT}/proxy_${FLAVOR}",
}
arm.write_text(json.dumps(doc, indent=2) + "\n")

seq = Path("${MANIFEST}")
sdoc = json.loads(seq.read_text())
sdoc.setdefault("tests", {})["${TEST}"] = {
    "stamp": "${STAMP}",
    "arm_manifest": str(arm),
    "flavors": doc["flavors"],
}
seq.write_text(json.dumps(sdoc, indent=2) + "\n")
PY
  done

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "[dry-run] skip selection wait for test=${TEST}"
    return 0
  fi

  echo "waiting for selection done test=${TEST} (min_ranked=${MIN_RANKED}; NOT waiting on cosim)"
  "${PY}" "${SCRIPT_DIR}/wait_hlsfactory_test_selection_done.py" \
    --manifest "${ARM_MANIFEST}" \
    --min-ranked "${MIN_RANKED}" \
    --poll-sec "${POLL_SEC}"
  echo "selection done for test=${TEST}; advancing"
}

for TEST in "${TESTS[@]}"; do
  launch_test "${TEST}"
done

echo
echo "=== sequence complete (cosim may still trail async) ==="
echo "manifest=${MANIFEST}"
cat "${MANIFEST}"
