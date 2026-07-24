#!/usr/bin/env bash
# Launch Devstral-2 RAG2 flavors in parallel: rag2_skills + rag2_ns (2x GPU).
#
# Usage:
#   ./scripts/pc2/start_chathls_devstral_rag2_parallel.sh [--dry-run]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

STAMP_BASE="$(date -u +%Y%m%d_%H%M%S)"
SEQ_ROOT="${C2HLS_ROOT}/artifacts/pc2/devstral_rag2_parallel_${STAMP_BASE}"
mkdir -p "${SEQ_ROOT}"
STATE_JSON="${SEQ_ROOT}/parallel_state.json"

EXTRA=()
if [[ "${DRY_RUN}" -eq 1 ]]; then
  EXTRA+=(--dry-run)
fi

echo "=== Devstral-2 RAG2 parallel: rag2_skills + rag2_ns ==="
echo "seq_root=${SEQ_ROOT}"
echo "dry_run=${DRY_RUN}"

"${C2HLS_PYTHON:-python3}" - "${SEQ_ROOT}" "${DRY_RUN}" "${STAMP_BASE}" <<'PY'
import json, sys, time
from pathlib import Path
root, dry, stamp = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
doc = {
    "seq_root": str(root),
    "method": "rag2",
    "model": "devstral-2",
    "mode": "parallel",
    "dry_run": bool(int(dry)),
    "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "flavors": ["rag2_skills", "rag2_ns"],
    "campaigns": {},
    "status": "starting",
}
(root / "parallel_state.json").write_text(json.dumps(doc, indent=2) + "\n")
PY

PIDS=()
for flavor in rag2_skills rag2_ns; do
  stamp="${STAMP_BASE}_${flavor}"
  log="${SEQ_ROOT}/${flavor}.log"
  echo "launching flavor=${flavor} stamp=${stamp} log=${log}"
  (
    set +e
    "${SCRIPT_DIR}/start_chathls_devstral_rag2_one.sh" \
      --flavor "${flavor}" \
      --stamp "${stamp}" \
      "${EXTRA[@]}"
    ec=$?
    echo "flavor=${flavor} exit=${ec}" >> "${log}"
    exit "${ec}"
  ) >"${log}" 2>&1 &
  PIDS+=("$!")
  "${C2HLS_PYTHON:-python3}" - "${STATE_JSON}" "${flavor}" "${stamp}" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
doc = json.loads(p.read_text())
flavor, stamp = sys.argv[2], sys.argv[3]
prefix = (
    "batch_parallel_chathls_fd_rag2"
    if flavor == "rag2_skills"
    else "batch_parallel_chathls_fd_rag2_ns"
)
doc["campaigns"][flavor] = {
    "stamp": stamp,
    "campaign_root": f"/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/{prefix}_{stamp}",
    "status": "launching",
}
p.write_text(json.dumps(doc, indent=2) + "\n")
PY
done

echo "waiting for both starters to finish submit (pids=${PIDS[*]}) ..."
FAIL=0
for i in "${!PIDS[@]}"; do
  pid="${PIDS[$i]}"
  if ! wait "${pid}"; then
    FAIL=1
    echo "WARNING: starter pid=${pid} failed" >&2
  fi
done

"${C2HLS_PYTHON:-python3}" - "${STATE_JSON}" "${FAIL}" <<'PY'
import json, sys, time
from pathlib import Path
p = Path(sys.argv[1])
doc = json.loads(p.read_text())
fail = int(sys.argv[2])
doc["status"] = "failed_submit" if fail else "submitted"
doc["submit_finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
for flavor, meta in doc.get("campaigns", {}).items():
    root = Path(meta["campaign_root"])
    cj = root / "campaign.json"
    if cj.is_file():
        cdoc = json.loads(cj.read_text())
        meta["gpu_job_id"] = cdoc.get("gpu_job_id")
        meta["post_watcher_job_id"] = cdoc.get("post_watcher_job_id")
        meta["status"] = cdoc.get("campaign_status", "running")
p.write_text(json.dumps(doc, indent=2) + "\n")
PY

echo "=== Devstral-2 RAG2 parallel submit done ==="
echo "seq_root=${SEQ_ROOT}"
echo "state=${STATE_JSON}"
cat "${STATE_JSON}"
if [[ "${FAIL}" -ne 0 ]]; then
  exit 1
fi
