#!/usr/bin/env bash
# Launch Haiku 4.5 skills + noskills top-5 flash→dataflow in parallel.
#
# Usage:
#   ./scripts/pc2/start_hlsfactory_haiku45_flash_dataflow_dual.sh [--dry-run] [--stamp STAMP]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

STAMP="${BATCH_PARALLEL_STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
DRY_RUN=0
BENCH_SET="top5"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stamp) shift; STAMP="$1"; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --bench-set) shift; BENCH_SET="$1"; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

ROOT="${C2HLS_ROOT}/artifacts/pc2/hlsfactory_haiku45_${BENCH_SET}_${STAMP}"
mkdir -p "${ROOT}"
MANIFEST="${ROOT}/dual_manifest.json"
echo "{}" > "${MANIFEST}"

echo "=== HLSFactory Claude Haiku 4.5 dual launch stamp=${STAMP} bench_set=${BENCH_SET} dry_run=${DRY_RUN} ==="

for FLAVOR in skills noskills; do
  ARGS=(--flavor "${FLAVOR}" --stamp "${STAMP}" --bench-set "${BENCH_SET}")
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    ARGS+=(--dry-run)
  fi
  echo "launching flavor=${FLAVOR}"
  bash "${SCRIPT_DIR}/start_hlsfactory_haiku45_flash_dataflow_one.sh" "${ARGS[@]}"
  case "${FLAVOR}" in
    skills) PREFIX="batch_parallel_hlsfactory_haiku45_${BENCH_SET}_skills" ;;
    noskills) PREFIX="batch_parallel_hlsfactory_haiku45_${BENCH_SET}_noskills" ;;
  esac
  CAMPAIGN_ROOT="${C2HLS_ROOT}/artifacts/pc2/${PREFIX}_${STAMP}"
  "${C2HLS_PYTHON:-python3}" - <<PY
import json
from pathlib import Path
p = Path("${MANIFEST}")
doc = json.loads(p.read_text()) if p.is_file() else {}
doc["${FLAVOR}"] = {
    "campaign_root": "${CAMPAIGN_ROOT}",
    "stamp": "${STAMP}",
    "bench_set": "${BENCH_SET}",
}
p.write_text(json.dumps(doc, indent=2) + "\n")
PY
done

echo "dual launch done manifest=${MANIFEST}"
cat "${MANIFEST}"
