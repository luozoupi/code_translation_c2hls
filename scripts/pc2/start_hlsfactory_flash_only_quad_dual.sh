#!/usr/bin/env bash
# Launch next5 flash-only duals for Sonnet-5, Luna, Grok-4.5, Haiku-4.5
# (skills + noskills each). No dataflow / lat_opt / RAG / RAG2.
#
# Usage:
#   ./scripts/pc2/start_hlsfactory_flash_only_quad_dual.sh [--stamp STAMP] [--dry-run]
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

ROOT="${C2HLS_ROOT}/artifacts/pc2/hlsfactory_next5_flash_${STAMP}"
mkdir -p "${ROOT}"
MANIFEST="${ROOT}/quad_dual_manifest.json"
echo "{}" > "${MANIFEST}"

echo "=== HLSFactory next5 FLASH-ONLY quad dual stamp=${STAMP} dry_run=${DRY_RUN} ==="
echo "benches: gemm floyd-warshall bicg doitgen fdtd-2d"

for MODEL in sonnet5 luna grok45 haiku45; do
  for FLAVOR in skills noskills; do
    ARGS=(--model "${MODEL}" --flavor "${FLAVOR}" --stamp "${STAMP}")
    if [[ "${DRY_RUN}" -eq 1 ]]; then
      ARGS+=(--dry-run)
    fi
    echo "launching model=${MODEL} flavor=${FLAVOR}"
    bash "${SCRIPT_DIR}/start_hlsfactory_flash_only_one.sh" "${ARGS[@]}"
    PREFIX="batch_parallel_hlsfactory_${MODEL}_next5_flash_${FLAVOR}"
    CAMPAIGN_ROOT="${C2HLS_ROOT}/artifacts/pc2/${PREFIX}_${STAMP}"
    "${C2HLS_PYTHON:-python3}" - <<PY
import json
from pathlib import Path
p = Path("${MANIFEST}")
doc = json.loads(p.read_text()) if p.is_file() else {}
doc.setdefault("${MODEL}", {})
doc["${MODEL}"]["${FLAVOR}"] = {
    "campaign_root": "${CAMPAIGN_ROOT}",
    "stamp": "${STAMP}",
    "bench_set": "next5_flash",
    "dataflow": False,
    "latency_opt": False,
}
p.write_text(json.dumps(doc, indent=2) + "\n")
PY
  done
done

echo "quad dual launch done manifest=${MANIFEST}"
cat "${MANIFEST}"
echo "STAMP=${STAMP}"
