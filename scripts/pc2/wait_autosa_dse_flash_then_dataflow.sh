#!/usr/bin/env bash
# Wait for AutoSA DSE flash batch_parallel campaign, then:
#   1) write matrix.json from variant cells
#   2) export flash_selected_bundle
#   3) run post-flash dataflow with cosim + multi-round repairs (no GPU borrow)
#   4) export dataflow_selected_bundle
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/setup_vitis_env.sh"
pc2_setup_vitis_env
cd "${C2HLS_ROOT}"

CAMPAIGN_ROOT=""
POLL_SEC="${POST_FLASH_POLL_SEC:-120}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --campaign-root) shift; CAMPAIGN_ROOT="$1"; shift ;;
    --poll-sec) shift; POLL_SEC="$1"; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "${CAMPAIGN_ROOT}" ]]; then
  echo "ERROR: --campaign-root required" >&2
  exit 2
fi

PY="${C2HLS_PYTHON:-${C2HLS_ROOT}/.venv/bin/python}"
if [[ ! -x "${PY}" ]]; then
  PY=python3
fi

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] waiting for campaign complete: ${CAMPAIGN_ROOT}"
while true; do
  status="$("${PY}" - <<PY
import json
from pathlib import Path
p = Path("${CAMPAIGN_ROOT}") / "campaign.json"
if not p.is_file():
    print("missing")
else:
    print(json.loads(p.read_text()).get("campaign_status", "unknown"))
PY
)"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] campaign_status=${status}"
  case "${status}" in
    complete|completed|failed|aborted) break ;;
  esac
  sleep "${POLL_SEC}"
done

if [[ "${status}" != "complete" && "${status}" != "completed" ]]; then
  echo "campaign ended with status=${status}; still exporting whatever cells exist"
fi

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] building matrix.json"
"${PY}" - <<PY
import json
from pathlib import Path
from datetime import datetime, timezone

root = Path("${CAMPAIGN_ROOT}")
variant_root = root / "variants"
rows = []
for cell in sorted(variant_root.glob("*/*/*")):
    if not cell.is_dir():
        continue
    if not (cell / f"{cell.parent.name}_multistep_results.json").is_file() and not any(cell.glob("*_final.cpp")) and not any(cell.glob("*_selected.cpp")):
        if not (cell / "reference_validation.json").is_file() and not (cell / "pipelined").is_dir():
            continue
    bench = cell.parent.name
    rows.append({
        "bench": bench,
        "cell_dir": str(cell.resolve()),
        "status": "unknown",
        "model": cell.name,
        "variant": cell.parent.parent.name,
    })
matrix = {
    "schema": "batch_parallel_matrix_v1",
    "campaign_root": str(root.resolve()),
    "created_at": datetime.now(timezone.utc).isoformat(),
    "rows": rows,
}
(root / "matrix.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
(root / "reports").mkdir(parents=True, exist_ok=True)
(root / "reports" / "matrix_meta.json").write_text(json.dumps(matrix, indent=2) + "\n", encoding="utf-8")
print(f"wrote matrix.json with {len(rows)} cells")
PY

FLASH_BUNDLE="${C2HLS_ROOT}/artifacts/pc2/flash_selected_bundle/$(basename "${CAMPAIGN_ROOT}")"
DATAFLOW_BUNDLE="${C2HLS_ROOT}/artifacts/pc2/dataflow_selected_bundle/$(basename "${CAMPAIGN_ROOT}")"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] exporting flash_selected -> ${FLASH_BUNDLE}"
"${PY}" "${C2HLS_ROOT}/scripts/pc2/export_flash_selected_bundle.py" --pc2 \
  --matrix-root "${CAMPAIGN_ROOT}" \
  --out-root "${C2HLS_ROOT}/artifacts/pc2/flash_selected_bundle"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] starting post-flash dataflow (cosim off, borrow off)"
export C2HLS_POST_FLASH_MATRIX_ROOT="${CAMPAIGN_ROOT}"
export C2HLS_RUN_COSIM=0
export C2HLS_REFERENCE_COSIM=0
export C2HLS_COSIM_REQUIRED=0
export C2HLS_DATAFLOW_REPAIR_ROUNDS="${C2HLS_DATAFLOW_REPAIR_ROUNDS:-4}"
export C2HLS_DATAFLOW_CONTRACT_ROUNDS="${C2HLS_DATAFLOW_CONTRACT_ROUNDS:-4}"
export C2HLS_POST_FLASH_RESULTS_SUFFIX="autosa_dse_nocosim_repairs"

"${SCRIPT_DIR}/start_post_flash_dataflow.sh" \
  --submit \
  --force \
  --no-borrow-gpu \
  --no-auto-stop-gpu \
  --matrix-root "${CAMPAIGN_ROOT}" \
  --prompt-policy system_skills \
  --contract-turns "${C2HLS_DATAFLOW_CONTRACT_ROUNDS}"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] waiting for dataflow summary under campaign"
while true; do
  if compgen -G "${CAMPAIGN_ROOT}/post_flash_dataflow_summary_*.json" > /dev/null; then
    break
  fi
  sleep "${POLL_SEC}"
done

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] exporting dataflow_selected -> ${DATAFLOW_BUNDLE}"
mkdir -p "${DATAFLOW_BUNDLE}"
"${PY}" "${C2HLS_ROOT}/scripts/pc2/export_post_flash_dataflow_csynth_bundle.py" \
  --matrix-root "${CAMPAIGN_ROOT}" \
  --flash-bundle-root "${FLASH_BUNDLE}" \
  --kernel-bundle "${DATAFLOW_BUNDLE}" \
  --force \
  || true

if [[ -d "${FLASH_BUNDLE}" ]]; then
  ln -sfn "${FLASH_BUNDLE}" "${CAMPAIGN_ROOT}/flash_selected"
fi
ln -sfn "${DATAFLOW_BUNDLE}" "${CAMPAIGN_ROOT}/dataflow_selected"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] done"
echo "flash_selected=${FLASH_BUNDLE}"
echo "dataflow_selected=${DATAFLOW_BUNDLE}"
