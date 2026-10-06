#!/usr/bin/env bash
# Post watcher: flash-only campaigns.
#   per bench: flash done → async flash ranked cosim
#   campaign complete → catch-up remaining flash cosim + export flash_selected
# NO dataflow. NO lat_opt. NO RAG/RAG2.
#
# Usage:
#   ./scripts/pc2/wait_hlsfactory_flash_cosim_only.sh --campaign-root DIR [--dry-run]
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
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --campaign-root) shift; CAMPAIGN_ROOT="$1"; shift ;;
    --poll-sec) shift; POLL_SEC="$1"; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "${CAMPAIGN_ROOT}" ]]; then
  echo "ERROR: --campaign-root required" >&2
  exit 2
fi

PY="${C2HLS_PYTHON:-${C2HLS_ROOT}/.venv/bin/python}"
[[ -x "${PY}" ]] || PY=python3

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] wait_hlsfactory_flash_cosim_only start"
echo "campaign=${CAMPAIGN_ROOT}"
echo "lat_opt=${C2HLS_POST_FLASH_LATENCY_OPT:-0} rag2=${C2HLS_RAG2:-0} dry_run=${DRY_RUN}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "[dry-run] would stream flash ranked cosim only (no dataflow)"
  exit 0
fi

submit_ready_flash_ranked_cosim() {
  local ready
  ready="$(
    CAMPAIGN_ROOT="${CAMPAIGN_ROOT}" "${PY}" - <<'PY'
import os
import sqlite3
from pathlib import Path

root = Path(os.environ["CAMPAIGN_ROOT"])
db = root / "queue.db"
if not db.is_file():
    raise SystemExit(0)
already = set()
job_ids = root / "flow" / "flash_ranked_cosim" / "job_ids.txt"
if job_ids.is_file():
    for line in job_ids.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) >= 2:
            already.add(parts[1])
con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
try:
    rows = con.execute(
        "SELECT DISTINCT bench FROM bench_lock "
        "WHERE bench_status IN ('done','failed') ORDER BY bench"
    ).fetchall()
finally:
    con.close()
for (bench,) in rows:
    if not bench or bench in already:
        continue
    cell_ready = False
    for cell in (root / "variants").glob(f"*/{bench}/*"):
        if not cell.is_dir():
            continue
        if (
            (cell / f"{bench}_multistep_results.json").is_file()
            or any(cell.glob("*_final.cpp"))
            or any(cell.glob("*_selected.cpp"))
        ):
            cell_ready = True
            break
    if cell_ready:
        print(bench)
PY
  )"
  if [[ -z "${ready}" ]]; then
    return 0
  fi
  local args=()
  local b
  while IFS= read -r b; do
    [[ -n "${b}" ]] || continue
    args+=(--bench "${b}")
  done <<< "${ready}"
  if [[ "${#args[@]}" -eq 0 ]]; then
    return 0
  fi
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] per-bench flash ranked cosim for: ${ready//$'\n'/ }"
  bash "${SCRIPT_DIR}/submit_campaign_flash_ranked_cosim.sh" \
    --campaign-root "${CAMPAIGN_ROOT}" \
    "${args[@]}"
}

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] streaming: flash ranked cosim only (no dataflow)"
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
  submit_ready_flash_ranked_cosim || true
  case "${status}" in
    complete|completed|failed|aborted) break ;;
  esac
  sleep "${POLL_SEC}"
done

if [[ "${status}" != "complete" && "${status}" != "completed" ]]; then
  echo "campaign ended with status=${status}; still exporting whatever cells exist"
fi
submit_ready_flash_ranked_cosim || true

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] building matrix.json"
"${PY}" - <<PY
import json
from pathlib import Path
from datetime import datetime, timezone

root = Path("${CAMPAIGN_ROOT}")
variant_root = root / "variants"
best = {}
for cell in sorted(variant_root.glob("*/*/*")):
    if not cell.is_dir():
        continue
    bench = cell.parent.name
    if not (cell / f"{bench}_multistep_results.json").is_file() and not any(cell.glob("*_final.cpp")) and not any(cell.glob("*_selected.cpp")):
        if not (cell / "reference_validation.json").is_file() and not (cell / "pipelined").is_dir():
            continue
    score = (
        0 if "claude" in cell.name else (1 if "deepseek" in cell.name else 2),
        0 if "devstral" not in cell.name else 3,
        cell.name,
    )
    prev = best.get(bench)
    if prev is None or score < prev[0]:
        best[bench] = (score, {
            "bench": bench,
            "cell_dir": str(cell.resolve()),
            "status": "unknown",
            "model": cell.name,
            "variant": cell.parent.parent.name,
        })
rows = [best[b][1] for b in sorted(best)]
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

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] catch-up flash ranked cosim for any remaining matrix rows"
bash "${SCRIPT_DIR}/submit_campaign_flash_ranked_cosim.sh" --campaign-root "${CAMPAIGN_ROOT}"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] exporting flash_selected bundle"
"${PY}" "${C2HLS_ROOT}/scripts/pc2/export_flash_selected_bundle.py" --pc2 \
  --matrix-root "${CAMPAIGN_ROOT}" \
  --out-root "${C2HLS_ROOT}/artifacts/pc2/flash_selected_bundle" || true

"${PY}" - <<PY
import json
from pathlib import Path
from datetime import datetime, timezone
p = Path("${CAMPAIGN_ROOT}") / "campaign.json"
doc = json.loads(p.read_text()) if p.is_file() else {}
doc["post_flash_cosim_only"] = {
    "finished_at": datetime.now(timezone.utc).isoformat(),
    "flash_ranked_cosim_async": True,
    "dataflow": False,
}
doc["selection_complete"] = True
p.write_text(json.dumps(doc, indent=2) + "\n")
PY

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] flash-cosim-only path complete (cosim may still be running async)"
echo "campaign=${CAMPAIGN_ROOT}"
