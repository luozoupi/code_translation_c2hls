#!/usr/bin/env bash
# HLSFactory post watcher for lat-opt / RAG2 tests:
#   per bench: flash(+lat_opt) done → rank → async ranked flash cosim
#   per bench: flash ranked cosim PASS → start that bench's dataflow immediately
#              (does NOT wait for campaign-complete)
#   campaign complete → catch-up remaining DF + final export
#
# Usage:
#   ./scripts/pc2/wait_hlsfactory_flash_lat_dataflow.sh --campaign-root DIR [--dry-run]
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

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] wait_hlsfactory_flash_lat_dataflow start"
echo "campaign=${CAMPAIGN_ROOT}"
echo "lat_opt=${C2HLS_POST_FLASH_LATENCY_OPT:-0} chain_flash=${C2HLS_LATENCY_OPT_CHAIN_FLASH:-0} chain_df=${C2HLS_LATENCY_OPT_CHAIN_DATAFLOW:-0}"
echo "rag2=${C2HLS_RAG2:-0} dry_run=${DRY_RUN}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "[dry-run] would stream flash ranked cosim + DF-on-cosim-pass, then catch-up at campaign complete"
  exit 0
fi

# Ranked flash cosim launches as soon as each bench finishes flash(+lat_opt),
# not after the whole campaign completes. Idempotent via job_ids.txt.
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
    # Cell must have a selectable kernel (done can briefly precede artifacts).
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

# Start dataflow for benches whose flash ranked cosim already PASSED.
# Does NOT wait for campaign-complete. Idempotent via parallel_dataflow/jobs.jsonl.
submit_ready_dataflow_after_flash_cosim_pass() {
  local ready
  ready="$(
    CAMPAIGN_ROOT="${CAMPAIGN_ROOT}" "${PY}" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["CAMPAIGN_ROOT"])
already = set()
jobs = root / "flow" / "parallel_dataflow" / "jobs.jsonl"
if jobs.is_file():
    for line in jobs.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        b = row.get("bench")
        if b:
            already.add(b)

ready = []
for cell in sorted((root / "variants").glob("*/hlsfactory_*/*")):
    if not cell.is_dir() or "failed" in cell.name:
        continue
    bench = cell.parent.name
    if bench in already or bench in ready:
        continue
    # Prefer explicit flash ranked-cosim pass artifact.
    passed = False
    for p in cell.glob("*_flash_cosim_opt_result.json"):
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if doc.get("passed") is True or doc.get("success") is True:
            passed = True
            break
        status = str(doc.get("status") or "").lower()
        if status in {"pass", "passed", "ok", "success"}:
            passed = True
            break
    if passed:
        ready.append(bench)

for b in ready:
    print(b)
PY
  )"
  if [[ -z "${ready}" ]]; then
    return 0
  fi
  local csv
  csv="$(echo "${ready}" | paste -sd, -)"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] per-bench DATAFLOW after flash cosim PASS: ${csv}"
  local df_args=(
    --campaign-root "${CAMPAIGN_ROOT}"
    --benches "${csv}"
    --job-prefix "${PC2_BATCH_JOB_PREFIX:-bphfpdf}"
    --no-cancel-post
    --append
    --skip-export
  )
  if [[ "${C2HLS_DATAFLOW_EXCLUSIVE:-0}" == "1" ]]; then
    df_args+=(--exclusive)
  else
    df_args+=(--no-exclusive)
  fi
  local backend="${C2HLS_PROXY_BACKEND:-}"
  # openai/xai/grok: always shared campaign proxy (no DeepSeek/Anthropic per-bench).
  if [[ "${backend}" == "openai" || "${backend}" == "xai" || "${backend}" == "grok" ]]; then
    df_args+=(--shared-proxy --proxy-backend "${backend}")
  elif [[ "${C2HLS_PER_BENCH_PROXY:-0}" == "1" ]]; then
    df_args+=(--per-bench-proxy)
    [[ -n "${backend}" ]] && df_args+=(--proxy-backend "${backend}")
  else
    df_args+=(--shared-proxy)
    [[ -n "${backend}" ]] && df_args+=(--proxy-backend "${backend}")
  fi
  bash "${SCRIPT_DIR}/start_hlsfactory_parallel_dataflow.sh" "${df_args[@]}"
}

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] streaming: flash ranked cosim + DF-on-cosim-pass (no campaign-complete gate for DF)"
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
  submit_ready_dataflow_after_flash_cosim_pass || true
  case "${status}" in
    complete|completed|failed|aborted) break ;;
  esac
  sleep "${POLL_SEC}"
done

if [[ "${status}" != "complete" && "${status}" != "completed" ]]; then
  echo "campaign ended with status=${status}; still exporting whatever cells exist"
fi
# Final catch-up in case the last bench finished between poll ticks.
submit_ready_flash_ranked_cosim || true
submit_ready_dataflow_after_flash_cosim_pass || true

MIN_EXPORTABLE="${POST_FLASH_MIN_EXPORTABLE:-1}"
EXPORT_WAIT_SEC="${POST_FLASH_EXPORT_WAIT_SEC:-7200}"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] gating on exportable flash kernels (min=${MIN_EXPORTABLE})"
if ! "${PY}" "${SCRIPT_DIR}/post_flash_export_gate.py" \
  --campaign-root "${CAMPAIGN_ROOT}" \
  --min-exportable "${MIN_EXPORTABLE}" \
  --poll-sec "${POLL_SEC}" \
  --max-wait-sec "${EXPORT_WAIT_SEC}"; then
  echo "ERROR: export gate failed; not starting post-flash dataflow catch-up" >&2
  exit 3
fi

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

FLASH_BUNDLE="${C2HLS_ROOT}/artifacts/pc2/flash_selected_bundle/$(basename "${CAMPAIGN_ROOT}")"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] catch-up flash ranked cosim for any remaining matrix rows (skip already submitted)"
bash "${SCRIPT_DIR}/submit_campaign_flash_ranked_cosim.sh" --campaign-root "${CAMPAIGN_ROOT}"

# One more DF stream pass for any late cosim passes.
submit_ready_dataflow_after_flash_cosim_pass || true

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] exporting flash_selected -> ${FLASH_BUNDLE}"
"${PY}" "${C2HLS_ROOT}/scripts/pc2/export_flash_selected_bundle.py" --pc2 \
  --matrix-root "${CAMPAIGN_ROOT}" \
  --out-root "${C2HLS_ROOT}/artifacts/pc2/flash_selected_bundle"

# Catch-up: any flash-finished benches not yet DF'd (e.g. cosim failed/skipped).
export C2HLS_POST_FLASH_MATRIX_ROOT="${CAMPAIGN_ROOT}"
export C2HLS_RUN_COSIM=0
export C2HLS_REFERENCE_COSIM=0
export C2HLS_COSIM_REQUIRED=0
export C2HLS_COSIM_XELAB_MT_OFF="${C2HLS_COSIM_XELAB_MT_OFF:-1}"
export C2HLS_DATAFLOW_REPAIR_ROUNDS="${C2HLS_DATAFLOW_REPAIR_ROUNDS:-4}"
export C2HLS_DATAFLOW_CONTRACT_ROUNDS="${C2HLS_DATAFLOW_CONTRACT_ROUNDS:-4}"
export C2HLS_POST_FLASH_RESULTS_SUFFIX="${C2HLS_POST_FLASH_RESULTS_SUFFIX:-hlsfactory_cosim_repairs}"
export C2HLS_DATAFLOW_EXCLUSIVE="${C2HLS_DATAFLOW_EXCLUSIVE:-0}"
export C2HLS_DATAFLOW_WORKER_CPUS="${C2HLS_DATAFLOW_WORKER_CPUS:-16}"
export C2HLS_DATAFLOW_WORKER_MEM_GB="${C2HLS_DATAFLOW_WORKER_MEM_GB:-64}"
export C2HLS_RANKED_COSIM_AFTER_DATAFLOW=1
export C2HLS_POST_FLASH_PROMPT_POLICY="${C2HLS_POST_FLASH_PROMPT_POLICY:-system_skills}"

ep_url="${BATCH_PARALLEL_EXTERNAL_ENDPOINT_URL:-}"
ep_model="${BATCH_PARALLEL_EXTERNAL_MODEL:-${C2HLS_MODEL:-deepseek-v4-flash}}"
if [[ -z "${ep_url}" && -f "${CAMPAIGN_ROOT}/llm_endpoint.json" ]]; then
  ep_url="$("${PY}" -c "import json;print(json.load(open('${CAMPAIGN_ROOT}/llm_endpoint.json')).get('url',''))")"
  ep_model="$("${PY}" -c "import json;d=json.load(open('${CAMPAIGN_ROOT}/llm_endpoint.json'));print(d.get('model') or '${ep_model}')")"
fi
if [[ -n "${ep_url}" ]]; then
  export OPENAI_BASE_URL="${ep_url}"
  export CHATHLS_API_BASE="${OPENAI_BASE_URL}"
  export C2HLS_MODEL="${ep_model}"
  export OPENAI_API_KEY="${OPENAI_API_KEY:-${CHATHLS_API_KEY:-EMPTY}}"
  export CHATHLS_API_KEY="${CHATHLS_API_KEY:-${OPENAI_API_KEY}}"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] external_llm dataflow via ${OPENAI_BASE_URL} model=${C2HLS_MODEL}"
  if ! curl -sf --max-time 10 "${OPENAI_BASE_URL}/models" >/dev/null; then
    echo "ERROR: external endpoint not reachable: ${OPENAI_BASE_URL}" >&2
    exit 4
  fi
fi

PREFIX="${PC2_BATCH_JOB_PREFIX:-bphfpdf}"
DF_ARGS=(
  --campaign-root "${CAMPAIGN_ROOT}"
  --remaining
  --job-prefix "${PREFIX}"
  --no-cancel-post
  --append
)
if [[ "${C2HLS_DATAFLOW_EXCLUSIVE:-0}" == "1" ]]; then
  DF_ARGS+=(--exclusive)
else
  DF_ARGS+=(--no-exclusive)
fi
backend="${C2HLS_PROXY_BACKEND:-}"
if [[ "${backend}" == "openai" || "${backend}" == "xai" || "${backend}" == "grok" ]]; then
  DF_ARGS+=(--shared-proxy --proxy-backend "${backend}")
elif [[ "${C2HLS_PER_BENCH_PROXY:-0}" == "1" ]]; then
  DF_ARGS+=(--per-bench-proxy)
  [[ -n "${backend}" ]] && DF_ARGS+=(--proxy-backend "${backend}")
else
  DF_ARGS+=(--shared-proxy)
  [[ -n "${backend}" ]] && DF_ARGS+=(--proxy-backend "${backend}")
fi
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] catch-up parallel dataflow for any remaining flash benches (+ final export)"
# --remaining may exit 3 if empty; treat as ok when streaming already covered all.
set +e
bash "${SCRIPT_DIR}/start_hlsfactory_parallel_dataflow.sh" "${DF_ARGS[@]}"
df_rc=$?
set -e
if [[ "${df_rc}" -ne 0 ]]; then
  # If all benches already submitted via streaming, rebuild export only.
  if [[ -f "${CAMPAIGN_ROOT}/flow/parallel_dataflow/jobs.jsonl" ]]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] remaining DF submit returned ${df_rc}; ensuring final export"
    BENCH_CSV="$(
      JOB_LIST="${CAMPAIGN_ROOT}/flow/parallel_dataflow/jobs.jsonl" "${PY}" - <<'PY'
import json
import os
from pathlib import Path
bs = []
p = Path(os.environ["JOB_LIST"])
for line in p.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line:
        continue
    try:
        row = json.loads(line)
    except Exception:
        continue
    if isinstance(row, dict):
        b = row.get("bench")
        if b and b not in bs:
            bs.append(b)
print(",".join(bs))
PY
    )"
    if [[ -n "${BENCH_CSV}" ]]; then
      bash "${SCRIPT_DIR}/start_hlsfactory_parallel_dataflow.sh" \
        --campaign-root "${CAMPAIGN_ROOT}" \
        --benches "${BENCH_CSV}" \
        --job-prefix "${PREFIX}" \
        --no-cancel-post \
        --append \
        --shared-proxy \
        --no-exclusive \
        ${backend:+--proxy-backend ${backend}} \
        || true
    fi
  else
    echo "ERROR: parallel dataflow catch-up failed rc=${df_rc}" >&2
    exit 5
  fi
fi

LAUNCH_JSON="${CAMPAIGN_ROOT}/flow/parallel_dataflow/launch.json"
EXPORT_JOB="$("${PY}" -c "import json;from pathlib import Path;p=Path('${LAUNCH_JSON}');
print(json.load(open(p)).get('export_job_id','') if p.is_file() else '')")"
if [[ -z "${EXPORT_JOB}" ]]; then
  echo "WARNING: no export_job_id recorded (streaming-only or empty remaining)" >&2
else
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] waiting for parallel export job ${EXPORT_JOB} (selection; not cosim)"
  while true; do
    st="$(squeue -j "${EXPORT_JOB}" -h -o '%T' 2>/dev/null || true)"
    if [[ -z "${st}" ]]; then
      break
    fi
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] export_job=${EXPORT_JOB} state=${st}"
    sleep "${POLL_SEC}"
  done
fi

"${PY}" - <<PY
import json
from pathlib import Path
from datetime import datetime, timezone
p = Path("${CAMPAIGN_ROOT}") / "campaign.json"
doc = json.loads(p.read_text()) if p.is_file() else {}
doc["post_flash_lat_dataflow"] = {
    "finished_at": datetime.now(timezone.utc).isoformat(),
    "export_job_id": "${EXPORT_JOB}",
    "flash_ranked_cosim_async": True,
    "dataflow_on_flash_cosim_pass": True,
    "dataflow_run_cosim": 0,
    "ranked_cosim_after_dataflow": True,
}
doc["selection_complete"] = True
p.write_text(json.dumps(doc, indent=2) + "\n")
PY

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] selection path complete (cosim may still be running async)"
echo "campaign=${CAMPAIGN_ROOT}"
