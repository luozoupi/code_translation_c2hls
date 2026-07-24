#!/usr/bin/env bash
# Clean MachSuite flash restart after breakage:
#   1) isolate hard benches (fail open jobs)
#   2) requeue orphan/stale claims
#   3) ensure one watcher + long post walltime
#   4) recover flash compute/GPU
#   5) re-run partial dataflow for flash-done benches (skip bfs_bulk success)
#
# Usage:
#   ./scripts/pc2/restart_machsuite_flash_partial_dataflow.sh
#   ./scripts/pc2/restart_machsuite_flash_partial_dataflow.sh --dry-run
#   ./scripts/pc2/restart_machsuite_flash_partial_dataflow.sh --skip-dataflow
#   ./scripts/pc2/restart_machsuite_flash_partial_dataflow.sh \
#       --campaign-root artifacts/pc2/batch_parallel_machsuite_fd_20260710_machsuite_flash_dataflow
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

CAMPAIGN_ROOT="${C2HLS_ROOT}/artifacts/pc2/batch_parallel_machsuite_fd_20260710_machsuite_flash_dataflow"
DRY_RUN=0
SKIP_DATAFLOW=0
SKIP_FLASH=0
HARD_BENCHES_DEFAULT="machsuite_aes_table,machsuite_viterbi,machsuite_md_grid,machsuite_backprop,machsuite_gemm_blocked"
HARD_BENCHES="${MACHSUITE_HARD_BENCHES:-${HARD_BENCHES_DEFAULT}}"
DATAFLOW_BENCHES_DEFAULT="machsuite_aes_tableless,machsuite_bfs_queue,machsuite_fft_transpose,machsuite_gemm_ncubed,machsuite_md_knn,machsuite_nw,machsuite_sort_merge,machsuite_sort_radix,machsuite_spmv_crs,machsuite_spmv_ellpack,machsuite_stencil2D,machsuite_stencil3D"
# bfs_bulk already succeeded in partial dataflow; omit unless forced.
DATAFLOW_BENCHES="${MACHSUITE_PARTIAL_DATAFLOW_BENCHES:-${DATAFLOW_BENCHES_DEFAULT}}"
POST_WALLTIME="${PC2_MACHSUITE_POST_WALLTIME:-7-00:00:00}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --campaign-root) shift; CAMPAIGN_ROOT="$1"; shift ;;
    --hard-benches) shift; HARD_BENCHES="$1"; shift ;;
    --dataflow-benches) shift; DATAFLOW_BENCHES="$1"; shift ;;
    --post-walltime) shift; POST_WALLTIME="$1"; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --skip-dataflow) SKIP_DATAFLOW=1; shift ;;
    --skip-flash) SKIP_FLASH=1; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

if [[ ! -d "${CAMPAIGN_ROOT}" ]]; then
  echo "ERROR: campaign root missing: ${CAMPAIGN_ROOT}" >&2
  exit 2
fi
CAMPAIGN_ROOT="$(cd "${CAMPAIGN_ROOT}" && pwd)"

export BATCH_PARALLEL_CONFIG="${BATCH_PARALLEL_CONFIG:-${SCRIPT_DIR}/batch_parallel_machsuite_flash_dataflow.json}"
export BATCH_PARALLEL_VARIANT="${BATCH_PARALLEL_VARIANT:-tier_b_aav_n}"
export BATCH_PARALLEL_CAMPAIGN_ROOT="${CAMPAIGN_ROOT}"
export PC2_BATCH_JOB_PREFIX="${PC2_BATCH_JOB_PREFIX:-bpmachfd}"
export PC2_FORCE_WALLTIME="${PC2_BATCH_PARALLEL_WALLTIME:-48:00:00}"
export C2HLS_RUN_COSIM=1
export C2HLS_REFERENCE_COSIM=1
export C2HLS_MAX_REPAIR_ATTEMPT="${C2HLS_MAX_REPAIR_ATTEMPT:-7}"
export C2HLS_STALE_CLAIM_S="${C2HLS_STALE_CLAIM_S:-1800}"
export C2HLS_DATAFLOW_REPAIR_ROUNDS="${C2HLS_DATAFLOW_REPAIR_ROUNDS:-4}"
export C2HLS_DATAFLOW_CONTRACT_ROUNDS="${C2HLS_DATAFLOW_CONTRACT_ROUNDS:-4}"
export C2HLS_POST_FLASH_RESULTS_SUFFIX="${C2HLS_POST_FLASH_RESULTS_SUFFIX:-machsuite_partial_cosim_repairs}"

PY="${C2HLS_PYTHON:-${C2HLS_ROOT}/.venv/bin/python3}"
[[ -x "${PY}" ]] || PY=python3

echo "=== MachSuite clean restart ==="
echo "campaign=${CAMPAIGN_ROOT}"
echo "hard_benches=${HARD_BENCHES}"
echo "dataflow_benches=${DATAFLOW_BENCHES}"
echo "post_walltime=${POST_WALLTIME}"
echo "max_repair_attempt=${C2HLS_MAX_REPAIR_ATTEMPT} stale_claim_s=${C2HLS_STALE_CLAIM_S}"
echo "dry_run=${DRY_RUN} skip_flash=${SKIP_FLASH} skip_dataflow=${SKIP_DATAFLOW}"

# ---------------------------------------------------------------------------
# 0) Kill duplicate MachSuite watchers (keep at most one after recover)
# ---------------------------------------------------------------------------
echo "[1/6] killing duplicate bpmachfd watchers"
for pid in $(pgrep -u "$(id -un)" -f 'batch_parallel_watch_session.sh' || true); do
  if tr '\0' '\n' < "/proc/${pid}/environ" 2>/dev/null | rg -q 'PC2_BATCH_JOB_PREFIX=bpmachfd|machsuite_flash_dataflow'; then
    echo "  kill watcher pid=${pid}"
    if [[ "${DRY_RUN}" -eq 0 ]]; then
      kill "${pid}" 2>/dev/null || true
    fi
  fi
done

# ---------------------------------------------------------------------------
# 1) Isolate hard benches + patch campaign config + requeue orphans/stale
# ---------------------------------------------------------------------------
echo "[2/6] isolate hard benches + requeue stale/orphan claims"
mkdir -p "${CAMPAIGN_ROOT}/flow"
export RESTART_STATE_FILE="${CAMPAIGN_ROOT}/flow/restart_state.env"
if [[ "${DRY_RUN}" -eq 0 ]]; then
  HARD_BENCHES="${HARD_BENCHES}" CAMPAIGN_ROOT="${CAMPAIGN_ROOT}" \
  BATCH_PARALLEL_CONFIG="${BATCH_PARALLEL_CONFIG}" \
  BATCH_PARALLEL_VARIANT="${BATCH_PARALLEL_VARIANT}" \
  C2HLS_ROOT="${C2HLS_ROOT}" \
  C2HLS_STALE_CLAIM_S="${C2HLS_STALE_CLAIM_S}" \
  RESTART_STATE_FILE="${RESTART_STATE_FILE}" \
  "${PY}" - <<'PY'
import json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(os.environ["C2HLS_ROOT"]) / "scripts" / "pc2"))
from batch_parallel_config import load_config
from batch_parallel_queue import BatchParallelQueue

root = Path(os.environ["CAMPAIGN_ROOT"])
hard = [b.strip() for b in os.environ["HARD_BENCHES"].split(",") if b.strip()]
cfg = load_config()
queue = BatchParallelQueue(root / "queue.db")

failed = queue.fail_benches(
    hard,
    error="isolated_hard_bench_for_smooth_restart",
    variant=os.environ.get("BATCH_PARALLEL_VARIANT") or "tier_b_aav_n",
)
print(f"isolated_failed_jobs={len(failed)} benches={hard}")

stale_s = float(os.environ.get("C2HLS_STALE_CLAIM_S") or cfg.stale_claim_s or 1800)
stale = queue.requeue_stale_claimed(max_age_s=stale_s)
orphans = queue.requeue_orphaned_claimed()
cleared = queue.clear_node_slot_assignments()
print(f"stale_requeued={len(stale)} orphan_requeued={len(orphans)} cleared_slots={cleared}")

# Merge new harness knobs into campaign.json stored config.
camp_path = root / "campaign.json"
camp = json.loads(camp_path.read_text())
stored = dict(camp.get("config") or {})
stored["stale_claim_s"] = float(cfg.stale_claim_s)
stored["max_repair_attempt"] = int(cfg.max_repair_attempt)
pilot = dict(stored.get("pilot") or {})
stored["pilot"] = pilot
camp["config"] = stored
active = camp.get("active_variants") or [os.environ.get("BATCH_PARALLEL_VARIANT") or "tier_b_aav_n"]
complete = queue.campaign_complete(list(active))
if complete:
    from datetime import datetime, timezone
    camp["campaign_status"] = "complete"
    camp["completed_at"] = datetime.now(timezone.utc).isoformat()
    print("campaign_complete=1 after isolate (flash done + hard failed)")
else:
    camp["campaign_status"] = "running"
    camp["completed_at"] = None
    print("campaign_complete=0 remaining work still open")
camp_path.write_text(json.dumps(camp, indent=2) + "\n")
print("updated campaign.json")
# Signal to bash
Path(os.environ.get("RESTART_STATE_FILE", root / "flow" / "restart_state.env")).write_text(
    f"CAMPAIGN_COMPLETE={'1' if complete else '0'}\n"
)
PY
  # shellcheck disable=SC1090
  source "${CAMPAIGN_ROOT}/flow/restart_state.env"
else
  echo "  (dry-run) would fail hard benches and requeue claims"
  CAMPAIGN_COMPLETE=0
fi

# ---------------------------------------------------------------------------
# 2) Restart flash GPU+compute via recover (one watcher) — only if work remains
# ---------------------------------------------------------------------------
if [[ "${SKIP_FLASH}" -eq 0 && "${CAMPAIGN_COMPLETE:-0}" != "1" ]]; then
  echo "[3/6] recover flash compute/GPU"
  if [[ "${DRY_RUN}" -eq 0 ]]; then
    # Cancel any leftover bpmachfd jobs first.
    mapfile -t OLD_JOBS < <(squeue -u "$(id -un)" -h -o '%i %j' | awk '/bpmachfd/{print $1}')
    if [[ "${#OLD_JOBS[@]}" -gt 0 ]]; then
      echo "  scancel ${OLD_JOBS[*]}"
      scancel "${OLD_JOBS[@]}" || true
      sleep 3
    fi
    BATCH_PARALLEL_CAMPAIGN_ROOT="${CAMPAIGN_ROOT}" \
      BATCH_PARALLEL_VARIANT="${BATCH_PARALLEL_VARIANT}" \
      BATCH_PARALLEL_CONFIG="${BATCH_PARALLEL_CONFIG}" \
      PC2_BATCH_JOB_PREFIX="${PC2_BATCH_JOB_PREFIX}" \
      PC2_FORCE_WALLTIME="${PC2_FORCE_WALLTIME}" \
      "${SCRIPT_DIR}/recover_batch_parallel_compute.sh"
  else
    echo "  (dry-run) would run recover_batch_parallel_compute.sh"
  fi
elif [[ "${CAMPAIGN_COMPLETE:-0}" == "1" ]]; then
  echo "[3/6] skip flash recover (campaign already complete after isolate)"
else
  echo "[3/6] skip flash recover"
fi

# ---------------------------------------------------------------------------
# 3) Resubmit long-lived post watcher (flash→dataflow handoff)
# ---------------------------------------------------------------------------
echo "[4/6] submit post watcher walltime=${POST_WALLTIME}"
WATCH_LOG="${CAMPAIGN_ROOT}/flow/post_flash_dataflow_watcher.log"
mkdir -p "${CAMPAIGN_ROOT}/flow"
if [[ "${DRY_RUN}" -eq 0 ]]; then
  # Cancel any leftover bpmachfd-post.
  mapfile -t POST_OLD < <(squeue -u "$(id -un)" -h -o '%i %j' | awk '/bpmachfd-post/{print $1}')
  if [[ "${#POST_OLD[@]}" -gt 0 ]]; then
    scancel "${POST_OLD[@]}" || true
  fi
  POST_JOB="$(
    sbatch --parsable \
      --chdir="${C2HLS_ROOT}" \
      --job-name="bpmachfd-post" \
      --output="${CAMPAIGN_ROOT}/flow/post_watcher-%j.out" \
      --error="${CAMPAIGN_ROOT}/flow/post_watcher-%j.err" \
      --account="${PC2_SLURM_ACCOUNT:-hpc-prf-llmfpga}" \
      --partition="${PC2_COMPUTE_PARTITION}" \
      --cpus-per-task=2 \
      --mem=8G \
      --time="${POST_WALLTIME}" \
      --wrap="bash ${SCRIPT_DIR}/wait_machsuite_flash_then_dataflow.sh --campaign-root ${CAMPAIGN_ROOT} >> ${WATCH_LOG} 2>&1"
  )"
  echo "  post_job=${POST_JOB}"
else
  echo "  (dry-run) would sbatch bpmachfd-post"
fi

# ---------------------------------------------------------------------------
# 4) Partial dataflow for remaining 12 (dedicated GPU, 72h)
# ---------------------------------------------------------------------------
if [[ "${SKIP_DATAFLOW}" -eq 0 ]]; then
  echo "[5/6] partial dataflow for ${DATAFLOW_BENCHES}"
  PARTIAL="${CAMPAIGN_ROOT}/partial_dataflow_flash_done_13"
  mkdir -p "${PARTIAL}"
  if [[ "${DRY_RUN}" -eq 0 ]]; then
    DATAFLOW_BENCHES="${DATAFLOW_BENCHES}" CAMPAIGN_ROOT="${CAMPAIGN_ROOT}" PARTIAL="${PARTIAL}" \
    "${PY}" - <<'PY'
import json, os
from datetime import datetime, timezone
from pathlib import Path

cam = Path(os.environ["CAMPAIGN_ROOT"])
partial = Path(os.environ["PARTIAL"])
benches = [b.strip() for b in os.environ["DATAFLOW_BENCHES"].split(",") if b.strip()]
rows = []
for bench in benches:
    cell = next((cam / "variants" / "tier_b_aav_n" / bench).glob("devstral2__*"))
    rows.append({
        "bench": bench,
        "cell_dir": str(cell.resolve()),
        "status": "flash_done",
        "model": cell.name,
        "variant": "tier_b_aav_n",
    })
(partial / "matrix.json").write_text(json.dumps(rows, indent=2) + "\n")
meta = {
    "schema": "machsuite_partial_dataflow_v1",
    "note": "Restart partial dataflow; flash continues on non-hard benches.",
    "parent_campaign": str(cam),
    "created_at": datetime.now(timezone.utc).isoformat(),
    "benches": benches,
    "skipped_success": ["machsuite_bfs_bulk"],
}
(partial / "README.json").write_text(json.dumps(meta, indent=2) + "\n")
print(f"wrote {partial}/matrix.json n={len(rows)}")
PY
    "${PY}" "${SCRIPT_DIR}/export_flash_selected_bundle.py" --pc2 \
      --matrix-root "${PARTIAL}" \
      --out-root "${C2HLS_ROOT}/artifacts/pc2/flash_selected_bundle" \
      --benches "${DATAFLOW_BENCHES}"
    ln -sfn "${C2HLS_ROOT}/artifacts/pc2/flash_selected_bundle/$(basename "${PARTIAL}")" "${PARTIAL}/flash_selected"
    ln -sfn "${PARTIAL}" "${CAMPAIGN_ROOT}/partial_dataflow"

    export C2HLS_POST_FLASH_MATRIX_ROOT="${PARTIAL}"
    export C2HLS_RUN_COSIM=1
    export PC2_FORCE_WALLTIME=72:00:00
    # Drop campaign root so start_session/submit_gpu do not leak it into the
    # dedicated post-flash GPU job (which would write llm_endpoint.json under
    # the campaign while watch/compute expect the session dir).
    env -u BATCH_PARALLEL_CAMPAIGN_ROOT \
      C2HLS_POST_FLASH_MATRIX_ROOT="${PARTIAL}" \
      C2HLS_RUN_COSIM=1 \
      PC2_FORCE_WALLTIME=72:00:00 \
      "${SCRIPT_DIR}/start_post_flash_dataflow.sh" \
      --submit \
      --force \
      --no-borrow-gpu \
      --no-auto-stop-gpu \
      --matrix-root "${PARTIAL}" \
      --benches "${DATAFLOW_BENCHES}" \
      --prompt-policy system_skills \
      --contract-turns "${C2HLS_DATAFLOW_CONTRACT_ROUNDS}"
  else
    echo "  (dry-run) would export+submit partial dataflow"
  fi
else
  echo "[5/6] skip partial dataflow"
fi

echo "[6/6] done"
echo "flash watch:   tail -f ${CAMPAIGN_ROOT}/flow/watch.log"
echo "post watch:    tail -f ${CAMPAIGN_ROOT}/flow/post_flash_dataflow_watcher.log"
echo "squeue:        squeue -u \$USER | rg 'bpmachfd|post_flash_dataflow'"
