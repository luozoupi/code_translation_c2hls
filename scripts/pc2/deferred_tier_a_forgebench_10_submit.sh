#!/usr/bin/env bash
# Wait for tier_A r2 Slurm jobs, then submit forgebench-only tier_a batch_parallel (10 benches).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

LOG="${C2HLS_ROOT}/artifacts/pc2/sessions/deferred_forgebench_10_submit.log"
mkdir -p "$(dirname "${LOG}")"

WAIT_JOBS=(1931245 1931246)
POLL_S=30
AUTOSTOP_GRACE_S=130

BENCHES="forgebench_attention_op_p1,forgebench_conv_A,forgebench_diff_dims_p1,forgebench_diff_orders_p1,forgebench_gpt_transformer_p1,forgebench_llama_transformer_p2,forgebench_mlp,forgebench_mult_op_p1,forgebench_tiled_attn_p1,forgebench_vec_mtx_p1"

log() {
  echo "[$(date -Iseconds)] $*" | tee -a "${LOG}"
}

log "deferred submit: waiting for jobs ${WAIT_JOBS[*]} to leave queue"
for j in "${WAIT_JOBS[@]}"; do
  while squeue -h -j "${j}" 2>/dev/null | grep -q .; do
    sleep "${POLL_S}"
  done
  log "job ${j} no longer in squeue"
done

log "sleep ${AUTOSTOP_GRACE_S}s for r2 session auto-stop"
sleep "${AUTOSTOP_GRACE_S}"

STAMP="20260701_tier_a_bp_forgebench10"
CONFIG="${SCRIPT_DIR}/batch_parallel_tier_a_forgebench10.json"

log "submitting forgebench-only tier_a batch_parallel: stamp=${STAMP}"
log "benches=${BENCHES} config=${CONFIG}"

export BATCH_PARALLEL_CONFIG="${CONFIG}"
export BATCH_PARALLEL_STAMP="${STAMP}"
export C2HLS_TIER_A_FLASH_BENCHES="${BENCHES}"
export PC2_TIER_A_FLASH_WALLTIME="${PC2_TIER_A_FLASH_WALLTIME:-12:00:00}"

"${C2HLS_ROOT}/scripts/pc2/start_tier_a_batch_parallel.sh" \
  --stamp "${STAMP}" 2>&1 | tee -a "${LOG}"

log "deferred submit complete"
