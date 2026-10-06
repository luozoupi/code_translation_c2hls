#!/usr/bin/env bash
# autosa_mm only: flash + ping-pong/DATAFLOW enforcement. Does not run the other GEMMs.
# Enforcement-only on a frozen flash or compute-rewrite seed:
#   C2HLS_SKIP_FLASH=1 C2HLS_FLASH_SEED_DIR=... \
#     ./scripts/pc2/start_autosa_mm_enforcement.sh --seed-flash "$C2HLS_FLASH_SEED_DIR" \
#     --endpoint-url http://login5:18092/v1
#   DIR needs autosa_mm_flash_opt.cpp + autosa_mm_flash_opt_report.json.
#   Opt-in load_B inside tile DATAFLOW: --load-b-in-df or C2HLS_PP_LOAD_B_IN_DF=1.
#   mmflow DSE 13160/352 copy: artifacts/pc2/seeds/mmflow_dse_13160_352
#   Do not point --seed-flash at mmflow itself (that seeds flash 139484/10).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BATCH_PARALLEL_CONFIG="${BATCH_PARALLEL_CONFIG:-${SCRIPT_DIR}/batch_parallel_autosa_flash_enforcement_mm.json}"
export BATCH_PARALLEL_ARTIFACT_PREFIX="${BATCH_PARALLEL_ARTIFACT_PREFIX:-batch_parallel_autosa_mm_enf_aav_n_gf}"
export PC2_BATCH_JOB_PREFIX="${PC2_BATCH_JOB_PREFIX:-mmenf}"
export AUTOSA_ENFORCEMENT_KERNELS="${AUTOSA_ENFORCEMENT_KERNELS:-mm}"
# Kill hung csynth instead of sitting 4h (AutoSA kernel meta is 14400).
export C2HLS_SYNTH_TIMEOUT="${C2HLS_SYNTH_TIMEOUT:-3600}"
exec "${SCRIPT_DIR}/start_autosa_flash_enforcement.sh" "$@"
