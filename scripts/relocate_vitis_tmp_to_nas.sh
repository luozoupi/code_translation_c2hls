#!/usr/bin/env bash
set -euo pipefail

SOURCE="${C2HLS_TMP_SOURCE:-/mnt/data/luo00466/tmp}"
DESTINATION="${C2HLS_TMP_DESTINATION:-/mnt/data2/luo00466/tmp_1/c2hls_vitis_tmp_20260725}"
LOG_DIR="${C2HLS_RELOCATION_LOG_DIR:-/home/luo00466/code_translation-c2hls-hpca2027/artifacts/storage/relocate_vitis_tmp_20260725}"
LOG_FILE="${LOG_DIR}/relocation.log"
LOCK_FILE="${LOG_DIR}/relocation.lock"
ACTIVE_PATTERN='run_agentic_sweep.py|vitis-run|vitis_hls|csim_design|csynth_design|xsim|run_vllm|run_.*matrix|run_.*c2hls'

mkdir -p "${LOG_DIR}"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  printf 'A relocation process already holds %s\n' "${LOCK_FILE}" >&2
  exit 1
fi

exec > >(tee -a "${LOG_FILE}") 2>&1

printf 'START %s\n' "$(date --iso-8601=seconds)"
printf 'SOURCE %s\n' "${SOURCE}"
printf 'DESTINATION %s\n' "${DESTINATION}"

if [[ ! -d "${SOURCE}" ]]; then
  printf 'Source directory does not exist; nothing to relocate.\n'
  exit 0
fi

active_processes="$(pgrep -af "${ACTIVE_PATTERN}" || true)"
if [[ -n "${active_processes}" ]]; then
  printf 'Refusing relocation while matching experiment processes are active:\n%s\n' \
    "${active_processes}"
  exit 2
fi

mkdir -p "${DESTINATION}"
find "${SOURCE}" -mindepth 1 -maxdepth 1 -printf '%P\n' \
  > "${LOG_DIR}/top_level_entries.before.txt"
du -sb "${SOURCE}" > "${LOG_DIR}/source_size.before.txt"
df -h "${SOURCE}" "${DESTINATION}" > "${LOG_DIR}/filesystem.before.txt"

# --remove-source-files removes a local file only after its destination copy succeeds.
# The dated destination is intentionally separate from older NAS scratch archives.
nice -n 10 ionice -c 2 -n 7 \
  rsync \
    -a \
    --numeric-ids \
    --partial \
    --remove-source-files \
    --human-readable \
    --info=progress2,stats2 \
    -- "${SOURCE}/" "${DESTINATION}/"

find "${SOURCE}" -mindepth 1 -depth -type d -empty -delete
find "${SOURCE}" -mindepth 1 -maxdepth 1 -printf '%P\n' \
  > "${LOG_DIR}/top_level_entries.after.txt"
du -sb "${SOURCE}" > "${LOG_DIR}/source_size.after.txt"
du -sb "${DESTINATION}" > "${LOG_DIR}/destination_size.after.txt"
df -h "${SOURCE}" "${DESTINATION}" > "${LOG_DIR}/filesystem.after.txt"

if find "${SOURCE}" -mindepth 1 -print -quit | grep -q .; then
  printf 'PARTIAL %s: source retains entries; rerun this script to resume.\n' \
    "$(date --iso-8601=seconds)"
  exit 3
fi

touch "${LOG_DIR}/COMPLETED"
printf 'DONE %s\n' "$(date --iso-8601=seconds)"
