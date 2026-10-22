#!/bin/bash
# resubmit_soda_otus.sh [--dry-run] : on Otus, submit the 85 SODA cosims listed in otus_soda_resubmit.txt
# (pending on Fir when the study moved, 2026-10-22) with job_soda_otus.sh. Skips a design whose cosim/ already
# has a cosim report. Appends "<job name> <design> <job id> <time>" to otus_soda_submitted.txt.
set -eo pipefail
cd "$(dirname "$(readlink -f "$0")")"
DRY=0; [ "${1:-}" = --dry-run ] && DRY=1
[ "${CC_CLUSTER:-}" = fir ] && [ "${1:-}" != --dry-run ] && { echo "this is Fir: only --dry-run here"; exit 1; }
grep -v '^#' otus_soda_resubmit.txt | while read -r NAME DES TIME; do
  D=$(readlink -f "$DES")
  if ls "$D"/cosim/*cosim.rpt >/dev/null 2>&1; then echo "skip (done): $DES"; continue; fi
  [ -f "$D/kernel.cpp" ] && [ -f "$D/config.json" ] || { echo "skip (no kernel/config): $DES"; continue; }
  CMD=(sbatch --parsable -J "$NAME" -t "$TIME" -o "$D/job_cosim-%j.out" job_soda_otus.sh "$D" cosim)
  if [ $DRY = 1 ]; then echo "${CMD[*]}"; continue; fi
  J=$("${CMD[@]}")
  echo "$NAME $DES $J $TIME" | tee -a otus_soda_submitted.txt
done
