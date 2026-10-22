#!/bin/bash
# resubmit_lud_otus.sh [--dry-run] : on Otus, submit the LUD csynth pending on Fir on 2026-10-22 (Fir job 65903242):
# dse/n32_st5/out, csynth, 192G, 2 days, with job_vitis_otus.sh. Skipped if its csynth report already exists.
set -eo pipefail
HERE=$(dirname "$(readlink -f "$0")")        # study/lud
D=$HERE/dse/n32_st5/out
[ "${CC_CLUSTER:-}" = fir ] && [ "${1:-}" != --dry-run ] && { echo "this is Fir: only --dry-run here"; exit 1; }
if ls "$D"/hls_prj_csynth/solution1/syn/report/csynth.xml >/dev/null 2>&1; then echo "lud n32_st5: csynth report exists, skipped"; exit 0; fi
CMD=(sbatch --parsable -J lud_csynth -t 2-00:00:00 --mem=192G -o "$D/job_csynth-%j.out"
     --export=ALL,STUDY_DIR="$(dirname "$HERE")" "$HERE/job_vitis_otus.sh" "$D" csynth)
if [ "${1:-}" = --dry-run ]; then echo "${CMD[*]}"; exit 0; fi
J=$("${CMD[@]}"); echo "lud_csynth n32_st5 $J" | tee -a "$HERE/otus_lud_submitted.txt"
