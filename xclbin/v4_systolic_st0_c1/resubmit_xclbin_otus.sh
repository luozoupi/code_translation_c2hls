#!/bin/bash
# resubmit_xclbin_otus.sh [--dry-run] : on Otus, submit the U280 xclbin build pending on Fir on 2026-10-22 (Fir job
# 64165394: 32 cpus, 448G, 2 days) with sbatch_build_otus.sh. Skipped if out/kernel0.xclbin exists. TIME=<limit> overrides
# the 2 days; MEM=<mem> the 448G (if Otus normal nodes have less, ask for a large-memory partition with PARTITION=<name>).
set -eo pipefail
HERE=$(dirname "$(readlink -f "$0")")
[ "${CC_CLUSTER:-}" = fir ] && [ "${1:-}" != --dry-run ] && { echo "this is Fir: only --dry-run here"; exit 1; }
[ -f "$HERE/out/kernel0.xclbin" ] && { echo "out/kernel0.xclbin exists, skipped"; exit 0; }
mkdir -p "$HERE/logs"
CMD=(sbatch --parsable --chdir="$HERE" -t "${TIME:-2-00:00:00}" --mem="${MEM:-448G}" ${PARTITION:+-p "$PARTITION"}
     -o "$HERE/logs/build-%j.out" -e "$HERE/logs/build-%j.err" "$HERE/sbatch_build_otus.sh")
if [ "${1:-}" = --dry-run ]; then echo "${CMD[*]}"; exit 0; fi
J=$("${CMD[@]}"); echo "$J" | tee "$HERE/job_id_otus.txt"
