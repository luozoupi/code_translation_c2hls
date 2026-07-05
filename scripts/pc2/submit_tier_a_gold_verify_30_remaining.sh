#!/usr/bin/env bash
# Submit Slurm array gold gate (synth + csim) for tier_A_ready remaining 30 benches.
#
# Usage:
#   ./scripts/pc2/submit_tier_a_gold_verify_30_remaining.sh
#   ./scripts/pc2/submit_tier_a_gold_verify_30_remaining.sh --dry-run
#   STAMP=my_stamp ./scripts/pc2/submit_tier_a_gold_verify_30_remaining.sh
#
# After all array tasks finish:
#   python3 scripts/pc2/merge_tier_a_gold_matrix.py artifacts/pc2/tier_a_gold_verify_${STAMP}
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

BENCH_LIST="${SCRIPT_DIR}/tier_a_30_remaining_benches.txt"
STAMP="${TIER_A_GOLD_STAMP:-tier_a_30_remaining_$(date -u +%Y%m%d_%H%M%S)}"
OUT="${C2HLS_ROOT}/artifacts/pc2/tier_a_gold_verify_${STAMP}"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help)
      sed -n '2,12p' "$0"
      exit 0
      ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

mapfile -t BENCHES < "${BENCH_LIST}"
N="${#BENCHES[@]}"
if [[ "${N}" -eq 0 ]]; then
  echo "empty bench list: ${BENCH_LIST}" >&2
  exit 1
fi
LAST=$((N - 1))

echo "bench list: ${BENCH_LIST} (${N} benches)"
echo "stamp:      ${STAMP}"
echo "output:     ${OUT}"
echo "array:      0-${LAST}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "dry-run ok"
  exit 0
fi

mkdir -p "${OUT}"
JOB_ID="$(
  export C2HLS_ROOT TIER_A_GOLD_BENCH_LIST="${BENCH_LIST}" TIER_A_GOLD_STAMP="${STAMP}"
  sbatch --parsable \
    --array="0-${LAST}" \
    --chdir="${C2HLS_ROOT}" \
    --export=ALL,C2HLS_ROOT,TIER_A_GOLD_BENCH_LIST,TIER_A_GOLD_STAMP \
    "${SCRIPT_DIR}/run_tier_a_gold_verify_array.sbatch.sh"
)"
JOB_ID="${JOB_ID%%;*}"

cat > "${OUT}/submit.json" <<EOF
{
  "job_id": "${JOB_ID}",
  "stamp": "${STAMP}",
  "bench_list": "${BENCH_LIST}",
  "bench_count": ${N},
  "array": "0-${LAST}",
  "output_dir": "${OUT}"
}
EOF

echo "submitted array job ${JOB_ID}"
echo "monitor: squeue -j ${JOB_ID}"
echo "merge when done:"
echo "  python3 scripts/pc2/merge_tier_a_gold_matrix.py ${OUT}"
