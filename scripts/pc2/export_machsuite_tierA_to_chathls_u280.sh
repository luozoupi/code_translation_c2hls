#!/usr/bin/env bash
# Export machsuite + forgebench + hp_fft + spector (68) into ChatHLS benchmark_optimization.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

CHATHLS_ROOT="${CHATHLS_ROOT:-/scratch/hpc-prf-llmfpga/asa582/projects/test-chathls/ChatHLS-ACL-26}"
OUT_ROOT="${CHATHLS_OUT_ROOT:-${CHATHLS_ROOT}/benchmark/benchmark_optimization}"
BENCH_LIST="${CHATHLS_BENCH_LIST:-${CHATHLS_ROOT}/scripts/pc2/chathls_u280_machsuite_tierA_68_benches.txt}"
MACHSUITE_ROOT="${C2HLS_ROOT}/benchmarks"
TIER_A_READY="${C2HLS_ROOT}/related_work/benchmarks/HLSFactory_benchmarks/tier_A_ready"

if [[ ! -f "${BENCH_LIST}" ]]; then
  echo "missing bench list: ${BENCH_LIST}" >&2
  exit 2
fi

echo "exporting 68 benches from:"
echo "  machsuite: ${MACHSUITE_ROOT}"
echo "  tier_A:    ${TIER_A_READY}"
echo "  list:      ${BENCH_LIST}"
echo "  out:       ${OUT_ROOT}"

"${C2HLS_PYTHON:-python3}" "${SCRIPT_DIR}/export_c2hls_bench_to_chathls.py" \
  --out-root "${OUT_ROOT}" \
  --benchmarks-root "${MACHSUITE_ROOT}" \
  --benchmarks-root "${TIER_A_READY}" \
  --bench-list "${BENCH_LIST}" \
  --strict

echo "export complete -> ${OUT_ROOT}"
