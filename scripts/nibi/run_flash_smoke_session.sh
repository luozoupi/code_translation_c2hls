#!/usr/bin/env bash
# Nibi flash smoke: one GPU node runs vLLM (GLM) + flash batch (Vitis csynth/csim).
#
# Usage:
#   ./scripts/nibi/run_flash_smoke_session.sh --dry-run
#   ./scripts/nibi/run_flash_smoke_session.sh --submit
#   ./scripts/nibi/run_flash_smoke_session.sh --submit --benches hlsfactory_gemm
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

PY="${C2HLS_PYTHON:-python3}"
STAMP="${C2HLS_NIBI_FLASH_SMOKE_STAMP:-$(date +%Y%m%d_%H%M%S)}"
BENCHES="${C2HLS_NIBI_FLASH_SMOKE_BENCHES:-hlsfactory_gemm}"
DRY_RUN=0
SUBMIT=0

usage() {
  cat <<EOF
Usage: $0 [--dry-run | --submit] [options]

Modes (exactly one required):
  --dry-run    Preflight + manifest plan; no Slurm jobs
  --submit     Submit one-node GPU flash smoke job

Options:
  --stamp STAMP       Artifact stamp (default: date-based)
  --benches A,B       Comma-separated benchmark names (default: hlsfactory_gemm)
  -h, --help          Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --submit) SUBMIT=1; shift ;;
    --stamp) shift; STAMP="$1"; shift ;;
    --benches) shift; BENCHES="$1"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "${DRY_RUN}" -eq 1 && "${SUBMIT}" -eq 1 ]]; then
  echo "ERROR: use --dry-run or --submit, not both" >&2
  exit 2
fi
if [[ "${DRY_RUN}" -eq 0 && "${SUBMIT}" -eq 0 ]]; then
  echo "ERROR: specify --dry-run or --submit" >&2
  usage >&2
  exit 2
fi

export C2HLS_NIBI_FLASH_SMOKE_STAMP="${STAMP}"
export C2HLS_NIBI_FLASH_SMOKE_BENCHES="${BENCHES}"

worker_cmd=(
  "${PY}" scripts/nibi/run_flash_smoke_batch.py
  --nibi
  --benches "${BENCHES}"
  --stamp "${STAMP}"
)

echo "Nibi flash smoke stamp=${STAMP} walltime=${NIBI_WALLTIME}"
echo "benches=${BENCHES} partition=${NIBI_GPU_PARTITION} account=${NIBI_SLURM_ACCOUNT}"
echo ""

if [[ "${DRY_RUN}" -eq 1 ]]; then
  "${worker_cmd[@]}" --dry-run --skip-preflight
  blockers="$("${PY}" -c "
import sys
sys.path.insert(0, 'scripts/nibi')
from flash_lib import preflight_blockers
for b in preflight_blockers():
    print(b)
" 2>/dev/null || true)"
  if [[ -n "${blockers}" ]]; then
    echo ""
    echo "Preflight notes (non-fatal for dry-run):"
    echo "${blockers}"
  fi
  echo "dry-run ok"
  exit 0
fi

mkdir -p artifacts/nibi/slurm
SBATCH="${SCRIPT_DIR}/run_flash_smoke_one_node.sbatch.sh"
export C2HLS_NIBI_FLASH_SMOKE_STAMP="${STAMP}"
export C2HLS_NIBI_FLASH_SMOKE_BENCHES="${BENCHES}"

JOB_ID="$(sbatch --parsable \
  --account="${NIBI_SLURM_ACCOUNT}" \
  --partition="${NIBI_GPU_PARTITION}" \
  --time="${NIBI_WALLTIME}" \
  --gres="gpu:h100:${NIBI_GPU_GPUS}" \
  --cpus-per-task="${NIBI_GPU_CPUS_PER_TASK}" \
  --mem="${NIBI_GPU_MEM}" \
  --job-name="c2hls-nibi-flash-smoke" \
  --output="artifacts/nibi/slurm/flash-smoke-${STAMP}-%j.out" \
  --error="artifacts/nibi/slurm/flash-smoke-${STAMP}-%j.err" \
  --export=ALL,C2HLS_ROOT="${C2HLS_ROOT}",C2HLS_SITE=nibi,C2HLS_NIBI_FLASH_SMOKE_STAMP="${STAMP}",C2HLS_NIBI_FLASH_SMOKE_BENCHES="${BENCHES}" \
  "${SBATCH}")"

echo "Submitted Nibi flash smoke job ${JOB_ID}"
echo "Artifacts: artifacts/nibi/flash_smoke_${STAMP}/"
echo "Logs: artifacts/nibi/slurm/flash-smoke-${STAMP}-${JOB_ID}.out"
