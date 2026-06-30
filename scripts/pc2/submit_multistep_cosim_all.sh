#!/usr/bin/env bash
# Build multistep cosim manifest and submit individual full-size jobs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

STAMP="${C2HLS_MULTISTEP_COSIM_STAMP:-$(date +%Y%m%d_%H%M%S)}"
export C2HLS_MULTISTEP_COSIM_STAMP="${STAMP}"
ARTIFACT_GLOB="${C2HLS_MULTISTEP_COSIM_ARTIFACT_GLOB:-multistep_fixed_cosim_*}"
ARTIFACT_FILTER=""
FULL_SIZE=1
SKIP_VERIFY=0
MAX_PARALLEL="${PC2_COSIM_ARRAY_MAX_PARALLEL:-40}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stamp) STAMP="$2"; export C2HLS_MULTISTEP_COSIM_STAMP="${STAMP}"; shift 2 ;;
    --artifact-glob) ARTIFACT_GLOB="$2"; shift 2 ;;
    --artifact) ARTIFACT_FILTER="${ARTIFACT_FILTER} $2"; shift 2 ;;
    --skip-verify) SKIP_VERIFY=1; shift ;;
    --max-parallel) MAX_PARALLEL="$2"; shift 2 ;;
    -h|--help)
      echo "usage: $0 [--stamp STAMP] [--artifact-glob GLOB] [--artifact BASENAME]..."
      exit 0
      ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

COSIM_ROOT="${C2HLS_MULTISTEP_COSIM_ROOT:-${C2HLS_ROOT}/artifacts/pc2/multistep_cosim}"
RUN_ROOT="${COSIM_ROOT}/${STAMP}"
export C2HLS_MULTISTEP_COSIM_ROOT="${COSIM_ROOT}"
export C2HLS_FLASH_COSIM_RUN_ROOT="${RUN_ROOT}"
mkdir -p "${RUN_ROOT}/submissions" "${COSIM_ROOT}/slurm"

export C2HLS_FLASH_COSIM_FULL_SIZE=1
export C2HLS_MULTISTEP_COSIM_FULL_SIZE=1
export C2HLS_COSIM_TIMEOUT="${C2HLS_COSIM_TIMEOUT:-43200}"

BUILD_ARGS=(--stamp "${STAMP}" --artifact-glob "${ARTIFACT_GLOB}")
for art in ${ARTIFACT_FILTER}; do
  [[ -n "${art}" ]] && BUILD_ARGS+=(--artifact "${art}")
done

if [[ ! -f "${RUN_ROOT}/manifest.json" ]]; then
  pc2_log "building multistep cosim manifest stamp=${STAMP}"
  python3 "${SCRIPT_DIR}/build_multistep_cosim_manifest.py" "${BUILD_ARGS[@]}"
fi

CELL_COUNT="$(python3 - <<'PY' "${RUN_ROOT}/manifest.json"
import json, sys
from pathlib import Path
print(len(json.loads(Path(sys.argv[1]).read_text()).get("cells", [])))
PY
)"
if [[ "${CELL_COUNT}" -le 0 ]]; then
  echo "ERROR: manifest has zero cells" >&2
  exit 2
fi

WALLTIME="${PC2_COSIM_WALLTIME:-13:00:00}"
CPUS="${PC2_COSIM_CPUS:-8}"
MEM="${PC2_COSIM_MEM:-32G}"
PARTITION="${PC2_COSIM_PARTITION:-${PC2_COMPUTE_PARTITION:-normal}}"

pc2_log "submitting ${CELL_COUNT} individual multistep cosim jobs stamp=${STAMP}"
python3 - <<'PY' "${RUN_ROOT}/manifest.json" "${SCRIPT_DIR}/submit_flash_cosim_one.sh" "${RUN_ROOT}"
import json, subprocess, sys
from pathlib import Path
manifest = Path(sys.argv[1])
submit = sys.argv[2]
run_root = sys.argv[3]
for cell in json.loads(manifest.read_text()).get("cells", []):
    subprocess.run([submit, cell["cell_id"], run_root], check=True)
PY

pc2_log "submitted ${CELL_COUNT} jobs -> ${RUN_ROOT}"
