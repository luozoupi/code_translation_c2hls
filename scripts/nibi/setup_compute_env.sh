#!/usr/bin/env bash
# Install c2hls Python venv on Nibi login/compute nodes (no GPU required).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TARGET="${NIBI_COMPUTE_VENV:-/home/asa582/scratch/asa582/packages/c2hls-venv}"
WHEEL_DIRS=(
  /cvmfs/soft.computecanada.ca/custom/python/wheelhouse/gentoo2023/x86-64-v3
  /cvmfs/soft.computecanada.ca/custom/python/wheelhouse/gentoo2023/generic
  /cvmfs/soft.computecanada.ca/custom/python/wheelhouse/generic
)

module load python/3.11.5
export PYTHONNOUSERSITE=1

if [[ ! -d "${TARGET}" ]]; then
  python -m venv "${TARGET}"
fi
# shellcheck disable=SC1091
source "${TARGET}/bin/activate"

FIND_ARGS=()
for d in "${WHEEL_DIRS[@]}"; do
  FIND_ARGS+=(--find-links="$d")
done

pip install --upgrade pip
pip install --no-index "${FIND_ARGS[@]}" \
  openai python-dotenv tqdm httpx pydantic annotated-types anyio distro h11 httpcore idna jiter sniffio typing_extensions certifi

echo "OK: c2hls venv at ${TARGET}"
"${TARGET}/bin/python" -c "import openai; print('openai', openai.__version__)"
