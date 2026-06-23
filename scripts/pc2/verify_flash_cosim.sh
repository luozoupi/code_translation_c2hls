#!/usr/bin/env bash
# Verify Vitis env + cosim manifest before Slurm submission.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

RUN_ROOT="${1:-${C2HLS_FLASH_COSIM_RUN_ROOT:-}}"
if [[ -z "${RUN_ROOT}" ]]; then
  echo "usage: $0 <cosim_run_root>" >&2
  exit 2
fi

export C2HLS_SITE=pc2
export C2HLS_RUN_COSIM=1
# shellcheck disable=SC1091
source "${C2HLS_ROOT}/scripts/setup_emu_env.sh"

MANIFEST="${RUN_ROOT}/manifest.json"
if [[ ! -f "${MANIFEST}" ]]; then
  echo "ERROR: missing manifest: ${MANIFEST}" >&2
  exit 2
fi

echo "== Vitis =="
command -v vitis-run
vitis-run --version 2>&1 | head -3 || true
echo "C2HLS_VITIS_SETTINGS=${C2HLS_VITIS_SETTINGS:-}"
echo "C2HLS_TMP_ROOT=${C2HLS_TMP_ROOT:-}"

echo "== Manifest =="
python3 - <<'PY' "${MANIFEST}"
import json, sys
from pathlib import Path
manifest = json.loads(Path(sys.argv[1]).read_text())
cells = manifest.get("cells", [])
print(f"cells={len(cells)}")
missing = [c for c in cells if not Path(c["final_cpp"]).exists()]
print(f"missing_final_cpp={len(missing)}")
if missing[:3]:
    for c in missing[:3]:
        print("  ", c["cell_id"], c["final_cpp"])
unsupported = [c for c in cells if not c.get("supports_cosim")]
print(f"unsupported_cosim={len(unsupported)}")
PY

echo "== Dry-run first cell =="
FIRST_CELL="$(python3 - <<'PY' "${MANIFEST}"
import json, sys
from pathlib import Path
cells = json.loads(Path(sys.argv[1]).read_text()).get("cells", [])
print(json.dumps(cells[0]) if cells else "")
PY
)"
if [[ -n "${FIRST_CELL}" ]]; then
  C2HLS_ROOT="${C2HLS_ROOT}" FIRST_CELL="${FIRST_CELL}" python3 - <<'PY'
import json, os, sys
from pathlib import Path
sys.path.insert(0, os.environ["C2HLS_ROOT"])
from scripts.pc2.flash_cosim_lib import CosimCell, load_cosim_inputs
cell = CosimCell(**json.loads(os.environ["FIRST_CELL"]))
bench_dir = Path(os.environ["C2HLS_ROOT"]) / "benchmarks" / cell.bench
inputs = load_cosim_inputs(bench_dir)
print(f"cell_id={cell.cell_id}")
print(f"bench={cell.bench} top={inputs['top_function']}")
print(f"final_cpp={cell.final_cpp}")
print(f"cosim_tb={inputs['meta'].get('cosim_testbench_file')}")
print(f"cosim_support={inputs['meta'].get('cosim_support_files')}")
PY
fi

echo "OK: cosim preflight passed for ${RUN_ROOT}"
