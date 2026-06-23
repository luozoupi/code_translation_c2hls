#!/bin/bash
# Cleanup pass for the Opus campaign:
#   Batch A: retry the 11 flash-extended cells that died on transient Argo
#            API outages (now protected by C2HLS_LLM_MAX_RETRIES=8).
#   Batch B: retry the 2 multistep cells that hit the 12h cap (ludcmp, symm)
#            with an 18h cap.
# Sequential (no concurrency -> no Vitis contention). Old framework on disk.
set -u

export VITIS_SETTINGS=/tools/Xilinx/Vitis_HLS/2023.2/settings64.sh
source "$VITIS_SETTINGS" >/dev/null 2>&1
export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/lib:${LIBRARY_PATH:-}
export ANTHROPIC_BASE_URL=https://apps.inside.anl.gov/argoapi
export C2HLS_CLAUDE_KEY_FILE=/mnt/e/courses/UMN/c2hls/api-key.txt
export C2HLS_TARGET_PART=xcu280-fsvh2892-2L-e
export C2HLS_DISABLE_COSIM_SHRINK=1
export HLS_COSIM_TIMEOUT=14400
export C2HLS_LLM_MAX_RETRIES=8
export TMPDIR=/tmp

cd /mnt/e/courses/UMN/c2hls/code_translation_c2hls
MODEL=claude-opus-4-8
PKG=/mnt/e/courses/UMN/c2hls/hls_full_optimization_skills_schema_1_1_package
SKILLS_BASE="$PKG/skills.json"
SKILLS_EXT="$PKG/skills_extension.json"

echo "=== strip failed cells so matrix_sweep retries only them ==="
python3 - <<'PY'
import json, shutil
from pathlib import Path
ROOT = Path('.')
def strip(d, benches):
    p = ROOT/d/'matrix.json'
    m = json.loads(p.read_text())
    bad = {(b, 'on') for b in benches}
    keep = [e for e in m if (e['bench'], e.get('skills')) not in bad]
    print(f'  {d}: {len(m)} -> {len(keep)}')
    shutil.copy2(p, p.with_suffix('.json.bak.cleanup'))
    p.write_text(json.dumps(keep, indent=2, default=str))
    for b in benches:
        for sub in (f'opus__flash__skills', f'opus__multistep__skills'):
            cd = ROOT/d/b/sub
            if cd.exists():
                shutil.rmtree(cd); print(f'    rm {cd}')
strip('results_matrix_u280_fullcosim_extended_OPUS',
      ['hlsfactory_covariance','hlsfactory_doitgen','hlsfactory_durbin',
       'hlsfactory_fdtd-2d','hlsfactory_floyd-warshall','hlsfactory_gemm',
       'hlsfactory_gesummv','hlsfactory_gramschmidt','hlsfactory_jacobi-1d',
       'hlsfactory_jacobi-2d','hlsfactory_lu'])
strip('results_matrix_u280_multistep_base_OPUS',
      ['hlsfactory_ludcmp','hlsfactory_symm'])
PY

EXCLUDE="hlsfactory_heat-3d hlsfactory_seidel-2d"
BENCHES=$(python3 -c "
import pathlib
excl = set('${EXCLUDE}'.split())
print(','.join(sorted(p.name for p in pathlib.Path('benchmarks').iterdir()
                      if p.name.startswith('hlsfactory_') and p.name not in excl)))
")

echo ""
echo "###### BATCH A: flash-extended retry (11 cells)  $(date '+%F %T') ######"
python3 matrix_sweep.py --benches "$BENCHES" --models "$MODEL" --modes flash \
    --skills-modes on --skills-path "${SKILLS_BASE}:${SKILLS_EXT}" \
    --turns 5 --quality-repair-turns 4 \
    --out results_matrix_u280_fullcosim_extended_OPUS --cell-timeout 43200 2>&1
echo "batchA-exit=$?  $(date '+%F %T')"

echo ""
echo "###### BATCH B: multistep retry @18h cap (2 cells)  $(date '+%F %T') ######"
python3 matrix_sweep.py --benches "$BENCHES" --models "$MODEL" --modes multistep \
    --skills-modes on --skills-path "${SKILLS_BASE}" \
    --turns 5 --quality-repair-turns 4 \
    --out results_matrix_u280_multistep_base_OPUS --cell-timeout 64800 2>&1
echo "batchB-exit=$?  $(date '+%F %T')"
echo "###### OPUS CLEANUP DONE  $(date '+%F %T') ######"
