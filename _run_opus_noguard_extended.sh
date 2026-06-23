#!/bin/bash
# Opus 4.8 counterpart of the Sonnet no-guardrails ablation: flash, base +
# skills_extension_noguard.json (the 2 prohibition guard skills dropped, only
# constructive loop_tripcount kept). Completes the 2x3 model x guard matrix:
#   Sonnet: base / ext-with-guards / ext-NO-guard  (all done)
#   Opus:   base / ext-with-guards (done) / ext-NO-guard  <- THIS
#
# Research question: Opus already resisted the bad guards (ext-guards 1.003x vs
# Sonnet 1.268x). Does removing them recover Opus toward its base 0.633x?
#
# OLD framework (same as all other Opus/Sonnet runs). max_retries=8 active.
# Out dir: results_matrix_u280_fullcosim_noguard_OPUS
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

MODEL=claude-opus-4-8
PKG=/mnt/e/courses/UMN/c2hls/hls_full_optimization_skills_schema_1_1_package
SKILLS_BASE="$PKG/skills.json"
SKILLS_NOGUARD="$PKG/skills_extension_noguard.json"
SKILLS_PATH="${SKILLS_BASE}:${SKILLS_NOGUARD}"

cd /mnt/e/courses/UMN/c2hls/code_translation_c2hls

EXCLUDE="hlsfactory_heat-3d hlsfactory_seidel-2d"
BENCHES=$(python3 -c "
import pathlib
excl = set('${EXCLUDE}'.split())
print(','.join(sorted(p.name for p in pathlib.Path('benchmarks').iterdir()
                      if p.name.startswith('hlsfactory_') and p.name not in excl)))
")

echo "=================================================================="
echo "OPUS NO-GUARDRAILS extended sweep  START  $(date '+%F %T')"
echo "  Model:        $MODEL"
echo "  Skills:       base + skills_extension_noguard.json (guards dropped)"
echo "  Skills path:  $SKILLS_PATH"
echo "  Out dir:      results_matrix_u280_fullcosim_noguard_OPUS"
echo "=================================================================="

python3 matrix_sweep.py \
    --benches "$BENCHES" --models "$MODEL" --modes flash --skills-modes on \
    --skills-path "$SKILLS_PATH" --turns 5 --quality-repair-turns 4 \
    --out results_matrix_u280_fullcosim_noguard_OPUS --cell-timeout 43200 \
  2>&1
echo "opus-noguard-exit=$?  $(date '+%F %T')"
