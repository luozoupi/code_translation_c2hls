#!/bin/bash
# ABLATION: Sonnet 4.6 flash extended-skills sweep with the GUARD skills
# removed, to test whether the preventative "avoid/DO NOT" instructions drove
# the Phase-9 regression.
#
# Identical to the Phase-9 extended-skills run (_run_phase9_extended_skills.sh)
# EXCEPT the skill extension is skills_extension_noguard.json (the 2 guard
# skills hls-guard-fp-reduction-order-preserving + hls-guard-device-budget-
# artix7-100t dropped; only the constructive loop_tripcount skill kept).
#
# Single variable vs Phase-9: the 2 guard skills. Base skills.json unchanged.
# Runs on the OLD framework (same as Phase-8/9 Sonnet runs) — the working tree
# has the old c2hls.py/hls_eval.py swapped in for the duration of the Opus
# campaign, which is exactly what we want for comparability. DO NOT git restore
# the framework files until this sweep finishes too.
#
# Out dir: results_matrix_u280_fullcosim_noguard
set -u

export VITIS_SETTINGS=/tools/Xilinx/Vitis_HLS/2023.2/settings64.sh
source "$VITIS_SETTINGS" >/dev/null 2>&1
export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/lib:${LIBRARY_PATH:-}
export ANTHROPIC_BASE_URL=https://apps.inside.anl.gov/argoapi
export C2HLS_CLAUDE_KEY_FILE=/mnt/e/courses/UMN/c2hls/api-key.txt
export C2HLS_TARGET_PART=xcu280-fsvh2892-2L-e
export C2HLS_DISABLE_COSIM_SHRINK=1
export HLS_COSIM_TIMEOUT=14400
export TMPDIR=/tmp

MODEL=claude-sonnet-4-6
SKILLS_BASE=/mnt/e/courses/UMN/c2hls/hls_full_optimization_skills_schema_1_1_package/skills.json
SKILLS_NOGUARD=/mnt/e/courses/UMN/c2hls/hls_full_optimization_skills_schema_1_1_package/skills_extension_noguard.json
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
echo "SONNET NO-GUARDRAILS extended sweep  START  $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Model:        $MODEL"
echo "  Framework:    OLD (Sonnet-matching, swapped working tree)"
echo "  Skills:       base + skills_extension_noguard.json (guards dropped)"
echo "  Skills path:  $SKILLS_PATH"
echo "  Target:       $C2HLS_TARGET_PART @ 3.33 ns, full-size cosim"
echo "  Out dir:      results_matrix_u280_fullcosim_noguard"
echo "=================================================================="

python3 matrix_sweep.py \
    --benches "$BENCHES" --models "$MODEL" --modes flash --skills-modes on \
    --skills-path "$SKILLS_PATH" --turns 5 --quality-repair-turns 4 \
    --out results_matrix_u280_fullcosim_noguard --cell-timeout 43200 \
  2>&1
echo "noguard-sweep-exit=$?  $(date '+%Y-%m-%d %H:%M:%S')"
