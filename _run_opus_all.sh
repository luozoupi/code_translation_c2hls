#!/bin/bash
# Opus 4.8 FULL REPLICATION of the Sonnet 4.6 sweep matrix, on the
# c2hls_enhanced framework. Mirrors the Sonnet phase-8/9/10 axes exactly,
# only swapping --models claude-opus-4-8 and _OPUS output dirs.
#
# NOTE ON COMPARABILITY: the Sonnet baseline data was generated on the OLD
# pre-agent-split framework; this Opus run is on the ENHANCED framework.
# The Opus-vs-Sonnet comparison therefore mixes model + framework version.
# (User chose this explicitly on 2026-06-15.)
#
# Three sub-sweeps run sequentially; each uses matrix_sweep.py's matrix.json
# resume so a crash/restart picks up where it left off. Gold is reused from
# the existing cache (reference-kernel synth is model-independent).
#
#   1. flash, skills off + on (base)   -> results_matrix_u280_fullcosim_OPUS        (52 cells)
#   2. flash, skills on (base+ext)     -> results_matrix_u280_fullcosim_extended_OPUS (26 cells)
#   3. multistep, skills on (base)     -> results_matrix_u280_multistep_base_OPUS     (26 cells)
#
# Total 104 cells. Est. multi-day to ~2 weeks wall (long-tail cosim + Opus
# generation latency).
set -u

export VITIS_SETTINGS=/tools/Xilinx/Vitis_HLS/2023.2/settings64.sh
source "$VITIS_SETTINGS" >/dev/null 2>&1
export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/lib:${LIBRARY_PATH:-}
export ANTHROPIC_BASE_URL=https://apps.inside.anl.gov/argoapi
# Argo auth = Argonne username, stored in the local api-key.txt (4 bytes).
# The orchestrator's key loader reads ANTHROPIC_API_KEY env, else this file.
# Point it at the local copy so we don't depend on the collaborator's
# hardcoded /home/luo00466/claude-api-key.txt default.
export C2HLS_CLAUDE_KEY_FILE=/mnt/e/courses/UMN/c2hls/api-key.txt
export C2HLS_TARGET_PART=xcu280-fsvh2892-2L-e
export C2HLS_DISABLE_COSIM_SHRINK=1
export HLS_COSIM_TIMEOUT=14400
export TMPDIR=/tmp

MODEL=claude-opus-4-8
SKILLS_BASE=/mnt/e/courses/UMN/c2hls/hls_full_optimization_skills_schema_1_1_package/skills.json
SKILLS_EXT=/mnt/e/courses/UMN/c2hls/hls_full_optimization_skills_schema_1_1_package/skills_extension.json
SKILLS_BASEEXT="${SKILLS_BASE}:${SKILLS_EXT}"

cd /mnt/e/courses/UMN/c2hls/code_translation_c2hls

# 26-bench set (heat-3d, seidel-2d excluded — gold cosim > 4h, same as Sonnet).
EXCLUDE="hlsfactory_heat-3d hlsfactory_seidel-2d"
BENCHES=$(python3 -c "
import pathlib
excl = set('${EXCLUDE}'.split())
print(','.join(sorted(p.name for p in pathlib.Path('benchmarks').iterdir()
                      if p.name.startswith('hlsfactory_') and p.name not in excl)))
")

echo "=================================================================="
echo "OPUS 4.8 FULL REPLICATION  START  $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Model:        $MODEL"
echo "  Framework:    OLD (Sonnet-matching: main+stash, flat structure, gold-cache)"
echo "                NOTE: working tree files c2hls.py/hls_eval.py/prompt_c2hls.py/"
echo "                rubric.py/report.py are the OLD framework, swapped in via"
echo "                'git show stash@{0}:<f>'. git restore them after the run."
echo "  Target:       $C2HLS_TARGET_PART @ 3.33 ns, full-size cosim"
echo "  Cosim cap:    ${HLS_COSIM_TIMEOUT}s   Per-cell cap: 43200s"
echo "  Benches:      $(echo $BENCHES | tr ',' '\n' | wc -l)"
echo "=================================================================="

run_axis () {
  local label="$1"; shift
  echo ""
  echo "###### AXIS: $label  $(date '+%Y-%m-%d %H:%M:%S') ######"
  python3 matrix_sweep.py "$@" 2>&1
  echo "###### AXIS $label exit=$?  $(date '+%Y-%m-%d %H:%M:%S') ######"
}

# 1. flash, skills off + on (base)
run_axis "flash_off+on_base" \
    --benches "$BENCHES" --models "$MODEL" --modes flash --skills-modes on,off \
    --skills-path "$SKILLS_BASE" --turns 5 --quality-repair-turns 4 \
    --out results_matrix_u280_fullcosim_OPUS --cell-timeout 43200

# 2. flash, skills on (base + extension)
run_axis "flash_on_extended" \
    --benches "$BENCHES" --models "$MODEL" --modes flash --skills-modes on \
    --skills-path "$SKILLS_BASEEXT" --turns 5 --quality-repair-turns 4 \
    --out results_matrix_u280_fullcosim_extended_OPUS --cell-timeout 43200

# 3. multistep, skills on (base)
run_axis "multistep_on_base" \
    --benches "$BENCHES" --models "$MODEL" --modes multistep --skills-modes on \
    --skills-path "$SKILLS_BASE" --turns 5 --quality-repair-turns 4 \
    --out results_matrix_u280_multistep_base_OPUS --cell-timeout 43200

echo ""
echo "=================================================================="
echo "OPUS 4.8 FULL REPLICATION  DONE  $(date '+%Y-%m-%d %H:%M:%S')"
echo "=================================================================="
