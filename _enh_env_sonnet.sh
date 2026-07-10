# Shared env for the ENHANCED-framework SONNET 4.6 sweeps (Argo direct endpoint).
# Mirror of the Opus _enh_env.sh (archived) with MODEL=claude-sonnet-4-6 and pinned
# to the miniconda python that has anthropic/openai/dotenv/globus_sdk. Sourced by
# _run_sonnet_campaign.sh; NOT executable on its own.
# Run in the Ubuntu-22.04 distro: system python3 has anthropic/openai/dotenv/globus_sdk,
# g++ is at /usr/bin/g++, Vitis HLS at /tools/Xilinx. (NOT the default 'Ubuntu' distro.)
PY=python3

export VITIS_SETTINGS=/tools/Xilinx/Vitis_HLS/2023.2/settings64.sh
source "$VITIS_SETTINGS" >/dev/null 2>&1
export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/lib:${LIBRARY_PATH:-}
export ANTHROPIC_BASE_URL=https://apps.inside.anl.gov/argoapi
export C2HLS_CLAUDE_KEY_FILE=/mnt/e/courses/UMN/c2hls/api-key.txt
export C2HLS_TARGET_PART=xcu280-fsvh2892-2L-e
export C2HLS_DISABLE_COSIM_SHRINK=1
export C2HLS_COSIM_TIMEOUT=14400
export HLS_COSIM_TIMEOUT=14400
export C2HLS_SYNTH_TIMEOUT=2400
export C2HLS_LLM_MAX_RETRIES=8
export C2HLS_DYNAMIC_ROUTING=1          # skill_library routing path
export C2HLS_SKILL_LIBRARY_PERSIST=0    # freeze library: clean ablation
export TMPDIR=/tmp

MODEL=claude-sonnet-4-6
SKILLS_BASE=/mnt/e/courses/UMN/c2hls/hls_full_optimization_skills_schema_1_1_package/skills.json

cd /mnt/e/courses/UMN/c2hls/code_translation_c2hls

EXCLUDE="hlsfactory_heat-3d hlsfactory_seidel-2d"
BENCHES=$("$PY" -c "
import pathlib
excl = set('${EXCLUDE}'.split())
print(','.join(sorted(p.name for p in pathlib.Path('benchmarks').iterdir()
                      if p.name.startswith('hlsfactory_') and p.name not in excl)))
")
