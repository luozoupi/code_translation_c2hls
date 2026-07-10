#!/bin/bash
# Sonnet 4.6 enhanced-framework counterpart to the Opus campaign — Core-3 arms:
#   1. one-shot (flash, turns=1, no repair, skills off, routing off)
#   2. multistep curated (routed skills)
#   3. multistep all-positive (41 skills/step)
# Same knobs/timeouts as the Opus _enh_env; MODEL=claude-sonnet-4-6; out dirs *_SONNET.
# Serial (cosim = one vitis_hls at a time). Launch detached with setsid.
set -u
cd /mnt/e/courses/UMN/c2hls/code_translation_c2hls

# Don't contend with any active cosim.
while pgrep -x vitis_hls >/dev/null 2>&1; do
  echo "SONNET: vitis_hls active, waiting...  $(date '+%F %T')"; sleep 60
done
echo "######## SONNET 4.6 CORE-3 CAMPAIGN START $(date '+%F %T') ########"

# ---- arm 1: one-shot baseline ----
(
  source ./_enh_env_sonnet.sh
  export C2HLS_DYNAMIC_ROUTING=0
  unset C2HLS_SKILLS_ALL_POSITIVE 2>/dev/null || true
  export C2HLS_COSIM_TIMEOUT=3600           # naive one-shot kernels fail fast to csynth
  OUT=results_matrix_u280_ENH_oneshot_SONNET
  echo "### [1/3] ONE-SHOT START $(date '+%F %T')  model=$MODEL  -> $OUT"
  "$PY" matrix_sweep.py --benches "$BENCHES" --models "$MODEL" --modes flash \
      --skills-modes off --turns 1 --quality-repair-turns 0 \
      --out "$OUT" --cell-timeout 43200 2>&1
  echo "sonnet-oneshot-exit=$?  $(date '+%F %T')"
)

# ---- arms 2+3: multistep curated then all-positive ----
run_ms() {  # $1 = allpos (0|1)
  source ./_enh_env_sonnet.sh
  export C2HLS_COSIM_TIMEOUT=3600           # 1h cap per step cosim
  if [ "$1" = "1" ]; then
    export C2HLS_SKILLS_ALL_POSITIVE=1; TAG=allpositive
  else
    unset C2HLS_SKILLS_ALL_POSITIVE 2>/dev/null || true; TAG=curated
  fi
  OUT=results_matrix_u280_ENH_${TAG}_multistep_skills_SONNET
  echo "### MULTISTEP $TAG START $(date '+%F %T')  model=$MODEL  -> $OUT"
  "$PY" matrix_sweep.py --benches "$BENCHES" --models "$MODEL" --modes multistep \
      --skills-modes on --skills-path "$SKILLS_BASE" --turns 5 --quality-repair-turns 4 \
      --out "$OUT" --cell-timeout 64800 2>&1
  echo "sonnet-multistep-${TAG}-exit=$?  $(date '+%F %T')"
}

echo "### [2/3] curated multistep"
run_ms 0
echo "### [3/3] all-positive multistep"
run_ms 1
echo "######## SONNET 4.6 CORE-3 CAMPAIGN END $(date '+%F %T') ########"
