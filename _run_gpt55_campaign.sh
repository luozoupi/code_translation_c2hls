#!/bin/bash
# GPT-5.5 enhanced-framework counterpart, Core-3 arms (oneshot + multistep curated
# + multistep all-positive). Argo OpenAI-compat routing. Serial cosim. out dirs *_GPT55.
set -u
cd /mnt/e/courses/UMN/c2hls/code_translation_c2hls
while pgrep -x vitis_hls >/dev/null 2>&1; do echo "GPT55: vitis busy, waiting $(date '+%F %T')"; sleep 60; done
echo "######## GPT-5.5 CORE-3 CAMPAIGN START $(date '+%F %T') ########"

( source ./_enh_env_gpt55.sh
  export C2HLS_DYNAMIC_ROUTING=0
  unset C2HLS_SKILLS_ALL_POSITIVE 2>/dev/null || true
  export C2HLS_COSIM_TIMEOUT=3600
  OUT=results_matrix_u280_ENH_oneshot_GPT55
  echo "### [1/3] ONE-SHOT START $(date '+%F %T')  model=$MODEL -> $OUT"
  "$PY" matrix_sweep.py --benches "$BENCHES" --models "$MODEL" --modes flash \
      --skills-modes off --turns 1 --quality-repair-turns 0 --out "$OUT" --cell-timeout 43200 2>&1
  echo "gpt55-oneshot-exit=$?  $(date '+%F %T')" )

run_ms() {
  source ./_enh_env_gpt55.sh
  export C2HLS_COSIM_TIMEOUT=3600
  if [ "$1" = "1" ]; then export C2HLS_SKILLS_ALL_POSITIVE=1; TAG=allpositive
  else unset C2HLS_SKILLS_ALL_POSITIVE 2>/dev/null || true; TAG=curated; fi
  OUT=results_matrix_u280_ENH_${TAG}_multistep_skills_GPT55
  echo "### MULTISTEP $TAG START $(date '+%F %T')  model=$MODEL -> $OUT"
  "$PY" matrix_sweep.py --benches "$BENCHES" --models "$MODEL" --modes multistep \
      --skills-modes on --skills-path "$SKILLS_BASE" --turns 5 --quality-repair-turns 4 \
      --out "$OUT" --cell-timeout 64800 2>&1
  echo "gpt55-multistep-${TAG}-exit=$?  $(date '+%F %T')"
}
echo "### [2/3] curated multistep"; run_ms 0
echo "### [3/3] all-positive multistep"; run_ms 1
echo "######## GPT-5.5 CORE-3 CAMPAIGN END $(date '+%F %T') ########"
