#!/bin/bash
# Wait for serve job to expose dpo, then launch held-out base vs DPO A/B and compare.
set -euo pipefail

C2HLS="${C2HLS_ROOT:-/scratch/hpc-prf-llmfpga/asa582/projects/c2hls}"
EVAL_ROOT="${1:?eval root}"
SERVE_JOB="${2:?serve job id}"
CONFIG="${C2HLS}/rl/eval/batch_parallel_heldout_u280.json"
LOG="${EVAL_ROOT}/autolaunch.log"

log(){ echo "[$(date -Is)] $*" | tee -a "$LOG"; }

log "waiting for serve=$SERVE_JOB dpo healthy under $EVAL_ROOT"

URL=""
for i in $(seq 1 20000); do  # ~many days at 60s
  st=$(squeue -j "$SERVE_JOB" -h -o '%T' 2>/dev/null || echo GONE)
  if [[ "$st" == "FAILED" || "$st" == "CANCELLED" || "$st" == "TIMEOUT" || "$st" == "GONE" ]]; then
    # maybe completed? check endpoint file
    if [[ -f "$EVAL_ROOT/llm_endpoint.json" ]]; then
      :
    else
      log "SERVE_DEAD state=$st"; exit 1
    fi
  fi
  if [[ -f "$EVAL_ROOT/llm_endpoint.json" ]]; then
    job=$(python3 -c "import json;print(json.load(open('$EVAL_ROOT/llm_endpoint.json')).get('job_id',''))" 2>/dev/null || true)
    url=$(python3 -c "import json;print(json.load(open('$EVAL_ROOT/llm_endpoint.json')).get('url',''))" 2>/dev/null || true)
    if [[ "$job" == "$SERVE_JOB" && -n "$url" ]]; then
      models=$(curl -sf --max-time 5 "$url/models" 2>/dev/null | python3 -c "import sys,json; print(','.join(m['id'] for m in json.load(sys.stdin)['data']))" 2>/dev/null || true)
      if (( i % 10 == 1 )); then log "poll=$i state=$st models=$models"; fi
      if echo "$models" | grep -q 'dpo'; then
        URL="$url"
        log "HEALTHY url=$URL models=$models"
        break
      fi
    elif (( i % 30 == 1 )); then
      log "poll=$i state=$st waiting endpoint job=$job"
    fi
  elif (( i % 30 == 1 )); then
    log "poll=$i state=$st no endpoint yet"
  fi
  sleep 60
done

if [[ -z "$URL" ]]; then
  log "gave up waiting for dpo"; exit 1
fi

cd "$C2HLS"
export OPENAI_API_KEY=local-vllm
export C2HLS_PART=xcu280-fsvh2892-2L-e
export C2HLS_CLOCK_NS=3.33
export C2HLS_RUN_COSIM=1
export C2HLS_REFERENCE_COSIM=1
export C2HLS_COSIM_TIMEOUT=7200
export C2HLS_DEEPSEEK_SKIP_PEAK=1
export PC2_WALLTIME=24:00:00
export PC2_HELPER_WALLTIME=72:00:00

: > "$EVAL_ROOT/arms_heldout4.txt"
declare -A CAMPS

for arm_model in "base:mistralai/Devstral-2-123B-Instruct-2512" "dpo:dpo"; do
  arm=${arm_model%%:*}; model=${arm_model#*:}
  stamp="heldout4_${arm}_$(date -u +%Y%m%d_%H%M%S)"
  log "launching arm=$arm model=$model stamp=$stamp"
  BATCH_PARALLEL_CONFIG="$CONFIG" \
  BATCH_PARALLEL_EXTERNAL_ENDPOINT_URL="$URL" \
  BATCH_PARALLEL_EXTERNAL_MODEL="$model" \
  C2HLS_MODEL="$model" \
  ./scripts/pc2/start_batch_parallel_campaign.sh --external-llm --stamp "$stamp" \
    | tee "$EVAL_ROOT/launch4_${arm}.log"
  camp=$(rg -o 'campaign_root=\S+' "$EVAL_ROOT/launch4_${arm}.log" | head -1 | cut -d= -f2-)
  echo "$arm|$model|$stamp|$camp" >> "$EVAL_ROOT/arms_heldout4.txt"
  CAMPS[$arm]="$camp"
  # mark skip peak in campaign.json
  python3 - <<PY
import json
from pathlib import Path
p=Path("$camp")/"campaign.json"
d=json.loads(p.read_text())
d["skip_peak_pause"]=True
p.write_text(json.dumps(d, indent=2)+"\n")
PY
done

BASE="${CAMPS[base]}"
DPO="${CAMPS[dpo]}"
log "base=$BASE dpo=$DPO"

# Wait until drain is RUNNING for both (codegen path)
for i in $(seq 1 120); do
  ok=1
  for camp in "$BASE" "$DPO"; do
    drain=$(python3 -c "import json;print((json.load(open('$camp/campaign.json')).get('helper_jobs') or {}).get('drain') or '')")
    st=$(squeue -j "$drain" -h -o '%T' 2>/dev/null || echo GONE)
    log "drain $drain state=$st camp=$(basename "$camp")"
    if [[ "$st" != "RUNNING" ]]; then ok=0; fi
  done
  if [[ "$ok" == 1 ]]; then log "both drains RUNNING"; break; fi
  sleep 30
done

# Monitor until results + compare
for i in $(seq 1 2000); do
  bn=$(find "$BASE" -name '*_multistep_results.json' 2>/dev/null | wc -l)
  dn=$(find "$DPO" -name '*_multistep_results.json' 2>/dev/null | wc -l)
  bc=$(rg -c '"event":"codegen_start"' "$BASE/flow/events.jsonl" 2>/dev/null || echo 0)
  dc=$(rg -c '"event":"codegen_start"' "$DPO/flow/events.jsonl" 2>/dev/null || echo 0)
  syn=$(squeue -u "$USER" -h -o '%j' 2>/dev/null | rg -c 'heldout4_' || true)
  syn=${syn:-0}
  if (( i % 5 == 1 )); then
    log "poll=$i base_res=$bn dpo_res=$dn codegen_starts=$bc/$dc synthish=$syn"
  fi
  # abort?
  if [[ -f "$BASE/flow/aborted" || -f "$DPO/flow/aborted" ]]; then
    log "abort marker seen; comparing anyway"; break
  fi
  if [[ "$bn" -ge 3 && "$dn" -ge 3 && "$syn" -eq 0 ]]; then
    log "both arms complete"; break
  fi
  sleep 60
done

python3 "$C2HLS/rl/scripts/compare_heldout_ab.py" \
  --base-campaign "$BASE" \
  --dpo-campaign "$DPO" \
  --out "$EVAL_ROOT/compare_heldout4.md" \
  2>&1 | tee "$EVAL_ROOT/compare_heldout4.run.log" | tee -a "$LOG"
log "wrote $EVAL_ROOT/compare_heldout4.md"
