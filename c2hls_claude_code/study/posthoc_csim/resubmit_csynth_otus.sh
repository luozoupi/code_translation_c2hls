#!/bin/bash
# resubmit_csynth_otus.sh [--dry-run|--check|--compare] : on Otus, the 21 csynths pending on Fir on 2026-10-22
# (otus_csynth_resubmit.txt: 19 post-hoc kernels under posthoc_csim/csynth/, 2 team kernels in their exp_post/ dirs, each
# with the task.json its Fir job used), submitted with job_csynth_otus.sh. A dir whose csynth_summary.json says
# success is skipped.
#   --check   first: csynth of team v3 (Fir: exp_post/v3_csynth) into posthoc_csim/otus_check_v3/
#   --compare after it: every resource and latency field must equal Fir's; only then submit the 21
set -eo pipefail
P=$(dirname "$(readlink -f "$0")"); STUDY=$(dirname "$P")
MODE=${1:-}
[ "${CC_CLUSTER:-}" = fir ] && [ "$MODE" != --dry-run ] && { echo "this is Fir: only --dry-run here"; exit 1; }
REF=$STUDY/agentfw/runs/mm1024_team_r1/exp_post/v3_csynth; CHK=$P/otus_check_v3
sub() {  # sub <job name> <dir> <task.json> <time> <cpus> <mem>
  local c=(sbatch --parsable -J "$1" -t "$4" --cpus-per-task="$5" --mem="$6" -o "$2/csynth-%j.out" -e "$2/csynth-%j.err"
           --export=ALL,STUDY_DIR="$STUDY" "$P/job_csynth_otus.sh" "$2" "$3")
  if [ "$MODE" = --dry-run ]; then echo "${c[*]}"; else echo "$("${c[@]}") $1 ${2#$STUDY/}" | tee -a "$P/otus_csynth_submitted.txt"; fi
}
case "$MODE" in
--check)
  mkdir -p "$CHK"; cp "$REF/kernel.cpp" "$CHK/"
  sub otus_check_v3 "$CHK" "$STUDY/agentfw/runs/mm1024_team_r1/task.json" 12:00:00 8 64G
  echo "when it finishes: $0 --compare"; exit 0 ;;
--compare)
  python3 - "$REF/csynth_summary.json" "$CHK/csynth_summary.json" <<'PY'
import json, sys
f, o = (json.load(open(p))["report"] for p in sys.argv[1:])
keys = ["latency_cycles", "latency_cycles_worst", "interval", "dsp", "lut", "ff", "bram", "uram"]
bad = [k for k in keys if f.get(k) != o.get(k)]
for k in keys: print(f"{k:22} Fir {f.get(k)!s:>10}  Otus {o.get(k)!s:>10}  {'DIFFERS' if k in bad else 'ok'}")
print("MATCH: submit the 21 with no option" if not bad else "MISMATCH: don't mix Otus csynths with Fir's")
sys.exit(1 if bad else 0)
PY
  exit $? ;;
esac
grep -v '^#' "$P/otus_csynth_resubmit.txt" | while read -r NAME DIR TASK TIME CPUS MEM FIR; do
  D=$STUDY/$DIR
  [ -f "$D/kernel.cpp" ] || { echo "skip (no kernel.cpp): $DIR"; continue; }
  if [ -f "$D/csynth_summary.json" ] && python3 -c "import json,sys; sys.exit(0 if json.load(open('$D/csynth_summary.json')).get('success') else 1)"; then
    echo "skip (done): $DIR"; continue; fi
  sub "$NAME" "$D" "$STUDY/$TASK" "$TIME" "$CPUS" "$MEM"
done
