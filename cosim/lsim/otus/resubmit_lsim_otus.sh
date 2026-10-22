#!/bin/bash
# resubmit_lsim_otus.sh [--dry-run|--check] : on Otus, submit the 3 LightningSim jobs pending on Fir on 2026-10-22.
#   lsim_soda2    (Fir 66100654) names_lsim_soda2.txt,    v2023, PAR=12, 16 cpus, 64G, 24 h
#   lsim_soda_pat (Fir 66218914) names_lsim_soda_pat.txt, v2023, PAR=8,   8 cpus, 64G, 12 h
#   lsim_proj     (Fir 66181502) SODA csynth of designs/contrast_u16_cr into ROOT/contrast_u16_cr/proj.tar.gz, 192G, 24 h
# A name whose ROOT/NAME/lsim_v2023.json already exists is left out (ALL=1 keeps it); a list left empty is not submitted.
# --check : first run, before the rest: one project Fir already finished, written as lsim_otuscheck.json; its "end"
#           cycle must equal the one in lsim_v2023.json from Fir.
# Env: LSIM_T (setup_lsim_otus.sh dir), LSIM_ROOT (the packed projects, rsynced from Fir's /scratch/asa582/tmp/c2hls/lsim/soda).
set -eo pipefail
HERE=$(dirname "$(readlink -f "$0")")        # cosim/lsim/otus
LS=$(dirname "$HERE")                         # cosim/lsim
CC=$(dirname "$(dirname "$LS")")              # c2hls_claude_code
W=$(dirname "$CC")                            # workspaces
ROOT=${LSIM_ROOT:-$(dirname "$W")/tmp/c2hls/lsim/soda}
LSIM_T=${LSIM_T:-/scratch/hpc-prf-llmfpga/asa582/projects/tools/lightningsim}
MODE=${1:-}
[ "${CC_CLUSTER:-}" = fir ] && [ "$MODE" != --dry-run ] && { echo "this is Fir: only --dry-run here"; exit 1; }
[ -d "$ROOT" ] || { echo "no LSIM_ROOT $ROOT"; exit 1; }
[ -x "$LSIM_T/env/bin/lightningsim" ] || { echo "no LightningSim in $LSIM_T: run setup_lsim_otus.sh"; exit 1; }
mkdir -p "$LS/logs"
sub() { if [ "$MODE" = --dry-run ]; then echo "sbatch $*"; else J=$(sbatch --parsable "$@"); echo "$J $*" | tee -a "$HERE/otus_lsim_submitted.txt"; fi; }
lsim() {   # lsim <job name> <names file> <par> <cpus> <time>
  local names
  names=$(grep -v '^#' "$HERE/$2" | while read -r n; do
    [ -f "$ROOT/$n/proj.tar.gz" ] || { echo "  missing $ROOT/$n/proj.tar.gz" >&2; continue; }
    { [ "${ALL:-0}" = 1 ] || [ ! -f "$ROOT/$n/lsim_v2023.json" ]; } && echo "$n" || true; done | paste -sd:)
  echo "$1: $(tr ':' '\n' <<< "$names" | grep -c .) projects" >&2
  [ -n "$names" ] || return 0
  sub -J "$1" --cpus-per-task="$4" --mem=64G -t "$5" -o "$LS/logs/lsim-%j.out" -e "$LS/logs/lsim-%j.err" \
      --export=ALL,LSIM_HERE="$LS",LSIM_T="$LSIM_T",ROOT="$ROOT",MODEL=v2023,PAR="$3",ONLY="$names" \
      "$HERE/sbatch_lsim_otus.sh"
}
if [ "$MODE" = --check ]; then
  n=$(for x in $(grep -v '^#' "$HERE/names_lsim_soda2.txt"); do [ -f "$ROOT/$x/lsim_v2023.json" ] && echo "$x" || true; done | head -1)
  [ -n "$n" ] || { echo "no finished Fir project to compare against"; exit 1; }
  echo "check project: $n (Fir result: $(python3 -c "import json;print(json.load(open('$ROOT/$n/lsim_v2023.json'))['end'])"))"
  echo "when done: python3 -c \"import json;print(json.load(open('$ROOT/$n/lsim_otuscheck.json'))['end'])\"  # must be equal"
  sub -J lsim_otuscheck --cpus-per-task=4 --mem=32G -t 2:00:00 -o "$LS/logs/lsim-%j.out" -e "$LS/logs/lsim-%j.err" \
      --export=ALL,LSIM_HERE="$LS",LSIM_T="$LSIM_T",ROOT="$ROOT",MODEL=v2023,TAG=otuscheck,PAR=1,ONLY="$n" \
      "$HERE/sbatch_lsim_otus.sh"
  exit 0
fi
lsim lsim_soda2    names_lsim_soda2.txt    12 16 24:00:00
lsim lsim_soda_pat names_lsim_soda_pat.txt  8  8 12:00:00
P=$CC/study/soda/lsim; mkdir -p "$P/logs"
if [ -f "$ROOT/contrast_u16_cr/proj.tar.gz" ]; then echo "lsim_proj: $ROOT/contrast_u16_cr/proj.tar.gz exists, skipped"
else sub -J lsim_proj -t 24:00:00 --mem=192G -o "$P/logs/proj_contrast_u16_cr-%j.out" \
         "$P/job_proj_otus.sh" "$CC/study/soda/dse/designs/contrast_u16_cr" "$ROOT/contrast_u16_cr"; fi
