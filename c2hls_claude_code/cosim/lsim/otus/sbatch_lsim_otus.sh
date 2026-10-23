#!/bin/bash
# LightningSim over every ROOT/*/proj.tar.gz (or a ':'-separated ONLY list), one node.
#   sbatch --export=ALL,ROOT=/scratch/hpc-prf-llmfpga/asa582/projects/tmp/c2hls/lsim/mb,MODEL=v2023,PAR=8 sbatch_lsim.sh
# Each project is unpacked in $SLURM_TMPDIR; results go to ROOT/NAME/lsim_MODEL.{json,log}.
# Otus copy of ../sbatch_lsim.sh: native Vitis and LightningSim (LSIM_T), no container; node-local dir falls back when
# $SLURM_TMPDIR is unset; LSIM_HERE (cosim/lsim) and the output path come from resubmit_lsim_otus.sh.
#SBATCH --job-name=lsim_run
#SBATCH --account=hpc-prf-llmfpga
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=0-06:00:00
set -uo pipefail
HERE=${LSIM_HERE:?}
: "${LSIM_T:?}"
module load lang Python/3.11.5-GCCcore-13.2.0 2>/dev/null || true
PY=python3
ulimit -c 0
set +u; source ${OTUS_VITIS_SETTINGS:-/scratch/hpc-prf-llmfpga/asa582/projects/workspaces/c2hls_claude_code/otus/vitis_settings_otus.sh} >/dev/null; set -u
export LSIM_T PY
set +e
ROOT=${ROOT:?}; MODEL=${MODEL:-v2023}; PAR=${PAR:-8}; TAG=${TAG:-$MODEL}
export ROOT MODEL TAG HERE LSIM_AXI_MODEL=$MODEL
L=${SLURM_TMPDIR:-${OTUS_LOCAL_TMP:-/tmp}/$USER/$SLURM_JOB_ID}
W="$L/lsim"; mkdir -p "$W"; export W
[ -n "${SLURM_TMPDIR:-}" ] || trap 'rm -rf "$L"' EXIT
if [ -n "${ONLY:-}" ]; then tr ':' '\n' <<< "$ONLY" > "$W/names.txt"
else for t in "$ROOT"/*/proj.tar.gz; do basename "$(dirname "$t")"; done > "$W/names.txt"; fi

one() {
  local n=$1 d="$W/$1" o="$ROOT/$1" t0 t1 rc
  mkdir -p "$d/tmp" && tar -C "$d" -xzf "$o/proj.tar.gz"
  t0=$(date +%s.%N)
  if [ "${IO:-0}" = 1 ]; then
    LSIM_TMPDIR="$d/tmp" LSIM_IO_OUT="$o/lsim_io_$TAG.json" bash "$HERE/otus/run_lsim_otus.sh" "$d/hls_proj/sol1" > "$o/lsim_io_$TAG.log" 2>&1
  fi
  LSIM_TMPDIR="$d/tmp" bash "$HERE/otus/run_lsim_otus.sh" "$d/hls_proj/sol1" > "$o/lsim_$TAG.log" 2>&1
  rc=$?
  t1=$(date +%s.%N)
  "$PY" - "$o/lsim_$TAG.log" "$o/lsim_$TAG.json" "$n" "$MODEL" "$rc" "$t0" "$t1" <<'EOF'
import json, re, sys
log, out, name, model, rc, t0, t1 = sys.argv[1:]
m = re.search(r"^\[\s*(\d+)-\s*(\d+)\]\s+(\S+)", open(log, errors="replace").read(), re.M)
json.dump({"name": name, "model": model, "rc": int(rc), "start": int(m.group(1)) if m else None,
           "end": int(m.group(2)) if m else None, "top": m.group(3) if m else None,
           "wall_s": round(float(t1) - float(t0), 2)}, open(out, "w"), indent=1)
EOF
  rm -rf "$d"
}
export -f one
xargs -a "$W/names.txt" -P "$PAR" -I{} bash -c 'one {}'
echo "done $(wc -l < "$W/names.txt")"
