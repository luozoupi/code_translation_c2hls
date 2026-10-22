#!/bin/bash
# LightningSim over every ROOT/*/proj.tar.gz (or a ':'-separated ONLY list), one node.
#   sbatch --export=ALL,ROOT=/scratch/asa582/tmp/c2hls/lsim/mb,MODEL=v2023,PAR=8 sbatch_lsim.sh
# Each project is unpacked in $SLURM_TMPDIR; results go to ROOT/NAME/lsim_MODEL.{json,log}.
#SBATCH --job-name=lsim_run
#SBATCH --account=def-zhenman
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=0-06:00:00
#SBATCH --output=/scratch/asa582/workspaces/c2hls_claude_code/cosim/lsim/logs/lsim-%j.out
#SBATCH --error=/scratch/asa582/workspaces/c2hls_claude_code/cosim/lsim/logs/lsim-%j.err
set -uo pipefail
HERE=/scratch/asa582/workspaces/c2hls_claude_code/cosim/lsim
source /scratch/asa582/workspaces/dse_v3_claude/env.sh
ulimit -c 0
export APPTAINER_BINDPATH="/scratch/${USER},${HOME}/.Xilinx,${SLURM_TMPDIR}"
dse_v3_source_vitis >/dev/null
set +e
ROOT=${ROOT:?}; MODEL=${MODEL:-v2023}; PAR=${PAR:-8}; TAG=${TAG:-$MODEL}
export ROOT MODEL TAG HERE LSIM_AXI_MODEL=$MODEL
W="$SLURM_TMPDIR/lsim"; mkdir -p "$W"; export W
if [ -n "${ONLY:-}" ]; then tr ':' '\n' <<< "$ONLY" > "$W/names.txt"
else for t in "$ROOT"/*/proj.tar.gz; do basename "$(dirname "$t")"; done > "$W/names.txt"; fi

one() {
  local n=$1 d="$W/$1" o="$ROOT/$1" t0 t1 rc
  mkdir -p "$d/tmp" && tar -C "$d" -xzf "$o/proj.tar.gz"
  t0=$(date +%s.%N)
  if [ "${IO:-0}" = 1 ]; then
    LSIM_TMPDIR="$d/tmp" LSIM_IO_OUT="$o/lsim_io_$TAG.json" bash "$HERE/run_lsim.sh" "$d/hls_proj/sol1" > "$o/lsim_io_$TAG.log" 2>&1
  fi
  LSIM_TMPDIR="$d/tmp" bash "$HERE/run_lsim.sh" "$d/hls_proj/sol1" > "$o/lsim_$TAG.log" 2>&1
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
