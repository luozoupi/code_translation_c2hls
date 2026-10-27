#!/bin/bash
# eval_job.sh : frp_eval.py (FRP rule variants) on finished projects, one node.
#   sbatch --export=ALL,DIRS=<dir>:<dir>...,TAG=<lsim json tag holding the cosim>,PAR=8 eval_job.sh
#   PYSCRIPT=fifo_cmp.py OUTNAME=fifo_cmp.json runs the per-FIFO LSIM-vs-VCD comparison instead (same argv).
# Each DIR holds proj.tar.gz (+ tb.cpp/kernel.cpp for fx_gen designs) and lsim_TAG.json with "cosim"; the result is
# DIR/frp_eval.json. NPTR: 2 for fx_gen designs, else counted from kernel.cpp's s_axilite pragmas (SODA).
#SBATCH --job-name=lsim_eval
#SBATCH --account=hpc-prf-llmfpga
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=0-12:00:00
set -uo pipefail
HERE=/scratch/hpc-prf-llmfpga/asa582/projects/workspaces/c2hls_claude_code/study/lsim_fix
LS=/scratch/hpc-prf-llmfpga/asa582/projects/workspaces/c2hls_claude_code/cosim/lsim
T=${LSIM_T:-/scratch/hpc-prf-llmfpga/asa582/projects/tools/lightningsim}
ulimit -c 0
set +u; source /scratch/hpc-prf-llmfpga/asa582/projects/workspaces/c2hls_claude_code/otus/vitis_settings_otus.sh >/dev/null; set -u
L=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}
W="$L/ev"; mkdir -p "$W"
[ -n "${SLURM_TMPDIR:-}" ] || trap 'rm -rf "$L"' EXIT
export PYSCRIPT=${PYSCRIPT:-frp_eval.py} W HERE LS T TAG=${TAG:?} OUTNAME=${OUTNAME:-frp_eval.json}
tr ':' '\n' <<< "${DIRS:?}" > "$W/dirs.txt"
one() {
  local o=$1 d="$W/$(basename "$1")"
  mkdir -p "$d/tmp" && tar -C "$d" -xzf "$o/proj.tar.gz"
  [ -f "$o/tb.cpp" ] && cp "$o"/{tb.cpp,kernel.cpp} "$d/"
  local nptr cos
  if [ -f "$o/tb.cpp" ]; then nptr=2; else
    nptr=$(grep -E "#pragma HLS interface s_axilite port = " "$d/kernel.cpp" | grep -vc "port = return"); fi
  cos=$(python3 -c "import json;print(json.load(open('$o/lsim_$TAG.json')).get('cosim'))" 2>/dev/null)
  python3 "$LS/axi_cfg.py" "$d/hls_proj/sol1" "$d/tmp/axi.json" > /dev/null 2>&1
  ( cd / && env -u LD_LIBRARY_PATH -u PYTHONPATH PATH="$T/env/bin:$PATH" TMPDIR="$d/tmp" LSIM_AXI_MODEL=v2023 \
      LSIM_AXI_IFCFG="$d/tmp/axi.json" LDFLAGS="-Wl,-rpath,$T/env/lib" \
      "$T/env/bin/python" "$HERE/${PYSCRIPT:-frp_eval.py}" "$d/hls_proj/sol1" "$o/$OUTNAME" "$nptr" "$cos" ) > "$o/${OUTNAME%.json}.log" 2>&1
  tail -n 1 "$o/${OUTNAME%.json}.log"
  rm -rf "$d"
}
export -f one
xargs -a "$W/dirs.txt" -P "${PAR:-8}" -I{} bash -c 'one {}'
echo done
