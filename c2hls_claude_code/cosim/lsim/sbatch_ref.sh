#!/bin/bash
# Vitis 2023.2 csynth + cosim of the m_axi microbenchmarks (reference cycles), one node.
#   sbatch --export=ALL,OUT=/scratch/hpc-prf-llmfpga/asa582/projects/tmp/c2hls/lsim/mb,ONLY=rd_n1024:wr_n1024,TRACE=port,PAR=12 sbatch_ref.sh
# Each config runs in $SLURM_TMPDIR; the whole project is tarred to $OUT/NAME/proj.tar.gz (one file, for
# LightningSim later), and the cosim report, the AESL AXI memory models and a log tail are kept unpacked.
#SBATCH --job-name=lsim_ref
#SBATCH --account=hpc-prf-llmfpga
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=0-12:00:00
#SBATCH --output=/scratch/hpc-prf-llmfpga/asa582/projects/workspaces/c2hls_claude_code/cosim/lsim/logs/ref-%j.out
#SBATCH --error=/scratch/hpc-prf-llmfpga/asa582/projects/workspaces/c2hls_claude_code/cosim/lsim/logs/ref-%j.err
set -uo pipefail
HERE=/scratch/hpc-prf-llmfpga/asa582/projects/workspaces/c2hls_claude_code/cosim/lsim
source /scratch/hpc-prf-llmfpga/asa582/projects/workspaces/dse_v3_claude/env.sh
ulimit -c 0
export APPTAINER_BINDPATH="/scratch/${USER},${HOME}/.Xilinx,${SLURM_TMPDIR}"
dse_v3_source_vitis >/dev/null
set +e
OUT=${OUT:?}; PAR=${PAR:-12}; TRACE=${TRACE:-none}
mkdir -p "$OUT"
W="$SLURM_TMPDIR/mb"
mkdir -p "$W/home/.Xilinx" "$W/tmp"
"$PY" "$HERE/mb_gen.py" "$W/cfg" ${ONLY:+--only "$ONLY"} --trace "$TRACE" > "$W/names.txt"

run_one() {
  local name=$1 d="$W/cfg/$1" o="$OUT/$1" t0 t1
  mkdir -p "$o"
  t0=$(date +%s)
  ( cd "$d" && HOME="$W/home" TMPDIR="$W/tmp" vitis-run --tcl --input_file run.tcl ) > "$d/vitis.log" 2>&1
  local rc=$?
  t1=$(date +%s)
  if [ "${VCD:-0}" = 1 ]; then bash "$HERE/vcd_rerun.sh" "$d" >> "$d/vitis.log" 2>&1; fi
  cp "$d"/{kernel.cpp,tb.cpp,run.tcl,config.json} "$o/"
  tail -n 80 "$d/vitis.log" > "$o/vitis_tail.log"
  grep -E "Passed!|Failed with|ERROR|CRITICAL" "$d/vitis.log" | head -n 40 > "$o/vitis_grep.log"
  cp "$d"/hls_proj/sol1/sim/report/*cosim.rpt "$o/" 2>/dev/null
  cp "$d"/hls_proj/sol1/syn/report/csynth.rpt "$o/" 2>/dev/null
  cp "$d"/hls_proj/sol1/sim/verilog/*.result.lat.rb "$o/" 2>/dev/null
  cp "$d"/axi.vcd.gz "$o/" 2>/dev/null
  mkdir -p "$o/aesl" && cp "$d"/hls_proj/sol1/sim/verilog/AESL_axi_*.v "$d"/hls_proj/sol1/sim/verilog/apatb_mb_top.v "$o/aesl/" 2>/dev/null
  tar -C "$d" -czf "$o/proj.tar.gz" --exclude='hls_proj/sol1/sim/verilog/xsim.dir' --exclude='*.wdb' --exclude='*.vcd*' hls_proj kernel.cpp tb.cpp
  printf '{"name":"%s","vitis_rc":%d,"wall_s":%d}\n' "$name" "$rc" "$((t1 - t0))" > "$o/run.json"
  rm -rf "$d/hls_proj"
}
export -f run_one
export W OUT HERE

xargs -a "$W/names.txt" -P "$PAR" -I{} bash -c 'run_one {}'
echo "done $(wc -l < "$W/names.txt") configs"
