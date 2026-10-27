#!/bin/bash
# fx_job.sh : Vitis 2023.2 csim + csynth + cosim (trace_level port) of fx_gen.py microbenchmarks on Otus, then a VCD of
# every signal under the DUT (xsim rerun of the cosim snapshot) and LightningSim on the same project.
#   sbatch --export=ALL,OUT=<dir>,ONLY=a:b,PAR=8,TAG=<lsim tag> fx_job.sh
# Per config, OUT/NAME/ gets: kernel.cpp tb.cpp config.json csynth.rpt *cosim.rpt vitis_grep.log vitis_tail.log
#   all.vcd.gz  proj.tar.gz  lsim_TAG.{log,json}  frp.txt (FRP instances in the RTL)
#SBATCH --job-name=lsim_fx
#SBATCH --account=hpc-prf-llmfpga
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=0-12:00:00
set -uo pipefail
HERE=/scratch/hpc-prf-llmfpga/asa582/projects/workspaces/c2hls_claude_code/study/lsim_fix
LS=/scratch/hpc-prf-llmfpga/asa582/projects/workspaces/c2hls_claude_code/cosim/lsim
export LSIM_T=${LSIM_T:-/scratch/hpc-prf-llmfpga/asa582/projects/tools/lightningsim}
module load lang Python/3.11.5-GCCcore-13.2.0 2>/dev/null || true
ulimit -c 0
set +u; source /scratch/hpc-prf-llmfpga/asa582/projects/workspaces/c2hls_claude_code/otus/vitis_settings_otus.sh >/dev/null; set -u
OUT=${OUT:?}; PAR=${PAR:-8}; TAG=${TAG:-fx}; VCD=${VCD:-1}
export OUT TAG VCD HERE LS
L=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}
W="$L/fx"; mkdir -p "$W"; export W
[ -n "${SLURM_TMPDIR:-}" ] || trap 'rm -rf "$L"' EXIT
python3 "$HERE/fx_gen.py" "$W/cfg" ${ONLY:+--only "$ONLY"} > "$W/names.txt"

one() {
  local n=$1 d="$W/cfg/$1" o="$OUT/$1"
  mkdir -p "$o" "$d/home/.Xilinx" "$d/tmp"
  ( cd "$d" && HOME="$d/home" TMPDIR="$d/tmp" vitis-run --tcl --input_file run.tcl ) > "$d/vitis.log" 2>&1
  cp "$d"/{kernel.cpp,tb.cpp,run.tcl,config.json} "$o/"
  cp "$d"/hls_proj/sol1/syn/report/csynth.rpt "$o/" 2>/dev/null
  cp "$d"/hls_proj/sol1/sim/report/*cosim.rpt "$o/" 2>/dev/null
  grep -E "Passed!|Failed with|ERROR|CRITICAL|DEADLOCK" "$d/vitis.log" | head -n 40 > "$o/vitis_grep.log"
  tail -n 80 "$d/vitis.log" > "$o/vitis_tail.log"
  grep -l "frp_pipeline_valid" "$d"/hls_proj/sol1/syn/verilog/*.v 2>/dev/null | xargs -r -n1 basename > "$o/frp.txt"
  local v="$d/hls_proj/sol1/sim/verilog"
  if [ "$VCD" = 1 ] && [ -f "$v/run_xsim.sh" ]; then
    local snap top tb inst
    top=$(sed -n 's/^`define AUTOTB_DUT_INST //p' "$v"/*.autotb.v | head -1)
    snap=$(grep "xelab" "$v/run_xsim.sh" | sed -n "s/.* -s \([A-Za-z0-9_]*\).*/\1/p" | head -1)
    tb=$(ls "$v"/*.tcl | grep -v -e check_sim -e dataflow_monitor -e run_sim -e vcd | head -1)
    inst="/apatb_${snap}_top/${top}"
    sed "s#^run all\$#open_vcd all.vcd\nlog_vcd [get_objects -r ${inst}/*]\nrun all\nclose_vcd#" "$tb" > "$v/vcd.tcl"
    ( cd "$v" && HOME="$d/home" xsim --noieeewarnings "$snap" -tclbatch vcd.tcl ) > "$d/vcd_xsim.log" 2>&1
    [ -s "$v/all.vcd" ] && gzip -c "$v/all.vcd" > "$o/all.vcd.gz" && rm -f "$v/all.vcd"
    tail -n 5 "$d/vcd_xsim.log" > "$o/vcd_xsim_tail.log"
  fi
  rm -rf "$d/hls_proj/sol1/sim/verilog/xsim.dir" "$d/hls_proj/sol1/sim/verilog/"*.wdb
  tar -C "$d" -czf "$o/proj.tar.gz" hls_proj
  mkdir -p "$d/ltmp"
  LSIM_TMPDIR="$d/ltmp" bash "$LS/otus/run_lsim_otus.sh" "$d/hls_proj/sol1" > "$o/lsim_$TAG.log" 2>&1
  local rc=$?
  python3 - "$o" "$TAG" "$rc" <<'EOF'
import json, math, re, sys
from pathlib import Path
o, tag, rc = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
m = re.search(r"^\[\s*(\d+)-\s*(\d+)\]\s+(\S+)", (o / f"lsim_{tag}.log").read_text(errors="replace"), re.M)
e = int(m.group(2)) if m else None
P, K = 8, 51                     # n = 2 pointer arguments
mp = P * math.ceil((e - 14) / P) + K if e is not None else None
c = None
for f in o.glob("*cosim.rpt"):
    r = re.search(r"\|\s*Verilog\|\s*(\w+)\|\s*(\d+)\|\s*(\d+)\|\s*(\d+)\|", f.read_text(errors="replace"))
    if r and r.group(1) == "Pass": c = int(r.group(4))
json.dump({"name": o.name, "rc": rc, "end": e, "mapped": mp, "cosim": c,
           "err_pct": round(100 * (mp - c) / c, 4) if c and mp else None, "diff": (mp - c) if c and mp else None,
           "frp": (o / "frp.txt").read_text().split()}, open(o / f"lsim_{tag}.json", "w"), indent=1)
EOF
  rm -rf "$d"
}
export -f one
xargs -a "$W/names.txt" -P "$PAR" -I{} bash -c 'one {}'
echo "done $(wc -l < "$W/names.txt")"
