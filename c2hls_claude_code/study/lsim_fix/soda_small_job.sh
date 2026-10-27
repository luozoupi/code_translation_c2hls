#!/bin/bash
# soda_small_job.sh : small SODA designs (gen_small.py) -> csynth + cosim (trace_level port, SODA host as TB), a VCD of
# every signal under the DUT, and LightningSim on the same project. Like fx_job.sh, for SODA's testbench.
#   sbatch --export=ALL,ONLY=designs/blur_u8:designs_patched/heat3d_u2_stp,PAR=8,TAG=<lsim tag> soda_small_job.sh
# OUT/<set>__<name>/ gets csynth.rpt *cosim.rpt vitis_grep.log all.vcd.gz proj.tar.gz lsim_TAG.{log,json} frp.txt
#SBATCH --job-name=lsim_ss
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
export LSIM_T=${LSIM_T:-/scratch/hpc-prf-llmfpga/asa582/projects/tools/lightningsim}
module load lang Python/3.11.5-GCCcore-13.2.0 2>/dev/null || true
ulimit -c 0
set +u; source /scratch/hpc-prf-llmfpga/asa582/projects/workspaces/c2hls_claude_code/otus/vitis_settings_otus.sh >/dev/null; set -u
OUT=${OUT:-/scratch/hpc-prf-llmfpga/asa582/projects/tmp/c2hls/lsim/soda_small}; PAR=${PAR:-8}; TAG=${TAG:-ss}; VCD=${VCD:-1}
export OUT TAG VCD HERE LS
L=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}
W="$L/ss"; mkdir -p "$W"; export W
[ -n "${SLURM_TMPDIR:-}" ] || trap 'rm -rf "$L"' EXIT
tr ':' '\n' <<< "${ONLY:?}" > "$W/names.txt"

one() {
  local rel=$1 D="$HERE/soda_small/$1" n="${1//\//__}"
  local d="$W/$n" o="$OUT/$n"
  mkdir -p "$o" "$d/home/.Xilinx" "$d/tmp"
  cp "$D"/{kernel.cpp,host.cpp,frt.h,config.json} "$d/"
  local top argv
  top=$(python3 -c "import json;print(json.load(open('$D/config.json'))['top'])")
  argv=$(python3 -c "import json;print(json.load(open('$D/config.json'))['argv'])")
  cat > "$d/run.tcl" <<TCL
open_project hls_proj
set_top $top
add_files kernel.cpp -cflags "-std=c++14"
add_files -tb host.cpp -cflags "-std=c++14 -DSODA_TEST_MAIN -DSODA_CPP_BINDING -I$d"
open_solution "sol1" -flow_target vitis
set_part {xcu280-fsvh2892-2L-e}
create_clock -period 3.33 -name default
config_compile -name_max_length 80
csynth_design
cosim_design -trace_level port -argv "$argv"
exit
TCL
  ( cd "$d" && HOME="$d/home" TMPDIR="$d/tmp" vitis-run --tcl --input_file run.tcl ) > "$d/vitis.log" 2>&1
  cp "$d"/{kernel.cpp,config.json} "$o/"
  cp "$d"/hls_proj/sol1/syn/report/csynth.rpt "$o/" 2>/dev/null
  cp "$d"/hls_proj/sol1/sim/report/*cosim.rpt "$o/" 2>/dev/null
  grep -E "INFO: (PASS|FAIL)|ERROR|CRITICAL|DEADLOCK|Verilog\|" "$d/vitis.log" | head -n 40 > "$o/vitis_grep.log"
  tail -n 80 "$d/vitis.log" > "$o/vitis_tail.log"
  grep -l "frp_pipeline_valid" "$d"/hls_proj/sol1/syn/verilog/*.v 2>/dev/null | xargs -r -n1 basename > "$o/frp.txt"
  local v="$d/hls_proj/sol1/sim/verilog"
  if [ "$VCD" = 1 ] && [ -f "$v/run_xsim.sh" ]; then
    local snap dut tb inst
    dut=$(sed -n 's/^`define AUTOTB_DUT_INST //p' "$v"/*.autotb.v | head -1)
    snap=$(grep "xelab" "$v/run_xsim.sh" | sed -n "s/.* -s \([A-Za-z0-9_]*\).*/\1/p" | head -1)
    tb=$(ls "$v"/*.tcl | grep -v -e check_sim -e dataflow_monitor -e run_sim -e vcd | head -1)
    inst="/apatb_${snap}_top/${dut}"
    sed "s#^run all\$#open_vcd all.vcd\nlog_vcd [get_objects -r ${inst}/*]\nrun all\nclose_vcd#" "$tb" > "$v/vcd.tcl"
    ( cd "$v" && HOME="$d/home" xsim --noieeewarnings "$snap" -tclbatch vcd.tcl ) > "$d/vcd_xsim.log" 2>&1
    [ -s "$v/all.vcd" ] && gzip -c "$v/all.vcd" > "$o/all.vcd.gz" && rm -f "$v/all.vcd"
    tail -n 5 "$d/vcd_xsim.log" > "$o/vcd_xsim_tail.log"
  fi
  # LightningSim reads argv from hls.app and compiles the TB from sol1 (see study/soda/lsim/job_proj.sh)
  python3 - "$d/hls_proj/hls.app" "$argv" "$d" <<'PY'
import sys
p, argv, d = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(p).read()
if "<Simulation" not in s:
    s = s.replace("</AutoPilot:project>", f'    <Simulation argv="{argv}"/>\n</AutoPilot:project>')
s = s.replace("-I../../../../.", "-I../..").replace(f"-I{d}", "-I../..")   # cosim needs the absolute path
open(p, "w").write(s)
PY
  rm -rf "$v/xsim.dir" "$v"/*.wdb
  tar -C "$d" --exclude='hls_proj/sol1/sim' -czf "$o/proj.tar.gz" hls_proj kernel.cpp host.cpp frt.h
  rm -rf "$d/hls_proj/sol1/sim"
  mkdir -p "$d/ltmp"
  ( cd "$d" && LSIM_TMPDIR="$d/ltmp" bash "$LS/otus/run_lsim_otus.sh" "$d/hls_proj/sol1" ) > "$o/lsim_$TAG.log" 2>&1
  local rc=$?
  python3 - "$o" "$TAG" "$rc" <<'EOF'
import json, math, re, sys
from pathlib import Path
o, tag, rc = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
log = (o / f"lsim_{tag}.log").read_text(errors="replace")
m = re.search(r"^\[\s*(\d+)-\s*(\d+)\]\s+(\S+)", log, re.M)
e = int(m.group(2)) if m else None
n = len(re.findall(r"#pragma HLS interface s_axilite port = (?!return)\w+", (o / "kernel.cpp").read_text()))
P, K = 6 + n, 27 + 12 * n
mp = P * math.ceil((e - 14) / P) + K if e is not None else None
c = None
for f in o.glob("*cosim.rpt"):
    r = re.search(r"\|\s*Verilog\|\s*(\w+)\|\s*(\d+)\|\s*(\d+)\|\s*(\d+)\|", f.read_text(errors="replace"))
    if r and r.group(1) == "Pass": c = int(r.group(4))
json.dump({"name": o.name, "rc": rc, "end": e, "mapped": mp, "cosim": c, "deadlock": "Deadlock detected" in log,
           "err_pct": round(100 * (mp - c) / c, 4) if c and mp else None, "diff": (mp - c) if c and mp else None,
           "frp": (o / "frp.txt").read_text().split()}, open(o / f"lsim_{tag}.json", "w"), indent=1)
EOF
  rm -rf "$d"
}
export -f one
xargs -a "$W/names.txt" -P "$PAR" -I{} bash -c 'one {}'
echo "done $(wc -l < "$W/names.txt")"
