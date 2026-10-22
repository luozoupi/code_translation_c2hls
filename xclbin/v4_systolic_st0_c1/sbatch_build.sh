#!/bin/bash
# Build the U280 xclbin for autosa_mm (v4_systolic_st0_c1). Keeps only the xclbin, its info, reports and gzipped logs.
#SBATCH --job-name=xclbin_v4_systolic_st0_c1
#SBATCH --account=def-zhenman
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=448G
#SBATCH --time=5-00:00:00
#SBATCH --chdir=/scratch/asa582/workspaces/c2hls_claude_code/xclbin/v4_systolic_st0_c1
#SBATCH --output=/scratch/asa582/workspaces/c2hls_claude_code/xclbin/v4_systolic_st0_c1/logs/build-%j.out
#SBATCH --error=/scratch/asa582/workspaces/c2hls_claude_code/xclbin/v4_systolic_st0_c1/logs/build-%j.err
set -euo pipefail
B=/scratch/asa582/workspaces/c2hls_claude_code/xclbin/v4_systolic_st0_c1
source /scratch/asa582/workspaces/dse_v3_claude/env.sh
ulimit -c 0  # no core dumps
export APPTAINER_BINDPATH="/scratch/${USER},${HOME}/.Xilinx,${SLURM_TMPDIR}"
dse_v3_source_vitis
PLAT=/opt/xilinx/platforms/xilinx_u280_gen3x16_xdma_1_202211_1/xilinx_u280_gen3x16_xdma_1_202211_1.xpfm
# Everything bulky (HLS project, Vivado project, all four implementation runs) goes to a temp dir that is deleted.
T=${SLURM_TMPDIR}/xclbin_v4_systolic_st0_c1   # node-local disk: Slurm deletes it when the job ends, even if killed
mkdir -p "$T" "$B/out/reports" "$B/out/logs"
cleanup() {
  # keep: reports v++ copied, the timing/utilization reports of every implementation run, gzipped logs; then delete.
  find "$T" \( -name "*timing_summary_routed.rpt" -o -name "*utilization_placed.rpt" -o -name "*utilization_routed.rpt" \
       -o -name "*route_status.rpt" -o -name "*slr_util*.rpt" -o -name "*kernel_util*.rpt" -o -name "*csynth.rpt" \
       -o -name "*.link_summary" -o -name "*.compile_summary" -o -name "system_estimate*" \) -size -20M 2>/dev/null |
    while read -r f; do d="$B/out/reports/$(dirname "${f#$T/}")"; mkdir -p "$d"; cp "$f" "$d/"; done
  for f in $(find "$T" -maxdepth 6 \( -name "v++*.log" -o -name "vivado.log" -o -name "runme.log" \) 2>/dev/null); do
    d="$B/out/logs/$(dirname "${f#$T/}")"; mkdir -p "$d"; gzip -c "$f" > "$d/$(basename "$f").gz"; done
  rm -rf "$T"
}
trap cleanup EXIT
run() { apptainer exec "$XILINX_SIF" bash -lc "export HOME=$T LIBRARY_PATH=/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu; ulimit -s unlimited || true; cd $T && $*"; }
cp "$B/kernel.cpp" "$B/kernel.h" "$B/link.cfg" "$T/"
date; echo "== compile (HLS)"
if [ -f "$B/out/kernel0.xo" ]; then echo "reusing $B/out/kernel0.xo"; cp "$B/out/kernel0.xo" "$T/"
else
  run "v++ -c -t hw --platform $PLAT -k kernel0 -I. kernel.cpp -o kernel0.xo --temp_dir $T/_x_c --log_dir $T/logs_c --report_dir $T/reports_c"
  cp "$T/kernel0.xo" "$B/out/"   # keep the compiled kernel even if the link fails
fi
date; echo "== link (synth + 4 parallel implementation strategies)"
run "v++ -l -t hw --platform $PLAT --config link.cfg kernel0.xo -o kernel0.xclbin --temp_dir $T/_x_l --log_dir $T/logs_l --report_dir $T/reports_l"
date; echo "== package"
cp "$T/kernel0.xclbin" "$T/kernel0.xo" "$B/out/"
run "xclbinutil --info --input kernel0.xclbin" > "$B/out/kernel0.xclbin.info"
grep -iE "Frequency|clock" "$B/out/kernel0.xclbin.info" | head
date; echo "done"
