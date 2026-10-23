#!/bin/bash
# Build the U280 xclbin for autosa_mm (v4_systolic_st0_c1). Keeps only the xclbin, its info, reports and gzipped logs.
# Otus copy of sbatch_build.sh: native Vitis 2023.2 (+ XRT for xclbinutil) instead of the Fir container; the U280 platform
# comes from OTUS_U280_XPFM or a search of the usual platform dirs; node-local dir falls back when $SLURM_TMPDIR is unset.
# Submitted by resubmit_xclbin_otus.sh (which sets --chdir, --output, --time).
#SBATCH --job-name=xclbin_v4_systolic_st0_c1
#SBATCH --account=hpc-prf-llmfpga
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=448G
#SBATCH --time=2-00:00:00
set -euo pipefail
B=${XCLBIN_DIR:-$PWD}   # resubmit_xclbin_otus.sh sets --chdir; SLURM_SUBMIT_DIR is wherever sbatch ran
[ -f "$B/link.cfg" ] || { echo "submit from xclbin/v4_systolic_st0_c1 (or set XCLBIN_DIR)"; exit 1; }
ulimit -c 0  # no core dumps
set +u
source "${OTUS_VITIS_SETTINGS:-/scratch/hpc-prf-llmfpga/asa582/projects/workspaces/c2hls_claude_code/otus/vitis_settings_otus.sh}" >/dev/null
XRT=${OTUS_XRT_SETUP:-/opt/xilinx/xrt/setup.sh}; [ -f "$XRT" ] && source "$XRT" >/dev/null || echo "no XRT setup at $XRT: xclbinutil may be missing"
set -u
command -v v++
XPFM=xilinx_u280_gen3x16_xdma_1_202211_1.xpfm
PLAT=${OTUS_U280_XPFM:-$(for d in ${PLATFORM_REPO_PATHS:-} /opt/xilinx/platforms /opt/software/FPGA/Xilinx/platforms /opt/software/FPGA/platforms; do
  [ -d "$d" ] && find "$d" -maxdepth 3 -name "$XPFM" 2>/dev/null || true; done | sed -n 1p)}   # || true: a missing last dir would end the script under pipefail
[ -n "$PLAT" ] && [ -f "$PLAT" ] || { echo "U280 platform $XPFM not found: set OTUS_U280_XPFM"; exit 1; }
echo "platform $PLAT"
# Everything bulky (HLS project, Vivado project, all four implementation runs) goes to a temp dir that is deleted.
T=${SLURM_TMPDIR:-${OTUS_LOCAL_TMP:-/tmp}/$USER/$SLURM_JOB_ID}/xclbin_v4_systolic_st0_c1   # node-local disk; cleanup() deletes it
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
run() { bash -c "export HOME=$T; ulimit -s unlimited || true; cd $T && $*"; }
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
grep -iE "Frequency|clock" "$B/out/kernel0.xclbin.info" | sed -n 1,10p || true
date; echo "done"
