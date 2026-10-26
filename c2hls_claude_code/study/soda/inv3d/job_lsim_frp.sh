#!/bin/bash
# job_lsim_frp.sh <proj.tar.gz> <out.json> : lsim_frp.py on one packed SODA project (job_proj_otus.sh layout).
# Environment as cosim/lsim/otus/run_lsim_otus.sh (native Vitis 2023.2, LightningSim fork in LSIM_T).
#SBATCH --account=hpc-prf-llmfpga
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=06:00:00
set -eo pipefail
TGZ=$(readlink -f $1); OUT=$(readlink -f -m $2)
HERE=/scratch/hpc-prf-llmfpga/asa582/projects/workspaces/c2hls_claude_code/study/soda/inv3d
LS=/scratch/hpc-prf-llmfpga/asa582/projects/workspaces/c2hls_claude_code/cosim/lsim
LSIM_T=${LSIM_T:-/scratch/hpc-prf-llmfpga/asa582/projects/tools/lightningsim}; ENV=$LSIM_T/env
ulimit -c 0
source ${OTUS_VITIS_SETTINGS:-/scratch/hpc-prf-llmfpga/asa582/projects/workspaces/c2hls_claude_code/otus/vitis_settings_otus.sh} >/dev/null
L=${SLURM_TMPDIR:-${OTUS_LOCAL_TMP:-/tmp}/$USER/$SLURM_JOB_ID}; [ -n "${SLURM_TMPDIR:-}" ] || trap 'rm -rf "$L"' EXIT
W=$L/frp; mkdir -p $W/tmp; tar -C $W -xzf $TGZ
export TMPDIR=$W/tmp LSIM_AXI_MODEL=v2023
python3 $LS/axi_cfg.py $W/hls_proj/sol1 $TMPDIR/axi_cfg.json >&2; export LSIM_AXI_IFCFG=$TMPDIR/axi_cfg.json
export PATH="$ENV/bin:$PATH"; unset LD_LIBRARY_PATH PYTHONPATH; export LDFLAGS="-Wl,-rpath,$ENV/lib"
cd / && python $HERE/lsim_frp.py $W/hls_proj/sol1 $OUT
