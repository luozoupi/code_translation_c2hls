#!/bin/bash
# job_csynth_otus.sh <kernel dir> <task.json> : csynth of <kernel dir>/kernel.cpp on Otus (Vitis HLS 2023.2 natively,
# U280, 3.33 ns) through agentfw/tools/hls_eval_job_otus.py; writes csynth_summary.{txt,json} there. Same settings as the
# Fir post-hoc job.sh files (8 cpus, 64G, 12 h, timeouts 200000 s); submitted by resubmit_csynth_otus.sh.
#SBATCH --account=hpc-prf-llmfpga
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
set -eo pipefail
STUDY=${STUDY_DIR:?}
module load lang Python/3.11.5-GCCcore-13.2.0 2>/dev/null || true
export C2HLS_VITIS_SETTINGS=${OTUS_VITIS_SETTINGS:-/scratch/hpc-prf-llmfpga/asa582/projects/workspaces/c2hls_claude_code/otus/vitis_settings_otus.sh}
W=$(dirname "$(dirname "$STUDY")")                        # workspaces (study -> c2hls_claude_code -> ..)
[ -f "$W/hls_eval.py" ] && W=$(dirname "$W")   # combined branch: the study is a subfolder of the judge clone
B=$(dirname "$W")                                          # projects root
export C2HLS_TMP_ROOT=$B/tmp/c2hls; mkdir -p "$C2HLS_TMP_ROOT"
unset C2HLS_USE_CONTAINER XILINX_SIF C2HLS_DSE_V3 C2HLS_DSE_V3_HARNESS C2HLS_DSE_V4 C2HLS_DSE_V2_GRID C2HLS_SITE C2HLS_VITIS_JOBS C2HLS_VITIS_JOBS_FROM_SLURM DSE_V3_TRIAL_ROOT
export C2HLS_CSIM_TIMEOUT=200000 C2HLS_SYNTH_TIMEOUT=200000
python3 "$STUDY/agentfw/tools/hls_eval_job_otus.py" csynth "$1" "$2"
