#!/bin/bash
# Build Verilator v5.044 (the version TAPA pins; >= 5.044 is needed for the Vitis HLS m_axi master) into /scratch/asa582/tools.
#SBATCH --job-name=build_verilator
#SBATCH --account=hpc-prf-llmfpga
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/hpc-prf-llmfpga/asa582/projects/workspaces/c2hls_claude_code/cosim/verilator/logs/build_verilator-%j.out
set -euo pipefail
ulimit -c 0
VER=v5.044
PREFIX=/scratch/asa582/tools/verilator-$VER
SRC=$SLURM_TMPDIR/verilator-src     # node-local; Slurm deletes it
git clone -q --depth 1 --branch $VER https://github.com/verilator/verilator.git "$SRC"
cd "$SRC"
unset VERILATOR_ROOT
autoconf
./configure --prefix="$PREFIX"
make -j"$SLURM_CPUS_PER_TASK"
make install
"$PREFIX/bin/verilator" --version
echo "installed: $PREFIX"
