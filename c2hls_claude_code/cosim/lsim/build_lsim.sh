#!/bin/bash
# Rebuild the LightningSim Rust core from the fork and install it into the env (no LLVM rebuild).
#   build_lsim.sh
set -euo pipefail
T=/scratch/asa582/tools/lightningsim
export PATH="$T/rustenv/bin:$T/env/bin:$PATH"
export CARGO_HOME="$T/cargo" CARGO_TARGET_DIR="/scratch/hpc-prf-llmfpga/asa582/projects/tmp/c2hls/lsim/cargo-target"
export TMPDIR=/scratch/hpc-prf-llmfpga/asa582/projects/tmp/c2hls/lsim/tmp
unset PYTHONPATH PIP_CONFIG_FILE
mkdir -p "$TMPDIR"
cd "$T/src/backend"
# The conda build of the package patches CONDA_PREFIX into runner.py; the fork's source has the placeholder.
sed -i "s#^CONDA_PREFIX = Path(r\"\"\"[^\"]*\"\"\")#CONDA_PREFIX = Path(r\"\"\"$T/env\"\"\")#" lightningsim/runner.py
"$T/env/bin/python" -m pip install --no-deps --no-build-isolation --ignore-installed . 2>&1 | tail -n 3
sed -i "s#^CONDA_PREFIX = Path(r\"\"\"$T/env\"\"\")#CONDA_PREFIX = Path(r\"\"\"/opt/anaconda1anaconda2anaconda3\"\"\")#" lightningsim/runner.py
cd / && "$T/env/bin/python" -c "import lightningsim._core as c, lightningsim.runner as r; print('installed', c.__file__, r.CONDA_PREFIX)"
