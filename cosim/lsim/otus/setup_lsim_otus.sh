#!/bin/bash
# setup_lsim_otus.sh : rebuild LightningSim v0.2.6 with the vitis2023-axi fork (the Vitis HLS 2023.2 m_axi model) on Otus.
# Run once on a login node (needs internet: micromamba, conda-forge, sharc-lab channel, GitHub, crates.io).
#   LSIM_T=<install dir> bash setup_lsim_otus.sh      (default /scratch/hpc-prf-llmfpga/asa582/projects/tools/lightningsim)
# Same layout as Fir's /scratch/asa582/tools/lightningsim: bin/micromamba, env (exact package list from Fir),
# rustenv (rust 1.85), src (upstream clone + the 3 fork commits), cargo. Checks that the source tree is Fir's tree.
# No llvm-project submodule (119K files): the conda lightningsim package ships its LLVM; only the Rust core is rebuilt.
set -euo pipefail
HERE=$(dirname "$(readlink -f "$0")")
T=${LSIM_T:-/scratch/hpc-prf-llmfpga/asa582/projects/tools/lightningsim}
mkdir -p "$T/bin" "$T/tmp"
export MAMBA_ROOT_PREFIX=$T/mamba
[ -x "$T/bin/micromamba" ] || curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C "$T" bin/micromamba
[ -d "$T/env" ]     || "$T/bin/micromamba" create -y -p "$T/env"     --file "$HERE/lsim_env_explicit.txt"
[ -d "$T/rustenv" ] || "$T/bin/micromamba" create -y -p "$T/rustenv" --file "$HERE/rust_env_explicit.txt"
if [ ! -d "$T/src/.git" ]; then
  git clone --filter=blob:none https://github.com/sharc-lab/LightningSim.git "$T/src"
  git -C "$T/src" checkout -B vitis2023-axi 1abd9a6b679e07fceff79b6617eb993c465be6fd
  git -C "$T/src" -c user.name="$USER" -c user.email="$USER@localhost" am "$HERE"/fork_patches/*.patch
fi
TREE=$(git -C "$T/src" rev-parse 'HEAD^{tree}')
[ "$TREE" = "$(cat "$HERE/fork_tree.txt")" ] || { echo "fork tree $TREE differs from Fir's $(cat "$HERE/fork_tree.txt")"; exit 1; }
echo "fork source matches Fir (tree $TREE)"
# build_lsim.sh with T and the cargo target dir moved under $T
export PATH="$T/rustenv/bin:$T/env/bin:$PATH"
export CARGO_HOME="$T/cargo" CARGO_TARGET_DIR="$T/cargo-target" TMPDIR="$T/tmp"
unset PYTHONPATH PIP_CONFIG_FILE
cd "$T/src/backend"
sed -i "s#^CONDA_PREFIX = Path(r\"\"\"[^\"]*\"\"\")#CONDA_PREFIX = Path(r\"\"\"$T/env\"\"\")#" lightningsim/runner.py
"$T/env/bin/python" -m pip install --no-deps --no-build-isolation --ignore-installed . 2>&1 | tail -n 3
sed -i "s#^CONDA_PREFIX = Path(r\"\"\"$T/env\"\"\")#CONDA_PREFIX = Path(r\"\"\"/opt/anaconda1anaconda2anaconda3\"\"\")#" lightningsim/runner.py
cd / && "$T/env/bin/python" -c "import lightningsim._core as c, lightningsim.runner as r; print('installed', c.__file__, r.CONDA_PREFIX)"
