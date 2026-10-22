#!/usr/bin/env bash
# Generate an AutoSA LU design (no Slurm, ~5 s, login-node safe) inside the AutoSA container.
# usage: gen_lu.sh <N> <space_time 0..5> <outdir>   e.g. gen_lu.sh 32 3 $PWD/out_n32_st3
set -euo pipefail
N=${1:?N}; ST=${2:?space_time}; OUT=$(readlink -f "${3:?outdir}")
SIF=${AUTOSA_SIF:-/scratch/asa582/workspaces/AutoSA/autosa_docker.sif}
SRC=$(cd "$(dirname "$0")" && pwd)/upstream/AutoSA/autosa_tests/lu   # identical to container copy (rev b61a1b4)
module load apptainer/1.3.5 2>/dev/null || true
mkdir -p "$OUT"/{in,out/src,out/latency_est,out/resource_est}   # AutoSA needs these to exist
cp "$SRC"/kernel.c "$SRC"/simd_info.json "$OUT/in/"
sed "s/#define N .*/#define N $N/" "$SRC/kernel.h" > "$OUT/in/kernel.h"
# --cleanenv is REQUIRED: host PYTHONPATH leaks a py>=3.9 sympy into the container's python3.8
apptainer exec --cleanenv --bind "$OUT":/work "$SIF" sh -c "cd /usr/src/docker_autosa && ./autosa /work/in/kernel.c \
  --config=./autosa_config/autosa_config.json --target=autosa_hls_c --output-dir=/work/out \
  --sa-sizes='{kernel[]->space_time[$ST];kernel[]->array_part[-1,-1,-1];kernel[]->latency[]}' \
  --simd-info=/work/in/simd_info.json --use-cplusplus-template --no-reschedule --live-range-reordering --hls"
test -s "$OUT/out/src/kernel_kernel.cpp" && echo "OK: $OUT/out/src/kernel_kernel.cpp"
