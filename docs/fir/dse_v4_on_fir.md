# DSE v4 on Fir

Run one config, `st0_c1`, on Alliance Canada Fir. Leave the other 29 configs unsubmitted until that job is legal.

This login could not open an SSH session to Fir (the `id_ed25519fir` key is not loaded in an agent here), so the node list below is from the Fir site docs and from `c2hls_paths.py` / `scripts/fir/`. After you log in, `sinfo -s` is the check that the partitions are actually up.

## What Fir is

Fir is the Alliance cluster at SFU that replaced Cedar. SSH from a machine where `~/.ssh/id_ed25519fir` is unlocked:

```bash
ssh fir
```

That uses the existing config: user `asa582`, jump host `fir.alliancecan.ca`, then `login3.int.fir.alliancecan.ca`. The short login name is `login3`.

| Resource | What the job uses |
| --- | --- |
| Base CPU node | 192 cores, 750 GB, 2× AMD EPYC 9655. v4 asks for 8 CPUs and 64 GB. |
| GPU nodes | 4× H100 80 GB. v4 does not use them. |
| FPGA cards | None. Csim and csynth still target part `xcu280-fsvh2892-2L-e` at 3.33 ns, from the device database inside the Vitis container. |
| Internet on compute nodes | Open. A compute job can call `https://api.deepseek.com` itself. |
| Storage | `$SCRATCH` is `/scratch/asa582`. Scratch is purged on the Cedar policy (about six weeks, with email warnings). |
| `/tmp` | RAM disk. Bytes written there count against the job memory cgroup. Vitis temp stays on scratch. |
| Wall time | 1 hour minimum, 7 days maximum. Slurm routes by `--time` when `--partition` is omitted. |

Time bins used by this Alliance generation (Fir’s own cap is 7 days, so there is no 28-day bin):

| Bin | Max wall |
| --- | --- |
| `b1` | 3 h |
| `b2` | 12 h |
| `b3` | 24 h |
| `b4` | 3 d |
| `b5` | 7 d |

`sinfo -s` on the login is authoritative. A job that sets both `--partition` and a `--time` outside that partition is rejected.

Accounts already used by this repo:

- CPU: `def-zhenman` (`FIR_COMPUTE_SLURM_ACCOUNT` in `scripts/fir/common.sh`)
- GPU: `def-zhenman_gpu` (`FIR_DEFAULTS` in `c2hls_paths.py`)

v4 is a CPU job, so it charges `def-zhenman`. Confirm with `sacctmgr show assoc user=asa582 format=Account,QOS%40`.

## What the PC2 launcher will not do on Fir

`scripts/pc2/start_dse_v4_one.sh` and `scripts/pc2/write_dse_v4_sbatch.sh` are Otus scripts.

- The starter exits unless `whoami` is `haqc2`.
- The generated script asks for `--partition=normal`, `--qos=cont`, and `--account=hpc-prf-llmfpga`.
- The proxy helper looks for ChatHLS at `/scratch/hpc-prf-llmfpga/asa582/projects/test-chathls/ChatHLS-ACL-26`.
- `scripts/pc2/run_dse_v4_one.py --pc2` calls `configure_site("pc2")`, which points Vitis at `/opt/software/FPGA/Xilinx/...` on Otus.

Do not sbatch those files on Fir.

The existing Fir session (`scripts/fir/start_session.sh`) is also the wrong shape. It starts a 4-GPU vLLM job for Devstral and only then a compute job. v4 calls `deepseek-v4-flash`.

## What already works on Fir

`configure_site("fir")` and `scripts/fir/fir_container_env.sh` are the Vitis path:

- Image: `/scratch/asa582/containers/xilinx_vitis_2023.2.standalone.sif`
- `module load apptainer/1.3.5`
- `scripts/fir/bin/vitis-run` runs `vitis-run` inside that image
- Part and clock in `scripts/fir/vitis_paths.env` are already `xcu280-fsvh2892-2L-e` and `3.33`
- Temp dir: `/scratch/asa582/tmp/c2hls`

`hls_eval.py` uses `vitis-run` from `PATH` and skips `settings64.sh` when that wrapper is visible. The compute script must source `scripts/fir/fir_container_env.sh` before Python starts.

`run_dse_v4_one.py` without `--pc2` does not call `configure_site`. That is what you want: Fir defaults would otherwise fill `C2HLS_MODEL` with Devstral and `OPENAI_BASE_URL` with `http://127.0.0.1:8000/v1`. Export the DeepSeek model and URL in the job, then run the script with no site flag.

## Files that have to be on Fir

The Fir clone is `/scratch/asa582/workspaces/code_translation_c2hls`. A `git pull` on `c2hls_enhanced_l_pc2_api_layout` brings the `st0_c1` inputs. The sibling AutoSA tree is not required for this one config.

`resolve_autosa_docs_dir()` uses `../AutoSA/docs` only when that tree contains `mm_codegen_factors.md`. The Fir AutoSA checkout does not, so the job reads the tracked pack:

- `inputs/dse_v4/docs/mm_codegen_factors.md`
- `inputs/dse_v4/docs/mm_configs/stream_stitching.md`
- `inputs/dse_v4/docs/mm_configs/st0/ap128_128_8__lat1_32__simd8.md`
- `inputs/dse_v4/artifacts/dse/campaigns/20260927_mm1024_u280_st0/u280_paper/validation/mm1024/candidate_1/autosa_out/src/kernel_kernel.cpp`
- the sibling `kernel_kernel.h` and `kernel_host.cpp` in that same `src/` directory

The n1024 bench is tracked despite `artifacts/pc2/*/`:

`artifacts/pc2/autosa_mm_ijk_benches/n1024/autosa_mm/testbench.cpp`

The other 29 gold trees are not in this pack. Do not copy `artifacts/pc2/autosa_mm_variant_sweep_20260918` or the paused `20260919` records into the output directory.

`DeepSeek_API` is already named in the Fir `~/.bashrc`. A non-interactive shell skips that file, which is why the variable is unset. The submit block below reads the assignment and exports it. Do not print the value, and do not write it into `fir.env` or the sbatch file.

## One job, st0_c1

On a Fir login:

```bash
ROOT=/scratch/asa582/workspaces/code_translation_c2hls
cd "${ROOT}"
git pull

test -f /scratch/asa582/containers/xilinx_vitis_2023.2.standalone.sif
test -f inputs/dse_v4/docs/mm_codegen_factors.md
test -f inputs/dse_v4/docs/mm_configs/stream_stitching.md
test -f inputs/dse_v4/docs/mm_configs/st0/ap128_128_8__lat1_32__simd8.md
test -f inputs/dse_v4/artifacts/dse/campaigns/20260927_mm1024_u280_st0/u280_paper/validation/mm1024/candidate_1/autosa_out/src/kernel_kernel.cpp
test -f artifacts/pc2/autosa_mm_ijk_benches/n1024/autosa_mm/testbench.cpp

module load apptainer/1.3.5
# shellcheck disable=SC1091
source scripts/fir/fir_container_env.sh
command -v vitis-run
sinfo -s | awk 'NR==1 || /cpubase/'
sacctmgr -n show assoc user=asa582 format=Account
```

`vitis-run` must print a path under `scripts/fir/bin`. `sinfo` must show CPU partitions up. Account `def-zhenman` must be in the association list.

Unset the empty placeholder before loading the key, then submit. The key stays in the job environment via `--export=ALL`. It is not written into the script.

```bash
unset OPENAI_API_KEY CHATHLS_API_KEY
if [[ -z "${DeepSeek_API:-}" ]]; then
  while IFS= read -r line; do
    case "${line}" in
      DeepSeek_API=*|export\ DeepSeek_API=*) eval "${line}" ;;
    esac
  done < "${HOME}/.bashrc"
fi
export OPENAI_API_KEY="${DeepSeek_API:?DeepSeek_API is unset}"
export OPENAI_BASE_URL=https://api.deepseek.com/v1
export CHATHLS_API_BASE=https://api.deepseek.com/v1
export C2HLS_MODEL=deepseek-v4-flash
export C2HLS_DSE_MODEL=deepseek-v4-flash
python3 -c 'import os; v=os.environ["OPENAI_API_KEY"]; assert len(v)>=20 and v.lower()!="empty"; print("api_key_loaded len=%d" % len(v))'

OUT=${ROOT}/artifacts/fir/dse_v4_st0_c1_$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "${OUT}"

sbatch --export=ALL <<EOF
#!/bin/bash
#SBATCH --job-name=st0_c1
#SBATCH --account=def-zhenman
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --chdir=${ROOT}
#SBATCH --output=${OUT}/st0_c1-%j.out
#SBATCH --error=${OUT}/st0_c1-%j.err

set -euo pipefail
cd ${ROOT}
module load apptainer/1.3.5
source scripts/fir/fir_container_env.sh
export C2HLS_TMP_ROOT=/scratch/asa582/tmp/c2hls
export C2HLS_VITIS_USER_HOME=/scratch/asa582/tmp/vitis_user_home
mkdir -p "\${C2HLS_TMP_ROOT}" "\${C2HLS_VITIS_USER_HOME}"
unset C2HLS_DSE_V3 C2HLS_DSE_V3_HARNESS C2HLS_DSE_SOURCE_KERNEL
unset C2HLS_VITIS_JOBS C2HLS_VITIS_JOBS_FROM_SLURM
command -v vitis-run
python3 scripts/pc2/run_dse_v4_one.py \\
  --config st0_c1 \\
  --out-dir ${OUT}/runs/st0_c1
EOF
```

There is no `--partition` and no `--qos`. A 24-hour request is inside the 7-day cap and lands in the `b3` CPU bin. Twelve hours, the Otus wall, lands in `b2` and can kill a 1024³ csim before the 86400-second tool timeout. The Python ceilings stay 86400 either way.

Eight cores and 64 GB is about 8 GB per core. Base nodes are about 4 GB per core, so the submit plugin may place this on `cpubase_bynode_b3` instead of `cpubase_bycore_b3`. That still runs. Check with `scontrol show job JOBID | grep Partition`.

Do not pass `--pc2`. Do not pass `--fir` (that flag is not on this script). Do not start `scripts/fir/start_session.sh`.

## After it is queued

```bash
squeue -u asa582 -n st0_c1
scontrol show job JOBID | awk '/JobState|Reason|Partition|TimeLimit|Command/'
```

A pending reason of `Priority` or `Resources` means the bin is busy. `PartitionTimeLimit` or `QOSMaxWallDurationPerJobLimit` means the wall does not fit the bin that was selected.

The run is legal only when `runs/st0_c1` records structural checks and a csim log that prints `Passed!` on the full 1024³ sum. `legal=True` is that bar. A Vitis `CSim done with 0 errors` line is not a substitute if the bench printed `Failed with N errors!`.

## If the first job fails before Vitis

| Symptom | Cause |
| --- | --- |
| `vitis-run` missing | `apptainer` module or the SIF path. `source scripts/fir/fir_container_env.sh` must run in the job, not only on the login. |
| AutoSA instruction or gold kernel missing | `inputs/dse_v4` is not in this checkout. Pull `c2hls_enhanced_l_pc2_api_layout` again. |
| HTTP 401 from DeepSeek | `OPENAI_API_KEY` was `EMPTY` or unset in the job. `sbatch --export=ALL` only keeps variables that were exported in the login shell. |
| Calls to `127.0.0.1:8000` | `C2HLS_MODEL` / `OPENAI_BASE_URL` were not exported, and something applied the Fir Devstral defaults. |
| `settings64.sh` under `/opt/software/FPGA` | The command included `--pc2`. |
| Job killed at the memory cgroup during csim | Vitis wrote under `/tmp`. `C2HLS_TMP_ROOT` must stay on `/scratch/asa582/tmp/c2hls`. |
| `OUT_OF_MEMORY` in the first seconds | `--mem` was omitted. Fir’s default memory is small and is enforced. |

## What stays unchanged

- One config until `st0_c1` is legal.
- One DeepSeek client for that job. A second concurrent job needs its own process, not a shared queue with more than one upstream worker.
- Cosim stays off.
- Do not edit `c2hls_paths.py` `PC2_DEFAULTS`, the live `I=J=K=64` kernel header, or the shared n1024 `testbench.cpp`.
