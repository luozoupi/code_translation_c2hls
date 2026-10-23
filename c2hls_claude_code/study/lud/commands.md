# LUD / AutoSA: reproducible commands (U280, 3.33 ns, Vitis HLS 2023.2)

All commands below were run or syntax-checked on the login node unless marked "NOT RUN". No Slurm job was submitted.
Paths: `W=/scratch/hpc-prf-llmfpga/asa582/projects/workspaces`, study dir `S=$W/c2hls_claude_code/study/lud`.

## 0. Two gotchas found while reproducing (both verified)

1. `--live-range-reordering` is mandatory for the LU kernel. `docs/examples/lu.rst` omits it; `autosa_tests/lu/README.md` and the earlier
   LU DSE campaign (`20260709_full21_dse`) include it. Without it `src/autosa` segfaults (rc=139) after "IO Grouping".
2. Run the container with `--cleanenv`. The host `PYTHONPATH` (`/home/asa582/scratch/packages/sympy`, needs py>=3.9) leaks in and
   `autosa_scripts/codegen.py` dies with `ImportError: Python version 3.9 or above is required for SymPy` (container python is 3.8.10).
   Also the `--output-dir` must already contain `src/`, `latency_est/`, `resource_est/` (the real campaign runner pre-creates them).

## 1. Container

```bash
module load apptainer/1.3.5
SIF=/scratch/hpc-prf-llmfpga/asa582/projects/workspaces/AutoSA/autosa_docker.sif      # Ubuntu 20.04, python 3.8.10, AutoSA at /usr/src/docker_autosa
# AutoSA source in the container = git b61a1b4132d631600696feba59eb606acb34d304 (Nov 23 2021, "Add min/max template in tapa code")
# = HEAD of https://github.com/UCLA-VAST/AutoSA as cloned on 2026-10-13.  The container has no vivado_hls/vitis_hls.
```
The `autosa_sandbox/` and `autosa_scripts/*`, `autosa_tests/lu`, `src/` dirs of the local checkout `$W/AutoSA` are EMPTY stubs (files only exist
inside the .sif and in `$W/AutoSA_20260709_0438.tar`); `scripts/pc2` and the U280 MM campaign scripts are not on this machine.

## 2. Generate the paper/README LU array (N=32 default, space_time 3) - tested, 5 s

```bash
$S/gen_lu.sh 32 3 $S/out_n32_st3        # wrapper below
```
Equivalent explicit form (what the wrapper runs):
```bash
G=$S/out_n32_st3; mkdir -p $G/{in,out/src,out/latency_est,out/resource_est}
cp $S/upstream/AutoSA/autosa_tests/lu/{kernel.c,kernel.h,simd_info.json} $G/in/
apptainer exec --cleanenv --bind $G:/work $SIF sh -c 'cd /usr/src/docker_autosa && ./autosa /work/in/kernel.c \
  --config=./autosa_config/autosa_config.json --target=autosa_hls_c --output-dir=/work/out \
  --sa-sizes="{kernel[]->space_time[3];kernel[]->array_part[-1,-1,-1];kernel[]->latency[]}" \
  --simd-info=/work/in/simd_info.json --use-cplusplus-template --no-reschedule --live-range-reordering --hls'
```
Upstream README command (OpenCL host, for xclbin) = same without `--hls`, and with `--output-dir=./autosa.tmp/output`.
Output: `out/src/{kernel_kernel.cpp,kernel_kernel.h,kernel_host.cpp,kernel.c,kernel.h,cmd}`; top is C++ `void kernel0(A_t1 *A, L_t1 *L, U_t16 *U)`
(A,L 32-bit scalar ports, U 512-bit), `#pragma HLS DATAFLOW`, 528 `PE_wrapper<p0,p1>` instances for N=32 (N(N+1)/2).
Verified outputs live in `$S/gen_test2/` (`lu_st3_ok`, `n16`, `n24`, `script_check`, `st{0..5}_m1`).

Other problem sizes (paper: 12/16/20/24): edit `#define N` in `kernel.h` (`gen_lu.sh <N> ...` does it). Verified: N=16 -> 136 PEs, N=24 -> 300 PEs.
Other space-time ids (array_part[-1,-1,-1], N=32): 3 -> 528 PEs, 4 -> 528 PEs, 5 -> 1024 PEs (square), 0/1/2 -> 1-D, 32 PEs. All generate.

What fails (verified, N=32): any real array partitioning, e.g. `array_part[16,16,16]` -> `isl_schedule_node.c:565: cannot handle extension nodes`, rc=247
(also with `latency[2,2]`). `simd[k]` is accepted but produces byte-identical code (no effect).

## 3. C simulation (needs Vitis HLS, not in the AutoSA sif) - NOT RUN

Upstream (Vivado HLS 2019.2, U200 part) `autosa_tests/lu/hls_script.tcl`: `csim_design` only. For U280/2023.2 use `$S/hls_csynth_u280.tcl`
(csim line commented in; host `kernel_host.cpp` prints `Passed!` / `Failed with N errors!`).
```bash
# inside the Fir Vitis container (same as dse_v4_claude env.sh: dse_v4_source_vitis)
module load apptainer/1.3.5; source /scratch/hpc-prf-llmfpga/asa582/projects/workspaces/code_translation_c2hls/scripts/fir/fir_container_env.sh
cd $S/out_n32_st3/out && vitis-run --tcl --input_file $S/hls_csynth_u280.tcl      # edit: uncomment csim_design for functional check
```
This must go through sbatch (login node forbids long Vitis jobs; project policy): wrap exactly like `dse_v4_claude/sbatch_csynth.sh`
(`--account=hpc-prf-llmfpga --partition=normal`, 8 cpu, 64G, >= 12 h; N=32 has ~30k lines of HLS C++ with 528 PEs so expect long csynth).

## 4. Csynth target (U280, 3.33 ns, 2023.2)

`$S/hls_csynth_u280.tcl` mirrors `code_translation_c2hls/scripts/autosa_dse_hls_common.py::_write_csynth_tcl`:
`open_solution "solution1" -flow_target vitis`, `set_part {xcu280-fsvh2892-2L-e}`, `create_clock -period 3.33 -name default`,
`config_compile -name_max_length 50`, `csynth_design`. (Env: `C2HLS_PART=xcu280-fsvh2892-2L-e`, `C2HLS_CLOCK_NS=3.33`, `scripts/fir/vitis_paths.env`.)
Metric convention of this project: latency min/max cycles + DSP; ignore estimated clock (memory note `csynth-clock-estimate-not-a-criterion`).

## 5. Hardware flow (xclbin; paper-style, NOT needed for csynth) - NOT RUN

Upstream `autosa_tests/lu/Makefile` is hard-wired to `PLATFORM := xilinx_u250_xdma_201830_2`, `--kernel_frequency 250`,
`--vivado.prop=run.impl_1.STRATEGY=Performance_EarlyBlockPlacement`, and needs `connectivity.cfg` (listed in README but NOT in the repo).
For U280 change `PLATFORM := xilinx_u280_xdma_201920_3` (as in `autosa_tests/mm_hbm/Makefile`) or the local `xilinx_u280_gen3x16_xdma_1_202211_1`
(C2HLS_DEVICE_PLATFORM), `--kernel_frequency 300` (3.33 ns), and write
```
[connectivity]
sp=kernel0_1.A:DDR[0]
sp=kernel0_1.L:DDR[1]
sp=kernel0_1.U:DDR[0]
```
(the stock `autosa_scripts/vitis_scripts/connectivity.cfg` names ports A/B/C; LU's are A/L/U). No FPGA card on Fir; hw flow is out of scope.

## 6. AutoSA's own DSE tools (what exists; none support LU as shipped)

```bash
# Old tuner (autosa_scripts/optimizer.py), per docs/tutorials/auto_tuning_exhaustive.rst, MM example:
python3 ./autosa_scripts/optimizer.py -c './autosa ./autosa_tests/mm/kernel.c --target=autosa_hls_c --simd-info=./autosa_tests/mm/simd_info.json --host-serialize --hls --sa-sizes="{kernel[]->space_time[3]}"' \
  --info autosa_config/hw_info.json -s autosa_config/optimizer_settings.json -train -p xilinx     # then --search ... --tuning-method=0
# New tuner / Odyssey (autosa_scripts/tuner/main.py, autosa_scripts/odyssey/main.py): tasks only mm / cnn (+mobilenet/resnet/vgg layers); python deps: numpy, sklearn, xgboost.
```
The campaign that previously tried LU here (`20260708_dse_fix*`, `20260709_full21_dse`, kernel id `lu`, space_time [3]) ran
`./autosa autosa_tests/lu/kernel.c --config=... --target=autosa_hls_c --simd-info=... --host-serialize --hls --tuning-method=0 --use-cplusplus-template --no-reschedule --live-range-reordering --config=<job>/autosa_config.json --output-dir=<job>/output --sa-sizes={kernel[]->space_time[3];kernel[]->array_part[8,16,8]}`
and produced NO usable LU design (tuning.json = `{"latency":{"tilable_loops":[]}}`, no aggregate rows) - consistent with the array_part failure above.

## 7. Proposed U280 / 3.33 ns study (summary; details in LUD_RESEARCH.md section 7)

```bash
for N in 12 16 20 24 32; do for ST in 3 4 5; do $S/gen_lu.sh $N $ST $S/runs/n${N}_st${ST}; done; done      # ~5 s each, login node OK
# then per design: sbatch a csynth job with hls_csynth_u280.tcl (Fir Vitis 2023.2 container), collect csynth.xml latency min/max, DSP/FF/LUT/BRAM
```
