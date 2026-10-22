# AutoSA LU decomposition (LUD): research notes for reproduction on U280 @ 3.33 ns

Date of research: 2026-10-13. Legend: **[C]** certain (read in a file/paper or executed), **[I]** inferred, **[U]** unknown.
Everything lives under `/scratch/asa582/workspaces/c2hls_claude_code/study/lud/` (`S` below). Commands: `S/commands.md`.

## 1. Bottom line

- AutoSA's FPGA'21 paper reports LU only as a small "stress test" **[C]**: N = 12/16/20/24, Alveo U250, results from **RTL simulation at an assumed 250 MHz**,
  latency 3.42 / 5.23 / 7.68 / 10.44 us, 6.8x geomean vs LAPACK on a Xeon. No resource, DSP, frequency, or throughput number is given for LU, and no
  exact `--sa-sizes` is quoted in the paper. There is **no** "best LU configuration" in the paper beyond "loops j and k as the space loops, 2-D triangular array".
- The only reproducible recipe is the repo's `autosa_tests/lu` (README / docs `lu.rst`): `space_time[3]`, `array_part[-1,-1,-1]` (= no array partitioning), `latency[]` (none),
  default `N = 32`, flags `--use-cplusplus-template --no-reschedule --live-range-reordering`. I reproduced generation in the container (N=16/24/32).
- LU in AutoSA has effectively **no tunable design space** in this revision: array partitioning and latency hiding fail or are empty, SIMD is a no-op, and
  neither the AutoSA tuner nor Odyssey has an LU task. The knobs that exist: problem size N, space_time id (0..5), clock/part. A previous LU DSE campaign on this
  cluster (`20260708_dse_fix*`, `20260709_full21_dse`) produced nothing usable.
- The MM U280 campaign configs (`st0_c1` = array_part 128,128,8 / latency 1,32 / simd 8 / 128 PEs) do **not** transfer to LU.

## 2. Sources inspected

| Item | Where | Notes |
|---|---|---|
| Local checkout | `/scratch/asa582/workspaces/AutoSA` | **Skeleton**: `autosa_tests/*`, `autosa_scripts/*`, `autosa_config/*_libs`, `src/*`, `autosa_sandbox` are empty dirs; only `autosa_docker.sif` (1.5 GB), `docs/*`, `artifacts/`, `scripts/pc2/` are all empty directories, and the dir is not a git repo; only `autosa_docker.sif` (1.5 GB) is real **[C]**. Real content is inside the sif and in `/scratch/asa582/workspaces/AutoSA_20260709_0438.tar` (34 GB). |
| AutoSA revision | container `/usr/src/docker_autosa`, `git log -1` | `b61a1b4132d631600696feba59eb606acb34d304` (2021-11-23 "Add min/max template in tapa code") **[C]**; equals HEAD of github UCLA-VAST/AutoSA master on 2026-10-13 (shallow clone has only this commit) **[C]**. Tarball `.git/HEAD` = `refs/heads/master`. |
| Upstream clone | `S/upstream/AutoSA` (49 MB) | `autosa_tests/lu`, `docs/examples/lu.rst`, `autosa_scripts/{tuner,odyssey,optimizer.py,...}` byte-identical to container for the files I diffed (tuner, lu/kernel.h, lu/README.md) **[C]**. No LU config other than the one in README/lu.rst anywhere in the repo (grep of `autosa_tests`, `docs`, tuner/odyssey tasks) **[C]**. |
| Paper PDF | `S/upstream/paper/autosa_fpga21.pdf` (+ `.txt`) | https://par.nsf.gov/servlets/purl/10250371 (NSF-PAR copy of Wang, Guo, Cong, FPGA'21, DOI 10.1145/3431920.3439292, pp. 93-104). The ACM and cadlab/vast.cs.ucla.edu URLs returned 403/404/no route from this host. |
| Odyssey paper | `S/upstream/paper/odyssey_2111.14252.pdf` | arXiv 2111.14252, Basalama/Wang/Cong, "Search for Optimal Systolic Arrays...". Mentions LU only in related work (irregular domains need manual effort); all results are MM/CNN on U250 **[C]**. |
| Docs | https://autosa.readthedocs.io/en/latest/examples/lu.html | Same text as `docs/examples/lu.rst` in the repo **[C]**. |
| Previous MM campaign docs | `code_translation_c2hls/{docs/pc2,inputs/dse_v4/docs}`, `dse_v4_claude/autosa_pe128_simd8_latency.md` | See section 6. |
| Prior LU DSE attempt | extracted: `S/local_snapshot_dse/AutoSA/artifacts/dse/campaigns/*/{campaign.json,optimizer_settings_lu.json,exhaustive/lu/...}` | See section 5. |

## 3. LU benchmark definition **[C]**

Source: `autosa_tests/lu/kernel.c` (upstream = container), header `kernel.h`: `typedef float data_t; #define N 32` (a commented `//#define N 3`). Files in dir:
`Makefile, README.md, add_batch.py, hls_script.tcl, kernel.c, kernel.h, simd_info.json` (README also lists `connectivity.cfg`, which does **not** exist in the repo).

- Algorithm: PolyBench LU (paper Sec. 8.2: "We choose the algorithm implemented in PolyBench [56]"), but **rewritten with uniform dependences** by hand
  (docs note: "the input source code has been modified to make sure all dependences are uniform. AutoSA lacks the ability to automatically uniformize the program").
  Not in place: `lu_device(A, L, U)` reads `A`, writes `L` and `U` (U is written transposed: `U[j][i]`). Compute is on scalar temporaries
  `prev_V, V, U_tmp, L_tmp` (N x N, locals in the scop).
- Loop nest inside `#pragma scop`: `for k { for j=k..N-1 { for i=k..N-1 { ... }}}` (pyramid domain) with 9 statements (paper Table 4: "LU Decomposition [n]: [12/16/20/24], #Stmts 9, LOC 27"):
  ```
  prev_V[i][j] = (k==0) ? A[i][j] : V[i][j];
  if (j==k) { U_tmp[i][j]=prev_V[i][j]; U[j][i]=U_tmp[i][j]; }                 // U row k
  else { U_tmp[i][j]=U_tmp[i][j-1];
         if (i==k) { L_tmp[i][j]=prev_V[i][j]/U_tmp[i][j]; L[i][j]=L_tmp[i][j]; }   // L column k (the division)
         else      { L_tmp[i][j]=L_tmp[i-1][j]; }
         V[i][j] = prev_V[i][j] - L_tmp[i][j]*U_tmp[i][j]; }                    // rank-1 update
  ```
  Float data type, no pivoting, A is made SPD-ish by `init_array` (B = A*A^T). Checker: `|L_golden-L| > 0.001` or same for U increments `err`; prints `Passed!` / `Failed with %d errors!`.
- `simd_info.json`: `{"kernel3": {"reduction": ["n"]}}` (a leftover; no effect, see section 4).
- Statement kinds are non-uniform (3 node types per paper Fig. 12a), domain is a pyramid; hence C++ templates (`--use-cplusplus-template`) so PEs (p0,p1) differ in function, and `--no-reschedule`
  (ISL rescheduling would destroy the permutable band).

## 4. What I executed (container, login node, no Slurm) **[C]**

See `commands.md` section 0/2. Results (N=32, `array_part[-1,-1,-1]`, `latency[]`, flags as README):

| space_time | outcome | PEs |
|---|---|---|
| 0, 1, 2 | generates (1-D arrays) | 32 |
| 3 (README/doc choice) | generates | **528 = N(N+1)/2**, p0,p1 in 0..31 (triangular) |
| 4 | generates | 528 |
| 5 | generates | 1024 (N x N) |
| 3, N=16 / 24 | generates | 136 / 300 |
| 3 without `--live-range-reordering` | **segfault** (rc 139) | - |
| 3 with `array_part[16,16,16]` (also `latency[2,2]`) | `isl_schedule_node.c:565: cannot handle extension nodes`, rc 247 | - |
| 3 with `simd[4]` or `simd[1]` | generates; output byte-identical to no-simd | 528 |

Generated top (st3, N=32): `kernel_kernel.cpp` 29,772 lines; `void kernel0(A_t1 *A, L_t1 *L, U_t16 *U)` (C++ symbol, `#pragma HLS DATAFLOW`, m_axi bundles gmem_A/L/U);
modules A_IO_L1/L2/L3_in, L_drain_IO_L1/L2/L3_out, U_drain_IO_L1/L2/L3_out, `PE<p0,p1>`. A and L ports are 32-bit scalars; only U is 512-bit packed (U_t16). Files: `S/gen_test2/lu_st3_ok/`.
Meaning of `array_part[-1,-1,-1]`: AutoSA source `autosa_trans.cpp` ~line 1260: "if all tiling factors are -1, we will skip array partitioning" -> the PE grid is the full iteration extent (hence N(N+1)/2 PEs).
`space_time[3]` = pair of loops (k,j) in the kernel's loop order (k,j,i) **[I]** (AutoSA enumerates 1-D choices then pairs); this matches the paper's "choosing the loops j and k as the space loops ... 2D systolic array in a triangular shape ... A fed only to the first row, L drained from all PEs, U only from diagonal PEs" **[I]**, since st3/st4 both give 528 PEs, I did not prove which of 3/4 is the paper's Fig. 12b; the README/doc picks 3.

## 5. DSE tools: inputs, device models, target/clock

**Old tuner** `autosa_scripts/optimizer.py` (+ `optimizer_prune.py`, `resource_model.py`, `latency_model.py`; doc `docs/tutorials/auto_tuning_exhaustive.rst`, genetic variant `auto_tuning_genetic.rst`):
inputs = the AutoSA command (`-c`, space_time must be given explicitly), `--info hw_info.json` (resource caps), `-s optimizer_settings.json`, `-p xilinx`, `--training/--search`.
Searches array_part / latency / SIMD factors (divisor tiling, PE-count pruning, `reg_size`, resource ratios), metric `latency`, `search.cycle_period` default 5 ns, regression resource model trained on 16 HLS samples
(FF/LUT/DSP; BRAM analytic), latency model from `latency_est/*_loop_info.json`. Paper Sec. 7. Examples/settings only for MM, CNN, MTTKRP, TTM(c) (`autosa_config/optimizer_settings_libs/*`).
**New tuner** `autosa_scripts/tuner` (`main.py --task mm|cnn --cst hw_cst --designs ...`; `cst/hw_cst.json` = U250 caps with ratio 0.7; `task/{cnn,mm,mm2}.json`) and **Odyssey** `autosa_scripts/odyssey` (`main.py`, `workload/*.json`, `cst/{u250,vu9p,hw_cst,single_test}.json`,
`designs_lib/gemm/kernel*_*.json`, scripts `run_arch*.sh`): affine-perfect-nest MM/conv workloads, evolutionary/ML search, designs are templated by hand-written JSON descriptions - **no LU workload** **[C]**.
Device models: `autosa_config/hw_info_libs/hw_info.json.{u200,u250,ku3}`; default `hw_info.json` = U250 (DSP 12288, BRAM18K 5376, FF 3,456,000, LUT 1,728,000, URAM 1280). **U280 is not a shipped device model** **[C]**
(U280 only appears as a platform string in `mm_hbm` and in AutoBridge docs). The U280 MM campaign must have used a custom hw_info/config (not on this machine) **[U]**. U280 caps for a custom file (public datasheet, **[I]** not from AutoSA): DSP 9024, BRAM18K 4032, URAM 960, LUT 1,303,680, FF 2,607,360.
Clock/part are **not** set by AutoSA codegen: `autosa_scripts/hls_scripts/hls_script*.tcl` hard-code `set_part {xcu200-fsgd2104-2-e}` and `create_clock -period 5`; `autosa_tests/lu/Makefile` hard-codes U250 platform + `--kernel_frequency 250`; the tuner's latency->seconds uses `cycle_period` (5 ns).
So retargeting to U280 / 3.33 ns is purely a script/tcl change (see `commands.md`, `hls_csynth_u280.tcl`).

**Prior LU DSE here** (`S/local_snapshot_dse/...`): `campaign.json` kernel entry `{"id":"lu","kernel_c":"autosa_tests/lu/kernel.c","simd_info":"autosa_tests/lu/simd_info.json","param_names":null,"space_time":[3],"odyssey_enabled":false}`, optimizer settings `optimizer_settings_lu.json` = the MM-style defaults
(PE_num [8,32], reg_size [32,256], search timeout 720 min). Training jobs were generated with `--host-serialize --hls --tuning-method=0 --use-cplusplus-template --no-reschedule --live-range-reordering ... --sa-sizes={kernel[]->space_time[3];kernel[]->array_part[8,16,8]}` (also [8,32,8], [16,16,8]); each `tuning.json` is `{"latency":{"tilable_loops":[]}}`;
no LU rows in `aggregate/exhaustive_top5.jsonl`. I.e. the tuner flow yielded no LU designs **[C]**. (Why exactly: not investigated beyond reproducing the `array_part` failure above.)

## 6. Previous MM campaign procedure (for parity)

- `dse_v4_claude/autosa_pe128_simd8_latency.md`: AutoSA campaign table, 1024^3 U280, 128 PEs x SIMD 8; `st0_c1` = `space_time[0]; array_part[128,128,8]; latency[1,32]; simd[8]`, latency min/max 1,169,498 / 2,130,396 cycles. Those are **AutoSA csynth latency min/max** (campaign artifacts `artifacts/dse/campaigns/20260927_mm1024_u280_st{0,3,4}/u280_paper/validation/mm1024/candidate_N/autosa_out/`).
- `code_translation_c2hls/inputs/dse_v4/docs/mm_codegen_factors.md`: MM flags `--host-serialize --hls --tuning-method=0` (+ `--local-reduce --reduce-op=+ --simd-touch-space --array-contraction` for ST4); campaign `autosa_config.json` has `array_part_L2.enable=0`, `hbm.enable=0`.
- `code_translation_c2hls/scripts/autosa_dse_hls_common.py`: csynth tcl = `open_solution "solution1" -flow_target vitis; set_part xcu280-fsvh2892-2L-e; create_clock -period 3.33; config_compile -name_max_length 50; csynth_design`, run with `vitis-run --tcl --input_file`. Fir container: `scripts/fir/fir_container_env.sh` (`XILINX_SIF=/scratch/asa582/containers/xilinx_vitis_2023.2.standalone.sif`, `module load apptainer/1.3.5`).
- `docs/pc2/2026-09-07-autosa-other-kernels-spend-dsp-plan.md`: `autosa_lu` is already in the c2hls kernel inventory: N=32 float, rank-1 csynth **none**, ~12 KB working set, "no 5000 DSP legal - k is sequential (div + rank-1 update); ablation + highest legal DSP".
- Not available on this machine **[U]**: the AutoSA-side campaign driver (`AutoSA/scripts/pc2/*`, `artifacts/dse/mm1024_u280_three_rankings.csv`, the U280 `hw_info`/`optimizer_settings`); they lived on the PC2 filesystem (`/scratch/hpc-prf-llmfpga/asa582/projects/AutoSA`), which is not mounted.

## 7. Paper details for LU (FPGA'21, Wang/Guo/Cong) **[C]**

PDF page numbers (pdf p.) / paper sections:
- Abstract/Intro (p.1-2): AutoSA "supports applications with complicated and irregular dependence structure such as LU decomposition".
- Table 4, Sec. 8 (p.8): benchmark description, `LU Decomposition [n]: [12/16/20/24], #Stmts 9, LOC 27`. Sec. 8 text: "We use Xilinx Vitis 2019.2 ... target Xilinx Alveo U250 ... We also adopt AutoBridge".
- Sec. 8.2 "Case Study 2: LU Decomposition" (p.9-10), quotes: "We choose the algorithm implemented in PolyBench [56]. Figure 12a shows the dependence graph of the LU algorithm"; challenges: "The iteration domain is in a pyramid shape..." and "The statement instances are non-uniform and conduct different computations ... three types of nodes";
  "We use the LU decomposition as a stress test to assess the robustness of our framework"; "Figure 12b shows an example array generated by AutoSA by choosing the loops j and k as the space loops. This mapping leads to a 2D systolic array in a triangular shape. In this array, data of matrix A are fed only to the first row of the array. The final results of matrix L are drained out from all the PEs, while the results of matrix U are drained only from the PEs on the diagonal of the array."
- Table 8 (p.10) "LU performance comparison with LAPACK":

  | N | 12 | 16 | 20 | 24 | Geo. mean |
  |---|---|---|---|---|---|
  | LAPACK (Xeon E5-2699 v3, 189 GB, avg of 10000 calls) | 11.69 us | 41.06 us | 78.09 us | 83.38 us | |
  | AutoSA | 3.42 us | 5.23 us | 7.68 us | 10.44 us | |
  | Speedup | 3.4x | 7.8x | 10.2x | 8.0x | 6.8x |

  "The FPGA results here are collected from RTL simulation with the assumed frequency of 250MHz." "As there is no reported numbers for LU decomposition in the previous automation frameworks ... we compare ... with LAPACK". "all the PEs are fully-pipelined and the systolic array extracts the maximal pipeline parallelism ... dataflow-like architecture."
  Implied cycles at 250 MHz **[I]**: 855 / 1308 / 1920 / 2610 cycles (3.42 us x 250 MHz etc.), growing ~N^2 (not computed by authors).
- Not reported for LU **[U]**: array_part / latency / SIMD values, `--sa-sizes`, resource use, achieved MHz, DSP efficiency, GFLOPs. Fig. 13 ablation (array partitioning, latency hiding, SIMD, double buffering) covers only CNN, MTTKRP, TTMc - **not LU**. The only "128 PE, SIMD 8, best 1-D" statement in the paper is for **MM design 1** (Table 5: 128x8, 346 MHz, 555 GFLOPs, DSP 42%) - same family as our `st0_c1`, not LU.
- Other papers: Odyssey (arXiv 2111.14252) no LU results. No other AutoSA-family paper with LU results found (search); "FLuD" (UMN) is an unrelated manual LU array.

## 8. Plan for LU on U280 @ 3.33 ns (proposal)

Goal definition (to confirm with the user **[U]**): "reproduce AutoSA's LU array and measure csynth latency/DSP on xcu280 at 3.33 ns", then feed the C2HLS/LLM comparison the same way as MM (latency min/max + DSP; ignore the clock estimate).

1. **Sizes**: primary N in {12, 16, 20, 24} (paper, Table 8) plus N=32 (repo default / c2hls `autosa_lu`). Resource sanity **[I]**: 2-D triangular array has N(N+1)/2 PEs each with a float mul+sub (and a divider in the i==k PEs); N=32 -> 528 PEs, N=24 -> 300, N=64 would be 2080 PEs and likely exceeds the 9024-DSP U280 budget if ~5 DSP/PE. So large N (e.g. 1024 as in MM) is impossible without array partitioning, which currently fails in AutoSA.
2. **Design points** (the whole DSE is tiny): `space_time in {3,4,5}` (2-D; optionally 0,1,2 as 1-D 32-PE baselines) x N. `array_part[-1,-1,-1]`, `latency[]`, no simd. 5 sizes x 3 st = 15 generations (~5 s each, login node) + 15 csynth jobs (sbatch, Fir Vitis 2023.2 container, `hls_csynth_u280.tcl`, >=12 h wall like `sbatch_csynth.sh`; large DATAFLOW with up to ~1000 PE instances may take hours).
3. **Metric**: csynth latency min/max cycles, DSP/FF/LUT/BRAM, at part `xcu280-fsvh2892-2L-e`, 3.33 ns; csim `Passed!` via `kernel_host.cpp` for correctness (tolerance 1e-3 in-source). Cosim optional (paper numbers are RTL sim at 250 MHz, so cosim cycles x 3.33 ns would be the like-for-like comparison with Table 8 **[I]**; project default is cosim off). Convert cycles->us with the *target* clock for comparison.
4. **If a real DSE is wanted** (tiling/latency/SIMD) it requires fixing/extending AutoSA (array_part crash `cannot handle extension nodes`, tuner `tilable_loops=[]`), i.e. new work beyond reproduction **[U]** whether feasible. Otherwise the "DSE" for the LLM study is over hand-written dataflow rewrites of the LU kernel, using the AutoSA design as the rank-1 reference.
5. For a faithful "U280 AutoSA" resource model add `autosa_config/hw_info_libs/hw_info.json.u280` (values above) - only relevant for the tuner, which cannot run LU anyway.

## 9. Certain / inferred / unknown summary

- **Certain**: LU source, README/doc command, flags (+ the mandatory `--live-range-reordering` and `--cleanenv`), U250/Vitis 2019.2 testing env, paper Table 8 numbers & RTL-sim@250 MHz, AutoSA rev b61a1b4 (container == upstream HEAD), PE counts for N=16/24/32, array_part/latency/simd behaviour, no tuner/Odyssey LU task, no U280 device model in repo, hard-coded U200/U250 in scripts.
- **Inferred**: space_time[3] = (k,j) = the paper's j,k triangular array (3 vs 4 not discriminated); implied cycle counts; U280 resource caps; DSP/PE estimate; cosim-vs-csynth comparability.
- **Unknown**: paper's own sizes/`sa-sizes` beyond "j,k space loops" (the authors' artifact/slides not found), LU resource/frequency in the paper, whether any later AutoSA commit supports LU partitioning (shallow clone only; HEAD == container; full history/issues were not searched offline), the AutoSA-side U280 campaign scripts/hw_info used for MM (not mounted), actual LU csynth numbers on U280 (no job run), whether 2023.2 csynth of the 30k-line N=32 design completes within reasonable time.
