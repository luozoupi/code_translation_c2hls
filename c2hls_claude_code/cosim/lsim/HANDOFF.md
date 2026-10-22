# Handoff: fast Vitis-2023.2-cosim cycle estimates with LightningSim (v2023 AXI model)

State as of 2026-10-11. See `README.md` for the full method, and the `T*_*.md` tables for the results.

## 1. Goal and ground rules

- **Reference:** cycle counts from **Vitis HLS 2023.2 C/RTL cosim** are the only reference.
  - csynth latency is not a reference; it assumes ideal memory.
  - Verilator cycles (`../verilator/`) are not a reference either. That flow is for correctness only, and its memory answers almost at once.
  - Cosim cycles above csynth are expected.
- **Aim:** a simulator that reproduces cosim cycles in seconds to minutes instead of days. LightningSimV2 was the only candidate, since OmniSim's code is not released.
- **Constraints:**
  - The c2hls checkout is read-only.
  - Never touch the running 1024³ Vitis cosim jobs (`../jobs.txt`).
  - Storage is limited by file count, so keep Vitis projects as one `proj.tar.gz` each. Never clone LightningSim with its `llvm-project` (119K files).
  - Temp files go under `/scratch/asa582/tmp/c2hls` (`/tmp` is a memory disk); jobs run on node-local `$SLURM_TMPDIR`.
  - Slurm account `def-zhenman`. No tight polling: chain jobs with `--dependency`, or check on demand.

## 2. Implementation state

| Item | Location |
|---|---|
| LightningSim fork (local branch `vitis2023-axi`, 4 commits on upstream v0.2.6 `1abd9a6`, **not pushed**) | `/scratch/asa582/tools/lightningsim/src` |
| Env (conda/micromamba), Rust 1.85 build env | `/scratch/asa582/tools/lightningsim/{env,rustenv,cargo}` |
| Patch vs upstream | `lightningsim-vitis2023-axi.patch` |
| Harness, tables, README | this directory (`cosim/lsim/`) |
| Projects and data (tarballs, VCDs, dumps) | `/scratch/asa582/tmp/c2hls/lsim/{mb,mb_vcd,mb_vcd2,designs,designs_vcd2}` |

- **Switch:** `LSIM_AXI_MODEL=v2021|v2023`. The default is `v2021`, which reproduces stock LightningSim bit for bit; `run_lsim.sh` defaults to `v2023`.
- **Changed Rust files:** `backend/lightningsim-core/{axi_interface.rs, edge.rs, lib.rs, builder/{axi_builder.rs, axi_rctl.rs, event.rs, mod.rs}}`.
- **Changed Python:** `backend/lightningsim/trace_file.py` (`_apply_axi_config`).
- **Per-port adapter config:** `axi_cfg.py` reads each `<top>_<bundle>_m_axi #(...)` instance in `syn/verilog/<top>.v` and the csynth argument→bundle map. It writes JSON, passed through `LSIM_AXI_IFCFG` → `SimulationBuilder.set_axi_config`. The trace names interfaces by bundle (`gmem0`, …).
- **Constants:** runtime-settable through `lightningsim._core.set_axi_constants([...])` or `LSIM_AXI_CONST`.
- **Rebuild:** `build_lsim.sh` (Rust core only; no LLVM rebuild).

## 3. Model (L = INTERFACE `latency` pragma, default 64; len in beats)

### Cosim memory (`AESL_axi_master_*.v`)
- One in-order server per port starts a burst at `t_req + L` and sends 1 beat/cycle.
- On the bus, AR→R = L+1 and last W→B = L+2.

### Reads
- Bursts are split at `min(max_read_burst_length, 4 KB)`.
- One read-credit transaction per burst, with capacity `num_read_outstanding`.
- `first_read(req) ≥ readreq + L + r0`
- `first_read(burst j) ≥ last_read(burst j−N) + L + r1`, where the credit is released when the kernel has consumed the burst.

### Writes
2023.2 adapter path: store FIFO (`WBUFF = max_write_burst_length`) → one-burst throttle FIFO (CONSERVATIVE) → bus.

There is one virtual **AW node** per burst:

```
AW(j) = max( last_write(j) + wa,
             writereq + len(j) + wq,
             AW(j−1) + len(j−1),
             AW(burst holding beat e(j)−T) + offset + wf,          T = write burst beats
             AW(j−N) + len(j−N) + L + 1 + max(wc + len(j), wc2) )   N = num_write_outstanding
kernel write i ≥ AW(burst holding beat i−(WBUFF+T)) + offset + wk
write response  ≥ AW(last burst) + len + L + 1 + wb       (+ legacy lower bound last_write + L + wl)
```

- **Constants** (order r0, r1, wa, wq, wf, wc, wk, wb, wl, wc2): `[11, 7, 5, 10, 3, 5, 1, 3, 0, 11]`, all taken from per-event cosim VCD timing.
- A free fit to report totals drove r0 to 0, so it was **rejected**. That fit is kept as `fit_free_rejected.json`.

### Cosim report mapping (`compare.py`)

```
report = P·ceil((E − END_OFFSET − 3)/P) + K,   P = 6 + n,   K = 27 + 12n
```

- n = number of s_axilite pointer arguments; E = LightningSim end cycle.
- `END_OFFSET = 11` (v2023 only) removes LightningSim's schedule tail after the final write response; the RTL raises `ap_done` in the same cycle as B.
- The mapping is exact on the DUT `ap_done` of all 2-pointer microbenchmarks (P=8) and both 3-pointer self-tests (P=9).

## 4. Results

| Tier | Stock v2021 | **v2023** |
|---|---|---|
| T1: 70 microbenchmarks (`T1_microbench.md`) | 28/70 within ±5 cycles of `ap_done` | 68/70 within ±5 cycles (mean 2.8); 58/70 within ±1% of the report (the rest are one poll quantum on tiny kernels, plus `wr_n1024_out4` at −1.3%) |
| T2: `autosa_mm` self-tests, 16³ (`T2_selftest.md`) | −0.70%, −2.59% | −0.01%, 0.00% |
| T3: 11 designs at 256³ vs our cosims (`T3_n256.md`) | 1.5–31% low | **mean 0.34%, max 0.87%, 11/11 within ±1%** |
| T4: 1024³ (`T4_n1024.md`, `t4_table.py`) | predictions only | predictions only; 8–44 s per design, 587 s for design 12 |

- **Why v2021 fails:** confirmed by per-event dumps of designs 01, 03, 04, 07 and 13 at 256³ (`designs/*_n256/lsim_io_io_v20{21,23}.json`).
  - About **97% of the gap is the write side.** v2021 has no write limit and no response model, so it streams single-beat C writes at about 1.0 cycle/beat; cosim runs them at about 4.7.
  - About **3% (a constant 509 cycles) is reads**, from v2021's old latency constants.
- **Key physical finding:** most designs issue single-beat requests (csynth "Widened 1"). At L=64 with 16 outstanding, cosim moves about one beat every 4.5 cycles.
  - The microbenchmark `rdsc_n1024` reports 4,702 cycles in cosim; v2023 is off by −0.06%.
  - As a result, the 1024³ predictions are about 1.4–2.5M cycles for designs 01, 02, 03, 04, 07, 09, 12 and 13, and about 0.86–1.0M for 05, 08, 10 and 11.

## 5. Unresolved issues

1. **Design 08 at 256³ is not representative.** With `NJ = J/64 = 4` accumulators instead of 16, HLS cannot pipeline `compute_k_compute_jj`. csynth itself says 196.3M cycles, and LightningSim's 195.9M is within 0.23%.
   - The 192 GB cosim rerun (job 64256469) and its LightningSim dependents (64256471, 64256472) were **cancelled on 2026-10-11** at the user's request, before they started; they could not have finished.
   - 08 is now T4-only, like 02. The first 256³ attempt is kept in `designs/08_dspredo_plain_r3_n256.oom48G/`.
2. **Design 02** can't be built below 1024³ (`C_LOAD_START` = 8192 requires K = 1024). T4-only.
3. **RTL value-check failures at 256³** for designs 03, 05 and 12 (65,536, 65,536 and 65,528 errors). Their T3 cycle counts come from wrong RTL.
   - Design 12 is also wrong in Verilator at 1024³ (finishes at 533K with all outputs wrong). LightningSim's 2.35M for 12 times the C code.
4. **`st0_c1`** (dse_v4 128-PE kernel, no cosim job): the LightningSim v2023 run failed after 27 min.
   - The error is in `trace_file.py:153`: `int(basic_block)` received `0x5652763ae150`. A `trace_bb`/`loop_bb` trace line carries a pointer instead of a basic-block number, probably interleaved or corrupted trace output from the dataflow design's many processes.
   - Not investigated yet. Inputs are in `/scratch/asa582/tmp/c2hls/lsim/st0c1_inputs/` and the project in `designs/st0_c1_n1024/`.
5. **Known model limits:**
   - Write-credit-bound long bursts: `wr_n1024_out4` is −1.3%.
   - The schedule-tail offset is handled as a constant (±3 cycles).
   - Blocking FIFOs only.
   - No RTL bugs are visible to LightningSim.

## 6. Next steps

1. **T4:** as the 13 Vitis 1024³ cosims finish (`../NAME/cosim_summary.json`; jobs in `../jobs.txt`, running 1–2.3 days so far), run `$PY t4_table.py` and check v2023 against them, aiming for ±3%.
2. Optionally debug `st0_c1`'s trace parse (§5.4): inspect the raw trace lines with `lightningsim --debug`, and check whether the trace writes are thread-safe (`src/trace.cpp`).

## 7. Commands

```bash
source /scratch/asa582/workspaces/dse_v3_claude/env.sh          # gives $PY; sbatch scripts source Vitis themselves
cd /scratch/asa582/workspaces/c2hls_claude_code/cosim/lsim
sbatch --export=ALL,NAME=x,KERNEL=/path/kernel.cpp,SIZE=1024[,COSIM=1] sbatch_design.sh        # project (+cosim)
sbatch --export=ALL,ROOT=/scratch/asa582/tmp/c2hls/lsim/designs,ONLY=x_n1024,MODEL=v2023[,IO=1] sbatch_lsim.sh
$PY compare.py /scratch/asa582/tmp/c2hls/lsim/designs --tags v2021,v2023 --match '_n256$'   # T3-style table
$PY t4_table.py                                                                              # T4 table
sbatch --export=ALL,OUT=/scratch/asa582/tmp/c2hls/lsim/mb,ONLY=a:b[,TRACE=port,VCD=1] sbatch_ref.sh  # microbench cosim
$PY io_compare.py <axi_timing.json> <lsim_io.json>                                           # per-event check
```
