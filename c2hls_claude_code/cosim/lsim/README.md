# Fast cycle estimates: LightningSim with a Vitis HLS 2023.2 `m_axi` model

**Goal:** cycle counts that match **Vitis HLS 2023.2 C/RTL cosim**, the only reference. They are produced in seconds to minutes instead of days.

Verilator (`../verilator/`) is for correctness only. Its AXI memory has no latency, so its cycle counts track csynth's.

## What was wrong with LightningSim on our designs

- **Version.** LightningSim v0.2.6 (= LightningSimV2, github.com/sharc-lab/LightningSim) models `m_axi` with constants calibrated on the **2021.1** adapter.
- **Read latency.** Fixed `latency + 12`.
- **Read credit.** 16 "bursts", split only at 4 KB.
- **Write response.** `latency + 7` after the last write.
- **Missing write behaviour.** No write throttling and no write credit.
- **Measured error on our 2023.2 microbenchmarks (stock):** mean |error| 7–8%, up to 82%. `v2021` rows in `T1_microbench.md`.
- **OmniSim** (the successor) is not released.

## What the 2023.2 cosim actually does (measured)

Measured on cosim VCDs of 70 microbenchmarks (`mb_gen.py`, `vcd_rerun.sh`, `parse_axi_vcd.py`), at both the bus ports and the adapter's kernel-side `I_*` ports.

- **Cosim memory** (`AESL_axi_master_*.v`): requests are queued, and one in-order server per port answers each burst at `t_request + latency`. `latency` is the INTERFACE `latency` pragma, default 64.
  - Data moves at 1 beat/cycle.
  - B arrives at last W + `latency` + 2.
- **Read side of the adapter.**
  - Bursts are split at `min(max_read_burst_length beats, 4 KB)`.
  - AR goes out one per cycle.
  - A burst may issue only when the kernel has *consumed* the burst `num_read_outstanding` earlier (read credit).
  - AR→R is `latency + 1`.
- **Write side of the adapter** (`CONSERVATIVE = 1`): kernel → `store` FIFO (`WBUFF_DEPTH = max_write_burst_length`) → `throttle` FIFO (one burst) → bus.
  - Bursts are split like reads.
  - AW and the W beats of a burst leave together, only when the whole burst is in the throttle FIFO, and bursts leave serially.
  - A one-burst throttle FIFO means 16-beat bursts are spaced 18 cycles apart and 64-beat bursts 66.
  - Burst j + `num_write_outstanding` waits for B of burst j: its AW comes `max(11, 5 + len)` cycles after that B.
  - The kernel stalls when `WBUFF + throttle` beats are buffered.
- **Cosim report** (Verilog latency). It also counts the testbench's AXI-Lite programming and its 8- or 9-cycle status poll:
  `report = P·ceil((ap_done − 3)/P) + K`, `P = 6 + n`, `K = 27 + 12n`, with n = number of s_axilite pointer arguments.
  This is exact on all 55 two-pointer microbenchmarks (P=8) and both three-pointer `autosa_mm` self-tests (P=9).

## The v2023 model (fork)

- **Fork:** `/scratch/asa582/tools/lightningsim/src`, branch `vitis2023-axi` (local commits), patch `lightningsim-vitis2023-axi.patch`.
- **Selection:** `LSIM_AXI_MODEL=v2023`. The default is `v2021`, and the fork in `v2021` mode reproduces stock LightningSim bit for bit on all 55 original microbenchmarks.
- **Per-interface config:** `axi_cfg.py` reads each `<top>_<bundle>_m_axi #(...)` instance (burst lengths, outstanding counts, widths) and the csynth argument→bundle map. It passes them through `LSIM_AXI_IFCFG` → `SimulationBuilder.set_axi_config`.
- **Reads:** one read-credit transaction per burst, with capacity `num_read_outstanding`.
  - request → first read: `L + r0`;
  - last read of burst j → first read of burst j+N: `L + r1`.
- **Writes:** one **AW node per burst** is added to the graph. It is the max of:

  | Source | Delay |
  |---|---|
  | kernel's last write of the burst | `wa` |
  | write request | `len + wq` |
  | AW(j−1) | `len(j−1)` |
  | AW of the burst holding beat e(j) − throttle depth | offset + `wf` |
  | AW(j−N) | `len + L + 1 + max(wc + len(j), wc2)` |

  - A kernel write waits for beat i − (WBUFF + throttle) to leave the bus (+ `wk`).
  - The write response is the last burst's B: AW + len + L + 1 + `wb`.
- **Constants** (cycles): r0=11, r1=7, wa=5, wq=10, wf=3, wc=5, wk=1, wb=3, wc2=11. They are runtime-settable via `lightningsim._core.set_axi_constants`, as used by `fit_lsim.py`.
  - All of them were read off the cosim VCDs, per event. A free fit to the report drove r0 to 0 to cancel the schedule tail below, so it was rejected.
- **End offset:** LightningSim keeps 7–14 cycles of schedule after the final write response, which the RTL does not have. `compare.py` subtracts `END_OFFSET = 11` before applying the report mapping.

## Results

Error is (predicted − cosim report)/cosim. `dut_done` is the DUT's own `ap_done` from the VCD.

| Tier | Set | Stock LightningSim (v2021) | **v2023 fork** |
|---|---|---|---|
| T1 | 70 m_axi microbenchmarks (`T1_microbench.md`) | mean 9.4%, max 81.5%; 28/70 within ±5 cycles of `ap_done` | **mean 0.46%; 68/70 within ±5 cycles of `ap_done` (mean 2.8 cycles)**. Report-level: 58/70 within ±1% or ±5 cycles; the rest are one 8-cycle poll quantum off on 200–1,300-cycle kernels, and `wr_n1024_out4` (−1.3%, write-credit bound) |
| T2 | `autosa_mm` self-tests at 16³ (`T2_selftest.md`) | −0.70%, −2.59% | **−0.01%, 0.00%** (70,875 vs 70,884; 68,679 vs 68,679) |
| T3 | designs at 256³ against our own Vitis cosim (`T3_n256.md`); 11 have cycle counts | mean 15.2%, max 31.2%; 0/11 within ±1% | **mean 0.34%, max 0.87%; 11/11 within ±1%**. Designs 03, 05 and 12 fail the RTL value check at 256³ (wrong outputs), so their cycles are of wrong RTL. 08: cosim OOM-killed at 48 GB, no reference, and LightningSim predicts 195.9M cycles (suspect). 02 can't be built at 256³ |
| T4 | 13 designs + `st0_c1` at 1024³ against the running Vitis cosim jobs (`T4_n1024.md`, `t4_table.py`) | predictions done; the reference is pending | predictions done (8–44 s each, 587 s for design 12); the reference is pending |

**Single-beat requests make cosim latency-bound.** Most designs issue one-beat requests (csynth: "Widened 1", "Inferred 16" floats). With `latency` 64 and 16 outstanding requests, cosim moves only about one beat every 4.5 cycles, while csynth and Verilator assume one per cycle.

- **The microbenchmark** `rdsc_n1024` (one-beat reads at II=1):
  - Vitis cosim reports **4,702 cycles**, against about 1,100 for one beat per cycle;
  - v2023 predicts 4,699 (−0.06%);
  - stock LightningSim is off by −2.8%, and by −75% at 4 outstanding;
  - single-beat writes (`wrsc_n1024`): v2023 −0.10%.
- **What this means at 1024³:** the T4 predictions are 1.4–2.5M cycles for designs 01, 02, 03, 04, 07, 09, 12 and 13, and 0.86–1.0M for 05, 08, 10 and 11. Neither csynth nor Verilator is a reference: both assume about one beat per cycle, so higher cosim counts are expected. The only checks are Vitis cosim itself: T3 (our 256³ cosims) now, T4 as the 1024³ jobs finish.

Design 02 can't be built below 1024³: its `C_LOAD_START` = 8192 needs `K/16·256` > 8192 + 2048. It gets T4 only.

## How to run

```bash
cd /scratch/asa582/workspaces/c2hls_claude_code/cosim/lsim
# 1. a Vitis 2023.2 project (csynth is enough for LightningSim; --cosim adds the reference cycles)
sbatch --job-name=ld_x --export=ALL,NAME=x,KERNEL=/path/kernel.cpp,SIZE=1024 sbatch_design.sh
# 2. LightningSim (v2023) on it, inside the Vitis container -> designs/x_n1024/lsim_v2023.json (end cycle)
sbatch --export=ALL,ROOT=/scratch/asa582/tmp/c2hls/lsim/designs,ONLY=x_n1024,MODEL=v2023 sbatch_lsim.sh
# 3. predicted cosim report and comparison
$PY compare.py /scratch/asa582/tmp/c2hls/lsim/designs --tags v2023 --match '_n1024$'
# T3 (once our 256^3 cosims and the queued LightningSim jobs finish) and T4 (as the Vitis 1024^3 jobs finish)
$PY compare.py /scratch/asa582/tmp/c2hls/lsim/designs --tags v2021,v2023 --match '_n256$' --md T3_n256.md --csv T3_n256.csv
$PY t4_table.py
```

- **Rebuild the fork:** `build_lsim.sh`. It uses the Rust 1.85 env in `/scratch/asa582/tools/lightningsim/rustenv`, and no LLVM rebuild is needed.
- **Microbenchmark reference:** `sbatch_ref.sh` (`ONLY=a:b`, `TRACE=port VCD=1` for VCDs).
- **Fit:** `sbatch_fit.sh`.
- **Per-event check:** `IO=1` in `sbatch_lsim.sh` → `lsim_io.py`, then `io_compare.py`.

| File | Role |
|---|---|
| `mb_gen.py` | microbenchmark kernels/testbenches (rd, wr, cp, df, chain; 512/32-bit) |
| `sbatch_ref.sh`, `vcd_rerun.sh`, `parse_axi_vcd.py` | Vitis cosim reference, VCD of bus + adapter `I_*` ports, per-transaction timing |
| `run_design.py`, `sbatch_design.sh` | real designs at size N (csynth, or csynth + cosim), project kept as one tarball |
| `run_lsim.sh`, `sbatch_lsim.sh`, `lsim_io.py` | LightningSim in the Vitis container; per-event m_axi dump |
| `axi_cfg.py` | per-interface adapter config from the generated RTL |
| `compare.py`, `io_compare.py`, `fit_lsim.py`, `sbatch_fit.sh` | report mapping and tables; per-event diff; constant fit |
| `t4_table.py` | 1024³ table: csynth, Verilator (not reference), LightningSim v2021/v2023, Vitis cosim reference |
| `build_lsim.sh`, `lightningsim-vitis2023-axi.patch` | rebuild/install the fork; its diff from upstream v0.2.6 |

**Data:**
- `/scratch/asa582/tmp/c2hls/lsim/{mb,mb_vcd,mb_vcd2,designs,designs_vcd2}`: one `proj.tar.gz` per project, to limit the file count.
- The env, fork and Rust toolchain take about 18.6K files in `/scratch/asa582/tools/lightningsim`.

## Limitations

- **Write-credit-bound transfers with long bursts** are off by up to about 1.3%: `wr_n1024_out4`. Single-beat credit-bound writes are within 0.1%.
- **Design 12:** LightningSim times the C code at 2.35M cycles. The Verilator run of its RTL finishes at 533K but with every output wrong (`../verilator/runs/12_ladder_agent_r2_L3`). The C model and the RTL disagree, so LightningSim's estimate for 12 assumes the RTL would behave like the C code.
- **Report quantum:** cosim reports in 8- or 9-cycle poll quanta, so a ±1-cycle estimate can show as ±8 on a tiny kernel. Above about 10K cycles this is under 0.1%.
- **Same assumptions as LightningSim:**
  - blocking FIFOs only, and none of our designs use non-blocking FIFO accesses;
  - only the HLS schedule and the C trace are simulated, so RTL bugs are invisible. Use the Verilator flow for correctness.
- **What "reference" means:** Vitis 2023.2 cosim, whose memory is a fixed-latency model and not HBM or DDR. Matching real-board DRAM timing is out of scope.
