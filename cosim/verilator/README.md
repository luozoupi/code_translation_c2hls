# Verilator RTL-correctness cosim for vanilla Vitis HLS kernels

This is a port of TAPA's Verilator fast-cosim approach (github.com/tuna/tapa, MIT, commit d249ff9, `fpga-runtime/frt-cosim`) to plain Vitis HLS 2023.2 RTL. It needs no TAPA `.xo`, no Rust and no Bazel.

| File | What it does |
|---|---|
| `rtl_prep.py` | Port of TAPA's `generate_xilinx_fp_ip_models` and `fix_combinational_nba`. Each AMD `floating_point` IP (from its `*_ip.tcl`: add/sub/mul, fp32/fp64, `c_latency`) becomes a behavioral SV stand-in with the same AXIS ports and latency; the arithmetic is a DPI call. All other RTL is exactly what HLS generated. |
| `fp_dpi.c` | The DPI arithmetic: IEEE round-to-nearest-even, plus denormal flush-to-zero (as AMD cores do; TAPA's DPI does not flush). |
| `tb_gen.py` | C++ testbench. Its AXI memory model and AXI-Lite start sequence are ported from TAPA `tb_verilator.cpp.j2`, with one memory per `m_axi` bundle. It adds a cycle counter, a heartbeat (cycles/s), a cycle limit, and an early check of C writes against the golden. |
| `run_vcosim.py` | The flow: csynth (c2hls `hls_eval`) → bench data and golden (`../fastcosim/gen_data.cpp`, 1024³ `static_assert`) → RTL prep → Verilator 5.044 build → run → `C_out.bin` vs golden (\|diff\| > 0.001 counts as an error). |
| `sbatch_vcosim.sh` | One node, 16 CPUs, 128 GB. Work goes to `$SLURM_TMPDIR`; results go to `runs/NAME/` (`summary.json`, `simulate.log` updated live, `tb.cpp`, `ip_models/`). |

Verilator is v5.044, installed at `/scratch/asa582/tools/verilator-v5.044` (`sbatch_build_verilator.sh`). TAPA requires at least 5.044 for the Vitis `m_axi` master.

**Scope:** this is a correctness screen. Everything except the arithmetic inside AMD's floating-point cores is the real generated RTL: pipelines, dependence hazards, FIFOs, AXI. The memory model is functional, so cycle counts are not DRAM-accurate. Vitis cosim with the real IP is the final word on latency.

**Self-tests (16³):**

| Kernel | Result | RTL cycles (= csynth) |
|---|---|---:|
| good | Passed! | 64,774 |
| bad | Failed with 1 errors!, also caught by the early check | 65,030 |

Each took about 7 s to build and 0.1 s to simulate.
