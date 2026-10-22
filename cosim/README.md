# C/RTL cosim of the top designs (1024³ bench)

- `run_cosim_one.py NAME KERNEL.cpp`: c2hls `hls_eval.run_cosim` (full `cosim_design` inside the Vitis container, `-trace_level none`, so no waveform) on the n1024 bench.
- The testbench is the bench's `testbench.cpp` with reporting changes only: `static_assert(I == 1024 && J == 1024 && K == 1024)`, a `TB_SIZE` line, a nonzero return on mismatch, and one verdict line per run in `tb_results.txt`. Data, golden model and the 0.001 tolerance are unchanged.
- **Correct** means:
  - two verdict lines, the second being the RTL-output check, with `err=0` at 1024³;
  - and Vitis reports `C/RTL co-simulation finished: PASS`.
- **What's kept** per design in `NAME/`: `cosim_summary.json`, `tb_results.txt`, `cosim_log_excerpt.txt`, the csynth and cosim reports (`lat.rpt`, `*_cosim.rpt`), the kernel and the testbench. The Vitis work directory is deleted after each run.
- `sbatch_cosim.sh`: one node, 16 CPUs, 128 GB, 7 days. Job IDs are in `jobs.txt`.
- **Self-test at 16³** (`selftest/`): the bench's naive kernel passes (two verdicts, PASS, 70,884 cycles). A kernel with one wrong element fails (`Failed with 1 errors!`, cosim aborted).
- **xelab crash fix:** the first attempt crashed in `xelab` (`XSIM 43-3316 SIGSEGV`) on large designs, before any simulation (logs in `failed_first_attempt/`). The jobs now use c2hls's `xelab -mt off` path (`C2HLS_COSIM_XELAB_MT_OFF=1`), and `run_cosim_one.py` runs the resulting `sim.sh` inside the Vitis container (c2hls runs it outside, where `vitis_hls` is missing). The 16³ self-test passes on this path (the last of 3 verdicts is the RTL check), and the bad kernel is still rejected.
