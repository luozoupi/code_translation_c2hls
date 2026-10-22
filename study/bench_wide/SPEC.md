# bench_wide: strict testbenches for the 58-kernel Refined + Res.-Req study

Copies of `code_translation_c2hls/benchmarks/<bench>/` (read-only source; never edit it). The 10 MachSuite twins of
self-contained benches were dropped (aes_table, aes_tableless, fft_transpose, gemm_ncubed, md_knn, nw, sort_merge,
spmv_crs, stencil2D, viterbi). Only `testbench.cpp` (and, if needed, a new helper file next to it) may be changed here;
`plain.cpp`, the header, `metadata.json`, `hls_baseline.cpp`, `gold_hls_source.cpp` and data files stay as copied.
Keep the original as `testbench.orig.cpp` when you replace one.

## What a strict testbench must do

1. **Independent golden.** Compute the expected outputs with a reference implementation compiled into the testbench:
   copy the body of `plain.cpp`'s top function into the testbench as `<top>_ref` (static, same signature). Or, for
   MachSuite benches, use `check.data` if it covers every output. Never derive the golden from the kernel under test.
2. **Same inputs.** Generate inputs deterministically (fixed seed, or the bench's own init / input.data). The reference
   and the kernel under test get identical copies of every input and every in-out array.
3. **Poison pure outputs.** Every array that `plain.cpp` only writes (never reads before writing) is filled with a
   poison value before the kernel call: 12345.0 for float/double, 0x5A5A5A5A pattern for ints (or -1 when 0x5A.. is a
   legal value is unclear). Arrays that `plain.cpp` reads and writes (in-out) get the real initial data, identical for
   reference and kernel. Read plain.cpp carefully to classify each argument.
4. **Check every output element** of every output and in-out array (and scalar outputs) against the golden.
   Tolerance: integers exact; float `fabs(d-r) <= 1e-3 * fmax(1.0, fabs(r))`; double
   `fabs(d-r) <= 1e-6 * fmax(1.0, fabs(r))`. NaN in the kernel output is an error. If an algorithm legitimately admits
   several correct outputs (e.g. ties), check an invariant that pins correctness completely and say so in a comment.
5. **Report.** Count mismatches. On any mismatch print exactly `Failed with %d errors!` and `return 1;`. Otherwise
   print exactly `Passed!` and `return 0;`. Print at most the first 10 mismatches before that line.
6. Keep the kernel call exactly as the original testbench made it (same top name from metadata `translated_hls_top`,
   same argument types, header include) so the c2hls flow still links. Do not change the problem size.
7. Keep run time small (csim of the reference should finish in well under a minute on one core).

## Validation (required for every bench you touch)

With g++ (HLS include path `/scratch/asa582/tmp/c2hls/vitis_hls_2023.2_include` for ap_int/hls headers if needed),
build `testbench.cpp` + a kernel file + any support files the original build used (see metadata.json and
dataset_hls_csim.tcl), from a scratch dir under `/home/asa582/.claude/jobs/72130906/tmp/bench_wide_check/<bench>/`:
- **control 1 (must pass):** kernel = `plain.cpp`. Must print `Passed!` and exit 0.
- **control 2 (must fail):** kernel = a copy of plain.cpp whose top function returns immediately (writes nothing).
  Must print `Failed with N errors!` and exit 1.
- **control 3 (must fail):** kernel = plain.cpp with one plausible bug (e.g. last output element skipped, or a loop
  bound off by one). Must print `Failed with N errors!`.
Also build with `hls_baseline.cpp` as the kernel and report whether it passes (it is the flow's starting point; it
should pass; if it does not, report it, do not edit it).

Write `bench_wide/<bench>/TB_NOTES.md` (5–15 lines): outputs/in-outs classification, golden source, tolerance,
poison, the three control results and the hls_baseline result. Return a table with one row per bench.
