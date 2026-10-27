# Scalar multiply-accumulate

`scalar_mac/` is an original, small example written for this project. The files
are MIT licensed, copyright 2026 luozoupi. They contain no copied legacy kernels,
datasets, experiment results, or model responses.

The C interface is `uint32_t scalar_mac(uint32_t x, uint32_t y, uint32_t bias)`.
The result is `x * y + bias`, modulo 2^32. The reference uses repeated addition;
the test cases keep `y` at 15 or below. Some cases deliberately exercise
unsigned wraparound. Compile `reference.c` and `testbench.c` as C, and each
candidate as C++. The conditional `extern "C"` declaration in `kernel.h` keeps
the ABI identical.
Only the reference source and `kernel.h` belong in generation prompts.
`testbench.c` and `test_vectors.h` are validation fixtures.

`responses.json` contains two illustrative saved responses. The first translates
the loop to C++; the second replaces the loop with multiplication. They are
fixtures, not recorded provider outputs or measured optimization results. The
default configuration uses native validation only and selects a passing
candidate. The second response can support a later measured optimization demo.

`scalar_mac.v` and `sim_main.cpp` exercise the Verilator backend with the same
arithmetic and private test vectors. The RTL is hand authored. It was not
synthesized from either C++ candidate, so its cycle count does not establish a
candidate's hardware performance. `provenance.json` records that distinction.

The RTL has scalar `x`, `y`, `bias`, and `result` ports, plus `clk`, synchronous
active-low `rst_n`, `start`, and `done`. A rising edge with `start` high produces
the result and asserts `done`. The harness first applies one reset edge, then
counts processing edges from each start through its done edge. Reset is excluded
from `HLSMITH_CYCLES:<integer>`. Each transaction has an eight-cycle wait limit.
There are no vendor IP blocks, simulator delays, or SystemC dependencies.

Both native and RTL harnesses print one deterministic `HLSMITH_RESULT` line.
The RTL harness also prints the separate cycle marker. `candidate_wrong.cpp`
and `rtl-wrong.json` are intentional failures: they compile and run, but add one
to every answer. They demonstrate why tool success alone cannot qualify a result.
