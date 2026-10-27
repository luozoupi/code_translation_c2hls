# Small HLSFactory-derived pilots

These three double-precision kernels exercise the standalone runner with small
arrays. They are new portability pilots, not reruns of the SIGMETRICS paper and
not measurements of fresh model-generated code.

| Kernel | Pilot dimensions | Preserved export dimensions | Workload |
| --- | --- | --- | --- |
| MVT | `N=8` | `N=120` | Two matrix-vector products with different access directions |
| ATAX | `M=6, N=8` | `M=116, N=124` | A matrix-vector product followed by its transpose product |
| Jacobi 1D | `N=10, TSTEPS=4` | `N=120, TSTEPS=40` | Repeated nearest-neighbor stencil updates |

Each `vendor/` directory preserves the four frozen public-C export files
(`plain.cpp`, header, testbench and metadata) without changing their bytes.
[manifest.json](manifest.json) records their source paths, sizes and SHA-256
hashes, bound to the original campaign manifest. These are existing C2HLS
exports labelled HLSFactory; the old metadata does not identify an exact
upstream HLSFactory commit. They must not be described as byte-identical copies
of the current upstream repository.

The separate `reference.c` and `kernel.h` adapters retain the kernel body and
double precision. They change the include name, provide a C11-compatible
interface with conditional C linkage for C++, and set the smaller dimensions.
The Jacobi coefficient remains `0.33333`. The new C testbenches use three
deterministic asymmetric, zero and boundary cases, and print all pointer arrays
to stdout with 17 significant digits. The historical testbenches remain in
`vendor/`; their input sizes, input values and stderr dump format are not used
as the new pilot contract.

`candidate.cpp` contains the same public kernel body. `responses.json` is a
saved portability baseline made from that source, explicitly labelled as such.
It contains no provider response and does not demonstrate LLM translation,
optimization quality or a speedup.

With Bambu and its compatible Verilator available, run one kernel using a new
output directory:

```sh
hlsmith run examples/hlsfactory/mvt/kernel-bambu.json \
  --responses examples/hlsfactory/mvt/responses.json \
  --max-calls 1 --backend bambu \
  --out runs/hlsfactory-bambu-pilot/mvt-example
```

Use the corresponding `atax` or `jacobi-1d` paths for the other kernels. The
native compile/run limits are 30/5 seconds; each Bambu job has a 120-second
limit. The 3.33 ns setting is a compiler scheduling request. No U280 fit,
physical clock frequency or board-performance claim follows from these runs.

`kernel-bambu.json` selects a separate testbench protocol adapter. During the
first real Bambu probes, automatic pointer-size inference described a vector
as one double and a matrix as one row. `testbench.bambu.c` registers each full
array extent with the installed MDPI `m_param_alloc` API before calling the
unchanged kernel. The sizes use array types such as `sizeof(double[N])` and
`sizeof(double[N][N])`, rather than `sizeof` on decayed pointer parameters.
This does not set `CUSTOM_VERIFICATION` or use `--tb-param-size`, which would
disable automatic top-level checking.

The hook is conditional on `__BAMBU_SIM__`. Both native execution and the
instrumented harness include `pilot_testbench.inc`, a private runtime alias of
the original `testbench.c`. This avoids duplicating or editing the three test
cases. `kernel.json`, the original testbench and the earlier failed runs remain
separate records. [native-extent-checks.json](native-extent-checks.json) confirms
identical native stdout hashes and verifies the hook's complete argument sizes
with a local stub; that stub check is not a real MDPI or RTL measurement.

Keep the two checks separate: the native check compares the candidate with the
original C adapter, while Bambu compares the candidate C/C++ with the generated
RTL. Successful checks cover the registered testbench calls. Preserve failures
and timeouts rather than changing shapes or data types within an existing run.

[native-checks.json](native-checks.json) records the local software check: all
three C11/C++17 pairs matched, and three deliberately incorrect early-return
candidates were rejected. It contains no Bambu or RTL measurements. Native
execution times are host observations and are not FPGA latency estimates.

## Source and license notices

The source algorithms come from PolyBench/C through HLSFactory-labelled local
exports. The [HLSFactory project](https://github.com/sharc-lab/HLSFactory)
states that its datasets retain their upstream licenses. The preserved exports
and these derived kernel adapters are not relabelled with this runner's MIT
license. See [LICENSE.PolyBench.txt](LICENSE.PolyBench.txt), the accompanying
[HLSFactory data license](LICENSE.HLSFactory-data.txt), and
[UPSTREAM-NOTICE.md](UPSTREAM-NOTICE.md). License download URLs and exact hashes
are recorded in the manifest.
