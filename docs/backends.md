# Optional backends

Run the commands below from the project root after installing the package.
Every output directory must be new. Native validation compares original C with
the candidate C++ on the supplied testbench before optional hardware checks.

| Backend | What its result establishes |
| --- | --- |
| Native C/C++ | Candidate output agreement with an independent original-C execution |
| Supplied RTL + Verilator | Output agreement and harness-defined cycles for that RTL |
| Bambu + Verilator | Candidate C/C++ compiled to RTL and checked by generated co-simulation |
| Vitis HLS | HLS latency, clock, and resource estimates under configured budgets |
| LightningSim | Diagnostic trace latency estimate using a Vitis HLS schedule |

These checks cover executed test cases. They do not establish physical U280
implementation, timing closure, board performance, or paper-wide equivalence.
Verilator simulates RTL; it does not translate C to RTL.

## Supplied RTL with Verilator

Install Verilator and `make`, then check the supplied scalar fixture:

```sh
hlsmith validate examples/scalar_mac/kernel.json \
  --candidate examples/scalar_mac/candidate_01.cpp \
  --rtl-config examples/scalar_mac/rtl.json --out results/rtl
hlsmith report results/rtl
```

The fixture executes 12 test cases and reports `HLSMITH_CYCLES:12`. Its RTL is
hand authored, with no established binding to the C++ candidate. Those cycles
belong to that RTL fixture. RTL and harness paths resolve relative to `rtl.json`.
A supplied harness must print numeric result tokens matching the native oracle.
An optional single `HLSMITH_CYCLES:<integer>` line records its counted cycles;
missing cycles leave functional checking separate from latency qualification.

## Bambu-generated RTL

The adapter supports [Bambu 2024.10](https://github.com/ferrandi/PandA-bambu/releases/tag/2024.10)
and Verilator. The optional local `.tools/bambu/bambu` launcher uses an extracted
AppImage without FUSE. Put its directory on `PATH` or configure
`bambu.executable` in your kernel configuration.

```sh
export PATH="$PWD/.tools/bambu:$PATH"
hlsmith run examples/hlsfactory/mvt/kernel-bambu.json \
  --backend bambu --responses examples/hlsfactory/mvt/responses.json \
  --max-calls 1 --out results/mvt-check
```

`validate --backend bambu` checks an explicit candidate through the same native
and generated-RTL gates. The adapter uses a 64-bit host ABI and
`OMP_NUM_THREADS=1`. Conditional testbench `m_param_alloc` hooks declare complete
pointer extents. Automatic C/RTL value checking remains enabled at zero ULP;
neither `--tb-param-size` nor `CUSTOM_VERIFICATION` is used.

The three [HLSFactory mini pilots](../examples/hlsfactory/README.md) retain
double precision but use reduced shapes and new test cases. Their saved responses
are public-C portability baselines, not provider outputs. The
[pilot receipt](evidence/HLSFACTORY_PILOT.json) preserves successful runs and
earlier setup failures.

Optional `--optimize N` ranks the sum of executed RTL cycles for the fixed
testbench. This is an experimental cycle-only objective. Resource eligibility,
physical timing fit, and physical implementation stay unmeasured. A 3.33 ns
clock request is a scheduling input, not a measured physical frequency.

## Vitis synthesis and optimization

Use `--backend vitis` or its alias `--synthesize`. Optimization requires the
executable and `vitis.part`, `vitis.clock_ns`, and `vitis.resource_limits` in the
kernel configuration. Merge the following settings into your configuration and
replace the part and budgets for your study:

```json
{
  "vitis": {
    "executable": "vitis_hls",
    "part": "YOUR_FPGA_PART",
    "clock_ns": 5.0,
    "resource_limits": {"lut": 5000, "ff": 10000, "bram": 16, "dsp": 32}
  },
  "lightningsim": {
    "executable": "lightningsim",
    "timeout": 120,
    "hls_root": "/path/to/Xilinx/Vitis_HLS/2023.2"
  }
}
```

These are illustrative per-kernel budgets, not device capacities. After setting
your configuration, a bounded saved-response optimization run is:

```sh
hlsmith run examples/scalar_mac/kernel.json \
  --responses examples/scalar_mac/responses.json --max-calls 3 \
  --synthesize --optimize 1 --out results/hls-opt
```

Selection uses qualified HLS worst-case latency estimates under the declared
clock and resource budgets. CPU wall time is not the hardware objective.

## LightningSim trace estimates

[LightningSim](https://github.com/sharc-lab/LightningSim) combines dynamic LLVM
execution traces with static HLS schedules to model cycles, pipeline stalls,
FIFO interactions, and deadlocks; see the [paper](https://arxiv.org/abs/2304.11219).
The adapter records the top-module trace interval, which may span multiple calls.
It is a diagnostic estimate, not a correctness check or executed-RTL proof.
Reports keep `trace_estimate_cycles` separate from executed `rtl_cycles`.

The package retains zero third-party Python runtime dependencies. Install the
optional tool in a separate Python 3.10 prefix from the project root, unless
that prefix already exists:

```sh
conda create --yes --prefix .tools/lightningsim \
  --solver libmamba --no-default-packages \
  --override-channels --strict-channel-priority --repodata-fn repodata.json \
  -c https://sharc-lab.github.io/LightningSim/repo -c conda-forge \
  python=3.10 lightningsim=0.2.6
```

Use `"lightningsim"` if the executable is on `PATH`. Otherwise, its configured
path resolves relative to `kernel.json`; for the two toy examples the local
prefix is `../../.tools/lightningsim/bin/lightningsim`. Set `hls_root` to your
Vitis HLS installation, or omit it after sourcing its `settings64.sh` to set
`XILINX_HLS`.

With both tools configured, request an estimate alongside synthesis:

```sh
hlsmith run examples/array_sum/kernel.json \
  --responses examples/array_sum/responses.json --max-calls 1 \
  --synthesize --lightningsim --out results/trace
```

For an existing Vitis solution, replace these workspace paths:

```sh
hlsmith validate examples/array_sum/kernel.json \
  --candidate examples/array_sum/candidate.cpp \
  --lightningsim-solution path/to/workspace/hls_project/solution \
  --lightningsim-workspace path/to/workspace --out results/trace-existing
```

The estimator uses a self-contained workspace copy, bounded at 512 MiB,
10,000 files, and 20,000 entries. Symlinks and external file dependencies are
unsupported. An existing solution's binding to the candidate remains unverified.
An otherwise passing check with an unavailable or partial estimate reports
`incomplete_estimation`; missing tool configuration is rejected before generation.
Trace estimates do not change candidate ranking: the HLS objective stays unchanged.

Native and Vitis HLS 2023.2 checks passed for both toys (`array_sum`: 142 HLS worst
cycles). LightningSim 0.2.6 timed out resolving schedules on both: 60 seconds for
the scalar and 15 seconds for the memory pilot, with no trace estimate. The
recorded workflow status is `incomplete_estimation`. Compatibility is not
validated; no LightningSim latency, accuracy, or speedup result is claimed.
See the [preserved validation receipt](evidence/LIGHTNINGSIM_VALIDATION.json).
