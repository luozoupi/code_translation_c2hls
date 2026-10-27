# HLSmith Minimal

A small C-to-HLS workbench: translate a C kernel to C++, check it against an
independent C execution, and optionally explore measured HLS improvements.
One CLI provides `run`, `validate`, and `report`.

The default prompt profile is **`minimal-researcher-v1`**, a new profile for this
package. Results from it do not reproduce the submitted paper's experiments.

## Install

Use Python 3.10 or newer. The Python runtime has no third-party dependencies.
Native checks need `gcc` and `g++`; RTL checks also need Verilator and `make`.
From this project directory:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install .
```

## Start offline

The original scalar example and two saved responses run without an API key:

```sh
hlsmith run examples/scalar_mac/kernel.json \
  --responses examples/scalar_mac/responses.json \
  --max-calls 2 --out runs/offline
hlsmith report runs/offline
```

This uses the first passing candidate, then independently rechecks the selected
source. Original C and C testbenches compile as C; candidates compile as C++.
Saved responses are illustrative fixtures, not model experiment results.
Native-only runs establish output agreement on the supplied tests; they make no
hardware speedup claim. Each `--out` directory must be new.

An explicit candidate uses the same checks:

```sh
hlsmith validate examples/scalar_mac/kernel.json \
  --candidate examples/scalar_mac/candidate_01.cpp --out runs/native
```

Paths inside `kernel.json` resolve relative to that file. Public `headers` enter
generation prompts; the testbench and `test_files` are validation-only inputs.
Expected outputs and test vectors are never added to generation or repair prompts.

## Check supplied RTL with Verilator

```sh
hlsmith validate examples/scalar_mac/kernel.json \
  --candidate examples/scalar_mac/candidate_01.cpp \
  --rtl-config examples/scalar_mac/rtl.json --out runs/rtl
hlsmith report runs/rtl
```

The real fixture executes 12 test cases and reports `HLSMITH_CYCLES:12`.
Its RTL is **hand authored**, with no established binding to the C++ candidate.
The cycle count belongs to that RTL fixture, not to the translation above.
RTL and harness paths resolve relative to `rtl.json`.

This deliberately wrong fixture should fail with exit code 1:

```sh
hlsmith validate examples/scalar_mac/kernel.json \
  --candidate examples/scalar_mac/candidate_01.cpp \
  --rtl-config examples/scalar_mac/rtl-wrong.json --out runs/rtl-wrong
```

## Use DeepSeek

Set the key in your shell environment; keep it out of configuration and source:

```sh
export DEEPSEEK_API_KEY='your-key'
hlsmith run examples/scalar_mac/kernel.json \
  --max-api-usd 1 --max-calls 3 --out runs/deepseek
```

This command makes live requests. The default alias is `deepseek-flash`, with
thinking disabled. The declared release `DeepSeek-V4.1-Flash-1017` is recorded
metadata; the hosted alias can change and does not pin model weights.

A default live request reserves all 1,048,576 input tokens plus 8,192 output
tokens: about **$0.3244** at configured peak rates, a conservative estimate rather
than an invoice. HTTP retries are disabled. Missing or uncertain usage retains
the reservation and stops further requests. Runs preserve requests, responses,
and ledgers. No API calls were spent building or testing this project.

## Optional synthesis and optimization

| Tool | Role |
| --- | --- |
| Native C/C++ | Compare candidate outputs with independent original-C outputs. |
| Verilator | Execute supplied RTL and measure its harness-defined cycles. |
| External Vitis HLS | Produce HLS latency, clock, and resource estimates. |
| Optional LightningSim | Estimate dynamic trace latency using a Vitis HLS schedule. |

Verilator does not translate C to RTL. These checks do not establish physical
U280 implementation. LightningSim is an optional external tool, described below.

Optimization requires `--synthesize`, an available Vitis executable, and explicit
`vitis.part`, `vitis.clock_ns`, and `vitis.resource_limits` in `kernel.json`.
Merge this section into your configuration, then set the part and study budgets:

```json
{
  "vitis": {
    "executable": "vitis_hls",
    "part": "YOUR_FPGA_PART",
    "clock_ns": 5.0,
    "resource_limits": {"lut": 5000, "ff": 10000, "bram": 16, "dsp": 32}
  },
  "lightningsim": {
    "executable": "../../.tools/lightningsim/bin/lightningsim",
    "timeout": 120,
    "hls_root": "/path/to/Xilinx/Vitis_HLS/2023.2"
  }
}
```

These numbers illustrate per-kernel budgets, not device capacities. After setting
your configuration, a bounded saved-response optimization run is:

```sh
hlsmith run examples/scalar_mac/kernel.json \
  --responses examples/scalar_mac/responses.json --max-calls 3 \
  --synthesize --optimize 1 --out runs/hls-opt
```

Selection uses qualified HLS worst-case latency estimates under the declared
clock/resource budgets. CPU wall time is not the hardware objective.

## Optional LightningSim trace estimates

[LightningSim](https://github.com/sharc-lab/LightningSim) combines dynamic LLVM
execution traces with static HLS schedules to model cycles, pipeline stalls,
FIFO interactions, and deadlocks; see the [original paper](https://arxiv.org/abs/2304.11219).
This adapter records the top-module trace interval, which may span multiple calls.
It provides diagnostic estimates, not LLVM correctness or executed-RTL proof.
Reports keep `trace_estimate_cycles` separate from executed `rtl_cycles`.

The default Python package still has zero third-party runtime dependencies.
Install the optional tool in a separate Python 3.10 prefix from this project root
(skip this if the prefix already exists):

```sh
conda create --yes --prefix .tools/lightningsim \
  --solver libmamba --no-default-packages \
  --override-channels --strict-channel-priority --repodata-fn repodata.json \
  -c https://sharc-lab.github.io/LightningSim/repo -c conda-forge \
  python=3.10 lightningsim=0.2.6
```

The configuration above resolves its executable relative to
the example's `kernel.json`; an executable on `PATH` can be named
`"lightningsim"` instead. Set `hls_root` to your actual Vitis HLS installation,
or omit it after sourcing that installation's `settings64.sh` to set `XILINX_HLS`.

With Vitis and LightningSim configured, this pilot currently returns `incomplete_estimation`:

```sh
hlsmith run examples/array_sum/kernel.json \
  --responses examples/array_sum/responses.json --max-calls 1 \
  --synthesize --lightningsim --out runs/trace
```

For an existing Vitis solution, replace these workspace paths:

```sh
hlsmith validate examples/array_sum/kernel.json \
  --candidate examples/array_sum/candidate.cpp \
  --lightningsim-solution path/to/workspace/hls_project/solution \
  --lightningsim-workspace path/to/workspace --out runs/trace-existing
```

The estimator executes a self-contained workspace copy, bounded at 512 MiB,
10,000 files and 20,000 entries. Symlinks and external file dependencies are
unsupported. An existing solution's binding to the candidate remains unverified.
An otherwise passing check with an unavailable or partial estimate reports
`incomplete_estimation`; missing tool configuration is rejected early.
Trace cycles do not change candidate ranking: the HLS objective stays unchanged.
Native and Vitis HLS 2023.2 checks passed for both toys (`array_sum`: 142 HLS worst cycles).
LightningSim 0.2.6 timed out during schedule resolution on both; the memory
pilot stopped after 15 seconds, producing no trace estimate.
It reports `incomplete_estimation`. Compatibility is not validated,
and no LightningSim latency, accuracy, or speedup result is claimed.

## Small by design

`src/hlsmith_minimal/` has seven Python files: `cli.py` for commands, `core.py` for
the run/check/select loop, `model.py` for requests and accounting, `tools.py` for
native/RTL/synthesis tools, `lightningsim.py` for optional trace estimates, and
`__init__.py`/`__main__.py` for package entry points. One
`prompt.md` holds the researcher-facing instructions.

This separate package imports no legacy project modules and changes no existing
campaigns. Validation covers executed test outputs, not all possible inputs or
paper-wide performance equivalence. The [examples](examples/README.md) have their
own [MIT license](examples/LICENSE); a project-wide license has not been selected.
