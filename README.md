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

Before each request, accounting reserves the full 1,048,576-token input context
plus the 8,192-token output allowance: about **$0.3244 per request** at the default
peak rates. This is a conservative estimate, not a provider invoice. There are
no automatic request retries. Missing or uncertain usage retains the reservation
and stops further requests. Request, response, and ledger records stay in the
run directory. No API calls were spent building or testing this project.

## Optional synthesis and optimization

| Tool | Role |
| --- | --- |
| Native C/C++ | Compare candidate outputs with independent original-C outputs. |
| Verilator | Execute supplied RTL and measure its harness-defined cycles. |
| External Vitis HLS | Produce HLS latency, clock, and resource estimates. |

Verilator does not translate C to RTL. The Vitis wrapper was not live-tested in
this build. LightningSim is not bundled; a future optional trace backend would
still require Vitis. None of these checks establish physical U280 implementation.

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

## Small by design

`src/hlsmith_minimal/` has six Python files: `cli.py` for commands, `core.py` for
the run/check/select loop, `model.py` for requests and accounting, `tools.py` for
external tools, and `__init__.py`/`__main__.py` for package entry points. One
`prompt.md` holds the researcher-facing instructions.

This separate package imports no legacy project modules and changes no existing
campaigns. Validation covers executed test outputs, not all possible inputs or
paper-wide performance equivalence. The [examples](examples/README.md) have their
own [MIT license](examples/LICENSE); a project-wide license has not been selected.
