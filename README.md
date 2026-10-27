# HLSmith Nano

Translate a plain C kernel to C++, check it against an independent C execution,
and optionally compile it to RTL or improve its measured latency. Three commands
cover the workflow: `run`, `validate`, and `report`.

Version 0.4.0 on [`HLSmiths-nano`](https://github.com/luozoupi/code_translation_c2hls/tree/HLSmiths-nano)
is a standalone package with a new `minimal-researcher-v1` prompt.
[`v8`](https://github.com/luozoupi/code_translation_c2hls/tree/v8) retains the full
research implementation. Nano results do not reproduce the paper's experiments.

## Install

Use Python 3.10+, `gcc`, and `g++`. The Python runtime has no third-party
dependencies. Bambu RTL checks need Bambu 2024.10, Verilator, and `make`.
Clang is needed only to generate a testbench or add pointer-extent wrappers.

```sh
git clone --branch HLSmiths-nano --single-branch https://github.com/luozoupi/code_translation_c2hls.git hlsmith-nano
cd hlsmith-nano
python3 -m venv .venv
. .venv/bin/activate
python -m pip install .
```

## Start with your C source

With the tools on `PATH` and a model key in your environment, supply a C/C++
testbench that calls the top function and prints the results to compare:

```sh
export DEEPSEEK_API_KEY='your-key'
hlsmith run kernel.c --top kernel --testbench tb.c \
  --max-api-usd 1 --max-calls 3 --out results/kernel
hlsmith report results/kernel
```

Plain C `run` defaults to Bambu: native checking runs first, then Bambu generates
RTL and checks it with Verilator. Use `--backend native` for software checks or
`--backend vitis --lightningsim --config target.json` for Vitis/trace estimates.
Each `--out` directory must be new.

Omit `--testbench` to generate three deterministic input cases. For a kernel
with array parameter `a` and scalar loop bound `n`, declare its full shape and
pin the bound to your intended domain:

```sh
hlsmith run kernel.c --top kernel --size a=8x8 --value n=8 \
  --max-api-usd 1 --max-calls 3 --out results/generated-tests
```

Repeat `--size` for every pointer parameter, even when the source declares array
dimensions. Repeat `--value` to fix scalar inputs, or `--header path.h` for public
headers. Generated tests support numeric scalars and fixed-size arrays with
separate buffers. Defaults are small positive values, not inferred application
preconditions. Aliasing, complex interfaces, and mutable state need your own
testbench. Supplied testbenches keep their cases; `--size` can add Bambu pointer
extent registration without changing those cases.

Generated inputs are checked against original C with address/undefined-behavior
sanitizers before any API request. Tests observe actual return values and all
array elements, never generated expected answers. The harness is frozen across
candidates and final checking. This is limited test coverage, not all-input proof.
Public headers enter prompts; testbench-only files and oracle outputs stay private.

To check an existing candidate without a model call:

```sh
hlsmith validate kernel.c --top kernel --testbench tb.c \
  --candidate candidate.cpp --out results/check
```

`validate` defaults to native; add `--backend bambu` for generated-RTL checks.
Each run retains a reusable `inputs/kernel.json` and copied, hashed inputs.

## Run offline or use detailed settings

Existing JSON projects remain supported and default to native checking:

```sh
hlsmith run examples/scalar_mac/kernel.json \
  --responses examples/scalar_mac/responses.json \
  --max-calls 2 --out results/offline
```

Saved responses are illustrative fixtures, never an API fallback. For plain C,
`--config target.json` supplies provider, validation, backend, and private
`test_files` settings. See [backend configuration](docs/backends.md) and the
[HLSFactory mini pilots](examples/hlsfactory/README.md).

## What the results mean

Bambu/Verilator records checked RTL cycles. Vitis records HLS estimates;
LightningSim records separate trace estimates, without asserting RTL correctness.
Host checks with unmodified LightningSim 0.2.6 and Vitis 2023.2 produced scalar
and array trace estimates of 11 and 163 cycles. Backend estimates have different
scopes and are not interchangeable. No result establishes physical FPGA fit or
board performance; [evidence records](docs/evidence/README.md) state their limits.

Live DeepSeek requests require `--max-api-usd`. The default alias is
`deepseek-flash`, thinking disabled; release metadata does not pin hosted weights.
Each default request reserves about $0.3244 at configured peak rates, covering
1,048,576 input and 8,192 output tokens. This is a reservation, not an invoice.
HTTP retries are off; uncertain usage retains its reservation and stops requests.

`src/hlsmith_minimal/` has nine Python files and one prompt. `examples/` holds
runnable inputs, `tests/` holds six focused modules, and `docs/evidence/` preserves
receipts. See [test instructions](tests/README.md). New output belongs in ignored
`results/`; historical `runs/` and optional `.tools/` use external compatibility
links described in the [cleanup record](docs/CLEANUP.md). Toy examples have an
[MIT license](examples/LICENSE); HLSFactory imports retain their
[upstream notices](examples/hlsfactory/UPSTREAM-NOTICE.md). A project-wide license
has not been selected.
