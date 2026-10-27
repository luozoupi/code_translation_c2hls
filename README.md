# HLSmith Nano

A small C-to-HLS workbench: translate C to C++, compare its outputs with an
independent C execution, and optionally compile it to RTL or explore measured
improvements. Three commands cover the workflow: `run`, `validate`, and `report`.

[`HLSmiths-nano`](https://github.com/luozoupi/code_translation_c2hls/tree/HLSmiths-nano)
is the lightweight development branch.
[`v8`](https://github.com/luozoupi/code_translation_c2hls/tree/v8), formerly
`hlsmith-clean`, retains the full research implementation and old `c2hls` CLI.
Nano keeps the existing `hlsmith` CLI and `hlsmith-minimal` package version
`0.3.0`. Its new `minimal-researcher-v1` prompt and results do not reproduce the
submitted paper's experiments.

## Install

Use Python 3.10 or newer, `gcc`, and `g++`. The Python runtime has no third-party
dependencies. For a fresh checkout:

```sh
git clone --branch HLSmiths-nano --single-branch https://github.com/luozoupi/code_translation_c2hls.git hlsmith-nano
cd hlsmith-nano
python3 -m venv .venv
. .venv/bin/activate
python -m pip install .
```

## Start offline

The scalar example includes two illustrative responses and needs no API key:

```sh
hlsmith run examples/scalar_mac/kernel.json \
  --responses examples/scalar_mac/responses.json \
  --max-calls 2 --out results/offline
hlsmith report results/offline
```

The default backend selects the first candidate that passes native checks, then
independently rechecks the selected source. Each `--out` directory must be new.
Saved responses are fixtures, not recorded model experiments.

To check your own candidate:

```sh
hlsmith validate examples/scalar_mac/kernel.json \
  --candidate examples/scalar_mac/candidate_01.cpp --out results/native
```

Paths in `kernel.json` resolve relative to that file. Public `headers` enter
generation prompts; the testbench, `test_files`, expected outputs, and test
vectors stay outside generation and repair prompts. Native checks establish
output agreement on supplied inputs, without a hardware performance claim.

## Compile to RTL with Bambu

With [Bambu 2024.10](https://github.com/ferrandi/PandA-bambu/releases/tag/2024.10),
Verilator, and `make` available, run a small HLSFactory-derived pilot:

```sh
export PATH="$PWD/.tools/bambu:$PATH"
hlsmith run examples/hlsfactory/mvt/kernel-bambu.json \
  --backend bambu --responses examples/hlsfactory/mvt/responses.json \
  --max-calls 1 --out results/mvt-check
hlsmith report results/mvt-check
```

The local optional launcher lives at `.tools/bambu/bambu`; another Bambu
installation can be selected in the configuration. Original-C/native checking
runs before Bambu compiles the candidate and checks generated RTL with Verilator.

The three small double-precision pilots passed three checked RTL invocations
each: MVT 4,626 cycles/call, ATAX 3,492, and Jacobi-1D 3,440. Their responses are
public-C portability baselines. These are functional plumbing tests, not fresh
LLM results, paper QoR, or U280 implementation evidence. See the
[pilot guide](examples/hlsfactory/README.md) and [recorded evidence](docs/evidence/README.md).

## Use DeepSeek

Set the key in your environment, then give live requests an explicit budget:

```sh
export DEEPSEEK_API_KEY='your-key'
hlsmith run examples/scalar_mac/kernel.json \
  --max-api-usd 1 --max-calls 3 --out results/deepseek
```

The default hosted alias is `deepseek-flash`, with thinking disabled. Recorded
release metadata does not pin hosted model weights. Each default request reserves
1,048,576 input tokens plus 8,192 output tokens: about $0.3244 at configured peak
rates. This is a conservative reservation, not an invoice. HTTP retries are off;
uncertain usage keeps its reservation and stops further requests. Saved-response
runs never fall back to the API.

## Optional backends and project layout

[Backend instructions](docs/backends.md) cover supplied RTL, Bambu, Vitis
synthesis/optimization, and optional LightningSim estimates. Verilator simulates
RTL; LightningSim uses a Vitis schedule. The current LightningSim setup timed
out on both toys and is not qualified for latency estimates. No backend here
establishes physical FPGA fit or board performance.

| Directory | Purpose |
| --- | --- |
| `src/hlsmith_minimal/` | Eight Python files and one researcher-facing prompt |
| `examples/` | Runnable scalar, array, and HLSFactory mini examples |
| `tests/` | Focused offline correctness and accounting checks |
| `docs/evidence/` | Preserved validation receipts and their scope |

See [test instructions](tests/README.md) and [example notes](examples/README.md).
New output belongs in `results/` (ignored by Git). Historical `runs/` and local
`.tools/` resolve through compatibility links to external storage; see the
[cleanup record](docs/CLEANUP.md). This package imports no legacy project modules. Toy examples have an [MIT license](examples/LICENSE);
HLSFactory imports retain their [upstream notices](examples/hlsfactory/UPSTREAM-NOTICE.md).
A project-wide license has not been selected.
