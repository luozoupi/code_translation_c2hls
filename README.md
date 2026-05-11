# C-to-HLS Code Translation Pipeline

A multi-agent LLM-driven pipeline that translates plain C/C++ kernels into Xilinx
Vitis HLS optimised code, validates the output through Vitis HLS synthesis,
optionally measures kernel runtime via XSIM hardware emulation (`make check
TARGET=hw_emu`), and scores quality against ground-truth HLS baselines.

## Overview

The pipeline takes pragma-free C code (derived from known-good HLS benchmarks
with pragmas stripped) and asks an LLM to re-introduce HLS optimisations in
phases. The generated code is synthesised with Vitis HLS, optionally
co-simulated for correctness, optionally driven through `v++` hardware
emulation against the rodinia-hls-nova testbenches, and scored against the
ground-truth HLS baseline using a 9-metric rubric.

```
gold_hls_source.cpp ──(strip pragmas)──> plain.cpp ──(LLM agents)──> generated HLS
        │                                                                    │
        ├──(synthesize as ground truth)──> GT report  <──(rubric)──── gen csynth report
        └──(reference hw_emu)──>          ref cycles  <──(direct)──── XSIM kernel cycles
```

The orchestrator runs three coordinating agents:
- **TranslatorAgent** — Phase A compile-check + Phase B initial translation
- **SynthesisAgent** — Phase B synth/csim/cosim + repair loop with structured
  feedback (per-attempt error history, repair guidance, profile-bottleneck
  signals, regression guard, revert-on-streak)
- **QualityRepairAgent** — post-synthesis quality-driven candidate generation
  to close the gap to ground truth

After the agentic loop completes, the optional **hw_emu post-step** runs the
final kernel through `v++` for an authoritative XSIM cycle measurement that
matches what `make check TARGET=hw_emu` would produce on a deployed kernel.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Benchmark Preparation](#benchmark-preparation)
4. [Running Translations](#running-translations)
5. [Verifying Nova Benchmarks via Direct Vitis Runs](#verifying-nova-benchmarks-via-direct-vitis-runs)
6. [Evaluation & Scoring](#evaluation--scoring)
7. [JSONL Export & Comparison](#jsonl-export--comparison)
8. [HTML Report Generation](#html-report-generation)
9. [Pipeline Architecture](#pipeline-architecture)
10. [Benchmark Corpus](#benchmark-corpus)
11. [File Reference](#file-reference)
12. [Troubleshooting](#troubleshooting)

---

## Prerequisites

| Component | Used for | Required for |
|---|---|---|
| **Vitis HLS 2023.2** (or compatible) | `csynth_design`, `csim_design`, `cosim_design` | All runs |
| **Python 3.10+** | Orchestrator + helpers | All runs |
| **g++** | Phase A compile-check | All runs |
| **LLM backend** (Anthropic / OpenAI / vLLM) | Phase B translation | LLM runs only |
| **XRT 2023.2** (user-mode tarball OK) | `v++` host runtime | hw_emu |
| **U280 dev platform** (`xilinx_u280_gen3x16_xdma_1_202211_1.xpfm`) | `v++` link target | hw_emu |
| **Khronos OpenCL headers** (`CL/cl.h`, `CL/cl2.hpp`) | host program compile | hw_emu |
| **glibc multi-arch headers** (`/usr/include/x86_64-linux-gnu`) | gcc multilib lookup | hw_emu (Ubuntu 22.04) |

The pipeline runs end-to-end **without** XRT / platform / OpenCL headers — those
are only needed to invoke `make check TARGET=sw_emu` / `TARGET=hw_emu` on the
nova benchmarks. csynth/csim/cosim via `vitis_hls` directly works with just the
HLS install.

---

## Environment Setup

### Step 1: Install Vitis HLS

Download the **Vitis Unified Installer** from
[Xilinx Downloads](https://www.xilinx.com/support/download/index.html/content/xilinx/en/downloadNav/vitis.html)
(requires a free AMD account).

Headless install, Vitis HLS only + Virtex UltraScale+ HBM device family:

```bash
chmod +x FPGAs_AdaptiveSoCs_Unified_2023.2_*_Lin64.bin
./FPGAs_AdaptiveSoCs_Unified_2023.2_*_Lin64.bin --noexec --keep --nox11 \
    --target /tmp/xinstall_2023.2_extract
# Generate auth token (interactive — needs AMD email + password):
/tmp/xinstall_2023.2_extract/xsetup -b AuthTokenGen
# Edit install_2023.2_config.txt to enable only "Virtex UltraScale+ HBM:1"
# (covers xcu280 / xcu50). Set Destination=/path/to/Xilinx
/tmp/xinstall_2023.2_extract/xsetup -a XilinxEULA,3rdPartyEULA -b Install \
    -c install_2023.2_config.txt
```

A pre-tested install_config.txt template ships in
[install_2023.2_config.txt](install_2023.2_config.txt).

### Step 2: Install XRT + U280 Platform (optional, for hw_emu)

Skip this step if you don't need `make check TARGET=sw_emu/hw_emu` on the nova
benches. csynth/csim/cosim via `vitis_hls` work without XRT.

```bash
# XRT user-mode (no sudo needed; emulation doesn't need the kernel module):
mkdir -p /path/to/XRT_2023.2 && cd /path/to/XRT_2023.2
curl -L -o xrt.deb \
  'https://www.xilinx.com/bin/public/openDownload?filename=xrt_202320.2.16.204_22.04-amd64-xrt.deb'
dpkg-deb -x xrt.deb .

# U280 dev platform (.xpfm + hw_emu.xsa). Get the .deb from your AMD account
# at https://www.xilinx.com/products/boards-and-kits/alveo/u280.html#deploy
mkdir -p /path/to/U280_PLATFORM && cd /path/to/U280_PLATFORM
dpkg-deb -x /path/to/xilinx-u280-gen3x16-xdma-1-202211-1-dev_*_all.deb .

# Khronos OpenCL headers (open source, no auth):
mkdir -p /path/to/opencl_headers/CL && cd /path/to/opencl_headers/CL
for f in cl.h opencl.h cl_platform.h cl_ext.h cl_egl.h cl_d3d10.h \
         cl_d3d11.h cl_dx9_media_sharing.h cl_gl.h cl_gl_ext.h \
         cl_layer.h cl_half.h cl_icd.h cl_version.h; do
  curl -sL -O "https://raw.githubusercontent.com/KhronosGroup/OpenCL-Headers/main/CL/$f"
done
curl -sL -O "https://raw.githubusercontent.com/KhronosGroup/OpenCL-CLHPP/v2.0.12/include/CL/cl2.hpp"
```

Then point [scripts/setup_emu_env.sh](scripts/setup_emu_env.sh) at your
install paths (the file is short; see the `XRT_DIR`, `PLATFORM_REPO_PATHS`,
`CPLUS_INCLUDE_PATH` lines).

### Step 3: Source the environment

```bash
# Always required
source /path/to/Xilinx/Vitis/2023.2/settings64.sh

# Only required for hw_emu
source scripts/setup_emu_env.sh
```

### Step 4: Python environment

```bash
conda create -n c2hls python=3.10 -y
conda activate c2hls
pip install -r requirements.txt
```

### Step 5: Configure LLM access

Create a `.env` file at the repo root:

```bash
# Anthropic Claude
ANTHROPIC_API_KEY=sk-ant-...

# OpenAI (optional)
OPENAI_API_KEY=sk-...

# Local vLLM (optional)
OPENAI_BASE_URL=http://127.0.0.1:8000/v1
```

The pipeline auto-detects model class from the model id — Claude models go
through Anthropic, GPT models go through OpenAI, anything else through your
vLLM endpoint.

### Step 6: Project environment variables

| Variable | Default | Purpose |
|---|---|---|
| `C2HLS_VITIS_SETTINGS` | `/mnt/data/luo00466/Xilinx/2025.2/Vitis/settings64.sh` | Path to Vitis HLS `settings64.sh` sourced before every synth call |
| `C2HLS_PART` | `xc7a100t-csg324-1` | Target FPGA part id (e.g. `xcu280-fsvh2892-2L-e`) |
| `C2HLS_CLOCK_NS` | `4` | Target clock period for csynth `create_clock` (3.33 = 300 MHz) |
| `C2HLS_FLOW_TARGET` | `vitis` | Vitis HLS solution flow: `vitis` (kernel/v++ flow) or `vivado` (raw IP). Use `vitis` for cross-tool comparison against v++ deployment numbers. |
| `C2HLS_HW_EMU_FINAL` | `0` | When `1`, run a final hw_emu measurement on the orchestrator's accepted code post-completion (slow, +30 min/bench) |
| `C2HLS_HW_EMU_CLOCK_NS` | `3.33` | XSIM clock period for cycle conversion. Use 3.33 for the U280 platform; independent from `C2HLS_CLOCK_NS`. |
| `C2HLS_HW_EMU_TIMEOUT` | `21600` | Wall-time budget for one `make check TARGET=hw_emu` (seconds) |
| `C2HLS_SYNTH_TIMEOUT` | `1200` | Wall-time budget for one `csynth_design` |
| `C2HLS_CSIM_TIMEOUT` | `180` | Wall-time budget for `csim_design` |
| `C2HLS_COSIM_TIMEOUT` | `1200` | Wall-time budget for `cosim_design` |
| `C2HLS_MODEL` | `nvidia/OpenCodeReasoning-Nemotron-1.1-32B` | Default LLM model id (overridden by `--model`) |
| `C2HLS_TRANSLATOR_MODEL` | (= `C2HLS_MODEL`) | Override only the TranslatorAgent's model |
| `C2HLS_SYNTHESIS_MODEL` | (= `C2HLS_MODEL`) | Override only the SynthesisAgent's model |
| `C2HLS_QUALITY_REPAIR_MODEL` | (= `C2HLS_MODEL`) | Override only the QualityRepairAgent's model |
| `C2HLS_QUALITY_REPAIR_TURNS` | `2` | Max candidate attempts in the quality-repair loop |
| `C2HLS_STEP_REGRESSION_THRESHOLD` | `1.10` | Multistep regression-guard threshold (1.10 = >10% latency increase triggers retry-then-revert) |
| `C2HLS_SYNTH_REVERT_THRESHOLD` | `0` | Revert-on-streak: N consecutive same-class errors trigger reversion to last-good (0 = disabled) |
| `C2HLS_VERIFY_RUNS` | `1` | When >1, repeat csynth N times and average for stability measurement |

A typical `.env` for U280 + 2023.2 + Claude:

```bash
ANTHROPIC_API_KEY=sk-ant-...
C2HLS_VITIS_SETTINGS=/mnt/data/luo00466/Xilinx/Vitis/2023.2/settings64.sh
C2HLS_PART=xcu280-fsvh2892-2L-e
C2HLS_CLOCK_NS=3.33
C2HLS_FLOW_TARGET=vitis
C2HLS_MODEL=claude-haiku-4-5-20251001
```

---

## Benchmark Preparation

### Pre-built corpus (recommended)

The `benchmarks/` directory ships **21 prepared benchmarks**:

- 9 from rodinia-hls (StreamCluster, hotspot, kmeans, knn, lavaMD, lud, nw,
  pathfinder, srad)
- 8 from ML4Accel-Dataset (aes, fft, gemm_ncubed, md_knn, sort_merge,
  spmv_crs, stencil2D, viterbi)
- 4 from rodinia-hls-nova (cfd_flux, cfd_step_factor, lc_gicov, lc_mgvf)

No preparation step is needed — skip to [Running Translations](#running-translations).

### Regenerating from upstream (optional)

```bash
# rodinia-hls (legacy + nova)
git clone https://github.com/UCLA-VAST/rodinia-hls.git /home/$USER/rodinia-hls
git clone https://your.git.host/rodinia-hls-nova.git /home/$USER/rodinia-hls-nova
git clone https://github.com/UIUC-ChenLab/ML4Accel-Dataset.git /home/$USER/ML4Accel-Dataset

# Edit the source-path constants near the top of:
#   prepare_benchmarks.py        (legacy rodinia + ML4Accel)
#   prepare_nova_benchmarks.py   (nova: cfd, leukocyte sub-kernels)

python prepare_benchmarks.py
python prepare_nova_benchmarks.py
```

Each benchmark dir gets:
- `gold_hls_source.cpp` — original upstream HLS code
- `hls_baseline.cpp` — localised gold (include paths rewritten for our work_dir)
- `plain.cpp` — gold with `#pragma HLS`, `extern "C"`, and `ap_int` includes stripped — **the LLM input**
- `<bench>.h` — header (also localised)
- `testbench.cpp` — for csim/cosim (when supported by upstream)
- `metadata.json` — provenance + variant list + sha256 digest
- `hls_<bench>_<step>.cpp` — per-step GT variants (rodinia/nova only) for multistep mode

---

## Running Translations

### Single-shot mode

The translator does Phase A → Phase B → Phase C in one pass:

```bash
source $C2HLS_VITIS_SETTINGS
python c2hls.py --bench aes --model claude-haiku-4-5-20251001 --turns 3
```

Output lands at `results/aes/`:
- `aes_generated.cpp` — final accepted HLS code
- `aes_synth_report.json` — Vitis csynth report
- `aes_results.json` — Phase C comparison + run attribution + (optional) hw_emu
- `aes_history.json` — full LLM conversation transcript with model attribution

### Multi-step mode

Each optimisation step (tiling → pipeline → unroll → doublebuffer →
coalescing) is applied incrementally; each step's output is compared against
the corresponding rodinia variant at that step:

```bash
python c2hls.py --bench knn --multistep --model claude-haiku-4-5-20251001
```

Output at `results_multistep/knn/`:
- `knn_multistep_results.json` — top-level results with per-step records
- `steps/<i>_<step_name>.cpp` — accepted code at each step
- `steps/<i>_<step_name>_report.json` — per-step synth + csim + GT pairing

### Multi-step with hw_emu post-step

Adds an authoritative XSIM cycle measurement on the final kernel:

```bash
source scripts/setup_emu_env.sh   # XRT + U280 platform + OpenCL headers
export C2HLS_HW_EMU_FINAL=1
python c2hls.py --bench pathfinder --multistep --model claude-haiku-4-5-20251001
```

The hw_emu post-step adds `hw_emu` to the saved results JSON with
`kernel_runtime_us`, `kernel_runtime_cycles`, `passed`, etc.

### Run all 21 benchmarks

```bash
python c2hls.py --all --multistep --model claude-haiku-4-5-20251001
```

### Per-agent model selection

Run different models on different agents (e.g. cheap haiku for translation,
sonnet for repair):

```bash
export C2HLS_TRANSLATOR_MODEL=claude-haiku-4-5-20251001
export C2HLS_SYNTHESIS_MODEL=claude-haiku-4-5-20251001
export C2HLS_QUALITY_REPAIR_MODEL=claude-sonnet-4-6
python c2hls.py --bench knn
```

### CLI reference

| Flag | Default | Description |
|---|---|---|
| `--bench NAME` | (required unless `--all`) | Benchmark to run |
| `--all` | — | Run every benchmark in `benchmarks/index.json` |
| `--multistep` | off | Use the per-step optimisation chain instead of single-shot |
| `--steps tiling,pipeline,...` | DEFAULT_OPT_STEPS | Custom step order for `--multistep` |
| `--model ID` | `$C2HLS_MODEL` | LLM model id (Anthropic / OpenAI / vLLM auto-routed by name) |
| `--turns N` | 3 | Max repair attempts per phase/step |
| `--quality-repair-turns N` | `$C2HLS_QUALITY_REPAIR_TURNS` (=2) | Max QualityRepair candidates |
| `--output-dir PATH` | `results/` or `results_multistep/` | Where to save artifacts |

---

## Verifying Nova Benchmarks via Direct Vitis Runs

This validates that your local Vitis 2023.2 + XRT + U280 platform install
reproduces the upstream rodinia-hls-nova reference's `sw_emu` and `hw_emu`
results bit-identically — no LLM in the loop. Use it after a fresh install
to confirm the toolchain works, or as ground-truth data for the agentic
pipeline to compare against.

### Reference data

Pre-computed reference numbers live in `results/references_philip/`:
- `sw_emu_vitis_2023.2__device_xilinx_u280_gen3x16_xdma_1_202211_1.jsonl`
- `hw_emu_vitis_2023.2__device_xilinx_u280_gen3x16_xdma_1_202211_1.jsonl`

Both follow the schema-1.0 record format (sw_run / rtl_sim payloads); the
hw_emu reference includes per-variant `kernel_runtime_cycles`,
`kernel_runtime_us`, and `kernel_clock_freq_mhz: 300.0`.

### Driver

[run_nova_direct_emu.py](run_nova_direct_emu.py) iterates a list of nova
benches, runs `make check TARGET=sw_emu` on every variant (~30 s each, fast)
and `make check TARGET=hw_emu` on a configurable subset (slow,
~10 min — 6 hr each), and emits a comparison JSONL plus a delta markdown
table.

```bash
source scripts/setup_emu_env.sh
export C2HLS_HW_EMU_STEPS=baseline,coalescing  # which steps to hw_emu
export C2HLS_HW_EMU_CLOCK_NS=3.33               # U280 platform clock
export C2HLS_HW_EMU_TIMEOUT=21600               # 6 hours per variant
python run_nova_direct_emu.py
```

The bench list at the top of the script controls which benches run:

```python
NOVA_BENCHES = [
    (("cfd", "cfd_step_factor"),  NOVA_ROOT / "cfd" / "cfd_step_factor",  "cfd_step_factor"),
    (("pathfinder",),             NOVA_ROOT / "pathfinder",               "pathfinder"),
    (("leukocyte", "lc_dilate"),  NOVA_ROOT / "leukocyte" / "lc_dilate",  "dilate"),
    (("nw",),                     NOVA_ROOT / "nw",                       "nw"),
    # ...
]
```

### Output

- [artifacts/nova_direct_emu.jsonl](artifacts/) — one record per variant, each
  carrying `sw_emu`, `hw_emu`, `ref_runtime_us`, `ref_runtime_cycles`,
  `ratio_us`, `ratio_cy`
- [artifacts/nova_direct_emu_vs_ref.md](artifacts/) — markdown delta table
  for human review

### Validated reference results

End-to-end on this box (Vitis 2023.2, U280, 3.33 ns), a typical successful
validation run produces exact runtime_us matches and cycle-ratios within 0.1 %
of upstream:

| bench | variant | hw_emu us (ours/ref) | cycles (ours/ref) | ratio_cy |
|---|---|---|---|---|
| cfd_flux | baseline | 86.737 / 86.737 | 26,047 / 26,021 | **1.001×** |
| cfd_step_factor | baseline | 17.874 / 17.874 | 5,367 / 5,362 | **1.001×** |
| knn | baseline | 3,496.756 / 3,496.756 | 1,050,076 / 1,049,027 | **1.001×** |
| knn | coalescing | 1,673.361 / 1,673.361 | 502,510 / 502,008 | **1.001×** |
| lc_dilate | baseline | 12,072.491 / 12,072.491 | 3,625,372 / 3,621,747 | **1.001×** |
| lc_dilate | coalescing | 102.034 / 102.034 | 30,640 / 30,610 | **1.001×** |
| nw | baseline | 97,019.513 / 97,019.513 | 29,134,988 / 29,105,854 | **1.001×** |
| pathfinder | baseline | 7,033.563 / 7,033.563 | 2,112,181 / 2,110,069 | **1.001×** |
| pathfinder | coalescing | 73.098 / 73.098 | 21,951 / 21,929 | **1.001×** |

The ~0.1 % cycle delta is integer-rounding from the
`cycles = int(us × 300.30)` conversion. `lud` is excluded from validation —
upstream wall-time is 6.7–12 h per variant; impractical to re-run locally.

### Standalone hw_emu on a single source file

If you just want to confirm hw_emu works for one bench without running the
full validation harness, use [run_hw_emu.py](run_hw_emu.py):

```bash
source scripts/setup_emu_env.sh
python run_hw_emu.py \
  --nova-bench-dir /home/$USER/rodinia-hls-nova/Benchmarks/pathfinder/pathfinder_0_baseline \
  --kernel-cpp benchmarks/pathfinder/hls_baseline.cpp \
  --kernel-basename pathfinder \
  --output artifacts/hw_emu_pathfinder.json
```

---

## Evaluation & Scoring

### Score with the rubric

```bash
python rubric.py --results results          # single-shot
python rubric.py --results results_multistep --multistep
python rubric.py --results results --json   # programmatic output
```

### Rubric metrics

The rubric is a 9-point comparison of generated vs ground-truth Vitis reports:

| Metric | Weight | Description |
|---|---|---|
| Synth status | gate | Pass/fail (zero-score if fail) |
| Csim correctness | gate | When supported, must match GT byte-for-byte |
| Cosim correctness | gate | When supported, RTL sim must match testbench |
| Latency (`latency_ns`) | 30% | Average-case real-time latency from csynth |
| Fmax (`fmax_mhz`) | 10% | `1000 / EstimatedClockPeriod` |
| LUT usage | 10% | vs GT |
| FF usage | 10% | vs GT |
| BRAM usage | 10% | vs GT |
| DSP usage | 10% | vs GT |
| ADP composite | 10% | `latency × normalised_area` |
| Feasibility | 10% | Fits target device (`xcu280` / `xcu50` / `xc7a100t`) |

Latency uses **average-case** (matches v++ deployment numbers); for
data-independent designs Average == Worst, but for double-buffered designs
worst-case includes warm-up/drain iterations that don't reflect steady state.

---

## JSONL Export & Comparison

The pipeline emits results in a canonical JSONL schema (v1.0) with three
record types: `sw_run`, `hls_synth`, `rtl_sim`. This format matches the
upstream rodinia-hls-nova reference data, enabling direct comparison.

```bash
python export_schema_jsonl.py \
    --results results \
    --multistep results_multistep \
    --stability artifacts/stability \
    --benchmarks benchmarks \
    --output artifacts/
```

Output: `artifacts/schema_records.jsonl` with the schema-1.0 records.

Key features:
- Per-record `run.target` is `vitis.csynth` / `vitis.csim` / `vitis.cosim` /
  `vitis.hw_emu`
- AI-generated records have `implementation.origin = c2hls_orchestrator`,
  with `origin_meta` carrying `model`, `model_translator`, `model_synthesis`,
  `model_quality_repair`, `generated_at`, `phase`/`step`
- Multistep emits **paired AI + GT records per step** (same step name on both),
  so step-N comparison is meaningful
- hw_emu records include `kernel_runtime_us` + `kernel_runtime_cycles` +
  `kernel_clock_freq_mhz`
- `AreaEstimates.AvailableResources` is populated from the `_DEVICE_TABLE`
  in `rubric.py` (xcu50, xcu280, xc7a100t, xc7a200t)

---

## HTML Report Generation

```bash
python report.py --results results --output report.html
python report.py --results results_multistep --multistep --output report_multistep.html
```

The output is a self-contained HTML page with per-benchmark cards (latency,
resources, csim/cosim status, agent transcripts, side-by-side gen vs GT
synth reports).

---

## Pipeline Architecture

The orchestrator (`C2HLSOrchestrator`) runs three coordinating agents that
share conversation history:

### TranslatorAgent

- **Phase A** — compile-check the input plain C with `g++ -c`. Fix on
  failure (up to `--turns` attempts).
- **Phase B initial translation** — emit the first HLS code via
  `q_translate_c_to_hls` system prompt, asking for `extern "C" workload()`,
  unified `bundle=control` s_axilite ports, and PIPELINE/UNROLL/
  ARRAY_PARTITION pragmas appropriate for the U280 / Virtex UltraScale+ HBM
  target.

### SynthesisAgent

Runs the synth/csim/cosim chain with structured repair feedback:

- **Per-attempt history**: each repair prompt now includes a "previous
  attempts in this phase" block summarising error class + first error-line
  per turn, plus a single-sentence "name the mistake category and the
  smallest fix" lead-in. Breaks the LLM out of two-failure-mode oscillations.
- **Repair guidance**: error-pattern → hint mappings (`_build_repair_guidance`)
  cover compile errors (redefinition, undeclared identifier, function-scope
  pragma misplacement, wrong signature), synthesis errors
  (`HLS 214-219` axilite-bundle-split with explicit fix), and synthesis
  timeouts (simplification hints).
- **Profile-bottleneck signals**: when a partial report is available,
  `_build_profile_signal` extracts II>1 loops, fmax slack, resource ceilings
  for the LLM to act on.
- **Regression guard** (multistep only): after each step accepts new code,
  `_step_regression_reasons` checks for >10 % latency growth or
  3+ resource regressions; on regression, retry once with explicit
  feedback, then revert.
- **Revert-on-streak** (`C2HLS_SYNTH_REVERT_THRESHOLD`): if N consecutive
  attempts in the same error class fail, revert to the best previously
  recorded state.
- **Best-state snapshot**: every successful synth becomes a revert target.

### QualityRepairAgent

After Phase B accepts a code, runs up to N (=2 by default) candidate
generations driven by quality guidance comparing the gen vs GT report. Each
candidate goes through the full synth/csim/cosim chain; only candidates that
preserve correctness AND improve the focus metric AND beat the score by
`QUALITY_SCORE_EPSILON` (0.25) are accepted.

### Optional: hw_emu post-step

When `C2HLS_HW_EMU_FINAL=1`, after the agentic loop accepts a final code,
[hls_eval.run_hw_emu_via_nova](hls_eval.py) stages a private copy of the
matching upstream nova benchmark, swaps in the LLM's kernel cpp (rewriting
`#include "support/common/X"` back to upstream's `#include "../../../common/X"`),
runs `make check TARGET=hw_emu`, and parses `profile_kernels.csv` for the
authoritative kernel runtime. The result lands in `results['hw_emu']` with
`{passed, success, kernel_runtime_us, kernel_runtime_cycles, error}`.

This is gated because hw_emu is slow (~30 min per kernel for typical
variants, hours for large ones).

### Phase C (multistep)

After the chain finishes, `validate_gold_reference` selects the corresponding
GT variant per step (using each variant's *own* upstream header, so
per-variant `#define`s like `TILE_SIZE` and `COALESCING_5_512bit` are
respected) and synthesises it under the same Vitis flow. The results JSON
records paired AI vs GT reports per step, plus the rubric score.

---

## Benchmark Corpus

### 21 benchmarks, three sources

| Benchmark | Source | Domain | csim | cosim | nova hw_emu |
|---|---|---|:---:|:---:|:---:|
| StreamCluster | rodinia-hls | Clustering | — | — | — |
| hotspot | rodinia-hls | Physics | — | — | — |
| kmeans | rodinia-hls | Clustering | — | — | — |
| knn | rodinia-hls | Classification | — | — | ✓ |
| lavaMD | rodinia-hls | Mol. dynamics | — | — | — |
| lud | rodinia-hls | Linear algebra | — | — | (slow) |
| nw | rodinia-hls | Bioinformatics | ✓ | ✓ | ✓ |
| pathfinder | rodinia-hls | DP | — | — | ✓ |
| srad | rodinia-hls | Image proc | — | — | — |
| aes | ML4Accel | Crypto | ✓ | — | — |
| fft | ML4Accel | Signal proc | ✓ | — | — |
| gemm_ncubed | ML4Accel | Linear algebra | ✓ | — | — |
| md_knn | ML4Accel | Mol. dynamics | ✓ | — | — |
| sort_merge | ML4Accel | Sorting | ✓ | — | — |
| spmv_crs | ML4Accel | Sparse linalg | ✓ | ✓ | — |
| stencil2D | ML4Accel | Stencil | ✓ | — | — |
| viterbi | ML4Accel | HMM | ✓ | — | — |
| **cfd_flux** | rodinia-hls-nova | CFD | — | — | ✓ |
| **cfd_step_factor** | rodinia-hls-nova | CFD | — | — | ✓ |
| **lc_gicov** | rodinia-hls-nova | Image (leukocyte) | — | — | ✓ |
| **lc_mgvf** | rodinia-hls-nova | Image (leukocyte) | — | — | ✓ |

The "nova hw_emu" column flags benchmarks where `make check TARGET=hw_emu`
runs against an upstream nova counterpart for authoritative cycle
measurement; for others, csynth + cosim provide the cycle estimate via
`vitis_hls`.

### Benchmark directory structure

```
benchmarks/
├── index.json                       # Corpus manifest
├── pathfinder/                      # Standard rodinia-style bench
│   ├── pathfinder.h                 # Cleaned header
│   ├── gold_hls_source.cpp          # Original upstream HLS code
│   ├── hls_baseline.cpp             # Localised gold (= GT for single-shot)
│   ├── plain.cpp                    # Stripped C input (LLM input)
│   ├── testbench.cpp                # csim/cosim driver (when supported)
│   ├── hls_pathfinder_1_tiling.cpp  # Per-step GT variants for multistep
│   ├── hls_pathfinder_3_unroll.cpp  # (note: indexes match rodinia naming)
│   ├── hls_pathfinder_4_doublebuffer.cpp
│   ├── hls_pathfinder_5_coalescing.cpp
│   ├── support/
│   │   └── common/
│   │       ├── mc.h                 # MARS_WIDE_BUS macros for coalescing
│   │       └── mars_wide_bus_*.h
│   └── metadata.json                # Provenance + variants list
└── ...
```

### Results directory structure

```
results/                              # Single-shot
├── all_results.json
└── pathfinder/
    ├── pathfinder_generated.cpp     # Final accepted HLS code
    ├── pathfinder_synth_report.json # csynth report
    ├── pathfinder_results.json      # Phase C comparison + run attribution + hw_emu (optional)
    └── pathfinder_history.json      # Full LLM transcript w/ per-agent model attribution

results_multistep/                    # Multi-step
└── pathfinder/
    ├── pathfinder_final.cpp
    ├── pathfinder_multistep_results.json
    ├── pathfinder_history.json
    └── steps/
        ├── 0_tiling.cpp
        ├── 0_tiling_report.json     # AI report + GT report at this step
        ├── 1_pipeline.cpp
        ├── 1_pipeline_report.json
        └── ...

artifacts/                            # JSONL exports + reports
├── schema_records.jsonl              # Canonical schema-1.0 records
├── nova_direct_emu.jsonl             # Direct hw_emu validation
├── nova_direct_emu_vs_ref.md         # hw_emu cycle delta vs reference
├── stability/                        # Repeat-N stability records
└── ...
```

---

## File Reference

### Core

| File | Purpose |
|---|---|
| [c2hls.py](c2hls.py) | Main pipeline — `C2HLSOrchestrator` + agents (`TranslatorAgent`, `SynthesisAgent`, `QualityRepairAgent`), `run_benchmark()`, `run_benchmark_multistep()`, hw_emu post-step |
| [hls_eval.py](hls_eval.py) | Vitis HLS / v++ runner: `run_hls_synthesis`, `run_csim`, `run_cosim`, `run_sw_emu_via_nova`, `run_hw_emu_via_nova` |
| [prompt_c2hls.py](prompt_c2hls.py) | All LLM prompts — system instruction, Phase A/B/quality-repair templates, per-optimisation-step prompts (with attempt-history feedback fields) |
| [rubric.py](rubric.py) | 9-metric scoring rubric with U280 / U50 / Artix-7 device tables |
| [report.py](report.py) | HTML report generator |

### Benchmark prep

| File | Purpose |
|---|---|
| [prepare_benchmarks.py](prepare_benchmarks.py) | Generates `benchmarks/` from rodinia-hls + ML4Accel-Dataset upstream repos |
| [prepare_nova_benchmarks.py](prepare_nova_benchmarks.py) | Adds nova benchmarks (cfd_flux/cfd_step_factor/lc_gicov/lc_mgvf) to corpus |
| [validate_corpus.py](validate_corpus.py) | Sanity-checks the corpus (no HLS leakage in plain.cpp, signature-compat against testbenches) |

### Validation drivers

| File | Purpose |
|---|---|
| [run_nova_direct_emu.py](run_nova_direct_emu.py) | Direct sw_emu/hw_emu on nova benches — no LLM, validates v++/XSIM cycle counts vs reference |
| [run_hw_emu.py](run_hw_emu.py) | Standalone hw_emu wrapper for one bench/variant |
| [verify_corpus_stability.py](verify_corpus_stability.py) | Repeat-N csynth on each bench — measures Vitis determinism for a target/clock/version tuple |
| [run_2023_2_synth_comparison.py](run_2023_2_synth_comparison.py) | Direct csynth on nw/pathfinder/knn vs upstream csynth reference |

### Multi-bench drivers

| File | Purpose |
|---|---|
| [run_multistep_haiku.py](run_multistep_haiku.py) | 4-bench multistep with haiku-4.5 |
| [run_multistep_hwemu.py](run_multistep_hwemu.py) | 4-bench multistep + hw_emu post-step |
| [run_3bench_haiku_sonnet.py](run_3bench_haiku_sonnet.py) | pathfinder/knn/nw with haiku + sonnet for cross-model comparison |
| [run_remaining_haiku.py](run_remaining_haiku.py) | All 14 benches not in the 3-bench set |
| [run_patched_benchmarks.py](run_patched_benchmarks.py) | Re-runs benches affected by a recent pipeline patch |

### JSONL export

| File | Purpose |
|---|---|
| [export_schema_jsonl.py](export_schema_jsonl.py) | Emits canonical schema-1.0 records (sw_run / hls_synth / rtl_sim) from `results/`, `results_multistep/`, `artifacts/stability/` |
| [export_rl_corpus.py](export_rl_corpus.py) | RL trajectory dataset export from accepted runs |
| [export_ml4accel_points.py](export_ml4accel_points.py) | ML4Accel design-point normaliser |

### Setup helpers

| File | Purpose |
|---|---|
| [scripts/setup_emu_env.sh](scripts/setup_emu_env.sh) | One-line emu env setup: sources Vitis settings, XRT, sets PLATFORM_REPO_PATHS + CPLUS_INCLUDE_PATH |
| [install_2023.2_config.txt](install_2023.2_config.txt) | Minimal Vitis 2023.2 install config (HLS only + Virtex UltraScale+ HBM) |

### Reference data

| Path | Purpose |
|---|---|
| [results/references_philip/](results/references_philip/) | Upstream sw_emu + hw_emu JSONL references for the 17 nova benchmarks |
| [csynth_vitis_2023.2__device_xilinx_u280_*.jsonl](csynth_vitis_2023.2__device_xilinx_u280_gen3x16_xdma_1_202211_1.jsonl) | Reference csynth records (legacy — prefer `results/references_philip/` for sw_emu/hw_emu) |

---

## Troubleshooting

### Vitis HLS not found

```
vitis_hls: command not found
```

Source the Vitis settings file before running, or set
`C2HLS_VITIS_SETTINGS` to its path:

```bash
source /path/to/Xilinx/Vitis/2023.2/settings64.sh
# or
export C2HLS_VITIS_SETTINGS=/path/to/Xilinx/Vitis/2023.2/settings64.sh
```

### "Part 'xcu280-fsvh2892-2L-e' is not installed"

The Vitis install lacks the Virtex UltraScale+ HBM device family. Re-run
the installer with `Modules=Virtex UltraScale+ HBM:1`, or via xsetup:

```bash
xsetup -b Add -c install_2023.2_config.txt
```

### `make check TARGET=sw_emu` errors with `CL/cl.h not found`

Khronos OpenCL headers aren't on the include path. Either install
`opencl-headers` via apt, or download from Khronos and point
`CPLUS_INCLUDE_PATH` at them — see [scripts/setup_emu_env.sh](scripts/setup_emu_env.sh).

### `make check TARGET=hw_emu` errors with `bits/wordsize.h not found`

Ubuntu 22.04 multi-arch glibc headers under
`/usr/include/x86_64-linux-gnu`. Add to `C_INCLUDE_PATH` /
`CPLUS_INCLUDE_PATH` (also handled by `setup_emu_env.sh`).

### "Authentication token expired" during Vitis install

The web installer's auth token is valid for ~14 days. Re-run interactively:

```bash
xsetup -b AuthTokenGen
```

### hw_emu on lc_dilate / nested benches fails with `'../../../common/mc.h' file not found`

The bench's header includes a *group-level* `common/` (e.g. `leukocyte/common/mc.h`),
not the canonical `Benchmarks/common/`. The staging logic in
[hls_eval._stage_nova_workdir](hls_eval.py) creates symlinks for both. If
this fails, check that your nova source tree has both `Benchmarks/common/`
(with `libs/xcl2/xcl2.mk`) AND any group-level `common/` dirs.

### Synthesis timeout

Some kernels exceed the default 1200 s `csynth_design` budget. Bump it:

```bash
export C2HLS_SYNTH_TIMEOUT=2400
```

If multiple consecutive timeouts happen on the same kernel, the LLM is
likely stuck in a fix-the-symptom loop. The `attempt_history` feedback
inserted into repair prompts surfaces this; see also `_classify_synth_error`
+ `C2HLS_SYNTH_REVERT_THRESHOLD` for revert-on-streak.

### "undef" latency in csynth reports

Vitis HLS reports `undef` for top-level latency when loops have variable
trip counts. The pipeline parses `Average-caseRealTimeLatency` first and
falls back to `Worst-caseRealTimeLatency`; for designs with conditional
prologue/epilogue (double-buffered chains), Average is ~2× lower than
Worst — we use Average to match v++ deployment numbers.

### Cycle vs latency_ns mismatch

If your hw_emu cycle count is off by ~17 %, the cycle conversion may have
used the wrong clock period. The U280 platform uses 3.33 ns (300 MHz);
make sure `C2HLS_HW_EMU_CLOCK_NS=3.33` is exported (independent from
`C2HLS_CLOCK_NS`, which targets csynth's `create_clock`).

### Qwen models on vLLM

Qwen models require `enable_thinking: false` passed via `extra_body` in the
OpenAI client. The pipeline applies this automatically when the model id
matches `qwen` (case-insensitive).

### API key not found

```
AssertionError: Missing Anthropic API key. Set ANTHROPIC_API_KEY env var or create a .env file.
```

Either:
- Set the env var: `export ANTHROPIC_API_KEY=sk-ant-...`
- Create `.env` at repo root with `ANTHROPIC_API_KEY=sk-ant-...`
- Save to the file referenced by `C2HLS_CLAUDE_KEY_FILE` (default
  `/home/luo00466/claude-api-key.txt`)
