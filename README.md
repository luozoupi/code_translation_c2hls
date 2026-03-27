# C-to-HLS Code Translation Pipeline

An LLM-based pipeline that translates plain C/C++ kernels into Xilinx Vitis HLS optimized code, validates the output through Vitis HLS synthesis, and evaluates quality against ground-truth HLS baselines.

## Overview

The pipeline takes pragma-free C code (derived from known-good HLS benchmarks with pragmas stripped) and asks an LLM to re-introduce HLS optimizations. The generated code is then synthesized with Vitis HLS, optionally verified with C-simulation/co-simulation, and scored against the ground-truth HLS baseline using a 9-metric rubric.

```
gold_hls_source.cpp   ──(strip pragmas)──>   plain.cpp   ──(LLM)──>   generated HLS
        │                                                                    │
        └──(synthesize as ground truth)──>   GT report  <──(compare)──  gen report
```

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Benchmark Preparation](#benchmark-preparation)
4. [Running Translations](#running-translations)
5. [Evaluation & Scoring](#evaluation--scoring)
6. [HTML Report Generation](#html-report-generation)
7. [Pipeline Architecture](#pipeline-architecture)
8. [Benchmark Corpus](#benchmark-corpus)
9. [File Reference](#file-reference)
10. [Troubleshooting](#troubleshooting)

---

## Prerequisites

- **Xilinx Vitis HLS 2025.2** (or compatible version)
- **Python 3.10+** with pip
- **g++** (for C/C++ compilation checks)
- **LLM backend** — one of:
  - Anthropic API key (for Claude models)
  - OpenAI API key (for GPT models)
  - Local vLLM server (for open-source models like Qwen, Nemotron)

---

## Environment Setup

### Step 1: Install Vitis HLS (headless / no-GUI)

If Vitis HLS is not already installed, download the **Vitis Unified Installer** from
[Xilinx Downloads](https://www.xilinx.com/support/download/index.html/content/xilinx/en/downloadNav/vitis.html) (requires a free AMD account).

For headless (CLI-only) installation on a Linux server:

```bash
# Download the installer (example for 2025.2; adjust version as needed)
chmod +x Xilinx_Unified_2025.2_XXXX_Lin64.bin

# Run headless install — select "Vitis HLS" only to save disk space (~15 GB vs ~100 GB full)
./Xilinx_Unified_2025.2_XXXX_Lin64.bin -- --batch ConfigGen
# Edit ~/.Xilinx/install_config.txt to set:
#   Destination  : /opt/Xilinx/2025.2   (or your preferred path)
#   Modules      : Vitis HLS:1           (enable only HLS)
./Xilinx_Unified_2025.2_XXXX_Lin64.bin -- --batch Install --agree XilinxEULA,3rdPartyEULA --config ~/.Xilinx/install_config.txt
```

After installation, verify:
```bash
source <XILINX_INSTALL_DIR>/Vitis/settings64.sh
which vitis_hls   # should print the path to the vitis_hls binary
```

### Step 2: Source Vitis HLS in your shell

Add this to your `.bashrc` or run before each session:

```bash
source <XILINX_INSTALL_DIR>/Vitis/settings64.sh
```

Replace `<XILINX_INSTALL_DIR>` with your actual installation path (e.g., `/opt/Xilinx/2025.2`).

> **Important:** You must also update `VITIS_SETTINGS` in `hls_eval.py` to match your installation path:
> ```python
> VITIS_SETTINGS = "<XILINX_INSTALL_DIR>/Vitis/settings64.sh"
> ```

### Step 3: Set up Python environment

```bash
# Option A: conda
conda create -n c2hls python=3.10 -y
conda activate c2hls

# Option B: venv
python3 -m venv .venv && source .venv/bin/activate
```

### Step 4: Install Python dependencies

```bash
pip install openai anthropic python-dotenv
```

### Step 5: Configure LLM access

The pipeline auto-detects the backend from the `--model` argument:

| Model prefix | Backend | Configuration |
|---|---|---|
| `claude-*` | Anthropic API | Set `ANTHROPIC_API_KEY` env var or create a `.env` file |
| `gpt-*`, `o1-*`, `o3-*`, `o4-*` | OpenAI API | Set `OPENAI_API_KEY` env var or create a `.env` file |
| Everything else | vLLM (local) | Set `OPENAI_BASE_URL` (default: `http://127.0.0.1:8000/v1`) |

Create a `.env` file in the project root (git-ignored):
```bash
# For Anthropic Claude models
ANTHROPIC_API_KEY=sk-ant-...

# For OpenAI models
OPENAI_API_KEY=sk-...

# For local vLLM (optional, these are the defaults)
OPENAI_BASE_URL=http://127.0.0.1:8000/v1
OPENAI_API_KEY=EMPTY
```

**Example — using a local vLLM server:**
```bash
# Start vLLM in a separate terminal
vllm serve Qwen/Qwen3.5-35B-A3B --port 8000

# The pipeline will auto-detect non-Claude/GPT model names as vLLM
python c2hls.py --bench aes --model Qwen/Qwen3.5-35B-A3B
```

### Step 6: Configure project paths

Several paths in the codebase need to match your environment. Update these before first use:

| File | Variable | Description |
|---|---|---|
| `hls_eval.py` | `VITIS_SETTINGS` | Path to Vitis `settings64.sh` |
| `prepare_benchmarks.py` | `ROOT` | Project root directory |
| `prepare_benchmarks.py` | `RODINIA_DIR` | Path to rodinia-hls benchmark repo |
| `prepare_benchmarks.py` | `ML4ACCEL_DIR` | Path to ML4Accel-Dataset repo |

If you only use the pre-built benchmarks in `benchmarks/` (already included in this repo), you can skip configuring `prepare_benchmarks.py`.

---

## Benchmark Preparation

### Using pre-built benchmarks (recommended)

The `benchmarks/` directory is included in this repository with all 17 pre-processed benchmarks. **No preparation step is needed** — skip to [Running Translations](#running-translations).

### Regenerating benchmarks from source (optional)

If you want to regenerate from the upstream repos:

1. Clone the upstream benchmark repos:
   ```bash
   git clone https://github.com/UCLA-VAST/rodinia-hls.git /path/to/rodinia-hls
   git clone https://github.com/UIUC-ChenLab/ML4Accel-Dataset.git /path/to/ML4Accel-Dataset
   ```

2. Update paths in `prepare_benchmarks.py`:
   ```python
   ROOT = Path("/path/to/this/project")
   RODINIA_DIR = Path("/path/to/rodinia-hls/Benchmarks")
   ML4ACCEL_DIR = Path("/path/to/ML4Accel-Dataset/fpga_ml_dataset/HLS_dataset")
   ```

3. Run the preparation script:
   ```bash
   python prepare_benchmarks.py
   ```

This reads gold HLS sources and creates three versions per benchmark:

| File | Description |
|---|---|
| `gold_hls_source.cpp` | Original HLS code from upstream (pragmas + `extern "C"`) |
| `hls_baseline.cpp` | Localized copy of gold (include paths adjusted) — used as **ground truth** |
| `plain.cpp` | Gold with all HLS pragmas and `extern "C"` stripped — used as **LLM input** |

The stripping removes:
- `#pragma HLS ...` and `#pragma ACCEL ...` directives
- `extern "C" { ... }` wrapper blocks
- `#include <ap_int.h>` and support-library includes

Each benchmark directory also contains:
- A **header file** (e.g., `aes.h`, `nw.h`) shared between plain and HLS versions
- A **testbench** (`testbench.cpp`) for benchmarks supporting C-simulation
- A **`metadata.json`** recording provenance, SHA256 hashes, and strip report

**Note:** `gold_hls_source.cpp` and `hls_baseline.cpp` are often byte-identical. They differ only for benchmarks where include path localization was needed.

### Verifying the preparation

```bash
# Check a specific benchmark's strip report
python -c "import json; print(json.dumps(json.load(open('benchmarks/aes/metadata.json'))['strip_report'], indent=2))"
```

Expected output confirms all pragmas were removed:
```json
{
  "removed_hls_pragmas": 5,
  "removed_extern_c_blocks": 1,
  "plain_contains_hls_pragmas": false
}
```

---

## Running Translations

### Run a single benchmark (single-shot mode)

```bash
python c2hls.py --bench aes --model claude-haiku-4-5-20251001 --turns 3
```

This runs the full three-phase pipeline on one benchmark:
1. **Phase A** — Validates `plain.cpp` compiles with `g++ -c`
2. **Phase B** — LLM translates to HLS, iterates on compilation/synthesis failures (up to `--turns` attempts)
3. **Phase C** — Compares generated synthesis report against ground-truth baseline

Results are saved to `results/<benchmark>/`.

### Run all benchmarks

```bash
python c2hls.py --all --model claude-haiku-4-5-20251001 --turns 3
```

Runs all 17 benchmarks sequentially. Results are saved per-benchmark under `results/`, plus a combined `results/all_results.json`.

### Run multi-step incremental optimization

```bash
# All optimization steps (default order: tiling -> pipeline -> unroll -> doublebuffer -> coalescing)
python c2hls.py --bench nw --multistep --model claude-haiku-4-5-20251001

# Specific steps only
python c2hls.py --bench StreamCluster --multistep --steps tiling,pipeline --model claude-haiku-4-5-20251001

# All benchmarks, multi-step
python c2hls.py --all --multistep --model claude-haiku-4-5-20251001
```

Multi-step mode applies optimizations incrementally:
1. First generates a baseline HLS translation (same as single-shot Phase B)
2. Then applies each optimization step sequentially, re-synthesizing after each
3. Each step uses the previous step's output as input, with a step-specific prompt

Available optimization steps: `tiling`, `pipeline`, `unroll`, `doublebuffer`, `coalescing`

### CLI reference

```
python c2hls.py [-h]
    --bench BENCH               Benchmark name (from benchmarks/ directory)
    --bench-dir BENCH_DIR       Direct path to benchmark directory
    --output-dir OUTPUT_DIR     Output directory for results
    --model MODEL               LLM model ID (default: nvidia/OpenCodeReasoning-Nemotron-1.1-32B)
    --turns TURNS               Max fix attempts per phase (default: 3)
    --quality-repair-turns N    Max post-synthesis quality repair attempts
    --all                       Run all benchmarks
    --multistep                 Run multi-step incremental optimization
    --steps STEPS               Comma-separated optimization steps
```

---

## Evaluation & Scoring

### Score results with the rubric

```bash
# Score single-shot results
python rubric.py --results results

# Score multi-step results
python rubric.py --results results --multistep

# Output as JSON (for programmatic use)
python rubric.py --results results --json
```

### Rubric metrics

The rubric evaluates generated HLS code across 9 weighted metrics against the ground-truth baseline:

| Metric | Weight | Description |
|---|---|---|
| M1. Synthesis Success | 5% | Did the code synthesize at all? (binary) |
| M2. C-Simulation | 10% | Functional correctness vs testbench (binary) |
| M3. Co-Simulation | 10% | RTL matches C behaviour (binary) |
| M4. Latency | 20% | Cycle count vs GT (lower ratio = better) |
| M5. Clock Frequency | 8% | Fmax / timing closure quality (higher = better) |
| M6. Throughput / II | 7% | Initiation interval vs GT (lower = better) |
| M7. Resource Efficiency | 17% | BRAM/DSP/FF/LUT vs GT, weighted by scarcity |
| M8. Area-Delay Product | 13% | Combined latency x normalized area efficiency |
| M9. Device Feasibility | 10% | Hard resource pressure vs Artix-7 100T limits |

**Target device:** xc7a100t-csg324-1 (Artix-7 100T) — BRAM=270, DSP=240, FF=126800, LUT=63400
**Clock target:** 10 ns (100 MHz)

### Grade scale

| Grade | Score | Meaning |
|---|---|---|
| A | 90-100 | Excellent — matches or improves on GT |
| B | 75-89 | Good — close to GT with minor overhead |
| C | 60-74 | Acceptable — functional but notable gaps |
| D | 40-59 | Below average — large overhead or timing issues |
| F | 0-39 | Poor — synthesis failure, infeasible, or extreme overhead |

---

## HTML Report Generation

Generate an interactive HTML report from your results:

```bash
# From single-shot results
python report.py --results results --output report.html

# From multi-step results
python report.py --results results --multistep --output report_multistep.html
```

The report includes:
- Overall summary with grade distribution and pass rates
- Per-benchmark score breakdown with bar charts
- Resource usage comparison (Generated vs Ground Truth)
- Latency and Fmax ratio visualizations
- Device utilization heatmap
- Detailed per-benchmark drill-down tables

Open the generated `.html` file in any browser — it is fully self-contained (no external dependencies).

---

## Pipeline Architecture

### Phase A: Input Validation

- **Input:** `plain.cpp` (HLS-pragma-free C code) + header file
- **Process:** Compile with `g++ -c` to verify it is valid C/C++
- **On failure:** LLM attempts to fix compilation errors (up to `--turns` attempts)
- **Output:** Validated C source code

### Phase B: LLM Translation + Synthesis Validation

- **Input:** Validated C code from Phase A
- **Process:**
  1. LLM translates C to HLS code — adds `extern "C"` wrapper, HLS INTERFACE pragmas, PIPELINE/UNROLL/ARRAY_PARTITION directives
  2. Generated code is compiled with `g++ -c` (fix if needed)
  3. Vitis HLS synthesis (`csynth_design`) validates the design
  4. If synthesis fails, the error log is fed back to the LLM for repair (up to `--turns` attempts)
  5. On synthesis success: optionally runs C-simulation (csim) and co-simulation (cosim)
- **Output:** Synthesizable HLS code + synthesis report (latency, Fmax, resources)

### Phase C: Quality Comparison

- **Input:** Phase B synthesis report + ground-truth baseline report
- **Process:** Metric-by-metric comparison (latency ratio, resource ratios, Fmax ratio, etc.)
- **Output:** Comparison dictionary saved to `<bench>_results.json`

### Translation prompt requirements

The LLM is instructed to:
1. Wrap the kernel in a `workload()` function with `extern "C"` linkage
2. Add HLS INTERFACE pragmas (`m_axi` for pointer args, `s_axilite` for control)
3. Add optimization pragmas (PIPELINE, UNROLL, ARRAY_PARTITION)
4. Preserve algorithm correctness — no behavioral changes
5. Include the benchmark header file exactly once
6. Keep all `#pragma HLS` inside function bodies

---

## Benchmark Corpus

### 17 benchmarks from two sources

| Benchmark | Source | Domain | Csim | Cosim |
|---|---|---|---|---|
| StreamCluster | rodinia-hls | Clustering | No | No |
| hotspot | rodinia-hls | Physics simulation | No | No |
| kmeans | rodinia-hls | Clustering | No | No |
| knn | rodinia-hls | Classification | No | No |
| lavaMD | rodinia-hls | Molecular dynamics | No | No |
| lud | rodinia-hls | Linear algebra | No | No |
| nw | rodinia-hls | Bioinformatics | Yes | Yes |
| pathfinder | rodinia-hls | Dynamic programming | No | No |
| srad | rodinia-hls | Image processing | No | No |
| aes | ML4Accel | Cryptography | Yes | No |
| fft | ML4Accel | Signal processing | Yes | No |
| gemm_ncubed | ML4Accel | Linear algebra | Yes | No |
| md_knn | ML4Accel | Molecular dynamics | Yes | No |
| sort_merge | ML4Accel | Sorting | Yes | No |
| spmv_crs | ML4Accel | Sparse linear algebra | Yes | Yes |
| stencil2D | ML4Accel | Stencil computation | Yes | No |
| viterbi | ML4Accel | HMM decoding | Yes | No |

### Benchmark directory structure

```
benchmarks/
+-- index.json                   # Corpus manifest
+-- aes/
|   +-- aes.h                    # Shared header
|   +-- gold_hls_source.cpp      # Original HLS code (with pragmas)
|   +-- hls_baseline.cpp         # Localized ground truth (= identical to gold)
|   +-- plain.cpp                # Stripped C code (LLM input)
|   +-- testbench.cpp            # C-simulation testbench
|   +-- metadata.json            # Provenance + strip report
+-- nw/
|   +-- nw.h
|   +-- gold_hls_source.cpp
|   +-- hls_baseline.cpp
|   +-- plain.cpp
|   +-- testbench.cpp
|   +-- hls_nw_1_tiling.cpp      # Multi-step GT variants
|   +-- hls_nw_2_pipeline.cpp
|   +-- hls_nw_3_unroll.cpp
|   +-- hls_nw_4_doublebuffer.cpp
|   +-- hls_nw_5_coalescing.cpp
|   +-- metadata.json
+-- ...
```

### Results directory structure

```
results/
+-- all_results.json             # Combined results from --all runs
+-- aes/
|   +-- aes_generated.cpp        # LLM-generated HLS code
|   +-- aes_synth_report.json    # Vitis HLS synthesis report
|   +-- aes_results.json         # Phase C comparison + metadata
|   +-- aes_history.json         # LLM conversation history
+-- ...
```

---

## File Reference

| File | Purpose |
|---|---|
| `c2hls.py` | Main pipeline — `C2HLSOrchestrator` class, `run_benchmark()`, `run_benchmark_multistep()` |
| `hls_eval.py` | Vitis HLS synthesis runner, XML + text report parser, csim/cosim support |
| `prompt_c2hls.py` | All LLM prompts — system instruction, Phase A/B repair prompts, per-optimization-step prompts |
| `prepare_benchmarks.py` | Generates `benchmarks/` from rodinia-hls and ML4Accel-Dataset upstream repos |
| `rubric.py` | 9-metric scoring rubric — compares generated vs GT synthesis reports |
| `report.py` | HTML report generator — produces self-contained interactive reports |
| `benchmarks/` | Prepared benchmark data (plain C input + HLS ground truth + testbenches) |
| `results/` | Pipeline output (generated HLS code, synthesis reports, comparisons) |

---

## Troubleshooting

### Vitis HLS not found

```
vitis_hls: command not found
```

Ensure you have sourced the Vitis settings file and that `VITIS_SETTINGS` in `hls_eval.py` points to the correct path:
```bash
source <XILINX_INSTALL_DIR>/Vitis/settings64.sh
```

### Synthesis timeout (600s)

Some benchmarks (e.g., viterbi) produce code that is too complex for Vitis HLS to synthesize within the 600-second timeout. This typically happens when the LLM aggressively partitions large arrays. Try a stronger model or reduce array sizes in the prompt.

### kmeans Phase A failure

The kmeans benchmark references `../../../common/mc.h` in its header. If this path is unresolvable, Phase A compilation will fail. Ensure the support files in the benchmark directory are intact.

### "undef" latency in reports

Vitis HLS reports `undef` for top-level latency when loops have variable trip counts. The pipeline falls back to the maximum loop-level latency from the text report (`_parse_ns_value()` handles unit suffixes like `"11.193 ms"`).

### Fmax extraction

Fmax is computed as `1000 / EstimatedClockPeriod` from `csynth.xml`. It is not directly available in the text report.

### Qwen models on vLLM

Qwen models require `enable_thinking: false` passed via `extra_body` in the OpenAI client. This is handled automatically in `c2hls.py`.

### API key not found

```
AssertionError: Missing Anthropic API key. Set ANTHROPIC_API_KEY env var or create a .env file.
```

Set the appropriate environment variable or create a `.env` file in the project root.
