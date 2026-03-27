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
6. [Pipeline Architecture](#pipeline-architecture)
7. [Benchmark Corpus](#benchmark-corpus)
8. [File Reference](#file-reference)
9. [Troubleshooting](#troubleshooting)

---

## Prerequisites

- **Vitis HLS 2025.2** installed at `/mnt/data/luo00466/Xilinx/2025.2/`
- **Conda** with the `py310_2` environment
- **LLM backend** — one of:
  - Anthropic API key (for Claude models)
  - OpenAI API key (for GPT models)
  - Local vLLM server (for open-source models like Qwen, Nemotron)

---

## Environment Setup

### Step 1: Source Vitis HLS

```bash
source /mnt/data/luo00466/Xilinx/2025.2/Vitis/settings64.sh
```

This makes `vitis_hls` available in your PATH. Required for synthesis (Phase B) and ground-truth report generation.

### Step 2: Activate conda environment

```bash
conda activate py310_2
```

### Step 3: Install Python dependencies (if needed)

```bash
pip install openai anthropic python-dotenv
```

### Step 4: Configure LLM access

The pipeline auto-detects the backend from the `--model` argument:

| Model prefix | Backend | Configuration |
|---|---|---|
| `claude-*` | Anthropic API | Set `ANTHROPIC_API_KEY` env var, or place key in `/home/luo00466/claude-api-key.txt` |
| `gpt-*`, `o1-*`, `o3-*`, `o4-*` | OpenAI API | Set `OPENAI_API_KEY` env var, or place key in `/home/luo00466/gpt-key.txt` |
| Everything else | vLLM (local) | Set `OPENAI_BASE_URL` (default: `http://127.0.0.1:8000/v1`) |

**Example — using Claude Haiku:**
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

**Example — using a local vLLM server:**
```bash
# Start vLLM (in a separate terminal / tmux)
# vllm serve Qwen/Qwen3.5-35B-A3B --port 8000

export OPENAI_API_KEY=EMPTY
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
```

---

## Benchmark Preparation

### Step 5: Generate benchmark data

```bash
python prepare_benchmarks.py
```

This reads gold HLS sources from two upstream repos and creates a clean benchmark corpus in `benchmarks/`:

- **rodinia-hls** (9 benchmarks): StreamCluster, hotspot, kmeans, knn, lavaMD, lud, nw, pathfinder, srad
- **ML4Accel-Dataset** (8 benchmarks): aes, fft, gemm_ncubed, md_knn, sort_merge, spmv_crs, stencil2D, viterbi

For each benchmark, three files are produced:

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

### Step 6: Run a single benchmark (single-shot mode)

```bash
python c2hls.py --bench aes --model claude-haiku-4-5-20251001 --turns 3
```

This runs the full three-phase pipeline on one benchmark:
1. **Phase A** — Validates `plain.cpp` compiles with `g++ -c`
2. **Phase B** — LLM translates to HLS, iterates on compilation/synthesis failures (up to `--turns` attempts)
3. **Phase C** — Compares generated synthesis report against ground-truth baseline

Results are saved to `results/<benchmark>/`.

### Step 7: Run all benchmarks

```bash
python c2hls.py --all --model claude-haiku-4-5-20251001 --turns 3
```

Runs all 17 benchmarks sequentially. Results are saved per-benchmark under `results/`, plus a combined `results/all_results.json`.

### Step 8: Run multi-step incremental optimization

```bash
# All optimization steps (default order: tiling → pipeline → unroll → doublebuffer → coalescing)
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

### Step 9: Score results with the rubric

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
| M8. Area-Delay Product | 13% | Combined latency × normalized area efficiency |
| M9. Device Feasibility | 10% | Hard resource pressure vs Artix-7 100T limits |

**Target device:** xc7a100t-csg324-1 (Artix-7 100T) — BRAM=270, DSP=240, FF=126800, LUT=63400
**Clock target:** 10 ns (100 MHz)

### Grade scale

| Grade | Score | Meaning |
|---|---|---|
| A | 90–100 | Excellent — matches or improves on GT |
| B | 75–89 | Good — close to GT with minor overhead |
| C | 60–74 | Acceptable — functional but notable gaps |
| D | 40–59 | Below average — large overhead or timing issues |
| F | 0–39 | Poor — synthesis failure, infeasible, or extreme overhead |

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
├── index.json                   # Corpus manifest
├── aes/
│   ├── aes.h                    # Shared header
│   ├── gold_hls_source.cpp      # Original HLS code (with pragmas)
│   ├── hls_baseline.cpp         # Localized ground truth (≈ identical to gold)
│   ├── plain.cpp                # Stripped C code (LLM input)
│   ├── testbench.cpp            # C-simulation testbench
│   └── metadata.json            # Provenance + strip report
├── nw/
│   ├── nw.h
│   ├── gold_hls_source.cpp
│   ├── hls_baseline.cpp
│   ├── plain.cpp
│   ├── testbench.cpp
│   ├── hls_nw_1_tiling.cpp      # Multi-step GT variants
│   ├── hls_nw_2_pipeline.cpp
│   ├── hls_nw_3_unroll.cpp
│   ├── hls_nw_4_doublebuffer.cpp
│   ├── hls_nw_5_coalescing.cpp
│   └── metadata.json
└── ...
```

### Results directory structure

```
results/
├── all_results.json             # Combined results from --all runs
├── aes/
│   ├── aes_generated.cpp        # LLM-generated HLS code
│   ├── aes_synth_report.json    # Vitis HLS synthesis report
│   ├── aes_results.json         # Phase C comparison + metadata
│   └── aes_history.json         # LLM conversation history
└── ...
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
| `benchmarks/` | Prepared benchmark data (plain C input + HLS ground truth + testbenches) |
| `results/` | Pipeline output (generated HLS code, synthesis reports, comparisons) |

---

## Troubleshooting

### Synthesis timeout (600s)

Some benchmarks (e.g., viterbi) produce code that is too complex for Vitis HLS to synthesize within the 600-second timeout. This typically happens when the LLM aggressively partitions large arrays. Try a stronger model or reduce array sizes in the prompt.

### kmeans Phase A failure

The kmeans benchmark references `../../../common/mc.h` in its header. If this path is unresolvable, Phase A compilation will fail. Ensure the rodinia-hls common directory is accessible.

### "undef" latency in reports

Vitis HLS reports `undef` for top-level latency when loops have variable trip counts. The pipeline falls back to the maximum loop-level latency from the text report (`_parse_ns_value()` handles unit suffixes like `"11.193 ms"`).

### Fmax extraction

Fmax is computed as `1000 / EstimatedClockPeriod` from `csynth.xml`. It is not directly available in the text report.

### Qwen models on vLLM

Qwen models require `enable_thinking: false` passed via `extra_body` in the OpenAI client. This is handled automatically in `c2hls.py`.

### API key not found

```
AssertionError: Missing Anthropic API key. Set ANTHROPIC_API_KEY or populate /home/luo00466/claude-api-key.txt.
```

Set the appropriate environment variable or create the key file for your chosen backend.
