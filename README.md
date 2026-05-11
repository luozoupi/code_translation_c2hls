# C-to-HLS Code Translation Pipeline

A multi-agent LLM-driven pipeline that translates plain C/C++ kernels into
Xilinx Vitis HLS optimised code, validates the output through Vitis HLS
synthesis (csynth / csim / cosim), optionally measures kernel runtime via
XSIM hardware emulation, and scores quality against ground-truth HLS
baselines.

## Overview

The pipeline takes pragma-free C code (derived from known-good HLS benchmarks
with pragmas stripped) and uses LLM agents to re-introduce HLS optimisations
in phases. Generated code is synthesised with Vitis HLS, validated for
functional correctness (csim/cosim), and scored against the ground-truth using
a 9-metric rubric.

```
gold_hls_source.cpp ──(strip pragmas)──> plain.cpp ──(LLM agents)──> generated HLS
        │                                                                    │
        ├──(synthesize as ground truth)──> GT report  <──(rubric)──── gen csynth report
        └──(reference hw_emu)──>          ref cycles  <──(direct)──── XSIM kernel cycles
```

### Multi-step optimisation chain

In multistep mode the orchestrator applies five incremental optimisation steps,
each validated against the corresponding GT variant:

```
baseline ──> tiling ──> pipeline ──> unroll ──> doublebuffer ──> coalescing
   │             │           │           │              │               │
Phase 8     Phase 9      Phase 9     Phase 9        Phase 9         Phase 9
(align)    (repair)    (repair)    (repair)        (repair)        (repair)
   │                                                                    │
Phase 6a ─────────────── best-so-far across all steps ─────────────────┘
```

### Agent decomposition

The orchestrator runs three coordinating agents:

- **TranslatorAgent** — Phase A compile-check + Phase B initial translation
- **SynthesisAgent** — synth / csim / cosim chain with structured repair
  feedback, per-loop bottleneck signals (Pillar 1), regression guard, and
  correctness-repair loop (Phase 9)
- **QualityRepairAgent** — post-synthesis quality-driven candidate generation
  to close the gap to the ground-truth baseline

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Benchmark Preparation](#benchmark-preparation)
4. [Running Translations](#running-translations)
5. [Pipeline Architecture](#pipeline-architecture)
6. [Per-loop Feedback System (Pillar 1)](#per-loop-feedback-system-pillar-1)
7. [Skill Library (Pillar 3)](#skill-library-pillar-3)
8. [Bottleneck Router (Pillar 5)](#bottleneck-router-pillar-5)
9. [Verifying Nova Benchmarks](#verifying-nova-benchmarks)
10. [Evaluation & Scoring](#evaluation--scoring)
11. [Dataset Pipeline](#dataset-pipeline)
12. [JSONL Export & Comparison](#jsonl-export--comparison)
13. [Benchmark Corpus](#benchmark-corpus)
14. [File Reference](#file-reference)
15. [Environment Variable Reference](#environment-variable-reference)
16. [Troubleshooting](#troubleshooting)

---

## Prerequisites

| Component | Used for | Required for |
|---|---|---|
| **Vitis HLS 2023.2** (or compatible) | `csynth_design`, `csim_design`, `cosim_design` | All runs |
| **Python 3.10+** | Orchestrator + helpers | All runs |
| **g++** | Phase A compile-check | All runs |
| **LLM backend** (Anthropic / OpenAI / vLLM) | Phase B translation | LLM runs only |
| **XRT 2023.2** (user-mode tarball OK) | `v++` host runtime | hw_emu only |
| **U280 dev platform** (`xilinx_u280_gen3x16_xdma_1_202211_1.xpfm`) | `v++` link target | hw_emu only |
| **Khronos OpenCL headers** (`CL/cl.h`, `CL/cl2.hpp`) | host program compile | hw_emu only |

The pipeline runs end-to-end **without** XRT / platform / OpenCL headers — those
are only needed for `make check TARGET=sw_emu/hw_emu` on nova benchmarks.

---

## Environment Setup

### Step 1: Install Vitis HLS

```bash
chmod +x FPGAs_AdaptiveSoCs_Unified_2023.2_*_Lin64.bin
./FPGAs_AdaptiveSoCs_Unified_2023.2_*_Lin64.bin --noexec --keep --nox11 \
    --target /tmp/xinstall_2023.2_extract
/tmp/xinstall_2023.2_extract/xsetup -b AuthTokenGen
# Edit install_2023.2_config.txt: enable only "Virtex UltraScale+ HBM:1"
/tmp/xinstall_2023.2_extract/xsetup -a XilinxEULA,3rdPartyEULA -b Install \
    -c install_2023.2_config.txt
```

A pre-tested config template ships as [install_2023.2_config.txt](install_2023.2_config.txt).

### Step 2: Install XRT + U280 Platform (optional, hw_emu only)

```bash
mkdir -p /path/to/XRT_2023.2 && cd /path/to/XRT_2023.2
dpkg-deb -x xrt_202320.2.16.204_22.04-amd64-xrt.deb .
mkdir -p /path/to/U280_PLATFORM
dpkg-deb -x xilinx-u280-gen3x16-xdma-1-202211-1-dev_*_all.deb /path/to/U280_PLATFORM
# Khronos OpenCL headers (open source):
mkdir -p /path/to/opencl_headers/CL
# ... see scripts/setup_emu_env.sh for the curl commands
```

Then edit `scripts/setup_emu_env.sh` with your install paths.

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

Create `.env` at the repo root:

```bash
ANTHROPIC_API_KEY=sk-ant-...
# Optional
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=http://127.0.0.1:8000/v1   # local vLLM endpoint
```

The pipeline auto-routes by model id: Claude models → Anthropic SDK,
GPT models → OpenAI SDK, anything else → vLLM via OpenAI-compat API.

### Step 6: Typical `.env` for U280 + Vitis 2023.2 + Claude

```bash
ANTHROPIC_API_KEY=sk-ant-...
C2HLS_VITIS_SETTINGS=/mnt/data/luo00466/Xilinx/Vitis/2023.2/settings64.sh
C2HLS_PART=xcu280-fsvh2892-2L-e
C2HLS_CLOCK_NS=3.33
C2HLS_VITIS_VERSION=2023.2
C2HLS_FLOW_TARGET=vitis
```

---

## Benchmark Preparation

### Pre-built corpus (recommended)

The `benchmarks/` directory ships **22 prepared benchmarks** — no
preparation step needed, skip to [Running Translations](#running-translations).

| Source | Count | Benchmarks |
|--------|------:|-----------|
| rodinia-hls | 9 | StreamCluster, hotspot, kmeans, knn, lavaMD, lud, nw, pathfinder, srad |
| ML4Accel-Dataset | 8 | aes, fft, gemm_ncubed, md_knn, sort_merge, spmv_crs, stencil2D, viterbi |
| rodinia-hls-nova | 5 | cfd_flux, cfd_step_factor, lc_dilate, lc_gicov, lc_mgvf |

### Regenerating from upstream

```bash
git clone https://github.com/UCLA-VAST/rodinia-hls.git ~/rodinia-hls
git clone <nova-repo> ~/rodinia-hls-nova
git clone https://github.com/UIUC-ChenLab/ML4Accel-Dataset.git ~/ML4Accel-Dataset

# Edit source-path constants near the top of each script:
python prepare_benchmarks.py        # rodinia-hls + ML4Accel
python prepare_nova_benchmarks.py   # nova: cfd_flux, cfd_step_factor, lc_*
```

Each benchmark directory contains:
- `plain.cpp` — pragma-free C input (LLM input)
- `gold_hls_source.cpp` — original upstream HLS source
- `hls_baseline.cpp` — localised gold (GT for single-shot mode)
- `hls_<bench>_<N>_<step>.cpp` — per-step GT variants for multistep mode
- `<bench>.h` — cleaned header
- `metadata.json` — provenance, variant list, sha256 digest

---

## Running Translations

### Single-shot mode

```bash
python c2hls.py --bench aes --model claude-haiku-4-5-20251001 --turns 3
```

### Multi-step mode (recommended)

```bash
python c2hls.py --bench knn --multistep \
    --model claude-haiku-4-5-20251001 \
    --strategy dynamic --turns 4 \
    --out results_phase2/knn_haiku_u280_v2023
```

The `--strategy dynamic` flag enables the bottleneck router (Pillar 5),
which selects the next optimisation step based on the synthesised bottleneck
profile rather than the fixed `tiling→pipeline→unroll→doublebuffer→coalescing`
order.

### Multi-step with all Phase 8–11 features enabled

```bash
export C2HLS_PHASE8_BASELINE_ALIGN=1    # Phase 8: retranslate if baseline < GT quality
export C2HLS_PHASE8_FMAX_FLOOR=0.80     # Phase 11: reject baselines with Fmax < 80% of GT
export C2HLS_PHASE5_GT_PREPOP=1         # Pre-synthesise GT variants before opt loop
export C2HLS_PHASE7A=1                  # Harvest static report data (burst.xml etc.)

python c2hls.py --bench StreamCluster --multistep \
    --model claude-sonnet-4-6 \
    --strategy dynamic --turns 4 \
    --out results_phase2/streamcluster_sonnet_u280
```

### Hardware emulation post-step

```bash
source scripts/setup_emu_env.sh
export C2HLS_HW_EMU_FINAL=1
python c2hls.py --bench knn --multistep --model claude-haiku-4-5-20251001
```

### Run all benchmarks

```bash
python c2hls.py --all --multistep --model claude-haiku-4-5-20251001
```

### Per-agent model routing

Run cheap Haiku for translation and expensive Sonnet for repair:

```bash
export C2HLS_TRANSLATOR_MODEL=claude-haiku-4-5-20251001
export C2HLS_SYNTHESIS_MODEL=claude-haiku-4-5-20251001
export C2HLS_QUALITY_REPAIR_MODEL=claude-sonnet-4-6
python c2hls.py --bench knn --multistep
```

### CLI reference

| Flag | Default | Description |
|---|---|---|
| `--bench NAME` | required | Benchmark name (matches `benchmarks/<name>/`) |
| `--all` | — | Run every benchmark in `benchmarks/index.json` |
| `--multistep` | off | Incremental per-step optimisation chain |
| `--steps S1,S2,...` | all 5 steps | Custom step order for multistep |
| `--strategy dynamic\|static` | `static` | Step selection: fixed order or bottleneck-routed |
| `--model ID` | `$C2HLS_MODEL` | LLM model id (auto-routed by name) |
| `--turns N` | `3` | Max repair attempts per phase/step |
| `--out PATH` | `results/` or `results_multistep/` | Output directory |

---

## Pipeline Architecture

### Phase sequence (multistep)

```
Phase A  — compile-check plain.cpp with g++
Phase B  — initial HLS translation (TranslatorAgent)
Phase 8  — baseline alignment: if agent baseline > 1.20× GT cycles OR
           Fmax < floor×GT_fmax, retranslate with gap feedback (up to 4 attempts)
Phase 5b — pre-synthesise all GT step variants into cache (gated by env var)
Phase C  — compare baseline vs GT baseline (rubric)

For each optimisation step (tiling → pipeline → unroll → doublebuffer → coalescing):
  ├─ _optimization_step_attempt (SynthesisAgent):
  │    ├─ Prompt includes: {synth_report} with Pillar 1 per-loop bottlenecks
  │    │                   baseline-vs-current scope diff block
  │    │                   per-step resource constraint block
  │    │                   profile signal (timing/resource overflow flags)
  │    │                   skill library match (if dynamic strategy)
  │    ├─ LLM generates HLS code
  │    ├─ Compile-error repair loop (up to N attempts)
  │    ├─ Synthesis (csynth_design)
  │    └─ Phase 9: if csim/cosim fails → correctness-repair re-prompt
  │
  ├─ Regression guard (_step_regression_reasons):
  │    ├─ Tier 1: reject if latency > step_threshold OR ≥3 resources > per-step ceiling
  │    └─ Tier 2 override: accept if latency ≤ 0.5× AND all resources < device capacity
  │       (allows aggressive DSP parallelisation that halves latency)
  │
  ├─ GT-shape alignment (trajectory_alignment.py):
  │    └─ Keep enabling regressions consistent with GT trajectory shape
  │
  └─ Phase 6a: update best-so-far pointer across all completed steps

Phase 6a final: promote best-so-far step as the output
```

### TranslatorAgent

- **Phase A** — compile-check with `g++ -c`; fix on failure
- **Phase B** — emit HLS code via `q_translate_c_to_hls`, targeting
  `extern "C" workload()`, unified `bundle=control` s_axilite, and
  appropriate PIPELINE/UNROLL/ARRAY_PARTITION pragmas

### SynthesisAgent

Runs the synth/csim/cosim chain with full structured feedback:

- **Per-attempt history** — each repair prompt includes error class + first
  error line per prior attempt, breaking the LLM out of oscillation loops
- **Repair guidance** — pattern→hint mappings covering compile errors,
  `HLS 214-219` axilite-bundle-split, synthesis timeouts
- **Pillar 1 per-loop feedback** — `format_report_summary` includes top-6
  per-loop bottleneck records (II violations, pipeline-blocked warnings,
  interval > latency flags) extracted by `hls_feedback.py`
- **Baseline scope diff** — each step prompt includes a diff between the
  baseline's per-loop bottlenecks and the current step's, showing which
  loops regressed and which new bottlenecks were introduced
- **Per-step resource constraints** — explicit budget block (e.g.
  "DSP max=15, device_cap=9024") injected before code generation, plus
  guidance on when to prefer local-buffer staging vs DSP parallelisation
- **Profile-bottleneck signals** — `_build_profile_signal` flags timing
  violations, Fmax below target, and resource overflow
- **Regression guard** — per-step latency and resource ceilings
  (calibrated against GT reference; see `STEP_REGRESSION_THRESHOLDS`);
  two-tier override accepts latency-halving steps that fit on chip
- **Phase 9 correctness-repair** — if csim/cosim fails after csynth
  passes, re-prompt with the testbench failure log; up to 3 repair rounds
- **GT-shape-aware revert** (`trajectory_alignment.py`) — keeps
  intermediate regressions that match the GT trajectory shape (enabling
  steps that are a structural prerequisite for later gains)
- **Revert-on-streak** — N consecutive same-class errors trigger reversion
  to last-known-good state (`C2HLS_SYNTH_REVERT_THRESHOLD`)

### QualityRepairAgent

After Phase B accepts a code, runs up to N candidate generations driven
by quality guidance (metric-level comparison vs GT). Each candidate goes
through the full synth/csim/cosim chain; only candidates that preserve
correctness AND improve the focus metric are accepted.

### Optional hw_emu post-step

When `C2HLS_HW_EMU_FINAL=1`, after the loop completes,
`hls_eval.run_hw_emu_via_nova` stages a private copy of the nova
benchmark, swaps in the LLM's kernel, runs `make check TARGET=hw_emu`,
and parses `profile_kernels.csv` for authoritative kernel runtime.

---

## Per-loop Feedback System (Pillar 1)

Implemented in [`hls_feedback.py`](hls_feedback.py) and wired into
`hls_eval.run_hls_synthesis` and `format_report_summary` in `hls_eval.py`.

### What is collected

After every `csynth_design` run, `attach_feedback` parses:

- **Per-scope records** from `csynth.rpt` loop table and `csynth.xml`:
  `{scope_id, kind, II_target, II_achieved, trip_count, pipelined,
  issue, violation, slack_ns, bram, dsp, ff, lut}`
- **Scheduler-blame strings** from `vitis_hls.log`:
  "Unable to enforce II=1 due to recurrence on `accum`" → typed as
  `loop_carried_dep`; "Memory port limit exceeded" → `port_conflict`;
  "cannot pipeline" → `pipeline_blocked`
- **Derived bottleneck records**: `{scope_id, kind, evidence, severity,
  metric, source_location}` — the structured signal downstream agents act on
- **Burst / diagnostic reports** (Phase 7a): `burst.xml`,
  `fe_messages.xml`, `be_messages.xml`, `csynth_design_size.rpt`

### How it reaches the LLM

`format_report_summary(report)` now appends:

```
per_loop_bottlenecks (top issues limiting performance):
  - workload/compute_dist/VITIS_LOOP_28: ii_target_miss | pipelined loop
    achieved II=64 (>1) [kernel.cpp:28]
  - workload: interval_exceeds_latency | interval 149838 > latency 149837
    cycles [kernel.cpp:3]
```

Additionally, every optimization step prompt receives two extra blocks:

1. **Baseline-vs-current scope diff** — `_render_baseline_scope_diff`
   shows which loops appeared as new bottlenecks since the baseline and
   which resolved, with a latency ratio and explicit target instruction.

2. **Per-step resource constraints** — `_render_step_resource_constraints`
   shows the exact DSP/BRAM/FF/LUT budget for this step, the two-tier
   override rule, and strategy guidance for AXI II violations:

```
RESOURCE CONSTRAINTS for the `tiling` step:
  dsp   : current=    12 → max=     15 (limit 1.30×)  device_cap=9024
  bram  : current=    30 → max=    120 (limit 4.00×)  device_cap=4032
  ...
Two-tier override: if latency_ns ≤ 0.5× AND fits on chip, accepted.
Strategy: prefer LOCAL BUFFER staging over unrolling for AXI II violations.
```

---

## Skill Library (Pillar 3)

Implemented in [`skill_library.py`](skill_library.py).

A YAML/JSON store of confidence-tagged pattern→strategy entries, loaded
by the orchestrator and queried per bottleneck kind at each step.

### Entry schema

```python
{
  "id":          "axi_burst_widening",
  "pattern":     "m_axi port in pipelined loop; latency dominated by DRAM bandwidth",
  "strategy":    "Add max_read_burst_length=64 num_read_outstanding=16 to m_axi pragma",
  "template":    "#pragma HLS INTERFACE m_axi port=X bundle=gmem max_read_burst_length=64 ...",
  "confidence":  "high",          # high / medium / low / avoid
  "kind":        ["ii_target_miss", "port_conflict"],
  "vitis_versions": ["2023.2", "2025.2"],
  "fpgas":       ["xcu280-fsvh2892-2L-e"],
  "avoid":       "Do NOT combine with array_partition cyclic",
  "stats":       {"occurrences": 0, "sec_pass": 0, "mean_advantage": null}
}
```

### Querying

```python
from skill_library import SkillLibrary
lib = SkillLibrary("skills.yaml")
matches = lib.query(bottleneck_kind="ii_target_miss", vitis_version="2023.2",
                    fpga="xcu280-fsvh2892-2L-e")
block = render_skill_set_for_prompt(matches, max_skills=2)
```

Matched skills are injected into the optimization prompt when
`--strategy dynamic` is active and the bottleneck router identifies a match.

### Updating skill statistics

After each step completes, the advantage signal
(`latency_improvement / baseline_latency`) is recorded against matched
skills. Skills with consistently negative advantage are demoted toward
`avoid`.

---

## Bottleneck Router (Pillar 5)

Implemented in [`bottleneck_router.py`](bottleneck_router.py).

When `--strategy dynamic` is active, replaces the fixed
`tiling→pipeline→unroll→doublebuffer→coalescing` order with a
bottleneck-driven selection:

| Bottleneck kind | Preferred next step |
|----------------|-------------------|
| `ii_target_miss` from loop-carried dep | `pipeline` → reassociate |
| `port_conflict` on array | `tiling` → array_partition + local buffer |
| Large trip-count hot loop | `tiling` or `doublebuffer` |
| Low Fmax / long combinational | `pipeline` (pipeline depth) |
| Bandwidth-limited (AXI) | `coalescing` → burst widening |

The router reads `feedback["bottlenecks"]` from the current step's report
and returns a `RoutingDecision(step_name, bottleneck_kind, rationale)`.

---

## Verifying Nova Benchmarks

Run [run_nova_direct_emu.py](run_nova_direct_emu.py) to validate that
your Vitis + XRT + U280 install reproduces upstream hw_emu cycle counts.

```bash
source scripts/setup_emu_env.sh
export C2HLS_HW_EMU_STEPS=baseline,coalescing
export C2HLS_HW_EMU_CLOCK_NS=3.33
python run_nova_direct_emu.py
```

Pre-computed reference numbers live in
[results/references_philip/](results/references_philip/).

Typical validation result: cycle ratios within 0.1 % of upstream
(integer-rounding from `cycles = int(us × 300.30)`).

---

## Evaluation & Scoring

### Rubric

```bash
python rubric.py --results results          # single-shot
python rubric.py --results results_multistep --multistep
```

### Metrics (9-point)

| Metric | Weight | Description |
|---|---|---|
| Synth status | gate | Pass/fail |
| Csim correctness | gate | Must match GT output |
| Cosim correctness | gate | RTL sim match |
| Latency (`latency_ns`) | 30 % | Average-case real-time latency |
| Fmax | 10 % | `1000 / EstimatedClockPeriod` |
| LUT / FF / BRAM / DSP | 10 % each | Resource usage vs GT |
| ADP composite | 10 % | `latency × normalised_area` |
| Device feasibility | 10 % | Fits target device |

### Best-vs-best scoring (multistep)

In multistep mode, the headline metric is:

```
cycles_ratio = agent_best_cycles / gt_ref_best_cycles
```

where `agent_best_cycles` is the lowest-cycle csim-passing step across
the full trajectory (Phase 6a best-so-far), and `gt_ref_best_cycles`
is the GT coalescing variant's cycle count.

---

## Dataset Pipeline

The [`dataset_pipeline/`](dataset_pipeline/) package implements the
C2HLS-Trajectory dataset (Pillar 8): per-step trajectory records usable
for training and evaluating HLS optimization agents.

### Schema (v2.0)

```python
{
  "kernel_id":        "knn",
  "version":          "2023.2",
  "fpga":             "xcu280-fsvh2892-2L-e",
  "step_name":        "doublebuffer",
  "parent_hash":      "<canonical AST hash of parent step>",
  "candidate_hash":   "<canonical AST hash of this code>",
  "pragma_diff":      "<unified diff of pragma lines>",
  "bottleneck_record": {...},         # Pillar 1 typed bottleneck
  "csynth_metrics_per_scope": [...],  # per-loop II / slack / resources
  "csim_pass":        true,
  "cosim_pass":       null,
  "relative_advantage": 0.127,        # (prev_lat - new_lat) / baseline_lat
  "skill_hits":       ["axi_burst_widening"],
  "status":           "improved",     # improved / regressed / absorbed / errored
  "rationale":        "..."           # LLM-emitted explanation
}
```

### Generating records

```bash
cd dataset_pipeline
python replay.py --bench knn --results ../results_phase2/knn_haiku_phase9_u280_v2023
python merge.py --output ../dataset/c2hls_trajectory_v2_2023.2_u280.jsonl
```

---

## JSONL Export & Comparison

```bash
python export_schema_jsonl.py \
    --results results \
    --multistep results_multistep \
    --benchmarks benchmarks \
    --output artifacts/
```

Output: `artifacts/schema_records.jsonl` with schema-1.0 records
(`sw_run` / `hls_synth` / `rtl_sim`), paired AI + GT records per step,
and `origin_meta` carrying model attribution and phase.

---

## Benchmark Corpus

### 22 benchmarks

| Benchmark | Source | csim | hw_emu |
|---|---|:---:|:---:|
| StreamCluster | rodinia-hls | — | — |
| hotspot | rodinia-hls | — | — |
| kmeans | rodinia-hls | — | — |
| knn | rodinia-hls | — | ✓ |
| lavaMD | rodinia-hls | — | — |
| lud | rodinia-hls | — | (slow) |
| nw | rodinia-hls | ✓ | ✓ |
| pathfinder | rodinia-hls | — | ✓ |
| srad | rodinia-hls | — | — |
| aes | ML4Accel | ✓ | — |
| fft | ML4Accel | ✓ | — |
| gemm_ncubed | ML4Accel | ✓ | — |
| md_knn | ML4Accel | ✓ | — |
| sort_merge | ML4Accel | ✓ | — |
| spmv_crs | ML4Accel | ✓ | — |
| stencil2D | ML4Accel | ✓ | — |
| viterbi | ML4Accel | ✓ | — |
| **cfd_flux** | rodinia-hls-nova | — | ✓ |
| **cfd_step_factor** | rodinia-hls-nova | — | ✓ |
| **lc_dilate** | rodinia-hls-nova | — | ✓ |
| **lc_gicov** | rodinia-hls-nova | — | ✓ |
| **lc_mgvf** | rodinia-hls-nova | — | ✓ |

### Directory structure

```
benchmarks/
├── index.json                        # Corpus manifest (22 entries)
├── knn/
│   ├── knn.h                         # Cleaned header
│   ├── plain.cpp                     # Pragma-free C input (LLM input)
│   ├── gold_hls_source.cpp           # Original upstream HLS
│   ├── hls_baseline.cpp              # Localised gold (GT for single-shot)
│   ├── hls_knn_1_tiling.cpp          # Per-step GT variants
│   ├── hls_knn_2_pipeline.cpp
│   ├── hls_knn_3_unroll.cpp
│   ├── hls_knn_4_doublebuffer.cpp
│   ├── hls_knn_5_coalescing.cpp      # <- GT ref-best (262,480 cyc on U280)
│   └── metadata.json
└── ...

results_phase2/                       # Multistep results (gitignored)
└── knn_haiku_phase9_u280_v2023/
    ├── knn_multistep_results.json    # Full trajectory + Phase 6a best
    ├── knn_final.cpp                 # Accepted final code
    ├── knn_history.json              # LLM conversation transcript
    └── steps/                        # Per-step code + reports

artifacts/                            # Markdown + comparison reports
├── phase9_e2e_knn_haiku_vs_sonnet_u280_v2023.md
├── phase10_pillar1_scope_feedback_comparison.md
├── phase11_sc_sonnet_comparison_and_pillar3_decision.md
└── ...
```

---

## File Reference

### Core pipeline

| File | Purpose |
|---|---|
| [c2hls.py](c2hls.py) | Main orchestrator — `C2HLSOrchestrator`, `TranslatorAgent`, `SynthesisAgent`, `QualityRepairAgent`, all phases (8/9/6a), multistep loop, regression guard |
| [hls_eval.py](hls_eval.py) | Vitis HLS runner: `run_hls_synthesis`, `run_csim`, `run_cosim`, `run_hw_emu_via_nova`, `format_report_summary` (includes Pillar 1 bottlenecks) |
| [hls_feedback.py](hls_feedback.py) | **Pillar 1**: per-scope II / slack / issue extraction from `csynth.rpt` / `csynth.xml` / `vitis_hls.log`; `attach_feedback`, `derive_bottleneck_records` |
| [prompt_c2hls.py](prompt_c2hls.py) | All LLM prompts — system instructions, Phase A/B/quality-repair templates, per-step optimisation prompts, correctness-repair prompt |
| [rubric.py](rubric.py) | 9-metric scoring rubric; `_device_limits_for_part` for U280/U50/Artix-7 capacity tables |
| [skill_library.py](skill_library.py) | **Pillar 3**: confidence-tagged pattern→strategy skill store; `SkillLibrary.query`, `render_skill_set_for_prompt`, advantage-based confidence updates |
| [bottleneck_router.py](bottleneck_router.py) | **Pillar 5**: bottleneck→step routing; `select_next_step`, `RoutingDecision` |
| [trajectory_alignment.py](trajectory_alignment.py) | GT-shape consistency check for enabling regressions; `is_consistent_with_gt_trajectory` |
| [candidate_cache.py](candidate_cache.py) | **Pillar 4**: canonical-AST hash → synthesis result cache (sqlite); deduplicates identical edits across reruns |
| [report.py](report.py) | HTML report generator |

### Benchmark prep & validation

| File | Purpose |
|---|---|
| [prepare_benchmarks.py](prepare_benchmarks.py) | Generates `benchmarks/` from rodinia-hls + ML4Accel upstream repos |
| [prepare_nova_benchmarks.py](prepare_nova_benchmarks.py) | Adds nova benchmarks (cfd_flux, cfd_step_factor, lc_dilate, lc_gicov, lc_mgvf) |
| [validate_corpus.py](validate_corpus.py) | Sanity-checks corpus (no pragma leakage in plain.cpp, signature compat) |
| [verify_corpus_stability.py](verify_corpus_stability.py) | Repeat-N csynth for Vitis determinism measurement |

### hw_emu / reference validation

| File | Purpose |
|---|---|
| [run_nova_direct_emu.py](run_nova_direct_emu.py) | Direct sw_emu/hw_emu on nova benches — validates cycle counts vs reference |
| [run_hw_emu.py](run_hw_emu.py) | Standalone hw_emu wrapper for a single bench/variant |
| [run_2023_2_synth_comparison.py](run_2023_2_synth_comparison.py) | Direct csynth comparison vs upstream reference |
| [build_hwemu_reference_candidate.py](build_hwemu_reference_candidate.py) | Builds hw_emu reference candidate for a given benchmark |
| [compare_jsonl_to_references.py](compare_jsonl_to_references.py) | Delta table between a generated JSONL and the reference JSONL |
| [robustness.py](robustness.py) | Robustness testing utilities (multi-run variance, noise injection) |

### Multi-bench run drivers

| File | Purpose |
|---|---|
| [run_multistep_haiku.py](run_multistep_haiku.py) | 4-bench multistep with Haiku 4.5 |
| [run_multistep_hwemu.py](run_multistep_hwemu.py) | 4-bench multistep + hw_emu post-step |
| [run_3bench_haiku_sonnet.py](run_3bench_haiku_sonnet.py) | pathfinder/knn/nw with Haiku + Sonnet cross-model comparison |
| [run_remaining_haiku.py](run_remaining_haiku.py) | All 14 benches not in the 3-bench set |
| [run_requested_hwemu_matrix.py](run_requested_hwemu_matrix.py) | hw_emu matrix across bench × variant |
| [run_requested_agentic_hwemu_smoke.py](run_requested_agentic_hwemu_smoke.py) | Agentic multistep + hw_emu smoke test |

### Dataset pipeline

| File | Purpose |
|---|---|
| [dataset_pipeline/schema.py](dataset_pipeline/schema.py) | Trajectory record schema (v2.0) |
| [dataset_pipeline/recorder.py](dataset_pipeline/recorder.py) | Records step events during live runs |
| [dataset_pipeline/replay.py](dataset_pipeline/replay.py) | Replays existing results into schema records |
| [dataset_pipeline/merge.py](dataset_pipeline/merge.py) | Merges records across runs into a single JSONL |
| [dataset_pipeline/external_adapter.py](dataset_pipeline/external_adapter.py) | Adapts HLSFactory / HLSyn records to c2hls schema |

### JSONL export

| File | Purpose |
|---|---|
| [export_schema_jsonl.py](export_schema_jsonl.py) | Canonical schema-1.0 JSONL from `results/` + `results_multistep/` |

### Setup helpers

| File | Purpose |
|---|---|
| [scripts/setup_emu_env.sh](scripts/setup_emu_env.sh) | Sources Vitis settings, XRT, sets PLATFORM_REPO_PATHS + CPLUS_INCLUDE_PATH |
| [install_2023.2_config.txt](install_2023.2_config.txt) | Minimal Vitis 2023.2 install config (HLS only + Virtex UltraScale+ HBM) |

### Reference data

| Path | Purpose |
|---|---|
| [results/references_philip/](results/references_philip/) | sw_emu + hw_emu JSONL references for 17 nova benchmarks |

---

## Environment Variable Reference

### Vitis / hardware

| Variable | Default | Purpose |
|---|---|---|
| `C2HLS_VITIS_SETTINGS` | *(required)* | Path to `settings64.sh` sourced before every synth call |
| `C2HLS_VITIS_VERSION` | `2023.2` | Version string (injected into reports and skill-library version tags) |
| `C2HLS_PART` | `xc7a100t-csg324-1` | Target FPGA part id |
| `C2HLS_CLOCK_NS` | `4` | Target clock period for `create_clock` (3.33 = 300 MHz) |
| `C2HLS_FLOW_TARGET` | `vitis` | Vitis HLS flow: `vitis` (kernel/v++) or `vivado` (raw IP) |
| `C2HLS_SYNTH_TIMEOUT` | `1200` | Wall-time budget for `csynth_design` (seconds) |
| `C2HLS_CSIM_TIMEOUT` | `180` | Wall-time budget for `csim_design` |
| `C2HLS_COSIM_TIMEOUT` | `1200` | Wall-time budget for `cosim_design` |

### Phase 8 — Baseline alignment

| Variable | Default | Purpose |
|---|---|---|
| `C2HLS_PHASE8_BASELINE_ALIGN` | `0` | Enable Phase 8 retranslation loop |
| `C2HLS_PHASE8_FMAX_FLOOR` | `0.80` | Reject baseline if `agent_fmax < floor × gt_fmax`; catches structurally slow translations that happen to match GT cycle count |
| `C2HLS_PHASE8_MAX_ALIGN_ATTEMPTS` | `4` | Max retranslation attempts before giving up and accepting best seen |

### Phase 9 — Correctness repair

| Variable | Default | Purpose |
|---|---|---|
| `C2HLS_DISABLE_CORRECTNESS_REPAIR` | `0` | Set to `1` to disable Phase 9 csim/cosim repair loop |

### Multistep optimisation

| Variable | Default | Purpose |
|---|---|---|
| `C2HLS_PHASE5_GT_PREPOP` | `0` | Pre-synthesise all GT step variants into cache before the optimisation loop |
| `C2HLS_PHASE7A` | `0` | Harvest static report data (burst.xml, fe_messages.xml, etc.) after each step |
| `C2HLS_GT_AWARE_REVERT` | `1` | Keep enabling regressions consistent with GT trajectory shape |
| `C2HLS_STEP_REGRESSION_THRESHOLD` | *(per-step)* | Global override for regression guard latency threshold (overrides per-step calibration) |
| `C2HLS_STEP_REGRESSION_THRESHOLDS_JSON` | — | Full per-step threshold dict override as JSON string |

### LLM / agents

| Variable | Default | Purpose |
|---|---|---|
| `C2HLS_MODEL` | `claude-haiku-4-5-20251001` | Default model id for all agents |
| `C2HLS_TRANSLATOR_MODEL` | `= C2HLS_MODEL` | Override TranslatorAgent model only |
| `C2HLS_SYNTHESIS_MODEL` | `= C2HLS_MODEL` | Override SynthesisAgent model only |
| `C2HLS_QUALITY_REPAIR_MODEL` | `= C2HLS_MODEL` | Override QualityRepairAgent model only |
| `C2HLS_FEEDBACK_MODEL` | `= C2HLS_MODEL` | Override FeedbackAgent (LLM-aided regression composition) |
| `C2HLS_QUALITY_REPAIR_TURNS` | `2` | Max candidate attempts in quality-repair loop |
| `C2HLS_SYNTH_REVERT_THRESHOLD` | `0` | Revert-on-streak: N consecutive same-class errors trigger revert (0 = disabled) |
| `C2HLS_VERIFY_RUNS` | `1` | When > 1, repeat csynth N times and average (stability measurement) |

### hw_emu

| Variable | Default | Purpose |
|---|---|---|
| `C2HLS_HW_EMU_FINAL` | `0` | Run hw_emu post-step on final accepted code |
| `C2HLS_HW_EMU_CLOCK_NS` | `3.33` | Clock period for hw_emu cycle conversion (U280 platform = 3.33) |
| `C2HLS_HW_EMU_TIMEOUT` | `21600` | Wall-time budget for one `make check TARGET=hw_emu` (6 hours) |
| `C2HLS_HW_EMU_STEPS` | `baseline,coalescing` | Which variants to hw_emu in `run_nova_direct_emu.py` |

---

## Troubleshooting

### Vitis HLS not found

```bash
source /path/to/Xilinx/Vitis/2023.2/settings64.sh
# or
export C2HLS_VITIS_SETTINGS=/path/to/Xilinx/Vitis/2023.2/settings64.sh
```

### "Part 'xcu280-fsvh2892-2L-e' is not installed"

Re-run the Vitis installer with `Modules=Virtex UltraScale+ HBM:1`:

```bash
xsetup -b Add -c install_2023.2_config.txt
```

### Phase 8 baseline keeps failing alignment

If Phase 8 retranslates 4 times and gives up with a high-latency baseline,
the LLM is producing structurally GT-like translations and can't find the
optimised shortcut. Lower the cycle tolerance or increase max attempts:

```bash
export C2HLS_PHASE8_MAX_ALIGN_ATTEMPTS=8
```

Or check whether the GT reference used for alignment is the coalescing
variant (fast) vs the unoptimised baseline (slow) — Phase C's GT
synthesis target determines the comparison point.

### `mc.h: No such file or directory` in coalescing step

The coalescing code includes `../../../common/mc.h`. This is automatically
repaired by the compile-error loop (one LLM call). If it exhausts attempts
(usually S_AXILITE bundle errors follow mc.h), check the `BUNDLES` pragma
style — Vitis 2023.2 requires all ports on a single `bundle=control`.

### Synthesis timeout on GT doublebuffer

GT doublebuffer synthesis can exceed 20 min on complex kernels (knn).
The `C2HLS_SYNTH_TIMEOUT` applies to both agent and GT synthesis. Add a
separate GT-only cap (planned as `C2HLS_GT_SYNTH_TIMEOUT`):

```bash
export C2HLS_SYNTH_TIMEOUT=2400        # agent: allow 40 min
# GT synthesis uses same var; workaround: kill via ps grep on the temp dir
```

### "undef" latency in csynth reports

Variable trip-count loops cause `undef` top-level latency. The pipeline
uses `Average-caseRealTimeLatency`, falling back to max-loop latency from
the text report. For variable-length designs, cycles may show `None` —
Phase 6a uses `latency_ns` (wall-clock) for comparison in this case.

### Qwen models on vLLM

Qwen models require `enable_thinking: false` via `extra_body`. The
pipeline applies this automatically for model ids matching `qwen`
(case-insensitive).

### hw_emu `CL/cl.h not found`

Khronos OpenCL headers aren't on the include path. Install
`opencl-headers` via apt or download from Khronos and set
`CPLUS_INCLUDE_PATH` — see [scripts/setup_emu_env.sh](scripts/setup_emu_env.sh).

### hw_emu `bits/wordsize.h not found`

Ubuntu 22.04 multi-arch glibc issue. Add
`/usr/include/x86_64-linux-gnu` to `C_INCLUDE_PATH` / `CPLUS_INCLUDE_PATH`
(handled by `setup_emu_env.sh`).
