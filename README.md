# C2HLS Agentic Translation Framework

C2HLS is a multi-agent framework for translating plain C/C++ kernels into
Vitis HLS kernels, then validating and scoring the generated designs against
reference HLS implementations. Its main target today is Rodinia/Rodinia-Nova
kernel optimization on Vitis 2023.2 / U280, with external-dataset support for
HLSFactory and hls-eval style C kernels.

The project is intentionally more than a prompt wrapper. It contains a corpus
layer, a reference-validation layer, an agentic translation and optimization
loop, Vitis synthesis/csim/cosim runners, optional Nova `sw_emu`/`hw_emu`
measurement, and canonical JSONL export for side-by-side comparison.

## Overview

The normal input is `plain.cpp`: a benchmark kernel with HLS pragmas and
platform-specific optimization code stripped. The framework asks the agents to
recover a legal Vitis kernel first, then improve it through staged HLS
optimization. Reference results come either from local Vitis runs or, for
trusted Rodinia/Rodinia-Nova entries, from direct JSONL artifacts produced by
known-good benchmark runs.

```
upstream HLS benchmark
        |
        v
benchmarks/<name>/
  plain.cpp            # LLM input
  metadata.json        # source repo, variants, provenance
  hls_<step>.cpp       # reference HLS variants when available
        |
        v
TranslatorAgent -> SynthesisAgent -> QualityRepairAgent
        |
        v
generated HLS + csynth/csim/cosim reports + optional hw_emu profile
        |
        v
schema-1.0 JSONL + markdown summaries + reference deltas
```

### Architecture at a glance

| Layer | Main files | Responsibility |
|---|---|---|
| Corpus and manifest | `benchmarks/`, `benchmarks_external/`, `prepare_*.py` | Materialize stripped C inputs, headers, testbenches, metadata, and reference variants |
| Reference validation | `c2hls.py`, `run_nova_direct_emu.py`, `results/references_philip/` | Prove or trust GT references; Rodinia/Rodinia-Nova can use direct JSONL evidence via `trusted_external` |
| Agentic workflow | `c2hls.py`, `prompt_c2hls.py` | Phase A compile repair, Phase B translation, multistep optimization, correctness repair |
| Performance feedback | `hls_eval.py`, `hls_feedback.py`, `bottleneck_router.py`, `skill_library.py` | Parse Vitis reports, classify bottlenecks, route steps, inject skills and resource constraints |
| Experiment drivers | `run_agentic_sweep.py`, `run_*smoke*.py`, `run_requested_hwemu_matrix.py` | Launch repeatable sweeps and direct-reference matrix runs |
| Results contract | `export_schema_jsonl.py`, `compare_jsonl_to_references.py`, `scripts/validate_jsonl_semantics.py` | Emit and validate reference-compatible schema-1.0 JSONL |

### Multi-step optimization chain

In multistep mode Phase B is deliberately conservative by default:
`C2HLS_PHASEB_MODE=functional` asks the translator to produce a correct,
testbench-compatible kernel, not an aggressively optimized one. The later step
agents then apply the same broad optimization sequence used by the reference
benchmark families.

```
plain.cpp
  |
  v
Phase A compile repair
  |
  v
Phase B functional HLS baseline
  |
  +--> optional Phase 8 baseline alignment
  |
  v
tiling -> pipeline -> unroll -> doublebuffer -> coalescing
  |         |          |         |              |
  +---------+----------+---------+--------------+
                  Phase 6a best-so-far selection
```

Each optimization step may evaluate multiple independent candidates and, in
exhaustive mode, multiple synth-tested attempts per candidate. The sweep mode
used for recent large runs is:

```
C2HLS_CANDIDATES_PER_STEP=5
C2HLS_ATTEMPTS_PER_CANDIDATE=5
C2HLS_EXHAUSTIVE_CANDIDATE_ATTEMPTS=1
```

The saved result records selected candidate/attempt indices plus min/max/avg
telemetry, so bad attempts are visible rather than silently dropped.

### Evaluation isolation policy

Reference HLS code and absolute reference metrics are controller-side evidence,
not prompt material. The agents may see their own synthesis/csim/cosim reports,
configured target context, device utilization, bottleneck diagnostics, curated
skills, and ratio-only directional feedback such as generated/reference latency
or Fmax ratios. Phase 8 baseline alignment and quality repair follow the same
rule: they can ask for a better structure, but they must not expose reference
source, exact reference cycle counts, or exact reference resource counts.

The default hardware target is Vitis 2023.2 on U280
(`xcu280-fsvh2892-2L-e`) at 3.33 ns. Unknown target/device information is
recorded as fallback metadata instead of being silently treated as successful
evidence.

### Agent decomposition

The orchestrator coordinates three agent roles:

- **TranslatorAgent**: Phase A input compile-check and Phase B initial HLS
  translation.
- **SynthesisAgent**: Vitis compile/synthesis/csim/cosim loop, structured
  error repair, Phase B correctness gating, per-loop bottleneck feedback,
  regression guard, and Phase 9 correctness repair.
- **QualityRepairAgent**: optional post-synthesis improvement loop that
  generates and accepts only candidates that preserve correctness while
  improving the focused quality metric.

The intended end-to-end workflow is therefore not "one prompt creates final
HLS." It is a measured trajectory: recover functionality, synthesize, inspect
feedback, optimize step by step, and export every material status.

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
16. [Performance Hardening Roadmap](#performance-hardening-roadmap)
17. [Troubleshooting](#troubleshooting)

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

### Flash mode

Flash mode keeps Phase B as a functional HLS baseline, then runs exactly one
aggressive all-in optimisation step. It reuses the normal candidate/attempt
search, CSim correctness gates, regression guard, and timing-clean selection,
but avoids the five-step trajectory. This is useful for compact
HLSFactory/PolyBench-style kernels where a simple pipeline/unroll/local-buffer
rewrite often beats a full Rodinia-style path.

```bash
python c2hls.py --bench-dir benchmarks_external/HLSFactory/polybench_float_small/hlsfactory_2mm \
    --multistep --strategy flash \
    --model claude-haiku-4-5-20251001 --turns 4 \
    --candidates-per-step 5 --attempts-per-candidate 5 \
    --exhaustive-candidate-attempts
```

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

### Agentic sweep runner

[run_agentic_sweep.py](run_agentic_sweep.py) is the preferred driver for
repeatable benchmark sweeps. It writes a live log, per-benchmark result dirs,
a summary JSON, a markdown table, and validated schema-1.0 JSONL after every
completed benchmark.

```bash
env \
  C2HLS_SWEEP_STAMP=rodinia_nova_haiku_u280 \
  C2HLS_SWEEP_BENCHES=knn,lud,nw,pathfinder \
  C2HLS_SWEEP_MODELS=haiku \
  C2HLS_SWEEP_HW_EMU=1 \
  C2HLS_SWEEP_REFERENCE_VALIDATE_MODE=trusted_external \
  C2HLS_SWEEP_CANDIDATES_PER_STEP=5 \
  C2HLS_SWEEP_ATTEMPTS_PER_CANDIDATE=5 \
  C2HLS_SWEEP_EXHAUSTIVE_CANDIDATE_ATTEMPTS=1 \
  /home/luo00466/.conda/envs/py310_2/bin/python run_agentic_sweep.py
```

For Rodinia/Rodinia-Nova sweeps, `trusted_external` avoids re-running CSim on
known-good reference kernels and instead uses the direct Vitis JSONL evidence.
Generated kernels still have to pass the framework's compile/synthesis/csim
checks where those checks are supported.

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
| `--strategy static\|dynamic\|flash\|combo_full\|combo_progressive\|forward_eval` | `static` | Step selection/mode. `flash` means one all-in optimisation step after the functional baseline |
| `--model ID` | `$C2HLS_MODEL` | LLM model id (auto-routed by name) |
| `--turns N` | `3` | Max repair attempts per phase/step |
| `--out PATH` | `results/` or `results_multistep/` | Output directory |

---

## Pipeline Architecture

### Control flow

```
load benchmark inputs
  |
  +-- load metadata.json and reference variants
  +-- validate/trust references
  |     - all/selected/preferred/baseline: local Vitis validation
  |     - trusted_external: Rodinia/Rodinia-Nova direct JSONL evidence
  |
  v
Phase A: compile-check plain.cpp with g++
  |
  v
Phase B: generate a functional HLS baseline
  |
  +-- optional Phase 8 baseline alignment
  |
  v
for each selected optimization step:
  |
  +-- generate candidate code
  +-- preflight compile
  +-- csynth_design
  +-- csim/cosim when supported
  +-- repair with structured error and report feedback
  +-- record candidate/attempt telemetry
  +-- apply regression/resource guard
  |
  v
Phase 6a: promote best passing step
  |
  +-- optional final Nova hw_emu
  |
  v
results JSON + schema-1.0 JSONL + comparison markdown
```

### TranslatorAgent

- **Phase A** — compile-check with `g++ -c`; fix on failure
- **Phase B** — emit HLS code via `q_translate_c_to_hls`, targeting
  `extern "C" workload()`, unified `bundle=control` s_axilite, and
  appropriate PIPELINE/UNROLL/ARRAY_PARTITION pragmas in legacy single-shot
  mode. In multistep mode the default is `C2HLS_PHASEB_MODE=functional`,
  which emits only a correct Vitis kernel baseline and leaves optimization
  pragmas to the step agents.
- **Input cleanliness requirement** — `plain.cpp` and included local headers
  must be compileable without upstream HLS helper dependencies. If a stripped
  input still includes `ap_int.h`, MARS wide-bus helpers, or benchmark-specific
  HLS transport code, Phase A correctly blocks the run and records the failure.

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
  passes, re-prompt with the testbench failure log. This applies both to
  Phase B's functional baseline and to later optimization steps, so the
  workflow does not start optimization from a synth-only but functionally
  broken kernel.
- **GT-shape-aware revert** (`trajectory_alignment.py`) — keeps
  intermediate regressions that match the GT trajectory shape (enabling
  steps that are a structural prerequisite for later gains)
- **Revert-on-streak** — N consecutive same-class errors trigger reversion
  to last-known-good state (`C2HLS_SYNTH_REVERT_THRESHOLD`)
- **Candidate search telemetry** — when exhaustive search is enabled, every
  candidate attempt is synthesized or explicitly failed, and min/max/avg
  latency/resource statistics are retained in the step record.

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

The hw_emu result is intentionally profile-visible. Missing profile CSV,
timeout, variant mismatch, testbench failure, and skipped runs are recorded
under `results["hw_emu"]` and propagated into sweep summaries/JSONL metadata.
This is important because csynth latency can look good while final emulation
times out, crashes, or fails the benchmark testbench.

The emulation wrapper writes `make check` output to a real log file while the
run is active. If the log reaches an explicit terminal failure marker such as
`Benchmark results are incorrect` or `make: ***` and the subprocess tree does
not exit, the wrapper terminates it after a short settle window and appends a
visible `[C2HLS]` note to the log. This prevents known-failed hw_emu runs from
occupying the full timeout while still preserving the failure evidence.

SRAD has an additional local CSim caveat: its tiled kernel uses a halo-row
contract and the upstream vectorized bottom-tile boundary condition. The local
testbench golden mirrors that condition so generated code is checked against
Rodinia/Nova semantics, not a scalar boundary variant. Phase B also records a
visible `srad_halo_copy_offset` preflight patch if an LLM shifts copy-back
offsets from `(t*TILE_ROWS+1)*COLS` to `t*TILE_ROWS*COLS`.

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

A JSON store of confidence-tagged pattern→strategy entries, loaded from
`skills/skills.json` by the orchestrator and queried per bottleneck kind at
each step. Built-in defaults are merged into the store when new skills are
added by framework updates. The framework also imports the curated schema-1.1
package in
`hls_full_optimization_skills_schema_1_1_package/skills.json`, which adds
compound coalescing, tiling, pipeline, unroll, double-buffer, and multibank
recipes plus explicit guardrails.

The important behavior change is that coalescing is treated as a compound HLS
rewrite, not as "add an `m_axi` pragma and hope." The rendered prompt now
includes `required_steps` and `guards`, so the agent is reminded to check
burst-friendly contiguous access, local staging, lane-level compute parallelism,
tail handling, resource growth, and synthesis-report evidence.

### Entry schema

```python
{
  "id":          "hls-coalescing-512-compound-transform",
  "pattern":     "m_axi port in pipelined loop; latency dominated by DRAM bandwidth",
  "strategy":    "request 512-bit widening, reshape access, stage locally, exploit LANES compute",
  "template":    "#pragma HLS INTERFACE m_axi port=X bundle=gmem max_widen_bitwidth=512 ...",
  "confidence":  "high",          # high / medium / low / avoid
  "kind":        "compound_transformation",
  "bottleneck_kinds": ["memory_bandwidth", "axi_burst_failed"],
  "applicable_versions": ["2023.2"],
  "applicable_fpgas": ["xcu280-fsvh2892-2L-e"],
  "tags":        ["coalescing", "max_widen_bitwidth", "lane-parallelism"],
  "required_steps": ["rewrite global accesses to contiguous unit-stride loops", "..."],
  "guards":      ["do not treat interface pragmas alone as complete coalescing", "..."],
  "occurrences": 0,
  "sec_pass": 0,
  "mean_advantage": 0.0
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
The selected router skill is passed directly into the step prompt so the
agent sees the intended recipe rather than an unrelated top-bottleneck match.
The router maps the curated `hls-*` skill ids back to executable steps such as
`coalescing`, `tiling`, `pipeline`, `unroll`, and `doublebuffer`.

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

During agentic sweeps, use `C2HLS_REFERENCE_VALIDATE_MODE=trusted_external`
for Rodinia/Rodinia-Nova. In that mode the framework trusts direct JSONL
records for reference `hls_synth`, `sw_emu`, and `hw_emu` status instead of
requiring each reference kernel to pass the stripped-C validation path. This
matches the benchmark's own direct-run contract and prevents known-good HLS
references from being rejected by the translator-oriented input checks.

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

Validate the envelope and semantic checks before treating a file as an
intended result artifact:

```bash
python export_schema_jsonl.py --validate-jsonl artifacts/schema_records.jsonl
python scripts/validate_jsonl_semantics.py artifacts/schema_records.jsonl
```

Direct Nova/reference records should use `origin=rodinia_hls_benchmark`.
Generated records should use `origin=c2hls_orchestrator`. Every record must
have one payload only: `sw_run`, `rtl_sim`, or `hls_synth`.

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
| [scripts/prepare_hlsfactory_external_benches.py](scripts/prepare_hlsfactory_external_benches.py) | Materializes HLSFactory PolyBench cases into `benchmarks_external/HLSFactory/polybench_float_small/` |
| [scripts/prepare_hls_eval_external_benches.py](scripts/prepare_hls_eval_external_benches.py) | Materializes hls-eval cases into flat c2hls benchmark dirs under `benchmarks_external/hls_eval/` |

### JSONL export

| File | Purpose |
|---|---|
| [export_schema_jsonl.py](export_schema_jsonl.py) | Canonical schema-1.0 JSONL from `results/` + `results_multistep/` |
| [scripts/validate_jsonl_semantics.py](scripts/validate_jsonl_semantics.py) | Additional JSONL payload checks for timing/resource validity and suspicious external-dataset synth records |

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
| `C2HLS_TMP_ROOT` | `/mnt/data/luo00466/tmp` | Scratch root for C2HLS compile probes, HLS synth/csim/cosim dirs, direct emu staging, and inherited `TMPDIR`/`TEMP`/`TMP` |
| `C2HLS_SYNTH_TIMEOUT` | `1200` | Wall-time budget for `csynth_design` (seconds) |
| `C2HLS_CSIM_TIMEOUT` | `180` | Wall-time budget for `csim_design` |
| `C2HLS_COSIM_TIMEOUT` | `1200` | Wall-time budget for `cosim_design` |
| `C2HLS_REFERENCE_VALIDATE_MODE` | `trusted_external` in sweeps, `all` otherwise | `trusted_external` uses direct JSONL reference artifacts for `rodinia-hls` / `rodinia-hls-nova` and records `reference_source=direct_jsonl`; non-trusted datasets use local Vitis validation. `external` requires a trusted direct record and fails explicitly if one is missing |
| `C2HLS_REFERENCE_JSONL_PATHS` | — | Optional `:`-separated extra direct-reference JSONL files to include when resolving trusted external reference status |

### Phase 8 — Baseline alignment

| Variable | Default | Purpose |
|---|---|---|
| `C2HLS_PHASE8_BASELINE_ALIGN` | `0` | Enable Phase 8 retranslation loop |
| `C2HLS_PHASE8_FMAX_FLOOR` | `0.80` | Reject baseline if `agent_fmax < floor × gt_fmax`; catches structurally slow translations that happen to match GT cycle count |
| `C2HLS_PHASE8_MAX_ALIGN_ATTEMPTS` | `4` | Max retranslation attempts before giving up and accepting best seen |

### Phase 9 — Correctness repair

| Variable | Default | Purpose |
|---|---|---|
| `C2HLS_DISABLE_CORRECTNESS_REPAIR` | `0` | Set to `1` to disable generated-code csim/cosim repair in Phase B and Phase 9 |

### Multistep optimisation

| Variable | Default | Purpose |
|---|---|---|
| `C2HLS_PHASEB_MODE` | `functional` for multistep, `optimized` for single-shot | Phase B prompt mode. `functional` emits only legal/testbench-compatible HLS; `optimized` preserves legacy Phase B optimization behavior |
| `C2HLS_STRATEGY` / `C2HLS_SWEEP_STRATEGY` | `dynamic` in sweeps, `static` in CLI unless set | `flash` runs one all-in optimisation step; `combo_full` is the older all-techniques prompt; `dynamic` enables router/skill-library step selection |
| `C2HLS_CANDIDATES_PER_STEP` | `1` | Number of independent LLM candidates per optimization step. Accepts an integer or JSON, e.g. `{"coalescing":3,"default":1}` |
| `C2HLS_ATTEMPTS_PER_CANDIDATE` | current repair turn limit | Number of fully evaluated attempts per candidate when exhaustive candidate attempts are enabled. Bounded to 1–10 |
| `C2HLS_EXHAUSTIVE_CANDIDATE_ATTEMPTS` | `0` | Set to `1` to synthesize every candidate attempt, then select the best passing attempt per candidate and record min/max/avg metrics |
| `C2HLS_SKILL_LIBRARY_PERSIST` | `1` | Persist skill-library bootstrap entries and per-skill trajectory statistics to `skills/skills.json` |
| `C2HLS_PHASEB_FAST_CANDIDATE_RATIO` | `0.80` | Record Phase B as a fast candidate when its baseline cycles are below this fraction of the reference baseline |
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
| `C2HLS_EMU_TERMINAL_SETTLE_S` | `10` | Seconds to wait after a terminal hw_emu/sw_emu failure marker before stopping a stuck process tree |

---

## Performance Hardening Roadmap

Recent Rodinia/Nova and HLSFactory sweeps show the framework is now getting
past the earlier reference-validation blocker, but several performance and
validity issues still need targeted hardening. These are the next practical
patches to improve generated-kernel quality without moving to reinforcement
learning.

### 1. Strengthen stripped-input hygiene

Observed issue: `lc_dilate` can still fail Phase A because `plain.cpp` or a
local header reaches `support/common/mc.h`, which includes `ap_int.h`.

Planned work:

- Add a corpus sanitizer that fails on `ap_int.h`, `ap_uint`, `MARS_*`,
  `memcpy_wide_bus_*`, and upstream transport helpers in LLM inputs.
- Expand `validate_corpus.py` so this is a hard corpus error before a sweep.
- Prefer benchmark-local scalar/plain-C headers for Phase A; keep HLS helper
  headers only inside reference variants.

### 2. Make resource feasibility part of candidate selection

Observed issue: some candidates report excellent latency but unrealistic
resource usage or suspiciously tiny cycles.

Planned work:

- Reject selected candidates that exceed device capacity before Phase 6a best
  promotion, not only after summary export.
- Add a resource-normalized score for tie-breaking: latency improvement must
  pay for DSP/BRAM/LUT/FF growth.
- Route suspicious records through `scripts/validate_jsonl_semantics.py`, for
  example external-dataset records with implausible cycle counts or no
  meaningful loop scopes.
- Record `selection_rejected_reason` for each rejected fast candidate so the
  search remains auditable.

### 3. Improve HLS error classifiers and repair prompts

Observed recurrent errors include undeclared temporaries (`sum`, `accum`,
`min_dist`), unsupported pointer selection, invalid dataflow on shared AXI
bundles, invalid burst values, and Vitis pragma conflicts.

Planned work:

- Add typed repair classes for those Vitis error families.
- Feed the exact class, source line, and minimal fix pattern into the next
  SynthesisAgent prompt.
- Detect repeated same-class failures earlier and switch strategy rather than
  spending all attempts on small rewrites of the same broken idea.

### 4. Use adaptive candidate budgets

The exhaustive `5 candidates x 5 attempts` mode is useful for measurement, but
it is expensive and can waste time on unpromising steps.

Planned work:

- Keep 5x5 for benchmark-quality sweeps, but add an adaptive mode for routine
  development: stop a candidate early after repeated compile/csim failure, and
  escalate only promising candidates.
- Use Haiku for broad candidate generation and reserve Sonnet for selected
  repair/escalation cases such as repeated synthesis failures or near-GT
  bottleneck closure.
- Cache and deduplicate canonical AST-equivalent attempts before Vitis runs.

### 5. Mine reference trajectories into the skill library

The reference variants already encode useful human HLS decisions.

Planned work:

- Diff `baseline -> tiling -> pipeline -> unroll -> doublebuffer ->
  coalescing` variants and extract pragma/code motifs into skill candidates.
- Attach each skill to bottleneck classes and resource envelopes.
- Promote skills only when live sweeps show positive advantage; demote skills
  that repeatedly cause csim failure, resource overflow, or Vitis conflicts.

### 6. Harden final hw_emu accounting

Observed issue: several designs pass csynth/csim but timeout or fail in final
`hw_emu`.

Planned work:

- Preserve and link final `make`, `v++`, XSIM, `profile_kernels.csv`, and
  `xrt.ini` artifacts for each hw_emu attempt.
- Classify timeout, missing profile CSV, XSIM crash, and testbench mismatch as
  distinct statuses in JSONL metadata.
- Default problematic XSIM waveform cases to `debug_mode=off` when the goal is
  profiling rather than waveform capture.
- Keep variant-aware staging strict: if the final selected step cannot be
  mapped to the corresponding Nova variant, skip with a profiled reason rather
  than silently falling back.

### 7. Separate external-dataset evaluation from Nova emulation

HLSFactory and hls-eval do not have Rodinia/Nova host harnesses, so the current
framework correctly skips Nova-style `hw_emu` for those records.

Planned work:

- Treat external datasets as csynth/csim-first benchmarks unless a dataset-local
  host/emulation harness is available.
- Add dataset-specific emulation adapters only where the host contract is
  explicit and reproducible.
- Compare external results through canonical JSONL and semantic validators,
  not through Rodinia-specific hw_emu expectations.

These improvements are all compatible with the current agentic design: they
make the agents' feedback sharper, reduce invalid candidate wins, and make
failures easier to profile without changing the core translator/synthesis/
quality-repair architecture.

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
