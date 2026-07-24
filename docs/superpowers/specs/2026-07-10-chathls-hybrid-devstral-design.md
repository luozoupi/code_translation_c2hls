# ChatHLS hybrid backend + Devstral escalation (PC2)

Date: 2026-07-10  
Status: draft for user review  
Primary code: `/scratch/hpc-prf-llmfpga/asa582/projects/test-chathls/ChatHLS-ACL-26`  
Models: `test-chathls/models/{ChatHLS-HLSFixer,ChatHLS-HLSTuner}`, `devstral2/models/Devstral-2-123B-Instruct-2512`  
Slurm patterns: `c2hls/scripts/pc2/` + `devstral2/serve_fp8_single_node.slurm`  
Spec path note: written under `/pc2/users/h/haqc2/docs/superpowers/specs/` because the scratch project quota is currently exceeded (cannot create files under `c2hls/` or `test-chathls/`).

## Goal

Run ChatHLS’s `benchmark_optimization` suite on PC2 (Alveo **U280**, **Vitis 2023.2**) with a paper-faithful hybrid LLM setup:

- Keep **HLSFixer** and **HLSTuner** as local Hugging Face fine-tunes.
- Replace commercial API roles (**transform** + **debug escalation / `debug_multi`**) with **Devstral-2-123B** via an OpenAI-compatible vLLM endpoint (or a real commercial key later).
- First deliverable: smoke run on **`benchmark_optimization/gemm`**.

## Non-goals (this pass)

- Full `benchmark_optimization` batch beyond gemm smoke.
- `benchmark_gen` / Rodinia / HLSFactory integration.
- Replacing HLSFixer or HLSTuner with Devstral.
- Permanently rewriting upstream benchmark TCL trees (prefer run-local U280 patch).

## Background

Upstream ChatHLS backends today are mutually exclusive: `api`, `hf`, or `agent`. The paper’s HLSFixer flow uses a fine-tuned analysis model first, then escalates to general LLMs (`debug_multi` / LLM-as-judge) when the first repair fails. Transform defaults to a general model (`deepseek-v3.2`). Without a commercial API key, those general-LLM calls must target local Devstral.

Existing assets on disk:

- ChatHLS checkout + HF weights under `test-chathls/`.
- Devstral FP8 weights + vLLM venv under `devstral2/`.
- c2hls PC2 GPU serve + Vitis 2023.2 / U280 compute patterns under `c2hls/scripts/pc2/`.

## Design

### 1. Hybrid LLM routing

Add `--llm-backend hybrid` (also `CHATHLS_LLM_BACKEND=hybrid`).

| Call site | Transport | Model |
|-----------|-----------|--------|
| Transform | HTTP OpenAI-compatible | `CHATHLS_GENERAL_MODEL` (Devstral) |
| Debug 1st (`debug` analysis + modify) | Local HF | HLSFixer |
| Debug escalate (`debug_multi` analysis ×3 + score) | HTTP | Devstral |
| Debug escalate modify | Local HF | HLSFixer |
| Optimize analysis + modify | Local HF | HLSTuner |

Preserve existing `api` / `hf` / `agent` behavior for upstream compatibility.

Implementation sketch:

- Extend `ChatHLSConfig` / CLI choices with `hybrid`.
- In `LLMAdapter._invoke_llm`, route by **role** (or by model path heuristic): HF paths for Fixer/Tuner directories; HTTP for the general model name.
- `debug_multi` must call HTTP for analysis/score and HF for the final modify step (today all share one backend).

### 2. Config / environment (API key optional)

| Env | Purpose | PC2 Devstral default |
|-----|---------|----------------------|
| `CHATHLS_API_BASE` or `OPENAI_BASE_URL` | Base URL ending in `/v1` | From GPU endpoint file |
| `CHATHLS_API_KEY` or `OPENAI_API_KEY` | Bearer token | `EMPTY` |
| `CHATHLS_GENERAL_MODEL` | Transform + escalation model id | `mistralai/Devstral-2-123B-Instruct-2512` |
| `CHATHLS_LLM_BACKEND` | Backend | `hybrid` |
| `CHATHLS_FPGA_PART` | Skills / metadata | `xcu280-fsvh2892-2L-e` |
| `CHATHLS_VITIS_VERSION` | Tooling / skills | `2023.2` |

Rules:

- Hybrid requires a reachable base URL; key may be empty or `EMPTY` (vLLM).
- Same HTTP path works with a commercial key + hosted base later.
- Loosen `api` key requirement the same way for consistency (base URL still required).

### 3. PC2 Slurm (c2hls-style split jobs)

```
[GPU job]  4×H100 vLLM serves Devstral  →  writes endpoint URL
                ↓
[Compute job]  waits for URL → loads HLSFixer+HLSTuner on GPU
               → Vitis 2023.2 hybrid workflow on gemm (U280)
```

Deliverables (under ChatHLS repo scripts when quota allows; otherwise stage under home/`c2hls` and copy):

1. Submit helper that starts GPU serve + compute with dependency / endpoint handoff (mirror c2hls `submit_gpu` + `compute_worker` ideas; reuse `devstral2` serve recipe: TP=4, served name above).
2. Compute sbatch: load Vitis 2023.2 / U280 modules (align with `c2hls/scripts/pc2/setup_vitis_env.sh`), activate ChatHLS venv, set hybrid env, run smoke.
3. Endpoint discovery: compute reads `OPENAI_BASE_URL` from a session/endpoint JSON written by the GPU job (same pattern as c2hls `llm_endpoint.json`).

GPU sizing: Devstral FP8 smoke = **1 node × 4 H100, TP=4** per `devstral2/ALLOCATION.txt`. Compute job needs at least **1 H100** for HF Fixer/Tuner (~28G each; load one at a time or both if memory allows).

### 4. U280 + gemm smoke

- Upstream `benchmark/benchmark_optimization/gemm/run_hls.tcl` uses `xczu7ev-…`. For smoke, **run-local** copy/patch `set_part {xcu280-fsvh2892-2L-e}` (do not permanently edit the suite in this pass unless batching later).
- Set `CHATHLS_FPGA_PART=xcu280-fsvh2892-2L-e`.
- Invoke:

```bash
./run_chathls.sh \
  --repo-root . \
  --llm-backend hybrid \
  --project-dir <u280-patched-gemm> \
  --kernel-name gemm \
  --top-function gemm \
  --source-file gemm.cpp \
  --run-name opt-gemm-hybrid-u280
```

Note: `benchmark_optimization` is project mode; transform usually copies the project and may not call Devstral unless code generation is triggered. Escalation still needs Devstral when the first HLSFixer repair fails.

## Success criteria

1. Hybrid backend unit-level: HF path used for Fixer/Tuner; HTTP path used for general model without requiring a commercial key.
2. GPU job exposes a healthy `/v1/models` endpoint for Devstral.
3. Compute job completes a gemm workflow attempt on U280 / Vitis 2023.2 and writes `runs/opt-gemm-hybrid-u280-*/summary.json`.
4. If first debug fails, logs show HTTP escalation (`debug_multi`) hitting Devstral, not DeepSeek/OpenAI hosted APIs.

## Risks / open points

- **Quota**: scratch project quota is exceeded; cannot write under `c2hls/` or `test-chathls/` until space is freed. Spec lives in home; implementation may need cleanup first.
- **Two GPU allocations**: Devstral (4×H100) + HF compute (1×H100) may queue; optional later: borrow an already-serving Devstral endpoint (c2hls `borrow_gpu` pattern).
- **HF + Vitis on one node**: existing ChatHLS rodinia batch already co-locates HF + Vitis on `gpu_h100`; keep that pattern for compute.
- **Part mismatch**: any unpatched TCL will synthesize for the wrong device; smoke must assert U280 in the effective `run_hls.tcl`.

## Implementation plan (next, after spec approval)

Tracked separately via writing-plans after user approves this spec.
