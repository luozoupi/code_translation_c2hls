# Offline SFT → DPO for C→HLS (Devstral) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (or subagent-driven-development) to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fine-tune an open-weight model (primary: Devstral-2-123B via QLoRA) on the team offline RL corpus so that C→HLS generation improves **pass quality** first and **latency** second, without running Vitis during training.

**Architecture:** Offline only. (1) SFT on cosim-verified / validated-positive chat trajectories. (2) DPO on preference pairs ranked by joint pass tier + hybrid latency. (3) Serve LoRA + evaluate with real HLS (csim/cosim). GLM-4.7-FP8 stays inference/eval-only; do not use the MoE FP8 serve checkpoint as the TRL train target.

**Tech Stack:** Python 3, TRL, PEFT/QLoRA, bitsandbytes, Hugging Face `datasets`, Slurm `gpu_h100` (4× H100), Devstral weights under `devstral2/`, training kit under `c2hls/rl/`.

**Runbook:** `c2hls/rl/README.md`

---

## Objectives and success metrics

Primary (in order):

1. **Pass quality** on held-out benches (joint tiers below).
2. **Latency** among passes, using hybrid latency.

Do **not** treat train loss as success. Report vs base Devstral (and optionally GLM-4.7 serve) on the same prompts/benches.

### Pass tiers (joint; csim ⊄ cosim)

Cosim = RTL synth + RTL sim. It implies a synthesizable RTL path, **not** csim.

| Tier | Meaning |
|------|---------|
| 4 | `csim_passed` **and** `cosim_passed` (best) |
| 3 | `cosim_passed` only |
| 2 | `csim_passed` only |
| 1 | `synth_passed` only |
| 0 | fail / unknown |

### Hybrid latency

```text
effective_latency =
    cosim_cycles       if cosim_passed and cosim_cycles is not None
    else latency_cycles   # csynth estimate
```

Lower is better. Use the same rule for DPO ranking and for eval tables.

### Eval summary (what to publish)

Per model / adapter, on a fixed held-out set:

- Rate at tier ≥ 2, ≥ 3, = 4
- Median / best `effective_latency` among tier ≥ 3 (and separately among tier = 4)
- Optional: csynth-only latency table for benches without cosim

---

## Data

| Path | Role | Approx size |
|------|------|-------------|
| `rl/rl_dataset.zip` → `rl/extracted/rl_dataset/` | Clean cosim-verified SFT + original DPO | 401 SFT / 302 DPO |
| `rl/rl_corpus.zip` → `rl/extracted/rl_corpus/` | Large agentic TRL export + metric points | ~3.7k SFT train; cosim subsets |
| `rl/prepared/sft/` | TRL chat SFT JSONL | 401 / 41 |
| `rl/prepared/dpo_hybrid/` | DPO pairs under revised tier + hybrid latency | 878 / 98 |
| `rl/prepared/dpo/` | Fallback from `rl_dataset` only (cosim cycles) | 276 / 26 |

Default DPO training input: **`prepared/dpo_hybrid`**.

| `rl/prepared/mined_sft/` | Fresh mine from local `artifacts/pc2` + `results_matrix_*` + `results/` | 184 validated_positive (558 raw) |
| `rl/prepared/sft_combined/` | Merge: mined + `rl_dataset` SFT + agentic_trl | ~4394 / 146 / 2 train/val/test |

Mine / merge:

```bash
python rl/scripts/mine_runs_to_sft.py
python rl/scripts/merge_sft_corpora.py
```

---

## Model roles

| Model | Role |
|-------|------|
| Devstral-2-123B (`devstral2/models/Devstral-2-123B-Instruct-2512`) | **Train** with QLoRA SFT → DPO on 4× H100 |
| GLM-4.7-FP8 (`glm4.7` serve stack) | **Serve / eval / teacher** only — not this TRL recipe |
| Dense small GLM (9B/32B) if available | Optional alternate train target via `rl/slurm/train_glm_*.slurm` |

---

## File map

| Path | Role |
|------|------|
| `rl/README.md` | Operator runbook |
| `rl/scripts/prepare_trl_datasets.py` | Normalize `rl_dataset` → `prepared/sft` + `prepared/dpo` |
| `rl/scripts/build_hybrid_latency_dpo.py` | Build `prepared/dpo_hybrid` with joint tiers + hybrid latency |
| `rl/scripts/train_sft.py` | TRL QLoRA SFT |
| `rl/scripts/train_dpo.py` | TRL DPO (optional merge of SFT adapter first) |
| `rl/scripts/setup_trl_venv.sh` | Dedicated train venv (not vLLM serve env) |
| `rl/requirements-trl.txt` | Train deps |
| `rl/slurm/train_devstral_sft.slurm` | 4× H100 SFT job |
| `rl/slurm/train_devstral_dpo.slurm` | 4× H100 DPO job (defaults to `dpo_hybrid`) |
| `rl/slurm/train_glm_{sft,dpo}.slurm` | Experimental dense-GLM path |
| `rl/outputs/` | Adapters + logs (created at job time) |
| `rl/logs/` | Slurm stdout/stderr |

---

### Task 0: Confirm data and policy artifacts

**Files:** `rl/extracted/`, `rl/prepared/`, `rl/scripts/build_hybrid_latency_dpo.py`

- [x] Extract `rl_corpus.zip` / `rl_dataset.zip` under `rl/extracted/`
- [x] Run `prepare_trl_datasets.py` → `prepared/sft`, `prepared/dpo`
- [x] Run `build_hybrid_latency_dpo.py` with revised tiers → `prepared/dpo_hybrid`
- [x] Document policy in `rl/README.md`

**Verify:**

```bash
python3 -c "import json;from pathlib import Path
m=json.loads(Path('rl/prepared/dpo_hybrid/manifest.json').read_text())
assert m['policy']['tiers']['4'].startswith('csim')
print(m['pairs_train'], m['pairs_val'], m['latency_source_counts'])"
```

---

### Task 1: Training environment

**Files:** `rl/scripts/setup_trl_venv.sh`, `rl/requirements-trl.txt`

- [ ] **Step 1: Create venv on a GPU node**

```bash
cd /scratch/hpc-prf-llmfpga/asa582/projects/c2hls/rl
bash scripts/setup_trl_venv.sh
source .venv-trl/bin/activate
python -c "import torch, trl, peft; print(torch.__version__, trl.__version__, torch.cuda.is_available())"
```

Expected: CUDA available; `trl` / `peft` import OK.

---

### Task 2: Devstral QLoRA SFT (pass-rate foundation)

**Files:** `rl/slurm/train_devstral_sft.slurm`, `rl/scripts/train_sft.py`, `rl/prepared/sft/`

- [ ] **Step 1: Submit starter SFT on cosim-verified set**

```bash
cd /scratch/hpc-prf-llmfpga/asa582/projects/c2hls/rl
sbatch slurm/train_devstral_sft.slurm
```

- [ ] **Step 2: Confirm adapter written**

```bash
# after job completes
ls -la outputs/devstral_sft_<JOBID>/adapter
```

- [ ] **Step 3 (optional scale): SFT on agentic set**

```bash
python scripts/prepare_trl_datasets.py --include-agentic
# point train-file at prepared/agentic_sft/train.jsonl (edit env/script or second submit)
```

**Done when:** Adapter dir exists; train log shows decreasing loss; no OOM after first steps.

**GPU-hour ballpark (starter):** ~4–12 GPU-hours (4× H100 × ~1–3 h wall).

---

### Task 3: Devstral DPO (tier + hybrid latency)

**Files:** `rl/slurm/train_devstral_dpo.slurm`, `rl/scripts/train_dpo.py`, `rl/prepared/dpo_hybrid/`

- [ ] **Step 1: Rebuild hybrid pairs if policy/code changed**

```bash
python scripts/build_hybrid_latency_dpo.py
```

- [ ] **Step 2: Submit DPO from SFT adapter**

```bash
SFT_ADAPTER=/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/rl/outputs/devstral_sft_<JOBID>/adapter \
  sbatch slurm/train_devstral_dpo.slurm
```

**Done when:** `outputs/devstral_dpo_<JOBID>/adapter` exists.

**GPU-hour ballpark (hybrid DPO):** ~4–12 GPU-hours starter; more if data scaled.

---

### Task 4: Serve adapter and wire into C2HLS

**Files:** `devstral2/` serve scripts; C2HLS `OPENAI_BASE_URL` / session endpoint

- [ ] **Step 1: Serve base Devstral + LoRA** (vLLM LoRA) **or** merge adapter offline and serve merged weights
- [ ] **Step 2: Point a small C2HLS smoke** at the endpoint (1–2 benches)
- [ ] **Step 3: Run held-out eval campaign** (same benches/protocol as baseline)

**Done when:** Endpoint healthy; smoke produces HLS candidates end-to-end.

---

### Task 5: Held-out evaluation report

**Files:** new under `docs/pc2/` (suggested): `docs/pc2/2026-07-19-offline-sft-dpo-devstral-eval.md`

- [ ] **Step 1: Score base Devstral** on held-out set (tier rates + hybrid latency)
- [ ] **Step 2: Score SFT-only adapter**
- [ ] **Step 3: Score SFT→DPO adapter**
- [ ] **Step 4: Write compare table** (pass tiers + latency; cosim when available else csynth)

**Success criteria:**

- Tier ≥ 3 and/or tier = 4 rate ≥ base Devstral
- Among tier ≥ 3, median hybrid latency ≤ base (prefer strict improvement)
- No large regression on tier ≥ 2 rate

---

### Task 6 (optional): Scale / iterate if offline plateaus

- [ ] More DPO pairs from new cosim runs (same tier + hybrid latency policy)
- [ ] Second SFT pass on `agentic_trl_chat_v1` then DPO again
- [ ] Online loop only if needed: generate → HLS → keep winners → refresh SFT/DPO

**GPU-hour ballpark if scaling agentic SFT seriously:** ~40–120+ GPU-hours for SFT alone.

---

## Out of scope (this plan)

- Full-parameter fine-tune of Devstral 123B or GLM-4.7 MoE FP8
- Training GLM-4.7-FP8 serve weights with the current TRL/bitsandbytes recipe
- Replacing C2HLS agent loop with a pure single-shot model (eval may still be single-shot or agentic; training is offline)

---

## Quick command cheat sheet

```bash
cd /scratch/hpc-prf-llmfpga/asa582/projects/c2hls/rl

bash scripts/setup_trl_venv.sh
source .venv-trl/bin/activate

python scripts/prepare_trl_datasets.py
python scripts/build_hybrid_latency_dpo.py

sbatch slurm/train_devstral_sft.slurm
SFT_ADAPTER=$PWD/outputs/devstral_sft_<JOBID>/adapter \
  sbatch slurm/train_devstral_dpo.slurm
```
