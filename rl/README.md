# Offline SFT → DPO for C→HLS (Devstral / GLM)

Training kit for the team packs in `rl_corpus.zip` / `rl_dataset.zip`.

**Plan:** [`docs/superpowers/plans/2026-07-19-offline-sft-dpo-devstral.md`](../docs/superpowers/plans/2026-07-19-offline-sft-dpo-devstral.md)

## Mine your own runs into SFT

Durable sources only (`artifacts/pc2`, `results_matrix_*`, `results/`).  
Skips `c2hls_tmp/batch_parallel_*` (HLS workdirs without chat/pass JSON).

```bash
cd /scratch/hpc-prf-llmfpga/asa582/projects/c2hls/rl
python scripts/mine_runs_to_sft.py
# → prepared/mined_sft/  (184 validated_positive from local runs)

python scripts/merge_sft_corpora.py
# → prepared/sft_combined/  (~4.5k: mined + rl_dataset + agentic_trl)
```

## Layout

```text
rl/
  rl_*.zip                 # originals from team
  extracted/
    rl_dataset/            # cosim-verified SFT + DPO (start here)
    rl_corpus/             # large agentic TRL export + metric_points
  prepared/                # TRL-ready JSONL (after prepare script)
  scripts/
    prepare_trl_datasets.py
    train_sft.py           # QLoRA SFT
    train_dpo.py           # offline DPO
    setup_trl_venv.sh
  slurm/
    train_devstral_sft.slurm
    train_devstral_dpo.slurm
    train_glm_sft.slurm    # dense GLM only (not 4.7-FP8 MoE serve)
    train_glm_dpo.slurm
```

## What to train on

| Dataset | Use when |
|---------|----------|
| `extracted/rl_dataset` | **Default.** 401 SFT + 302 DPO, all cosim-verified on U280 |
| `extracted/rl_corpus/agentic_trl_chat_v1` | Scale SFT (~3.7k); noisier / longer contexts |
| `cosim_passed.jsonl` | Small reward/DPO prototyping subset |

## Latency / pass-rate ranking policy

For preferences (DPO) and eval:

1. **Pass first** with a *joint* tier (csim ⊄ cosim):
   - cosim = RTL synth + RTL sim; it does **not** imply csim
   - best = csim **and** cosim, then cosim-only, then csim-only, then synth-only
2. **Latency next**, with hybrid metric:

```text
effective_latency =
    cosim_cycles     if cosim_passed
    else latency_cycles   # csynth
```

Build pairs:

```bash
python scripts/build_hybrid_latency_dpo.py
# → prepared/dpo_hybrid/{train,val}.jsonl
```

`train_devstral_dpo.slurm` uses `dpo_hybrid` by default.

## Recommended model policy

| Model | Role |
|-------|------|
| **Devstral-2-123B** (`devstral2/models/...`) | Primary fine-tune target via **QLoRA** on 4×H100 |
| **GLM-4.7-FP8** (`glm4.7` serve stack) | Keep for **inference / teacher / eval**, not for this TRL recipe |
| Smaller dense GLM (9B/32B) if you have weights | Optional alternate fine-tune via `train_glm_*.slurm` |

Full fine-tune of 123B / 358B MoE is out of scope here. Offline RL = **SFT then DPO on fixed labels**.

## One-time setup

```bash
cd /scratch/hpc-prf-llmfpga/asa582/projects/c2hls/rl

# On a GPU node (or interactive allocation) with CUDA modules:
bash scripts/setup_trl_venv.sh
source .venv-trl/bin/activate

python scripts/prepare_trl_datasets.py
# optional larger SFT:
# python scripts/prepare_trl_datasets.py --include-agentic
```

## Train: Devstral path (recommended)

```bash
# 1) SFT
sbatch slurm/train_devstral_sft.slurm

# 2) After SFT finishes, DPO from that adapter
SFT_ADAPTER=/scratch/.../c2hls/rl/outputs/devstral_sft_<JOBID>/adapter \
  sbatch slurm/train_devstral_dpo.slurm
```

Optional knobs (env before `sbatch`):

```bash
MAX_SEQ_LENGTH=8192 NUM_EPOCHS=2 LR=1e-4 GRAD_ACCUM=16 LORA_R=16
```

## Train: GLM path (only if you have a dense trainable GLM)

```bash
MODEL_PATH=/path/to/GLM-4-9B-Chat sbatch slurm/train_glm_sft.slurm
MODEL_PATH=... SFT_ADAPTER=.../adapter sbatch slurm/train_glm_dpo.slurm
```

Do **not** point these at `models/GLM-4.7-FP8` serve weights unless you have a known-working MoE+PEFT train stack (this kit does not provide one).

## After training

1. Serve base model + LoRA (vLLM LoRA, or merge adapter offline).
2. Point C2HLS / ChatHLS at that endpoint.
3. Score held-out benches with csim/cosim — that is the real offline-RL metric, not train loss.

## TRL data formats (after prepare)

**SFT** `prepared/sft/*.jsonl`:

```json
{"messages": [{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

**DPO** `prepared/dpo/*.jsonl`:

```json
{
  "prompt": [{"role":"system","content":"..."},{"role":"user","content":"..."}],
  "chosen": [{"role":"assistant","content":"..."}],
  "rejected": [{"role":"assistant","content":"..."}]
}
```
