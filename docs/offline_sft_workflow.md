# Offline SFT workflow for C2HLS agents

This document records the supervised fine-tuning pipeline used by the local
C2HLS experiments through 2026-07-31. The completed training runs use QLoRA on
Qwen3.6-27B. A model-tokenized Gemma 4 31B corpus also exists, but no completed
Gemma adapter was found, so Gemma is listed as prepared rather than trained.

Offline SFT here means completion-only next-token training on previously saved
agent trajectories. It does not call an LLM API or Vitis during optimization,
and it is not online RL, DPO, GRPO, or reward-model training. Vitis is used to
qualify teacher examples before training and to evaluate generated kernels
after training.

## End-to-end data flow

```text
Sonnet 4.6 C2HLS traces
  -> exact model-call/candidate-event attribution
  -> CSim/CSynth/timing/resource quality filter
  -> benchmark-disjoint train/validation/test split
  -> target-model chat-template rendering and tokenization
  -> completion-only NF4 QLoRA
  -> local OpenAI-compatible PEFT endpoint
  -> direct one-shot and role-routed agentic evaluation
  -> Vitis 2023.2 CSim and CSynth
```

The canonical high-integrity source campaign is the Sonnet 4.6, skill-v2,
flash-mode sweep at:

```text
/home/luo00466/code_translation-c2hls-hpca2027/results_sweeps/
agentic_no_streamcluster_hpca_skillv2_hlsfactory_sonnet46_flash_skills5_csim_csynth_20260724
```

The data and training utilities currently live in the sibling working tree
`/home/luo00466/code_translation-c2hls`. Large corpora, adapters, and trainer
state live under `/mnt/data2/luo00466/c2hls_sft`.

## Canonical role-attributed corpus

The corpus builder joins four pieces of evidence for each supervised target:

1. The exact messages sent for a model call.
2. The exact assistant response returned for that call.
3. The recorded agent role and candidate-evaluation index.
4. The same-index Vitis candidate-validation event.

An example is retained only when it contains code and the associated candidate
has all of the following:

- `correctness_status == passed`, including CSim;
- `synthesis_status == passed`, including CSynth;
- `timing_met == true`;
- `resource_fit == true`; and
- no recorded failure class.

COSIM was not required for this corpus. Local home and temporary paths are
sanitized. The source campaign was run under the reference-blind HPCA profile;
the sanitizer itself only removes paths and must not be treated as a general
reference-leakage detector.

This is positive-only behavioral cloning. Failed candidates are recorded in
rejection counts but are not negative training examples, and cycle counts do
not weight the cross-entropy loss. All feasible attributed responses can be
retained rather than only the minimum-cycle response. Consequently, the
adapter learns patterns associated with valid agent actions but is not trained
to minimize latency directly.

The active roles represented by model calls in that profile are `translator`,
`synthesis`, and `orchestrator`. `feedback` was deterministic and its LLM path
was disabled; `quality_repair` was dormant without a reference report. Those
two roles therefore have no training examples.

The accepted corpus contains 204 examples from 135 result files:

| Split | Benchmarks | Rows | Role composition |
|---|---|---:|---|
| Train | 22 remaining HLSFactory kernels | 167 | translator 72, synthesis 18, orchestrator 77 |
| Validation | `gramschmidt` | 5 | translator 5 |
| Test | `durbin`, `floyd_warshall`, `gemm`, `trmm` | 32 | translator 11, synthesis 1, orchestrator 20 |

The split is by benchmark, not by response, code hash, or random row. The
builder fails if a benchmark appears in more than one split. At the 3,072-token
training cap, two train completions cannot fit while preserving the required
prompt context; the actual Qwen run therefore uses 165 train examples,
including 75 orchestrator examples.

Build the role-attributed JSONL files with:

```bash
SOURCE_REPO=/home/luo00466/code_translation-c2hls
HPCA_REPO=/home/luo00466/code_translation-c2hls-hpca2027
PY=/home/luo00466/.conda/envs/py310_2/bin/python
RESULTS_ROOT="$HPCA_REPO/results_sweeps/agentic_no_streamcluster_hpca_skillv2_hlsfactory_sonnet46_flash_skills5_csim_csynth_20260724"
ROLE_DATA="$SOURCE_REPO/artifacts/rl_corpus/active_llm_actors_sonnet46_skillv2_flash_20260724"

cd "$SOURCE_REPO"
"$PY" scripts/build_role_attributed_sft_dataset.py \
  --results-root "$RESULTS_ROOT" \
  --output-dir "$ROLE_DATA"
```

Omitting `--role` selects all three active roles. Repeating `--role` can create
a role-specific corpus, for example `--role translator`.

The authoritative output manifest is:

```text
/home/luo00466/code_translation-c2hls/artifacts/rl_corpus/
active_llm_actors_sonnet46_skillv2_flash_20260724/manifest.json
```

## Model-aware export

Raw conversations are converted into one canonical system/user/assistant
exchange and rendered with the target checkpoint's own chat template. Qwen
and Gemma are exported separately because their templates and token boundaries
differ. Thinking is disabled for both profiles.

The export writes compressed Parquet containing precomputed `input_ids` and a
same-length `completion_mask`. Prompt and framework-context tokens have mask
zero. The complete assistant target and closing turn tokens have mask one.
Training must use `completion_only_loss=True`, `assistant_only_loss=False`, and
`packing=False`.

Long prompts retain their beginning and most recent context around an explicit
truncation marker. Assistant completions are never truncated. A record is
rejected when the full completion plus 256 prompt tokens cannot fit.

Export Qwen data:

```bash
QWEN_CORPUS=/mnt/data2/luo00466/c2hls_sft/corpus/qwen3.6-27b_active_llm_actors_max8192_20260724

cd "$SOURCE_REPO"
"$PY" scripts/export_model_aware_sft_dataset.py \
  --input-dir "$ROLE_DATA" \
  --profile qwen3.6-27b \
  --output-dir "$QWEN_CORPUS" \
  --max-length 8192 \
  --min-prompt-tokens 256
```

The equivalent Gemma export already present on this host was produced with:

```bash
GEMMA_CHECKPOINT=/mnt/data2/li004074/gemma-4-31B-it
GEMMA_CORPUS=/mnt/data2/luo00466/c2hls_sft/corpus/gemma-4-31b-it_active_llm_actors_max8192_20260725

cd "$SOURCE_REPO"
"$PY" scripts/export_model_aware_sft_dataset.py \
  --input-dir "$ROLE_DATA" \
  --profile gemma-4-31b-it \
  --checkpoint "$GEMMA_CHECKPOINT" \
  --output-dir "$GEMMA_CORPUS" \
  --max-length 8192 \
  --min-prompt-tokens 256
```

Always inspect `manifest.json` and `rejected.jsonl` before training. The
manifest records checkpoint identity, tokenizer, chat-template hash, split
counts, length distributions, skipped rows, and the TRL loss contract.

## Completed Qwen QLoRA configuration

The canonical completed adapter is:

```text
/mnt/data2/luo00466/c2hls_sft/runs/
qwen3.6-27b_active_llm_actors_qlora_r16_1epoch_3072_20260724/adapter
```

It was trained from this immutable local base snapshot:

```text
/mnt/data/vllm_models/hub/models--Qwen--Qwen3.6-27B/
snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9
```

| Setting | Value |
|---|---|
| Method | QLoRA, completion-only SFT |
| Base quantization | 4-bit NF4, double quantization |
| Compute dtype | BF16 |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 |
| Trainable parameters | 116,727,808 of 27,473,456,368, or 0.4249% |
| Target modules | 496 language-model projection modules |
| Sequence length | 3,072 tokens |
| Batch / accumulation | 1 / 1 |
| Optimizer | paged AdamW 8-bit |
| Learning rate | 1e-4, cosine decay, 5% warmup |
| Gradient checkpointing | enabled, non-reentrant |
| Epochs / optimizer steps | 1 / 165 |
| Seed | 42 |
| Train / validation rows | 165 / 5 |
| Training loss / validation loss | 0.14331 / 0.06521 |
| Validation token accuracy | 0.98058 |

The LoRA suffix targets are `q_proj`, `k_proj`, `v_proj`, `o_proj`,
`gate_proj`, `up_proj`, `down_proj`, `in_proj_qkv`, `in_proj_z`, `in_proj_a`,
`in_proj_b`, and `out_proj`. The launcher uses the conditional-generation
wrapper, prepares the 4-bit model for training, and verifies that every
trainable parameter is inside the language model rather than the vision tower.

The historical run used physical GPU 7, an RTX PRO 6000 Blackwell with about
95 GiB, and required at least 80 GiB free before loading. It took 1,813.9
seconds. The adapter is 233,607,464 bytes and has SHA-256:

```text
38ec522e68ac3997f15457abfc055f514d814212e417bc4009b6c7412b9d96ad
```

Use a fresh output directory when reproducing it:

```bash
RUN_DIR=/mnt/data2/luo00466/c2hls_sft/runs/qwen3.6-27b_active_llm_actors_qlora_r16_repro

cd "$SOURCE_REPO"
"$PY" scripts/train_agentic_sft_qlora.py \
  --gpu 7 \
  --dataset-dir "$QWEN_CORPUS" \
  --output-dir "$RUN_DIR" \
  --sequence-length 3072 \
  --min-prompt-tokens 256 \
  --max-steps 165 \
  --max-train-samples 0 \
  --max-eval-samples 0 \
  --rank 16 \
  --alpha 32 \
  --dropout 0.05 \
  --learning-rate 1e-4 \
  --lr-scheduler-type cosine \
  --warmup-ratio 0.05 \
  --gradient-accumulation-steps 1 \
  --seed 42 \
  --min-free-gib 80
```

`max-train-samples=0` and `max-eval-samples=0` mean all eligible rows. The
trainer sets `CUDA_VISIBLE_DEVICES` from `--gpu`, requires exactly one visible
GPU, and forces Hugging Face offline mode. At 3,072 tokens it may trim prompt
context again while retaining the complete assistant target.

The run directory contains `run_manifest.json`, `run.log`,
`step_metrics.jsonl`, `trainer/trainer_state.json`, and the `adapter/`
directory. Treat `run_manifest.json` plus the adapter hash as the training
identity. A low validation loss is only a language-model metric and is not
evidence of lower HLS latency.

The run manifest records Python 3.10.18, PyTorch 2.9.0+cu128, Transformers
5.4.0, TRL 0.27.2, PEFT 0.18.1, and bitsandbytes 0.49.1. The current training
environment reports datasets 4.4.1; future runs should add that package version
to their own manifest rather than infer it from this host.

## Serving the adapter

The completed experiments did not serve the adapter with vLLM. They used
`serve_agentic_peft_openai.py`, a single-request Transformers/PEFT server that
loads the base in 4-bit NF4, attaches the adapter, and exposes OpenAI-compatible
`/v1/models` and `/v1/chat/completions` endpoints.

```bash
MODEL_ID=qwen3.6-27b-c2hls-active-actors-sft-r16-3072

cd "$SOURCE_REPO"
"$PY" scripts/serve_agentic_peft_openai.py \
  --adapter "$RUN_DIR/adapter" \
  --model-id "$MODEL_ID" \
  --gpu 7 \
  --host 127.0.0.1 \
  --port 30105 \
  --max-input-tokens 32768 \
  --max-new-tokens 4096 \
  --min-free-gib 80 \
  --manifest "$RUN_DIR/server_30105_manifest.json" \
  --log-file "$RUN_DIR/server_30105.log"
```

Verify the served identity before an experiment:

```bash
curl --fail http://127.0.0.1:30105/healthz
curl --fail http://127.0.0.1:30105/v1/models | jq -r '.data[0].id'
```

The server truncates overlong input by preserving the system prefix and latest
context and records truncation counts in its manifest. Evaluation should fail
or be flagged when delivery differs between a base/SFT A/B pair.

## Assigning the adapter to agents

The orchestrator itself uses the default model and `OPENAI_BASE_URL`. The four
phase-specific model slots are controlled independently:

| Runtime role | Model variable | Endpoint variable |
|---|---|---|
| Translator | `C2HLS_TRANSLATOR_MODEL` | `C2HLS_TRANSLATOR_BASE_URL` |
| Synthesis | `C2HLS_SYNTHESIS_MODEL` | `C2HLS_SYNTHESIS_BASE_URL` |
| Quality repair | `C2HLS_QUALITY_REPAIR_MODEL` | `C2HLS_QUALITY_REPAIR_BASE_URL` |
| Feedback | `C2HLS_FEEDBACK_MODEL` | `C2HLS_FEEDBACK_BASE_URL` |

Assign the active-role adapter to every runtime slot with:

```bash
SFT_URL=http://127.0.0.1:30105/v1
export OPENAI_API_KEY=EMPTY
export OPENAI_BASE_URL="$SFT_URL"
export C2HLS_SWEEP_MODELS="$MODEL_ID"
export C2HLS_MODEL_REVISION=38ec522e68ac3997f15457abfc055f514d814212e417bc4009b6c7412b9d96ad
export C2HLS_TRANSLATOR_MODEL="$MODEL_ID"
export C2HLS_SYNTHESIS_MODEL="$MODEL_ID"
export C2HLS_QUALITY_REPAIR_MODEL="$MODEL_ID"
export C2HLS_FEEDBACK_MODEL="$MODEL_ID"
```

When every role uses the default endpoint, the per-role base URL variables can
remain unset. For a selective-agent A/B, leave `OPENAI_BASE_URL` on the base
model endpoint and set only the selected role's model and base URL. For an
orchestrator-only A/B, point the default endpoint at the adapter and explicitly
route every phase-specific role back to the base endpoint.

The experiment named `qwen_sft_all_agents_valtest5_nocosim_20260724` assigned
an orchestrator-only adapter to every slot. It is an all-slot deployment, not
an adapter trained on all roles. The later
`qwen_active_actor_sft_valtest5_nocosim_20260724` run uses the mixed
translator/synthesis/orchestrator corpus and is the correct all-active-role
training experiment. It still sets `C2HLS_FEEDBACK_LLM=0`, and quality repair
remains dormant in the reference-blind profile.

## Held-out evaluation

The canonical evaluation uses only the validation and test kernels:
`gramschmidt`, `durbin`, `floyd_warshall`, `gemm`, and `trmm`. It compares:

- direct one-shot generation;
- flash agentic generation with skillless and all-positive skill-v2 modes; and
- multistep agentic generation with skillless and all-positive skill-v2 modes.

Generation uses temperature 0, top-p 1, seed 42, at most 4,096 completion
tokens, one candidate per step, and a pinned adapter hash. Validation uses
Vitis 2023.2, `xcu280-fsvh2892-2L-e`, 3.33 ns, CSim, and CSynth. COSIM was
disabled for this campaign.

The exact completed launcher and its state record are:

```text
/home/luo00466/code_translation-c2hls-hpca2027/artifacts/experiment_matrix/
run_qwen_active_actor_sft_valtest5_nocosim_20260724.sh

/home/luo00466/code_translation-c2hls-hpca2027/artifacts/experiment_matrix/logs/
qwen_active_actor_sft_valtest5_nocosim_20260724/experiment_state.json
```

Its direct one-shot outputs passed both CSim and CSynth on all five held-out
kernels. Agentic summaries are stored under
`code_translation-c2hls-hpca2027/artifacts/` with the same run ID. Results must
be compared against a base-model control with the same prompts, decoding,
candidate budget, skill snapshot, toolchain, and validation policy.

## Historical completed adapters

| Adapter | Supervision scope | Training |
|---|---|---|
| `qwen3.6-27b_qlora_r16_smoke10_1024_20260716` | Mixed historical trajectory smoke | 16 rows, 10 steps, length 1,024 |
| `qwen3.6-27b_translator_qlora_r16_1epoch_2048_20260717` | Assistant turn 0, translator proxy | 157 train rows, 1 epoch, length 2,048 |
| `qwen3.6-27b_orchestrator_qlora_r16_1epoch_3072_20260721` | Orchestrator optimization turns | 185 train rows, 1 epoch, length 3,072 |
| `qwen3.6-27b_active_llm_actors_qlora_r16_1epoch_3072_20260724` | Exact translator, synthesis, and orchestrator attribution | 165 train rows, 1 epoch, length 3,072 |

All four use Qwen3.6-27B, NF4 double-quantized QLoRA, rank 16, alpha 32,
dropout 0.05, completion-only loss, and language-model-only adapters. The
first is an integration smoke rather than a quality result. The last is the
preferred adapter for whole-framework evaluation.

## Gemma status and required work

The Gemma 4 31B active-role corpus is complete and has the same 167/5/32 split
as Qwen after model-specific tokenization. No Gemma `adapter_config.json` was
found under `/mnt/data2/luo00466/c2hls_sft` or
`/mnt/data2/luo00466/c2hls_rl`. The checked-in trainer also describes and
enforces the tested Qwen path; a Gemma adapter must not be reported as trained.

Before a Gemma full run:

1. Add an explicit Gemma model profile to the trainer rather than relying on
   Qwen defaults.
2. Discover and freeze Gemma language projection targets, then assert that no
   vision parameter is trainable.
3. Run `--load-only`, a 10-step QLoRA smoke, adapter reload, and one generation
   smoke.
4. Train one benchmark-disjoint epoch with a fresh output directory.
5. Serve the pinned adapter and repeat the same direct and agentic held-out
   Vitis matrix used for Qwen.

This preserves a model-matched comparison. Reusing Qwen target assumptions or
calling the prepared Gemma Parquet a trained model would invalidate it.

## Verification checklist

Before accepting an SFT result:

- check that train, validation, and test benchmark sets are disjoint;
- retain the role-attribution and model-tokenization manifests;
- retain rejected rows rather than silently dropping them;
- verify completion masks are one contiguous non-empty suffix;
- hash the base snapshot, adapter weights, source corpus, and launch script;
- verify the served model ID and endpoint before generation;
- audit server context truncation in matched A/B experiments;
- use the same decoding and synthesis budget for base and SFT conditions;
- report CSim/CSynth/COSIM status independently from trainer loss; and
- label COSIM as not run when it was not executed.

Relevant unit tests in the source repository are:

```bash
cd /home/luo00466/code_translation-c2hls
/home/luo00466/.conda/envs/py310_2/bin/python -m pytest \
  tests/test_build_role_attributed_sft_dataset.py \
  tests/test_export_model_aware_sft_dataset.py \
  tests/test_train_agentic_sft_qlora.py \
  tests/test_serve_agentic_peft_openai.py \
  tests/test_filter_agentic_sft_role.py
```
