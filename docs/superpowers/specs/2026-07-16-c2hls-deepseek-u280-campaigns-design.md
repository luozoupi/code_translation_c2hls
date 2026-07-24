# Design: c2hls DeepSeek U280 Campaigns (vs ChatHLS Hybrid)

**Date:** 2026-07-16  
**Status:** approved  
**Plan:** `docs/superpowers/plans/2026-07-16-c2hls-deepseek-u280-campaigns.md`  
**Goal:** Run c2hls ChatHLS-bench flash+dataflow campaigns with DeepSeek (API model id `deepseek-chat`) on U280 @ 3.33 ns, fairer comparison to ChatHLS hybrid U280 results that already used DeepSeek + HLSTuner/HLSFixer.

## Problem

Prior c2hls latency ranking vs ChatHLS mixed two confounds:

1. **Model:** c2hls used Devstral-2 123B; ChatHLS U280 used DeepSeek via API.
2. **Stack:** ChatHLS also runs HLSTuner + HLSFixer on GPU; c2hls does not.

We already have Devstral c2hls results (RAG+skills / noRAG+skills / RAG-noskills). We need matching **DeepSeek** c2hls runs on the **same FPGA/clock** as:

`test-chathls/ChatHLS-ACL-26/artifacts/pc2/sessions/hybrid-u280-split-20260717-001649/`

Compute nodes have **no internet**; only login nodes can reach `api.deepseek.com`.

## Decisions (locked)

| Item | Choice |
|------|--------|
| Campaigns | **All three**, sequential: RAG+skills → noRAG+skills → RAG-noskills |
| Model request id | `deepseek-chat` (account maps to DeepSeek V4 flash; same as ChatHLS) |
| Target | U280 part, **3.33 ns / 300 MHz** (already c2hls ChatHLS defaults) |
| LLM serving | **No GPU vLLM**; login-node DeepSeek **queue proxy** |
| Proxy concurrency | **`workers=1`** (fully serial upstream) |
| Compute layout | **16 nodes**, one per bench: **csim + csynth + cosim** on that node |
| Scheduling | Sequential campaigns |
| Peak pricing (Beijing UTC+8) | Windows **09:00–12:00** and **14:00–18:00** |
| Peak behavior | Gate **campaign start**; during peak **pause codegen/DeepSeek only**; Vitis synth/csim/cosim continue; resume LLM off-peak |
| API key | From login env / `~/.bashrc` (`DeepSeek_API` / `DeepSeek_SPI` / `OPENAI_API_KEY`) — same as ChatHLS `setup_deepseek_api.sh` |

## Non-goals

- Porting HLSTuner / HLSFixer into c2hls for this campaign.
- Running DeepSeek and Devstral campaigns in parallel.
- Direct internet from compute nodes.
- Changing ChatHLS hybrid scripts except optional reuse of their proxy as a library path.

## Architecture

```
login node
  ├── deepseek_queue_proxy (workers=1) ──HTTPS──► api.deepseek.com
  ├── peak scheduler / sequential launcher
  └── llm_endpoint.json  (url=http://<login>:<port>/v1)

normal partition (16×)
  └── per-bench HLS worker: gold + phase_b/flash csim+csynth+cosim

normal partition (helpers)
  ├── watch / coordinator
  └── gpu_drain (codegen) ──HTTP──► login proxy
        └── skips claim while Beijing peak OR campaign peak_paused
```

### LLM path

1. Start (or reuse) login DeepSeek queue proxy (prefer wrapping/calling ChatHLS’s existing `deepseek_queue_proxy.py` + `setup_deepseek_api.sh` to avoid drift).
2. Write campaign `llm_endpoint.json` with OpenAI-compatible `url`, `model=deepseek-chat`, mark borrowed/external so coordinator does **not** submit/park a GPU vLLM job.
3. `batch_parallel_gpu_drain.py` loads `OPENAI_BASE_URL` from endpoint file; health via `/models` or `/health`.
4. Set `C2HLS_MODEL` / campaign `pilot.model` to `deepseek-chat`.
5. Real `OPENAI_API_KEY` in drain env (not `EMPTY`).

### Peak gate

- Source of truth: Asia/Shanghai (UTC+8).
- Peak intervals: `[09:00, 12:00)` and `[14:00, 18:00)`.
- **Launcher:** block `start` until off-peak (or `--force` for emergencies).
- **Drain:** if peak, sleep/poll; do not claim codegen jobs; do not call proxy.
- **Synth/cosim workers:** unchanged (keep claiming Vitis work).
- Optional: proxy itself may remain up during peak (idle); drain is the pause point so queued HTTP clients don’t pile up.

### Combined HLS node (one per bench)

Today: 16 synth nodes (csim+csynth) + 16 cosim nodes.

Required: **16 dual-role nodes** (or synth workers that run cosim inline after synth success for the same bench).

Design requirement:

- Preserve queue semantics (job kinds `synth` / `cosim` may remain) but **co-locate** execution on one Slurm allocation per bench index, **or** collapse cosim into synth worker after successful synth.
- Max parallelism across benches: **16** Vitis nodes.
- Codegen remains on the shared drain (not on Vitis nodes).

### Sequential campaign runner

Single entry script (conceptual):

1. Ensure proxy up (workers=1), endpoint healthy from a compute reachability probe.
2. Wait until off-peak.
3. Run campaign A (RAG+skills, DeepSeek).
4. On complete → wait off-peak → campaign B (noRAG+skills).
5. On complete → wait off-peak → campaign C (RAG-noskills).
6. Tear down proxy when all done (or leave up if shared with ChatHLS — prefer dedicated c2hls proxy port to avoid colliding with an active ChatHLS session).

Artifact prefixes (suggested):

- `batch_parallel_chathls_fd_ds_rag_<stamp>`
- `batch_parallel_chathls_fd_ds_skills_<stamp>`
- `batch_parallel_chathls_fd_ds_rag_ns_<stamp>`

Job name prefix distinct (e.g. `bpchds`) so `scancel`/stop doesn’t hit Devstral campaigns.

### Config variants

Reuse existing ChatHLS flash+streaming-dataflow wiring:

| Campaign | Skills | RAG scrape |
|----------|--------|------------|
| RAG+skills | 90-skills flash + dataflow no_RMW skills | ug1399+ug902 (+ prior RAG corpus policy for skills run) |
| noRAG+skills | skills on | RAG off |
| RAG-noskills | skills off | ug1399+ug902 scrape |

Streaming flash→dataflow watcher remains (same as prior ChatHLS FD campaigns), with long endpoint wait and Slurm helpers (not login nohup).

## Comparison protocol

After each campaign (or all three):

- Best-of flash/dataflow `latency_cycles` vs  
  `hybrid-u280-split-20260717-001649/final_latency_csynth.csv` (`csynth_best_cycles`, `passed_optimization=True`).
- Report geomean(lat/U280_ChatHLS) and per-bench wins.
- Also rank the three DeepSeek c2hls configs against each other (same model).

Caveat still documented: ChatHLS retains HLSTuner/HLSFixer; this isolates the **model** confound only.

## Risks / mitigations

| Risk | Mitigation |
|------|------------|
| Compute cannot reach login proxy | Reachability gate before seeding compute (ChatHLS pattern) |
| Peak pause starves flash while gold cosim runs long | Acceptable; codegen resumes off-peak |
| Proxy workers=1 slows flash wall | Intentional for quota; sequential campaigns already serialize |
| Combined node change breaks cosim queue | Feature-flag / DeepSeek-only config; keep legacy 32-node mode for Devstral scripts |
| Accidental GPU vLLM submit | External-endpoint mode must skip `batch_parallel_submit_gpu.sh` |
| Key leakage in logs | Never print API key; reuse ChatHLS setup script patterns |

## Success criteria

1. Three DeepSeek campaigns complete (or clearly fail per bench) without requiring compute internet.
2. No DeepSeek chat/completions during Beijing peak windows (codegen paused).
3. Campaigns only start off-peak.
4. U280 @ 3.33 ns confirmed in selected synth reports (`requested_clock_period_ns=3.33`).
5. Latency table vs U280 ChatHLS CSV published for DeepSeek c2hls RAG+skills (and the other two).

## Implementation sketch (not the plan)

- Reuse ChatHLS proxy scripts from `test-chathls/.../scripts/pc2/` or vendor a thin wrapper under `c2hls/scripts/pc2/`.
- Add `external` / `deepseek_proxy` GPU mode to batch_parallel start + coordinator (no scancel/submit GPU).
- Peak helpers: `is_deepseek_peak_beijing(now)`; drain + launcher hooks.
- Combined HLS worker path for DeepSeek config JSON.
- `start_chathls_deepseek_u280_sequence.sh` orchestrating A→B→C.

## Spec self-review

- No TBDs left for locked product choices.
- Scope excludes HLSTuner/HLSFixer (explicit).
- Combined-node mechanism left as “inline cosim **or** dual-role allocation” — plan must pick one concrete mechanism.
- Peak timezone fixed to Asia/Shanghai.
- Does not commit to copying proxy code vs invoking ChatHLS path — plan must pick one.
