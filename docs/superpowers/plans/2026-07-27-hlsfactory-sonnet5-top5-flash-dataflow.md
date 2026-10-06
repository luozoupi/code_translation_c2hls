# HLSFactory Sonnet 5 Top5 Flash+Dataflow Implementation Plan

> **For agentic workers:** Implement task-by-task. Cosim must not block dataflow.

**Goal:** Run skills + noskills HLSFactory flash→dataflow on 5 benches with `claude-sonnet-5` via login-node Anthropic queue proxies (compute has no internet).

**Architecture:** OpenAI-compat Anthropic proxy on login; c2hls uses OpenAI client when `OPENAI_BASE_URL` is set (even for `claude*`). Flash uses shared high-worker proxy; dataflow uses per-bench Anthropic proxies + exclusive Vitis jobs. Waiter = `wait_hlsfactory_flash_lat_dataflow.sh` with lat_opt=0 so ranked cosim is async and DF does not wait on flash/lat cosim.

**Tech Stack:** bash/Slurm, Anthropic OpenAI-compat API, batch_parallel, existing DF launcher.

---

### Task 1: Anthropic queue proxy
### Task 2: Route Claude via OPENAI_BASE_URL
### Task 3: Launch scripts + top5 JSON
### Task 4: Parallel DF Anthropic proxies + RAG scrub in export
### Task 5: Tests + dry-run
