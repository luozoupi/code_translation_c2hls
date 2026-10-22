# c2hls flash flow with Claude Code as the LLM

The c2hls flash flow is unchanged: its prompts, skills, repair loops, gates, post-flash DSE and stream chains, and artifacts all come from the read-only checkout at `/scratch/asa582/workspaces/code_translation_c2hls`. The only change is the model endpoint. c2hls sends every LLM call through an OpenAI client when `OPENAI_BASE_URL` is set (`C2HLSOrchestrator._call_llm_with_model`). Here that URL points at `claude_code_shim.py`, which answers each request with one headless `claude -p`.

| File | Role |
|---|---|
| `claude_code_shim.py` | OpenAI-compatible `/v1/chat/completions` server. Each request runs one stateless `claude -p`: the request's system messages replace Claude Code's system prompt, all tools are off, no skills, MCP, settings, CLAUDE.md or memory, and the working directory is empty. Logs every call. |
| `start_shim.sh` | Starts the shim on this login node (compute nodes have no internet) on a free port. Writes `shim_url.txt` and `shim.pid`. |
| `run_flash_cc.py` | Runs one flash cell exactly like the fir batch worker: `configure_site("fir")`, then `apply_mm_flow_flavor`, then `configure_autosa_campaign_env`, then bench timeouts, then `run_benchmark_multistep`. Sets the shim URL and model last. `--dry-run` prints the configuration. |
| `sbatch_flash_cc.sh` | Slurm wrapper: fir Vitis container, cosim off, 48 h. |

## Run

```bash
cd /scratch/asa582/workspaces/c2hls_claude_code
./start_shim.sh 2                      # on a login node; prints http://loginN:PORT/v1 (2 concurrent claude -p)
source /scratch/asa582/workspaces/dse_v3_claude/env.sh
"$PY" run_flash_cc.py --flavor skills --shim-url "$(cat shim_url.txt)" --dry-run   # check the config
sbatch --export=ALL,FLAVOR=skills sbatch_flash_cc.sh                             # full flash + DSE + stream
kill "$(cat shim.pid)"                 # stop the shim when no job needs it
```

Flavors, as defined in `scripts/pc2/autosa_flash_lib.py`:

| Flavor | Variant | Skills | Phase B | Turns | DSE + stream |
|---|---|---|---|---:|---|
| `skills` | aav_n_gf | all + avoids, 90-skill GEMM-flatten pack | yes | 3 | yes |
| `aav_n_90` | aav_n_90 | all + avoids, 90-skill pack | yes | 3 | yes |
| `noskills` | noskills | off | yes | 3 | yes |
| `zero_shot` | zero_shot | off | skipped | 1 | no |
| `one_shot` | one_shot | off | skipped | 1 | no |

The bench defaults to `artifacts/pc2/autosa_mm_ijk_benches/n1024/autosa_mm` in the checkout, which is only read. It is 1024³ fp32 on xcu280 at 3.33 ns, with the same `kernel.h` and `testbench.cpp` as the DSE v3 trials and the clean baselines. Use `--bench-root` or `--bench` for another autosa_ready-style bench.

## Agent mode (flash step)

`AGENT=1 ./start_shim.sh 2` starts a second shim (`shim_agent_url.txt`) in which flash-step requests run as a tool-using Claude Code session instead of one tool-less reply. Those are the requests whose prompt starts with `FLASH MODE:`, `HARD REJECT —` or the quality-repair text. Every other call (Phase B, compile, synthesis and correctness repairs) stays plain.

Each agent session gets its own workspace under `logs/shim_agent_*/call_NNNNN/workspace/`:
- the request's current kernel as `kernel.cpp`, plus the bench's `kernel.h` and `testbench.cpp`;
- `./hls_check.sh compile|csim|csynth`, which runs the same c2hls checks (`compile_check_cpp`, `hls_eval.run_csim`, `hls_eval.run_hls_synthesis`) on a compute node through `srun` and blocks until done. It allows `MAX_CHECKS` (default 12) checks per session.

The session receives c2hls's own system and user prompt (skills included), plus a short appendix that describes the workspace and repeats the output contract. Its tools are Read, Write, Edit, Glob, Grep and Bash, but Bash may only run `./hls_check.sh`; `--restricted` keeps file access inside the workspace. The final reply must carry the kernel in a ```` ```cpp ```` fence. If it does not, the shim appends the workspace `kernel.cpp`. c2hls then runs its own gates, csynth and csim exactly as in plain mode.

```bash
AGENT=1 ./start_shim.sh 2
sbatch --export=ALL,FLAVOR=skills,SKILLS_JSON='skills_ii_target_miss_solutions_added(90skills)_plus_gemm_flatten_v2.json',CLAUDE_CODE_SHIM_URL=$(cat shim_agent_url.txt) sbatch_flash_cc.sh
```

Per agent call, the log directory holds `agent_transcript.jsonl` (every event of the session), `agent_tool_calls.json`, the workspace with `checks/` (each check's summary), and the `calls.jsonl` line with `mode`, `checks_used` and the tool-call count. `sbatch_flash_cc.sh` raises c2hls's LLM timeout to 14 h (`CC_LLM_TIMEOUT`), because one agent session can run several csynth checks.

## Output

- `runs/<flavor>[_tag]_<stamp>/<bench>/`: the normal c2hls cell artifacts (`*_multistep_results.json`, `*_history.json` with every message and LLM usage, `*_flash_opt.cpp`, `*_selected.cpp`, `*_final.cpp`, `steps/`, post-flash DSE folders), plus `cc_run_plan.json` and `cc_run_summary.json`.
- `logs/shim_<stamp>/`: `calls.jsonl` (one line per LLM call: tokens, time, cost, finish reason) and `call_NNNNN/` (system prompt, prompt, reply, raw CLI result).
- `logs/flash_cc-<jobid>.{out,err}`: the Slurm logs.

## Notes

- `claude -p` uses this account's Claude Code login. Usage counts against it, and `calls.jsonl` records the per-call cost the CLI reports.
- `--workers` limits concurrent `claude -p` calls. Extra requests wait up to 2 h (`--queue-timeout`). When `OPENAI_BASE_URL` is set, c2hls DSE v2 drops to one LLM worker anyway.
- c2hls's `max_tokens` becomes `CLAUDE_CODE_MAX_OUTPUT_TOKENS`, capped at 64,000. A reply cut off at the cap is returned with `finish_reason=length`, so c2hls's truncated-fence continuation logic still works. Multi-turn requests (continuations) are rendered as one transcript.
- Temperature is never set by c2hls, and the shim does not set one either.
