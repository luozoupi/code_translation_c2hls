# agentfw: a general Claude Code agent for HLS optimization

This is a separate contribution from c2hls and HLSmith. There is no flow, no flash/repair loop, no ladder and no fixed
agent roles. Claude Code gets a task workspace, a small set of HLS tools and a goal. It decides how to work: what to
read, what to try, when to synthesize, and when to stop.

## Workspace (created by `new_task.py <task> <ws> <variant>`)

| File | Role |
|---|---|
| `CLAUDE.md` | Task brief. It holds the goal, the hard constraints (correct, fits the U280, keeps the top signature), the device facts, the tools and the finish criteria (best `kernel.cpp` plus `REPORT.md`). It gives no architecture hints and no resource target. |
| `kernel.cpp` | Starts as the reference implementation. The agent edits it. |
| `task.json`, header, testbench, reference | Task definition. The testbench pre-fills outputs with garbage, so read-modify-write designs fail. |
| `bin/hls` | The only way to run Vitis: `submit csim/csynth` (async, Slurm), `wait`, `result`, `diag` (dfdiag), `floor`, `list`. Wall time is 1 h → 3 h → 6 h on timeout. |
| `exp/` | One directory per experiment (kernel snapshot, job scripts, summaries, Vitis work dir) and `log.jsonl` of every tool action. |
| `knowledge/` | Only in the `know` and `team` variants: skill packs, dataflow/ping-pong notes, how the diagnosis works, and AutoSA material. |

## Variants (organization is the experimental variable)

| Variant | Tools | Question |
|---|---|---|
| `base` | Read/Write/Edit/Glob/Grep + `bin/hls` and read-only shell | How far does one agent get with only feedback from real csim/csynth? |
| `know` | `base` + `knowledge/` + WebSearch/WebFetch | Does knowledge acquisition (docs, papers, skills) help, and how much does the agent use it? |
| `team` | `know` + the Agent tool (subagents) | Does self-organized delegation help, at what cost? |

The launcher `run_agent.sh <ws> [hours]` runs a headless `claude -p` with a stream-json transcript. Every tool call,
token count and cost is in `transcript.jsonl`, and every experiment is in `exp/log.jsonl`. Effort is measured the same
way as in `study/effort/`.

## Evaluation

- **Tasks:** `mm1024` (C = A·Bᵀ, fixed bench), then LUD once task 6 provides the bench.
- **Replicates:** 3 per variant.
- **Metrics:**
  - best correct worst-case csynth latency;
  - resources;
  - × compute floor;
  - dfdiag class (true overlap?);
  - experiments run, wall time, tokens, cost;
  - knowledge-acquisition share (reads, web) versus experimentation share.
- **Baselines:** the controlled ladder (plain Opus and DeepSeek), the c2hls agent-mode runs, the manual true-dataflow
  design (task 2), and AutoSA st0_c6 (1,048,845 / 1,054,811 cycles).
