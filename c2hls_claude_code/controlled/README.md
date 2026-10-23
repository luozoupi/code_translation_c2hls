# Controlled DeepSeek vs Claude Code rerun

The goal is to compare the two LLMs on the c2hls flash flow so that **only the model differs**.

The first campaign (`runs/` for Claude Code, `runs_deepseek/` for DeepSeek) had non-model confounds. Each is removed here:

| Confound in the first campaign | Control here |
|---|---|
| 65,536-token cap shared with DeepSeek's chain of thought. 37% of DeepSeek flash calls were truncated. | Every call gets `max(c2hls budget, CTRL_MIN_MAX_TOKENS=262144)` on both sides. DeepSeek accepts 262,144; the Claude shim caps at its CLI limit of 64,000, which Claude never reached. Truncated or empty replies are retried, up to 2 times. |
| c2hls used DeepSeek's reasoning text as the "answer" when content was empty, which gave compile errors. | Answer = `message.content` only (`ctrl_llm.py`). |
| Each model wrote its own Phase-B baseline, so flash prompts and latency baselines differed. | Both models are seeded with the same baseline kernel: the bench's `hls_baseline.cpp` (`--seed-code`). Zero-shot needs no seed: it starts flash from `plain.cpp` on both sides. |
| Multi-turn format differed: real chat for DeepSeek, a flattened transcript for the Claude shim. | Every call is flattened with `claude_code_shim.render()` into `[system, user]`. Both models receive byte-identical text. |
| Reasoning settings were unrecorded provider defaults. | `C2HLS_THINKING=enabled` (DeepSeek) is set and recorded. Claude uses its CLI default thinking. Both reason. |
| Proxy: 600 s timeout, no retries, no logging. One 502 ended a stage. | `ds_proxy.py`: 3,600 s timeout, per-call JSON log (`ds_proxy_calls.jsonl`), key read from the environment only. Transport errors are retried up to 3 times, identically for both models. |
| c2hls's line-based pragma preflight inserts an `s_axilite` line after each `m_axi` line. A pragma continued with `\` is split, and its tail becomes stray C++ (`'depth' was not declared`). DeepSeek writes continued pragmas and Claude doesn't. Found in this rerun (zero-shot r1 and r3); it may also explain some first-campaign compile errors. | `ctrl_llm.py` joins `\`-continued preprocessor lines before the preflight, which is C translation phase 2, so meaning is unchanged. Logged as `joined_pragma_continuations`. Zero-shot runs made before the fix are in `../runs_controlled/_superseded_before_pragma_fix/` and were rerun. |
| 24 DeepSeek jobs submitted at once. Claude jobs were staggered. | Both models are submitted interleaved, with the same `--begin` offsets. |
| Unequal replicates: Claude had 1 run for some variants. | 3 replicates per variant per model. |

## Files

| File | Role |
|---|---|
| `ctrl_llm.py` | Runtime patch of `C2HLSOrchestrator._call_llm_with_model`, enabled by `run_flash_cc.py --controlled`. Writes a per-try log to `<cell>/ctrl_llm_calls.jsonl`, including a `prompt_sha` used to check first-prompt identity across models. |
| `ds_proxy.py` | DeepSeek forwarding proxy on a login node: `http://login3:18094/v1`, see `ds_proxy_url.txt` |

## Runs

Runs go to `../runs_controlled/`, with job IDs in `../runs_controlled/jobs.txt`.

Variants, each with 3 replicates per model:
- `zero_shot`;
- `skills --refined` (v2);
- `skills --refined --dsp-window 6317:7219` (70–80% DSP window).

Shared environment:

```
CONTROLLED=1 C2HLS_THINKING=enabled C2HLS_FLASH_MAX_TOKENS=262144 CTRL_MIN_MAX_TOKENS=262144
SEED_CODE=<bench>/hls_baseline.cpp   (refined variants)
```

## Remaining, inherent differences (the model)

These belong to the model and are what the comparison measures:
- reply content and code quality;
- reasoning length;
- how each model responds to repair prompts.

Synthesis time (24 h csynth cap, same on both) also depends on the design each model writes.

## Wall-time change (2026-10-12)

The 12 refined-v2 and DSP-window jobs were cut from 2 days to 12 hours, then to **3 hours each**, the same for both models, so they could get through the queue. In the first campaign these flows either finished in under 30 minutes or hung in csynth until the 24-hour timeout, and none of the hung runs produced a passing design.

Those jobs still carry c2hls's 24-hour csynth timeout, which can't be changed for a queued job. A design whose csynth runs past ~2.5 hours is therefore killed by Slurm (job state TIMEOUT) instead of timing out inside the flow. Such a run counts as a failure for the model that wrote the design.

## Compile-check headers and memory (2026-10-12)

**Missing headers.** c2hls's quick `g++` compile check found no Vitis HLS headers on the host (Vitis exists only inside the container). Any design that includes `<hls_stream.h>` or `<ap_int.h>` therefore failed with `fatal error: hls_stream.h: No such file or directory` before synthesis.
- **Who it hit:** only Claude, in 4 of 6 runs. Three runs lost one attempt each; window r1 lost all three and failed. This biased the comparison against Claude.
- **Fix:** the headers were copied once from the container to `/scratch/hpc-prf-llmfpga/asa582/projects/tmp/c2hls/vitis_hls_2023.2_include`, and every controlled run now sets `C2HLS_COMPILE_INCLUDE_PATHS` (a c2hls setting) to that path. It is recorded in each `cc_run_plan.json`.

**Memory.** Job memory was raised from 64 GB to 128 GB for both models. DeepSeek window r3 lost two attempts to csynth `Out of Memory`.

**Reruns.** All 12 refined-v2 and window runs were rerun under these settings (3 h limit each). The runs made before the fix are in `../runs_controlled/_superseded_before_header_fix/`.

## Timeout resubmission (2026-10-13)

Any refined-v2 or window run whose 3 h job ends in TIMEOUT is resubmitted once with **6 hours**, with the same settings and the same rule for both models (`resubmit_timeouts.sh`). The timed-out cells go to `../runs_controlled/_superseded_timeout_3h/`, and their old job IDs to that folder's `jobs.txt`.

## No-skills and DSP-redo variants (2026-10-13)

Two more first-campaign variants were added under the same controls, with 3 replicates per model, interleaved, 6 h, 128 GB, the shared seed kernel, and the header path (`submit_extra.sh`):
- `noskills` (`FLAVOR=noskills`);
- `dspredo` (`FLAVOR=skills,DSP_REDO=1`: DSP floor 300, fill to at least 90% of 9024).

Tags are `ctl_<m>_noskills_r<n>` and `ctl_<m>_dspredo_r<n>`. `compare.py` reports them.

## Flash with skills, no DSP gates (2026-10-13)

`skills` (`FLAVOR=skills`, default pack: 90 skills + gemm-flatten v1, the same pack as controlled DSP redo, no DSP floor or cap) fills the gap between no-skills and DSP redo. It has 3 replicates per model and the same settings as the other variants (`submit_skills.sh`). Tags are `ctl_<m>_skills_r<n>`.
All 6 were still pending, so their limit was cut from 6 h to 3 h, then to **1 h** (`scontrol update`). A run that times out is resubmitted once at 6 h by `resubmit_timeouts.sh`, the same rule as refined v2 and the window.

## Claude isolation audit and shim fix (2026-10-13)

**Audit of every Claude call of the controlled campaign** (63 calls in `logs/shim_20261007T023853/`):
- All reported Claude runs went through the plain shim on port 38585. The tool-enabled agent shim on 36013 was not used.
- The flags were `claude -p --tools "" --disable-slash-commands --strict-mcp-config --setting-sources "" --no-session-persistence`, with `--system-prompt-file` (c2hls's system prompt replaces Claude Code's own) and an empty temporary working directory.
- Per call:
  - 0 web search or web fetch requests, and 0 subagents;
  - only `claude-opus-5-5` was used;
  - fast mode was off;
  - 62 calls had 1 turn, and 1 call had 2 turns (an internal continuation of a long thinking reply, still with no tool use).
- Claude Code still injects a few environment lines into the context (the empty working directory and the account email). It has no tools, no files, no CLAUDE.md and no memory.

**Shim bug found and fixed.** When `claude -p` returned an error together with text, for example "You've hit your session limit ...", the shim returned that text as an HTTP 200 answer. That happened in 24 calls on 2026-10-12, 20:16–20:17, all inside runs that were superseded before the header fix. None of them appear in the reported tables, and every reported Claude call is a real reply. The shim now returns 502 for any CLI error, so the controlled layer treats it as a transport error and retries. The patched shim was restarted on the same port, with logs in `logs/shim_20261013T1130/` and the start script `logs/start_shim_38585.sh`.

## Skills + refined without the DSP floor (2026-10-13)

`refnf` is refined v2 (refined prompts, refined v2 pack, balance gate ≤ 1.5× compute floor, 9024 device cap) with the DSP floor and fill gate removed. It uses the new `run_flash_cc.py --no-dsp-floor` flag (`NO_DSP_FLOOR=1`). It has 3 replicates per model, a 1 h limit, and otherwise the same settings as the rest (`submit_refnf.sh`). It fills the rung between flash + skills and refined v2.

## Neutral refined expertise (2026-10-13)

The `refnf` prompts still ask, in words, for most of the DSP. Examples: "size the design to the target device", "the parallel datapath should use most of the device's DSP budget", "a small datapath on a large device is a performance bug". The pack also has the device-budget skills. So `refnf` is really "a resource requirement in words, with no gate".

`refneu` (`--refined-neutral`, `REFINED_NEUTRAL=1`) is refined expertise with no resource requirement:
- **Prompts:** the hold-back cues stay removed. Every sizing push is deleted or turned into the cap "every resource below 100%". `flash_refined.NEUTRAL_*` asserts each edit, and the rendered prompts are checked for banned phrases.
- **Resource block:** the DEVICE BUDGET block becomes DEVICE LIMITS.
- **Repair template:** the balance repair template drops all DSP-fill wording.
- **Skill pack:** `..._refined_v2_neutral.json` (136 skills, from `make_neutral_skills.py`). It drops `hls-size-parallel-datapath-to-device-budget` and `avoid-idle-device-on-compute-bound-kernel`, and rewrites the three "device budget" wordings as a cap. The two balance skills lose their "filled / after scaling" assumptions, and their ids are renamed to `hls-balance-compute-and-transfer` and `avoid-transfer-bound-datapath`.
- **Gates:** no DSP floor or fill gate. The balance gate (latency at most 1.5× the compute floor) and the 9024 cap stay.

It has 3 replicates per model and a 1 h limit (`submit_refneu.sh`).
