# Effort and knowledge-acquisition measurements: Opus 5.5 (Claude Code) in the HLS experiments

Generated 2026-10-13T14:51:29 from logs only (read-only). Machine-readable per-run records and aggregates: `effort.json`. Values are measured; "n/a" = not recorded (reason given). Medians are over cells (runs) unless stated.

## 1. Method and data sources

* **Controlled campaign (plain mode)**: `runs_controlled/*_ctl_cc_<variant>_r<n>_*/autosa_mm/` (latest stamp per tag; `_superseded*` dirs and smoke dirs excluded from the main tables, see Appendix A). Per LLM try: `ctrl_llm_calls.jsonl` (outcome, prompt/completion tokens, elapsed_s, len_try/tx_try retry counters). Flash attempts and reject stages: `autosa_mm_multistep_results.json` (`steps[0].attempt_results` when present, which exists only for failed flash steps) plus the "Previous attempts in this phase: - Attempt N: STAGE" list in the last user prompt of `autosa_mm_history.json` (the success case carries no attempt_results; the last successful attempt has no reject stage). The per-attempt budget is 3 (`attempt_count`) for all skills/noskills variants and 1 for zero-shot.
* **Claude CLI own accounting** (thinking tokens, cost, turns, web, denials, subagents): `logs/shim_20261007T023853/call_*/cli_stdout.json` and `logs/shim_20261013T1130/call_*/cli_stdout.json`. These runs are `claude -p --tools ""`, so tool activity is zero by construction; the CLI JSON confirms it (num_turns = 1 on every matched call, web_search = web_fetch = 0, permission_denials = 0, subagents spawned = 0).
* **Agent mode**: `logs/shim_agent_20261007T045824/` (`calls.jsonl`, `call_*/agent_transcript.jsonl` stream-json with every tool_use, final `result` event with usage/permission_denials/api_error, `workspace/checks/<n>_<mode>.txt` = each hls_check result). Agent cells: `runs/ladder_agent_r*/skills_L*`, `skills_agent_plus_gemm_v2`, `skills_dspredo_agent`, `skills_refined_agent_r1`, `skills_refined_v2_agent_r1` (11 cells; all `OPENAI_BASE_URL=http://login3:36013/v1`).
* **Reasoning** = `usage.output_tokens_details.thinking_tokens` (CLI). Thinking text itself is redacted in transcripts (empty `thinking` blocks with signatures), so only token counts exist. For DeepSeek: `reasoning_tokens` in `ctrl_llm_calls.jsonl`.
* **Knowledge acquisition** = Read + Grep + Glob + read-only Bash (cat/ls/head/tail/grep/find/...) + web search/fetch. **Implementation/experimentation** = Edit + Write + mutating Bash (sed -i, cp, python, redirects) + executed hls_check runs. **Reasoning** = thinking tokens. The Bash read-only vs mutating split is a regex heuristic on the command text (hls_check.sh commands are classified exactly).

## 2. Matching shim calls to runs, and confidence

**Controlled runs.** `ctrl_llm_calls.jsonl.t` is the local PDT request *start*; the shim `calls.jsonl.t` is the request *end* (verified: shim end = ctrl t + elapsed_s, e.g. skills_r1 10:56:19 + 84.1 s = 10:57:43 = shim call 166). Rule applied: shim `out_tokens` must equal ctrl `completion_tokens` exactly AND |shim_end - (ctrl_t + elapsed_s)| <= 15 s, each shim call used once. Result: 70 of 70 Claude ctrl calls (all Claude cells including smoke and superseded) matched; max time offset 0.8 s; shim `in_tokens` equal to ctrl `prompt_tokens` on 70 of 70. Confidence: **very high** (exact output-token equality plus sub-second timing). Shim prompt_chars differs from ctrl prompt_chars by ~1.5k chars (shim measures the sent user text); sizes were not used for matching. The 176 plain-shim calls (168 in `shim_20261007T023853`, 8 in `shim_20261013T1130`) include 106 unmatched calls (dates 2026-10-07, 2026-10-08, 2026-10-09, 2026-10-13): they belong to other campaigns under `runs/` (ladder, ablation, refined, win7080, team_site, etc.) and to a one-line ping (restart log n=1); they are out of scope and are not attributed to any run here. DeepSeek calls do not pass through the Claude shim; their fields come only from `ctrl_llm_calls.jsonl`.

**Agent runs.** Each cell's `autosa_mm_history.json` `llm_usage.events[]` carries a UTC timestamp and output_tokens per LLM call; shim t is PDT end time. Rule: event UTC - 7 h within 5 s of shim t AND equal output_tokens. Result: 25 of 25 agent-shim calls matched one-to-one to 25 pipeline events across 11 cells (max offset 1.0 s). Confidence: **very high**. Each agent cell's first event is a `plain`-mode call (prompt 4,283 chars, 389-806 output tokens, baseline translation) routed through the same shim; the remaining events are `agent`-mode calls (shim regex on FLASH MODE / HARD REJECT prompts). The agent shim log has 7 plain + 18 agent calls = 25, all mapped to cell events. The 7 cells that start from the plain C code (ladder L1 x3 and the four single-cell families) begin with a plain-mode baseline call; ladder L2/L3 cells are seeded from the previous stage and have none.

## 3. Controlled campaign, Claude Opus 5.5, plain mode: per run

| variant | rep | LLM calls | out tok | thinking tok | think % | LLM wall s | cost USD | turns | non-ok calls | len/tx retries | flash rejects | reject stages | flash ok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| zs | r1 | 1 | 3,200 | 1,539 | 48% | 66.6 | 0.067 | 1 | 0 | 0/0 | 0 | - | yes |
| zs | r2 | 1 | 2,682 | 832 | 31% | 35.8 | 0.057 | 1 | 0 | 0/0 | 0 | - | yes |
| zs | r3 | 1 | 2,708 | 997 | 37% | 35.3 | 0.057 | 1 | 0 | 0/0 | 0 | - | yes |
| noskills | r1 | 1 | 4,812 | 2,678 | 56% | 54.6 | 0.136 | 1 | 0 | 0/0 | 0 | - | yes |
| noskills | r2 | 1 | 4,672 | 2,533 | 54% | 47.0 | 0.128 | 1 | 0 | 0/0 | 0 | - | yes |
| noskills | r3 | 1 | 5,272 | 3,306 | 63% | 53.4 | 0.140 | 1 | 0 | 0/0 | 0 | - | yes |
| skills | r1 | 1 | 7,654 | 4,927 | 64% | 84.1 | 0.790 | 1 | 0 | 0/0 | 0 | - | yes |
| skills | r2 | 1 | 4,859 | 2,850 | 59% | 49.5 | 0.730 | 1 | 0 | 0/0 | 0 | - | yes |
| skills | r3 | 1 | 7,080 | 4,095 | 58% | 71.0 | 0.774 | 1 | 0 | 0/0 | 0 | - | yes |
| dspredo | r1 | 2 | 24,294 | 17,636 | 73% | 241.8 | 1.815 | 2 | 0 | 0/0 | 1 | a0:SYNTH_OTHER | yes |
| dspredo | r2 | 2 | 19,125 | 12,899 | 67% | 208.3 | 1.708 | 2 | 0 | 0/0 | 1 | a0:SYNTH_OTHER | yes |
| dspredo | r3 | 2 | 20,366 | 14,233 | 70% | 245.1 | 1.733 | 2 | 0 | 0/0 | 0 | n/a | incomplete |
| refneu | r1 | 2 | 25,774 | 17,612 | 68% | 251.4 | 1.951 | 2 | 0 | 0/0 | 1 | a0:SYNTH_OTHER | yes |
| refneu | r2 | 1 | 28,655 | 23,643 | 83% | 288.6 | 1.264 | 1 | 0 | 0/0 | 0 | - | yes |
| refneu | r3 | 1 | 10,371 | 7,468 | 72% | 223.2 | 0.892 | 1 | 0 | 0/0 | 0 | - | yes |
| refnf | r1 | 1 | 9,457 | 6,165 | 65% | 106.8 | 0.888 | 1 | 0 | 0/0 | 0 | - | yes |
| refnf | r2 | 1 | 9,377 | 6,150 | 66% | 87.5 | 0.880 | 1 | 0 | 0/0 | 0 | - | yes |
| refnf | r3 | 1 | 17,191 | 12,485 | 73% | 170.2 | 1.036 | 1 | 0 | 0/0 | 0 | - | yes |
| refv2 | r1 | 2 | 44,718 | 35,613 | 80% | 460.1 | 2.394 | 2 | 0 | 0/0 | 1 | a0:CSIM_FAILED | yes |
| refv2 | r2 | 1 | 30,648 | 25,171 | 82% | 302.5 | 1.309 | 1 | 0 | 0/0 | 0 | - | yes |
| refv2 | r3 | 3 | 35,767 | 21,159 | 59% | 329.0 | 3.046 | 3 | 0 | 0/0 | 2 | a0:SYNTH_OTHER, a1:SYNTH_OTHER | yes |
| win7080 | r1 | 1 | 27,423 | 21,690 | 79% | 302.1 | 1.250 | 1 | 0 | 0/0 | 0 | - | yes |
| win7080 | r2 | 1 | 24,655 | 19,304 | 78% | 250.5 | 1.188 | 1 | 0 | 0/0 | 0 | - | yes |
| win7080 | r3 | 1 | 35,168 | 29,435 | 84% | 344.0 | 1.398 | 1 | 0 | 0/0 | 0 | - | yes |


Every ctrl row has outcome=ok (0 truncated/empty/transport errors; len_try = tx_try = 0). "LLM calls" = flash attempts made (1 call per attempt; budget 3, zero-shot 1). Web search/fetch = 0, permission denials = 0, subagents = 0, tool calls = 0 for every Claude controlled call (CLI-measured). "flash ok" is the flash step result; `skills_ctl_cc_dspredo_r3` has no results json (incomplete, reason not recorded; its 2 LLM calls are still counted).


### 3a. Controlled Claude, by variant (medians per cell, totals over cells)

| variant | cells | calls med/total | output tok med/total | thinking tok med/total | LLM wall s med/total | cost USD med/total | flash rejects | reject stages | flash ok / complete cells |
|---|---|---|---|---|---|---|---|---|---|
| zs | 3 | 1.0 / 3 | 2,708 / 8,590 | 997 / 3,368 | 36 / 138 | 0.06 / 0.18 | 0 | - | 3/3 |
| noskills | 3 | 1.0 / 3 | 4,812 / 14,756 | 2,678 / 8,517 | 53 / 155 | 0.14 / 0.40 | 0 | - | 3/3 |
| skills | 3 | 1.0 / 3 | 7,080 / 19,593 | 4,095 / 11,872 | 71 / 205 | 0.77 / 2.29 | 0 | - | 3/3 |
| dspredo | 3 | 2.0 / 6 | 20,366 / 63,785 | 14,233 / 44,768 | 242 / 695 | 1.73 / 5.26 | 2 | SYNTH_OTHERx2 | 2/2 |
| refneu | 3 | 1.0 / 4 | 25,774 / 64,800 | 17,612 / 48,723 | 251 / 763 | 1.26 / 4.11 | 1 | SYNTH_OTHERx1 | 3/3 |
| refnf | 3 | 1.0 / 3 | 9,457 / 36,025 | 6,165 / 24,800 | 107 / 364 | 0.89 / 2.80 | 0 | - | 3/3 |
| refv2 | 3 | 2.0 / 6 | 35,767 / 111,133 | 25,171 / 81,943 | 329 / 1,092 | 2.39 / 6.75 | 3 | CSIM_FAILEDx1, SYNTH_OTHERx2 | 3/3 |
| win7080 | 3 | 1.0 / 3 | 27,423 / 87,246 | 21,690 / 70,429 | 302 / 897 | 1.25 / 3.84 | 0 | - | 3/3 |
| **all** | 24 | - / 31 | - / 405,928 | - / 294,420 | - / 4,308 | - / 25.63 |  |  |  |


Reading: the zero-shot prompt (1.3k chars) costs ~1k thinking tokens and ~36 s; adding the ~205k-char skills prompt raises thinking to 4-6k; the DSP-targeted variants (win7080, refneu, refv2, dspredo) raise it 5-10x and are the only ones that needed retries. Wall time scales with output tokens at roughly 100 tok/s.


## 4. DeepSeek (deepseek-v4-flash) controlled contrast

| variant | cells | complete cells (results json) | calls med/total | completion tok med/total | reasoning tok med/total | LLM wall s med/total | reject stages (from complete cells) | flash ok cells |
|---|---|---|---|---|---|---|---|---|
| zs | 3 | 3 | 1 / 3 | 13,097 / 41,597 | 11,884 / 38,110 | 77 / 238 | - | 3 |
| noskills | 3 | 3 | 1 / 3 | 46,324 / 147,280 | 45,176 / 144,281 | 275 / 833 | - | 3 |
| skills | 3 | 3 | 1 / 3 | 34,261 / 115,156 | 33,133 / 111,500 | 192 / 609 | - | 3 |
| dspredo | 3 | 0 | 2 / 7 | 67,443 / 220,435 | 65,113 / 210,900 | 369 / 1,213 | - | 0 |
| refneu | 3 | 3 | 1 / 4 | 57,462 / 204,738 | 55,996 / 198,274 | 277 / 997 | SYNTH_OTHERx1 | 3 |
| refnf | 3 | 3 | 1 / 3 | 49,743 / 174,918 | 48,016 / 169,624 | 248 / 876 | - | 3 |
| refv2 | 3 | 1 | 3 / 7 | 171,176 / 569,348 | 165,504 / 555,847 | 957 / 2,964 | dsp_fillx3 | 0 |
| win7080 | 3 | 1 | 2 / 7 | 175,016 / 566,190 | 171,172 / 552,431 | 986 / 3,222 | dsp_fillx3 | 0 |
| **all** | 24 |  | - / 37 | - / 2,039,662 | - / 1,980,967 | - / 10,952 |  |  |


Contrast: over the 24 non-smoke cells per model, DeepSeek used 2,039,662 completion tokens (1,980,967 reasoning, 97%) in 37 calls and 10,952 s; Opus used 405,928 output tokens (294,420 thinking, 73%) in 31 calls and 4,308 s. DeepSeek therefore emits ~5x more tokens for the same prompts, almost all reasoning. DeepSeek cost: n/a (no cost field in ctrl_llm_calls.jsonl). Incomplete DeepSeek cells (no results json; reason not recorded; the `_superseded_timeout_3h` dir name suggests a 3 h timeout, cause for the current incomplete cells not verified): dspredo_r1, dspredo_r2, dspredo_r3, refv2_r2, refv2_r3, win7080_r1, win7080_r3. For these only the LLM-call rows exist, so flash attempts/reject stages are n/a. In DeepSeek `ctrl_llm_calls.jsonl` all outcomes are `ok`; 2 non-call event rows (`joined_pragma_continuations`) are post-processing, not retries.


### 4a. DeepSeek per run

| variant | rep | calls | completion tok | reasoning tok | wall s | reject stages | flash ok |
|---|---|---|---|---|---|---|---|
| zs | r1 | 1 | 6,719 | 5,581 | 34.8 | - | yes |
| zs | r2 | 1 | 21,781 | 20,645 | 125.9 | - | yes |
| zs | r3 | 1 | 13,097 | 11,884 | 76.9 | - | yes |
| noskills | r1 | 1 | 27,644 | 26,869 | 159.5 | - | yes |
| noskills | r2 | 1 | 73,312 | 72,236 | 398.4 | - | yes |
| noskills | r3 | 1 | 46,324 | 45,176 | 275.1 | - | yes |
| skills | r1 | 1 | 30,368 | 29,151 | 158.4 | - | yes |
| skills | r2 | 1 | 34,261 | 33,133 | 192.2 | - | yes |
| skills | r3 | 1 | 50,527 | 49,216 | 258.6 | - | yes |
| dspredo | r1 | 2 | 66,700 | 64,582 | 360.1 | n/a | incomplete |
| dspredo | r2 | 3 | 86,292 | 81,205 | 483.8 | n/a | incomplete |
| dspredo | r3 | 2 | 67,443 | 65,113 | 369.2 | n/a | incomplete |
| refneu | r1 | 1 | 57,462 | 55,996 | 277.3 | - | yes |
| refneu | r2 | 1 | 51,410 | 49,772 | 243.9 | - | yes |
| refneu | r3 | 2 | 95,866 | 92,506 | 476.1 | a0:SYNTH_OTHER | yes |
| refnf | r1 | 1 | 76,833 | 74,648 | 384.2 | - | yes |
| refnf | r2 | 1 | 49,743 | 48,016 | 247.8 | - | yes |
| refnf | r3 | 1 | 48,342 | 46,960 | 243.6 | - | yes |
| refv2 | r1 | 3 | 293,460 | 287,133 | 1,422.1 | a0:dsp_fill, a1:dsp_fill, a2:dsp_fill | NO |
| refv2 | r2 | 1 | 104,712 | 103,210 | 585.3 | n/a | incomplete |
| refv2 | r3 | 3 | 171,176 | 165,504 | 956.6 | n/a | incomplete |
| win7080 | r1 | 2 | 175,016 | 171,172 | 986.4 | n/a | incomplete |
| win7080 | r2 | 3 | 276,823 | 270,412 | 1,574.0 | a0:dsp_fill, a1:dsp_fill, a2:dsp_fill | NO |
| win7080 | r3 | 2 | 114,351 | 110,847 | 662.1 | n/a | incomplete |


## 5. Agent mode (Claude Code with Read/Write/Edit/Glob/Grep and Bash restricted to ./hls_check.sh): per run

| cell | LLM calls (plain+agent) | agent calls | CLI turns | out tok | thinking tok | LLM wall s | cost USD | 429 usage-limit agent calls | perm. denials | pipeline flash rejects |
|---|---|---|---|---|---|---|---|---|---|---|
| ladder_agent_r1_20261009T052542Z/L1_20261009T052 | 4 | 3 | 16 | 27,313 | 14,420 | 5,001 | 2.59 | 3 | 1 | a0:compile_check, a1:dsp_floor, a2:dsp_floor |
| ladder_agent_r2_20261009T052542Z/L1_20261009T052 | 3 | 2 | 17 | 39,373 | 15,927 | 534 | 2.86 | 0 | 3 | a0:SYNTH_OTHER |
| ladder_agent_r2_20261009T052542Z/L2_20261009T061 | 1 | 1 | 11 | 13,077 | 3,032 | 444 | 1.27 | 0 | 1 | - |
| ladder_agent_r2_20261009T052542Z/L3_20261009T063 | 1 | 1 | 7 | 19,658 | 8,780 | 1,368 | 1.39 | 1 | 1 | - |
| ladder_agent_r3_20261009T052541Z/L1_20261009T052 | 3 | 2 | 21 | 56,564 | 29,420 | 2,253 | 3.41 | 0 | 3 | a0:SYNTH_OTHER |
| ladder_agent_r3_20261009T052541Z/L2_20261009T063 | 3 | 3 | 16 | 9,807 | 1,780 | 4,277 | 2.29 | 2 | 1 | a0:SYNTH_OTHER, a1:SYNTH_OTHER |
| ladder_agent_r3_20261009T052541Z/L3_20261009T101 | 1 | 1 | 7 | 20,227 | 4,973 | 9,224 | 2.27 | 0 | 1 | - |
| agent_plus_gemm_v2_20261007T120001Z | 2 | 1 | 10 | 10,009 | 4,639 | 371 | 1.14 | 0 | 1 | - |
| dspredo_agent_20261007T231953Z | 2 | 1 | 11 | 40,463 | 33,795 | 1,259 | 1.94 | 1 | 1 | - |
| refined_agent_r1_20261008T022545Z | 2 | 1 | 10 | 22,676 | 11,976 | 971 | 1.57 | 0 | 1 | - |
| refined_v2_agent_r1_20261008T075311Z | 3 | 2 | 12 | 24,125 | 11,179 | 2,007 | 2.31 | 0 | 0 | a0:SYNTH_OTHER |


Per-call details (turns, tool use, files, hls_check results) are in `effort.json` under each record's `calls`. LLM wall = sum of shim elapsed_s (includes hls_check execution time inside the agent loop, e.g. csynth, so it is not pure model time). "429 usage-limit": the Claude subscription session limit was hit (CLI `api_error=usage_limit_reached`); 3 calls hit it mid-run after doing tool work (calls 4, 13, 19: the shim still returned the workspace kernel), and 4 further calls (20-23) failed immediately with 0 tokens (the pipeline retried). These count in "calls" and cost 0 USD when 0-token.


### 5a. Knowledge acquisition vs implementation/experimentation vs reasoning (agent cells)

| cell | Read | Grep | Glob | Bash read-only | web search+fetch | Edit | Write | Bash mutating | hls_check compile/csim/csynth (executed) | hls_check DENIED | thinking tok |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ladder_agent_r1_20261009T052542Z/L1_20261009 | 0 | 0 | 0 | 1 | 0 | 5 | 2 | 1 | 0/2/2 | 0 | 14,420 |
| ladder_agent_r2_20261009T052542Z/L1_20261009 | 0 | 3 | 0 | 2 | 0 | 0 | 2 | 2 | 1/2/1 | 1 | 15,927 |
| ladder_agent_r2_20261009T052542Z/L2_20261009 | 3 | 0 | 0 | 0 | 0 | 4 | 0 | 1 | 0/1/1 | 0 | 3,032 |
| ladder_agent_r2_20261009T052542Z/L3_20261009 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0/1/1 | 1 | 8,780 |
| ladder_agent_r3_20261009T052541Z/L1_20261009 | 2 | 0 | 0 | 2 | 0 | 3 | 2 | 1 | 1/3/2 | 2 | 29,420 |
| ladder_agent_r3_20261009T052541Z/L2_20261009 | 1 | 0 | 0 | 2 | 0 | 7 | 0 | 0 | 0/1/1 | 1 | 1,780 |
| ladder_agent_r3_20261009T052541Z/L3_20261009 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0/1/1 | 1 | 4,973 |
| agent_plus_gemm_v2_20261007T120001Z | 0 | 0 | 0 | 1 | 0 | 1 | 1 | 0 | 0/2/2 | 1 | 4,639 |
| dspredo_agent_20261007T231953Z | 0 | 0 | 0 | 1 | 0 | 2 | 2 | 0 | 0/1/2 | 1 | 33,795 |
| refined_agent_r1_20261008T022545Z | 0 | 0 | 0 | 2 | 0 | 0 | 2 | 1 | 0/2/2 | 0 | 11,976 |
| refined_v2_agent_r1_20261008T075311Z | 0 | 0 | 0 | 3 | 0 | 0 | 2 | 0 | 0/2/2 | 0 | 11,179 |


**All 11 agent cells** (25 LLM calls: 18 agent + 7 plain): calls median 2.0, total 25; agent calls median 1.0, total 18; CLI turns median 11.0, total 138; output tokens median 22,676.0, total 283,292; thinking tokens median 11,179.0, total 139,921; LLM wall s median 1,367.7, total 27,708; cost USD median 2.27, total 23.03; permission denials total 14. Tool calls total: {'Bash': 65, 'Write': 15, 'Edit': 22, 'Grep': 3, 'Read': 8}. hls_check executed: {'csim': 18, 'csynth': 17, 'compile': 2}; outcomes: every executed check returned success (csynth OK x17, csim PASSED x18, compile OK x2); no failing hls_check was recorded (failures appear only as pipeline-level rejects after the agent returns). Web search/fetch: 0. Subagents: 0 (CLI `subagent_stats.spawned` = 0 everywhere). Glob: 0. Files read with the Read tool: only the agent's own `workspace/kernel.cpp` (8 Reads); no external docs, no `kernel.h`/`testbench.cpp` via Read (testbench was inspected through Bash `cat` in at least call 2). Grep patterns (3, all in one cell) searched the cell's own prior report text for loop names, not docs.


Permission denials (14 in total across agent calls) are Bash commands outside the allowed `./hls_check.sh compile|csim|csynth` form (compound `cd ...; sed -i ...; ./hls_check.sh ...` or `cp ... && sed ...`);  (the examples inspected are compound shell commands; not every denial command was individually reviewed); they are listed with the full command in the CLI result `permission_denials` of each `call_*/agent_transcript.jsonl`.


### 5b. Agent run families (aggregate)

| family | cells | calls med/total | out tok med/total | thinking med/total | wall s med/total | cost USD med/total | turns total | Reads | tool calls total | hls_check by mode | 429s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ladder_agent | 7 | 3 / 16 | 20,227 / 186,019 | 8,780 / 78,332 | 2,253 / 23,100 | 2.29 / 16.08 | 95 | 8 | {'Bash': 41, 'Write': 8, 'Edit': 19, 'Grep': 3, 'Read': 8} | {'csim': 11, 'csynth': 9, 'compile': 2} | 6 |
| skills_agent_plus_gemm_v2 | 1 | 2 / 2 | 10,009 / 10,009 | 4,639 / 4,639 | 371 / 371 | 1.14 / 1.14 | 10 | 0 | {'Bash': 6, 'Write': 1, 'Edit': 1} | {'csynth': 2, 'csim': 2} | 0 |
| skills_dspredo_agent | 1 | 2 / 2 | 40,463 / 40,463 | 33,795 / 33,795 | 1,259 / 1,259 | 1.94 / 1.94 | 11 | 0 | {'Bash': 5, 'Write': 2, 'Edit': 2} | {'csim': 1, 'csynth': 2} | 1 |
| skills_refined_agent | 1 | 2 / 2 | 22,676 / 22,676 | 11,976 / 11,976 | 971 / 971 | 1.57 / 1.57 | 10 | 0 | {'Bash': 6, 'Write': 2} | {'csim': 2, 'csynth': 2} | 0 |
| skills_refined_v2_agent | 1 | 3 / 3 | 24,125 / 24,125 | 11,179 / 11,179 | 2,007 / 2,007 | 2.31 / 2.31 | 12 | 0 | {'Bash': 7, 'Write': 2} | {'csim': 2, 'csynth': 2} | 0 |


`ladder_agent` = 7 cells (r1: L1; r2, r3: L1-L3), the only multi-stage family; others are single-cell families (n=1), so their "median" is that one value.


## 6. Plain vs agent contrast (Claude)

Controlled plain skills-prompt cells: median thinking 4,095 tok and 71 s per cell (1 call, 0 tool calls). Agent cells: median thinking 11,179 tok, 22,676 output tok, 11 turns, $2.27 per cell, with ~1.5 csynth + 1.6 csim runs per cell. In agent mode, effort goes to experimentation (hls_check) and implementation, not to knowledge acquisition: reading activity is minimal (8 Reads of its own file, 3 Greps, 0 web).


## Appendix A. Superseded and smoke cells (counted separately, excluded above)

| dir | model | variant | rep | stamp | calls | completion tok | thinking/reasoning tok | wall s | cost USD (cc) | outcomes | results json |
|---|---|---|---|---|---|---|---|---|---|---|---|
| smoke | cc | zs | smoke | 20261012T104035Z | 1 | 2,876 | 1,169 | 42.9 | 0.07 | {'ok': 1} | results |
| smoke | ds | zs | smoke | 20261012T104034Z | 1 | 21,683 | 20,194 | 105.0 | n/a | {'ok': 1} | results |
| _superseded_before_header_fix | cc | refv2 | r1 | 20261013T024044Z | 3 | 43,800 | 30,782 | 442.3 | 3.21 | {'ok': 3} | results |
| _superseded_before_header_fix | cc | refv2 | r2 | 20261013T025626Z | 2 | 43,061 | 34,165 | 949.8 | 2.35 | {'ok': 2} | results |
| _superseded_before_header_fix | cc | refv2 | r3 | 20261013T025626Z | 1 | 22,429 | 18,563 | 222.0 | 1.14 | {'ok': 1} | results |
| _superseded_before_header_fix | cc | win7080 | r1 | 20261013T025626Z | 25 | 47,088 | 39,092 | 1,176.8 | 1.64 | {'ok': 25} | results |
| _superseded_before_header_fix | cc | win7080 | r2 | 20261013T025626Z | 2 | 78,147 | 67,064 | 1,074.4 | 3.60 | {'ok': 2} | results |
| _superseded_before_header_fix | cc | win7080 | r3 | 20261013T025626Z | 2 | 38,700 | 28,552 | 936.1 | 2.26 | {'ok': 2} | results |
| _superseded_before_header_fix | ds | refv2 | r1 | 20261013T024044Z | 2 | 153,014 | 149,433 | 841.5 | n/a | {'ok': 2} | no results |
| _superseded_before_header_fix | ds | refv2 | r2 | 20261013T025626Z | 2 | 170,191 | 166,786 | 1,002.0 | n/a | {'ok': 2} | no results |
| _superseded_before_header_fix | ds | refv2 | r3 | 20261013T025626Z | 2 | 160,327 | 157,134 | 913.4 | n/a | {'ok': 2} | no results |
| _superseded_before_header_fix | ds | win7080 | r1 | 20261013T025626Z | 2 | 335,376 | 332,184 | 1,939.5 | n/a | {'ok': 2} | no results |
| _superseded_before_header_fix | ds | win7080 | r2 | 20261013T025626Z | 2 | 189,369 | 185,662 | 1,101.0 | n/a | {'ok': 2} | no results |
| _superseded_before_header_fix | ds | win7080 | r3 | 20261013T025626Z | 3 | 112,884 | 108,002 | 642.9 | n/a | {'ok': 3} | results |
| _superseded_before_pragma_fix | cc | zs | r1 | 20261012T120021Z | 1 | 2,433 | 826 | 35.9 | 0.06 | {'ok': 1} | results |
| _superseded_before_pragma_fix | cc | zs | r2 | 20261012T120403Z | 1 | 3,021 | 1,123 | 28.8 | 0.06 | {'ok': 1} | results |
| _superseded_before_pragma_fix | cc | zs | r3 | 20261012T120747Z | 1 | 2,827 | 1,337 | 32.6 | 0.06 | {'ok': 1} | results |
| _superseded_before_pragma_fix | ds | zs | r1 | 20261012T120402Z | 1 | 18,238 | 16,988 | 96.9 | n/a | {'ok': 1} | results |
| _superseded_before_pragma_fix | ds | zs | r2 | 20261012T120747Z | 1 | 25,476 | 24,234 | 118.5 | n/a | {'ok': 1} | results |
| _superseded_before_pragma_fix | ds | zs | r3 | 20261012T120747Z | 1 | 13,030 | 11,293 | 73.7 | n/a | {'ok': 1} | results |
| _superseded_timeout_3h | ds | refneu | r3 | 20261013T195001Z | 1 | 74,997 | 73,680 | 361.8 | n/a | {'ok': 1} | no results |
| _superseded_timeout_3h | ds | refv2 | r2 | 20261013T052202Z | 2 | 192,868 | 189,259 | 1,069.6 | n/a | {'ok': 2} | no results |
| _superseded_timeout_3h | ds | refv2 | r3 | 20261013T053610Z | 2 | 144,441 | 141,045 | 770.5 | n/a | {'ok': 2} | no results |
| _superseded_timeout_3h | ds | win7080 | r1 | 20261013T051501Z | 2 | 117,299 | 113,950 | 683.2 | n/a | {'ok': 2} | no results |
| _superseded_timeout_3h | ds | win7080 | r2 | 20261013T053244Z | 2 | 178,871 | 174,299 | 974.8 | n/a | {'ok': 2} | no results |
| _superseded_timeout_3h | ds | win7080 | r3 | 20261013T053943Z | 2 | 236,121 | 232,132 | 1,320.1 | n/a | {'ok': 2} | no results |


Superseded Claude subtotal: 9 cells, 38 calls, 221,504 thinking tok, $14.38, 4,899 s. Note `win7080_r1` (before_header_fix) made 25 calls (cause not investigated; the superseded dir name says before_header_fix), the highest-call-count cell in the study; it is superseded and not in the main tables. Superseded DeepSeek: 15 cells. The `_superseded_timeout_3h` dirs contain ctrl logs without some result files.


## Appendix B. Unavailable or limited

* **Thinking text / reasoning content**: redacted (empty thinking blocks with signatures); only token counts. Per-turn thinking tokens are not recorded (only the call total; `system.thinking_tokens` stream events are estimates and not used).
* **Plain-mode "knowledge acquisition"**: none exists by construction (no tools); the only knowledge input is the prompt (skills text ~205k chars in skill variants vs 1.3k in zs). Prompt tokens are reported in effort.json (`prompt_tokens`); cache read/creation tokens appear in each shim call's cli fields (cache_* = 0 for most plain calls).
* **DeepSeek**: no cost, no turns, no tool fields; incomplete cells have no flash-attempt data; reasoning tokens as reported by the API.
* **Failure stage of the final failed flash attempt**: for successful runs, the last attempt has no reject stage (success). For failed flash steps the stages come from `attempt_results`. For incomplete cells: n/a.
* **Reject-stage coverage in agent cells**: pipeline-level flash attempts (reject stages) are from the cell history; they are separate from the agent's internal hls_check iterations (agent loops inside a single call).
* **hls_check outcomes**: only successful results exist in the checks dirs; whether the agent's check was refused (denied) is taken from CLI `permission_denials`. A transcript-vs-executed mismatch (e.g. 5 hls_check commands in transcript vs 4 check files) = denied commands.
* **Other campaigns** (106 plain-shim calls, ~930k output tokens, in `runs/` plain ladder/ablation/refined runs) are not attributed to runs here.
* **Bash classification** read-only vs mutating is heuristic.
* **Time**: LLM wall times are shim elapsed_s (agent calls include hls_check tool time).


## 7. Agentic Claude Code outside c2hls

Sources: `/scratch/asa582/clean_u280/`, `~/.claude/projects/-scratch-asa582-clean-u280-run2/*.jsonl`, `c2hls_claude_code/ablation/`, and `~/.claude/projects/-scratch-asa582-workspaces-dse-v4-claude/*.jsonl` (streamed with Python; transcripts never loaded whole). Note: only the `dse_v4_claude` project directory exists under `~/.claude/projects`; there is **no separate dse_v3_claude session directory** (the st0_c1 design session itself read dse_v3_claude files). Counting rules: tokens come from the `usage` of each assistant message (deduplicated by message id; Claude Code writes each content block as its own line); thinking tokens are only available where the CLI emitted a final `result` (clean runs, C0d, shim) and **not** in interactive session transcripts (n/a; only `thinkingDurationMs` sums). Interactive sessions are forked/copied (`forkedFrom`), so a session file repeats its parent's history; to avoid double counting, per-session figures below are *own* messages and tool calls (anything already seen in an earlier session, ordered by first timestamp, is attributed to the earlier one); raw per-file totals are in effort.json (`raw_*`). Wall time = "active seconds" (sum of gaps between assistant messages, each capped at 600 s; idle days between turns excluded). Bash classes are regex heuristics on command text: a command mentioning csynth/csim counts even if it only polls or reads logs, so they are upper bounds on runs.

### 7.1 Clean U280 runs (`/scratch/asa582/clean_u280/`): no skills/config/spec, one shot, tools Read/Write/Edit/Glob/Grep, **no Bash** (cannot run Vitis; no feedback)

Launch: `claude -p --restricted --tools Read,Write,Edit,Glob,Grep --disable-slash-commands --strict-mcp-config --no-session-persistence`, prompt = one sentence (PROMPT.txt) (from README.md). Judged afterwards by plain Vitis csim/csynth.


| run | turns | out tok | thinking tok | think % | wall s | cost USD | tool calls | web | denials | subagents | csim correct | csynth cycles |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gen(run1) | 6 | 27,251 | 21,051 | 77% | 266 | 0.776 | n/a (not saved) | 0 | 0 | 0 | True | 2,106,102 |
| run3 | 5 | 27,852 | 21,212 | 76% | 274 | 0.789 | {'Read': 2, 'Glob': 1, 'Write': 1} | 0 | 0 | 0 | False | 1,052,795 |
| run4 | 5 | 18,101 | 14,242 | 79% | 196 | 0.517 | {'Glob': 1, 'Read': 2, 'Write': 1} | 0 | 0 | 0 | True | 4,751,407 |
| run5 | 5 | 30,549 | 25,473 | 83% | 311 | 0.865 | {'Glob': 1, 'Read': 2, 'Write': 1} | 0 | 0 | 0 | True | 1,056,942 |
| run6 | 5 | 37,312 | 31,380 | 84% | 405 | 1.055 | {'Read': 2, 'Glob': 1, 'Write': 1} | 0 | 0 | 0 | True | n/a |
| run7 | 5 | 39,922 | 35,250 | 88% | 406 | 1.128 | {'Read': 2, 'Glob': 1, 'Write': 1} | 0 | 0 | 0 | True | 1,114,236 |


6 one-shot runs: median 23,342 thinking tok, 29,200 output tok, 293 s, $0.83; totals 148,608 thinking tok, $5.13. Knowledge acquisition in run3-7 is identical every time: Glob 1 + Read 2 (the seed `kernel.cpp` and `kernel.h`), then Write 1 (the result); 0 Grep, 0 web, 0 docs. All effort is reasoning (~76-90% of output tokens are thinking) and one write. Turns = 5 (CLI turn count incl. tool turns). gen/run2-7 csim correctness and cycles above are from each `*_eval/csim_summary.txt` and `csynth_summary.json` (run3 failed csim).


Interactive run2 session (same prompt, VS Code-style interactive Claude Code in `clean_u280/run2`, transcripts in `~/.claude`):


| session | title | assistant msgs | out tok | tool calls | files read | active s | thinking s (duration) | cost USD | csim ok | csynth cycles |
|---|---|---|---|---|---|---|---|---|---|---|
| 63de8597 | U280 kernel.cpp performance optimization | 6 | 34,677 | {'Read': 2, 'Glob': 1, 'Write': 1, 'Edit': 4, 'mcp__ide__getDiagnostics': 1} | run2/kernel.cpp; run2/kernel.h | 340 | 285 | 1.074 | True | 1,057,915 |
| acd21b7e | Unfinished work | 6 | 3,593 | {'Glob': 5, 'Read': 2, 'Grep': 1} | run2/kernel.h; run2/kernel.cpp | 95 | 31 | 0.229 | None | n/a |


The 63de8597 session wrote the kernel and then made 4 Edits to it (self-revision without any compile feedback; one `mcp__ide__getDiagnostics` call); acd21b7e ("unfinished work") only read the two files and grepped, no edits. Thinking tokens: n/a (duration only).


### 7.2 Prompt ablation (`c2hls_claude_code/ablation/`): NOT tool-enabled sessions

Launcher: `ablation/run_ablation.py generate` posts each prompt to the **plain (tool-less) shim** `/chat/completions` (`call_shim`), one request per rep, then `submit` runs csim+csynth. So variants C0, C1, F0-F4 are single plain LLM calls: no tools, no reads, no turns beyond 1, no Bash (zero knowledge acquisition by construction). `usage.json` holds only prompt/completion tokens; thinking tokens, cost and wall time were recovered by matching each run to the shim `calls.jsonl`/`cli_stdout.json` on exact completion-token equality, restricted to shim calls on 2026-10-07 and unique (all matches listed below as "unique"). C0d_default_system is different: `cli.json` per run (a `claude -p` call with the default Claude Code system prompt, run in /tmp/c0d-*; no launcher script found in the repo; 1 turn, 0 tool calls observed in the CLI result, no web, no denials).


| variant | runs | completion tok median | completion tok total | thinking tok median | thinking tok total | wall s median | cost USD total | thinking/cost matched | csim passed | median csynth cycles |
|---|---|---|---|---|---|---|---|---|---|---|
| C0_clean | 5 | 5,791 | 35,938 | 3,169 | 21,408 | 58 | 0.76 | 5/5 | 4/5 | 10,528,573 |
| C0d_default_system | 2 | 11,615 | 23,230 | 7,425 | 14,850 | 115 | 0.50 | 2/2 | 1/2 | 4,237,108 |
| C1_clean_plus_skills | 5 | 8,301 | 42,709 | 5,696 | 26,497 | 80 | 4.10 | 5/5 | 5/5 | 2,493,752 |
| F0_full | 5 | 6,957 | 40,582 | 4,264 | 27,304 | 77 | 4.21 | 5/5 | 5/5 | 9,632,584 |
| F1_no_resource_caps | 2 | 6,606 | 13,211 | 3,995 | 7,990 | 63 | 1.61 | 2/2 | 2/2 | 12,877,168 |
| F2_no_conservative_guidance | 5 | 8,487 | 49,465 | 5,262 | 33,182 | 88 | 4.37 | 5/5 | 5/5 | 5,531,473 |
| F3_no_caps_no_guidance | 2 | 7,031 | 14,062 | 4,162 | 8,323 | 66 | 1.63 | 2/2 | 2/2 | 11,680,568 |
| F4_no_skills | 2 | 4,486 | 8,971 | 2,572 | 5,143 | 44 | 0.25 | 2/2 | 2/2 | 52,479,592 |


Per-run values are in effort.json (`agentic_outside_c2hls.ablation`). Unmatched rows show n/a for thinking/cost/wall (non-unique or no shim record).


### 7.3 DSE v3/v4 Claude Code sessions (`~/.claude/projects/-scratch-asa582-workspaces-dse-v4-claude/`)

Session `82ac5e3f` (and its byte-identical copy `738dfb6f`, 0 own messages) is the **actual st0_c1 design session** ("Build st0_c1 in this directory. Read CLAUDE.md and prompt_user.txt..."), i.e. the flow in `dse_v4_claude/CLAUDE.md` (edit kernel.cpp, `check_structural.py`, then sbatch csynth/csim). The other sessions are coordination/infrastructure work on the same project dir (TAPA fast-cosim, LightningSim, ablation tracking, csynth analysis), not per-design generation; they are reported separately as overhead. Session 72130906 (the live coordinating session; excluded from per-design effort) is reported only as orchestration overhead.


| session | title | own assistant msgs | own out tok | own cache-read tok | active h | Read | Grep | Glob | Bash read-only | web search+fetch | Edit+Write | Bash mutating | Bash sbatch | check_structural | Bash csim/eval-mentions | Bash csynth-mentions | Agent (subagent) calls | Skill calls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 82ac5e3f | st0_c1 kernel build with dataflow | 110 | 133,254 | 19,792,504 | 2.5 | 7 | 0 | 0 | 23 | 0 | 13 | 5 | 7 | 2 | 10 | 26 | 0 | 0 |
| 738dfb6f | st0_c1 kernel build with dataflow | 0 | 0 | 0 | 2.5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5b47bff3 | tapa-fast-cosim-autosa-mm | 525 | 561,405 | 165,629,245 | 11.6 | 4 | 0 | 0 | 106 | 7 | 21 | 141 | 47 | 0 | 64 | 76 | 3 | 0 |
| 346624e7 | csynth analysis only | 132 | 105,033 | 53,686,347 | 6.1 | 1 | 0 | 0 | 16 | 1 | 5 | 44 | 13 | 0 | 6 | 3 | 11 | 0 |
| 72130906* | tapa-fast-cosim-autosa-mm | 524 | 418,536 | 237,717,075 | 20.8 | 0 | 0 | 0 | 78 | 0 | 12 | 185 | 18 | 0 | 18 | 37 | 37 | 0 |
| f0958df1 | csynth analysis only ⑂ | 59 | 71,555 | 34,274,941 | 7.8 | 1 | 0 | 0 | 13 | 3 | 5 | 15 | 7 | 0 | 1 | 9 | 1 | 0 |
| 98a90d7b | csynth analysis only ⑂ | 256 | 324,683 | 89,103,446 | 4.2 | 16 | 0 | 0 | 47 | 1 | 25 | 74 | 38 | 0 | 10 | 26 | 0 | 0 |
| d622e92a | csynth analysis only ⑂ | 0 | 0 | 0 | 2.9 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |


`*` = excluded orchestration session (own-message figures are after attributing the shared prefix to earlier sessions, so they understate its raw totals). Subagent (Agent tool) token usage is **not recorded** in these transcripts (sidechain messages = 0), so subagent effort is n/a; only the number of Agent calls is known. Cost: the transcript `cost-state` entry gives session-cumulative USD, which is inherited by forks, so per-session cost is shown only where it is unambiguous (below). Thinking tokens: n/a (not in usage); `thinkingDurationMs` sums are in effort.json.


**Design session 82ac5e3f (st0_c1)**: 110 assistant messages, 133,254 output tokens, 19,792,504 cache-read tokens, active 2.5 h, session cost $14.13 (cost-state). Tool calls {'Bash': 70, 'Read': 7, 'Write': 9, 'ToolSearch': 1, 'Edit': 4}. Knowledge acquisition: Read x7 of `dse_v4_claude/prompt_user.txt, dse_v3_claude/prompt_user.txt, dse_v3_claude/kernel.cpp, dse_v3_claude/CLAUDE.md, dse_v3_claude/pe64_simd16/kernel.cpp, dse_v3_claude/pe128_simd8/kernel.cpp, dse_v3_claude/check_dataflow.py`, plus 23 read-only Bash commands (cat/ls/grep style; top cat'd files {'post_flash_dse_v4.py': 2, 'README.md': 1, 'kernel.h': 1, 'sbatch_csynth.sh': 1, 'prompt_user.txt;': 1, 'CLAUDE.md': 1, 'kernel.cpp;': 1, 'post_flash_dse_v3.py;': 1, 'b4r9u0ux7.output;': 1, 'common.py': 1, 'bcpvrmha2.output;': 1, 'bvird1xcg.output;': 1, 'sol1.log': 1, 'bvy1m7y85.output': 1}); 0 Grep, 0 Glob, 0 web, 0 Skill, 0 subagents. Implementation/experimentation: Write x9 + Edit x4 (files {'/scratch/asa582/workspaces/dse_v3_claude/check_dataflow.py': 4, '/scratch/asa582/workspaces/dse_v3_claude/kernel.cpp': 2, '/scratch/asa582/workspaces/dse_v4_claude/gen_kernel.py': 1, '/scratch/asa582/workspaces/dse_v3_claude/kernel_dataflow.cpp': 1, '/scratch/asa582/workspaces/dse_v3_claude/DATAFLOW.md': 1, '/scratch/asa582/workspaces/dse_v3_claude/CLAUDE.md': 1, '/scratch/asa582/workspaces/dse_v3_claude/pe64_simd16/kernel.cpp': 1, '/scratch/asa582/workspaces/dse_v3_claude/pe128_simd8/kernel.cpp': 1, '/scratch/asa582/workspaces/dse_v3_claude/gen_dataflow.py': 1}), `check_structural.py` x2, sbatch x7, Bash commands mentioning csynth x26 / csim-or-eval x10, Slurm status polls x18. Reasoning: thinking-duration sum 631 s; thinking tokens n/a.


**Orchestration overhead (72130906, raw file totals, includes inherited prefix and this ongoing task)**: 782 assistant messages, 629,548 output tokens, cost-state $81.61, tool calls {'Bash': 575, 'Edit': 4, 'Write': 16, 'ToolSearch': 3, 'WebFetch': 4, 'WebSearch': 4, 'TaskStop': 5, 'Read': 1, 'Agent': 48, 'Artifact': 11, 'SendMessage': 2}. Only the own (non-inherited) part is in the table above.


**Infrastructure/coordination sessions other than the design session** (5b47bff3, 346624e7, f0958df1, 98a90d7b; own parts): 972 assistant messages, 1,062,676 output tokens, Bash 789, Agent calls 15, Reads 22, web 12 (LightningSim/OmniSim searches, GitHub fetch of sharc-lab/LightningSim). Files read there are source files of the tools being modified (tapa_fast_cosim templates, lightningsim Rust sources), i.e. tool-development, not HLS-design knowledge. Whole-session cost-state totals (cumulative, not additive across forks): 5b47bff3 $21.78, f0958df1 $56.54 (fork), 72130906 $81.61.


### 7.4 Aggregate by flow type (knowledge acquisition / implementation / reasoning)

| flow | n | turns | knowledge acquisition | implementation / experimentation | reasoning (thinking tok) | output tok | wall | cost | note |
|---|---|---|---|---|---|---|---|---|---|
| c2hls controlled plain (Opus, no tools) | 24 | 1/call | 0 by construction (prompt only: 1.3k chars zs ... ~205k chars skills) | 0 | 294,420 thinking (73% of out) | 405,928 out | 4,308 s total | $25.63 | sections 2-3 |
| ablation plain shim (C0,C1,F0-F4; no tools) | 26 | 1/call | 0 (prompt only) | 0 | 129,847 | 204,938 | 1998 s | $16.92 | plain tool-less; shim-matched |
| ablation C0d default-system `claude -p` | 2 | 1/call | 0 | 0 | 14,850 | 23,230 | 230 s | $0.50 | no tool use observed |
| clean one-shot with file tools, no Bash (gen, run3-7) | 6 | 5/run | Glob 1 + Read 2 per run (seed files only); 0 Grep/web | Write 1 per run (no compile/run possible) | 148,608 | 180,987 | 1858 s | $5.13 | no feedback loop |
| c2hls agent mode (shim_agent, tools + hls_check) | 11 | median 11 CLI turns/cell | see 5a | Edit 22 + Write 15 + hls_check 37 executed (csim 18, csynth 17, compile 2) | 139,921 thinking | 283,292 out | 27,708 s | $23.03 | section 5 |
| DSE v4 st0_c1 design session (82ac5e3f) | 1 | 110 asst msgs | Read 7 + read-only Bash 23; web 0 | Edit+Write 13, sbatch 7, check_structural 2 | n/a (duration 631 s) | 133,254 | 9127 s active | $14.13 | tools + Slurm; main-model usage only |


Reading: only the DSE design session and c2hls agent mode have an experiment loop (compile/csynth/csim feedback). In every flow measured, explicit knowledge acquisition (Read/Grep/Glob/web) is small (0-16 reads, 0-3 greps, 0-1 web); the DSE design session reads its CLAUDE.md, prompt, and sibling kernels (7 files) and then spends its effort on edit/csynth iterations. Web access occurred only in the infrastructure/coordination sessions (LightningSim/OmniSim searches), never in a design-generation flow.


### 7.5 Gaps
* No dse_v3 session transcripts exist under `~/.claude/projects` (only the dse_v4 project dir, plus two empty-memory c0d dirs).
* Interactive sessions: thinking tokens n/a; subagent usage n/a; cost only as session-cumulative `cost-state` (forks inherit it); per-session cost for forks not attributable.
* Run3-7 and gen: per-tool inputs for gen are not saved (gen_result.json is only the final result); gen tool counts are therefore n/a (not assumed equal to run3-7).
* Ablation C0d launcher script not found; its configuration is inferred from cli.json only. Ablation thinking/cost/wall for non-uniquely-matched rows is n/a.
* Active-time uses a 600 s idle cap (heuristic).
