# Study progress: effort, true dataflow, DSP 50–65%, general agent, LUD

This is the persistent record for the seven-part study requested on 2026-10-13. Update it after every step; it must
let a new session resume without the chat history. Measured values only. Anything estimated is marked "(est.)".

Root: `/scratch/hpc-prf-llmfpga/asa582/projects/workspaces/c2hls_claude_code/study/`. Earlier work stays where it is and is never overwritten.

## Fixed rules

- Device xcu280-fsvh2892-2L-e (Alveo U280), clock 3.33 ns, Vitis HLS 2023.2 (Fir container). Check the generated TCL.
- **Csynth wall time:** submit at 1 h. On TIMEOUT, resubmit at 3 h. On a second TIMEOUT, resubmit at 6 h. On a third,
  count it as a failure.
- **Correctness:** strict csim on the 1024³ bench. Csynth alone never counts as correct.
- **Kernel semantics:** C = A·Bᵀ (overwrite), as AutoSA computes. The user confirmed this on 2026-10-13. Old results used
  the c2hls bench, whose `plain.cpp` says C += but whose testbench cannot tell the two apart. New experiments use the fixed
  bench (task 0). Old results are kept and labelled with the bench they used.
- Storage is limited: no waveforms, node-local temp for simulators.
- The c2hls checkout (`code_translation_c2hls`) is read-only.

## Inventory (2026-10-13, start of study)

| Item | State | Where |
|---|---|---|
| Controlled Opus 5.5 vs DeepSeek v4.1-flash ladder (8 steps × 3 reps × 2 models) | Done except neutral refined Opus r1 and DeepSeek r3 (check) | `runs_controlled/`, `controlled/`, page `controlled/story/` |
| Story page | Published (v10) | https://claude.ai/artifact/MQ2wMCbXqZGTjLFKiNZAND |
| Load/compute/store breakdown, 35 designs | Done | `controlled/story/bottleneck/breakdown.json` |
| Structure columns (DATAFLOW / ping-pong / streams), source and log heuristics | Done, shallow | `controlled/story/make_story.py:structure()` |
| Earlier c2hls ping-pong judge (design notes) | Exists in c2hls docs, not in our tooling | `code_translation_c2hls/docs/pc2/2026-09-09-dataflow-pingpong-judge.md` |
| Claude agent mode (shim with tools + `hls_check.sh`) | Exists; used only for ladder and refined-agent runs | `claude_code_shim.py`, `agent/` |
| Vitis cosim of 13 earlier designs | Running for days; no cycle counts yet | `cosim/` |
| LightningSim v2023 predictions | Done for the 13 cosim designs | `cosim/lsim/T4_n1024.md` |
| AutoSA checkout and container | Present; `autosa_tests/lu` exists | `/scratch/hpc-prf-llmfpga/asa582/projects/workspaces/AutoSA` |
| Bench C-semantics fix | **Done**: `study/bench_fixed/` (C = A·Bᵀ, C poisoned with 12345.0f); re-csim of the 41 old kernels submitted | task 0 |
| 1. Effort and knowledge-extraction analysis | **Done for all existing runs** (`study/effort/EFFORT.md`); agentfw runs to be added | task 1 |
| 2. Manual true dataflow of refnf r2 | **Running**: v0 (C= baseline), v1 (DATAFLOW tile loop, PIPO A/C, B streamed in during tile 0) | `study/manual_df/` |
| 3. DSP 50–65% sets | **Submitted**: 15 jobs (s5065 cc/ds, r5065 cc/ds, a5065 agent) | `study/dsp5065/` |
| 4. General Claude Code agentic framework | **Built; 3 pilots running** (base, know, team on mm1024) | `study/agentfw/` (DESIGN.md) |
| 5. Dataflow/ping-pong diagnostics | **v1 tool works** on known cases | `study/dfdiag/dfdiag.py` |
| 6. AutoSA LUD DSE and ablation | **Research agent running** (repo, paper, configs) | `study/lud/` |

## Plan (dependency order)

0. **Fixed bench** `study/bench_fixed/`. `plain.cpp` writes C = A·Bᵀ. The testbench fills C with a NaN/garbage pattern
   before the call and checks every element. Then re-csim every existing passing kernel on it (cheap: 2 CPU, 16 GB) and
   label each one `C=`-correct or `C+=`-only. No reruns of old LLM jobs.
1. **Effort analysis**, from logs only:
   - controlled plain runs: `ctrl_llm_calls.jsonl`, shim `cli_stdout.json`, which has thinking tokens, turns,
     server_tool_use and duration;
   - agent runs: shim_agent `agent_tool_calls.json` / transcripts.
   Plain mode has tools disabled, so knowledge extraction there is zero by construction; report that as measured.
2. **Manual true DATAFLOW + ping-pong of `ctl_cc_refnf_r2`.** Hand rewrite, csim on the fixed bench, csynth, and the
   task-5 diagnostics.
5. **Diagnostic tool `study/dfdiag/`.** Parse source, top and module csynth reports, dataflow logs and channel types.
   Classify into: ineffective pragma / task concurrency / correct ping-pong / partial overlap / full overlap. Validate on
   known cases (refv2 r3, win r1 II=16 loader, agent L2, task-2 rewrite).
3. **DSP 50–65% sets** (floor 4,512, cap 5,865 of 9,024) for every test type with >60% DSP in its results: DSP redo
   (stock skills plus requirement), refined + requirement, refined + window, agentic ladder. Both models where the type
   is model-comparable. Fixed bench.
4. **General Claude Code agent framework** `study/agentfw/`. A headless Claude Code with HLS tools (csim, csynth,
   report parser, dfdiag) and knowledge tools; no fixed ladder. Evaluated on MM and LUD against the controlled baselines.
6. **AutoSA LUD.** Fetch the repo and paper; find the best published LUD configuration; run AutoSA on U280 at 3.33 ns.
   Then the ablation (zero-shot, plain ladder, agentic) on LUD with the same methods.
7. **Integration:** the page gets new sections. Every number keeps its provenance.

## Log

- 2026-10-13: study started; inventory above; plan written.

- 2026-10-13: fixed bench built (`study/bench_fixed/`). Re-csim of the 41 old passing kernels plus 2 controls (`study/recsim/`).
  First 20 results: all 17 finished Opus kernels **fail** with C poisoned (they read C, as `plain.cpp`'s `C = C + A·B`
  asks); the 3 DeepSeek no-skills kernels pass (they overwrite C). This is a spec mismatch in the old bench, not a model
  error.
- 2026-10-13: effort analysis of the c2hls runs done (`study/effort/EFFORT.md`).
  - **Controlled Opus:** 31 calls, 405,928 output tokens (294,420 thinking), $25.63, 0 tool calls by design.
  - **DeepSeek:** 2.04M completion tokens.
  - **Agent mode:** 25 calls; 65 Bash, 22 Edit, 15 Write, 8 Read (own kernel only), 3 Grep, 0 web; 37 hls_checks
    (18 csim, 17 csynth, 2 compile); 14 permission denials; 7 usage-limit errors.
  - Agentic sessions outside c2hls (clean_u280, ablation/, DSE v3/v4) are being added.
- 2026-10-13: `study/jobs/ledger.py` enforces 1 h → 3 h → 6 h wall time (`ledger.py escalate`). Agent shim for the
  study runs on login3:36014 with the fixed bench (`logs/start_shim_agent_36014.sh`; checks at 1 h).
  `sbatch_flash_cc.sh` gained `BENCH_ROOT`.
- 2026-10-13: dfdiag v1 classifies:
  - refv2 r3: PINGPONG, FULL (16 tiles at interval 41,018; compute-bound);
  - win r1: TASK_CONCURRENCY, but paced by an II=16 A loader;
  - win r3: TASK_CONCURRENCY, FULL, compute-bound;
  - refv2 r2: INEFFECTIVE (DATAFLOW over loaders only; compute serial);
  - refnf r2: NO_DATAFLOW.
- 2026-10-13: effort analysis extended to the agentic sessions outside c2hls (EFFORT.md section 7):
  - clean U280 one-shots: Glob 1, Read 2, Write 1, median 23k thinking tokens;
  - C0–F4 ablation: tool-less shim calls;
  - DSE v4 st0_c1 session `82ac5e3f`: 133k output tokens, 7 Reads, 23 read-only Bash, 13 Edit/Write, 7 sbatch, 0 web.

  No existing flow did any real knowledge acquisition: no web use, and reads only of its own files and seeds.
- 2026-10-13: manual_df v0 (C= baseline of refnf r2) csynth = 1,246,225 cycles, 5,120 DSP. v1 pending.
- 2026-10-13: agentfw built (`tools/hls`, `new_task.py`, `run_agent.sh`, `knowledge/`). The smoke csim of the reference
  passes on the fixed bench.
  - The first pilots were aborted after a few turns: symlinked bin/ and knowledge/ were blocked by Claude Code path
    security, and compound shell commands needed approval. Kept in `runs/_aborted_pilot/`.
  - Relaunched 15:02 PDT on login3 with copied files and a one-command-per-call rule: `runs/mm1024_{base,know,team}_r1`,
    10 h cap each.
- 2026-10-13: the story page artifact (old account) is no longer reachable from this account. It must be republished
  from `controlled/story/story.html` as a new artifact.
- 2026-10-13: **task 2 result.** Manual v1 (true DATAFLOW tile loop + PIPO ping-pong + B streamed during tile 0) gives
  1,062,813 / 1,071,597 cycles at 5,120 DSP. csim passes on the fixed bench. That is −14.7% vs v0 (1,246,225), and
  dfdiag says PINGPONG/FULL. v2 (flattened compute) is submitted. Details in `manual_df/RESULTS.md`.
- 2026-10-13: bench controls OK: the fixed reference passes and the old C += reference fails (1,048,576 errors).
- 2026-10-13: LUD.
  - **Research:** `lud/LUD_RESEARCH.md`. The paper has LU only at N = 12–24 on a U250 (RTL sim, 250 MHz), with no
    sa-sizes; the repo has one recipe (space_time 3, array_part −1). AutoSA cannot tile LU and ships no U280 model.
  - **DSE:** N ∈ {12, 16, 20, 24, 32} × space_time 0–5, generation with a 180 s timeout; csim and csynth via the ledger
    (`lud/dse.sh`).
  - **Bench:** `lud/bench/n32/lud`, N=32, L/U poisoned. The checker fixes AutoSA's own testbench, which compared L only
    where nothing is written.
  - **Ablation:** zero-shot smoke for both models submitted (`lud/submit_ablation.sh`).
- 2026-10-13 (evening): results batch.
  - **Manual v2** (flattened compute) 1,062,797 / 1,071,582 cycles, 5,120 DSP, csim passes; same as v1 (no gain).
  - **LUD DSE csim** (30 designs). AutoSA's host check: st2/st4/st5 pass at every N; st0/st1/st3 fail at every N.
    `lud/check/parse_printed.py` compares the printed matrices with a numpy Doolittle LU: L is exact in all designs,
    and in st0/1/3 U is wrong (row k shifted one column; U[k][k] missing). The AutoSA golden itself is right.
    Regenerating without `--hls` (`lud/check/nohls_*`) gives an identical kernel body (only extern "C" / header
    inlining differ), so this is AutoSA codegen at rev b61a1b4, including the README recipe (st3).
  - **LUD DSE csynth** (table `lud/check/dse_table.md`):
    - st2 fails pre-synthesis at every N (HLS 200-970: stream not produced/consumed exactly once);
    - st5 at N=24/32 ran out of memory at 64 GB and was resubmitted at 192 GB (`*_mem192`);
    - st3/st4 at N≥20 timed out at 1 h and are running at 3 h;
    - correct and synthesized so far: n12_st4 1,318/1,450 cycles (330 DSP), n16_st4 1,755/1,995 (600 DSP).
  - **LUD zero-shot smoke** (N=32, strict bench):
    - Opus: csim pass, 6,125/9,677 cycles, 155 DSP;
    - DeepSeek: csim pass, 35,341/130,573 cycles, 3 DSP.

    Submitted the rest of the LUD ablation (zs r2–3; noskills/skills/refneu r1–3; both models; 22 jobs).
  - **DSP 50–65%**:
    - r5065 cc r1/r2: 1,114,698 / 1,066,430 cycles;
    - r5065 ds r2: 1,351,825;
    - s5065 cc r1/r2/r3: 1,268,881 / 4,200,655 / 7,368,021;
    - s5065 ds r2: 23,186,945;
    - s5065 ds r1: no accepted design (DeepSeek never reached ≥4,512 DSP; falls back to baseline).

    All csim pass on the fixed bench. a5065 ag r1/r2 were **invalid**: every call returned `usage_limit_reached` at
    19:09, so the flow fell back to the baseline. Moved to `dsp5065/runs/_invalid_usage_limit/` and rerun as
    `st_a5065_ag_r{1,2}b`.
  - **agentfw pilots** were stopped by the Claude session limit after about 2.1 h each:
    - base $5.77, 103 turns;
    - know $5.39, 93 turns, 1 WebSearch;
    - team $4.35, 77 turns, 0 Agent calls.

    Best correct designs:
    - **base `v4_ti171_f7`: 404,451 / 404,379 cycles**; DSP 8,552 (95%), LUT 1,136,287 (87%), FF 1,901,558 (73%),
      BRAM 2,871 (71%), URAM 256. The csim'd kernel is identical; there is no TRIPCOUNT; the flat loop is 404,331
      iterations. It uses BIND_OP (fmul fulldsp, some fadd in fabric). dfdiag: TASK_CONCURRENCY/FULL,
      compute-bound. Manual A double-buffering inside one loop is not detected by dfdiag (limitation). Caveat: two
      `DEPENDENCE false` pragmas that csim cannot verify; 95% DSP / 87% LUT is unlikely to route.
    - know `v3_27x64`: 1,092,653 cycles (8,641 DSP).
    - team: best synthesized design ~4.2 M cycles; v3/v4 csynth timed out.

    `run_agent.sh` gained `RESUME=1` (`transcript_resumeN.jsonl`); all three were resumed on login3 for 8 h.
- 2026-10-13: `jobs/ledger.py escalate` now also handles OUT_OF_MEMORY. It resubmits at the same wall-time tier with
  --mem 192G, then 512G, and counts an OOM at 512G as a failure. The memory tier carries over into later timeout
  escalations. The LUD st5 N=24/32 csynth OOMs were folded into their original tags (jobs 64870336/7, 192G); the
  duplicate `_mem192` tags are marked `merged_into`. Note: only csim and csynth are ever run (no v++/xclbin).
- 2026-10-14: **/scratch file-count quota hit (997K/1,000K files)** on the evening of 2026-10-13.
  - **Damage:**
    - 8 LUD csynth (n16_st5, n20_st4/5, n24_st3/4/5, n32_st3/4) ended "RTL creation cannot open file";
    - st_r5065_ds_r1/r3 failed at start (exit 0:53);
    - a5065 ag r1b (every check EDQUOT) and r2b (final evaluation failed), and s5065 ds r3 (synthesis failed): all
      quota-invalid;
    - the agentfw know/team sessions ended early (best results stand).
  - **Freed so far:**
    - the user removed the LUD csim projects;
    - the user archived and removed 499 c2hls temp dirs older than 2 days (archive:
      `~/c2hls_tmp_reports_20261014.tar.gz`, list: `/scratch/asa582/c2hls_tmp_cleanup_20261014.list`);
    - `.autopilot`/RTL/impl removed from finished LUD csynth projects;
    - `jobs/slim_workdir.sh` does the same for all finished `hls_synth__`/`hls_csim__` dirs and study work dirs.
      It skips cosim dirs and anything modified in the last 90 min.
  - **Prevention:** Vitis now runs in node-local `$SLURM_TMPDIR` (or `$HLS_LOCAL_ROOT`), and `jobs/keep_copy.sh`
    copies back only reports (.rpt/.xml), sources, json, logs, tcl and txt. Applied to:
    - `sbatch_flash_cc.sh`: copies to `/scratch/hpc-prf-llmfpga/asa582/projects/tmp/c2hls_kept/<TAG>/...`;
    - `lud/job_vitis.sh`;
    - `agentfw/tools/hls_eval_job.py`;
    - `jobs/run_eval_local.py` (used by `manual_df/mk.sh`);
    - `agent/hls_check.py`.

    Backups are `*.bak_20261014`.
  - **Skills packs used** (identified by content hash of each run's `skills_source.json`; `study/effort/which_pack.py`):
    - "skills" / "DSP redo" / s5065 (mm and LUD): `skills_ii_target_miss_solutions_added(90skills)_gemm_flatten_v1.json`
      (129 skills).
    - refv2 / refnf / win7080 / r5065: `..._plus_gemm_flatten_v2_refined_v2.json` (138).
    - refneu: `..._plus_gemm_flatten_v2_refined_v2_neutral.json` (136).
    - zero-shot / no-skills: a 55-entry source (not given to the model).
    - Never used: the plain `(90skills).json` (123), `_gemm_flatten_v2.json` (104), `_plus_gemm_flatten_v2.json` (134),
      and `_no_RMW_m_axi.json` (92).
  - LUD ablation (24 runs) all done, all csim pass; table pending.
  - lud skills ds r3 = baseline fallback by c2hls's timing-regression gate (est. slack −0.641 ns).
- 2026-10-14: **90-skills ablation** submitted (`s90/submit.sh`, 18 jobs, fixed benches, node-local Vitis).
  - Pack: plain `skills_ii_target_miss_solutions_added(90skills).json` (123 skills, packaged-only, no gemm_flatten).
  - mm: `s90` (flash + pack) and `s90dsp` (+ DSP floor).
  - LUD N=32: `lud_s90`.
  - Both models, r1–3.
  - To verify: each run's `skills_source.json` hash must match that pack (`effort/which_pack.py`).
- 2026-10-14 03:10 PDT: the user is away for about 8 h.
  - `ledger.py escalate` now also retries jobs that FAILED/NODE_FAIL within 2 min of starting (same tier, at most twice).
  - The escalation loop (background) runs until no ledger job is active.
  - Then: collect LUD DSE, DSP 50–65% reruns and the s90 ablation (verify the pack hash); update the story page.
- 2026-10-14 12:30 PDT: overnight results collected.
  - **s90 (plain 90-skill pack; all runs verified to load the 123-skill pack):**
    - mm Opus 23.1M / 69.3M / 86.1M (80–320 DSP); DeepSeek 158M–305M;
    - s90dsp Opus 2.85M / 1.92M / **1.05M** (8,320–8,640 DSP); DeepSeek r2 4.04M, r1/r3 running at 6 h;
    - LUD Opus 8,308 / 13,924 / 5,302; DeepSeek 40K–57K.

    All csim pass.
  - **DSP 50–65%:**
    - a5065 r1c 1,066,430, r2c 1,116,262 (5,120 DSP);
    - s5065 ds r3b: no design (stayed at 768 DSP);
    - r5065 ds r1/r3: timed out at 6 h, counted as failures.
  - **LUD DSE:** n16_st5 1,925, n20_st4 2,546, n24_st3 3,049 (wrong U), n24_st4 3,141 worst cycles. n32_st5 timed out at
    6 h (failure). n20_st5, n24_st5, n32_st3 and n32_st4 are running at the 6 h tier.
  - **agentfw:** no csim-verified + synthesized + fitting design for team (its fastest uses 10,240 DSP).
  - **Story page v11:** new generator `controlled/story/make_story_v11.py` → `story_v11.html`. Old files kept. Published
    as a new artifact: https://claude.ai/artifact/3QDoWgS4ZBitkPv3fLzz9q (republish the same scratchpad path after the
    remaining jobs finish).
- 2026-10-14: agentfw explainer page published: https://claude.ai/artifact/GYomaEtSoFbkdctUEvnjK4 (source copy: `agentfw/hls_agent_workbench.html`).
- 2026-10-14: final overnight jobs. These timed out at the 6 h tier and are counted as failures:
  - LUD n20_st5, n24_st5, n32_st3, n32_st4 (n32_st5 earlier). AutoSA has no correct design at N=32 that finishes
    csynth; its largest correct result is n24_st4 at 3,141 cycles (1,380 DSP).
  - s90dsp ds r1 and r3.

  No ledger job is active. Story page republished (v11.1, same URL).
- 2026-10-15: story page v12 (`controlled/story/make_story_v12.py` → `story_v12.html`, same URL). Adds a chart of the
  13–14 Oct fixed-bench runs (s90, s90dsp, s5065, r5065, a5065, manual_df, agentfw), the refneu rung text and csim
  (all 6 pass), and the zero-C confirmation (`verify_zeroC/`: 13/13 edited pass, 13/13 originals fail).
  **Not run yet:** the supervisor's 2026-10-13 agentic ablation (agent mode with no skills / skills / refined /
  + DSP requirement; agentic zero-shot + skills; plain zero-shot + skills for both models). It was cut off by the usage
  limit and never submitted.
- 2026-10-15: **agentic ablation submitted** (supervisor's 2026-10-13 request), `agentic_ablation/submit.sh`, runs in
  `agentic_ablation/runs/`, job list `agentic_ablation/submitted_20261015.txt`. Fixed bench, 3 reps.
  - Agent mode (Opus 5.5 + tools + real csim/csynth, 12 checks), tags `ag_<rung>_r<n>`. Rungs: zs, zsk, noskills,
    skills, dspredo, refneu, refnf, refv2, win7080. They run in 6 Slurm singleton slots (`agabl_s0-5`) and start at
    the 3 h tier.
  - Agent shim: login3:36015, 6 workers, `logs/start_shim_agent_abl_36015.sh`. Its match also covers the zero-shot
    prompt.
  - Plain mode, both models: `pl_zs_*`, `pl_zsk_*`.
  - zsk = the bare zero-shot prompt plus the Expertise rung's recorded skill block, verbatim
    (`agentic_ablation/skills_block_90gemm_v1.txt`, 195,467 chars, sha1 ae2cf639…), added by
    `flash_refined/zs_skills.py` (`--zs-skills-block`, `ZS_SKILLS_BLOCK`).
  - Shims restarted on login3: plain 38585, DeepSeek proxy 18094 (key exported from `~/.bashrc`), agent 36015.
  - `ledger.py submit` gained `--tier N` (backup `ledger.py.bak_20261015`).
- 2026-10-15 16:50 PDT: per the user, Claude jobs now run **one at a time**. The 27 `ag_*` and 6 `pl_*_cc` jobs from
  submit.sh were cancelled before starting. `agentic_ablation/serial_driver.py` runs on login3
  (pid in `driver.pid`, log `driver.log`, state `driver_state.json`). It is a plain loop, no LLM:
  - it submits one tag at a time, r1 of every rung first;
  - before each job it probes Claude through the plain shim;
  - on a usage-limit error (shim `calls.jsonl` or an agent result event) it moves the run to
    `runs/_invalid_usage_limit/`, sleeps until the named reset time + 5 min, and reruns the tag;
  - it escalates TIMEOUT 1 h → 3 h → 6 h, and reruns an OOM at 192G.

  The `pl_*_ds` jobs (DeepSeek, no Claude usage) still run in parallel. Restarting the driver resumes from its state.
- 2026-10-15: **58-kernel Refined + Res.-Req study** (user approved the per-kernel floors).
  - **Benches:** `bench_wide/` holds 58 unique c2hls benchmarks (10 MachSuite twins dropped), all with strict
    testbenches. They print `Passed!` / `Failed with N errors!`. Golden is independent and pure outputs are poisoned.
    Controls 1–3 and the hls_baseline check pass for every bench (see each `TB_NOTES.md`).
  - **Floors:** `bench_wide/FLOORS.md` and `floors_final.json`, from `op_counts.json`.
  - **Runner:** `run_flash_cc.py --refined` now reads the metadata header (backup `run_flash_cc.py.bak_20261015`).
  - **Job settings:** `bench_wide_runs/params.py` builds each job's export:
    - DSP window floor..8121, or NO_DSP_FLOOR when the floor is 0;
    - per-kernel balance gate via FLASH_REFINED_TOTAL_MACS = DSP-weighted work and DSP_PER_MAC = 1.
  - **Opus:** `bench_wide_runs/bench_driver.py` on login3 (`driver.pid`, `driver.log`), serial, reusing the ablation
    driver's usage-limit handling. It starts when the ablation driver logs "driver done". Tags `bw_<bench>_cc`.
  - **DeepSeek:** smoke runs `bwsmoke_{hlsfactory_atax,hotspot,aes}_ds`; the full `bw_<bench>_ds` set is submitted after
    they work.
- 2026-10-16: smoke v1 (DeepSeek: atax, hotspot, aes) found two setup bugs, both fixed.
  1. The capped floors (8121) gave a zero-width window, and hotspot was rejected at 7,568 DSP. The 5 capped kernels
     (heat-3d, hotspot, lavaMD, lc_mgvf, backprop) now use floor 6317, so their window is 6317–8121.
  2. The balance gate ignored data movement: atax was rejected against a 290-cycle compute floor although moving its
     116 KB takes at least 1,843 cycles. `flash_refined.balance_error` now uses
     floor = max(compute floor, FLASH_REFINED_MEM_CYCLES), and `params.py` passes bytes/64 per kernel. mm runs leave
     it unset, so their gate is unchanged. Backup: `flash_refined.py.bak_20261016`.

  smoke v1 moved to `bench_wide_runs/runs/_smoke_v1/`. bench_driver was restarted (still waiting for the ablation).
  smoke v2 (atax, hotspot) submitted.
- 2026-10-16: smoke v2.
  - atax passes: 2,426 cycles at 1,372 DSP, csim OK. The gate rejected 4,393 cycles (2.38× the memory floor) and
    accepted 2,426.
  - hotspot timed out at 1 h in csynth.
  - **Full DeepSeek set submitted:** 58 × `bw_<bench>_ds` at the 3 h tier, job list `bench_wide_runs/ds_submitted.txt`.
  - `ledger.py escalate` gained `--match REGEX`, so the driver-managed tags (`ag_*`, `pl_*`, `bw_*_cc`) are never
    escalated twice.
  - `bench_wide_runs/ds_escalate_loop.sh` runs on login3: a plain loop, every 20 min, `^bw_.*_ds$` only. It exits when
    no such job is left; log in `ds_escalate.log`.
- 2026-10-16: **LUD long csynth reruns** (user asked for 12 h → 24 h → 2 days).
  - Resubmitted the 5 that timed out at 6 h: n20_st5, n24_st5, n32_st3, n32_st4, n32_st5. The st5 jobs at N=24/32 keep
    192G.
  - `ledger.py submit` gained `--tiers a,b,c` (stored per record and used by escalate) and `--mem-override`. Backup:
    `ledger.py.bak_20261016`.
  - `lud/csynth_escalate_loop.sh` runs on login3: a plain loop, every 30 min, only those 5 tags; log in
    `lud/csynth_escalate.log`.
- 2026-10-16: **AutoSA dangling-stream repair** (new flow, the ablation partner of plain AutoSA).
  - **The bug:** for LU space_time 2, AutoSA wires the last PE's output streams (fifo_L_tmp_0_PE_N and
    fifo_L_drain_PE_N-1) into the kernel0 DATAFLOW region, but no module reads them. The writes are dead for that PE,
    but Vitis cannot prove it and stops with HLS 200-970.
  - **The fix:** `lud/autosa_fix/fix_dangling_streams.py`. Repair on error: only the streams named in HLS 200-970 are
    moved into a per-call wrapper as locals. A csim-only guard aborts with "AUTOSA_DFIX" if any of them ever receives
    data. AutoSA and `dse/` are untouched.
  - **g++ check, n12_st2 repaired:** runs clean, guard silent, **L and U both exact**.
  - **Runs:** `lud/autosa_fix/run_dfix_flow.sh` writes `lud/dse_dfix/<design>/` for every design with HLS 200-970
    (st2 at N=12, 16, 20, 24, 32). It submits csim (1 h) and csynth (3 h → 12 h → 24 h → 2 days, per-tag tiers).
  - **Escalation:** `jobs/escalate_loop.sh '^lud_dfix_'` runs on login3.
- 2026-10-16: **SODA** (user asked for the SODA stencil kernels; LLM side = Refined + Res.-Req only, Opus + DeepSeek, 1 run).
  - Repo cloned to `/scratch/hpc-prf-llmfpga/asa582/projects/workspaces/soda_src/soda` (UCLA-VAST/soda @720e4e2); sodac in
    `soda_src/venv`.
  - Research is in `soda/SODA_RESEARCH.md`. 11 kernels; single tile only (the multi-tile host is buggy); replication
    crashes; the border is undefined, so compare the interior only.
  - Study params: 2D 1024² (contrast 512²), 3D 128³; iterate jacobi2d 10, jacobi3d 6, seidel2d 9, heat3d 5, else 1.
  - **Benches:** strict benches for `bench_soda/<k>/` are being built by an agent; plain.cpp is cross-checked against
    SODA's golden.
  - **Csynth probe:** `soda/job_csynth.sh` on jacobi2d u8 q10 (does csynth report a latency for SODA's data-driven
    loops?).

## 2026-10-16 cosim speed test (for 1024^3 within Fir's 7-day cap)
- Fir hard cap 7 days (b6 28-day partitions DOWN; submit filter rejects >7 d, also on cpupreempt). Of 13 1024^3 cosims only 01 finished (1,593,018 cycles; LightningSim v2023 1,596,519, +0.22%); 8 hit the 580,000 s sim timeout, 4 still running.
- `cosim/speedtest/`: design 01 at 256^3, jobs 65388083 (baseline), 65388084 (cosim_design -O), 65388085 (-O + xelab --O3). sim.sh wall in `cosim/speed_01_n256_*/sim_wall_s.txt`. Earlier baseline: 18,522 s total wall.

## 2026-10-16 58-bench DeepSeek results
- `study/bench_wide_runs/collect.py` -> `results_ds.csv`. 55/58 done, all csim pass; 34 improved (4x to 34,000x); 21 kept the baseline (1.0x) because all 3 candidates failed the gate (44 of 63 rejections = dsp_fill).
- User decision: (a) keep the results as the gated flow produces them; no reporting of rejected candidates, no gate change for Opus.

## 2026-10-16 SODA started (floors approved)
- Cycles for SODA designs come from cosim only (csynth latency is "?" for SODA's data-driven modules); csynth gives resources. The best LLM SODA designs get cosim too, so both sides use the same cycle measure.
- SODA DSE: `study/soda/dse/gen_dse.py` (burst 512, single tile, study iterate, unroll {1,2,4,8,16} float / {8,16,32} int16, CR on/off for contrast/erosion/xcorr = 58 configs; m_axi depth + 1 MiB host padding for cosim; g++ golden check). `job_soda.sh <dir> csynth|cosim` (SODA test main as TB). Smoke cosims: blur_u32 65389575, jacobi2d_u4 65389576 (tiers 3h,12h,24h,2d).
- LLM: `study/bench_soda_runs/` (params.py, soda_driver.py). DeepSeek sd_<k>_ds x11 submitted (ds_submitted.txt), escalate loop on login1 (ds_escalate.log). Opus sd_<k>_cc: soda_driver.py on login3, waits for the 58-bench Opus driver "driver done".
- 2026-10-17 doitgen: HLSFactory hls_baseline.cpp failed csynth (SYNCHK 200-43); user approved a minimal fix (local sum_l buffer; orig kept as hls_baseline.orig.cpp; see TB_NOTES). DeepSeek bw_hlsfactory_doitgen_ds resubmitted. Opus picks up the fixed baseline automatically.
- 2026-10-17 submitted csynth+cosim for the other 56 SODA DSE configs (soda_dse_<cfg>_cosim, tiers 3h,12h,24h,2d; list in soda/dse/cosim_submitted.txt); escalation by the login3 loop (bench_soda_runs/escalate.log).
- 2026-10-17 SODA csynth-with-tripcount (report-only loop_tripcount from SODA's host formulas; add_tripcount.py -> designs_tc/): 58 csynth jobs soda_dse_tc_<cfg>_csynth (tc_submitted.txt). For csynth-vs-csynth comparison with LLM designs until cosim is in.
- 2026-10-17 SODA terminology (user): **naive baseline generated from SODA** = bench_soda/<k>/hls_baseline.cpp (c2hls starting point); **SODA target** = sodac-generated kernels (soda/dse/designs, designs_tc). Every SODA table names its measure (csynth or cosim) per column.
- 2026-10-17 SODA goal (user): start from the naive baseline generated from SODA; success = beat the SODA target (best SODA config per kernel), cosim vs cosim decides.
- 2026-10-18 LAAFD-matched SODA targets: gen_dse.py adds burst 128 / unroll 4 for every kernel (`<k>_u4_b128[_cr]`, 14) and, at LAAFD sizes (contrast 1024², 3D 256³), `<k>_u4_b128[_cr]_laafd` (5); all g++ golden PASS; csynth (tc) + cosim submitted (tc_submitted.txt, cosim_submitted.txt). The 5 non-LAAFD-size b128 dirs for contrast/3D are extra points only.
- 2026-10-18 SODA target 4× gap diagnosed (`soda/dse/diag/`, job_diag.sh keeps full log + module reports): Vitis 2023.2 drops `pipeline II=1` on SODA modules with a data-driven `for (bool enable…)` exit and no line buffer (Module0/1/4–8): unpipelined, 4 cycles/iter → blur_u4_b128 1,050,885 = 4 × 262,658. Not the cause: FIFO depth 32, blocking reads, style flp/frp, while loop, no labels, dependence pragma (all still 1,050,885 or unpipelined). Fixed trip count (measured 262,658 / 131,329 for unpack/pack) pipelines all modules; cosim streams ~1 iter/cycle but deadlocks near the end (~264.5K cycles) — patch termination not yet fixed. Decision pending (user): compare against SODA as generated or as patched.
- 2026-10-19 cosim speed test: sim.sh wall base 13,761 s, -O 13,853 s, -O + --O3 14,492 s: no gain.
- 2026-10-19 ledger.py: escalate message used TIERS[tier] (IndexError with per-tag tiers, after the resubmit was recorded); fixed to tl[tier] (backup ledger.py.bak_20261019).
- 2026-10-19 58-bench (csynth, vs bench reference): DeepSeek 56 ok, 31 >1×, geomean 18.8×; Opus 57 ok (viterbi at 6 h), 44 >1×, geomean 159.6×; head-to-head Opus faster 36 / DeepSeek 9 / tie 9. Opus 24K–32K× outliers (floyd-warshall 0 DSP, covariance, correlation, gemm, trmm, syrk) unverified.
- 2026-10-19 **SODA target = patched SODA (user decision) — reporting caveat, keep:** the patched SODA target is OUR modified SODA. sodac's loops run until the end-of-stream flag (one design, any size); Vitis 2023.2 does not pipeline them (≈4× slow). Our patch replaces each data-driven loop with a fixed trip count measured for one exact problem size (e.g. 262,658 / 131,329 for 1024² blur at 128-bit u4) plus blocking reads, so each patched design works only at that size. Any "beats SODA" claim must say "SODA hand-patched for Vitis 2023.2 (fixed trip counts)" and show the unpatched (as-generated) numbers beside it. Status: first patch (fixed counts, empty() guards kept) deadlocked in RTL — guarded empty iterations consumed the count; corrected patch (fixed counts + blocking reads, g++ golden PASS) cosim job 66013498 pending, not yet verified.
- 2026-10-19 **login3 rebooted (~12:00, load ~1,100 after)**: all login3 processes died (Opus 58-bench driver, Opus SODA driver, SODA + LUD escalation loops, plain Claude shim 38585, DeepSeek proxy 18094). Restarted on **login2**: escalation loops (regex now also lsim_proj_/soda_diag_), shim 38585 (`logs/start_shim_38585_login2.sh`), ds_proxy 18094, bench_driver, soda_driver. URLs moved to login2: shim_url.txt, controlled/ds_proxy_url.txt, agentic_ablation/serial_driver.py PLAIN_SHIM (backups *.bak_20261020). Catch-up escalation pass resubmitted ~30 SODA as-generated cosims at 2 d. bench_driver resubmitted bw_viterbi_cc as try 3 (6 h, job 66038607).
- 2026-10-19 **bench_soda sizes = LAAFD** (user): contrast 1024² (was 512²), denoise3d/jacobi3d/heat3d 256³ (was 128³). Old bench kept in `bench_soda_v1_20261016sizes/`. gen_bench.py edited (backup .bak_20261019). New `bench_soda/check_bench.py` reruns the controls (plain pass, early-return fail, one-bug fail, hls_baseline pass): all 4 OK, written to TB_NOTES. floors_soda.json: cells/mem_cycles/balance_work scaled; floors unchanged (ratio). DeepSeek reruns sd_{contrast,denoise3d,jacobi3d,heat3d}_ds (66038585–88, 6 h→12 h→24 h; 3D at 192G). Opus SODA queue (soda_driver) starts after bench_driver's "driver done" and uses the new bench. The 512²/128³ DeepSeek runs and SODA target dirs stay as old-size data only.
- 2026-10-19 LightningSim on SODA targets: `study/soda/lsim/job_proj.sh` (csynth → proj.tar.gz with SODA host as TB, argv in hls.app <Simulation>), 81 jobs lsim_proj_* (77 configs + 4 blur diag variants), projects in /scratch/hpc-prf-llmfpga/asa582/projects/tmp/c2hls/lsim/soda/; LightningSim (v2023) batch 66030863 on the first 15.
- 2026-10-20 **patched SODA target verified on blur_u4_b128**: fixed counts + blocking reads (diag/v_fixblock_cosim) cosim **350,460 cycles, golden PASS** (as generated 1,050,885; LAAFD ideal 264,192). Remaining 33% gap above ideal unexplained (all modules II=1, bursts inferred) — check with LightningSim.
- 2026-10-20 `soda/dse/patch_soda.py` (measured per-function trip counts from an instrumented g++ run at the config size; rejects if instances disagree; golden check) → `designs_patched/`. 75/77 PASS (identical to the verified blur variant); contrast_u16{,_cr} rerun in job 66100567 (3 h limit hit). 72 patched cosims submitted, tags `soda_dse_pat_<cfg>_cosim` (int kernels 3h→12h→24h; float 2d→7d), list `patched_cosim_submitted.txt`. Not submitted: the 3 float 256³ `_laafd` (would exceed 7 days).
- 2026-10-20 as-generated float cosims: not hung, 13–100K cycles/h (DSP48E1 unisim models + OPMODE warning flood, 0.5–0.9 GB logs); most need 55–100 h, 256³ ones 7–27 days. Ledger: the 3 as-generated 256³ float cosims (65861616/7/8) marked failed (given up); scancel denied to Claude — user to cancel.
- 2026-10-20 LightningSim SODA batches: 66030863 (15 projects), 66100654 (65 projects); 80/81 projects built (contrast_u16_cr OOM → 192G retry).
- 2026-10-20 agentfw team: post-hoc csim of its synthesized fitting kernels all pass (exp_post/): best v3_ti4 **4,195,795 cycles, 1,024 DSP** (v2_ti4_kc16 4,197,493; v1_ti4 4,260,001). csynth of its csim-passing v2/v3/v4_ti36/v4_ti40 pending (12 h→24 h→2 d). These are post-session evaluations of unchanged agent kernels; label them so. Team made 0 Agent (subagent) calls.
- 2026-10-20 patched SODA cosims, 24 done (all golden PASS): blur/sobel2d 3.0× faster than as generated (cycles = 1.34× module iterations, e.g. blur_u32 44,028); erosion/xcorr **slower** when patched (no CR: 6.7–7.0× iterations, e.g. erosion_u16 444,780 vs 333,765 as generated; CR: 2.2–2.4×, ≈ as generated for erosion, 2.5–2.7× faster for xcorr). Cause unknown. The width-32 erosion/xcorr cosims (patched and as generated) fail with an xelab SIGSEGV (snapshot not created) — retry with `ulimit -s unlimited`.
- 2026-10-20 DeepSeek sd_contrast_ds at 1024²: fell back to the baseline again (2,282,749,953 cycles, 5 DSP).
- 2026-10-20 **LightningSim on SODA**: first runs failed (Vitis rewrote the TB's -I../.. to -I../../../../.; fixed in all 80 projects and job_proj.sh). Rerun (66131309): exact match on the two blocking-read blur designs — diag_v_blockall 1,050,885 vs cosim 1,050,885, diag_v_fixblock (patched) 350,460 vs 350,460 (mapped, n=3 s_axilite). The 21 as-generated designs fail to link: LightningSim does not support non-blocking FIFO reads (`fifo.empty()` → `_autotb_FifoCanRead`). So LightningSim covers the patched SODA target but not SODA as generated. Submitted 75 project builds `lsim_proj_pat_<cfg>` and a waiter that runs LightningSim on them and writes `soda/lsim/lsim_vs_cosim.csv`.
- 2026-10-20 **LightningSim on SODA**: designs with SODA's non-blocking `fifo.empty()` cannot be simulated (link error `_autotb_FifoCanRead_i513`, NbReadReq unsupported). Blocking-read designs work and match cosim exactly: diag_v_blockall 1,050,885 = cosim, diag_v_fixblock (patched blur_u4_b128) 350,460 = cosim (0.00%). Fix needed first: Vitis stores the TB `-I../..` as `-I../../../../.` in hls.app; all 80 tarballs rewritten, job_proj.sh fixed. Submitted lsim_proj_pat_<cfg> for 75 patched designs (→ /scratch/hpc-prf-llmfpga/asa582/projects/tmp/c2hls/lsim/soda/pat_<cfg>); a waiter submits LightningSim on them when built. compare_soda.py maps pat_ → designs_patched/.
- 2026-10-20 LUD figures on the story page (make_story_v13.py): AutoSA LU design space (30 points, correct / wrong U / csynth error / pending) and the LUD ablation at N=32; lud/check/dse_table.json regenerated (n20_st5 2,456; n24_st5 3,044; n32_st3 4,305 wrong U; n32_st4 4,395 correct; n32_st5 pending).
- 2026-10-20 **Supervisor's question: HLSmith's agentic flow vs Claude Code's own agent mode** (skills and resource requirement accepted). `study/exp_hlsmith/driver.py` on login2 (serial Opus; reuses agentic_ablation/serial_driver.py; adds logs/shim_20261020_login2 to the usage-limit scan):
  - **C (first):** agentfw variant `hlsm` (new_task.py, backup .bak_20261020): free Claude Code agent, base tools, the same 138-skill refined v2 pack as ag_win7080 (knowledge_hlsm/), DSP window 6,317–7,219 as a hard requirement, same seed (hls_baseline.cpp) and fixed bench, 10 h cap; runs agentfw/runs/mm1024_hlsm_r1..3; compare with ag_win7080_r1..3 (591K/836K/792K). r1 started 22:15.
  - **A:** plain HLSmith on the fixed bench, pl_<rung>_cc_r1..3 for noskills…win7080 (pairs with ag_*); DeepSeek pl_<rung>_ds_r* submitted directly (21 jobs, ds_submitted.txt).
  - **B:** agentic HLSmith on the 8 benches where plain Opus stayed at 1.0× (agb_<bench>_r1; machsuite_backprop excluded: support files); per-bench agent shim on login2:36016 started by the driver. claude_code_shim.py and agent/hls_check.{py,sh.template} gained --agent-header / AGENT_HEADER (default kernel.h; backups .bak_20261020).
  - Opus SODA queue stays parked until this driver is done (keep Claude jobs serial).
- 2026-10-20 **Experiment D**: Claude Code's free agent (agentfw `hlsm`: refined v2 pack + DSP window 6,317–7,219, same seed/bench, 10 h) driven by **DeepSeek v4-flash**, vs plain HLSmith DeepSeek (exp A pl_<rung>_ds). Runs agentfw/runs/mm1024_hlsm_ds_r1..3, launched in parallel on login2 22:2x with `agentfw/run_agent_ds.sh` (new file): fresh CLAUDE_CONFIG_DIR (no Anthropic login), ANTHROPIC_BASE_URL → `study/exp_hlsmith/ds_anthropic_proxy.py` on login2:18095 → api.deepseek.com/anthropic; the proxy adds the key from the environment and **refuses any request containing the user's email** (log ds_anthropic_calls.jsonl, sizes/usage only; 1 blocked = the guard test). Claude Code's total_cost_usd for these sessions uses Anthropic prices — ignore it; use proxy token counts. Restart the proxy if login2 reboots (the sessions die with it).
- 2026-10-20 cost table: log-miner extraction of per-run calls/tokens/$/wall/Vitis checks for the ablation, controlled ladder, agentfw and LUD runs (in progress).
- 2026-10-20 **Matched budget for C/D** (side-agent warning; user: keep budgets almost equal). Measured agentic HLSmith per run (ag_win7080): 1 agent session, 2–4 of its 12 allowed checks, 0.2–0.3 h agent time, $1.3–1.7, 3 h job cap (refv2/refnf: ≤16 checks, ≤6 h). Budget mode in agentfw/tools/hls (task.json "budget"; backup tools/hls.bak_20261020): ≤12 submissions, one job at a time, 1 h each, no escalation; run_agent(_ds).sh take the session cap from the task (3 h). Task `agentfw/tasks/mm1024_b12`. C r2/r3 workspaces pre-created budgeted (the driver uses them); C r1 (started 22:15, 10 h, unbounded) and D runs mm1024_hlsm_ds_r1..3 (unbounded) continue as the unbounded-budget variant. Budgeted D: mm1024_hlsm_b12_ds_r1..3 launched on login2. A budgeted C r1 (mm1024_hlsm_r4) still to run after the driver.
- 2026-10-21 **C/D first results** (score = fastest csim-verified, synthesized design with DSP 6,317–7,219 and every resource <100%, from exp/ records):
  - C r1 (Opus free agent + HLSmith pack + window, unbounded; 21 submits, ~1.5 h, exit 127 but REPORT.md written): **425,879 cycles, 7,160 DSP, LUT 93.4%** (fadd in fabric, fmul meddsp; genuine fp32). C r2 (matched budget, running): 562,781, 7,193 DSP, LUT 92%, 9/12 submits so far. C r3 not started. Agentic HLSmith window: 590,653 / 835,709 / 792,060 at LUT 65% / 35% / 46%, $1.3–1.7, 16–21 min.
  - D (DeepSeek free agent): unbounded r1, r2 no in-window verified design; **r3 "292,030 cycles" is INVALID** — kernel quantizes A, B to unsigned 18-bit fixed point (assumes inputs in [0,1), as the fixed bench's rand()/RAND_MAX gives) and passes the 0.001 tolerance; not fp32. Matched budget: r1 none (3 h timeout), r2 none, r3 948,196 cycles, 6,820 DSP, LUT 37%.
  - Scan of all HLSmith/plain/agentic final kernels for fixed-point tricks: none (matches are 512-bit packing reinterpret casts). **The fixed bench must be hardened** (signed, wide-range, hidden inputs) before final numbers.
  - Exp A DeepSeek plain: most rungs done (e.g. refnf 1.25M/1.29M at 5,120 DSP; dspredo 2.08M/4.90M); win7080 ds still running.
- 2026-10-21 **Hardened bench** `study/bench_hard/n1024/autosa_mm` (agents never saw it): two input sets per run (signed [-4,4); wide range ±2^±10 with zeros), new seed, double golden, tolerance 1e-4·Σ|ab|+1e-6, NaN check, poisoned C, kernel called twice. `verify_hard.py` (g++ -O2, Vitis headers) → verify_hard_b.csv (240 kernels): every HLSmith final passes (agentic_ablation 39/39, dsp5065 13/13, s90 10/10, exp_hlsmith 17/17, manual_df 3/3, verify_zeroC 13/13); runs_controlled 30 fail as expected (old C += spec). Free agents: all headline designs pass; failures are probe kernels, DeepSeek ds_r3's fixed-point variants (11), and hlsm_r2/v5 (never csim'd; its twin v6 passes).
- 2026-10-21 **C final (LUT ≤ 80% applied post hoc to both sides; neither was told)**: free agent r1 unbounded 532,752 (73% LUT; uncapped 425,879 @ 93%); matched r2 808,541 (54%; uncapped 538,205 @ 97%) $6.04, 12 checks; matched r3 496,924 (77%; uncapped 480,684 @ 80.3%) $4.96, 10 checks. Agentic HLSmith window 590,653 (65%) / 835,709 (35%) / 792,060 (46%), $1.3–1.7, 2–4 checks. ⇒ parity on quality under equal caps; HLSmith ~3.5× cheaper, ≤⅓ the Vitis checks, lower logic use; free agent's uncapped lead comes from 80–97% LUT.
- 2026-10-22 **login2 rebooted 2026-10-21 17:56**: the exp driver, plain shim 38585 and DS proxy 18095 died (C and D had finished; D proxy not needed). Restarted on login2 01:47: shim 38585 (`logs/start_shim_38585_login2.sh`), `exp_hlsmith/driver.py`, which now adopts a tag's still-queued job after a restart instead of resubmitting (adopt(); backup driver.py.bak_20261022); it adopted pl_refv2_cc_r2 job 66683742. 17 of 32 tags left (A r2 rest, A r3, B). Cost section + C rows in `controlled/story/make_story_v14.py` (exp_hlsmith/cost_table/exp_cd.json), published as a new artifact. C r4 (budgeted, runs/mm1024_hlsm_r4 from tasks/mm1024_b12) added to the driver queue after pl_refv2_cc_r2; driver restarted 01:56 (pid 93132), 18 of 33 tags left.

## 2026-10-22 — exit-127 sessions checked; launcher fixed

- C r1 (`mm1024_hlsm_r1`) and D unbounded r1–r3 (`mm1024_hlsm_ds_r1..r3`) ended with exit 127 and an empty `transcript.jsonl`.
  Cause: `run_agent.sh` / `run_agent_ds.sh` were edited (budget-hours line) at 2026-10-20 22:27:56 PDT while those four
  sessions ran. Bash had already started `claude` with the intended flags; after it exited, bash read the edited file at
  the old byte offset, ran a fragment of the `--model` line (127), and that fragment's `> $OUTF` truncated the transcript.
- Checked from Claude Code's own session logs (now copied into each run as `transcript_native.jsonl`, with `LAUNCHER_NOTE.txt`):
  tools used were only Read/Write/Edit/Glob/Grep/Bash; no WebSearch/WebFetch/Agent/MCP; one model throughout
  (claude-opus-5-5, or deepseek-v4-flash); no permission denials. The sessions ran under their limits; results stand.
- Each session had its own auto-memory directory (per workspace path); none read another run's memory.
- Fix: both launchers' bodies are now one `{ ...; exit; }` block, so bash parses the whole script before running it
  (backups `.bak_20261022`). Tested by editing a wrapped script mid-run. C r4 (not started yet) uses the fixed launcher.

## 2026-10-22 — 80% LUT cap on every number, with one fallback rule

- `make_story_v11.LUT_CAP` (0.8 × 1,303,680) now applies in the shared readers (v11 cell/agentfw_runs, v12 mm_cell kind "cap",
  v13 ablation, v14 cost table); the canvas header states it once. Backups `make_story_v1[1-4].py.bak_20261022c`.
- Fallback (`make_story_v13.fallbacks`): an over-cap run uses its best design under the cap that the run csim'd on the fixed
  bench and synthesized byte-identically: controller history plus, with an agent backend, the session's own checks
  (`logs/<shim>/call_*/workspace/checks`). Same rule as the free agents' in-session designs.
- Affected: agent backend Expertise + DSP floor r1 (656,422 at 82.9% LUT): no fallback (its 721,966 / 76.7% LUT check-3
  kernel was never csim'd), so the rung's best is 868,033 and the gain over plain is 1.1×. Free agent base r1: 404,451 at
  87% LUT → 468,427 (v3b_ti147_f4, 70% LUT). know r1 658,279 is at 79.6% (inside). Experiment C rows were already capped.
- Survey synced (backup `agentic_hls_survey.html.bak_20261022b`): 468K, 534K–868K, gains 1.1× / 1.3–2.2×, rule stated.

## 2026-10-22 — post-hoc csim, both sides (`study/posthoc_csim/`)

- `enumerate.py` lists every design synthesized in its session, never csim'd there, under the LUT cap and DSP total, and
  faster than its run's counted best, on both sides (agentfw sessions; agent-backend sessions' own checks, worst case read
  from each check's csynth.xml, since the check .txt gives best case). It also checked symmetry: no csim-verified in-session
  agent-backend design beats its run's final.
- Candidates: agent backend Expertise + DSP floor r1 checks 2/3/4; free-agent base r1 probe_ops/probe_widen. (team's were
  csim'd post hoc on 2026-10-20 in exp_post/.) `submit.py`: Vitis csim on the fixed bench (ws/ = base task files; same
  testbench.cpp and kernel.h md5 as the agent sessions). `hard.py`: hardened bench.
- Results: dspredo r1 check 3, **721,966 / 8,704 DSP / 76.7% LUT, passes csim and hardened** → that run's fallback; rung best
  721,966; gain over plain 1.4×. Checks 2, 4 pass. Base probes fail (not mm designs).
- team: exp_post v3_csynth (csim'd in session, synthesized post hoc) = **527,571 / 8,192 DSP / 75.5% LUT**, passes hardened →
  team's post-session best (was 4,195,795). v4_ti36/v4_ti40 post-hoc csynths (66116541/2) still pending.
- Pages rebuilt and republished (canvas v6, survey v6).
- Other half of the post-hoc rule, both sides: `enumerate_csynth.py` → 19 kernels csim-passed in session but never
  synthesized (13 Claude Code on its own: know ×4, team v3_kc16, DeepSeek C/D runs ×8; 6 agent backend: dspredo r1 1_csim,
  dspredo r2 3_csim, refnf r1 1_csim/8_csim, zsk r1 7_csim, zsk r2 3_csim). `submit_csynth.py` submitted jobs
  66762762–66762780 (12 h, `posthoc_csim/csynth/<run>/`); team v4_ti36/v4_ti40 (66116541/2) pending since 10-20.
  When done: count a result only where it fits the cap (and the DSP window for C/D), passes the hardened bench, and beats the
  run's counted best; then rebuild both pages.
- Rule (user, 2026-10-22: "count it on both sides"): each run counts its best verified design — csim'd on the fixed bench,
  byte-identical kernel synthesized, LUT ≤ 80%, rung DSP gate (floor 90% / window 6,317–7,219 / C–D window), passes the
  hardened bench — from the session or checked post hoc. Code: `v11.session_best` (Claude Code), `v13.fallbacks`/`run_best`
  (HLSmith; source final/fallback/better). Matched-budget C/D rows: in-session only. Cost-table free rows now computed
  (FREE/exp_cd numbers no longer typed in). Pages rebuilt; numbers unchanged until the 19 csynths land.
- When the csynths finish: `sbatch study/posthoc_csim/job_hard.sh` (hard.py now includes csynth/), then
  `python3 controlled/story/make_story_v14.py`, check the survey numbers, publish both.
- 2026-10-22 **Otus move prepared (nothing changed on Fir):** `study/OTUS_MIGRATION.md` (full plan), `study/posthoc_csim/OTUS_RUNBOOK.md`. SODA: `soda/dse/otus_soda_resubmit.txt` (the 85 cosims pending on Fir: 37 designs/ + 48 designs_patched/, 2 d, 8 cpus, 64G), `soda/dse/job_soda_otus.sh` (job_soda.sh with hpc-prf-llmfpga/normal, native Vitis 2023.2, node-local fallback), `soda/dse/resubmit_soda_otus.sh [--dry-run]`. Driver: `exp_hlsmith/otus_reset_state.py [--apply]` drops the Fir job id of unfinished tags (now only pl_refv2_cc_r2 / 66683742, tier 2, tries 3) and reopens tags marked CANCELLED; run it on Otus after driver.py has stopped.
- 2026-10-22 **Otus move, lsim + LUD + LightningSim fork:**
  - `cosim/lsim/otus/` holds:
    - the fork as 3 `format-patch` files plus a git bundle (branch vitis2023-axi on 1abd9a6 v0.2.6; tree da0dae7f);
    - exact micromamba package lists for `env` and `rustenv`;
    - `setup_lsim_otus.sh`, which rebuilds the fork and checks the tree hash;
    - `run_lsim_otus.sh` and `sbatch_lsim_otus.sh`, which run natively without the container;
    - `names_lsim_soda2.txt` (65 names, 19 finished on Fir) and `names_lsim_soda_pat.txt` (75);
    - `resubmit_lsim_otus.sh [--dry-run|--check]`, which covers lsim_soda2, lsim_soda_pat and lsim_proj for contrast_u16_cr.
  - `study/soda/lsim/job_proj_otus.sh`.
  - LUD: `study/lud/job_vitis_otus.sh` and `study/lud/resubmit_lud_otus.sh` (n32_st5 csynth).
  - The packed projects in `/scratch/hpc-prf-llmfpga/asa582/projects/tmp/c2hls/lsim/soda` (3.8 GB) were added to the rsync list.
  - All 3 resubmit scripts refuse a real submission when `CC_CLUSTER=fir`. The dry runs on Fir were checked.
- 2026-10-22 **Otus move, xclbin + sd_heat3d_ds:**
  - `xclbin/v4_systolic_st0_c1/sbatch_build_otus.sh` uses native Vitis and XRT. It searches for the U280 platform or takes `OTUS_U280_XPFM`, and stops early if it finds none. `resubmit_xclbin_otus.sh` submits it with Fir job 64165394's settings: 32 cpus, 448G, 2 days.
  - `otus/` (at the c2hls_claude_code root) holds:
    - `requirements_c2hls_venv.txt`: exact Fir venv versions, no torch, torchvision or Pillow_SIMD; the c2hls/hls_eval imports don't need them;
    - `setup_venv_otus.sh`;
    - `sbatch_flash_cc_otus.sh`: the same `run_flash_cc.py` command line, dse_v3 env inlined, native Vitis, include dir from `$XILINX_HLS`; it can replace sbatch_flash_cc.sh;
    - `start_ds_proxy_otus.sh` (controlled/ds_proxy.py, 18094);
    - `resubmit_sd_heat3d_otus.sh`: the `--export` matches Fir job 66475350 exactly apart from the include dir.
  - Found: the 18094 DeepSeek proxy isn't running on Fir login2, so 66475350's LLM calls would fail if it started on Fir.
- 2026-10-22 **Otus move, post-hoc csynths:**
  - `posthoc_csim/otus_csynth_resubmit.txt` is built from the Fir queue: 19 posthoc_csynth + 2 afw_post, each with its dir, its task.json, 12 h, 8 cpus, 64G and its Fir job id. The team pair uses `agentfw/runs/mm1024_team_r1/task.json`; it has the same content and the same kernel.h md5 as `posthoc_csim/ws`.
  - `agentfw/tools/hls_eval_job_otus.py` finds the judge and keep_copy.sh from its own location, with a node-local fallback. Otherwise it's identical to hls_eval_job.py.
  - `posthoc_csim/job_csynth_otus.sh` and `posthoc_csim/resubmit_csynth_otus.sh [--dry-run|--check|--compare]`. `--check` runs a csynth of team v3 into `otus_check_v3/`; `--compare` checks latency (best and worst), interval, DSP, LUT, FF, BRAM and URAM against Fir's `exp_post/v3_csynth`. The compare logic was tested on Fir with a copy of the reference.
  - OTUS_RUNBOOK.md is marked superseded for the full move.
- 2026-10-22 **Study git repo:** `git init -b main` in c2hls_claude_code with a code-only `.gitignore`. The first commit is 3bf6d39: 3,131 files, 2.2 MB packed. Ignored: logs/runs/ablation results, Vitis trees, reports and logs, generated SODA/LUD designs, transcripts and configs, and the clean AutoSA clone (its commit is in `study/lud/upstream/AutoSA.commit`). Two files with personal data stay out of git: `ds_anthropic_proxy.py` (its guard lists the user's addresses) and the fork `.bundle` (author email). The fork patches now carry a noreply author and still rebuild tree da0dae7f (tested in a temporary worktree). The repo-local identity is `ahmadsb101 <ahmadsb101@users.noreply.github.com>`. Staged blobs were scanned for keys and emails: clean. No remote yet; the user pushes.
- 2026-10-22 **Migration plan fix (side-agent note):** step 5 copied results with `--exclude '*_work/'`, which would have dropped every agent-backend session check: 212 `checks/N_*_work/kernel.cpp` and 104 `N_*_work/hls_proj/*/syn/report/csynth.xml`. make_story_v13 `fallbacks()` and posthoc_csim/enumerate*.py read those. The rsync now excludes only Vitis internals (.autopilot, verilog, vhdl, systemc, impl, csim/build, *.o, *.vcd, *.wdb). Dry run on Fir: 107,727 files and 5.47 GB, against 97,961 and 4.29 GB before; the sample check folder keeps kernel.cpp and syn/report/csynth.xml.
- 2026-10-22 **Otus move, last gaps:**
  - **Agent checks** were still on Fir's container and account.
    - `study/agentfw/tools/hls_otus`: the free-agent tool, for C r4 and C/D. Tested in a throwaway copy of the r4 workspace with a fake sbatch: it writes an Otus job (account, partition, native Vitis, the runner from bin/).
    - `agent/hls_check_otus.sh.template` and `agent/hls_check_otus.py`: HLSmith with an agent backend, for experiment B. The template was rendered as the shim does, and the outer and inner shells parse.
    - `hls_eval_job_otus.py` now finds study/ by walking up to `jobs/keep_copy.sh`, so it works from a workspace's bin/.
  - **Flash job:** the driver's `sbatch_flash_cc.sh` is replaced on Otus by `otus/sbatch_flash_cc_otus.sh`.
  - **Hardened bench:** `verify_hard.py` reads `/scratch/hpc-prf-llmfpga/asa582/projects/tmp/c2hls/vitis_hls_2023.2_include`, which is now in the step-5 rsync, with a symlink fallback to `$XILINX_HLS/include`.
  - **Installer:** `otus/apply_otus_site.sh [--dry-run]` installs everything above and also:
    - copies the tools into the unfinished workspaces (mm1024_hlsm_r4, smoke_base);
    - sets the shim's `--agent-account` default;
    - rewrites the claude binary path;
    - restores the judge skill snapshot with an md5 check.
    It refuses to run on Fir; the dry run on Fir was checked.
  - **Judge skill files:** not committed, at the user's choice. The snapshot is in `study/judge_skills_fir_20261022/` (3 JSON files, README with md5s and diff sizes, MD5SUMS). The judge repo is unchanged. OTUS_MIGRATION step 2 was rewritten to match.
- 2026-10-22 **Otus setup in one script** (`otus/otus_setup.sh <stage>`: check, clone, home, copy, site, venv, lsim, verify, compare, dryrun; OTUS_MIGRATION.md has the table).
  - Combined layout (study inside the judge clone, `$W/c2hls_claude_code` a link): `apply_otus_site.sh`, `setup_venv_otus.sh`, `resubmit_lsim_otus.sh`, `sbatch_flash_cc_otus.sh` and `job_csynth_otus.sh` took `W`/`B` one level too deep through `readlink -f`/`$SLURM_SUBMIT_DIR`; fixed (if the parent holds `hls_eval.py`, go up one more).
  - `site` tested on a copy of the combined layout on Fir: 992 files changed and committed, no Fir path or `login2` host left, skill md5s OK. Bugs found by the test and fixed: `apply_otus_site.sh` exited on SIGPIPE (`git status | head`) under pipefail; the two path seds could rewrite each other (placeholders now); `logs/start_*.sh` were skipped; the plan's `s|login2|…|` also renamed `logs/shim_20261020_login2/` references (now the bare host word only, `*.md` left as written).
  - Plan step 1 now saves the pending-job arguments before the `scancel` (the step-7 `scontrol` with a comma list fails; one job at a time).
- 2026-10-22 **`dse_v4_claude` Otus copies** (that folder, not in git; the step-5 copy carries them): `env_otus.sh` (paths from the folder's location; `dse_v4_source_vitis` sets `C2HLS_VITIS_SETTINGS` and unsets the container variables), `sbatch_csynth_otus.sh`, `sbatch_csim_otus.sh` (hpc-prf-llmfpga/normal, 8 cpus, 64G, 12 h / 24 h as on Fir; refuse on Fir or outside the folder). Tested in an Otus-like copy on Fir: paths right, `check_structural.py` passes. Its `CLAUDE.md` has an "On Otus" section.
- 2026-10-22 17:40 **Fir moved on**: `pl_refv2_cc_r2` (66683742) COMPLETED in 1 h at 17:15; the driver started **C r4** (`afw_hlsm_r4`, 3 h budget, login2) at 17:17, and it submitted `afw_csim` 66851704. 113 jobs queued.
  - Freeze plan (OTUS_MIGRATION step 1) changed: stop the driver only (C r4's `run_agent.sh` keeps running: `pkill -f "python3 driver.py"` signals PID 93132 alone, checked); wait for C r4's `exit_code`; then save the queue, stop the services, `scancel -u asa582`.
  - `otus_reset_state.py` now handles agent tags: `exit_code` present → final `exit N` (tested on a copy); missing → a warning, because the Otus driver would resume the session and mix the two clusters.
- 2026-10-22 **Globus instead of rsync** (Otus needs a VPN; Fir has none): `otus/pack_for_otus.sh` (Fir: freeze checks, then tarballs + MANIFEST.tsv + SHA256SUMS in `/scratch/hpc-prf-llmfpga/asa582/projects/tmp/c2hls/otus_transfer/`) and `otus_setup.sh unpack` (Otus: checksums, then unpack per manifest from `$B/transfer_from_fir`). Tested end to end on small trees. The clone now uses HTTPS (public repo, no GitHub key on Otus); `check` also tests GitHub and only warns when there's no ssh to Fir.
- 2026-10-23 **On Otus (login5, user haqc2, shared with Philip: dotfiles left as they are)**. Fir is not frozen: all Fir jobs keep running as a contingency; the pack ran live (`LIVE=1`, `PACKED_AT` 2026-10-23T03:25:49Z); Fir's driver is stopped. C r4 (`afw_hlsm_r4`) ended on Fir at 20:08 PDT with exit 0 and no result (its v1 csim 66851704 never left the Fir queue); how to count or rerun it is open.
  - Done: `check`, `clone`, `home`, `venv`, `lsim` (fork tree da0dae7f; conda cache pinned to the install dir), Globus + `unpack` (checksums OK, 115,878 study files), `site` (a20b1b3, local only: Otus paths and Fir's judge skill files, not to be pushed). Compute nodes reach login5 ports.
  - `dryrun`: shim URL `login5:18094`, no `login2` host left; driver state previews `afw_hlsm_r4` final = exit 0. `site` had missed the claude path in `agentfw/run_agent{,_ds}.sh` (fixed, e8bedc1). Fir paths left on purpose: Fir-only scripts (`cosim/lsim/{build,run}_lsim.sh`, `sbatch_fit.sh`, `ablation/run_ablation.py`), records and logs.
- 2026-10-23 **Otus: Vitis needed libtinfo.so.5** on compute nodes (first verify 3811767 failed in 11 s: `libxv_commontasks.so: libtinfo.so.5`). PC2 ships it in `/opt/software/FPGA/Xilinx/vitis_vivado_libs` (its `xilinx/vitis/23.2` module adds it). New `otus/vitis_settings_otus.sh` sources settings64.sh and adds that folder; all 13 Otus Vitis scripts and `dse_v4_claude/env_otus.sh` use it (4975197). Verify resubmitted: 3811943.
  - User decisions: `sd_heat3d_ds` runs on Otus (user cancels it on Fir); **C r4 reruns on Otus**. Fir's C r4 workspace kept as `agentfw/runs/mm1024_hlsm_r4_fir`; a fresh `mm1024_hlsm_r4` made with `new_task.py tasks/mm1024_b12 … hlsm` (same task.json and CLAUDE.md as Fir's, budget 12 checks / 3 h, Otus tools). The driver starts it fresh (no transcript) after compare passes.
- 2026-10-23 12:30 **Otus step 7 started** (user: go on without waiting for the verify compare; verify csynth 3811943 still running, past compilation, so the libtinfo fix works).
  - Services on login5 (Python 3.11: `module load lang; module load Python/3.11.5-GCCcore-13.2.0`): plain shim on **38585** (`PORT=38585 bash start_shim.sh`; new `PORT=` override, because `serial_driver.py` probes `login5:38585`; the first try on a random port left the driver waiting 10 min), DeepSeek proxy `login5:18094` (`otus/start_ds_proxy_otus.sh`), `driver.py` (pid 3992046). **C r4 restarted fresh on Otus at 12:41** (`afw_hlsm_r4`, `claude -p`, 3 h budget, workspace `agentfw/runs/mm1024_hlsm_r4`; Fir's is `mm1024_hlsm_r4_fir`).
  - Submitted: `sd_heat3d_ds` 3813693 (cancelled on Fir by the user), `lud_csynth` 3813694, xclbin 3813695, 37 `soda_cosim` + 48 `soda_pat_cosim` (…3813780), 19 `posthoc_csynth` + 2 `afw_post` (…3813801), `lsim_soda2` 3813803 (46 projects), `lsim_soda_pat` 3813804 (30), `lsim_proj` 3813805.
  - These went in before the verify compare: if team v3's Otus csynth (3811943) differs from Fir in any field, the csynth/SODA/LUD results from Otus are not comparable with Fir's and get flagged.
  - LightningSim check: `resubmit_lsim_otus.sh --check` exited silently (`head -1` SIGPIPE under pipefail; fixed with `sed -n 1p`). Its pick `diag_blur_u4_b128_fifo32` has `end: None` on Fir, so that check (3813807) can't compare anything (can be cancelled). Real check: 3813809 on `diag_v_fixblock_cosim`, Fir end **350408**; Otus writes `lsim_otuscheck.json` in `/scratch/hpc-prf-llmfpga/asa582/projects/tmp/c2hls/lsim/soda/diag_v_fixblock_cosim/`.
- 2026-10-23 14:30 **Otus: two library bugs found after step 7, fixed, affected jobs rerun.**
  - **LightningSim** (`lsim_soda2` 3813803, `lsim_soda_pat` 3813804 and check 3813809 "COMPLETED" in minutes with `end: null`): LightningSim pops `LD_LIBRARY_PATH` and links the testbench with the conda env's g++, so at run time it got Otus' old `/lib64/libstdc++.so.6` (no `GLIBCXX_3.4.30`) and exited at load ("kernel did not run. Did the testbench call it?"; Fir's container had a new enough libstdc++). Fix in `cosim/lsim/otus/run_lsim_otus.sh`: `LDFLAGS=-Wl,-rpath,$ENV/lib` (Vitis' `Makefile.sysc.rules` appends to `LDFLAGS`). **Check: `diag_v_fixblock_cosim` gives 350408 cycles on Otus = Fir.** The 76 failed result files were moved to `/scratch/hpc-prf-llmfpga/asa582/projects/tmp/c2hls/lsim/otus_failed_libstdcxx_20261023/`; rerun as `lsim_soda2` 3814216 and `lsim_soda_pat` 3814217. A duplicate `lsim_proj` (3814218) the resubmit queued next to the running 3813805 was cancelled. 3813807 was a useless check (its project has `end: null` on Fir).
  - **Vitis testbenches (cosim/csim)**: all 84 finished SODA cosims failed with `C TB simulation failed, nonzero return value '1'`: `cosim.tv.exe` needed `GLIBCXX_3.4.30`. Cause: `module load lang Python/3.11.5-GCCcore-13.2.0` puts GCC 13's lib64 on `LD_LIBRARY_PATH`/`LIBRARY_PATH`; Vitis' g++ 8.3 linked against it and ran with Vitis' own older libstdc++. Fix in `otus/vitis_settings_otus.sh` (every Otus Vitis job sources it, the judge too: `source <wrapper> && cmd`): drop `/EB-SW/software/GCCcore/` dirs from both paths. Test: contrast_u1 cosim passed the C TB and ran xsim. The 85 SODA `cosim/` dirs moved to `/scratch/hpc-prf-llmfpga/asa582/projects/tmp/c2hls/soda_otus_failed_gcccore_20261023/` (3813744, still running in the bad env, cancelled); all 85 resubmitted (`otus_soda_submitted.txt`, second block, …3815629).
  - **Not affected**: csynth only (posthoc csynths, LUD, verify); `sd_heat3d_ds` 3813693 (csims ran in the bad env, but the judge applies the resource-cap and DSP-fill rejects before the csim gate, so no csim verdict was used: flash failed after 3 attempts on BRAM 103% / DSP-fill / a synthesis error — a valid result); C r4 (its only csim, 3815087 at 13:47, ran after the fix).
- 2026-10-23 13:51 **C r4 on Otus done** (`afw_hlsm_r4`, exit 0, 1 h 10 min): 836,464 cycles, DSP 7,041 (78 %), LUT 865,749, FF 1,122,814, BRAM 2,025, URAM 176; csim `correct=True` / `Passed!` (`agentfw/runs/mm1024_hlsm_r4/REPORT.md`). Fir's r4 (no csim result) stays in `mm1024_hlsm_r4_fir`. Driver moved on to `pl_win7080_cc_r2` (3815098).
- Posthoc csynths so far on Otus: 3 success (call_00006_1, call_00009_8, mm1024_hlsm_b12_ds_r2/exp_v1_csim), 2 real kernel errors (pragma conflict; invalid BIND_STORAGE ram_t2p+lutram).
- 2026-10-23 15:15 **SODA cosims on Otus after the fix**: the C TB now passes (`INFO: PASS!`). First 8 results: 3 genuine RTL deadlocks (patched denoise2d_u1, denoise2d_u2, denoise3d_u1: "DEADLOCK DETECTED", a design outcome), 5 `xelab` SIGSEGV (patched contrast_u4, contrast_u4_b128, contrast_u4_b128_laafd, denoise2d_u16, seidel2d_u8). The SIGSEGV is **not Otus-specific**: Fir had it on large designs (erosion/xcorr u32, designs/contrast_u8, designs/denoise3d_u16; see 2026-10-20). It crashes compiling `$unit_dataflow_monitor_sv` (Vitis' deadlock monitor). `ulimit -s unlimited` does not help (debug 3816751). Testing `cosim_design -disable_deadlock_detection` (debug 3816896, scratch copy only; production `job_soda_otus.sh` unchanged): it would keep the cycle counts but a deadlock would then hang to the time limit instead of being reported — **a method choice for the user**.
  - 15:27: `-disable_deadlock_detection` does not help either (debug 3816896): the monitor is still compiled and xelab now segfaults after `work.glbl`, at its final step. Not pursued further; open for the user (ideas not tried: `xelab -mt off` via `config_cosim`/`cosim_design -O` off, more memory, Vitis 2024.x). Production SODA jobs unchanged; the SIGSEGV designs stay failed, as they were on Fir.
- 2026-10-23 17:25 **Otus verify csynth matches Fir** (team v3, job 3811943, 5 h 58 min): `otus_setup.sh compare` gives MATCH on every field — 527,571 cycles (best = worst), interval 527,442, DSP 8,192, LUT 984,616, FF 1,706,888, BRAM 2,197, URAM 0. So the Otus csynth, SODA and LUD numbers can be compared with Fir's. The 21 csynths it gated were already submitted (…3813801).
  - **xclbin 3813695 failed in 6 s:** `sbatch_build_otus.sh` took the build folder from `$SLURM_SUBMIT_DIR`, but `resubmit_xclbin_otus.sh` submits with `--chdir`, so it never found `link.cfg`. Now it uses `${XCLBIN_DIR:-$PWD}`. Resubmitted as **3820062** (448G fits, since normal nodes have 730G; U280 platform `/opt/software/FPGA/Xilinx/platforms/xilinx_u280_gen3x16_xdma_1_202211_1/`; `xclbinutil` comes from Vitis' bin, and Otus has no `/opt/xilinx/xrt`).
  - **lsim_soda_pat 3814217 OUT_OF_MEMORY**: 25 of 60 done; 35 died (20 rc 137 killed, 15 rc 135 bus error) with PAR=8 in 64G. Node `/tmp` on Otus is GPFS, not tmpfs, so the memory is LightningSim itself. Their json/log moved to `/scratch/hpc-prf-llmfpga/asa582/projects/tmp/c2hls/lsim/otus_failed_oom_20261023/`; rerun as **3820398** (PAR=4, 192G, 24 h, ONLY = the 35), submitted directly, not through `resubmit_lsim_otus.sh`, because that script would also resubmit `lsim_proj` (3813805, still running).
  - **lsim_soda2 3814216**: all 46 failed at link with `undefined reference to _autotb_FifoCanRead_*` (non-blocking FIFO reads, which LightningSim does not support). This is the same error as the Fir failures of 10-20/21 in the same folder, so it is a tool limitation, not an Otus problem. The 15 patched deadlocks in `lsim/soda` (contrast, denoise2d) are Fir results from before the move.
  - **SODA cosims (Otus resubmit, 20 of 85 done):** 2 pass (patched jacobi2d_u1 1,069,740 cycles; jacobi2d_u2 535,194), 3 RTL deadlocks (patched denoise2d_u1/u2, denoise3d_u1), 15 xelab SIGSEGV (patched contrast_u4* ×6, contrast_u8/_cr, denoise2d_u16, denoise3d_u16, heat3d_u16, jacobi2d_u16, jacobi3d_u16, seidel2d_u16, seidel2d_u8). Same families as the **unpatched** failures on Fir (10-18: contrast_u8/_cr/u16/u4_cr…, denoise3d_u16, heat3d_u16, jacobi2d_u16, jacobi3d_u16, seidel2d_u16), so this matches Fir and is not an Otus regression. 65 still running.
  - Driver (HLSmith with an agent backend, `exp_hlsmith/driver.py`, pid 3992046): pl_win7080_cc_r2 (1 h timeout, then COMPLETED at 3 h: 884,837 cycles, DSP 6,400, csim True; r1 was 870,852), pl_noskills_cc_r3 (138,611,372, DSP 38, csim True; r2 35,750,152), pl_skills_cc_r3 (19,948,552, DSP 320, csim True; r2 35,839,468). These short runs (2–6 min) are normal: r1/r2 took 4–8 min. Now pl_dspredo_cc_r3 3820021.
  - 17:40: the first xclbin resubmit (3820062) failed in 5 s for a second reason: the U280 platform search loop ended on a missing folder, so it returned 1, and under `pipefail` the script exited with no message. Fixed (`|| true`, `sed -n 1p`), and the last `grep | head` is guarded too. Resubmitted as **3820417**: running, and `v++ -c` is compiling kernel0 for `xilinx_u280_gen3x16_xdma_1_202211_1`.
- 2026-10-23 18:10 check: 4 more SODA cosims pass (patched heat3d_u1 3,769,974 cycles; jacobi2d_u4 357,120; jacobi3d_u1 3,824,883; seidel2d_u1 1,067,931), so it is now 6 pass / 3 deadlock / 15 xelab SIGSEGV of 24; 61 running. The lsim_soda_pat rerun 3820398 is running with no memory kills so far: its first 3 (pat_denoise3d_u2/u4, pat_contrast_u4_b128_cr_laafd) end in LightningSim "Deadlock detected", like the patched contrast/denoise lsim deadlocks on Fir. xclbin 3820417 is at HLS block synthesis (139/140). Driver: pl_dspredo_cc_r3 and pl_refneu_cc_r3 done; pl_refnf_cc_r3 3821728 running.
- 2026-10-23 19:10 check:
  - SODA cosims, 28 of 85 done: **10 pass** (patched jacobi2d_u1 1,069,740 / u2 535,194 / u4 357,120 / u4_b128 357,120 / u8 178,947; jacobi3d_u1 3,824,883 / u2 1,912,752; seidel2d_u1 1,067,931 / u2 534,393; heat3d_u1 3,769,974), 3 deadlock, 15 xelab SIGSEGV; 57 running. Classifier: `bash /scratch/hpc-prf-llmfpga/asa582/projects/tmp/claude-95510/-scratch-hpc-prf-llmfpga-asa582-projects-workspaces-dse-v4-claude/aa860cea-994c-4932-be44-bfdfec62033d/scratchpad/soda_classify.sh`.
  - Posthoc csynths: `mm1024_hlsm_b12_ds_r3/exp_v3_widen1024` 948,192 cycles, DSP 6,803, LUT 786,705; `mm1024_team_r1/exp_v3_kc16_csim` 527,477 cycles, DSP 8,192, LUT 1,838,044 (above the U280's 1.30 M LUT); `mm1024_hlsm_ds_r2/exp_v3_csim` failed with `ERROR: [HLS 200-103] Unexpected exception occurred: Out of Memory` during scheduling (334,561 instructions after Array/Struct; sacct MaxRSS only 10 GB of 64G, so it may be a Vitis limit, not the cgroup). Its summary moved to `oom_3813790/`, and it was rerun at 192G as **3825753**.
  - xclbin 3820417: HLS done; link at implementation (4 strategies running).
  - Driver: pl_refnf_cc_r3, pl_refv2_cc_r3 and pl_win7080_cc_r3 done; now **agb_fft_r1** (HLSmith with an agent backend on FFT, agent shim login5:36016). Its csim/csynth checks take seconds because the kernel is small (csim PASSED, csynth LUT ~150–170 k), so they are real.
- 2026-10-23 20:15 check:
  - **lsim_soda_pat rerun 3820398 OOM again at 192G** (MaxRSS 192 GB, PAR=4), but 31 of the 35 have a result now: 23 with an `end` cycle, 8 LightningSim deadlocks (patched contrast_u1_cr, contrast_u2_cr, contrast_u4_b128_laafd, contrast_u4_b128_cr_laafd, denoise3d_u1/u2/u4/u16). The 4 still killed (pat_jacobi3d_u4_b128_laafd, pat_heat3d_u4_b128_laafd, pat_denoise3d_u4_b128_laafd, pat_jacobi3d_u8) were rerun one at a time as **3826441** (PAR=1, 480G, 24 h; normal nodes have 730G); their failed json/log are in `otus_failed_oom_20261023/<name>/rerun1/`.
  - LightningSim vs cosim on the same patched designs agrees to within 0.04 %: jacobi2d_u8 178,886 vs 178,947; seidel2d_u4 356,672 vs 356,724; jacobi2d_u4 357,068 vs 357,120; seidel2d_u1 1,067,879 vs 1,067,931; seidel2d_u2 534,341 vs 534,393.
  - SODA cosims: 12 pass (+ seidel2d_u4 356,724, seidel2d_u4_b128 356,733), 3 deadlock, 15 xelab SIGSEGV; 55 running.
  - afw_post `mm1024_team_r1/exp_post/v4_ti36_csynth` done (7 h 10 min): 911,371 cycles, interval 983,532, DSP 8,834, LUT 1,244,462, FF 2,069,037, BRAM 2,197. v4_ti40 is still running.
  - xclbin 3820417: still in implementation (0 of 5 runs complete). Driver: agb_fft_r1 still running.
- 2026-10-23 21:15 check:
  - afw_post `mm1024_team_r1/exp_post/v4_ti40_csynth` done: 883,480 cycles, DSP 8,706, LUT 1,454,056, FF 2,271,406, BRAM 2,197 (with v4_ti36's 911,371 at 20:15, the team pair is complete).
  - Posthoc `skills_ag_dspredo_r2_.../call_00016_3_csim`: 724,446 cycles, DSP 8,323, LUT 844,430, FF 1,361,090, BRAM 120.
  - SODA cosims: 14 pass (+ patched heat3d_u2 1,885,266, contrast_u1 468,963), 3 deadlock, 15 xelab SIGSEGV; 53 running.
  - Driver: **agb_fft_r1** done (1 h 31 min; HLSmith with an agent backend): baseline 36,494 cycles → 367 cycles, DSP 6,800, LUT 952,742, csim True. Now agb_hlsfactory_cholesky_r1 (3826558).
  - xclbin 3820417: still in implementation (4 strategies, 0 of 5 complete). lsim 3826441 (the 4 big ones, 480G) is running.
- 2026-10-23 22:15 check:
  - SODA cosims: 17 pass, 3 deadlock, 15 xelab SIGSEGV of 35; 50 running. New passes and other non-jacobi/seidel passes: designs_patched/heat3d_u4 1319877;designs_patched/heat3d_u4_b128 1319877;designs_patched/jacobi3d_u4 1339146.
  - Driver (HLSmith with an agent backend, `exp_hlsmith/runs/skills_agb_hlsfactory_*_r1_*`):
    - **agb_hlsfactory_cholesky_r1**: 68,337,001 → 5,336 cycles, DSP 5,946, LUT 419,007, csim passed. N = 120; the final kernel is a blocked right-looking Cholesky (4×4 pivot blocks, an II=1 trailing update of 256 MACs per cycle), so ~576 k MACs / 256 ≈ 2,250 cycles: plausible.
    - **agb_hlsfactory_fdtd-2d_r1** (6 min 45 s): 98,261,801 → 3,818 cycles, DSP 3,806, LUT 501,848, csim passed.
    - Now agb_hlsfactory_gramschmidt_r1 (3828007).
  - xclbin 3820417: still 0 of 5 implementation runs complete (about 3 h in implementation).
- 2026-10-23 23:15 check: **lsim_proj 3813805 done** (9 h 39 min): `/scratch/hpc-prf-llmfpga/asa582/projects/tmp/c2hls/lsim/soda/contrast_u16_cr/proj.tar.gz` (184 MB, "proj ok"); its LightningSim run was submitted as **3831009** (`lsim_u16cr`, PAR=1, 192G). SODA cosims: 19 pass (+ patched jacobi3d_u4_b128 1,339,137, jacobi3d_u8 669,915), 3 deadlock, 15 xelab SIGSEGV; 48 running. Driver on agb_hlsfactory_gramschmidt_r1. xclbin 3820417: still 0 of 5 implementation runs complete.
- 2026-10-24 00:15 check:
  - lsim_u16cr 3831009 (contrast_u16_cr): rc 1 after 25 s, `undefined reference to _autotb_FifoCanRead_i33/_i513`. This is the same LightningSim limitation as the 46 lsim_soda2 projects (non-blocking FIFO reads), so the unpatched contrast_u16_cr has no LightningSim number.
  - Driver: **agb_hlsfactory_gramschmidt_r1** done (1 h 7 min): no gain, final = baseline = 2,270,081 cycles, DSP 660, LUT 66,797, csim True. Now agb_hlsfactory_jacobi-2d_r1 (3831013).
  - SODA cosims unchanged: 19 pass / 3 deadlock / 15 SIGSEGV; 48 running. xclbin 3820417: about 5 h in implementation, still 0 of 5 complete (2-day limit).
- 2026-10-24 01:20 check:
  - **3 posthoc csynths TIMEOUT at 12 h** (3813787–89): `mm1024_hlsm_ds_r1/exp_v2_T64J32K4_csim`, `mm1024_hlsm_ds_r1/exp_v4_T67J128K1_csim`, `mm1024_hlsm_ds_r2/exp_v2a_csim` (MaxRSS ~5 GB). Their non-csim twins in the agent sessions on Fir also never finished csynth (`agentfw/runs/.../exp/v2_T64J32K4`, `v4_T67J128K1`: 3 TIMEOUTs each; `v2a_128x64`: 2), so the slowness is in these kernels, not Otus. Rerun once with a 3-day limit (normal allows 21 days): **3831181, 3831182, 3831183**. If they time out again, treat them as "csynth does not finish".
  - Driver: agb_hlsfactory_jacobi-2d_r1 done (1 h 2 min): 3,112,241 → 4,453 cycles, DSP 7,040, LUT 1,233,442, csim True. Now agb_hlsfactory_lu_r1.
  - xclbin 3820417: 1 of 5 implementation runs complete. SODA cosims unchanged (19/3/15; 48 running).
- 2026-10-24 02:20 check:
  - **lsim_soda_pat 3826441 (the 4 big ones, PAR=1, 480G) done** (5 h 38 min, MaxRSS 192 GB for one project): pat_jacobi3d_u4_b128_laafd 8,783,426 cycles; pat_heat3d_u4_b128_laafd 8,717,645; pat_jacobi3d_u8 574,472; pat_denoise3d_u4_b128_laafd LightningSim deadlock. **All 60 lsim_soda_pat projects now have a result**: 26 with an `end` cycle, 34 deadlock or link failures. (The laafd variants are ~7.6× slower than the plain u4_b128 ones: 8.78 M vs 1.15 M.)
  - Posthoc `mm1024_hlsm_ds_r2/exp_v3_csim` rerun at 192G (3825753): **the same `HLS 200-103 Out of Memory`**, again at MaxRSS 10.6 GB. So it is a Vitis-internal limit on this kernel, not the job's memory, and that is its final result.
  - Posthoc 3813783 `mm1024_hlsm_b12_ds_r1/exp_v2_csim`: success, 2,291,745 cycles (finished earlier and not logged until now).
  - xclbin 3820417: 3 of 5 implementation runs complete. Driver: agb_hlsfactory_lu_r1 running. SODA cosims unchanged (19/3/15; 48 running).
- 2026-10-24 03:15 check: SODA cosims 20 pass (+ patched heat3d_u8 660,231), 3 deadlock, 15 SIGSEGV; 47 running (all 37 unpatched `designs/` still running at about 12 h of their 2 days). xclbin 3820417: 4 of 5 implementation runs complete. Driver: agb_hlsfactory_lu_r1 at about 3 h (its 3 h job limit: the driver retries a TIMEOUT at a longer limit).
- 2026-10-24 04:15 check: agb_hlsfactory_lu_r1 hit its 3 h limit; the driver reran it at 6 h (3831887, try 2). xclbin 4 of 5 implementation runs done. SODA cosims unchanged (20/3/15; 47 running).
- 2026-10-24 05:15 check: no change. All within their limits: lud_csynth 16.7 h / 2 d, 37 unpatched soda_cosim 14.8 h / 2 d, xclbin 11.9 h / 2 d (4 of 5 implementation runs), 3 posthoc reruns 4 h / 3 d, agb_hlsfactory_lu_r1 try 2 1.8 h / 6 h.
- 2026-10-24 06:50 check:
  - **xclbin 3820417 FAILED after 12 h 26 min: the U280 link does not route.** HLS compile and synthesis finished. All 4 multi-strategy implementation runs failed: `Design is not routable as its global congestion level is 7` and `Routing results verification failed due to partially-conflicted nets` (in `A_IO_L1_in_wrapper_100/101`). Then `VPL 60-704 ... none of the implementation runs can meet timing` and `v++ 60-626 Kernel link failed`. Placed utilization (impl_Performance_HighUtilSLRs): LUT 804,682 (61.7 %), FF 1,057,493 (40.6 %), BRAM 293.5 (14.6 %), DSP 5,124 (56.8 %), URAM 0. So this is congestion (128 PEs, wide IO modules), not capacity. Kept: `xclbin/v4_systolic_st0_c1/out/kernel0.xo` (rerun the link only), `out/reports/` (per-strategy utilization/timing), `out/logs/`. **Not resubmitted**: a new link setup (lower kernel clock, SLR assignment or floorplan in `link.cfg`, different strategies) is a design decision, open for the user. The Fir run never started (it was pending), so there is no Fir result to compare.
  - Driver: agb_hlsfactory_lu_r1 done on try 2 (2 h 27 min): 134,372,041 → 4,664 cycles, DSP 6,792, LUT 626,013, csim True. Now agb_hlsfactory_ludcmp_r1 (3832058).
  - SODA cosims unchanged (20/3/15; 47 running).
- 2026-10-24 07:40 **xclbin link retried at 250 MHz (user's request): job 3832310.** It reuses `out/kernel0.xo` (no HLS recompile) with `xclbin/v4_systolic_st0_c1/link_250mhz.cfg`, which is link.cfg plus `[clock] freqHz=250000000:kernel0_1.ap_clk`. The same HBM mapping, global retiming and 4 strategies. `sbatch_build_otus.sh` now takes `LINK_CFG` (default link.cfg); submitted with `LINK_CFG=link_250mhz.cfg bash resubmit_xclbin_otus.sh`. The cfgen line shows `-clock.freqHz 250000000:kernel0_1.ap_clk` (only a deprecation warning: v++ prefers `freqhz`). The failed 300 MHz reports/logs moved to `out_300mhz_failed_3820417/`.
- 2026-10-24 08:20 check: no new results. xclbin 250 MHz (3832310) at block-level synthesis (141/142). Driver: agb_hlsfactory_ludcmp_r1 running (2.5 h of 3 h). SODA cosims unchanged (20/3/15; 47 running).
- 2026-10-24 09:50 check: agb_hlsfactory_ludcmp_r1 hit its 3 h limit; the driver reran it at 6 h (3832533, try 2). xclbin 250 MHz (3832310) now in implementation (4 strategies, 0 of 5 complete). SODA cosims unchanged (20/3/15; 47 running).
- 2026-10-24 11:20 check:
  - Driver: agb_hlsfactory_ludcmp_r1 done on try 2 (1 h 6 min): 10,095,723 → 5,187 cycles, DSP 7,944, LUT 616,208, csim True. Now agb_lud_r1 (3833226).
  - xclbin 250 MHz (3832310): still in implementation, 0 of 5 runs complete, 4 in progress.
  - SODA cosims unchanged (20 pass / 3 deadlock / 15 xelab_segv; 47 running). lud_csynth and the 3 posthoc reruns still running.
- 2026-10-24 12:50 check: no job ended. xclbin 250 MHz (3832310): all 4 implementation runs are routing (Phase 4 rip-up and reroute, global iteration 0); router estimate is global/short congestion level 6 (64x64). The 300 MHz run (3820417) ended with "not routable, global congestion level 7". Seen via srun --overlap on cn0607.
- 2026-10-24 14:20 check:
  - Driver: agb_lud_r1 hit its 3 h limit; the driver reran it at 6 h (3838306, try 2).
  - xclbin 250 MHz (3832310): all 4 runs still in rip-up and reroute; overlapping nodes falling (impl_1 138,833 → 81,337; SpreadLogic_high 250,615 → 136,379; HighUtilSLRs 487,569 → 267,332; ExplorePostRoutePhysOpt 464,168 → 255,477).
  - Nothing else ended.
- 2026-10-24 15:50 check, xclbin 250 MHz (3832310): 3 of 5 strategies have failed routing.
  - impl_1: route_design failed (partially-conflicted nets).
  - Congestion_SSI_SpreadLogic_high: "Routing results verification failed due to partially-conflicted nets".
  - Performance_HighUtilSLRs: "46505 signals failed to route due to routing congestion", intermediate WNS -4.231 ns.
  - Still open: Performance_ExplorePostRoutePhysOpt, in reroute, global iteration 1; Performance_NetDelay_high, just started (5th run, waited for a free slot).
  - Lowering the clock did not fix the congestion. The next lever is placement/floorplan, not frequency. That is the user's decision.
- 2026-10-24 17:20 check, xclbin 250 MHz (3832310): 4 of 5 strategies have failed routing. Performance_ExplorePostRoutePhysOpt also ended with "route_design failed due to earlier errors". The last one, Performance_NetDelay_high, is in placement (Phase 3.4). Nothing else ended. agb_lud_r1 try 2 is at 4.2 h of 6 h.
- 2026-10-24 18:50 check:
  - Posthoc 3-day reruns, 2 of 3 done (both success=True):
    - mm1024_hlsm_ds_r1/exp_v2_T64J32K4_csim (3831181, 16.5 h): 772,269 cycles, DSP 6,240, LUT 1,573,076, FF 1,915,782, BRAM 2,138.
    - mm1024_hlsm_ds_r2/exp_v2a_csim (3831183, 17.2 h): 749,101 cycles, DSP 6,144, LUT 1,454,646, FF 1,629,012, BRAM 2,138.
    - Both exceed the U280's 1,303,680 LUTs, so neither fits. Both reach only II=4 on the k loop (carried dependence on the accumulator store).
    - mm1024_hlsm_ds_r1/exp_v4_T67J128K1_csim (3831182) still running.
  - xclbin 250 MHz (3832310): Performance_NetDelay_high is now routing (Phase 2.4); the other 4 strategies failed.
- 2026-10-24 20:20 check:
  - Driver: agb_lud_r1 timed out at 6 h too (3838306) and was counted as failed. "driver done" at 19:10; all driver tags are finished.
  - Leftover: the lud agent session (login5 pid 782864, workspace logs/shim_agent_exp_lud/call_00005) kept running after the job ended. It is still using Claude and submitting flash_agent_csynth jobs (3847345). Nothing collects its result.
    - My kill/scancel was refused by the permission check; it is left for the user to stop.
  - xclbin 250 MHz: Performance_NetDelay_high still routing; the other 4 strategies failed.
- 2026-10-24 21:50 check: the leftover lud agent (pid 782864) is gone and no flash_agent jobs remain. xclbin NetDelay_high still routing. Nothing else ended.
- 2026-10-24 23:20: **xclbin 250 MHz (3832310) FAILED** after 14.5 h. All 5 strategies failed routing (partially-conflicted nets / congestion): "none of the implementation runs can meet timing".
  - Placed utilization: LUT 815,416 (62.6%), DSP 5,124 (56.8%).
  - Per SLR, LUT is SLR0 8.7%, SLR1 91.5%, SLR2 88.4%; DSP is SLR0 0, SLR1 80.7%, SLR2 86.1%.
  - The placer packs the kernel into SLR1 and SLR2 and leaves SLR0 almost empty. That, not the clock, is why routing fails. The next lever is SLR assignment/floorplan (or a smaller array), which is the user's decision.
  - Reports and logs moved to `xclbin/v4_systolic_st0_c1/out_250mhz_failed_3832310/`; `out/kernel0.xo` kept.
- 2026-10-25 02:20: posthoc rerun mm1024_hlsm_ds_r1/exp_v4_T67J128K1_csim (3831182) done after 24.1 h, success=True.
  - Result: latency 1,021,869 cycles, interval 1,364,910, DSP 6,432, LUT 1,648,507 (> 1,303,680, does not fit), FF 1,979,351, BRAM 2,138. cmp_k reaches only II=4.
  - All 3 slow posthoc csynths are now finished; none fits the U280 by LUT. Remaining: lud_csynth, 37 soda_cosim (unpatched), 10 soda_pat_cosim.
- 2026-10-25 12:45:
  - lud_csynth n32_st5 (3813694) TIMEOUT at the 2-day limit. On Fir it already timed out at 12 h, 24 h and the escalation tiers, so this is final: AutoSA LUD n32_st5 csynth does not finish in 2 days.
  - The 47 SODA cosims (37 unpatched, 10 patched) reach their 2-day limit at 13:27 CET. Classifier is unchanged before that (20 pass / 3 deadlock / 15 xelab_segv).
- 2026-10-25 13:40: **All 47 remaining SODA cosims TIMEOUT at the 2-day limit.**
  - All 37 unpatched designs (the naive baseline generated from SODA).
  - 10 patched: contrast_u1_cr, contrast_u2, contrast_u2_cr, denoise2d_u4/u4_b128/u8, denoise3d_u2/u4/u4_b128/u8.
  - Final Otus SODA cosim tally, 85 designs:
    - 20 pass (all patched);
    - 3 deadlock (patched denoise2d_u1/u2, denoise3d_u1);
    - 15 xelab SIGSEGV (patched contrast_u4/u8 family and all u16);
    - 47 timeout at 2 days.
  - No Otus jobs left in the queue. Open decisions for the user: xclbin next step (SLR assignment / smaller array / stop), whether to rerun the 47 timeouts longer (partition max 21 days) or report them as "cosim does not finish in 2 days".
- 2026-10-25: Baseline check, agb_hlsfactory_* vs bench_wide: same baseline reports. bench_wide `results_*.csv` gt_lat/lat are worst-case latency (latency_cycles_worst); my agb notes quoted typical latency.
  - cholesky typical/worst: 68,337,001 / 270,927,961.
  - lu: 134,372,041 / 539,513,401.
  - ludcmp: 10,095,723 / 32,168,883.
  - The agb results are equal typical and worst except gramschmidt (2,270,081 typical / 4,529,281 worst = baseline).
- 2026-10-25 **LightningSim vs cosim, all pairs** (`study/soda/lsim/compare_pat_otus.py` → `lsim_vs_cosim_otus_20261025.csv`):
  - **Pairs:** 37 patched designs have both a cosim pass and a LightningSim end cycle. 11 patched designs have LightningSim only (cosim timed out or crashed). Unpatched designs have cosim only (FifoCanRead).
  - **1D/2D designs, 27 pairs:** blur, sobel2d, erosion, xcorr, contrast, jacobi2d, seidel2d. The end cycle is 0.003–0.12% below cosim. With the compare.py report mapping the error is 0.000% to −0.005%.
  - **3D designs, 10 pairs (heat3d, jacobi3d): LightningSim is 14–20% low**, in the same pattern for both:
    - u1: −20.0%;
    - u2: −16.0%;
    - u4, u4_b128 and u8: −14.3%.
  - This corrects the earlier "agrees to within 0.04%" note, which only held for the 1D/2D designs. Cause not investigated.
  - The 3D LightningSim-only numbers (u16 and *_laafd) are therefore not trustworthy.
- 2026-10-25 **3D LightningSim gap, investigation** (`study/soda/inv3d/`):
  - **Finding:** only the 3D designs (heat3d, jacobi3d) contain Vitis free-running pipelines (FRP). Their Module6Func, the 7-input weighted sum (PipeLatency 49), gets a `frp_fifoout` output buffer of depth 50.
  - **Separation is perfect over the 37 pairs:** all 27 designs without FRP match cosim (0.000% to −0.005%); all 10 designs with FRP are −14% to −20%.
  - **LightningSim (fork in /scratch/hpc-prf-llmfpga/asa582/projects/tools/lightningsim/src) has no FRP model.** Every pipeline is simulated as a stall pipeline.
  - **Both simulators are slower than the ideal ~2.26M cycles of heat3d_u1** (II=1 at 2,261,648 trips), so a back-pressure limit exists in the fork/join of shallow FIFOs (depth 3–15). FRP changes how much.
  - **Test:** heat3d_u1 with `style = stp` on Module6's pipeline (only change; `inv3d/heat3d_u1_stp/`). Jobs: cosim 3933015, LightningSim project 3933016, LightningSim 3933017.
  - If cosim(stp) ≈ LightningSim, FRP is the cause.
- 2026-10-26 **3D gap resolved: FRP is the cause.** The stp cosim (job 3933015, 3 h 49 min) **passed** (`INFO: PASS`). It gives **3,016,143 cycles**.
  - That is exactly LightningSim's mapped number (end 3,016,091 → 3,016,143 with the compare.py mapping), so the error is 0.000%.
  - The original design, where Vitis picked FRP for Module6, takes 3,769,974 cycles in cosim. Forcing `style = stp` on that one pipeline makes the design 20% faster in cosim, and LightningSim then matches it exactly.
  - **Conclusion:** LightningSim is exact for stall pipelines. It underestimates designs that contain Vitis free-running pipelines (here −14% to −20%). Flag the 3D LightningSim-only numbers (u16, *_laafd) as such, or rebuild them with stp.
  - **Side result:** for these SODA 3D kernels FRP is slower than stp. The FRP output buffer's credit check plus shallow FIFOs throttle the fork/join.
- 2026-10-26 **stp rebuild of the other 13 patched 3D designs** (user: "yes, rebuild the 3D ones with stp and rerun lsim").
  - Script: `study/soda/inv3d/submit_stp_all.sh`. Each design is copied to `inv3d/<d>_stp`, and `style = stp` is appended to every pipeline pragma.
  - Each design gets a csynth project job and a LightningSim job (afterok). Results go to `/scratch/hpc-prf-llmfpga/asa582/projects/tmp/c2hls/lsim/inv3d/<d>_stp/lsim_v2023.json`.
  - Job IDs are in `inv3d/submitted.txt` (3973980–3974005). The laafd and u8 designs get 480G, because they ran out of memory at 192G before.
  - Check every built RTL for `frp_` before using a number.
- 2026-10-26 **stp rebuild results (11 of 13 done; both laafd still running, 1.5 h in):**
  - No rebuilt RTL contains `frp_fifoout`. The check works: the original pat_heat3d_u2 RTL has 12 hits.
  - **Every LightningSim end cycle is identical to the earlier number for the same design**, e.g. heat3d_u16 283,421 and jacobi3d_u8 574,472.
  - This is expected: LightningSim always simulated stall semantics, so forcing stp changes nothing it sees.
  - **Meaning:** the old 3D LightningSim numbers are correct predictions for the stp variant of each design (validated by cosim only for heat3d_u1, exact). They are not predictions for the designs as Vitis builds them by default, which are 14–20% slower in cosim.
  - Cosim of the other stp variants would validate the rest; not submitted.
- 2026-10-26 **laafd stp done:** heat3d_u4_b128_laafd 8,717,645 and jacobi3d_u4_b128_laafd 8,783,426, both identical to the old LightningSim numbers. All 13 rebuilt 3D designs are done, with no frp and no change in LightningSim.
