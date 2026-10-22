# flash_refined

The c2hls flash step with its hold-back cues removed. It is applied at runtime by `run_flash_cc.py --refined`, because the c2hls checkout is read only; nothing in it is modified. The original skill packs are unchanged as well: the refined pack is a new file in this directory.

| File | What it is |
|---|---|
| `flash_refined.py` | `apply()`: replaces the flash system and user prompts, replaces the flash resource block, sets the flow environment and selects the refined skill pack |
| `make_refined_skills.py` | builds the refined pack from pack 5; every edit is an asserted exact-substring replacement |
| `skills_ii_target_miss_solutions_added(90skills)_plus_gemm_flatten_v2_refined.json` | version 1 pack: 136 skills (134 from the source pack + 2 new). Its `"refined"` block lists every edit. |
| `skills_ii_target_miss_solutions_added(90skills)_plus_gemm_flatten_v2_refined_v2.json` | version 2 pack: 138 skills (version 1 + 2 balance skills) |

## What changes (flash step only)

| Source of the hold-back | Original c2hls | flash_refined |
|---|---|---|
| System prompt `Instruction_c2hls_flash` (`prompt_c2hls.py:627–651`) | "modest unroll, array partition on small buffers, DATAFLOW only when buffering stays simple, … narrow host ABI"; "Prefer surgical pragma/bundle edits on the existing structure" | unroll and partition sized to the device budget; DATAFLOW or ping-pong to overlap memory with compute; 512-bit access, with wide internal types allowed and the top-level argument types unchanged; "scale the datapath toward the device budget"; full restructuring allowed. Bundle, correctness and loop-label rules are kept verbatim. |
| User prompt `q_optimize_flash` (`prompt_c2hls.py:975–1050`) | item 3 "small … bounded power-of-two factor"; item 4 "partition only small hot buffers"; item 5 "DATAFLOW only when … simple"; item 6 "Do NOT introduce `ap_uint<512>` …"; "Prefer a compact … modest unroll"; "Prefer surgical edits" | 6 asserted replacements in the same spirit; every other line (ABI, bundles, burst legality, labels, interface check, output format) is unchanged |
| Resource block `_render_step_resource_constraints` (`c2hls.py:2636`) | "RESOURCE CONSTRAINTS … max = 12× current DSP … automatic revert", plus "Reserve DSP-heavy unrolling for the `unroll` and later steps" | DEVICE BUDGET: current usage and device capacity per resource; must be faster; every resource below 100%; no growth cap. Other steps keep the original block. |
| Regression guard `STEP_REGRESSION_THRESHOLDS["flash"]` (`c2hls.py:1978`) | per-step ratios (DSP 12×, BRAM 6×, LUT 6×, …) | `C2HLS_STEP_REGRESSION_THRESHOLDS_JSON`: flash keeps latency 1.0× (must not get slower), resource ratios removed |
| DSP floor (`autosa_flow_gates.py`) | off unless the DSP-redo launcher sets it | on, as in `scripts/pc2/start_autosa_mm_flash_dsp_redo.sh`: `DSP_REDO=1`, `MIN_DSP=300`, `MAX_DSP=9024`, `DSP_FILL_PCT=90` (fill gate plus the <100% resource-cap gate) |
| Skills | pack 5, `…plus_gemm_flatten_v2.json` (134 skills) | refined pack: 7 hold-back wordings rewritten (wide bus only "if explicitly allowed", "conservative" unroll factors, "conservative BRAM budget", separate bundles only "when supported", "reserve conservative factors"); "resource explosion" defined as exceeding device capacity or timing (4 places); plus `hls-size-parallel-datapath-to-device-budget` and `avoid-idle-device-on-compute-bound-kernel`. Legality guards are unchanged. |

## Version 2: the compute/transfer balance rule (default; `--refined-version 1` reproduces version 1)

Version 1 made every flash run fill the device on the first reply, but 2 of 3 plain replicates were memory-bound. Their DSPs were filled, but B was reloaded every tile, C was moved one float per beat (II=16), and there was no overlap. Version 2 adds:

- **Prompts:** a balance bullet in the system prompt and a balance paragraph in the user prompt. Per-tile compute must exceed per-tile transfers: keep reused operands resident, move every array (C included) in 512-bit beats at II=1, and overlap loads and stores with compute.
- **Check (flow):** `c2hls.flash_dsp_fill_reject_error` is wrapped. After the original fill check passes, a flash design is rejected when `latency_cycles > 1.5 × compute floor`. The compute floor is the total MACs (I×J×K from the bench's `kernel.h`) divided by the design's own DSP count / 5 DSP per fp32 MAC. The rejection names the longest pipelined loops. It goes through c2hls's existing fill-repair path, whose template `hls_flash_dsp_fill_fix` is replaced with one that covers both causes (DSP not filled, or compute/transfer imbalance). Tunables: `FLASH_REFINED_BALANCE_RATIO` (1.5) and `FLASH_REFINED_DSP_PER_MAC` (5).
- **Skills:** `hls-balance-compute-and-transfer-after-scaling` and `avoid-transfer-bound-filled-datapath`, in the version 2 pack.

On the earlier reports, the check rejects refined r2 (4.43×), refined r3 (3.19×), DSP-redo r2 (1.97×), DSP-redo agent (16.3×) and the 640-DSP plain run (1.76×). It passes DSP-redo r1 and r3, refined r1, the refined agent run, and clean run2.

## Run

```bash
cd /scratch/asa582/workspaces/c2hls_claude_code
"$PY" run_flash_cc.py --flavor skills --refined --shim-url "$(cat shim_url.txt)" --dry-run --show-refined   # render what c2hls will send
sbatch --export=ALL,FLAVOR=skills,REFINED=1 sbatch_flash_cc.sh                                              # plain
sbatch --export=ALL,FLAVOR=skills,REFINED=1,CLAUDE_CODE_SHIM_URL=$(cat shim_agent_url.txt) sbatch_flash_cc.sh  # agent mode
```

`--skills-json` still overrides the pack. Post-flash stages stay off unless `--post-flash` is given.

## DSP window and the DSP ladder (`dsp_window.py`)

`run_flash_cc.py --refined --dsp-window LO,HI` replaces the ≥ 90% fill target with a csynth DSP window. HI is a hard roof (c2hls's ceiling gate, `C2HLS_FLASH_MAX_DSP=HI`). A design below LO is rejected by an exact check, unless another resource is already ≥ 90% of the device (c2hls's own escape). Every prompt and repair text that said "fill the device" states the window instead. The balance check (version 2) is unchanged. `--stage-note-file` appends a stage instruction to the flash guidance. `--seed-code FILE` makes Phase B start from FILE instead of an LLM translation; c2hls csynths and csims it as that step's baseline.

| Run | Command | Window |
|---|---|---|
| 70–80% window | `sbatch --export=ALL,FLAVOR=skills,REFINED=1,DSP_WINDOW=6317:7219,TAG=win7080_r1 sbatch_flash_cc.sh` | 6,317–7,219 |
| DSP ladder | `sbatch --export=ALL,TAG=r1 sbatch_flash_ladder.sh` (`run_flash_ladder.py`) | L1 from plain C: 1,805–2,707 (20–30%)<br>L2 seeded with L1: 1.8–2.2× L1's DSP ("double it")<br>L3 seeded with L2: max(70%, 1.1× L2) to 7,219 ("increase it") |

The 80% roof (7,219) holds at every ladder stage, and every stage must be faster than its seed. Ladder output goes to `runs/ladder_<tag>_<stamp>/` (`skills_L{1,2,3}_*/autosa_mm`, `L*_note.txt`, `ladder_summary.json`).
