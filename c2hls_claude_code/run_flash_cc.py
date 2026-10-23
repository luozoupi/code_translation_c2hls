#!/usr/bin/env python3
"""Run one c2hls flash cell (autosa bench) with Claude Code as the LLM, through claude_code_shim.py.

The flow is c2hls's own, unchanged: configure_site("fir") -> apply_mm_flow_flavor(flavor) ->
configure_autosa_campaign_env(variant) -> bench timeouts -> run_benchmark_multistep(...), the same
calls the fir batch worker makes (scripts/fir/batch_parallel/dispatch.py run_flash_bench). Only the
LLM endpoint differs: OPENAI_BASE_URL points at the shim, so every flash, repair, DSE and stream
call is a `claude -p`. All output goes to --out-root (outside the c2hls checkout).

  run_flash_cc.py --flavor skills --shim-url http://login3:18300/v1 [--bench autosa_mm] [--dry-run]

Flavors (scripts/pc2/autosa_flash_lib.py): skills (aav_n_gf + post-flash DSE/stream), aav_n_90,
noskills, zero_shot, one_shot.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

C2HLS = Path(os.environ.get("C2HLS_ROOT", "/scratch/hpc-prf-llmfpga/asa582/projects/workspaces/code_translation_c2hls"))
sys.path.insert(0, str(C2HLS))
sys.path.insert(0, str(C2HLS / "scripts"))
sys.path.insert(0, str(C2HLS / "scripts" / "pc2"))

HERE = Path(__file__).resolve().parent
# Every switch that chains a post-flash stage after the flash step (c2hls.py, after save_multistep_results).
POST_FLASH_SWITCHES = ("C2HLS_DSE_V2", "C2HLS_DSE_V2_CHAIN_FLASH", "C2HLS_DSE_V4", "C2HLS_POST_FLASH_DSE",
                       "C2HLS_DSE_CHAIN_FLASH", "C2HLS_POST_FLASH_STREAM", "C2HLS_STREAM_CHAIN_FLASH",
                       "C2HLS_POST_FLASH_PRAGMA_OPT", "C2HLS_PRAGMA_OPT_CHAIN_FLASH",
                       "C2HLS_POST_FLASH_LATENCY_OPT", "C2HLS_LATENCY_OPT_CHAIN_FLASH")
SHOWN_ENV = ("C2HLS_STRATEGY", "BATCH_PARALLEL_VARIANT", "C2HLS_MM_FLOW_FLAVOR", "C2HLS_SKILL_MODE",
             "C2HLS_SKILL_PROMPT_MODE", "C2HLS_PACKAGED_SKILLS_JSON", "C2HLS_PACKAGED_SKILLS_ONLY",
             "C2HLS_FLASH_SKILL_ENTRIES_JSON", "C2HLS_POST_FLASH_NO_SKILLS", "C2HLS_SKIP_PHASE_B",
             "C2HLS_FLASH_OPT_PROMPT_MODE", "C2HLS_ONE_SHOT", "C2HLS_FLASH_ONLY", "C2HLS_TURNS",
             "C2HLS_POST_FLASH_DSE", "C2HLS_DSE_CHAIN_FLASH", "C2HLS_POST_FLASH_STREAM", "C2HLS_STREAM_CHAIN_FLASH",
             "C2HLS_DSE_V2", "C2HLS_DSE_V2_CHAIN_FLASH", "C2HLS_RUN_COSIM", "C2HLS_PART", "C2HLS_CLOCK_NS",
             "C2HLS_CSIM_TIMEOUT", "C2HLS_SYNTH_TIMEOUT", "C2HLS_LLM_TIMEOUT", "C2HLS_MODEL", "OPENAI_BASE_URL", "C2HLS_THINKING", "C2HLS_REASONING_EFFORT", "C2HLS_COMPILE_INCLUDE_PATHS", "C2HLS_FLASH_MAX_TOKENS", "C2HLS_LLM_MAX_TOKENS",
             "C2HLS_TMP_ROOT", "C2HLS_AUTOSA_READY_ROOT", "C2HLS_SITE", "C2HLS_FLASH_DSP_REDO", "C2HLS_FLASH_MIN_DSP", "C2HLS_FLASH_MAX_DSP", "C2HLS_FLASH_DSP_FILL_PCT", "C2HLS_STEP_REGRESSION_THRESHOLDS_JSON", "FLASH_WINDOW_LO", "FLASH_WINDOW_HI") + POST_FLASH_SWITCHES


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--flavor", default="skills", help="skills | aav_n_90 | noskills | zero_shot | one_shot")
    ap.add_argument("--bench", default="autosa_mm")
    ap.add_argument("--bench-root", default=str(C2HLS / "artifacts/pc2/autosa_mm_ijk_benches/n1024"),
                    help="autosa_ready-style root holding <bench>/metadata.json (read only); the default n1024 "
                         "autosa_mm has the same kernel.h and testbench as the DSE trials")
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--turns", type=int, default=None, help="repair turns (default: C2HLS_TURNS or 3)")
    ap.add_argument("--shim-url", default=os.environ.get("CLAUDE_CODE_SHIM_URL", ""))
    ap.add_argument("--out-root", default=str(HERE / "runs"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--dry-run", action="store_true", help="configure and print the plan; no LLM or Vitis")
    ap.add_argument("--refined", action="store_true",
                    help="flash_refined: hold-back cues removed from the flash prompts/flow, DSP floor on, refined "
                         "skill pack (flash_refined/flash_refined.py); needs a skills flavor")
    ap.add_argument("--refined-version", type=int, default=2,
                    help="flash_refined version: 1 = hold-back fixes + DSP floor; 2 = plus the compute/transfer balance rule")
    ap.add_argument("--show-refined", action="store_true", help="with --dry-run: print the rendered refined prompts")
    ap.add_argument("--dsp-window", default="",
                    help="with --refined: LO,HI (or LO:HI, for sbatch --export) csynth DSP window instead of the >=90%% fill target; HI is a hard roof "
                         "(flash_refined/dsp_window.py)")
    ap.add_argument("--stage-note-file", default="", help="with --dsp-window: text appended to the flash guidance (ladder stage)")
    ap.add_argument("--controlled", action="store_true",
                    help="controlled model comparison (controlled/ctrl_llm.py): one prompt format for every backend, "
                         "answer = message.content only, retries on truncation/empty/transport; per-try log in the cell")
    ap.add_argument("--seed-code", default="",
                    help="Phase B starts from this kernel (no LLM translation); c2hls csynths/csims it as the baseline")
    ap.add_argument("--dsp-redo", action="store_true",
                    help="flash DSP floor/fill gates as in scripts/pc2/start_autosa_mm_flash_dsp_redo.sh: "
                         "DSP_REDO=1, MIN_DSP=300, MAX_DSP=9024, DSP_FILL_PCT=90")
    ap.add_argument("--refined-neutral", action="store_true",
                    help="with --refined: neutral refined prompts and pack (hold-back cues removed, no wording that "
                         "asks to size the design to the device); no DSP floor/fill gate; balance gate and 9024 cap stay")
    ap.add_argument("--no-dsp-floor", action="store_true",
                    help="with --refined: drop the DSP floor/fill gate that --refined turns on (DSP_REDO, MIN_DSP, "
                         "FILL_PCT); refined prompts, refined pack, balance gate and the 9024 device cap stay")
    ap.add_argument("--post-flash", action="store_true",
                    help="keep the flavor's post-flash chains (DSE, stream, pragma/latency opt); default: all off")
    ap.add_argument("--skills-json", default="",
                    help="skill pack for the flash prompt (a file name in the c2hls skills package, or a path); "
                         "default: the flavor's own pack")
    ap.add_argument("--zs-skills-block", default="",
                    help="with --flavor zero_shot: append this skill block (a recorded Expertise-rung injection) to the "
                         "zero-shot flash prompt (flash_refined/zs_skills.py)")
    args = ap.parse_args()
    if args.zs_skills_block and args.flavor != "zero_shot":
        raise SystemExit("--zs-skills-block needs --flavor zero_shot")
    if not args.shim_url:
        raise SystemExit("--shim-url (or CLAUDE_CODE_SHIM_URL) is required; start it with start_shim.sh")

    os.environ["C2HLS_AUTOSA_READY_ROOT"] = args.bench_root
    # c2hls re-detects the site from C2HLS_SITE when it is imported; without this export it falls
    # back to "team", where all-skills prompt modes silently degrade to 4 bottleneck skills.
    os.environ["C2HLS_SITE"] = "fir"
    from c2hls_paths import configure_site
    configure_site("fir")
    from autosa_flash_lib import apply_bench_timeouts_from_meta, apply_mm_flow_flavor, resolve_autosa_benches
    from batch_parallel_autosa_lib import configure_autosa_campaign_env

    flavor_info = apply_mm_flow_flavor(args.flavor)
    configure_autosa_campaign_env(os.environ["BATCH_PARALLEL_VARIANT"])
    (bench, bench_dir), = resolve_autosa_benches([args.bench], Path(args.bench_root))
    meta = json.loads((bench_dir / "metadata.json").read_text())
    apply_bench_timeouts_from_meta(meta)
    if not args.post_flash:
        for key in POST_FLASH_SWITCHES:
            os.environ[key] = "0"
    if args.dsp_redo:
        os.environ.update({"C2HLS_FLASH_DSP_REDO": "1", "C2HLS_FLASH_MIN_DSP": "300",
                           "C2HLS_FLASH_MAX_DSP": "9024", "C2HLS_FLASH_DSP_FILL_PCT": "90"})
    refined_info = None
    if args.refined:
        if os.environ.get("C2HLS_SKILL_MODE") != "skill_on":
            raise SystemExit("--refined needs a skills flavor (skills or aav_n_90)")
        sys.path.insert(0, str(HERE / "flash_refined"))
        import flash_refined
        hdr_path = bench_dir / (meta.get("header_file") or "kernel.h")
        hdr = hdr_path.read_text() if hdr_path.is_file() else ""   # non-mm benches: no I/J/K, balance gate MACs from env
        dims = {}
        for line in hdr.splitlines():
            m = __import__("re").match(r"\s*#\s*define\s+(I|J|K)\s+(\d+)\b", line)
            if m:
                dims[m.group(1)] = int(m.group(2))  # last active define wins (the 1024 block)
        if {"I", "J", "K"} <= set(dims):
            os.environ["FLASH_REFINED_TOTAL_MACS"] = str(dims["I"] * dims["J"] * dims["K"])
        refined_info = flash_refined.apply(version=args.refined_version, neutral=args.refined_neutral)
        if args.no_dsp_floor:
            for key in ("C2HLS_FLASH_DSP_REDO", "C2HLS_FLASH_MIN_DSP", "C2HLS_FLASH_DSP_FILL_PCT"):
                os.environ.pop(key, None)
            refined_info["dsp_floor"] = "off"
        if args.dsp_window:
            import dsp_window
            lo, hi = (int(x) for x in args.dsp_window.replace(":", ",").split(","))
            note = Path(args.stage_note_file).read_text().strip() if args.stage_note_file else ""
            refined_info["dsp_window"] = dsp_window.apply_window(lo, hi, stage_note=note)
    elif args.dsp_window or args.stage_note_file or args.no_dsp_floor or args.refined_neutral:
        raise SystemExit("--dsp-window / --stage-note-file / --no-dsp-floor need --refined")
    if args.seed_code:
        sys.path.insert(0, str(HERE / "flash_refined"))
        import dsp_window
        seed_info = dsp_window.apply_seed(str(Path(args.seed_code).resolve()))
        if refined_info is not None:
            refined_info["seed"] = seed_info
    zs_skills_info = None
    if args.zs_skills_block:
        sys.path.insert(0, str(HERE / "flash_refined"))
        import zs_skills
        zs_skills_info = zs_skills.apply(str(Path(args.zs_skills_block).resolve()))
    if args.skills_json:
        pack = Path(args.skills_json)
        if not pack.is_absolute():
            pack = C2HLS / "hls_full_optimization_skills_schema_1_1_package" / pack
        if not pack.is_file():
            raise SystemExit(f"skill pack not found: {pack}")
        if os.environ.get("C2HLS_SKILL_MODE") != "skill_on":
            raise SystemExit(f"flavor {args.flavor} runs without skills; --skills-json needs skills or aav_n_90")
        os.environ["C2HLS_PACKAGED_SKILLS_JSON"] = str(pack.resolve())
        os.environ["C2HLS_PACKAGED_SKILLS_ONLY"] = "1"
    # LLM endpoint last, so nothing from the site env or the flavor overrides it.
    os.environ["OPENAI_BASE_URL"] = args.shim_url.rstrip("/")
    os.environ["OPENAI_API_KEY"] = "EMPTY"
    os.environ["C2HLS_MODEL"] = args.model
    os.environ["C2HLS_LLM_TIMEOUT"] = os.environ.get("CC_LLM_TIMEOUT", "7200")
    os.environ.pop("C2HLS_BATCH_LLM_HOOK_MODULE", None)
    turns = args.turns if args.turns is not None else int(os.environ.get("C2HLS_TURNS", "3") or 3)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    cell = Path(args.out_root) / f"{args.flavor}{('_' + args.tag) if args.tag else ''}_{stamp}" / bench
    plan = {"post_flash": bool(args.post_flash), "bench": bench, "bench_dir": str(bench_dir), "cell_dir": str(cell), "model": args.model, "turns": turns,
            "flavor": flavor_info, "zs_skills": zs_skills_info, "env": {k: os.environ.get(k, "") for k in SHOWN_ENV}}
    print(json.dumps(plan, indent=2))
    base = args.shim_url.rstrip("/")
    # Claude Code shim: <url>/health; ChatHLS DeepSeek queue proxy: health at the host root.
    for health in (base + "/health", base.removesuffix("/v1") + "/health"):
        try:
            with urllib.request.urlopen(health, timeout=10) as r:
                print("endpoint:", r.read().decode())
            break
        except Exception as exc:  # noqa: BLE001
            err = exc
    else:
        raise SystemExit(f"LLM endpoint not reachable at {args.shim_url}: {err}")
    import c2hls
    from c2hls_paths import active_site
    effective_mode = c2hls._skill_prompt_mode()
    print(f"effective: site={active_site()} skill_prompt_mode={effective_mode}")
    wanted = os.environ.get("C2HLS_SKILL_PROMPT_MODE", "")
    if os.environ.get("C2HLS_SKILL_MODE") == "skill_on" and wanted and effective_mode != wanted:
        raise SystemExit(f"c2hls would use skill mode {effective_mode!r}, not {wanted!r}; refusing to run")
    plan["effective"] = {"site": active_site(), "skill_prompt_mode": effective_mode,
                         "refined": refined_info is not None, "refined_info": refined_info,
                         "skills_json": os.environ.get("C2HLS_PACKAGED_SKILLS_JSON", "")}
    if args.show_refined:
        import prompt_c2hls
        print("=== flash system prompt ===\n" + prompt_c2hls.llm_system_instruction(zero_shot=False, step_name="flash"))
        print("=== flash user prompt template ===\n" + prompt_c2hls.flash_optimization_prompt(zero_shot=False, skip_phase_b=False))
        sample = {"latency_ns": 25818000000.0, "dsp": 5, "bram": 30, "ff": 18633, "lut": 9731}
        print("=== flash resource block (sample report) ===\n"
              + c2hls._render_step_resource_constraints("flash", sample, os.environ.get("C2HLS_PART", "")))
        print("=== non-flash step block (unchanged) ===\n"
              + c2hls._render_step_resource_constraints("tiling", sample, os.environ.get("C2HLS_PART", ""))[:300])
    if args.dry_run:
        return 0

    cell.mkdir(parents=True, exist_ok=True)
    if args.controlled:
        sys.path.insert(0, str(HERE / "controlled"))
        os.environ["CTRL_LLM_LOG"] = str(cell / "ctrl_llm_calls.jsonl")
        import ctrl_llm
        plan["controlled"] = ctrl_llm.apply()
    (cell / "cc_run_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    from c2hls import run_benchmark_multistep
    t0 = time.time()
    try:
        result = run_benchmark_multistep(str(bench_dir), output_dir=str(cell), gpt_model=args.model,
                                         turns_limitation=turns, steps=None)
        status = "ok" if result.get("success") else "fail"
    except Exception as exc:  # noqa: BLE001
        result, status = {"success": False, "phase": "exception", "error": str(exc)}, "error"
    summary = {"status": status, "wallclock_s": round(time.time() - t0, 1), "cell_dir": str(cell),
               "phase": result.get("phase"), "error": result.get("error"),
               "steps": [{k: s.get(k) for k in ("step_name", "success", "error")} for s in (result.get("steps") or [])
                         if isinstance(s, dict)]}
    (cell / "cc_run_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps(summary, indent=2, default=str))
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
