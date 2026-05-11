"""Phase 2 end-to-end smoke on knn (rodinia-hls-nova benchmark).

Runs `c2hls.run_benchmark_multistep` twice on the same kernel:

1. **STATIC order**  — `C2HLS_DYNAMIC_ROUTING=0`, the existing
   tiling → pipeline → unroll → doublebuffer → coalescing progression.
2. **DYNAMIC routing** — `C2HLS_DYNAMIC_ROUTING=1`, Phase 2 behavior:
   the bottleneck-router consults Pillar 1 feedback after every step
   and consults the Pillar 3 skill library for the next move.

Both runs use:
- Vitis HLS 2025.2 (the only one installed) on `xc7a100t-csg324-1`
  (the part Phase 1 confirmed working). NOTE: this is NOT the same
  toolchain as `results/references_philip/` (Vitis 2023.2 + U280) —
  absolute numbers won't match, but trajectory shape and ratios will.
- claude-haiku-4-5-20251001 for the agent calls.
- Multistep with all 5 default steps; csynth + csim where available
  (knn supports csim, not cosim).

After both runs finish, the existing `dataset_pipeline.replay_existing_results`
upgrades the saved `*_multistep_results.json` into v2 trajectory jsonls,
and `merge_with_references` joins them with philip's reference jsonls.
The script writes a markdown artifact summarizing what changed between
static and dynamic, and how each compares to the reference.

Usage:
    cd /home/luo00466/code_translation-c2hls
    python tests/test_phase2_e2e_knn.py [--out artifacts/phase2_e2e_knn_<ts>.md]
                                         [--bench knn]
                                         [--turns 2]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent

# Set environment BEFORE importing c2hls so DEFAULT_PART / clock pick them up.
os.environ.setdefault("C2HLS_PART", "xc7a100t-csg324-1")
os.environ.setdefault("C2HLS_CLOCK_NS", "4.0")
os.environ.setdefault(
    "C2HLS_VITIS_SETTINGS",
    "/mnt/data/luo00466/Xilinx/2025.2/Vitis/settings64.sh",
)
os.environ.setdefault("C2HLS_VITIS_VERSION", "2025.2")

sys.path.insert(0, str(REPO_ROOT))


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )


def _run_one(*, bench_dir: Path, output_dir: Path,
             dynamic_routing: bool, model: str, turns: int) -> Dict[str, Any]:
    """Run one multistep pass and return a summary."""
    os.environ["C2HLS_DYNAMIC_ROUTING"] = "1" if dynamic_routing else "0"
    output_dir.mkdir(parents=True, exist_ok=True)

    from c2hls import run_benchmark_multistep   # noqa: WPS433

    label = "dynamic" if dynamic_routing else "static"
    logging.info("=== knn / %s order — starting (output_dir=%s) ===",
                 label, output_dir)
    t0 = time.time()
    result = run_benchmark_multistep(
        str(bench_dir),
        output_dir=str(output_dir),
        gpt_model=model,
        turns_limitation=turns,
    )
    wall = time.time() - t0
    logging.info("=== knn / %s order — done in %.1fs (success=%s) ===",
                 label, wall, result.get("success"))
    summary = {
        "mode": label,
        "wall_seconds": round(wall, 1),
        "success": bool(result.get("success")),
        "phase": result.get("phase"),
        "error": result.get("error", ""),
        "step_summaries": [],
    }
    for step in result.get("steps") or []:
        sname = step.get("step_name", "?")
        rep = step.get("report") or step.get("rejected_report") or {}
        rd = step.get("routing_decision") or {}
        summary["step_summaries"].append({
            "step_name": sname,
            "success": bool(step.get("success")),
            "step_effect": step.get("step_effect"),
            "error": step.get("error", "")[:200],
            "latency_cycles": rep.get("latency_cycles"),
            "latency_ns": rep.get("latency_ns"),
            "interval": rep.get("interval"),
            "bram": rep.get("bram"),
            "dsp": rep.get("dsp"),
            "ff": rep.get("ff"),
            "lut": rep.get("lut"),
            "fmax_mhz": rep.get("fmax_mhz"),
            "routing_reason": rd.get("reason"),
            "skill_id": rd.get("skill_id"),
            "warnings": step.get("warnings") or [],
        })
    summary["robustness_log"] = result.get("robustness_log") or []
    return summary


def _build_comparison(static_summary: Dict[str, Any],
                       dynamic_summary: Dict[str, Any],
                       reference_jsonl: List[str],
                       v2_jsonl: Path,
                       merged_jsonl: Path,
                       artifact_path: Path,
                       run_meta: Any) -> None:
    """Write the markdown comparison artifact."""
    from dataset_pipeline import (
        replay_existing_results,
        merge_with_references,
    )
    from dataset_pipeline.schema import RunMeta

    # 1. Replay both result trees into one v2 jsonl.
    summary = replay_existing_results(
        results_dirs=[
            str(REPO_ROOT / "results_phase2"),
        ],
        output_jsonl=str(v2_jsonl),
        run_meta=run_meta,
        origin_version="phase2-e2e-knn",
    )

    # 2. Merge with philip's references.
    merged = merge_with_references(
        generated_jsonl=str(v2_jsonl),
        reference_paths=[p for p in reference_jsonl if Path(p).exists()],
        output_jsonl=str(merged_jsonl),
    )

    # 3. Pull reference numbers for knn from the merged stream so we
    # can compare side-by-side.
    ref_by_step: Dict[str, Dict[str, Any]] = {}
    if Path(merged_jsonl).exists():
        with merged_jsonl.open() as f:
            for line in f:
                rec = json.loads(line)
                impl = rec.get("implementation") or {}
                if impl.get("origin", "").startswith("rodinia"):
                    p = rec.get("problem", {})
                    if "knn" in (p.get("group_path") or []):
                        v = rec.get("variant") or {}
                        rt = rec.get("report_type")
                        ref_by_step[(rt, v.get("name", "?"))] = rec

    # 4. Render markdown.
    lines: List[str] = []
    lines.append(f"# Phase 2 End-to-End: knn (multistep, static vs dynamic)\n")
    lines.append(f"- generated: {_dt.datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- v2 trajectory: `{v2_jsonl}`  ({summary['records_written']} records)")
    lines.append(f"- merged with refs: `{merged_jsonl}`  ({merged['merged_records']} total, "
                 f"{merged['reference_records']} ref + {merged['generated_records']} gen)")
    lines.append("")
    lines.append("Both modes ran on `benchmarks/knn/`, Vitis HLS 2025.2 on "
                 "`xc7a100t-csg324-1` at 4 ns clock, model `claude-haiku-4-5-20251001`. "
                 "Philip's reference jsonl was generated on **Vitis 2023.2 + xilinx_u280** — "
                 "different toolchain and FPGA, so absolute numbers should be compared "
                 "as ratios, not parities.")
    lines.append("")

    # Mode summaries
    for s in (static_summary, dynamic_summary):
        lines.append(f"## {s['mode']} mode summary")
        lines.append("")
        lines.append(f"- success: {s['success']}")
        lines.append(f"- wall: {s['wall_seconds']}s")
        if s.get("error"):
            lines.append(f"- error: `{s['error'][:200]}`")
        if s.get("robustness_log"):
            lines.append("- robustness events:")
            for ev in s["robustness_log"]:
                lines.append(f"  - `{ev}`")
        lines.append("")
        lines.append("| step | success | effect | lat_cyc | ii | BRAM | DSP | FF | LUT | Fmax | routing |")
        lines.append("|------|:-------:|--------|--------:|---:|-----:|----:|---:|---:|-----:|---------|")
        for ss in s["step_summaries"]:
            routing = ss.get("routing_reason", "") or ""
            lines.append(
                f"| {ss['step_name']} | {ss['success']} | {ss.get('step_effect') or '-'} | "
                f"{ss.get('latency_cycles', '-')} | {ss.get('interval', '-')} | "
                f"{ss.get('bram', '-')} | {ss.get('dsp', '-')} | {ss.get('ff', '-')} | "
                f"{ss.get('lut', '-')} | {ss.get('fmax_mhz', '-')} | "
                f"`{(routing[:60] + '…') if len(routing) > 60 else routing}` |"
            )
        lines.append("")

    # Static vs Dynamic head-to-head
    lines.append("## Static vs Dynamic head-to-head\n")
    s_steps = {s["step_name"]: s for s in static_summary["step_summaries"]}
    d_steps = {s["step_name"]: s for s in dynamic_summary["step_summaries"]}
    all_steps = list(dict.fromkeys(
        [s["step_name"] for s in static_summary["step_summaries"]]
        + [s["step_name"] for s in dynamic_summary["step_summaries"]]
    ))
    lines.append("| step | static lat | dynamic lat | Δlatency | static effect | dynamic effect |")
    lines.append("|------|-----------:|------------:|---------:|---------------|----------------|")
    for st in all_steps:
        s = s_steps.get(st, {})
        d = d_steps.get(st, {})
        s_lat = s.get("latency_cycles")
        d_lat = d.get("latency_cycles")
        delta = ""
        if s_lat and d_lat:
            try:
                ratio = float(d_lat) / float(s_lat)
                delta = f"{ratio:.3f}x"
            except (TypeError, ValueError):
                pass
        lines.append(
            f"| {st} | {s_lat or '-'} | {d_lat or '-'} | {delta or '-'} | "
            f"{s.get('step_effect') or '-'} | {d.get('step_effect') or '-'} |"
        )
    lines.append("")

    # Reference comparison
    lines.append("## vs philip's reference jsonl (Vitis 2023.2 / U280)\n")
    if not ref_by_step:
        lines.append("_no reference rows tagged for `knn` were located in the merged stream._\n")
    else:
        lines.append("Reference rows tagged for `knn` (csynth report_type, by step name):\n")
        lines.append("| step | ref lat_cyc | ref ii | ref BRAM | ref DSP | ref FF | ref LUT |")
        lines.append("|------|-----------:|-------:|---------:|--------:|-------:|--------:|")
        for (rt, vname), rec in sorted(ref_by_step.items()):
            if rt != "hls_synth":
                continue
            hs = rec.get("hls_synth") or {}
            lines.append(
                f"| {vname} | {hs.get('latency_cycles', '-')} | "
                f"{hs.get('interval', '-')} | {hs.get('bram', '-')} | "
                f"{hs.get('dsp', '-')} | {hs.get('ff', '-')} | "
                f"{hs.get('lut', '-')} |"
            )
        lines.append("")
        lines.append("Disclaimer: reference and our runs use different (Vitis version, FPGA) "
                     "cells. Compare *trends* (do larger trip counts produce larger latency? "
                     "Does pipeline reduce II?) rather than absolute magnitudes.")
    lines.append("")
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text("\n".join(lines), encoding="utf-8")
    logging.info("artifact written: %s", artifact_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    parser.add_argument("--out", type=Path,
                        default=REPO_ROOT / "artifacts" / f"phase2_e2e_knn_{timestamp}.md")
    parser.add_argument("--bench", default="knn")
    parser.add_argument("--model", default="claude-haiku-4-5-20251001")
    parser.add_argument("--turns", type=int, default=2)
    parser.add_argument("--skip-static", action="store_true")
    parser.add_argument("--skip-dynamic", action="store_true")
    args = parser.parse_args()

    _setup_logging()

    bench_dir = REPO_ROOT / "benchmarks" / args.bench
    if not bench_dir.is_dir():
        logging.error("bench dir missing: %s", bench_dir)
        return 2

    out_static = REPO_ROOT / "results_phase2" / f"{args.bench}_static"
    out_dynamic = REPO_ROOT / "results_phase2" / f"{args.bench}_dynamic"

    static_summary = {"mode": "static", "wall_seconds": 0.0,
                       "success": False, "step_summaries": [],
                       "robustness_log": [], "error": "skipped"}
    dynamic_summary = {"mode": "dynamic", "wall_seconds": 0.0,
                        "success": False, "step_summaries": [],
                        "robustness_log": [], "error": "skipped"}

    if not args.skip_static:
        static_summary = _run_one(
            bench_dir=bench_dir, output_dir=out_static,
            dynamic_routing=False, model=args.model, turns=args.turns,
        )
    if not args.skip_dynamic:
        dynamic_summary = _run_one(
            bench_dir=bench_dir, output_dir=out_dynamic,
            dynamic_routing=True, model=args.model, turns=args.turns,
        )

    # Save raw mode summaries next to the markdown artifact for later
    # programmatic inspection.
    raw_path = args.out.with_suffix(".json")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps({
        "static": static_summary,
        "dynamic": dynamic_summary,
        "env": {
            "C2HLS_PART": os.environ.get("C2HLS_PART"),
            "C2HLS_CLOCK_NS": os.environ.get("C2HLS_CLOCK_NS"),
            "C2HLS_VITIS_VERSION": os.environ.get("C2HLS_VITIS_VERSION"),
            "model": args.model,
            "turns": args.turns,
        },
    }, indent=2), encoding="utf-8")

    from dataset_pipeline.schema import RunMeta
    run_meta = RunMeta(
        target="vitis.csynth",
        vitis_version=os.environ.get("C2HLS_VITIS_VERSION", "2025.2"),
        device=os.environ.get("C2HLS_PART", "xc7a100t-csg324-1"),
        flow_target="vitis",
        clock_ns=float(os.environ.get("C2HLS_CLOCK_NS", "4.0")),
    )
    v2_jsonl = REPO_ROOT / "artifacts" / f"phase2_e2e_knn_v2_{timestamp}.jsonl"
    merged_jsonl = REPO_ROOT / "artifacts" / f"phase2_e2e_knn_merged_{timestamp}.jsonl"
    refs = [
        str(REPO_ROOT / "csynth_vitis_2023.2__device_xilinx_u280_gen3x16_xdma_1_202211_1.jsonl"),
        str(REPO_ROOT / "results/references_philip/hw_emu_vitis_2023.2"
            "__device_xilinx_u280_gen3x16_xdma_1_202211_1.jsonl"),
        str(REPO_ROOT / "results/references_philip/sw_emu_vitis_2023.2"
            "__device_xilinx_u280_gen3x16_xdma_1_202211_1.jsonl"),
    ]
    _build_comparison(
        static_summary, dynamic_summary,
        reference_jsonl=refs,
        v2_jsonl=v2_jsonl, merged_jsonl=merged_jsonl,
        artifact_path=args.out,
        run_meta=run_meta,
    )

    print(f"\nartifact: {args.out}")
    print(f"raw    : {raw_path}")
    print(f"v2     : {v2_jsonl}")
    print(f"merged : {merged_jsonl}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
