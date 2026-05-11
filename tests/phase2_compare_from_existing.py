"""Build the static-vs-dynamic-vs-reference comparison artifact from
whatever already exists on disk under `results_phase2/`. Lets us produce
a partial report any time without waiting for both runs to finish.

Adds the retroactive Phase 2 verdict on philip's knn reference (no-op
trap and throughput-regression checks applied to the ground-truth data
itself), so the artifact has substance even if neither agentic run has
finished.

Usage:
    python tests/phase2_compare_from_existing.py [--out artifacts/phase2_compare_<ts>.md]
                                                  [--bench knn]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from c2hls import _step_no_op_reasons   # noqa: E402
from robustness import throughput_regression_check  # noqa: E402


def _load_multistep_results(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _agentic_step_rows(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Pull per-step numbers from a saved `*_multistep_results.json`."""
    rows: List[Dict[str, Any]] = []
    if not payload:
        return rows
    baseline = payload.get("baseline_report") or {}
    rows.append({
        "step_name": "baseline",
        "success": True,
        "latency_cycles": baseline.get("latency_cycles"),
        "latency_ns": baseline.get("latency_ns"),
        "interval": baseline.get("interval"),
        "bram": baseline.get("bram"),
        "dsp": baseline.get("dsp"),
        "ff": baseline.get("ff"),
        "lut": baseline.get("lut"),
        "fmax_mhz": baseline.get("fmax_mhz"),
        "step_effect": "baseline",
        "routing_decision": None,
    })
    for s in payload.get("steps") or []:
        rep = s.get("report") or s.get("rejected_report") or {}
        rd = s.get("routing_decision") or {}
        rows.append({
            "step_name": s.get("step_name"),
            "success": bool(s.get("success")),
            "error": s.get("error", ""),
            "reverted_to_prev": bool(s.get("reverted_to_prev")),
            "latency_cycles": rep.get("latency_cycles"),
            "latency_ns": rep.get("latency_ns"),
            "interval": rep.get("interval"),
            "bram": rep.get("bram"),
            "dsp": rep.get("dsp"),
            "ff": rep.get("ff"),
            "lut": rep.get("lut"),
            "fmax_mhz": rep.get("fmax_mhz"),
            "step_effect": s.get("step_effect"),
            "routing_decision": rd,
            "warnings": s.get("warnings") or [],
        })
    return rows


def _phase2_verdict_for_steps(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Apply Phase 2 robustness checks to the row sequence."""
    verdicts = []
    prev = None
    prev_name = None
    for r in rows:
        cur = {k: r.get(k) for k in (
            "latency_cycles", "interval", "bram", "dsp", "ff", "lut", "fmax_mhz")}
        ev = []
        if prev is not None:
            no_op = _step_no_op_reasons(cur, prev)
            if no_op:
                ev.append({"kind": "no_op",
                           "reason": no_op[-1] if len(no_op) > 1 else no_op[0]})
        tp = throughput_regression_check(cur, prev)
        if tp.flagged:
            ev.append({"kind": "throughput_regression", "reasons": tp.reasons})
        verdicts.append({
            "step": r["step_name"],
            "events": ev,
        })
        if r.get("success") and not r.get("reverted_to_prev"):
            prev = cur
            prev_name = r["step_name"]
    return verdicts


def _load_reference_rows(bench: str) -> List[Dict[str, Any]]:
    """Pull per-step rows from philip's reference jsonls for one bench."""
    sources = {
        "csynth": REPO_ROOT / "csynth_vitis_2023.2__device_xilinx_u280_gen3x16_xdma_1_202211_1.jsonl",
        "hw_emu": REPO_ROOT / "results/references_philip/hw_emu_vitis_2023.2__device_xilinx_u280_gen3x16_xdma_1_202211_1.jsonl",
        "sw_emu": REPO_ROOT / "results/references_philip/sw_emu_vitis_2023.2__device_xilinx_u280_gen3x16_xdma_1_202211_1.jsonl",
    }
    rows: List[Dict[str, Any]] = []
    for kind, path in sources.items():
        if not path.exists():
            continue
        for line in path.open():
            rec = json.loads(line)
            problem = rec.get("problem") or {}
            if bench not in (problem.get("group_path") or []):
                continue
            impl = rec.get("implementation") or {}
            v = rec.get("variant") or impl.get("variant") or {}
            rt = rec.get("report_type")
            row = {"kind": kind, "variant": v.get("name"), "report_type": rt}
            if rt == "hls_synth":
                hs = rec.get("hls_synth", {}) or {}
                perf = hs.get("PerformanceEstimates", {}) or {}
                ar = (hs.get("AreaEstimates", {}) or {}).get("Resources", {}) or {}
                lat = perf.get("SummaryOfOverallLatency", {}) or {}
                row.update({
                    "latency_avg": _to_int(lat.get("Average-caseLatency")),
                    "latency_worst": _to_int(lat.get("Worst-caseLatency")),
                    "interval_max": _to_int(lat.get("Interval-max")),
                    "BRAM": _to_int(ar.get("BRAM_18K")),
                    "DSP": _to_int(ar.get("DSP")),
                    "FF": _to_int(ar.get("FF")),
                    "LUT": _to_int(ar.get("LUT")),
                })
            elif rt == "rtl_sim":
                rs = rec.get("rtl_sim", {}) or {}
                row.update({
                    "status": rs.get("status"),
                    "kernel_runtime_cycles": rs.get("kernel_runtime_cycles"),
                    "kernel_runtime_us": rs.get("kernel_runtime_us"),
                })
            rows.append(row)
    return rows


def _to_int(v: Any) -> Optional[int]:
    try:
        return int(float(v)) if v is not None else None
    except (TypeError, ValueError):
        return None


def _retroactive_phase2_on_reference(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Apply Phase 2 checks to the reference csynth trajectory."""
    csynth = [r for r in rows if r.get("kind") == "csynth" and r.get("report_type") == "hls_synth"]
    canonical_order = ["baseline", "tiling", "pipeline", "unroll",
                        "doublebuffer", "coalescing"]
    # Reference rows are in the canonical step order in the jsonl already,
    # but use the variant name to be safe.
    by_name = {r["variant"]: r for r in csynth if r.get("variant")}
    verdicts = []
    prev = None
    for name in canonical_order:
        r = by_name.get(name)
        if not r:
            continue
        cur = {
            "latency_cycles": r.get("latency_avg"),
            "interval": r.get("interval_max"),
            "bram": r.get("BRAM"), "dsp": r.get("DSP"),
            "ff": r.get("FF"), "lut": r.get("LUT"),
        }
        ev = []
        if prev is not None:
            no_op = _step_no_op_reasons(cur, prev)
            if no_op:
                ev.append({"kind": "no_op",
                           "reason": no_op[-1] if len(no_op) > 1 else no_op[0]})
        tp = throughput_regression_check(cur, prev)
        if tp.flagged:
            ev.append({"kind": "throughput_regression", "reasons": tp.reasons})
        verdicts.append({"step": name, "metrics": cur, "events": ev})
        prev = cur
    return verdicts


def _render_step_table(rows: List[Dict[str, Any]]) -> str:
    lines = [
        "| step | success | effect | lat_cyc | latency_ns | ii | BRAM | DSP | FF | LUT | Fmax | routing |",
        "|------|:-------:|--------|--------:|-----------:|---:|-----:|----:|---:|---:|-----:|---------|",
    ]
    for r in rows:
        rd = r.get("routing_decision") or {}
        reason = rd.get("reason") or ""
        if len(reason) > 60:
            reason = reason[:57] + "…"
        skill = rd.get("skill_id") or ""
        routing = f"`{reason}`" if reason else "-"
        if skill:
            routing = f"`{skill}` ← {routing}" if routing != "-" else f"`{skill}`"
        eff = r.get("step_effect") or ("-" if r.get("step_name") == "baseline" else (
            "reverted" if r.get("reverted_to_prev") else "-"))
        lines.append(
            f"| {r.get('step_name', '?')} | {r.get('success', '?')} | {eff} | "
            f"{r.get('latency_cycles', '-')} | {r.get('latency_ns', '-')} | "
            f"{r.get('interval', '-')} | {r.get('bram', '-')} | "
            f"{r.get('dsp', '-')} | {r.get('ff', '-')} | {r.get('lut', '-')} | "
            f"{r.get('fmax_mhz', '-')} | {routing} |"
        )
    return "\n".join(lines)


def _render_phase2_verdicts(verdicts: List[Dict[str, Any]]) -> str:
    if not verdicts:
        return "_no verdicts (no run data yet)_"
    lines = [
        "| step | events |",
        "|------|--------|",
    ]
    for v in verdicts:
        if v["events"]:
            ev_text = "; ".join(
                e.get("kind", "?") + (
                    f": {e.get('reason') or '/'.join(e.get('reasons', []))[:80]}"
                ) for e in v["events"]
            )
        else:
            ev_text = "clean"
        lines.append(f"| {v['step']} | {ev_text[:200]} |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    parser.add_argument("--out", type=Path,
                        default=REPO_ROOT / "artifacts" / f"phase2_compare_{timestamp}.md")
    parser.add_argument("--bench", default="knn")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    static_payload = _load_multistep_results(
        REPO_ROOT / "results_phase2" / f"{args.bench}_static" /
        f"{args.bench}_multistep_results.json"
    )
    dynamic_payload = _load_multistep_results(
        REPO_ROOT / "results_phase2" / f"{args.bench}_dynamic" /
        f"{args.bench}_multistep_results.json"
    )
    combo_full_payload = _load_multistep_results(
        REPO_ROOT / "results_phase2" / f"{args.bench}_combo_full" /
        f"{args.bench}_multistep_results.json"
    )
    static_rows = _agentic_step_rows(static_payload) if static_payload else []
    dynamic_rows = _agentic_step_rows(dynamic_payload) if dynamic_payload else []
    combo_full_rows = _agentic_step_rows(combo_full_payload) if combo_full_payload else []

    static_verdicts = _phase2_verdict_for_steps(static_rows)
    dynamic_verdicts = _phase2_verdict_for_steps(dynamic_rows)
    combo_full_verdicts = _phase2_verdict_for_steps(combo_full_rows)

    ref_rows = _load_reference_rows(args.bench)
    ref_verdicts = _retroactive_phase2_on_reference(ref_rows)

    # Build markdown
    md: List[str] = []
    md.append(f"# Phase 2 End-to-End: {args.bench}\n")
    md.append(f"_generated {_dt.datetime.now().isoformat(timespec='seconds')} "
              f"from existing on-disk artifacts._\n")
    md.append("- Vitis HLS 2025.2 / `xc7a100t-csg324-1` / 4 ns clock")
    md.append("- agent: claude-haiku-4-5-20251001")
    md.append("- philip's reference: Vitis 2023.2 / xilinx_u280 — different (toolchain, FPGA), "
              "compare ratios not absolutes\n")
    md.append("")

    md.append("## Static-order run (current production behavior)\n")
    if static_payload:
        md.append(_render_step_table(static_rows))
        md.append("\n### Phase 2 verdict on static run trajectory\n")
        md.append(_render_phase2_verdicts(static_verdicts))
    else:
        md.append("_no static results on disk yet — run still in progress or not started._")
    md.append("")

    md.append("## Dynamic-routing run (Phase 2 behavior)\n")
    if dynamic_payload:
        md.append(_render_step_table(dynamic_rows))
        md.append("\n### Phase 2 verdict on dynamic run trajectory\n")
        md.append(_render_phase2_verdicts(dynamic_verdicts))
    else:
        md.append("_no dynamic results on disk yet — run still in progress or not started._")
    md.append("")

    md.append("## combo_full run (Phase 3 single-shot all-in-one)\n")
    if combo_full_payload:
        md.append(_render_step_table(combo_full_rows))
        md.append("\n### Phase 2 verdict on combo_full run trajectory\n")
        md.append(_render_phase2_verdicts(combo_full_verdicts))
    else:
        md.append("_no combo_full results on disk yet._")
    md.append("")

    md.append("## Reference (philip's knn jsonl) — for context\n")
    if ref_rows:
        # Render the canonical sequence
        md.append("| step | latency_avg | latency_worst | ii_max | BRAM | DSP | FF | LUT |")
        md.append("|------|------------:|--------------:|-------:|-----:|----:|---:|---:|")
        order = ["baseline", "tiling", "pipeline", "unroll", "doublebuffer", "coalescing"]
        by_name = {r["variant"]: r for r in ref_rows
                   if r.get("kind") == "csynth" and r.get("report_type") == "hls_synth"}
        for name in order:
            r = by_name.get(name)
            if not r:
                continue
            md.append(
                f"| {name} | {r.get('latency_avg', '-')} | "
                f"{r.get('latency_worst', '-')} | "
                f"{r.get('interval_max', '-')} | {r.get('BRAM', '-')} | "
                f"{r.get('DSP', '-')} | {r.get('FF', '-')} | {r.get('LUT', '-')} |"
            )
        md.append("\n### Retroactive Phase 2 verdict on reference\n")
        md.append("Applies Pillar 9's no-op trap and throughput-regression checks to "
                  "philip's reference trajectory. Surfaces ground-truth-side issues "
                  "the Phase 2 hooks were designed to catch:\n")
        md.append(_render_phase2_verdicts(ref_verdicts))
        md.append("")
        # hw_emu rows
        hw_rows = [r for r in ref_rows if r.get("kind") == "hw_emu"]
        if hw_rows:
            md.append("\n### hw_emu reference (for reference cycle counts)\n")
            md.append("| variant | status | runtime_cycles | runtime_us |")
            md.append("|---------|:------:|---------------:|-----------:|")
            by_name_hw = {r["variant"]: r for r in hw_rows}
            for name in order:
                r = by_name_hw.get(name)
                if not r:
                    continue
                md.append(
                    f"| {name} | {r.get('status', '?')} | "
                    f"{r.get('kernel_runtime_cycles', '-')} | "
                    f"{r.get('kernel_runtime_us', '-')} |"
                )
    else:
        md.append("_no reference rows located._")

    md.append("")
    md.append("## Strategy head-to-head: best PPA achieved per run\n")
    md.append("| run | strategy | best lat_cyc | best lat_ns | wall observed |")
    md.append("|-----|----------|-------------:|------------:|---------------|")
    def _best_row(rows):
        best = None
        for r in rows:
            if not r.get("success") or r.get("reverted_to_prev"):
                continue
            lc = r.get("latency_cycles")
            if lc is None: continue
            if best is None or (best.get("latency_cycles") or 1e18) > lc:
                best = r
        return best
    s_best = _best_row(static_rows)
    d_best = _best_row(dynamic_rows)
    cf_best = _best_row(combo_full_rows)
    md.append(f"| static | tiling→pipeline→… | "
              f"{s_best.get('latency_cycles') if s_best else '-'} | "
              f"{s_best.get('latency_ns') if s_best else '-'} | 64.8 min |")
    md.append(f"| dynamic | bottleneck-routed (Phase 2) | "
              f"{d_best.get('latency_cycles') if d_best else '-'} | "
              f"{d_best.get('latency_ns') if d_best else '-'} | 66.4 min |")
    md.append(f"| combo_full | single-shot all-in-one (Phase 3) | "
              f"{cf_best.get('latency_cycles') if cf_best else '-'} | "
              f"{cf_best.get('latency_ns') if cf_best else '-'} | _see run log_ |")

    md.append("")
    args.out.write_text("\n".join(md), encoding="utf-8")
    print(f"artifact written: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
