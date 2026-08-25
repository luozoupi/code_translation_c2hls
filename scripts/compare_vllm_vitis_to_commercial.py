#!/usr/bin/env python3
"""Compare vLLM Vitis smoke results with commercial-model HLSFactory records."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
DEFAULT_COMMERCIAL = (
    REPO
    / "artifacts"
    / "hlsfactory_multistep_sonnet46_skill_on_website_revstyle_combined_20260615.jsonl"
)
DEFAULT_OUT_PREFIX = (
    REPO
    / "artifacts"
    / "vllm_vitis_smoke"
    / "vllm_vs_sonnet46_multistep_20260707"
)


def _bench_key(name: str) -> str:
    raw = str(name or "")
    raw = raw.removeprefix("hlsfactory_")
    return raw.replace("_", "-")


def _as_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    text = str(value).replace(",", "").strip()
    try:
        return int(float(text))
    except ValueError:
        return None


def _commercial_role(
    record: dict[str, Any],
    *,
    c2hls_any_variant_final: bool,
) -> str | None:
    impl = record.get("implementation") or {}
    origin = impl.get("origin")
    variant_name = (impl.get("variant") or {}).get("name")
    if origin == "hlsfactory_benchmark" and variant_name == "baseline":
        return "baseline"
    if origin == "c2hls_orchestrator" and (
        variant_name == "final" or c2hls_any_variant_final
    ):
        return "commercial_final"
    return None


def _load_commercial(
    path: Path,
    *,
    c2hls_any_variant_final: bool,
) -> dict[str, dict[str, dict[str, Any]]]:
    data: dict[str, dict[str, dict[str, Any]]] = {}
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            role = _commercial_role(
                record,
                c2hls_any_variant_final=c2hls_any_variant_final,
            )
            if not role:
                continue
            bench = _bench_key("/".join((record.get("problem") or {}).get("group_path") or []))
            if not bench:
                continue
            slot = data.setdefault(bench, {}).setdefault(role, {})
            report_type = record.get("report_type")
            if report_type == "hls_synth":
                synth = record.get("hls_synth") or {}
                latency = (
                    (synth.get("PerformanceEstimates") or {})
                    .get("SummaryOfOverallLatency", {})
                    .get("Worst-caseLatency")
                )
                cycles = _as_int(latency)
                resources = (synth.get("AreaEstimates") or {}).get("Resources") or {}
                slot["synth_status"] = synth.get("status")
                if cycles is not None or slot.get("synth_cycles") is None:
                    slot["synth_cycles"] = cycles
                for field, resource_key in (
                    ("bram", "BRAM_18K"),
                    ("dsp", "DSP"),
                    ("ff", "FF"),
                    ("lut", "LUT"),
                ):
                    value = _as_int(resources.get(resource_key))
                    if value is not None or slot.get(field) is None:
                        slot[field] = value
            elif report_type == "sw_run":
                slot["csim_status"] = (record.get("sw_run") or {}).get("status")
            elif report_type == "rtl_sim":
                rtl = record.get("rtl_sim") or {}
                slot["cosim_status"] = rtl.get("status")
                slot["cosim_cycles"] = _as_int(rtl.get("kernel_runtime_cycles"))
    return data


def _load_vllm_summary(path: Path) -> list[dict[str, Any]]:
    summary = json.loads(path.read_text())
    rows = []
    for row in summary.get("rows") or []:
        bench = _bench_key(row.get("benchmark"))
        synth = row.get("synth") or {}
        csim = row.get("csim") or {}
        rows.append(
            {
                "benchmark": bench,
                "vllm_source_summary": str(path),
                "vllm_synth_status": synth.get("status"),
                "vllm_csim_status": csim.get("status"),
                "vllm_cycles": _as_int(synth.get("latency_cycles")),
                "vllm_fmax_mhz": synth.get("fmax_mhz"),
                "vllm_bram": _as_int(synth.get("bram")),
                "vllm_dsp": _as_int(synth.get("dsp")),
                "vllm_ff": _as_int(synth.get("ff")),
                "vllm_lut": _as_int(synth.get("lut")),
                "vllm_error": synth.get("error") or csim.get("error") or "",
                "vllm_generated_code_path": row.get("generated_code_path"),
            }
        )
    return rows


def _ratio(numerator: int | None, denominator: int | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def build_comparison(
    commercial_path: Path,
    vllm_summaries: list[Path],
    *,
    c2hls_any_variant_final: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    commercial = _load_commercial(
        commercial_path,
        c2hls_any_variant_final=c2hls_any_variant_final,
    )
    rows = []
    seen = set()
    for summary_path in vllm_summaries:
        for vllm in _load_vllm_summary(summary_path):
            bench = vllm["benchmark"]
            if bench in seen:
                continue
            seen.add(bench)
            base = commercial.get(bench, {}).get("baseline", {})
            final = commercial.get(bench, {}).get("commercial_final", {})
            v_cycles = vllm.get("vllm_cycles")
            c_cycles = final.get("synth_cycles")
            b_cycles = base.get("synth_cycles")
            ratio_comm = _ratio(v_cycles, c_cycles)
            ratio_base = _ratio(v_cycles, b_cycles)
            note = ""
            if vllm.get("vllm_synth_status") != "pass":
                note = (vllm.get("vllm_error") or "vLLM synth did not pass")[:180]
            elif vllm.get("vllm_csim_status") != "pass":
                note = "vLLM synth passed but csim did not pass"
            elif ratio_comm is not None:
                note = "vLLM faster than commercial" if ratio_comm < 1.0 else "commercial faster than vLLM"
            rows.append(
                {
                    **vllm,
                    "commercial_final_synth_status": final.get("synth_status"),
                    "commercial_final_csim_status": final.get("csim_status"),
                    "commercial_final_cosim_status": final.get("cosim_status"),
                    "commercial_final_cycles": c_cycles,
                    "commercial_final_cosim_cycles": final.get("cosim_cycles"),
                    "commercial_final_bram": final.get("bram"),
                    "commercial_final_dsp": final.get("dsp"),
                    "commercial_final_ff": final.get("ff"),
                    "commercial_final_lut": final.get("lut"),
                    "hlsfactory_baseline_cycles": b_cycles,
                    "vllm_over_commercial_cycles": ratio_comm,
                    "vllm_over_hlsfactory_baseline_cycles": ratio_base,
                    "note": note,
                }
            )

    comparable = [
        row for row in rows
        if row.get("vllm_cycles") is not None and row.get("commercial_final_cycles") is not None
    ]
    ratios = [row["vllm_over_commercial_cycles"] for row in comparable if row.get("vllm_over_commercial_cycles") is not None]
    summary = {
        "commercial_path": str(commercial_path),
        "c2hls_any_variant_final": c2hls_any_variant_final,
        "vllm_summaries": [str(path) for path in vllm_summaries],
        "counts": {
            "rows": len(rows),
            "vllm_synth_pass": sum(1 for row in rows if row.get("vllm_synth_status") == "pass"),
            "vllm_csim_pass": sum(1 for row in rows if row.get("vllm_csim_status") == "pass"),
            "commercial_synth_pass": sum(1 for row in rows if row.get("commercial_final_synth_status") == "pass"),
            "commercial_csim_pass": sum(1 for row in rows if row.get("commercial_final_csim_status") == "pass"),
            "commercial_cosim_pass": sum(1 for row in rows if row.get("commercial_final_cosim_status") == "pass"),
            "cycle_comparable": len(comparable),
            "vllm_faster_than_commercial": sum(1 for row in comparable if row["vllm_over_commercial_cycles"] < 1.0),
            "commercial_faster_than_vllm": sum(1 for row in comparable if row["vllm_over_commercial_cycles"] > 1.0),
            "tie": sum(1 for row in comparable if row["vllm_over_commercial_cycles"] == 1.0),
        },
        "ratio_vllm_over_commercial_cycles": {
            "median": statistics.median(ratios) if ratios else None,
            "min": min(ratios) if ratios else None,
            "max": max(ratios) if ratios else None,
        },
    }
    return rows, summary


def write_outputs(rows: list[dict[str, Any]], summary: dict[str, Any], out_prefix: Path) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out_prefix.with_suffix(".csv")
    json_path = out_prefix.with_suffix(".summary.json")
    md_path = out_prefix.with_suffix(".md")
    fields = [
        "benchmark",
        "vllm_synth_status",
        "vllm_csim_status",
        "vllm_cycles",
        "commercial_final_synth_status",
        "commercial_final_csim_status",
        "commercial_final_cosim_status",
        "commercial_final_cycles",
        "commercial_final_cosim_cycles",
        "hlsfactory_baseline_cycles",
        "vllm_over_commercial_cycles",
        "vllm_over_hlsfactory_baseline_cycles",
        "vllm_bram",
        "vllm_dsp",
        "vllm_ff",
        "vllm_lut",
        "commercial_final_bram",
        "commercial_final_dsp",
        "commercial_final_ff",
        "commercial_final_lut",
        "note",
        "vllm_generated_code_path",
        "vllm_source_summary",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})

    json_path.write_text(json.dumps({**summary, "rows": rows}, indent=2, sort_keys=True) + "\n")

    lines = [
        "# vLLM vs Sonnet 4.6 Multistep",
        "",
        f"- Commercial reference: `{summary['commercial_path']}`",
        f"- Rows compared: `{summary['counts']['rows']}`",
        f"- vLLM synth/csim pass: `{summary['counts']['vllm_synth_pass']}` / `{summary['counts']['vllm_csim_pass']}`",
        f"- Commercial final synth/csim/cosim pass: `{summary['counts']['commercial_synth_pass']}` / `{summary['counts']['commercial_csim_pass']}` / `{summary['counts']['commercial_cosim_pass']}`",
        f"- Cycle-comparable rows: `{summary['counts']['cycle_comparable']}`",
        f"- vLLM faster than commercial by synth cycles: `{summary['counts']['vllm_faster_than_commercial']}`",
        f"- Commercial faster than vLLM by synth cycles: `{summary['counts']['commercial_faster_than_vllm']}`",
        "",
        "| bench | vLLM synth | vLLM csim | vLLM cycles | Sonnet synth | Sonnet csim | Sonnet cosim | Sonnet cycles | direct baseline cycles | vLLM/Sonnet | note |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        note = str(row.get("note") or "")
        if len(note) > 90:
            note = note[:87] + "..."
        lines.append(
            "| {benchmark} | {vsynth} | {vcsim} | {vcycles} | {csynth} | {ccsim} | {ccosim} | {ccycles} | {bcycles} | {ratio} | {note} |".format(
                benchmark=row.get("benchmark"),
                vsynth=row.get("vllm_synth_status") or "",
                vcsim=row.get("vllm_csim_status") or "",
                vcycles=_fmt(row.get("vllm_cycles")),
                csynth=row.get("commercial_final_synth_status") or "",
                ccsim=row.get("commercial_final_csim_status") or "",
                ccosim=row.get("commercial_final_cosim_status") or "",
                ccycles=_fmt(row.get("commercial_final_cycles")),
                bcycles=_fmt(row.get("hlsfactory_baseline_cycles")),
                ratio=_fmt(row.get("vllm_over_commercial_cycles")),
                note=note.replace("|", "/"),
            )
        )
    md_path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commercial-jsonl", type=Path, default=DEFAULT_COMMERCIAL)
    parser.add_argument("--vllm-summary", type=Path, action="append", required=True)
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_OUT_PREFIX)
    parser.add_argument(
        "--c2hls-any-variant-final",
        action="store_true",
        help="Treat any c2hls_orchestrator implementation variant as the commercial final record.",
    )
    args = parser.parse_args()
    rows, summary = build_comparison(
        args.commercial_jsonl,
        args.vllm_summary,
        c2hls_any_variant_final=args.c2hls_any_variant_final,
    )
    write_outputs(rows, summary, args.out_prefix)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
