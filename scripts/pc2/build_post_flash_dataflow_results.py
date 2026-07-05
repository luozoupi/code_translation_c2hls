#!/usr/bin/env python3
"""Build post-flash DATAFLOW results tree: kernel bundle + csynth logs + analysis MD.

Creates::

    <results-root>/
      kernel_bundle/hlsfactory_<bench>/{flash,dataflow}/kernel.cpp
      kernel_bundle/hlsfactory_<bench>/{flash_csynth,dataflow_csynth}/...
      reports/post_flash_dataflow_results.md
      reports/metrics.json
      run_summary.json

Example::

    python3 scripts/pc2/build_post_flash_dataflow_results.py \\
        --matrix-root artifacts/pc2/flash_all_new_skills_avoids_global_20260623_024548 \\
        --summary post_flash_dataflow_summary_20260704_052531.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "pc2"))

from export_post_flash_dataflow_csynth_bundle import export_csynth_bundle
from post_flash_dataflow import artifact_paths, resolve_selected_kernel
from post_flash_mem_parallel import discover_matrix_cells


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")


def _parse_int(text: Optional[str]) -> Optional[int]:
    if text is None:
        return None
    s = str(text).strip().replace(",", "")
    if not s or s.lower() in {"", "undef", "-", "n/a", "na"}:
        return None
    try:
        return int(float(s))
    except (TypeError, ValueError):
        return None


def _latency_triple_from_csynth(xml_path: Path) -> dict[str, Optional[int]]:
    out: dict[str, Optional[int]] = {
        "best": None,
        "avg": None,
        "worst": None,
        "interval_min": None,
        "interval_max": None,
    }
    if not xml_path.is_file():
        return out
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return out
    lat = root.find(".//PerformanceEstimates/SummaryOfOverallLatency")
    if lat is None:
        return out
    out["best"] = _parse_int(lat.findtext("Best-caseLatency"))
    out["avg"] = _parse_int(lat.findtext("Average-caseLatency"))
    out["worst"] = _parse_int(lat.findtext("Worst-caseLatency"))
    out["interval_min"] = _parse_int(lat.findtext("Interval-min"))
    out["interval_max"] = _parse_int(lat.findtext("Interval-max"))
    return out


def _resources_from_csynth(xml_path: Path) -> dict[str, Optional[int]]:
    out = {k: None for k in ("bram", "dsp", "ff", "lut", "uram")}
    if not xml_path.is_file():
        return out
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return out
    resources = root.find(".//AreaEstimates/Resources")
    if resources is None:
        return out
    mapping = {
        "bram": "BRAM_18K",
        "dsp": "DSP",
        "ff": "FF",
        "lut": "LUT",
        "uram": "URAM",
    }
    for key, tag in mapping.items():
        out[key] = _parse_int(resources.findtext(tag))
    return out


def _max_final_ii(log_path: Path) -> Optional[int]:
    if not log_path.is_file():
        return None
    text = log_path.read_text(encoding="utf-8", errors="replace")
    iis = [
        int(m.group(1))
        for m in re.finditer(
            r"Target II = \d+, Final II = (\d+)",
            text,
        )
    ]
    return max(iis) if iis else None


def _speedup(base: Optional[int], opt: Optional[int]) -> Optional[float]:
    if base is None or opt is None or opt <= 0:
        return None
    return base / opt


def _geom_mean(values: list[float]) -> Optional[float]:
    if not values:
        return None
    return math.exp(sum(math.log(v) for v in values) / len(values))


def _fmt_cycles(v: Optional[int]) -> str:
    if v is None:
        return "undef"
    return f"{v:,}"


def _fmt_speedup(v: Optional[float]) -> str:
    if v is None:
        return "n/a"
    return f"{v:.2f}×"


def _fmt_pct_change(base: Optional[int], opt: Optional[int]) -> str:
    if base is None or opt is None or base == 0:
        return "n/a"
    pct = (opt - base) / base * 100.0
    sign = "+" if pct > 0 else ""
    return f"{sign}{pct:.1f}%"


def _copy_kernel(src: Path, dest: Path, force: bool) -> bool:
    if not src.is_file():
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        return True
    shutil.copy2(src, dest)
    return True


def _resolve_old_dataflow_kernel(old_bundle: Path, bench: str) -> Optional[Path]:
    for rel in (
        f"{bench}/dataflow.cpp",
        f"{bench}/dataflow/kernel.cpp",
    ):
        path = old_bundle / rel
        if path.is_file():
            return path
    return None


def _kernel_code_audit(code: str) -> dict[str, Any]:
    bundles = sorted(set(re.findall(r"bundle=(gmem\d*)", code)))
    distinct_gmemn = sorted({
        b for b in bundles if re.fullmatch(r"gmem\d+", b)
    })
    loop_labels = len(re.findall(r"^\s*\w+:\s*for\s*\(", code, re.MULTILINE))
    shared_gmem_only = bundles == ["gmem"] or (
        "gmem" in bundles and not distinct_gmemn
    )
    return {
        "m_axi_bundles": bundles,
        "distinct_gmemn": distinct_gmemn,
        "distinct_gmemn_count": len(distinct_gmemn),
        "loop_label_count": loop_labels,
        "shared_gmem_only": shared_gmem_only,
    }


def build_results(
    *,
    matrix_root: Path,
    flash_bundle_root: Path,
    results_root: Path,
    summary_path: Optional[Path],
    old_kernel_bundle: Optional[Path] = None,
    prompt_policy: str = "system_skills",
    force: bool = False,
) -> dict[str, Any]:
    from post_flash_dataflow import kernel_bundle_dir_name, resolve_prompt_policy

    policy = resolve_prompt_policy(prompt_policy)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    kernel_bundle = results_root / kernel_bundle_dir_name(policy)
    reports_dir = results_root / "reports"
    kernel_bundle.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    if summary_path and summary_path.is_file():
        shutil.copy2(summary_path, results_root / "run_summary.json")

    # Export csynth trees + logs into kernel_bundle.
    export_manifest = export_csynth_bundle(
        matrix_root=matrix_root,
        flash_bundle_root=flash_bundle_root,
        kernel_bundle=kernel_bundle,
        force=force,
        remove_legacy_top_level=False,
    )
    shutil.copy2(
        kernel_bundle / "csynth_bundle_manifest.json",
        results_root / "csynth_bundle_manifest.json",
    )

    if old_kernel_bundle is None:
        old_kernel_bundle = matrix_root / "post_flash_dataflow_kernel_bundle"
    old_kernel_bundle = old_kernel_bundle.resolve()

    run_summary: list[dict[str, Any]] = []
    if summary_path and summary_path.is_file():
        data = _load_json(summary_path)
        if isinstance(data, list):
            run_summary = data

    rows: list[dict[str, Any]] = []
    cells = discover_matrix_cells(matrix_root)

    for cell in cells:
        bench = str(cell.get("bench") or "")
        if not bench.startswith("hlsfactory_"):
            continue
        short = bench.removeprefix("hlsfactory_")
        cell_dir = Path(cell["cell_dir"])
        bench_dir = kernel_bundle / bench

        flash_kernel_src = flash_bundle_root / bench / "selected" / "kernel.cpp"
        flash_kernel_dst = bench_dir / "flash" / "kernel.cpp"
        _copy_kernel(flash_kernel_src, flash_kernel_dst, force)

        df_paths = artifact_paths(cell_dir, bench)
        df_kernel_src = df_paths["kernel"]
        df_kernel_dst = bench_dir / "dataflow" / "kernel.cpp"
        has_dataflow_cpp = _copy_kernel(df_kernel_src, df_kernel_dst, force)

        summary_row = next((r for r in run_summary if r.get("bench") == bench), {})
        df_success = bool(summary_row.get("success"))
        df_skipped = bool(summary_row.get("skipped"))
        df_error = str(summary_row.get("error") or "")

        flash_xml = bench_dir / "flash_csynth" / "csynth.xml"
        df_xml = bench_dir / "dataflow_csynth" / "csynth.xml"
        flash_lat = _latency_triple_from_csynth(flash_xml)
        df_lat = _latency_triple_from_csynth(df_xml)

        flash_report_path = bench_dir / "flash_csynth" / "synth_report.json"
        df_report_path = bench_dir / "dataflow_csynth" / "dataflow_report.json"
        if not df_report_path.is_file():
            alt = bench_dir / "dataflow_csynth" / "dataflow_result.json"
            df_report_path = alt if alt.is_file() else df_report_path

        flash_report = _load_json(flash_report_path) if flash_report_path.is_file() else {}
        df_report = _load_json(df_report_path) if df_report_path.is_file() else {}
        if isinstance(df_report, dict) and "synth_report" in df_report:
            inner = df_report.get("synth_report")
            if isinstance(inner, dict):
                df_report = inner

        flash_res = _resources_from_csynth(flash_xml)
        if not any(v is not None for v in flash_res.values()) and isinstance(flash_report, dict):
            flash_res = {k: _parse_int(flash_report.get(k)) for k in flash_res}
        df_res = _resources_from_csynth(df_xml)
        if not any(v is not None for v in df_res.values()) and isinstance(df_report, dict):
            df_res = {k: _parse_int(df_report.get(k)) for k in df_res}

        flash_ii = _max_final_ii(bench_dir / "flash_csynth" / "vitis_hls.log")
        df_ii = _max_final_ii(bench_dir / "dataflow_csynth" / "vitis_hls.log")

        flash_interval = _parse_int(
            flash_report.get("interval") if isinstance(flash_report, dict) else None
        ) or flash_lat.get("interval_max")
        df_interval = _parse_int(
            df_report.get("interval") if isinstance(df_report, dict) else None
        ) or df_lat.get("interval_max")

        old_df_lat: dict[str, Optional[int]] = {
            "best": None, "avg": None, "worst": None,
            "interval_min": None, "interval_max": None,
        }
        old_df_interval: Optional[int] = None
        old_df_ii: Optional[int] = None
        old_audit: dict[str, Any] = {}
        old_kernel_path = _resolve_old_dataflow_kernel(old_kernel_bundle, bench)
        if old_kernel_path:
            old_audit = _kernel_code_audit(old_kernel_path.read_text(encoding="utf-8"))
            old_df_xml = old_kernel_bundle / bench / "dataflow_csynth" / "csynth.xml"
            old_df_lat = _latency_triple_from_csynth(old_df_xml)
            old_df_interval = old_df_lat.get("interval_max")
            old_df_ii = _max_final_ii(
                old_kernel_bundle / bench / "dataflow_csynth" / "vitis_hls.log"
            )

        new_audit: dict[str, Any] = {}
        if df_kernel_dst.is_file():
            new_audit = _kernel_code_audit(df_kernel_dst.read_text(encoding="utf-8"))

        prev_df_speedup = _speedup(old_df_lat.get("worst"), df_lat.get("worst"))

        row = {
            "bench": bench,
            "short": short,
            "dataflow_success": df_success,
            "dataflow_skipped": df_skipped,
            "dataflow_error_head": df_error[:240],
            "has_dataflow_cpp": has_dataflow_cpp,
            "flash": {
                "latency": flash_lat,
                "interval": flash_interval,
                "max_final_ii": flash_ii,
                "resources": flash_res,
            },
            "dataflow": {
                "latency": df_lat,
                "interval": df_interval,
                "max_final_ii": df_ii,
                "resources": df_res,
            },
            "prev_dataflow": {
                "latency": old_df_lat,
                "interval": old_df_interval,
                "max_final_ii": old_df_ii,
                "kernel_path": str(old_kernel_path) if old_kernel_path else None,
                "code_audit": old_audit,
            },
            "dataflow_code_audit": new_audit,
            "speedup": {
                "worst": _speedup(flash_lat.get("worst"), df_lat.get("worst")),
                "avg": _speedup(flash_lat.get("avg"), df_lat.get("avg")),
                "best": _speedup(flash_lat.get("best"), df_lat.get("best")),
                "interval": _speedup(flash_interval, df_interval),
                "prev_df_worst": prev_df_speedup,
            },
        }
        rows.append(row)

    rows.sort(key=lambda r: r["short"])

    pass_rows = [r for r in rows if r["dataflow_success"]]
    fail_rows = [r for r in rows if not r["dataflow_success"] and not r["dataflow_skipped"]]
    skip_rows = [r for r in rows if r["dataflow_skipped"]]

    def _collect_speedups(key: str) -> list[float]:
        out: list[float] = []
        for r in pass_rows:
            sp = r["speedup"].get(key)
            if sp is not None and sp > 0:
                out.append(sp)
        return out

    def _collect_prev_df_speedups() -> list[float]:
        out: list[float] = []
        for r in rows:
            sp = r["speedup"].get("prev_df_worst")
            if sp is not None and sp > 0:
                out.append(sp)
        return out

    prev_df_speedups = _collect_prev_df_speedups()
    code_audit_rows = [r for r in rows if r.get("prev_dataflow", {}).get("kernel_path")]

    summary_stats = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "matrix_root": str(matrix_root.resolve()),
        "results_root": str(results_root.resolve()),
        "prompt_policy": policy,
        "kernel_bundle_dir": kernel_bundle.name,
        "old_kernel_bundle": str(old_kernel_bundle),
        "skills_overlay": "flash_no_RMW_m_axi_skill_entries.json",
        "dataflow_run_summary": str(summary_path) if summary_path else None,
        "counts": {
            "total": len(rows),
            "dataflow_pass": len(pass_rows),
            "dataflow_fail": len(fail_rows),
            "skipped": len(skip_rows),
        },
        "geom_mean_speedup_passes_only": {
            "worst": _geom_mean(_collect_speedups("worst")),
            "avg": _geom_mean(_collect_speedups("avg")),
            "best": _geom_mean(_collect_speedups("best")),
            "interval": _geom_mean(_collect_speedups("interval")),
        },
        "geom_mean_prev_df_vs_new_worst": _geom_mean(prev_df_speedups),
        "prev_df_regressions_worst_lt_1": [
            r["short"]
            for r in rows
            if r["speedup"].get("prev_df_worst") is not None and r["speedup"]["prev_df_worst"] < 1.0
        ],
        "prev_df_wins_worst_gt_1_05": [
            r["short"]
            for r in rows
            if r["speedup"].get("prev_df_worst") is not None and r["speedup"]["prev_df_worst"] > 1.05
        ],
        "code_audit": {
            "old_with_distinct_gmemn": sum(
                1 for r in code_audit_rows
                if (r.get("prev_dataflow", {}).get("code_audit") or {}).get("distinct_gmemn_count", 0) > 0
            ),
            "new_with_distinct_gmemn": sum(
                1 for r in rows
                if (r.get("dataflow_code_audit") or {}).get("distinct_gmemn_count", 0) > 0
            ),
            "old_with_loop_labels": sum(
                1 for r in code_audit_rows
                if (r.get("prev_dataflow", {}).get("code_audit") or {}).get("loop_label_count", 0) > 0
            ),
            "new_with_loop_labels": sum(
                1 for r in rows
                if (r.get("dataflow_code_audit") or {}).get("loop_label_count", 0) > 0
            ),
        },
        "regressions_worst_lt_1": [
            r["short"]
            for r in pass_rows
            if r["speedup"].get("worst") is not None and r["speedup"]["worst"] < 1.0
        ],
        "wins_worst_gte_1_5": [
            r["short"]
            for r in pass_rows
            if r["speedup"].get("worst") is not None and r["speedup"]["worst"] >= 1.5
        ],
    }

    metrics_path = reports_dir / "metrics.json"
    _write_json(metrics_path, {"summary": summary_stats, "benches": rows})

    md_lines = [
        "# Post-flash DATAFLOW results (flash overlay skills)",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## Scope",
        "",
        f"- **Matrix:** `{matrix_root.name}`",
        f"- **Prompt policy:** `{policy}` (`system_skills` = skills in system; `user_skills` = skills in user)",
        "- **Flash step:** `all_new_skills_avoids_global` (90-skill base + `flash_no_RMW_m_axi_skill_entries.json` overlay)",
        f"- **DATAFLOW step:** task functions + `#pragma HLS DATAFLOW`, policy `{policy}`",
        f"- **Run summary:** `{summary_path.name if summary_path else 'n/a'}`",
        f"- **Previous DATAFLOW bundle:** `{old_kernel_bundle.name}` (pre-skills-in-prompt run)",
        "",
        "## DATAFLOW run outcome",
        "",
        f"| Outcome | Count |",
        f"|---------|------:|",
        f"| csim + csynth pass | {len(pass_rows)} |",
        f"| failed | {len(fail_rows)} |",
        f"| skipped (no flash kernel) | {len(skip_rows)} |",
        "",
        "## Methodology",
        "",
        "| Metric | Flash (baseline) | DATAFLOW (optimized) |",
        "|--------|------------------|----------------------|",
        "| Latency | `Best/Average/Worst-caseLatency` from `csynth.xml` top `SummaryOfOverallLatency` | same |",
        "| Interval | `Interval-max` (fallback: JSON `interval`) | same |",
        "| Speedup | flash ÷ dataflow per latency kind; **>1× = DATAFLOW faster** | |",
        "| Resources | BRAM, DSP, FF, LUT, URAM from parsed synth report | same |",
        "| Hot-loop II | max `Final II` from `vitis_hls.log` pipelining lines | same |",
        "",
        "Geometric means below use **DATAFLOW passes only** where both sides have defined latency.",
        "",
        "## Previous vs new DATAFLOW (worst latency)",
        "",
        f"| Metric | Value |",
        f"|--------|------:|",
        f"| Geo-mean **prev÷new** worst (>1× = new faster) | **{_fmt_speedup(summary_stats.get('geom_mean_prev_df_vs_new_worst'))}** |",
        f"| Benches new slower than prev (prev÷new < 1×) | {len(summary_stats['prev_df_regressions_worst_lt_1'])} |",
        f"| Benches new faster than prev (prev÷new > 1.05×) | {len(summary_stats['prev_df_wins_worst_gt_1_05'])} |",
        "",
        f"- **Regressions vs prev DATAFLOW:** "
        + (", ".join(f"`{x}`" for x in summary_stats["prev_df_regressions_worst_lt_1"]) or "none"),
        f"- **Wins vs prev DATAFLOW:** "
        + (", ".join(f"`{x}`" for x in summary_stats["prev_df_wins_worst_gt_1_05"]) or "none"),
        "",
        "## Code audit: gmemN bundles & loop labels",
        "",
        "Skills require `bundle=gmem0/gmem1/...` per port (`hls-distinct-gmem-bundle-per-port`) "
        "and mandatory `#pragma HLS DATAFLOW` with labeled loops. "
        "Compare prev vs new runs for compliance (gmemN count, loop labels).",
        "",
        f"| Audit | Previous DATAFLOW | New (skills run) |",
        f"|-------|------------------:|-----------------:|",
        f"| Kernels with distinct `gmemN` | {summary_stats['code_audit']['old_with_distinct_gmemn']} | {summary_stats['code_audit']['new_with_distinct_gmemn']} |",
        f"| Kernels with labeled `name: for` loops | {summary_stats['code_audit']['old_with_loop_labels']} | {summary_stats['code_audit']['new_with_loop_labels']} |",
        "",
        "## Summary vs flash (geometric mean, passes only)",
        "",
        "| Pairing | Geo-mean speedup | n |",
        "|---------|-----------------:|--:|",
    ]
    for key, label in (
        ("worst", "Worst ÷ Worst"),
        ("avg", "Avg ÷ Avg"),
        ("best", "Best ÷ Best"),
        ("interval", "Interval ÷ Interval"),
    ):
        gm = summary_stats["geom_mean_speedup_passes_only"].get(key)
        n = len(_collect_speedups(key))
        md_lines.append(f"| {label} | **{_fmt_speedup(gm)}** | {n} |")

    md_lines.extend([
        "",
        f"- **Regressions (worst < 1×):** {len(summary_stats['regressions_worst_lt_1'])} — "
        + (", ".join(f"`{x}`" for x in summary_stats["regressions_worst_lt_1"]) or "none"),
        f"- **Wins (worst ≥ 1.5×):** {len(summary_stats['wins_worst_gte_1_5'])} — "
        + (", ".join(f"`{x}`" for x in summary_stats["wins_worst_gte_1_5"]) or "none"),
        "",
        "## Full table (all benches)",
        "",
        "| Bench | DF ok | Prev DF worst | New DF worst | **Prev÷New** | "
        "Flash worst | **Flash÷New** | "
        "gmemN (prev→new) | loop labels (prev→new) |",
        "|-------|:-----:|--------------:|-------------:|-------------:|"
        "------------:|-------------:|"
        "-----------------:|---------------------:|",
    ])

    for r in rows:
        pl = r.get("prev_dataflow", {}).get("latency") or {}
        dl = r["dataflow"]["latency"]
        sp = r["speedup"]
        old_audit = r.get("prev_dataflow", {}).get("code_audit") or {}
        new_audit = r.get("dataflow_code_audit") or {}
        gmem_note = (
            f"{old_audit.get('distinct_gmemn_count', 0)}→{new_audit.get('distinct_gmemn_count', 0)}"
        )
        label_note = (
            f"{old_audit.get('loop_label_count', 0)}→{new_audit.get('loop_label_count', 0)}"
        )
        ok = "✓" if r["dataflow_success"] else ("—" if r["dataflow_skipped"] else "✗")
        md_lines.append(
            f"| `{r['short']}` | {ok} | "
            f"{_fmt_cycles(pl.get('worst'))} | {_fmt_cycles(dl.get('worst'))} | "
            f"**{_fmt_speedup(sp.get('prev_df_worst'))}** | "
            f"{_fmt_cycles(r['flash']['latency'].get('worst'))} | "
            f"**{_fmt_speedup(sp.get('worst'))}** | "
            f"{gmem_note} | {label_note} |"
        )

    md_lines.extend([
        "",
        "## Full table — latency / interval / II / resources (all benches)",
        "",
        "| Bench | DF ok | Flash worst/avg/best | DF worst/avg/best | "
        "**Worst÷Worst** | **Avg÷Avg** | **Best÷Best** | "
        "Flash interval | DF interval | **Intv÷Intv** | "
        "Flash IIₘₐₓ | DF IIₘₐₓ | BRAM Δ | LUT Δ |",
        "|-------|:-----:|---------------------:|------------------:|"
        "--------------:|-----------:|------------:|"
        "---------------:|------------:|-------------:|"
        "------------:|-----------:|-------:|------:|",
    ])

    for r in rows:
        fl = r["flash"]["latency"]
        dl = r["dataflow"]["latency"]
        sp = r["speedup"]
        fr = r["flash"]["resources"]
        dr = r["dataflow"]["resources"]
        bram_d = (
            dr["bram"] - fr["bram"]
            if dr["bram"] is not None and fr["bram"] is not None
            else None
        )
        lut_d = (
            dr["lut"] - fr["lut"]
            if dr["lut"] is not None and fr["lut"] is not None
            else None
        )
        ok = "✓" if r["dataflow_success"] else ("—" if r["dataflow_skipped"] else "✗")
        md_lines.append(
            f"| `{r['short']}` | {ok} | "
            f"{_fmt_cycles(fl.get('worst'))} / {_fmt_cycles(fl.get('avg'))} / {_fmt_cycles(fl.get('best'))} | "
            f"{_fmt_cycles(dl.get('worst'))} / {_fmt_cycles(dl.get('avg'))} / {_fmt_cycles(dl.get('best'))} | "
            f"**{_fmt_speedup(sp.get('worst'))}** | **{_fmt_speedup(sp.get('avg'))}** | "
            f"**{_fmt_speedup(sp.get('best'))}** | "
            f"{_fmt_cycles(r['flash']['interval'])} | {_fmt_cycles(r['dataflow']['interval'])} | "
            f"**{_fmt_speedup(sp.get('interval'))}** | "
            f"{r['flash']['max_final_ii'] or '—'} | {r['dataflow']['max_final_ii'] or '—'} | "
            f"{bram_d if bram_d is not None else 'n/a'} | {lut_d if lut_d is not None else 'n/a'} |"
        )

    if fail_rows:
        md_lines.extend(["", "## DATAFLOW failures", ""])
        for r in fail_rows:
            md_lines.append(f"- `{r['short']}`: {r['dataflow_error_head'] or 'unknown error'}")

    if pass_rows:
        md_lines.extend([
            "",
            "## Interval changes (passes only)",
            "",
            "| Bench | Flash interval | DF interval | Δ% | Speedup |",
            "|-------|---------------:|------------:|---:|--------:|",
        ])
        for r in pass_rows:
            fi = r["flash"]["interval"]
            di = r["dataflow"]["interval"]
            md_lines.append(
                f"| `{r['short']}` | {_fmt_cycles(fi)} | {_fmt_cycles(di)} | "
                f"{_fmt_pct_change(fi, di)} | **{_fmt_speedup(r['speedup'].get('interval'))}** |"
            )

        md_lines.extend([
            "",
            "## Hot-loop II (max Final II from vitis_hls.log, passes only)",
            "",
            "| Bench | Flash IIₘₐₓ | DF IIₘₐₓ | Δ |",
            "|-------|------------:|---------:|--:|",
        ])
        for r in pass_rows:
            fii = r["flash"]["max_final_ii"]
            dii = r["dataflow"]["max_final_ii"]
            delta = (
                f"{dii - fii:+d}"
                if fii is not None and dii is not None
                else "n/a"
            )
            md_lines.append(
                f"| `{r['short']}` | {fii if fii is not None else '—'} | "
                f"{dii if dii is not None else '—'} | {delta} |"
            )

        md_lines.extend([
            "",
            "## Resource deltas (passes only)",
            "",
            "| Bench | ΔBRAM | ΔDSP | ΔFF | ΔLUT | ΔURAM |",
            "|-------|------:|-----:|----:|-----:|------:|",
        ])
        for r in pass_rows:
            fr = r["flash"]["resources"]
            dr = r["dataflow"]["resources"]

            def _d(k: str) -> str:
                a, b = fr.get(k), dr.get(k)
                if a is None or b is None:
                    return "n/a"
                return f"{b - a:+d}"

            md_lines.append(
                f"| `{r['short']}` | {_d('bram')} | {_d('dsp')} | {_d('ff')} | "
                f"{_d('lut')} | {_d('uram')} |"
            )

    md_lines.extend([
        "",
        "## Artifact layout",
        "",
        "```",
        f"{results_root.name}/",
        "  kernel_bundle/hlsfactory_<bench>/",
        "    flash/kernel.cpp",
        "    dataflow/kernel.cpp",
        "    flash_csynth/{csynth.xml,vitis_hls.log,sol1.log,*.rpt}",
        "    dataflow_csynth/{csynth.xml,vitis_hls.log,sol1.log,*.rpt}",
        "  reports/post_flash_dataflow_results.md",
        "  reports/metrics.json",
        "  run_summary.json",
        "```",
        "",
    ])

    md_path = reports_dir / "post_flash_dataflow_results.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    return {
        "results_root": str(results_root.resolve()),
        "kernel_bundle": str(kernel_bundle.resolve()),
        "report_md": str(md_path.resolve()),
        "metrics_json": str(metrics_path.resolve()),
        "summary": summary_stats,
        "export_manifest": export_manifest,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument(
        "--flash-bundle-root",
        type=Path,
        default=None,
        help="default: artifacts/pc2/flash_selected_bundle/<matrix-name>",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=None,
        help="default: <matrix-root>/post_flash_dataflow_results_<stamp>",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="post_flash_dataflow_summary_*.json (default: newest in matrix-root)",
    )
    parser.add_argument(
        "--old-kernel-bundle",
        type=Path,
        default=None,
        help="Previous DATAFLOW bundle (default: <matrix-root>/post_flash_dataflow_kernel_bundle)",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    matrix_root = args.matrix_root.resolve()
    flash_bundle_root = (
        args.flash_bundle_root.resolve()
        if args.flash_bundle_root
        else (REPO / "artifacts/pc2/flash_selected_bundle" / matrix_root.name).resolve()
    )

    summary_path = args.summary
    if summary_path is None:
        candidates = sorted(matrix_root.glob("post_flash_dataflow_summary_*.json"))
        summary_path = candidates[-1] if candidates else None
    elif not summary_path.is_absolute():
        summary_path = matrix_root / summary_path

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    results_root = (
        args.results_root.resolve()
        if args.results_root
        else (matrix_root / f"post_flash_dataflow_results_{stamp}_flash_overlay").resolve()
    )

    out = build_results(
        matrix_root=matrix_root,
        flash_bundle_root=flash_bundle_root,
        results_root=results_root,
        summary_path=summary_path,
        old_kernel_bundle=(
            args.old_kernel_bundle.resolve()
            if args.old_kernel_bundle
            else None
        ),
        force=args.force,
    )
    s = out["summary"]
    print(f"results: {out['results_root']}")
    print(f"report:  {out['report_md']}")
    print(
        f"passes: {s['counts']['dataflow_pass']}/{s['counts']['total']} | "
        f"geom worst={_fmt_speedup(s['geom_mean_speedup_passes_only']['worst'])} "
        f"avg={_fmt_speedup(s['geom_mean_speedup_passes_only']['avg'])} "
        f"best={_fmt_speedup(s['geom_mean_speedup_passes_only']['best'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
