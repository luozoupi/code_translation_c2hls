#!/usr/bin/env python3
"""Aggregate post-flash DATAFLOW runs vs flash-selected into one comparison report.

Scans matrix-root for packaged result bundles, run summaries/plans, the original
kernel bundle, and current per-cell success state. Emits:

  <matrix-root>/reports/post_flash_dataflow_run_comparison.md
  <matrix-root>/reports/post_flash_dataflow_run_comparison.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "pc2"))


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
        "best": None, "avg": None, "worst": None,
        "interval_min": None, "interval_max": None,
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
        "bram": "BRAM_18K", "dsp": "DSP", "ff": "FF", "lut": "LUT", "uram": "URAM",
    }
    for key, tag in mapping.items():
        out[key] = _parse_int(resources.findtext(tag))
    return out


def _speedup(base: Optional[int], opt: Optional[int]) -> Optional[float]:
    if base is None or opt is None or opt <= 0:
        return None
    return base / opt


def _geom_mean(values: list[float]) -> Optional[float]:
    if not values:
        return None
    return math.exp(sum(math.log(v) for v in values) / len(values))


def _fmt_speedup(v: Optional[float]) -> str:
    if v is None:
        return "n/a"
    return f"{v:.3f}×"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_run_id(name: str) -> str:
    m = re.search(r"(\d{8}_\d{6})", name)
    return m.group(1) if m else name


def _find_plan(matrix: Path, stamp: str) -> Optional[dict[str, Any]]:
    plan_path = matrix / f"post_flash_dataflow_plan_{stamp}.json"
    if plan_path.is_file():
        return _load_json(plan_path)
    return None


def _find_meta(matrix: Path, stamp: str) -> dict[str, Any]:
    meta_path = matrix / f"post_flash_dataflow_summary_meta_{stamp}.json"
    if meta_path.is_file():
        return _load_json(meta_path)
    return {}


def _summary_stats(summary_path: Path) -> dict[str, Any]:
    data = _load_json(summary_path)
    if isinstance(data, list):
        attempted = [r for r in data if not r.get("skipped")]
        passed = [r for r in attempted if r.get("success")]
        failed = [r for r in attempted if not r.get("success")]
        return {
            "total": len(data),
            "attempted": len(attempted),
            "dataflow_pass": len(passed),
            "dataflow_fail": len(failed),
            "passed_benches": [r.get("bench") for r in passed],
            "failed_benches": [r.get("bench") for r in failed],
        }
    return data if isinstance(data, dict) else {}


def _bench_rows_from_bundle(
    bundle_root: Path,
    *,
    pass_only: bool = True,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    kb: Optional[Path] = None
    if (bundle_root / "kernel_bundle").is_dir():
        kb = bundle_root / "kernel_bundle"
    else:
        for alt in sorted(bundle_root.glob("kernel_bundle_*")):
            kb = alt
            break
    if kb is None and any(bundle_root.glob("hlsfactory_*")):
        kb = bundle_root

    if kb is None or not kb.is_dir():
        return rows

    for bench_dir in sorted(kb.iterdir()):
        if not bench_dir.is_dir() or not bench_dir.name.startswith("hlsfactory_"):
            continue
        bench = bench_dir.name
        flash_xml = bench_dir / "flash_csynth" / "csynth.xml"
        df_xml = bench_dir / "dataflow_csynth" / "csynth.xml"
        if not flash_xml.is_file() or not df_xml.is_file():
            continue
        flash_lat = _latency_triple_from_csynth(flash_xml)
        df_lat = _latency_triple_from_csynth(df_xml)
        flash_res = _resources_from_csynth(flash_xml)
        df_res = _resources_from_csynth(df_xml)
        row = {
            "bench": bench,
            "short": bench.replace("hlsfactory_", ""),
            "flash": {"latency": flash_lat, "resources": flash_res},
            "dataflow": {"latency": df_lat, "resources": df_res},
            "speedup": {
                "worst": _speedup(flash_lat.get("worst"), df_lat.get("worst")),
                "avg": _speedup(flash_lat.get("avg"), df_lat.get("avg")),
                "best": _speedup(flash_lat.get("best"), df_lat.get("best")),
                "interval": _speedup(
                    flash_lat.get("interval_max") or flash_lat.get("interval_min"),
                    df_lat.get("interval_max") or df_lat.get("interval_min"),
                ),
            },
            "resource_delta": {
                k: (df_res.get(k) - flash_res.get(k))
                if df_res.get(k) is not None and flash_res.get(k) is not None
                else None
                for k in ("bram", "dsp", "ff", "lut", "uram")
            },
        }
        rows.append(row)
    return rows


def _aggregate_rows(rows: list[dict[str, Any]], *, pass_benches: Optional[set[str]] = None) -> dict[str, Any]:
    if pass_benches is not None:
        rows = [r for r in rows if r["bench"] in pass_benches]
    out: dict[str, Any] = {"bench_count": len(rows)}
    for key in ("worst", "avg", "best", "interval"):
        vals = [r["speedup"][key] for r in rows if r.get("speedup", {}).get(key) is not None]
        out[f"geom_mean_speedup_{key}"] = _geom_mean(vals)
        out[f"n_speedup_{key}"] = len(vals)
        out[f"df_faster_{key}"] = sum(1 for v in vals if v > 1.0)
    for res in ("bram", "dsp", "ff", "lut", "uram"):
        deltas = [
            r["resource_delta"][res]
            for r in rows
            if r.get("resource_delta", {}).get(res) is not None
        ]
        out[f"mean_delta_{res}"] = (sum(deltas) / len(deltas)) if deltas else None
        out[f"increased_{res}"] = sum(1 for d in deltas if d > 0)
        out[f"decreased_{res}"] = sum(1 for d in deltas if d < 0)
    return out


def _rows_from_metrics(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for b in metrics.get("benches", []):
        if not b.get("dataflow_success"):
            continue
        fr = b.get("flash", {}).get("resources", {})
        dr = b.get("dataflow", {}).get("resources", {})
        rows.append({
            "bench": b["bench"],
            "short": b.get("short", b["bench"]),
            "speedup": b.get("speedup", {}),
            "resource_delta": {
                k: (dr.get(k) - fr.get(k)) if dr.get(k) is not None and fr.get(k) is not None else None
                for k in ("bram", "dsp", "ff", "lut", "uram")
            },
        })
    return rows


def _current_pass_benches(matrix: Path) -> set[str]:
    passed: set[str] = set()
    for cell in sorted(matrix.iterdir()):
        if not cell.is_dir() or not cell.name.startswith("hlsfactory_"):
            continue
        bench = cell.name
        subs = [p for p in cell.iterdir() if p.is_dir()]
        if not subs:
            continue
        result = subs[0] / f"{bench}_dataflow_result.json"
        if result.is_file() and _load_json(result).get("success"):
            passed.add(bench)
    return passed


def discover_runs(matrix: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []

    # Original kernel bundle (baseline DATAFLOW before later reruns)
    kb = matrix / "post_flash_dataflow_kernel_bundle"
    if kb.is_dir():
        rows = _bench_rows_from_bundle(kb, pass_only=False)
        runs.append({
            "run_id": "original_kernel_bundle",
            "label": "Original DATAFLOW bundle (first batch)",
            "kind": "baseline",
            "results_root": str(kb),
            "conditions": {
                "prompt_policy": "legacy (skills in system)",
                "skills": "flash_no_RMW_m_axi overlay",
                "contract_check": False,
                "repair_rounds": 4,
                "scope": "full matrix (~25 kernels)",
                "notes": "Pre skills-in-prompt churn; packaged csynth for flash + dataflow",
            },
            "outcome": {"dataflow_pass": len(rows), "total": 28},
            "aggregate": _aggregate_rows(rows),
            "rows": rows,
        })

    # Packaged result bundles
    for bundle in sorted(matrix.glob("post_flash_dataflow_results_*")):
        metrics_path = bundle / "reports" / "metrics.json"
        if not metrics_path.is_file():
            continue
        metrics = _load_json(metrics_path)
        summary = metrics.get("summary", {})
        stamp = _parse_run_id(bundle.name)
        meta = _find_meta(matrix, stamp)
        plan = _find_plan(matrix, stamp)
        summary_path = summary.get("dataflow_run_summary", "")
        sp = Path(summary_path) if summary_path else matrix / f"post_flash_dataflow_summary_{stamp}.json"
        run_stats = _summary_stats(sp) if sp.is_file() else {}

        suffix_m = re.search(r"post_flash_dataflow_results_\d{8}_\d{6}_(.+)$", bundle.name)
        suffix = suffix_m.group(1) if suffix_m else bundle.name.split("_", 4)[-1]

        conditions = {
            "prompt_policy": summary.get("prompt_policy") or meta.get("prompt_policy") or (plan or {}).get("prompt_policy") or "system_skills (default)",
            "results_suffix": meta.get("results_suffix") or (plan or {}).get("results_suffix") or suffix,
            "skills": summary.get("skills_overlay", "flash_no_RMW_m_axi_skill_entries.json"),
            "contract_check": (plan or {}).get("contract_check"),
            "contract_rounds": (plan or {}).get("contract_rounds"),
            "repair_rounds": (plan or {}).get("repair_rounds", 4),
            "scope": f"batch ({run_stats.get('attempted', run_stats.get('total', '?'))} cells)",
        }
        if plan and plan.get("cells"):
            benches = [c["bench"] for c in plan["cells"]]
            if len(benches) <= 10:
                conditions["scope"] = f"targeted ({len(benches)}): " + ", ".join(
                    b.replace("hlsfactory_", "") for b in benches
                )

        pass_rows = _rows_from_metrics(metrics)
        runs.append({
            "run_id": bundle.name,
            "label": suffix.replace("_", " "),
            "kind": "packaged",
            "stamp": stamp,
            "results_root": str(bundle),
            "summary_path": str(sp) if sp.is_file() else None,
            "conditions": conditions,
            "outcome": summary.get("counts") or {
                "dataflow_pass": run_stats.get("dataflow_pass"),
                "dataflow_fail": run_stats.get("dataflow_fail"),
                "total": run_stats.get("total"),
            },
            "aggregate": {
                **{f"geom_mean_speedup_{k}": summary.get("geom_mean_speedup_passes_only", {}).get(k)
                   for k in ("worst", "avg", "best", "interval")},
                **_aggregate_rows(pass_rows),
            },
            "code_audit": summary.get("code_audit"),
            "rows": pass_rows,
        })

    # Current best per cell (latest success flags)
    passed = _current_pass_benches(matrix)
    if passed and kb.is_dir():
        all_rows = _bench_rows_from_bundle(kb, pass_only=False)
        pass_rows = [r for r in all_rows if r["bench"] in passed]
        runs.append({
            "run_id": "current_cell_success",
            "label": "Current best per cell (success flags)",
            "kind": "snapshot",
            "results_root": str(matrix),
            "conditions": {
                "prompt_policy": "mixed (latest success per bench)",
                "contract_check": "partial (latest runs)",
                "scope": f"{len(passed)} passing cells",
                "notes": "Flash csynth from kernel bundle; DATAFLOW csynth from kernel bundle (may lag latest reruns for some benches)",
            },
            "outcome": {"dataflow_pass": len(passed), "total": 28},
            "aggregate": _aggregate_rows(pass_rows),
            "passed_benches": sorted(passed),
            "rows": pass_rows,
        })

    return runs


def render_markdown(matrix: Path, runs: list[dict[str, Any]]) -> str:
    lines = [
        "# Post-flash DATAFLOW run comparison vs flash-selected",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        f"**Matrix:** `{matrix.name}`",
        "",
        "**Baseline:** flash-selected kernel (`*_final.cpp`) csynth latency/resources.",
        "",
        "**Speedup:** flash ÷ DATAFLOW per metric; **>1× = DATAFLOW faster**.",
        "",
        "Geometric means use **passing benches only** where both sides have csynth data.",
        "",
        "## Run index",
        "",
        "| Run | Kind | Pass | Policy | Suffix | Contract | Scope |",
        "|-----|------|-----:|--------|--------|----------|-------|",
    ]
    for r in runs:
        c = r.get("conditions", {})
        o = r.get("outcome", {})
        pass_n = o.get("dataflow_pass", "?")
        total = o.get("total", "?")
        lines.append(
            f"| `{r['run_id']}` | {r.get('kind','')} | {pass_n}/{total} | "
            f"{c.get('prompt_policy','—')} | {c.get('results_suffix','—')} | "
            f"{c.get('contract_check','—')} | {str(c.get('scope',''))[:60]} |"
        )

    lines += [
        "",
        "## Geo-mean speedup vs flash (passes only)",
        "",
        "| Run | Worst | Average | Best | Interval | n(worst) |",
        "|-----|------:|--------:|-----:|---------:|---------:|",
    ]
    for r in runs:
        a = r.get("aggregate", {})
        lines.append(
            f"| `{r['run_id']}` | {_fmt_speedup(a.get('geom_mean_speedup_worst'))} | "
            f"{_fmt_speedup(a.get('geom_mean_speedup_avg'))} | "
            f"{_fmt_speedup(a.get('geom_mean_speedup_best'))} | "
            f"{_fmt_speedup(a.get('geom_mean_speedup_interval'))} | "
            f"{a.get('n_speedup_worst', '—')} |"
        )

    lines += [
        "",
        "## Resource delta vs flash (mean Δ on passing benches)",
        "",
        "Positive Δ = DATAFLOW uses **more** of that resource than flash.",
        "",
        "| Run | Δ BRAM | Δ DSP | Δ FF | Δ LUT | Δ URAM | increased BRAM |",
        "|-----|-------:|------:|-----:|------:|-------:|---------------:|",
    ]
    for r in runs:
        a = r.get("aggregate", {})
        def d(k):
            v = a.get(k)
            if v is None:
                return "n/a"
            return f"{v:+.1f}"
        lines.append(
            f"| `{r['run_id']}` | {d('mean_delta_bram')} | {d('mean_delta_dsp')} | "
            f"{d('mean_delta_ff')} | {d('mean_delta_lut')} | {d('mean_delta_uram')} | "
            f"{a.get('increased_bram', '—')} |"
        )

    lines += [
        "",
        "## Test conditions (detail)",
        "",
    ]
    for r in runs:
        lines.append(f"### `{r['run_id']}` — {r.get('label', '')}")
        lines.append("")
        c = r.get("conditions", {})
        for k, v in c.items():
            lines.append(f"- **{k}:** {v}")
        o = r.get("outcome", {})
        if o:
            lines.append(f"- **outcome:** {o}")
        if r.get("summary_path"):
            lines.append(f"- **summary:** `{Path(r['summary_path']).name}`")
        if r.get("results_root"):
            lines.append(f"- **artifacts:** `{r['results_root']}`")
        audit = r.get("code_audit")
        if audit:
            lines.append(f"- **code_audit:** {audit}")
        lines.append("")

    lines += [
        "## Interpretation notes",
        "",
        "1. **Original kernel bundle** is the cleanest apples-to-apples baseline (~1.2× geo-mean on all latency kinds).",
        "2. **Full skills-in-prompt batch** (`flash_overlay`, 22 pass) regressed to ~0.99× vs flash — decorative DATAFLOW, no gmemN rebundling.",
        "3. **Targeted reruns** (`fanout_fix`, `parallel_fix`) have small n; geo-means are not comparable to full-matrix runs.",
        "4. **Current cell success** (25 pass) mixes kernels from multiple runs; treat as operational snapshot, not a single experiment.",
        "5. Packaged bundles are the authoritative source per run; rebuild with `build_post_flash_dataflow_results.py` after new batches.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate DATAFLOW runs vs flash")
    parser.add_argument("--matrix-root", type=str, required=True)
    args = parser.parse_args()

    matrix = Path(args.matrix_root).expanduser()
    if not matrix.is_absolute():
        matrix = REPO / matrix
    if not matrix.is_dir():
        print(f"matrix root missing: {matrix}", file=sys.stderr)
        return 1

    runs = discover_runs(matrix)
    out_dir = matrix / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "matrix_root": str(matrix),
        "run_count": len(runs),
        "runs": runs,
    }
    json_path = out_dir / "post_flash_dataflow_run_comparison.json"
    md_path = out_dir / "post_flash_dataflow_run_comparison.md"
    json_path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(matrix, runs), encoding="utf-8")
    print(f"wrote {md_path}")
    print(f"wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
