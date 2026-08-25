#!/usr/bin/env python3
"""Run Vitis cosim on vLLM-generated kernels and compare with gold references."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

os.environ.setdefault(
    "C2HLS_VITIS_SETTINGS",
    "/mnt/data/luo00466/Xilinx/Vitis/2023.2/settings64.sh",
)
os.environ.setdefault("C2HLS_VITIS_VERSION", "2023.2")
os.environ.setdefault("C2HLS_PART", "xcu280-fsvh2892-2L-e")
os.environ.setdefault("C2HLS_CLOCK_NS", "3.33")
os.environ.setdefault("C2HLS_FLOW_TARGET", "vitis")

import hls_eval  # noqa: E402
from c2hls import _load_benchmark_inputs  # noqa: E402


DEFAULT_HLSFACTORY_ROOT = REPO / "benchmarks_external" / "HLSFactory" / "polybench_float_small"
DEFAULT_REFERENCE_JSONL = (
    REPO
    / "artifacts"
    / "hlsfactory_multistep_sonnet46_skill_on_website_revstyle_combined_20260615.jsonl"
)
DEFAULT_OUTPUT_DIR = REPO / "artifacts" / "vllm_cosim_compare"


def _bench_key(value: Any) -> str:
    raw = str(value or "")
    raw = raw.removeprefix("hlsfactory_")
    raw = raw.replace("-", "_").replace("/", "_")
    return f"hlsfactory_{raw}" if raw else ""


def _as_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(str(value).replace(",", "")))
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _ratio(numerator: int | None, denominator: int | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def _status(result: dict[str, Any] | None) -> str:
    if not result:
        return "not_run"
    if result.get("success") and result.get("passed"):
        return "pass"
    err = str(result.get("error") or "").lower()
    if "timed out" in err or "timeout" in err:
        return "timeout"
    return "fail"


def _short_error(result: dict[str, Any] | None, max_chars: int = 1000) -> str:
    if not isinstance(result, dict):
        return ""
    return str(result.get("error") or "")[:max_chars]


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _load_gold_cosim(path: Path) -> dict[str, dict[str, Any]]:
    gold: dict[str, dict[str, Any]] = {}
    for record in _read_jsonl(path):
        if record.get("report_type") != "rtl_sim":
            continue
        impl = record.get("implementation") or {}
        if impl.get("origin") != "hlsfactory_benchmark":
            continue
        bench = _bench_key("/".join((record.get("problem") or {}).get("group_path") or []))
        rtl = record.get("rtl_sim") or {}
        gold[bench] = {
            "gold_cosim_status": rtl.get("status"),
            "gold_cosim_cycles": _as_int(rtl.get("kernel_runtime_cycles")),
            "gold_cosim_runtime_us": _as_float(rtl.get("kernel_runtime_us")),
            "gold_cosim_clock_freq_mhz": _as_float(rtl.get("kernel_clock_freq_mhz")),
            "gold_reference_jsonl": str(path),
        }
    return gold


def _load_vllm_rows(summary_paths: list[Path], *, benchmarks: set[str]) -> list[dict[str, Any]]:
    rows = []
    seen: set[str] = set()
    for path in summary_paths:
        summary = json.loads(path.read_text())
        for row in summary.get("rows") or []:
            bench = _bench_key(row.get("benchmark"))
            if not bench or bench in seen:
                continue
            if benchmarks and bench not in benchmarks:
                continue
            synth = row.get("synth") or {}
            csim = row.get("csim") or {}
            if synth.get("status") != "pass" or csim.get("status") != "pass":
                continue
            code_path = Path(str(row.get("generated_code_path") or ""))
            if not code_path.is_absolute():
                code_path = (REPO / code_path).resolve()
            if not code_path.exists():
                continue
            seen.add(bench)
            rows.append(
                {
                    "benchmark": bench,
                    "source_summary": str(path),
                    "generated_code_path": str(code_path),
                    "vllm_synth_cycles": _as_int(synth.get("latency_cycles")),
                    "vllm_synth_status": synth.get("status"),
                    "vllm_csim_status": csim.get("status"),
                }
            )
    return rows


def _run_case(
    row: dict[str, Any],
    *,
    hlsfactory_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    bench = row["benchmark"]
    bench_dir = hlsfactory_root / bench
    case_dir = output_dir / bench
    case_dir.mkdir(parents=True, exist_ok=True)

    code = Path(row["generated_code_path"]).read_text()
    inputs = _load_benchmark_inputs(str(bench_dir))
    meta = inputs["meta"]
    top = meta.get("hls_top", "workload")
    started = time.time()
    cosim = hls_eval.run_cosim(
        code,
        inputs.get("testbench_code", ""),
        inputs.get("header_code", ""),
        header_name=inputs.get("header_name") or "kernel.h",
        top_function=top,
        part=hls_eval.DEFAULT_PART,
        clock_ns=hls_eval.DEFAULT_CLOCK_NS,
        work_dir=str(case_dir / "cosim_work"),
        extra_files=inputs.get("extra_files", []),
        interface_depths=meta.get("cosim_depths") or {},
    )
    elapsed = round(time.time() - started, 3)
    return {
        **row,
        "vllm_cosim_status": _status(cosim),
        "vllm_cosim_runtime_seconds": elapsed,
        "vllm_cosim_cycles": _as_int(cosim.get("kernel_runtime_cycles")) if isinstance(cosim, dict) else None,
        "vllm_cosim_runtime_us": _as_float(cosim.get("kernel_runtime_us")) if isinstance(cosim, dict) else None,
        "vllm_cosim_clock_freq_mhz": _as_float(cosim.get("kernel_clock_freq_mhz")) if isinstance(cosim, dict) else None,
        "vllm_cosim_error": _short_error(cosim),
        "vllm_cosim_work_dir": str(case_dir / "cosim_work"),
    }


def _write_outputs(rows: list[dict[str, Any]], summary: dict[str, Any], output_dir: Path) -> None:
    result_jsonl = output_dir / "results.jsonl"
    summary_json = output_dir / "summary.json"
    summary_md = output_dir / "summary.md"
    with result_jsonl.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    lines = [
        "# vLLM Gemma Cosim vs HLSFactory Gold",
        "",
        f"- JSONL: `{result_jsonl}`",
        f"- Vitis: `{summary['vitis']['version']}` / `{summary['vitis']['part']}` / `{summary['vitis']['clock_ns']} ns`",
        f"- Gold reference: `{summary['gold_reference_jsonl']}`",
        "",
        "| bench | vLLM cosim | vLLM cycles | gold cosim | gold cycles | vLLM/gold | synth cycles | error |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        err = row.get("vllm_cosim_error") or ""
        if len(err) > 100:
            err = err[:97] + "..."
        ratio = row.get("vllm_over_gold_cosim_cycles")
        lines.append(
            "| {bench} | {vstatus} | {vcycles} | {gstatus} | {gcycles} | {ratio} | {synth} | {err} |".format(
                bench=row.get("benchmark"),
                vstatus=row.get("vllm_cosim_status") or "",
                vcycles=row.get("vllm_cosim_cycles") or "",
                gstatus=row.get("gold_cosim_status") or "",
                gcycles=row.get("gold_cosim_cycles") or "",
                ratio=f"{ratio:.4g}" if isinstance(ratio, float) else "",
                synth=row.get("vllm_synth_cycles") or "",
                err=err.replace("|", "/"),
            )
        )
    summary_md.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-summary", type=Path, action="append", required=True)
    parser.add_argument("--reference-jsonl", type=Path, default=DEFAULT_REFERENCE_JSONL)
    parser.add_argument("--hlsfactory-root", type=Path, default=DEFAULT_HLSFACTORY_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-name", default=f"gemma4_31b_cosim_compare_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--benchmark", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    output_dir = (args.output_dir / args.run_name).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    gold = _load_gold_cosim(args.reference_jsonl)
    wanted = {_bench_key(item) for item in args.benchmark if item}
    selected = _load_vllm_rows(args.vllm_summary, benchmarks=wanted)
    if args.limit:
        selected = selected[: args.limit]

    rows = []
    for index, row in enumerate(selected, 1):
        print(f"[{index}/{len(selected)}] {row['benchmark']}", flush=True)
        result = _run_case(row, hlsfactory_root=args.hlsfactory_root, output_dir=output_dir)
        result.update(gold.get(result["benchmark"], {}))
        result["vllm_over_gold_cosim_cycles"] = _ratio(
            result.get("vllm_cosim_cycles"),
            result.get("gold_cosim_cycles"),
        )
        rows.append(result)
        print(
            json.dumps(
                {
                    "benchmark": result["benchmark"],
                    "vllm_cosim": result["vllm_cosim_status"],
                    "vllm_cycles": result.get("vllm_cosim_cycles"),
                    "gold_cycles": result.get("gold_cosim_cycles"),
                    "ratio": result.get("vllm_over_gold_cosim_cycles"),
                    "seconds": result["vllm_cosim_runtime_seconds"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    comparable = [
        row for row in rows
        if row.get("vllm_cosim_cycles") is not None and row.get("gold_cosim_cycles") is not None
    ]
    ratios = [row["vllm_over_gold_cosim_cycles"] for row in comparable if row.get("vllm_over_gold_cosim_cycles") is not None]
    summary = {
        "schema_version": "vllm_cosim_compare_v1",
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(output_dir),
        "gold_reference_jsonl": str(args.reference_jsonl),
        "vllm_summaries": [str(path) for path in args.vllm_summary],
        "vitis": {
            "settings": os.getenv("C2HLS_VITIS_SETTINGS"),
            "version": os.getenv("C2HLS_VITIS_VERSION", "2023.2"),
            "part": hls_eval.DEFAULT_PART,
            "clock_ns": hls_eval.DEFAULT_CLOCK_NS,
            "flow_target": hls_eval.DEFAULT_FLOW_TARGET,
        },
        "counts": {
            "selected": len(selected),
            "vllm_cosim_pass": sum(1 for row in rows if row.get("vllm_cosim_status") == "pass"),
            "vllm_cosim_fail": sum(1 for row in rows if row.get("vllm_cosim_status") == "fail"),
            "vllm_cosim_timeout": sum(1 for row in rows if row.get("vllm_cosim_status") == "timeout"),
            "gold_cosim_pass": sum(1 for row in rows if row.get("gold_cosim_status") == "pass"),
            "cycle_comparable": len(comparable),
            "vllm_faster_than_gold": sum(1 for row in comparable if row["vllm_over_gold_cosim_cycles"] < 1.0),
            "gold_faster_than_vllm": sum(1 for row in comparable if row["vllm_over_gold_cosim_cycles"] > 1.0),
        },
        "ratio_vllm_over_gold_cosim_cycles": {
            "min": min(ratios) if ratios else None,
            "median": sorted(ratios)[len(ratios) // 2] if ratios else None,
            "max": max(ratios) if ratios else None,
        },
        "rows": rows,
    }
    _write_outputs(rows, summary, output_dir)
    print(json.dumps(summary["counts"], indent=2, sort_keys=True))
    print(f"summary={output_dir / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
