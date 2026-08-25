#!/usr/bin/env python3
"""Run a small Vitis csynth/csim smoke on generated vLLM C2HLS outputs."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
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


DEFAULT_OUTPUT_DIR = REPO / "artifacts" / "vllm_vitis_smoke"
DEFAULT_HLSFACTORY_ROOT = REPO / "benchmarks_external" / "HLSFactory" / "polybench_float_small"


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as f:
        for line_index, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            record["_source_jsonl"] = str(path)
            record["_source_line_index"] = line_index
            yield record


def _largest_fenced_block(text: str) -> str:
    blocks = re.findall(r"```(?:[A-Za-z0-9_+.-]+)?\s*\n(.*?)```", text, flags=re.DOTALL)
    if not blocks:
        return ""
    return max(blocks, key=len).strip()


def _extract_code(record: dict[str, Any]) -> tuple[str, str]:
    content = str(((record.get("response") or {}).get("content")) or "")
    fenced = _largest_fenced_block(content)
    if fenced:
        return fenced + "\n", "largest_fenced_block"
    return content.strip() + "\n", "raw_response"


def _status(result: dict[str, Any] | None, *, passed_key: bool = False) -> str:
    if not result:
        return "not_run"
    if passed_key:
        if result.get("success") and result.get("passed"):
            return "pass"
    elif result.get("success"):
        return "pass"
    err = str(result.get("error") or "").lower()
    if "timed out" in err or "timeout" in err:
        return "timeout"
    return "fail"


def _select_records(
    records: list[dict[str, Any]],
    *,
    benchmarks: set[str],
    require_stop: bool,
    limit: int,
) -> list[dict[str, Any]]:
    selected = []
    seen: set[str] = set()
    for record in records:
        bench = str(((record.get("record") or {}).get("benchmark")) or "")
        if not bench:
            continue
        if benchmarks and bench not in benchmarks:
            continue
        if bench in seen:
            continue
        if record.get("status") != "ok":
            continue
        if require_stop and record.get("finish_reason") != "stop":
            continue
        code, _ = _extract_code(record)
        if "#pragma HLS" not in code:
            continue
        selected.append(record)
        seen.add(bench)
        if limit and len(selected) >= limit:
            break
    return selected


def _short_error(result: dict[str, Any] | None, max_chars: int = 1000) -> str:
    if not isinstance(result, dict):
        return ""
    err = result.get("error") or ""
    return str(err)[:max_chars]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as output:
        output.write(json.dumps(payload, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def _load_latest_rows(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return latest
    with path.open() as source:
        for line in source:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            benchmark = str(row.get("benchmark") or "")
            if benchmark:
                latest[benchmark] = row
    return latest


def _run_case(
    record: dict[str, Any],
    *,
    hlsfactory_root: Path,
    artifact_dir: Path,
    work_root: Path,
    run_csim: bool,
    correctness_first: bool,
) -> dict[str, Any]:
    bench = str((record.get("record") or {}).get("benchmark"))
    bench_dir = hlsfactory_root / bench
    case_artifact_dir = artifact_dir / bench
    case_work_dir = work_root / bench
    case_artifact_dir.mkdir(parents=True, exist_ok=True)
    case_work_dir.mkdir(parents=True, exist_ok=True)

    code, extraction = _extract_code(record)
    code_path = case_artifact_dir / "generated.cpp"
    code_path.write_text(code)

    inputs = _load_benchmark_inputs(str(bench_dir))
    meta = inputs["meta"]
    top = meta.get("hls_top", "workload")
    header_code = inputs.get("header_code", "")
    header_name = inputs.get("header_name") or "kernel.h"
    extra_files = inputs.get("extra_files", [])
    testbench_code = inputs.get("testbench_code", "")

    csim = None
    csim_elapsed = None
    csim_status = "not_run"
    supports_csim = bool(
        run_csim and testbench_code and meta.get("supports_csim")
    )
    if supports_csim and correctness_first:
        csim_t0 = time.time()
        csim = hls_eval.run_csim(
            code,
            testbench_code,
            header_code,
            header_name=header_name,
            top_function=top,
            part=hls_eval.DEFAULT_PART,
            clock_ns=hls_eval.DEFAULT_CLOCK_NS,
            work_dir=str(case_work_dir / "csim_work"),
            extra_files=extra_files,
        )
        csim_elapsed = round(time.time() - csim_t0, 3)
        csim_status = _status(csim, passed_key=True)

    synth = None
    synth_elapsed = None
    if supports_csim and correctness_first and csim_status != "pass":
        synth_status = "skipped"
        synth_skip_reason = f"correctness_first_csim_{csim_status}"
    else:
        synth_t0 = time.time()
        synth = hls_eval.run_hls_synthesis(
            code,
            header_code,
            header_name=header_name,
            top_function=top,
            part=hls_eval.DEFAULT_PART,
            clock_ns=hls_eval.DEFAULT_CLOCK_NS,
            work_dir=str(case_work_dir / "csynth_work"),
            extra_files=extra_files,
        )
        synth_elapsed = round(time.time() - synth_t0, 3)
        synth_status = _status(synth)
        synth_skip_reason = None

    if (
        supports_csim
        and not correctness_first
        and synth_status == "pass"
    ):
        csim_t0 = time.time()
        csim = hls_eval.run_csim(
            code,
            testbench_code,
            header_code,
            header_name=header_name,
            top_function=top,
            part=hls_eval.DEFAULT_PART,
            clock_ns=hls_eval.DEFAULT_CLOCK_NS,
            work_dir=str(case_work_dir / "csim_work"),
            extra_files=extra_files,
        )
        csim_elapsed = round(time.time() - csim_t0, 3)
        csim_status = _status(csim, passed_key=True)

    report = synth.get("report") if isinstance(synth, dict) else {}
    return {
        "schema_version": "vllm_vitis_smoke_v1",
        "benchmark": bench,
        "source_jsonl": record.get("_source_jsonl"),
        "source_line_index": record.get("_source_line_index"),
        "vllm": {
            "finish_reason": record.get("finish_reason"),
            "response_chars": ((record.get("signals") or {}).get("response_chars")),
            "largest_fenced_block_chars": ((record.get("signals") or {}).get("largest_fenced_block_chars")),
            "row_index": ((record.get("record") or {}).get("row_index")),
        },
        "extraction": extraction,
        "generated_code_path": str(code_path),
        "work_root": str(case_work_dir),
        "vitis": {
            "version": os.getenv("C2HLS_VITIS_VERSION", "2023.2"),
            "part": hls_eval.DEFAULT_PART,
            "clock_ns": hls_eval.DEFAULT_CLOCK_NS,
            "flow_target": hls_eval.DEFAULT_FLOW_TARGET,
        },
        "synth": {
            "status": synth_status,
            "runtime_seconds": synth_elapsed,
            "work_dir": (
                str(case_work_dir / "csynth_work")
                if synth is not None
                else None
            ),
            "skip_reason": synth_skip_reason,
            "error": _short_error(synth),
            "latency_cycles": report.get("latency_cycles") if isinstance(report, dict) else None,
            "latency_ns": report.get("latency_ns") if isinstance(report, dict) else None,
            "fmax_mhz": report.get("fmax_mhz") if isinstance(report, dict) else None,
            "bram": report.get("bram") if isinstance(report, dict) else None,
            "dsp": report.get("dsp") if isinstance(report, dict) else None,
            "ff": report.get("ff") if isinstance(report, dict) else None,
            "lut": report.get("lut") if isinstance(report, dict) else None,
        },
        "csim": {
            "status": csim_status,
            "runtime_seconds": csim_elapsed,
            "work_dir": (
                str(case_work_dir / "csim_work")
                if csim is not None
                else None
            ),
            "error": _short_error(csim),
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_name = (
        args.run_name
        or f"vllm_vitis_smoke_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir = Path(args.output_dir).resolve() / run_name
    work_root = (
        Path(args.work_root).resolve() / run_name
        if args.work_root
        else output_dir / "work"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)
    out_jsonl = output_dir / "results.jsonl"
    out_summary = output_dir / "summary.json"
    out_md = output_dir / "summary.md"
    heartbeat_path = (
        Path(args.heartbeat_path).resolve()
        if args.heartbeat_path
        else output_dir / "heartbeat.json"
    )

    records: list[dict[str, Any]] = []
    for input_path in args.input_jsonl:
        records.extend(_read_jsonl(Path(input_path)))
    selected = _select_records(
        records,
        benchmarks=set(args.benchmark or []),
        require_stop=not args.allow_non_stop,
        limit=args.limit,
    )

    if args.resume:
        latest_by_benchmark = _load_latest_rows(out_jsonl)
    else:
        out_jsonl.write_text("")
        latest_by_benchmark = {}

    selected_benchmarks = [
        str((record.get("record") or {}).get("benchmark") or "")
        for record in selected
    ]
    latest_by_benchmark = {
        benchmark: row
        for benchmark, row in latest_by_benchmark.items()
        if benchmark in selected_benchmarks
    }

    def checkpoint(
        state: str,
        current_benchmark: str | None = None,
    ) -> dict[str, Any]:
        rows = [
            latest_by_benchmark[benchmark]
            for benchmark in selected_benchmarks
            if benchmark in latest_by_benchmark
        ]
        summary = {
            "schema_version": "vllm_vitis_smoke_v1_summary",
            "created_at": dt.datetime.now().isoformat(timespec="seconds"),
            "state": state,
            "current_benchmark": current_benchmark,
            "inputs": [str(Path(p)) for p in args.input_jsonl],
            "output_dir": str(output_dir),
            "work_root": str(work_root),
            "jsonl": str(out_jsonl),
            "selected": selected_benchmarks,
            "completed": [row["benchmark"] for row in rows],
            "counts": {
                "selected": len(selected),
                "completed": len(rows),
                "pending": max(0, len(selected) - len(rows)),
                "synth_pass": sum(
                    1 for row in rows if row["synth"]["status"] == "pass"
                ),
                "synth_fail": sum(
                    1 for row in rows if row["synth"]["status"] == "fail"
                ),
                "synth_timeout": sum(
                    1 for row in rows if row["synth"]["status"] == "timeout"
                ),
                "synth_skipped": sum(
                    1 for row in rows if row["synth"]["status"] == "skipped"
                ),
                "synth_not_run": sum(
                    1 for row in rows if row["synth"]["status"] == "not_run"
                ),
                "csim_pass": sum(
                    1 for row in rows if row["csim"]["status"] == "pass"
                ),
                "csim_fail": sum(
                    1 for row in rows if row["csim"]["status"] == "fail"
                ),
                "csim_timeout": sum(
                    1 for row in rows if row["csim"]["status"] == "timeout"
                ),
                "csim_not_run": sum(
                    1 for row in rows if row["csim"]["status"] == "not_run"
                ),
                "infrastructure_error": sum(
                    1 for row in rows if row.get("infrastructure_error")
                ),
            },
            "policy": {
                "correctness_first": args.correctness_first,
                "run_csim": not args.no_csim,
                "resume": args.resume,
                "retry_failed": args.retry_failed,
            },
            "vitis": {
                "settings": os.getenv("C2HLS_VITIS_SETTINGS"),
                "version": os.getenv("C2HLS_VITIS_VERSION", "2023.2"),
                "part": hls_eval.DEFAULT_PART,
                "clock_ns": hls_eval.DEFAULT_CLOCK_NS,
                "flow_target": hls_eval.DEFAULT_FLOW_TARGET,
                "synth_timeout_seconds": hls_eval.SYNTH_TIMEOUT,
                "csim_timeout_seconds": hls_eval.CSIM_TIMEOUT,
                "user_home": os.getenv("C2HLS_VITIS_USER_HOME"),
            },
            "rows": rows,
        }
        _write_json(out_summary, summary)
        _write_json(
            heartbeat_path,
            {
                "schema_version": "vllm_vitis_smoke_v1_heartbeat",
                "updated_at": dt.datetime.now().isoformat(timespec="seconds"),
                "pid": os.getpid(),
                "state": state,
                "current_benchmark": current_benchmark,
                "selected": len(selected),
                "completed": len(rows),
                "summary": str(out_summary),
                "work_root": str(work_root),
            },
        )

        lines = [
            "# vLLM Vitis Smoke",
            "",
            f"- State: `{state}`",
            f"- JSONL: `{out_jsonl}`",
            f"- Work root: `{work_root}`",
            f"- Vitis: `{summary['vitis']['version']}` / `{summary['vitis']['part']}` / `{summary['vitis']['clock_ns']} ns`",
            f"- Correctness first: `{args.correctness_first}`",
            "",
            "| bench | synth | csim | cycles | fmax MHz | BRAM | DSP | FF | LUT | error |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        for row in rows:
            err = (
                row["synth"]["error"]
                or row["csim"]["error"]
                or str(row.get("infrastructure_error") or "")
            )
            if len(err) > 120:
                err = err[:117] + "..."
            lines.append(
                "| {bench} | {synth} | {csim} | {cycles} | {fmax} | {bram} | {dsp} | {ff} | {lut} | {err} |".format(
                    bench=row["benchmark"],
                    synth=row["synth"]["status"],
                    csim=row["csim"]["status"],
                    cycles=row["synth"]["latency_cycles"] or "-",
                    fmax=row["synth"]["fmax_mhz"] or "-",
                    bram=row["synth"]["bram"] or "-",
                    dsp=row["synth"]["dsp"] or "-",
                    ff=row["synth"]["ff"] or "-",
                    lut=row["synth"]["lut"] or "-",
                    err=err.replace("|", "/"),
                )
            )
        temporary_md = out_md.with_suffix(".md.tmp")
        temporary_md.write_text("\n".join(lines) + "\n")
        temporary_md.replace(out_md)
        return summary

    checkpoint("running")
    consecutive_infrastructure_errors = 0
    state = "complete"
    for idx, record in enumerate(selected, 1):
        bench = str((record.get("record") or {}).get("benchmark") or "")
        existing = latest_by_benchmark.get(bench)
        if existing and not args.retry_failed:
            print(f"[{idx}/{len(selected)}] {bench} resume_skip", flush=True)
            continue
        if (
            existing
            and args.retry_failed
            and existing.get("synth", {}).get("status") == "pass"
            and (
                args.no_csim
                or existing.get("csim", {}).get("status") in {"pass", "not_run"}
            )
        ):
            print(f"[{idx}/{len(selected)}] {bench} resume_skip_pass", flush=True)
            continue

        print(f"[{idx}/{len(selected)}] {bench}", flush=True)
        checkpoint("running", bench)
        try:
            row = _run_case(
                record,
                hlsfactory_root=Path(args.hlsfactory_root),
                artifact_dir=output_dir,
                work_root=work_root,
                run_csim=not args.no_csim,
                correctness_first=args.correctness_first,
            )
            consecutive_infrastructure_errors = 0
        except Exception as exc:  # noqa: BLE001 - preserve campaign failure
            error = {"type": type(exc).__name__, "message": str(exc)}
            row = {
                "schema_version": "vllm_vitis_smoke_v1",
                "benchmark": bench,
                "source_jsonl": record.get("_source_jsonl"),
                "source_line_index": record.get("_source_line_index"),
                "work_root": str(work_root / bench),
                "infrastructure_error": error,
                "synth": {
                    "status": "not_run",
                    "runtime_seconds": None,
                    "work_dir": None,
                    "skip_reason": "infrastructure_error",
                    "error": str(error),
                    "latency_cycles": None,
                    "latency_ns": None,
                    "fmax_mhz": None,
                    "bram": None,
                    "dsp": None,
                    "ff": None,
                    "lut": None,
                },
                "csim": {
                    "status": "not_run",
                    "runtime_seconds": None,
                    "work_dir": None,
                    "error": str(error),
                },
            }
            consecutive_infrastructure_errors += 1

        _append_jsonl(out_jsonl, row)
        latest_by_benchmark[bench] = row
        checkpoint("running")
        print(
            json.dumps(
                {
                    "benchmark": row["benchmark"],
                    "synth": row["synth"]["status"],
                    "csim": row["csim"]["status"],
                    "latency_cycles": row["synth"]["latency_cycles"],
                    "synth_seconds": row["synth"]["runtime_seconds"],
                    "csim_seconds": row["csim"]["runtime_seconds"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if (
            consecutive_infrastructure_errors
            >= args.max_consecutive_infrastructure_errors
        ):
            state = "aborted_consecutive_infrastructure_errors"
            break

    summary = checkpoint(state)
    print(json.dumps(summary["counts"], indent=2, sort_keys=True))
    print(f"summary={out_summary}", flush=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", action="append", required=True)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--run-name", default="")
    parser.add_argument(
        "--work-root",
        default="",
        help="Base for bulky Vitis work directories; run_name is appended.",
    )
    parser.add_argument(
        "--heartbeat-path",
        default="",
        help="Atomic heartbeat JSON path. Defaults inside the compact output.",
    )
    parser.add_argument("--hlsfactory-root", default=str(DEFAULT_HLSFACTORY_ROOT))
    parser.add_argument("--benchmark", action="append", default=[])
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--allow-non-stop", action="store_true")
    parser.add_argument("--no-csim", action="store_true")
    parser.add_argument(
        "--correctness-first",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run supported CSim before CSynth and skip invalid candidates.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--retry-failed",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--max-consecutive-infrastructure-errors",
        type=int,
        default=3,
    )
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
