#!/usr/bin/env python3
"""Emit schema-1.0 JSONL for hlsfactory baseline csynth/csim runs.

Default corpus is benchmarks/ (hls_baseline.cpp). Set
C2HLS_HLSFACTORY_BASELINE_CORPUS=benchmarks_cosim to run csynth/csim on
benchmarks_cosim/hls_baseline_cosim.cpp instead (full cosim rtl_sim is merged
later via export_pc2_baseline_cosim_jsonl.py).

Example (PC2, naive):
  module load fpga xilinx/xrt/2.16
  C2HLS_HLSFACTORY_BASELINE_COSIM=0 python3 misc/generate_hlsfactory_baseline_jsonl.py

Example (PC2, fixed cosim corpus):
  C2HLS_HLSFACTORY_BASELINE_CORPUS=benchmarks_cosim \
  C2HLS_HLSFACTORY_BASELINE_STAMP=20260616_benchmarks \
  C2HLS_HLSFACTORY_BASELINE_JSONL=misc/hlsfactory_cosim_baseline_u280_20260616_benchmarks.jsonl \
  python3 misc/generate_hlsfactory_baseline_jsonl.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from c2hls_paths import apply_runtime_defaults, configure_site  # noqa: E402

configure_site("pc2")
apply_runtime_defaults()


def _load_hls_eval():
    """Import hls_eval on Python 3.9 (needs postponed annotations)."""
    path = REPO / "hls_eval.py"
    source = "from __future__ import annotations\n" + path.read_text()
    spec = importlib.util.spec_from_file_location("hls_eval", path)
    module = importlib.util.module_from_spec(spec)
    exec(compile(source, str(path), "exec"), module.__dict__)  # noqa: S102
    return module


hls_eval = _load_hls_eval()

from export_schema_jsonl import (  # noqa: E402
    SCHEMA_VERSION,
    TARGET_CSIM,
    TARGET_COSIM,
    TARGET_CSYNTH,
    _build_hls_synth_payload,
    _build_implementation,
    _build_problem,
    _build_run,
    validate_jsonl,
)

CORPUS = os.getenv("C2HLS_HLSFACTORY_BASELINE_CORPUS", "benchmarks").strip()
if CORPUS not in ("benchmarks", "benchmarks_cosim"):
    raise SystemExit(f"unsupported C2HLS_HLSFACTORY_BASELINE_CORPUS={CORPUS!r}")

STAMP = os.getenv("C2HLS_HLSFACTORY_BASELINE_STAMP") or datetime.now().strftime("%Y%m%d_%H%M%S")
_DEFAULT_JSONL_NAME = (
    f"hlsfactory_cosim_baseline_u280_{STAMP}.jsonl"
    if CORPUS == "benchmarks_cosim"
    else f"hlsfactory_baseline_u280_{STAMP}.jsonl"
)
OUT_JSONL = Path(
    os.getenv(
        "C2HLS_HLSFACTORY_BASELINE_JSONL",
        str(REPO / "misc" / _DEFAULT_JSONL_NAME),
    )
)
BENCHMARKS_DIR = REPO / CORPUS
ORIGIN_VERSION_PREFIX = (
    "benchmarks_cosim_baseline" if CORPUS == "benchmarks_cosim" else "benchmarks_hls_baseline"
)
HLSFACTORY_UPSTREAM_REPO = "https://github.com/sharc-lab/HLSFactory/blob/main"


def _upstream_github_url(remote_path: str | None) -> str | None:
    if not remote_path:
        return None
    marker = "HLSFactory/"
    if marker in remote_path:
        return f"{HLSFACTORY_UPSTREAM_REPO}/{remote_path.split(marker, 1)[1]}"
    return None


def _resolve_origin_paths(bench_dir: Path, meta: dict[str, Any], variant_file: str) -> dict[str, Any]:
    """Map provenance to local corpus files.

    Corpus chain (see prepare_hlsfactory_cosim_benchmarks.py):
      hls_baseline.cpp -> hls_baseline_cosim.cpp (+ interfaces / body fixes)
        -> plain.cpp (LLM input, pragmas stripped)
        -> gold_hls_source.cpp (cosim oracle, same as hls_baseline_cosim.cpp)
    - plain.cpp: LLM translation input (no HLS pragmas)
    - hls_baseline_cosim.cpp: ground-truth HLS baseline (csynth/cosim GT)
    - gold_hls_source.cpp: cosim functional oracle (identical to hls_baseline_cosim.cpp)
    """
    gt_file = variant_file or meta.get("gold_hls_baseline_file", "hls_baseline.cpp")
    baseline_path = bench_dir / gt_file
    plain_file = meta.get("plain_c_file", "plain.cpp")
    plain_path = bench_dir / plain_file
    gold_file = meta.get("gold_hls_source_file", "gold_hls_source.cpp")
    gold_path = bench_dir / gold_file

    remote = meta.get("algorithm_source_path") or meta.get("gold_hls_source_path")
    if not remote:
        variants = meta.get("variants") or []
        if variants:
            remote = variants[0].get("source_path")

    return {
        "source_path": str(plain_path.resolve()) if plain_path.is_file() else str(baseline_path.resolve()),
        "plain_path": str(plain_path.resolve()) if plain_path.is_file() else None,
        "baseline_path": str(baseline_path.resolve()),
        "gold_hls_source_path": str(gold_path.resolve()) if gold_path.is_file() else None,
        "source_file": gt_file,
        "upstream_source_url": _upstream_github_url(remote),
    }


def _load_bench_inputs(bench_dir: Path) -> dict[str, Any]:
    meta = json.loads((bench_dir / "metadata.json").read_text())
    header_name = meta.get("header_file") or "kernel.h"
    header_code = (bench_dir / header_name).read_text() if (bench_dir / header_name).exists() else ""
    if CORPUS == "benchmarks_cosim":
        gt_file = meta.get("cosim_kernel_file", "hls_baseline_cosim.cpp")
        tb_file = meta.get("cosim_testbench_file", "testbench_cosim.cpp")
    else:
        gt_file = meta.get("gold_hls_baseline_file", "hls_baseline.cpp")
        tb_file = meta.get("testbench_file", "testbench.cpp")
    baseline_code = (bench_dir / gt_file).read_text()
    testbench_code = (bench_dir / tb_file).read_text() if (bench_dir / tb_file).exists() else ""
    variant = (meta.get("variants") or [{}])[0]
    extra_files: list[dict[str, str]] = []
    extra_file_paths: set[str] = set()
    if CORPUS == "benchmarks_cosim":
        for rel_path in meta.get("cosim_support_files", []):
            file_path = bench_dir / rel_path
            if file_path.exists():
                extra_files.append({
                    "path": rel_path,
                    "content": file_path.read_text(),
                    "tb": True,
                })
                extra_file_paths.add(rel_path)
        gold_src = meta.get("gold_hls_source_file") or "gold_hls_source.cpp"
        gold_path = bench_dir / gold_src
        if gold_path.exists() and gold_src not in extra_file_paths:
            extra_files.append({
                "path": gold_src,
                "content": gold_path.read_text(),
                "tb": False,
            })
            extra_file_paths.add(gold_src)
    else:
        for rel_path in meta.get("support_files", []):
            file_path = bench_dir / rel_path
            if file_path.exists():
                extra_files.append({"path": rel_path, "content": file_path.read_text()})
                extra_file_paths.add(rel_path)
    support_dir = bench_dir / "support"
    if support_dir.exists():
        for file_path in sorted(support_dir.rglob("*")):
            if not file_path.is_file():
                continue
            rel_path = str(file_path.relative_to(bench_dir))
            if rel_path in extra_file_paths:
                continue
            extra_files.append({"path": rel_path, "content": file_path.read_text()})
            extra_file_paths.add(rel_path)
    return {
        "meta": meta,
        "header_name": header_name,
        "header_code": header_code,
        "baseline_code": baseline_code,
        "testbench_code": testbench_code,
        "variant_name": (
            "baseline_cosim"
            if CORPUS == "benchmarks_cosim"
            else variant.get("name", f"{meta['benchmark']}_0_baseline")
        ),
        "variant_file": gt_file if CORPUS == "benchmarks_cosim" else variant.get("file", gt_file),
        "origin_paths": _resolve_origin_paths(bench_dir, meta, variant.get("file", gt_file)),
        "extra_files": extra_files,
    }


def _bench_dirs() -> list[Path]:
    raw = os.getenv("C2HLS_HLSFACTORY_BASELINE_BENCHES", "").strip()
    dirs = sorted(
        p for p in BENCHMARKS_DIR.glob("hlsfactory_*")
        if p.is_dir() and (p / "metadata.json").is_file()
    )
    if raw:
        wanted = {item.strip() for item in raw.split(",") if item.strip()}
        dirs = [
            p for p in dirs
            if p.name in wanted
            or p.name.removeprefix("hlsfactory_") in wanted
        ]
    return dirs


def _status_from_result(result: dict[str, Any] | None, *, passed_key: bool = False) -> str:
    if not result:
        return "not_run"
    if passed_key:
        if result.get("success") and result.get("passed"):
            return "pass"
    elif result.get("success"):
        return "pass"
    err = str(result.get("error") or "")
    if "timed out" in err.lower() or "timeout" in err.lower():
        return "timeout"
    return "fail"


def _top_model_payload(report: dict[str, Any], status: str, top: str) -> dict[str, Any]:
    payload = _build_hls_synth_payload(
        report,
        hls_eval.DEFAULT_PART,
        hls_eval.DEFAULT_CLOCK_NS,
        status=status,
    )
    ua = payload.get("UserAssignments")
    if isinstance(ua, dict):
        ua["TopModelName"] = top
    return payload


def _jsonl_record(
    *,
    report_type: str,
    target: str,
    runtime_seconds: float | None,
    meta: dict[str, Any],
    variant_name: str,
    origin_meta: dict[str, Any],
    payload_key: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "report_type": report_type,
        "run": _build_run(
            target,
            hls_eval.DEFAULT_PART,
            runtime_seconds,
            {
                "vitis_version": os.getenv("C2HLS_VITIS_VERSION", "2023.2"),
                "flow_target": hls_eval.DEFAULT_FLOW_TARGET,
                "clock_ns": hls_eval.DEFAULT_CLOCK_NS,
            },
        ),
        "problem": _build_problem(meta),
        "implementation": _build_implementation(
            meta,
            variant_name=variant_name,
            origin_version=f"{ORIGIN_VERSION_PREFIX}_{STAMP}",
            origin_meta=origin_meta,
        ),
        payload_key: payload,
    }


def main() -> int:
    run_csim = os.getenv("C2HLS_HLSFACTORY_BASELINE_CSIM", "1") != "0"
    run_cosim = os.getenv("C2HLS_HLSFACTORY_BASELINE_COSIM", "0") != "0"
    bench_dirs = _bench_dirs()
    if not bench_dirs:
        print("No benchmarks/hlsfactory_* directories found", file=sys.stderr)
        return 1

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSONL.write_text("")
    records_written = 0
    rows: list[dict[str, Any]] = []

    print(f"corpus={CORPUS} benches={len(bench_dirs)} jsonl={OUT_JSONL}", flush=True)
    print(
        f"vitis={os.getenv('C2HLS_VITIS_VERSION')} part={hls_eval.DEFAULT_PART} "
        f"clock={hls_eval.DEFAULT_CLOCK_NS} csim={run_csim} cosim={run_cosim}",
        flush=True,
    )

    for idx, bench_dir in enumerate(bench_dirs, 1):
        inputs = _load_bench_inputs(bench_dir)
        meta = dict(inputs["meta"])
        short_name = bench_dir.name.removeprefix("hlsfactory_").replace("-", "_")
        meta["group_path"] = [short_name]
        top = meta.get("hls_top", "workload")
        candidate_code = inputs["baseline_code"]
        variant_name = inputs["variant_name"]
        print(f"[{idx}/{len(bench_dirs)}] {bench_dir.name} variant={variant_name}", flush=True)

        provenance = meta.get("provenance") or {}
        paths = inputs["origin_paths"]
        origin_meta = {
            "direct_script": Path(__file__).name,
            "corpus": CORPUS,
            "source_repo": meta.get("source_repo"),
            "source_path": paths["source_path"],
            "plain_path": paths["plain_path"],
            "baseline_path": paths["baseline_path"],
            "gold_hls_source_path": paths["gold_hls_source_path"],
            "upstream_source_url": paths["upstream_source_url"],
            "source_file": paths["source_file"],
            "benchmark_dir": str(bench_dir.resolve()),
            "parent_benchmark_dir": meta.get("parent_benchmark_dir"),
            "cosim_kernel_file": meta.get("cosim_kernel_file"),
            "reference_role": "hlsfactory_ground_truth",
            "gold_hls_baseline_sha256": provenance.get("gold_hls_baseline_sha256"),
            "plain_c_sha256": provenance.get("plain_c_sha256"),
            "synth_work_dir": None,
            "error": None,
        }

        synth_t0 = time.time()
        synth = hls_eval.run_hls_synthesis(
            candidate_code,
            inputs.get("header_code", ""),
            header_name=inputs.get("header_name") or "kernel.h",
            top_function=top,
            part=hls_eval.DEFAULT_PART,
            clock_ns=hls_eval.DEFAULT_CLOCK_NS,
            extra_files=inputs.get("extra_files", []),
        )
        synth_elapsed = round(time.time() - synth_t0, 3)
        synth_status = _status_from_result(synth)
        report = synth.get("report") or {}
        origin_meta["synth_work_dir"] = synth.get("work_dir")
        if synth_status != "pass":
            origin_meta["error"] = (synth.get("error") or "")[:300]

        synth_record = _jsonl_record(
            report_type="hls_synth",
            target=TARGET_CSYNTH,
            runtime_seconds=synth_elapsed,
            meta=meta,
            variant_name=variant_name,
            origin_meta=origin_meta,
            payload_key="hls_synth",
            payload=_top_model_payload(report, synth_status, top),
        )
        with OUT_JSONL.open("a") as f:
            f.write(json.dumps(synth_record) + "\n")
        records_written += 1

        csim_status = "not_run"
        if synth_status == "pass" and run_csim and inputs.get("testbench_code") and meta.get("supports_csim"):
            csim_t0 = time.time()
            csim = hls_eval.run_csim(
                candidate_code,
                inputs.get("testbench_code", ""),
                inputs.get("header_code", ""),
                header_name=inputs.get("header_name") or "kernel.h",
                top_function=top,
                part=hls_eval.DEFAULT_PART,
                clock_ns=hls_eval.DEFAULT_CLOCK_NS,
                extra_files=inputs.get("extra_files", []),
            )
            csim_elapsed = round(time.time() - csim_t0, 3)
            csim_status = _status_from_result(csim, passed_key=True)
            csim_record = _jsonl_record(
                report_type="sw_run",
                target=TARGET_CSIM,
                runtime_seconds=csim_elapsed,
                meta=meta,
                variant_name=variant_name,
                origin_meta={
                    **origin_meta,
                    "csim_work_dir": csim.get("work_dir") if isinstance(csim, dict) else None,
                    "error": (csim.get("error") or "")[:300] if csim_status != "pass" else None,
                },
                payload_key="sw_run",
                payload={
                    "status": csim_status,
                    "error": (csim.get("error") or "")[:300] if csim_status != "pass" else None,
                },
            )
            with OUT_JSONL.open("a") as f:
                f.write(json.dumps(csim_record) + "\n")
            records_written += 1

        cosim_status = "not_run"
        cosim_cycles = None
        if synth_status == "pass" and run_cosim and inputs.get("testbench_code") and meta.get("supports_cosim"):
            cosim_t0 = time.time()
            cosim = hls_eval.run_cosim(
                candidate_code,
                inputs.get("testbench_code", ""),
                inputs.get("header_code", ""),
                header_name=inputs.get("header_name") or "kernel.h",
                top_function=top,
                part=hls_eval.DEFAULT_PART,
                clock_ns=hls_eval.DEFAULT_CLOCK_NS,
                extra_files=inputs.get("extra_files", []),
                interface_depths=meta.get("cosim_depths") or {},
            )
            cosim_elapsed = round(time.time() - cosim_t0, 3)
            cosim_status = _status_from_result(cosim, passed_key=True)
            cosim_cycles = cosim.get("kernel_runtime_cycles") if isinstance(cosim, dict) else None
            cosim_record = _jsonl_record(
                report_type="rtl_sim",
                target=TARGET_COSIM,
                runtime_seconds=cosim_elapsed,
                meta=meta,
                variant_name=variant_name,
                origin_meta={
                    **origin_meta,
                    "cosim_work_dir": cosim.get("work_dir") if isinstance(cosim, dict) else None,
                    "error": (cosim.get("error") or "")[:300] if cosim_status != "pass" else None,
                },
                payload_key="rtl_sim",
                payload={
                    "status": cosim_status,
                    "kernel_runtime_cycles": cosim_cycles,
                    "kernel_runtime_us": cosim.get("kernel_runtime_us") if isinstance(cosim, dict) else None,
                    "kernel_clock_freq_mhz": cosim.get("kernel_clock_freq_mhz") if isinstance(cosim, dict) else None,
                    "error": (cosim.get("error") or "")[:300] if cosim_status != "pass" else None,
                },
            )
            with OUT_JSONL.open("a") as f:
                f.write(json.dumps(cosim_record) + "\n")
            records_written += 1

        rows.append({
            "bench": bench_dir.name,
            "synth_status": synth_status,
            "csim_status": csim_status,
            "cosim_status": cosim_status,
            "latency_cycles": report.get("latency_cycles"),
            "cosim_cycles": cosim_cycles,
            "error": origin_meta.get("error"),
        })

    validation = validate_jsonl(OUT_JSONL, verbose=True)
    summary_path = OUT_JSONL.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(
            {
                "corpus": CORPUS,
                "stamp": STAMP,
                "jsonl": str(OUT_JSONL),
                "records_written": records_written,
                "validation": validation,
                "rows": rows,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"jsonl": str(OUT_JSONL), "summary": str(summary_path), "validation": validation}, indent=2))
    return 0 if validation.get("invalid", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
