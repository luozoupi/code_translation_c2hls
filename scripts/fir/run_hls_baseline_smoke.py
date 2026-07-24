#!/usr/bin/env python3
"""Fir HLS baseline smoke — csynth + csim via hls_eval (no LLM).

Validates the C2HLS evaluation path on Fir with the Apptainer Vitis SIF.
Artifacts::

    artifacts/fir/hls_baseline_smoke_<stamp>/matrix.json

Examples::

    bash scripts/fir/run_hls_baseline_smoke.sh
    python3 scripts/fir/run_hls_baseline_smoke.py --fir --benches hlsfactory_2mm,hlsfactory_lu,hlsfactory_3mm
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from c2hls_paths import BENCHMARKS_DIR, configure_site, site_artifacts_dir

DEFAULT_BENCHES = ("hlsfactory_2mm", "hlsfactory_lu", "hlsfactory_3mm")
ARTIFACT_PREFIX = "hls_baseline_smoke"


def _load_bench_inputs(bench_dir: Path) -> dict[str, Any]:
    meta = json.loads((bench_dir / "metadata.json").read_text())
    header_name = meta.get("header_file") or "kernel.h"
    header_code = ""
    header_path = bench_dir / header_name
    if header_path.is_file():
        header_code = header_path.read_text()

    gt_file = (
        meta.get("gold_hls_baseline_file")
        or meta.get("preferred_gt_file")
        or "hls_baseline.cpp"
    )
    gt_path = bench_dir / gt_file
    if not gt_path.is_file():
        raise FileNotFoundError(f"missing ground truth {gt_path}")

    tb_file = meta.get("testbench_file") or "testbench.cpp"
    tb_path = bench_dir / tb_file
    testbench_code = tb_path.read_text() if tb_path.is_file() else ""

    extra_files: list[tuple[str, str]] = []
    for rel in meta.get("support_files") or []:
        p = bench_dir / rel
        if p.is_file():
            extra_files.append((rel, p.read_text()))

    return {
        "meta": meta,
        "header_name": header_name,
        "header_code": header_code,
        "ground_truth_code": gt_path.read_text(),
        "testbench_code": testbench_code,
        "extra_files": extra_files,
    }


def _split_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _resolve_benches(requested: list[str]) -> list[tuple[str, Path]]:
    available: dict[str, Path] = {}
    for meta_path in sorted(BENCHMARKS_DIR.glob("*/metadata.json")):
        try:
            meta = json.loads(meta_path.read_text())
        except json.JSONDecodeError:
            continue
        name = meta.get("benchmark") or meta_path.parent.name
        available[name] = meta_path.parent
    missing = [name for name in requested if name not in available]
    if missing:
        raise ValueError(f"unknown benchmark(s): {missing}")
    return [(name, available[name]) for name in requested]


def _status_from_result(result: dict[str, Any] | None, *, passed_key: str = "success") -> str:
    if not result:
        return "not_run"
    return "pass" if result.get(passed_key) else "fail"


def main() -> int:
    parser = argparse.ArgumentParser(description="Fir HLS baseline smoke (csynth + csim, no LLM)")
    parser.add_argument("--fir", action="store_true", required=True)
    parser.add_argument("--benches", type=str, default=",".join(DEFAULT_BENCHES))
    parser.add_argument("--out-root", type=str, default="")
    parser.add_argument("--stamp", type=str, default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    os.environ["C2HLS_SITE"] = "fir"
    configure_site("fir")

    import hls_eval  # noqa: E402

    stamp = args.stamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(
        args.out_root
        or site_artifacts_dir("fir") / f"{ARTIFACT_PREFIX}_{stamp}"
    )
    benches = _resolve_benches(_split_csv(args.benches))

    plan = {
        "site": "fir",
        "kind": ARTIFACT_PREFIX,
        "stamp": stamp,
        "out_root": str(out_root),
        "benches": [name for name, _ in benches],
        "vitis_settings": hls_eval.VITIS_SETTINGS,
        "vitis_run": shutil.which("vitis-run") or "",
        "use_container": os.getenv("C2HLS_USE_CONTAINER", ""),
        "xilinx_sif": os.getenv("XILINX_SIF", ""),
        "tmp_root": os.getenv("C2HLS_TMP_ROOT", ""),
        "part": hls_eval.DEFAULT_PART,
        "clock_ns": hls_eval.DEFAULT_CLOCK_NS,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "manifest.json").write_text(json.dumps(plan, indent=2) + "\n")

    print(f"site=fir benches={len(benches)} out_root={out_root}")
    print(f"vitis-run={plan['vitis_run']} container={plan['use_container']}")
    for bench, _ in benches:
        print(f"  - {bench}")

    if args.dry_run:
        print("dry-run ok")
        return 0

    rows: list[dict[str, Any]] = []
    for bench, bench_dir in benches:
        try:
            inputs = _load_bench_inputs(bench_dir)
        except (OSError, json.JSONDecodeError, FileNotFoundError) as exc:
            rows.append({
                "bench": bench,
                "top": "",
                "synth_status": "fail",
                "csim_status": "not_run",
                "error": str(exc),
            })
            continue

        meta = inputs["meta"]
        top = meta.get("hls_top") or meta.get("kernel_top") or "workload"
        code = inputs["ground_truth_code"]
        tb = inputs.get("testbench_code") or ""
        variant = meta.get("baseline_variant") or meta.get("gold_hls_baseline_file") or "hls_baseline.cpp"

        print(f"START {bench} top={top}", flush=True)
        t0 = time.time()

        synth = hls_eval.run_hls_synthesis(
            code,
            inputs.get("header_code", ""),
            header_name=inputs.get("header_name") or "kernel.h",
            top_function=top,
            part=hls_eval.DEFAULT_PART,
            clock_ns=hls_eval.DEFAULT_CLOCK_NS,
            extra_files=inputs.get("extra_files", []),
        )
        synth_status = _status_from_result(synth)

        csim_status = "not_run"
        csim = None
        if synth_status == "pass" and tb and meta.get("supports_csim", True):
            csim = hls_eval.run_csim(
                code,
                tb,
                inputs.get("header_code", ""),
                header_name=inputs.get("header_name") or "kernel.h",
                top_function=top,
                part=hls_eval.DEFAULT_PART,
                clock_ns=hls_eval.DEFAULT_CLOCK_NS,
                extra_files=inputs.get("extra_files", []),
            )
            csim_status = _status_from_result(csim)

        elapsed = round(time.time() - t0, 1)
        row = {
            "bench": bench,
            "top": top,
            "variant": variant,
            "synth_status": synth_status,
            "csim_status": csim_status,
            "wallclock_s": elapsed,
            "synth_work_dir": synth.get("work_dir") if isinstance(synth, dict) else None,
            "csim_work_dir": csim.get("work_dir") if isinstance(csim, dict) else None,
            "synth_error": (synth.get("error") or "")[:500] if synth_status != "pass" else None,
            "csim_error": (csim.get("error") or "")[:500] if csim_status == "fail" else None,
        }
        rows.append(row)
        ok = synth_status == "pass" and csim_status in ("pass", "not_run")
        print(
            f"DONE {bench} synth={synth_status} csim={csim_status} ok={ok} elapsed={elapsed}s",
            flush=True,
        )

    (out_root / "matrix.json").write_text(json.dumps(rows, indent=2) + "\n")
    failed = [
        r["bench"]
        for r in rows
        if r["synth_status"] != "pass" or r["csim_status"] not in ("pass", "not_run")
    ]
    if failed:
        print(f"FAIL benches: {failed}", flush=True)
        return 1
    print("PASS all benches", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
