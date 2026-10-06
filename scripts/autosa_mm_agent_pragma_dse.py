#!/usr/bin/env python3
"""Pragma-only DSE on the frozen autosa_mm flash kernel.

No LLM. No loop/algorithm rewrite. Search PIPELINE-adjacent UNROLL,
ARRAY_PARTITION, and optional DATAFLOW. Score vs AutoSA rank1 (4228 cycles).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
DEFAULT_SEED = (
    REPO
    / "artifacts/pc2/batch_parallel_autosa_mm_gap_20260818_mm_gap_flash_abi"
    / "variants/autosa_nav_n/autosa_mm/devstral2__flash__autosa__nav_n"
    / "autosa_mm_selected.cpp"
)
DEFAULT_BENCH = REPO / "related_work/benchmarks/autosa_ready/autosa_mm"
RANK1_CYCLES = 4228
DEFAULT_PART = "xcu280-fsvh2892-2L-e"
DEFAULT_CLOCK_NS = 3.33
COMPUTE_MARK = "    // Compute matrix multiplication"

Point = dict[str, Any]
PartSpec = tuple[str, int, int] | None


def load_seed_kernel(path: Path | None = None) -> str:
    seed_path = path or DEFAULT_SEED
    return seed_path.read_text(encoding="utf-8")


def _point(
    pid: str,
    *,
    unroll_k: int = 1,
    unroll_j: int = 1,
    part_a: PartSpec = None,
    part_b: PartSpec = None,
    part_c: PartSpec = None,
    dataflow: bool = False,
) -> Point:
    return {
        "id": pid,
        "unroll_k": unroll_k,
        "unroll_j": unroll_j,
        "part_a": part_a,
        "part_b": part_b,
        "part_c": part_c,
        "dataflow": dataflow,
    }


def enumerate_points() -> list[Point]:
    """Coordinated recipes, not a full cartesian product."""

    def part(kind: str, factor: int, dim: int) -> PartSpec:
        return (kind, factor, dim)

    return [
        _point("baseline"),
        _point("partAB_cyc4", part_a=part("cyclic", 4, 2), part_b=part("cyclic", 4, 2)),
        _point("partAB_cyc8", part_a=part("cyclic", 8, 2), part_b=part("cyclic", 8, 2)),
        _point("unroll_k4", unroll_k=4),
        _point("unroll_k8", unroll_k=8),
        _point(
            "uk2_partAB_cyc2",
            unroll_k=2,
            part_a=part("cyclic", 2, 2),
            part_b=part("cyclic", 2, 2),
        ),
        _point(
            "uk4_partAB_cyc4",
            unroll_k=4,
            part_a=part("cyclic", 4, 2),
            part_b=part("cyclic", 4, 2),
        ),
        _point(
            "uk8_partAB_cyc8",
            unroll_k=8,
            part_a=part("cyclic", 8, 2),
            part_b=part("cyclic", 8, 2),
        ),
        _point(
            "uk16_partAB_cyc16",
            unroll_k=16,
            part_a=part("cyclic", 16, 2),
            part_b=part("cyclic", 16, 2),
        ),
        _point(
            "uk4_partABC_cyc4",
            unroll_k=4,
            part_a=part("cyclic", 4, 2),
            part_b=part("cyclic", 4, 2),
            part_c=part("cyclic", 4, 2),
        ),
        _point(
            "uk8_partABC_cyc8",
            unroll_k=8,
            part_a=part("cyclic", 8, 2),
            part_b=part("cyclic", 8, 2),
            part_c=part("cyclic", 8, 2),
        ),
        _point(
            "uj4_partC_cyc4",
            unroll_j=4,
            part_b=part("cyclic", 4, 1),
            part_c=part("cyclic", 4, 2),
        ),
        _point(
            "uj8_partC_cyc8",
            unroll_j=8,
            part_b=part("cyclic", 8, 1),
            part_c=part("cyclic", 8, 2),
        ),
        _point("dataflow", dataflow=True),
        _point(
            "dataflow_uk4_partAB_cyc4",
            unroll_k=4,
            part_a=part("cyclic", 4, 2),
            part_b=part("cyclic", 4, 2),
            dataflow=True,
        ),
        _point(
            "partAB_complete_d2",
            part_a=part("complete", 0, 2),
            part_b=part("complete", 0, 2),
        ),
        _point(
            "partABC_complete_d2",
            part_a=part("complete", 0, 2),
            part_b=part("complete", 0, 2),
            part_c=part("complete", 0, 2),
        ),
        _point(
            "partAB_complete_all",
            part_a=part("complete", 0, 0),
            part_b=part("complete", 0, 0),
        ),
        _point(
            "uk4_partAB_complete_d2",
            unroll_k=4,
            part_a=part("complete", 0, 2),
            part_b=part("complete", 0, 2),
        ),
        _point(
            "uk8_partAB_complete_d2",
            unroll_k=8,
            part_a=part("complete", 0, 2),
            part_b=part("complete", 0, 2),
        ),
        _point(
            "uk16_partAB_complete_d2",
            unroll_k=16,
            part_a=part("complete", 0, 2),
            part_b=part("complete", 0, 2),
        ),
        _point(
            "uj8_partBC_complete",
            unroll_j=8,
            part_b=part("complete", 0, 1),
            part_c=part("complete", 0, 2),
        ),
        _point(
            "dataflow_uk4_partAB_complete_d2",
            unroll_k=4,
            part_a=part("complete", 0, 2),
            part_b=part("complete", 0, 2),
            dataflow=True,
        ),
    ]


def _partition_pragma(variable: str, spec: PartSpec) -> str:
    if not spec:
        return ""
    kind, factor, dim = spec
    if kind == "complete":
        return f"#pragma HLS ARRAY_PARTITION variable={variable} complete dim={dim}"
    return (
        f"#pragma HLS ARRAY_PARTITION variable={variable} {kind} "
        f"factor={factor} dim={dim}"
    )


def apply_pragmas(seed: str, point: Point) -> str:
    if COMPUTE_MARK not in seed:
        raise ValueError("frozen kernel is missing the compute nest marker")
    head, compute = seed.split(COMPUTE_MARK, 1)

    part_lines = [
        _partition_pragma("local_A", point.get("part_a")),
        _partition_pragma("local_B", point.get("part_b")),
        _partition_pragma("local_C", point.get("part_c")),
    ]
    part_block = "".join(f"    {line}\n" for line in part_lines if line)
    if part_block:
        needle = "    data_t local_C[I][J];\n"
        if needle not in head:
            raise ValueError("frozen kernel is missing local_C declaration")
        head = head.replace(needle, needle + part_block, 1)

    if point.get("dataflow"):
        iface = "#pragma HLS INTERFACE s_axilite port=return bundle=control\n"
        if iface not in head:
            raise ValueError("frozen kernel is missing return s_axilite pragma")
        head = head.replace(iface, iface + "#pragma HLS DATAFLOW\n", 1)

    unroll_j = int(point.get("unroll_j") or 1)
    if unroll_j > 1:
        j_loop = "        for (int j = 0; j < J; j++) {\n"
        if j_loop not in compute:
            raise ValueError("expected a compute j loop")
        compute = compute.replace(
            j_loop,
            j_loop + f"#pragma HLS UNROLL factor={unroll_j}\n",
            1,
        )

    unroll_k = int(point.get("unroll_k") or 1)
    if unroll_k > 1:
        k_pipe = "#pragma HLS PIPELINE II=1\n                local_C[i][j] +="
        if k_pipe not in compute:
            raise ValueError("expected pipelined compute MAC")
        compute = compute.replace(
            k_pipe,
            f"#pragma HLS PIPELINE II=1\n#pragma HLS UNROLL factor={unroll_k}\n                local_C[i][j] +=",
            1,
        )
    return head + COMPUTE_MARK + compute


def _compact_report(report: dict[str, Any] | None) -> dict[str, Any]:
    report = report or {}
    keep = (
        "latency_cycles",
        "latency_cycles_worst",
        "dsp",
        "bram",
        "lut",
        "ff",
        "uram",
        "interval",
        "estimated_clock_period_ns",
        "slack_ns",
        "fmax_mhz",
    )
    return {k: report.get(k) for k in keep}


def _run_point(
    *,
    code: str,
    header_code: str,
    testbench_code: str,
    work_dir: Path,
    part: str,
    clock_ns: float,
    run_csim: bool,
) -> dict[str, Any]:
    from hls_eval import run_csim as hls_csim, run_hls_synthesis

    result: dict[str, Any] = {"csim": None, "csynth": None}
    if run_csim:
        csim = hls_csim(
            code,
            testbench_code,
            header_code=header_code,
            header_name="kernel.h",
            top_function="autosa_mm",
            part=part,
            clock_ns=clock_ns,
            work_dir=str(work_dir / "csim"),
        )
        result["csim"] = {
            "success": bool(csim.get("success")),
            "passed": bool(csim.get("passed") or csim.get("success")),
            "error": csim.get("error"),
        }
        if not result["csim"]["passed"]:
            result["ok"] = False
            result["error"] = csim.get("error") or "csim failed"
            return result

    synth = run_hls_synthesis(
        code,
        header_code=header_code,
        header_name="kernel.h",
        top_function="autosa_mm",
        part=part,
        clock_ns=clock_ns,
        work_dir=str(work_dir / "csynth"),
    )
    result["csynth"] = {
        "success": bool(synth.get("success")),
        "error": synth.get("error"),
        "report": _compact_report(
            synth.get("report") if isinstance(synth.get("report"), dict) else {}
        ),
    }
    result["ok"] = bool(synth.get("success"))
    if not result["ok"]:
        result["error"] = synth.get("error") or "csynth failed"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=Path, default=DEFAULT_SEED)
    parser.add_argument("--bench", type=Path, default=DEFAULT_BENCH)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--point-id", action="append", dest="point_ids")
    parser.add_argument("--skip-csim", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--part", default=DEFAULT_PART)
    parser.add_argument("--clock-ns", type=float, default=DEFAULT_CLOCK_NS)
    args = parser.parse_args()

    seed = load_seed_kernel(args.seed)
    points = enumerate_points()
    if args.point_ids:
        wanted = set(args.point_ids)
        points = [p for p in points if p["id"] in wanted]
        missing = wanted - {p["id"] for p in points}
        if missing:
            raise SystemExit(f"unknown point id(s): {sorted(missing)}")

    if args.dry_run:
        payload = []
        for point in points:
            code = apply_pragmas(seed, point)
            payload.append({"id": point["id"], "point": point, "bytes": len(code)})
        print(json.dumps({"n": len(payload), "points": payload}, indent=2, default=str))
        return 0

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = args.out or (REPO / "artifacts/pc2" / f"autosa_mm_agent_pragma_dse_{stamp}")
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "seed.cpp").write_text(seed, encoding="utf-8")

    header_code = (args.bench / "kernel.h").read_text(encoding="utf-8")
    testbench_code = (args.bench / "testbench.cpp").read_text(encoding="utf-8")
    do_csim = not args.skip_csim

    rows: list[dict[str, Any]] = []
    for point in points:
        pid = point["id"]
        point_dir = out / "points" / pid
        point_dir.mkdir(parents=True, exist_ok=True)
        code = apply_pragmas(seed, point)
        (point_dir / "kernel.cpp").write_text(code, encoding="utf-8")
        (point_dir / "point.json").write_text(
            json.dumps(point, indent=2) + "\n", encoding="utf-8"
        )
        print(f"==> {pid}", flush=True)
        run = _run_point(
            code=code,
            header_code=header_code,
            testbench_code=testbench_code,
            work_dir=point_dir,
            part=args.part,
            clock_ns=args.clock_ns,
            run_csim=do_csim,
        )
        report = ((run.get("csynth") or {}).get("report") or {})
        latency = report.get("latency_cycles")
        row = {
            "id": pid,
            "ok": bool(run.get("ok")),
            "error": run.get("error"),
            "csim_passed": ((run.get("csim") or {}).get("passed") if do_csim else None),
            "latency_cycles": latency,
            "dsp": report.get("dsp"),
            "bram": report.get("bram"),
            "lut": report.get("lut"),
            "ff": report.get("ff"),
            "vs_rank1": (
                None if latency is None else round(float(latency) / RANK1_CYCLES, 3)
            ),
            "point": point,
        }
        (point_dir / "result.json").write_text(
            json.dumps(run, indent=2) + "\n", encoding="utf-8"
        )
        rows.append(row)
        print(
            json.dumps(
                {k: row[k] for k in ("id", "ok", "latency_cycles", "dsp", "vs_rank1", "error")}
            ),
            flush=True,
        )

    legal = [r for r in rows if r.get("ok") and r.get("latency_cycles") is not None]
    best = min(legal, key=lambda r: r["latency_cycles"]) if legal else None
    summary = {
        "stamp": stamp,
        "seed": str(args.seed.resolve()),
        "out": str(out),
        "rank1_cycles": RANK1_CYCLES,
        "part": args.part,
        "clock_ns": args.clock_ns,
        "n_points": len(rows),
        "n_ok": len(legal),
        "best": best,
        "rows": rows,
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    with (out / "summary.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "id",
                "ok",
                "csim_passed",
                "latency_cycles",
                "dsp",
                "bram",
                "lut",
                "ff",
                "vs_rank1",
                "error",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in writer.fieldnames})
    print(json.dumps({"out": str(out), "best": best}, indent=2, default=str))
    return 0 if legal else 1


if __name__ == "__main__":
    raise SystemExit(main())
