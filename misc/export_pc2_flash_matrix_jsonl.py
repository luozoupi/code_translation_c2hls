#!/usr/bin/env python3
"""Export Devstral PC2 flash matrix artifacts to schema-1.0 JSONL.

Reads artifacts/pc2/<flash_run>/matrix.json + per-cell *_multistep_results.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
PC2 = REPO / "artifacts" / "pc2"
BENCHMARKS = REPO / "benchmarks"

from export_schema_jsonl import (  # noqa: E402
    SCHEMA_VERSION,
    TARGET_CSIM,
    TARGET_CSYNTH,
    _build_hls_synth_payload,
    _build_implementation,
    _build_problem,
    _build_run,
    validate_jsonl,
)

SUITE = "hlsfactory_polybench_float_small"
DEFAULT_PART = "xcu280-fsvh2892-2L-e"
DEFAULT_CLOCK_NS = 3.33


def _bench_meta(bench: str) -> dict[str, Any]:
    bench_dir = BENCHMARKS / bench
    if not bench_dir.is_dir():
        alt = bench.replace("-", "_")
        for p in BENCHMARKS.glob("hlsfactory_*"):
            if p.name.removeprefix("hlsfactory_").replace("-", "_") == alt.replace("hlsfactory_", ""):
                bench_dir = p
                break
    meta = json.loads((bench_dir / "metadata.json").read_text())
    short = bench_dir.name.removeprefix("hlsfactory_")
    meta = dict(meta)
    meta["group_path"] = [short]
    return meta


def _status_from_block(block: dict[str, Any] | None) -> str | None:
    if not block:
        return None
    if block.get("passed") is True:
        return "pass"
    status = (block.get("status") or "").lower()
    if status in {"passed", "pass"}:
        return "pass"
    if status == "timeout" or block.get("timed_out"):
        return "timeout"
    if block.get("ran") is False:
        return None
    if block.get("success") is False or block.get("passed") is False:
        return "fail"
    return None


def _load_manifest(artifact_dir: Path) -> dict[str, Any]:
    for name in ("manifest.json", "plan.json"):
        path = artifact_dir / name
        if path.is_file():
            try:
                return json.loads(path.read_text())
            except json.JSONDecodeError:
                pass
    return {}


def _top_model_payload(report: dict[str, Any], status: str, top: str, part: str, clock_ns: float) -> dict[str, Any]:
    payload = _build_hls_synth_payload(report, part, clock_ns, status=status)
    ua = payload.get("UserAssignments")
    if isinstance(ua, dict):
        ua["TopModelName"] = top
    return payload


def export_artifact_dir(artifact_dir: Path) -> list[dict[str, Any]]:
    matrix_path = artifact_dir / "matrix.json"
    if not matrix_path.is_file():
        raise FileNotFoundError(f"missing matrix.json: {artifact_dir}")

    manifest = _load_manifest(artifact_dir)
    setup = manifest.get("setup") or artifact_dir.name
    stamp = manifest.get("stamp") or artifact_dir.name.rsplit("_", 2)[-2:]
    origin_version = f"pc2_{artifact_dir.name}"

    records: list[dict[str, Any]] = []
    for row in json.loads(matrix_path.read_text()):
        bench = row.get("bench") or ""
        cell_dir = Path(row.get("cell_dir") or "")
        if not bench or not cell_dir.is_dir():
            continue

        result_path = cell_dir / f"{bench}_multistep_results.json"
        if not result_path.is_file():
            continue
        try:
            data = json.loads(result_path.read_text())
        except json.JSONDecodeError:
            continue

        meta = _bench_meta(bench)
        top = meta.get("hls_top") or meta.get("kernel_top") or "workload"
        run_meta = data.get("run") or {}
        part = run_meta.get("part") or DEFAULT_PART
        clock_ns = float(run_meta.get("clock_ns") or DEFAULT_CLOCK_NS)
        model = row.get("model") or run_meta.get("model") or manifest.get("model")
        mode = row.get("mode") or "flash"
        skills = row.get("skills") or ("on" if "noskills" not in setup else "off")

        origin_meta = {
            "model": model,
            "model_tag": manifest.get("model_tag") or "devstral2",
            "mode": mode,
            "skills": skills,
            "phase": data.get("phase") or "flash",
            "artifact_dir": str(artifact_dir),
            "artifact_setup": setup,
            "artifact_stamp": stamp if isinstance(stamp, str) else "_".join(stamp),
            "cell_dir": str(cell_dir),
            "matrix_status": row.get("status"),
            "wallclock_s": row.get("wallclock_s"),
        }

        synth = data.get("final_report") or {}
        if not synth:
            steps = data.get("steps") or []
            if steps:
                synth = steps[-1].get("report") or {}
        if not synth and isinstance(row.get("summary"), dict):
            synth = row["summary"].get("synth_report") or {}

        csim = None
        steps = data.get("steps") or []
        if steps:
            csim = steps[-1].get("csim")
        if not csim:
            csim = data.get("csim") or data.get("baseline_csim")

        variant = {"index": 0, "name": "final"}
        impl_common = dict(
            meta=meta,
            variant_name="final",
            variant_index=0,
            origin_override="c2hls_orchestrator",
            origin_version=origin_version,
            origin_meta=origin_meta,
        )

        if synth:
            synth_status = "pass" if synth.get("latency_cycles") is not None else "fail"
            records.append({
                "schema_version": SCHEMA_VERSION,
                "report_type": "hls_synth",
                "run": _build_run(TARGET_CSYNTH, part, row.get("wallclock_s"), run_meta),
                "problem": {**_build_problem(meta), "suite": SUITE},
                "implementation": _build_implementation(**impl_common),
                "hls_synth": _top_model_payload(synth, synth_status, top, part, clock_ns),
            })

        csim_status = _status_from_block(csim)
        if csim_status:
            records.append({
                "schema_version": SCHEMA_VERSION,
                "report_type": "sw_run",
                "run": _build_run(TARGET_CSIM, part, row.get("wallclock_s"), run_meta),
                "problem": {**_build_problem(meta), "suite": SUITE},
                "implementation": _build_implementation(**impl_common),
                "sw_run": {
                    "status": csim_status,
                    "error": (csim.get("error") or "")[:300] if csim_status != "pass" else None,
                },
            })

    return records


def discover_devstral_flash_dirs(pc2_root: Path = PC2) -> list[Path]:
    out: list[Path] = []
    for path in sorted(pc2_root.iterdir()):
        if not path.is_dir() or not path.name.startswith("flash_"):
            continue
        matrix = path / "matrix.json"
        if not matrix.is_file():
            continue
        try:
            rows = json.loads(matrix.read_text())
        except json.JSONDecodeError:
            continue
        if len(rows) != 28:
            continue
        if rows and "Devstral" not in (rows[0].get("model") or ""):
            continue
        out.append(path)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-dir",
        action="append",
        dest="artifact_dirs",
        help="PC2 artifact directory name under artifacts/pc2/ (repeatable)",
    )
    parser.add_argument(
        "--all-devstral-flash",
        action="store_true",
        help="Export all 28-bench Devstral flash artifact dirs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "misc" / "devstral2_flash_pc2_schema.jsonl",
    )
    args = parser.parse_args()

    if args.all_devstral_flash:
        dirs = discover_devstral_flash_dirs()
    elif args.artifact_dirs:
        dirs = [PC2 / name for name in args.artifact_dirs]
    else:
        dirs = [
            PC2 / "flash_noskills_20260620_004507",
            PC2 / "flash_all_skills_no_avoids_global_20260620_113247",
            PC2 / "flash_skills_20260620_004507",
            PC2 / "flash_bn_skills_new_2_2_20260621_020847",
        ]

    records: list[dict[str, Any]] = []
    per_dir: dict[str, int] = {}
    for artifact_dir in dirs:
        if not artifact_dir.is_dir():
            print(f"skip missing: {artifact_dir}", file=sys.stderr)
            continue
        batch = export_artifact_dir(artifact_dir)
        per_dir[artifact_dir.name] = len(batch)
        records.extend(batch)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    validation = validate_jsonl(args.output, verbose=True)
    summary = {"output": str(args.output), "artifact_dirs": per_dir, "records": len(records), "validation": validation}
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if validation.get("invalid", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
