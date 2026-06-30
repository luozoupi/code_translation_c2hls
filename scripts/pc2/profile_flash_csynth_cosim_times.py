#!/usr/bin/env python3
"""Profile csynth and cosim wall times per kernel × variant × phase (phase_b vs flash/selected)."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TMP = REPO / "c2hls_tmp"
VARIANTS = ("aav_n", "aav_o", "nav_n", "nav_o", "noskills")

VITIS_RE = re.compile(r"vitis-run 60-791\] Total elapsed time: (?:(\d+)h )?(\d+)m (\d+)s")
HLS_RE = re.compile(r"Total elapsed time: ([0-9.]+) seconds")


def parse_log_seconds(log_path: Path) -> float | None:
    if not log_path.is_file():
        return None
    try:
        with log_path.open("rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - 65536))
            tail = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return None
    matches = list(VITIS_RE.finditer(tail))
    if matches:
        last = matches[-1]
        hours = int(last.group(1) or 0)
        return hours * 3600 + int(last.group(2)) * 60 + int(last.group(3))
    hls_matches = list(HLS_RE.finditer(tail))
    if hls_matches:
        return float(hls_matches[-1].group(1))
    return None


def workdir_from_report(report: dict) -> str | None:
    feedback = report.get("feedback") or {}
    for scope in feedback.get("scopes") or []:
        loc = scope.get("source_location") or ""
        if "c2hls_tmp/hls_synth" in loc:
            return loc.split("c2hls_tmp/")[1].split("/")[0]
    return None


def classify_synth_dir(name: str) -> str | None:
    if "phase_b" in name:
        return "phase_b"
    if "flash" in name or "selected" in name:
        return "flash"
    return None


def bench_variant_from_dir(name: str) -> tuple[str | None, str | None]:
    bench_match = re.search(r"hlsfactory_([^_]+(?:-\d+d)?)", name)
    bench = bench_match.group(1) if bench_match else None
    variant_match = re.search(r"fixed_cosim_([a-z_]+)_", name)
    variant = variant_match.group(1) if variant_match else None
    return bench, variant


def load_final_csynth_rows(flash_stamp: str) -> list[dict]:
    rows: list[dict] = []
    for variant in VARIANTS:
        artifact = REPO / f"artifacts/pc2/flash_fixed_cosim_{variant}_{flash_stamp}"
        if not artifact.is_dir():
            continue
        for bench_dir in sorted(artifact.glob("hlsfactory_*")):
            bench = bench_dir.name
            cell = bench_dir / f"devstral2__flash__fixed_cosim__{variant}"
            if not cell.is_dir():
                continue
            for phase, names in (
                ("phase_b", (f"{bench}_phase_b_report.json",)),
                ("flash", (f"{bench}_selected_report.json", f"{bench}_flash_opt_report.json")),
            ):
                report_path = next((cell / name for name in names if (cell / name).is_file()), None)
                if report_path is None:
                    continue
                report = json.loads(report_path.read_text(encoding="utf-8"))
                work_dir = workdir_from_report(report)
                seconds = None
                if work_dir:
                    seconds = parse_log_seconds(TMP / work_dir / "logs" / "hls_run_tcl.log")
                rows.append(
                    {
                        "variant": variant,
                        "bench": bench.removeprefix("hlsfactory_"),
                        "phase": phase,
                        "csynth_final_s": seconds,
                        "csynth_work_dir": work_dir,
                        "report_file": str(report_path.relative_to(REPO)),
                    }
                )
    return rows


def load_synth_attempts() -> dict[tuple[str | None, str, str], list[float]]:
    attempts: dict[tuple[str | None, str, str], list[float]] = defaultdict(list)
    for work_dir in TMP.glob("hls_synth__*"):
        phase = classify_synth_dir(work_dir.name)
        bench, variant = bench_variant_from_dir(work_dir.name)
        if not phase or not bench:
            continue
        if "fixed_cosim" in work_dir.name:
            if not any(f"fixed_cosim_{variant_key}" in work_dir.name for variant_key in VARIANTS):
                continue
        seconds = parse_log_seconds(work_dir / "logs" / "hls_run_tcl.log")
        if seconds is None:
            continue
        attempts[(variant, bench, phase)].append(seconds)
    return attempts


def load_cosim_rows(selected_root: Path, phase_b_root: Path) -> list[dict]:
    rows: list[dict] = []
    for stamp_root, phase in ((selected_root, "flash"), (phase_b_root, "phase_b")):
        if not stamp_root.is_dir():
            continue
        for result_path in stamp_root.glob("cells/*/cosim_result.json"):
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            provenance = payload.get("provenance") or {}
            rows.append(
                {
                    "variant": provenance.get("variant"),
                    "bench": (provenance.get("bench") or "").removeprefix("hlsfactory_"),
                    "phase": phase,
                    "cosim_s": payload.get("runtime_seconds"),
                    "cosim_status": payload.get("status"),
                    "cosim_passed": payload.get("passed"),
                }
            )
    return rows


def build_profile(
    *,
    flash_stamp: str,
    selected_cosim_root: Path,
    phase_b_cosim_root: Path,
) -> list[dict]:
    csynth_rows = load_final_csynth_rows(flash_stamp)
    attempts = load_synth_attempts()
    cosim_rows = load_cosim_rows(selected_cosim_root, phase_b_cosim_root)
    cosim_map = {(r["variant"], r["bench"], r["phase"]): r for r in cosim_rows}

    profile: list[dict] = []
    for row in csynth_rows:
        key = (row["variant"], row["bench"], row["phase"])
        cosim = cosim_map.get(key, {})
        bench_key = row["bench"]
        att = attempts.get((row["variant"], bench_key, row["phase"]), [])
        if not att:
            att = attempts.get((None, bench_key, row["phase"]), [])
        profile.append(
            {
                **row,
                "cosim_s": cosim.get("cosim_s"),
                "cosim_status": cosim.get("cosim_status"),
                "cosim_passed": cosim.get("cosim_passed"),
                "csynth_attempts": len(att),
                "csynth_attempt_total_s": sum(att) if att else None,
                "csynth_attempt_max_s": max(att) if att else None,
                "csynth_attempt_median_s": statistics.median(att) if att else None,
            }
        )
    return profile


def print_summary(profile: list[dict]) -> None:
    print("variant  bench          phase    csynth_s  cosim_s   cosim_st  att  att_max_s")
    for row in sorted(profile, key=lambda r: (r["variant"], r["bench"], r["phase"])):
        cs = f"{row['csynth_final_s']:.0f}" if row.get("csynth_final_s") else "-"
        co = f"{row['cosim_s']:.0f}" if row.get("cosim_s") else "-"
        print(
            f"{row['variant']:8} {row['bench']:14} {row['phase']:8} {cs:>8} {co:>9} "
            f"{str(row.get('cosim_status') or '-'):9} {row.get('csynth_attempts', 0):>3} "
            f"{row.get('csynth_attempt_max_s') or 0:>8.0f}"
        )

    for phase in ("phase_b", "flash"):
        csynth = [r["csynth_final_s"] for r in profile if r["phase"] == phase and r.get("csynth_final_s")]
        cosim = [r["cosim_s"] for r in profile if r["phase"] == phase and r.get("cosim_s")]
        print(f"\n{phase} csynth: n={len(csynth)} median={statistics.median(csynth):.0f}s "
              f"max={max(csynth):.0f}s sum={sum(csynth)/3600:.1f}h")
        if cosim:
            print(
                f"{phase} cosim: n={len(cosim)} median={statistics.median(cosim):.0f}s "
                f"max={max(cosim):.0f}s sum={sum(cosim)/3600:.1f}h"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flash-stamp", default="20260628_fixed_cosim_flash_r2_pipelined")
    parser.add_argument(
        "--selected-cosim-root",
        type=Path,
        default=REPO / "artifacts/pc2/flash_cosim/fixed_cosim_flash_20260628",
    )
    parser.add_argument(
        "--phase-b-cosim-root",
        type=Path,
        default=REPO / "artifacts/pc2/flash_cosim/fixed_cosim_flash_phase_b_20260628",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=REPO / "artifacts/pc2/analysis/20260628_fixed_cosim_flash_r2_pipelined/csynth_cosim_time_profile.csv",
    )
    args = parser.parse_args()

    profile = build_profile(
        flash_stamp=args.flash_stamp,
        selected_cosim_root=args.selected_cosim_root,
        phase_b_cosim_root=args.phase_b_cosim_root,
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(profile[0].keys()) if profile else []
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(profile)

    print_summary(profile)
    print(f"\nWrote {args.output_csv}")
    print("Note: pipelined LLM+Vitis run had C2HLS_RUN_COSIM off; cosim times are post-batch full-size runs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
