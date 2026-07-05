#!/usr/bin/env python3
"""Reconstruct GPU-hour usage per test/campaign from Slurm sacct.

GPU-hours are computed as wall_elapsed_hours × allocated H100 count (typically 4
for a full gpu_h100 node). Only top-level jobs (-X) on gpu_h100 (or with GPU
AllocTRES) are included.

Example:
  python3 scripts/pc2/analysis/gpu_hours_from_sacct.py
  python3 scripts/pc2/analysis/gpu_hours_from_sacct.py --starttime 2026-07-01
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
DEFAULT_OUT = REPO / "artifacts/pc2/analysis/gpu_hours_per_test_sacct.csv"


def parse_elapsed_hours(text: str) -> float:
    s = (text or "").strip()
    if not s or s == "00:00:00":
        return 0.0
    days = 0
    if "-" in s:
        day_part, s = s.split("-", 1)
        days = int(day_part)
    parts = s.split(":")
    if len(parts) == 3:
        hours, minutes, seconds = (int(parts[0]), int(parts[1]), int(parts[2]))
    elif len(parts) == 2:
        hours = 0
        minutes, seconds = int(parts[0]), int(parts[1])
    else:
        return 0.0
    return days * 24 + hours + minutes / 60 + seconds / 3600


def parse_ngpus(alloc_tres: str) -> int:
    m = re.search(r"gres/gpu:h100=(\d+)", alloc_tres or "")
    if m:
        return int(m.group(1))
    m = re.search(r"gres/gpu=(\d+)", alloc_tres or "")
    return int(m.group(1)) if m else 0


def normalize_test(job_name: str) -> str:
    name = job_name.strip()
    if name.startswith("bp-llm-"):
        return name[len("bp-llm-") :]
    if name.startswith("c2hls-llm-"):
        return name[len("c2hls-llm-") :]
    if name.startswith("c2hls-llm"):
        return "c2hls-llm"
    return name


def load_campaign_job_map(artifacts_pc2: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if not artifacts_pc2.is_dir():
        return mapping
    for campaign_path in artifacts_pc2.glob("*/campaign.json"):
        try:
            doc = json.loads(campaign_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        stamp = campaign_path.parent.name
        gpu_job_id = doc.get("gpu_job_id")
        if gpu_job_id:
            mapping[str(gpu_job_id)] = stamp
        for row in doc.get("compute_jobs") or []:
            slurm_job_id = row.get("slurm_job_id")
            if slurm_job_id:
                mapping[str(slurm_job_id)] = stamp
    return mapping


def fetch_sacct_rows(*, user: str, starttime: str) -> list[dict[str, Any]]:
    cmd = [
        "sacct",
        "-u",
        user,
        f"--starttime={starttime}",
        "--format=JobID,JobName,Account,Partition,State,Start,End,Elapsed,AllocTRES",
        "-n",
        "-P",
        "-X",
    ]
    try:
        raw = subprocess.check_output(cmd, text=True, errors="replace")
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"sacct failed: {exc}") from exc

    rows: list[dict[str, Any]] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) < 9:
            continue
        job_id, job_name, account, partition, state, start, end, elapsed, tres = parts[:9]
        if "." in job_id:
            continue
        if partition != "gpu_h100" and "gres/gpu" not in (tres or ""):
            continue
        ngpus = parse_ngpus(tres)
        if ngpus <= 0:
            continue
        elapsed_h = parse_elapsed_hours(elapsed)
        gpu_hours = elapsed_h * ngpus
        if gpu_hours <= 0 and state not in {"RUNNING", "PENDING"}:
            continue
        rows.append(
            {
                "jobid": job_id,
                "test": normalize_test(job_name),
                "jobname": job_name,
                "account": account,
                "partition": partition,
                "state": state,
                "start": start,
                "end": end,
                "elapsed_h": round(elapsed_h, 4),
                "ngpus": ngpus,
                "gpu_hours": round(gpu_hours, 4),
            }
        )
    return rows


def attach_campaigns(rows: list[dict[str, Any]], job_map: dict[str, str]) -> None:
    for row in rows:
        campaign = job_map.get(row["jobid"])
        if campaign is None and row["jobname"].startswith("bp-llm-"):
            campaign = row["test"]
        row["campaign"] = campaign or ""


def aggregate_by_test(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_test: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"gpu_hours": 0.0, "jobs": 0, "accounts": set()}
    )
    for row in rows:
        info = by_test[row["test"]]
        info["gpu_hours"] += row["gpu_hours"]
        info["jobs"] += 1
        info["accounts"].add(row["account"])
    return by_test


def aggregate_by_account(rows: list[dict[str, Any]]) -> dict[str, float]:
    by_account: dict[str, float] = defaultdict(float)
    for row in rows:
        by_account[row["account"]] += row["gpu_hours"]
    return by_account


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "jobid",
        "test",
        "campaign",
        "jobname",
        "account",
        "partition",
        "state",
        "start",
        "end",
        "elapsed_h",
        "ngpus",
        "gpu_hours",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["start"]))


def print_report(
    *,
    rows: list[dict[str, Any]],
    by_test: dict[str, dict[str, Any]],
    by_account: dict[str, float],
    starttime: str,
    user: str,
) -> None:
    total_gpu_hours = sum(row["gpu_hours"] for row in rows)
    print("=" * 80)
    print(f"GPU HOURS BY TEST/CAMPAIGN (sacct since {starttime}, user={user})")
    print("GPU-hours = wall_elapsed_hours × allocated H100 count")
    print("=" * 80)
    print(f"{'Test/Campaign':<55} {'Jobs':>5} {'GPU-h':>10} {'Account(s)':<20}")
    print("-" * 80)
    for test, info in sorted(by_test.items(), key=lambda item: -item[1]["gpu_hours"]):
        accounts = ",".join(sorted(info["accounts"]))
        print(f"{test[:55]:<55} {info['jobs']:>5} {info['gpu_hours']:>10.2f} {accounts[:20]:<20}")
    print("-" * 80)
    print(f"{'TOTAL':<55} {len(rows):>5} {total_gpu_hours:>10.2f}")

    batch_parallel = {
        test: info["gpu_hours"]
        for test, info in by_test.items()
        if test.startswith("batch_parallel")
    }
    if batch_parallel:
        print("\n" + "=" * 80)
        print("BATCH PARALLEL CAMPAIGNS")
        print("=" * 80)
        for name, gpu_hours in sorted(batch_parallel.items(), key=lambda item: -item[1]):
            print(f"  {name:<60} {gpu_hours:10.2f} GPU-h")

    print("\n" + "=" * 80)
    print("BY SLURM ACCOUNT")
    print("=" * 80)
    for account, gpu_hours in sorted(by_account.items(), key=lambda item: -item[1]):
        print(f"  {account:<25} {gpu_hours:10.2f} GPU-h")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", default="haqc2", help="Slurm user (default: haqc2)")
    parser.add_argument(
        "--starttime",
        default="2026-06-15",
        help="sacct --starttime value (default: 2026-06-15)",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO,
        help="c2hls repo root for campaign.json lookup",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=DEFAULT_OUT,
        help=f"per-job CSV output (default: {DEFAULT_OUT})",
    )
    parser.add_argument("--json", action="store_true", help="emit summary JSON to stdout")
    args = parser.parse_args()

    job_map = load_campaign_job_map(args.repo_root / "artifacts/pc2")
    rows = fetch_sacct_rows(user=args.user, starttime=args.starttime)
    attach_campaigns(rows, job_map)
    by_test = aggregate_by_test(rows)
    by_account = aggregate_by_account(rows)

    write_csv(args.out_csv, rows)
    if args.json:
        payload = {
            "user": args.user,
            "starttime": args.starttime,
            "total_gpu_hours": round(sum(row["gpu_hours"] for row in rows), 4),
            "job_count": len(rows),
            "by_test": {
                test: {
                    "gpu_hours": round(info["gpu_hours"], 4),
                    "jobs": info["jobs"],
                    "accounts": sorted(info["accounts"]),
                }
                for test, info in sorted(by_test.items(), key=lambda item: -item[1]["gpu_hours"])
            },
            "by_account": {
                account: round(gpu_hours, 4)
                for account, gpu_hours in sorted(by_account.items(), key=lambda item: -item[1])
            },
            "csv": str(args.out_csv.resolve()),
        }
        json.dump(payload, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        print_report(
            rows=rows,
            by_test=by_test,
            by_account=by_account,
            starttime=args.starttime,
            user=args.user,
        )
        print(f"\nWrote per-job CSV: {args.out_csv.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
