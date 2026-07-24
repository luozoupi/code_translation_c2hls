#!/usr/bin/env python3
"""Requeue Fir batch_parallel benches for rerun (e.g. LLM errors, csynth timeout)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "fir"))

from batch_parallel.config import campaign_paths, load_campaign
from batch_parallel.queue import FirBatchParallelQueue
from explorer.metrics import bench_run_issues_from_multistep_doc


def _multistep_doc_for_bench(campaign_root: Path, bench: str) -> dict | None:
    bench_dir = campaign_root / bench
    if not bench_dir.is_dir():
        return None
    for cell in sorted(bench_dir.iterdir()):
        if not cell.is_dir():
            continue
        for name in (f"{bench}_multistep_results.json", "multistep.json"):
            path = cell / name
            if path.is_file():
                return json.loads(path.read_text(encoding="utf-8"))
    return None


def benches_with_run_issues(
    campaign_root: Path,
    *,
    issue_kinds: set[str],
) -> list[str]:
    found: list[str] = []
    for bench_dir in sorted(campaign_root.iterdir()):
        if not bench_dir.is_dir() or not bench_dir.name.startswith("hlsfactory_"):
            continue
        bench = bench_dir.name
        doc = _multistep_doc_for_bench(campaign_root, bench)
        if not doc:
            continue
        issues = set(bench_run_issues_from_multistep_doc(doc))
        if issues & issue_kinds:
            found.append(bench)
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description="Requeue Fir campaign benches for rerun")
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument(
        "--issues",
        default="llm_connection_error,llm_timeout,csynth_timeout",
        help="Comma-separated run_issues to scan for (default: llm_connection_error,llm_timeout,csynth_timeout)",
    )
    parser.add_argument(
        "--bench",
        action="append",
        dest="benches",
        default=[],
        help="Explicit bench name (repeatable); skips issue scan when set",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    campaign_root = args.campaign_root.resolve()
    paths = campaign_paths(campaign_root)
    queue = FirBatchParallelQueue(paths["queue_db"])
    campaign = load_campaign(campaign_root)

    if args.benches:
        targets = sorted(set(args.benches))
    else:
        kinds = {s.strip() for s in args.issues.split(",") if s.strip()}
        targets = benches_with_run_issues(campaign_root, issue_kinds=kinds)

    if not targets:
        print("No benches matched for requeue")
        return 0

    print(f"Campaign: {campaign_root.name}")
    print(f"Targets ({len(targets)}):")
    for bench in targets:
        row = next((j for j in queue.all_jobs() if j["bench"] == bench), None)
        status = row["status"] if row else "missing"
        print(f"  {bench} [{status}]")

    if args.dry_run:
        print("Dry run — no queue changes")
        return 0

    requeued = queue.requeue_benches(targets)
    print(f"Requeued {len(requeued)} benches: {', '.join(requeued)}")

    campaign["campaign_status"] = "running"
    campaign_path = campaign_root / "campaign.json"
    campaign_path.write_text(json.dumps(campaign, indent=2) + "\n", encoding="utf-8")
    print("Set campaign_status=running")
    print(f"Pending flash jobs: {queue.pending_flash_count()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
