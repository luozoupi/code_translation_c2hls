#!/usr/bin/env python3
"""Worker process for one Fir batch_parallel slot."""

from __future__ import annotations

import argparse
import logging
import os
import socket
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "fir"))

from batch_parallel.config import benches_for_campaign, campaign_paths, load_campaign, load_config
from batch_parallel.dispatch import run_flash_bench
from batch_parallel.flow import FirBatchParallelFlow
from batch_parallel.queue import FirBatchParallelQueue, is_retryable_error


def release_compute_after_bench(cfg) -> bool:
    """One bench per compute node: exit the Slurm allocation when it finishes."""
    return bool(cfg.compute_nodes_match_benches) and cfg.workers_per_node == 1


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Fir batch_parallel worker")
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--node-index", type=int, required=True)
    parser.add_argument("--worker-slot", type=int, required=True)
    args = parser.parse_args()

    campaign_root = Path(args.campaign_root)
    paths = campaign_paths(campaign_root)
    cfg = load_config()
    campaign = load_campaign(campaign_root)
    pilot = (campaign.get("config") or {}).get("pilot") or {}
    model_id = str(pilot.get("model") or cfg.model)
    turns = int(pilot.get("turns") or cfg.turns)

    queue = FirBatchParallelQueue(paths["queue_db"])
    flow = FirBatchParallelFlow(campaign_root)
    worker_id = f"flash-n{args.node_index}-s{args.worker_slot}"

    queue.register_node_slot(
        node_index=args.node_index,
        worker_slot=args.worker_slot,
        hostname=socket.gethostname(),
        slurm_job_id=os.getenv("SLURM_JOB_ID", ""),
    )

    release_after_bench = release_compute_after_bench(cfg)
    finished_bench = False

    while True:
        if queue.campaign_complete():
            return 0

        job = queue.claim(
            node_index=args.node_index,
            worker_slot=args.worker_slot,
            worker_id=worker_id,
        )
        if job is None:
            if release_after_bench and finished_bench:
                flow.emit(
                    "node_released",
                    node_index=args.node_index,
                    worker_slot=args.worker_slot,
                    worker_id=worker_id,
                    reason="bench_done",
                )
                logging.info(
                    "node %s slot %s released (bench finished)",
                    args.node_index,
                    args.worker_slot,
                )
                return 0
            queue.heartbeat_node_slot(node_index=args.node_index, worker_slot=args.worker_slot)
            flow.write_node_map(queue.snapshot_node_map())
            time.sleep(cfg.poll_sec)
            continue

        def _release_node(*, reason: str) -> int:
            flow.emit(
                "node_released",
                bench=job.bench,
                node_index=args.node_index,
                worker_slot=args.worker_slot,
                worker_id=worker_id,
                reason=reason,
            )
            logging.info(
                "node %s slot %s released after %s (%s)",
                args.node_index,
                args.worker_slot,
                job.bench,
                reason,
            )
            return 0

        flow.emit(
            "flash_start",
            bench=job.bench,
            node_index=args.node_index,
            worker_slot=args.worker_slot,
            worker_id=worker_id,
        )
        try:
            row = run_flash_bench(
                campaign_root=campaign_root,
                bench=job.bench,
                model_id=model_id,
                turns=turns,
                endpoint_file=paths["endpoint"],
                campaign=campaign,
                job_id=job.id,
                worker_id=worker_id,
            )
            success = row.get("status") == "ok"
            error = str(row.get("error") or "")
            if success:
                queue.complete(
                    job.id,
                    success=True,
                    error=error,
                    result_path=str(row.get("result_path") or ""),
                )
                flow.emit(
                    "flash_done",
                    bench=job.bench,
                    success=True,
                    node_index=args.node_index,
                    worker_slot=args.worker_slot,
                )
                finished_bench = True
                if release_after_bench:
                    return _release_node(reason="success")
            elif is_retryable_error(error):
                queue.requeue(job.id)
                flow.emit(
                    "flash_requeued",
                    bench=job.bench,
                    error=error,
                    node_index=args.node_index,
                    worker_slot=args.worker_slot,
                )
                finished_bench = True
                if release_after_bench:
                    return _release_node(reason="requeued")
            else:
                queue.complete(
                    job.id,
                    success=False,
                    error=error,
                    result_path=str(row.get("result_path") or ""),
                )
                flow.emit(
                    "flash_done",
                    bench=job.bench,
                    success=False,
                    node_index=args.node_index,
                    worker_slot=args.worker_slot,
                )
                finished_bench = True
                if release_after_bench:
                    return _release_node(reason="failed")
        except Exception as exc:
            error = str(exc)
            if is_retryable_error(error):
                queue.requeue(job.id)
                flow.emit(
                    "flash_requeued",
                    bench=job.bench,
                    error=error,
                    node_index=args.node_index,
                    worker_slot=args.worker_slot,
                )
                finished_bench = True
                if release_after_bench:
                    return _release_node(reason="requeued_exception")
            else:
                queue.complete(job.id, success=False, error=error)
                flow.emit(
                    "flash_failed",
                    bench=job.bench,
                    error=error,
                    node_index=args.node_index,
                    worker_slot=args.worker_slot,
                )
                finished_bench = True
                if release_after_bench:
                    return _release_node(reason="exception")


if __name__ == "__main__":
    raise SystemExit(main())
