#!/usr/bin/env python3
"""Worker process for one batch_parallel task slot (synth or cosim)."""

from __future__ import annotations

import argparse
import logging
import os
import socket
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "pc2"))

from batch_parallel_config import campaign_paths, load_campaign, load_config
from batch_parallel_dispatch import (
    campaign_benches_resolved,
    cell_dir_for_job,
    configure_campaign_env,
    resolve_bench_map,
    run_batch_parallel_job,
    seed_kwargs_for_campaign,
)
from batch_parallel_flow import BatchParallelFlow
from batch_parallel_queue import BatchParallelQueue
from c2hls_paths import configure_site


def worker_loop(args: argparse.Namespace) -> int:
    configure_site("pc2")
    campaign_root = Path(args.campaign_root)
    paths = campaign_paths(campaign_root)
    cfg = load_config()
    campaign = load_campaign(campaign_root)
    configure_campaign_env(campaign, args.variant)
    active_variants = campaign.get("active_variants") or [args.variant]
    benches_order = campaign_benches_resolved(campaign, cfg)
    pilot = (campaign.get("config") or {}).get("pilot") or {}
    model_id = str(pilot.get("model") or cfg.model)
    turns = int(pilot.get("turns") or cfg.turns)
    bench_map = resolve_bench_map(campaign, cfg, benches_order)
    queue = BatchParallelQueue(paths["queue_db"])
    flow = BatchParallelFlow(campaign_root)
    kind = "synth" if args.role == "synth" else "cosim"
    worker_id = f"{args.role}-{args.variant}-n{args.node_index}-s{args.worker_slot}"
    seed_kwargs = seed_kwargs_for_campaign(campaign, cfg)

    queue.register_node_slot(
        variant=args.variant,
        role=args.role,
        node_index=args.node_index,
        worker_slot=args.worker_slot,
        hostname=socket.gethostname(),
        slurm_job_id=os.getenv("SLURM_JOB_ID", ""),
    )

    while True:
        if queue.campaign_complete(active_variants):
            return 0
        queue.maybe_seed_next_bench(
            args.variant,
            benches_order,
            max_inflight=cfg.max_inflight_benches,
            seed_kwargs=seed_kwargs,
        )
        job = queue.claim(
            kind=kind,
            variant=args.variant,
            role=args.role,
            node_index=args.node_index,
            worker_slot=args.worker_slot,
            worker_id=worker_id,
        )
        if job is None:
            queue.heartbeat_node_slot(
                variant=args.variant,
                role=args.role,
                node_index=args.node_index,
                worker_slot=args.worker_slot,
            )
            flow.write_node_map(queue.snapshot_node_map())
            time.sleep(cfg.poll_sec)
            continue

        event = f"{kind}_start"
        flow.emit(
            event,
            scope="variant",
            variant=args.variant,
            bench=job.bench,
            phase=job.phase,
            role=args.role,
            node_index=args.node_index,
            worker_slot=args.worker_slot,
            pending_codegen=queue.pending_codegen(),
        )
        if kind == "cosim":
            flow.emit(
                "cosim_start",
                scope="variant",
                variant=args.variant,
                bench=job.bench,
                phase=job.phase,
                role=args.role,
                node_index=args.node_index,
                worker_slot=args.worker_slot,
            )

        cell = cell_dir_for_job(campaign_root, campaign, job, model_id)
        cell.mkdir(parents=True, exist_ok=True)
        try:
            run_batch_parallel_job(
                job=job,
                queue=queue,
                campaign=campaign,
                bench_dir=bench_map[job.bench],
                cell_dir=cell,
                model_id=model_id,
                turns=turns,
            )
            queue.complete(job.id)
            flow.emit(
                f"{kind}_done",
                scope="variant",
                variant=args.variant,
                bench=job.bench,
                phase=job.phase,
                role=args.role,
                node_index=args.node_index,
                worker_slot=args.worker_slot,
            )
        except Exception as exc:
            queue.complete(job.id, error=str(exc))
            flow.emit(
                f"{kind}_failed",
                scope="variant",
                variant=args.variant,
                bench=job.bench,
                error=str(exc),
                role=args.role,
                node_index=args.node_index,
                worker_slot=args.worker_slot,
            )
        queue.maybe_seed_next_bench(
            args.variant,
            benches_order,
            max_inflight=cfg.max_inflight_benches,
            seed_kwargs=seed_kwargs,
        )
        flow.write_node_map(queue.snapshot_node_map())


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="batch_parallel worker slot")
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--role", choices=["synth", "cosim"], required=True)
    parser.add_argument("--node-index", type=int, required=True)
    parser.add_argument("--worker-slot", type=int, required=True)
    args = parser.parse_args()
    return worker_loop(args)


if __name__ == "__main__":
    raise SystemExit(main())
