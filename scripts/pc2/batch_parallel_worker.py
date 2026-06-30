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

from batch_parallel_bench import execute_job
from batch_parallel_config import campaign_paths, campaign_benches, load_campaign, load_config
from batch_parallel_flow import BatchParallelFlow
from batch_parallel_queue import BatchParallelQueue
from c2hls_paths import configure_site
from flash_fixed_cosim_lib import VARIANTS, configure_fixed_cosim_flash_env, resolve_cosim_benches
from run_flash_fixed_cosim_batch import _cell_dir, model_cell_tag


def _bench_map(benches: list[str]) -> dict[str, Path]:
    return {name: path for name, path in resolve_cosim_benches(benches)}


def worker_loop(args: argparse.Namespace) -> int:
    configure_site("pc2")
    variant = VARIANTS[args.variant]
    configure_fixed_cosim_flash_env(variant)
    campaign_root = Path(args.campaign_root)
    paths = campaign_paths(campaign_root)
    cfg = load_config()
    campaign = load_campaign(campaign_root)
    active_variants = campaign.get("active_variants") or [args.variant]
    benches_order = campaign_benches(campaign, cfg)
    pilot = (campaign.get("config") or {}).get("pilot") or {}
    model_id = str(pilot.get("model") or cfg.model)
    turns = int(pilot.get("turns") or cfg.turns)
    model_tag = model_cell_tag(model_id)
    bench_map = _bench_map(benches_order)
    cell_root = campaign_root / "variants" / args.variant
    queue = BatchParallelQueue(paths["queue_db"])
    flow = BatchParallelFlow(campaign_root)
    kind = "synth" if args.role == "synth" else "cosim"
    worker_id = f"{args.role}-{args.variant}-n{args.node_index}-s{args.worker_slot}"

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

        cell = _cell_dir(cell_root, job.bench, model_tag, variant)
        cell.mkdir(parents=True, exist_ok=True)
        try:
            execute_job(
                job=job,
                queue=queue,
                bench_dir=bench_map[job.bench],
                cell_dir=cell,
                variant_key=args.variant,
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
