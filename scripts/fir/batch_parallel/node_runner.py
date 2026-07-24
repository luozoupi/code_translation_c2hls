#!/usr/bin/env python3
"""Node runner: spawns worker_slot processes on one Fir Slurm compute allocation."""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SCRIPT_DIR = REPO / "scripts" / "fir"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(SCRIPT_DIR))

from batch_parallel.config import campaign_paths, load_config
from batch_parallel.queue import FirBatchParallelQueue


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Fir batch_parallel node runner")
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--node-index", type=int, required=True)
    args = parser.parse_args()

    cfg = load_config()
    workers = cfg.workers_per_node
    paths = campaign_paths(Path(args.campaign_root))
    queue = FirBatchParallelQueue(paths["queue_db"])

    for slot in range(workers):
        queue.register_node_slot(
            node_index=args.node_index,
            worker_slot=slot,
            hostname=os.uname().nodename,
            slurm_job_id=os.getenv("SLURM_JOB_ID", ""),
        )

    py = os.getenv("C2HLS_PYTHON", "python3")
    worker_script = SCRIPT_DIR / "batch_parallel" / "worker.py"
    procs: list[subprocess.Popen] = []
    for slot in range(workers):
        cmd = [
            py,
            str(worker_script),
            "--campaign-root", args.campaign_root,
            "--node-index", str(args.node_index),
            "--worker-slot", str(slot),
        ]
        logging.info("spawn worker slot %s: %s", slot, " ".join(cmd))
        procs.append(subprocess.Popen(cmd))

    exit_code = 0
    while procs:
        for proc in list(procs):
            code = proc.poll()
            if code is None:
                continue
            procs.remove(proc)
            if code != 0:
                exit_code = code
        if procs:
            time.sleep(2.0)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
